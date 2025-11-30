package main

import (
	"context"
	"encoding/json"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/sashabaranov/go-openai"
	"github.com/sourcenetwork/defradb/client"
	"github.com/sourcenetwork/defradb/node"
)

const (
	ollamaBaseURL  = "http://localhost:11434/v1"
	llmModel       = "gemma:2b"
	embeddingModel = "nomic-embed-text"
	schemaDef      = `type Wiki {
        text: String
        category: String
        text_v: [Float32!] @embedding(fields: ["text"], provider: "ollama", model: "nomic-embed-text")
    }`
)

var (
	db              *node.Node
	once            sync.Once
	initErr         error
	systemPromptTpl = template.Must(template.New("system_prompt").Parse(`
You are a helpful assistant with access to a knowlege base, tasked with answering questions about the world and its history, people, places and other things.
Answer the question in a very concise manner. Use an unbiased and journalistic tone. Do not repeat text. Don't make anything up. If you are not sure about something, just say that you don't know.
{{- if . }}
Answer the question solely based on the provided search results from the knowledge base. If the search results from the knowledge base are not relevant to the question at hand, just say that you don't know. Don't make anything up.

<context>
    {{- range . }}
    - {{ . }}
    {{- end }}
</context>
{{- end }}
Don't mention the knowledge base, context or search results in your answer.
`))
)

func initDefraNode() (*node.Node, error) {
	ctx := context.Background()

	n, err := node.New(
		ctx,
		node.WithBadgerInMemory(true),
		node.WithDisableAPI(true),
		node.WithDisableP2P(true),
	)
	if err != nil {
		return nil, err
	}

	if err := n.Start(ctx); err != nil {
		return nil, err
	}

	_, err = n.DB.AddSchema(ctx, schemaDef)
	if err != nil {
		return nil, err
	}

	return n, nil
}

func loadWikiData(n *node.Node) error {
	ctx := context.Background()

	f, err := os.Open("wiki.jsonl")
	if err != nil {
		return err
	}
	defer f.Close()

	d := json.NewDecoder(f)
	for {
		var article struct {
			Text     string `json:"text"`
			Category string `json:"category"`
		}

		err := d.Decode(&article)
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		content := "search_document: " + article.Text

		createResult := n.DB.ExecRequest(
			ctx,
			`mutation CreateWiki($input: [WikiMutationInputArg!]!) {
                create_Wiki(input: $input) { _docID }
            }`,
			client.WithVariables(map[string]interface{}{
				"input": map[string]interface{}{
					"text":     content,
					"category": article.Category,
				},
			}),
		)

		if len(createResult.GQL.Errors) > 0 {
			return createResult.GQL.Errors[0]
		}
	}

	return nil
}

func setupKnowledgeBase() error {
	n, err := initDefraNode()
	if err != nil {
		return err
	}

	if err := loadWikiData(n); err != nil {
		return err
	}

	db = n
	return nil
}

type askReq struct {
	Question string `json:"question"`
}
type askResp struct {
	Answer string `json:"answer"`
}

func main() {
	once.Do(func() {
		initErr = setupKnowledgeBase()
	})
	if initErr != nil {
		log.Fatalf("Failed KB setup: %v", initErr)
	}
	log.Println("Knowledge base initialized and HTTP API server is running at :8080")

	http.HandleFunc("/ask", askHandler)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func askHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method == "OPTIONS" {
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		return
	}
	if r.Method != "POST" {
		http.Error(w, "only POST allowed", http.StatusMethodNotAllowed)
		return
	}
	var req askReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || strings.TrimSpace(req.Question) == "" {
		http.Error(w, "invalid payload", http.StatusBadRequest)
		return
	}
	answer, err := handleRAGRequest(r.Context(), req.Question)
	if err != nil {
		log.Printf("Pipeline error: %v", err)
		http.Error(w, "internal error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(askResp{Answer: answer})
}

func handleRAGRequest(ctx context.Context, question string) (string, error) {
	// 1. Build query embedding
	openAIClient := openai.NewClientWithConfig(openai.ClientConfig{
		BaseURL:    ollamaBaseURL,
		HTTPClient: http.DefaultClient,
	})
	queryWithPrefix := "search_query: " + question
	embeddingResp, err := openAIClient.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{queryWithPrefix},
		Model: embeddingModel,
	})
	if err != nil {
		return "", err
	}
	// 2. Retrieve top relevant docs
	queryResult := db.DB.ExecRequest(
		ctx,
		`query Search($queryVector: [Float32!]!) {
            Wiki(
                filter: {_alias: {sim: {_gt: 0.63}}},
                limit: 2,
                order: {_alias: {sim: DESC}}
            ) {
                text
                sim: _similarity(text_v: {vector: $queryVector})
            }
        }`,
		client.WithVariables(map[string]interface{}{
			"queryVector": embeddingResp.Data[0].Embedding,
		}),
	)
	contexts := []string{}
	if found, ok := queryResult.GQL.Data.(map[string]interface{})["Wiki"].([]interface{}); ok {
		for _, resAny := range found {
			res, ok := resAny.(map[string]interface{})
			if !ok {
				continue
			}
			content := strings.TrimPrefix(res["text"].(string), "search_document: ")
			contexts = append(contexts, content)
		}
	}
	// 3. Ask LLM with retrieved context
	answer := askLLM(ctx, contexts, question)
	return answer, nil
}

func askLLM(ctx context.Context, contexts []string, question string) string {
	openAIClient := openai.NewClientWithConfig(openai.ClientConfig{
		BaseURL:    ollamaBaseURL,
		HTTPClient: http.DefaultClient,
	})
	sb := &strings.Builder{}
	_ = systemPromptTpl.Execute(sb, contexts)
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: sb.String()},
		{Role: openai.ChatMessageRoleUser, Content: "Question: " + question},
	}
	res, err := openAIClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    llmModel,
		Messages: messages,
	})
	if err != nil {
		log.Printf("LLM error: %v", err)
		return "Sorry, I couldn't generate an answer."
	}
	return strings.TrimSpace(res.Choices[0].Message.Content)
}
