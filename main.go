package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/pinecone-io/go-pinecone/v4/pinecone"
	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/structpb"
)

// Constantes
const (
	// Modelos Gemini
	embeddingModel  = "models/embedding-001"
	generationModel = "gemini-2.5-flash"

	// Configurações Pinecone
	// O índice deve ter dimensão 768 para ser compatível com o embedding-001
	pineconeIndexName = "gemini-index" // Mude para o nome do seu índice
	INDEX_HOST        = "https://gemini-index-3ke0q3k.svc.aped-4627-b74a.pinecone.io"
)

// https://gemini-index-3ke0q3k.svc.aped-4627-b74a.pinecone.io
// Document representa um chunk de texto e seu ID único
type Document struct {
	ID      string
	Content string
}

func main() {
	ctx := context.Background()

	// --- 1. Inicializar Clientes ---
	geminiClient, pineconeClient, err := initializeClients(ctx)
	if err != nil {
		log.Fatalf("Falha na inicialização dos clientes: %v", err)
	}

	// --- 2. Fase de Ingestão (Pré-requisito RAG) ---
	// Você precisa rodar esta fase pelo menos uma vez para popular o Pinecone.
	// O documento baseia-se em um fato pouco conhecido.
	doc := Document{
		ID:      "doc-golang-origem",
		Content: "O Go (também conhecido como Golang) foi projetado no Google por Robert Griesemer, Rob Pike e Ken Thompson. Sua primeira versão pública foi lançada em novembro de 2009.",
	}

	err = ingestDocument(ctx, geminiClient, pineconeClient, doc)
	if err != nil {
		log.Fatalf("Falha na ingestão do documento: %v", err)
	}

	// --- 3. Consulta RAG (Retrieval Augmented Generation) ---
	query := "Quem são os criadores do Go e quando foi lançado?"

	fmt.Printf("\n============================================\n")
	fmt.Printf("PERGUNTA: %s\n", query)
	fmt.Printf("============================================\n")

	// A. Embed a Consulta (Pergunta)
	queryEmbedding, err := createEmbedding(ctx, geminiClient, query)
	if err != nil {
		log.Fatalf("Falha ao criar embedding da consulta: %v", err)
	}

	// B. Buscar Contexto (Retrieval)
	contextDocs, err := retrieveContext(ctx, pineconeClient, queryEmbedding)
	if err != nil {
		log.Fatalf("Falha ao buscar contexto no Pinecone: %v", err)
	}

	// C. Aumentar e Gerar Resposta (Augmentation & Generation)
	response, err := generateRAGResponse(ctx, geminiClient, query, contextDocs)
	if err != nil {
		log.Fatalf("Falha ao gerar resposta RAG: %v", err)
	}

	// --- 4. Resultado ---
	fmt.Println("\n--- Contexto Recuperado do Pinecone ---")
	for i, doc := range contextDocs {
		fmt.Printf("[%d] ID: %s, Conteúdo: \"%s\"\n", i+1, doc.ID, doc.Content)
	}
	fmt.Println("\n--- Resposta Gerada pelo Gemini (Augmented) ---")
	fmt.Println(response)
}

// -----------------------------------------------------
// --- Funções de Inicialização ---
// -----------------------------------------------------

func initializeClients(ctx context.Context) (*genai.Client, *pinecone.Client, error) {
	// 1. Inicialização do Cliente Gemini
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return nil, nil, fmt.Errorf("GEMINI_API_KEY não definida")
	}
	geminiClient, err := genai.NewClient(ctx, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("erro ao criar cliente Gemini: %w", err)
	}
	log.Println("Cliente Gemini inicializado.")

	// 2. Inicialização do Cliente Pinecone
	pcAPIKey := os.Getenv("PINECONE_API_KEY")

	if pcAPIKey == "" {
		return nil, nil, fmt.Errorf("PINECONE_API_KEY não definida")
	}

	// O cliente Pinecone Go usa uma configuração de ambiente/servidor
	pineconeClient, err := pinecone.NewClient(pinecone.NewClientParams{ApiKey: pcAPIKey})
	if err != nil {
		return nil, nil, fmt.Errorf("erro ao criar cliente Pinecone: %w", err)
	}
	log.Println("Cliente Pinecone inicializado.")

	return geminiClient, pineconeClient, nil
}

// -----------------------------------------------------
// --- Funções RAG ---
// -----------------------------------------------------

// createEmbedding gera o vetor para um texto usando a API de Embeddings do Gemini.
func createEmbedding(ctx context.Context, client *genai.Client, text string) ([]float32, error) {
	contents := []*genai.Content{
		genai.NewContentFromText(text, genai.RoleUser),
	}
	result, err := client.Models.EmbedContent(ctx,
		embeddingModel,
		contents,
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}
	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("embedding vazio retornado")
	}
	embeddings, err := json.MarshalIndent(result.Embeddings, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(embeddings))
	resp := result.Embeddings[0].Values

	return resp, nil
}

// ingestDocument cria o embedding do documento e o insere no Pinecone.
func ingestDocument(ctx context.Context, geminiClient *genai.Client, pcClient *pinecone.Client, doc Document) error {
	log.Printf("1. Gerando embedding para o documento ID: %s...", doc.ID)
	vector, err := createEmbedding(ctx, geminiClient, doc.Content)
	if err != nil {
		return fmt.Errorf("falha na ingestão (embedding): %w", err)
	}

	log.Printf("2. Inserindo (Upsert) o vetor no Pinecone (ID: %s)...", doc.ID)

	index, err := pcClient.Index(pinecone.NewIndexConnParams{Host: INDEX_HOST})
	metadataMap := map[string]interface{}{
		"content": doc.Content,
	}
	metadata, err := structpb.NewStruct(metadataMap)
	if err != nil {
		log.Fatalf("Failed to create metadata map: %v", err)
	}
	vectors := []*pinecone.Vector{
		{
			Id:       doc.ID,
			Values:   &vector,
			Metadata: metadata,
		},
	}
	_, err = index.UpsertVectors(ctx, vectors)

	if err != nil {
		return fmt.Errorf("falha na ingestão (upsert Pinecone): %w", err)
	}
	log.Println("3. Documento ingerido com sucesso no Pinecone.")
	// Dar um tempo para o Pinecone processar a inserção, especialmente em índices starter
	time.Sleep(2 * time.Second)

	return nil
}

// retrieveContext busca os documentos mais relevantes no Pinecone.
func retrieveContext(ctx context.Context, pcClient *pinecone.Client, queryEmbedding []float32) ([]Document, error) {
	log.Println("Buscando contexto relevante no Pinecone...")

	index, err := pcClient.Index(pinecone.NewIndexConnParams{Host: INDEX_HOST})

	// Consulta para os 3 vetores mais próximos
	resp, err := index.QueryByVectorValues(ctx, &pinecone.QueryByVectorValuesRequest{
		Vector: queryEmbedding,
		TopK:   3,
		// MetadataFilter:  metadata,
		IncludeMetadata: true, // Importante para recuperar o texto original
	})

	if err != nil {
		return nil, fmt.Errorf("falha na consulta ao Pinecone: %w", err)
	}

	var documents []Document
	for _, match := range resp.Matches {
		if match.Vector.Metadata != nil {
			content, ok := match.Vector.Metadata.AsMap()["content"].(string)
			if !ok {
				log.Printf("Aviso: Metadado 'content' ausente ou não é string para ID: %s", match.Vector.Id)
				continue
			}
			documents = append(documents, Document{
				ID:      match.Vector.Id,
				Content: content,
			})
		}

	}
	return documents, nil
}

// generateRAGResponse constrói o prompt RAG e chama o modelo Gemini para gerar a resposta.
func generateRAGResponse(ctx context.Context, client *genai.Client, query string, contextDocs []Document) (string, error) {
	// Constrói o contexto recuperado em um formato legível
	contextText := ""
	for i, doc := range contextDocs {
		contextText += fmt.Sprintf("--- CONTEXTO %d ---\n%s\n", i+1, doc.Content)
	}

	// A Instrução do Sistema (System Instruction) é o coração do RAG
	systemInstruction := `Você é um assistente RAG útil e conciso. 
    Use SOMENTE os 'CONTEXTOS DE RECUPERAÇÃO' fornecidos abaixo para responder à 'PERGUNTA'. 
    Se a resposta NÃO puder ser encontrada nos contextos fornecidos, responda que 'A informação não está disponível no contexto fornecido.'
    Seja direto e cite os fatos dos contextos.`

	// Prompt final para o LLM
	fullPrompt := fmt.Sprintf("%s\n\nCONTEXTOS DE RECUPERAÇÃO:\n%s\n\nPERGUNTA: %s", systemInstruction, contextText, query)

	log.Println("Chamando Gemini com o prompt RAG...")

	temperature := float32(0.7)
	topK := float32(40)
	topP := float32(0.95)
	maxTokens := int32(2048)
	config := &genai.GenerateContentConfig{
		Temperature:     &temperature,
		TopK:            &topK,
		TopP:            &topP, // This line is causing the error
		MaxOutputTokens: maxTokens,
	}
	resp, err := client.Models.GenerateContent(ctx, generationModel, genai.Text(fullPrompt), config)
	if err != nil {
		return "", fmt.Errorf("falha na chamada GenerateContent do Gemini: %w", err)
	}

	// Extrai a resposta de texto
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		return fmt.Sprintf("%v", resp.Candidates[0].Content.Parts[0]), nil
	}

	return "Não foi possível gerar uma resposta ou a resposta está vazia.", nil
}
