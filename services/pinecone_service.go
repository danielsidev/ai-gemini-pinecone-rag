package services

import (
	"ai_gemini_pinecone_rag/constants"
	"ai_gemini_pinecone_rag/models"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/pinecone-io/go-pinecone/v4/pinecone"
	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/structpb"
)

// Generate vector for a text using a Gemini Embeddings API.
func CreateEmbedding(ctx context.Context, client *genai.Client, text string) ([]float32, error) {
	contents := []*genai.Content{
		genai.NewContentFromText(text, genai.RoleUser),
	}
	result, err := client.Models.EmbedContent(ctx,
		constants.EmbeddingModel,
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

// ingestDocument create embedding to document and insert on Pinecone.
func IngestDocument(ctx context.Context, geminiClient *genai.Client, pcClient *pinecone.Client, doc models.Document) error {
	log.Printf("1. Generating embedding for document ID: %s...", doc.ID)
	vector, err := CreateEmbedding(ctx, geminiClient, doc.Content)
	if err != nil {
		return fmt.Errorf("fail on insert (embedding): %w", err)
	}

	log.Printf("2. Inserting (Upsert) vector on Pinecone (ID: %s)...", doc.ID)

	index, err := pcClient.Index(pinecone.NewIndexConnParams{Host: constants.INDEX_HOST})
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
		return fmt.Errorf("failn in insert (upsert Pinecone): %w", err)
	}
	log.Println("3. Document inserted with succes on Pinecone.")
	time.Sleep(2 * time.Second)

	return nil
}

// retrieveContext to find document more relevants on Pinecone.
func RetrieveContext(ctx context.Context, pcClient *pinecone.Client, queryEmbedding []float32) ([]models.Document, error) {
	log.Println("Buscando contexto relevante no Pinecone...")

	index, err := pcClient.Index(pinecone.NewIndexConnParams{Host: constants.INDEX_HOST})

	// Query to 3 vectors  closest
	resp, err := index.QueryByVectorValues(ctx, &pinecone.QueryByVectorValuesRequest{
		Vector:          queryEmbedding,
		TopK:            3,
		IncludeMetadata: true, // Important to retrieve original text
	})

	if err != nil {
		return nil, fmt.Errorf("fail to find on Pinecone: %w", err)
	}

	var documents []models.Document
	for _, match := range resp.Matches {
		if match.Vector.Metadata != nil {
			content, ok := match.Vector.Metadata.AsMap()["content"].(string)
			if !ok {
				log.Printf("Warning: Metadata 'content' off or is not string  ID: %s", match.Vector.Id)
				continue
			}
			documents = append(documents, models.Document{
				ID:      match.Vector.Id,
				Content: content,
			})
		}

	}
	return documents, nil
}
