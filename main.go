package main

import (
	"ai_gemini_pinecone_rag/ai"
	"ai_gemini_pinecone_rag/rag"
	"ai_gemini_pinecone_rag/services"
	"context"
	"fmt"
	"log"
)

func main() {
	ctx := context.Background()

	// --- 1. Starting Clients ---
	geminiClient, err := ai.ConnGemini(ctx)
	if err != nil {
		log.Fatalf("gemini client error: %v", err)
	}
	pineconeClient, err := rag.ConnPinecone(ctx)

	if err != nil {
		log.Fatalf("pinecone client error: %v", err)
	}

	// Find RAG (Retrieval Augmented Generation) ---
	query := "Who are the creators of Go and when was it released?"

	fmt.Printf("\n============================================\n")
	fmt.Printf("ASK: %s\n", query)
	fmt.Printf("============================================\n")

	// A. Generate Embed to Find (Ask)
	queryEmbedding, err := services.CreateEmbedding(ctx, geminiClient, query)
	if err != nil {
		log.Fatalf("Fail to create embedding from query: %v", err)
	}

	// B. To Find Context (Retrieval)
	contextDocs, err := services.RetrieveContext(ctx, pineconeClient, queryEmbedding)
	if err != nil {
		log.Fatalf("Fail to find context from pinecone: %v", err)
	}

	// C. Increase and Generate Answer (Augmentation & Generation)
	response, err := services.GenerateRAGResponse(ctx, geminiClient, query, contextDocs)
	if err != nil {
		log.Fatalf("Fail to generate answer RAG: %v", err)
	}

	// --- 4. Result ---
	fmt.Println("\n--- Context Retrieve from Pinecone ---")
	for i, doc := range contextDocs {
		fmt.Printf("[%d] ID: %s, Content: \"%s\"\n", i+1, doc.ID, doc.Content)
	}
	fmt.Println("\n--- Response Generate from Gemini (Augmented) ---")
	fmt.Println(response)
}
