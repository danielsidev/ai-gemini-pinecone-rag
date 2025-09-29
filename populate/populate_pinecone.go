package main

import (
	"ai_gemini_pinecone_rag/ai"
	"ai_gemini_pinecone_rag/models"
	"ai_gemini_pinecone_rag/rag"
	"ai_gemini_pinecone_rag/services"
	"context"
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
	doc := models.Document{
		ID:      "doc-golang-origin",
		Content: "Go (also known as Golang) was designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson. Its first public version was released in November 2009.",
	}

	err = services.IngestDocument(ctx, geminiClient, pineconeClient, doc)
	if err != nil {
		log.Fatalf("Fail on insert document: %v", err)
	}

	log.Println("Document inserted in pinecone with success!")

}
