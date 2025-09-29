package services

import (
	"ai_gemini_pinecone_rag/constants"
	"ai_gemini_pinecone_rag/models"
	"context"
	"fmt"
	"log"

	"google.golang.org/genai"
)

func GenerateRAGResponse(ctx context.Context, client *genai.Client, query string, contextDocs []models.Document) (string, error) {
	// Constrói o contexto recuperado em um formato legível
	contextText := ""
	for i, doc := range contextDocs {
		contextText += fmt.Sprintf("--- CONTEXTO %d ---\n%s\n", i+1, doc.Content)
	}

	// System Instruction is core of RAG
	systemInstruction := `You are a helpful and concise RAG assistant.
Use ONLY the 'RETRIEVAL CONTEXTS' provided below to answer the 'QUESTION'.
If the answer CANNOT be found in the contexts provided, respond that 'The information is not available in the context provided.'
Be direct and cite the facts from the contexts.`

	// Final Prompt to  LLM
	fullPrompt := fmt.Sprintf("%s\n\nRETRIVE CONTEXT :\n%s\n\nASK: %s", systemInstruction, contextText, query)

	log.Println("Calling Gemini with RAG Prompt...")

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
	resp, err := client.Models.GenerateContent(ctx, constants.GenerationModel, genai.Text(fullPrompt), config)
	if err != nil {
		return "", fmt.Errorf("falha na chamada GenerateContent do Gemini: %w", err)
	}

	// Extrai a resposta de texto
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		return fmt.Sprintf("%v", resp.Candidates[0].Content.Parts[0]), nil
	}

	return "Não foi possível gerar uma resposta ou a resposta está vazia.", nil
}
