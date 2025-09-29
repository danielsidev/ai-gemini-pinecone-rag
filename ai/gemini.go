package ai

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"
)

func ConnGemini(ctx context.Context) (*genai.Client, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY is not defined.")
	}
	geminiClient, err := genai.NewClient(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("erro to create gemini client: %w", err)
	}
	log.Println("Gemini client started.")
	return geminiClient, nil
}
