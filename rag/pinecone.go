package rag

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/pinecone-io/go-pinecone/v4/pinecone"
)

func ConnPinecone(ctx context.Context) (*pinecone.Client, error) {

	pcAPIKey := os.Getenv("PINECONE_API_KEY")

	if pcAPIKey == "" {
		return nil, fmt.Errorf("PINECONE_API_KEY is not defined")
	}

	pineconeClient, err := pinecone.NewClient(pinecone.NewClientParams{ApiKey: pcAPIKey})
	if err != nil {
		return nil, fmt.Errorf("erro to create pinecone client: %w", err)
	}
	log.Println("Pinecone client started.")

	return pineconeClient, nil
}
