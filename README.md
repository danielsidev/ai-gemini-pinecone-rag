# RAG  with Gemini and Pinecone

An Example of RAG with Golang, Gemini and Pinecone.

## Requirements

We need some API KEYs:

- PINECONE AOI KEY (create an free account in [https://app.pinecone.io/](https://app.pinecone.io/) and generate your api key)
- GEMINI API KEY ( generate your free api key in [https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key) )


## Envs 
We use this ENVs:

- PINECONE_API_KEY
- PINECONE_INDEX
- GEMINI_API_KEY

```
export PINECONE_API_KEY=your-pinecone-api-key
export PINECONE_INDEX=your-pinecone-index
export GEMINI_API_KEY=your-gemini-api-key

```

## Populate Pinecone

First, It's necessary populate pinecone with an example of document.

For this, execute the following commands:

```
go run populate/populate_pinecone.go


```

## Execute
To start this example, after populate pinecone, execute the following commands:


```
go run main.go

```