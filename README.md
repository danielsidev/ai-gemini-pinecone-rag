# RAG  with Gemini and Pinecone

An Example of RAG with Golang, Gemini and Pinecone.

![Gemini Flash](https://img.shields.io/badge/Gemini-2.0_Flash-886CE4?style=for-the-badge&logo=googlegemini&logoColor=white) ![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=for-the-badge&logoColor=white) <img src="https://s3.amazonaws.com/appforest_uf/f1679157815668x357855949495047500/io6cC6vZ_400x400.png" alt="Claude AI" width="28" height="28"/> ![Go](https://img.shields.io/badge/Go-1.23+-00ADD8?style=for-the-badge&logo=go&logoColor=white)

## Requirements

We need some API KEYs:

- PINECONE API KEY (create an free account in [https://app.pinecone.io/](https://app.pinecone.io/) and generate your api key)
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