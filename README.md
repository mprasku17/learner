# Mini Project 2: Sherlock Holmes RAG

This project builds a lightweight Retrieval-Augmented Generation (RAG) pipeline over a small Sherlock Holmes corpus. The system reads story text, chunks it into manageable segments, embeds each chunk with OpenAI embeddings, stores the vectors in Qdrant, retrieves relevant passages for a user question, and generates an answer grounded in the provided corpus.

## Overview

The workflow is:

1. Load story text from the `corpus/` folder
2. Split documents into chunks with overlap
3. Generate embeddings using OpenAI's `text-embedding-3-small` model
4. Store vectors in Qdrant
5. Query the vector store with a natural-language question
6. Retrieve the most relevant chunks
7. Ask an LLM to answer using only the retrieved context
8. Validate answers against predefined questions

## Project structure

```text
learner/
├── .env.example          # Example environment variables
├── .env                  # Local secrets (not committed)
├── corpus/               # Sherlock Holmes source text files
├── data/                 # Evaluation questions and answer fixtures
├── mp2_rag.py            # Main project implementation
├── mp2_rag_starter.py    # Starter reference implementation
├── MP2_FAQ.md            # Common issues and guidance
├── requirements.txt      # Python dependencies
├── test.py               # Local testing helper
└── README.md             # Project documentation
```

## Tech stack

- Python 3.10+
- OpenAI API for embeddings and chat generation
- Qdrant vector database
- `python-dotenv` for loading local environment variables

## Prerequisites

Before running the project, make sure you have:

- An OpenAI API key
- A running Qdrant instance
- Python dependencies installed

## Setup

1. Create and activate a virtual environment if needed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the sample environment file and fill in the real values:

```bash
cp .env.example .env
```

Then update `.env` with your credentials:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional if using the default
export QDRANT_URL="http://localhost:6333"            # or your cloud Qdrant URL
export QDRANT_API_KEY="your-qdrant-api-key"          # optional for local Qdrant
```

## Running Qdrant locally

If you want to use a local Qdrant database via Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Then set:

```bash
export QDRANT_URL="http://localhost:6333"
unset QDRANT_API_KEY
```

## Usage

Load the corpus and build the vector collection:

```bash
source .env
python mp2_rag.py ingest
```

Start an interactive Q&A session:

```bash
python mp2_rag.py ask
```

Run validation against the predefined question set:

```bash
python mp2_rag.py validate
```

## Notes

- The project uses `gpt-4o-mini` for generation and `text-embedding-3-small` for embeddings.
- The chunk size is tuned for narrative text and uses a small overlap to preserve context.
- The retrieval layer stores metadata including the source story and section to support citation-aware responses.
- The validation command checks whether retrieved passages match expected story sources and answer facts.

## Troubleshooting

Common issues are covered in `MP2_FAQ.md`, including:

- missing `OPENAI_API_KEY`
- Qdrant connectivity problems
- local-vs-cloud Qdrant configuration
- Windows shell differences with `source .env`

## License

This project is intended for educational and assignment use within the course context.
