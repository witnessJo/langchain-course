# RAG Tutorial with LangChain and Chroma

This example loads a text document, stores its embeddings in a local Chroma database, and answers questions using the retrieved context.

## Setup

1. Install dependencies with `uv sync`.
2. Set `OPENAI_API_KEY` in `.env`.
3. Run `uv run python ingestion.py` to populate `./chroma_db`.
4. Run `uv run python main.py` to ask a question.

Both scripts use the `medium-blogs` collection in `./chroma_db`.
