# RAG Observer — Event-Driven PDF Ingestion & Question Answering

A local Retrieval-Augmented Generation (RAG) demo for ingesting PDFs and answering questions over their contents.

This repository uses:
- Streamlit for the user interface
- Inngest for event-driven workflows
- FastAPI for serving the workflow endpoint
- Gemini for embeddings and text generation
- Qdrant for vector search
- LlamaIndex for PDF reading and text chunking

## Project Structure

- `streamlit_app.py` — Streamlit frontend for upload and query.
- `main.py` — FastAPI + Inngest workflow definitions.
- `data_loader.py` — PDF loading, chunking, and Gemini embedding.
- `vector_db.py` — Qdrant wrapper for vector storage and search.
- `custom_types.py` — Pydantic models for typed workflow outputs.

## How It Works

### Ingesting PDFs

1. User uploads a PDF via Streamlit.
2. `streamlit_app.py` saves the file under `uploads/`.
3. Streamlit sends an Inngest event named `rag/ingest_pdf`.
4. The FastAPI worker receives the event at `/api/inngest`.
5. `main.py` runs the ingest workflow:
   - `load_chunk_pdf()` loads and chunks the PDF text.
   - `embed_text()` generates embeddings using Gemini.
   - `QdrantStorage.upsert()` stores vectors and metadata in Qdrant.

Stored payloads include:
- `source`: the PDF filename
- `text`: the chunk content

### Answering Questions

1. User submits a question in Streamlit.
2. Streamlit sends an Inngest event named `rag/query_pdf_ai`.
3. The worker embeds the question and queries Qdrant.
4. Qdrant returns the highest-scoring document chunks.
5. The chunks are composed into a prompt.
6. Gemini generates an answer using only the retrieved context.
7. Streamlit displays the answer and the source filenames.

## Requirements

- Python 3.12+
- Qdrant
- Gemini API key
- Inngest dev server
- `uv` CLI (used in run examples)

## For Other Users

If you want to try this project locally, follow these steps:

1. Clone the repository.
2. Create and activate a Python 3.12+ virtual environment.
3. Install the required libraries.
4. Create a `.env` file with your Gemini API key.
5. Start Qdrant, the FastAPI worker, the Inngest dev server, and Streamlit.
6. Open the Streamlit app, upload a PDF, and ask a question.

Basic setup commands:

```powershell
uv run python -m venv .venv
.\.venv\Scripts\Activate.ps1
uv run pip install -r requirements.txt
```


## Environment Variables

Create a `.env` file in the project root with one of the supported Gemini key names:

```env
GEMINI_API=your_gemini_api_key
```

Optional override for the Inngest local API base URL:

```env
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

## Running Locally

### 1. Start Qdrant

```powershell
docker run -p 6333:6333 qdrant/qdrant
```

Expected URL:

```text
http://localhost:6333
```

### 2. Start the FastAPI worker

```powershell
uv run uvicorn main:app
```

This exposes:

```text
http://127.0.0.1:8000/api/inngest
```

### 3. Start Inngest dev server

```powershell
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
```

### 4. Start Qdrant locally

```powershell
docker run -d --name qdrant_v2 -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 5. Start Streamlit

```powershell
uv run streamlit run streamlit_app.py
```

Open the app at:

```text
http://localhost:8501
```

## Notes

- `streamlit_app.py` only sends Inngest events and polls for run output.
- The actual ingestion and query workflows execute in `main.py` via FastAPI and Inngest.
- If the FastAPI server is not running, ingestion and querying will fail.

## Troubleshooting

### Port 8000 is already in use

```powershell
netstat -ano | Select-String ':8000'
```

Then stop the process by PID:

```powershell
Stop-Process -Id <PID> -Force
```

### Inngest cannot connect to localhost:8000

Make sure the FastAPI worker is running:

```powershell
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

### Qdrant connection fails

Ensure Qdrant is running and accessible at `http://localhost:6333`.

### Query times out

Check FastAPI and Inngest logs. The main failure points are:
- Gemini embedding API
- Qdrant search
- Gemini answer generation

