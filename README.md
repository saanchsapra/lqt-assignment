
# RAG PDF Assistant

A complete, local-first Retrieval-Augmented Generation (RAG) app for querying one or more PDFs.  
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)  
- **Vector DB:** FAISS (cosine similarity via inner product on normalized vectors)  
- **UI:** Streamlit  
- **Generation:** OpenAI (if `OPENAI_API_KEY` is set) or Hugging Face (`google/flan-t5-base`) fallback.

## ✨ Features
- Upload multiple PDFs, extract text per page (PyMuPDF)
- Paragraph-based chunking with overlap
- Persisted FAISS index + JSON metadata
- Top-K semantic retrieval with scores
- RAG answer with inline citations like `[file.pdf p.X]`
- Fully local option using HF model (no external API required)
- Configurable via environment variables

## 🧱 Project Structure
```
app/
  app.py                 # Streamlit UI
  backend/
    config.py
    pdf_utils.py
    models.py
    vectordb.py
    ingest.py
    rag.py
  data/
    uploads/             # your PDFs
    storage/             # FAISS index + metadata.json
  tests/
requirements.txt
docker/Dockerfile
.env.example
```

## 🚀 Quickstart

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) (Optional) Configure
Copy `.env.example` to `.env` and edit, or set env vars directly:
```
# Generation backend: "openai" or "hf"
GENERATION_BACKEND=hf
# If using OpenAI:
OPENAI_API_KEY=sk-...
# Retrieval & chunking
TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=120
# Embedding model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
# HF local generation model
HF_MODEL_NAME=google/flan-t5-base
HF_MAX_INPUT_TOKENS=2048
HF_MAX_NEW_TOKENS=256
```

### 3) Run the app
```bash
cd app
streamlit run app.py
```
Open the URL shown (usually http://localhost:8501).

### 4) Use
1. Upload one or more PDFs.
2. Click **Ingest PDFs** to build/update the index.
3. Ask a question. The answer is generated from retrieved context with citations.

> To rebuild from scratch, delete files in `app/data/storage` and re-ingest.

## 🐳 Docker
```
docker build -t rag-pdf-assistant -f docker/Dockerfile .
docker run --rm -p 8501:8501 -v $PWD/app/data:/app/app/data rag-pdf-assistant
```

## 🧪 Tests
```
pytest -q
```

## 📎 Notes
- For best results, keep chunk size ~800 and overlap ~120.
- Large PDFs are processed page-by-page to keep memory usage low.
- If HF model download is slow, you can switch to OpenAI by setting `GENERATION_BACKEND=openai` and providing an API key.

## ❗ Limitations
- The HF fallback model is small; quality will be simpler than GPT-class models.
- This is a minimal reference; add auth, rate limits, and better logging for production.
