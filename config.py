
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
STORAGE_DIR = DATA_DIR / "storage"

# Vector DB file names
INDEX_PATH = STORAGE_DIR / "faiss_index.bin"
META_PATH = STORAGE_DIR / "metadata.json"

# Embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# RAG generation backend: "openai" or "hf"
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "openai")

# OpenAI settings (used if GENERATION_BACKEND == "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hugging Face Transformers model (used if GENERATION_BACKEND == "hf")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/flan-t5-base")
HF_MAX_INPUT_TOKENS = int(os.getenv("HF_MAX_INPUT_TOKENS", "2048"))
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
