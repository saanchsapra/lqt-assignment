
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME

_model = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")
