
import json
from typing import List, Dict, Tuple
import faiss
from pathlib import Path
from .config import INDEX_PATH, META_PATH

class VectorDB:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = None
        self.metadata: List[Dict] = []
        self._load()

    def _load(self):
        if Path(INDEX_PATH).exists() and Path(META_PATH).exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            # cosine similarity via inner product on normalized vectors
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    def save(self):
        Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add(self, vectors, metadatas: List[Dict]):
        assert vectors.shape[0] == len(metadatas)
        if not isinstance(self.index, faiss.IndexFlatIP):
            raise ValueError("Index type mismatch.")
        self.index.add(vectors)
        for m in metadatas:
            self.metadata.append(m)

    def search(self, query_vec, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        scores, idxs = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((float(score), self.metadata[idx]))
        return results
