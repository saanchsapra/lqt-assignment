
from pathlib import Path
from typing import List
from .config import UPLOADS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from .pdf_utils import extract_pages, make_doc_chunks
from .models import embed_texts, get_embedding_model
from .vectordb import VectorDB

def ingest_pdfs(pdf_paths: List[str]):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure model is loaded to know embedding dim
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()
    vdb = VectorDB(dim=dim)

    all_texts = []
    all_metas = []

    for p in pdf_paths:
        src_name = Path(p).name
        pages = extract_pages(p)
        chunks = make_doc_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP, src_name)
        texts = [c["text"] for c in chunks]
        metas = [dict(source=src_name, page=chunk["metadata"]["page"], text=chunk["text"]) for chunk in chunks]

        all_texts.extend(texts)
        all_metas.extend(metas)

    if all_texts:
        vectors = embed_texts(all_texts)
        vdb.add(vectors, all_metas)
        vdb.save()
