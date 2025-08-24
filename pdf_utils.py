
from typing import List, Dict
import fitz  # PyMuPDF
import re

def extract_pages(pdf_path: str) -> List[Dict]:
    # Extract text per page from a PDF using PyMuPDF.
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        # Normalize whitespace
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    # Paragraph-first chunking with overlap; falls back to hard splits if needed.
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= chunk_size:
            buf += "\n" + p
        else:
            chunks.append(buf.strip())
            tail = buf[-chunk_overlap:] if chunk_overlap > 0 else ""
            buf = (tail + "\n" + p).strip()
            while len(buf) > chunk_size:
                chunks.append(buf[:chunk_size].strip())
                tail = buf[:chunk_size][-chunk_overlap:] if chunk_overlap > 0 else ""
                buf = (tail + buf[chunk_size:]).strip()
    if buf:
        while len(buf) > chunk_size:
            chunks.append(buf[:chunk_size].strip())
            tail = buf[:chunk_size][-chunk_overlap:] if chunk_overlap > 0 else ""
            buf = (tail + buf[chunk_size:]).strip()
        if buf:
            chunks.append(buf.strip())
    return chunks

def make_doc_chunks(pages: List[Dict], chunk_size: int, chunk_overlap: int, source_name: str) -> List[Dict]:
    # Create chunk dicts with metadata from page texts.
    docs = []
    for p in pages:
        page_no = p["page"]
        for ch in chunk_text(p["text"], chunk_size, chunk_overlap):
            docs.append({
                "text": ch,
                "metadata": {"source": source_name, "page": page_no}
            })
    return docs
