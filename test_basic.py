
from app.backend.pdf_utils import chunk_text

def test_chunker_basic():
    txt = ("para1\n\npara2 ") * 100
    chunks = chunk_text(txt, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
