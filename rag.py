
from typing import List, Dict
from .models import embed_texts
from .vectordb import VectorDB
from .config import TOP_K, GENERATION_BACKEND, OPENAI_API_KEY, HF_MODEL_NAME, HF_MAX_INPUT_TOKENS, HF_MAX_NEW_TOKENS

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    qvec = embed_texts([query])
    vdb = VectorDB()
    results = vdb.search(qvec[0], top_k=top_k)
    enriched = []
    for score, meta in results:
        enriched.append({
            "score": score,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "text": meta.get("text", ""),
        })
    return enriched

def generate_answer(query: str, contexts: List[Dict]) -> str:
    context_text = "\n\n".join([f"[{c['source']} p.{c['page']}] {c['text']}" for c in contexts])
    prompt = 'You are a helpful assistant. Answer the user\\'s question *using only* the provided context.\\n' \
             'If the answer is not in the context, say you do not know and suggest where it might be found.\\n' \
             'Be concise and cite sources inline like [source p.X].\\n\\n' \
             f'Question: {query}\\n\\nContext:\\n{context_text}\\n\\nAnswer:'

    if GENERATION_BACKEND == "openai":
        if not OPENAI_API_KEY:
            return "OpenAI API key not set. Set OPENAI_API_KEY or switch GENERATION_BACKEND to 'hf'."
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You answer strictly from the provided context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI error: {e}"

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=HF_MAX_INPUT_TOKENS)
        outputs = model.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
    except Exception:
        joined = "\\n\\n".join(c["text"] for c in contexts)
        return "(Fallback extractive answer)\\nRelevant excerpts:\\n" + joined[:1200]
