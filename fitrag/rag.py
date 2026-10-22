"""RAG pipeline: Ingest -> Retrieve (dense embeddings + FAISS) -> Generate -> Cite."""
import os, glob
import numpy as np
import faiss
from fastembed import TextEmbedding

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.55"))  # cosine similarity; below => "I don't know"
TOP_K = 3
IDK = "I don't know. I couldn't find this in the health & fitness documents."

# 1. INGEST: read markdown files, chunk by paragraph, keep source metadata
def load_chunks():
    chunks = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.md"))):
        raw = open(path, encoding="utf-8").read()
        title = raw.splitlines()[0].lstrip("# ").strip()
        for para in [p.strip() for p in raw.split("\n\n")]:
            if para.startswith("#"):
                para = "\n".join(para.splitlines()[1:]).strip()
            if len(para.split()) >= 8:
                chunks.append({"source": os.path.basename(path), "title": title, "text": para})
    return chunks

CHUNKS = load_chunks()
_model = TextEmbedding(EMBED_MODEL, cache_dir=os.getenv("FASTEMBED_CACHE_PATH") or None)

def _norm(x):
    x = np.asarray(x, dtype="float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)

_vecs = _norm(list(_model.embed([f"{c['title']}. {c['text']}" for c in CHUNKS])))
INDEX = faiss.IndexFlatIP(_vecs.shape[1])   # inner product on unit vectors = cosine
INDEX.add(_vecs)

# 2. RETRIEVE: dense top-k from FAISS
def retrieve(question, k=TOP_K):
    q = _norm(list(_model.query_embed([question])))
    scores, ids = INDEX.search(q, k)
    return [dict(CHUNKS[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

# 3. GENERATE: LLM answers only from retrieved context.
# Provider is chosen by whichever key is set: GROQ_API_KEY (free), GEMINI_API_KEY (free), ANTHROPIC_API_KEY.
SYSTEM = ("You answer health and fitness questions using ONLY the numbered context. "
          "Cite sources like [1]. If the context does not clearly answer the question, "
          f"reply exactly: {IDK} Do not use outside knowledge. Do not give medical diagnoses.")

def generate(question, hits):
    import requests
    context = "\n\n".join(f"[{i+1}] ({h['title']}) {h['text']}" for i, h in enumerate(hits))
    user = f"Context:\n{context}\n\nQuestion: {question}"
    groq, gem, ant = (os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"))
    try:
        if groq:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq}"}, timeout=60,
                json={"model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), "temperature": 0,
                      "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        if gem:
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": gem}, timeout=60,
                json={"systemInstruction": {"parts": [{"text": SYSTEM}]},
                      "contents": [{"role": "user", "parts": [{"text": user}]}],
                      "generationConfig": {"temperature": 0}})
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        if ant:
            import anthropic
            msg = anthropic.Anthropic(api_key=ant).messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"), max_tokens=400, system=SYSTEM,
                messages=[{"role": "user", "content": user}])
            return msg.content[0].text
    except Exception as e:
        return f"(LLM call failed: {e})\n\nBest passage:\n{hits[0]['text']}"
    return "(No API key set - showing best passage)\n\n" + hits[0]["text"]

# 4. CITE: return answer together with the passages used
def ask(question):
    hits = retrieve(question)
    if not hits or hits[0]["score"] < MIN_SCORE:
        return {"answer": IDK, "sources": [], "top_score": hits[0]["score"] if hits else 0.0}
    return {"answer": generate(question, hits), "sources": hits, "top_score": hits[0]["score"]}
