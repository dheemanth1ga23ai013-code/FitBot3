# FitDocs - Health & Fitness RAG (From Text to Insight Hack Hour)

**Pipeline:** Ingest (markdown -> paragraph chunks) -> Retrieve (dense embeddings via FastEmbed BGE-small + FAISS, top-3, cosine threshold) -> Generate (Claude, context-only) -> Cite (passages + scores shown).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."                # Windows: set ANTHROPIC_API_KEY=...
streamlit run app.py
```
The first run downloads the ~130 MB embedding model. Without an API key the app still runs and shows the best retrieved passage.
Tune the refusal threshold with `export MIN_SCORE=0.5` (cosine similarity).

## Test sheet (5 questions: 3 pass, 2 designed failure cases)
Run `python evaluate.py`, then paste the real outcomes below.

| # | Question | Designed as | Actual result |
|---|----------|-------------|---------------|
| 1 | Protein per kg for muscle gain? | Pass | |
| 2 | Why do my legs hurt two days after squats? | Pass (dense retrieval bridges the wording gap) | |
| 3 | Who won the 2022 World Cup? (out of domain) | Pass ("I don't know") | |
| 4 | How much protein should a pregnant woman eat? | FAIL - near-topic overreach | |
| 5 | Give me a full weekly plan combining cardio, strength and rest | FAIL - multi-document synthesis | |

**Fixes to try:** add a reranker or a stricter "does the context fully answer this?" check for #4; raise top-k or add query decomposition for #5.

## Deploy (Streamlit Community Cloud, free)
1. Push this folder to a public GitHub repo.
2. Go to share.streamlit.io -> New app -> pick repo, branch, main file `app.py`.
3. Advanced settings -> Secrets: `GROQ_API_KEY = "gsk_..."`
4. Deploy and share the live link.

## Deploy on Render
1. Push this folder to GitHub (`render.yaml` must be in the repo root).
2. Render dashboard -> **New +** -> **Blueprint** -> connect the repo -> **Apply**.
3. When prompted, enter `GROQ_API_KEY` (free; or add GEMINI_API_KEY / ANTHROPIC_API_KEY in Environment).
4. Wait for the build (3-6 min). Your app is live at `https://fitdocs-rag.onrender.com`.
Manual alternative: New + -> Web Service, Build Command `pip install -r requirements.txt && python -c "import rag"`, Start Command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`, and add the env vars from render.yaml.
