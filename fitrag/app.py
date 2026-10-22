import streamlit as st
from rag import ask, CHUNKS, MIN_SCORE

st.set_page_config(page_title="FitDocs RAG", page_icon="🏋️")
st.title("🏋️ FitDocs - Ask Your Health & Fitness Documents")
st.caption("Answers come only from the documents in /data. Unanswerable questions get “I don't know”.")

with st.sidebar:
    st.subheader("Knowledge base")
    for s in sorted({c["title"] for c in CHUNKS}):
        st.write("•", s)
    st.write(f"{len(CHUNKS)} chunks · min cosine {MIN_SCORE}")
    st.info("Educational only. Not medical advice.")

examples = ["How much protein do I need to build muscle?",
            "How many minutes of cardio per week should adults do?",
            "What is DOMS?"]
q = st.text_input("Ask a question", placeholder=examples[0])
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        q = ex

if q:
    with st.spinner("Searching documents..."):
        r = ask(q)
    st.subheader("Answer")
    st.write(r["answer"])
    st.caption(f"Top retrieval score: {r['top_score']:.2f}")
    if r["sources"]:
        st.subheader("Sources")
        for i, s in enumerate(r["sources"], 1):
            with st.expander(f"[{i}] {s['title']} · {s['source']} · score {s['score']:.2f}"):
                st.write(s["text"])
