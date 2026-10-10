import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.pipeline import build_pipeline

st.set_page_config(page_title="Self-Reflective RAG", page_icon="🔎", layout="wide")


@st.cache_resource
def get_pipeline():
    return build_pipeline()


st.title("🔎 Self-Reflective RAG")
st.caption("Retrieves, grades its own context, verifies its answer, and retries when unsure.")

mode = st.sidebar.radio("Mode", ["self_rag", "baseline"], help="Compare against plain RAG")
question = st.text_input("Ask a question about the indexed documents")

if st.button("Ask", type="primary") and question.strip():
    pipe = get_pipeline()
    with st.spinner("Thinking..."):
        res = pipe.answer(question) if mode == "self_rag" else pipe.baseline(question)

    if res.refused:
        st.warning(res.answer)
    else:
        st.success(res.answer)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{res.confidence:.0%}")
    c2.metric("Attempts", res.attempts)
    c3.metric("LLM calls", res.llm_calls)
    c4.metric("Latency (s)", res.latency_s)

    with st.expander("Sources"):
        for s in res.sources:
            st.markdown(f"**{s['source']}** (p.{s['page']}, score {s['score']})")
            st.caption(s["snippet"] + "...")

    with st.expander("Reasoning trace (what the system did)", expanded=True):
        for step in res.trace:
            st.json(step)
