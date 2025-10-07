import streamlit as st
from src.rag.retriever import build_hybrid_retriever
from src.rag.llm_client import LLMClient
from src.utils.helpers import load_yaml

st.set_page_config(page_title="Q&A with Citations", layout="wide")
st.title("Q&A over 10-Ks (Hybrid Retrieval + Rerank)")

cfg = load_yaml("config/config.yaml")
qs = st.text_input("Ask a question", placeholder="e.g., What are the key risk factors mentioned?")
ticker = st.session_state.get("ticker", "AAPL")

if "qa_init" not in st.session_state:
    st.session_state.llm = LLMClient()
    st.session_state.qa_init = True

if st.button("Search") and qs:
    retriever = build_hybrid_retriever(ticker, k=6, bm25_weight=0.4, dense_weight=0.6)
    docs = retriever.get_relevant_documents(qs) if hasattr(retriever, "get_relevant_documents") else []
    if not docs:
        st.warning("No indexed documents found. Go to Setup to index 10-Ks for your ticker.")
    else:
        st.subheader("Top Contexts")
        previews = []
        for d in docs[:5]:
            meta = getattr(d, "metadata", {}) or {}
            section = meta.get("section", "Section")
            source = meta.get("source", "")
            citation = f"{section} [{source}]" if source else section
            text = getattr(d, "page_content", "")[:180].replace("\n", " ")
            previews.append(f"- [{citation}] {text}...")
        st.markdown("\n".join(previews))
        context = "\n\n".join(f"[{(getattr(d, 'metadata', {}) or {}).get('section','Section')}] {getattr(d, 'page_content','')}" for d in docs[:4])
        prompt = f"Answer the question strictly based on the context. Cite the sections.\n\nContext:\n{context}\n\nQuestion: {qs}\nAnswer:"
        ans = st.session_state.llm.generate_answer(prompt)
        st.subheader("Answer")
        st.write(ans)