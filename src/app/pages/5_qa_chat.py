import streamlit as st
from src.storage.vector_store import VectorStore
from src.rag.retriever import HybridRetriever
from src.rag.reranker import Reranker
from src.rag.llm_client import LLMClient
from src.utils.helpers import load_yaml

st.set_page_config(page_title="Q&A with Citations", layout="wide")
st.title("Q&A over 10-Ks (Hybrid Retrieval + Rerank)")

cfg = load_yaml("config/config.yaml")
qs = st.text_input("Ask a question", placeholder="e.g., What are the key risk factors mentioned?")
ticker = st.session_state.get("ticker", "AAPL")

if "qa_init" not in st.session_state:
    st.session_state.vs = VectorStore(persist_dir=cfg["storage"]["chroma_dir"], embedding_model=cfg["rag"]["embedding_model"])
    st.session_state.ret = HybridRetriever(st.session_state.vs)
    try:
        st.session_state.rr = Reranker(cfg["rag"]["reranker_model"])
    except Exception:
        st.session_state.rr = None
    st.session_state.llm = LLMClient()
    st.session_state.qa_init = True

where = {"ticker": {"$eq": ticker}} if ticker else None

if st.button("Search") and qs:
    hits = st.session_state.ret.hybrid_search(qs, k=cfg["rag"]["top_k"], alpha=cfg["rag"]["alpha"], where=where)
    hits = st.session_state.ret.add_citations(hits)
    if not hits:
        st.warning("No indexed documents found. Go to Setup to index 10-Ks.")
    else:
        st.subheader("Top Contexts")
        for h in hits[:5]:
            st.markdown(f"- [{h['citation']}] {h['text'][:180]}...")
        if st.session_state.rr:
            ranked = st.session_state.rr.rerank(qs, [h["text"] for h in hits], top_k=min(5, len(hits)))
            hits = [hits[i] for i, _ in ranked]
        context = "\n\n".join(f"[{h['citation']}] {h['text']}" for h in hits[:4])
        prompt = f"Answer the question strictly based on the context. Cite the sections.\n\nContext:\n{context}\n\nQuestion: {qs}\nAnswer:"
        ans = st.session_state.llm.generate_answer(prompt)
        st.subheader("Answer")
        st.write(ans)