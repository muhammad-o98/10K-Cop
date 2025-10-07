import streamlit as st
from src.pipeline.orchestrator import build_index_for_ticker_years
from src.storage.vector_store import VectorStore
from src.utils.helpers import load_yaml
from src.utils.logger import get_logger

logger = get_logger("setup_page")

st.set_page_config(page_title="Setup", layout="wide")
st.title("Setup and Ingestion")

cfg = load_yaml("config/config.yaml")

colA, colB = st.columns([2,1])
with colA:
    ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "AAPL")).strip().upper()
    years_str = st.text_input("Years (space-separated)", value="2023 2022").strip()
    if st.button("Download + Index 10-Ks"):
        years = [int(y) for y in years_str.split() if y.isdigit()]
        with st.status("Ingesting and indexing...", expanded=True) as status:
            try:
                build_index_for_ticker_years(ticker, years)
                status.update(label="Done", state="complete")
                st.success(f"Indexed {ticker} for years: {years}")
            except Exception as e:
                st.exception(e)
with colB:
    st.subheader("Vector Store")
    vs = VectorStore(persist_dir=cfg["storage"]["chroma_dir"], embedding_model=cfg["rag"]["embedding_model"])
    try:
        count = vs.col.count()
        st.metric("Indexed chunks", count)
        peek = vs.col.peek()
        if peek.get("metadatas"):
            st.caption("Sample metadata")
            st.json(peek["metadatas"][0])
    except Exception as e:
        st.warning(f"Chroma not initialized yet: {e}")
st.caption("Tip: Use the sidebar to set a global ticker and date range used by other pages.")