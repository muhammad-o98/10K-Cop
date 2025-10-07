import os
import sys

# Ensure project root is on sys.path so 'src' imports work whether launched via Streamlit or python
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import datetime as dt
from src.utils.helpers import load_yaml
from src.utils.logger import get_logger

logger = get_logger("app")

@st.cache_data
def load_config(path: str = "config/config.yaml"):
    return load_yaml(path)

def render_sidebar(cfg):
    st.sidebar.title("10K Cop")
    st.sidebar.caption("CFO-grade insights from 10-Ks")
    # Global selector
    default_ticker = st.session_state.get("ticker", "AAPL")
    ticker = st.sidebar.text_input("Ticker", value=default_ticker, help="Enter a stock ticker, e.g., AAPL")
    today = dt.date.today()
    default_start = st.session_state.get("start_date", today - dt.timedelta(days=365*3))
    default_end = st.session_state.get("end_date", today)
    dates = st.sidebar.date_input("Date range", value=(default_start, default_end))
    if isinstance(dates, tuple):
        start_date, end_date = dates
    else:
        start_date, end_date = default_start, default_end
    st.session_state["ticker"] = ticker.strip().upper()
    st.session_state["start_date"] = start_date
    st.session_state["end_date"] = end_date
    st.sidebar.divider()
    st.sidebar.caption(f"Model: {cfg['rag']['llm_model']} | Emb: {cfg['rag']['embedding_model']}")
    st.sidebar.page_link("src/app/pages/0_setup.py", label="Setup", icon="🧩")
    st.sidebar.page_link("src/app/pages/1_overview.py", label="Overview", icon="📊")
    st.sidebar.page_link("src/app/pages/2_ratios.py", label="Ratios", icon="🧮")
    st.sidebar.page_link("src/app/pages/3_sentiment.py", label="Sentiment", icon="🧠")
    st.sidebar.page_link("src/app/pages/4_forecasts.py", label="Forecasts", icon="📈")
    st.sidebar.page_link("src/app/pages/5_qa_chat.py", label="Q&A Chat", icon="💬")
    st.sidebar.page_link("src/app/pages/6_health.py", label="Health", icon="✅")

def main():
    st.set_page_config(page_title="10K Cop", layout="wide")
    cfg = load_config()
    render_sidebar(cfg)
    st.title("10K Cop")
    st.markdown("Select a page from the left to get started. Use the sidebar to set a global ticker and date range.")

if __name__ == "__main__":
    main()