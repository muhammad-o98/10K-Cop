import streamlit as st
from src.ingestion.price_fetcher import PriceFetcher
from src.app.visualizations import trend_lines

st.set_page_config(page_title="Overview", layout="wide")
st.title("Company Overview")

ticker = st.session_state.get("ticker", "AAPL")
start_date = st.session_state.get("start_date")
end_date = st.session_state.get("end_date")

st.subheader(f"Selected Ticker: {ticker}")
st.caption(f"Date range: {start_date} to {end_date}")

pf = PriceFetcher()
try:
    df = pf.fetch_prices(ticker, start=str(start_date), end=str(end_date))
    df = pf.calculate_indicators(df)
    st.dataframe(df.tail(), use_container_width=True)
    st.plotly_chart(trend_lines(df, x="date", ys=["close", "sma_20", "sma_50"]), use_container_width=True)
except Exception as e:
    st.warning(f"Could not fetch data for {ticker}: {e}")

st.info("Use the Setup page to ingest/index 10-Ks for the chatbot.")