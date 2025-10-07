import streamlit as st
import pandas as pd
from src.ingestion.price_fetcher import PriceFetcher
from src.ingestion.macro_fetcher import MacroFetcher
from src.ml.forecaster import Forecaster
from src.app.visualizations import trend_lines
from src.utils.helpers import load_yaml

st.set_page_config(page_title="Forecasts", layout="wide")
st.title("Forecasts (ARIMA baseline)")

cfg = load_yaml("config/config.yaml")
ticker = st.session_state.get("ticker", "AAPL")
start_date = st.session_state.get("start_date")
end_date = st.session_state.get("end_date")

col1, col2, col3 = st.columns(3)
with col1:
    h = st.number_input("Forecast horizon (days)", min_value=1, max_value=60, value=5)
with col2:
    use_macro = st.checkbox("Include macro features", value=False)
with col3:
    order = tuple(cfg.get("ml", {}).get("arima_order", [1,1,1])) if isinstance(cfg.get("ml", {}).get("arima_order", [1,1,1]), (list, tuple)) else (1,1,1)

pf = PriceFetcher()
prices = None
try:
    prices = pf.fetch_prices(ticker, start=str(start_date), end=str(end_date))
    prices = prices[["date","close"]].dropna().sort_values("date")
    st.subheader(f"{ticker} Price History")
    st.plotly_chart(trend_lines(prices, x="date", ys=["close"]), use_container_width=True)
except Exception as e:
    st.warning(f"Could not fetch prices: {e}")

macro_df = None
if use_macro:
    try:
        mf = MacroFetcher()
        series_ids = cfg.get("macro", {}).get("series", {})
        if series_ids:
            macro_df = mf.get_macro_features(series_ids)
    except Exception as e:
        st.warning(f"Macro fetch failed: {e}")

if prices is not None and not prices.empty:
    f = Forecaster()
    features = f.prepare_features(prices, macro_df)
    series = features.set_index("date")["close"]
    fc = f.arima_forecast(series, order=order, horizon=h)
    fc_df = pd.DataFrame({"date": pd.date_range(prices["date"].iloc[-1] + pd.Timedelta(days=1), periods=h), "forecast": fc.values})
    st.subheader("Forecast")
    show_df = pd.concat([
        prices.rename(columns={"close":"value"})[["date","value"]].assign(kind="history"),
        fc_df.rename(columns={"forecast":"value"}).assign(kind="forecast")
    ], ignore_index=True)
    st.plotly_chart(trend_lines(show_df, x="date", ys=["value"]), use_container_width=True)
    st.dataframe(fc_df, use_container_width=True)