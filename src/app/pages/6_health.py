import os
import requests
import streamlit as st
from src.utils.helpers import get_env

st.set_page_config(page_title="Health", layout="wide")
st.title("System Health")

st.subheader("Environment")
env_vars = ["USER_AGENT", "FRED_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "LOG_LEVEL"]
rows = []
for e in env_vars:
    val = get_env(e)
    rows.append((e, "SET" if val else "MISSING", val or ""))
st.table(rows)

st.subheader("Ollama")
ollama_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
ok = False
try:
    r = requests.get(f"{ollama_url}/api/tags", timeout=5)
    ok = r.ok
except Exception:
    ok = False
st.metric("Ollama reachable", "Yes" if ok else "No")
st.caption(f"Checked: {ollama_url}/api/tags")

st.subheader("Paths")
for p in ["data/raw", "data/processed", "data/analytics", "data/analytics/chroma"]:
    st.write(f"{p}: {'OK' if os.path.exists(p) else 'Missing'}")