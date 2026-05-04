"""
app.py — Streamlit entrypoint

Run locally:
    streamlit run app/app.py

With Ollama backend:
    GEMMA_BACKEND=ollama GEMMA_MODEL=gemma3:9b-instruct-q4_K_M streamlit run app/app.py

With OpenAI fallback (default):
    GEMMA_BACKEND=openai streamlit run app/app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Job Application AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pages defined as st.Page objects for st.navigation
analyze_page = st.Page("pages/analyze.py", title="Analyze Resume", icon="📄", default=True)
search_page = st.Page("pages/search.py", title="Find Jobs", icon="🔍")
history_page = st.Page("pages/history.py", title="History", icon="📋")

nav = st.navigation([analyze_page, search_page, history_page])

# Sidebar: backend indicator
import os
backend = os.getenv("GEMMA_BACKEND", "lmstudio").lower()
if backend == "lmstudio":
    host = os.getenv("LM_STUDIO_HOST", "http://localhost:1234")
    model_name = os.getenv("LM_STUDIO_MODEL", "") or "auto-detect"
    backend_label = f"LM Studio  ({host})"
elif backend == "ollama":
    model_name = os.getenv("GEMMA_MODEL", "gemma3:9b-instruct-q4_K_M")
    backend_label = "Ollama"
else:
    model_name = os.getenv("OPENAI_SCORING_MODEL", "gpt-4.1-nano")
    backend_label = "OpenAI"
with st.sidebar:
    st.markdown("### Job Application AI")
    st.caption(f"Backend: **{backend_label}**  \nModel: `{model_name}`")
    session_count = len(st.session_state.get("analyses", []))
    if session_count:
        st.caption(f"Analyses this session: **{session_count}**")

nav.run()
