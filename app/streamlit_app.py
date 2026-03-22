"""
DeepValue AI — Streamlit Dashboard

Main entry point.  Sidebar navigation to three views:
    1. Analizador Individual — single ticker deep-dive
    2. Screener S&P 500     — batch scan for opportunities
    3. Backtesting           — historical simulation with benchmark
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `import core` works
# even when Streamlit is launched with `cd app && streamlit run ...`.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepValue AI",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("DeepValue AI")
st.sidebar.caption("Sistema de soporte a la inversión con ML")

page = st.sidebar.radio(
    "Navegación",
    [
        "Analizador Individual",
        "Screener S&P 500",
        "Backtesting",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Los modelos se generan con **`make pipeline`**.\n\n"
    "Sin modelos entrenados la app no puede generar predicciones."
)

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "Analizador Individual":
    from page_analyzer import render
    render()
elif page == "Screener S&P 500":
    from page_screener import render
    render()
else:
    from page_backtesting import render
    render()
