"""
DeepValue AI — Streamlit Dashboard

Main entry point.  Sidebar navigation to three views:
    1. Individual Analyzer — single ticker deep-dive
    2. S&P 500 Screener   — batch scan for opportunities
    3. Backtesting         — historical simulation with benchmark
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `import core` works
# even when Streamlit is launched with `cd app && streamlit run ...`.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st  # noqa: E402

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
st.sidebar.caption("ML-powered investment decision support system")

page = st.sidebar.radio(
    "Navigation",
    [
        "Individual Analyzer",
        "S&P 500 Screener",
        "Backtesting",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Models are generated with **`make pipeline`**.\n\n"
    "Without trained models the app cannot generate predictions."
)

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page == "Individual Analyzer":
    from page_analyzer import render
    render()
elif page == "S&P 500 Screener":
    from page_screener import render
    render()
else:
    from page_backtesting import render
    render()
