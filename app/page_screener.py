"""
S&P 500 screener page.

Scans the S&P 500 (or a small test list) using the production model
and displays a ranked table of investment opportunities with filters.
"""

import pandas as pd
import streamlit as st

from core.screener_engine import scan_sp500

_TEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "V", "PG"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render():
    st.title("S&P 500 Screener")
    st.markdown(
        "Scans the S&P 500 with the production model (19 features) "
        "and displays opportunities ranked by probability."
    )

    # --- Controls ---
    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.radio(
            "Scan mode",
            ["Quick test (10 tickers)", "Full S&P 500 (~15 min)"],
            horizontal=True,
        )
    with col2:
        show_all = st.checkbox("Include tickers without signal", value=True)

    if st.button("Scan", type="primary"):
        tickers = _TEST_TICKERS if "Quick" in mode else None

        with st.spinner("Scanning... this may take several minutes."):
            try:
                results = scan_sp500(tickers=tickers, include_failing=show_all)
            except FileNotFoundError:
                st.error(
                    "Model not found. Run **`make pipeline`** to train."
                )
                return

        if results.empty:
            st.warning("No results found.")
            return

        st.session_state["screener_results"] = results

    # --- Results ---
    if "screener_results" in st.session_state:
        _display_results(st.session_state["screener_results"])


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

_DISPLAY_COLS = [
    "ticker", "close", "probability", "signal_strength",
    "signal_freshness_days", "sma_headroom_pct", "passes_filters",
    "pe_ratio", "peg_ratio", "fcf_yield",
]

_FORMAT_MAP = {
    "probability": "{:.1%}",
    "signal_strength": "{:.1%}",
    "sma_headroom_pct": "{:.1%}",
    "close": "${:.2f}",
    "pe_ratio": "{:.1f}",
    "peg_ratio": "{:.2f}",
    "fcf_yield": "{:.2%}",
}


def _display_results(df: pd.DataFrame):
    # Summary
    n_pass = int(df["passes_filters"].sum())
    st.success(f"**{n_pass}** opportunities out of **{len(df)}** tickers analyzed.")

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        min_prob = st.slider("Minimum probability", 0.0, 1.0, 0.0, 0.05)
    with col_f2:
        only_passing = st.checkbox("Only tickers passing filters", value=False)

    filtered = df[df["probability"] >= min_prob].copy()
    if only_passing:
        filtered = filtered[filtered["passes_filters"]].copy()

    if filtered.empty:
        st.info("No tickers match the selected filters.")
        return

    # Select available columns
    available = [c for c in _DISPLAY_COLS if c in filtered.columns]
    fmt = {k: v for k, v in _FORMAT_MAP.items() if k in available}

    # Apply row highlighting: green for passing, neutral otherwise.
    # Use semi-transparent colors that work on both light and dark themes,
    # and force text color so it's always readable.
    def _highlight_passing(row):
        if row.get("passes_filters"):
            return [
                "background-color: rgba(76, 175, 80, 0.18); color: inherit"
            ] * len(row)
        return [""] * len(row)

    styled = (
        filtered[available]
        .style
        .format(fmt, na_rep="N/A")
        .apply(_highlight_passing, axis=1)
    )

    st.dataframe(styled, use_container_width=True, height=600)

    # Download button
    csv = filtered[available].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="screener_results.csv",
        mime="text/csv",
    )
