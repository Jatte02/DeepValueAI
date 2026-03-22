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
    st.title("Screener S&P 500")
    st.markdown(
        "Escanea el S&P 500 con el modelo de producción (19 features) "
        "y muestra las oportunidades ordenadas por probabilidad."
    )

    # --- Controls ---
    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.radio(
            "Modo de escaneo",
            ["Test rápido (10 tickers)", "S&P 500 completo (~15 min)"],
            horizontal=True,
        )
    with col2:
        show_all = st.checkbox("Incluir tickers sin señal", value=True)

    if st.button("Escanear", type="primary"):
        tickers = _TEST_TICKERS if "Test" in mode else None

        with st.spinner("Escaneando... esto puede tardar varios minutos."):
            try:
                results = scan_sp500(tickers=tickers, include_failing=show_all)
            except FileNotFoundError:
                st.error(
                    "Modelo no encontrado. Ejecuta **`make pipeline`** para entrenar."
                )
                return

        if results.empty:
            st.warning("No se encontraron resultados.")
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
    st.success(f"**{n_pass}** oportunidades de **{len(df)}** tickers analizados.")

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        min_prob = st.slider("Probabilidad mínima", 0.0, 1.0, 0.0, 0.05)
    with col_f2:
        only_passing = st.checkbox("Solo tickers que pasan filtros", value=False)

    filtered = df[df["probability"] >= min_prob].copy()
    if only_passing:
        filtered = filtered[filtered["passes_filters"]].copy()

    if filtered.empty:
        st.info("Ningún ticker cumple los filtros seleccionados.")
        return

    # Select available columns
    available = [c for c in _DISPLAY_COLS if c in filtered.columns]
    fmt = {k: v for k, v in _FORMAT_MAP.items() if k in available}

    # Apply row highlighting: green for passing, neutral otherwise
    def _highlight_passing(row):
        if row.get("passes_filters"):
            return ["background-color: #e8f5e9"] * len(row)
        return [""] * len(row)

    styled = (
        filtered[available]
        .style
        .format(fmt, na_rep="N/D")
        .apply(_highlight_passing, axis=1)
    )

    st.dataframe(styled, use_container_width=True, height=600)

    # Download button
    csv = filtered[available].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV",
        data=csv,
        file_name="screener_results.csv",
        mime="text/csv",
    )
