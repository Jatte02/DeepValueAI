"""
Individual ticker analyzer page.

Downloads OHLCV + fundamentals for a single ticker, runs the
production model (19 features), and displays:
    - Signal card (BUY / HOLD) with probability and confidence
    - Price chart with SMA 200
    - Technical indicator subplots (RSI, Williams %R, MACD)
    - Fundamental metrics summary
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.config import FEATURE_COLUMNS, FUNDAMENTAL_FEATURES, PATHS
from core.data_service import build_feature_row, download_ohlcv
from core.prediction_service import generate_signal, load_model, load_threshold


# ---------------------------------------------------------------------------
# Cached model loading — loaded once, reused across reruns
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo de producción...")
def _load_production_model():
    model = load_model(PATHS["model_file"])
    threshold = load_threshold(PATHS["threshold_file"])
    return model, threshold


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render():
    st.title("Analizador Individual")
    st.markdown(
        "Introduce un ticker del S&P 500 para obtener la predicción del modelo, "
        "indicadores técnicos y métricas fundamentales."
    )

    # --- Input ---
    col_input, col_period = st.columns([2, 1])
    with col_input:
        ticker = st.text_input(
            "Ticker", value="AAPL", max_chars=10,
        ).upper().strip()
    with col_period:
        months_back = st.selectbox(
            "Mostrar últimos",
            [3, 6, 12, 24, 60],
            index=2,
            format_func=lambda m: f"{m} meses",
        )

    if st.button("Analizar", type="primary"):
        if not ticker:
            st.warning("Introduce un ticker válido.")
            return
        _run_analysis(ticker, months_back)


# ---------------------------------------------------------------------------
# Core analysis flow
# ---------------------------------------------------------------------------

def _run_analysis(ticker: str, months_back: int):
    # Load model
    try:
        model, threshold = _load_production_model()
    except FileNotFoundError:
        st.error(
            "Modelo no encontrado. Ejecuta **`make pipeline`** para entrenar el modelo."
        )
        return

    # Download OHLCV
    with st.spinner(f"Descargando datos de {ticker}..."):
        data = download_ohlcv([ticker, "^GSPC"])

    if ticker not in data:
        st.error(f"No se pudieron descargar datos para **{ticker}**.")
        return

    market_df = data.get("^GSPC")

    # Feature engineering
    with st.spinner("Calculando features..."):
        df = build_feature_row(ticker, data[ticker], market_df=market_df)

    if df.empty:
        st.error("Datos insuficientes para calcular indicadores técnicos.")
        return

    # Prediction on latest row
    latest = df.iloc[[-1]]
    X = latest[FEATURE_COLUMNS].values
    prob = float(model.predict_proba(X)[0, 1])
    signal_info = generate_signal(prob, threshold)

    # --- Display ---
    _show_signal_card(signal_info, threshold, df, ticker)

    # Filter to display window
    cutoff = df.index.max() - pd.DateOffset(months=months_back)
    df_display = df.loc[df.index >= cutoff]

    _plot_price_chart(df_display, ticker)
    _plot_technicals(df_display)
    _show_fundamentals(df)


# ---------------------------------------------------------------------------
# Signal card
# ---------------------------------------------------------------------------

_SIGNAL_COLORS = {"BUY": "#4caf50", "HOLD": "#9e9e9e"}


def _show_signal_card(signal_info: dict, threshold: float, df: pd.DataFrame, ticker: str):
    signal = signal_info["signal"]
    prob = signal_info["probability"]
    confidence = signal_info["confidence"]
    color = _SIGNAL_COLORS.get(signal, "#9e9e9e")

    latest_close = df["Close"].iloc[-1]
    latest_sma = df["sma_200"].iloc[-1] if "sma_200" in df.columns else np.nan

    st.markdown(
        f"""
        <div style="
            background-color: {color}20;
            border-left: 6px solid {color};
            padding: 1rem 1.5rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        ">
            <h2 style="margin:0; color:{color}">{ticker} — {signal}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Probabilidad", f"{prob:.1%}")
    c2.metric("Confianza", confidence)
    c3.metric("Threshold", f"{threshold:.1%}")
    c4.metric("Cierre", f"${latest_close:,.2f}")
    c5.metric(
        "SMA 200",
        f"${latest_sma:,.2f}" if pd.notna(latest_sma) else "N/D",
    )


# ---------------------------------------------------------------------------
# Price chart
# ---------------------------------------------------------------------------

def _plot_price_chart(df: pd.DataFrame, ticker: str):
    st.subheader("Precio y SMA 200")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Precio", line=dict(color="#2196F3", width=2),
    ))
    if "sma_200" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_200"],
            name="SMA 200", line=dict(color="#FF9800", width=1.5, dash="dash"),
        ))

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Precio ($)",
        height=420,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def _plot_technicals(df: pd.DataFrame):
    st.subheader("Indicadores Técnicos")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("RSI (14)", "Williams %R (14)", "MACD Histogram"),
        vertical_spacing=0.08,
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                   line=dict(color="#2196F3")),
        row=1, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=1, col=1)

    # Williams %R
    fig.add_trace(
        go.Scatter(x=df.index, y=df["williams_r"], name="Williams %R",
                   line=dict(color="#9C27B0")),
        row=2, col=1,
    )
    fig.add_hline(y=-80, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
    fig.add_hline(y=-20, line_dash="dash", line_color="red", line_width=1, row=2, col=1)

    # MACD Histogram
    macd_vals = df["macd_histogram"]
    colors = ["#4caf50" if v >= 0 else "#f44336" for v in macd_vals]
    fig.add_trace(
        go.Bar(x=df.index, y=macd_vals, name="MACD Hist", marker_color=colors),
        row=3, col=1,
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

_FUND_LABELS = {
    "pe_ratio": ("PE Ratio", "num"),
    "peg_ratio": ("PEG Ratio", "num"),
    "op_margin": ("Margen Operativo", "pct"),
    "revenue_growth": ("Crecimiento Ingresos", "pct"),
    "debt_equity": ("Deuda / Equity", "num"),
    "current_ratio": ("Current Ratio", "num"),
    "cash_covers_debt": ("Cash / Deuda", "num"),
    "fcf_yield": ("FCF Yield", "pct"),
}


def _show_fundamentals(df: pd.DataFrame):
    st.subheader("Métricas Fundamentales")

    latest = df.iloc[-1]
    cols = st.columns(4)

    for i, (key, (label, fmt)) in enumerate(_FUND_LABELS.items()):
        val = latest.get(key)
        with cols[i % 4]:
            if pd.notna(val):
                display = f"{val:.2%}" if fmt == "pct" else f"{val:.2f}"
            else:
                display = "N/D"
            st.metric(label, display)
