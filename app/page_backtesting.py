"""
Backtesting visualization page.

Runs the historical simulation via ``core.backtesting_engine.run_backtest``
and displays:
    - Key metric cards
    - Equity curve vs S&P 500 benchmark
    - Drawdown chart
    - Full 22-metric breakdown (5 tiers)
    - Monthly returns heatmap
    - Trade log table
"""

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtesting_engine import run_backtest
from core.data_service import get_sp500_tickers

_TEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "V", "PG"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render():
    st.title("Backtesting")
    st.markdown(
        "Simulates the DeepValue AI strategy on historical data and compares "
        "against a passive S&P 500 buy-and-hold benchmark."
    )

    # --- Configuration ---
    col1, col2, col3 = st.columns(3)
    with col1:
        mode = st.radio(
            "Universe",
            ["Test (10 tickers)", "Full S&P 500"],
            horizontal=True,
        )
    with col2:
        # SimFin free tier fundamentals start ~2020-05; use 2020-06 as safe default
        start_date = st.date_input("Start date", value=pd.Timestamp("2020-06-01"))
        end_date = st.date_input("End date", value=date.today())
    with col3:
        capital = st.number_input(
            "Initial capital ($)", value=100_000, step=10_000, min_value=10_000,
        )

    if st.button("Run Backtest", type="primary"):
        tickers = _TEST_TICKERS if "Test" in mode else get_sp500_tickers()

        with st.spinner("Running backtest... this may take several minutes."):
            try:
                result = run_backtest(
                    tickers=tickers,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    initial_capital=float(capital),
                )
            except FileNotFoundError:
                st.error(
                    "Backtest model not found. "
                    "Run **`make pipeline`** to train."
                )
                return
            except RuntimeError as e:
                st.error(f"Backtest error: {e}")
                return

        st.session_state["backtest_result"] = result

    # --- Display ---
    if "backtest_result" in st.session_state:
        _display_results(st.session_state["backtest_result"])


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _display_results(result):
    metrics = result.metrics

    _show_key_metrics(metrics)
    _plot_equity_curve(result)
    _plot_drawdown(result.equity_curve)
    _show_monthly_heatmap(metrics)
    _show_all_metrics(metrics)
    _show_trade_log(result.trades)


# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------

def _show_key_metrics(m: dict):
    st.subheader("Summary")

    cards = [
        ("Total Return", m.get("total_return"), "{:.2%}"),
        ("Annualized Return", m.get("annualized_return"), "{:.2%}"),
        ("Sharpe Ratio", m.get("sharpe_ratio"), "{:.2f}"),
        ("Max Drawdown", m.get("max_drawdown"), "{:.2%}"),
        ("Win Rate", m.get("win_rate"), "{:.1%}"),
        ("Alpha vs S&P 500", m.get("alpha"), "{:.2%}"),
    ]

    cols = st.columns(len(cards))
    for col, (label, val, fmt) in zip(cols, cards, strict=True):
        display = fmt.format(val) if val is not None else "N/A"
        col.metric(label, display)


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def _plot_equity_curve(result):
    st.subheader("Equity Curve vs S&P 500")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve.values,
        name="DeepValue AI",
        line=dict(color="#2196F3", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=result.benchmark_curve.index,
        y=result.benchmark_curve.values,
        name="S&P 500 (Buy & Hold)",
        line=dict(color="#FF9800", width=2, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=480,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(t=30, b=40),
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def _plot_drawdown(equity_curve: pd.Series):
    st.subheader("Drawdown")

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#f44336", width=1),
        fillcolor="rgba(244, 67, 54, 0.25)",
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown",
        height=300,
        template="plotly_white",
        yaxis_tickformat=".1%",
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Monthly returns heatmap
# ---------------------------------------------------------------------------

def _show_monthly_heatmap(metrics: dict):
    monthly_raw = metrics.get("monthly_returns")
    if not monthly_raw:
        return

    st.subheader("Monthly Returns")

    # Build a DataFrame with year x month
    records = []
    for date_str, ret in monthly_raw.items():
        dt = pd.Timestamp(date_str)
        records.append({"year": dt.year, "month": dt.month, "return": ret})

    df = pd.DataFrame(records)
    if df.empty:
        return

    pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="first")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(pivot.columns)]

    # Color scale: red for negative, green for positive
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index],
        colorscale=[
            [0.0, "#f44336"],
            [0.5, "#ffffff"],
            [1.0, "#4caf50"],
        ],
        zmid=0,
        text=[[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Return", tickformat=".0%"),
    ))
    fig.update_layout(
        height=max(200, 60 * len(pivot)),
        template="plotly_white",
        margin=dict(t=20, b=30),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# All 22 metrics — 5 tiers
# ---------------------------------------------------------------------------

_METRIC_TIERS = {
    "Return": [
        ("total_return", "Total Return", "pct"),
        ("annualized_return", "Annualized Return", "pct"),
        ("roi_on_invested", "ROI on Invested Capital", "pct"),
        ("benchmark_return", "Benchmark Return", "pct"),
        ("alpha", "Alpha", "pct"),
    ],
    "Risk": [
        ("max_drawdown", "Max Drawdown", "pct"),
        ("max_drawdown_duration", "Max DD Duration (days)", "int"),
        ("volatility", "Annualized Volatility", "pct"),
        ("value_at_risk", "VaR 95%", "pct"),
        ("conditional_var", "CVaR 95%", "pct"),
    ],
    "Risk-Adjusted": [
        ("sharpe_ratio", "Sharpe Ratio", "dec"),
        ("sortino_ratio", "Sortino Ratio", "dec"),
        ("calmar_ratio", "Calmar Ratio", "dec"),
        ("omega_ratio", "Omega Ratio", "dec"),
        ("recovery_factor", "Recovery Factor", "dec"),
    ],
    "Trade Quality": [
        ("win_rate", "Win Rate", "pct"),
        ("profit_factor", "Profit Factor", "dec"),
        ("avg_win_vs_avg_loss", "Avg Win / Avg Loss", "dec"),
        ("num_trades", "Total Trades", "int"),
    ],
    "Consistency": [
        ("positive_months_pct", "Positive Months", "pct"),
        ("ulcer_index", "Ulcer Index", "dec"),
    ],
}


def _fmt_metric(val, kind: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if kind == "pct":
        return f"{val:.2%}"
    if kind == "int":
        return f"{val:,.0f}"
    return f"{val:.2f}"


def _show_all_metrics(metrics: dict):
    with st.expander("All metrics (22 metrics, 5 tiers)"):
        for tier_name, items in _METRIC_TIERS.items():
            st.markdown(f"**{tier_name}**")
            tier_cols = st.columns(len(items))
            for col, (key, label, kind) in zip(tier_cols, items, strict=True):
                val = metrics.get(key)
                col.metric(label, _fmt_metric(val, kind))
            st.markdown("---")


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

_TRADE_FMT = {
    "price": "${:.2f}",
    "shares": "{:.2f}",
    "value": "${:,.0f}",
    "return_pct": "{:.2%}",
}


def _show_trade_log(trades: pd.DataFrame):
    st.subheader("Trade Log")

    if trades.empty:
        st.info("No trades were executed during this period.")
        return

    # Summary
    n_buys = int((trades["action"] == "BUY").sum())
    n_sells = int(trades["action"].isin(["SELL", "SELL_PARTIAL"]).sum())
    st.caption(f"{n_buys} buys · {n_sells} sells · {len(trades)} total trades")

    # Display
    display_cols = [c for c in ["date", "ticker", "action", "price", "shares",
                                 "value", "reason", "return_pct"] if c in trades.columns]
    fmt = {k: v for k, v in _TRADE_FMT.items() if k in display_cols}

    st.dataframe(
        trades[display_cols].style.format(fmt, na_rep="—"),
        use_container_width=True,
        height=400,
    )

    # Download
    csv = trades.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download trades (CSV)",
        data=csv,
        file_name="backtest_trades.csv",
        mime="text/csv",
    )
