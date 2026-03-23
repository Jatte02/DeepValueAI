# DeepValue AI

> Intelligent asset selection system for the US stock market.
>
> **Master's Thesis (TFM)** — IMF Smart Education
>
> **Author:** Giancarlos Estévez

DeepValue AI is a Decision Support System (DSS) that combines
Fundamental Analysis, Macroeconomic Indicators, NLP Sentiment,
and Machine Learning to identify investment opportunities with
high probability of medium-term appreciation.

The system scans **~3,600 US stocks** (full SimFin universe) using
**34 features** across 5 categories and **5 years** of historical data
with strict point-in-time safety to prevent look-ahead bias.

---

## Features

- **Individual Analyzer** — Deep audit of any company: fundamental, technical, and real-time AI prediction.
- **Visual Backtesting** — Historical simulator applying the hybrid strategy (AI + Technical Filter) over 5 years.
- **S&P 500 Screener** — Mass scanner filtering companies that meet quality and probability criteria.

---

## Feature Categories (34 total)

| Category | Count | Source | Examples |
|----------|-------|--------|----------|
| Technical | 11 | Yahoo Finance | Williams %R, RSI, MACD, SMA distance, volume trend |
| Fundamental | 8 | SimFin | PE ratio, PEG, FCF yield, debt/equity, operating margin |
| VIX / Volatility | 4 | Yahoo Finance | VIX level, regime, 5-day change, SMA distance |
| Macroeconomic | 6 | FRED API | Fed rate, unemployment, GDP growth, CPI inflation |
| Sentiment / NLP | 5 | SEC EDGAR + FinBERT | Sentiment mean/std, news volume, max/min scores |

---

## Project Structure
```
DeepValueAI/
├── core/                          # Business logic (UI-independent)
│   ├── config.py                  # Paths, constants, feature mapping
│   ├── data_service.py            # Data download and feature engineering
│   ├── prediction_service.py      # Model loading and inference
│   ├── backtesting_engine.py      # Historical simulation engine
│   ├── screener_engine.py         # S&P 500 scanning engine
│   ├── fundamental_database.py    # Historical fundamentals (SimFin, PIT-safe)
│   ├── macro_database.py          # Macro data (FRED API, PIT-safe)
│   ├── news_database.py           # News corpus (SEC EDGAR 8-K filings)
│   └── sentiment_pipeline.py      # NLP sentiment scoring (FinBERT)
│
├── app/                           # Streamlit UI layer
│   ├── streamlit_app.py           # Entry point, sidebar, routing
│   ├── page_analyzer.py           # Individual analysis page
│   ├── page_backtesting.py        # Backtesting page
│   └── page_screener.py           # Screener page
│
├── ml_pipeline/                   # Training pipeline
│   ├── generate_dataset.py        # ETL: download and process historical data
│   └── train_model.py             # Train, compare, and select best model
│
├── models/                        # Trained model artifacts
├── data/                          # Generated datasets (Parquet + CSV)
└── tests/                         # Unit tests (179 tests)
```

---

## Quick Start
```bash
# Clone the repository
git clone https://github.com/Jatte02/DeepValueAI.git
cd DeepValueAI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install dependencies
make install

# Set up API keys (copy and edit .env)
cp .env.example .env
# Add your SimFin and FRED API keys to .env

# Download intermediate datasets (run once)
python -m core.fundamental_database --source simfin
python -m core.macro_database
python -m core.news_database --source edgar
python -m core.sentiment_pipeline

# Generate training data (~1-2 hours for ~3,600 tickers)
make dataset

# Train and compare models (~10-15 min)
make train

# Launch the dashboard
make run
```

---

## Available Commands
```bash
make help       # Show all commands
make run        # Launch Streamlit dashboard
make test       # Run unit tests
make lint       # Check code quality
make install    # Install dependencies
make dataset    # Generate training dataset
make train      # Train ML models
make pipeline   # Full pipeline: data -> model
```

---

## Data Pipeline

```
SimFin API ──> fundamentals_features.parquet ──┐
FRED API   ──> macro_features.parquet ─────────┤
SEC EDGAR  ──> headlines_raw.parquet           │
    └──> FinBERT ──> sentiment_scores.parquet ─┤
Yahoo Finance (OHLCV + VIX) ───────────────────┤
                                               ▼
                            generate_dataset (ETL merge)
                                               │
                                               ▼
                            training_dataset.csv (34 features)
                                               │
                                               ▼
                             train_model (3 classifiers)
                                               │
                                               ▼
                            best_model.pkl + optimal_threshold.txt
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Data | yfinance, SimFin, FRED API, SEC EDGAR |
| NLP | HuggingFace Transformers (FinBERT), PyTorch |
| Technical Analysis | pandas_ta (SMA, Williams %R, RSI, MACD) |
| Machine Learning | scikit-learn (HistGradientBoosting, RandomForest, LogReg) |
| Visualization | Streamlit, Plotly, matplotlib, seaborn |
| Storage | Parquet (pyarrow), CSV |
| Quality | pytest (179 tests), ruff |

---

## Architecture Principles

- **Model-agnostic**: Any scikit-learn compatible classifier can be trained, compared, and deployed.
- **Point-in-time safe**: Every data source uses its real publication date — no look-ahead bias.
- **Separation of concerns**: `core/` has zero UI dependencies. Streamlit pages are thin renderers.
- **Single source of truth**: Data fetching, feature engineering, and prediction each live in one place.
- **Testable**: Every engine runs without Streamlit. `make test` verifies everything (179 tests).

---

## License

Developed as part of the Master's Thesis in Data Science and Business Analytics at IMF Smart Education.
