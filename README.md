# DeepValue AI

> Intelligent asset selection system for the S&P 500 market.
>
> **Master's Thesis (TFM)** — IMF Smart Education
>
> **Author:** Giancarlos Estévez

DeepValue AI is a Decision Support System (DSS) that combines
Fundamental Analysis with Machine Learning to identify investment
opportunities with high probability of medium-term appreciation.

The system is **model-agnostic**: it can train, compare, and deploy
any scikit-learn compatible classifier (XGBoost, Random Forest,
LightGBM, Gradient Boosting, etc.) and automatically selects the
best performer.

---

## Features

- **Individual Analyzer** — Deep audit of any company: fundamental, technical, and real-time AI prediction.
- **Visual Backtesting** — Historical simulator applying the hybrid strategy (AI + Technical Filter) over 5 years.
- **S&P 500 Screener** — Mass scanner filtering companies that meet quality and probability criteria.

---

## Project Structure
```
DeepValueAI/
├── core/                      # Business logic (UI-independent)
│   ├── config.py              # Paths, constants, feature mapping
│   ├── data_service.py        # Data download and feature engineering
│   ├── prediction_service.py  # Model loading and inference
│   ├── backtesting_engine.py  # Historical simulation engine
│   └── screener_engine.py     # S&P 500 scanning engine
│
├── app/                       # Streamlit UI layer
│   ├── streamlit_app.py       # Entry point, sidebar, routing
│   └── pages/
│       ├── analyzer.py        # Individual analysis page
│       ├── backtesting.py     # Backtesting page
│       └── screener.py        # Screener page
│
├── ml_pipeline/               # Training pipeline
│   ├── generate_dataset.py    # ETL: download and process historical data
│   └── train_model.py         # Train, compare, and select best model
│
├── models/                    # Trained model artifacts
├── data/                      # Generated datasets
└── tests/                     # Unit tests
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

# Generate training data (~30 min for 500 companies)
make dataset

# Train and compare models (~2 min)
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
make pipeline   # Full pipeline: data → model
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Data | yfinance, pandas, numpy |
| Technical Analysis | pandas_ta (SMA, Williams %R) |
| Machine Learning | scikit-learn, XGBoost, LightGBM, Random Forest |
| Visualization | Streamlit, matplotlib, seaborn |
| Quality | pytest, ruff |

---

## Architecture Principles

- **Model-agnostic**: Any scikit-learn compatible classifier can be trained, compared, and deployed.
- **Separation of concerns**: `core/` has zero UI dependencies. Streamlit pages are thin renderers.
- **Single source of truth**: Data fetching, feature engineering, and prediction each live in one place.
- **Testable**: Every engine runs without Streamlit. `make test` verifies everything.

---

## License

Developed as part of the Master's Thesis in Data Science and Business Analytics at IMF Smart Education.