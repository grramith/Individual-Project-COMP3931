# Hybrid Dynamic Ensemble for Magnificent Seven Return Forecasting

**COMP3931 Individual Project | University of Leeds**

This project builds a machine learning pipeline to forecast next-day stock returns for the Magnificent Seven. It combines linear models, tree-based ensembles, and an LSTM under an adaptive weighting framework. A backtested trading strategy and full statistical evaluation are included. The results are reported honestly, including where things did not work.

---

## Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running the Pipeline](#running-the-pipeline)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Known Limitations](#known-limitations)

---

## Overview

Together, the Magnificent Seven (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA) make up around 33.7% of the S&P 500. Their short-term returns are hard to forecast. Daily price movements are noisy, and the relationships between predictors and returns can change quickly. This project tests whether a dynamically weighted ensemble of different model types can improve prediction accuracy and backtested trading performance.

The pipeline is fully reproducible from the raw data stage. It downloads raw price and macroeconomic data, builds the feature set, and splits the data in strict chronological order to avoid temporal leakage. Four classical regressors and an LSTM are then trained on the prepared data. The Hybrid Dynamic Ensemble combines the Random Forest, Gradient Boosting, and LSTM predictions using a rolling performance-based weighting scheme. The backtest adds a one-day signal lag, VIX-based threshold adjustment, fractional position sizing, drawdown control, and five basis-point transaction costs.

The main finding is that adaptive weighting did not provide a reliable improvement over simpler baselines. The model weights stayed close to a uniform allocation across the test period, suggesting that recent prediction errors were too noisy to identify the strongest model consistently. The HDE produced positive trading performance and reduced drawdown, but it did not outperform Buy-and-Hold on total return or Sharpe ratio. This result is still useful, as it highlights the difficulty of finding stable daily return signals in highly liquid mega-cap equities.

---



## Architecture

```
+-----------------------------------------------------------------+
|                         Data Pipeline                           |
|       Adjusted price data (2015-2025) + FRED macro data         |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                      Feature Engineering                        |
|     Returns | Momentum | MA Ratios | MACD | RSI | Volatility    |
|         SPY return | Macroeconomic variables (5)                |
+-----------------------------------------------------------------+
                                |
                                v
+------------+------------+---------------+-----------------------+
|   Linear   |   Ridge    | Random Forest |   Gradient Boosting   |
| Regression | Regression |    (tuned)    |        (tuned)        |
+------------+------------+---------------+-----------------------+
|            LSTM (2-layer, per-ticker sequences)                 |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                Hybrid Dynamic Ensemble (HDE)                    |
|   Rolling performance scoring | Exponential weight smoothing    |
|       Bias correction | Adaptive model weighting                |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|                      Backtesting Engine                         |
|    One-day signal lag | VIX-based threshold adjustment          |
|    Fractional position sizing | Drawdown tapering               |
|             5 bps transaction costs                             |
+-----------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------+
|              Evaluation and Statistical Testing                 |
|   Predictive metrics | Backtest metrics | PT test | DM test     |
|   Block bootstrap confidence intervals | Regime analysis        |
+-----------------------------------------------------------------+
```

**Key design decisions:**

- **Temporal integrity:** The train/val/test split is strictly chronological. Training runs through 2022, validation is 2023, and the test set is 2024-2025. The StandardScaler is fitted on training data only.
- **Per-ticker LSTM sequences:** Sequences are built separately for each ticker so rolling windows don't leak across stocks.
- **Warm-up handling:** During the LSTM warm-up window, the ensemble runs on RF and GB only. LSTM is added once it has enough history.
- **Bias correction:** Before combining models, the ensemble subtracts each model's recent weighted prediction bias. This cuts down on systematic over- or under-shooting.
- **Reproducibility:** Random seed 42 is set globally. All artifacts are deterministic across runs on the same hardware.

---

## Tech Stack

| Purpose | Library | Version |
|---------|---------|---------|
| Equity data | yfinance | 0.2.66 |
| Macro data | fredapi | 0.5.2 |
| Env vars | python-dotenv | 1.0.1 |
| Data wrangling | pandas, numpy | 2.3.3, 1.26.4 |
| Classical ML | scikit-learn | 1.5.2 |
| Model serialisation | joblib | >= 1.3.0 |
| Deep learning | PyTorch | 2.4.1 |
| Statistical tests | scipy | 1.13.1 |
| Plots | matplotlib | 3.9.2 |
| Tests | pytest | 8.3.3 |
| Notebooks | jupyter, ipykernel | >= 1.0.0, >= 6.25.0 |

**Python:** 3.11+

The experiments for this project were run on a Mac with an Apple M2 chip and 16 GB of memory, running macOS 26.3.1. The LSTM training loop used MPS acceleration where available, with CPU fallback where needed.

---

## Setup

### Prerequisites

- Python 3.11+
- pip
- A FRED API key, free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html). This is only needed for script 02, which pulls macro data.

### Install

```bash
# Clone the repository
git clone <repo-url>
cd <repo-directory>

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root and add your FRED API key:

```env
FRED_API_KEY=your_fred_api_key_here
```

The key is only needed for `scripts/02_feature_engineering.py`. This script downloads the macroeconomic variables used in the project, including the Federal Funds Rate, 10-year Treasury yield, VIX, CPI, and unemployment rate.

If data/raw/macro_fred.csv already exists from a previous run, you can skip script 02 and continue from script 03.

---

## Running the pipeline

Run the full pipeline:

```bash
python main.py
```

This runs scripts 01 to 07 in order. Each script checks its inputs first, and the pipeline stops if any step fails.

Run the Chapter 4 evaluation only:

```bash
python run_evaluation.py
```

This requires `data/results/hde_final_results.csv` and produces the tables, figures, and summary files used in Chapter 4.

### Running scripts individually

Each stage can also be run on its own:

```bash
python scripts/01_data_collection.py
python scripts/02_feature_engineering.py
python scripts/03_build_master_dataset.py
python scripts/04_regression_data_preprocessing.py
python scripts/05_train_baseline_regressors.py
python scripts/06_train_lstm_regressor.py
python scripts/07_build_enhanced_hde.py
```

Run them in order, as each script depends on outputs from the previous stage.

---

## Results summary

Outputs are saved in `data/results/`. The `evaluation/` folder contains the Chapter 4 tables and figures.

### Predictive performance, test set 2024

| Model | MAE | Directional Accuracy |
|-------|-----|----------------------|
| Linear Regression | 0.0173 | 50.2% |
| Ridge Regression | 0.0174 | 50.1% |
| Random Forest | **0.0167** | 52.1% |
| Gradient Boosting | 0.0183 | 51.0% |
| LSTM | 0.0181 | **53.9%** |
| HDE | 0.0179 | 51.5% |

### Strategy performance, test set 2024

| Strategy | Total Return | Sharpe | Max Drawdown |
|----------|--------------|--------|--------------|
| HDE | 32.4% | 1.09 | -18.1% |
| Equal-weight ensemble | 9.4% | 0.36 | n/a |
| Buy-and-hold | **110.9%** | **1.51** | -29.0% |

Best validation configuration: `window=10`, `decay=0.9`, `threshold=0.0005`, `VIX_low=16.6`, `VIX_high=22.55`.

## Project Structure

```
.
├── config.py                         # Shared paths, tickers, dates, and random seed
├── main.py                           # Runs the main pipeline from scripts 01 to 07
├── run_evaluation.py                 # Runs the Chapter 4 evaluation scripts
├── README.md                         # Project overview and setup guide
├── requirements.txt                  # Python package requirements
├── .env                              # Local only. Stores FRED_API_KEY.
├── data/
│   ├── raw/                          # Raw price and macro data
│   │   ├── prices.csv
│   │   └── macro_fred.csv
│   ├── processed/                    # Cleaned feature dataset
│   │   └── master_dataset.csv
│   ├── modeling/                     # Arrays and metadata used for training
│   │   ├── feature_names.csv
│   │   ├── scaler.pkl
│   │   ├── train_metadata.csv
│   │   ├── val_metadata.csv
│   │   ├── test_metadata.csv
│   │   ├── X_train.npy
│   │   ├── X_val.npy
│   │   ├── X_test.npy
│   │   ├── y_train_returns.npy
│   │   ├── y_val_returns.npy
│   │   └── y_test_returns.npy
│   └── results/
│       ├── hde_final_results.csv     # Final HDE predictions and weights
│       ├── backtest_summary.json     # Main strategy performance summary
│       ├── best_ensemble_config.json # Best validation-tuned HDE settings
│       ├── baseline_regression_results.csv
│       ├── ensemble_tuning_log.csv
│       ├── hyperparameter_tuning_log.csv
│       ├── lstm_predictions.csv
│       ├── lstm_tuning_log.csv
│       ├── per_stock_metrics.csv
│       ├── portfolio_backtest.csv
│       ├── rolling_window_evaluation.csv
│       └── evaluation/               # Chapter 4 tables, figures, and summaries
├── models/
│   ├── baselines/                    # Saved baseline regression models
│   │   ├── Linear_Regression.pkl
│   │   ├── Ridge_Regression.pkl
│   │   ├── RF_Regressor.pkl
│   │   └── GB_Regressor.pkl
│   └── lstm/                         # Saved LSTM model and selected config
│       ├── best_lstm.pth
│       └── best_config.json
├── scripts/
│   ├── 01_data_collection.py         # Downloads adjusted price data
│   ├── 02_feature_engineering.py     # Fetches FRED data and macro variables
│   ├── 03_build_master_dataset.py    # Builds the final feature dataset
│   ├── 04_regression_data_preprocessing.py # Splits and scales the data
│   ├── 05_train_baseline_regressors.py     # Trains Linear, Ridge, RF, and GB
│   ├── 06_train_lstm_regressor.py           # Trains and tunes the LSTM model
│   ├── 07_build_enhanced_hde.py             # Builds and evaluates the HDE
│   ├── 07.1_sensitivity.py                  # Optional sensitivity analysis
│   └── chapter4_evaluation/
│       ├── 01_shared_infrastructure.py      # Shared backtest and helper code
│       ├── 02_inferential_toolbox.py        # Bootstrap, DM, PT, and Sharpe tests
│       ├── 03_predictive_performance.py     # Predictive performance tables
│       ├── 04_weight_drawdown_diagnostics.py
│       ├── 05_regime_robustness_summary.py
│       └── 06_enhanced_backtest.py
└── tests/
    ├── test_backtest.py              # Backtest and trading-rule checks
    ├── test_ensemble.py              # Ensemble weighting and bias correction tests
    ├── test_features.py              # Technical indicator tests
    ├── test_integration.py           # End-to-end output checks
    ├── test_metrics.py               # Evaluation metric tests
    ├── test_models.py                # Model shape and prediction checks
    └── test_preprocessing.py         # Split, scaling, and missing-value tests
```

---

## Testing

Run the full test suite:

```bash
pytest tests/
```

Run with detailed output:

```bash
pytest tests/ -v
```

Stop after the first failure:

```bash
pytest tests/ -x
```

Run a single test file:

```bash
pytest tests/test_ensemble.py -v
```

The test suite covers the main parts of the project: model setup, ensemble weighting, feature calculations, preprocessing, performance metrics, backtesting, and end-to-end output checks.

The tests use small synthetic examples rather than the full dataset. They run quickly and do not require downloaded data or trained models.
---

## Known Limitations

**Dynamic weighting barely moves.** The main takeaway from the project is that the adaptive weighting mechanism settles near uniform weights (roughly 1/3 each for RF, GB, and LSTM) and stays there. The "dynamic" part adds little over a plain equal-weight ensemble on this dataset.

**LSTM warm-up gap.** For the first `seq_len` trading days of the test period, LSTM predictions do not exist. The ensemble falls back to RF and GB during that window, which creates a small inconsistency in the earliest test-set predictions.

**FRED API key needed for macro data.** Script 02 will not run without a valid FRED API key in `.env`. If you have an existing `macro_fred.csv`, drop it into `data/raw/` and start from script 03.

**Models and raw data are not in the repo.** The repository ships with result CSVs and evaluation outputs. The trained `.pkl` and `.pth` files and the raw downloaded data are all gitignored. A clean run from scratch needs internet access for yfinance and FRED, and takes roughly 20 to 40 minutes. The LSTM tuning step is the slowest part.

**The test window is one bull-market year.** The backtest only covers 2024, which was an unusually strong year for the Magnificent Seven. Buy-and-hold benefits a lot from that. Testing across a longer window with different regimes would likely give a fairer comparison.

**No live trading.** This is a research pipeline. There is no order management, slippage modelling, or live data feed.
