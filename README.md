# Hybrid Dynamic Ensemble for Magnificent Seven Return Forecasting

## Overview

This repository contains the implementation for a final-year COMP3931 Individual Project at the University of Leeds. The project investigates whether a Hybrid Dynamic Ensemble (HDE) can improve next-day return forecasting for the Magnificent Seven equities by combining linear models, tree-based ensembles, and an LSTM model under an adaptive weighting framework.

The system provides an end-to-end forecasting and evaluation pipeline covering data acquisition, feature engineering, temporal preprocessing, model training, dynamic ensemble construction, backtesting, statistical testing, and Chapter 4 result generation.

## Project Context

Daily equity return prediction is difficult because the signal-to-noise ratio is low, predictor relationships change across market regimes, and statistical accuracy does not necessarily translate into profitable trading after transaction costs. This project addresses that problem by evaluating whether dynamic model weighting can improve predictive and portfolio performance relative to simpler baselines.

The empirical setting is the Magnificent Seven equity universe:

- Apple
- Microsoft
- Alphabet
- Amazon
- Nvidia
- Meta
- Tesla

SPY is used as both a market benchmark and a market return feature.

The project evaluates three main success criteria:

1. Whether the HDE improves predictive accuracy relative to linear baselines.
2. Whether directional accuracy exceeds naive and practical baselines.
3. Whether the trading strategy improves risk-adjusted performance relative to buy-and-hold.

## Repository Structure

```text
.
├── config.py
├── main.py
├── run_evaluation.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   ├── prices.csv
│   │   └── macro_fred.csv
│   ├── processed/
│   │   └── master_dataset.csv
│   ├── modeling/
│   │   ├── X_train.npy
│   │   ├── X_val.npy
│   │   ├── X_test.npy
│   │   ├── y_train_returns.npy
│   │   ├── y_val_returns.npy
│   │   ├── y_test_returns.npy
│   │   ├── train_metadata.csv
│   │   ├── val_metadata.csv
│   │   ├── test_metadata.csv
│   │   ├── feature_names.csv
│   │   └── scaler.pkl
│   └── results/
│       ├── baseline_regression_results.csv
│       ├── hyperparameter_tuning_log.csv
│       ├── lstm_predictions.csv
│       ├── lstm_tuning_log.csv
│       ├── hde_final_results.csv
│       ├── ensemble_tuning_log.csv
│       ├── best_ensemble_config.json
│       ├── portfolio_backtest.csv
│       ├── backtest_summary.json
│       ├── per_stock_metrics.csv
│       ├── rolling_window_evaluation.csv
│       └── evaluation/
│           ├── all_test_predictions.csv
│           ├── chapter_4_summary.json
│           ├── table_4_1_predictive_performance.csv
│           ├── table_4_1_dm_matrix.csv
│           ├── table_4_2_baseline_ladder.csv
│           ├── table_4_2_sharpe_tests.csv
│           ├── table_4_3_drawdown_decomposition.csv
│           ├── weight_diagnostics.csv
│           ├── tx_cost_sensitivity.csv
│           ├── regime_features.csv
│           ├── regime_regression.csv
│           ├── per_ticker_alpha.csv
│           ├── figure_4_1_weight_trajectories.png
│           ├── figure_4_2_rolling_sharpe.png
│           └── figure_4_3_regime_scatter.png
├── models/
│   ├── baselines/
│   │   ├── Linear_Regression.pkl
│   │   ├── Ridge_Regression.pkl
│   │   ├── RF_Regressor.pkl
│   │   └── GB_Regressor.pkl
│   └── lstm/
│       ├── best_lstm.pth
│       └── best_config.json
├── notebooks/
│   ├── final_mag7.ipynb
│   └── chapter_4_evaluation.ipynb
├── scripts/
│   ├── 01_data_collection.py
│   ├── 02_feature_engineering.py
│   ├── 03_build_master_dataset.py
│   ├── 04_regression_data_preprocessing.py
│   ├── 05_train_baseline_regressors.py
│   ├── 06_train_lstm_regressor.py
│   ├── 07_build_enhanced_hde.py
│   ├── 07.1_sensitivity.py
│   └── chapter4_evaluation/
│       ├── 01_shared_infrastructure.py
│       ├── 02_inferential_toolbox.py
│       ├── 03_predictive_performance.py
│       ├── 04_weight_drawdown_diagnostics.py
│       ├── 05_regime_robustness_summary.py
│       └── 06_enhanced_backtest.py
└── tests/
    ├── test_backtest.py
    ├── test_ensemble.py
    ├── test_features.py
    ├── test_integration.py
    ├── test_metrics.py
    ├── test_models.py
    └── test_preprocessing.py