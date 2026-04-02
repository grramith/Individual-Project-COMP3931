import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_hidden)).squeeze(-1)


def compute_ensemble_predictions(
    metadata,
    X_data,
    y_data,
    rf,
    gb,
    lstm_preds=None,
    lstm_meta=None,
    window=10,
):
    meta = metadata.copy()
    meta["Actual"] = y_data
    meta["Pred_RF"] = rf.predict(X_data)
    meta["Pred_GB"] = gb.predict(X_data)

    if lstm_preds is not None and lstm_meta is not None and len(lstm_preds) > 0:
        lstm_df = lstm_meta.copy()
        lstm_df["Date"] = pd.to_datetime(lstm_df["Date"])
        lstm_df["Pred_LSTM"] = lstm_preds
        meta["Date"] = pd.to_datetime(meta["Date"])
        meta = meta.merge(
            lstm_df[["Date", "Ticker", "Pred_LSTM"]],
            on=["Date", "Ticker"],
            how="left",
        )
    else:
        meta["Pred_LSTM"] = np.nan

    final_results = []

    for ticker in meta["Ticker"].unique():
        ticker_df = meta[meta["Ticker"] == ticker].copy().sort_values("Date")
        n = len(ticker_df)

        weight_rf = np.full(n, 1 / 3)
        weight_gb = np.full(n, 1 / 3)
        weight_lstm = np.full(n, 1 / 3)
        ensemble_pred = np.zeros(n)

        rf_preds = ticker_df["Pred_RF"].values
        gb_preds = ticker_df["Pred_GB"].values
        lstm_preds_t = ticker_df["Pred_LSTM"].values

        for i in range(n):
            if np.isnan(lstm_preds_t[i]):
                ensemble_pred[i] = 0.5 * rf_preds[i] + 0.5 * gb_preds[i]
                weight_rf[i] = 0.5
                weight_gb[i] = 0.5
                weight_lstm[i] = 0.0
            else:
                ensemble_pred[i] = (rf_preds[i] + gb_preds[i] + lstm_preds_t[i]) / 3.0

        ticker_df["Weight_RF"] = weight_rf
        ticker_df["Weight_GB"] = weight_gb
        ticker_df["Weight_LSTM"] = weight_lstm
        ticker_df["Ensemble_Delta"] = ensemble_pred
        final_results.append(ticker_df)

    return pd.concat(final_results, ignore_index=True)


def sharpe_from_signals(
    results_df,
    threshold=0.001,
    use_fractional=True,
    allow_short=True,
    tx_cost=0.0005,
):
    all_rets = []

    for ticker in results_df["Ticker"].unique():
        ticker_df = results_df[results_df["Ticker"] == ticker].copy().sort_values("Date")
        preds = ticker_df["Ensemble_Delta"].values
        actual = ticker_df["Actual"].values
        n = len(ticker_df)

        position = np.zeros(n)
        strategy_rets = np.zeros(n)

        for i in range(1, n):
            pred = preds[i - 1]

            if use_fractional:
                if pred > threshold:
                    position[i] = min(pred / (threshold * 5 + 1e-9), 1.0)
                elif allow_short and pred < -threshold:
                    position[i] = max(pred / (threshold * 5 + 1e-9), -1.0)
                else:
                    position[i] = 0.0
            else:
                if pred > threshold:
                    position[i] = 1.0
                elif allow_short and pred < -threshold:
                    position[i] = -1.0
                else:
                    position[i] = 0.0

            pos_change = abs(position[i] - position[i - 1])
            strategy_rets[i] = position[i] * actual[i] - pos_change * tx_cost

        ticker_df["Position"] = position
        ticker_df["Strategy_Ret"] = strategy_rets
        all_rets.append(ticker_df)

    combined = pd.concat(all_rets, ignore_index=True)
    portfolio_ret = combined.groupby("Date")["Strategy_Ret"].mean()

    if portfolio_ret.std() == 0:
        return 0.0, combined

    sharpe = (portfolio_ret.mean() / portfolio_ret.std()) * np.sqrt(252)
    return sharpe, combined


def build_enhanced_hde():
    X_val = np.load("data/modeling/X_val.npy")
    X_test = np.load("data/modeling/X_test.npy")
    y_val = np.load("data/modeling/y_val_returns.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")

    val_meta = pd.read_csv("data/modeling/val_metadata.csv")
    test_meta = pd.read_csv("data/modeling/test_metadata.csv")
    full_df = pd.read_csv("data/processed/master_dataset.csv")

    val_meta["Date"] = pd.to_datetime(val_meta["Date"])
    test_meta["Date"] = pd.to_datetime(test_meta["Date"])
    full_df["Date"] = pd.to_datetime(full_df["Date"])

    rf = joblib.load("models/baselines/RF_Regressor.pkl")
    gb = joblib.load("models/baselines/GB_Regressor.pkl")

    val_results = compute_ensemble_predictions(val_meta, X_val, y_val, rf, gb)
    val_sharpe, _ = sharpe_from_signals(val_results, threshold=0.001)

    test_results = compute_ensemble_predictions(test_meta, X_test, y_test, rf, gb)

    print("Validation Sharpe:", round(val_sharpe, 3))
    print("Validation ensemble rows:", len(val_results))
    print("Test ensemble rows:", len(test_results))

    return val_results, test_results


if __name__ == "__main__":
    build_enhanced_hde()