import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.metrics import mean_absolute_error


def compute_ensemble_predictions(metadata, X_data, y_data, rf, gb, lstm_preds, lstm_meta):
    meta = metadata.copy()
    meta["Actual"] = y_data
    meta["Pred_RF"] = rf.predict(X_data)
    meta["Pred_GB"] = gb.predict(X_data)

    if lstm_preds is not None and lstm_meta is not None and len(lstm_preds) > 0:
        lstm_df = lstm_meta.copy()
        lstm_df["Pred_LSTM"] = lstm_preds
        lstm_df["Date"] = pd.to_datetime(lstm_df["Date"])
        meta["Date"] = pd.to_datetime(meta["Date"])
        meta = meta.merge(
            lstm_df[["Date", "Ticker", "Pred_LSTM"]],
            on=["Date", "Ticker"],
            how="left"
        )
    else:
        meta["Pred_LSTM"] = np.nan

    final_results = []

    for ticker in meta["Ticker"].unique():
        t_df = meta[meta["Ticker"] == ticker].copy().sort_values("Date")
        p_rf = t_df["Pred_RF"].values
        p_gb = t_df["Pred_GB"].values
        p_lstm = t_df["Pred_LSTM"].values

        ensemble = np.zeros(len(t_df))

        for i in range(len(t_df)):
            if np.isnan(p_lstm[i]):
                ensemble[i] = 0.5 * p_rf[i] + 0.5 * p_gb[i]
            else:
                ensemble[i] = (p_rf[i] + p_gb[i] + p_lstm[i]) / 3.0

        t_df["Ensemble_Delta"] = ensemble
        final_results.append(t_df)

    return pd.concat(final_results)


def build_enhanced_hde():
    X_test = np.load("data/modeling/X_test.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")
    test_meta = pd.read_csv("data/modeling/test_metadata.csv")

    rf = joblib.load("models/baselines/RF_Regressor.pkl")
    gb = joblib.load("models/baselines/GB_Regressor.pkl")

    lstm_test_df = pd.read_csv("data/results/lstm_predictions.csv")
    lstm_test_preds = lstm_test_df["Pred_LSTM"].values
    lstm_test_meta = lstm_test_df[["Date", "Ticker"]].copy()

    test_ensemble = compute_ensemble_predictions(
        test_meta, X_test, y_test, rf, gb, lstm_test_preds, lstm_test_meta
    )

    valid = test_ensemble.dropna(subset=["Ensemble_Delta"])
    ens_mae = mean_absolute_error(valid["Actual"], valid["Ensemble_Delta"])

    print(f"Ensemble MAE: {ens_mae:.6f}")
    return test_ensemble


if __name__ == "__main__":
    build_enhanced_hde()