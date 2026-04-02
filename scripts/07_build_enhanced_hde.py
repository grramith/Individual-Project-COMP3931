import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error


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
    vix_low=20.0,
    vix_high=30.0,
    use_fractional=True,
    allow_short=True,
    dd_limit=0.15,
    pos_scale=1.0,
    tx_cost=0.0005,
):
    all_rets = []

    for ticker in results_df["Ticker"].unique():
        ticker_df = results_df[results_df["Ticker"] == ticker].copy().sort_values("Date")
        preds = ticker_df["Ensemble_Delta"].values
        actual = ticker_df["Actual"].values
        vix = ticker_df["VIX_Value"].values if "VIX_Value" in ticker_df.columns else np.zeros(len(ticker_df))
        n = len(ticker_df)

        position = np.zeros(n)
        equity = [1.0]
        peak = 1.0
        strategy_rets = np.zeros(n)

        for i in range(1, n):
            pred = preds[i - 1]
            vix_value = vix[i - 1]

            if vix_value > vix_high:
                eff_threshold = threshold * 3.0
            elif vix_value > vix_low:
                eff_threshold = threshold * 1.5
            else:
                eff_threshold = threshold

            if use_fractional:
                denom = eff_threshold * 5 + 1e-9
                if pred > eff_threshold:
                    position[i] = min(pred / denom * pos_scale, 1.0)
                elif allow_short and pred < -eff_threshold:
                    position[i] = max(pred / denom * pos_scale, -1.0)
                else:
                    position[i] = 0.0
            else:
                if pred > eff_threshold:
                    position[i] = 1.0
                elif allow_short and pred < -eff_threshold:
                    position[i] = -1.0
                else:
                    position[i] = 0.0

            dd = (equity[-1] - peak) / peak if peak > 0 else 0.0
            if dd < -dd_limit:
                severity = min((abs(dd) - dd_limit) / dd_limit, 1.0)
                position[i] *= max(1.0 - severity, 0.0)

            pos_change = abs(position[i] - position[i - 1])
            strategy_rets[i] = position[i] * actual[i] - pos_change * tx_cost
            equity.append(equity[-1] * (1 + strategy_rets[i]))
            peak = max(peak, equity[-1])

        ticker_df["Position"] = position
        ticker_df["Strategy_Ret"] = strategy_rets
        all_rets.append(ticker_df)

    combined = pd.concat(all_rets, ignore_index=True)
    portfolio_ret = combined.groupby("Date")["Strategy_Ret"].mean()

    if portfolio_ret.std() == 0:
        return 0.0, combined

    sharpe = (portfolio_ret.mean() / portfolio_ret.std()) * np.sqrt(252)
    return sharpe, combined


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_lstm_val_predictions(X_val, y_val, val_meta):
    with open("models/lstm/best_config.json", "r") as f:
        lstm_cfg = json.load(f)

    device = get_device()

    model = LSTMRegressor(
        input_size=X_val.shape[1],
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=2,
        dropout=lstm_cfg["dropout"],
    ).to(device)

    model.load_state_dict(torch.load("models/lstm/best_lstm.pth", map_location=device))
    model.eval()

    seq_len = lstm_cfg["seq_len"]
    val_sequences = []
    val_seq_meta = []

    for ticker in val_meta["Ticker"].unique():
        ticker_mask = val_meta["Ticker"].values == ticker
        X_ticker = X_val[ticker_mask]
        meta_ticker = val_meta[ticker_mask].reset_index(drop=True)

        for i in range(seq_len, len(X_ticker)):
            val_sequences.append(X_ticker[i - seq_len:i])
            val_seq_meta.append(
                {
                    "Date": meta_ticker.iloc[i]["Date"],
                    "Ticker": ticker,
                }
            )

    val_sequences = np.array(val_sequences)
    val_seq_meta = pd.DataFrame(val_seq_meta)

    with torch.no_grad():
        preds = model(torch.FloatTensor(val_sequences).to(device)).cpu().numpy()

    return preds, val_seq_meta


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

    lstm_test_df = pd.read_csv("data/results/lstm_predictions.csv")
    lstm_test_df["Date"] = pd.to_datetime(lstm_test_df["Date"])
    lstm_test_preds = lstm_test_df["Pred_LSTM"].values
    lstm_test_meta = lstm_test_df[["Date", "Ticker"]].copy()

    lstm_val_preds, val_seq_meta = generate_lstm_val_predictions(X_val, y_val, val_meta)

    vix_col = [col for col in full_df.columns if "vix" in col.lower()][0]
    vix_data = full_df[["Date", "Ticker", vix_col]].copy()
    vix_data = vix_data.rename(columns={vix_col: "VIX_Value"})

    val_meta = val_meta.merge(vix_data, on=["Date", "Ticker"], how="left")
    test_meta = test_meta.merge(vix_data, on=["Date", "Ticker"], how="left")

    train_vix = full_df[full_df["Date"] < "2023-01-01"][vix_col].dropna()
    vix_50th = float(train_vix.quantile(0.50))
    vix_75th = float(train_vix.quantile(0.75))

    rf = joblib.load("models/baselines/RF_Regressor.pkl")
    gb = joblib.load("models/baselines/GB_Regressor.pkl")

    print("Tuning ensemble parameters on validation set...")
    param_grid = [
        {
            "window": w,
            "threshold": th,
            "fractional": fr,
            "allow_short": sh,
            "dd_limit": dd,
        }
        for w in [10, 20, 30]
        for th in [0.0, 0.0005, 0.001, 0.002]
        for fr in [True, False]
        for sh in [True, False]
        for dd in [0.10, 0.15, 0.20]
    ]

    best_sharpe = -999
    best_params = None
    tuning_results = []

    for params in param_grid:
        val_results = compute_ensemble_predictions(
            val_meta,
            X_val,
            y_val,
            rf,
            gb,
            lstm_val_preds,
            val_seq_meta,
            window=params["window"],
        )

        sharpe, _ = sharpe_from_signals(
            val_results,
            threshold=params["threshold"],
            vix_low=vix_50th,
            vix_high=vix_75th,
            use_fractional=params["fractional"],
            allow_short=params["allow_short"],
            dd_limit=params["dd_limit"],
        )

        tuning_results.append({**params, "val_sharpe": sharpe})

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params

    print("Configurations tested:", len(param_grid))
    print("Best validation Sharpe:", round(best_sharpe, 3))
    print("Best parameters:", best_params)

    os.makedirs("data/results", exist_ok=True)
    pd.DataFrame(tuning_results).to_csv("data/results/ensemble_tuning_log.csv", index=False)

    test_results = compute_ensemble_predictions(
        test_meta,
        X_test,
        y_test,
        rf,
        gb,
        lstm_test_preds,
        lstm_test_meta,
        window=best_params["window"],
    )

    test_results.to_csv("data/results/hde_final_results.csv", index=False)

    valid = test_results.dropna(subset=["Ensemble_Delta"])
    ens_mae = mean_absolute_error(valid["Actual"], valid["Ensemble_Delta"])
    ens_dir = np.mean((valid["Ensemble_Delta"] > 0) == (valid["Actual"] > 0))

    print("Test MAE:", round(ens_mae, 6))
    print("Test directional accuracy:", f"{ens_dir:.2%}")

    return test_results, best_params


if __name__ == "__main__":
    build_enhanced_hde()