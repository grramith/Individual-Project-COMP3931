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
    lstm_preds,
    lstm_meta,
    window=10,
    decay=0.95,
    weight_smooth_alpha=0.15,
):
    eps = 1e-6

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

        actual = ticker_df["Actual"].values
        pred_rf = ticker_df["Pred_RF"].values.copy()
        pred_gb = ticker_df["Pred_GB"].values.copy()
        pred_lstm = ticker_df["Pred_LSTM"].values.copy()

        smooth_rf, smooth_gb, smooth_lstm = 1 / 3, 1 / 3, 1 / 3

        for i in range(min(window, n)):
            if np.isnan(pred_lstm[i]):
                ensemble_pred[i] = 0.5 * pred_rf[i] + 0.5 * pred_gb[i]
            else:
                ensemble_pred[i] = (pred_rf[i] + pred_gb[i] + pred_lstm[i]) / 3.0

        for i in range(window, n):
            decay_weights = np.array([decay ** (window - 1 - j) for j in range(window)])
            decay_weights /= decay_weights.sum()

            hist_actual = actual[i - window:i]
            hist_rf = pred_rf[i - window:i]
            hist_gb = pred_gb[i - window:i]

            bias_rf = float(np.dot(decay_weights, hist_rf - hist_actual))
            bias_gb = float(np.dot(decay_weights, hist_gb - hist_actual))

            pred_rf_corrected = pred_rf[i] - bias_rf
            pred_gb_corrected = pred_gb[i] - bias_gb

            mae_rf = float(np.dot(decay_weights, np.abs(hist_rf - hist_actual)))
            mae_gb = float(np.dot(decay_weights, np.abs(hist_gb - hist_actual)))

            dir_rf = float(
                np.dot(decay_weights, ((hist_rf > 0) == (hist_actual > 0)).astype(float))
            )
            dir_gb = float(
                np.dot(decay_weights, ((hist_gb > 0) == (hist_actual > 0)).astype(float))
            )

            score_rf = 0.7 / (mae_rf + eps) + 0.3 * dir_rf
            score_gb = 0.7 / (mae_gb + eps) + 0.3 * dir_gb

            hist_lstm = pred_lstm[i - window:i]

            if np.any(np.isnan(hist_lstm)):
                total_score = score_rf + score_gb
                raw_rf = score_rf / total_score
                raw_gb = score_gb / total_score
                raw_lstm = 0.0
                pred_lstm_corrected = 0.0
            else:
                bias_lstm = float(np.dot(decay_weights, hist_lstm - hist_actual))
                pred_lstm_corrected = pred_lstm[i] - bias_lstm

                mae_lstm = float(np.dot(decay_weights, np.abs(hist_lstm - hist_actual)))
                dir_lstm = float(
                    np.dot(decay_weights, ((hist_lstm > 0) == (hist_actual > 0)).astype(float))
                )
                score_lstm = 0.7 / (mae_lstm + eps) + 0.3 * dir_lstm

                total_score = score_rf + score_gb + score_lstm
                raw_rf = score_rf / total_score
                raw_gb = score_gb / total_score
                raw_lstm = score_lstm / total_score

            alpha = weight_smooth_alpha
            smooth_rf = (1 - alpha) * smooth_rf + alpha * raw_rf
            smooth_gb = (1 - alpha) * smooth_gb + alpha * raw_gb
            smooth_lstm = (1 - alpha) * smooth_lstm + alpha * raw_lstm

            weight_sum = smooth_rf + smooth_gb + smooth_lstm
            weight_rf[i] = smooth_rf / weight_sum
            weight_gb[i] = smooth_gb / weight_sum
            weight_lstm[i] = smooth_lstm / weight_sum

            ensemble_pred[i] = (
                weight_rf[i] * pred_rf_corrected
                + weight_gb[i] * pred_gb_corrected
                + weight_lstm[i] * pred_lstm_corrected
            )

        ticker_df["Weight_RF"] = weight_rf
        ticker_df["Weight_GB"] = weight_gb
        ticker_df["Weight_LSTM"] = weight_lstm
        ticker_df["Ensemble_Delta"] = ensemble_pred
        final_results.append(ticker_df)

    return pd.concat(final_results, ignore_index=True)


def sharpe_from_signals(
    results_df,
    threshold,
    vix_low,
    vix_high,
    use_fractional,
    allow_short,
    dd_limit,
    pos_scale=1.0,
    tx_cost=0.0005,
):
    all_rets = []

    for ticker in results_df["Ticker"].unique():
        ticker_df = results_df[results_df["Ticker"] == ticker].copy().sort_values("Date")
        pred = ticker_df["Ensemble_Delta"].values
        actual = ticker_df["Actual"].values
        vix = ticker_df["VIX_Value"].values if "VIX_Value" in ticker_df.columns else np.zeros(len(ticker_df))
        n = len(ticker_df)

        position = np.zeros(n)
        equity = [1.0]
        peak = 1.0
        strat_rets = np.zeros(n)

        for i in range(1, n):
            p = pred[i - 1]
            v = vix[i - 1]

            if v > vix_high:
                eff_threshold = threshold * 3.0
            elif v > vix_low:
                eff_threshold = threshold * 1.5
            else:
                eff_threshold = threshold

            if use_fractional:
                denom = eff_threshold * 5 + 1e-9
                if p > eff_threshold:
                    position[i] = min(p / denom * pos_scale, 1.0)
                elif allow_short and p < -eff_threshold:
                    position[i] = max(p / denom * pos_scale, -1.0)
                else:
                    position[i] = 0.0
            else:
                if p > eff_threshold:
                    position[i] = 1.0
                elif allow_short and p < -eff_threshold:
                    position[i] = -1.0
                else:
                    position[i] = 0.0

            dd = (equity[-1] - peak) / peak if peak > 0 else 0.0
            if dd < -dd_limit:
                severity = min((abs(dd) - dd_limit) / dd_limit, 1.0)
                position[i] *= max(1.0 - severity, 0.0)

            pos_change = abs(position[i] - position[i - 1])
            ret = position[i] * actual[i] - pos_change * tx_cost
            strat_rets[i] = ret

            equity.append(equity[-1] * (1 + ret))
            peak = max(peak, equity[-1])

        ticker_df["Strategy_Ret"] = strat_rets
        ticker_df["Position"] = position
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
        mask = val_meta["Ticker"].values == ticker
        X_ticker = X_val[mask]
        meta_ticker = val_meta[mask].reset_index(drop=True)

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

    print(f"VIX regime thresholds: 50th={vix_50th:.1f}, 75th={vix_75th:.1f}")

    rf = joblib.load("models/baselines/RF_Regressor.pkl")
    gb = joblib.load("models/baselines/GB_Regressor.pkl")

    print("\nTuning ensemble parameters on validation set...")
    print("=" * 60)

    param_grid = [
        {
            "window": w,
            "decay": d,
            "threshold": th,
            "fractional": fr,
            "allow_short": sh,
            "dd_limit": dd,
        }
        for w in [10, 20, 30]
        for d in [0.90, 0.95, 1.00]
        for th in [0.0, 0.0005, 0.001, 0.002]
        for fr in [True, False]
        for sh in [True, False]
        for dd in [0.10, 0.15, 0.20]
    ]

    best_sharpe = -999
    best_params = None
    tuning_results = []

    for params in param_grid:
        val_ensemble = compute_ensemble_predictions(
            val_meta,
            X_val,
            y_val,
            rf,
            gb,
            lstm_val_preds,
            val_seq_meta,
            window=params["window"],
            decay=params["decay"],
        )

        sharpe, _ = sharpe_from_signals(
            val_ensemble,
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

    print("\nApplying best parameters to test set...")

    test_ensemble = compute_ensemble_predictions(
        test_meta,
        X_test,
        y_test,
        rf,
        gb,
        lstm_test_preds,
        lstm_test_meta,
        window=best_params["window"],
        decay=best_params["decay"],
    )

    test_ensemble.to_csv("data/results/hde_final_results.csv", index=False)

    best_config = {
        **best_params,
        "vix_low": vix_50th,
        "vix_high": vix_75th,
        "val_sharpe": best_sharpe,
    }

    with open("data/results/best_ensemble_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    valid = test_ensemble.dropna(subset=["Ensemble_Delta"])
    ens_mae = mean_absolute_error(valid["Actual"], valid["Ensemble_Delta"])
    ens_dir = np.mean((valid["Ensemble_Delta"] > 0) == (valid["Actual"] > 0))

    print("\nHDE v3 test metrics:")
    print(f"MAE: {ens_mae:.6f}")
    print(f"Directional accuracy: {ens_dir:.2%}")
    print("Saved results to data/results/hde_final_results.csv")

    return test_ensemble, best_config


if __name__ == "__main__":
    build_enhanced_hde()