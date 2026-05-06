import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import time

import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)


# HDE: rolling weighted blend of RF, GB, LSTM
def compute_ensemble_predictions(
    metadata, X_data, y_data, rf, gb,
    lstm_preds, lstm_meta,
    window=10, decay=0.95, weight_smooth_alpha=0.15
):

    eps = 1e-6

    # attach actuals + RF/GB preds
    meta = metadata.copy()
    meta["Actual"]  = y_data
    meta["Pred_RF"] = rf.predict(X_data)
    meta["Pred_GB"] = gb.predict(X_data)

    # join LSTM preds on Date+Ticker
    if lstm_preds is not None and lstm_meta is not None and len(lstm_preds) > 0:
        lstm_df = lstm_meta.copy()
        lstm_df["Pred_LSTM"] = lstm_preds
        lstm_df["Date"] = pd.to_datetime(lstm_df["Date"])
        meta["Date"]    = pd.to_datetime(meta["Date"])
        meta = meta.merge(lstm_df[["Date", "Ticker", "Pred_LSTM"]],
                          on=["Date", "Ticker"], how="left")
    else:
        # no LSTM, fill NaN
        meta["Pred_LSTM"] = np.nan

    final_results = []

    # weights adapt per ticker
    for ticker in meta["Ticker"].unique():
        t_df = meta[meta["Ticker"] == ticker].copy().sort_values("Date")
        n    = len(t_df)

        # weight history
        w_rf   = np.full(n, 1/3)
        w_gb   = np.full(n, 1/3)
        w_lstm = np.full(n, 1/3)

        # ensemble output
        ens_delta = np.zeros(n)

        act    = t_df["Actual"].values
        p_rf   = t_df["Pred_RF"].values.copy()
        p_gb   = t_df["Pred_GB"].values.copy()
        p_lstm = t_df["Pred_LSTM"].values.copy()

        # init smoothed weights
        sw_rf, sw_gb, sw_lstm = 1/3, 1/3, 1/3

        # equal-weight during warm-up
        for t in range(min(window, n)):
            if np.isnan(p_lstm[t]):
                ens_delta[t] = 0.5 * p_rf[t] + 0.5 * p_gb[t]
            else:
                ens_delta[t] = (p_rf[t] + p_gb[t] + p_lstm[t]) / 3.0

        for t in range(window, n):
            
            # exponential decay weights
            decay_w = np.array([decay ** (window - 1 - i) for i in range(window)])
            decay_w /= decay_w.sum()

            hist_act = act[t - window : t]
            hist_rf  = p_rf[t - window : t]
            hist_gb  = p_gb[t - window : t]

            # bias correct
            bias_rf = float(np.dot(decay_w, hist_rf - hist_act))
            bias_gb = float(np.dot(decay_w, hist_gb - hist_act))
            pred_rf_corrected = p_rf[t] - bias_rf
            pred_gb_corrected = p_gb[t] - bias_gb

            # score = error + direction
            ew_mae_rf  = float(np.dot(decay_w, np.abs(hist_rf - hist_act)))
            ew_mae_gb  = float(np.dot(decay_w, np.abs(hist_gb - hist_act)))
            dir_rf     = float(np.dot(decay_w, ((hist_rf > 0) == (hist_act > 0)).astype(float)))
            dir_gb     = float(np.dot(decay_w, ((hist_gb > 0) == (hist_act > 0)).astype(float)))
            score_rf   = 0.7 / (ew_mae_rf + eps) + 0.3 * dir_rf
            score_gb   = 0.7 / (ew_mae_gb + eps) + 0.3 * dir_gb

            # LSTM may still be missing
            hist_lstm = p_lstm[t - window : t]
            if np.any(np.isnan(hist_lstm)):
                total      = score_rf + score_gb
                raw_rf     = score_rf / total
                raw_gb     = score_gb / total
                raw_lstm   = 0.0
                pred_lstm_corrected = 0.0
            
            # else score all three
            else:
                bias_lstm   = float(np.dot(decay_w, hist_lstm - hist_act))
                pred_lstm_corrected = p_lstm[t] - bias_lstm
                ew_mae_lstm = float(np.dot(decay_w, np.abs(hist_lstm - hist_act)))
                dir_lstm    = float(np.dot(decay_w, ((hist_lstm > 0) == (hist_act > 0)).astype(float)))
                score_lstm  = 0.7 / (ew_mae_lstm + eps) + 0.3 * dir_lstm
                total       = score_rf + score_gb + score_lstm
                raw_rf      = score_rf  / total
                raw_gb      = score_gb  / total
                raw_lstm    = score_lstm / total

            # EMA smooth the weights
            a = weight_smooth_alpha
            sw_rf   = (1 - a) * sw_rf   + a * raw_rf
            sw_gb   = (1 - a) * sw_gb   + a * raw_gb
            sw_lstm = (1 - a) * sw_lstm + a * raw_lstm
            sw_sum  = sw_rf + sw_gb + sw_lstm

            w_rf[t]   = sw_rf   / sw_sum
            w_gb[t]   = sw_gb   / sw_sum
            w_lstm[t] = sw_lstm / sw_sum

            # blend
            ens_delta[t] = (w_rf[t]   * pred_rf_corrected +
                            w_gb[t]   * pred_gb_corrected +
                            w_lstm[t] * pred_lstm_corrected)

        # store back
        t_df["Weight_RF"]     = w_rf
        t_df["Weight_GB"]     = w_gb
        t_df["Weight_LSTM"]   = w_lstm
        t_df["Ensemble_Delta"] = ens_delta
        final_results.append(t_df)

    return pd.concat(final_results)


# backtest signals -> Sharpe
def sharpe_from_signals(results_df, threshold, vix_low, vix_high,
                        use_fractional, allow_short, dd_limit,
                        use_vix_filter=True, use_taper=True,
                        pos_scale=1, tx_cost=0.0005):
    all_rets = []

    # one ticker at a time
    for ticker in results_df["Ticker"].unique():
        t_df  = results_df[results_df["Ticker"] == ticker].copy().sort_values("Date")
        pred   = t_df["Ensemble_Delta"].values
        actual = t_df["Actual"].values
        vix    = t_df["VIX_Value"].values if "VIX_Value" in t_df.columns else np.zeros(len(t_df))
        n      = len(t_df)

        position   = np.zeros(n)
        equity     = [1.0]
        peak       = 1.0
        strat_rets = np.zeros(n)

        # one-day signal lag
        for i in range(1, n):
            p   = pred[i - 1]
            v   = vix[i - 1]

            # widen threshold in high-VIX regimes
            if use_vix_filter:
                if v > vix_high:
                    eff_threshold = threshold * 3.0
                elif v > vix_low:
                    eff_threshold = threshold * 1.5
                else:
                    eff_threshold = threshold
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

            # taper on drawdown
            if use_taper:
                dd = (equity[-1] - peak) / peak if peak > 0 else 0
                if dd < -dd_limit:
                    severity = min((abs(dd) - dd_limit) / dd_limit, 1.0)
                    position[i] *= max(1.0 - severity, 0.0)

            # tx cost on position change
            pos_change  = abs(position[i] - position[i - 1])
            ret         = position[i] * actual[i] - pos_change * tx_cost
            strat_rets[i] = ret

            # update equity
            equity.append(equity[-1] * (1 + ret))
            peak = max(peak, equity[-1])

        # save
        t_df["Strategy_Ret"] = strat_rets
        t_df["Position"]     = position
        all_rets.append(t_df)

    combined      = pd.concat(all_rets)
    portfolio_ret = combined.groupby("Date")["Strategy_Ret"].mean()

    # avoid /0
    if portfolio_ret.std() == 0:
        return 0.0, combined
    sharpe = (portfolio_ret.mean() / portfolio_ret.std()) * np.sqrt(252)
    return sharpe, combined


# tune + evaluate HDE
def build_enhanced_hde():
    # load arrays
    X_val   = np.load("data/modeling/X_val.npy")
    X_test  = np.load("data/modeling/X_test.npy")
    y_val   = np.load("data/modeling/y_val_returns.npy")
    y_test  = np.load("data/modeling/y_test_returns.npy")

    # metadata + master
    val_meta  = pd.read_csv("data/modeling/val_metadata.csv")
    test_meta = pd.read_csv("data/modeling/test_metadata.csv")
    full_df   = pd.read_csv("data/processed/master_dataset.csv")

    val_meta["Date"]  = pd.to_datetime(val_meta["Date"])
    test_meta["Date"] = pd.to_datetime(test_meta["Date"])

    # LSTM test preds from script 06
    lstm_test_df    = pd.read_csv("data/results/lstm_predictions.csv")
    lstm_test_preds = lstm_test_df["Pred_LSTM"].values
    lstm_test_meta  = lstm_test_df[["Date", "Ticker"]].copy()

    # re-run LSTM on val for tuning

    with open("models/lstm/best_config.json") as f:
        lstm_cfg = json.load(f)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # reload LSTM
    lstm_model = LSTMRegressor(
        X_val.shape[1], lstm_cfg["hidden_size"], 2, lstm_cfg["dropout"]
    ).to(device)
    lstm_model.load_state_dict(
        torch.load("models/lstm/best_lstm.pth", map_location=device)
    )
    lstm_model.eval()

    # val sequences
    sl = lstm_cfg["seq_len"]
    val_seqs, val_seq_meta = [], []
    for ticker in val_meta["Ticker"].unique():
        mask       = val_meta["Ticker"].values == ticker
        X_tick     = X_val[mask]
        y_tick     = y_val[mask]
        meta_tick  = val_meta[mask].reset_index(drop=True)
        for i in range(sl, len(X_tick)):
            val_seqs.append(X_tick[i - sl : i])
            val_seq_meta.append({"Date": meta_tick.iloc[i]["Date"], "Ticker": ticker})
    val_seqs    = np.array(val_seqs)
    val_seq_meta = pd.DataFrame(val_seq_meta)

    # val inference
    with torch.no_grad():
        lstm_val_preds = lstm_model(
            torch.FloatTensor(val_seqs).to(device)
        ).cpu().numpy()

    # pull VIX into metadata
    vix_col  = [c for c in full_df.columns if "vix" in c.lower()][0]
    vix_data = full_df[["Date", "Ticker", vix_col]].copy()
    vix_data.rename(columns={vix_col: "VIX_Value"}, inplace=True)
    vix_data["Date"] = pd.to_datetime(vix_data["Date"])

    val_meta  = val_meta.merge(vix_data, on=["Date", "Ticker"], how="left")
    test_meta = test_meta.merge(vix_data, on=["Date", "Ticker"], how="left")

    # VIX thresholds from train data only
    train_vix = full_df[pd.to_datetime(full_df["Date"]) < "2023-01-01"][vix_col].dropna()
    vix_50th  = float(train_vix.quantile(0.50))   # low to elevated
    vix_75th  = float(train_vix.quantile(0.75))   # elevated to high
    print(f"VIX regime thresholds (training data):  50th={vix_50th:.1f}  75th={vix_75th:.1f}")

    # load RF and GB
    rf = joblib.load("models/baselines/RF_Regressor.pkl")
    gb = joblib.load("models/baselines/GB_Regressor.pkl")

    # tune on val
    print("\nTuning ensemble parameters on validation set (2023):")

    # grid search
    param_grid = [
        {"window": w, "decay": d, "threshold": th,
         "fractional": fr, "allow_short": sh,
         "dd_limit": dd}
        for w  in [10, 20, 30]
        for d  in [0.90, 0.95, 1.00]
        for th in [0.0005, 0.001, 0.002]
        for fr in [True, False]
        for sh in [True, False]
        for dd in [0.10, 0.15, 0.20]
    ]

    best_sharpe   = -999
    best_params   = None
    tuning_results = []

    start_grid = time.time()
    # run each config
    for params in param_grid:
        val_ens = compute_ensemble_predictions(
            val_meta, X_val, y_val, rf, gb,
            lstm_val_preds, val_seq_meta,
            window=params["window"], decay=params["decay"]
        )
        sharpe, _ = sharpe_from_signals(
            val_ens,
            threshold    = params["threshold"],
            vix_low      = vix_50th,
            vix_high     = vix_75th,
            use_fractional = params["fractional"],
            allow_short  = params["allow_short"],
            dd_limit     = params["dd_limit"],
        )
        tuning_results.append({**params, "val_sharpe": sharpe})
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params

    # report best
    print(f"Grid search completed in {(time.time() - start_grid) / 60:.1f} minutes")
    print(f"Configurations tested:   {len(param_grid)}")
    print(f"Best Validation Sharpe:  {best_sharpe:.3f}")
    print(f"Best Parameters:         {best_params}")

    os.makedirs("data/results", exist_ok=True)
    pd.DataFrame(tuning_results).to_csv("data/results/ensemble_tuning_log.csv", index=False)

    # apply to test
    print(f"\nApplying best parameters to test set (2024+)...")

    test_ensemble = compute_ensemble_predictions(
        test_meta, X_test, y_test, rf, gb,
        lstm_test_preds, lstm_test_meta,
        window=best_params["window"], decay=best_params["decay"]
    )
    test_ensemble.to_csv("data/results/hde_final_results.csv", index=False)

    # save config
    best_config = {
        **best_params,
        "vix_low":     vix_50th,
        "vix_high":    vix_75th,
        "val_sharpe":  best_sharpe,
    }
    with open("data/results/best_ensemble_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # final metrics
    valid    = test_ensemble.dropna(subset=["Ensemble_Delta"])
    ens_mae  = mean_absolute_error(valid["Actual"], valid["Ensemble_Delta"])
    ens_dir  = np.mean((valid["Ensemble_Delta"] > 0) == (valid["Actual"] > 0))
    print(f"\nHDE Test Metrics:")
    print(f"  MAE:                  {ens_mae:.6f}")
    print(f"  Directional Accuracy: {ens_dir:.2%}")
    print(f"  Results saved to data/results/hde_final_results.csv")

    return test_ensemble, best_config


if __name__ == "__main__":
    build_enhanced_hde()
