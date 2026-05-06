import numpy as np
import pandas as pd
import joblib
import json
import torch
import torch.nn as nn


# local copy so we can load the saved weights
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :])).squeeze(-1)


# same as script 07 but with mae_weight exposed
def compute_ensemble_predictions(
    metadata, X_data, y_data, rf, gb,
    lstm_preds, lstm_meta,
    window=10, decay=0.95, weight_smooth_alpha=0.15,
    mae_weight=0.7
):
    dir_weight = 1.0 - mae_weight
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
        # no LSTM, fall back to RF+GB
        meta["Pred_LSTM"] = np.nan

    final_results = []

    # weights adapt per ticker
    for ticker in meta["Ticker"].unique():
        t_df = meta[meta["Ticker"] == ticker].copy().sort_values("Date")
        n = len(t_df)

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

        # main loop
        for t in range(window, n):

            # exponential decay weights
            decay_w = np.array([decay ** (window - 1 - i) for i in range(window)])
            decay_w /= decay_w.sum()

            hist_act = act[t - window:t]
            hist_rf  = p_rf[t - window:t]
            hist_gb  = p_gb[t - window:t]

            # bias correct
            bias_rf = float(np.dot(decay_w, hist_rf - hist_act))
            bias_gb = float(np.dot(decay_w, hist_gb - hist_act))
            pred_rf_corrected = p_rf[t] - bias_rf
            pred_gb_corrected = p_gb[t] - bias_gb

            # score = error + direction
            ew_mae_rf = float(np.dot(decay_w, np.abs(hist_rf - hist_act)))
            ew_mae_gb = float(np.dot(decay_w, np.abs(hist_gb - hist_act)))
            dir_rf    = float(np.dot(decay_w, ((hist_rf > 0) == (hist_act > 0)).astype(float)))
            dir_gb    = float(np.dot(decay_w, ((hist_gb > 0) == (hist_act > 0)).astype(float)))

            # sweep happens here
            score_rf = mae_weight / (ew_mae_rf + eps) + dir_weight * dir_rf
            score_gb = mae_weight / (ew_mae_gb + eps) + dir_weight * dir_gb

            # LSTM may still be missing
            hist_lstm = p_lstm[t - window:t]
            if np.any(np.isnan(hist_lstm)):
                total    = score_rf + score_gb
                raw_rf   = score_rf / total
                raw_gb   = score_gb / total
                raw_lstm = 0.0
                pred_lstm_corrected = 0.0

            # else score all three
            else:
                bias_lstm = float(np.dot(decay_w, hist_lstm - hist_act))
                pred_lstm_corrected = p_lstm[t] - bias_lstm
                ew_mae_lstm = float(np.dot(decay_w, np.abs(hist_lstm - hist_act)))
                dir_lstm    = float(np.dot(decay_w, ((hist_lstm > 0) == (hist_act > 0)).astype(float)))
                score_lstm  = mae_weight / (ew_mae_lstm + eps) + dir_weight * dir_lstm
                total    = score_rf + score_gb + score_lstm
                raw_rf   = score_rf / total
                raw_gb   = score_gb / total
                raw_lstm = score_lstm / total

            # EMA smooth
            a = weight_smooth_alpha
            sw_rf   = (1 - a) * sw_rf   + a * raw_rf
            sw_gb   = (1 - a) * sw_gb   + a * raw_gb
            sw_lstm = (1 - a) * sw_lstm + a * raw_lstm
            sw_sum  = sw_rf + sw_gb + sw_lstm

            w_rf[t]   = sw_rf / sw_sum
            w_gb[t]   = sw_gb / sw_sum
            w_lstm[t] = sw_lstm / sw_sum

            # blend
            ens_delta[t] = (w_rf[t] * pred_rf_corrected +
                            w_gb[t] * pred_gb_corrected +
                            w_lstm[t] * pred_lstm_corrected)

        # store back
        t_df["Ensemble_Delta"] = ens_delta
        final_results.append(t_df)

    return pd.concat(final_results)


# slim backtest, returns Sharpe only
def sharpe_from_signals(results_df, threshold, vix_low, vix_high,
                        use_fractional, allow_short, dd_limit,
                        pos_scale=1, tx_cost=0.0005):
    all_rets = []

    # one ticker at a time
    for ticker in results_df["Ticker"].unique():
        t_df = results_df[results_df["Ticker"] == ticker].copy().sort_values("Date")
        pred   = t_df["Ensemble_Delta"].values
        actual = t_df["Actual"].values
        vix    = t_df["VIX_Value"].values if "VIX_Value" in t_df.columns else np.zeros(len(t_df))
        n = len(t_df)

        position   = np.zeros(n)
        equity     = [1.0]
        peak       = 1.0
        strat_rets = np.zeros(n)

        # one-day signal lag
        for i in range(1, n):
            p = pred[i - 1]
            v = vix[i - 1]

            # widen threshold when VIX is high
            if v > vix_high:
                eff_threshold = threshold * 3.0
            elif v > vix_low:
                eff_threshold = threshold * 1.5
            else:
                eff_threshold = threshold

            # size the position
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
            dd = (equity[-1] - peak) / peak if peak > 0 else 0
            if dd < -dd_limit:
                severity = min((abs(dd) - dd_limit) / dd_limit, 1.0)
                position[i] *= max(1.0 - severity, 0.0)

            # tx cost on position change
            pos_change     = abs(position[i] - position[i - 1])
            ret            = position[i] * actual[i] - pos_change * tx_cost
            strat_rets[i]  = ret

            # update equity
            equity.append(equity[-1] * (1 + ret))
            peak = max(peak, equity[-1])

        # save
        t_df["Strategy_Ret"] = strat_rets
        all_rets.append(t_df)

    # daily portfolio mean
    combined      = pd.concat(all_rets)
    portfolio_ret = combined.groupby("Date")["Strategy_Ret"].mean()

    # avoid /0
    if portfolio_ret.std() == 0:
        return 0.0, combined
    sharpe = (portfolio_ret.mean() / portfolio_ret.std()) * np.sqrt(252)
    return sharpe, combined


# entry point
def main():

    # load val arrays
    X_val    = np.load("data/modeling/X_val.npy")
    y_val    = np.load("data/modeling/y_val_returns.npy")
    val_meta = pd.read_csv("data/modeling/val_metadata.csv")
    full_df  = pd.read_csv("data/processed/master_dataset.csv")
    val_meta["Date"] = pd.to_datetime(val_meta["Date"])

    # load RF and GB
    rf = joblib.load("models/baselines/RF_Regressor.pkl")
    gb = joblib.load("models/baselines/GB_Regressor.pkl")

    # re-run LSTM on val
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
        mask      = val_meta["Ticker"].values == ticker
        X_tick    = X_val[mask]
        meta_tick = val_meta[mask].reset_index(drop=True)
        for i in range(sl, len(X_tick)):
            val_seqs.append(X_tick[i - sl:i])
            val_seq_meta.append({"Date": meta_tick.iloc[i]["Date"], "Ticker": ticker})
    val_seqs     = np.array(val_seqs)
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
    val_meta = val_meta.merge(vix_data, on=["Date", "Ticker"], how="left")

    # VIX thresholds from train data only
    train_vix = full_df[pd.to_datetime(full_df["Date"]) < "2023-01-01"][vix_col].dropna()
    vix_50th  = float(train_vix.quantile(0.50))
    vix_75th  = float(train_vix.quantile(0.75))

    # hold everything else at the tuned config
    best_params = {"window": 10, "decay": 0.9, "threshold": 0.0,
                   "fractional": True, "allow_short": False, "dd_limit": 0.15}

    # Test 1: blend ratio
    print("TEST 1: MAE/DirAcc blend ratio sensitivity")
    for mae_w in [0.6, 0.7, 0.8]:
        dir_w = round(1.0 - mae_w, 1)
        val_ens = compute_ensemble_predictions(
            val_meta, X_val, y_val, rf, gb,
            lstm_val_preds, val_seq_meta,
            window=best_params["window"],
            decay=best_params["decay"],
            weight_smooth_alpha=0.15,
            mae_weight=mae_w
        )
        sharpe, _ = sharpe_from_signals(
            val_ens,
            threshold=best_params["threshold"],
            vix_low=vix_50th, vix_high=vix_75th,
            use_fractional=best_params["fractional"],
            allow_short=best_params["allow_short"],
            dd_limit=best_params["dd_limit"]
        )
        print(f"  Blend {mae_w:.1f}/{dir_w:.1f}  -->  Val Sharpe = {sharpe:.3f}")

    # Test 2: smoothing alpha
    print()
    print("TEST 2: EMA smoothing alpha sensitivity")
    for alpha in [0.05, 0.15, 0.30]:
        val_ens = compute_ensemble_predictions(
            val_meta, X_val, y_val, rf, gb,
            lstm_val_preds, val_seq_meta,
            window=best_params["window"],
            decay=best_params["decay"],
            weight_smooth_alpha=alpha,
            mae_weight=0.7
        )
        sharpe, _ = sharpe_from_signals(
            val_ens,
            threshold=best_params["threshold"],
            vix_low=vix_50th, vix_high=vix_75th,
            use_fractional=best_params["fractional"],
            allow_short=best_params["allow_short"],
            dd_limit=best_params["dd_limit"]
        )
        print(f"  alpha = {alpha:.2f}  -->  Val Sharpe = {sharpe:.3f}")

    print()



if __name__ == "__main__":
    main()