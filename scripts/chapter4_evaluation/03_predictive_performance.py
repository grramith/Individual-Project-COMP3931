def preds_for_model(col_name):
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value", col_name]].copy()
    df.rename(columns={col_name: "Prediction"}, inplace=True)
    return df.dropna(subset=["Prediction"])

def build_buy_and_hold():
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value"]].copy()
    df["Prediction"] = 1.0
    return df

def build_momentum_12_1():
    master = pd.read_csv("data/processed/master_dataset.csv", parse_dates=["Date"])
    ret_col = "Return_1d" if "Return_1d" in master.columns else "Target_Return"
    frames = []
    for ticker in PREDS["Ticker"].unique():
        h = master[master["Ticker"] == ticker].sort_values("Date").copy()
        h["ret12"] = (1 + h[ret_col]).rolling(252).apply(np.prod, raw=True) - 1
        h["ret1"] = (1 + h[ret_col]).rolling(21).apply(np.prod, raw=True) - 1
        h["mom_12_1"] = h["ret12"] - h["ret1"]
        frames.append(h[["Date", "Ticker", "mom_12_1"]])
    mom = pd.concat(frames, ignore_index=True)
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value"]].merge(
        mom, on=["Date", "Ticker"], how="left")
    df.rename(columns={"mom_12_1": "Prediction"}, inplace=True)
    df["Prediction"] = df["Prediction"] / 252
    return df.dropna(subset=["Prediction"])

def build_equal_weight_ensemble():
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value",
                "Pred_RF", "Pred_GB", "Pred_LSTM"]].copy()
    df["Prediction"] = df[["Pred_RF", "Pred_GB", "Pred_LSTM"]].mean(axis=1)
    return df.dropna(subset=["Prediction"])