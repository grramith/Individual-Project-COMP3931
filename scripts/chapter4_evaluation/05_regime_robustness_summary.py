# Chapter 4 - regime regression, robustness checks, display items, summary report
# Phases 5-7 + final summary, lifted out of the eval notebook

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from utils.strategies import run_strategy, preds_for_model


# Phase 5 - generate walk-forward results inline if not already cached
def generate_walk_forward_if_missing():
    # GB stand-in for the full HDE pipeline - retraining HDE in every window would take hours
    # and the GB constituent's regime stability is the proxy reported in 4.4 of the chapter
    wf_path = "data/results/rolling_window_evaluation.csv"
    if os.path.exists(wf_path):
        print(f"  Walk-forward CSV already exists at {wf_path}")
        return pd.read_csv(wf_path, parse_dates=["Window_Start"])

    print("  Walk-forward CSV not found — generating now (≈2-3 min)")
    print("  Approach: GradientBoostingRegressor retrained on each expanding window")
    print("            (matches the methodology of the existing pipeline)")

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error

    df = pd.read_csv("data/processed/master_dataset.csv", parse_dates=["Date"])
    target_col = "Target_Return"
    drop_cols = ["Date", "Ticker", "Adj_Close", "Target_Direction",
                 "Target_Return", "Return_1d"]
    features = [c for c in df.columns if c not in drop_cols]

    # Eight semi-annual evaluation windows starting Jan 2021
    eval_start_dates = pd.to_datetime([
        "2021-01-01", "2021-07-01",
        "2022-01-01", "2022-07-01",
        "2023-01-01", "2023-07-01",
        "2024-01-01", "2024-07-01",
    ])
    WINDOW_SIZE_DAYS = int(126 * 1.5)

    rows = []
    for start in eval_start_dates:
        end = start + pd.Timedelta(days=WINDOW_SIZE_DAYS)
        train = df[df["Date"] < start]
        test = df[(df["Date"] >= start) & (df["Date"] < end)]
        if len(test) == 0 or len(train) == 0:
            continue

        # Refit scaler per window so it only sees data available at training time
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[features].values)
        Xte = scaler.transform(test[features].values)
        ytr = train[target_col].values
        yte = test[target_col].values

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)

        mae = mean_absolute_error(yte, preds)
        dir_acc = float(np.mean((preds > 0) == (yte > 0)))
        rows.append({
            "Window_Start": start,
            "Train_Size": len(train),
            "Test_Size": len(test),
            "MAE": mae,
            "Dir_Accuracy": dir_acc,
        })
        print(f"    {start.date()}: train={len(train):>6}  test={len(test):>5}  "
              f"MAE={mae:.6f}  DirAcc={dir_acc:.2%}")

    wf = pd.DataFrame(rows)
    wf.to_csv(wf_path, index=False)
    print(f"\n  Saved → {wf_path}")
    print(f"  Mean DirAcc: {wf['Dir_Accuracy'].mean():.2%}  "
          f"(σ = {wf['Dir_Accuracy'].std():.2%})")
    print(f"  Mean MAE:    {wf['MAE'].mean():.6f}  "
          f"(σ = {wf['MAE'].std():.6f})")
    return wf


def regime_regression(eval_dir):
    print("\n" + "=" * 78)
    print("4.4 — WALK-FORWARD + REGIME REGRESSION")
    print("=" * 78)

    wf = generate_walk_forward_if_missing()
    if wf is None or len(wf) == 0:
        print("  [error] walk-forward generation failed")
        return None, None

    master = pd.read_csv("data/processed/master_dataset.csv", parse_dates=["Date"])
    ret_col = "Return_1d" if "Return_1d" in master.columns else "Target_Return"
    vix_col = [c for c in master.columns if "vix" in c.lower()][0]

    # Term spread is optional - skip cleanly if the dataset doesn't carry a yield-curve column
    term_col = None
    for cand in ["Term_Spread", "Yield_Spread", "T10Y3M", "t10y3m_spread"]:
        if cand in master.columns:
            term_col = cand
            break

    regime_rows = []
    WINDOW_DAYS = int(126 * 1.5)  # matches Script 09
    for _, r in wf.iterrows():
        start = pd.Timestamp(r["Window_Start"])
        end = start + pd.Timedelta(days=WINDOW_DAYS)
        win = master[(master["Date"] >= start) & (master["Date"] < end)]
        if len(win) == 0:
            continue

        daily = win.groupby("Date").agg({
            ret_col: "mean",
            vix_col: "mean",
        }).reset_index()

        # Per-window regime descriptors used as candidate predictors of accuracy
        realised_vol = daily[ret_col].std() * np.sqrt(252)
        mean_vix = daily[vix_col].mean()
        cum_return = (1 + daily[ret_col]).prod() - 1

        # Average pairwise correlation across the seven tickers - proxy for cross-sectional dispersion
        wide = win.pivot_table(index="Date", columns="Ticker", values=ret_col)
        corr_mat = wide.corr()
        avg_corr = (corr_mat.values[np.triu_indices_from(corr_mat.values, k=1)]).mean()

        row = {
            "Window_Start": r["Window_Start"],
            "DirAcc": r["Dir_Accuracy"],
            "MAE": r["MAE"],
            "Mean_VIX": mean_vix,
            "Realised_Vol": realised_vol,
            "Avg_Pair_Corr": avg_corr,
            "Cum_Return": cum_return,
        }
        if term_col:
            row["Term_Spread"] = win[term_col].mean()
        regime_rows.append(row)

    regimes = pd.DataFrame(regime_rows)
    print("\nPer-window regime features:")
    print(regimes.round(4).to_string(index=False))

    # Univariate OLS - n=8 means these are exploratory associations, not confirmatory tests
    print("\nUnivariate OLS: DirAcc ~ regime_var   (n = {})".format(len(regimes)))
    candidate_vars = ["Mean_VIX", "Realised_Vol", "Avg_Pair_Corr", "Cum_Return"]
    if term_col:
        candidate_vars.append("Term_Spread")

    uni_rows = []
    for v in candidate_vars:
        x = regimes[v].values
        y = regimes["DirAcc"].values
        if len(x) < 3 or np.std(x) == 0:
            continue
        slope, intercept, r, p, se = sp_stats.linregress(x, y)
        uni_rows.append({
            "Variable": v,
            "Coefficient": round(slope, 6),
            "Std_Error": round(se, 6),
            "R_squared": round(r ** 2, 4),
            "p_value": round(p, 4),
        })
    uni_table = pd.DataFrame(uni_rows).sort_values("R_squared", ascending=False)
    print(uni_table.to_string(index=False))

    print("\nCaveat: n = {} windows. These p-values are exploratory and should".format(len(regimes)))
    print("not be interpreted as confirmatory evidence. Report as motivation for")
    print("a regime-conditional architecture (Future Work 4.7).")

    regimes.to_csv(f"{eval_dir}/regime_features.csv", index=False)
    uni_table.to_csv(f"{eval_dir}/regime_regression.csv", index=False)
    return regimes, uni_table