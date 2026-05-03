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


# Phase 6 - per-ticker alpha. One-sample t-test against zero across tickers
def per_ticker_alpha(strategies, eval_dir):
    print("\n" + "=" * 78)
    print("4.3.3 — Per-ticker alpha cross-sectional test")
    print("=" * 78)
    hde = strategies["e_HDE"]["combined"]
    bh = strategies["a_BuyHold"]["combined"]

    rows = []
    for ticker in hde["Ticker"].unique():
        h = hde[hde["Ticker"] == ticker]
        b = bh[bh["Ticker"] == ticker]
        m = h[["Date", "Strategy_Ret"]].merge(
            b[["Date", "Strategy_Ret"]], on="Date", suffixes=("_hde", "_bh"))
        alpha_series = m["Strategy_Ret_hde"] - m["Strategy_Ret_bh"]
        mean_alpha_ann = alpha_series.mean() * 252
        t_stat, p_val = sp_stats.ttest_1samp(alpha_series, 0.0)
        rows.append({
            "Ticker": ticker,
            "Annualised_Alpha_%": round(mean_alpha_ann * 100, 2),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 4),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # Cross-sectional t-test - tests whether the average alpha across tickers is distinguishable from zero
    alphas = table["Annualised_Alpha_%"].values
    t_cs, p_cs = sp_stats.ttest_1samp(alphas, 0.0)
    print(f"\nCross-sectional mean alpha: {alphas.mean():.2f}%  "
          f"t={t_cs:.3f}  p={p_cs:.4f}")
    if p_cs > 0.05:
        print("  → Mean per-ticker alpha is NOT distinguishable from zero.")
        print("     The HDE's cross-sectional contribution is not statistically")
        print("     separable from a zero-alpha strategy after accounting for")
        print("     cross-sectional variance.")

    table.to_csv(f"{eval_dir}/per_ticker_alpha.csv", index=False)
    return table


# Phase 6 - sweep transaction costs from 0 to 30 bps to check the strategy's break-even point
def tx_cost_sensitivity(eval_dir):
    print("\n" + "=" * 78)
    print("4.3 — Transaction cost sensitivity (HDE)")
    print("=" * 78)
    hde_preds = preds_for_model("Pred_HDE")
    rows = []
    for bps in [0, 5, 10, 15, 20, 30]:
        res = run_strategy(f"{bps} bps", hde_preds, tx_cost=bps / 10000)
        s = res["stats"]
        rows.append({
            "TX_cost_bps": bps,
            "Total_Return_%": round(s["total_return_pct"], 1),
            "Sharpe": round(s["sharpe"], 3),
            "Max_DD_%": round(s["max_drawdown"] * 100, 1),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(f"{eval_dir}/tx_cost_sensitivity.csv", index=False)
    return table


# Phase 7 - display items for the chapter
def build_display_items(strategies, regimes, regime_regression_table, eval_dir):
    print("\n" + "=" * 78)
    print("Phase 7 — Display items")
    print("=" * 78)

    # Figure 4.2 - rolling 60-day Sharpe across the three headline strategies
    hde_port = strategies["e_HDE"]["portfolio"]
    eq_port = strategies["d_EqualWeight"]["portfolio"]
    bh_port = strategies["a_BuyHold"]["portfolio"]

    def rolling_sharpe(s, window=60):
        return s.rolling(window).mean() / s.rolling(window).std() * np.sqrt(252)

    merged = hde_port[["Date", "Strategy_Ret"]].rename(
        columns={"Strategy_Ret": "HDE"}).merge(
        eq_port[["Date", "Strategy_Ret"]].rename(
            columns={"Strategy_Ret": "EqualWt"}),
        on="Date", how="inner").merge(
        bh_port[["Date", "Strategy_Ret"]].rename(
            columns={"Strategy_Ret": "BuyHold"}),
        on="Date", how="inner")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(merged["Date"], rolling_sharpe(merged["HDE"]),
            label="Full HDE", color="#2563eb", lw=1.5)
    ax.plot(merged["Date"], rolling_sharpe(merged["EqualWt"]),
            label="Equal-weight static ensemble", color="#ef4444", lw=1.5, ls="--")
    ax.plot(merged["Date"], rolling_sharpe(merged["BuyHold"]),
            label="Buy & Hold", color="gray", lw=1.5, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Figure 4.2 — 60-day rolling Sharpe ratio comparison",
                 fontweight="bold")
    ax.set_ylabel("Rolling Sharpe (annualised)")
    ax.set_xlabel("Date")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{eval_dir}/figure_4_2_rolling_sharpe.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {eval_dir}/figure_4_2_rolling_sharpe.png")

    # Figure 4.3 - directional accuracy against whichever regime variable explained the most variance
    if regimes is not None and regime_regression_table is not None and len(regime_regression_table):
        top_var = regime_regression_table.iloc[0]["Variable"]
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(regimes[top_var], regimes["DirAcc"] * 100, s=100, color="#2563eb")

        for _, r in regimes.iterrows():
            ax.annotate(
                pd.Timestamp(r["Window_Start"]).strftime("%Y-%m"),
                (r[top_var], r["DirAcc"] * 100),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8
            )

        x = regimes[top_var].values
        y = regimes["DirAcc"].values * 100

        if len(x) >= 2:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(
                xs,
                np.polyval(coef, xs),
                ls="--",
                color="red",
                alpha=0.7,
                label=f"OLS Regression  R²={regime_regression_table.iloc[0]['R_squared']:.3f}"
            )

        ax.axhline(50, color="gray", lw=0.5, ls=":")
        ax.axhline(53, color="gray", lw=0.5, ls=":", alpha=0.5)

        ax.set_title(
            "Walk-forward Directional Accuracy versus Mean VIX",
            fontweight="bold"
        )
        ax.set_xlabel("Mean VIX")
        ax.set_ylabel("Directional Accuracy (%)")

        legend = ax.legend(
            fontsize=9,
            title="Key",
            title_fontsize=10
        )
        legend.get_title().set_fontweight("bold")

        ax.grid(True, alpha=0.3)

        # Padding so edge points and labels sit inside the grid
        ax.margins(x=0.08, y=0.08)

        plt.tight_layout()
        plt.savefig(f"{eval_dir}/figure_4_3_regime_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved → {eval_dir}/figure_4_3_regime_scatter.png")