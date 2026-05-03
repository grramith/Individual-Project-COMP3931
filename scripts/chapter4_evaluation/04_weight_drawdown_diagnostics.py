# Chapter 4 §4.5 - dynamic weighting diagnosis (Figure 4.1) and Table 4.3 drawdown decomposition
# Pulled out of the eval notebook so main.py can call this without re-running the whole cell

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.statistical_tests import sharpe_difference_test
from utils.bootstrap import select_block_length, block_bootstrap
from utils.strategies import run_strategy, preds_for_model


# Figure 4.1 - weight trajectories and deviation statistics
def weight_diagnosis(strategies, eval_dir):
    print("\n" + "=" * 78)
    print("§4.5 — DYNAMIC WEIGHTING MECHANISM DIAGNOSIS")
    print("=" * 78)

    hde = pd.read_csv("data/results/hde_final_results.csv", parse_dates=["Date"])
    w_cols = ["Weight_RF", "Weight_GB", "Weight_LSTM"]

    # Aggregate to one row per date - average across tickers gives the portfolio-level weight story
    daily = hde.groupby("Date")[w_cols].mean().sort_index()

    # Summary statistics on the daily portfolio-level weight series
    print("\nTime-series statistics of daily portfolio-level weights:")
    stats = pd.DataFrame({
        "mean": daily.mean(),
        "std":  daily.std(),
        "min":  daily.min(),
        "max":  daily.max(),
    })
    stats["max_abs_dev_from_uniform"] = (daily - 1/3).abs().max()
    stats["frac_within_±0.05"] = ((daily - 1/3).abs() < 0.05).mean()
    print(stats.round(4).to_string())

    # Convergence metric - how often all three weights sit inside a tight band around 1/3
    all_within = ((daily - 1/3).abs() < 0.05).all(axis=1).mean()
    print(f"\nFraction of days ALL THREE weights within ±0.05 of 1/3:  {all_within:.2%}")
    print(f"Max abs deviation of any weight on any day from 1/3:     "
          f"{(daily - 1/3).abs().values.max():.4f}")

    # Figure 4.1 - three weight series with the uniform reference and ±0.05 band
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index, daily["Weight_RF"], label="Random Forest", color="#3b82f6", lw=1.5)
    ax.plot(daily.index, daily["Weight_GB"], label="Gradient Boosting", color="#ef4444", lw=1.5)
    ax.plot(daily.index, daily["Weight_LSTM"], label="LSTM", color="#10b981", lw=1.5)
    ax.axhline(1/3, color="black", ls="--", alpha=0.5, label="Uniform (1/3)")
    ax.fill_between(daily.index, 1/3 - 0.05, 1/3 + 0.05,
                    color="gray", alpha=0.15, label="±0.05 band")
    ax.set_title("Ensemble weight trajectories across the test period",
                 fontweight="bold")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.set_ylim(0.2, 0.45)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{eval_dir}/figure_4_1_weight_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {eval_dir}/figure_4_1_weight_trajectories.png")

    stats.to_csv(f"{eval_dir}/weight_diagnostics.csv")

    # Critical test - HDE vs equal-weight static. If JKM p > 0.05 the dynamic mechanism is empirically inert
    print("\nCritical test — does the dynamic weighting add anything?")
    hde_rets = strategies["e_HDE"]["daily_returns"]
    eq_rets = strategies["d_EqualWeight"]["daily_returns"]
    hde_port = strategies["e_HDE"]["portfolio"][["Date", "Strategy_Ret"]].rename(
        columns={"Strategy_Ret": "hde"})
    eq_port = strategies["d_EqualWeight"]["portfolio"][["Date", "Strategy_Ret"]].rename(
        columns={"Strategy_Ret": "eq"})
    merged = hde_port.merge(eq_port, on="Date", how="inner")

    test = sharpe_difference_test(merged["hde"].values, merged["eq"].values)
    print(f"  HDE Sharpe:        {test['sr1']:+.3f}")
    print(f"  Equal-wt Sharpe:   {test['sr2']:+.3f}")
    print(f"  Δ Sharpe:          {test['diff']:+.3f}")
    print(f"  JKM z-statistic:   {test['z']:+.3f}")
    print(f"  p-value:           {test['p_value']:.4f}")
    if test["p_value"] > 0.05:
        print("  → HDE and equal-weight ensemble are STATISTICALLY INDISTINGUISHABLE")
        print("     The dynamic weighting mechanism is empirically inert.")
    elif test["diff"] > 0:
        print("  → HDE significantly outperforms equal-weight ensemble")
    else:
        print("  → Equal-weight ensemble significantly outperforms HDE")
        print("     The dynamic weighting is actively harmful.")

    # Bootstrap the daily-return difference directly as a robustness check on JKM
    diff_rets = merged["hde"].values - merged["eq"].values
    bl = select_block_length(diff_rets)
    mean_diff, (lo, hi), _ = block_bootstrap(diff_rets, np.mean,
                                              n_boot=5000, block_len=bl)
    print(f"\n  Bootstrap mean daily return difference (HDE − EqualWt):")
    print(f"  point={mean_diff*1e4:+.2f} bps  CI=[{lo*1e4:+.2f}, {hi*1e4:+.2f}] bps")

    return daily, stats, test


# Phase 4 - 2x2x2 factorial ablation of the three defensive components
def drawdown_decomposition(eval_dir):
    print("\n" + "=" * 78)
    print("TABLE 4.3 — Drawdown Decomposition (full 2×2×2 factorial)")
    print("=" * 78)
    print("All eight combinations of the three defensive components on HDE predictions.")
    print("Base configuration uses the tuned HDE parameters from §4.3.\n")

    hde_preds = preds_for_model("Pred_HDE")
    rows = []
    for use_thresh in [False, True]:
        for use_vix in [False, True]:
            for use_tap in [False, True]:
                res = run_strategy(
                    f"T={int(use_thresh)}/V={int(use_vix)}/D={int(use_tap)}",
                    hde_preds,
                    use_threshold=use_thresh,
                    use_vix_filter=use_vix,
                    use_taper=use_tap,
                )
                s = res["stats"]
                rows.append({
                    "Threshold": use_thresh,
                    "VIX_filter": use_vix,
                    "Taper": use_tap,
                    "Sharpe": round(s["sharpe"], 3),
                    "Max_DD_%": round(s["max_drawdown"] * 100, 2),
                    "Total_Return_%": round(s["total_return_pct"], 1),
                    "Exposure_%": round(s["avg_exposure"] * 100, 1),
                })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # Marginal effect of each component, averaged across the other two - main-effects attribution
    print("\nMarginal contribution to drawdown improvement")
    print("(averaged across all configurations of the other two components):")

    def marginal_effect(component):
        # positive value = enabling the component reduces drawdown (off mean is more negative than on mean)
        on = table[table[component] == True]["Max_DD_%"].mean()
        off = table[table[component] == False]["Max_DD_%"].mean()
        return off - on

    for comp in ["Threshold", "VIX_filter", "Taper"]:
        effect = marginal_effect(comp)
        print(f"  {comp:<12}  ΔMax_DD when enabled: {effect:+.2f} pp  "
              f"({'reduces' if effect > 0 else 'worsens'} drawdown)")

    # Headline number - total drawdown reduction from running all three vs none
    full = table[(table["Threshold"]) & (table["VIX_filter"]) & (table["Taper"])]
    none = table[(~table["Threshold"]) & (~table["VIX_filter"]) & (~table["Taper"])]
    if len(full) and len(none):
        total_improvement = none.iloc[0]["Max_DD_%"] - full.iloc[0]["Max_DD_%"]
        print(f"\n  Total drawdown reduction (all-on vs all-off): {total_improvement:+.2f} pp")

    table.to_csv(f"{eval_dir}/table_4_3_drawdown_decomposition.csv", index=False)
    return table