# Builds the weight and drawdown diagnostics for Chapter 4.

import matplotlib.pyplot as plt


def weight_diagnosis():
    # Summarise whether adaptive weights actually move away from equal weighting.
    print("\n" + "=" * 78)
    print("§4.5 — DYNAMIC WEIGHTING MECHANISM DIAGNOSIS")
    print("=" * 78)

    hde = pd.read_csv("data/results/hde_final_results.csv", parse_dates=["Date"])
    w_cols = ["Weight_RF", "Weight_GB", "Weight_LSTM"]

    # Average across tickers to describe the portfolio-level weight path.
    daily = hde.groupby("Date")[w_cols].mean().sort_index()

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

    # This checks how close the dynamic weights stay to a simple 1/3 split.
    all_within = ((daily - 1/3).abs() < 0.05).all(axis=1).mean()
    print(f"\nFraction of days ALL THREE weights within ±0.05 of 1/3:  {all_within:.2%}")
    print(f"Max abs deviation of any weight on any day from 1/3:     "
          f"{(daily - 1/3).abs().values.max():.4f}")

    # Save the weight path used in the dynamic-weighting discussion.
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
    plt.savefig(f"{EVAL_DIR}/figure_4_1_weight_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {EVAL_DIR}/figure_4_1_weight_trajectories.png")

    stats.to_csv(f"{EVAL_DIR}/weight_diagnostics.csv")

    # HDE versus equal-weight is the direct test of whether dynamic weighting matters.
    print("\nCritical test — does the dynamic weighting add anything?")
    hde_rets = STRATEGIES["e_HDE"]["daily_returns"]
    eq_rets = STRATEGIES["d_EqualWeight"]["daily_returns"]
    hde_port = STRATEGIES["e_HDE"]["portfolio"][["Date", "Strategy_Ret"]].rename(
        columns={"Strategy_Ret": "hde"})
    eq_port = STRATEGIES["d_EqualWeight"]["portfolio"][["Date", "Strategy_Ret"]].rename(
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

    # Check the same comparison as a direct daily-return difference.
    diff_rets = merged["hde"].values - merged["eq"].values
    bl = select_block_length(diff_rets)
    mean_diff, (lo, hi), _ = block_bootstrap(diff_rets, np.mean,
                                              n_boot=5000, block_len=bl)
    print(f"\n  Bootstrap mean daily return difference (HDE − EqualWt):")
    print(f"  point={mean_diff*1e4:+.2f} bps  CI=[{lo*1e4:+.2f}, {hi*1e4:+.2f}] bps")

    return daily, stats, test


WEIGHTS, WEIGHT_STATS, HDE_VS_EQUAL_TEST = weight_diagnosis()