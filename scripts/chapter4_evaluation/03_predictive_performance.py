# Builds Tables 4.1 and 4.2 from the shared prediction and backtest setup.

# Strategy builders all return the same schema expected by run_backtest().
def preds_for_model(col_name):
    # Use one prediction column at a time while keeping the shared metadata fixed.
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value", col_name]].copy()
    df.rename(columns={col_name: "Prediction"}, inplace=True)
    return df.dropna(subset=["Prediction"])


def build_buy_and_hold():
    # Constant positive signal gives the always-long benchmark.
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value"]].copy()
    df["Prediction"] = 1.0
    return df


def build_momentum_12_1():
    # Use 12-month momentum excluding the most recent month as a non-ML comparator.
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
        mom, on=["Date", "Ticker"], how="left"
    )
    df.rename(columns={"mom_12_1": "Prediction"}, inplace=True)
    # Put annual momentum onto a daily-return scale for the shared threshold rule.
    df["Prediction"] = df["Prediction"] / 252
    return df.dropna(subset=["Prediction"])


def build_equal_weight_ensemble():
    # Fixed 1/3 weights isolate whether dynamic weighting adds anything.
    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value",
                "Pred_RF", "Pred_GB", "Pred_LSTM"]].copy()
    df["Prediction"] = df[["Pred_RF", "Pred_GB", "Pred_LSTM"]].mean(axis=1)
    return df.dropna(subset=["Prediction"])


def build_table_4_1():
    # Summarise forecast accuracy before adding the trading overlay.
    print("\n" + "=" * 78)
    print("TABLE 4.1 — Predictive Performance (95% block bootstrap CIs)")
    print("=" * 78)

    rows = []
    errors = {}

    for model in ["Linear", "Ridge", "RF", "GB", "LSTM", "HDE"]:
        col = f"Pred_{model}"
        if col not in PREDS.columns or PREDS[col].isna().all():
            continue

        sub = PREDS[["Date", "Ticker", "Actual", col]].dropna().copy()
        sub["Error"] = sub["Actual"] - sub[col]

        pred = sub[col].values
        actual = sub["Actual"].values
        errs = sub["Error"].values

        # Keep dated errors so the DM tests compare aligned forecasts.
        errors[model] = sub[["Date", "Ticker", "Error"]].copy()

        abs_errs = np.abs(errs)
        bl = select_block_length(abs_errs)

        mae_pt, (mae_lo, mae_hi), _ = block_bootstrap(
            abs_errs, np.mean, n_boot=5000, block_len=bl
        )

        # Resample prediction and actual together so directional accuracy stays paired.
        paired = np.column_stack([pred, actual])

        def dir_stat(p):
            return float(np.mean((p[:, 0] > 0) == (p[:, 1] > 0)))

        da_pt, (da_lo, da_hi), _ = block_bootstrap(
            paired, dir_stat, n_boot=5000, block_len=bl
        )

        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        pt_res = pesaran_timmermann(pred, actual, null=0.5)

        rows.append({
            "Model": model,
            "N": len(sub),
            "MAE": mae_pt,
            "MAE_CI_lo": mae_lo,
            "MAE_CI_hi": mae_hi,
            "DirAcc": da_pt,
            "DA_CI_lo": da_lo,
            "DA_CI_hi": da_hi,
            "R2": r2,
            "PT_p_vs_0.5": pt_res["p_value"],
            "block_len": bl,
        })

    table = pd.DataFrame(rows)

    # Run DM tests only after aligning models on the same Date and Ticker rows.
    models = list(errors.keys())
    dm_p = pd.DataFrame(index=models, columns=models, dtype=float)

    for a in models:
        for b in models:
            if a == b:
                dm_p.loc[a, b] = np.nan
                continue

            merged = errors[a].merge(
                errors[b],
                on=["Date", "Ticker"],
                how="inner",
                suffixes=(f"_{a}", f"_{b}")
            )

            if len(merged) < 10:
                dm_p.loc[a, b] = np.nan
                continue

            _, p = diebold_mariano(
                merged[f"Error_{a}"].values,
                merged[f"Error_{b}"].values,
                loss="abs"
            )
            dm_p.loc[a, b] = p

    # Traded-day accuracy is reported separately because flat days dilute the signal test.
    hde = PREDS[["Pred_HDE", "Actual"]].dropna()
    hde_traded = hde[hde["Pred_HDE"].abs() > HDE_CONFIG["threshold"]]

    if len(hde_traded) > 0:
        pt_traded_50 = pesaran_timmermann(
            hde_traded["Pred_HDE"].values,
            hde_traded["Actual"].values,
            null=0.5
        )
        pt_traded_53 = pesaran_timmermann(
            hde_traded["Pred_HDE"].values,
            hde_traded["Actual"].values,
            null=0.53
        )
        pt_traded_55 = pesaran_timmermann(
            hde_traded["Pred_HDE"].values,
            hde_traded["Actual"].values,
            null=0.55
        )
    else:
        pt_traded_50 = pt_traded_53 = pt_traded_55 = None

    print(table.to_string(
        index=False,
        formatters={
            "MAE": "{:.5f}".format,
            "MAE_CI_lo": "{:.5f}".format,
            "MAE_CI_hi": "{:.5f}".format,
            "DirAcc": "{:.4f}".format,
            "DA_CI_lo": "{:.4f}".format,
            "DA_CI_hi": "{:.4f}".format,
            "R2": "{:+.4f}".format,
            "PT_p_vs_0.5": "{:.4f}".format,
        }
    ))

    print("\nDiebold–Mariano pairwise p-values (MAE, HLN-corrected):")
    print(dm_p.round(4).to_string())

    print("\nHDE directional accuracy on TRADED days only:")
    for label, r in [("vs 0.50", pt_traded_50),
                     ("vs 0.53", pt_traded_53),
                     ("vs 0.55", pt_traded_55)]:
        if r is None:
            continue
        print(f"  {label}:  hit={r['hit_rate']:.4f}  p={r['p_value']:.4f}  ({r['test']})")

    table.to_csv(f"{EVAL_DIR}/table_4_1_predictive_performance.csv", index=False)
    dm_p.to_csv(f"{EVAL_DIR}/table_4_1_dm_matrix.csv")
    return table, dm_p, errors


def diebold_mariano(e1, e2, h=1, loss="abs"):
    # Fail loudly here because the caller should already have aligned the errors.
    e1, e2 = np.asarray(e1), np.asarray(e2)

    if len(e1) != len(e2):
        raise ValueError(
            f"Diebold-Mariano requires aligned error series of equal length, "
            f"got {len(e1)} and {len(e2)}."
        )

    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sq":
        d = e1 ** 2 - e2 ** 2
    else:
        raise ValueError(loss)

    T = len(d)
    d_bar = np.mean(d)

    gamma_0 = np.var(d, ddof=0)
    gamma = [gamma_0]
    for k in range(1, h):
        gk = np.mean((d[:-k] - d_bar) * (d[k:] - d_bar))
        gamma.append(gk)

    var_d = gamma[0] + 2 * sum(gamma[1:])
    var_d = max(var_d, 1e-12) / T

    dm = d_bar / np.sqrt(var_d)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * hln_factor
    p = 2 * (1 - sp_stats.t.cdf(abs(dm_hln), df=T - 1))
    return float(dm_hln), float(p)


TABLE_4_1, DM_MATRIX, FORECAST_ERRORS = build_table_4_1()


def run_strategy(label, preds_df, **override):
    # Start from the tuned HDE overlay, then override only when a baseline requires it.
    kwargs = dict(
        threshold=HDE_CONFIG["threshold"],
        vix_low=HDE_CONFIG["vix_low"],
        vix_high=HDE_CONFIG["vix_high"],
        use_fractional=HDE_CONFIG.get("fractional", True),
        allow_short=HDE_CONFIG.get("allow_short", False),
        dd_limit=HDE_CONFIG["dd_limit"],
    )
    kwargs.update(override)
    result = run_backtest(preds_df, **kwargs)
    result["label"] = label
    return result


def build_table_4_2():
    # Build the baseline ladder from passive exposure through the full HDE.
    print("\n" + "=" * 78)
    print("TABLE 4.2 — Baseline Ladder (95% CIs, paired tests vs HDE)")
    print("=" * 78)

    strategies = {}

    # Buy and Hold turns the overlay off so it remains a clean market benchmark.
    bh_preds = build_buy_and_hold()
    strategies["a_BuyHold"] = run_strategy(
        "Buy & Hold", bh_preds,
        threshold=0.0, use_threshold=False, use_vix_filter=False,
        use_taper=False, use_fractional=False, allow_short=False,
    )

    # Momentum gives a standard finance baseline under the same overlay.
    try:
        mom_preds = build_momentum_12_1()
        strategies["b_Momentum"] = run_strategy("12-1 Momentum", mom_preds)
    except Exception as e:
        print(f"  [warn] momentum baseline failed: {e}")

    # OLS checks whether a simple linear signal benefits from the overlay.
    if "Pred_Linear" in PREDS.columns and not PREDS["Pred_Linear"].isna().all():
        strategies["c_OLS_overlay"] = run_strategy(
            "OLS + overlay", preds_for_model("Pred_Linear"))

    # Equal weighting keeps the HDE constituents but removes adaptive weighting.
    strategies["d_EqualWeight"] = run_strategy(
        "Equal-weight static ens.", build_equal_weight_ensemble())

    # Full HDE is the proposed method.
    strategies["e_HDE"] = run_strategy("Full HDE", preds_for_model("Pred_HDE"))

    rows = []
    for key, res in strategies.items():
        s = res["stats"]
        rets = res["daily_returns"]
        bl = select_block_length(rets)
        sr_pt, (sr_lo, sr_hi), _ = block_bootstrap(
            rets, lambda x: sharpe_annualised(x), n_boot=5000, block_len=bl)
        rows.append({
            "Strategy": res["label"],
            "Total Return %": round(s["total_return_pct"], 1),
            "Sharpe": round(s["sharpe"], 3),
            "Sharpe_CI_lo": round(sr_lo, 3),
            "Sharpe_CI_hi": round(sr_hi, 3),
            "Sortino": round(s["sortino"], 3),
            "Calmar": round(s["calmar"], 3),
            "Max DD %": round(s["max_drawdown"] * 100, 1),
            "Exposure %": round(s["avg_exposure"] * 100, 1),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # Pair strategies by date before testing Sharpe differences.
    print("\nJobson–Korkie–Memmel Sharpe tests (vs Full HDE):")
    hde_rets = strategies["e_HDE"]["daily_returns"]
    pair_pvals = {}
    for key, res in strategies.items():
        if key == "e_HDE":
            continue
        hde_port = strategies["e_HDE"]["portfolio"][["Date", "Strategy_Ret"]].rename(
            columns={"Strategy_Ret": "hde"})
        other_port = res["portfolio"][["Date", "Strategy_Ret"]].rename(
            columns={"Strategy_Ret": "other"})
        merged = hde_port.merge(other_port, on="Date", how="inner")
        t = sharpe_difference_test(merged["hde"].values, merged["other"].values)
        print(f"  HDE vs {res['label']:<26}  "
              f"ΔSR={t['diff']:+.3f}  z={t['z']:+.2f}  p={t['p_value']:.4f}")
        pair_pvals[f"HDE_vs_{key}"] = t["p_value"]

    # Holm correction keeps the ladder comparisons from being over-read.
    adj = holm_correction(pair_pvals)
    print("\nHolm-corrected p-values:")
    for k, v in adj.items():
        mark = "★" if v["reject"] else " "
        print(f"  {mark} {k:<30}  raw={v['raw']:.4f}  adj={v['adj']:.4f}")

    table.to_csv(f"{EVAL_DIR}/table_4_2_baseline_ladder.csv", index=False)
    pd.DataFrame(adj).T.to_csv(f"{EVAL_DIR}/table_4_2_sharpe_tests.csv")
    return strategies, table, adj


STRATEGIES, TABLE_4_2, LADDER_PVALS = build_table_4_2()