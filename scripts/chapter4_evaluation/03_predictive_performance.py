
# Strategy builders - each returns a preds_df shaped for run_backtest
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
    # fall back to Target_Return if the daily return column isn't present
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
    # Rescale annualised momentum to a daily-return-equivalent so the same threshold logic applies
    df["Prediction"] = df["Prediction"] / 252
    return df.dropna(subset=["Prediction"])


def build_equal_weight_ensemble():

    df = PREDS[["Date", "Ticker", "Actual", "VIX_Value",
                "Pred_RF", "Pred_GB", "Pred_LSTM"]].copy()
    df["Prediction"] = df[["Pred_RF", "Pred_GB", "Pred_LSTM"]].mean(axis=1)
    return df.dropna(subset=["Prediction"])

def build_table_4_1():

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

        # store errors for the pairwise DM matrix below
        errors[model] = sub[["Date", "Ticker", "Error"]].copy()

        abs_errs = np.abs(errs)
        bl = select_block_length(abs_errs)

        mae_pt, (mae_lo, mae_hi), _ = block_bootstrap(
            abs_errs, np.mean, n_boot=5000, block_len=bl
        )

        # paired bootstrap for directional accuracy - column-stack so blocks resample (pred, actual) jointly
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

    # Pairwise DM matrix - only on the intersection of (Date, Ticker) so error series are aligned
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

    # Restrict the PT test to traded days only - HDE's untraded days dilute the hit rate
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

# Strict version of DM that errors out on length mismatches - safer than the Phase 1 version when the caller is responsible for alignment
def diebold_mariano(e1, e2, h=1, loss="abs"):
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


# Table 4.2 - the baseline ladder
def run_strategy(label, preds_df, **override):
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
    print("\n" + "=" * 78)
    print("TABLE 4.2 — Baseline Ladder (95% CIs, paired tests vs HDE)")
    print("=" * 78)

    strategies = {}

    # (a) Buy & Hold - all overlay flags off so the comparison isolates the prediction signal from the overlay
    bh_preds = build_buy_and_hold()
    strategies["a_BuyHold"] = run_strategy(
        "Buy & Hold", bh_preds,
        threshold=0.0, use_threshold=False, use_vix_filter=False,
        use_taper=False, use_fractional=False, allow_short=False,
    )

    # (b) 12-1 momentum - classic asset-pricing baseline, run through the same overlay as HDE
    try:
        mom_preds = build_momentum_12_1()
        strategies["b_Momentum"] = run_strategy("12-1 Momentum", mom_preds)
    except Exception as e:
        print(f"  [warn] momentum baseline failed: {e}")

    # (c) Linear regression predictions - tests whether overlay-on-OLS is enough
    if "Pred_Linear" in PREDS.columns and not PREDS["Pred_Linear"].isna().all():
        strategies["c_OLS_overlay"] = run_strategy(
            "OLS + overlay", preds_for_model("Pred_Linear"))

    # (d) Equal-weight ensemble - same constituents as HDE but fixed 1/3 weights, isolates the dynamic-weighting contribution
    strategies["d_EqualWeight"] = run_strategy(
        "Equal-weight static ens.", build_equal_weight_ensemble())

    # (e) Full HDE
    strategies["e_HDE"] = run_strategy("Full HDE", preds_for_model("Pred_HDE"))

    # Summary table with bootstrap Sharpe CIs
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

    # Pairwise Sharpe tests against HDE - merged on Date so JKM gets paired returns
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

    # Holm correction across the four ladder comparisons - controls familywise error
    adj = holm_correction(pair_pvals)
    print("\nHolm-corrected p-values:")
    for k, v in adj.items():
        mark = "★" if v["reject"] else " "
        print(f"  {mark} {k:<30}  raw={v['raw']:.4f}  adj={v['adj']:.4f}")

    table.to_csv(f"{EVAL_DIR}/table_4_2_baseline_ladder.csv", index=False)
    pd.DataFrame(adj).T.to_csv(f"{EVAL_DIR}/table_4_2_sharpe_tests.csv")
    return strategies, table, adj


STRATEGIES, TABLE_4_2, LADDER_PVALS = build_table_4_2()
