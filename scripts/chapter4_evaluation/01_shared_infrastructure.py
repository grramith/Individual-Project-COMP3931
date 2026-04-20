import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

def _find_project_root():
    sentinel_dirs = {'data', 'scripts', 'models'}
    candidate = Path.cwd().resolve()
    while True:
        children = {p.name for p in candidate.iterdir() if p.is_dir()}
        if sentinel_dirs <= children:
            return candidate
        if candidate.parent == candidate:
            raise RuntimeError("Could not locate project root (need data/, scripts/, models/)")
        candidate = candidate.parent

PROJECT_ROOT = _find_project_root()
os.chdir(PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")

assert Path("data/results/best_ensemble_config.json").exists()
EVAL_DIR = "data/results/evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)

TRADING_DAYS = 252
TX_COST_DEFAULT = 0.0005
INITIAL_CAPITAL = 1000.0

def sharpe_annualised(rets, periods=TRADING_DAYS):
    rets = np.asarray(rets)
    if len(rets) == 0 or np.std(rets, ddof=1) == 0:
        return 0.0
    return (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(periods)

def sortino_annualised(rets, periods=TRADING_DAYS):
    rets = np.asarray(rets)
    downside = rets[rets < 0]
    if len(downside) == 0 or np.std(downside, ddof=1) == 0:
        return 0.0
    return (np.mean(rets) / np.std(downside, ddof=1)) * np.sqrt(periods)

def max_drawdown(equity):
    equity = np.asarray(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())

def calmar_ratio(rets, equity, periods=TRADING_DAYS):
    ann_ret = np.mean(rets) * periods
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return ann_ret / mdd

def run_backtest_per_ticker(preds_df, threshold=0.0, vix_low=18.0, vix_high=22.0,
                             use_fractional=True, allow_short=False, dd_limit=0.15,
                             use_threshold=True, use_vix_filter=True, use_taper=True,
                             tx_cost=TX_COST_DEFAULT, initial_capital=INITIAL_CAPITAL):
    per_ticker = []
    eff_threshold_base = threshold if use_threshold else 0.0

    for ticker in preds_df["Ticker"].unique():
        t = preds_df[preds_df["Ticker"] == ticker].copy().sort_values("Date").reset_index(drop=True)
        n = len(t)
        if n < 2:
            continue

        pred = t["Prediction"].values
        actual = t["Actual"].values
        vix = t["VIX_Value"].values if "VIX_Value" in t.columns else np.zeros(n)

        position = np.zeros(n)
        strat_rets = np.zeros(n)
        equity = np.zeros(n)
        equity[0] = initial_capital
        peak = initial_capital

        for i in range(1, n):
            p = pred[i - 1]
            v = vix[i - 1]

            if use_vix_filter and use_threshold:
                if v > vix_high:
                    eff = eff_threshold_base * 3.0
                elif v > vix_low:
                    eff = eff_threshold_base * 1.5
                else:
                    eff = eff_threshold_base
            else:
                eff = eff_threshold_base

            denom = eff * 5 + 1e-9
            if use_fractional and use_threshold:
                if p > eff:
                    position[i] = min(p / denom, 1.0)
                elif allow_short and p < -eff:
                    position[i] = max(p / denom, -1.0)
                else:
                    position[i] = 0.0
            else:
                if p > eff:
                    position[i] = 1.0
                elif allow_short and p < -eff:
                    position[i] = -1.0
                else:
                    position[i] = 0.0

            if use_taper:
                current_dd = (equity[i - 1] - peak) / peak if peak > 0 else 0.0
                if current_dd < -dd_limit:
                    severity = min((abs(current_dd) - dd_limit) / dd_limit, 1.0)
                    position[i] *= max(1.0 - severity, 0.0)

            pos_change = abs(position[i] - position[i - 1])
            ret = position[i] * actual[i] - pos_change * tx_cost
            strat_rets[i] = ret
            equity[i] = equity[i - 1] * (1 + ret)
            peak = max(peak, equity[i])

        t["Position"] = position
        t["Strategy_Ret"] = strat_rets
        t["Equity"] = equity
        per_ticker.append(t)

    return per_ticker

def aggregate_portfolio(per_ticker, initial_capital=INITIAL_CAPITAL):
    if not per_ticker:
        return None

    combined = pd.concat(per_ticker, ignore_index=True)

    port = combined.groupby("Date").agg(
        Actual=("Actual", "mean"),
        Strategy_Ret=("Strategy_Ret", "mean"),
        Position=("Position", "mean"),
    ).reset_index().sort_values("Date").reset_index(drop=True)

    port["Equity"] = initial_capital * (1 + port["Strategy_Ret"]).cumprod()
    port["Market_Cum"] = initial_capital * (1 + port["Actual"]).cumprod()

    rets = port["Strategy_Ret"].values
    mkt = port["Actual"].values

    stats = {
        "total_return_pct": (port["Equity"].iloc[-1] / initial_capital - 1) * 100,
        "sharpe": sharpe_annualised(rets),
        "sortino": sortino_annualised(rets),
        "max_drawdown": max_drawdown(port["Equity"].values),
        "calmar": calmar_ratio(rets, port["Equity"].values),
        "avg_exposure": float(np.mean(port["Position"])),
        "n_days": len(rets),
        "market_total_return_pct": (port["Market_Cum"].iloc[-1] / initial_capital - 1) * 100,
        "market_sharpe": sharpe_annualised(mkt),
        "market_max_drawdown": max_drawdown(port["Market_Cum"].values),
    }

    return {"per_ticker": per_ticker, "combined": combined, "portfolio": port,
            "stats": stats, "daily_returns": rets, "market_returns": mkt}

def run_backtest(preds_df, **kwargs):
    per_ticker = run_backtest_per_ticker(preds_df, **kwargs)
    return aggregate_portfolio(per_ticker, kwargs.get("initial_capital", INITIAL_CAPITAL))