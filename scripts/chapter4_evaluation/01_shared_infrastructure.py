# Phase 0 - shared infrastructure
# One backtest engine + one prediction loader so every variant in the chapter is compared on identical footing

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# Walk up from cwd until we hit the project root, so the notebook runs from any subdir
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
# Fail fast if the upstream HDE config is missing - nothing downstream will work without it
assert Path("data/results/best_ensemble_config.json").exists()
EVAL_DIR = "data/results/evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)

TRADING_DAYS = 252
TX_COST_DEFAULT = 0.0005  # 5 bps - matches the existing backtester which i had used for comparison purposes
INITIAL_CAPITAL = 1000.0


# Reusable backtest engine - every strategy in the chapter routes through this so results are comparable
def run_backtest(
    preds_df,
    threshold=0.0,
    vix_low=18.0,
    vix_high=22.0,
    use_fractional=True,
    allow_short=False,
    dd_limit=0.15,
    use_threshold=True,
    use_vix_filter=True,
    use_taper=True,
    tx_cost=TX_COST_DEFAULT,
    initial_capital=INITIAL_CAPITAL,
):

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
            # use yesterday's prediction to avoid look-ahead bias
            p = pred[i - 1]
            v = vix[i - 1]

            # scale threshold up in elevated-VIX regimes to reduce whipsaws
            if use_vix_filter and use_threshold:
                if v > vix_high:
                    eff = eff_threshold_base * 3.0
                elif v > vix_low:
                    eff = eff_threshold_base * 1.5
                else:
                    eff = eff_threshold_base
            else:
                eff = eff_threshold_base

            # fractional sizing: position scales linearly with signal strength
            denom = eff * 5 + 1e-9
            if use_fractional and use_threshold:
                if p > eff:
                    position[i] = min(p / denom, 1.0)
                elif allow_short and p < -eff:
                    position[i] = max(p / denom, -1.0)
                else:
                    position[i] = 0.0
            else:
                # binary sign-based entry when threshold is disabled or non-fractional
                if p > eff:
                    position[i] = 1.0
                elif allow_short and p < -eff:
                    position[i] = -1.0
                else:
                    position[i] = 0.0

            # drawdown taper - linearly reduce exposure once dd_limit is breached
            if use_taper:
                current_dd = (equity[i - 1] - peak) / peak if peak > 0 else 0.0
                if current_dd < -dd_limit:
                    severity = min((abs(current_dd) - dd_limit) / dd_limit, 1.0)
                    position[i] *= max(1.0 - severity, 0.0)

            # net return = directional P&L minus proportional transaction cost
            pos_change = abs(position[i] - position[i - 1])
            ret = position[i] * actual[i] - pos_change * tx_cost
            strat_rets[i] = ret
            equity[i] = equity[i - 1] * (1 + ret)
            peak = max(peak, equity[i])

        t["Position"] = position
        t["Strategy_Ret"] = strat_rets
        t["Equity"] = equity
        per_ticker.append(t)

    if not per_ticker:
        return None

    combined = pd.concat(per_ticker, ignore_index=True)

    # equal-weight across tickers per day - mirrors a naive Mag-7 basket
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

    return {
        "per_ticker": per_ticker,
        "combined": combined,
        "portfolio": port,
        "stats": stats,
        "daily_returns": rets,
        "market_returns": mkt,
    }


# Standard performance metrics, all annualised to 252 trading days
def sharpe_annualised(rets, periods=TRADING_DAYS):
    rets = np.asarray(rets)
    if len(rets) == 0 or np.std(rets, ddof=1) == 0:
        return 0.0
    return (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(periods)

def sortino_annualised(rets, periods=TRADING_DAYS):
    # penalise only downside deviation - better for asymmetric return profiles
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


# Pulls every model's test-set predictions into one wide table keyed on (Date, Ticker)
def load_all_test_predictions():
    X_test = np.load("data/modeling/X_test.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")
    test_meta = pd.read_csv("data/modeling/test_metadata.csv", parse_dates=["Date"])

    # load each baseline pkl and generate predictions on the held-out test set
    baselines = {}
    model_files = {
        "Linear": "models/baselines/Linear_Regression.pkl",
        "Ridge":  "models/baselines/Ridge_Regression.pkl",
        "RF":     "models/baselines/RF_Regressor.pkl",
        "GB":     "models/baselines/GB_Regressor.pkl",
    }
    for name, path in model_files.items():
        if not os.path.exists(path):
            print(f"  [warn] missing {path} — {name} column will be NaN")
            baselines[name] = np.full(len(X_test), np.nan)
            continue
        try:
            baselines[name] = joblib.load(path).predict(X_test)
            print(f"  [ok]   loaded {name}")
        except Exception as e:
            # Keep going if one model fails to unpickle - the table just drops that row
            print(f"  [warn] couldn't load {name} ({type(e).__name__})")
            print("         try re-running scripts/05_train_baseline_regressors.py")
            print(f"         skipping {name} for now — it won’t appear in Table 4.1")
            baselines[name] = np.full(len(X_test), np.nan)

    preds = test_meta[["Date", "Ticker"]].copy()
    preds["Actual"] = y_test
    for name, arr in baselines.items():
        preds[f"Pred_{name}"] = arr

    # bring in the HDE's blended predictions and per-constituent weights
    hde = pd.read_csv("data/results/hde_final_results.csv", parse_dates=["Date"])
    hde_slim = hde[["Date", "Ticker", "Ensemble_Delta", "VIX_Value",
                    "Weight_RF", "Weight_GB", "Weight_LSTM"]].copy()
    hde_slim.rename(columns={"Ensemble_Delta": "Pred_HDE"}, inplace=True)
    preds = preds.merge(hde_slim, on=["Date", "Ticker"], how="left")

    # LSTM predictions live in a separate CSV produced by script 06
    lstm_path = "data/results/lstm_predictions.csv"
    if os.path.exists(lstm_path):
        lstm = pd.read_csv(lstm_path, parse_dates=["Date"])
        lstm = lstm[["Date", "Ticker", "Pred_LSTM"]]
        preds = preds.merge(lstm, on=["Date", "Ticker"], how="left")
    else:
        preds["Pred_LSTM"] = np.nan

    # the HDE warm-up period produces NaN predictions - drop those rows
    preds = preds.dropna(subset=["Pred_HDE"]).reset_index(drop=True)
    preds = preds.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print(f"Loaded {len(preds)} test observations across {preds['Ticker'].nunique()} tickers")
    print(f"Date range: {preds['Date'].min().date()} → {preds['Date'].max().date()}")
    print(f"Columns: {list(preds.columns)}")
    return preds


# Reuse the tuned HDE config so every strategy in the ladder gets the same overlay parameters
with open("data/results/best_ensemble_config.json") as f:
    HDE_CONFIG = json.load(f)

print("HDE tuned config (applied to all strategies in the ladder):")
for k, v in HDE_CONFIG.items():
    print(f"  {k}: {v}")

PREDS = load_all_test_predictions()
PREDS.to_csv(f"{EVAL_DIR}/all_test_predictions.csv", index=False)
print(f"\nsaved combined predictions to {EVAL_DIR}/all_test_predictions.csv")
