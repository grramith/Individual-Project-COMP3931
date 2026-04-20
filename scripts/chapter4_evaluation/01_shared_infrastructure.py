import numpy as np

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