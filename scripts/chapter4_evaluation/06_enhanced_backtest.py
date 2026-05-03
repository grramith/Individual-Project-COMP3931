import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calculate_max_drawdown(equity_curve):
    # Peak-to-trough as a negative fraction so it slots straight into reporting
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def calculate_sharpe(returns, periods=252):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods)


def run_enhanced_backtest():
    # Pull tuned thresholds from Script 07 instead of hardcoding - keeps backtest in sync with config
    path = 'data/results/hde_final_results.csv'
    config_path = 'data/results/best_ensemble_config.json'
    if not os.path.exists(path) or not os.path.exists(config_path):
        print('Error: Run Script 07 (Enhanced HDE) first.')
        return

    df = pd.read_csv(path, parse_dates=['Date'])
    with open(config_path) as f:
        config = json.load(f)

    # Tuned parameters from validation - vix keys vary across config versions, so handle both layouts
    THRESHOLD     = config['threshold']
    VIX_LOW       = config.get('vix_low', config.get('vix_threshold', 18.0))
    VIX_HIGH      = config.get('vix_high', config.get('vix_threshold', 22.0))
    USE_FRACTIONAL = config['fractional']
    ALLOW_SHORT   = config.get('allow_short', False)
    DD_LIMIT      = config['dd_limit']
    POS_SCALE     = config.get('pos_scale', 1)
    TX_COST       = 0.0005  # 5 bps round-trip
    INITIAL_CAPITAL = 1000  # NAV-normalised so equity curves are comparable from t=0

    print(f'Backtest Configuration (tuned on validation):')
    print(f'  Confidence Threshold: {THRESHOLD}')
    print(f'  VIX Regime (low/high): {VIX_LOW:.1f} / {VIX_HIGH:.1f}')
    print(f'  Fractional Sizing:    {USE_FRACTIONAL}')
    print(f'  Short Selling:        {ALLOW_SHORT}')
    print(f'  Drawdown Limit:       {DD_LIMIT:.0%}')
    print(f'  Transaction Cost:     {TX_COST*10000:.0f} bps')
    print('=' * 70)

    final_returns = []
    per_stock_metrics = []

    for ticker in df['Ticker'].unique():
        t_df = df[df['Ticker'] == ticker].copy().sort_values('Date')
        n = len(t_df)

        pred = t_df['Ensemble_Delta'].values
        actual = t_df['Actual'].values
        vix = t_df['VIX_Value'].values if 'VIX_Value' in t_df.columns else np.zeros(n)

        position = np.zeros(n)
        strat_rets = np.zeros(n)
        equity = np.zeros(n)
        equity[0] = INITIAL_CAPITAL
        peak = INITIAL_CAPITAL

        for i in range(1, n):
            # Signal lag - trade tomorrow on yesterday's signal to avoid look-ahead
            p = pred[i - 1]

            # Multi-level VIX regime - threshold widens in higher-vol regimes to reduce false trades
            v = vix[i-1]
            if v > VIX_HIGH:
                eff_threshold = THRESHOLD * 3.0
            elif v > VIX_LOW:
                eff_threshold = THRESHOLD * 1.5
            else:
                eff_threshold = THRESHOLD

            # Position sizing - long always, short only when explicitly enabled
            denom = eff_threshold * 5 + 1e-9
            if USE_FRACTIONAL:
                if p > eff_threshold:
                    position[i] = min(p / denom * POS_SCALE, 1.0)
                elif ALLOW_SHORT and p < -eff_threshold:
                    position[i] = max(p / denom * POS_SCALE, -1.0)
                else:
                    position[i] = 0.0
            else:
                if p > eff_threshold:
                    position[i] = 1.0
                elif ALLOW_SHORT and p < -eff_threshold:
                    position[i] = -1.0
                else:
                    position[i] = 0.0

            # Gradual taper instead of hard cutoff - exposure shrinks linearly past the DD limit
            current_dd = (equity[i-1] - peak) / peak
            if current_dd < -DD_LIMIT:
                severity = min((abs(current_dd) - DD_LIMIT) / DD_LIMIT, 1.0)
                position[i] *= max(1.0 - severity, 0.0)

            # Net return after costs - cost charged on absolute change in position
            pos_change = abs(position[i] - position[i-1])
            ret = position[i] * actual[i] - pos_change * TX_COST
            strat_rets[i] = ret

            equity[i] = equity[i-1] * (1 + ret)
            peak = max(peak, equity[i])

        t_df['Position'] = position
        t_df['Strategy_Ret'] = strat_rets
        t_df['Equity'] = equity

        # Per-stock metrics for the cross-sectional report
        stock_sharpe = calculate_sharpe(pd.Series(strat_rets))
        traded = t_df[t_df['Position'] > 0]
        hit_rate = (traded['Actual'] > 0).mean() if len(traded) > 0 else 0.0
        eq_series = pd.Series(equity[1:])
        max_dd = calculate_max_drawdown(eq_series)
        total_ret = (equity[n-1] / INITIAL_CAPITAL - 1) * 100

        per_stock_metrics.append({
            'Ticker': ticker, 'Sharpe': stock_sharpe, 'Hit_Rate': hit_rate,
            'Max_Drawdown': max_dd, 'Total_Return_Pct': total_ret,
            'Avg_Position': position.mean(), 'Num_Trades': int((np.diff(position) != 0).sum())
        })
        print(f'  {ticker}: Sharpe={stock_sharpe:.2f} | Hit={hit_rate:.1%} | '
              f'MaxDD={max_dd:.1%} | Return={total_ret:.1f}% | '
              f'AvgPos={position.mean():.2f} | Trades={per_stock_metrics[-1]["Num_Trades"]}')

        final_returns.append(t_df)

    processed_df = pd.concat(final_returns)

    # Equal-weight aggregation across tickers gives the headline portfolio numbers
    portfolio = processed_df.groupby('Date').agg({
        'Actual': 'mean',
        'Strategy_Ret': 'mean',
        'Weight_RF': 'mean',
        'Weight_GB': 'mean',
        'Weight_LSTM': 'mean',
        'Position': 'mean',
    }).reset_index()

    portfolio['Market_Cum'] = INITIAL_CAPITAL * (1 + portfolio['Actual']).cumprod()
    portfolio['HDE_Cum'] = INITIAL_CAPITAL * (1 + portfolio['Strategy_Ret']).cumprod()

    m_sharpe = calculate_sharpe(portfolio['Actual'])
    s_sharpe = calculate_sharpe(portfolio['Strategy_Ret'])
    m_dd = calculate_max_drawdown(portfolio['Market_Cum'])
    s_dd = calculate_max_drawdown(portfolio['HDE_Cum'])
    portfolio_hit = (processed_df[processed_df['Position'] > 0]['Actual'] > 0).mean()
    total_market = (portfolio['Market_Cum'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    total_strat = (portfolio['HDE_Cum'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    avg_exposure = portfolio['Position'].mean()

    print(f'\n{"=" * 70}')
    print(f'PORTFOLIO RESULTS')
    print(f'{"=" * 70}')
    print(f'{"Metric":<25} {"Buy & Hold":>15} {"HDE v2":>15}')
    print(f'{"-"*55}')
    print(f'{"Total Return":<25} {total_market:>14.1f}% {total_strat:>14.1f}%')
    print(f'{"Sharpe Ratio":<25} {m_sharpe:>15.2f} {s_sharpe:>15.2f}')
    print(f'{"Max Drawdown":<25} {m_dd:>14.1%} {s_dd:>14.1%}')
    print(f'{"Hit Rate":<25} {"N/A":>15} {portfolio_hit:>14.1%}')
    print(f'{"Avg Exposure":<25} {"100%":>15} {avg_exposure:>14.1%}')

    return (processed_df, portfolio, per_stock_metrics, config,
            DD_LIMIT, INITIAL_CAPITAL,
            m_sharpe, s_sharpe, m_dd, s_dd,
            portfolio_hit, total_market, total_strat, avg_exposure)


if __name__ == '__main__':
    run_enhanced_backtest()