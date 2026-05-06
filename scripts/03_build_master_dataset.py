import pandas as pd
import numpy as np
import os

# Wilder RSI
def calculate_rsi_wilder(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    # avoid divide by zero
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def build_dataset():
    prices_path = "data/raw/prices.csv"
    macro_path = "data/raw/macro_fred.csv" 

    if not os.path.exists(prices_path) or not os.path.exists(macro_path):
        print("Error: Raw files not found.")
        return

    # load inputs
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    
    # SPY as market proxy
    spy_ret = prices['SPY'].pct_change() if 'SPY' in prices.columns else None
    
    mag_7 = [t for t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'] if t in prices.columns]
    final_dfs = []

    print("Building Research-Grade Features")

    for ticker in mag_7:
        # one stock at a time
        df = pd.DataFrame(prices[ticker]).rename(columns={ticker: 'Adj_Close'})

        # returns
        df['Return_1d'] = df['Adj_Close'].pct_change()
        df['Return_5d'] = df['Adj_Close'].pct_change(5)
        df['Return_21d'] = df['Adj_Close'].pct_change(21)
        df['Market_Return'] = spy_ret

        # MA ratios
        df['MA10_Ratio'] = df['Adj_Close'] / df['Adj_Close'].rolling(window=10, min_periods=10).mean()
        df['MA50_Ratio'] = df['Adj_Close'] / df['Adj_Close'].rolling(window=50, min_periods=50).mean()

        # MACD
        ema12 = df['Adj_Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Adj_Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (ema12 - ema26) / df['Adj_Close']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        df['RSI'] = calculate_rsi_wilder(df['Adj_Close'])

        # 20d realised vol
        df['Vol_20d'] = df['Return_1d'].rolling(window=20, min_periods=20).std()

        # 10d momentum
        df['Momentum_10d'] = df['Adj_Close'].pct_change(10)

        # ffill macro to avoid look-ahead
        df = df.join(macro, how='left').ffill()

        # next-day return + sign
        df['Target_Return'] = df['Return_1d'].shift(-1)
        df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
        
        df['Ticker'] = ticker
        df = df.reset_index().rename(columns={"index": "Date"})
        
        final_dfs.append(df)

    # stack into one frame
    master_df = pd.concat(final_dfs, axis=0)
    master_df = master_df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    
    # drop infs and NaNs
    master_df = master_df.replace([np.inf, -np.inf], np.nan).dropna()

    os.makedirs("data/processed", exist_ok=True)
    master_df.to_csv("data/processed/master_dataset.csv", index=False)
    
    print(f"Final Master Dataset Saved (Long Format)")
    print(f"Total Observations: {len(master_df)}")
    print(f"Features: {[c for c in master_df.columns if c not in ['Date','Ticker','Adj_Close','Target_Return','Target_Direction']]}")
    print(f"Date range: {master_df['Date'].min()} to {master_df['Date'].max()}")
    return master_df

if __name__ == "__main__":
    master_df = build_dataset()