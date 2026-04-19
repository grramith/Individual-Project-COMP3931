import yfinance as yf
import os
import pandas as pd

# Configuration for the "Research Baseline"
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'SPY']
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"  # Ensures data stops at 2025-12-31
OUTPUT_PATH = "data/raw/prices.csv"

def collect_prices():
    print(f"Downloading Mag 7 + Benchmark: {START_DATE} to {END_DATE}")
    
    # auto_adjust=True is critical for dividends/splits in dissertation research
    df = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)
    
    # yfinance returns multi-level columns, just need Close prices
    if isinstance(df.columns, pd.MultiIndex):
        data = df['Close']
    else:
        data = df
        
    data.index.name = 'Date'
    
    # Save the data
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data.to_csv(OUTPUT_PATH)
    
    print(f"Saved {len(data)} rows to {OUTPUT_PATH}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Tickers: {list(data.columns)}")
    return data

if __name__ == "__main__":
    collect_prices()