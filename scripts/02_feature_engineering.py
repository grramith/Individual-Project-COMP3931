import pandas as pd
from fredapi import Fred
import os
from dotenv import load_dotenv
load_dotenv()

# config
FRED_API_KEY = os.environ.get('FRED_API_KEY')
OUTPUT_PATH = 'data/raw/macro_fred.csv'
START_DATE = '2014-01-01'
END_DATE = '2026-01-01' # cuts off at end of 2025


# FRED series IDs
MACRO_SERIES = {
    'fed_funds_rate': 'DFF',
    'us10y_yield': 'DGS10',
    'vix': 'VIXCLS',
    'cpi': 'CPIAUCSL',
    'unemployment_rate': 'UNRATE'
}

if not FRED_API_KEY:
    raise EnvironmentError(
        "If not working check if the .env file and the python -dotenv package has been installed (Personal Note: this error has happened before)"
    )

def collect_macro_data():
    fred = Fred(api_key=FRED_API_KEY)
    macro_frames = []
    print("Fetching Macro Indicators:")

    for name, series_id in MACRO_SERIES.items():
        print(f"  - {name}")
        # force numeric
        s = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
        s = pd.to_numeric(s, errors='coerce')
        s.name = name
        macro_frames.append(s)


    # CPI and unemployment are monthly, ffill to daily
    macro_df = pd.concat(macro_frames, axis=1)
    macro_df.index.name = 'Date'

    macro_df = macro_df.ffill()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    macro_df.to_csv(OUTPUT_PATH)
    
    print(f"Saved {len(macro_df)} rows to {OUTPUT_PATH}")
    return macro_df

if __name__ == "__main__":
    collect_macro_data()