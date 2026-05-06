import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def prepare_regression_data():
    # load master dataset
    input_path = "data/processed/master_dataset.csv"
    if not os.path.exists(input_path):
        print("Error: Master dataset not found. Run Script 03 first.")
        return

    df = pd.read_csv(input_path, parse_dates=['Date'])
    
    # target
    target_col = 'Target_Return'
    
    # drop leakage-prone cols
    drop_cols = ['Date', 'Ticker', 'Adj_Close', 'Target_Direction', 'Target_Return', 'Return_1d']
    features = [col for col in df.columns if col not in drop_cols]
    
    # chronological split
    val_start = '2023-01-01'
    test_start = '2024-01-01'
    
    train_df = df[df['Date'] < val_start].copy()
    val_df = df[(df['Date'] >= val_start) & (df['Date'] < test_start)].copy()
    test_df = df[df['Date'] >= test_start].copy()
    
    # check for overlap
    assert train_df["Date"].max() < val_df["Date"].min(), "Leakage: train/val overlap"
    assert val_df["Date"].max() < test_df["Date"].min(), "Leakage: val/test overlap"
    print("splits look clean")
    print(f"Temporal Split:")
    print(f"  Train:      {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(train_df)} samples)")
    print(f"  Validation: {val_df['Date'].min().date()} to {val_df['Date'].max().date()} ({len(val_df)} samples)")
    print(f"  Test:       {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(test_df)} samples)")
    
    X_train = train_df[features]
    y_train = train_df[target_col]
    
    X_val = val_df[features]
    y_val = val_df[target_col]
    
    X_test = test_df[features]
    y_test = test_df[target_col]
    
    # fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)     
    X_val_scaled = scaler.transform(X_val)            
    X_test_scaled = scaler.transform(X_test)            
    
    # save splits
    os.makedirs("data/modeling", exist_ok=True)
    
    np.save("data/modeling/X_train.npy", X_train_scaled)
    np.save("data/modeling/X_val.npy", X_val_scaled)
    np.save("data/modeling/X_test.npy", X_test_scaled)
    np.save("data/modeling/y_train_returns.npy", y_train.values)
    np.save("data/modeling/y_val_returns.npy", y_val.values)
    np.save("data/modeling/y_test_returns.npy", y_test.values)
    
    joblib.dump(scaler, "data/modeling/scaler.pkl")
    
    # metadata for backtest joins
    train_df[['Date', 'Ticker', 'Adj_Close']].to_csv("data/modeling/train_metadata.csv", index=False)
    val_df[['Date', 'Ticker', 'Adj_Close']].to_csv("data/modeling/val_metadata.csv", index=False)
    test_df[['Date', 'Ticker', 'Adj_Close']].to_csv("data/modeling/test_metadata.csv", index=False)
    
    # save feature names
    pd.Series(features).to_csv("data/modeling/feature_names.csv", index=False)

    print(f"\nPreprocessing Complete")
    print(f"Features ({len(features)}): {features}")
    print(f"All files saved to data/modeling/")

if __name__ == "__main__":
    prepare_regression_data()