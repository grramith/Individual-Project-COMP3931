import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time


class LSTMRegressor(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
       # Use the final timestep representation for prediction
        lstm_out, _ = self.lstm(x)
        # Use the last time step's output
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)

def create_sequences_per_ticker(X, y, metadata, seq_len):

    sequences, targets, meta_rows = [], [], []
    
    # Group by ticker
    tickers = metadata['Ticker'].unique()
    
    for ticker in tickers:
        mask = metadata['Ticker'].values == ticker
        X_tick = X[mask]
        y_tick = y[mask]
        meta_tick = metadata[mask].reset_index(drop=True)
        
        # Create sequences within this ticker
        for i in range(seq_len, len(X_tick)):
            sequences.append(X_tick[i - seq_len:i])
            targets.append(y_tick[i])
            meta_rows.append({
                'Date': meta_tick.iloc[i]['Date'],
                'Ticker': ticker
            })
    
    return (
        np.array(sequences), 
        np.array(targets),
        pd.DataFrame(meta_rows)
    )


def train_lstm():
    # Load scaled inputs and regression targets
    X_train = np.load("data/modeling/X_train.npy")
    X_val = np.load("data/modeling/X_val.npy")
    X_test = np.load("data/modeling/X_test.npy")
    y_train = np.load("data/modeling/y_train_returns.npy")
    y_val = np.load("data/modeling/y_val_returns.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")
    
   # Load metadata so sequence construction can stay ticker-specific
    meta_train = pd.read_csv("data/modeling/train_metadata.csv")
    meta_val = pd.read_csv("data/modeling/val_metadata.csv")
    meta_test = pd.read_csv("data/modeling/test_metadata.csv")
    
     # Use the best available device on the current machine (In this we don't actually use CUDA since all the code has been ran on Apple Silicon, but this is future-proofing for anyone who might want to run it on a GPU and for good coding practice))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Keep the search space small and practical
    hyperparam_configs = [
        {"seq_len": 10, "hidden_size": 64, "lr": 0.001, "dropout": 0.3},
        {"seq_len": 20, "hidden_size": 64, "lr": 0.001, "dropout": 0.3},
        {"seq_len": 10, "hidden_size": 128, "lr": 0.0005, "dropout": 0.2},
        {"seq_len": 20, "hidden_size": 128, "lr": 0.0005, "dropout": 0.2},
    ]
    
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    PATIENCE = 10  # Early stopping patience
    
    best_val_loss = float('inf')
    best_config = None
    best_model_state = None
    tuning_log = []
    
    print("Tuning LSTM hyperparameters on validation set...")
    print("=" * 70)
    start_lstm = time.time()
    
    for cfg in hyperparam_configs:
        seq_len = cfg["seq_len"]
        print(f"\nConfig: {cfg}")
        
        # Build rolling windows separately for each ticker
        X_tr_seq, y_tr_seq, _ = create_sequences_per_ticker(
            X_train, y_train, meta_train, seq_len
        )
        X_val_seq, y_val_seq, _ = create_sequences_per_ticker(
            X_val, y_val, meta_val, seq_len
        )
        
        if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
            print(f"  Skipping: insufficient data for seq_len={seq_len}")
            continue
        
         # Convert sequence arrays into PyTorch datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr_seq),
            torch.FloatTensor(y_tr_seq)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialise model
        input_size = X_train.shape[1]
        model = LSTMRegressor(
            input_size=input_size,
            hidden_size=cfg["hidden_size"],
            num_layers=2,
            dropout=cfg["dropout"]
        ).to(device)
        
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = nn.MSELoss()
        
        # Training with early stopping
        best_epoch_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(NUM_EPOCHS):
            # Training step
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimiser.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(train_dataset)
            
            # Training step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_loss += loss.item() * len(X_batch)
            val_loss /= len(val_dataset)
            
            # Keep the best validation checkpoint seen so far
            if val_loss < best_epoch_val_loss:
                best_epoch_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress every 10 epochs so training is easier to track
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        tuning_log.append({
            "config": str(cfg),
            "best_val_loss": best_epoch_val_loss,
            "stopped_epoch": epoch + 1
        })
        
        print(f"  >> Best Val Loss: {best_epoch_val_loss:.6f}")
        
        # Update the overall best configuration if this run improved validation loss
        if best_epoch_val_loss < best_val_loss:
            best_val_loss = best_epoch_val_loss
            best_config = cfg
            best_model_state = best_state

    print(f"LSTM tuning completed in {(time.time() - start_lstm) / 60:.1f} minutes")
    # Stop early if no valid configuration produced a usable model
    if best_config is None:
        print("ERROR: No valid configuration found!")
        return None, None
    
    print(f"\n{'=' * 70}")
    print(f"Best LSTM Config: {best_config}")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    
    # Rebuild the best model and use it for final test inference
    seq_len = best_config["seq_len"]
    input_size = X_train.shape[1]
    
    final_model = LSTMRegressor(
        input_size=input_size,
        hidden_size=best_config["hidden_size"],
        num_layers=2,
        dropout=best_config["dropout"]
    ).to(device)
    final_model.load_state_dict(best_model_state)
    
    # Build ticker-safe test sequences with the selected sequence length
    X_test_seq, y_test_seq, meta_test_seq = create_sequences_per_ticker(
        X_test, y_test, meta_test, seq_len
    )
    
    final_model.eval()
    with torch.no_grad():
        test_preds = final_model(
            torch.FloatTensor(X_test_seq).to(device)
        ).cpu().numpy()
    
    # Report final predictive and directional performance on the test set
    test_mae = np.mean(np.abs(y_test_seq - test_preds))
    test_dir_acc = np.mean((test_preds > 0) == (y_test_seq > 0))
    
    print(f"\nTest Performance:")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  Directional Accuracy: {test_dir_acc:.2%}")
    print(f"  Test sequences: {len(test_preds)} (from {len(y_test)} raw test rows)")

    # Create output folders before saving model artefacts and prediction files
    os.makedirs("models/lstm", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    
    torch.save(best_model_state, "models/lstm/best_lstm.pth")
    
    with open("models/lstm/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    # Save predictions with Date and Ticker so they can be merged later without misalignment
    lstm_results = meta_test_seq.copy()
    lstm_results['Pred_LSTM'] = test_preds
    lstm_results['Actual'] = y_test_seq
    lstm_results.to_csv("data/results/lstm_predictions.csv", index=False)
    
    # Save the full tuning history for later analysis
    pd.DataFrame(tuning_log).to_csv("data/results/lstm_tuning_log.csv", index=False)
    
    print(f"\nLSTM predictions saved with Date+Ticker alignment to data/results/lstm_predictions.csv")
    print(f"Model saved to models/lstm/")
    return test_preds, y_test_seq

if __name__ == "__main__":
    train_lstm()