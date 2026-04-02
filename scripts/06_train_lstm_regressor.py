import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    """
    Two-layer LSTM for daily return prediction.
    Dropout is applied between LSTM layers and before the output
    to reduce overfitting on noisy financial data.
    """
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
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)

def create_sequences_per_ticker(X, y, metadata, seq_len):
    """
    Create sequences within each ticker separately so windows
    do not mix rows from different stocks.
    """
    sequences, targets, meta_rows = [], [], []
    tickers = metadata["Ticker"].unique()

    for ticker in tickers:
        mask = metadata["Ticker"].values == ticker
        X_tick = X[mask]
        y_tick = y[mask]
        meta_tick = metadata[mask].reset_index(drop=True)

        for i in range(seq_len, len(X_tick)):
            sequences.append(X_tick[i - seq_len:i])
            targets.append(y_tick[i])
            meta_rows.append({
                "Date": meta_tick.iloc[i]["Date"],
                "Ticker": ticker
            })

    return (
        np.array(sequences),
        np.array(targets),
        pd.DataFrame(meta_rows)
    )