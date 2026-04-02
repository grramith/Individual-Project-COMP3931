import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)


def create_sequences_per_ticker(X, y, metadata, seq_len):
    sequences = []
    targets = []

    tickers = metadata["Ticker"].unique()

    for ticker in tickers:
        mask = metadata["Ticker"].values == ticker
        X_ticker = X[mask]
        y_ticker = y[mask]

        for i in range(seq_len, len(X_ticker)):
            sequences.append(X_ticker[i - seq_len:i])
            targets.append(y_ticker[i])

    return np.array(sequences), np.array(targets)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_lstm():
    X_train = np.load("data/modeling/X_train.npy")
    X_val = np.load("data/modeling/X_val.npy")
    X_test = np.load("data/modeling/X_test.npy")

    y_train = np.load("data/modeling/y_train_returns.npy")
    y_val = np.load("data/modeling/y_val_returns.npy")
    y_test = np.load("data/modeling/y_test_returns.npy")

    meta_train = pd.read_csv("data/modeling/train_metadata.csv")
    meta_val = pd.read_csv("data/modeling/val_metadata.csv")
    meta_test = pd.read_csv("data/modeling/test_metadata.csv")

    device = get_device()
    print(f"Using device: {device}")

    hyperparam_configs = [
        {"seq_len": 10, "hidden_size": 64, "lr": 0.001, "dropout": 0.3},
        {"seq_len": 20, "hidden_size": 64, "lr": 0.001, "dropout": 0.3},
        {"seq_len": 10, "hidden_size": 128, "lr": 0.0005, "dropout": 0.2},
        {"seq_len": 20, "hidden_size": 128, "lr": 0.0005, "dropout": 0.2},
    ]

    num_epochs = 50
    batch_size = 64
    patience = 10

    best_val_loss = float("inf")
    best_config = None
    best_model_state = None

    print("Tuning LSTM hyperparameters on validation set...")
    print("=" * 70)

    for cfg in hyperparam_configs:
        seq_len = cfg["seq_len"]
        print(f"\nConfig: {cfg}")

        X_train_seq, y_train_seq = create_sequences_per_ticker(
            X_train, y_train, meta_train, seq_len
        )
        X_val_seq, y_val_seq = create_sequences_per_ticker(
            X_val, y_val, meta_val, seq_len
        )

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            print(f"Skipping seq_len={seq_len} because there is not enough data")
            continue

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = LSTMRegressor(
            input_size=X_train.shape[1],
            hidden_size=cfg["hidden_size"],
            num_layers=2,
            dropout=cfg["dropout"]
        ).to(device)

        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        best_epoch_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimiser.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_dataset)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_loss += loss.item() * len(X_batch)

            val_loss /= len(val_dataset)

            if val_loss < best_epoch_val_loss:
                best_epoch_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        print(f"  Best Val Loss: {best_epoch_val_loss:.6f}")

        if best_epoch_val_loss < best_val_loss:
            best_val_loss = best_epoch_val_loss
            best_config = cfg
            best_model_state = best_state

    if best_config is None:
        print("No valid LSTM config found")
        return None, None

    print("=" * 70)
    print(f"Best LSTM Config: {best_config}")
    print(f"Best Val Loss: {best_val_loss:.6f}")

    final_model = LSTMRegressor(
        input_size=X_train.shape[1],
        hidden_size=best_config["hidden_size"],
        num_layers=2,
        dropout=best_config["dropout"]
    ).to(device)
    final_model.load_state_dict(best_model_state)

    X_test_seq, y_test_seq = create_sequences_per_ticker(
        X_test, y_test, meta_test, best_config["seq_len"]
    )

    final_model.eval()
    with torch.no_grad():
        test_preds = final_model(torch.FloatTensor(X_test_seq).to(device)).cpu().numpy()

    test_mae = np.mean(np.abs(y_test_seq - test_preds))
    test_dir_acc = np.mean((test_preds > 0) == (y_test_seq > 0))

    print("\nTest Performance:")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  Directional Accuracy: {test_dir_acc:.2%}")
    print(f"  Test sequences: {len(test_preds)}")

    return test_preds, y_test_seq


if __name__ == "__main__":
    train_lstm()