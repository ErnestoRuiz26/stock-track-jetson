import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("models", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)

FEATURE_COLS = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Volatility', 'Volume_Change']
TARGET_COL   = 'Target'
WINDOW       = 20

# ── 2. Chronological train/test split ────────────────────────────────────────
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

# ── 3. Build sequences ────────────────────────────────────────────────────────
def make_sequences(data, feature_cols, target_col, window):
    X, y = [], []
    values  = data[feature_cols].values
    targets = data[target_col].values
    for i in range(len(data) - window):
        X.append(values[i : i + window])
        y.append(targets[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X_train, y_train = make_sequences(train_df, FEATURE_COLS, TARGET_COL, WINDOW)
X_test,  y_test  = make_sequences(test_df,  FEATURE_COLS, TARGET_COL, WINDOW)

# ── 4. Normalize ──────────────────────────────────────────────────────────────
n_train, n_steps, n_features = X_train.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, n_steps, n_features)
X_test  = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], n_steps, n_features)

joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved.")

# ── 5. DataLoaders ────────────────────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=32, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
    batch_size=32
)

# ── 6. Model definition ───────────────────────────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.classifier(last_step)

model = StockLSTM(n_features=n_features)

# ── 7. Training ───────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            correct += (preds.argmax(1) == yb).sum().item()
            total   += len(yb)

    val_acc = correct / total
    scheduler.step(1 - val_acc)

    if (epoch + 1) % 10 == 0:
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.3f}")

print("\nTraining complete.")
torch.save(model.state_dict(), "models/stock_lstm.pt")
print("Model weights saved.")