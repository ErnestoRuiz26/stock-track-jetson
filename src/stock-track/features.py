import pandas as pd
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/raw/aapl_prices.csv", index_col=0, parse_dates=True)

# --- Feature engineering ---
df['MA_5']  = df['Close'].rolling(5).mean()
df['MA_20'] = df['Close'].rolling(20).mean()

delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))

df['Volatility']    = df['Close'].rolling(10).std()
df['Volume_Change'] = df['Volume'].pct_change()

# --- Label: 1 if tomorrow closes higher, 0 if lower ---
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop rows with NaN (from rolling windows)
df.dropna(inplace=True)

df.to_csv("data/features.csv")
print(f"Saved {len(df)} rows to data/features.csv")
print(df[['Close', 'MA_5', 'RSI', 'Target']].tail())