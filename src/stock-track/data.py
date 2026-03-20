import yfinance as yf
import os

TICKER = "AAPL"
PERIOD = "5y"

os.makedirs("data", exist_ok=True)

ticker = yf.Ticker(TICKER)
df = ticker.history(period=PERIOD)

df.to_csv("data/raw/aapl_prices.csv")
print(f"Saved {len(df)} rows to data/raw/aapl_prices.csv")