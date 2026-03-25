import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TICKER     = "AAPL"
WINDOW     = 20
FEATURES   = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Volatility', 'Volume_Change']
ENGINE_PATH = "models/stock_lstm.trt"
SCALER_PATH = "models/scaler.pkl"

# ── 1. Fetch latest data ──────────────────────────────────────────────────────
print(f"Fetching latest {TICKER} data...")
df = yf.Ticker(TICKER).history(period="60d")

# ── 2. Build features (same as training) ─────────────────────────────────────
df['MA_5']  = df['Close'].rolling(5).mean()
df['MA_20'] = df['Close'].rolling(20).mean()

delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))

df['Volatility']    = df['Close'].rolling(10).std()
df['Volume_Change'] = df['Volume'].pct_change()
df.dropna(inplace=True)

# ── 3. Build the input window ─────────────────────────────────────────────────
window_data = df[FEATURES].values[-WINDOW:]          # last 20 days
scaler      = joblib.load(SCALER_PATH)
window_norm = scaler.transform(window_data)           # normalize
input_data  = window_norm.astype(np.float32)          # shape: (20, 7)
input_data  = input_data[np.newaxis, :, :]            # shape: (1, 20, 7)

# ── 4. Load TensorRT engine ───────────────────────────────────────────────────
logger = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# ── 5. Allocate GPU memory and run inference ──────────────────────────────────
input_nbytes  = input_data.nbytes
output_data   = np.empty((1, 2), dtype=np.float32)
output_nbytes = output_data.nbytes

d_input  = cuda.mem_alloc(input_nbytes)
d_output = cuda.mem_alloc(output_nbytes)

cuda.memcpy_htod(d_input, input_data)
context.execute_v2(bindings=[int(d_input), int(d_output)])
cuda.memcpy_dtoh(output_data, d_output)

# ── 6. Interpret and print result ─────────────────────────────────────────────
import torch
probs      = torch.softmax(torch.tensor(output_data[0]), dim=0)
prediction = "UP ▲" if probs[1] > probs[0] else "DOWN ▼"
confidence = probs.max().item() * 100

today_close = df['Close'].iloc[-1]
print(f"\n{'─'*40}")
print(f"  Ticker:     {TICKER}")
print(f"  Last close: ${today_close:.2f}")
print(f"  Prediction: {prediction}")
print(f"  Confidence: {confidence:.1f}%")
print(f"{'─'*40}\n")