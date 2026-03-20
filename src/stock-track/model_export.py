import torch
import joblib
from models import StockLSTM   # reuses the class definition

WINDOW      = 20
N_FEATURES  = 7

# Recreate model and load weights saved during training
# (add torch.save to 3_train_model.py first — see note below)
model = StockLSTM(n_features=N_FEATURES)
model.load_state_dict(torch.load("models/stock_lstm.pt"))
model.eval()

dummy_input = torch.randn(1, WINDOW, N_FEATURES)

torch.onnx.export(
    model,
    dummy_input,
    "models/stock_lstm.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=11
)

print("Exported to models/stock_lstm.onnx")