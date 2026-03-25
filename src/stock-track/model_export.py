# 4_export_model.py — full corrected file

import torch
import sys
sys.path.append('.')
from models import StockLSTM   # reuse class definition

WINDOW     = 20
N_FEATURES = 7

# Load weights into the trained model
trained_model = StockLSTM(n_features=N_FEATURES, dropout=0.3)
trained_model.load_state_dict(torch.load("models/stock_lstm.pt", map_location="cpu"))

# Create an export copy with dropout disabled
export_model = StockLSTM(n_features=N_FEATURES, dropout=0.0)
export_model.load_state_dict(trained_model.state_dict())
export_model.eval()

dummy_input = torch.randn(1, WINDOW, N_FEATURES)

torch.onnx.export(
    export_model,
    dummy_input,
    "models/stock_lstm.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=11,
    do_constant_folding=True,
    training=torch.onnx.TrainingMode.EVAL,
    dynamo=False 
)

print("Exported to models/stock_lstm.onnx")
import onnx
model_check = onnx.load("models/stock_lstm.onnx")
onnx.checker.check_model(model_check)
print(f"ONNX opset: {model_check.opset_import[0].version}")
print("Model is valid.")