"""
Model loading logic.

Responsibilities:
- Load TorchScript model from artifacts/
- Prepare model for inference
"""
from pathlib import Path
import torch

arifact_path = Path(__file__).resolve().parents[2] / "artifacts"
model_path = arifact_path / "digits_mlp.ts"

def loading_model():
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    return model