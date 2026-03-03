"""
FastAPI application entry point.

Responsibilities:
- Initialize FastAPI app
- Load exported TorchScript model
- Define POST /predict endpoint
"""
from fastapi import FastAPI
import torch
from src.api.model_loader import loading_model, pred
from src.api.schemas import Request, Prediction


app = FastAPI()
model = loading_model()

@app.post("/predict", response_model=Prediction)
def predict(request:Request):
    ten = torch.tensor(request.features, dtype=torch.float32).unsqueeze(0)
    log = pred(model, ten)
    prob = torch.softmax(log, dim=1)
    pred_class = torch.argmax(prob, dim=1).item()
    conf = torch.max(prob, dim =1).values.item()

    return Prediction(prediction = pred_class, confidence=conf)

