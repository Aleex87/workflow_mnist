"""
FastAPI application entry point.

Responsibilities:
- Initialize FastAPI app
- Load exported TorchScript model
- Define POST /predict endpoint
"""
from fastapi import FastAPI
import torch
from os import path
from model_loader import loading_model, pred
from src.api.schemas import Request, Prediction


app = FastAPI()
model = loading_model()

@app.post(respose_pred=Prediction)
def predict(request:Request):
    ten = torch.tensor(request.features, dtype= Float32).unsqueeze(0)
    log = pred(model, ten)
    prob = torch.softmax(log, dim=1)
    pred_class = torch.argmax(prob, dim=1).item()
    confidence = torch.max(prob, dim =1).values.item()

    return Prediction(prediction = pred_class, confidence=confidence)

