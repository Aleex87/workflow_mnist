"""
Pydantic schemas for request and response validation.

Responsibilities:
- Define input schema for prediction
- Define response schema
"""
from fastAPI import FastAPI


class request(Basemodel):
    pred: int = Filed


