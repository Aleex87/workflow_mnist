"""
Pydantic schemas for request and response validation.

Responsibilities:
- Define input schema for prediction
- Define response schema
"""
from pydantic import BaseModel, Field, conlist, confloat

amount_of_pixels = confloat(ge=0, le=16)
#conflotat gives a float between 0 and 16

class Request(BaseModel):
    features: conlist(amount_of_pixels, min_lenght=64, max_length=64)
#give amount pf pixels and define the amount of pixels

class Prediction(BaseModel):
    predicrtion: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)