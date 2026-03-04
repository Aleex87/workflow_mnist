# workflow_mnist

# Digits Classification API

Authors  
Alessandro Abbate  
Alicia Gezelius

This project implements a simple machine learning workflow where a PyTorch model is integrated into a FastAPI application and distributed using Docker containers.

The model classifies handwritten digits from the sklearn `digits` dataset (8x8 images).

---

# How to run the project

## Run locally

Install dependencies
 > uv sync 

 
Start the API

> uvicorn src.api.main:app --reload


Open Swagger documentation

http://localhost:8000/docs


---

## Run using Docker

Build the container

> docker build -t digits-api .


Run the container

> docker run -p 8000:8000 digits-api


Open the API documentation

> http://localhost:8000/docs


---

# API Endpoint

POST `/predict`

The API expects a JSON body containing 64 pixel values.

Example:

{
"features": [64 values]
}


The API returns:

- predicted digit
- confidence score

---

# Example requests

You can copy and paste these examples directly into Swagger.

Example 1

{
"features":[
0,0,5,13,9,1,0,0,
0,0,13,15,10,15,5,0,
0,3,15,2,0,11,8,0,
0,4,12,0,0,8,8,0,
0,5,8,0,0,9,8,0,
0,4,11,0,1,12,7,0,
0,2,14,5,10,12,0,0,
0,0,6,13,10,0,0,0
]
}

Example 2 

{
"features":[
0,0,0,12,13,5,0,0,
0,0,4,16,8,0,0,0,
0,0,8,16,16,6,0,0,
0,0,8,16,16,8,0,0,
0,0,8,16,16,8,0,0,
0,0,7,16,16,6,0,0,
0,0,2,12,16,4,0,0,
0,0,0,2,6,0,0,0
]
}

Example 3

{
"features":[
0,0,0,0,14,13,1,0,
0,0,0,5,16,16,2,0,
0,0,0,12,15,16,2,0,
0,0,0,3,12,16,6,0,
0,0,0,0,11,16,8,0,
0,0,0,0,12,16,7,0,
0,0,0,1,15,16,3,0,
0,0,0,0,13,10,0,0
]
}


---

# Project structure

src/model
model definition, training and export

src/api
FastAPI application

artifacts
exported TorchScript model

notebooks
EDA and training notebooks

tests
example requests


## Model — Digits Classification (MLP)

This part of the project focuses on building and validating a PyTorch model
before integrating it into the API.

The development process is divided into two main notebooks:

---

###  1 `notebooks/eda.ipynb`

Purpose: Explore and understand the dataset.

In this notebook we:

- Load the `sklearn.datasets.load_digits()` dataset
- Inspect shapes and input dimensions
- Analyze pixel value range
- Verify class balance
- Visualize digit samples
- Define the preprocessing strategy

Main conclusions:

- 1797 samples
- 8×8 grayscale images
- Flattened input dimension: 64
- Pixel range: [0, 16]
- Problem type: supervised multi-class classification (10 classes)

Preprocessing decision:

- Flatten 8×8 → 64 features
- Normalize with: `x_norm = x_raw / 16.0`
- Use stratified train/test split

---

### 2 `notebooks/traning.ipynb`

Purpose: Train and evaluate the MLP model.

In this notebook we:

- Perform train/test split (80/20, stratified)
- Normalize inputs
- Convert data to PyTorch tensors
- Use DataLoader for mini-batch training
- Define an MLP architecture:
  
  64 → 128 → 64 → 10 (ReLU activations)

- Train for 25 epochs
- Evaluate test accuracy
- Plot training loss and accuracy curves

Final performance:

- Test Accuracy ≈ 97%
- Smooth convergence
- No strong overfitting observed

---

### Why these notebooks matter

The notebooks serve as:

- Experimental validation of the model
- Definition of the preprocessing pipeline
- Reference implementation for production code

The logic implemented here will be transferred into:

- `src/model/model.py`
- `src/model/train.py`
- `src/model/export.py`

This ensures consistency between experimentation and deployment.

### API Input 

Request:
{
  "features": [64 float values]
}

- Length must be exactly 64
- Values must be in the range [0, 16]
- Order: row-major flatten of the 8×8 image

Preprocessing inside API:
x_norm = x_raw / 16.0

### API Output

{
  "prediction": int (0–9),
  "confidence": float (0–1)
}

______________ API _______________________


