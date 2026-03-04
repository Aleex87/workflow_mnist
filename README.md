# workflow_mnist

___________ EDA AND TRANING __________________

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

So for the api part i first made some paths to the model and artifacts. Then there are two funktions, the first just loads in the model and puts it on the cpu since it is a finished
model and the traning is done, and for the same reason the second funktion turns of the gradietns calcuations since that is only needed in the traning fase. In the schemas the amount of pixels is 
that goes in how to return the predtiocon and confidence. In main is the post and fastapi application. I did have some trouble with the request and apperenlty i needed the unqueeze to add a batch dimention since the the model needs a batch size and features according to chatgpt. Then we take the raw output and make it into probabilitys with softmax() and then using argmax to find the class. Then we do teh conficede and return the confidence and predicted class. 



______________ DEPLOY ____________________

In order to deploy it with docker we created a dockerfile and the first time we buildt it took almost 30 min just for the container. Appertenly that is beacsue the cuda wheels needed fot this the gpu and we did first are huge and take up alot of space. 


