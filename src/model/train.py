"""
This file handles the training pipeline.

This script:
- Loads the dataset
- Performs stratified train/test split
- Normalizes inputs
- Trains the model
- Saves trained weights to artifacts/
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import DigitsMLP

# === Configuration ===
SEED = 42
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
TEST_SIZE = 0.2


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print("Device:", device)

    # === Load dataset ===
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # === Normalization 
    X_train = X_train / 16.0
    X_test = X_test / 16.0

    # === To tensors ===
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    # === Dataset + loaders ===
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # === Model / loss / optimizer ===
    model = DigitsMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # === Training loop ===
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # === Evaluation ===
        model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.append(preds)
                all_targets.append(yb.numpy())

        all_preds_np = np.concatenate(all_preds)
        all_targets_np = np.concatenate(all_targets)

        acc = accuracy_score(all_targets_np, all_preds_np)

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Test Acc: {acc:.4f}")

    # === Save artifacts 

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    weights_path = artifacts_dir / "digits_mlp_state_dict.pt"
    torch.save(model.state_dict(), weights_path)

    metadata = {
        "model_name": "DigitsMLP",
        "input_dim": 64,
        "num_classes": 10,
        "normalization": "x_norm = x_raw / 16.0",
        "expected_input": "features: list[float] length 64, raw range [0,16] (API normalizes)",
    }

    metadata_path = artifacts_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\nTraining completed.")
    print(f"Weights saved to: {weights_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()