"""
Legacy training script — 1-lead 100 Hz CNN (old model, kept for reference).

IMPORTANT: This script uses the old architecture (ecg_cnn.py) and the old
preprocessed data (data/processed/). It is NOT used in the current inference
pipeline. The active training script is training/train_1lead_500hz.py.

Run from project root:
    venv/Scripts/python -m training.train
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

# Legacy model (100 Hz, kernel sizes scaled for 100 Hz)
# model/ecg_cnn.py must exist; this is the old architecture.
try:
    from model.ecg_cnn import ECGCNN
except ImportError:
    raise ImportError(
        "model/ecg_cnn.py not found. This is the legacy training script for the "
        "old 100 Hz model. Use training/train_1lead_500hz.py for current training."
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = "data/processed"
MODEL_PATH = "model/ecg_cnn_k15.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS     = 30
BATCH_SIZE = 32
LR         = 1e-3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[Legacy] Device: {DEVICE}")

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.int64)
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy")).astype(np.float32)
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.int64)

    # (N, L) -> (N, 1, L)
    X_train = X_train[:, np.newaxis, :]
    X_val   = X_val[:,   np.newaxis, :]

    # Weighted sampler
    n_pos   = int(y_train.sum())
    n_neg   = int((y_train == 0).sum())
    weights = np.where(y_train == 1, 1.0 / n_pos, 1.0 / n_neg)
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    model     = ECGCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in val_dl:
                X = X.to(DEVICE)
                probs = torch.softmax(model(X), dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.numpy())

        auc = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch:2d}/{EPOCHS} | val_AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved -> {MODEL_PATH}")

    print(f"Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
