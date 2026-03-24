"""
Training script — image-extracted 1-lead CNN.

Trains on signals extracted from synthetic ECG paper images, so the model
learns the domain of image-extracted waveforms rather than clean digital signals.

Run from project root:
    venv/Scripts/python -m training.train_image

Output: model/ecg_cnn_image.pth
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score

from model.ecg_cnn_500hz import ECGCNN500Hz
from dataset.ecg_dataset import ECGDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = r"D:\Major project\ECG_MI_Project\data\processed_1lead_image"
MODEL_PATH = "model/ecg_cnn_image.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10

# Stability on Windows WDDM GPU
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_weighted_sampler(dataset):
    """Over-sample minority class to handle class imbalance."""
    labels  = dataset.y
    n_pos   = int(labels.sum())
    n_neg   = int((labels == 0).sum())
    weights = np.where(labels == 1, 1.0 / n_pos, 1.0 / n_neg)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )
    return sampler


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss   = criterion(logits, y)
            total_loss += loss.item() * len(y)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_probs)
    preds    = (np.array(all_probs) >= 0.5).astype(int)
    f1       = f1_score(all_labels, preds, zero_division=0)
    return avg_loss, auc, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_DIR}")

    # Datasets
    train_ds = ECGDataset(DATA_DIR, split="train", augment=True)
    val_ds   = ECGDataset(DATA_DIR, split="val",   augment=False)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    sampler  = make_weighted_sampler(train_ds)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

    # Model — same architecture as production 1-lead 500Hz model
    model     = ECGCNN500Hz().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    best_auc   = 0.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0

        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

        train_loss /= len(train_dl.dataset)

        # --- Validate ---
        val_loss, val_auc, val_f1 = evaluate(model, val_dl, criterion)
        scheduler.step(val_auc)

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_AUC={val_auc:.4f} | "
            f"val_F1={val_f1:.4f}"
        )

        # --- Early stopping / checkpoint ---
        if val_auc > best_auc:
            best_auc   = val_auc
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved best model (AUC={best_auc:.4f}) -> {MODEL_PATH}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
