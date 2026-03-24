"""
Training script — 12-lead CNN at 500 Hz.

Run from project root:
    venv/Scripts/python -m training.train_12lead

Saves best model (lowest val loss) to model/ecg_cnn_12lead.pth
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from model.ecg_cnn_12lead import ECGCNN12Lead
from dataset.ecg_dataset import ECGDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH  = r"D:\Major project\ECG_MI_Project\data\processed_12lead_full"
MODEL_SAVE = "model/ecg_cnn_12lead.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10

# Windows: num_workers must be 0 to avoid shared memory errors
NUM_WORKERS  = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device : {DEVICE}")
    print(f"Data   : {DATA_PATH}")
    print(f"Output : {MODEL_SAVE}")

    train_ds = ECGDataset(DATA_PATH, split="train", is_12lead=True, augment=True)
    val_ds   = ECGDataset(DATA_PATH, split="val",   is_12lead=True, augment=False)

    print(f"Train  : {len(train_ds)} samples  "
          f"(normal={int((train_ds.y==0).sum())}, "
          f"abnormal={int((train_ds.y==1).sum())})")
    print(f"Val    : {len(val_ds)} samples")

    # Class weights
    n_normal   = int((train_ds.y == 0).sum())
    n_abnormal = int((train_ds.y == 1).sum())
    n_total    = n_normal + n_abnormal
    w_normal   = n_total / (2.0 * n_normal)
    w_abnormal = n_total / (2.0 * n_abnormal)
    class_weights = torch.tensor([w_normal, w_abnormal], dtype=torch.float32).to(DEVICE)
    print(f"Class weights: normal={w_normal:.3f}, abnormal={w_abnormal:.3f}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    model     = ECGCNN12Lead().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )

    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):

        # --- Train ---
        model.train()
        train_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_dl.dataset)

        # --- Validate ---
        model.eval()
        val_loss   = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                loss   = criterion(logits, y)
                val_loss += loss.item() * len(y)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.cpu().numpy())
        val_loss /= len(val_dl.dataset)
        val_auc   = roc_auc_score(all_labels, all_probs)
        val_f1    = f1_score(all_labels, (np.array(all_probs) >= 0.5).astype(int), zero_division=0)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_AUC={val_auc:.4f} | "
              f"val_F1={val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  -> Saved best model  (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping: no improvement for {PATIENCE} epochs.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {MODEL_SAVE}")


if __name__ == "__main__":
    main()
