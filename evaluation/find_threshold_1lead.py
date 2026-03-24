"""
Find optimal classification threshold — 1-lead model.

Sweeps thresholds 0.10 → 0.90 on the test set and reports:
  ROC-AUC, Accuracy, Sensitivity, Specificity, F1, Best threshold (by F1)

Run from project root:
    venv/Scripts/python -m evaluation.find_threshold_1lead
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

from model.ecg_cnn_1lead import ECGCNN1Lead
from dataset.ecg_dataset import ECGDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH  = r"D:\Major project\ECG_MI_Project\data\processed_1lead_500hz"
MODEL_PATH = "model/ecg_cnn_1lead.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model():
    model = ECGCNN1Lead().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def get_probs(model, loader):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            probs = F.softmax(model(X), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    return np.array(all_probs), np.array(all_labels)


def metrics_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    acc   = (preds == labels).mean()
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    f1   = f1_score(labels, preds, zero_division=0)
    return acc, sens, spec, f1


def sweep_and_print(probs, labels, model_name):
    auc = roc_auc_score(labels, probs)
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  ROC-AUC: {auc:.4f}   Test samples: {len(labels)}")
    print(f"{'='*60}")
    print(f"  {'Thresh':>6}  {'Acc':>6}  {'Sens':>6}  {'Spec':>6}  {'F1':>6}")
    print(f"  {'-'*38}")

    thresholds = np.arange(0.10, 0.91, 0.01)
    best_f1    = -1
    best_t     = 0.5
    best_row   = None

    for t in thresholds:
        acc, sens, spec, f1 = metrics_at_threshold(probs, labels, t)
        marker = " *" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1  = f1
            best_t   = t
            best_row = (acc, sens, spec, f1)
        if abs(t - round(t * 10) / 10) < 0.001:  # print every 0.10
            print(f"  {t:6.2f}  {acc:6.4f}  {sens:6.4f}  {spec:6.4f}  {f1:6.4f}{marker}")

    print(f"\n  Best threshold (F1): {best_t:.2f}")
    print(f"    Accuracy    : {best_row[0]:.4f}")
    print(f"    Sensitivity : {best_row[1]:.4f}")
    print(f"    Specificity : {best_row[2]:.4f}")
    print(f"    F1 Score    : {best_row[3]:.4f}")

    return auc, best_t, best_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    model = load_model()

    test_ds = ECGDataset(DATA_PATH, split="test", is_12lead=False, augment=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    probs, labels = get_probs(model, test_dl)
    auc, best_t, best_row = sweep_and_print(probs, labels, "1-Lead CNN (ecg_cnn_1lead.pth)")

    print(f"\nUpdate inference/predict.py:")
    print(f"  THRESHOLD_1LEAD = {best_t:.2f}")


if __name__ == "__main__":
    main()
