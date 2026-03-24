"""
Find optimal classification threshold — image-extracted 1-lead model.

Evaluates model/ecg_cnn_image.pth on the image-extracted validation set
and prints four threshold candidates.

Run from project root:
    venv/Scripts/python -m evaluation.find_threshold_image

Update THRESHOLD_IMAGE in inference/predict.py with the chosen value,
then run eval_test_image.py for unbiased test-set metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    confusion_matrix, precision_recall_curve
)

from model.ecg_cnn_500hz import ECGCNN500Hz
from dataset.ecg_dataset import ECGDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "model/ecg_cnn_image.pth"
DATA_DIR   = r"D:\Major project\ECG_MI_Project\data\processed_1lead_image"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            probs = F.softmax(model(X), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    return np.array(all_probs), np.array(all_labels)


def threshold_metrics(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1          = f1_score(labels, preds, zero_division=0)
    return sensitivity, specificity, f1


def find_best_f1_threshold(probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx  = np.argmax(f1_scores[:-1])
    return thresholds[best_idx], f1_scores[best_idx]


def find_youden_threshold(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden   = tpr - fpr
    best_idx = np.argmax(youden)
    return thresholds[best_idx], youden[best_idx]


def find_sensitivity_threshold(probs, labels, min_sensitivity):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    idx = np.where(tpr >= min_sensitivity)[0]
    if len(idx) == 0:
        return thresholds[-1]
    return thresholds[idx[-1]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Image Model threshold results")
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_PATH}")
    print(f"Data:   {DATA_DIR}")

    model = ECGCNN500Hz().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    val_ds = ECGDataset(DATA_DIR, split="val", augment=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nVal set: {len(val_ds)} samples")

    probs, labels = collect_probs(model, val_dl)

    auc = roc_auc_score(labels, probs)
    print(f"Val ROC-AUC: {auc:.4f}")

    print("\n--- Threshold options ---")

    # 1. Best F1
    t_f1, best_f1 = find_best_f1_threshold(probs, labels)
    sens, spec, f1 = threshold_metrics(probs, labels, t_f1)
    print(f"\n[1] Best F1 threshold = {t_f1:.4f}")
    print(f"    F1={f1:.4f}  Sensitivity={sens:.4f}  Specificity={spec:.4f}")

    # 2. Youden's J
    t_youden, youden_j = find_youden_threshold(probs, labels)
    sens, spec, f1 = threshold_metrics(probs, labels, t_youden)
    print(f"\n[2] Youden's J threshold = {t_youden:.4f}  (J={youden_j:.4f})")
    print(f"    F1={f1:.4f}  Sensitivity={sens:.4f}  Specificity={spec:.4f}")

    # 3. Sensitivity >= 0.90
    t_90 = find_sensitivity_threshold(probs, labels, 0.90)
    sens, spec, f1 = threshold_metrics(probs, labels, t_90)
    print(f"\n[3] Sensitivity >= 0.90 threshold = {t_90:.4f}")
    print(f"    F1={f1:.4f}  Sensitivity={sens:.4f}  Specificity={spec:.4f}")

    # 4. Sensitivity >= 0.95
    t_95 = find_sensitivity_threshold(probs, labels, 0.95)
    sens, spec, f1 = threshold_metrics(probs, labels, t_95)
    print(f"\n[4] Sensitivity >= 0.95 threshold = {t_95:.4f}")
    print(f"    F1={f1:.4f}  Sensitivity={sens:.4f}  Specificity={spec:.4f}")

    print("\nUpdate THRESHOLD_IMAGE in inference/predict.py with the chosen value.")


if __name__ == "__main__":
    main()
