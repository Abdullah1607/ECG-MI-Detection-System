"""
Final unbiased test-set evaluation — both 1-lead and 12-lead models.

Run AFTER updating THRESHOLD_1LEAD / THRESHOLD_12LEAD in inference/predict.py.

Run from project root:
    venv/Scripts/python -m evaluation.eval_test
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, confusion_matrix,
    classification_report
)

from model.ecg_cnn_500hz import ECGCNN500Hz
from model.ecg_cnn_12lead import ECGCNN12Lead
from dataset.ecg_dataset import ECGDataset, ECGDataset12Lead
from inference.predict import THRESHOLD_1LEAD, THRESHOLD_12LEAD

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, threshold, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            probs = F.softmax(model(X), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)

    auc             = roc_auc_score(labels, probs)
    tn, fp, fn, tp  = confusion_matrix(labels, preds).ravel()
    sensitivity     = tp / (tp + fn + 1e-8)
    specificity     = tn / (tn + fp + 1e-8)
    f1              = f1_score(labels, preds, zero_division=0)
    ppv             = tp / (tp + fp + 1e-8)   # precision
    npv             = tn / (tn + fn + 1e-8)

    return {
        "AUC":         round(float(auc), 4),
        "Sensitivity": round(float(sensitivity), 4),
        "Specificity": round(float(specificity), 4),
        "F1":          round(float(f1), 4),
        "PPV":         round(float(ppv), 4),
        "NPV":         round(float(npv), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "N":  int(len(labels)),
        "threshold": threshold,
    }


def print_results(name, r):
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Test set size : {r['N']} samples")
    print(f"  Threshold     : {r['threshold']}")
    print(f"  AUC           : {r['AUC']}")
    print(f"  Sensitivity   : {r['Sensitivity']}")
    print(f"  Specificity   : {r['Specificity']}")
    print(f"  F1 score      : {r['F1']}")
    print(f"  PPV (precision): {r['PPV']}")
    print(f"  NPV           : {r['NPV']}")
    print(f"  Confusion matrix:")
    print(f"    TP={r['TP']}  FP={r['FP']}")
    print(f"    FN={r['FN']}  TN={r['TN']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    # ---- 1-lead ----
    print("\nLoading 1-lead model...")
    model_1l = ECGCNN500Hz().to(DEVICE)
    model_1l.load_state_dict(
        torch.load("model/ecg_cnn_500hz.pth", map_location=DEVICE, weights_only=True)
    )

    test_ds_1l = ECGDataset("data/processed_1lead_500hz", split="test", augment=False)
    test_dl_1l = DataLoader(test_ds_1l, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    r1 = evaluate_model(model_1l, test_dl_1l, THRESHOLD_1LEAD, DEVICE)
    print_results("1-lead CNN (Lead II, 500 Hz)", r1)

    # ---- 12-lead ----
    print("\nLoading 12-lead model...")
    model_12l = ECGCNN12Lead().to(DEVICE)
    model_12l.load_state_dict(
        torch.load("model/ecg_cnn_12lead.pth", map_location=DEVICE, weights_only=True)
    )

    test_ds_12l = ECGDataset12Lead("data/processed_12lead_full", split="test", augment=False)
    test_dl_12l = DataLoader(test_ds_12l, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    r12 = evaluate_model(model_12l, test_dl_12l, THRESHOLD_12LEAD, DEVICE)
    print_results("12-lead CNN (500 Hz)", r12)

    print("\nDone.")


if __name__ == "__main__":
    main()
