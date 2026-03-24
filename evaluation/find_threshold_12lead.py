"""
Find optimal threshold — 12-lead model, with side-by-side 1-lead comparison.

Sweeps thresholds 0.10 -> 0.90 on the test set for both models and
prints a side-by-side comparison table.

Run from project root:
    venv/Scripts/python -m evaluation.find_threshold_12lead
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

from model.ecg_cnn_1lead  import ECGCNN1Lead
from model.ecg_cnn_12lead import ECGCNN12Lead
from dataset.ecg_dataset  import ECGDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_1LEAD  = r"D:\Major project\ECG_MI_Project\data\processed_1lead_500hz"
DATA_12LEAD = r"D:\Major project\ECG_MI_Project\data\processed_12lead_full"
MODEL_1LEAD  = "model/ecg_cnn_1lead.pth"
MODEL_12LEAD = "model/ecg_cnn_12lead.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_1lead():
    m = ECGCNN1Lead().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_1LEAD, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def load_12lead():
    m = ECGCNN12Lead().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_12LEAD, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def get_probs(model, loader):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            probs = F.softmax(model(X), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    return np.array(all_probs), np.array(all_labels)


def sweep(probs, labels):
    auc = roc_auc_score(labels, probs)
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_f1 = -1
    best_t  = 0.5
    best_m  = None
    results = {}
    for t in thresholds:
        preds = (probs >= t).astype(int)
        acc   = (preds == labels).mean()
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        f1   = f1_score(labels, preds, zero_division=0)
        results[round(t, 2)] = (acc, sens, spec, f1)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = round(t, 2)
            best_m  = (acc, sens, spec, f1)
    return auc, best_t, best_m, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}\n")

    # --- 1-Lead ---
    print("Evaluating 1-lead model...")
    m1    = load_1lead()
    ds1   = ECGDataset(DATA_1LEAD,  split="test", is_12lead=False, augment=False)
    dl1   = DataLoader(ds1, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    p1, l1 = get_probs(m1, dl1)
    auc1, bt1, bm1, res1 = sweep(p1, l1)
    del m1

    # --- 12-Lead ---
    print("Evaluating 12-lead model...")
    m12   = load_12lead()
    ds12  = ECGDataset(DATA_12LEAD, split="test", is_12lead=True,  augment=False)
    dl12  = DataLoader(ds12, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    p12, l12 = get_probs(m12, dl12)
    auc12, bt12, bm12, res12 = sweep(p12, l12)
    del m12

    # --- Side-by-side table ---
    print(f"\n{'='*76}")
    print(f"  {'THRESHOLD SWEEP — TEST SET COMPARISON':^72}")
    print(f"{'='*76}")
    print(f"  {'':>6}  {'--- 1-Lead CNN ---':^26}  {'--- 12-Lead CNN ---':^26}")
    print(f"  {'Thresh':>6}  {'Sens':>6} {'Spec':>6} {'F1':>6}  {'Sens':>6} {'Spec':>6} {'F1':>6}")
    print(f"  {'-'*68}")

    for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        r1  = res1.get(t,  (0, 0, 0, 0))
        r12 = res12.get(t, (0, 0, 0, 0))
        m1_mark  = " *" if t == bt1  else "  "
        m12_mark = " *" if t == bt12 else "  "
        print(f"  {t:6.2f}  "
              f"{r1[1]:6.4f} {r1[2]:6.4f} {r1[3]:6.4f}{m1_mark} "
              f"{r12[1]:6.4f} {r12[2]:6.4f} {r12[3]:6.4f}{m12_mark}")

    print(f"\n  * = best F1 threshold")
    print(f"\n  1-Lead  ROC-AUC : {auc1:.4f}   Best threshold: {bt1:.2f}  "
          f"F1={bm1[3]:.4f}  Sens={bm1[1]:.4f}  Spec={bm1[2]:.4f}")
    print(f"  12-Lead ROC-AUC : {auc12:.4f}   Best threshold: {bt12:.2f}  "
          f"F1={bm12[3]:.4f}  Sens={bm12[1]:.4f}  Spec={bm12[2]:.4f}")

    print(f"\nUpdate inference/predict.py:")
    print(f"  THRESHOLD_1LEAD  = {bt1:.2f}")
    print(f"  THRESHOLD_12LEAD = {bt12:.2f}")


if __name__ == "__main__":
    main()
