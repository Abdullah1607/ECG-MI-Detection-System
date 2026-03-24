"""
Model inference — supports 1-lead, 12-lead, and image-extracted CNN.

Thresholds are placeholders; update after running:
    venv/Scripts/python -m evaluation.find_threshold
    venv/Scripts/python -m evaluation.find_threshold_12lead
    venv/Scripts/python -m evaluation.find_threshold_image
"""

import torch
import torch.nn.functional as F
import numpy as np

from model.ecg_cnn_1lead   import ECGCNN1Lead
from model.ecg_cnn_500hz   import ECGCNN500Hz
from model.ecg_cnn_12lead  import ECGCNN12Lead

# ---------------------------------------------------------------------------
# Thresholds — update after evaluation
# ---------------------------------------------------------------------------
THRESHOLD_1LEAD       = 0.48   # digital file upload (1-lead)
THRESHOLD_1LEAD_IMAGE = 0.60   # legacy: image upload before image model existed
THRESHOLD_12LEAD      = 0.43   # digital file upload (12-lead)
THRESHOLD_IMAGE       = 0.91    # image model: best F1 (Sens=0.826, Spec=0.618)

MODEL_PATH_1LEAD  = "model/ecg_cnn_1lead.pth"
MODEL_PATH_12LEAD = "model/ecg_cnn_12lead.pth"
MODEL_PATH_IMAGE  = "model/ecg_cnn_image.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model_1lead  = None
_model_12lead = None
_model_image  = None


# ---------------------------------------------------------------------------
# Model loaders (lazy singletons)
# ---------------------------------------------------------------------------

def get_model_1lead():
    global _model_1lead
    if _model_1lead is None:
        m = ECGCNN1Lead().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH_1LEAD, map_location=DEVICE, weights_only=True))
        m.eval()
        _model_1lead = m
    return _model_1lead


def get_model_12lead():
    global _model_12lead
    if _model_12lead is None:
        m = ECGCNN12Lead().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH_12LEAD, map_location=DEVICE, weights_only=True))
        m.eval()
        _model_12lead = m
    return _model_12lead


def get_model_image():
    global _model_image
    if _model_image is None:
        m = ECGCNN500Hz().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH_IMAGE, map_location=DEVICE, weights_only=True))
        m.eval()
        _model_image = m
    return _model_image


# keep alias used by explain.py
def get_model():
    return get_model_1lead()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_ecg(signal, model_type: str = "1lead", threshold: float = None):
    """
    Run inference on an ECG signal.

    Parameters
    ----------
    signal     : numpy array
                 (5000,)    for model_type=="1lead" or "image"
                 (12,5000)  for model_type=="12lead"
    model_type : "1lead" | "12lead" | "image"

    Returns
    -------
    prob_abnormal : float
    label         : "Normal" | "Abnormal"
    confidence    : float
    """
    signal = np.asarray(signal, dtype=np.float32)

    if model_type == "1lead":
        if signal.shape != (5000,):
            raise ValueError(f"1-lead signal must be (5000,), got {signal.shape}")
        x     = torch.tensor(signal).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,5000)
        model = get_model_1lead()
        thr   = THRESHOLD_1LEAD

    elif model_type == "12lead":
        if signal.shape != (12, 5000):
            raise ValueError(f"12-lead signal must be (12,5000), got {signal.shape}")
        x     = torch.tensor(signal).unsqueeze(0).to(DEVICE)  # (1,12,5000)
        model = get_model_12lead()
        thr   = THRESHOLD_12LEAD

    elif model_type == "image":
        if signal.shape != (5000,):
            raise ValueError(f"image signal must be (5000,), got {signal.shape}")
        x     = torch.tensor(signal).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,5000)
        model = get_model_image()
        thr   = THRESHOLD_IMAGE

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use '1lead', '12lead', or 'image'.")

    # Allow caller to override the default threshold
    if threshold is not None:
        thr = threshold

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)

    prob_abnormal = probs[0, 1].item()
    label         = "Abnormal" if prob_abnormal >= thr else "Normal"
    confidence    = prob_abnormal if label == "Abnormal" else (1.0 - prob_abnormal)

    return prob_abnormal, label, confidence
