"""
ECG file loading and preprocessing for inference.

Supports:
  .csv           -> 1-lead (Lead II)
  .dat + .hea    -> 12-lead (all leads) or 1-lead fallback
"""

import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample as scipy_resample

TARGET_FS     = 500
TARGET_LENGTH = 5000
LEAD_INDEX    = 1    # Lead II (0-indexed)
N_LEADS       = 12


def _normalize(sig: np.ndarray) -> np.ndarray:
    """Z-score normalisation."""
    sig  = sig.astype(np.float32)
    std  = sig.std()
    return (sig - sig.mean()) / (std + 1e-8)


def _resample(sig: np.ndarray, orig_fs: int) -> np.ndarray:
    if orig_fs == TARGET_FS:
        return sig.astype(np.float32)
    n_out = int(len(sig) * TARGET_FS / orig_fs)
    return scipy_resample(sig, n_out).astype(np.float32)


def _pad_trim(sig: np.ndarray) -> np.ndarray:
    if len(sig) >= TARGET_LENGTH:
        return sig[:TARGET_LENGTH]
    return np.pad(sig, (0, TARGET_LENGTH - len(sig)))


def preprocess_signal(sig: np.ndarray) -> np.ndarray:
    """
    Normalize + pad/trim a 1D signal to (5000,).
    Used for image-extracted signals.
    """
    sig = _normalize(sig)
    return _pad_trim(sig)


# ---------------------------------------------------------------------------
# 1-lead loaders
# ---------------------------------------------------------------------------

def load_csv_ecg(csv_path: str) -> np.ndarray:
    """
    Load single-column CSV ECG.
    Returns normalized (5000,) float32.
    """
    df = pd.read_csv(csv_path)
    sig = df.iloc[:, 0].values.astype(np.float32)
    sig = _normalize(sig)
    return _pad_trim(sig)


def load_csv_12lead(csv_path: str) -> np.ndarray:
    """
    Load 12-column CSV ECG (one column per lead).
    Returns normalized (12, 5000) float32, each lead independently Z-scored.
    """
    df = pd.read_csv(csv_path)
    out = np.zeros((N_LEADS, TARGET_LENGTH), dtype=np.float32)
    for i in range(N_LEADS):
        lead = df.iloc[:, i].values.astype(np.float32)
        lead = _normalize(lead)
        out[i] = _pad_trim(lead)
    return out


def load_wfdb_1lead(dat_path: str, hea_path: str) -> np.ndarray:
    """
    Load Lead II from a WFDB record.
    Returns normalized (5000,) float32.
    """
    record_path = dat_path.replace(".dat", "")
    signal, fields = wfdb.rdsamp(record_path)

    if signal.shape[1] <= LEAD_INDEX:
        # Fall back to first available lead
        lead = signal[:, 0].astype(np.float32)
    else:
        lead = signal[:, LEAD_INDEX].astype(np.float32)

    fs   = fields.get("fs", TARGET_FS)
    lead = _resample(lead, fs)
    lead = _normalize(lead)
    return _pad_trim(lead)


# ---------------------------------------------------------------------------
# 12-lead loader
# ---------------------------------------------------------------------------

def load_wfdb_12lead(dat_path: str, hea_path: str) -> np.ndarray:
    """
    Load all 12 leads from a WFDB record.
    Returns normalized (12, 5000) float32, each lead independently Z-scored.
    """
    record_path = dat_path.replace(".dat", "")
    signal, fields = wfdb.rdsamp(record_path)

    fs      = fields.get("fs", TARGET_FS)
    n_leads = signal.shape[1]
    out     = np.zeros((N_LEADS, TARGET_LENGTH), dtype=np.float32)

    for i in range(N_LEADS):
        if i < n_leads:
            lead = signal[:, i].astype(np.float32)
            lead = _resample(lead, fs)
            lead = _normalize(lead)
            lead = _pad_trim(lead)
            out[i] = lead
        # else: remain zeros (zero-padded if file has fewer than 12 leads)

    return out


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def load_ecg(files: list):
    """
    Detect file type and load appropriately.

    Returns
    -------
    (signal, model_type)
        signal     : numpy array
                     (5000,)    float32  if model_type == "1lead"
                     (12,5000)  float32  if model_type == "12lead"
        model_type : "1lead" or "12lead"
    """
    exts = [os.path.splitext(f)[1].lower() for f in files]

    if ".csv" in exts:
        if len(files) != 1:
            raise ValueError("Upload exactly ONE CSV file.")
        path = files[0]
        df   = pd.read_csv(path)
        ncols = df.shape[1]
        if ncols == 12:
            return load_csv_12lead(path), "12lead"
        elif ncols == 1:
            return load_csv_ecg(path), "1lead"
        else:
            # Fallback: extract column index 1 as Lead II
            sig = df.iloc[:, 1].values.astype(np.float32)
            sig = _normalize(sig)
            return _pad_trim(sig), "1lead"

    if ".dat" in exts and ".hea" in exts:
        dat = [f for f in files if f.endswith(".dat")][0]
        hea = [f for f in files if f.endswith(".hea")][0]
        signal = load_wfdb_12lead(dat, hea)
        return signal, "12lead"

    raise ValueError("Unsupported format. Upload .csv OR .dat + .hea pair.")
