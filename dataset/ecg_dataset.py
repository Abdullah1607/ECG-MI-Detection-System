"""
ECG Dataset — folder-based per-file loading.

Expected directory layout:
    data_root/
        train/
            normal/     *.npy  shape (5000,)      label 0
            abnormal/   *.npy  shape (5000,)      label 1
        val/  ...
        test/ ...

For 12-lead, shape is (12, 5000) per file.

Usage:
    ds = ECGDataset(data_root, split="train", is_12lead=False, augment=True)
    signal, label = ds[0]   # signal: (1, 5000) or (12, 5000)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    Loads pre-processed ECG .npy files from a folder hierarchy.

    Parameters
    ----------
    data_root : root directory that contains train/val/test subdirs
    split     : "train", "val", or "test"
    is_12lead : if True, expect (12, 5000) files; else (5000,) files
    augment   : apply on-the-fly augmentation (train only)
    """

    def __init__(self, data_root: str, split: str, is_12lead: bool = False, augment: bool = False):
        self.is_12lead = is_12lead
        self.augment   = augment

        normal_dir   = os.path.join(data_root, split, "normal")
        abnormal_dir = os.path.join(data_root, split, "abnormal")

        if not os.path.isdir(normal_dir):
            raise FileNotFoundError(f"Normal directory not found: {normal_dir}")
        if not os.path.isdir(abnormal_dir):
            raise FileNotFoundError(f"Abnormal directory not found: {abnormal_dir}")

        self.samples = []  # list of (path, label) tuples

        for fname in sorted(os.listdir(normal_dir)):
            if fname.endswith(".npy"):
                self.samples.append((os.path.join(normal_dir, fname), 0))

        for fname in sorted(os.listdir(abnormal_dir)):
            if fname.endswith(".npy"):
                self.samples.append((os.path.join(abnormal_dir, fname), 1))

        self.y = np.array([lbl for _, lbl in self.samples], dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # Augmentation — applied per-signal (or per-lead for 12-lead)
    # ------------------------------------------------------------------
    def _augment_1d(self, sig: np.ndarray) -> np.ndarray:
        # Gaussian noise
        if np.random.rand() < 0.50:
            sig = sig + 0.02 * sig.std() * np.random.randn(len(sig)).astype(np.float32)
        # Amplitude scaling
        if np.random.rand() < 0.60:
            sig = sig * np.random.uniform(0.8, 1.2)
        # Baseline wander
        if np.random.rand() < 0.40:
            t    = np.linspace(0, 2 * np.pi, len(sig), dtype=np.float32)
            sig  = sig + 0.1 * np.sin(np.random.uniform(0.1, 0.5) * t)
        # Time shift
        if np.random.rand() < 0.40:
            sig = np.roll(sig, np.random.randint(-200, 201))
        # Polarity flip
        if np.random.rand() < 0.20:
            sig = -sig
        return sig

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        sig = np.load(path)  # (5000,) or (12, 5000)

        if self.is_12lead:
            sig = sig.copy().astype(np.float32)  # (12, 5000)
            if self.augment:
                for i in range(sig.shape[0]):
                    sig[i] = self._augment_1d(sig[i])
            tensor = torch.tensor(sig)            # (12, 5000)
        else:
            sig = sig.copy().astype(np.float32)   # (5000,)
            if self.augment:
                sig = self._augment_1d(sig)
            tensor = torch.tensor(sig).unsqueeze(0)  # (1, 5000)

        return tensor, torch.tensor(label, dtype=torch.long)
