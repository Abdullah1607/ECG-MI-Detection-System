import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGCNN500Hz(nn.Module):
    """
    1D-CNN for Lead II ECG classification — optimised for 500 Hz input.

    Kernel sizes are scaled relative to the original 100 Hz model so that
    each filter covers the same time window in milliseconds:
        conv1  k=75  → 150 ms  (P-wave detection)
        conv2  k=35  →  70 ms  (QRS complex)
        conv3  k=25  →  50 ms  (ST segment / fine features)
        conv4  k=13  →  26 ms  (sub-QRS features)

    Input:  (batch, 1, 5000)   — 10 s at 500 Hz, Z-scored
    Output: 2 classes (Normal=0 / Abnormal=1)

    After 4 × MaxPool(2):  5000 → 2500 → 1250 → 625 → 312
    FC input: 128 * 312 = 39 936
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1,   32,  kernel_size=75, padding=37)
        self.conv2 = nn.Conv1d(32,  64,  kernel_size=35, padding=17)
        self.conv3 = nn.Conv1d(64,  128, kernel_size=25, padding=12)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=13, padding=6)

        self.bn1   = nn.BatchNorm1d(32)
        self.bn2   = nn.BatchNorm1d(64)
        self.bn3   = nn.BatchNorm1d(128)
        self.bn4   = nn.BatchNorm1d(128)

        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(0.4)

        self.fc1   = nn.Linear(128 * 312, 256)
        self.fc2   = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 32,  2500)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, 64,  1250)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (B, 128,  625)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # (B, 128,  312)

        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
