import torch.nn as nn
import torch.nn.functional as F


class ECGCNN1Lead(nn.Module):
    """
    1D-CNN for single-lead (Lead II) ECG classification at 500 Hz.

    Input:  (batch, 1, 5000)
    Output: (batch, 2)  — logits for [Normal, Abnormal]

    Feature map sizes after each MaxPool(2):
        conv1: (B, 32, 2500)
        conv2: (B, 64, 1250)
        conv3: (B, 128, 625)
        FC in:  128 * 625 = 80 000
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1,   32,  kernel_size=15, padding=7)
        self.bn1   = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32,  64,  kernel_size=7,  padding=3)
        self.bn2   = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64,  128, kernel_size=5,  padding=2)
        self.bn3   = nn.BatchNorm1d(128)

        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(0.3)

        self.fc1   = nn.Linear(128 * 625, 256)
        self.fc2   = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (B, 32,  2500)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B, 64,  1250)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # (B, 128,  625)

        x = x.view(x.size(0), -1)                        # (B, 80000)
        x = self.drop(F.relu(self.fc1(x)))               # (B, 256)
        return self.fc2(x)                               # (B, 2)
