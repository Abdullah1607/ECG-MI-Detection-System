import torch.nn as nn
import torch.nn.functional as F


class ECGCNNImage(nn.Module):
    """
    1D-CNN for single-lead ECG classification at 500 Hz.
    Used for the image-extracted signal model (ecg_cnn_image.pth).

    Input:  (batch, 1, 5000)
    Output: (batch, 2)  — logits for [Normal, Abnormal]

    Kernel sizes chosen for 500 Hz temporal coverage:
        conv1: k=75  (150 ms — P-wave)
        conv2: k=35  (70 ms  — QRS)
        conv3: k=25  (50 ms  — ST/fine features)
        conv4: k=13  (26 ms  — sub-QRS)

    Feature map sizes after each MaxPool(2):
        conv1: (B, 32,  2500)
        conv2: (B, 64,  1250)
        conv3: (B, 128,  625)
        conv4: (B, 128,  312)
        FC in:  128 * 312 = 39 936
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1,   32,  kernel_size=75, padding=37)
        self.bn1   = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32,  64,  kernel_size=35, padding=17)
        self.bn2   = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64,  128, kernel_size=25, padding=12)
        self.bn3   = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=13, padding=6)
        self.bn4   = nn.BatchNorm1d(128)

        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(0.4)

        self.fc1   = nn.Linear(128 * 312, 256)
        self.fc2   = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (B, 32,  2500)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B, 64,  1250)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # (B, 128,  625)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # (B, 128,  312)

        x = x.view(x.size(0), -1)                        # (B, 39936)
        x = self.drop(F.relu(self.fc1(x)))               # (B, 256)
        return self.fc2(x)                               # (B, 2)
