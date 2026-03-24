import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model.ecg_cnn import ECGCNN


# -------- CONFIG --------
MODEL_PATH = "model/ecg_cnn_k15.pth"
SAMPLE_PATH = "data/processed/test/abnormal"  # folder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------


def load_sample(sample_dir):
    files = [f for f in os.listdir(sample_dir) if f.endswith(".npy")]
    path = os.path.join(sample_dir, files[0])
    signal = np.load(path)
    return signal, path


def compute_saliency(model, signal):
    model.eval()

    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = x.to(DEVICE)
    x.requires_grad = True

    output = model(x)
    pred_class = output.argmax(dim=1)

    score = output[0, pred_class]
    score.backward()

    saliency = x.grad.abs().squeeze().cpu().numpy()
    return saliency, pred_class.item()


def plot_saliency(signal, saliency, title):
    saliency = saliency / (saliency.max() + 1e-8)

    plt.figure(figsize=(12, 4))
    plt.plot(signal, label="ECG", alpha=0.7)
    plt.plot(saliency * np.max(signal), label="Saliency", color="red", alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    import os

    model = ECGCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    signal, path = load_sample(SAMPLE_PATH)
    saliency, pred = compute_saliency(model, signal)

    label = "Abnormal" if pred == 1 else "Normal"
    print("Prediction:", label)
    print("Sample:", path)

    plot_saliency(signal, saliency, f"Saliency Map ({label})")


if __name__ == "__main__":
    main()