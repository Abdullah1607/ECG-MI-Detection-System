"""
Explainability — saliency maps and Grad-CAM for both model types.
"""

import torch
import numpy as np

from inference.predict import get_model_1lead, get_model_12lead, DEVICE
from inference.gradcam_1d import GradCAM1D

# Last conv layer name per model
_GRADCAM_LAYER = {
    "1lead":  "conv3",
    "12lead": "conv4",
}


def compute_saliency(signal, model_type: str = "1lead") -> np.ndarray:
    """
    Gradient-based saliency map.

    Parameters
    ----------
    signal     : numpy array (5000,) for 1-lead or (12,5000) for 12-lead
    model_type : "1lead" or "12lead"

    Returns
    -------
    saliency : (5000,) float32, normalised to [0, 1]
    """
    signal = np.asarray(signal, dtype=np.float32)

    if model_type == "1lead":
        model = get_model_1lead()
        x     = torch.tensor(signal).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,5000)
    else:
        model = get_model_12lead()
        x     = torch.tensor(signal).unsqueeze(0).to(DEVICE)               # (1,12,5000)

    x.requires_grad = True
    model.eval()

    output     = model(x)
    pred_class = output.argmax(dim=1).item()
    output[0, pred_class].backward()

    grad = x.grad.abs()  # (1,1,5000) or (1,12,5000)

    if model_type == "12lead":
        saliency = grad.squeeze(0).mean(dim=0).cpu().numpy()  # average across leads -> (5000,)
    else:
        saliency = grad.squeeze().cpu().numpy()               # (5000,)

    mx = saliency.max()
    if mx > 1e-8:
        saliency = saliency / mx
    return saliency.astype(np.float32)


def compute_gradcam(signal, model_type: str = "1lead"):
    """
    Grad-CAM heatmap via GradCAM1D helper.

    Parameters
    ----------
    signal     : numpy array (5000,) for 1-lead or (12,5000) for 12-lead
    model_type : "1lead" or "12lead"

    Returns
    -------
    gradcam    : (5000,) float32, normalised to [0, 1]
    raw_signal : (5000,) float32 — Lead II for display
    """
    signal = np.asarray(signal, dtype=np.float32)

    if model_type == "1lead":
        model      = get_model_1lead()
        x          = torch.tensor(signal).unsqueeze(0).unsqueeze(0).to(DEVICE)
        raw_signal = signal                # (5000,)
    else:
        model      = get_model_12lead()
        x          = torch.tensor(signal).unsqueeze(0).to(DEVICE)
        raw_signal = signal[1]             # Lead II (index 1) for display

    target_layer = _GRADCAM_LAYER[model_type]
    cam          = GradCAM1D(model, target_layer_name=target_layer)
    gradcam      = cam.generate(x, target_class=1)  # (5000,)
    cam.remove_hooks()

    return gradcam.astype(np.float32), raw_signal.astype(np.float32)


def compute_saliency_12lead(signal) -> np.ndarray:
    """
    Per-lead gradient saliency for the 12-lead model.

    Parameters
    ----------
    signal : numpy array (12, 5000) float32

    Returns
    -------
    saliency : (12, 5000) float32, each lead normalised to [0, 1]
    """
    signal = np.asarray(signal, dtype=np.float32)
    model  = get_model_12lead()
    model.eval()

    x = torch.tensor(signal).unsqueeze(0).to(DEVICE)  # (1, 12, 5000)
    x.requires_grad = True

    output     = model(x)
    pred_class = output.argmax(dim=1).item()
    output[0, pred_class].backward()

    grad = x.grad.abs().squeeze(0).cpu().numpy()  # (12, 5000)
    for i in range(12):
        mx = grad[i].max()
        if mx > 1e-8:
            grad[i] = grad[i] / mx
    return grad.astype(np.float32)


def compute_gradcam_12lead(signal) -> np.ndarray:
    """
    Per-lead Grad-CAM for the 12-lead model using input gradients.

    Each lead gets its own activation map derived from the gradient of the
    class-1 logit w.r.t. the input tensor, giving genuinely different
    per-lead importance scores.

    Parameters
    ----------
    signal : numpy array (12, 5000) float32

    Returns
    -------
    gradcam_leads : (12, 5000) float32, each lead independently normalised to [0, 1]
    """
    from scipy.ndimage import uniform_filter1d

    signal = np.asarray(signal, dtype=np.float32)
    model  = get_model_12lead()
    model.eval()

    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    output = model(x)
    model.zero_grad()
    output[0, 1].backward()

    # (1, 12, 5000) → (12, 5000)
    input_grads = x.grad.abs().squeeze(0).cpu().numpy()

    gradcam_leads = np.zeros((12, 5000), dtype=np.float32)
    for i in range(12):
        lead_grad = input_grads[i]
        lead_min  = lead_grad.min()
        lead_max  = lead_grad.max()
        if lead_max > lead_min:
            gradcam_leads[i] = (lead_grad - lead_min) / (lead_max - lead_min)
        else:
            gradcam_leads[i] = lead_grad

    # Smooth out spike noise with a 50-sample rolling average
    for i in range(12):
        gradcam_leads[i] = uniform_filter1d(gradcam_leads[i], size=50)

    return gradcam_leads.astype(np.float32)
