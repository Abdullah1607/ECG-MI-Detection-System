"""
Grad-CAM for 1D CNN ECG models.

Works by hooking into the last Conv1D layer, capturing:
  - Forward activations  (shape: batch × channels × time)
  - Backward gradients   (shape: batch × channels × time)

The CAM is: ReLU( sum_c( weight_c * activation_c ) )
where weight_c = global average of gradient over time dimension.

Usage
-----
    cam = GradCAM1D(model, target_layer_name="conv_layers.4")
    heatmap = cam.generate(signal_tensor, target_class=1)
    cam.remove_hooks()
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping for 1D CNNs.

    Parameters
    ----------
    model             : trained PyTorch model (eval mode expected)
    target_layer_name : dot-separated attribute path to the target Conv1d layer
                        e.g. "features.2"  or  "conv_blocks.2.conv"
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks = []

        # Resolve nested attribute path  e.g. "conv_blocks.2.conv"
        layer = self._get_layer(model, target_layer_name)
        self._register_hooks(layer)

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------
    def _register_hooks(self, layer: torch.nn.Module):
        def forward_hook(module, input, output):
            # output: (batch, channels, time)
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: (batch, channels, time)
            self._gradients = grad_output[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Call this when done to avoid memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Attribute path resolver
    # ------------------------------------------------------------------
    @staticmethod
    def _get_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
        parts = name.split(".")
        layer = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    # ------------------------------------------------------------------
    # CAM generation
    # ------------------------------------------------------------------
    def generate(
        self,
        signal: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single ECG signal.

        Parameters
        ----------
        signal       : tensor of shape (1, 1, T) or (1, C, T)
                       batch size must be 1
        target_class : class index to explain (0=normal, 1=abnormal)
                       if None, uses the predicted class
        normalize    : if True, output is min-max normalized to [0, 1]

        Returns
        -------
        heatmap : np.ndarray of shape (T,) — same length as input signal
        """
        assert signal.shape[0] == 1, "Batch size must be 1 for Grad-CAM"

        signal = signal.requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        logits = self.model(signal)              # (1, num_classes) or (1, 1)

        # Handle both sigmoid (binary) and softmax (multi-class) outputs
        if logits.shape[-1] == 1:
            # Binary with sigmoid
            probs = torch.sigmoid(logits)
            if target_class is None:
                target_class = int((probs > 0.5).item())
            score = probs[0, 0] if target_class == 1 else 1 - probs[0, 0]
        else:
            # Multi-class with softmax
            probs = torch.softmax(logits, dim=-1)
            if target_class is None:
                target_class = int(torch.argmax(probs, dim=-1).item())
            score = probs[0, target_class]

        # Backward pass
        score.backward()

        # Activations: (1, C, T)  →  (C, T)
        activations = self._activations[0]   # (C, T)
        gradients   = self._gradients[0]     # (C, T)

        # Global average pool gradients over time  →  (C,)
        weights = gradients.mean(dim=-1)

        # Weighted sum of activation maps  →  (T,)
        cam = torch.einsum("c,ct->t", weights, activations)

        # ReLU — keep only positive contributions toward the target class
        cam = F.relu(cam)

        # Upsample to original signal length
        signal_length = signal.shape[-1]
        cam_upsampled = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),    # (1, 1, T_conv)
            size=signal_length,
            mode="linear",
            align_corners=False,
        ).squeeze().cpu().numpy()             # (signal_length,)

        if normalize:
            cam_min = cam_upsampled.min()
            cam_max = cam_upsampled.max()
            if cam_max - cam_min > 1e-8:
                cam_upsampled = (cam_upsampled - cam_min) / (cam_max - cam_min)
            else:
                cam_upsampled = np.zeros_like(cam_upsampled)

        return cam_upsampled

    # ------------------------------------------------------------------
    # Convenience: generate + overlay on raw signal
    # ------------------------------------------------------------------
    def generate_overlay(
        self,
        signal: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns both the raw signal (numpy) and the Grad-CAM heatmap.
        Useful for feeding directly into visualization.
        """
        heatmap = self.generate(signal, target_class=target_class)
        raw     = signal.squeeze().detach().cpu().numpy()
        return raw, heatmap


# ------------------------------------------------------------------
# Smoothed Grad-CAM (reduces noise in saliency)
# ------------------------------------------------------------------
class SmoothGradCAM1D(GradCAM1D):
    """
    Averages Grad-CAM over N passes with Gaussian noise injected into input.
    Reduces gradient noise for more stable saliency maps.

    Parameters
    ----------
    n_samples   : number of noisy forward/backward passes (default: 20)
    noise_level : std of Gaussian noise relative to signal std (default: 0.15)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer_name: str,
        n_samples: int = 20,
        noise_level: float = 0.15,
    ):
        super().__init__(model, target_layer_name)
        self.n_samples   = n_samples
        self.noise_level = noise_level

    def generate(
        self,
        signal: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:

        signal_std = signal.std().item()
        noise_std  = self.noise_level * signal_std

        accumulated = np.zeros(signal.shape[-1], dtype=np.float32)

        for _ in range(self.n_samples):
            noise         = torch.randn_like(signal) * noise_std
            noisy_signal  = (signal + noise).detach()
            cam           = super().generate(noisy_signal, target_class, normalize=False)
            accumulated  += cam

        averaged = accumulated / self.n_samples

        if normalize:
            mn, mx = averaged.min(), averaged.max()
            if mx - mn > 1e-8:
                averaged = (averaged - mn) / (mx - mn)
            else:
                averaged = np.zeros_like(averaged)

        return averaged


# ------------------------------------------------------------------
# Visualization helper
# ------------------------------------------------------------------
def plot_gradcam_overlay(
    signal_np: np.ndarray,
    heatmap_np: np.ndarray,
    sampling_rate: int = 500,
    title: str = "Grad-CAM ECG Overlay",
    figsize: tuple = (14, 4),
    colormap: str = "jet",
    ax=None,
    segment_labels: dict = None,
):
    """
    Plot ECG signal with Grad-CAM heatmap overlaid as a colored background.

    Parameters
    ----------
    signal_np     : raw ECG signal (1D numpy array)
    heatmap_np    : Grad-CAM values (1D numpy array, same length)
    sampling_rate : samples per second (for x-axis in seconds)
    ax            : optional existing matplotlib Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection

    time_axis = np.arange(len(signal_np)) / sampling_rate

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Build colored line segments from heatmap
    points  = np.array([time_axis, signal_np]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap  = plt.get_cmap(colormap)
    norm  = mcolors.Normalize(vmin=0.0, vmax=1.0)
    lc    = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, alpha=0.9)
    lc.set_array(heatmap_np[:-1])
    ax.add_collection(lc)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Grad-CAM Importance", pad=0.01)

    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_ylim(signal_np.min() - 0.1, signal_np.max() + 0.1)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title)

    # ECG segment markers — override via segment_labels parameter if needed
    if segment_labels is None:
        segment_labels = {
            "P":   (0.5, 1.0),
            "QRS": (1.2, 2.2),
            "ST":  (2.2, 3.3),
            "T":   (3.3, 4.3),
        }
    for label, (xmin, xmax) in segment_labels.items():
        if xmax <= time_axis[-1]:
            ax.axvspan(xmin, xmax, alpha=0.06, color="gray")
            ax.text((xmin + xmax) / 2, signal_np.max() + 0.05, label,
                    ha="center", fontsize=7, color="dimgray")

    return fig, ax
