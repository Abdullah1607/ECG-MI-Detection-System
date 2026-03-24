import os
import cv2
import numpy as np
from PIL import Image
from scipy.signal import resample, butter, filtfilt
from scipy.ndimage import uniform_filter1d

TARGET_LENGTH = 5000


# -------------------------------
# Load Image
# -------------------------------
def load_image(path):
    return Image.open(path).convert("RGB")


# -------------------------------
# Preprocess
# -------------------------------
def preprocess(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Resize (optional)
    if gray.shape[1] > 3000:
        scale = 3000 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale)

    # Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )

    return gray, binary


# -------------------------------
# Remove ECG grid lines
# -------------------------------
def remove_grid(binary):
    # Only remove horizontal grid lines, NOT vertical
    # Vertical removal destroys QRS spikes
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    cleaned = cv2.subtract(binary, horizontal_lines)
    return cleaned


# -------------------------------
# Find Rhythm Strip (robust)
# -------------------------------
def find_rhythm_strip(binary):
    h, w = binary.shape

    # Search 58%-76%: rhythm strip lives here; footer text is below 77%
    search_start = int(h * 0.58)
    search_end   = int(h * 0.76)
    region = binary[search_start:search_end, :]

    row_density = region.sum(axis=1).astype(np.float32)
    # Smooth to merge closely-spaced trace rows
    kernel = np.ones(10) / 10
    smoothed = np.convolve(row_density, kernel, mode='same')
    threshold = smoothed.max() * 0.20

    active = np.where(smoothed > threshold)[0]

    if len(active) < 5:
        # fallback: middle of search window
        r1 = search_start + (search_end - search_start) // 3
        r2 = search_start + (search_end - search_start) * 2 // 3
    else:
        # Find contiguous segments; take the LAST (lowest = rhythm strip)
        breaks = np.where(np.diff(active) > 15)[0]
        seg_starts = np.append(active[0], active[breaks + 1])
        seg_ends   = np.append(active[breaks], active[-1])
        r1 = search_start + int(seg_starts[-1])
        r2 = search_start + int(seg_ends[-1])
        pad = max(20, int((r2 - r1) * 0.20))
        r1 = max(0, r1 - pad)
        r2 = min(h, r2 + pad)

    return r1, r2, int(w * 0.08), int(w * 0.93)


# -------------------------------
# Extract waveform (robust)
# -------------------------------
def extract_waveform(binary_strip, gray_strip):
    h, w = binary_strip.shape
    signal = np.full(w, np.nan)
    inverted = 255 - gray_strip.astype(np.float64)

    for col in range(w):
        rows = np.where(binary_strip[:, col] > 0)[0]
        if len(rows) == 0:
            continue
        # Weight by darkness — trace is darker than grid remnants
        weights = inverted[rows, col]
        if weights.sum() > 0:
            signal[col] = np.average(rows, weights=weights)
        else:
            signal[col] = np.median(rows)

    x = np.arange(w)
    valid = ~np.isnan(signal)
    if valid.any():
        signal[~valid] = np.interp(x[~valid], x[valid], signal[valid])

    signal = h - signal
    return signal


# -------------------------------
# Bandpass Filter (PTB-XL match)
# -------------------------------
def bandpass(signal, fs=500):
    b, a = butter(4, [0.5/(fs/2), 40/(fs/2)], btype='band')
    return filtfilt(b, a, signal)


# -------------------------------
# Normalize
# -------------------------------
def normalize(signal):
    return (signal - signal.mean()) / (signal.std() + 1e-8)


# -------------------------------
# Clean Signal (safe)
# -------------------------------
def clean(signal):
    # Light smoothing (don’t destroy QRS)
    signal = uniform_filter1d(signal, size=3)

    # Clip outliers
    mean, std = signal.mean(), signal.std()
    signal = np.clip(signal, mean - 3*std, mean + 3*std)

    return signal


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def extract_lead_ii_from_image(image_path):
    img = load_image(image_path)

    # Rotate if portrait
    if img.width < img.height:
        img = img.rotate(90, expand=True)

    gray, binary = preprocess(img)

    # 🔥 NEW: remove grid
    binary = remove_grid(binary)

    # Find strip
    r1, r2, c1, c2 = find_rhythm_strip(binary)

    strip = binary[r1:r2, c1:c2]

    # Extract waveform
    gray_strip = gray[r1:r2, c1:c2]
    signal = extract_waveform(strip, gray_strip)

    # Clean
    signal = clean(signal)

    # Resample
    signal = resample(signal, TARGET_LENGTH)

    # Filter
    signal = bandpass(signal)

    # Normalize
    signal = normalize(signal)
    
   
    return signal.astype(np.float32)