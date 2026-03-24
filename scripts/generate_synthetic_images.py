import os
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import cv2
import tempfile
from pathlib import Path

INPUT_BASE = r"D:\Major project\ECG_MI_Project\data\processed_1lead_500hz"
OUTPUT_BASE = r"D:\Major project\ECG_MI_Project\data\processed_1lead_image"
SPLITS = ["train", "val", "test"]
CLASSES = ["normal", "abnormal"]

def render_ecg_as_image(signal, add_noise=True):
    """
    Render a Lead II signal as a realistic ECG paper image.
    Returns: path to temporary PNG file (caller must delete)
    """
    # Use Figure directly (not plt.subplots) to avoid pyplot global state accumulation
    fig = Figure(figsize=(11, 2.5), dpi=72)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')

    # ECG paper grid
    ax.set_xlim(0, len(signal))
    y_range = max(abs(signal.min()), abs(signal.max())) * 1.3 + 0.5
    ax.set_ylim(-y_range, y_range)

    # Minor grid (1mm = small squares)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.grid(which='minor', color='#F4A0A0', alpha=0.6, linewidth=0.4)

    # Major grid (5mm = large squares)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.grid(which='major', color='#E06060', alpha=0.7, linewidth=0.8)

    # Add noise to signal for realism
    if add_noise:
        noise_level = np.random.uniform(0.01, 0.06)
        noisy = signal + np.random.normal(0, noise_level, len(signal))
    else:
        noisy = signal

    # Plot signal
    ax.plot(np.arange(len(signal)), noisy,
            color='black', linewidth=0.9, alpha=0.95)

    # Remove axes decorations
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # Save to temp file via canvas
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    canvas.print_figure(tmp_path, dpi=72, bbox_inches='tight',
                        facecolor='white', format='png')
    fig.clf()
    del canvas, fig

    if add_noise:
        # Apply rotation and blur using cv2
        angle = np.random.uniform(-1, 1)
        img = cv2.imread(tmp_path)
        os.unlink(tmp_path)

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

        # Slight blur to simulate scan
        if np.random.random() > 0.5:
            img = cv2.GaussianBlur(img, (3, 3), 0.5)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
            tmp2_path = tmp2.name
        cv2.imwrite(tmp2_path, img)
        return tmp2_path
    else:
        return tmp_path


def process_file(npy_path, out_path):
    """Load signal, render as image, extract back, save"""
    from inference.ecg_image_extractor import extract_lead_ii_from_image

    signal = np.load(npy_path)

    # Render as image
    img_path = render_ecg_as_image(signal, add_noise=True)

    try:
        # Extract signal from image
        extracted = extract_lead_ii_from_image(img_path)
        np.save(out_path, extracted)
        return True
    except Exception as e:
        # If extraction fails, skip this file
        return False
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


def main():
    import sys
    sys.path.insert(0, '.')

    total = 0
    success = 0

    for split in SPLITS:
        for cls in CLASSES:
            in_dir = os.path.join(INPUT_BASE, split, cls)
            out_dir = os.path.join(OUTPUT_BASE, split, cls)
            os.makedirs(out_dir, exist_ok=True)

            files = [f for f in os.listdir(in_dir) if f.endswith('.npy')]
            print(f"\n{split}/{cls}: {len(files)} files")

            for i, fname in enumerate(files):
                in_path = os.path.join(in_dir, fname)
                out_path = os.path.join(out_dir, fname)

                if os.path.exists(out_path):
                    success += 1
                    total += 1
                    continue

                ok = process_file(in_path, out_path)
                total += 1
                if ok:
                    success += 1

                if (i + 1) % 10 == 0:
                    gc.collect()

                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(files)} done ({success}/{total} success)", flush=True)

    print(f"\nDone! {success}/{total} files processed successfully")
    print(f"Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
