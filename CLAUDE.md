# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Strategy

Refer to PROJECT_GUIDE.md for:
- Workflow rules
- What to avoid
- Final phase execution strategy

## Running the Application

Two processes must run simultaneously. **Always use `venv/Scripts/python` or activate the venv first.**

**Start the FastAPI backend (must come first):**
```bash
cd "D:/Major project/ECG_MI_Project"
venv/Scripts/python -m uvicorn backend.main:app --reload
```

**Start the Streamlit frontend (second terminal):**
```bash
cd "D:/Major project/ECG_MI_Project"
venv/Scripts/python -m streamlit run app.py
```

Frontend connects to `http://127.0.0.1:8000`. Backend changes require a restart.

## Training and Evaluation

**All scripts must be run as modules from the project root** (not as `python scripts/foo.py`) to avoid import errors:

```bash
# 1. Preprocess PTB-XL data (run once before training)
venv/Scripts/python scripts/preprocess.py        # 1-lead: records500 → data/processed_1lead_500hz/
venv/Scripts/python scripts/preprocess_12lead.py # 12-lead: records500 → data/processed_12lead_full/

# 2. Train models
venv/Scripts/python -m training.train_1lead_500hz  # → model/ecg_cnn_500hz.pth
venv/Scripts/python -m training.train_12lead       # → model/ecg_cnn_12lead.pth

# 3. Find optimal thresholds (run after training, update inference/predict.py)
venv/Scripts/python -m evaluation.find_threshold        # 1-lead (uses val split)
venv/Scripts/python -m evaluation.find_threshold_12lead # 12-lead (uses val split)

# 4. Final test-set evaluation (unbiased metrics for reporting)
venv/Scripts/python -m evaluation.eval_test
```

Threshold scripts print four options (best F1, Youden's J, sensitivity >= 0.90, sensitivity >= 0.95). Update `THRESHOLD_1LEAD` / `THRESHOLD_12LEAD` in `inference/predict.py`. Then run `eval_test.py` to confirm test-set performance.

## Ad-hoc Smoke Tests

Manual tests in the project root (not pytest):
```bash
venv/Scripts/python test_loader.py      # ECG file loading
venv/Scripts/python test_predict.py     # End-to-end inference
venv/Scripts/python test_explain.py     # Saliency/GradCAM
venv/Scripts/python test_explanation.py # Explanation text (no data file needed)
venv/Scripts/python test_dataset.py     # Dataset class
```

## Architecture

### Request Flow

```
User (browser)
  └─► app.py  (Streamlit UI)
        ├─► /analyze        POST multipart  → digital file (.csv or .dat+.hea)
        └─► /analyze_signal POST JSON       → pre-extracted signal array (from image)

backend/main.py  (FastAPI)
  ├─ load_ecg()          → Z-scored Lead II signal (5000,) — resamples to 500Hz if needed
  ├─ preprocess_signal() → called for /analyze_signal (image-extracted signals need Z-scoring)
  ├─ predict_ecg()       → prob, label, confidence
  ├─ compute_saliency()  → gradient-based saliency map (5000,)
  ├─ compute_gradcam()   → Grad-CAM map (5000,)
  └─ response JSON: prediction, probabilities, saliency, gradcam, ecg_signal
```

### Signal Preprocessing Contract

All signals must be **Z-score normalised and exactly 5000 samples** before reaching the model:
- `load_ecg()` handles this for digital uploads (`/analyze`). If the uploaded file is not 500 Hz, `ecg_loader.py` **resamples** it to 500 Hz using `scipy.signal.resample` before normalising.
- `preprocess_signal()` is called explicitly in `/analyze_signal` for image-extracted signals.
- **Neither `predict_ecg()` nor `compute_saliency()` normalise their input** — preprocessing must happen upstream.

### Models

### Models

| Model | Architecture file | Weights | Input shape | Val AUC | Test AUC | Sensitivity | Specificity | F1 | Training data |
|---|---|---|---|---|---|---|---|---|---|
| 1-lead (Lead II) | `model/ecg_cnn_500hz.py` | `model/ecg_cnn_500hz.pth` | `(1, 5000)` | 0.894 | 0.8997 | 0.9217 | 0.5876 | 0.8215 | PTB-XL `records500` (~15k records) |
| 12-lead | `model/ecg_cnn_12lead.py` | `model/ecg_cnn_12lead.pth` | `(12, 5000)` | 0.918 | 0.9172 | 0.9071 | 0.6801 | 0.841 | PTB-XL `records500` (~15k records) |


Both output logits for 2 classes (Normal=0, Abnormal=1); `softmax[:, 1]` gives `prob_abnormal`.

Current thresholds in `inference/predict.py` (selected for sensitivity >= 0.90 on validation set):

- `THRESHOLD_1LEAD = 0.24`
  - Test Sensitivity: 0.9217
  - Test Specificity: 0.5876

- `THRESHOLD_12LEAD = 0.28`
  - Test Sensitivity: 0.9071
  - Test Specificity: 0.6801

Run `venv/Scripts/python -m evaluation.eval_test` for final unbiased test-set metrics.

**Preserved legacy files** (not used in inference):
- `model/ecg_cnn.py` / `model/ecg_cnn_k15.pth` — old 1-lead model trained on 100 Hz zero-padded data (AUC 0.789)
- `data/processed/` — old 100 Hz preprocessed data
- `data/processed_12lead/` — old capped 12-lead data (2500/class)
- `training/train.py` — old 1-lead training script

### Model Architectures

Both models use the same kernel sizes scaled for 500 Hz temporal coverage:
- `conv1`: k=75 (150 ms — P-wave)
- `conv2`: k=35 (70 ms — QRS)
- `conv3`: k=25 (50 ms — ST/fine features)
- `conv4`: k=13 (26 ms — sub-QRS)
- 4× MaxPool(2): 5000 → 312; FC: 128×312 → 256 → 2; Dropout 0.4

`ECGCNN500Hz` (`model/ecg_cnn_500hz.py`): 1 input channel.
`ECGCNN12Lead` (`model/ecg_cnn_12lead.py`): 12 input channels (one per lead).

### Image Input Path

`inference/ecg_image_extractor.py` extracts a Lead II signal from an ECG image:
1. Convert PDF → PNG via poppler (path hardcoded to `poppler/poppler-25.12.0/Library/bin`)
2. Remove pink/red ECG grid lines via HSV masking
3. Detect lead row bands via horizontal projection profiling
4. **Orientation-aware Lead II extraction:**
   - **Portrait** (`h > w × 1.3`): each row is one lead → use `regions[1]` full width
   - **Landscape** (`w > h × 1.3`): 4×3 grid → if 4+ regions, use last (rhythm strip); else crop `regions[1]` to first quarter (`w // 4` columns)
5. Resample to 5000 samples via `scipy.signal.resample`

The extracted signal goes to `/analyze_signal` where it is Z-score normalised before inference.
For the 12-lead model with image input, Lead II is tiled across all 12 channels (`np.tile`). This is an approximation.

### Explainability

Two methods in `inference/explain.py`, always computed on the 1-lead signal:
- **Saliency**: vanilla gradient (`x.grad.abs()`) — fast, noisy
- **Grad-CAM**: hooks on `conv4` (both 1-lead and 12-lead) via `inference/gradcam_1d.py`

Both return `(5000,)` arrays normalised to `[0, 1]`. In `app.py`, used **only for colour-coding** waveform plots; Y-axis always uses `raw_ecg` (the `ecg_signal` field from the backend response).

### Explanation Text and Report Pipeline

```
generate_explanation()    (inference/explanation_text.py)
  → maps saliency onto named ECG segments (P, QRS, ST, T, TP)
  → returns user_text, clinical_text, confidence_text

generate_report()         (inference/report.py)
  → wraps result into a structured dict (sampling_rate_hz: 500)

generate_pdf_report()     (inference/ecg_pdf_report.py)
  → ReportLab Platypus PDF; embeds the saliency heatmap matplotlib figure
```

### Data Augmentation (training only)

`dataset/ecg_dataset.py` `ECGDataset(augment=True)` applies per-sample:
1. Gaussian noise (50% chance, σ = 0.02 × signal std)
2. Amplitude scaling (60% chance, ×Uniform(0.8, 1.2))
3. Baseline wander (40% chance, low-freq sinusoid)
4. Time shift (40% chance, ±200 samples via `np.roll`)
5. Polarity flip (20% chance)

`ECGDataset12Lead` in `train_12lead.py` applies same augmentations per-lead independently. Always pass `augment=False` for val/test.

## Key Data Paths

```
data/ptbxl/                    Raw PTB-XL metadata + records100/ + records500/
data/processed_1lead_500hz/    Active 1-lead preprocessed data (500Hz, ~15k records)
data/processed_12lead_full/    Active 12-lead preprocessed data (500Hz, ~15k records)
model/ecg_cnn_500hz.pth        Active 1-lead model weights
model/ecg_cnn_12lead.pth       Active 12-lead model weights
poppler/                       PDF-to-image converter (Windows binary)
```

## Windows-Specific Notes

- Always use `venv/Scripts/python` (not system Python) — packages are installed in the venv.
- 12-lead `DataLoader` must use `num_workers=0` on Windows to avoid shared memory errors (error code 1455).
- Unicode characters (`≥`, `—`) in print statements cause `UnicodeEncodeError` in the Windows terminal (cp1252). Use ASCII equivalents (`>=`, `--`) in scripts that print to stdout.
- Run training scripts with `python -m training.train_xxx` from project root, not `python training/train_xxx.py`, to resolve module imports.

## Abnormal Class Definition

Both preprocessing scripts label a record as **abnormal** if any SCP code maps to diagnostic class `{"MI", "STTC", "CD", "HYP"}`. This is broad cardiac abnormality detection, not MI-only.
