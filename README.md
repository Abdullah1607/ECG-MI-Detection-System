# ECG Cardiac Abnormality Detection

A deep learning system for detecting cardiac abnormalities from ECG signals using 1-lead, 12-lead, and image-extracted input pipelines. Built on the PTB-XL dataset with a FastAPI backend and Streamlit frontend.

> **Disclaimer:** This tool is for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis. Always consult a qualified healthcare professional for medical decisions.

---

## System Requirements

- Python 3.10 or higher
- CUDA-capable GPU recommended (NVIDIA, 4 GB+ VRAM) — CPU works but training is slow
- 8 GB RAM minimum (16 GB recommended for 12-lead training)
- ~20 GB free disk space (PTB-XL dataset + preprocessed data)
- Windows 10/11 or Linux (Windows paths used in scripts — adjust if on Linux)
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) for PDF input (Windows binary included at `poppler/`)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Abdullah1607/ECG_MI_Project_v2.git
cd ECG_MI_Project_v2

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Download PTB-XL Dataset

1. Register at [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
2. Download and extract to `data/ptbxl/`

Expected structure:
```
data/ptbxl/
    ptbxl_database.csv
    scp_statements.csv
    records100/
    records500/
```

---

## Preprocessing

Run once before training. Scripts must be run as modules from the project root.

```bash
# 1-lead (Lead II) at 500 Hz
venv/Scripts/python scripts/preprocess.py
# Output: data/processed_1lead_500hz/

# 12-lead at 500 Hz
venv/Scripts/python scripts/preprocess_12lead.py
# Output: data/processed_12lead_full/

# Synthetic ECG images for image model (optional)
venv/Scripts/python -m scripts.generate_synthetic_images
# Output: data/processed_1lead_image/
```

---

## Training

```bash
# 1-lead CNN
venv/Scripts/python -m training.train_1lead_500hz
# Output: model/ecg_cnn_500hz.pth

# 12-lead CNN
venv/Scripts/python -m training.train_12lead
# Output: model/ecg_cnn_12lead.pth

# Image-domain CNN (requires generate_synthetic_images first)
venv/Scripts/python -m training.train_image
# Output: model/ecg_cnn_image.pth
```

---

## Threshold Evaluation

After training, find the optimal classification threshold on the validation set:

```bash
venv/Scripts/python -m evaluation.find_threshold         # 1-lead
venv/Scripts/python -m evaluation.find_threshold_12lead  # 12-lead
venv/Scripts/python -m evaluation.find_threshold_image   # image model
```

Each script prints four threshold options (best F1, Youden's J, sensitivity ≥ 0.90, sensitivity ≥ 0.95). Update `THRESHOLD_1LEAD`, `THRESHOLD_12LEAD`, and `THRESHOLD_IMAGE` in `inference/predict.py` with your chosen values.

---

## Final Test-Set Evaluation

```bash
venv/Scripts/python -m evaluation.eval_test
```

Reports unbiased test-set metrics (AUC, sensitivity, specificity, F1) for the 1-lead model.

---

## Running the Application

Two processes must run simultaneously. Start the backend first.

**Terminal 1 — FastAPI backend:**
```bash
cd "D:/Major project/ECG_MI_Project_v2"
venv/Scripts/python -m uvicorn backend.main:app --reload
# Runs on http://127.0.0.1:8000
```

**Terminal 2 — Streamlit frontend:**
```bash
cd "D:/Major project/ECG_MI_Project_v2"
venv/Scripts/python -m streamlit run app.py
# Opens in browser at http://localhost:8501
```

---

## Model Performance

| Model | Input | Val AUC | Test AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| 1-Lead CNN (`ecg_cnn_500hz.pth`) | `.dat+.hea` or `.csv` (Lead II) | 0.894 | 0.900 | 0.922 | 0.588 | 0.822 |
| 12-Lead CNN (`ecg_cnn_12lead.pth`) | `.dat+.hea` or `.csv` (12 leads) | 0.918 | 0.917 | 0.907 | 0.680 | 0.841 |
| Image CNN (`ecg_cnn_image.pth`) | PNG / JPG / PDF scan | 0.808 | — | 0.826 | 0.618 | 0.780 |

Thresholds selected for sensitivity ≥ 0.90 (digital models) and best F1 (image model).

**Abnormal class definition:** MI, STTC (ST/T-wave changes), CD (conduction disturbances), HYP (hypertrophy) — broad cardiac abnormality detection, not MI-only.

---

## Project Structure

```
ECG_MI_Project_v2/
├── app.py                        # Streamlit frontend
├── backend/
│   └── main.py                   # FastAPI backend
├── model/
│   ├── ecg_cnn_500hz.py          # 1-lead CNN architecture
│   ├── ecg_cnn_12lead.py         # 12-lead CNN architecture
│   └── ecg_cnn_1lead.py          # Lightweight 1-lead CNN (image model)
├── inference/
│   ├── predict.py                # Model inference + thresholds
│   ├── explain.py                # Saliency + Grad-CAM
│   ├── gradcam_1d.py             # 1D Grad-CAM implementation
│   ├── ecg_loader.py             # Signal loading + Z-score normalisation
│   ├── ecg_image_extractor.py    # Lead II extraction from ECG images
│   ├── explanation_text.py       # Clinical explanation text generation
│   ├── report.py                 # Structured report generation
│   └── ecg_pdf_report.py         # PDF report generation (ReportLab)
├── training/
│   ├── train_1lead_500hz.py      # 1-lead training script
│   ├── train_12lead.py           # 12-lead training script
│   └── train_image.py            # Image-domain training script
├── evaluation/
│   ├── find_threshold.py         # Threshold search — 1-lead
│   ├── find_threshold_12lead.py  # Threshold search — 12-lead
│   ├── find_threshold_image.py   # Threshold search — image model
│   └── eval_test.py              # Final test-set evaluation
├── dataset/
│   └── ecg_dataset.py            # PyTorch Dataset with augmentation
├── scripts/
│   ├── preprocess.py             # PTB-XL 1-lead preprocessing
│   ├── preprocess_12lead.py      # PTB-XL 12-lead preprocessing
│   └── generate_synthetic_images.py  # Synthetic ECG image generation
├── poppler/                      # PDF-to-image converter (Windows)
└── data/                         # Excluded from git — see Download section
```

---

## Known Limitations

- **Image model is experimental.** Trained on synthetically generated ECG paper images — no large public ECG image dataset with diagnostic labels exists. Performance on real scanned ECGs may differ from synthetic test results (AUC 0.808).
- **Broad abnormality detection.** The model detects MI, STTC, CD, and HYP collectively — it is not a specific MI detector.
- **Lead II only for 1-lead and image paths.** Pathologies primarily visible in other leads may be missed.
- **PTB-XL label quality.** Labels are derived from automated ECG statements and are not independently adjudicated by cardiologists.
- **Windows paths.** Several scripts hardcode Windows-style paths (`D:\Major project\...`). Update `DATA_DIR` / `INPUT_BASE` / `OUTPUT_BASE` variables in preprocessing and training scripts before running on other systems.
- **num_workers=0** required on Windows for the 12-lead DataLoader due to shared memory limitations.

---

## Disclaimer

This software is provided for **research and educational purposes only**. It has not been validated for clinical use, is not a certified medical device, and must not be used to make or influence clinical decisions. The authors accept no liability for any harm resulting from its use. Always seek the advice of a qualified medical professional for any health concerns.
