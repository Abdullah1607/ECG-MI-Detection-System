from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import numpy as np

from inference.ecg_loader import load_ecg, preprocess_signal
from inference.ecg_image_extractor import extract_lead_ii_from_image
from inference.predict import predict_ecg
from inference.explain import compute_saliency, compute_gradcam, compute_gradcam_12lead, compute_saliency_12lead

app = FastAPI(
    title="ECG Abnormality Detection API",
    description="1-lead and 12-lead 1D-CNN inference with saliency and Grad-CAM.",
    version="2.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "models": ["1lead", "12lead"]}


# --------------------------------------------------
# Digital file upload: .csv / .dat + .hea
# --------------------------------------------------
@app.post("/analyze")
async def analyze_ecg(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    temp_dir   = tempfile.mkdtemp()
    file_paths = []

    try:
        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in (".csv", ".dat", ".hea"):
                raise HTTPException(status_code=400,
                                    detail=f"Unsupported file: {file.filename}")
            path = os.path.join(temp_dir, file.filename)
            with open(path, "wb") as f:
                f.write(await file.read())
            file_paths.append(path)

        signal, model_type = load_ecg(file_paths)

        prob, label, confidence = predict_ecg(signal, model_type)
        saliency                = compute_saliency(signal, model_type)
        gradcam, raw_signal     = compute_gradcam(signal, model_type)

        response_body = {
            "prediction":           label,
            "abnormal_probability": round(float(prob), 4),
            "confidence":           round(float(confidence), 4),
            "model_used":           model_type,
            "saliency":             saliency.tolist(),
            "gradcam":              gradcam.tolist(),
            "signal_12lead":        None,
            "gradcam_12lead":       None,
            "signal_1lead":         None,
        }

        if model_type == "12lead":
            _LEAD_NAMES = ['I','II','III','aVR','aVL','aVF',
                           'V1','V2','V3','V4','V5','V6']
            gradcam_12lead              = compute_gradcam_12lead(signal)
            saliency_12lead             = compute_saliency_12lead(signal)
            lead_scores                 = gradcam_12lead.mean(axis=1)
            best_idx                    = int(lead_scores.argmax())
            response_body["signal_12lead"]        = signal.tolist()
            response_body["gradcam_12lead"]       = gradcam_12lead.tolist()
            response_body["saliency_12lead"]      = saliency_12lead.tolist()
            response_body["best_lead_name"]       = _LEAD_NAMES[best_idx]
            response_body["best_lead_activation"] = float(lead_scores[best_idx])
        else:
            response_body["signal_1lead"] = signal.tolist()

        return JSONResponse(response_body)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts a PNG, JPG, or PDF ECG image.
    Extracts Lead II, runs image-domain CNN inference, and returns results.
    """
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        signal = extract_lead_ii_from_image(tmp_path)
        os.unlink(tmp_path)

        signal_proc = preprocess_signal(signal)
        prob, label, confidence = predict_ecg(signal_proc, model_type="image")
        saliency = compute_saliency(signal_proc, "1lead")
        gradcam, _ = compute_gradcam(signal_proc, "1lead")
        return JSONResponse({
            "status":               "ok",
            "prediction":           label,
            "abnormal_probability": round(float(prob), 4),
            "confidence":           round(float(confidence), 4),
            "model_used":           "image",
            "saliency":             saliency.tolist(),
            "gradcam":              gradcam.tolist(),
            "signal_1lead":         signal_proc.tolist(),
            "extracted_signal":     signal_proc.tolist(),
            "signal_12lead":        None,
            "gradcam_12lead":       None,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
