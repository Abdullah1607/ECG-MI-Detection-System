from datetime import datetime
from inference.disclaimer import CLINICAL_DISCLAIMER


def generate_report(
    label: str,
    prob_abnormal: float,
    user_text: str,
    clinical_text: str,
    confidence_text: str,
    model_type: str = "1lead",
    best_lead_name: str = None,
    best_lead_activation: float = None,
) -> dict:
    """
    Build a structured report dictionary.

    Parameters
    ----------
    label           : "Normal" or "Abnormal"
    prob_abnormal   : float probability of abnormal class
    user_text       : patient-friendly explanation
    clinical_text   : clinician summary
    confidence_text : confidence disclaimer string
    model_type      : "1lead" or "12lead"

    Returns
    -------
    dict with sections: system, model_result, interpretation, signal, model_info, disclaimer
    """
    conf = prob_abnormal if label == "Abnormal" else (1.0 - prob_abnormal)

    if model_type == "12lead":
        signal_section = {
            "length_samples":   5000,
            "duration_seconds": 10.0,
            "sampling_rate_hz": 500,
            "num_leads":        12,
            "lead_names":       ["I", "II", "III", "aVR", "aVL", "aVF",
                                 "V1", "V2", "V3", "V4", "V5", "V6"],
        }
        model_info = {
            "architecture":   "12-Lead 1D-CNN",
            "input_channels": 12,
            "trained_on":     "PTB-XL 500Hz",
            "roc_auc":        0.8991,
            "threshold":      0.43,
        }
    else:
        signal_section = {
            "length_samples":   5000,
            "duration_seconds": 10.0,
            "sampling_rate_hz": 500,
            "num_leads":        1,
            "lead_names":       ["Lead II"],
        }
        model_info = {
            "architecture":   "Lead II 1D-CNN",
            "input_channels": 1,
            "trained_on":     "PTB-XL 500Hz",
            "roc_auc":        0.8528,
            "threshold":      0.48,
        }

    if model_type == "12lead" and best_lead_name:
        viz_type      = "12-Lead Global Heatmap + Per-Lead Grad-CAM"
        activated_lead = best_lead_name
        clinical_note  = (
            f"The model focused most strongly on lead {best_lead_name}, "
            f"suggesting pathological patterns in that lead's territory."
        )
    else:
        viz_type       = "Lead II Grad-CAM + Saliency"
        activated_lead = "Lead II"
        clinical_note  = (
            "This model focuses on temporal regions within Lead II "
            "to detect myocardial infarction-related patterns."
        )

    explainability = {
        "method":                    "Grad-CAM + Saliency",
        "model_type":                model_type,
        "visualization_type":        viz_type,
        "most_activated_lead":       activated_lead,
        "most_activated_lead_score": (
            round(float(best_lead_activation), 4)
            if best_lead_activation is not None else None
        ),
        "clinical_note": clinical_note,
    }

    return {
        "system": {
            "name":                     "ECG Abnormality Detection System",
            "version":                  "2.0",
            "analysis_timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "model_result": {
            "prediction":           label,
            "is_abnormal":          label != "Normal",
            "abnormal_probability": round(prob_abnormal, 4),
            "normal_probability":   round(1.0 - prob_abnormal, 4),
            "probability_percent":  f"{prob_abnormal * 100:.1f}%",
            "confidence":           round(conf, 4),
            "model_type":           model_type,
        },
        "interpretation": {
            "patient_explanation": user_text,
            "clinical_summary":    clinical_text,
            "confidence_note":     confidence_text,
        },
        "signal":         signal_section,
        "model_info":     model_info,
        "explainability": explainability,
        "disclaimer":     CLINICAL_DISCLAIMER,
    }
