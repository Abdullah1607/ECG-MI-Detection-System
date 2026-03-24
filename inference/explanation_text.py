"""
Rule-based textual explanation for ECG predictions.
"""


_USER_NORMAL = (
    "Your ECG signal does not show strong indicators of myocardial infarction "
    "or significant ischemic changes based on the AI analysis. "
    "While this result is reassuring, it is not a substitute for clinical evaluation. "
    "If you have symptoms such as chest pain, shortness of breath, or palpitations, "
    "please seek immediate medical attention."
)

_USER_ABNORMAL_MOD = (
    "The AI analysis has detected patterns in your ECG that may be associated with "
    "cardiac abnormalities, including possible ischemic or myocardial injury. "
    "Borderline findings were identified that warrant clinical review. "
    "Please consult a healthcare professional as soon as possible."
)

_USER_ABNORMAL_HIGH = (
    "The AI analysis has detected strong indicators of cardiac abnormality in your ECG. "
    "These patterns may be consistent with myocardial infarction (heart attack) or "
    "severe ischemic injury. "
    "This is a high-priority finding — please seek urgent medical evaluation immediately."
)

_CLINICAL_NORMAL = (
    "AI model output: Normal classification. "
    "No significant ST-segment deviations, pathological Q-waves, or T-wave inversions "
    "were identified as the dominant drivers of this classification. "
    "Standard follow-up per clinical protocol is recommended."
)

_CLINICAL_ABNORMAL = (
    "AI model output: Abnormal classification. "
    "The model's attention (via Grad-CAM) is focused on features consistent with "
    "ischemic injury patterns — potential ST-segment deviation, T-wave changes, "
    "or morphological QRS abnormalities. "
    "Differential includes STEMI, NSTEMI, unstable angina, or non-ischemic "
    "cardiomyopathy. Clinical correlation with patient history, symptoms, "
    "troponin levels, and repeat ECG is strongly advised."
)

_CLINICAL_12LEAD_NOTE = (
    " The 12-lead model incorporates spatial information from all lead vectors, "
    "providing higher specificity for localizing territory of injury."
)

_CLINICAL_1LEAD_NOTE = (
    " The 1-lead (Lead II) model was used — single-lead analysis has reduced "
    "sensitivity for detecting lateral or posterior territory events."
)


def generate_explanation(label: str, prob_abnormal: float, model_type: str = "1lead"):
    """
    Generate layered textual explanation.

    Parameters
    ----------
    label         : "Normal" or "Abnormal"
    prob_abnormal : float in [0, 1]
    model_type    : "1lead" or "12lead"

    Returns
    -------
    user_text       : patient-friendly explanation
    clinical_text   : clinician-facing summary
    confidence_text : confidence disclaimer
    """

    # --- Patient text ---
    if label == "Normal":
        user_text = _USER_NORMAL
    elif prob_abnormal >= 0.70:
        user_text = _USER_ABNORMAL_HIGH
    else:
        user_text = _USER_ABNORMAL_MOD

    # --- Clinical text ---
    clinical_text = _CLINICAL_ABNORMAL if label == "Abnormal" else _CLINICAL_NORMAL
    if model_type == "12lead":
        clinical_text += _CLINICAL_12LEAD_NOTE
    else:
        clinical_text += _CLINICAL_1LEAD_NOTE

    # --- Confidence text ---
    conf = prob_abnormal if label == "Abnormal" else (1.0 - prob_abnormal)
    if conf >= 0.90:
        conf_level = "very high"
    elif conf >= 0.75:
        conf_level = "high"
    elif conf >= 0.60:
        conf_level = "moderate"
    else:
        conf_level = "low"

    model_label = "12-lead" if model_type == "12lead" else "1-lead (Lead II)"
    confidence_text = (
        f"Model confidence: {conf_level} ({conf * 100:.1f}%). "
        f"Model: {model_label} CNN trained on PTB-XL (500 Hz). "
        "This AI prediction is for research and educational use only. "
        "A qualified clinician must validate all findings."
    )

    return user_text, clinical_text, confidence_text
