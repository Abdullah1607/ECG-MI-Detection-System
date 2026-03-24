"""
ECG Analysis PDF Report Generator
Uses ReportLab Platypus for structured, clinical-grade PDF output.

Accepts the structured report dict from ecg_report.py and
an optional matplotlib Figure of the Grad-CAM overlay.

Usage
-----
    from ecg_pdf_report import generate_pdf_report

    fig, _ = plot_gradcam_overlay(signal_np, heatmap_np)
    pdf_path = generate_pdf_report(report_dict, gradcam_fig=fig)
"""

import io
import os
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether,
)

# ------------------------------------------------------------------
# Color palette (clinical, professional)
# ------------------------------------------------------------------
BRAND_BLUE      = colors.HexColor("#1A3C5E")
BRAND_TEAL      = colors.HexColor("#0E7C7B")
ALERT_RED       = colors.HexColor("#C0392B")
ALERT_ORANGE    = colors.HexColor("#E67E22")
ALERT_YELLOW    = colors.HexColor("#F1C40F")
NEUTRAL_GRAY    = colors.HexColor("#7F8C8D")
LIGHT_GRAY      = colors.HexColor("#ECF0F1")
DARK_TEXT       = colors.HexColor("#2C3E50")
WHITE           = colors.white

URGENCY_COLORS = {
    "critical":  ALERT_RED,
    "urgent":    ALERT_ORANGE,
    "elevated":  ALERT_YELLOW,
    "routine":   BRAND_TEAL,
}


# ------------------------------------------------------------------
# Style definitions
# ------------------------------------------------------------------
def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=BRAND_BLUE,
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="Helvetica",
            fontSize=10,
            textColor=NEUTRAL_GRAY,
            spaceAfter=2,
            alignment=TA_CENTER,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=BRAND_BLUE,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyText",
            fontName="Helvetica",
            fontSize=9,
            textColor=DARK_TEXT,
            leading=14,
            spaceAfter=4,
        ),
        "body_bold": ParagraphStyle(
            "BodyBold",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=DARK_TEXT,
            leading=14,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=NEUTRAL_GRAY,
            leading=11,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            textColor=NEUTRAL_GRAY,
            leading=11,
            borderPadding=(6, 6, 6, 6),
            backColor=LIGHT_GRAY,
        ),
        "urgency_label": ParagraphStyle(
            "UrgencyLabel",
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
    }
    return styles


# ------------------------------------------------------------------
# Figure → ReportLab Image (from matplotlib Figure or raw PNG bytes)
# ------------------------------------------------------------------
def _fig_to_rl_image(fig: matplotlib.figure.Figure, width_mm: float = 170) -> RLImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = RLImage(buf)
    aspect = fig.get_figheight() / fig.get_figwidth()
    img.drawWidth  = width_mm * mm
    img.drawHeight = width_mm * mm * aspect
    return img


def _bytes_to_rl_image(png_bytes: bytes, width_pt: float = 480) -> RLImage:
    """Convert raw PNG bytes to a ReportLab Image scaled to width_pt points."""
    buf = io.BytesIO(png_bytes)
    img = RLImage(buf)
    # Read actual pixel dimensions to preserve aspect ratio
    import struct
    # PNG header: width at bytes 16-20, height at 20-24
    try:
        w_px = struct.unpack(">I", png_bytes[16:20])[0]
        h_px = struct.unpack(">I", png_bytes[20:24])[0]
        aspect = h_px / w_px
    except Exception:
        aspect = 0.25   # fallback
    img.drawWidth  = width_pt
    img.drawHeight = width_pt * aspect
    return img


# ------------------------------------------------------------------
# Section builders
# ------------------------------------------------------------------
def _header_section(report: dict, styles: dict) -> list:
    elements = []

    system = report.get("system", {})
    rec    = report.get("recording", {})

    elements.append(Paragraph("ECG Analysis Report", styles["title"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        system.get("name", "ECG Abnormality Detection System"),
        styles["subtitle"],
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        f"Model v{system.get('model_version', '1.0')}  |  "
        f"Analysis time: {system.get('analysis_timestamp_local', 'N/A')}",
        styles["subtitle"],
    ))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_BLUE, spaceAfter=10))

    return elements


def _urgency_banner(report: dict, styles: dict) -> list:
    urgency_block = report.get("urgency", {})
    if not urgency_block:
        return []

    level    = urgency_block.get("level", "routine")
    guidance = urgency_block.get("guidance", "")
    color    = URGENCY_COLORS.get(level, BRAND_TEAL)

    banner_data = [[
        Paragraph(f"TRIAGE LEVEL: {level.upper()}", styles["urgency_label"]),
    ]]
    banner = Table(banner_data, colWidths=[170*mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), color),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))

    elements = [
        Spacer(1, 4),
        banner,
        Spacer(1, 4),
        Paragraph(guidance, styles["small"]),
        Paragraph(
            urgency_block.get("note", ""),
            styles["small"],
        ),
        Spacer(1, 6),
    ]
    return elements


def _model_result_section(report: dict, styles: dict) -> list:
    result = report.get("model_result", {})
    elements = [Paragraph("Model Result", styles["section_heading"])]

    is_abnormal = result.get("is_abnormal", False)
    pred_color  = ALERT_RED if is_abnormal else BRAND_TEAL

    result_data = [
        ["Prediction",            result.get("prediction", "N/A")],
        ["Classification",        "ABNORMAL" if is_abnormal else "NORMAL"],
        ["Abnormal Probability",  result.get("probability_percent", "N/A")],
        ["Normal Probability",    f"{round((1 - result.get('abnormal_probability', 0)) * 100, 1)}%"],
    ]

    t = Table(result_data, colWidths=[55*mm, 110*mm])
    t.setStyle(TableStyle([
        ("FONTNAME",       (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",       (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("TEXTCOLOR",      (0,0), (0,-1),  NEUTRAL_GRAY),
        ("TEXTCOLOR",      (1,1), (1,1),   pred_color),
        ("FONTNAME",       (1,1), (1,1),   "Helvetica-Bold"),
        ("BACKGROUND",     (0,0), (-1,0),  LIGHT_GRAY),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, LIGHT_GRAY]),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor("#D5D8DC")),
        ("TOPPADDING",     (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
        ("LEFTPADDING",    (0,0), (-1,-1), 6),
    ]))
    elements.append(t)
    return elements


def _interpretation_section(report: dict, styles: dict) -> list:
    interp   = report.get("interpretation", {})
    elements = []

    elements.append(Paragraph("Patient Explanation", styles["section_heading"]))
    elements.append(Paragraph(
        interp.get("patient_explanation", "Not available."), styles["body"]
    ))

    elements.append(Paragraph("Clinical Summary", styles["section_heading"]))
    elements.append(Paragraph(
        interp.get("clinical_summary", "Not available."), styles["body"]
    ))

    elements.append(Paragraph("Confidence Note", styles["section_heading"]))
    elements.append(Paragraph(
        interp.get("confidence_note", "Not available."), styles["small"]
    ))

    return elements


def _signal_and_model_section(report: dict, styles: dict) -> list:
    sig   = report.get("signal", {})
    model = report.get("model", {})

    elements = [Paragraph("Technical Metadata", styles["section_heading"])]

    metadata_rows = [
        ["Signal Length",    f"{sig.get('length_samples', 'N/A')} samples"],
        ["Duration",         f"{sig.get('duration_seconds', 'N/A')} s"],
        ["Sampling Rate",    f"{sig.get('sampling_rate_hz', 'N/A')} Hz"],
        ["Leads Analyzed",   f"{sig.get('num_leads', 'N/A')}"],
        ["Lead Names",       ", ".join(sig.get("lead_names", []))],
        ["Model Architecture", model.get("architecture", "N/A")],
        ["Model Version",    model.get("version", "N/A")],
    ]

    t = Table(metadata_rows, colWidths=[55*mm, 110*mm])
    t.setStyle(TableStyle([
        ("FONTNAME",       (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",       (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 8.5),
        ("TEXTCOLOR",      (0,0), (0,-1),  NEUTRAL_GRAY),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, LIGHT_GRAY]),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor("#D5D8DC")),
        ("TOPPADDING",     (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 3),
        ("LEFTPADDING",    (0,0), (-1,-1), 6),
    ]))
    elements.append(t)
    return elements


PINK_BG  = colors.HexColor("#FFF0F3")
PINK_HDR = colors.HexColor("#C0606080")  # semi-transparent rose for header row
PINK_BOX = colors.HexColor("#FFE0E6")


def _explainability_section(report: dict, styles: dict) -> list:
    expl = report.get("explainability")
    if not expl:
        return []

    elements = [Paragraph("Explainability Summary", styles["section_heading"])]

    rows = [
        ["Visualization Method",  expl.get("visualization_type", "N/A")],
        ["Most Activated Lead",   expl.get("most_activated_lead", "N/A")],
        ["Activation Score",
         str(expl.get("most_activated_lead_score", "N/A"))],
        ["XAI Method",            expl.get("method", "N/A")],
    ]

    t = Table(rows, colWidths=[55*mm, 110*mm])
    t.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",       (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR",      (0, 0), (0, -1),  NEUTRAL_GRAY),
        ("BACKGROUND",     (0, 0), (-1, -1), PINK_BG),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#E8A0A0")),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 4))

    note_style = ParagraphStyle(
        "ExplNote",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        textColor=DARK_TEXT,
        leading=13,
        backColor=PINK_BG,
        borderPadding=(6, 8, 6, 8),
    )
    elements.append(Paragraph(expl.get("clinical_note", ""), note_style))
    elements.append(Spacer(1, 6))
    return elements


def _disclaimer_section(report: dict, styles: dict) -> list:
    elements = [
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=0.8, color=NEUTRAL_GRAY, spaceAfter=8),
        Paragraph("Disclaimer", styles["section_heading"]),
        Spacer(1, 4),
        Paragraph(report.get("disclaimer", ""), styles["disclaimer"]),
    ]
    return elements


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def generate_pdf_report(
    report: dict,
    output_path: str = "ecg_report.pdf",
    gradcam_fig: Optional[matplotlib.figure.Figure] = None,
) -> str:
    """
    Generate a PDF ECG analysis report.

    Parameters
    ----------
    report      : structured report dict from ecg_report.generate_report()
    output_path : file path for the PDF output
    gradcam_fig : optional matplotlib Figure (Grad-CAM overlay) to embed

    Returns
    -------
    output_path : str — path of the generated PDF
    """

    styles = _build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=18*mm,
        bottomMargin=18*mm,
        title="ECG Analysis Report",
        author="ECG MI Detection System",
    )

    story = []

    story += _header_section(report, styles)
    story += _urgency_banner(report, styles)
    story.append(KeepTogether(_model_result_section(report, styles)))
    story += _interpretation_section(report, styles)

    # Embed Grad-CAM figure if provided
    if gradcam_fig is not None:
        story.append(Paragraph("Grad-CAM Saliency Overlay", styles["section_heading"]))
        story.append(Paragraph(
            "The heatmap below shows the regions of the ECG signal that most "
            "strongly influenced the model's classification decision. "
            "Red/yellow regions indicate high model attention; blue indicates low attention.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))
        story.append(_fig_to_rl_image(gradcam_fig, width_mm=170))
        story.append(Spacer(1, 6))

    story.append(KeepTogether(_signal_and_model_section(report, styles)))
    story += _disclaimer_section(report, styles)

    doc.build(story)
    print(f"[PDF] Report saved to: {output_path}")
    return output_path


def generate_pdf_report_bytes(
    report: dict,
    fig_gradcam: bytes = None,
    fig_saliency: bytes = None,
    fig_heatmap: bytes = None,
) -> bytes:
    """
    Generate PDF report and return as bytes (for Streamlit download).

    Parameters
    ----------
    report       : structured report dict from inference/report.py
    fig_gradcam  : PNG bytes of the Grad-CAM figure
    fig_saliency : PNG bytes of the Saliency figure
    fig_heatmap  : PNG bytes of the Global Heatmap figure (12-lead only)

    Returns
    -------
    bytes : PDF file contents
    """
    buf    = io.BytesIO()
    styles = _build_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=18*mm,
        bottomMargin=18*mm,
        title="ECG Analysis Report",
        author="ECG MI Detection System",
    )

    story = []

    # 1. Header
    story += _header_section(report, styles)
    story += _urgency_banner(report, styles)

    # 2. Patient Result
    story.append(KeepTogether(_model_result_section(report, styles)))

    # 3. Explainability Summary
    story += _explainability_section(report, styles)

    # 4. Embedded figures
    _FIG_WIDTH_PT = 480   # max width in points (~169 mm)

    if fig_heatmap is not None:
        story.append(Paragraph("Figure 3: Global Heatmap (12-Lead)",
                               styles["section_heading"]))
        story.append(Paragraph(
            "Rows = leads, columns = time. Warm colours (red) indicate high Grad-CAM "
            "activation; cool colours (blue) indicate low activation.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))
        story.append(_bytes_to_rl_image(fig_heatmap, width_pt=_FIG_WIDTH_PT))
        story.append(Spacer(1, 8))

    if fig_gradcam is not None:
        story.append(Paragraph("Figure 1: Grad-CAM Activation Map",
                               styles["section_heading"]))
        story.append(Paragraph(
            "Colour-coded background segments show where the model focused most "
            "strongly. Red = high activation, blue = low activation.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))
        story.append(_bytes_to_rl_image(fig_gradcam, width_pt=_FIG_WIDTH_PT))
        story.append(Spacer(1, 8))

    if fig_saliency is not None:
        story.append(Paragraph("Figure 2: Saliency Map",
                               styles["section_heading"]))
        story.append(Paragraph(
            "Individual samples are coloured by gradient magnitude. "
            "Hot colours (white/yellow/red) mark the most influential time points.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))
        story.append(_bytes_to_rl_image(fig_saliency, width_pt=_FIG_WIDTH_PT))
        story.append(Spacer(1, 8))

    # 5. Clinical Interpretation
    story += _interpretation_section(report, styles)

    # 6. Technical metadata
    story.append(KeepTogether(_signal_and_model_section(report, styles)))

    # 7. Disclaimer
    story += _disclaimer_section(report, styles)

    doc.build(story)
    return buf.getvalue()


# ------------------------------------------------------------------
# Example
# ------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import sys
    sys.path.insert(0, ".")

    DISCLAIMER = (
        "This report is generated by an automated AI decision-support system. "
        "It does not constitute a medical diagnosis and must not be used as the "
        "sole basis for any clinical decision. All findings must be correlated with "
        "patient history and the judgment of a licensed clinician."
    )

    sample_report = {
        "system": {
            "name": "ECG Ischemia & MI Detection System",
            "model_version": "1.2.0",
            "analysis_timestamp_utc": "2026-03-07T10:00:00Z",
            "analysis_timestamp_local": "2026-03-07 12:00:00",
        },
        "recording": {
            "patient_id": "PT-00421",
            "recording_id": "REC-20260307-001",
        },
        "model_result": {
            "prediction": "Inferior STEMI",
            "is_abnormal": True,
            "abnormal_probability": 0.91,
            "normal_probability": 0.09,
            "probability_percent": "91.0%",
        },
        "urgency": {
            "level": "critical",
            "guidance": "Immediate clinical review required. Findings are consistent with an acute ST-elevation pattern.",
            "note": "Urgency level is an algorithmic heuristic and does not replace clinical triage judgment.",
        },
        "interpretation": {
            "patient_explanation": (
                "The automated system detected ECG patterns that may be associated with "
                "a heart attack affecting the lower part of your heart. A clinician should "
                "review this result immediately."
            ),
            "clinical_summary": (
                "Model classified this ECG as Inferior STEMI with very high confidence (91%). "
                "Saliency is sharply localized on the ST segment. Inferior lead territory "
                "(II, III, aVF) implicated — likely RCA territory. ST-elevation pattern is dominant. "
                "Correlate with troponin, clinical presentation, and full 12-lead review."
            ),
            "confidence_note": (
                "Model confidence: very high (91.0%). Confidence reflects internal pattern-matching "
                "certainty based on training data. It does not imply clinical diagnostic certainty."
            ),
        },
        "signal": {
            "length_samples": 5000,
            "duration_seconds": 10.0,
            "sampling_rate_hz": 500,
            "num_leads": 12,
            "lead_names": ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"],
        },
        "model": {
            "architecture": "1D CNN",
            "version": "1.2.0",
        },
        "disclaimer": DISCLAIMER,
    }

    # Generate a mock Grad-CAM figure
    rng = np.random.default_rng(0)
    signal_np  = rng.normal(0, 0.5, 5000)
    heatmap_np = rng.uniform(0, 0.3, 5000)
    heatmap_np[2250:3250] += 0.65

    fig, ax = plt.subplots(figsize=(14, 3))
    time_axis = np.arange(5000) / 500
    sc = ax.scatter(time_axis, signal_np, c=heatmap_np, cmap="jet", s=1, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Grad-CAM Importance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title("Grad-CAM Overlay — Inferior STEMI")
    ax.set_facecolor("#f9f9f9")

    generate_pdf_report(sample_report, output_path="ecg_report.pdf", gradcam_fig=fig)
    plt.close(fig)