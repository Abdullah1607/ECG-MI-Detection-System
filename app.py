import streamlit as st
import tempfile
import os
import json
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.ticker as mticker

from inference.disclaimer import CLINICAL_DISCLAIMER
from inference.explanation_text import generate_explanation
from inference.report import generate_report
from inference.ecg_pdf_report import generate_pdf_report_bytes
from inference.ecg_loader import load_ecg, preprocess_signal
from inference.ecg_image_extractor import extract_lead_ii_from_image
from inference.predict import predict_ecg
from inference.explain import (compute_saliency, compute_gradcam,
                                compute_gradcam_12lead, compute_saliency_12lead)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Abnormality Detection",
    page_icon="🫀",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Risk indicator boxes */
.risk-high {
    background-color: #4a0000;
    border-left: 5px solid #e05c5c;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 8px 0;
    color: #ffd5d5;
    font-size: 1.05rem;
    font-weight: 600;
}
.risk-moderate {
    background-color: #3a2800;
    border-left: 5px solid #f0a500;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 8px 0;
    color: #ffe8a0;
    font-size: 1.05rem;
    font-weight: 600;
}
.risk-low {
    background-color: #002a14;
    border-left: 5px solid #3ddc84;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 8px 0;
    color: #b2f5c8;
    font-size: 1.05rem;
    font-weight: 600;
}
/* Metric card tweaks */
div[data-testid="metric-container"] {
    background-color: #1c2333;
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #2e3d5a;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🫀 ECG Abnormality Detection System")
st.markdown(
    "#### AI-Powered Myocardial Infarction & Ischemia Screening",
    unsafe_allow_html=False,
)

with st.expander("⚠️ Clinical Disclaimer", expanded=True):
    st.warning(CLINICAL_DISCLAIMER)

st.divider()

# ── Helpers ────────────────────────────────────────────────────────────────────

def risk_box(label: str, prob: float):
    if prob > 0.6:
        msg = "🔴 HIGH RISK: Strong abnormal ECG indicators detected"
        cls = "risk-high"
    elif prob >= 0.3:
        msg = "🟡 MODERATE RISK: Borderline findings detected — clinical review advised"
        cls = "risk-moderate"
    else:
        msg = "🟢 LOW RISK: ECG appears within normal limits"
        cls = "risk-low"
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)


# ── Publication-quality ECG plot helpers ───────────────────────────────────────

_ECG_BG         = "#fff0f3"   # light pink paper
_ECG_GRID_COLOR = "#e07070"   # classic ECG red grid
_TICK_COLOR     = "#3a0000"   # dark red — readable on light bg

_TICK_SAMPLES   = np.arange(0, 5001, 500)          # every 500 samples
_TICK_MS        = [int(s / 500 * 1000) for s in _TICK_SAMPLES]   # 0…10000 ms


def _apply_ecg_paper(ax):
    """Draw ECG paper grid: major every 500 samples, minor every 100 samples."""
    ax.set_facecolor(_ECG_BG)
    # Minor grid (small squares — 100 samples = 0.04 s)
    ax.set_xticks(np.arange(0, 5001, 100), minor=True)
    ax.xaxis.grid(True, which="minor", color=_ECG_GRID_COLOR, alpha=0.40, linewidth=0.3)
    ax.yaxis.grid(True, which="minor", color=_ECG_GRID_COLOR, alpha=0.40, linewidth=0.3)
    # Major grid (large squares — 500 samples = 0.2 s)
    ax.set_xticks(_TICK_SAMPLES)
    ax.xaxis.grid(True, which="major", color=_ECG_GRID_COLOR, alpha=0.75, linewidth=0.8)
    ax.yaxis.grid(True, which="major", color=_ECG_GRID_COLOR, alpha=0.75, linewidth=0.8)
    ax.set_xticklabels(_TICK_MS, fontsize=8, color=_TICK_COLOR)
    ax.tick_params(which="both", colors=_TICK_COLOR, labelsize=8)
    ax.set_xlabel("Time (ms)", color=_TICK_COLOR, fontsize=9)
    ax.set_ylabel("Amplitude (mV)", color=_TICK_COLOR, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_ECG_GRID_COLOR)


def _add_colorbar(fig, ax, mappable, label: str):
    cb = fig.colorbar(mappable, ax=ax, pad=0.01, aspect=25)
    cb.set_label(label, color=_TICK_COLOR, fontsize=8)
    cb.ax.set_facecolor(_ECG_BG)
    cb.ax.yaxis.set_tick_params(color=_TICK_COLOR, labelcolor=_TICK_COLOR, labelsize=7)


def _plot_saliency_panel(fig, ax, raw: np.ndarray, sal_norm: np.ndarray):
    """ECG paper background + glow + scatter saliency overlay."""
    xs = np.arange(len(raw))
    _apply_ecg_paper(ax)
    ax.set_xlim(0, 5000)
    ax.set_ylim(raw.min() - 0.3, raw.max() + 0.3)
    # Glow layers (subtle dark red on light background)
    ax.plot(xs, raw, linewidth=6,   alpha=0.06, color="#c00000", zorder=2)
    ax.plot(xs, raw, linewidth=3,   alpha=0.12, color="#c00000", zorder=3)
    ax.plot(xs, raw, linewidth=0.8, alpha=0.95, color="#1a0000", zorder=4)
    # Scatter heatmap
    sc = ax.scatter(xs, raw, c=sal_norm, cmap="hot", vmin=0, vmax=1,
                    s=sal_norm * 8 + 1, alpha=0.85, linewidths=0, zorder=5)
    _add_colorbar(fig, ax, sc, "Saliency")
    ax.set_title("Saliency Map — Neural Attention", color=_TICK_COLOR, fontsize=11, pad=6)


def _plot_gradcam_panel(fig, ax, raw: np.ndarray, cam_norm: np.ndarray):
    """ECG paper background + axvspan region shading + ECG line."""
    xs = np.arange(len(raw))
    _apply_ecg_paper(ax)
    ax.set_xlim(0, 5000)
    ax.set_ylim(raw.min() - 0.3, raw.max() + 0.3)
    # Region shading — 50-sample segments
    cmap_jet = plt.colormaps["jet"]
    for i in range(100):              # 5000 / 50 = 100 segments
        x0, x1  = i * 50, (i + 1) * 50
        cam_val = float(cam_norm[x0:x1].mean())
        ax.axvspan(x0, x1, alpha=cam_val * 0.6, color=cmap_jet(cam_val), linewidth=0)
    # ECG line on top
    ax.plot(xs, raw, linewidth=0.8, alpha=0.95, color="#1a0000", zorder=5)
    # Fake mappable for colorbar
    sm = plt.cm.ScalarMappable(cmap="jet", norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    _add_colorbar(fig, ax, sm, "CAM Activation")
    ax.set_title("Grad-CAM — Pathology Regions", color=_TICK_COLOR, fontsize=11, pad=6)


def _plot_for_pdf(ax, raw: np.ndarray, sal_norm: np.ndarray):
    """Light-background version used when building the PDF."""
    xs = np.arange(len(raw))
    norm = mcolors.Normalize(vmin=0, vmax=1)
    pts  = np.array([xs, raw]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap="hot", norm=norm, linewidth=1.2)
    lc.set_array(sal_norm[:-1])
    ax.add_collection(lc)
    ax.set_xlim(0, 5000)
    ax.set_ylim(raw.min() - 0.2, raw.max() + 0.2)
    ax.set_title("ECG — Saliency Heatmap (Lead II)", fontsize=10)
    ax.set_xlabel("Sample (500 Hz)", fontsize=8)
    ax.set_ylabel("Amplitude (mV)", fontsize=8)


def _fig_to_bytes(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ── Safe reshape helper ───────────────────────────────────────────────────────

def _safe_reshape_12lead(arr: np.ndarray, name: str = "signal"):
    """
    Safely reshape a flat array into (12, 5000).
    - If size == 60000 → reshape directly.
    - If size == 5000  → place into lead 0, pad rest with zeros.
    - Otherwise        → return None (caller should handle).
    """
    arr = arr.astype(np.float32).ravel()
    if arr.size == 60000:
        return arr.reshape(12, 5000)
    elif arr.size == 5000:
        out = np.zeros((12, 5000), dtype=np.float32)
        out[0] = arr
        return out
    else:
        return None


# ── Input section ──────────────────────────────────────────────────────────────
st.subheader("📂 Upload ECG")

col_l, col_r = st.columns(2)

with col_l:
    input_method = st.radio(
        "Input type",
        ["Digital ECG File", "ECG Image / PDF"],
        horizontal=True,
        label_visibility="collapsed",
    )

with col_r:
    if input_method == "Digital ECG File":
        st.info(
            "**CSV** (.csv — single column)  →  1-Lead CNN\n\n"
            "**CSV** (.csv — 12 columns)  →  12-Lead CNN\n\n"
            "**WFDB** (.dat + .hea)  →  12-Lead CNN",
        )
    else:
        st.warning(
            "⚠️ Image upload is for signal visualization only. "
            "AI analysis requires digital ECG files. "
            "Supported formats for AI analysis: WFDB (.dat + .hea) or CSV."
        )

if input_method == "Digital ECG File":
    uploaded_files = st.file_uploader(
        "Upload .csv, or both .dat and .hea",
        type=["csv", "dat", "hea"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    uploaded_image = None
else:
    uploaded_image = st.file_uploader(
        "Upload ECG image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    uploaded_files = None

run = st.button("▶ Run Analysis", type="primary", use_container_width=False)

# ── Analysis ───────────────────────────────────────────────────────────────────
if run:

    # ── Image path ───────────────────────────────────────────────────────────────
    if input_method == "ECG Image / PDF":
        if uploaded_image is None:
            st.error("Please upload an ECG image or PDF first.")
            st.stop()

        with st.spinner("Extracting signal and running AI analysis…"):
            suffix = os.path.splitext(uploaded_image.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_image.getvalue())
                tmp_path = tmp.name
            try:
                signal_raw  = extract_lead_ii_from_image(tmp_path)
            finally:
                os.unlink(tmp_path)
            signal_proc          = preprocess_signal(signal_raw)
            prob, label, confidence = predict_ecg(signal_proc, model_type="image")
            saliency_arr         = compute_saliency(signal_proc, "1lead")
            gradcam_arr, _       = compute_gradcam(signal_proc, "1lead")

        result = {
            "prediction":           label,
            "abnormal_probability": round(float(prob), 4),
            "confidence":           round(float(confidence), 4),
            "model_used":           "image",
            "saliency":             saliency_arr.tolist(),
            "gradcam":              gradcam_arr.tolist(),
            "signal_1lead":         signal_proc.tolist(),
            "extracted_signal":     signal_proc.tolist(),
            "signal_12lead":        None,
            "gradcam_12lead":       None,
        }
        label      = result["prediction"]
        prob       = result["abnormal_probability"]
        confidence = result["confidence"]
        saliency   = np.array(result["saliency"])
        gradcam    = np.array(result["gradcam"])
        raw_ecg    = np.array(result["signal_1lead"], dtype=np.float32)
        model_used = "image"
        best_lead_name       = None
        best_lead_activation = None

        user_text, clinical_text, confidence_text = generate_explanation(label, prob, model_used)
        report = generate_report(
            label, prob, user_text, clinical_text, confidence_text, model_used,
            best_lead_name=None, best_lead_activation=None,
        )

        st.divider()
        st.warning(
            "⚠️ EXPERIMENTAL: Image-based analysis uses a model trained on synthetic "
            "ECG images (ROC-AUC: 0.808). This is less accurate than digital file input "
            "(ROC-AUC: 0.899). False negatives occur in ~1 in 6 abnormal cases. "
            "Always confirm with digital ECG analysis and clinical evaluation."
        )

    # ── Digital file path ────────────────────────────────────────────────────────
    else:
        if not uploaded_files:
            st.error("Please upload at least one ECG file.")
            st.stop()

        tmp_dir    = tempfile.mkdtemp()
        file_paths = []
        try:
            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                file_paths.append(path)

            signal, model_type       = load_ecg(file_paths)
            prob, label, confidence  = predict_ecg(signal, model_type)
            saliency_arr             = compute_saliency(signal, model_type)
            gradcam_arr, _           = compute_gradcam(signal, model_type)

            _LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
            result = {
                "prediction":           label,
                "abnormal_probability": round(float(prob), 4),
                "confidence":           round(float(confidence), 4),
                "model_used":           model_type,
                "saliency":             saliency_arr.tolist(),
                "gradcam":              gradcam_arr.tolist(),
                "signal_12lead":        None,
                "gradcam_12lead":       None,
                "signal_1lead":         None,
            }
            if model_type == "12lead":
                gradcam_12lead              = compute_gradcam_12lead(signal)
                saliency_12lead             = compute_saliency_12lead(signal)
                lead_scores                 = gradcam_12lead.mean(axis=1)
                best_idx                    = int(lead_scores.argmax())
                result["signal_12lead"]         = signal.tolist()
                result["gradcam_12lead"]        = gradcam_12lead.tolist()
                result["saliency_12lead"]       = saliency_12lead.tolist()
                result["best_lead_name"]        = _LEAD_NAMES[best_idx]
                result["best_lead_activation"]  = float(lead_scores[best_idx])
            else:
                result["signal_1lead"] = signal.tolist()

        except Exception as e:
            st.error(f"Analysis error: {e}")
            st.stop()
        finally:
            for path in file_paths:
                if os.path.exists(path):
                    os.remove(path)
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

        label      = result["prediction"]
        prob       = result["abnormal_probability"]
        confidence = result["confidence"]
        saliency   = np.array(result["saliency"])
        gradcam    = np.array(result["gradcam"])
        raw_ecg    = np.array(result.get("signal_1lead") or result.get("extracted_signal") or result.get("signal_12lead", [[]])[1])
        model_used = result.get("model_used", "1lead")

        # Best-lead metadata (12lead only)
        best_lead_name       = result.get("best_lead_name")
        best_lead_activation = result.get("best_lead_activation")

        # Generate text & report
        user_text, clinical_text, confidence_text = generate_explanation(label, prob, model_used)
        report = generate_report(
            label, prob, user_text, clinical_text, confidence_text, model_used,
            best_lead_name=best_lead_name,
            best_lead_activation=best_lead_activation,
        )

        st.divider()

    # ── Shared display ────────────────────────────────────────────────────────

    # ── 🩺 Clinical Risk Indicator ────────────────────────────────────────────
    st.subheader("🩺 Clinical Risk Indicator")
    risk_box(label, prob)

    # ── Model info ────────────────────────────────────────────────────────────
    if model_used == "12lead":
        model_label = "12-Lead CNN (WFDB input)"
        _threshold  = 0.43
    elif model_used == "image":
        model_label = "Image CNN (synthetic training) | AUC: 0.808 | EXPERIMENTAL"
        _threshold  = 0.91
    else:
        model_label = "1-Lead CNN (Lead II)"
        _threshold  = 0.48
    st.info(f"**Model used:** {model_label}  |  **Threshold:** {_threshold}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction",            label)
    c2.metric("Abnormal Probability",  f"{prob * 100:.1f}%")
    c3.metric("Model Confidence",      f"{confidence * 100:.1f}%")

    # ── 📊 Probability Distribution ───────────────────────────────────────────
    st.subheader("📊 Probability Distribution")
    prob_normal = 1.0 - prob
    col_a, col_b = st.columns([6, 1])
    with col_a:
        st.markdown("🔴 **Abnormal ECG likelihood**")
        st.progress(float(prob))
    with col_b:
        st.markdown(f"### {prob * 100:.1f}%")
    col_c, col_d = st.columns([6, 1])
    with col_c:
        st.markdown("🟢 **Normal ECG likelihood**")
        st.progress(float(prob_normal))
    with col_d:
        st.markdown(f"### {prob_normal * 100:.1f}%")

    # ── 📈 ECG Signal Visualization ───────────────────────────────────────────

    import matplotlib.gridspec as gridspec

    # ── palette ──
    _VBG = '#FFF0F3'   # figure / axes background
    _VMJ = '#E8A0A0'   # major grid
    _VMN = '#F0C0C0'   # minor grid
    _VSG = '#8B0000'   # ECG signal (dark red)
    _VTX = '#2C0A0A'   # axis text / labels
    _VSP = '#C97070'   # spines / tick marks

    # ── helpers ──
    def _vstyle(ax, show_xaxis=False):
        ax.set_facecolor(_VBG)
        ax.set_xticks(range(0, 5001, 500))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(100))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
        ax.grid(which='major', color=_VMJ, alpha=0.7, linewidth=0.6)
        ax.grid(which='minor', color=_VMN, alpha=0.4, linewidth=0.2)
        for spine in ax.spines.values():
            spine.set_color(_VSP)
        if show_xaxis:
            ax.set_xticklabels([f"{int(t/500*1000)}" for t in range(0, 5001, 500)],
                               color=_VTX, fontsize=7)
            ax.set_xlabel('Time (ms)', color=_VTX, fontsize=8)
            ax.set_ylabel('Amplitude (mV)', color=_VTX, fontsize=8)
            ax.tick_params(colors=_VTX)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(which='both', length=0)

    def _vfig(figsize):
        plt.style.use('default')
        f = plt.figure(figsize=figsize)
        f.patch.set_facecolor(_VBG)
        return f

    def _vcbar(fig, ax, mappable, label):
        cb = fig.colorbar(mappable, ax=ax, pad=0.01)
        cb.set_label(label, color=_VTX, fontsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=_VTX, fontsize=7)
        return cb

    LEAD_LAYOUT = [
        (0, 0, 'I',    0), (0, 1, 'aVR', 3), (0, 2, 'V1',  6), (0, 3, 'V4',  9),
        (1, 0, 'II',   1), (1, 1, 'aVL', 4), (1, 2, 'V2',  7), (1, 3, 'V5', 10),
        (2, 0, 'III',  2), (2, 1, 'aVF', 5), (2, 2, 'V3',  8), (2, 3, 'V6', 11),
    ]

    # ════════════════════════════════════════════════════════════════════════════
    # 12-LEAD VISUALIZATIONS
    # ════════════════════════════════════════════════════════════════════════════
    if result.get("model_used") == "12lead" and result.get("signal_12lead"):

        sig12 = np.array(result["signal_12lead"], dtype=np.float32).reshape(12, 5000)

        cam12_raw = result.get("gradcam_12lead")
        cam12 = np.array(cam12_raw if cam12_raw else np.zeros((12, 5000)),
                         dtype=np.float32).reshape(12, 5000)

        sal12_raw = result.get("saliency_12lead")
        sal12 = np.array(sal12_raw if sal12_raw else np.zeros((12, 5000)),
                         dtype=np.float32).reshape(12, 5000)

        # ── Plot 1: Global Grad-CAM heatmap (12-lead × time) ─────────────────
        st.markdown("### 🌡️ Global Grad-CAM Heatmap")
        # Downsample time axis: (12, 5000) → (12, 500)
        cam_down = cam12.reshape(12, 500, 10).mean(axis=2)
        # Global normalisation only — preserves real differences between leads
        cam_norm = (cam_down - cam_down.min()) / (cam_down.max() - cam_down.min() + 1e-8)

        fig1, ax1 = plt.subplots(figsize=(16, 5))
        fig1.patch.set_facecolor(_VBG)
        ax1.set_facecolor('#FFF8FA')
        im = ax1.imshow(cam_norm, aspect='auto', cmap='RdYlBu_r',
                        interpolation='bilinear', vmin=0, vmax=1)
        ax1.set_yticks(range(12))
        ax1.set_yticklabels(['I','II','III','aVR','aVL','aVF',
                             'V1','V2','V3','V4','V5','V6'],
                            fontsize=10, color=_VTX)
        ax1.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499])
        ax1.set_xticklabels([0, 1000, 2000, 3000, 4000, 5000,
                             6000, 7000, 8000, 9000, 10000],
                            fontsize=9, color=_VTX)
        ax1.set_xlabel('Time (ms)', color=_VTX, fontsize=10)
        ax1.set_ylabel('ECG Lead',  color=_VTX, fontsize=10)
        for _i in range(1, 12):
            ax1.axhline(_i - 0.5, color=_VSP, linewidth=0.5, alpha=0.5)
        ax1.set_title('Grad-CAM Activation — All 12 Leads',
                      color=_VTX, fontsize=11, fontweight='bold')
        for spine in ax1.spines.values():
            spine.set_color(_VSP)
        cbar1 = plt.colorbar(im, ax=ax1)
        cbar1.set_label('Grad-CAM Activation', fontsize=9, color=_VTX)
        cbar1.ax.yaxis.set_tick_params(color=_VTX)
        plt.setp(cbar1.ax.yaxis.get_ticklabels(), color=_VTX)
        plt.tight_layout()
        st.session_state['fig_heatmap_bytes'] = _fig_to_bytes(fig1)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        st.caption("Warm colours = high model attention | Cool colours = low attention | "
                   "Shows which leads and time regions contributed most to the prediction.")
        st.markdown("""
**How to read this heatmap:**
- Each row represents one ECG lead (I, II, III, aVR, aVL, aVF, V1–V6)
- Each column represents a time point (0–10,000 ms)
- **Warm colours (red/yellow)** = the model paid strong attention to this region
- **Cool colours (blue)** = low model attention, likely normal segment
- Vertical bright columns = the model focused on those heartbeat cycles across multiple leads
- If certain leads (e.g. V1–V4) show consistently high activation, this may suggest anterior ischemia
- If inferior leads (II, III, aVF) are most activated, this may suggest inferior MI patterns
""")

        # ── Plot 2: Most-activated lead ───────────────────────────────────────
        st.markdown("### 🔍 Most Activated Lead")
        lead_scores   = cam12.mean(axis=1)         # (12,) mean activation per lead
        top_lead_idx  = int(lead_scores.argmax())
        lead_names_12 = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        top_name      = lead_names_12[top_lead_idx]

        fig2, ax2 = plt.subplots(figsize=(16, 3))
        fig2.patch.set_facecolor(_VBG)
        _vstyle(ax2, show_xaxis=True)

        norm2 = mcolors.Normalize(vmin=0, vmax=1)
        cmap2 = plt.colormaps['RdYlBu_r']
        seg   = 50
        for s in range(0, 5000, seg):
            e  = min(s + seg, 5000)
            cv = float(cam12[top_lead_idx, s:e].mean())
            ax2.axvspan(s, e, alpha=max(cv * 0.55, 0.05),
                        color=cmap2(norm2(cv)), zorder=1)
        ax2.plot(np.arange(5000), sig12[top_lead_idx],
                 color=_VSG, linewidth=1.0, alpha=0.95, zorder=2)
        ax2.set_title(f'Lead {top_name} — Highest Mean Grad-CAM Activation',
                      color=_VTX, fontsize=10, fontweight='bold')
        ax2.text(0.99, 0.90,
                 f'Lead {top_name}  (mean={lead_scores[top_lead_idx]:.2f})',
                 transform=ax2.transAxes, ha='right', va='top',
                 color=_VSG, fontsize=9, fontstyle='italic',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#FFE0E0', ec=_VSP, alpha=0.9))
        sm2 = cm.ScalarMappable(cmap='RdYlBu_r', norm=norm2)
        sm2.set_array([])
        _vcbar(fig2, ax2, sm2, 'CAM Activation')
        plt.tight_layout()
        st.session_state['fig_gradcam_bytes'] = _fig_to_bytes(fig2)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.markdown(f"""
**Why Lead {top_name}?**
The model assigned the highest mean activation score to **Lead {top_name}**, \
meaning this lead contributed most strongly to the prediction.
- **Red/yellow regions** on the waveform highlight the exact time segments the model focused on
- These regions often correspond to ST changes, abnormal Q waves, or T-wave inversions
- This does NOT confirm a diagnosis — it shows where the model's attention was concentrated
- A qualified clinician must interpret these findings in clinical context
""")

        # ── Plot 3: 12-lead grid (3×4 + rhythm strip) ────────────────────────
        st.markdown("### 📋 12-Lead ECG Grid")
        fig3 = _vfig((22, 12))
        gs3  = gridspec.GridSpec(4, 4, figure=fig3, hspace=0.55, wspace=0.35)

        norm3 = mcolors.Normalize(vmin=0, vmax=1)
        cmap3 = plt.colormaps['RdYlBu_r']

        for (row, col, name, idx) in LEAD_LAYOUT:
            ax = fig3.add_subplot(gs3[row, col])
            _vstyle(ax, show_xaxis=False)
            for s in range(0, 5000, seg):
                e  = min(s + seg, 5000)
                cv = float(cam12[idx, s:e].mean())
                ax.axvspan(s, e, alpha=max(cv * 0.5, 0.04),
                           color=cmap3(norm3(cv)), zorder=1)
            ax.plot(np.arange(5000), sig12[idx],
                    color=_VSG, linewidth=0.8, alpha=0.95, zorder=2)
            ax.text(0.02, 0.88, name, transform=ax.transAxes,
                    color=_VTX, fontsize=9, fontweight='bold')

        ax_r = fig3.add_subplot(gs3[3, :])
        _vstyle(ax_r, show_xaxis=True)
        for s in range(0, 5000, seg):
            e  = min(s + seg, 5000)
            cv = float(cam12[1, s:e].mean())
            ax_r.axvspan(s, e, alpha=max(cv * 0.5, 0.04),
                         color=cmap3(norm3(cv)), zorder=1)
        ax_r.plot(np.arange(5000), sig12[1],
                  color=_VSG, linewidth=0.9, alpha=0.95, zorder=2)
        ax_r.text(0.01, 0.88, 'II — Rhythm Strip', transform=ax_r.transAxes,
                  color=_VTX, fontsize=9, fontweight='bold')

        sm3 = cm.ScalarMappable(cmap='RdYlBu_r', norm=norm3)
        sm3.set_array([])
        cbar3 = fig3.colorbar(sm3, ax=fig3.get_axes(), shrink=0.45, pad=0.01)
        cbar3.set_label('Grad-CAM Activation', color=_VTX, fontsize=9)
        plt.setp(cbar3.ax.yaxis.get_ticklabels(), color=_VTX, fontsize=7)
        fig3.suptitle('12-Lead ECG — Grad-CAM Overlay',
                      color=_VTX, fontsize=13, fontweight='bold', y=1.005)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        st.caption("Warm (red) = high CAM activation | Dark red line = ECG signal")
        st.markdown("""
**12-Lead Grid with Grad-CAM overlay:**
- Each lead panel shows the raw ECG signal overlaid with Grad-CAM shading
- **Warmer background colour** in a panel = that lead contributed more to the prediction
- **Rhythm strip (bottom)** shows Lead II across the full 10-second recording
- Compare activation patterns across limb leads (I, II, III, aVR, aVL, aVF) \
and precordial leads (V1–V6) to identify which cardiac territory may be affected
""")

        # ── Plot 4: Combined saliency on highest-saliency lead ───────────────
        _sal12_lead_means = sal12.mean(axis=1)          # (12,)
        _sal_best_idx     = int(_sal12_lead_means.argmax())
        sal_ref_name      = lead_names_12[_sal_best_idx]
        _sal_best_score   = float(_sal12_lead_means[_sal_best_idx])

        st.markdown(f"### 🧠 Combined Saliency Map (Lead {sal_ref_name})")
        sal_avg  = sal12.mean(axis=0)          # (5000,) combined across leads
        sal_avg /= (sal_avg.max() + 1e-8)
        sig_ref  = sig12[_sal_best_idx]        # signal of highest-saliency lead

        fig4, ax4 = plt.subplots(figsize=(16, 3))
        fig4.patch.set_facecolor(_VBG)
        _vstyle(ax4, show_xaxis=True)
        ax4.plot(np.arange(5000), sig_ref,
                 color=_VSG, linewidth=5,   alpha=0.05, zorder=1)
        ax4.plot(np.arange(5000), sig_ref,
                 color=_VSG, linewidth=2.5, alpha=0.10, zorder=2)
        ax4.plot(np.arange(5000), sig_ref,
                 color=_VSG, linewidth=0.9, alpha=0.95, zorder=3)
        sc4 = ax4.scatter(np.arange(5000), sig_ref,
                          c=sal_avg, cmap='hot',
                          s=sal_avg * 10 + 0.5, alpha=0.85,
                          linewidths=0, zorder=4)
        _vcbar(fig4, ax4, sc4, 'Saliency')
        ax4.set_title(f'Combined Saliency — Lead {sal_ref_name} (Highest Saliency Lead)',
                      color=_VTX, fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.session_state['fig_saliency_bytes'] = _fig_to_bytes(fig4)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)
        st.caption(f"Hot colours = points most influential to the model's decision | "
                   f"Signal shown: Lead {sal_ref_name} (highest mean saliency) | "
                   f"Saliency averaged across all 12 leads")
        st.info(
            f"Lead **{sal_ref_name}** had the highest mean saliency score "
            f"({_sal_best_score:.4f}), indicating it contains the most influential "
            f"time steps for this prediction."
        )
        st.markdown("""
**Saliency Map explanation:**
- Saliency shows **fine-grained, time-step level importance** using input gradients
- **Bright/hot coloured points** on the ECG line = these exact time steps had the \
greatest influence on the model's prediction
- Unlike Grad-CAM (which works at the feature map level), saliency operates directly \
on the raw input signal
- High saliency at QRS peaks = model sensitive to depolarization morphology
- High saliency at ST segment = model detecting possible ST elevation/depression
""")

    # ════════════════════════════════════════════════════════════════════════════
    # 1-LEAD VISUALIZATIONS
    # ════════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("### 📈 ECG Signal Visualization")
        sig1         = np.array(result.get("signal_1lead") or
                                result.get("extracted_signal") or [],
                                dtype=np.float32)
        saliency_arr = np.array(result["saliency"], dtype=np.float32)
        gradcam_arr  = np.array(result["gradcam"],  dtype=np.float32)
        samples      = np.arange(5000)
        norm1        = mcolors.Normalize(vmin=0, vmax=1)
        cmap_cam     = plt.colormaps['RdYlBu_r']
        seg          = 50

        # ── Plot A: Grad-CAM overlay ──────────────────────────────────────────
        st.markdown("#### Grad-CAM — Pathology Regions")
        figA, axA = plt.subplots(figsize=(16, 3))
        figA.patch.set_facecolor(_VBG)
        _vstyle(axA, show_xaxis=True)
        for s in range(0, 5000, seg):
            e  = min(s + seg, 5000)
            cv = float(gradcam_arr[s:e].mean())
            axA.axvspan(s, e, alpha=max(cv * 0.55, 0.04),
                        color=cmap_cam(norm1(cv)), zorder=1)
        axA.plot(samples, sig1, color=_VSG, linewidth=1.0, alpha=0.95, zorder=2)
        smA = cm.ScalarMappable(cmap='RdYlBu_r', norm=norm1)
        smA.set_array([])
        _vcbar(figA, axA, smA, 'CAM Activation')
        axA.set_title('Grad-CAM Activation — Lead II', color=_VTX,
                      fontsize=10, fontweight='bold')
        axA.text(0.99, 0.90, 'Gradient-weighted Class Activation Map',
                 transform=axA.transAxes, ha='right', va='top',
                 color=_VSG, fontsize=8, fontstyle='italic')
        plt.tight_layout()
        st.session_state['fig_gradcam_bytes'] = _fig_to_bytes(figA)
        st.session_state['fig_heatmap_bytes'] = None
        st.pyplot(figA, use_container_width=True)
        plt.close(figA)
        st.markdown("""
**Grad-CAM explanation:**
- Red/warm regions indicate where in the Lead II recording the model focused most
- The model analyzes temporal patterns within Lead II to detect MI-related changes
- Key regions of interest: QRS complex morphology, ST segment, T-wave shape
- This model focuses on temporal regions within Lead II to detect \
myocardial infarction-related patterns
- Results must be validated by a qualified clinician
""")

        # ── Plot B: Saliency scatter ──────────────────────────────────────────
        st.markdown("#### Saliency Map — Neural Attention")
        figB, axB = plt.subplots(figsize=(16, 3))
        figB.patch.set_facecolor(_VBG)
        _vstyle(axB, show_xaxis=True)
        axB.plot(samples, sig1, color=_VSG, linewidth=5,   alpha=0.05, zorder=1)
        axB.plot(samples, sig1, color=_VSG, linewidth=2.5, alpha=0.10, zorder=2)
        axB.plot(samples, sig1, color=_VSG, linewidth=0.9, alpha=0.95, zorder=3)
        scB = axB.scatter(samples, sig1, c=saliency_arr, cmap='hot',
                          s=saliency_arr * 10 + 0.5, alpha=0.85,
                          linewidths=0, zorder=4)
        _vcbar(figB, axB, scB, 'Saliency')
        axB.set_title('Saliency Map — Gradient Attention', color=_VTX,
                      fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.session_state['fig_saliency_bytes'] = _fig_to_bytes(figB)
        st.pyplot(figB, use_container_width=True)
        plt.close(figB)
        st.markdown("""
**Saliency explanation:**
- Hot coloured scatter points show time steps with highest gradient magnitude
- These are the exact signal points that most influenced the model's decision
- Concentrated saliency around QRS complexes suggests morphology-based detection
- Saliency around ST segment suggests possible ST deviation detection
""")

        # ── Plot C: Raw ECG + prediction annotation ───────────────────────────
        st.markdown("#### Raw ECG Signal")
        figC, axC = plt.subplots(figsize=(16, 3))
        figC.patch.set_facecolor(_VBG)
        _vstyle(axC, show_xaxis=True)
        axC.plot(samples, sig1, color=_VSG, linewidth=1.0, alpha=0.95)
        pred_txt = (f"Prediction: {label}\n"
                    f"P(Abnormal): {prob*100:.1f}%\n"
                    f"Confidence: {confidence*100:.1f}%")
        axC.text(0.99, 0.97, pred_txt,
                 transform=axC.transAxes, ha='right', va='top',
                 fontsize=8, color=_VTX,
                 bbox=dict(boxstyle='round,pad=0.4', fc='#FFE0E0',
                           ec=_VSP, alpha=0.92))
        axC.set_title('Raw ECG — Lead II (Z-scored)', color=_VTX,
                      fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(figC, use_container_width=True)
        plt.close(figC)

    # sal_norm kept for PDF section below
    sal_norm = np.array(result["saliency"], dtype=np.float32)
    sal_norm = sal_norm / (sal_norm.max() + 1e-8)

    # ── 💡 Interpretation ─────────────────────────────────────────────────────
    st.subheader("💡 Interpretation")

    tab_patient, tab_clinical = st.tabs(["For Patient", "For Clinician / Technical Summary"])
    with tab_patient:
        st.write(user_text)
        st.caption(confidence_text)
    with tab_clinical:
        st.write(clinical_text)
        st.caption(confidence_text)

    # ── Research note (image input only) ─────────────────────────────────────
    if model_used == "image":
        st.info(
            "🔬 Research Note: The image analysis model was trained on synthetically "
            "generated ECG paper images derived from PTB-XL digital signals. "
            "No large-scale public ECG image dataset with standardized diagnostic labels "
            "exists, making this an open research problem. "
            "The model achieves ROC-AUC of 0.808 on synthetic test images. "
            "Performance on real scanned ECGs may vary. "
            "This feature demonstrates proof-of-concept feasibility. "
            "Digital ECG input (.dat+.hea or .csv) remains the recommended pathway "
            "for reliable AI analysis."
        )

    # ── 📋 PDF Report ─────────────────────────────────────────────────────────
    st.subheader("📋 Full Report")

    pdf_bytes = generate_pdf_report_bytes(
        report,
        fig_gradcam=st.session_state.get('fig_gradcam_bytes'),
        fig_saliency=st.session_state.get('fig_saliency_bytes'),
        fig_heatmap=st.session_state.get('fig_heatmap_bytes'),
    )

    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name="ecg_report.pdf",
        mime="application/pdf",
        use_container_width=False,
    )

    #with st.expander("View JSON Report", expanded=False):
        # Remove disclaimer from displayed JSON to keep it tidy
        #display_report = {k: v for k, v in report.items() if k != "disclaimer"}
        #st.json(display_report)

    st.divider()
    st.caption(
        "This AI system is for research and educational use only. "
        "It is not a certified medical device and must not replace professional clinical diagnosis."
    )