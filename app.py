"""
app.py  –  Streamlit Deepfake Detection Frontend

Changes from original
─────────────────────
• Removed orphaned `DeepfakeGradCAM` import: the class is now internal to
  predict.py (renamed DualBranchGradCAM); app.py never instantiated it
  directly, so the import was always dead code that would cause an ImportError
  if the class name ever changed.
• _side_by_side() accepts optional `spatial_heatmap` / `fft_heatmap` args
  and renders them in a collapsible expander so power users can drill into
  which branch drove the decision.
• The `warning` key from predict_video (frame-padding notice) is surfaced as
  a st.warning() banner so analysts know when inference ran on padded frames.

Run with:
    streamlit run app.py
"""

import os
import tempfile
from xml.parsers.expat import model

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from dataloaders.transforms import get_val_transforms
from models.fusion import DeepfakeDetector
# DualBranchGradCAM is instantiated *inside* predict(); app.py never needs it directly.
from predict import predict
from utils.face_extractor import FaceExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_PATH   = "weights/best_model.pth"
SEQ_LENGTH     = 8
IMG_SIZE       = 224
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACCEPTED_TYPES = ["mp4", "avi", "mov", "jpg", "jpeg", "png"]


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources  (loaded once per process, shared across all reruns)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def load_model() -> DeepfakeDetector:
    """
    Load EfficientNet-B4 + FFT fusion model from disk.
    @st.cache_resource ensures weights are loaded exactly once per process,
    even when multiple browser sessions are connected simultaneously.
    """
    model = DeepfakeDetector(sequence_length=SEQ_LENGTH).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        st.error(
            f"Model weights not found at `{WEIGHTS_PATH}`. "
            "Please train the model first or place `best_model.pth` in `weights/`."
        )
        st.stop()

    state_dict = torch.load(
        WEIGHTS_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_extractor() -> FaceExtractor:
    return FaceExtractor()


@st.cache_resource(show_spinner=False)
def load_transform():
    return get_val_transforms(IMG_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _verdict_badge(label: str, confidence: float) -> None:
    """Render a large colour-coded verdict block."""
    colour = "#d62828" if label == "FAKE" else "#2d6a4f"
    icon   = "🚨"       if label == "FAKE" else "✅"
    st.markdown(
        f"""
        <div style="background-color:{colour};border-radius:12px;
                    padding:20px 30px;text-align:center;margin-bottom:16px;">
            <span style="font-size:2.5rem;">{icon}</span><br/>
            <span style="color:white;font-size:2rem;font-weight:700;">{label}</span><br/>
            <span style="color:#ffffffcc;font-size:1.1rem;">
                {confidence * 100:.1f}% confidence
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _confidence_bar(label: str, confidence: float) -> None:
    """Render a styled confidence progress bar."""
    pct        = int(confidence * 100)
    bar_colour = "#d62828" if label == "FAKE" else "#2d6a4f"
    st.markdown(f"**Confidence:** {pct}%")
    st.markdown(
        f"""
        <div style="background:#e0e0e0;border-radius:8px;height:18px;width:100%;">
          <div style="background:{bar_colour};width:{pct}%;height:18px;
                      border-radius:8px;transition:width 0.4s ease;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _side_by_side(
    face: np.ndarray,
    heatmap: np.ndarray | None,
    spatial_heatmap: np.ndarray | None = None,
    fft_heatmap: np.ndarray | None = None,
) -> None:
    """
    Show the analysed face crop alongside the combined Grad-CAM overlay.

    When dual-branch heatmaps are available they are rendered in a collapsible
    expander so the primary view stays uncluttered while power users can still
    drill into which branch (spatial vs frequency) drove the prediction.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Analysed Face Crop**")
        st.image(face, use_container_width=True)

    with col2:
        if heatmap is not None:
            st.markdown("**Grad-CAM Heatmap (Combined)**")
            st.image(heatmap, use_container_width=True)
            st.caption(
                "🔴 Red / warm = high activation (suspicious region).  "
                "Blended from spatial branch (65 %) + frequency branch (35 %)."
            )
        else:
            st.info("Grad-CAM not available for this input.")

    # Branch-specific drill-down in a collapsible section
    if spatial_heatmap is not None or fft_heatmap is not None:
        with st.expander("🔬 Branch-specific Grad-CAM (spatial vs frequency)"):
            b1, b2 = st.columns(2)
            with b1:
                if spatial_heatmap is not None:
                    st.markdown("**Spatial Branch**")
                    st.image(spatial_heatmap, use_container_width=True)
                    st.caption("Blending / texture artefacts at jaw, eyes, hairline.")
            with b2:
                if fft_heatmap is not None:
                    st.markdown("**Frequency Branch**")
                    st.image(fft_heatmap, use_container_width=True)
                    st.caption("Spectral GAN fingerprints: grid / ring patterns.")


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Deepfake Detector",
        page_icon="🔍",
        layout="wide",
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔍 Multi-Domain Deepfake Detector")
    st.markdown(
        "Powered by **EfficientNet-B4 + FFT Fusion** · "
        "Dual-Branch Grad-CAM · FaceForensics++ trained"
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        use_gradcam = st.toggle("Show Grad-CAM heatmap", value=True)
        seq_len     = st.slider(
            "Frames to sample (video only)",
            min_value=4, max_value=16, value=SEQ_LENGTH, step=2,
        )
        st.markdown("---")
        st.markdown(
            f"**Device:** `{DEVICE}`  \n"
            f"**Model:** EfficientNet-B4 + FFT  \n"
            f"**Weights:** `{WEIGHTS_PATH}`"
        )

    # ── Load resources ────────────────────────────────────────────────────────
    model     = load_model()
    extractor = load_extractor()
    transform = load_transform()

    # ── Upload widget ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a video or image to analyse",
        type=ACCEPTED_TYPES,
        help="Supported: MP4, AVI, MOV (video)  •  JPG, PNG (image)",
    )

    if uploaded is None:
        st.info("👆 Upload a file above to get started.")
        return

    # Write the upload to a temp file so OpenCV can access it on disk
    suffix = f".{uploaded.name.rsplit('.', 1)[-1].lower()}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.markdown(
        f"**File:** `{uploaded.name}`   |   **Size:** {uploaded.size / 1024:.1f} KB"
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    with st.spinner("Detecting faces and running inference…"):
        result = predict(
            input_path=tmp_path,
            model=model,
            extractor=extractor,
            transform=transform,
            device=DEVICE,
            sequence_length=seq_len,
            use_gradcam=use_gradcam,
        )

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    # ── Display results ───────────────────────────────────────────────────────
    st.divider()

    if "error" in result:
        st.error(f"❌ {result['error']}")
        return

    label      = result["label"]
    confidence = result["confidence"]
    face       = result.get("face")
    heatmap    = result.get("heatmap")

    if result.get("gradcam_error"):
        st.warning(result["gradcam_error"])
    # Surface the padding warning from predict_video so analysts know
    # that some frames were duplicated rather than freshly detected.
    if result.get("warning"):
        st.warning(f"⚠️ {result['warning']}")

    # Verdict + confidence
    col_verdict, col_conf = st.columns([1, 2])

    with col_verdict:
        _verdict_badge(label, confidence)

    with col_conf:
        st.markdown("### Confidence Score")
        _confidence_bar(label, confidence)
        st.markdown("")
        fake_pct = confidence * 100 if label == "FAKE" else (1 - confidence) * 100
        real_pct = 100 - fake_pct
        st.markdown(
            f"| Class | Score |\n|---|---|\n"
            f"| 🔴 FAKE | `{fake_pct:.1f}%` |\n"
            f"| 🟢 REAL | `{real_pct:.1f}%` |"
        )

    # Face crop + Grad-CAM side by side
    if face is not None:
        st.divider()
        st.subheader("📊 Visual Explanation (Dual-Branch Grad-CAM)")
        _side_by_side(
            face,
            heatmap,
            spatial_heatmap=result.get("spatial_heatmap"),
            fft_heatmap=result.get("fft_heatmap"),
        )

    # ── Interpretation guide ──────────────────────────────────────────────────
    with st.expander("ℹ️ How to interpret these results"):
        st.markdown(
            """
            **Confidence score** reflects how certain the model is about its prediction.
            A score above 70 % is considered reliable.

            **Combined Grad-CAM** blends two independent evidence streams:
            - 🔴 **Red / warm** → high activation (suspicious region)
            - 🔵 **Blue / cool** → low activation (ignored by the model)

            **Spatial branch** identifies face regions with unnatural blending:
            soft jaw lines, mismatched skin texture around eyes, hair-edge artefacts.

            **Frequency branch** identifies spectral GAN fingerprints:
            up-sampling grid patterns and JPEG ring artefacts that survive
            social-media re-encoding.

            Common manipulation tells the model detects:
            - Soft / blurred boundaries at jaw and hairline
            - Skin texture mismatch around the eyes
            - Spectral grid patterns from GAN up-sampling
            """
        )


if __name__ == "__main__":
    main()