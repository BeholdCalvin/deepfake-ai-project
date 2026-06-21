"""
predict.py

Objectives addressed
────────────────────
2 – Unified image / video inference (robust dimension handling):
      • Images  → [B, C, H, W]       → 4-D single-frame path, LSTM bypassed.
      • Videos  → [B, T, C, H, W]    → 5-D sequence path.
      • Fewer-than-expected faces are padded (last valid frame repeated) rather
        than hard-failing so short clips with occasional occlusion still work.

3 – Dual-branch Grad-CAM:
      Gradients flow through BOTH the EfficientNet-B4 spatial stream AND the
      FFT-CNN frequency stream.  The blended heatmap shows face-region blending
      artefacts (spatial) *and* spectral GAN fingerprints (frequency) together.
      Individual branch overlays are returned for analyst drill-down.
"""

import os
from typing import Optional

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataloaders.transforms import get_val_transforms
from models.fusion import DeepfakeDetector, load_checkpoint
from utils.face_extractor import FaceExtractor


# ─────────────────────────────────────────────────────────────────────────────
# File-type helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
)
_VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm"}
)


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _VIDEO_EXTS


def _tensor_to_rgb_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Invert ImageNet normalisation on a CHW float32 tensor → HWC float32 [0,1].

    show_cam_on_image() requires a float32 RGB array in [0, 1]; this function
    undoes the Normalize(mean=_MEAN, std=_STD) applied during preprocessing.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = tensor.cpu().numpy().transpose(1, 2, 0)   # CHW → HWC
    return np.clip(img * std + mean, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM target
# ─────────────────────────────────────────────────────────────────────────────

class BinaryOutputTarget:
    """
    Grad-CAM target for a binary classifier that emits shape [B] logits.

    pytorch-grad-cam's built-in ClassifierOutputTarget expects [B, num_classes].
    DeepfakeDetector returns a single logit per sample (squeezed to [B]), which
    can become a 0-D scalar inside GradCAM's per-sample loop.  This shim handles
    all output shapes without the IndexError that ClassifierOutputTarget raises.
    """

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 0:
            return model_output                    # scalar → return as-is
        if model_output.ndim == 1:
            # [1] single-sample case or [B] batch – take element 0
            return model_output.squeeze(0) if model_output.numel() == 1 \
                   else model_output[0]
        return model_output[0, 0]                  # [B, 1] defensive case


# ─────────────────────────────────────────────────────────────────────────────
# Dual-Branch Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class DualBranchGradCAM:
    """
    Grad-CAM over BOTH the spatial (EfficientNet-B4) and frequency (FFT-CNN)
    branches of DeepfakeDetector.

    Why dual-branch?
    ────────────────
    • Spatial CAM : highlights face regions with unnatural blending – soft jaw
      lines, mismatched texture at eye boundaries, identity-swap artefacts.
      Anchored at model.gradcam_spatial_layer (EfficientNet-B4 conv_head).
    • Frequency CAM : highlights where periodic spectral fingerprints drove the
      decision – GAN up-sampling grids, JPEG ring artefacts at specific spatial
      loci.  Anchored at model.gradcam_fft_layer (FFTBranch last Conv2d).

    Dynamic, per-sample blend weight (NOT a fixed prior)
    ──────────────────────────────────────────────────────
    A static 65/35 blend is misleading: for any given image the network may
    have relied almost entirely on one branch, and a fixed prior would hide
    that.  Instead, _compute_branch_weights() measures each branch's ACTUAL
    contribution to this specific prediction via a Grad×Input attribution at
    the fusion boundary:

      1. A forward hook on `model.fusion_fc` captures its input tensor –
         the concatenation cat([spatial_1792, freq_512]) – with
         `.retain_grad()` enabled.
      2. backward() on the logit propagates a gradient into that captured
         tensor.
      3. |gradient * activation|, summed separately over the spatial slice
         (first 1792 dims) and the frequency slice (last 512 dims), gives a
         magnitude-aware estimate of how much each branch's features moved
         the final logit for THIS sample.
      4. The two scores are normalised to sum to 1.0 and used as the blend
         weight – so a sample where the model leaned 90 % on the frequency
         branch is correctly blended 90/10, not forced into 65/35.

    This is the standard "Gradient × Input" attribution method, applied at
    the fusion layer instead of at input pixels — cheap (one extra
    forward+backward) and model-agnostic given the branch dimensions exposed
    on DeepfakeDetector (SPATIAL_DIM, FFT_DIM).

    Implementation note
    ────────────────────
    Three independent forward+backward passes run per generate() call: one
    for branch-weight attribution, one for the spatial CAM, one for the FFT
    CAM.  pytorch-grad-cam resets its own activation/gradient buffers at the
    start of every __call__, and model.zero_grad() is called between passes,
    so none of the three interfere with each other.
    """

    # Fallback prior used ONLY if gradient-based attribution degenerates
    # (e.g. a zero-gradient edge case on a saturated logit).  Not used in
    # the normal path — kept so generate() never has to return without a
    # usable blend.
    _FALLBACK_SPATIAL_WEIGHT: float = 0.65
    _FALLBACK_FFT_WEIGHT:     float = 0.35

    def __init__(self, model: DeepfakeDetector, device: torch.device) -> None:
        if not hasattr(model, "gradcam_spatial_layer"):
            raise AttributeError(
                "model.gradcam_spatial_layer not found. "
                "Ensure fusion.py is the updated version exposing both CAM anchors."
            )
        if not hasattr(model, "gradcam_fft_layer"):
            raise AttributeError(
                "model.gradcam_fft_layer not found. "
                "Ensure fusion.py is the updated version exposing both CAM anchors."
            )
        if not hasattr(model, "fusion_fc"):
            raise AttributeError(
                "model.fusion_fc not found. Branch-weight attribution hooks "
                "the fusion_fc input tensor; this requires the standard "
                "DeepfakeDetector architecture from fusion.py."
            )

        self.model  = model
        self.device = device

        # Two separate GradCAM instances: each registers its own forward /
        # backward hooks on the respective target layer.
        self._spatial_cam = GradCAM(
            model=model,
            target_layers=[model.gradcam_spatial_layer],
        )
        self._fft_cam = GradCAM(
            model=model,
            target_layers=[model.gradcam_fft_layer],
        )

    @staticmethod
    def _normalise(cam: np.ndarray) -> np.ndarray:
        """Min-max normalise a Grad-CAM map to [0, 1]."""
        cam  = np.clip(cam, 0.0, None)
        vmax = float(cam.max())
        return cam / (vmax + 1e-8) if vmax > 0 else cam

    def _compute_branch_weights(self, input_tensor: torch.Tensor) -> tuple[float, float]:
        """
        Measure each branch's real contribution to this prediction via
        Grad×Input attribution at the fusion boundary (see class docstring).

        Args:
            input_tensor: [1, C, H, W] single frame, already on self.device.

        Returns:
            (spatial_weight, fft_weight) – non-negative floats summing to 1.0.
        """
        captured: dict[str, torch.Tensor] = {}

        def _capture_fused_input(module, inputs, output):
            # inputs[0] is the tensor passed INTO fusion_fc, i.e. exactly
            # cat([spatial_features, freq_features], dim=1) — see
            # DeepfakeDetector._extract_frame_features in fusion.py.
            fused = inputs[0]
            fused.retain_grad()   # fused is a non-leaf tensor; grad isn't
            captured["fused"] = fused  # kept by default without retain_grad()

        handle = self.model.fusion_fc.register_forward_hook(_capture_fused_input)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(input_tensor)        # [1] – single-image path
            logits.sum().backward()
        finally:
            handle.remove()

        fused_act  = captured["fused"].detach()       # [1, 2304]
        fused_grad = captured["fused"].grad           # [1, 2304] or None

        # Always clear parameter grads afterwards so this attribution pass
        # never leaks gradient state into the CAM passes that follow.
        self.model.zero_grad(set_to_none=True)

        if fused_grad is None:
            return self._FALLBACK_SPATIAL_WEIGHT, self._FALLBACK_FFT_WEIGHT

        contribution = (fused_grad * fused_act).abs()[0]          # [2304]
        spatial_score = contribution[: DeepfakeDetector.SPATIAL_DIM].sum().item()
        fft_score     = contribution[DeepfakeDetector.SPATIAL_DIM:].sum().item()
        total         = spatial_score + fft_score

        if total <= 1e-12:
            # Degenerate case: essentially zero gradient reached the fusion
            # boundary (e.g. a fully saturated sigmoid).  Fall back rather
            # than divide by ~0 and produce a meaningless weight.
            return self._FALLBACK_SPATIAL_WEIGHT, self._FALLBACK_FFT_WEIGHT

        return spatial_score / total, fft_score / total

    def generate(self, input_tensor: torch.Tensor) -> dict[str, object]:
        """
        Compute Grad-CAM overlays for both branches, blended using each
        branch's measured contribution to this specific prediction.

        Args:
            input_tensor: [1, C, H, W] single frame (any device).

        Returns:
            dict with:
              'combined'       – HWC uint8 RGB, dynamically-weighted blend
              'spatial'        – HWC uint8 RGB, spatial branch only
              'fft'            – HWC uint8 RGB, frequency branch only
              'spatial_weight' – float in [0, 1], this sample's spatial share
              'fft_weight'     – float in [0, 1], this sample's frequency share
        """
        inp     = input_tensor.to(self.device)
        targets = [BinaryOutputTarget()]

        # 1) Measure actual per-sample branch contribution BEFORE running the
        #    CAM passes, so this pass's backward() can't be confused with
        #    GradCAM's internal backward calls.
        self.model.eval()
        spatial_weight, fft_weight = self._compute_branch_weights(inp)

        # 2) Two independent CAM passes: each GradCAM resets its activation /
        #    gradient buffers at call time, so results are fresh and
        #    non-interfering despite sharing the same underlying model.
        with torch.enable_grad():
            spatial_raw = self._spatial_cam(input_tensor=inp, targets=targets)[0]
            fft_raw     = self._fft_cam(input_tensor=inp,     targets=targets)[0]

        spatial_norm = self._normalise(spatial_raw)
        fft_norm     = self._normalise(fft_raw)

        # Blend using the MEASURED contribution, not a fixed prior.
        combined = self._normalise(
            spatial_weight * spatial_norm + fft_weight * fft_norm
        )

        rgb_float = _tensor_to_rgb_numpy(inp.squeeze(0).cpu())  # float32 [0,1]

        return {
            "combined":       show_cam_on_image(rgb_float, combined,     use_rgb=True),
            "spatial":        show_cam_on_image(rgb_float, spatial_norm, use_rgb=True),
            "fft":            show_cam_on_image(rgb_float, fft_norm,     use_rgb=True),
            "spatial_weight": spatial_weight,
            "fft_weight":     fft_weight,
        }


# Deprecated alias – retained for backward compatibility with scripts that
# import DeepfakeGradCAM by name.  Will be removed in a future major version.
DeepfakeGradCAM = DualBranchGradCAM


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_gradcam(
    gradcam: Optional[DualBranchGradCAM],
    face_tensor: torch.Tensor,
) -> dict[str, object]:
    """
    Safely execute Grad-CAM and return a flat dict of result keys.

    Returns a dict with keys: heatmap, spatial_heatmap, fft_heatmap,
    spatial_weight, fft_weight, gradcam_error.  All values are None when
    Grad-CAM is disabled or generation fails, so callers can spread this
    directly into their result dict without branching on success/failure.

    Named dict keys (vs. a positional tuple) avoid silent argument-order
    bugs if the return shape ever changes again.
    """
    empty: dict[str, object] = {
        "heatmap": None, "spatial_heatmap": None, "fft_heatmap": None,
        "spatial_weight": None, "fft_weight": None, "gradcam_error": None,
    }
    if gradcam is None:
        return empty
    try:
        out = gradcam.generate(face_tensor)
        return {
            "heatmap":         out["combined"],
            "spatial_heatmap": out["spatial"],
            "fft_heatmap":     out["fft"],
            "spatial_weight":  out["spatial_weight"],
            "fft_weight":      out["fft_weight"],
            "gradcam_error":   None,
        }
    except Exception as exc:
        return {**empty, "gradcam_error": f"Grad-CAM unavailable: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Inference functions
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(
    image_path: str,
    model: DeepfakeDetector,
    extractor: FaceExtractor,
    transform,
    device: torch.device,
    gradcam: Optional[DualBranchGradCAM] = None,
) -> dict:
    """
    Run inference on a single still image.

    Returns a dict with keys:
      label, confidence, face, heatmap, spatial_heatmap, fft_heatmap,
      gradcam_error
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"error": f"Cannot read image: {image_path}"}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces   = extractor.extract_batch([img_rgb])

    if not faces or faces[0] is None:
        return {"error": "No face detected in image."}

    face        = faces[0]                                             # HWC uint8 RGB
    # Move to device immediately: ensures EfficientNet and FFT kernels
    # run on the same accelerator without a silent CPU fallback.
    face_tensor = transform(image=face)["image"].unsqueeze(0).to(device)  # [1,C,H,W]

    model.eval()
    with torch.no_grad():
        logits = model(face_tensor)                                    # single-image path
        prob   = torch.sigmoid(logits).item()

    label      = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if label == "FAKE" else (1.0 - prob)

    combined_result = _run_gradcam(gradcam, face_tensor)

    return {
        "label":      label,
        "confidence": confidence,
        "face":       face,
        **combined_result,   # heatmap, spatial_heatmap, fft_heatmap, spatial_weight, fft_weight, gradcam_error
    }


def predict_video(
    video_path: str,
    model: DeepfakeDetector,
    extractor: FaceExtractor,
    transform,
    device: torch.device,
    sequence_length: int = 8,
    gradcam: Optional[DualBranchGradCAM] = None,
) -> dict:
    """
    Run inference on a video by sampling `sequence_length` evenly-spaced frames.

    Padding strategy: if fewer faces than `sequence_length` are detected the
    last valid frame is repeated.  This prevents hard failures on clips with
    brief occlusion at the cost of slightly reduced temporal diversity.
    A 'warning' key is added to the result dict when padding is applied.

    Returns a dict with keys:
      label, confidence, face, heatmap, spatial_heatmap, fft_heatmap,
      gradcam_error, warning (optional)
    """
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return {"error": "Cannot read video file or file has zero frames."}

    indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

    frames:     list[torch.Tensor] = []
    face_crops: list[np.ndarray]   = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces     = extractor.extract_batch([frame_rgb])

        if faces and faces[0] is not None:
            face        = faces[0]
            face_tensor = transform(image=face)["image"]
            frames.append(face_tensor)
            face_crops.append(face)

    cap.release()

    if len(frames) == 0:
        return {"error": "No faces detected in any sampled frame."}

    warning: Optional[str] = None

    # Pad with the last detected face rather than raising an error so that
    # clips with occasional occlusion (hats, hands, profile turns) still
    # produce a usable inference result.
    if len(frames) < sequence_length:
        n_missing = sequence_length - len(frames)
        warning = (
            f"Only {len(frames)} / {sequence_length} frames had detectable faces; "
            f"{n_missing} frame(s) padded with the last valid detection."
        )
        for _ in range(n_missing):
            # Clone to avoid tensor aliasing: duplicate references to the same
            # storage would be silently overwritten if any in-place op ran.
            frames.append(frames[-1].clone())
            face_crops.append(face_crops[-1].copy())

    # [1, T, C, H, W] ── dispatches to the video-sequence path in forward()
    seq_tensor = torch.stack(frames).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(seq_tensor)
        prob   = torch.sigmoid(logits).item()

    label      = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if label == "FAKE" else (1.0 - prob)

    # Grad-CAM on the middle frame: more stable than the first/last frame
    # which often suffer from motion blur or lighting transitions.
    mid_idx    = len(frames) // 2
    mid_tensor = frames[mid_idx].unsqueeze(0).to(device)               # [1,C,H,W]

    combined_result = _run_gradcam(gradcam, mid_tensor)

    return {
        "label":      label,
        "confidence": confidence,
        "face":       face_crops[mid_idx],
        "warning":    warning,
        **combined_result,   # heatmap, spatial_heatmap, fft_heatmap, spatial_weight, fft_weight, gradcam_error
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    input_path: str,
    model: DeepfakeDetector,
    extractor: FaceExtractor,
    transform,
    device: torch.device,
    sequence_length: int = 8,
    use_gradcam: bool    = True,
) -> dict:
    """
    Auto-detect input type (image vs video) and dispatch to the correct
    inference function.  Returns a unified result dict regardless of input type.
    """
    cam_helper = DualBranchGradCAM(model, device) if use_gradcam else None

    if _is_image(input_path):
        return predict_image(
            input_path, model, extractor, transform, device, cam_helper
        )
    elif _is_video(input_path):
        return predict_video(
            input_path, model, extractor, transform, device,
            sequence_length, cam_helper,
        )
    else:
        ext = os.path.splitext(input_path)[1]
        return {
            "error": (
                f"Unsupported file extension '{ext}'. "
                f"Images: {sorted(_IMAGE_EXTS)}  "
                f"Videos: {sorted(_VIDEO_EXTS)}"
            )
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    WEIGHTS_PATH = "weights/best_model.pth"
    INPUT_PATH   = r"C:\Users\rohan\Downloads\df\multi_domain_deepfake\data\raw\test2\1.mp4"
    SEQ_LENGTH   = 8
    IMG_SIZE     = 224
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    face_extractor = FaceExtractor()
    val_transform  = get_val_transforms(IMG_SIZE)
    detector       = DeepfakeDetector(sequence_length=SEQ_LENGTH).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
    else:
        load_checkpoint(detector, WEIGHTS_PATH, DEVICE)
        print(f"\nAnalyzing: {os.path.basename(INPUT_PATH)}")
        result = predict(
            INPUT_PATH, detector, face_extractor, val_transform, DEVICE, SEQ_LENGTH
        )
        print("-" * 40)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Result     : {result['label']}")
            print(f"Confidence : {result['confidence'] * 100:.2f}%")
            if result.get("heatmap") is not None:
                out_path    = "gradcam_output.png"
                heatmap_bgr = cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, heatmap_bgr)
                print(f"Grad-CAM heatmap saved to: {out_path}")
            if result.get("warning"):
                print(f"Warning: {result['warning']}")
        print("-" * 40)