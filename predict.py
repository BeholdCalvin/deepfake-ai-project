"""
predict.py

Objectives addressed:
  2 – Unified image / video inference:
        • Detects input type from file extension.
        • Images are processed as a single frame (no LSTM path).
        • Videos sample `sequence_length` evenly-spaced frames.
        • The model's forward() handles both shapes transparently.
  3 – Grad-CAM integration:
        • Uses pytorch-grad-cam targeting the EfficientNet-B4 conv_head layer.
        • Returns a coloured heatmap overlaid on the face crop so analysts can
          see which facial region drove the FAKE/REAL decision.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image

# ── Grad-CAM ─────────────────────────────────────────────────────────────────
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Project imports ───────────────────────────────────────────────────────────
from dataloaders.transforms import get_val_transforms
from models.fusion import DeepfakeDetector
from utils.face_extractor import FaceExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _VIDEO_EXTS


def _tensor_to_rgb_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a CHW float32 normalised tensor back to HWC uint8 RGB for
    Grad-CAM overlay rendering.  Uses ImageNet stats to invert normalisation.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = tensor.cpu().numpy().transpose(1, 2, 0)   # CHW → HWC
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img   # float32 in [0, 1], RGB


class BinaryOutputTarget:
    """
    Grad-CAM target for binary classifiers that return a single logit.

    `ClassifierOutputTarget` expects a 2-D tensor like [B, num_classes], but
    `DeepfakeDetector` returns shape [B]. When Grad-CAM iterates sample-by-sample,
    each item can become a 0-D scalar tensor, so we must handle all cases.
    """

    def __init__(self, category: int = 0):
        self.category = category

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 0:
            return model_output
        if model_output.ndim == 1:
            if model_output.numel() == 1:
                return model_output.squeeze(0)
            return model_output[self.category]
        return model_output[:, self.category]

# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeGradCAM:
    """
    Thin wrapper around pytorch-grad-cam for DeepfakeDetector.

    The target layer is model.backbone.conv_head (the last depthwise conv
    before global average pooling in EfficientNet-B4).  Gradients flow
    through the spatial branch, ignoring the FFT path, which gives a
    spatially-interpretable heatmap on the face image.
    """

    def __init__(self, model: DeepfakeDetector, device: torch.device):
        self.model  = model
        self.device = device
        self.cam    = GradCAM(
            model=model,
            target_layers=[model.gradcam_target_layer],
        )

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        input_tensor : [1, C, H, W]  (single frame, already on CPU or GPU)
        returns      : HWC uint8 BGR heatmap overlaid on the face (for cv2).
        """
        input_tensor = input_tensor.to(self.device)

        # The detector returns a single binary logit per sample, so Grad-CAM
        # needs a custom target that can handle scalar / 1-D outputs.
        targets = [BinaryOutputTarget(0)]

        # Ensure gradients are enabled even if inference previously ran under
        # torch.no_grad() elsewhere in the pipeline.
        with torch.enable_grad():
            grayscale_cam = self.cam(
                input_tensor=input_tensor,
                targets=targets,
            )                                    # [1, H, W]

        grayscale_cam = grayscale_cam[0]         # [H, W]

        # Reconstruct visible face for overlay
        rgb_float = _tensor_to_rgb_numpy(input_tensor.squeeze(0).cpu())

        # show_cam_on_image expects float32 RGB in [0,1]
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        return overlay   # HWC uint8 RGB


# ─────────────────────────────────────────────────────────────────────────────
# Inference functions
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(
    image_path: str,
    model: DeepfakeDetector,
    extractor: FaceExtractor,
    transform,
    device: torch.device,
    gradcam: DeepfakeGradCAM | None = None,
) -> dict:
    """
    Run inference on a single still image.
    Returns a result dict with keys: label, confidence, heatmap (or None).
    """
    # Load and convert to RGB
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"error": f"Cannot read image: {image_path}"}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Face detection (MTCNN expects a batch)
    faces = extractor.extract_batch([img_rgb])
    if not faces or faces[0] is None:
        return {"error": "No face detected in image."}

    face       = faces[0]                                       # HWC uint8 RGB
    face_tensor = transform(image=face)["image"].unsqueeze(0).to(device)  # [1,C,H,W]

    model.eval()
    with torch.no_grad():
        logits = model(face_tensor)
        prob   = torch.sigmoid(logits).item()

    label      = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if label == "FAKE" else (1.0 - prob)

    # Grad-CAM heatmap (Objective 3)
    heatmap = None
    gradcam_error = None
    if gradcam is not None:
        try:
            heatmap = gradcam.generate(face_tensor)
        except Exception as exc:
            gradcam_error = f"Grad-CAM unavailable: {exc}"

    return {
        "label":         label,
        "confidence":    confidence,
        "heatmap":       heatmap,
        "face":          face,
        "gradcam_error": gradcam_error,
    }


def predict_video(
    video_path: str,
    model: DeepfakeDetector,
    extractor: FaceExtractor,
    transform,
    device: torch.device,
    sequence_length: int = 8,
    gradcam: DeepfakeGradCAM | None = None,
) -> dict:
    """
    Run inference on a video by sampling `sequence_length` evenly-spaced frames.
    Returns a result dict with keys: label, confidence, heatmap (or None).
    The heatmap corresponds to the most suspicious frame (highest FAKE prob).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return {"error": "Cannot read video file."}

    indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

    frames       = []   # tensors for model input
    face_crops   = []   # raw numpy faces for heatmap selection

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

    if len(frames) < sequence_length:
        return {
            "error": (
                f"Only {len(frames)} faces detected; need {sequence_length}. "
                "Try a video with clearer face visibility."
            )
        }

    # Build sequence batch: [1, T, C, H, W]
    seq_tensor = torch.stack(frames).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(seq_tensor)
        prob   = torch.sigmoid(logits).item()

    label      = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if label == "FAKE" else (1.0 - prob)

    # ── Grad-CAM on the most representative frame ─────────────────────────────
    # We pick the middle frame as a stable representative; for higher fidelity
    # you could run per-frame CAM and pick the argmax of individual probs.
    heatmap = None
    gradcam_error = None
    if gradcam is not None:
        mid_idx    = len(frames) // 2
        mid_tensor = frames[mid_idx].unsqueeze(0).to(device)    # [1,C,H,W]
        try:
            heatmap = gradcam.generate(mid_tensor)
        except Exception as exc:
            gradcam_error = f"Grad-CAM unavailable: {exc}"

    return {
        "label":         label,
        "confidence":    confidence,
        "heatmap":       heatmap,
        "face":          face_crops[len(face_crops) // 2],
        "gradcam_error": gradcam_error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point (Objective 2)
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    input_path: str,
    model: DeepfakeDetector,
    extractor: FaceExtractor,
    transform,
    device: torch.device,
    sequence_length: int = 8,
    use_gradcam: bool = True,
) -> dict:
    """
    Auto-detect input type and dispatch to the correct inference function.
    Returns a unified result dict.
    """
    cam_helper = DeepfakeGradCAM(model, device) if use_gradcam else None

    if _is_image(input_path):
        return predict_image(input_path, model, extractor, transform, device, cam_helper)
    elif _is_video(input_path):
        return predict_video(input_path, model, extractor, transform, device, sequence_length, cam_helper)
    else:
        return {"error": f"Unsupported file type: {os.path.splitext(input_path)[1]}"}


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

    detector = DeepfakeDetector(sequence_length=SEQ_LENGTH).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
    else:
        detector.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))

        print(f"\nAnalyzing: {os.path.basename(INPUT_PATH)}")
        result = predict(INPUT_PATH, detector, face_extractor, val_transform, DEVICE, SEQ_LENGTH)
        print("-" * 40)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Result     : {result['label']}")
            print(f"Confidence : {result['confidence'] * 100:.2f}%")

            if result.get("heatmap") is not None:
                out_path = "gradcam_output.png"
                heatmap_bgr = cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, heatmap_bgr)
                print(f"Grad-CAM heatmap saved to: {out_path}")
        print("-" * 40)
