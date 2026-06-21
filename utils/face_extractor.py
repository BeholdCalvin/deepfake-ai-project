"""
utils/face_extractor.py

Aligns face crops to a canonical 224×224 pose using MTCNN-detected 5-point
landmarks (left eye, right eye, nose, left mouth, right mouth) before
returning the crop.

Why alignment matters for deepfake detection
─────────────────────────────────────────────
Deepfake generators synthesise faces in a normalised coordinate space.
After alignment, manipulation artefacts (soft jaw edges, blending seams,
texture discontinuities) appear at consistent spatial locations across all
samples.  This makes it easier for the EfficientNet backbone and the
downstream Grad-CAM maps to localise them reliably.

Alignment method
────────────────
cv2.estimateAffinePartial2D computes a 4-DOF similarity transform (rotation,
uniform scale, translation – no shear).  A full 6-DOF affine is deliberately
avoided: non-uniform shearing distorts facial geometry and can introduce
spurious boundary artefacts that confuse the detector.

Fallback
────────
If landmark detection fails (profile views, heavy occlusion, low resolution)
the code falls back to a margin-padded bounding-box crop resized to
output_size × output_size.  Both paths return the same dtype / shape so
downstream code never needs to branch on which path ran.
"""

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Reference landmarks for 224×224 aligned face  (ArcFace / InsightFace standard)
# ─────────────────────────────────────────────────────────────────────────────
# Derived by scaling the standard 112×112 InsightFace reference points × 2.
# Layout: both eyes in the upper-centre with enough chin / forehead margin for
# the model to observe blending artefacts at face boundaries.
#
# Order: [left_eye, right_eye, nose_tip, left_mouth, right_mouth]  (x, y)
_REF_LANDMARKS_224 = np.array([
    [ 76.58,  103.39],   # left  eye
    [147.06,  103.00],   # right eye
    [112.05,  143.47],   # nose tip
    [ 83.09,  184.73],   # left  mouth corner
    [141.46,  184.41],   # right mouth corner
], dtype=np.float32)


class FaceExtractor:
    """
    Detects faces with MTCNN, aligns them using 5-point landmark affine
    transformation, and returns fixed-size RGB crops.

    Every returned crop is `output_size × output_size × 3` uint8 RGB so
    that the transform pipeline (Albumentations → EfficientNet) receives a
    consistent input shape regardless of the original video resolution.
    """

    def __init__(
        self,
        device: str      = "cuda",
        margin: float    = 0.3,
        output_size: int = 224,
    ) -> None:
        """
        Args:
            device:      Preferred compute device ('cuda' or 'cpu').
                         Automatically falls back to CPU when CUDA unavailable.
            margin:      Fractional padding added around bounding boxes in the
                         fallback crop path (0.3 = 30 % on each side).
            output_size: Side length in pixels of the returned square crop.
        """
        self.device      = torch.device(device if torch.cuda.is_available() else "cpu")
        self.margin      = margin
        self.output_size = output_size

        print(f"[FaceExtractor] device={self.device}, output_size={self.output_size}")

        # keep_all=False + select_largest=True: return only the most prominent
        # face per frame.  Avoids multi-face index tracking in talking-head
        # interview footage (the most common deepfake scenario).
        self.mtcnn = MTCNN(
            keep_all=False,
            select_largest=True,
            device=self.device,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_batch(
        self,
        frames_rgb: list[np.ndarray],
    ) -> list[np.ndarray | None]:
        """
        Detect, align, and crop the primary face from each frame in the batch.

        Args:
            frames_rgb: List of HWC uint8 NumPy arrays in RGB colour order.

        Returns:
            List of HWC uint8 RGB arrays, each of shape
            (output_size, output_size, 3), or None where no face was found.
        """
        pil_imgs = [Image.fromarray(f) for f in frames_rgb]

        # landmarks=True requests 5-point facial keypoints alongside boxes.
        # Returns: boxes_list, probs_list, landmarks_list  (each len == batch).
        boxes_list, _, landmarks_list = self.mtcnn.detect(pil_imgs, landmarks=True)

        # Guard: MTCNN should always return a list the same length as input,
        # but defensive handling avoids hard crashes on edge-case batch sizes.
        if landmarks_list is None:
            landmarks_list = [None] * len(frames_rgb)

        results: list[np.ndarray | None] = []

        for frame, box, lm in zip(frames_rgb, boxes_list, landmarks_list):
            if box is None:
                # MTCNN found no face in this frame
                results.append(None)
                continue

            face: np.ndarray | None = None

            # ── Preferred path: landmark-based affine alignment ───────────
            if lm is not None:
                try:
                    # lm shape: [n_faces, 5, 2]; index [0] is the top-1 face.
                    keypoints = lm[0].astype(np.float32)               # [5, 2]
                    face = self._align_face(frame, keypoints)
                except Exception:
                    face = None  # fall through to bounding-box path

            # ── Fallback path: margin-padded bounding-box crop ────────────
            if face is None:
                face = self._crop_bbox(frame, box[0])

            results.append(face)

        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _align_face(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,   # [5, 2] in (x, y) pixel order
    ) -> np.ndarray | None:
        """
        Compute a similarity transform mapping detected landmarks onto the
        canonical reference positions, then warp the frame into a
        output_size × output_size canvas.

        Args:
            frame:     HWC uint8 RGB source frame (full resolution).
            landmarks: [5, 2] float32 detected keypoints in (x, y) order.

        Returns:
            HWC uint8 RGB aligned crop of shape (output_size, output_size, 3),
            or None if the transformation matrix cannot be estimated.
        """
        # Scale reference points if output_size != 224 (e.g. for ablation runs
        # at 256 or 384 px without changing the canonical geometry).
        ref = _REF_LANDMARKS_224
        if self.output_size != 224:
            ref = ref * (self.output_size / 224.0)

        # LMEDS is more robust to occasional landmark regression outliers than
        # the default RANSAC estimator when using only 5 correspondences.
        M, _ = cv2.estimateAffinePartial2D(landmarks, ref, method=cv2.LMEDS)
        if M is None:
            return None

        aligned = cv2.warpAffine(
            frame, M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            # BORDER_REPLICATE fills edge pixels with the nearest valid value;
            # avoids black / green borders that could bias the FFT spectrum.
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned  # HWC uint8 RGB

    def _crop_bbox(
        self,
        frame: np.ndarray,
        box: np.ndarray,   # [x1, y1, x2, y2] float32 from MTCNN
    ) -> np.ndarray:
        """
        Fallback: expand the bounding box by `margin` on each side, clamp to
        frame boundaries, crop, and resize to output_size × output_size.

        Args:
            frame: HWC uint8 RGB source frame.
            box:   [x1, y1, x2, y2] bounding box from MTCNN (float32).

        Returns:
            HWC uint8 RGB face crop of shape (output_size, output_size, 3).
        """
        x1, y1, x2, y2 = (int(b) for b in box)
        h, w            = frame.shape[:2]
        bw, bh          = x2 - x1, y2 - y1

        # Expand box symmetrically so chin and forehead context are included;
        # deepfake boundaries often appear just outside the tight face box.
        x1 = max(0, int(x1 - bw * self.margin))
        y1 = max(0, int(y1 - bh * self.margin))
        x2 = min(w, int(x2 + bw * self.margin))
        y2 = min(h, int(y2 + bh * self.margin))

        crop = frame[y1:y2, x1:x2]
        return cv2.resize(
            crop,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LINEAR,
        )  # HWC uint8 RGB