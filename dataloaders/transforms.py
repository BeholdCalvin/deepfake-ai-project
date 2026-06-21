"""
dataloaders/transforms.py

Objective 1 – Robustness Against Compression:
  Training transforms inject heavy JPEG / WebP compression, Gaussian noise,
  motion blur, and down-scale augmentations.  This forces the model to rely on
  spatial blending artefacts (e.g. soft face boundaries) rather than high-
  frequency spectral fingerprints that social-media re-encoding destroys.

  Probability schedule:
    • 65 % chance of at least one degradation (JPEG / WebP / Downscale)
    • 45 % chance of noise, 30 % of motion blur
    → The model sees clean faces ~10 % of the time; heavily degraded ~30 %.

Albumentations API compatibility:
  Several Albumentations transforms changed their keyword arguments between
  the 1.x and 2.x series.  Each affected transform has a thin factory
  function (_image_compression, _downscale, _gauss_noise) that tries the
  2.x call first and falls back to the 1.x equivalent on TypeError.
  This keeps the pipeline runnable on both library versions.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet channel statistics – EfficientNet-B4 was pretrained with these exact
# values.  Deviating shifts the backbone's feature distribution and slows or
# destabilises convergence.
_MEAN: list[float] = [0.485, 0.456, 0.406]
_STD:  list[float] = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Version-compatible transform factories
# ─────────────────────────────────────────────────────────────────────────────

def _image_compression(compression_type: str, p: float = 1.0) -> A.ImageCompression:
    """
    Build an ImageCompression transform compatible with Albumentations 1.x and 2.x.

    JPEG / WebP compression simulates the quality loss applied by social-media
    platforms (typical quality range 30–75), ensuring the model does not rely
    on pristine-frame spectral fingerprints that real-world re-sharing destroys.
    """
    try:
        # Albumentations 2.x
        return A.ImageCompression(
            quality_range=(20, 75),
            compression_type=compression_type,
            p=p,
        )
    except TypeError:
        # Albumentations 1.x
        image_type = (
            A.ImageCompression.ImageType.JPEG
            if compression_type.lower() == "jpeg"
            else A.ImageCompression.ImageType.WEBP
        )
        return A.ImageCompression(
            quality_lower=20,
            quality_upper=75,
            image_type=image_type,
            p=p,
        )


def _downscale(p: float = 1.0) -> A.Downscale:
    """
    Build a Downscale transform compatible with Albumentations 1.x and 2.x.

    Down-sample then up-sample to simulate low-resolution source artefacts
    (e.g. deepfakes rendered at 128 px then interpolated to 720p).
    """
    try:
        # Albumentations 2.x: scale_range + interpolation_pair dict
        return A.Downscale(
            scale_range=(0.4, 0.85),
            interpolation_pair={
                "downscale": 1,   # cv2.INTER_LINEAR
                "upscale":   3,   # cv2.INTER_CUBIC
            },
            p=p,
        )
    except TypeError:
        # Albumentations 1.x: scale_min / scale_max positional args
        return A.Downscale(scale_min=0.4, scale_max=0.85, p=p)


def _gauss_noise(p: float = 0.45) -> A.GaussNoise:
    """
    Build a GaussNoise transform compatible with Albumentations 1.x and 2.x.

    Noise injection forces the model to rely on structural manipulation
    artefacts (blending boundaries, texture mismatches) rather than clean
    high-frequency spectral differences that vanish in noisy conditions.
    """
    try:
        # Albumentations 2.x: std_range in normalised [0, 1] space
        return A.GaussNoise(std_range=(0.02, 0.12), p=p)
    except TypeError:
        # Albumentations 1.x: var_limit in uint8-squared space.
        # std ∈ [0.02, 0.12] on [0,1] images ≈ [5, 30] on [0,255] images
        # → var ≈ [25, 900] in uint8² units.
        return A.GaussNoise(var_limit=(25.0, 256.0), p=p)


# ─────────────────────────────────────────────────────────────────────────────
# Transform pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> A.Compose:
    """
    Augmentation pipeline for training.

    Designed to maximise robustness against the compression and noise
    conditions encountered when deepfake videos are shared on social media
    (Objective 1).

    Args:
        img_size: Square output side length in pixels.
                  Must match the EfficientNet-B4 input resolution (default 224).

    Input:  HWC uint8 NumPy array (RGB face crop).
    Output: CHW float32 torch.Tensor (ImageNet-normalised).
    """
    return A.Compose([
        # ── Geometry ─────────────────────────────────────────────────────────
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
            border_mode=0, p=0.4,
        ),

        # ── Compression / quality degradation (Objective 1 core) ─────────────
        # At least one degradation fires ~65 % of the time, covering the range
        # of quality loss applied by YouTube, Instagram, WhatsApp, etc.
        A.OneOf([
            _image_compression("jpeg"),      # social-media recompression
            _image_compression("webp"),      # increasingly common on mobile
            _downscale(),                    # low-res source video artefacts
        ], p=0.65),

        # ── Sensor / transmission noise ───────────────────────────────────────
        _gauss_noise(p=0.45),

        # ── Camera motion artefacts ───────────────────────────────────────────
        A.MotionBlur(blur_limit=(3, 9), p=0.30),

        # ── Colour / photometric jitter ───────────────────────────────────────
        A.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.12,
            p=0.45,
        ),
        # Rare grayscale conversion: forces the model to learn features that
        # do not depend on colour channels (important for B&W archival footage).
        A.ToGray(p=0.05),

        # ── Normalise and convert ─────────────────────────────────────────────
        # ImageNet stats are mandatory: EfficientNet-B4 BatchNorm statistics
        # were computed on ImageNet-normalised inputs; mismatched stats cause
        # systematic activation bias and slow convergence.
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),   # HWC uint8 → CHW float32
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Deterministic inference pipeline: resize, normalise, convert to tensor.

    No augmentation is applied to ensure fully reproducible evaluation metrics.
    The ImageNet normalisation matches get_train_transforms so the model
    receives identically distributed inputs at train and inference time.

    Args:
        img_size: Square output side length in pixels (must match training value).

    Input:  HWC uint8 NumPy array (RGB face crop).
    Output: CHW float32 torch.Tensor (ImageNet-normalised).
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])