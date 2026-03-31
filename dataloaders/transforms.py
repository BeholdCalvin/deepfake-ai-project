"""
dataloaders/transforms.py

Objective 1 – Robustness Against Compression:
  Training transforms inject heavy JPEG / WebP compression, Gaussian noise,
  motion blur, and down-scale augmentations.  This forces the model to rely on
  spatial blending artefacts (e.g. soft face boundaries) rather than high-
  frequency spectral fingerprints that social-media re-encoding destroys.

  Probability schedule:
    • 60 % chance of at least one degradation (ImageCompression OR Downscale)
    • 40 % chance of noise, 30 % of motion blur
    → The model sees clean faces ~15 % of the time; heavily degraded ~30 %.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics (EfficientNet-B4 expects these)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _image_compression(compression_type: str, p: float = 1.0) -> A.ImageCompression:
    """
    Build an ImageCompression transform compatible with Albumentations 1.x and 2.x.
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


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """
    Augmentation pipeline for training.
    Input:  HWC uint8 NumPy array (RGB face crop).
    Output: CHW float32 torch.Tensor (normalised).
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
        # At least one degradation path fires ~60 % of the time.
        A.OneOf([
            # JPEG compression simulates social-media re-encoding
            _image_compression("jpeg", p=1.0),
            # WebP compression – increasingly common on mobile platforms
            _image_compression("webp", p=1.0),
            # Downscale then resize up – mimics low-res source videos
            A.Downscale(
                scale_range=(0.4, 0.85),
                interpolation_pair={
                    "downscale": 1,   # INTER_LINEAR
                    "upscale":   3,   # INTER_CUBIC
                },
                p=1.0,
            ),
        ], p=0.65),

        # ── Sensor / transmission noise ───────────────────────────────────────
        A.GaussNoise(std_range=(0.02, 0.12), p=0.45),

        # ── Camera motion artefacts ───────────────────────────────────────────
        A.MotionBlur(blur_limit=(3, 9), p=0.30),

        # ── Colour / photometric jitter ───────────────────────────────────────
        A.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.12,
            p=0.45,
        ),
        A.ToGray(p=0.05),          # rare grayscale forces channel-agnostic features

        # ── Normalise & convert ───────────────────────────────────────────────
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),              # HWC → CHW, float32
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Deterministic pipeline for validation / inference.
    No augmentation – only resize and normalise.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
