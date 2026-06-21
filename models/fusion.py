"""
models/fusion.py

Architecture: EfficientNet-B4 (spatial) + FFT-CNN (frequency) fusion with
              optional LSTM temporal modelling.

Key design decisions
────────────────────
• EfficientNet-B4 via `timm`: 1792-d ImageNet-pretrained spatial prior.
• FFTBranch: log-magnitude spectrum exposes GAN up-sampling grid artefacts
  that survive JPEG re-encoding and are invisible to spatial CNNs.
• Fusion: concat(1792 + 512) → Linear(2304→512) → LSTM(T steps) → head.
• Dynamic single-frame path: 4-D input bypasses the LSTM entirely so the
  same model handles still images and video sequences.
• Two Grad-CAM anchor layers are exposed:
    gradcam_spatial_layer  – last conv in EfficientNet-B4 (face-region heatmap)
    gradcam_fft_layer      – last conv in FFTBranch     (spectral-artefact heatmap)
"""

import torch
import torch.nn as nn
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Frequency Branch
# ─────────────────────────────────────────────────────────────────────────────

class FFTBranch(nn.Module):
    """
    Converts an RGB frame to a 2-D FFT log-magnitude map and extracts
    discriminative frequency features.

    Design rationale
    ────────────────
    • Deepfake generators (GAN up-sampling, face-swap blending) leave spectral
      fingerprints – periodic grid artefacts visible as ring / cross patterns
      in the centred FFT magnitude – that survive JPEG re-encoding better than
      pixel-level blending cues.
    • We convert to grayscale *before* the FFT so that channel correlations
      do not dominate; luminance carries the dominant structural frequencies.
    • fftshift is applied over spatial dims only (dim=(-2,-1)) to avoid
      permuting the batch / channel dimensions – a subtle but crashy bug.
    • Per-sample normalisation of log-magnitude keeps gradients stable when
      mixing pristine-quality frames with heavily-compressed ones.
    """

    def __init__(self, out_features: int = 512) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # ── Block 1: 224×224 → 112×112 ──────────────────────────────────
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),    # [0]
            nn.BatchNorm2d(32),                                          # [1]
            nn.GELU(),                                                   # [2]
            nn.MaxPool2d(2),                                             # [3]

            # ── Block 2: 112×112 → 56×56 ────────────────────────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),    # [4]
            nn.BatchNorm2d(64),                                          # [5]
            nn.GELU(),                                                   # [6]
            nn.MaxPool2d(2),                                             # [7]

            # ── Block 3: 56×56 spatial map (Grad-CAM anchor) ────────────────
            # This is the last Conv2d before global pooling; hooking here
            # gives a 56×56 activation map → upsampled to 224×224 by CAM.
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),   # [8] ← CAM target
            nn.BatchNorm2d(128),                                         # [9]
            nn.GELU(),                                                   # [10]
            nn.AdaptiveAvgPool2d((4, 4)),                                # [11]
        )
        self.fc = nn.Linear(128 * 4 * 4, out_features)

        # Expose the last spatial conv for external Grad-CAM hooks so
        # callers never need to hard-code the Sequential index.
        #
        # WHY object.__setattr__ instead of self.gradcam_target = ...?
        # nn.Module.__setattr__ intercepts every assignment where the value
        # is an nn.Module and registers it as a NAMED CHILD in self._modules.
        # That creates a second path to the same weight tensors in state_dict
        # (e.g. "fft_branch.gradcam_target.weight" alongside the real path
        # "fft_branch.conv.8.weight").  Old checkpoints trained without these
        # aliases will then fail load_state_dict with "missing keys".
        # object.__setattr__ bypasses the hook and stores a plain Python
        # reference in the instance __dict__ that is invisible to state_dict
        # serialisation but still fully accessible as an attribute.
        object.__setattr__(self, 'gradcam_target', self.conv[8])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] ImageNet-normalised RGB frame tensor.
        Returns:
            [B, out_features] frequency feature vector.
        """
        gray        = x.mean(dim=1, keepdim=True)                      # [B, 1, H, W]
        fft_complex = torch.fft.fft2(gray)
        # Shift low frequencies to the centre. dim=(-2,-1) is critical –
        # omitting it shifts the batch dimension and produces random NaNs.
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        # +1e-8 prevents log(0) which produces -inf and kills gradients.
        magnitude   = torch.log(torch.abs(fft_shifted) + 1e-8)         # [B, 1, H, W]

        # Per-sample z-score normalisation: keeps the dynamic range
        # consistent across mixed-quality / mixed-codec video sources.
        mu        = magnitude.mean(dim=[-2, -1], keepdim=True)
        sigma     = magnitude.std(dim=[-2, -1], keepdim=True) + 1e-8
        magnitude = (magnitude - mu) / sigma

        feat = self.conv(magnitude)                                     # [B, 128, 4, 4]
        return self.fc(feat.flatten(1))                                 # [B, out_features]


# ─────────────────────────────────────────────────────────────────────────────
# Main Detector
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    Dual-stream detector:

        Input  →  [EfficientNet-B4 spatial stream]  →  1792-d
               →  [FFT frequency stream]            →   512-d
               →  concat → Linear → (optional LSTM) → classifier

    Sequence handling
    ─────────────────
    • [B, C, H, W]      → single-image path; LSTM bypassed entirely.
    • [B, T, C, H, W]   → video path; LSTM used when T > 1, bypassed when T == 1.

    Grad-CAM anchors
    ────────────────
    Two target layers are exposed for dual-branch explainability:
    • gradcam_spatial_layer : EfficientNet-B4 conv_head (last depthwise conv
                              before global-average-pool); produces a spatially-
                              detailed face-region heatmap.
    • gradcam_fft_layer     : FFTBranch.conv[8] (last Conv2d before GAP);
                              highlights spectral-artefact regions of the face.
    • gradcam_target_layer  : backward-compat alias → gradcam_spatial_layer.
    """

    SPATIAL_DIM: int = 1792                        # EfficientNet-B4 pool output
    FFT_DIM:     int = 512
    FUSED_DIM:   int = SPATIAL_DIM + FFT_DIM       # 2304

    def __init__(
        self,
        sequence_length: int = 8,
        hidden_size: int     = 512,
        lstm_layers: int     = 2,
        dropout: float       = 0.4,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length

        # ── Spatial backbone ─────────────────────────────────────────────────
        # pretrained=True: ImageNet weights provide a rich spatial prior that
        # reduces the labelled data needed to learn subtle blending artefacts.
        # num_classes=0 + global_pool="avg" strips the head → [B, 1792].
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # ── Frequency branch ─────────────────────────────────────────────────
        self.fft_branch = FFTBranch(out_features=self.FFT_DIM)

        # ── Fusion projection ─────────────────────────────────────────────────
        # LayerNorm instead of BatchNorm: batch sizes can be very small (e.g. 1)
        # when sampling long video sequences on memory-constrained GPUs.
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.FUSED_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Temporal model ───────────────────────────────────────────────────
        # Unidirectional LSTM: causal design allows future extension to real-time
        # streaming without architectural changes.  Hidden/cell states are
        # zero-initialised by PyTorch on the same device as the input tensor,
        # so no explicit device management is required.
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),   # single logit; apply sigmoid at inference
        )

        # ── Grad-CAM target layers ────────────────────────────────────────────
        # All stored via object.__setattr__ to bypass nn.Module.__setattr__.
        # nn.Module.__setattr__ intercepts any assignment where the value is an
        # nn.Module and registers it as a NAMED CHILD in self._modules, which
        # emits a duplicate key path in state_dict (e.g. the real weight lives
        # at "backbone.conv_head.weight" but a second entry would appear as
        # "gradcam_spatial_layer.weight").  Old checkpoints lack these alias
        # keys, so load_state_dict raises "Missing key(s) in state_dict".
        # object.__setattr__ writes directly into instance __dict__: the attr
        # is fully accessible as model.X but is invisible to state_dict, so
        # existing checkpoints load without modification.
        #
        # Spatial: last depthwise conv before global-avg-pool in EfficientNet-B4.
        object.__setattr__(self, 'gradcam_spatial_layer', self.backbone.conv_head)
        # Frequency: last Conv2d before AdaptiveAvgPool2d in FFTBranch (56x56).
        object.__setattr__(self, 'gradcam_fft_layer', self.fft_branch.gradcam_target)
        # Backward-compat alias for scripts that reference gradcam_target_layer.
        object.__setattr__(self, 'gradcam_target_layer', self.backbone.conv_head)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dual-stream feature extraction for a flat batch of frames.

        Args:
            x: [B, C, H, W] – may be B = batch × T when called from forward().
        Returns:
            [B, hidden_size] fused feature vectors.
        """
        spatial: torch.Tensor = self.backbone(x)                       # [B, 1792]
        freq:    torch.Tensor = self.fft_branch(x)                     # [B,  512]
        fused:   torch.Tensor = torch.cat([spatial, freq], dim=1)      # [B, 2304]
        return self.fusion_fc(fused)                                   # [B, hidden]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unified forward for both image and video inputs.

        Args:
            x: Either
               • [B, C, H, W]    – single image  (LSTM bypassed)
               • [B, T, C, H, W] – video sequence (LSTM active when T > 1)
        Returns:
            [B] raw logits.  Apply torch.sigmoid() for probabilities.
        """
        # ── Single-image path (4-D input) ─────────────────────────────────
        if x.dim() == 4:
            feat   = self._extract_frame_features(x)                   # [B, hidden]
            logits = self.classifier(feat)                             # [B, 1]
            return logits.squeeze(1)                                   # [B]

        # ── Video-sequence path (5-D input) ───────────────────────────────
        B, T, C, H, W = x.shape

        # Process all frames in parallel: flatten to [B*T, C, H, W], extract
        # features, then reshape back to [B, T, hidden].
        feat_flat = self._extract_frame_features(x.view(B * T, C, H, W))  # [B*T, hidden]
        feat_seq  = feat_flat.view(B, T, -1)                              # [B, T, hidden]

        if T == 1:
            # Single temporal step: bypass LSTM to avoid stateful padding
            # overhead and to stay consistent with the 4-D single-image path.
            out = feat_seq.squeeze(1)                                  # [B, hidden]
        else:
            # LSTM zero-initialises h0/c0 on the same device as feat_seq,
            # so no explicit .to(device) call is needed here.
            lstm_out, _ = self.lstm(feat_seq)                          # [B, T, hidden]
            out = lstm_out[:, -1, :]                                   # last timestep

        logits = self.classifier(out)                                  # [B, 1]
        return logits.squeeze(1)                                       # [B]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint compatibility
# ─────────────────────────────────────────────────────────────────────────────
#
# A transitional version of this file assigned Grad-CAM target-layer
# references as plain attributes, e.g.:
#     self.gradcam_target_layer = self.backbone.conv_head
# nn.Module.__setattr__ auto-registers any nn.Module-valued attribute as a
# NAMED CHILD, so any checkpoint saved while that version was active contains
# a SECOND, redundant copy of those weights under the alias path (e.g.
# "gradcam_target_layer.weight") in addition to the real path
# ("backbone.conv_head.weight").  The current model (using object.__setattr__,
# see __init__ above) has no such keys, so a strict load_state_dict() on an
# old checkpoint raises "Unexpected key(s)".
#
# The fix here is intentionally narrow: strip ONLY these known alias
# prefixes, then load with strict=True for everything else.  A blanket
# strict=False would also silently swallow genuine shape/architecture
# mismatches — exactly the kind of bug you want loud, not hidden.

_LEGACY_GRADCAM_ALIAS_PREFIXES: tuple[str, ...] = (
    "gradcam_target_layer.",
    "gradcam_spatial_layer.",
    "gradcam_fft_layer.",
    "fft_branch.gradcam_target.",
)


def strip_legacy_gradcam_alias_keys(state_dict: dict) -> dict:
    """
    Remove duplicate Grad-CAM alias weights left behind by the transitional
    buggy version of DeepfakeDetector (see module-level comment above).

    Safe no-op on checkpoints that never had these keys — every key with a
    different prefix passes through unchanged.  Returns a NEW dict; never
    mutates the input, so the raw checkpoint object remains untouched for
    debugging if needed.

    Args:
        state_dict: Raw dict loaded via torch.load(..., weights_only=True).

    Returns:
        A copy of state_dict with legacy alias keys removed.
    """
    cleaned: dict = {}
    removed: list[str] = []

    for key, value in state_dict.items():
        if key.startswith(_LEGACY_GRADCAM_ALIAS_PREFIXES):
            removed.append(key)
            continue
        cleaned[key] = value

    if removed:
        print(
            f"[DeepfakeDetector] Discarded {len(removed)} legacy Grad-CAM "
            f"alias key(s) from checkpoint (duplicate weights, safe to drop): "
            f"{removed}"
        )

    return cleaned


def load_checkpoint(
    model: "DeepfakeDetector",
    checkpoint_path: str,
    device: torch.device,
) -> "DeepfakeDetector":
    """
    Load a .pth checkpoint into `model` with legacy-alias compatibility.

    This is the SINGLE place checkpoint-loading logic lives; app.py,
    predict.py, and evaluate.py all call this instead of duplicating
    torch.load + load_state_dict, so a future fix to checkpoint handling
    only needs to happen once.

    Args:
        model:           A freshly-constructed DeepfakeDetector (correct
                         architecture; weights will be overwritten).
        checkpoint_path: Path to a .pth file saved via torch.save(model.state_dict()).
        device:          Device to map the checkpoint tensors onto.

    Returns:
        The same `model` instance, with weights loaded and set to eval().

    Raises:
        RuntimeError: if the checkpoint has any UNEXPECTED or MISSING key
                      beyond the known legacy aliases — a genuine
                      architecture mismatch should still fail loudly.
    """
    raw_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cleaned_state_dict = strip_legacy_gradcam_alias_keys(raw_state_dict)

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    return model