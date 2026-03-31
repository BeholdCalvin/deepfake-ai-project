"""
models/fusion.py

Architecture: Spatial (EfficientNet-B4) + Frequency (FFT-CNN) fusion with
              optional LSTM temporal modelling.

Key design decisions:
  - EfficientNet-B4 via `timm` replaces the custom CNN backbone. It provides
    1792-dim features pretrained on ImageNet, giving us a rich spatial prior.
  - FFTBranch converts each frame to a log-magnitude spectrum and extracts
    512-dim frequency features via a small CNN.  Compression artefacts show up
    as ring patterns in the spectrum that spatial CNNs miss.
  - Fusion: concat(1792 + 512) → Linear(2304→512) → LSTM(T steps) → head.
  - Dynamic single-frame path: if the input has T=1 (or is a raw 4-D tensor),
    the LSTM is bypassed entirely, so the model works for both images and videos.
  - `gradcam_target_layer` is exposed so the predict / app scripts can hook
    Grad-CAM without hardcoding layer names.
"""

import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# Frequency Branch
# ---------------------------------------------------------------------------

class FFTBranch(nn.Module):
    """
    Converts an RGB frame to a 2-D FFT log-magnitude map and extracts
    discriminative frequency features.  Deepfake generators leave spectral
    fingerprints (e.g. grid artefacts from up-sampling) that survive JPEG
    compression better than pixel-level blending artefacts.
    """

    def __init__(self, out_features: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),                                      # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),                                      # /4

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),                         # → 4×4
        )
        self.fc = nn.Linear(128 * 4 * 4, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, 3, H, W]  (normalised RGB)
        gray = x.mean(dim=1, keepdim=True)                        # [B,1,H,W]
        fft_complex = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft_complex)
        magnitude = torch.log(torch.abs(fft_shifted) + 1e-8)     # [B,1,H,W]

        # per-sample normalisation keeps gradients stable across compressions
        mu = magnitude.mean(dim=[-2, -1], keepdim=True)
        sigma = magnitude.std(dim=[-2, -1], keepdim=True) + 1e-8
        magnitude = (magnitude - mu) / sigma

        feat = self.conv(magnitude)                               # [B,128,4,4]
        feat = self.fc(feat.flatten(1))                           # [B, out_features]
        return feat


# ---------------------------------------------------------------------------
# Main Detector
# ---------------------------------------------------------------------------

class DeepfakeDetector(nn.Module):
    """
    Dual-stream detector:

        Input  →  [EfficientNet-B4 spatial stream]  →  1792-d
               →  [FFT frequency stream]            →   512-d
               →  concat → Linear → (optional LSTM) → classifier
    """

    SPATIAL_DIM = 1792   # EfficientNet-B4 penultimate feature size
    FFT_DIM     = 512
    FUSED_DIM   = SPATIAL_DIM + FFT_DIM   # 2304

    def __init__(
        self,
        sequence_length: int = 8,
        hidden_size: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.sequence_length = sequence_length

        # ── Spatial backbone ────────────────────────────────────────────────
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0,          # strip the classification head
            global_pool="avg",      # returns [B, 1792]
        )

        # ── Frequency branch ────────────────────────────────────────────────
        self.fft_branch = FFTBranch(out_features=self.FFT_DIM)

        # ── Fusion projection ───────────────────────────────────────────────
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.FUSED_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Temporal model (used only for T > 1) ────────────────────────────
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ── Head ─────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Expose the layer Grad-CAM should hook into (last conv before global pool)
        self.gradcam_target_layer = self.backbone.conv_head

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a flat batch of frames independently.
        x : [B, C, H, W]
        returns : [B, hidden_size]
        """
        spatial = self.backbone(x)                                # [B, 1792]
        freq    = self.fft_branch(x)                              # [B, 512]
        fused   = torch.cat([spatial, freq], dim=1)               # [B, 2304]
        return self.fusion_fc(fused)                              # [B, hidden]

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          • [B, C, H, W]          → single-image path  (T bypass)
          • [B, T, C, H, W]       → video sequence path (LSTM when T > 1)
        Returns logits [B].
        """
        if x.dim() == 4:
            # ── Single image input ──────────────────────────────────────────
            feat   = self._extract_frame_features(x)              # [B, hidden]
            logits = self.classifier(feat)                        # [B, 1]
            return logits.squeeze(1)

        B, T, C, H, W = x.shape

        # ── Video sequence input ────────────────────────────────────────────
        x_flat    = x.view(B * T, C, H, W)
        feat_flat = self._extract_frame_features(x_flat)          # [B*T, hidden]
        feat_seq  = feat_flat.view(B, T, -1)                      # [B, T, hidden]

        if T == 1:
            # Single temporal step → bypass LSTM, use features directly
            out = feat_seq.squeeze(1)                             # [B, hidden]
        else:
            lstm_out, _ = self.lstm(feat_seq)                     # [B, T, hidden]
            out = lstm_out[:, -1, :]                              # last timestep

        logits = self.classifier(out)                             # [B, 1]
        return logits.squeeze(1)
