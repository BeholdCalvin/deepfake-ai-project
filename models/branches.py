import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class PixelBranch(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        laplacian = torch.tensor([[[[-1., -1., -1.],
                                    [-1.,  8., -1.],
                                    [-1., -1., -1.]]]], dtype=torch.float32)
        self.register_buffer('filter', laplacian.repeat(3, 1, 1, 1) / 3.0) 
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        residual = F.conv2d(x, self.filter, groups=3, padding=1)
        return self.net(residual)

class FrequencyBranch(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        fft_complex = torch.fft.fft2(x)
        # BUG FIX: Add dim=(-2, -1) to only shift spatial dimensions
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        # Add 1e-8 to prevent log(0) which causes NaN gradients
        log_magnitude = torch.log(torch.abs(fft_shifted) + 1e-8) 
        return self.net(log_magnitude)

class SpatialBranch(nn.Module):
    def __init__(self, out_features=256):
        super().__init__()
        # Use pretrained EfficientNet for spatial features
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Remove classifier head
        self.backbone.classifier = nn.Identity() 
        self.fc = nn.Linear(1280, out_features) # 1280 is effnet_b0 output dim

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)