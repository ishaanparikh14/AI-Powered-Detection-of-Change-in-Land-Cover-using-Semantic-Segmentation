"""
src/model.py — U-Net segmentation model.

Primary: segmentation_models_pytorch (smp) with a pretrained ResNet-34 encoder.
Fallback: hand-written U-Net (no external dependency) activated automatically
          if smp is not installed.

Interface contract (fixed — other modules depend on this):
    from src.model import UNet
    model = UNet(num_classes=5)          # in_channels=3
    logits = model(x)                    # x: (B,3,256,256) → (B,5,256,256)
    pred   = logits.argmax(dim=1)        # (B,256,256)  values 0..4

Save/load:
    torch.save(model.state_dict(), "checkpoints/best_model.pth")
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location="cpu"))
"""

from __future__ import annotations

import torch
import torch.nn as nn

# --- Inline defaults (avoids collision with config/ package) ---
# inference.py always passes explicit num_classes; these are only fallbacks.
IN_CHANNELS      = 3
NUM_CLASSES      = 5
ENCODER_NAME     = "resnet34"
ENCODER_WEIGHTS  = "imagenet"


# ---------------------------------------------------------------------------
# Try segmentation_models_pytorch first
# ---------------------------------------------------------------------------
try:
    import segmentation_models_pytorch as smp  # type: ignore

    class UNet(nn.Module):
        """
        Thin wrapper around smp.Unet so the rest of the codebase always
        imports `UNet` with a consistent interface.
        """

        def __init__(
            self,
            num_classes: int = NUM_CLASSES,
            in_channels: int = IN_CHANNELS,
            encoder_name: str = ENCODER_NAME,
            encoder_weights: str | None = ENCODER_WEIGHTS,
        ):
            super().__init__()
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,   # raw logits; softmax/argmax done outside
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    _BACKEND = "segmentation_models_pytorch"

except ImportError:
    # ------------------------------------------------------------------
    # Fallback: hand-written U-Net (no extra dependencies)
    # ------------------------------------------------------------------

    class _DoubleConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class _Down(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.pool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                _DoubleConv(in_ch, out_ch),
            )

        def forward(self, x):
            return self.pool_conv(x)

    class _Up(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = _DoubleConv(in_ch, out_ch)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            # Pad x1 to match x2 if sizes differ (odd input dimensions)
            dh = x2.size(2) - x1.size(2)
            dw = x2.size(3) - x1.size(3)
            if dh > 0 or dw > 0:
                import torch.nn.functional as F
                x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class UNet(nn.Module):
        """
        Classic U-Net (no pretrained encoder).
        Channels: 64→128→256→512→1024 (bottleneck).
        """

        def __init__(
            self,
            num_classes: int = NUM_CLASSES,
            in_channels: int = IN_CHANNELS,
            **_kwargs,   # absorb unused encoder_name / encoder_weights
        ):
            super().__init__()
            self.inc   = _DoubleConv(in_channels, 64)
            self.down1 = _Down(64,  128)
            self.down2 = _Down(128, 256)
            self.down3 = _Down(256, 512)
            self.down4 = _Down(512, 1024)
            self.up1   = _Up(1024, 512)
            self.up2   = _Up(512,  256)
            self.up3   = _Up(256,  128)
            self.up4   = _Up(128,   64)
            self.outc  = nn.Conv2d(64, num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x  = self.up1(x5, x4)
            x  = self.up2(x,  x3)
            x  = self.up3(x,  x2)
            x  = self.up4(x,  x1)
            return self.outc(x)

    _BACKEND = "fallback (hand-written UNet — install segmentation_models_pytorch for ResNet-34 encoder)"


def build_model(num_classes: int = NUM_CLASSES) -> UNet:
    """Convenience factory used in train.py and inference."""
    print(f"[model] Backend: {_BACKEND}")
    return UNet(num_classes=num_classes)
