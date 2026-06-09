"""
eval_full.py  — Evaluate checkpoints/full_model.pth
The file uses a different U-Net architecture (dconv_down1 keys),
so we define a matching architecture inline and evaluate it.
"""
import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.insert(0, ".")
from src.dataset import DeepGlobeDataset, get_image_paths
from ml_config import (
    NUM_CLASSES, IGNORE_INDEX, SEED, CLASS_NAMES,
    CLASS_COLORS, IMAGENET_MEAN, IMAGENET_STD,
)
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Architecture that matches the dconv_down / dconv_up key naming ───────────
class ResBlock(nn.Module):
    """Conv-BN-ReLU x2 + skip connection (matches .conv + .skip keys)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class ExternalUNet(nn.Module):
    """
    Residual U-Net with 7 output classes and a bottleneck block.
    Reconstructed from full_model.pth key names and shapes.
    conv_last shape [7, 64] → 7-class model (original DeepGlobe 7 classes).
    """
    def __init__(self, n_class=7):
        super().__init__()
        # Encoder
        self.dconv_down1 = ResBlock(3,   64)
        self.dconv_down2 = ResBlock(64,  128)
        self.dconv_down3 = ResBlock(128, 256)
        self.dconv_down4 = ResBlock(256, 512)

        # Bottleneck
        self.bottleneck = ResBlock(512, 512)

        # Transposed conv bridges (dconv1..4)
        self.dconv1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dconv2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dconv3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dconv4 = nn.ConvTranspose2d(64,   64, 2, stride=2)

        # Decoder (input = upsample + skip → concat)
        self.dconv_up4 = ResBlock(512 + 512, 256)
        self.dconv_up3 = ResBlock(256 + 256, 128)
        self.dconv_up2 = ResBlock(128 + 128,  64)
        self.dconv_up1 = ResBlock(64  +  64,  64)

        self.maxpool  = nn.MaxPool2d(2)
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        c1 = self.dconv_down1(x)
        c2 = self.dconv_down2(self.maxpool(c1))
        c3 = self.dconv_down3(self.maxpool(c2))
        c4 = self.dconv_down4(self.maxpool(c3))
        bn = self.bottleneck(self.maxpool(c4))

        u4 = self.dconv_up4(torch.cat([self.dconv1(bn), c4], dim=1))
        u3 = self.dconv_up3(torch.cat([self.dconv2(u4), c3], dim=1))
        u2 = self.dconv_up2(torch.cat([self.dconv3(u3), c2], dim=1))
        u1 = self.dconv_up1(torch.cat([self.dconv4(u2), c1], dim=1))
        return self.conv_last(u1)


# ── Load ─────────────────────────────────────────────────────────────────────
# full_model.pth has 7 output classes (original DeepGlobe labelling).
# We evaluate on the 7-class raw predictions then remap to compare.
model = ExternalUNet(n_class=7)
state = torch.load("checkpoints/full_model.pth", map_location=device)
model.load_state_dict(state)
model = model.to(device)
model.eval()
print(f"[eval] Loaded  full_model.pth  ({sum(p.numel() for p in model.parameters()):,} params)  ->  {device}")
print("[eval] This model outputs 7 classes (original DeepGlobe).")
print("[eval] Remapping to our 5-class scheme for fair comparison.")
print()

# DeepGlobe 7-class index -> our 5-class index mapping
# Original class order in Kaggle kernels typically:
# 0=urban, 1=agriculture, 2=rangeland, 3=forest, 4=water, 5=barren, 6=unknown
# Rangeland(2) -> Agriculture(2), unknown(6) -> ignore(255)
REMAP_7_TO_5 = {0: 1, 1: 2, 2: 2, 3: 0, 4: 3, 5: 4, 6: 255}

# ── Val split ─────────────────────────────────────────────────────────────────
all_paths = get_image_paths()
rng       = np.random.default_rng(SEED)
indices   = rng.permutation(len(all_paths))
n_val     = max(1, int(len(all_paths) * 0.2))
val_paths = [all_paths[i] for i in indices[:n_val]]

val_ds     = DeepGlobeDataset(val_paths, augment=False)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

# ── Inference ─────────────────────────────────────────────────────────────────
all_preds, all_targets = [], []
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs  = imgs.to(device)
        raw   = model(imgs).argmax(dim=1).cpu()  # 7-class predictions
        # Remap 7 -> 5
        remapped = torch.full_like(raw, 255)
        for src, dst in REMAP_7_TO_5.items():
            remapped[raw == src] = dst
        all_preds.append(remapped)
        all_targets.append(masks)

all_preds   = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

valid       = all_targets != IGNORE_INDEX
pv          = all_preds[valid]
tv          = all_targets[valid]
total_valid = valid.sum().item()

# ── Metrics ───────────────────────────────────────────────────────────────────
SEP = "=" * 62
print()
print(SEP)
print(f"  full_model.pth  —  {len(val_paths)} val images   |   {device}")
print(SEP)
print(f"  {'Class':<15}  {'IoU':>7}   {'GT %':>6}   {'Pred %':>7}")
print("-" * 62)

ious = []
for c, name in enumerate(CLASS_NAMES):
    inter  = int(((pv == c) & (tv == c)).sum())
    union  = int(((pv == c) | (tv == c)).sum())
    gt_pct = int((tv == c).sum()) / total_valid * 100
    pr_pct = int((pv == c).sum()) / total_valid * 100
    iou    = inter / union if union > 0 else float("nan")
    ious.append(iou)
    flag   = "  <<< low" if (not np.isnan(iou) and iou < 0.5) else ""
    print(f"  {name:<15}  {iou:>7.4f}   {gt_pct:>5.1f}%   {pr_pct:>6.1f}%{flag}")

mean_iou = float(np.nanmean(ious))
print("-" * 62)
print(f"  {'mean IoU':<15}  {mean_iou:>7.4f}")
print(SEP)

correct = int((pv == tv).sum())
acc     = correct / total_valid * 100
print(f"  Pixel accuracy : {acc:.2f}%")
print()

print(f"  {'Class':<15}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
print("-" * 62)
for c, name in enumerate(CLASS_NAMES):
    tp   = int(((pv == c) & (tv == c)).sum())
    fp   = int(((pv == c) & (tv != c)).sum())
    fn   = int(((pv != c) & (tv == c)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"  {name:<15}  {prec:>10.4f}  {rec:>8.4f}  {f1:>8.4f}")
print(SEP)

# ── Side-by-side comparison summary ──────────────────────────────────────────
print()
print("  COMPARISON vs best_model.pth (0.7140 mIoU)")
print("-" * 62)
best = {"Forest":0.7490,"Urban":0.7064,"Agriculture":0.8500,"Water":0.6809,"Barren":0.5840}
for c, name in enumerate(CLASS_NAMES):
    diff = ious[c] - best[name]
    arrow = "↑" if diff > 0.002 else ("↓" if diff < -0.002 else "~")
    print(f"  {name:<15}  full:{ious[c]:.4f}   best:{best[name]:.4f}   {arrow} {diff:+.4f}")
print("-" * 62)
print(f"  {'mean IoU':<15}  full:{mean_iou:.4f}   best:0.7140   {'↑' if mean_iou>0.7140 else '↓'} {mean_iou-0.7140:+.4f}")
print(SEP)
print()
print("Done.")
