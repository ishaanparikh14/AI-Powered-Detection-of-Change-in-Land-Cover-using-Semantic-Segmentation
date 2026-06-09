"""
train_full.py  — Train/fine-tune full_model.pth architecture.

Architecture: Residual U-Net (7-class → adapted to 5-class)
Exactly matches the key names and channel widths in full_model.pth.

Usage:
    # Fine-tune from full_model.pth (recommended):
    python train_full.py --resume checkpoints/full_model.pth --epochs 30

    # Train from scratch:
    python train_full.py --epochs 50
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import DeepGlobeDataset, get_image_paths
from src.loss    import CombinedLoss
from ml_config import (
    BATCH_SIZE, CHECKPOINT_DIR, IGNORE_INDEX,
    EPOCHS, NUM_CLASSES, NUM_WORKERS, SEED, LR,
    CLASS_NAMES,
)

# ── Exact architecture matching full_model.pth key names + shapes ─────────────
class ResBlock(nn.Module):
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


class ResUNet(nn.Module):
    """
    Residual U-Net — matches full_model.pth key names exactly.
    n_class=5  for our 5-class training.
    n_class=7  to load original full_model.pth weights then swap head.
    """
    def __init__(self, n_class=5):
        super().__init__()
        # Encoder
        self.dconv_down1 = ResBlock(3,    64)
        self.dconv_down2 = ResBlock(64,  128)
        self.dconv_down3 = ResBlock(128, 256)
        self.dconv_down4 = ResBlock(256, 512)

        # Bottleneck  (512 → 1024)
        self.bottleneck  = ResBlock(512, 1024)

        # Transpose conv bridges (halve channels)
        self.dconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dconv2 = nn.ConvTranspose2d(512,  256, 2, stride=2)
        self.dconv3 = nn.ConvTranspose2d(256,  128, 2, stride=2)
        self.dconv4 = nn.ConvTranspose2d(128,   64, 2, stride=2)

        # Decoder  (upsample + skip → concat)
        self.dconv_up4 = ResBlock(512 + 512,  512)
        self.dconv_up3 = ResBlock(256 + 256,  256)
        self.dconv_up2 = ResBlock(128 + 128,  128)
        self.dconv_up1 = ResBlock( 64 +  64,   64)

        self.maxpool   = nn.MaxPool2d(2)
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


def build_model_from_full(path: str) -> ResUNet:
    """
    Load full_model.pth (7-class) into ResUNet,
    then replace the final conv_last with a fresh 5-class head.
    Encoder + decoder weights are reused; only the head is re-initialised.
    """
    # 1. Load into 7-class model (exact weight match)
    model7 = ResUNet(n_class=7)
    state  = torch.load(path, map_location="cpu")
    model7.load_state_dict(state)
    print(f"[model] Loaded 7-class weights from {path}")

    # 2. Swap head to 5-class
    model7.conv_last = nn.Conv2d(64, 5, 1)
    nn.init.kaiming_normal_(model7.conv_last.weight)
    nn.init.zeros_(model7.conv_last.bias)
    print("[model] Replaced conv_last  7 → 5 classes (head re-initialised)")
    return model7


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_mean_iou(preds, targets, num_classes=NUM_CLASSES,
                     ignore_index=IGNORE_INDEX):
    valid = targets != ignore_index
    pv, tv = preds[valid], targets[valid]
    ious = []
    for c in range(num_classes):
        inter = ((pv == c) & (tv == c)).sum().item()
        union = ((pv == c) | (tv == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ── Train / val loops ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [train]", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total):
    model.eval()
    running = 0.0
    all_p, all_t = [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [val]  ", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        running += criterion(logits, masks).item() * imgs.size(0)
        all_p.append(logits.argmax(1).cpu())
        all_t.append(masks.cpu())
    miou = compute_mean_iou(torch.cat(all_p), torch.cat(all_t))
    return running / len(loader.dataset), miou


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--batch",     type=int,   default=2)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--workers",   type=int,   default=0)
    p.add_argument("--seed",      type=int,   default=SEED)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--resume",    type=str,   default=None,
                   help="Path to full_model.pth (7-class) to resume from")
    p.add_argument("--out",       type=str,
                   default="checkpoints/full_model_5class.pth",
                   help="Where to save the best checkpoint")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    print(f"[train] Device : {device}")
    print(f"[train] Epochs : {args.epochs}  Batch : {args.batch}  LR : {args.lr}")

    # Data
    all_paths = get_image_paths()
    rng       = np.random.default_rng(args.seed)
    idx       = rng.permutation(len(all_paths))
    n_val     = max(1, int(len(all_paths) * args.val_split))
    val_paths   = [all_paths[i] for i in idx[:n_val]]
    train_paths = [all_paths[i] for i in idx[n_val:]]
    print(f"[train] Split  : {len(train_paths)} train / {len(val_paths)} val")

    train_ds = DeepGlobeDataset(train_paths, augment=True)
    val_ds   = DeepGlobeDataset(val_paths,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type=="cuda"),
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type=="cuda"))

    # Model
    if args.resume:
        model = build_model_from_full(args.resume)
    else:
        model = ResUNet(n_class=5)
        print("[model] Training ResUNet from scratch (5 classes)")
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")

    # Optimiser — lower LR for pretrained encoder, higher for new head
    head_params = list(model.conv_last.parameters())
    head_ids    = {id(p) for p in head_params}
    base_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.Adam([
        {"params": base_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.001
    )
    criterion = CombinedLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss          = train_one_epoch(model, train_loader, optimizer,
                                              criterion, device, epoch, args.epochs)
        val_loss, val_miou  = validate(model, val_loader, criterion,
                                       device, epoch, args.epochs)
        scheduler.step()

        marker = ""
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), args.out)
            marker = "  <- best"

        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"train_loss: {train_loss:.4f}  "
              f"val_loss: {val_loss:.4f}  "
              f"val_mIoU: {val_miou:.4f}{marker}")

    print(f"\n[train] Done.  Best val mIoU: {best_miou:.4f}")
    print(f"[train] Saved → {args.out}")


if __name__ == "__main__":
    main()
