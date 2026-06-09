"""
src/train.py — Training script for Module A.

Run:
    python src/train.py                        # uses defaults from config.py
    python src/train.py --epochs 30 --batch 4  # override on the fly

Outputs:
    checkpoints/best_model.pth  — state_dict of the epoch with best val mIoU
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sure project root is on the path (works whether you run from root or src/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_config import (
    BATCH_SIZE, CHECKPOINT_DIR, BEST_MODEL_PATH,
    EPOCHS, IGNORE_INDEX, LR, NUM_CLASSES, NUM_WORKERS, SEED,
)
from src.dataset import DeepGlobeDataset, get_image_paths
from src.loss    import CombinedLoss
from src.model   import build_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metric: mean IoU (excludes ignore_index pixels)
# ---------------------------------------------------------------------------
def compute_mean_iou(
    preds:   torch.Tensor,   # (B, H, W) long — argmax predictions
    targets: torch.Tensor,   # (B, H, W) long
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Returns mean IoU as a Python float."""
    ious = []
    valid_mask = targets != ignore_index
    preds_v   = preds[valid_mask]
    targets_v = targets[valid_mask]

    for cls in range(num_classes):
        pred_cls   = preds_v   == cls
        target_cls = targets_v == cls
        inter = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            # Class absent in this batch — skip rather than penalise
            continue
        ious.append(inter / union)

    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model:      torch.nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  torch.nn.Module,
    device:     torch.device,
    epoch:      int,
    total:      int,
) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [train]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)           # (B, C, H, W)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model:     torch.nn.Module,
    loader:    DataLoader,
    criterion: torch.nn.Module,
    device:    torch.device,
    epoch:     int,
    total:     int,
) -> tuple[float, float]:
    """Returns (val_loss, val_mIoU)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [val]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)     # (B, H, W)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    miou = compute_mean_iou(all_preds, all_targets)

    return running_loss / len(loader.dataset), miou


# ---------------------------------------------------------------------------
# Quick visual sanity check (saved to checkpoints/sample_pred.png)
# ---------------------------------------------------------------------------
def save_sample_prediction(
    model:  torch.nn.Module,
    dataset: DeepGlobeDataset,
    device: torch.device,
    save_path: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")           # headless — safe on Colab / server
        import matplotlib.pyplot as plt
        from ml_config import CLASS_NAMES, CLASS_COLORS

        model.eval()
        with torch.no_grad():
            image_t, mask_t = dataset[0]
            logits  = model(image_t.unsqueeze(0).to(device))
            pred    = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            mask_np = mask_t.numpy()

        # De-normalise image for display
        from ml_config import IMAGENET_MEAN, IMAGENET_STD
        img_np = image_t.permute(1, 2, 0).numpy()
        img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1)

        def colorise(index_map: np.ndarray) -> np.ndarray:
            h, w = index_map.shape
            rgb  = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, color in enumerate(CLASS_COLORS):
                rgb[index_map == idx] = color
            return rgb

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np);            axes[0].set_title("Image");     axes[0].axis("off")
        axes[1].imshow(colorise(mask_np)); axes[1].set_title("GT mask");   axes[1].axis("off")
        axes[2].imshow(colorise(pred));    axes[2].set_title("Prediction"); axes[2].axis("off")

        # Legend
        from matplotlib.patches import Patch
        legend = [
            Patch(color=[c / 255 for c in color], label=CLASS_NAMES[i])
            for i, color in enumerate(CLASS_COLORS)
        ]
        axes[2].legend(handles=legend, loc="lower right", fontsize=7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[train] Sample prediction saved → {save_path}")
    except Exception as exc:
        print(f"[train] (Could not save sample prediction: {exc})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepGlobe U-Net (Module A)")
    parser.add_argument("--epochs",    type=int,   default=EPOCHS)
    parser.add_argument("--batch",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--workers",   type=int,   default=NUM_WORKERS)
    parser.add_argument("--seed",      type=int,   default=SEED)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data used for validation (default 0.2)")
    parser.add_argument("--resume",    type=str,   default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/best_model.pth)")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    print(f"[train] Device : {device}")
    print(f"[train] Epochs : {args.epochs}  |  Batch : {args.batch}  |  LR : {args.lr}")

    # ----- Data ---------------------------------------------------------------
    all_paths = get_image_paths()
    print(f"[train] Found {len(all_paths)} image/mask pairs in dataset")

    rng = np.random.default_rng(args.seed)
    indices  = rng.permutation(len(all_paths))
    n_val    = max(1, int(len(all_paths) * args.val_split))
    val_idx  = indices[:n_val]
    train_idx = indices[n_val:]

    train_paths = [all_paths[i] for i in train_idx]
    val_paths   = [all_paths[i] for i in val_idx]
    print(f"[train] Split  : {len(train_paths)} train / {len(val_paths)} val")

    train_ds = DeepGlobeDataset(train_paths, augment=True)
    val_ds   = DeepGlobeDataset(val_paths,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    # ----- Model, optimiser, loss ---------------------------------------------
    model     = build_model(num_classes=NUM_CLASSES).to(device)

    # Resume from checkpoint if requested
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"[train] Resumed weights from: {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = CombinedLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ----- Training loop ------------------------------------------------------
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        val_loss, val_miou = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )
        scheduler.step()

        # Checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            ckpt_marker = "  ← best"
        else:
            ckpt_marker = ""

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss: {train_loss:.4f}  "
            f"val_loss: {val_loss:.4f}  "
            f"val_mIoU: {val_miou:.4f}"
            f"{ckpt_marker}"
        )

    print(f"\n[train] Training complete. Best val mIoU: {best_miou:.4f}")
    print(f"[train] Best checkpoint saved to: {BEST_MODEL_PATH}")

    # ----- Sanity-check: one prediction image ---------------------------------
    sample_path = os.path.join(CHECKPOINT_DIR, "sample_pred.png")
    save_sample_prediction(model, val_ds, device, sample_path)


if __name__ == "__main__":
    main()
