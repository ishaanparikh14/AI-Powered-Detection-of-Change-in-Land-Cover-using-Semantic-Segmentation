"""
eval.py  — Full evaluation of best_model.pth on the val split.
Run:  python eval.py
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, ".")
from src.model   import UNet
from src.dataset import DeepGlobeDataset, get_image_paths
from ml_config import (
    NUM_CLASSES, IGNORE_INDEX, SEED, CLASS_NAMES,
    CLASS_COLORS, IMAGENET_MEAN, IMAGENET_STD,
)
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Val split (identical seed / ratio used during training) ─────────────────
all_paths = get_image_paths()
rng       = np.random.default_rng(SEED)
indices   = rng.permutation(len(all_paths))
n_val     = max(1, int(len(all_paths) * 0.2))
val_paths = [all_paths[i] for i in indices[:n_val]]

val_ds     = DeepGlobeDataset(val_paths, augment=False)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

# ── Load model ───────────────────────────────────────────────────────────────
model = UNet(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model = model.to(device)
model.eval()
print(f"[eval] Loaded  checkpoints/best_model.pth  →  {device}")

# ── Run inference ────────────────────────────────────────────────────────────
all_preds, all_targets = [], []
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs  = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(masks)

all_preds   = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

valid        = all_targets != IGNORE_INDEX
pv           = all_preds[valid]
tv           = all_targets[valid]
total_valid  = valid.sum().item()

# ── Per-class IoU ────────────────────────────────────────────────────────────
SEP = "=" * 62
print()
print(SEP)
print(f"  Evaluation on {len(val_paths)} val images   |   device: {device}")
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

# ── Pixel accuracy ───────────────────────────────────────────────────────────
correct = int((pv == tv).sum())
acc     = correct / total_valid * 100
print(f"  Pixel accuracy : {acc:.2f}%")
print()

# ── Per-class Precision / Recall / F1 ────────────────────────────────────────
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

# ── Visual sample: save 3 val images as side-by-side PNG ────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def colorise(idx_map):
        h, w = idx_map.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for i, col in enumerate(CLASS_COLORS):
            rgb[idx_map == i] = col
        return rgb

    n_samples = min(3, len(val_ds))
    fig, axes = plt.subplots(n_samples, 3, figsize=(13, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(range(n_samples)):
            img_t, mask_t = val_ds[idx]
            logits = model(img_t.unsqueeze(0).to(device))
            pred   = logits.argmax(1).squeeze(0).cpu().numpy()
            mask   = mask_t.numpy()

            # De-normalise for display
            img_np = img_t.permute(1, 2, 0).numpy()
            img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img_np = np.clip(img_np, 0, 1)

            axes[row][0].imshow(img_np);             axes[row][0].set_title("Satellite image"); axes[row][0].axis("off")
            axes[row][1].imshow(colorise(mask));     axes[row][1].set_title("Ground truth");    axes[row][1].axis("off")
            axes[row][2].imshow(colorise(pred));     axes[row][2].set_title("Model prediction");axes[row][2].axis("off")

    legend_patches = [
        Patch(color=[c / 255 for c in col], label=CLASS_NAMES[i])
        for i, col in enumerate(CLASS_COLORS)
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02)
    )
    plt.suptitle(f"Module A  —  val mIoU: {mean_iou:.4f}", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = "checkpoints/eval_samples.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sample predictions saved → {out_path}")

    # ── 1. Confusion Matrix ──────────────────────────────────────────────────────
    print("  Generating Confusion Matrix plot...")
    conf_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    # pv and tv are PyTorch tensors, valid elements only
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            conf_matrix[i, j] = int(((tv == i) & (pv == j)).sum())

    # Normalize confusion matrix by row (True label)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_norm = np.divide(conf_matrix, row_sums, out=np.zeros_like(conf_matrix, dtype=float), where=row_sums!=0)

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = conf_matrix_norm.max() / 2.
    for i in range(conf_matrix_norm.shape[0]):
        for j in range(conf_matrix_norm.shape[1]):
            plt.text(j, i, format(conf_matrix_norm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if conf_matrix_norm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = "checkpoints/confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    # ── 2. Metrics Bar Chart ─────────────────────────────────────────────────────
    print("  Generating Metrics Bar Chart...")
    metrics_data = {
        'IoU': ious,
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for c in range(len(CLASS_NAMES)):
        tp = int(((pv == c) & (tv == c)).sum())
        fp = int(((pv == c) & (tv != c)).sum())
        fn = int(((pv != c) & (tv == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics_data['Precision'].append(prec)
        metrics_data['Recall'].append(rec)
        metrics_data['F1-Score'].append(f1)

    x = np.arange(len(CLASS_NAMES))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 1.5*width, metrics_data['IoU'], width, label='IoU', color='#1f77b4')
    rects2 = ax.bar(x - 0.5*width, metrics_data['Precision'], width, label='Precision', color='#ff7f0e')
    rects3 = ax.bar(x + 0.5*width, metrics_data['Recall'], width, label='Recall', color='#2ca02c')
    rects4 = ax.bar(x + 1.5*width, metrics_data['F1-Score'], width, label='F1-Score', color='#d62728')

    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)

    plt.tight_layout()
    metrics_path = "checkpoints/metrics_bar_chart.png"
    plt.savefig(metrics_path, dpi=120)
    plt.close()
    print(f"  Metrics bar chart saved → {metrics_path}")

except Exception as e:
    print(f"  (Visual save skipped: {e})")

print()
print("Done.")
