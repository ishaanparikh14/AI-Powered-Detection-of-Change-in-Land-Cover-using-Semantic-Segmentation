"""
backend/inference.py — Tile, infer, and stitch segmentation maps.

Workflow:
  1. Split a large satellite image into overlapping TILE_SIZE x TILE_SIZE tiles.
  2. Run best_model.pth on each tile.
  3. Stitch predictions back into a full-resolution label map.
  4. Return label map (H, W) int  and colour overlay (H, W, 3) uint8.
"""

import logging
import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MODEL_PATH, NUM_CLASSES, TILE_SIZE, OVERLAP,
    BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_COLORS_RGB, IGNORE_INDEX,
)

logger = logging.getLogger(__name__)

# ── Model singleton ───────────────────────────────────────────────────────────
_model  = None
_device = None


def _load_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load src/model.py under a unique alias so it never collides with the
    # config/ package or config.py (model.py no longer imports from config).
    import importlib.util
    root     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(root, "src", "model.py")
    spec     = importlib.util.spec_from_file_location("_wg_model", src_path)
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    UNet = model_mod.UNet

    m = UNet(num_classes=NUM_CLASSES)

    # Handle multiple checkpoint formats:
    #   1. Raw state dict (OrderedDict)
    #   2. {'state_dict': ...}
    #   3. {'model_state_dict': ...}
    checkpoint = torch.load(MODEL_PATH, map_location=_device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint          # assume it IS the state dict
    else:
        state = checkpoint

    m.load_state_dict(state)
    m.to(_device).eval()
    _model = m

    logger.info("Model loaded from %s on %s", MODEL_PATH, _device)
    return _model, _device



# ── Pre/post processing helpers ───────────────────────────────────────────────
def _normalise(img_uint8: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, H, W) float32 ImageNet-normalised."""
    x = img_uint8.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    x    = (x - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def _colorise(label_map: np.ndarray) -> np.ndarray:
    """(H, W) int → (H, W, 3) uint8 colour overlay."""
    h, w  = label_map.shape
    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS_RGB.items():
        rgb[label_map == cls_idx] = color
    return rgb


# ── Tiling ────────────────────────────────────────────────────────────────────
def _extract_tiles(image: np.ndarray, tile_size: int, overlap: int):
    """
    Yield (tile_img, row_start, col_start) for every tile position.
    Tiles are padded to tile_size if the image edge is reached.
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap
    row = 0
    while row < h:
        col = 0
        while col < w:
            r1, c1 = row, col
            r2, c2 = min(row + tile_size, h), min(col + tile_size, w)
            tile = image[r1:r2, c1:c2]
            # Pad to tile_size x tile_size if necessary
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            yield tile, r1, c1, r2, c2
            col += stride
            if col >= w:
                break
        row += stride
        if row >= h:
            break


# ── Main inference entry point ────────────────────────────────────────────────
def run_inference(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run segmentation on a full satellite image.

    Parameters
    ----------
    image_rgb : np.ndarray  (H, W, 3)  uint8  RGB

    Returns
    -------
    label_map    : np.ndarray  (H, W)      int   class indices 0..4
    colour_map   : np.ndarray  (H, W, 3)   uint8  RGB colour overlay
    """
    model, device = _load_model()
    h, w = image_rgb.shape[:2]

    # Accumulate logit sum + count per pixel for overlap averaging
    logit_sum = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    tiles     = list(_extract_tiles(image_rgb, TILE_SIZE, OVERLAP))
    logger.info("Running inference on %d tiles (image %dx%d)", len(tiles), h, w)

    # Process in batches
    for batch_start in range(0, len(tiles), BATCH_SIZE):
        batch = tiles[batch_start: batch_start + BATCH_SIZE]
        tensors = torch.cat([_normalise(t[0]) for t in batch], dim=0).to(device)

        with torch.no_grad():
            logits = model(tensors)                        # (B, C, 256, 256)
            probs  = F.softmax(logits, dim=1).cpu().numpy()

        for i, (_, r1, c1, r2, c2) in enumerate(batch):
            tile_h = r2 - r1
            tile_w = c2 - c1
            p_crop = probs[i, :, :tile_h, :tile_w]        # (C, tile_h, tile_w)
            logit_sum[:, r1:r2, c1:c2] += p_crop
            count_map[r1:r2, c1:c2]    += 1.0

    # Average overlapping regions
    count_map = np.maximum(count_map, 1.0)
    avg_probs  = logit_sum / count_map[np.newaxis, :, :]   # (C, H, W)
    label_map  = avg_probs.argmax(axis=0).astype(np.int32) # (H, W)
    colour_map = _colorise(label_map)

    logger.info("Inference complete. Unique classes: %s", np.unique(label_map))
    return label_map, colour_map
