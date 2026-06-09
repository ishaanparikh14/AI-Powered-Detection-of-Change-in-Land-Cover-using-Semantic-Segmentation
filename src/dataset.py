"""
src/dataset.py — DeepGlobe Land Cover dataset loader.

Responsibilities
----------------
1. Scan DATA_DIR for *_sat.jpg / *_mask.png pairs.
2. Convert RGB mask → per-pixel class index using exact colour lookup.
   Any colour not in the lookup table is remapped to IGNORE_INDEX (255).
3. Resize both image and mask to IMG_SIZE×IMG_SIZE (nearest for mask).
4. Apply albumentations augmentation on the training split.
5. Return (image_tensor [3,H,W] float32, mask_tensor [H,W] int64).
"""

import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import config from the parent directory regardless of how the script is run
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_config import (
    DATA_DIR, COLOR_TO_INDEX, IGNORE_INDEX,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


# ---------------------------------------------------------------------------
# Precomputed lookup table for fast RGB → index conversion
# Shape: (256, 256, 256)  dtype: uint8
# Built once at import time; negligible memory (~16 MB).
# ---------------------------------------------------------------------------
def _build_lut() -> np.ndarray:
    lut = np.full((256, 256, 256), IGNORE_INDEX, dtype=np.uint8)
    for (r, g, b), idx in COLOR_TO_INDEX.items():
        lut[r, g, b] = idx
    return lut


_LUT = _build_lut()


def rgb_mask_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an H×W×3 uint8 RGB mask to an H×W uint8 index mask.
    Uses the precomputed lookup table for O(H×W) speed.
    """
    r, g, b = mask_rgb[:, :, 0], mask_rgb[:, :, 1], mask_rgb[:, :, 2]
    return _LUT[r, g, b]


# ---------------------------------------------------------------------------
# Albumentations transforms
# ---------------------------------------------------------------------------
def _train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.05, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DeepGlobeDataset(Dataset):
    """
    Parameters
    ----------
    image_paths : list[str]   Absolute paths to *_sat.jpg files.
    augment     : bool        If True applies training augmentations.
    """

    def __init__(self, image_paths: list, augment: bool = False):
        self.image_paths = image_paths
        self.mask_paths  = [p.replace("_sat.jpg", "_mask.png") for p in image_paths]
        self.transform   = _train_transforms() if augment else _val_transforms()

        # Sanity-check that every mask exists
        missing = [m for m in self.mask_paths if not os.path.exists(m)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} mask file(s) not found. First missing: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # --- Load image (BGR → RGB) -----------------------------------------
        img_bgr = cv2.imread(self.image_paths[idx])
        if img_bgr is None:
            raise IOError(f"Cannot read image: {self.image_paths[idx]}")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # H×W×3 uint8

        # --- Load mask (BGR → RGB) ------------------------------------------
        mask_bgr = cv2.imread(self.mask_paths[idx])
        if mask_bgr is None:
            raise IOError(f"Cannot read mask: {self.mask_paths[idx]}")
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)  # H×W×3 uint8

        # --- Remap RGB → class index (0..4 or 255) --------------------------
        mask = rgb_mask_to_index(mask_rgb)  # H×W uint8

        # --- Apply transforms -----------------------------------------------
        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]              # torch.float32 [3,H,W]
        mask_t  = augmented["mask"].long()        # torch.int64   [H,W]

        return image_t, mask_t


# ---------------------------------------------------------------------------
# Helper: scan DATA_DIR and return sorted list of satellite image paths
# ---------------------------------------------------------------------------
def get_image_paths(data_dir: str = DATA_DIR) -> list:
    """Return sorted list of absolute paths to all *_sat.jpg files."""
    pattern = os.path.join(data_dir, "*_sat.jpg")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No *_sat.jpg files found in '{data_dir}'. "
            "Check that DATA_DIR in config.py points to the correct folder."
        )
    return paths
