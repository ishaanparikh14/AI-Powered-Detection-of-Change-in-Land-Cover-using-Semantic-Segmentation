"""
src/loss.py — Combined segmentation loss.

    loss = (1 - DICE_WEIGHT) * CrossEntropyLoss  +  DICE_WEIGHT * DiceLoss

Both components respect ignore_index=255 so unknown/background pixels are
excluded from both the gradient and the metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_config import IGNORE_INDEX, DICE_WEIGHT, NUM_CLASSES


class MulticlassDiceLoss(nn.Module):
    """
    Soft Dice loss averaged over the valid classes.
    Pixels with label == ignore_index are zeroed out before the computation.

    Parameters
    ----------
    num_classes  : int
    ignore_index : int   Label value to exclude (default 255).
    smooth       : float Laplace smoothing to avoid division by zero.
    """

    def __init__(
        self,
        num_classes:  int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        smooth:       float = 1.0,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)  raw (unnormalised) scores
        targets : (B, H, W)     long  — values in [0, C-1] or ignore_index
        """
        probs = F.softmax(logits, dim=1)   # (B, C, H, W)

        # Build a valid-pixel mask: 1 where we should count, 0 otherwise
        valid = (targets != self.ignore_index).float()   # (B, H, W)

        # Replace ignore pixels in targets with 0 so one_hot doesn't crash
        targets_clean = targets.clone()
        targets_clean[targets == self.ignore_index] = 0

        # One-hot encode: (B, H, W) → (B, C, H, W)
        one_hot = F.one_hot(targets_clean, self.num_classes)   # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()          # (B, C, H, W)

        # Apply valid mask to both predictions and targets
        valid_4d = valid.unsqueeze(1)          # (B, 1, H, W) — broadcasts over C
        probs   = probs   * valid_4d
        one_hot = one_hot * valid_4d

        # Dice per class, then mean
        dims    = (0, 2, 3)                    # reduce over batch, H, W
        inter   = (probs * one_hot).sum(dims)  # (C,)
        union   = probs.sum(dims) + one_hot.sum(dims)  # (C,)
        dice    = (2.0 * inter + self.smooth) / (union + self.smooth)  # (C,)

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum: CE + Dice.

    Parameters
    ----------
    dice_weight  : float   Weight of Dice term (0 = CE only, 1 = Dice only).
    num_classes  : int
    ignore_index : int
    """

    def __init__(
        self,
        dice_weight:  float = DICE_WEIGHT,
        num_classes:  int   = NUM_CLASSES,
        ignore_index: int   = IGNORE_INDEX,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = MulticlassDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (1.0 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
