"""
config.py — Central configuration for Module A (DeepGlobe training pipeline).
All other modules import from here; change values once, applies everywhere.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "dataset", "train")   # <id>_sat.jpg + <id>_mask.png
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
NUM_CLASSES  = 5
CLASS_NAMES  = ["Forest", "Urban", "Agriculture", "Water", "Barren"]
IGNORE_INDEX = 255   # pixels remapped here are excluded from loss & metric

# ---------------------------------------------------------------------------
# RGB colour → class index  (exact DeepGlobe mask colours — PNG, no JPEG noise)
# Rangeland (magenta) is merged into Agriculture (index 2).
# Unknown / black → IGNORE_INDEX so it is masked out everywhere.
# ---------------------------------------------------------------------------
COLOR_TO_INDEX = {
    (0,   255, 0):   0,    # Forest        → 0
    (0,   255, 255): 1,    # Urban         → 1
    (255, 255, 0):   2,    # Agriculture   → 2
    (255, 0,   255): 2,    # Rangeland     → 2  (merged)
    (0,   0,   255): 3,    # Water         → 3
    (255, 255, 255): 4,    # Barren        → 4
    (0,   0,   0):   IGNORE_INDEX,  # Unknown
}

# Visualisation palette — one RGB colour per class index (for inference overlays)
CLASS_COLORS = [
    (0,   255, 0),    # 0 Forest
    (0,   255, 255),  # 1 Urban
    (255, 255, 0),    # 2 Agriculture
    (0,   0,   255),  # 3 Water
    (255, 255, 255),  # 4 Barren
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
IN_CHANNELS  = 3
ENCODER_NAME = "resnet34"          # used by segmentation_models_pytorch
ENCODER_WEIGHTS = "imagenet"       # pretrained weights

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE    = 256
BATCH_SIZE  = 2      # RTX 3050 4GB — keep VRAM under 3.5GB
NUM_WORKERS = 0      # Windows: 0 avoids DataLoader multiprocess issues
LR          = 1e-4
EPOCHS      = 50
SEED        = 42

# ImageNet normalisation (applied when using a pretrained encoder)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Loss weighting
# ---------------------------------------------------------------------------
DICE_WEIGHT = 0.5    # final loss = CE * (1-DICE_WEIGHT) + Dice * DICE_WEIGHT
