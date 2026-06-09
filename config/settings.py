"""
config/settings.py — Central configuration for the Western Ghats Change Detection System.
All regions, class mappings, model paths, and thresholds live here.
"""

import os

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
NUM_CLASSES   = 5
TILE_SIZE     = 256        # pixels — model input size
OVERLAP       = 32         # pixel overlap between tiles to avoid edge artefacts
BATCH_SIZE    = 8          # tiles per inference batch

# ── Earth Engine ──────────────────────────────────────────────────────────────
GEE_PROJECT   = "unified-hull-470511-t2"
GEE_DATASET   = "COPERNICUS/S2_SR_HARMONIZED"
GEE_BANDS     = ["B4", "B3", "B2"]          # Red, Green, Blue
CLOUD_THRESH  = 20                            # max cloud cover %
IMAGE_SCALE   = 10                            # metres per pixel (Sentinel-2 native)
OUTPUT_SIZE   = 1024                          # px — downloaded image size per region

# ── Supported regions and bounding boxes [west, south, east, north] ──────────
REGIONS = {
    "Bandipur":   [76.15, 11.55, 76.85, 12.05],
    "Wayanad":    [75.75, 11.45, 76.45, 12.15],
    "Nagarhole":  [75.90, 11.75, 76.40, 12.30],
    "Coorg":      [75.50, 12.00, 76.20, 12.75],
    "Sakleshpur": [75.60, 12.70, 75.95, 13.15],
    "Kudremukh":  [75.00, 13.00, 75.45, 13.45],
}

# ── Land-cover class definitions ──────────────────────────────────────────────
CLASS_NAMES = {
    0: "Forest",
    1: "Agriculture",
    2: "Urban",
    3: "Water",
    4: "Barren",
}

CLASS_COLORS_HEX = {
    0: "#1a9641",   # Forest      — green
    1: "#ffffb2",   # Agriculture — yellow
    2: "#d7191c",   # Urban       — red
    3: "#2c7bb6",   # Water       — blue
    4: "#d9d9d9",   # Barren      — grey
}

CLASS_COLORS_RGB = {
    0: (26,  150,  65),   # Forest
    1: (255, 255, 178),   # Agriculture
    2: (215,  25,  28),   # Urban
    3: ( 44, 123, 182),   # Water
    4: (217, 217, 217),   # Barren
}

IGNORE_INDEX = 255

# ── Alert thresholds (forest loss %) ─────────────────────────────────────────
ALERT_THRESHOLDS = {
    "Critical Risk": 20.0,
    "High Risk":     10.0,
    "Moderate Risk":  5.0,
    "Low Risk":       0.0,
}

# ── Change transitions to track ───────────────────────────────────────────────
TRACKED_TRANSITIONS = [
    ("Forest", "Urban"),
    ("Forest", "Agriculture"),
    ("Forest", "Barren"),
    ("Agriculture", "Urban"),
]

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

for _d in [EXPORTS_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── ImageNet normalisation (must match training) ──────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
