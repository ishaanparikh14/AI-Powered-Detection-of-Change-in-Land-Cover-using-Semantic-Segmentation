"""
backend/change_detection.py — Land-cover change analysis between two label maps.

Computes:
  - Per-class area percentages for Year 1 and Year 2
  - Transition matrix (which class converted to which)
  - Tracked transition areas (Forest→Urban, Forest→Agriculture, etc.)
  - Forest loss / gain / net change
  - Alert level
  - Recommendations
"""

import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    CLASS_NAMES, NUM_CLASSES, ALERT_THRESHOLDS,
    TRACKED_TRANSITIONS,
)

logger = logging.getLogger(__name__)

# Reverse lookup: name → index
_NAME_TO_IDX = {v: k for k, v in CLASS_NAMES.items()}
FOREST_IDX   = _NAME_TO_IDX["Forest"]


def _class_percentages(label_map: np.ndarray) -> Dict[str, float]:
    """Return {class_name: percentage_of_valid_pixels} for a label map."""
    valid = label_map.flatten()
    total = len(valid)
    if total == 0:
        return {name: 0.0 for name in CLASS_NAMES.values()}
    pct = {}
    for idx, name in CLASS_NAMES.items():
        pct[name] = float((valid == idx).sum()) / total * 100.0
    return pct


def _transition_matrix(label1: np.ndarray, label2: np.ndarray) -> np.ndarray:
    """
    Return (NUM_CLASSES, NUM_CLASSES) matrix where
    matrix[i, j] = number of pixels that changed from class i → class j.
    """
    mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    flat1 = label1.flatten()
    flat2 = label2.flatten()
    # Only count valid pixels
    mask = (flat1 >= 0) & (flat1 < NUM_CLASSES) & (flat2 >= 0) & (flat2 < NUM_CLASSES)
    flat1, flat2 = flat1[mask], flat2[mask]
    np.add.at(mat, (flat1, flat2), 1)
    return mat


def _alert_level(forest_loss_pct: float) -> str:
    """Map forest loss % to alert level string."""
    for level, threshold in sorted(
        ALERT_THRESHOLDS.items(), key=lambda x: x[1], reverse=True
    ):
        if forest_loss_pct >= threshold:
            return level
    return "Low Risk"


def _recommendations(
    alert_level: str,
    transitions: Dict[str, float],
    forest_loss_pct: float,
) -> List[str]:
    recs = []
    if alert_level == "Critical Risk":
        recs.append("URGENT: Immediate field inspection and satellite monitoring required.")
        recs.append("Notify relevant forest authorities and environmental agencies.")
    elif alert_level == "High Risk":
        recs.append("Field inspection strongly recommended within 30 days.")
        recs.append("Potential deforestation hotspot detected — escalate for review.")
    elif alert_level == "Moderate Risk":
        recs.append("Monitor the area for further changes in the next season.")

    if transitions.get("Forest → Urban", 0) > 0.5:
        recs.append("Urban expansion detected near forest boundary.")
    if transitions.get("Forest → Agriculture", 0) > 0.5:
        recs.append("Forest conversion into agricultural land observed.")
    if transitions.get("Forest → Barren", 0) > 0.5:
        recs.append("Barren land formation from forest area — possible logging or fire.")
    if transitions.get("Agriculture → Urban", 0) > 1.0:
        recs.append("Agricultural land being converted to urban use.")
    if forest_loss_pct < 0:
        recs.append("Net forest gain detected — positive trend, continue monitoring.")
    if not recs:
        recs.append("No significant changes detected. Continue routine monitoring.")
    return recs


def _generate_academic_analysis(class_changes: Dict[str, float], region: str) -> Dict:
    """Generate dynamic academic reasoning and conclusions."""
    drivers = []
    
    f_change = class_changes.get("Forest", 0)
    u_change = class_changes.get("Urban", 0)
    a_change = class_changes.get("Agriculture", 0)
    w_change = class_changes.get("Water", 0)
    b_change = class_changes.get("Barren", 0)

    if f_change < -0.1:
        drivers.append(f"A significant reduction of {abs(f_change):.2f}% in dense forest cover indicates acute anthropogenic pressures. Studies such as those by the Indian Institute of Science (IISc) on the Western Ghats highlight that unbridled agricultural expansion and infrastructure projects are prime catalysts for this deforestation.")
    elif f_change > 0.1:
        drivers.append(f"A positive trend with a {f_change:.2f}% increase in forest cover suggests effective local conservation efforts and natural regeneration, aligning with recent afforestation drives under the Green India Mission.")
    else:
        drivers.append("The forest cover has remained relatively stable, suggesting an equilibrium between localized exploitation and conservation measures in the observed period.")

    if u_change > 0.1:
        drivers.append(f"The {u_change:.2f}% expansion in built-up urban areas is a direct consequence of rapid demographic shifts and infrastructure development. Recent environmental appraisals emphasize that unregulated urbanization in the eco-sensitive zones of the Western Ghats severely fragments primary habitats.")
    
    if a_change > 0.1:
        drivers.append(f"An increase of {a_change:.2f}% in agricultural land, often at the expense of forest fringes, points towards the proliferation of commercial cash crops (tea, coffee, rubber), a trend extensively documented by the Gadgil Committee report.")
    elif a_change < -0.1:
        drivers.append(f"A decrease of {abs(a_change):.2f}% in agricultural land may indicate land degradation, abandonment, or conversion to built-up areas and barren tracts.")

    if w_change < -0.1:
        drivers.append(f"A concerning reduction of {abs(w_change):.2f}% in water bodies correlates with reduced groundwater recharge due to catchment degradation, exacerbating hydrological vulnerability in the region as noted by the Central Water Commission.")

    if b_change > 0.1:
        drivers.append(f"The {b_change:.2f}% increase in barren land highlights soil erosion and localized resource extraction (such as quarrying), which strip the land of its natural ecological resilience.")

    if len(drivers) == 0:
        drivers.append("Minor shifts in land cover classes suggest natural seasonal variations or marginal human interventions without significant landscape alteration.")

    reasoning_text = " ".join(drivers)

    steps = [
        "**Enforcement of Eco-Sensitive Zones (ESZs):** Strict implementation of zoning laws as recommended by the Kasturirangan Committee to restrict hazardous industrial and mining activities in vulnerable tracts.",
        "**Agroforestry Integration:** Transitioning from monoculture plantations to biodiversity-friendly agroforestry systems in buffer areas to restore soil fertility and create wildlife corridors.",
        "**Hydrological Restoration:** Rejuvenating degraded catchments through watershed management and afforestation to secure the water table and sustain regional hydrology."
    ]

    conclusions = []
    conclusions.append(f"**Habitat Fragmentation & Edge Effects:** The observed transitions in {region} reveal a concerning pattern. The conversion of natural habitats to anthropogenic land uses isolates endemic species and diminishes overall biodiversity capacity.")
    
    if f_change < 0 and u_change > 0:
        conclusions.append("**Unsustainable Urban Trajectory:** The direct replacement of ecological carbon sinks with impervious urban surfaces poses severe risks to localized climate resilience. It increases susceptibility to extreme weather events like landslides and flash floods, a recurring issue documented in recent regional climatic anomaly reports.")
    elif f_change < 0 and a_change > 0:
        conclusions.append("**Agricultural Encroachment:** The encroachment of commercial agriculture into forested tracts continues to be a dominant driver of ecological degradation, necessitating immediate policy interventions for sustainable land-use.")
    else:
        conclusions.append("**Ecological Vulnerability:** The land-cover dynamics underscore a precarious balance. Without proactive management, even minor anthropogenic perturbations could trigger irreversible ecological decline in this global biodiversity hotspot.")

    conclusions.append("**Imperative for Data-Driven Policy:** Real-time, satellite-derived insights emphasize the critical need for executing ground-level environmental regulations promptly. Continued unmonitored land-cover alterations will irreversibly compromise the long-term ecological and economic stability of peninsular India.")

    return {
        "reasoning": reasoning_text,
        "steps": steps,
        "conclusions": conclusions
    }



def analyze(
    label1: np.ndarray,
    label2: np.ndarray,
    region_name: str,
    year1: int,
    year2: int,
) -> Dict:
    """
    Full change-detection analysis.

    Parameters
    ----------
    label1, label2 : (H, W) int   class index maps for year1, year2

    Returns
    -------
    dict with all statistics, alerts, and recommendations
    """
    logger.info("Running change detection: %s  %d→%d", region_name, year1, year2)

    pct1 = _class_percentages(label1)
    pct2 = _class_percentages(label2)

    # Per-class change
    class_changes = {
        name: pct2[name] - pct1[name]
        for name in CLASS_NAMES.values()
    }

    # Transition matrix
    trans_mat  = _transition_matrix(label1, label2)
    total_px   = trans_mat.sum()

    # Tracked transitions as % of total pixels
    tracked = {}
    for src_name, dst_name in TRACKED_TRANSITIONS:
        si = _NAME_TO_IDX[src_name]
        di = _NAME_TO_IDX[dst_name]
        key = f"{src_name} → {dst_name}"
        tracked[key] = float(trans_mat[si, di]) / max(total_px, 1) * 100.0

    # Forest metrics
    forest_year1_px = int((label1 == FOREST_IDX).sum())
    forest_year2_px = int((label2 == FOREST_IDX).sum())
    forest_loss_px  = max(0, forest_year1_px - forest_year2_px)
    forest_gain_px  = max(0, forest_year2_px - forest_year1_px)
    net_change_px   = forest_year2_px - forest_year1_px

    total_valid     = label1.size
    forest_loss_pct = float(forest_loss_px) / max(total_valid, 1) * 100.0
    forest_gain_pct = float(forest_gain_px) / max(total_valid, 1) * 100.0
    net_change_pct  = float(net_change_px)  / max(total_valid, 1) * 100.0

    alert = _alert_level(forest_loss_pct)
    recs  = _recommendations(alert, tracked, net_change_pct)
    academic_analysis = _generate_academic_analysis(class_changes, region_name)

    result = {
        "region":       region_name,
        "year1":        year1,
        "year2":        year2,
        "class_pct_y1": pct1,
        "class_pct_y2": pct2,
        "class_changes": class_changes,
        "transitions":  tracked,
        "forest": {
            "year1_pct":     pct1["Forest"],
            "year2_pct":     pct2["Forest"],
            "loss_pct":      forest_loss_pct,
            "gain_pct":      forest_gain_pct,
            "net_change_pct": net_change_pct,
        },
        "alert_level":      alert,
        "recommendations":  recs,
        "academic_analysis": academic_analysis,
        "transition_matrix": trans_mat.tolist(),
    }
    logger.info("Alert level: %s  |  Forest loss: %.2f%%", alert, forest_loss_pct)
    return result


def generate_change_map(label1: np.ndarray, label2: np.ndarray) -> np.ndarray:
    """
    Create a colour-coded change map (H, W, 3) uint8.

    Colours:
      Red    — Forest loss  (Forest → non-Forest)
      Green  — Forest gain  (non-Forest → Forest)
      Orange — Urbanisation (any → Urban)
      Grey   — No change
      White  — Other change
    """
    h, w = label1.shape
    change_map = np.zeros((h, w, 3), dtype=np.uint8)
    change_map[:] = [200, 200, 200]   # default grey = no change

    urban_idx = _NAME_TO_IDX["Urban"]

    # No change
    no_change = label1 == label2
    change_map[no_change] = [200, 200, 200]

    # Any change
    changed = label1 != label2
    change_map[changed] = [255, 255, 255]   # other change — white

    # Forest loss
    forest_loss = (label1 == FOREST_IDX) & (label2 != FOREST_IDX)
    change_map[forest_loss] = [220, 50, 50]

    # Forest gain
    forest_gain = (label1 != FOREST_IDX) & (label2 == FOREST_IDX)
    change_map[forest_gain] = [50, 200, 50]

    # Urbanisation
    urbanised = (label1 != urban_idx) & (label2 == urban_idx)
    change_map[urbanised] = [255, 140, 0]

    return change_map
