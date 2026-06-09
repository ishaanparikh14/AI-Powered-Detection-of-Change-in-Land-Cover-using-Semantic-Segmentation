"""
backend/gee_loader.py — Google Earth Engine satellite imagery fetcher.

Fetches Sentinel-2 SR imagery for a given region and year,
applies cloud filtering, selects the best composite, and
returns a numpy RGB array ready for tiling + inference.
"""

import io
import logging
import os
import sys

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    GEE_PROJECT, GEE_DATASET, GEE_BANDS, CLOUD_THRESH,
    IMAGE_SCALE, OUTPUT_SIZE, REGIONS,
)

logger = logging.getLogger(__name__)

# Lazy-import earthengine so the rest of the app works even if ee isn't installed
_ee = None


def _get_ee():
    global _ee
    if _ee is None:
        try:
            import ee
            ee.Initialize(project=GEE_PROJECT)
            _ee = ee
            logger.info("Earth Engine initialised (project=%s)", GEE_PROJECT)
        except Exception as exc:
            logger.error("Earth Engine init failed: %s", exc)
            raise RuntimeError(
                f"Earth Engine initialisation failed: {exc}\n"
                "Run 'earthengine authenticate' and ensure the ee Python package is installed."
            )
    return _ee


def fetch_sentinel2(region_name: str, year: int) -> np.ndarray:
    """
    Fetch a cloud-free Sentinel-2 RGB composite for a region and year.

    Parameters
    ----------
    region_name : str   One of the keys in REGIONS config.
    year        : int   e.g. 2020 or 2024

    Returns
    -------
    np.ndarray  shape (H, W, 3)  uint8  RGB image
    """
    if region_name not in REGIONS:
        raise ValueError(f"Unknown region '{region_name}'. Valid: {list(REGIONS)}")

    ee = _get_ee()

    bbox  = REGIONS[region_name]        # [west, south, east, north]
    aoi   = ee.Geometry.Rectangle(bbox)

    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    logger.info("Fetching Sentinel-2 for %s %d  bbox=%s", region_name, year, bbox)

    collection = (
        ee.ImageCollection(GEE_DATASET)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESH))
        .select(GEE_BANDS)
    )

    count = collection.size().getInfo()
    logger.info("  Found %d images after cloud filter", count)

    if count == 0:
        # Relax cloud filter and retry
        logger.warning("  No images found with threshold %d%%, relaxing to 50%%", CLOUD_THRESH)
        collection = (
            ee.ImageCollection(GEE_DATASET)
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
            .select(GEE_BANDS)
        )
        count = collection.size().getInfo()
        if count == 0:
            raise RuntimeError(
                f"No Sentinel-2 imagery found for {region_name} in {year}. "
                "Try a different year or check GEE availability."
            )

    # Median composite → minimises cloud/shadow artefacts
    image = collection.median().clip(aoi)

    # Visualise: scale DN 0-3000 to 0-255 for RGB display
    vis_params = {
        "min": 0,
        "max": 3000,
        "bands": GEE_BANDS,
    }

    url = image.getThumbURL({
        "region":      aoi,
        "dimensions":  OUTPUT_SIZE,
        "format":      "png",
        "min":          0,
        "max":          3000,
        "bands":        GEE_BANDS,
    })

    logger.info("  Downloading thumbnail from GEE …")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    img_pil  = Image.open(io.BytesIO(response.content)).convert("RGB")
    img_np   = np.array(img_pil)                          # (H, W, 3) uint8
    logger.info("  Downloaded image shape: %s", img_np.shape)

    return img_np


def fetch_both_years(region_name: str, year1: int, year2: int):
    """
    Convenience wrapper — fetch imagery for two years.

    Returns
    -------
    img1, img2 : np.ndarray  (H, W, 3)  uint8
    """
    img1 = fetch_sentinel2(region_name, year1)
    img2 = fetch_sentinel2(region_name, year2)
    return img1, img2
