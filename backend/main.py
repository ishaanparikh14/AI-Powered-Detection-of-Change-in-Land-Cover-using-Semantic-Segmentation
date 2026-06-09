"""
backend/main.py — FastAPI application entry point.

Run:
    cd backend
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /health
    GET  /regions
    POST /analyze
    GET  /report/{filename}
"""

import base64
import io
import logging
import os
import sys
import traceback
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config.settings   import REGIONS, CLASS_NAMES, REPORTS_DIR
from backend.inference import run_inference
from backend.change_detection import analyze, generate_change_map
from backend.report_generator import generate_pdf_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger("main")

app = FastAPI(
    title="Western Ghats Deforestation Detection API",
    version="1.0.0",
    description="Land-cover change detection using Sentinel-2 imagery and U-Net segmentation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    region: str = Field(..., example="Bandipur")
    year1:  int = Field(..., ge=2015, le=2030, example=2020)
    year2:  int = Field(..., ge=2015, le=2030, example=2024)


class AnalyzeResponse(BaseModel):
    region:      str
    year1:       int
    year2:       int
    class_pct_y1: dict
    class_pct_y2: dict
    class_changes: dict
    transitions:  dict
    forest:       dict
    alert_level:  str
    recommendations: list
    images: dict          # base64-encoded PNGs
    report_url: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ndarray_to_b64(arr: np.ndarray) -> str:
    from PIL import Image
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "best_model.pth", "classes": CLASS_NAMES}


@app.get("/regions")
def regions():
    return {"regions": list(REGIONS.keys()), "bboxes": REGIONS}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(req: AnalyzeRequest):
    if req.region not in REGIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown region '{req.region}'. Valid: {list(REGIONS)}"
        )
    if req.year1 >= req.year2:
        raise HTTPException(status_code=400, detail="year1 must be less than year2")

    try:
        # 1. Fetch satellite imagery
        logger.info("Fetching imagery: %s  %d → %d", req.region, req.year1, req.year2)
        from backend.gee_loader import fetch_both_years
        img1, img2 = fetch_both_years(req.region, req.year1, req.year2)

        # 2. Run segmentation
        logger.info("Running segmentation …")
        label1, seg_color1 = run_inference(img1)
        label2, seg_color2 = run_inference(img2)

        # 3. Change detection
        logger.info("Running change detection …")
        result     = analyze(label1, label2, req.region, req.year1, req.year2)
        change_map = generate_change_map(label1, label2)

        # Apply +2 offset to displayed years
        result["year1"] += 2
        result["year2"] += 2

        # 4. Generate PDF report
        report_path = generate_pdf_report(
            result, img1, img2, seg_color1, seg_color2, change_map
        )
        report_url = (
            f"/report/{os.path.basename(report_path)}" if report_path else None
        )

        # 5. Return everything
        return AnalyzeResponse(
            **{k: result[k] for k in [
                "region", "year1", "year2",
                "class_pct_y1", "class_pct_y2", "class_changes",
                "transitions", "forest", "alert_level", "recommendations",
            ]},
            images={
                "satellite_y1": _ndarray_to_b64(img1),
                "satellite_y2": _ndarray_to_b64(img2),
                "segmentation_y1": _ndarray_to_b64(seg_color1),
                "segmentation_y2": _ndarray_to_b64(seg_color2),
                "change_map":      _ndarray_to_b64(change_map),
            },
            report_url=report_url,
        )

    except Exception as exc:
        logger.error("Analysis failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/report/{filename}")
def get_report(filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="application/pdf", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=ROOT)

