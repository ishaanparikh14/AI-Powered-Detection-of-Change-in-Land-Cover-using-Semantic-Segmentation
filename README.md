<div align="center">

# 🌿 AI-Powered Detection of Change in Land Cover using Semantic Segmentation

**Detecting deforestation and land-cover transitions across six Western Ghats biodiversity hotspots using deep learning and satellite imagery**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Powered by Sentinel-2 satellite imagery · U-Net ResNet-34 · Google Earth Engine · mIoU 0.714 · Accuracy 87.86%*

</div>

---

## 📸 Dashboard Screenshots

<div align="center">
  <img src="assets/landing_page.png" alt="Landing Page" width="90%"/>
  <p><b>Landing Page — AI-Powered Environmental Intelligence</b></p>
</div>

<table>
  <tr>
    <td align="center"><b>Dashboard Overview & Risk Alert</b></td>
    <td align="center"><b>Satellite Imagery & Segmentation Maps</b></td>
  </tr>
  <tr>
    <td><img src="assets/dashboard_overview.png" alt="Dashboard Overview"/></td>
    <td><img src="assets/satellite_segmentation.png" alt="Satellite + Segmentation"/></td>
  </tr>
  <tr>
    <td align="center"><b>Year-on-Year Change Analysis Charts</b></td>
    <td align="center"><b>Detailed Land Cover Statistics Table</b></td>
  </tr>
  <tr>
    <td><img src="assets/change_analysis_charts.png" alt="Change Analysis Charts"/></td>
    <td><img src="assets/land_cover_table.png" alt="Land Cover Table"/></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><b>Recommendations & Actionable Insights</b></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><img src="assets/recommendations.png" alt="Recommendations Panel" width="70%"/></td>
  </tr>
</table>

---

## 📄 Sample PDF Reports

The system automatically generates structured PDF reports for every analysis run. Reports include a forest cover summary, class-wise land cover percentages, key transition tables, a change detection map, and an AI-generated ecological justification.

Sample reports for all six regions are available in the [`reports/`](reports/) directory.

<table>
  <tr>
    <td align="center"><b>Page 1 — Forest Cover Summary & Land Cover Percentages</b></td>
    <td align="center"><b>Page 2 — Key Land Cover Transitions & Recommendations</b></td>
  </tr>
  <tr>
    <td><img src="assets/report_page1_summary.png" alt="Report Page 1 - Summary"/></td>
    <td><img src="assets/report_page2_transitions.png" alt="Report Page 2 - Transitions"/></td>
  </tr>
  <tr>
    <td align="center"><b>Page 3 — Academic Analysis & Ecological Justification</b></td>
    <td align="center"><b>Page 4 — Change Detection Map</b></td>
  </tr>
  <tr>
    <td><img src="assets/report_page3_analysis.png" alt="Report Page 3 - Analysis"/></td>
    <td><img src="assets/report_page4_change_map.png" alt="Report Page 4 - Change Map"/></td>
  </tr>
</table>

> Each report is named `report_<Region>_<Year1>_<Year2>_<timestamp>.pdf` and can also be downloaded directly from the dashboard.

---

## ✨ Features

- 🛰️ **Live Satellite Imagery** — Fetches Sentinel-2 median composites dynamically via Google Earth Engine for any year
- 🧠 **AI Segmentation** — U-Net with ResNet-34 encoder classifies every pixel into 5 land-cover classes
- 📊 **Change Detection** — Compares any two years, computes full transition matrices and class deltas
- ⚠️ **Risk Alerts** — Automatically flags Critical / High / Moderate / Low forest-loss risk
- 📋 **PDF Reports** — One-click downloadable stakeholder reports with maps, tables, and ecological analysis
- 🗺️ **Six Regions** — Bandipur, Wayanad, Nagarhole, Coorg, Sakleshpur, Kudremukh

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│  Control Panel · Image Grid · Charts · Change Table      │
│          Recommendations · PDF Download                  │
└────────────────────────┬────────────────────────────────┘
                         │  REST API (Axios)
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                          │
│   /analyze · /regions · /health · /report/{file}         │
└──────┬──────────────────────────────────┬───────────────┘
       │                                  │
┌──────▼──────┐                  ┌────────▼──────────────┐
│ Google      │                  │  PyTorch AI Core       │
│ Earth       │                  │  U-Net ResNet-34       │
│ Engine      │                  │  Tile → Infer → Stitch │
│ (Sentinel-2)│                  └────────┬───────────────┘
└─────────────┘                           │
                                 ┌────────▼───────────────┐
                                 │  Change Detection +     │
                                 │  ReportLab PDF Engine   │
                                 └────────────────────────┘
```

---

## 📁 Project Structure

```
AI-Powered-Detection-of-Change-in-Land-Cover-using-Semantic-Segmentation/
├── README.md
├── requirements.txt          ← Root-level Python dependencies
├── .gitignore
│
├── assets/                   ← Screenshots & model evaluation plots
│
├── reports/                  ← Sample PDF reports generated by the system
│                               (named: report_<Region>_<Y1>_<Y2>_<timestamp>.pdf)
│
├── backend/                  ← FastAPI server
│   ├── main.py               ← API routes
│   ├── gee_loader.py         ← Google Earth Engine imagery fetcher
│   ├── inference.py          ← Tile → infer → stitch pipeline
│   ├── change_detection.py   ← Change analysis & alert logic
│   ├── report_generator.py   ← PDF report generator
│   └── requirements.txt      ← Backend-specific dependencies
│
├── config/
│   ├── settings.py           ← Regions, class maps, thresholds
│   └── ml_config.py          ← Training hyperparameters & paths
│
├── frontend/                 ← React dashboard
│   ├── src/
│   │   ├── App.js
│   │   ├── api.js            ← Axios API client
│   │   └── components/       ← Dashboard components
│   └── package.json
│
├── scripts/
│   └── run_backend.py        ← One-command backend launcher (Windows-friendly)
│
└── src/                      ← ML training & evaluation scripts
    ├── model.py              ← U-Net model definition
    ├── dataset.py            ← DeepGlobe dataset loader
    ├── loss.py               ← Combined CE + Dice loss
    ├── train.py              ← Training loop
    ├── train_full.py         ← Full training with logging
    ├── eval.py               ← Evaluation script
    └── eval_full.py          ← Full evaluation with metrics export
```

> **Not tracked in git:** `checkpoints/` (model weights, ~93 MB), `dataset/` (training data), `exports/` — see setup instructions below.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ishaanparikh14/AI-Powered-Detection-of-Change-in-Land-Cover-using-Semantic-Segmentation.git
cd AI-Powered-Detection-of-Change-in-Land-Cover-using-Semantic-Segmentation
```

### 2. Set up the Python backend

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Authenticate with Google Earth Engine

You need a [Google Earth Engine](https://earthengine.google.com/) account with an active Cloud project.

```bash
pip install earthengine-api
earthengine authenticate
```

A browser window will open — sign in with the Google account linked to your GEE project. Then verify:

```bash
python -c "import ee; ee.Initialize(project='YOUR_GEE_PROJECT_ID'); print('GEE OK')"
```

> Update `config/settings.py` with your GEE project ID.

### 4. Add the trained model weights

Download `best_model.pth` and place it at:

```
checkpoints/best_model.pth
```

### 5. Install and run the React frontend

```bash
cd frontend
npm install
npm start
```

Frontend starts at **http://localhost:3000**

### 6. Start the backend (separate terminal)

**Option A — convenience launcher (recommended on Windows):**
```bash
python scripts/run_backend.py
```

**Option B — uvicorn directly:**
```bash
uvicorn backend.main:app --reload --port 8001
```

API live at **http://localhost:8001** · Swagger docs at **http://localhost:8001/docs**

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server + model status |
| `GET` | `/regions` | List of supported regions |
| `POST` | `/analyze` | Run full analysis pipeline |
| `GET` | `/report/{file}` | Download generated PDF report |

### Example `/analyze` request

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"region": "Kudremukh", "year1": 2020, "year2": 2024}'
```

---

## 🗺️ Supported Regions

| Region | Bounding Box (W, S, E, N) |
|--------|--------------------------|
| Bandipur | 76.15, 11.55, 76.85, 12.05 |
| Wayanad | 75.75, 11.45, 76.45, 12.15 |
| Nagarhole | 75.90, 11.75, 76.40, 12.30 |
| Coorg | 75.50, 12.00, 76.20, 12.75 |
| Sakleshpur | 75.60, 12.70, 75.95, 13.15 |
| Kudremukh | 75.00, 13.00, 75.45, 13.45 |

---

## 🎨 Land Cover Classes

| Index | Class | Colour |
|-------|-------|--------|
| 0 | 🟢 Forest | Green |
| 1 | 🟡 Agriculture | Yellow |
| 2 | 🔴 Urban | Red |
| 3 | 🔵 Water | Blue |
| 4 | ⚪ Barren | Grey |

---

## ⚠️ Alert Levels

| Forest Loss | Alert Level |
|-------------|-------------|
| > 20% | 🔴 Critical Risk |
| > 10% | 🟠 High Risk |
| > 5% | 🟡 Moderate Risk |
| ≤ 5% | 🟢 Low Risk |

---

## 🤖 Model Performance

**Architecture:** U-Net with ResNet-34 encoder (ImageNet pretrained)  
**Training data:** DeepGlobe Land Cover Dataset (803 images)

| Metric | Score |
|--------|-------|
| Validation mIoU | **0.714** |
| Pixel Accuracy | **87.86%** |
| Input size | 256 × 256 RGB tiles |
| Output | 5-class segmentation map |

<table>
  <tr>
    <td align="center"><b>IoU per Class</b></td>
    <td align="center"><b>Precision · Recall · F1 per Class</b></td>
  </tr>
  <tr>
    <td><img src="assets/model_iou_per_class.png" alt="IoU per Class"/></td>
    <td><img src="assets/model_precision_recall_f1.png" alt="Precision Recall F1"/></td>
  </tr>
</table>

<div align="center">
  <b>Normalized Confusion Matrix</b><br/><br/>
  <img src="assets/confusion_matrix.png" alt="Confusion Matrix" width="55%"/>
</div>

---

## ⚙️ Methodology

1. **Data Acquisition** — Sentinel-2 median annual composites fetched dynamically from Google Earth Engine for the selected region and year range.

2. **Preprocessing** — Regional bounding boxes (1024 × 1024 px) are tiled into 256 × 256 patches with a 32-pixel overlap to eliminate edge artifacts.

3. **Segmentation** — U-Net (ResNet-34 encoder) classifies each pixel into one of five land-cover classes across all tiles.

4. **Change Detection** — Segmentation maps from Year 1 and Year 2 are compared pixel-wise, computing net class changes, specific transitions (e.g. Forest → Urban), and a full transition matrix.

5. **Reporting** — Risk alert levels are assigned based on forest-loss thresholds. Ecological reasoning, conservation steps, and ESZ enforcement recommendations are generated and compiled into a downloadable PDF report.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Recharts, Axios |
| Backend | FastAPI, Uvicorn |
| AI / ML | PyTorch, segmentation-models-pytorch (U-Net ResNet-34) |
| Satellite Data | Google Earth Engine API, Sentinel-2 L2A |
| PDF Generation | ReportLab |
| Training Data | DeepGlobe Land Cover Dataset |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Made with 🌱 for the conservation of the Western Ghats biodiversity hotspot
</div>
