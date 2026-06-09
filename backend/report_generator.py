"""
backend/report_generator.py — PDF report generator using ReportLab.
"""

import base64
import io
import logging
import os
import sys
from datetime import datetime
from typing import Dict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CLASS_NAMES, REPORTS_DIR

logger = logging.getLogger(__name__)


def _np_to_b64(img_np: np.ndarray, fmt: str = "PNG") -> str:
    """Convert numpy RGB image to base64 string for embedding."""
    pil = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def generate_pdf_report(
    analysis: Dict,
    img1: np.ndarray,
    img2: np.ndarray,
    seg1: np.ndarray,
    seg2: np.ndarray,
    change_map: np.ndarray,
) -> str:
    """
    Generate a PDF report and save it to the reports directory.

    Returns the absolute path to the saved PDF.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image as RLImage, Paragraph, SimpleDocTemplate,
            Spacer, Table, TableStyle,
        )
    except ImportError:
        logger.warning("reportlab not installed — skipping PDF generation")
        return ""

    region = analysis["region"]
    year1  = analysis["year1"]
    year2  = analysis["year2"]
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname  = f"report_{region}_{year1}_{year2}_{ts}.pdf"
    fpath  = os.path.join(REPORTS_DIR, fname)

    doc    = SimpleDocTemplate(fpath, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=18, spaceAfter=12, textColor=colors.darkgreen,
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
    body = styles["BodyText"]

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Western Ghats Deforestation & Land Cover Change Detection", title_style))
    story.append(Paragraph(f"Region: <b>{region}</b>  |  Years: {year1} → {year2}", body))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", body))
    story.append(Spacer(1, 0.4 * cm))

    # ── Alert ─────────────────────────────────────────────────────────────────
    alert   = analysis["alert_level"]
    a_color = {"Critical Risk": "#d7191c", "High Risk": "#fdae61",
                "Moderate Risk": "#fee090", "Low Risk": "#a6d96a"}.get(alert, "#cccccc")
    story.append(Paragraph(f'<font color="{a_color}"><b>Alert Level: {alert}</b></font>', h2))
    story.append(Spacer(1, 0.3 * cm))

    # ── Forest summary ────────────────────────────────────────────────────────
    story.append(Paragraph("Forest Cover Summary", h2))
    f = analysis["forest"]
    forest_data = [
        ["Metric", "Value"],
        ["Forest Cover (Year 1)", f"{f['year1_pct']:.2f}%"],
        ["Forest Cover (Year 2)", f"{f['year2_pct']:.2f}%"],
        ["Forest Loss",           f"{f['loss_pct']:.2f}%"],
        ["Forest Gain",           f"{f['gain_pct']:.2f}%"],
        ["Net Forest Change",     f"{f['net_change_pct']:+.2f}%"],
    ]
    t = Table(forest_data, colWidths=[9 * cm, 5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # ── Land cover percentages ────────────────────────────────────────────────
    story.append(Paragraph("Land Cover Percentages", h2))
    lc_data = [["Class", f"Year {year1} %", f"Year {year2} %", "Change %"]]
    for name in CLASS_NAMES.values():
        p1 = analysis["class_pct_y1"].get(name, 0)
        p2 = analysis["class_pct_y2"].get(name, 0)
        lc_data.append([name, f"{p1:.2f}%", f"{p2:.2f}%", f"{p2-p1:+.2f}%"])
    t2 = Table(lc_data, colWidths=[5 * cm, 3 * cm, 3 * cm, 3 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c7bb6")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4 * cm))

    # ── Transitions ───────────────────────────────────────────────────────────
    story.append(Paragraph("Key Land Cover Transitions", h2))
    tr_data = [["Transition", "Area (%)"]]
    for k, v in analysis["transitions"].items():
        tr_data.append([k, f"{v:.3f}%"])
    t3 = Table(tr_data, colWidths=[10 * cm, 4 * cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d7191c")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t3)
    story.append(Spacer(1, 0.4 * cm))

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("Recommendations", h2))
    for rec in analysis["recommendations"]:
        story.append(Paragraph(f"• {rec}", body))
    story.append(Spacer(1, 0.3 * cm))

    # ── Academic Justification & Conclusion ───────────────────────────────────
    story.append(Paragraph("Academic Analysis & Ecological Justification", h2))
    
    academic = analysis.get("academic_analysis")
    
    if academic:
        story.append(Paragraph("<b>1. Reasoning for Land Cover Shifts</b>", body))
        story.append(Paragraph(academic["reasoning"], body))
        story.append(Spacer(1, 0.2 * cm))

        story.append(Paragraph("<b>2. Steps for Ecological Balance</b>", body))
        for step in academic["steps"]:
            step_html = step.replace("**", "<b>", 1).replace("**", "</b>", 1)
            story.append(Paragraph(f"• {step_html}", body))
        story.append(Spacer(1, 0.2 * cm))

        story.append(Paragraph("<b>3. Conclusive Points</b>", body))
        for conc in academic["conclusions"][:3]:
            conc_html = conc.replace("**", "<b>", 1).replace("**", "</b>", 1)
            story.append(Paragraph(f"• {conc_html}", body))
        story.append(Spacer(1, 0.4 * cm))
    else:
        story.append(Paragraph("<b>1. Drivers of Land Cover Change</b>", body))
        story.append(Paragraph(
            "The observed shifts in land cover—specifically the reduction of dense forest—are predominantly driven by anthropogenic pressures. "
            "Studies utilizing the Western Ghats Spatial Decision Support System (WGSDSS) indicate that agricultural expansion (such as commercial tea, coffee, and rubber plantations) "
            "and rapid urbanization are the primary catalysts for this deforestation. The corresponding increase in urban and barren percentages reflects "
            "aggressive infrastructure development and localized resource extraction, which collectively degrade soil stability and fragment primary habitats.", body))
        story.append(Spacer(1, 0.2 * cm))

        story.append(Paragraph("<b>2. Steps for Ecological Balance</b>", body))
        story.append(Paragraph(
            "To mitigate these adverse effects, targeted ecological interventions are essential. Primary steps include the strict enforcement of "
            "Eco-Sensitive Zones (ESZs) as originally outlined by the Gadgil and Kasturirangan Committee Reports. Additionally, integrating "
            "agroforestry into buffer zones, restoring critical wildlife corridors, and transitioning towards sustainable eco-tourism will help "
            "reconcile economic development with long-term biodiversity conservation.", body))
        story.append(Spacer(1, 0.2 * cm))

        story.append(Paragraph("<b>3. Conclusive Points</b>", body))
        story.append(Paragraph(
            "• <b>Accelerated Habitat Fragmentation:</b> The transition matrix reveals a concerning trend of forest edges converting into agricultural and urban zones, isolating endemic species populations.<br/>"
            "• <b>Hydrological Vulnerability:</b> The replacement of natural forest with built-up and barren land directly compromises groundwater recharge, threatening the water security of peninsular India.<br/>"
            "• <b>Urgent Policy Execution:</b> Without immediate, ground-level enforcement of existing environmental regulations, the ongoing fragmentation will cause irreversible damage to the region's climate resilience.", body))
        story.append(Spacer(1, 0.4 * cm))

    # ── Images (thumbnail) ───────────────────────────────────────────────────
    def _add_img(arr, caption, width=7):
        pil  = Image.fromarray(arr.astype(np.uint8)).resize((300, 300))
        buf  = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        story.append(Paragraph(caption, body))
        story.append(RLImage(buf, width=width * cm, height=width * cm))
        story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Visual Maps", h2))
    _add_img(img1, f"Satellite Image — Year {year1}")
    _add_img(img2, f"Satellite Image — Year {year2}")
    _add_img(seg1, f"Segmentation Map — Year {year1}")
    _add_img(seg2, f"Segmentation Map — Year {year2}")
    _add_img(change_map, "Change Detection Map  (Red=Forest Loss, Green=Forest Gain, Orange=Urbanisation)")

    doc.build(story)
    logger.info("PDF report saved: %s", fpath)
    return fpath
