#!/usr/bin/env python3
"""Embed the plot catalog into a static offline JuMACS browser page."""
import json
import os
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def build_index():
    catalog = json.loads((ROOT / "site/catalog.json").read_text())
    comparison = ROOT / "site/comparison_catalog.json"
    if comparison.exists():
        catalog.extend(json.loads(comparison.read_text()))
    compact = ROOT / "site/compact_catalog.json"
    if compact.exists():
        catalog.extend(json.loads(compact.read_text()))
    template = (ROOT / "scripts/index_template.html").read_text()
    payload = json.dumps(catalog, ensure_ascii=False).replace("</", "<\\/")
    destination = ROOT / "site/index.html"
    base = os.environ.get("JUMACS_PRODUCT_URL_BASE", "").strip()
    if base:
        links = "Compact application files: " + " and ".join(
            f'<a href="{escape(base.rstrip("/") + "/" + filename, quote=True)}">{label}</a>'
            for label, filename in (
                ("GEOSCCM→WACCM-X", "jumacs_geosccm_waccmx_1985-2014_5deg_1km.nc"),
                ("EMAC→WACCM-X", "jumacs_emac_waccmx_1985-2014_5deg_1km.nc"),
            )
        ) + "."
    else:
        links = "Compact NetCDF products are stored separately under products/climatology/combined."
    destination.write_text(template.replace("__JUMACS_CATALOG__", payload).replace("__JUMACS_DATA_LINKS__", links))
    return destination

if __name__ == "__main__":
    print(build_index())
