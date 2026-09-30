"""Read CEDA's live refD1 monthly directory and catalogue metadata."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import requests

from .config import ROOT, load_config

FAMILIES = ("Amon", "AmonZ")
FILE_RE = re.compile(r"_(\d{6})-(\d{6})\.nc$")
TARGETS = ["temperature", "surface_pressure", "geopotential_height", "O3", "H2O", "CO2", "CH4", "N2O", "CO", "HNO3", "NO", "NO2", "HCl", "ClO", "ClONO2", "BrO", "HOCl", "N2O5", "HNO4", "SF6", "CFC-11", "CFC-12", "CFC-113", "CFC-114", "CFC-115", "HCFC-22", "HCFC-141b", "HCFC-142b", "Halon-1211", "Halon-1301", "Halon-2402", "Cly", "Bry", "NOy"]
CATALOGUE = "https://catalogue.ceda.ac.uk/uuid/{}/"


def _get(url):
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _entries(url):
    soup = _get(url)
    path = urlparse(url).path.rstrip("/") + "/"
    result = []
    for row in soup.select("tr"):
        link = row.select_one("td a[href]")
        if not link:
            continue
        name = link.get_text(" ", strip=True)
        href = link.get("href", "")
        if not name or (not href.startswith(path) and not name.endswith(".nc")):
            continue
        cells = row.select("td")
        size_text = cells[2].get_text(" ", strip=True) if len(cells) > 2 else ""
        result.append((name, urljoin(url, href), size_text))
    return result


def _size_bytes(label):
    match = re.search(r"([\d.]+)\s*([KMGT]?B)", label, re.I)
    if not match:
        return None
    return round(float(match[1]) * 1000 ** {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}[match[2].upper()])


def catalogue_variables(uuid):
    soup = _get(CATALOGUE.format(uuid))
    result = {}
    for card in soup.select(".card-body"):
        attributes = {}
        for item in card.select("li"):
            key = item.select_one("i")
            if key:
                attributes[key.get_text(strip=True)] = item.get_text(" ", strip=True).split(":", 1)[-1].strip()
        if "var_id" in attributes:
            result[attributes["var_id"]] = attributes
    return result


def _leaf(task):
    family, variable, base = task
    files = []
    for grid, grid_url, _ in _entries(base):
        if not grid.startswith("g"):
            continue
        for version, version_url, _ in _entries(grid_url):
            if not version.startswith("v"):
                continue
            for filename, url, size in _entries(version_url):
                match = FILE_RE.search(filename)
                if not match or not filename.startswith(variable + "_" + family + "_") or "_refD1_" not in filename:
                    continue
                files.append({"family": family, "variable": variable, "grid": grid, "version": version,
                              "filename": filename, "source_url": url.split("?", 1)[0],
                              "start": match[1], "end": match[2], "size_bytes": _size_bytes(size),
                              "size_label": size, "checksum": None})
    return files


def _remote_dimensions(item):
    variable = item["variable"]
    url = item["source_url"].replace("https://dap.ceda.ac.uk/", "https://dap.ceda.ac.uk/thredds/dodsC/") + ".dds"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        match = re.search(r"\b(?:Float32|Float64|Int16|Int32)\s+" + re.escape(variable) + r"((?:\[[^\]]+\])+)", response.text)
        dims = re.findall(r"\[([^=\]]+)", match[1]) if match else []
        return [d.strip() for d in dims]
    except requests.RequestException:
        return []


def inspect_model(model, workers=12):
    config = load_config(model)
    if model == "WACCM-X":
        return inspect_waccmx(config)
    base = config["model"]["archive_base"].rstrip("/")
    tasks = []
    for family in FAMILIES:
        for variable, url, _ in _entries(base + "/" + family + "/"):
            if re.fullmatch(r"[a-z][a-z0-9]*", variable):
                tasks.append((family, variable, url))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        groups = list(pool.map(_leaf, tasks))
    files = sorted((item for group in groups for item in group), key=lambda x: (x["family"], x["variable"], x["start"]))
    metadata = catalogue_variables(config["model"]["dataset_uuid"])
    representatives = {}
    for item in files:
        representatives.setdefault((item["family"], item["variable"]), item)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        dimensions = dict(zip(representatives, pool.map(_remote_dimensions, representatives.values())))
    variables = []
    for family, variable, _ in tasks:
        matching = [f for f in files if f["family"] == family and f["variable"] == variable]
        attrs = metadata.get(variable, {})
        variables.append({"family": family, "variable": variable, "units": attrs.get("units", ""),
                          "standard_name": attrs.get("standard_name", ""), "long_name": attrs.get("long_name", ""),
                          "first_month": min((f["start"] for f in matching), default=""),
                          "last_month": max((f["end"] for f in matching), default=""),
                          "files": len(matching), "bytes": sum(f["size_bytes"] or 0 for f in matching),
                          "grid": ";".join(sorted({f["grid"] for f in matching})),
                          "dimensions": ";".join(dimensions.get((family, variable), [])),
                          "vertical_coordinate": next((d for d in dimensions.get((family, variable), []) if d in ("lev", "plev")), "")})
    root = ROOT / "products/diagnostics" / "inspection" / model
    root.mkdir(parents=True, exist_ok=True)
    inventory = {"model": model, "experiment": "refD1", "dataset_uuid": config["model"]["dataset_uuid"],
                 "archive_base": base, "inspected_utc": datetime.now(timezone.utc).isoformat(),
                 "files": files, "catalogue_metadata": metadata}
    (root / "archive_inventory.json").write_text(json.dumps(inventory, indent=2))
    with (root / "variables.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(variables[0]) if variables else ["variable"])
        writer.writeheader(); writer.writerows(variables)
    summary = {"model": model, "dataset_uuid": config["model"]["dataset_uuid"],
               "archive_base": base, "monthly_families": list(FAMILIES), "file_count": len(files),
               "listed_size_bytes": sum(f["size_bytes"] or 0 for f in files),
               "first_month": min((f["start"] for f in files), default=None),
               "last_month": max((f["end"] for f in files), default=None),
               "variables_by_family": {family: sorted({f["variable"] for f in files if f["family"] == family}) for family in FAMILIES},
               "sf6_found": any(f["variable"] == "sf6" for f in files),
               "notes": "CEDA listing sizes are rounded estimates; dimensions are from representative OPeNDAP DDS; coefficients require NetCDF or DAS."}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def inspect_waccmx(config):
    """Inventory the published monthly _zm files and one representative OPeNDAP header."""
    base = config["model"]["archive_base"].rstrip("/")
    files = []
    for name, url, size in _entries(base + "/"):
        match = re.search(r"\.(\d{4})-(\d{2})_zm\.nc$", name)
        if match:
            files.append({"family": "monthly_zonal", "variable": "*", "filename": name,
                "source_url": url.split("?", 1)[0], "start": match[1]+match[2],
                "end": match[1]+match[2], "size_bytes": _size_bytes(size),
                "size_label": size, "checksum": None})
    if not files:
        raise FileNotFoundError(f"No WACCM-X monthly zonal files at {base}")
    representative = next((f for f in files if f["start"] == "200001"), files[0])
    dds_url = representative["source_url"].replace("https://dap.ceda.ac.uk/", "https://dap.ceda.ac.uk/thredds/dodsC/") + ".dds"
    response = requests.get(dds_url, timeout=45); response.raise_for_status()
    dimensions = {name: re.findall(r"\[([^=\]]+)", dims) for name, dims in
        re.findall(r"\bFloat32\s+(\w+)((?:\[[^\]]+\])+)", response.text)}
    coordinate_variables = {name: [d.strip() for d in re.findall(r"\[([^=\]]+)", dims)] for name, dims in
        re.findall(r"\bFloat64\s+(\w+)((?:\[[^\]]+\])*)", response.text)
        if name in ("lev", "ilev", "hyam", "hybm", "hyai", "hybi", "P0", "lat", "lon", "time")}
    metadata = catalogue_variables(config["model"]["dataset_uuid"])
    variables = []
    for name, dims in sorted(dimensions.items()):
        attrs = metadata.get(name, {})
        variables.append({"family": "monthly_zonal", "variable": name,
            "units": attrs.get("units", ""), "standard_name": attrs.get("standard_name", ""),
            "long_name": attrs.get("long_name", ""), "first_month": min(f["start"] for f in files),
            "last_month": max(f["end"] for f in files), "files": len(files),
            "bytes": sum(f["size_bytes"] or 0 for f in files), "grid": "native zonal",
            "dimensions": ";".join(dims), "vertical_coordinate": next((d for d in dims if d in ("lev", "ilev")), "")})
    root = ROOT / "products/diagnostics" / "inspection" / "WACCM-X"; root.mkdir(parents=True, exist_ok=True)
    inventory = {"model": "WACCM-X", "experiment": config["model"]["experiment"],
        "dataset_uuid": config["model"]["dataset_uuid"], "archive_base": base,
        "inspected_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "catalogue_metadata": metadata, "dimensions": dimensions, "coordinate_variables": coordinate_variables}
    (root / "archive_inventory.json").write_text(json.dumps(inventory, indent=2))
    with (root / "variables.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(variables[0])); writer.writeheader(); writer.writerows(variables)
    summary = {"model": "WACCM-X", "dataset_uuid": config["model"]["dataset_uuid"],
        "archive_base": base, "monthly_families": ["monthly_zonal"], "file_count": len(files),
        "listed_size_bytes": sum(f["size_bytes"] or 0 for f in files),
        "first_month": min(f["start"] for f in files), "last_month": max(f["end"] for f in files),
        "variables_by_family": {"monthly_zonal": sorted(dimensions)}, "sf6_found": "SF6" in dimensions,
        "coordinate_variables": coordinate_variables,
        "notes": "Monthly zonal source files contain many variables; archive listing sizes are rounded."}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_inventory(model):
    path = ROOT / "products/diagnostics" / "inspection" / model / "archive_inventory.json"
    if not path.exists():
        raise FileNotFoundError(f"Run inspect --model {model} first: {path}")
    return json.loads(path.read_text())


def variable_matrix():
    inventories = {m: load_inventory(m) for m in ("GEOSCCM", "EMAC", "WACCM-X")}
    configs = {m: load_config(m) for m in inventories}
    coverage = {}
    for model in inventories:
        path = ROOT / "products/diagnostics" / "coverage" / model / "vertical_coverage.json"
        coverage[model] = {row["variable"]: row for row in json.loads(path.read_text())} if path.exists() else {}
    names = list(dict.fromkeys(TARGETS + [k for c in configs.values() for k in c["variables"]]))
    rows = []
    for canonical in names:
        row = {"canonical_species": canonical}
        units = []
        for model in inventories:
            variable = configs[model]["variables"].get(canonical, "")
            found = bool(variable and (variable in inventories[model].get("dimensions", {}) if model == "WACCM-X" else any(f["variable"] == variable for f in inventories[model]["files"])))
            row[model + "_variable"] = variable if found else ""
            row[model + "_found"] = found
            row[model + "_time_coverage"] = (f"{min(f['start'] for f in inventories[model]['files'])}-{max(f['end'] for f in inventories[model]['files'])}" if found else "")
            row[model + "_units"] = inventories[model]["catalogue_metadata"].get(variable, {}).get("units", "") if found else ""
            measured = coverage[model].get(variable, {})
            row[model + "_vertical_coverage"] = (f"{measured['highest_valid_pressure_pa']}-{measured['lowest_valid_pressure_pa']} Pa" if measured and measured["highest_valid_pressure_pa"] is not None else "pending data-based diagnostic" if found else "")
            if found:
                units.append(inventories[model]["catalogue_metadata"].get(variable, {}).get("units", ""))
        row["units"] = ";".join(sorted(set(filter(None, units))))
        row["notes"] = "family sum" if canonical in ("Cly", "Bry", "NOy") else ""
        rows.append(row)
    path = ROOT / "products/comparison" / "variable_matrix.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return path
