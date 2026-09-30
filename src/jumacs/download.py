"""Selective, incremental monthly downloads with a preflight size estimate."""
import hashlib
import json
from pathlib import Path
import re
import requests

from .archive import load_inventory
from .config import ROOT, load_config, is_waccmx, model_slug


def select_files(model, start_year, end_year, variables=None):
    config = load_config(model)
    inventory = load_inventory(model)
    if start_year > end_year:
        raise ValueError("start year exceeds end year")
    if is_waccmx(config):
        first = min(int(f["start"][:4]) for f in inventory["files"])
        last = max(int(f["end"][:4]) for f in inventory["files"])
        if start_year < first or end_year > last:
            raise ValueError(f"WACCM-X monthly zonal archive covers {first}-{last}; requested {start_year}-{end_year}")
    wanted = set(variables or config["variables"].values())
    wanted = {config["variables"].get(v, v) for v in wanted}
    if is_waccmx(config):
        available = set(inventory.get("dimensions", {}))
        selected = [f for f in inventory["files"] if start_year <= int(f["start"][:4]) <= end_year] if wanted & available else []
        return selected, sorted(wanted - available)
    # Precomputed AmonZ is required for species that have no Amon field.
    available_amon = {f["variable"] for f in inventory["files"] if f["family"] == "Amon"}
    preferred = config.get("preferred_families", {})
    selected = []
    for f in inventory["files"]:
        if f["variable"] not in wanted or int(f["end"][:4]) < start_year or int(f["start"][:4]) > end_year:
            continue
        family = preferred.get(f["variable"])
        if family and f["family"] != family:
            continue
        if not family and f["family"] == "AmonZ" and f["variable"] in available_amon:
            continue
        selected.append(f)
    # A CEDA variable can have multiple published versions for the same months.
    # Keep the latest version so each monthly timestamp enters a product once.
    latest = {}
    for f in selected:
        key = (f["family"], f["variable"], f.get("grid", ""), f["start"], f["end"])
        if key not in latest or f.get("version", "") > latest[key].get("version", ""):
            latest[key] = f
    selected = list(latest.values())
    missing = sorted(wanted - {f["variable"] for f in selected})
    return selected, missing


def manifest_path(model):
    config = load_config(model)
    slug = config["model"].get("slug") or (model_slug(model) + "_refd1" if not is_waccmx(config) else "waccmx")
    return ROOT / "products/manifests" / f"{slug}.json"


def _record(config, model, f):
    local = ROOT / config["paths"]["raw"] / (f["filename"] if is_waccmx(config) else f["family"] + "/" + f["variable"] + "/" + f["filename"])
    complete = local.exists() and (not f["size_bytes"] or abs(local.stat().st_size - f["size_bytes"]) < f["size_bytes"] * .15)
    return {**f, "model": model, "experiment": config["model"]["experiment"], "local_path": str(local.relative_to(ROOT)),
            "download_status": "complete" if complete else "pending"}


def plan(model, start_year, end_year, variables=None):
    files, missing = select_files(model, start_year, end_year, variables)
    config = load_config(model)
    records = [_record(config, model, f) for f in files]
    payload = {"model": model, "experiment": config["model"]["experiment"], "start_year": start_year, "end_year": end_year,
               "selected_bytes_estimate": sum(f["size_bytes"] or 0 for f in files),
               "size_is_rounded_listing_estimate": True, "missing_variables": missing, "files": records}
    path = manifest_path(model); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def full_plan(model):
    """Plan a complete mirror of the configured archive member: every inventoried file and family."""
    config = load_config(model)
    inventory = load_inventory(model)
    files = list(inventory["files"])
    waccmx = is_waccmx(config)
    plannable = [f for f in files if f.get("source_url")]
    latest = {}
    for f in plannable:
        key = (f["family"], f["filename"]) if waccmx else (f["family"], f["variable"], f.get("grid", ""), f["start"], f["end"])
        if key not in latest or f.get("version", "") > latest[key].get("version", ""):
            latest[key] = f
    records = [_record(config, model, latest[key]) for key in sorted(latest)]
    families = {}
    for f in files:
        families.setdefault(f["family"], {"inventory": 0, "planned": 0})["inventory"] += 1
    for r in records:
        families[r["family"]]["planned"] += 1
    members = sorted({m for r in records for m in re.findall(r"r\d+i\d+p\d+f\d+", r["filename"])})
    complete = [r for r in records if r["download_status"] == "complete"]
    pending = [r for r in records if r["download_status"] != "complete"]
    model_config = config["model"]
    payload = {"model": model, "experiment": model_config["experiment"], "mode": "full_archive",
               "dataset_uuid": model_config["dataset_uuid"],
               "member": ",".join(members) if members else "n/a",
               "first_month": min(f["start"] for f in files), "last_month": max(f["end"] for f in files),
               "inventory_file_count": len(files), "planned_file_count": len(records),
               "deduplicated_count": len(plannable) - len(records),
               "unplannable_count": len(files) - len(plannable),
               "inventory_bytes": sum(f["size_bytes"] or 0 for f in files),
               "planned_bytes": sum(r["size_bytes"] or 0 for r in records),
               "missing_remote_size_count": sum(1 for f in files if not f["size_bytes"]),
               "families": {k: families[k] for k in sorted(families)},
               "already_complete_file_count": len(complete),
               "already_complete_bytes": sum(r["size_bytes"] or 0 for r in complete),
               "pending_file_count": len(pending),
               "pending_bytes": sum(r["size_bytes"] or 0 for r in pending),
               "selected_bytes_estimate": sum(r["size_bytes"] or 0 for r in records),
               "size_is_rounded_listing_estimate": True, "files": records}
    if model_config.get("dataset_uuid_scope"):
        payload["dataset_uuid_scope"] = model_config["dataset_uuid_scope"]
    path = manifest_path(model); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def _transfer(record):
    dest = ROOT / record["local_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if record["download_status"] == "complete":
        return "complete"
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    url = record["source_url"] + "?download=1"
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        with part.open("ab" if offset else "wb") as stream:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
    expected = record.get("size_bytes")
    if expected and abs(part.stat().st_size - expected) > expected * .15:
        raise IOError(f"Downloaded size differs from CEDA rounded listing: {dest}")
    part.replace(dest)
    if record.get("checksum"):
        algorithm, expected_hash = record["checksum"].split(":", 1)
        digest = hashlib.new(algorithm)
        with dest.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            dest.unlink(); raise IOError(f"Checksum mismatch: {dest}")
    return "complete"


def download(payload):
    path = manifest_path(payload["model"])
    for record in payload["files"]:
        try:
            record["download_status"] = _transfer(record)
        except Exception as exc:
            record["download_status"] = "failed"
            record["error"] = str(exc)
            path.write_text(json.dumps(payload, indent=2))
            raise
        path.write_text(json.dumps(payload, indent=2))
    return payload
