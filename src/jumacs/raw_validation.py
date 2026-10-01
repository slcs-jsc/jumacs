"""Offline validation of the locally mirrored raw archive (manifests + data/raw only).

Never contacts CEDA, never regenerates inventories, never downloads or modifies raw files.
Deep checks are limited to a small deterministic set of representative files.
"""
import csv
import json

from .config import ROOT, load_config, model_metadata, registry
from .download import manifest_path

NETCDF_MAGICS = (b"CDF\x01", b"CDF\x02", b"CDF\x03", b"\x89HDF\r\n\x1a\n", b"\x0e\x03\x13\x01")
CANONICAL_PRIORITY = ("temperature", "surface_pressure", "geopotential_height", "H2O", "O3", "HNO3", "OH")
REFERENCE_PERIOD = (1985, 2014)
MAX_REPRESENTATIVES = 6
SIZE_TOLERANCE = 0.15

RAW_VALIDATION_COLUMNS = ["model", "kind", "status", "manifest_path", "expected_files", "present_files",
                          "missing_files", "partial_files", "invalid_signature_files", "xarray_failures",
                          "listed_bytes", "present_bytes", "representative_files_checked", "time_decode_ok",
                          "reasons", "notes"]


def _manifest(model, root=ROOT):
    path = manifest_path(model) if root == ROOT else root / manifest_path(model).relative_to(ROOT)
    return path


def _signature_ok(path):
    try:
        with path.open("rb") as stream:
            head = stream.read(8)
    except OSError:
        return False
    return any(head.startswith(magic) for magic in NETCDF_MAGICS)


def select_representatives(model, files, root=ROOT, limit=MAX_REPRESENTATIVES):
    """Deterministic small selection: canonical priority, both families, reference period, smaller files."""
    config = load_config(model)
    families = sorted({f["family"] for f in files})
    wanted = [c for c in CANONICAL_PRIORITY if c in config.get("variables", {})]
    selected, reasons = [], []
    used = set()

    def candidates(family, canonical):
        native = config["variables"][canonical]
        pool = [f for f in files if f["family"] == family and f["variable"] == native and f["filename"] not in used]
        def rank(f):
            overlap = not (int(f["end"][:4]) < REFERENCE_PERIOD[0] or int(f["start"][:4]) > REFERENCE_PERIOD[1])
            return (0 if overlap else 1, f.get("size_bytes") or 0, f["filename"])
        return sorted(pool, key=rank)

    for family in families:
        for canonical in wanted:
            pool = candidates(family, canonical)
            if pool:
                pick = pool[0]
                used.add(pick["filename"])
                selected.append(pick)
                overlaps = not (int(pick["end"][:4]) < REFERENCE_PERIOD[0] or int(pick["start"][:4]) > REFERENCE_PERIOD[1])
                reasons.append(f"{pick['filename']}: family {family}, canonical {canonical}"
                               + (", overlaps 1985-2014" if overlaps else ""))
                break
    for family in families:
        leftover = sorted((f for f in files if f["family"] == family and f["filename"] not in used),
                          key=lambda f: (f.get("size_bytes") or 0, f["filename"]))
        if leftover and len(selected) < limit:
            pick = leftover[0]
            used.add(pick["filename"])
            selected.append(pick)
            reasons.append(f"{pick['filename']}: family {family}, smallest available fallback")
    return selected[:limit], reasons


def _open_representatives(model, files, root):
    import xarray as xr
    from .config import is_waccmx
    config = load_config(model)
    checked, failures, time_decode = [], [], None
    for record in files:
        path = root / record["local_path"]
        try:
            with xr.open_dataset(path, decode_times=True, use_cftime=True) as ds:
                native = record["variable"]
                if not is_waccmx(config) and native not in ds and not any(native == v for v in ds.variables):
                    failures.append(f"{path.name}: variable {native} absent")
                    continue
                time_ok = "time" in ds.dims and ds.sizes["time"] > 0 and "time" in ds.variables
                if time_ok:
                    try:
                        ds["time"].encoding["units"]
                        values = ds["time"].values
                        time_ok = values.size > 0 and hasattr(values.ravel()[0], "calendar")
                    except Exception:
                        time_ok = False
                if is_waccmx(config):
                    time_ok = "time" in ds.dims
                checked.append(path.name)
                if time_decode is None:
                    time_decode = bool(time_ok)
                elif time_ok is False:
                    time_decode = False
        except Exception as exc:
            failures.append(f"{path.name}: xarray open failed: {exc}")
    return checked, failures, time_decode


def validate_model(model, root=ROOT):
    meta = model_metadata(model)
    path = _manifest(model, root)
    row = {"model": model, "kind": meta["kind"], "status": "ok", "manifest_path": str(path.relative_to(root)),
           "expected_files": 0, "present_files": 0, "missing_files": 0, "partial_files": 0,
           "invalid_signature_files": 0, "xarray_failures": 0, "listed_bytes": 0, "present_bytes": 0,
           "representative_files_checked": 0, "time_decode_ok": None, "reasons": [], "notes": []}
    reasons, notes = row["reasons"], row["notes"]
    if not registry()[model].exists():
        raise FileNotFoundError(path)
    if not (load_config(model).get("capabilities") or {}).get("download"):
        row["status"] = "not_applicable"
        reasons.append("capability_missing:download")
        return row
    if not path.exists():
        row["status"] = "not_downloaded"
        reasons.append("manifest_missing")
        return row
    payload = json.loads(path.read_text())
    records = payload.get("files", [])
    if not records:
        row["status"] = "not_downloaded"
        reasons.append("manifest_without_files")
        return row
    row["expected_files"] = len(records)
    row["listed_bytes"] = sum(r.get("size_bytes") or 0 for r in records)
    raw_root = root / load_config(model)["paths"]["raw"]
    partial = sorted(str(p.relative_to(root)) for p in raw_root.rglob("*.part")) if raw_root.exists() else []
    missing, size_mismatch, bad_signature = [], [], []
    for record in records:
        local = root / record["local_path"]
        if local.exists():
            row["present_files"] += 1
            actual = local.stat().st_size
            row["present_bytes"] += actual
            expected = record.get("size_bytes")
            if expected and abs(actual - expected) > expected * SIZE_TOLERANCE:
                size_mismatch.append(local.name)
            if not _signature_ok(local):
                row["invalid_signature_files"] += 1
                bad_signature.append(local.name)
        else:
            missing.append(record["local_path"])
    row["missing_files"] = len(missing)
    row["partial_files"] = len(partial)
    row["xarray_failures"] = 0
    present = [r for r in records if (root / r["local_path"]).exists()]
    representatives, why = select_representatives(model, present, root)
    if representatives:
        checked, failures, time_decode = _open_representatives(model, representatives, root)
        row["representative_files_checked"] = len(checked)
        row["xarray_failures"] = len(failures)
        row["time_decode_ok"] = time_decode
        for failure in failures:
            reasons.append("xarray_open_failed:" + failure)
        if time_decode is False:
            reasons.append("time_decode_failed")
        notes.append("representatives: " + "; ".join(why))
    notes.append("full-file hashing intentionally not performed (metadata-only audit)")
    if missing:
        reasons.append(f"missing_expected_files:{len(missing)}:" + "; ".join(missing[:5]))
    if partial:
        reasons.append(f"leftover_part_files:{len(partial)}:" + "; ".join(partial[:5]))
    if size_mismatch:
        reasons.append(f"size_mismatch:{len(size_mismatch)}:" + "; ".join(size_mismatch[:5]))
    if bad_signature:
        reasons.append(f"invalid_signature:{len(bad_signature)}:" + "; ".join(bad_signature[:5]))
    if row["present_files"] == 0:
        row["status"] = "not_downloaded"
    elif row["xarray_failures"]:
        row["status"] = "invalid"
    elif row["invalid_signature_files"]:
        row["status"] = "invalid"
    elif missing or partial or size_mismatch:
        row["status"] = "incomplete"
    elif row["time_decode_ok"] is False:
        row["status"] = "invalid"
    return row


def _md_table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    return lines


def markdown_raw_validation_report(rows):
    lines = ["# JuMACS raw archive validation", "",
             "Offline validation of locally mirrored raw archives against the download manifests "
             "(`jumacs validate-raw`). No network access, no transfer, raw files read-only.", "",
             "Statuses: `ok`, `incomplete` (missing/partial/size-mismatched files), `invalid` "
             "(unreadable/bad signature/time decode), `not_downloaded`, `not_applicable`.", ""]
    lines += _md_table(RAW_VALIDATION_COLUMNS, [{**row, "reasons": "; ".join(row["reasons"])} for row in rows])
    problems = [row for row in rows if row["status"] != "ok"]
    lines += ["", "## Exceptions", ""]
    if problems:
        lines += [f"- **{row['model']}** ({row['status']}): " + ("; ".join(row["reasons"]) or "see table")
                  for row in problems]
    else:
        lines.append("- all validated models complete and readable at metadata level.")
    lines.append("")
    return "\n".join(lines)


def validate_models(models=None, root=ROOT):
    models = tuple(models) if models else tuple(registry())
    return [validate_model(model, root) for model in models]


def write_raw_validation(rows=None, models=None, root=ROOT):
    rows = list(rows) if rows is not None else validate_models(models, root)
    diagnostics = root / "products/diagnostics/raw_validation"
    diagnostics.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (diagnostics / f"{row['model']}.json").write_text(json.dumps(row, indent=2))
    comparison = root / "products/comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    csv_path = comparison / "raw_validation.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_VALIDATION_COLUMNS)
        writer.writeheader()
        writer.writerows({**row, "reasons": "; ".join(row["reasons"])} for row in rows)
    md_path = comparison / "raw_validation.md"
    md_path.write_text(markdown_raw_validation_report(rows))
    return [csv_path, md_path]
