"""Offline coordinate-convention audit of downloaded raw archives.

Answers: what coordinate conventions does each model actually use, and can the
current JuMACS reader support them? Descriptive only: no remapping, no config edits,
no capability changes, no network access.
"""
import csv
import json
import re

from .config import ROOT, load_config, model_metadata, registry, is_waccmx
from .coordinates import formula_terms
from .raw_validation import select_representatives, _manifest

COORDINATE_AUDIT_COLUMNS = ["model", "status", "families", "horizontal_grid", "vertical_coordinate",
                            "time_decode", "reader_compatibility", "main_action", "representatives",
                            "reasons", "notes"]

HYBRID_STD = {"atmosphere_hybrid_sigma_pressure_coordinate": "hybrid_sigma_pressure",
              "atmosphere_hybrid_height_coordinate": "hybrid_height",
              "altitude": "altitude"}

FORMULA_RE = re.compile(r"\b(ap|a|b|ps|p0)\s*:\s*([A-Za-z0-9_\-.]+)")


def _first(ds, names):
    for name in names:
        if name and name in ds.variables:
            return name
    return None


def _detected(ds, standard_name=None, units_prefix=None):
    hits = []
    for name, var in ds.variables.items():
        if standard_name and var.attrs.get("standard_name") == standard_name:
            hits.append(name)
        elif units_prefix and str(var.attrs.get("units", "")).lower().startswith(units_prefix):
            hits.append(name)
    return sorted(hits)


def _resolve(ds, cfg_name, standard_name=None, fallbacks=()):
    if cfg_name and cfg_name in ds:
        return cfg_name, True
    detected = _detected(ds, standard_name)
    if detected:
        return detected[0], False
    for name in fallbacks:
        if name in ds:
            return name, name == cfg_name
    return None, False


def audit_horizontal(ds, config, variable):
    names = config.get("coordinates") or {}
    lat, mapped_lat = _resolve(ds, names.get("latitude"), "latitude", ("lat", "latitude"))
    lon, mapped_lon = _resolve(ds, names.get("longitude"), "longitude", ("lon", "longitude"))
    spatial = [d for d in ds[variable].dims if d != "time"]
    lat_dims = list(ds[lat].dims) if lat in ds else []
    lon_dims = list(ds[lon].dims) if lon in ds else []
    info = {"latitude": lat, "longitude": lon, "mapped_latitude": mapped_lat, "mapped_longitude": mapped_lon,
            "latitude_dims": lat_dims, "longitude_dims": lon_dims}
    curvilinear = (lat in ds and ds[lat].ndim >= 2) or (lon in ds and ds[lon].ndim >= 2)
    scalar_lon = lon in ds and ds[lon].ndim == 0
    regular = (lon in ds and lon in ds[variable].dims and lat in ds and lat in ds[variable].dims)
    curved_ok = curvilinear and lat in ds and lon in ds and bool(lat_dims) and all(d in spatial for d in lat_dims)
    if not regular and not curved_ok:
        info["grid"] = "already_zonal" if (lon not in ds or scalar_lon) and len(spatial) <= 2 \
            else "coordinate_names_missing"
        return info
    lats = ds[lat].values
    lons = ds[lon].values
    info["n_lat"] = int(lats.shape[0])
    info["n_lon"] = int(lons.shape[-1])
    info["latitude_units"] = ds[lat].attrs.get("units", "")
    info["longitude_units"] = ds[lon].attrs.get("units", "")
    flat = lons.ravel()
    info["longitude_min"] = float(flat.min())
    info["longitude_max"] = float(flat.max())
    span = info["longitude_max"] - info["longitude_min"]
    unique = sorted(set(flat.tolist()))
    gaps = [unique[i + 1] - unique[i] for i in range(len(unique) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    effective = span + median_gap
    info["longitude_monotonic"] = bool(flat.size <= 1 or flat.tolist() == sorted(flat.tolist())
                                       or flat.tolist() == sorted(flat.tolist(), reverse=True))
    info["global_coverage"] = bool(effective >= 350)
    info["regular_spacing"] = bool(max(gaps) <= 1.5 * median_gap and min(gaps) >= 0.5 * median_gap) if gaps else True
    if effective < 350:
        info["longitude_convention"] = "partial"
    elif info["longitude_min"] >= -1:
        info["longitude_convention"] = "0..360"
    else:
        info["longitude_convention"] = "-180..180"
    info["grid"] = "curvilinear" if curvilinear else ("regular_latlon" if info["global_coverage"] else "regular_partial")
    return info


def audit_vertical(ds, config, variable):
    names = config.get("coordinates") or {}
    var = ds[variable]
    gridish = {"lat", "lon", "latitude", "longitude"}
    level_names = [d for d in var.dims if d != "time" and d in ds.variables]
    level_candidates = [d for d in level_names if d not in gridish
                        and ds[d].attrs.get("standard_name") not in ("latitude", "longitude")]
    explicit = _first(ds, [names.get("level")] if names.get("level") else [])
    level = explicit if explicit in level_candidates else None
    if not level:
        for std, kind in [("air_pressure", "pressure_level"),
                          ("atmosphere_hybrid_sigma_pressure_coordinate", "hybrid_sigma_pressure"),
                          ("atmosphere_hybrid_height_coordinate", "hybrid_height"), ("altitude", "altitude")]:
            hits = [h for h in _detected(ds, std) if h in var.dims and h in level_candidates]
            if hits:
                level = hits[0]
                break
    if not level:
        for cand in level_candidates:
            if any(k in ds[cand].attrs for k in ("formula_terms", "formula_term", "z_factors")):
                level = cand
                break
    info = {"level_dim": None, "level_variable": level, "level_from_config": bool(explicit),
            "detected_level_candidates": level_candidates}
    if not level:
        if not level_candidates:
            info["coordinate_type"] = "none"
        elif any(s in ds.variables or any("sigma" in c.lower() for c in level_candidates) for s in ("sigma",)):
            info["coordinate_type"] = "pure_sigma"
        else:
            info["coordinate_type"] = "unknown"
        return info
    info["level_dim"] = level if level in var.dims else None
    coord = ds[level]
    info["n_levels"] = int(coord.size)
    info["units"] = coord.attrs.get("units", "")
    info["standard_name"] = coord.attrs.get("standard_name", "")
    info["positive"] = coord.attrs.get("positive", "")
    raw_terms = coord.attrs.get("formula_terms") or coord.attrs.get("z_factors")
    non_cf = None if raw_terms else coord.attrs.get("formula_term")
    terms = formula_terms(coord)
    alt_terms = {k: v for k, v in FORMULA_RE.findall(non_cf)} if non_cf else {}
    info["formula_terms"] = dict(terms or alt_terms)
    info["formula_terms_source"] = "observed" if raw_terms else (
        "observed_non_cf" if alt_terms else ("inferred_from_config" if terms else None))
    if info["standard_name"] in HYBRID_STD:
        info["coordinate_type"] = HYBRID_STD[info["standard_name"]]
    elif info["units"].lower() in ("pa", "pascal", "pascals") or info["standard_name"] == "air_pressure":
        info["coordinate_type"] = "pressure_level"
    elif terms or alt_terms:
        info["coordinate_type"] = "hybrid_sigma_pressure" \
            if (info["formula_terms"].get("ap") or info["formula_terms"].get("a")) else "other"
    else:
        info["coordinate_type"] = "unknown"
    terms = info["formula_terms"]
    a_name = terms.get("ap") or terms.get("a") or names.get("hybrid_a")
    b_name = terms.get("b") or names.get("hybrid_b")
    ps_name = terms.get("ps") or _first(ds, [names.get("surface_pressure")] if names.get("surface_pressure") else []) \
        or _first(ds, _detected(ds, "surface_air_pressure"))
    p0_name = terms.get("p0") or names.get("reference_pressure")
    info["hybrid_a_name"], info["hybrid_b_name"] = a_name, b_name
    info["hybrid_a_present"] = a_name in ds if a_name else None
    info["hybrid_b_present"] = b_name in ds if b_name else None
    info["surface_pressure_name"] = ps_name
    info["surface_pressure_present"] = ps_name in ds if ps_name else None
    info["reference_pressure_name"] = p0_name
    info["reference_pressure_present"] = p0_name in ds if p0_name else None
    if info["coordinate_type"] == "pressure_level" and coord.ndim == 1 and coord.size:
        info["top_pressure"] = float(min(coord.values.tolist()))
        info["bottom_pressure"] = float(max(coord.values.tolist()))
    return info


def audit_time(ds, config, path):
    if "time" not in ds.variables and "time" not in ds.dims:
        return {"time_variable": None, "time_decode_ok": False}
    time = ds["time"]
    info = {"time_variable": "time", "units": time.attrs.get("units", ""),
            "bounds_variable": time.attrs.get("bounds", ""), "time_decode_ok": True}
    try:
        values = time.values
        first, last = values.ravel()[0], values.ravel()[-1]
        info["calendar"] = getattr(first, "calendar", "")
        info["first_time"] = str(first)
        info["last_time"] = str(last)
        info["n_times"] = int(values.size)
        cadence = None
        if values.size > 1:
            step = values.ravel()[min(3, values.size - 1)] - values.ravel()[0]
            days = step.days + step.seconds / 86400
            days /= min(3, values.size - 1)
            cadence = round(days, 2)
            info["cadence_days"] = cadence
        info["monthly"] = bool(cadence and 26 <= cadence <= 32) or (is_waccmx(config) and values.size == 1)
        if is_waccmx(config):
            info["timestamp_position"] = ("interval midpoint per JuMACS WACCM-X convention "
                                          "(filename month + time_bnds lower bound); not rewritten here")
    except Exception as exc:
        info["time_decode_ok"] = False
        info["error"] = str(exc)
    return info


def audit_variable(ds, path, native):
    da = ds[native]
    info = {"file": str(path), "native": native, "dims": list(da.dims), "shape": list(da.shape),
            "dtype": str(da.dtype)}
    for key in ("units", "standard_name", "long_name", "_FillValue", "missing_value"):
        if key in da.attrs:
            info[key] = da.attrs[key]
    fill = da.encoding.get("_FillValue")
    if fill is not None and "_FillValue" not in info:
        info["_FillValue"] = fill
    try:
        small = da.isel({dim: 0 for dim in da.dims if dim not in (da.dims[-1],)})
        info["sample_read_ok"] = bool(small.size) and small.load().notnull().any().item()
    except Exception:
        info["sample_read_ok"] = None
    return info


def _assess(config, horizontals, verticals, times, model):
    reasons, features = [], []
    waccmx = is_waccmx(config)
    worst = 0
    def bump(value):
        nonlocal worst
        worst = max(worst, value)
    for h in horizontals:
        grid = h["grid"]
        if grid == "coordinate_names_missing":
            reasons.append("coordinate_names_missing"); bump(1)
        elif grid == "curvilinear":
            reasons.append("curvilinear_grid_support_needed"); bump(2)
        elif grid == "regular_partial":
            reasons.append("partial_longitude_coverage"); bump(2)
        elif grid == "regular_latlon":
            if not (h["mapped_latitude"] and h["mapped_longitude"]):
                reasons.append("horizontal_mapping_missing"); bump(1)
            if h.get("longitude_convention") == "-180..180":
                features.append("longitude_-180_180_handled_by_mod_in_zonal")
        elif grid == "already_zonal":
            features.append("already_zonal_supported_by_zonal_mean")
    for v in verticals:
        ctype = v["coordinate_type"]
        if ctype == "none":
            pass
        elif ctype == "unknown":
            reasons.append("vertical_coordinate_unrecognized"); bump(2)
        elif ctype == "pure_sigma":
            reasons.append("vertical_requires_sigma_to_pressure_conversion"); bump(2)
        elif ctype in ("hybrid_sigma_pressure", "hybrid_height"):
            if v["hybrid_a_present"] is False or v["hybrid_b_present"] is False:
                reasons.append("hybrid_coefficients_missing_in_file"); bump(2)
            elif v["formula_terms_source"] == "observed_non_cf":
                reasons.append("non_cf_formula_term_attribute"); bump(2)
            elif v["formula_terms_source"] != "observed":
                if v["level_from_config"] and v["hybrid_a_present"] and v["hybrid_b_present"]:
                    reasons.append("hybrid_coefficients_inferred_from_config"); bump(1)
                else:
                    reasons.append("hybrid_coefficients_mapping_missing"); bump(1)
            if v["surface_pressure_name"] is None or v["surface_pressure_present"] is False:
                reasons.append("surface_pressure_mapping_missing"); bump(1)
            if ctype == "hybrid_height":
                reasons.append("pressure_formula_not_supported"); bump(2)
        elif ctype == "pressure_level":
            if not (v["level_from_config"] or v["standard_name"] == "air_pressure"):
                reasons.append("vertical_mapping_missing"); bump(1)
        elif ctype == "altitude":
            reasons.append("altitude_coordinate_needs_height_conversion"); bump(2)
    for t in times:
        if not t.get("time_decode_ok"):
            reasons.append("time_decode_failed"); bump(3)
    if waccmx:
        features.append("waccmx_separate_whole_atmosphere_handling")
    classes = ["supported_now", "small_mapping_change", "reader_extension_needed", "unsupported"]
    classification = classes[worst]
    if waccmx and classification == "supported_now":
        reasons.append("waccmx_separate_handling_preserved")
    return classification, sorted(set(reasons)), sorted(set(features))


def audit_model(model, root=ROOT):
    meta = model_metadata(model)
    config = load_config(model)
    row = {"model": model, "kind": meta["kind"], "status": "supported", "families": "", "horizontal_grid": "",
           "vertical_coordinate": "", "time_decode": "", "reader_compatibility": "", "main_action": "",
           "representatives": [], "per_file": [], "reasons": [], "notes": []}
    path = _manifest(model, root)
    if not path.exists():
        row.update(status="not_downloaded", reader_compatibility="unknown", main_action="download the archive",
                   reasons=["manifest_missing"])
        return row
    payload = json.loads(path.read_text())
    present = [r for r in payload.get("files", []) if (root / r["local_path"]).exists()]
    if not present:
        row.update(status="not_downloaded", reader_compatibility="unknown", main_action="download the archive",
                   reasons=["no_local_files"])
        return row
    representatives, why = select_representatives(model, present, root)
    row["reasons"].extend(f"selected:{w}" for w in why)
    import xarray as xr
    horizontals, verticals, times, grids, vtypes = [], [], [], set(), set()
    for record in representatives:
        local = root / record["local_path"]
        try:
            with xr.open_dataset(local, decode_times=True, use_cftime=True) as ds:
                native = record["variable"]
                if native not in ds:
                    mapped = config.get("variables", {}).get(native, native)
                    if mapped in ds:
                        native = mapped
                    else:
                        wanted = [config["variables"][c] for c in ("temperature", "O3") if c in config.get("variables", {})]
                        native = next((w for w in wanted if w in ds), None) \
                            or next((v for v in ds.data_vars if "time" in ds[v].dims), None)
                if native is None or native not in ds:
                    continue
                h = audit_horizontal(ds, config, native)
                v = audit_vertical(ds, config, native)
                t = audit_time(ds, config, local)
                entry = {"file": record["local_path"], "family": record["family"],
                         "variable": native, "canonical": _canonical(config, native),
                         "horizontal": h, "vertical": v, "time": t, "variable_metadata": audit_variable(ds, local, native)}
                row["per_file"].append(entry)
                row["representatives"].append(record["local_path"])
                horizontals.append(h); verticals.append(v); times.append(t)
                grids.add(h["grid"]); vtypes.add(v["coordinate_type"])
        except Exception as exc:
            row["reasons"].append(f"unreadable:{local.name}:{exc}")
    if not row["per_file"]:
        row.update(status="unsupported", reader_compatibility="unsupported", main_action="investigate unreadable files",
                   reasons=["no_readable_representatives"])
        return row
    classification, reasons, features = _assess(config, horizontals, verticals, times, model)
    status_map = {"supported_now": "supported", "small_mapping_change": "needs_mapping",
                  "reader_extension_needed": "reader_extension_needed", "unsupported": "unsupported"}
    actions = {"supported_now": "none - ready for processing tests",
               "small_mapping_change": "add coordinates: mapping to model YAML",
               "reader_extension_needed": "extend reader/coordinates module",
               "unsupported": "coordinate structure not processable by JuMACS"}
    row.update(status=status_map[classification], reader_compatibility=classification,
               main_action=actions[classification],
               families=";".join(sorted({e["family"] for e in row["per_file"]})),
               horizontal_grid=";".join(sorted(grids)), vertical_coordinate=";".join(sorted(vtypes)),
               time_decode="yes" if all(t.get("time_decode_ok") for t in times) else "no",
               reasons=sorted(set(row["reasons"])) + reasons, notes=features)
    return row


def _canonical(config, native):
    for canonical, name in config.get("variables", {}).items():
        if name == native:
            return canonical
    return None


def _md_table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    return lines


def markdown_coordinate_audit_report(rows):
    lines = ["# JuMACS coordinate audit", "",
             "Offline audit of actual coordinate conventions in the downloaded raw archives and of current "
             "JuMACS reader compatibility (`jumacs coordinate-audit`). Descriptive only: no coordinates are "
             "remapped and no model configurations are modified.", "",
             "Statuses: `supported`, `needs_mapping`, `reader_extension_needed`, `unsupported`, `not_downloaded`.", ""]
    lines += _md_table(COORDINATE_AUDIT_COLUMNS, [{**row, "reasons": "; ".join(row["reasons"])} for row in rows])
    lines += ["", "## Detail", ""]
    for row in rows:
        if not row["per_file"]:
            continue
        lines.append(f"### {row['model']}")
        lines.append("")
        for entry in row["per_file"]:
            h, v, t = entry["horizontal"], entry["vertical"], entry["time"]
            lines.append(f"- `{entry['file']}` ({entry['family']}, {entry['variable']}"
                         + (f", canonical {entry['canonical']}" if entry["canonical"] else "") + "): "
                         f"grid={h['grid']} lat={h.get('latitude')}({h.get('n_lat')}) "
                         f"lon={h.get('longitude')}({h.get('n_lon')}) conv={h.get('longitude_convention', 'n/a')}; "
                         f"vertical={v['coordinate_type']} levels={v.get('n_levels')} "
                         f"terms={v.get('formula_terms_source') or 'n/a'} ps={v.get('surface_pressure_name') or 'n/a'}; "
                         f"time={'ok' if t.get('time_decode_ok') else 'failed'} "
                         f"calendar={t.get('calendar', '')} monthly={t.get('monthly')}")
        lines.append("")
    return "\n".join(lines)


def audit_models(models=None, root=ROOT):
    models = tuple(models) if models else tuple(registry())
    return [audit_model(model, root) for model in models]


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def write_coordinate_audit(rows=None, models=None, root=ROOT):
    rows = list(rows) if rows is not None else audit_models(models, root)
    diagnostics = root / "products/diagnostics/coordinate_audit"
    diagnostics.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (diagnostics / f"{row['model']}.json").write_text(json.dumps(row, indent=2, default=_json_default))
    comparison = root / "products/comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    csv_path = comparison / "coordinate_audit.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COORDINATE_AUDIT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "reasons": "; ".join(row["reasons"])} for row in rows)
    md_path = comparison / "coordinate_audit.md"
    md_path.write_text(markdown_coordinate_audit_report(rows))
    return [csv_path, md_path]
