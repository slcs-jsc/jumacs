"""Descriptive inventory of existing native and application climatologies."""

import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import config
from .climatology import product_name

FIELDS = ("model", "canonical_variable", "native_variable", "long_name", "units",
          "dimensionality", "native_product", "application_product", "native_available",
          "application_available", "waccmx_extended", "bottom_pressure_hpa",
          "top_pressure_hpa", "finite_fraction")


def _fields(path, skipped):
    """Read field metadata only; calculate coverage only for application fields."""
    result = {}
    if not path.is_file():
        return result
    with xr.open_dataset(path, decode_cf=False) as dataset:
        for name in dataset.data_vars:
            if not name.endswith("_mean"):
                continue
            field = dataset[name]
            native = name[:-5]
            dimensions = field.dims
            if dimensions == ("time", "lat"):
                dimensionality = "2D"
            elif len(dimensions) == 3 and dimensions[0] == "time" and dimensions[-1] == "lat":
                dimensionality = "3D"
            else:
                skipped.append({"product": str(path), "field": name,
                                "reason": f"unsupported dimensions {dimensions}"})
                continue
            item = {"long_name": str(field.attrs.get("long_name", "")),
                    "units": str(field.attrs.get("units", "")),
                    "dimensionality": dimensionality, "waccmx_extended": None,
                    "bottom_pressure_hpa": None, "top_pressure_hpa": None,
                    "finite_fraction": None}
            if "pressure" in dimensions and dimensions == ("time", "pressure", "lat"):
                values = np.asarray(field.values)
                finite = np.isfinite(values)
                levels = np.flatnonzero(finite.any(axis=(0, 2)))
                item["finite_fraction"] = float(finite.mean())
                if levels.size:
                    pressure = np.asarray(dataset.pressure.values, dtype=float) / 100.0
                    item["bottom_pressure_hpa"] = float(np.max(pressure[levels]))
                    item["top_pressure_hpa"] = float(np.min(pressure[levels]))
            applied = field.attrs.get("extension_applied")
            donor = field.attrs.get("extension_donor", dataset.attrs.get("extension_donor"))
            if applied == "false":
                item["waccmx_extended"] = False
            elif applied == "true" and donor == "WACCM-X":
                item["waccmx_extended"] = True
            result[native] = item
    return result


def _summary(rows):
    available = [row for row in rows if row["native_available"] or row["application_available"]]
    application = [row for row in rows if row["application_available"]]
    models = sorted({row["model"] for row in application})
    return {
        "application_models": len(models),
        "canonical_variables": len({row["canonical_variable"] for row in available
                                    if row["canonical_variable"]}),
        "variables_3d": sum(row["dimensionality"] == "3D" for row in available),
        "variables_2d": sum(row["dimensionality"] == "2D" for row in available),
        "application_model_variables": len(application),
        "waccmx_extended": sum(row["waccmx_extended"] is True for row in application),
        "unmapped_available": sum(not row["canonical_variable"] for row in available),
        "per_model": {model: {
            "total": sum(row["model"] == model for row in available),
            "3D": sum(row["model"] == model and row["dimensionality"] == "3D" for row in available),
            "2D": sum(row["model"] == model and row["dimensionality"] == "2D" for row in available),
        } for model in sorted({row["model"] for row in available})},
    }


def _markdown(rows, summary, start_year, end_year):
    lines = [f"# JuMACS climatology inventory, {start_year}–{end_year}", "",
             (f"{summary['application_models']} application models; "
             f"{summary['canonical_variables']} canonical variables; "
             f"{summary['variables_3d']} 3D and {summary['variables_2d']} 2D available model variables; "
             f"{summary['application_model_variables']} application model-variable fields; "
             f"{summary['waccmx_extended']} extended with WACCM-X."), ""]
    for model, counts in summary["per_model"].items():
        lines.append(f"- {model}: {counts['total']} variables ({counts['3D']} 3D, {counts['2D']} 2D)")
    lines += ["", ("The table includes configured variables absent from existing products. "
                   "Pressure limits mark levels with any finite mean value; no usability threshold is applied."), "",
              ("| Model | Canonical variable | Native name | Grid | Native | Application | "
               "Pressure range (hPa) | WACCM-X extended |"),
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        bottom, top = row["bottom_pressure_hpa"], row["top_pressure_hpa"]
        pressure = f"{bottom:.4g}–{top:.4g}" if bottom is not None and top is not None else ""
        flag = row["waccmx_extended"]
        lines.append("| " + " | ".join((row["model"], row["canonical_variable"] or "—",
                     row["native_variable"], row["dimensionality"] or "—",
                     "yes" if row["native_available"] else "no",
                     "yes" if row["application_available"] else "no", pressure,
                     "unknown" if flag is None else "yes" if flag else "no")) + " |")
    return "\n".join(lines) + "\n"


def build_inventory(start_year=None, end_year=None, *, model=None, root=None):
    """Inspect existing period products and write CSV, JSON and Markdown reports."""
    root = config.ROOT if root is None else Path(root)
    period = config.reference_period()["reference_period"]
    start_year = period["start_year"] if start_year is None else start_year
    end_year = period["end_year"] if end_year is None else end_year
    if start_year > end_year:
        raise ValueError("start year exceeds end year")
    native_root = root / "products" / "climatology"
    app_root = root / "products" / "application"
    registered = config.registry()
    folders = {p.name for base in (native_root, app_root) if base.exists()
               for p in base.iterdir() if p.is_dir() and p.name in registered}
    models = [model] if model else sorted(folders)
    rows = []
    skipped = []
    for name in models:
        mapping = config.load_config(name)["variables"]
        native_path = native_root / name / product_name(name, start_year, end_year)
        app_path = app_root / name / (
            f"jumacs_{config.model_slug(name)}_application_climatology_{start_year}-{end_year}.nc")
        native_fields, app_fields = _fields(native_path, skipped), _fields(app_path, skipped)
        inverse = {native: canonical for canonical, native in mapping.items()}
        # Include configured absences and any present fields that lack a mapping.
        names = sorted(set(mapping.values()) | native_fields.keys() | app_fields.keys())
        for native in names:
            info = app_fields.get(native) or native_fields.get(native) or {}
            rows.append({"model": name, "canonical_variable": inverse.get(native),
                         "native_variable": native, "long_name": info.get("long_name", ""),
                         "units": info.get("units", ""),
                         "dimensionality": info.get("dimensionality"),
                         "native_product": native_path.relative_to(root).as_posix() if native_path.is_file() else None,
                         "application_product": app_path.relative_to(root).as_posix() if app_path.is_file() else None,
                         "native_available": native in native_fields,
                         "application_available": native in app_fields,
                         "waccmx_extended": info.get("waccmx_extended") if native in app_fields else None,
                         "bottom_pressure_hpa": info.get("bottom_pressure_hpa") if native in app_fields else None,
                         "top_pressure_hpa": info.get("top_pressure_hpa") if native in app_fields else None,
                         "finite_fraction": info.get("finite_fraction") if native in app_fields else None})
    summary = _summary(rows)
    document = {"period": {"start_year": start_year, "end_year": end_year},
                "summary": summary, "rows": rows, "skipped": skipped}
    directory = root / "products" / "catalog"
    directory.mkdir(parents=True, exist_ok=True)
    basename = directory / "climatology_inventory"
    with basename.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    basename.with_suffix(".json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    basename.with_suffix(".md").write_text(_markdown(rows, summary, start_year, end_year), encoding="utf-8")
    return document
