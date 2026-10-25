"""Descriptive inventory of existing native and harmonized climatologies."""

import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import config
from .climatology import product_name
from .vertical import is_latitude_dimension

FIELDS = ("model", "canonical_variable", "native_variable", "long_name", "units",
          "dimensionality", "native_product", "harmonized_product", "native_available",
          "harmonized_available", "bottom_altitude_km", "top_altitude_km", "finite_fraction")


def _dimensionality(field, harmonized):
    """Grid label for a mean field on either the native or the harmonized layout."""
    dimensions = field.dims
    if not dimensions or not is_latitude_dimension(dimensions[-1]):
        return None
    if harmonized:
        if dimensions == ("month", "lat"):
            return "2D"
        if dimensions == ("month", "altitude", "lat"):
            return "3D"
        return None
    if len(dimensions) == 2 and dimensions[0] == "time":
        return "2D"
    if len(dimensions) == 3 and dimensions[0] == "time":
        return "3D"
    return None


def _fields(path, skipped, *, harmonized=False):
    """Read field metadata; calculate altitude coverage only for harmonized fields."""
    result = {}
    if path is None or not path.is_file():
        return result
    with xr.open_dataset(path) as dataset:
        for name in dataset.data_vars:
            if not name.endswith("_mean"):
                continue
            field = dataset[name]
            dimensionality = _dimensionality(field, harmonized)
            if dimensionality is None:
                skipped.append({"product": str(path), "field": name,
                                "reason": f"unsupported dimensions {field.dims}"})
                continue
            stem = name[:-5]
            native = str(field.attrs.get("source_variable") or stem) if harmonized else stem
            item = {"long_name": str(field.attrs.get("long_name", "")),
                    "units": str(field.attrs.get("units", "")), "dimensionality": dimensionality,
                    "canonical_variable": str(field.attrs.get("canonical_variable") or "") or None,
                    "bottom_altitude_km": None, "top_altitude_km": None, "finite_fraction": None}
            if harmonized and dimensionality == "3D":
                finite = np.isfinite(np.asarray(field.values, dtype="float64"))
                levels = np.flatnonzero(finite.any(axis=(0, 2)))
                item["finite_fraction"] = float(finite.mean())
                if levels.size:
                    altitude = np.asarray(dataset["altitude"].values, dtype="float64")
                    item["bottom_altitude_km"] = float(np.min(altitude[levels]))
                    item["top_altitude_km"] = float(np.max(altitude[levels]))
            result[native] = item
    return result


def _harmonized_product(folder, start_year, end_year):
    if not folder.is_dir():
        return None
    matches = sorted(folder.glob(f"*_{start_year}-{end_year}_*deg_*km.nc"))
    return matches[0] if matches else None


def _summary(rows):
    available = [row for row in rows if row["native_available"] or row["harmonized_available"]]
    harmonized = [row for row in rows if row["harmonized_available"]]
    return {
        "harmonized_models": len({row["model"] for row in harmonized}),
        "canonical_variables": len({row["canonical_variable"] for row in available
                                    if row["canonical_variable"]}),
        "variables_3d": sum(row["dimensionality"] == "3D" for row in available),
        "variables_2d": sum(row["dimensionality"] == "2D" for row in available),
        "harmonized_model_variables": len(harmonized),
        "unmapped_available": sum(not row["canonical_variable"] for row in available),
        "per_model": {model: {
            "total": sum(row["model"] == model for row in available),
            "3D": sum(row["model"] == model and row["dimensionality"] == "3D" for row in available),
            "2D": sum(row["model"] == model and row["dimensionality"] == "2D" for row in available),
        } for model in sorted({row["model"] for row in available})},
    }


def _markdown(rows, summary, start_year, end_year):
    lines = [f"# JuMACS climatology inventory, {start_year}–{end_year}", "",
             (f"{summary['harmonized_models']} harmonized models; "
              f"{summary['canonical_variables']} canonical variables; "
              f"{summary['variables_3d']} 3D and {summary['variables_2d']} 2D available model variables; "
              f"{summary['harmonized_model_variables']} harmonized model-variable fields."), ""]
    for model, counts in summary["per_model"].items():
        lines.append(f"- {model}: {counts['total']} variables ({counts['3D']} 3D, {counts['2D']} 2D)")
    lines += ["", ("The table includes configured variables absent from existing products. "
                   "Altitude limits mark levels with any finite harmonized mean value; no usability "
                   "threshold is applied. Products are never merged across models."), "",
              ("| Model | Canonical variable | Native name | Grid | Native | Harmonized | "
               "Altitude range (km) |"),
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        bottom, top = row["bottom_altitude_km"], row["top_altitude_km"]
        altitude = f"{bottom:.4g}–{top:.4g}" if bottom is not None and top is not None else ""
        lines.append("| " + " | ".join((row["model"], row["canonical_variable"] or "—",
                     row["native_variable"], row["dimensionality"] or "—",
                     "yes" if row["native_available"] else "no",
                     "yes" if row["harmonized_available"] else "no", altitude)) + " |")
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
    harmonized_root = root / "products" / "harmonized"
    registered = config.registry()
    folders = {p.name for base in (native_root, harmonized_root) if base.exists()
               for p in base.iterdir() if p.is_dir() and p.name in registered}
    models = [model] if model else sorted(folders)
    rows = []
    skipped = []
    for name in models:
        mapping = config.load_config(name)["variables"]
        inverse = {native: canonical for canonical, native in mapping.items()}
        native_path = native_root / name / product_name(name, start_year, end_year)
        harmonized_path = _harmonized_product(harmonized_root / name, start_year, end_year)
        native_fields = _fields(native_path, skipped)
        harmonized_fields = _fields(harmonized_path, skipped, harmonized=True)
        # Include configured absences and any present fields that lack a mapping.
        names = sorted(set(mapping.values()) | native_fields.keys() | harmonized_fields.keys())
        for native in names:
            native_info = native_fields.get(native)
            harmonized_info = harmonized_fields.get(native)
            info = harmonized_info or native_info or {}
            rows.append({"model": name,
                         "canonical_variable": inverse.get(native) or info.get("canonical_variable"),
                         "native_variable": native, "long_name": info.get("long_name", ""),
                         "units": info.get("units", ""), "dimensionality": info.get("dimensionality"),
                         "native_product": native_path.relative_to(root).as_posix() if native_path.is_file() else None,
                         "harmonized_product": (harmonized_path.relative_to(root).as_posix()
                                                if harmonized_path and harmonized_path.is_file() else None),
                         "native_available": native in native_fields,
                         "harmonized_available": native in harmonized_fields,
                         "bottom_altitude_km": harmonized_info and harmonized_info["bottom_altitude_km"],
                         "top_altitude_km": harmonized_info and harmonized_info["top_altitude_km"],
                         "finite_fraction": harmonized_info and harmonized_info["finite_fraction"]})
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
