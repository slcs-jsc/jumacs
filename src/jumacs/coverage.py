"""Observed valid vertical extent of each native-grid monthly zonal field."""
import csv
import json
from contextlib import nullcontext
import numpy as np
import xarray as xr
from .config import ROOT, load_config
from .comparison import _pressure


def coverage_for_field(field, pressure, height=None):
    level = next((d for d in ("lev", "plev", "ilev") if d in field.dims), None)
    if level is None or pressure is None:
        return {"lowest_valid_pressure_pa": None, "highest_valid_pressure_pa": None,
                "approximate_altitude_min_km": None, "approximate_altitude_max_km": None,
                "median_upper_valid_altitude_km": None, "altitude_method": "unavailable",
                "vertical_levels": field.sizes.get(level, 0) if level else 0}
    valid = np.isfinite(field) & np.isfinite(pressure) & (pressure > 0)
    sampled = pressure.where(valid)
    lower = float(sampled.max(skipna=True))
    upper = float(sampled.min(skipna=True))
    if not np.isfinite(lower) or not np.isfinite(upper):
        lower = upper = None
    method = "7 km scale-height estimate; limited above ~80 km"
    altitude = lambda p: round(7.0 * np.log(101325.0 / p), 2) if p else None
    bottom_km, top_km = altitude(lower), altitude(upper)
    median_top_km = None
    if height is not None and set(height.dims) == set(field.dims) and all(height.sizes[d] == field.sizes[d] and height[d].equals(field[d]) for d in field.dims if d in field.coords and d in height.coords):
        valid_height = height.where(valid & np.isfinite(height))
        bottom_km = round(float(valid_height.min(skipna=True)) / 1000, 2)
        top_km = round(float(valid_height.max(skipna=True)) / 1000, 2)
        median_top_km = round(float(valid_height.max(level, skipna=True).median(skipna=True)) / 1000, 2)
        method = "native geopotential height"
    return {"lowest_valid_pressure_pa": lower, "highest_valid_pressure_pa": upper,
            "approximate_altitude_min_km": bottom_km, "approximate_altitude_max_km": top_km,
            "median_upper_valid_altitude_km": median_top_km,
            "altitude_method": method, "vertical_levels": field.sizes[level]}


def coverage_model(model):
    config = load_config(model)
    rows = []
    height_name = config["variables"].get("geopotential_height")
    height_path = ROOT / config["paths"]["zonal"] / f"{height_name}_monthly_zonal.nc" if height_name else None
    height_source = xr.open_dataset(height_path, use_cftime=True) if height_path and height_path.exists() else nullcontext(None)
    with height_source as height_ds:
        height = height_ds[height_name] if height_ds is not None else None
        for path in sorted((ROOT / config["paths"]["zonal"]).glob("*_monthly_zonal.nc")):
            name = path.name.removesuffix("_monthly_zonal.nc")
            with xr.open_dataset(path, use_cftime=True) as ds:
                row = {"model": model, "variable": name,
                       "canonical_species": next((k for k, v in config["variables"].items() if v == name), name),
                       "units": ds[name].attrs.get("units", ""),
                       "first_month": str(ds.time.values[0])[:7], "last_month": str(ds.time.values[-1])[:7]}
                row.update(coverage_for_field(ds[name], _pressure(ds), height))
                rows.append(row)
    dest = ROOT / "products/diagnostics" / "coverage" / model
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "vertical_coverage.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (dest / "vertical_coverage.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return dest / "vertical_coverage.csv"


def coverage_matrix():
    from .archive import TARGETS
    models = ("GEOSCCM", "EMAC", "WACCM-X")
    reports = {}
    for model in models:
        path = ROOT / "products/diagnostics" / "coverage" / model / "vertical_coverage.json"
        reports[model] = {row["canonical_species"]: row for row in json.loads(path.read_text())} if path.exists() else {}
    names = list(dict.fromkeys(TARGETS + [name for report in reports.values() for name in report]))
    rows = []
    for name in names:
        row = {"species": name}
        for model in models:
            entry = reports[model].get(name, {})
            row[model + "_top_pa"] = entry.get("highest_valid_pressure_pa", "")
            row[model + "_bottom_pa"] = entry.get("lowest_valid_pressure_pa", "")
            row[model + "_top_km"] = entry.get("approximate_altitude_max_km", "")
            row[model + "_top_median_km"] = entry.get("median_upper_valid_altitude_km", "")
            row[model + "_altitude_method"] = entry.get("altitude_method", "")
            row[model + "_levels"] = entry.get("vertical_levels", "")
        rows.append(row)
    dest = ROOT / "products/comparison" / "vertical_coverage_matrix.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return dest


def coverage_plot():
    import matplotlib.pyplot as plt
    targets = ("temperature", "O3", "H2O", "CO2", "CH4", "N2O", "CO", "NO", "CFC-11", "CFC-12", "SF6")
    models = ("GEOSCCM", "EMAC", "WACCM-X")
    reports = {}
    for model in models:
        path = ROOT / "products/diagnostics" / "coverage" / model / "vertical_coverage.json"
        reports[model] = {r["canonical_species"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for model, offset, color in zip(models, (-.24, 0, .24), ("#097b87", "#cf7245", "#5b55aa")):
        x, y = [], []
        for index, target in enumerate(targets):
            value = reports[model].get(target, {}).get("highest_valid_pressure_pa")
            if value is not None and value > 0:
                x.append(index + offset); y.append(value)
        ax.scatter(x, y, s=44, color=color, label=model, zorder=3)
    ax.set(xticks=range(len(targets)), xticklabels=targets, yscale="log", ylabel="Lowest pressure with finite values (Pa)",
           title="JuMACS species-specific upper valid pressure")
    ax.invert_yaxis()
    ax.grid(axis="y", which="both", alpha=.2)
    ax.legend()
    ax.text(.01, -.17, "Finite-value extent only; trace concentrations can be negligible at the top level.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    dest = ROOT / "products/diagnostics" / "coverage" / "vertical_coverage.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    return dest
