"""One command per model: monthly zonal series and one verified CF-1.13 climatology product."""
import numpy as np
import xarray as xr

from . import cf
from .archive import discovered_zm_variables
from .climatology import build_climatology
from .netcdf import open_cftime_dataset
from .config import ROOT, is_waccmx, load_config, model_period, models_with_capability, model_names, ready_model_names, vertical_grid
from .zonal import build_zonal
from .vertical import pressure_report, product_pressure, vertical_dimension


def buildable_model_names():
    ready = set(ready_model_names())
    return tuple(model for model in models_with_capability("zonal_processing") if model in ready)


def model_list(selection):
    """Resolve a --model value: one name, a comma-separated list, or all buildable models."""
    if selection == "all":
        return buildable_model_names()
    names = tuple(part.strip() for part in selection.split(",") if part.strip())
    if not names:
        raise ValueError("--model needs a model name, a comma-separated list, or all")
    unknown = [name for name in names if name not in model_names()]
    if unknown:
        raise ValueError(f"unknown model(s) {', '.join(unknown)}; see config/models/ or run jumacs inspect --model all")
    return names


def configured_variables(model, config):
    names = set(config["variables"].values())
    if is_waccmx(config):
        names |= set(discovered_zm_variables(model))
    return sorted(names)


def coverage_years(path):
    with open_cftime_dataset(path) as ds:
        return int(ds.time.values[0].year), int(ds.time.values[-1].year)


def build_model(model, start_year=None, end_year=None, variables=None):
    config = load_config(model)
    period = model_period(model)
    start = start_year or period["start_year"]
    end = end_year or period["end_year"]
    if variables:
        wanted = sorted({config["variables"].get(name, name) for name in variables})
    else:
        wanted = configured_variables(model, config)
    zonal_base = ROOT / config["paths"]["zonal"]
    report = {"model": model, "experiment": config["model"]["experiment"], "climatology_period": f"{start}-{end}",
              "variables_configured": len(wanted), "zonal_built": [], "zonal_reused": [], "skipped": [],
              "checks": {}, "warnings": []}
    available = []
    for name in wanted:
        zonal_path = zonal_base / f"{name}_monthly_zonal.nc"
        if zonal_path.exists():
            report["zonal_reused"].append(name)
        else:
            try:
                build_zonal(model, name)
            except FileNotFoundError as exc:
                report["skipped"].append({"variable": name, "reason": str(exc)})
                continue
            report["zonal_built"].append(name)
        if not zonal_path.exists():
            report["skipped"].append({"variable": name, "reason": "zonal product was not written"})
            continue
        first, last = coverage_years(zonal_path)
        if last < start or first > end:
            report["skipped"].append({"variable": name, "reason": f"zonal data spans {first}-{last}, outside {start}-{end}"})
            continue
        available.append(name)
    report["variables_processed"] = available
    if not available:
        report["ok"] = False
        report["error"] = f"no variable of {model} has monthly zonal data in {start}-{end}; run jumacs download --model {model} --start-year {start} --end-year {end}"
        return report
    product_path = build_climatology(model, start, end, available)
    report["product"] = str(product_path)
    report["conventions"] = cf.CONVENTIONS
    with xr.open_dataset(product_path) as product:
        grid = vertical_grid()
        report["application_pressure_grid"] = {"coordinate": grid["coordinate"],
                                              "level_count": len(grid["levels"]),
                                              "pressure_min_pa": pressure_report(min(grid["levels"])),
                                              "pressure_max_pa": pressure_report(max(grid["levels"])),
                                              "interpolation": grid["interpolation"],
                                              "extrapolation": grid["extrapolation"],
                                              "applied_at": "combination; products stay on native grids"}
        report["vertical"] = {key: str(product.attrs[key]) for key in
                              ("vertical_coordinate", "vertical_level_count", "vertical_level_counts",
                               "vertical_interpolation") if key in product.attrs}
        for name in available:
            report["checks"][name] = check_variable(name, product)
    lost = [name for name in available if not report["checks"].get(name, {}).get("present")]
    report["variables_in_product"] = sum(1 for check in report["checks"].values() if check.get("present"))
    report["ok"] = bool(not lost and all(check["ok"] for check in report["checks"].values()))
    if lost:
        report["error"] = f"product lacks {', '.join(lost)}: {product_path}"
    return report


def vertical_pressure_monotonic(pressure, axis):
    """True when every column of the pressure description runs one way only, ignoring missing levels."""
    diffs = np.diff(np.asarray(pressure, float), axis=axis)
    steps = diffs[np.isfinite(diffs)]
    if steps.size == 0:
        return False
    return bool(np.all(steps >= 0.0) or np.all(steps <= 0.0))


def check_variable(name, combined):
    """What one field of a product looks like: it is reported on the grid the model published it on."""
    mean = combined.get(f"{name}_mean")
    if mean is None or f"{name}_n_years" not in combined:
        return {"present": False, "ok": False}
    check = {"present": True, "finite_fraction": round(float(np.isfinite(mean).mean()), 4),
             "n_years_min": int(combined[f"{name}_n_years"].min()),
             "n_years_max": int(combined[f"{name}_n_years"].max()),
             "dimensions": [str(dim) for dim in mean.dims]}
    check["ok"] = True
    if mean.ndim == 3:
        pressure = product_pressure(combined, mean)
        level = vertical_dimension(mean, pressure) if pressure is not None else vertical_dimension(mean)
        check["vertical_coordinate"] = str(level)
        check["vertical_level_count"] = int(mean.sizes.get(level, 0)) if level else 0
        if pressure is None:
            check["ok"] = False
            check["error"] = ("three-dimensional field carries no air pressure: the product needs a pressure-valued "
                              "coordinate or a published pressure field; rebuild it with jumacs climatology")
            return check
        values = np.asarray(pressure, float)
        check["pressure_representation"] = str(mean.attrs.get("native_pressure_kind",
                                                              "coordinate" if pressure.ndim == 1 else "profile"))
        check["pressure_finite_fraction"] = round(float(np.mean(np.isfinite(values) & (values > 0.0))), 4)
        check["pressure_min_pa"] = pressure_report(np.nanmin(values))
        check["pressure_max_pa"] = pressure_report(np.nanmax(values))
        check["pressure_monotonic"] = vertical_pressure_monotonic(values, list(pressure.dims).index(level))
        check["pressure_plausible"] = bool(check["pressure_max_pa"] <= 120000.0 and check["pressure_min_pa"] > 0.0)
        if not check["pressure_monotonic"] or not check["pressure_plausible"]:
            check["ok"] = False
            check["error"] = "the air pressure of the field is not monotonic, positive and plausible"
    else:
        check["vertical_treatment"] = "two-dimensional field kept without a vertical dimension"
    if check["finite_fraction"] == 0.0:
        check["warning"] = "no finite values in the monthly mean climatology"
    return check


def build_models(selection, start_year=None, end_year=None, variables=None):
    for model in model_list(selection):
        if load_config(model)["model"].get("status", "ready") != "ready":
            yield {"model": model, "ok": False, "error": f"archive identifiers unresolved in config/models/{model}.yaml"}
            continue
        if model not in buildable_model_names():
            yield {"model": model, "ok": False,
                   "error": f"{model} does not combine status: ready with capabilities.zonal_processing in config/models/{model}.yaml"}
            continue
        yield build_model(model, start_year, end_year, variables)
