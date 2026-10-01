"""One command per model: monthly zonal series, climatology, and a verified combined product."""
import numpy as np
import xarray as xr

from .archive import discovered_zm_variables
from .climatology import build_climatology
from .config import ROOT, is_waccmx, load_config, model_period, models_with_capability, model_names, ready_model_names, vertical_grid
from .zonal import build_zonal


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
    with xr.open_dataset(path, use_cftime=True) as ds:
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
    outputs = build_climatology(model, start, end, available)
    combined_path = outputs[-1]
    report["climatology_products"] = [str(path) for path in outputs[:-1]]
    report["combined_product"] = str(combined_path)
    with xr.open_dataset(combined_path) as combined:
        coordinate = vertical_grid()["coordinate"]
        if coordinate in combined.coords:
            levels = np.asarray(combined[coordinate].values, float)
            report["vertical_grid"] = {"coordinate": coordinate, "level_count": int(levels.size),
                                       "pressure_min_pa": round(float(levels.min()), 4),
                                       "pressure_max_pa": round(float(levels.max()), 4),
                                       "monotonic": bool(np.all(np.diff(levels) <= 0) or np.all(np.diff(levels) >= 0)),
                                       "interpolation": "linear_log_pressure", "extrapolation": "none"}
        for name in available:
            report["checks"][name] = check_variable(name, combined)
    lost = [name for name in available if name not in report["checks"]]
    report["variables_in_combined"] = len(report["checks"])
    report["ok"] = bool(not lost and all(check["ok"] for check in report["checks"].values()))
    if lost:
        report["error"] = f"combined product lacks {', '.join(lost)}: {combined_path}"
    return report


def check_variable(name, combined):
    mean = combined.get(f"{name}_mean")
    if mean is None or f"{name}_n_years" not in combined:
        return {"present": False, "ok": False}
    coordinate = vertical_grid()["coordinate"]
    check = {"present": True, "finite_fraction": round(float(np.isfinite(mean).mean()), 4),
             "n_years_min": int(combined[f"{name}_n_years"].min()),
             "n_years_max": int(combined[f"{name}_n_years"].max()),
             "dimensions": [str(dim) for dim in mean.dims]}
    check["ok"] = True
    if mean.ndim == 3:
        if coordinate not in mean.dims:
            check["ok"] = False
            check["error"] = f"three-dimensional field is not on the common '{coordinate}' grid: {check['dimensions']}"
            return check
        pressure = np.asarray(combined[coordinate].values, float)
        check["pressure_min_pa"] = round(float(pressure.min()), 4)
        check["pressure_max_pa"] = round(float(pressure.max()), 4)
        check["pressure_monotonic"] = bool(np.all(np.diff(pressure) <= 0) or np.all(np.diff(pressure) >= 0))
        check["pressure_plausible"] = bool(pressure.max() <= 120000.0 and pressure.min() > 0.0)
        check["finite_fraction_on_common_grid"] = check["finite_fraction"]
        top = mean.attrs.get("native_pressure_min_pa")
        bottom = mean.attrs.get("native_pressure_max_pa")
        if top and bottom:
            oriented = mean.transpose(*[coordinate] + [dim for dim in mean.dims if dim != coordinate])
            finite_per_level = np.isfinite(oriented.values).sum(axis=tuple(range(1, oriented.ndim)))
            outside = (pressure < top * 0.999) | (pressure > bottom * 1.001)
            leak = int(np.sum(finite_per_level[outside]))
            check["levels_outside_native_coverage"] = int(outside.sum())
            check["values_outside_native_coverage"] = leak
            if leak:
                check["ok"] = False
                check["error"] = f"{leak} finite values outside the native pressure range {top}-{bottom} Pa"
        if not check["pressure_plausible"] or not check["pressure_monotonic"]:
            check["ok"] = False
            check["error"] = "the common pressure coordinate is not monotonic, positive and plausible"
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
