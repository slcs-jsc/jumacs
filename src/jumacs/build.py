"""One command per model: monthly zonal series, climatology, and a verified combined product."""
import numpy as np
import xarray as xr

from .archive import discovered_zm_variables
from .climatology import build_climatology
from .config import ROOT, is_waccmx, load_config, model_period, models_with_capability, model_names, ready_model_names
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
        for name in available:
            report["checks"][name] = check_variable(name, combined)
    lost = [name for name in available if name not in report["checks"]]
    report["variables_in_combined"] = len(report["checks"])
    report["ok"] = bool(not lost and all(check["present"] for check in report["checks"].values()))
    if lost:
        report["error"] = f"combined product lacks {', '.join(lost)}: {combined_path}"
    return report


def check_variable(name, combined):
    mean = combined.get(f"{name}_mean")
    if mean is None or f"{name}_n_years" not in combined:
        return {"present": False}
    check = {"present": True, "finite_fraction": round(float(np.isfinite(mean).mean()), 4),
             "n_years_min": int(combined[f"{name}_n_years"].min()), "dimensions": [str(dim) for dim in mean.dims]}
    level = next((dim for dim in mean.dims if dim not in ("month", "lat")), None)
    pressure = combined.get(f"{name}_air_pressure")
    if pressure is not None and level is not None:
        profile = np.asarray(pressure.isel({dim: 0 for dim in pressure.dims if dim != level}), float)
        profile = np.squeeze(profile)
        profile = profile[np.isfinite(profile)]
        if profile.size:
            check["pressure_min_pa"] = round(float(profile.min()), 3)
            check["pressure_max_pa"] = round(float(profile.max()), 3)
            check["pressure_monotonic"] = bool(np.all(np.diff(profile) <= 0) or np.all(np.diff(profile) >= 0))
            check["pressure_plausible"] = bool(profile.max() <= 120000.0 and profile.min() > 0.0)
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
