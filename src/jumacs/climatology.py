"""Period-specific monthly statistics on native model grids."""
import numpy as np
import xarray as xr

from .config import ROOT, load_config, reference_period, is_waccmx, model_slug


def _slug(model, config):
    slug = config["model"].get("slug")
    if slug:
        return slug
    return "waccmx" if is_waccmx(config) else model_slug(model) + "_refd1"


def product_name(model, start_year, end_year):
    config = load_config(model)
    return f"jumacs_{_slug(model, config)}_climatology_{start_year}-{end_year}.nc"


def variable_product_name(model, name, start_year, end_year):
    config = load_config(model)
    return f"jumacs_{_slug(model, config)}_{name}_climatology_{start_year}-{end_year}.nc"


def monthly_climatology(data, start_year, end_year):
    if start_year > end_year:
        raise ValueError("start year exceeds end year")
    selected = data.where((data.time.dt.year >= start_year) & (data.time.dt.year <= end_year), drop=True)
    if selected.sizes.get("time", 0) == 0:
        raise ValueError("No months in requested period")
    months = [(int(t.year), int(t.month)) for t in selected.time.values]
    if len(months) != len(set(months)):
        raise ValueError("Duplicate monthly samples")
    group = selected.groupby("time.month")
    out = xr.Dataset({"mean": group.mean("time", skipna=True),
                      "sigma": group.std("time", skipna=True, ddof=0),
                      "minimum": group.min("time", skipna=True),
                      "maximum": group.max("time", skipna=True),
                      "n_years": group.count("time")})
    out.attrs = {"start_year": start_year, "end_year": end_year,
                 "definition": "monthly across-year statistics, no trend or bias correction",
                 "source_units": data.attrs.get("units", "")}
    return out


def build_climatology(model, start_year, end_year, variables=None):
    config = load_config(model)
    base = ROOT / config["paths"]["zonal"]
    names = [config["variables"].get(v, v) for v in variables] if variables else [p.name.removesuffix("_monthly_zonal.nc") for p in base.glob("*_monthly_zonal.nc")]
    if not names:
        raise FileNotFoundError(f"No zonal files in {base}")
    outputs = []
    coverage_bounds = []
    from netCDF4 import Dataset
    combined = ROOT / config["paths"]["climatology"] / product_name(model, start_year, end_year)
    combined.parent.mkdir(parents=True, exist_ok=True)
    temporary_combined = combined.with_suffix(".nc.tmp")
    if temporary_combined.exists():
        temporary_combined.unlink()
    reference = reference_period()["reference_period"]
    with Dataset(temporary_combined, "w") as root:
        root.setncatts({"project": "JuMACS", "model": model, "experiment": config["model"]["experiment"],
            "source_dataset": config["model"]["dataset_uuid"], "source_archive": config["model"]["archive_base"],
            "climatology_period": f"{start_year}-{end_year}", "reference_period_years": end_year-start_year+1,
            "nominal_reference_year": reference["nominal_reference_year"], "bias_correction": "none",
            "trend_correction": "none", "model_combination": "none",
            "history": "calendar-month statistics from native-grid monthly zonal time series"})
    for name in sorted(set(names)):
        path = base / f"{name}_monthly_zonal.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path, use_cftime=True) as ds:
            coverage_bounds.extend((str(ds.time.values[0]), str(ds.time.values[-1])))
            stats = monthly_climatology(ds[name], start_year, end_year)
            if "air_pressure" in ds:
                p = ds["air_pressure"].where((ds.time.dt.year >= start_year) & (ds.time.dt.year <= end_year), drop=True)
                stats["air_pressure"] = p.groupby("time.month").mean("time", skipna=True)
            stats.attrs.update({"project": "JuMACS", "model": model, "experiment": config["model"]["experiment"],
                "source_dataset": config["model"]["dataset_uuid"], "source_archive": config["model"]["archive_base"],
                "source_variable": name, "source_units": ds[name].attrs.get("units", ""),
                "output_units": ds[name].attrs.get("units", ""), "vertical_coordinate": config["coordinates"]["level"],
                "time_coverage": f"{str(ds.time.values[0])} to {str(ds.time.values[-1])}",
                "climatology_period": f"{start_year}-{end_year}", "reference_period_years": end_year-start_year+1,
                "nominal_reference_year": reference["nominal_reference_year"], "bias_correction": "none",
                "trend_correction": "none", "history": "monthly zonal values grouped by calendar month"})
            dest = ROOT / config["paths"]["climatology"] / variable_product_name(model, name, start_year, end_year)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(".nc.tmp")
            stats.to_netcdf(temporary, engine="netcdf4")
            temporary.replace(dest)
            stats.to_netcdf(temporary_combined, engine="netcdf4", mode="a", group=f"variables/{name}")
            outputs.append(dest)
    with Dataset(temporary_combined, "a") as root:
        root.time_coverage = f"{min(coverage_bounds)} to {max(coverage_bounds)}"
        root.native_units = "see per-variable groups"
        root.output_units = "see per-variable groups"
        root.vertical_coordinate = "native per-variable; see groups"
    temporary_combined.replace(combined)
    outputs.append(combined)
    return outputs
