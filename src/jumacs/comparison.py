"""Diagnostics on a common valid domain; no model mean is constructed."""
import json
import numpy as np
import xarray as xr

from .config import ROOT, load_config
from .vertical import interpolate_log_pressure


def compatible_units(first, second):
    aliases = {"mol/mol": "mol mol-1", "mol mol^-1": "mol mol-1"}
    return bool(first and second and aliases.get(first, first) == aliases.get(second, second))


def compare_fields(geos, emac, geos_pressure, emac_pressure, target_pa, labels=("GEOSCCM", "EMAC")):
    if not compatible_units(geos.attrs.get("units"), emac.attrs.get("units")):
        raise ValueError("Incompatible units")
    g = interpolate_log_pressure(geos, geos_pressure, target_pa)
    e = interpolate_log_pressure(emac, emac_pressure, target_pa)
    # EMAC latitudes inside GEOSCCM's valid range form the common grid.
    common_lat = e.lat.values[(e.lat.values >= g.lat.min().item()) & (e.lat.values <= g.lat.max().item())]
    if len(common_lat) == 0:
        raise ValueError("No overlapping latitude domain")
    if np.array_equal(g.lat.values, common_lat):
        g = g.sel(lat=common_lat)
    elif g.sizes["lat"] == 1:
        g = g.sel(lat=common_lat) if np.array_equal(g.lat.values, common_lat) else g.interp(lat=common_lat)
    else:
        g = g.interp(lat=common_lat)
    e = e.sel(lat=common_lat)
    g, e = xr.align(g, e, join="inner")
    if not any(g.sizes.get(dim, 0) for dim in ("time", "month")):
        raise ValueError("No overlapping calendar months")
    both = np.isfinite(g) & np.isfinite(e)
    difference = (g - e).where(both)
    g.attrs["units"] = geos.attrs["units"]
    e.attrs["units"] = emac.attrs["units"]
    difference.attrs["units"] = geos.attrs["units"]
    relative = xr.where(both & (e != 0), 100 * difference / e, np.nan)
    relative.attrs["units"] = "%"
    relative.attrs["definition"] = f"100 * ({labels[0]} - {labels[1]}) / {labels[1]}; undefined where denominator is zero"
    return xr.Dataset({labels[0]: g.where(both), labels[1]: e.where(both),
                       "difference": difference, "relative_difference": relative})


def _pressure(ds):
    if "air_pressure" in ds:
        return ds["air_pressure"]
    for name in ("lev", "plev", "pressure"):
        if name in ds.coords and ds[name].attrs.get("units", "").lower() in ("pa", "pascal", "pascals"):
            return ds[name]
    return None


def compare_period(start_year, end_year, pressure_pa, models=("GEOSCCM", "EMAC")):
    """Compare climatological monthly means from two single-file products."""
    from .climatology import product_name
    configs = {m: load_config(m) for m in models}
    paths = {m: ROOT / configs[m]["paths"]["climatology"] / product_name(m, start_year, end_year) for m in configs}
    absent = [m for m in models if not paths[m].exists()]
    if absent:
        raise FileNotFoundError(f"climatology product missing for {', '.join(absent)}; "
                                f"run jumacs build --model {absent[0]} --start-year {start_year} --end-year {end_year}")
    report = {"start_year": start_year, "end_year": end_year, "products": {m: str(paths[m]) for m in models},
              "compared": [], "skipped": {}}
    with xr.open_dataset(paths[models[0]]) as first, xr.open_dataset(paths[models[1]]) as second:
        open_datasets = {models[0]: first, models[1]: second}
        for canonical in sorted(set(configs[models[0]]["variables"]) & set(configs[models[1]]["variables"])):
            names = {m: configs[m]["variables"][canonical] for m in configs}
            fields = {m: open_datasets[m].get(f"{names[m]}_mean") for m in models}
            if any(field is None for field in fields.values()):
                report["skipped"][canonical] = "variable not in both products"; continue
            units = {m: fields[m].attrs.get("units", "") for m in models}
            if not compatible_units(units[models[0]], units[models[1]]):
                report["skipped"][canonical] = "units differ"; continue
            pressures = {m: _pressure(open_datasets[m]) for m in models}
            if any(pressure is None for pressure in pressures.values()):
                report["skipped"][canonical] = "common pressure coordinate unavailable"; continue
            left, right = (fields[m].load().assign_attrs(units=units[m]) for m in models)
            try:
                out = compare_fields(left, right, pressures[models[0]], pressures[models[1]], pressure_pa, models)
            except ValueError as exc:
                report["skipped"][canonical] = str(exc); continue
            dest = ROOT / "products/comparison" / f"jumacs_{models[0].lower()}_vs_{models[1].lower()}_{canonical.lower()}_{start_year}-{end_year}.nc"
            dest.parent.mkdir(parents=True, exist_ok=True)
            out.attrs = {"project": "JuMACS", "models": ", ".join(models), "operation": f"{models[0]} minus {models[1]}; no model mean",
                         "climatology_period": f"{start_year}-{end_year}",
                         "comment": "climatological monthly means compared on the shared JuMACS pressure grid"}
            temporary = dest.with_suffix(".nc.tmp")
            out.to_netcdf(temporary, engine="netcdf4")
            temporary.replace(dest)
            report["compared"].append(canonical)
    path = ROOT / "products/comparison" / f"comparison_{models[0].lower()}_vs_{models[1].lower()}_{start_year}-{end_year}.json"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2))
    return path
