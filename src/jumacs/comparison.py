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
    if g.sizes.get("month", 1) == 0:
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
    configs = {m: load_config(m) for m in models}
    report = {"start_year": start_year, "end_year": end_year, "compared": [], "skipped": {}}
    for canonical in sorted(set(configs[models[0]]["variables"]) & set(configs[models[1]]["variables"])):
        names = {m: configs[m]["variables"][canonical] for m in configs}
        from .climatology import variable_product_name
        paths = {m: ROOT / configs[m]["paths"]["climatology"] / variable_product_name(m, names[m], start_year, end_year) for m in configs}
        if not all(p.exists() for p in paths.values()):
            report["skipped"][canonical] = "climatology missing"; continue
        with xr.open_dataset(paths[models[0]]) as gd, xr.open_dataset(paths[models[1]]) as ed:
            if not compatible_units(gd.attrs.get("source_units"), ed.attrs.get("source_units")):
                report["skipped"][canonical] = "units differ"; continue
            gp, ep = _pressure(gd), _pressure(ed)
            if gp is None or ep is None:
                report["skipped"][canonical] = "native pressure coordinate unavailable"; continue
            g, e = gd["mean"], ed["mean"]
            g.attrs["units"] = gd.attrs.get("source_units", "")
            e.attrs["units"] = ed.attrs.get("source_units", "")
            try:
                out = compare_fields(g, e, gp, ep, pressure_pa, models)
            except ValueError as exc:
                report["skipped"][canonical] = str(exc); continue
            dest = ROOT / "products/comparison" / f"jumacs_{models[0].lower()}_vs_{models[1].lower()}_{canonical.lower()}_{start_year}-{end_year}.nc"
            dest.parent.mkdir(parents=True, exist_ok=True)
            out.attrs = {"project": "JuMACS", "models": ", ".join(models), "operation": f"{models[0]} minus {models[1]}; no model mean", "climatology_period": f"{start_year}-{end_year}"}
            temporary = dest.with_suffix(".nc.tmp")
            out.to_netcdf(temporary, engine="netcdf4")
            temporary.replace(dest)
            report["compared"].append(canonical)
    path = ROOT / "products/comparison" / f"comparison_{models[0].lower()}_vs_{models[1].lower()}_{start_year}-{end_year}.json"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2))
    return path
