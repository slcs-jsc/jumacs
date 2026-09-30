"""Native monthly time series; average longitude only for Amon sources."""
import numpy as np
import xarray as xr

from .config import ROOT, load_config, is_waccmx
from .reader import open_source, source_files
from .coordinates import hybrid_pressure


def zonal_mean(ds, variable, longitude="lon", source_kind=None):
    data = ds[variable]
    if longitude in data.dims:
        data = data.where(np.isfinite(data)).mean(longitude, skipna=True, keep_attrs=True)
        method = "unweighted valid-value longitude mean"
    else:
        method = ("CEDA WACCM-X monthly _zm field (already zonal)" if source_kind == "monthly_zonal_multivariable"
                  else "CEDA AmonZ monthly zonal field (already zonal)")
    data.attrs = {**data.attrs, "zonal_method": method}
    return data


def build_zonal(model, variable):
    config = load_config(model)
    name = config["variables"].get(variable, variable)
    arrays = []
    pressures = []
    source_attrs = []
    signature = None
    longitude_checks = []
    for path in source_files(model, name):
        with open_source(path, model, name) as ds:
            lon_name = config["coordinates"]["longitude"]
            if lon_name in ds[name].dims:
                longitudes = np.sort(np.unique(np.mod(ds[lon_name].values, 360)))
                gaps = np.diff(np.r_[longitudes, longitudes[0] + 360])
                longitude_checks.append(bool(len(longitudes) > 1 and np.max(gaps) <= 1.5 * np.median(gaps)))
            level_name = config["coordinates"]["level"]
            is_hybrid = (level_name in ds and (ds[level_name].attrs.get("standard_name")
                         == "atmosphere_hybrid_sigma_pressure_coordinate" or is_waccmx(config)))
            # Limit the largest in-memory full field to 12 monthly samples.
            for start in range(0, ds.sizes["time"], 12):
                chunk = ds.isel(time=slice(start, start + 12))
                arr = zonal_mean(chunk, name, lon_name, config["model"].get("source_kind")).load()
                current = (tuple((dim, arr.sizes[dim]) for dim in arr.dims if dim != "time"),
                           arr.attrs.get("units", ""),
                           tuple((dim, tuple(arr[dim].values.tolist())) for dim in arr.dims if dim != "time" and dim in arr.coords))
                if signature is not None and current != signature:
                    raise ValueError(f"Grid, dimensions, or units changed for {model} {name}: {path.name}")
                signature = current
                arrays.append(arr)
                if is_hybrid:
                    pressure = hybrid_pressure(chunk, config)
                    pressure_attrs = dict(pressure.attrs)
                    if lon_name in pressure.dims:
                        pressure = pressure.mean(lon_name, skipna=True)
                    pressure.attrs = pressure_attrs
                    pressures.append(pressure.load())
            source_attrs.append(path.name)
    data = xr.concat(arrays, dim="time").sortby("time")
    months = [(int(t.year), int(t.month)) for t in data.time.values]
    if len(months) != len(set(months)):
        raise ValueError(f"Duplicate months for {model} {name}")
    out = data.to_dataset(name=name)
    if pressures:
        out["air_pressure"] = xr.concat(pressures, dim="time").sortby("time")
    out.attrs = {"project": "JuMACS", "model": model, "experiment": config["model"]["experiment"], "source_variable": name,
                 "source_dataset": config["model"]["dataset_uuid"], "source_archive": config["model"]["archive_base"],
                 "source_files": ";".join(source_attrs), "native_grid_preserved": "true",
                 "monthly_time_label": "source interval midpoint" if is_waccmx(config) else "source time coordinate",
                 "source_longitude_global": str(all(longitude_checks)) if longitude_checks else "not_applicable_already_zonal"}
    dest = ROOT / config["paths"]["zonal"] / f"{name}_monthly_zonal.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".nc.tmp")
    out.to_netcdf(temporary, engine="netcdf4")
    temporary.replace(dest)
    return dest
