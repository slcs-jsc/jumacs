"""Native monthly time series; average longitude only for Amon sources."""
import numpy as np
import xarray as xr

from .config import ROOT, is_waccmx, load_config
from .coordinates import hybrid_pressure
from .reader import open_source, source_files
from .vertical import pressure_report


def mask_near_fill(data, near_fill_relative_tolerance=None):
    """Mask archived values whose magnitude is within the tolerance of a declared fill marker.

    Two quirks of the archives decide the shape of this mask. First, the markers are read
    from the decoding encoding and from the attributes: xarray removes _FillValue from the
    encoding when a file declares several fill markers (it then reports the value in the
    attributes instead), and a marker that only lives in the attributes is still a marker.
    Second, the comparison is on magnitude: a file can store the negative of its declared
    marker as junk (CESM2-WACCM does, on EP fluxes), and a fill marker identifies a
    bit pattern, not a sign the archived junk respects.
    """
    if near_fill_relative_tolerance:
        markers = [data.encoding.get("_FillValue"), data.encoding.get("missing_value"),
                   data.attrs.get("_FillValue"), data.attrs.get("missing_value")]
        for marker in {None if m is None else abs(float(m)) for m in markers}:
            if marker is None:
                continue
            attrs = dict(data.attrs)
            data = data.where(~np.isclose(np.abs(data), marker, rtol=near_fill_relative_tolerance, atol=0.0))
            data.attrs = attrs
    return data


def _pressure_units_factor(units):
    return {"Pa": 1.0, "hPa": 100.0}.get(str(units).strip())


def _surface_pressure_context(model, config, name, level_name, stamps):
    """Month index and grid of the model's surface-pressure files, or None when unusable.

    Files are opened per chunk later, so the (possibly multi-decade) surface-pressure
    payload is read only for the months actually being processed.
    """
    if is_waccmx(config) or "surface_pressure" not in config.get("variables", {}):
        return None
    surface_name = config["variables"]["surface_pressure"]
    lat_name, lon_name = config["coordinates"]["latitude"], config["coordinates"]["longitude"]
    try:
        files = source_files(model, surface_name)
    except FileNotFoundError:
        return None
    index, grids, units = {}, [], None
    for path in files:
        with open_source(path, model, surface_name) as ds:
            data = ds[surface_name]
            if set(data.dims) != {"time", lat_name, lon_name}:
                return None
            if units is None:
                units = data.attrs.get("units")
            elif units != data.attrs.get("units"):
                return None
            grids.append(np.asarray(data[lat_name].values, float))
            for position, stamp in enumerate(data.time.values):
                index.setdefault(str(stamp), (path, position))
    if any(not np.array_equal(grid, grids[0]) for grid in grids[1:]):
        return None
    units_factor = _pressure_units_factor(units)
    if units_factor is None or not all(stamp in index for stamp in stamps):
        return None
    return {"model": model, "variable": surface_name, "index": index, "lat": grids[0],
            "longitude": lon_name, "latitude": lat_name, "units": units, "units_factor": units_factor,
            "level_name": level_name}


def _surface_pressure_ceiling(data, context):
    """Maximum surface pressure around each latitude circle for the chunk's months.

    Only the maximum over longitude is defensible here: it is the single condition
    (pressure level above that maximum) that proves every longitude of a latitude
    circle lies below the model surface. A mean, median, or below-ground fraction
    cannot distinguish a fully buried circle from a partly buried one, and a zonal
    mean source field has no longitude dimension left to repair the latter with.
    """
    if not np.array_equal(context["lat"], np.asarray(data[context["latitude"]].values, float)):
        return None
    rows = []
    for stamp in data["time"].values:
        entry = context["index"].get(str(stamp))
        if entry is None:
            return None
        path, position = entry
        with open_source(path, context["model"], context["variable"]) as ds:
            profile = ds[context["variable"]].isel(time=position).max(context["longitude"])
        rows.append(np.asarray(profile.values, float))
    return xr.DataArray(np.stack(rows), dims=["time", context["latitude"]],
                        coords={"time": data["time"].values, context["latitude"]: data[context["latitude"]].values},
                        attrs={"units": context["units"], "description": "maximum surface pressure around latitude circle"})


def mask_fully_below_surface(data, ceiling, level="plev", level_units_factor=1.0, ceiling_units_factor=1.0):
    """Set zonal values to NaN where the pressure level lies below ground on a whole latitude circle.

    The ceiling is the maximum surface pressure around each latitude circle, so a
    pressure level above that ceiling is below the model surface at every longitude
    and the published zonal value is invalid. Partially below-ground circles stay
    unchanged: the archived field is already a zonal mean and its longitude-resolved
    source is unavailable, so no valid replacement value exists there.
    """
    levels = xr.DataArray(np.asarray(data[level].values, float) * level_units_factor, dims=[level],
                          coords={level: data[level].values})
    below = (levels > ceiling * ceiling_units_factor).transpose(*data.dims)
    attrs = dict(data.attrs)
    data = data.where(~below)
    data.attrs = attrs
    return data


def zonal_mean(ds, variable, longitude="lon", source_kind=None, near_fill_relative_tolerance=None):
    data = mask_near_fill(ds[variable], near_fill_relative_tolerance)
    if longitude in data.dims:
        if data.sizes[longitude] == 1:
            data = data.isel({longitude: 0}, drop=True)
            method = "published zonal field with singleton longitude dimension"
        else:
            data = data.where(np.isfinite(data)).mean(longitude, skipna=True, keep_attrs=True)
            method = "unweighted valid-value longitude mean"
    else:
        method = ("CEDA WACCM-X monthly _zm field (already zonal)" if source_kind == "monthly_zonal_multivariable"
                  else "CEDA AmonZ monthly zonal field (already zonal)")
    data.attrs = {**data.attrs, "zonal_method": method}
    return data


def is_hybrid_level(ds, config):
    level_name = config["coordinates"]["level"]
    return bool(level_name in ds and (ds[level_name].attrs.get("standard_name")
                == "atmosphere_hybrid_sigma_pressure_coordinate" or is_waccmx(config)))


def _pressure_profile(air_pressure, level):
    profile = air_pressure.isel(time=0) if "time" in air_pressure.dims else air_pressure
    for dim in tuple(profile.dims):
        if dim != level:
            profile = profile.mean(dim)
    return np.asarray(profile.values, float).ravel()


def zonal_smoke(model, variable):
    config = load_config(model)
    coords = config.get("coordinates", {})
    name = config["variables"].get(variable, variable)
    report = {"model": model, "variable": variable, "source_variable": name,
              "coordinate_mapping": coords, "ok": False}
    if "longitude" not in coords or "level" not in coords:
        report["reason"] = "coordinates: mapping incomplete (need at least longitude and level)"
        return report, None
    files = source_files(model, name)
    report["source_file"] = files[0].name
    report["source_file_count"] = len(files)
    with open_source(files[0], model, name) as ds:
        sample = ds.isel(time=slice(0, 1))
        lon_name = coords["longitude"]
        arr = zonal_mean(sample, name, lon_name, config["model"].get("source_kind"),
                         config.get("near_fill_relative_tolerance")).load()
        out = arr.to_dataset(name=name)
        finite = float(np.isfinite(arr.values).mean())
        report["zonal_dims"] = {d: int(s) for d, s in arr.sizes.items()}
        report["zonal_finite_fraction"] = round(finite, 4)
        report["longitude_removed"] = lon_name not in arr.dims
        report["field_min"] = float(np.nanmin(arr.values)) if finite else None
        report["field_max"] = float(np.nanmax(arr.values)) if finite else None
        level_name = coords["level"]
        hybrid = is_hybrid_level(sample, config)
        if hybrid:
            report["vertical_route"] = "hybrid_reconstruction"
            pressure = hybrid_pressure(sample, config)
            if lon_name in pressure.dims:
                pressure = pressure.mean(lon_name, skipna=True)
            out["air_pressure"] = pressure.load()
        else:
            coord = coords.get("pressure") if coords.get("pressure") in sample.coords else level_name
            report["vertical_route"] = f"pressure_level:{coord}" if coord in sample.coords else f"native_level:{coord}"
            if coord in sample.coords:
                pressure = xr.DataArray(
                    np.asarray(sample[coord].values, float), dims=[coord],
                    attrs={"units": sample[coord].attrs.get("units", "Pa"),
                           "standard_name": "air_pressure", "source_coordinate": coord})
                out["air_pressure"] = pressure.broadcast_like(arr).transpose(*arr.dims)
                level_name = coord
    if "air_pressure" in out:
        profile = _pressure_profile(out["air_pressure"], level_name)
        profile = profile[np.isfinite(profile)]
        if profile.size:
            diffs = np.diff(profile)
            report["pressure_units"] = out["air_pressure"].attrs.get("units")
            report["pressure_min_pa"] = pressure_report(profile.min())
            report["pressure_max_pa"] = pressure_report(profile.max())
            report["pressure_monotonic_decreasing"] = bool(profile[0] >= profile[-1])
            report["pressure_decreasing_fraction"] = round(float((diffs < 0).mean()), 4)
            report["pressure_plausible"] = bool(profile.max() <= 120000.0 and profile.min() > 0.0 and profile.max() >= 1000.0)
        else:
            report["reason"] = "pressure profile empty (non-finite)"
    else:
        report["reason"] = "no recoverable pressure coordinate (hybrid coefficients or plev absent)"
    report["ok"] = bool(report.get("zonal_finite_fraction", 0) >= 0.9 and report.get("longitude_removed", False)
                        and report.get("pressure_plausible", False) and report.get("pressure_decreasing_fraction", 0) >= 0.9)
    dest = ROOT / "products/diagnostics/zonal_smoke" / f"{model}_{name}_zonal_smoke.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(dest)
    report["output_file"] = str(dest)
    return report, dest


def build_zonal(model, variable):
    config = load_config(model)
    name = config["variables"].get(variable, variable)
    level_name = config["coordinates"]["level"]
    arrays = []
    pressures = []
    source_attrs = []
    signature = None
    longitude_checks = []
    surface_context = None
    if (config["coordinates"].get("pressure") and config["coordinates"].get("pressure") == level_name
            and "surface_pressure" in config.get("variables", {})):
        stamps = []
        for path in source_files(model, name):
            with open_source(path, model, name) as ds:
                if level_name not in ds[name].dims or is_hybrid_level(ds, config):
                    stamps = []
                    break
                stamps.extend(str(stamp) for stamp in ds[name].time.values)
        if stamps:
            surface_context = _surface_pressure_context(model, config, name, level_name, stamps)
    masked_months = 0
    for path in source_files(model, name):
        with open_source(path, model, name) as ds:
            lon_name = config["coordinates"]["longitude"]
            if lon_name in ds[name].dims and ds[name].sizes[lon_name] > 1:
                longitudes = np.sort(np.unique(np.mod(ds[lon_name].values, 360)))
                gaps = np.diff(np.r_[longitudes, longitudes[0] + 360])
                longitude_checks.append(bool(len(longitudes) > 1 and np.max(gaps) <= 1.5 * np.median(gaps)))
            is_hybrid = is_hybrid_level(ds, config)
            # Limit the largest in-memory full field to 12 monthly samples.
            for start in range(0, ds.sizes["time"], 12):
                chunk = ds.isel(time=slice(start, start + 12))
                arr = zonal_mean(chunk, name, lon_name, config["model"].get("source_kind"),
                                 config.get("near_fill_relative_tolerance")).load()
                if config.get("squeeze_singleton_level") and arr.sizes.get(config["coordinates"]["level"]) == 1:
                    arr = arr.squeeze(config["coordinates"]["level"], drop=True)
                current = (tuple((dim, arr.sizes[dim]) for dim in arr.dims if dim != "time"),
                           arr.attrs.get("units", ""),
                           tuple((dim, tuple(arr[dim].values.tolist())) for dim in arr.dims if dim != "time" and dim in arr.coords))
                if signature is not None and current != signature:
                    raise ValueError(f"Grid, dimensions, or units changed for {model} {name}: {path.name}")
                signature = current
                if surface_context is not None and level_name in arr.dims and "time" in arr.dims:
                    ceiling = _surface_pressure_ceiling(arr, surface_context)
                    level_factor = _pressure_units_factor(arr[level_name].attrs.get("units"))
                    if ceiling is None or level_factor is None:
                        surface_context = None
                    else:
                        masked = mask_fully_below_surface(arr, ceiling, level=level_name,
                                                          level_units_factor=level_factor,
                                                          ceiling_units_factor=surface_context["units_factor"])
                        if int(np.isfinite(masked.values).sum()) < int(np.isfinite(arr.values).sum()):
                            masked_months += 1
                        arr = masked
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
    foreign = [c for c in out.coords if c not in out.dims
               and (out[c].ndim == 0 or not set(out[c].dims) <= set(data.dims))]
    if foreign:
        out = out.drop_vars(foreign)
    if pressures:
        out["air_pressure"] = xr.concat(pressures, dim="time").sortby("time")
    out.attrs = {"project": "JuMACS", "model": model, "experiment": config["model"]["experiment"], "source_variable": name,
                 "source_dataset": config["model"]["dataset_uuid"], "source_archive": config["model"]["archive_base"],
                 "source_files": ";".join(source_attrs), "native_grid_preserved": "true",
                 "monthly_time_label": "source interval midpoint" if is_waccmx(config) else "source time coordinate",
                 "source_longitude_global": str(all(longitude_checks)) if longitude_checks else "not_applicable_already_zonal"}
    if masked_months:
        out.attrs["below_ground_mask"] = "pressure_level > maximum surface pressure around latitude circle"
        out.attrs["below_ground_surface_pressure_variable"] = config["variables"]["surface_pressure"]
        out.attrs["below_ground_limitation"] = ("partially below-ground latitude circles remain unchanged: "
                                                "the source field is already a zonal mean")
    dest = ROOT / config["paths"]["zonal"] / f"{name}_monthly_zonal.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".nc.tmp")
    out.to_netcdf(temporary, engine="netcdf4")
    temporary.replace(dest)
    return dest
