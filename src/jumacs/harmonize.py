"""Harmonized application products: one file per model on a 1 km altitude grid.

Each source model keeps its own scientifically valid vertical range. A product runs
from the surface at 0 km up to the highest level the model resolves, never padded to a
common top and never clipped at one. What the models share is the format, not a common
ceiling: twelve climatological months, five degree latitude bands, one kilometre
altitude spacing, and one naming and statistics convention. There is no vertical blend
with WACCM-X and no extension above a model's own top.
"""

import numpy as np
import xarray as xr

from .climatology import product_name
from .config import ROOT, load_config, model_metadata
from .remap import BAND_WIDTH_DEGREES, area_weighted_bands, latitude_band_weights, latitude_bands

ALTITUDE_STEP_KM = 1
STATISTICS = ("mean", "sigma", "minimum", "maximum")
GRAVITY = 9.80665
HARMONIZED_SUBDIR = "harmonized"

# U.S. Standard Atmosphere height of a pressure level, used only as an independent
# yardstick to tell geopotential (m^2 s^-2) apart from geopotential height (m) when a
# source mislabels one as the other. It is never used to place data on the altitude grid.
_ISA_PRESSURE_PA = np.array([101325.0, 22632.0, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734, 0.01])
_ISA_HEIGHT_M = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0, 110000.0])

# Pressure on a model's own levels is published under these names; when a vertical
# dimension is itself a pressure axis the coordinate stands in for them.
_PRESSURE_FIELDS = ("air_pressure", "air_pressure_lev", "air_pressure_lev_2")


def interpolate_profile(height_km, values, target_km):
    """Interpolate valid native levels onto a target altitude without extending their range."""
    good = np.isfinite(height_km) & np.isfinite(values)
    if good.sum() < 2:
        return np.full(np.shape(target_km), np.nan)
    height, values = height_km[good], values[good]
    order = np.argsort(height)
    height, values = height[order], values[order]
    height, unique = np.unique(height, return_index=True)
    values = values[unique]
    return np.interp(target_km, height, values, left=np.nan, right=np.nan)


def _isa_height_m(pressure_pa):
    pressure = np.log(np.clip(np.asarray(pressure_pa, dtype="float64"), 1e-6, None))
    anchor_p = np.log(_ISA_PRESSURE_PA[::-1])
    anchor_h = _ISA_HEIGHT_M[::-1]
    return np.interp(pressure, anchor_p, anchor_h)


def sanitize_geopotential(z_native, pressure_pa):
    """Return geopotential height in metres, correcting a field stored as geopotential.

    A field is read as geopotential rather than height when its ratio to the standard
    atmosphere height over the stratosphere sits near one over gravity. Without a
    paired pressure to compare against the values are taken at face value, which is the
    safe reading for the models that publish true height.
    """
    z = np.asarray(z_native, dtype="float64")
    p = np.asarray(pressure_pa, dtype="float64")
    band = np.isfinite(z) & np.isfinite(p) & (p >= 1.0e3) & (p <= 3.0e4) & (z > 0.0)
    if band.sum() < 4:
        return z
    ratio = z[band] / _isa_height_m(p[band])
    scale = float(np.nanmedian(ratio))
    if 5.0 < scale < 20.0:
        return z / GRAVITY
    return z


def _pressure_on_dim(ds, dim):
    """Pressure in Pa on a vertical dimension, as a DataArray, or None if unresolvable."""
    for name in _PRESSURE_FIELDS:
        if name in ds and dim in ds[name].dims:
            field = ds[name]
            scale = 100.0 if field.attrs.get("units") == "hPa" else 1.0
            return (field * scale).assign_attrs(units="Pa")
    if dim in ds.coords:
        coord = ds[dim]
        values = np.asarray(coord.values, dtype="float64")
        positive = values.size > 1 and bool(np.isfinite(values).all()) and bool((values > 1.0).all())
        if coord.attrs.get("units") in ("Pa", "hPa") or positive:
            scale = 100.0 if coord.attrs.get("units") == "hPa" else 1.0
            return (coord * scale).assign_attrs(units="Pa")
    return None


def _valid_top_km(altitude_km):
    """Highest altitude reached, following the surface-upward monotonic rise and stopping there."""
    profile = np.nanmean(np.where(np.isfinite(altitude_km), altitude_km, np.nan), axis=(0, 2))
    top = 0
    for index in range(1, profile.size):
        if np.isfinite(profile[index]) and np.isfinite(profile[index - 1]) and profile[index] > profile[index - 1]:
            top = index
        else:
            break
    return float(profile[top]) if np.isfinite(profile[top]) else 0.0


def _altitude_for_field(field_dim, height_dim, z_km, pressure_height, pressure_field):
    """Altitude in km on a field's own levels, from the model's own height-pressure profile."""
    if field_dim == height_dim:
        return z_km
    return _map_through_log_pressure(z_km, pressure_height, pressure_field)


def _pressure_values(resolved, dims, months, latitudes):
    """Pressure in Pa aligned to (month, level, latitude), from a field or a 1D level coordinate."""
    if resolved is None:
        return None
    if resolved.ndim >= 3:
        return np.asarray(resolved.transpose(*dims).values, dtype="float64")
    values = np.asarray(resolved.values, dtype="float64")
    return np.broadcast_to(values[np.newaxis, :, np.newaxis], (months, values.size, latitudes)).astype("float64")


def _map_through_log_pressure(z_km, pressure_height, pressure_field):
    """Altitude of each field level by reading the model's height as a function of -log pressure."""
    months, _, latitudes = pressure_field.shape
    mapped = np.full(pressure_field.shape, np.nan)
    log_height = -np.log(np.where(pressure_height > 0, pressure_height, np.nan))
    log_target = -np.log(np.where(pressure_field > 0, pressure_field, np.nan))
    for month in range(months):
        for column in range(latitudes):
            mapped[month, :, column] = interpolate_profile(
                log_height[month, :, column], z_km[month, :, column], log_target[month, :, column])
    return mapped


def _to_altitude_grid(altitude_km, value, target_km):
    months, _, latitudes = value.shape
    grid = np.full((months, len(target_km), latitudes), np.nan)
    for month in range(months):
        for column in range(latitudes):
            grid[month, :, column] = interpolate_profile(altitude_km[month, :, column], value[month, :, column],
                                                          target_km)
    return grid


def harmonized_model_names():
    """The ready CCMI models plus WACCM-X, the models a harmonized product is built for."""
    from .config import ready_ccmi_model_names
    return (*ready_ccmi_model_names(), "WACCM-X")


def harmonized_path(model, start_year, end_year, latitude_step=5, altitude_step=ALTITUDE_STEP_KM):
    metadata = model_metadata(model)
    name = f"jumacs_{metadata['slug']}_{start_year}-{end_year}_{latitude_step}deg_{altitude_step}km.nc"
    return ROOT / "products" / HARMONIZED_SUBDIR / metadata["name"] / name


def _product_variables(config):
    skip = {"geopotential_height", "surface_pressure"}
    return [canonical for canonical in config["variables"] if canonical not in skip]


def harmonize_model(model, start_year=None, end_year=None, latitude_step=5, altitude_step=ALTITUDE_STEP_KM):
    """Build one model's harmonized product and return its path and a short report."""
    from .config import reference_period
    reference = reference_period()["reference_period"]
    start_year = start_year or reference["start_year"]
    end_year = end_year or reference["end_year"]
    config = load_config(model)
    source = ROOT / config["paths"]["climatology"] / product_name(model, start_year, end_year)
    if not source.exists():
        raise FileNotFoundError(f"native climatology missing for {model}: {source}")
    with xr.open_dataset(source) as ds:
        prepared = _harmonize_open_dataset(ds, model, config, start_year, end_year, latitude_step, altitude_step)
    dest = harmonized_path(model, start_year, end_year, latitude_step, altitude_step)
    _write(prepared["dataset"], dest)
    return dest, prepared["report"]


def _harmonize_open_dataset(ds, model, config, start_year, end_year, latitude_step, altitude_step):
    height_source = config["variables"].get("geopotential_height")
    if not height_source or f"{height_source}_mean" not in ds:
        raise ValueError(f"{model}: geopotential height is required to place fields on altitude")
    height = ds[f"{height_source}_mean"]
    height_dim = height.dims[1]
    months, latitudes = height.shape[0], height.shape[2]
    pressure_height = _pressure_on_dim(ds, height_dim)
    if pressure_height is None:
        raise ValueError(f"{model}: no pressure found on the geopotential height dimension {height_dim!r}")
    pressure_height_values = _pressure_values(pressure_height, height.dims, months, latitudes)
    z_m = sanitize_geopotential(height.values, pressure_height_values)
    z_km = z_m / 1000.0
    sanitized = not np.allclose(z_m, np.asarray(height.values, dtype="float64"), equal_nan=True)

    latitude = np.asarray(ds["lat"].values, dtype="float64")
    band_edges, band_centers = latitude_bands(BAND_WIDTH_DEGREES)
    if abs(latitude_step - BAND_WIDTH_DEGREES) > 1e-9:
        band_edges, band_centers = latitude_bands(latitude_step)
    weights = latitude_band_weights(latitude, band_edges)

    top_km = _valid_top_km(z_km)
    top_level = int(np.floor(top_km))
    if top_level < 1:
        raise ValueError(f"{model}: derived valid top is not above the surface ({top_km:.2f} km)")
    target_km = np.arange(0, top_level + 1, altitude_step, dtype="float64")

    altitude_cache = {height_dim: z_km}
    data = {}
    carried = {}
    for canonical in _product_variables(config):
        source_variable = config["variables"][canonical]
        key = canonical.lower().replace("-", "_")
        variable_report = {}
        for statistic in STATISTICS:
            field_name = f"{source_variable}_{statistic}"
            if field_name not in ds or ds[field_name].ndim != 3 or "lat" not in ds[field_name].dims:
                continue
            field = ds[field_name]
            field_dim = field.dims[1]
            if field_dim not in altitude_cache:
                resolved = _pressure_on_dim(ds, field_dim)
                if resolved is None:
                    break
                pressure_field = _pressure_values(resolved, field.dims, months, latitudes)
                altitude_cache[field_dim] = _map_through_log_pressure(z_km, pressure_height_values, pressure_field)
            altitude = altitude_cache[field_dim]
            gridded = _to_altitude_grid(altitude, np.asarray(field.values, dtype="float64"), target_km)
            band_values = area_weighted_bands(
                xr.DataArray(gridded, dims=("month", "altitude", "lat"),
                             coords={"lat": ("lat", latitude)}), weights, band_centers)
            data[f"{key}_{statistic}"] = (("month", "altitude", "lat"), np.asarray(band_values.values, "float32"),
                                           _field_attrs(field, canonical, source_variable))
            variable_report[statistic] = int(np.count_nonzero(np.isfinite(gridded)))
        if variable_report:
            carried[key] = variable_report

    dataset = xr.Dataset(
        data,
        coords={"month": np.arange(1, 13, dtype="int8"),
                "altitude": ("altitude", target_km.astype("float64"),
                            {"units": "km", "standard_name": "altitude", "axis": "Z", "positive": "up",
                             "long_name": "altitude above the surface on a one kilometre grid"}),
                "lat": ("lat", band_centers.astype("float64"),
                        {"units": "degrees_north", "standard_name": "latitude",
                         "long_name": f"{latitude_step:g} degree latitude band centre",
                         "cell_methods": "area_weighted zonal mean"})},
    )
    metadata = model_metadata(model)
    dataset.attrs = {
        "project": "JuMACS", "product_type": "harmonized_climatology", "model": metadata["name"],
        "display_name": metadata["display_name"], "institution": metadata["institution"],
        "model_kind": metadata["kind"], "experiment": config["model"].get("experiment", ""),
        "source_dataset": config["model"].get("dataset_uuid", ""),
        "climatology_period": f"{start_year}-{end_year}", "reference_period_years": end_year - start_year + 1,
        "altitude_step_km": altitude_step, "latitude_band_degrees": float(latitude_step),
        "vertical_extent_km": [0.0, float(target_km[-1])], "model_valid_top_km": round(top_km, 3),
        "vertical_domain": "each model keeps its own valid top; no common ceiling, no padding, no extension",
        "geopotential_units_corrected": int(sanitized),
        "missing_policy": "NaN outside a model's valid vertical range; no extrapolation above the model top",
        "statistics": ", ".join(STATISTICS),
        "history": f"native monthly zonal climatology remapped to a {altitude_step} km altitude grid and "
                   f"{latitude_step:g} degree area-weighted latitude bands",
    }
    report = {"model": model, "valid_top_km": round(top_km, 3), "altitude_levels": len(target_km),
              "latitude_bands": len(band_centers), "geopotential_corrected": bool(sanitized),
              "variables": len(carried), "statistics": carried}
    return {"dataset": dataset, "report": report}


def _field_attrs(field, canonical, source_variable):
    attrs = {"units": field.attrs.get("units", ""), "canonical_variable": canonical,
             "source_variable": source_variable,
             "cell_methods": "area_weighted zonal mean; climatological mean and spread across years"}
    for key in ("standard_name", "long_name"):
        if key in field.attrs:
            attrs[key] = field.attrs[key]
    return attrs


def _write(dataset, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
                for name in dataset.data_vars}
    temp = dest.with_suffix(".nc.tmp")
    dataset.to_netcdf(temp, engine="netcdf4", encoding=encoding)
    temp.replace(dest)


def harmonize_all(models=None, start_year=None, end_year=None, latitude_step=5, altitude_step=ALTITUDE_STEP_KM):
    models = tuple(models) if models else harmonized_model_names()
    return [harmonize_model(model, start_year, end_year, latitude_step, altitude_step) for model in models]
