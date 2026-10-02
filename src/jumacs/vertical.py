"""Shared vertical machinery: native pressure recovery and log-pressure regridding."""
import numpy as np
import xarray as xr

LEVEL_PRIORITY = ("lev", "plev", "pressure")
NON_VERTICAL = ("time", "lat", "month", "lon", "latitude")
PA_TO_HPA = {"pa": 1.0, "pascal": 1.0, "pascals": 1.0, "hpa": 100.0, "hectopascal": 100.0, "hectopascals": 100.0}

PRESSURE_ATTRS = {"standard_name": "air_pressure", "units": "Pa", "positive": "down",
                  "long_name": "air pressure on the common JuMACS pressure grid"}


def pressure_report(value):
    """Pressure in Pa for reports and attributes: significant digits, never a rounded 0.0."""
    number = float(value)
    return float(f"{number:.6g}")


def vertical_dimension(field, pressure=None):
    """The native vertical dimension of a field, checked against the pressure array."""
    candidates = [dim for dim in field.dims if dim not in NON_VERTICAL]
    if pressure is not None:
        shared = [dim for dim in candidates if dim in pressure.dims]
        if shared:
            candidates = shared
    for dim in LEVEL_PRIORITY:
        if dim in candidates:
            return dim
    return candidates[0] if candidates else None


def pressure_scale(attrs):
    return PA_TO_HPA.get(str(attrs.get("units", "")).strip().lower())


def native_pressure(ds, field):
    """Pressure in Pa for a native monthly zonal field, or None when unrecoverable.

    Hybrid-sigma sources carry a reconstructed ``air_pressure(time, lev, lat)``
    variable; pressure-level sources keep their own ``plev``-style coordinate.
    """
    level = vertical_dimension(field)
    if level is None:
        return None
    if "air_pressure" in ds and level in ds["air_pressure"].dims:
        pressure = ds["air_pressure"]
        scale = pressure_scale(pressure.attrs) or 1.0
        return (pressure * scale).assign_attrs(units="Pa")
    if level in ds.coords:
        attrs = ds[level].attrs
        if "air_pressure" in str(attrs.get("standard_name", "")) or pressure_scale(attrs):
            scale = pressure_scale(attrs) or 1.0
            return (ds[level] * scale).rename("air_pressure").assign_attrs(units="Pa")
    return None


def interpolate_log_pressure(field, pressure, target_pa):
    """Linear interpolation in log pressure on one column; never extrapolates."""
    level = next((dim for dim in pressure.dims if dim in field.dims and dim in LEVEL_PRIORITY), None)
    if level is None:
        level = next((dim for dim in pressure.dims if dim in field.dims and dim not in NON_VERTICAL), None)
    if level is None:
        raise ValueError("No shared native vertical dimension")
    targets = np.asarray(target_pa, dtype=float)
    if np.any(targets <= 0):
        raise ValueError("Pressure levels must be positive")

    def one_column(values, p):
        valid = np.isfinite(values) & np.isfinite(p) & (p > 0)
        if valid.sum() < 2:
            return np.full(targets.shape, np.nan)
        x = np.log(p[valid]); y = values[valid]
        order = np.argsort(x)
        x, y = x[order], y[order]
        return np.interp(np.log(targets), x, y, left=np.nan, right=np.nan)

    result = xr.apply_ufunc(one_column, field, pressure, input_core_dims=[[level], [level]],
                            output_core_dims=[["pressure"]], exclude_dims={level} if level == "pressure" else set(),
                            vectorize=True, dask="allowed",
                            output_dtypes=[float], dask_gufunc_kwargs={"output_sizes": {"pressure": len(targets)}})
    result = result.transpose(..., "pressure")
    if "lat" in result.dims:
        result = result.transpose(..., "lat")
    return result.assign_coords(pressure=(("pressure",), targets, dict(PRESSURE_ATTRS)))


def regrid_to_common_grid(field, pressure, levels):
    """Move a native field onto the common pressure grid and keep its units."""
    units = field.attrs.get("units", "")
    regridded = interpolate_log_pressure(field, pressure, levels)
    regridded.attrs = dict(field.attrs)
    if units:
        regridded.attrs["units"] = units
    regridded.attrs["vertical_interpolation"] = "linear in log(pressure)"
    regridded.attrs["vertical_extrapolation"] = "none"
    return regridded
