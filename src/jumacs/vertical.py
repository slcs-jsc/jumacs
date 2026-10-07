"""Shared vertical machinery: native pressure recovery and log-pressure regridding."""
import numpy as np
import xarray as xr

LEVEL_PRIORITY = ("lev", "plev", "pressure")
NON_VERTICAL = ("time", "lat", "month", "lon", "latitude")


def is_latitude_dimension(name):
    """Whether one dimension name is a native latitude axis: 'lat', or 'lat_2', 'lat_3', ... of a second grid."""
    if name == "lat":
        return True
    if not name.startswith("lat_"):
        return False
    return name[len("lat_"):].isdigit()

# The fewest levels two profiles must hold usable in common before anything can be joined on them: a single
# shared level fixes no transition, so the extension stage refuses it and the coverage diagnostics do not
# offer it as a candidate.
MINIMUM_OVERLAP_LEVELS = 2
PA_TO_HPA = {"pa": 1.0, "pascal": 1.0, "pascals": 1.0, "hpa": 100.0, "hectopascal": 100.0, "hectopascals": 100.0}

PRESSURE_ATTRS = {"standard_name": "air_pressure", "units": "Pa", "positive": "down",
                  "long_name": "air pressure"}


def pressure_report(value):
    """Pressure in Pa for reports and attributes: significant digits, never a rounded 0.0."""
    number = float(value)
    return float(f"{number:.6g}")


def vertical_dimension(field, pressure=None):
    """The native vertical dimension of a field, checked against the pressure array."""
    candidates = [dim for dim in field.dims if dim not in NON_VERTICAL and not is_latitude_dimension(dim)]
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


def product_pressure(ds, field):
    """Pressure in Pa for a field of a published climatology product.

    Hybrid products publish a climatological ``air_pressure(time, lev, lat)``
    field and point at it with ``pressure_field``; pressure-level products carry
    their native pressure-valued coordinate instead, which ``native_pressure``
    finds. Nothing is interpolated here: the pressure is read as published.
    """
    named = str(field.attrs.get("pressure_field", "")).strip()
    if named and named in ds:
        pressure = ds[named]
        scale = pressure_scale(pressure.attrs) or 1.0
        return (pressure * scale).assign_attrs(units="Pa")
    return native_pressure(ds, field)


def own_level_coordinate(array, level):
    """The array with its own copy of the metadata of the vertical coordinate it is asked to consume.

    ``apply_ufunc`` rebuilds the coordinates of the dimensions it consumes and writes the result back into
    the coordinate objects it was handed, which for a native product are the very objects the dataset holds:
    reading the pressure straight off a pressure-level product and interpolating one column stripped the
    units and the standard name from the level coordinate of the product itself, so every field of that grid
    read afterwards as having no pressure at all and was dropped without a word. Owning a copy here keeps
    the published product as it was written.
    """
    if level not in array.coords:
        return array
    detached = array.copy(deep=False)
    detached.coords[level] = array[level].variable.copy()
    return detached


def interpolate_log_pressure(field, pressure, target_pa, bridge_gaps=True):
    """Linear interpolation in log pressure on one column; never extrapolates.

    With ``bridge_gaps`` the missing samples of a column are set aside first, so a level is filled whenever two
    remaining samples bracket it, however far apart. Without it the missing samples stay in place and act as
    barriers: only levels whose own two neighbours are both present are filled, and a gap in the middle of a
    profile remains a gap rather than becoming a long-distance interpolation.
    """
    level = next((dim for dim in pressure.dims if dim in field.dims and dim in LEVEL_PRIORITY), None)
    if level is None:
        level = next((dim for dim in pressure.dims if dim in field.dims and dim not in NON_VERTICAL
                      and not is_latitude_dimension(dim)), None)
    if level is None:
        raise ValueError("No shared native vertical dimension")
    field, pressure = own_level_coordinate(field, level), own_level_coordinate(pressure, level)
    targets = np.asarray(target_pa, dtype=float)
    if np.any(targets <= 0):
        raise ValueError("Pressure levels must be positive")

    def one_column(values, p):
        placed = np.isfinite(p) & (p > 0)
        if bridge_gaps:
            valid = placed & np.isfinite(values)
            if valid.sum() < 2:
                return np.full(targets.shape, np.nan)
            x = np.log(p[valid]); y = values[valid]
            order = np.argsort(x)
            x, y = x[order], y[order]
            unique, first = np.unique(x, return_index=True)
            if unique.size != x.size:
                x, y = unique, y[np.sort(first)]
                if x.size < 2:
                    return np.full(targets.shape, np.nan)
            return np.interp(np.log(targets), x, y, left=np.nan, right=np.nan)
        if placed.sum() < 2:
            return np.full(targets.shape, np.nan)
        x = np.log(p[placed]); y = np.where(np.isfinite(values[placed]), values[placed], np.nan)
        order = np.argsort(x, kind="stable")
        return np.interp(np.log(targets), x[order], y[order], left=np.nan, right=np.nan)

    result = xr.apply_ufunc(one_column, field, pressure, input_core_dims=[[level], [level]],
                            output_core_dims=[["pressure"]], exclude_dims={level} if level == "pressure" else set(),
                            vectorize=True, dask="allowed",
                            output_dtypes=[float], dask_gufunc_kwargs={"output_sizes": {"pressure": len(targets)}})
    result = result.transpose(..., "pressure")
    axis = next((dim for dim in result.dims if is_latitude_dimension(dim)), None)
    if axis is not None:
        result = result.transpose(..., axis)
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
