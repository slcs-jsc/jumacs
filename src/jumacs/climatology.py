"""Period-specific monthly statistics on one common pressure grid."""
import datetime as dt

import numpy as np
import xarray as xr

from . import cf
from .config import ROOT, load_config, reference_period, is_waccmx, model_slug, vertical_grid
from .vertical import PRESSURE_ATTRS, native_pressure, regrid_to_common_grid, vertical_dimension


class MissingPressureCoordinate(ValueError):
    """A three-dimensional field has no recoverable native pressure coordinate."""


def _slug(model, config):
    slug = config["model"].get("slug")
    if slug:
        return slug
    return "waccmx" if is_waccmx(config) else model_slug(model) + "_refd1"


def product_name(model, start_year, end_year):
    config = load_config(model)
    return f"jumacs_{_slug(model, config)}_climatology_{start_year}-{end_year}.nc"


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


def on_common_grid(ds, name, start_year, end_year, grid, native_coordinate=""):
    """Move one native monthly zonal variable onto the common pressure grid.

    Interpolation happens here, before any statistic is taken, because hybrid
    pressure levels change with time and latitude: averaging first and
    interpolating the mean afterwards would mix different air masses.
    Two-dimensional fields (surface and tropopause quantities) are returned
    unchanged.
    """
    field = ds[name]
    level = vertical_dimension(field)
    if level is None:
        return field, {"regridded_to_common_pressure_grid": "false",
                       "vertical_treatment": "two-dimensional field; no vertical interpolation"}
    pressure = native_pressure(ds, field)
    if pressure is None:
        raise MissingPressureCoordinate(
            f"{name}: the zonal product has neither an air_pressure variable nor a pressure-valued "
            f"'{level}' coordinate, so it cannot be placed on the common pressure grid")
    inside = (ds.time.dt.year >= start_year) & (ds.time.dt.year <= end_year)
    native = pressure.where(inside, drop=True) if "time" in pressure.dims else pressure
    regridded = regrid_to_common_grid(field.where(inside, drop=True), native, grid["levels"])
    provenance = {"regridded_to_common_pressure_grid": "true",
                  "native_level_dimension": level,
                  "native_level_count": int(field.sizes[level]),
                  "native_vertical_coordinate": native_coordinate or level,
                  "vertical_interpolation": "linear in log(pressure) on every month and latitude",
                  "vertical_extrapolation": "none; targets outside a profile's finite range are NaN"}
    finite = native.where(np.isfinite(native) & (native > 0))
    if finite.size:
        provenance["native_pressure_min_pa"] = round(float(finite.min()), 4)
        provenance["native_pressure_max_pa"] = round(float(finite.max()), 4)
    return regridded, provenance


def variable_statistics(name, stats, source_attrs, regrid_provenance, standard_names=None):
    """Rename one variable's statistics and describe them for CF."""
    renamed = stats.rename({statistic: f"{name}_{statistic}" for statistic in stats.data_vars})
    units = source_attrs.get("units", "")
    curated = (standard_names or {}).get(name) or (standard_names or {}).get(name.lower())
    if curated:
        standard_name = cf.curated_standard_name(curated, units)
        standard_name_source = "config" if standard_name else ""
    else:
        standard_name = cf.valid_standard_name(source_attrs.get("standard_name"))
        standard_name_source = "archive" if standard_name else ""
    base_long_name = source_attrs.get("long_name") or source_attrs.get("original_name") or name
    original_name = str(source_attrs.get("original_name") or "")
    for statistic in cf.STATISTIC_ORDER:
        field = renamed[f"{name}_{statistic}"].astype(cf.STATISTIC_DTYPES[statistic])
        attrs = {"long_name": f"{base_long_name}: {cf.statistic_long_name(statistic)}",
                 "units": "1" if statistic == "n_years" else units,
                 "source_variable": name, "cell_methods": cf.cell_methods(statistic)}
        if standard_name and statistic != "n_years":
            attrs["standard_name"] = standard_name
            attrs["standard_name_source"] = standard_name_source
        if original_name and original_name != name:
            attrs["original_name"] = original_name
        if statistic == "mean":
            for key, value in regrid_provenance.items():
                attrs[key] = value
        field.attrs = attrs
        renamed[f"{name}_{statistic}"] = field
    return renamed[[f"{name}_{statistic}" for statistic in cf.STATISTIC_ORDER]]


def product_attributes(model, config, start_year, end_year, names, coverage_bounds, grid, reference):
    """Model-wide provenance, recorded once per product instead of per variable."""
    coordinate = grid["coordinate"]
    model_info = config["model"]
    return {
        "Conventions": cf.CONVENTIONS,
        "title": f"{model_info.get('display_name', model)} {model_info['experiment']} monthly zonal climatology "
                 f"{start_year}-{end_year} (JuMACS)",
        "project": "JuMACS",
        "institution": model_info.get("institution", "unresolved"),
        "source": f"{model_info.get('display_name', model)} ({model_info['source_id']}) {model_info['experiment']} "
                  f"r1i1p1f1 via {model_info['archive_base']}",
        "model": model, "experiment": model_info["experiment"],
        "source_dataset": model_info["dataset_uuid"], "source_archive": model_info["archive_base"],
        "climatology_period": f"{start_year}-{end_year}",
        "reference_period_years": end_year - start_year + 1,
        "nominal_reference_year": reference["nominal_reference_year"],
        "time_coverage_start": f"{start_year}-01-01",
        "time_coverage_end": f"{end_year}-12-31",
        "source_time_coverage_start": str(min(coverage_bounds)) if coverage_bounds else "",
        "source_time_coverage_end": str(max(coverage_bounds)) if coverage_bounds else "",
        "product": "monthly climatology of zonal means on one common pressure grid",
        "horizontal_grid": "zonal mean over all longitudes; latitude is the only horizontal dimension",
        "statistics": " ".join(cf.STATISTIC_ORDER),
        "variable_naming": ("one field per source variable and statistic, named <variable>_<statistic>: "
                            "<variable>_mean, <variable>_sigma, <variable>_minimum, <variable>_maximum, "
                            "<variable>_n_years"),
        "variable_count": len(names),
        "bias_correction": "none", "trend_correction": "none", "model_combination": "none",
        "vertical_coordinate": f"{coordinate} (Pa); one common grid shared by every variable",
        "vertical_grid": grid["description"] or f"{len(grid['levels'])} levels, "
                                                f"{grid['levels'][-1]:g} to {grid['levels'][0]:g} Pa",
        "vertical_level_count": len(grid["levels"]),
        "vertical_interpolation": "monthly zonal fields regridded linearly in log(pressure) before any statistic",
        "vertical_extrapolation": "none; values outside a source profile's finite range stay NaN",
        "missing_value_policy": "NaN where no source value exists; n_years counts finite contributions",
        "comment": ("CF-1.13 climatological time axis: 12 monthly cells summarising every year in the period, "
                    "with across-year statistics in cell_methods and month intervals in climatology_bounds"),
        "history": (f"created {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}: monthly zonal values "
                    "regridded to the common pressure grid, grouped by calendar month, written as one "
                    f"{cf.CONVENTIONS} NetCDF per model"),
    }


def assemble_product(model, start_year, end_year, parts, coverage_bounds=(), names=None):
    """Merge per-variable statistics into the single CF-1.13 product dataset."""
    config = load_config(model)
    grid = vertical_grid()
    reference = reference_period()["reference_period"]
    coordinate = grid["coordinate"]
    names = sorted(names or {part[key].attrs["source_variable"] for part in parts for key in part.data_vars})
    months = [part.rename({"month": "time"}) for part in parts]
    combined = xr.merge(months, join="outer", compat="override")
    stray = sorted(dim for dim in combined.dims if dim not in (coordinate, "time", "lat", cf.BOUNDS_DIMENSION))
    if stray:
        raise RuntimeError(f"Climatology product kept per-variable vertical dimensions {stray}; every "
                           f"three-dimensional field must share the common '{coordinate}' grid")
    months = [int(month) for month in combined["time"].values]
    absent = sorted(set(range(1, 13)) - set(months))
    if absent:
        raise RuntimeError(f"Climatological month cells {absent} have no data; a monthly climatology needs all "
                           f"twelve calendar months in the period")
    if months != list(range(1, 13)):
        combined = combined.sel(time=list(range(1, 13)))
    cf_time = cf.climatology_time(start_year, end_year, reference["nominal_reference_year"])
    combined = combined.assign_coords(time=(("time",), cf_time["values"], cf_time["attrs"]))
    combined["climatology_bounds"] = (("time", cf.BOUNDS_DIMENSION), cf_time["bounds"])
    combined["climatology_bounds"].attrs = {
        "units": cf_time["units"], "calendar": cf_time["calendar"],
        "long_name": "beginning and end of the climatological month interval",
        "comment": ("bounds of the monthly interval each statistic summarises, taken from a common non-leap year; "
                    "climatology_bounds carries no fill value")}
    pressure_attrs = dict(PRESSURE_ATTRS)
    pressure_attrs.update({"axis": "Z", "comment": "shared vertical grid; every species uses these levels"})
    levels = np.asarray(grid["levels"], "float64")
    if coordinate in combined.coords and combined[coordinate].size != levels.size:
        raise RuntimeError(f"'{coordinate}' carries {combined[coordinate].size} levels instead of the configured "
                           f"{levels.size}; every three-dimensional field must be regridded before assembly")
    combined = combined.assign_coords(**{coordinate: ((coordinate,), levels, pressure_attrs)})
    if "lat" in combined.coords:
        lat_attrs = {"standard_name": "latitude", "units": "degrees_north", "axis": "Y", "long_name": "latitude"}
        combined = combined.assign_coords(lat=(("lat",), np.asarray(combined.lat.values, "float64"), lat_attrs))
    for field in combined.data_vars:
        if field == "climatology_bounds":
            continue
        variable = combined[field]
        order = ["time", coordinate, "lat"] if variable.ndim == 3 else ["time", "lat"]
        combined[field] = variable.transpose(*order)
    combined.attrs = product_attributes(model, config, start_year, end_year, names, coverage_bounds, grid, reference)
    return combined


def product_encoding(names, coordinate):
    """Compact, CF-safe encodings: float64 coordinates, float32 data, int16 counts."""
    encoding = {"time": {"dtype": "float64"},
                "climatology_bounds": {"dtype": "float64", "_FillValue": None, "zlib": True},
                coordinate: {"dtype": "float64", "zlib": True},
                "lat": {"dtype": "float64", "zlib": True}}
    for name in names:
        for statistic in cf.STATISTIC_ORDER:
            dtype = cf.STATISTIC_DTYPES[statistic]
            entry = {"dtype": dtype, "zlib": True, "complevel": 4}
            if dtype != "int16":
                entry["_FillValue"] = np.float32(np.nan)
            encoding[f"{name}_{statistic}"] = entry
    return encoding


def write_product(model, start_year, end_year, parts, coverage_bounds=()):
    """Write the one product per model atomically and replace it only when it validates."""
    config = load_config(model)
    grid = vertical_grid()
    names = sorted({part[key].attrs["source_variable"] for part in parts for key in part.data_vars})
    dataset = assemble_product(model, start_year, end_year, parts, coverage_bounds, names)
    dest = ROOT / config["paths"]["climatology"] / product_name(model, start_year, end_year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".nc.tmp")
    encoding = product_encoding(names, grid["coordinate"])
    try:
        dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        with xr.open_dataset(temporary, decode_cf=False) as written:
            cf.assert_product(written, names=names, grid=grid)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(dest)
    return dest


def build_climatology(model, start_year, end_year, variables=None):
    """One CF-1.13 NetCDF per model and period; returns the product path."""
    config = load_config(model)
    base = ROOT / config["paths"]["zonal"]
    names = [config["variables"].get(v, v) for v in variables] if variables else [p.name.removesuffix("_monthly_zonal.nc") for p in base.glob("*_monthly_zonal.nc")]
    if not names:
        raise FileNotFoundError(f"No zonal files in {base}")
    names = sorted(set(names))
    parts = []
    coverage_bounds = []
    grid = vertical_grid()
    for name in names:
        path = base / f"{name}_monthly_zonal.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path, use_cftime=True) as ds:
            coverage_bounds.extend((str(ds.time.values[0]), str(ds.time.values[-1])))
            source_attrs = dict(ds[name].attrs)
            field, regrid_provenance = on_common_grid(ds, name, start_year, end_year, grid,
                                                      config["coordinates"].get("level", ""))
            stats = monthly_climatology(field, start_year, end_year)
            parts.append(variable_statistics(name, stats, source_attrs, regrid_provenance,
                                             config.get("standard_names") or {}))
    return write_product(model, start_year, end_year, parts, coverage_bounds)

