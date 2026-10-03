"""Period-specific monthly statistics on the native grid of one model.

An individual JuMACS climatology keeps the model where it is: twelve calendar
months, the native latitude grid, and the native vertical levels. The shared
124 level application pressure grid and the fixed 5 degree latitude bands belong
to the combination stage, so nothing here interpolates vertically or horizontally.
What an individual product must do instead is publish the pressure belonging to
its native levels, so that a later stage can place them on any grid it wants.
"""
import datetime as dt
import warnings

import numpy as np
import xarray as xr

from . import cf
from .config import ROOT, load_config, reference_period, is_waccmx, model_slug, vertical_grid
from .vertical import PRESSURE_ATTRS, native_pressure, pressure_report, vertical_dimension
from .netcdf import open_cftime_dataset


class MissingPressureCoordinate(ValueError):
    """A three-dimensional field has no recoverable native pressure coordinate."""


LATITUDE_SNAP_TOLERANCE_DEGREES = 1e-5
GRID_TOLERANCE = 1e-4


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


def native_monthly_field(ds, name, start_year, end_year, native_coordinate=""):
    """One monthly zonal series kept on the native vertical grid of the model.

    An individual product is the climatology of a model *as the model produced it*:
    statistics are taken on its own levels and only later stages move fields onto
    the shared application grid. The pressure belonging to those native levels is
    returned with the field, because a hybrid level is not a fixed pressure and a
    published climatology has to say where its levels actually were.
    """
    inside = (ds.time.dt.year >= start_year) & (ds.time.dt.year <= end_year)
    field = ds[name].where(inside, drop=True)
    level = vertical_dimension(field)
    if level is None:
        return field, None, {"vertical_treatment": "two-dimensional field; no vertical dimension",
                             "on_application_pressure_grid": "false"}
    pressure = native_pressure(ds, field)
    if pressure is None:
        raise MissingPressureCoordinate(
            f"{name}: the zonal product has neither an air_pressure variable nor a pressure-valued "
            f"'{level}' coordinate, so the native levels of this field cannot be described")
    if "time" in pressure.dims:
        pressure = pressure.where(inside, drop=True)
    provenance = {"vertical_treatment": "climatological statistics on the native vertical levels of the model",
                  "on_application_pressure_grid": "false",
                  "native_level_dimension": level,
                  "native_level_count": int(field.sizes[level]),
                  "native_vertical_coordinate": native_coordinate or level}
    if "air_pressure" in ds and level in ds["air_pressure"].dims:
        provenance["native_pressure_units"] = str(ds["air_pressure"].attrs.get("units", "")) or "Pa"
    elif level in ds.coords:
        provenance["native_pressure_units"] = str(ds[level].attrs.get("units", "")) or "Pa"
    provenance["native_pressure_kind"] = "coordinate" if pressure.ndim == 1 else "profile"
    finite = pressure.where(np.isfinite(pressure) & (pressure > 0))
    if finite.size:
        provenance["native_pressure_min_pa"] = pressure_report(float(finite.min()))
        provenance["native_pressure_max_pa"] = pressure_report(float(finite.max()))
    return field, pressure, provenance


def level_pressure_signature(pressure, level):
    """The pressure of every native level, reduced to one number per level for recognition.

    A pressure coordinate is already one number per level; a hybrid profile is reduced to the median
    of its finite pressures. A grid that is really a fixed pressure level shows the same signature
    however the model describes it, while a grid that moves with latitude shows a profile that no
    pressure coordinate of the same length will claim.
    """
    values = np.asarray(pressure.values, "float64")
    if values.ndim == 1:
        return values
    axis = pressure.dims.index(level)
    columns = np.moveaxis(values, axis, 0).reshape(values.shape[axis], -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(columns, axis=1)


def _same_level_grid(known, signature):
    """Whether two level signatures describe the same native grid, compared level by level in Pa."""
    if np.asarray(known).shape != np.asarray(signature).shape:
        return False
    return bool(np.allclose(known, signature, rtol=GRID_TOLERANCE, atol=0.0, equal_nan=True))


def level_grid_name(grids, level, pressure):
    """The product name of one native vertical dimension, so distinct grids get distinct names.

    GEOSCCM keeps its chemistry on a 39 level pressure grid and its dynamics on a 72 level hybrid
    grid and calls both dimensions 'lev'. Merged under one name the dimension becomes the union of
    two unrelated grids and every field of the other grid turns into rows of NaN, so the second grid
    of a source name is published as 'lev_2' and so on. Nothing is resampled: each grid keeps its own
    levels and its own pressure description, and the source name survives in the provenance.

    Grids are recognised by the pressure of their levels and by the way that pressure is expressed, never
    by their length: two grids of the same length that sit on different pressures are two grids, and a grid
    whose levels are given as a pressure coordinate is never merged with a grid whose levels are given by a
    (time, level, lat) pressure profile, even when both name the same pressures. A pressure-valued dimension
    coordinate applies to every variable that shares its dimension under CF, and no attribute can withdraw
    it again, so a fixed coordinate next to a moving profile would be read as the pressure of the profiled
    variable too. The level count of a grid is recorded with its name, so the naming of a grid does not
    depend on which variable of the model is read first.
    """
    signature = level_pressure_signature(pressure, level)
    kind = pressure_kind(pressure)
    known_grids = grids.setdefault(level, [])
    for name, known in known_grids:
        if known["kind"] == kind and _same_level_grid(known["signature"], signature):
            return name
    name = level if not known_grids else f"{level}_{len(known_grids) + 1}"
    known_grids.append((name, {"kind": kind, "level_count": signature.size, "signature": signature}))
    return name


def pressure_kind(pressure):
    """How one variable expresses the pressure of its levels: a coordinate, or a profile over them."""
    return "coordinate" if np.asarray(pressure).ndim == 1 else "profile"


def register_pressure(pressures, level, pressure, name, coverage, units="", source_level=""):
    """Record how the native levels of one product dimension are expressed as pressure.

    A pressure-valued coordinate is published as the level coordinate; a time- and latitude-dependent
    hybrid pressure is published as a climatology field. Which of the two a grid is is settled by
    level_grid_name, which never puts a coordinate-described grid and a profile-described grid on one
    dimension, so one entry describes exactly one representation and a product dimension can never carry
    a fixed coordinate that some of its variables do not share. Two variables of one grid must agree about
    the pressure of those levels over the same period; a silent pick between disagreeing pressures would
    hide a broken grid, so a disagreement raises.
    """
    entry = pressures.get(level)
    if entry is None:
        entry = pressures[level] = {"kind": "coordinate" if pressure.ndim == 1 else "hybrid",
                                    "pressure": pressure, "variables": {name},
                                    "source_variable": name, "coverage": coverage, "units": units,
                                    "source_level": source_level or level}
        return
    if entry["kind"] != ("coordinate" if pressure.ndim == 1 else "hybrid"):
        raise RuntimeError(
            f"variable '{name}' describes the pressure of the native '{entry['source_level']}' levels as a "
            f"{'profile over the levels' if pressure.ndim > 1 else 'level coordinate'}, but '{entry['source_variable']}' "
            f"describes the same levels the other way; a product dimension carries one representation of its "
            f"pressure, so the two variables are published on dimensions of their own")
    if entry["coverage"] == coverage and entry["pressure"].sizes == pressure.sizes:
        _check_pressure_agreement(level, entry["pressure"], entry["source_variable"], pressure, name)
    entry["variables"].add(name)
    entry["units"] = entry["units"] or units


def _check_pressure_agreement(level, known, known_variable, pressure, name):
    first = np.asarray(known.values, "float64")
    second = np.asarray(pressure.values, "float64")
    both = np.isfinite(first) & np.isfinite(second)
    if both.any() and not np.allclose(first[both], second[both], rtol=GRID_TOLERANCE, atol=0.0):
        raise RuntimeError(
            f"variables '{known_variable}' and '{name}' disagree about the pressure of the native "
            f"'{level}' levels over the same period (largest relative difference "
            f"{float(np.max(np.abs(first[both] - second[both]) / first[both])):g}); the monthly zonal "
            f"intermediates of this model must describe one vertical grid")


def pressure_climatology(pressure):
    """The climatological pressure of native levels: a monthly mean over the years."""
    if "time" not in pressure.dims:
        return pressure
    return pressure.groupby("time.month").mean("time", skipna=True)


def pressure_field_name(pressures, level):
    return "air_pressure" if len(pressures) == 1 else f"air_pressure_{level}"


def variable_statistics(name, stats, source_attrs, vertical_provenance, standard_names=None):
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
            for key, value in vertical_provenance.items():
                attrs[key] = value
        field.attrs = attrs
        renamed[f"{name}_{statistic}"] = field
    return renamed[[f"{name}_{statistic}" for statistic in cf.STATISTIC_ORDER]]


def product_attributes(model, config, start_year, end_year, names, coverage_bounds, vertical, reference):
    """Model-wide provenance, recorded once per product instead of per variable."""
    model_info = config["model"]
    application = vertical_grid()
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
        "product": "monthly climatology of zonal means on the native vertical and horizontal grid of one model",
        "horizontal_grid": "zonal mean over all longitudes; latitude is the only horizontal dimension and it is "
                           "the native latitude grid of the model",
        "statistics": " ".join(cf.STATISTIC_ORDER),
        "variable_naming": ("one field per source variable and statistic, named <variable>_<statistic>: "
                            "<variable>_mean, <variable>_sigma, <variable>_minimum, <variable>_maximum, "
                            "<variable>_n_years"),
        "variable_count": len(names),
        "bias_correction": "none", "trend_correction": "none", "model_combination": "none",
        "vertical_coordinate": vertical["description"],
        "vertical_level_count": vertical["level_count"],
        "vertical_level_counts": vertical["level_counts"],
        "vertical_interpolation": "none; every statistic is computed on the native vertical levels of the model",
        "vertical_extrapolation": "none; a level the model does not reach simply does not exist here",
        "application_pressure_grid": (f"{len(application['levels'])} levels from {application['levels'][0]:g} to "
                                      f"{application['levels'][-1]:g} Pa; this shared grid is applied when models "
                                      f"are combined, not in this product"),
        "latitude_count": vertical["latitude_count"],
        "application_latitude_bands": "the fixed 5 degree area-weighted latitude bands are applied when models "
                                      "are combined, not in this product",
        "missing_value_policy": "NaN where no source value exists; n_years counts finite contributions",
        "comment": ("CF-1.13 climatological time axis: 12 monthly cells summarising every year in the period, "
                    "with across-year statistics in cell_methods and month intervals in climatology_bounds; "
                    "hybrid level heights are published as a climatological air_pressure field so that the "
                    "native levels stay resolvable as pressure"),
        "history": (f"created {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}: monthly zonal values "
                    "grouped by calendar month on the native vertical grid of the model, written as one "
                    f"{cf.CONVENTIONS} NetCDF per model"),
    }


def _part_variable(part):
    for key in part.data_vars:
        return str(part[key].attrs.get("source_variable", ""))
    return "<unnamed>"


def _check_latitude(model, variable, anchor_variable, anchor, latitude):
    if latitude.ndim != 1:
        raise RuntimeError(f"latitude grid for variable '{variable}' of {model} is {latitude.ndim}-dimensional; "
                           f"the climatology product needs one 1-D latitude coordinate per field")
    if latitude.size != anchor.size:
        raise RuntimeError(f"latitude grid for variable '{variable}' of {model} has {latitude.size} latitude points, "
                           f"the canonical grid taken from '{anchor_variable}' has {anchor.size}; distinct latitude "
                           f"grids are neither unioned nor interpolated")
    if not np.array_equal(np.argsort(latitude, kind="stable"), np.argsort(anchor, kind="stable")):
        raise RuntimeError(f"latitude grid for variable '{variable}' of {model} is ordered differently from the "
                           f"canonical grid taken from '{anchor_variable}' ({latitude.size} latitude points); "
                           f"fields are never reordered during assembly")
    if not np.allclose(latitude, anchor, rtol=0.0, atol=LATITUDE_SNAP_TOLERANCE_DEGREES):
        difference = float(np.max(np.abs(latitude - anchor)))
        raise RuntimeError(
            f"latitude grid for variable '{variable}' of {model} differs from the canonical model grid taken from "
            f"'{anchor_variable}' beyond the {LATITUDE_SNAP_TOLERANCE_DEGREES} degree tolerance: {latitude.size} "
            f"latitude points, maximum absolute coordinate difference {difference} degrees")


def normalize_latitude(parts, model=""):
    """Pin latitude grids that differ only by roundoff onto the canonical grid of the product.

    Variable groups of one model can carry the same physical latitude axis with harmless
    float64 roundoff of order 1e-6 degrees. An outer merge reads those as two different
    axes and publishes their union, which leaves most latitude rows NaN and makes the
    field look unusable. Grids that agree to LATITUDE_SNAP_TOLERANCE_DEGREES are therefore
    snapped to the coordinate of the first assembled part; the values of the data are
    never interpolated, reordered, or rounded. Grids that differ by more than that, or
    that disagree in size or order, are a real mismatch and raise instead.
    """
    anchor = None
    anchor_variable = ""
    normalized = []
    for part in parts:
        if "lat" not in part.coords or "lat" not in part.dims:
            normalized.append(part)
            continue
        latitude = np.asarray(part["lat"].values, "float64")
        variable = _part_variable(part)
        if anchor is None:
            anchor, anchor_variable = latitude, variable
            normalized.append(part)
            continue
        _check_latitude(model, variable, anchor_variable, anchor, latitude)
        normalized.append(part.assign_coords(lat=(("lat",), anchor, dict(part["lat"].attrs))))
    return normalized


def _grid_order(name):
    """The order the numbered dimensions of one source name were published in."""
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else 0, name)


def _shared_source_note(pressures, level):
    """Warn a reader of one coordinate that a sibling grid carries the same source name but other pressures.

    A dimension coordinate applies to every variable that shares its dimension, so the coordinate of a grid
    has to say out loud that the variables of a sibling grid of the same source name are not covered by it.
    """
    source = str(pressures.get(level, {}).get("source_level", level))
    others = sorted((other for other, entry in pressures.items()
                     if other != level and str(entry.get("source_level", other)) == source), key=_grid_order)
    if not others:
        return ""
    named = [other for other in sorted(pressures, key=_grid_order)
             if str(pressures[other].get("source_level", other)) == source]
    quoted = " and ".join("'" + name + "'" for name in named)
    siblings = " and ".join("'" + name + "'" for name in others)
    return (f"; the source name '{source}' is published as {quoted} here, one dimension per grid, so the "
            f"variables of {siblings} take their pressure from {'that dimension' if len(others) == 1 else 'those dimensions'}"
            f" and not from these levels")


def _hybrid_pressure_attrs(level, name, start_year, end_year, source_variable, source_level=""):
    comment = (f"climatological pressure of the native vertical levels, published per month and latitude "
               f"because a {level} level is not a fixed pressure; the vertical coordinate '{level}' is "
               f"the native level index of the model and carries no pressure of its own")
    if source_level and source_level != level:
        comment += (f"; the source names both of its vertical grids '{source_level}', so this one is published "
                    f"as '{level}'")
    return dict(PRESSURE_ATTRS, positive="down", axis="Z",
                long_name=f"air pressure of the native '{level}' levels: monthly mean {start_year}-{end_year}",
                cell_methods="time: mean", source_variable=source_variable, comment=comment)


def _level_index_attrs(level, source_level, pressure_field, shared=""):
    """Describe a native level coordinate that is an index, not a pressure.

    The hybrid formula of the source cannot be evaluated in this product, because its terms
    (a, b, p0 and the surface pressure) are not carried over as they stand, so the coordinate says
    what it is and points at the published pressure of its levels instead of naming a formula.
    """
    attrs = {"units": "1", "positive": "down", "axis": "Z",
             "long_name": f"native '{level}' level index of the model",
             "comment": (f"the index of a native vertical level of the model, not a pressure; the pressure of these "
                         f"levels is the field '{pressure_field}' (time, {level}, lat) in Pa, and the hybrid formula "
                         f"of the source is not reproducible here because its terms are not part of this product") + shared}
    if source_level and source_level != level:
        attrs["source_level_coordinate"] = source_level
    return attrs


def _pressure_coordinate_attrs(level, native_units, source_level="", shared=""):
    comment = ("native pressure levels of the source, in Pa; the shared application pressure grid is "
               "applied when models are combined, not in this product")
    if source_level and source_level != level:
        comment += (f"; the source names both of its vertical grids '{source_level}', so this one is published "
                    f"as '{level}'")
    attrs = dict(PRESSURE_ATTRS, positive="down", axis="Z",
                 long_name=f"air pressure on the native '{level}' pressure levels of the model",
                 comment=comment + shared)
    if native_units:
        attrs["source_coordinate_units"] = native_units
    return attrs


def _entry_kind(entry):
    """The representation one registry entry describes, also for a registry built by hand."""
    kind = entry.get("kind")
    if kind:
        return str(kind)
    return "coordinate" if np.asarray(entry["pressure"].values).ndim == 1 else "hybrid"


def hybrid_pressure_field(entry):
    """The climatological pressure field of a canonical grid, when a hybrid profile describes it.

    A grid that is a fixed pressure is described by its coordinate alone and publishes no field; a grid
    whose levels move is described by a field over time and latitude, and by nothing else.
    """
    return entry["pressure"] if _entry_kind(entry) == "hybrid" else None


def _entry_level_count(entry, level):
    pressure = entry["pressure"]
    if level in pressure.sizes:
        return int(pressure.sizes[level])
    if pressure.ndim == 1:
        return int(pressure.size)
    axes = [dim for dim in pressure.dims if dim not in {"time", "month", "lat", cf.BOUNDS_DIMENSION}]
    return int(pressure.sizes[axes[0]]) if axes else int(pressure.size)


def level_coordinate_description(entry, level, published_field="", shared=""):
    """The coordinate a product level dimension must carry, values and attributes together.

    A canonical grid that is a fixed pressure is published as a pressure-valued coordinate; a grid whose
    levels move is published as the native level index, because its pressure belongs to the published
    pressure field and not to the index. An index is published with units of one and without a pressure
    standard_name, so no reader can take it for the pressure of the levels it numbers. Either way one grid
    gets exactly one coordinate in a product, and a grid that shares its source name with another grid says
    in its own attributes which dimension describes the pressure of the other one's variables.
    """
    if _entry_kind(entry) == "coordinate":
        return (np.asarray(entry["pressure"].values, "float64"),
                _pressure_coordinate_attrs(level, entry.get("units", ""), entry.get("source_level", level), shared))
    count = _entry_level_count(entry, level)
    return (np.arange(count, dtype="float64"),
            _level_index_attrs(level, entry.get("source_level", level), published_field, shared))


def pin_level_coordinates(parts, pressures, published, model=""):
    """Write the canonical coordinate of every native grid onto the parts that carry it.

    Parts of one grid can still arrive with different coordinates under one name: one carries the pressure
    of the levels, another only a bare level index, and their numbers are unrelated. Merged they become
    their union, which turns most of both fields into rows of missing values, so the coordinate of a grid is
    settled from the registry before the merge and written on every part of it. Only the coordinate is
    written, never a value of a field, so every level keeps the value the model gave it.
    """
    if not pressures:
        return list(parts)
    pinned = []
    for part in parts:
        updates, variable = {}, _part_variable(part)
        for level, entry in sorted(pressures.items()):
            if level not in part.dims:
                continue
            values, attrs = level_coordinate_description(entry, level, published.get(level, ""),
                                                         _shared_source_note(pressures, level))
            if part.sizes[level] != values.size:
                raise RuntimeError(
                    f"variable '{variable}' of {model} has {part.sizes[level]} levels on the native "
                    f"'{entry.get('source_level', level)}' grid, but the product dimension '{level}' describes "
                    f"the {values.size} levels taken from '{entry.get('source_variable', '')}'; a native grid "
                    f"that differs from another is published as a dimension of its own")
            updates[level] = ((level,), values, attrs)
        pinned.append(part.assign_coords(**updates) if updates else part)
    return pinned


def _vertical_summary(pressures, published, combined, vertical_dims):
    descriptions, counts = [], {}
    for level in vertical_dims:
        counts[level] = int(combined.sizes[level])
        entry = pressures.get(level, {})
        source_level = str(entry.get("source_level", level))
        renamed = "" if source_level == level else f", named '{source_level}' in the source"
        shared = _shared_source_note(pressures, level)
        if level in published:
            descriptions.append(f"'{level}' native level index of the model{renamed}; its pressure is "
                                f"'{published[level]}' (time, {level}, lat) in Pa{shared}")
        else:
            descriptions.append(f"'{level}' native pressure levels in Pa (standard_name air_pressure)"
                                f"{renamed}{shared}")
    return {"description": "; ".join(descriptions) or
                      "no three-dimensional fields; two-dimensional fields carry no vertical dimension",
            "level_count": max(counts.values(), default=0),
            "level_counts": "; ".join(f"{level}={count}" for level, count in sorted(counts.items())) or "none",
            "latitude_count": int(combined.sizes.get("lat", 0))}


def assemble_product(model, start_year, end_year, parts, coverage_bounds=(), names=None, pressures=None):
    """Merge per-variable statistics into the single CF-1.13 product dataset."""
    config = load_config(model)
    reference = reference_period()["reference_period"]
    pressures = pressures or {}
    names = sorted(names or {part[key].attrs["source_variable"] for part in parts for key in part.data_vars})
    months = [part.rename({"month": "time"}) for part in normalize_latitude(parts, model)]
    published = {level: pressure_field_name(pressures, level) for level, entry in sorted(pressures.items())
                 if hybrid_pressure_field(entry) is not None}
    months = pin_level_coordinates(months, pressures, published, model)
    published_parts = []
    for level, name in sorted(published.items()):
        field = hybrid_pressure_field(pressures[level]).rename({"month": "time"}).rename(name)
        values, attrs = level_coordinate_description(pressures[level], level, name,
                                                     _shared_source_note(pressures, level))
        field = field.assign_coords(**{level: ((level,), values, attrs)})
        field.attrs = _hybrid_pressure_attrs(level, name, start_year, end_year,
                                             pressures[level]["source_variable"],
                                             pressures[level].get("source_level", level))
        published_parts.append(field)
    combined = xr.merge(months + published_parts, join="outer", compat="override")
    vertical_dims = sorted({vertical_dimension(variable) for variable in combined.data_vars.values()
                            if variable.ndim == 3} - {None})
    stray = sorted(dim for dim in combined.dims
                   if dim not in {"time", "lat", cf.BOUNDS_DIMENSION} | set(vertical_dims))
    if stray:
        raise RuntimeError(f"Climatology product kept unexpected dimensions {stray}; an individual product holds "
                           f"only time, latitude, and the native vertical dimensions {vertical_dims or '()'} of "
                           f"the model")
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
    for level, entry in sorted(pressures.items()):
        count = _entry_level_count(entry, level)
        if level not in combined.coords or combined[level].size != count:
            raise RuntimeError(f"'{level}' is not the coordinate its fields were built on; the "
                               f"monthly zonal intermediates must be regenerated")
    for field, variable in combined.data_vars.items():
        if field in published.values() or variable.ndim != 3:
            continue
        level = vertical_dimension(variable)
        if level not in pressures:
            continue
        entry = pressures[level]
        named = published.get(level, "")
        if _entry_kind(entry) == "hybrid" and named:
            combined[field].attrs["pressure_field"] = named
        else:
            combined[field].attrs["pressure_coordinate"] = level
    if "lat" in combined.coords:
        lat_attrs = {"standard_name": "latitude", "units": "degrees_north", "axis": "Y", "long_name": "latitude"}
        combined = combined.assign_coords(lat=(("lat",), np.asarray(combined.lat.values, "float64"), lat_attrs))
    for field in combined.data_vars:
        if field == "climatology_bounds":
            continue
        variable = combined[field]
        level = vertical_dimension(variable)
        order = ["time", level, "lat"] if level is not None else ["time", "lat"]
        combined[field] = variable.transpose(*order)
    combined.attrs = product_attributes(model, config, start_year, end_year, names, coverage_bounds,
                                        _vertical_summary(pressures, published, combined, vertical_dims), reference)
    return combined


def product_encoding(names, vertical_dims=(), pressure_fields=()):
    """Compact, CF-safe encodings: float64 coordinates, float32 data, int16 counts."""
    encoding = {"time": {"dtype": "float64"},
                "climatology_bounds": {"dtype": "float64", "_FillValue": None, "zlib": True},
                "lat": {"dtype": "float64", "zlib": True}}
    for level in vertical_dims:
        encoding[level] = {"dtype": "float64", "zlib": True}
    for name in pressure_fields:
        encoding[name] = {"dtype": "float64", "zlib": True, "complevel": 4, "_FillValue": np.float64(np.nan)}
    for name in names:
        for statistic in cf.STATISTIC_ORDER:
            dtype = cf.STATISTIC_DTYPES[statistic]
            entry = {"dtype": dtype, "zlib": True, "complevel": 4}
            if dtype != "int16":
                entry["_FillValue"] = np.float32(np.nan)
            encoding[f"{name}_{statistic}"] = entry
    return encoding


def write_product(model, start_year, end_year, parts, coverage_bounds=(), pressures=None):
    """Write the one product per model atomically and replace it only when it validates."""
    config = load_config(model)
    names = sorted({part[key].attrs["source_variable"] for part in parts for key in part.data_vars})
    dataset = assemble_product(model, start_year, end_year, parts, coverage_bounds, names, pressures)
    dest = ROOT / config["paths"]["climatology"] / product_name(model, start_year, end_year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".nc.tmp")
    pressure_fields = [name for name in dataset.data_vars if name.startswith("air_pressure")]
    vertical_dims = [level for level in pressures or {} if level in dataset.coords]
    encoding = product_encoding(names, vertical_dims, pressure_fields)
    try:
        dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        with xr.open_dataset(temporary, decode_cf=False) as written:
            cf.assert_product(written, names=names, grid=vertical_grid())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(dest)
    return dest


def build_climatology(model, start_year, end_year, variables=None):
    """One CF-1.13 NetCDF per model and period on the native grid; returns the product path."""
    config = load_config(model)
    base = ROOT / config["paths"]["zonal"]
    names = [config["variables"].get(v, v) for v in variables] if variables else [p.name.removesuffix("_monthly_zonal.nc") for p in base.glob("*_monthly_zonal.nc")]
    if not names:
        raise FileNotFoundError(f"No zonal files in {base}")
    names = sorted(set(names))
    parts = []
    coverage_bounds = []
    pressures = {}
    grids = {}
    for name in names:
        path = base / f"{name}_monthly_zonal.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with open_cftime_dataset(path) as ds:
            coverage_bounds.extend((str(ds.time.values[0]), str(ds.time.values[-1])))
            source_attrs = dict(ds[name].attrs)
            field, pressure, provenance = native_monthly_field(ds, name, start_year, end_year,
                                                              config["coordinates"].get("level", ""))
            stats = monthly_climatology(field, start_year, end_year)
            level = ""
            if pressure is not None:
                source_level = provenance["native_level_dimension"]
                level_pressure = pressure_climatology(pressure)
                level = level_grid_name(grids, source_level, level_pressure)
                if level != source_level:
                    stats = stats.rename({source_level: level})
                    level_pressure = level_pressure.rename({source_level: level})
                provenance["native_source_level_dimension"] = source_level
                provenance["native_level_dimension"] = level
            parts.append(variable_statistics(name, stats, source_attrs, provenance,
                                             config.get("standard_names") or {}))
            if pressure is not None:
                register_pressure(pressures, level, level_pressure,
                                  name, (int(ds.time.dt.year.min()), int(ds.time.dt.year.max())),
                                  units=provenance.get("native_pressure_units", ""), source_level=source_level)
    return write_product(model, start_year, end_year, parts, coverage_bounds, pressures)

