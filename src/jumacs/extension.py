"""Extending one application-grid climatology upward with one whole-atmosphere donor.

Stage 1 moves a model's published climatology onto the shared pressure grid and the fixed
latitude bands. That grid reaches to 0.002 Pa, well above where a CCMI model has chemistry,
so a remapped CCMI field is simply missing in the upper part of it. This module fills
exactly that gap and nothing else: where the donor (WACCM-X, the only donor in this stage)
reaches higher than the base model and the two overlap, the values are blended over a
finite number of overlapping levels and the donor continues the column upward.

Three choices give the blend its shape.

The decision is made per month and per latitude column, never on a domain average. The
missing-value pattern of a remapped field varies with month and with latitude - terrain
erases the lowest levels over mountains, chemistry runs out at different heights in
different seasons - so a single global transition would blend where a column has nothing
to blend and refuse where two profiles agree perfectly.

The blend is linear in log pressure, which is the variable the vertical grid is uniform in
and the variable the interpolation of Stage 1 was linear in. Across twelve application
levels the pressure changes by a factor of a hundred, so a blend linear in pressure itself
would spend almost all of its weight on the bottom of the transition.

The donor is allowed to continue the atmosphere, not to repair it. A donor value is taken
where the base is missing only above the transition of a column that qualified, so a hole
in the middle of the base profile stays a hole: it says the base model did not resolve
that value, and a donor that happens to have one there does not contradict that. The
statistics are blended each on its own, which is a merge of two climatologies rather than
a reconstruction of a sample distribution - the band mean of a standard deviation is not a
standard deviation, and the attributes say so.
"""
import numpy as np
import xarray as xr

from . import cf, config, remap, vertical

EXTENSION_STATISTICS = remap.REMAP_STATISTICS

MINIMUM_OVERLAP_LEVELS = vertical.MINIMUM_OVERLAP_LEVELS

EXTENDED_DONOR_FALLBACK = "WACCM-X"

EXTENSION_METHOD = ("linear blend in log pressure over the uppermost overlapping application levels; below the "
                    "transition the base model stands alone and above it the donor does")
EXTENSION_WEIGHTING = "linear in log pressure: 0 at the lowest transition level, 1 at the highest"
EXTENSION_ELIGIBILITY = ("a donor value where the base is missing is taken only above the transition of a column "
                         "whose donor reaches above the base top; a missing value inside the base profile stays "
                         "missing and is not repaired by the donor")
COORDINATE_TOLERANCE = 1e-12


def product_description(donor):
    """How an extended dataset describes itself: one model below, the donor above."""
    return f"single-model climatology on the JuMACS application grid with {donor} upper-atmosphere extension"


class ExtensionProblem(ValueError):
    """Two application-grid datasets that cannot be blended as they stand."""


def transition_level_count(variable, requested=None):
    """How many overlapping levels a variable transitions over: the override, else the configured default.

    Widths are counted in application levels rather than kilometres, so they keep their
    meaning if the pressure grid changes. The default and the per-variable overrides both
    come from the configuration, which is the only place the number is written.
    """
    if requested is not None:
        count = int(requested)
        if count < MINIMUM_OVERLAP_LEVELS:
            raise ExtensionProblem(f"a transition of {requested} levels cannot be blended; at least "
                                   f"{MINIMUM_OVERLAP_LEVELS} overlapping levels are needed")
        return count
    settings = config.extension_settings()
    overrides = settings["transition_levels_by_variable"]
    if variable in overrides:
        return overrides[variable]
    lowered = {str(key).lower(): value for key, value in overrides.items()}
    if str(variable).lower() in lowered:
        return lowered[str(variable).lower()]
    return int(settings["transition_levels"])


def blend_weights(levels, first, last):
    """The donor weight of every level of a column, for a transition spanning ``first`` to ``last``.

    The levels run from high pressure to low, so ``first`` is the bottom of the transition
    and ``last`` its top: the weight is 0 at the bottom, 1 at the top, and linear in
    ``log(pressure)`` between them, clipped outside. One interval only is refused, because
    a transition has to have two ends to be linear across.
    """
    grid = np.asarray(levels, dtype="float64")
    if not 0 <= first < last < grid.size:
        raise ExtensionProblem(f"a transition needs two distinct levels, found indexes {first} and {last} of "
                               f"{grid.size} levels")
    bottom, top = float(grid[first]), float(grid[last])
    if not bottom > top > 0.0:
        raise ExtensionProblem(f"a transition must run from a higher to a lower pressure, found {bottom} Pa and "
                               f"{top} Pa")
    span = np.log(bottom) - np.log(top)
    return np.clip((np.log(bottom) - np.log(grid)) / span, 0.0, 1.0)


def extension_outcome(base, donor, levels, transition_levels):
    """What one column allows: whether it extends, over which levels, and why not when it does not.

    The top of a profile is the lowest-pressure level that holds a finite value, which is
    the same reading of "usable" the coverage matrix uses column by column. The transition
    takes the uppermost overlapping levels, because the point of the blend is to arrive at
    the donor value where the base has run out.
    """
    grid = np.asarray(levels, dtype="float64")
    base_column = np.asarray(base, dtype="float64")
    donor_column = np.asarray(donor, dtype="float64")
    if base_column.shape != donor_column.shape or base_column.shape != grid.shape:
        raise ExtensionProblem("a column and its donor and the pressure levels must have the same shape, found "
                               f"{base_column.shape}, {donor_column.shape} and {grid.shape}")
    count = int(transition_levels)
    if count < MINIMUM_OVERLAP_LEVELS:
        raise ExtensionProblem(f"a transition of {transition_levels} levels cannot be blended; at least "
                               f"{MINIMUM_OVERLAP_LEVELS} overlapping levels are needed")
    overlap = np.flatnonzero(np.isfinite(base_column) & np.isfinite(donor_column))
    if overlap.size < MINIMUM_OVERLAP_LEVELS:
        return {"extended": False, "reason": "no_overlap", "overlap_levels": int(overlap.size),
                "transition_levels_used": 0, "window": None,
                "base_top_pressure_pa": None, "donor_top_pressure_pa": None}
    base_top = float(grid[np.isfinite(base_column)].min())
    donor_top = float(grid[np.isfinite(donor_column)].min())
    if not donor_top < base_top:
        return {"extended": False, "reason": "donor_not_higher", "overlap_levels": int(overlap.size),
                "transition_levels_used": 0, "window": None,
                "base_top_pressure_pa": base_top, "donor_top_pressure_pa": donor_top}
    selected = overlap[-count:]
    window = (int(selected[0]), int(selected[-1]))
    return {"extended": True, "reason": "extended", "overlap_levels": int(overlap.size),
            "transition_levels_used": int(selected.size), "window": window,
            "base_top_pressure_pa": base_top, "donor_top_pressure_pa": donor_top}


def extend_column(base, donor, levels, transition_levels):
    """One month and one latitude: the base profile with the donor continued above its transition.

    Below the transition the base stands alone, internal gaps included. Inside the window a
    finite value on one side is used where the other side is missing. Above the window the
    donor is all there is, so a finite donor value is taken and a missing one stays missing.
    """
    values = np.asarray(base, dtype="float64").copy()
    donor_values = np.asarray(donor, dtype="float64")
    outcome = extension_outcome(values, donor_values, levels, transition_levels)
    if not outcome["extended"]:
        return values, outcome
    first, last = outcome["window"]
    weights = blend_weights(levels, first, last)
    for index in range(first, last + 1):
        base_value, donor_value = values[index], donor_values[index]
        if np.isfinite(base_value) and np.isfinite(donor_value):
            values[index] = (1.0 - weights[index]) * base_value + weights[index] * donor_value
        elif np.isfinite(donor_value):
            values[index] = donor_value
        elif not np.isfinite(base_value):
            values[index] = np.nan
    above = slice(last + 1, values.size)
    values[above] = np.where(np.isfinite(donor_values[above]), donor_values[above], np.nan)
    return values, outcome


def _is_vertical(field):
    """Whether a field sits on (time, pressure, lat) and so has something to extend."""
    return list(field.dims) == ["time", "pressure", "lat"]


def _field_dimensions(field, name):
    """The (time, pressure, lat) shape a three-dimensional field must have to be blended."""
    if field.ndim != 3:
        raise ExtensionProblem(f"{name} must be three-dimensional (time, pressure, lat), found {list(field.dims)}")
    if list(field.dims) != ["time", field.dims[1], "lat"]:
        raise ExtensionProblem(f"{name} must be ordered (time, pressure, lat), found {list(field.dims)}")


def _same_coordinate(base, donor, name):
    """Whether two coordinates hold the same numbers, checked before any arithmetic.

    xarray would happily align mismatched coordinates and quietly interpolate or subset
    the result; here a mismatch is an error, because the grid is what makes the two
    datasets blendable and Stage 1 is the step responsible for producing it.
    """
    if name not in base.coords or name not in donor.coords:
        raise ExtensionProblem(f"{name} must be a coordinate of both datasets")
    left = np.asarray(base.coords[name].values, dtype="float64")
    right = np.asarray(donor.coords[name].values, dtype="float64")
    if left.shape != right.shape:
        raise ExtensionProblem(f"{name} differs between the two datasets: {left.shape[0]} values against "
                               f"{right.shape[0]}")
    if not np.allclose(left, right, rtol=COORDINATE_TOLERANCE, atol=0.0):
        raise ExtensionProblem(f"{name} differs between the two datasets; both must already sit on the same "
                               "application grid, and neither is aligned or interpolated here")
    return left


def _compatible_units(base, donor, name):
    """Units of a pair, compared in one folded spelling; a mismatch is refused, never converted."""
    base_units, donor_units = str(base.attrs.get("units", "")), str(donor.attrs.get("units", ""))
    if not base_units or not donor_units:
        raise ExtensionProblem(f"{name} must carry its units on both datasets, found {base_units!r} and "
                               f"{donor_units!r}")
    if cf._units_key(base_units) != cf._units_key(donor_units):
        raise ExtensionProblem(f"{name} is in {base_units!r} in the base dataset and {donor_units!r} in the donor; "
                               "values are not converted by this step")
    return base_units


def extend_field(base_field, donor_field, levels=None, transition_levels=None):
    """One statistic field of the base with the donor blended into its upper part, column by column.

    The grid of the two fields is checked before anything is read: same dimensions, same
    pressure levels, same latitude bands, same months, same units. The arithmetic is done
    on positional arrays, so no xarray alignment can slip in between.
    """
    name = str(getattr(base_field, "name", "") or "field")
    grid = np.asarray(config.vertical_grid()["levels"] if levels is None else levels, dtype="float64")
    _field_dimensions(base_field, name)
    if list(donor_field.dims) != list(base_field.dims):
        raise ExtensionProblem(f"{name} differs between the two datasets: {list(base_field.dims)} against "
                               f"{list(donor_field.dims)}")
    if base_field.shape != donor_field.shape:
        raise ExtensionProblem(f"{name} differs in size between the two datasets: {base_field.shape} against "
                               f"{donor_field.shape}")
    _compatible_units(base_field, donor_field, name)
    level = base_field.dims[1]
    if level != "pressure" or not np.array_equal(np.asarray(base_field.coords[level].values, float), grid):
        raise ExtensionProblem(f"{name} must sit on the {grid.size} levels of the application pressure grid")
    for coordinate in ("time", "lat"):
        _same_coordinate(base_field, donor_field, coordinate)
    count = transition_level_count(base_field.attrs.get("source_variable", name), transition_levels)
    values = np.asarray(base_field, dtype="float64")
    donors = np.asarray(donor_field, dtype="float64")
    result = np.empty_like(values)
    outcomes = []
    for first in range(values.shape[0]):
        for last in range(values.shape[2]):
            result[first, :, last], outcome = extend_column(values[first, :, last], donors[first, :, last],
                                                            grid, count)
            outcomes.append(outcome)
    return result, _field_diagnostics(name, base_field, count, outcomes)


def _field_diagnostics(name, base_field, count, outcomes):
    """A short account of what the blend did to one field, for a smoke test to read."""
    extended = [outcome for outcome in outcomes if outcome["extended"]]
    used = [outcome["transition_levels_used"] for outcome in extended]
    base_tops = [outcome["base_top_pressure_pa"] for outcome in extended]
    donor_tops = [outcome["donor_top_pressure_pa"] for outcome in extended]
    rejected = {}
    for outcome in outcomes:
        if not outcome["extended"]:
            rejected[outcome["reason"]] = rejected.get(outcome["reason"], 0) + 1

    def span(values):
        return None if not values else [float(min(values)), float(max(values))]

    reason = ("extended" if not rejected else "; ".join(
        f"{why} in {tally} of {len(outcomes)} columns" for why, tally in sorted(rejected.items())))
    return {"field": name, "variable": str(base_field.attrs.get("source_variable", name)),
            "reason": reason,
            "extension_applied": bool(extended), "columns_total": len(outcomes), "columns_extended": len(extended),
            "columns_rejected": len(outcomes) - len(extended),
            "columns_no_overlap": sum(1 for outcome in outcomes if outcome["reason"] == "no_overlap"),
            "columns_donor_not_higher": sum(1 for outcome in outcomes if outcome["reason"] == "donor_not_higher"),
            "transition_levels_requested": count,
            "transition_levels_min_used": min(used) if used else None,
            "transition_levels_max_used": max(used) if used else None,
            "base_top_pressure_range": span(base_tops), "donor_top_pressure_range": span(donor_tops)}


def _extended_attributes(attrs, diagnostics):
    """Attributes of a field the blend touched; identity and native provenance stay as they were."""
    attrs["extension_applied"] = "true" if diagnostics["extension_applied"] else "false"
    attrs["extension_method"] = EXTENSION_METHOD
    attrs["extension_weighting"] = EXTENSION_WEIGHTING
    attrs["extension_transition_levels"] = int(diagnostics["transition_levels_requested"])
    attrs["extension_transition_levels_min_used"] = ("" if diagnostics["transition_levels_min_used"] is None
                                                     else diagnostics["transition_levels_min_used"])
    attrs["extension_transition_levels_max_used"] = ("" if diagnostics["transition_levels_max_used"] is None
                                                     else diagnostics["transition_levels_max_used"])
    attrs["extension_columns_extended"] = int(diagnostics["columns_extended"])
    attrs["extension_columns_total"] = int(diagnostics["columns_total"])
    attrs["extension_columns_rejected"] = int(diagnostics["columns_rejected"])
    attrs["extension_note"] = diagnostics["reason"]
    return attrs


def assert_same_application_grid(base, donor):
    """The two datasets are on one and the same application grid, or the blend refuses them.

    Pressure is checked against the configured grid rather than only against each other,
    because two datasets can agree with each other on a grid that is not the project's.
    The latitude axis is checked to be the fixed band grid edge to edge, at whatever band
    width Stage 1 was asked for, so two datasets on different band widths cannot meet.
    """
    grid = np.asarray(config.vertical_grid()["levels"], dtype="float64")
    pressure = _same_coordinate(base, donor, "pressure")
    if pressure.size != grid.size or not np.allclose(pressure, grid, rtol=COORDINATE_TOLERANCE, atol=0.0):
        raise ExtensionProblem(f"pressure must be the {grid.size} configured application levels; both datasets "
                               "must be remapped onto them first")
    latitudes = _same_coordinate(base, donor, "lat")
    _same_coordinate(base, donor, "time")
    if latitudes.size < 2:
        raise ExtensionProblem("the latitude band axis needs at least two bands")
    width = 2.0 * float(latitudes[0] + 90.0)
    centers = np.empty(0)
    try:
        centers = remap.latitude_bands(width)[1] if width > 0.0 else centers
    except ValueError:
        centers = np.empty(0)
    if centers.size != latitudes.size or not np.allclose(centers, latitudes, rtol=0.0, atol=1e-9):
        raise ExtensionProblem("lat must be the fixed latitude band centers of the application grid, produced by "
                               "the remapping step")
    return grid, latitudes, width


def donor_label(donor):
    """How the donor dataset is named in the provenance of the result."""
    return str(donor.attrs.get("model") or donor.attrs.get("source") or EXTENDED_DONOR_FALLBACK)


def extend_product(base, donor, names=(), statistics=EXTENSION_STATISTICS, transition_levels=None,
                   validate=True):
    """One model's application-grid climatology, continued upward with the donor, and what that did.

    Only a base field the donor also holds is extended; a base field without a donor
    counterpart is carried unchanged and a donor-only field is left out, because this step
    extends a model's own fields rather than adding the donor's species. Two-dimensional
    fields have no vertical dimension to extend and are carried as they are. Statistics are
    blended each on their own and none is derived from another.

    Returns the extended dataset and one diagnostic row per base field.
    """
    fields, stems = remap.statistic_fields(base, names, statistics)
    grid, latitudes, width = assert_same_application_grid(base, donor)
    donor_name = donor_label(donor)
    blendable = {field for field in fields if field in donor.data_vars and _is_vertical(base[field])}
    bounds = cf.time_bounds_variable(base["time"])
    extended, carried, rows = {}, {}, []
    for field in base.data_vars:
        if field == bounds:
            continue
        if field in blendable:
            result, diagnostics = extend_field(base[field], donor[field], grid, transition_levels)
            attrs = _extended_attributes(dict(base[field].attrs), diagnostics)
            attrs["extension_donor"] = donor_name
            if "vertical_treatment" in attrs:
                attrs["native_vertical_treatment"] = attrs.pop("vertical_treatment")
            attrs["vertical_treatment"] = (f"climatological statistics of the base model below the transition and "
                                           f"of {donor_name} above it, blended in log pressure")
            extended[field] = xr.DataArray(result.astype("float32"), dims=base[field].dims,
                                           coords=base[field].coords, name=field, attrs=attrs)
            rows.append(diagnostics)
            continue
        reason = (f"{field} is not one of the {', '.join(statistics)} statistics" if field not in fields
                  else "two-dimensional field: there is no vertical dimension to extend" if base[field].ndim == 2
                  else f"{donor_name} holds no {field} field" if field not in donor.data_vars
                  else f"the two datasets hold {field} on different dimensions")
        attrs = dict(base[field].attrs, extension_applied="false", extension_reason=reason)
        (carried if "lat" in base[field].dims else extended)[field] = base[field].assign_attrs(attrs)
        rows.append({"field": field, "variable": str(base[field].attrs.get("source_variable", field)),
                     "extension_applied": False, "columns_total": 0, "columns_extended": 0, "columns_rejected": 0,
                     "columns_no_overlap": 0, "columns_donor_not_higher": 0,
                     "transition_levels_requested": None, "transition_levels_min_used": None,
                     "transition_levels_max_used": None, "base_top_pressure_range": None,
                     "donor_top_pressure_range": None, "reason": reason})
    product = xr.Dataset({**extended, **carried}, coords={
        "time": base["time"], "pressure": base["pressure"], "lat": base["lat"]}, attrs=dict(base.attrs))
    if bounds and bounds in base:
        product[bounds] = base[bounds]
    applied = [row["field"] for row in rows if row["extension_applied"]]
    rejected = [row for row in rows if row["columns_rejected"]]
    settings = config.extension_settings()
    product.attrs.update({
        "product": product_description(donor_name),
        "title": (f"{base.attrs.get('model', '')} monthly zonal climatology {base.attrs.get('climatology_period', '')} "
                  f"on the JuMACS application grid, extended upward with {donor_name}").strip(),
        "extension_base": str(base.attrs.get("model", "")),
        "extension_donor": donor_name,
        "extension_method": EXTENSION_METHOD,
        "extension_weighting": EXTENSION_WEIGHTING,
        "extension_eligibility": EXTENSION_ELIGIBILITY,
        "extension_rejected": "; ".join(f"{row['field']}: {row['reason']}" for row in rejected),
        "default_transition_levels": int(settings["transition_levels"]),
        "transition_levels_by_variable": ", ".join(f"{key}={value}" for key, value in
                                                   sorted(settings["transition_levels_by_variable"].items())),
        "extension_transition_level_unit": "application pressure levels, not kilometres",
        "extended_fields": " ".join(applied),
        "unextended_fields": " ".join(row["field"] for row in rows if not row["extension_applied"]),
        "statistics": " ".join(statistics),
        "base_product": str(base.attrs.get("product", "")),
        "donor_product": str(donor.attrs.get("product", "")),
        "donor_latitude_bands": int(donor.sizes.get("lat", 0)),
        "latitude_band_width_degrees": float(width),
    })
    if validate:
        cf.assert_product(product, names=stems, grid=config.vertical_grid(), statistics=tuple(statistics),
                          kind="application", latitude_bands=int(latitudes.size))
    return product, rows
