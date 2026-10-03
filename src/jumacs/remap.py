"""Remapping of one published climatology onto the fixed application grid.

A product of one model stays on the grid of that model; this module only answers the
next question, which is what that same climatology looks like on the grid the project
combines models on. Nothing is written here: the result is an in-memory dataset, and a
single remapped model is neither a native product (it no longer holds the model as it
ran) nor a combined product (no second model has entered), so it is described as what
it is.

Two rules shape the arithmetic. The vertical step comes first: on a hybrid grid the
pressure of a level is a field over month and latitude, so averaging over latitude
before interpolating would average the pressure of one level across latitudes where it
is not that pressure, and every value above it would be moved by the wrong amount. The
horizontal step is an area mean rather than a point sample: a 5 degree band near the
pole is a small slice of the sphere and a 5 degree band at the equator is a large one,
so the weight of a native latitude cell is the spherical zone area of its overlap with
the band, which is exactly the difference of the sines of its bounding latitudes.
"""
import numpy as np
import xarray as xr

from . import cf, config, vertical

BAND_WIDTH_DEGREES = 5.0

REMAP_STATISTICS = ("mean", "sigma", "minimum", "maximum")

BAND_LATITUDE_ATTRS = {"standard_name": "latitude", "units": "degrees_north", "axis": "Y",
                       "long_name": "latitude band center"}

VERTICAL_INTERPOLATION = "linear in log(pressure), per calendar month and per native latitude"
VERTICAL_TREATMENT = ("climatological statistics of the model, interpolated onto the shared application pressure "
                      "grid from the native levels")
HORIZONTAL_REMAPPING = "area-weighted latitude-band mean"
LATITUDE_WEIGHTING = ("spherical zone area of the overlap with each native latitude cell: "
                      "sin(north edge) - sin(south edge), with the latitudes in radians")
REMAPPING_ORDER = ("vertical interpolation onto the shared pressure grid first, then area-weighted latitude-band "
                   "averaging: the pressure of a hybrid level varies with latitude, so latitude must not be "
                   "averaged before the vertical step")
MISSING_TREATMENT = ("only finite values contribute; a missing contributor carries zero weight and the remaining "
                     "area weights are renormalised; a cell without any finite contributor is missing")

PRODUCT_DESCRIPTION = "single-model climatology remapped to the JuMACS application grid"
COUNT_NOTE = ("n_years is not remapped: it counts the years that contributed a finite value on the native level, "
              "and no interpolation of that count is meaningful; the count and the coverage diagnostics stay in "
              "the individual product of this model")

REMAP_COMMENT = ("remapped onto the shared pressure grid and fixed latitude bands; cell_methods describes the "
                 "statistic as it was accumulated on the native grid of the model, and the latitude treatment "
                 "added here is recorded in horizontal_remapping and latitude_weighting")

# Global attributes that describe the grid of the individual product. They stay true of
# that product and would be read as false of this one, so they are kept under their own
# name prefixed with 'native_' while the canonical name describes the grid the remap put
# the fields on.
NATIVE_ATTRIBUTES = ("horizontal_grid", "vertical_coordinate", "vertical_level_count", "vertical_level_counts",
                     "latitude_count", "missing_value_policy", "application_pressure_grid",
                     "application_latitude_bands")


def latitude_bands(width_degrees=BAND_WIDTH_DEGREES):
    """Edges and centres of the fixed latitude bands, edge to edge from -90 to +90 degrees."""
    width = float(width_degrees)
    if width <= 0 or abs(180.0 / width - round(180.0 / width)) > 1e-9:
        raise ValueError(f"the latitude band width must divide the 180 degrees from pole to pole exactly, found "
                         f"{width_degrees}")
    count = int(round(180.0 / width))
    edges = -90.0 + width * np.arange(count + 1, dtype="float64")
    edges[0], edges[-1] = -90.0, 90.0
    return edges, (edges[:-1] + edges[1:]) / 2.0


def latitude_cell_bounds(centers):
    """The cell each native latitude value stands for, returned in ascending order.

    Interior boundaries are the midpoints of adjacent centres, which is the only
    boundary two neighbouring cells can agree on; the outermost cells are extended to
    the poles, because a zonal mean axis that stops short of -90 and +90 still holds
    the whole globe and dropping the polar caps would silently discard them. A
    descending axis is read as such and answered ascending; the values themselves are
    never reordered here, only the boundaries of the cells they belong to.
    """
    values = np.asarray(centers, dtype="float64")
    if values.ndim != 1 or values.size < 2:
        raise ValueError("latitude must be a one-dimensional axis of at least two points")
    if not bool(np.isfinite(values).all()) or values.min() < -90.0 or values.max() > 90.0:
        raise ValueError("latitude must be finite and inside -90..90 degrees")
    ascending = bool(values[-1] > values[0])
    forward = values if ascending else values[::-1]
    if not bool(np.all(np.diff(forward) > 0.0)):
        raise ValueError("latitude must be strictly monotonic, ascending or descending")
    bounds = np.concatenate(([-90.0], (forward[1:] + forward[:-1]) / 2.0, [90.0]))
    if not bool(np.all(np.diff(bounds) > 0.0)):
        raise ValueError("the latitude cells derived from the axis are not in increasing order")
    return bounds


def latitude_band_weights(centers, edges=None):
    """Area overlap of every native latitude cell with every band, as a (bands x native) weight matrix.

    The area of a spherical zone between two latitudes is ``2*pi*|sin(phi_north) -
    sin(phi_south)|``, so the difference of sines is the area weight a zonal mean
    deserves: a degree of latitude near the pole covers far less of the sphere than a
    degree at the equator. Neither the degree width alone nor ``cos(centre)`` is that
    area: the first ignores the curvature and the second samples the band at one point
    instead of integrating it, which is wrong for a cell as wide as 5 degrees.
    """
    values = np.asarray(centers, dtype="float64")
    band_edges = np.asarray(edges, dtype="float64") if edges is not None else latitude_bands()[0]
    if band_edges.ndim != 1 or band_edges.size < 2:
        raise ValueError("latitude band edges must be a one-dimensional axis of at least two points")
    if not bool(np.all(np.diff(band_edges) > 0.0)) or band_edges[0] < -90.0 or band_edges[-1] > 90.0:
        raise ValueError("latitude band edges must be increasing and inside -90..90 degrees")
    cell_edges = latitude_cell_bounds(values)
    sine_cells = np.sin(np.radians(cell_edges))
    sine_bands = np.sin(np.radians(band_edges))
    lower = np.maximum(sine_cells[:-1][np.newaxis, :], sine_bands[:-1][:, np.newaxis])
    upper = np.minimum(sine_cells[1:][np.newaxis, :], sine_bands[1:][:, np.newaxis])
    weights = np.maximum(0.0, upper - lower)
    return weights if bool(values[-1] > values[0]) else weights[:, ::-1]


def area_weighted_bands(field, weights, centers, dim="lat"):
    """Area-weighted mean of a field over the native cells falling in each band.

    Missing values are taken out of both the numerator and the denominator, so a cell
    without data neither contributes nor dilutes the mean: the surviving weights are
    renormalised by their own sum, which is the same rule the vertical step uses. Where
    every native cell of a band is missing the band is missing too - it is not filled
    and it is not borrowed from the pole.
    """
    moved = field.transpose(..., dim)
    values = np.asarray(moved, dtype="float64")
    finite = np.isfinite(values)
    matrix = np.asarray(weights, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[1] != values.shape[-1]:
        raise ValueError(f"latitude band weights must be (bands x native latitude) with {values.shape[-1]} "
                         f"native latitudes, found {matrix.shape}")
    if np.asarray(centers).size != matrix.shape[0]:
        raise ValueError("the number of latitude band centers must match the rows of the weight matrix")
    numerator = np.tensordot(np.where(finite, values, 0.0), matrix.T, axes=([-1], [0]))
    support = np.tensordot(finite.astype("float64"), matrix.T, axes=([-1], [0]))
    filled = np.where(support > 0.0, support, 1.0)
    result = np.where(support > 0.0, numerator / filled, np.nan)
    coords = {name: variable for name, variable in moved.coords.items()
              if name != dim and dim not in variable.dims}
    remapped = xr.DataArray(result, dims=moved.dims[:-1] + (dim,),
                            coords={**coords, dim: (dim, np.asarray(centers, dtype="float64"))},
                            name=moved.name)
    remapped.attrs = dict(moved.attrs)
    return remapped


def _is_pressure_name(stem):
    """Whether a variable name is a published pressure, which the application grid never carries.

    An individual product publishes the pressure of its native levels as its own field, and on the
    application grid that role belongs to the shared ``pressure`` coordinate, so such a field is left
    out of a remap whether it appears alone or with a statistic suffix.
    """
    return stem == cf.PRESSURE_FIELD or stem.startswith(f"{cf.PRESSURE_FIELD}_")


def _suffixed_fields(ds, statistics):
    """The fields of a product whose name ends in one of the statistic suffixes, mapped to their stems.

    Published pressures are skipped, and the suffix is read as the last name of the field,
    which is how a product writes one field per variable and statistic.
    """
    found = {}
    for field in ds.data_vars:
        statistic = next((name for name in statistics if field.endswith(f"_{name}")), None)
        if statistic is None:
            continue
        stem = field[: -len(statistic) - 1]
        if not _is_pressure_name(stem):
            found[field] = stem
    return found


def statistic_fields(ds, names=(), statistics=REMAP_STATISTICS):
    """The statistic fields a remap carries, and the variable names they belong to.

    A name is matched without regard to case, because the products are written in the
    spelling the source uses and a caller has no reason to know it.
    """
    requested = tuple(statistics)
    if "n_years" in requested:
        raise ValueError(f"n_years cannot be remapped: {COUNT_NOTE}")
    wanted = {str(name).lower() for name in names}
    found = {field: stem for field, stem in _suffixed_fields(ds, requested).items()
             if not wanted or stem.lower() in wanted}
    if wanted:
        missing = sorted(wanted - {stem.lower() for stem in found.values()})
        if missing:
            raise ValueError(f"the product holds no {', '.join(missing)} statistic to remap")
    incomplete = sorted(f"{stem}_{name}" for stem in set(found.values()) for name in requested
                        if f"{stem}_{name}" not in found)
    if incomplete:
        raise ValueError(f"a remap carries every statistic of a variable it carries, and the product holds no "
                         f"{', '.join(incomplete)}")
    return sorted(found), tuple(sorted(set(found.values())))


def _pressure_coordinate_attributes(levels):
    """Attributes of the shared pressure coordinate the remapped fields land on."""
    return dict(vertical.PRESSURE_ATTRS, positive="down", axis="Z",
                long_name="air pressure on the shared application grid",
                comment=(f"{levels.size} levels, from {levels.max():g} Pa to {levels.min():g} Pa, evenly spaced in "
                         f"log(pressure); every field of this dataset is placed on these levels"))


def _field_attributes(source, remapped, width_degrees):
    """Attributes of one remapped field: identity kept, native vertical references dropped.

    The identity of a field - its name in the source, its units, its standard name, the
    cell method that says how the statistic was accumulated - survives, because the remap
    changes where the values sit and not what they are. The interpolation step drops the
    attributes on its way through, so they are taken from the field as published.
    """
    attrs = dict(source.attrs)
    for stale in ("pressure_field", "pressure_coordinate"):
        attrs.pop(stale, None)
    attrs["horizontal_remapping"] = HORIZONTAL_REMAPPING
    attrs["latitude_band_width_degrees"] = float(width_degrees)
    attrs["latitude_weighting"] = LATITUDE_WEIGHTING
    attrs["comment"] = REMAP_COMMENT
    if remapped.ndim == 3:
        attrs["vertical_interpolation"] = VERTICAL_INTERPOLATION
        attrs["vertical_extrapolation"] = "none"
        attrs["on_application_pressure_grid"] = "true"
        attrs["native_level_dimension"] = str(source.attrs.get("native_level_dimension", source.dims[1]))
        if "vertical_treatment" in attrs:
            attrs["native_vertical_treatment"] = attrs.pop("vertical_treatment")
        attrs["vertical_treatment"] = VERTICAL_TREATMENT
    return attrs


def remap_field(ds, name, levels, weights, band_centers, width_degrees=BAND_WIDTH_DEGREES):
    """One statistic field of a product onto the application grid, vertical step first.

    The native pressure of the field is read exactly as published - a pressure-valued
    level coordinate or the hybrid ``air_pressure`` field the product names - and the
    column interpolation is the shared one, so a level outside what the model reached
    stays missing instead of being extrapolated or bridged across a gap. A field with
    two dimensions is a column-integrated or single-per-column quantity: it gets the
    latitude step only and keeps its shape without a vertical dimension.
    """
    field = ds[name]
    grid = np.asarray(levels, dtype="float64")
    if field.ndim == 3:
        pressure = vertical.product_pressure(ds, field)
        if pressure is None:
            raise ValueError(f"{name} is three-dimensional on '{field.dims[1]}' but the product publishes no "
                             f"pressure for those levels, so it cannot be placed on the application grid")
        moved = vertical.interpolate_log_pressure(field.astype("float64"), pressure.astype("float64"), grid,
                                                 bridge_gaps=False)
    elif field.ndim == 2:
        moved = field.astype("float64")
    else:
        raise ValueError(f"{name} must be two- or three-dimensional, found {field.dims}")
    remapped = area_weighted_bands(moved, weights, band_centers).assign_attrs(
        _field_attributes(field, moved, width_degrees))
    return remapped.astype("float32")


def remap_product(ds, names=(), statistics=REMAP_STATISTICS, levels=None, width_degrees=BAND_WIDTH_DEGREES,
                  validate=True):
    """One model's published climatology on the shared pressure grid and the fixed latitude bands.

    Every statistic field is moved by the same geometric procedure and on its own: the
    mean, the standard deviation, the minimum and the maximum of a native level are each
    placed on the target grid as they stand, which is what a single-model remap can do
    honestly - the band mean of a monthly standard deviation is not the standard
    deviation of the band, and the attributes say where each value came from. The count
    of contributing years is refused rather than invented. Fields that are neither
    statistics nor published pressures and that still carry a latitude dimension are
    refused too, because there is no rule here for remapping them.
    """
    if "lat" not in ds.coords:
        raise ValueError("the product has no lat coordinate to remap")
    grid = np.asarray(config.vertical_grid()["levels"] if levels is None else levels, dtype="float64")
    band_edges, band_centers = latitude_bands(width_degrees)
    weights = latitude_band_weights(ds["lat"].values, band_edges)
    fields, stems = statistic_fields(ds, names, statistics)
    if not fields:
        raise ValueError("the product holds no statistic field to remap")
    carried = set(fields) | set(_suffixed_fields(ds, cf.STATISTIC_ORDER)) | set(cf._pressure_fields(ds))
    for other in ds.data_vars:
        if other not in carried and "lat" in ds[other].dims:
            raise ValueError(f"{other} is neither a statistic field nor a published pressure field but varies with "
                             f"latitude; it cannot be remapped by any rule of this module")
    remapped = {name: remap_field(ds, name, grid, weights, band_centers, width_degrees) for name in fields}
    product = xr.Dataset(remapped, coords={
        "time": ds["time"],
        "pressure": ("pressure", grid, _pressure_coordinate_attributes(grid)),
        "lat": ("lat", band_centers, dict(BAND_LATITUDE_ATTRS)),
    }, attrs=dict(ds.attrs))
    bounds = cf.time_bounds_variable(ds["time"])
    if bounds and bounds in ds:
        product[bounds] = ds[bounds]
    for native_attr in NATIVE_ATTRIBUTES:
        if native_attr in product.attrs:
            product.attrs[f"native_{native_attr}"] = product.attrs.pop(native_attr)
    bands = int(band_centers.size)
    product.attrs.update({
        "product": PRODUCT_DESCRIPTION,
        "native_product": str(ds.attrs.get("product", "")),
        "title": (f"{ds.attrs.get('model', '')} {ds.attrs.get('experiment', '')} monthly zonal climatology "
                  f"{ds.attrs.get('climatology_period', '')} on the JuMACS application grid").strip(),
        "vertical_interpolation": VERTICAL_INTERPOLATION,
        "vertical_extrapolation": "none",
        "horizontal_remapping": HORIZONTAL_REMAPPING,
        "horizontal_grid": (f"zonal mean over all longitudes; latitude is the fixed grid of {bands} area-weighted "
                            f"{width_degrees:g} degree bands, not the native latitude grid of the model"),
        "latitude_band_width_degrees": float(width_degrees),
        "latitude_band_count": bands,
        "latitude_count": bands,
        "latitude_weighting": LATITUDE_WEIGHTING,
        "remapping_order": REMAPPING_ORDER,
        "missing_value_policy": MISSING_TREATMENT,
        "application_pressure_grid": (f"{grid.size} levels from {grid.max():g} to {grid.min():g} Pa, applied here by "
                                      f"interpolation in log(pressure)"),
        "application_latitude_bands": (f"{bands} bands of {width_degrees:g} degrees from -90 to 90 degrees, applied "
                                       f"here by area-weighted averaging"),
        "statistics": " ".join(statistics),
        "count_statistics": COUNT_NOTE,
        "variable_count": len(stems),
    })
    if validate:
        cf.assert_product(product, names=stems, grid=config.vertical_grid(), statistics=tuple(statistics),
                          kind="application", latitude_bands=bands)
    return product
