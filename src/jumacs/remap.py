"""Fixed latitude-band grid helpers shared by the harmonized climatology products.

A harmonized product is a zonal mean, so the only horizontal grid it needs is the fixed
set of latitude bands the project publishes on. Nothing is written here: these helpers
turn a native latitude axis into area-weighted bands, and the caller assembles the
dataset.

The horizontal treatment is an area mean rather than a point sample: a 5 degree band near
the pole is a small slice of the sphere and a 5 degree band at the equator is a large one,
so the weight of a native latitude cell is the spherical zone area of its overlap with the
band, which is exactly the difference of the sines of its bounding latitudes.
"""
import numpy as np
import xarray as xr

BAND_WIDTH_DEGREES = 5.0


def latitude_bands(width_degrees=BAND_WIDTH_DEGREES):
    """Edges and centres of the fixed latitude bands, edge to edge from -90 to +90 degrees."""
    width = float(width_degrees)
    if width <= 0 or abs(180.0 / width - round(180.0 / width)) > 1e-9:
        raise ValueError(f"the latitude band width must divide the 180 degrees from pole to pole exactly, found "
                         f"{width_degrees}")
    count = round(180.0 / width)
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
