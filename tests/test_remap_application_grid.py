"""Remapping one model's product onto the shared pressure grid and the fixed latitude bands."""
import numpy as np
import pytest
import xarray as xr

from jumacs import cf, config, remap
from jumacs.climatology import assemble_product, variable_statistics

GRID = config.vertical_grid()
LEVELS = np.asarray(GRID["levels"], float)
LAT = np.linspace(-88.57216851400727, 88.57216851400727, 96)
SOCOL_PLEV = np.logspace(5.0, -0.5, 39)


def part(name, lat, value, levels=SOCOL_PLEV, level="plev", units="mol mol-1"):
    stats = xr.Dataset(
        {statistic: (("month", level, "lat"), np.full((12, len(levels), lat.size), value, float))
         for statistic in cf.STATISTIC_ORDER},
        coords={"month": list(range(1, 13)), level: (level, levels, {"units": "Pa"}), "lat": ("lat", lat)})
    return variable_statistics(name, stats, {"units": units}, {})


def index_part(name, lat, value, count, level="lev"):
    stats = xr.Dataset(
        {statistic: (("month", level, "lat"), np.full((12, count, lat.size), value, float))
         for statistic in cf.STATISTIC_ORDER},
        coords={"month": list(range(1, 13)), level: (level, np.arange(count, dtype=float), {"units": "1"}),
                "lat": ("lat", lat)})
    return variable_statistics(name, stats, {"units": "K"}, {})


def hybrid_field(lat, count, level="lev", surface=1.0):
    values = np.broadcast_to((np.logspace(5.0, 0.0, count) * surface)[None, :, None], (12, count, lat.size)).copy()
    return xr.DataArray(values, dims=("month", level, "lat"),
                        coords={"month": list(range(1, 13)), "lat": ("lat", lat)})


def socol_pressures(pressure_field):
    return {"plev": {"kind": "coordinate", "pressure": xr.DataArray(SOCOL_PLEV.copy(), dims=("plev",)),
                     "source_variable": "br", "coverage": (2000, 2001), "units": "Pa"},
            "lev": {"kind": "hybrid", "pressure": pressure_field, "source_variable": "ta",
                    "coverage": (2000, 2001), "units": "Pa", "source_level": "lev"}}


def product(**kwargs):
    return assemble_product("SOCOL", 2000, 2001,
                            [part("o3", LAT, 3.0), index_part("ta", LAT, 250.0, 47)], (), ["br", "ta"],
                            socol_pressures(hybrid_field(LAT, 47)))


def band_field(values, pressures, lat, level="lev"):
    """A field on a native grid with its hybrid pressure, for a direct call to remap_field."""
    array = np.asarray(values, float)
    coords = {"time": [0.0], level: (level, np.arange(array.shape[0], dtype=float)),
              "lat": ("lat", np.asarray(lat, float))}
    field = xr.DataArray(array[None], dims=("time", level, "lat"), coords=coords,
                         attrs={"units": "K", "pressure_field": "air_pressure"})
    return xr.Dataset({"ta_mean": field, "air_pressure": (("time", level, "lat"), np.asarray(pressures, float)[None])})


def test_the_fixed_bands_run_edge_to_edge_from_pole_to_pole():
    edges, centers = remap.latitude_bands()
    assert edges[0] == -90.0 and edges[-1] == 90.0
    assert edges.size == 37 and centers.size == 36
    assert np.allclose(np.diff(edges), 5.0)
    assert centers[0] == -87.5 and centers[-1] == 87.5
    assert np.array_equal(centers, (edges[:-1] + edges[1:]) / 2.0)
    assert remap.latitude_bands(10.0)[1].size == 18
    with pytest.raises(ValueError, match="divide the 180 degrees"):
        remap.latitude_bands(7.0)


def test_every_band_weight_is_the_area_of_its_overlap():
    lat = -89.5 + np.arange(180, dtype=float)
    weights = remap.latitude_band_weights(lat)
    assert weights.shape == (36, 180)
    assert (weights >= 0.0).all()
    sine_bands = np.sin(np.radians(remap.latitude_bands()[0]))
    sine_cells = np.sin(np.radians(remap.latitude_cell_bounds(lat)))
    assert np.allclose(weights.sum(axis=1), np.diff(sine_bands))
    assert np.allclose(weights.sum(axis=0), np.diff(sine_cells))
    assert weights.sum() == pytest.approx(2.0)


def test_the_same_latitude_width_near_a_pole_covers_less_area():
    lat = -89.5 + np.arange(180, dtype=float)
    weights = remap.latitude_band_weights(lat)
    assert weights[0].sum() < weights[18].sum()
    assert weights[:, 0].sum() < weights[:, 90].sum()


def test_cell_boundaries_are_midpoints_inside_and_the_poles_outside():
    centers = np.array([-60.0, -30.0, 0.0, 30.0])
    bounds = remap.latitude_cell_bounds(centers)
    assert np.array_equal(bounds, [-90.0, -45.0, -15.0, 15.0, 90.0])
    assert np.array_equal(remap.latitude_cell_bounds(centers[::-1]), bounds)
    edges = remap.latitude_bands()[0]
    forward = remap.latitude_band_weights(centers, edges)
    reversed_axis = remap.latitude_band_weights(centers[::-1], edges)
    assert np.allclose(reversed_axis, forward[:, ::-1])
    with pytest.raises(ValueError, match="strictly monotonic"):
        remap.latitude_cell_bounds([0.0, 30.0, -60.0])
    with pytest.raises(ValueError, match="one-dimensional axis of at least two"):
        remap.latitude_cell_bounds([0.0])
    with pytest.raises(ValueError, match="inside -90..90"):
        remap.latitude_cell_bounds([-91.0, 0.0, 90.0])


def test_a_band_holding_native_cells_exactly_returns_them_weighted_by_area():
    _, centers = remap.latitude_bands()
    weights = remap.latitude_band_weights(centers)
    values = np.arange(centers.size, dtype=float)
    field = xr.DataArray(values[None, :], dims=("time", "lat"), coords={"time": [0.0], "lat": ("lat", centers)})
    banded = remap.area_weighted_bands(field, weights, centers)
    assert np.allclose(banded.values[0], values)
    assert np.array_equal(banded.lat.values, centers)
    two = xr.DataArray(np.array([[1.0, 3.0]]), dims=("time", "lat"), coords={"time": [0.0], "lat": [-2.5, 2.5]})
    equal = remap.latitude_band_weights([-2.5, 2.5], [-2.5, 2.5])
    assert np.allclose(equal[0, 0], equal[0, 1])
    assert remap.area_weighted_bands(two, equal, [0.0]).values[0, 0] == pytest.approx(2.0)


def test_a_missing_native_cell_leaves_the_others_to_carry_the_band():
    two = xr.DataArray(np.array([[1.0, np.nan], [np.nan, np.nan]]), dims=("time", "lat"),
                       coords={"time": [0.0, 1.0], "lat": [-2.5, 2.5]})
    weights = remap.latitude_band_weights([-2.5, 2.5], [-2.5, 2.5])
    banded = remap.area_weighted_bands(two, weights, [0.0])
    assert banded.values[0, 0] == pytest.approx(1.0)
    assert np.isnan(banded.values[1, 0])
    assert np.isfinite(banded.sel(time=0)).all()


def test_the_vertical_step_comes_before_the_latitude_mean():
    lat = [-2.5, 2.5]
    surface = [1.0e5, 4.0e4]
    top = 1.0e3
    ds = band_field([[0.0, 0.0], [1.0, 1.0]], [[surface[0], surface[1]], [top, top]], lat)
    target = np.array([3.0e4])
    weights = remap.latitude_band_weights(lat, [-2.5, 2.5])
    banded = remap.remap_field(ds, "ta_mean", target, weights, [0.0])
    placed = [np.log(surface[index] / 3.0e4) / np.log(surface[index] / top) for index in range(2)]
    expected = sum(placed) / len(placed)
    assert banded.values[0, 0, 0] == pytest.approx(expected, rel=1e-6)
    averaged_first = np.log((surface[0] + surface[1]) / 2.0 / 3.0e4) / np.log((surface[0] + surface[1]) / 2.0 / top)
    assert banded.values[0, 0, 0] != pytest.approx(averaged_first, rel=1e-3)
    assert banded.dims == ("time", "pressure", "lat")


def test_a_two_dimensional_field_keeps_its_shape_without_a_vertical_step():
    lat = -89.5 + np.arange(180, dtype=float)
    values = np.full((2, lat.size), 3.0)
    field = xr.DataArray(values, dims=("time", "lat"), coords={"time": [0.0, 1.0], "lat": ("lat", lat)},
                         attrs={"units": "m", "vertical_treatment": "two-dimensional field; no vertical dimension"})
    ds = xr.Dataset({"toz_mean": field})
    edges, centers = remap.latitude_bands()
    banded = remap.remap_field(ds, "toz_mean", LEVELS, remap.latitude_band_weights(lat, edges), centers)
    assert banded.dims == ("time", "lat")
    assert banded.shape == (2, 36)
    assert np.allclose(banded.values, 3.0)
    assert banded.dtype == np.dtype("float32")
    assert "vertical_interpolation" not in banded.attrs
    assert banded.attrs["units"] == "m"
    assert banded.attrs["vertical_treatment"] == "two-dimensional field; no vertical dimension"


def test_a_level_the_model_does_not_reach_stays_missing():
    lat = [-2.5, 2.5]
    ds = band_field([[0.0, 0.0], [1.0, 1.0]], [[1.0e5, 1.0e5], [10.0, 10.0]], lat)
    weights = remap.latitude_band_weights(lat, [-2.5, 2.5])
    target = np.array([1.0e6, 5.0e4, 1.0, 0.002])
    banded = remap.remap_field(ds, "ta_mean", target, weights, [0.0])
    assert np.isnan(banded.values[0, 0, 0])
    assert np.isfinite(banded.values[0, 1, 0])
    assert np.isnan(banded.values[0, 2, 0]) and np.isnan(banded.values[0, 3, 0])


def test_a_missing_native_level_divides_the_column_rather_than_being_bridged():
    lat = [-1.0, 1.0]
    ds = band_field([[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]],
                    [[1.0e5, 1.0e5], [1.0e4, 1.0e4], [1.0e3, 1.0e3], [1.0e2, 1.0e2]], lat)
    weights = remap.latitude_band_weights(lat, [-90.0, 90.0])
    banded = remap.remap_field(ds, "ta_mean", np.array([5.0e4, 5.0e2]), weights, [0.0])
    assert np.isfinite(banded.values[0, 0, 0])
    assert np.isnan(banded.values[0, 1, 0])


def test_a_product_reaches_the_application_grid_and_still_validates():
    remapped = remap.remap_product(product())
    assert dict(remapped.sizes) == {"time": 12, "pressure": LEVELS.size, "lat": 36, "nv": 2}
    assert np.array_equal(remapped.pressure.values, LEVELS)
    assert remapped.pressure.attrs["axis"] == "Z" and remapped.pressure.attrs["positive"] == "down"
    assert np.array_equal(remapped.lat.values, remap.latitude_bands()[1])
    assert remapped.lat.dtype == np.dtype("float64") and remapped.lat.attrs["units"] == "degrees_north"
    assert remapped["climatology_bounds"].dims == ("time", "nv")
    assert remapped.time.attrs == product().time.attrs
    for field in ("o3_mean", "o3_sigma", "o3_minimum", "o3_maximum", "ta_mean"):
        variable = remapped[field]
        assert variable.dims == ("time", "pressure", "lat")
        assert variable.dtype == np.dtype("float32")
        assert variable.attrs["cell_methods"] == cf.cell_methods(field.rsplit("_", 1)[1])
        finite = np.isfinite(variable.values)
        assert finite.any() and np.allclose(variable.values[finite], 3.0 if field.startswith("o3") else 250.0)
        assert variable.attrs["on_application_pressure_grid"] == "true"
        assert "pressure_field" not in variable.attrs


def test_the_published_pressures_and_the_counts_are_left_behind():
    source = product()
    assert any(name.startswith("air_pressure") for name in source.data_vars)
    remapped = remap.remap_product(source)
    assert not [name for name in remapped.data_vars if name.startswith("air_pressure")]
    assert not [name for name in remapped.data_vars if name.endswith("_n_years")]
    assert any(name.endswith("_n_years") for name in source.data_vars)
    with pytest.raises(ValueError, match="n_years cannot be remapped"):
        remap.remap_product(source, statistics=("mean", "n_years"))
    assert "n_years" in remapped.attrs["count_statistics"]


def test_the_method_of_the_remap_is_written_into_the_product_it_describes():
    source = product()
    source["ta_mean"].attrs["vertical_treatment"] = "climatological statistics of the model on its native levels"
    remapped = remap.remap_product(source)
    assert remapped.attrs["product"] == remap.PRODUCT_DESCRIPTION
    assert remapped.attrs["native_product"] == product().attrs["product"]
    assert remapped.attrs["vertical_interpolation"] == remap.VERTICAL_INTERPOLATION
    assert remapped.attrs["vertical_extrapolation"] == "none"
    assert remapped.attrs["horizontal_remapping"] == remap.HORIZONTAL_REMAPPING
    assert remapped.attrs["latitude_band_count"] == 36
    assert remapped.attrs["native_latitude_count"] == LAT.size
    assert remapped.attrs["native_horizontal_grid"] == product().attrs["horizontal_grid"]
    assert "36" in remapped.attrs["horizontal_grid"]
    assert "36" in remapped.attrs["application_latitude_bands"]
    assert "vertical" in remapped.attrs["remapping_order"]
    assert remapped["ta_mean"].attrs["vertical_treatment"] == remap.VERTICAL_TREATMENT
    assert remapped["ta_mean"].attrs["native_vertical_treatment"] == (
        source["ta_mean"].attrs["vertical_treatment"])


def test_only_the_named_variables_are_remapped_and_names_ignore_case():
    remapped = remap.remap_product(product(), names=("O3",))
    assert sorted(remapped.data_vars) == ["climatology_bounds", "o3_maximum", "o3_mean", "o3_minimum", "o3_sigma"]
    with pytest.raises(ValueError, match="holds no xyz statistic"):
        remap.remap_product(product(), names=("xyz",))


def test_a_variable_without_every_statistic_is_not_carried_partly():
    source = product()
    del source["o3_sigma"]
    with pytest.raises(ValueError, match="o3_sigma"):
        remap.remap_product(source, names=("o3",))


def test_a_field_that_is_neither_a_statistic_nor_a_published_pressure_is_refused():
    source = product()
    source["raw_thing"] = (("time", "lat"), np.zeros((12, LAT.size)))
    with pytest.raises(ValueError, match="raw_thing"):
        remap.remap_product(source, names=("o3",))


def test_the_band_width_is_a_choice_that_the_product_records():
    remapped = remap.remap_product(product(), names=("o3",), width_degrees=10.0)
    assert remapped.sizes["lat"] == 18
    assert remapped.attrs["latitude_band_width_degrees"] == 10.0
    assert np.allclose(remapped.lat.values, [-85.0 + 10.0 * index for index in range(18)])
