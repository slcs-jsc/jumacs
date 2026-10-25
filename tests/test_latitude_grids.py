"""Models with more than one native latitude grid publish each grid as its own dimension."""
import numpy as np
import pytest
import xarray as xr

import jumacs.climatology as climatology_module
from jumacs import cf, inventory
from jumacs.harmonize import _harmonize_open_dataset, _isa_height_m
from jumacs.climatology import (
    LATITUDE_SNAP_TOLERANCE_DEGREES,
    assemble_product,
    assign_latitude_grid,
    build_climatology,
    variable_statistics,
)
from jumacs.config import load_config, vertical_grid
from jumacs.vertical import is_latitude_dimension

GRID = vertical_grid()
LEVELS = np.asarray(GRID["levels"], float)
# A vertical grid of the model itself, which the application grid deliberately is not.
NATIVE = np.logspace(5.0, -0.5, 67)

# A 96-point archive latitude axis and a 72-point axis of the kind an EP flux variable can carry.
LAT = np.linspace(-88.57216851400727, 88.57216851400727, 96)
LAT2 = np.linspace(-87.5, 87.5, 72)
LAT3 = np.linspace(-86.0, 86.0, 25)


def gridded_parts(specs, model="IPSL-CM6A-ATM-LR-REPROBUS"):
    grids = {}
    return [assign_latitude_grid(part(name, lat, value), grids, model, name) for name, lat, value in specs]


def jitter(lat, scale):
    return lat + scale * np.where(np.arange(lat.size) % 2, 1.0, -1.0)


def part(name, lat, value, levels=NATIVE, level="plev"):
    stats = xr.Dataset(
        {statistic: (("month", level, "lat"), np.full((12, len(levels), lat.size), value, float))
         for statistic in cf.STATISTIC_ORDER},
        coords={"month": list(range(1, 13)), level: (level, levels, {"units": "Pa"}), "lat": ("lat", lat)})
    return variable_statistics(name, stats, {"units": "mol mol-1"}, {})


def write_zonal(path, name, lat, value):
    ds = xr.Dataset(
        {name: (("time", "plev", "lat"), np.broadcast_to(np.array(value, float), (24, NATIVE.size, lat.size)).copy(),
                {"units": "mol mol-1"})},
        coords={"time": xr.date_range("2000-01", periods=24, freq="MS", use_cftime=True),
                "plev": ("plev", NATIVE, {"units": "Pa", "standard_name": "air_pressure"}),
                "lat": ("lat", lat)})
    ds.to_netcdf(path)


def build_product(tmp_path, monkeypatch, grids):
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    zonal = tmp_path / load_config("IPSL-CM6A-ATM-LR-REPROBUS")["paths"]["zonal"]
    zonal.mkdir(parents=True)
    for name, (lat, value) in grids.items():
        write_zonal(zonal / f"{name}_monthly_zonal.nc", name, lat, value)
    product = build_climatology("IPSL-CM6A-ATM-LR-REPROBUS", 2000, 2001)
    with xr.open_dataset(product, decode_cf=False) as written:
        return written.load()


def test_a_second_native_latitude_grid_becomes_its_own_dimension(tmp_path, monkeypatch):
    ds = build_product(tmp_path, monkeypatch, {"o3": (LAT, 3.0e-6), "epfy": (LAT2, 1.0e-3)})
    assert {dim for dim in ds.dims if is_latitude_dimension(dim)} == {"lat", "lat_2"}
    for name, lat in (("epfy", LAT2), ("o3", LAT)):
        axis = ds[f"{name}_mean"].attrs["latitude_axis"]
        assert ds[f"{name}_mean"].dims == ("time", "plev", axis)
        assert ds.sizes[axis] == lat.size
        assert np.array_equal(ds[axis].values, lat)
    assert ds.lat.dtype == np.dtype("float64") and ds.lat_2.dtype == np.dtype("float64")
    for coordinate in ("lat", "lat_2"):
        assert ds[coordinate].attrs["standard_name"] == "latitude"
        assert ds[coordinate].attrs["units"] == "degrees_north"
        assert ds[coordinate].attrs["axis"] == "Y"
    assert "comment" not in ds.lat.attrs
    assert "lat_2" in ds.lat_2.attrs["comment"]
    assert "latitude_axis" in ds.lat_2.attrs["comment"]
    for name in ("o3", "epfy"):
        assert np.isfinite(ds[f"{name}_mean"].values).all()
    cf.assert_product(ds, names=("o3", "epfy"), grid=GRID)


def test_discovery_order_names_the_grids_lat_lat_2_and_lat_3(tmp_path, monkeypatch):
    ds = build_product(tmp_path, monkeypatch, {"o3": (LAT, 3.0e-6), "epfy": (LAT2, 1.0e-3), "zg": (LAT3, 100.0)})
    assert ds.sizes["lat"] == LAT2.size
    assert ds.sizes["lat_2"] == LAT.size
    assert ds.sizes["lat_3"] == LAT3.size
    assert ds.epfy_mean.attrs["latitude_axis"] == "lat"
    assert ds.o3_mean.attrs["latitude_axis"] == "lat_2"
    assert ds.zg_mean.attrs["latitude_axis"] == "lat_3"
    assert ds.attrs["latitude_count"] == LAT2.size
    assert ds.attrs["latitude_counts"] == f"lat={LAT2.size}; lat_2={LAT.size}; lat_3={LAT3.size}"


def test_near_identical_grids_still_collapse_to_one_dimension(tmp_path, monkeypatch):
    ds = build_product(tmp_path, monkeypatch, {"o3": (LAT, 3.0e-6), "epfy": (jitter(LAT, 4.0e-6), 1.0e-3)})
    assert {dim for dim in ds.dims if is_latitude_dimension(dim)} == {"lat"}
    assert ds.sizes["lat"] == 96
    assert np.abs(ds.lat.values - LAT).max() <= LATITUDE_SNAP_TOLERANCE_DEGREES
    for name in ("o3", "epfy"):
        assert ds[f"{name}_mean"].dims == ("time", "plev", "lat")
        assert np.isfinite(ds[f"{name}_mean"].values).all()
    assert ds.attrs["latitude_counts"] == "lat=96"


def test_a_grid_only_beyond_tolerance_stays_a_grid_of_its_own_without_padding(tmp_path, monkeypatch):
    beyond = jitter(LAT2, 1.0e-9)
    beyond[30] += 0.5
    ds = build_product(tmp_path, monkeypatch, {"o3": (LAT, 3.0e-6), "epfy": (LAT2, 1.0e-3), "zg": (beyond, 100.0)})
    assert ds.sizes["lat"] == 72 and ds.sizes["lat_2"] == 96 and ds.sizes["lat_3"] == 72
    assert np.array_equal(ds.lat.values, LAT2) and np.array_equal(ds.lat_3.values, beyond)
    assert ds.zg_mean.dims == ("time", "plev", "lat_3")
    assert ds.zg_mean.attrs["latitude_axis"] == "lat_3"
    assert np.isfinite(ds.zg_mean.values).all() and np.isfinite(ds.epfy_mean.values).all()


def test_a_latitude_axis_beyond_tolerance_below_the_snapping_level_is_refused():
    beyond = jitter(LAT, 7.6e-13)
    beyond[40] += 1e-3
    with pytest.raises(RuntimeError, match="differs from the canonical model grid"):
        assemble_product("IPSL-CM6A-ATM-LR-REPROBUS", 2000, 2001, [part("o3", LAT, 1.0), part("epfy", beyond, 2.0)])


def height_part(lat):
    """A geopotential profile on the model's own native levels, in m2 s-2."""
    profile = 9.80665 * _isa_height_m(NATIVE)
    stats = xr.Dataset(
        {statistic: (("month", "plev", "lat"),
                     np.broadcast_to(profile[None, :, None], (12, NATIVE.size, lat.size)).copy())
         for statistic in cf.STATISTIC_ORDER},
        coords={"month": list(range(1, 13)), "plev": ("plev", NATIVE, {"units": "Pa"}), "lat": ("lat", lat)})
    return variable_statistics("zg", stats, {"units": "m2 s-2"}, {})


def harmonized_grids_dataset():
    model = "IPSL-CM6A-ATM-LR-REPROBUS"
    grids = {}
    parts = [assign_latitude_grid(part(name, lat, value), grids, model, name)
             for name, lat, value in (("o3", LAT, 1.0), ("epfy", LAT2, 2.0))]
    parts.append(assign_latitude_grid(height_part(LAT), grids, model, "zg"))
    product = assemble_product(model, 2000, 2001, parts)
    prepared = _harmonize_open_dataset(product, model, load_config(model), 2000, 2001, 5, 1)
    return product, prepared["dataset"]


def test_harmonize_grids_native_latitude_fields_on_the_canonical_bands():
    product, harmonized = harmonized_grids_dataset()
    assert product.o3_mean.attrs["latitude_axis"] == "lat"
    assert product.epfy_mean.attrs["latitude_axis"] == "lat_2"
    assert product.attrs["latitude_counts"] == f"lat={LAT.size}; lat_2={LAT2.size}"
    assert "lat_2" not in harmonized.dims
    # A field carried only on a second latitude grid has no band weights and stays out of the product.
    assert "epfy_mean" not in harmonized.data_vars
    field = harmonized["o3_mean"]
    assert field.dims == ("month", "altitude", "lat")
    assert harmonized.sizes["lat"] == 36
    np.testing.assert_allclose(field.values[np.isfinite(field.values)], 1.0)
    assert float(harmonized.altitude[0]) == 0.0
    assert float(harmonized.altitude[-1]) == float(np.floor(_isa_height_m(NATIVE).max() / 1000.0))
    assert cf.validate_product(harmonized, names=("o3",), kind="harmonized") == []


def test_the_harmonized_contract_still_refuses_a_latitude_axis_named_lat_2():
    _, harmonized = harmonized_grids_dataset()
    assert cf.validate_product(harmonized.rename({"lat": "lat_2"}), names=("o3",), kind="harmonized") != []


def test_the_inventory_recognizes_fields_on_a_second_latitude_grid(tmp_path):
    lat = np.linspace(-85.0, 85.0, 72)
    ds = xr.Dataset(
        {"o3_mean": (("time", "lat_2"), np.zeros((2, lat.size), np.float32), {"units": "mol mol-1"}),
         "epfy_mean": (("time", "pressure", "lat_2"), np.zeros((2, 3, lat.size), np.float32), {"units": "m2 s-2"})},
        coords={"time": [0.0, 1.0], "pressure": ("pressure", [100000., 1000., 10.]), "lat_2": ("lat_2", lat)})
    path = tmp_path / "ipsl_cm6a_refd1_2000_2001.nc"
    ds.to_netcdf(path)
    skipped = []
    fields = inventory._fields(path, skipped)
    assert skipped == []
    assert fields["o3"]["dimensionality"] == "2D"
    assert fields["epfy"]["dimensionality"] == "3D"


def test_profile_interpolation_carries_the_first_valid_value_down_to_the_surface():
    from jumacs.harmonize import interpolate_profile
    height = np.array([2.0, 3.0, 4.0])
    values = np.array([10.0, 20.0, 30.0])
    got = interpolate_profile(height, values, np.arange(0.0, 5.0))
    np.testing.assert_allclose(got, [10.0, 10.0, 10.0, 20.0, 30.0])
    flipped = interpolate_profile(height[::-1], values[::-1], np.arange(0.0, 5.0))
    np.testing.assert_allclose(flipped, [10.0, 10.0, 10.0, 20.0, 30.0])


def test_profile_interpolation_never_bridges_a_missing_level_inside_a_profile():
    from jumacs.harmonize import interpolate_profile
    height = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([1.0, 2.0, np.nan, np.nan, 4.0])
    got = interpolate_profile(height, values, np.arange(0.0, 6.0))
    np.testing.assert_allclose(got, [1.0, 2.0, np.nan, np.nan, 4.0, 4.0], equal_nan=True)


def test_profile_interpolation_carries_the_last_valid_value_up_to_the_model_top():
    from jumacs.harmonize import interpolate_profile
    height = np.array([0.0, 1.0, 2.0])
    values = np.array([5.0, 7.0, 9.0])
    got = interpolate_profile(height, values, np.arange(0.0, 6.0))
    np.testing.assert_allclose(got, [5.0, 7.0, 9.0, 9.0, 9.0, 9.0])


def test_upper_carry_is_constant_and_uses_the_last_valid_value_not_a_slope():
    from jumacs.harmonize import interpolate_profile
    height = np.array([0.0, 2.0, 4.0])
    values = np.array([10.0, 20.0, 30.0])
    got = interpolate_profile(height, values, np.arange(0.0, 8.0))
    np.testing.assert_allclose(got, [10.0, 15.0, 20.0, 25.0, 30.0, 30.0, 30.0, 30.0])


def test_upper_carry_follows_the_final_run_when_a_profile_has_an_internal_gap():
    from jumacs.harmonize import interpolate_profile
    height = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    values = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    got = interpolate_profile(height, values, np.arange(0.0, 7.0))
    np.testing.assert_allclose(got, [1.0, np.nan, 3.0, np.nan, np.nan, 6.0, 6.0], equal_nan=True)


def test_carry_directions_survive_a_descending_native_axis():
    from jumacs.harmonize import interpolate_profile
    height = np.array([3.0, 2.0, 1.0])
    values = np.array([30.0, 20.0, 10.0])
    got = interpolate_profile(height, values, np.arange(0.0, 6.0))
    np.testing.assert_allclose(got, [10.0, 10.0, 20.0, 30.0, 30.0, 30.0])


def test_completion_attributes_summarise_source_extent_and_edge_carries():
    from jumacs.harmonize import _edge_completion_attrs
    altitude = np.zeros((2, 4, 2))
    altitude[:] = np.array([0.0, 10.0, 20.0, 30.0])[None, :, None]
    values = np.array([[[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                       [[np.nan, np.nan], [2.0, np.nan], [4.0, 5.0], [6.0, 7.0]]], dtype="float64")
    attrs = _edge_completion_attrs(altitude, values, 30.0)
    assert attrs["source_valid_bottom_km_min"] == 0.0
    assert attrs["source_valid_bottom_km_max"] == 20.0
    assert attrs["source_valid_top_km"] == 30.0
    assert attrs["lower_completion_km_min"] == 0.0
    assert attrs["lower_completion_km_max"] == 20.0
    assert attrs["upper_completion_km"] == 0.0


def test_completion_attributes_are_independent_of_native_level_order():
    from jumacs.harmonize import _edge_completion_attrs
    altitude = np.zeros((1, 4, 1))
    altitude[0, :, 0] = [0.0, 10.0, 20.0, 30.0]
    values = np.array([[[np.nan], [2.0], [4.0], [6.0]]], dtype="float64")
    ascending = _edge_completion_attrs(altitude, values, 40.0)
    descending = _edge_completion_attrs(altitude[:, ::-1, :].copy(), values[:, ::-1, :].copy(), 40.0)
    assert ascending == descending
    assert ascending["source_valid_bottom_km"] == 10.0
    assert ascending["source_valid_top_km"] == 30.0
    assert ascending["upper_completion_km"] == 10.0


def capped_harmonized(monkeypatch, cap):
    """The two-grid synthetic product harmonized under a forced product cap on the model."""
    import jumacs.harmonize as harmonize_module
    monkeypatch.setattr(harmonize_module, "PRODUCT_TOP_CAP_KM", {"IPSL-CM6A-ATM-LR-REPROBUS": cap})
    model = "IPSL-CM6A-ATM-LR-REPROBUS"
    grids = {}
    parts = [assign_latitude_grid(part("o3", LAT, 1.0), grids, model, "o3"),
             assign_latitude_grid(height_part(LAT), grids, model, "zg")]
    product = assemble_product(model, 2000, 2001, parts)
    return _harmonize_open_dataset(product, model, load_config(model), 2000, 2001, 5, 1)


def test_a_product_cap_truncates_the_published_grid_and_says_so_in_metadata(monkeypatch):
    prepared = capped_harmonized(monkeypatch, 30.0)["dataset"]
    assert float(prepared.altitude[0]) == 0.0 and float(prepared.altitude[-1]) == 30.0
    assert prepared.sizes["altitude"] == 31
    assert prepared.attrs["model_valid_top_km"] == 30.0
    assert prepared.attrs["source_valid_top_km"] == pytest.approx(round(_isa_height_m(NATIVE).max() / 1000.0, 3))
    assert prepared.attrs["vertical_extent_policy"] == "source extent capped at 30 km for IPSL-CM6A-ATM-LR-REPROBUS"
    assert prepared.attrs["lower_boundary_completion"] == "constant nearest valid value"
    assert prepared.attrs["upper_boundary_completion"] == "constant nearest valid value to model top"
    assert prepared.attrs["internal_gap_filling"] == 0  # netCDF has no boolean; 0 is the false value
    assert np.isfinite(prepared["o3_mean"].values).all()
    assert cf.validate_product(prepared, names=("o3",), kind="harmonized") == []


def test_an_uncapped_model_keeps_its_full_extent_and_carries_no_cap_metadata():
    harmonized = harmonized_grids_dataset()[1]
    assert float(harmonized.altitude[-1]) == float(np.floor(_isa_height_m(NATIVE).max() / 1000.0))
    for key in ("vertical_extent_policy", "upper_extrapolation", "source_valid_top_km",
                "gradient_extrapolation"):
        assert key not in harmonized.attrs


def test_the_published_cap_is_waccmx_only_and_sits_at_150_km():
    from jumacs.harmonize import PRODUCT_TOP_CAP_KM
    assert PRODUCT_TOP_CAP_KM == {"WACCM-X": 150.0}
