"""Models with more than one native latitude grid publish each grid as its own dimension."""
import numpy as np
import pytest
import xarray as xr

import jumacs.climatology as climatology_module
from jumacs import cf, inventory, remap
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

# The 73-point axis of the NIWA-UKCA2 archive and the 72-point axis of its EP fluxes.
LAT = np.linspace(-88.57216851400727, 88.57216851400727, 96)
LAT2 = np.linspace(-87.5, 87.5, 72)
LAT3 = np.linspace(-86.0, 86.0, 25)


def gridded_parts(specs, model="NIWA-UKCA2"):
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


def build_niwa(tmp_path, monkeypatch, grids):
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    zonal = tmp_path / load_config("NIWA-UKCA2")["paths"]["zonal"]
    zonal.mkdir(parents=True)
    for name, (lat, value) in grids.items():
        write_zonal(zonal / f"{name}_monthly_zonal.nc", name, lat, value)
    product = build_climatology("NIWA-UKCA2", 2000, 2001)
    with xr.open_dataset(product, decode_cf=False) as written:
        return written.load()


def test_a_second_native_latitude_grid_becomes_its_own_dimension(tmp_path, monkeypatch):
    ds = build_niwa(tmp_path, monkeypatch, {"o3": (LAT, 3.0e-6), "epfy": (LAT2, 1.0e-3)})
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
    ds = build_niwa(tmp_path, monkeypatch, {"o3": (LAT, 3.0e-6), "epfy": (LAT2, 1.0e-3), "zg": (LAT3, 100.0)})
    assert ds.sizes["lat"] == LAT2.size
    assert ds.sizes["lat_2"] == LAT.size
    assert ds.sizes["lat_3"] == LAT3.size
    assert ds.epfy_mean.attrs["latitude_axis"] == "lat"
    assert ds.o3_mean.attrs["latitude_axis"] == "lat_2"
    assert ds.zg_mean.attrs["latitude_axis"] == "lat_3"
    assert ds.attrs["latitude_count"] == LAT2.size
    assert ds.attrs["latitude_counts"] == f"lat={LAT2.size}; lat_2={LAT.size}; lat_3={LAT3.size}"


def test_near_identical_grids_still_collapse_to_one_dimension(tmp_path, monkeypatch):
    ds = build_niwa(tmp_path, monkeypatch, {"o3": (LAT, 3.0e-6), "epfy": (jitter(LAT, 4.0e-6), 1.0e-3)})
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
    ds = build_niwa(tmp_path, monkeypatch, {"o3": (LAT, 3.0e-6), "epfy": (LAT2, 1.0e-3), "zg": (beyond, 100.0)})
    assert ds.sizes["lat"] == 72 and ds.sizes["lat_2"] == 96 and ds.sizes["lat_3"] == 72
    assert np.array_equal(ds.lat.values, LAT2) and np.array_equal(ds.lat_3.values, beyond)
    assert ds.zg_mean.dims == ("time", "plev", "lat_3")
    assert ds.zg_mean.attrs["latitude_axis"] == "lat_3"
    assert np.isfinite(ds.zg_mean.values).all() and np.isfinite(ds.epfy_mean.values).all()


def test_a_latitude_axis_beyond_tolerance_below_the_snapping_level_is_refused():
    beyond = jitter(LAT, 7.6e-13)
    beyond[40] += 1e-3
    with pytest.raises(RuntimeError, match="differs from the canonical model grid"):
        assemble_product("NIWA-UKCA2", 2000, 2001, [part("o3", LAT, 1.0), part("epfy", beyond, 2.0)])


def test_remap_places_every_grid_on_the_canonical_application_bands(tmp_path, monkeypatch):
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    product = assemble_product("NIWA-UKCA2", 2000, 2001, gridded_parts([("o3", LAT, 1.0), ("epfy", LAT2, 2.0)]))
    assert product.o3_mean.attrs["latitude_axis"] == "lat"
    assert product.epfy_mean.attrs["latitude_axis"] == "lat_2"
    assert product.attrs["latitude_counts"] == f"lat={LAT.size}; lat_2={LAT2.size}"
    remapped = remap.remap_product(product)
    assert "lat_2" not in remapped.dims and "lat_2" not in remapped.coords
    for name, value in (("o3", 1.0), ("epfy", 2.0)):
        field = remapped[f"{name}_mean"]
        assert field.dims == ("time", "pressure", "lat")
        assert "latitude_axis" not in field.attrs
        finite = np.isfinite(field.values)
        assert finite.any()
        np.testing.assert_allclose(field.values[finite], value)
    assert remapped.sizes["lat"] == 36
    assert remapped.attrs["native_latitude_counts"] == product.attrs["latitude_counts"]
    assert cf.validate_product(remapped, names=("o3", "epfy"), kind="application") == []


def test_the_application_contract_still_refuses_a_latitude_axis_named_lat_2():
    product = assemble_product("NIWA-UKCA2", 2000, 2001, gridded_parts([("o3", LAT, 1.0), ("epfy", LAT2, 2.0)]))
    remapped = remap.remap_product(product).rename({"lat": "lat_2"})
    assert cf.validate_product(remapped, names=("o3", "epfy"), kind="application") != []


def test_the_inventory_recognizes_fields_on_a_second_latitude_grid(tmp_path):
    lat = np.linspace(-85.0, 85.0, 72)
    ds = xr.Dataset(
        {"o3_mean": (("time", "lat_2"), np.zeros((2, lat.size), np.float32), {"units": "mol mol-1"}),
         "epfy_mean": (("time", "pressure", "lat_2"), np.zeros((2, 3, lat.size), np.float32), {"units": "m2 s-2"})},
        coords={"time": [0.0, 1.0], "pressure": ("pressure", [100000., 1000., 10.]), "lat_2": ("lat_2", lat)})
    path = tmp_path / "niwa_ukca2_refd1_2000_2001.nc"
    ds.to_netcdf(path)
    skipped = []
    fields = inventory._fields(path, skipped)
    assert skipped == []
    assert fields["o3"]["dimensionality"] == "2D"
    assert fields["epfy"]["dimensionality"] == "3D"
