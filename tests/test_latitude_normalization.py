"""Nearly identical latitude grids must become one axis, not a union."""
import warnings

import numpy as np
import pytest
import xarray as xr

import jumacs.climatology as climatology_module
from jumacs import cf
from jumacs.climatology import (LATITUDE_SNAP_TOLERANCE_DEGREES, assemble_product, build_climatology,
                                normalize_latitude, variable_statistics, write_product)
from jumacs.config import load_config, vertical_grid

GRID = vertical_grid()
LEVELS = np.asarray(GRID["levels"], float)
# A vertical grid of the model itself, which the application grid deliberately is not.
NATIVE = np.logspace(5.0, -0.5, 67)

# A 96-point zonal axis of the SOCOL type, with the extrema seen in the archive.
LAT = np.linspace(-88.57216851400727, 88.57216851400727, 96)


def jitter(lat, scale):
    return lat + scale * np.where(np.arange(lat.size) % 2, 1.0, -1.0)


def part(name, lat, value, levels=LEVELS, level="pressure"):
    stats = xr.Dataset(
        {statistic: (("month", level, "lat"), np.full((12, len(levels), lat.size), value, float))
         for statistic in cf.STATISTIC_ORDER},
        coords={"month": list(range(1, 13)), level: (level, levels, {"units": "Pa"}), "lat": ("lat", lat)})
    return variable_statistics(name, stats, {"units": "mol mol-1"}, {})


def index_part(name, lat, value, count, level="lev"):
    stats = xr.Dataset(
        {statistic: (("month", level, "lat"), np.full((12, count, lat.size), value, float))
         for statistic in cf.STATISTIC_ORDER},
        coords={"month": list(range(1, 13)), level: (level, np.arange(count, dtype=float), {"units": "1"}),
                "lat": ("lat", lat)})
    return variable_statistics(name, stats, {"units": "K"}, {})


def hybrid_field(lat, count, level="lev"):
    values = np.broadcast_to(np.logspace(5.0, 0.0, count)[None, :, None], (12, count, lat.size)).copy()
    return xr.DataArray(values, dims=("month", level, "lat"),
                        coords={"month": list(range(1, 13)), "lat": ("lat", lat)})


def test_nearly_identical_grids_are_snapped_instead_of_unioned():
    parts = [part("o3", LAT, 3.0), part("br", jitter(LAT, 7.6e-13), 5.0)]
    product = assemble_product("SOCOL", 2000, 2001, parts)
    assert product.sizes["lat"] == LAT.size
    assert np.array_equal(product.lat.values, LAT)
    for name, value in (("o3", 3.0), ("br", 5.0)):
        field = product[f"{name}_mean"]
        assert field.dims == ("time", "pressure", "lat")
        assert np.isfinite(field.values).all()
        assert np.array_equal(field.values, np.full(field.shape, value, field.dtype))


def test_three_socol_like_grids_keep_one_ninety_six_point_axis():
    float32_roundtrip = LAT.astype(np.float32).astype(np.float64)
    assert np.abs(float32_roundtrip - LAT).max() < LATITUDE_SNAP_TOLERANCE_DEGREES
    grids = [LAT, jitter(LAT, 7.6e-13), jitter(LAT, 5.2e-7) + 2.0e-6, float32_roundtrip]
    parts = [part(name, lat, float(index)) for index, (name, lat) in
             enumerate(zip(("ch4", "o3", "ta", "h2o"), grids))]
    product = assemble_product("SOCOL", 2000, 2001, parts)
    assert product.sizes["lat"] == 96
    assert np.array_equal(product.lat.values, LAT)
    for index, name in enumerate(("ch4", "o3", "ta", "h2o")):
        field = product[f"{name}_mean"]
        assert field.shape == (12, LEVELS.size, 96)
        assert np.isfinite(field.values).all()
        assert np.array_equal(field.values, np.full(field.shape, float(index), field.dtype))


def test_a_latitude_grid_outside_tolerance_is_refused():
    mismatched = jitter(LAT, 7.6e-13)
    mismatched[40] += 1e-3
    with pytest.raises(RuntimeError) as error:
        assemble_product("SOCOL", 2000, 2001, [part("o3", LAT, 1.0), part("ta", mismatched, 2.0)])
    message = str(error.value)
    assert "latitude grid for variable 'ta' of SOCOL differs from the canonical model grid" in message
    assert "'o3'" in message
    assert f"beyond the {LATITUDE_SNAP_TOLERANCE_DEGREES} degree tolerance" in message
    assert "96 latitude points" in message
    assert "maximum absolute coordinate difference" in message


def test_a_different_latitude_point_count_is_refused():
    with pytest.raises(RuntimeError, match=r"variable 'ta' of SOCOL has 95 latitude points, the canonical grid "
                                           r"taken from 'o3' has 96"):
        assemble_product("SOCOL", 2000, 2001, [part("o3", LAT, 1.0), part("ta", LAT[1:], 2.0)])


def test_a_reversed_latitude_axis_is_refused_instead_of_reordered():
    with pytest.raises(RuntimeError, match="ordered differently from the canonical grid"):
        assemble_product("SOCOL", 2000, 2001, [part("o3", LAT, 1.0), part("ta", LAT[::-1].copy(), 2.0)])


def test_already_identical_latitude_coordinates_pass_through_untouched():
    parts = [part("o3", LAT, 1.0), part("br", LAT.copy(), 2.0)]
    normalized = normalize_latitude(parts, "SOCOL")
    assert len(normalized) == 2
    for original, snapped in zip(parts, normalized):
        assert np.array_equal(original.lat.values, snapped.lat.values)
        assert snapped.lat.attrs == original.lat.attrs
    product = assemble_product("SOCOL", 2000, 2001, normalized)
    assert product.sizes["lat"] == 96
    assert np.array_equal(product.lat.values, LAT)


SOCOL_PLEV = np.logspace(5.0, -0.5, 39)


def socol_pressures(pressure_field):
    return {"plev": {"kind": "coordinate", "pressure": xr.DataArray(SOCOL_PLEV.copy(), dims=("plev",)),
                     "source_variable": "br", "coverage": (2000, 2001), "units": "Pa"},
            "lev": {"kind": "hybrid", "pressure": pressure_field, "source_variable": "ta",
                    "coverage": (2000, 2001), "units": "Pa", "source_level": "lev"}}


def test_a_published_pressure_field_of_roundoff_latitudes_keeps_the_canonical_axis():
    parts = [part("br", LAT, 5.0, levels=SOCOL_PLEV, level="plev"), index_part("ta", LAT, 250.0, 47)]
    product = assemble_product("SOCOL", 2000, 2001, parts, (), ["br", "ta"],
                               socol_pressures(hybrid_field(jitter(LAT, 4.0e-6), 47)))
    assert product.sizes["lat"] == 96
    assert len(set(product.lat.values.tolist())) == 96
    assert np.array_equal(product.lat.values, LAT)
    assert np.array_equal(product.air_pressure_lev.lat.values, LAT)
    assert product.air_pressure_lev.dims == ("time", "lev", "lat")
    assert product.air_pressure_lev.shape == (12, 47, 96)
    assert np.isfinite(product.air_pressure_lev.values).all()
    for name, value, level in (("br_mean", 5.0, "plev"), ("ta_mean", 250.0, "lev")):
        field = product[name]
        assert field.dims == ("time", level, "lat")
        assert np.isfinite(field.values).all()
        assert np.array_equal(field.values, np.full(field.shape, value, field.dtype))
    assert product.attrs["latitude_count"] == 96
    for name in ("br", "ta"):
        counts = product[f"{name}_n_years"].values
        assert counts.dtype == np.int16 and (counts == counts.flat[0]).all()
    cf.assert_product(product, names=("br", "ta"), grid=GRID)


def test_a_published_pressure_field_of_other_latitudes_is_refused_instead_of_unioned():
    beyond = jitter(LAT, 4.0e-6)
    beyond[40] += 1e-3
    with pytest.raises(RuntimeError) as error:
        assemble_product("SOCOL", 2000, 2001, [part("br", LAT, 5.0, levels=SOCOL_PLEV, level="plev"),
                                              index_part("ta", LAT, 250.0, 47)], (), ["br", "ta"],
                         socol_pressures(hybrid_field(beyond, 47)))
    message = str(error.value)
    assert "latitude grid for variable 'air_pressure_lev' of SOCOL differs from the canonical model grid" in message
    assert f"beyond the {LATITUDE_SNAP_TOLERANCE_DEGREES} degree tolerance" in message
    assert "96 latitude points" in message


def test_a_published_pressure_field_axis_of_another_size_is_refused():
    with pytest.raises(RuntimeError, match=r"variable 'air_pressure_lev' of SOCOL has 95 latitude points, the "
                                           r"canonical grid taken from 'br' has 96"):
        assemble_product("SOCOL", 2000, 2001, [part("br", LAT, 5.0, levels=SOCOL_PLEV, level="plev"),
                                              index_part("ta", LAT, 250.0, 47)], (), ["br", "ta"],
                         socol_pressures(hybrid_field(LAT[1:], 47)))


def test_roundoff_latitudes_of_a_published_pressure_field_keep_the_counts_integral(tmp_path, monkeypatch):
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    parts = [part("br", LAT, 5.0, levels=SOCOL_PLEV, level="plev"), index_part("ta", LAT, 250.0, 47)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = write_product("SOCOL", 2000, 2001, parts, (), socol_pressures(hybrid_field(jitter(LAT, 4.0e-6), 47)))
    assert [str(item.message) for item in caught if "as an integer dtype" in str(item.message)] == []
    with xr.open_dataset(path, decode_cf=False) as written:
        assert written.sizes["lat"] == 96
        assert np.array_equal(written.lat.values, LAT)
        assert np.array_equal(written.air_pressure_lev.lat.values, LAT)
        for name in ("br", "ta"):
            assert written[f"{name}_n_years"].dtype == np.int16
            assert np.isfinite(written[f"{name}_mean"].values).all()
            assert np.isfinite(written[f"{name}_n_years"].values).all()
        assert np.isfinite(written.air_pressure_lev.values).all()


def write_zonal(path, name, lat, value):
    ds = xr.Dataset(
        {name: (("time", "plev", "lat"), np.broadcast_to(np.array(value, float), (24, NATIVE.size, lat.size)).copy(),
                {"units": "mol mol-1"})},
        coords={"time": xr.date_range("2000-01", periods=24, freq="MS", use_cftime=True),
                "plev": ("plev", NATIVE, {"units": "Pa", "standard_name": "air_pressure"}),
                "lat": ("lat", lat)})
    ds.to_netcdf(path)


def test_socol_like_zonal_grids_yield_one_axis_in_the_final_product(tmp_path, monkeypatch):
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    zonal = tmp_path / load_config("SOCOL")["paths"]["zonal"]
    zonal.mkdir(parents=True)
    write_zonal(zonal / "o3_monthly_zonal.nc", "o3", LAT, 3.0e-6)
    write_zonal(zonal / "ta_monthly_zonal.nc", "ta", LAT.astype(np.float32).astype(np.float64), 250.0)
    product = build_climatology("SOCOL", 2000, 2001)
    with xr.open_dataset(product, decode_cf=False) as ds:
        cf.assert_product(ds, names=("o3", "ta"), grid=GRID)
        assert ds.sizes["lat"] == 96
        assert np.array_equal(ds.lat.values, LAT)
        assert ds.lat.dtype == np.dtype("float64")
        for name in ("o3", "ta"):
            assert np.isfinite(ds[f"{name}_mean"].values).mean() == 1.0
            assert ds[f"{name}_mean"].dims == ("time", "plev", "lat")
        assert ds.sizes["plev"] == NATIVE.size and np.allclose(ds.plev.values, NATIVE)
        assert ds.o3_mean.attrs["on_application_pressure_grid"] == "false"
