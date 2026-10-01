import numpy as np
import pytest
import xarray as xr

import jumacs.config as config_module
from jumacs.climatology import (MissingPressureCoordinate, build_climatology, combined_variables,
                                monthly_climatology, on_common_grid, write_combined)
from jumacs.config import vertical_grid
from jumacs.vertical import interpolate_log_pressure, native_pressure, regrid_to_common_grid

GRID = vertical_grid()


def level_near(pressure_pa):
    """The common-grid label closest to a pressure, because labels are exact grid values."""
    return min(GRID["levels"], key=lambda value: abs(np.log10(value) - np.log10(pressure_pa)))


MIDDLE = level_near(3162.0)
ABOVE_TOP = level_near(562.3)


def profile(levels, periods=12):
    return np.broadcast_to(np.asarray(levels, float)[:, None], (periods, len(levels), 1)).copy()


def times(periods=12, start="2000-01"):
    return xr.date_range(start, periods=periods, freq="MS", use_cftime=True)


def dataset(name, values, pressure, coords=None, pressure_name="air_pressure", level="lev", units="Pa"):
    values = np.asarray(values, float)
    size = values.ndim
    times_ = times(values.shape[0])
    dims = ["time", level, "lat"] if size == 3 else ["time", "lat"]
    data = xr.Dataset({name: (dims, values, {"units": "mol mol-1"})},
                      coords={"time": times_, "lat": [0.] if size == 3 else [-45., 45.]})
    if pressure is None:
        return data
    pressure = np.asarray(pressure, float)
    if coords is not None:
        data = data.assign_coords({level: (level, pressure, dict(coords))})
    else:
        data[pressure_name] = (("time", level, "lat"), pressure, {"units": units})
    return data


def test_common_grid_is_explicit_log_uniform_and_monotonic():
    levels = np.asarray(GRID["levels"], float)
    assert GRID["coordinate"] == "pressure" and GRID["units"] == "Pa"
    assert GRID["interpolation"] == "linear_log_pressure" and GRID["extrapolation"] == "none"
    assert levels.size == 101 and levels[0] == 1e5 and levels[-1] == pytest.approx(1e-5)
    assert all(a > b for a, b in zip(levels, levels[1:]))
    exponents = np.log10(levels)
    assert exponents[0] == 5.0 and exponents[-1] == pytest.approx(-5.0)
    steps = np.diff(exponents)
    assert np.allclose(steps, -0.1, rtol=0.0, atol=1e-9)
    assert np.allclose(1.0 / -steps, 10.0, rtol=1e-8)
    assert np.allclose(levels[:-1] / levels[1:], 10 ** 0.1, rtol=1e-9)
    for decade in (1e4, 1e3, 1e2, 1.0, 1e-2, 1e-3, 1e-4):
        assert any(abs(level - decade) <= 1e-9 * decade for level in levels)


@pytest.mark.parametrize("section,message", [
    ({"units": "hPa", "levels": [1e5, 1e3]}, "units must be Pa"),
    ({"levels": [1e5, 1e4, 1e5]}, "strictly monotonic"),
    ({"levels": [1e5, 0.0]}, "positive"),
    ({"levels": [1e5]}, "at least two"),
    ({"interpolation": "linear", "levels": [1e5, 1e3]}, "linear_log_pressure"),
    ({"extrapolation": "linear", "levels": [1e5, 1e3]}, "extrapolation must be none"),
    ({}, "needs a vertical_grid section"),
])
def test_common_grid_configuration_is_validated(monkeypatch, section, message):
    monkeypatch.setattr(config_module, "reference_period", lambda: {"vertical_grid": section})
    with pytest.raises(ValueError, match=message):
        config_module.vertical_grid()


def test_native_pressure_recovers_hybrid_and_fixed_levels():
    hybrid = dataset("o3", np.zeros((2, 3, 1)), profile([1e5, 1e4, 1e3], 2))
    assert native_pressure(hybrid, hybrid.o3).dims == ("time", "lev", "lat")
    hectopascal = dataset("ta", np.zeros((2, 2, 1)), [1000., 100.], level="plev",
                          coords={"units": "hPa", "standard_name": "air_pressure"})
    recovered = native_pressure(hectopascal, hectopascal.ta)
    assert recovered.dims == ("plev",)
    assert np.allclose(recovered.values, [1e5, 1e4]) and recovered.attrs["units"] == "Pa"
    nameless = dataset("ta", np.zeros((2, 2, 1)), [1000., 100.], coords={"units": "m"})
    assert native_pressure(nameless, nameless.ta) is None


def test_interpolation_precedes_the_statistics():
    pressure = np.where(np.arange(12)[:, None, None] % 2 == 0,
                       profile([1e5, 1e4, 1e3]), profile([1e4, 1e3, 1e2]))
    ds = dataset("o3", np.broadcast_to(np.array([0., 1., 2.])[None, :, None], (12, 3, 1)), pressure)
    field, provenance = on_common_grid(ds, "o3", 2000, 2000, GRID)
    assert field.dims == ("time", "pressure", "lat")
    assert provenance["regridded_to_common_pressure_grid"] == "true"
    climatology = monthly_climatology(field, 2000, 2000)
    january = climatology["mean"].sel(month=1, pressure=MIDDLE).item()
    february = climatology["mean"].sel(month=2, pressure=MIDDLE).item()
    native_mean = ds.o3.mean("time")
    pressure_mean = ds.air_pressure.mean("time")
    after_the_mean = interpolate_log_pressure(native_mean, pressure_mean, GRID["levels"]).sel(pressure=MIDDLE).isel(lat=0).item()
    assert january == pytest.approx(1.5, rel=1e-4)
    assert february == pytest.approx(0.5, rel=1e-4)
    assert after_the_mean == pytest.approx(1.2404, abs=2e-3)
    assert abs(january - after_the_mean) > 0.2 and abs(february - after_the_mean) > 0.2
    assert abs(climatology["mean"].sel(pressure=MIDDLE).mean().item() - after_the_mean) > 0.2
    assert climatology["n_years"].sel(month=1, pressure=MIDDLE).item() == 1


def test_targets_outside_a_profile_stay_nan_and_count_zero_years():
    ds = dataset("o3", np.broadcast_to(np.array([0., 1.])[None, :, None], (12, 2, 1)), profile([1e5, 1e3]))
    field, _ = on_common_grid(ds, "o3", 2000, 2000, GRID)
    climatology = monthly_climatology(field, 2000, 2000)
    above_top = np.asarray(GRID["levels"]) < 1e3
    mean = climatology["mean"].values[:, above_top, :]
    assert np.isnan(mean).all()
    assert climatology["n_years"].values[:, above_top, :].max() == 0
    assert climatology["mean"].sel(month=1, pressure=1e5).item() == pytest.approx(0.0)
    assert climatology["n_years"].sel(month=1, pressure=1e3).item() == 1
    assert climatology["n_years"].sel(month=1, pressure=ABOVE_TOP).item() == 0


def test_native_pressure_ordering_and_units_do_not_matter():
    common = {"coords": {"units": "Pa", "standard_name": "air_pressure"}, "level": "plev"}
    descending = dataset("ta", [[[0.], [1.], [2.]]] * 2, [1e5, 1e4, 1e3], **common)
    ascending = dataset("ta", [[[2.], [1.], [0.]]] * 2, [1e3, 1e4, 1e5], **common)
    descending = descending.transpose("time", "plev", "lat")
    ascending = ascending.transpose("time", "plev", "lat")
    def moved(source):
        return regrid_to_common_grid(source.ta, native_pressure(source, source.ta), GRID["levels"]).transpose("time", "pressure", "lat")

    low, high = moved(descending), moved(ascending)
    assert np.allclose(np.nan_to_num(low.values, nan=-1), np.nan_to_num(high.values, nan=-1))
    assert np.isfinite(low.values).sum() > 0
    assert low.pressure.attrs["units"] == "Pa" and low.pressure.attrs["positive"] == "down"
    assert low.attrs["units"] == "mol mol-1" and low.attrs["vertical_extrapolation"] == "none"


def test_broken_pressure_values_are_dropped_not_interpolated():
    pressure = np.array([[[1e5], [1e4], [1e3]], [[np.nan], [np.nan], [1e3]]])
    ds = dataset("o3", [[[0.], [1.], [2.]], [[5.], [5.], [5.]]], pressure)
    field, provenance = on_common_grid(ds, "o3", 2000, 2000, GRID)
    climatology = monthly_climatology(field, 2000, 2000)
    assert climatology["mean"].sel(month=1, pressure=MIDDLE).item() == pytest.approx(1.5, rel=1e-4)
    assert climatology["n_years"].sel(month=1, pressure=MIDDLE).item() == 1
    assert provenance["native_pressure_min_pa"] == 1000.0
    assert provenance["native_pressure_max_pa"] == 100000.0


def test_three_dimensional_field_without_any_pressure_information_is_rejected():
    ds = dataset("o3", np.zeros((2, 3, 1)), np.zeros((2, 3, 1)), pressure_name="not_pressure")
    ds = ds.drop_vars("not_pressure")
    ds = ds.assign_coords(lev=("lev", [0., 0.5, 1.], {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"}))
    with pytest.raises(MissingPressureCoordinate, match="common pressure grid"):
        on_common_grid(ds, "o3", 2000, 2000, GRID)


def test_two_dimensional_fields_keep_their_own_shape(tmp_path, monkeypatch):
    import jumacs.climatology as climatology_module
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    zonal = tmp_path / "data/processed/GEOSCCM/refD1"
    zonal.mkdir(parents=True)
    mixed = xr.Dataset({
        "o3": (("time", "plev", "lat"), np.broadcast_to(np.array([0., 1.])[None, :, None], (24, 2, 2)).copy(), {"units": "mol mol-1"}),
        "trop": (("time", "lat"), np.full((24, 2), 25000.), {"units": "Pa"}),
    }, coords={"time": times(24), "plev": ("plev", [1e5, 1e3], {"units": "Pa", "standard_name": "air_pressure"}), "lat": [-45., 45.]})
    mixed.to_netcdf(zonal / "o3_monthly_zonal.nc")
    mixed[["trop"]].to_netcdf(zonal / "trop_monthly_zonal.nc")
    outputs = build_climatology("GEOSCCM", 2000, 2001, ["o3", "trop"])
    with xr.open_dataset(outputs[0]) as three_d:
        assert set(three_d["mean"].dims) == {"month", "pressure", "lat"}
        assert three_d.sizes["pressure"] == 101
        assert three_d.attrs["regridded_to_common_pressure_grid"] == "true"
        assert three_d.attrs["native_level_dimension"] == "plev"
        assert three_d.attrs["vertical_coordinate"] == "pressure (Pa), common grid"
        assert three_d["mean"].attrs["vertical_extrapolation"] == "none"
    with xr.open_dataset(outputs[1]) as two_d:
        assert set(two_d["mean"].dims) == {"month", "lat"}
        assert "pressure" not in two_d.dims
        assert two_d.attrs["regridded_to_common_pressure_grid"] == "false"
        assert two_d.attrs["vertical_coordinate"] == "none (two-dimensional field)"
        assert "two-dimensional" in two_d.attrs["vertical_treatment"]
    with xr.open_dataset(outputs[2]) as combined:
        assert set(combined.o3_mean.dims) == {"month", "pressure", "lat"}
        assert set(combined.trop_mean.dims) == {"month", "lat"}
        assert combined.sizes["pressure"] == 101
        assert combined.attrs["vertical_level_count"] == len(GRID["levels"]) == 101
        assert not [dim for dim in combined.dims if dim not in ("month", "pressure", "lat")]


def test_combined_product_refuses_separate_vertical_dimensions(tmp_path, monkeypatch):
    import jumacs.climatology as climatology_module
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    folder = tmp_path / "products/climatology/CMAM"
    folder.mkdir(parents=True)
    for name, levels in (("o3", [1e5, 1e4]), ("br", [1e5, 1e4])):
        xr.Dataset({f"{name}_mean": (("month", f"{name}_plev", "lat"), np.ones((1, 2, 1)))},
                   coords={"month": [1], f"{name}_plev": (f"{name}_plev", levels), "lat": [0.]},
                   attrs={"source_variable": name}).to_netcdf(folder / f"jumacs_cmam_refd1_{name}_climatology_2000-2001.nc")
    with pytest.raises(RuntimeError, match="per-variable vertical dimensions"):
        write_combined("CMAM", 2000, 2001, ["o3", "br"])


def test_shared_coordinate_survives_the_combined_merge():
    coords = {"month": [1], "pressure": ("pressure", [1e5, 1e4], {"units": "Pa"}), "lat": [0.]}
    parts = [combined_variables(name, xr.Dataset({"mean": (("month", "pressure", "lat"), np.ones((1, 2, 1)))}, coords=coords))
             for name in ("o3", "br")]
    merged = xr.merge(parts, join="outer", compat="override")
    assert list(merged.dims) == ["month", "pressure", "lat"]
    assert set(merged.data_vars) == {"o3_mean", "br_mean"}
