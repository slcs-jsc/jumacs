import math

import numpy as np
import pytest
import xarray as xr

import jumacs.climatology as climatology_module
import jumacs.config as config_module
from jumacs import cf
from jumacs.climatology import (MissingPressureCoordinate, assemble_product, build_climatology, monthly_climatology,
                                native_monthly_field, pressure_climatology, product_name, register_pressure,
                                variable_statistics)
from jumacs.config import load_config, vertical_grid
from jumacs.vertical import (interpolate_log_pressure, native_pressure, pressure_report, product_pressure,
                             regrid_to_common_grid)

GRID = vertical_grid()
LEVELS = np.asarray(GRID["levels"], float)


def level_near(pressure_pa):
    """The application-grid level closest to a pressure, because grid levels are exact values."""
    return min(LEVELS, key=lambda value: abs(np.log10(value) - np.log10(pressure_pa)))


PROBE = level_near(3162.0)
LOG_PROBE = float(np.log10(PROBE))


def profile(levels, periods=12):
    return np.broadcast_to(np.asarray(levels, float)[:, None], (periods, len(levels), 1)).copy()


def times(periods=12, start="2000-01"):
    return xr.date_range(start, periods=periods, freq="MS", use_cftime=True)


def zonal_dataset(name, values, periods=12, level="lev", lats=(0.,), units="mol mol-1"):
    """A monthly zonal series as the model produced it: native levels, no grid yet."""
    values = np.asarray(values, float)
    dims = ("time", level, "lat") if values.ndim == 3 else ("time", "lat")
    return xr.Dataset({name: (dims, values, {"units": units})},
                      coords={"time": times(periods), "lat": (("lat",), np.asarray(lats, float))})


def with_pressure(data, levels, units="Pa", level="lev"):
    return data.assign_coords(**{level: (level, np.asarray(levels, float),
                                         {"standard_name": "air_pressure", "units": units})})


def with_hybrid_pressure(data, pressure, units="Pa", level="lev"):
    data["air_pressure"] = (("time", level, "lat"), np.asarray(pressure, float), {"units": units})
    return data


def store(workspace, model, name, data):
    directory = workspace / load_config(model)["paths"]["zonal"]
    directory.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(directory / f"{name}_monthly_zonal.nc")


def test_the_application_grid_has_exact_endpoints_and_a_resolution_derived_length():
    assert GRID["coordinate"] == "pressure" and GRID["units"] == "Pa"
    assert GRID["interpolation"] == "linear_log_pressure" and GRID["extrapolation"] == "none"
    assert LEVELS[0] == 1e5 and LEVELS[-1] == 0.002
    assert all(a > b for a, b in zip(LEVELS, LEVELS[1:])) and LEVELS.min() > 0
    assert LEVELS.dtype == np.float64 or all(isinstance(level, float) for level in LEVELS)
    exponents = np.log10(LEVELS)
    assert exponents[0] == 5.0 and exponents[-1] == pytest.approx(np.log10(0.002))
    steps = np.diff(exponents)
    assert np.allclose(steps, GRID["delta_log10_pressure"], rtol=0.0, atol=1e-12)
    assert np.allclose(LEVELS[:-1] / LEVELS[1:], 10 ** -GRID["delta_log10_pressure"], rtol=1e-9)
    assert np.allclose(LEVELS[1:] / LEVELS[:-1], 10 ** GRID["delta_log10_pressure"], rtol=1e-9)


def test_level_count_and_spacing_follow_the_configured_resolution_not_a_written_down_length():
    settings = config_module.reference_period()["vertical_grid"]
    decades = np.log10(settings["max_pressure_pa"] / settings["min_pressure_pa"])
    assert settings["max_pressure_pa"] == 100000.0 and settings["min_pressure_pa"] == 0.002
    assert GRID["intervals"] == round(decades * settings["intervals_per_decade"])
    assert LEVELS.size == GRID["intervals"] + 1
    assert 15.0 <= GRID["intervals"] / decades <= 17.0
    altitude_step_km = 7.0 * np.log(10.0) * abs(GRID["delta_log10_pressure"])
    assert 0.9 <= altitude_step_km <= 1.1
    assert "levels" not in settings


@pytest.mark.parametrize("section,message", [
    ({"units": "hPa", "max_pressure_pa": 1e5, "min_pressure_pa": 1e3, "intervals_per_decade": 10}, "units must be Pa"),
    ({"interpolation": "linear", "max_pressure_pa": 1e5, "min_pressure_pa": 1e3, "intervals_per_decade": 10},
     "linear_log_pressure"),
    ({"extrapolation": "linear", "max_pressure_pa": 1e5, "min_pressure_pa": 1e3, "intervals_per_decade": 10},
     "extrapolation must be none"),
    ({"max_pressure_pa": 1e5, "min_pressure_pa": 1e3, "intervals_per_decade": 10, "levels": [1e5, 1e3]},
     "no longer accepted"),
    ({"max_pressure_pa": 1e5, "min_pressure_pa": 1e3}, "missing intervals_per_decade"),
    ({"max_pressure_pa": 1e5, "min_pressure_pa": 1e3, "intervals_per_decade": 0}, "at least 1"),
    ({"max_pressure_pa": 1e3, "min_pressure_pa": 1e5, "intervals_per_decade": 10}, "must be the larger"),
    ({"max_pressure_pa": -1e5, "min_pressure_pa": 1e3, "intervals_per_decade": 10}, "positive"),
    ({"max_pressure_pa": 1e5, "min_pressure_pa": 0.0, "intervals_per_decade": 10}, "positive"),
    ({}, "needs a vertical_grid section"),
])
def test_the_application_grid_configuration_is_validated(monkeypatch, section, message):
    monkeypatch.setattr(config_module, "reference_period", lambda: {"vertical_grid": section})
    with pytest.raises(ValueError, match=message):
        config_module.vertical_grid()


def test_a_written_down_level_list_is_rejected_rather_than_silently_ignored(monkeypatch):
    monkeypatch.setattr(config_module, "reference_period",
                        lambda: {"vertical_grid": {"levels": list(np.logspace(5, -5, 101))}})
    with pytest.raises(ValueError, match="intervals_per_decade"):
        config_module.vertical_grid()


def test_pressure_reporting_keeps_significant_digits_at_both_ends_of_the_grid():
    assert pressure_report(LEVELS.min()) == 0.002 and pressure_report(LEVELS.max()) == 100000.0
    assert f"{pressure_report(LEVELS.min()):g}" == "0.002"
    assert pressure_report(np.float64(12345.6789)) == 12345.7
    assert pressure_report(101325.2646) == 101325.0
    assert pressure_report(0.078812345) == 0.0788123


def test_native_pressure_recovers_hybrid_and_fixed_levels():
    hybrid = with_hybrid_pressure(zonal_dataset("o3", np.zeros((2, 3, 1)), periods=2), profile([1e5, 1e4, 1e3], 2))
    assert native_pressure(hybrid, hybrid.o3).dims == ("time", "lev", "lat")
    hectopascal = with_pressure(zonal_dataset("ta", np.zeros((2, 2, 1)), periods=2, level="plev"), [1000., 100.],
                                units="hPa", level="plev")
    recovered = native_pressure(hectopascal, hectopascal.ta)
    assert recovered.dims == ("plev",)
    assert np.allclose(recovered.values, [1e5, 1e4]) and recovered.attrs["units"] == "Pa"
    nameless = zonal_dataset("ta", np.zeros((2, 2, 1)), periods=2).assign_coords(
        lev=("lev", [1., 2.], {"units": "m"}))
    assert native_pressure(nameless, nameless.ta) is None


def test_a_native_field_describes_its_own_levels_and_never_the_application_grid():
    data = with_hybrid_pressure(zonal_dataset("o3", np.zeros((12, 3, 1))),
                                profile([1e5, 1e4, 1e3]) * np.array([1.0, 1.0, 0.9])[:, None])
    field, pressure, provenance = native_monthly_field(data, "o3", 2000, 2000)
    assert field.dims == ("time", "lev", "lat")
    assert provenance["on_application_pressure_grid"] == "false"
    assert provenance["native_level_dimension"] == "lev" and provenance["native_level_count"] == 3
    assert provenance["native_vertical_coordinate"] == "lev"
    assert provenance["native_pressure_kind"] == "profile" and provenance["native_pressure_units"] == "Pa"
    assert provenance["native_pressure_min_pa"] == 900.0 and provenance["native_pressure_max_pa"] == 100000.0
    hectopascal = with_pressure(zonal_dataset("ta", np.zeros((12, 2, 1)), level="plev"), [1000., 100.],
                                units="hPa", level="plev")
    _, pressure, provenance = native_monthly_field(hectopascal, "ta", 2000, 2000)
    assert provenance["native_pressure_kind"] == "coordinate" and provenance["native_pressure_units"] == "hPa"
    assert provenance["native_pressure_min_pa"] == 10000.0 and provenance["native_pressure_max_pa"] == 100000.0
    flat = zonal_dataset("trop", np.zeros((12, 1)), units="Pa")
    field, pressure, provenance = native_monthly_field(flat, "trop", 2000, 2000)
    assert pressure is None and field.dims == ("time", "lat")
    assert provenance["on_application_pressure_grid"] == "false" and "native_level_count" not in provenance


def test_the_statistics_come_first_and_the_application_grid_only_later():
    pressure = np.concatenate([profile([1e5, 1e4, 1e3], 12), profile([1e4, 1e3, 1e2], 12)])
    data = with_hybrid_pressure(zonal_dataset("o3", np.broadcast_to(np.array([0., 1., 2.])[None, :, None],
                                                                    (24, 3, 1)), periods=24), pressure)
    field, native, provenance = native_monthly_field(data, "o3", 2000, 2001)
    assert provenance["native_pressure_min_pa"] == 100.0 and provenance["native_pressure_max_pa"] == 100000.0
    climatology = monthly_climatology(field, 2000, 2001)
    assert climatology["mean"].dims == ("month", "lev", "lat") and "pressure" not in climatology.dims
    published = pressure_climatology(native)
    top, middle = np.log10([5.5e4, 5.5e3])
    statistics_first = interpolate_log_pressure(climatology["mean"], published, GRID["levels"])
    assert statistics_first.sel(month=1, pressure=PROBE).item() == pytest.approx((top - LOG_PROBE) / (top - middle),
                                                                                 rel=1e-6)
    grid_first = interpolate_log_pressure(field, native, GRID["levels"]).groupby("time.month").mean("time")
    assert grid_first.sel(month=1, pressure=PROBE).item() == pytest.approx(((5.0 - LOG_PROBE) + (4.0 - LOG_PROBE)) / 2,
                                                                           rel=1e-6)
    difference = statistics_first - grid_first
    assert abs(difference.sel(month=1, pressure=PROBE).item()) > 0.2
    assert not np.allclose(statistics_first.values, grid_first.values, equal_nan=True)


def test_levels_the_model_never_reached_stay_nan_after_the_climatology():
    data = with_pressure(zonal_dataset("o3", np.broadcast_to(np.array([0., 1.])[None, :, None], (12, 2, 1))),
                         [1e5, 1e3])
    field, native, _ = native_monthly_field(data, "o3", 2000, 2000)
    climatology = monthly_climatology(field, 2000, 2000)
    assert np.isfinite(climatology["mean"].values).all()
    assert climatology["n_years"].dims == ("month", "lev", "lat") and climatology["n_years"].values.max() == 1
    placed = interpolate_log_pressure(climatology["mean"], native, GRID["levels"])
    above_top = LEVELS < 1e3
    assert np.isnan(placed.values[:, above_top, :]).all()
    assert np.isfinite(placed.values[:, ~above_top, :]).all()
    assert placed.sel(month=1, pressure=1e5).item() == pytest.approx(0.0)
    deepest_inside = LEVELS[LEVELS >= 1e3][-1]
    assert placed.sel(month=1, pressure=deepest_inside).item() == pytest.approx(
        float((np.log(deepest_inside) - np.log(1e5)) / (np.log(1e3) - np.log(1e5))))


def test_a_level_with_a_broken_pressure_is_dropped_and_not_stepped_on():
    pressure = np.array([[[1e5], [np.nan], [1e3]], [[1e5], [1e4], [1e3]]])
    data = with_hybrid_pressure(zonal_dataset("o3", np.broadcast_to(np.array([0., 10., 2.])[None, :, None],
                                                                   (2, 3, 1)), periods=2), pressure)
    field, native, provenance = native_monthly_field(data, "o3", 2000, 2000)
    assert provenance["native_pressure_min_pa"] == 1000.0 and provenance["native_pressure_max_pa"] == 100000.0
    climatology = monthly_climatology(field, 2000, 2000)
    published = pressure_climatology(native)
    assert np.isnan(published.sel(month=1, lev=1).item())
    placed = interpolate_log_pressure(climatology["mean"], published, GRID["levels"])
    january = placed.sel(month=1, pressure=PROBE).item()
    february = placed.sel(month=2, pressure=PROBE).item()
    assert january == pytest.approx(5.0 - LOG_PROBE, rel=1e-6)
    assert february == pytest.approx(10.0 - 8.0 * (4.0 - LOG_PROBE), rel=1e-6)
    assert abs(january - february) > 4.0


def test_the_native_to_grid_interpolation_ignores_level_order_and_units():
    def moved(values, levels, units):
        data = with_pressure(zonal_dataset("ta", values, periods=2, level="plev"), levels, units=units, level="plev")
        data = data.transpose("time", "plev", "lat")
        return regrid_to_common_grid(data.ta, native_pressure(data, data.ta), GRID["levels"]).transpose(
            "time", "pressure", "lat")

    low = moved([[[0.], [1.], [2.]]] * 2, [1e5, 1e4, 1e3], "Pa")
    high = moved([[[2.], [1.], [0.]]] * 2, [1e3, 1e4, 1e5], "Pa")
    assert np.allclose(np.nan_to_num(low.values, nan=-1), np.nan_to_num(high.values, nan=-1))
    assert np.isfinite(low.values).sum() > 0
    assert low.pressure.attrs["units"] == "Pa" and low.pressure.attrs["positive"] == "down"
    assert low.attrs["units"] == "mol mol-1" and low.attrs["vertical_extrapolation"] == "none"


def test_a_three_dimensional_field_without_any_pressure_description_is_refused():
    data = zonal_dataset("o3", np.zeros((2, 3, 1)), periods=2).assign_coords(
        lev=("lev", [0., 0.5, 1.], {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"}))
    with pytest.raises(MissingPressureCoordinate, match="native levels of this field cannot be described"):
        native_monthly_field(data, "o3", 2000, 2000)


def test_variables_that_disagree_about_the_native_pressure_are_refused():
    months = list(range(1, 13))
    first = xr.DataArray(profile([1e5, 1e4, 1e3]), dims=("month", "lev", "lat"),
                         coords={"month": months, "lat": [0.]})
    pressures = {}
    register_pressure(pressures, "lev", first, "o3", (2000, 2001), units="Pa")
    register_pressure(pressures, "lev", first.copy(deep=True), "br", (2000, 2001), units="Pa")
    assert pressures["lev"]["source_variable"] == "o3" and pressures["lev"]["kind"] == "hybrid"
    with pytest.raises(RuntimeError, match="disagree about the pressure"):
        register_pressure(pressures, "lev", first * 1.5, "br", (2000, 2001), units="Pa")
    register_pressure(pressures, "lev", first * 1.5, "bro", (1985, 2014), units="Pa")
    assert pressures["lev"]["source_variable"] == "o3"


def test_two_dimensional_and_three_dimensional_fields_share_one_native_product(tmp_path, monkeypatch):
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    values = np.broadcast_to(np.array([0., 1.])[None, :, None], (24, 2, 2)).copy()
    store(tmp_path, "GEOSCCM", "o3",
          with_pressure(zonal_dataset("o3", values, lats=(-45., 45.), periods=24), [1e5, 1e3]))
    store(tmp_path, "GEOSCCM", "trop",
          zonal_dataset("trop", np.full((24, 2), 25000.), lats=(-45., 45.), periods=24, units="Pa"))
    product = build_climatology("GEOSCCM", 2000, 2001, ["o3", "trop"])
    assert product == tmp_path / load_config("GEOSCCM")["paths"]["climatology"] / product_name("GEOSCCM", 2000, 2001)
    assert sorted(path.name for path in product.parent.iterdir()) == [product.name]
    with xr.open_dataset(product, decode_cf=False) as ds:
        cf.assert_product(ds, names=("o3", "trop"), grid=GRID)
        assert list(ds.o3_mean.dims) == ["time", "lev", "lat"]
        assert list(ds.trop_mean.dims) == ["time", "lat"]
        assert ds.sizes["lev"] == 2 and ds.vertical_level_count == 2
        assert not [name for name in ds.data_vars if name.startswith("air_pressure")]
        assert ds.o3_mean.attrs["pressure_coordinate"] == "lev"
        assert ds.o3_mean.attrs["on_application_pressure_grid"] == "false"
        assert ds.o3_mean.attrs["native_level_dimension"] == "lev"
        assert ds.o3_mean.attrs["native_pressure_min_pa"] == 1000.0
        assert ds.lev.attrs["standard_name"] == "air_pressure" and ds.lev.attrs["units"] == "Pa"
        assert ds.attrs["vertical_coordinate"].startswith("'lev' native pressure levels in Pa")
        assert "124 levels" in ds.attrs["application_pressure_grid"]
        assert ds.attrs["latitude_count"] == 2 and ds.sizes["lat"] == 2
        assert not [dim for dim in ds.dims if dim not in ("time", "lev", "lat", cf.BOUNDS_DIMENSION)]


def test_a_hybrid_product_publishes_the_climatological_pressure_of_its_native_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    pressure = profile([1e5, 1e4, 1e3], 24) * np.array([1.0, 0.5])[None, None, :]
    data = zonal_dataset("o3", np.broadcast_to(np.array([0., 1., 2.])[None, :, None], (24, 3, 2)), lats=(-45., 45.),
                         periods=24)
    data = data.assign_coords(lev=("lev", np.arange(3.), {"units": "1", "long_name": "model level index"}))
    store(tmp_path, "CMAM", "o3", with_hybrid_pressure(data, pressure))
    product = build_climatology("CMAM", 2000, 2001, ["o3"])
    with xr.open_dataset(product, decode_cf=False) as ds:
        cf.assert_product(ds, names=("o3",), grid=GRID)
        assert list(ds.air_pressure.dims) == ["time", "lev", "lat"]
        assert ds.air_pressure.attrs["standard_name"] == "air_pressure" and ds.air_pressure.attrs["units"] == "Pa"
        assert ds.air_pressure.attrs["positive"] == "down" and ds.air_pressure.attrs["axis"] == "Z"
        assert ds.air_pressure.attrs["cell_methods"] == "time: mean"
        assert ds.air_pressure.attrs["source_variable"] == "o3"
        assert np.allclose(ds.air_pressure.values, pressure[:12])
        assert ds.o3_mean.attrs["pressure_field"] == "air_pressure"
        assert "standard_name" not in ds.lev.attrs and ds.lev.attrs["units"] == "1"
        assert ds.attrs["vertical_coordinate"].startswith("'lev' native level index of the model")
        recovered = product_pressure(ds, ds.o3_mean)
        assert np.array_equal(recovered.values, ds.air_pressure.values) and recovered.attrs["units"] == "Pa"
        placed = interpolate_log_pressure(ds.o3_mean.isel(time=0), recovered.isel(time=0), GRID["levels"])
        between_tops = LEVELS[(LEVELS < 1e3) & (LEVELS > 500.0)][0]
        column = placed.isel(pressure=int(np.flatnonzero(LEVELS == between_tops)[0]))
        assert np.isnan(column.values[0]) and np.isfinite(column.values[1])


def test_a_product_built_on_the_application_pressure_grid_is_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    store(tmp_path, "GEOSCCM", "o3",
          with_pressure(zonal_dataset("o3", np.zeros((24, LEVELS.size, 1)), periods=24), LEVELS))
    with pytest.raises(cf.ProductProblem, match="must be rebuilt with jumacs climatology"):
        build_climatology("GEOSCCM", 2000, 2001, ["o3"])
    assert not list((tmp_path / load_config("GEOSCCM")["paths"]["climatology"]).iterdir())


def statistics(dims, coords):
    lengths = [len(np.atleast_1d(coords[dim][1] if isinstance(coords[dim], tuple) else coords[dim]))
               for dim in dims]
    return xr.Dataset({statistic: (dims, np.ones(lengths)) for statistic in cf.STATISTIC_ORDER}, coords=coords)


def native_part(name, level, levels, units, attrs=None):
    coords = {"month": list(range(1, 13)), "lat": [0.], level: (level, levels, dict(attrs or {"units": "Pa"}))}
    return variable_statistics(name, statistics(("month", level, "lat"), coords), {"units": units}, {})


def coordinate_pressure(levels, units="Pa"):
    return {"kind": "coordinate", "pressure": xr.DataArray(np.asarray(levels, float), dims=("plev",)),
            "source_variable": "o3", "coverage": (2000, 2001), "units": units}


def test_every_native_vertical_dimension_of_a_model_is_described_in_the_product():
    fixed = np.logspace(5, 2, 20) * 100.0
    hybrid = xr.DataArray(np.broadcast_to(np.logspace(5, 0, 6)[None, :, None], (12, 6, 1)).copy(),
                          dims=("month", "lev", "lat"), coords={"month": list(range(1, 13)), "lat": [0.]})
    parts = [native_part("o3", "plev", fixed, "mol mol-1", {"standard_name": "air_pressure", "units": "Pa"}),
             native_part("ta", "lev", np.arange(6.), "K", {"units": "1"})]
    pressures = {"plev": coordinate_pressure(fixed),
                 "lev": {"kind": "hybrid", "pressure": hybrid, "source_variable": "ta",
                         "coverage": (2000, 2001), "units": "Pa"}}
    product = assemble_product("CMAM", 2000, 2001, parts, (), ["o3", "ta"], pressures)
    cf.assert_product(product, names=("o3", "ta"), grid=GRID)
    expected = {f"{name}_{statistic}" for name in ("o3", "ta") for statistic in cf.STATISTIC_ORDER}
    assert set(product.data_vars) == expected | {"climatology_bounds", "air_pressure_lev"}
    assert list(product.o3_mean.dims) == ["time", "plev", "lat"]
    assert list(product.ta_mean.dims) == ["time", "lev", "lat"]
    assert product.o3_mean.attrs["pressure_coordinate"] == "plev"
    assert product.ta_mean.attrs["pressure_field"] == "air_pressure_lev"
    assert product.air_pressure_lev.attrs["cell_methods"] == "time: mean"
    assert product.plev.attrs["units"] == "Pa" and product.plev.attrs["source_coordinate_units"] == "Pa"
    assert np.allclose(product.plev.values, fixed)
    assert product.attrs["vertical_level_counts"] == "lev=6; plev=20"
    assert product.attrs["vertical_level_count"] == 20


@pytest.mark.parametrize("levels,message", [
    ([1e5, 1e5, 1e3], "must not repeat a pressure level"),
    ([1e5, 0.0, 1e3], "must be finite and positive"),
])
def test_a_native_pressure_coordinate_that_cannot_be_interpolated_is_refused(levels, message):
    parts = [native_part("o3", "plev", levels, "mol mol-1", {"standard_name": "air_pressure", "units": "Pa"})]
    product = assemble_product("CMAM", 2000, 2001, parts, (), ["o3"], {"plev": coordinate_pressure(levels)})
    with pytest.raises(cf.ProductProblem, match=message):
        cf.assert_product(product, names=("o3",), grid=GRID)


def test_a_coordinate_and_a_hybrid_grid_of_the_same_size_do_not_become_one_axis():
    parts = [native_part("o3", "plev", np.logspace(5, 2, 6) * 100.0, "mol mol-1",
                         {"standard_name": "air_pressure", "units": "Pa"}),
             native_part("ta", "lev", np.arange(6.), "K", {"units": "1"})]
    product = assemble_product("CMAM", 2000, 2001, parts, (), ["o3", "ta"], {"plev": coordinate_pressure(
        np.logspace(5, 2, 6) * 100.0)})
    with pytest.raises(cf.ProductProblem, match="neither names a pressure field nor has a pressure-valued"):
        cf.assert_product(product, names=("o3", "ta"), grid=GRID)


def test_transition_widths_are_counted_in_application_grid_levels():
    extension = config_module.extension_settings()
    assert extension["transition_levels"] == 12
    assert extension["transition_levels_by_variable"] == {}
    span_km = 7.0 * abs(GRID["delta_log10_pressure"]) * (extension["transition_levels"] - 1) * math.log(10.0)
    assert 10.0 <= span_km <= 12.0


def test_a_per_variable_transition_width_is_configuration_only(monkeypatch):
    monkeypatch.setattr(config_module, "reference_period",
                        lambda: {"extension": {"transition_levels": 12,
                                               "transition_levels_by_variable": {"br": 20}}})
    assert config_module.extension_settings()["transition_levels_by_variable"] == {"br": 20}
    monkeypatch.setattr(config_module, "reference_period", lambda: {"extension": {"transition_levels": 1}})
    with pytest.raises(ValueError, match="at least 2"):
        config_module.extension_settings()
