import numpy as np
import xarray as xr

from jumacs.compact import blend_profiles, interpolate_profile, _on_grid, _unit_key


def test_altitude_interpolation_has_no_extrapolation():
    result = interpolate_profile(np.array([1., 3.]), np.array([10., 30.]), np.array([0., 1., 2., 3., 4.]))
    assert np.isnan(result[[0, 4]]).all()
    np.testing.assert_allclose(result[1:4], [10., 20., 30.])


def test_cosine_transition_and_missing_upper_species():
    height = np.array([50., 55., 60., 65., 70.])
    lower = np.full((1, 5, 1), 2.)
    upper = np.full((1, 5, 1), 4.)
    lower[:, -1] = np.nan
    np.testing.assert_allclose(blend_profiles(lower, upper, height, 55., 65.)[0, :, 0], [2., 2., 3., 4., 4.])
    upper[:, 3:] = np.nan
    assert np.isnan(blend_profiles(lower, upper, height, 55., 65.)[0, 3:, 0]).all()


def test_mole_fraction_units_agree():
    assert _unit_key("mol mol-1") == _unit_key("mol/mol")


def test_pressure_level_chemistry_uses_model_height_relation():
    months = np.arange(1, 13)
    height = xr.Dataset({
        "mean": (("month", "lev", "lat"), np.broadcast_to(np.array([0., 10000., 20000.])[None, :, None], (12, 3, 2))),
        "air_pressure": (("lev", "month", "lat"), np.broadcast_to(np.array([100000., 10000., 1000.])[:, None, None], (3, 12, 2)))
    }, coords={"month": months, "lev": [1, 2, 3], "lat": [-5., 5.]})
    chemistry = xr.Dataset({"mean": (("month", "plev", "lat"), np.broadcast_to(np.array([2., 4.])[None, :, None], (12, 2, 2)))},
                           coords={"month": months, "plev": xr.DataArray([100000., 1000.], dims="plev", attrs={"units":"Pa"}), "lat":[-5., 5.]})
    result = _on_grid(chemistry, height, np.array([0.]), np.array([0., 10., 20.]))
    np.testing.assert_allclose(result[:, :, 0], np.tile([2., 3., 4.], (12, 1)))
