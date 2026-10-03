"""Blending one application-grid climatology with a whole-atmosphere donor above its own top."""
import numpy as np
import pytest
import xarray as xr

from jumacs import cf, config, coverage, extension, remap, vertical
from jumacs.climatology import assemble_product, variable_statistics

LEVELS = np.asarray(config.vertical_grid()["levels"], float)
TOP = LEVELS.size - 1
BANDS = remap.latitude_bands()[1]
NATIVE_LAT = np.linspace(-88.57216851400727, 88.57216851400727, 96)
STATISTICS = ("mean", "sigma", "minimum", "maximum")

BASE_TOP = 60
DONOR_TOP = 100
TRANSITION = config.extension_settings()["transition_levels"]
WINDOW = (BASE_TOP - TRANSITION + 1, BASE_TOP)
BASE_VALUE = {"mean": 1.0, "sigma": 101.0, "minimum": 201.0, "maximum": 301.0}
DONOR_VALUE = {"mean": 1001.0, "sigma": 2001.0, "minimum": 3001.0, "maximum": 4001.0}


def identical(left, right):
    """Whether two arrays hold the same numbers, with a missing value matching a missing value."""
    return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)


def column(base_top=BASE_TOP, donor_top=DONOR_TOP, gap=None, donor_gap=None):
    """A base profile that ends at ``base_top`` and a donor that reaches to ``donor_top``."""
    levels = np.arange(LEVELS.size)
    base = np.where(levels <= base_top, 1.0 + levels, np.nan)
    donor = np.where(levels <= donor_top, 100.0 + levels, np.nan)
    if gap is not None:
        base[gap] = np.nan
    if donor_gap is not None:
        donor[donor_gap] = np.nan
    return base, donor


def field(values, name="ta_mean", units="K", time=2, lat=BANDS.size, attrs=None):
    """A three-dimensional statistic field holding ``values`` in every month and band it is given."""
    profile = np.asarray(values, np.float64)
    if profile.ndim == 1:
        profile = profile[None, :, None]
    elif profile.ndim == 2:
        profile = profile[:, :, None]
    array = np.broadcast_to(profile, (time, LEVELS.size, lat)).astype(np.float64)
    coords = {"time": ("time", np.arange(time, dtype=float)), "pressure": ("pressure", LEVELS),
              "lat": ("lat", BANDS[:lat])}
    merged = {"source_variable": name.rsplit("_", 1)[0], "units": units}
    merged.update(attrs or {})
    return xr.DataArray(array, dims=("time", "pressure", "lat"), coords=coords, name=name, attrs=merged)


def flat_field(values, name="toz_mean", units="DU"):
    array = np.asarray(values, np.float64)
    return xr.DataArray(array, dims=("time", "lat"),
                        coords={"time": ("time", np.arange(array.shape[0], dtype=float)), "lat": ("lat", BANDS)},
                        name=name, attrs={"source_variable": name.rsplit("_", 1)[0], "units": units})


def native_product(value=250.0, units="K", name="ta", width_degrees=remap.BAND_WIDTH_DEGREES):
    """A published product on a native grid, remapped so that it is known to validate."""
    stats = xr.Dataset({statistic: (("month", "lev", "lat"), np.full((12, 47, NATIVE_LAT.size), value, float))
                        for statistic in cf.STATISTIC_ORDER},
                       coords={"month": list(range(1, 13)), "lev": ("lev", np.arange(47.0), {"units": "1"}),
                               "lat": ("lat", NATIVE_LAT)})
    part = variable_statistics(name, stats, {"units": units}, {})
    hybrid = xr.DataArray(
        np.broadcast_to(np.logspace(5.0, 0.0, 47)[None, :, None], (12, 47, NATIVE_LAT.size)).copy(),
        dims=("month", "lev", "lat"), coords={"month": list(range(1, 13)), "lat": ("lat", NATIVE_LAT)})
    pressures = {"lev": {"kind": "hybrid", "pressure": hybrid, "source_variable": name, "coverage": (2000, 2001),
                         "units": "Pa", "source_level": "lev"}}
    return remap.remap_product(assemble_product("SOCOL", 2000, 2001, [part], (), [name], pressures),
                               width_degrees=width_degrees)


def application_pair(base_top=BASE_TOP, donor_top=DONOR_TOP):
    """Two remapped products on one grid: the base ends lower, the donor is labelled WACCM-X."""
    base, donor = native_product(), native_product()
    donor.attrs["model"] = "WACCM-X"
    for statistic in STATISTICS:
        for dataset, top, value in ((base, base_top, BASE_VALUE[statistic]), (donor, donor_top, DONOR_VALUE[statistic])):
            values = np.full((12, LEVELS.size, BANDS.size), np.nan, np.float32)
            values[:, :top + 1, :] = value
            dataset[f"ta_{statistic}"].values = values
    return base, donor


def weight(index, first=WINDOW[0], last=WINDOW[1]):
    span = np.log(LEVELS[first]) - np.log(LEVELS[last])
    return float(np.clip((np.log(LEVELS[first]) - np.log(LEVELS[index])) / span, 0.0, 1.0))


def test_the_transition_width_is_read_from_the_configuration_once(monkeypatch):
    assert extension.transition_level_count("ta") == 12
    assert extension.transition_level_count("ta") == config.extension_settings()["transition_levels"]
    monkeypatch.setattr(config, "extension_settings",
                        lambda: {"transition_levels": 12, "transition_levels_by_variable": {"o3": 20}})
    assert extension.transition_level_count("o3") == 20
    assert extension.transition_level_count("O3") == 20
    assert extension.transition_level_count("ta") == 12
    assert extension.transition_level_count("o3", 8) == 8
    with pytest.raises(extension.ExtensionProblem, match="at least 2 overlapping levels"):
        extension.transition_level_count("o3", 1)


def test_the_donor_weight_runs_from_zero_at_the_bottom_to_one_at_the_top():
    weights = extension.blend_weights(LEVELS, *WINDOW)
    assert weights[WINDOW[0]] == 0.0 and weights[WINDOW[1]] == 1.0
    assert (np.diff(weights[WINDOW[0]:WINDOW[1] + 1]) > 0.0).all()
    assert (weights[:WINDOW[0] + 1] == 0.0).all() and (weights[WINDOW[1]:] == 1.0).all()
    middle = WINDOW[0] + TRANSITION // 2
    assert weights[middle] == pytest.approx(weight(middle))
    assert 0.0 < weights[middle] < 1.0


def test_a_transition_needs_two_levels_and_a_downward_pressure_run():
    with pytest.raises(extension.ExtensionProblem, match="two distinct levels"):
        extension.blend_weights(LEVELS, 40, 40)
    with pytest.raises(extension.ExtensionProblem, match="higher to a lower pressure"):
        extension.blend_weights(LEVELS[::-1], 40, 50)
    with pytest.raises(extension.ExtensionProblem, match="two distinct levels"):
        extension.blend_weights(LEVELS, 60, 49)


def test_the_blend_is_linear_in_log_pressure_between_pure_base_and_pure_donor():
    base, donor = column()
    values, outcome = extension.extend_column(base, donor, LEVELS, TRANSITION)
    assert outcome == {"extended": True, "reason": "extended", "overlap_levels": BASE_TOP + 1,
                       "transition_levels_used": TRANSITION, "window": WINDOW,
                       "base_top_pressure_pa": float(LEVELS[BASE_TOP]),
                       "donor_top_pressure_pa": float(LEVELS[DONOR_TOP])}
    assert identical(values[:WINDOW[0]], base[:WINDOW[0]])
    assert values[WINDOW[0]] == pytest.approx(base[WINDOW[0]])
    assert values[WINDOW[1]] == pytest.approx(donor[WINDOW[1]])
    for index in range(WINDOW[0] + 1, WINDOW[1]):
        expected = (1.0 - weight(index)) * base[index] + weight(index) * donor[index]
        assert values[index] == pytest.approx(expected)
        assert base[index] < values[index] < donor[index]
    assert np.allclose(values[WINDOW[1] + 1:DONOR_TOP + 1], donor[WINDOW[1] + 1:DONOR_TOP + 1])
    assert np.isnan(values[DONOR_TOP + 1:]).all()


def test_only_the_uppermost_configured_levels_blend_and_the_rest_of_the_overlap_stays_base():
    base, donor = column()
    values, outcome = extension.extend_column(base, donor, LEVELS, TRANSITION)
    assert outcome["overlap_levels"] == BASE_TOP + 1 and outcome["transition_levels_used"] == TRANSITION
    below = np.arange(WINDOW[0] - 8, WINDOW[0])
    assert np.isfinite(base[below]).all() and np.isfinite(donor[below]).all()
    assert identical(values[below], base[below])


def test_a_shorter_overlap_blends_over_every_level_it_can():
    for overlap in (5, 3, 2):
        top = overlap - 1
        base, donor = column(base_top=top)
        values, outcome = extension.extend_column(base, donor, LEVELS, TRANSITION)
        assert outcome["window"] == (0, top) and outcome["transition_levels_used"] == overlap
        assert values[0] == pytest.approx(base[0]) and values[top] == pytest.approx(donor[top])


def test_one_overlapping_level_is_not_a_blend():
    base, donor = column(base_top=4)
    base[:4] = np.nan
    values, outcome = extension.extend_column(base, donor, LEVELS, TRANSITION)
    assert outcome["extended"] is False and outcome["reason"] == "no_overlap"
    assert outcome["overlap_levels"] == 1 and outcome["window"] is None
    assert identical(values, base)
    alone, _ = extension.extend_column(np.full(LEVELS.size, np.nan), donor, LEVELS, TRANSITION)
    assert np.isnan(alone).all()


def test_a_donor_that_does_not_reach_higher_leaves_the_column_untouched():
    base, donor = column(donor_top=BASE_TOP - 20)
    values, outcome = extension.extend_column(base, donor, LEVELS, TRANSITION)
    assert outcome["extended"] is False and outcome["reason"] == "donor_not_higher"
    assert outcome["base_top_pressure_pa"] == pytest.approx(float(LEVELS[BASE_TOP]))
    assert outcome["base_top_pressure_pa"] < outcome["donor_top_pressure_pa"]
    assert identical(values, base)
    tied_base, tied_donor = column(donor_top=BASE_TOP)
    tied, outcome = extension.extend_column(tied_base, tied_donor, LEVELS, TRANSITION)
    assert outcome["reason"] == "donor_not_higher" and identical(tied, tied_base)


def test_inside_the_transition_the_side_that_is_there_is_used_and_neither_is_invented():
    base, donor = column(gap=BASE_TOP - 2)
    donor[WINDOW[0] + 1] = np.nan
    base[BASE_TOP - 1] = donor[BASE_TOP - 1] = np.nan
    values, _ = extension.extend_column(base, donor, LEVELS, TRANSITION)
    assert np.isnan(base[BASE_TOP - 2]) and np.isfinite(donor[BASE_TOP - 2])
    assert values[BASE_TOP - 2] == pytest.approx(donor[BASE_TOP - 2])
    assert np.isfinite(base[WINDOW[0] + 1]) and np.isnan(donor[WINDOW[0] + 1])
    assert values[WINDOW[0] + 1] == pytest.approx(base[WINDOW[0] + 1])
    assert np.isnan(base[BASE_TOP - 1]) and np.isnan(donor[BASE_TOP - 1])
    assert np.isnan(values[BASE_TOP - 1])


def test_a_hole_inside_the_base_profile_is_not_repaired_from_above():
    hole = WINDOW[0] - 12
    base, donor = column(gap=hole)
    values, outcome = extension.extend_column(base, donor, LEVELS, TRANSITION)
    assert outcome["extended"] and hole < outcome["window"][0]
    assert np.isfinite(donor[hole]) and np.isnan(base[hole])
    assert np.isnan(values[hole])
    assert identical(np.isnan(values[:WINDOW[0]]), np.isnan(base[:WINDOW[0]]))


def test_above_the_transition_the_donor_continues_the_column_and_its_gaps_stay_gaps():
    base, donor = column(donor_gap=WINDOW[1] + 6)
    values, _ = extension.extend_column(base, donor, LEVELS, TRANSITION)
    assert np.allclose(values[WINDOW[1] + 1:WINDOW[1] + 6], donor[WINDOW[1] + 1:WINDOW[1] + 6])
    assert np.isnan(donor[WINDOW[1] + 6]) and np.isnan(values[WINDOW[1] + 6])


def test_the_outcome_of_a_column_is_reported_for_a_smoke_test_to_read():
    base, donor = column()
    result, diagnostics = extension.extend_field(field(base, lat=3), field(donor, lat=3), LEVELS, TRANSITION)
    assert result.shape == (2, LEVELS.size, 3)
    assert diagnostics["field"] == "ta_mean" and diagnostics["variable"] == "ta"
    assert diagnostics["reason"] == "extended"
    assert diagnostics["extension_applied"] is True
    assert diagnostics["columns_total"] == 6 and diagnostics["columns_extended"] == 6
    assert diagnostics["columns_no_overlap"] == 0 and diagnostics["columns_donor_not_higher"] == 0
    assert diagnostics["transition_levels_requested"] == TRANSITION
    assert diagnostics["transition_levels_min_used"] == diagnostics["transition_levels_max_used"] == TRANSITION
    assert diagnostics["base_top_pressure_range"] == [float(LEVELS[BASE_TOP])] * 2
    assert diagnostics["donor_top_pressure_range"] == [float(LEVELS[DONOR_TOP])] * 2


def test_the_decision_is_made_in_every_month_and_latitude_column_separately():
    good_base, good_donor = column()
    flat_base, flat_donor = column(donor_top=BASE_TOP - 10)
    empty_base, empty_donor = column()
    empty_donor[:] = np.nan
    assert extension.extension_outcome(empty_base, empty_donor, LEVELS, TRANSITION)["reason"] == "no_overlap"
    base_values = np.stack([np.repeat(good_base[:, None], 3, axis=1),
                            np.column_stack([flat_base, empty_base, good_base])])
    donor_values = np.stack([np.repeat(good_donor[:, None], 3, axis=1),
                             np.column_stack([flat_donor, empty_donor, good_donor])])
    result, diagnostics = extension.extend_field(field(base_values, time=2, lat=3),
                                                 field(donor_values, time=2, lat=3), LEVELS, TRANSITION)
    assert diagnostics["columns_total"] == 6 and diagnostics["columns_extended"] == 4
    assert diagnostics["columns_no_overlap"] == 1 and diagnostics["columns_donor_not_higher"] == 1
    assert diagnostics["transition_levels_min_used"] == diagnostics["transition_levels_max_used"] == TRANSITION
    expected, _ = extension.extend_column(good_base, good_donor, LEVELS, TRANSITION)
    assert identical(result[0, :, 0], expected) and identical(result[1, :, 2], expected)
    assert identical(result[1, :, 0], flat_base) and identical(result[1, :, 1], empty_base)


def test_the_same_latitude_is_decided_differently_in_two_months():
    summer_base, summer_donor = column()
    winter_base, winter_donor = column(donor_top=BASE_TOP - 30)
    base = np.stack([summer_base, winter_base])[:, :, None]
    donor = np.stack([summer_donor, winter_donor])[:, :, None]
    result, diagnostics = extension.extend_field(field(base, time=2, lat=1), field(donor, time=2, lat=1),
                                                 LEVELS, TRANSITION)
    assert diagnostics["columns_total"] == 2 and diagnostics["columns_extended"] == 1
    assert np.isfinite(result[0, BASE_TOP + 1, 0]) and np.isnan(result[1, BASE_TOP + 1, 0])


def test_a_field_without_a_donor_counterpart_is_carried_and_a_donor_only_field_never_arrives():
    base, donor = application_pair()
    for statistic in STATISTICS:
        base[f"cl_{statistic}"] = base[f"ta_{statistic}"]
        base[f"toz_{statistic}"] = base[f"ta_{statistic}"].isel(pressure=0)
        donor[f"no2_{statistic}"] = donor[f"ta_{statistic}"]
    extended, rows = extension.extend_product(base, donor, names=("ta", "cl", "toz"), validate=False)
    assert "no2_mean" not in extended.data_vars and "no2_sigma" not in extended.data_vars
    assert identical(extended["cl_mean"].values, base["cl_mean"].values)
    assert extended["cl_mean"].attrs["extension_applied"] == "false"
    assert extended["cl_mean"].attrs["extension_reason"] == "WACCM-X holds no cl_mean field"
    assert extended["toz_mean"].dims == ("time", "lat")
    assert identical(extended["toz_mean"].values, base["toz_mean"].values)
    assert "no vertical dimension" in extended["toz_mean"].attrs["extension_reason"]
    assert extended["ta_mean"].attrs["extension_applied"] == "true"
    assert {row["field"] for row in rows} == {f"{stem}_{statistic}" for stem in ("ta", "cl", "toz")
                                              for statistic in STATISTICS}
    assert sum(1 for row in rows if row["extension_applied"]) == len(STATISTICS)


def test_the_count_of_years_is_carried_as_it_was_and_never_blended():
    base, donor = application_pair()
    del base["ta_mean"], donor["ta_mean"]
    base["ta_n_years"] = xr.DataArray(np.full((12, LEVELS.size, BANDS.size), 30, np.int16),
                                      dims=base["ta_sigma"].dims, coords=base["ta_sigma"].coords,
                                      attrs={"units": "1", "source_variable": "ta"})
    extended, _ = extension.extend_product(base, donor, statistics=("sigma", "minimum", "maximum"), validate=False)
    assert identical(extended["ta_n_years"].values, base["ta_n_years"].values)
    assert extended["ta_n_years"].attrs["extension_applied"] == "false"
    assert extended["ta_n_years"].attrs["extension_reason"] == (
        "ta_n_years is not one of the sigma, minimum, maximum statistics")


def test_every_statistic_is_blended_on_its_own_and_none_is_derived_from_another():
    base, donor = application_pair()
    extended, _ = extension.extend_product(base, donor)
    middle = WINDOW[0] + TRANSITION // 2
    for statistic in STATISTICS:
        below, base_value, donor_value = 4, BASE_VALUE[statistic], DONOR_VALUE[statistic]
        values = extended[f"ta_{statistic}"].values
        assert values[3, below, 7] == pytest.approx(base_value, rel=1e-6)
        assert values[3, WINDOW[0], 7] == pytest.approx(base_value, rel=1e-6)
        assert values[3, middle, 7] == pytest.approx((1 - weight(middle)) * base_value + weight(middle) * donor_value,
                                                     rel=1e-5)
        assert values[3, WINDOW[1], 7] == pytest.approx(donor_value, rel=1e-6)
        assert values[3, BASE_TOP + 4, 7] == pytest.approx(donor_value, rel=1e-6)
    spread = extended["ta_sigma"].values[:, BASE_TOP + 3, :] - extended["ta_mean"].values[:, BASE_TOP + 3, :]
    assert np.allclose(spread, DONOR_VALUE["sigma"] - DONOR_VALUE["mean"])
    lower = extended["ta_sigma"].values[:, 4, :] - extended["ta_mean"].values[:, 4, :]
    assert np.allclose(lower, BASE_VALUE["sigma"] - BASE_VALUE["mean"])


def test_the_levels_the_base_model_cannot_fill_are_filled_and_the_product_still_validates():
    base, donor = application_pair()
    assert np.isnan(base["ta_mean"].values[:, BASE_TOP + 1:, :]).all()
    extended, rows = extension.extend_product(base, donor)
    cf.assert_product(extended, names=("ta",), grid=config.vertical_grid(), kind="application",
                      latitude_bands=BANDS.size)
    assert dict(extended.sizes) == {"time": 12, "pressure": LEVELS.size, "lat": BANDS.size, "nv": 2}
    assert extended["ta_mean"].dtype == np.dtype("float32")
    assert np.isfinite(extended["ta_mean"].values[:, BASE_TOP + 1:DONOR_TOP + 1, :]).all()
    assert np.isnan(extended["ta_mean"].values[:, DONOR_TOP + 1:, :]).all()
    assert identical(extended.pressure.values, LEVELS) and identical(extended.lat.values, BANDS)
    assert identical(extended["climatology_bounds"].values, base["climatology_bounds"].values)
    assert len(rows) == len(STATISTICS) and all(row["extension_applied"] for row in rows)
    assert all(row["columns_extended"] == 12 * BANDS.size for row in rows)


def test_the_blend_is_written_into_the_attributes_of_the_dataset_and_of_each_field():
    base, donor = application_pair()
    base["ta_mean"].attrs["vertical_treatment"] = "climatological statistics of the model on its native levels"
    extended, _ = extension.extend_product(base, donor)
    assert extended.attrs["product"] == ("single-model climatology on the JuMACS application grid with WACCM-X "
                                         "upper-atmosphere extension")
    assert extended.attrs["extension_donor"] == "WACCM-X" and extended.attrs["extension_base"] == "SOCOL"
    assert extended.attrs["extension_method"] == extension.EXTENSION_METHOD
    assert extended.attrs["extension_weighting"] == extension.EXTENSION_WEIGHTING
    assert extended.attrs["extension_eligibility"] == extension.EXTENSION_ELIGIBILITY
    assert extended.attrs["default_transition_levels"] == 12
    assert extended.attrs["extension_transition_level_unit"] == "application pressure levels, not kilometres"
    assert extended.attrs["extended_fields"].split() == sorted(f"ta_{statistic}" for statistic in STATISTICS)
    assert extended.attrs["unextended_fields"] == ""
    assert extended.attrs["base_product"] == base.attrs["product"]
    assert extended.attrs["donor_product"] == donor.attrs["product"]
    attrs = extended["ta_mean"].attrs
    assert attrs["extension_donor"] == "WACCM-X" and attrs["extension_applied"] == "true"
    assert attrs["extension_transition_levels"] == 12
    assert attrs["extension_columns_extended"] == 12 * BANDS.size
    assert attrs["native_vertical_treatment"] == "climatological statistics of the model on its native levels"
    assert "WACCM-X above" in attrs["vertical_treatment"]
    assert attrs["cell_methods"] == cf.cell_methods("mean") and attrs["units"] == "K"


def test_an_unqualified_column_is_reported_as_rejected_rather_than_silently_filled():
    base, donor = application_pair()
    donor["ta_mean"].values[0, :, :] = np.nan
    donor["ta_mean"].values[1, :, :] = base["ta_mean"].values[1, :, :]
    extended, rows = extension.extend_product(base, donor)
    row = next(row for row in rows if row["field"] == "ta_mean")
    assert row["columns_extended"] == (12 - 2) * BANDS.size
    assert row["columns_no_overlap"] == BANDS.size and row["columns_donor_not_higher"] == BANDS.size
    assert row["reason"] == (f"donor_not_higher in {BANDS.size} of {12 * BANDS.size} columns; no_overlap in "
                             f"{BANDS.size} of {12 * BANDS.size} columns")
    assert extended.attrs["extension_rejected"].startswith("ta_mean: ")
    assert "ta_sigma" not in extended.attrs["extension_rejected"]
    assert extended["ta_mean"].attrs["extension_applied"] == "true"
    assert np.isnan(extended["ta_mean"].values[0, BASE_TOP + 1, 0])
    assert identical(extended["ta_mean"].values[1, :, 0], base["ta_mean"].values[1, :, 0])


def test_different_units_are_refused_rather_than_converted():
    base, donor = column()
    with pytest.raises(extension.ExtensionProblem, match="are not converted"):
        extension.extend_field(field(base), field(donor, units="Pa"), LEVELS, TRANSITION)
    with pytest.raises(extension.ExtensionProblem, match="must carry its units"):
        extension.extend_field(field(base), field(donor).assign_attrs(units=""), LEVELS, TRANSITION)
    values, _ = extension.extend_field(field(base, units="m s-1"), field(donor, units="m/s"), LEVELS, TRANSITION)
    assert np.isfinite(values[:, :DONOR_TOP + 1]).all()


def test_two_datasets_that_do_not_sit_on_the_same_grid_are_refused_not_aligned():
    base, donor = application_pair()
    with pytest.raises(extension.ExtensionProblem, match="pressure differs"):
        extension.assert_same_application_grid(base, donor.assign_coords(pressure=("pressure", LEVELS * 1.001)))
    with pytest.raises(extension.ExtensionProblem, match="lat differs"):
        extension.assert_same_application_grid(base.assign_coords(lat=("lat", BANDS + 0.5)), donor)
    narrow = native_product(width_degrees=10.0)
    assert narrow.sizes["lat"] == BANDS.size // 2
    with pytest.raises(extension.ExtensionProblem, match="lat differs"):
        extension.assert_same_application_grid(base, narrow)
    _, latitudes, width = extension.assert_same_application_grid(narrow, narrow)
    assert width == pytest.approx(10.0) and latitudes.size == BANDS.size // 2
    odd = base.assign_coords(lat=("lat", BANDS + 0.01))
    with pytest.raises(extension.ExtensionProblem, match="fixed latitude band centers"):
        extension.assert_same_application_grid(odd, donor.assign_coords(lat=("lat", BANDS + 0.01)))
    off_grid_levels = ("pressure", np.logspace(5.0, -3.0, LEVELS.size))
    with pytest.raises(extension.ExtensionProblem, match="configured application levels"):
        extension.assert_same_application_grid(base.assign_coords(pressure=off_grid_levels),
                                              donor.assign_coords(pressure=off_grid_levels))
    assert extension.assert_same_application_grid(base, donor)[2] == pytest.approx(5.0)


def test_the_blend_of_two_matched_products_refuses_a_grid_mismatch_before_blending():
    base, donor = application_pair()
    donor = donor.assign_coords(time=("time", base["time"].values + 1.0))
    with pytest.raises(extension.ExtensionProblem, match="time differs"):
        extension.extend_product(base, donor)


def test_a_field_that_is_not_on_the_application_grid_is_refused_by_the_field_check():
    base, donor = column()
    off_grid = field(base).assign_coords(pressure=("pressure", np.logspace(5, -3, LEVELS.size)))
    with pytest.raises(extension.ExtensionProblem, match="application pressure grid"):
        extension.extend_field(off_grid, field(donor), LEVELS, TRANSITION)
    with pytest.raises(extension.ExtensionProblem, match="ordered"):
        extension.extend_field(field(base).transpose("time", "lat", "pressure"), field(donor), LEVELS, TRANSITION)
    with pytest.raises(extension.ExtensionProblem, match="must be three-dimensional"):
        extension.extend_field(flat_field(np.zeros((12, BANDS.size))), flat_field(np.zeros((12, BANDS.size))))
    with pytest.raises(extension.ExtensionProblem, match="differs in size"):
        extension.extend_field(field(base, lat=18), field(donor, lat=BANDS.size), LEVELS, TRANSITION)
    with pytest.raises(extension.ExtensionProblem, match="same shape"):
        extension.extension_outcome(base, donor[:-1], LEVELS, TRANSITION)
    with pytest.raises(extension.ExtensionProblem, match="at least 2 overlapping levels"):
        extension.extension_outcome(base, donor, LEVELS, 1)


def test_the_coverage_matrix_and_the_blend_ask_for_the_same_overlap():
    assert extension.MINIMUM_OVERLAP_LEVELS == vertical.MINIMUM_OVERLAP_LEVELS
    assert coverage.MINIMUM_OVERLAP_LEVELS == vertical.MINIMUM_OVERLAP_LEVELS


def test_a_column_the_coverage_matrix_calls_an_extension_candidate_is_really_extended():
    base, donor = column()
    base_mask, donor_mask = np.isfinite(base), np.isfinite(donor)
    base_top, _ = coverage._range_of(LEVELS, base_mask)
    donor_top, _ = coverage._range_of(LEVELS, donor_mask)
    assert bool((donor_mask & (LEVELS < base_top)).any())
    assert int((base_mask & donor_mask).sum()) >= extension.MINIMUM_OVERLAP_LEVELS
    outcome = extension.extension_outcome(base, donor, LEVELS, TRANSITION)
    assert outcome["extended"] and outcome["reason"] == "extended"
    assert outcome["base_top_pressure_pa"] == pytest.approx(base_top)
    assert outcome["donor_top_pressure_pa"] == pytest.approx(donor_top)
    assert outcome["overlap_levels"] == int((base_mask & donor_mask).sum())
    lower_donor = column(donor_top=BASE_TOP - 10)
    tied_donor = column(donor_top=BASE_TOP)
    no_donor = (column()[0], np.full(LEVELS.size, np.nan))
    for base_profile, donor_profile in (lower_donor, tied_donor, no_donor):
        mask, reach = np.isfinite(base_profile), np.isfinite(donor_profile)
        top, _ = coverage._range_of(LEVELS, mask)
        upward = bool(top is not None and (reach & (LEVELS < top)).any())
        both = int((mask & reach).sum())
        outcome = extension.extension_outcome(base_profile, donor_profile, LEVELS, TRANSITION)
        assert outcome["extended"] is False
        assert not (upward and both >= extension.MINIMUM_OVERLAP_LEVELS)
        assert outcome["reason"] == ("no_overlap" if both < extension.MINIMUM_OVERLAP_LEVELS else "donor_not_higher")


def test_the_blend_needs_the_grid_it_is_given_and_the_levels_it_thinks_it_has():
    base, donor = column()
    assert extension.MINIMUM_OVERLAP_LEVELS == 2
    assert extension.EXTENSION_STATISTICS == remap.REMAP_STATISTICS == cf.APPLICATION_STATISTIC_ORDER
    values, outcome = extension.extend_column(base, donor, LEVELS, TRANSITION)
    assert identical(values[:WINDOW[0]], base[:WINDOW[0]]) and outcome["transition_levels_used"] == TRANSITION
    with pytest.raises(extension.ExtensionProblem, match="at least 2 overlapping levels"):
        extension.extend_column(base, donor, LEVELS, 0)
    assert extension.product_description("WACCM-X").endswith("WACCM-X upper-atmosphere extension")
    assert extension.donor_label(xr.Dataset(attrs={"model": "WACCM-X"})) == "WACCM-X"
    assert extension.donor_label(xr.Dataset()) == extension.EXTENDED_DONOR_FALLBACK
