"""Extension coverage is built from published products and never changes them.

Products stay on the native grid of their model, so coverage measures what a field
would provide once the combination moves it onto the shared application pressure
grid. That move is reproduced here in memory, exactly as the combination performs
it, and is never written back into a product.
"""
import csv
import json
import numpy as np
import pytest
import xarray as xr
from jumacs import cf, config, coverage

START, END = 1985, 2014
LATS = (-45.0, 0.0, 45.0)
SAMPLE, LEVEL = 0.9, 0.5
LEVEL_COUNT = len(config.vertical_grid()["levels"])
NATIVE_COUNT = 100
NATIVE_LEVEL = "plev"
FULL = (0, NATIVE_COUNT)
MODEL_WINDOW = (0, 70)


def levels():
    return np.asarray(config.vertical_grid()["levels"], "float64")


def native_levels():
    """The native grid of one model: its own levels, not the application grid."""
    return np.logspace(5, -3.5, NATIVE_COUNT)


def span(window, native=None):
    """The pressure range a block of native levels spans: (highest, lowest)."""
    native = native_levels() if native is None else native
    return float(native[window[0]]), float(native[window[1] - 1])


def inside(grid, top_pa, bottom_pa):
    """Application grid levels a native profile reaching top..bottom can supply."""
    grid = np.asarray(grid, float)
    return grid[(grid <= top_pa) & (grid >= bottom_pa)]


def donor_reaching(target, count=400):
    """A donor grid and window leaving exactly `target` usable levels in common with MODEL_WINDOW.

    A finer native grid than the model\'s is used so that the overlap can be adjusted one level at a
    time, whichever resolution the shared application grid happens to have.
    """
    fine = np.logspace(5.0, -3.5, count)
    model_top = span(MODEL_WINDOW)[1]
    for start in range(count):
        if inside(levels(), float(fine[start]), model_top).size == target:
            return fine, (start, count)
    raise AssertionError(f"no donor window on {count} levels leaves exactly {target} levels in common")


def write_product(workspace, model, fields, native=None):
    """A minimal but contract-valid individual product: native levels, pressure-valued coordinate."""
    pressure = np.asarray(native if native is not None else native_levels(), "float64")
    reference = config.reference_period()["reference_period"]["nominal_reference_year"]
    time = cf.climatology_time(START, END, reference)
    variables = {"climatology_bounds": (("time", "nv"), time["bounds"], {"units": time["units"]})}
    for stem, spec in fields.items():
        flat = spec.get("two_dimensional", False)
        dims = ("time", "lat") if flat else ("time", NATIVE_LEVEL, "lat")
        shape = (12, len(LATS)) if flat else (12, pressure.size, len(LATS))
        mean = np.zeros(shape, "float32") if flat else np.full(shape, np.nan, "float32")
        if flat:
            mean[...] = spec.get("value", 1.0)
        else:
            for block in [spec.get("levels", (0, pressure.size))] + spec.get("blocks", []):
                months = block[2] if len(block) > 2 else spec.get("sample_count")
                rows = slice(None) if months is None else slice(0, months)
                mean[rows, block[0]:block[1], :] = spec.get("value", 1.0)
        for statistic in cf.STATISTIC_ORDER:
            values = (np.ones(shape, "int16") if statistic == "n_years" else
                      (mean if statistic == "mean" else np.abs(mean).astype("float32")))
            units = "1" if statistic == "n_years" else spec["units"]
            variables[f"{stem}_{statistic}"] = (dims, values, {"units": units,
                                                               "cell_methods": cf.cell_methods(statistic)})
    coordinates = {
        "time": (("time",), time["values"], dict(time["attrs"])),
        NATIVE_LEVEL: ((NATIVE_LEVEL,), pressure, {"standard_name": "air_pressure", "units": "Pa",
                                                   "positive": "down", "axis": "Z"}),
        "lat": (("lat",), np.asarray(LATS, "float64"), {"standard_name": "latitude",
                                                        "units": "degrees_north"})}
    attrs = {"Conventions": cf.CONVENTIONS, "climatology_period": f"{START}-{END}",
             "time_coverage_start": f"{START}-01-01", "time_coverage_end": f"{END}-12-31",
             "reference_period_years": END - START + 1, "title": "synthetic extension coverage product"}
    directory = workspace / config.load_config(model)["paths"]["climatology"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"jumacs_{config.model_metadata(model)['slug']}_climatology_{START}-{END}.nc"
    xr.Dataset(variables, coordinates, attrs).to_netcdf(
        path, encoding={"climatology_bounds": {"_FillValue": None}})
    return path


def write_hybrid_product(workspace, model, native, reaches_pa):
    """A hybrid product whose native pressure depends on latitude, published as a climatology field."""
    native = np.asarray(native, "float64")
    reference = config.reference_period()["reference_period"]["nominal_reference_year"]
    time = cf.climatology_time(START, END, reference)
    shape = (12, native.size, len(LATS))
    pressure = np.broadcast_to(native[None, :, None], shape).copy()
    for index, limit in reaches_pa.items():
        pressure[:, native < limit, index] = np.nan
    variables = {"climatology_bounds": (("time", "nv"), time["bounds"], {"units": time["units"]}),
                 "air_pressure": (("time", "lev", "lat"), pressure,
                                  {"standard_name": "air_pressure", "units": "Pa", "positive": "down", "axis": "Z",
                                   "cell_methods": "time: mean"})}
    values = np.where(np.isfinite(pressure), 1e-6, np.nan).astype("float32")
    for statistic in cf.STATISTIC_ORDER:
        data = np.ones(shape, "int16") if statistic == "n_years" else (values if statistic == "mean" else
                                                                       np.abs(values).astype("float32"))
        variables[f"o3_{statistic}"] = (("time", "lev", "lat"), data,
                                        {"units": "1" if statistic == "n_years" else "1e-6",
                                         "cell_methods": cf.cell_methods(statistic),
                                         "pressure_field": "air_pressure"})
    coordinates = {"time": (("time",), time["values"], dict(time["attrs"])),
                   "lev": (("lev",), np.arange(native.size, dtype="float64"), {"axis": "Z", "positive": "down"}),
                   "lat": (("lat",), np.asarray(LATS, "float64"), {"standard_name": "latitude",
                                                                   "units": "degrees_north"})}
    attrs = {"Conventions": cf.CONVENTIONS, "climatology_period": f"{START}-{END}",
             "time_coverage_start": f"{START}-01-01", "time_coverage_end": f"{END}-12-31",
             "reference_period_years": END - START + 1, "title": "synthetic hybrid coverage product"}
    directory = workspace / config.load_config(model)["paths"]["climatology"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"jumacs_{config.model_metadata(model)['slug']}_climatology_{START}-{END}.nc"
    xr.Dataset(variables, coordinates, attrs).to_netcdf(
        path, encoding={"climatology_bounds": {"_FillValue": None}})
    return path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage, "ROOT", tmp_path)
    return tmp_path


def row_for(rows, model, variable):
    return next(row for row in rows if row["model"] == model and row["variable"] == variable)


def read_csv(path):
    with path.open(newline="") as stream:
        return {(row["model"], row["variable"]): row for row in csv.DictReader(stream)}


def matrix(models, donor="WACCM-X", sample_fraction=SAMPLE, level_fraction=LEVEL):
    return coverage.extension_matrix([model for model in models if model != donor], donor, START, END,
                                     sample_fraction, level_fraction)


def test_an_overlapping_model_that_stops_lower_is_an_extension_candidate(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 70)}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    reached = inside(grid, *span((0, 70)))
    assert ozone["extension_status"] == "extension_candidate"
    assert ozone["native_level_count"] == NATIVE_COUNT
    assert ozone["min_usable_pressure_pa"] == pytest.approx(reached.min())
    assert ozone["max_usable_pressure_pa"] == pytest.approx(grid[0])
    assert ozone["waccmx_min_usable_pressure_pa"] == pytest.approx(grid[-1])
    assert ozone["overlap_level_count"] == reached.size
    assert ozone["overlap_min_pressure_pa"] == pytest.approx(reached.min())
    assert ozone["overlap_max_pressure_pa"] == pytest.approx(grid[0])
    assert ozone["waccmx_extends_upward"] is True
    assert ozone["application_focus"] is True and ozone["group"] == "ozone"


def test_nothing_is_invented_above_or_below_the_native_reach_of_the_model(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 70)}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    fractions = row_for(matrix(["CMAM", "WACCM-X"])[1], "CMAM", "O3")["level_finite_fractions"]
    top, bottom = span((0, 70))
    assert len(fractions) == LEVEL_COUNT
    assert all(fractions[index] == 0.0 for index in np.flatnonzero(grid < bottom))
    assert all(fractions[index] == 1.0 for index in np.flatnonzero((grid <= top) & (grid >= bottom)))
    assert ozone_native(workspace) == (bottom, top)


def ozone_native(workspace):
    """The pressure range where the field itself has samples, not merely the levels it was written on."""
    entry = coverage.product_inventory("CMAM", START, END, SAMPLE)["fields"]["o3"]
    return entry["native_pressure_min_pa"], entry["native_pressure_max_pa"]


def test_the_lower_extent_is_the_valid_extent_so_terrain_thinning_does_not_hide_1000_hpa(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (1, 70), "blocks": [(0, 1, 6)]}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["max_valid_pressure_pa"] == pytest.approx(100000.0) == pytest.approx(grid[0])
    assert ozone["max_any_pressure_pa"] == ozone["max_valid_pressure_pa"]
    assert ozone["max_usable_pressure_pa"] == pytest.approx(inside(grid, *span((1, 70))).max())
    assert ozone["max_usable_pressure_pa"] < ozone["max_valid_pressure_pa"]
    fractions = ozone["level_finite_fractions"]
    thinned = [index for index, value in enumerate(fractions) if 0.0 < value < SAMPLE]
    assert thinned and grid[thinned[0]] == pytest.approx(grid[0])
    assert fractions[thinned[0]] == pytest.approx(0.5)
    assert ozone["levels_any_finite"] == inside(grid, *span((0, 70))).size
    assert ozone["levels_usable"] == inside(grid, *span((1, 70))).size
    assert ozone["contiguous_usable"] is True


def test_a_sparse_upper_level_does_not_become_the_usable_top(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 70), "blocks": [(88, 91, 3)]}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["min_any_pressure_pa"] == pytest.approx(inside(grid, *span((88, 91))).min())
    assert ozone["min_usable_pressure_pa"] == pytest.approx(inside(grid, *span((0, 70))).min())
    assert ozone["min_any_pressure_pa"] < ozone["min_usable_pressure_pa"]
    assert ozone["overlap_min_pressure_pa"] == ozone["min_usable_pressure_pa"]
    assert ozone["levels_any_finite"] == inside(grid, *span((0, 70))).size + inside(grid, *span((88, 91))).size
    assert ozone["waccmx_extends_upward"] is True and ozone["extension_status"] == "extension_candidate"


def test_the_valid_lower_bound_is_descriptive_and_leaves_the_decision_alone(workspace):
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (1, 70), "blocks": [(0, 1, 6)]}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    thinned = row_for(matrix(["CMAM", "WACCM-X"])[1], "CMAM", "O3")
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (1, 70)}})
    plain = row_for(matrix(["CMAM", "WACCM-X"])[1], "CMAM", "O3")
    decided = ("overlap_exists", "overlap_level_count", "overlap_min_pressure_pa", "overlap_max_pressure_pa",
               "min_usable_pressure_pa", "waccmx_min_usable_pressure_pa", "waccmx_extends_upward",
               "extension_status")
    assert {key: thinned[key] for key in decided} == {key: plain[key] for key in decided}
    assert thinned["max_valid_pressure_pa"] > plain["max_valid_pressure_pa"]
    assert thinned["max_usable_pressure_pa"] == plain["max_usable_pressure_pa"]


def test_ranges_that_do_not_meet_leave_no_overlap(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 20)}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": (60, NATIVE_COUNT)}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["extension_status"] == "no_overlap"
    assert ozone["overlap_exists"] is False and ozone["overlap_level_count"] == 0
    assert ozone["overlap_min_pressure_pa"] is None and ozone["overlap_max_pressure_pa"] is None
    assert ozone["waccmx_extends_upward"] is False
    assert ozone["min_usable_pressure_pa"] == pytest.approx(inside(grid, *span((0, 20))).min())
    assert ozone["waccmx_min_usable_pressure_pa"] == pytest.approx(grid[-1])


def test_overlap_without_upward_reach_is_not_a_candidate(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": FULL}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": (0, 50)}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["extension_status"] == "overlap_no_extension"
    assert ozone["overlap_level_count"] == inside(grid, *span((0, 50))).size
    assert ozone["overlap_min_pressure_pa"] == pytest.approx(inside(grid, *span((0, 50))).min())
    assert ozone["waccmx_extends_upward"] is False


def test_an_overlap_thinner_than_a_transition_needs_is_not_a_candidate(workspace):
    grid = levels()
    donor_grid, window = donor_reaching(1)
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": MODEL_WINDOW}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": window}}, native=donor_grid)
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["overlap_level_count"] == 1
    assert ozone["waccmx_extends_upward"] is True
    assert ozone["extension_status"] == "overlap_no_extension"
    shared = inside(grid, float(donor_grid[window[0]]), span(MODEL_WINDOW)[1])
    assert ozone["overlap_min_pressure_pa"] == pytest.approx(inside(grid, *span(MODEL_WINDOW)).min())
    assert ozone["overlap_max_pressure_pa"] == pytest.approx(shared.max())


def test_the_shared_minimum_overlap_of_levels_is_what_makes_a_candidate(workspace):
    grid = levels()
    donor_grid, window = donor_reaching(coverage.MINIMUM_OVERLAP_LEVELS)
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": MODEL_WINDOW}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": window}}, native=donor_grid)
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["overlap_level_count"] == coverage.MINIMUM_OVERLAP_LEVELS == 2
    assert ozone["waccmx_extends_upward"] is True
    assert ozone["extension_status"] == "extension_candidate"
    assert ozone["overlap_min_pressure_pa"] == pytest.approx(inside(grid, *span(MODEL_WINDOW)).min())
    assert ozone["overlap_max_pressure_pa"] == pytest.approx(
        inside(grid, float(donor_grid[window[0]]), span(MODEL_WINDOW)[1]).max())


def test_overlap_is_taken_from_the_masks_so_internal_gaps_stay_visible(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 20), "blocks": [(60, 81)]}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    blocks = inside(grid, *span((0, 20))).size + inside(grid, *span((60, 81))).size
    gap = (grid <= float(native_levels()[19])) & (grid >= float(native_levels()[60]))
    assert ozone["levels_usable"] == blocks
    assert ozone["contiguous_usable"] is False
    assert ozone["min_usable_pressure_pa"] == pytest.approx(inside(grid, *span((60, 81))).min())
    assert ozone["max_usable_pressure_pa"] == pytest.approx(grid[0])
    assert ozone["overlap_level_count"] == blocks
    assert ozone["extension_status"] == "extension_candidate"
    fractions = ozone["level_finite_fractions"]
    assert all(fractions[index] == 0.0 for index in np.flatnonzero(gap))
    assert fractions.count(0.0) == LEVEL_COUNT - blocks


def test_a_level_needs_most_of_its_samples_before_it_counts(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 70), "sample_count": 6}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    reached = inside(grid, *span((0, 70)))
    assert ozone["levels_any_finite"] == reached.size and ozone["levels_usable"] == 0
    assert ozone["min_usable_pressure_pa"] is None
    assert ozone["min_any_pressure_pa"] == pytest.approx(reached.min())
    assert ozone["contiguous_usable"] is None and ozone["broadly_usable"] is False
    assert ozone["extension_status"] == "no_overlap"
    assert all(ozone["level_finite_fractions"][index] == pytest.approx(0.5) for index in
               np.flatnonzero((grid <= reached.max()) & (grid >= reached.min())))
    _, relaxed = matrix(["CMAM", "WACCM-X"], sample_fraction=0.5)
    assert row_for(relaxed, "CMAM", "O3")["extension_status"] == "extension_candidate"


def test_coverage_follows_the_latitude_rows_that_the_model_reaches(workspace):
    grid = levels()
    native = np.logspace(5, -3.5, 60)
    write_hybrid_product(workspace, "CMAM", native, {1: 100.0})
    entry = coverage.product_inventory("CMAM", START, END, SAMPLE)["fields"]["o3"]
    edge = min(float(value) for value in native if value >= 100.0)
    assert entry["native_level_count"] == native.size
    assert entry["native_pressure_min_pa"] == pytest.approx(float(native[-1]))
    assert entry["native_pressure_max_pa"] == pytest.approx(float(native[0]))
    assert entry["levels_usable"] == inside(grid, float(native[0]), edge).size
    assert entry["min_usable_pressure_pa"] == pytest.approx(inside(grid, float(native[0]), edge).min())
    assert entry["min_any_pressure_pa"] == pytest.approx(grid[-1])
    partial = [index for index, value in enumerate(entry["level_finite_fractions"]) if 0.0 < value < SAMPLE]
    assert partial and all(entry["level_finite_fractions"][index] == pytest.approx(2 / 3) for index in partial)
    assert entry["grid_finite_fraction"] > entry["finite_fraction"]


def test_a_missing_waccmx_variable_is_kept_apart_from_an_unmapped_species(workspace):
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 80)},
                                      "ch4": {"units": "1e-6", "levels": (0, 80)},
                                      "clo": {"units": "1e-12", "levels": (20, 90)}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    assert row_for(rows, "CMAM", "O3")["extension_status"] == "extension_candidate"
    methane = row_for(rows, "CMAM", "CH4")
    assert methane["extension_status"] == "no_waccmx_variable" and methane["waccmx_available"] is False
    assert methane["present"] is True and methane["overlap_level_count"] == 0
    chlorine = row_for(rows, "CMAM", "ClO")
    assert chlorine["extension_status"] == "mapping_unresolved"
    assert chlorine["present"] is True and chlorine["waccmx_available"] is False
    assert chlorine["min_usable_pressure_pa"] is not None


def test_absent_and_two_dimensional_variables_are_not_applicable(workspace):
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 80)},
                                      "ps": {"units": "Pa", "two_dimensional": True}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL},
                                         "PS": {"units": "Pa", "two_dimensional": True}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    surface = row_for(rows, "CMAM", "surface_pressure")
    assert surface["extension_status"] == "not_applicable" and surface["is_3d"] is False
    assert surface["present"] is True and surface["levels_usable"] is None and surface["units"] == "Pa"
    assert surface["native_level_count"] == 0 and surface["native_pressure_max_pa"] is None
    assert surface["grid_finite_fraction"] is None
    temperature = row_for(rows, "CMAM", "temperature")
    assert temperature["extension_status"] == "not_applicable" and temperature["present"] is False
    assert temperature["product_variable"] == "" and temperature["level_finite_fractions"] == []


def test_family_sums_stay_apart_from_their_constituents(workspace):
    write_product(workspace, "CMAM", {"clo": {"units": "1e-12", "levels": (20, 90)},
                                      "cly": {"units": "1e-12", "levels": (20, 85)}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    family, species = row_for(rows, "CMAM", "Cly"), row_for(rows, "CMAM", "ClO")
    assert family["group"] == species["group"] == "inorganic_chlorine"
    assert family["product_variable"] == "cly_mean" and species["product_variable"] == "clo_mean"
    assert family is not species


def test_an_unmapped_product_variable_keeps_its_own_name(workspace):
    write_product(workspace, "CMAM", {"qrl_tot": {"units": "K/s", "levels": (0, 100)}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    extra = row_for(rows, "CMAM", "qrl_tot")
    assert extra["group"] == "" and extra["present"] is True
    assert extra["extension_status"] == "mapping_unresolved"


def test_waccmx_is_the_reference_and_never_a_row_of_its_own(workspace):
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    with pytest.raises(coverage.CoverageMatrixProblem, match="select at least one CCMI model"):
        coverage.split_models(coverage.product_models("all", START, END))
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 70)}})
    models, donor = coverage.split_models(coverage.product_models("CMAM,WACCM-X", START, END))
    assert (models, donor) == (["CMAM"], "WACCM-X")
    assert coverage.split_models(["GEOSCCM", "CMAM", "WACCM-X"])[1] == "WACCM-X"
    with pytest.raises(coverage.CoverageMatrixProblem, match="a WACCM-X product is required"):
        coverage.split_models(["CMAM", "GEOSCCM"])


def test_selected_models_must_have_a_product(workspace):
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6"}})
    assert coverage.product_models("cmam_refd1", START, END) == ["CMAM"]
    assert coverage.product_models("all", START, END) == ["CMAM"]
    with pytest.raises(coverage.CoverageMatrixProblem, match="no product"):
        coverage.product_models("CMAM,GEOSCCM", START, END)
    with pytest.raises(coverage.CoverageMatrixProblem, match="unknown model"):
        coverage.product_models("Nope", START, END)
    assert coverage.product_models("auto", START, END) == ["CMAM"]
    with pytest.raises(coverage.CoverageMatrixProblem, match="no climatology products"):
        coverage.product_models("all", START - 10, END - 10)


def test_a_native_grid_of_any_resolution_is_accepted(workspace):
    coarse = np.logspace(5, -1, 41)
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6"}}, native=coarse)
    entry = coverage.product_inventory("CMAM", START, END, SAMPLE)["fields"]["o3"]
    assert entry["native_level_count"] == 41
    assert entry["levels_usable"] == inside(levels(), float(coarse[0]), float(coarse[-1])).size
    assert float(entry["native_pressure_min_pa"]) == pytest.approx(float(coarse[-1]))


def test_a_product_on_the_application_pressure_grid_demands_a_rebuild(workspace):
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6"}}, native=levels())
    with pytest.raises(coverage.CoverageMatrixProblem, match="must be rebuilt with jumacs climatology"):
        coverage.product_inventory("CMAM", START, END, SAMPLE)


def test_a_product_on_the_application_grid_is_rejected_level_by_level(workspace):
    path = write_product(workspace, "CMAM", {"o3": {"units": "1e-6"}}, native=levels())
    with xr.open_dataset(path, decode_cf=False) as ds:
        with pytest.raises(cf.ProductProblem, match="application pressure grid"):
            cf.assert_product(ds, names=("o3",), grid=config.vertical_grid())


def test_a_three_dimensional_field_without_any_pressure_description_is_refused(workspace):
    path = write_product(workspace, "CMAM", {"o3": {"units": "1e-6"}})
    dataset = xr.load_dataset(path)
    dataset = dataset.rename({NATIVE_LEVEL: "ilev"}).assign_coords(
        ilev=("ilev", np.arange(dataset.sizes[NATIVE_LEVEL], dtype="float64")))
    dataset.to_netcdf(path)
    with pytest.raises(coverage.CoverageMatrixProblem, match="pressure-valued"):
        coverage.product_inventory("CMAM", START, END, SAMPLE)


def test_an_older_grouped_product_is_refused_rather_than_read(workspace):
    path = write_product(workspace, "CMAM", {"o3": {"units": "1e-6"}})
    xr.load_dataset(path).drop_vars("climatology_bounds").to_netcdf(path)
    with pytest.raises(coverage.CoverageMatrixProblem, match="climatology_bounds.*jumacs climatology"):
        coverage.product_inventory("CMAM", START, END, SAMPLE)


def test_files_keep_pa_in_machine_outputs_and_hpa_in_markdown(workspace):
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 70)},
                                      "ch4": {"units": "1e-6", "levels": (0, 70)},
                                      "ps": {"units": "Pa", "two_dimensional": True}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL},
                                         "PS": {"units": "Pa", "two_dimensional": True}})
    summary = coverage.coverage_matrix_command("CMAM,WACCM-X")
    assert "1985-2014: 1 CCMI product(s) compared with WACCM-X" in summary
    assert "extension candidates: CMAM O3" in summary
    table = read_csv(workspace / "products/comparison/coverage_matrix.csv")
    ozone, methane, surface = table[("CMAM", "O3")], table[("CMAM", "CH4")], table[("CMAM", "surface_pressure")]
    assert ozone["max_usable_pressure_pa"] == "100000.0"
    assert ozone["max_valid_pressure_pa"] == "100000.0" and ozone["waccmx_max_valid_pressure_pa"] == "100000.0"
    assert float(ozone["min_usable_pressure_pa"]) == pytest.approx(inside(levels(), *span((0, 70))).min())
    assert float(ozone["overlap_min_pressure_pa"]) == pytest.approx(inside(levels(), *span((0, 70))).min())
    assert float(ozone["waccmx_min_usable_pressure_pa"]) == pytest.approx(levels()[-1])
    assert float(ozone["native_pressure_min_pa"]) == pytest.approx(float(native_levels()[69]))
    assert int(ozone["native_level_count"]) == NATIVE_COUNT
    assert float(ozone["finite_fraction"]) == pytest.approx(70 / NATIVE_COUNT)
    assert float(ozone["grid_finite_fraction"]) == pytest.approx(inside(levels(), *span((0, 70))).size / LEVEL_COUNT)
    assert ozone["extension_status"] == "extension_candidate"
    assert methane["extension_status"] == "no_waccmx_variable"
    assert surface["extension_status"] == "not_applicable" and surface["levels_usable"] == ""
    assert surface["native_level_count"] == "0" and surface["grid_finite_fraction"] == ""
    assert all(column not in table[("CMAM", "O3")] for column in
               ("ccmi_common_min_usable_pressure_pa", "ccmi_union_pressure_range", "ccmi_all_present"))
    assert not any(model == "WACCM-X" for model, _ in table)
    document = json.loads((workspace / "products/comparison/coverage_matrix.json").read_text())
    assert document["ccmi_models"] == ["CMAM"] and document["donor_model"] == "WACCM-X"
    assert document["usable_sample_fraction"] == SAMPLE and document["usable_level_fraction"] == LEVEL
    assert document["extension_statuses"] == list(coverage.EXTENSION_STATUSES)
    assert document["pressure_coordinate"]["level_count"] == len(levels())
    assert document["donor_fields"]["O3"]["levels_usable"] == len(levels())
    assert document["products"]["CMAM"].endswith("jumacs_cmam_refd1_climatology_1985-2014.nc")
    record = row_for(document["rows"], "CMAM", "O3")
    reached = inside(levels(), *span((0, 70)))
    assert record["level_finite_fractions"] == [1.0 if pressure in reached else 0.0 for pressure in levels()]
    assert record["application_focus"] is True
    for key in ("max_valid_pressure_pa", "grid_finite_fraction", "native_level_count",
                "native_pressure_min_pa", "native_pressure_max_pa"):
        assert key in document["definitions"]
    markdown = (workspace / "products/comparison/coverage_matrix.md").read_text()
    assert "| CMAM | O3 |" in markdown
    assert "| Model | Variable | CCMI range | WACCM-X range | Usable overlap | Extends upward | Status |" in markdown
    assert "descriptive" in markdown and "1000 hPa counts" in markdown
    assert "1000 hPa" in markdown and "2e-05 hPa" in markdown and "no blending" in markdown
    assert markdown.count("| --- | --- | --- | --- | --- | --- | --- |") == 2
    assert "Mapped but absent from the WACCM-X product: CH4" in markdown
    assert "Two-dimensional, so no vertical extension applies: surface_pressure" in markdown


def test_broadly_usable_is_a_label_and_changes_no_decision(workspace):
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 40)}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    strict = coverage.coverage_matrix_command("CMAM,WACCM-X")
    ozone = read_csv(workspace / "products/comparison/coverage_matrix.csv")[("CMAM", "O3")]
    assert ozone["broadly_usable"] == "False" and ozone["extension_status"] == "extension_candidate"
    relaxed = coverage.coverage_matrix_command("CMAM,WACCM-X", usable_level_fraction=0.1)
    widened = read_csv(workspace / "products/comparison/coverage_matrix.csv")[("CMAM", "O3")]
    assert widened["broadly_usable"] == "True" and widened["extension_status"] == ozone["extension_status"]
    assert relaxed == strict


def test_matrices_are_reproducible_and_read_products_only(workspace, monkeypatch):
    product = write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 70)}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})

    def forbidden(*args, **kwargs):
        raise AssertionError("the coverage matrix must read climatology products only")

    monkeypatch.setattr(coverage, "open_cftime_dataset", forbidden)
    first = coverage.coverage_matrix_command("CMAM,WACCM-X")
    files = sorted((workspace / "products/comparison").glob("coverage_matrix.*"))
    assert len(files) == 3
    before = [path.read_bytes() for path in files]
    product_bytes = product.read_bytes()
    assert first == coverage.coverage_matrix_command("CMAM,WACCM-X")
    assert before == [path.read_bytes() for path in files]
    assert product.read_bytes() == product_bytes


def test_an_impossible_usable_fraction_is_refused(workspace):
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6"}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    with pytest.raises(coverage.CoverageMatrixProblem, match="coverage.usable_level_fraction"):
        coverage.coverage_matrix_command("CMAM,WACCM-X", usable_level_fraction=0.0)
    with pytest.raises(coverage.CoverageMatrixProblem, match="coverage.usable_level_fraction"):
        coverage.coverage_matrix_command("CMAM,WACCM-X", usable_level_fraction=1.5)
    with pytest.raises(coverage.CoverageMatrixProblem, match="coverage.usable_sample_fraction"):
        coverage.coverage_matrix_command("CMAM,WACCM-X", usable_sample_fraction=0.0)
