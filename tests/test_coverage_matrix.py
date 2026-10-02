"""Extension coverage is built from published products and never changes them."""
import csv
import json
import numpy as np
import pytest
import xarray as xr
from jumacs import cf, config, coverage

START, END = 1985, 2014
LATS = (-45.0, 0.0, 45.0)
SAMPLE, LEVEL = 0.9, 0.5
FULL = (0, 101)


def levels():
    return np.asarray(config.vertical_grid()["levels"], "float64")


def write_product(workspace, model, fields, grid_levels=None):
    """A minimal but contract-valid climatology product for one model."""
    coordinate = config.vertical_grid()["coordinate"]
    pressure = np.asarray(grid_levels if grid_levels is not None else levels(), "float64")
    reference = config.reference_period()["reference_period"]["nominal_reference_year"]
    time = cf.climatology_time(START, END, reference)
    variables = {"climatology_bounds": (("time", "nv"), time["bounds"], {"units": time["units"]})}
    for stem, spec in fields.items():
        flat = spec.get("two_dimensional", False)
        dims = ("time", "lat") if flat else ("time", coordinate, "lat")
        shape = (12, len(LATS)) if flat else (12, pressure.size, len(LATS))
        mean = np.zeros(shape, "float32") if flat else np.full(shape, np.nan, "float32")
        if flat:
            mean[...] = spec.get("value", 1.0)
        else:
            for window in [spec.get("levels", (0, pressure.size))] + spec.get("blocks", []):
                months = window[2] if len(window) > 2 else spec.get("sample_count")
                block = slice(None) if months is None else slice(0, months)
                mean[block, window[0]:window[1], :] = spec.get("value", 1.0)
        for statistic in cf.STATISTIC_ORDER:
            values = (np.ones(shape, "int16") if statistic == "n_years" else
                      (mean if statistic == "mean" else np.abs(mean).astype("float32")))
            units = "1" if statistic == "n_years" else spec["units"]
            variables[f"{stem}_{statistic}"] = (dims, values, {"units": units,
                                                               "cell_methods": cf.cell_methods(statistic)})
    coordinates = {
        "time": (("time",), time["values"], dict(time["attrs"])),
        "pressure": (("pressure",), pressure, {"standard_name": "air_pressure", "units": "Pa",
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
    assert ozone["extension_status"] == "extension_candidate"
    assert ozone["min_usable_pressure_pa"] == pytest.approx(grid[69])
    assert ozone["max_usable_pressure_pa"] == pytest.approx(grid[0])
    assert ozone["waccmx_min_usable_pressure_pa"] == pytest.approx(grid[100])
    assert ozone["overlap_level_count"] == 70
    assert ozone["overlap_min_pressure_pa"] == pytest.approx(grid[69])
    assert ozone["overlap_max_pressure_pa"] == pytest.approx(grid[0])
    assert ozone["waccmx_extends_upward"] is True
    assert ozone["application_focus"] is True and ozone["group"] == "ozone"


def test_the_lower_extent_is_the_valid_extent_so_terrain_thinning_does_not_hide_1000_hpa(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (1, 70), "blocks": [(0, 1, 6)]}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["max_valid_pressure_pa"] == pytest.approx(100000.0) == pytest.approx(grid[0])
    assert ozone["max_any_pressure_pa"] == ozone["max_valid_pressure_pa"]
    assert ozone["max_usable_pressure_pa"] == pytest.approx(grid[1])
    assert ozone["level_finite_fractions"][0] == pytest.approx(0.5)
    assert ozone["levels_any_finite"] == 70 and ozone["levels_usable"] == 69
    assert ozone["contiguous_usable"] is True


def test_a_sparse_upper_level_does_not_become_the_usable_top(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 70), "blocks": [(90, 91, 3)]}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["min_any_pressure_pa"] == pytest.approx(grid[90])
    assert ozone["min_usable_pressure_pa"] == pytest.approx(grid[69])
    assert ozone["min_any_pressure_pa"] < ozone["min_usable_pressure_pa"]
    assert ozone["overlap_min_pressure_pa"] == pytest.approx(grid[69])
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
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": (60, 101)}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["extension_status"] == "no_overlap"
    assert ozone["overlap_exists"] is False and ozone["overlap_level_count"] == 0
    assert ozone["overlap_min_pressure_pa"] is None and ozone["overlap_max_pressure_pa"] is None
    assert ozone["waccmx_extends_upward"] is False
    assert ozone["min_usable_pressure_pa"] == pytest.approx(grid[19])
    assert ozone["waccmx_min_usable_pressure_pa"] == pytest.approx(grid[100])


def test_overlap_without_upward_reach_is_not_a_candidate(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": FULL}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": (0, 50)}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["extension_status"] == "overlap_no_extension"
    assert ozone["overlap_level_count"] == 50
    assert ozone["overlap_min_pressure_pa"] == pytest.approx(grid[49])
    assert ozone["waccmx_extends_upward"] is False


def test_overlap_is_taken_from_the_masks_so_internal_gaps_stay_visible(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 20), "blocks": [(60, 81)]}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["levels_usable"] == 41
    assert ozone["contiguous_usable"] is False
    assert ozone["min_usable_pressure_pa"] == pytest.approx(grid[80])
    assert ozone["max_usable_pressure_pa"] == pytest.approx(grid[0])
    assert ozone["overlap_level_count"] == 41
    assert ozone["extension_status"] == "extension_candidate"
    assert ozone["level_finite_fractions"].count(0.0) == 60


def test_a_level_needs_most_of_its_samples_before_it_counts(workspace):
    grid = levels()
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6", "levels": (0, 70), "sample_count": 6}})
    write_product(workspace, "WACCM-X", {"O3": {"units": "1e-6", "levels": FULL}})
    _, rows = matrix(["CMAM", "WACCM-X"])
    ozone = row_for(rows, "CMAM", "O3")
    assert ozone["levels_any_finite"] == 70 and ozone["levels_usable"] == 0
    assert ozone["min_usable_pressure_pa"] is None and ozone["min_any_pressure_pa"] == pytest.approx(grid[69])
    assert ozone["contiguous_usable"] is None and ozone["broadly_usable"] is False
    assert ozone["extension_status"] == "no_overlap"
    _, relaxed = matrix(["CMAM", "WACCM-X"], sample_fraction=0.5)
    assert row_for(relaxed, "CMAM", "O3")["extension_status"] == "extension_candidate"


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


def test_a_product_on_another_pressure_grid_is_refused(workspace):
    write_product(workspace, "CMAM", {"o3": {"units": "1e-6"}}, grid_levels=np.logspace(5, -1, 41))
    with pytest.raises(coverage.CoverageMatrixProblem, match="configured common grid"):
        coverage.product_inventory("CMAM", START, END, SAMPLE)


def test_an_older_grouped_product_is_refused_rather_than_read(workspace):
    path = write_product(workspace, "CMAM", {"o3": {"units": "1e-6"}})
    xr.load_dataset(path).drop_vars("climatology_bounds").to_netcdf(path)
    with pytest.raises(coverage.CoverageMatrixProblem, match="climatology_bounds.*jumacs build"):
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
    assert float(ozone["min_usable_pressure_pa"]) == pytest.approx(levels()[69])
    assert float(ozone["overlap_min_pressure_pa"]) == pytest.approx(levels()[69])
    assert float(ozone["waccmx_min_usable_pressure_pa"]) == pytest.approx(levels()[100])
    assert ozone["extension_status"] == "extension_candidate"
    assert methane["extension_status"] == "no_waccmx_variable"
    assert surface["extension_status"] == "not_applicable" and surface["levels_usable"] == ""
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
    assert record["level_finite_fractions"] == [1.0] * 70 + [0.0] * (len(levels()) - 70)
    assert record["application_focus"] is True
    assert "max_valid_pressure_pa" in document["definitions"]
    markdown = (workspace / "products/comparison/coverage_matrix.md").read_text()
    assert "| CMAM | O3 |" in markdown
    assert "| Model | Variable | CCMI range | WACCM-X range | Usable overlap | Extends upward | Status |" in markdown
    assert "descriptive" in markdown and "1000 hPa counts" in markdown
    assert "1000 hPa" in markdown and "1e-07 hPa" in markdown and "no blending" in markdown
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
