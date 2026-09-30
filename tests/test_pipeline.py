import json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from jumacs.config import load_config
from jumacs.archive import TARGETS
from jumacs.coordinates import hybrid_pressure
from jumacs.zonal import zonal_mean
from jumacs.climatology import monthly_climatology
from jumacs.comparison import compare_fields
from jumacs.download import select_files
from jumacs.reader import open_source
from jumacs.archive import _size_bytes


def test_configs_separate_and_yamls_not_boolean():
    g, e = load_config("GEOSCCM"), load_config("EMAC")
    assert g["model"]["dataset_uuid"] != e["model"]["dataset_uuid"]
    assert g["model"]["experiment"] == e["model"]["experiment"] == "refD1"
    assert g["coordinates"]["hybrid_a"] == "a"
    assert e["coordinates"]["hybrid_a"] == "ap"
    assert g["variables"]["NO"] == e["variables"]["NO"] == "no"
    assert "Halon-2402" in g["variables"] and "Halon-2402" not in e["variables"]
    assert "SF6" in TARGETS and "SF6" not in g["variables"] and "SF6" not in e["variables"]
    assert "CO2" not in g["variables"] and e["variables"]["CO2"] == "co2"


def test_zonal_valid_values_unweighted():
    data = xr.Dataset({"o3": (("time", "lev", "lat", "lon"), np.array([[[[1., 3., np.nan], [2., 4., 6.]]]]))},
                      coords={"time": [0], "lev": [1], "lat": [-30, 30], "lon": [0, 120, 240]})
    out = zonal_mean(data, "o3")
    np.testing.assert_allclose(out.values[0, 0], [2., 4.])


def test_arbitrary_period_statistics_and_counts():
    times = xr.cftime_range("1980-01", periods=12 * 40, freq="MS")
    values = np.array([t.year - 1980 for t in times], dtype=float)
    data = xr.DataArray(values, dims="time", coords={"time": times}, attrs={"units": "mol mol-1"})
    out = monthly_climatology(data, 2000, 2018)
    assert out["mean"].sel(month=1).item() == 29
    assert out["n_years"].sel(month=1).item() == 19
    assert out["minimum"].sel(month=1).item() == 20
    assert out["maximum"].sel(month=1).item() == 38


def test_hybrid_pressure_model_specific_formulas():
    ps = xr.DataArray([[100000.]], dims=("lat", "lon"))
    geos = xr.Dataset({"a": ("lev", [.1, .2]), "b": ("lev", [.5, .3]), "p0": 100000., "ps": ps}, coords={"lev": [1, 2]})
    emac = xr.Dataset({"ap": ("lev", [10000., 20000.]), "b": ("lev", [.5, .3]), "ps": ps}, coords={"lev": [1, 2]})
    np.testing.assert_allclose(hybrid_pressure(geos, load_config("GEOSCCM")).values[:, 0, 0], [60000, 50000])
    np.testing.assert_allclose(hybrid_pressure(emac, load_config("EMAC")).values[:, 0, 0], [60000, 50000])


def test_comparison_interpolates_overlap_only():
    g = xr.DataArray([[[1., 2.], [3., 4.]]], dims=("month", "lat", "lev"),
                     coords={"month": [1], "lat": [-10., 10.], "lev": [100000., 10000.]}, attrs={"units": "mol mol-1"})
    e = xr.DataArray([[[2., 3.], [4., 5.], [6., 7.]]], dims=("month", "lat", "plev"),
                     coords={"month": [1], "lat": [-20., 0., 20.], "plev": [100000., 10000.]}, attrs={"units": "mol mol-1"})
    result = compare_fields(g, e, g.lev, e.plev, [100000., 10000., 1000.])
    assert list(result.lat.values) == [0.]
    assert np.isnan(result["difference"].sel(pressure=1000.)).all()
    assert set(result.data_vars) == {"GEOSCCM", "EMAC", "difference", "relative_difference"}


def test_manifest_selection_missing_variable(monkeypatch):
    from jumacs import download
    inventory = {"files": [{"variable": "cfcl3", "family": "AmonZ", "start": "196001", "end": "201812"}]}
    monkeypatch.setattr(download, "load_inventory", lambda model: inventory)
    selected, missing = select_files("EMAC", 2000, 2018, ["CFC-11", "SF6"])
    assert len(selected) == 1 and missing == ["SF6"]


def test_reader_preserves_model_metadata_and_rejects_other_model(tmp_path):
    path = tmp_path / "o3_AmonZ_GEOSCCM_refD1_r1i1p1f1_grz_200001-200012.nc"
    ds = xr.Dataset({"o3": (("time", "lev", "lat"), np.ones((1, 2, 1)), {"units": "mol mol-1"})},
                    coords={"time": xr.cftime_range("2000-01", periods=1), "lev": [100000., 10000.], "lat": [0.]},
                    attrs={"institution": "synthetic"})
    ds.to_netcdf(path)
    with open_source(path, "GEOSCCM", "O3") as opened:
        assert opened.attrs["model"] == "GEOSCCM"
        assert opened.attrs["institution"] == "synthetic"
        assert opened.o3.attrs["units"] == "mol mol-1"
    with pytest.raises(ValueError):
        open_source(path, "EMAC", "O3")


def test_archive_size_parser_and_sf6_absence_in_synthetic_inventory():
    assert _size_bytes("4.0 MB") == 4_000_000
    inventory = [{"variable": "cfcl3"}, {"variable": "cf2cl2"}]
    assert not any(row["variable"] == "sf6" for row in inventory)


def test_comparison_rejects_unit_mismatch():
    field = xr.DataArray([1., 2.], dims="lev", coords={"lev": [100000., 10000.]}, attrs={"units": "mol mol-1"})
    other = field.copy(); other.attrs["units"] = "kg kg-1"
    with pytest.raises(ValueError, match="units"):
        compare_fields(field, other, field.lev, other.lev, [50000.])


def test_additional_chemistry_prefers_small_zonal_family(monkeypatch):
    from jumacs import download
    inventory = {"files": [
        {"variable": "oh", "family": "Amon", "start": "196001", "end": "201812"},
        {"variable": "oh", "family": "AmonZ", "start": "196001", "end": "201812"},
        {"variable": "ta", "family": "Amon", "start": "196001", "end": "201812"},
        {"variable": "ta", "family": "AmonZ", "start": "196001", "end": "201812"},
    ]}
    monkeypatch.setattr(download, "load_inventory", lambda model: inventory)
    for model in ("GEOSCCM", "EMAC"):
        config = load_config(model)
        assert config["variables"]["temperature"] == "ta"
        assert config["variables"]["geopotential_height"] == "zg"
        selected, missing = select_files(model, 2000, 2018, ["OH", "temperature"])
        assert not missing
        assert {(row["variable"], row["family"]) for row in selected} == {("oh", "AmonZ"), ("ta", "Amon")}


def test_duplicate_archive_versions_select_latest(monkeypatch):
    from jumacs import download
    inventory = {"files": [
        {"variable": "vtem", "family": "AmonZ", "grid": "grz", "version": "v20211012", "start": "196001", "end": "199912"},
        {"variable": "vtem", "family": "AmonZ", "grid": "grz", "version": "v20220415", "start": "196001", "end": "199912"},
    ]}
    monkeypatch.setattr(download, "load_inventory", lambda model: inventory)
    selected, missing = select_files("GEOSCCM", 1980, 2018, ["TEM_meridional_wind"])
    assert not missing
    assert len(selected) == 1 and selected[0]["version"] == "v20220415"
    g, e = load_config("GEOSCCM"), load_config("EMAC")
    assert g["preferred_families"]["ua"] == e["preferred_families"]["ua"] == "AmonZ"
    assert "tropopause_altitude" not in g["variables"]
    assert e["variables"]["tropopause_altitude"] == "ztp"
