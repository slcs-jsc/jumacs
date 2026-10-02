import json
import numpy as np
import pytest
import xarray as xr

from jumacs.build import build_model, build_models, configured_variables, model_list
from jumacs.config import load_config, vertical_grid


class MissingRawSources:
    """Stand in for zonal processing so unit tests never touch a real raw archive."""

    def __init__(self):
        self.calls = []

    def __call__(self, model, name):
        self.calls.append(name)
        raise FileNotFoundError(f"no downloaded monthly files for {model} {name}")


def isolate(tmp_path, monkeypatch):
    """Confine build tests to tmp_path and replace the directly imported zonal entry point."""
    import jumacs.build as build_module
    import jumacs.climatology as climatology_module
    import jumacs.zonal as zonal_module
    missing = MissingRawSources()
    monkeypatch.setattr(build_module, "ROOT", tmp_path)
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    monkeypatch.setattr(zonal_module, "ROOT", tmp_path)
    monkeypatch.setattr(build_module, "build_zonal", missing)
    return missing


def synthetic_zonal(path, name, value, with_pressure):
    times = xr.date_range("1985-01", periods=30 * 12, freq="MS", use_cftime=True)
    coords = {"time": times, "lev": xr.DataArray([100000., 10000., 1000.], dims="lev", attrs={"units": "Pa"}), "lat": [-45., 45.]}
    data = xr.Dataset({name: (("time", "lev", "lat"), np.full((360, 3, 2), value), {"units": "mol mol-1"})}, coords=coords)
    if with_pressure:
        data["air_pressure"] = (("time", "lev", "lat"), np.broadcast_to(np.array([100000., 10000., 1000.])[None, :, None], (360, 3, 2)).copy(), {"units": "Pa"})
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(path)


def test_build_model_processes_every_configured_variable_and_keeps_them_all(tmp_path, monkeypatch):
    missing = isolate(tmp_path, monkeypatch)
    zonal = tmp_path / load_config("CMAM")["paths"]["zonal"]
    synthetic_zonal(zonal / "ta_monthly_zonal.nc", "ta", 220.0, True)
    synthetic_zonal(zonal / "o3_monthly_zonal.nc", "o3", 8e-6, True)
    synthetic_zonal(zonal / "h2o_monthly_zonal.nc", "h2o", 5e-6, False)
    configured = set(configured_variables("CMAM", load_config("CMAM")))
    report = build_model("CMAM", 1985, 2014)
    assert report["ok"] is True
    assert report["climatology_period"] == "1985-2014"
    assert set(report["variables_processed"]) == {"ta", "o3", "h2o"}
    assert report["zonal_built"] == []
    assert set(report["zonal_reused"]) == {"ta", "o3", "h2o"}
    assert set(missing.calls) == configured - {"ta", "o3", "h2o"}
    assert len(report["skipped"]) == len(missing.calls) == len(configured) - 3
    assert all("no downloaded monthly files" in row["reason"] for row in report["skipped"])
    combined = xr.open_dataset(report["product"], decode_cf=False)
    assert report["conventions"] == "CF-1.13"
    assert report["variables_in_product"] == 3
    levels = combined["lev"].values
    assert np.array_equal(levels, np.array([100000., 10000., 1000.]))
    assert combined.attrs["vertical_level_count"] == 3 and combined.attrs["vertical_level_counts"] == "lev=3"
    for name, value in (("ta", 220.0), ("o3", 8e-6), ("h2o", 5e-6)):
        mean = combined[f"{name}_mean"].isel(time=0)
        assert np.allclose(mean.values, value)
        assert combined[f"{name}_n_years"].isel(time=0).max().item() == 30
        check = report["checks"][name]
        assert check["dimensions"] == ["time", "lev", "lat"]
        assert check["vertical_coordinate"] == "lev" and check["vertical_level_count"] == 3
        assert check["n_years_max"] == 30 and check["n_years_min"] == 30
        assert check["finite_fraction"] == 1.0
        assert check["pressure_min_pa"] == 1000.0 and check["pressure_max_pa"] == 100000.0
        assert combined[f"{name}_mean"].attrs["native_level_count"] == 3
        assert combined[f"{name}_mean"].attrs["on_application_pressure_grid"] == "false"
    assert report["checks"]["ta"]["pressure_plausible"] and report["checks"]["ta"]["pressure_monotonic"]
    assert report["checks"]["ta"]["pressure_representation"] == "profile"
    assert report["checks"]["h2o"]["pressure_representation"] == "coordinate"
    grid = report["application_pressure_grid"]
    assert grid["level_count"] == len(vertical_grid()["levels"]) == 124
    assert grid["pressure_min_pa"] == 0.002 and grid["pressure_max_pa"] == 100000.0
    assert grid["applied_at"].startswith("combination")
    assert report["vertical"]["vertical_level_count"] == "3"
    assert f"{grid['pressure_min_pa']:g}" == "0.002"
    assert combined.attrs["variable_count"] == 3
    assert combined.attrs["vertical_coordinate"].startswith("'lev'")
    assert combined.attrs["vertical_coordinate"].endswith("standard_name air_pressure)")
    assert combined.attrs["application_pressure_grid"].startswith("124 levels from 100000 to 0.002 Pa")
    assert combined.attrs["vertical_interpolation"].startswith("none")
    assert combined.attrs["vertical_extrapolation"].startswith("none")
    assert combined["lev"].attrs["units"] == "Pa"
    assert combined["lev"].attrs["standard_name"] == "air_pressure"
    assert not [dim for dim in combined.dims if dim.endswith(("_lev", "_plev"))]
    assert combined["o3_mean"].attrs["source_variable"] == "o3"
    assert combined["o3_mean"].attrs["cell_methods"] == "longitude: mean time: mean within years time: mean over years"
    assert combined["o3_mean"].attrs["native_pressure_min_pa"] == 1000.0
    assert combined["o3_sigma"].attrs["cell_methods"].endswith("time: standard_deviation over years")
    assert combined["o3_mean"].dtype == "float32"
    assert combined["o3_n_years"].dtype == "int16"
    assert combined["time"].dtype == "float64"
    assert combined["time"].attrs["climatology"] == "climatology_bounds"
    assert "time:bounds" not in combined["time"].attrs
    assert combined["climatology_bounds"].dims == ("time", "nv")
    assert "source_units" not in combined["o3_n_years"].attrs
    assert set(combined["o3_n_years"].attrs) == {"units", "long_name", "cell_methods", "source_variable",
                                                "pressure_coordinate"}
    combined.close()


def test_build_model_reports_a_period_without_data(tmp_path, monkeypatch):
    missing = isolate(tmp_path, monkeypatch)
    synthetic_zonal(tmp_path / load_config("CMAM")["paths"]["zonal"] / "ta_monthly_zonal.nc", "ta", 220.0, True)
    report = build_model("CMAM", 1950, 1960)
    assert report["ok"] is False
    assert report["variables_processed"] == []
    skipped = {row["variable"]: row["reason"] for row in report["skipped"]}
    assert "outside 1950-1960" in skipped["ta"]
    assert "no downloaded monthly files" in skipped["br"]
    assert "br" in missing.calls and "ta" not in missing.calls


def test_model_list_accepts_one_model_comma_lists_and_all():
    assert model_list("CMAM") == ("CMAM",)
    assert model_list("CMAM, SOCOL") == ("CMAM", "SOCOL")
    assert {"CMAM", "SOCOL", "GEOSCCM", "EMAC", "WACCM-X"} <= set(model_list("all"))
    assert "MIROC-ES2H" not in model_list("all")
    with pytest.raises(ValueError, match="unknown model"):
        model_list("CMAM,NOPE")
    reports = list(build_models("MIROC-ES2H"))
    assert reports[0]["ok"] is False and "unresolved" in reports[0]["error"]
