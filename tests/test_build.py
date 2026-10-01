import json
import numpy as np
import pytest
import xarray as xr

from jumacs.build import build_model, build_models, model_list
from jumacs.config import load_config


def synthetic_zonal(path, name, value, with_pressure):
    times = xr.date_range("1985-01", periods=30 * 12, freq="MS", use_cftime=True)
    coords = {"time": times, "lev": xr.DataArray([100000., 10000., 1000.], dims="lev", attrs={"units": "Pa"}), "lat": [-45., 45.]}
    data = xr.Dataset({name: (("time", "lev", "lat"), np.full((360, 3, 2), value), {"units": "mol mol-1"})}, coords=coords)
    if with_pressure:
        data["air_pressure"] = (("time", "lev", "lat"), np.broadcast_to(np.array([100000., 10000., 1000.])[None, :, None], (360, 3, 2)).copy(), {"units": "Pa"})
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(path)


def test_build_model_processes_every_configured_variable_and_keeps_them_all(tmp_path, monkeypatch):
    import jumacs.build as build_module
    import jumacs.climatology as climatology_module
    monkeypatch.setattr(build_module, "ROOT", tmp_path)
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    zonal = tmp_path / load_config("CMAM")["paths"]["zonal"]
    synthetic_zonal(zonal / "ta_monthly_zonal.nc", "ta", 220.0, True)
    synthetic_zonal(zonal / "o3_monthly_zonal.nc", "o3", 8e-6, True)
    synthetic_zonal(zonal / "h2o_monthly_zonal.nc", "h2o", 5e-6, False)
    report = build_model("CMAM", 1985, 2014)
    assert report["ok"] is True
    assert report["climatology_period"] == "1985-2014"
    assert set(report["variables_processed"]) == {"ta", "o3", "h2o"}
    assert report["zonal_built"] == []
    assert set(report["zonal_reused"]) == {"ta", "o3", "h2o"}
    assert len(report["skipped"]) == len(load_config("CMAM")["variables"]) - 3
    assert all("no downloaded monthly files" in row["reason"].lower() for row in report["skipped"])
    combined = xr.open_dataset(report["combined_product"])
    for name, value in (("ta", 220.0), ("o3", 8e-6), ("h2o", 5e-6)):
        assert np.allclose(combined[f"{name}_mean"].sel(month=1).values, value)
        assert combined[f"{name}_n_years"].sel(month=1).min().item() == 30
        assert report["checks"][name]["finite_fraction"] == 1.0
        assert report["checks"][name]["n_years_min"] == 30
    assert report["checks"]["ta"]["pressure_plausible"] and report["checks"]["ta"]["pressure_monotonic"]
    assert report["checks"]["ta"]["dimensions"] == ["month", "ta_lev", "lat"]
    assert report["checks"]["h2o"]["dimensions"] == ["month", "h2o_lev", "lat"]
    assert combined.attrs["variable_count"] == 3
    assert combined["o3_mean"].attrs["source_variable"] == "o3"
    assert combined["o3_air_pressure"].attrs["source_variable"] == "o3"
    assert combined["o3_air_pressure"].attrs["units"] == "Pa"
    assert "source_units" not in combined["o3_air_pressure"].attrs
    combined.close()


def test_build_model_reports_a_period_without_data(tmp_path, monkeypatch):
    import jumacs.build as build_module
    import jumacs.climatology as climatology_module
    monkeypatch.setattr(build_module, "ROOT", tmp_path)
    monkeypatch.setattr(climatology_module, "ROOT", tmp_path)
    synthetic_zonal(tmp_path / load_config("CMAM")["paths"]["zonal"] / "ta_monthly_zonal.nc", "ta", 220.0, True)
    report = build_model("CMAM", 1950, 1960)
    assert report["ok"] is False
    skipped = {row["variable"]: row["reason"] for row in report["skipped"]}
    assert "outside 1950-1960" in skipped["ta"]
    assert "no downloaded monthly files" in skipped["br"].lower()


def test_model_list_accepts_one_model_comma_lists_and_all():
    assert model_list("CMAM") == ("CMAM",)
    assert model_list("CMAM, SOCOL") == ("CMAM", "SOCOL")
    assert {"CMAM", "SOCOL", "GEOSCCM", "EMAC", "WACCM-X"} <= set(model_list("all"))
    assert "MIROC-ES2H" not in model_list("all")
    with pytest.raises(ValueError, match="unknown model"):
        model_list("CMAM,NOPE")
    reports = list(build_models("MIROC-ES2H"))
    assert reports[0]["ok"] is False and "unresolved" in reports[0]["error"]
