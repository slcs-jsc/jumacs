import numpy as np
import xarray as xr
import cftime
from pathlib import Path

from jumacs.config import load_config, reference_period
from jumacs.archive import TARGETS
from jumacs.climatology import monthly_climatology, product_name, build_climatology
from jumacs.coverage import coverage_for_field
from jumacs.coordinates import hybrid_pressure
from jumacs.comparison import compare_fields, compatible_units
from jumacs.evaluation import evaluate_fields
from jumacs.reader import open_source
from jumacs.download import select_files
from jumacs.zonal import zonal_mean


def test_reference_and_waccmx_config():
    period = reference_period()["reference_period"]
    assert (period["start_year"], period["end_year"], period["nominal_reference_year"]) == (1985, 2014, 2000)
    assert load_config("WACCM-X")["model"]["dataset_uuid"] == "dc91f5e39ae34fd883af81dfdbaf659c"
    assert product_name("GEOSCCM", 1985, 2014) == "jumacs_geosccm_refd1_climatology_1985-2014.nc"
    assert "SF6" in TARGETS and "CFC-115" in TARGETS
    assert compatible_units("mol/mol", "mol mol-1")
    assert not compatible_units("kg kg-1", "mol mol-1")


def test_default_period_statistics_have_30_years():
    times = xr.date_range("1980-01", periods=39*12, freq="MS", use_cftime=True)
    values = xr.DataArray([t.year for t in times], dims="time", coords={"time": times})
    result = monthly_climatology(values, 1985, 2014)
    assert result.n_years.sel(month=1).item() == 30
    assert result.minimum.sel(month=1).item() == 1985
    assert result.maximum.sel(month=1).item() == 2014


def test_waccmx_ready_made_zonal_reader_and_pressure(tmp_path):
    name = "f.e210.FXHIST.f19_f19.h1a.cam.h0.2000-01_zm.nc"
    ds = xr.Dataset({"O3": (("time", "lev", "lat"), np.ones((1,2,2)), {"units": "mol/mol"}),
        "hyam": ("lev", [0.1, 0.2]), "hybm": ("lev", [0.5, 0.3]),
        "P0": 100000., "PS": (("time", "lat"), [[100000.,100000.]])},
        coords={"time": xr.date_range("2000-01", periods=1, use_cftime=True), "lev": [1,2], "lat": [-10.,10.]})
    path = tmp_path / name; ds.to_netcdf(path)
    with open_source(path, "WACCM-X", "O3") as opened:
        assert opened.O3.dims == ("time", "lev", "lat")
        assert opened.attrs["model"] == "WACCM-X"
        assert opened.time.dt.month.item() == 1
        assert "already zonal" in zonal_mean(opened, "O3", source_kind="monthly_zonal_multivariable").attrs["zonal_method"]
        pressure = hybrid_pressure(opened, load_config("WACCM-X"))
        np.testing.assert_allclose(pressure.values[:,0,0], [60000.,50000.])


def test_waccmx_uses_time_bounds_month_not_endpoint(tmp_path):
    filename = "f.e210.FXHIST.f19_f19.h1a.cam.h0.2014-12_zm.nc"
    start = cftime.DatetimeGregorian(2014,12,1)
    end = cftime.DatetimeGregorian(2015,1,1)
    ds = xr.Dataset({"O3": (("time","lev","lat"), np.ones((1,1,1))),
        "time_bnds": (("time","nbnd"), [[start,end]])},
        coords={"time":[end],"lev":[1.],"lat":[0.]})
    path = tmp_path / filename; ds.to_netcdf(path)
    with open_source(path,"WACCM-X","O3") as opened:
        assert int(opened.time.dt.year.item()) == 2014
        assert int(opened.time.dt.month.item()) == 12
        assert opened.attrs["original_time_coordinate"].startswith("2015-01")


def test_valid_vertical_coverage_and_relative_difference():
    pressure = xr.DataArray([100000.,10000.,1000.], dims="lev", coords={"lev":[0,1,2]}, attrs={"units":"Pa"})
    field = xr.DataArray([1.,2.,np.nan], dims="lev", coords={"lev":[0,1,2]})
    extent = coverage_for_field(field, pressure)
    assert extent["lowest_valid_pressure_pa"] == 100000.
    assert extent["highest_valid_pressure_pa"] == 10000.
    assert extent["vertical_levels"] == 3
    native_height = xr.DataArray([1000.,20000.,100000.], dims="lev", coords={"lev":[0,1,2]})
    native_extent = coverage_for_field(field, pressure, native_height)
    assert native_extent["approximate_altitude_max_km"] == 20.
    assert native_extent["median_upper_valid_altitude_km"] == 20.
    assert native_extent["altitude_method"] == "native geopotential height"
    a = xr.DataArray([[[2.,4.]]], dims=("month","lat","lev"), coords={"month":[1],"lat":[0.],"lev":[100000.,10000.]}, attrs={"units":"mol/mol"})
    b = a/2; b.attrs["units"] = "mol/mol"
    out = compare_fields(a,b,a.lev,b.lev,[50000.,1000.])
    assert "relative_difference" in out
    assert np.isnan(out.difference.sel(pressure=1000.)).all()
    evaluation = evaluate_fields(a,b,a.lev,b.lev,[50000.])
    assert float(evaluation.rms_difference.sel(month=1)) > 0


def test_evaluation_pressure_coordinate_and_month_overlap():
    model = xr.DataArray(np.ones((2,2,2)), dims=("month","lat","lev"),
        coords={"month":[1,2],"lat":[-10.,10.],"lev":[100000.,10000.]}, attrs={"units":"mol mol-1"})
    observed = xr.DataArray(np.ones((1,2,2))*2, dims=("month","lat","pressure"),
        coords={"month":[1],"lat":[-10.,10.],"pressure":[100000.,10000.]}, attrs={"units":"mol/mol"})
    result = evaluate_fields(model,observed,model.lev,observed.pressure,[50000.,1000.])
    assert result.sizes["month"] == 1
    assert np.isnan(result.difference.sel(pressure=1000.)).all()


def test_climatology_provenance_and_grouped_product(tmp_path, monkeypatch):
    import jumacs.climatology as module
    monkeypatch.setattr(module, "ROOT", tmp_path)
    zonal = tmp_path / "data/processed/GEOSCCM/refD1"; zonal.mkdir(parents=True)
    times = xr.date_range("1985-01", periods=30*12, freq="MS", use_cftime=True)
    data = xr.Dataset({"o3": (("time","plev","lat"), np.ones((360,2,1)), {"units":"mol/mol"})},
        coords={"time":times,"plev":xr.DataArray([100000.,10000.],dims="plev",attrs={"units":"Pa","standard_name":"air_pressure"}),"lat":[0.]})
    data.to_netcdf(zonal / "o3_monthly_zonal.nc")
    outputs = build_climatology("GEOSCCM",1985,2014,["O3"])
    assert len(outputs) == 2
    with xr.open_dataset(outputs[-1]) as root:
        assert root.attrs["nominal_reference_year"] == 2000
        assert root.attrs["reference_period_years"] == 30
        assert root.attrs["model"] == "GEOSCCM"
    with xr.open_dataset(outputs[-1], group="variables/o3") as group:
        assert group.n_years.sel(month=1).min().item() == 30
        assert group.attrs["bias_correction"] == "none"


def test_availability_matrix_keeps_models_separate(tmp_path, monkeypatch):
    import csv
    import jumacs.archive as archive
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    inventories = {
        "GEOSCCM": {"files":[{"variable":"cfcl3","start":"198501","end":"201412"}],"catalogue_metadata":{"cfcl3":{"units":"mol mol-1"}}},
        "EMAC": {"files":[],"catalogue_metadata":{}},
        "WACCM-X": {"files":[{"variable":"*","start":"195001","end":"201512"}],"dimensions":{"CFC11":["time","lev","lat"]},"catalogue_metadata":{"CFC11":{"units":"mol/mol"}}},
    }
    monkeypatch.setattr(archive, "load_inventory", lambda model: inventories.get(model))
    path = archive.variable_matrix()
    rows = {r["canonical_species"]: r for r in csv.DictReader(path.open())}
    assert rows["CFC-11"]["GEOSCCM_available"] == "True"
    assert rows["CFC-11"]["GEOSCCM_status"] == "ok"
    assert rows["CFC-11"]["GEOSCCM_time_coverage"] == "198501-201412"
    assert rows["CFC-11"]["EMAC_available"] == "False"
    assert rows["CFC-11"]["EMAC_status"] == "absent"
    assert rows["CFC-11"]["EMAC_time_coverage"] == ""
    assert rows["CFC-11"]["WACCM-X_available"] == "True"
    assert all(rows["SF6"][m+"_available"] == "False" for m in inventories)
    stub = next(m for m in archive.model_names() if m not in inventories and m != "WACCM-X")
    assert rows["CFC-11"][stub+"_status"] in {"unresolved", "not_inspected"}
    assert rows["CFC-11"][stub+"_time_coverage"] != "unresolved"


def test_native_variable_report_separates_mapped_and_native_only(tmp_path, monkeypatch):
    import csv
    import jumacs.archive as archive
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    inventories = {
        "GEOSCCM": {"files":[{"variable":"o3","start":"198501","end":"201412"}],"catalogue_metadata":{}},
        "WACCM-X": {"files":[{"variable":"*","start":"195001","end":"201512"}],
                    "dimensions":{"O3":["time","lev","lat"], "N2_vmr":["time","lev","lat"]}, "catalogue_metadata":{}},
    }
    monkeypatch.setattr(archive, "load_inventory", lambda model: inventories.get(model))
    rows = {(r["model"], r["native_variable"]): r for r in csv.DictReader(archive.native_variable_report().open())}
    assert rows[("WACCM-X", "O3")]["mapping_status"] == "mapped"
    assert rows[("WACCM-X", "O3")]["canonical_species"] == "O3"
    assert rows[("WACCM-X", "N2_vmr")]["mapping_status"] == "mapped"
    assert rows[("WACCM-X", "N2_vmr")]["canonical_species"] == "N2"
    assert rows[("GEOSCCM", "o3")]["mapping_status"] == "mapped"


def test_capabilities_gate_commands():
    from jumacs.config import models_with_capability, has_capability
    assert set(models_with_capability("compact_waccmx")) == {"GEOSCCM", "EMAC"}
    assert has_capability("WACCM-X", "zonal_processing") and not has_capability("WACCM-X", "compact_waccmx")
    assert set(models_with_capability("archive_inventory")) == {
        "GEOSCCM", "EMAC", "WACCM-X", "ACCESS-CM2-Chem", "CCSR-NIES-MIROC32", "CESM2-WACCM", "CMAM",
        "CNRM-MOCAGE", "IPSL-CM6A-ATM-LR-REPROBUS", "NIWA-UKCA2", "SOCOL", "UKESM1-StratTrop"}
    assert set(models_with_capability("zonal_processing")) == {"GEOSCCM", "EMAC", "WACCM-X"}
    assert not any(has_capability("MIROC-ES2H", cap) for cap in
                   ("archive_inventory", "download", "zonal_processing", "climatology", "compact_waccmx"))
    for stub in ("SOCOL", "CMAM"):
        assert has_capability(stub, "archive_inventory") and has_capability(stub, "download")
        assert not any(has_capability(stub, cap) for cap in ("zonal_processing", "climatology", "compact_waccmx"))


def test_compact_rejects_models_without_capability():
    import pytest
    from jumacs.compact import build_compact
    with pytest.raises(ValueError, match="not registered for compact"):
        build_compact("SOCOL")
    with pytest.raises(ValueError, match="not registered for compact"):
        build_compact("WACCM-X")


def test_model_period_defaults_to_reference_configuration():
    from jumacs.config import model_period
    assert model_period("GEOSCCM") == {"start_year": 1985, "end_year": 2014}
    assert model_period("WACCM-X")["start_year"] == 1985


def test_download_cli_is_plan_only_without_execute(monkeypatch):
    import jumacs.cli as cli
    calls = []
    payload = {"model": "GEOSCCM", "files": [{"filename": "x.nc"}], "selected_bytes_estimate": 5}
    monkeypatch.setattr(cli, "plan", lambda model, start, end, variable: calls.append(("plan", model)) or payload)
    monkeypatch.setattr(cli, "download", lambda p: calls.append(("download", p["model"])))
    cli.main(["download", "--model", "GEOSCCM"])
    assert calls == [("plan", "GEOSCCM")]
    cli.main(["download", "--model", "GEOSCCM", "--execute"])
    assert calls == [("plan", "GEOSCCM"), ("plan", "GEOSCCM"), ("download", "GEOSCCM")]


def test_judac_helper_rejects_non_manifest_json(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("judac_download", Path(__file__).resolve().parents[1] / "scripts/judac_download.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory_like = {"model": "GEOSCCM", "inspected_utc": "x", "files": [{"variable": "o3"}]}
    matrix_like = {"model": "GEOSCCM", "rows": []}
    manifest = {"model": "GEOSCCM", "experiment": "refD1", "selected_bytes_estimate": 1,
                "files": [{"filename": "o3.nc", "local_path": "data/raw/x.nc",
                           "source_url": "https://example.invalid/o3.nc", "download_status": "pending"}]}
    assert module.is_manifest(manifest)
    assert not module.is_manifest(inventory_like)
    assert not module.is_manifest(matrix_like)
    manifests_dir = tmp_path / "products/manifests"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "inventory.json").write_text(__import__("json").dumps(inventory_like))
    (manifests_dir / "manifest.json").write_text(__import__("json").dumps(manifest))
    module.ROOT = tmp_path
    assert set(module.discover_manifests()) == {"GEOSCCM"}


def test_waccmx_selects_only_monthly_zonal_period(monkeypatch):
    import jumacs.download as module
    inventory = {"dimensions":{"O3":["time","lev","lat"]},"files":[
        {"filename":"1984_zm.nc","start":"198412","end":"198412"},
        {"filename":"1985_zm.nc","start":"198501","end":"198501"},
        {"filename":"2015_zm.nc","start":"201501","end":"201501"}]}
    monkeypatch.setattr(module, "load_inventory", lambda model: inventory)
    selected, missing = select_files("WACCM-X",1985,2014,["O3","SF6"])
    assert [row["filename"] for row in selected] == ["1985_zm.nc"]
    assert missing == ["SF6"]
    import pytest
    with pytest.raises(ValueError, match="covers"):
        select_files("WACCM-X",1985,2016,["O3"])


def test_renamed_cli_uses_reference_period_by_default(monkeypatch):
    import jumacs.cli as cli
    calls = []
    monkeypatch.setattr(cli, "build_climatology", lambda model,start,end,variables: calls.append((model,start,end,variables)) or [])
    cli.main(["climatology","--model","GEOSCCM"])
    assert calls == [("GEOSCCM",1985,2014,None)]


def test_registry_kinds_and_statuses():
    from jumacs.config import model_names, ccmi_model_names, ready_ccmi_model_names, ready_model_names, model_metadata, species_registry
    names = model_names()
    assert {"GEOSCCM", "EMAC", "WACCM-X"} <= set(names)
    assert len(names) == len(set(names))
    assert "WACCM-X" not in ccmi_model_names()
    assert set(ready_ccmi_model_names()) >= {"GEOSCCM", "EMAC"}
    assert set(ready_model_names()) >= {"GEOSCCM", "EMAC", "WACCM-X"}
    assert len(ccmi_model_names()) >= 12
    geosccm = model_metadata("GEOSCCM")
    assert geosccm["kind"] == "ccmi" and geosccm["status"] == "ready" and geosccm["slug"] == "geosccm_refd1"
    waccmx = model_metadata("WACCM-X")
    assert waccmx["kind"] == "whole_atmosphere" and waccmx["slug"] == "waccmx"
    unresolved = [name for name in ccmi_model_names() if model_metadata(name)["status"] == "unresolved"]
    assert set(unresolved) == {"MIROC-ES2H"}
    assert all(model_metadata(name)["institution"] == "unresolved" for name in unresolved)
    canonical = {entry["canonical"] for entry in species_registry()}
    assert {"O3", "CFC-11", "SF6", "Cly", "Bry", "NOy"} <= canonical


def test_download_plan_writes_manifest_without_transfer(monkeypatch, tmp_path):
    import json
    import jumacs.download as download_module
    inventory = {"model": "GEOSCCM", "experiment": "refD1",
                 "files": [{"family": "Amon", "variable": "o3", "version": "v1", "grid": "GISS MODELS",
                            "filename": "o3_Amon_GISS-MODELS-0001-0012_198501-198512.nc",
                            "start": "198501", "end": "198512",
                            "source_url": "https://example.invalid/o3.nc", "size_bytes": 10}],
                 "catalogue_metadata": {"o3": {"units": "mole mole-1"}}}
    monkeypatch.setattr(download_module, "ROOT", tmp_path)
    monkeypatch.setattr(download_module, "load_inventory", lambda model: inventory)
    payload = download_module.plan("GEOSCCM", 1985, 1985, ["O3"])
    path = download_module.manifest_path("GEOSCCM")
    assert path == tmp_path / "products/manifests/geosccm_refd1.json"
    assert json.loads(path.read_text())["files"][0]["download_status"] == "pending"
    assert payload["missing_variables"] == []
    assert payload["files"][0]["local_path"].startswith(load_config("GEOSCCM")["paths"]["raw"])
    assert not any(tmp_path.rglob("*.nc"))
