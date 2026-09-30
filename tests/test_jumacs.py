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
    monkeypatch.setattr(archive, "load_inventory", lambda model: inventories[model])
    path = archive.variable_matrix()
    rows = {r["canonical_species"]: r for r in csv.DictReader(path.open())}
    assert rows["CFC-11"]["GEOSCCM_found"] == "True"
    assert rows["CFC-11"]["EMAC_found"] == "False"
    assert rows["CFC-11"]["WACCM-X_found"] == "True"
    assert all(rows["SF6"][m+"_found"] == "False" for m in inventories)


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
