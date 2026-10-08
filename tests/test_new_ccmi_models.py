"""The additional refD1 models use their published AmonZ pressure grids."""

from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
import xarray as xr

from jumacs import cf
from jumacs.application import build_application_product
from jumacs.build import build_model, configured_variables
from jumacs.config import load_config
from jumacs.download import select_files
from jumacs.reader import open_source, source_files
from jumacs.zonal import zonal_mean

MODELS = ("IPSL-CM6A-ATM-LR-REPROBUS", "CCSR-NIES-MIROC32", "NIWA-UKCA2")


def _raw_file(root, model, family, name, dataset):
    config = load_config(model)
    archive = config["model"]["archive_model"]
    path = root / config["paths"]["raw"] / family / name / (
        f"{name}_{family}_{archive}_refD1_r1i1p1f1_gnz_198501-198612.nc")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    return path


@pytest.mark.parametrize("model", MODELS)
def test_new_model_native_and_application_products(tmp_path, monkeypatch, model):
    import jumacs.build as build_module
    import jumacs.climatology as climatology_module
    import jumacs.config as config_module
    import jumacs.zonal as zonal_module

    for module in (build_module, climatology_module, config_module, zonal_module):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    copyfile(Path(__file__).resolve().parents[1] / "config" / "climatology.yaml",
             tmp_path / "config" / "climatology.yaml")
    config = load_config(model)
    level = config["coordinates"]["level"]
    latitude = [45., 0., -45.] if model == "CCSR-NIES-MIROC32" else [-45., 0., 45.]
    time = xr.date_range("1985-01", periods=24, freq="MS", use_cftime=True)
    coordinates = {"time": time, level: (level, [100000., 10000., 3.],
                                              {"units": "Pa", "standard_name": "air_pressure"}),
                   "lat": latitude}
    dimensions = ("time", level, "lat")
    if model == "CCSR-NIES-MIROC32":
        coordinates["lon"] = [0.]
        dimensions += ("lon",)
    temperature = xr.Dataset({"ta": (dimensions, np.full((24, 3, 3) + ((1,) if model == "CCSR-NIES-MIROC32" else ()), 230.),
                                    {"units": "K"})}, coords=coordinates)
    _raw_file(tmp_path, model, "AmonZ", "ta", temperature)
    longitude = [0., 90., 180., 270.]
    surface_dims = ("time", "lat", "lon")
    surface_coords = {"time": time, "lat": latitude, "lon": longitude}
    if model == "CCSR-NIES-MIROC32":
        surface_dims = ("time", "lev", "lat", "lon")
        surface_coords["lev"] = ("lev", [100.], {"units": "Pa"})
    pressure = xr.Dataset({"ps": (surface_dims, np.full((24,) + ((1,) if model == "CCSR-NIES-MIROC32" else ()) + (3, 4), 100000.),
                                  {"units": "Pa"})}, coords=surface_coords)
    _raw_file(tmp_path, model, "Amon", "ps", pressure)

    assert source_files(model, "temperature")[0].parent.parent.name == "AmonZ"
    assert source_files(model, "surface_pressure")[0].parent.parent.name == "Amon"
    report = build_model(model, 1985, 1986, ["temperature", "surface_pressure"])
    assert report["ok"] is True
    with xr.open_dataset(report["product"], decode_cf=False) as native:
        cf.assert_product(native, names=("ta", "ps"), kind="individual")
        assert native["ta_mean"].ndim == 3
        assert native["ps_mean"].dims == ("time", "lat")
    path = build_application_product(model, 1985, 1986, extend=False, root=tmp_path)
    with xr.open_dataset(path, decode_cf=False) as application:
        cf.assert_product(application, names=("ta", "ps"), kind="application")
        assert application.sizes["pressure"] == 124
        assert application.sizes["lat"] == 36
        assert application["ta_mean"].ndim == 3
        assert application["ps_mean"].ndim == 2


@pytest.mark.parametrize("model", MODELS)
def test_archive_family_selection_agrees_between_download_and_reader(monkeypatch, model):
    from jumacs import download

    inventory = {"files": [
        {"variable": "ta", "family": family, "start": "198501", "end": "198612"}
        for family in ("Amon", "AmonZ")
    ] + [{"variable": "ps", "family": "Amon", "start": "198501", "end": "198612"}]}
    monkeypatch.setattr(download, "load_inventory", lambda selected: inventory)
    selected, missing = select_files(model, 1985, 1986, ["temperature", "surface_pressure"])
    assert not missing
    assert {(row["variable"], row["family"]) for row in selected} == {("ta", "AmonZ"), ("ps", "Amon")}


def test_niwa_default_build_excludes_ep_fluxes_and_c2h6():
    config = load_config("NIWA-UKCA2")
    configured = configured_variables("NIWA-UKCA2", config)
    assert "epfy" not in configured and "epfz" not in configured
    assert "c2h6" not in configured
    assert set(config["exclude_from_climatology"]) == {"c2h6", "epfy", "epfz"}
    assert "abs_magnitude_limits" not in config
    assert config["variables"]["EP_flux_meridional"] == "epfy"


def test_ccsr_near_fill_values_are_missing_before_longitude_mean(tmp_path):
    model = "CCSR-NIES-MIROC32"
    marker = np.float32(1e20)
    near_marker = np.nextafter(marker, np.float32(0))
    values = np.array([[[[near_marker, 220.], [near_marker, near_marker]],
                        [[230., 250.], [240., 260.]]]], dtype="float32")
    ds = xr.Dataset({"ta": (("time", "lev", "lat", "lon"), values, {"units": "K"})},
                    coords={"time": xr.date_range("1985-01", periods=1, use_cftime=True),
                            "lev": ("lev", [100000., 10000.], {"units": "Pa"}),
                            "lat": [-30., 30.], "lon": [0., 180.]})
    path = _raw_file(tmp_path, model, "Amon", "ta", ds)
    ds.to_netcdf(path, encoding={"ta": {"_FillValue": marker}})
    with open_source(path, model, "temperature") as source:
        assert np.isfinite(source.ta.isel(time=0, lev=0, lat=0, lon=0).item())
        out = zonal_mean(source, "ta", near_fill_relative_tolerance=load_config(model)["near_fill_relative_tolerance"])
        np.testing.assert_allclose(out.isel(time=0, lev=0, lat=0).item(), 220.)
        assert np.isnan(out.isel(time=0, lev=0, lat=1).item())
        np.testing.assert_allclose(out.isel(time=0, lev=1).values, [240., 250.])


PUBLISHED_LEVEL_MODELS = ("ACCESS-CM2-Chem", "UKESM1-StratTrop")
FILE_STAMPS = {"ACCESS-CM2-Chem": "r1i1p1f1_gn_198501-198612",
               "UKESM1-StratTrop": "r1i1p1f2_grz_19850101-19870101"}


def _stamped_raw_file(root, model, family, name, dataset):
    config = load_config(model)
    archive = config["model"]["archive_model"]
    path = root / config["paths"]["raw"] / family / name / (
        f"{name}_{family}_{archive}_refD1_{FILE_STAMPS[model]}.nc")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    return path


def test_access_and_ukesm_are_buildable_from_published_pressure_levels():
    for model in PUBLISHED_LEVEL_MODELS:
        config = load_config(model)
        assert config["capabilities"]["zonal_processing"] is True
        assert config["capabilities"]["climatology"] is True
        assert config["coordinates"] == {"latitude": "lat", "longitude": "lon",
                                         "level": "plev", "pressure": "plev"}
        assert config["preferred_family_when_available"] == "AmonZ"
    access = load_config("ACCESS-CM2-Chem")
    assert set(access["exclude_from_climatology"]) == {"c2h6"}
    assert "c2h6" in access["variables"].values()
    assert "c2h6" not in configured_variables("ACCESS-CM2-Chem", access)
    assert "exclude_from_climatology" not in load_config("UKESM1-StratTrop")


@pytest.mark.parametrize("model", PUBLISHED_LEVEL_MODELS)
def test_access_and_ukesm_native_and_application_products(tmp_path, monkeypatch, model):
    import jumacs.build as build_module
    import jumacs.climatology as climatology_module
    import jumacs.config as config_module
    import jumacs.zonal as zonal_module

    for module in (build_module, climatology_module, config_module, zonal_module):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    copyfile(Path(__file__).resolve().parents[1] / "config" / "climatology.yaml",
             tmp_path / "config" / "climatology.yaml")
    time = xr.date_range("1985-01", periods=24, freq="MS", use_cftime=True)
    temperature = xr.Dataset(
        {"ta": (("time", "plev", "lat"), np.full((24, 3, 3), 230.), {"units": "K"})},
        coords={"time": time, "lat": [-45., 0., 45.],
                "plev": ("plev", [100000., 10000., 3.],
                         {"units": "Pa", "standard_name": "air_pressure", "positive": "down"})})
    _stamped_raw_file(tmp_path, model, "AmonZ", "ta", temperature)
    pressure = xr.Dataset(
        {"ps": (("time", "lat", "lon"), np.full((24, 3, 4), 100000.), {"units": "Pa"})},
        coords={"time": time, "lat": [-45., 0., 45.], "lon": [0., 90., 180., 270.]})
    _stamped_raw_file(tmp_path, model, "Amon", "ps", pressure)

    assert source_files(model, "temperature")[0].parent.parent.name == "AmonZ"
    assert source_files(model, "surface_pressure")[0].parent.parent.name == "Amon"
    report = build_model(model, 1985, 1986, ["temperature", "surface_pressure"])
    assert report["ok"] is True
    with xr.open_dataset(report["product"], decode_cf=False) as native:
        cf.assert_product(native, names=("ta", "ps"), kind="individual")
        assert native["ta_mean"].ndim == 3
        assert native["ps_mean"].dims == ("time", "lat")
    path = build_application_product(model, 1985, 1986, extend=False, root=tmp_path)
    with xr.open_dataset(path, decode_cf=False) as application:
        cf.assert_product(application, names=("ta", "ps"), kind="application")
        assert application.sizes["pressure"] == 124
        assert application.sizes["lat"] == 36


def test_cesm2_waccm_is_buildable_from_published_pressure_levels():
    config = load_config("CESM2-WACCM")
    assert config["capabilities"]["zonal_processing"] is True
    assert config["capabilities"]["climatology"] is True
    assert config["coordinates"] == {"latitude": "lat", "longitude": "lon",
                                     "level": "plev", "pressure": "plev"}
    assert config["preferred_family_when_available"] == "AmonZ"
    assert config["near_fill_relative_tolerance"] == pytest.approx(1e-06)
    assert set(config["exclude_from_climatology"]) == {"c2h2", "c2h6", "co2", "epfy"}
    assert {"c2h2", "c2h6", "co2", "epfy"} <= set(config["variables"].values())
    configured = configured_variables("CESM2-WACCM", config)
    assert {"c2h2", "c2h6", "co2", "epfy"}.isdisjoint(configured)
    assert "ta" in configured and "epfz" in configured


def test_cesm2_waccm_archive_family_selection_prefers_amonz(monkeypatch):
    from jumacs import download

    inventory = {"files": [
        {"variable": "ta", "family": family, "start": "198501", "end": "198612"}
        for family in ("Amon", "AmonZ")
    ] + [{"variable": "ps", "family": "Amon", "start": "198501", "end": "198612"}]}
    monkeypatch.setattr(download, "load_inventory", lambda selected: inventory)
    selected, missing = select_files("CESM2-WACCM", 1985, 1986, ["temperature", "surface_pressure"])
    assert not missing
    assert {(row["variable"], row["family"]) for row in selected} == {("ta", "AmonZ"), ("ps", "Amon")}


def test_cesm2_waccm_native_and_application_products(tmp_path, monkeypatch):
    import jumacs.build as build_module
    import jumacs.climatology as climatology_module
    import jumacs.config as config_module
    import jumacs.zonal as zonal_module

    model = "CESM2-WACCM"
    for module in (build_module, climatology_module, config_module, zonal_module):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    copyfile(Path(__file__).resolve().parents[1] / "config" / "climatology.yaml",
             tmp_path / "config" / "climatology.yaml")
    time = xr.date_range("1985-01", periods=24, freq="MS", use_cftime=True)
    temperature = xr.Dataset(
        {"ta": (("time", "plev", "lat"), np.full((24, 3, 3), 230.), {"units": "K"})},
        coords={"time": time, "lat": [-45., 0., 45.],
                "plev": ("plev", [100000., 10000., 3.],
                         {"units": "Pa", "standard_name": "air_pressure", "positive": "down"})})
    _raw_file(tmp_path, model, "AmonZ", "ta", temperature)
    pressure = xr.Dataset(
        {"ps": (("time", "lat", "lon"), np.full((24, 3, 4), 100000.), {"units": "Pa"})},
        coords={"time": time, "lat": [-45., 0., 45.], "lon": [0., 90., 180., 270.]})
    _raw_file(tmp_path, model, "Amon", "ps", pressure)

    assert source_files(model, "temperature")[0].parent.parent.name == "AmonZ"
    assert source_files(model, "surface_pressure")[0].parent.parent.name == "Amon"
    report = build_model(model, 1985, 1986, ["temperature", "surface_pressure"])
    assert report["ok"] is True
    with xr.open_dataset(report["product"], decode_cf=False) as native:
        cf.assert_product(native, names=("ta", "ps"), kind="individual")
        assert native["ta_mean"].ndim == 3
        assert native["ps_mean"].dims == ("time", "lat")
    path = build_application_product(model, 1985, 1986, extend=False, root=tmp_path)
    with xr.open_dataset(path, decode_cf=False) as application:
        cf.assert_product(application, names=("ta", "ps"), kind="application")
        assert application.sizes["pressure"] == 124
        assert application.sizes["lat"] == 36


def test_cesm2_waccm_epfz_near_fill_escapees_are_missing_before_statistics(tmp_path):
    model = "CESM2-WACCM"
    marker = np.float32(1e20)
    escapee = np.nextafter(marker, np.float32(np.inf))
    values = np.array([[[escapee, -marker, 3.0e6, escapee]]], dtype="float32")
    ds = xr.Dataset({"epfz": (("time", "plev", "lat"), values, {"units": "m3 s-2"})},
                    coords={"time": xr.date_range("1985-01", periods=1, use_cftime=True),
                            "plev": ("plev", [100000.], {"units": "Pa"}),
                            "lat": [-45., -15., 15., 45.]})
    path = _raw_file(tmp_path, model, "AmonZ", "epfz", ds)
    ds.to_netcdf(path, encoding={"epfz": {"_FillValue": marker}})
    with open_source(path, model, "EP_flux_vertical") as source:
        out = zonal_mean(source, "epfz",
                         near_fill_relative_tolerance=load_config(model)["near_fill_relative_tolerance"])
        np.testing.assert_allclose(out.isel(time=0, plev=0).values, [np.nan, np.nan, 3.0e6, np.nan], equal_nan=True)

