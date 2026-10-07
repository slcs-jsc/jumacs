"""The three additional refD1 models use their published AmonZ pressure grids."""

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


def test_niwa_ep_fluxes_join_the_product_on_their_own_latitude_grid():
    config = load_config("NIWA-UKCA2")
    configured = configured_variables("NIWA-UKCA2", config)
    assert {"epfy", "epfz"} <= set(configured)
    assert "c2h6" not in configured
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
