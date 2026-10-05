import numpy as np
import xarray as xr

import jumacs.config as config_module
import jumacs.zonal as zonal_module
from jumacs.config import load_config
from jumacs.coordinates import hybrid_pressure
from jumacs.zonal import is_hybrid_level, zonal_smoke

MAPPED = ("CMAM", "CNRM-MOCAGE", "SOCOL")
DEFERRED = ("CESM2-WACCM",)


def _times(n=3):
    return xr.cftime_range("1990-01", periods=n, freq="MS", calendar="standard")


def _hybrid_ds(a_name="ap", a_units="Pa", a_values=(10000.0, 20000.0), b_values=(0.9, 0.5), p0=None):
    times = _times()
    lon = np.arange(0.0, 360.0, 90.0)
    lat = np.array([-45.0, 0.0, 45.0])
    nlev = len(a_values)
    if a_name == "ap":
        terms = "ap: ap b: b ps: ps"
    else:
        terms = "a: a b: b ps: ps" + (" p0: p0" if p0 is not None else "")
    values = np.linspace(210.0, 250.0, len(times) * nlev * len(lat) * len(lon)).reshape(
        len(times), nlev, len(lat), len(lon)).astype("float32")
    ds = xr.Dataset({"ta": (("time", "lev", "lat", "lon"), values,
                            {"units": "K", "standard_name": "air_temperature"})},
                    coords={"time": times, "lev": ("lev", np.arange(1.0, nlev + 1)),
                            "lat": ("lat", lat), "lon": ("lon", lon)})
    ds.lev.attrs = {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate", "units": "1",
                    "positive": "down", "formula_terms": terms}
    ds[a_name] = ("lev", list(a_values), {"units": a_units})
    ds["b"] = ("lev", list(b_values), {"units": "1"})
    ds["ps"] = (("time", "lat", "lon"), np.full((len(times), len(lat), len(lon)), 100000.0, dtype="float32"),
                {"units": "Pa", "standard_name": "surface_air_pressure"})
    if p0 is not None:
        ds["p0"] = ((), float(p0), {"units": "Pa", "standard_name": "reference_pressure"})
    return ds


def _dimless_ds():
    return _hybrid_ds(a_name="a", a_units="1", a_values=(0.8, 0.3), b_values=(0.2, 0.6), p0=100000.0)


def _plev_4d_ds():
    times = _times()
    lon = np.arange(0.0, 360.0, 90.0)
    lat = np.array([-45.0, 0.0, 45.0])
    plev = [100000.0, 10000.0, 1000.0]
    ds = xr.Dataset({"ta": (("time", "plev", "lat", "lon"),
                            np.linspace(200.0, 260.0, len(times) * 3 * len(lat) * len(lon)).reshape(
                                len(times), 3, len(lat), len(lon)).astype("float32"),
                            {"units": "K", "standard_name": "air_temperature"})},
                    coords={"time": times, "plev": ("plev", plev), "lat": ("lat", lat), "lon": ("lon", lon)})
    ds.plev.attrs = {"standard_name": "air_pressure", "units": "Pa", "positive": "down"}
    return ds


def _profile(pressure):
    return [round(float(v), 3) for v in pressure.isel(time=0).mean(("lat", "lon")).values]


def _store_raw(root, model, relative, ds):
    dest = root / load_config(model)["paths"]["raw"] / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dest)
    return dest


def test_new_mappings_present():
    for model in MAPPED:
        coords = load_config(model)["coordinates"]
        assert coords["latitude"] == "lat" and coords["longitude"] == "lon"
        assert coords["level"] == "lev" and coords["surface_pressure"] == "ps"
        assert coords["pressure"] == "plev" and coords["hybrid_b"] == "b"
    cmam = load_config("CMAM")["coordinates"]
    assert cmam["hybrid_a"] == "ap" and "reference_pressure" not in cmam
    for model in ("CNRM-MOCAGE", "SOCOL"):
        coords = load_config(model)["coordinates"]
        assert coords["hybrid_a"] == "a" and coords["reference_pressure"] == "p0"


def test_deferred_models_left_unchanged():
    for model in DEFERRED:
        assert load_config(model).get("coordinates", {}) == {}


def test_mapped_names_present_and_gate_triggers():
    cmam_config = load_config("CMAM")
    ap_ds = _hybrid_ds(a_name="ap")
    assert is_hybrid_level(ap_ds, cmam_config) is True
    for key in ("hybrid_a", "hybrid_b", "surface_pressure"):
        assert cmam_config["coordinates"][key] in ap_ds
    for model in ("CNRM-MOCAGE", "SOCOL"):
        config = load_config(model)
        dimless = _dimless_ds()
        assert is_hybrid_level(dimless, config) is True
        for key in ("hybrid_a", "hybrid_b", "surface_pressure", "reference_pressure"):
            assert config["coordinates"][key] in dimless


def test_hybrid_reconstruction_ap_style():
    pressure = hybrid_pressure(_hybrid_ds(a_name="ap", a_values=(10000.0, 20000.0), b_values=(0.9, 0.5)),
                               load_config("CMAM"))
    assert pressure.attrs["units"] == "Pa"
    assert _profile(pressure) == [100000.0, 70000.0]


def test_hybrid_reconstruction_dimensionless_style():
    for model in ("CNRM-MOCAGE", "SOCOL"):
        pressure = hybrid_pressure(_dimless_ds(), load_config(model))
        assert pressure.attrs["units"] == "Pa"
        assert _profile(pressure) == [100000.0, 90000.0]


def test_plev_coordinate_is_not_hybrid():
    assert is_hybrid_level(_plev_4d_ds(), load_config("CMAM")) is False


def test_emac_and_geosccm_regression():
    emac = load_config("EMAC")
    assert is_hybrid_level(_hybrid_ds(a_name="ap"), emac) is True
    assert _profile(hybrid_pressure(_hybrid_ds(a_name="ap", a_values=(10000.0, 20000.0), b_values=(0.9, 0.5)), emac)) == [100000.0, 70000.0]
    geosccm = load_config("GEOSCCM")
    assert geosccm["coordinates"]["hybrid_a"] == "a" and geosccm["coordinates"]["reference_pressure"] == "p0"
    assert _profile(hybrid_pressure(_dimless_ds(), geosccm)) == [100000.0, 90000.0]


def test_waccmx_forces_hybrid_route():
    waccmx = load_config("WACCM-X")
    minimal = xr.Dataset(coords={"lev": ("lev", [1.0, 2.0])})
    assert is_hybrid_level(minimal, waccmx) is True


def test_zonal_smoke_hybrid_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    monkeypatch.setattr(zonal_module, "ROOT", tmp_path)
    _store_raw(tmp_path, "CMAM", "Amon/ta/ta_Amon_CMAM_refD1_r1i1p1f1_gn_199001-199003.nc",
               _hybrid_ds(a_name="ap", a_values=(10000.0, 20000.0), b_values=(0.9, 0.5)))
    report, dest = zonal_smoke("CMAM", "temperature")
    assert report["ok"] is True
    assert report["vertical_route"] == "hybrid_reconstruction"
    assert report["longitude_removed"] is True
    assert report["zonal_finite_fraction"] == 1.0
    assert report["pressure_min_pa"] == 70000.0 and report["pressure_max_pa"] == 100000.0
    assert report["pressure_decreasing_fraction"] == 1.0
    assert dest is not None and dest.exists()


def test_zonal_smoke_pressure_level_route(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    monkeypatch.setattr(zonal_module, "ROOT", tmp_path)
    _store_raw(tmp_path, "CMAM", "Amon/ta/ta_Amon_CMAM_refD1_r1i1p1f1_gn_199001-199003.nc", _plev_4d_ds())
    report, dest = zonal_smoke("CMAM", "temperature")
    assert report["ok"] is True
    assert report["vertical_route"] == "pressure_level:plev"
    assert report["pressure_min_pa"] == 1000.0 and report["pressure_max_pa"] == 100000.0
    assert report["pressure_decreasing_fraction"] == 1.0
    assert dest is not None and dest.exists()


def test_zonal_smoke_deferred_reports_incomplete_mapping():
    report, dest = zonal_smoke("CESM2-WACCM", "temperature")
    assert report["ok"] is False
    assert dest is None
    assert "mapping incomplete" in report["reason"]
