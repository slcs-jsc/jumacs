import numpy as np
import xarray as xr

from jumacs import config as config_module
from jumacs import reader, zonal

TIMES = xr.date_range("2000-01-01", periods=2, freq="MS", use_cftime=True)
LAT = [-80., 0.]
LON = [0., 120., 240.]


def _config(variables=None):
    return {"model": {"name": "TestModel", "display_name": "TestModel", "kind": "ccmi", "status": "ready",
                      "experiment": "refD1", "archive_model": "TestModel", "dataset_uuid": "x", "archive_base": "x"},
            "coordinates": {"latitude": "lat", "longitude": "lon", "level": "plev", "pressure": "plev"},
            "paths": {"raw": "raw", "zonal": "processed"},
            "variables": variables if variables is not None else {"CFC-11": "cfcl3", "surface_pressure": "ps"}}


def _build(tmp_path, monkeypatch, field, ps, plev=(100000.,), ps_units="Pa", variables=None):
    field_path = tmp_path / "cfcl3_AmonZ_TestModel_refD1_r1i1p1f1_gn_200001-200002.nc"
    xr.Dataset({"cfcl3": (("time", "plev", "lat"), field, {"units": "mol mol-1"})},
               coords={"time": TIMES, "plev": ("plev", list(plev), {"units": "Pa"}), "lat": LAT}
               ).to_netcdf(field_path, engine="netcdf4")
    files = {"cfcl3": [field_path]}
    if ps is not None:
        ps_path = tmp_path / "ps_Amon_TestModel_refD1_r1i1p1f1_gn_200001-200002.nc"
        xr.Dataset({"ps": (("time", "lat", "lon"), ps, {"units": ps_units})},
                   coords={"time": TIMES, "lat": LAT, "lon": LON}
                   ).to_netcdf(ps_path, engine="netcdf4")
        files["ps"] = [ps_path]
    cfg = _config(variables)
    monkeypatch.setattr(zonal, "ROOT", tmp_path)
    monkeypatch.setattr(zonal, "load_config", lambda model: cfg)
    monkeypatch.setattr(reader, "load_config", lambda model: cfg)
    monkeypatch.setattr(zonal, "source_files", lambda model, variable: files[variable])
    with xr.open_dataset(zonal.build_zonal("TestModel", "CFC-11")) as result:
        return result.cfcl3.load(), result.attrs


def test_fully_below_ground_circle_becomes_nan(tmp_path, monkeypatch):
    ps = np.stack([np.array([[70000., 75000., 72000.], [101000., 101500., 102000.]])] * 2)
    field = np.full((2, 1, 2), 5.0)
    values, attrs = _build(tmp_path, monkeypatch, field, ps)
    assert np.isnan(values.isel(time=0, plev=0, lat=0).item())
    assert values.isel(time=0, plev=0, lat=1).item() == 5.0
    assert attrs["below_ground_mask"] == "pressure_level > maximum surface pressure around latitude circle"
    assert attrs["below_ground_surface_pressure_variable"] == "ps"
    assert "already a zonal mean" in attrs["below_ground_limitation"]


def test_partially_below_ground_circle_preserved(tmp_path, monkeypatch):
    ps = np.stack([np.array([[70000., 75000., 101000.], [101000., 101500., 102000.]])] * 2)
    values, attrs = _build(tmp_path, monkeypatch, np.full((2, 1, 2), 5.0), ps)
    assert np.isfinite(values.isel(lat=0).values).all()
    assert "below_ground_mask" not in attrs


def test_hpa_surface_pressure_units(tmp_path, monkeypatch):
    ps = np.stack([np.array([[700., 750., 720.], [1010., 1015., 1020.]])] * 2)
    values, _ = _build(tmp_path, monkeypatch, np.full((2, 1, 2), 5.0), ps, ps_units="hPa")
    assert np.isnan(values.isel(time=0, plev=0, lat=0).item())
    assert values.isel(time=0, plev=0, lat=1).item() == 5.0


def test_pressure_levels_masked_independently(tmp_path, monkeypatch):
    ps = np.stack([np.array([[88000., 90000., 89000.], [103000., 103000., 103000.]])] * 2)
    values, _ = _build(tmp_path, monkeypatch, np.full((2, 2, 2), 5.0), ps, plev=(100000., 85000.))
    assert np.isnan(values.isel(time=0, plev=0, lat=0).item())
    assert values.isel(time=0, plev=1, lat=0).item() == 5.0
    assert np.isfinite(values.isel(lat=1).values).all()


def test_mask_depends_on_month(tmp_path, monkeypatch):
    ps = np.array([[[70000., 70000., 70000.], [101000., 101000., 101000.]],
                   [[105000., 105000., 105000.], [101000., 101000., 101000.]]])
    values, _ = _build(tmp_path, monkeypatch, np.full((2, 1, 2), 5.0), ps)
    assert np.isnan(values.isel(time=0, plev=0, lat=0).item())
    assert values.isel(time=1, plev=0, lat=0).item() == 5.0


def test_no_surface_pressure_mapping_preserves_values(tmp_path, monkeypatch):
    values, attrs = _build(tmp_path, monkeypatch, np.full((2, 1, 2), 5.0), None,
                           variables={"CFC-11": "cfcl3"})
    assert (values.values == 5.0).all()
    assert "below_ground_mask" not in attrs


def test_hybrid_field_preserves_current_behavior(tmp_path, monkeypatch):
    times = xr.date_range("2000-01-01", periods=24, freq="MS", use_cftime=True)
    source = xr.Dataset(
        {
            "ta": (("time", "lev", "lat", "lon"), np.full((24, 2, 2, 3), 3.), {"units": "K"}),
            "ps": (("time", "lat", "lon"), np.full((24, 2, 3), 100000.)),
            "a": ("lev", [.1, .2]),
            "b": ("lev", [.5, .3]),
            "p0": ("constant", [100000.]),
        },
        coords={"time": times, "lev": ("lev", [1., 2.],
                                       {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"}),
                "lat": LAT, "lon": LON},
    )
    path = tmp_path / "ta_Amon_GEOSCCM_refD1_r1i1p1f1_gr_200001-200112.nc"
    source.to_netcdf(path, engine="netcdf4")
    monkeypatch.setattr(zonal, "ROOT", tmp_path)
    monkeypatch.setattr(zonal, "source_files", lambda model, variable: [path])
    with xr.open_dataset(zonal.build_zonal("GEOSCCM", "ta")) as result:
        assert (result.ta.values == 3.0).all()
        assert "below_ground_mask" not in result.attrs


def test_access_surface_pressure_discovery_across_families(tmp_path, monkeypatch):
    raw = tmp_path / "data/raw/ACCESS-CM2-Chem/refD1"
    cfcl3_dir = raw / "AmonZ/cfcl3"
    ps_dir = raw / "Amon/ps"
    cfcl3_dir.mkdir(parents=True)
    ps_dir.mkdir(parents=True)
    (cfcl3_dir / "cfcl3_AmonZ_ACCESS-CM2-Chem_refD1_r1i1p1f1_gn_200001-200002.nc").touch()
    (ps_dir / "ps_Amon_ACCESS-CM2-Chem_refD1_r1i1p1f1_gn_200001-200002.nc").touch()
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    assert config_module.load_config("ACCESS-CM2-Chem")["variables"]["CFC-11"] == "cfcl3"
    assert config_module.load_config("ACCESS-CM2-Chem")["variables"]["surface_pressure"] == "ps"
    field_files = reader.source_files("ACCESS-CM2-Chem", "CFC-11")
    surface_files = reader.source_files("ACCESS-CM2-Chem", "surface_pressure")
    assert [path.parent.parent.name for path in field_files] == ["AmonZ"]
    assert [path.parent.parent.name for path in surface_files] == ["Amon"]


def test_access_surface_pressure_file_opens(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    path = tmp_path / "ps_Amon_ACCESS-CM2-Chem_refD1_r1i1p1f1_gn_200001-200002.nc"
    xr.Dataset({"ps": (("time", "lat", "lon"), np.full((2, 2, 3), 100000.), {"units": "Pa"})},
               coords={"time": TIMES, "lat": LAT, "lon": LON}
               ).to_netcdf(path, engine="netcdf4")
    with reader.open_source(path, "ACCESS-CM2-Chem", "surface_pressure") as ds:
        assert set(ds.ps.dims) == {"time", "lat", "lon"}
        assert np.isfinite(ds.ps.values).all()
