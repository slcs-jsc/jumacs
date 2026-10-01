import numpy as np
import xarray as xr

from jumacs import diagnostics

PRESSURE = [100000., 10000., 1000.]


def write_product(tmp_path, model, filename, dataset):
    destination = tmp_path / "products/climatology" / model / filename
    destination.parent.mkdir(parents=True)
    dataset.to_netcdf(destination, engine="netcdf4")


def zonal_product(fields):
    coords = {"time": ("time", np.arange(12.0), {"units": "days since 2000-01-01", "calendar": "gregorian"}),
              "pressure": ("pressure", PRESSURE, {"standard_name": "air_pressure", "units": "Pa", "positive": "down", "axis": "Z"}),
              "lat": ("lat", [-30., 30.], {"units": "degrees_north", "axis": "Y"})}
    return xr.Dataset(fields, coords=coords, attrs={"Conventions": "CF-1.13"})


def test_quicklook_uses_the_shared_pressure_coordinate(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    mean = np.arange(12 * 3 * 2, dtype=float).reshape(12, 3, 2)
    write_product(tmp_path, "CMAM", "jumacs_cmam_refd1_climatology_1985-2014.nc",
                  zonal_product({"o3_mean": (("time", "pressure", "lat"), mean, {"units": "mol mol-1"})}))
    paths = diagnostics.quicklooks("CMAM", 1985, 2014)
    assert [path.name for path in paths] == [f"o3_1985-2014_{month:02d}.png" for month in (1, 4, 7, 10)]
    assert all(path.is_file() for path in paths)


def test_quicklook_skips_fields_without_a_vertical_axis(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    fields = {"ta_mean": (("time", "lat"), np.zeros((12, 2)), {"units": "K"}),
              "o3_mean": (("time", "pressure", "lat"), np.zeros((12, 3, 2)), {"units": "mol mol-1"})}
    write_product(tmp_path, "CMAM", "jumacs_cmam_refd1_climatology_1985-2014.nc", zonal_product(fields))
    plotted = {path.name.split("_")[0] for path in diagnostics.quicklooks("CMAM", 1985, 2014)}
    assert plotted == {"o3"}


def test_quicklook_is_silent_without_a_product(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    assert diagnostics.quicklooks("CMAM", 1985, 2014) == []
