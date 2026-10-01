import numpy as np
import xarray as xr

from jumacs import diagnostics


def test_hybrid_quicklook_uses_pressure_field(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    destination = tmp_path / "products/climatology/GEOSCCM/jumacs_geosccm_refd1_o3_climatology_2000-2018.nc"
    destination.parent.mkdir(parents=True)
    dataset = xr.Dataset(
        {
            "mean": (("month", "lev", "lat"), np.array([[[1., 2.], [3., 4.]]])),
            "air_pressure": (("month", "lev", "lat"), np.array([[[50000., 45000.], [10000., 9000.]]])),
        },
        coords={"month": [1], "lev": ("lev", [.1, .2], {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"}),
                "lat": [-10., 10.]},
        attrs={"source_units": "mol mol-1"},
    )
    dataset.to_netcdf(destination, engine="netcdf4")
    paths = diagnostics.quicklooks("GEOSCCM", 2000, 2018)
    assert len(paths) == 1 and paths[0].is_file()


def test_common_grid_quicklook_uses_the_shared_pressure_coordinate(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    destination = tmp_path / "products/climatology/CMAM/jumacs_cmam_refd1_o3_climatology_1985-2014.nc"
    destination.parent.mkdir(parents=True)
    levels = [100000., 10000., 1000.]
    xr.Dataset({"mean": (("month", "pressure", "lat"), np.arange(1 * 3 * 2, dtype=float).reshape(1, 3, 2))},
               coords={"month": [1],
                       "pressure": ("pressure", levels, {"standard_name": "air_pressure", "units": "Pa", "positive": "down"}),
                       "lat": [-30., 30.]},
               attrs={"source_units": "mol mol-1"}).to_netcdf(destination, engine="netcdf4")
    paths = diagnostics.quicklooks("CMAM", 1985, 2014)
    assert len(paths) == 1 and paths[0].is_file()
