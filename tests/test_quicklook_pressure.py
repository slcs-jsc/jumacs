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
