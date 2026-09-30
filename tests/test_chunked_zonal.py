import numpy as np
import xarray as xr

from jumacs import zonal


def test_chunked_hybrid_zonal_product(tmp_path, monkeypatch):
    times = xr.date_range("2000-01-01", periods=24, freq="MS", use_cftime=True)
    values = np.broadcast_to(np.array([1., 3., 5.]), (24, 2, 2, 3)).copy()
    source = xr.Dataset(
        {
            "ta": (("time", "lev", "lat", "lon"), values, {"units": "K"}),
            "ps": (("time", "lat", "lon"), np.full((24, 2, 3), 100000.)),
            "a": ("lev", [.1, .2]),
            "b": ("lev", [.5, .3]),
            "p0": ("constant", [100000.]),
        },
        coords={"time": times, "lev": ("lev", [1., 2.], {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"}),
                "lat": [-30., 30.], "lon": [0., 120., 240.]},
    )
    path = tmp_path / "ta_Amon_GEOSCCM_refD1_r1i1p1f1_gr_200001-200112.nc"
    source.to_netcdf(path, engine="netcdf4")
    monkeypatch.setattr(zonal, "ROOT", tmp_path)
    monkeypatch.setattr(zonal, "source_files", lambda model, variable: [path])
    output = zonal.build_zonal("GEOSCCM", "ta")
    with xr.open_dataset(output) as result:
        assert result.sizes["time"] == 24
        assert result.attrs["source_longitude_global"] == "True"
        np.testing.assert_allclose(result.ta.values, 3.)
        assert "constant" not in result.air_pressure.dims
        np.testing.assert_allclose(result.air_pressure.isel(time=0, lat=0).values, [60000., 50000.])
