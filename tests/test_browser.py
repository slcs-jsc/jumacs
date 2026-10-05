"""Static application-product browser wiring and regeneration."""

import json
import re
from pathlib import Path

import numpy as np
import xarray as xr

from jumacs.browser import (
    PLOT_LATITUDE_EDGES,
    PLOT_PRESSURES_HPA,
    ZONAL_MONTHS,
    _band_samples,
    _height_to_pressure,
    _plot_paths,
    _plot_zonal,
    _pressure_to_height,
    build_site,
    discover_application_products,
)
from jumacs.cli import main


def _product(root, model="SOCOL", name="ta"):
    folder = root / "products" / "application" / model
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"jumacs_{model.lower()}_application_climatology_1985-2014.nc"
    values = np.ones((12, 4, 3))
    xr.Dataset({f"{name}_mean": (("time", "pressure", "lat"), values, {"units": "K"}),
                "surface_mean": (("time", "lat"), np.ones((12, 3)))},
               coords={"time": range(12), "pressure": [10000., 1000., 100., 10.],
                       "lat": [-62.5, 2.5, 57.5]}).to_netcdf(path)
    return path


def test_discovery_months_and_paths(tmp_path):
    product = _product(tmp_path)
    assert discover_application_products(tmp_path, 1985, 2014) == [("SOCOL", product)]
    assert ZONAL_MONTHS == (1, 4, 7, 10)
    assert _plot_paths("SOCOL", "ta") == (Path("plots/SOCOL/ta_zonal.png"),
                                          Path("plots/SOCOL/ta_annual.png"))


def test_site_links_and_regeneration(tmp_path, monkeypatch):
    from jumacs import browser

    def placeholder(*args):
        args[-1].write_bytes(b"PNG")

    monkeypatch.setattr(browser, "_plot_zonal", placeholder)
    monkeypatch.setattr(browser, "_plot_annual", placeholder)
    product = _product(tmp_path)
    first = build_site(1985, 2014, root=tmp_path)
    site = first["site"]
    assert first["models"] == ["SOCOL"]
    assert first["variables"] == {"SOCOL": 1}
    assert first["png_files"] == 2 and first["netcdf_files"] == 1
    assert first["skipped"] == [("SOCOL", "surface_mean", "not a 12-month time×pressure×lat field")]
    assert (site / "products/application/SOCOL" / product.name).read_bytes() == product.read_bytes()
    html = (site / "index.html").read_text()
    assert "SOCOL" in html and "temperature" in html and "K" in html
    assert '<select id="model">' in html and 'id="variables"' in html
    assert 'data-view="zonal"' in html and 'data-view="annual"' in html
    assert 'data-view="timeline"' in html
    catalog = json.loads(re.search(r'<script id="catalog" type="application/json">(.*?)</script>',
                                  html, re.DOTALL).group(1))
    assert len(catalog) == 1
    assert catalog[0]["model"] == "SOCOL" and catalog[0]["variable"] == "ta"
    for link in [catalog[0]["product"], *catalog[0]["views"].values()]:
        assert not Path(link).is_absolute() and "://" not in link
        assert (site / link).is_file()
    for link in re.findall(r'(?:href|src)="([^"]+)"', html):
        if link.startswith("#"):
            continue
        assert not Path(link).is_absolute() and "://" not in link
        assert (site / link).is_file()
    stale = site / "plots" / "OLD" / "stale.png"
    stale.parent.mkdir()
    stale.write_bytes(b"old")
    _product(tmp_path, name="o3")
    second = build_site(1985, 2014, root=tmp_path)
    assert second["png_files"] == 2
    assert not stale.exists()
    assert not (site / "plots/SOCOL/ta_zonal.png").exists()
    assert (site / "plots/SOCOL/o3_zonal.png").exists()


def test_two_models_render_in_parallel_without_mixing_outputs(tmp_path):
    _product(tmp_path, model="CMAM")
    _product(tmp_path, model="SOCOL")
    report = build_site(1985, 2014, root=tmp_path, workers=2)
    assert report["models"] == ["CMAM", "SOCOL"]
    assert report["png_files"] == 4 and report["netcdf_files"] == 2
    for model in report["models"]:
        assert (report["site"] / f"plots/{model}/ta_zonal.png").stat().st_size > 1000
        assert (report["site"] / f"plots/{model}/ta_annual.png").stat().st_size > 1000
        assert (report["site"] / f"products/application/{model}/jumacs_{model.lower()}_application_climatology_1985-2014.nc").exists()


def test_monthly_timeline_uses_existing_zonal_series(tmp_path, monkeypatch):
    from jumacs import browser

    def placeholder(*args):
        args[-1].write_bytes(b"PNG")

    monkeypatch.setattr(browser, "_plot_zonal", placeholder)
    monkeypatch.setattr(browser, "_plot_annual", placeholder)
    _product(tmp_path)
    zonal = tmp_path / "data/processed/SOCOL/refD1/ta_monthly_zonal.nc"
    zonal.parent.mkdir(parents=True)
    xr.Dataset({"ta": (("time", "plev", "lat"), np.ones((24, 2, 3)), {"units": "K"})},
               coords={"time": np.arange("1985-01", "1987-01", dtype="datetime64[M]").astype("datetime64[ns]"),
                       "plev": ("plev", [30000.0, 3000.0], {"units": "Pa"}),
                       "lat": [-62.5, 2.5, 57.5]}).to_netcdf(zonal)
    result = build_site(1985, 2014, root=tmp_path)
    assert result["png_files"] == 3 and result["timeline_skipped"] == []
    assert (result["site"] / "plots/SOCOL/ta_timeline.png").stat().st_size > 1000
    html = (result["site"] / "index.html").read_text()
    assert "plots/SOCOL/ta_timeline.png" in html


def test_exact_pressures_and_five_area_weighted_bands_keep_missing_barriers():
    assert PLOT_PRESSURES_HPA == (1000.0, 100.0, 10.0, 1.0)
    assert PLOT_LATITUDE_EDGES == (-90.0, -65.0, -20.0, 20.0, 65.0, 90.0)
    pressure = np.array([100000.0, 20000.0, 5000.0, 2000.0, 500.0, 100.0])
    latitude = np.array([-75.0, -50.0, -25.0, -15.0, 15.0, 25.0, 50.0, 75.0])
    values = np.log(pressure[:, None]) + latitude[None, :] / 100.0
    values[3] = np.nan
    field = xr.DataArray(values[None], dims=("time", "plev", "lat"),
                         coords={"time": [0], "plev": pressure, "lat": latitude})
    samples = _band_samples(field, field.plev, latitude)
    assert samples.shape == (5, 4, 1)
    np.testing.assert_allclose(samples[2, [0, 1, 3], 0], np.log([100000.0, 10000.0, 100.0]))
    assert np.isnan(samples[:, 2, 0]).all()


def test_timeline_sampling_interpolates_hybrid_columns_before_band_mean():
    latitudes = np.array([-45.0, 45.0])
    field = xr.DataArray([[[0.0, 0.0], [1.0, 2.0], [2.0, 2.0]]],
                         dims=("time", "lev", "lat"), coords={"time": [0], "lev": [0, 1, 2],
                                                                   "lat": latitudes})
    pressure = xr.DataArray([[[100000.0, 100000.0], [10000.0, 20000.0], [1000.0, 1000.0]]],
                            dims=("time", "lev", "lat"), coords=field.coords)
    samples = _band_samples(field, pressure, latitudes)
    np.testing.assert_allclose(samples[2, 1, 0], 1.5)
    assert np.isnan(samples[2, 3, 0])  # 1 hPa is above both native columns.


def test_zonal_plot_has_working_approximate_height_scale(tmp_path):
    pressure = np.geomspace(1000.0, 0.00002, 124)
    height = _pressure_to_height([1000.0, 100.0, 10.0, 1.0])
    np.testing.assert_allclose(height, [0.0, 7 * np.log(10), 14 * np.log(10), 21 * np.log(10)])
    np.testing.assert_allclose(_height_to_pressure(height), [1000.0, 100.0, 10.0, 1.0])
    path = tmp_path / "zonal.png"
    _plot_zonal(np.ones((12, 124, 36)), pressure, np.arange(-87.5, 90.0, 5.0),
                "SOCOL", "temperature", "K", path)
    assert path.stat().st_size > 1000


def test_browse_cli(monkeypatch, capsys):
    from jumacs import browser

    calls = []

    def build(start, end, *, model, workers):
        calls.append((start, end, model, workers))
        return {"site": Path("site"), "models": ["SOCOL"]}

    monkeypatch.setattr(browser, "build_site", build)
    main(["browse", "--start-year", "1985", "--end-year", "2014", "--model", "SOCOL", "--workers", "6"])
    assert calls == [(1985, 2014, "SOCOL", 6)]
    assert '"models": [\n    "SOCOL"' in capsys.readouterr().out
