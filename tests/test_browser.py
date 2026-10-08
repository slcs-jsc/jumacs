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
    _display_scale,
    _height_to_pressure,
    _map_values,
    _plot_map,
    _plot_paths,
    _plot_zonal,
    _pressure_to_height,
    _reference_year_source,
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

    def placeholder(*args, **kwargs):
        args[-1].write_bytes(b"PNG")

    monkeypatch.setattr(browser, "_plot_zonal", placeholder)
    monkeypatch.setattr(browser, "_plot_annual", placeholder)
    product = _product(tmp_path)
    first = build_site(1985, 2014, root=tmp_path)
    site = first["site"]
    assert first["models"] == ["SOCOL"]
    assert first["variables"] == {"SOCOL": 1}
    assert first["png_files"] == 2 and first["netcdf_files"] == 1
    assert first["skipped"] == [("SOCOL", "surface_mean", "no geographical source for 2D field")]
    assert (site / "products/application/SOCOL" / product.name).read_bytes() == product.read_bytes()
    html = (site / "index.html").read_text()
    assert "JuMACS climatology atlas" in html
    assert "JuMACS application climatology atlas" not in html
    assert "SOCOL" in html and "temperature" in html and "K" in html
    assert '<select id="model">' in html and 'id="variables"' in html
    assert 'data-view="zonal"' in html and 'data-view="annual"' in html
    assert 'data-view="timeline"' in html
    assert 'data-view="map"' in html
    assert "single example year 2000" in html
    catalog = json.loads(re.search(r'<script id="catalog" type="application/json">(.*?)</script>',
                                  html, re.DOTALL).group(1))
    assert len(catalog) == 1
    assert catalog[0]["model"] == "SOCOL" and catalog[0]["variable"] == "ta"
    assert catalog[0]["canonical"] == "temperature"
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


def test_plot_notes_accompany_their_plot_views(tmp_path, monkeypatch):
    from jumacs import browser

    def placeholder(*args, **kwargs):
        args[-1].write_bytes(b"PNG")

    monkeypatch.setattr(browser, "_plot_zonal", placeholder)
    monkeypatch.setattr(browser, "_plot_annual", placeholder)
    _product(tmp_path)
    site = build_site(1985, 2014, root=tmp_path)["site"]
    html = (site / "index.html").read_text()
    style = (site / "assets" / "style.css").read_text()
    notes = dict(re.findall(r'<p class="plot-note" data-note="(\w+)"[^>]*>(.*?)</p>', html, re.DOTALL))
    assert set(notes) == {"zonal", "timeline", "annual", "map"}
    figure_end, section_end = html.index("</figure>"), html.index("</section></main>")
    for match in re.finditer(r'<p class="plot-note"', html):
        assert figure_end < match.start() < section_end
    assert "January, April, July and October" in notes["zonal"]
    assert "7 km scale height" in notes["zonal"]
    for view in ("annual", "timeline"):
        assert "1000, 100, 10 and 1 hPa" in notes[view]
        assert "Missing values are not filled" in notes[view]
    assert "longer source period" in notes["timeline"]
    assert "single example year 2000" in notes["map"]
    assert "January, April, July and October" in notes["map"]
    assert 'class="note"' not in html
    assert "Cross sections:" not in html and "Monthly time series:" not in html
    assert "note.dataset.note !== view" in html
    assert ".plot-note{" in style and ".note{" not in style


def test_catalog_keeps_canonical_identity_across_different_native_names():
    from jumacs.browser import _html

    records = [(model, Path(f"{model}.nc"), [{"native": native, "canonical": "temperature",
                                               "label": "temperature", "units": "K", "views": {}}])
               for model, native in (("SOCOL", "ta"), ("WACCM-X", "T"))]
    html = _html(records, 1985, 2014, [])
    catalog = json.loads(re.search(r'<script id="catalog" type="application/json">(.*?)</script>',
                                   html, re.DOTALL).group(1))
    assert [row["variable"] for row in catalog] == ["ta", "T"]
    assert [row["canonical"] for row in catalog] == ["temperature", "temperature"]


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

    def placeholder(*args, **kwargs):
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


def test_plot_units_and_timeline_layout(tmp_path, monkeypatch):
    from jumacs import browser

    assert _display_scale(np.array([2e-6]), "mol mol-1") == (1e6, "ppmv")
    assert _display_scale(np.array([2e-8]), "mol mol-1") == (1e9, "ppbv")
    assert _display_scale(np.array([2e-10]), "mol mol-1") == (1e12, "pptv")
    assert _display_scale(np.array([250.0]), "K") == (1.0, "K")

    product = _product(tmp_path, name="o3")
    with xr.open_dataset(product) as ds:
        changed = ds.load()
    changed["o3_mean"][:] = 2e-8
    changed["o3_mean"].attrs["units"] = "mol mol-1"
    changed.to_netcdf(product)
    plots = {}

    def zonal(values, *args):
        plots["zonal"] = (values.copy(), args[-2])
        args[-1].write_bytes(b"PNG")

    def annual(field, *args, scale):
        plots["annual"] = (field.attrs["units"], args[-2], scale)
        args[-1].write_bytes(b"PNG")

    monkeypatch.setattr(browser, "_plot_zonal", zonal)
    monkeypatch.setattr(browser, "_plot_annual", annual)
    build_site(1985, 2014, root=tmp_path)
    np.testing.assert_allclose(plots["zonal"][0], 20.0)
    assert plots["zonal"][1] == "ppbv"
    assert plots["annual"] == ("mol mol-1", "ppbv", 1e9)

    original = browser.plt.subplots
    layouts = []

    def subplots(*args, **kwargs):
        layouts.append(args[:2])
        return original(*args, **kwargs)

    monkeypatch.setattr(browser.plt, "subplots", subplots)
    samples = np.ones((5, 4, 12))
    browser._plot_band_series(samples, np.arange(12), "SOCOL", "ozone", "ppbv",
                              "monthly zonal means", "Year", tmp_path / "timeline.png", timeline=True)
    assert layouts == [(4, 1)]


def test_geographical_maps_use_single_source_year_and_native_pressure(tmp_path):
    folder = tmp_path / "data/raw/SOCOL/refD1/Amon/ta"
    folder.mkdir(parents=True)
    source = folder / "ta_Amon_SOCOL_refD1_gn_r1i1p1f1_199901-200112.nc"
    time = np.arange("1999-01", "2002-01", dtype="datetime64[M]").astype("datetime64[ns]")
    values = np.full((36, 2, 3, 4), 200.0)
    values[:, 1] = 220.0
    xr.Dataset({"ta": (("time", "lev", "lat", "lon"), values, {"units": "K"}),
                "ps": (("time", "lat", "lon"), np.full((36, 3, 4), 100000.0), {"units": "Pa"}),
                "a": ("lev", [1.0, 0.1]), "b": ("lev", [0.0, 0.0]),
                "p0": ((), 100000.0, {"units": "Pa"})},
               coords={"time": time, "lev": ("lev", [0, 1], {"units": "1"}),
                       "lat": [-60.0, 0.0, 60.0], "lon": [0.0, 90.0, 180.0, 270.0]}).to_netcdf(source)
    from jumacs.config import load_config

    assert _reference_year_source(tmp_path, load_config("SOCOL"), "ta", 2000) == source
    maps, lat, lon, levels, units = _map_values(source, "SOCOL", "ta", 2000)
    assert maps.shape == (4, 4, 3, 4)
    assert levels == PLOT_PRESSURES_HPA and units == "K"
    np.testing.assert_allclose(maps[:, 0], 200.0)
    np.testing.assert_allclose(maps[:, 1], 220.0)
    assert np.isnan(maps[:, 2:]).all()  # No extrapolation above the native model top.
    np.testing.assert_array_equal(lon, [-180.0, -90.0, 0.0, 90.0])
    path = tmp_path / "map.png"
    assert _plot_map(maps, lat, lon, levels, "SOCOL", "temperature", units, 2000, path)
    assert path.stat().st_size > 1000

    surface = source.with_name("ps_Amon_SOCOL_refD1_gn_r1i1p1f1_199901-200112.nc")
    surface.write_bytes(source.read_bytes())
    maps, _, _, levels, units = _map_values(surface, "SOCOL", "ps", 2000)
    assert maps.shape == (4, 1, 3, 4) and levels is None and units == "Pa"
    np.testing.assert_allclose(maps, 100000.0)


def test_geographical_maps_use_separate_published_pressure(tmp_path):
    folder = tmp_path / "data/raw/NIWA-UKCA2/refD1/Amon"
    time = np.arange("2000-01", "2001-01", dtype="datetime64[M]").astype("datetime64[ns]")
    coords = {"time": time, "lat": [-45.0, 45.0], "lon": [0.0, 180.0]}
    name = "ta_Amon_NIWA-UKCA2_refD1_r1i1p1f1_gn_200001-200012.nc"
    source = folder / "ta" / name
    source.parent.mkdir(parents=True)
    xr.Dataset({"ta": (("time", "lev", "lat", "lon"),
                       np.broadcast_to(np.array([210.0, 230.0])[None, :, None, None],
                                       (12, 2, 2, 2)).copy(), {"units": "K"})},
               coords={**coords, "lev": [0, 1]}).to_netcdf(source)
    pressure = folder / "pa" / name.replace("ta_", "pa_", 1)
    pressure.parent.mkdir(parents=True)
    xr.Dataset({"pa": (("time", "lev", "lat", "lon"),
                       np.broadcast_to(np.array([100000.0, 10000.0])[None, :, None, None],
                                       (12, 2, 2, 2)), {"units": "Pa"})},
               coords={**coords, "lev": [0, 1]}).to_netcdf(pressure)
    maps, _, _, levels, _ = _map_values(source, "NIWA-UKCA2", "ta", 2000)
    assert levels == PLOT_PRESSURES_HPA
    np.testing.assert_allclose(maps[:, 0], 210.0)
    np.testing.assert_allclose(maps[:, 1], 230.0)
    assert np.isnan(maps[:, 2:]).all()


def test_surface_pressure_appears_as_map_only_view(tmp_path, monkeypatch):
    from jumacs import browser

    product = _product(tmp_path)
    with xr.open_dataset(product) as ds:
        changed = ds.load()
    changed["ps_mean"] = (("time", "lat"), np.full((12, 3), 100000.0), {"units": "Pa"})
    changed.to_netcdf(product)
    folder = tmp_path / "data/raw/SOCOL/refD1/Amon/ps"
    folder.mkdir(parents=True)
    source = folder / "ps_Amon_SOCOL_refD1_gn_r1i1p1f1_200001-200012.nc"
    time = np.arange("2000-01", "2001-01", dtype="datetime64[M]").astype("datetime64[ns]")
    xr.Dataset({"ps": (("time", "lat", "lon"), np.full((12, 3, 4), 100000.0),
                       {"units": "Pa"})},
               coords={"time": time, "lat": [-60.0, 0.0, 60.0],
                       "lon": [0.0, 90.0, 180.0, 270.0]}).to_netcdf(source)

    monkeypatch.setattr(browser, "_plot_zonal", lambda *args: args[-1].write_bytes(b"PNG"))
    monkeypatch.setattr(browser, "_plot_annual", lambda *args, **kwargs: args[-1].write_bytes(b"PNG"))

    def map_plot(*args):
        args[-1].write_bytes(b"PNG")
        return True

    monkeypatch.setattr(browser, "_plot_map", map_plot)
    site = build_site(1985, 2014, root=tmp_path)["site"]
    html = (site / "index.html").read_text()
    catalog = json.loads(re.search(r'<script id="catalog" type="application/json">(.*?)</script>',
                                  html, re.DOTALL).group(1))
    surface = next(row for row in catalog if row["variable"] == "ps")
    assert surface["views"] == {"map": "plots/SOCOL/ps_map.png"}
    assert (site / surface["views"]["map"]).is_file()
    inventory = json.loads(re.search(r'<script id="inventory-data" type="application/json">(.*?)</script>',
                                     html, re.DOTALL).group(1))
    pressure = next(row for row in inventory if row["native_variable"] == "ps")
    assert pressure["map_plot"] == surface["views"]["map"]
    assert not Path(pressure["map_plot"]).is_absolute()


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
