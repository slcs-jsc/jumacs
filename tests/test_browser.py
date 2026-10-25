"""Static harmonized-product browser wiring and regeneration."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from jumacs import browser
from jumacs.browser import (
    EXTENT_KEYS,
    HARMONIZED_NAME,
    STALE_PATHS,
    ZONAL_MONTHS,
    _color_scale,
    _display_name,
    _display_scale,
    _human_size,
    _section_path,
    build_site,
    discover_harmonized_products,
)
from jumacs.cli import main

NAME = "jumacs_socol_1985-2014_5deg_1km.nc"

NINE_MODELS = ["CCSR-NIES-MIROC32", "CESM2-WACCM", "CMAM", "CNRM-MOCAGE", "EMAC", "GEOSCCM",
               "IPSL-CM6A-ATM-LR-REPROBUS", "SOCOL", "WACCM-X"]

TA_ATTRS = {"units": "K", "canonical_variable": "temperature", "source_variable": "ta",
            "source_valid_bottom_km_min": 0.03, "source_valid_bottom_km_max": 2.4,
            "source_valid_top_km": 55.0, "upper_completion_km": 5.0}


def _ta_field(months=12):
    return (("month", "altitude", "lat"), np.full((months, 3, 3), 230.0, "float32"), dict(TA_ATTRS))


def _product(root, model="SOCOL", slug="socol", start=1985, end=2014, months=12, fields=None,
             valid_top=60.0, attrs_extra=None):
    folder = root / "products" / "harmonized" / model
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"jumacs_{slug}_{start}-{end}_5deg_1km.nc"
    if fields is None:
        fields = {"ta_mean": _ta_field(months)}
    dataset = xr.Dataset(fields, coords={"month": np.arange(1, months + 1),
                                         "altitude": [0.0, 30.0, 60.0], "lat": [-87.5, 0.0, 87.5]})
    dataset.attrs = {"product_type": "harmonized_climatology", "model": model,
                     "display_name": f"{model} climatology" if model != "SOCOL" else "SOCOL stratosphere",
                     "climatology_period": f"{start}-{end}", "model_valid_top_km": valid_top,
                     "latitude_band_degrees": 5.0, "altitude_step_km": 1.0,
                     "institution": "Example Institute",
                     "lower_boundary_completion": "constant nearest valid value",
                     "upper_boundary_completion": "constant nearest valid value to model top"}
    dataset.attrs.update(attrs_extra or {})
    dataset.to_netcdf(path)
    return path


def _json(html, script_id):
    return json.loads(re.search(rf'<script id="{script_id}" type="application/json">(.*?)</script>',
                                html, re.DOTALL).group(1))


def _no_plot(monkeypatch):
    monkeypatch.setattr(browser, "_plot_section", lambda *args: args[-1].write_bytes(b"PNG"))


def test_discover_harmonized_products(tmp_path):
    product = _product(tmp_path)
    folder = tmp_path / "products/harmonized/SOCOL"
    (folder / "jumacs_socol_1980-2009_5deg_1km.nc").write_bytes(b"x")
    (folder / "notes.txt").write_text("x")
    assert discover_harmonized_products(tmp_path, 1985, 2014) == [("SOCOL", product)]
    assert discover_harmonized_products(tmp_path, 1985, 2014, model="SOCOL") == [("SOCOL", product)]
    with pytest.raises(FileNotFoundError):
        discover_harmonized_products(tmp_path, 1985, 2014, model="CMAM")
    with pytest.raises(FileNotFoundError):
        discover_harmonized_products(tmp_path, 1990, 2014)
    with pytest.raises(FileNotFoundError):
        discover_harmonized_products(tmp_path / "nothing", 1985, 2014)
    assert HARMONIZED_NAME.fullmatch(product.name)["model"] == "socol"
    assert HARMONIZED_NAME.fullmatch(f"{product.name}.bak") is None


def test_section_paths_display_scales_and_sizes():
    assert ZONAL_MONTHS == (1, 4, 7, 10)
    assert _section_path("SOCOL", "ta") == Path("plots/SOCOL/ta_section.png")
    assert _section_path("SOCOL", "O3_strat") == Path("plots/SOCOL/O3_strat_section.png")
    assert _display_scale(np.array([2e-6]), "mol mol-1") == (1e6, "ppmv")
    assert _display_scale(np.array([2e-8]), "mol mol-1") == (1e9, "ppbv")
    assert _display_scale(np.array([2e-10]), "mol mol-1") == (1e12, "pptv")
    assert _display_scale(np.array([2e-13]), "mol mol-1") == (1e15, "ppqv")
    assert _display_scale(np.array([250.0]), "K") == (1.0, "K")
    assert _display_scale(np.array([100000.0]), "Pa") == (0.01, "hPa")
    assert _display_scale(np.array([10000.0]), "m") == (0.001, "km")
    assert _human_size(8) == "8 B"
    assert _human_size(1500) == "1.5 kB"
    assert _human_size(6_500_000) == "6.5 MB"


def test_display_names_are_public_scientific_labels():
    assert _display_name("surface_pressure") == "Surface pressure"
    assert _display_name("mean_age") == "Mean age"
    assert _display_name("O3_strat") == "Stratospheric ozone"
    assert _display_name("tropospheric_ozone_column") == "Tropospheric ozone column"
    assert _display_name("tropopause_pressure") == "Tropopause pressure"
    assert _display_name("tropopause_temperature") == "Tropopause temperature"
    assert _display_name("tropopause_altitude") == "Tropopause altitude"
    assert _display_name("geopotential_height") == "Geopotential height"
    assert _display_name("zonal_wind") == "Zonal wind"
    assert _display_name("meridional_wind") == "Meridional wind"
    assert _display_name("total_ozone_column") == "Total ozone column"
    assert _display_name("EP_flux_meridional") == "Meridional EP flux"
    assert _display_name("EP_flux_vertical") == "Vertical EP flux"
    assert _display_name("TEM_meridional_wind") == "TEM meridional wind"
    assert _display_name("TEM_vertical_velocity") == "TEM vertical velocity"
    assert _display_name("sulfate_SAD") == "Sulfate aerosol surface area density"
    assert _display_name("PSC_NAT_SAD") == "PSC nitric acid surface area density"
    assert _display_name("PSC_ice_SAD") == "PSC ice surface area density"
    for formula in ("O3", "H2O", "CH4", "N2O", "CO", "CO2", "NO", "NO2", "HCl", "CFC-11", "CFC-12"):
        assert _display_name(formula) == formula


def test_color_scale_is_log_only_for_positive_multi_decade_fields():
    assert _color_scale(np.array([230.0, 250.0, 270.0])) == "linear"
    assert _color_scale(np.full((4, 4), 7.0)) == "linear"
    assert _color_scale(np.array([-5.0, 1e-9, 1e-6])) == "linear"
    assert _color_scale(np.array([0.0, 1e-6, 1.0])) == "linear"
    assert _color_scale(np.array([np.nan, 1e-9, 5e-7])) == "log"
    assert _color_scale(np.array([1e-6, 1e-3])) == "log"


def test_site_links_and_regeneration(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    product = _product(tmp_path, fields={
        "ta_mean": _ta_field(),
        "o3_mean": (("month", "altitude", "lat"), np.full((12, 3, 3), np.nan, "float32"),
                    {"units": "mol mol-1", "canonical_variable": "O3", "source_variable": "o3"}),
        "ps_mean": (("month", "lat"), np.ones((12, 3), "float32"), {"units": "Pa"}),
    })
    first = build_site(1985, 2014, model="SOCOL", root=tmp_path)
    site = first["site"]
    assert first["models"] == ["SOCOL"]
    assert first["variables"] == {"SOCOL": 1}
    assert first["png_files"] == 1 and first["netcdf_files"] == 1
    assert first["skipped"] == [("SOCOL", "o3_mean", "no finite values in displayed months"),
                                ("SOCOL", "ps_mean", "not a 12-month month×altitude×lat field")]
    assert (site / "products/harmonized/SOCOL" / product.name).read_bytes() == product.read_bytes()
    html = (site / "index.html").read_text()
    assert "JuMACS harmonized model climatology browser" in html
    assert '<select id="model">' in html and 'id="variables"' in html
    catalog = _json(html, "catalog")
    assert len(catalog) == 1
    assert catalog[0]["model"] == "SOCOL" and catalog[0]["variable"] == "ta"
    assert catalog[0]["canonical"] == "temperature"
    assert catalog[0]["display"] == "Temperature" and catalog[0]["units"] == "K"
    assert catalog[0]["source"] == "ta"
    assert catalog[0]["product"] == f"products/harmonized/SOCOL/{NAME}"
    assert catalog[0]["views"] == {"section": "plots/SOCOL/ta_section.png"}
    assert "native" not in catalog[0]
    assert catalog[0]["meta"]["source_valid_top_km"] == 55.0
    assert catalog[0]["meta"]["source_valid_bottom_km_min"] == 0.03
    for link in re.findall(r'(?:href|src)="([^"]+)"', html):
        if link.startswith(("#", "mailto:")) or "://" in link:
            continue
        assert not Path(link).is_absolute() and ".." not in link
        assert (site / link).is_file(), link
    for row in catalog:
        assert (site / row["product"]).is_file()
        assert (site / row["views"]["section"]).is_file()
    for forbidden in ("products/combined", "products/application", "products/climatology",
                      "plots-ready", "compact_catalog", "comparison_catalog"):
        assert forbidden not in html
    stale = site / "plots/OLD/stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    product.unlink()
    _product(tmp_path, fields={
        "o3_mean": (("month", "altitude", "lat"), np.full((12, 3, 3), 2e-6, "float32"),
                    {"units": "mol mol-1", "canonical_variable": "O3", "source_variable": "o3"})})
    second = build_site(1985, 2014, model="SOCOL", root=tmp_path)
    assert second["png_files"] == 1
    assert not stale.exists()
    assert not (site / "plots/SOCOL/ta_section.png").exists()
    assert (site / "plots/SOCOL/o3_section.png").exists()


def test_non_climatological_product_is_skipped(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    _product(tmp_path, months=6)
    report = build_site(1985, 2014, model="SOCOL", root=tmp_path)
    assert report["models"] == ["SOCOL"]
    assert report["variables"] == {"SOCOL": 0}
    assert report["skipped"] == [("SOCOL", NAME, "not a 12-month month×altitude×lat climatology")]


def test_legacy_and_removed_paths_are_deleted(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    _product(tmp_path)
    site = tmp_path / "site"
    for name in ("products/application/SOCOL", "products/combined", "products/application-ready",
                 "products/climatology/SOCOL", "plots-ready/SOCOL"):
        legacy = site / name
        legacy.mkdir(parents=True)
        (legacy / "old.nc").write_bytes(b"old")
    for name in ("compact_catalog.json", "comparison_catalog.json"):
        (site / name).write_text("{}")
    build_site(1985, 2014, model="SOCOL", root=tmp_path)
    for stale in STALE_PATHS:
        assert not (site / stale).exists(), stale


def test_overview_lists_every_model_with_its_own_top(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    extra = {"WACCM-X": {"source_valid_top_km": 478.842,
                         "vertical_extent_policy": "source extent capped at 150 km for WACCM-X"}}
    tops = {"WACCM-X": 150.0, "CMAM": 92.348}
    for model in NINE_MODELS:
        slug = {"CMAM": "cmam_refd1", "WACCM-X": "waccmx"}.get(model, model.lower())
        _product(tmp_path, model=model, slug=slug, valid_top=tops.get(model, 71.0),
                 attrs_extra=extra.get(model, {}))
    report = build_site(1985, 2014, root=tmp_path, workers=3)
    assert sorted(report["models"]) == sorted(NINE_MODELS)
    html = (report["site"] / "index.html").read_text()
    assert html.count("<tr><td><strong>") == 9
    assert "__MODEL_COUNT__" not in html and "__OVERVIEW_ROWS__" not in html
    assert "holds 9 harmonized products for 1985–2014" in " ".join(html.split())
    assert "92.348 km" in html
    assert "150 km (capped; source top 478.842 km)" in html
    assert "jumacs_cmam_refd1_1985-2014_5deg_1km.nc" in html
    for model in NINE_MODELS:
        assert f'data-model="{model}"' in html
    infos = _json(html, "model-info")
    assert len(infos) == 9
    assert {info["model"] for info in infos} == set(NINE_MODELS)


def test_absent_variables_are_omitted_without_placeholders(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    _product(tmp_path)
    _product(tmp_path, model="EMAC", slug="emac", fields={
        "o3_mean": (("month", "altitude", "lat"), np.full((12, 3, 3), 4e-6, "float32"),
                    {"units": "mol mol-1", "canonical_variable": "O3", "source_variable": "o3"})})
    report = build_site(1985, 2014, root=tmp_path, workers=2)
    assert report["variables"] == {"EMAC": 1, "SOCOL": 1}
    catalog = _json((report["site"] / "index.html").read_text(), "catalog")
    emac = [row for row in catalog if row["model"] == "EMAC"]
    socol = [row for row in catalog if row["model"] == "SOCOL"]
    assert [row["canonical"] for row in emac] == ["O3"]
    assert [row["canonical"] for row in socol] == ["temperature"]
    assert all("placeholder" not in json.dumps(row).lower() for row in catalog)


def test_one_shared_section_plot_per_variable(tmp_path):
    _product(tmp_path)
    report = build_site(1985, 2014, root=tmp_path)
    site = report["site"]
    assert report["variables"] == {"SOCOL": 1}
    assert report["png_files"] == 1
    plot = site / "plots/SOCOL/ta_section.png"
    assert plot.is_file() and plot.stat().st_size > 1000
    assert not list((site / "plots").rglob("*map*"))
    assert not list((site / "plots").rglob("*comparison*"))


def test_downloads_are_harmonized_products_only(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    _product(tmp_path)
    report = build_site(1985, 2014, model="SOCOL", root=tmp_path)
    site = report["site"]
    html = (site / "index.html").read_text()
    assert f'href="products/harmonized/SOCOL/{NAME}"' in html
    assert "native NetCDF" not in html
    assert "products/climatology" not in html
    assert report["netcdf_files"] == 1
    assert "climatology_files" not in report
    published = list((site / "products").rglob("*.nc"))
    assert [path.name for path in published] == [NAME]


def test_vertical_extent_policy_and_provenance_are_public(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    _product(tmp_path, model="WACCM-X", slug="waccmx", valid_top=150.0, attrs_extra={
        "source_valid_top_km": 478.842,
        "vertical_extent_policy": "source extent capped at 150 km for WACCM-X"})
    report = build_site(1985, 2014, model="WACCM-X", root=tmp_path)
    html = (report["site"] / "index.html").read_text()
    text = " ".join(html.split())
    assert "constant carry of the nearest valid value" in text
    assert "interior gaps are preserved" in text
    assert "never filled, bridged or extrapolated" in text
    assert "JuMACS harmonizes, and never merges" in text
    assert "never merged, blended or bias corrected" in text
    assert "model_valid_top_km" in html
    infos = _json(html, "model-info")
    assert infos[0]["extent_policy"] == "source extent capped at 150 km for WACCM-X"
    assert infos[0]["source_top_km"] == 478.842
    assert "Source dataset" in html and "metadata" in html
    for label in ("Source-valid bottom", "Source-valid top", "Lower edge completion",
                  "Upper edge completion", "Vertical extent policy"):
        assert label in html
    assert all(key in html or key in json.dumps(_json(html, "catalog")[0]["meta"])
               for key in EXTENT_KEYS)


def test_navigation_is_overview_models_about_methodology(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    _product(tmp_path)
    report = build_site(1985, 2014, model="SOCOL", root=tmp_path)
    html = (report["site"] / "index.html").read_text()
    style = (report["site"] / "assets" / "style.css").read_text()
    assert '<button id="overview-nav" class="active">Overview</button>' in html
    assert '<button id="models-nav">Models</button>' in html
    assert '<button id="methodology-nav">Methodology</button>' in html
    assert '<button id="about-nav">About</button>' in html
    for section in ("overview", "models", "methodology-view", "about-view"):
        assert f'id="{section}"' in html
    assert "#overview[hidden]" in style and "#methodology-view[hidden]" in style
    assert "inventory" not in html.lower()
    assert "Data availability" not in html
    assert "Data provenance" not in html


def test_about_contact_and_release_header(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    _product(tmp_path)
    report = build_site(1985, 2014, model="SOCOL", root=tmp_path)
    html = (report["site"] / "index.html").read_text()
    text = " ".join(html.split())
    assert '<section id="about-view" hidden>' in html
    assert "Jülich Multi-source Atmospheric Climatology and Statistics" in html
    assert "Climatology System" not in html
    assert "Dr. Lars Hoffmann" in html
    assert "Jülich Supercomputing Centre (JSC)" in html
    assert 'href="mailto:l.hoffmann@fz-juelich.de"' in html
    assert '<a href="https://datapub.fz-juelich.de/slcs/jumacs/">Published data</a>' in html
    assert '<a href="https://github.com/slcs-jsc/jumacs">Source code on GitHub</a>' in html
    assert ("<p>JuMACS v0.1 · Initial public release · "
            "Harmonized multi-source model climatologies</p>") in html
    assert "Feedback and scientific collaboration are very welcome." in html
    assert "1985–2014" in html


def test_variable_selector_shows_display_names_without_native_aliases(tmp_path, monkeypatch):
    _no_plot(monkeypatch)
    _product(tmp_path)
    html = (build_site(1985, 2014, model="SOCOL", root=tmp_path)["site"] / "index.html").read_text()
    catalog = _json(html, "catalog")
    assert catalog[0]["display"] == "Temperature"
    assert catalog[0]["variable"] == "ta" and catalog[0]["canonical"] == "temperature"
    assert "button.textContent = row.display;" in html
    assert "row.label + ' · ' + row.variable" not in html
    assert "current.model + ' · ' + current.display" in html
    assert "row.model === model.value && row.canonical === canonical" in html


def test_catalog_keeps_canonical_identity_across_different_native_names():
    records = [(model, Path(f"{model}.nc"),
                [{"native": native, "canonical": "temperature", "display": "Temperature",
                  "units": "K", "source": native, "meta": {}, "views": {}}])
               for model, native in (("SOCOL", "ta"), ("CESM2-WACCM", "T"))]
    models = [{"model": model, "display": model, "period": "1985-2014", "variables": 1}
              for model in ("SOCOL", "CESM2-WACCM")]
    html = browser._html(models, records, 1985, 2014)
    catalog = _json(html, "catalog")
    assert [row["variable"] for row in catalog] == ["ta", "T"]
    assert [row["canonical"] for row in catalog] == ["temperature", "temperature"]
    assert [row["display"] for row in catalog] == ["Temperature", "Temperature"]


def test_two_models_render_in_parallel_without_mixing_outputs(tmp_path):
    socol = _product(tmp_path)
    cmam = _product(tmp_path, model="CMAM", slug="cmam_refd1", valid_top=92.348)
    report = build_site(1985, 2014, root=tmp_path, workers=2)
    assert report["models"] == ["CMAM", "SOCOL"]
    assert report["png_files"] == 2 and report["netcdf_files"] == 2
    assert report["variables"] == {"CMAM": 1, "SOCOL": 1}
    for model, path in (("CMAM", cmam), ("SOCOL", socol)):
        published = report["site"] / f"products/harmonized/{model}/{path.name}"
        assert published.is_file() and published.read_bytes() == path.read_bytes()
        assert (report["site"] / f"plots/{model}/ta_section.png").stat().st_size > 1000


def test_plot_section_uses_shared_range_and_altitude_axis(tmp_path):
    rng = np.random.default_rng(3)
    values = 10 ** rng.random((12, 3, 3))
    values[0, 0, 0] = np.nan
    path = tmp_path / "section.png"
    browser._plot_section(values, np.array([0.0, 30.0, 60.0]), np.array([-45.0, 0.0, 45.0]),
                          "SOCOL", "Temperature", "K", path)
    assert path.is_file() and path.stat().st_size > 1000
    flat = tmp_path / "flat.png"
    browser._plot_section(np.full((12, 3, 3), 7.0), np.array([0.0, 30.0, 60.0]),
                          np.array([-45.0, 0.0, 45.0]), "SOCOL", "Flat", "", flat)
    assert flat.is_file() and flat.stat().st_size > 1000
    spread = np.logspace(-9.0, -6.0, 12 * 3 * 3, dtype="float32").reshape(12, 3, 3)
    logp = tmp_path / "log.png"
    browser._plot_section(spread, np.array([0.0, 30.0, 60.0]),
                          np.array([-45.0, 0.0, 45.0]), "SOCOL", "O3", "mol mol-1", logp)
    assert logp.is_file() and logp.stat().st_size > 1000


def test_plot_units_are_scaled_without_changing_products(tmp_path, monkeypatch):
    captured = {}

    def fake(values, altitude, latitudes, model, label, units, path):
        captured.update(values=values.copy(), units=units, label=label)
        path.write_bytes(b"PNG")

    monkeypatch.setattr(browser, "_plot_section", fake)
    _product(tmp_path, fields={
        "o3_mean": (("month", "altitude", "lat"), np.full((12, 3, 3), 2e-8, "float32"),
                    {"units": "mol mol-1", "canonical_variable": "O3", "source_variable": "o3"})})
    build_site(1985, 2014, model="SOCOL", root=tmp_path)
    np.testing.assert_allclose(captured["values"], 20.0)
    assert captured["units"] == "ppbv" and captured["label"] == "O3"


def test_build_site_requires_at_least_one_worker(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        build_site(1985, 2014, root=tmp_path, workers=0)


def test_browse_cli(monkeypatch, capsys):
    calls = []

    def build(start, end, *, model, workers):
        calls.append((start, end, model, workers))
        return {"site": Path("site"), "models": ["SOCOL"]}

    monkeypatch.setattr(browser, "build_site", build)
    main(["browse", "--start-year", "1985", "--end-year", "2014", "--model", "SOCOL", "--workers", "6"])
    assert calls == [(1985, 2014, "SOCOL", 6)]
    assert '"models": [\n    "SOCOL"' in capsys.readouterr().out
