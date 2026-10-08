"""Inventory of existing products and its browser representation."""

import csv
import json
import re

import numpy as np
import xarray as xr

from jumacs import browser, inventory
from jumacs.cli import main
from jumacs.climatology import product_name


def _products(root):
    native = root / "products/climatology/SOCOL" / product_name("SOCOL", 1985, 2014)
    native.parent.mkdir(parents=True)
    xr.Dataset({"ta_mean": (("time", "plev", "lat"), np.ones((12, 2, 2)),
                            {"units": "K", "long_name": "air temperature"}),
                "ps_mean": (("time", "lat"), np.ones((12, 2)), {"units": "Pa"})},
               coords={"time": range(12), "plev": [10000., 1000.], "lat": [-45., 45.]}).to_netcdf(native)
    app = root / "products/application/SOCOL/jumacs_socol_application_climatology_1985-2014.nc"
    app.parent.mkdir(parents=True)
    values = np.ones((12, 3, 2))
    values[:, 0, :] = np.nan
    values[0, 1, 0] = np.nan
    xr.Dataset({"ta_mean": (("time", "pressure", "lat"), values,
                            {"units": "K", "long_name": "air temperature",
                             "extension_applied": "true", "extension_donor": "WACCM-X"}),
                "ps_mean": (("time", "lat"), np.ones((12, 2)),
                            {"units": "Pa", "extension_applied": "false"})},
               coords={"time": range(12), "pressure": [100000., 10000., 1000.],
                       "lat": [-45., 45.]}).to_netcdf(app)
    return native, app


def test_inventory_mapping_dimensions_coverage_and_files(tmp_path):
    native, app = _products(tmp_path)
    report = inventory.build_inventory(1985, 2014, model="SOCOL", root=tmp_path)
    rows = {row["canonical_variable"]: row for row in report["rows"]}
    ta = rows["temperature"]
    assert ta["native_variable"] == "ta" and ta["dimensionality"] == "3D"
    assert ta["native_available"] and ta["application_available"]
    assert ta["native_product"] == native.relative_to(tmp_path).as_posix()
    assert ta["application_product"] == app.relative_to(tmp_path).as_posix()
    assert ta["bottom_pressure_hpa"] == 100.0 and ta["top_pressure_hpa"] == 10.0
    assert ta["finite_fraction"] == 47 / 72
    assert ta["waccmx_extended"] is True
    ps = rows["surface_pressure"]
    assert ps["dimensionality"] == "2D" and ps["waccmx_extended"] is False
    assert ps["bottom_pressure_hpa"] is None and ps["top_pressure_hpa"] is None
    assert not rows["O3"]["native_available"] and not rows["O3"]["application_available"]
    folder = tmp_path / "products/catalog"
    with (folder / "climatology_inventory.csv").open() as stream:
        written = list(csv.DictReader(stream))
    assert len(written) == len(report["rows"])
    assert json.loads((folder / "climatology_inventory.json").read_text()) == report
    markdown = (folder / "climatology_inventory.md").read_text()
    assert "SOCOL:" in markdown and "temperature" in markdown


def test_inventory_handles_missing_product_and_unknown_extension(tmp_path):
    _, app = _products(tmp_path)
    (tmp_path / "products/climatology/SOCOL" / product_name("SOCOL", 1985, 2014)).unlink()
    with xr.open_dataset(app) as source:
        changed = source.load()
    changed["ta_mean"].attrs.pop("extension_applied")
    changed.to_netcdf(app)
    report = inventory.build_inventory(model="SOCOL", root=tmp_path)
    ta = next(row for row in report["rows"] if row["canonical_variable"] == "temperature")
    assert not ta["native_available"] and ta["application_available"]
    assert ta["native_product"] is None and ta["waccmx_extended"] is None
    app.unlink()
    report = inventory.build_inventory(model="SOCOL", root=tmp_path)
    assert all(not row["native_available"] and not row["application_available"] for row in report["rows"])


def test_inventory_in_site_and_cli(tmp_path, monkeypatch, capsys):
    _products(tmp_path)
    monkeypatch.setattr(browser, "_plot_zonal", lambda *args: args[-1].write_bytes(b"PNG"))
    monkeypatch.setattr(browser, "_plot_annual", lambda *args, **kwargs: args[-1].write_bytes(b"PNG"))
    site = browser.build_site(1985, 2014, model="SOCOL", root=tmp_path)["site"]
    html = (site / "index.html").read_text()
    assert "Data availability" in html and 'id="inventory-search"' in html
    assert "Longitude-resolved" in html and "Dimensionality" in html
    assert "Climatology grid" not in html and "Native name" not in html
    assert 'data-sort="map_plot">Longitude-resolved</button>' in html
    assert "2D/3D</button>" not in html
    assert "Click a column heading to sort (↕)" in html
    assert "inventoryDescending ? ' ↓' : ' ↑' : ' ↕'" in html
    assert 'data-sort="model"' in html and 'data-sort="canonical_variable"' in html
    assert "aria-sort" in html and "inventoryDescending" in html
    rows = json.loads(re.search(r'<script id="inventory-data" type="application/json">(.*?)</script>',
                               html, re.DOTALL).group(1))
    ta = next(row for row in rows if row["canonical_variable"] == "temperature")
    assert ta["native_variable"] == "ta"
    assert ta["dimensionality"] == "3D" and ta["map_plot"] is None
    link = ta["application_product"]
    assert link == "products/application/SOCOL/jumacs_socol_application_climatology_1985-2014.nc"
    assert (site / link).is_file()
    assert not link.startswith("/") and ".." not in link
    monkeypatch.setattr(inventory, "build_inventory", lambda start, end, *, model: {
        "summary": {"application_models": 1, "per_model": {"SOCOL": {"total": 2}}}})
    main(["inventory", "--model", "SOCOL", "--start-year", "1985", "--end-year", "2014"])
    assert '"application_models": 1' in capsys.readouterr().out
