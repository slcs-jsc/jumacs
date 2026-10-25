"""Inventory of existing products and its browser representation."""

import csv
import json

import numpy as np
import xarray as xr

from jumacs import browser, inventory
from jumacs.cli import main
from jumacs.climatology import product_name

HARMONIZED_LINK = "products/harmonized/SOCOL/jumacs_socol_1985-2014_5deg_1km.nc"


def _products(root):
    native = root / "products/climatology/SOCOL" / product_name("SOCOL", 1985, 2014)
    native.parent.mkdir(parents=True)
    xr.Dataset({"ta_mean": (("time", "plev", "lat"), np.ones((12, 2, 2)),
                            {"units": "K", "long_name": "air temperature"}),
                "ps_mean": (("time", "lat"), np.ones((12, 2)), {"units": "Pa"})},
               coords={"time": range(12), "plev": [10000., 1000.], "lat": [-45., 45.]}).to_netcdf(native)
    harmonized = root / HARMONIZED_LINK
    harmonized.parent.mkdir(parents=True)
    values = np.ones((12, 3, 2))
    values[:, 0, :] = np.nan
    values[0, 1, 0] = np.nan
    xr.Dataset({"temperature_mean": (("month", "altitude", "lat"), values.astype("float32"),
                                      {"units": "K", "long_name": "air temperature",
                                       "canonical_variable": "temperature", "source_variable": "ta"}),
                "temperature_sigma": (("month", "altitude", "lat"), np.ones((12, 3, 2), "float32"),
                                      {"units": "K"})},
               coords={"month": np.arange(1, 13, dtype="int8"), "altitude": [0., 1., 2.],
                       "lat": [-2.5, 2.5]}).to_netcdf(harmonized)
    return native, harmonized


def test_inventory_mapping_dimensions_coverage_and_files(tmp_path):
    native, harmonized = _products(tmp_path)
    report = inventory.build_inventory(1985, 2014, model="SOCOL", root=tmp_path)
    rows = {row["canonical_variable"]: row for row in report["rows"]}
    ta = rows["temperature"]
    assert ta["native_variable"] == "ta" and ta["dimensionality"] == "3D"
    assert ta["native_available"] and ta["harmonized_available"]
    assert ta["native_product"] == native.relative_to(tmp_path).as_posix()
    assert ta["harmonized_product"] == harmonized.relative_to(tmp_path).as_posix()
    assert ta["bottom_altitude_km"] == 1.0 and ta["top_altitude_km"] == 2.0
    assert ta["finite_fraction"] == 47 / 72
    ps = rows["surface_pressure"]
    assert ps["dimensionality"] == "2D"
    assert ps["native_available"] and not ps["harmonized_available"]
    assert ps["bottom_altitude_km"] is None and ps["top_altitude_km"] is None
    assert ps["harmonized_product"] is not None
    assert not rows["O3"]["native_available"] and not rows["O3"]["harmonized_available"]
    assert report["summary"]["harmonized_models"] == 1
    assert report["summary"]["harmonized_model_variables"] == 1
    folder = tmp_path / "products/catalog"
    with (folder / "climatology_inventory.csv").open() as stream:
        written = list(csv.DictReader(stream))
    assert len(written) == len(report["rows"])
    assert json.loads((folder / "climatology_inventory.json").read_text()) == report
    markdown = (folder / "climatology_inventory.md").read_text()
    assert "SOCOL:" in markdown and "temperature" in markdown
    assert "Altitude range (km)" in markdown


def test_inventory_handles_missing_and_unmatched_products(tmp_path):
    native, harmonized = _products(tmp_path)
    native.unlink()
    report = inventory.build_inventory(model="SOCOL", root=tmp_path)
    ta = next(row for row in report["rows"] if row["canonical_variable"] == "temperature")
    assert not ta["native_available"] and ta["harmonized_available"]
    assert ta["native_product"] is None and ta["harmonized_product"] == HARMONIZED_LINK
    assert ta["units"] == "K" and ta["dimensionality"] == "3D"
    harmonized.unlink()
    report = inventory.build_inventory(model="SOCOL", root=tmp_path)
    assert all(not row["native_available"] and not row["harmonized_available"] for row in report["rows"])
    assert not report["skipped"]


def test_inventory_cli_and_site_omits_inventory_view(tmp_path, monkeypatch, capsys):
    _products(tmp_path)
    site = browser.build_site(1985, 2014, model="SOCOL", root=tmp_path)["site"]
    html = (site / "index.html").read_text()
    assert "Data availability" not in html and 'id="inventory-search"' not in html
    assert 'id="inventory-data"' not in html and "inventoryDescending" not in html
    for stale in ("products/combined", "products/application-ready", "plots-ready",
                  "compact_catalog.json", "comparison_catalog.json"):
        assert stale not in html
    assert "products/harmonized/SOCOL/jumacs_socol_1985-2014_5deg_1km.nc" in html
    monkeypatch.setattr(inventory, "build_inventory",
                        lambda start, end, *, model, root=None: {
                            "summary": {"harmonized_models": 1, "per_model": {"SOCOL": {"total": 2}}}})
    main(["inventory", "--model", "SOCOL", "--start-year", "1985", "--end-year", "2014"])
    assert '"harmonized_models": 1' in capsys.readouterr().out
