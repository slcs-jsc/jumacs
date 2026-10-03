"""Static application-product browser wiring and regeneration."""

import re
from pathlib import Path

import numpy as np
import xarray as xr

from jumacs.browser import (
    ZONAL_MONTHS,
    _plot_paths,
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


def test_browse_cli(monkeypatch, capsys):
    from jumacs import browser

    calls = []

    def build(start, end, *, model):
        calls.append((start, end, model))
        return {"site": Path("site"), "models": ["SOCOL"]}

    monkeypatch.setattr(browser, "build_site", build)
    main(["browse", "--start-year", "1985", "--end-year", "2014", "--model", "SOCOL"])
    assert calls == [(1985, 2014, "SOCOL")]
    assert '"models": [\n    "SOCOL"' in capsys.readouterr().out
