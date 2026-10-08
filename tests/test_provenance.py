"""Source data provenance catalog: rows, files, volumes and its browser page."""

import csv
import json
import re

import numpy as np
import xarray as xr

from jumacs import browser, config, provenance
from jumacs.cli import main


def test_provenance_rows_identifiers_licences_and_volumes(tmp_path):
    raw = tmp_path / "data/raw/SOCOL/refD1/Amon/ta/r1i1p1f1"
    raw.mkdir(parents=True)
    (raw / "ta.nc").write_bytes(b"x" * 2500)
    (tmp_path / "data/raw/SOCOL/refD1/fx.nc").write_bytes(b"y" * 1000)
    document = provenance.build_provenance(root=tmp_path)
    rows = {row["dataset"]: row for row in document["sources"]}
    assert set(rows) == {config.load_config(m)["model"].get("display_name", m) for m in config.ready_model_names()}
    assert not any("ES2H" in name for name in rows)
    socol = rows["SOCOL"]
    assert socol["member"] == "r1i1p1f1" and socol["institution"] == "ETH-PMOD"
    assert socol["role"] == "primary climatology" and socol["experiment"] == "refD1"
    assert socol["period_used"] == "1985–2014"
    assert socol["source_archive"] == "CEDA"
    assert socol["doi"] is None
    assert socol["persistent_identifier"].startswith("https://data.ceda.ac.uk/")
    assert socol["license"] == "OGL v3" and socol["accessed"] == "2026-09-30"
    assert socol["raw_data_used_bytes"] == 3500 and socol["raw_data_used_human"] == "3.5 KB"
    assert rows["UKESM1-StratTrop"]["member"] == "r1i1p1f2"
    assert rows["EMAC-CCMI2"]["member"] == "r1i1p1f1"
    assert "collection-level" in rows["CESM2-WACCM"]["notes"]
    assert rows["CESM2-WACCM"]["persistent_identifier"].endswith("/refD1/r1i1p1f1")
    assert "omitted from the climatology" in rows["NIWA-UKCA2"]["notes"]
    assert rows["EMAC-CCMI2"]["license"] == "unknown" and rows["EMAC-CCMI2"]["accessed"] is None
    assert rows["EMAC-CCMI2"]["raw_data_used_bytes"] is None
    assert rows["EMAC-CCMI2"]["raw_data_used_human"] == "unknown"
    waccmx = rows["WACCM-X transient 1950-2015"]
    assert waccmx["role"] == "upper-atmosphere extension" and waccmx["institution"] == "NCAR"
    assert waccmx["experiment"] == "transient-1950-2015" and waccmx["member"] == ""
    assert waccmx["license"] == "unknown" and waccmx["doi"] is None
    assert rows["NIWA-UKCA2"]["license"] == "OGL v3"


def test_provenance_writes_catalog_files(tmp_path):
    document = provenance.build_provenance(root=tmp_path)
    folder = tmp_path / "products/catalog"
    with (folder / "data_provenance.csv").open() as stream:
        written = list(csv.DictReader(stream))
    assert [row["dataset"] for row in written] == [row["dataset"] for row in document["sources"]]
    assert written[0]["doi"] == "" and written[0]["license"]
    stored = json.loads((folder / "data_provenance.json").read_text())
    assert stored == document
    assert stored["size_units"].startswith("decimal")


def test_provenance_page_in_site(tmp_path, monkeypatch):
    def placeholder(*args, **kwargs):
        args[-1].write_bytes(b"PNG")

    monkeypatch.setattr(browser, "_plot_zonal", placeholder)
    monkeypatch.setattr(browser, "_plot_annual", placeholder)
    folder = tmp_path / "products/application/SOCOL"
    folder.mkdir(parents=True)
    xr.Dataset({"ta_mean": (("time", "pressure", "lat"), np.ones((12, 2, 2)), {"units": "K"})},
               coords={"time": range(12), "pressure": [10000., 1000.],
                       "lat": [-45., 45.]}).to_netcdf(
        folder / "jumacs_socol_application_climatology_1985-2014.nc")
    report = browser.build_site(1985, 2014, root=tmp_path)
    assert report["provenance_sources"] == len(config.ready_model_names())
    html = (report["site"] / "index.html").read_text()
    assert "Data provenance" in html and 'id="provenance-view"' in html
    assert 'id="provenance-nav"' in html
    embedded = json.loads(re.search(
        r'<script id="provenance-data" type="application/json">(.*?)</script>', html, re.DOTALL).group(1))
    datasets = {row["dataset"] for row in embedded}
    for name in ("CESM2-WACCM", "ACCESS-CM2-Chem", "UKESM1-StratTrop", "SOCOL", "GEOSCCM",
                 "WACCM-X transient 1950-2015"):
        assert name in datasets
    assert all(row["persistent_identifier"] for row in embedded)
    assert (tmp_path / "products/catalog/data_provenance.csv").is_file()


def test_provenance_cli(monkeypatch, capsys):
    monkeypatch.setattr(provenance, "build_provenance",
                        lambda *, root=None: {"sources": [{"dataset": "SOCOL"}]})
    main(["provenance"])
    assert json.loads(capsys.readouterr().out) == {"datasets": ["SOCOL"]}
