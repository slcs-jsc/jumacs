import json

import pytest

import jumacs.download as download_module
from jumacs.download import full_plan


GN = "https://dap.ceda.ac.uk/badc/ccmi/data/post-cmip6/ccmi-2022/ETH-PMOD/SOCOL/refD1/gn"
GNZ = "https://dap.ceda.ac.uk/badc/ccmi/data/post-cmip6/ccmi-2022/ETH-PMOD/SOCOL/refD1/gnz"


def _entry(family, variable, filename, grid, version="v20210101", size=100, start="196001", end="201812", url=None):
    return {"family": family, "variable": variable, "grid": grid, "version": version, "filename": filename,
            "source_url": url or f"https://example.invalid/{filename}", "start": start, "end": end,
            "size_bytes": size, "checksum": None}


def _socol_inventory():
    return {"model": "SOCOL", "experiment": "refD1", "member": "r1i1p1f1",
            "members_in_archive": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"], "files": [
        _entry("Amon", "o3", "o3_Amon_SOCOL_refD1_gn_r1i1p1f1_196001-201812.nc", "gn", url=GN + "/Amon/o3/r1i1p1f1/v20210531/o3_Amon_SOCOL_refD1_gn_r1i1p1f1_196001-201812.nc"),
        _entry("Amon", "o3", "o3_Amon_SOCOL_refD1_gn_r2i1p1f1_196001-201812.nc", "r2i1p1f1", url=GN + "/Amon/o3/r2i1p1f1/v20210531/o3_Amon_SOCOL_refD1_gn_r2i1p1f1_196001-201812.nc"),
        _entry("AmonZ", "clo", "clo_AmonZ_SOCOL_refD1_gnz_r1i1p1f1_196001-201812.nc", "r1i1p1f1", url=GNZ + "/AmonZ/clo/r1i1p1f1/v20210531/clo_AmonZ_SOCOL_refD1_gnz_r1i1p1f1_196001-201812.nc"),
        _entry("AmonZ", "clo", "clo_AmonZ_SOCOL_refD1_gnz_r3i1p1f1_196001-201812.nc", "r3i1p1f1", url=GNZ + "/AmonZ/clo/r3i1p1f1/v20210531/clo_AmonZ_SOCOL_refD1_gnz_r3i1p1f1_196001-201812.nc")]}


@pytest.fixture
def patched(tmp_path, monkeypatch):
    def apply(inventories):
        monkeypatch.setattr(download_module, "ROOT", tmp_path)
        monkeypatch.setattr(download_module, "load_inventory", lambda model: inventories[model])
        return tmp_path
    return apply


def test_full_plan_selects_every_file_and_dedupes_superseded_versions(patched, monkeypatch):
    inventory = {"model": "GEOSCCM", "experiment": "refD1", "files": [
        _entry("Amon", "o3", "o3_r1i1p1f1_old.nc", "gr", version="v20210101", size=100),
        _entry("Amon", "o3", "o3_r1i1p1f1_new.nc", "gr", version="v20220101", size=120),
        _entry("AmonZ", "clo", "clo_r1i1p1f1.nc", "grz", size=80)]}
    root = patched({"GEOSCCM": inventory})
    payload = full_plan("GEOSCCM")
    assert payload["inventory_file_count"] == 3 and payload["planned_file_count"] == 2
    assert payload["deduplicated_count"] == 1 and payload["unplannable_count"] == 0
    assert payload["inventory_bytes"] == 300 and payload["planned_bytes"] == 200
    assert {r["filename"] for r in payload["files"]} == {"o3_r1i1p1f1_new.nc", "clo_r1i1p1f1.nc"}
    assert payload["pending_file_count"] == 2 and payload["pending_bytes"] == 200
    assert payload["mode"] == "full_archive" and payload["member"] == "r1i1p1f1"
    assert json.loads(download_module.manifest_path("GEOSCCM").read_text()) == payload
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("archive_download", Path(__file__).resolve().parents[1] / "scripts/archive_download.py")
    judac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judac)
    assert judac.is_manifest(payload)
    assert not any(root.rglob("*.nc"))


def test_full_plan_socol_split_archive_bases(patched):
    inventory = _socol_inventory()
    patched({"SOCOL": inventory})
    payload = full_plan("SOCOL")
    assert payload["planned_file_count"] == 2 == payload["inventory_file_count"]
    assert payload["planned_bytes"] == payload["inventory_bytes"] == 200
    assert payload["families"] == {"Amon": {"inventory": 1, "planned": 1}, "AmonZ": {"inventory": 1, "planned": 1}}
    urls = [r["source_url"] for r in payload["files"]]
    assert sum(u.startswith(GN + "/") for u in urls) == 1 and sum(u.startswith(GNZ + "/") for u in urls) == 1
    paths = [r["local_path"] for r in payload["files"]]
    assert len(set(paths)) == 2
    assert payload["member"] == "r1i1p1f1"
    assert payload["members_in_archive_excluded"] == ["r2i1p1f1", "r3i1p1f1"]
    assert not any("r2i1p1f1" in r["filename"] or "r3i1p1f1" in r["filename"] for r in payload["files"])


def test_full_plan_ukesm1_keeps_eight_digit_dates_and_all_files(patched):
    inventory = {"model": "UKESM1-StratTrop", "experiment": "refD1", "files": [
        _entry("Amon", "o3", "o3_Amon_UKESM1-StratTrop_refD1_gn_r1i1p1f2_19600101-20190101.nc", "gn",
               start="19600101", end="20190101", size=500),
        _entry("AmonZ", "clo", "clo_AmonZ_UKESM1-StratTrop_refD1_grz_r1i1p1f2_19600101-20190101.nc", "grz",
               start="19600101", end="20190101", size=300)]}
    patched({"UKESM1-StratTrop": inventory})
    payload = full_plan("UKESM1-StratTrop")
    assert payload["first_month"] == "19600101" and payload["last_month"] == "20190101"
    assert payload["planned_file_count"] == 2
    assert {r["filename"] for r in payload["files"]} == {f["filename"] for f in inventory["files"]}
    assert all(len(r["start"]) == 8 for r in payload["files"])


def test_full_plan_collection_level_uuid_does_not_block(patched):
    inventory = {"model": "CESM2-WACCM", "experiment": "refD1", "files": [
        _entry("Amon", "o3", "o3.nc", "gn", size=10), _entry("AmonZ", "clo", "clo.nc", "gnz", size=20)]}
    patched({"CESM2-WACCM": inventory})
    payload = full_plan("CESM2-WACCM")
    assert payload["planned_file_count"] == 2
    assert payload["dataset_uuid_scope"] == "collection"


def test_full_plan_marks_existing_complete_files(patched):
    from jumacs.config import load_config, ROOT
    entry = _entry("Amon", "o3", "o3.nc", "gr", size=100)
    root = patched({"GEOSCCM": {"model": "GEOSCCM", "experiment": "refD1", "files": [entry]}})
    local = root / load_config("GEOSCCM")["paths"]["raw"] / "Amon" / "o3" / "o3.nc"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x" * 100)
    payload = full_plan("GEOSCCM")
    assert payload["already_complete_file_count"] == 1 and payload["already_complete_bytes"] == 100
    assert payload["pending_file_count"] == 0 and payload["pending_bytes"] == 0


def test_full_plan_waccmx_includes_every_zm_file(patched):
    files = [_entry("monthly_zonal", "*", f"a_{month}_zm.nc", "", start=month, end=month, size=1000)
             for month in ("195001", "195002", "201512")]
    patched({"WACCM-X": {"model": "WACCM-X", "experiment": "transient-1950-2015", "files": files}})
    payload = full_plan("WACCM-X")
    assert payload["planned_file_count"] == 3 and payload["inventory_bytes"] == payload["planned_bytes"] == 3000
    assert payload["families"] == {"monthly_zonal": {"inventory": 3, "planned": 3}}
    assert payload["first_month"] == "195001" and payload["last_month"] == "201512"
    from jumacs.config import load_config
    raw = load_config("WACCM-X")["paths"]["raw"]
    assert all(r["local_path"] == f"{raw}/{r['filename']}" for r in payload["files"])


def test_download_cli_all_files_is_plan_only_and_rejects_window(monkeypatch, patched):
    import jumacs.cli as cli
    calls = []
    inventory = {"model": "GEOSCCM", "experiment": "refD1", "files": [_entry("Amon", "o3", "o3.nc", "gr")]}
    patched({"GEOSCCM": inventory})
    monkeypatch.setattr(cli, "download", lambda p: calls.append("download"))
    cli.main(["download", "--model", "GEOSCCM", "--all-files"])
    assert calls == [] and download_module.manifest_path("GEOSCCM").exists()
    with pytest.raises(SystemExit):
        cli.main(["download", "--model", "GEOSCCM", "--all-files", "--start-year", "1990"])


def test_download_audit_summary_totals_exclusions_and_determinism(tmp_path, monkeypatch):
    import jumacs.audit as audit
    import requests
    inventories = {"SOCOL": _socol_inventory(),
                   "UKESM1-StratTrop": {"model": "UKESM1-StratTrop", "experiment": "refD1", "files": [
                       _entry("Amon", "o3", "o3_19600101-20190101.nc", "gn", start="19600101", end="20190101", size=500)]},
                   "WACCM-X": {"model": "WACCM-X", "experiment": "transient-1950-2015", "files": [
                       _entry("monthly_zonal", "*", "a_zm.nc", "", size=1000)]}}
    monkeypatch.setattr(download_module, "ROOT", tmp_path)
    monkeypatch.setattr(download_module, "load_inventory", lambda model: inventories[model])
    monkeypatch.setattr(audit, "load_inventory", lambda model: inventories[model])
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network access")))
    models = ("SOCOL", "UKESM1-StratTrop", "WACCM-X")
    first = audit.write_download_plan_summary(models, root=tmp_path)
    second = audit.write_download_plan_summary(models, root=tmp_path)
    assert [p.name for p in first] == ["download_plan_summary.csv", "download_plan_summary.md"]
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes()
    csv_text = first[0].read_text()
    assert "MIROC-ES2H" not in csv_text
    report = first[1].read_text()
    assert "**CCMI only**: 2 sources" in report and "**WACCM-X (separate)**: 1 sources" in report
    assert "**Grand total**: 3 sources" in report
    assert "excluded (unavailable): MIROC-ES2H" in report
    assert "no audit mismatches" in report
    from jumacs.audit import audit_row
    socol = audit_row("SOCOL")
    assert socol["audit_status"] == "ok" and socol["planned_files"] == 2
    assert socol["member"] == "r1i1p1f1" and "r2i1p1f1" not in socol["member"]
    assert "gn + gnz" in socol["notes"]


def test_download_audit_cli_dispatch(monkeypatch, tmp_path, capsys):
    import jumacs.audit as audit
    import jumacs.cli as cli
    monkeypatch.setattr(audit, "audit_row", lambda model: {"model": model})
    monkeypatch.setattr(audit, "write_download_plan_summary",
                        lambda models=None: [tmp_path / "download_plan_summary.csv", tmp_path / "download_plan_summary.md"])
    cli.main(["download-audit"])
    out = capsys.readouterr().out
    assert "download_plan_summary.csv" in out and "download_plan_summary.md" in out
