import json

import pytest

import jumacs.archive as archive_module
import jumacs.download as download_module
from jumacs.archive import filter_selected_member, member_of
from jumacs.config import load_config
from jumacs.download import full_plan, select_files


def _entry(family, variable, filename, grid, size=100, url=None):
    return {"family": family, "variable": variable, "grid": grid, "version": "v20210101", "filename": filename,
            "source_url": url or f"https://example.invalid/{filename}", "start": "196001", "end": "201812",
            "size_bytes": size, "checksum": None}


def _socol_files():
    return [
        _entry("Amon", "o3", "o3_Amon_SOCOL_refD1_gn_r1i1p1f1_196001-201812.nc", "r1i1p1f1"),
        _entry("Amon", "o3", "o3_Amon_SOCOL_refD1_gn_r2i1p1f1_196001-201812.nc", "r2i1p1f1"),
        _entry("Amon", "o3", "o3_Amon_SOCOL_refD1_gn_r3i1p1f1_196001-201812.nc", "r3i1p1f1"),
        _entry("AmonZ", "clo", "clo_AmonZ_SOCOL_refD1_gnz_r1i1p1f1_196001-201812.nc", "r1i1p1f1"),
        _entry("AmonZ", "clo", "clo_AmonZ_SOCOL_refD1_gnz_r2i1p1f1_196001-201812.nc", "r2i1p1f1"),
    ]


def test_socol_config_pins_member_r1i1p1f1():
    assert load_config("SOCOL")["model"]["member"] == "r1i1p1f1"


def test_member_of_reads_grid_directory_and_filename_token():
    assert member_of({"grid": "r2i1p1f1", "filename": "o3_Amon_SOCOL_refD1_gn_r2i1p1f1_196001-201812.nc"}) == "r2i1p1f1"
    assert member_of({"grid": "gn", "filename": "o3_Amon_X_refD1_gn_r1i1p1f2_196001-201812.nc"}) == "r1i1p1f2"
    assert member_of({"grid": "gr", "filename": "o3.nc"}) is None


def test_filter_keeps_selected_member_and_member_unknown_entries():
    kept = filter_selected_member({"model": {"member": "r1i1p1f1"}}, _socol_files())
    assert [f["filename"] for f in kept] == [_socol_files()[0]["filename"], _socol_files()[3]["filename"]]
    assert all("r1i1p1f1" in f["filename"] for f in kept)
    extra = _entry("Amon", "ps", "ps_Amon_SOCOL_refD1_gn_196001-201812.nc", "gn")
    assert extra in filter_selected_member({"model": {"member": "r1i1p1f1"}}, _socol_files() + [extra])


def test_filter_is_identity_without_configured_member():
    files = _socol_files()
    assert filter_selected_member({"model": {}}, files) == files
    assert filter_selected_member({"model": {}}, files) is not files


def test_load_inventory_applies_member_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_module, "ROOT", tmp_path)
    root = tmp_path / "products/diagnostics/inspection/SOCOL"
    root.mkdir(parents=True)
    (root / "archive_inventory.json").write_text(json.dumps(
        {"model": "SOCOL", "experiment": "refD1", "files": _socol_files()}))
    inventory = archive_module.load_inventory("SOCOL")
    assert len(inventory["files"]) == 2
    assert all("r1i1p1f1" in f["filename"] for f in inventory["files"])


@pytest.fixture
def patched(tmp_path, monkeypatch):
    def apply(inventories):
        monkeypatch.setattr(download_module, "ROOT", tmp_path)
        monkeypatch.setattr(download_module, "load_inventory", lambda model: inventories[model])
        return tmp_path
    return apply


def test_full_plan_and_window_select_only_configured_member(patched):
    files = _socol_files()
    patched({"SOCOL": {"model": "SOCOL", "experiment": "refD1", "files": files}})
    payload = full_plan("SOCOL")
    assert payload["planned_file_count"] == payload["inventory_file_count"] == 2
    assert payload["planned_bytes"] == sum(f["size_bytes"] for f in files if "r1i1p1f1" in f["filename"])
    selected, missing = select_files("SOCOL", 1960, 2018, variables=["O3", "ClO"])
    assert len(selected) == 2 and missing == []
    assert all("r1i1p1f1" in f["filename"] for f in selected)


def test_models_without_configured_member_keep_all_realizations(patched):
    files = [_entry("Amon", "o3", "o3_Amon_GEOSCCM_refD1_gr_r1i1p1f1_196001-201812.nc", "gr"),
             _entry("Amon", "o3", "o3_Amon_GEOSCCM_refD1_gr_r2i1p1f1_196001-201812.nc", "gr")]
    patched({"GEOSCCM": {"model": "GEOSCCM", "experiment": "refD1", "files": files}})
    assert "member" not in load_config("GEOSCCM")["model"]
    assert full_plan("GEOSCCM")["planned_file_count"] == 2
    selected, _ = select_files("GEOSCCM", 1960, 2018, variables=["O3"])
    assert len(selected) == 2
