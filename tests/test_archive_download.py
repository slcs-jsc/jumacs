import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/archive_download.py"
    spec = importlib.util.spec_from_file_location("archive_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, data, status, headers):
        self.data = data
        self.status_code = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data


def test_judac_transfer_and_resume(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = b"\x89HDF\r\n\x1a\n" + b"synthetic-netcdf"
    destination = tmp_path / "raw/test.nc"
    record = {"local_path": str(destination), "source_url": "https://example.test/test.nc",
              "size_bytes": len(payload), "download_status": "pending"}
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response(payload, 200, {"Content-Length": str(len(payload))}))
    module.transfer(record)
    assert destination.read_bytes() == payload
    assert record["download_status"] == "complete"
    destination.unlink()
    partial = destination.with_suffix(".nc.part")
    partial.write_bytes(payload[:8])
    record["download_status"] = "pending"
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response(payload[8:], 206, {"Content-Range": "bytes 8-{}/{}".format(len(payload)-1, len(payload))}))
    module.transfer(record)
    assert destination.read_bytes() == payload


def test_model_choices_follow_registry_download_capability():
    module = _module()
    from jumacs.config import models_with_capability
    accepted = set(models_with_capability("download"))
    assert "CMAM" in accepted and "WACCM-X" in accepted and "GEOSCCM" in accepted
    assert "MIROC-ES2H" not in accepted
    assert set(module.download_models()) == accepted
    for name in ("CMAM", "GEOSCCM", "WACCM-X", "all"):
        assert module.build_parser().parse_args(["--model", name]).model == name
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--model", "MIROC-ES2H"])


def test_model_choices_are_not_hardcoded(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "models_with_capability", lambda capability: ("FOO", "BAR"))
    parser = module.build_parser()
    model_action = next(a for a in parser._actions if a.dest == "model")
    assert set(model_action.choices) == {"FOO", "BAR", "all"}


def test_all_model_dry_run_covers_every_download_capable_manifest(tmp_path, monkeypatch, capsys):
    module = _module()
    monkeypatch.setattr(module, "download_models", lambda: ("CMAM", "WACCM-X"))
    manifests = {}
    for model, slug in (("CMAM", "cmam_refd1"), ("WACCM-X", "waccmx")):
        path = tmp_path / f"{slug}.json"
        path.write_text(__import__("json").dumps({"model": model, "experiment": "refD1",
            "selected_bytes_estimate": 10, "files": [{"filename": "a.nc", "local_path": "raw/a.nc",
            "source_url": "https://example.invalid/a.nc", "download_status": "pending", "size_bytes": 10}]}))
        manifests[model] = path
    monkeypatch.setattr(module, "discover_manifests", lambda: manifests)
    monkeypatch.setattr("sys.argv", ["archive_download.py"])
    module.main()
    out = capsys.readouterr().out
    assert "CMAM: 1 files, 1 pending" in out and "WACCM-X: 1 files, 1 pending" in out
    assert "Dry run only" in out


def test_missing_manifest_fails_loudly(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "download_models", lambda: ("CMAM",))
    monkeypatch.setattr(module, "discover_manifests", lambda: {})
    monkeypatch.setattr("sys.argv", ["archive_download.py", "--model", "CMAM"])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    assert "jumacs download --model CMAM" in str(exit_info.value)
