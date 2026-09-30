import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/judac_download.py"
    spec = importlib.util.spec_from_file_location("judac_download", path)
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
