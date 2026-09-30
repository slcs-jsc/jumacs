#!/usr/bin/env python3
"""Transfer prepared JuMACS manifests where Internet egress is available.

Create manifests with `jumacs download` on JUWELS first. This
transfer helper needs only requests plus the JuMACS registry in src/, so it
runs on JUDAC's system Python 3.9 as well as on a JUWELS login node.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from jumacs.config import models_with_capability

MANIFEST_KEYS = ("model", "experiment", "files", "selected_bytes_estimate")
RECORD_KEYS = ("filename", "local_path", "source_url", "download_status")


def is_manifest(data):
    """True only for download manifests written by `jumacs download` on JUWELS."""
    if not isinstance(data, dict) or any(key not in data for key in MANIFEST_KEYS):
        return False
    files = data["files"]
    if not isinstance(files, list):
        return False
    return all(isinstance(record, dict) and all(key in record for key in RECORD_KEYS) for record in files)


def load_manifest(path):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Not a readable manifest: {}".format(path)) from exc
    if not is_manifest(data):
        raise ValueError("Not a JuMACS download manifest (missing model/experiment/files records): {}".format(path))
    return data


def discover_manifests():
    found = {}
    for path in sorted((ROOT / "products/manifests").glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not is_manifest(data) or not isinstance(data["model"], str):
            continue
        found[data["model"]] = path
    return found


def save(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(path)


def transfer(record):
    destination = (ROOT / record["local_path"]).resolve()
    if ROOT.resolve() not in destination.parents:
        raise ValueError("Manifest destination is outside the project; regenerate it on JUWELS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if record["download_status"] == "complete" and destination.is_file():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": "bytes={}-".format(offset)} if offset else {}
    with requests.get(record["source_url"] + "?download=1", headers=headers,
                      stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        append = bool(offset and response.status_code == 206)
        if offset and not append:
            offset = 0
        expected_exact = None
        if append:
            match = re.search(r"/(\d+)$", response.headers.get("Content-Range", ""))
            if match:
                expected_exact = int(match.group(1))
        elif response.headers.get("Content-Length"):
            expected_exact = int(response.headers["Content-Length"])
        with partial.open("ab" if append else "wb") as target:
            for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                if block:
                    target.write(block)
    actual = partial.stat().st_size
    if expected_exact is not None and actual != expected_exact:
        raise IOError("Incomplete transfer: {}: {} != {}".format(destination.name, actual, expected_exact))
    rounded = record.get("size_bytes")
    if rounded and abs(actual - rounded) > rounded * .15:
        raise IOError("Size differs from rounded CEDA listing: {}".format(destination.name))
    with partial.open("rb") as stream:
        signature = stream.read(8)
    if not (signature.startswith(b"CDF") or signature == b"\x89HDF\r\n\x1a\n"):
        raise IOError("Response is not a NetCDF/HDF file: {}".format(destination.name))
    partial.replace(destination)
    record["download_status"] = "complete"
    record.pop("error", None)


def download_models():
    """Registry-driven accepted models: every source with download capability."""
    return models_with_capability("download")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*download_models(), "all"), default="all")
    parser.add_argument("--execute", action="store_true", help="Transfer pending files")
    return parser


def main():
    manifests = discover_manifests()
    args = build_parser().parse_args()
    models = download_models() if args.model == "all" else (args.model,)
    selected = []
    for model in models:
        path = manifests.get(model)
        if path is None:
            raise SystemExit("No manifest for {}; create it first with: jumacs download --model {}".format(model, model))
        manifest = load_manifest(path)
        if manifest["model"] != model:
            raise ValueError("Incorrect manifest: {}".format(path))
        files = manifest["files"]
        pending = sum(not (row["download_status"] == "complete" and (ROOT / row["local_path"]).is_file()) for row in files)
        print("{}: {} files, {} pending, ~{:.2f} GB listed".format(model, len(files), pending, manifest["selected_bytes_estimate"] / 1e9), flush=True)
        selected.append((path, manifest))
    if not args.execute:
        print("Dry run only. Add --execute to transfer.")
        return
    failures = []
    for path, manifest in selected:
        for index, record in enumerate(manifest["files"], 1):
            if record["download_status"] == "complete" and (ROOT / record["local_path"]).is_file():
                continue
            print("{} {}/{} {}".format(manifest["model"], index, len(manifest["files"]), record["filename"]), flush=True)
            for attempt in range(1, 4):
                try:
                    transfer(record)
                except Exception as exc:
                    record["download_status"] = "failed"
                    record["error"] = str(exc)
                    save(path, manifest)
                    print("Attempt {}/3 failed: {}: {}".format(attempt, record["filename"], exc), flush=True)
                    if attempt < 3:
                        time.sleep(5 * attempt)
                else:
                    save(path, manifest)
                    break
            if record["download_status"] != "complete":
                failures.append(record["filename"])
    if failures:
        print("Failed after retries: {}".format(", ".join(failures)), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
