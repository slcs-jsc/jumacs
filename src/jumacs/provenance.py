"""Publication-oriented provenance catalog for the source datasets JuMACS uses."""

import csv
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from . import config

FIELDS = ("dataset", "institution", "role", "experiment", "member", "period_used",
          "source_archive", "doi", "persistent_identifier", "dataset_uuid",
          "license", "raw_data_used_bytes", "raw_data_used_human", "accessed", "notes")

_MEMBER = re.compile(r"r\d+i\d+p\d+f\d+")
_ARCHIVE_NAMES = {"data.ceda.ac.uk": "CEDA"}


def _archive_urls(entry):
    return str(entry["model"].get("archive_base", "")).split()


def _member(entry):
    member = entry["model"].get("member")
    if member:
        return str(member)
    for url in _archive_urls(entry):
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        if _MEMBER.fullmatch(tail):
            return tail
    return ""


def _directory_size(path):
    if not path.is_dir():
        return None
    return sum(item.stat().st_size for item in path.rglob("*")
               if item.is_file() and not item.is_symlink())


def _human_bytes(size):
    if size is None:
        return "unknown"
    for unit, scale in (("GB", 10 ** 9), ("MB", 10 ** 6), ("KB", 10 ** 3)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} B"


def _notes(entry):
    notes = []
    if entry["model"].get("dataset_uuid_scope") == "collection":
        notes.append("collection-level persistent identifier: the dataset UUID covers the "
                     "model's archive collection, not this activity alone")
    omitted = entry.get("exclude_from_climatology") or []
    if omitted:
        notes.append("omitted from the climatology for source-data reasons: " + ", ".join(omitted))
    return "; ".join(notes)


def _row(name, root):
    entry = config.load_config(name)
    model = entry["model"]
    urls = _archive_urls(entry)
    netloc = urlsplit(urls[0]).netloc if urls else ""
    period = config.model_period(name)
    bytes_used = _directory_size(root / entry["paths"]["raw"])
    return {"dataset": model.get("display_name", name),
            "institution": model.get("institution", "unresolved"),
            "role": "upper-atmosphere extension" if config.is_waccmx(entry) else "primary climatology",
            "experiment": model.get("experiment", ""),
            "member": _member(entry),
            "period_used": f"{period['start_year']}–{period['end_year']}",
            "source_archive": _ARCHIVE_NAMES.get(netloc, netloc),
            "doi": model.get("doi"),
            "persistent_identifier": urls[0] if urls else None,
            "dataset_uuid": model.get("dataset_uuid"),
            "license": model.get("license", "unknown"),
            "raw_data_used_bytes": bytes_used,
            "raw_data_used_human": _human_bytes(bytes_used),
            "accessed": str(model["verified"]) if model.get("verified") else None,
            "notes": _notes(entry)}


def build_provenance(*, root=None):
    """Write the provenance catalog for every ready source and return the document."""
    root = config.ROOT if root is None else Path(root)
    rows = [_row(name, root) for name in sorted(config.ready_model_names())]
    document = {"size_units": "decimal units (1 GB = 10^9 bytes)",
                "raw_data_used_note": "bytes in regular files under the configured raw source directory",
                "sources": rows}
    directory = root / "products" / "catalog"
    directory.mkdir(parents=True, exist_ok=True)
    basename = directory / "data_provenance"
    with basename.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    basename.with_suffix(".json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document
