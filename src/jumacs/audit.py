"""Audit complete-archive download plans against the local CEDA inventories (no network access)."""
import csv

from .archive import load_inventory
from .config import ROOT, model_metadata, model_names, ready_ccmi_model_names
from .download import full_plan

COMPARISON_SUBDIR = "products/comparison"

AUDIT_COLUMNS = ["model", "kind", "status", "member", "first_month", "last_month", "families",
                 "inventory_files", "planned_files", "inventory_bytes", "planned_bytes", "planned_gb",
                 "already_complete_files", "already_complete_bytes", "pending_files", "pending_bytes",
                 "pending_gb", "audit_status", "notes"]

MODEL_NOTES = {"SOCOL": "two configured archive bases (gn + gnz); all inventoried members included",
               "UKESM1-StratTrop": "8-digit YYYYMMDD date stamps preserved",
               "CESM2-WACCM": "collection-level dataset_uuid (no per-experiment refD1 record published)",
               "WACCM-X": "monthly zonal multivariable _zm.nc files, kept separate from CCMI totals"}


def _gb(value):
    return round(value / 1e9, 2)


def audit_row(model):
    payload = full_plan(model)
    inventory_files = load_inventory(model)["files"]
    meta = model_metadata(model)
    inventory_count = len(inventory_files)
    inventory_bytes = sum(f["size_bytes"] or 0 for f in inventory_files)
    family_inventory = {}
    for f in inventory_files:
        family_inventory[f["family"]] = family_inventory.get(f["family"], 0) + 1
    local_paths = [record["local_path"] for record in payload["files"]]
    problems = []
    if payload["inventory_file_count"] != inventory_count:
        problems.append(f"manifest inventory count {payload['inventory_file_count']} != listing {inventory_count}")
    if payload["inventory_bytes"] != inventory_bytes:
        problems.append(f"manifest inventory bytes {payload['inventory_bytes']} != listing {inventory_bytes}")
    if payload["planned_file_count"] + payload["deduplicated_count"] + payload["unplannable_count"] != inventory_count:
        problems.append(f"planned {payload['planned_file_count']} + dedup {payload['deduplicated_count']}"
                        f" + unplannable {payload['unplannable_count']} != listing {inventory_count}")
    if len(set(local_paths)) != len(local_paths):
        problems.append("duplicate local destination paths")
    for family, counts in family_inventory.items():
        planned = payload["families"].get(family, {}).get("planned", 0)
        if planned == 0:
            problems.append(f"family {family} inventoried ({counts} files) but absent from the plan")
    notes = []
    if payload["deduplicated_count"]:
        notes.append(f"{payload['deduplicated_count']} superseded version entries deduplicated"
                     f" ({payload['inventory_bytes'] - payload['planned_bytes']:,} bytes not re-listed)")
    if payload["missing_remote_size_count"]:
        notes.append(f"{payload['missing_remote_size_count']} files without remote size metadata")
    if model in MODEL_NOTES:
        notes.append(MODEL_NOTES[model])
    if problems:
        notes.insert(0, "AUDIT FAILURE: " + "; ".join(problems))
    return {"model": model, "kind": meta["kind"], "status": meta["status"], "member": payload["member"],
            "first_month": payload["first_month"], "last_month": payload["last_month"],
            "families": "; ".join(f"{name} {counts['inventory']}/{counts['planned']}"
                                  for name, counts in payload["families"].items()),
            "inventory_files": payload["inventory_file_count"], "planned_files": payload["planned_file_count"],
            "inventory_bytes": payload["inventory_bytes"], "planned_bytes": payload["planned_bytes"],
            "planned_gb": _gb(payload["planned_bytes"]),
            "already_complete_files": payload["already_complete_file_count"],
            "already_complete_bytes": payload["already_complete_bytes"],
            "pending_files": payload["pending_file_count"], "pending_bytes": payload["pending_bytes"],
            "pending_gb": _gb(payload["pending_bytes"]),
            "audit_status": "ok" if not problems else "mismatch", "notes": "; ".join(notes)}


def _totals(rows, label):
    total = {"label": label, "models": len(rows),
             "inventory_files": sum(row["inventory_files"] for row in rows),
             "planned_files": sum(row["planned_files"] for row in rows),
             "planned_bytes": sum(row["planned_bytes"] for row in rows),
             "already_complete_files": sum(row["already_complete_files"] for row in rows),
             "pending_files": sum(row["pending_files"] for row in rows),
             "pending_bytes": sum(row["pending_bytes"] for row in rows)}
    return total


def _md_table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    return lines


def markdown_audit_report(rows):
    ccmi = [row for row in rows if row["kind"] == "ccmi"]
    waccmx = [row for row in rows if row["kind"] != "ccmi"]
    excluded = [model for model in model_names()
                if model_metadata(model)["kind"] == "ccmi" and model_metadata(model)["status"] != "ready"]
    lines = ["# JuMACS download plan audit", "",
             "Complete-archive plans derived offline from `products/diagnostics/inspection/` "
             "(`jumacs download-audit`). No transfer has been executed.", ""]
    lines += ["## Per-model audit", ""]
    lines += _md_table(AUDIT_COLUMNS, rows)
    lines += ["", "## Totals", ""]
    for total in (_totals(ccmi, "CCMI only"), _totals(waccmx, "WACCM-X (separate)"), _totals(rows, "Grand total")):
        if not total["models"]:
            continue
        lines.append(f"- **{total['label']}**: {total['models']} sources, {total['inventory_files']} inventory files, "
                     f"{total['planned_files']} planned files (~{_gb(total['planned_bytes'])} GB planned), "
                     f"{total['already_complete_files']} already complete, "
                     f"{total['pending_files']} pending (~{_gb(total['pending_bytes'])} GB pending)")
    mismatches = [row for row in rows if row["audit_status"] != "ok"]
    lines += ["", "## Exceptions and warnings", ""]
    if mismatches:
        lines += [f"- **{row['model']}**: {row['notes']}" for row in mismatches]
    else:
        lines.append("- no audit mismatches: every inventoried file is planned or an understood exception "
                     "(superseded versions deduplicated to the latest release).")
    if excluded:
        lines += [f"- excluded (unavailable): {', '.join(excluded)} - no executable plan generated."]
    lines.append("")
    return "\n".join(lines)


def write_download_plan_summary(models=None, root=ROOT):
    models = tuple(models) if models else (*ready_ccmi_model_names(), "WACCM-X")
    rows = [audit_row(model) for model in models]
    comparison = root / COMPARISON_SUBDIR
    comparison.mkdir(parents=True, exist_ok=True)
    csv_path = comparison / "download_plan_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    md_path = comparison / "download_plan_summary.md"
    md_path.write_text(markdown_audit_report(rows))
    return [csv_path, md_path]
