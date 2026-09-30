"""Offline derived summaries of the inspected CEDA inventories (no network access)."""
import csv
import json

from .config import ROOT, registry, load_config, model_metadata, species_registry

INSPECTION_SUBDIR = "products/diagnostics/inspection"
COMPARISON_SUBDIR = "products/comparison"

MODEL_COLUMNS = ["model", "kind", "status", "dataset_uuid", "file_count", "first_month", "last_month",
                 "amon_variable_count", "amonz_variable_count", "native_variable_count",
                 "mapped_species_count", "amonz_only_variable_count"]
SPECIES_COLUMNS = ["species", "group", "models_available", "models", "amonz_only_models", "waccmx_available"]


def _inventory(root, model):
    path = root / INSPECTION_SUBDIR / model / "archive_inventory.json"
    return json.loads(path.read_text()) if path.exists() else None


def _natives(inventory):
    if "dimensions" in inventory:
        return set(inventory["dimensions"])
    return {entry["variable"] for entry in inventory.get("files", ())}


def _families(inventory):
    families = {}
    for entry in inventory.get("files", ()):
        families.setdefault(entry["family"], set()).add(entry["variable"])
    return families


def _usable(inventory):
    return inventory is not None and inventory.get("status") != "unresolved"


def model_summary_rows(root=ROOT):
    rows = []
    for model in registry():
        config = load_config(model)
        meta = model_metadata(model)
        inventory = _inventory(root, model)
        status = meta["status"] if meta["status"] != "ready" else ("ready" if _usable(inventory) else "not_inspected")
        if inventory is not None and inventory.get("status") == "unresolved":
            status = "unresolved"
        row = {"model": model, "kind": meta["kind"], "status": status,
               "dataset_uuid": config["model"].get("dataset_uuid", "")}
        for column in MODEL_COLUMNS[4:]:
            row[column] = ""
        if not _usable(inventory):
            rows.append(row)
            continue
        natives = _natives(inventory)
        families = _families(inventory)
        monthly = "Amon" in families or "AmonZ" in families
        amon, amonz = families.get("Amon", set()), families.get("AmonZ", set())
        files = inventory.get("files", ())
        native_to_canonical = {native: canonical for canonical, native in config["variables"].items()}
        mapped = {canonical for native, canonical in native_to_canonical.items() if native in natives}
        species_targets = {entry["canonical"] for entry in species_registry()}
        row.update({"file_count": len(files) if files else "",
                    "first_month": min((entry["start"] for entry in files), default=""),
                    "last_month": max((entry["end"] for entry in files), default=""),
                    "amon_variable_count": len(amon) if monthly else "",
                    "amonz_variable_count": len(amonz) if monthly else "",
                    "native_variable_count": len(natives),
                    "mapped_species_count": len(mapped & species_targets),
                    "amonz_only_variable_count": len(amonz - amon) if monthly else ""})
        rows.append(row)
    return rows


def species_summary_rows(root=ROOT):
    models = {model: model_metadata(model) for model in registry()}
    prepared = {}
    for model, meta in models.items():
        inventory = _inventory(root, model)
        if meta["kind"] != "ccmi" or meta["status"] != "ready" or not _usable(inventory):
            continue
        families = _families(inventory)
        natives = _natives(inventory)
        prepared[model] = (natives, families.get("AmonZ", set()), families.get("Amon", set()),
                           load_config(model)["variables"])
    waccmx = next((model for model, meta in models.items() if meta["kind"] == "whole_atmosphere"), None)
    waccmx_inventory = _inventory(root, waccmx) if waccmx else None
    waccmx_natives = _natives(waccmx_inventory) if _usable(waccmx_inventory) else set()
    waccmx_variables = load_config(waccmx)["variables"] if waccmx else {}
    rows = []
    for entry in species_registry():
        canonical, group = entry["canonical"], entry.get("group", "")
        available = [model for model, (natives, _, _, variables) in sorted(prepared.items())
                     if variables.get(canonical, "") and variables.get(canonical) in natives]
        amonz_only = [model for model in available
                      if prepared[model][2] and prepared[model][1] and prepared[model][3].get(canonical)
                      not in prepared[model][2]]
        rows.append({"species": canonical, "group": group, "models_available": len(available),
                     "models": ";".join(available), "amonz_only_models": len(amonz_only),
                     "waccmx_available": "yes" if waccmx_variables.get(canonical, "") and waccmx_variables.get(canonical) in waccmx_natives else "no"})
    return rows


def _md_table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(value) for value in (row[c] for c in columns)) + " |" for row in rows]
    return lines


def _richness_section(root, model_rows):
    groups = {}
    for entry in species_registry():
        groups.setdefault(entry.get("group", ""), []).append(entry["canonical"])
    species_rows = species_summary_rows(root)
    by_species = {row["species"]: row for row in species_rows}
    ccmi_models = [row["model"] for row in model_rows if row["kind"] == "ccmi" and row["status"] == "ready"]
    header = ["group", "species_tracked"] + ccmi_models + ["WACCM-X"]
    lines = []
    for group in sorted(groups):
        canonicals = groups[group]
        cells = [group, len(canonicals)]
        for model in ccmi_models:
            cells.append(sum(1 for c in canonicals if model in by_species[c]["models"].split(";")))
        cells.append(sum(1 for c in canonicals if by_species[c]["waccmx_available"] == "yes"))
        lines.append(cells)
    return lines, header


def markdown_report(root=ROOT):
    model_rows = model_summary_rows(root)
    species_rows = species_summary_rows(root)
    ccmi_rows = [row for row in model_rows if row["kind"] == "ccmi"]
    waccmx_rows = [row for row in model_rows if row["kind"] != "ccmi"]
    inspected = [row for row in model_rows if row["status"] == "ready"]
    ready_ccmi = [row for row in ccmi_rows if row["status"] == "ready"]
    lines = ["# JuMACS archive summary", "",
             "Derived offline from the inventories in `products/diagnostics/inspection/` (`jumacs summary`).",
             "Cly, Bry and NOy are family sums and never substitute for individual species.", ""]
    lines += ["## Archive overview", ""]
    lines += _md_table(MODEL_COLUMNS, model_rows)
    lines += ["", "## Species coverage", "",
              f"{len(species_rows)} canonical species are tracked across {len(ready_ccmi)} inspected CCMI-ready models.", ""]
    full = [row["species"] for row in species_rows if ready_ccmi and row["models_available"] == len(ready_ccmi)]
    none = [row["species"] for row in species_rows if row["models_available"] == 0]
    partial = [row for row in species_rows if 0 < row["models_available"] < len(ready_ccmi)]
    lines += [f"- present in all inspected CCMI models: {', '.join(full) if full else 'none'}",
              f"- present in no inspected CCMI model: {', '.join(none) if none else 'none'}",
              f"- partial coverage: {len(partial)} species (see `species_summary.csv`)", ""]
    lines += ["## Chemistry richness by species group", ""]
    richness, header = _richness_section(root, model_rows)
    lines += _md_table(header, [dict(zip(header, cells)) for cells in richness])
    lines += ["", "## Amon versus AmonZ", ""]
    amonz_rows = [row for row in inspected if row["amonz_only_variable_count"] != ""]
    lines += _md_table(["model", "amon_variable_count", "amonz_variable_count", "amonz_only_variable_count"], amonz_rows)
    amonz_only_species = [row["species"] for row in species_rows if row["models_available"] and row["amonz_only_models"] == row["models_available"]]
    lines += ["", f"Canonical species available only in AmonZ (no Amon instance anywhere): "
                  f"{', '.join(amonz_only_species) if amonz_only_species else 'none'}", ""]
    lines += ["## Gaps and unresolved items", ""]
    for row in model_rows:
        if row["status"] == "unresolved":
            lines.append(f"- **{row['model']}**: archive identifiers unresolved; see the evidence comment in `config/models/{row['model']}.yaml`.")
        elif row["status"] == "not_inspected":
            lines.append(f"- **{row['model']}**: registered but not inspected yet (`jumacs inspect --model {row['model']}`).")
    if none:
        lines.append(f"- species not found in any inspected CCMI model: {', '.join(none)}.")
    lines += ["", "## WACCM-X (whole atmosphere, kept separate)", ""]
    for row in waccmx_rows:
        if row["status"] == "ready":
            lines.append(f"- {row['model']}: monthly zonal-mean multivariable files, {row['file_count']} files, "
                         f"{row['first_month']}-{row['last_month']}, {row['native_variable_count']} native variables, "
                         f"{row['mapped_species_count']} mapped canonical species. WACCM-X is excluded from the CCMI "
                         "coverage counts above and is only flagged per species via `waccmx_available`.")
        else:
            lines.append(f"- {row['model']}: status {row['status']}.")
    lines.append("")
    return "\n".join(lines)


def write_summaries(root=ROOT):
    comparison = root / COMPARISON_SUBDIR
    comparison.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, columns, rows in (("model_summary.csv", MODEL_COLUMNS, model_summary_rows(root)),
                                ("species_summary.csv", SPECIES_COLUMNS, species_summary_rows(root))):
        path = comparison / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    report = comparison / "summary.md"
    report.write_text(markdown_report(root))
    paths.append(report)
    return paths
