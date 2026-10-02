"""Vertical coverage: observed extent of native-grid monthly zonal fields, and per-model readiness for a
WACCM-X extension of the common-grid climatology products."""
import csv
import json
from contextlib import nullcontext
import numpy as np
import xarray as xr
from .config import ROOT, load_config
from .netcdf import open_cftime_dataset
from .comparison import _pressure


def coverage_for_field(field, pressure, height=None):
    level = next((d for d in ("lev", "plev", "ilev") if d in field.dims), None)
    if level is None or pressure is None:
        return {"lowest_valid_pressure_pa": None, "highest_valid_pressure_pa": None,
                "approximate_altitude_min_km": None, "approximate_altitude_max_km": None,
                "median_upper_valid_altitude_km": None, "altitude_method": "unavailable",
                "vertical_levels": field.sizes.get(level, 0) if level else 0}
    valid = np.isfinite(field) & np.isfinite(pressure) & (pressure > 0)
    sampled = pressure.where(valid)
    lower = float(sampled.max(skipna=True))
    upper = float(sampled.min(skipna=True))
    if not np.isfinite(lower) or not np.isfinite(upper):
        lower = upper = None
    method = "7 km scale-height estimate; limited above ~80 km"
    altitude = lambda p: round(7.0 * np.log(101325.0 / p), 2) if p else None
    bottom_km, top_km = altitude(lower), altitude(upper)
    median_top_km = None
    if height is not None and set(height.dims) == set(field.dims) and all(height.sizes[d] == field.sizes[d] and height[d].equals(field[d]) for d in field.dims if d in field.coords and d in height.coords):
        valid_height = height.where(valid & np.isfinite(height))
        bottom_km = round(float(valid_height.min(skipna=True)) / 1000, 2)
        top_km = round(float(valid_height.max(skipna=True)) / 1000, 2)
        median_top_km = round(float(valid_height.max(level, skipna=True).median(skipna=True)) / 1000, 2)
        method = "native geopotential height"
    return {"lowest_valid_pressure_pa": lower, "highest_valid_pressure_pa": upper,
            "approximate_altitude_min_km": bottom_km, "approximate_altitude_max_km": top_km,
            "median_upper_valid_altitude_km": median_top_km,
            "altitude_method": method, "vertical_levels": field.sizes[level]}


def coverage_model(model):
    config = load_config(model)
    rows = []
    height_name = config["variables"].get("geopotential_height")
    height_path = ROOT / config["paths"]["zonal"] / f"{height_name}_monthly_zonal.nc" if height_name else None
    height_source = open_cftime_dataset(height_path) if height_path and height_path.exists() else nullcontext(None)
    with height_source as height_ds:
        height = height_ds[height_name] if height_ds is not None else None
        for path in sorted((ROOT / config["paths"]["zonal"]).glob("*_monthly_zonal.nc")):
            name = path.name.removesuffix("_monthly_zonal.nc")
            with open_cftime_dataset(path) as ds:
                row = {"model": model, "variable": name,
                       "canonical_species": next((k for k, v in config["variables"].items() if v == name), name),
                       "units": ds[name].attrs.get("units", ""),
                       "first_month": str(ds.time.values[0])[:7], "last_month": str(ds.time.values[-1])[:7]}
                row.update(coverage_for_field(ds[name], _pressure(ds), height))
                rows.append(row)
    dest = ROOT / "products/diagnostics" / "coverage" / model
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "vertical_coverage.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (dest / "vertical_coverage.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return dest / "vertical_coverage.csv"


def coverage_matrix():
    from .archive import TARGETS
    from .config import ready_model_names
    models = ready_model_names()
    reports = {}
    for model in models:
        path = ROOT / "products/diagnostics" / "coverage" / model / "vertical_coverage.json"
        reports[model] = {row["canonical_species"]: row for row in json.loads(path.read_text())} if path.exists() else {}
    names = list(dict.fromkeys(TARGETS + [name for report in reports.values() for name in report]))
    rows = []
    for name in names:
        row = {"species": name}
        for model in models:
            entry = reports[model].get(name, {})
            row[model + "_top_pa"] = entry.get("highest_valid_pressure_pa", "")
            row[model + "_bottom_pa"] = entry.get("lowest_valid_pressure_pa", "")
            row[model + "_top_km"] = entry.get("approximate_altitude_max_km", "")
            row[model + "_top_median_km"] = entry.get("median_upper_valid_altitude_km", "")
            row[model + "_altitude_method"] = entry.get("altitude_method", "")
            row[model + "_levels"] = entry.get("vertical_levels", "")
        rows.append(row)
    dest = ROOT / "products/comparison" / "vertical_coverage_matrix.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return dest


def coverage_plot():
    import matplotlib.pyplot as plt
    targets = ("temperature", "O3", "H2O", "CO2", "CH4", "N2O", "CO", "NO", "CFC-11", "CFC-12", "SF6")
    from .config import ready_model_names
    models = tuple(m for m in ready_model_names() if (ROOT / "products/diagnostics" / "coverage" / m / "vertical_coverage.json").exists())
    reports = {}
    for model in models:
        path = ROOT / "products/diagnostics" / "coverage" / model / "vertical_coverage.json"
        reports[model] = {r["canonical_species"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    import numpy as np
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    offsets = np.linspace(-.24, .24, len(models)) if len(models) > 1 else [0.0]
    for index, (model, offset) in enumerate(zip(models, offsets)):
        x, y = [], []
        for index_target, target in enumerate(targets):
            value = reports[model].get(target, {}).get("highest_valid_pressure_pa")
            if value is not None and value > 0:
                x.append(index_target + offset); y.append(value)
        ax.scatter(x, y, s=44, color=palette[index % len(palette)], label=model, zorder=3)
    ax.set(xticks=range(len(targets)), xticklabels=targets, yscale="log", ylabel="Lowest pressure with finite values (Pa)",
           title="JuMACS species-specific upper valid pressure")
    ax.invert_yaxis()
    ax.grid(axis="y", which="both", alpha=.2)
    ax.legend()
    ax.text(.01, -.17, "Finite-value extent only; trace concentrations can be negligible at the top level.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    dest = ROOT / "products/diagnostics" / "coverage" / "vertical_coverage.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    return dest


PRODUCT_MEAN_SUFFIX = "_mean"
APPROXIMATE_SCALE_HEIGHT_KM = 7.0
APPROXIMATE_SURFACE_PRESSURE_PA = 101325.0
EXTENSION_STATUSES = ("not_applicable", "mapping_unresolved", "no_waccmx_variable", "no_overlap",
                      "overlap_no_extension", "extension_candidate")


class CoverageMatrixProblem(ValueError):
    """A coverage matrix cannot be built from the products that were asked for."""


def product_path(model, start_year, end_year):
    """The single climatology product a model contributes for a period."""
    from .climatology import product_name
    return ROOT / load_config(model)["paths"]["climatology"] / product_name(model, start_year, end_year)


def product_models(spec, start_year, end_year):
    """Models to matrix, given names, slugs, 'all', or 'auto'; an explicit model must have a product."""
    from .config import model_metadata, model_names, ready_ccmi_model_names
    known = {model_metadata(name)["slug"]: name for name in model_names()}
    known.update({name.lower(): name for name in known.values()})
    if spec in (None, "", "all", "auto"):
        pool = list(model_names()) if spec != "auto" else [*ready_ccmi_model_names(), "WACCM-X"]
        selected = [name for name in pool if product_path(name, start_year, end_year).exists()]
        if not selected:
            raise CoverageMatrixProblem(
                f"no climatology products for {start_year}-{end_year}; run jumacs build or choose another period")
        return selected
    selected = []
    for entry in [part.strip() for part in str(spec).split(",") if part.strip()]:
        name = known.get(entry.lower())
        if name is None:
            raise CoverageMatrixProblem(f"unknown model {entry!r}; registered: {', '.join(sorted(model_names()))}")
        path = product_path(name, start_year, end_year)
        if not path.exists():
            raise CoverageMatrixProblem(f"{name} has no product at {path}; build it first")
        selected.append(name)
    return list(dict.fromkeys(selected))


def split_models(models):
    """The CCMI models to assess and the whole-atmosphere model they would be extended with."""
    from .config import is_waccmx
    donors = [model for model in models if is_waccmx(load_config(model))]
    if len(donors) != 1:
        raise CoverageMatrixProblem("extension coverage compares each CCMI model with WACCM-X, so a WACCM-X "
                                    "product is required: select it too or build it with jumacs build")
    return ccmi_models(models, donors[0]), donors[0]


def ccmi_models(models, donor):
    """The models to assess; the donor is the reference, never a row of its own."""
    remaining = [model for model in models if model != donor]
    if not remaining:
        raise CoverageMatrixProblem(f"extension coverage compares CCMI models with {donor}, so select at least one "
                                    "CCMI model alongside it")
    return remaining


def approximate_altitude_km(pressure):
    """Approximate altitude of a pressure level; crude above ~80 km, where a single scale height fails."""
    if pressure is None or pressure <= 0:
        return None
    return APPROXIMATE_SCALE_HEIGHT_KM * np.log(APPROXIMATE_SURFACE_PRESSURE_PA / pressure)


def product_field_coverage(ds, name, coordinate, sample_fraction):
    """Availability of one climatology field on the common grid; nothing is interpolated or filled.

    A pressure level is usable when at least ``sample_fraction`` of its months and latitudes are finite. The
    per-level fractions travel with the field, so coverage that is broken in the middle stays visible.
    """
    field = ds[name]
    finite = np.isfinite(np.asarray(field, float))
    fraction = round(float(finite.mean()), 6) if finite.size else 0.0
    if coordinate not in field.dims:
        return {"variable": name, "is_3d": False, "units": field.attrs.get("units", ""),
                "finite_fraction": fraction, "levels_any_finite": None,
                "levels_usable": None, "contiguous_usable": None, "max_any_pressure_pa": None,
                "min_any_pressure_pa": None, "max_usable_pressure_pa": None, "min_usable_pressure_pa": None,
                "level_finite_fractions": []}
    pressure = np.asarray(ds[coordinate], float)
    axis = list(field.dims).index(coordinate)
    per_level = finite.mean(axis=tuple(i for i in range(finite.ndim) if i != axis))
    present = per_level > 0.0
    usable = per_level >= sample_fraction
    reached = np.flatnonzero(usable)
    lowest = np.flatnonzero(present)
    return {"variable": name, "is_3d": True, "units": field.attrs.get("units", ""),
            "finite_fraction": fraction,
            "levels_any_finite": int(present.sum()), "levels_usable": int(usable.sum()),
            "contiguous_usable": None if reached.size == 0 else
            bool(reached[-1] - reached[0] + 1 == reached.size),
            "max_any_pressure_pa": float(pressure[lowest].max()) if lowest.size else None,
            "min_any_pressure_pa": float(pressure[lowest].min()) if lowest.size else None,
            "max_usable_pressure_pa": float(pressure[usable].max()) if reached.size else None,
            "min_usable_pressure_pa": float(pressure[usable].min()) if reached.size else None,
            "level_finite_fractions": [round(float(value), 6) for value in per_level]}


def read_product(path):
    """Open one climatology product; the coverage step only reads, and never writes, products."""
    return xr.open_dataset(path, decode_cf=False)


def product_inventory(model, start_year, end_year, sample_fraction):
    """Every mean field of one product, after the product is checked against the configured common grid."""
    from .cf import ProductProblem, assert_product
    from .config import vertical_grid
    grid = vertical_grid()
    path = product_path(model, start_year, end_year)
    with read_product(path) as ds:
        stems = sorted(name[: -len(PRODUCT_MEAN_SUFFIX)] for name in ds.data_vars
                       if name.endswith(PRODUCT_MEAN_SUFFIX))
        try:
            assert_product(ds, names=tuple(stems), grid=grid)
        except ProductProblem as error:
            raise CoverageMatrixProblem(f"{model}: {path.name} does not meet the product contract: {error}; if the "
                                        "product predates the single-file CF-1.13 layout, rebuild it with jumacs build") from error
        fields = {stem: product_field_coverage(ds, stem + PRODUCT_MEAN_SUFFIX, grid["coordinate"], sample_fraction)
                  for stem in stems}
    return {"path": str(path), "fields": fields}


def product_species_mapping(model):
    """Source variable name to canonical species, read from the model's own variables mapping."""
    return {source: canonical for canonical, source in load_config(model)["variables"].items()}


def usable_mask(entry, sample_fraction):
    """The boolean mask over the common grid of levels where one field is usable; None when it has no levels."""
    if entry is None or not entry["is_3d"]:
        return None
    return np.asarray(entry["level_finite_fractions"], float) >= sample_fraction


def model_variables(model, fields):
    """Canonical variables one model is asked about: those its configuration maps, plus anything it holds."""
    from .config import species_registry
    registry = [entry["canonical"] for entry in species_registry()]
    named = [product_species_mapping(model).get(stem, stem) for stem in fields]
    in_scope = set(load_config(model)["variables"]) | set(named)
    return ([canonical for canonical in registry if canonical in in_scope] +
            [name for name in dict.fromkeys(named) if name not in registry])


def _range_of(pressure, mask):
    """Top (lowest pressure) and bottom (highest pressure) of a usable mask, or two None values."""
    if mask is None or not bool(mask.any()):
        return None, None
    reached = pressure[mask]
    return float(reached.min()), float(reached.max())


def extension_row(model, variable, entry, donor_entry, donor_mapped, group, focus, pressure, sample_fraction,
                  level_fraction):
    """One CCMI model and variable against WACCM-X, decided from the level masks rather than endpoints."""
    ccmi_mask = usable_mask(entry, sample_fraction)
    donor_mask = usable_mask(donor_entry, sample_fraction)
    both = None if ccmi_mask is None or donor_mask is None else np.logical_and(ccmi_mask, donor_mask)
    ccmi_top, ccmi_bottom = _range_of(pressure, ccmi_mask)
    donor_top, donor_bottom = _range_of(pressure, donor_mask)
    overlap_top, overlap_bottom = _range_of(pressure, both)
    overlap_levels = 0 if both is None else int(both.sum())
    extends_upward = False
    if both is not None and both.any() and ccmi_top is not None:
        extends_upward = bool((donor_mask & (pressure < ccmi_top)).any())
    if entry is None or not entry["is_3d"]:
        status = "not_applicable"
    elif donor_entry is None:
        status = "no_waccmx_variable" if donor_mapped else "mapping_unresolved"
    elif not donor_entry["is_3d"]:
        status = "no_waccmx_variable"
    elif overlap_levels == 0:
        status = "no_overlap"
    else:
        status = "extension_candidate" if extends_upward else "overlap_no_extension"
    return {"model": model, "variable": variable, "group": group, "application_focus": variable in focus,
            "product_variable": entry["variable"] if entry else "", "units": entry["units"] if entry else "",
            "present": entry is not None, "is_3d": bool(entry["is_3d"]) if entry else False,
            "finite_fraction": entry["finite_fraction"] if entry else None,
            "levels_any_finite": entry["levels_any_finite"] if entry else None,
            "levels_usable": entry["levels_usable"] if entry else None,
            "broadly_usable": None if entry is None or not entry["is_3d"] else
            entry["levels_usable"] / pressure.size >= level_fraction,
            "contiguous_usable": entry["contiguous_usable"] if entry else None,
            "max_any_pressure_pa": entry["max_any_pressure_pa"] if entry else None,
            "min_any_pressure_pa": entry["min_any_pressure_pa"] if entry else None,
            "max_usable_pressure_pa": ccmi_bottom, "min_usable_pressure_pa": ccmi_top,
            "waccmx_available": donor_entry is not None and donor_entry["is_3d"],
            "waccmx_max_usable_pressure_pa": donor_bottom, "waccmx_min_usable_pressure_pa": donor_top,
            "overlap_exists": overlap_levels > 0, "overlap_level_count": overlap_levels,
            "overlap_max_pressure_pa": overlap_bottom, "overlap_min_pressure_pa": overlap_top,
            "waccmx_extends_upward": extends_upward, "extension_status": status,
            "level_finite_fractions": list(entry["level_finite_fractions"]) if entry else []}


def extension_rows(models, donor, inventories, sample_fraction, level_fraction):
    """One row per CCMI model and canonical variable; WACCM-X is the reference, never a row of its own."""
    from .config import coverage_settings, species_registry, vertical_grid
    registry = {entry["canonical"]: entry.get("group", "") for entry in species_registry()}
    pressure = np.asarray(vertical_grid()["levels"], float)
    focus = set(coverage_settings()["application_focus"])
    donor_fields = inventories[donor]["fields"]
    donor_map = load_config(donor)["variables"]
    rows = []
    for model in models:
        inverse = product_species_mapping(model)
        by_canonical = {canonical: source for source, canonical in inverse.items()}
        fields = inventories[model]["fields"]
        for variable in model_variables(model, fields):
            stem = by_canonical.get(variable, variable)
            donor_stem = donor_map.get(variable)
            rows.append(extension_row(model, variable, fields.get(stem),
                                      donor_fields.get(donor_stem) if donor_stem else None, donor_stem is not None,
                                      registry.get(variable, ""), focus, pressure, sample_fraction, level_fraction))
    return rows


def extension_matrix(models, donor, start_year, end_year, sample_fraction, level_fraction):
    """Read the products of the selected models and compare each with the whole-atmosphere donor."""
    inventories = {model: product_inventory(model, start_year, end_year, sample_fraction)
                   for model in [*models, donor]}
    return inventories, extension_rows(models, donor, inventories, sample_fraction, level_fraction)


def _hpa(pressure):
    return f"{pressure / 100.0:g} hPa" if pressure is not None else "—"


def _span(top, bottom):
    return "—" if top is None and bottom is None else f"{_hpa(top)} – {_hpa(bottom)}"


def _km(pressure):
    altitude = approximate_altitude_km(pressure)
    return f"~{altitude:.0f} km" if altitude is not None else "—"


def _cell(value):
    return "" if value is None else value


EXTENSION_COLUMNS = ("model", "variable", "group", "application_focus", "product_variable", "units", "present",
                     "is_3d", "finite_fraction", "levels_any_finite", "levels_usable", "broadly_usable",
                     "contiguous_usable", "max_any_pressure_pa", "min_any_pressure_pa", "max_usable_pressure_pa",
                     "min_usable_pressure_pa", "waccmx_available", "waccmx_max_usable_pressure_pa",
                     "waccmx_min_usable_pressure_pa", "overlap_exists", "overlap_level_count",
                     "overlap_max_pressure_pa", "overlap_min_pressure_pa", "waccmx_extends_upward",
                     "extension_status")


def extension_csv(path, rows):
    """One row per CCMI model and canonical variable, pressures in Pa."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, EXTENSION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell(row[column]) for column in EXTENSION_COLUMNS})
    return path


def extension_markdown(path, models, donor, rows, sample_fraction, level_fraction, period):
    """Human summary of extension readiness. Pressures are hPa here; the CSV and JSON keep Pa."""
    candidates = [row for row in rows if row["extension_status"] == "extension_candidate"]
    focus_rows = [row for row in rows if row["application_focus"]]
    missing = sorted({row["variable"] for row in rows if row["extension_status"] == "no_waccmx_variable"})
    unmapped = sorted({row["variable"] for row in rows if row["extension_status"] == "mapping_unresolved"})
    flat = sorted({row["variable"] for row in rows if row["present"] and not row["is_3d"]})
    header = ("| Model | Variable | CCMI usable | WACCM-X usable | Overlap | Extends upward | Status |",
              "| --- | --- | --- | --- | --- | --- | --- |")

    def line(row):
        return (f"| {row['model']} | {row['variable']} | "
                f"{_span(row['min_usable_pressure_pa'], row['max_usable_pressure_pa'])} | "
                f"{_span(row['waccmx_min_usable_pressure_pa'], row['waccmx_max_usable_pressure_pa'])} | "
                f"{_span(row['overlap_min_pressure_pa'], row['overlap_max_pressure_pa'])} | "
                f"{'yes' if row['waccmx_extends_upward'] else 'no'} | {row['extension_status']} |")

    lines = [f"# JuMACS WACCM-X extension coverage {period[0]}-{period[1]}", "",
              "For every CCMI model and variable: where that model has usable values, where WACCM-X has usable "
              "values, and whether the two overlap. Diagnostics only: no blending, no filling, no ranking, and "
              "no change to products or interpolation.", "",
              f"- Products: {', '.join(models)}, compared with {donor}.",
              f"- A pressure level is usable when at least {sample_fraction:g} of its months and latitudes are "
              f"finite; *any finite* is reported separately and is weaker.",
              "- Overlap is the level-by-level AND of the two usable masks, so coverage broken in the middle "
              "stays broken instead of being bridged by endpoint ranges.",
              "- Extends upward means WACCM-X is usable at lower pressure, that is higher altitude, than the top "
              "of the CCMI usable range.",
              f"- *Broadly usable* (CSV and JSON) only labels a field finite on at least {level_fraction:g} of the "
              "levels; it plays no part in the overlap or extension decision.",
              "- Status: not_applicable (absent or two-dimensional), mapping_unresolved (no configured WACCM-X "
              "counterpart), no_waccmx_variable (mapped but absent from the product), no_overlap, "
              "overlap_no_extension, extension_candidate.",
              "- hPa here for readability; the CSV and JSON keep Pa. Altitudes from a single 7 km scale height "
              "are indicative only, and poor above about 80 km.", ""]
    lines += [f"## Extension candidates ({len(candidates)})", "",
              "WACCM-X overlaps the model and reaches higher than it; nothing is joined yet.", ""]
    lines += list(header) + ([line(row) for row in candidates] if candidates else ["None."])
    lines += ["", f"## Application variables ({len(focus_rows)})", "",
              "The configured `coverage.application_focus` species for every CCMI product in scope.", ""]
    lines += list(header) + ([line(row) for row in focus_rows] if focus_rows else ["None."])
    lines += ["", "## Not extendable in WACCM-X", ""]
    lines += [f"- No configured WACCM-X counterpart: {', '.join(unmapped) if unmapped else 'none'}.",
              f"- Mapped but absent from the WACCM-X product: {', '.join(missing) if missing else 'none'}.",
              f"- Two-dimensional, so no vertical extension applies: {', '.join(flat) if flat else 'none'}."]
    if candidates:
        lines += ["", "## Approximate altitude reached above each model", "",
                  "Altitudes come from a single 7 km scale height and are a poor guide above about 80 km; the "
                  "pressure levels in the CSV and JSON are the authoritative statement.", "",
                  "| Model | Variable | CCMI top | WACCM-X top | Approximate WACCM-X top |",
                  "| --- | --- | --- | --- | --- |"]
        lines += [f"| {row['model']} | {row['variable']} | {_hpa(row['min_usable_pressure_pa'])} | "
                  f"{_hpa(row['waccmx_min_usable_pressure_pa'])} | "
                  f"{_km(row['waccmx_min_usable_pressure_pa'])} |" for row in candidates]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def extension_json(path, models, donor, inventories, rows, sample_fraction, level_fraction, period):
    """Full record: every row, its per-level usable evidence, and the WACCM-X masks beside it."""
    from .config import vertical_grid
    grid = vertical_grid()
    donor_fields = inventories[donor]["fields"]
    donor_map = load_config(donor)["variables"]
    document = {
        "description": "Where each CCMI model has usable values on the common grid, and where WACCM-X could "
                       "extend it upward.",
        "note": "Diagnostics only: no blending, no filling, no ranking, and no change to products or interpolation.",
        "period": {"start_year": period[0], "end_year": period[1]},
        "ccmi_models": models,
        "donor_model": donor,
        "usable_sample_fraction": sample_fraction,
        "usable_level_fraction": level_fraction,
        "extension_statuses": list(EXTENSION_STATUSES),
        "definitions": {
            "usable_level": f"a pressure level where at least {sample_fraction:g} of the months and latitudes are "
                            "finite",
            "broadly_usable": f"a field usable on at least {level_fraction:g} of the levels; a descriptive label "
                              "only, not a condition for overlap or extension",
            "min_usable_pressure_pa": "smallest pressure on a usable level: the top of the usable range",
            "max_usable_pressure_pa": "largest pressure on a usable level: the bottom of the usable range",
            "min_any_pressure_pa": "smallest pressure with any finite value: the top of the raw extent",
            "contiguous_usable": "the usable levels form one unbroken block; null when there are none",
            "overlap": "levels usable in both the CCMI model and WACCM-X, combined level by level",
            "waccmx_extends_upward": "WACCM-X is usable at lower pressure, that is higher altitude, than the top "
                                     "of the CCMI usable range",
            "not_applicable": "the CCMI model lacks the variable or the field is two-dimensional",
            "mapping_unresolved": "no WACCM-X counterpart is configured, and none is guessed",
            "no_waccmx_variable": "the WACCM-X counterpart is configured but the product has no such field",
            "no_overlap": "both sources have fields but no usable level in common",
            "overlap_no_extension": "they overlap and WACCM-X reaches no higher",
            "extension_candidate": "they overlap and WACCM-X reaches to lower pressure",
            "level_finite_fractions": "finite fraction of months and latitudes per level, in grid order",
            "approximate_altitude": "7 km scale-height estimate, limited above about 80 km"},
        "pressure_coordinate": {"name": grid["coordinate"], "units": grid["units"], "level_count": len(grid["levels"]),
                                "levels_pa": list(grid["levels"])},
        "products": {model: inventories[model]["path"] for model in inventories},
        "donor_fields": {variable: donor_fields[donor_map[variable]] for variable in donor_map
                         if donor_map[variable] in donor_fields},
        "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2))
    return path


def coverage_matrix_command(model="all", usable_level_fraction=None, usable_sample_fraction=None,
                            start_year=None, end_year=None):
    """Build the per-model WACCM-X extension coverage from existing products: CSV, Markdown, and JSON."""
    from .config import coverage_settings, reference_period, vertical_grid
    reference = reference_period()["reference_period"]
    start_year = start_year or reference["start_year"]
    end_year = end_year or reference["end_year"]
    settings = coverage_settings()
    for key, override in (("usable_level_fraction", usable_level_fraction),
                          ("usable_sample_fraction", usable_sample_fraction)):
        if override is None:
            continue
        value = float(override)
        if not 0.0 < value <= 1.0:
            raise CoverageMatrixProblem(f"coverage.{key} must be in (0, 1]")
        settings[key] = value
    sample_fraction, level_fraction = settings["usable_sample_fraction"], settings["usable_level_fraction"]
    models, donor = split_models(product_models(model, start_year, end_year))
    inventories, rows = extension_matrix(models, donor, start_year, end_year, sample_fraction, level_fraction)
    directory = ROOT / "products/comparison"
    csv_path = extension_csv(directory / "coverage_matrix.csv", rows)
    markdown_path = extension_markdown(directory / "coverage_matrix.md", models, donor, rows, sample_fraction,
                                      level_fraction, (start_year, end_year))
    json_path = extension_json(directory / "coverage_matrix.json", models, donor, inventories, rows,
                               sample_fraction, level_fraction, (start_year, end_year))
    counts = {status: sum(1 for row in rows if row["extension_status"] == status) for status in EXTENSION_STATUSES}
    candidates = [f"{row['model']} {row['variable']}" for row in rows
                  if row["extension_status"] == "extension_candidate"]
    return "\n".join([
        f"extension coverage for {start_year}-{end_year}: {len(models)} CCMI product(s) compared with {donor}",
        f"a level is usable when at least {sample_fraction:g} of its months and latitudes are finite, of "
        f"{len(vertical_grid()['levels'])} levels",
        f"{len(rows)} model-variable pairs: " + ", ".join(f"{status}={counts[status]}" for status in EXTENSION_STATUSES),
        "extension candidates: " + (", ".join(candidates) if candidates else "none"),
        f"wrote {csv_path}, {markdown_path}, {json_path}"])
