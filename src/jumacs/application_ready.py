"""Application-ready JuMACS product: complete the compact EMAC/WACCM-X climatology.

Vertical sources are used in strict priority EMAC -> WACCM-X -> IG2 v3.1 -> constant
edge fallback.  The combined compact product already merges EMAC and WACCM-X, so this
stage only fills the still-missing upper tail with IG2 through a short raised-cosine
transition, then carries the last valid value upward and downward to remove edge gaps.
"""

import json
import math
from pathlib import Path

import numpy as np
import xarray as xr

from . import config, ig2

DEFAULT_TRANSITION_WIDTH_KM = 8.0


def _default_paths(model, start_year, end_year, latitude_step, altitude_step):
    period = f"{start_year}-{end_year}"
    grid = f"{latitude_step}deg_{altitude_step}km"
    combined = (config.ROOT / "products/climatology/combined"
                / f"jumacs_{model.lower()}_waccmx_{period}_{grid}.nc")
    ig2_dir = config.ROOT / "data/raw/IG2/v3.1"
    output = (config.ROOT / "products/application-ready"
              / f"jumacs_{model.lower()}_waccmx_ig2_{period}_{grid}.nc")
    return combined, ig2_dir, output


def _raised_cosine_weight(altitudes, top_altitude, width):
    """Weight rising 0 -> 1 from ``top_altitude - width`` to ``top_altitude``."""
    if width <= 0:
        raise ValueError("transition width must be positive")
    phase = np.clip((altitudes - (top_altitude - width)) / width, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * phase)


def _blend(existing, donor, weight, positive):
    """Log-space blend for positive values, ordinary linear blend otherwise."""
    if positive and existing > 0.0 and donor > 0.0:
        return math.exp((1.0 - weight) * math.log(existing) + weight * math.log(donor))
    return (1.0 - weight) * existing + weight * donor


def _edge_and_transition(field, altitude, donor_column, donor_spec, width):
    """Return the completed field plus per-variable cell counts (never touches internal gaps)."""
    months, nalt, nlat = field.shape
    out = field.copy()
    counts = {"ig2_upper_cells": 0, "transition_blend_cells": 0,
              "constant_upper_cells": 0, "constant_lower_cells": 0,
              "internal_nan_cells": 0, "fully_missing_columns": 0}
    positive = bool(donor_spec["positive"]) if donor_spec else False
    donor_top = donor_spec["donor_top_km"] if donor_spec else 0.0
    for month in range(months):
        for j in range(nlat):
            column = out[month, :, j]
            finite = np.isfinite(column)
            if not finite.any():
                counts["fully_missing_columns"] += 1
                continue
            first = int(np.argmax(finite))
            last = nalt - 1 - int(np.argmax(finite[::-1]))
            counts["internal_nan_cells"] += int(np.isnan(column[first + 1:last]).sum())
            if first > 0:
                out[month, :first, j] = column[first]
                counts["constant_lower_cells"] += first
            if donor_column is not None:
                reference = donor_column[month, :, j]
                anchor_altitude = altitude[last]
                anchor_value = float(column[last])
                band_top = anchor_altitude + width
                for k in range(last + 1, nalt):
                    if altitude[k] > donor_top:
                        break
                    donor = reference[k]
                    if not np.isfinite(donor) or donor <= 0.0:
                        break
                    if altitude[k] < band_top and anchor_value > 0.0:
                        weight = float(_raised_cosine_weight(altitude[k:k + 1], band_top, width)[0])
                        out[month, k, j] = _blend(anchor_value, donor, weight, positive)
                        counts["transition_blend_cells"] += 1
                    else:
                        out[month, k, j] = donor
                        counts["ig2_upper_cells"] += 1
            filled = np.isfinite(out[month, :, j])
            last_filled = nalt - 1 - int(np.argmax(filled[::-1]))
            if last_filled < nalt - 1:
                out[month, last_filled + 1:, j] = out[month, last_filled, j]
                counts["constant_upper_cells"] += (nalt - 1 - last_filled)
    return out, counts


def _seam_log_step(original, completed, donor, altitude, last, donor_top):
    """Adjacent-level log10 steps around and above the donor seam for smoothness checks."""
    top_index = int(np.searchsorted(altitude, donor_top, side="right") - 1)
    steps = []
    for k in range(max(last, 0), min(len(altitude) - 1, top_index)):
        a, b = completed[k], completed[k + 1]
        if a > 0 and b > 0:
            steps.append(abs(math.log10(b) - math.log10(a)))
    hard_seam = None
    if last + 1 <= top_index:
        base = original[last]
        top = donor[last + 1]
        if base and base > 0 and top and top > 0:
            hard_seam = abs(math.log10(top) - math.log10(base))
    return (max(steps) if steps else 0.0), hard_seam


def _annual_metrics(series):
    series = np.asarray(series, dtype=float)
    if np.any(series <= 0):
        positive = series[series > 0]
        if positive.size < 2:
            return {"max_monthly_log_step": None, "dec_jan_log_step": None}
    logs = np.log10(np.where(series > 0, series, np.nan))
    steps = np.abs(np.diff(np.concatenate([logs, logs[:1]])))
    steps = steps[np.isfinite(steps)]
    dec_jan = abs(logs[-1] - logs[0]) if np.isfinite(logs[-1]) and np.isfinite(logs[0]) else None
    return {"max_monthly_log_step": float(steps.max()) if steps.size else None,
            "dec_jan_log_step": float(dec_jan) if dec_jan is not None else None}


def _latitude_metrics(profile):
    profile = np.asarray(profile, dtype=float)
    if np.any(profile <= 0):
        return {"max_lat_log_step": None, "step_at_regime_knot": None}
    logs = np.log10(profile)
    steps = np.abs(np.diff(logs))
    return {"max_lat_log_step": float(steps.max()) if steps.size else None}


def _smoothness(donor_columns, original_data, completed_data, altitude, latitudes, months):
    representative = {"equator": int(np.argmin(np.abs(latitudes))),
                      "midlatitude": int(np.argmin(np.abs(np.abs(latitudes) - 45.0))),
                      "polar": int(np.argmin(np.abs(np.abs(latitudes) - 75.0)))}
    july = int(np.argmin(np.abs(np.asarray(months) - 7)))
    alt_80 = int(np.argmin(np.abs(altitude - 80.0)))
    report = {}
    for key, spec in ig2.IG2_DONORS.items():
        if key not in completed_data:
            continue
        original = original_data[key]
        completed = completed_data[key]
        column = donor_columns[key]
        vertical = {}
        for label, j in representative.items():
            finite = np.isfinite(original[july, :, j])
            last = int(len(altitude) - 1 - np.argmax(finite[::-1])) if finite.any() else 0
            max_step, hard_seam = _seam_log_step(original[july, :, j], completed[july, :, j],
                                                 column[july, :, j], altitude, last, spec["donor_top_km"])
            vertical[label] = {"blend_top_km": float(altitude[last]), "seam_log_step_after_blend": round(max_step, 4),
                               "hard_graft_seam_log_step": round(hard_seam, 4) if hard_seam else None}
        latitude_slice = completed[july, alt_80, :]
        seasonal = column[:, alt_80, representative["polar"]]
        report[key] = {
            "ig2_species": spec["ig2_species"],
            "vertical_transition": vertical,
            "latitude_at_80km_july": _latitude_metrics(latitude_slice),
            "annual_cycle_75N_80km": _annual_metrics(seasonal),
        }
    return report


def _make_plots(directory, original_data, completed_data, donor_columns, altitude, latitudes, months, donors_used):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    equator = int(np.argmin(np.abs(latitudes)))
    polar = int(np.argmin(np.abs(np.abs(latitudes) - 75.0)))
    july = int(np.argmin(np.abs(np.asarray(months) - 7)))

    for key in donors_used:
        fig, axes = plt.subplots(1, 2, figsize=(9, 7), sharey=True, squeeze=False)
        for ax, label, j in ((axes[0, 0], "equator", equator), (axes[0, 1], "75N", polar)):
            ax.plot(original_data[key][july, :, j], altitude, label="existing EMAC/WACCM-X", lw=1.8)
            ax.plot(donor_columns[key][july, :, j], altitude, label="interpolated IG2", ls="--", lw=1.2)
            ax.plot(completed_data[key][july, :, j], altitude, label="final blend", lw=1.8)
            ax.set_xscale("log")
            ax.set_ylim(0, 120)
            ax.set_ylabel("altitude (km)")
            ax.set_title(f"{key} @ July, {label}")
            ax.grid(True, alpha=0.3)
        axes[0, 0].set_xlabel("mixing ratio (mol/mol)")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(directory / f"profile_{key}.png", dpi=130)
        plt.close(fig)

    if donors_used:
        key = donors_used[0]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for alt in (70, 80, 90):
            index = int(np.argmin(np.abs(altitude - alt)))
            axes[0].plot(latitudes, completed_data[key][july, index, :], marker="o", ms=3, lw=1, label=f"{alt} km")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("latitude")
        axes[0].set_ylabel(f"{key} (mol/mol)")
        axes[0].set_title(f"{key} latitude cross-section, July")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        for label, j in (("equator", equator), ("75N", polar)):
            axes[1].plot(np.asarray(months), donor_columns[key][:, int(np.argmin(np.abs(altitude - 80.0))), j],
                         marker="o", label=label)
        axes[1].set_yscale("log")
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xlabel("month")
        axes[1].set_ylabel(f"{key} (mol/mol)")
        axes[1].set_title(f"{key} annual cycle, IG2 80 km")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / "regime_interpolation.png", dpi=130)
        plt.close(fig)
    return directory


def build_application_ready(model="EMAC", start_year=None, end_year=None,
                            latitude_step=5, altitude_step=1, combined_path=None,
                            ig2_dir=None, output_path=None,
                            transition_width_km=DEFAULT_TRANSITION_WIDTH_KM, plots_dir=None):
    reference = config.reference_period()["reference_period"]
    start_year = start_year or reference["start_year"]
    end_year = end_year or reference["end_year"]
    default_combined, default_ig2, default_output = _default_paths(
        model, start_year, end_year, latitude_step, altitude_step)
    combined_path = Path(combined_path) if combined_path else default_combined
    ig2_dir = Path(ig2_dir) if ig2_dir else default_ig2
    output_path = Path(output_path) if output_path else default_output
    if not combined_path.exists():
        raise FileNotFoundError(f"compact product not found: {combined_path}")

    with xr.open_dataset(combined_path) as source:
        base = source.load()
    altitude = base.altitude.values.astype(float)
    latitudes = base.lat.values.astype(float)
    months = base.month.values
    width = float(transition_width_km)

    regime_data = ig2.load_regimes(ig2_dir)
    if regime_data["nalt"] != altitude.size or not np.allclose(regime_data["altitude_km"], altitude):
        raise ValueError("IG2 altitude grid does not match the compact product grid")

    donors_used = [key for key in ig2.IG2_DONORS if key in base.data_vars]
    donor_columns = {key: ig2.donor_column(regime_data, ig2.IG2_DONORS[key]["ig2_species"], months, latitudes)
                     for key in donors_used}

    original_data = {key: base[key].values.astype(float) for key in base.data_vars}
    completed_data = {}
    per_variable = {}
    data_vars = {}
    totals = {"ig2_upper_cells": 0, "transition_blend_cells": 0, "constant_upper_cells": 0,
              "constant_lower_cells": 0, "internal_nan_cells": 0, "fully_missing_columns": 0}
    for key in base.data_vars:
        spec = ig2.IG2_DONORS.get(key)
        column = donor_columns.get(key)
        completed, counts = _edge_and_transition(original_data[key], altitude, column, spec, width)
        completed_data[key] = completed
        for name, value in counts.items():
            totals[name] += value
        attrs = dict(base[key].attrs)
        blended = counts["ig2_upper_cells"] + counts["transition_blend_cells"]
        attrs["ig2_completed"] = "true" if blended else "false"
        if spec:
            attrs["ig2_species"] = spec["ig2_species"]
            attrs["ig2_donor_top_km"] = spec["donor_top_km"]
        attrs["completion_transition_width_km"] = width
        attrs["cells_filled_ig2_upper"] = counts["ig2_upper_cells"]
        attrs["cells_blended_ig2_transition"] = counts["transition_blend_cells"]
        attrs["cells_filled_constant_upper"] = counts["constant_upper_cells"]
        attrs["cells_filled_constant_lower"] = counts["constant_lower_cells"]
        attrs["internal_nan_cells"] = counts["internal_nan_cells"]
        attrs["fully_missing_columns"] = counts["fully_missing_columns"]
        attrs["completion_method"] = ("EMAC/WACCM-X -> IG2 raised-cosine transition -> IG2 upper "
                                      "-> constant edge fill" if spec else
                                      "EMAC/WACCM-X -> constant upper/lower edge fill")
        data_vars[key] = (("month", "altitude", "lat"), completed.astype("float32"), attrs)
        per_variable[key] = dict(source_status=base[key].attrs.get("source_status"),
                                 ig2_species=spec["ig2_species"] if spec else None,
                                 **counts)

    attributes = dict(base.attrs)
    attributes.update({
        "product_type": "application_ready_vertical_completion",
        "base_product": combined_path.name,
        "vertical_completion": "EMAC -> WACCM-X -> IG2 v3.1 -> constant edge fallback",
        "ig2_role": "upper-tail completion where EMAC and WACCM-X provide no data",
        "ig2_version": ig2.VERSION,
        "ig2_date": ig2.DATE,
        "ig2_regimes": ", ".join(ig2.STANDARD_PROFILES),
        "ig2_species_mapped": ", ".join(f"{key}:{ig2.IG2_DONORS[key]['ig2_species']}" for key in donors_used),
        "ig2_latitude_interpolation": "piecewise-linear in |latitude| over 0/45/75 regime knots, clamped beyond 75",
        "ig2_southern_hemisphere": "mirror of northern-hemisphere regimes by |latitude| (IG2 provides NH regimes only)",
        "ig2_seasonal_interpolation": ("single-harmonic polar summer/winter blend anchored at January; "
                                       "tropical and midlatitude archetypes carry no IG2 seasonality"),
        "vertical_transition_width_km": width,
        "vertical_transition_method": "raised cosine from (seam - width) to seam; existing below, IG2 above",
        "vertical_blend_space": "logarithmic for strictly positive trace-gas donors, linear otherwise",
        "donor_log_space_species": ", ".join(k for k in donors_used if ig2.IG2_DONORS[k]["positive"]),
        "donor_linear_species": ", ".join(k for k in donors_used if not ig2.IG2_DONORS[k]["positive"]),
        "lower_boundary_completion": "nearest valid value carried downward (includes the sub-surface lowest level)",
        "upper_boundary_completion": "IG2 donor where available, otherwise last valid value carried upward",
        "internal_gaps_filled": "false",
        "history": base.attrs.get("history", "") + "; IG2 v3.1 upper completion and constant edge fill",
    })

    result = xr.Dataset(data_vars, coords={"month": base.month.values, "altitude": base.altitude.values,
                                           "lat": base.lat.values})
    result.altitude.attrs = base.altitude.attrs
    result.lat.attrs = base.lat.attrs
    result.month.attrs = base.month.attrs
    result.attrs = attributes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {key: {"dtype": "float32", "zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
                for key in result.data_vars}
    temp = output_path.with_suffix(".nc.tmp")
    result.to_netcdf(temp, engine="netcdf4", encoding=encoding)
    temp.replace(output_path)

    smoothness = _smoothness(donor_columns, original_data, completed_data, altitude, latitudes, months)
    flagged = [key for key, metrics in smoothness.items()
               if any(v["seam_log_step_after_blend"] and v["seam_log_step_after_blend"] > 0.35
                      for v in metrics["vertical_transition"].values())]
    report = {
        "product": str(output_path),
        "base_product": str(combined_path),
        "period": f"{start_year}-{end_year}",
        "grid": {"month": len(months), "altitude_km": float(altitude[-1]), "latitude": len(latitudes)},
        "transition_width_km": width,
        "vertical_transition_method": attributes["vertical_transition_method"],
        "blend_space": {"log_species": attributes["donor_log_space_species"],
                        "linear_species": attributes["donor_linear_species"] or "none"},
        "latitude_method": attributes["ig2_latitude_interpolation"],
        "southern_hemisphere": attributes["ig2_southern_hemisphere"],
        "season_method": attributes["ig2_seasonal_interpolation"],
        "ig2_species_mapped": attributes["ig2_species_mapped"],
        "totals": totals,
        "per_variable": per_variable,
        "smoothness": smoothness,
        "variables_flagged_for_hard_seam": flagged,
    }
    report_directory = output_path.parent.parent / "diagnostics" / "application-ready"
    report_directory.mkdir(parents=True, exist_ok=True)
    report_name = output_path.stem.replace("jumacs_", "completion_report_") + ".json"
    (report_directory / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if plots_dir:
        _make_plots(plots_dir, original_data, completed_data, donor_columns, altitude, latitudes, months, donors_used)
    return report
