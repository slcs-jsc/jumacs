from pathlib import Path
import math
import re
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "config" / "models"
WACCMX_CONFIG = ROOT / "config" / "waccmx.yaml"


def model_slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_file(path):
    with path.open() as stream:
        return yaml.safe_load(stream)


def registry():
    """All registered sources keyed by model name: CCMI model configs plus WACCM-X."""
    entries = {}
    for path in sorted(MODELS_DIR.glob("*.yaml")):
        config = _load_file(path)
        entries[config["model"]["name"]] = path
    entries["WACCM-X"] = WACCMX_CONFIG
    return entries


def model_names():
    return tuple(registry())


def ccmi_model_names():
    return tuple(name for name in registry() if load_config(name)["model"].get("kind", "ccmi") == "ccmi")


def ready_model_names():
    return tuple(name for name in registry() if load_config(name)["model"].get("status", "ready") == "ready")


def ready_ccmi_model_names():
    return tuple(name for name in ccmi_model_names() if load_config(name)["model"].get("status", "ready") == "ready")


def model_metadata(name):
    config = load_config(name)
    model = config["model"]
    return {"name": name, "display_name": model.get("display_name", name),
            "institution": model.get("institution", "unresolved"), "source_id": model.get("source_id", name),
            "kind": model.get("kind", "ccmi"), "status": model.get("status", "ready"),
            "slug": model.get("slug") or (model_slug(name) if name != "WACCM-X" else "waccmx")}


def is_waccmx(config):
    return config["model"].get("kind") == "whole_atmosphere" or config["model"].get("source_kind") == "monthly_zonal_multivariable"


def capabilities(model):
    return dict(load_config(model).get("capabilities", {}))


def has_capability(model, capability):
    return bool(capabilities(model).get(capability, False))


def models_with_capability(capability):
    return tuple(name for name in registry() if has_capability(name, capability))


def load_config(model):
    entries = registry()
    if model not in entries:
        raise ValueError(f"Unknown model {model!r}; registered: {', '.join(sorted(entries))}")
    config = _load_file(entries[model])
    if config["model"]["name"] != model:
        raise ValueError("Configuration must identify the requested model")
    if config["model"].get("kind", "ccmi") == "ccmi" and config["model"].get("status", "ready") == "ready" and config["model"]["experiment"] != "refD1":
        raise ValueError("CCMI configuration must use the refD1 experiment")
    return config


def require_ready(model):
    model_config = load_config(model)["model"]
    if model_config.get("status", "ready") != "ready":
        raise ValueError(f"{model} archive identifiers are unresolved; complete config/models/{model}.yaml before use")
    return model_config


def reference_period():
    return _load_file(ROOT / "config" / "climatology.yaml")


def pressure_levels(max_pressure_pa, min_pressure_pa, intervals_per_decade):
    """Descending log-uniform pressure levels with both endpoints exact.

    The level count follows from the requested resolution: one interval every
    `intervals_per_decade` steps in log10(pressure), rounded to the nearest whole
    interval so the endpoints stay untouched and the spacing in log-pressure is
    constant. The pressure mapping to altitude is only approximate, so altitude is
    never used to build the grid.
    """
    if max_pressure_pa <= 0 or min_pressure_pa <= 0:
        raise ValueError("vertical_grid pressures must be positive values in Pa")
    if max_pressure_pa <= min_pressure_pa:
        raise ValueError("vertical_grid.max_pressure_pa must be the larger (lower) pressure")
    if intervals_per_decade < 1:
        raise ValueError("vertical_grid.intervals_per_decade must be at least 1")
    decades = math.log10(max_pressure_pa / min_pressure_pa)
    intervals = max(1, round(decades * intervals_per_decade))
    ratio = min_pressure_pa / max_pressure_pa
    levels = [max_pressure_pa * ratio ** (index / intervals) for index in range(intervals + 1)]
    levels[0], levels[-1] = float(max_pressure_pa), float(min_pressure_pa)
    return tuple(levels)


def vertical_grid():
    """Project-wide pressure coordinate used by every climatology product.

    The configuration states the endpoints and the target resolution; the levels
    themselves are derived here, so there is one definition and no second list to
    keep in step.
    """
    grid = reference_period().get("vertical_grid")
    if not grid:
        raise ValueError("config/climatology.yaml needs a vertical_grid section with pressure endpoints in Pa "
                         "and a resolution in intervals per decade")
    if grid.get("units", "Pa") != "Pa":
        raise ValueError("vertical_grid.units must be Pa")
    if grid.get("interpolation", "linear_log_pressure") != "linear_log_pressure":
        raise ValueError("vertical_grid.interpolation must be linear_log_pressure")
    if grid.get("extrapolation", "none") != "none":
        raise ValueError("vertical_grid.extrapolation must be none")
    if grid.get("levels"):
        raise ValueError("vertical_grid.levels is no longer accepted; set max_pressure_pa, min_pressure_pa and "
                         "intervals_per_decade so the level count follows from the requested resolution")
    missing = [key for key in ("max_pressure_pa", "min_pressure_pa", "intervals_per_decade") if key not in grid]
    if missing:
        raise ValueError(f"vertical_grid is missing {', '.join(missing)}")
    levels = pressure_levels(float(grid["max_pressure_pa"]), float(grid["min_pressure_pa"]),
                             int(grid["intervals_per_decade"]))
    if any(level <= 0 for level in levels):
        raise ValueError("vertical_grid levels must all be positive pressures in Pa")
    if not all(a > b for a, b in zip(levels, levels[1:])):
        raise ValueError("vertical_grid levels must be strictly monotonic decreasing")
    return {"coordinate": grid.get("coordinate", "pressure"), "units": "Pa",
            "interpolation": "linear_log_pressure", "extrapolation": "none",
            "description": grid.get("description", ""), "levels": levels,
            "intervals": len(levels) - 1,
            "delta_log10_pressure": math.log10(levels[-1] / levels[0]) / (len(levels) - 1)}


def extension_settings():
    """Vertical transition width for the planned CCMI + WACCM-X combination.

    Widths are counted in common-grid levels, not in kilometres, so the same
    configuration keeps its meaning when the pressure grid changes; 12 levels span
    11 intervals, about 11 km on the ~1 km grid.
    """
    extension = reference_period().get("extension") or {}
    levels = int(extension.get("transition_levels", 12))
    if levels < 2:
        raise ValueError("extension.transition_levels must be at least 2")
    overrides = dict(extension.get("transition_levels_by_variable") or {})
    for variable, value in overrides.items():
        if int(value) < 2:
            raise ValueError(f"extension.transition_levels_by_variable.{variable} must be at least 2")
    return {"transition_levels": levels,
            "transition_levels_by_variable": {str(key): int(value) for key, value in overrides.items()}}



def coverage_settings():
    """Coverage-matrix thresholds and the application focus list; diagnostics only, never products."""
    coverage = reference_period().get("coverage") or {}

    def fraction(key, default):
        value = float(coverage.get(key, default))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"coverage.{key} must be in (0, 1]")
        return value

    return {"usable_level_fraction": fraction("usable_level_fraction", 0.5),
            "usable_sample_fraction": fraction("usable_sample_fraction", 0.9),
            "application_focus": tuple(coverage.get("application_focus", ()))}


def model_period(model):
    """Per-model processing window; config/climatology.yaml is the single default source."""
    period = load_config(model).get("period")
    if period:
        return period
    reference = reference_period()["reference_period"]
    return {"start_year": reference["start_year"], "end_year": reference["end_year"]}


def species_registry():
    return _load_file(ROOT / "config" / "species.yaml")["species"]


def project_path(relative):
    return ROOT / relative
