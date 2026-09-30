from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"GEOSCCM": "geosccm_refd1.yaml", "EMAC": "emac_refd1.yaml", "WACCM-X": "waccmx.yaml"}


def load_config(model):
    if model not in CONFIGS:
        raise ValueError(f"Unknown model {model!r}; use {', '.join(CONFIGS)}")
    with (ROOT / "config" / CONFIGS[model]).open() as stream:
        config = yaml.safe_load(stream)
    if config["model"]["name"] != model or (model != "WACCM-X" and config["model"]["experiment"] != "refD1"):
        raise ValueError("Configuration must identify the requested model and experiment")
    return config


def reference_period():
    with (ROOT / "config" / "climatology.yaml").open() as stream:
        return yaml.safe_load(stream)


def project_path(relative):
    return ROOT / relative
