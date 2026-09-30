"""Path and site checks that need no archive access."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import yaml

from jumacs.config import ROOT, load_config
from jumacs.download import manifest_path


def test_repository_relative_paths():
    layout = yaml.safe_load((ROOT / "config/jumacs.yaml").read_text())["paths"]
    assert layout == {"raw": "data/raw", "processed": "data/processed",
                      "products": "products", "site": "site", "cache": "cache", "tmp": "tmp"}
    for model in ("GEOSCCM", "EMAC", "WACCM-X"):
        cfg = load_config(model)
        for key, prefix in (("raw", "data/raw/"), ("zonal", "data/processed/"),
                            ("climatology", "products/climatology/")):
            value = cfg["paths"][key]
            assert not Path(value).is_absolute() and value.startswith(prefix)
        assert manifest_path(model).parent == ROOT / "products/manifests"


def test_site_build_has_only_relative_default_links(tmp_path, monkeypatch):
    path = ROOT / "scripts/build_index.py"
    spec = importlib.util.spec_from_file_location("site_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.delenv("JUMACS_PRODUCT_URL_BASE", raising=False)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "site").mkdir()
    (tmp_path / "site/catalog.json").write_text(json.dumps([{"model": "GEOSCCM", "views": {"timeline": "plots/o3.png"}}]))
    (tmp_path / "scripts/index_template.html").write_text('<a href="plots/o3.png">plot</a> __JUMACS_CATALOG__ __JUMACS_DATA_LINKS__')
    text = module.build_index().read_text()
    assert 'href="plots/o3.png"' in text
    assert "compact/jumacs_" not in text
    assert "products/climatology/combined" in text


def test_mirror_rejects_unset_destination():
    env = os.environ.copy()
    for key in ("JUMACS_LOCAL_SITE_MIRROR", "JUMACS_WEB_SITE_MIRROR", "JUMACS_WEB_DATA_MIRROR"):
        env.pop(key, None)
    for mode in ("local", "web", "data"):
        result = subprocess.run(["bash", str(ROOT / "scripts/mirror.sh"), mode], env=env, capture_output=True, text=True)
        assert result.returncode == 2 and "unset or empty" in result.stderr
