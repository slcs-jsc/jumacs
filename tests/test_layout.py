"""Path and site checks that need no archive access."""
import importlib.util
import json
import os
import shutil
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
    for key in ("JUMACS_WEB_SITE_MIRROR", "JUMACS_WEB_DATA_MIRROR"):
        env.pop(key, None)
    for mode in ("web", "data"):
        result = subprocess.run(["bash", str(ROOT / "scripts/mirror.sh"), mode], env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 2 and "unset or empty" in result.stderr


def test_local_mirror_copies_self_contained_hpc_site(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / ".git").mkdir()
    shutil.copy2(ROOT / "scripts/mirror.sh", checkout / "scripts/mirror.sh")
    hpc = tmp_path / "hpc"
    (hpc / "site/products/application/SOCOL").mkdir(parents=True)
    (hpc / "site/index.html").write_text("site")
    (hpc / "site/products/application/SOCOL/jumacs_socol_application_climatology_1985-2014.nc").write_bytes(b"synthetic")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_rsync = fake_bin / "rsync"
    fake_rsync.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$JUMACS_RSYNC_LOG"\n')
    fake_rsync.chmod(0o755)
    log = tmp_path / "rsync.log"
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    env["JUMACS_RSYNC_LOG"] = str(log)
    env["JUMACS_HPC_ROOT"] = str(hpc)
    result = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "local"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 1
    assert "--delete" in calls[0] and str(hpc / "site") in calls[0]
    assert str(checkout / "site") in calls[0]


def test_local_mirror_rejects_missing_hpc_source(tmp_path):
    env = os.environ.copy()
    env["JUMACS_HPC_ROOT"] = str(tmp_path / "missing")
    result = subprocess.run(["bash", str(ROOT / "scripts/mirror.sh"), "local"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "Mirror source is missing" in result.stderr


def test_web_mirror_uses_complete_local_site(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/mirror.sh", checkout / "scripts/mirror.sh")
    site = checkout / "site"
    site.mkdir()
    (site / "index.html").write_text("site")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_rsync = fake_bin / "rsync"
    fake_rsync.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$JUMACS_RSYNC_LOG"\n')
    fake_rsync.chmod(0o755)
    log = tmp_path / "rsync.log"
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    env["JUMACS_RSYNC_LOG"] = str(log)
    env["JUMACS_WEB_SITE_MIRROR"] = "datapub.fz-juelich.de:/var/www/jumacs/"
    missing = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "web"],
                             env=env, capture_output=True, text=True, check=False)
    assert missing.returncode == 2 and "generated application site" in missing.stderr
    (site / "products/application/SOCOL").mkdir(parents=True)
    (site / "products/application/SOCOL/application.nc").write_bytes(b"synthetic")
    result = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "web"],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        f"-av --delete {site}/ datapub.fz-juelich.de:/var/www/jumacs/"]


def test_download_all_does_not_leak_its_arguments_to_python_setup(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    for name in ("download_all.sh", "python_setup.sh"):
        shutil.copy2(ROOT / "scripts" / name, checkout / "scripts" / name)
    venv = checkout / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text('#!/bin/sh\nexec python3 "$@"\n')
    (venv / "bin" / "python").chmod(0o755)
    (venv / "bin" / "activate").write_text('export VIRTUAL_ENV="${VIRTUAL_ENV:-$PWD/.venv}"\n')
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/bin/sh\ncase "$1" in\n'
        '  -) echo SOCOL ;;\n'
        '  -m) exit 0 ;;\n'
        '  --version) echo "Python 3.12 (mock)" ;;\n'
        '  scripts/archive_download.py) echo "$3: 1 files, 0 pending, ~0.00 GB listed" ;;\n'
        '  *) exit 9 ;;\nesac\n'
    )
    fake_python.chmod(0o755)
    (fake_bin / "module").write_text("#!/bin/sh\nexit 0\n")
    (fake_bin / "module").chmod(0o755)
    # Isolated environment: download_all.sh must resolve `python` to the mock, so
    # the ambient interpreter (active venv, PYTHONPATH, Lmod, shell functions) is
    # deliberately excluded rather than inherited.
    env = {"PATH": str(fake_bin) + os.pathsep + os.defpath + os.pathsep + "/usr/bin:/bin",
           "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "LANG": "C"}
    for args in ([], ["--execute"]):
        result = subprocess.run(["bash", str(checkout / "scripts/download_all.sh")] + args,
                                env=env, capture_output=True, text=True, cwd=checkout, check=False)
        assert "Unknown option" not in result.stdout + result.stderr, result.stdout + result.stderr
        assert "Python 3.12 (mock)" in result.stdout, result.stdout + result.stderr
        assert result.returncode == 0, result.stdout + result.stderr
        assert "All download-capable JuMACS sources are complete." in result.stdout
