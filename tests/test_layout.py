"""Path and site checks that need no archive access."""
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


def test_mirror_rejects_unset_destination():
    env = os.environ.copy()
    env.pop("JUMACS_WEB_SITE_MIRROR", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/mirror.sh"), "web"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "unset or empty" in result.stderr


def test_mirror_rejects_removed_data_mode():
    result = subprocess.run(["bash", str(ROOT / "scripts/mirror.sh"), "data"], env=os.environ.copy(),
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "Usage" in result.stderr


def _fake_rsync(tmp_path, env):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_rsync = fake_bin / "rsync"
    fake_rsync.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$JUMACS_RSYNC_LOG"\n')
    fake_rsync.chmod(0o755)
    log = tmp_path / "rsync.log"
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    env["JUMACS_RSYNC_LOG"] = str(log)
    return log


def test_local_mirror_copies_self_contained_hpc_site(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / ".git").mkdir()
    shutil.copy2(ROOT / "scripts/mirror.sh", checkout / "scripts/mirror.sh")
    hpc = tmp_path / "hpc/site"
    (hpc / "products/harmonized/SOCOL").mkdir(parents=True)
    (hpc / "index.html").write_text("site")
    (hpc / "products/harmonized/SOCOL/jumacs_socol_1985-2014_5deg_1km.nc").write_bytes(b"synthetic")
    env = os.environ.copy()
    log = _fake_rsync(tmp_path, env)
    env["JUMACS_HPC_MIRROR"] = str(hpc)
    result = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "local"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 1
    assert "--delete" in calls[0] and str(hpc) in calls[0]
    assert str(checkout / "site") in calls[0]


def _bare_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / ".git").mkdir()
    shutil.copy2(ROOT / "scripts/mirror.sh", checkout / "scripts/mirror.sh")
    return checkout


def test_local_mirror_default_source_is_judac(tmp_path):
    checkout = _bare_checkout(tmp_path)
    env = os.environ.copy()
    log = _fake_rsync(tmp_path, env)
    env.pop("JUMACS_HPC_MIRROR", None)
    result = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "local"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        f"-av --delete judac.fz-juelich.de:/p/data1/slmet/model_data/jumacs/site/ {checkout / 'site'}/"]


def test_local_mirror_pulls_remote_site_without_local_source_check(tmp_path):
    checkout = _bare_checkout(tmp_path)
    env = os.environ.copy()
    log = _fake_rsync(tmp_path, env)
    env["JUMACS_HPC_MIRROR"] = "override.example.org:/srv/jumacs/site"
    result = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "local"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        f"-av --delete override.example.org:/srv/jumacs/site/ {checkout / 'site'}/"]


def test_local_mirror_rejects_missing_hpc_source(tmp_path):
    env = os.environ.copy()
    env["JUMACS_HPC_MIRROR"] = str(tmp_path / "missing")
    result = subprocess.run(["bash", str(ROOT / "scripts/mirror.sh"), "local"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "Mirror source is missing" in result.stderr


def test_local_mirror_rejects_self_mirror(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / ".git").mkdir()
    (checkout / "site").mkdir()
    shutil.copy2(ROOT / "scripts/mirror.sh", checkout / "scripts/mirror.sh")
    env = os.environ.copy()
    _fake_rsync(tmp_path, env)
    env["JUMACS_HPC_MIRROR"] = str(checkout / "site")
    result = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "local"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "same directory" in result.stderr


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
    assert missing.returncode == 2 and "generated site" in missing.stderr
    (site / "products/harmonized/SOCOL").mkdir(parents=True)
    (site / "products/harmonized/SOCOL/harmonized.nc").write_bytes(b"synthetic")
    result = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "web"],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        f"-av --delete {site}/ datapub.fz-juelich.de:/var/www/jumacs/"]


def _web_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/mirror.sh", checkout / "scripts/mirror.sh")
    site = checkout / "site"
    (site / "products/harmonized/SOCOL").mkdir(parents=True)
    (site / "index.html").write_text("new-index-content")
    (site / "products/harmonized/SOCOL/harmonized.nc").write_bytes(b"new")
    return checkout, site


def test_web_mirror_dry_run_passes_dry_run_and_delete_to_rsync(tmp_path):
    checkout, site = _web_checkout(tmp_path)
    env = os.environ.copy()
    log = _fake_rsync(tmp_path, env)
    env["JUMACS_WEB_SITE_MIRROR"] = "datapub.fz-juelich.de:/mnt/slcs/jumacs/"
    result = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "web", "--dry-run"],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        f"-av --delete --dry-run {site}/ datapub.fz-juelich.de:/mnt/slcs/jumacs/"]


def test_web_mirror_dry_run_cannot_mutate_destination(tmp_path):
    checkout, _site = _web_checkout(tmp_path)
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "index.html").write_text("old")
    (remote / "stale.png").write_text("stale")
    env = os.environ.copy()
    env["JUMACS_WEB_SITE_MIRROR"] = str(remote)
    dry = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "web", "--dry-run"],
                         env=env, capture_output=True, text=True, check=False)
    assert dry.returncode == 0, dry.stderr
    assert "deleting stale.png" in dry.stdout
    assert (remote / "index.html").read_text() == "old"
    assert (remote / "stale.png").is_file()
    assert not (remote / "products").exists()
    real = subprocess.run(["bash", str(checkout / "scripts/mirror.sh"), "web"],
                          env=env, capture_output=True, text=True, check=False)
    assert real.returncode == 0, real.stderr
    assert (remote / "index.html").read_text() == "new-index-content"
    assert not (remote / "stale.png").exists()
    assert (remote / "products/harmonized/SOCOL/harmonized.nc").read_bytes() == b"new"


def test_mirror_dry_run_still_rejects_unset_destination():
    env = os.environ.copy()
    env.pop("JUMACS_WEB_SITE_MIRROR", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/mirror.sh"), "web", "--dry-run"],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "unset or empty" in result.stderr


def test_mirror_rejects_unknown_option():
    for args in (["web", "--oops"], ["web", "--dry-run", "extra"]):
        result = subprocess.run(["bash", str(ROOT / "scripts/mirror.sh")] + args,
                                env=os.environ.copy(), capture_output=True, text=True, check=False)
        assert result.returncode == 2 and "Usage" in result.stderr


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
