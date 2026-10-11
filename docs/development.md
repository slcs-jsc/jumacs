# Development and reproducibility

`jumacs/` is both the Git root and the HPC working directory. Run JuMACS commands there. Tracked material is limited to `src/`, `config/`, `tests/`, `docs/`, `scripts/`, and root metadata (`README.md`, `LICENSE`, `CITATION.cff`, `pyproject.toml`, `Makefile`, `.gitignore`). The full model files in `data/raw/`, monthly zonal series in `data/processed/`, generated scientific files in `products/`, and the static browser in `site/` are ignored by Git. `cache/` and `tmp/` are ignored work areas. The path defaults are documented in `config/jumacs.yaml` and resolved from the repository root.

| Directory | Purpose | Git |
| --- | --- | --- |
| `src/`, `config/`, `tests/`, `docs/`, `scripts/` | Reproducible code, settings, tests and instructions (`config/models/` holds one registry file per source) | Track |
| `data/raw/` | Unmodified downloaded files for registered sources | Ignore |
| `data/processed/` | Native monthly zonal time series | Ignore |
| `products/` | Climatologies, comparisons, manifests and diagnostics | Ignore |
| `site/` | Generated static browser, plots and published NetCDF products | Ignore |
| `cache/`, `tmp/` | Logs, caches and temporary work | Ignore |

## Workflow on JUWELS

From the repository and HPC project root `/p/data1/slmet/model_data/jumacs`:

```bash
source scripts/python_setup.sh
python -m pytest -q
jumacs inspect --model all            # archive contents, variable mappings, readiness
jumacs download --model CMAM --start-year 1985 --end-year 2014  # size plan only
jumacs build --model CMAM --start-year 1985 --end-year 2014     # zonal series + climatology + report
jumacs quicklook --model CMAM --start-year 1985 --end-year 2014
jumacs build --model CMAM,SOCOL --start-year 1985 --end-year 2014   # several models in one call
jumacs build --model all              # every ready model with validated zonal processing
jumacs application --model SOCOL --start-year 1985 --end-year 2014  # pressure-grid product
jumacs browse --start-year 1985 --end-year 2014  # static site from existing application products
jumacs download --model GEOSCCM --all-files  # complete-archive mirror plan (all families/members, latest versions)
jumacs download-audit  # audit all mirror plans, write products/comparison/download_plan_summary.{csv,md}
```

`build` takes one model (or list, or `all`) as the unit of work: it reads the variables from `config/models/<model>.yaml`, reuses any existing `data/processed/.../VARIABLE_monthly_zonal.nc`, builds the missing monthly zonal series, computes the climatology, and prints one JSON report per model (`zonal_built`, `zonal_reused`, `skipped` with reasons, `variables_processed`, `product`, `conventions`, `variables_in_product`, the native `vertical` grid actually written, the `application_pressure_grid` that combination will apply, per-variable `checks`, `ok`). Variables without downloaded data, or whose zonal series does not overlap the requested period, are skipped and named in the report instead of failing the run; the run fails only when no variable is processable. Registry mappings are never bypassed: a model must combine `status: ready` with `capabilities.zonal_processing: true`, so explicit `--variable` does not unlock unvalidated coordinate mappings. The per-step commands (`zonal`, `climatology`, `zonal-smoke`, `validate-raw`, `coordinate-audit`, `post-download-audit`) remain available and are labelled advanced in `jumacs --help`.

IPSL-CM6A-ATM-LR-REPROBUS and CCSR-NIES-MIROC32 use their published AmonZ pressure-level fields for 3D climatologies and Amon fields for 2D quantities. CESM2-WACCM omits `c2h2`, `c2h6`, `co2` and `epfy` from the default build. These mappings remain in the registry, but the default build does not silently misplace them.

`scripts/python_setup.sh` loads the Python module, creates/activates `.venv` and installs JuMACS with the test extra; re-run it with `source scripts/python_setup.sh --update` to refresh pip, JuMACS and dependencies.

The installed command is `jumacs` (editable install from the setup script); `python3 -m jumacs.cli` also works with manually loaded modules and `PYTHONPATH="$PWD/src"`. `--model` accepts any registered model name or `all`. The `download` command is plan-only: it prints and saves a size estimate without transfer. `download --model MODEL --all-files` plans a complete-archive mirror instead of the mapped-species window: every file in the model's inventory across all families and members of the configured archive trees, superseded versions deduplicated to the latest release; `download-audit` rebuilds those mirror plans for every available source and writes the audit to `products/comparison/download_plan_summary.csv`/`.md` (WACCM-X reported separately; unresolved models excluded).

Transfers need general Internet access, which JUWELS **compute nodes do not have**; there is therefore no download wrapper in `scripts/slurm/`. Execute a reviewed plan only where Internet egress works: with `python3 scripts/archive_download.py --model MODEL --execute` (or `--all`), which auto-discovers manifests in `products/manifests/` and needs only `requests` — run it on **JUDAC** (preferred for bulk transfer) or on a **JUWELS login node** (`jumacs download --model MODEL --execute` or `make download-execute MODEL=...` remain available for small selections). Keep bulk work on JUDAC; publication copies the generated site from JUDAC with plain rsync over SSH (`make mirror-local`), so no filesystem mount is involved. Heavy processing still runs through Slurm, not on the login node, and Slurm jobs must not fetch data at run time:

```bash
sbatch -A YOUR_ACCOUNT scripts/slurm/build.sh GEOSCCM 1985 2014   # zonal + climatology
sbatch -A YOUR_ACCOUNT scripts/slurm/build.sh EMAC 1985 2014
sbatch -A YOUR_ACCOUNT scripts/slurm/compare.sh 1985 2014
sbatch -A YOUR_ACCOUNT scripts/slurm/plot_catalog.sh
sbatch -A YOUR_ACCOUNT scripts/slurm/plot_comparisons.sh
sbatch -A YOUR_ACCOUNT scripts/slurm/compact.sh GEOSCCM
sbatch -A YOUR_ACCOUNT scripts/slurm/compact.sh EMAC
```

`scripts/slurm/build.sh` is the normal submission path because it wraps `jumacs build`: it reuses existing monthly zonal intermediates, builds only what is missing, and validates the climatology afterwards. The per-stage wrappers (`scripts/slurm/zonal_model.sh`, `zonal_array.sh`, `climatology.sh`, `submit_processing.py`) and the matching `jumacs zonal` / `jumacs climatology` commands, `make zonal MODEL=...` and `make climatology MODEL=...` are advanced entry points for re-running one stage after a failed build or for debugging a single variable; running `climatology.sh` alone skips the coverage checks that `build` performs.

For WACCM-X, first inspect and plan download, then transfer the selected monthly `_zm.nc` files from JUDAC. Afterward, submit zonal tasks by variable and its climatology. `make inspect`, `make test`, `make download MODEL=...` (plan), `make download-execute MODEL=...`, `make build MODEL=...`, `make zonal MODEL=...`, `make climatology MODEL=...`, `make coverage MODEL=...`, and `make compare` are shortcuts. The default 1985–2014 reference period is defined only in `config/climatology.yaml`; the Makefile and CLI omit year flags when unset and read that file, and a model YAML may add a `period:` override. `--start-year` and `--end-year` allow any covered inclusive period. The monthly zonal series remain intact.
