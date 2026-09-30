# JuMACS

**Jülich Multi-source Atmospheric Climatology System** provides monthly zonal atmospheric reference climatologies derived from internally consistent chemistry-climate simulations. The default is **1985–2014 inclusive** (30 years), with `nominal_reference_year=2000` as an epoch label. It does not equate trend-sensitive concentrations with year-2000 abundances; it applies no detrending, normalization, bias correction, satellite adjustment, model weighting, or ensemble averaging.

```text
GEOSCCM refD1 ──→ monthly zonal series ──→ native climatology ──┐
EMAC refD1 ─────→ monthly zonal series ──→ native climatology ───┼──→ comparisons/evaluation
WACCM-X ───────→ monthly zonal series ──→ native climatology ───┘
```

GEOSCCM and EMAC are independent CCMI-2022 refD1 primary model sources. A first compact application product extends **each CCMI model separately** with WACCM-X over their common valid height range. There is no GEOSCCM–EMAC average. Native model products remain unchanged. Satellite climatologies are evaluation data only. They never fill cells or change production climatologies. Direct satellite fusion is inappropriate here because observing periods, instrument biases, retrieval characteristics, trends in long-lived gases, and vertical/temporal overlap differ.

## Sources

- [GEOSCCM refD1, CEDA UUID 0689a7f2e0964a7b89e395ee68d7eee5](https://catalogue.ceda.ac.uk/uuid/0689a7f2e0964a7b89e395ee68d7eee5/)
- [EMAC-CCMI2 refD1, CEDA UUID 9b15ae551fda4035a7940a3adbe31691](https://catalogue.ceda.ac.uk/uuid/9b15ae551fda4035a7940a3adbe31691/)
- [WACCM-X transient 1950–2015 monthly means, CEDA UUID dc91f5e39ae34fd883af81dfdbaf659c](https://catalogue.ceda.ac.uk/uuid/dc91f5e39ae34fd883af81dfdbaf659c/)

The WACCM-X archive publishes ready-made monthly zonal `_zm.nc` files; JuMACS uses these directly. Its file `time` coordinate is the **next month's first day**, while `time_bnds` and the filename identify the represented month. The reader labels samples with the interval midpoint, so `1985-01_zm.nc` contributes to January 1985. Its 126 hybrid midpoint levels use `p=hyam*P0+hybm*PS`. GEOSCCM full 3-D fields use `p=a*p0+b*ps` (72 levels); EMAC uses `p=ap+b*ps` (90 levels). CCMI `AmonZ` chemistry uses native pressure coordinates in Pa. Actual valid coverage is calculated from finite species values and pressure, not inferred from model top. Altitude comes from native geopotential height where coordinates align, particularly for WACCM-X. Other levels receive a labelled `7 ln(101325/p)` km scale-height estimate, which is unreliable above roughly 80 km.

`config/models/` is the source registry: one YAML file per CCMI-2022 refD1 model plus `config/waccmx.yaml` for the separate whole-atmosphere source. GEOSCCM and EMAC are `status: ready`; the other ten CCMI refD1 models are registered with `status: unresolved` placeholders (CEDA identifiers to confirm, never guessed) and are skipped by data commands until their `institution`, `source archive_model` and `dataset_uuid` are completed. Each source also declares explicit `capabilities` (`archive_inventory`, `download`, `zonal_processing`, `climatology`, `compact_waccmx`); commands such as `compact` accept only models whose capability is enabled, and unregistered capability defaults to false. `config/species.yaml` defines the canonical species list and groups used by the inventory and coverage matrices. `inspect --model all` writes model-specific archive inventories plus `products/comparison/variable_matrix.csv` (per-model `<model>_status` of `ok`/`absent`/`no_mapping`/`unresolved`/`not_inspected` and `<model>_available`) and `products/comparison/native_variables.csv` across the full registry. The native-variable report keeps source fields that exist only in a monthly multi-variable file, such as WACCM-X native species, separate from the canonical species list until a mapping is deliberately configured. Missing species remain missing. No family sum replaces an individual gas. Neither CCMI refD1 source has a monthly SF6 directory, and the inspected WACCM-X zonal file has no three-dimensional SF6 field. WACCM-X zonal files contain many variables together, so selecting one variable for download still selects the monthly multi-variable files.

## Workflow on JUWELS

From the repository and HPC project root `/p/data1/slmet/model_data/jumacs`:

```bash
module load Stages/2026 GCCcore/14.3.0 SciPy-Stack/2025b netcdf4-python/1.7.2 BeautifulSoup/4.14.2 PyYAML/6.0.2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest -q
python3 -m jumacs.cli inspect --model GEOSCCM
python3 -m jumacs.cli inspect --model EMAC
python3 -m jumacs.cli inspect --model WACCM-X
python3 -m jumacs.cli download --model WACCM-X --start-year 1985 --end-year 2014  # size plan only
```

The installed command is `jumacs` (`pip install -e .`); `python3 -m jumacs.cli` works with the modules and `PYTHONPATH`. `--model` accepts any registered model name or `all`. The `download` command is plan-only: it prints and saves a size estimate without transfer.

Transfers need general Internet access, which JUWELS **compute nodes do not have**; there is therefore no download wrapper in `scripts/slurm/`. Execute a reviewed plan only where Internet egress works: on **JUDAC** (`python3 scripts/judac_download.py --model MODEL --execute`, auto-discovers manifests in `products/manifests/`; preferred for bulk transfer) or on a **JUWELS login node** (`jumacs download --model MODEL --execute` or `make download-execute MODEL=...`). Login-node transfers are for small selections only; keep bulk work on JUDAC and mount the shared tree over SSHFS (`~/jumount`) when copying from JUDAC. Heavy processing still runs through Slurm, not on the login node, and Slurm jobs must not fetch data at run time:

```bash
sbatch -A YOUR_ACCOUNT scripts/slurm/climatology.sh GEOSCCM 1985 2014
sbatch -A YOUR_ACCOUNT scripts/slurm/climatology.sh EMAC 1985 2014
sbatch -A YOUR_ACCOUNT scripts/slurm/compare.sh 1985 2014
sbatch -A YOUR_ACCOUNT scripts/slurm/plot_catalog.sh
sbatch -A YOUR_ACCOUNT scripts/slurm/plot_comparisons.sh
sbatch -A YOUR_ACCOUNT scripts/slurm/compact.sh GEOSCCM
sbatch -A YOUR_ACCOUNT scripts/slurm/compact.sh EMAC
```

For WACCM-X, first inspect and plan download, then transfer the selected monthly `_zm.nc` files from JUDAC. Afterward, submit zonal tasks by variable and its climatology. `make inspect`, `make test`, `make download MODEL=...` (plan), `make download-execute MODEL=...`, `make zonal MODEL=...`, `make climatology MODEL=...`, `make coverage MODEL=...`, and `make compare` are shortcuts. The default 1985–2014 reference period is defined only in `config/climatology.yaml`; the Makefile and CLI omit year flags when unset and read that file, and a model YAML may add a `period:` override. `--start-year` and `--end-year` allow any covered inclusive period. The monthly zonal series remain intact.

## Products

Each `data/processed/MODEL/.../VARIABLE_monthly_zonal.nc` retains time, latitude, native level, native units, source file names, and model identity. `jumacs climatology --model GEOSCCM` creates `products/climatology/GEOSCCM/jumacs_geosccm_refd1_climatology_1985-2014.nc` with one NetCDF group per variable, plus independently readable per-variable NetCDF files. Each group has calendar-month `mean`, population `sigma`, `minimum`, `maximum`, and cellwise `n_years`. Products record source dataset/archive, experiment, coverage, period, native/output units, vertical coordinate, and processing history.

`jumacs coverage --model MODEL` writes finite-value-based pressure limits, native geopotential-height ranges (where available), a representative median top height and level count per species to `products/diagnostics/coverage/MODEL/`. `jumacs compare --models GEOSCCM EMAC` writes original model fields, absolute and relative difference on a common pressure grid in the overlapping domain. Relative difference is `100 × (first − second) / second` and is undefined where the second field is zero. Interpolation is in log pressure and never extrapolates. `jumacs evaluate` accepts a separate evaluation NetCDF with the requested field and pressure coordinate, and produces difference, relative difference and RMS; it does not rank or adjust models. Plots from `quicklook`, `trends`, the model atlas and representative comparison maps are diagnostics only.

`jumacs compact --model GEOSCCM` and `jumacs compact --model EMAC` create separate application files in `products/climatology/combined/`. The default grid is monthly × altitude (0–120 km in 1 km steps) × latitude (85°S–85°N in 5° steps); longitude is absent because the source fields are zonal means. Fields are stored as compressed float32. The altitude axis is **geopotential height**, mapped from each source's monthly native geopotential-height profile; pressure-level chemistry is mapped using that model's pressure–height relation. Interpolation does not extrapolate below the lowest or above the highest valid native point, so the exact 0 km cell can be missing. Where both models provide a species, a raised-cosine transition spans 55–65 km. That band lies inside the checked overlap of the available fields; the command refuses an incomplete overlap. Source-variable names and `source_status` identify whether a field is blended or supplied by only one source. Species absent from WACCM-X remain missing above the CCMI valid top. A compact product is a first application candidate, not a replacement for either native climatology.

The compact file is a straightforward month/height/latitude input for a JuMACS-aware JURASSIC or MPTRAC adapter. The current MPTRAC `read_clim_zm()` interface expects **separate** pressure-grid files with `time,press,lat` and variable-specific names, so it cannot directly read this multi-variable altitude-grid file. A pressure-grid adapter and application-side integration still need validation before claiming plug-and-play use.

`docs/evaluation/` documents independent satellite comparisons (for example SPARC, SWOOSH, GOZCARDS, ACE-FTS, MIPAS or MLS). No satellite dataset is fetched by this project. Data and generated products are excluded from Git. The MIT licence covers project code; cite the CEDA datasets and follow their licences when using their data.

## Repository and mirror layout

`jumacs/` is both the Git root and the HPC working directory. Tracked material is limited to `src/`, `config/`, `tests/`, `docs/`, `scripts/`, and root metadata (`README.md`, `LICENSE`, `CITATION.cff`, `pyproject.toml`, `Makefile`, `.gitignore`). The full model files in `data/raw/`, monthly zonal series in `data/processed/`, generated scientific files in `products/`, and the static browser in `site/` are ignored by Git. `cache/` and `tmp/` are ignored work areas. The path defaults are documented in `config/jumacs.yaml` and resolved from the repository root. `products/release/` is reserved for explicitly selected public datasets; nothing is added automatically.

`make site` rebuilds `site/index.html` from existing plot catalogs without redrawing the plots. Internal browser links are relative. By default the browser does not link to NetCDF products because `site/` must also work when mirrored by itself. Set `JUMACS_PRODUCT_URL_BASE` during site generation only if those selected compact files are published at that URL.

Run the local mirror **from the notebook Git clone**. It reads the mounted HPC tree at `~/jumount/data/slmet/model_data/jumacs` by default and writes the static browser to the clone's ignored `site/` directory. It also copies exactly five finished 1985–2014 NetCDF products to the clone's ignored `products/climatology/`: the grouped GEOSCCM, EMAC and WACCM-X climatologies and the two compact extensions. Per-variable files and the full monthly zonal series are excluded. The site is an exact mirror; NetCDF files are copied incrementally without deletion.

```bash
cd ~/wrk/clim/jumacs
make mirror-local
```

Set `JUMACS_HPC_ROOT` if the HPC tree is mounted elsewhere. The web and publication commands still require explicit destinations:

```bash
export JUMACS_WEB_SITE_MIRROR=user@web:/path/to/site
make mirror-web
export JUMACS_WEB_DATA_MIRROR=user@web:/path/to/data
make publish-data  # only products/release/
```

The intended arrangement is GitHub for code and documentation, the HPC working repository for full data and products, a local Git checkout with a mirrored site, and a web server with the mirrored site plus selected released data. JuMACS does not use GitHub Pages. See [docs/README.md](docs/README.md) for the directory map.
