# Project layout

The repository root is the HPC working directory. Run JuMACS commands there.

| Directory | Purpose | Git |
| --- | --- | --- |
| `src/`, `config/`, `tests/`, `docs/`, `scripts/` | Reproducible code, settings, tests and instructions (`config/models/` holds one registry file per source) | Track |
| `data/raw/` | Unmodified downloaded files for registered sources (GEOSCCM, EMAC, WACCM-X today) | Ignore |
| `data/processed/` | Native monthly zonal time series | Ignore |
| `products/` | Climatologies, comparisons, manifests and diagnostics | Ignore |
| `site/` | Generated static browser and plots | Ignore |
| `cache/`, `tmp/` | Logs, caches and temporary work | Ignore |

`products/release/` is reserved for files explicitly selected for publication. Nothing is copied there automatically.

`config/climatology.yaml` defines the reference period and application pressure grid: 124 levels from `100000` Pa to `0.002` Pa, uniform in log pressure. Individual climatologies keep their native model grids. The application builder remaps each model onto the common pressure and latitude grid and can extend it upward with WACCM-X; the transition width is configured under `extension:`.

Each model and period yields **one** CF-1.13 NetCDF in `products/climatology/MODEL/` holding every processed species as `<variable>_<statistic>` (`mean`, `sigma`, `minimum`, `maximum`, `n_years`) on the model's own vertical grid and level name, with `native_*` attributes recording that grid, on a twelve-cell climatological `time` axis with `climatology_bounds` and `cell_methods` stating the across-year statistic. See [../README.md](../README.md#products) for the attribute conventions.

```text
HPC site/ (including application NCs) ──rsync──> local clone/site/ ──rsync──> web site mirror
HPC products/release/              ──rsync──> public data location
```

Run `make mirror-local` in the local Git clone to copy the complete generated site, including NetCDF downloads. It uses `~/jumount/data/slmet/model_data/jumacs` as the default source; set `JUMACS_HPC_ROOT` if the mount differs. Run `make mirror-web` from that local clone after setting `JUMACS_WEB_SITE_MIRROR` to the web-server destination. It requires a generated application site. `make publish-data` remains a separate, explicitly configured release-data mirror. No mirror runs during builds or tests. GitHub hosts code and documentation; the HPC filesystem hosts complete scientific data and generated products. No GitHub Pages configuration is used.
