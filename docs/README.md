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

`site/` is the single publication tree: the generated browser plus the NetCDF downloads it serves, under `site/products/application/` and `site/products/combined/`.

`config/climatology.yaml` defines the reference period and application pressure grid: 124 levels from `100000` Pa to `0.002` Pa, uniform in log pressure. Individual climatologies keep their native model grids. The application builder remaps each model onto the common pressure and latitude grid and can extend it upward with WACCM-X; the transition width is configured under `extension:`.

Each model and period yields **one** CF-1.13 NetCDF in `products/climatology/MODEL/` holding every processed species as `<variable>_<statistic>` (`mean`, `sigma`, `minimum`, `maximum`, `n_years`) on the model's own vertical grid and level name, with `native_*` attributes recording that grid, on a twelve-cell climatological `time` axis with `climatology_bounds` and `cell_methods` stating the across-year statistic. See [../README.md](../README.md#products) for the attribute conventions.

```text
HPC site/ (browser + application and combined NCs) ──rsync/SSH──> local clone/site/ ──rsync──> web mirror
```

Run `make mirror-local` in the local Git clone to pull the complete generated site, including all NetCDF downloads, directly from JUDAC over SSH. The source is `JUMACS_HPC_MIRROR`, by default `judac.fz-juelich.de:/p/data1/slmet/model_data/jumacs/site`. Run `make mirror-web` from that local clone after setting `JUMACS_WEB_SITE_MIRROR` to the web-server destination. It requires a generated application site. Both mirrors use `rsync --delete` and refuse unsafe destinations. No mirror runs during builds or tests. GitHub hosts code and documentation; the HPC filesystem hosts complete scientific data and generated products. No GitHub Pages configuration is used.
