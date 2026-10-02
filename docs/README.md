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

`config/climatology.yaml` is the single definition of both the reference period and the project-wide common pressure grid (`vertical_grid`: endpoints of `100000` Pa and `0.002` Pa — roughly the surface to about 120 km — uniform in `log10(pressure)` with `intervals_per_decade: 16`, which gives 123 intervals and 124 levels of about 1 km, `linear_log_pressure`, `extrapolation: none`). Every climatology product is written on that axis as the shared `pressure` coordinate; the level count follows from the endpoints and the resolution, so changing those numbers and rebuilding is the only way to change the grid, and a product written on another axis is refused rather than resampled. The planned combination step takes its transition width from the `extension:` section as a number of common-grid levels (`transition_levels: 12`, about 11 km), optionally per variable, and is configuration only — no fields are blended yet.

Each model and period yields **one** CF-1.13 NetCDF in `products/climatology/MODEL/` holding every processed species as `<variable>_<statistic>` (`mean`, `sigma`, `minimum`, `maximum`, `n_years`) on a twelve-cell climatological `time` axis with `climatology_bounds` and `cell_methods` stating the across-year statistic. See [../README.md](../README.md#products) for the attribute conventions.

```text
HPC site/                          ──rsync──> local clone/site/
                                   └─rsync──> web site mirror
HPC five finished climatology NCs  ──rsync──> local clone/products/climatology/
HPC products/release/              ──rsync──> public data location
```

Run `make mirror-local` in the local Git clone. It uses `~/jumount/data/slmet/model_data/jumacs` as the default source; set `JUMACS_HPC_ROOT` if the mount differs. It copies only the three model climatology files (one CF-1.13 NetCDF each) and two compact extension NetCDF files for 1985–2014. `make mirror-web` and `make publish-data` still require non-empty destination environment variables. No mirror runs during builds or tests. GitHub hosts code and documentation; the HPC filesystem hosts complete scientific data and generated products. No GitHub Pages configuration is used.
