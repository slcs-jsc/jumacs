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

```text
HPC site/                          ──rsync──> local clone/site/
                                   └─rsync──> web site mirror
HPC five finished climatology NCs  ──rsync──> local clone/products/climatology/
HPC products/release/              ──rsync──> public data location
```

Run `make mirror-local` in the local Git clone. It uses `~/jumount/data/slmet/model_data/jumacs` as the default source; set `JUMACS_HPC_ROOT` if the mount differs. It copies only the three grouped model climatologies and two compact extension NetCDF files for 1985–2014. `make mirror-web` and `make publish-data` still require non-empty destination environment variables. No mirror runs during builds or tests. GitHub hosts code and documentation; the HPC filesystem hosts complete scientific data and generated products. No GitHub Pages configuration is used.
