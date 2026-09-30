# Project layout

The repository root is the HPC working directory. Run JuMACS commands there.

| Directory | Purpose | Git |
| --- | --- | --- |
| `src/`, `config/`, `tests/`, `docs/`, `scripts/` | Reproducible code, settings, tests and instructions | Track |
| `data/raw/` | Unmodified downloaded GEOSCCM, EMAC and WACCM-X files | Ignore |
| `data/processed/` | Native monthly zonal time series | Ignore |
| `products/` | Climatologies, comparisons, manifests and diagnostics | Ignore |
| `site/` | Generated static browser and plots | Ignore |
| `cache/`, `tmp/` | Logs, caches and temporary work | Ignore |

`products/release/` is reserved for files explicitly selected for publication. Nothing is copied there automatically.

```text
HPC site/                 ──rsync──> local site mirror
                          └─rsync──> web site mirror
HPC products/release/     ──rsync──> public data location
```

`make mirror-local`, `make mirror-web` and `make publish-data` require their respective environment variables. They refuse an empty destination before invoking `rsync`. They are not run by the build or test targets. GitHub hosts code and documentation; the HPC filesystem hosts complete scientific data and generated products. No GitHub Pages configuration is used.
