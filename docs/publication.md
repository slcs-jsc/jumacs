# Publication workflow

GitHub hosts code and documentation; the HPC filesystem hosts complete scientific data and generated products. No GitHub Pages configuration is used. The single publication tree is `site/`: the generated browser plus the downloadable NetCDF products it serves (`site/products/harmonized/` and `site/products/climatology/`).

```text
HPC site/ (browser + harmonized and native NCs) ──rsync/SSH──> local clone/site/ ──rsync──> web mirror
```

```bash
cd /p/data1/slmet/model_data/jumacs
jumacs inventory
jumacs browse --start-year 1985 --end-year 2014
# Back in the local Git checkout:
make mirror-local
JUMACS_WEB_SITE_MIRROR=user@datapub.fz-juelich.de:/absolute/path/ make mirror-web
```

`make mirror-local` pulls this complete `site/` tree, including its harmonized and native NetCDF downloads, into the local checkout's ignored `site/` with rsync over SSH. The source is `JUMACS_HPC_MIRROR`, by default `judac.fz-juelich.de:/p/data1/slmet/model_data/jumacs/site`. Run `make mirror-web` from that local checkout with `JUMACS_WEB_SITE_MIRROR=user@datapub.fz-juelich.de:/absolute/path/` set to the chosen web destination. It refuses a site without an `index.html` and the generated `products/harmonized/` tree. Both mirrors use `rsync --delete` and refuse unsafe destinations; no mirror runs during builds or tests. `make site` is a shortcut for `jumacs browse`. See [development and reproducibility](development.md) for the directory map.
