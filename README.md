# JuMACS

**Jülich Multi-source Atmospheric Climatology and Statistics** provides monthly zonal atmospheric reference climatologies derived from internally consistent chemistry-climate simulations. The default period is **1985–2014 inclusive** (30 years), with `nominal_reference_year=2000` as an epoch label; it does not equate trend-sensitive concentrations with year-2000 abundances. JuMACS applies no detrending, normalization, bias correction, satellite adjustment, model weighting, or ensemble averaging.

## Published data

The reference climatologies, application-grid products and the static data browser are published at
[https://datapub.fz-juelich.de/slcs/jumacs/](https://datapub.fz-juelich.de/slcs/jumacs/) (Jülich Supercomputing Centre (JSC) DataPub).

## What JuMACS provides

- One CF-1.13 NetCDF climatology per model and period, on the model's own vertical grid, with `mean`, `sigma`, `minimum`, `maximum` and `n_years` on a climatological month axis.
- Application products remapped onto one project-wide pressure grid (124 levels, 100000 Pa to 0.002 Pa, uniform in log pressure), optionally extended upward with WACCM-X for upper-atmosphere work.
- A separate altitude-grid product from the older `compact` pipeline.
- A static browser with latitude cross sections, the annual cycle, time series, single-year maps and a data-availability view.
- Quality-control and diagnostics tooling: post-download audit, coverage matrices, model comparison and satellite evaluation.

## Data sources

Eight CCMI-2022 refD1 chemistry-climate models (GEOSCCM, EMAC, CCSR-NIES-MIROC32, CESM2-WACCM, CMAM, CNRM-MOCAGE, IPSL-CM6A-ATM-LR-REPROBUS, SOCOL) archived at CEDA, plus the WACCM-X transient simulation for the upper-atmosphere extension. Satellite datasets are evaluation data only and never inputs. Two archived models were excluded from the active dataset after data-quality checks; the records, licences and curation rationale are documented in [docs/data_sources.md](docs/data_sources.md).

## Documentation

Detailed documentation lives in [docs/](docs/README.md):

- [Data sources and curation](docs/data_sources.md)
- [Methodology](docs/methodology.md)
- [Published products and the data browser](docs/products.md)
- [Quality control and evaluation](docs/quality_control.md)
- [Development and reproducibility](docs/development.md)
- [Publication workflow](docs/publication.md)

## Repository and reproducibility

This repository is both the Git root and the HPC working directory: pipeline code, the source registry (`config/models/`), tests, Slurm scripts and documentation. The `jumacs` command runs the full workflow — inspect, download plans, zonal series, climatologies, application products, diagnostics and site generation — on JUWELS/JUDAC. Raw data, processed series, generated products and the site are excluded from Git; see [docs/development.md](docs/development.md).

## Status

JuMACS is under active development. The `v0.1` release marks the first published dataset; development continues in the repository.

## License and citation

The MIT license covers the project code. When using the data, cite the CEDA dataset records listed in the model configurations and follow their licences; see [CITATION.cff](CITATION.cff).

Contact: [l.hoffmann@fz-juelich.de](mailto:l.hoffmann@fz-juelich.de)
