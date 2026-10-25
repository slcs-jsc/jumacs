# Quality control and evaluation

## Post-download audit

After a transfer and before enabling any processing, run the offline post-download audit. It reads only local raw files and manifests — no CEDA access, no re-planning, no downloads, no config changes — and opens at most a few small representative files per model (metadata plus tiny slices; no hashing, no full-field reads):

```bash
jumacs validate-raw --model all       # transfer integrity: completeness, .part strays, signatures, readability
jumacs coordinate-audit --model all   # conventions: grid, vertical coordinate, time, reader compatibility
jumacs post-download-audit --model all  # both stages in one run
```

`validate-raw` reports per model `ok`, `incomplete`, `invalid`, `not_downloaded` or `not_applicable` (manifest vs. disk counts, leftover `.part` files, NetCDF/HDF5 signatures, time decoding, size tolerance against manifest listing estimates) and writes `products/diagnostics/raw_validation/MODEL.json` plus `products/comparison/raw_validation.{csv,md}`. `coordinate-audit` classifies each model's observed horizontal grid (regular, curvilinear, already zonal, partial longitude), vertical coordinate (pressure levels, hybrid sigma-pressure/hybrid height with observed vs. inferred `formula_terms`, pure sigma, altitude) and time handling, and grades reader compatibility `supported_now`, `small_mapping_change`, `reader_extension_needed` or `unsupported` with machine-readable reasons and a `main_action` per model; outputs are `products/diagnostics/coordinate_audit/MODEL.json` and `products/comparison/coordinate_audit.{csv,md}`. Treat `incomplete`/`invalid` or any `reader_extension_needed` as a gate: fix the transfer or extend the reader before submitting zonal/climatology jobs. The audit is descriptive evidence; it never edits `config/models/*.yaml` or enables processing steps.

## Coverage and comparison diagnostics

`jumacs coverage --model MODEL` writes finite-value-based pressure limits, native geopotential-height ranges (where available), a representative median top height and level count per species to `products/diagnostics/coverage/MODEL/`. `jumacs compare --models GEOSCCM EMAC` writes original model fields, absolute and relative difference on a common pressure grid in the overlapping domain. Relative difference is `100 × (first − second) / second` and is undefined where the second field is zero. Interpolation is in log pressure and never extrapolates; `compare`/`evaluate` take their own query levels from `--pressure-hpa`, which is a report grid only and does not change the product axis — and because products are stored on native levels, that query grid is exactly where the two models are brought onto one vertical axis, with levels a model does not reach left missing there. `jumacs evaluate` accepts a separate evaluation NetCDF with the requested field and pressure coordinate, and produces difference, relative difference and RMS; it does not rank or adjust models. Plots from `quicklook`, `trends`, the model atlas and representative comparison maps are diagnostics only.


## Evaluation data

Satellite climatologies and observations are independent evaluation sources, never inputs to the primary JuMACS climatology. They must not fill gaps, correct model biases, normalize trends, or select a model automatically. Independent satellite comparisons use sources such as SPARC, SWOOSH, GOZCARDS, ACE-FTS, MIPAS and MLS; no satellite datasets are fetched by this project. Store generated machine-readable evaluation results under `products/comparison/evaluation/` and any browser plots under `site/`. The `jumacs evaluate` command compares compatible fields on an overlapping month, latitude and pressure domain. Retain each source's provenance and licence.
