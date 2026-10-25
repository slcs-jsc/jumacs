# Published products and the data browser

JuMACS harmonizes climatologies; it does not merge them. The published product is one validated harmonized NetCDF file per model, and the website presents those files model by model.

- **Harmonized products (published)** — one validated file per model in `products/harmonized/MODEL/`, on 5° area-weighted latitude bands and a 1 km altitude grid over the model's own validated vertical range, with no blending, padding or gap filling (see [methodology](methodology.md)).
- **Native climatologies (intermediate)** — one CF-1.13 file per model and period in `products/climatology/MODEL/`, on the model's own vertical grid. These remain on the processing filesystem as pipeline intermediates and evidence of provenance; they are not copied into the site and are not site downloads.

Site generation copies only the harmonized NetCDF files into `site/products/harmonized/` so they publish with the browser; see the [publication workflow](publication.md).

## Generating the browser

Generate the site **on the HPC system**, where the harmonized NetCDF files are stored (`jumacs browse --start-year 1985 --end-year 2014`). The site has four views — **Overview**, **Models**, **About** and **Methodology**:

- The **Overview** is one table of every harmonized product: model, covered period, variable count, valid top (WACCM-X is shown as capped at 150 km with its source top of 478.842 km stated), file size, a Browse action and a NetCDF download.
- **Models** drills from a model into its variables; each variable renders a January/April/July/October grid of latitude–altitude cross sections of the area-weighted 5° latitude bands on one shared color scale. Scaling is variable-specific: long-lived tracers such as N2O, CH4, CO2, CFCs and related halocarbons use linear color scales for easier interpretation of spatial gradients; logarithmic scales are reserved for selected short-lived or strongly stratified species with genuinely large dynamic ranges, with altitude in km up to that model's own top and readable mixing-ratio units (ppmv, ppbv or pptv) where appropriate; the NetCDF units remain unchanged. Every variable carries its harmonized product as a NetCDF download and an expandable metadata block with its provenance and vertical-extent attributes (`source_valid_bottom_km`, `source_valid_top_km`, `lower_completion_km`, `upper_completion_km`).
- **About** states the harmonizes-not-merges scope, licensing and citation; **Methodology** summarizes the vertical rules and links to this documentation.

Variables a model does not have are simply omitted; nothing is padded from another model. The command removes stale generated plots, copied products and any leftover legacy product directories, and the generated `site/` tree has only relative links and is the complete publication unit.

## Product inventory

`jumacs inventory` remains a command-line tool: it inspects existing native and harmonized climatology products and writes an overview to `products/catalog/climatology_inventory.{csv,json,md}` for operators. It is no longer part of the site; the Overview table is generated directly from the harmonized products on disk.
