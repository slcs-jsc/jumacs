# Published products and the data browser

JuMACS publishes two kinds of NetCDF products plus a static browser:

- **Native climatologies** — one CF-1.13 file per model and period in `products/climatology/MODEL/`, on the model's own vertical grid (see [methodology](methodology.md)).
- **Harmonized products** — one validated file per model in `products/harmonized/MODEL/`, on 5° area-weighted latitude bands and a 1 km altitude grid over the model's own validated vertical range, with no blending, padding or gap filling (see [methodology](methodology.md)).

Site generation copies the harmonized and native NetCDF files into `site/products/harmonized/` and `site/products/climatology/` so they publish with the browser; see the [publication workflow](publication.md).

## Generating the browser

Generate the site **on the HPC system**, where the harmonized NetCDF files are stored (`jumacs browse --start-year 1985 --end-year 2014`). The atlas renders every harmonized three-dimensional field as a January/April/July/October latitude–altitude cross section of the area-weighted 5° latitude bands, with altitude on the vertical axis and readable mixing-ratio units (ppmv, ppbv or pptv) where appropriate; the NetCDF units remain unchanged. The **Data availability** view lists every native and harmonized file with its coverage and a working download link, the **Products** view presents one card per harmonized file with its valid top and file size, the **Provenance** view lists the source datasets, and the **About** view states the methodology and release status. The command copies harmonized files into `site/products/harmonized/` and native climatology files into `site/products/climatology/`, and removes stale generated plots, copied products and any leftover legacy product directories. The generated `site/` tree has only relative links and is the complete publication unit.

## Data availability inventory

`jumacs inventory` inspects existing native and harmonized climatology products and writes a searchable data-availability overview to `products/catalog/climatology_inventory.{csv,json,md}`. `jumacs browse` refreshes the same inventory before generating the website's **Data availability** view; the Markdown lists the current contents without maintaining a second hand-written inventory here.
