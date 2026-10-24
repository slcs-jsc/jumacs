# Published products and the data browser

JuMACS publishes four kinds of NetCDF products plus a static browser:

- **Native climatologies** — one CF-1.13 file per model and period in `products/climatology/MODEL/`, on the model's own vertical grid (see [methodology](methodology.md)).
- **Application products** — one validated file per model in `products/application/MODEL/`, remapped onto the common 124-level pressure grid and optionally extended upward with WACCM-X.
- **Combined compact products** — separate altitude-grid files in `products/climatology/combined/` from the older `compact` pipeline.
- **Application-ready products** — altitude-grid files in `products/application-ready/` that extend a combined compact product to a finite 0–120 km column, completing the upper tail with IG2 v3.1 standard atmospheres and edge-filling the remaining gaps (see [methodology](methodology.md)).

Site generation copies the application, combined and application-ready products into `site/products/application/`, `site/products/combined/` and `site/products/application-ready/` so they publish with the browser; see the [publication workflow](publication.md).

## Generating the browser

Generate the atlas **on the HPC system**, where the application NetCDF files, monthly zonal series and raw geographical fields are stored. The page offers January/April/July/October cross sections, the twelve-month climatological annual cycle, and the original multi-year monthly time series where available. Line plots compare five area-weighted latitude bands at 1000, 100, 10 and 1 hPa; monthly time series use four vertically stacked panels. Geographical maps use the four monthly raw fields from the single example year 2000, at those same pressures where native data permit, or at the surface for 2D fields. They are labelled as single-year examples, not 1985–2014 climatological means. Plot axes use readable mixing-ratio units (ppmv, ppbv or pptv) where appropriate; NetCDF units remain unchanged. The command copies application NetCDF files into `site/products/application/` and the combined compact products into `site/products/combined/`. Any application-ready product is published under a dedicated **Application-ready** view with its download, a latitude-altitude cross section for every three-dimensional field (drawn into `site/plots-ready/`) and its IG2 completion provenance. The generated `site/` tree has only relative links and is the complete publication unit. Re-running the command replaces generated plots and copied products, removing stale entries.

## Data availability inventory

`jumacs inventory` inspects existing native and application climatologies and writes a searchable data-availability overview to `products/catalog/climatology_inventory.{csv,json,md}`. `jumacs browse` refreshes the same inventory before generating the website's **Data availability** view; the Markdown lists the current contents without maintaining a second hand-written inventory here.
