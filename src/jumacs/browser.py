"""Generate a self-contained static browser for harmonized climatologies."""

import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from html import escape
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import config

ZONAL_MONTHS = (1, 4, 7, 10)
MONTH_NAMES = {1: "January", 4: "April", 7: "July", 10: "October"}

HARMONIZED_NAME = re.compile(
    r"jumacs_(?P<model>[a-z0-9_-]+?)_(?P<start>\d{4})-(?P<end>\d{4})"
    r"_(?P<latitude>\d+)deg_(?P<altitude>\d+)km\.nc")

VARIABLE_DISPLAY_NAMES = {
    "surface_pressure": "Surface pressure",
    "temperature": "Temperature",
    "geopotential_height": "Geopotential height",
    "zonal_wind": "Zonal wind",
    "meridional_wind": "Meridional wind",
    "mean_age": "Mean age",
    "total_ozone_column": "Total ozone column",
    "tropospheric_ozone_column": "Tropospheric ozone column",
    "O3_strat": "Stratospheric ozone",
    "tropopause_pressure": "Tropopause pressure",
    "tropopause_temperature": "Tropopause temperature",
    "tropopause_altitude": "Tropopause altitude",
    "EP_flux_meridional": "Meridional EP flux",
    "EP_flux_vertical": "Vertical EP flux",
    "TEM_meridional_wind": "TEM meridional wind",
    "TEM_vertical_velocity": "TEM vertical velocity",
    "sulfate_SAD": "Sulfate aerosol surface area density",
    "PSC_NAT_SAD": "PSC nitric acid surface area density",
    "PSC_ice_SAD": "PSC ice surface area density",
}


def _display_name(canonical):
    """Public scientific label for a canonical variable (display-only)."""
    mapped = VARIABLE_DISPLAY_NAMES.get(canonical)
    if mapped is not None:
        return mapped
    spaced = canonical.replace("_", " ")
    return spaced[:1].upper() + spaced[1:]


def discover_harmonized_products(root, start_year, end_year, model=None):
    """Return existing harmonized products for the requested period."""
    base = Path(root) / "products" / "harmonized"
    folders = [base / model] if model else sorted(base.iterdir()) if base.exists() else []
    products = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob(f"*_{start_year}-{end_year}_*deg_*km.nc")):
            match = HARMONIZED_NAME.fullmatch(path.name)
            if match is not None:
                products.append((folder.name, path))
    if not products:
        raise FileNotFoundError(f"No harmonized products for {start_year}-{end_year} in {base}")
    return products


def _section_path(model, variable):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", variable)
    return Path("plots") / model / f"{safe_name}_section.png"


def _human_size(nbytes):
    value = float(nbytes)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000.0


def _display_scale(values, units):
    """Choose readable plot units without changing the stored product units."""
    if units in ("mol mol-1", "mol/mol"):
        finite = np.abs(values[np.isfinite(values)])
        peak = float(np.percentile(finite, 98)) if finite.size else 0.0
        return ((1e6, "ppmv") if peak >= 1e-6 else (1e9, "ppbv") if peak >= 1e-9
                else (1e12, "pptv") if peak >= 1e-12 else (1e15, "ppqv"))
    if units == "Pa":
        return 0.01, "hPa"
    if units == "m":
        return 0.001, "km"
    return 1.0, units


def _plot_section(values, altitude, latitudes, model, label, units, path):
    """Draw a January/April/July/October latitude-altitude cross section grid."""
    displayed = values[np.array(ZONAL_MONTHS) - 1]
    finite = displayed[np.isfinite(displayed)]
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        width = max(abs(low) * 0.01, 1e-12)
        low, high = low - width, high + width
    ceiling = float(altitude.max())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, month, panel in zip(axes.flat, ZONAL_MONTHS, displayed):
        image = ax.pcolormesh(latitudes, altitude, np.ma.masked_invalid(panel),
                              shading="auto", cmap="viridis", vmin=low, vmax=high,
                              rasterized=True)
        ax.set(title=MONTH_NAMES[month], xlim=(float(latitudes.min()), float(latitudes.max())),
               ylim=(0.0, ceiling))
    for ax in axes[-1]:
        ax.set_xlabel("Latitude (°N)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Altitude (km)")
    fig.suptitle(f"{model} · {label} · zonal climatology")
    fig.colorbar(image, ax=axes, label=units or "Value", shrink=0.85)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _product_card(model_name, path):
    """One download card per harmonized product, described by its own attributes."""
    with xr.open_dataset(path) as dataset:
        display = str(dataset.attrs.get("display_name") or model_name)
        period = str(dataset.attrs.get("climatology_period") or "")
        valid_top = dataset.attrs.get("model_valid_top_km")
        latitude_step = dataset.attrs.get("latitude_band_degrees", 5)
        altitude_step = dataset.attrs.get("altitude_step_km", 1)
    valid = f" · valid top {valid_top:g} km" if isinstance(valid_top, (int, float)) else ""
    latitude_text = f"{latitude_step:g}" if isinstance(latitude_step, (int, float)) else str(latitude_step)
    altitude_text = f"{altitude_step:g}" if isinstance(altitude_step, (int, float)) else str(altitude_step)
    card = [f'<h3>{escape(display)}</h3>',
            f'<p class="product-sub">Harmonized climatology · {escape(period.replace("-", "–"))}</p>',
            f'<p class="product-meta">12 months · {latitude_text}° latitude · {altitude_text} km altitude'
            f'{escape(valid)} · NetCDF · {_human_size(path.stat().st_size)}</p>',
             f'<a class="download" href="{escape(path.relative_to(path.parents[3]).as_posix(), quote=True)}"'
            f' download>Download NetCDF</a>',
            f'<p class="product-file"><code>{escape(path.name)}</code></p>']
    return f'<article class="product">{"".join(card)}</article>'


def _html(records, start_year, end_year, inventory_rows, provenance_rows=(), product_cards=(),
          native_map=None):
    native_map = native_map or {}
    catalog = []
    for model, product, fields in records:
        for field in fields:
            catalog.append({"model": model, "variable": field["native"],
                            "canonical": field["canonical"], "label": field["label"],
                            "display": _display_name(field["canonical"]), "units": field["units"],
                            "product": f"products/harmonized/{model}/{product.name}",
                            "native": native_map.get(model),
                            "views": {name: path.as_posix() for name, path in field["views"].items()}})
    payload = json.dumps(catalog, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    inventory_payload = json.dumps(inventory_rows, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    provenance_payload = json.dumps(provenance_rows, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    products_html = "".join(product_cards) or '<p>No harmonized products are published for this period yet.</p>'
    return (HTML_TEMPLATE.replace("__PERIOD__", f"{start_year}–{end_year}")
            .replace("__CATALOG__", payload).replace("__INVENTORY__", inventory_payload)
            .replace("__PROVENANCE__", provenance_payload)
            .replace("__PRODUCT_COUNT__", str(len(product_cards)))
            .replace("__PRODUCTS__", products_html))


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>JuMACS · Climatology atlas</title><link rel="stylesheet" href="assets/style.css"></head>
<body><header><h1>JuMACS climatology atlas</h1>
<p>JuMACS v0.1 · Initial public release · Harmonized multi-source model climatologies</p>
<nav><button id="atlas-nav" class="active">Plots</button><button id="inventory-nav">Data availability</button>
<button id="products-nav">Products</button>
<button id="provenance-nav">Data provenance</button>
<button id="about-nav">About &amp; contact</button></nav></header>
<main id="atlas"><aside><label for="model">Model / product</label><select id="model"></select>
<label for="search">Variable</label><input id="search" type="search" placeholder="Search variable name">
<div id="variables" aria-label="Available variables"></div></aside>
<section class="viewer"><div class="topline"><div><h2 id="title"></h2><p class="meta" id="meta"></p></div>
<div class="controls"><button id="prev" title="Previous variable">← Previous</button>
<button id="next" title="Next variable">Next →</button></div></div>
<figure><img id="plot" alt=""><figcaption><span id="caption"></span>
<span><a class="download" id="png" download>Download PNG</a>
<a class="download" id="product" download>Download NetCDF</a>
<a class="download" id="native" download>Download native NetCDF</a></span></figcaption></figure>
<p class="plot-note">January, April, July and October climatological means on the model's own altitude grid,
shown as latitude–altitude cross sections of area-weighted 5° latitude bands.
Each model keeps its own valid top: values above it are absent, not filled or extrapolated, and interior
gaps are preserved. Surface-only fields without an altitude dimension are available as NetCDF downloads
through the data-availability and product pages.</p>
</section></main><section id="inventory-view" hidden><h2>Data availability</h2>
<p>Availability covers the native monthly climatology products and the harmonized products.
Harmonized fields are area-weighted zonal means: their spatial axes are altitude × latitude or latitude only.
The altitude range lists the levels that hold any finite climatological mean; no usability threshold is
applied, and interior gaps are preserved. Products are never merged across models.
Click a column heading to sort (↕); ↑ and ↓ show the current order.</p>
<label for="inventory-search">Filter model or variable</label><input id="inventory-search" type="search" placeholder="Search inventory">
<div class="table-wrap"><table><thead><tr>
<th><button class="sort-button" data-sort="model">Model</button></th>
<th><button class="sort-button" data-sort="canonical_variable">Variable</button></th>
<th><button class="sort-button" data-sort="units">Units</button></th>
<th><button class="sort-button" data-sort="dimensionality">Grid</button></th>
<th><button class="sort-button" data-sort="native_available">Native</button></th>
<th><button class="sort-button" data-sort="harmonized_available">Harmonized</button></th>
<th><button class="sort-button" data-sort="top_altitude_km">Altitude range (km)</button></th>
</tr></thead><tbody id="inventory-body"></tbody></table></div></section>
<section id="products-view" hidden><h2>Harmonized climatology products</h2>
<div class="product-intro"><p>JuMACS publishes one harmonized climatology product per model.
Each product holds twelve climatological months on 5° area-weighted latitude bands and a 1 km altitude grid,
spanning the surface to the model's own valid top as derived from its data. The current public set contains
__PRODUCT_COUNT__ products for __PERIOD__.</p>
<p>Products are independent per model: there is no merging or blending between models, no donor extension,
no bias correction and no gap filling. Missing values remain missing. The
<a href="https://github.com/slcs-jsc/jumacs/blob/main/docs/methodology.md">methodology</a> documents how
each native climatology is placed on altitude and latitude bands.</p></div>
<div class="products-grid">__PRODUCTS__</div>
<p class="products-note">Native-model climatology files remain available through the data-availability view;
this page focuses on the harmonized products intended for convenient download and use.</p></section>
<section id="provenance-view" hidden><h2>Data provenance</h2>
<p>Every source dataset JuMACS processes, with its archive, citation identifiers, licence and the volume of
raw archive data mirrored for the configured reference period. Sizes are decimal units (1 GB = 10^9 bytes)
of the regular files held under each source directory, so they describe this installation rather than the
full archive. Identifier links point to a registered DOI where one exists and otherwise to the stable
archive record; a dataset UUID covers the archive collection or dataset as noted.</p>
<div id="provenance-list"></div>
</section>
<section id="about-view" hidden><h2>About JuMACS</h2>
<p>JuMACS — Jülich Multi-source Atmospheric Climatology and Statistics — provides reproducible monthly
atmospheric climatologies and associated statistics for scientific applications.</p>
<p>JuMACS harmonizes published chemistry–climate model climatologies from multiple sources onto common
representations while keeping every source-model product independent. Each harmonized product preserves the
original model climatology and its provenance on 5° area-weighted latitude bands and a 1 km altitude grid up
to the model's own valid top. Products are never merged, blended or bias corrected across models.</p>
<p>The project provides native-model climatologies, harmonized products, diagnostic visualizations, and
detailed source-data provenance. Original source datasets remain subject to their respective licences and citation
requirements.</p>
<p class="disclaimer">JuMACS is under active development. The public site provides the current model-based
climatologies, harmonized products, diagnostic atlas, and source-data provenance. Additional datasets, evaluation
results, and functionality may be added in future releases.</p>
<h2>Contact</h2>
<address class="contact">Dr. Lars Hoffmann<br>Jülich Supercomputing Centre (JSC)<br>Forschungszentrum Jülich<br>Germany<br>
<a href="mailto:l.hoffmann@fz-juelich.de">l.hoffmann@fz-juelich.de</a></address>
<p>Feedback and scientific collaboration are very welcome.</p>
<h2>Project links</h2>
<ul class="project-links">
<li><a href="https://datapub.fz-juelich.de/slcs/jumacs/">Published data</a></li>
<li><a href="https://github.com/slcs-jsc/jumacs">Source code on GitHub</a></li>
</ul>
</section>
<script id="catalog" type="application/json">__CATALOG__</script>
<script id="inventory-data" type="application/json">__INVENTORY__</script>
<script id="provenance-data" type="application/json">__PROVENANCE__</script>
<script>
const catalog = JSON.parse(document.getElementById('catalog').textContent);
const inventory = JSON.parse(document.getElementById('inventory-data').textContent);
const provenance = JSON.parse(document.getElementById('provenance-data').textContent);
const model = document.getElementById('model');
const search = document.getElementById('search');
const list = document.getElementById('variables');
for (const name of [...new Set(catalog.map(row => row.model))]) {
  const option = document.createElement('option'); option.value = name; option.textContent = name; model.append(option);
}
let current = null;
function available() {
  const query = search.value.toLowerCase();
  return catalog.filter(row => row.model === model.value &&
    (row.display + ' ' + row.canonical + ' ' + row.variable).toLowerCase().includes(query));
}
function render() {
  const rows = available();
  if (!rows.includes(current)) current = rows[0] || null;
  list.replaceChildren();
  for (const row of rows) {
    const button = document.createElement('button');
    button.className = 'item' + (row === current ? ' active' : '');
    button.textContent = row.display;
    button.onclick = () => { current = row; render(); };
    list.append(button);
  }
  if (!current) {
    document.getElementById('title').textContent = 'No matching variable';
    document.getElementById('plot').removeAttribute('src');
    return;
  }
  document.getElementById('title').textContent = current.model + ' · ' + current.display;
  document.getElementById('meta').textContent = current.canonical + ' · ' + current.units + ' · __PERIOD__';
  const image = document.getElementById('plot');
  image.src = current.views.section;
  image.alt = current.model + ' ' + current.display + ' cross sections';
  document.getElementById('png').href = current.views.section;
  document.getElementById('product').href = current.product;
  const native = document.getElementById('native');
  native.hidden = !current.native;
  if (current.native) native.href = current.native;
  document.getElementById('caption').textContent =
    'Monthly climatology · four latitude–altitude cross sections';
}
model.onchange = () => {
  const canonical = current && current.canonical;
  const match = catalog.find(row => row.model === model.value && row.canonical === canonical);
  if (match && search.value && !available().includes(match)) search.value = '';
  current = match || null;
  render();
};
search.oninput = () => { current = null; render(); };
function move(step) {
  const rows = available(), index = rows.indexOf(current);
  if (rows.length) { current = rows[(index + step + rows.length) % rows.length]; render(); }
}
document.getElementById('prev').onclick = () => move(-1);
document.getElementById('next').onclick = () => move(1);
document.onkeydown = event => {
  if (event.target.matches('input,select')) return;
  if (event.key === 'ArrowLeft') move(-1);
  if (event.key === 'ArrowRight') move(1);
};
const inventoryCollator = new Intl.Collator(undefined, {numeric: true, sensitivity: 'base'});
let inventorySort = null, inventoryDescending = false;
function compareInventory(a, b, key) {
  const left = a[key], right = b[key];
  if (left == null) return right == null ? 0 : 1;
  if (right == null) return -1;
  if (typeof left === 'string') return inventoryCollator.compare(left, right);
  if (typeof left === 'boolean') return Number(left) - Number(right);
  return Number(left) - Number(right);
}
function renderInventory() {
  const query = document.getElementById('inventory-search').value.toLowerCase();
  const body = document.getElementById('inventory-body');
  body.replaceChildren();
  const rows = inventory.filter(row =>
    (row.model + ' ' + (row.canonical_variable || '') + ' ' + row.native_variable)
      .toLowerCase().includes(query));
  rows.sort((a, b) => {
    const primary = compareInventory(a, b, inventorySort);
    if (primary) return a[inventorySort] == null || b[inventorySort] == null ? primary :
      inventoryDescending ? -primary : primary;
    return compareInventory(a, b, 'model') ||
      compareInventory(a, b, 'canonical_variable') || compareInventory(a, b, 'native_variable');
  });
  for (const row of rows) {
    const tr = document.createElement('tr');
    const altitude = row.bottom_altitude_km == null ? '' :
      `${row.bottom_altitude_km.toPrecision(3)}–${row.top_altitude_km.toPrecision(3)}`;
    const values = [row.model, row.canonical_variable || '—', row.units, row.dimensionality || '—',
      row.native_product ? 'yes' : 'no', row.harmonized_product ? 'yes' : 'no', altitude];
    for (const [index, value] of values.entries()) {
      const cell = document.createElement('td');
      const link_to = index === 4 ? row.native_product : index === 5 ? row.harmonized_product : null;
      if (link_to) {
        const link = document.createElement('a');
        link.href = link_to;
        link.textContent = 'NetCDF';
        cell.append(link);
      } else cell.textContent = value;
      tr.append(cell);
    }
    body.append(tr);
  }
}
document.getElementById('inventory-search').oninput = renderInventory;
for (const button of document.querySelectorAll('.sort-button')) {
  const label = button.textContent;
  button.onclick = () => {
    inventoryDescending = inventorySort === button.dataset.sort && !inventoryDescending;
    inventorySort = button.dataset.sort;
    for (const heading of document.querySelectorAll('.sort-button')) {
      const active = heading.dataset.sort === inventorySort;
      heading.parentElement.setAttribute('aria-sort', active ?
        inventoryDescending ? 'descending' : 'ascending' : 'none');
      const direction = active ? inventoryDescending ? 'descending' : 'ascending' : 'sortable';
      heading.textContent = heading.dataset.label + (active ? inventoryDescending ? ' ↓' : ' ↑' : ' ↕');
      heading.setAttribute('aria-label', heading.dataset.label + ', ' + direction);
    }
    renderInventory();
  };
  button.dataset.label = label;
  button.textContent = label + ' ↕';
  button.setAttribute('aria-label', label + ', sortable');
}
document.querySelector('.sort-button[data-sort="model"]').click();
const views = [['atlas-nav', 'atlas'], ['inventory-nav', 'inventory-view'],
               ['products-nav', 'products-view'],
               ['provenance-nav', 'provenance-view'], ['about-nav', 'about-view']];
for (const [button, active] of views) {
  document.getElementById(button).onclick = () => {
    for (const [, id] of views) document.getElementById(id).hidden = id !== active;
    for (const [name, target] of views)
      document.getElementById(name).classList.toggle('active', target === active);
  };
}
function renderProvenance() {
  const list = document.getElementById('provenance-list');
  list.textContent = '';
  const text = value => value === null || value === undefined || value === '' ? 'unknown' : String(value);
  const meta = (term, value) => {
    const pair = document.createElement('div');
    const dt = document.createElement('dt'); dt.textContent = term;
    const dd = document.createElement('dd'); dd.textContent = value;
    pair.append(dt, dd); return pair;
  };
  const sorted = [...provenance].sort((a, b) =>
    a.role.localeCompare(b.role) || a.dataset.localeCompare(b.dataset));
  for (const row of sorted) {
    const card = document.createElement('article'); card.className = 'dataset';
    const head = document.createElement('div'); head.className = 'dataset-head';
    const title = document.createElement('h3'); title.textContent = text(row.dataset);
    const sub = document.createElement('p'); sub.className = 'dataset-sub';
    sub.textContent = `${text(row.institution)} · ${text(row.role)}`;
    head.append(title, sub);
    const dl = document.createElement('dl'); dl.className = 'dataset-meta';
    dl.append(meta('Experiment · member', row.member ? `${row.experiment} · ${row.member}` : text(row.experiment)));
    if (row.period_used) dl.append(meta('Period used', text(row.period_used)));
    if (row.source_archive) dl.append(meta('Source', text(row.source_archive)));
    if (row.doi || row.persistent_identifier) {
      const pair = document.createElement('div');
      const dt = document.createElement('dt'); dt.textContent = 'Identifier';
      const dd = document.createElement('dd');
      const link = document.createElement('a');
      if (row.doi) { link.href = 'https://doi.org/' + row.doi; link.textContent = row.doi; }
      else {
        link.href = row.persistent_identifier;
        const ceda = row.persistent_identifier.includes('catalogue.ceda.ac.uk');
        const collection = (row.notes || '').includes('collection-level');
        link.textContent = ceda ? (collection ? 'CEDA collection record' : 'CEDA dataset record')
                                : 'Source record';
      }
      dd.append(link); pair.append(dt, dd); dl.append(pair);
    }
    if (row.license) dl.append(meta('License', text(row.license)));
    if (row.raw_data_used_human) dl.append(meta('Raw data used', text(row.raw_data_used_human)));
    if (row.accessed) dl.append(meta('Accessed', text(row.accessed)));
    if (row.notes) dl.append(meta('Notes', row.notes));
    card.append(head, dl); list.append(card);
  }
}
renderProvenance();
renderInventory();
render();
</script></body></html>
"""


STYLE = """:root{--ink:#183349;--muted:#617686;--line:#dce7ec;--paper:#f5f9fa;--accent:#087f89}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
header{background:#12364b;color:white;padding:1.2rem max(1.5rem,calc((100vw - 1420px)/2))}
header h1{margin:0;font-size:1.6rem}header p{margin:.3rem 0 0;color:#cee2e9;font-size:.92rem}
header nav{display:flex;gap:.5rem;margin-top:.7rem}header nav button.active{background:var(--accent);color:white}
main{max-width:1420px;min-height:calc(100vh - 98px);margin:auto;display:grid;grid-template-columns:280px minmax(0,1fr)}
main[hidden],#inventory-view[hidden],#products-view[hidden],#provenance-view[hidden],#about-view[hidden]{display:none}
#inventory-view,#products-view,#provenance-view,#about-view{max-width:1420px;margin:auto;padding:1.5rem}
#about-view{max-width:60rem}
#about-view p{max-width:75ch}
#about-view h2:not(:first-child){margin-top:1.6rem}
#about-view a{color:var(--accent)}
#about-view address.contact{font-style:normal;line-height:1.6;margin:0}
.project-links{margin:.6rem 0 0;padding-left:1.2rem;display:grid;gap:.45rem;max-width:75ch}
#products-view p{max-width:90ch}
.product-intro{background:white;border:1px solid var(--line);border-radius:.8rem;padding:.4rem 1.2rem;margin-top:1rem}
.product-intro p{margin:.7rem 0}
.product-intro a{color:var(--accent)}
.products-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(21rem,1fr));gap:.9rem;margin-top:1.1rem}
.product{background:white;border:1px solid var(--line);border-radius:.8rem;padding:1rem 1.2rem;display:grid;gap:.45rem;align-content:start}
.product h3{margin:0;font-size:1.05rem;overflow-wrap:anywhere}
.product-sub{margin:0;color:var(--muted);font-size:.85rem}
.product-meta{margin:0;color:var(--muted);font-size:.85rem}
.product a.download{justify-self:start}
.product-file{margin:0;color:var(--muted);font-size:.78rem;overflow-wrap:anywhere}
.products-note{margin:1.1rem 0 0;color:#3f6472;font-size:.86rem;max-width:90ch}
#inventory-search{max-width:28rem}.table-wrap{overflow:auto;margin-top:1rem}
#provenance-list{display:grid;gap:.9rem;margin-top:1.1rem}
.dataset{background:white;border:1px solid var(--line);border-radius:.8rem;padding:1rem 1.2rem;display:grid;grid-template-columns:minmax(11rem,16rem) minmax(0,1fr);gap:.5rem 1.6rem;align-items:start}
.dataset-head h3{margin:0;font-size:1.05rem}
.dataset-sub{margin:.25rem 0 0;color:var(--muted);font-size:.85rem}
.dataset-meta{display:grid;grid-template-columns:auto minmax(0,1fr);gap:.3rem 1rem;margin:0}
.dataset-meta>div{display:contents}
.dataset-meta dt{color:var(--muted);font-size:.78rem;font-weight:750;text-transform:uppercase;padding-top:.1rem}
.dataset-meta dd{margin:0;font-size:.9rem;overflow-wrap:anywhere}
.dataset-meta a{color:var(--accent)}
table{width:100%;border-collapse:collapse;background:white;font-size:.9rem}
th,td{padding:.55rem .7rem;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:#e9f4f5;position:sticky;top:0}td a{color:#087f89}
th button.sort-button{border:0;background:transparent;padding:0;color:inherit;font:inherit;font-weight:700;text-align:left;white-space:nowrap;cursor:pointer}
th button.sort-button:hover,th button.sort-button:focus-visible{color:var(--accent);text-decoration:underline}
aside{background:white;border-right:1px solid var(--line);padding:1.2rem}
label{display:block;margin:.75rem 0 .3rem;color:var(--muted);font-size:.78rem;font-weight:750;text-transform:uppercase}
select,input{width:100%;border:1px solid #b8cbd4;border-radius:.5rem;background:white;padding:.65rem .75rem;color:var(--ink);font:inherit}
#variables{display:grid;gap:.2rem;max-height:calc(100vh - 290px);overflow:auto;margin-top:.7rem}
.item{border:0;border-radius:.5rem;background:transparent;color:var(--ink);cursor:pointer;font:inherit;padding:.5rem .65rem;text-align:left}
.item:hover{background:#edf6f7}.item.active{background:#dff0f1;color:#075e65;font-weight:700}
.viewer{min-width:0;padding:1.35rem 1.6rem 2rem}.topline{display:flex;justify-content:space-between;align-items:start;gap:1rem;flex-wrap:wrap}
h2{margin:0;font-size:1.5rem}.meta{color:var(--muted);font-size:.9rem;margin:.2rem 0 0}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:.4rem}
button:not(.item),a.download{border:1px solid #bfcfd6;border-radius:.48rem;background:white;padding:.55rem .8rem;color:var(--ink);font:inherit;cursor:pointer;text-decoration:none}
a.download[hidden]{display:none}
button:hover:not(:disabled),a.download:hover{border-color:var(--accent)}
figure{background:white;border:1px solid var(--line);border-radius:.8rem;margin:1.2rem 0 .6rem;padding:.65rem;box-shadow:0 8px 28px #12364b0a}
figure img{display:block;width:100%;max-height:calc(100vh - 220px);object-fit:contain}
figcaption{display:flex;justify-content:space-between;align-items:center;gap:1rem;color:var(--muted);font-size:.88rem;margin-top:.7rem}
figcaption span:last-child{display:flex;gap:.4rem;flex-wrap:wrap}
.plot-note{margin:.15rem 0 0;color:#3f6472;font-size:.86rem;max-width:90ch}
@media(max-width:760px){main{display:block}aside{border-right:0;border-bottom:1px solid var(--line)}
#variables{display:flex;max-height:none;overflow:auto}.item{white-space:nowrap}.viewer{padding:1rem}figure img{max-height:none}
.dataset{grid-template-columns:minmax(0,1fr)}.products-grid{grid-template-columns:minmax(0,1fr)}}
"""


def _render_model(model_name, product, root, staging, native_rel):
    """Render one model in its own staging folder; safe to run in a worker process."""
    fields = []
    skipped = []
    model_config = config.load_config(model_name)
    inverse = {native: canonical for canonical, native in model_config["variables"].items()}
    with xr.open_dataset(product) as dataset:
        if ("month" not in dataset.coords or "altitude" not in dataset.coords
                or "lat" not in dataset.coords or dataset.coords["month"].size != 12):
            skipped.append((model_name, product.name, "not a 12-month month×altitude×lat climatology"))
            return (model_name, product, fields), skipped, native_rel
        altitude = np.asarray(dataset.altitude.values, dtype=float)
        latitudes = np.asarray(dataset.lat.values, dtype=float)
        for name in sorted(dataset.data_vars):
            if not name.endswith("_mean"):
                continue
            field = dataset[name]
            if field.dims != ("month", "altitude", "lat") or field.sizes["month"] != 12:
                skipped.append((model_name, name, "not a 12-month month×altitude×lat field"))
                continue
            values = np.asarray(field.values, dtype=float)
            if not np.isfinite(values[np.array(ZONAL_MONTHS) - 1]).any():
                skipped.append((model_name, name, "no finite values in displayed months"))
                continue
            stem = name[:-5]
            canonical = str(field.attrs.get("canonical_variable") or inverse.get(stem, stem))
            label = canonical.replace("_", " ")
            units = str(field.attrs.get("units", ""))
            scale, plot_units = _display_scale(values, units)
            path = _section_path(model_name, stem)
            (staging / path).parent.mkdir(parents=True, exist_ok=True)
            _plot_section(values * scale, altitude, latitudes, model_name,
                          _display_name(canonical), plot_units, staging / path)
            fields.append({"native": stem, "canonical": canonical, "label": label,
                           "units": units, "views": {"section": path}})
    copy = staging / "products" / "harmonized" / model_name / product.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(product, copy)
    return (model_name, product, fields), skipped, native_rel


def _publish(staging, site, name):
    source = staging / name
    target = site / name
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    elif target.exists() or target.is_symlink():
        target.unlink()
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)


def build_site(start_year=None, end_year=None, *, model=None, root=None, workers=4):
    """Plot existing harmonized products and publish a deployable tree below ``root/site``."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    root = config.ROOT if root is None else Path(root)
    period = config.reference_period()["reference_period"]
    start_year = period["start_year"] if start_year is None else start_year
    end_year = period["end_year"] if end_year is None else end_year
    products = discover_harmonized_products(root, start_year, end_year, model)
    from .inventory import build_inventory
    from .provenance import build_provenance
    inventory = build_inventory(start_year, end_year, model=model, root=root)
    provenance = build_provenance(root=root)
    native_map = {}
    for row in inventory["rows"]:
        if row["native_product"] and row["model"] not in native_map:
            native_map[row["model"]] = row["native_product"]
    site = root / "site"
    site.mkdir(parents=True, exist_ok=True)
    staging = site / ".browse-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    records = []
    skipped = []
    try:
        tasks = [(name, product, root, staging, native_map.get(name)) for name, product in products]
        if workers == 1 or len(tasks) == 1:
            results = [_render_model(*task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                     mp_context=get_context("spawn")) as pool:
                results = list(pool.map(_render_model, *zip(*tasks)))
        for record, model_skipped, _ in results:
            records.append(record)
            skipped.extend(model_skipped)
        assets = staging / "assets"
        assets.mkdir()
        (assets / "style.css").write_text(STYLE, encoding="utf-8")
        site_inventory = []
        for row in inventory["rows"]:
            entry = dict(row)
            if entry["native_product"] and not (root / entry["native_product"]).is_file():
                entry["native_product"] = None
            site_inventory.append(entry)
        published_native = set()
        for row in site_inventory:
            rel = row.get("native_product")
            if not rel or rel in published_native:
                continue
            source = root / rel
            if not source.is_file():
                continue
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            published_native.add(rel)
        product_cards = [_product_card(name, path) for name, path in products]
        (staging / "index.html").write_text(
            _html(records, start_year, end_year, site_inventory, provenance["sources"],
                  product_cards, native_map),
            encoding="utf-8")
        for name in ("plots", "plots-ready", "assets", "index.html",
                     "products/harmonized", "products/climatology"):
            _publish(staging, site, name)
        for stale in ("products/application", "products/combined", "products/application-ready"):
            old = site / stale
            if old.is_dir() and not old.is_symlink():
                shutil.rmtree(old)
            elif old.exists() or old.is_symlink():
                old.unlink()
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"site": site, "models": [name for name, _, _ in records],
            "provenance_sources": len(provenance["sources"]),
            "variables": {name: len(fields) for name, _, fields in records},
            "png_files": sum(len(field["views"]) for _, _, fields in records for field in fields),
            "netcdf_files": len(records), "climatology_files": len(published_native),
            "product_files": [path.name for _, path in products], "skipped": skipped}
