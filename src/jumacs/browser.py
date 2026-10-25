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

EXTENT_KEYS = ("source_valid_bottom_km", "source_valid_top_km",
               "lower_completion_km", "upper_completion_km")

STALE_PATHS = ("products/application", "products/combined", "products/application-ready",
               "products/climatology", "plots-ready", "compact_catalog.json",
               "comparison_catalog.json")


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


def _color_scale(displayed):
    """Log scale for strictly positive fields spanning at least two decades."""
    finite = displayed[np.isfinite(displayed)]
    if finite.size and float(finite.min()) > 0.0:
        if float(finite.max()) / float(finite.min()) >= 100.0:
            return "log"
    return "linear"


def _plot_section(values, altitude, latitudes, model, label, units, path):
    """Draw a January/April/July/October latitude-altitude cross section grid."""
    displayed = values[np.array(ZONAL_MONTHS) - 1]
    finite = displayed[np.isfinite(displayed)]
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        width = max(abs(low) * 0.01, 1e-12)
        low, high = low - width, high + width
    norm = (matplotlib.colors.LogNorm(vmin=max(low, np.finfo(float).tiny), vmax=high)
            if _color_scale(displayed) == "log"
            else matplotlib.colors.Normalize(vmin=low, vmax=high))
    ceiling = float(altitude.max())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, month, panel in zip(axes.flat, ZONAL_MONTHS, displayed):
        image = ax.pcolormesh(latitudes, altitude, np.ma.masked_invalid(panel),
                              shading="auto", cmap="viridis", norm=norm, rasterized=True)
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


def _model_info(model, path, variables):
    with xr.open_dataset(path) as dataset:
        attrs = dataset.attrs
    info = {"model": model,
            "display": str(attrs.get("display_name") or model),
            "period": str(attrs.get("climatology_period") or ""),
            "institution": str(attrs.get("institution") or ""),
            "source_dataset": str(attrs.get("source_dataset") or ""),
            "variables": variables,
            "size_bytes": path.stat().st_size,
            "size_human": _human_size(path.stat().st_size),
            "product": f"products/harmonized/{model}/{path.name}",
            "product_file": path.name,
            "valid_top_km": attrs.get("model_valid_top_km"),
            "source_top_km": attrs.get("source_valid_top_km"),
            "extent_policy": str(attrs.get("vertical_extent_policy") or ""),
            "lower_completion": str(attrs.get("lower_boundary_completion") or ""),
            "upper_completion": str(attrs.get("upper_boundary_completion") or "")}
    return info


def _overview_row(info):
    valid_top = info.get("valid_top_km")
    top = f"{valid_top:g} km" if isinstance(valid_top, (int, float)) else "—"
    if info.get("source_top_km"):
        top += f" (capped; source top {info['source_top_km']:g} km)"
    period = info.get("period", "").replace("-", "–")
    return (f'<tr><td><strong>{escape(info.get("display") or info["model"])}</strong><br>'
            f'<code class="file">{escape(info.get("product_file", ""))}</code></td>'
            f'<td>{escape(period)}</td><td class="num">{info.get("variables", 0)}</td>'
            f'<td>{escape(top)}</td><td class="num">{escape(info.get("size_human", ""))}</td>'
            f'<td class="actions"><button class="browse-button" data-model="{escape(info["model"])}">'
            f'Browse</button> <a class="download" href="{escape(info.get("product", "#"), quote=True)}"'
            f' download>NetCDF</a></td></tr>')


def _html(models, records, start_year, end_year):
    catalog = []
    for model, product, fields in records:
        for field in fields:
            catalog.append({"model": model, "variable": field["native"],
                            "canonical": field["canonical"], "display": field["display"],
                            "units": field["units"], "source": field["source"],
                            "product": f"products/harmonized/{model}/{product.name}",
                            "views": {name: path.as_posix() for name, path in field["views"].items()},
                            "meta": field["meta"]})

    def embed(value):
        return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")

    overview_rows = "".join(_overview_row(info) for info in models)
    return (HTML_TEMPLATE.replace("__PERIOD__", f"{start_year}–{end_year}")
            .replace("__CATALOG__", embed(catalog))
            .replace("__MODELS__", embed(models))
            .replace("__MODEL_COUNT__", str(len(models)))
            .replace("__OVERVIEW_ROWS__", overview_rows))


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>JuMACS · Harmonized model climatology browser</title><link rel="stylesheet" href="assets/style.css"></head>
<body><header><h1>JuMACS harmonized model climatology browser</h1>
<p>JuMACS v0.1 · Initial public release · Harmonized multi-source model climatologies</p>
<nav><button id="overview-nav" class="active">Overview</button><button id="models-nav">Models</button>
<button id="methodology-nav">Methodology</button>
<button id="about-nav">About</button></nav></header>
<section id="overview"><div class="prose">
<h2>Overview</h2>
<p>JuMACS harmonizes published chemistry–climate model climatologies onto common grids and units.
It does not merge them: every model keeps an independent product with its own variables and its own
valid altitude extent. The current public set holds __MODEL_COUNT__ harmonized products for __PERIOD__,
each on 12 climatological months, 5° area-weighted latitude bands and a 1 km altitude grid.
Select a model to browse its variables, or download a product directly.</p>
<div class="table-wrap"><table><thead><tr>
<th>Model / product file</th><th>Period</th><th>Variables</th><th>Valid top</th><th>Size</th><th>Access</th>
</tr></thead><tbody>__OVERVIEW_ROWS__</tbody></table></div>
<p class="note">A model's valid top is the highest altitude its own data support, read from the product
metadata. WACCM-X source data extend far above 150 km; its published product is deliberately capped at
150 km and is not presented with its native top.</p>
</div></section>
<main id="models" hidden><aside><label for="model">Model / product</label><select id="model"></select>
<label for="search">Variable</label><input id="search" type="search" placeholder="Search variable name">
<div id="variables" aria-label="Available variables"></div></aside>
<section class="viewer"><div class="topline"><div><h2 id="title"></h2><p class="meta" id="meta"></p></div>
<div class="controls"><button id="prev" title="Previous variable">← Previous</button>
<button id="next" title="Next variable">Next →</button></div></div>
<figure><img id="plot" alt=""><figcaption><span id="caption"></span>
<span><a class="download" id="png" download>Download PNG</a>
<a class="download" id="product" download>Download NetCDF</a></span></figcaption></figure>
<p class="plot-note">January, April, July and October climatological means as latitude–altitude cross
sections on the model's own altitude grid up to its valid top, with one common color scale across the
four panels (logarithmic where a strictly positive field spans several decades).
Profile edges are completed by a constant carry of the nearest valid value; interior gaps are preserved
as missing and never filled, bridged or extrapolated. Variables a model does not provide are simply
absent from its list.</p>
<details id="metadata-view"><summary>Product and vertical-extent metadata</summary>
<dl id="metadata" class="dataset-meta"></dl></details>
</section></main>
<section id="methodology-view" hidden><div class="prose">
<h2>Methodology</h2>
<p>JuMACS harmonizes, and never merges. Each model's native monthly zonal-mean climatology is placed on
a common representation independently of all other models; nothing is blended, interpolated between
models or bias corrected.</p>
<ul>
<li><strong>Grid:</strong> 12 climatological months, 5° area-weighted latitude bands, 1 km altitude
levels from the surface to the model's own valid top.</li>
<li><strong>Valid extent:</strong> each variable keeps the vertical extent its source data support; the
model valid top is derived from the data and stored in the product metadata
(<code>model_valid_top_km</code>).</li>
<li><strong>Edge completion:</strong> where a profile's valid range does not reach the surface or the
model top, the nearest valid value is carried constant to that edge
(<code>lower_boundary_completion</code> / <code>upper_boundary_completion</code>). This is a constant
completion, not an extrapolation.</li>
<li><strong>Gaps:</strong> internal missing levels are never bridged; they remain missing in the
product and appear as white in the plots.</li>
<li><strong>WACCM-X:</strong> source data reach roughly 479 km; the published product is capped at
150 km (<code>vertical_extent_policy</code>), so its top must not be read as the native model top.</li>
</ul>
<p>Per-variable provenance and vertical-extent attributes are visible under every plot, and the full
NetCDF attributes travel with each download. Details:
<a href="https://github.com/slcs-jsc/jumacs/blob/main/docs/methodology.md">methodology</a> and
<a href="https://github.com/slcs-jsc/jumacs/blob/main/docs/products.md">product description</a>.</p>
</div></section>
<section id="about-view" hidden><div class="prose"><h2>About JuMACS</h2>
<p>JuMACS — Jülich Multi-source Atmospheric Climatology and Statistics — provides reproducible monthly
atmospheric climatologies and associated statistics for scientific applications.</p>
<p>JuMACS harmonizes published chemistry–climate model climatologies from multiple sources onto common
representations while keeping every source-model product independent. Each harmonized product preserves
the original model climatology and its provenance on 5° area-weighted latitude bands and a 1 km altitude
grid up to the model's own valid top. Products are never merged, blended or bias corrected across
models.</p>
<p>The project provides harmonized model climatology products, diagnostic visualizations, and detailed
per-variable provenance. Original source datasets remain subject to their respective licences and
citation requirements.</p>
<p class="disclaimer">JuMACS is under active development. The public site provides the current harmonized
model climatologies, diagnostic visualizations, and source-data provenance. Additional datasets,
evaluation results, and functionality may be added in future releases.</p>
<h2>Contact</h2>
<address class="contact">Dr. Lars Hoffmann<br>Jülich Supercomputing Centre (JSC)<br>Forschungszentrum Jülich<br>Germany<br>
<a href="mailto:l.hoffmann@fz-juelich.de">l.hoffmann@fz-juelich.de</a></address>
<p>Feedback and scientific collaboration are very welcome.</p>
<h2>Project links</h2>
<ul class="project-links">
<li><a href="https://datapub.fz-juelich.de/slcs/jumacs/">Published data</a></li>
<li><a href="https://github.com/slcs-jsc/jumacs">Source code on GitHub</a></li>
</ul>
</div></section>
<script id="catalog" type="application/json">__CATALOG__</script>
<script id="model-info" type="application/json">__MODELS__</script>
<script>
const catalog = JSON.parse(document.getElementById('catalog').textContent);
const modelInfo = Object.fromEntries(JSON.parse(document.getElementById('model-info').textContent)
  .map(info => [info.model, info]));
const model = document.getElementById('model');
const search = document.getElementById('search');
const list = document.getElementById('variables');
for (const info of JSON.parse(document.getElementById('model-info').textContent)) {
  const option = document.createElement('option');
  option.value = info.model; option.textContent = info.display; model.append(option);
}
let current = null;
function available() {
  const query = search.value.toLowerCase();
  return catalog.filter(row => row.model === model.value &&
    (row.display + ' ' + row.canonical + ' ' + row.variable).toLowerCase().includes(query));
}
function extent(meta, base) {
  if (meta[base + '_min'] !== undefined) return meta[base + '_min'] + '–' + meta[base + '_max'] + ' km';
  if (meta[base] !== undefined) return meta[base] + ' km';
  return null;
}
function metadataRows(row) {
  const info = modelInfo[row.model] || {};
  const rows = [['Canonical variable', row.canonical], ['Native field', row.variable],
    ['Source variable', row.source], ['Units', row.units || '—'],
    ['Source-valid bottom', extent(row.meta, 'source_valid_bottom_km')],
    ['Source-valid top', extent(row.meta, 'source_valid_top_km')],
    ['Lower edge completion', extent(row.meta, 'lower_completion_km')],
    ['Upper edge completion', extent(row.meta, 'upper_completion_km')],
    ['Product file', info.product_file], ['Period', info.period],
    ['Institution', info.institution], ['Source dataset', info.source_dataset],
    ['Model valid top', info.valid_top_km === undefined ? null : info.valid_top_km + ' km'],
    ['Edge completion policy', [info.lower_completion, info.upper_completion]
      .filter(Boolean).join('; ')],
    ['Vertical extent policy', info.extent_policy]];
  return rows.filter(([, value]) => value !== null && value !== undefined && value !== '');
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
  document.getElementById('caption').textContent =
    'Monthly climatology · four latitude–altitude cross sections';
  const dl = document.getElementById('metadata');
  dl.replaceChildren();
  for (const [term, value] of metadataRows(current)) {
    const dt = document.createElement('dt'); dt.textContent = term;
    const dd = document.createElement('dd'); dd.textContent = value;
    dl.append(dt, dd);
  }
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
const views = [['overview-nav', 'overview'], ['models-nav', 'models'],
               ['methodology-nav', 'methodology-view'], ['about-nav', 'about-view']];
function showView(active) {
  for (const [, id] of views) document.getElementById(id).hidden = id !== active;
  for (const [name, target] of views)
    document.getElementById(name).classList.toggle('active', target === active);
}
for (const [button, target] of views) {
  document.getElementById(button).onclick = () => showView(target);
}
for (const button of document.querySelectorAll('.browse-button')) {
  button.onclick = () => {
    model.value = button.dataset.model;
    current = null; search.value = '';
    model.onchange();
    showView('models');
  };
}
render();
</script></body></html>
"""


STYLE = """:root{--ink:#183349;--muted:#617686;--line:#dce7ec;--paper:#f5f9fa;--accent:#087f89}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
header{background:#12364b;color:white;padding:1.2rem max(1.5rem,calc((100vw - 1420px)/2))}
header h1{margin:0;font-size:1.6rem}header p{margin:.3rem 0 0;color:#cee2e9;font-size:.92rem}
header nav{display:flex;gap:.5rem;margin-top:.7rem}header nav button.active{background:var(--accent);color:white}
header nav button{border:0;border-radius:.45rem;background:#1d4a63;color:#d8e8ee;padding:.5rem .85rem;font:inherit;cursor:pointer}
main{max-width:1420px;min-height:calc(100vh - 98px);margin:auto;display:grid;grid-template-columns:280px minmax(0,1fr)}
main[hidden],#overview[hidden],#methodology-view[hidden],#about-view[hidden]{display:none}
#overview,#methodology-view,#about-view{max-width:1420px;margin:auto;padding:1.5rem}
.prose{max-width:90ch}
.prose p{max-width:90ch}
.prose li{max-width:90ch;margin:.25rem 0}
.prose a{color:var(--accent)}
.prose h2:not(:first-child){margin-top:1.6rem}
address.contact{font-style:normal;line-height:1.6;margin:0}
.project-links{margin:.6rem 0 0;padding-left:1.2rem;display:grid;gap:.45rem;max-width:75ch}
.disclaimer{background:#eef7f3;border-left:3px solid var(--accent);padding:.7rem 1rem;border-radius:.4rem}
.table-wrap{overflow:auto;margin-top:1rem;background:white;border:1px solid var(--line);border-radius:.8rem}
table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{padding:.6rem .8rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:#e9f4f5;position:sticky;top:0}
td.num{text-align:right;white-space:nowrap}
td.actions{white-space:nowrap}
code.file{font-size:.78rem;color:var(--muted);overflow-wrap:anywhere}
.note{color:#3f6472;font-size:.86rem}
aside{background:white;border-right:1px solid var(--line);padding:1.2rem}
label{display:block;margin:.75rem 0 .3rem;color:var(--muted);font-size:.78rem;font-weight:750;text-transform:uppercase}
select,input{width:100%;border:1px solid #b8cbd4;border-radius:.5rem;background:white;padding:.65rem .75rem;color:var(--ink);font:inherit}
#variables{display:grid;gap:.2rem;max-height:calc(100vh - 290px);overflow:auto;margin-top:.7rem}
.item{border:0;border-radius:.5rem;background:transparent;color:var(--ink);cursor:pointer;font:inherit;padding:.5rem .65rem;text-align:left}
.item:hover{background:#edf6f7}.item.active{background:#dff0f1;color:#075e65;font-weight:700}
.viewer{min-width:0;padding:1.35rem 1.6rem 2rem}.topline{display:flex;justify-content:space-between;align-items:start;gap:1rem;flex-wrap:wrap}
h2{margin:0;font-size:1.5rem}.meta{color:var(--muted);font-size:.9rem;margin:.2rem 0 0}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:.4rem}
button:not(.item):not(nav button),a.download,.browse-button{border:1px solid #bfcfd6;border-radius:.48rem;background:white;padding:.55rem .8rem;color:var(--ink);font:inherit;cursor:pointer;text-decoration:none}
a.download[hidden]{display:none}
button:hover:not(:disabled),a.download:hover{border-color:var(--accent)}
figure{background:white;border:1px solid var(--line);border-radius:.8rem;margin:1.2rem 0 .6rem;padding:.65rem;box-shadow:0 8px 28px #12364b0a}
figure img{display:block;width:100%;max-height:calc(100vh - 220px);object-fit:contain}
figcaption{display:flex;justify-content:space-between;align-items:center;gap:1rem;color:var(--muted);font-size:.88rem;margin-top:.7rem}
figcaption span:last-child{display:flex;gap:.4rem;flex-wrap:wrap}
.plot-note{margin:.15rem 0 0;color:#3f6472;font-size:.86rem;max-width:90ch}
#metadata-view{background:white;border:1px solid var(--line);border-radius:.8rem;padding:.7rem 1.1rem;margin:1rem 0 0}
#metadata-view summary{cursor:pointer;font-weight:700;color:var(--ink)}
.dataset-meta{display:grid;grid-template-columns:auto minmax(0,1fr);gap:.3rem 1rem;margin:.7rem 0 0}
.dataset-meta dt{color:var(--muted);font-size:.78rem;font-weight:750;text-transform:uppercase;padding-top:.1rem}
.dataset-meta dd{margin:0;font-size:.9rem;overflow-wrap:anywhere}
@media(max-width:760px){main{display:block}aside{border-right:0;border-bottom:1px solid var(--line)}
#variables{display:flex;max-height:none;overflow:auto}.item{white-space:nowrap}.viewer{padding:1rem}figure img{max-height:none}}
"""


def _render_model(model_name, product, root, staging):
    """Render one model in its own staging folder; safe to run in a worker process."""
    fields = []
    skipped = []
    model_config = config.load_config(model_name)
    inverse = {native: canonical for canonical, native in model_config["variables"].items()}
    with xr.open_dataset(product) as dataset:
        if ("month" not in dataset.coords or "altitude" not in dataset.coords
                or "lat" not in dataset.coords or dataset.coords["month"].size != 12):
            skipped.append((model_name, product.name, "not a 12-month month×altitude×lat climatology"))
            return (model_name, product, fields), {}, skipped
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
            units = str(field.attrs.get("units", ""))
            scale, plot_units = _display_scale(values, units)
            meta = {}
            for key in EXTENT_KEYS:
                for suffix in ("", "_min", "_max"):
                    attr = f"{key}{suffix}"
                    if attr in field.attrs:
                        meta[attr] = float(field.attrs[attr])
            path = _section_path(model_name, stem)
            (staging / path).parent.mkdir(parents=True, exist_ok=True)
            _plot_section(values * scale, altitude, latitudes, model_name,
                          _display_name(canonical), plot_units, staging / path)
            fields.append({"native": stem, "canonical": canonical,
                           "display": _display_name(canonical), "units": units,
                           "source": str(field.attrs.get("source_variable") or stem),
                           "meta": meta, "views": {"section": path}})
    info = _model_info(model_name, product, len(fields))
    copy = staging / "products" / "harmonized" / model_name / product.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(product, copy)
    return (model_name, product, fields), info, skipped


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
    site = root / "site"
    site.mkdir(parents=True, exist_ok=True)
    staging = site / ".browse-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    records = []
    infos = []
    skipped = []
    try:
        tasks = [(name, product, root, staging) for name, product in products]
        if workers == 1 or len(tasks) == 1:
            results = [_render_model(*task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                     mp_context=get_context("spawn")) as pool:
                results = list(pool.map(_render_model, *zip(*tasks)))
        for record, info, model_skipped in results:
            records.append(record)
            if info:
                infos.append(info)
            skipped.extend(model_skipped)
        infos.sort(key=lambda info: info["display"])
        assets = staging / "assets"
        assets.mkdir()
        (assets / "style.css").write_text(STYLE, encoding="utf-8")
        (staging / "index.html").write_text(_html(infos, records, start_year, end_year),
                                            encoding="utf-8")
        for name in ("plots", "assets", "index.html", "products/harmonized"):
            _publish(staging, site, name)
        for stale in STALE_PATHS:
            old = site / stale
            if old.is_dir() and not old.is_symlink():
                shutil.rmtree(old)
            elif old.exists() or old.is_symlink():
                old.unlink()
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"site": site, "models": [name for name, _, _ in records],
            "variables": {name: len(fields) for name, _, fields in records},
            "png_files": sum(len(fields) for _, _, fields in records),
            "netcdf_files": len(records),
            "product_files": [path.name for _, path in products], "skipped": skipped}
