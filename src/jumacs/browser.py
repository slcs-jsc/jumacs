"""Generate a self-contained static browser for application climatologies."""

import json
import re
import shutil
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import config, remap, vertical

ZONAL_MONTHS = (1, 4, 7, 10)
MONTH_NAMES = {1: "January", 4: "April", 7: "July", 10: "October"}
PLOT_PRESSURES_HPA = (1000.0, 100.0, 10.0, 1.0)
PLOT_LATITUDE_EDGES = (-90.0, -65.0, -20.0, 20.0, 65.0, 90.0)
PLOT_LATITUDE_NAMES = ("Polar South (90–65°S)", "Midlatitudes South (65–20°S)",
                       "Tropics (20°S–20°N)", "Midlatitudes North (20–65°N)",
                       "Polar North (65–90°N)")
PLOT_LATITUDE_CENTERS = tuple((south + north) / 2 for south, north in pairwise(PLOT_LATITUDE_EDGES))


def discover_application_products(root, start_year, end_year, model=None):
    """Return existing application products for the requested period."""
    base = Path(root) / "products" / "application"
    folders = [base / model] if model else sorted(base.iterdir()) if base.exists() else []
    products = []
    for folder in folders:
        if not folder.is_dir():
            continue
        name = folder.name
        filename = f"jumacs_{config.model_slug(name)}_application_climatology_{start_year}-{end_year}.nc"
        path = folder / filename
        if path.is_file():
            products.append((name, path))
    if not products:
        raise FileNotFoundError(f"No application products for {start_year}-{end_year} in {base}")
    return products


def _plot_paths(model, native_name):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", native_name)
    return (Path("plots") / model / f"{safe_name}_zonal.png",
            Path("plots") / model / f"{safe_name}_annual.png")


def _timeline_path(model, native_name):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", native_name)
    return Path("plots") / model / f"{safe_name}_timeline.png"


def _band_samples(field, pressure, latitudes):
    """Sample exact pressures per column, then average over five spherical-area bands."""
    sampled = vertical.interpolate_log_pressure(
        field, pressure, np.asarray(PLOT_PRESSURES_HPA) * 100.0, bridge_gaps=False)
    weights = remap.latitude_band_weights(latitudes, PLOT_LATITUDE_EDGES)
    bands = remap.area_weighted_bands(sampled, weights, PLOT_LATITUDE_CENTERS)
    return bands.transpose("lat", "pressure", "time").values


def _pressure_to_height(pressure_hpa):
    """Indicative height for a fixed 7 km scale height and 1000 hPa reference."""
    return -7.0 * np.log(np.asarray(pressure_hpa) / 1000.0)


def _height_to_pressure(height_km):
    return 1000.0 * np.exp(-np.asarray(height_km) / 7.0)


def _plot_band_series(samples, x, model, label, units, title, xlabel, destination):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    colors = ("#2b8cbe", "#f28e2b", "#4a9d65", "#9467bd", "#c64b5d")
    for level, (pressure, ax) in enumerate(zip(PLOT_PRESSURES_HPA, axes.flat)):
        for band, (name, color) in enumerate(zip(PLOT_LATITUDE_NAMES, colors)):
            ax.plot(x, samples[band, level], color=color, lw=1.3, label=name)
        ax.set(title=f"{pressure:g} hPa", ylabel=units or "Value")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    for ax in axes[1, :]:
        ax.set_xlabel(xlabel)
    if xlabel == "Month":
        for ax in axes.flat:
            ax.set_xticks(np.arange(1, 13))
            ax.set_xlim(1, 12)
    fig.suptitle(f"{model} · {label} · {title}")
    fig.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_timeline(source, native, model, label, destination):
    """Plot the actual monthly zonal series, using its published native pressure."""
    with xr.open_dataset(source) as ds:
        field = ds[native]
        pressure = vertical.native_pressure(ds, field)
        if pressure is None or "time" not in field.dims or "lat" not in field.dims:
            return False
        samples = _band_samples(field, pressure, ds.lat.values)
        years = ds.time.dt.year.values
        months = ds.time.dt.month.values
        units = field.attrs.get("units", "")
    if units in ("mol mol-1", "mol/mol"):
        finite = np.abs(samples[np.isfinite(samples)])
        peak = float(np.percentile(finite, 98)) if finite.size else 0.0
        power, units = ((6, "ppmv") if peak >= 1e-6 else (9, "ppbv") if peak >= 1e-9
                        else (12, "pptv") if peak >= 1e-12 else (15, "ppqv"))
        samples = samples * 10 ** power
    elif units == "Pa":
        samples, units = samples / 100.0, "hPa"
    elif units == "m":
        samples, units = samples / 1000.0, "km"
    x = years + (months - 0.5) / 12
    _plot_band_series(samples, x, model, label, units,
                      f"monthly zonal means {years.min()}–{years.max()}", "Year", destination)
    return True


def _plot_zonal(values, pressure_hpa, latitudes, model, label, units, path):
    displayed = values[np.array(ZONAL_MONTHS) - 1]
    finite = displayed[np.isfinite(displayed)]
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        width = max(abs(low) * 0.01, 1e-12)
        low, high = low - width, high + width
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, month, panel in zip(axes.flat, ZONAL_MONTHS, displayed):
        image = ax.pcolormesh(latitudes, pressure_hpa, np.ma.masked_invalid(panel),
                              shading="auto", cmap="viridis", vmin=low, vmax=high,
                              rasterized=True)
        ax.set(title=MONTH_NAMES[month], yscale="log", ylim=(pressure_hpa.max(), pressure_hpa.min()))
    for ax in axes[-1]:
        ax.set_xlabel("Latitude (°N)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pressure (hPa)")
    for ax in axes[:, 1]:
        height = ax.twinx()
        height.set_yscale("log")
        height.set_ylim(ax.get_ylim())
        ticks_km = np.arange(0, _pressure_to_height(pressure_hpa.min()), 20)
        height.set_yticks(_height_to_pressure(ticks_km))
        height.set_yticklabels([f"{value:g}" for value in ticks_km])
        height.minorticks_off()
        height.set_ylabel("Approx. height (km)")
    fig.suptitle(f"{model} · {label} · zonal climatology")
    fig.colorbar(image, ax=axes, label=units or "Value", shrink=0.85)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_annual(field, model, label, units, path):
    samples = _band_samples(field, field.pressure, field.lat.values)
    _plot_band_series(samples, np.arange(1, 13), model, label, units,
                      "Annual cycle", "Month", path)


def _html(records, start_year, end_year, inventory_rows):
    catalog = []
    for model, product, fields in records:
        for field in fields:
            catalog.append({"model": model, "variable": field["native"],
                            "label": field["label"], "units": field["units"],
                            "product": f"products/application/{model}/{product.name}",
                            "views": {name: path.as_posix() for name, path in field["views"].items()}})
    payload = json.dumps(catalog, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    inventory_payload = json.dumps(inventory_rows, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    return (HTML_TEMPLATE.replace("__PERIOD__", f"{start_year}–{end_year}")
            .replace("__CATALOG__", payload).replace("__INVENTORY__", inventory_payload))


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>JuMACS · Application climatology atlas</title><link rel="stylesheet" href="assets/style.css"></head>
<body><header><h1>JuMACS application climatology atlas</h1>
<p>Model climatologies on the shared pressure grid · __PERIOD__</p>
<nav><button id="atlas-nav" class="active">Plots</button><button id="inventory-nav">Data availability</button></nav></header>
<main id="atlas"><aside><label for="model">Model / product</label><select id="model"></select>
<label for="search">Variable</label><input id="search" type="search" placeholder="Search variable name">
<div id="variables" aria-label="Available variables"></div></aside>
<section class="viewer"><div class="topline"><div><h2 id="title"></h2><p class="meta" id="meta"></p></div>
<div class="controls"><button id="prev" title="Previous variable">← Previous</button>
<button id="next" title="Next variable">Next →</button></div></div>
<div class="controls tabs"><button class="tab active" data-view="zonal">Four cross sections</button>
<button class="tab" data-view="timeline">Monthly time series</button>
<button class="tab" data-view="annual">Annual cycle</button></div>
<figure><img id="plot" alt=""><figcaption><span id="caption"></span>
<span><a class="download" id="png" download>Download PNG</a>
<a class="download" id="product" download>Download NetCDF</a></span></figcaption></figure>
<p class="note">Cross sections: January, April, July and October on the application pressure grid.
The right axes on April and October show an approximate height using a fixed 7 km scale height.
Annual cycle: twelve climatological months at exactly 1000, 100, 10 and 1 hPa.
Monthly time series: original monthly zonal data at the same four pressures;
both line plots compare tropical, northern/southern midlatitude and northern/southern polar area means.
The monthly series may span a longer source period than the application climatology and appear only where the source series exists.
Some 1000 hPa curves are absent where source data do not reach that pressure. No missing values are filled.</p>
</section></main><section id="inventory-view" hidden><h2>Data availability</h2>
<p>Existing native and application climatologies. Pressure limits use any finite mean value.</p>
<label for="inventory-search">Filter model or variable</label><input id="inventory-search" type="search" placeholder="Search inventory">
<div class="table-wrap"><table><thead><tr>
<th><button class="sort-button" data-sort="model">Model</button></th>
<th><button class="sort-button" data-sort="canonical_variable">Variable</button></th>
<th><button class="sort-button" data-sort="native_variable">Native name</button></th>
<th><button class="sort-button" data-sort="dimensionality">2D/3D</button></th>
<th><button class="sort-button" data-sort="units">Units</button></th>
<th><button class="sort-button" data-sort="application_available">Application product</button></th>
<th><button class="sort-button" data-sort="bottom_pressure_hpa">Vertical range (hPa)</button></th>
<th><button class="sort-button" data-sort="waccmx_extended">WACCM-X extended</button></th>
</tr></thead><tbody id="inventory-body"></tbody></table></div></section>
<script id="catalog" type="application/json">__CATALOG__</script>
<script id="inventory-data" type="application/json">__INVENTORY__</script>
<script>
const catalog = JSON.parse(document.getElementById('catalog').textContent);
const inventory = JSON.parse(document.getElementById('inventory-data').textContent);
const model = document.getElementById('model');
const search = document.getElementById('search');
const list = document.getElementById('variables');
for (const name of [...new Set(catalog.map(row => row.model))]) {
  const option = document.createElement('option'); option.value = name; option.textContent = name; model.append(option);
}
let current = null, view = 'zonal';
function available() {
  const query = search.value.toLowerCase();
  return catalog.filter(row => row.model === model.value &&
    (row.label + ' ' + row.variable).toLowerCase().includes(query));
}
function render() {
  const rows = available();
  if (!rows.includes(current)) current = rows[0] || null;
  list.replaceChildren();
  for (const row of rows) {
    const button = document.createElement('button');
    button.className = 'item' + (row === current ? ' active' : '');
    button.textContent = row.label.toLowerCase() === row.variable.toLowerCase()
      ? row.label : row.label + ' · ' + row.variable;
    button.onclick = () => { current = row; render(); };
    list.append(button);
  }
  if (!current) {
    document.getElementById('title').textContent = 'No matching variable';
    document.getElementById('plot').removeAttribute('src');
    return;
  }
  if (!current.views[view]) view = 'zonal';
  document.getElementById('title').textContent = current.model + ' · ' + current.label;
  document.getElementById('meta').textContent = current.variable + ' · ' + current.units + ' · __PERIOD__';
  const image = document.getElementById('plot');
  image.src = current.views[view];
  image.alt = current.model + ' ' + current.label + ' ' + view;
  document.getElementById('png').href = current.views[view];
  document.getElementById('product').href = current.product;
  document.getElementById('caption').textContent = {
    zonal: 'Monthly climatology · four latitude–pressure sections',
    timeline: 'Original monthly zonal time series', annual: 'Twelve-month climatological annual cycle'
  }[view];
  for (const tab of document.querySelectorAll('.tab')) {
    tab.disabled = !current.views[tab.dataset.view];
    tab.classList.toggle('active', tab.dataset.view === view);
  }
}
model.onchange = () => { current = null; render(); };
search.oninput = () => { current = null; render(); };
for (const tab of document.querySelectorAll('.tab')) tab.onclick = () => { view = tab.dataset.view; render(); };
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
    const pressure = row.bottom_pressure_hpa == null ? '' :
      `${row.bottom_pressure_hpa.toPrecision(3)}–${row.top_pressure_hpa.toPrecision(3)}`;
    const values = [row.model, row.canonical_variable || '—', row.native_variable,
      row.dimensionality || '—', row.units, null, pressure,
      row.waccmx_extended == null ? 'unknown' : row.waccmx_extended ? 'yes' : 'no'];
    for (const value of values) {
      const cell = document.createElement('td');
      if (value === null && row.application_available && row.application_product) {
        const link = document.createElement('a');
        link.href = row.application_product;
        link.textContent = 'NetCDF';
        cell.append(link);
      } else cell.textContent = value === null ? '—' : value;
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
      heading.textContent = heading.dataset.label + (active ? inventoryDescending ? ' ↓' : ' ↑' : '');
    }
    renderInventory();
  };
  button.dataset.label = label;
}
document.querySelector('.sort-button[data-sort="model"]').click();
for (const [button, show] of [['atlas-nav', false], ['inventory-nav', true]]) {
  document.getElementById(button).onclick = () => {
    document.getElementById('atlas').hidden = show;
    document.getElementById('inventory-view').hidden = !show;
    document.getElementById('atlas-nav').classList.toggle('active', !show);
    document.getElementById('inventory-nav').classList.toggle('active', show);
  };
}
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
main[hidden],#inventory-view[hidden]{display:none}
#inventory-view{max-width:1420px;margin:auto;padding:1.5rem}
#inventory-search{max-width:28rem}.table-wrap{overflow:auto;margin-top:1rem}
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
.tab.active{background:var(--accent);border-color:var(--accent);color:white}.tab:disabled{opacity:.4;cursor:not-allowed}
button:hover:not(:disabled),a.download:hover{border-color:var(--accent)}.tabs{margin-top:1.2rem}
figure{background:white;border:1px solid var(--line);border-radius:.8rem;margin:1rem 0 .6rem;padding:.65rem;box-shadow:0 8px 28px #12364b0a}
figure img{display:block;width:100%;max-height:calc(100vh - 220px);object-fit:contain}
figcaption{display:flex;justify-content:space-between;align-items:center;gap:1rem;color:var(--muted);font-size:.88rem;margin-top:.7rem}
figcaption span:last-child{display:flex;gap:.4rem;flex-wrap:wrap}
.note{background:#e9f4f5;border-left:3px solid var(--accent);padding:.65rem .9rem;margin-top:1.3rem;color:#335c69;font-size:.88rem}
@media(max-width:760px){main{display:block}aside{border-right:0;border-bottom:1px solid var(--line)}
#variables{display:flex;max-height:none;overflow:auto}.item{white-space:nowrap}.viewer{padding:1rem}figure img{max-height:none}}
"""


def build_site(start_year=None, end_year=None, *, model=None, root=None):
    """Plot existing products and publish a deployable tree below ``root/site``."""
    root = config.ROOT if root is None else Path(root)
    period = config.reference_period()["reference_period"]
    start_year = period["start_year"] if start_year is None else start_year
    end_year = period["end_year"] if end_year is None else end_year
    products = discover_application_products(root, start_year, end_year, model)
    from .inventory import build_inventory
    inventory = build_inventory(start_year, end_year, model=model, root=root)
    site = root / "site"
    site.mkdir(parents=True, exist_ok=True)
    staging = site / ".browse-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    records = []
    skipped = []
    timeline_skipped = []
    try:
        for model_name, product in products:
            fields = []
            model_config = config.load_config(model_name)
            mapping = {native: canonical for canonical, native in model_config["variables"].items()}
            with xr.open_dataset(product) as dataset:
                pressure_hpa = np.asarray(dataset.pressure.values, dtype=float) / 100.0
                latitudes = np.asarray(dataset.lat.values, dtype=float)
                for name in sorted(dataset.data_vars):
                    if not name.endswith("_mean"):
                        continue
                    field = dataset[name]
                    if field.dims != ("time", "pressure", "lat") or field.sizes["time"] != 12:
                        skipped.append((model_name, name, "not a 12-month time×pressure×lat field"))
                        continue
                    native = name[:-5]
                    values = np.asarray(field.values)
                    if not np.isfinite(values[np.array(ZONAL_MONTHS) - 1]).any():
                        skipped.append((model_name, name, "no finite values in displayed months"))
                        continue
                    paths = _plot_paths(model_name, native)
                    for path in paths:
                        (staging / path).parent.mkdir(parents=True, exist_ok=True)
                    label = mapping.get(native, native).replace("_", " ")
                    units = str(field.attrs.get("units", ""))
                    _plot_zonal(values, pressure_hpa, latitudes, model_name, label, units, staging / paths[0])
                    _plot_annual(field, model_name, label, units, staging / paths[1])
                    views = {"zonal": paths[0], "annual": paths[1]}
                    source = root / model_config["paths"]["zonal"] / f"{native}_monthly_zonal.nc"
                    if source.is_file():
                        timeline = _timeline_path(model_name, native)
                        if _plot_timeline(source, native, model_name, label, staging / timeline):
                            views["timeline"] = timeline
                        else:
                            timeline_skipped.append((model_name, native, "no usable native pressure profile"))
                    else:
                        timeline_skipped.append((model_name, native, "monthly zonal source missing"))
                    fields.append({"native": native, "label": label, "units": units, "views": views})
            copy = staging / "products" / "application" / model_name / product.name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(product, copy)
            records.append((model_name, product, fields))
        assets = staging / "assets"
        assets.mkdir()
        (assets / "style.css").write_text(STYLE, encoding="utf-8")
        site_inventory = []
        copied = {name for name, _, _ in records}
        for row in inventory["rows"]:
            entry = dict(row)
            if row["model"] in copied and row["application_product"]:
                entry["application_product"] = row["application_product"]
            else:
                entry["application_product"] = None
            site_inventory.append(entry)
        (staging / "index.html").write_text(_html(records, start_year, end_year, site_inventory), encoding="utf-8")
        for name in ("plots", "assets", "index.html"):
            old = site / name
            if old.is_dir() and not old.is_symlink():
                shutil.rmtree(old)
            elif old.exists() or old.is_symlink():
                old.unlink()
            new = staging / name
            if new.exists():
                new.replace(old)
        application = site / "products" / "application"
        application.parent.mkdir(exist_ok=True)
        if application.is_dir() and not application.is_symlink():
            shutil.rmtree(application)
        elif application.exists() or application.is_symlink():
            application.unlink()
        (staging / "products" / "application").replace(application)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"site": site, "models": [name for name, _, _ in records],
            "variables": {name: len(fields) for name, _, fields in records},
            "png_files": sum(len(field["views"]) for _, _, fields in records for field in fields),
            "netcdf_files": len(records), "skipped": skipped,
            "timeline_skipped": timeline_skipped}
