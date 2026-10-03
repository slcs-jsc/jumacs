"""Generate a self-contained static browser for application climatologies."""

import json
import re
import shutil
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import config

ZONAL_MONTHS = (1, 4, 7, 10)
MONTH_NAMES = {1: "January", 4: "April", 7: "July", 10: "October"}
REFERENCE_LATITUDES = (-60.0, 0.0, 60.0)
REFERENCE_PRESSURES_HPA = (100.0, 10.0, 1.0)


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


def _plot_timeline(source, native, model, label, destination):
    """Plot the actual monthly zonal series, using its published native pressure."""
    with xr.open_dataset(source) as ds:
        field = ds[native]
        level = next((dim for dim in ("lev", "plev") if dim in field.dims), None)
        if level is None or "time" not in field.dims or "lat" not in field.dims:
            return False
        order = ("time", level, "lat")
        values = field.transpose(*order).values
        if "air_pressure" in ds:
            pressure = ds.air_pressure.transpose(*order).values / 100.0
        else:
            coordinate = ds[level]
            factor = {"Pa": 0.01, "hPa": 1.0, "mbar": 1.0}.get(coordinate.attrs.get("units"))
            if factor is None:
                return False
            pressure = np.broadcast_to(coordinate.values[None, :, None] * factor, values.shape)
        latitudes = ds.lat.values
        years = ds.time.dt.year.values
        months = ds.time.dt.month.values
        units = field.attrs.get("units", "")
    if units in ("mol mol-1", "mol/mol"):
        finite = np.abs(values[np.isfinite(values)])
        peak = float(np.percentile(finite, 98)) if finite.size else 0.0
        power, units = ((6, "ppmv") if peak >= 1e-6 else (9, "ppbv") if peak >= 1e-9
                        else (12, "pptv") if peak >= 1e-12 else (15, "ppqv"))
        values = values * 10 ** power
    elif units == "Pa":
        values, units = values / 100.0, "hPa"
    elif units == "m":
        values, units = values / 1000.0, "km"
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = years + (months - 0.5) / 12
    with np.errstate(divide="ignore", invalid="ignore"):
        for target, color in ((300.0, "#078c9d"), (30.0, "#df7344")):
            distance = np.where((pressure > 0) & np.isfinite(pressure),
                                np.abs(np.log(pressure / target)), np.inf)
            index = np.argmin(distance, axis=1)
            layer = np.take_along_axis(values, index[:, None, :], axis=1)[:, 0, :]
            for south, north, style, band in ((-20, 20, "-", "tropics"),
                                               (45, 75, "--", "45–75°N")):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    series = np.nanmean(layer[:, (latitudes >= south) & (latitudes <= north)], axis=1)
                ax.plot(x, series, color=color, ls=style, lw=1.3,
                        label=f"{target:g} hPa · {band}")
    ax.set(xlim=(years.min(), years.max() + 1), xlabel="Year", ylabel=units or "Value",
           title=f"{model} · {label} · monthly zonal means {years.min()}–{years.max()}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=125)
    plt.close(fig)
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
    fig.suptitle(f"{model} · {label} · zonal climatology")
    fig.colorbar(image, ax=axes, label=units or "Value", shrink=0.85)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_annual(values, pressure_hpa, latitudes, model, label, units, path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    months = np.arange(1, 13)
    for ax, latitude in zip(axes, REFERENCE_LATITUDES):
        lat_index = int(np.abs(latitudes - latitude).argmin())
        for pressure in REFERENCE_PRESSURES_HPA:
            pressure_index = int(np.abs(np.log(pressure_hpa / pressure)).argmin())
            ax.plot(months, values[:, pressure_index, lat_index], marker="o", markersize=3,
                    label=f"{pressure_hpa[pressure_index]:g} hPa")
        ax.set_ylabel(f"{latitudes[lat_index]:g}°N\n{units or 'Value'}")
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize="small")
    axes[-1].set(xlabel="Month", xticks=months, xlim=(1, 12))
    fig.suptitle(f"{model} · {label} · Annual cycle")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _html(records, start_year, end_year):
    catalog = []
    for model, product, fields in records:
        for field in fields:
            catalog.append({"model": model, "variable": field["native"],
                            "label": field["label"], "units": field["units"],
                            "product": f"products/application/{model}/{product.name}",
                            "views": {name: path.as_posix() for name, path in field["views"].items()}})
    payload = json.dumps(catalog, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    return (HTML_TEMPLATE.replace("__PERIOD__", f"{start_year}–{end_year}")
            .replace("__CATALOG__", payload))


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>JuMACS · Application climatology atlas</title><link rel="stylesheet" href="assets/style.css"></head>
<body><header><h1>JuMACS application climatology atlas</h1>
<p>Model climatologies on the shared pressure grid · __PERIOD__</p></header>
<main><aside><label for="model">Model / product</label><select id="model"></select>
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
Annual cycle: twelve climatological months near 60°S, the equator and 60°N at 100, 10 and 1 hPa.
Monthly time series: original monthly zonal data at 300 and 30 hPa for 20°S–20°N and 45–75°N;
these may span a longer source period than the application climatology and appear only where the source series exists.
No missing values are filled.</p>
</section></main><script id="catalog" type="application/json">__CATALOG__</script>
<script>
const catalog = JSON.parse(document.getElementById('catalog').textContent);
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
render();
</script></body></html>
"""


STYLE = """:root{--ink:#183349;--muted:#617686;--line:#dce7ec;--paper:#f5f9fa;--accent:#087f89}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
header{background:#12364b;color:white;padding:1.2rem max(1.5rem,calc((100vw - 1420px)/2))}
header h1{margin:0;font-size:1.6rem}header p{margin:.3rem 0 0;color:#cee2e9;font-size:.92rem}
main{max-width:1420px;min-height:calc(100vh - 98px);margin:auto;display:grid;grid-template-columns:280px minmax(0,1fr)}
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
                    _plot_annual(values, pressure_hpa, latitudes, model_name, label, units, staging / paths[1])
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
        (staging / "index.html").write_text(_html(records, start_year, end_year), encoding="utf-8")
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
