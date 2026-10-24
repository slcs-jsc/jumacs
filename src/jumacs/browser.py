"""Generate a self-contained static browser for application climatologies."""

import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from html import escape
from itertools import pairwise
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import config, coordinates, ig2, remap, vertical, zonal
from .reader import open_source

ZONAL_MONTHS = (1, 4, 7, 10)
MONTH_NAMES = {1: "January", 4: "April", 7: "July", 10: "October"}
PLOT_PRESSURES_HPA = (1000.0, 100.0, 10.0, 1.0)
PLOT_LATITUDE_EDGES = (-90.0, -65.0, -20.0, 20.0, 65.0, 90.0)
PLOT_LATITUDE_NAMES = ("Polar South (90–65°S)", "Midlatitudes South (65–20°S)",
                       "Tropics (20°S–20°N)", "Midlatitudes North (20–65°N)",
                       "Polar North (65–90°N)")
PLOT_LATITUDE_CENTERS = tuple((south + north) / 2 for south, north in pairwise(PLOT_LATITUDE_EDGES))

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


READY_NAME = re.compile(
    r"jumacs_(?P<model>[a-z0-9-]+?)_waccmx_ig2_(?P<start>\d{4})-(?P<end>\d{4})"
    r"_(?P<latitude>\d+)deg_(?P<altitude>\d+)km\.nc")


def discover_application_ready_products(root, start_year, end_year, model=None):
    """Return existing application-ready (IG2-completed) products for the period."""
    base = Path(root) / "products" / "application-ready"
    if not base.is_dir():
        return []
    products = []
    for path in sorted(base.glob("*.nc")):
        match = READY_NAME.fullmatch(path.name)
        if not match or int(match["start"]) != start_year or int(match["end"]) != end_year:
            continue
        if model is not None and model.lower() != match["model"]:
            continue
        products.append((match["model"], path))
    return products


def _plot_paths(model, native_name):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", native_name)
    return (Path("plots") / model / f"{safe_name}_zonal.png",
            Path("plots") / model / f"{safe_name}_annual.png")


def _timeline_path(model, native_name):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", native_name)
    return Path("plots") / model / f"{safe_name}_timeline.png"


def _map_path(model, native_name):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", native_name)
    return Path("plots") / model / f"{safe_name}_map.png"


def _reference_year_source(root, model_config, native, year):
    """Find an existing geographical Amon or AmonZ file covering the nominal year."""
    for table in ("Amon", "AmonZ"):
        folder = root / model_config["paths"]["raw"] / table
        found = _year_file(folder / native, native, year)
        if found is not None:
            return found
    return None


def _year_file(folder, native, year):
    for path in sorted(folder.glob(f"{native}_*.nc")):
        match = re.search(r"_(\d{6,8})-(\d{6,8})\.nc$", path.name)
        if match and match[1][:6] <= f"{year}01" and match[2][:6] >= f"{year}10":
            return path
    return None


def _published_pressure(source, model, year, field):
    """Read a separate geographical air-pressure field when hybrid terms are absent."""
    path = _year_file(source.parent.parent / "pa", "pa", year)
    if path is None:
        raise ValueError("No reconstructable or published geographical pressure")
    with open_source(path, model, "pa") as ds:
        months = ds.time.dt.month.values
        years = ds.time.dt.year.values
        indices = [np.flatnonzero((years == year) & (months == month)) for month in ZONAL_MONTHS]
        if any(len(found) != 1 for found in indices):
            raise ValueError("Published pressure lacks a required example month")
        pressure = ds["pa"].isel(time=[int(found[0]) for found in indices])
        if (pressure.sizes.get(vertical.vertical_dimension(field)) != field.sizes[vertical.vertical_dimension(field)]
                or not np.array_equal(pressure.time.values, field.time.values)):
            raise ValueError("Published pressure grid or time differs from field")
        for name in ("lat", "lon"):
            if name in field.coords and name in pressure.coords and not np.array_equal(field[name], pressure[name]):
                raise ValueError(f"Published pressure {name} differs from field")
        return pressure.load()


def _map_values(source, model, native, year):
    """Read four geographical monthly fields; interpolate 3D fields to four pressures."""
    model_config = config.load_config(model)
    with open_source(source, model, native) as ds:
        months = ds.time.dt.month.values
        years = ds.time.dt.year.values
        indices = [np.flatnonzero((years == year) & (months == month)) for month in ZONAL_MONTHS]
        if any(len(found) != 1 for found in indices):
            raise ValueError(f"Expected January/April/July/October {year} in {source.name}")
        sample = ds.isel(time=[int(found[0]) for found in indices])
        field = zonal.mask_near_fill(sample[native], model_config.get("near_fill_relative_tolerance"))
        latitude = model_config["coordinates"]["latitude"]
        longitude = model_config["coordinates"]["longitude"]
        if (latitude not in field.dims or longitude not in field.dims or
                field.sizes[longitude] < 2 or sample[latitude].ndim != 1 or sample[longitude].ndim != 1):
            raise ValueError("No geographical latitude–longitude field")
        extra = [dim for dim in field.dims if dim not in ("time", latitude, longitude)]
        if len(extra) == 1 and field.sizes[extra[0]] == 1:
            field = field.squeeze(extra[0], drop=True)
            extra = []
        if not extra:
            values = field.transpose("time", latitude, longitude).values[:, None]
            levels = None
        elif len(extra) == 1:
            pressure = vertical.native_pressure(sample, field)
            if pressure is None:
                try:
                    pressure = coordinates.hybrid_pressure(sample, model_config)
                except (KeyError, ValueError):
                    pressure = _published_pressure(source, model, year, field)
            values = vertical.interpolate_log_pressure(
                field, pressure, np.asarray(PLOT_PRESSURES_HPA) * 100.0,
                bridge_gaps=False).transpose("time", "pressure", latitude, longitude).values
            levels = PLOT_PRESSURES_HPA
        else:
            raise ValueError("Field has more than one native vertical dimension")
        latitudes = np.asarray(sample[latitude].values, dtype=float)
        longitudes = np.asarray(sample[longitude].values, dtype=float)
        units = str(field.attrs.get("units", ""))
    lat_order = np.argsort(latitudes)
    longitudes = (longitudes + 180.0) % 360.0 - 180.0
    lon_order = np.argsort(longitudes)
    values = values[:, :, lat_order, :][:, :, :, lon_order]
    return values, latitudes[lat_order], longitudes[lon_order], levels, units


def _plot_map(values, latitudes, longitudes, levels, model, label, units, year, path):
    """Plot four source months on a latitude–longitude grid, one row per pressure."""
    scale, units = _display_scale(values, units)
    values = values * scale
    rows = values.shape[1]
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3 * rows + 1.5),
                             constrained_layout=True, squeeze=False)
    has_data = False
    for row, panels in enumerate(axes):
        displayed = values[:, row]
        finite = displayed[np.isfinite(displayed)]
        if finite.size:
            has_data = True
            low, high = float(finite.min()), float(finite.max())
            if low == high:
                width = max(abs(low) * 0.01, 1e-12)
                low, high = low - width, high + width
        for ax, month, panel in zip(panels, ZONAL_MONTHS, displayed):
            ax.set(title=MONTH_NAMES[month], xlim=(-180, 180), ylim=(-90, 90))
            if finite.size:
                image = ax.pcolormesh(longitudes, latitudes, np.ma.masked_invalid(panel),
                                      shading="auto", cmap="viridis", vmin=low, vmax=high,
                                      rasterized=True)
            else:
                ax.text(0.5, 0.5, "No native data", ha="center", va="center", transform=ax.transAxes)
            if row == rows - 1:
                ax.set_xlabel("Longitude (°E)")
        panels[0].set_ylabel("Latitude (°N)" + (f"\n{levels[row]:g} hPa" if levels else ""))
        if finite.size:
            fig.colorbar(image, ax=panels, label=units or "Value", shrink=0.85)
    fig.suptitle(f"{model} · {label} · geographical monthly fields ({year}; single year)")
    if has_data:
        fig.savefig(path, dpi=120)
    plt.close(fig)
    return has_data


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


def _plot_band_series(samples, x, model, label, units, title, xlabel, destination, *, timeline=False):
    rows, columns = (4, 1) if timeline else (2, 2)
    figsize = (13, 10) if timeline else (12, 8)
    fig, axes = plt.subplots(rows, columns, figsize=figsize, sharex=True,
                             constrained_layout=True, squeeze=False)
    colors = ("#2b8cbe", "#f28e2b", "#4a9d65", "#9467bd", "#c64b5d")
    for level, (pressure, ax) in enumerate(zip(PLOT_PRESSURES_HPA, axes.flat)):
        for band, (name, color) in enumerate(zip(PLOT_LATITUDE_NAMES, colors)):
            ax.plot(x, samples[band, level], color=color, lw=1.3, label=name)
        ax.set(title=f"{pressure:g} hPa", ylabel=units or "Value")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    if xlabel == "Month":
        for ax in axes.flat:
            ax.set_xticks(np.arange(1, 13))
            ax.set_xlim(1, 12)
    fig.suptitle(f"{model} · {label} · {title}")
    fig.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
    scale, units = _display_scale(samples, units)
    samples = samples * scale
    x = years + (months - 0.5) / 12
    _plot_band_series(samples, x, model, label, units,
                      f"monthly zonal means {years.min()}–{years.max()}", "Year", destination,
                      timeline=True)
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


def _plot_annual(field, model, label, units, path, *, scale=1.0):
    samples = _band_samples(field, field.pressure, field.lat.values) * scale
    _plot_band_series(samples, np.arange(1, 13), model, label, units,
                      "Annual cycle", "Month", path)


def _ready_plot_path(model_dir, variable):
    safe_model = re.sub(r"[^A-Za-z0-9_-]+", "_", model_dir)
    safe_var = re.sub(r"[^A-Za-z0-9_-]+", "_", variable)
    return Path("plots-ready") / safe_model / f"{safe_var}_section.png"


def _plot_ready_section(values, altitude, latitudes, model, label, units, path):
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
    fig.suptitle(f"{model} · {label} · application-ready zonal section")
    fig.colorbar(image, ax=axes, label=units or "Value", shrink=0.85)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _ready_item(info, ready_base, combined_base):
    display = info["display"]
    card = [f'<h3>{escape(display)}</h3>',
            f'<p class="product-sub">Application-ready · IG2 v{ig2.VERSION} upper-tail completion</p>']
    meta = (f'<p class="product-meta">12 months · {info["latitude"]}° latitude · '
            f'{info["altitude"]} km altitude · NetCDF'
            f'{escape(" · " + info["size"]) if info.get("size") else ""}</p>')
    card.append(meta)
    href = escape(f"{ready_base.rstrip('/')}/{info['filename']}", quote=True)
    card.append(f'<a class="download" href="{href}" download>Download NetCDF</a>')
    if info.get("base_product"):
        derived = escape(f"{combined_base.rstrip('/')}/{info['base_product']}", quote=True)
        card.append('<p class="product-meta">Derived from '
                    f'<a href="{derived}">the compact combined product</a>.</p>')
    card.append(f'<p class="product-file"><code>{escape(info["filename"])}</code></p>')
    head = f'<article class="product">{"".join(card)}</article>'
    figures = []
    for entry in info["entries"]:
        alt = escape(f"{display} {entry['display']} application-ready zonal section", quote=True)
        source = escape(entry["url"], quote=True)
        completion = "IG2 upper completion" if entry["completed"] else "edge-fill completion"
        label = escape(entry["display"])
        units = escape(entry["units"] or "Value")
        figures.append(
            f'<figure><img loading="lazy" src="{source}" alt="{alt}">'
            f'<figcaption><span>{label} · {units}</span>'
            f'<span>{escape(completion)}</span></figcaption></figure>')
    plots = f'<div class="ready-plots">{"".join(figures)}</div>' if figures else ""
    return f'<section class="ready-product">{head}{plots}</section>'


COMBINED_NAME = re.compile(
    r"jumacs_(?P<model>[a-z0-9-]+?)_waccmx_(?P<start>\d{4})-(?P<end>\d{4})"
    r"_(?P<latitude>\d+)deg_(?P<altitude>\d+)km\.nc")


def _model_display_name(slug):
    for name in config.model_names():
        if name.lower() == slug:
            return config.model_metadata(name)["display_name"]
    return slug.upper()


def _model_for_slug(slug):
    for name in config.model_names():
        if name.lower() == slug:
            return name, config.model_metadata(name)["display_name"]
    return slug, slug.upper()


def _human_size(nbytes):
    value = float(nbytes)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000.0


def _format_step(step, default):
    if step is None:
        return default
    return f"{step:g}"


def _combined_item(name, base, sizes=None):
    href = escape(f"{base.rstrip('/')}/{name}", quote=True)
    size = f" · {_human_size(sizes[name])}" if sizes and name in sizes else ""
    match = COMBINED_NAME.fullmatch(name)
    if match:
        card = [f'<h3>{escape(_model_display_name(match["model"]))}</h3>',
                f'<p class="product-sub">Combined with WACCM-X · {match["start"]}–{match["end"]}</p>',
                '<p class="product-meta">12 months · '
                f'{match["latitude"]}° latitude · {match["altitude"]} km altitude · NetCDF{escape(size)}</p>']
    else:
        card = [f'<h3>{escape(name)}</h3>']
    card.append(f'<a class="download" href="{href}" download>Download NetCDF</a>')
    card.append(f'<p class="product-file"><code>{escape(name)}</code></p>')
    return f'<article class="product">{"".join(card)}</article>'


def _html(records, start_year, end_year, inventory_rows, provenance_rows=(), combined_files=(),
          combined_sizes=None, ready_products=()):
    catalog = []
    for model, product, fields in records:
        for field in fields:
            catalog.append({"model": model, "variable": field["native"],
                            "canonical": field["canonical"], "label": field["label"],
                            "display": _display_name(field["canonical"]), "units": field["units"],
                            "product": f"products/application/{model}/{product.name}",
                            "views": {name: path.as_posix() for name, path in field["views"].items()}})
    payload = json.dumps(catalog, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    inventory_payload = json.dumps(inventory_rows, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    provenance_payload = json.dumps(provenance_rows, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    base = os.environ.get("JUMACS_PRODUCT_URL_BASE", "").strip() or "products/combined"
    combined_html = "".join(_combined_item(name, base, combined_sizes) for name in combined_files) \
        or '<p>No combined products are published for this period yet.</p>'
    ready_base = os.environ.get("JUMACS_READY_URL_BASE", "").strip() or "products/application-ready"
    ready_html = "".join(_ready_item(info, ready_base, base) for info in ready_products) \
        or '<p>No application-ready products are published for this period yet.</p>'
    year = config.reference_period()["reference_period"]["nominal_reference_year"]
    return (HTML_TEMPLATE.replace("__PERIOD__", f"{start_year}–{end_year}")
            .replace("__MAP_YEAR__", str(year))
            .replace("__CATALOG__", payload).replace("__INVENTORY__", inventory_payload)
            .replace("__PROVENANCE__", provenance_payload)
            .replace("__COMBINED_COUNT__", str(len(combined_files)))
            .replace("__COMBINED__", combined_html)
            .replace("__READY_COUNT__", str(len(ready_products)))
            .replace("__READY__", ready_html)
            .replace("__IG2_VERSION__", ig2.VERSION))


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>JuMACS · Climatology atlas</title><link rel="stylesheet" href="assets/style.css"></head>
<body><header><h1>JuMACS climatology atlas</h1>
<p>JuMACS v0.1 · Initial public release · Harmonized multi-source model climatologies</p>
<nav><button id="atlas-nav" class="active">Plots</button><button id="inventory-nav">Data availability</button>
<button id="combined-nav">Products</button><button id="ready-nav">Application-ready</button>
<button id="provenance-nav">Data provenance</button>
<button id="about-nav">About &amp; contact</button></nav></header>
<main id="atlas"><aside><label for="model">Model / product</label><select id="model"></select>
<label for="search">Variable</label><input id="search" type="search" placeholder="Search variable name">
<div id="variables" aria-label="Available variables"></div></aside>
<section class="viewer"><div class="topline"><div><h2 id="title"></h2><p class="meta" id="meta"></p></div>
<div class="controls"><button id="prev" title="Previous variable">← Previous</button>
<button id="next" title="Next variable">Next →</button></div></div>
<div class="controls tabs"><button class="tab active" data-view="zonal">Four cross sections</button>
<button class="tab" data-view="timeline">Monthly time series</button>
<button class="tab" data-view="annual">Annual cycle</button>
<button class="tab" data-view="map">Global maps</button></div>
<figure><img id="plot" alt=""><figcaption><span id="caption"></span>
<span><a class="download" id="png" download>Download PNG</a>
<a class="download" id="product" download>Download NetCDF</a></span></figcaption></figure>
<p class="plot-note" data-note="zonal" hidden>January, April, July and October on the JuMACS application pressure grid.
The right axes on the April and October panels show an approximate height using a fixed 7 km scale height.</p>
<p class="plot-note" data-note="timeline" hidden>Original monthly zonal data at exactly 1000, 100, 10 and 1 hPa.
The curves compare tropical, northern and southern midlatitude, and northern and southern polar area means.
The time series may span a longer source period than the 1985–2014 application climatology and are shown only
where the corresponding source series exists. Some 1000 hPa curves may therefore be absent.
Missing values are not filled.</p>
<p class="plot-note" data-note="annual" hidden>Twelve climatological months sampled at exactly 1000, 100, 10 and 1 hPa.
Curves are shown only where the source data reach the requested pressure level. Missing values are not filled.</p>
<p class="plot-note" data-note="map" hidden>Geographical monthly fields from the single example year __MAP_YEAR__,
shown for January, April, July and October. For 3D fields, maps are sampled at 1000, 100, 10 and 1 hPa
where the native pressure grid permits; for 2D fields, four surface maps are shown.
These maps are examples from one year and are not 1985–2014 climatological means.</p>
</section></main><section id="inventory-view" hidden><h2>Data availability</h2>
<p>Application climatologies are zonal means: their spatial axes are pressure × latitude or latitude only.
Longitude-resolved marks variables whose raw archive provides a geographical longitude × latitude
field mappable to pressure (shown as an example-year map); sources published only as zonal means, or
whose archived vertical levels cannot be mapped to pressure, show no.
Pressure limits use any finite mean value. Click a column heading to sort (↕); ↑ and ↓ show the current order.</p>
<label for="inventory-search">Filter model or variable</label><input id="inventory-search" type="search" placeholder="Search inventory">
<div class="table-wrap"><table><thead><tr>
<th><button class="sort-button" data-sort="model">Model</button></th>
<th><button class="sort-button" data-sort="canonical_variable">Variable</button></th>
<th><button class="sort-button" data-sort="units">Units</button></th>
<th><button class="sort-button" data-sort="dimensionality">Dimensionality</button></th>
<th><button class="sort-button" data-sort="map_plot">Longitude-resolved</button></th>
<th><button class="sort-button" data-sort="bottom_pressure_hpa">Vertical range (hPa)</button></th>
<th><button class="sort-button" data-sort="waccmx_extended">WACCM-X extended</button></th>
<th><button class="sort-button" data-sort="application_available">Application product</button></th>
</tr></thead><tbody id="inventory-body"></tbody></table></div></section>
<section id="provenance-view" hidden><h2>Data provenance</h2>
<p>Every source dataset JuMACS processes, with its archive, citation identifiers, licence and the volume of
raw archive data mirrored for the configured reference period. Sizes are decimal units (1 GB = 10^9 bytes)
of the regular files held under each source directory, so they describe this installation rather than the
full archive. Identifier links point to a registered DOI where one exists and otherwise to the stable
archive record; a dataset UUID covers the archive collection or dataset as noted.</p>
<div id="provenance-list"></div>
</section>
<section id="combined-view" hidden><h2>Combined climatology products</h2>
<div class="product-intro"><p>JuMACS provides one compact combined climatology product per active primary CCMI model.
Each file merges the harmonized lower and middle-atmosphere climatology with WACCM-X upper-atmosphere information
where the established extension rules apply. The current public set contains __COMBINED_COUNT__ curated
primary-model products.</p>
<p>Every product covers 12 climatological months at 5° latitude resolution on a 1 km altitude grid spanning
approximately 0–120 km as supported by the individual fields, and is distributed as NetCDF. Not every variable
necessarily reaches 120 km: fields without an established extension stop at the available model top.
No bias correction or gap filling is applied. The
<a href="https://github.com/slcs-jsc/jumacs/blob/main/docs/methodology.md">combination methodology</a> documents
the extension rules and the blending range.</p></div>
<div class="products-grid">__COMBINED__</div>
<p class="products-note">Model-specific application and native climatology products remain available through the plots
and data-availability views; this page focuses on the compact combined products intended for convenient
download and use.</p></section>
<section id="ready-view" hidden><h2>Application-ready products</h2>
<div class="product-intro"><p>Application-ready products extend each compact combined climatology into a complete
profile with no open boundaries: wherever a model or the WACCM-X extension leaves the upper tail open, the profile
is continued with IG2 v__IG2_VERSION__ standard atmospheres through a short raised-cosine transition, and the lowest
and highest levels are carried to the domain edges. Interior gaps are preserved and no bias correction is applied.
Each file holds 12 climatological months at 5° latitude on a 1 km altitude grid (approximately 0–120 km) as NetCDF,
ready for direct use in modelling and radiative-transfer tools such as JURASSIC and MPTRAC.</p>
<p>The current set contains __READY_COUNT__ published product(s) for this period. The
<a href="https://github.com/slcs-jsc/jumacs/blob/main/docs/methodology.md">combination methodology</a> documents the
vertical extension and blending, and IG2 v__IG2_VERSION__ provides the upper-tail reference atmospheres.</p></div>
__READY__</section>
<section id="about-view" hidden><h2>About JuMACS</h2>
<p>JuMACS — Jülich Multi-source Atmospheric Climatology and Statistics — provides reproducible monthly
atmospheric climatologies and associated statistics for scientific applications.</p>
<p>JuMACS combines published chemistry–climate model data from multiple sources into harmonized products on a
common application grid while preserving the individual source-model climatologies and their provenance.
The current products are designed in particular for use with atmospheric modelling and radiative-transfer tools
such as JURASSIC and MPTRAC.</p>
<p>The project provides native-model climatologies, harmonized application products, diagnostic visualizations, and
detailed source-data provenance. Original source datasets remain subject to their respective licences and citation
requirements.</p>
<p class="disclaimer">JuMACS is under active development. The public site provides the current model-based
climatologies, application products, diagnostic atlas, and source-data provenance. Additional datasets, evaluation
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
let current = null, view = 'zonal';
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
    for (const note of document.querySelectorAll('.plot-note')) note.hidden = true;
    return;
  }
  if (!current.views[view]) view = current.views.zonal ? 'zonal' : Object.keys(current.views)[0];
  document.getElementById('title').textContent = current.model + ' · ' + current.display;
  document.getElementById('meta').textContent = current.canonical + ' · ' + current.units + ' · __PERIOD__';
  const image = document.getElementById('plot');
  image.src = current.views[view];
  image.alt = current.model + ' ' + current.display + ' ' + view;
  document.getElementById('png').href = current.views[view];
  document.getElementById('product').href = current.product;
  document.getElementById('caption').textContent = {
    zonal: 'Monthly climatology · four latitude–pressure sections',
    timeline: 'Original monthly zonal time series', annual: 'Twelve-month climatological annual cycle',
    map: 'Geographical monthly fields from __MAP_YEAR__ (single year)'
  }[view];
  for (const tab of document.querySelectorAll('.tab')) {
    tab.disabled = !current.views[tab.dataset.view];
    tab.classList.toggle('active', tab.dataset.view === view);
  }
  for (const note of document.querySelectorAll('.plot-note')) note.hidden = note.dataset.note !== view;
}
model.onchange = () => {
  const canonical = current && current.canonical;
  const match = catalog.find(row => row.model === model.value && row.canonical === canonical);
  if (match && search.value && !available().includes(match)) search.value = '';
  current = match || null;
  render();
};
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
    const values = [row.model, row.canonical_variable || '—', row.units,
      row.dimensionality || '—', row.map_plot ? 'yes' : 'no', pressure,
      row.waccmx_extended == null ? 'unknown' : row.waccmx_extended ? 'yes' : 'no', null];
    for (const [index, value] of values.entries()) {
      const cell = document.createElement('td');
      if (index === 7 && row.application_available && row.application_product) {
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
               ['combined-nav', 'combined-view'], ['ready-nav', 'ready-view'],
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
main[hidden],#inventory-view[hidden],#provenance-view[hidden],#combined-view[hidden],#ready-view[hidden],#about-view[hidden]{display:none}
#inventory-view,#provenance-view,#combined-view,#ready-view,#about-view{max-width:1420px;margin:auto;padding:1.5rem}
#about-view{max-width:60rem}
#about-view p{max-width:75ch}
#about-view h2:not(:first-child){margin-top:1.6rem}
#about-view a{color:var(--accent)}
#about-view address.contact{font-style:normal;line-height:1.6;margin:0}
.project-links{margin:.6rem 0 0;padding-left:1.2rem;display:grid;gap:.45rem;max-width:75ch}
#combined-view p{max-width:90ch}
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
#ready-view p{max-width:90ch}
.ready-product{display:grid;gap:1rem;margin-top:1.2rem}
.ready-product .product{max-width:44rem}
.ready-plots{display:grid;grid-template-columns:repeat(auto-fill,minmax(30rem,1fr));gap:1rem}
.ready-plots figure{margin:0}
.ready-plots figcaption span:last-child{color:var(--accent)}
.ready-plots img{background:white}
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
.tab.active{background:var(--accent);border-color:var(--accent);color:white}.tab:disabled{opacity:.4;cursor:not-allowed}
button:hover:not(:disabled),a.download:hover{border-color:var(--accent)}.tabs{margin-top:1.2rem}
figure{background:white;border:1px solid var(--line);border-radius:.8rem;margin:1rem 0 .6rem;padding:.65rem;box-shadow:0 8px 28px #12364b0a}
figure img{display:block;width:100%;max-height:calc(100vh - 220px);object-fit:contain}
figcaption{display:flex;justify-content:space-between;align-items:center;gap:1rem;color:var(--muted);font-size:.88rem;margin-top:.7rem}
figcaption span:last-child{display:flex;gap:.4rem;flex-wrap:wrap}
.plot-note{margin:.15rem 0 0;color:#3f6472;font-size:.86rem;max-width:90ch}
@media(max-width:760px){main{display:block}aside{border-right:0;border-bottom:1px solid var(--line)}
#variables{display:flex;max-height:none;overflow:auto}.item{white-space:nowrap}.viewer{padding:1rem}figure img{max-height:none}
.dataset{grid-template-columns:minmax(0,1fr)}.products-grid{grid-template-columns:minmax(0,1fr)}.ready-plots{grid-template-columns:minmax(0,1fr)}}
"""


def _render_model(model_name, product, root, staging):
    """Render one model in its own staging folder; safe to run in a worker process."""
    fields = []
    skipped = []
    timeline_skipped = []
    map_skipped = []
    model_config = config.load_config(model_name)
    map_year = config.reference_period()["reference_period"]["nominal_reference_year"]
    mapping = {native: canonical for canonical, native in model_config["variables"].items()}
    with xr.open_dataset(product) as dataset:
        pressure_hpa = np.asarray(dataset.pressure.values, dtype=float) / 100.0
        latitudes = np.asarray(dataset.lat.values, dtype=float)
        for name in sorted(dataset.data_vars):
            if not name.endswith("_mean"):
                continue
            field = dataset[name]
            three_dimensional = field.dims == ("time", "pressure", "lat")
            two_dimensional = field.dims == ("time", "lat")
            if field.sizes.get("time") != 12 or not (three_dimensional or two_dimensional):
                skipped.append((model_name, name, "not a 12-month time×pressure×lat or time×lat field"))
                continue
            native = name[:-5]
            canonical = mapping.get(native, native)
            label = canonical.replace("_", " ")
            units = str(field.attrs.get("units", ""))
            views = {}
            if three_dimensional:
                values = np.asarray(field.values)
                if not np.isfinite(values[np.array(ZONAL_MONTHS) - 1]).any():
                    skipped.append((model_name, name, "no finite values in displayed months"))
                    continue
                paths = _plot_paths(model_name, native)
                for path in paths:
                    (staging / path).parent.mkdir(parents=True, exist_ok=True)
                scale, plot_units = _display_scale(values, units)
                _plot_zonal(values * scale, pressure_hpa, latitudes, model_name, label,
                            plot_units, staging / paths[0])
                _plot_annual(field, model_name, label, plot_units, staging / paths[1], scale=scale)
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
            map_source = _reference_year_source(root, model_config, native, map_year)
            if map_source is None:
                map_skipped.append((model_name, native, f"no Amon/AmonZ source covering {map_year}"))
            else:
                try:
                    map_values, map_lat, map_lon, map_levels, map_units = _map_values(
                        map_source, model_name, native, map_year)
                    map_path = _map_path(model_name, native)
                    (staging / map_path).parent.mkdir(parents=True, exist_ok=True)
                    if _plot_map(map_values, map_lat, map_lon, map_levels, model_name,
                                 label, map_units, map_year, staging / map_path):
                        views["map"] = map_path
                    else:
                        map_skipped.append((model_name, native, "no finite map values"))
                except (KeyError, ValueError) as exc:
                    map_skipped.append((model_name, native, str(exc)))
            if not views:
                skipped.append((model_name, name, "no geographical source for 2D field"))
                continue
            fields.append({"native": native, "canonical": canonical, "label": label,
                           "units": units, "views": views})
    copy = staging / "products" / "application" / model_name / product.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(product, copy)
    return (model_name, product, fields), skipped, timeline_skipped, map_skipped


def _render_ready_product(model_dir, display, product, staging):
    """Render latitude-altitude cross sections for one application-ready product."""
    entries = []
    skipped = []
    base_product = None
    latitude_step = None
    altitude_step = None
    with xr.open_dataset(product) as dataset:
        base_product = dataset.attrs.get("base_product")
        if "altitude" not in dataset.coords or "lat" not in dataset.coords or "month" not in dataset.coords:
            return entries, skipped, base_product, latitude_step, altitude_step
        altitude = np.asarray(dataset.altitude.values, dtype=float)
        latitudes = np.asarray(dataset.lat.values, dtype=float)
        if dataset.coords["month"].size != 12:
            return entries, skipped, base_product, latitude_step, altitude_step
        if altitude.size > 1:
            altitude_step = float(np.round(np.median(np.diff(altitude)), 6))
        if latitudes.size > 1:
            latitude_step = float(np.round(np.median(np.diff(latitudes)), 6))
        for name in sorted(dataset.data_vars):
            field = dataset[name]
            if field.dims != ("month", "altitude", "lat") or field.sizes["month"] != 12:
                skipped.append((model_dir, name, "not a 12-month month×altitude×lat field"))
                continue
            values = np.asarray(field.values, dtype=float)
            displayed = values[np.array(ZONAL_MONTHS) - 1]
            if not np.isfinite(displayed).any():
                skipped.append((model_dir, name, "no finite values in displayed months"))
                continue
            units = str(field.attrs.get("units", ""))
            scale, plot_units = _display_scale(values, units)
            completed = str(field.attrs.get("ig2_completed", "")).lower() == "true"
            path = _ready_plot_path(model_dir, name)
            (staging / path).parent.mkdir(parents=True, exist_ok=True)
            _plot_ready_section(values * scale, altitude, latitudes, display,
                                _display_name(name), plot_units, staging / path)
            entries.append({"canonical": name, "display": _display_name(name),
                            "units": plot_units, "completed": completed, "url": path.as_posix()})
    return entries, skipped, base_product, latitude_step, altitude_step


def build_site(start_year=None, end_year=None, *, model=None, root=None, workers=4):
    """Plot existing products and publish a deployable tree below ``root/site``."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    root = config.ROOT if root is None else Path(root)
    period = config.reference_period()["reference_period"]
    start_year = period["start_year"] if start_year is None else start_year
    end_year = period["end_year"] if end_year is None else end_year
    products = discover_application_products(root, start_year, end_year, model)
    ready_products = discover_application_ready_products(root, start_year, end_year, model)
    from .inventory import build_inventory
    from .provenance import build_provenance
    inventory = build_inventory(start_year, end_year, model=model, root=root)
    provenance = build_provenance(root=root)
    site = root / "site"
    site.mkdir(parents=True, exist_ok=True)
    staging = site / ".browse-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    records = []
    skipped = []
    timeline_skipped = []
    map_skipped = []
    try:
        tasks = [(name, product, root, staging) for name, product in products]
        if workers == 1 or len(tasks) == 1:
            results = [_render_model(*task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                     mp_context=get_context("spawn")) as pool:
                results = list(pool.map(_render_model, *zip(*tasks)))
        for record, model_skipped, model_timeline_skipped, model_map_skipped in results:
            records.append(record)
            skipped.extend(model_skipped)
            timeline_skipped.extend(model_timeline_skipped)
            map_skipped.extend(model_map_skipped)
        assets = staging / "assets"
        assets.mkdir()
        (assets / "style.css").write_text(STYLE, encoding="utf-8")
        site_inventory = []
        copied = {name for name, _, _ in records}
        maps = {(name, field["native"]): field["views"]["map"].as_posix()
                for name, _, fields in records for field in fields if "map" in field["views"]}
        for row in inventory["rows"]:
            entry = dict(row)
            if row["model"] in copied and row["application_product"]:
                entry["application_product"] = row["application_product"]
            else:
                entry["application_product"] = None
            entry["map_plot"] = maps.get((row["model"], row["native_variable"]))
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
        combined_files = []
        combined_sizes = {}
        combined_source = root / "products" / "climatology" / "combined"
        if combined_source.is_dir():
            target_dir = staging / "products" / "combined"
            for source in sorted(combined_source.glob("*.nc")):
                if model is not None and model.lower() not in source.name.lower():
                    continue
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target_dir / source.name)
                combined_files.append(source.name)
                combined_sizes[source.name] = source.stat().st_size
        ready_infos = []
        ready_skipped = []
        for slug, path in ready_products:
            model_dir, display = _model_for_slug(slug)
            entries, model_skipped, base_product, latitude_step, altitude_step = \
                _render_ready_product(model_dir, display, path, staging)
            ready_skipped.extend(model_skipped)
            target = staging / "products" / "application-ready" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            ready_infos.append({"model_dir": model_dir, "display": display,
                                "filename": path.name, "size": _human_size(path.stat().st_size),
                                "latitude": _format_step(latitude_step, "5"),
                                "altitude": _format_step(altitude_step, "1"),
                                "base_product": base_product, "entries": entries})
        (staging / "index.html").write_text(
            _html(records, start_year, end_year, site_inventory, provenance["sources"], combined_files,
                  combined_sizes, ready_infos),
            encoding="utf-8")
        for name in ("plots", "plots-ready", "assets", "index.html"):
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
        climatology = site / "products" / "climatology"
        staged_climatology = staging / "products" / "climatology"
        if climatology.is_dir() and not climatology.is_symlink():
            shutil.rmtree(climatology)
        elif climatology.exists() or climatology.is_symlink():
            climatology.unlink()
        if staged_climatology.exists():
            staged_climatology.replace(climatology)
        published_combined = site / "products" / "combined"
        staged_combined = staging / "products" / "combined"
        if published_combined.is_dir() and not published_combined.is_symlink():
            shutil.rmtree(published_combined)
        elif published_combined.exists() or published_combined.is_symlink():
            published_combined.unlink()
        if staged_combined.exists():
            staged_combined.replace(published_combined)
        published_ready = site / "products" / "application-ready"
        staged_ready = staging / "products" / "application-ready"
        if published_ready.is_dir() and not published_ready.is_symlink():
            shutil.rmtree(published_ready)
        elif published_ready.exists() or published_ready.is_symlink():
            published_ready.unlink()
        if staged_ready.exists():
            staged_ready.replace(published_ready)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"site": site, "models": [name for name, _, _ in records],
            "provenance_sources": len(provenance["sources"]),
            "variables": {name: len(fields) for name, _, fields in records},
            "png_files": sum(len(field["views"]) for _, _, fields in records for field in fields),
            "netcdf_files": len(records), "climatology_files": len(published_native),
            "combined_files": combined_files, "skipped": skipped,
            "ready_files": [path.name for _, path in ready_products], "ready_skipped": ready_skipped,
            "timeline_skipped": timeline_skipped, "map_skipped": map_skipped}
