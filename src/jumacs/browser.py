"""Generate a self-contained static browser for application climatologies."""

import re
import shutil
from html import escape
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
    sections = []
    for model, product, fields in records:
        cards = []
        download = f"products/application/{model}/{product.name}"
        for field in fields:
            zonal, annual = field["paths"]
            label = escape(field["label"])
            units = escape(field["units"])
            cards.append(
                f'<article class="variable"><h3>{label} <small>({escape(field["native"])})</small></h3>'
                f'<p>Units: {units or "unspecified"} · <a href="{escape(download)}" download>Download NetCDF</a></p>'
                f'<div class="previews"><a href="{zonal.as_posix()}"><img src="{zonal.as_posix()}" '
                f'alt="{label} zonal climatology"></a><a href="{annual.as_posix()}">'
                f'<img src="{annual.as_posix()}" alt="{label} annual cycle"></a></div></article>')
        sections.append(f'<section id="{escape(model)}"><h2>{escape(model)}</h2>' + "\n".join(cards) + "</section>")
    navigation = " ".join(f'<a href="#{escape(model)}">{escape(model)}</a>' for model, _, _ in records)
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>JuMACS application climatologies {start_year}–{end_year}</title>'
            '<link rel="stylesheet" href="assets/style.css"></head><body><main>'
            f'<header><h1>JuMACS application climatologies</h1><p>{start_year}–{end_year} · '
            'Zonal means on the shared pressure and latitude grid</p></header>'
            f'<nav>{navigation}</nav>' + "\n".join(sections) + "</main></body></html>\n")


STYLE = """body{font:16px/1.5 system-ui,sans-serif;color:#152b3a;background:#f4f7f9;margin:0}
main{max-width:1200px;margin:auto;padding:1.5rem}header,nav,section{margin-bottom:2rem}
nav{display:flex;gap:1rem;flex-wrap:wrap}a{color:#075b88}section{border-top:2px solid #b8cbd5}
.variable{background:white;border:1px solid #d9e2e8;border-radius:8px;padding:1rem;margin:1rem 0}
h3{margin:0}small{font-weight:normal;color:#526774}.variable p{margin:.25rem 0 1rem}
.previews{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}
.previews img{display:block;width:100%;height:auto;border:1px solid #e4e9ec}
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
    try:
        for model_name, product in products:
            fields = []
            mapping = {native: canonical for canonical, native in
                       config.load_config(model_name)["variables"].items()}
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
                    fields.append({"native": native, "label": label, "units": units, "paths": paths})
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
            "png_files": sum(2 * len(fields) for _, _, fields in records),
            "netcdf_files": len(records), "skipped": skipped}
