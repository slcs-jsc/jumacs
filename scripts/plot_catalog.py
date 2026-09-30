#!/usr/bin/env python3
"""Build lightweight JuMACS plots from model-specific monthly zonal products."""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import xarray as xr

from jumacs.config import ROOT, load_config

DEST = ROOT / "site"
MONTHS = (1, 4, 7, 10)
PERIOD = (1985, 2014)


def scale(a, units):
    good = np.abs(a[np.isfinite(a)])
    peak = float(np.percentile(good, 98)) if good.size else 0.0
    if units in ("mol mol-1", "mol/mol"):
        power, label = ((6, "ppmv") if peak >= 1e-6 else
                        (9, "ppbv") if peak >= 1e-9 else
                        (12, "pptv") if peak >= 1e-12 else (15, "ppqv"))
        return a * 10**power, label
    if units == "Pa":
        return a / 100, "hPa"
    if units == "m":
        return a / 1000, "km"
    return a, units or "native units"


def read(src, var):
    with xr.open_dataset(src, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        da = ds[var]
        if "time" not in da.dims or "lat" not in da.dims:
            return None
        level = next((d for d in ("lev", "plev") if d in da.dims), None)
        order = ("time", level, "lat") if level else ("time", "lat")
        a, units = scale(da.transpose(*order).values, da.attrs.get("units", ""))
        p = None
        if level:
            if "air_pressure" in ds:
                p = ds.air_pressure.transpose(*order).values / 100
            else:
                z = ds[level]
                factor = {"Pa": .01, "hPa": 1, "mbar": 1}.get(z.attrs.get("units"))
                if factor is not None:
                    p = np.broadcast_to(z.values[None, :, None] * factor, a.shape)
        t = ds.time.values
        return a, p, ds.lat.values, np.array([v.year for v in t]), np.array([v.month for v in t]), units


def timeline(model, name, a, p, lat, years, months, units, dest):
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=125)
    x = years + (months - .5) / 12
    if p is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            for pressure, color in ((300, "#078c9d"), (30, "#df7344")):
                distance = np.where((p > 0) & np.isfinite(p), np.abs(np.log(p / pressure)), np.inf)
                index = np.argmin(distance, axis=1)
                layer = np.take_along_axis(a, index[:, None, :], axis=1)[:, 0, :]
                for lo, hi, style, band in ((-20, 20, "-", "tropics"), (45, 75, "--", "45–75°N")):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        y = np.nanmean(layer[:, (lat >= lo) & (lat <= hi)], axis=1)
                    ax.plot(x, y, color=color, ls=style, lw=1.3, label=f"{pressure} hPa · {band}")
    else:
        for lo, hi, color, band in ((-20, 20, "#078c9d", "tropics"), (45, 75, "#df7344", "45–75°N")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                y = np.nanmean(a[:, (lat >= lo) & (lat <= hi)], axis=1)
            ax.plot(x, y, color=color, lw=1.3, label=band)
    ax.set(xlim=(years.min(), years.max() + 1), xlabel="Year", ylabel=units,
           title=f"{model} · {name} · monthly zonal means {years.min()}–{years.max()}")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(dest, facecolor="white")
    plt.close(fig)


def sections(model, name, a, p, lat, years, months, units, dest, pressure_bounds=(1000., 1.)):
    high_pressure, low_pressure = pressure_bounds
    fields, press = [], []
    for month in MONTHS:
        keep = (years >= PERIOD[0]) & (years <= PERIOD[1]) & (months == month)
        if not keep.any():
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fields.append(np.nanmean(a[keep], axis=0))
            press.append(np.nanmean(p[keep], axis=0))
    values = np.concatenate([v[(q >= low_pressure) & (q <= high_pressure) & np.isfinite(v)] for v, q in zip(fields, press)])
    if not values.size:
        return False
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        high = low + max(abs(low) * .01, 1e-15)
    norm = LogNorm(vmin=low, vmax=high) if low > 0 and high / low > 500 else None
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), dpi=125, sharex=True, sharey=True)
    mesh = None
    for ax, month, v, q in zip(axes.flat, MONTHS, fields, press):
        grid = np.broadcast_to(lat[None, :], v.shape)
        masked = np.where((q >= low_pressure) & (q <= high_pressure), v, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mesh = ax.pcolormesh(grid, q, masked, shading="nearest", cmap="viridis",
                                 norm=norm, vmin=None if norm else low, vmax=None if norm else high,
                                 rasterized=True)
        ax.set(title={1:"January",4:"April",7:"July",10:"October"}[month], yscale="log",
               ylim=(high_pressure, low_pressure), xlim=(-90, 90))
    for ax in axes[1]:
        ax.set_xlabel("Latitude (°N)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pressure (hPa)")
    fig.suptitle(f"{model} · {name} · monthly climatology {PERIOD[0]}–{PERIOD[1]}", weight="bold")
    fig.subplots_adjust(left=.08, right=.88, bottom=.08, top=.91, hspace=.2, wspace=.12)
    fig.colorbar(mesh, ax=axes, fraction=.028, pad=.03, label=units)
    fig.savefig(dest, facecolor="white")
    plt.close(fig)
    return True


def main():
    records = []
    for model in ("GEOSCCM", "EMAC", "WACCM-X"):
        cfg = load_config(model)
        folder = DEST / "plots" / model
        folder.mkdir(parents=True, exist_ok=True)
        for src in sorted((ROOT / cfg["paths"]["zonal"]).glob("*_monthly_zonal.nc")):
            var = src.name.removesuffix("_monthly_zonal.nc")
            name = next((k for k, v in cfg["variables"].items() if v == var), var)
            result = read(src, var)
            if result is None:
                continue
            a, p, lat, years, months, units = result
            path = folder / f"{var}_timeline.png"
            if not path.exists():
                timeline(model, name, a, p, lat, years, months, units, path)
            views = {"timeline": str(path.relative_to(DEST))}
            if p is not None:
                path = folder / f"{var}_sections_{PERIOD[0]}-{PERIOD[1]}.png"
                if path.exists() or sections(model, name, a, p, lat, years, months, units, path):
                    views["sections"] = str(path.relative_to(DEST))
                if model == "WACCM-X":
                    path = folder / f"{var}_upper_{PERIOD[0]}-{PERIOD[1]}.png"
                    if path.exists() or sections(model, name, a, p, lat, years, months, units, path, (1., 1e-10)):
                        views["upper"] = str(path.relative_to(DEST))
            records.append(dict(model=model, variable=var, label=name, units=units,
                                first_year=int(years.min()), last_year=int(years.max()),
                                months=len(years), views=views))
            print(model, var, ",".join(views), flush=True)
    (DEST / "catalog.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    from build_index import build_index
    print("INDEX", build_index(), flush=True)
    print("CATALOG", len(records), "variables", flush=True)


if __name__ == "__main__":
    main()
