#!/usr/bin/env python3
"""Plot representative model differences on their common pressure/latitude grid."""
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from jumacs.config import ROOT

DEST = ROOT / "site"
MONTHS = (1, 4, 7, 10)
TARGETS = ("temperature", "O3", "H2O", "CH4", "N2O", "CFC-11", "CFC-12")
PAIRS = (("GEOSCCM", "EMAC"), ("GEOSCCM", "WACCM-X"), ("EMAC", "WACCM-X"))


def plot(path, first, second, species, dest):
    with xr.open_dataset(path) as ds:
        fields = [ds.difference.sel(month=month).transpose("pressure", "lat").values for month in MONTHS]
        finite = np.concatenate([np.abs(f[np.isfinite(f)]) for f in fields])
        if not finite.size:
            return False
        limit = float(np.percentile(finite, 98))
        if limit <= 0:
            limit = 1e-30
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=125, sharex=True, sharey=True)
        mesh = None
        for ax, month, values in zip(axes.flat, MONTHS, fields):
            mesh = ax.pcolormesh(ds.lat.values, ds.pressure.values / 100., values,
                                 shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
            ax.set(title={1:"January",4:"April",7:"July",10:"October"}[month], yscale="log",
                   ylim=(1000, 1), xlim=(-90, 90))
        for ax in axes[1]: ax.set_xlabel("Latitude (°N)")
        for ax in axes[:, 0]: ax.set_ylabel("Pressure (hPa)")
        fig.suptitle(f"{first} − {second} · {species} · 1985–2014", weight="bold")
        fig.subplots_adjust(left=.08, right=.87, bottom=.08, top=.91, hspace=.2, wspace=.12)
        fig.colorbar(mesh, ax=axes, fraction=.028, pad=.03, label=ds.difference.attrs.get("units", "native units"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, facecolor="white")
        plt.close(fig)
    return True


def main():
    records = []
    for first, second in PAIRS:
        report = ROOT / "products/comparison" / f"comparison_{first.lower()}_vs_{second.lower()}_1985-2014.json"
        if not report.exists():
            continue
        compared = set(json.loads(report.read_text())["compared"])
        for species in TARGETS:
            if species not in compared:
                continue
            src = ROOT / "products/comparison" / f"jumacs_{first.lower()}_vs_{second.lower()}_{species.lower()}_1985-2014.nc"
            dest = DEST / "plots" / "Comparison" / f"{first.lower()}_vs_{second.lower()}_{species.lower()}_difference.png"
            if src.exists() and plot(src, first, second, species, dest):
                records.append({"model":"Comparison", "variable": species,
                    "label": f"{first} − {second} · {species}", "units":"difference in native units",
                    "first_year":1985, "last_year":2014, "months":360,
                    "views":{"comparison":str(dest.relative_to(DEST))}})
    source = ROOT / "products/diagnostics/coverage/vertical_coverage.png"
    if source.exists():
        dest = DEST / "plots/Comparison/vertical_coverage.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        records.append({"model":"Comparison", "variable":"vertical_coverage",
            "label":"Vertical coverage", "units":"Pa", "first_year":1950,
            "last_year":2018, "months":0, "views":{"coverage":str(dest.relative_to(DEST))}})
    (DEST / "comparison_catalog.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(len(records), "comparison and coverage plots")


if __name__ == "__main__":
    main()
