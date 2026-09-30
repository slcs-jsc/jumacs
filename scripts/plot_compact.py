#!/usr/bin/env python3
"""Plot compact application fields and add them to the offline atlas."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from jumacs.config import ROOT


def main():
    records = []
    for model in ("GEOSCCM", "EMAC"):
        name = f"jumacs_{model.lower()}_waccmx_1985-2014_5deg_1km.nc"
        source = ROOT / "products/climatology/combined" / name
        if not source.exists():
            continue
        with xr.open_dataset(source) as ds:
            for species in ("o3", "ch4", "cfc_11", "cfc_12", "temperature", "no", "co2"):
                if species not in ds:
                    continue
                field = ds[species]
                months = (1, 4, 7, 10)
                values = [field.sel(month=m).values for m in months]
                finite = np.concatenate([v[np.isfinite(v)] for v in values])
                if finite.size == 0:
                    continue
                lo, hi = np.nanpercentile(finite, [1, 99])
                if lo == hi:
                    hi = lo + 1
                fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, dpi=125)
                for ax, month, value in zip(axes.flat, months, values):
                    mesh = ax.pcolormesh(ds.lat, ds.altitude, value, vmin=lo, vmax=hi,
                                         shading="auto", cmap="viridis", rasterized=True)
                    ax.axhspan(55, 65, color="white", alpha=.12)
                    ax.axhline(55, color="white", lw=.5, alpha=.8)
                    ax.axhline(65, color="white", lw=.5, alpha=.8)
                    ax.set(title={1:"January",4:"April",7:"July",10:"October"}[month],
                           xlim=(-85, 85), ylim=(0, 120))
                for ax in axes[1]: ax.set_xlabel("Latitude (°N)")
                for ax in axes[:, 0]: ax.set_ylabel("Geopotential height (km)")
                fig.suptitle(f"{model} → WACCM-X · {field.attrs['canonical_species']} · 1985–2014", weight="bold")
                fig.subplots_adjust(left=.08, right=.87, bottom=.09, top=.9, hspace=.22, wspace=.12)
                fig.colorbar(mesh, ax=axes, fraction=.028, pad=.03, label=field.attrs["units"])
                dest = ROOT / "site/plots" / f"{model}-WACCM-X" / f"{species}_compact.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(dest, facecolor="white")
                plt.close(fig)
                records.append({"model": f"{model}→WACCM-X", "variable": species,
                                "label": field.attrs["canonical_species"], "units": field.attrs["units"],
                                "first_year": 1985, "last_year": 2014, "months": 12,
                                "views": {"compact": str(dest.relative_to(ROOT / "site"))}})
    output = ROOT / "site/compact_catalog.json"
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"{len(records)} compact field plots")


if __name__ == "__main__":
    main()
