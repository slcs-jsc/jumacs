"""Independent data checks, trend plots, and quick-look sections."""
import json
from pathlib import Path
import numpy as np
import xarray as xr

from .config import ROOT, load_config


def validate(model):
    config = load_config(model)
    base = ROOT / config["paths"]["zonal"]
    report = {"model": model, "experiment": config["model"]["experiment"], "variables": {}}
    for path in sorted(base.glob("*_monthly_zonal.nc")):
        name = path.name.removesuffix("_monthly_zonal.nc")
        with xr.open_dataset(path, use_cftime=True) as ds:
            data = ds[name]
            stamps = [(int(t.year), int(t.month)) for t in data.time.values]
            expected = [(y, m) for y in range(min(y for y, _ in stamps), max(y for y, _ in stamps) + 1) for m in range(1, 13)]
            missing = [f"{y:04d}-{m:02d}" for y, m in expected if (y, m) not in stamps]
            latitude = ds.get(config["coordinates"]["latitude"])
            report["variables"][name] = {"first_month": f"{min(stamps)[0]:04d}-{min(stamps)[1]:02d}",
                "last_month": f"{max(stamps)[0]:04d}-{max(stamps)[1]:02d}", "missing_months": missing,
                "duplicate_months": len(stamps) - len(set(stamps)), "units": data.attrs.get("units", ""),
                "dimensions": list(data.dims), "finite_fraction": float(np.isfinite(data).mean()),
                "latitude_range": [float(latitude.min()), float(latitude.max())] if latitude is not None else None,
                "longitude_removed": config["coordinates"]["longitude"] not in data.dims,
                "source_longitude_global": ds.attrs.get("source_longitude_global", "unknown"),
                "zonal_method": data.attrs.get("zonal_method", "")}
    dest = ROOT / "products/diagnostics" / "validation" / f"{model}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    return dest


def trend_plots(model):
    import matplotlib.pyplot as plt
    config = load_config(model)
    outputs = []
    for canonical in ("CH4", "N2O", "CO2", "SF6", "CFC-11", "CFC-12"):
        name = config["variables"].get(canonical)
        if not name:
            continue
        path = ROOT / config["paths"]["zonal"] / f"{name}_monthly_zonal.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path, use_cftime=True) as ds:
            data = ds[name]
            dims = [d for d in data.dims if d != "time"]
            series = data.mean(dims, skipna=True)
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(range(series.sizes["time"]), series.values)
            ax.set_xticks([i for i, t in enumerate(series.time.values) if int(t.month) == 1 and int(t.year) % 5 == 0],
                          [str(int(t.year)) for t in series.time.values if int(t.month) == 1 and int(t.year) % 5 == 0])
            ax.set(title=f"{model} {canonical}: monthly native-grid mean", ylabel=data.attrs.get("units", ""))
            fig.tight_layout()
            dest = ROOT / "products/diagnostics" / "trends" / model / f"{name}.png"
            dest.parent.mkdir(parents=True, exist_ok=True); fig.savefig(dest, dpi=150); plt.close(fig); outputs.append(dest)
    return outputs


def quicklooks(model, start_year, end_year):
    import matplotlib.pyplot as plt
    config = load_config(model)
    outputs = []
    targets = ("temperature", "geopotential_height", "O3", "H2O", "CO2", "CH4", "N2O", "CO", "HNO3", "NO", "NO2", "HCl", "ClO", "ClONO2", "BrO", "HOCl", "N2O5", "HNO4", "CFC-11", "CFC-12", "SF6")
    for canonical in targets:
        name = config["variables"].get(canonical)
        from .climatology import variable_product_name
        path = ROOT / config["paths"]["climatology"] / variable_product_name(model, name, start_year, end_year) if name else None
        if path is None or not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            for month in (1, 4, 7, 10):
                if month not in ds.month:
                    continue
                field = ds["mean"].sel(month=month)
                if "lat" not in field.dims or not any(d in field.dims for d in ("lev", "plev")):
                    continue
                level = "lev" if "lev" in field.dims else "plev"
                values = field.transpose(level, "lat").values
                if "air_pressure" in ds:
                    pressure = ds["air_pressure"].sel(month=month).transpose(level, "lat").values
                elif field[level].attrs.get("standard_name") == "air_pressure":
                    pressure = np.broadcast_to(field[level].values[:, None], values.shape)
                else:
                    continue
                latitude = np.broadcast_to(field.lat.values[None, :], values.shape)
                fig, ax = plt.subplots(figsize=(7, 4))
                mesh = ax.pcolormesh(latitude, pressure / 100., values, shading="auto")
                fig.colorbar(mesh, ax=ax, label=ds.attrs.get("source_units", ""))
                ax.set_yscale("log")
                ax.invert_yaxis()
                ax.set_xlabel("Latitude (degrees north)")
                ax.set_ylabel("Pressure (hPa)")
                ax.set_title(f"{model} {canonical} {start_year}-{end_year} month {month}")
                dest = ROOT / "products/diagnostics" / "quicklook" / model / f"{name}_{start_year}-{end_year}_{month:02d}.png"
                dest.parent.mkdir(parents=True, exist_ok=True); fig.savefig(dest, dpi=150); plt.close(fig); outputs.append(dest)
    return outputs
