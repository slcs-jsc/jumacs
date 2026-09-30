"""Read each source NetCDF without changing its model identity or grid."""
import re
import cftime
import xarray as xr

from .config import load_config


def open_source(path, model, variable):
    config = load_config(model)
    name = config["variables"].get(variable, variable)
    if model == "WACCM-X":
        valid_name = path.name.endswith("_zm.nc") and ".cam.h0." in path.name
    else:
        valid_name = "_refD1_" in path.name and f"_{config['model']['archive_model']}_" in path.name
    if not valid_name:
        raise ValueError(f"Unexpected model or experiment in {path.name}")
    ds = xr.open_dataset(path, decode_times=True, use_cftime=True)
    if name not in ds:
        ds.close(); raise KeyError(f"{name} absent from {path.name}")
    if model == "WACCM-X":
        match = re.search(r"\.(\d{4})-(\d{2})_zm\.nc$", path.name)
        year, month = int(match[1]), int(match[2])
        if ds.sizes.get("time") != 1:
            ds.close(); raise ValueError(f"Expected one monthly sample in {path.name}")
        original_time = str(ds.time.values[0])
        if "time_bnds" in ds:
            lower, upper = ds.time_bnds.values[0]
            if (lower.year, lower.month) != (year, month):
                ds.close(); raise ValueError(f"WACCM-X time bounds disagree with filename: {path.name}")
            timestamp = lower + (upper - lower) / 2
        else:
            timestamp = cftime.datetime(year, month, 15, calendar=ds.time.values[0].calendar)
        time_attrs = dict(ds.time.attrs)
        ds = ds.assign_coords(time=("time", [timestamp]))
        ds.time.attrs = time_attrs
        ds.attrs["original_time_coordinate"] = original_time
        ds.attrs["monthly_time_label"] = "interval midpoint; source filename and lower time bound identify calendar month"
    ds.attrs = {**ds.attrs, "model": model, "experiment": config["model"]["experiment"],
                "source_variable": name, "source_file": path.name}
    return ds


def source_files(model, variable):
    from .config import ROOT
    config = load_config(model)
    name = config["variables"].get(variable, variable)
    base = ROOT / config["paths"]["raw"]
    if model == "WACCM-X":
        files = sorted(base.glob("*_zm.nc"))
        if files:
            return files
    files = sorted(base.glob(f"*/{name}/{name}_*_{config['model']['archive_model']}_refD1_*.nc"))
    preferred = config.get("preferred_families", {}).get(name)
    if preferred:
        files = [path for path in files if path.parent.parent.name == preferred]
    elif any(path.parent.parent.name == "Amon" for path in files):
        files = [path for path in files if path.parent.parent.name == "Amon"]
    if not files:
        raise FileNotFoundError(f"No downloaded monthly files for {model} {name}")
    return files
