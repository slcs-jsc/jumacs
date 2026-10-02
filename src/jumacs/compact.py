"""Compact altitude-grid application products from separate native climatologies."""

from pathlib import Path

import numpy as np
import xarray as xr

from .cf import PRESSURE_FIELD
from .climatology import product_name
from .config import ROOT, load_config, has_capability, reference_period


def interpolate_profile(height_km, values, target_km):
    """Interpolate valid native levels without extending their vertical domain."""
    good = np.isfinite(height_km) & np.isfinite(values)
    if good.sum() < 2:
        return np.full(target_km.shape, np.nan)
    height, values = height_km[good], values[good]
    order = np.argsort(height)
    height, values = height[order], values[order]
    height, unique = np.unique(height, return_index=True)
    values = values[unique]
    return np.interp(target_km, height, values, left=np.nan, right=np.nan)


def blend_profiles(lower, upper, heights_km, start_km, end_km):
    """Raised-cosine transition; preserve missing cells rather than filling gaps."""
    if not 0 <= start_km < end_km:
        raise ValueError("Transition heights must increase")
    phase = np.clip((heights_km - start_km) / (end_km - start_km), 0, 1)
    weight = 0.5 - 0.5 * np.cos(np.pi * phase)
    shape = (1, len(heights_km), 1)
    weight = weight.reshape(shape)
    both = np.isfinite(lower) & np.isfinite(upper)
    result = np.where(both, (1-weight)*lower + weight*upper, np.nan)
    result = np.where((weight == 0) & np.isfinite(lower), lower, result)
    result = np.where((weight == 1) & np.isfinite(upper), upper, result)
    return result


def _unit_key(unit):
    return {"molmol-1": "mole_fraction", "mol/mol": "mole_fraction"}.get(
        (unit or "").replace(" ", ""), (unit or "").replace(" ", ""))


def _read(model, canonical, start_year, end_year):
    """Extract one variable's climatological mean from the model's single product."""
    config = load_config(model)
    name = config["variables"].get(canonical)
    if not name:
        return None
    path = ROOT / config["paths"]["climatology"] / product_name(model, start_year, end_year)
    if not path.exists():
        return None
    field = f"{name}_mean"
    with xr.open_dataset(path) as ds:
        if field not in ds or ds[field].ndim != 3 or "lat" not in ds[field].dims:
            return None
        keep, rename = [field], {field: "mean"}
        published = str(ds[field].attrs.get("pressure_field", "")).strip()
        if published in ds and published not in keep:
            keep.append(published)
            rename[published] = PRESSURE_FIELD
        selected = ds[keep].rename(rename).load()
    selected["mean"].attrs.setdefault("units", "")
    selected.attrs["output_units"] = selected["mean"].attrs["units"]
    return selected


def _on_grid(ds, height_ds, latitude, height_km):
    """Map each monthly zonal profile via model height, including pressure levels."""
    field = ds["mean"].interp(lat=latitude).values
    z = height_ds["mean"].interp(lat=latitude).values / 1000.0
    if field.shape[1] != z.shape[1] or ds["mean"].dims[1] != height_ds["mean"].dims[1]:
        # Pressure levels: determine altitude from the model's own
        # monthly geopotential-height/pressure profile in log pressure.
        height_level = height_ds["mean"].dims[1]
        source = height_ds["air_pressure"] if "air_pressure" in height_ds else height_ds[height_level]
        pressure = source.interp(lat=latitude).transpose(*height_ds["mean"].dims).values
        level = ds["mean"].dims[1]
        level_coord = ds[level] if level in ds else ds[height_level]
        target_pa = level_coord.values
        if level_coord.attrs.get("units") == "hPa":
            target_pa = target_pa * 100.0

        mapped = np.full_like(field, np.nan, dtype=float)
        for m in range(12):
            for j in range(len(latitude)):
                mapped[m, :, j] = interpolate_profile(
                    -np.log(pressure[m, :, j]), z[m, :, j], -np.log(target_pa))
        z = mapped
    output = np.full((12, len(height_km), len(latitude)), np.nan, dtype=float)
    for m in range(12):
        for j in range(len(latitude)):
            output[m, :, j] = interpolate_profile(z[m, :, j], field[m, :, j], height_km)
    return output


def build_compact(lower_model, start_year=None, end_year=None,
                  latitude_step=5, altitude_step=1, transition_start=55,
                  transition_end=65, variables=None):
    """Create one lower-model→WACCM-X product; never average CCMI models."""
    if not has_capability(lower_model, "compact_waccmx"):
        raise ValueError(f"{lower_model} is not registered for compact WACCM-X extension; "
                         "set capabilities.compact_waccmx in config/models after validating the pairing")
    reference = reference_period()["reference_period"]
    start_year = start_year or reference["start_year"]
    end_year = end_year or reference["end_year"]
    if 170 % latitude_step or 120 % altitude_step:
        raise ValueError("Grid steps must evenly divide 170° latitude and 120 km")
    latitude = np.arange(-85, 86, latitude_step, dtype=float)
    height_km = np.arange(0, 121, altitude_step, dtype=float)
    lower_config = load_config(lower_model)
    upper_config = load_config("WACCM-X")
    lower_height = _read(lower_model, "geopotential_height", start_year, end_year)
    upper_height = _read("WACCM-X", "geopotential_height", start_year, end_year)
    if lower_height is None or upper_height is None:
        raise FileNotFoundError("Both native geopotential-height climatologies are required")
    names = variables or list(dict.fromkeys((*lower_config["variables"], *upper_config["variables"])))
    data = {}
    for canonical in names:
        if canonical in ("geopotential_height", "surface_pressure"):
            continue
        lower = _read(lower_model, canonical, start_year, end_year)
        upper = _read("WACCM-X", canonical, start_year, end_year)
        if lower is None and upper is None:
            continue
        lower_units = lower.attrs.get("output_units") if lower is not None else None
        upper_units = upper.attrs.get("output_units") if upper is not None else None
        if lower is not None and upper is not None:
            if _unit_key(lower_units) != _unit_key(upper_units):
                raise ValueError(f"Incompatible units for {canonical}: {lower_units}, {upper_units}")
        low = _on_grid(lower, lower_height, latitude, height_km) if lower is not None else None
        high = _on_grid(upper, upper_height, latitude, height_km) if upper is not None else None
        if low is not None and high is not None:
            # Require every latitude/month profile to cover the transition.
            transition = (height_km >= transition_start) & (height_km <= transition_end)
            if not (np.isfinite(low[:, transition]).all() and np.isfinite(high[:, transition]).all()):
                raise ValueError(f"Incomplete overlap for {canonical} at {transition_start}–{transition_end} km")
            merged = blend_profiles(low, high, height_km, transition_start, transition_end)
            status = "lower_model_to_waccmx"
        elif low is not None:
            merged, status = low, "lower_model_only"
        else:
            merged, status = high, "waccmx_only"
        if not np.isfinite(merged).any():
            continue
        key = canonical.lower().replace("-", "_")
        data[key] = (("month", "altitude", "lat"), merged.astype("float32"),
                     {"units": lower_units or upper_units, "canonical_species": canonical,
                      "source_status": status,
                      "lower_source_variable": lower_config["variables"].get(canonical, "missing") if lower is not None else "missing",
                      "upper_source_variable": upper_config["variables"].get(canonical, "missing") if upper is not None else "missing"})
    ds = xr.Dataset(data, coords={"month": np.arange(1, 13, dtype="int8"),
                                  "altitude": height_km.astype("float32"),
                                  "lat": latitude.astype("float32")})
    ds.altitude.attrs = {"units": "km", "long_name": "geopotential height above mean sea level"}
    ds.lat.attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds.attrs = {"project": "JuMACS", "product_type": "compact_vertical_extension",
                "lower_model": lower_model, "lower_experiment": "refD1", "upper_model": "WACCM-X",
                "upper_experiment": "transient-1950-2015",
                "lower_source_dataset": lower_config["model"]["dataset_uuid"],
                "upper_source_dataset": upper_config["model"]["dataset_uuid"],
                "climatology_period": f"{start_year}-{end_year}",
                "reference_period_years": end_year-start_year+1,
                "nominal_reference_year": (reference["nominal_reference_year"]
                                           if (start_year, end_year) == (reference["start_year"], reference["end_year"])
                                           else int(round((start_year + end_year) / 2))),
                "latitude_step_degrees": latitude_step, "altitude_step_km": altitude_step,
                "transition_start_km": transition_start, "transition_end_km": transition_end,
                "transition_method": "raised cosine on geopotential height where both fields are finite",
                "missing_policy": "NaN outside each source's valid vertical range; no extrapolation",
                "bias_correction": "none", "trend_correction": "none",
                "history": "native monthly zonal climatologies interpolated to latitude and geopotential height; source-specific vertical blend"}
    dest = ROOT / "products/climatology/combined" / f"jumacs_{lower_model.lower()}_waccmx_{start_year}-{end_year}_{latitude_step}deg_{altitude_step}km.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".nc.tmp")
    encoding = {key: {"dtype": "float32", "zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)} for key in ds.data_vars}
    ds.to_netcdf(temp, engine="netcdf4", encoding=encoding)
    temp.replace(dest)
    return dest
