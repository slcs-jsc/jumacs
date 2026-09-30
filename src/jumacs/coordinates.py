"""CF hybrid pressure on each model's own native levels."""
import re
import xarray as xr


def formula_terms(level):
    text = level.attrs.get("formula_terms", level.attrs.get("z_factors", ""))
    return dict(re.findall(r"(ap|a|b|ps|p0)\s*:\s*([A-Za-z][A-Za-z0-9_]*)", text))


def hybrid_pressure(ds, model_config, surface_pressure=None):
    names = model_config["coordinates"]
    level = ds[names["level"]]
    terms = formula_terms(level)
    a_name = terms.get("ap") or terms.get("a") or names["hybrid_a"]
    b_name = terms.get("b") or names["hybrid_b"]
    ps_name = terms.get("ps") or names["surface_pressure"]
    if a_name not in ds or b_name not in ds:
        raise ValueError("Hybrid coefficients absent; inspect NetCDF formula_terms")
    ps = surface_pressure if surface_pressure is not None else ds.get(ps_name)
    if ps is None:
        raise ValueError("Surface pressure required for hybrid pressure")
    if a_name == "ap" or "ap" in terms or ds[a_name].attrs.get("units", "").lower() in ("pa", "pascal", "pascals"):
        pressure = ds[a_name] + ds[b_name] * ps
        formula = f"{a_name} + {b_name} * {ps_name}"
    else:
        p0_name = terms.get("p0", names.get("reference_pressure", "p0"))
        if p0_name not in ds:
            raise ValueError("Dimensionless hybrid a requires p0")
        p0 = ds[p0_name].squeeze(drop=True)
        pressure = ds[a_name] * p0 + ds[b_name] * ps
        formula = f"{a_name} * {p0_name} + {b_name} * {ps_name}"
    pressure.name = "air_pressure"
    pressure.attrs = {"units": "Pa", "standard_name": "air_pressure", "formula": formula}
    return pressure
