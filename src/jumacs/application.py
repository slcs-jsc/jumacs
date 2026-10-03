"""Write one model's application-grid climatology, optionally extended by WACCM-X."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from . import cf, config, extension, remap
from .climatology import product_name


def build_application_product(model, start_year=None, end_year=None, *, extend=True, root=None):
    """Remap one published CCMI product and write a validated application product."""
    if model == "WACCM-X" or model not in config.ccmi_model_names():
        raise ValueError(f"{model} must be a registered CCMI model")
    period = config.model_period(model)
    start_year = period["start_year"] if start_year is None else start_year
    end_year = period["end_year"] if end_year is None else end_year
    if start_year > end_year:
        raise ValueError("start year exceeds end year")
    root = config.ROOT if root is None else Path(root)
    base_config = config.load_config(model)
    donor_config = config.load_config("WACCM-X")
    base_path = root / base_config["paths"]["climatology"] / product_name(model, start_year, end_year)
    donor_path = root / donor_config["paths"]["climatology"] / product_name("WACCM-X", start_year, end_year)
    grid = config.vertical_grid()

    with xr.open_dataset(base_path, decode_cf=False) as source:
        _, base_names = remap.statistic_fields(source)
        cf.assert_product(source, names=base_names, grid=grid, kind="individual")
        product = remap.remap_product(source)

    paired = {}
    if extend:
        with xr.open_dataset(donor_path, decode_cf=False) as source:
            _, donor_names = remap.statistic_fields(source)
            cf.assert_product(source, names=donor_names, grid=grid, kind="individual")
            base_mapping = base_config["variables"]
            donor_mapping = donor_config["variables"]
            paired = {base_mapping[canonical]: (canonical, donor_mapping[canonical])
                      for canonical in base_mapping
                      if base_mapping[canonical] in base_names and canonical in donor_mapping
                      and donor_mapping[canonical] in donor_names}
            if paired:
                donor = remap.remap_product(source, names=[item[1] for item in paired.values()])
                rename = {f"{donor_native}_{statistic}": f"{base_native}_{statistic}"
                          for base_native, (_, donor_native) in paired.items()
                          for statistic in cf.APPLICATION_STATISTIC_ORDER}
                donor = donor.rename(rename)
                product, _ = extension.extend_product(product, donor)

    for base_native, (canonical, donor_native) in paired.items():
        for statistic in cf.APPLICATION_STATISTIC_ORDER:
            field = product[f"{base_native}_{statistic}"]
            if field.attrs.get("extension_applied") == "true":
                field.attrs.update(canonical_variable=canonical, base_native_variable=base_native,
                                   donor_native_variable=donor_native)

    product.attrs.update({
        "base_model": model,
        "base_input_product": str(base_path),
        "donor_model": "WACCM-X" if extend else "none",
        "donor_input_product": str(donor_path) if extend else "",
        "application_grid_remapping": remap.REMAPPING_ORDER,
        "waccmx_extension_method": extension.EXTENSION_METHOD if paired else "none",
        "history": (str(product.attrs.get("history", "")) + "\n" +
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}: remapped {model} "
                    f"onto the JuMACS application grid" +
                    (" and extended upward with WACCM-X" if paired else "")).strip(),
    })
    cf.assert_product(product, names=base_names, grid=grid, kind="application", latitude_bands=36)
    destination = root / "products" / "application" / model / (
        f"jumacs_{config.model_slug(model)}_application_climatology_{start_year}-{end_year}.nc")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".nc.tmp")
    encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 4,
                       "_FillValue": np.float32(np.nan)}
                for name in product.data_vars if name != "climatology_bounds"}
    encoding["climatology_bounds"] = {"dtype": "float64", "_FillValue": None, "zlib": True}
    encoding.update({name: {"dtype": "float64"} for name in ("time", "pressure", "lat")})
    # Native files were opened without CF decoding for validation, so their statistic
    # attributes can still contain _FillValue. The output encoding owns that key.
    for name in product.data_vars:
        product[name].attrs.pop("_FillValue", None)
    try:
        product.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        with xr.open_dataset(temporary, decode_cf=False) as written:
            cf.assert_product(written, names=base_names, grid=grid, kind="application", latitude_bands=36)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    return destination
