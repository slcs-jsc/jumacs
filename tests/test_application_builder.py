"""The small persistent wrapper around the existing remap and extension stages."""

import numpy as np
import xarray as xr

from jumacs import cf, config, remap
from jumacs.application import build_application_product
from jumacs.climatology import variable_statistics, write_product
from jumacs.cli import main


LAT = np.array([-60.0, 0.0, 60.0])
BASE_PRESSURE = np.geomspace(100000.0, 10.0, 9)
DONOR_PRESSURE = np.geomspace(100000.0, 0.002, 17)


def native_part(name, pressure, value, units):
    data = xr.Dataset(
        {statistic: (("month", "plev", "lat"),
                     np.full((12, pressure.size, LAT.size), value, dtype="float64"))
         for statistic in cf.STATISTIC_ORDER},
        coords={"month": np.arange(1, 13), "plev": ("plev", pressure), "lat": ("lat", LAT)})
    return variable_statistics(name, data, {"units": units}, {})


def native_product(model, pressure, specifications, tmp_path):
    parts = [native_part(name, pressure, value, units) for name, value, units in specifications]
    pressures = {"plev": {"kind": "coordinate", "pressure": xr.DataArray(pressure, dims="plev"),
                          "source_variable": specifications[0][0], "coverage": (1985, 2014),
                          "units": "Pa", "source_level": "plev"}}
    return write_product(model, 1985, 2014, parts, pressures=pressures)


def test_builder_pairs_canonical_names_and_writes_valid_application_product(tmp_path, monkeypatch):
    import jumacs.climatology as climatology
    monkeypatch.setattr(climatology, "ROOT", tmp_path)
    base_path = native_product("SOCOL", BASE_PRESSURE,
                               [("ta", 220.0, "K"), ("o3", 2.0, "mol mol-1")], tmp_path)
    donor_path = native_product("WACCM-X", DONOR_PRESSURE,
                                [("T", 260.0, "K"), ("CH4", 1.0, "mol mol-1")], tmp_path)
    with xr.open_dataset(base_path, decode_cf=False) as source:
        unextended = remap.remap_product(source)

    output = build_application_product("SOCOL", 1985, 2014, root=tmp_path)
    assert output.name == "jumacs_socol_application_climatology_1985-2014.nc"
    assert output.exists() and not output.with_suffix(".nc.tmp").exists()
    with xr.open_dataset(output, decode_cf=False) as written:
        cf.assert_product(written, names=("ta", "o3"), grid=config.vertical_grid(),
                          kind="application", latitude_bands=36)
        assert dict(written.sizes) == {"time": 12, "pressure": 124, "lat": 36, "nv": 2}
        assert not ({"lev", "lev_2", "plev", "ilev"} & set(written.dims))
        assert not any(name.endswith("_n_years") for name in written.data_vars)
        assert not any(name.startswith("CH4_") for name in written.data_vars)
        assert set(written.data_vars) == {"climatology_bounds"} | {
            f"{name}_{statistic}" for name in ("ta", "o3")
            for statistic in cf.APPLICATION_STATISTIC_ORDER}
        assert np.array_equal(written["o3_mean"].values, unextended["o3_mean"].values, equal_nan=True)
        assert np.isfinite(written["ta_mean"].isel(pressure=-1)).all()
        assert written["ta_mean"].attrs["canonical_variable"] == "temperature"
        assert written["ta_mean"].attrs["base_native_variable"] == "ta"
        assert written["ta_mean"].attrs["donor_native_variable"] == "T"
        assert written["ta_mean"].attrs["extension_donor"] == "WACCM-X"
        assert written.attrs["base_input_product"] == str(base_path)
        assert written.attrs["donor_input_product"] == str(donor_path)
        assert written.attrs["climatology_period"] == "1985-2014"


def test_application_cli_passes_model_period_and_extension_choice(monkeypatch, capsys):
    import jumacs.application as application
    calls = []

    def build(model, start_year, end_year, *, extend):
        calls.append((model, start_year, end_year, extend))
        return "/tmp/application.nc"

    monkeypatch.setattr(application, "build_application_product", build)
    main(["application", "--model", "SOCOL", "--start-year", "1985", "--end-year", "2014"])
    main(["application", "--model", "SOCOL", "--no-extension"])
    assert calls == [("SOCOL", 1985, 2014, True), ("SOCOL", None, None, False)]
    assert capsys.readouterr().out.splitlines() == ["/tmp/application.nc", "/tmp/application.nc"]
