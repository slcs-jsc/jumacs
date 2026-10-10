"""The small persistent wrapper around the existing remap and extension stages."""

import numpy as np
import xarray as xr

from jumacs import cf, climatology, config, remap
from jumacs.application import build_application_product, lower_boundary_invalid_mask
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


def test_lower_boundary_mask_geometry_on_the_application_grid():
    grid = np.asarray(config.vertical_grid()["levels"], dtype="float64")
    _, centers = remap.latitude_bands()
    assert np.array_equal(centers, np.arange(-87.5, 90.0, 5.0))
    invalid = lower_boundary_invalid_mask(grid, centers)

    def row(pressure):
        return invalid[int(np.argmin(np.abs(grid - pressure)))]

    assert invalid[0].all()
    for band in (0, 1, 2):
        assert invalid[:4, band].all(), (band, grid[:4])
    for pressure in (86577.84, 74957.22):
        masked = row(pressure)
        assert masked[:5].all() and not masked[5]
        assert masked[24] and masked[25] and masked[26] and not masked[23] and not masked[27]
        assert masked.sum() == 8
    masked = row(64896.35)
    assert masked[:4].all() and not masked[4]
    assert not masked[24:27].any()
    assert row(56185.85).sum() == 0
    assert (grid >= 95000.0).nonzero()[0].size == 1 and invalid[grid >= 95000.0].all()


def test_builder_masks_access_lower_boundary_and_leaves_other_models_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(climatology, "ROOT", tmp_path)
    native_product("ACCESS-CM2-Chem", BASE_PRESSURE, [("ta", 220.0, "K"), ("cf2cl2", 5.0, "ppt")], tmp_path)
    native_product("WACCM-X", DONOR_PRESSURE, [("T", 260.0, "K")], tmp_path)
    output = build_application_product("ACCESS-CM2-Chem", 1985, 2014, root=tmp_path)

    def at(written, name, pressure, lat):
        p = int(np.argmin(np.abs(written["pressure"].values - pressure)))
        return written[name].isel(pressure=p, lat=int((lat + 87.5) / 5)).values

    with xr.open_dataset(output, decode_cf=False) as written:
        assert np.isnan(written["cf2cl2_mean"].isel(pressure=0).values).all()
        for statistic in cf.APPLICATION_STATISTIC_ORDER:
            for pressure, lat in ((86577.84, -67.5), (86577.84, 37.5), (74957.22, 32.5), (74957.22, -67.5),
                                  (64896.35, -72.5)):
                assert np.isnan(at(written, f"cf2cl2_{statistic}", pressure, lat)).all(), (statistic, pressure, lat)
            for pressure, lat in ((86577.84, -62.5), (86577.84, 27.5), (86577.84, 47.5), (74957.22, -62.5),
                                  (74957.22, 47.5), (64896.35, 37.5), (64896.35, -67.5), (56185.85, -72.5)):
                assert np.isfinite(at(written, f"cf2cl2_{statistic}", pressure, lat)).all(), (statistic, pressure, lat)
        assert np.isnan(at(written, "ta_mean", 100000.0, 0.0)).all()
        assert np.isfinite(at(written, "ta_mean", 0.002, 0.0)).all()
        assert written.attrs["lower_boundary_quality_mask_scope"] == "application product only"
        assert "lower_boundary_quality_mask_rule" in written.attrs
        assert "masked the diagnosed lower-boundary cells" in written.attrs["history"]

    native_product("SOCOL", BASE_PRESSURE, [("ta", 220.0, "K")], tmp_path)
    socol = build_application_product("SOCOL", 1985, 2014, root=tmp_path)
    with xr.open_dataset(socol, decode_cf=False) as written:
        assert "lower_boundary_quality_mask" not in written.attrs
        assert np.isfinite(written["ta_mean"].isel(pressure=0).values).all()


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
