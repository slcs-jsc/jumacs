import math

import numpy as np
import xarray as xr

from jumacs import ig2
from jumacs.application_ready import (
    _blend,
    _edge_and_transition,
    build_application_ready,
)

ALT = np.arange(121).astype(float)


def test_latitude_weights_are_continuous_and_clamped():
    for magnitude in (0.0, 22.5, 45.0, 60.0, 75.0, 85.0, 90.0):
        assert math.isclose(sum(ig2.latitude_weights(magnitude)), 1.0)
    assert ig2.latitude_weights(0.0) == (1.0, 0.0, 0.0)
    assert ig2.latitude_weights(75.0) == ig2.latitude_weights(90.0) == (0.0, 0.0, 1.0)
    assert ig2.latitude_weights(-45.0) == (0.0, 1.0, 0.0)
    below = ig2.latitude_weights(44.99)
    above = ig2.latitude_weights(45.01)
    assert all(abs(a - b) < 1e-3 for a, b in zip(below, above))


def test_summer_weight_anchors_january_as_winter():
    assert ig2.summer_weight(1) == 0.0
    assert math.isclose(ig2.summer_weight(7), 1.0)
    assert math.isclose(ig2.summer_weight(4), 0.5)


def test_donor_column_uses_regime_and_seasonal_mapping():
    nalt = ALT.size
    profiles = {
        "tropical": np.full(nalt, 10.0),
        "midlatitude_day": np.full(nalt, 12.0),
        "midlatitude_night": np.full(nalt, 8.0),
        "polar_summer": np.full(nalt, 15.0),
        "polar_winter": np.full(nalt, 5.0),
    }
    regime_data = {"altitude_km": ALT, "nalt": nalt,
                   "regimes": {name: {"X": arr} for name, arr in profiles.items()}}
    column = ig2.donor_column(regime_data, "X", [1, 7], [0.0, 45.0, 75.0])
    assert column.shape == (2, nalt, 3)
    np.testing.assert_allclose(column[:, :, 0], 10.0)
    np.testing.assert_allclose(column[:, :, 1], 10.0)
    np.testing.assert_allclose(column[0, :, 2], 5.0)
    np.testing.assert_allclose(column[1, :, 2], 15.0)


def test_blend_log_and_linear():
    assert _blend(2.0, 4.0, 0.5, positive=False) == 3.0
    assert math.isclose(_blend(1.0, 100.0, 0.5, positive=True), 10.0)
    assert _blend(-1.0, 4.0, 0.5, positive=True) == 1.5


def test_edge_and_transition_preserves_existing_and_caps_donor():
    field = np.full((1, 121, 1), np.nan)
    field[0, 2:61, 0] = np.linspace(1e-6, 5e-6, 59)
    field[0, 30, 0] = np.nan
    donor = (1e-5 + ALT * 1e-7)[None, :, None]
    donor_column = np.tile(donor, (1, 1, 1))
    spec = {"positive": True, "donor_top_km": 99.0}
    original = field.copy()
    completed, counts = _edge_and_transition(field, ALT, donor_column, spec, width=8.0)
    assert completed[0, 0, 0] == completed[0, 1, 0] == original[0, 2, 0]
    assert counts["constant_lower_cells"] == 2
    assert math.isnan(completed[0, 30, 0]) and counts["internal_nan_cells"] == 1
    preserved = [k for k in range(2, 61) if k != 30]
    np.testing.assert_array_equal(completed[0, preserved, 0], original[0, preserved, 0])
    assert np.isfinite(completed[0, 61:100, 0]).all()
    assert abs(math.log10(completed[0, 61, 0] / completed[0, 60, 0])) < 0.05
    assert completed[0, 120, 0] == completed[0, 99, 0]
    assert completed[0, 120, 0] != donor[0, 120, 0]
    assert counts["ig2_upper_cells"] > 0 and counts["transition_blend_cells"] > 0
    assert counts["constant_upper_cells"] == 21


def _write_atm(path, species):
    lines = [str(ALT.size), "*HGT [km] " + " ".join(f"{v:.6e}" for v in ALT)]
    for name, (unit, values) in species.items():
        lines.append(f"*{name} [{unit}] " + " ".join(f"{v:.6e}" for v in values))
    lines.append("*END")
    path.write_text("\n".join(lines) + "\n")


def _write_ig2(dirpath):
    dirpath.mkdir(parents=True, exist_ok=True)
    offsets = {"tropical": 10, "midlatitude_day": 12, "midlatitude_night": 8,
               "polar_summer": 15, "polar_winter": 5}
    files = {"tropical": "mipas_eqn_d2.atm", "midlatitude_day": "mipas_midl_day_d2.atm",
             "midlatitude_night": "mipas_midl_ngt_d2.atm", "polar_summer": "mipas_psum_d2.atm",
             "polar_winter": "mipas_pwin_d2.atm"}
    for regime, filename in files.items():
        offset = offsets[regime]
        species = {spec["ig2_species"]: ("ppmv", (offset + ALT * 0.1)) for spec in ig2.IG2_DONORS.values()}
        _write_atm(dirpath / filename, species)


def _write_combined(path, lats):
    months = np.arange(1, 13)
    data_vars = {}
    tops = {"o3": 70}
    for key in ["o3", "h2o", "n2o", "co", "h2o2", "hcfc_22", "clo"]:
        top = tops.get(key, 60)
        profile = np.where(ALT <= top, 1e-6 + 1e-8 * ALT, np.nan)
        profile[0] = np.nan
        field = np.broadcast_to(profile[None, :, None], (12, ALT.size, lats.size)).astype("float32")
        data_vars[key] = (("month", "altitude", "lat"), field, {"units": "mol/mol"})
    ds = xr.Dataset(data_vars, coords={"month": months, "altitude": ALT, "lat": lats})
    ds.to_netcdf(path)


def test_build_application_ready(tmp_path):
    ig2_dir = tmp_path / "ig2"
    _write_ig2(ig2_dir)
    lats = np.arange(-85, 86, 5).astype(float)
    combined = tmp_path / "combined.nc"
    _write_combined(combined, lats)
    output = tmp_path / "out" / "ready.nc"
    report = build_application_ready("EMAC", 1985, 2014, combined_path=combined,
                                     ig2_dir=ig2_dir, output_path=output)
    assert output.exists()
    with xr.open_dataset(output) as ds:
        j = int(np.argmin(np.abs(ds.lat.values)))
        assert not np.isnan(ds["h2o"].values).any()
        assert ds["h2o"].attrs["ig2_completed"] == "true"
        assert ds["o3"].attrs["ig2_completed"] == "false"
        np.testing.assert_array_equal(ds["h2o"].values[:, 0, j], ds["h2o"].values[:, 1, j])
        original = np.where((ALT <= 60) & (ALT > 0), 1e-6 + 1e-8 * ALT, np.nan)
        finite = np.isfinite(original)
        np.testing.assert_allclose(ds["h2o"].values[0, finite, j], original[finite], rtol=1e-5)
        assert ds["clo"].values[0, 120, j] == ds["clo"].values[0, 99, j]
    seam = report["smoothness"]["h2o"]["vertical_transition"]["equator"]
    assert seam["seam_log_step_after_blend"] < 0.3
    assert seam["seam_log_step_after_blend"] < 0.5 * seam["hard_graft_seam_log_step"]
    assert ds.attrs["internal_gaps_filled"] == "false"
    assert ds.attrs["ig2_version"] == ig2.VERSION
    assert report["totals"]["ig2_upper_cells"] > 0
    assert report["variables_flagged_for_hard_seam"] == []
