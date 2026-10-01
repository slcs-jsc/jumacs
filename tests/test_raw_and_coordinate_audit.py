import json
import numpy as np

import xarray as xr

from jumacs.config import ROOT, load_config, registry
from jumacs.download import manifest_path
from jumacs.raw_validation import select_representatives, validate_model, write_raw_validation
from jumacs.coordinate_audit import audit_model, write_coordinate_audit


def _times(n=4, calendar="standard", start="1990-01"):
    return xr.cftime_range(start, periods=n, freq="MS", calendar=calendar)


def _store(root, model, family, variable, filename, ds, start=None, end=None):
    raw = root / load_config(model)["paths"]["raw"]
    dest = raw / family / variable / filename if family else raw / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dest)
    times = ds.time.values
    if start is None:
        start = f"{times[0].year}{times[0].month:02d}"
    if end is None:
        end = f"{times[-1].year}{times[-1].month:02d}"
    return {"family": family or "data", "variable": variable, "filename": filename, "grid": "gn",
            "version": "v1", "start": start,
            "end": end, "size_bytes": int(dest.stat().st_size),
            "checksum": None, "model": model, "local_path": str(dest.relative_to(root))}


def _manifest(root, model, records):
    path = root / manifest_path(model).relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "mode": "full_archive", "files": records}))
    return path


def _hybrid_ds(lon_values=None, coord_names=("lat", "lon"), calendar="standard", terms="ap: ap b: b ps: ps",
               include_ap=True, include_ps=True, lat2d=False, time_strings=False, n_times=4):
    times = _times(n_times, calendar)
    lon = np.array(lon_values if lon_values is not None else np.arange(0.0, 360.0, 45.0))
    lat = np.array([-60.0, -30.0, 30.0, 60.0])
    lev = [1.0, 2.0]
    lat_name, lon_name = coord_names
    values = np.linspace(200.0, 260.0, n_times * 2 * 4 * len(lon)).reshape(n_times, 2, 4, len(lon))
    if lat2d:
        dims = ("time", "lev", "y", "x")
        spatial = ("y", "x")
        coords = {"time": times, "lev": ("lev", lev),
                  lat_name: (("y", "x"), np.tile(lat[:, None], (1, len(lon)))),
                  lon_name: (("y", "x"), np.tile(lon[None, :], (4, 1)))}
    else:
        dims = ("time", "lev", lat_name, lon_name)
        spatial = (lat_name, lon_name)
        coords = {"time": times, "lev": ("lev", lev), lat_name: (lat_name, lat), lon_name: lon}
    data = np.zeros(values.shape, dtype=np.float32)
    data[:] = values
    ds = xr.Dataset({"ta": (dims, data, {"units": "K", "standard_name": "air_temperature",
                                         "long_name": "temperature", "_FillValue": np.float32(-999.0)})},
                    coords=coords)
    ds.lev.attrs = {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate", "positive": "up",
                    "units": "1"}
    if terms:
        ds.lev.attrs["formula_terms"] = terms
    if include_ap:
        ds["ap"] = ("lev", [10000.0, 20000.0], {"units": "Pa"})
        ds["b"] = ("lev", [0.9, 0.5])
    if include_ps:
        ds["ps"] = (("time",) + spatial, np.full((n_times, 4, len(lon)), 100000.0, dtype=np.float32),
                    {"units": "Pa", "standard_name": "surface_air_pressure"})
    if calendar == "standard" and not time_strings:
        ds.time.attrs = {"bounds": "time_bnds"}
    if time_strings:
        ds = ds.assign_coords(time=("time", [str(t) for t in times]))
    return ds


def _plev_amonz_ds(model="EMAC", variable="o3", native="o3", n_times=4):
    times = _times(n_times)
    plev = [100000.0, 10000.0, 1000.0]
    lat = np.array([-60.0, -30.0, 30.0, 60.0])
    ds = xr.Dataset({native: (("time", "plev", "lat"),
                              np.linspace(1e-6, 9e-6, n_times * 3 * 4).reshape(n_times, 3, 4).astype(np.float32),
                              {"units": "mol mol-1", "standard_name": "mole_fraction_of_ozone_in_air"})},
                    coords={"time": times, "plev": ("plev", plev), "lat": ("lat", lat)})
    ds.plev.attrs = {"standard_name": "air_pressure", "units": "Pa", "positive": "down"}
    ds["ps"] = (("time", "lat"), np.full((n_times, 4), 100000.0, dtype=np.float32),
                {"units": "Pa", "standard_name": "surface_air_pressure"})
    return ds


def _waccmx_ds():
    times = _times(1)
    ds = xr.Dataset({"T": (("time", "lev", "lat"), np.linspace(180.0, 300.0, 2 * 3).reshape(1, 2, 3),
                           {"units": "K", "standard_name": "air_temperature"}),
                     "PS": (("time", "lat"), np.full((1, 3), 100000.0), {"units": "Pa",
                                                                         "standard_name": "surface_air_pressure"}),
                     "hyam": ("lev", [1.0, 0.1]), "hybm": ("lev", [0.0, 0.9]),
                     "P0": ((), 100000.0, {"units": "Pa"})},
                    coords={"time": times, "lev": ("lev", [1.0, 2.0]), "lat": ("lat", [-45.0, 0.0, 45.0])})
    ds.lev.attrs = {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate", "positive": "up",
                    "units": "1", "formula_terms": "a: hyam b: hybm ps: PS p0: P0"}
    return ds


def test_validate_complete_archive(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", _hybrid_ds())
    _manifest(tmp_path, "EMAC", [record])
    row = validate_model("EMAC", tmp_path)
    assert row["status"] == "ok"
    assert row["expected_files"] == row["present_files"] == 1
    assert row["missing_files"] == row["partial_files"] == row["invalid_signature_files"] == 0
    assert row["representative_files_checked"] == 1 and row["time_decode_ok"] is True
    assert row["reasons"] == []
    assert row["present_bytes"] == row["listed_bytes"]


def test_validate_missing_file(tmp_path):
    records = [_store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", _hybrid_ds())]
    records.append({"family": "Amon", "variable": "o3", "filename": "absent.nc", "start": "199001", "end": "199012",
                    "size_bytes": 100, "local_path": "data/raw/EMAC/refD1/Amon/o3/absent.nc"})
    _manifest(tmp_path, "EMAC", records)
    row = validate_model("EMAC", tmp_path)
    assert row["status"] == "incomplete" and row["missing_files"] == 1
    assert any(reason.startswith("missing_expected_files:1") for reason in row["reasons"])


def test_validate_leftover_part_file(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", _hybrid_ds())
    _manifest(tmp_path, "EMAC", [record])
    stray = tmp_path / "data/raw/EMAC/refD1/Amon/o3/o3_partial.nc.part"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"partial")
    row = validate_model("EMAC", tmp_path)
    assert row["status"] == "incomplete" and row["partial_files"] == 1
    assert any(reason.startswith("leftover_part_files:1") for reason in row["reasons"])
    assert stray.exists()


def test_validate_invalid_signature(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", _hybrid_ds())
    bad = tmp_path / "data/raw/EMAC/refD1/Amon/h2o/h2o_bad.nc"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a netcdf file at all")
    record2 = dict(record, variable="h2o", filename=bad.name, local_path=str(bad.relative_to(tmp_path)),
                   size_bytes=bad.stat().st_size)
    _manifest(tmp_path, "EMAC", [record, record2])
    row = validate_model("EMAC", tmp_path)
    assert row["status"] == "invalid" and row["invalid_signature_files"] == 1
    assert any(reason.startswith("invalid_signature:1") for reason in row["reasons"])


def test_validate_xarray_open_failure(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", _hybrid_ds())
    bad = tmp_path / "data/raw/EMAC/refD1/Amon/o3/o3_truncated.nc"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    record2 = dict(record, variable="o3", filename=bad.name, local_path=str(bad.relative_to(tmp_path)),
                   size_bytes=bad.stat().st_size)
    _manifest(tmp_path, "EMAC", [record, record2])
    row = validate_model("EMAC", tmp_path)
    assert row["status"] == "invalid" and row["xarray_failures"] >= 1
    assert any(reason.startswith("xarray_open_failed") for reason in row["reasons"])


def test_validate_size_mismatch_and_not_downloaded(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", _hybrid_ds())
    mismatch = dict(record, size_bytes=record["size_bytes"] * 3)
    _manifest(tmp_path, "EMAC", [mismatch])
    assert validate_model("EMAC", tmp_path)["status"] == "incomplete"
    empty = dict(record, local_path="data/raw/EMAC/refD1/Amon/ta/gone.nc")
    _manifest(tmp_path, "EMAC", [empty])
    assert validate_model("EMAC", tmp_path)["status"] == "not_downloaded"


def test_validate_manifest_missing_and_not_applicable(tmp_path):
    assert validate_model("EMAC", tmp_path)["status"] == "not_downloaded"
    assert validate_model("EMAC", tmp_path)["reasons"] == ["manifest_missing"]
    row = validate_model("MIROC-ES2H", tmp_path)
    assert row["status"] == "not_applicable" and row["reasons"] == ["capability_missing:download"]


def test_representative_selection_rules(tmp_path):
    files = [
        {"family": "Amon", "variable": "o3", "filename": "late_o3.nc", "start": "201501", "end": "201812", "size_bytes": 10},
        {"family": "Amon", "variable": "ta", "filename": "big_ta.nc", "start": "196001", "end": "201812", "size_bytes": 900},
        {"family": "Amon", "variable": "ta", "filename": "small_ta.nc", "start": "196001", "end": "201812", "size_bytes": 100},
        {"family": "AmonZ", "variable": "o3", "filename": "amonz_o3.nc", "start": "196001", "end": "201812", "size_bytes": 50},
        {"family": "AmonZ", "variable": "oh", "filename": "amonz_oh.nc", "start": "199001", "end": "199012", "size_bytes": 20},
    ]
    selected, reasons = select_representatives("EMAC", files, tmp_path)
    names = [f["filename"] for f in selected]
    assert names[0] == "small_ta.nc"
    assert "amonz_o3.nc" in names
    assert "big_ta.nc" not in names
    assert "late_o3.nc" not in names[:2]
    assert any("canonical temperature" in reason and "1985-2014" in reason for reason in reasons)
    again, _ = select_representatives("EMAC", list(reversed(files)), tmp_path)
    assert [f["filename"] for f in again] == names


def test_representatives_overlap_reference_period_and_families(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", _hybrid_ds())
    amonz = _store(tmp_path, "EMAC", "AmonZ", "o3", "o3_AmonZ_EMAC_refD1_grz_r1i1p1f1_199001-199012.nc",
                   _plev_amonz_ds())
    _manifest(tmp_path, "EMAC", [record, amonz])
    row = validate_model("EMAC", tmp_path)
    assert row["status"] == "ok" and row["representative_files_checked"] == 2
    assert any("Amon," in note for note in row["notes"])


def test_coordinate_audit_pressure_level_and_already_zonal(tmp_path):
    record = _store(tmp_path, "EMAC", "AmonZ", "o3", "o3_AmonZ_EMAC_refD1_grz_r1i1p1f1_199001-199012.nc",
                    _plev_amonz_ds())
    _manifest(tmp_path, "EMAC", [record])
    row = audit_model("EMAC", tmp_path)
    entry = row["per_file"][0]
    assert entry["horizontal"]["grid"] == "already_zonal"
    assert entry["vertical"]["coordinate_type"] == "pressure_level"
    assert entry["vertical"]["n_levels"] == 3
    assert entry["vertical"]["top_pressure"] == 1000.0
    assert entry["vertical"]["bottom_pressure"] == 100000.0
    assert row["status"] == "supported" and row["reader_compatibility"] == "supported_now"


def test_coordinate_audit_hybrid_observed_terms(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", _hybrid_ds())
    _manifest(tmp_path, "EMAC", [record])
    row = audit_model("EMAC", tmp_path)
    vertical = row["per_file"][0]["vertical"]
    assert vertical["coordinate_type"] == "hybrid_sigma_pressure"
    assert vertical["formula_terms"] == {"ap": "ap", "b": "b", "ps": "ps"}
    assert vertical["formula_terms_source"] == "observed"
    assert vertical["surface_pressure_name"] == "ps"
    horizontal = row["per_file"][0]["horizontal"]
    assert horizontal["grid"] == "regular_latlon"
    assert horizontal["longitude_convention"] == "0..360"
    assert horizontal["n_lat"] == 4 and horizontal["n_lon"] == 8
    assert row["status"] == "supported" and row["reader_compatibility"] == "supported_now"
    assert "selected:" in row["reasons"][0]


def test_coordinate_audit_formula_missing_coefficient(tmp_path):
    ds = _hybrid_ds(terms="ap: ap_missing b: b ps: ps")
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", ds)
    _manifest(tmp_path, "EMAC", [record])
    row = audit_model("EMAC", tmp_path)
    vertical = row["per_file"][0]["vertical"]
    assert vertical["hybrid_a_present"] is False
    assert "hybrid_coefficients_missing_in_file" in row["reasons"]
    assert row["reader_compatibility"] == "reader_extension_needed"


def test_coordinate_audit_inferred_from_config_and_missing_ps(tmp_path):
    ds = _hybrid_ds(terms=None, include_ps=False)
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", ds)
    _manifest(tmp_path, "EMAC", [record])
    row = audit_model("EMAC", tmp_path)
    vertical = row["per_file"][0]["vertical"]
    assert vertical["formula_terms_source"] is None
    assert vertical["hybrid_a_name"] == "ap" and vertical["hybrid_a_present"] is True
    assert "hybrid_coefficients_inferred_from_config" in row["reasons"]
    assert "surface_pressure_mapping_missing" in row["reasons"]
    assert row["status"] == "needs_mapping" and row["reader_compatibility"] == "small_mapping_change"
    assert "YAML" in row["main_action"]


def test_coordinate_audit_longitude_conventions(tmp_path):
    centered = _hybrid_ds(lon_values=np.arange(-180.0, 180.0, 45.0))
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", centered)
    _manifest(tmp_path, "EMAC", [record])
    row = audit_model("EMAC", tmp_path)
    horizontal = row["per_file"][0]["horizontal"]
    assert horizontal["longitude_convention"] == "-180..180"
    assert horizontal["longitude_min"] == -180.0 and horizontal["longitude_max"] == 135.0
    assert horizontal["global_coverage"] is True and horizontal["longitude_monotonic"] is True
    assert "longitude_-180_180_handled_by_mod_in_zonal" in row["notes"]
    assert row["status"] == "supported"


def test_coordinate_audit_small_mapping_change(tmp_path):
    ds = _hybrid_ds(coord_names=("latitude", "longitude"))
    ds.latitude.attrs["standard_name"] = "latitude"
    ds.latitude.attrs["units"] = "degrees_north"
    ds.longitude.attrs["standard_name"] = "longitude"
    ds.longitude.attrs["units"] = "degrees_east"
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", ds)
    _manifest(tmp_path, "EMAC", [record])
    row = audit_model("EMAC", tmp_path)
    assert row["per_file"][0]["horizontal"]["grid"] == "regular_latlon"
    assert "horizontal_mapping_missing" in row["reasons"]
    assert row["status"] == "needs_mapping" and row["reader_compatibility"] == "small_mapping_change"


def test_coordinate_audit_curvilinear_needs_reader(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc",
                    _hybrid_ds(lat2d=True))
    _manifest(tmp_path, "EMAC", [record])
    row = audit_model("EMAC", tmp_path)
    assert row["per_file"][0]["horizontal"]["grid"] == "curvilinear"
    assert "curvilinear_grid_support_needed" in row["reasons"]
    assert row["reader_compatibility"] == "reader_extension_needed"


def test_coordinate_audit_nonstandard_calendar_and_monthly(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc",
                    _hybrid_ds(calendar="360_day"))
    _manifest(tmp_path, "EMAC", [record])
    row = audit_model("EMAC", tmp_path)
    time = row["per_file"][0]["time"]
    assert time["calendar"] == "360_day" and time["time_decode_ok"] is True
    assert time["monthly"] is True and time["n_times"] == 4
    assert row["time_decode"] == "yes"


def test_coordinate_audit_time_decode_failure(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc",
                    _hybrid_ds(time_strings=True), start="199001", end="199004")
    _manifest(tmp_path, "EMAC", [record])
    row = audit_model("EMAC", tmp_path)
    assert row["per_file"][0]["time"]["time_decode_ok"] is False
    assert "time_decode_failed" in row["reasons"]
    assert row["status"] == "unsupported"


def test_coordinate_audit_waccmx_separate_handling(tmp_path):
    record = _store(tmp_path, "WACCM-X", None, "T", "T_1990-01_zm.nc", _waccmx_ds())
    _manifest(tmp_path, "WACCM-X", [record])
    row = audit_model("WACCM-X", tmp_path)
    entry = row["per_file"][0]
    assert entry["horizontal"]["grid"] == "already_zonal"
    assert entry["vertical"]["formula_terms"] == {"a": "hyam", "b": "hybm", "ps": "PS", "p0": "P0"}
    assert entry["vertical"]["coordinate_type"] == "hybrid_sigma_pressure"
    assert entry["time"]["monthly"] is True
    assert "waccmx_separate_whole_atmosphere_handling" in row["notes"]
    assert "waccmx_separate_handling_preserved" in row["reasons"]
    assert row["status"] == "supported"


def test_coordinate_audit_variable_metadata_no_full_load(tmp_path):
    record = _store(tmp_path, "EMAC", "Amon", "ta", "ta_Amon_EMAC_refD1_gn_r1i1p1f1_199001-199012.nc", _hybrid_ds())
    _manifest(tmp_path, "EMAC", [record])
    metadata = audit_model("EMAC", tmp_path)["per_file"][0]["variable_metadata"]
    assert metadata["native"] == "ta" and metadata["dims"] == ["time", "lev", "lat", "lon"]
    assert metadata["units"] == "K" and metadata["standard_name"] == "air_temperature"
    assert metadata["dtype"].startswith("float32") and "_FillValue" in metadata
    assert metadata["sample_read_ok"] is True


def test_not_downloaded_coordinate_audit_and_unreadable(tmp_path):
    assert audit_model("EMAC", tmp_path)["status"] == "not_downloaded"
    bad = tmp_path / "data/raw/EMAC/refD1/Amon/ta/bad.nc"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 32)
    record = {"family": "Amon", "variable": "ta", "filename": "bad.nc", "start": "199001", "end": "199012",
              "size_bytes": bad.stat().st_size, "local_path": str(bad.relative_to(tmp_path))}
    _manifest(tmp_path, "EMAC", [record])
    assert audit_model("EMAC", tmp_path)["status"] == "unsupported"


def test_all_model_aggregation_and_deterministic_order(tmp_path):
    raw_rows = write_raw_validation(models=None, root=tmp_path)
    expected_order = list(registry())
    statuses = json.loads((tmp_path / "products/diagnostics/raw_validation/EMAC.json").read_text())
    assert statuses["status"] == "not_downloaded"
    coordinate_rows = write_coordinate_audit(models=None, root=tmp_path)
    for paths in (raw_rows, coordinate_rows):
        assert paths[0].exists() and paths[1].exists()
    csv_text = (tmp_path / "products/comparison/coordinate_audit.csv").read_text().splitlines()
    assert [line.split(",")[0] for line in csv_text[1:]] == expected_order
    assert "not_downloaded" in (tmp_path / "products/comparison/coordinate_audit.md").read_text()
