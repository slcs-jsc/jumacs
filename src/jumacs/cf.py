"""CF-1.13 conventions for the published climatology: time axis, cell methods, validation.

The published product is a climatology, not a time series. CF describes that
case with a climatological time axis: ``time`` carries one representative point
per calendar month, ``climatology_bounds`` brackets the interval the statistic
summarises, and the variable attribute ``climatology`` names that bounds
variable. The across-year statistic itself is recorded in ``cell_methods``, so
no statistic name is baked into ``standard_name``.

The month intervals are taken from a common year that is not a leap year, so
February always has 28 days and the annual cycle has exactly twelve intervals.
The reference date of the time units follows the configured nominal reference
year, which keeps the axis tied to the documented climatological normal.
"""
import datetime as dt
import re
import numpy as np

CONVENTIONS = "CF-1.13"
BOUNDS_DIMENSION = "nv"
STATISTIC_ORDER = ("mean", "sigma", "minimum", "maximum", "n_years")
STATISTIC_DTYPES = {"mean": "float32", "sigma": "float32", "minimum": "float32",
                    "maximum": "float32", "n_years": "int16"}

CELL_METHODS = {
    "mean": "longitude: mean time: mean within years time: mean over years",
    "sigma": "longitude: mean time: mean within years time: standard_deviation over years",
    "minimum": "longitude: mean time: mean within years time: minimum over years",
    "maximum": "longitude: mean time: mean within years time: maximum over years",
    "n_years": "longitude: mean time: sample_size over years",
}

STATISTIC_LONG_NAME = {
    "mean": "climatological mean",
    "sigma": "climatological standard deviation",
    "minimum": "climatological minimum",
    "maximum": "climatological maximum",
    "n_years": "number of years contributing a finite value",
}

TIME_ATTRS = {"axis": "T", "standard_name": "time",
              "climatology": "climatology_bounds",
              "long_name": "climatological month"}

_NAME_SYNTAX = re.compile(r"^[a-z][a-z0-9_]*$")
_MIXING_PATTERN = re.compile(r"^(mole|mass)_(fraction|mixing_ratio)_of_[a-z0-9_]+_in_(air|ambient_air)$")
# Names that appear in the CCMI-2022 zonal products and are unambiguously CF.
ALLOWED_NAMES = {
    "air_temperature", "geopotential_height", "geopotential", "surface_temperature",
    "surface_air_pressure", "air_pressure", "specific_humidity", "relative_humidity",
    "cloud_liquid_water_content", "cloud_mixing_ratio", "cloud_area_fraction",
    "atmosphere_mass_streamfunction", "atmosphere_meridional_mass_streamfunction",
    "eastward_wind", "northward_wind", "vertical_wind", "atmosphere_vertical_velocity",
    "ozone_mixing_ratio", "water_vapor_mixing_ratio",
    "atmosphere_hybrid_sigma_pressure_coordinate", "atmosphere_sigma_coordinate",
    "barotropic_streamfunction", "atmosphere_kinetic_energy_content",
    "equivalent_effective_temperature_of_radiation", "air_content_of_ozone",
    "atmosphere_mass_content_of_ozone", "atmosphere_mass_content_of_tracer",
    "tropopause_air_pressure", "tropopause_air_temperature",
    "tropopause_geopotential_height", "atmosphere_mole_fraction_of_species",
    "lagrangian_tendency_of_air_pressure",
}


def valid_standard_name(name):
    """A CF name only when it is syntactically valid and recognised.

    Nothing is invented here and no CF table is downloaded: an unrecognised
    source name is dropped so the field keeps a correct ``long_name`` instead of
    carrying a fabricated ``standard_name``.
    """
    text = str(name or "").strip()
    if not text or not _NAME_SYNTAX.match(text) or "__" in text or text.endswith("_"):
        return None
    if text in ALLOWED_NAMES or _MIXING_PATTERN.match(text):
        return text
    if text.startswith(("atmosphere_", "sea_", "at_")) and text.count("_") >= 2 and text.endswith("_air"):
        return text
    return None


# Unit agreement for names assigned from model configuration. A curated name is a
# claim made by this project rather than read from the archive, so it is published
# only when the field's units match what the CF name requires (units are compared
# after folding negative exponents into slash form, so "m s-1" equals "m/s").
EXPECTED_UNITS = {
    "air_temperature": {"k"},
    "surface_temperature": {"k"},
    "surface_air_pressure": {"pa"},
    "air_pressure": {"pa"},
    "eastward_wind": {"m/s"},
    "northward_wind": {"m/s"},
    "lagrangian_tendency_of_air_pressure": {"pa/s"},
    "geopotential_height": {"m2/s2"},
    "geopotential": {"m2/s2"},
}
MOLE_FRACTION_UNITS = {"mol/mol", "mole/mole", "1"}
MASS_FRACTION_UNITS = {"kg/kg", "1"}

_NEGATIVE_EXPONENT = re.compile(r"([a-z0-9]+)-(\d+)")


def _units_key(units):
    """Fold a units string into one comparable spelling: ``m s-1`` and ``m/s`` agree."""
    text = str(units or "").strip().lower().replace("**", "").replace("^", "")
    text = _NEGATIVE_EXPONENT.sub(lambda m: "/" + m.group(1) + ("" if m.group(2) == "1" else m.group(2)), text)
    return re.sub(r"\s+", "", text)


def curated_standard_name(candidate, units):
    """A configured ``standard_name``, accepted only when its units agree.

    Returns ``None`` for an unrecognised name, for missing units, and when a
    listed name and the field's units disagree - for example ``geopotential_height``
    (CF units ``m2 s-2``) on a field stored in metres.
    """
    name = valid_standard_name(candidate)
    key = _units_key(units)
    if not name or not key:
        return None
    if name.startswith(("mole_fraction_of_", "mole_mixing_ratio_of_")):
        return name if key in MOLE_FRACTION_UNITS else None
    if name.startswith(("mass_fraction_of_", "mass_mixing_ratio_of_")):
        return name if key in MASS_FRACTION_UNITS else None
    expected = EXPECTED_UNITS.get(name)
    if expected is None:
        return name
    return name if key in expected else None


def _is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def climatology_time(start_year, end_year, nominal_reference_year):
    """CF climatological time coordinate: midpoints, bounds, units and calendar."""
    reference_year = int(nominal_reference_year)
    common_year = reference_year if not _is_leap(reference_year) else reference_year + 1
    reference = dt.datetime(reference_year, 1, 1)
    edges, centres = [], []
    for month in range(1, 13):
        start = dt.datetime(common_year, month, 1)
        stop = dt.datetime(common_year + 1, 1, 1) if month == 12 else dt.datetime(common_year, month + 1, 1)
        low = (start - reference).total_seconds() / 86400.0
        high = (stop - reference).total_seconds() / 86400.0
        edges.append((low, high))
        centres.append(0.5 * (low + high))
    units = f"days since {reference_year}-01-01"
    attrs = dict(TIME_ATTRS)
    attrs.update({"units": units, "calendar": "gregorian",
                  "climatology_month_intervals_year": common_year,
                  "comment": ("climatological time axis: one point per calendar month, sampled from every year in "
                              f"{start_year}-{end_year}; bounds span the month in a common non-leap year")})
    return {"values": np.asarray(centres, dtype="float64"),
            "bounds": np.asarray(edges, dtype="float64"),
            "units": units, "calendar": "gregorian", "attrs": attrs}


def cell_methods(statistic):
    try:
        return CELL_METHODS[statistic]
    except KeyError:
        raise ValueError(f"No CF cell_methods defined for statistic {statistic!r}") from None


def statistic_long_name(statistic):
    try:
        return STATISTIC_LONG_NAME[statistic]
    except KeyError:
        raise ValueError(f"No CF long_name defined for statistic {statistic!r}") from None


def time_bounds_variable(time):
    """The bounds variable named by ``time:climatology``, or None."""
    named = str(time.attrs.get("climatology", "")).strip()
    return named or None


class ProductProblem(ValueError):
    """A published product that does not meet the CF or methodology contract."""


def validate_product(ds, names=(), grid=None, statistics=STATISTIC_ORDER):
    """Structural and methodological checks on a written product (no external checker)."""
    problems = []
    grid = grid or {}
    coordinate = grid.get("coordinate", "pressure")
    if str(ds.attrs.get("Conventions", "")).strip() != CONVENTIONS:
        problems.append(f"Conventions must be {CONVENTIONS}, found {ds.attrs.get('Conventions')!r}")
    period = str(ds.attrs.get("climatology_period", "")).strip()
    span = re.fullmatch(r"(\d{4})-(\d{4})", period)
    if not span:
        problems.append(f"climatology_period must read 'YYYY-YYYY', found {period!r}")
    else:
        first, last = span.groups()
        if str(ds.attrs.get("time_coverage_start", "")).strip() != f"{first}-01-01":
            problems.append(f"time_coverage_start must be {first}-01-01 (the climatological period), found "
                            f"{ds.attrs.get('time_coverage_start')!r}; the span of the source series belongs in "
                            f"source_time_coverage_start")
        if str(ds.attrs.get("time_coverage_end", "")).strip() != f"{last}-12-31":
            problems.append(f"time_coverage_end must be {last}-12-31 (the climatological period), found "
                            f"{ds.attrs.get('time_coverage_end')!r}")
        years = ds.attrs.get("reference_period_years")
        if years is not None and int(years) != int(last) - int(first) + 1:
            problems.append(f"reference_period_years must be {int(last) - int(first) + 1} for {period}, found {years}")
    if "time" not in ds.dims or ds.sizes.get("time") != 12:
        problems.append(f"the time axis must hold 12 climatological months, found {ds.sizes.get('time')}")
    if "time" not in ds.coords:
        problems.append("time must be a coordinate variable")
    else:
        time = ds["time"]
        if time.dtype != np.float64:
            problems.append(f"time must be float64, found {time.dtype}")
        bounds_name = time_bounds_variable(time)
        if not bounds_name:
            problems.append("time:climatology must name the climatology bounds variable")
        elif bounds_name not in ds:
            problems.append(f"bounds variable {bounds_name!r} named by time:climatology is missing")
        else:
            bounds = ds[bounds_name]
            if list(bounds.dims) != ["time", BOUNDS_DIMENSION]:
                problems.append(f"{bounds_name} must have dimensions (time, {BOUNDS_DIMENSION}), found {list(bounds.dims)}")
            if bounds.dtype != np.float64:
                problems.append(f"{bounds_name} must be float64, found {bounds.dtype}")
            if "_FillValue" in bounds.encoding or "missing_value" in bounds.attrs:
                problems.append(f"{bounds_name} must not carry a fill value")
        if "units" not in time.attrs or "calendar" not in time.attrs:
            problems.append("time needs units and calendar")
    if coordinate not in ds.coords:
        problems.append(f"the shared {coordinate} coordinate is missing")
    else:
        pressure = ds[coordinate]
        if pressure.dtype != np.float64:
            problems.append(f"{coordinate} must be float64, found {pressure.dtype}")
        if pressure.attrs.get("standard_name") != "air_pressure" or pressure.attrs.get("units") != "Pa":
            problems.append(f"{coordinate} must be air_pressure in Pa")
        if pressure.attrs.get("positive") != "down" or pressure.attrs.get("axis") != "Z":
            problems.append(f"{coordinate} must carry positive=down and axis=Z")
        if grid.get("levels") is not None and not np.allclose(pressure.values, np.asarray(grid["levels"], float)):
            problems.append(f"{coordinate} is not the configured common grid")
    if "lat" not in ds.coords:
        problems.append("lat is missing")
    else:
        if ds["lat"].dtype != np.float64:
            problems.append(f"lat must be float64, found {ds['lat'].dtype}")
        if ds["lat"].attrs.get("units") != "degrees_north":
            problems.append("lat must be degrees_north")
    if any(name.endswith("_air_pressure") for name in ds.data_vars):
        problems.append("per-variable air_pressure fields must not be published; the pressure grid is shared")
    expected = {f"{name}_{statistic}" for name in names for statistic in statistics}
    present = set(ds.data_vars)
    for missing in sorted(expected - present):
        problems.append(f"missing field {missing}")
    for statistic, dtype in STATISTIC_DTYPES.items():
        for field in sorted(field for field in present if field.endswith(f"_{statistic}")):
            variable = ds[field]
            if variable.dtype != np.dtype(dtype):
                problems.append(f"{field} must be {dtype}, found {variable.dtype}")
            if variable.attrs.get("cell_methods", "") != cell_methods(statistic):
                problems.append(f"{field} cell_methods must be {cell_methods(statistic)!r}, "
                                f"found {variable.attrs.get('cell_methods')!r}")
            if statistic == "n_years":
                if variable.attrs.get("standard_name"):
                    problems.append(f"{field} must not claim a standard_name (it is a count)")
                if variable.attrs.get("units") != "1":
                    problems.append(f"{field} must have units of 1 (a count)")
            elif not variable.attrs.get("units"):
                problems.append(f"{field} must carry its units")
            standard_name = variable.attrs.get("standard_name", "")
            if standard_name and valid_standard_name(standard_name) is None:
                problems.append(f"{field} carries an unrecognised standard_name {standard_name!r}")
            if variable.ndim == 3 and list(variable.dims) != ["time", coordinate, "lat"]:
                problems.append(f"{field} must be ordered (time, {coordinate}, lat), found {list(variable.dims)}")
            elif variable.ndim == 2 and list(variable.dims) != ["time", "lat"]:
                problems.append(f"{field} must be ordered (time, lat), found {list(variable.dims)}")
            elif variable.ndim not in (2, 3):
                problems.append(f"{field} must be two- or three-dimensional, found {variable.ndim}")
    return problems


def assert_product(ds, **kwargs):
    problems = validate_product(ds, **kwargs)
    if problems:
        raise ProductProblem("; ".join(problems))
    return problems
