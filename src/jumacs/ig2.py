"""IG2 standard-atmosphere profiles used for upper-tail completion.

IG2 (MIPAS reference atmospheres, v3.1) provides five archetypal profiles on a
0-120 km, 1 km grid that matches the compact JuMACS altitude grid.  Only the
profiles that carry a genuine upper-atmosphere gradient for a given species are
used as donors; the latitude and seasonal mapping below turns the five discrete
archetypes into a continuous field with no hard regime or month boundaries.
"""

import math
import re
from pathlib import Path

import numpy as np

VERSION = "3.1"
DATE = "2001-05-11"

# The five standard archetypes (companion max/min/sigma files are statistics, not profiles).
STANDARD_PROFILES = {
    "tropical": "mipas_eqn_d2.atm",
    "midlatitude_day": "mipas_midl_day_d2.atm",
    "midlatitude_night": "mipas_midl_ngt_d2.atm",
    "polar_summer": "mipas_psum_d2.atm",
    "polar_winter": "mipas_pwin_d2.atm",
}

# One central mapping: JuMACS variable key -> IG2 block name, with the highest
# altitude the IG2 profile can be trusted as a donor and whether values are
# strictly positive (so the vertical blend runs in log space). Species whose IG2
# upper tail is physically negligible yet disagrees with the lower model by many
# decades at the seam (e.g. HCFC-22 / F22) are intentionally omitted so they fall
# back to a constant upper carry instead of grafting a large log-space jump.
IG2_DONORS = {
    "h2o": {"ig2_species": "H2O", "donor_top_km": 120.0, "positive": True},
    "n2o": {"ig2_species": "N2O", "donor_top_km": 120.0, "positive": True},
    "co": {"ig2_species": "CO", "donor_top_km": 120.0, "positive": True},
    "h2o2": {"ig2_species": "H2O2", "donor_top_km": 120.0, "positive": True},
    "clo": {"ig2_species": "ClO", "donor_top_km": 99.0, "positive": True},
}

# Northern-hemisphere regime latitudes; the southern hemisphere mirrors by |latitude|.
REGIME_LATITUDE_KNOTS = (0.0, 45.0, 75.0)

_NUMBER = re.compile(r"^[-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?$")


def _is_number(token):
    return bool(_NUMBER.match(token))


def parse_atm(path):
    """Parse one MIPAS ``.atm`` file into an altitude grid and named level arrays."""
    tokens = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        tokens.extend(stripped.split())
    index = 0
    while index < len(tokens) and not tokens[index].startswith("*"):
        index += 1
    nalt = int(float(tokens[0])) if tokens and _is_number(tokens[0]) else None
    fields = {}
    while index < len(tokens):
        if not tokens[index].startswith("*"):
            index += 1
            continue
        name = tokens[index][1:]
        if name.upper() == "END":
            break
        cursor = index + 1
        unit_parts = []
        while cursor < len(tokens) and not tokens[cursor].startswith("*") and not _is_number(tokens[cursor]):
            unit_parts.append(tokens[cursor].strip("[]").lower())
            cursor += 1
        values = []
        while cursor < len(tokens) and not tokens[cursor].startswith("*"):
            if _is_number(tokens[cursor]):
                values.append(float(tokens[cursor].lower().replace("d", "e")))
            cursor += 1
        fields[name] = {"unit": "".join(unit_parts), "values": np.asarray(values, dtype=float)}
        index = cursor
    height = fields.get("HGT", {}).get("values")
    if height is None or (nalt is not None and len(height) != nalt):
        raise ValueError(f"{path} does not contain a consistent {nalt}-level height grid")
    return nalt, height, fields


def load_regimes(ig2_dir):
    """Load the five standard archetypes with mixing ratios in mol/mol."""
    directory = Path(ig2_dir)
    regimes = {}
    height = None
    for regime, filename in STANDARD_PROFILES.items():
        _nalt, profile_height, fields = parse_atm(directory / filename)
        if height is None:
            height = profile_height
        elif profile_height.shape != height.shape or not np.allclose(profile_height, height):
            raise ValueError(f"{filename} altitude grid differs from the other IG2 archetypes")
        converted = {}
        for name, block in fields.items():
            values = block["values"].copy()
            if block["unit"] == "ppmv":
                values = values * 1e-6
            converted[name.upper()] = values
        regimes[regime] = converted
    return {"altitude_km": height, "nalt": height.size, "regimes": regimes}


def latitude_weights(abs_latitude):
    """Continuous (tropical, midlatitude, polar) weights over |latitude|; clamped at 75 deg."""
    magnitude = abs(float(abs_latitude))
    if magnitude <= REGIME_LATITUDE_KNOTS[1]:
        span = magnitude / REGIME_LATITUDE_KNOTS[1]
        return (1.0 - span, span, 0.0)
    if magnitude <= REGIME_LATITUDE_KNOTS[2]:
        span = (magnitude - REGIME_LATITUDE_KNOTS[1]) / (REGIME_LATITUDE_KNOTS[2] - REGIME_LATITUDE_KNOTS[1])
        return (0.0, 1.0 - span, span)
    return (0.0, 0.0, 1.0)


def summer_weight(month):
    """Single-harmonic summer fraction anchored so January is the winter minimum."""
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * (month - 1) / 12.0))


def donor_column(regime_data, ig2_species, months, latitudes):
    """Continuous IG2 field (month, altitude, latitude) from the five archetypes.

    Midlatitude daytime and nighttime archetypes are averaged; the tropical and
    midlatitude archetypes carry no IG2 seasonal information so they stay fixed,
    while the polar archetype rotates smoothly between winter and summer.
    """
    regimes = regime_data["regimes"]
    for regime in STANDARD_PROFILES:
        if ig2_species.upper() not in regimes[regime]:
            raise KeyError(f"IG2 species {ig2_species!r} missing from archetype {regime!r}")
    tropical = regimes["tropical"][ig2_species.upper()]
    midlatitude = 0.5 * (regimes["midlatitude_day"][ig2_species.upper()]
                         + regimes["midlatitude_night"][ig2_species.upper()])
    polar_summer = regimes["polar_summer"][ig2_species.upper()]
    polar_winter = regimes["polar_winter"][ig2_species.upper()]
    column = np.empty((len(months), tropical.size, len(latitudes)), dtype=float)
    for j, latitude in enumerate(latitudes):
        tropical_w, mid_w, polar_w = latitude_weights(latitude)
        for index, month in enumerate(months):
            summer = summer_weight(int(month))
            polar = summer * polar_summer + (1.0 - summer) * polar_winter
            column[index, :, j] = tropical_w * tropical + mid_w * midlatitude + polar_w * polar
    return column
