"""Canonical dialect taxonomy and per-dataset mappings.

The canonical set is derived from GBI's six broad categories plus a small
set of non-UK/IE labels needed for VCTK and CMU Arctic speakers.

Canonical labels
----------------
scottish, welsh, irish, southern_english, midlands_english, northern_english,
american, canadian, indian, other
"""

from __future__ import annotations

# GBI utterance-ID prefix → canonical label (ground truth: no mapping needed)
GBI_CODE_TO_DIALECT: dict[str, str] = {
    "sc": "scottish",
    "we": "welsh",
    "ir": "irish",
    "so": "southern_english",
    "mi": "midlands_english",
    "no": "northern_english",
}

# VCTK accent column → canonical label (for non-English speakers, direct)
# For accent="English" the region column is needed; see vctk_to_canonical().
_VCTK_ACCENT_DIRECT: dict[str, str] = {
    "Scottish":       "scottish",
    "Welsh":          "welsh",
    "Irish":          "irish",
    "NorthernIrish":  "irish",   # VCTK uses "NorthernIrish" without space
    "American":       "american",
    "Canadian":       "canadian",
    "Indian":         "indian",
}

# VCTK region column → canonical label (used when accent == "English")
# Source: VCTK 0.92 speaker-info.txt region values.
# Regions not listed here fall through to "other".
_VCTK_REGION_TO_DIALECT: dict[str, str] = {
    # Southern
    "Southern England":        "southern_english",
    "South East England":      "southern_english",
    "South West England":      "southern_english",
    "East Anglia":             "southern_english",
    "Home Counties":           "southern_english",
    "London":                  "southern_english",
    # Midlands
    "Midlands":                "midlands_english",
    "West Midlands":           "midlands_english",
    "East Midlands":           "midlands_english",
    "Staffordshire":           "midlands_english",
    "Shropshire":              "midlands_english",
    # Northern
    "Yorkshire":               "northern_english",
    "Lancashire":              "northern_english",
    "Greater Manchester":      "northern_english",
    "Merseyside":              "northern_english",
    "Tyneside":                "northern_english",
    "Geordie":                 "northern_english",
    "Humberside":              "northern_english",
    "Cumbria":                 "northern_english",
}


def vctk_to_canonical(accent: str, region: str) -> str:
    """Map a VCTK (accent, region) pair to a canonical dialect label."""
    if accent != "English":
        return _VCTK_ACCENT_DIRECT.get(accent, "other")
    return _VCTK_REGION_TO_DIALECT.get(region, "other")


# CMU Arctic accent column → canonical label
ARCTIC_ACCENT_TO_DIALECT: dict[str, str] = {
    "US":         "american",
    "American":   "american",
    "Canadian":   "canadian",
    "Scottish":   "scottish",
    "Indian":     "indian",
    "Filipino":   "other",
    "Korean":     "other",
}


def arctic_to_canonical(accent: str) -> str:
    return ARCTIC_ACCENT_TO_DIALECT.get(accent, "other")
