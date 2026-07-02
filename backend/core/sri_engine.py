"""
SafeHer AI — Milestone 4
backend/core/sri_engine.py

Deterministic, rule-based SafeHer Risk Index (SRI) engine.

This module is intentionally kept STATELESS and ML-free so that it
can be swapped for a machine-learning model in a later milestone
without modifying the routing engine or API layer.

Public API
----------
calculate_edge_risk_profile(road_type, is_lit, road_length,
                            distance_to_safe_haven) -> dict
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Road-type penalty table (lower = safer)
# ---------------------------------------------------------------------------
_ROAD_TYPE_PENALTIES: dict[str, float] = {
    "motorway":     5.0,
    "trunk":        7.0,    # between motorway and primary
    "primary":     10.0,
    "primary_link": 10.0,
    "secondary":   15.0,
    "secondary_link": 15.0,
    "tertiary":    20.0,
    "tertiary_link": 20.0,
    "unclassified": 25.0,
    "residential": 35.0,
    "service":     40.0,
    "footway":     45.0,
    "path":        50.0,
    "cycleway":    45.0,
    "pedestrian":  40.0,
    "living_street": 35.0,
}

_DEFAULT_ROAD_PENALTY = 30.0   # "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _road_type_penalty(road_type: str | None) -> float:
    """Return the road-type penalty for the given OSM highway tag."""
    if not road_type:
        return _DEFAULT_ROAD_PENALTY
    key = str(road_type).strip().lower()
    return _ROAD_TYPE_PENALTIES.get(key, _DEFAULT_ROAD_PENALTY)


def _lighting_penalty(is_lit: bool | None) -> float:
    """Return 0 when the road is lit, 15 otherwise."""
    return 0.0 if is_lit else 15.0


def _road_length_penalty(road_length: float | None) -> float:
    """Return a penalty based on segment length in metres."""
    if road_length is None or road_length < 0:
        return 5.0      # sensible default for unknown length
    if road_length < 50:
        return 0.0
    if road_length < 100:
        return 3.0
    if road_length < 250:
        return 5.0
    if road_length < 500:
        return 8.0
    return 12.0


def _safe_haven_bonus(distance_m: float | None) -> float:
    """Return a safety bonus based on distance to nearest safe haven (metres).

    A larger bonus means the edge is SAFER (subtracted from risk).
    """
    if distance_m is None:
        return 0.0
    if distance_m <= 100:
        return 20.0
    if distance_m <= 250:
        return 15.0
    if distance_m <= 500:
        return 8.0
    if distance_m <= 1000:
        return 3.0
    return 0.0


def _risk_category(score: float) -> str:
    """Map a clamped 0-100 score to a risk category string."""
    if score <= 25:
        return "LOW"
    if score <= 50:
        return "MODERATE"
    if score <= 75:
        return "HIGH"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def calculate_edge_risk_profile(
    road_type: str | None,
    is_lit: bool | None,
    road_length: float | None,
    distance_to_safe_haven: float | None,
) -> dict:
    """
    Calculate the rule-based SRI profile for a single road edge.

    Parameters
    ----------
    road_type : str or None
        OSM highway tag value (e.g. 'residential', 'primary').
    is_lit : bool or None
        True when the road is marked as lit in OSM.
    road_length : float or None
        Length of the road segment in metres.
    distance_to_safe_haven : float or None
        Geodesic distance (metres) to the nearest Safe Haven.
        None means no Safe Haven found within search radius.

    Returns
    -------
    dict with keys:
        sri_score          (float, 0–100)
        confidence_score   (float, always 1.0 — deterministic engine)
        risk_category      (str: LOW | MODERATE | HIGH | CRITICAL)
        risk_attributions  (dict of individual penalty / bonus values)
    """
    rtp  = _road_type_penalty(road_type)
    lp   = _lighting_penalty(is_lit)
    rlp  = _road_length_penalty(road_length)
    shb  = _safe_haven_bonus(distance_to_safe_haven)

    raw_score = rtp + lp + rlp - shb
    score     = max(0.0, min(100.0, raw_score))

    return {
        "sri_score":        round(score, 4),
        "confidence_score": 1.0,
        "risk_category":    _risk_category(score),
        "risk_attributions": {
            "road_type_penalty":    round(rtp,  4),
            "lighting_penalty":     round(lp,   4),
            "road_length_penalty":  round(rlp,  4),
            "safe_haven_bonus":     round(shb,  4),
        },
    }
