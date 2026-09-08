"""
SafeHer AI — Milestone 6
backend/core/safety_score.py

Deterministic, rule-based Safety Score Engine.

Design principle
----------------
This module is a pure function: given a feature dict, it returns a
structured safety assessment.  No database calls, no side effects.

The interface is intentionally minimal so that this module can be
swapped for an ML model in a later milestone without touching the API
layer.

Scoring formula
---------------
  base             = 100
  - 0.5  × effective_average_risk
  - 3    × incident_reports_near_route
  - 5    × high_risk_zones
  + 2    × safe_havens_nearby
  clamped to [0, 100]

Safety levels
-------------
  80–100  SAFE
  60–79   LOW RISK
  40–59   MODERATE
  20–39   HIGH RISK
   0–19   CRITICAL

Public API
----------
calculate_safety_score(features: dict) -> dict
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_LEVEL_THRESHOLDS = [
    (80, "SAFE"),
    (60, "LOW RISK"),
    (40, "MODERATE"),
    (20, "HIGH RISK"),
    (0,  "CRITICAL"),
]

_CONFIDENCE_RULE_BASED = 0.95


def _safety_level(score: float) -> str:
    """Map a clamped 0-100 score to its descriptive safety level."""
    for threshold, label in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "CRITICAL"


# ---------------------------------------------------------------------------
# Positive factor generators
# ---------------------------------------------------------------------------

def _positive_factors(features: dict) -> list[str]:
    """Generate human-readable positive factor sentences from feature values."""
    factors: list[str] = []

    sh = features.get("safe_havens_nearby", 0)
    if sh >= 3:
        factors.append(f"{sh} safe havens (hospitals, police stations, pharmacies) are within 200 m of your route.")
    elif sh >= 1:
        factors.append(f"{sh} safe haven(s) detected nearby — emergency access is available.")

    inc = features.get("incident_reports_near_route", 0)
    if inc == 0:
        factors.append("No active community incidents have been reported near this route.")
    elif inc <= 3:
        factors.append("Very few recent incident reports near this route.")

    eff = features.get("effective_average_risk", 0.0)
    if eff <= 25.0:
        factors.append("Average effective route risk is low — roads along this path are generally safe.")
    elif eff <= 40.0:
        factors.append("Moderate average route risk — the path avoids the most hazardous segments.")

    dist = features.get("distance_meters", 0.0)
    if dist <= 600:
        factors.append("Short travel distance minimises overall exposure time.")
    elif dist <= 1500:
        factors.append("Moderate travel distance — route is compact and well-connected.")

    hrz = features.get("high_risk_zones", 0)
    if hrz == 0:
        factors.append("No road segments with critically elevated effective risk were found on this route.")

    mode = features.get("routing_mode", "fastest")
    if mode == "safest":
        factors.append("Safest routing mode selected — the algorithm actively avoids high-risk and incident-affected roads.")
    elif mode == "balanced":
        factors.append("Balanced routing mode provides a good compromise between travel speed and road safety.")

    return factors


# ---------------------------------------------------------------------------
# Negative factor generators
# ---------------------------------------------------------------------------

def _negative_factors(features: dict) -> list[str]:
    """Generate human-readable negative factor sentences from feature values."""
    factors: list[str] = []

    inc = features.get("incident_reports_near_route", 0)
    if inc > 10:
        factors.append(f"{inc} recent community incidents reported within 200 m of this route — elevated caution required.")
    elif inc > 3:
        factors.append(f"{inc} active incident reports detected near the route.")

    hrz = features.get("high_risk_zones", 0)
    if hrz >= 3:
        factors.append(f"{hrz} road segments have an elevated effective risk score (> 50) — these areas may be unsafe.")
    elif hrz >= 1:
        factors.append(f"{hrz} road segment(s) with elevated effective risk detected on this path.")

    eff = features.get("effective_average_risk", 0.0)
    if eff >= 60.0:
        factors.append(f"High average effective route risk ({eff:.1f}/100) — multiple risk factors compound along this path.")
    elif eff >= 40.0:
        factors.append(f"Moderate-to-high average effective risk ({eff:.1f}/100) — proceed with awareness.")

    sh = features.get("safe_havens_nearby", 0)
    if sh == 0:
        factors.append("No nearby safe havens (hospitals, police, pharmacies) detected along this route.")

    dist = features.get("distance_meters", 0.0)
    if dist > 3000:
        factors.append("Long travel distance increases cumulative exposure time on the road.")

    mode = features.get("routing_mode", "fastest")
    if mode == "fastest":
        factors.append("Fastest routing mode prioritises speed over safety — consider 'balanced' or 'safest' modes.")

    return factors


# ---------------------------------------------------------------------------
# Recommendation generators
# ---------------------------------------------------------------------------

def _recommendations(features: dict, safety_score: float) -> list[str]:
    """Generate contextual, risk-relevant recommendations."""
    recs: list[str] = []

    inc = features.get("incident_reports_near_route", 0)
    hrz = features.get("high_risk_zones", 0)
    eff = features.get("effective_average_risk", 0.0)
    sh  = features.get("safe_havens_nearby", 0)
    dist = features.get("distance_meters", 0.0)
    mode = features.get("routing_mode", "fastest")

    # Always recommended regardless of score
    recs.append("Share your live location with a trusted contact before starting your journey.")
    recs.append("Keep emergency contacts accessible (Police: 100, Ambulance: 108, Women helpline: 1091).")

    if safety_score < 80 or eff >= 40.0:
        recs.append("Prefer daytime travel — visibility and bystander presence significantly improve safety.")

    if safety_score < 60 or hrz >= 1:
        recs.append("Stay on main, well-lit roads and avoid narrow lanes or isolated shortcuts.")

    if inc > 3:
        recs.append(f"Avoid areas with recent incident activity — {inc} reports were filed near this route.")

    if sh == 0:
        recs.append("Identify safe refuges (shops, ATMs, public spaces) along the route before departing.")

    if mode == "fastest" and safety_score < 70:
        recs.append("Switch to 'balanced' or 'safest' routing mode to reduce exposure to risky road segments.")

    if dist > 2000:
        recs.append("For long journeys, plan rest points in safe, populated areas.")

    if safety_score < 40:
        recs.append("Consider an alternative route or travel with a companion for this path.")

    if safety_score < 20:
        recs.append("CRITICAL: This route has a very high risk rating. Strongly consider an alternative path or postpone travel.")

    return recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_safety_score(features: dict) -> dict:
    """
    Calculate a deterministic safety assessment from the route feature dict.

    Parameters
    ----------
    features : dict
        Output of extract_route_features().

    Returns
    -------
    dict with keys:
        safety_score      (float, 0–100)
        safety_level      (str)
        confidence        (float, always 0.95 for rule-based engine)
        positive_factors  (list[str])
        negative_factors  (list[str])
        recommendations   (list[str])
    """
    eff = features.get("effective_average_risk",      0.0)
    inc = features.get("incident_reports_near_route", 0)
    hrz = features.get("high_risk_zones",             0)
    sh  = features.get("safe_havens_nearby",          0)

    raw_score = (
        100.0
        - 0.5 * eff
        - 3.0 * inc
        - 5.0 * hrz
        + 2.0 * sh
    )

    score = round(max(0.0, min(100.0, raw_score)), 2)
    level = _safety_level(score)

    pos_factors = _positive_factors(features)
    neg_factors = _negative_factors(features)
    recs        = _recommendations(features, score)

    logger.info(
        "Safety score computed: %.2f (%s) — pos=%d neg=%d recs=%d",
        score, level, len(pos_factors), len(neg_factors), len(recs),
    )

    return {
        "safety_score":     score,
        "safety_level":     level,
        "confidence":       _CONFIDENCE_RULE_BASED,
        "positive_factors": pos_factors,
        "negative_factors": neg_factors,
        "recommendations":  recs,
    }
