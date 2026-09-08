"""
SafeHer AI — Milestone 6
backend/core/route_analyzer.py

Extracts a flat feature dictionary from the route result produced by
compute_shortest_path() + the routes.py enrichment layer.

Design principle: this module is STATELESS and READ-ONLY.
It never calls the database; it only transforms the already-computed
route dict into a feature vector that safety_score.py can consume.

Public API
----------
extract_route_features(route_result, mode) -> dict
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def extract_route_features(
    route_result: dict,
    mode: str = "fastest",
) -> dict:
    """
    Extract a normalised feature dictionary from a route response dict.

    Parameters
    ----------
    route_result : dict
        The full dict returned by compute_shortest_path() and enriched by
        routes.py (i.e. it already contains nearby_safe_havens,
        incident_reports_near_route, effective_average_risk, etc.).
    mode : str
        The routing mode used ('fastest' | 'balanced' | 'safest').

    Returns
    -------
    dict with keys:
        average_route_risk          (float)   — base SRI average
        effective_average_risk      (float)   — SRI + incident overlay
        maximum_edge_risk           (float)   — worst single edge SRI
        minimum_edge_risk           (float)   — best single edge SRI
        incident_reports_near_route (int)
        high_risk_zones             (int)
        safe_havens_nearby          (int)
        distance_meters             (float)
        node_count                  (int)
        routing_mode                (str)
        risk_category               (str)
    """
    safe_havens_nearby = len(route_result.get("nearby_safe_havens", []))

    features = {
        "average_route_risk":          float(route_result.get("average_route_risk",          0.0)),
        "effective_average_risk":      float(route_result.get("effective_average_risk",      0.0)),
        "maximum_edge_risk":           float(route_result.get("maximum_edge_risk",           0.0)),
        "minimum_edge_risk":           float(route_result.get("minimum_edge_risk",           0.0)),
        "incident_reports_near_route": int(  route_result.get("incident_reports_near_route", 0  )),
        "high_risk_zones":             int(  route_result.get("high_risk_zones",             0  )),
        "safe_havens_nearby":          safe_havens_nearby,
        "distance_meters":             float(route_result.get("distance_meters",             0.0)),
        "node_count":                  int(  route_result.get("node_count",                  0  )),
        "routing_mode":                str(mode),
        "risk_category":               str(  route_result.get("risk_category",               "UNKNOWN")),
    }

    logger.debug("Extracted features: %s", features)
    return features
