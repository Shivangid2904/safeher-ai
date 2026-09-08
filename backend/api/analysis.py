"""
SafeHer AI — Milestone 6
backend/api/analysis.py

Flask Blueprint: analysis_bp
Prefix: /api

Endpoints:
  GET  /api/analysis/health   — liveness check
  POST /api/route/analyze     — compute route + full safety analysis

The /api/route/analyze endpoint reuses EXACTLY the same routing pipeline
as /api/route (same validation, same compute_shortest_path call, same
safe-haven and incident enrichment), then passes the enriched result
to the route_analyzer and safety_score modules.

Nothing in this blueprint modifies routing behaviour.
"""

from __future__ import annotations

import json
import logging

from flask import Blueprint, request, jsonify

from core.routing_service import (
    compute_shortest_path,
    find_nearest_node,
    get_graph,
    get_db_connection,
)
from core.route_analyzer  import extract_route_features
from core.safety_score    import calculate_safety_score

logger = logging.getLogger(__name__)

analysis_bp = Blueprint("analysis_bp", __name__, url_prefix="/api")

VALID_MODES  = {"fastest", "balanced", "safest"}
DEFAULT_MODE = "fastest"


# ---------------------------------------------------------------------------
# Helpers (mirrors routes.py — no shared state so we keep them local)
# ---------------------------------------------------------------------------

def _fetch_nearby_safe_havens(geojson_geometry: dict) -> list[dict]:
    """Return safe havens within 200 m of the route LineString."""
    geojson_str = json.dumps(geojson_geometry)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, category, latitude, longitude
                FROM safe_havens
                WHERE ST_DWithin(
                    geom::geography,
                    ST_SetSRID(ST_GeomFromGeoJSON(%(g)s), 4326)::geography,
                    200
                );
                """,
                {"g": geojson_str},
            )
            rows = cur.fetchall()
        return [
            {"name": r[0], "category": r[1], "latitude": r[2], "longitude": r[3]}
            for r in rows
        ]
    except Exception as exc:
        logger.warning("safe_havens query failed: %s", exc)
        return []
    finally:
        conn.close()


def _count_active_reports_near_route(geojson_geometry: dict) -> int:
    """Count active community reports (last 7 days) within 200 m of the route."""
    geojson_str = json.dumps(geojson_geometry)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM community_reports
                WHERE created_at >= NOW() - INTERVAL '7 days'
                  AND ST_DWithin(
                      geom::geography,
                      ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)::geography,
                      200
                  );
                """,
                (geojson_str,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except Exception as exc:
        logger.error("Report count query failed: %s", exc)
        return 0
    finally:
        conn.close()


def _validate_and_parse(data: dict | None) -> tuple | str:
    """
    Validate the request body and return parsed values or an error string.
    Returns (start_lat, start_lon, end_lat, end_lon, mode) or a str error.
    """
    if not data:
        return "Request body must be valid JSON."
    required = ["start_lat", "start_lon", "end_lat", "end_lon"]
    missing = [f for f in required if f not in data]
    if missing:
        return f"Missing required fields: {missing}"
    try:
        start_lat = float(data["start_lat"])
        start_lon = float(data["start_lon"])
        end_lat   = float(data["end_lat"])
        end_lon   = float(data["end_lon"])
    except (TypeError, ValueError):
        return "All coordinate fields must be valid floats."
    mode = str(data.get("mode", DEFAULT_MODE)).strip().lower()
    if mode not in VALID_MODES:
        mode = DEFAULT_MODE
    return start_lat, start_lon, end_lat, end_lon, mode


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@analysis_bp.get("/analysis/health")
def analysis_health():
    """GET /api/analysis/health — liveness check for the analysis service."""
    return jsonify({
        "status":    "ok",
        "service":   "SafeHer Route Analysis",
        "milestone": 6,
    }), 200


# ---------------------------------------------------------------------------
# Analyze endpoint
# ---------------------------------------------------------------------------

@analysis_bp.post("/route/analyze")
def analyze_route():
    """
    POST /api/route/analyze

    Accepts the same JSON body as POST /api/route:
        {
            "start_lat": float,
            "start_lon": float,
            "end_lat":   float,
            "end_lon":   float,
            "mode":      string  (optional, default "fastest")
        }

    Returns:
        {
            "route":    { ...full route response, identical to /api/route... },
            "analysis": {
                "safety_score":     float   (0-100),
                "safety_level":     str,
                "confidence":       float,
                "positive_factors": [str, ...],
                "negative_factors": [str, ...],
                "recommendations":  [str, ...],
                "features":         { ...extracted feature dict... }
            }
        }

    Response 400: invalid / missing input
    Response 422: no path found
    Response 500: unexpected server error
    """
    data   = request.get_json(silent=True)
    parsed = _validate_and_parse(data)

    if isinstance(parsed, str):
        return jsonify({"error": parsed}), 400

    start_lat, start_lon, end_lat, end_lon, mode = parsed

    try:
        # ── Step 1: compute route (identical pipeline to /api/route) ─────
        G          = get_graph()
        start_node = find_nearest_node(start_lat, start_lon)
        end_node   = find_nearest_node(end_lat, end_lon)
        route      = compute_shortest_path(start_node, end_node, G, mode=mode)

        # ── Step 2: enrich with safe havens (Milestone 3) ────────────────
        line_geom = route["geojson"]["geometry"]
        route["nearby_safe_havens"] = _fetch_nearby_safe_havens(line_geom)

        # ── Step 3: enrich with incident count (Milestone 5) ─────────────
        reports_count = _count_active_reports_near_route(line_geom)
        route["incident_reports_near_route"] = reports_count
        route["geojson"]["properties"]["incident_reports_near_route"] = reports_count

        # ── Step 4: extract features ──────────────────────────────────────
        features = extract_route_features(route, mode=mode)

        # ── Step 5: calculate safety score ───────────────────────────────
        assessment = calculate_safety_score(features)

        return jsonify({
            "route":    route,
            "analysis": {
                **assessment,
                "features": features,
            },
        }), 200

    except ValueError as exc:
        logger.warning("Route not found: %s", exc)
        return jsonify({"error": str(exc)}), 422

    except Exception as exc:
        logger.exception("Unexpected error in /api/route/analyze: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
