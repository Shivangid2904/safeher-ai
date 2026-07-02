"""
SafeHer AI — Milestone 5
backend/api/routes.py

Flask Blueprint: routing_bp
Prefix: /api

Endpoints:
  POST /api/route   — compute route between two lat/lon points
  GET  /api/health  — liveness check

Milestone 5 additions (backward-compatible):
  • Accepts 'mode' field in JSON body (fastest | balanced | safest).
  • Queries the database for active community reports (last 7 days) within 200m of the path.
  • Returns incident_reports_near_route, effective_average_risk, high_risk_zones.
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

logger = logging.getLogger(__name__)

routing_bp = Blueprint("routing_bp", __name__, url_prefix="/api")

VALID_MODES  = {"fastest", "balanced", "safest"}
DEFAULT_MODE = "fastest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_nearby_safe_havens(geojson_geometry: dict) -> list[dict]:
    """
    Return safe havens within 200 m of the route LineString.

    geojson_geometry is the 'geometry' sub-dict (type+coordinates) from the
    route result — NOT the Feature wrapper.
    """
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
                    ST_SetSRID(ST_GeomFromGeoJSON(%(geojson_line)s), 4326)::geography,
                    200
                );
                """,
                {"geojson_line": geojson_str},
            )
            rows = cur.fetchall()

        return [
            {
                "name":      row[0],
                "category":  row[1],
                "latitude":  row[2],
                "longitude": row[3],
            }
            for row in rows
        ]

    except Exception as exc:
        logger.warning(
            "Could not query safe_havens (table may not exist yet): %s", exc
        )
        return []
    finally:
        conn.close()


def _count_active_reports_near_route(geojson_geometry: dict) -> int:
    """
    Count the number of active community reports (last 7 days) within 200m of the route.
    """
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
                (geojson_str,)
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except Exception as exc:
        logger.error("Failed to query active reports near route: %s", exc)
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Health Check (Milestone 2 endpoint — unchanged)
# ---------------------------------------------------------------------------

@routing_bp.get("/health")
def health():
    return jsonify({
        "status":    "ok",
        "service":   "SafeHer Routing Engine",
        "milestone": 5,
    }), 200


# ---------------------------------------------------------------------------
# Route endpoint
# ---------------------------------------------------------------------------

@routing_bp.post("/route")
def route():
    """
    POST /api/route

    Request body (JSON):
        {
            "start_lat": float,
            "start_lon": float,
            "end_lat":   float,
            "end_lon":   float,
            "mode":      string   (optional, default "fastest")
        }

    Response 200:
        {
            "distance_meters":             float,
            "node_count":                  int,
            "route_nodes":                 [int, ...],
            "average_route_risk":          float,
            "minimum_edge_risk":           float,
            "maximum_edge_risk":           float,
            "risk_category":               str,
            "effective_average_risk":      float,
            "incident_reports_near_route": int,
            "high_risk_zones":             int,
            "geojson":                     { GeoJSON Feature },
            "nearby_safe_havens":          [ {name, category, latitude, longitude}, ... ]
        }

    Response 400: invalid / missing input
    Response 422: no path exists
    Response 500: unexpected server error
    """
    data = request.get_json(silent=True)

    # ── Validate input ────────────────────────────────────────────────────
    required_fields = ["start_lat", "start_lon", "end_lat", "end_lon"]
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        start_lat = float(data["start_lat"])
        start_lon = float(data["start_lon"])
        end_lat   = float(data["end_lat"])
        end_lon   = float(data["end_lon"])
    except (TypeError, ValueError):
        return jsonify({"error": "All coordinate fields must be valid floats."}), 400

    # ── Routing mode ──────────────────────────────────────────────────────
    mode = str(data.get("mode", DEFAULT_MODE)).strip().lower()
    if mode not in VALID_MODES:
        logger.warning("Invalid routing mode '%s'; defaulting to 'fastest'.", mode)
        mode = DEFAULT_MODE

    # ── Routing workflow ──────────────────────────────────────────────────
    try:
        G          = get_graph()
        start_node = find_nearest_node(start_lat, start_lon)
        end_node   = find_nearest_node(end_lat, end_lon)
        
        # Calculate dynamic route using incident engine penalties
        result = compute_shortest_path(start_node, end_node, G, mode=mode)

        # ── Milestone 3 safe-havens overlay ──────────────────────────────
        line_geometry = result["geojson"]["geometry"]
        result["nearby_safe_havens"] = _fetch_nearby_safe_havens(line_geometry)

        # ── Milestone 5 active reports counter ───────────────────────────
        reports_count = _count_active_reports_near_route(line_geometry)
        
        # Write final values into route response
        result["incident_reports_near_route"] = reports_count
        result["geojson"]["properties"]["incident_reports_near_route"] = reports_count

        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Route not found: %s", exc)
        return jsonify({"error": str(exc)}), 422

    except Exception as exc:
        logger.exception("Unexpected error during routing: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
