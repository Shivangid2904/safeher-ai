"""
SafeHer AI — Milestone 2
backend/api/routes.py

Flask Blueprint: routing_bp
Prefix: /api

Endpoints:
  POST /api/route   — compute shortest path between two lat/lon points
  GET  /api/health  — liveness check
"""

import logging

from flask import Blueprint, request, jsonify

from core.routing_service import (
    compute_shortest_path,
    find_nearest_node,
    get_graph,
)

logger = logging.getLogger(__name__)

routing_bp = Blueprint("routing_bp", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@routing_bp.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "SafeHer Routing Engine",
        "milestone": 2,
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
            "end_lon":   float
        }

    Response 200:
        {
            "distance_meters": float,
            "node_count":      int,
            "route_nodes":     [int, ...],
            "geojson":         { GeoJSON Feature }
        }

    Response 400: invalid / missing input
    Response 422: no path exists
    Response 500: unexpected server error
    """
    data = request.get_json(silent=True)

    # ── validate input ────────────────────────────────────────────────────
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

    # ── routing workflow ──────────────────────────────────────────────────
    try:
        G          = get_graph()                                # cached — no DB hit
        start_node = find_nearest_node(start_lat, start_lon)
        end_node   = find_nearest_node(end_lat, end_lon)
        result     = compute_shortest_path(start_node, end_node, G)
        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Route not found: %s", exc)
        return jsonify({"error": str(exc)}), 422

    except Exception as exc:
        logger.exception("Unexpected error during routing: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
