"""
SafeHer AI — Milestone 3
backend/api/safe_havens.py

Flask Blueprint: safe_havens_bp
Prefix: /api

Endpoints:
  GET /api/safe-havens           — all safe havens, optional ?category= filter
  GET /api/safe-havens/nearby    — safe havens within radius of a lat/lon point
  GET /api/safe-havens/health    — liveness check
"""

import json
import logging

from flask import Blueprint, request, jsonify

from core.routing_service import get_db_connection

logger = logging.getLogger(__name__)

safe_havens_bp = Blueprint("safe_havens_bp", __name__, url_prefix="/api")

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@safe_havens_bp.get("/safe-havens/health")
def safe_havens_health():
    return jsonify({
        "status": "ok",
        "service": "Safe Havens",
        "milestone": 3,
    }), 200


# ---------------------------------------------------------------------------
# All safe havens
# ---------------------------------------------------------------------------

@safe_havens_bp.get("/safe-havens")
def get_safe_havens():
    """
    GET /api/safe-havens
    Optional query param: category  ('police' | 'hospital' | 'pharmacy')
    """
    category = request.args.get("category", "").strip() or None

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if category:
                cur.execute(
                    """
                    SELECT id, name, category, latitude, longitude
                    FROM safe_havens
                    WHERE category = %s
                    ORDER BY id;
                    """,
                    (category,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, name, category, latitude, longitude
                    FROM safe_havens
                    ORDER BY id;
                    """
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    havens = [
        {
            "id":        row[0],
            "name":      row[1],
            "category":  row[2],
            "latitude":  row[3],
            "longitude": row[4],
        }
        for row in rows
    ]

    return jsonify({"count": len(havens), "safe_havens": havens}), 200


# ---------------------------------------------------------------------------
# Nearby safe havens
# ---------------------------------------------------------------------------

@safe_havens_bp.get("/safe-havens/nearby")
def get_nearby_safe_havens():
    """
    GET /api/safe-havens/nearby?lat=<float>&lon=<float>[&radius=<float>]

    radius in metres, default 1000, silently clamped to [1, 5000].
    Results ordered by distance ASC.
    Uses ST_DWithin with ::geography cast for metre-accurate distance.
    """
    # ── validate lat / lon ────────────────────────────────────────────────
    lat_raw = request.args.get("lat")
    lon_raw = request.args.get("lon")

    if not lat_raw or not lon_raw:
        return jsonify({"error": "Query params 'lat' and 'lon' are required."}), 400

    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except ValueError:
        return jsonify({"error": "'lat' and 'lon' must be valid floats."}), 400

    # ── radius: default 1000, clamp to [1, 5000] silently ────────────────
    try:
        radius = float(request.args.get("radius", 1000))
    except ValueError:
        radius = 1000.0
    radius = max(1.0, min(5000.0, radius))

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    name,
                    category,
                    latitude,
                    longitude,
                    ST_Distance(
                        geom::geography,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                    ) AS distance_meters
                FROM safe_havens
                WHERE ST_DWithin(
                    geom::geography,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                    %s
                )
                ORDER BY distance_meters ASC;
                """,
                (lon, lat, lon, lat, radius),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    havens = [
        {
            "id":               row[0],
            "name":             row[1],
            "category":         row[2],
            "latitude":         row[3],
            "longitude":        row[4],
            "distance_meters":  round(float(row[5]), 2),
        }
        for row in rows
    ]

    return jsonify({
        "count":         len(havens),
        "radius_meters": radius,
        "safe_havens":   havens,
    }), 200
