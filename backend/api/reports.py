"""
SafeHer AI — Milestone 5
backend/api/reports.py

Flask Blueprint: reports_bp
Prefix: /api

Endpoints:
  • GET  /api/reports/health      — check health of reports service
  • GET  /api/reports             — fetch all community reports
  • GET  /api/reports/nearby      — fetch reports within a radius of a coordinate
  • GET  /api/reports/statistics  — fetch reporting metrics & aggregated stats
  • POST /api/reports             — submit a new community report
"""

from __future__ import annotations

import logging
import uuid
from flask import Blueprint, request, jsonify

from core.routing_service import get_db_connection

logger = logging.getLogger(__name__)

reports_bp = Blueprint("reports_bp", __name__, url_prefix="/api")

VALID_CATEGORIES = {
    "harassment",
    "stalking",
    "poor_lighting",
    "unsafe_area",
    "suspicious_activity"
}


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@reports_bp.get("/reports/health")
def reports_health():
    """GET /api/reports/health — checks service health."""
    return jsonify({
        "status":    "ok",
        "service":   "Community Reports",
        "milestone": 5,
    }), 200


# ---------------------------------------------------------------------------
# GET /api/reports — Fetch all reports
# ---------------------------------------------------------------------------

@reports_bp.get("/reports")
def get_reports():
    """
    GET /api/reports
    Fetches all reports, using ST_X(geom) and ST_Y(geom) to obtain coordinates.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    report_id,
                    category,
                    severity,
                    description,
                    ST_Y(geom) AS latitude,
                    ST_X(geom) AS longitude,
                    created_at
                FROM community_reports
                ORDER BY created_at DESC;
            """)
            rows = cur.fetchall()
            
        reports = [
            {
                "report_id":   row[0],
                "category":    row[1],
                "severity":    row[2],
                "description": row[3],
                "latitude":    float(row[4]),
                "longitude":   float(row[5]),
                "created_at":  row[6].isoformat() if row[6] else None,
            }
            for row in rows
        ]
        
        return jsonify({"count": len(reports), "reports": reports}), 200
        
    except Exception as exc:
        logger.exception("Failed to get community reports: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# GET /api/reports/nearby — Find nearby reports
# ---------------------------------------------------------------------------

@reports_bp.get("/reports/nearby")
def get_nearby_reports():
    """
    GET /api/reports/nearby?lat=<float>&lon=<float>[&radius=<float>]
    Finds reports within radius (default 500m) using ST_DWithin geography,
    sorted by distance ASC.
    """
    lat_raw = request.args.get("lat")
    lon_raw = request.args.get("lon")
    
    if not lat_raw or not lon_raw:
        return jsonify({"error": "Query parameters 'lat' and 'lon' are required."}), 400
        
    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except ValueError:
        return jsonify({"error": "'lat' and 'lon' must be valid floats."}), 400
        
    try:
        radius = float(request.args.get("radius", 500.0))
    except ValueError:
        radius = 500.0
        
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    report_id,
                    category,
                    severity,
                    description,
                    ST_Y(geom) AS latitude,
                    ST_X(geom) AS longitude,
                    created_at,
                    ST_Distance(
                        geom::geography,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                    ) AS distance_meters
                FROM community_reports
                WHERE ST_DWithin(
                    geom::geography,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                    %s
                )
                ORDER BY distance_meters ASC;
            """, (lon, lat, lon, lat, radius))
            rows = cur.fetchall()
            
        reports = [
            {
                "report_id":       row[0],
                "category":        row[1],
                "severity":        row[2],
                "description":     row[3],
                "latitude":        float(row[4]),
                "longitude":       float(row[5]),
                "created_at":      row[6].isoformat() if row[6] else None,
                "distance_meters": round(float(row[7]), 2),
            }
            for row in rows
        ]
        
        return jsonify({
            "count":          len(reports),
            "radius_meters":  radius,
            "reports":        reports
        }), 200
        
    except Exception as exc:
        logger.exception("Failed to get nearby reports: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# GET /api/reports/statistics — Reporting statistics
# ---------------------------------------------------------------------------

@reports_bp.get("/reports/statistics")
def get_reports_statistics():
    """
    GET /api/reports/statistics
    Returns total count, count by category, count by severity, and latest report time.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Total and latest report
            cur.execute("""
                SELECT 
                    COUNT(*),
                    MAX(created_at)
                FROM community_reports;
            """)
            total, latest = cur.fetchone()
            
            # Count by category
            cur.execute("""
                SELECT category, COUNT(*) 
                FROM community_reports 
                GROUP BY category;
            """)
            cat_rows = cur.fetchall()
            category_counts = {cat: 0 for cat in VALID_CATEGORIES}
            for cat, count in cat_rows:
                if cat in category_counts:
                    category_counts[cat] = int(count)
                    
            # Count by severity
            cur.execute("""
                SELECT severity, COUNT(*) 
                FROM community_reports 
                GROUP BY severity 
                ORDER BY severity;
            """)
            sev_rows = cur.fetchall()
            severity_counts = {i: 0 for i in range(1, 6)}
            for sev, count in sev_rows:
                if sev is not None and int(sev) in severity_counts:
                    severity_counts[int(sev)] = int(count)
                    
        return jsonify({
            "total_reports":          int(total) if total else 0,
            "latest_report_timestamp": latest.isoformat() if latest else None,
            "reports_by_category":     category_counts,
            "reports_by_severity":     severity_counts
        }), 200
        
    except Exception as exc:
        logger.exception("Failed to query report statistics: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# POST /api/reports — Create report
# ---------------------------------------------------------------------------

@reports_bp.post("/reports")
def create_report():
    """
    POST /api/reports
    Submits a new report. Validates input properties, maps lat/lon into PostGIS Point,
    and defaults credibility and status metadata fields.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400
        
    # Required parameters check
    required = ["latitude", "longitude", "category", "severity", "description"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required parameters: {missing}"}), 400
        
    # Parse coordinates
    try:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
    except (TypeError, ValueError):
        return jsonify({"error": "'latitude' and 'longitude' must be valid floats."}), 400
        
    # Validate severity
    try:
        severity = int(data["severity"])
        if not (1 <= severity <= 5):
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({"error": "'severity' must be an integer between 1 and 5."}), 400
        
    # Validate category
    category = str(data["category"]).strip().lower()
    if category not in VALID_CATEGORIES:
        return jsonify({"error": f"Invalid category. Must be one of: {list(VALID_CATEGORIES)}"}), 400
        
    description = str(data["description"]).strip()
    if not description:
        return jsonify({"error": "'description' cannot be empty."}), 400
        
    report_id = str(uuid.uuid4())
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO community_reports (
                    report_id,
                    category,
                    severity,
                    description,
                    geom,
                    initial_credibility,
                    current_credibility,
                    status,
                    created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                    1.0, 1.0, 'PENDING', NOW()
                );
                """,
                (report_id, category, severity, description, lon, lat)
            )
        conn.commit()
        
        logger.info("Successfully created community report with ID: %s", report_id)
        return jsonify({
            "message":   "Report created successfully.",
            "report_id": report_id,
        }), 201
        
    except Exception as exc:
        conn.rollback()
        logger.exception("Failed to insert community report: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        conn.close()
