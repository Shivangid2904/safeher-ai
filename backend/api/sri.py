"""
SafeHer AI — Milestone 4
backend/api/sri.py

Flask Blueprint: sri_bp
Prefix: /api/sri

Endpoints
---------
GET /api/sri/health
    Liveness check for the risk engine service.

GET /api/sri/statistics
    Aggregate statistics (count, avg, min, max) over edge_risk_profiles.

GET /api/sri/edge?u=<int>&v=<int>
    Full SRI profile for a specific road edge (u, v).
    Returns the first matching row (lowest edge_key) when multiple
    parallel edges exist for the same node pair.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from core.routing_service import get_db_connection

logger = logging.getLogger(__name__)

sri_bp = Blueprint("sri_bp", __name__, url_prefix="/api/sri")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@sri_bp.get("/health")
def health():
    """GET /api/sri/health — SRI engine liveness."""
    return jsonify({
        "status":    "ok",
        "service":   "SafeHer Risk Engine",
        "milestone": 4,
    }), 200


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@sri_bp.get("/statistics")
def statistics():
    """
    GET /api/sri/statistics

    Returns aggregate SRI statistics from edge_risk_profiles.

    Response 200:
        {
            "edge_count":  int,
            "average_sri": float,
            "minimum_sri": float,
            "maximum_sri": float,
            "category_distribution": {
                "LOW":      int,
                "MODERATE": int,
                "HIGH":     int,
                "CRITICAL": int
            }
        }
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # ── Aggregate stats ───────────────────────────────────────────
            cur.execute("""
                SELECT
                    COUNT(*)             AS edge_count,
                    AVG(sri_score)       AS avg_sri,
                    MIN(sri_score)       AS min_sri,
                    MAX(sri_score)       AS max_sri
                FROM edge_risk_profiles
                WHERE sri_score IS NOT NULL;
            """)
            row = cur.fetchone()
            if row is None or row[0] == 0:
                return jsonify({
                    "error": "No SRI data found. Run generate_sri.py first."
                }), 404

            edge_count, avg_sri, min_sri, max_sri = row

            # ── Category distribution ─────────────────────────────────────
            cur.execute("""
                SELECT risk_category, COUNT(*) AS cnt
                FROM edge_risk_profiles
                WHERE sri_score IS NOT NULL
                GROUP BY risk_category;
            """)
            category_rows = cur.fetchall()
            distribution = {cat: 0 for cat in ("LOW", "MODERATE", "HIGH", "CRITICAL")}
            for cat, cnt in category_rows:
                if cat in distribution:
                    distribution[cat] = int(cnt)

        return jsonify({
            "edge_count":           int(edge_count),
            "average_sri":          round(float(avg_sri),  2),
            "minimum_sri":          round(float(min_sri),  2),
            "maximum_sri":          round(float(max_sri),  2),
            "category_distribution": distribution,
        }), 200

    except Exception as exc:
        logger.exception("Error fetching SRI statistics: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Edge detail
# ---------------------------------------------------------------------------

@sri_bp.get("/edge")
def edge_profile():
    """
    GET /api/sri/edge?u=<int>&v=<int>

    Returns the full SRI breakdown for the road edge identified by
    the (u, v) node pair.  When multiple parallel edges share the same
    (u, v), the one with the lowest edge_key is returned.

    Response 200:
        {
            "edge_u":            int,
            "edge_v":            int,
            "edge_key":          int,
            "sri_score":         float,
            "confidence_score":  float,
            "risk_category":     str,
            "risk_attributions": {
                "road_type_penalty":   float,
                "lighting_penalty":    float,
                "road_length_penalty": float,
                "safe_haven_bonus":    float
            },
            "last_calculated_at": str (ISO-8601)
        }

    Response 400: missing or non-integer parameters
    Response 404: no profile found for the given (u, v) pair
    """
    u_raw = request.args.get("u")
    v_raw = request.args.get("v")

    if u_raw is None or v_raw is None:
        return jsonify({"error": "Query parameters 'u' and 'v' are required."}), 400

    try:
        u = int(u_raw)
        v = int(v_raw)
    except ValueError:
        return jsonify({"error": "Parameters 'u' and 'v' must be integers."}), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    edge_u,
                    edge_v,
                    edge_key,
                    sri_score,
                    confidence_score,
                    risk_category,
                    risk_attributions,
                    last_calculated_at
                FROM edge_risk_profiles
                WHERE edge_u = %s AND edge_v = %s
                ORDER BY edge_key ASC
                LIMIT 1;
            """, (u, v))
            row = cur.fetchone()

        if row is None:
            return jsonify({
                "error": f"No SRI profile found for edge ({u}, {v})."
            }), 404

        edge_u, edge_v, edge_key, sri_score, confidence_score, \
            risk_category, risk_attributions, last_calculated_at = row

        return jsonify({
            "edge_u":             int(edge_u),
            "edge_v":             int(edge_v),
            "edge_key":           int(edge_key),
            "sri_score":          round(float(sri_score),         2) if sri_score         is not None else None,
            "confidence_score":   round(float(confidence_score),  2) if confidence_score  is not None else None,
            "risk_category":      risk_category,
            "risk_attributions":  risk_attributions,
            "last_calculated_at": last_calculated_at.isoformat() if last_calculated_at else None,
        }), 200

    except Exception as exc:
        logger.exception("Error fetching edge profile: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        conn.close()
