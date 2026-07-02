"""
SafeHer AI — Milestone 5
backend/core/incident_engine.py

Core engine to calculate dynamic incident-based risk penalties on road segments.
Calculates penalties dynamically at runtime to keep graph cache immutable.

Scoring Rules
-------------
Distance:
  • <= 50m      : +20
  • <= 100m     : +12
  • <= 200m     : +5
  • otherwise   : 0

Time Decay:
  • 0–24 hours  : 100% (x1.0)
  • 1–3 days    : 70%  (x0.7)
  • 3–7 days    : 40%  (x0.4)
  • > 7 days    : 0%   (x0)

Clamping:
  • Total penalty per edge is clamped to [0.0, 20.0].
"""

from __future__ import annotations

import logging
import psycopg2

logger = logging.getLogger(__name__)


def calculate_decayed_penalty(distance_meters: float, age_hours: float) -> float:
    """
    Calculate the decayed penalty for a report at a given distance and age.

    Parameters
    ----------
    distance_meters : float
        Geodesic distance between the road edge and the report.
    age_hours : float
        Time elapsed since the report was submitted.

    Returns
    -------
    float
        The calculated penalty value.
    """
    # ── Distance penalty ──────────────────────────────────────────────────
    if distance_meters <= 50.0:
        base_penalty = 20.0
    elif distance_meters <= 100.0:
        base_penalty = 12.0
    elif distance_meters <= 200.0:
        base_penalty = 5.0
    else:
        return 0.0

    # ── Time decay multiplier ─────────────────────────────────────────────
    if age_hours <= 24.0:
        multiplier = 1.0
    elif age_hours <= 72.0:  # 3 days
        multiplier = 0.7
    elif age_hours <= 168.0:  # 7 days
        multiplier = 0.4
    else:
        return 0.0

    return base_penalty * multiplier


def get_active_incident_penalties(conn: psycopg2.extensions.connection) -> dict[tuple[int, int], float]:
    """
    Fetch all active incident reports within the last 7 days and compute
    incident penalties for all road edges within 200m.

    Uses a single PostGIS query with ST_DWithin and ST_Distance to find
    all edge-incident pairs, then calculates decayed penalties in Python
    and aggregates them by (u, v) edge identifiers.

    Parameters
    ----------
    conn : psycopg2 connection

    Returns
    -------
    dict
        A map of (u, v) -> incident_penalty, where penalty is clamped to [0.0, 20.0].
    """
    query = """
        SELECT
            re.u,
            re.v,
            ST_Distance(re.geom::geography, r.geom::geography) AS distance_meters,
            EXTRACT(EPOCH FROM (NOW() - r.created_at)) / 3600.0 AS age_hours
        FROM road_edges re
        JOIN community_reports r
            ON ST_DWithin(re.geom::geography, r.geom::geography, 200)
        WHERE r.created_at >= NOW() - INTERVAL '7 days';
    """

    penalties: dict[tuple[int, int], float] = {}

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

        for u, v, dist, age in rows:
            penalty = calculate_decayed_penalty(float(dist), float(age))
            edge_key = (int(u), int(v))
            penalties[edge_key] = penalties.get(edge_key, 0.0) + penalty

        # Clamp all penalties to [0.0, 20.0]
        for edge_key in penalties:
            penalties[edge_key] = max(0.0, min(20.0, penalties[edge_key]))

        logger.info("Computed dynamic penalties for %d active edges.", len(penalties))
        return penalties

    except Exception as exc:
        logger.error("Failed to query active incident penalties: %s", exc)
        return {}


def calculate_incident_penalty(edge_geom_hex_or_text: str | bytes, conn: psycopg2.extensions.connection) -> float:
    """
    Compute the aggregated incident penalty for a single edge geometry.
    Mainly used for verification/testing.

    Parameters
    ----------
    edge_geom_hex_or_text : str or bytes
        The geometry representing the road edge (WKT string or WKB hex).
    conn : psycopg2 connection

    Returns
    -------
    float
        The calculated penalty clamped to [0.0, 20.0].
    """
    # Detect WKB vs WKT to form the correct SQL casts
    is_wkb = isinstance(edge_geom_hex_or_text, (bytes, memoryview)) or (
        isinstance(edge_geom_hex_or_text, str) and
        all(c in '0123456789abcdefABCDEF' for c in edge_geom_hex_or_text)
    )

    geom_cast = "%s::geometry" if is_wkb else "ST_GeomFromText(%s, 4326)"

    query = f"""
        SELECT
            ST_Distance({geom_cast}::geography, r.geom::geography) AS distance_meters,
            EXTRACT(EPOCH FROM (NOW() - r.created_at)) / 3600.0 AS age_hours
        FROM community_reports r
        WHERE r.created_at >= NOW() - INTERVAL '7 days'
          AND ST_DWithin({geom_cast}::geography, r.geom::geography, 200);
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query, (edge_geom_hex_or_text, edge_geom_hex_or_text))
            rows = cur.fetchall()

        total_penalty = 0.0
        for dist, age in rows:
            total_penalty += calculate_decayed_penalty(float(dist), float(age))

        return max(0.0, min(20.0, total_penalty))

    except Exception as exc:
        logger.error("Failed to compute single incident penalty: %s", exc)
        return 0.0
