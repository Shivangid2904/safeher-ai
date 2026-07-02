"""
SafeHer AI — Milestone 4
backend/scripts/generate_sri.py

Bulk-generate the SafeHer Risk Index (SRI) for every road edge and
persist the results into the existing edge_risk_profiles table.

Data sources:
  • road_edges.highway_type   — from populate_road_metadata.py
  • road_edges.is_lit         — from populate_road_metadata.py
  • road_edges.length         — from Milestone 1 import
  • safe_havens (nearest)     — computed via PostGIS ST_Distance

The script uses:
  ON CONFLICT (edge_u, edge_v, edge_key) DO UPDATE
so it is safe to run multiple times.

Usage:
    cd backend
    python scripts/generate_sri.py
"""

import sys
import os
import json
import logging
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from core.routing_service import get_db_connection
from core.sri_engine import calculate_edge_risk_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Maximum distance (metres) to search for a nearby safe haven
SAFE_HAVEN_SEARCH_RADIUS_M = 1000


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

EDGE_QUERY = """
SELECT DISTINCT ON (u, v, key)
    u,
    v,
    key,
    length,
    highway_type,
    is_lit,
    geom
FROM road_edges
ORDER BY u, v, key, length ASC;
"""

NEAREST_SAFE_HAVEN_QUERY = """
SELECT
    ST_Distance(
        re_geom::geography,
        sh.geom::geography
    ) AS distance_m
FROM safe_havens sh
WHERE ST_DWithin(
    %s::geography,
    sh.geom::geography,
    %s
)
ORDER BY re_geom::geography <-> sh.geom::geography
LIMIT 1;
"""

UPSERT_QUERY = """
INSERT INTO edge_risk_profiles
    (edge_u, edge_v, edge_key,
     sri_score, confidence_score, risk_category,
     risk_attributions, last_calculated_at)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, NOW())
ON CONFLICT (edge_u, edge_v, edge_key) DO UPDATE SET
    sri_score          = EXCLUDED.sri_score,
    confidence_score   = EXCLUDED.confidence_score,
    risk_category      = EXCLUDED.risk_category,
    risk_attributions  = EXCLUDED.risk_attributions,
    last_calculated_at = EXCLUDED.last_calculated_at;
"""


def _find_nearest_safe_haven_distance(
    cur,
    geom_wkb_hex: str,
    radius_m: float,
) -> float | None:
    """
    Return the geodesic distance (metres) to the nearest Safe Haven,
    or None if no Safe Haven exists within radius_m.
    """
    cur.execute(
        """
        SELECT ST_Distance(
                   %s::geography,
                   sh.geom::geography
               ) AS distance_m
        FROM safe_havens sh
        WHERE ST_DWithin(
            %s::geography,
            sh.geom::geography,
            %s
        )
        ORDER BY %s::geography <-> sh.geom::geography
        LIMIT 1;
        """,
        (geom_wkb_hex, geom_wkb_hex, radius_m, geom_wkb_hex),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    start = time.perf_counter()

    conn = get_db_connection()
    try:
        # ── Fetch all edges ────────────────────────────────────────────────
        with conn.cursor() as cur:
            cur.execute(EDGE_QUERY)
            edges = cur.fetchall()

        if not edges:
            logger.error("No edges found in road_edges.  Run init_db.py first.")
            return

        logger.info("Processing %d edges…", len(edges))

        scores: list[float] = []
        batch: list[tuple] = []

        # ── Process each edge ──────────────────────────────────────────────
        with conn.cursor() as cur:
            for (u, v, key, length, highway_type, is_lit, geom_hex) in edges:
                # Find nearest Safe Haven using PostGIS
                dist_m = _find_nearest_safe_haven_distance(
                    cur, geom_hex, SAFE_HAVEN_SEARCH_RADIUS_M
                )

                # Calculate the SRI profile
                profile = calculate_edge_risk_profile(
                    road_type=highway_type,
                    is_lit=bool(is_lit) if is_lit is not None else False,
                    road_length=float(length) if length is not None else None,
                    distance_to_safe_haven=dist_m,
                )

                scores.append(profile["sri_score"])
                batch.append((
                    int(u),
                    int(v),
                    int(key),
                    profile["sri_score"],
                    profile["confidence_score"],
                    profile["risk_category"],
                    json.dumps(profile["risk_attributions"]),
                ))

        # ── Bulk upsert ────────────────────────────────────────────────────
        with conn.cursor() as cur:
            cur.executemany(UPSERT_QUERY, batch)
        conn.commit()

        elapsed = time.perf_counter() - start

        # ── Print statistics ───────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("SRI Generation Complete")
        logger.info("=" * 60)
        logger.info("  Edges processed : %d",    len(scores))
        logger.info("  Average SRI     : %.2f",  sum(scores) / len(scores))
        logger.info("  Minimum SRI     : %.2f",  min(scores))
        logger.info("  Maximum SRI     : %.2f",  max(scores))
        logger.info("  Execution time  : %.2fs", elapsed)
        logger.info("=" * 60)

    except Exception as exc:
        conn.rollback()
        logger.error("SRI generation failed: %s", exc, exc_info=True)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
