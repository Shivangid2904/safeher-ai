"""
SafeHer AI — Milestone 4
backend/scripts/populate_road_metadata.py

One-time idempotent migration that extends road_edges with:
  • highway_type  VARCHAR(50)  — OSM highway tag
  • is_lit        BOOLEAN      — True when OSM lit == 'yes'

The script NEVER downloads new OSM data.  It reads the cached
OSMnx graph that was already stored on disk during Milestone 1.

Usage:
    cd backend
    python scripts/populate_road_metadata.py
"""

import sys
import os
import logging
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import osmnx as ox
from core.routing_service import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OSMnx graph parameters — must match Milestone 1 exactly
# ---------------------------------------------------------------------------
GRAPH_CENTER   = (16.5062, 80.6480)   # Vijayawada
GRAPH_DIST_M   = 1000
NETWORK_TYPE   = "drive"


def _add_columns_if_missing(conn) -> None:
    """Add highway_type and is_lit columns to road_edges if they do not exist."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE road_edges
                ADD COLUMN IF NOT EXISTS highway_type VARCHAR(50),
                ADD COLUMN IF NOT EXISTS is_lit BOOLEAN DEFAULT FALSE;
        """)
    conn.commit()
    logger.info("Columns highway_type / is_lit ensured on road_edges.")


def _load_cached_graph() -> dict:
    """
    Load the OSMnx graph from the local cache.

    Returns a dict keyed by (u, v, key) → {"highway": str, "lit": bool}

    OSMnx uses a requests-cache automatically.  We force cache-only by
    temporarily setting a very large expiry so the disk cache is always used.
    """
    logger.info("Loading OSMnx graph from local cache (no network download)…")

    # Ensure ox uses its cache folder and never hits the network
    ox.settings.use_cache = True
    ox.settings.log_console = False

    try:
        G = ox.graph_from_point(
            GRAPH_CENTER,
            dist=GRAPH_DIST_M,
            network_type=NETWORK_TYPE,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load OSMnx graph. Make sure the local cache exists. Error: {exc}"
        ) from exc

    edge_meta: dict = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        highway_raw = data.get("highway", None)
        if isinstance(highway_raw, list):
            highway_raw = highway_raw[0]
        highway_type = str(highway_raw).strip() if highway_raw else None

        lit_raw = data.get("lit", "no")
        if isinstance(lit_raw, list):
            lit_raw = lit_raw[0]
        is_lit = str(lit_raw).strip().lower() == "yes"

        edge_meta[(int(u), int(v), int(key))] = {
            "highway_type": highway_type,
            "is_lit": is_lit,
        }

    logger.info("Graph loaded — %d edges found in cache.", len(edge_meta))
    return edge_meta


def _count_unpopulated(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM road_edges
            WHERE highway_type IS NULL;
        """)
        return cur.fetchone()[0]


def _populate_metadata(conn, edge_meta: dict) -> int:
    """
    Update road_edges rows whose highway_type is NULL.
    Matches on (u, v, key).

    Returns the number of rows updated.
    """
    updated = 0
    rows = []

    with conn.cursor() as cur:
        cur.execute("""
            SELECT u, v, key FROM road_edges
            WHERE highway_type IS NULL;
        """)
        rows = cur.fetchall()

    if not rows:
        logger.info("All rows are already populated — nothing to do.")
        return 0

    logger.info("%d road_edges rows need populating…", len(rows))

    batch = []
    for (u, v, key) in rows:
        meta = edge_meta.get((int(u), int(v), int(key)))
        if meta is None:
            # Fallback: mark as unknown / unlit so the row is no longer NULL
            meta = {"highway_type": "unknown", "is_lit": False}
        batch.append((meta["highway_type"], meta["is_lit"], int(u), int(v), int(key)))

    with conn.cursor() as cur:
        cur.executemany("""
            UPDATE road_edges
            SET    highway_type = %s,
                   is_lit       = %s
            WHERE  u   = %s
              AND  v   = %s
              AND  key = %s
              AND  highway_type IS NULL;
        """, batch)
        updated = cur.rowcount

    conn.commit()
    return updated


def run() -> None:
    start = time.perf_counter()

    conn = get_db_connection()
    try:
        # Step 1 — ensure columns exist
        _add_columns_if_missing(conn)

        # Step 2 — quick exit if already done
        unpopulated = _count_unpopulated(conn)
        if unpopulated == 0:
            logger.info("road_edges is already fully populated — no work needed.")
            return

        logger.info("%d rows still need highway_type / is_lit.", unpopulated)

        # Step 3 — load cached graph metadata
        edge_meta = _load_cached_graph()

        # Step 4 — populate
        updated = _populate_metadata(conn, edge_meta)
        elapsed = time.perf_counter() - start
        logger.info(
            "Migration complete: %d rows updated in %.2fs.", updated, elapsed
        )

    finally:
        conn.close()


if __name__ == "__main__":
    run()
