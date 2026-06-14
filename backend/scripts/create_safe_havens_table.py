"""
SafeHer AI — Milestone 3
backend/scripts/create_safe_havens_table.py

Idempotent migration script: creates the safe_havens table and its indexes.
Safe to run multiple times (uses IF NOT EXISTS throughout).

Usage:
    cd backend
    python scripts/create_safe_havens_table.py
"""

import sys
import os
import logging

# Make backend/ importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from core.routing_service import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS safe_havens (
    id          SERIAL PRIMARY KEY,
    osm_id      BIGINT  NOT NULL,
    name        TEXT,
    category    TEXT    NOT NULL,
    latitude    FLOAT   NOT NULL,
    longitude   FLOAT   NOT NULL,
    geom        GEOMETRY(Point, 4326) NOT NULL,
    CONSTRAINT safe_havens_osm_id_unique UNIQUE (osm_id)
);
"""

CREATE_GIST_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS safe_havens_geom_gist
    ON safe_havens USING GIST (geom);
"""


def run_migration() -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            logger.info("Creating safe_havens table (IF NOT EXISTS)…")
            cur.execute(CREATE_TABLE_SQL)

            logger.info("Creating GIST index on safe_havens.geom (IF NOT EXISTS)…")
            cur.execute(CREATE_GIST_INDEX_SQL)

        conn.commit()
        logger.info("Migration complete — safe_havens table is ready.")
    except Exception as exc:
        conn.rollback()
        logger.error("Migration failed: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_migration()
