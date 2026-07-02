"""
SafeHer AI — Milestone 4
backend/scripts/create_edge_risk_table.py

Ensures the edge_risk_profiles table exists with the correct schema and indexes.

The existing edge_risk_profiles table from models.py has:
  edge_u BIGINT, edge_v BIGINT, edge_key INTEGER,
  sri_score DOUBLE PRECISION, confidence_score DOUBLE PRECISION,
  risk_category VARCHAR, risk_attributions JSONB,
  last_calculated_at TIMESTAMP
  PRIMARY KEY (edge_u, edge_v, edge_key)

This script creates the table (IF NOT EXISTS) and any missing indexes.
It NEVER drops or recreates the table.

Usage:
    cd backend
    python scripts/create_edge_risk_table.py
"""

import sys
import os
import logging

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
CREATE TABLE IF NOT EXISTS edge_risk_profiles (
    edge_u             BIGINT            NOT NULL,
    edge_v             BIGINT            NOT NULL,
    edge_key           INTEGER           NOT NULL DEFAULT 0,
    sri_score          DOUBLE PRECISION,
    confidence_score   DOUBLE PRECISION,
    risk_category      VARCHAR(20),
    risk_attributions  JSONB,
    last_calculated_at TIMESTAMP,
    PRIMARY KEY (edge_u, edge_v, edge_key)
);
"""

CREATE_INDEX_SRI = """
CREATE INDEX IF NOT EXISTS idx_edge_risk_sri_score
    ON edge_risk_profiles (sri_score);
"""

CREATE_INDEX_CATEGORY = """
CREATE INDEX IF NOT EXISTS idx_edge_risk_category
    ON edge_risk_profiles (risk_category);
"""


def run() -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            logger.info("Creating edge_risk_profiles table (IF NOT EXISTS)…")
            cur.execute(CREATE_TABLE_SQL)

            logger.info("Creating index on sri_score (IF NOT EXISTS)…")
            cur.execute(CREATE_INDEX_SRI)

            logger.info("Creating index on risk_category (IF NOT EXISTS)…")
            cur.execute(CREATE_INDEX_CATEGORY)

        conn.commit()
        logger.info("edge_risk_profiles table and indexes are ready.")

    except Exception as exc:
        conn.rollback()
        logger.error("Migration failed: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
