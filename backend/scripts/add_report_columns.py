"""
SafeHer AI — Milestone 5
backend/scripts/add_report_columns.py

Database migration script to extend the existing community_reports table.
Adds the following columns using raw SQL (psycopg2) if they are missing:
  • severity: INTEGER (risk severity rating, 1 to 5)
  • description: TEXT (text description of the incident)
  • created_at: TIMESTAMP WITH TIME ZONE (incident report time)

Does NOT create latitude/longitude columns; coordinates are derived from PostGIS geom.
Script is idempotent and safe to run multiple times.

Usage:
    cd backend
    python scripts/add_report_columns.py
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


def migrate() -> None:
    """Run the database migration to add severity, description, and created_at columns."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            logger.info("Inspecting columns of community_reports table...")
            
            # Add severity column
            cur.execute("""
                ALTER TABLE community_reports 
                ADD COLUMN IF NOT EXISTS severity INTEGER;
            """)
            
            # Add description column
            cur.execute("""
                ALTER TABLE community_reports 
                ADD COLUMN IF NOT EXISTS description TEXT;
            """)
            
            # Add created_at column
            cur.execute("""
                ALTER TABLE community_reports 
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
            """)
            
            # Re-create spatial index on geom point just in case
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_community_reports_geom_gist 
                ON community_reports USING GIST (geom);
            """)
            
        conn.commit()
        logger.info("Database migration completed successfully! community_reports table extended.")
        
    except Exception as exc:
        conn.rollback()
        logger.exception("Migration failed: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
