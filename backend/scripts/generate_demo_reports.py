"""
SafeHer AI — Milestone 5
backend/scripts/generate_demo_reports.py

Generates 200 realistic community incident reports clustered around:
  • Vijayawada Railway Station (Transit Center)
  • Pandit Nehru Bus Stand (Transit Center)
  • Besant Road (Commercial district)
  • Siddhartha Medical College (University district)

Timestamps are randomly distributed over the last 7 days.
Severity ranges from 1 to 5.
Coordinates are jittered and unique to prevent overlapping/duplicate entries.

Usage:
    cd backend
    python scripts/generate_demo_reports.py
"""

import sys
import os
import random
import uuid
import logging
from datetime import datetime, timedelta, timezone

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

# Bounding box centers (lat, lon) inside the Vijayawada road network area
CLUSTERS = {
    "railway_station": (16.5090, 80.6420),
    "bus_stand":       (16.5015, 80.6440),
    "commercial_area": (16.5050, 80.6465),
    "university_area": (16.5030, 80.6520),
}

CATEGORIES = [
    "harassment",
    "stalking",
    "poor_lighting",
    "unsafe_area",
    "suspicious_activity"
]

DESCRIPTIONS = {
    "harassment": [
        "Group of boys passing remarks at passersby near the corner shop.",
        "Catcalling reported near the tea stall during evening hours.",
        "Stray boys loitering and making inappropriate comments."
    ],
    "stalking": [
        "Unidentified individual following women walking home from the bus stop.",
        "A suspicious bike following pedestrians for several blocks.",
        "Suspicious man watching the crossroad corner for long hours."
    ],
    "poor_lighting": [
        "Streetlights completely broken along this lane.",
        "Pitch black corner near the abandoned building, unsafe for walking.",
        "Dim lighting makes the road hard to navigate in the evening."
    ],
    "unsafe_area": [
        "Unregulated auto stand with aggressive drivers loitering.",
        "High density of abandoned spots with poor visibility.",
        "Isolated shortcut lane where multiple safety issues are regularly felt."
    ],
    "suspicious_activity": [
        "Group of individuals drinking alcohol in public and blocking the footpath.",
        "Suspicious vehicle parked with no license plates near the residential corner.",
        "Strangers arguing and blocking passage on the side walkway."
    ]
}


def _clear_existing_reports(conn) -> None:
    """Clear all existing records from community_reports table."""
    with conn.cursor() as cur:
        logger.info("Clearing previous records in community_reports...")
        cur.execute("DELETE FROM community_reports;")
    conn.commit()


def run() -> None:
    conn = get_db_connection()
    try:
        _clear_existing_reports(conn)
        
        # Keep track of generated coordinates to guarantee uniqueness
        seen_coordinates = set()
        
        logger.info("Generating 200 clustered community reports...")
        
        now = datetime.now(timezone.utc)
        batch = []
        
        for i in range(200):
            # Pick a cluster randomly
            cluster_name, (center_lat, center_lon) = random.choice(list(CLUSTERS.items()))
            
            # Apply tiny random jitter (approx. within 150m of cluster center)
            while True:
                jitter_lat = center_lat + random.uniform(-0.0015, 0.0015)
                jitter_lon = center_lon + random.uniform(-0.0015, 0.0015)
                coord_key = (round(jitter_lat, 6), round(jitter_lon, 6))
                
                if coord_key not in seen_coordinates:
                    seen_coordinates.add(coord_key)
                    lat, lon = coord_key
                    break
                    
            category = random.choice(CATEGORIES)
            severity = random.choice([1, 2, 3, 4, 5])
            description = random.choice(DESCRIPTIONS[category])
            
            # Time decay distribution: mostly recent (0-3 days), some older (3-7 days)
            if random.random() < 0.6:
                # 0 - 3 days old
                age_delta = timedelta(
                    days=random.randint(0, 2),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
            else:
                # 3 - 7 days old
                age_delta = timedelta(
                    days=random.randint(3, 6),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
            report_time = now - age_delta
            report_id = str(uuid.uuid4())
            
            batch.append((
                report_id,
                category,
                severity,
                description,
                lon, lat,
                1.0, 1.0, 'PENDING',
                report_time
            ))
            
        with conn.cursor() as cur:
            cur.executemany(
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
                    %s, %s, %s,
                    %s
                );
                """,
                batch
            )
            
        conn.commit()
        logger.info("Demo reports generation complete! 200 reports generated and saved.")
        
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to generate demo reports: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
