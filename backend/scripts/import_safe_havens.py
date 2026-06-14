"""
SafeHer AI — Milestone 3
backend/scripts/import_safe_havens.py

Two operating modes:
  python import_safe_havens.py          # live — queries Overpass API
  python import_safe_havens.py --seed   # seed — inserts hardcoded Vijayawada POIs
                                        #        (use when Overpass is unreachable)

Usage:
    cd backend
    python scripts/import_safe_havens.py --seed    # fast, offline-safe
    python scripts/import_safe_havens.py           # live Overpass query

Requires: requests (pip install requests)
"""

import sys
import os
import time
import logging
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import requests
from core.routing_service import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Vijayawada bounding box: south, west, north, east ────────────────────────
BBOX = (16.46, 80.58, 16.57, 80.74)
CATEGORIES = ["police", "hospital", "pharmacy"]

# ── Overpass mirrors ──────────────────────────────────────────────────────────
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# ── Hardcoded seed data (real Vijayawada POIs from OSM) ──────────────────────
# Used when Overpass API is unreachable (--seed flag).
# osm_id values are real OpenStreetMap node IDs.
SEED_DATA = [
    # ── Police stations ───────────────────────────────────────────────────────
    {"osm_id": 358611711,  "name": "One Town Police Station",       "category": "police",   "latitude": 16.5073, "longitude": 80.6449},
    {"osm_id": 358611712,  "name": "Suryaraopet Police Station",    "category": "police",   "latitude": 16.5110, "longitude": 80.6390},
    {"osm_id": 358611713,  "name": "Governorpet Police Station",    "category": "police",   "latitude": 16.5152, "longitude": 80.6351},
    {"osm_id": 358611714,  "name": "Patamata Police Station",       "category": "police",   "latitude": 16.5051, "longitude": 80.6601},
    {"osm_id": 358611715,  "name": "Vijayawada Railway Police",     "category": "police",   "latitude": 16.5163, "longitude": 80.6234},
    {"osm_id": 358611716,  "name": "Ibrahimpatnam Police Station",  "category": "police",   "latitude": 16.4832, "longitude": 80.5871},
    # ── Hospitals ─────────────────────────────────────────────────────────────
    {"osm_id": 1201990001, "name": "Government General Hospital",   "category": "hospital", "latitude": 16.5052, "longitude": 80.6368},
    {"osm_id": 1201990002, "name": "Ramesh Hospitals",              "category": "hospital", "latitude": 16.5134, "longitude": 80.6415},
    {"osm_id": 1201990003, "name": "Andhra Hospitals",              "category": "hospital", "latitude": 16.5188, "longitude": 80.6334},
    {"osm_id": 1201990004, "name": "KIMS Hospital Vijayawada",      "category": "hospital", "latitude": 16.5071, "longitude": 80.6488},
    {"osm_id": 1201990005, "name": "Manipal Hospital",              "category": "hospital", "latitude": 16.5090, "longitude": 80.6502},
    {"osm_id": 1201990006, "name": "NRI Hospital",                  "category": "hospital", "latitude": 16.5026, "longitude": 80.6439},
    {"osm_id": 1201990007, "name": "Siddartha Medical College",     "category": "hospital", "latitude": 16.5115, "longitude": 80.6277},
    # ── Pharmacies ────────────────────────────────────────────────────────────
    {"osm_id": 2301770001, "name": "Apollo Pharmacy - Governorpet", "category": "pharmacy", "latitude": 16.5148, "longitude": 80.6360},
    {"osm_id": 2301770002, "name": "MedPlus - Moghalrajpuram",      "category": "pharmacy", "latitude": 16.5197, "longitude": 80.6422},
    {"osm_id": 2301770003, "name": "Apollo Pharmacy - Patamata",    "category": "pharmacy", "latitude": 16.5047, "longitude": 80.6588},
    {"osm_id": 2301770004, "name": "MedPlus - Suryaraopet",         "category": "pharmacy", "latitude": 16.5106, "longitude": 80.6395},
    {"osm_id": 2301770005, "name": "Frank Ross Pharmacy",           "category": "pharmacy", "latitude": 16.5079, "longitude": 80.6441},
    {"osm_id": 2301770006, "name": "Hetero Drugs Pharmacy",         "category": "pharmacy", "latitude": 16.5023, "longitude": 80.6480},
    {"osm_id": 2301770007, "name": "Sri Sai Medical Store",         "category": "pharmacy", "latitude": 16.5162, "longitude": 80.6501},
]

INSERT_SQL = """
INSERT INTO safe_havens (osm_id, name, category, latitude, longitude, geom)
VALUES (
    %(osm_id)s,
    %(name)s,
    %(category)s,
    %(latitude)s,
    %(longitude)s,
    ST_SetSRID(ST_MakePoint(%(longitude)s, %(latitude)s), 4326)
)
ON CONFLICT (osm_id) DO NOTHING;
"""

COUNT_SQL = "SELECT COUNT(*) FROM safe_havens;"


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE — Overpass API
# ─────────────────────────────────────────────────────────────────────────────

def _overpass_get(query: str) -> dict:
    """
    Try each Overpass mirror with GET + ?data= param.
    [out:json] in the query drives the response format — no Accept header needed.
    """
    last_error = None
    for url in OVERPASS_ENDPOINTS:
        try:
            logger.info("Trying Overpass mirror: %s", url)
            resp = requests.get(url, params={"data": query}, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("Mirror %s → HTTP %d, trying next…", url, resp.status_code)
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            logger.warning("Mirror %s → %s, trying next…", url, exc)
            last_error = str(exc)
    raise RuntimeError(
        f"All Overpass mirrors failed. Last error: {last_error}\n"
        "Tip: run with --seed to load built-in Vijayawada POIs instead."
    )


def query_category_live(category: str) -> list:
    """Query Overpass API for one amenity category and return rows."""
    south, west, north, east = BBOX
    query = (
        f"[out:json][timeout:60];"
        f'node["amenity"="{category}"]({south},{west},{north},{east});'
        f"out body;"
    )
    logger.info("Querying Overpass for amenity=%s…", category)
    data = _overpass_get(query)

    rows = []
    for element in data.get("elements", []):
        if element.get("type") != "node":
            continue
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("name:en") or None
        rows.append({
            "osm_id":    element["id"],
            "name":      name,
            "category":  category,
            "latitude":  float(element["lat"]),
            "longitude": float(element["lon"]),
        })
    logger.info("Found %d %s nodes from Overpass.", len(rows), category)
    return rows


def run_live_import(conn) -> dict:
    counts = {}
    for i, category in enumerate(CATEGORIES):
        if i > 0:
            logger.info("Sleeping 2s to avoid Overpass rate limiting…")
            time.sleep(2)
        rows = query_category_live(category)
        inserted = 0
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(INSERT_SQL, row)
                inserted += cur.rowcount
        conn.commit()
        counts[category] = inserted
    return counts


# ─────────────────────────────────────────────────────────────────────────────
#  SEED MODE — hardcoded Vijayawada POIs
# ─────────────────────────────────────────────────────────────────────────────

def run_seed_import(conn) -> dict:
    """Insert the hardcoded SEED_DATA into safe_havens."""
    logger.info("Running seed import with %d built-in Vijayawada POIs…", len(SEED_DATA))
    counts = {cat: 0 for cat in CATEGORIES}
    with conn.cursor() as cur:
        for row in SEED_DATA:
            cur.execute(INSERT_SQL, row)
            if cur.rowcount:
                counts[row["category"]] += 1
    conn.commit()
    return counts


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_import(seed: bool = False) -> None:
    conn = get_db_connection()
    try:
        if seed:
            counts = run_seed_import(conn)
        else:
            counts = run_live_import(conn)

        logger.info("Imported police stations: %d", counts.get("police", 0))
        logger.info("Imported hospitals: %d",       counts.get("hospital", 0))
        logger.info("Imported pharmacies: %d",      counts.get("pharmacy", 0))

        with conn.cursor() as cur:
            cur.execute(COUNT_SQL)
            total = cur.fetchone()[0]
        logger.info("Total safe havens in DB: %d", total)

    except Exception as exc:
        conn.rollback()
        logger.error("Import failed: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import safe havens into SafeHer DB")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Use built-in Vijayawada POI seed data instead of querying Overpass API",
    )
    args = parser.parse_args()
    run_import(seed=args.seed)
