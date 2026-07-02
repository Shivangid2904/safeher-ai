"""
SafeHer AI — Milestone 4
backend/scripts/test_sri.py

End-to-end validation script.  Verifies the entire SRI pipeline:

  1. road_edges metadata (highway_type, is_lit)
  2. edge_risk_profiles exists and is populated
  3. No NULL sri_score values
  4. Risk values are clamped in [0, 100]
  5. Routing works in all three modes
  6. Route response includes risk statistics
  7. /api/sri/statistics endpoint works
  8. /api/sri/edge endpoint works

Usage:
    cd backend
    python scripts/test_sri.py
"""

import sys
import os
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from core.routing_service import get_db_connection
from core.sri_engine import calculate_edge_risk_profile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:5000"

# A pair of coordinates known to produce a valid route in Vijayawada
START_LAT, START_LON = 16.5062, 80.6480
END_LAT,   END_LON   = 16.5100, 80.6550

PASS  = "[PASS]"
FAIL  = "[FAIL]"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check(condition: bool, message: str) -> None:
    if condition:
        print(f"  {PASS}  {message}")
    else:
        print(f"  {FAIL}  {message}")
        sys.exit(1)


def _api_get(path: str) -> dict:
    url = API_BASE + path
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def _api_post(path: str, payload: dict) -> dict:
    url  = API_BASE + path
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

def test_road_metadata(conn) -> None:
    print("\n[1] Road Metadata Migration")
    with conn.cursor() as cur:
        # Column highway_type exists
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'road_edges'
              AND column_name = 'highway_type';
        """)
        _check(cur.fetchone() is not None, "highway_type column exists in road_edges")

        # Column is_lit exists
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'road_edges'
              AND column_name = 'is_lit';
        """)
        _check(cur.fetchone() is not None, "is_lit column exists in road_edges")

        # At least some edges have highway_type populated
        cur.execute("""
            SELECT COUNT(*) FROM road_edges
            WHERE highway_type IS NOT NULL;
        """)
        populated_count = cur.fetchone()[0]
        _check(populated_count > 0, f"highway_type populated in {populated_count} edges")


def test_edge_risk_profiles(conn) -> None:
    print("\n[2] edge_risk_profiles Table")
    with conn.cursor() as cur:
        # Table exists
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'edge_risk_profiles';
        """)
        _check(cur.fetchone() is not None, "edge_risk_profiles table exists")

        # Every edge has an SRI entry
        cur.execute("SELECT COUNT(*) FROM road_edges;")
        edge_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM edge_risk_profiles;")
        sri_count = cur.fetchone()[0]

        _check(sri_count > 0, f"edge_risk_profiles has {sri_count} rows")
        _check(
            sri_count >= edge_count * 0.95,
            f"SRI coverage >= 95%: {sri_count}/{edge_count} edges scored"
        )


def test_no_null_scores(conn) -> None:
    print("\n[3] NULL & Range Check")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM edge_risk_profiles
            WHERE sri_score IS NULL;
        """)
        null_count = cur.fetchone()[0]
        _check(null_count == 0, f"No NULL sri_score values (null count: {null_count})")

        cur.execute("""
            SELECT COUNT(*) FROM edge_risk_profiles
            WHERE sri_score < 0 OR sri_score > 100;
        """)
        out_of_range = cur.fetchone()[0]
        _check(out_of_range == 0, f"All SRI values in [0, 100] (out-of-range: {out_of_range})")

        cur.execute("""
            SELECT COUNT(*) FROM edge_risk_profiles
            WHERE confidence_score != 1.0;
        """)
        conf_wrong = cur.fetchone()[0]
        _check(conf_wrong == 0, f"All confidence_score values are 1.0 (wrong: {conf_wrong})")

        cur.execute("""
            SELECT COUNT(*) FROM edge_risk_profiles
            WHERE risk_category IS NULL
               OR risk_category NOT IN ('LOW','MODERATE','HIGH','CRITICAL');
        """)
        bad_cat = cur.fetchone()[0]
        _check(bad_cat == 0, f"All risk_category values are valid (invalid: {bad_cat})")


def test_sri_engine_unit() -> None:
    print("\n[4] SRI Engine Unit Tests")

    # Known safe road: primary, lit, short, near safe haven
    r = calculate_edge_risk_profile("primary", True, 40, 80)
    score = r["sri_score"]
    _check(0 <= score <= 100, f"Primary+lit+short+near_haven score in range: {score}")
    expected = max(0.0, min(100.0, 10 + 0 + 0 - 20))
    _check(abs(score - expected) < 0.01, f"Primary+lit+short+near_haven score={score} (expected {expected})")

    # Riskiest road: path, unlit, long, far from safe havens
    r2 = calculate_edge_risk_profile("path", False, 600, None)
    score2 = r2["sri_score"]
    _check(0 <= score2 <= 100, f"Path+unlit+long+no_haven score in range: {score2}")
    expected2 = min(100.0, 50 + 15 + 12 - 0)
    _check(abs(score2 - expected2) < 0.01, f"Path+unlit+long+no_haven score={score2} (expected {expected2})")

    # Categories
    _check(calculate_edge_risk_profile("motorway", True, 40, 50)["risk_category"] == "LOW",
           "Low risk category correct")
    _check(calculate_edge_risk_profile("residential", False, 200, 400)["risk_category"] in ("MODERATE","HIGH","CRITICAL"),
           "Higher risk category correct")


def test_statistics_api() -> None:
    print("\n[5] /api/sri/statistics")
    try:
        data = _api_get("/api/sri/statistics")
    except RuntimeError as exc:
        _check(False, f"API call failed: {exc}")
        return

    _check("edge_count"  in data, "Response has edge_count")
    _check("average_sri" in data, "Response has average_sri")
    _check("minimum_sri" in data, "Response has minimum_sri")
    _check("maximum_sri" in data, "Response has maximum_sri")
    _check(data["edge_count"] > 0, f"edge_count > 0: {data.get('edge_count')}")
    _check(
        0 <= data["minimum_sri"] <= data["maximum_sri"] <= 100,
        f"SRI range valid: min={data['minimum_sri']}, max={data['maximum_sri']}"
    )


def test_sri_edge_api(conn) -> None:
    print("\n[6] /api/sri/edge")
    # Get a real edge (u, v) from the database
    with conn.cursor() as cur:
        cur.execute("""
            SELECT edge_u, edge_v FROM edge_risk_profiles
            WHERE sri_score IS NOT NULL
            ORDER BY edge_u LIMIT 1;
        """)
        row = cur.fetchone()

    if row is None:
        _check(False, "No scored edges found — run generate_sri.py first")
        return

    u, v = row
    try:
        data = _api_get(f"/api/sri/edge?u={u}&v={v}")
    except RuntimeError as exc:
        _check(False, f"API call failed: {exc}")
        return

    _check("sri_score"         in data, "Response has sri_score")
    _check("confidence_score"  in data, "Response has confidence_score")
    _check("risk_category"     in data, "Response has risk_category")
    _check("risk_attributions" in data, "Response has risk_attributions")
    _check(data.get("confidence_score") == 1.0, "confidence_score is 1.0")


def test_routing_modes() -> None:
    print("\n[7] Routing Modes")

    for mode in ("fastest", "balanced", "safest"):
        try:
            data = _api_post("/api/route", {
                "start_lat": START_LAT,
                "start_lon": START_LON,
                "end_lat":   END_LAT,
                "end_lon":   END_LON,
                "mode":      mode,
            })
        except RuntimeError as exc:
            _check(False, f"Mode '{mode}' API call failed: {exc}")
            continue

        _check("distance_meters"    in data, f"Mode '{mode}': has distance_meters")
        _check("average_route_risk" in data, f"Mode '{mode}': has average_route_risk")
        _check("minimum_edge_risk"  in data, f"Mode '{mode}': has minimum_edge_risk")
        _check("maximum_edge_risk"  in data, f"Mode '{mode}': has maximum_edge_risk")
        _check("risk_category"      in data, f"Mode '{mode}': has risk_category")
        _check(
            0 <= data["average_route_risk"] <= 100,
            f"Mode '{mode}': average_route_risk in [0,100]: {data['average_route_risk']}"
        )


def test_backward_compatibility() -> None:
    print("\n[8] Backward Compatibility")

    # Existing endpoint must still work without 'mode'
    try:
        data = _api_post("/api/route", {
            "start_lat": START_LAT,
            "start_lon": START_LON,
            "end_lat":   END_LAT,
            "end_lon":   END_LON,
        })
    except RuntimeError as exc:
        _check(False, f"Route without mode failed: {exc}")
        return

    _check("distance_meters"    in data, "Route without mode: has distance_meters")
    _check("route_nodes"        in data, "Route without mode: has route_nodes")
    _check("geojson"            in data, "Route without mode: has geojson")
    _check("nearby_safe_havens" in data, "Route without mode: has nearby_safe_havens")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  SafeHer AI — Milestone 4 Validation")
    print("=" * 60)

    conn = get_db_connection()
    try:
        test_road_metadata(conn)
        test_edge_risk_profiles(conn)
        test_no_null_scores(conn)
        test_sri_engine_unit()
        test_statistics_api()
        test_sri_edge_api(conn)
        test_routing_modes()
        test_backward_compatibility()
    finally:
        conn.close()

    print()
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
