"""
SafeHer AI â€” Milestone 5
backend/scripts/test_reports.py

End-to-end validation for the entire Milestone 5 pipeline.

Test suites
-----------
[1]  Database schema â€” community_reports has required columns
[2]  community_reports populated with demo data
[3]  Incident engine â€” calculate_decayed_penalty() unit tests
[4]  Incident engine â€” get_active_incident_penalties() returns dict
[5]  POST /api/reports â€” report creation
[6]  GET  /api/reports â€” list all reports
[7]  GET  /api/reports/nearby â€” spatial search
[8]  GET  /api/reports/statistics â€” aggregated stats
[9]  GET  /api/reports/health â€” liveness
[10] POST /api/route (fastest) â€” returns incident fields
[11] POST /api/route (balanced) â€” routing mode
[12] POST /api/route (safest)   â€” routing mode
[13] POST /api/route without mode â€” backward compat
[14] Milestone 4 backward compat â€” SRI fields preserved
[15] Safe Havens API still works

Usage:
    # Start the Flask server FIRST, then:
    cd backend
    python scripts/test_reports.py
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
from core.incident_engine import calculate_decayed_penalty, get_active_incident_penalties

API_BASE  = "http://127.0.0.1:5000"
START_LAT, START_LON = 16.5062, 80.6480
END_LAT,   END_LON   = 16.5100, 80.6550

PASS = "[PASS]"
FAIL = "[FAIL]"
_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    label = PASS if condition else FAIL
    print(f"  {label}  {message}")
    if not condition:
        _failures.append(message)


def api_get(path: str) -> dict:
    url = API_BASE + path
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}: {exc.read().decode()}") from exc


def api_post(path: str, payload: dict) -> tuple[int, dict]:
    url  = API_BASE + path
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [1] Database schema
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_db_schema(conn) -> None:
    print("\n[1] Database Schema")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'community_reports';
        """)
        cols = {row[0] for row in cur.fetchall()}

    check("severity"   in cols, "community_reports has 'severity' column")
    check("description" in cols, "community_reports has 'description' column")
    check("created_at"  in cols, "community_reports has 'created_at' column")
    check("geom"        in cols, "community_reports has 'geom' column")
    check("report_id"   in cols, "community_reports has 'report_id' column")
    check("category"    in cols, "community_reports has 'category' column")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [2] Demo data present
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_demo_data(conn) -> None:
    print("\n[2] Demo Data Present")
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM community_reports WHERE severity IS NOT NULL;")
        count = cur.fetchone()[0]
    check(count >= 200, f"community_reports contains >= 200 rows with severity: {count}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [3] Incident engine unit tests
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_incident_engine_unit() -> None:
    print("\n[3] Incident Engine Unit Tests")

    # Distance bands
    check(calculate_decayed_penalty(40.0, 0.0)   == 20.0, "<=50m + 0h â†’ 20.0")
    check(calculate_decayed_penalty(75.0, 0.0)   == 12.0, "<=100m + 0h â†’ 12.0")
    check(calculate_decayed_penalty(150.0, 0.0)  ==  5.0, "<=200m + 0h â†’ 5.0")
    check(calculate_decayed_penalty(250.0, 0.0)  ==  0.0, ">200m â†’ 0.0")

    # Time decay
    check(calculate_decayed_penalty(40.0, 12.0)  == 20.0, "0-24h â†’ 100%: 20.0")
    check(calculate_decayed_penalty(40.0, 48.0)  == 14.0, "1-3d â†’ 70%: 14.0")
    check(calculate_decayed_penalty(40.0, 120.0) ==  8.0, "3-7d â†’ 40%: 8.0")
    check(calculate_decayed_penalty(40.0, 200.0) ==  0.0, ">7d â†’ ignored: 0.0")

    # Combined: <=100m, 3-7 days
    p = calculate_decayed_penalty(80.0, 100.0)
    check(abs(p - 4.8) < 0.01, f"<=100m + 3-7d â†’ 12*0.4=4.8: {p}")

    # Clamping
    from core.incident_engine import get_active_incident_penalties
    # Clamping tested implicitly via aggregation logic; unit assert boundary
    check(calculate_decayed_penalty(40.0, 0.0) <= 20.0, "Single penalty <=20")
    check(calculate_decayed_penalty(40.0, 0.0) >= 0.0,  "Single penalty >=0")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [4] get_active_incident_penalties
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_incident_penalties_dict(conn) -> None:
    print("\n[4] Active Incident Penalties Dict")
    penalties = get_active_incident_penalties(conn)
    check(isinstance(penalties, dict), f"Returns dict: {type(penalties).__name__}")
    # With 200 demo reports there should be some edge penalties
    check(len(penalties) > 0, f"Penalties computed for {len(penalties)} edges")
    for edge_key, val in penalties.items():
        check(isinstance(edge_key, tuple) and len(edge_key) == 2,
              f"Key is 2-tuple: {edge_key}")
        check(0.0 <= val <= 20.0, f"Penalty clamped [0,20]: {val}")
        break  # just check first entry


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [5] POST /api/reports â€” create report
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_create_report() -> None:
    print("\n[5] POST /api/reports")
    try:
        status, data = api_post("/api/reports", {
            "latitude":    16.5045,
            "longitude":   80.6460,
            "category":    "harassment",
            "severity":    3,
            "description": "Test report generated by test_reports.py",
        })
    except RuntimeError as exc:
        check(False, f"API call failed: {exc}")
        return

    check(status == 201, f"Status 201 Created: {status}")
    check("report_id" in data, "Response has 'report_id'")
    check("message"   in data, "Response has 'message'")


def test_create_report_validation() -> None:
    print("\n[5b] POST /api/reports â€” input validation")
    # Missing field
    try:
        _, _ = api_post("/api/reports", {
            "latitude": 16.50, "longitude": 80.64,
            "category": "harassment", "severity": 3,
            # missing description
        })
        check(False, "Should have rejected missing description")
    except RuntimeError as exc:
        check("400" in str(exc), f"Rejected missing description with 400: {exc}")

    # Invalid severity
    try:
        _, _ = api_post("/api/reports", {
            "latitude": 16.50, "longitude": 80.64,
            "category": "harassment", "severity": 9,
            "description": "bad severity",
        })
        check(False, "Should have rejected severity=9")
    except RuntimeError as exc:
        check("400" in str(exc), f"Rejected severity=9 with 400: {exc}")

    # Invalid category
    try:
        _, _ = api_post("/api/reports", {
            "latitude": 16.50, "longitude": 80.64,
            "category": "aliens", "severity": 2,
            "description": "bad category",
        })
        check(False, "Should have rejected invalid category")
    except RuntimeError as exc:
        check("400" in str(exc), f"Rejected invalid category with 400: {exc}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [6] GET /api/reports
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_get_reports() -> None:
    print("\n[6] GET /api/reports")
    try:
        data = api_get("/api/reports")
    except RuntimeError as exc:
        check(False, f"API call failed: {exc}")
        return

    check("count"   in data, "Response has 'count'")
    check("reports" in data, "Response has 'reports'")
    check(data["count"] >= 200, f"At least 200 reports returned: {data['count']}")

    r0 = data["reports"][0]
    check("report_id"  in r0, "Report has 'report_id'")
    check("category"   in r0, "Report has 'category'")
    check("severity"   in r0, "Report has 'severity'")
    check("description" in r0, "Report has 'description'")
    check("latitude"   in r0, "Report has 'latitude'")
    check("longitude"  in r0, "Report has 'longitude'")
    check("created_at" in r0, "Report has 'created_at'")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [7] GET /api/reports/nearby
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_nearby_reports() -> None:
    print("\n[7] GET /api/reports/nearby")
    try:
        data = api_get(f"/api/reports/nearby?lat={START_LAT}&lon={START_LON}&radius=1000")
    except RuntimeError as exc:
        check(False, f"API call failed: {exc}")
        return

    check("count"          in data, "Response has 'count'")
    check("radius_meters"  in data, "Response has 'radius_meters'")
    check("reports"        in data, "Response has 'reports'")
    check(data["count"] > 0,        f"Found reports within 1000m: {data['count']}")

    r0 = data["reports"][0]
    check("distance_meters" in r0, "Nearby report has 'distance_meters'")

    # Verify ordering
    if len(data["reports"]) > 1:
        d0 = data["reports"][0]["distance_meters"]
        d1 = data["reports"][1]["distance_meters"]
        check(d0 <= d1, f"Results ordered by distance ASC: {d0:.1f} <= {d1:.1f}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [8] GET /api/reports/statistics
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_statistics() -> None:
    print("\n[8] GET /api/reports/statistics")
    try:
        data = api_get("/api/reports/statistics")
    except RuntimeError as exc:
        check(False, f"API call failed: {exc}")
        return

    check("total_reports"           in data, "Response has 'total_reports'")
    check("latest_report_timestamp" in data, "Response has 'latest_report_timestamp'")
    check("reports_by_category"     in data, "Response has 'reports_by_category'")
    check("reports_by_severity"     in data, "Response has 'reports_by_severity'")
    check(data["total_reports"] >= 200, f"total_reports >= 200: {data['total_reports']}")

    cats = data["reports_by_category"]
    check("harassment"          in cats, "Category 'harassment' present")
    check("stalking"            in cats, "Category 'stalking' present")
    check("poor_lighting"       in cats, "Category 'poor_lighting' present")
    check("unsafe_area"         in cats, "Category 'unsafe_area' present")
    check("suspicious_activity" in cats, "Category 'suspicious_activity' present")

    sevs = data["reports_by_severity"]
    check(all(str(k) in sevs or k in sevs for k in range(1, 6)),
          "Severity keys 1-5 all present")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [9] GET /api/reports/health
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_reports_health() -> None:
    print("\n[9] GET /api/reports/health")
    try:
        data = api_get("/api/reports/health")
    except RuntimeError as exc:
        check(False, f"API call failed: {exc}")
        return

    check(data.get("status") == "ok", "status is 'ok'")
    check(data.get("milestone") == 5, "milestone is 5")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [10-12] POST /api/route â€” all routing modes
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_routing_mode(mode: str) -> None:
    print(f"\n[10-12] POST /api/route mode='{mode}'")
    try:
        _, data = api_post("/api/route", {
            "start_lat": START_LAT, "start_lon": START_LON,
            "end_lat":   END_LAT,   "end_lon":   END_LON,
            "mode":      mode,
        })
    except RuntimeError as exc:
        check(False, f"API call failed: {exc}")
        return

    # Milestone 4 backward-compat fields
    check("distance_meters"    in data, f"[{mode}] has 'distance_meters'")
    check("node_count"         in data, f"[{mode}] has 'node_count'")
    check("route_nodes"        in data, f"[{mode}] has 'route_nodes'")
    check("average_route_risk" in data, f"[{mode}] has 'average_route_risk'")
    check("minimum_edge_risk"  in data, f"[{mode}] has 'minimum_edge_risk'")
    check("maximum_edge_risk"  in data, f"[{mode}] has 'maximum_edge_risk'")
    check("risk_category"      in data, f"[{mode}] has 'risk_category'")
    check("geojson"            in data, f"[{mode}] has 'geojson'")

    # Milestone 5 new fields
    check("effective_average_risk"      in data, f"[{mode}] has 'effective_average_risk'")
    check("incident_reports_near_route" in data, f"[{mode}] has 'incident_reports_near_route'")
    check("high_risk_zones"             in data, f"[{mode}] has 'high_risk_zones'")

    # Value sanity
    eff = data["effective_average_risk"]
    check(isinstance(eff, (int, float)) and 0 <= eff <= 100,
          f"[{mode}] effective_average_risk in [0,100]: {eff}")
    inc = data["incident_reports_near_route"]
    check(isinstance(inc, int) and inc >= 0,
          f"[{mode}] incident_reports_near_route >= 0: {inc}")
    hiz = data["high_risk_zones"]
    check(isinstance(hiz, int) and hiz >= 0,
          f"[{mode}] high_risk_zones >= 0: {hiz}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [13] Backward compat â€” no mode field
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_route_no_mode() -> None:
    print("\n[13] POST /api/route â€” no mode (backward compat)")
    try:
        _, data = api_post("/api/route", {
            "start_lat": START_LAT, "start_lon": START_LON,
            "end_lat":   END_LAT,   "end_lon":   END_LON,
        })
    except RuntimeError as exc:
        check(False, f"API call failed: {exc}")
        return

    check("distance_meters"    in data, "Has 'distance_meters'")
    check("route_nodes"        in data, "Has 'route_nodes'")
    check("geojson"            in data, "Has 'geojson'")
    check("nearby_safe_havens" in data, "Has 'nearby_safe_havens'")
    check("average_route_risk" in data, "Has 'average_route_risk' (M4 compat)")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [14] Milestone 4 SRI APIs still work
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_sri_apis() -> None:
    print("\n[14] Milestone 4 SRI APIs")
    try:
        data = api_get("/api/sri/statistics")
        check("edge_count"   in data, "SRI statistics: has 'edge_count'")
        check("average_sri"  in data, "SRI statistics: has 'average_sri'")
        check(data["edge_count"] > 0, f"SRI edge_count > 0: {data.get('edge_count')}")
    except RuntimeError as exc:
        check(False, f"/api/sri/statistics failed: {exc}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  [15] Safe Havens API still works
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_safe_havens() -> None:
    print("\n[15] Safe Havens API")
    try:
        data = api_get("/api/safe-havens")
        check("safe_havens" in data, "Safe Havens: has 'safe_havens'")
        check("count"       in data, "Safe Havens: has 'count'")
    except RuntimeError as exc:
        check(False, f"/api/safe-havens failed: {exc}")

    try:
        data = api_get(f"/api/safe-havens/nearby?lat={START_LAT}&lon={START_LON}&radius=1000")
        check("safe_havens" in data, "Nearby Safe Havens: has 'safe_havens'")
    except RuntimeError as exc:
        check(False, f"/api/safe-havens/nearby failed: {exc}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Runner
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main() -> None:
    print("=" * 64)
    print("  SafeHer AI â€” Milestone 5 Validation")
    print("=" * 64)

    conn = get_db_connection()
    try:
        test_db_schema(conn)
        test_demo_data(conn)
        test_incident_engine_unit()
        test_incident_penalties_dict(conn)
    finally:
        conn.close()

    test_reports_health()
    test_create_report()
    test_create_report_validation()
    test_get_reports()
    test_nearby_reports()
    test_statistics()

    for m in ("fastest", "balanced", "safest"):
        test_routing_mode(m)

    test_route_no_mode()
    test_sri_apis()
    test_safe_havens()

    print()
    print("=" * 64)
    if _failures:
        print(f"  {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"    âœ—  {f}")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")
    print("=" * 64)


if __name__ == "__main__":
    main()

