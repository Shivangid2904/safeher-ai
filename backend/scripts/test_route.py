"""
SafeHer AI — Milestone 2
backend/scripts/test_route.py

Standalone integration test — no pytest required.
Runs two real Vijayawada route requests against the live server and
asserts the expected response structure and values.

Usage (server must already be running on localhost:5000):
    cd backend
    python scripts/test_route.py
"""

import sys
import json

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package is not installed. Run: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:5000"
ROUTE_URL = f"{BASE_URL}/api/route"

TESTS = [
    {
        "name": "Test 1 — Short route (central Vijayawada)",
        "payload": {
            "start_lat": 16.5062,
            "start_lon": 80.6480,
            "end_lat":   16.5120,
            "end_lon":   80.6550,
        },
    },
    {
        "name": "Test 2 — Longer route (north to central Vijayawada)",
        "payload": {
            "start_lat": 16.5193,
            "start_lon": 80.6350,
            "end_lat":   16.5062,
            "end_lon":   80.6480,
        },
    },
]


def run_test(test: dict) -> None:
    name = test["name"]
    payload = test["payload"]

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Request: {json.dumps(payload, indent=4)}")

    response = requests.post(ROUTE_URL, json=payload, timeout=30)

    # ── pretty-print key fields ───────────────────────────────────────────
    if response.status_code == 200:
        body = response.json()
        print(f"  Status           : {response.status_code}")
        print(f"  distance_meters  : {body.get('distance_meters')}")
        print(f"  node_count       : {body.get('node_count')}")
        print(f"  first 3 nodes    : {body.get('route_nodes', [])[:3]}")
        geojson_coords = (
            body.get("geojson", {})
                .get("geometry", {})
                .get("coordinates", [])
        )
        print(f"  first coordinate : {geojson_coords[0] if geojson_coords else 'N/A'}")
    else:
        print(f"  Status           : {response.status_code}")
        print(f"  Body             : {response.text}")

    # ── assertions ────────────────────────────────────────────────────────
    assert response.status_code == 200, (
        f"[FAIL] Expected status 200, got {response.status_code}. "
        f"Body: {response.text}"
    )

    body = response.json()

    assert body.get("distance_meters", 0) > 0, (
        f"[FAIL] distance_meters should be > 0, got {body.get('distance_meters')}"
    )

    assert body.get("node_count", 0) >= 2, (
        f"[FAIL] node_count should be >= 2, got {body.get('node_count')}"
    )

    geojson = body.get("geojson")
    assert geojson is not None, "[FAIL] 'geojson' key missing from response"
    assert geojson.get("type") == "Feature", (
        f"[FAIL] geojson.type should be 'Feature', got {geojson.get('type')}"
    )

    geometry = geojson.get("geometry", {})
    assert geometry.get("type") == "LineString", (
        f"[FAIL] geojson.geometry.type should be 'LineString', "
        f"got {geometry.get('type')}"
    )

    coordinates = geometry.get("coordinates", [])
    assert len(coordinates) > 0, (
        "[FAIL] geojson.geometry.coordinates should be non-empty"
    )

    print(f"  [PASS] All assertions passed for '{name}'")


def main() -> None:
    print("\nSafeHer AI — Milestone 2 Integration Tests")
    print(f"Target: {ROUTE_URL}\n")

    failed = False
    for test in TESTS:
        try:
            run_test(test)
        except AssertionError as exc:
            print(f"\n  {exc}")
            failed = True
        except requests.exceptions.ConnectionError:
            print(
                f"\n  [ERROR] Could not connect to {BASE_URL}. "
                "Is the Flask server running?\n"
                "  Start it with:  cd backend && python app.py"
            )
            sys.exit(1)
        except Exception as exc:
            print(f"\n  [ERROR] Unexpected exception: {exc}")
            failed = True

    print(f"\n{'=' * 60}")
    if failed:
        print("  SOME TESTS FAILED — see output above.")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
