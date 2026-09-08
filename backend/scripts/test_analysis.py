"""
SafeHer AI — Milestone 6
backend/scripts/test_analysis.py

Comprehensive validation script for the AI Route Safety Analysis & Explainable Score module.

Validates:
  1. GET /api/analysis/health endpoint
  2. Route feature extraction logic (extract_route_features)
  3. Safety score engine calculation (calculate_safety_score)
  4. Correct mapping of score -> safety levels
  5. Content checks for positive factors, negative factors, recommendations
  6. POST /api/route/analyze API endpoint response and schema
  7. Verification of backward compatibility with all milestones

Usage:
    # Make sure Flask server (python backend/app.py) is running first!
    cd backend
    python scripts/test_analysis.py
"""

import sys
import os
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.route_analyzer import extract_route_features
from core.safety_score   import calculate_safety_score
from core.routing_service import get_db_connection

API_BASE  = "http://127.0.0.1:5000"
START_LAT, START_LON = 16.5062, 80.6480
END_LAT,   END_LON   = 16.5100, 80.6550

PASS = "[PASS]"
FAIL = "[FAIL]"
_failures = []


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


# ═══════════════════════════════════════════════════════════════════
#  [1] Health Endpoint
# ═══════════════════════════════════════════════════════════════════

def test_health() -> None:
    print("\n[1] GET /api/analysis/health")
    try:
        data = api_get("/api/analysis/health")
        check(data.get("status") == "ok", "status is 'ok'")
        check(data.get("service") == "SafeHer Route Analysis", "service is correct")
        check(data.get("milestone") == 6, "milestone is 6")
    except Exception as exc:
        check(False, f"Failed to get health status: {exc}")


# ═══════════════════════════════════════════════════════════════════
#  [2] Feature Extraction Unit Test
# ═══════════════════════════════════════════════════════════════════

def test_feature_extraction() -> None:
    print("\n[2] Feature Extraction Unit Test")
    
    mock_route = {
        "average_route_risk": 20.5,
        "effective_average_risk": 28.2,
        "maximum_edge_risk": 55.0,
        "minimum_edge_risk": 10.0,
        "incident_reports_near_route": 4,
        "high_risk_zones": 1,
        "nearby_safe_havens": [{"name": "Police", "category": "police"}],
        "distance_meters": 1200.5,
        "node_count": 15,
        "risk_category": "MODERATE",
    }
    
    features = extract_route_features(mock_route, mode="safest")
    
    check(features["average_route_risk"] == 20.5, "average_route_risk matches")
    check(features["effective_average_risk"] == 28.2, "effective_average_risk matches")
    check(features["maximum_edge_risk"] == 55.0, "maximum_edge_risk matches")
    check(features["minimum_edge_risk"] == 10.0, "minimum_edge_risk matches")
    check(features["incident_reports_near_route"] == 4, "incident_reports_near_route matches")
    check(features["high_risk_zones"] == 1, "high_risk_zones matches")
    check(features["safe_havens_nearby"] == 1, "safe_havens_nearby count matches")
    check(features["distance_meters"] == 1200.5, "distance_meters matches")
    check(features["node_count"] == 15, "node_count matches")
    check(features["routing_mode"] == "safest", "routing_mode matches")
    check(features["risk_category"] == "MODERATE", "risk_category matches")


# ═══════════════════════════════════════════════════════════════════
#  [3] Score Calculation & Levels
# ═══════════════════════════════════════════════════════════════════

def test_score_calculation() -> None:
    print("\n[3] Safety Score Engine Logic")
    
    # ── Test 1: Ideal Safe Route ──────────────────────────────────────────
    # Formula: 100 - 0.5 * 10 - 3 * 0 - 5 * 0 + 2 * 3 = 100 - 5 - 0 - 0 + 6 = 101.0 -> Clamped 100.0
    safe_features = {
        "effective_average_risk": 10.0,
        "incident_reports_near_route": 0,
        "high_risk_zones": 0,
        "safe_havens_nearby": 3,
    }
    res_safe = calculate_safety_score(safe_features)
    check(res_safe["safety_score"] == 100.0, f"Ideal route score is 100.0: {res_safe['safety_score']}")
    check(res_safe["safety_level"] == "SAFE", "Ideal route safety level is SAFE")
    check(res_safe["confidence"] == 0.95, "Confidence score is 0.95")
    
    # ── Test 2: Moderate Risk Route ──────────────────────────────────────
    # Formula: 100 - 0.5 * 40 - 3 * 4 - 5 * 2 + 2 * 1 = 100 - 20 - 12 - 10 + 2 = 60.0
    mod_features = {
        "effective_average_risk": 40.0,
        "incident_reports_near_route": 4,
        "high_risk_zones": 2,
        "safe_havens_nearby": 1,
    }
    res_mod = calculate_safety_score(mod_features)
    check(res_mod["safety_score"] == 60.0, f"Moderate route score is 60.0: {res_mod['safety_score']}")
    check(res_mod["safety_level"] == "LOW RISK", f"Score 60 mapping is LOW RISK: {res_mod['safety_level']}")
    
    # ── Test 3: Critical Route ───────────────────────────────────────────
    # Formula: 100 - 0.5 * 80 - 3 * 15 - 5 * 5 + 2 * 0 = 100 - 40 - 45 - 25 + 0 = -10.0 -> Clamped 0.0
    crit_features = {
        "effective_average_risk": 80.0,
        "incident_reports_near_route": 15,
        "high_risk_zones": 5,
        "safe_havens_nearby": 0,
    }
    res_crit = calculate_safety_score(crit_features)
    check(res_crit["safety_score"] == 0.0, f"Critical route score is 0.0: {res_crit['safety_score']}")
    check(res_crit["safety_level"] == "CRITICAL", "Critical route safety level is CRITICAL")


# ═══════════════════════════════════════════════════════════════════
#  [4] Positive / Negative Factors & Recommendations
# ═══════════════════════════════════════════════════════════════════

def test_explanations() -> None:
    print("\n[4] Explainable Explanations & Recommendations")
    
    features = {
        "effective_average_risk": 15.0,
        "incident_reports_near_route": 5,
        "high_risk_zones": 1,
        "safe_havens_nearby": 0,
        "distance_meters": 3500.0,
        "routing_mode": "fastest",
    }
    
    res = calculate_safety_score(features)
    
    pos = res["positive_factors"]
    neg = res["negative_factors"]
    recs = res["recommendations"]
    
    # Verify presence of correct factors based on the features
    check(any("average effective route risk is low" in f.lower() for f in pos), "Low average risk listed in positive factors")
    check(any("incident reports detected near" in f.lower() for f in neg), "Active incident count listed in negative factors")
    check(any("no nearby safe havens" in f.lower() for f in neg), "No safe havens listed in negative factors")
    check(any("long travel distance" in f.lower() for f in neg), "Long travel distance listed in negative factors")
    check(any("fastest routing mode" in f.lower() for f in neg), "Fastest routing mode warning listed in negative factors")
    
    check(any("share your live location" in r.lower() for r in recs), "Live location sharing listed in recommendations")
    check(any("identify safe refuges" in r.lower() for r in recs), "Identifying safe refuges listed in recommendations")
    check(any("prefer daytime travel" in r.lower() for r in recs), "Prefer daytime travel listed in recommendations")


# ═══════════════════════════════════════════════════════════════════
#  [5] API Integration
# ═══════════════════════════════════════════════════════════════════

def test_api_integration() -> None:
    print("\n[5] POST /api/route/analyze API Integration")
    
    payload = {
        "start_lat": START_LAT,
        "start_lon": START_LON,
        "end_lat":   END_LAT,
        "end_lon":   END_LON,
        "mode":      "safest",
    }
    
    try:
        status, data = api_post("/api/route/analyze", payload)
        check(status == 200, f"API status code is 200: {status}")
        
        # Verify schema layout
        check("route" in data, "Response contains 'route'")
        check("analysis" in data, "Response contains 'analysis'")
        
        route = data["route"]
        check("distance_meters" in route, "Route payload contains 'distance_meters'")
        check("geojson" in route, "Route payload contains 'geojson'")
        check("nearby_safe_havens" in route, "Route payload contains 'nearby_safe_havens'")
        check("incident_reports_near_route" in route, "Route payload contains 'incident_reports_near_route'")
        
        analysis = data["analysis"]
        check("safety_score" in analysis, "Analysis payload contains 'safety_score'")
        check("safety_level" in analysis, "Analysis payload contains 'safety_level'")
        check("confidence" in analysis, "Analysis payload contains 'confidence'")
        check("positive_factors" in analysis, "Analysis payload contains 'positive_factors'")
        check("negative_factors" in analysis, "Analysis payload contains 'negative_factors'")
        check("recommendations" in analysis, "Analysis payload contains 'recommendations'")
        check("features" in analysis, "Analysis payload contains 'features'")
        
        # Values sanity
        check(0 <= analysis["safety_score"] <= 100, f"Safety score is in range [0, 100]: {analysis['safety_score']}")
        check(analysis["confidence"] == 0.95, "Confidence score is 0.95")
        
    except Exception as exc:
        check(False, f"Failed to test API integration: {exc}")


# ═══════════════════════════════════════════════════════════════════
#  [6] Backward Compatibility Checks
# ═══════════════════════════════════════════════════════════════════

def test_backward_compatibility() -> None:
    print("\n[6] Backward Compatibility Checks")
    
    # ── Verify Milestone 2, 3, 4 endpoints still work ──────────────────────
    try:
        data_route = api_get("/api/health")
        check(data_route.get("status") == "ok", "Routing API /api/health still works")
    except Exception as exc:
        check(False, f"Routing API health check failed: {exc}")
        
    try:
        data_havens = api_get("/api/safe-havens")
        check("safe_havens" in data_havens, "Safe Havens API /api/safe-havens still works")
    except Exception as exc:
        check(False, f"Safe Havens API query failed: {exc}")
        
    try:
        data_reports = api_get("/api/reports/statistics")
        check("total_reports" in data_reports, "Reports API /api/reports/statistics still works")
    except Exception as exc:
        check(False, f"Reports API query failed: {exc}")


# ═══════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 64)
    print("  SafeHer AI — Milestone 6 Validation")
    print("=" * 64)

    test_health()
    test_feature_extraction()
    test_score_calculation()
    test_explanations()
    test_api_integration()
    test_backward_compatibility()

    print()
    print("=" * 64)
    if _failures:
        print(f"  {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"    [FAIL]  {f}")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")
    print("=" * 64)


if __name__ == "__main__":
    main()
