"""
SafeHer AI — Milestone 5
backend/core/routing_service.py

Synchronous routing engine using psycopg2 + NetworkX.
Graph is built ONCE at module load / first access, then cached.

Milestone 5 additions (backward-compatible):
  • Immutability preservation: Cached graph is NEVER mutated.
  • Incident integration: get_active_incident_penalties() is loaded
    at the start of compute_shortest_path() to apply runtime-only modifiers.
  • calculate_edge_cost() is updated to dynamically add the incident_penalty
    to build the effective_risk.
  • compute_shortest_path() calculates and returns:
      - effective_average_risk (length-weighted average of effective risk)
      - high_risk_zones (edges with effective risk > 50.0)
      - incident_reports_near_route (defaulting to 0, computed via routes.py)
"""

from __future__ import annotations

import logging
import os

import networkx as nx
import psycopg2
from dotenv import load_dotenv

from core.incident_engine import get_active_incident_penalties

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment / connection
# ---------------------------------------------------------------------------
load_dotenv()


def _build_database_url() -> str:
    """
    Prefer DATABASE_URL env var; fall back to individual POSTGRES_* vars
    (matching the pattern already used in backend/config.py).
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user     = os.getenv("POSTGRES_USER",     "safeher_user")
    password = os.getenv("POSTGRES_PASSWORD", "safeher_pass")
    host     = os.getenv("POSTGRES_HOST",     "localhost")
    port     = os.getenv("POSTGRES_PORT",     "5432")
    db       = os.getenv("POSTGRES_DB",       "safeher")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_db_connection() -> psycopg2.extensions.connection:
    """Return a new psycopg2 connection. Caller is responsible for closing it."""
    dsn = _build_database_url()
    return psycopg2.connect(dsn)


# ---------------------------------------------------------------------------
# Module-level graph cache
# ---------------------------------------------------------------------------
_graph: nx.DiGraph | None = None


# ---------------------------------------------------------------------------
# Dynamic edge cost (never mutates the graph)
# ---------------------------------------------------------------------------
_VALID_MODES = {"fastest", "balanced", "safest"}
_DEFAULT_MODE = "fastest"


def calculate_edge_cost(edge: dict, mode: str, incident_penalty: float = 0.0) -> float:
    """
    Compute the traversal cost for a single edge given a routing mode and incident penalty.

    The edge dict must have keys 'distance' (float, metres) and
    'risk' (float, 0-100 SRI).

    Effective Risk = Base SRI + Incident Penalty (clamped to 100).

    Modes
    -----
    fastest  : cost = distance
    balanced : cost = 0.6 × distance + 0.4 × effective_risk
    safest   : cost = 0.2 × distance + 0.8 × effective_risk

    The graph itself is NEVER modified by this function.
    """
    distance = edge.get("distance", 1.0)
    base_risk = edge.get("risk", 30.0)
    
    # Calculate effective risk: Base SRI + dynamic incident penalty (clamped in [0.0, 100.0])
    effective_risk = max(0.0, min(100.0, base_risk + incident_penalty))

    if mode == "balanced":
        return 0.6 * distance + 0.4 * effective_risk
    if mode == "safest":
        return 0.2 * distance + 0.8 * effective_risk
    # default: fastest
    return distance


# ---------------------------------------------------------------------------
# Graph construction (Milestone 4 - unmodified)
# ---------------------------------------------------------------------------

def build_graph() -> nx.DiGraph:
    """
    Load ALL road_nodes and road_edges from PostGIS into a NetworkX DiGraph.

    Milestone 4 change: LEFT JOIN with edge_risk_profiles to store both
    'distance' (metres) and 'risk' (0-100 SRI score) on every edge.
    When no SRI record exists yet, 'risk' defaults to 30.0.
    """
    conn = get_db_connection()
    try:
        G = nx.DiGraph()

        # ── nodes ─────────────────────────────────────────────────────────
        with conn.cursor() as cur:
            cur.execute("SELECT osmid, y, x FROM road_nodes;")
            for osmid, y, x in cur.fetchall():
                G.add_node(int(osmid), lat=float(y), lon=float(x))

        # ── edges (with SRI join) ─────────────────────────────────────────
        duplicate_count = 0
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    re.u,
                    re.v,
                    re.length,
                    COALESCE(erp.sri_score, 30.0) AS risk
                FROM road_edges re
                LEFT JOIN edge_risk_profiles erp
                    ON  erp.edge_u   = re.u
                    AND erp.edge_v   = re.v
                    AND erp.edge_key = re.key;
            """)
            for u, v, length, risk in cur.fetchall():
                source   = int(u)
                target   = int(v)
                dist_m   = float(length) if length is not None else 1.0
                risk_val = float(risk)   if risk   is not None else 30.0

                if G.has_edge(source, target):
                    duplicate_count += 1
                    # Keep the shortest distance
                    if dist_m < G[source][target]["distance"]:
                        G[source][target]["distance"] = dist_m
                        G[source][target]["risk"]     = risk_val
                        G[source][target]["weight"]   = dist_m
                else:
                    G.add_edge(
                        source, target,
                        distance=dist_m,
                        risk=risk_val,
                        weight=dist_m,      # backward-compat alias
                    )

        logger.info(
            "Graph loaded: %d nodes, %d edges (duplicates resolved: %d)",
            G.number_of_nodes(), G.number_of_edges(), duplicate_count,
        )
        return G

    finally:
        conn.close()


def get_graph() -> nx.DiGraph:
    """
    Return the cached DiGraph, building it on first call.
    Thread-safe for single-threaded Flask dev server.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def find_nearest_node(lat: float, lon: float) -> int:
    """
    Return the osmid of the road_node closest to (lat, lon).

    Uses PostGIS KNN operator (<->) for index-accelerated nearest-neighbour.
    NOTE: ST_MakePoint(longitude, latitude) — positional order matters.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT osmid
                FROM road_nodes
                ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                LIMIT 1;
                """,
                (lon, lat),   # ST_MakePoint(lon, lat) — longitude FIRST
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError("road_nodes table is empty — cannot find nearest node.")
            node_id = int(row[0])
            logger.info("Nearest node to (%s, %s): %d", lat, lon, node_id)
            return node_id
    finally:
        conn.close()


def get_edge_geometry(
    conn: psycopg2.extensions.connection,
    source: int,
    target: int,
) -> list[list[float]]:
    """
    Fetch the LineString geometry for edge (source -> target).

    Returns a list of [lon, lat] coordinate pairs (GeoJSON order).
    Returns [] if no matching edge is found.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ST_AsGeoJSON(geom)::json->'coordinates'
            FROM road_edges
            WHERE u = %s AND v = %s
            LIMIT 1;
            """,
            (source, target),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return []
        return row[0]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def compute_shortest_path(
    start_node: int,
    end_node: int,
    G: nx.DiGraph,
    mode: str = "fastest",
) -> dict:
    """
    Compute the shortest path from start_node to end_node in G using
    Dijkstra's algorithm weighted by calculate_edge_cost(edge, mode, penalty).

    Milestone 5 additions:
      • Fetches all active incident penalties in a single database query.
      • Dynamically incorporates penalties into routing cost without mutating G.
      • Calculates and returns:
          - effective_average_risk
          - high_risk_zones
          - incident_reports_near_route (defaults to 0, count updated in routes.py)
      • Preserves all Milestone 4 return parameters for backward compatibility.
    """
    if mode not in _VALID_MODES:
        logger.warning("Unknown routing mode '%s'; defaulting to 'fastest'.", mode)
        mode = _DEFAULT_MODE

    # ── Load dynamic incident penalties ────────────────────────────────────
    conn = get_db_connection()
    try:
        incident_penalties = get_active_incident_penalties(conn)
    finally:
        conn.close()

    # Weight function passed to Dijkstra — reads edge attributes + dynamic penalty
    def _weight_fn(u: int, v: int, edge_data: dict) -> float:
        penalty = incident_penalties.get((u, v), 0.0)
        return calculate_edge_cost(edge_data, mode, incident_penalty=penalty)

    try:
        path: list[int] = nx.shortest_path(
            G, start_node, end_node, weight=_weight_fn
        )
    except nx.NodeNotFound as exc:
        raise ValueError(f"Node not found in graph: {exc}") from exc
    except nx.NetworkXNoPath as exc:
        raise ValueError(
            f"No path exists between node {start_node} and node {end_node}."
        ) from exc

    # ── Accumulate metrics along the path ──────────────────────────────────
    total_distance = 0.0
    total_effective_risk_distance = 0.0
    
    base_risks: list[float] = []
    high_risk_zones = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G[u][v]
        dist = edge_data.get("distance", 1.0)
        base_risk = edge_data.get("risk", 30.0)
        
        penalty = incident_penalties.get((u, v), 0.0)
        effective_risk = max(0.0, min(100.0, base_risk + penalty))

        total_distance += dist
        total_effective_risk_distance += effective_risk * dist
        base_risks.append(base_risk)

        if effective_risk > 50.0:
            high_risk_zones += 1

    # ── Calculate statistics ───────────────────────────────────────────────
    if base_risks:
        avg_base_risk = round(sum(base_risks) / len(base_risks), 2)
        min_base_risk = round(min(base_risks), 2)
        max_base_risk = round(max(base_risks), 2)
    else:
        avg_base_risk = min_base_risk = max_base_risk = 0.0

    effective_avg_risk = round(
        (total_effective_risk_distance / total_distance) if total_distance > 0 else 0.0, 
        2
    )

    def _risk_category(score: float) -> str:
        if score <= 25:
            return "LOW"
        if score <= 50:
            return "MODERATE"
        if score <= 75:
            return "HIGH"
        return "CRITICAL"

    # ── Fetch geometry from PostGIS ────────────────────────────────────────
    coordinates: list[list[float]] = []
    conn = get_db_connection()
    try:
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            coords = get_edge_geometry(conn, src, tgt)
            if not coords:
                logger.warning(
                    "No geometry for edge (%d -> %d), skipping", src, tgt
                )
                continue
            if coordinates and coords:
                coordinates.extend(coords[1:])
            else:
                coordinates.extend(coords)
    finally:
        conn.close()

    node_count       = len(path)
    distance_rounded = round(total_distance, 2)

    logger.info(
        "Route computed: %d nodes, %.2fm, mode=%s, base_avg_risk=%.2f, eff_avg_risk=%.2f",
        node_count, distance_rounded, mode, avg_base_risk, effective_avg_risk,
    )

    # Output structure keeps all Milestone 4 parameters and adds Milestone 5 ones
    return {
        "distance_meters":             distance_rounded,
        "node_count":                  node_count,
        "route_nodes":                 path,
        "average_route_risk":          avg_base_risk,     # base SRI stats (compat)
        "minimum_edge_risk":           min_base_risk,
        "maximum_edge_risk":           max_base_risk,
        "risk_category":               _risk_category(avg_base_risk),
        "effective_average_risk":      effective_avg_risk, # dynamic SRI stats (new)
        "incident_reports_near_route": 0,                 # set by route API wrapper
        "high_risk_zones":             high_risk_zones,
        "geojson": {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "distance_meters":             distance_rounded,
                "node_count":                  node_count,
                "mode":                        mode,
                "average_route_risk":          avg_base_risk,
                "minimum_edge_risk":           min_base_risk,
                "maximum_edge_risk":           max_base_risk,
                "risk_category":               _risk_category(avg_base_risk),
                "effective_average_risk":      effective_avg_risk,
                "incident_reports_near_route": 0,
                "high_risk_zones":             high_risk_zones,
            },
        },
    }
