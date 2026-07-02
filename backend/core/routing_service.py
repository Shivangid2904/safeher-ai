"""
SafeHer AI — Milestone 4
backend/core/routing_service.py

Synchronous routing engine using psycopg2 + NetworkX.
Graph is built ONCE at module load / first access, then cached.

Milestone 4 additions (backward-compatible):
  • build_graph() LEFT JOINs edge_risk_profiles to store 'distance' and
    'risk' on every NetworkX edge.
  • calculate_edge_cost(edge, mode) computes traversal cost dynamically
    so the graph itself is never mutated when routing modes change.
  • compute_shortest_path() accepts an optional `mode` argument and
    returns average_route_risk, minimum_edge_risk, maximum_edge_risk.

Schema (from db/models.py):
  road_nodes      : osmid (PK), y (lat), x (lon), geom
  road_edges      : u (source FK), v (target FK), key (INT), length, geom
  edge_risk_profiles : edge_u, edge_v, edge_key, sri_score
"""

from __future__ import annotations

import logging
import os

import networkx as nx
import psycopg2
from dotenv import load_dotenv

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
    """Return a new psycopg2 connection.  Caller is responsible for closing it."""
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


def calculate_edge_cost(edge: dict, mode: str) -> float:
    """
    Compute the traversal cost for a single edge given a routing mode.

    The edge dict must have keys 'distance' (float, metres) and
    'risk' (float, 0-100).

    Modes
    -----
    fastest  : cost = distance
    balanced : cost = 0.6 × distance + 0.4 × risk
    safest   : cost = 0.2 × distance + 0.8 × risk

    The graph itself is NEVER modified by this function.
    """
    distance = edge.get("distance", 1.0)
    risk     = edge.get("risk",     0.0)

    if mode == "balanced":
        return 0.6 * distance + 0.4 * risk
    if mode == "safest":
        return 0.2 * distance + 0.8 * risk
    # default: fastest
    return distance


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> nx.DiGraph:
    """
    Load ALL road_nodes and road_edges from PostGIS into a NetworkX DiGraph.

    Milestone 4 change: LEFT JOIN with edge_risk_profiles to store both
    'distance' (metres) and 'risk' (0-100 SRI score) on every edge.
    When no SRI record exists yet, 'risk' defaults to 30.0 (unknown road
    penalty — matches the sri_engine default for an unscored edge).

    Multi-edge handling (unchanged from Milestone 2):
    OSMnx may store several edges for the same (u, v) pair.  DiGraph keeps
    only one edge per (u, v) — we keep the one with the *smallest* length.
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
                    # Keep the shortest distance (Milestone 2 behaviour)
                    if dist_m < G[source][target]["distance"]:
                        G[source][target]["distance"] = dist_m
                        G[source][target]["risk"]     = risk_val
                        # Keep 'weight' in sync for generic nx algorithms
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
    Dijkstra's algorithm weighted by calculate_edge_cost(edge, mode).

    Milestone 4 additions:
      The response now includes:
        average_route_risk   — mean SRI score across all path edges
        minimum_edge_risk    — lowest SRI score on the path
        maximum_edge_risk    — highest SRI score on the path
        risk_category        — category derived from average_route_risk

    Parameters
    ----------
    start_node : int
    end_node   : int
    G          : nx.DiGraph
    mode       : str — 'fastest' | 'balanced' | 'safest'

    Returns
    -------
    dict with keys:
        distance_meters, node_count, route_nodes, geojson,
        average_route_risk, minimum_edge_risk, maximum_edge_risk,
        risk_category

    Raises ValueError for nx.NetworkXNoPath or nx.NodeNotFound.
    """
    if mode not in _VALID_MODES:
        logger.warning("Unknown routing mode '%s'; defaulting to 'fastest'.", mode)
        mode = _DEFAULT_MODE

    # Weight function passed to Dijkstra — reads edge data dynamically
    def _weight_fn(u: int, v: int, edge_data: dict) -> float:
        return calculate_edge_cost(edge_data, mode)

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

    # ── Accumulate distance and collect risk values ────────────────────────
    total_distance = 0.0
    risk_values: list[float] = []

    for i in range(len(path) - 1):
        edge_data = G[path[i]][path[i + 1]]
        total_distance += edge_data.get("distance", 0.0)
        risk_values.append(edge_data.get("risk", 30.0))

    # ── Route risk statistics ──────────────────────────────────────────────
    if risk_values:
        avg_risk = round(sum(risk_values) / len(risk_values), 2)
        min_risk = round(min(risk_values), 2)
        max_risk = round(max(risk_values), 2)
    else:
        avg_risk = min_risk = max_risk = 0.0

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
        "Route found: %d nodes, %.2fm, mode=%s, avg_risk=%.2f",
        node_count, distance_rounded, mode, avg_risk,
    )

    return {
        "distance_meters":    distance_rounded,
        "node_count":         node_count,
        "route_nodes":        path,
        "average_route_risk": avg_risk,
        "minimum_edge_risk":  min_risk,
        "maximum_edge_risk":  max_risk,
        "risk_category":      _risk_category(avg_risk),
        "geojson": {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "distance_meters":    distance_rounded,
                "node_count":         node_count,
                "mode":               mode,
                "average_route_risk": avg_risk,
                "minimum_edge_risk":  min_risk,
                "maximum_edge_risk":  max_risk,
                "risk_category":      _risk_category(avg_risk),
            },
        },
    }
