"""
SafeHer AI — Milestone 2
backend/core/routing_service.py

Synchronous routing engine using psycopg2 + NetworkX.
Graph is built ONCE at module load / first access, then cached.

Schema (from db/models.py):
  road_nodes : osmid (PK, BigInteger), y (lat, Float), x (lon, Float), geom
  road_edges : u (source FK), v (target FK), key (INT), length (Float), geom
"""

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
    user = os.getenv("POSTGRES_USER", "safeher_user")
    password = os.getenv("POSTGRES_PASSWORD", "safeher_pass")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "safeher")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_db_connection() -> psycopg2.extensions.connection:
    """Return a new psycopg2 connection.  Caller is responsible for closing it."""
    dsn = _build_database_url()
    return psycopg2.connect(dsn)


# ---------------------------------------------------------------------------
# Module-level graph cache
# ---------------------------------------------------------------------------
_graph: nx.DiGraph | None = None


def build_graph() -> nx.DiGraph:
    """
    Load ALL road_nodes and road_edges from PostGIS into a NetworkX DiGraph.

    Node key  : osmid  (BigInteger, the primary key of road_nodes)
    Node attrs: lat (y column), lon (x column)
    Edge weight: length in metres (road_edges.length); defaults to 1.0 when NULL.

    Multi-edge handling: OSMnx may store several edges for the same (u, v) pair
    (different OSM key values). DiGraph keeps only one edge per (u, v) — we
    keep the one with the *smallest* length.
    """
    conn = get_db_connection()
    try:
        G = nx.DiGraph()

        # ── nodes ────────────────────────────────────────────────────────────
        with conn.cursor() as cur:
            cur.execute(
                "SELECT osmid, y, x FROM road_nodes;"
            )
            for osmid, y, x in cur.fetchall():
                G.add_node(int(osmid), lat=float(y), lon=float(x))

        # ── edges ────────────────────────────────────────────────────────────
        duplicate_count = 0
        with conn.cursor() as cur:
            cur.execute(
                "SELECT u, v, length FROM road_edges;"
            )
            for u, v, length in cur.fetchall():
                source = int(u)
                target = int(v)

                if length is None:
                    logger.warning(
                        "NULL length on edge (%s -> %s); defaulting to 1.0",
                        source, target
                    )
                    length_m = 1.0
                else:
                    length_m = float(length)

                if G.has_edge(source, target):
                    duplicate_count += 1
                    current_weight = G[source][target]["weight"]
                    if length_m < current_weight:
                        G[source][target]["weight"] = length_m
                    # else: keep existing shorter edge — do nothing
                else:
                    G.add_edge(source, target, weight=length_m)

        logger.info(
            "Graph loaded: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges()
        )
        logger.info(
            "Duplicate edges encountered and resolved: %d", duplicate_count
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
        # row[0] is already a Python list of [lon, lat] lists (psycopg2 + json cast)
        return row[0]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def compute_shortest_path(
    start_node: int,
    end_node: int,
    G: nx.DiGraph,
) -> dict:
    """
    Compute the shortest path from start_node to end_node in G using
    Dijkstra's algorithm weighted by 'weight' (metres).

    Returns:
        {
          "distance_meters": float,
          "node_count": int,
          "route_nodes": [int, ...],
          "geojson": {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[lon, lat], ...]},
            "properties": {"distance_meters": float, "node_count": int}
          }
        }

    Raises ValueError for nx.NetworkXNoPath or nx.NodeNotFound.
    """
    try:
        path: list[int] = nx.shortest_path(G, start_node, end_node, weight="weight")
    except nx.NodeNotFound as exc:
        raise ValueError(f"Node not found in graph: {exc}") from exc
    except nx.NetworkXNoPath as exc:
        raise ValueError(
            f"No path exists between node {start_node} and node {end_node}."
        ) from exc

    # ── accumulate distance ───────────────────────────────────────────────
    total_distance = 0.0
    for i in range(len(path) - 1):
        edge_data = G[path[i]][path[i + 1]]
        total_distance += edge_data.get("weight", 0.0)

    # ── fetch geometry from PostGIS ───────────────────────────────────────
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
            # Avoid duplicating the junction point between consecutive edges
            if coordinates and coords:
                coordinates.extend(coords[1:])
            else:
                coordinates.extend(coords)
    finally:
        conn.close()

    node_count = len(path)
    distance_rounded = round(total_distance, 2)

    logger.info(
        "Route found: %d nodes, %.2fm", node_count, distance_rounded
    )

    return {
        "distance_meters": distance_rounded,
        "node_count": node_count,
        "route_nodes": path,
        "geojson": {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "distance_meters": distance_rounded,
                "node_count": node_count,
            },
        },
    }
