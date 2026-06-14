import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from db.models import RoadNode, RoadEdge
from config import Config

def verify_db():
    print(f"Connecting to {Config.DATABASE_URI}...")
    try:
        engine = create_engine(Config.DATABASE_URI)
        Session = sessionmaker(bind=engine)
        session = Session()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    # 4. Report counts
    node_count = session.query(RoadNode).count()
    edge_count = session.query(RoadEdge).count()
    
    print(f"\n--- POPULATION REPORT ---")
    print(f"Total Road Nodes: {node_count}")
    print(f"Total Road Edges: {edge_count}")
    
    if node_count == 0:
        print("Database is empty. Did init_db.py run successfully?")
        return
        
    print(f"\n--- SAMPLE RECORDS ---")
    sample_node = session.query(RoadNode).first()
    print(f"Sample Node: osmid={sample_node.osmid}, x={sample_node.x}, y={sample_node.y}")
    
    sample_edge = session.query(RoadEdge).first()
    print(f"Sample Edge: u={sample_edge.u}, v={sample_edge.v}, name={sample_edge.name}, length={sample_edge.length}")

    # 5. Verify geometry columns
    print(f"\n--- GEOMETRY VALIDATION ---")
    invalid_nodes = session.query(RoadNode).filter(text("ST_IsValid(geom) = false")).count()
    invalid_edges = session.query(RoadEdge).filter(text("ST_IsValid(geom) = false")).count()
    print(f"Invalid Node Geometries: {invalid_nodes}")
    print(f"Invalid Edge Geometries: {invalid_edges}")

    # 6. Verify spatial indexes
    print(f"\n--- SPATIAL INDEX VERIFICATION ---")
    index_query = text("""
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE tablename IN ('road_nodes', 'road_edges') AND indexdef LIKE '%gist%';
    """)
    indexes = session.execute(index_query).fetchall()
    print(f"Found {len(indexes)} spatial (GIST) indexes:")
    for idx in indexes:
        print(f" - {idx[0]}")

    # 7. Nearest spatial queries
    print(f"\n--- NEAREST NEIGHBOR QUERIES ---")
    # Vijayawada test coordinate
    test_lat, test_lon = 16.5062, 80.6480
    test_point = f"SRID=4326;POINT({test_lon} {test_lat})"
    
    print(f"Querying nearest node to {test_lat}, {test_lon}...")
    nearest_node = session.query(RoadNode).order_by(
        RoadNode.geom.distance_centroid(test_point)
    ).first()
    
    if nearest_node:
        print(f"Nearest Node: osmid={nearest_node.osmid}")
    
    print(f"Querying nearest edge to {test_lat}, {test_lon}...")
    nearest_edge = session.query(RoadEdge).order_by(
        RoadEdge.geom.distance_centroid(test_point)
    ).first()
    
    if nearest_edge:
        print(f"Nearest Edge: {nearest_edge.name} (length: {nearest_edge.length}m)")

    print("\n✅ Verification script completed.")

if __name__ == "__main__":
    verify_db()
