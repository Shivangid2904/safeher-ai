import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import osmnx as ox
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base, RoadNode, RoadEdge
from geoalchemy2.elements import WKTElement

from config import Config
DATABASE_URI = Config.DATABASE_URI

def init_db():
    print(f"Connecting to {DATABASE_URI}")
    engine = create_engine(DATABASE_URI)
    
    # Create tables
    print("Creating tables...")
    try:
        # Create extension if not exists requires superuser, but image postgis/postgis has it by default
        engine.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    except Exception as e:
        print("Note: Could not run CREATE EXTENSION postgis. Assuming it already exists or lacks permission.")
        pass
        
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    print("Downloading OSMnx graph for Vijayawada, India (small bounding box)...")
    # Small bounding box to avoid massive downloads
    # Coordinates from map.py: 16.5062, 80.6480
    try:
        G = ox.graph_from_point((16.5062, 80.6480), dist=1000, network_type='drive')
    except Exception as e:
        print(f"Error downloading graph: {e}")
        return
    
    print("Extracting nodes and edges...")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    
    print(f"Inserting {len(gdf_nodes)} nodes...")
    for index, row in gdf_nodes.iterrows():
        point = f"POINT({row['x']} {row['y']})"
        node = RoadNode(
            osmid=index,
            y=row['y'],
            x=row['x'],
            geom=WKTElement(point, srid=4326)
        )
        session.merge(node)
    
    session.commit()
    
    print(f"Inserting {len(gdf_edges)} edges...")
    for index, row in gdf_edges.iterrows():
        u, v, key = index
        
        # Some edges might not have a geometry column if they are straight lines
        if 'geometry' in row and row['geometry']:
            geom = WKTElement(row['geometry'].wkt, srid=4326)
        else:
            # Create straight line between u and v
            u_node = session.query(RoadNode).get(u)
            v_node = session.query(RoadNode).get(v)
            if u_node and v_node:
                geom = WKTElement(f"LINESTRING({u_node.x} {u_node.y}, {v_node.x} {v_node.y})", srid=4326)
            else:
                continue
                
        # Handle osmid being a list sometimes
        osmid = row.get('osmid', 0)
        if isinstance(osmid, list):
            osmid = osmid[0]
            
        name = row.get('name', '')
        if isinstance(name, list):
            name = name[0]
            
        edge = RoadEdge(
            u=u,
            v=v,
            key=key,
            osmid=int(osmid) if not pd.isna(osmid) else 0,
            name=name if isinstance(name, str) else str(name),
            length=row.get('length', 0.0),
            geom=geom
        )
        session.merge(edge)
        
    session.commit()
    print("Database initialization complete!")

if __name__ == "__main__":
    import pandas as pd
    init_db()
