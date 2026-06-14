from sqlalchemy import Column, Integer, String, Float, ForeignKey, BigInteger, DateTime, SmallInteger
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
from geoalchemy2 import Geometry
import uuid

Base = declarative_base()

class RoadNode(Base):
    __tablename__ = 'road_nodes'
    osmid = Column(BigInteger, primary_key=True)
    y = Column(Float, nullable=False)
    x = Column(Float, nullable=False)
    geom = Column(Geometry(geometry_type='POINT', srid=4326, spatial_index=True))

class RoadEdge(Base):
    __tablename__ = 'road_edges'
    u = Column(BigInteger, ForeignKey('road_nodes.osmid'), primary_key=True)
    v = Column(BigInteger, ForeignKey('road_nodes.osmid'), primary_key=True)
    key = Column(Integer, primary_key=True)
    osmid = Column(BigInteger)
    name = Column(String)
    length = Column(Float)
    geom = Column(Geometry(geometry_type='LINESTRING', srid=4326, spatial_index=True))

class EdgeRiskProfile(Base):
    __tablename__ = 'edge_risk_profiles'
    edge_u = Column(BigInteger, primary_key=True)
    edge_v = Column(BigInteger, primary_key=True)
    edge_key = Column(Integer, primary_key=True)
    sri_score = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    risk_category = Column(String) # 'MEASURED', 'INFERRED', 'UNKNOWN'
    risk_attributions = Column(JSONB)
    last_calculated_at = Column(DateTime)

class UserReputation(Base):
    __tablename__ = 'user_reputation'
    rep_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    reputation_score = Column(Float, default=0.3)
    reports_submitted = Column(Integer, default=0)
    reports_verified = Column(Integer, default=0)

class CommunityReport(Base):
    __tablename__ = 'community_reports'
    report_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rep_id = Column(UUID(as_uuid=True), ForeignKey('user_reputation.rep_id'))
    category = Column(String)
    initial_credibility = Column(Float)
    current_credibility = Column(Float)
    status = Column(String) # 'PENDING', 'VERIFIED', 'REJECTED', 'VELOCITY_LOCKED'
    geom = Column(Geometry(geometry_type='POINT', srid=4326, spatial_index=True))

class ReportVerification(Base):
    __tablename__ = 'report_verifications'
    verification_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey('community_reports.report_id'))
    rep_id = Column(UUID(as_uuid=True), ForeignKey('user_reputation.rep_id'))
    vote = Column(SmallInteger) # +1 or -1
    timestamp = Column(DateTime)
