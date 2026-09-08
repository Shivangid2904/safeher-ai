# SafeHer AI

## AI-Powered Safe Route Recommendation Platform

**Status:** Active Development

SafeHer AI is a geospatial safety platform designed to recommend safer travel routes rather than simply the shortest or fastest paths. The project combines Geographic Information Systems (GIS), graph algorithms, spatial databases, and safety analysis engines to build an intelligent navigation system focused on personal safety.

The platform integrates geospatial spatial database infrastructure, multi-objective routing (fastest, balanced, safest), a Safe Haven system, SafeHer Risk Index (SRI) scoring, dynamic community incident-aware routing, and explainable route safety analysis. Future development focuses on ML-based scoring (SHAP), user trust reputation systems, real-time navigation recalculation, and privacy-preserving safety features.

---

# Project Vision

Traditional navigation systems optimize routes based on travel time or distance.

SafeHer AI aims to optimize routes based on **safety**.

Instead of asking:

> "What is the shortest route?"

SafeHer AI answers:

> "What is the safest route given the current conditions?"

The platform is designed around four major pillars:

- Geospatial Routing
- Risk Intelligence
- Community Trust
- Privacy & Security

---

# Current Features

## Road Network Infrastructure

- Road network imported from OpenStreetMap using OSMnx
- PostgreSQL + PostGIS spatial database
- Spatial indexing using GiST indexes
- Geometry validation
- Nearest-node and nearest-edge spatial queries

---

## Routing Engine

- Directed road graph built using NetworkX
- Shortest-path computation with support for multi-objective routing (`fastest`, `balanced`, `safest`)
- Graph caching for improved performance
- GeoJSON route generation with edge-by-edge risk metadata
- Flask REST API (`/api/route`)

---

## Safe Haven Layer

SafeHer currently supports spatial querying and visualization of nearby:

- Police Stations
- Hospitals
- Pharmacies

Safe Havens are imported from OpenStreetMap into PostGIS and linked to generated routes within a 200m spatial buffer.

---

## SafeHer Risk Index (SRI)

- Base rule-based edge risk profiling scoring road segments from 0 (safest) to 100 (highest risk) based on road type penalties, lighting status, segment length, and safe haven proximity (Milestone 4)
- Persistent edge risk storage in PostGIS (`edge_risk_profiles`) with detailed risk attribution breakdowns
- REST API for SRI statistics (`/api/sri/statistics`) and individual edge profiles (`/api/sri/edge`)
- Effective risk calculation at routing time combining edge length, base SRI risk, and active Milestone 5 community incident penalties

---

## Dynamic Community Incident-Aware Routing

- Community incident report submission (`POST /api/reports`) and spatial querying (`/api/reports/nearby`, `/api/reports/statistics`)
- Dynamic distance-based penalty calculation (up to 200 m) combined with a 7-day exponential time-decay model
- Runtime edge risk overlay calculated dynamically during route computation without mutating base SRI scores or the immutable graph cache
- Route optimization automatically penalizes and avoids road segments with active incident reports

---

## Route Safety Analysis & Explainability

- Deterministic rule-based route safety assessment engine (`/api/route/analyze`)
- 0–100 overall route safety score and safety level classification (`SAFE`, `LOW RISK`, `MODERATE`, `HIGH RISK`, `CRITICAL`)
- Detailed feature extraction including effective route risk, high risk zone counts, active incident counts, and safe haven counts
- Contextual factor attribution (positive & negative factors) and deterministic safety recommendations derived from extracted route features

---

## Interactive Map

A browser-based interface built using Leaflet allows users to:

- Select start location and destination
- Choose routing mode (`fastest`, `balanced`, `safest`)
- Generate routes with edge-level risk coloring
- Visualize nearby Safe Havens and community incident reports
- Submit community incident reports directly from the interface
- View interactive Safety Analysis panel displaying safety score, confidence rating, positive/negative contributing factors, and recommendations

---

# Current Architecture

```
                                Leaflet Frontend
                                       │
                                Flask REST API
                                       │
 ┌───────────────────┬─────────────────┼───────────────────┬──────────────────┐
 │                   │                 │                   │                  │
Routing Service   Safe Haven Service  SRI Engine    Incident Engine    Route Analyzer & Safety Score Engine
(NetworkX Cache)    (Spatial Query)  (Base Scoring) (Dynamic Overlay)   (Rule Assessment)
 │                   │                 │                   │                  │
 └───────────────────┴─────────────────┼───────────────────┴──────────────────┘
                                       │
                  PostgreSQL / PostGIS (Spatial DB / Source of Truth)
                                       │
                     OpenStreetMap Road Data (OSMnx Import)
```

- **PostgreSQL / PostGIS**: Persistent spatial database and authoritative source of truth storing road edge geometries, safe havens, base SRI risk profiles, and community incident reports.
- **NetworkX Graph Cache**: In-memory directed graph loaded at startup for fast pathfinding and runtime dynamic weight evaluations without modifying PostGIS data.

---

# Planned & Future Capabilities

The platform will expand with the following upcoming architecture:

## 1. ML-Based SafeHer Risk Index & Predictive Analytics

The current deterministic SRI engine will be extended with machine learning models trained on historical data:

- Machine learning risk prediction (scikit-learn / TensorFlow)
- Real-time traffic and crowd density integration
- Temporal and weather-dependent risk adjustment

---

## 2. Advanced Explainable AI (SHAP)

Current explainability is rule-based. Future releases will introduce model-level feature attribution:

- SHAP (SHapley Additive exPlanations) for ML model interpretability
- Detailed feature influence breakdown per route recommendation
- Model confidence scoring and uncertainty visualization

---

## 3. Community Trust System

Building upon the current incident reporting module:

- User reputation scores and credibility weighting
- Multi-user report verification and anti-spam algorithms
- Community feedback loops on route safety quality

---

## 4. Privacy and Security (Milestone 7)

Future releases will include:

- User authentication and role-based access control
- Trusted contacts and automated location sharing
- End-to-end encryption for sensitive data
- One-touch SOS workflows
- Privacy-preserving location anonymization

---

## 5. Dynamic Real-Time Navigation Recalculation

Continuous background monitoring during active navigation:

- Real-time incident alert monitoring
- Automated rerouting when safer paths become available
- Turn-by-turn safety guidance

---

# Technology Stack

## Backend

- Python
- Flask & Flask-CORS
- NetworkX
- psycopg2
- SQLAlchemy & GeoAlchemy2

## Spatial Technologies

- PostgreSQL + PostGIS
- OpenStreetMap
- OSMnx
- Overpass API / overpy

## Frontend

- HTML5 / CSS3
- JavaScript (ES6+)
- Leaflet.js

## Infrastructure

- Docker & Docker Compose
- python-dotenv

## Planned Machine Learning Stack

- Scikit-learn
- TensorFlow
- SHAP

---

# Repository Structure

```
backend/
    api/
        analysis.py
        reports.py
        routes.py
        safe_havens.py
        sri.py
    core/
        incident_engine.py
        route_analyzer.py
        routing_service.py
        safety_score.py
        sri_engine.py
    db/
        models.py
    scripts/
app.py
config.py

frontend/
    index.html

research/

docker-compose.yml
requirements.txt
README.md
```

---

# Development Roadmap

## Milestone 1 — Spatial Infrastructure

Completed

- Dockerized PostgreSQL + PostGIS
- Road graph import using OSMnx
- Spatial indexing using GiST
- Graph validation and geometry verification

---

## Milestone 2 — Routing Engine

Completed

- Directed graph construction with NetworkX
- Shortest-path routing engine
- Route API (`/api/route`)
- GeoJSON output generation
- Graph caching

---

## Milestone 3 — Safe Haven Layer

Completed

- Safe Haven database schema and OSM import
- Nearby Safe Haven spatial queries (police, hospitals, pharmacies)
- Interactive Leaflet map integration
- Route proximity linking

---

## Milestone 4 — SafeHer Risk Index (SRI)

Completed

- Continuous edge risk scoring (road type, lighting, segment length, safe haven proximity)
- Dynamic edge weight calculation and effective route risk computation
- Multi-objective routing modes (`fastest`, `balanced`, `safest`)
- REST API endpoints (`/api/sri/statistics`, `/api/sri/edge`) and PostGIS persistence

---

## Milestone 5 — Dynamic Community Incident-Aware Routing

Completed

- Community incident report submission and query API (`/api/reports`)
- Distance-based (200m) and 7-day time-decayed penalty calculation
- Runtime edge risk overlay without graph cache mutation
- Incident-aware route optimization and frontend map visualization

---

## Milestone 6 — Explainable Route Safety Analysis

Completed

- Deterministic rule-based route safety assessment engine (`calculate_safety_score`)
- 0–100 overall route safety score and safety level classification (`SAFE` to `CRITICAL`)
- Contributing factor attribution (positive and negative factors) and contextual recommendations derived from route features
- Integrated `/api/route/analyze` endpoint and interactive frontend analysis panel
- *(Note: ML-based SHAP explainability is planned for future releases)*

---

## Milestone 7 — Privacy & Security

Planned

- Authentication and user management
- Trusted contacts and location sharing
- SOS workflows
- Privacy-preserving architecture

---

# Future Scope

The long-term objective is to evolve SafeHer AI into a comprehensive safety navigation platform capable of combining geospatial analytics, artificial intelligence, explainable machine learning, and privacy-first system design to support safer urban mobility.

---

# Author

**Shivangi Dubey**
