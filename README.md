# SafeHer AI

## AI-Powered Safe Route Recommendation Platform

**Status:** Active Development (Prototype)

SafeHer AI is a geospatial safety platform designed to recommend safer travel routes rather than simply the shortest or fastest paths. The project combines Geographic Information Systems (GIS), graph algorithms, spatial databases, and machine learning to build an intelligent navigation system focused on personal safety.

The current repository contains the working prototype, including the routing infrastructure, spatial database, and interactive route visualization. Future development will introduce AI-powered risk scoring, explainability, community intelligence, and privacy-preserving safety features.

---

# Project Vision

Traditional navigation systems optimize routes based on travel time or distance.

SafeHer AI aims to optimize routes based on **safety**.

Instead of asking:

> "What is the shortest route?"

SafeHer AI answers:

> "What is the safest route given the current conditions?"

The complete platform is designed around four major pillars:

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
- Shortest-path computation
- Graph caching for improved performance
- GeoJSON route generation
- Flask REST API

---

## Safe Haven Layer

SafeHer currently supports spatial querying and visualization of nearby:

- Police Stations
- Hospitals
- Pharmacies

Safe Havens are imported from OpenStreetMap and linked to generated routes.

---

## Interactive Map

A browser-based interface built using Leaflet allows users to:

- Select start location
- Select destination
- Generate a route
- Visualize nearby Safe Havens
- Display route statistics

---

# Current Architecture

```
                    Leaflet Frontend
                           │
                    Flask REST API
                           │
            ┌──────────────┴──────────────┐
            │                             │
     Routing Service             Safe Haven Service
            │                             │
            └──────────────┬──────────────┘
                           │
                    PostgreSQL/PostGIS
                           │
                 OpenStreetMap Road Graph
```

---

# Planned Architecture

The complete SafeHer platform extends beyond the current prototype.

## 1. SafeHer Risk Index (SRI)

Each road segment will receive a dynamic safety score derived from:

- Historical crime statistics
- Lighting conditions
- Crowd density
- Time of day
- Safe Haven proximity
- Community reports

---

## 2. Risk-Aware Route Recommendation

Instead of minimizing only distance,

the routing engine will minimize:

```
Travel Cost =
Travel Time
+ Risk Score
+ Uncertainty
```

Supported routing modes:

- Fastest
- Safest
- Balanced

---

## 3. Explainable AI

Every generated safety score will include an explanation showing the factors contributing to the decision.

Example:

```
Risk Score: 82

Primary Contributors

• Poor lighting
• Recent harassment reports
• No nearby Safe Havens

Confidence: 92%
```

The explainability module is planned using SHAP.

---

## 4. Community Trust System

Users will be able to submit reports including:

- Harassment
- Following behaviour
- Poor lighting
- Broken infrastructure
- Suspicious activity

Reports will influence routing only after credibility evaluation using:

- Reputation scores
- Report verification
- Time decay
- Spatial validation

---

## 5. Privacy and Security

Future releases will include:

- Anonymous reporting
- Trusted contacts
- End-to-end encryption
- SOS workflows
- Location anonymization
- Secure data retention
- Role-based access control

---

## 6. Dynamic Route Recalculation

During active navigation the routing engine will continuously monitor:

- Community reports
- Dynamic risk updates
- Road closures
- Emergency incidents

Routes will automatically update whenever a safer alternative becomes available.

---

# Technology Stack

## Backend

- Python
- Flask
- NetworkX
- psycopg2

## Spatial Technologies

- PostgreSQL
- PostGIS
- OpenStreetMap
- OSMnx
- Overpass API

## Frontend

- HTML
- CSS
- JavaScript
- Leaflet

## Infrastructure

- Docker
- Docker Compose

## Planned Machine Learning Stack

- TensorFlow
- Scikit-learn
- SHAP

---

# Repository Structure

```
backend/
    api/
    core/
    db/
    scripts/

frontend/

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
- Spatial indexing
- Graph validation
- Geometry verification

---

## Milestone 2 — Routing Engine

Completed

- Directed graph construction
- Shortest-path routing
- Route API
- GeoJSON output
- Graph caching

---

## Milestone 3 — Safe Haven Layer

Completed

- Safe Haven database
- Nearby Safe Haven queries
- Interactive Leaflet map
- Route visualization

---

## Milestone 4 — SafeHer Risk Index

Planned

- Continuous risk scoring
- Dynamic edge weights
- Multi-objective routing

---

## Milestone 5 — Community Intelligence

Planned

- Community reports
- Trust and reputation system
- Dynamic route updates

---

## Milestone 6 — Explainable AI

Planned

- SHAP explanations
- Confidence scoring
- Feature attribution

---

## Milestone 7 — Privacy & Security

Planned

- Authentication
- Trusted contacts
- SOS workflows
- Privacy-preserving architecture

---

# Future Scope

The long-term objective is to evolve SafeHer AI from a routing prototype into a comprehensive safety navigation platform capable of combining geospatial analytics, artificial intelligence, explainable machine learning, and privacy-first system design to support safer urban mobility.

---

# Author

**Shivangi Dubey**
