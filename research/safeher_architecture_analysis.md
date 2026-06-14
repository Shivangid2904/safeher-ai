# SafeHer AI - Project Analysis & Migration Plan

This document provides a comprehensive evaluation of the current SafeHer AI codebase and outlines the strategic roadmap for transforming it into a production-ready women's safety platform.

---

## 1. Current Architecture Analysis

The current project is a monolithic prototype designed for local execution.

*   **Backend Interface:** A lightweight Flask application (`backend/flask_api/app.py`) exposing a single POST endpoint (`/predict_risk`).
*   **Machine Learning Integration:** A Keras/TensorFlow model (`model.h5`) loaded directly into the Flask application's memory at startup. Input features are limited to: `lat`, `long`, `hour`, `crime_score`, and `crowd_density`.
*   **Data Pipeline:** The model relies on synthetic data (`data/synthetic/`) generated for training (`train.py`), completely lacking real-world datasets or real-time ingestion capabilities.
*   **Frontend/Visualization:** Consists of static HTML files and a Python script for map generation (`visualization/interactive_map.html`, `risk_map.html`, `map.py`), rather than a dynamic web application.
*   **Infrastructure:** Runs on the built-in Flask development server (`app.run(debug=True)`). No database, authentication, or containerization.

---

## 2. Technical Debt & Critical Vulnerabilities

Before scaling to production, several major architectural and code-level issues must be addressed:

*   **Blocking ML Inference:** Loading and executing TensorFlow models inside a synchronous Flask worker blocks the thread. Under concurrent load, the application will hang or crash.
*   **Zero Validation & Error Handling:** The `/predict_risk` endpoint blindly accepts JSON data without schema validation. Missing keys or incorrect data types will trigger ungraceful 500 exceptions.
*   **Hardcoded Configuration:** File paths (`../../ml_models/...`) are hardcoded, making the application brittle and deployment extremely difficult across different environments.
*   **Lack of Persistence:** Without a database, user requests, past predictions, and crowdsourced feedback are lost immediately.
*   **Synthetic Reliance:** The model cannot generalize to the real world until it is retrained on actual crime, lighting, and environmental datasets.
*   **Security Posture:** `CORS(app)` is overly permissive. There is no authentication mechanism, API rate limiting, or input sanitization.

---

## 3. Recommended Folder Structure

To support a scalable, microservices-oriented architecture, the repository should be restructured as follows:

```text
safeher-ai/
├── frontend/                 # Modern web application (React/Next.js/Vite)
│   ├── src/
│   │   ├── components/       # Map views, risk indicators, forms
│   │   ├── services/         # API integration clients
│   │   └── hooks/            # State management
│   └── package.json
├── backend/                  # API Gateway & Core Logic (FastAPI)
│   ├── app/
│   │   ├── api/              # API Endpoints (Routing)
│   │   ├── models/           # Database ORM models (SQLAlchemy)
│   │   ├── schemas/          # Data validation (Pydantic)
│   │   └── core/             # Security, CORS, config
│   ├── requirements.txt
│   └── Dockerfile
├── ml_service/               # Dedicated Model Inference Service
│   ├── serving/              # FastAPI/Triton inference wrapper
│   ├── training/             # Model training pipelines
│   └── models/               # Serialized model artifacts (.h5, .pkl)
├── data_pipeline/            # ETL scripts for real-world data ingestion
│   ├── extractors/           # Connectors to police/city APIs
│   └── processors/           # Data cleaning and feature engineering
├── deployment/               # Infrastructure as Code
│   ├── docker-compose.yml
│   └── k8s/
├── .env.example
├── .gitignore
└── README.md
```

---

## 4. Phase 1 MVP Architecture

The goal of Phase 1 is to establish a robust, deployable foundation using modern, production-ready technologies.

*   **Frontend Client:** A React (Vite) Single Page Application utilizing Mapbox GL JS or Leaflet for dynamic geospatial rendering.
*   **Core API (Backend):** Migrate from Flask to **FastAPI**. FastAPI provides native async support, automated OpenAPI documentation, and strict Pydantic validation for incoming requests.
*   **Database Engine:** **PostgreSQL** extended with **PostGIS** for efficient geospatial querying (e.g., finding nearby safe zones or historical incidents within a radius).
*   **Decoupled ML Serving:** Serve the Keras model in an isolated microservice (e.g., via a lightweight FastAPI wrapper or TensorFlow Serving). The Core API will communicate with the ML service over HTTP/gRPC.
*   **Orchestration:** Containerize all components using **Docker** and manage them locally and in staging via **Docker Compose**.

---

## 5. Migration Plan

A phased approach to smoothly transition from the current prototype to the MVP.

**Step 1: Containerization & Cleanup (Days 1-3)**
*   Write `Dockerfile`s for the existing Flask app.
*   Replace hardcoded paths in `app.py` with environment variables (`python-dotenv`).
*   Implement basic Pydantic or Marshmallow validation for the Flask endpoint.

**Step 2: Repository Restructuring & API Upgrade (Days 4-7)**
*   Reorganize folders according to the recommended structure.
*   Rewrite the Core API using FastAPI.
*   Set up PostgreSQL + PostGIS locally using Docker.
*   Implement SQLAlchemy ORM models and connect the FastAPI application to the database to log incoming prediction requests.

**Step 3: Decouple Machine Learning (Days 8-10)**
*   Extract the model loading and `predict()` logic into a standalone `ml_service`.
*   Update the Core API to make async HTTP requests to the `ml_service` rather than running inference locally.

**Step 4: Real Data Integration (Days 11-15)**
*   Identify 1-2 real-world open datasets (e.g., city crime data, street lighting).
*   Build basic Python extraction scripts in `data_pipeline/` to fetch, clean, and store this data in PostgreSQL.
*   Retrain the Keras model using the newly acquired real data and update the model artifacts.

**Step 5: Frontend Development (Days 16-21)**
*   Initialize a React/Vite project.
*   Build a responsive map component that fetches risk data from the new FastAPI backend.
*   Replace the static HTML visualization files.

**Step 6: Production Deployment Preparation (Days 22-25)**
*   Finalize `docker-compose.yml` encapsulating the Frontend, Core API, ML Service, and Database.
*   Configure a reverse proxy (Nginx or Traefik) to handle routing and SSL termination.
*   Implement basic GitHub Actions for CI (linting, basic testing).
