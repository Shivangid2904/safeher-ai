"""
SafeHer AI — Milestone 2
backend/app.py

Entry point for the routing engine Flask application.

Usage:
    cd backend
    python app.py
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Load .env BEFORE any other local imports so env vars are available
load_dotenv()

# Make sure `backend/` is on the Python path when running directly
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask
from flask_cors import CORS

from api.routes import routing_bp
from core.routing_service import get_graph

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)  # Allow cross-origin requests from the frontend

    # Register blueprints
    app.register_blueprint(routing_bp)

    # ── Warm the graph cache at startup ──────────────────────────────────
    # This ensures the first HTTP request doesn't bear the graph-build cost.
    logger.info("Warming graph cache at startup…")
    get_graph()
    logger.info("Graph cache ready.")

    @app.get("/")
    def index():
        return {"service": "SafeHer Routing Engine", "milestone": 2, "status": "running"}

    return app


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
