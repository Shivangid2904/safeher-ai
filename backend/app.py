"""
SafeHer AI — Milestone 3
backend/app.py

Entry point for the SafeHer routing + safe-havens Flask application.

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

from flask import Flask, send_from_directory
from flask_cors import CORS

from api.routes import routing_bp
from api.safe_havens import safe_havens_bp
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
# Paths
# ---------------------------------------------------------------------------
_BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIR = os.path.abspath(os.path.join(_BACKEND_DIR, "..", "frontend"))

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app)  # Allow cross-origin requests from the frontend

    # ── Register API Blueprints ───────────────────────────────────────────
    app.register_blueprint(routing_bp)
    app.register_blueprint(safe_havens_bp)

    # ── Warm the graph cache at startup ──────────────────────────────────
    # This ensures the first HTTP request doesn't bear the graph-build cost.
    logger.info("Warming graph cache at startup…")
    get_graph()
    logger.info("Graph cache ready.")

    # ── Serve the frontend ────────────────────────────────────────────────
    @app.get("/")
    def index():
        """Serve the Leaflet map frontend."""
        return send_from_directory(_FRONTEND_DIR, "index.html")

    # Allow serving other static assets from frontend/ if needed
    @app.get("/<path:filename>")
    def frontend_static(filename):
        return send_from_directory(_FRONTEND_DIR, filename)

    return app


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
