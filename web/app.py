"""Flask application factory and server entry point."""

import os
from flask import Flask, send_from_directory
try:
    from flask_cors import CORS
except ImportError:
    class CORS:
        def __init__(self, *args, **kwargs):
            pass

from web.routes import api_bp


def create_app() -> Flask:
    """Create and configure the Flask web application."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "static")

    app = Flask(
        __name__,
        static_folder=static_dir,
        static_url_path="",
    )

    # Increase max upload size to 500 MB (comfortably handles 300 MB CSV files)
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

    # Enable Cross-Origin Resource Sharing for API routes
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint
    app.register_blueprint(api_bp)

    # Serve the single-page frontend application
    @app.route("/")
    def serve_index():
        return send_from_directory(app.static_folder, "index.html")

    # Serve any static asset requested directly
    @app.route("/<path:path>")
    def serve_static(path):
        if os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Network Attack Forecasting Web Dashboard on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
