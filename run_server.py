"""Standalone server launch script for Network Attack Forecasting Dashboard."""

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.app import create_app


def main():
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")

    print("=" * 80)
    print("  NETWORK ATTACK FORECASTING — WORLD MODEL SOC DASHBOARD")
    print("=" * 80)
    print(f"  [+] Localhost Access URL : http://localhost:{port}")
    print(f"  [+] Network IP Access    : http://{host}:{port}")
    print(f"  [+] API Documentation    : http://localhost:{port}/api/status")
    print("  [+] Mode                 : Fully Offline / Local Python Runtime")
    print("=" * 80)
    print("  Press Ctrl+C to terminate the server.\n")

    app = create_app()
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
