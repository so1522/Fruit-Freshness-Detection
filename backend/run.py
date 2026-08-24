import os
import sys
import webbrowser
import threading
import time

# Ensure project directory is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app

def open_browser():
    """Wait briefly for Flask server to boot then open default web browser."""
    time.sleep(1.5)
    url = "http://127.0.0.1:5000"
    print(f"\n[Launcher] Opening Fruit Freshness Web App in browser: {url}\n")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=" * 65)
    print("      Fruit Freshness Detection System - Web Server Launcher      ")
    print("=" * 65)
    print("  Backend API: http://127.0.0.1:5000/api/health")
    print("  Web Dashboard: http://127.0.0.1:5000")
    print("=" * 65)

    # Launch browser thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Flask Web Server
    app.run(host="127.0.0.1", port=5000, debug=False)
