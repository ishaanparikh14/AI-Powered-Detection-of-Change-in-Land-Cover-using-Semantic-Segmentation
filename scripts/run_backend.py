"""
run_backend.py — Convenience launcher for the FastAPI backend.

Usage:
    python run_backend.py

This is equivalent to:
    cd backend && uvicorn main:app --reload --port 8001
but works from the project root on Windows.
"""

import os
import sys
import subprocess

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")

sys.path.insert(0, ROOT)

if __name__ == "__main__":
    print("=" * 60)
    print("  Western Ghats Deforestation Monitor — Backend")
    print("=" * 60)
    print(f"  API:  http://localhost:8001")
    print(f"  Docs: http://localhost:8001/docs")
    print("=" * 60)

    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--log-level", "info",
        ],
        cwd=ROOT,
    )
