"""HTTP API (FastAPI) for the Text-to-SQL service (Phase 4).

The package exposes the FastAPI ``app`` from :mod:`text2sql.api.app`.
Endpoints:

- ``GET  /health``           — liveness probe.
- ``POST /v1/query``         — generate SQL + validate + execute (+ optional detect).
- ``GET  /v1/schema``        — return the database schema as JSON.
- ``GET  /v1/history``       — list queries + latest feedback for a session.
- ``POST /v1/feedback``      — record human verdict for a previous query.
"""
from text2sql.api.app import app

__all__ = ["app"]
