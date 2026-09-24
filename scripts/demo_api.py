"""Standalone demo API server with stubbed pipeline + in-memory feedback store.

Used by ``scripts/demo.sh`` to launch a self-contained Phase 4 demo that
requires no database, no LLM API key, and no SQLite file on disk.

What it does:
  1. Imports ``text2sql.api.app`` so all routes, Pydantic schemas, history
     state, and CORS middleware are wired exactly as in production.
  2. Monkey-patches ``_run_pipeline_sync`` at module level to a deterministic
     stub that returns a fully-formed ``QueryResponse``-shaped dict.
  3. Overrides ``get_feedback_store`` to a temp-file-backed ``FeedbackStore``
     (so feedback is real SQLite under the hood, just not in the repo).
  4. Serves the patched app via ``uvicorn`` on 127.0.0.1:8000.

Run: ``python scripts/demo_api.py`` (typically invoked by ``scripts/demo.sh``).
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

# Make the repo root importable so ``text2sql`` resolves when this script
# is invoked directly: ``python scripts/demo_api.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import uvicorn  # noqa: E402

# NOTE: ``import text2sql.api.app as api_module`` does NOT give us the module.
# ``text2sql/api/__init__.py`` does ``from text2sql.api.app import app``, which
# binds the FastAPI instance (not the module) to the ``app`` attribute of the
# ``text2sql.api`` package. So ``text2sql.api.app`` resolves to the FastAPI
# instance, not the source module. ``importlib.import_module`` is the only
# safe path that returns the actual source module (it ignores package-level
# attribute shadowing).
import importlib  # noqa: E402

api_module = importlib.import_module("text2sql.api.app")
QueryResponse = api_module.QueryResponse  # Pydantic model — endpoint expects this type back
from text2sql.feedback.store import FeedbackStore  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [demo_api] %(message)s")
log = logging.getLogger("demo_api")


def _stub_pipeline(question: str, sql_override, detect, **kwargs) -> QueryResponse:
    """Deterministic ``QueryResponse`` for the demo. Avoids any DB / LLM call.

    Returns a real ``QueryResponse`` instance (not a dict) so the endpoint's
    ``_record_query_in_history`` helper can access ``response.session_id`` etc.
    via attribute access.
    """
    sid = kwargs.get("session_id", "demo")
    if sql_override:
        sql = sql_override
        source = "override"
    else:
        sql = "SELECT COUNT(*) AS user_count FROM users"
        if "list" in question.lower() or "name" in question.lower():
            sql = "SELECT id, name FROM users LIMIT 10"
        source = "stub"
    payload = {
        "query_id": str(uuid.uuid4()),
        "session_id": sid,
        "question": question,
        "sql": sql,
        "confidence": 0.92,
        "guardrail": {
            "sql": sql,
            "allowed": True,
            "warnings": [],
            "blocked_reason": None,
        },
        "execution": {
            "sql": sql,
            "confidence": 0.92,
            "results": [
                {"id": 1, "name": "alice", "user_count": 42},
                {"id": 2, "name": "bob", "user_count": 17},
            ],
            "row_count": 2,
            "execution_time_ms": 7,
            "warnings": [],
            "allowed": True,
            "blocked_reason": None,
        },
        "hallucination_report": {
            "overall_score": 88.5,
            "verdict": "high",
            "flagged": False,
            "outcomes": {
                "back_translation": {
                    "name": "back_translation",
                    "ran": True,
                    "score": 0.91,
                    "flagged": False,
                    "detail": "back-translated question matches input",
                },
                "sanity": {
                    "name": "sanity",
                    "ran": True,
                    "score": 1.0,
                    "flagged": False,
                    "detail": "no anomalies detected",
                },
            },
            "notes": [f"demo stub ({source}); full pipeline not exercised"],
        } if detect else None,
        "created_at": datetime.utcnow(),
    }
    return QueryResponse(**payload)


# Real SQLite feedback store on a temp file (cleaned at exit).
_TMP_DB_FH = tempfile.NamedTemporaryFile(suffix=".db", delete=False, prefix="phase4_demo_feedback_")
_TMP_DB_FH.close()
_TMP_DB_PATH = _TMP_DB_FH.name
_DEMO_STORE = FeedbackStore(db_path=_TMP_DB_PATH)


def _override_feedback_store() -> FeedbackStore:
    return _DEMO_STORE


# Apply patches BEFORE uvicorn starts handling requests.
api_module._run_pipeline_sync = _stub_pipeline
api_module.get_feedback_store = _override_feedback_store


def _cleanup(signum=None, frame=None) -> None:
    try:
        _DEMO_STORE.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        os.unlink(_TMP_DB_PATH)
    except FileNotFoundError:
        pass
    if signum is not None:
        # Re-raise default disposition so callers see the signal.
        sys.exit(0)


signal.signal(signal.SIGTERM, _cleanup)
signal.signal(signal.SIGINT, _cleanup)


if __name__ == "__main__":
    log.warning("stubbed pipeline ready; feedback DB = %s", _TMP_DB_PATH)
    uvicorn.run(api_module.app, host="127.0.0.1", port=8000, log_level="warning")
