"""API tests for the Phase 4 text-to-SQL FastAPI service.

Covers /health, /docs, /openapi.json, POST /v1/query, POST /v1/feedback,
GET /v1/history, and CORS behaviour. Uses FastAPI TestClient and monkeypatches
the top-level ``_run_pipeline_sync`` in ``text2sql.api.app`` to avoid invoking
the real pipeline. Feedback tests inject an in-memory ``FeedbackStore`` via
``app.dependency_overrides`` for ``get_feedback_store``.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import sys

# Trigger the package import so `__init__.py` runs and the submodule
# `text2sql.api.app` ends up in sys.modules. Then grab the actual module
# (the FastAPI instance is what `text2sql.api.app` resolves to as an attribute,
# because the package re-exports it).
import text2sql.api  # noqa: F401  (side-effect: populates sys.modules)
app_module = sys.modules["text2sql.api.app"]
from text2sql.api.app import (
    ExecutionResultModel,
    FeedbackResponse,
    GuardrailDecisionModel,
    HallucinationReportModel,
    HistoryItem,
    QueryResponse,
    app,
    get_feedback_store,
)
from text2sql.feedback.store import FeedbackStore


@pytest.fixture(autouse=True)
def _reset_app_state():
    """Reset app.state.history and DI overrides between tests."""
    app.state.history = {}
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def in_memory_store():
    return FeedbackStore(":memory:")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _fake_query_response(query_id: str = "q1", session_id: str = "s1") -> QueryResponse:
    sql = "SELECT COUNT(*) FROM users"
    return QueryResponse(
        query_id=query_id,
        session_id=session_id,
        question="count users",
        sql=sql,
        confidence=0.9,
        guardrail=GuardrailDecisionModel(sql=sql, allowed=True),
        execution=ExecutionResultModel(
            sql=sql,
            confidence=0.9,
            results=[{"count": 42}],
            row_count=1,
        ),
        hallucination_report=None,
        created_at=datetime.utcnow(),
    )


def _fake_pipeline_factory(query_id: str, session_id: str):
    """Build a fake ``_run_pipeline_sync`` that returns a canned QueryResponse.

    History recording now happens in the endpoint via ``_record_query_in_history``,
    so the fake only needs to return the response — no history mutation here.
    """

    def _fake_pipeline(**kwargs):
        return _fake_query_response(query_id, session_id)

    return _fake_pipeline


# ----------------------------------------------------------------------
# /health + /docs + /openapi.json
# ----------------------------------------------------------------------
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_docs_render(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert "/v1/query" in spec["paths"]
    assert "/v1/schema" in spec["paths"]
    assert "/v1/history" in spec["paths"]
    assert "/v1/feedback" in spec["paths"]


def test_docs_html_renders(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower()


# ----------------------------------------------------------------------
# POST /v1/query
# ----------------------------------------------------------------------
def test_query_endpoint_happy_path(client, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_run_pipeline_sync",
        lambda **kwargs: _fake_query_response("qid-1", "sess-x"),
    )
    r = client.post("/v1/query", json={"question": "count users"})
    assert r.status_code == 200
    body = r.json()
    assert body["sql"] == "SELECT COUNT(*) FROM users"
    assert body["session_id"] == "sess-x"
    assert body["query_id"] == "qid-1"


def test_query_endpoint_with_sql_override(client, monkeypatch):
    captured: dict = {}

    def fake_pipeline(**kwargs):
        captured["sql_override"] = kwargs.get("sql_override")
        captured["detect"] = kwargs.get("detect")
        return _fake_query_response(query_id="qid-2")

    monkeypatch.setattr(app_module, "_run_pipeline_sync", fake_pipeline)
    r = client.post(
        "/v1/query",
        json={"question": "override me", "sql_override": "SELECT 1", "detect_hallucination": True},
    )
    assert r.status_code == 200
    assert captured["sql_override"] == "SELECT 1"
    assert captured["detect"] is True


def test_query_endpoint_assigns_session_id_when_missing(client, monkeypatch):
    captured: dict = {}

    def fake_pipeline(**kwargs):
        captured["session_id"] = kwargs.get("session_id")
        return _fake_query_response(
            query_id="qid-3", session_id=captured["session_id"]
        )

    monkeypatch.setattr(app_module, "_run_pipeline_sync", fake_pipeline)
    r = client.post("/v1/query", json={"question": "x"})
    assert r.status_code == 200
    sid = captured["session_id"]
    assert sid  # auto-generated UUID: truthy
    # A UUID4 is 36 chars including 4 hyphens.
    assert isinstance(sid, str)
    assert len(sid) >= 32


def test_query_endpoint_malformed_json_returns_422(client):
    r = client.post(
        "/v1/query",
        data="not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


def test_query_endpoint_missing_question_returns_422(client):
    r = client.post("/v1/query", json={})
    assert r.status_code == 422


def test_query_endpoint_generator_failure_returns_500(monkeypatch):
    # If _run_pipeline_sync raises, FastAPI returns 500. Use a TestClient with
    # raise_server_exceptions=False so the exception is reported as a 500
    # rather than re-raised in the test (the default behaviour is to raise).
    def boom(**kwargs):
        raise RuntimeError("simulated generator failure")

    monkeypatch.setattr(app_module, "_run_pipeline_sync", boom)
    tc = TestClient(app, raise_server_exceptions=False)
    r = tc.post("/v1/query", json={"question": "x"})
    assert r.status_code == 500  # unhandled exception → 500


# ----------------------------------------------------------------------
# POST /v1/feedback
# ----------------------------------------------------------------------
def test_feedback_endpoint_records(client, in_memory_store):
    app.dependency_overrides[app_module.get_feedback_store] = lambda: in_memory_store
    r = client.post(
        "/v1/feedback",
        json={
            "query_id": "qid-1",
            "session_id": "sess-1",
            "verdict": "correct",
            "note": "great",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["query_id"] == "qid-1"
    assert body["verdict"] == "correct"
    assert body["note"] == "great"
    assert body["session_id"] == "sess-1"
    assert body["id"]  # uuid assigned by store
    # Side effect: in-memory store now contains the record.
    listed = in_memory_store.list_feedback("sess-1")
    assert len(listed) == 1
    assert listed[0].verdict == "correct"


def test_feedback_endpoint_invalid_verdict_returns_422(client, in_memory_store):
    app.dependency_overrides[app_module.get_feedback_store] = lambda: in_memory_store
    r = client.post(
        "/v1/feedback",
        json={
            "query_id": "qid-1",
            "session_id": "sess-1",
            "verdict": "totally-wrong",
        },
    )
    assert r.status_code == 422
    # No record should have been written.
    assert in_memory_store.list_feedback("sess-1") == []


# ----------------------------------------------------------------------
# GET /v1/history
# ----------------------------------------------------------------------
def test_history_endpoint_empty(client):
    r = client.get("/v1/history", params={"session_id": "empty"})
    assert r.status_code == 200
    assert r.json() == []


def test_history_endpoint_returns_queries(client, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_run_pipeline_sync",
        _fake_pipeline_factory("qid-h", "sess-h"),
    )
    TestClient(app).post("/v1/query", json={"question": "x", "session_id": "sess-h"})
    r = client.get("/v1/history", params={"session_id": "sess-h"})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["query_id"] == "qid-h"
    assert items[0]["session_id"] == "sess-h"
    assert items[0]["sql"] == "SELECT COUNT(*) FROM users"
    assert items[0]["feedback"] == []


def test_history_endpoint_joins_feedback(client, in_memory_store, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_run_pipeline_sync",
        _fake_pipeline_factory("qid-j", "sess-j"),
    )
    app.dependency_overrides[app_module.get_feedback_store] = lambda: in_memory_store
    TestClient(app).post("/v1/query", json={"question": "x", "session_id": "sess-j"})
    client.post(
        "/v1/feedback",
        json={"query_id": "qid-j", "session_id": "sess-j", "verdict": "correct"},
    )
    r = client.get("/v1/history", params={"session_id": "sess-j"})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["query_id"] == "qid-j"
    assert len(items[0]["feedback"]) == 1
    assert items[0]["feedback"][0]["verdict"] == "correct"
    assert items[0]["feedback"][0]["query_id"] == "qid-j"


# ----------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------
def test_cors_allows_streamlit_origin(client):
    r = client.get("/health", headers={"origin": "http://localhost:8501"})
    assert r.status_code == 200
    header_keys = {k.lower() for k in r.headers.keys()}
    assert "access-control-allow-origin" in header_keys


# ----------------------------------------------------------------------
# Sanity: response model import surface
# ----------------------------------------------------------------------
def test_response_models_importable():
    # Touch each model so a future symbol removal shows up as an ImportError.
    assert FeedbackResponse is not None
    assert HistoryItem is not None
    assert ExecutionResultModel is not None
    assert GuardrailDecisionModel is not None
    assert HallucinationReportModel is not None
