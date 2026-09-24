"""Phase 4 integration tests - end-to-end flow through the FastAPI surface.

These tests exercise the public API contract:
- POST /v1/query with stubbed completion_fn returns generated SQL + execution result.
- POST /v1/query with detect_hallucination=true returns hallucination_report.
- POST /v1/feedback records a verdict.
- GET /v1/history returns the original query with the feedback linked.
- Multiple queries in a session preserve order.

The pipeline is mocked at the `_run_pipeline_sync` boundary (it's a top-level
function in text2sql.api.app) so no real LLM, no real DB, no network.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from text2sql.api.app import (
    app,
    ExecutionResultModel,
    GuardrailDecisionModel,
    HallucinationReportModel,
    QueryResponse,
)
from text2sql.feedback.store import FeedbackStore

import sys

# IMPORTANT: `import text2sql.api.app as app_module` resolves to the FastAPI
# instance (because text2sql/api/__init__.py does `from text2sql.api.app import app`,
# which rebinds the `app` attribute on the package to the FastAPI instance,
# shadowing the submodule). To get the actual module object, fetch from sys.modules
# — this is the same pattern test_api.py uses.
app_module = sys.modules["text2sql.api.app"]


@pytest.fixture(autouse=True)
def _reset_app_state():
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


def _build_response(
    *,
    query_id: str,
    session_id: str,
    question: str,
    sql: str = "SELECT 42 AS answer",
    confidence: float = 0.85,
    detect: bool = False,
    flagged: bool = False,
):
    """Build a synthetic QueryResponse matching what the real pipeline returns."""
    return QueryResponse(
        query_id=query_id,
        session_id=session_id,
        question=question,
        sql=sql,
        confidence=confidence,
        guardrail=GuardrailDecisionModel(sql=sql, allowed=True, warnings=[]),
        execution=ExecutionResultModel(
            sql=sql,
            confidence=confidence,
            results=[{"answer": 42}],
            row_count=1,
            execution_time_ms=3,
            warnings=[],
            allowed=True,
        ),
        hallucination_report=(
            HallucinationReportModel(
                overall_score=92.0 if not flagged else 30.0,
                verdict="high" if not flagged else "low",
                flagged=flagged,
                outcomes={
                    "syntax": {"ran": True, "score": 1.0, "flagged": False},
                    "schema_coverage": {"ran": True, "score": 0.95, "flagged": False},
                },
                notes=[],
            )
            if detect
            else None
        ),
        created_at=datetime.utcnow(),
    )


def test_e2e_query_then_feedback_then_history(client, in_memory_store, monkeypatch):
    """End-to-end: ask -> record feedback -> history shows both linked."""
    qid = "q-e2e-1"
    sid = "sess-e2e"

    monkeypatch.setattr(
        app_module,
        "_run_pipeline_sync",
        lambda **kwargs: _build_response(
            query_id=qid, session_id=sid,
            question="What is the answer?",
            detect=kwargs.get("detect", False),
        ),
    )
    app.dependency_overrides[app_module.get_feedback_store] = lambda: in_memory_store

    # 1. Submit question
    r = client.post("/v1/query", json={"question": "What is the answer?", "session_id": sid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query_id"] == qid
    assert body["session_id"] == sid
    assert body["sql"] == "SELECT 42 AS answer"
    assert body["execution"]["results"] == [{"answer": 42}]
    assert body["execution"]["row_count"] == 1
    assert body["hallucination_report"] is None

    # 2. Record feedback
    r = client.post(
        "/v1/feedback",
        json={"query_id": qid, "session_id": sid, "verdict": "correct", "note": "spot on"},
    )
    assert r.status_code == 200, r.text
    fb = r.json()
    assert fb["query_id"] == qid
    assert fb["verdict"] == "correct"
    assert fb["note"] == "spot on"

    # 3. GET /v1/history returns the query with feedback linked
    r = client.get("/v1/history", params={"session_id": sid})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["query_id"] == qid
    assert items[0]["question"] == "What is the answer?"
    assert len(items[0]["feedback"]) == 1
    assert items[0]["feedback"][0]["verdict"] == "correct"
    assert items[0]["feedback"][0]["note"] == "spot on"


def test_e2e_query_with_detect_hallucination_flagged(client, in_memory_store, monkeypatch):
    """When detect_hallucination=true, response includes hallucination_report."""
    sid = "sess-detect"
    monkeypatch.setattr(
        app_module,
        "_run_pipeline_sync",
        lambda **kwargs: _build_response(
            query_id="q-detect", session_id=sid,
            question="weird question",
            detect=kwargs.get("detect", False),
            flagged=True,
        ),
    )
    app.dependency_overrides[app_module.get_feedback_store] = lambda: in_memory_store

    r = client.post(
        "/v1/query",
        json={"question": "weird question", "session_id": sid, "detect_hallucination": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hallucination_report"] is not None
    hr = body["hallucination_report"]
    assert hr["flagged"] is True
    assert hr["verdict"] == "low"
    assert hr["overall_score"] == 30.0
    assert "syntax" in hr["outcomes"]
    assert "schema_coverage" in hr["outcomes"]


def test_e2e_session_history_preserves_order(client, in_memory_store, monkeypatch):
    """Multiple queries in one session appear in submission order with separate feedback."""
    sid = "sess-multi"
    queries = [
        ("q-m-1", "first question", "SELECT 1", False),
        ("q-m-2", "second question", "SELECT 2", True),  # flagged
        ("q-m-3", "third question", "SELECT 3", False),
    ]
    counter = {"i": 0}

    def fake_pipeline(**kwargs):
        qid, q, sql, flagged = queries[counter["i"]]
        counter["i"] += 1
        return _build_response(
            query_id=qid, session_id=sid, question=q, sql=sql,
            detect=kwargs.get("detect", False), flagged=flagged,
        )

    monkeypatch.setattr(app_module, "_run_pipeline_sync", fake_pipeline)
    app.dependency_overrides[app_module.get_feedback_store] = lambda: in_memory_store

    # Submit 3 queries. q-m-2 enables hallucination detection so its
    # flagged report round-trips into /v1/history.
    for idx, (_, q, _, _) in enumerate(queries):
        payload = {"question": q, "session_id": sid}
        if idx == 1:
            payload["detect_hallucination"] = True
        r = client.post("/v1/query", json=payload)
        assert r.status_code == 200

    # Record feedback on 2 of them
    client.post("/v1/feedback", json={"query_id": "q-m-1", "session_id": sid, "verdict": "correct"})
    client.post("/v1/feedback", json={"query_id": "q-m-3", "session_id": sid, "verdict": "incorrect"})

    # History returns 3 items in order
    r = client.get("/v1/history", params={"session_id": sid})
    assert r.status_code == 200
    items = r.json()
    assert [it["query_id"] for it in items] == ["q-m-1", "q-m-2", "q-m-3"]

    # q-m-1 has feedback, q-m-2 has none, q-m-3 has feedback
    assert len(items[0]["feedback"]) == 1
    assert items[0]["feedback"][0]["verdict"] == "correct"
    assert items[1]["feedback"] == []
    assert len(items[2]["feedback"]) == 1
    assert items[2]["feedback"][0]["verdict"] == "incorrect"

    # Flagged query's hallucination_report is preserved
    assert items[1]["hallucination_report"] is not None
    assert items[1]["hallucination_report"]["flagged"] is True


def test_e2e_sql_override_re_executes(client, in_memory_store, monkeypatch):
    """sql_override flows through to _run_pipeline_sync (used by Streamlit's re-execute button)."""
    sid = "sess-override"
    captured = {}

    def fake_pipeline(**kwargs):
        captured["sql_override"] = kwargs.get("sql_override")
        captured["detect"] = kwargs.get("detect")
        return _build_response(
            query_id="q-ov", session_id=sid,
            question="override me", sql=kwargs.get("sql_override") or "SELECT 0",
        )

    monkeypatch.setattr(app_module, "_run_pipeline_sync", fake_pipeline)
    app.dependency_overrides[app_module.get_feedback_store] = lambda: in_memory_store

    r = client.post(
        "/v1/query",
        json={"question": "override me", "session_id": sid, "sql_override": "SELECT 99"},
    )
    assert r.status_code == 200
    assert captured["sql_override"] == "SELECT 99"
    assert r.json()["sql"] == "SELECT 99"


def test_e2e_feedback_persists_across_requests(client, in_memory_store, monkeypatch):
    """Feedback recorded in one request is visible in /v1/history on a subsequent request."""
    sid = "sess-persist"
    monkeypatch.setattr(
        app_module,
        "_run_pipeline_sync",
        lambda **kwargs: _build_response(
            query_id="q-persist", session_id=sid, question="persist me",
        ),
    )
    app.dependency_overrides[app_module.get_feedback_store] = lambda: in_memory_store

    # Query
    r1 = client.post("/v1/query", json={"question": "persist me", "session_id": sid})
    assert r1.status_code == 200

    # Feedback
    r2 = client.post("/v1/feedback", json={
        "query_id": "q-persist", "session_id": sid, "verdict": "partially_correct", "note": "close but wrong col"
    })
    assert r2.status_code == 200

    # History (fresh request)
    r3 = client.get("/v1/history", params={"session_id": sid})
    assert r3.status_code == 200
    items = r3.json()
    assert len(items) == 1
    assert len(items[0]["feedback"]) == 1
    assert items[0]["feedback"][0]["verdict"] == "partially_correct"
    assert items[0]["feedback"][0]["note"] == "close but wrong col"
