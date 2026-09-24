"""FastAPI service fronting the Phase 1-3 text-to-SQL pipeline (README 4.1)."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

from text2sql.config import get_settings
from text2sql.execution import QueryRunner, Sandbox
from text2sql.feedback.store import FeedbackStore, get_store
from text2sql.generation import SQLGenerator, GeneratedSQL
from text2sql.guardrails import GuardrailValidator
from text2sql.llm.embedder import default_embedder, Embedder
from text2sql.relevance import RelevanceFilter
from text2sql.schema import extract_schema
from text2sql.validation.detector import HallucinationDetector, HallucinationReport

logger = logging.getLogger("text2sql.api")

app = FastAPI(title="Text-to-SQL API", version="0.1.0",
              description="REST interface over the Phase 1-3 text-to-SQL pipeline.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    sql_override: Optional[str] = None
    detect_hallucination: bool = False


class GuardrailDecisionModel(BaseModel):
    sql: str
    allowed: bool
    warnings: list[str] = Field(default_factory=list)
    blocked_reason: Optional[str] = None


class ExecutionResultModel(BaseModel):
    sql: str
    confidence: float
    results: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: int = 0
    warnings: list[str] = Field(default_factory=list)
    allowed: bool = True
    blocked_reason: Optional[str] = None


class HallucinationReportModel(BaseModel):
    overall_score: float
    verdict: str
    flagged: bool
    outcomes: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    query_id: str
    session_id: str
    question: str
    sql: str
    confidence: float
    guardrail: GuardrailDecisionModel
    execution: Optional[ExecutionResultModel] = None
    hallucination_report: Optional[HallucinationReportModel] = None
    created_at: datetime


class SchemaResponse(BaseModel):
    tables: list[dict[str, Any]] = Field(default_factory=list)
    table_names: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    query_id: str
    session_id: str
    verdict: str
    note: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    query_id: str
    session_id: str
    verdict: str
    note: Optional[str] = None
    created_at: datetime


class HistoryItem(BaseModel):
    query_id: str
    session_id: str
    question: str
    sql: str
    confidence: float
    execution: Optional[ExecutionResultModel] = None
    hallucination_report: Optional[HallucinationReportModel] = None
    feedback: list[FeedbackResponse] = Field(default_factory=list)
    created_at: datetime


def get_provider_settings() -> dict[str, Any]:
    s = get_settings()
    return {
        "provider": s.llm_provider,
        "model": s.llm_model,
        "database_url": s.database_url,
        "limit_cap": s.limit_cap,
        "max_subquery_depth": s.max_subquery_depth,
        "max_explain_rows": s.max_explain_rows,
        "statement_timeout_ms": s.statement_timeout_ms,
    }


def get_embedder() -> Embedder:
    s = get_settings()
    return default_embedder(provider=s.llm_provider)


def get_feedback_store() -> FeedbackStore:
    return get_store("./feedback.db")


def get_history() -> dict[str, list[HistoryItem]]:
    return app.state.history


@app.on_event("startup")
async def _init_state():
    if not hasattr(app.state, "history"):
        app.state.history = {}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/schema", response_model=SchemaResponse)
async def get_schema_endpoint():
    cfg = get_provider_settings()
    try:
        engine = create_engine(cfg["database_url"])
        schema = extract_schema(engine)
        engine.dispose()
    except Exception as e:
        logger.warning("schema_extraction_failed: %s: %s", type(e).__name__, e)
        return SchemaResponse(tables=[], table_names=[])
    table_names = schema.table_names() if hasattr(schema, "table_names") else []
    try:
        dumped = schema.model_dump() if hasattr(schema, "model_dump") else {}
        tables = dumped.get("tables", []) if isinstance(dumped, dict) else []
    except Exception:
        tables = []
    return SchemaResponse(tables=tables, table_names=table_names)


def _run_pipeline_sync(
    question: str,
    sql_override: Optional[str],
    detect: bool,
    *,
    completion_fn: Optional[Callable],
    cfg: dict[str, Any],
    embedder: Embedder,
    history: dict[str, list[HistoryItem]],
    session_id: str,
) -> QueryResponse:
    engine = create_engine(cfg["database_url"])
    schema = extract_schema(engine)
    engine.dispose()
    try:
        filtered = RelevanceFilter(embedder=embedder).filter(question, schema)
    except Exception as e:
        logger.warning("relevance_filter_failed: %s: %s", type(e).__name__, e)
        filtered = schema

    query_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    sql = sql_override
    confidence = 0.0
    generated: Optional[GeneratedSQL] = None
    if sql is None:
        gen = SQLGenerator(provider=cfg["provider"], model=cfg["model"], completion_fn=completion_fn)
        generated = gen.generate(question, filtered)
        sql = generated.sql
        confidence = generated.confidence

    validator = GuardrailValidator(
        limit_cap=cfg["limit_cap"],
        max_subquery_depth=cfg["max_subquery_depth"],
        max_explain_rows=cfg["max_explain_rows"],
        statement_timeout_ms=cfg["statement_timeout_ms"],
        log_blocked=False,
    )
    decision = validator.validate(sql, schema=filtered, provider="api")

    exec_model: Optional[ExecutionResultModel] = None
    exec_result = None
    if decision.allowed:
        sandbox = Sandbox(dsn=cfg["database_url"])
        runner = QueryRunner(validator=validator, sandbox=sandbox, audit_path=Path("audit/queries.jsonl"))
        if generated is None:
            gen_obj = GeneratedSQL(
                sql=sql, explanation="user-override", confidence=confidence or 0.5,
                tables_touched=[], columns_touched=[],
            )
        else:
            gen_obj = generated
        exec_result = runner.run(gen_obj, question)
        exec_model = ExecutionResultModel(
            sql=exec_result.sql,
            confidence=exec_result.confidence,
            results=list(exec_result.results or []),
            row_count=exec_result.row_count,
            execution_time_ms=exec_result.execution_time_ms,
            warnings=list(exec_result.warnings or []),
            allowed=exec_result.allowed,
            blocked_reason=exec_result.blocked_reason,
        )

    hr_model: Optional[HallucinationReportModel] = None
    if detect and exec_result is not None:
        try:
            def _sandbox_callable(s: str) -> Any:
                sub_runner = QueryRunner(
                    validator=validator,
                    sandbox=Sandbox(dsn=cfg["database_url"]),
                    audit_path=Path("audit/queries.jsonl"),
                )
                return sub_runner.run(s, question)

            det = HallucinationDetector(schema=filtered)
            hr: HallucinationReport = det.detect(
                question=question,
                generated_sql=sql,
                execution_result=exec_result,
                schema=filtered,
                embedder=embedder,
                completion_fn=completion_fn,
                sandbox=_sandbox_callable,
            )
            outcomes: dict[str, Any] = {}
            for k, v in (hr.outcomes or {}).items():
                if hasattr(v, "model_dump"):
                    outcomes[k] = v.model_dump()
                elif hasattr(v, "dict"):
                    outcomes[k] = v.dict()
                else:
                    outcomes[k] = str(v)
            hr_model = HallucinationReportModel(
                overall_score=float(hr.overall_score),
                verdict=str(hr.verdict),
                flagged=bool(hr.flagged),
                outcomes=outcomes,
                notes=list(hr.notes or []),
            )
        except Exception as e:
            logger.warning("hallucination_detection_error: %s: %s", type(e).__name__, e)
            hr_model = None

    guardrail_model = GuardrailDecisionModel(
        sql=decision.sql,
        allowed=decision.allowed,
        warnings=list(decision.warnings or []),
        blocked_reason=decision.blocked_reason,
    )
    resp = QueryResponse(
        query_id=query_id,
        session_id=session_id,
        question=question,
        sql=sql,
        confidence=confidence,
        guardrail=guardrail_model,
        execution=exec_model,
        hallucination_report=hr_model,
        created_at=created_at,
    )
    return resp


def _record_query_in_history(response: QueryResponse, history: dict[str, list[HistoryItem]]) -> None:
    """Append the response into the in-memory history dict keyed by session_id.

    Extracted from _run_pipeline_sync so that tests can stub _run_pipeline_sync
    without losing history recording.
    """
    history.setdefault(response.session_id, []).append(
        HistoryItem(
            query_id=response.query_id,
            session_id=response.session_id,
            question=response.question,
            sql=response.sql,
            confidence=response.confidence,
            execution=response.execution,
            hallucination_report=response.hallucination_report,
            created_at=response.created_at,
        )
    )


def get_completion_fn() -> Optional[Callable]:
    """Override this dependency in tests; default returns None (no stub)."""
    return None


@app.post("/v1/query", response_model=QueryResponse)
async def post_query(
    req: QueryRequest,
    completion_fn: Optional[Callable] = Depends(get_completion_fn),
    embedder: Embedder = Depends(get_embedder),
    history: dict = Depends(get_history),
):
    session_id = req.session_id or str(uuid.uuid4())
    cfg = get_provider_settings()
    response = _run_pipeline_sync(
        question=req.question,
        sql_override=req.sql_override,
        detect=req.detect_hallucination,
        completion_fn=completion_fn,
        cfg=cfg,
        embedder=embedder,
        history=history,
        session_id=session_id,
    )
    _record_query_in_history(response, history)
    return response


@app.post("/v1/feedback", response_model=FeedbackResponse)
async def post_feedback(req: FeedbackRequest, store: FeedbackStore = Depends(get_feedback_store)):
    if req.verdict not in ("correct", "incorrect", "partially_correct"):
        raise HTTPException(status_code=422, detail=f"invalid verdict: {req.verdict!r}")
    try:
        rec = store.record_feedback(
            query_id=req.query_id,
            session_id=req.session_id,
            verdict=req.verdict,
            note=req.note,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"feedback_record_failed: {e}")
    return FeedbackResponse(
        id=rec.id, query_id=rec.query_id, session_id=rec.session_id,
        verdict=rec.verdict, note=rec.note, created_at=rec.created_at,
    )


@app.get("/v1/history", response_model=list[HistoryItem])
async def get_history_endpoint(
    session_id: str = Query(...),
    history: dict = Depends(get_history),
    store: FeedbackStore = Depends(get_feedback_store),
):
    items = list(history.get(session_id, []))
    fb_by_qid: dict[str, list[FeedbackResponse]] = {}
    for f in store.list_feedback(session_id):
        fb_by_qid.setdefault(f.query_id, []).append(
            FeedbackResponse(
                id=f.id, query_id=f.query_id, session_id=f.session_id,
                verdict=f.verdict, note=f.note, created_at=f.created_at,
            )
        )
    for it in items:
        it.feedback = fb_by_qid.get(it.query_id, [])
    return items
