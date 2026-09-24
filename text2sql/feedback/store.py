"""SQLite-backed feedback persistence for the Text-to-SQL pipeline.

Stores per-query human feedback (verdict + optional note) keyed by
``query_id`` and ``session_id``, and per-query execution records (the
generated SQL, result rows, and optional hallucination report) keyed by
``query_id`` so ``/v1/history`` can join them.

Uses stdlib ``sqlite3`` with ``check_same_thread=False`` and a
``threading.Lock`` so the same ``FeedbackStore`` instance is safe to
share across threads.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("text2sql.feedback.store")

Verdict = Literal["correct", "incorrect", "partially_correct"]


class FeedbackRecord(BaseModel):
    """One feedback entry persisted by FeedbackStore."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    session_id: str
    verdict: Verdict
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QueryRecord(BaseModel):
    """One executed query persisted by FeedbackStore.record_query()."""

    id: str
    session_id: str
    question: str
    generated_sql: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    hallucination_report: Optional[dict[str, Any]] = None
    created_at: datetime


class FeedbackStore:
    """SQLite-backed feedback + query persistence. Thread-safe."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS feedback (
        id          TEXT PRIMARY KEY,
        query_id    TEXT NOT NULL,
        session_id  TEXT NOT NULL,
        verdict     TEXT NOT NULL,
        note        TEXT,
        created_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
    CREATE INDEX IF NOT EXISTS idx_feedback_query   ON feedback(query_id);

    CREATE TABLE IF NOT EXISTS queries (
        id               TEXT PRIMARY KEY,
        session_id       TEXT NOT NULL,
        question         TEXT NOT NULL,
        generated_sql    TEXT NOT NULL,
        results_json     TEXT,
        columns_json     TEXT,
        row_count        INTEGER,
        hallucination_json TEXT,
        created_at       TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_queries_session ON queries(session_id);
    """

    def __init__(self, db_path: str | Path) -> None:
        # check_same_thread=False so the same connection can be touched across threads;
        # we still serialize with a lock for write atomicity.
        self._path = str(db_path)
        self._lock = threading.Lock()
        # Initialize schema on a single connection; keep it open for the store's lifetime.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Feedback (existing API — unchanged behaviour)
    # ------------------------------------------------------------------
    def record_feedback(
        self,
        query_id: str,
        session_id: str,
        verdict: Verdict,
        note: Optional[str] = None,
    ) -> FeedbackRecord:
        record = FeedbackRecord(query_id=query_id, session_id=session_id, verdict=verdict, note=note)
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback (id, query_id, session_id, verdict, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.query_id,
                    record.session_id,
                    record.verdict,
                    record.note,
                    record.created_at.isoformat(),
                ),
            )
            self._conn.commit()
        return record

    def list_feedback(self, session_id: str) -> list[FeedbackRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, query_id, session_id, verdict, note, created_at FROM feedback WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [
            FeedbackRecord(
                id=r[0],
                query_id=r[1],
                session_id=r[2],
                verdict=r[3],   # type: ignore[arg-type]
                note=r[4],
                created_at=datetime.fromisoformat(r[5]),
            )
            for r in rows
        ]

    def get_feedback(self, query_id: str) -> Optional[FeedbackRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, query_id, session_id, verdict, note, created_at FROM feedback WHERE query_id = ? ORDER BY created_at DESC LIMIT 1",
                (query_id,),
            ).fetchone()
        if row is None:
            return None
        return FeedbackRecord(
            id=row[0],
            query_id=row[1],
            session_id=row[2],
            verdict=row[3],   # type: ignore[arg-type]
            note=row[4],
            created_at=datetime.fromisoformat(row[5]),
        )

    # ------------------------------------------------------------------
    # Queries (Phase 4 extension)
    # ------------------------------------------------------------------
    def record_query(
        self,
        session_id: str,
        question: str,
        generated_sql: str,
        results: list[dict[str, Any]],
        columns: list[str],
        row_count: int,
        hallucination_report: Optional[dict[str, Any]] = None,
    ) -> str:
        """Persist one executed query; returns the assigned ``query_id``."""
        query_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        results_json = json.dumps(results)
        columns_json = json.dumps(columns)
        hallucination_json = (
            json.dumps(hallucination_report)
            if hallucination_report is not None
            else None
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO queries (id, session_id, question, generated_sql, "
                "results_json, columns_json, row_count, hallucination_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    query_id,
                    session_id,
                    question,
                    generated_sql,
                    results_json,
                    columns_json,
                    int(row_count),
                    hallucination_json,
                    created_at.isoformat(),
                ),
            )
            self._conn.commit()
        return query_id

    def list_queries(self, session_id: str) -> list[QueryRecord]:
        """Return all queries for ``session_id`` ordered by ``created_at`` ASC."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, question, generated_sql, results_json, "
                "columns_json, row_count, hallucination_json, created_at "
                "FROM queries WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [_row_to_query_record(r) for r in rows]

    def get_query(self, query_id: str) -> Optional[QueryRecord]:
        """Return one query by ``query_id`` or ``None`` when missing."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, session_id, question, generated_sql, results_json, "
                "columns_json, row_count, hallucination_json, created_at "
                "FROM queries WHERE id = ?",
                (query_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_query_record(row)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_query_record(row: tuple[Any, ...]) -> QueryRecord:
    """Translate a raw sqlite row into a typed QueryRecord."""
    (
        qid,
        session_id,
        question,
        generated_sql,
        results_json,
        columns_json,
        row_count,
        hallucination_json,
        created_at,
    ) = row
    return QueryRecord(
        id=qid,
        session_id=session_id,
        question=question,
        generated_sql=generated_sql,
        results=json.loads(results_json) if results_json else [],
        columns=json.loads(columns_json) if columns_json else [],
        row_count=int(row_count) if row_count is not None else 0,
        hallucination_report=(
            json.loads(hallucination_json) if hallucination_json else None
        ),
        created_at=datetime.fromisoformat(created_at),
    )


def get_store(db_path: str | Path = "./feedback.db") -> FeedbackStore:
    """Factory for the default file-backed store; tests pass ':memory:'."""
    return FeedbackStore(db_path)

