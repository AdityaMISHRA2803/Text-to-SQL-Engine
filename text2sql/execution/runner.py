"""Query execution runner (README 2.4).

Composes: validate (GuardrailValidator) -> sandbox.execute (Sandbox) -> audit (AuditLog).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import text

from text2sql.execution.audit import AuditLog
from text2sql.execution.sandbox import Sandbox
from text2sql.generation import GeneratedSQL
from text2sql.guardrails import GuardrailValidator


class ExecutionResult(BaseModel):
    """Result of executing one validated query."""

    sql: str
    confidence: float
    results: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: int = 0
    explain_plan: Optional[dict[str, Any]] = None
    warnings: list[str] = Field(default_factory=list)
    allowed: bool = True
    blocked_reason: Optional[str] = None


class QueryRunner:
    """Run generated SQL through guardrails + sandbox + audit log."""

    def __init__(
        self,
        *,
        validator: GuardrailValidator,
        sandbox: Sandbox,
        audit_path: Path | str = Path("audit/queries.jsonl"),
        explain: bool = False,
    ) -> None:
        self.validator = validator
        self.sandbox = sandbox
        self.audit = AuditLog(audit_path)
        self.explain = explain

    def run(self, generated: GeneratedSQL, question: str) -> ExecutionResult:
        """Validate -> execute (sandboxed) -> log -> return ExecutionResult.

        On validation failure, returns ExecutionResult with row_count=0,
        results=[], blocked_reason set, allowed=False. Audit log still gets
        a line so blocked queries are recorded.

        On runtime DB error, returns ExecutionResult with row_count=0,
        results=[], warnings=[str(exc)], allowed=True (it passed guardrails).
        """
        sql = generated.sql
        # 1) Validate (Layer 1 + Layer 2)
        validation = self.validator.validate(sql, provider="query_runner")
        if not validation.allowed:
            result = ExecutionResult(
                sql=sql,
                confidence=generated.confidence,
                allowed=False,
                blocked_reason=validation.blocked_reason,
                warnings=validation.warnings,
            )
            self._audit(question, generated, result, validation)
            return result

        # 2) Use the (possibly LIMIT-injected) sql from validation
        sql_to_run = validation.sql

        # 3) Execute via sandbox (ALWAYS rolls back)
        start = time.perf_counter()
        try:
            with self.sandbox.connection() as conn:
                rows = conn.execute(text(sql_to_run)).fetchall()
                elapsed_ms = int((time.perf_counter() - start) * 1000)

                # 4) Convert rows to list of dicts
                results = [dict(row._mapping) for row in rows]

                # 5) Optional EXPLAIN plan (do not rollback - in same tx; ok)
                explain_plan: Optional[dict[str, Any]] = None
                if self.explain:
                    plan_rows = conn.execute(
                        text(f"EXPLAIN (FORMAT JSON) {sql_to_run}")
                    ).fetchall()
                    if plan_rows:
                        explain_plan = json.loads(plan_rows[0][0])
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result = ExecutionResult(
                sql=sql,
                confidence=generated.confidence,
                results=[],
                row_count=0,
                execution_time_ms=elapsed_ms,
                warnings=[f"{type(exc).__name__}: {exc}"],
                allowed=True,
                blocked_reason=None,
            )
            self._audit(question, generated, result, validation)
            return result

        result = ExecutionResult(
            sql=sql,
            confidence=generated.confidence,
            results=results,
            row_count=len(results),
            execution_time_ms=elapsed_ms,
            explain_plan=explain_plan,
            warnings=validation.warnings,
            allowed=True,
            blocked_reason=None,
        )
        self._audit(question, generated, result, validation)
        return result

    def _audit(
        self,
        question: str,
        generated: GeneratedSQL,
        result: ExecutionResult,
        validation: Any,
    ) -> None:
        self.audit.append({
            "question": question,
            "sql": result.sql,
            "confidence": generated.confidence,
            "tables_touched": generated.tables_touched,
            "columns_touched": generated.columns_touched,
            "row_count": result.row_count,
            "execution_time_ms": result.execution_time_ms,
            "allowed": result.allowed,
            "blocked_reason": result.blocked_reason,
            "validation_warnings": validation.warnings,
        })


__all__ = ["ExecutionResult", "QueryRunner"]
