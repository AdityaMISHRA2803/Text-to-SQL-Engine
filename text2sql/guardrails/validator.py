"""GuardrailValidator: layered safety checks for generated SQL.

Design (see README 2.2):
    1. sqlparse.parse must yield >=1 statement
    2. No DDL tokens (CREATE/ALTER/DROP/TRUNCATE) outside string literals
    3. No DML write tokens (INSERT/UPDATE/DELETE/MERGE/UPSERT/GRANT/REVOKE)
       outside string literals
    4. Parenthesised subquery depth <= max_subquery_depth
    5. EXPLAIN (FORMAT JSON) row-estimate sum <= max_explain_rows
    6. If no LIMIT clause, append "LIMIT {limit_cap}" before the terminal
       semicolon (preserving any trailing comment)

String-literal handling: we strip single-quoted strings (with `''` escape)
before tokenising, so a string like `'DROP TABLE foo'` is not flagged as DDL.

Structured logging: every block emits a record via stdlib logging
at the `text2sql.guardrails` logger (WARNING level). The record payload is
{sql, reason, timestamp, provider, schema_hash}. Tests assert log
emission via `caplog`.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import sqlparse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger("text2sql.guardrails")

# Reason constants — kept as module-level strings so tests can assert exact
# values without string-typing into the source.
REASON_PARSE_FAILURE = "parse_failure"
REASON_DDL = "ddl"
REASON_DML_WRITE = "dml_write"
REASON_SUBQUERY_DEPTH = "subquery_depth"
REASON_EXPLAIN_ROWS = "explain_rows"

# Token-level detection: word-boundary regex, case-insensitive, applied AFTER
# stripping string literals.
_TOKEN_RE = re.compile(r"\b[A-Z_][A-Z_0-9]*\b", re.IGNORECASE)
_STRING_RE = re.compile(r"'(?:''|[^'])*'")  # SQL single-quoted string with '' escape

_DDL_TOKENS = frozenset({"CREATE", "ALTER", "DROP", "TRUNCATE"})
_DML_WRITE_TOKENS = frozenset({"INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "GRANT", "REVOKE"})


class ValidationResult(BaseModel):
    """Outcome of `GuardrailValidator.validate`."""

    sql: str
    allowed: bool
    warnings: list[str] = Field(default_factory=list)
    blocked_reason: Optional[str] = None

    def to_block_record(self, *, provider: str = "unknown", schema_hash: str = "") -> dict[str, Any]:
        """Return the structured-log payload for blocked queries."""
        return {
            "sql": self.sql,
            "reason": self.blocked_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "schema_hash": schema_hash,
        }


class GuardrailValidator:
    """Layered SQL safety checks. Construct once, call `.validate()` per query."""

    def __init__(
        self,
        *,
        limit_cap: int = 1000,
        max_subquery_depth: int = 3,
        max_explain_rows: int = 100_000,
        statement_timeout_ms: int = 30_000,
        explain_runner: Optional[Callable[[str], int]] = None,
        log_blocked: bool = True,
    ) -> None:
        if limit_cap <= 0:
            raise ValueError("limit_cap must be > 0")
        if max_subquery_depth < 1:
            raise ValueError("max_subquery_depth must be >= 1")
        if max_explain_rows <= 0:
            raise ValueError("max_explain_rows must be > 0")
        if statement_timeout_ms <= 0:
            raise ValueError("statement_timeout_ms must be > 0")

        self.limit_cap = limit_cap
        self.max_subquery_depth = max_subquery_depth
        self.max_explain_rows = max_explain_rows
        self.statement_timeout_ms = statement_timeout_ms
        self._explain_runner = explain_runner
        self.log_blocked = log_blocked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(
        self,
        sql: str,
        *,
        schema: Any = None,
        engine: Optional[Engine] = None,
        provider: str = "unknown",
    ) -> ValidationResult:
        """Run all checks; return ValidationResult (allowed=True if safe)."""
        if not sql or not sql.strip():
            return self._block(sql or "", REASON_PARSE_FAILURE, "SQL is empty", provider, schema)

        # 1. Parse check
        parsed = sqlparse.parse(sql)
        if not parsed or all(str(s).strip() == "" for s in parsed):
            return self._block(
                sql, REASON_PARSE_FAILURE, "sqlparse returned no statements", provider, schema
            )

        # Normalise: strip string literals so token-level checks don't false-positive.
        stripped = self._strip_string_literals(sql)

        # 2. DDL
        tokens = {t.upper() for t in _TOKEN_RE.findall(stripped)}
        ddl_hit = tokens & _DDL_TOKENS
        if ddl_hit:
            return self._block(sql, REASON_DDL, f"DDL token(s): {sorted(ddl_hit)}", provider, schema)

        # 3. DML writes
        dml_hit = tokens & _DML_WRITE_TOKENS
        if dml_hit:
            return self._block(
                sql, REASON_DML_WRITE, f"DML write token(s): {sorted(dml_hit)}", provider, schema
            )

        # 4. Subquery depth
        depth = self._walk_subquery_depth(stripped)
        if depth > self.max_subquery_depth:
            return self._block(
                sql, REASON_SUBQUERY_DEPTH,
                f"subquery depth {depth} > {self.max_subquery_depth}", provider, schema,
            )

        # 5. LIMIT injection (best-effort)
        limit_injected_sql, limit_warning = self._maybe_inject_limit(sql)
        warnings: list[str] = []
        if limit_warning:
            warnings.append(limit_warning)

        # 6. EXPLAIN-based row-estimate gate (only if not trivially small)
        try:
            runner = self._explain_runner or self._default_explain_runner(engine)
            estimate = runner(limit_injected_sql)
            if estimate > self.max_explain_rows:
                return self._block(
                    limit_injected_sql, REASON_EXPLAIN_ROWS,
                    f"EXPLAIN estimate {estimate} > {self.max_explain_rows}", provider, schema,
                )
        except Exception as exc:  # pragma: no cover - depends on engine
            warnings.append(f"EXPLAIN runner failed: {exc}")

        return ValidationResult(sql=limit_injected_sql, allowed=True, warnings=warnings)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _block(
        self, sql: str, reason: str, message: str, provider: str, schema: Any,
    ) -> ValidationResult:
        schema_hash = self._schema_hash(schema)
        if self.log_blocked:
            logger.warning(
                "guardrail_block reason=%s message=%s provider=%s schema_hash=%s sql=%s",
                reason, message, provider, schema_hash, sql,
            )
        return ValidationResult(
            sql=sql, allowed=False, warnings=[message], blocked_reason=reason,
        )

    @staticmethod
    def _strip_string_literals(sql: str) -> str:
        """Replace single-quoted strings with empty strings."""
        return _STRING_RE.sub("", sql)

    @staticmethod
    def _walk_subquery_depth(sql: str) -> int:
        """Walk the SQL and return the deepest (SELECT/WITH ...) nesting level."""
        s = re.sub(r"--[^\n]*", "", sql)
        upper = s.upper()
        max_subq = 0
        paren_stack: list[bool] = []  # True if the paren is SELECT/WITH-bearing
        i = 0
        while i < len(s):
            if s[i] == "(":
                rest = upper[i + 1:i + 30].lstrip()
                is_select = rest.startswith("SELECT") or rest.startswith("WITH")
                paren_stack.append(is_select)
                if is_select:
                    depth = sum(1 for x in paren_stack if x)
                    if depth > max_subq:
                        max_subq = depth
            elif s[i] == ")":
                if paren_stack:
                    paren_stack.pop()
            i += 1
        return max_subq

    @staticmethod
    def _has_limit_clause(stripped_sql: str) -> bool:
        return bool(re.search(r"\bLIMIT\s+\d+", stripped_sql, re.IGNORECASE))

    def _maybe_inject_limit(self, sql: str) -> tuple[str, Optional[str]]:
        """If SQL has no LIMIT clause, append 'LIMIT {limit_cap}' before ';'."""
        stripped = self._strip_string_literals(sql)
        if self._has_limit_clause(stripped):
            return sql, None
        # Preserve trailing comments and trailing semicolon.
        trailing_comment = ""
        body = sql.rstrip()
        m = re.search(r"(\s*--[^\n]*)$", body)
        if m:
            trailing_comment = m.group(1)
            body = body[: m.start()].rstrip()
        if body.endswith(";"):
            injected = body[:-1].rstrip() + f" LIMIT {self.limit_cap};" + trailing_comment
        else:
            injected = body + f" LIMIT {self.limit_cap};" + trailing_comment
        return injected, f"No LIMIT clause; injected LIMIT {self.limit_cap}"

    def _default_explain_runner(self, engine: Optional[Engine]) -> Callable[[str], int]:
        """Build an EXPLAIN runner from a SQLAlchemy engine, or a permissive fallback."""
        if engine is None:
            return lambda _sql: 0  # permissive: never block on missing engine

        def _run(sql: str) -> int:
            with engine.connect() as conn:
                conn.execute(text(f"SET LOCAL statement_timeout = {int(self.statement_timeout_ms)}"))
                result = conn.execute(text(f"EXPLAIN (FORMAT JSON) {sql}"))
                row = result.fetchone()
                if not row:
                    return 0
                plan = row[0]
                if isinstance(plan, str):
                    import json
                    plan = json.loads(plan)
                if isinstance(plan, list) and plan and isinstance(plan[0], dict):
                    return int(plan[0].get("Plan", {}).get("Plan Rows", 0) or 0)
                return 0

        return _run

    @staticmethod
    def _schema_hash(schema: Any) -> str:
        if schema is None:
            return ""
        try:
            text_repr = schema.to_compact_json()  # type: ignore[attr-defined]
        except Exception:
            text_repr = str(schema)
        return hashlib.sha256(text_repr.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "GuardrailValidator",
    "ValidationResult",
    "REASON_DDL",
    "REASON_DML_WRITE",
    "REASON_SUBQUERY_DEPTH",
    "REASON_EXPLAIN_ROWS",
    "REASON_PARSE_FAILURE",
]
