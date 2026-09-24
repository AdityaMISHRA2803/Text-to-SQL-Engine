"""Guardrail middleware for SQL safety (README 2.2).

The `GuardrailValidator` enforces a layered set of checks on every SQL
statement before execution:

1. Parse failure (sqlparse.parse returns empty)
2. DDL: CREATE/ALTER/DROP/TRUNCATE
3. DML writes: INSERT/UPDATE/DELETE/MERGE/UPSERT/GRANT/REVOKE
4. Subquery depth > max_subquery_depth
5. EXPLAIN row-estimate > max_explain_rows
6. (LIMIT injection is best-effort, not blocking — appended silently)

Every blocked query emits a structured log record via stdlib logging
under the `text2sql.guardrails` logger.
"""
from text2sql.guardrails.validator import (
    REASON_DDL,
    REASON_DML_WRITE,
    REASON_EXPLAIN_ROWS,
    REASON_PARSE_FAILURE,
    REASON_SUBQUERY_DEPTH,
    GuardrailValidator,
    ValidationResult,
)

__all__ = [
    "GuardrailValidator",
    "ValidationResult",
    "REASON_DDL",
    "REASON_DML_WRITE",
    "REASON_EXPLAIN_ROWS",
    "REASON_PARSE_FAILURE",
    "REASON_SUBQUERY_DEPTH",
]
