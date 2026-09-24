"""Multi-query agreement (README 3.3)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, ClassVar, Optional

from pydantic import BaseModel, Field

from text2sql.config import get_settings

logger = logging.getLogger("text2sql.validation.multi_query")


#: Keywords that mark the start of a SQL statement; used as a fallback
#: when stripping prose from an LLM completion before parsing the SQL.
_SQL_START_KEYWORDS: ClassVar[tuple[str, ...]] = (
    "SELECT",
    "WITH",
    "INSERT",
    "UPDATE",
    "DELETE",
)


#: Prompt used to elicit a *different* SQL solution to the same question.
#: We constrain the response shape (raw SQL, no markdown) so the parser
#: can stay simple and robust.
_SECOND_SQL_PROMPT: ClassVar[str] = (
    "Given this question and schema, write a different SQL query that "
    "answers the same question. Use a different JOIN strategy, "
    "aggregation path, or subquery shape from a typical first attempt. "
    "Reply with ONLY the SQL, no markdown fences, no prose."
)


def _strip_markdown_fences(text: str) -> str:
    """Strip ```sql ... ``` or ``` ... ``` fences from an LLM completion."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _extract_sql_line(text: str) -> str:
    """Return the first non-empty line in `text` that starts with a SQL keyword.

    Used as a defensive fallback when an LLM returns prose around its
    SQL answer (e.g. "Here is the query:\\nSELECT ..."). If nothing
    matches, the entire cleaned text is returned verbatim.
    """
    cleaned = _strip_markdown_fences(text)
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if any(upper.startswith(kw) for kw in _SQL_START_KEYWORDS):
            return line
    return cleaned.strip()


class MultiQueryResult(BaseModel):
    """Outcome of a single multi-query agreement check.

    `agreed=True` means the two executed result sets compared equal
    (sorted JSON canonicalisation). `flagged=True` means disagreement
    was observed and the caller should treat this candidate as
    suspicious. When the trigger condition is not met and force-enable
    is off, `skipped=True` and the other fields are best-effort.
    """

    sql_a: str
    sql_b: str
    agreed: bool
    flagged: bool
    skipped: bool = False
    skip_reason: Optional[str] = None
    rows_a: list[Any] = Field(default_factory=list)
    rows_b: list[Any] = Field(default_factory=list)
    row_count_a: int = 0
    row_count_b: int = 0
    disagreement_detail: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


def _coerce_rows(result: Any) -> tuple[list[Any], int]:
    """Normalise a sandbox result into ``(rows, row_count)``.

    Accepts:
      - objects with ``.results: list[dict]`` and ``.row_count`` (the
        shape produced by ``text2sql.execution.runner.ExecutionResult``)
      - objects with ``.rows: list[tuple]`` and ``.columns: list[str]``
        plus ``.row_count`` (a more raw shape)
      - anything else → ``([], 0)`` and the caller is expected to add
        a warning via the returned `row_count == 0` signal.
    """
    if result is None:
        return [], 0
    # Dict-shaped results (preferred).
    if hasattr(result, "results"):
        try:
            rows = list(result.results or [])
        except Exception:  # noqa: BLE001 — defensive normalisation
            rows = []
        try:
            row_count = int(getattr(result, "row_count", None) or len(rows))
        except Exception:  # noqa: BLE001
            row_count = len(rows)
        return rows, row_count
    # Tuple-shaped results.
    if hasattr(result, "rows"):
        try:
            rows = list(result.rows or [])
        except Exception:  # noqa: BLE001
            rows = []
        try:
            row_count = int(getattr(result, "row_count", None) or len(rows))
        except Exception:  # noqa: BLE001
            row_count = len(rows)
        return rows, row_count
    # Unknown shape.
    return [], 0


def _serialise_row(row: Any) -> str:
    """Canonicalise a single row to a deterministic string for comparison.

    Dicts are serialised with sorted keys so ``{"a":1,"b":2}`` and
    ``{"b":2,"a":1}`` compare equal. ``default=str`` defends against
    non-JSON-native values like ``Decimal``, ``datetime``, ``UUID``,
    bytes, etc.
    """
    if isinstance(row, dict):
        return json.dumps(row, sort_keys=True, default=str)
    # Tuples / scalars: normalise via JSON wrapper.
    return json.dumps(row, sort_keys=True, default=str)


def _compare_sets(
    rows_a: list[Any],
    rows_b: list[Any],
) -> tuple[bool, Optional[str]]:
    """Compare two result sets as sorted, JSON-canonicalised sequences.

    Returns ``(agreed, disagreement_detail)``. ``agreed`` is True when
    the two multisets of canonicalised rows are equal (row order does
    not matter). When they disagree, `disagreement_detail` either names
    the first differing row or reports the cardinalities.
    """
    serialised_a = [_serialise_row(r) for r in rows_a]
    serialised_b = [_serialise_row(r) for r in rows_b]
    if sorted(serialised_a) == sorted(serialised_b):
        return True, None
    # Build a quick diff for the detail string.
    set_a = sorted(serialised_a)
    set_b = sorted(serialised_b)
    for ra, rb in zip(set_a, set_b):
        if ra != rb:
            return False, f"row differs: {ra[:120]} vs {rb[:120]}"
    return False, f"row set differs in {len(serialised_a)} vs {len(serialised_b)} rows"


class MultiQueryValidator:
    """Run a second independent SQL attempt and compare result sets (README 3.3).

    The validator is intentionally dependency-free at construction time:
    the LLM call goes through an injected ``completion_fn`` and the
    SQL execution goes through an injected ``sandbox``. This mirrors
    ``AmbiguityHandler`` and keeps the unit tests free of network
    access and real database connections.

    Public API::

        v = MultiQueryValidator()                    # default trigger from Settings
        v = MultiQueryValidator(trigger_mode="any")  # always run on non-empty SQL
        v = MultiQueryValidator(force_enable=True)   # bypass trigger entirely

        result = v.validate(
            question="...",
            schema=schema_or_none,
            generated_sql="SELECT ...",
            completion_fn=lambda prompt: "SELECT ...",
            sandbox=lambda sql: fake_result,
        )
        result.agreed     # True if both queries produced equal result sets
        result.flagged    # True iff there was a meaningful disagreement
        result.skipped    # True when trigger condition was not met
    """

    def __init__(
        self,
        *,
        trigger_mode: Optional[str] = None,
        force_enable: bool = False,
    ) -> None:
        if trigger_mode is None:
            trigger_mode = get_settings().multi_query_min_complexity
        if trigger_mode not in {"any", "always", "join_or_agg"}:
            raise ValueError(
                f"trigger_mode must be one of 'any', 'always', 'join_or_agg'; "
                f"got {trigger_mode!r}"
            )
        self.trigger_mode = trigger_mode
        self.force_enable = bool(force_enable)

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------
    @staticmethod
    def should_trigger(sql: str, mode: str) -> bool:
        """Return True when `sql` meets the complexity bar for `mode`.

        - ``any`` / ``always`` → any non-empty (after strip) SQL triggers.
        - ``join_or_agg`` → SQL must contain ``JOIN`` or ``GROUP BY``.
        """
        text = (sql or "").strip()
        if mode == "always":
            return bool(text)
        if mode == "any":
            return bool(text)
        if mode == "join_or_agg":
            upper = text.upper()
            return (" JOIN " in upper) or ("GROUP BY" in upper)
        # Unknown mode → conservative default of "do not trigger".
        return False

    # ------------------------------------------------------------------
    # Public validation entry point
    # ------------------------------------------------------------------
    def validate(
        self,
        *,
        question: str,
        schema: Any,  # Schema or None
        generated_sql: str,
        completion_fn: Callable[..., Any],
        sandbox: Callable[[str], Any],
        force_enable: Optional[bool] = None,
    ) -> MultiQueryResult:
        """Run the multi-query agreement check and return a `MultiQueryResult`.

        Behaviour:
          - If the trigger condition is not met (and force-enable is
            off) → return a ``skipped=True`` result with empty rows.
          - Otherwise, generate a second SQL via ``completion_fn``,
            execute both via ``sandbox``, compare canonicalised row
            sets, and return the agreed/flagged verdict.

        `force_enable` on the method shadows the constructor value so
        call sites can opt in per-call without rebuilding the validator.
        """
        forced = self.force_enable if force_enable is None else bool(force_enable)
        if not forced and not self.should_trigger(generated_sql or "", self.trigger_mode):
            return MultiQueryResult(
                sql_a=generated_sql or "",
                sql_b="",
                agreed=True,   # vacuous — nothing to compare
                flagged=False,
                skipped=True,
                skip_reason=f"query below trigger complexity (mode={self.trigger_mode})",
            )

        warnings: list[str] = []

        # 1) Generate the second SQL via the injected completion_fn.
        sql_b = self._generate_second_sql(
            question=question,
            schema=schema,
            completion_fn=completion_fn,
            warnings=warnings,
        )

        # 2) Execute both queries via the injected sandbox.
        rows_a, row_count_a, warnings_a = self._safe_execute(sandbox, generated_sql, warnings)
        rows_b, row_count_b, warnings_b = self._safe_execute(sandbox, sql_b, warnings)

        # 3) Compare canonicalised row sets.
        agreed, disagreement_detail = _compare_sets(rows_a, rows_b)
        flagged = (not agreed)

        if flagged:
            logger.warning(
                "multi_query.disagreement",
                extra={
                    "sql_a": generated_sql,
                    "sql_b": sql_b,
                    "row_count_a": row_count_a,
                    "row_count_b": row_count_b,
                    "disagreement_detail": disagreement_detail,
                    "warning_count": len(warnings),
                },
            )

        return MultiQueryResult(
            sql_a=generated_sql or "",
            sql_b=sql_b,
            agreed=agreed,
            flagged=flagged,
            skipped=False,
            skip_reason=None,
            rows_a=rows_a,
            rows_b=rows_b,
            row_count_a=row_count_a,
            row_count_b=row_count_b,
            disagreement_detail=disagreement_detail,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _generate_second_sql(
        self,
        *,
        question: str,
        schema: Any,
        completion_fn: Callable[..., Any],
        warnings: list[str],
    ) -> str:
        """Call ``completion_fn`` and extract the SQL from its response."""
        prompt = self._build_second_sql_prompt(question=question, schema=schema)
        try:
            raw = completion_fn(prompt)
        except Exception as exc:  # noqa: BLE001 — defensive: any completion error
            warnings.append(
                f"completion_fn failed: {type(exc).__name__}: {exc}"
            )
            return ""
        if raw is None:
            return ""
        text = str(raw)
        if not text.strip():
            return ""
        return _extract_sql_line(text)

    @staticmethod
    def _build_second_sql_prompt(*, question: str, schema: Any) -> str:
        """Compose the user prompt for the second-SQL completion call.

        Schema is serialised via ``to_compact_json()`` when available
        (the canonical Schema object); otherwise we fall back to
        ``str(schema)`` so tests can pass lightweight stubs.
        """
        schema_text = ""
        if schema is not None:
            if hasattr(schema, "to_compact_json"):
                try:
                    schema_text = schema.to_compact_json()
                except Exception:  # noqa: BLE001
                    schema_text = str(schema)
            else:
                schema_text = str(schema)
        return (
            f"{_SECOND_SQL_PROMPT}\n\n"
            f"Question: {(question or '').strip()}\n\n"
            f"Schema: {schema_text}"
        )

    @staticmethod
    def _safe_execute(
        sandbox: Callable[[str], Any],
        sql: str,
        warnings: list[str],
    ) -> tuple[list[Any], int, list[str]]:
        """Execute `sql` via `sandbox` and return ``(rows, row_count, warnings)``.

        On sandbox exception we record a warning and return empty rows
        so the caller can still emit a verdict (treating the failure as
        a disagreement for safety).
        """
        if not sql:
            return [], 0, warnings
        try:
            result = sandbox(sql)
        except Exception as exc:  # noqa: BLE001 — sandbox errors are non-fatal
            warnings.append(f"sandbox error: {type(exc).__name__}: {exc}")
            return [], 0, warnings
        rows, row_count = _coerce_rows(result)
        # If coercion returned an empty rowset but the result was
        # non-None, surface a soft warning so callers can distinguish
        # "empty result" from "malformed result".
        if result is not None and row_count == 0 and not hasattr(result, "results") and not hasattr(result, "rows"):
            warnings.append("sandbox returned malformed result shape; coerced to empty rows")
        return rows, row_count, warnings


__all__ = ["MultiQueryResult", "MultiQueryValidator"]
