"""Hallucination detection orchestrator (README 3.5)."""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from text2sql.config import get_settings
from text2sql.scoring.scorer import ConfidenceScore, HallucinationScorer
from text2sql.validation.back_translator import (
    BackTranslationResult,
    BackTranslator,
)
from text2sql.validation.multi_query import MultiQueryResult, MultiQueryValidator
from text2sql.validation.sanity_checker import SanityCheckResult, SanityChecker

logger = logging.getLogger("text2sql.validation.detector")


#: Verdict threshold for "high" overall confidence.
_HIGH_THRESHOLD: float = 80.0
#: Verdict threshold for "moderate" overall confidence.
_MODERATE_THRESHOLD: float = 50.0


#: SQL keywords we exclude from identifier extraction when computing
#: schema coverage. Lower-cased.
_SQL_KEYWORDS: frozenset[str] = frozenset(
    {
        "select", "from", "where", "group", "by", "order", "join",
        "inner", "outer", "left", "right", "full", "on", "as", "and",
        "or", "not", "null", "true", "false", "limit", "offset",
        "having", "distinct", "all", "union", "intersect", "except",
        "case", "when", "then", "else", "end", "in", "is", "like",
        "ilike", "between", "exists", "any", "some", "count", "sum",
        "avg", "min", "max", "coalesce", "cast", "with", "recursive",
        "insert", "update", "delete", "merge", "into", "values", "set",
        "returning", "over", "partition", "rows", "range", "preceding",
        "following", "unbounded", "current", "row", "row_number", "rank",
        "dense_rank", "lead", "lag",
    }
)


#: Detect SQL statement opener, allowing leading whitespace and
#: single-line `--` comments before the keyword.
_SQL_START_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:--[^\n]*\n)*\s*(SELECT|WITH|INSERT|UPDATE|DELETE|MERGE)\b",
    re.IGNORECASE,
)


#: Match identifier-like tokens for schema coverage heuristics.
_IDENT_RE: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


#: Canonical signal names. Order is fixed so outcome iteration is stable.
_SIGNAL_NAMES: tuple[str, ...] = (
    "syntax",
    "back_translation",
    "sanity",
    "multi_query",
    "schema_coverage",
)


class SignalOutcome(BaseModel):
    """One per-signal outcome slot in HallucinationReport."""

    name: str
    ran: bool
    skipped_reason: Optional[str] = None
    score: float
    flagged: bool = False
    detail: str = ""


class HallucinationReport(BaseModel):
    """Aggregate hallucination-detection verdict for one SQL generation."""

    question: str
    generated_sql: str
    overall_score: float
    verdict: str  # "high" | "moderate" | "low"
    flagged: bool
    confidence: ConfidenceScore
    outcomes: dict[str, SignalOutcome] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _syntax_ok(sql: str) -> bool:
    """True when `sql` is non-empty and starts with a recognised SQL verb."""
    if not sql or not sql.strip():
        return False
    return bool(_SQL_START_RE.match(sql))


def _extract_identifiers(sql: str) -> set[str]:
    """Lower-case identifier tokens from `sql`, excluding SQL keywords.

    Pure regex extraction; we don't attempt to honour string literals or
    quoted identifiers. False positives are tolerable because they only
    inflate the candidate set (coverage becomes a lower bound).
    """
    if not sql:
        return set()
    raw = _IDENT_RE.findall(sql)
    out: set[str] = set()
    for tok in raw:
        low = tok.lower()
        if low in _SQL_KEYWORDS:
            continue
        out.add(low)
    return out


def _schema_token_set(schema: Any) -> set[str]:
    """Return the set of lowercase column/table names exposed by `schema`.

    Accepts:

    - An object with ``.tables: list[Table]`` where each ``Table`` has
      ``.name`` and ``.columns: list[Column]`` where each ``Column``
      has ``.name`` (the canonical ``Schema`` shape).
    - A ``{table_name: [col_name, ...]}`` mapping.
    - A flat iterable of strings (treated as already-resolved column
      names).

    Raises ``ValueError`` when none of these shapes apply.
    """
    if schema is None:
        raise ValueError("schema is None")

    # Canonical Schema / Table/Column object shape.
    tables = getattr(schema, "tables", None)
    if tables is not None:
        names: set[str] = set()
        for table in tables:
            t_name = getattr(table, "name", None)
            if t_name:
                names.add(str(t_name).lower())
            cols = getattr(table, "columns", None) or []
            for col in cols:
                c_name = getattr(col, "name", None)
                if c_name:
                    names.add(str(c_name).lower())
        return names

    # Dict-shaped: {table_name: [col_name, ...]} OR {table_name: {"columns": [...]}}.
    if isinstance(schema, dict):
        names = set()
        for key, value in schema.items():
            names.add(str(key).lower())
            if isinstance(value, dict):
                cols = value.get("columns") or value.get("cols") or []
                for col in cols:
                    if isinstance(col, str):
                        names.add(col.lower())
                    else:
                        c_name = getattr(col, "name", None) or (
                            str(col) if col is not None else None
                        )
                        if c_name:
                            names.add(str(c_name).lower())
            elif isinstance(value, (list, tuple, set)):
                for col in value:
                    if isinstance(col, str):
                        names.add(col.lower())
                    else:
                        c_name = getattr(col, "name", None) or (
                            str(col) if col is not None else None
                        )
                        if c_name:
                            names.add(str(c_name).lower())
            elif isinstance(value, str):
                names.add(value.lower())
        return names

    # Flat iterable of strings.
    if isinstance(schema, (list, tuple, set)):
        names = set()
        for item in schema:
            if isinstance(item, str):
                names.add(item.lower())
            else:
                c_name = getattr(item, "name", None)
                if c_name:
                    names.add(str(c_name).lower())
        return names

    raise ValueError(f"unsupported schema shape: {type(schema).__name__}")


def _map_verdict(score: float) -> str:
    """Map a 0-100 score to ``"high"`` / ``"moderate"`` / ``"low"``."""
    if score >= _HIGH_THRESHOLD:
        return "high"
    if score >= _MODERATE_THRESHOLD:
        return "moderate"
    return "low"


class HallucinationDetector:
    """Orchestrate the five hallucination signals into a single report.

    Public API::

        detector = HallucinationDetector()
        report = detector.detect(
            question="...",
            generated_sql="SELECT ...",
            execution_result=runner_result,   # or None
            schema=schema_or_none,
            embedder=my_embedder,             # optional
            completion_fn=my_llm_call,        # optional
            sandbox=lambda sql: ...,          # optional
        )
        report.verdict       # "high" | "moderate" | "low"
        report.overall_score # 0-100
        report.flagged       # True iff verdict == "low"
        report.outcomes      # per-signal SignalOutcome map
    """

    def __init__(
        self,
        *,
        scorer: Optional[HallucinationScorer] = None,
        back_translator: Optional[BackTranslator] = None,
        sanity_checker: Optional[SanityChecker] = None,
        multi_query_validator: Optional[MultiQueryValidator] = None,
        schema: Any = None,
    ) -> None:
        # Pull pass/warn thresholds from Settings so the scorer's verdict
        # boundaries stay aligned with the rest of the system.
        try:
            settings = get_settings()
        except Exception:  # noqa: BLE001 — defensive: Settings may not be ready
            settings = None

        self.scorer = scorer or HallucinationScorer()
        self.back_translator = back_translator or BackTranslator()
        self.sanity_checker = sanity_checker or SanityChecker()
        self.multi_query_validator = (
            multi_query_validator or MultiQueryValidator()
        )
        self.schema = schema
        # Keep a reference so tests / ops can confirm we wired settings.
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        *,
        question: str,
        generated_sql: str,
        execution_result: Any,
        schema: Any = None,
        embedder: Optional[Callable[..., Any]] = None,
        completion_fn: Optional[Callable[..., Any]] = None,
        sandbox: Optional[Callable[[str], Any]] = None,
    ) -> HallucinationReport:
        """Run all five signals and assemble a :class:`HallucinationReport`.

        Each signal runs defensively: any exception in a sub-component
        is captured, the corresponding signal is marked as not-ran with
        a neutral 0.5 score, and a warning is appended to the report so
        the orchestrator never crashes because one validator failed.
        """
        effective_schema = schema if schema is not None else self.schema
        outcomes: dict[str, SignalOutcome] = {}
        signals: dict[str, float] = {}
        warnings: list[str] = []

        # ---- syntax ----------------------------------------------------------
        outcome = self._score_syntax(generated_sql)
        outcomes[outcome.name] = outcome
        signals[outcome.name] = outcome.score

        # ---- back_translation ------------------------------------------------
        outcome = self._score_back_translation(
            question=question,
            generated_sql=generated_sql,
            schema=effective_schema,
            embedder=embedder,
            completion_fn=completion_fn,
        )
        outcomes[outcome.name] = outcome
        signals[outcome.name] = outcome.score
        if outcome.detail and outcome.detail.startswith("warning:"):
            warnings.append(outcome.detail)

        # ---- sanity ----------------------------------------------------------
        outcome = self._score_sanity(
            generated_sql=generated_sql,
            execution_result=execution_result,
            schema=effective_schema,
        )
        outcomes[outcome.name] = outcome
        signals[outcome.name] = outcome.score
        if outcome.detail and outcome.detail.startswith("warning:"):
            warnings.append(outcome.detail)

        # ---- multi_query -----------------------------------------------------
        outcome = self._score_multi_query(
            question=question,
            generated_sql=generated_sql,
            schema=effective_schema,
            completion_fn=completion_fn,
            sandbox=sandbox,
        )
        outcomes[outcome.name] = outcome
        signals[outcome.name] = outcome.score
        if outcome.detail and outcome.detail.startswith("warning:"):
            warnings.append(outcome.detail)

        # ---- schema_coverage -------------------------------------------------
        outcome = self._score_schema_coverage(
            generated_sql=generated_sql,
            schema=effective_schema,
        )
        outcomes[outcome.name] = outcome
        signals[outcome.name] = outcome.score
        if outcome.detail and outcome.detail.startswith("warning:"):
            warnings.append(outcome.detail)

        # ---- aggregate -------------------------------------------------------
        confidence = self.scorer.score(signals)
        verdict = _map_verdict(confidence.score)
        flagged = verdict == "low"

        report = HallucinationReport(
            question=question,
            generated_sql=generated_sql,
            overall_score=confidence.score,
            verdict=verdict,
            flagged=flagged,
            confidence=confidence,
            outcomes=outcomes,
            notes=list(confidence.notes),
            warnings=warnings,
        )

        ran_count = sum(1 for o in outcomes.values() if o.ran)
        skipped_count = len(outcomes) - ran_count
        flagged_signal_count = sum(1 for o in outcomes.values() if o.flagged)
        logger.info(
            "hallucination_detect.report",
            extra={
                "report_overall_score": report.overall_score,
                "report_verdict": report.verdict,
                "report_flagged": report.flagged,
                "signals_total": len(outcomes),
                "signals_ran": ran_count,
                "signals_skipped": skipped_count,
                "signals_flagged": flagged_signal_count,
            },
        )

        return report

    # ------------------------------------------------------------------
    # Per-signal implementations
    # ------------------------------------------------------------------
    def _score_syntax(self, generated_sql: str) -> SignalOutcome:
        ok = _syntax_ok(generated_sql)
        score = 1.0 if ok else 0.0
        detail = "sql starts with a recognised verb" if ok else (
            "sql is empty or does not start with a recognised verb"
        )
        return SignalOutcome(
            name="syntax",
            ran=True,
            score=score,
            flagged=score < 1.0,
            detail=detail,
        )

    def _score_back_translation(
        self,
        *,
        question: str,
        generated_sql: str,
        schema: Any,
        embedder: Optional[Callable[..., Any]],
        completion_fn: Optional[Callable[..., Any]],
    ) -> SignalOutcome:
        if embedder is None or completion_fn is None:
            return SignalOutcome(
                name="back_translation",
                ran=False,
                skipped_reason="embedder or completion_fn not provided",
                score=0.5,
                flagged=False,
                detail="neutral (embedder or completion_fn not provided)",
            )
        try:
            result: BackTranslationResult = self.back_translator.translate(
                sql=generated_sql,
                schema=schema,
                original_question=question,
                embedder=embedder,
                completion_fn=completion_fn,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            msg = f"back_translation error: {type(exc).__name__}: {exc}"
            logger.warning("hallucination_detect.back_translation_error", extra={"error": msg})
            return SignalOutcome(
                name="back_translation",
                ran=False,
                skipped_reason=msg,
                score=0.5,
                flagged=False,
                detail=f"warning: {msg}",
            )
        score = float(result.similarity)
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        detail = (
            f"warning: similarity={score:.3f} flagged={result.flagged}"
            if result.flagged
            else f"similarity={score:.3f}"
        )
        return SignalOutcome(
            name="back_translation",
            ran=True,
            score=score,
            flagged=bool(result.flagged),
            detail=detail,
        )

    def _score_sanity(
        self,
        *,
        generated_sql: str,
        execution_result: Any,
        schema: Any,
    ) -> SignalOutcome:
        if execution_result is None:
            return SignalOutcome(
                name="sanity",
                ran=False,
                skipped_reason="execution_result not provided",
                score=0.5,
                flagged=False,
                detail="neutral (execution_result not provided)",
            )
        try:
            result: SanityCheckResult = self.sanity_checker.check(
                sql=generated_sql,
                execution_result=execution_result,
                schema=schema,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            msg = f"sanity_check error: {type(exc).__name__}: {exc}"
            logger.warning("hallucination_detect.sanity_error", extra={"error": msg})
            return SignalOutcome(
                name="sanity",
                ran=False,
                skipped_reason=msg,
                score=0.5,
                flagged=False,
                detail=f"warning: {msg}",
            )

        # Spec: 1.0 if clean, otherwise 1.0 - 0.25 * len(anomalies), clamped
        # to [0.0, 0.75]. This ensures any anomaly penalises but never
        # zeros out the signal — the verdict is carried by `flagged`.
        if not result.flagged:
            score = 1.0
        else:
            score = 1.0 - 0.25 * len(result.anomalies)
            if score < 0.0:
                score = 0.0
            if score > 0.75:
                score = 0.75
        detail = (
            f"warning: {len(result.anomalies)} anomalies, flagged={result.flagged}"
            if result.flagged
            else "no anomalies"
        )
        return SignalOutcome(
            name="sanity",
            ran=True,
            score=score,
            flagged=bool(result.flagged),
            detail=detail,
        )

    def _score_multi_query(
        self,
        *,
        question: str,
        generated_sql: str,
        schema: Any,
        completion_fn: Optional[Callable[..., Any]],
        sandbox: Optional[Callable[[str], Any]],
    ) -> SignalOutcome:
        if completion_fn is None or sandbox is None:
            return SignalOutcome(
                name="multi_query",
                ran=False,
                skipped_reason="completion_fn or sandbox not provided",
                score=0.5,
                flagged=False,
                detail="neutral (completion_fn or sandbox not provided)",
            )
        try:
            result: MultiQueryResult = self.multi_query_validator.validate(
                question=question,
                schema=schema,
                generated_sql=generated_sql,
                completion_fn=completion_fn,
                sandbox=sandbox,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            msg = f"multi_query error: {type(exc).__name__}: {exc}"
            logger.warning("hallucination_detect.multi_query_error", extra={"error": msg})
            return SignalOutcome(
                name="multi_query",
                ran=False,
                skipped_reason=msg,
                score=0.5,
                flagged=False,
                detail=f"warning: {msg}",
            )

        # Spec mapping:
        #   - not flagged    → 1.0
        #   - skipped=True   → 0.5 (trigger not met; neutral)
        #   - flagged        → 0.0
        if result.flagged:
            score = 0.0
        elif result.skipped:
            score = 0.5
        else:
            score = 1.0

        if result.skipped:
            detail = (
                f"skipped: {result.skip_reason or 'trigger not met'}"
            )
        elif result.flagged:
            detail = (
                f"warning: result sets disagreed"
                + (
                    f" ({result.disagreement_detail})"
                    if result.disagreement_detail
                    else ""
                )
            )
        else:
            detail = "agreed"
        return SignalOutcome(
            name="multi_query",
            ran=not result.skipped,
            skipped_reason=result.skip_reason if result.skipped else None,
            score=score,
            flagged=bool(result.flagged),
            detail=detail,
        )

    def _score_schema_coverage(
        self,
        *,
        generated_sql: str,
        schema: Any,
    ) -> SignalOutcome:
        if schema is None:
            return SignalOutcome(
                name="schema_coverage",
                ran=False,
                skipped_reason="schema not provided",
                score=0.5,
                flagged=False,
                detail="neutral (schema not provided)",
            )

        try:
            schema_tokens = _schema_token_set(schema)
        except Exception as exc:  # noqa: BLE001 — defensive
            msg = f"schema not parseable: {type(exc).__name__}: {exc}"
            logger.warning(
                "hallucination_detect.schema_unparseable",
                extra={"error": msg},
            )
            return SignalOutcome(
                name="schema_coverage",
                ran=False,
                skipped_reason=msg,
                score=0.5,
                flagged=False,
                detail=f"warning: {msg}",
            )

        sql_tokens = _extract_identifiers(generated_sql)
        if not sql_tokens:
            coverage = 0.0
        else:
            overlap = sql_tokens & schema_tokens
            coverage = len(overlap) / len(sql_tokens)
        if coverage < 0.0:
            coverage = 0.0
        elif coverage > 1.0:
            coverage = 1.0
        flagged = coverage < 0.3
        detail = (
            f"warning: coverage={coverage:.2f} below 0.30"
            if flagged
            else f"coverage={coverage:.2f}"
        )
        return SignalOutcome(
            name="schema_coverage",
            ran=True,
            score=coverage,
            flagged=flagged,
            detail=detail,
        )


__all__ = [
    "HallucinationDetector",
    "HallucinationReport",
    "SignalOutcome",
]
