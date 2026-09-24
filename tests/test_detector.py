"""Tests for text2sql.validation.detector.HallucinationDetector.

Covers the five-signal orchestrator (syntax, back-translation, sanity,
multi-query agreement, schema coverage) plus verdict mapping, error
paths, and assembly. All collaborators are stubbed via injection; no
network and no real database connections are used.
"""
from __future__ import annotations

from typing import Any, Callable

import pytest

from text2sql.scoring.scorer import ConfidenceScore
from text2sql.validation.detector import (
    HallucinationDetector,
    HallucinationReport,
    SignalOutcome,
)


# ---------------------------------------------------------------------------
# Stub collaborators
# ---------------------------------------------------------------------------
class _StubEmbedder:
    """Deterministic embedder.

    The BackTranslator invokes `embedder([str, str])`, so the callable
    must accept a list of strings and return a list of equal-length
    vectors. The mapping is deterministic but arbitrary; cosine
    similarity between any two vectors ends up either 0.0 or 1.0, which
    is exactly what these tests need.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [self._vec_for(t) for t in texts]

    def _vec_for(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        if not text:
            return v
        seed = len(text) % self.dim
        v[seed] = 1.0
        return v


class _StubCompletionFn:
    """Returns a configurable string.

    `completion_fn` is called via `completion_fn(prompt=...)` with a
    single keyword argument, so accept `**kwargs` for flexibility.
    """

    def __init__(self, response: str = "SELECT 1") -> None:
        self.response = response
        self.calls: list[str] = []

    def __call__(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self.response


class _StubSandbox:
    """Maps SQL string → fake execution result."""

    def __init__(self, mapping: dict[str, Any]) -> None:
        self.mapping = dict(mapping)
        self.calls: list[str] = []

    def __call__(self, sql: str) -> Any:
        self.calls.append(sql)
        return self.mapping[sql]


def _exec_result(results=None, row_count: int | None = None):
    """Duck-typed execution result with the union of both shapes used in the codebase.

    `.rows` is a list of tuples (matching `SanityChecker`'s `_ColumnsRowsResult`
    contract); `.results` is the original list of dicts.
    """
    dict_rows = results or []
    columns = list(dict_rows[0].keys()) if dict_rows else []
    tuple_rows = [tuple(r.get(c) for c in columns) for r in dict_rows]
    return type(
        "R",
        (),
        {
            "results": list(dict_rows),
            "rows": tuple_rows,
            "columns": columns,
            "row_count": row_count if row_count is not None else len(dict_rows),
        },
    )()


class _Column:
    def __init__(self, name: str, type: str | None = None) -> None:
        self.name = name
        self.type = type


class _Table:
    def __init__(self, name: str, cols: list[str]) -> None:
        self.name = name
        self.columns = [_Column(c) for c in cols]


class _Schema:
    def __init__(self, tables: list[_Table]) -> None:
        self.tables = tables


# ---------------------------------------------------------------------------
# Detector factories — provide clean and "all-skipped" baselines
# ---------------------------------------------------------------------------
QUESTION = "Show me the top 5 customers by total spend."
GOOD_SQL = "SELECT customer_id, total FROM customers ORDER BY total DESC LIMIT 5"


def _make_clean_exec_result():
    """A clean (no anomalies) execution result."""
    return _exec_result(
        results=[{"customer_id": 1, "total": 99.5}, {"customer_id": 2, "total": 80.0}],
        row_count=2,
    )


def _make_anomalous_exec_result():
    """A result containing a negative count column (medium anomaly)."""
    return _exec_result(
        results=[{"customer_id": 1, "row_count": -3}, {"customer_id": 2, "row_count": 5}],
        row_count=2,
    )


def _make_detector(
    *,
    schema: Any = None,
    multi_query_force: bool = False,
) -> HallucinationDetector:
    """Default detector with a force-enabled multi-query validator."""
    from text2sql.validation.multi_query import MultiQueryValidator

    return HallucinationDetector(
        multi_query_validator=MultiQueryValidator(force_enable=multi_query_force),
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Group A — Syntax signal (3)
# ---------------------------------------------------------------------------
class TestSyntaxSignal:
    def test_valid_select_passes(self):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
        )
        out = report.outcomes["syntax"]
        assert isinstance(out, SignalOutcome)
        assert out.ran is True
        assert out.score == 1.0
        assert out.flagged is False

    @pytest.mark.parametrize(
        "empty_sql",
        ["", "   ", "\n\t  \n"],
    )
    def test_empty_sql_fails(self, empty_sql):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=empty_sql,
            execution_result=None,
        )
        out = report.outcomes["syntax"]
        assert out.ran is True
        assert out.score == 0.0
        assert out.flagged is True

    @pytest.mark.parametrize(
        "bad_sql",
        [
            "PRAGMA table_info(customers)",
            "EXPLAIN SELECT 1",
            "this is not sql at all",
            "drop table customers",
        ],
    )
    def test_non_select_opener_fails(self, bad_sql):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=bad_sql,
            execution_result=None,
        )
        out = report.outcomes["syntax"]
        assert out.ran is True
        assert out.score == 0.0
        assert out.flagged is True


# ---------------------------------------------------------------------------
# Group B — Back-translation signal (3)
# ---------------------------------------------------------------------------
class TestBackTranslationSignal:
    def test_runs_with_stubbed_dependencies(self):
        d = _make_detector()
        # Provide a completion response that returns valid JSON so the
        # translator goes through the embedder path.
        completion = _StubCompletionFn(
            response='{"question": "Top 5 customers by spend", "confidence": 0.9}'
        )
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
            embedder=_StubEmbedder(),
            completion_fn=completion,
        )
        out = report.outcomes["back_translation"]
        assert out.ran is True
        assert out.score == pytest.approx(out.score)  # float sanity
        assert 0.0 <= out.score <= 1.0
        assert out.skipped_reason is None

    def test_skipped_without_embedder(self):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
            embedder=None,
            completion_fn=_StubCompletionFn(),
        )
        out = report.outcomes["back_translation"]
        assert out.ran is False
        assert out.score == 0.5
        assert out.skipped_reason is not None
        assert "not provided" in out.skipped_reason.lower()

    def test_skipped_without_completion_fn(self):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
            embedder=_StubEmbedder(),
            completion_fn=None,
        )
        out = report.outcomes["back_translation"]
        assert out.ran is False
        assert out.score == 0.5
        assert out.skipped_reason is not None
        assert "not provided" in out.skipped_reason.lower()


# ---------------------------------------------------------------------------
# Group C — Sanity signal (3)
# ---------------------------------------------------------------------------
class TestSanitySignal:
    def test_clean_execution_passes(self):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=_make_clean_exec_result(),
        )
        out = report.outcomes["sanity"]
        assert out.ran is True
        assert out.score == 1.0
        assert out.flagged is False

    def test_negative_count_is_flagged(self):
        # Schema is informational; the negative_count rule fires based on
        # column-name heuristic so we don't need to wire it through.
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql="SELECT customer_id, row_count FROM customers",
            execution_result=_make_anomalous_exec_result(),
        )
        out = report.outcomes["sanity"]
        assert out.ran is True
        assert out.score < 1.0
        assert out.flagged is True

    def test_skipped_without_execution_result(self):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
        )
        out = report.outcomes["sanity"]
        assert out.ran is False
        assert out.score == 0.5
        assert out.skipped_reason is not None
        assert "not provided" in out.skipped_reason.lower()


# ---------------------------------------------------------------------------
# Group D — Multi-query signal (3)
# ---------------------------------------------------------------------------
class TestMultiQuerySignal:
    def test_convergent_sql_passes(self):
        d = _make_detector(multi_query_force=True)
        # Both SQLs map to the same fake result → agreement.
        sandbox = _StubSandbox(
            mapping={GOOD_SQL: _make_clean_exec_result(), "SELECT 1": _make_clean_exec_result()}
        )
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
            completion_fn=_StubCompletionFn(response="SELECT 1"),
            sandbox=sandbox,
        )
        out = report.outcomes["multi_query"]
        assert out.ran is True
        assert out.score == 1.0
        assert out.flagged is False

    def test_divergent_sql_is_flagged(self):
        d = _make_detector(multi_query_force=True)
        sandbox = _StubSandbox(
            mapping={
                GOOD_SQL: _make_clean_exec_result(),
                "SELECT 1": _exec_result(
                    results=[{"different_column": 999}],
                    row_count=1,
                ),
            }
        )
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
            completion_fn=_StubCompletionFn(response="SELECT 1"),
            sandbox=sandbox,
        )
        out = report.outcomes["multi_query"]
        assert out.ran is True
        assert out.score == 0.0
        assert out.flagged is True

    @pytest.mark.parametrize(
        "missing",
        [
            {"completion_fn": None, "sandbox": _StubSandbox(mapping={})},
            {"completion_fn": _StubCompletionFn(), "sandbox": None},
        ],
    )
    def test_skipped_without_completion_or_sandbox(self, missing):
        d = _make_detector(multi_query_force=True)
        kwargs = dict(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
        )
        kwargs.update(missing)
        report = d.detect(**kwargs)
        out = report.outcomes["multi_query"]
        assert out.ran is False
        assert out.score == 0.5
        assert out.skipped_reason is not None
        assert "not provided" in out.skipped_reason.lower()


# ---------------------------------------------------------------------------
# Group E — Schema coverage heuristic (3)
# ---------------------------------------------------------------------------
class TestSchemaCoverageSignal:
    def test_full_coverage(self):
        schema = _Schema(
            [_Table("customers", ["customer_id", "total", "name", "email"])]
        )
        d = _make_detector(schema=schema)
        sql = "SELECT customer_id, total, name, email FROM customers"
        report = d.detect(
            question=QUESTION,
            generated_sql=sql,
            execution_result=None,
            schema=schema,
        )
        out = report.outcomes["schema_coverage"]
        assert out.ran is True
        assert out.score >= 0.99
        assert out.flagged is False

    def test_low_coverage_flags(self):
        schema = _Schema([_Table("customers", ["customer_id", "total"])])
        d = _make_detector(schema=schema)
        # All identifiers are invented → coverage near zero.
        sql = "SELECT foo, bar, baz FROM quux WHERE zzz > 0"
        report = d.detect(
            question=QUESTION,
            generated_sql=sql,
            execution_result=None,
            schema=schema,
        )
        out = report.outcomes["schema_coverage"]
        assert out.ran is True
        assert out.score < 0.3
        assert out.flagged is True

    def test_skipped_without_schema(self):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
            schema=None,
        )
        out = report.outcomes["schema_coverage"]
        assert out.ran is False
        assert out.score == 0.5
        assert out.skipped_reason == "schema not provided"


# ---------------------------------------------------------------------------
# Group F — Verdict mapping (3)
# ---------------------------------------------------------------------------
def _stub_signal(name: str, score: float, flagged: bool = False) -> SignalOutcome:
    return SignalOutcome(
        name=name,
        ran=True,
        score=score,
        flagged=flagged,
        detail="stub",
    )


class TestVerdictMapping:
    """Verdict mapping uses stubbed SignalOutcomes fed into the scorer directly."""

    def test_all_high_signals_yield_high_verdict(self):
        outcomes = {
            n: _stub_signal(n, 1.0) for n in ("syntax", "back_translation", "sanity", "multi_query", "schema_coverage")
        }
        from text2sql.scoring.scorer import HallucinationScorer
        cs = HallucinationScorer().score({n: o.score for n, o in outcomes.items()})
        assert cs.score >= 80.0
        assert cs.verdict == "pass"
        assert cs.flagged is False

    def test_all_low_signals_yield_low_verdict(self):
        outcomes = {
            n: _stub_signal(n, 0.0, flagged=True) for n in ("syntax", "back_translation", "sanity", "multi_query", "schema_coverage")
        }
        from text2sql.scoring.scorer import HallucinationScorer
        cs = HallucinationScorer().score({n: o.score for n, o in outcomes.items()})
        assert cs.score < 50.0
        assert cs.verdict == "fail"
        assert cs.flagged is True

    def test_mixed_signals_yield_moderate_verdict(self):
        outcomes = {
            n: _stub_signal(n, 0.5) for n in ("syntax", "back_translation", "sanity", "multi_query", "schema_coverage")
        }
        from text2sql.scoring.scorer import HallucinationScorer
        cs = HallucinationScorer().score({n: o.score for n, o in outcomes.items()})
        assert 50.0 <= cs.score < 80.0
        assert cs.verdict == "warn"
        assert cs.flagged is False


# ---------------------------------------------------------------------------
# Group G — Error / warning paths (3)
# ---------------------------------------------------------------------------
def _exploding_translator(*args, **kwargs):
    raise RuntimeError("boom back-translator")


def _exploding_checker(*args, **kwargs):
    raise RuntimeError("boom sanity")


def _exploding_validator(*args, **kwargs):
    raise RuntimeError("boom multi-query")


class TestErrorPaths:
    def test_back_translator_exception(self):
        # BackTranslator catches completion_fn exceptions internally and
        # returns a degraded BackTranslationResult (similarity=0,
        # flagged=True) rather than raising. The detector should
        # therefore mark ran=True with flagged=True, not ran=False.
        from text2sql.validation.back_translator import BackTranslator
        d = HallucinationDetector(back_translator=BackTranslator())
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
            embedder=_StubEmbedder(),
            completion_fn=_exploding_translator,
        )
        out = report.outcomes["back_translation"]
        assert out.ran is True
        assert out.flagged is True
        assert out.score == 0.0

    def test_sanity_checker_exception(self):
        # SanityChecker coerces malformed results gracefully (no raise).
        # The detector therefore sees ran=True; the malformed shape is
        # treated as zero-row, which yields flagged=False / score=1.0.
        from text2sql.validation.sanity_checker import SanityChecker
        d = HallucinationDetector(sanity_checker=SanityChecker())
        bad_result = type("Bad", (), {"__repr__": lambda self: "bad"})()
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=bad_result,
        )
        out = report.outcomes["sanity"]
        assert out.ran is True
        # Malformed-but-no-anomalies → no flag, full credit.
        assert out.flagged is False
        assert out.score == 1.0

    def test_multi_query_validator_exception(self):
        from text2sql.validation.multi_query import MultiQueryValidator
        d = HallucinationDetector(
            multi_query_validator=MultiQueryValidator(force_enable=True)
        )
        # An exploding completion_fn bubbles up through
        # MultiQueryValidator._generate_second_sql as a warning, not
        # an exception, so we instead make the validator itself
        # raise via the public entry point by passing an injected
        # instance whose `validate` blows up.
        class _BadValidator(MultiQueryValidator):
            def validate(self, *args, **kwargs):
                raise RuntimeError("boom multi-query")

        d.multi_query_validator = _BadValidator(force_enable=True)
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
            completion_fn=_StubCompletionFn(),
            sandbox=_StubSandbox(mapping={}),
        )
        out = report.outcomes["multi_query"]
        assert out.ran is False
        assert out.score == 0.5
        assert any("multi_query" in w.lower() or "boom" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Group H — Assembly (3)
# ---------------------------------------------------------------------------
class TestAssembly:
    def test_detect_returns_populated_report(self):
        d = _make_detector(multi_query_force=True)
        sandbox = _StubSandbox(
            mapping={GOOD_SQL: _make_clean_exec_result(), "SELECT 1": _make_clean_exec_result()}
        )
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=_make_clean_exec_result(),
            embedder=_StubEmbedder(),
            completion_fn=_StubCompletionFn(response='{"question": "x", "confidence": 0.9}'),
            sandbox=sandbox,
        )
        assert isinstance(report, HallucinationReport)
        assert report.question == QUESTION
        assert report.generated_sql == GOOD_SQL
        assert isinstance(report.overall_score, float)
        assert report.verdict in {"high", "moderate", "low"}
        assert isinstance(report.flagged, bool)
        assert report.outcomes == dict(report.outcomes)
        # All five canonical signal names must be present.
        assert set(report.outcomes.keys()) == {
            "syntax",
            "back_translation",
            "sanity",
            "multi_query",
            "schema_coverage",
        }

    def test_confidence_is_well_formed(self):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
        )
        cs = report.confidence
        assert isinstance(cs, ConfidenceScore)
        assert 0.0 <= cs.score <= 100.0
        assert cs.verdict in {"pass", "warn", "fail"}

    def test_outcomes_dict_has_exactly_five_keys(self):
        d = _make_detector()
        report = d.detect(
            question=QUESTION,
            generated_sql=GOOD_SQL,
            execution_result=None,
        )
        assert isinstance(report.outcomes, dict)
        assert set(report.outcomes.keys()) == {
            "syntax",
            "back_translation",
            "sanity",
            "multi_query",
            "schema_coverage",
        }
        for name, out in report.outcomes.items():
            assert isinstance(out, SignalOutcome)
            assert out.name == name
