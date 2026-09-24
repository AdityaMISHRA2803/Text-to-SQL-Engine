"""Tests for the result sanity checker (Phase 2 — README 3.2).

These tests stub `execution_result` with a duck-typed object that exposes
`.columns` (list[str]) and `.rows` (list[tuple]) and `.row_count` (int).
The sanity checker also accepts `ExecutionResult`-shaped objects with
`.results` and `.row_count` (the runner's actual shape).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from text2sql.config import get_settings
from text2sql.schema import Column, Schema, Table
from text2sql.validation import SanityCheckResult, SanityChecker
from text2sql.validation.sanity_checker import SanityAnomaly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class StubResult:
    """Duck-typed result: .columns / .rows / .row_count."""

    columns: list[str]
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0

    def __post_init__(self) -> None:
        if self.row_count == 0 and self.rows:
            self.row_count = len(self.rows)


def _execution_result(
    columns: list[str],
    rows: list[tuple],
    row_count: Optional[int] = None,
) -> StubResult:
    return StubResult(
        columns=list(columns),
        rows=list(rows),
        row_count=row_count if row_count is not None else len(rows),
    )


def _col(name: str, type_: str = "VARCHAR", nullable: bool = True, pk: bool = False) -> Column:
    return Column(name=name, type=type_, nullable=nullable, primary_key=pk)


def _schema_with(columns_by_table: dict[str, list[Column]]) -> Schema:
    return Schema(
        tables=[
            Table(name=tname, columns=cols, primary_key=[c.name for c in cols if c.primary_key])
            for tname, cols in columns_by_table.items()
        ]
    )


# ===========================================================================
# 1) Pure unit — each rule independently
# ===========================================================================


def test_percentage_rule_fires_on_out_of_range_values() -> None:
    """A column named "discount_pct" with [-5, 50, 200] fires percentage_out_of_range."""
    result_obj = _execution_result(
        columns=["discount_pct"],
        rows=[(-5.0,), (50.0,), (200.0,)],
    )
    result = SanityChecker().check(
        "SELECT discount_pct FROM t",
        result_obj,
    )
    rules = [a.rule for a in result.anomalies]
    assert "percentage_out_of_range" in rules
    pct = next(a for a in result.anomalies if a.rule == "percentage_out_of_range")
    assert pct.severity == "medium"
    assert pct.column == "discount_pct"
    assert result.flagged is True


def test_negative_count_rule_fires_on_negative_values() -> None:
    """A column named "order_count" with [-1, 5] fires negative_count."""
    result_obj = _execution_result(
        columns=["order_count"],
        rows=[(-1,), (5,)],
    )
    result = SanityChecker().check(
        "SELECT order_count FROM t",
        result_obj,
    )
    rules = [a.rule for a in result.anomalies]
    assert "negative_count" in rules
    neg = next(a for a in result.anomalies if a.rule == "negative_count")
    assert neg.severity == "medium"
    assert neg.column == "order_count"


def test_null_heavy_fires_above_default_threshold() -> None:
    """Column with 60% NULLs (above default 50%) fires null_heavy_column."""
    result_obj = _execution_result(
        columns=["x", "y"],
        rows=[(1, None), (2, "a"), (3, None), (4, "b"), (5, None)],
    )
    result = SanityChecker().check("SELECT x, y FROM t", result_obj)
    rules = [a.rule for a in result.anomalies]
    assert "null_heavy_column" in rules
    heavy = next(a for a in result.anomalies if a.rule == "null_heavy_column")
    assert heavy.severity == "high"
    assert heavy.column == "y"


def test_null_heavy_does_not_fire_below_default_threshold() -> None:
    """Column with 40% NULLs (below default 50%) does NOT fire null_heavy_column."""
    result_obj = _execution_result(
        columns=["y"],
        rows=[("a",), (None,), ("b",), (None,), ("c",)],
    )
    result = SanityChecker().check("SELECT y FROM t", result_obj)
    rules = [a.rule for a in result.anomalies]
    assert "null_heavy_column" not in rules


# ===========================================================================
# 2) Empty-result rule
# ===========================================================================


def test_empty_result_triggers_low_severity_anomaly() -> None:
    """row_count=0 with default flag_empty=True emits empty_result (low)."""
    result_obj = _execution_result(columns=["x"], rows=[], row_count=0)
    result = SanityChecker().check("SELECT x FROM t", result_obj)
    assert any(a.rule == "empty_result" for a in result.anomalies)
    empty = next(a for a in result.anomalies if a.rule == "empty_result")
    assert empty.severity == "low"
    assert empty.detail == "Query returned 0 rows"


def test_empty_result_suppressed_when_flag_empty_false() -> None:
    """SanityChecker(flag_empty=False) suppresses the empty_result anomaly."""
    result_obj = _execution_result(columns=["x"], rows=[], row_count=0)
    result = SanityChecker(flag_empty=False).check("SELECT x FROM t", result_obj)
    assert all(a.rule != "empty_result" for a in result.anomalies)


# ===========================================================================
# 3) Suspicious aggregation
# ===========================================================================


def test_suspicious_aggregation_fires_for_sum_over_zero_rows() -> None:
    """SQL with `SUM(` and 0 rows triggers suspicious_aggregation (medium)."""
    result_obj = _execution_result(columns=["total"], rows=[], row_count=0)
    result = SanityChecker().check("SELECT SUM(amount) AS total FROM t", result_obj)
    rules = [a.rule for a in result.anomalies]
    assert "suspicious_aggregation" in rules
    susp = next(a for a in result.anomalies if a.rule == "suspicious_aggregation")
    assert susp.severity == "medium"


# ===========================================================================
# 4) Combined scenarios
# ===========================================================================


def test_two_anomalies_both_reported_and_flagged_true() -> None:
    """A result with two anomalies reports both and sets flagged=True."""
    result_obj = _execution_result(
        columns=["discount_pct", "order_count"],
        rows=[(-5.0, -1), (50.0, 5), (200.0, 10)],
    )
    result = SanityChecker().check("SELECT * FROM t", result_obj)
    rules = sorted({a.rule for a in result.anomalies})
    assert "percentage_out_of_range" in rules
    assert "negative_count" in rules
    assert result.flagged is True
    assert len(result.anomalies) == 2


def test_only_low_severity_does_not_flag() -> None:
    """Only low-severity anomalies (empty_result) keep flagged=False."""
    result_obj = _execution_result(
        columns=["x"],
        rows=[(1,), (2,), (3,)],
        row_count=0,  # row_count is 0 but we have rows? use explicit
    )
    # Pure empty case (rows=[] AND row_count=0).
    empty_obj = _execution_result(columns=["x"], rows=[], row_count=0)
    result = SanityChecker().check("SELECT x FROM t", empty_obj)
    rules = [a.rule for a in result.anomalies]
    assert rules == ["empty_result"]
    assert result.flagged is False


# ===========================================================================
# 5) Schema-aware rules
# ===========================================================================


def test_schema_aware_pct_column_fires() -> None:
    """Column named "pct" with values [150.0, 50.0] triggers percentage rule."""
    schema = _schema_with({"metrics": [_col("pct", "NUMERIC")]})
    result_obj = _execution_result(columns=["pct"], rows=[(150.0,), (50.0,)])
    result = SanityChecker().check("SELECT pct FROM metrics", result_obj, schema=schema)
    rules = [a.rule for a in result.anomalies]
    assert "percentage_out_of_range" in rules
    pct = next(a for a in result.anomalies if a.rule == "percentage_out_of_range")
    assert pct.column == "pct"


def test_schema_aware_qty_column_fires() -> None:
    """Column named "qty" with [-1] triggers negative_count."""
    schema = _schema_with({"orders": [_col("qty", "INTEGER")]})
    result_obj = _execution_result(columns=["qty"], rows=[(-1,)])
    result = SanityChecker().check("SELECT qty FROM orders", result_obj, schema=schema)
    rules = [a.rule for a in result.anomalies]
    assert "negative_count" in rules


# ===========================================================================
# 6) Threshold override
# ===========================================================================


def test_null_heavy_threshold_override() -> None:
    """SanityChecker(null_heavy_ratio=0.3) makes 40% NULLs trigger."""
    result_obj = _execution_result(
        columns=["y"],
        rows=[("a",), (None,), ("b",), (None,), ("c",)],
    )
    result = SanityChecker(null_heavy_ratio=0.3).check("SELECT y FROM t", result_obj)
    rules = [a.rule for a in result.anomalies]
    assert "null_heavy_column" in rules


# ===========================================================================
# 7) Happy paths
# ===========================================================================


def test_clean_result_no_anomalies() -> None:
    """Clean result with row_count>0 and no anomalies: flagged=False, anomalies=[]."""
    result_obj = _execution_result(
        columns=["name", "score"],
        rows=[("a", 10), ("b", 20), ("c", 30)],
    )
    result = SanityChecker().check("SELECT name, score FROM t", result_obj)
    assert result.flagged is False
    assert result.anomalies == []
    assert result.row_count == 3


def test_30_percent_nulls_not_flagged_with_default_threshold() -> None:
    """30% NULLs does not trip the default 50% threshold; flagged=False."""
    rows = [("a",), ("b",), ("c",), (None,), ("d",), ("e",), ("f",), ("g",), ("h",), ("i",)]
    # 1/10 = 10%, not 30% — but adjust to 30% (3/10):
    rows = [("a",), ("b",), ("c",), (None,), (None,), (None,), ("g",), ("h",), ("i",), ("j",)]
    result_obj = _execution_result(columns=["y"], rows=rows, row_count=10)
    result = SanityChecker().check("SELECT y FROM t", result_obj)
    assert all(a.rule != "null_heavy_column" for a in result.anomalies)
    assert result.flagged is False


# ===========================================================================
# 8) Logger
# ===========================================================================


def test_logger_warning_on_anomalies(caplog: pytest.LogCaptureFixture) -> None:
    """caplog captures WARNING when anomalies fire."""
    result_obj = _execution_result(columns=["discount_pct"], rows=[(150.0,)])
    with caplog.at_level(logging.WARNING, logger="text2sql.validation.sanity_checker"):
        SanityChecker().check("SELECT discount_pct FROM t", result_obj)
    assert any("sanity_check.anomalies" in rec.message for rec in caplog.records)


# ===========================================================================
# 9) Exit-criterion fixture (broken JOIN)
# ===========================================================================


def test_broken_join_fixture_triggers_null_heavy() -> None:
    """Broken JOIN: order_id and total_spend 60% NULL → null_heavy (high)."""
    schema = _schema_with(
        {
            "customers": [_col("customer_id", "INTEGER", pk=True)],
            "orders": [_col("order_id", "INTEGER", pk=True), _col("customer_id", "INTEGER")],
        }
    )
    # 5 rows, 3 of which have NULL order_id and total_spend (60% NULL each).
    rows = [
        (1, 101, 50.0),
        (2, 102, 30.0),
        (3, None, None),
        (4, None, None),
        (5, None, None),
    ]
    result_obj = _execution_result(
        columns=["customer_id", "order_id", "total_spend"],
        rows=rows,
    )
    sql = "SELECT c.customer_id, o.order_id, o.total_spend FROM customers c JOIN orders o"  # missing ON
    result = SanityChecker().check(sql, result_obj, schema=schema)

    rules = [a.rule for a in result.anomalies]
    assert rules.count("null_heavy_column") == 2  # order_id AND total_spend

    flagged_columns = {
        a.column for a in result.anomalies if a.rule == "null_heavy_column"
    }
    assert "order_id" in flagged_columns
    assert "total_spend" in flagged_columns

    for a in result.anomalies:
        if a.rule == "null_heavy_column":
            assert a.severity == "high"

    assert result.flagged is True


# ===========================================================================
# 10) Edge cases
# ===========================================================================


def test_all_null_column_fires() -> None:
    """100% NULL column fires null_heavy_column (every value is NULL)."""
    result_obj = _execution_result(
        columns=["x"],
        rows=[(None,), (None,), (None,), (None,)],
    )
    result = SanityChecker().check("SELECT x FROM t", result_obj)
    rules = [a.rule for a in result.anomalies]
    assert "null_heavy_column" in rules
    heavy = next(a for a in result.anomalies if a.rule == "null_heavy_column")
    assert heavy.column == "x"


def test_empty_rows_but_columns_present() -> None:
    """rows=[] but column headers present: empty_result fires (low)."""
    result_obj = _execution_result(columns=["a", "b"], rows=[], row_count=0)
    result = SanityChecker().check("SELECT a, b FROM t", result_obj)
    assert result.row_count == 0
    assert any(a.rule == "empty_result" for a in result.anomalies)


def test_suspicious_aggregation_lowercase_sum() -> None:
    """SQL with lowercase `sum(` still triggers suspicious_aggregation."""
    result_obj = _execution_result(columns=["total"], rows=[], row_count=0)
    # pydantic-ish SQL string with lowercase function.
    result = SanityChecker().check(
        "select sum(amount) as total from t",
        result_obj,
    )
    rules = [a.rule for a in result.anomalies]
    assert "suspicious_aggregation" in rules


# ===========================================================================
# 11) Public API surface
# ===========================================================================


def test_result_is_pydantic_sanity_check_result() -> None:
    """`check` returns a SanityCheckResult with the expected fields."""
    result_obj = _execution_result(columns=["x"], rows=[(1,)])
    result = SanityChecker().check("SELECT x FROM t", result_obj)
    assert isinstance(result, SanityCheckResult)
    assert hasattr(result, "sql")
    assert hasattr(result, "row_count")
    assert hasattr(result, "flagged")
    assert hasattr(result, "anomalies")
    assert hasattr(result, "warnings")


def test_settings_default_null_heavy_ratio_used_when_not_overridden() -> None:
    """SanityChecker() with no override uses Settings.null_heavy_ratio."""
    settings = get_settings()
    checker = SanityChecker()
    assert checker.null_heavy_ratio == float(settings.null_heavy_ratio)


def test_invalid_null_heavy_ratio_raises() -> None:
    """SanityChecker rejects null_heavy_ratio outside [0.0, 1.0]."""
    with pytest.raises(ValueError):
        SanityChecker(null_heavy_ratio=1.5)
    with pytest.raises(ValueError):
        SanityChecker(null_heavy_ratio=-0.1)


def test_execution_result_shaped_like_runner_results_is_accepted() -> None:
    """`ExecutionResult`-shaped object with `.results`/`.row_count` works."""

    @dataclass
    class RunnerStyleResult:
        results: list[dict]
        row_count: int
        columns: list = field(default_factory=list)
        rows: list = field(default_factory=list)

    result_obj = RunnerStyleResult(
        results=[{"name": "a", "score": 10}, {"name": "b", "score": 20}],
        row_count=2,
    )
    result = SanityChecker().check("SELECT name, score FROM t", result_obj)
    assert result.row_count == 2
    assert result.flagged is False
