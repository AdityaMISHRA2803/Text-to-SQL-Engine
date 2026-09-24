"""Tests for text2sql.validation.multi_query (README 3.3 exit criterion).

These tests exercise the multi-query agreement validator end-to-end with
stubbed ``completion_fn`` and ``sandbox`` callables — no network, no DB.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable

import pytest

from text2sql.validation.multi_query import MultiQueryResult, MultiQueryValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_result(rows: list[dict], row_count: int | None = None) -> Any:
    """Build a sandbox-shaped result with ``.results`` and ``.row_count``."""

    class _R:
        pass

    r = _R()
    r.results = rows
    r.row_count = row_count if row_count is not None else len(rows)
    return r


def _make_completion_fn(sql: str) -> Callable[..., str]:
    """Return a completion_fn that always emits ``sql`` regardless of prompt."""

    def fn(prompt: str, **kwargs) -> str:
        return sql

    return fn


def _mapping_sandbox(mapping: dict[str, Any]) -> tuple[Callable[[str], Any], list[str]]:
    """Return a sandbox callable + a list that records every SQL it sees."""

    calls: list[str] = []

    def sandbox(sql: str) -> Any:
        calls.append(sql)
        return mapping[sql]

    return sandbox, calls


# ---------------------------------------------------------------------------
# Group A — should_trigger (6)
# ---------------------------------------------------------------------------
class TestShouldTrigger:
    def test_empty_any_is_false(self) -> None:
        assert MultiQueryValidator.should_trigger("", "any") is False

    def test_empty_join_or_agg_is_false(self) -> None:
        assert MultiQueryValidator.should_trigger("", "join_or_agg") is False

    def test_join_triggers_join_or_agg(self) -> None:
        sql = "SELECT * FROM a JOIN b ON a.id = b.id"
        assert MultiQueryValidator.should_trigger(sql, "join_or_agg") is True

    def test_group_by_triggers_join_or_agg(self) -> None:
        sql = "SELECT x FROM t GROUP BY x"
        assert MultiQueryValidator.should_trigger(sql, "join_or_agg") is True

    def test_simple_select_triggers_always(self) -> None:
        assert MultiQueryValidator.should_trigger("SELECT 1", "always") is True

    def test_simple_select_triggers_any(self) -> None:
        assert MultiQueryValidator.should_trigger("SELECT 1", "any") is True


# ---------------------------------------------------------------------------
# Group B — skipped path (3)
# ---------------------------------------------------------------------------
class TestSkippedPath:
    def test_simple_select_with_default_mode_is_skipped(self) -> None:
        # Default mode is "join_or_agg" (from Settings); simple SELECT does not meet the bar.
        validator = MultiQueryValidator()
        sandbox, _ = _mapping_sandbox({})
        completion = _make_completion_fn("SELECT 2 FROM t")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT * FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.skipped is True
        assert result.flagged is False
        assert result.skip_reason is not None
        assert "trigger complexity" in result.skip_reason.lower()

    def test_force_enable_runs_validation_on_simple_select(self) -> None:
        validator = MultiQueryValidator(force_enable=False)
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT * FROM t": _make_result([{"a": 1}]),
                "SELECT 2 FROM t": _make_result([{"a": 1}]),
            }
        )
        completion = _make_completion_fn("SELECT 2 FROM t")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT * FROM t",
            completion_fn=completion,
            sandbox=sandbox,
            force_enable=True,
        )
        assert result.skipped is False
        # Validation actually ran — sandbox was called for both SQLs.
        assert len(result.rows_a) == 1
        assert len(result.rows_b) == 1

    def test_join_query_runs_even_with_force_enable_false(self) -> None:
        validator = MultiQueryValidator(force_enable=False)
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT * FROM a JOIN b ON a.id = b.id": _make_result([{"x": 1}]),
                "SELECT 2 FROM b": _make_result([{"x": 1}]),
            }
        )
        completion = _make_completion_fn("SELECT 2 FROM b")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT * FROM a JOIN b ON a.id = b.id",
            completion_fn=completion,
            sandbox=sandbox,
            force_enable=False,
        )
        # Trigger met, so validation ran even with force_enable=False.
        assert result.skipped is False


# ---------------------------------------------------------------------------
# Group C — convergent SQL (3)
# ---------------------------------------------------------------------------
class TestConvergentSQL:
    def test_identical_row_sets_agree(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT id FROM t WHERE id = 1": _make_result([{"id": 1}]),
                "SELECT id FROM t WHERE id = 1": _make_result([{"id": 1}]),
            }
        )
        # Different SQL, same result: completion emits a rephrased SELECT.
        completion = _make_completion_fn("SELECT id FROM t WHERE id = 1")
        result = validator.validate(
            question="give me id 1",
            schema=None,
            generated_sql="SELECT id FROM t WHERE id = 1",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.agreed is True
        assert result.flagged is False

    def test_empty_result_sets_on_both_sides_agree(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT * FROM t": _make_result([]),
                "SELECT 2 FROM t": _make_result([]),
            }
        )
        completion = _make_completion_fn("SELECT 2 FROM t")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT * FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.agreed is True
        assert result.flagged is False

    def test_single_row_both_sides_agree(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT count(*) FROM t": _make_result([{"count": 7}]),
                "SELECT COUNT(*) FROM t": _make_result([{"count": 7}]),
            }
        )
        completion = _make_completion_fn("SELECT COUNT(*) FROM t")
        result = validator.validate(
            question="how many",
            schema=None,
            generated_sql="SELECT count(*) FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.agreed is True
        assert result.flagged is False


# ---------------------------------------------------------------------------
# Group D — divergent SQL (4)
# ---------------------------------------------------------------------------
class TestDivergentSQL:
    def test_different_row_values_disagree(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT id FROM t WHERE x = 1": _make_result([{"id": 1}]),
                "SELECT id FROM t WHERE x = 2": _make_result([{"id": 2}]),
            }
        )
        completion = _make_completion_fn("SELECT id FROM t WHERE x = 2")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT id FROM t WHERE x = 1",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.agreed is False
        assert result.flagged is True
        assert result.disagreement_detail is not None

    def test_different_row_counts_disagree(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT id FROM t": _make_result(
                    [{"id": 1}, {"id": 2}, {"id": 3}], row_count=3
                ),
                "SELECT id FROM t2": _make_result([{"id": 1}], row_count=1),
            }
        )
        completion = _make_completion_fn("SELECT id FROM t2")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT id FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.agreed is False
        assert result.flagged is True
        assert result.row_count_a == 3
        assert result.row_count_b == 1

    def test_different_column_shape_disagrees(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT a, b FROM t": _make_result([{"a": 1, "b": 2}]),
                "SELECT a FROM t": _make_result([{"a": 1}]),
            }
        )
        completion = _make_completion_fn("SELECT a FROM t")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT a, b FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.agreed is False
        assert result.flagged is True
        assert result.disagreement_detail is not None

    @pytest.mark.parametrize(
        "sql_a_result, sql_b_result",
        [
            # Scenario 1: row count differs
            (
                _make_result([{"x": 1}, {"x": 2}], row_count=2),
                _make_result([{"x": 1}], row_count=1),
            ),
            # Scenario 2: row values differ
            (
                _make_result([{"name": "Alice"}]),
                _make_result([{"name": "Bob"}]),
            ),
            # Scenario 3: column shape differs
            (
                _make_result([{"a": 1, "b": 2}]),
                _make_result([{"a": 1}]),
            ),
        ],
    )
    def test_readme_exit_criterion_three_stubbed_scenarios(
        self, sql_a_result: Any, sql_b_result: Any
    ) -> None:
        """README 3.3 exit criterion: 3 stubbed scenarios must all flag disagreements."""
        validator = MultiQueryValidator(trigger_mode="any")
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT a FROM t": sql_a_result,
                "SELECT b FROM t": sql_b_result,
            }
        )
        completion = _make_completion_fn("SELECT b FROM t")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT a FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.flagged is True
        assert result.agreed is False


# ---------------------------------------------------------------------------
# Group E — defensive parsing (2)
# ---------------------------------------------------------------------------
class TestDefensiveParsing:
    def test_markdown_fence_is_stripped(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT * FROM t": _make_result([{"x": 1}]),
                "SELECT 1 FROM t": _make_result([{"x": 1}]),
            }
        )

        def completion(prompt: str, **kwargs) -> str:
            return "```sql\nSELECT 1 FROM t\n```"

        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT * FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.sql_b == "SELECT 1 FROM t"

    def test_prose_around_sql_is_stripped(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT * FROM t": _make_result([{"x": 1}]),
                "SELECT 1 FROM t": _make_result([{"x": 1}]),
            }
        )

        def completion(prompt: str, **kwargs) -> str:
            return "Here you go:\nSELECT 1 FROM t\nThat's it."

        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT * FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.sql_b == "SELECT 1 FROM t"


# ---------------------------------------------------------------------------
# Group F — edge cases (3)
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_decimal_values_compare_via_default_str(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        # Both sides identical Decimal → agrees.
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT price FROM items": _make_result(
                    [{"price": Decimal("9.99")}]
                ),
                "SELECT price FROM products": _make_result(
                    [{"price": Decimal("9.99")}]
                ),
            }
        )
        completion = _make_completion_fn("SELECT price FROM products")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT price FROM items",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.agreed is True
        assert result.flagged is False

        # Now flip side B to a different Decimal → disagrees.
        sandbox2, _ = _mapping_sandbox(
            {
                "SELECT price FROM items": _make_result(
                    [{"price": Decimal("9.99")}]
                ),
                "SELECT price FROM products": _make_result(
                    [{"price": Decimal("10.00")}]
                ),
            }
        )
        result2 = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT price FROM items",
            completion_fn=completion,
            sandbox=sandbox2,
        )
        assert result2.agreed is False
        assert result2.flagged is True

    def test_datetime_values_compare_consistently(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")
        dt = datetime(2024, 1, 15, 12, 30, 0)
        sandbox, _ = _mapping_sandbox(
            {
                "SELECT created_at FROM t": _make_result([{"created_at": dt}]),
                "SELECT created_at FROM u": _make_result(
                    [{"created_at": datetime(2024, 1, 15, 12, 30, 0)}]
                ),
            }
        )
        completion = _make_completion_fn("SELECT created_at FROM u")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT created_at FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        # Same instant on both sides → agrees.
        assert result.agreed is True
        assert result.flagged is False

        # A date-only value differs from a datetime value → disagrees.
        sandbox2, _ = _mapping_sandbox(
            {
                "SELECT created_at FROM t": _make_result([{"created_at": dt}]),
                "SELECT created_at FROM u": _make_result(
                    [{"created_at": date(2024, 1, 15)}]
                ),
            }
        )
        result2 = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT created_at FROM t",
            completion_fn=completion,
            sandbox=sandbox2,
        )
        assert result2.agreed is False
        assert result2.flagged is True

    def test_sandbox_exception_produces_warning_and_flagged(self) -> None:
        validator = MultiQueryValidator(trigger_mode="any")

        def sandbox(sql: str) -> Any:
            if sql == "SELECT * FROM t":
                return _make_result([{"x": 1}])
            # sql_b raises — safe-default divergence.
            raise RuntimeError("boom")

        completion = _make_completion_fn("SELECT bad FROM u")
        result = validator.validate(
            question="q",
            schema=None,
            generated_sql="SELECT * FROM t",
            completion_fn=completion,
            sandbox=sandbox,
        )
        assert result.flagged is True
        assert len(result.warnings) >= 1
        # Warning references the sandbox failure.
        assert any("sandbox error" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Group G — Settings default (1)
# ---------------------------------------------------------------------------
class TestSettingsDefault:
    def test_default_trigger_mode_reads_from_settings(self) -> None:
        # The constructor reads get_settings().multi_query_min_complexity
        # when trigger_mode is not passed. Verify that default is the
        # documented "join_or_agg" by inspecting both the Settings default
        # and the constructed validator.
        from text2sql.config import Settings, get_settings

        # 1) Settings default.
        settings = get_settings()
        assert settings.multi_query_min_complexity == "join_or_agg"

        # 2) Validator constructed with no args picks up the same default.
        validator = MultiQueryValidator()
        assert validator.trigger_mode == "join_or_agg"
        assert validator.trigger_mode == settings.multi_query_min_complexity

        # 3) Cross-check that a fresh Settings() instance also yields
        #    the same default (Pydantic-level, not env-overridden).
        fresh = Settings()
        assert fresh.multi_query_min_complexity == "join_or_agg"
