"""Tests for the guardrail validator (README 2.2).

Hand-built fixture verifies the five README expected behaviours:
- DROP TABLE customers → blocked with reason='ddl'
- INSERT INTO ... → blocked with reason='dml_write'
- 5-level nested subquery → blocked with reason='subquery_depth'
- no-LIMIT query → LIMIT 1000 appended
- EXPLAIN estimate > threshold → blocked with reason='explain_rows'

Plus ≥15 dangerous / ≥15 safe parametrised queries (30-query README fixture).
"""
from __future__ import annotations

import logging

import pytest

from text2sql.config import Settings
from text2sql.guardrails import (
    REASON_DDL,
    REASON_DML_WRITE,
    REASON_EXPLAIN_ROWS,
    REASON_PARSE_FAILURE,
    REASON_SUBQUERY_DEPTH,
    GuardrailValidator,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Hand-built fixture queries (the README's expected behaviour)
# ---------------------------------------------------------------------------


def test_drop_table_blocked_with_reason_ddl() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("DROP TABLE customers")
    assert result.allowed is False
    assert result.blocked_reason == REASON_DDL


def test_insert_blocked_with_reason_dml_write() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("INSERT INTO customers (email) VALUES ('x@y.com')")
    assert result.allowed is False
    assert result.blocked_reason == REASON_DML_WRITE


def test_five_level_nested_subquery_blocked_with_reason_subquery_depth() -> None:
    sql = (
        "SELECT * FROM "
        "(SELECT * FROM "
        "(SELECT * FROM "
        "(SELECT * FROM "
        "(SELECT * FROM t) a) b) c) d"
    )
    v = GuardrailValidator(max_subquery_depth=3, explain_runner=lambda _sql: 0)
    result = v.validate(sql)
    assert result.allowed is False
    assert result.blocked_reason == REASON_SUBQUERY_DEPTH


def test_no_limit_query_has_limit_injected() -> None:
    v = GuardrailValidator(limit_cap=1000, explain_runner=lambda _sql: 0)
    result = v.validate("SELECT * FROM orders")
    assert result.allowed is True
    assert "LIMIT 1000" in result.sql


def test_explain_estimate_exceeds_threshold_blocked() -> None:
    v = GuardrailValidator(max_explain_rows=1000, explain_runner=lambda _sql: 999_999)
    result = v.validate("SELECT * FROM orders")
    assert result.allowed is False
    assert result.blocked_reason == REASON_EXPLAIN_ROWS


# ---------------------------------------------------------------------------
# Settings defaults (4 tests)
# ---------------------------------------------------------------------------


def test_settings_default_limit_cap_is_1000() -> None:
    assert Settings().limit_cap == 1000


def test_settings_default_max_subquery_depth_is_3() -> None:
    assert Settings().max_subquery_depth == 3


def test_settings_default_max_explain_rows_is_100_000() -> None:
    assert Settings().max_explain_rows == 100_000


def test_settings_default_statement_timeout_is_30_000() -> None:
    assert Settings().statement_timeout_ms == 30_000


# ---------------------------------------------------------------------------
# Parse failure (2 tests)
# ---------------------------------------------------------------------------


def test_empty_sql_blocked_with_reason_parse_failure() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("")
    assert result.allowed is False
    assert result.blocked_reason == REASON_PARSE_FAILURE


def test_whitespace_sql_blocked_with_reason_parse_failure() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("   \n\t  ")
    assert result.allowed is False
    assert result.blocked_reason == REASON_PARSE_FAILURE


# ---------------------------------------------------------------------------
# DDL rejection (3 parametrised + 3 extras = 6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers",
    "CREATE TABLE foo (id INT)",
    "ALTER TABLE foo ADD COLUMN bar INT",
    "TRUNCATE TABLE orders",
])
def test_ddl_statements_are_blocked(sql: str) -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate(sql)
    assert result.allowed is False
    assert result.blocked_reason == REASON_DDL


def test_ddl_inside_string_literal_is_not_blocked() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("SELECT 'DROP TABLE customers' AS label FROM t LIMIT 1")
    assert result.allowed is True


def test_ddl_case_insensitive_detection() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("drop table customers")
    assert result.allowed is False
    assert result.blocked_reason == REASON_DDL


def test_ddl_keyword_in_any_position_is_blocked() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("SELECT * FROM customers; DROP TABLE customers;")
    assert result.allowed is False
    assert result.blocked_reason == REASON_DDL


# ---------------------------------------------------------------------------
# DML write rejection (4 parametrised + 3 extras = 7 tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql", [
    "INSERT INTO customers (email) VALUES ('a@b.com')",
    "UPDATE customers SET email = 'x'",
    "DELETE FROM orders WHERE id = 1",
    "GRANT SELECT ON customers TO PUBLIC",
])
def test_dml_writes_are_blocked(sql: str) -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate(sql)
    assert result.allowed is False
    assert result.blocked_reason == REASON_DML_WRITE


def test_dml_write_in_string_literal_is_not_blocked() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("SELECT 'INSERT INTO t VALUES (1)' AS label FROM t LIMIT 1")
    assert result.allowed is True


def test_merge_is_blocked_as_dml_write() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("MERGE INTO customers AS c USING staging AS s ON c.id = s.id")
    assert result.allowed is False
    assert result.blocked_reason == REASON_DML_WRITE


def test_revoke_is_blocked_as_dml_write() -> None:
    v = GuardrailValidator(explain_runner=lambda _sql: 0)
    result = v.validate("REVOKE SELECT ON customers FROM PUBLIC")
    assert result.allowed is False
    assert result.blocked_reason == REASON_DML_WRITE


# ---------------------------------------------------------------------------
# Subquery depth (3 tests)
# ---------------------------------------------------------------------------


def test_single_subquery_passes() -> None:
    v = GuardrailValidator(max_subquery_depth=3, explain_runner=lambda _sql: 0)
    sql = "SELECT * FROM (SELECT id FROM t) sub"
    result = v.validate(sql)
    assert result.allowed is True


def test_three_level_subquery_passes() -> None:
    v = GuardrailValidator(max_subquery_depth=3, explain_runner=lambda _sql: 0)
    sql = "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT id FROM t) a) b) c"
    result = v.validate(sql)
    assert result.allowed is True


def test_four_level_subquery_blocked() -> None:
    v = GuardrailValidator(max_subquery_depth=3, explain_runner=lambda _sql: 0)
    sql = "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT id FROM t) a) b) c) d"
    result = v.validate(sql)
    assert result.allowed is False
    assert result.blocked_reason == REASON_SUBQUERY_DEPTH


# ---------------------------------------------------------------------------
# LIMIT injection (4 tests)
# ---------------------------------------------------------------------------


def test_safe_query_with_no_limit_gets_limit_injected() -> None:
    v = GuardrailValidator(limit_cap=500, explain_runner=lambda _sql: 0)
    result = v.validate("SELECT * FROM orders")
    assert result.allowed is True
    assert "LIMIT 500" in result.sql
    assert any("No LIMIT" in w for w in result.warnings)


def test_query_with_existing_limit_is_not_modified() -> None:
    v = GuardrailValidator(limit_cap=500, explain_runner=lambda _sql: 0)
    result = v.validate("SELECT * FROM orders LIMIT 10")
    assert result.allowed is True
    assert result.sql == "SELECT * FROM orders LIMIT 10"
    assert result.warnings == []


def test_limit_injection_preserves_trailing_semicolon() -> None:
    v = GuardrailValidator(limit_cap=100, explain_runner=lambda _sql: 0)
    result = v.validate("SELECT * FROM orders;")
    assert result.sql.strip().endswith("LIMIT 100;")


def test_limit_injection_with_custom_cap() -> None:
    v = GuardrailValidator(limit_cap=42, explain_runner=lambda _sql: 0)
    result = v.validate("SELECT * FROM orders")
    assert "LIMIT 42" in result.sql


# ---------------------------------------------------------------------------
# EXPLAIN row-estimate gate (4 tests)
# ---------------------------------------------------------------------------


def test_explain_estimate_below_threshold_passes() -> None:
    v = GuardrailValidator(max_explain_rows=10_000, explain_runner=lambda _sql: 100)
    result = v.validate("SELECT * FROM orders LIMIT 5")
    assert result.allowed is True


def test_explain_estimate_above_threshold_blocked() -> None:
    v = GuardrailValidator(max_explain_rows=100, explain_runner=lambda _sql: 50_000)
    result = v.validate("SELECT * FROM orders")
    assert result.allowed is False
    assert result.blocked_reason == REASON_EXPLAIN_ROWS


def test_explain_runner_receives_limit_injected_sql() -> None:
    seen: list[str] = []

    def _runner(sql: str) -> int:
        seen.append(sql)
        return 0

    v = GuardrailValidator(limit_cap=500, explain_runner=_runner)
    v.validate("SELECT * FROM orders")
    assert len(seen) == 1
    assert "LIMIT 500" in seen[0]


def test_explain_runner_failure_emits_warning_but_passes() -> None:
    def _runner(sql: str) -> int:
        raise RuntimeError("EXPLAIN crashed")

    v = GuardrailValidator(explain_runner=_runner)
    result = v.validate("SELECT * FROM orders LIMIT 10")
    assert result.allowed is True
    assert any("EXPLAIN runner failed" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# ValidationResult / structured logging (4 tests)
# ---------------------------------------------------------------------------


def test_validation_result_default_allowed() -> None:
    r = ValidationResult(sql="SELECT 1", allowed=True)
    assert r.warnings == []
    assert r.blocked_reason is None


def test_blocked_record_has_required_fields() -> None:
    r = ValidationResult(sql="DROP TABLE t", allowed=False, blocked_reason=REASON_DDL, warnings=["x"])
    rec = r.to_block_record(provider="openai", schema_hash="abc123")
    assert rec["sql"] == "DROP TABLE t"
    assert rec["reason"] == REASON_DDL
    assert rec["provider"] == "openai"
    assert rec["schema_hash"] == "abc123"
    assert "timestamp" in rec


def test_block_event_is_logged_via_caplog(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="text2sql.guardrails")
    v = GuardrailValidator(explain_runner=lambda _sql: 0, log_blocked=True)
    v.validate("DROP TABLE customers")
    assert any("guardrail_block" in rec.message for rec in caplog.records)
    assert any("ddl" in rec.message for rec in caplog.records)


def test_block_event_not_logged_when_log_blocked_false(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="text2sql.guardrails")
    v = GuardrailValidator(explain_runner=lambda _sql: 0, log_blocked=False)
    v.validate("DROP TABLE customers")
    assert not any("guardrail_block" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Constructor validation (3 tests)
# ---------------------------------------------------------------------------


def test_constructor_rejects_zero_limit_cap() -> None:
    with pytest.raises(ValueError, match="limit_cap"):
        GuardrailValidator(limit_cap=0)


def test_constructor_rejects_zero_max_explain_rows() -> None:
    with pytest.raises(ValueError, match="max_explain_rows"):
        GuardrailValidator(max_explain_rows=0)


def test_constructor_rejects_zero_statement_timeout() -> None:
    with pytest.raises(ValueError, match="statement_timeout_ms"):
        GuardrailValidator(statement_timeout_ms=0)


# ---------------------------------------------------------------------------
# 30-query README fixture — ≥15 dangerous queries
# ---------------------------------------------------------------------------


DANGEROUS_QUERIES: list[tuple[str, str]] = [
    ("DROP TABLE customers", REASON_DDL),
    ("CREATE TABLE foo (id INT)", REASON_DDL),
    ("ALTER TABLE customers ADD COLUMN x INT", REASON_DDL),
    ("TRUNCATE orders", REASON_DDL),
    ("INSERT INTO customers (email) VALUES ('x@y.com')", REASON_DML_WRITE),
    ("UPDATE orders SET total = 0", REASON_DML_WRITE),
    ("DELETE FROM orders WHERE id = 1", REASON_DML_WRITE),
    ("MERGE INTO customers USING staging ON 1=1", REASON_DML_WRITE),
    ("GRANT ALL ON customers TO PUBLIC", REASON_DML_WRITE),
    ("REVOKE ALL ON customers FROM PUBLIC", REASON_DML_WRITE),
    (
        "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM t) a) b) c) d",
        REASON_SUBQUERY_DEPTH,
    ),
    # EXPLAIN-busting queries (runner injected below)
    ("SELECT * FROM customers JOIN orders ON 1=1", REASON_EXPLAIN_ROWS),
    ("SELECT * FROM customers, orders, products", REASON_EXPLAIN_ROWS),
    ("SELECT * FROM orders WHERE total > 0", REASON_EXPLAIN_ROWS),
]


@pytest.mark.parametrize("sql,expected_reason", DANGEROUS_QUERIES)
def test_dangerous_queries_blocked_with_correct_reason(sql: str, expected_reason: str) -> None:
    runner = (
        (lambda _sql: 999_999)
        if expected_reason == REASON_EXPLAIN_ROWS
        else (lambda _sql: 0)
    )
    v = GuardrailValidator(
        limit_cap=1000, max_subquery_depth=3, max_explain_rows=100_000,
        explain_runner=runner,
    )
    result = v.validate(sql)
    assert result.allowed is False, f"query={sql!r} expected block with reason={expected_reason}"
    assert result.blocked_reason == expected_reason, (
        f"query={sql!r} expected reason={expected_reason}, got {result.blocked_reason}"
    )


# ---------------------------------------------------------------------------
# 30-query README fixture — ≥15 safe queries
# ---------------------------------------------------------------------------


SAFE_QUERIES: list[str] = [
    "SELECT 1",
    "SELECT id FROM customers LIMIT 10",
    "SELECT id, email FROM customers ORDER BY id LIMIT 100",
    "SELECT * FROM orders WHERE total > 100 LIMIT 50",
    "SELECT COUNT(*) FROM customers",
    "SELECT c.id, c.email FROM customers c JOIN orders o ON c.id = o.customer_id LIMIT 20",
    "SELECT * FROM products ORDER BY price DESC LIMIT 25",
    "SELECT country, COUNT(*) FROM customers GROUP BY country LIMIT 50",
    "SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers LIMIT 10) LIMIT 5",
    "WITH top_customers AS (SELECT id FROM customers LIMIT 5) SELECT * FROM top_customers",
    "SELECT id FROM orders WHERE id = 1 LIMIT 1",
    "SELECT id FROM customers WHERE country = 'US' LIMIT 10",
    "SELECT AVG(total) FROM orders LIMIT 1",
    "SELECT * FROM products WHERE price < 50 LIMIT 100",
    "SELECT DISTINCT country FROM customers LIMIT 200",
]


@pytest.mark.parametrize("sql", SAFE_QUERIES)
def test_safe_queries_pass(sql: str) -> None:
    v = GuardrailValidator(
        limit_cap=1000, max_subquery_depth=3, max_explain_rows=100_000,
        explain_runner=lambda _sql: 10,
    )
    result = v.validate(sql)
    assert result.allowed is True, (
        f"query={sql!r} should pass; got blocked_reason={result.blocked_reason}"
    )
