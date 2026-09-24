"""Unit tests for evals/metrics.py — pure functions, no DB / no LLM."""
from __future__ import annotations

import pytest

from evals.metrics import (
    Counters,
    execution_match,
    normalize_sql,
    sql_exact_match,
)


# ---------------------------------------------------------------------------
# normalize_sql
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Whitespace
        ("SELECT  *   FROM   users", "select * from users"),
        ("select\t*\nfrom\nusers", "select * from users"),
        # Trailing semicolon
        ("SELECT 1;", "select 1"),
        ("SELECT 1;  ", "select 1"),
        # Case
        ("Select * FROM Users", "select * from users"),
        # Comments
        ("SELECT 1 -- get count\nFROM t", "select 1 from t"),
        ("/* hi */ SELECT 1 /* bye */ FROM t", "select 1 from t"),
        # String literals are placeholdered, not lowercased
        ("SELECT * FROM users WHERE country = 'USA'", "select * from users where country = ''"),
        ("SELECT * FROM users WHERE name = 'Alice'", "select * from users where name = ''"),
        # Already normalized
        ("select 1", "select 1"),
        # None / empty
        (None, ""),
        ("", ""),
    ],
)
def test_normalize_sql(raw, expected):
    assert normalize_sql(raw) == expected


def test_normalize_sql_preserves_string_literal_presence():
    """Two queries that differ only in literal contents should NOT match
    after normalization if we want them to. But after normalization both
    literals collapse to '' — which is intentional: structural equality
    means same shape, not same data values."""
    a = normalize_sql("SELECT * FROM t WHERE x = 'foo'")
    b = normalize_sql("SELECT * FROM t WHERE x = 'bar'")
    # They collapse to identical forms — by design.
    assert a == b


# ---------------------------------------------------------------------------
# sql_exact_match
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "generated,golden,expected",
    [
        ("SELECT * FROM users", "select * from users", True),
        ("SELECT  *  FROM  users", "select * from users", True),
        ("SELECT * FROM users;", "select * from users", True),
        ("SELECT * FROM customers", "select * from users", False),
        ("SELECT * FROM users WHERE id = 1", "select * from users", False),
        (None, None, True),
        (None, "select 1", False),
        ("select 1", None, False),
        ("", "", True),
    ],
)
def test_sql_exact_match(generated, golden, expected):
    assert sql_exact_match(generated, golden) is expected


# ---------------------------------------------------------------------------
# execution_match
# ---------------------------------------------------------------------------


def test_execution_match_empty():
    assert execution_match([], []) is True
    assert execution_match(None, None) is True
    assert execution_match([], None) is True


def test_execution_match_different_row_counts():
    gen = [{"a": 1}, {"a": 2}]
    gold = [{"a": 1}]
    assert execution_match(gen, gold) is False


def test_execution_match_different_columns():
    gen = [{"a": 1}]
    gold = [{"b": 1}]
    assert execution_match(gen, gold) is False


def test_execution_match_identical_rows():
    gen = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    gold = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    assert execution_match(gen, gold) is True


def test_execution_match_order_insensitive():
    gen = [{"id": 2, "name": "Bob"}, {"id": 1, "name": "Alice"}]
    gold = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    assert execution_match(gen, gold) is True


def test_execution_match_duplicates_matter():
    """Multiset semantics: two identical rows must both appear."""
    gen = [{"id": 1}, {"id": 1}, {"id": 2}]
    gold = [{"id": 1}, {"id": 2}]
    assert execution_match(gen, gold) is False


def test_execution_match_floats_rounded_to_2dp():
    """Postgres Decimal(10,2) -> float in Python can have binary noise.
    The matcher rounds to 2 dp so 19.989999 -> 19.99."""
    gen = [{"amount": 19.989999}]
    gold = [{"amount": 19.99}]
    assert execution_match(gen, gold) is True


def test_execution_match_handles_sqlalchemy_rows():
    """sqlalchemy.Row objects expose ._mapping — should be coerced to dict."""

    class FakeRow:
        def __init__(self, d):
            self._mapping = d

    gen = [FakeRow({"id": 1}), FakeRow({"id": 2})]
    gold = [{"id": 2}, {"id": 1}]
    assert execution_match(gen, gold) is True


def test_execution_match_none_values():
    gen = [{"id": 1, "name": None}]
    gold = [{"id": 1, "name": None}]
    assert execution_match(gen, gold) is True
    assert execution_match(gen, [{"id": 1, "name": ""}]) is False


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def test_counters_bump_and_get():
    c = Counters()
    assert c.get("a") == 0
    c.bump("a")
    c.bump("a")
    c.bump("a", n=3)
    assert c.get("a") == 5


def test_counters_rate():
    c = Counters()
    c.bump("num", 7)
    c.bump("denom", 10)
    assert c.rate("num", "denom") == 0.7


def test_counters_rate_no_divide_by_zero():
    c = Counters()
    assert c.rate("num", "denom") == 0.0


def test_counters_as_dict():
    c = Counters()
    c.bump("a")
    c.bump("b", n=2)
    assert c.as_dict() == {"a": 1, "b": 2}
