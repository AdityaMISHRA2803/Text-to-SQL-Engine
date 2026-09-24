"""Pure metrics primitives for the eval suite.

Everything here is dependency-light (only `sqlparse` from the project) so that
unit tests can exercise edge cases without a database, an LLM, or a network.

The runner (`evals/runner.py`) calls these to score generated outputs against
the golden dataset.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# SQL normalization
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_SEMI_RE = re.compile(r";\s*$")
_COMMENT_LINE_RE = re.compile(r"--[^\n]*")
_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL_RE = re.compile(r"'([^']*)'")


def normalize_sql(sql: str) -> str:
    """Canonicalize a SQL string for textual comparison.

    Steps (order matters):
      1. Strip block comments ``/* ... */``.
      2. Strip line comments ``-- ...`` (everything to end-of-line).
      3. Replace string literals with the empty string ``''`` so that
         `'USA'` and `'usa'` both collapse to the same placeholder. (We do
         not lowercase the literal contents; presence/absence matters for
         structural equality.)
      4. Lowercase all keywords.
      5. Collapse all whitespace runs to a single space.
      6. Strip a single trailing semicolon.
      7. Strip leading/trailing whitespace.

    The result is suitable for an equality check between a generated query
    and a golden query.
    """
    if sql is None:
        return ""
    s = str(sql)
    s = _COMMENT_BLOCK_RE.sub(" ", s)
    s = _COMMENT_LINE_RE.sub(" ", s)
    s = _STRING_LITERAL_RE.sub("''", s)
    s = s.lower()
    s = _WHITESPACE_RE.sub(" ", s)
    s = _TRAILING_SEMI_RE.sub("", s)
    return s.strip()


def sql_exact_match(generated: str | None, golden: str | None) -> bool:
    """True iff normalized(generated) == normalized(golden).

    Both arguments may be ``None`` (e.g. when a case has no golden SQL).
    Two Nones compare equal. A None on either side with a non-None on the
    other returns False.
    """
    if generated is None and golden is None:
        return True
    if generated is None or golden is None:
        return False
    return normalize_sql(generated) == normalize_sql(golden)


# ---------------------------------------------------------------------------
# Result-set comparison
# ---------------------------------------------------------------------------


def _row_key(row: dict[str, Any]) -> tuple:
    """Convert a result-row dict into a hashable, order-insensitive key.

    Floats are rounded to 2 decimal places to avoid spurious mismatches from
    the binary representation of decimals coming back from psycopg. None
    values are preserved (they sort before strings, which matches Postgres).
    Other values are stringified for safety.
    """
    out: list[Any] = []
    for v in row.values():
        if v is None:
            out.append((0, ""))
        elif isinstance(v, float):
            out.append((1, round(v, 2)))
        elif isinstance(v, (int,)):
            out.append((2, v))
        else:
            out.append((3, str(v)))
    return tuple(out)


def execution_match(
    generated_results: Iterable[dict[str, Any]] | None,
    golden_results: Iterable[dict[str, Any]] | None,
) -> bool:
    """Compare two result sets for semantic equality.

    Returns True iff:
      * Both are empty / None
      * Both have the same row count
      * Both contain the same column names (as a multiset)
      * Both have identical rows as multisets (order-insensitive)

    Tuples are converted to dicts on the fly so callers can pass raw
    SQLAlchemy ``Row._mapping`` objects.
    """
    gen_list = _to_dict_list(generated_results)
    gold_list = _to_dict_list(golden_results)

    if not gen_list and not gold_list:
        return True
    if len(gen_list) != len(gold_list):
        return False
    if _columns(gen_list[0]) != _columns(gold_list[0]):
        return False
    gen_keys = sorted(_row_key(r) for r in gen_list)
    gold_keys = sorted(_row_key(r) for r in gold_list)
    return gen_keys == gold_keys


def _to_dict_list(rows: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if rows is None:
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        if hasattr(r, "_mapping"):
            out.append(dict(r._mapping))
        elif isinstance(r, dict):
            out.append(r)
        else:
            out.append(dict(r))
    return out


def _columns(row: dict[str, Any]) -> frozenset[str]:
    return frozenset(row.keys())


# ---------------------------------------------------------------------------
# Aggregate counters
# ---------------------------------------------------------------------------


class Counters:
    """Thread-unsafe rolling tally, used by the runner to compute rates.

    Increment via ``counters.bump("sql_exact_match")`` and read totals via
    ``counters.get("sql_exact_match")``.
    """

    def __init__(self) -> None:
        self._tallies: dict[str, int] = {}

    def bump(self, key: str, n: int = 1) -> None:
        self._tallies[key] = self._tallies.get(key, 0) + n

    def get(self, key: str) -> int:
        return self._tallies.get(key, 0)

    def rate(self, num_key: str, denom_key: str) -> float:
        denom = self.get(denom_key)
        if denom == 0:
            return 0.0
        return self.get(num_key) / denom

    def as_dict(self) -> dict[str, int]:
        return dict(self._tallies)
