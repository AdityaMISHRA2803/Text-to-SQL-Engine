"""Result sanity checking (README 3.2).

Given an executed query's results, run a set of plausibility rules
(value ranges, non-negative counts, NULL-heavy columns, empty sets,
suspicious aggregations) and flag anomalies with specific reason strings.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from text2sql.config import get_settings

logger = logging.getLogger("text2sql.validation.sanity_checker")


#: Column-name tokens that suggest the column holds a percentage / ratio.
_PERCENT_TOKENS = ("percent", "pct", "rate", "ratio")

#: Column-name tokens that suggest the column holds a non-negative count.
_COUNT_TOKENS = ("count", "qty", "quantity", "total_count")

#: SQL type tokens (upper-cased) that count as integer-typed.
_INTEGER_TYPES = ("INTEGER", "BIGINT", "SMALLINT", "INT", "INT4", "INT8", "INT2")


class SanityAnomaly(BaseModel):
    """A single rule violation flagged by `SanityChecker`."""

    rule: str
    severity: str  # "low" | "medium" | "high"
    column: Optional[str] = None
    detail: str


class SanityCheckResult(BaseModel):
    """Aggregate outcome of running `SanityChecker.check`.

    `flagged` is True iff at least one anomaly has severity in
    {"medium", "high"}. Low-severity anomalies (e.g. `empty_result`) do
    NOT flip `flagged` to True on their own.
    """

    sql: str
    row_count: int
    flagged: bool
    anomalies: list[SanityAnomaly] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class _ColumnsRowsResult:
    """Adapter that exposes `columns` / `rows` / `row_count` from various shapes.

    The runner's `ExecutionResult` uses `results: list[dict]`. Tests and
    call-sites may also pass a plain object with `.columns` / `.rows` /
    `.row_count`. We resolve to a common tuple shape so the rule logic
    stays uniform.
    """

    def __init__(self, columns: list[str], rows: list[tuple], row_count: int) -> None:
        self.columns = list(columns)
        self.rows = list(rows)
        self.row_count = int(row_count)


def _coerce_result(execution_result: Any) -> _ColumnsRowsResult:
    """Return a `_ColumnsRowsResult` for whatever shape the caller passed.

    Accepts:
    - already-shaped objects with `.columns` / `.rows` / `.row_count`
    - `ExecutionResult`-like objects with `.results: list[dict]` and `.row_count`
    """
    if hasattr(execution_result, "columns") and hasattr(execution_result, "rows"):
        columns = list(execution_result.columns)
        rows = list(execution_result.rows)
        row_count = int(
            getattr(execution_result, "row_count", None) or len(rows)
        )
        return _ColumnsRowsResult(columns, rows, row_count)

    # ExecutionResult-like: .results is a list of dicts.
    if hasattr(execution_result, "results"):
        results = list(execution_result.results or [])
        columns: list[str] = []
        rows: list[tuple] = []
        for record in results:
            if isinstance(record, dict):
                if not columns:
                    columns = list(record.keys())
                rows.append(tuple(record.get(c) for c in columns))
            else:
                # Mapping-like (e.g. SQLAlchemy Row._mapping)
                try:
                    mapping = dict(record)
                except Exception:  # noqa: BLE001
                    continue
                if not columns:
                    columns = list(mapping.keys())
                rows.append(tuple(mapping.get(c) for c in columns))
        row_count = int(getattr(execution_result, "row_count", None) or len(rows))
        return _ColumnsRowsResult(columns, rows, row_count)

    raise TypeError(
        "execution_result must expose .columns/.rows/.row_count or .results/.row_count"
    )


def _build_schema_index(schema: Any) -> dict[str, dict[str, str]]:
    """Return ``{column_name: {"type": str, "table": str}}`` for fast lookup."""
    index: dict[str, dict[str, str]] = {}
    tables = getattr(schema, "tables", None) or []
    for table in tables:
        for col in getattr(table, "columns", None) or []:
            index[col.name] = {"type": str(col.type).upper(), "table": table.name}
    return index


def _is_percentage_column(name: str, schema_type: Optional[str]) -> bool:
    lowered = name.lower()
    if any(token in lowered for token in _PERCENT_TOKENS):
        return True
    return False


def _is_count_column(name: str, schema_type: Optional[str]) -> bool:
    lowered = name.lower()
    # Word-boundary regex so 'count' does not match inside 'discount'.
    if re.search(r"(?:^|[_\W])count(?:$|[_\W])", lowered):
        return True
    if any(token in lowered for token in ("qty", "quantity")):
        return True
    return False


class SanityChecker:
    """Run plausibility rules over an executed query's results.

    Public API::

        checker = SanityChecker()
        result = checker.check(sql, execution_result)
        result.flagged          # True if any medium/high anomaly fired
        result.anomalies        # list[SanityAnomaly]
    """

    def __init__(
        self,
        *,
        null_heavy_ratio: Optional[float] = None,
        flag_empty: bool = True,
    ) -> None:
        if null_heavy_ratio is None:
            null_heavy_ratio = float(get_settings().null_heavy_ratio)
        if not 0.0 <= null_heavy_ratio <= 1.0:
            raise ValueError("null_heavy_ratio must be in [0.0, 1.0]")
        self.null_heavy_ratio = float(null_heavy_ratio)
        self.flag_empty = bool(flag_empty)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check(
        self,
        sql: str,
        execution_result: Any,
        schema: Any = None,
    ) -> SanityCheckResult:
        """Run all rules and return a populated `SanityCheckResult`."""
        anomalies: list[SanityAnomaly] = []
        warnings: list[str] = []

        try:
            coerced = _coerce_result(execution_result)
        except TypeError as exc:
            return SanityCheckResult(
                sql=sql or "",
                row_count=0,
                flagged=False,
                anomalies=[],
                warnings=[f"unrecognised execution_result shape: {exc}"],
            )

        schema_index = _build_schema_index(schema) if schema is not None else {}

        # Rule 1: empty_result (low severity, configurable)
        if coerced.row_count == 0 and self.flag_empty:
            anomalies.append(
                SanityAnomaly(
                    rule="empty_result",
                    severity="low",
                    column=None,
                    detail="Query returned 0 rows",
                )
            )

        # Rule 5: suspicious_aggregation (must inspect SQL; cheap, run early)
        sql_text = sql or ""
        sql_upper = sql_text.upper()
        if "SUM(" in sql_upper and coerced.row_count == 0:
            anomalies.append(
                SanityAnomaly(
                    rule="suspicious_aggregation",
                    severity="medium",
                    column=None,
                    detail="SUM aggregation over zero rows is suspicious",
                )
            )

        # Rules 2-4 iterate columns; only meaningful when we have columns.
        if coerced.columns:
            self._check_columns(
                sql_text=sql_text,
                sql_upper=sql_upper,
                columns=coerced.columns,
                rows=coerced.rows,
                schema_index=schema_index,
                anomalies=anomalies,
                warnings=warnings,
            )

        flagged = any(a.severity in {"medium", "high"} for a in anomalies)

        if anomalies:
            logger.warning(
                "sanity_check.anomalies",
                extra={
                    "sql": sql_text,
                    "row_count": coerced.row_count,
                    "flagged": flagged,
                    "anomaly_rules": [a.rule for a in anomalies],
                    "anomaly_count": len(anomalies),
                },
            )

        return SanityCheckResult(
            sql=sql_text,
            row_count=coerced.row_count,
            flagged=flagged,
            anomalies=anomalies,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_columns(
        self,
        *,
        sql_text: str,
        sql_upper: str,
        columns: list[str],
        rows: list[tuple],
        schema_index: dict[str, dict[str, str]],
        anomalies: list[SanityAnomaly],
        warnings: list[str],
    ) -> None:
        n_rows = max(len(rows), 1)
        for col_idx, col_name in enumerate(columns):
            col_values = [row[col_idx] if col_idx < len(row) else None for row in rows]

            # Rule 4: null_heavy_column (high severity)
            null_count = sum(1 for v in col_values if v is None)
            if rows and (null_count / len(rows)) > self.null_heavy_ratio:
                anomalies.append(
                    SanityAnomaly(
                        rule="null_heavy_column",
                        severity="high",
                        column=col_name,
                        detail=(
                            f"column '{col_name}' has "
                            f"{null_count}/{len(rows)} NULL values "
                            f"({null_count / len(rows):.0%}), above threshold "
                            f"{self.null_heavy_ratio:.0%}"
                        ),
                    )
                )

            schema_info = schema_index.get(col_name)
            schema_type = schema_info["type"] if schema_info else None

            # Rule 2: percentage_out_of_range (medium)
            if self._is_percent_target(col_name, schema_type, sql_upper):
                sample = self._first_out_of_range_pct(col_values)
                if sample is not None:
                    anomalies.append(
                        SanityAnomaly(
                            rule="percentage_out_of_range",
                            severity="medium",
                            column=col_name,
                            detail=(
                                f"column '{col_name}' has value {sample!r} "
                                f"outside [0.0, 100.0]"
                            ),
                        )
                    )

            # Rule 3: negative_count (medium)
            if self._is_count_target(col_name, schema_type, sql_upper):
                sample = self._first_negative(col_values)
                if sample is not None:
                    anomalies.append(
                        SanityAnomaly(
                            rule="negative_count",
                            severity="medium",
                            column=col_name,
                            detail=(
                                f"column '{col_name}' has negative value "
                                f"{sample!r}; counts/quantities should be >= 0"
                            ),
                        )
                    )

    @staticmethod
    def _is_percent_target(name: str, schema_type: Optional[str], sql_upper: str) -> bool:
        if _is_percentage_column(name, schema_type):
            return True
        # Schema type is numeric AND SQL mentions /100 — value is meant as %.
        if schema_type and any(t in schema_type for t in ("NUMERIC", "FLOAT", "DOUBLE", "REAL", "DECIMAL")):
            if " / 100" in sql_upper or "/100" in sql_upper or "* 100" in sql_upper:
                return True
        return False

    @staticmethod
    def _is_count_target(name: str, schema_type: Optional[str], sql_upper: str) -> bool:
        if _is_count_column(name, schema_type):
            return True
        # INTEGER/BIGINT columns with a COUNT(*) in the SQL are aggregate counts.
        if schema_type and any(t in schema_type for t in _INTEGER_TYPES):
            if "COUNT(" in sql_upper:
                return True
        return False

    @staticmethod
    def _first_out_of_range_pct(values: list[Any]) -> Any:
        for v in values:
            if v is None:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                if v < 0.0 or v > 100.0:
                    return v
        return None

    @staticmethod
    def _first_negative(values: list[Any]) -> Any:
        for v in values:
            if v is None:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and v < 0:
                return v
        return None


__all__ = ["SanityAnomaly", "SanityCheckResult", "SanityChecker"]
