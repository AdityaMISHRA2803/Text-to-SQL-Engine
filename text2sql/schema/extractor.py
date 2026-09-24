"""Database schema introspection for the Text-to-SQL engine.

Returns a typed `Schema` Pydantic model describing the connected PostgreSQL
database: tables, columns with types and nullability, primary keys, declared
and inferred foreign-key relationships, and sample values for low-cardinality
categorical columns. The extractor is read-only — it never mutates the source
database. It accepts a SQLAlchemy Engine, a connection URL string, or None
(falls back to Settings().database_url), so it works against both the
docker-compose Postgres (dev) and the testcontainers fixture (tests).
"""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.sql import text as sa_text


class Column(BaseModel):
    """A single column in a database table."""
    name: str
    type: str  # PostgreSQL type as a human-readable string, e.g. "VARCHAR", "NUMERIC(10,2)", "INTEGER"
    nullable: bool
    primary_key: bool


class ForeignKey(BaseModel):
    """A foreign-key edge between two tables.

    `inferred=True` indicates this edge was derived from a naming convention
    (e.g. `orders.customer_id` → `customers.id`) rather than from an explicit
    `FOREIGN KEY` constraint in the database schema.
    """
    constrained_columns: list[str]
    referred_table: str
    referred_columns: list[str]
    inferred: bool = False


class Table(BaseModel):
    """A database table with its columns, keys, and sample categorical values."""
    name: str
    columns: list[Column]
    primary_key: list[str]
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    sample_values: dict[str, list[str]] = Field(default_factory=dict)


class Schema(BaseModel):
    """The full schema of a database — a collection of tables and their metadata."""
    tables: list[Table]

    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def to_markdown(self) -> str:
        """Human-readable Markdown rendering of the schema."""
        lines: list[str] = []
        lines.append(f"# Database Schema ({len(self.tables)} tables)")
        lines.append("")
        for table in self.tables:
            lines.append(f"## `{table.name}`")
            lines.append("")
            if table.primary_key:
                lines.append(f"**Primary key**: {', '.join(f'`{c}`' for c in table.primary_key)}")
                lines.append("")
            if table.columns:
                lines.append("| Column | Type | Nullable | PK |")
                lines.append("| --- | --- | --- | --- |")
                for col in table.columns:
                    lines.append(
                        f"| `{col.name}` | {col.type} | {'YES' if col.nullable else 'NO'} | {'YES' if col.primary_key else ''} |"
                    )
                lines.append("")
            if table.foreign_keys:
                lines.append("**Foreign keys**:")
                for fk in table.foreign_keys:
                    src = ", ".join(fk.constrained_columns)
                    dst = f"{fk.referred_table}({', '.join(fk.referred_columns)})"
                    tag = " (inferred)" if fk.inferred else ""
                    lines.append(f"- `{src}` → `{dst}`{tag}")
                lines.append("")
            if table.sample_values:
                lines.append("**Sample categorical values**:")
                for col_name, vals in table.sample_values.items():
                    lines.append(f"- `{col_name}`: {vals}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_compact_json(self) -> str:
        """Compact JSON representation suitable for inclusion in an LLM prompt."""
        import json
        return json.dumps(self.model_dump(), separators=(",", ":"))


# Thresholds for sample-value extraction.
DEFAULT_LOW_CARDINALITY_THRESHOLD = 20
DEFAULT_SAMPLE_LIMIT = 10


def _resolve_engine(engine_or_url: Union[Engine, str, None]) -> Engine:
    if engine_or_url is None:
        from text2sql.config import get_settings
        return create_engine(get_settings().database_url, future=True)
    if isinstance(engine_or_url, Engine):
        return engine_or_url
    return create_engine(engine_or_url, future=True)


def _format_type(col_info: dict) -> str:
    """Best-effort rendering of a SQLAlchemy column-info dict as a type string."""
    t = col_info.get("type")
    if t is None:
        return "UNKNOWN"
    return str(t)


def _build_prefix_map(all_tables: dict[str, Table]) -> dict[str, str]:
    """Build a map from any plausible column-prefix (singular or plural) to a table name.

    Handles both regular plurals (``order`` <-> ``orders``) and irregular
    plurals such as ``category`` <-> ``categories``.
    """
    prefix_to_table: dict[str, str] = {}
    for table_name in all_tables:
        # All name forms this table can be referenced by.
        forms: set[str] = {table_name}
        # Singularise the plural (or vice-versa, if the table itself is singular).
        if table_name.endswith("ies"):
            forms.add(table_name[:-3] + "y")          # categories -> category
        elif table_name.endswith("ses") or table_name.endswith("shes") \
                or table_name.endswith("ches") or table_name.endswith("xes"):
            forms.add(table_name[:-2])
        elif table_name.endswith("s"):
            forms.add(table_name[:-1])                 # orders -> order, customers -> customer, ...
        for form in forms:
            prefix_to_table[form] = table_name
            # Regular plural: order -> orders, customer -> customers.
            prefix_to_table[form + "s"] = table_name
            # Irregular plural for forms ending in "y": category -> categories.
            if form.endswith("y"):
                prefix_to_table[form[:-1] + "ies"] = table_name
    return prefix_to_table


def _infer_relationships(
    table_name: str,
    table_columns: list[Column],
    all_tables: dict[str, Table],
) -> list[ForeignKey]:
    """Infer likely FK edges from naming convention.

    A column ending in `_id` whose prefix matches another table name
    (singular, plural, or irregular plural like ``category``/``categories``)
    is treated as a reference to that table's ``id`` primary key, IF that
    table exists and has an ``id`` PK.
    """
    inferred: list[ForeignKey] = []
    prefix_to_table = _build_prefix_map(all_tables)
    for col in table_columns:
        if not col.name.endswith("_id") or col.name == "id":
            continue
        prefix = col.name[: -len("_id")]
        target = prefix_to_table.get(prefix)
        if target is not None and target != table_name:
            if "id" in all_tables[target].primary_key:
                inferred.append(
                    ForeignKey(
                        constrained_columns=[col.name],
                        referred_table=target,
                        referred_columns=["id"],
                        inferred=True,
                    )
                )
    return inferred


def _extract_sample_values(
    engine: Engine,
    table_name: str,
    columns: list[Column],
    threshold: int,
    sample_limit: int,
) -> dict[str, list[str]]:
    """For each low-cardinality column in `table_name`, return up to `sample_limit` distinct sample values."""
    samples: dict[str, list[str]] = {}
    with engine.connect() as conn:
        for col in columns:
            if col.primary_key:
                continue
            type_str = col.type.upper()
            if not any(t in type_str for t in ("CHAR", "TEXT", "VARCHAR", "UUID", "ENUM")):
                continue
            try:
                sql = sa_text(
                    f'SELECT "{col.name}" FROM "{table_name}" '
                    f'GROUP BY "{col.name}" ORDER BY COUNT(*) DESC'
                )
                rows = conn.execute(sql).fetchall()
            except Exception:
                continue
            values = [str(r[0]) for r in rows if r[0] is not None]
            if 0 < len(values) <= threshold:
                samples[col.name] = values[:sample_limit]
    return samples


def extract_schema(
    engine_or_url: Union[Engine, str, None] = None,
    *,
    low_cardinality_threshold: int = DEFAULT_LOW_CARDINALITY_THRESHOLD,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> Schema:
    """Introspect a database and return a typed `Schema` model.

    Args:
        engine_or_url: A SQLAlchemy Engine, a connection URL string, or None
            to fall back to Settings().database_url.
        low_cardinality_threshold: Max distinct values for a column to be
            considered categorical and have its sample values included.
        sample_limit: Max number of sample values to include per column.

    Returns:
        A `Schema` Pydantic model describing the connected database.
    """
    engine = _resolve_engine(engine_or_url)
    insp = inspect(engine)

    # Pass 1: extract declared structure for every table.
    tables_by_name: dict[str, Table] = {}
    for table_name in insp.get_table_names():
        cols: list[Column] = []
        for col_info in insp.get_columns(table_name):
            cols.append(
                Column(
                    name=col_info["name"],
                    type=_format_type(col_info),
                    nullable=bool(col_info.get("nullable", True)),
                    primary_key=False,
                )
            )
        pk_cols = list(insp.get_pk_constraint(table_name).get("constrained_columns") or [])
        for c in cols:
            if c.name in pk_cols:
                c.primary_key = True
        declared_fks: list[ForeignKey] = []
        for fk in insp.get_foreign_keys(table_name):
            if not fk.get("constrained_columns"):
                continue
            declared_fks.append(
                ForeignKey(
                    constrained_columns=list(fk["constrained_columns"]),
                    referred_table=fk.get("referred_table", ""),
                    referred_columns=list(fk.get("referred_columns") or []),
                    inferred=False,
                )
            )
        tables_by_name[table_name] = Table(
            name=table_name,
            columns=cols,
            primary_key=pk_cols,
            foreign_keys=declared_fks,
        )

    # Pass 2: add inferred relationships based on naming convention.
    for table_name, table in tables_by_name.items():
        inferred = _infer_relationships(table_name, table.columns, tables_by_name)
        existing_pairs = {
            (tuple(fk.constrained_columns), fk.referred_table) for fk in table.foreign_keys
        }
        for fk in inferred:
            key = (tuple(fk.constrained_columns), fk.referred_table)
            if key not in existing_pairs:
                table.foreign_keys.append(fk)

    # Pass 3: sample categorical values per table.
    for table_name, table in tables_by_name.items():
        table.sample_values = _extract_sample_values(
            engine, table_name, table.columns, low_cardinality_threshold, sample_limit
        )

    return Schema(tables=list(tables_by_name.values()))


__all__ = ["Column", "ForeignKey", "Table", "Schema", "extract_schema"]
