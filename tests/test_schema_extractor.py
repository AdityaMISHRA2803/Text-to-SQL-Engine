"""Tests for the schema extractor (Phase 1, Step 2)."""
from __future__ import annotations

import json

import pytest
from sqlalchemy.engine import Engine

from text2sql.schema import Column, ForeignKey, Schema, Table, extract_schema


SEEDED_TABLES = {"categories", "customers", "products", "orders", "order_items", "payments"}


def test_extract_returns_all_seeded_tables(fresh_db: Engine) -> None:
    schema = extract_schema(fresh_db)
    assert set(schema.table_names()) == SEEDED_TABLES


def test_every_table_has_columns(fresh_db: Engine) -> None:
    schema = extract_schema(fresh_db)
    for table in schema.tables:
        assert len(table.columns) > 0, f"Table {table.name} has no columns"


def test_every_column_has_a_type(fresh_db: Engine) -> None:
    schema = extract_schema(fresh_db)
    for table in schema.tables:
        for col in table.columns:
            assert col.type, f"Column {table.name}.{col.name} has empty type"
            assert col.type != "UNKNOWN", f"Column {table.name}.{col.name} has UNKNOWN type"
            assert col.type.strip(), f"Column {table.name}.{col.name} has whitespace-only type"


def test_primary_keys_are_detected(fresh_db: Engine) -> None:
    schema = extract_schema(fresh_db)
    for table in schema.tables:
        assert "id" in table.primary_key, f"Table {table.name} is missing 'id' in primary key"


def test_inferred_relationships_present(fresh_db: Engine) -> None:
    """The extractor should infer FK-like edges from `*_id` naming convention."""
    schema = extract_schema(fresh_db)
    by_name = {t.name: t for t in schema.tables}

    orders_fks = {(tuple(fk.constrained_columns), fk.referred_table) for fk in by_name["orders"].foreign_keys}
    assert (("customer_id",), "customers") in orders_fks, f"orders.customer_id → customers missing; got {orders_fks}"

    oi_fks = {(tuple(fk.constrained_columns), fk.referred_table) for fk in by_name["order_items"].foreign_keys}
    assert (("order_id",), "orders") in oi_fks, f"order_items.order_id → orders missing; got {oi_fks}"
    assert (("product_id",), "products") in oi_fks, f"order_items.product_id → products missing; got {oi_fks}"

    products_fks = {(tuple(fk.constrained_columns), fk.referred_table) for fk in by_name["products"].foreign_keys}
    assert (("category_id",), "categories") in products_fks, f"products.category_id → categories missing; got {products_fks}"

    payments_fks = {(tuple(fk.constrained_columns), fk.referred_table) for fk in by_name["payments"].foreign_keys}
    assert (("order_id",), "orders") in payments_fks, f"payments.order_id → orders missing; got {payments_fks}"


def test_inferred_flag_set_on_naming_convention_fks(fresh_db: Engine) -> None:
    schema = extract_schema(fresh_db)
    by_name = {t.name: t for t in schema.tables}
    customer_fk = next(
        (fk for fk in by_name["orders"].foreign_keys if "customer_id" in fk.constrained_columns),
        None,
    )
    assert customer_fk is not None
    assert customer_fk.inferred is True


def test_sample_values_for_low_cardinality_columns(fresh_db: Engine) -> None:
    """At least one categorical column on the e-commerce schema has sample values populated."""
    schema = extract_schema(fresh_db)
    by_name = {t.name: t for t in schema.tables}

    assert "country" in by_name["customers"].sample_values, (
        f"customers.country should have sample values; got columns: {list(by_name['customers'].sample_values.keys())}"
    )
    assert len(by_name["customers"].sample_values["country"]) > 0

    assert "status" in by_name["orders"].sample_values, (
        f"orders.status should have sample values; got columns: {list(by_name['orders'].sample_values.keys())}"
    )

    assert "method" in by_name["payments"].sample_values


def test_sample_values_excluded_for_high_cardinality_columns(fresh_db: Engine) -> None:
    """Customer emails are unique → no sample values should be populated for them."""
    schema = extract_schema(fresh_db)
    by_name = {t.name: t for t in schema.tables}
    assert "email" not in by_name["customers"].sample_values, "Unique emails should not be sampled"


def test_to_markdown_renders_expected_sections(fresh_db: Engine) -> None:
    schema = extract_schema(fresh_db)
    md = schema.to_markdown()
    assert md.startswith("# Database Schema")
    for table in schema.tables:
        assert f"## `{table.name}`" in md
    assert "Primary key" in md
    assert "Foreign keys" in md


def test_to_compact_json_is_valid_json(fresh_db: Engine) -> None:
    schema = extract_schema(fresh_db)
    compact = schema.to_compact_json()
    parsed = json.loads(compact)
    assert "tables" in parsed
    assert len(parsed["tables"]) == len(schema.tables)
    assert "\n" not in compact


def test_extract_accepts_url_string(pg_engine: Engine, fresh_db: Engine) -> None:
    """Passing a URL string should work just like passing an Engine against a seeded DB."""
    url = pg_engine.url.render_as_string(hide_password=False)
    schema = extract_schema(url)
    assert len(schema.tables) >= 6


def test_extract_is_read_only(fresh_db: Engine) -> None:
    """Extraction must not mutate the database — running it twice yields equal row counts."""
    from sqlalchemy import text

    def row_counts(engine: Engine) -> dict[str, int]:
        with engine.connect() as conn:
            return {
                t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() or 0
                for t in SEEDED_TABLES
            }

    before = row_counts(fresh_db)
    _ = extract_schema(fresh_db)
    after = row_counts(fresh_db)
    assert before == after, f"Extraction mutated row counts: before={before} after={after}"


def test_pydantic_models_round_trip(fresh_db: Engine) -> None:
    schema = extract_schema(fresh_db)
    json_str = schema.model_dump_json()
    restored = Schema.model_validate_json(json_str)
    assert restored.table_names() == schema.table_names()
    assert len(restored.tables[0].columns) == len(schema.tables[0].columns)
