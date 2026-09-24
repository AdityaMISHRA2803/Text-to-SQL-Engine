"""Tests for the prompt builder (Phase 1, Step 3)."""
from __future__ import annotations

import time

import pytest
from sqlalchemy.engine import Engine

from text2sql.prompts import PromptBuilder
from text2sql.prompts.builder import FewShot, estimate_tokens
from text2sql.schema import extract_schema


# A small but realistic glossary for testing.
SAMPLE_GLOSSARY = {
    "revenue": "SUM(quantity * unit_price) for paid orders",
    "active customer": "a customer who has placed at least one order in the last 90 days",
}


@pytest.fixture
def builder() -> PromptBuilder:
    """Default PromptBuilder; loads few_shots.json from disk."""
    return PromptBuilder()


@pytest.fixture
def schema(fresh_db: Engine):
    """Schema extracted from the seeded e-commerce DB."""
    return extract_schema(fresh_db)


def test_estimate_tokens_returns_positive_int() -> None:
    n = estimate_tokens("SELECT * FROM orders WHERE status = 'paid'")
    assert isinstance(n, int)
    assert n > 0


def test_estimate_tokens_handles_empty_string() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_scales_with_length() -> None:
    short = estimate_tokens("hi")
    long = estimate_tokens("hi " * 1000)
    assert long > short


def test_build_returns_string(builder: PromptBuilder, schema) -> None:
    prompt = builder.build("How many orders per customer?", schema)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_contains_relevant_table_names(builder: PromptBuilder, schema) -> None:
    """For a customer/order question, both table names should appear in the prompt."""
    prompt = builder.build("How many orders per customer?", schema)
    lower = prompt.lower()
    assert "orders" in lower
    assert "customers" in lower


def test_build_contains_at_least_one_few_shot(builder: PromptBuilder, schema) -> None:
    """Default builder must include at least one Q: ... SQL: ... pair."""
    prompt = builder.build("What's the total revenue by category?", schema)
    assert "Q:" in prompt
    assert "SQL:" in prompt
    assert "```sql" in prompt


def test_build_completes_under_500ms(builder: PromptBuilder, schema) -> None:
    """Average build() call must complete in under 500ms on the seeded fixture.

    We measure 10 invocations and require the average to be under 500ms — this
    is the spec target. The 10x averaging smooths out first-call overhead
    (e.g., JSON file load).
    """
    # Warm-up to absorb any first-call cost (file reads, etc.).
    _ = builder.build("warm up", schema)
    start = time.perf_counter()
    for _ in range(10):
        _ = builder.build("How many orders per customer?", schema)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / 10) * 1000
    assert avg_ms < 500, f"Average build() took {avg_ms:.1f}ms (limit: 500ms)"


def test_build_stays_under_token_cap(builder: PromptBuilder, schema) -> None:
    """The returned prompt must be at or under the configured token cap."""
    prompt = builder.build("How many orders per customer?", schema)
    assert estimate_tokens(prompt) <= builder.token_cap, (
        f"Prompt is {estimate_tokens(prompt)} tokens; cap is {builder.token_cap}"
    )


def test_build_truncates_when_cap_is_tiny(schema) -> None:
    """A tiny cap should trigger the truncation policy and emit a marker."""
    tiny = PromptBuilder(token_cap=200)
    prompt = tiny.build("How many orders per customer?", schema)
    # Even with a tiny cap, build() must not raise.
    assert isinstance(prompt, str)
    # The prompt should fit within the cap (truncation policy kicks in).
    assert estimate_tokens(prompt) <= tiny.token_cap, (
        f"Prompt is {estimate_tokens(prompt)} tokens; cap is {tiny.token_cap}"
    )
    # Truncation marker should be present.
    assert "truncated" in prompt


def test_build_with_empty_few_shots(schema) -> None:
    """An explicit empty few_shots list should produce a prompt with the (none provided) marker."""
    builder = PromptBuilder(few_shots=[])
    prompt = builder.build("How many orders per customer?", schema)
    assert "Examples" in prompt
    assert "none provided" in prompt


def test_build_with_glossary(schema) -> None:
    """Glossary terms should appear in the prompt when provided."""
    builder = PromptBuilder(glossary=SAMPLE_GLOSSARY)
    prompt = builder.build("What's our revenue from active customers?", schema)
    assert "Glossary" in prompt
    assert "revenue" in prompt.lower()


def test_build_with_empty_glossary(builder: PromptBuilder, schema) -> None:
    """No glossary should mean no Glossary section in the prompt."""
    prompt = builder.build("Anything", schema)
    assert "Glossary" not in prompt


def test_build_accepts_override_few_shots(schema) -> None:
    """Passing few_shots explicitly overrides the constructor's store."""
    custom = [FewShot(question="What is X?", sql="SELECT 1 AS x")]
    builder = PromptBuilder(few_shots=[])
    prompt = builder.build("Anything", schema, few_shots=custom)
    assert "What is X?" in prompt
    assert "SELECT 1 AS x" in prompt


def test_few_shot_selection_prefers_keyword_overlap() -> None:
    """Examples with shared keywords should rank above unrelated ones."""
    fs = [
        FewShot(question="Total payment volume per country", sql="SELECT ..."),
        FewShot(question="How many orders has each customer placed?", sql="SELECT ..."),
    ]
    builder = PromptBuilder(few_shots=fs, num_few_shots=1)
    selected = builder._select_few_shots("Tell me how many orders per customer")
    assert selected[0].question.startswith("How many orders")


def test_few_shot_selection_caps_at_num_few_shots() -> None:
    fs = [FewShot(question=f"Q{i}", sql=f"SELECT {i}") for i in range(10)]
    builder = PromptBuilder(few_shots=fs, num_few_shots=3)
    selected = builder._select_few_shots("anything")
    assert len(selected) == 3


def test_few_shot_selection_handles_empty_store() -> None:
    builder = PromptBuilder(few_shots=[])
    assert builder._select_few_shots("anything") == []


def test_build_contains_fk_graph_or_none_marker(builder: PromptBuilder, schema) -> None:
    """The FK-graph section should appear with either real edges or a (none) marker."""
    prompt = builder.build("Anything", schema)
    assert "Foreign Key Graph" in prompt


def test_build_contains_schema_section(builder: PromptBuilder, schema) -> None:
    """The schema section should appear with all 6 e-commerce tables listed."""
    prompt = builder.build("Anything", schema)
    assert "## Schema" in prompt
    for t in ("categories", "customers", "products", "orders", "order_items", "payments"):
        assert t in prompt


def test_few_shots_json_loadable() -> None:
    """The bundled few_shots.json must exist and be parseable."""
    from pathlib import Path

    from text2sql.prompts.builder import DEFAULT_FEW_SHOTS_PATH

    assert DEFAULT_FEW_SHOTS_PATH.exists(), f"Missing few_shots.json at {DEFAULT_FEW_SHOTS_PATH}"
    import json as _json

    data = _json.loads(DEFAULT_FEW_SHOTS_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 3, f"Expected at least 3 few-shot examples, got {len(data)}"
    for item in data:
        assert "question" in item and "sql" in item
