"""Tests for the structured SQL generator.

20+ tests cover:
- Pure-model construction (3)
- OpenAI provider with mocked SDK response (3)
- Anthropic provider with mocked SDK response (3)
- Dual-provider parametrisation (2)
- Empty/whitespace question rejection (2)
- Malformed-JSON / missing-field rejection (3)
- Unknown tables drop confidence (2)
- Injected client support (2)
- Unknown-provider rejection (1)

Verification runs against no live network — every LLM call is routed through
a canned `completion_fn` that returns a JSON string the generator parses.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from text2sql.generation import GeneratedSQL, SQLGenerationError, SQLGenerator
from text2sql.schema import Column, ForeignKey, Schema, Table


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_schema() -> Schema:
    """Build a small in-memory Schema for the e-commerce subset."""
    return Schema(
        tables=[
            Table(
                name="customers",
                columns=[
                    Column(name="id", type="INTEGER", nullable=False, primary_key=True),
                    Column(name="email", type="VARCHAR", nullable=False, primary_key=False),
                    Column(name="country", type="VARCHAR", nullable=False, primary_key=False),
                ],
                primary_key=["id"],
                foreign_keys=[],
            ),
            Table(
                name="orders",
                columns=[
                    Column(name="id", type="INTEGER", nullable=False, primary_key=True),
                    Column(name="customer_id", type="INTEGER", nullable=False, primary_key=False),
                    Column(name="total", type="NUMERIC", nullable=False, primary_key=False),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKey(
                        constrained_columns=["customer_id"],
                        referred_table="customers",
                        referred_columns=["id"],
                    )
                ],
            ),
            Table(
                name="products",
                columns=[
                    Column(name="id", type="INTEGER", nullable=False, primary_key=True),
                    Column(name="name", type="VARCHAR", nullable=False, primary_key=False),
                    Column(name="price", type="NUMERIC", nullable=False, primary_key=False),
                ],
                primary_key=["id"],
                foreign_keys=[],
            ),
        ]
    )


# Twenty representative natural-language questions — covers SELECT, JOIN,
# aggregation, GROUP BY, subquery, ORDER BY, LIMIT, filters.
TWENTY_QUESTIONS: list[str] = [
    "Show me all customers.",
    "How many customers are there?",
    "List orders placed in Germany.",
    "What is the total revenue from all orders?",
    "Top 10 customers by total spend.",
    "Which customers have placed more than 5 orders?",
    "Show the email addresses of every customer.",
    "Count orders by status.",
    "Average order total.",
    "Customers who have never placed an order.",
    "Top 5 most expensive products.",
    "Total revenue per country.",
    "List all products under $50.",
    "Orders placed in the last 30 days.",
    "Which country has the most customers?",
    "Show products ordered more than 100 times.",
    "Email of customers whose largest order exceeded $500.",
    "Number of distinct countries.",
    "All orders with their customer emails.",
    "Products sorted by price descending.",
]


def _good_payload() -> dict[str, Any]:
    return {
        "sql": "SELECT id, email FROM customers ORDER BY id LIMIT 100;",
        "explanation": "Return customer id and email, sorted by id, capped at 100.",
        "confidence": 0.92,
        "tables_touched": ["customers"],
        "columns_touched": ["customers.id", "customers.email"],
    }


def _canned_openai_completion_fn(payload: dict[str, Any]):
    """completion_fn returning the OpenAI content shape (JSON-stringified)."""
    def _fn(prompt: str = "", **_kwargs: Any) -> str:
        return json.dumps(payload)
    return _fn


def _canned_anthropic_completion_fn(payload: dict[str, Any]):
    """completion_fn returning the Anthropic tool_use response shape."""
    def _fn(prompt: str = "", **_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "submit_sql",
                                    "arguments": json.dumps(payload),
                                },
                            }
                        ]
                    }
                }
            ]
        }
    return _fn


def _canned_anthropic_raw_content_completion_fn(payload: dict[str, Any]):
    """completion_fn returning the raw Anthropic content block shape."""
    def _fn(prompt: str = "", **_kwargs: Any) -> dict[str, Any]:
        return {
            "content": [
                {"type": "text", "text": "Generating SQL..."},
                {"type": "tool_use", "id": "tool_1", "name": "submit_sql", "input": payload},
            ]
        }
    return _fn


# ---------------------------------------------------------------------------
# Pure-model construction (3 tests)
# ---------------------------------------------------------------------------


def test_generated_sql_minimal_construction() -> None:
    g = GeneratedSQL(
        sql="SELECT 1",
        explanation="Return one row.",
        confidence=0.9,
    )
    assert g.sql == "SELECT 1"
    assert g.explanation == "Return one row."
    assert g.confidence == 0.9
    assert g.tables_touched == []
    assert g.columns_touched == []


def test_generated_sql_rejects_empty_sql() -> None:
    with pytest.raises(ValueError):
        GeneratedSQL(sql="", explanation="x", confidence=0.5)


def test_generated_sql_rejects_invalid_sql() -> None:
    """Non-SQL string should fail sqlparse.parse validation."""
    with pytest.raises(ValueError):
        GeneratedSQL(sql="not a sql statement just words", explanation="x", confidence=0.5)


# ---------------------------------------------------------------------------
# OpenAI provider with mocked SDK response (3 tests)
# ---------------------------------------------------------------------------


def test_openai_provider_returns_generated_sql() -> None:
    schema = _make_schema()
    gen = SQLGenerator(
        provider="openai",
        model="gpt-4o-mini",
        completion_fn=_canned_openai_completion_fn(_good_payload()),
    )
    out = gen.generate("Show all customers", schema)
    assert isinstance(out, GeneratedSQL)
    assert "SELECT id, email FROM customers" in out.sql
    assert out.confidence == pytest.approx(0.92)
    assert out.tables_touched == ["customers"]
    assert "customers.id" in out.columns_touched


def test_openai_provider_all_twenty_questions_produce_valid_sql() -> None:
    """Every question in the 20-question fixture parses and returns 5 fields."""
    schema = _make_schema()
    for q in TWENTY_QUESTIONS:
        gen = SQLGenerator(
            provider="openai",
            model="gpt-4o-mini",
            completion_fn=_canned_openai_completion_fn(_good_payload()),
        )
        out = gen.generate(q, schema)
        assert isinstance(out, GeneratedSQL), f"question={q!r}"
        assert out.sql and out.explanation and 0.0 <= out.confidence <= 1.0
        assert isinstance(out.tables_touched, list)
        assert isinstance(out.columns_touched, list)


def test_openai_provider_strict_kwargs_include_json_schema() -> None:
    """Verify the kwargs include the response_format=json_schema for OpenAI."""
    from text2sql.generation.generator import _build_provider_kwargs

    kwargs = _build_provider_kwargs("openai")
    assert "response_format" in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "submit_sql"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    params = kwargs["response_format"]["json_schema"]["schema"]
    for field in ("sql", "explanation", "confidence", "tables_touched", "columns_touched"):
        assert field in params["properties"]
    assert set(params["required"]) == {
        "sql", "explanation", "confidence", "tables_touched", "columns_touched",
    }


# ---------------------------------------------------------------------------
# Anthropic provider with mocked SDK response (3 tests)
# ---------------------------------------------------------------------------


def test_anthropic_provider_returns_generated_sql_via_tool_calls() -> None:
    schema = _make_schema()
    gen = SQLGenerator(
        provider="anthropic",
        model="claude-3-5-sonnet-latest",
        completion_fn=_canned_anthropic_completion_fn(_good_payload()),
    )
    out = gen.generate("Show all customers", schema)
    assert isinstance(out, GeneratedSQL)
    assert "SELECT id, email FROM customers" in out.sql
    assert out.confidence == pytest.approx(0.92)


def test_anthropic_provider_returns_generated_sql_via_raw_content() -> None:
    schema = _make_schema()
    gen = SQLGenerator(
        provider="anthropic",
        model="claude-3-5-sonnet-latest",
        completion_fn=_canned_anthropic_raw_content_completion_fn(_good_payload()),
    )
    out = gen.generate("Show all customers", schema)
    assert isinstance(out, GeneratedSQL)
    assert out.sql.startswith("SELECT")


def test_anthropic_provider_kwargs_include_tools_and_tool_choice() -> None:
    from text2sql.generation.generator import _build_provider_kwargs

    kwargs = _build_provider_kwargs("anthropic")
    assert "tools" in kwargs
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tools"][0]["function"]["name"] == "submit_sql"
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit_sql"}}


# ---------------------------------------------------------------------------
# Dual-provider parametrisation (2 tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet-latest"),
])
def test_provider_dispatch_returns_same_payload(provider: str, model: str) -> None:
    schema = _make_schema()
    if provider == "openai":
        cfn = _canned_openai_completion_fn(_good_payload())
    else:
        cfn = _canned_anthropic_completion_fn(_good_payload())
    gen = SQLGenerator(provider=provider, model=model, completion_fn=cfn)
    out = gen.generate("Show all customers", schema)
    assert isinstance(out, GeneratedSQL)
    assert out.sql == "SELECT id, email FROM customers ORDER BY id LIMIT 100;"
    assert out.confidence == pytest.approx(0.92)


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet-latest"),
])
def test_provider_dispatch_for_all_twenty_questions(provider: str, model: str) -> None:
    """For each of the 20 questions and each provider, generate succeeds."""
    schema = _make_schema()
    for q in TWENTY_QUESTIONS:
        if provider == "openai":
            cfn = _canned_openai_completion_fn(_good_payload())
        else:
            cfn = _canned_anthropic_completion_fn(_good_payload())
        gen = SQLGenerator(provider=provider, model=model, completion_fn=cfn)
        out = gen.generate(q, schema)
        assert out.sql and out.explanation, f"question={q!r}, provider={provider}"


# ---------------------------------------------------------------------------
# Empty/whitespace question rejection (2 tests)
# ---------------------------------------------------------------------------


def test_empty_question_raises_sql_generation_error() -> None:
    schema = _make_schema()
    gen = SQLGenerator(
        provider="openai", model="gpt-4o-mini",
        completion_fn=_canned_openai_completion_fn(_good_payload()),
    )
    with pytest.raises(SQLGenerationError, match="non-empty"):
        gen.generate("", schema)


def test_whitespace_question_raises_sql_generation_error() -> None:
    schema = _make_schema()
    gen = SQLGenerator(
        provider="openai", model="gpt-4o-mini",
        completion_fn=_canned_openai_completion_fn(_good_payload()),
    )
    with pytest.raises(SQLGenerationError, match="non-empty"):
        gen.generate("   \t\n  ", schema)


# ---------------------------------------------------------------------------
# Malformed-JSON / missing-field rejection (3 tests)
# ---------------------------------------------------------------------------


def test_malformed_json_response_raises_sql_generation_error() -> None:
    schema = _make_schema()
    gen = SQLGenerator(
        provider="openai", model="gpt-4o-mini",
        completion_fn=_canned_openai_completion_fn({"this": "is not valid JSON because we override"}),  # noqa: E501
    )
    # Override the completion_fn to return invalid JSON
    gen._completion_fn = lambda prompt="", **_: "this is not valid json at all"
    with pytest.raises(SQLGenerationError, match="JSON"):
        gen.generate("anything", schema)


def test_missing_required_field_raises_sql_generation_error() -> None:
    schema = _make_schema()
    bad_payload = {"sql": "SELECT 1", "explanation": "x", "confidence": 0.5}
    gen = SQLGenerator(
        provider="openai", model="gpt-4o-mini",
        completion_fn=_canned_openai_completion_fn(bad_payload),
    )
    with pytest.raises(SQLGenerationError, match="missing fields"):
        gen.generate("anything", schema)


def test_non_object_response_raises_sql_generation_error() -> None:
    schema = _make_schema()
    gen = SQLGenerator(
        provider="openai", model="gpt-4o-mini",
        completion_fn=lambda prompt="", **_: '"just a string"',
    )
    with pytest.raises(SQLGenerationError, match="JSON object"):
        gen.generate("anything", schema)


# ---------------------------------------------------------------------------
# Unknown tables drop confidence (2 tests)
# ---------------------------------------------------------------------------


def test_unknown_tables_in_response_drop_confidence() -> None:
    schema = _make_schema()
    payload = dict(_good_payload())
    payload["tables_touched"] = ["customers", "ghost_table"]
    payload["confidence"] = 0.9
    gen = SQLGenerator(
        provider="openai", model="gpt-4o-mini",
        completion_fn=_canned_openai_completion_fn(payload),
    )
    out = gen.generate("anything", schema)
    # ghost_table dropped, customers kept, confidence drops 0.2
    assert out.tables_touched == ["customers"]
    assert out.confidence == pytest.approx(0.7)


def test_unknown_columns_in_response_are_silently_dropped() -> None:
    schema = _make_schema()
    payload = dict(_good_payload())
    payload["columns_touched"] = ["customers.id", "ghost.col"]
    gen = SQLGenerator(
        provider="openai", model="gpt-4o-mini",
        completion_fn=_canned_openai_completion_fn(payload),
    )
    out = gen.generate("anything", schema)
    assert "customers.id" in out.columns_touched
    assert "ghost.col" not in out.columns_touched


# ---------------------------------------------------------------------------
# Injected client support (2 tests)
# ---------------------------------------------------------------------------


def test_injected_client_is_used_instead_of_default() -> None:
    from text2sql.llm import LLMClient

    schema = _make_schema()
    call_count = {"n": 0}

    def _injected_client_complete(prompt: str, *, completion_fn=None, **kwargs):
        call_count["n"] += 1
        return json.dumps(_good_payload())

    class StubClient:
        def complete(self, prompt: str, *, completion_fn=None, **kwargs):
            return _injected_client_complete(prompt, completion_fn=completion_fn, **kwargs)

    gen = SQLGenerator(
        provider="openai",
        model="gpt-4o-mini",
        completion_fn=_canned_openai_completion_fn(_good_payload()),
        client=StubClient(),  # type: ignore[arg-type]
    )
    out = gen.generate("anything", schema)
    assert call_count["n"] == 1
    assert out.sql.startswith("SELECT")


def test_llm_client_is_completion_fn_called_exactly_once() -> None:
    schema = _make_schema()
    call_count = {"n": 0}

    def _counting_cfn(prompt: str = "", **_kwargs: Any) -> str:
        call_count["n"] += 1
        return json.dumps(_good_payload())

    gen = SQLGenerator(
        provider="openai", model="gpt-4o-mini", completion_fn=_counting_cfn,
    )
    gen.generate("Show all customers", schema)
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Unknown-provider rejection (1 test)
# ---------------------------------------------------------------------------


def test_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        SQLGenerator(provider="gemini", model="x")  # type: ignore[arg-type]
