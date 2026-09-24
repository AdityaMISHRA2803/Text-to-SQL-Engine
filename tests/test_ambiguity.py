"""Tests for the ambiguity handler (Phase 1, Step 5).

The handler routes every LLM call through the dual-provider abstraction
(`text2sql.llm.LLMClient`), and these tests verify the abstraction is
exercised end-to-end with mocked SDK responses for both OpenAI and
Anthropic. No live network calls. Deterministic canned JSON.
"""
from __future__ import annotations

from typing import Any

import pytest

from text2sql.ambiguity import AmbiguityHandler, ClarificationRequest, Interpretation
from text2sql.llm import LLMClient
from text2sql.schema import Column, ForeignKey, Schema, Table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _col(name: str, type_: str = "VARCHAR", nullable: bool = True, pk: bool = False) -> Column:
    return Column(name=name, type=type_, nullable=nullable, primary_key=pk)


def _ecommerce_schema() -> Schema:
    """A representative e-commerce schema mirroring the seeded DB."""
    return Schema(
        tables=[
            Table(
                name="customers",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("email", "VARCHAR"),
                    _col("country", "VARCHAR"),
                    _col("created_at", "TIMESTAMP"),
                ],
                primary_key=["id"],
                sample_values={"country": ["US", "CA", "UK", "DE"]},
            ),
            Table(
                name="orders",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("customer_id", "INTEGER"),
                    _col("status", "VARCHAR"),
                    _col("total_amount", "NUMERIC"),
                    _col("created_at", "TIMESTAMP"),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKey(constrained_columns=["customer_id"], referred_table="customers",
                               referred_columns=["id"], inferred=True),
                ],
                sample_values={"status": ["pending", "paid", "shipped", "delivered"]},
            ),
            Table(
                name="order_items",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("order_id", "INTEGER"),
                    _col("product_id", "INTEGER"),
                    _col("quantity", "INTEGER"),
                    _col("unit_price", "NUMERIC"),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKey(constrained_columns=["order_id"], referred_table="orders",
                               referred_columns=["id"], inferred=True),
                    ForeignKey(constrained_columns=["product_id"], referred_table="products",
                               referred_columns=["id"], inferred=True),
                ],
                sample_values={},
            ),
            Table(
                name="products",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("name", "VARCHAR"),
                    _col("category_id", "INTEGER"),
                    _col("price", "NUMERIC"),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKey(constrained_columns=["category_id"], referred_table="categories",
                               referred_columns=["id"], inferred=True),
                ],
                sample_values={},
            ),
            Table(
                name="categories",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("name", "VARCHAR"),
                ],
                primary_key=["id"],
                sample_values={},
            ),
            Table(
                name="payments",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("order_id", "INTEGER"),
                    _col("amount", "NUMERIC"),
                    _col("method", "VARCHAR"),
                    _col("status", "VARCHAR"),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKey(constrained_columns=["order_id"], referred_table="orders",
                               referred_columns=["id"], inferred=True),
                ],
                sample_values={
                    "method": ["credit_card", "paypal"],
                    "status": ["pending", "completed", "failed"],
                },
            ),
        ]
    )


# Mocked LLM response builders — both LiteLLM-normalised shapes work
# because `LLMClient._extract_content` accepts both object and dict
# forms.

def _openai_style_response(content: str) -> dict[str, Any]:
    """OpenAI / LiteLLM-normalised dict shape."""
    return {"choices": [{"message": {"content": content}}]}


def _anthropic_style_response(content: str) -> dict[str, Any]:
    """Anthropic / LiteLLM-normalised dict shape (same envelope as OpenAI)."""
    return {"choices": [{"message": {"content": content}}]}


# A canned ambiguous response for "What's the revenue?" — 2 interpretations.
AMBIGUOUS_REVENUE_JSON = """{
    "ambiguous": true,
    "interpretations": [
        {
            "sql": "SELECT SUM(total_amount) FROM orders WHERE status = 'paid';",
            "explanation": "Gross revenue as the sum of all paid order totals, before refunds or chargebacks."
        },
        {
            "sql": "SELECT SUM(amount) FROM payments WHERE status = 'completed';",
            "explanation": "Net revenue as the sum of all completed payment amounts, after any payment-processor fees."
        }
    ]
}"""

# A canned unambiguous response — single interpretation passthrough.
UNAMBIGUOUS_TOP_CUSTOMERS_JSON = """{
    "ambiguous": false,
    "interpretations": [
        {
            "sql": "SELECT c.id, c.email, SUM(o.total_amount) AS revenue FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.email ORDER BY revenue DESC LIMIT 10;",
            "explanation": "Top 10 customers ranked by total spend across all their orders."
        }
    ]
}"""


@pytest.fixture
def schema() -> Schema:
    return _ecommerce_schema()


# ---------------------------------------------------------------------------
# Pure-model tests (no LLM involved)
# ---------------------------------------------------------------------------


def test_interpreter_creation_with_minimal_fields() -> None:
    i = Interpretation(sql="SELECT 1;", explanation="trivial")
    assert i.sql == "SELECT 1;"
    assert i.explanation == "trivial"
    assert i.assumptions == []


def test_clarification_request_minimum_one_interpretation() -> None:
    cr = ClarificationRequest(
        question="Q?",
        interpretations=[Interpretation(sql="SELECT 1", explanation="x")],
        ambiguous=False,
    )
    assert cr.question == "Q?"
    assert len(cr.interpretations) == 1
    assert cr.ambiguous is False
    assert cr.model_id is None


def test_clarification_request_with_model_id() -> None:
    cr = ClarificationRequest(
        question="Q?",
        interpretations=[Interpretation(sql="SELECT 1", explanation="x")],
        ambiguous=True,
        model_id="openai/gpt-4o-mini",
    )
    assert cr.model_id == "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# Handler detect() — empty question defensive path
# ---------------------------------------------------------------------------


def test_detect_empty_question_returns_passthrough_without_llm_call(schema) -> None:
    """An empty question should return a single-interpretation passthrough and NOT call the LLM."""
    calls: list[Any] = []

    def fake_completion_fn(**kwargs):
        calls.append(kwargs)
        return _openai_style_response('{"ambiguous": false, "interpretations": [{"sql": "x", "explanation": "y"}]}')

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn
    )
    cr = handler.detect("", schema)
    assert isinstance(cr, ClarificationRequest)
    assert cr.ambiguous is False
    assert len(cr.interpretations) == 1
    assert cr.question == ""
    assert calls == []  # LLM was NOT called


def test_detect_whitespace_question_returns_passthrough(schema) -> None:
    calls: list[Any] = []

    def fake_completion_fn(**kwargs):
        calls.append(kwargs)
        return _openai_style_response("{}")

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn
    )
    cr = handler.detect("   \n\t  ", schema)
    assert cr.ambiguous is False
    assert len(cr.interpretations) == 1
    assert calls == []


# ---------------------------------------------------------------------------
# Handler detect() — ambiguous path
# ---------------------------------------------------------------------------


def test_detect_ambiguous_question_returns_multiple_interpretations_openai(schema) -> None:
    def fake_completion_fn(**kwargs):
        return _openai_style_response(AMBIGUOUS_REVENUE_JSON)

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn
    )
    cr = handler.detect("What's the revenue?", schema)
    assert isinstance(cr, ClarificationRequest)
    assert cr.ambiguous is True
    assert len(cr.interpretations) >= 2
    # Each interpretation must have non-empty SQL and explanation.
    for interp in cr.interpretations:
        assert isinstance(interp, Interpretation)
        assert interp.sql.strip()
        assert interp.explanation.strip()
    # The SQL sketches must be distinct (≥2 interpretations → not duplicates).
    sqls = [i.sql.strip() for i in cr.interpretations]
    assert len(set(sqls)) == len(sqls), f"SQL sketches were not distinct: {sqls}"
    # Question is echoed back.
    assert cr.question == "What's the revenue?"
    # model_id is populated.
    assert cr.model_id == "openai/gpt-4o-mini"


def test_detect_ambiguous_question_returns_multiple_interpretations_anthropic(schema) -> None:
    """Same handler behaviour with the Anthropic provider path."""
    def fake_completion_fn(**kwargs):
        return _anthropic_style_response(AMBIGUOUS_REVENUE_JSON)

    handler = AmbiguityHandler(
        provider="anthropic", model="claude-3-5-sonnet-latest", completion_fn=fake_completion_fn
    )
    cr = handler.detect("What's the revenue?", schema)
    assert cr.ambiguous is True
    assert len(cr.interpretations) >= 2
    assert cr.model_id == "anthropic/claude-3-5-sonnet-latest"


# ---------------------------------------------------------------------------
# Handler detect() — unambiguous passthrough path
# ---------------------------------------------------------------------------


def test_detect_unambiguous_question_returns_single_interpretation_openai(schema) -> None:
    def fake_completion_fn(**kwargs):
        return _openai_style_response(UNAMBIGUOUS_TOP_CUSTOMERS_JSON)

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn
    )
    cr = handler.detect("Who are the top 10 customers by spend?", schema)
    assert isinstance(cr, ClarificationRequest)
    assert cr.ambiguous is False
    assert len(cr.interpretations) == 1
    assert cr.interpretations[0].sql
    assert cr.interpretations[0].explanation


def test_detect_unambiguous_question_returns_single_interpretation_anthropic(schema) -> None:
    def fake_completion_fn(**kwargs):
        return _anthropic_style_response(UNAMBIGUOUS_TOP_CUSTOMERS_JSON)

    handler = AmbiguityHandler(
        provider="anthropic", model="claude-3-5-sonnet-latest", completion_fn=fake_completion_fn
    )
    cr = handler.detect("Who are the top 10 customers by spend?", schema)
    assert cr.ambiguous is False
    assert len(cr.interpretations) == 1


# ---------------------------------------------------------------------------
# Uniform-shape contract: ambiguous and unambiguous paths return the same type
# ---------------------------------------------------------------------------


def test_response_shape_is_identical_for_both_paths(schema) -> None:
    """Both ambiguous and unambiguous inputs must return ClarificationRequest with the same shape."""
    amb_handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini",
        completion_fn=lambda **_: _openai_style_response(AMBIGUOUS_REVENUE_JSON),
    )
    unamb_handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini",
        completion_fn=lambda **_: _openai_style_response(UNAMBIGUOUS_TOP_CUSTOMERS_JSON),
    )
    amb_cr = amb_handler.detect("Revenue?", schema)
    unamb_cr = unamb_handler.detect("Top customers?", schema)
    # Same type.
    assert type(amb_cr) is type(unamb_cr) is ClarificationRequest
    # Same fields.
    assert set(amb_cr.model_dump().keys()) == set(unamb_cr.model_dump().keys())
    # Both have non-empty interpretations list.
    assert len(amb_cr.interpretations) >= 1
    assert len(unamb_cr.interpretations) >= 1


# ---------------------------------------------------------------------------
# Defensive fallback paths
# ---------------------------------------------------------------------------


def test_detect_malformed_json_returns_passthrough(schema) -> None:
    def fake_completion_fn(**kwargs):
        return _openai_style_response("not valid json at all { ]")

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn
    )
    cr = handler.detect("Revenue?", schema)
    assert cr.ambiguous is False
    assert len(cr.interpretations) == 1


def test_detect_json_with_empty_interpretations_returns_passthrough(schema) -> None:
    def fake_completion_fn(**kwargs):
        return _openai_style_response('{"ambiguous": false, "interpretations": []}')

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn
    )
    cr = handler.detect("Revenue?", schema)
    assert cr.ambiguous is False
    assert len(cr.interpretations) == 1


def test_detect_llm_exception_returns_passthrough(schema) -> None:
    def fake_completion_fn(**kwargs):
        raise RuntimeError("network down")

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn
    )
    cr = handler.detect("Revenue?", schema)
    assert cr.ambiguous is False
    assert len(cr.interpretations) == 1


def test_detect_llm_returns_json_fence_is_stripped(schema) -> None:
    """The handler should tolerate markdown code fences around the JSON."""
    fenced = "```json\n" + AMBIGUOUS_REVENUE_JSON + "\n```"
    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini",
        completion_fn=lambda **_: _openai_style_response(fenced),
    )
    cr = handler.detect("Revenue?", schema)
    assert cr.ambiguous is True
    assert len(cr.interpretations) >= 2


def test_detect_llm_returns_json_with_trailing_comma_is_tolerated(schema) -> None:
    """The handler should tolerate trailing commas (common LLM output quirk)."""
    with_trailing = AMBIGUOUS_REVENUE_JSON.replace('}', ',}', 1)
    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini",
        completion_fn=lambda **_: _openai_style_response(with_trailing),
    )
    cr = handler.detect("Revenue?", schema)
    assert cr.ambiguous is True
    assert len(cr.interpretations) >= 2


# ---------------------------------------------------------------------------
# LLM is called exactly once per detect()
# ---------------------------------------------------------------------------


def test_detect_calls_llm_exactly_once_per_call(schema) -> None:
    call_count = {"n": 0}

    def fake_completion_fn(**kwargs):
        call_count["n"] += 1
        return _openai_style_response(AMBIGUOUS_REVENUE_JSON)

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn
    )
    handler.detect("Revenue?", schema)
    assert call_count["n"] == 1


def test_detect_repeated_calls_each_invoke_llm(schema) -> None:
    call_count = {"n": 0}

    def fake_completion_fn(**kwargs):
        call_count["n"] += 1
        return _openai_style_response(UNAMBIGUOUS_TOP_CUSTOMERS_JSON)

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn
    )
    handler.detect("Q1?", schema)
    handler.detect("Q2?", schema)
    handler.detect("Q3?", schema)
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# Both providers selected — confirms dual-provider abstraction end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet-latest"),
])
def test_detect_works_for_both_providers(provider, model, schema) -> None:
    """The same handler code path drives both OpenAI and Anthropic."""
    def fake_completion_fn(**kwargs):
        # Use whichever response shape the provider would return;
        # LiteLLM normalises both to the same envelope.
        return {"choices": [{"message": {"content": AMBIGUOUS_REVENUE_JSON}}]}

    handler = AmbiguityHandler(
        provider=provider, model=model, completion_fn=fake_completion_fn
    )
    cr = handler.detect("What's the revenue?", schema)
    assert cr.ambiguous is True
    assert len(cr.interpretations) >= 2
    assert cr.model_id == f"{provider}/{model}"


def test_detect_uses_provided_client_when_injected(schema) -> None:
    """When a client is explicitly injected, the handler must use it (not create its own)."""
    calls = []

    class StubClient:
        provider = "openai"
        model = "stub-model"
        def complete(self, prompt, *, completion_fn=None, **kwargs):
            calls.append(prompt)
            return '{"ambiguous": false, "interpretations": [{"sql": "SELECT 1", "explanation": "stub"}]}'

    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini",
        client=StubClient(),  # type: ignore[arg-type]
    )
    cr = handler.detect("Q?", schema)
    assert cr.ambiguous is False
    assert len(cr.interpretations) == 1
    assert len(calls) == 1


def test_ambiguous_response_with_three_distinct_interpretations(schema) -> None:
    """The handler should accept any number >= 2 of interpretations, not just 2."""
    three_json = """{
        "ambiguous": true,
        "interpretations": [
            {"sql": "SELECT SUM(total_amount) FROM orders;", "explanation": "All orders"},
            {"sql": "SELECT SUM(total_amount) FROM orders WHERE status='paid';", "explanation": "Paid only"},
            {"sql": "SELECT SUM(amount) FROM payments WHERE status='completed';", "explanation": "Payments only"}
        ]
    }"""
    handler = AmbiguityHandler(
        provider="openai", model="gpt-4o-mini",
        completion_fn=lambda **_: _openai_style_response(three_json),
    )
    cr = handler.detect("Revenue?", schema)
    assert cr.ambiguous is True
    assert len(cr.interpretations) == 3
    sqls = [i.sql for i in cr.interpretations]
    assert len(set(sqls)) == 3
