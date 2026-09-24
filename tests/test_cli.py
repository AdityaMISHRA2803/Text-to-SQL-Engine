"""End-to-end tests for the Phase-1 CLI pipeline.

The CLI must compose schema extraction -> relevance filtering -> prompt
building -> ambiguity detection and print both the constructed prompt
and the resulting ClarificationRequest. These tests run the pipeline
via `run_pipeline()` against the seeded testcontainers fixture with
injected embedder + completion_fn so no network or API keys are needed.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from text2sql.cli import (
    _format_clarification,
    _resolve_embedder,
    run_pipeline,
    _fake_embedder,
    _tokenize,
)
from text2sql.ambiguity import ClarificationRequest, Interpretation


def _canned_ambiguous_completion_fn(**kwargs):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "ambiguous": True,
                        "interpretations": [
                            {
                                "sql": "SELECT SUM(total_amount) FROM orders WHERE status = 'paid';",
                                "explanation": "Gross revenue as the sum of all paid order totals.",
                            },
                            {
                                "sql": "SELECT SUM(amount) FROM payments WHERE status = 'completed';",
                                "explanation": "Net revenue as the sum of all completed payment amounts.",
                            },
                        ],
                    }),
                },
            },
        ],
    }


def _canned_unambiguous_completion_fn(**kwargs):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "ambiguous": False,
                        "interpretations": [
                            {
                                "sql": "SELECT c.id, c.email, SUM(o.total_amount) AS revenue FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.email ORDER BY revenue DESC LIMIT 10;",
                                "explanation": "Top 10 customers ranked by total spend.",
                            },
                        ],
                    }),
                },
            },
        ],
    }


def _fake_embedder_for_tests():
    """Use the CLI's own fake embedder — it is deterministic and key-free."""
    return _fake_embedder()


# ---------------------------------------------------------------------------
# run_pipeline end-to-end
# ---------------------------------------------------------------------------


def test_run_pipeline_with_ambiguous_question_returns_prompt_and_clarification(fresh_db) -> None:
    """Full pipeline against the seeded Postgres fixture with mocked LLM.

    Asserts:
      - prompt is a non-empty string that mentions schema artifacts
      - ClarificationRequest.ambiguous is True
      - exactly 2 interpretations with distinct SQL sketches
      - model_id is populated
    """
    prompt, cr = run_pipeline(
        "What's the total revenue?",
        embedder=_fake_embedder_for_tests(),
        completion_fn=_canned_ambiguous_completion_fn,
        database_url=fresh_db.url.render_as_string(hide_password=False),
    )
    assert isinstance(prompt, str) and len(prompt) > 0
    # Prompt should contain schema artifacts (table names, column names).
    assert "orders" in prompt or "customers" in prompt or "payments" in prompt
    assert isinstance(cr, ClarificationRequest)
    assert cr.ambiguous is True
    assert len(cr.interpretations) == 2
    sqls = [i.sql for i in cr.interpretations]
    assert len(set(sqls)) == 2
    assert cr.model_id is not None and "/" in cr.model_id


def test_run_pipeline_with_unambiguous_question_returns_single_interpretation(fresh_db) -> None:
    prompt, cr = run_pipeline(
        "Show me the top 10 customers by total spend",
        embedder=_fake_embedder_for_tests(),
        completion_fn=_canned_unambiguous_completion_fn,
        database_url=fresh_db.url.render_as_string(hide_password=False),
    )
    assert isinstance(prompt, str) and len(prompt) > 0
    assert cr.ambiguous is False
    assert len(cr.interpretations) == 1
    assert cr.interpretations[0].sql


def test_run_pipeline_response_shape_is_uniform(fresh_db) -> None:
    """Both ambiguous and unambiguous inputs return the same ClarificationRequest type."""
    _, amb_cr = run_pipeline(
        "What's the revenue?",
        embedder=_fake_embedder_for_tests(),
        completion_fn=_canned_ambiguous_completion_fn,
        database_url=fresh_db.url.render_as_string(hide_password=False),
    )
    _, unamb_cr = run_pipeline(
        "Show customers",
        embedder=_fake_embedder_for_tests(),
        completion_fn=_canned_unambiguous_completion_fn,
        database_url=fresh_db.url.render_as_string(hide_password=False),
    )
    assert type(amb_cr) is type(unamb_cr) is ClarificationRequest
    assert set(amb_cr.model_dump().keys()) == set(unamb_cr.model_dump().keys())


def test_run_pipeline_without_completion_fn_returns_passthrough(fresh_db) -> None:
    """If no completion_fn is provided AND no API key is set, the handler
    catches the LLM error and returns a defensive passthrough single
    interpretation. The pipeline must not crash."""
    prompt, cr = run_pipeline(
        "What's the revenue?",
        embedder=_fake_embedder_for_tests(),
        completion_fn=None,
        database_url=fresh_db.url.render_as_string(hide_password=False),
    )
    assert isinstance(prompt, str) and len(prompt) > 0
    assert isinstance(cr, ClarificationRequest)
    assert cr.ambiguous is False
    assert len(cr.interpretations) >= 1


def test_run_pipeline_filters_schema_before_building_prompt(fresh_db) -> None:
    """The relevance filter should reduce the schema to relevant tables.

    A question about 'customers' should drop irrelevant tables (e.g.,
    payments) from the prompt. We can't assert exact drops because the
    fake embedder is weak, but we can verify the prompt mentions SOME
    tables (proves schema was extracted) and is non-empty.
    """
    prompt, _cr = run_pipeline(
        "Which customers spent the most?",
        embedder=_fake_embedder_for_tests(),
        completion_fn=_canned_unambiguous_completion_fn,
        database_url=fresh_db.url.render_as_string(hide_password=False),
    )
    # Customers, orders, order_items are plausibly relevant; payments may be filtered.
    assert "customers" in prompt.lower()
    # The prompt should contain some structure markers.
    assert "Schema" in prompt or "Tables" in prompt or "Columns" in prompt


# ---------------------------------------------------------------------------
# CLI formatting helper
# ---------------------------------------------------------------------------


def test_format_clarification_includes_all_interpretations() -> None:
    cr = ClarificationRequest(
        question="Revenue?",
        ambiguous=True,
        interpretations=[
            Interpretation(sql="SELECT 1", explanation="one"),
            Interpretation(sql="SELECT 2", explanation="two"),
        ],
        model_id="openai/gpt-4o-mini",
    )
    text = _format_clarification(cr)
    assert "Revenue?" in text
    assert "Ambiguous:   True" in text
    assert "Interpretations (2)" in text
    assert "SELECT 1" in text
    assert "SELECT 2" in text
    assert "openai/gpt-4o-mini" in text


def test_format_clarification_handles_no_assumptions() -> None:
    cr = ClarificationRequest(
        question="Q?",
        ambiguous=False,
        interpretations=[Interpretation(sql="SELECT 1", explanation="x")],
    )
    text = _format_clarification(cr)
    assert "Ambiguous:   False" in text
    assert "Interpretations (1)" in text


# ---------------------------------------------------------------------------
# Embedder resolution helper
# ---------------------------------------------------------------------------


def test_resolve_embedder_falls_back_to_fake_when_no_api_key(monkeypatch) -> None:
    """Without API keys for the chosen provider, _resolve_embedder should
    return the deterministic fake embedder rather than raising."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    # Need to invalidate the lru_cache on get_settings so env changes take effect.
    from text2sql.config import get_settings
    get_settings.cache_clear()
    try:
        embedder, source = _resolve_embedder("openai", api_key=None)
        assert callable(embedder)
        assert source == "fake"
        # And it actually returns the right shape.
        out = embedder(["hello world", "foo bar baz"])
        assert len(out) == 2
        assert all(len(v) > 0 for v in out)
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# _tokenize helper
# ---------------------------------------------------------------------------


def test_tokenize_drops_stopwords_and_short_tokens() -> None:
    toks = _tokenize("Show me the customers who bought something in 2024")
    assert "show" not in toks  # stopword
    assert "the" not in toks    # stopword
    assert "me" not in toks     # stopword
    assert "customers" in toks
    assert "bought" in toks
    assert "2024" in toks


def test_tokenize_handles_punctuation_and_case() -> None:
    toks = _tokenize("Hello, WORLD! Foo_Bar 123")
    assert "hello" in toks
    assert "world" in toks
    assert "foo_bar" in toks
    assert "123" in toks
