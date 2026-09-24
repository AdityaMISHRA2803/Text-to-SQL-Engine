"""Tests for the back-translation validator (Phase 2 — README 3.1).

The BackTranslator uses an injected `completion_fn` for the LLM call
and an injected `embedder` for the vectorisation step. These tests
exercise both injection points with canned stubs so no network is ever
touched. The exit-criterion fixture (5 known-wrong-SQL → 100% flagged)
mirrors the README 3.1 success criterion.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from text2sql.config import get_settings
from text2sql.schema import Column, Schema, Table
from text2sql.validation import BackTranslationResult, BackTranslator
from text2sql.validation.back_translator import cosine_similarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _col(name: str, type_: str = "VARCHAR", nullable: bool = True, pk: bool = False) -> Column:
    return Column(name=name, type=type_, nullable=nullable, primary_key=pk)


def _minimal_schema() -> Schema:
    """A small schema sufficient for back-translation tests."""
    return Schema(
        tables=[
            Table(
                name="customers",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("email", "VARCHAR"),
                    _col("country", "VARCHAR"),
                ],
                primary_key=["id"],
            ),
            Table(
                name="orders",
                columns=[
                    _col("id", "INTEGER", nullable=False, pk=True),
                    _col("customer_id", "INTEGER"),
                    _col("total_amount", "NUMERIC"),
                ],
                primary_key=["id"],
            ),
        ]
    )


def _openai_style_response(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def _anthropic_tool_use_response(input_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "back_translate",
                                "arguments": json.dumps(input_obj),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _string_embedder(vectors: list[list[float]]):
    """Build an embedder stub returning ``vectors`` regardless of input order.

    The translator always passes ``[original_question, parsed_question]``
    in that order, so we return the precomputed vectors in the same order.
    """

    def _embed(texts: list[str]) -> list[list[float]]:
        assert len(texts) == 2, "translator should embed exactly two strings"
        return [list(vectors[0]), list(vectors[1])]

    return _embed


# ---------------------------------------------------------------------------
# Pure-math tests: cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_returns_one() -> None:
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_identical_nonzero_vectors_returns_one() -> None:
    a = [3.0, 4.0, 0.0]
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_returns_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_clamps_to_zero() -> None:
    """Cosine of antiparallel vectors is -1, but we clamp to [0, 1]."""
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0


def test_cosine_similarity_empty_returns_zero() -> None:
    assert cosine_similarity([], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], []) == 0.0


def test_cosine_similarity_mismatched_lengths_returns_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_cosine_similarity_known_value() -> None:
    """[1, 0] vs [0.7071, 0.7071] -> ~0.7071."""
    assert cosine_similarity([1.0, 0.0], [0.70710678, 0.70710678]) == pytest.approx(
        0.70710678, abs=1e-6
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_default_threshold_comes_from_settings() -> None:
    bt = BackTranslator()
    assert bt.threshold == get_settings().back_translation_threshold
    assert bt.threshold == pytest.approx(0.75)


def test_threshold_override_via_constructor() -> None:
    bt = BackTranslator(threshold=0.5)
    assert bt.threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Translate happy-path tests
# ---------------------------------------------------------------------------


def test_translate_identical_question_similarity_one_not_flagged() -> None:
    schema = _minimal_schema()
    # Identical vectors → cosine = 1.0 → not flagged.
    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response(
            '{"question": "Show me the top 10 customers by total spend", '
            '"confidence": 0.95}'
        )

    embedder = _string_embedder([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bt = BackTranslator(threshold=0.75, completion_fn=fake_completion_fn)
    result = bt.translate(
        "SELECT * FROM customers LIMIT 10;",
        schema,
        original_question="Show me the top 10 customers by total spend",
        embedder=embedder,
    )
    assert isinstance(result, BackTranslationResult)
    assert result.similarity == pytest.approx(1.0)
    assert result.flagged is False
    assert result.reason is not None
    assert ">=" in result.reason
    assert result.warnings == []
    assert result.model_id == "openai/gpt-4o-mini"


def test_translate_clearly_different_question_flagged() -> None:
    schema = _minimal_schema()
    # Orthogonal vectors → cosine = 0.0 → flagged.
    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response(
            '{"question": "How many books are there?", "confidence": 0.6}'
        )

    embedder = _string_embedder([[1.0, 0.0], [0.0, 1.0]])
    bt = BackTranslator(threshold=0.75, completion_fn=fake_completion_fn)
    result = bt.translate(
        "SELECT COUNT(*) FROM products WHERE category='books';",
        schema,
        original_question="Show me the top 10 customers by total spend",
        embedder=embedder,
    )
    assert result.similarity == pytest.approx(0.0)
    assert result.flagged is True
    assert "below threshold" in (result.reason or "")
    assert result.back_translated_question == "How many books are there?"
    assert result.model_id == "openai/gpt-4o-mini"


def test_translate_threshold_edge_similarity_equal_not_flagged() -> None:
    """similarity == threshold must NOT be flagged (we use <, not <=)."""
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response(
            '{"question": "Same as threshold", "confidence": 0.5}'
        )

    # Two vectors with cosine exactly = 0.75.
    # Construct a pair: a=[1,0], b=[0.75, sqrt(1-0.75^2)]
    import math
    a = [1.0, 0.0]
    b = [0.75, math.sqrt(1 - 0.75 ** 2)]
    embedder = _string_embedder([a, b])
    bt = BackTranslator(threshold=0.75, completion_fn=fake_completion_fn)
    result = bt.translate(
        "SELECT 1;",
        schema,
        original_question="Q?",
        embedder=embedder,
    )
    assert result.similarity == pytest.approx(0.75, abs=1e-9)
    assert result.flagged is False, (
        f"similarity == threshold should NOT be flagged; got {result.similarity}"
    )


# ---------------------------------------------------------------------------
# Stubbed completion_fn shape variants
# ---------------------------------------------------------------------------


def test_translate_openai_shape_dict_response() -> None:
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response(
            '{"question": "Top 10 customers by spend", "confidence": 0.9}'
        )

    embedder = _string_embedder([[1.0, 0.0], [1.0, 0.0]])
    bt = BackTranslator(provider="openai", model="gpt-4o-mini", completion_fn=fake_completion_fn)
    result = bt.translate(
        "SELECT * FROM customers LIMIT 10;",
        schema,
        original_question="Show me the top 10 customers by total spend",
        embedder=embedder,
    )
    assert result.back_translated_question == "Top 10 customers by spend"
    assert result.similarity == pytest.approx(1.0)
    assert result.flagged is False
    assert result.model_id == "openai/gpt-4o-mini"


def test_translate_anthropic_tool_use_dict_response() -> None:
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _anthropic_tool_use_response(
            {"question": "Top customers", "confidence": 0.8}
        )

    embedder = _string_embedder([[1.0, 0.0], [0.9, 0.1]])
    bt = BackTranslator(provider="anthropic", model="claude-3-5-sonnet-latest", completion_fn=fake_completion_fn)
    result = bt.translate(
        "SELECT * FROM customers LIMIT 10;",
        schema,
        original_question="Show me the top 10 customers by total spend",
        embedder=embedder,
    )
    assert result.back_translated_question == "Top customers"
    assert result.model_id == "anthropic/claude-3-5-sonnet-latest"


def test_translate_raw_string_response_falls_back_defensively() -> None:
    """When the LLM returns a plain string (not JSON), the raw text is the question."""
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> str:
        return "How many products are in the books category?"

    embedder = _string_embedder([[1.0, 0.0], [0.0, 1.0]])
    bt = BackTranslator(completion_fn=fake_completion_fn)
    result = bt.translate(
        "SELECT COUNT(*) FROM products WHERE category='books';",
        schema,
        original_question="Show me the top 10 customers by total spend",
        embedder=embedder,
    )
    assert result.back_translated_question == "How many products are in the books category?"
    assert "fallback: non-JSON response" in result.warnings
    assert result.model_id is None  # model_id is None when fallback path was taken
    assert result.flagged is True


# ---------------------------------------------------------------------------
# Defensive parsing: markdown-fenced JSON
# ---------------------------------------------------------------------------


def test_translate_markdown_fenced_json_still_parses() -> None:
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> str:
        return '```json\n{"question": "Top 10 customers", "confidence": 0.8}\n```'

    embedder = _string_embedder([[1.0, 0.0], [1.0, 0.0]])
    bt = BackTranslator(completion_fn=fake_completion_fn)
    result = bt.translate(
        "SELECT * FROM customers LIMIT 10;",
        schema,
        original_question="Show me the top 10 customers by total spend",
        embedder=embedder,
    )
    assert result.back_translated_question == "Top 10 customers"
    assert result.warnings == []
    assert result.model_id == "openai/gpt-4o-mini"
    assert result.flagged is False


# ---------------------------------------------------------------------------
# Embedder errors: graceful degradation
# ---------------------------------------------------------------------------


def test_translate_embedder_error_returns_flagged_with_warning() -> None:
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response('{"question": "Anything", "confidence": 0.5}')

    def broken_embedder(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding API down")

    bt = BackTranslator(completion_fn=fake_completion_fn)
    result = bt.translate(
        "SELECT 1;",
        schema,
        original_question="Q?",
        embedder=broken_embedder,
    )
    assert result.flagged is True
    assert result.similarity == 0.0
    assert any("embedder" in w for w in result.warnings)
    assert any("RuntimeError" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Logging: WARNING emitted with structured payload when flagged
# ---------------------------------------------------------------------------


def test_translate_flagged_emits_warning_log_with_payload(caplog) -> None:
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response('{"question": "How many books?", "confidence": 0.2}')

    embedder = _string_embedder([[1.0, 0.0], [0.0, 1.0]])
    bt = BackTranslator(threshold=0.75, completion_fn=fake_completion_fn)

    with caplog.at_level(logging.WARNING, logger="text2sql.validation.back_translator"):
        result = bt.translate(
            "SELECT COUNT(*) FROM products WHERE category='books';",
            schema,
            original_question="Show me the top 10 customers by total spend",
            embedder=embedder,
        )

    assert result.flagged is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected at least one WARNING log record when flagged"
    payload = warnings[0].__dict__
    assert payload.get("sql") == "SELECT COUNT(*) FROM products WHERE category='books';"
    assert payload.get("original_question") == "Show me the top 10 customers by total spend"
    assert payload.get("back_translated_question") == "How many books?"
    assert payload.get("similarity") == pytest.approx(0.0)
    assert payload.get("threshold") == pytest.approx(0.75)
    assert payload.get("flagged") is True
    assert payload.get("model_id") == "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# Exit-criterion fixture: README 3.1 — 5 injected wrong-SQL → 100% flagged
# ---------------------------------------------------------------------------


_EXIT_CRITERION_ORIGINAL_QUESTION = "Show me the top 10 customers by total spend"


_EXIT_CRITERION_CASES = [
    (
        "SELECT COUNT(*) FROM products WHERE category='books';",
        "How many books are there?",
        [[1.0, 0.0], [0.0, 1.0]],  # orthogonal → cosine = 0.0
    ),
    (
        "SELECT AVG(price) FROM products;",
        "What is the average product price?",
        [[1.0, 0.0], [-1.0, 0.0]],  # opposite → cosine = 0.0 (clamped)
    ),
    (
        "DELETE FROM customers WHERE country='US';",
        "Delete all US customers",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # orthogonal → cosine = 0.0
    ),
    (
        "SELECT id FROM orders ORDER BY created_at DESC LIMIT 5;",
        "Most recent orders",
        [[1.0, 0.0], [0.0, 0.0]],  # zero vector → cosine = 0.0
    ),
    (
        "SELECT name FROM categories;",
        "List all category names",
        [[1.0, 0.0], [0.5, 0.866]],  # 60 degrees → cosine = 0.5, < 0.75
    ),
]


@pytest.mark.parametrize(
    "sql,back_question,vectors",
    _EXIT_CRITERION_CASES,
    ids=[f"case-{i}" for i in range(len(_EXIT_CRITERION_CASES))],
)
def test_exit_criterion_wrong_sql_all_flagged(
    sql: str, back_question: str, vectors: list[list[float]]
) -> None:
    """README 3.1 exit criterion: 5 known-wrong SQL must ALL be flagged."""
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response(
            json.dumps({"question": back_question, "confidence": 0.5})
        )

    embedder = _string_embedder(vectors)
    bt = BackTranslator(threshold=0.75, completion_fn=fake_completion_fn)
    result = bt.translate(
        sql,
        schema,
        original_question=_EXIT_CRITERION_ORIGINAL_QUESTION,
        embedder=embedder,
    )
    assert isinstance(result, BackTranslationResult)
    assert result.flagged is True, (
        f"Expected wrong-SQL fixture to be flagged. "
        f"sql={sql!r}, similarity={result.similarity}, threshold={result.threshold}"
    )
    assert result.reason is not None
    assert "below threshold" in result.reason


def test_exit_criterion_all_five_cases_in_one_loop() -> None:
    """Aggregated run: every one of the 5 fixtures must be flagged."""
    schema = _minimal_schema()
    bt = BackTranslator(threshold=0.75)

    flagged_count = 0
    for sql, back_question, vectors in _EXIT_CRITERION_CASES:
        # Each iteration gets fresh closure variables.
        def fake_completion_fn(
            _bq: str = back_question, **kwargs: Any
        ) -> dict[str, Any]:
            return _openai_style_response(
                json.dumps({"question": _bq, "confidence": 0.5})
            )

        embedder = _string_embedder(vectors)
        result = bt.translate(
            sql,
            schema,
            original_question=_EXIT_CRITERION_ORIGINAL_QUESTION,
            embedder=embedder,
            completion_fn=fake_completion_fn,
        )
        if result.flagged:
            flagged_count += 1

    assert flagged_count == 5, (
        f"README 3.1 exit criterion: expected all 5 wrong-SQL fixtures to be flagged, "
        f"got {flagged_count}/5"
    )


# ---------------------------------------------------------------------------
# Misc defensive paths
# ---------------------------------------------------------------------------


def test_translate_threshold_per_call_override() -> None:
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response('{"question": "Q", "confidence": 0.5}')

    # Similarity = 0.7. With default threshold (0.75) → flagged.
    # With per-call threshold=0.5 → not flagged.
    embedder = _string_embedder([[1.0, 0.0], [0.7, 0.714142842854285]])
    bt = BackTranslator(threshold=0.75, completion_fn=fake_completion_fn)
    flagged_default = bt.translate(
        "SELECT 1;",
        schema,
        original_question="Q?",
        embedder=embedder,
    )
    assert flagged_default.flagged is True

    not_flagged = bt.translate(
        "SELECT 1;",
        schema,
        original_question="Q?",
        embedder=_string_embedder([[1.0, 0.0], [0.7, 0.714142842854285]]),
        threshold=0.5,
    )
    assert not_flagged.flagged is False
    assert not_flagged.threshold == pytest.approx(0.5)


def test_translate_empty_original_question_returns_flagged() -> None:
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response('{"question": "Q", "confidence": 0.5}')

    bt = BackTranslator(completion_fn=fake_completion_fn)
    result = bt.translate(
        "SELECT 1;",
        schema,
        original_question="",
        embedder=_string_embedder([[1.0, 0.0], [1.0, 0.0]]),
    )
    assert result.flagged is True
    assert "empty original_question" in result.warnings


def test_translate_empty_sql_returns_flagged() -> None:
    schema = _minimal_schema()

    def fake_completion_fn(**kwargs: Any) -> dict[str, Any]:
        return _openai_style_response('{"question": "Q", "confidence": 0.5}')

    bt = BackTranslator(completion_fn=fake_completion_fn)
    result = bt.translate(
        "",
        schema,
        original_question="Q?",
        embedder=_string_embedder([[1.0, 0.0], [1.0, 0.0]]),
    )
    assert result.flagged is True
    assert "empty sql" in result.warnings


def test_llm_failure_returns_flagged_with_warning() -> None:
    """If the LLM call raises, the result is still populated and flagged."""
    schema = _minimal_schema()

    def boom_completion_fn(**kwargs: Any) -> None:
        raise RuntimeError("API down")

    bt = BackTranslator(completion_fn=boom_completion_fn)
    result = bt.translate(
        "SELECT 1;",
        schema,
        original_question="Q?",
        embedder=_string_embedder([[1.0, 0.0], [1.0, 0.0]]),
    )
    assert result.flagged is True
    assert any("LLM call failed" in w for w in result.warnings)
    assert result.model_id is None
