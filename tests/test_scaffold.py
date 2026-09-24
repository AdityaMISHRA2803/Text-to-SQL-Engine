"""Scaffold smoke tests for Phase 1.

These verify the package skeleton imports cleanly and that the
seed dataset round-trips through the test database. They will be
expanded in later phases as Schema, PromptBuilder, RelevanceFilter,
ClarificationRequest, and LLMClient gain real implementations.
"""
from __future__ import annotations

from sqlalchemy import text

from text2sql.ambiguity import ClarificationRequest
from text2sql.config import get_settings
from text2sql.llm import LLMClient
from text2sql.prompts import PromptBuilder
from text2sql.relevance import RelevanceFilter
from text2sql.schema import Schema
from text2sql.seed_dataset import Base


def test_imports() -> None:
    """All five Phase 1 stub classes must be importable."""
    assert Schema is not None
    assert PromptBuilder is not None
    assert RelevanceFilter is not None
    assert ClarificationRequest is not None
    assert LLMClient is not None


def test_settings_loads() -> None:
    s = get_settings()
    assert s.database_url.startswith("postgresql")
    assert s.llm_provider in {"openai", "anthropic"}


def test_llm_client_dispatches_to_injected_completion_fn() -> None:
    """LLMClient.complete must use the injected completion_fn instead of litellm."""
    client = LLMClient(provider="openai", model="gpt-4o-mini")

    class FakeMessage:
        content = "hello from fake llm"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_completion_fn(prompt, **_kwargs):  # noqa: ANN001 - injection shape
        assert prompt == "ping"
        return FakeResponse()

    out = client.complete("ping", completion_fn=fake_completion_fn)
    assert out == "hello from fake llm"


def test_llm_client_embed_injection() -> None:
    """LLMClient.embed must use the injected embedding_fn instead of litellm."""
    client = LLMClient(provider="openai", model="gpt-4o-mini")

    class FakeItem:
        embedding = [0.1, 0.2, 0.3]

    class FakeResponse:
        data = [FakeItem(), FakeItem()]

    def fake_embedding_fn(input, **_kwargs):  # noqa: A002 - matches litellm kwarg
        return FakeResponse()

    out = client.embed(["a", "b"], embedding_fn=fake_embedding_fn)
    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_seed_round_trip(fresh_db) -> None:
    """The fresh_db fixture should produce a populated schema."""
    with fresh_db.connect() as conn:
        for table in ("categories", "customers", "products", "orders", "order_items", "payments"):
            n = conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            assert n and n > 0, f"{table} should be non-empty after seed"


def test_seed_schema_tables_present(fresh_db) -> None:
    """All six tables should exist after the fixture runs Base.metadata.create_all."""
    expected = {"categories", "customers", "products", "orders", "order_items", "payments"}
    actual = set(Base.metadata.tables.keys())
    assert expected.issubset(actual)
