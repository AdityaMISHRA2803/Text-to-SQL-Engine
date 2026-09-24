"""Provider-agnostic embedder interface.

The `Embedder` type is a callable that takes a list of strings and returns
a list of equal-length float vectors. Any callable with this signature
satisfies the interface — no inheritance required. This makes it trivial
to swap providers (OpenAI text-embedding-3-small, Voyage, Cohere, local
sentence-transformers, or a deterministic test double) without touching
the relevance-filter code.

Why this lives in `text2sql/llm/`: the relevance filter imports `Embedder`
from here so the dependency arrow points from `relevance/` → `llm/`, never
the other way. Adding a new provider later means writing a new factory
function in this file (or in a sibling module) and nothing else changes.
"""
from __future__ import annotations

from typing import Callable, Optional

from text2sql.llm.client import LLMClient


#: Protocol type for any embedder. Signature: (texts) -> list[vector].
#: All texts in one call must be embedded into the same vector space so
#: cosine similarity is meaningful.
Embedder = Callable[[list[str]], list[list[float]]]


def default_embedder(
    provider: str = "openai",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Embedder:
    """Return a default `Embedder` for the given provider.

    OpenAI: routes to `LLMClient(provider="openai", model=model).embed`,
    which under the hood calls `litellm.embedding` with
    `text-embedding-3-small` (1536 dimensions). This is the recommended
    production path; latency is ~50-200ms per batch of ~20 strings.

    Anthropic: Anthropic has no native embeddings API as of this writing,
    so we raise `NotImplementedError` with an actionable message. The
    provider-pluggable design means callers can substitute Voyage AI,
    Cohere, or a local sentence-transformers model via the
    `embedder=` constructor argument of `RelevanceFilter` without
    touching this function.

    Args:
        provider: "openai" or "anthropic". Default "openai".
        model: Optional explicit model name. Defaults to
            "text-embedding-3-small" for OpenAI.
        api_key: Optional API key override; if None, falls back to the
            provider's key from Settings().

    Returns:
        An `Embedder` callable.

    Raises:
        NotImplementedError: If `provider="anthropic"`.
        ValueError: If `provider` is neither "openai" nor "anthropic".
    """
    if provider == "anthropic":
        raise NotImplementedError(
            "Anthropic has no native embeddings API. Provide a custom embedder "
            "(e.g. wrapping Voyage AI or a local sentence-transformers model) "
            "to RelevanceFilter(embedder=...)."
        )
    if provider != "openai":
        raise ValueError(f"Unsupported provider: {provider!r}; use 'openai' or pass a custom embedder")
    chosen_model = model or "text-embedding-3-small"
    client = LLMClient(provider="openai", model=chosen_model, api_key=api_key)

    def _embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return client.embed(texts)

    return _embed


__all__ = ["Embedder", "default_embedder"]
