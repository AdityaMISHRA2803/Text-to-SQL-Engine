"""LLM client abstraction over LiteLLM.

Provides a single interface for completion and embedding across providers,
with dependency-injection hooks so unit tests can swap in mocks without
hitting the network.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from text2sql.config import get_settings

try:
    import litellm  # type: ignore
except ImportError:  # pragma: no cover - litellm is a required dependency
    litellm = None  # type: ignore


EmbedFn = Callable[..., Any]
CompleteFn = Callable[..., Any]


class LLMClient:
    """Thin wrapper around ``litellm.completion`` / ``litellm.embedding``."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
    ) -> None:
        if provider not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported provider: {provider!r}")
        self.provider = provider
        self.model = model
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_api_key(self, override: Optional[str]) -> Optional[str]:
        if override is not None:
            return override
        if self._api_key is not None:
            return self._api_key
        settings = get_settings()
        if self.provider == "openai":
            return settings.openai_api_key
        return settings.anthropic_api_key

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            message = choices[0].message
            content = getattr(message, "content", None)
            if content is not None:
                return content
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if content is not None:
                    return content
        raise RuntimeError("Unexpected LLM response shape: no choices[0].message.content")

    @staticmethod
    def _extract_embeddings(response: Any) -> list[list[float]]:
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")
        if data is None:
            raise RuntimeError("Unexpected embedding response shape: missing `data`")
        embeddings: list[list[float]] = []
        for item in data:
            embedding = getattr(item, "embedding", None)
            if embedding is None and isinstance(item, dict):
                embedding = item.get("embedding")
            if embedding is None:
                raise RuntimeError("Missing embedding vector for item")
            embeddings.append(list(embedding))
        return embeddings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        api_key: Optional[str] = None,
        completion_fn: Optional[CompleteFn] = None,
        **kwargs: Any,
    ) -> str:
        """Return a string completion for ``prompt``.

        ``completion_fn`` is an injection point for tests; when provided
        it is called instead of ``litellm.completion``.
        """
        if completion_fn is not None:
            result = completion_fn(prompt=prompt, **kwargs)
            return self._extract_content(result)
        if litellm is None:
            raise RuntimeError("litellm is not installed; install with `pip install litellm`")

        resolved_key = self._resolve_api_key(api_key)
        model_name = f"{self.provider}/{self.model}"
        call_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if resolved_key is not None:
            call_kwargs["api_key"] = resolved_key
        call_kwargs.update(kwargs)
        response = litellm.completion(**call_kwargs)
        return self._extract_content(response)

    def embed(
        self,
        texts: list[str],
        *,
        api_key: Optional[str] = None,
        embedding_fn: Optional[EmbedFn] = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Return embedding vectors for ``texts``."""
        if embedding_fn is not None:
            result = embedding_fn(input=texts, **kwargs)
            return self._extract_embeddings(result)
        if litellm is None:
            raise RuntimeError("litellm is not installed; install with `pip install litellm`")
        if self.provider != "openai":
            raise NotImplementedError(
                "Use OpenAI embeddings or a local model for now; "
                "Anthropic has no native embeddings API"
            )
        resolved_key = self._resolve_api_key(api_key)
        call_kwargs: dict[str, Any] = {
            "model": "text-embedding-3-small",
            "input": texts,
        }
        if resolved_key is not None:
            call_kwargs["api_key"] = resolved_key
        call_kwargs.update(kwargs)
        response = litellm.embedding(**call_kwargs)
        return self._extract_embeddings(response)
