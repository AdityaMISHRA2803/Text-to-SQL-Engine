"""Completion-fn factories for the eval suite.

Two modes:

* **Stub** (``make_stub_completion_fn``) — pure-Python stand-in for tests that
  want to exercise the eval runner's plumbing without an LLM. Routes by
  substring match against a ``prompt_substring → response`` map.

* **Live** (``make_live_completion_fn``) — wraps ``litellm.completion`` and
  returns the raw response in whichever shape the generator / back-translator
  / multi-query validator expects. Requires ``OPENAI_API_KEY`` or
  ``ANTHROPIC_API_KEY`` depending on the provider.

The eval runner's stub-mode pipeline does **not** use a completion_fn at all —
it builds ``GeneratedSQL`` directly from the golden SQL. This module only
provides ``completion_fn`` factories for live mode and for unit tests of the
runner.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

import litellm  # type: ignore

from text2sql.config import get_settings


# ---------------------------------------------------------------------------
# Stub factory (unit tests only)
# ---------------------------------------------------------------------------


def make_stub_completion_fn(
    responses: Mapping[str, str],
    *,
    default_response: str = "{}",
) -> Callable[..., Any]:
    """Build a stub completion_fn that routes by substring.

    Args:
        responses: ordered mapping ``{prompt_substring: response_string}``.
            When the runtime prompt contains a key (case-insensitive), the
            corresponding value is returned as the completion. The first
            matching key wins.
        default_response: returned when no key matches.

    Returns:
        A ``completion_fn(prompt=..., **kwargs)``-shaped callable.

    Note:
        This is intentionally trivial — it exists to give unit tests of the
        runner's plumbing a predictable substitute for a real LLM. The eval
        runner's stub-mode pipeline constructs ``GeneratedSQL`` directly and
        never calls this factory.
    """
    keys: list[str] = [k.lower() for k in responses.keys()]

    def _fn(prompt: str = "", **_kwargs: Any) -> str:
        lowered = (prompt or "").lower()
        for key in keys:
            if key in lowered:
                return responses[key]
        return default_response

    return _fn


# ---------------------------------------------------------------------------
# Live factory
# ---------------------------------------------------------------------------


def make_live_completion_fn(
    provider: str,
    model: str,
    *,
    api_key: Optional[str] = None,
) -> Callable[..., Any]:
    """Build a completion_fn that delegates to ``litellm.completion``.

    The returned callable matches the contract used by
    ``SQLGenerator``/``BackTranslator``/``MultiQueryValidator``: it accepts
    ``(prompt=..., **kwargs)`` and returns the **raw** litellm response
    object. The generator's ``_normalise_response`` handles extracting
    content from whatever shape litellm produced.

    Args:
        provider: ``"openai"`` or ``"anthropic"``.
        model: model name without provider prefix.
        api_key: explicit key. If ``None``, the project's ``Settings`` is
            consulted for the matching provider key. Raises ``RuntimeError``
            if no key can be resolved.

    Returns:
        A ``completion_fn(prompt=..., **kwargs)``-shaped callable that
        calls ``litellm.completion`` and returns the raw response.
    """
    if provider not in {"openai", "anthropic"}:
        raise ValueError(f"Unsupported provider: {provider!r}")
    settings = get_settings()
    resolved_key = (
        api_key
        or (settings.openai_api_key if provider == "openai" else settings.anthropic_api_key)
        or os.environ.get("OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY")
    )
    if not resolved_key:
        raise RuntimeError(
            f"No API key resolved for provider {provider!r}. "
            "Set OPENAI_API_KEY / ANTHROPIC_API_KEY in the environment "
            "or pass api_key= explicitly."
        )

    model_name = f"{provider}/{model}"

    def _fn(prompt: str = "", **kwargs: Any) -> Any:
        call_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt or ""}],
            "api_key": resolved_key,
        }
        # If the caller passed response_format / tools (used by SQLGenerator),
        # let them override defaults.
        for k, v in kwargs.items():
            if v is not None:
                call_kwargs[k] = v
        return litellm.completion(**call_kwargs)

    return _fn
