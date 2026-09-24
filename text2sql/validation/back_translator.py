"""SQL-to-question back-translation (README 3.1).

After SQL generation, ask the LLM: "What question does this SQL answer?"
Embed both the back-translated question and the original user question;
compute cosine similarity. Flag below `settings.back_translation_threshold`
(default 0.75) as low-confidence translation.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from text2sql.config import get_settings
from text2sql.llm.client import LLMClient
from text2sql.schema import Schema


logger = logging.getLogger("text2sql.validation.back_translator")


#: A callable that turns a list of strings into a list of embedding vectors.
Embedder = Callable[[list[str]], list[list[float]]]


#: Prompt template for the back-translation LLM call.
BACK_TRANSLATION_PROMPT_TEMPLATE = """You are an expert SQL reader. Given a SQL query, describe in one natural-language sentence what question the SQL is answering.

Respond ONLY with a JSON object in this exact shape:
{{"question": "<one-sentence natural-language question>", "confidence": <float 0-1>}}

SQL:
{sql}
"""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two non-zero vectors in [0.0, 1.0].

    Implemented manually so we don't need numpy/scipy at test time. Returns
    ``0.0`` when either input has zero magnitude, when the inputs are of
    different lengths, or when the inputs are empty — the value is clamped
    to ``[0.0, 1.0]`` to absorb tiny floating-point overshoot.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cos = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    # Clamp to handle floating-point noise (e.g. 1.0000000002).
    if cos > 1.0:
        return 1.0
    if cos < 0.0:
        return 0.0
    return cos


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` fences if present (defensive parsing)."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _extract_first_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` substring from ``text``.

    Useful when the LLM adds trailing prose after a JSON object.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def _normalise_completion_response(response: Any) -> str:
    """Extract a JSON string from whatever shape ``completion_fn`` returned.

    Supported shapes (mirrors the dual-provider `LLMClient` abstraction):

    - ``str``: returned as-is (already a content string).
    - ``dict`` with ``choices[0].message.content``: standard OpenAI shape.
    - ``dict`` with ``choices[0].message.tool_calls[0].function.arguments``:
      Anthropic tool_use shape (litellm-normalised).
    - ``dict`` with top-level ``content=[{"type": "tool_use", "input": ...}]``:
      raw Anthropic shape.
    """
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        # Try to use LLMClient._extract_content on object-shaped responses
        # (e.g. pydantic/openai ModelResponse objects).
        try:
            return LLMClient._extract_content(response)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not extract content from response: {exc}") from exc

    # OpenAI / litellm-normalised Anthropic envelope.
    choices = response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        # Anthropic tool_use via litellm.
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments")
            if isinstance(arguments, str):
                return arguments
            if isinstance(arguments, dict):
                return json.dumps(arguments)
        content = message.get("content")
        if isinstance(content, str):
            return content
    # Raw Anthropic shape.
    content = response.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return json.dumps(block.get("input", {}))
        # Fallback: concatenate any text blocks.
        text_chunks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if text_chunks:
            return "".join(text_chunks)
    raise ValueError("Could not extract content from completion response")


class BackTranslationResult(BaseModel):
    """Outcome of `BackTranslator.translate`.

    Attributes:
        sql: The SQL that was passed to the back-translator.
        original_question: The user's original natural-language question.
        back_translated_question: The question the LLM claims the SQL answers.
        similarity: Cosine similarity between the embeddings of the two
            questions, clamped to ``[0.0, 1.0]``.
        flagged: ``True`` when ``similarity < threshold``.
        threshold: The threshold used for flagging (resolved from
            ``Settings.back_translation_threshold`` or an explicit override).
        reason: Human-readable explanation of the flagging decision.
        warnings: Non-fatal diagnostic messages (e.g. ``"fallback: non-JSON
            response"`` when the LLM returned prose instead of JSON).
        model_id: Identifier of the LLM that produced the back-translation,
            or ``None`` when the response was the defensive fallback.
    """

    sql: str
    original_question: str
    back_translated_question: str
    similarity: float = Field(ge=0.0, le=1.0)
    flagged: bool
    threshold: float = Field(ge=0.0, le=1.0)
    reason: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    model_id: Optional[str] = None


class BackTranslator:
    """Translate generated SQL back into a natural-language question and compare it to the original question.

    The translator routes its LLM call through `LLMClient` (or an
    injected ``completion_fn``) and uses an injected ``embedder`` to
    vectorise both questions. Cosine similarity is computed manually so
    we do not depend on numpy/scipy. Tests pass stubs for both, so no
    network is ever touched.

    Public API::

        bt = BackTranslator(provider="openai", model="gpt-4o-mini")
        result = bt.translate(
            sql,
            schema,
            original_question="Show me the top 10 customers by total spend",
            embedder=my_embedder,
        )
        result.flagged          # True / False
        result.similarity       # float in [0.0, 1.0]
        result.reason           # e.g. "similarity 0.42 below threshold 0.75"
    """

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        threshold: Optional[float] = None,
        completion_fn: Optional[Callable[..., Any]] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = float(get_settings().back_translation_threshold)
        self._completion_fn = completion_fn
        self._client = client or LLMClient(
            provider=provider, model=model, api_key=api_key
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def translate(
        self,
        sql: str,
        schema: Schema,
        *,
        original_question: str,
        embedder: Embedder,
        completion_fn: Optional[Callable[..., Any]] = None,
        threshold: Optional[float] = None,
    ) -> BackTranslationResult:
        """Back-translate ``sql`` to a question and compare against ``original_question``.

        Args:
            sql: The SQL statement to back-translate.
            schema: The schema the SQL is meant to run against. (Currently
                not embedded in the prompt; reserved for future cross-checks.)
            original_question: The user's original natural-language question.
            embedder: Callable taking a list of strings and returning a
                list of embedding vectors (same length as input).
            completion_fn: Optional per-call override for the LLM call.
                Falls back to the constructor-injected ``completion_fn``
                and finally to the real ``LLMClient.complete``.
            threshold: Optional per-call override for the flagging
                threshold. Falls back to the constructor-injected value
                and finally to ``Settings.back_translation_threshold``.

        Returns:
            A populated :class:`BackTranslationResult`.
        """
        effective_threshold = (
            float(threshold) if threshold is not None else self.threshold
        )
        effective_completion_fn = completion_fn or self._completion_fn

        # ---- Defensive inputs -------------------------------------------------
        if not original_question or not original_question.strip():
            warnings = ["empty original_question"]
            return self._build_result(
                sql=sql,
                original_question=original_question,
                back_translated_question="",
                similarity=0.0,
                threshold=effective_threshold,
                flagged=True,
                reason="original_question was empty; cannot compare",
                warnings=warnings,
                model_id=None,
            )

        if not sql or not sql.strip():
            warnings = ["empty sql"]
            return self._build_result(
                sql=sql or "",
                original_question=original_question,
                back_translated_question="",
                similarity=0.0,
                threshold=effective_threshold,
                flagged=True,
                reason="sql was empty; nothing to back-translate",
                warnings=warnings,
                model_id=None,
            )

        prompt = BACK_TRANSLATION_PROMPT_TEMPLATE.format(sql=sql.strip())
        model_id = f"{self.provider}/{self.model}"

        # ---- LLM call (with defensive fallback) ------------------------------
        # When a completion_fn is supplied we call it directly rather than
        # going through `LLMClient.complete`, because that method always
        # post-processes the response with `_extract_content`, which only
        # understands the OpenAI `choices[0].message.content` shape. Calling
        # completion_fn directly lets us normalise OpenAI / Anthropic /
        # raw-string shapes in `_normalise_completion_response`.
        try:
            if effective_completion_fn is not None:
                response = effective_completion_fn(prompt=prompt)
                raw = _normalise_completion_response(response)
            else:
                raw = self._client.complete(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "back_translation.llm_failed",
                extra={
                    "sql": sql,
                    "original_question": original_question,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return self._build_result(
                sql=sql,
                original_question=original_question,
                back_translated_question="",
                similarity=0.0,
                threshold=effective_threshold,
                flagged=True,
                reason=f"LLM call failed: {type(exc).__name__}",
                warnings=[f"LLM call failed: {type(exc).__name__}"],
                model_id=None,
            )

        # ---- Parse the LLM response ------------------------------------------
        parsed_question, confidence, parse_warnings = self._parse_back_translation(
            raw
        )
        warnings: list[str] = list(parse_warnings)
        resolved_model_id = model_id if not parse_warnings else None

        # ---- Embed both questions --------------------------------------------
        try:
            vectors = embedder([original_question, parsed_question])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "back_translation.embedder_failed",
                extra={
                    "sql": sql,
                    "original_question": original_question,
                    "back_translated_question": parsed_question,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            warnings.append(f"embedder failed: {type(exc).__name__}")
            return self._build_result(
                sql=sql,
                original_question=original_question,
                back_translated_question=parsed_question,
                similarity=0.0,
                threshold=effective_threshold,
                flagged=True,
                reason=f"embedder raised {type(exc).__name__}; flagged as low-confidence",
                warnings=warnings,
                model_id=resolved_model_id,
            )

        if (
            not isinstance(vectors, list)
            or len(vectors) != 2
            or not all(isinstance(v, list) for v in vectors)
        ):
            warnings.append("embedder returned unexpected shape")
            return self._build_result(
                sql=sql,
                original_question=original_question,
                back_translated_question=parsed_question,
                similarity=0.0,
                threshold=effective_threshold,
                flagged=True,
                reason="embedder did not return two vectors; flagged as low-confidence",
                warnings=warnings,
                model_id=resolved_model_id,
            )

        similarity = cosine_similarity(vectors[0], vectors[1])

        flagged = similarity < effective_threshold
        reason = self._make_reason(similarity, effective_threshold, flagged)

        result = self._build_result(
            sql=sql,
            original_question=original_question,
            back_translated_question=parsed_question,
            similarity=similarity,
            threshold=effective_threshold,
            flagged=flagged,
            reason=reason,
            warnings=warnings,
            model_id=resolved_model_id,
        )

        if flagged:
            payload = {
                "sql": sql,
                "original_question": original_question,
                "back_translated_question": parsed_question,
                "similarity": similarity,
                "threshold": effective_threshold,
                "model_id": resolved_model_id,
                "flagged": True,
                "confidence": confidence,
            }
            logger.warning("back_translation.low_similarity", extra=payload)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_back_translation(
        raw: str,
    ) -> tuple[str, Optional[float], list[str]]:
        """Parse the raw LLM response into ``(question, confidence, warnings)``.

        If the response is not valid JSON, the raw text is used as the
        back-translated question (after stripping markdown fences) and
        ``warnings=["fallback: non-JSON response"]`` is returned.
        """
        warnings: list[str] = []
        if raw is None:
            return "", None, ["fallback: non-JSON response"]
        text = _strip_markdown_fences(str(raw))
        if not text:
            return "", None, ["fallback: non-JSON response"]

        json_blob = _extract_first_json_object(text)
        try:
            data = json.loads(json_blob)
        except json.JSONDecodeError:
            # Defensive fallback: treat the raw response as the question.
            return text.strip(), None, ["fallback: non-JSON response"]

        if not isinstance(data, dict):
            return text.strip(), None, ["fallback: non-JSON response"]

        question = str(data.get("question", "")).strip()
        confidence_raw = data.get("confidence")
        confidence: Optional[float]
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence = None

        if not question:
            # JSON parsed but had no usable question field — fall back to
            # the raw text so we still surface *something* to the user.
            return text.strip(), confidence, ["fallback: empty 'question' field"]

        return question, confidence, warnings

    @staticmethod
    def _make_reason(similarity: float, threshold: float, flagged: bool) -> str:
        """Build a short human-readable reason string."""
        sim_str = f"{similarity:.4f}"
        thr_str = f"{threshold:.4f}"
        if flagged:
            return f"similarity {sim_str} below threshold {thr_str}"
        return f"similarity {sim_str} >= threshold {thr_str}"

    @staticmethod
    def _build_result(
        *,
        sql: str,
        original_question: str,
        back_translated_question: str,
        similarity: float,
        threshold: float,
        flagged: bool,
        reason: str,
        warnings: list[str],
        model_id: Optional[str],
    ) -> BackTranslationResult:
        return BackTranslationResult(
            sql=sql,
            original_question=original_question,
            back_translated_question=back_translated_question,
            similarity=similarity,
            flagged=flagged,
            threshold=threshold,
            reason=reason,
            warnings=list(warnings),
            model_id=model_id,
        )


__all__ = [
    "BackTranslationResult",
    "BackTranslator",
    "Embedder",
    "BACK_TRANSLATION_PROMPT_TEMPLATE",
    "cosine_similarity",
]
