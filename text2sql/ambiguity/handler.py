"""Ambiguity detection — when to ask the user a clarifying question.

Pipeline:
  1. Build a compact JSON summary of the schema (column names + types
     only — sample values are not needed for ambiguity detection and
     would burn tokens).
  2. Construct a prompt asking the LLM to output either:
       {"ambiguous": true,  "interpretations": [{sql, explanation}, ...]}  (>=2)
       {"ambiguous": false, "interpretations": [{sql, explanation}]}       (passthrough)
  3. Call the LLM through the dual-provider `LLMClient.complete()`
     abstraction. Test injection point: pass `completion_fn=` to the
     handler (or to `LLMClient` directly) to bypass the network.
  4. Parse the JSON response into a `ClarificationRequest`.
  5. Defensive fallback: if JSON parsing fails, if the LLM returns an
     empty interpretations list, or if the question is empty, return
     a single-interpretation passthrough with the question echoed and
     a generic SQL sketch. Better to under-detect ambiguity than to
     crash the pipeline.

The response SHAPE is identical regardless of which path triggered:
Phase 2 / downstream consumers always see a `ClarificationRequest`
with `interpretations: list[Interpretation]` and an `ambiguous: bool`
flag. The boolean lets callers branch if they want to, but they can
also just iterate `cr.interpretations` and treat each as a candidate.
"""
from __future__ import annotations

import json
import re
import textwrap
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from text2sql.llm.client import LLMClient
from text2sql.schema import Schema


#: System prompt template. The LLM is told to output strict JSON so we
#: can parse it deterministically. We also constrain the response with
#: field hints to keep the model on-task.
_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a SQL ambiguity detector for a text-to-SQL system.

    Given a natural-language question and a database schema, decide
    whether the question has one clear interpretation or multiple.

    Respond ONLY with a JSON object (no prose, no markdown fences) in
    this exact shape:

    {
      "ambiguous": <true|false>,
      "interpretations": [
        {"sql": "<example SQL sketch>", "explanation": "<short natural-language explanation>"},
        ...
      ]
    }

    Rules:
    - If the question is unambiguous (only one reasonable SQL
      interpretation), set "ambiguous": false and provide EXACTLY ONE
      interpretation.
    - If the question is ambiguous (multiple reasonable SQL
      interpretations, e.g. "revenue" without specifying gross/net or
      orders/payments), set "ambiguous": true and provide AT LEAST TWO
      interpretations with DISTINCT SQL sketches.
    - Each SQL sketch must be a syntactically plausible SELECT statement
      referencing the actual table and column names from the schema.
    - Each explanation should be one short sentence describing what
      that interpretation assumes.
""")


def _build_user_prompt(question: str, schema: Schema) -> str:
    """Compose the user-role prompt: schema summary + question."""
    schema_json = schema.to_compact_json()
    return (
        "Schema (JSON):\n"
        f"{schema_json}\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Now respond with the JSON object as specified."
    )


def _parse_llm_response(raw: str, question: str) -> ClarificationRequest:
    """Parse the LLM's raw text into a ClarificationRequest.

    Tolerant of small formatting quirks:
      - strips ```json ... ``` fences if present
      - extracts the first {...} JSON object from the response
      - tolerates trailing commas (via regex sanitisation)
    """
    text = raw.strip()
    # Strip markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # Extract the first {...} block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    # Remove trailing commas before } or ].
    text = re.sub(r",\s*([}\]])", r"\1", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response was not a JSON object")
    interpretations_data = data.get("interpretations") or []
    if not isinstance(interpretations_data, list) or len(interpretations_data) == 0:
        raise ValueError("LLM response had no interpretations array")
    interpretations = [
        Interpretation(
            sql=str(item.get("sql", "")).strip(),
            explanation=str(item.get("explanation", "")).strip(),
            assumptions=list(item.get("assumptions") or []),
        )
        for item in interpretations_data
    ]
    # Sanity: each interpretation must have non-empty sql and explanation.
    interpretations = [i for i in interpretations if i.sql and i.explanation]
    if not interpretations:
        raise ValueError("All interpretations were empty after sanitisation")
    explicit_flag = data.get("ambiguous")
    # Treat explicit True or missing as ambiguous when >=2 interpretations exist.
    # Trust explicit False (the LLM said the question is unambiguous).
    ambiguous = len(interpretations) >= 2 and explicit_flag is not False
    return ClarificationRequest(
        question=question,
        interpretations=interpretations,
        ambiguous=ambiguous,
    )


def _passthrough(question: str, reason: str) -> ClarificationRequest:
    """Build a single-interpretation passthrough used for unambiguous inputs and defensive fallbacks.

    The shape is identical to the ambiguous-path output so Phase 2
    consumers can iterate `cr.interpretations` uniformly.
    """
    sql = "-- (no SQL sketch produced; treat question as unambiguous)"
    explanation = (
        "Treated as unambiguous; using the most direct SQL interpretation. "
        f"Reason: {reason}"
    )
    return ClarificationRequest(
        question=question,
        interpretations=[Interpretation(sql=sql, explanation=explanation)],
        ambiguous=False,
    )


class Interpretation(BaseModel):
    """A single interpretation of an ambiguous (or unambiguous) question.

    `sql` is an example SQL sketch against the provided schema. It is
    NOT executed by this module — it exists so the user (or a downstream
    phase) can see what each interpretation would compile to.
    `explanation` is a short natural-language sentence describing the
    assumption that interpretation makes (e.g., "treats revenue as the
    sum of paid-order totals"). `assumptions` is an optional list of
    finer-grained assumptions if the LLM enumerated them.
    """

    sql: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)
    assumptions: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}


class ClarificationRequest(BaseModel):
    """Uniform response shape from `AmbiguityHandler.detect()`.

    Always populated with at least one `Interpretation`. When the input
    question is unambiguous, `ambiguous=False` and `interpretations`
    contains exactly one entry (passthrough). When the question has
    multiple plausible readings, `ambiguous=True` and `interpretations`
    contains ≥2 entries with distinct SQL sketches.

    Downstream consumers (Phase 2's SQL generator, the CLI, the API)
    can branch on `ambiguous` if they want to surface a clarification
    dialog to the user, OR they can simply iterate `interpretations`
    and let the LLM pick the best one — the shape is uniform.
    """

    question: str
    interpretations: list[Interpretation] = Field(..., min_length=1)
    ambiguous: bool
    model_id: Optional[str] = None


class AmbiguityHandler:
    """Detect ambiguity in a natural-language question via the dual-provider LLM abstraction.

    The handler always routes its LLM call through `LLMClient` so the
    same code path works for OpenAI and Anthropic — only `provider`
    and `model` change. Tests inject a `completion_fn` to return canned
    JSON without hitting the network.

    Public API:
        AmbiguityHandler(provider="openai", model="gpt-4o-mini",
                         *, completion_fn=my_mock, client=my_client)
        cr = handler.detect("What's the revenue?", schema)
        cr.ambiguous  # True
        len(cr.interpretations)  # >= 2
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        completion_fn: Optional[Callable[..., Any]] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._completion_fn = completion_fn
        self.client = client or LLMClient(provider=provider, model=model, api_key=api_key)

    def detect(self, question: str, schema: Schema) -> ClarificationRequest:
        """Return a `ClarificationRequest` for `question` against `schema`.

        Behaviour:
          - empty / whitespace-only question → passthrough single interpretation (no LLM call)
          - well-formed LLM JSON response with ≥2 interpretations → ambiguous ClarificationRequest
          - well-formed LLM JSON response with 1 interpretation → unambiguous passthrough
          - malformed / unparseable LLM response → defensive passthrough (ambiguous=False)
        """
        if not question or not question.strip():
            return _passthrough(question or "", "empty question")

        user_prompt = _build_user_prompt(question, schema)
        full_prompt = _SYSTEM_PROMPT + "\n\n" + user_prompt

        try:
            raw = self.client.complete(full_prompt, completion_fn=self._completion_fn)
        except Exception as exc:  # noqa: BLE001 — defensive: any LLM/SDK error → passthrough
            return _passthrough(question, f"LLM call failed: {type(exc).__name__}")

        try:
            cr = _parse_llm_response(raw, question)
        except (json.JSONDecodeError, ValueError, ValidationError):
            return _passthrough(question, "LLM response was not valid JSON")

        cr.model_id = f"{self.provider}/{self.model}"
        return cr


__all__ = ["AmbiguityHandler", "ClarificationRequest", "Interpretation"]
