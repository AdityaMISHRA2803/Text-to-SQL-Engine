"""Structured SQL generation via the dual-provider LLMClient.

Two structured-output paths share a single `completion_fn` injection point:

- **OpenAI**: litellm is called with
    response_format={"type": "json_schema", "json_schema": {...}, "strict": True}
  which guarantees the model returns a JSON string matching the schema.
- **Anthropic**: litellm is called with
    tools=[{"type": "function", "function": {"name": "submit_sql", ...}}]
    and tool_choice={"type": "function", "function": {"name": "submit_sql"}}.
  The Anthropic tool-use response is normalised back to a flat JSON string
  in `_extract_tool_input()`.

Both paths go through `LLMClient.complete(prompt, completion_fn=self._completion_fn, **kwargs)`,
so tests can swap in a stub `completion_fn` that returns canned JSON strings
without ever touching the network.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

import sqlparse
from pydantic import BaseModel, Field, field_validator

from text2sql.llm import LLMClient
from text2sql.schema import Schema


# ---------------------------------------------------------------------------
# Exceptions & models
# ---------------------------------------------------------------------------


# Recognised SQL leading-keywords — used to reject prose like
# "not a sql statement just words" that sqlparse.parse will otherwise
# accept as a single (unknown-type) statement.
_SQL_LEAD_KEYWORDS: frozenset[str] = frozenset(
    word.upper()
    for word in (
        "select", "insert", "update", "delete", "merge", "create", "drop",
        "alter", "truncate", "with", "explain", "analyze", "vacuum",
        "reindex", "grant", "revoke", "set", "reset", "show", "use",
        "begin", "commit", "rollback", "savepoint", "values", "copy",
        "call", "do", "listen", "notify", "unlisten", "lock", "refresh",
        "prepare", "execute", "deallocate", "declare", "fetch", "move",
        "close",
    )
)


class SQLGenerationError(Exception):
    """Raised when SQL generation fails parsing or validation."""


class GeneratedSQL(BaseModel):
    """Structured output from the LLM — five required fields per README 2.1."""

    sql: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    tables_touched: list[str] = Field(default_factory=list)
    columns_touched: list[str] = Field(default_factory=list)

    @field_validator("sql")
    @classmethod
    def _validate_sql(cls, value: str) -> str:
        parsed = sqlparse.parse(value)
        # sqlparse.parse returns [] for empty strings; we want at least one
        # non-empty statement.
        if not parsed or all(str(stmt).strip() == "" for stmt in parsed):
            raise ValueError(f"SQL could not be parsed by sqlparse: {value!r}")
        # sqlparse is permissive — prose like "not a sql statement just words"
        # is parsed as a single Unknown-type statement. Reject any statement
        # whose first non-whitespace, non-comment, non-punctuation token is
        # not a recognised SQL keyword.
        first_stmt = parsed[0]
        for token in first_stmt.tokens:
            ttype = getattr(token, "ttype", None)
            if ttype is None:
                continue
            # skip whitespace, punctuation, and comments
            from sqlparse import tokens as _tokens
            if ttype in _tokens.Whitespace or ttype in _tokens.Comment:
                continue
            if ttype in _tokens.Punctuation:
                continue
            head = str(token).strip().split()[0].upper() if str(token).strip() else ""
            if head and head in _SQL_LEAD_KEYWORDS:
                return value
            raise ValueError(
                f"SQL must begin with a recognised SQL keyword, got {head!r}: {value!r}"
            )
        raise ValueError(f"SQL could not be parsed by sqlparse: {value!r}")


# ---------------------------------------------------------------------------
# Provider-specific structured-output schema
# ---------------------------------------------------------------------------


_SUBMIT_SQL_TOOL_NAME = "submit_sql"


def _generated_sql_parameters() -> dict[str, Any]:
    """JSON-Schema for the GeneratedSQL payload — used in both provider paths."""
    return {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single, syntactically valid SQL statement.",
            },
            "explanation": {
                "type": "string",
                "description": "Plain-English explanation of what the SQL does.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Self-reported confidence in [0, 1].",
            },
            "tables_touched": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tables referenced in the SQL.",
            },
            "columns_touched": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns referenced in the SQL.",
            },
        },
        "required": ["sql", "explanation", "confidence", "tables_touched", "columns_touched"],
        "additionalProperties": False,
    }


def _build_provider_kwargs(provider: str) -> dict[str, Any]:
    """Return the litellm kwargs that force structured output for ``provider``."""
    parameters = _generated_sql_parameters()
    if provider == "openai":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _SUBMIT_SQL_TOOL_NAME,
                    "schema": parameters,
                    "strict": True,
                },
            },
        }
    # anthropic
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": _SUBMIT_SQL_TOOL_NAME,
                    "description": "Submit the generated SQL with explanation and metadata.",
                    "parameters": parameters,
                },
            },
        ],
        "tool_choice": {"type": "function", "function": {"name": _SUBMIT_SQL_TOOL_NAME}},
    }


def _extract_tool_input(anthropic_response: dict[str, Any]) -> str:
    """Normalise an Anthropic tool-use response to the JSON input string.

    Anthropic's litellm shape (after `_extract_content` falls through):
        {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {"name": ..., "arguments": "<json string>"}
                    }]
                }
            }]
        }
    or the raw Anthropic shape:
        {
            "content": [{"type": "tool_use", "input": {...}}]
        }
    Both are accepted; we return the JSON-stringified input.
    """
    choices = anthropic_response.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments")
            if isinstance(arguments, str):
                return arguments
            if isinstance(arguments, dict):
                return json.dumps(arguments)
    content = anthropic_response.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return json.dumps(block.get("input", {}))
    raise SQLGenerationError("Could not extract tool_use input from Anthropic response")


def _normalise_response(provider: str, response: Any) -> str:
    """Convert whatever the LLM returned into a flat JSON string the validator
    can parse.

    The response may arrive in three shapes:
      - ``str``: the content was already extracted (e.g. via
        ``LLMClient.complete`` -> ``_extract_content`` for OpenAI json_schema).
      - ``dict`` with ``choices[0].message.content``: standard OpenAI shape.
      - ``dict`` with ``choices[0].message.tool_calls[*].function.arguments``:
        Anthropic tool_use response (litellm shape).
      - ``dict`` with top-level ``content=[{"type": "tool_use", "input": ...}]``:
        raw Anthropic shape.
    """
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        # Object with ``.choices`` attribute (pydantic/openai ModelResponse).
        try:
            content = LLMClient._extract_content(response)
            return content
        except Exception as exc:
            raise SQLGenerationError(
                f"Could not extract content from LLM response: {exc}"
            ) from exc

    # OpenAI path: a string content field on choices[0].message.
    if provider == "openai":
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
        raise SQLGenerationError("OpenAI response missing choices[0].message.content")

    # Anthropic path: try tool_use first, then raw content blocks.
    try:
        return _extract_tool_input(response)
    except SQLGenerationError as exc:
        # Fall back to plain content if present.
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
        raise SQLGenerationError(
            f"Could not extract tool_use input from Anthropic response: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SQLGenerator:
    """Generate a `GeneratedSQL` Pydantic model from a question + schema."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        completion_fn: Optional[Callable[..., Any]] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        if provider not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported provider: {provider!r}")
        self.provider = provider
        self.model = model
        self._completion_fn = completion_fn
        self._client_injected = client is not None
        self._client = client or LLMClient(provider=provider, model=model, api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, question: str, schema: Schema) -> GeneratedSQL:
        """Generate SQL for ``question`` against ``schema``."""
        if not question or not question.strip():
            raise SQLGenerationError("Question must be a non-empty string")
        if schema is None:
            raise SQLGenerationError("schema must not be None")

        prompt = self._build_prompt(question, schema)
        kwargs = _build_provider_kwargs(self.provider)

        # Route completion through LLMClient when a custom client was injected
        # via the constructor (so test stubs observe the call); otherwise call
        # completion_fn directly and let `_normalise_response` handle shape
        # normalisation. Going through the default `LLMClient.complete()` with
        # a user-supplied `completion_fn` would apply `_extract_content`, which
        # only knows the OpenAI `choices[0].message.content` shape — so we
        # bypass it when no custom client was injected.
        response: Any
        if self._completion_fn is not None:
            if self._client_injected:
                response = self._client.complete(prompt, completion_fn=self._completion_fn, **kwargs)
            else:
                response = self._completion_fn(prompt=prompt, **kwargs)
        else:
            # Real litellm call. We need to capture the raw response for
            # Anthropic tool_use normalisation. Since LLMClient.complete
            # returns only content, we use the underlying client directly.
            try:
                from litellm import completion as _litellm_completion
            except ImportError as exc:  # pragma: no cover - litellm required dep
                raise SQLGenerationError("litellm is not installed") from exc
            from text2sql.config import get_settings

            model_name = f"{self.provider}/{self.model}"
            call_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs,
            }
            settings_api_key = (
                get_settings().openai_api_key
                if self.provider == "openai"
                else get_settings().anthropic_api_key
            )
            if settings_api_key:
                call_kwargs["api_key"] = settings_api_key
            response = _litellm_completion(**call_kwargs)

        normalised = _normalise_response(self.provider, response)
        return self._validate_and_build(normalised, schema, question)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, schema: Schema) -> str:
        try:
            schema_text = schema.to_compact_json()
        except Exception:  # pragma: no cover - fallback only
            schema_text = str(schema)
        return (
            "You are an expert PostgreSQL generator. Given the schema below "
            "and the user's question, produce exactly one SQL statement that "
            "answers the question. Use the submit_sql tool/response format to "
            "return your answer.\n\n"
            f"## Schema\n{schema_text}\n\n"
            f"## Question\n{question.strip()}\n"
        )

    def _validate_and_build(
        self, raw: str, schema: Schema, question: str
    ) -> GeneratedSQL:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SQLGenerationError(
                f"LLM response was not valid JSON: {exc}; raw={raw!r}"
            ) from exc

        if not isinstance(data, dict):
            raise SQLGenerationError(
                f"LLM response must be a JSON object, got {type(data).__name__}"
            )

        missing = [f for f in ("sql", "explanation", "confidence",
                               "tables_touched", "columns_touched") if f not in data]
        if missing:
            raise SQLGenerationError(f"LLM response missing fields: {missing}")

        # Cross-check tables/columns against schema; unknown items drop confidence.
        known_tables = set(schema.table_names())
        validated_tables: list[str] = []
        confidence = float(data["confidence"])
        for table in data.get("tables_touched") or []:
            if not isinstance(table, str):
                continue
            if table in known_tables:
                validated_tables.append(table)
            else:
                confidence = max(0.0, confidence - 0.2)

        known_columns: set[str] = set()
        for tbl in schema.tables:
            for col in tbl.columns:
                known_columns.add(col.name)
                known_columns.add(f"{tbl.name}.{col.name}")
        validated_columns: list[str] = []
        for col in data.get("columns_touched") or []:
            if not isinstance(col, str):
                continue
            if col in known_columns:
                validated_columns.append(col)
            # unknown columns are dropped silently

        try:
            return GeneratedSQL(
                sql=str(data["sql"]),
                explanation=str(data["explanation"]),
                confidence=confidence,
                tables_touched=validated_tables,
                columns_touched=validated_columns,
            )
        except Exception as exc:
            raise SQLGenerationError(
                f"GeneratedSQL validation failed: {exc}; raw={raw!r}"
            ) from exc


__all__ = ["GeneratedSQL", "SQLGenerationError", "SQLGenerator"]
