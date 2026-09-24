"""Structured SQL generation from natural-language questions.

Wraps the dual-provider LLMClient (OpenAI + Anthropic) with provider-specific
structured-output prompts (response_format for OpenAI, tools for Anthropic)
and returns a validated `GeneratedSQL` Pydantic model with five fields:
sql, explanation, confidence, tables_touched, columns_touched.
"""
from text2sql.generation.generator import (
    GeneratedSQL,
    SQLGenerationError,
    SQLGenerator,
)

__all__ = ["GeneratedSQL", "SQLGenerationError", "SQLGenerator"]
