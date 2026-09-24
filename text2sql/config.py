"""Application configuration backed by pydantic-settings.

Reads from environment variables (and optionally a `.env` file).

Guardrail thresholds (limit_cap, max_subquery_depth, max_explain_rows,
statement_timeout_ms) have defaults that match the README 2.2 spec and are
overridable via env vars (e.g. `LIMIT_CAP=500`, `MAX_SUBQUERY_DEPTH=5`).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://text2sql:text2sql@localhost:5432/text2sql"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"

    # --- Guardrail thresholds (README 2.2) -----------------------------------
    # Cap appended to queries without an explicit LIMIT.
    limit_cap: int = 1000
    # Maximum nesting depth of parenthesised subqueries.
    max_subquery_depth: int = 3
    # EXPLAIN Plan.Total Cost-style estimate threshold; reject above this.
    max_explain_rows: int = 100_000
    # PostgreSQL statement_timeout (ms) applied per execution by the runner.
    statement_timeout_ms: int = 30_000

    # --- Hallucination detection thresholds (README 2.3) --------------------
    # Thresholds used by the hallucination detection system to decide whether
    # a given signal warrants flagging a candidate query. Defaults match
    # README 2.3.1 and are env-overridable (e.g. `BACK_TRANSLATION_THRESHOLD`).
    back_translation_threshold: float = Field(
        0.75,
        description=(
            "Cosine similarity threshold below which a back-translation "
            "comparison is flagged as suspicious (README 2.3.1 default)."
        ),
    )
    null_heavy_ratio: float = Field(
        0.5,
        description=(
            "Fraction (0-1) of rows that must be NULL in a column for that "
            "column to be flagged as NULL-heavy, which usually indicates a "
            "bad JOIN producing NULLs on the unmatched side."
        ),
    )
    multi_query_min_complexity: Literal["any", "join_or_agg", "always"] = Field(
        "join_or_agg",
        description=(
            "Minimum query complexity required to trigger multi-query "
            "validation. 'any' / 'always' runs on every query; 'join_or_agg' "
            "restricts it to queries that contain a JOIN or GROUP BY clause."
        ),
    )

    # --- Hallucination detection signal weights -----------------------------
    # Weights for combining individual hallucination signals into a final
    # score. Defaults below are normalised and sum to exactly 1.0; they will
    # be re-normalised at scorer-time to defend against env-var drift.
    weight_syntax: float = Field(
        0.20,
        description="Weight for the SQL syntax validity signal (binary).",
    )
    weight_back_translation: float = Field(
        0.25,
        description="Weight for the back-translation alignment signal.",
    )
    weight_sanity: float = Field(
        0.20,
        description="Weight for the sanity check pass-rate signal.",
    )
    weight_multi_query: float = Field(
        0.20,
        description="Weight for the multi-query agreement signal.",
    )
    weight_schema_coverage: float = Field(
        0.15,
        description="Weight for the schema coverage signal.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache makes Settings effectively a singleton across the
    process, which is what we want for API key and DB URL handling.
    """
    return Settings()
