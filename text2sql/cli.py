"""Command-line entry point for the text2sql service.

Phase 1 pipeline:
    database -> extract_schema -> RelevanceFilter -> PromptBuilder -> AmbiguityHandler
                                                                       -> printed prompt + ClarificationRequest

Phase 2 execution pipeline (--execute):
    schema -> filter -> prompt -> SQLGenerator -> GuardrailValidator ->
    Sandbox -> AuditLog -> printed ExecutionResult
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy import create_engine

from text2sql.ambiguity import AmbiguityHandler, ClarificationRequest
from text2sql.config import get_settings
from text2sql.execution import AuditLog, QueryRunner, Sandbox
from text2sql.generation import SQLGenerator
from text2sql.guardrails import GuardrailValidator
from text2sql.llm import default_embedder
from text2sql.prompts import PromptBuilder
from text2sql.relevance import RelevanceFilter
from text2sql.schema import Schema, extract_schema
from text2sql.validation.detector import HallucinationDetector, HallucinationReport


# ---------------------------------------------------------------------------
# Embedder fallback: deterministic bag-of-words so the CLI runs without API keys
# ---------------------------------------------------------------------------

#: Small stopword list — anything ≤2 chars or in this set is dropped from
#: the vocabulary before building binary indicator vectors. Mirrors the
#: pattern used in the relevance-filter test suite.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "from", "into", "this", "are",
    "was", "were", "have", "has", "had", "but", "not", "all", "any",
    "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "many", "much", "most", "more", "less", "few", "show", "me",
    "give", "list", "find", "get", "tell",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords + ≤2-char tokens."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def _fake_embedder(dimension: int = 64) -> Callable[[list[str]], list[list[float]]]:
    """Build a deterministic bag-of-words + L2-normalised embedder.

    Used when the user has no API key — the CLI must still run end-to-end.
    Real production embeddings (text-embedding-3-small, Voyage, etc.)
    produce higher cosine scores for genuinely relevant pairs; the fake
    is intentionally weaker but stable across runs.
    """
    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            tokens = set(_tokenize(text))
            vec = [1.0 if i < len(tokens) else 0.0 for i in range(dimension)]
            # L2 normalise so cosine ≈ dot product.
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vec = [x / norm for x in vec]
            # Overlay token indicator bits (first len(tokens) positions).
            for i, _ in enumerate(tokens):
                if i < dimension:
                    vec[i] = max(vec[i], 1.0 / norm)
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors
    return embed


# ---------------------------------------------------------------------------
# CLI pipeline
# ---------------------------------------------------------------------------


def _resolve_embedder(provider: str, api_key: Optional[str]) -> tuple[Callable[[list[str]], list[list[float]]], str]:
    """Return (embedder, source_label).

    If an API key is available for the chosen provider, use the real
    default_embedder. Otherwise fall back to the deterministic fake so
    the CLI is still usable for local development.
    """
    settings = get_settings()
    has_key = (
        (provider == "openai" and (api_key or settings.openai_api_key))
        or (provider == "anthropic" and (api_key or settings.anthropic_api_key))
    )
    if has_key:
        try:
            return default_embedder(provider, api_key=api_key), f"real:{provider}"
        except NotImplementedError:
            pass
    return _fake_embedder(), "fake"


def run_pipeline(
    question: str,
    *,
    provider: Optional[str] = None,
    database_url: Optional[str] = None,
    embedder: Optional[Callable[[list[str]], list[list[float]]]] = None,
    completion_fn: Optional[Callable[..., Any]] = None,
) -> tuple[str, ClarificationRequest]:
    """Execute the Phase-1 pipeline and return (prompt, ClarificationRequest).

    `embedder` and `completion_fn` are injected so tests can run the
    full pipeline end-to-end without network or API keys.
    """
    settings = get_settings()
    provider = provider or settings.llm_provider
    database_url = database_url or settings.database_url

    engine = create_engine(database_url)
    schema = extract_schema(engine)
    engine.dispose()

    embedder, _source = (embedder, "injected") if embedder is not None else _resolve_embedder(provider, None)
    filtered_schema = RelevanceFilter(embedder=embedder).filter(question, schema)

    prompt = PromptBuilder().build(question, filtered_schema)

    handler = AmbiguityHandler(
        provider=provider,
        model=settings.llm_model,
        completion_fn=completion_fn,
    )
    clarification = handler.detect(question, filtered_schema)
    return prompt, clarification


def run_execute(
    question: str,
    *,
    provider: Optional[str] = None,
    database_url: Optional[str] = None,
    audit_path: Path = Path("audit/queries.jsonl"),
    embedder: Optional[Callable[[list[str]], list[list[float]]]] = None,
    completion_fn: Optional[Callable[..., Any]] = None,
    detect_hallucination: bool = False,
) -> Any:
    """Full Phase 2 pipeline: prompt -> generate -> validate -> sandbox -> audit.

    Returns the `ExecutionResult` from `QueryRunner.run`. The constructed
    prompt is echoed to stderr so stdout remains machine-parseable.
    """
    # Import here to avoid pulling pydantic at module-import time on lighter
    # callers (the Phase 1 CLI path doesn't need ExecutionResult).
    from text2sql.execution.runner import ExecutionResult  # noqa: WPS433

    settings = get_settings()
    provider = provider or settings.llm_provider
    database_url = database_url or settings.database_url

    # 1. Schema
    engine = create_engine(database_url)
    schema = extract_schema(engine)
    engine.dispose()

    # 2. Relevance-filter the schema
    embedder = embedder or _resolve_embedder(provider, None)[0]
    filtered = RelevanceFilter(embedder=embedder).filter(question, schema)

    # 3. Build prompt (so user can see it)
    prompt = PromptBuilder().build(question, filtered)

    # 4. Generate SQL via injected completion_fn (no real API call)
    generator = SQLGenerator(
        provider=provider,
        model=settings.llm_model,
        completion_fn=completion_fn,
    )
    generated = generator.generate(question, filtered)

    # 5. Validate
    validator = GuardrailValidator(
        limit_cap=settings.limit_cap,
        max_subquery_depth=settings.max_subquery_depth,
        max_explain_rows=settings.max_explain_rows,
        statement_timeout_ms=settings.statement_timeout_ms,
        log_blocked=False,  # do not log to stderr in CLI
    )

    # 6. Sandbox (readonly user, always rolls back)
    sandbox = Sandbox(dsn=database_url)

    # 7. Run
    runner = QueryRunner(
        validator=validator,
        sandbox=sandbox,
        audit_path=audit_path,
    )
    result = runner.run(generated, question)

    # 7b. Optional hallucination detection (Phase 3). Runs after execution so
    # the sanity signal can inspect execution_result. Never blocks the
    # execution output even if it raises.
    hallucination_report: Optional[HallucinationReport] = None
    if detect_hallucination:
        try:
            # Build a sandbox-callable wrapper so the detector can re-execute
            # the generated SQL via the same readonly connection.
            def _sandbox_callable(sql: str) -> Any:
                sub_runner = QueryRunner(
                    validator=validator, sandbox=sandbox, audit_path=audit_path
                )
                sub = sub_runner.run(sql, question)
                return sub

            detector = HallucinationDetector(schema=filtered)
            hallucination_report = detector.detect(
                question=question,
                generated_sql=generated.sql,
                execution_result=result,
                schema=filtered,
                embedder=embedder,
                completion_fn=completion_fn,
                sandbox=_sandbox_callable,
            )
        except Exception as exc:  # noqa: BLE001 — never block execution output
            print(
                f"hallucination_detection_error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            hallucination_report = None

    # 8. Echo the prompt to stderr so stdout stays clean for programmatic use
    print("=== PROMPT ===", file=sys.stderr)
    print(prompt, file=sys.stderr)
    print(file=sys.stderr)

    assert isinstance(result, ExecutionResult)  # for type-checkers
    return result, hallucination_report


def _format_clarification(cr: ClarificationRequest) -> str:
    lines: list[str] = []
    lines.append(f"Question:    {cr.question}")
    lines.append(f"Ambiguous:   {cr.ambiguous}")
    if cr.model_id:
        lines.append(f"Model:       {cr.model_id}")
    lines.append(f"Interpretations ({len(cr.interpretations)}):")
    for idx, interp in enumerate(cr.interpretations, start=1):
        lines.append(f"  [{idx}] SQL:         {interp.sql}")
        lines.append(f"      Explanation: {interp.explanation}")
        if interp.assumptions:
            for assumption in interp.assumptions:
                lines.append(f"      Assumption:  {assumption}")
    return "\n".join(lines)


def _format_execution(
    result: Any,
    audit_path: Path | str,
    hallucination_report: Optional[HallucinationReport] = None,
) -> str:
    audit_path = Path(audit_path)
    lines: list[str] = []
    lines.append("=== EXECUTION ===")
    lines.append(f"sql:               {result.sql}")
    lines.append(f"confidence:        {result.confidence:.3f}")
    lines.append(f"row_count:         {result.row_count}")
    lines.append(f"execution_time_ms: {result.execution_time_ms}")
    lines.append(f"audit_log:         {audit_path.resolve()}")
    if not result.allowed:
        lines.append(f"BLOCKED: {result.blocked_reason}")
    elif result.warnings:
        lines.append(f"warnings: {result.warnings}")

    if hallucination_report is not None:
        lines.append("")
        lines.append("=== SCORE ===")
        lines.append(f"overall_score:     {hallucination_report.overall_score:.1f}")
        lines.append(f"verdict:           {hallucination_report.verdict}")
        lines.append(f"flagged:           {str(hallucination_report.flagged).lower()}")
        lines.append("signals:")
        # Always render the 5 known signals in fixed order so downstream
        # consumers can grep reliably.
        for signal_name in (
            "syntax",
            "back_translation",
            "sanity",
            "multi_query",
            "schema_coverage",
        ):
            outcome = hallucination_report.outcomes.get(signal_name)
            if outcome is None:
                lines.append(f"  - {signal_name:<16} score=n/a flagged=n/a")
            else:
                lines.append(
                    f"  - {signal_name:<16} score={outcome.score:.2f} flagged={str(outcome.flagged).lower()}"
                )
        warnings_str = (
            ",".join(hallucination_report.warnings)
            if hallucination_report.warnings
            else "none"
        )
        lines.append(f"warnings:          {warnings_str}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.execute:
        result, hallucination_report = run_execute(
            args.question,
            provider=args.provider,
            database_url=args.database_url,
            audit_path=args.audit_path,
            detect_hallucination=args.detect_hallucination,
        )
        print(_format_execution(result, args.audit_path, hallucination_report))
        return 0

    prompt, clarification = run_pipeline(
        args.question,
        provider=args.provider,
        database_url=args.database_url,
    )

    print("=== PROMPT ===")
    print(prompt)
    print()
    print("=== AMBIGUITY ===")
    print(_format_clarification(clarification))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text2sql",
        description="Translate natural language questions into SQL.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Natural language question to translate.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider override (defaults to LLM_PROVIDER env var).",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Database URL override (defaults to DATABASE_URL env var).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Run the full Phase 2 pipeline: generate -> validate -> sandbox -> audit.",
    )
    parser.add_argument(
        "--audit-path",
        default="audit/queries.jsonl",
        help="Path to the append-only audit log (default: audit/queries.jsonl).",
    )
    parser.add_argument(
        "--detect-hallucination",
        action="store_true",
        default=False,
        help="Append a hallucination SCORE block to the execution output.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
