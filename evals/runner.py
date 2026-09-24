"""Eval runner — CLI entry point for `python -m evals.run`.

Loads golden cases from ``evals/golden/*.json``, runs each through the
text2sql pipeline (stub-mode or live-mode), scores the outputs against the
golden expectations, and writes a JSON + Markdown report.

Modes
-----

* ``stub`` (default): builds ``GeneratedSQL`` directly from the golden SQL,
  then exercises the rest of the pipeline (guardrail, sandbox, hallucination
  detection) with stub collaborators. **No LLM calls, no API keys required.**
  Measures downstream-pipeline accuracy only — SQL exact-match will be
  trivially 100% by construction.

* ``live``: invokes the real pipeline via ``cli.run_execute`` with a
  ``completion_fn`` that calls litellm. Requires an API key. Measures true
  end-to-end accuracy including LLM generation quality.

Exit code
---------

* ``0`` — every golden case ran to completion (regardless of pass/fail).
* ``1`` — fatal error (golden dataset missing, DB unreachable in a way the
  runner cannot recover from, etc.).

The pass/fail of individual cases is recorded in the report, not in the
process exit code, so CI can ingest the JSON without needing the runner to
agree with it on what "passing" means.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from text2sql.config import get_settings
from text2sql.execution import AuditLog, QueryRunner, Sandbox
from text2sql.generation import GeneratedSQL
from text2sql.guardrails import GuardrailValidator
from text2sql.validation.detector import HallucinationDetector

from .completion_fns import make_live_completion_fn
from .metrics import Counters, execution_match, sql_exact_match

logger = logging.getLogger("evals.runner")


# ---------------------------------------------------------------------------
# Public dataclasses — the shape the report writer consumes
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    difficulty: str
    notes: str
    golden_sql: Optional[str]
    generated_sql: Optional[str]
    sql_exact_match: Optional[bool]      # None when no golden SQL (ambiguous/unanswerable)
    execution_match: Optional[bool]      # None when DB unavailable or no golden SQL
    execution_skipped: bool
    execution_skip_reason: Optional[str]
    guardrail_allowed: Optional[bool]
    guardrail_blocked_reason: Optional[str]
    hallucination_flagged: Optional[bool]
    hallucination_should_flag: bool
    hallucination_verdict: Optional[str]
    hallucination_score: Optional[float]
    detector_notes: list[str] = field(default_factory=list)
    detector_warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EvalReport:
    timestamp: str
    mode: str
    provider: Optional[str]
    model: Optional[str]
    database_url: Optional[str]
    db_available: bool
    total_cases: int
    summary: dict[str, Any]
    per_category: dict[str, dict[str, Any]]
    cases: list[CaseResult]


# ---------------------------------------------------------------------------
# Golden loading
# ---------------------------------------------------------------------------


def load_golden_dataset(golden_dir: Path) -> list[dict[str, Any]]:
    """Load and validate every ``*.json`` file in ``golden_dir``."""
    if not golden_dir.is_dir():
        raise FileNotFoundError(f"Golden dataset directory not found: {golden_dir}")
    cases: list[dict[str, Any]] = []
    for fp in sorted(golden_dir.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{fp} must be a JSON list of cases, got {type(data).__name__}")
        for case in data:
            _validate_case(case, source=str(fp))
            cases.append(case)
    if len(cases) < 50:
        logger.warning(
            "Golden dataset has %d cases; README target is >=50. Add more in evals/golden/.",
            len(cases),
        )
    return cases


def _validate_case(case: dict[str, Any], source: str) -> None:
    required = {"id", "category", "question", "should_be_flagged", "difficulty"}
    missing = required - case.keys()
    if missing:
        raise ValueError(f"{source}: case missing required fields: {sorted(missing)}")
    if case.get("golden_sql") is not None and "expect" not in case:
        raise ValueError(
            f"{source}: case {case.get('id')!r} has golden_sql but no expect block"
        )


# ---------------------------------------------------------------------------
# DB availability
# ---------------------------------------------------------------------------


def _probe_database(database_url: Optional[str]) -> tuple[Optional[Engine], Optional[str]]:
    """Try to connect. Returns ``(engine, readonly_dsn)`` or ``(None, None)``."""
    settings = get_settings()
    url = database_url or settings.database_url
    try:
        admin_engine = create_engine(url, future=True)
        with admin_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database unreachable (%s); execution_match will be skipped.", exc)
        return None, None
    # Derive readonly DSN by swapping user/password — same pattern as conftest.
    from sqlalchemy.engine.url import make_url

    try:
        u = make_url(url)
        readonly_dsn = str(u.set(username="text2sql_ro", password="text2sql_ro"))
    except Exception:
        readonly_dsn = url
    return admin_engine, readonly_dsn


def _reseed(admin_engine: Engine) -> None:
    """Drop + recreate + reseed the database so result snapshots are stable."""
    from text2sql.seed_dataset import seed
    seed(admin_engine, row_count=200)


# ---------------------------------------------------------------------------
# Stub-mode pipeline
# ---------------------------------------------------------------------------


def _build_stub_completion_fn(original_question: str) -> Callable[..., Any]:
    """Build a stub completion_fn that returns ``original_question``.

    Used in stub mode so that golden-SQL cases get high back-translation
    similarity scores and therefore are *not* flagged as hallucinations.
    """
    payload = json.dumps({
        "question": original_question,
        "translation": original_question,
    })

    def _fn(prompt: str = "", **_kwargs: Any) -> str:
        return payload

    return _fn


def _make_fake_embedder() -> Callable[[list[str]], list[list[float]]]:
    """Deterministic bag-of-words embedder (no API key needed)."""
    try:
        from text2sql.cli import _fake_embedder
        return _fake_embedder()
    except Exception:  # pragma: no cover - fallback
        # Fallback in case _fake_embedder is unavailable in some env.
        import hashlib
        dim = 64

        def _hash(text: str) -> list[float]:
            vec = [0.0] * dim
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % dim] += 1.0
            return vec

        def embed(texts: list[str]) -> list[list[float]]:
            return [_hash(t) for t in texts]

        return embed


def _run_case_stub(
    case: dict[str, Any],
    *,
    admin_engine: Optional[Engine],
    readonly_dsn: Optional[str],
    counters: Counters,
) -> CaseResult:
    """Run a single golden case through the stub-mode pipeline."""
    cid = case["id"]
    category = case["category"]
    question = case["question"]
    golden_sql = case.get("golden_sql")
    should_flag = case["should_be_flagged"]

    result = CaseResult(
        case_id=cid,
        category=category,
        question=question,
        difficulty=case.get("difficulty", ""),
        notes=case.get("notes", ""),
        golden_sql=golden_sql,
        generated_sql=None,
        sql_exact_match=None,
        execution_match=None,
        execution_skipped=True,
        execution_skip_reason=None,
        guardrail_allowed=None,
        guardrail_blocked_reason=None,
        hallucination_flagged=None,
        hallucination_should_flag=should_flag,
        hallucination_verdict=None,
        hallucination_score=None,
    )

    # --- 1. SQL exact match ----------------------------------------------------
    if golden_sql is None:
        result.sql_exact_match = None  # not applicable
    else:
        match = sql_exact_match(golden_sql, golden_sql)
        result.sql_exact_match = match
        result.generated_sql = golden_sql
        if match:
            counters.bump("sql_exact_match_hits")

    # --- 2. Guardrail ---------------------------------------------------------
    validator = GuardrailValidator(explain_runner=lambda _sql: 0, log_blocked=False)
    if golden_sql is not None:
        try:
            vres = validator.validate(golden_sql)
        except Exception as exc:  # noqa: BLE001
            result.error = f"guardrail raised: {exc}"
            vres = None
        if vres is not None:
            result.guardrail_allowed = vres.allowed
            result.guardrail_blocked_reason = vres.blocked_reason
            counters.bump("guardrail_checked")
            if category == "dangerous":
                if not vres.allowed:
                    counters.bump("guardrail_blocked_when_expected")
                else:
                    counters.bump("guardrail_allowed_when_unsafe")
            else:
                if vres.allowed:
                    counters.bump("guardrail_allowed_when_expected")
                else:
                    counters.bump("guardrail_blocked_when_safe")

    # --- 3. Execution (skip if dangerous or no DB) -----------------------------
    if golden_sql is not None and result.guardrail_allowed and admin_engine is not None:
        execution_result = _execute_in_sandbox(golden_sql, readonly_dsn)
        result.execution_skipped = False
        counters.bump("execution_attempted")
        if execution_result is None:
            result.execution_skip_reason = "sandbox execution failed"
        else:
            # Fetch golden results directly from the admin engine.
            try:
                with admin_engine.connect() as conn:
                    rows = conn.execute(text(golden_sql)).fetchall()
                    golden_results = [dict(r._mapping) for r in rows]
            except Exception as exc:  # noqa: BLE001
                result.execution_skip_reason = f"golden fetch failed: {exc}"
            else:
                em = execution_match(execution_result, golden_results)
                result.execution_match = em
                if em:
                    counters.bump("execution_match_hits")
    elif golden_sql is not None and result.guardrail_allowed and admin_engine is None:
        result.execution_skip_reason = "database unavailable"

    # --- 4. Hallucination detection -------------------------------------------
    # Skip when no SQL was generated (ambiguous / unanswerable stubs).
    if golden_sql is None:
        # Stub mode cannot exercise LLM-driven ambiguity handling.
        result.hallucination_flagged = None
        result.detector_notes.append("stub-mode: no SQL to score")
    else:
        # Build a synthetic execution result for the detector. The detector
        # only consumes .results + .row_count.
        from text2sql.execution import ExecutionResult

        exec_for_detector = ExecutionResult(
            sql=golden_sql,
            confidence=0.9,
            results=[],
            row_count=0,
            execution_time_ms=0,
            warnings=[],
            allowed=True,
        )
        detector = HallucinationDetector(
            schema=None,  # schema_coverage signal will be skipped (neutral 0.5)
            multi_query_validator=None,  # multi_query signal will be skipped
        )
        try:
            report = detector.detect(
                question=question,
                generated_sql=golden_sql,
                execution_result=exec_for_detector,
                completion_fn=_build_stub_completion_fn(question),
                embedder=_make_fake_embedder(),
            )
        except Exception as exc:  # noqa: BLE001
            result.error = (result.error or "") + f"; detector raised: {exc}"
            report = None
        if report is not None:
            result.hallucination_flagged = bool(report.flagged)
            result.hallucination_verdict = report.verdict
            result.hallucination_score = report.overall_score
            result.detector_notes = list(report.notes or [])
            result.detector_warnings = list(report.warnings or [])
            counters.bump("hallucination_checked")
            if result.hallucination_flagged == should_flag:
                counters.bump("hallucination_correct")
            else:
                counters.bump("hallucination_wrong")

    return result


def _execute_in_sandbox(sql: str, readonly_dsn: Optional[str]) -> Optional[list[dict[str, Any]]]:
    """Run SQL inside the readonly sandbox and return rows."""
    if not readonly_dsn:
        return None
    try:
        sandbox = Sandbox(dsn=readonly_dsn)
        with sandbox.connection() as conn:
            rows = conn.execute(text(sql)).fetchall()
            return [dict(r._mapping) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sandbox execution failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Live-mode pipeline (delegates to cli.run_execute)
# ---------------------------------------------------------------------------


def _run_case_live(
    case: dict[str, Any],
    *,
    provider: str,
    model: str,
    api_key: Optional[str],
    database_url: Optional[str],
    admin_engine: Optional[Engine],
    readonly_dsn: Optional[str],
    counters: Counters,
) -> CaseResult:
    """Run a single golden case through the live LLM pipeline."""
    from text2sql.cli import run_execute

    cid = case["id"]
    category = case["category"]
    question = case["question"]
    golden_sql = case.get("golden_sql")
    should_flag = case["should_be_flagged"]

    result = CaseResult(
        case_id=cid,
        category=category,
        question=question,
        difficulty=case.get("difficulty", ""),
        notes=case.get("notes", ""),
        golden_sql=golden_sql,
        generated_sql=None,
        sql_exact_match=None,
        execution_match=None,
        execution_skipped=True,
        execution_skip_reason=None,
        guardrail_allowed=None,
        guardrail_blocked_reason=None,
        hallucination_flagged=None,
        hallucination_should_flag=should_flag,
        hallucination_verdict=None,
        hallucination_score=None,
    )

    try:
        completion_fn = make_live_completion_fn(provider, model, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        result.error = f"live completion_fn setup failed: {exc}"
        return result

    try:
        exec_result, hallucination_report = run_execute(
            question,
            provider=provider,
            database_url=database_url,
            completion_fn=completion_fn,
            embedder=_make_fake_embedder(),
            detect_hallucination=True,
        )
    except Exception as exc:  # noqa: BLE001
        result.error = f"run_execute raised: {exc}"
        return result

    result.generated_sql = exec_result.sql
    result.guardrail_allowed = exec_result.allowed
    result.guardrail_blocked_reason = exec_result.blocked_reason

    if golden_sql is not None:
        match = sql_exact_match(exec_result.sql, golden_sql)
        result.sql_exact_match = match
        if match:
            counters.bump("sql_exact_match_hits")

    counters.bump("guardrail_checked")
    if category == "dangerous":
        if not exec_result.allowed:
            counters.bump("guardrail_blocked_when_expected")
        else:
            counters.bump("guardrail_allowed_when_unsafe")
    else:
        if exec_result.allowed:
            counters.bump("guardrail_allowed_when_expected")
        else:
            counters.bump("guardrail_blocked_when_safe")

    # Execution match: only when SQL was allowed and DB is available.
    if exec_result.allowed and admin_engine is not None and golden_sql is not None:
        result.execution_skipped = False
        counters.bump("execution_attempted")
        try:
            with admin_engine.connect() as conn:
                rows = conn.execute(text(golden_sql)).fetchall()
                golden_results = [dict(r._mapping) for r in rows]
            em = execution_match(exec_result.results, golden_results)
            result.execution_match = em
            if em:
                counters.bump("execution_match_hits")
        except Exception as exc:  # noqa: BLE001
            result.execution_skip_reason = f"golden fetch failed: {exc}"
    elif exec_result.allowed and admin_engine is None:
        result.execution_skip_reason = "database unavailable"

    # Hallucination detection.
    if hallucination_report is not None:
        result.hallucination_flagged = bool(hallucination_report.flagged)
        result.hallucination_verdict = hallucination_report.verdict
        result.hallucination_score = hallucination_report.overall_score
        result.detector_notes = list(hallucination_report.notes or [])
        result.detector_warnings = list(hallucination_report.warnings or [])
        counters.bump("hallucination_checked")
        if result.hallucination_flagged == should_flag:
            counters.bump("hallucination_correct")
        else:
            counters.bump("hallucination_wrong")

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _build_summary(counters: Counters, results: list[CaseResult]) -> dict[str, Any]:
    """Roll up counters into a final summary dict."""
    sql_em_total = sum(1 for r in results if r.sql_exact_match is not None)
    sql_em_hits = sum(1 for r in results if r.sql_exact_match)
    sql_em_rate = (sql_em_hits / sql_em_total) if sql_em_total else 0.0

    exec_total = counters.get("execution_attempted")
    exec_hits = counters.get("execution_match_hits")
    exec_rate = (exec_hits / exec_total) if exec_total else 0.0

    gr_total = counters.get("guardrail_checked")
    gr_correct = (
        counters.get("guardrail_blocked_when_expected")
        + counters.get("guardrail_allowed_when_expected")
    )
    gr_rate = (gr_correct / gr_total) if gr_total else 0.0

    hal_total = counters.get("hallucination_checked")
    hal_correct = counters.get("hallucination_correct")
    hal_rate = (hal_correct / hal_total) if hal_total else 0.0

    return {
        "sql_exact_match": {
            "applicable": sql_em_total,
            "hits": sql_em_hits,
            "rate": round(sql_em_rate, 4),
        },
        "execution_match": {
            "attempted": exec_total,
            "hits": exec_hits,
            "rate": round(exec_rate, 4),
        },
        "guardrail_effectiveness": {
            "checked": gr_total,
            "correct": gr_correct,
            "rate": round(gr_rate, 4),
            "blocked_when_unsafe": counters.get("guardrail_blocked_when_expected"),
            "allowed_when_safe": counters.get("guardrail_allowed_when_expected"),
            "false_negatives": counters.get("guardrail_allowed_when_unsafe"),
            "false_positives": counters.get("guardrail_blocked_when_safe"),
        },
        "hallucination_detection": {
            "checked": hal_total,
            "correct": hal_correct,
            "rate": round(hal_rate, 4),
        },
        "totals": {
            "cases": len(results),
            "executed": sum(1 for r in results if not r.execution_skipped),
            "errors": sum(1 for r in results if r.error),
        },
    }


def _build_per_category(results: list[CaseResult]) -> dict[str, dict[str, Any]]:
    """Per-category roll-up so the report can show a category-by-category table."""
    per_cat: dict[str, dict[str, Any]] = {}
    for r in results:
        c = per_cat.setdefault(
            r.category,
            {
                "total": 0,
                "sql_em_applicable": 0,
                "sql_em_hits": 0,
                "exec_attempted": 0,
                "exec_hits": 0,
                "guardrail_correct": 0,
                "guardrail_total": 0,
                "hal_correct": 0,
                "hal_total": 0,
            },
        )
        c["total"] += 1
        if r.sql_exact_match is not None:
            c["sql_em_applicable"] += 1
            if r.sql_exact_match:
                c["sql_em_hits"] += 1
        if not r.execution_skipped and r.execution_match is not None:
            c["exec_attempted"] += 1
            if r.execution_match:
                c["exec_hits"] += 1
        if r.guardrail_allowed is not None:
            c["guardrail_total"] += 1
            if r.category == "dangerous" and not r.guardrail_allowed:
                c["guardrail_correct"] += 1
            elif r.category != "dangerous" and r.guardrail_allowed:
                c["guardrail_correct"] += 1
        if r.hallucination_flagged is not None:
            c["hal_total"] += 1
            if r.hallucination_flagged == r.hallucination_should_flag:
                c["hal_correct"] += 1
    # Round rates.
    for c in per_cat.values():
        c["sql_em_rate"] = round(
            (c["sql_em_hits"] / c["sql_em_applicable"]) if c["sql_em_applicable"] else 0.0, 4
        )
        c["exec_match_rate"] = round(
            (c["exec_hits"] / c["exec_attempted"]) if c["exec_attempted"] else 0.0, 4
        )
        c["guardrail_rate"] = round(
            (c["guardrail_correct"] / c["guardrail_total"]) if c["guardrail_total"] else 0.0, 4
        )
        c["hal_rate"] = round(
            (c["hal_correct"] / c["hal_total"]) if c["hal_total"] else 0.0, 4
        )
    return per_cat


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_eval(
    *,
    mode: str = "stub",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    database_url: Optional[str] = None,
    reseed: bool = False,
    golden_dir: Optional[Path] = None,
) -> EvalReport:
    """Run the full eval suite and return the assembled report."""
    if mode not in {"stub", "live"}:
        raise ValueError(f"Invalid mode: {mode!r}; expected 'stub' or 'live'")
    golden_path = golden_dir or (Path(__file__).parent / "golden")
    cases = load_golden_dataset(golden_path)

    admin_engine, readonly_dsn = _probe_database(database_url)
    db_available = admin_engine is not None

    if reseed and admin_engine is not None:
        logger.info("Reseeding database with row_count=200")
        _reseed(admin_engine)

    counters = Counters()
    results: list[CaseResult] = []
    started = time.monotonic()

    for case in cases:
        try:
            if mode == "stub":
                r = _run_case_stub(
                    case,
                    admin_engine=admin_engine,
                    readonly_dsn=readonly_dsn,
                    counters=counters,
                )
            else:
                if not provider or not model:
                    raise RuntimeError("live mode requires --provider and --model")
                r = _run_case_live(
                    case,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    database_url=database_url,
                    admin_engine=admin_engine,
                    readonly_dsn=readonly_dsn,
                    counters=counters,
                )
        except Exception as exc:  # noqa: BLE001
            r = CaseResult(
                case_id=case["id"],
                category=case["category"],
                question=case["question"],
                difficulty=case.get("difficulty", ""),
                notes=case.get("notes", ""),
                golden_sql=case.get("golden_sql"),
                generated_sql=None,
                sql_exact_match=None,
                execution_match=None,
                execution_skipped=True,
                execution_skip_reason=None,
                guardrail_allowed=None,
                guardrail_blocked_reason=None,
                hallucination_flagged=None,
                hallucination_should_flag=case["should_be_flagged"],
                hallucination_verdict=None,
                hallucination_score=None,
                error=f"runner exception: {exc}",
            )
        results.append(r)

    elapsed = time.monotonic() - started
    summary = _build_summary(counters, results)
    summary["elapsed_seconds"] = round(elapsed, 2)
    per_category = _build_per_category(results)

    return EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        provider=provider,
        model=model,
        database_url=database_url or get_settings().database_url,
        db_available=db_available,
        total_cases=len(results),
        summary=summary,
        per_category=per_category,
        cases=results,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evals.run",
        description="Run the text2sql eval suite (Phase 5).",
    )
    p.add_argument("--mode", choices=["stub", "live"], default="stub")
    p.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    p.add_argument("--model", default=None, help="LLM model name (without provider prefix)")
    p.add_argument("--api-key", default=None, help="Override API key (else reads from env/Settings)")
    p.add_argument("--database-url", default=None)
    p.add_argument("--reseed", action="store_true", help="Drop+reseed the DB before running")
    p.add_argument("--golden-dir", default=None, help="Override path to golden dataset")
    p.add_argument(
        "--report-dir",
        default="evals/reports",
        help="Where to write timestamped JSON + Markdown reports",
    )
    p.add_argument("--json-out", default=None, help="Override the JSON output path")
    p.add_argument("--md-out", default=None, help="Override the Markdown output path")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        report = run_eval(
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            database_url=args.database_url,
            reseed=args.reseed,
            golden_dir=Path(args.golden_dir) if args.golden_dir else None,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # Lazy import of report module to avoid circular import at module load.
    from .report import write_json_report, write_markdown_report

    ts = report.timestamp.replace(":", "-").replace("+00-00", "Z").rstrip("Z") + "Z"
    json_path = Path(args.json_out) if args.json_out else Path(args.report_dir) / f"phase5_{ts}.json"
    md_path = Path(args.md_out) if args.md_out else Path(args.report_dir) / f"phase5_{ts}.md"
    write_json_report(report, json_path)
    write_markdown_report(report, md_path)

    logger.info("Report written to %s and %s", json_path, md_path)
    print(f"Eval complete: {report.total_cases} cases, "
          f"{report.summary['sql_exact_match']['rate']*100:.1f}% SQL EM, "
          f"{report.summary['execution_match']['rate']*100:.1f}% exec match, "
          f"{report.summary['guardrail_effectiveness']['rate']*100:.1f}% guardrail, "
          f"{report.summary['hallucination_detection']['rate']*100:.1f}% hallucination.")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
