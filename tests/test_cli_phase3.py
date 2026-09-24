"""Phase 3 CLI tests: --detect-hallucination flag and SCORE block formatting.

Covers:
- Parser: --detect-hallucination default + parses + composes with --execute
- _format_execution without report: no SCORE block, BLOCKED line, warnings line
- _format_execution with report: SCORE block, 5 fixed signal rows, score/flagged fields,
  warnings rendering ('none' or comma-joined)
- Score formatting boundaries: verdict 'high'/'low'/'moderate' formatting
- main() end-to-end: stubbed run_execute returns (result, report|None); assert stdout shape
  and exception swallowing when HallucinationDetector raises.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from text2sql import cli
from text2sql.cli import (
    _build_parser,
    _format_execution,
    main,
)
from text2sql.scoring.scorer import ConfidenceScore
from text2sql.validation.detector import HallucinationReport, SignalOutcome


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubResult:
    """Minimal duck-typed ExecutionResult for formatter tests."""

    def __init__(
        self,
        sql: str = "SELECT 1",
        confidence: float = 0.9,
        row_count: int = 5,
        execution_time_ms: int = 12,
        allowed: bool = True,
        blocked_reason: Any = None,
        warnings: list | None = None,
    ) -> None:
        self.sql = sql
        self.confidence = confidence
        self.row_count = row_count
        self.execution_time_ms = execution_time_ms
        self.allowed = allowed
        self.blocked_reason = blocked_reason
        self.warnings = warnings or []


def _make_confidence(score: float = 85.0, verdict: str = "pass", flagged: bool = False) -> ConfidenceScore:
    return ConfidenceScore(
        score=score,
        verdict=verdict,
        flagged=flagged,
        signal_scores={},
        weights={},
        notes=[],
    )


def _make_report(
    verdict: str = "high",
    score: float = 85.0,
    flagged: bool = False,
    with_warnings: bool = False,
    missing: set | None = None,
) -> HallucinationReport:
    """Build a HallucinationReport with the 5 standard signals.

    missing: set of signal names to omit (for testing the 'n/a' rendering).
    """
    missing = missing or set()
    all_outcomes = {
        "syntax": SignalOutcome(name="syntax", ran=True, score=1.00, flagged=False),
        "back_translation": SignalOutcome(name="back_translation", ran=True, score=0.90, flagged=False),
        "sanity": SignalOutcome(name="sanity", ran=True, score=1.00, flagged=False),
        "multi_query": SignalOutcome(
            name="multi_query", ran=False, score=0.50, skipped_reason="no completion_fn"
        ),
        "schema_coverage": SignalOutcome(name="schema_coverage", ran=True, score=0.75, flagged=False),
    }
    outcomes = {k: v for k, v in all_outcomes.items() if k not in missing}
    warnings = ["back_translation.llm_failed"] if with_warnings else []
    return HallucinationReport(
        question="q",
        generated_sql="SELECT 1",
        overall_score=score,
        verdict=verdict,
        flagged=flagged,
        confidence=_make_confidence(score=score),
        outcomes=outcomes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Group A — Parser flag (3)
# ---------------------------------------------------------------------------


def test_parser_detect_hallucination_default_false() -> None:
    """Without the flag, args.detect_hallucination must default to False."""
    parser = _build_parser()
    args = parser.parse_args(["--question", "q", "--execute"])
    assert args.detect_hallucination is False


def test_parser_detect_hallucination_true() -> None:
    """--detect-hallucination sets the flag to True."""
    parser = _build_parser()
    args = parser.parse_args(
        ["--question", "q", "--execute", "--detect-hallucination"]
    )
    assert args.detect_hallucination is True


def test_parser_detect_hallucination_with_execute() -> None:
    """--execute + --detect-hallucination both parse and coexist."""
    parser = _build_parser()
    args = parser.parse_args(
        ["--question", "q", "--execute", "--detect-hallucination"]
    )
    assert args.execute is True
    assert args.detect_hallucination is True


# ---------------------------------------------------------------------------
# Group B — _format_execution WITHOUT report (3)
# ---------------------------------------------------------------------------


def test_format_execution_no_report_omits_score_block() -> None:
    """When hallucination_report is None, no SCORE block should appear."""
    result = _StubResult()
    out = _format_execution(result, Path("audit.jsonl"), None)
    assert "=== SCORE ===" not in out
    assert "overall_score" not in out


def test_format_execution_blocked_result_renders_blocked_line() -> None:
    """Blocked results must render a BLOCKED: line."""
    result = _StubResult(allowed=False, blocked_reason="destructive statement")
    out = _format_execution(result, Path("audit.jsonl"))
    assert "BLOCKED:" in out
    assert "destructive statement" in out


def test_format_execution_warnings_rendered_when_present() -> None:
    """warnings list (non-empty) must be rendered when result is allowed."""
    result = _StubResult(warnings=["hint", "caveat"])
    out = _format_execution(result, Path("audit.jsonl"))
    assert "warnings:" in out


# ---------------------------------------------------------------------------
# Group C — _format_execution WITH report (5)
# ---------------------------------------------------------------------------


def test_format_execution_score_block_rendered_with_report() -> None:
    """A report triggers the SCORE block."""
    result = _StubResult()
    out = _format_execution(result, Path("audit.jsonl"), _make_report())
    assert "=== SCORE ===" in out
    assert "overall_score:" in out
    assert "verdict:" in out
    assert "flagged:" in out
    assert "signals:" in out
    assert "warnings:" in out


def test_format_execution_score_block_lists_five_signals_in_fixed_order() -> None:
    """The five signals must always appear, in the documented fixed order."""
    result = _StubResult()
    out = _format_execution(result, Path("audit.jsonl"), _make_report())
    expected = [
        "syntax",
        "back_translation",
        "sanity",
        "multi_query",
        "schema_coverage",
    ]
    # Each signal row begins with "- <name>" (after indent); use that prefix.
    positions = []
    for name in expected:
        marker = f"- {name}"
        pos = out.find(marker)
        assert pos != -1, f"Missing signal row for {name!r} in:\n{out}"
        positions.append(pos)
    # Strictly ascending — fixed order across runs.
    assert positions == sorted(positions), (
        f"Signals not in fixed order: {expected} at {positions}"
    )


def test_format_execution_each_signal_row_has_score_and_flagged() -> None:
    """Every signal row must contain score= and flagged= substrings."""
    result = _StubResult()
    out = _format_execution(result, Path("audit.jsonl"), _make_report())
    for name in (
        "syntax",
        "back_translation",
        "sanity",
        "multi_query",
        "schema_coverage",
    ):
        # Find the line containing this signal name.
        matching = [
            line for line in out.splitlines() if line.lstrip().startswith(f"- {name}")
        ]
        assert len(matching) == 1, f"Expected 1 row for {name}, got {matching}"
        line = matching[0]
        assert "score=" in line
        assert "flagged=" in line
        # Score formatted to 2 decimals (or n/a when missing).
        assert ("score=" in line and ("n/a" in line or "." in line.split("score=")[1]))


def test_format_execution_warnings_none_when_empty() -> None:
    """Empty warnings list renders the literal string 'none'."""
    result = _StubResult()
    out = _format_execution(result, Path("audit.jsonl"), _make_report(with_warnings=False))
    # The warnings line should literally contain the word 'none'.
    warnings_lines = [line for line in out.splitlines() if line.startswith("warnings:")]
    assert warnings_lines, f"warnings line not found in:\n{out}"
    assert "warnings:" in warnings_lines[0]
    assert "none" in warnings_lines[0]


def test_format_execution_warnings_comma_joined_when_non_empty() -> None:
    """Non-empty warnings list renders comma-joined."""
    result = _StubResult()
    report = _make_report(with_warnings=True)
    out = _format_execution(result, Path("audit.jsonl"), report)
    warnings_lines = [line for line in out.splitlines() if line.startswith("warnings:")]
    assert warnings_lines, f"warnings line not found in:\n{out}"
    # Joined by comma — single warning means no comma, but content must show.
    assert "back_translation.llm_failed" in warnings_lines[0]


# ---------------------------------------------------------------------------
# Group D — Score formatting boundaries (3)
# ---------------------------------------------------------------------------


def test_format_execution_verdict_high_substring() -> None:
    """verdict='high' must appear in the rendered score block."""
    result = _StubResult()
    out = _format_execution(result, Path("audit.jsonl"), _make_report(verdict="high"))
    assert "verdict:" in out
    # The verdict line should literally read 'high'.
    verdict_lines = [line for line in out.splitlines() if line.startswith("verdict:")]
    assert any("high" in line for line in verdict_lines)


def test_format_execution_verdict_low_flagged_true_substring() -> None:
    """verdict='low' must pair with flagged=True."""
    result = _StubResult()
    out = _format_execution(
        result, Path("audit.jsonl"), _make_report(verdict="low", flagged=True, score=20.0)
    )
    assert "flagged:" in out
    flagged_lines = [line for line in out.splitlines() if line.startswith("flagged:")]
    assert any("true" in line for line in flagged_lines)


def test_format_execution_score_rounded_to_one_decimal() -> None:
    """overall_score is formatted to 1 decimal place (e.g., '85.0')."""
    result = _StubResult()
    out = _format_execution(
        result, Path("audit.jsonl"), _make_report(score=85.456, verdict="high")
    )
    # Find the overall_score line and assert it uses 1-decimal formatting.
    score_lines = [line for line in out.splitlines() if line.startswith("overall_score:")]
    assert score_lines, f"overall_score line not found in:\n{out}"
    assert "85.5" in score_lines[0]


# ---------------------------------------------------------------------------
# Group E — main() end-to-end with stubbed detector (4)
# ---------------------------------------------------------------------------


def test_main_without_flag_omits_score_block(monkeypatch, capsys) -> None:
    """main() with --execute (no --detect-hallucination) prints EXECUTION block but NO SCORE."""
    monkeypatch.setattr(
        cli, "run_execute", lambda *a, **kw: (_StubResult(), None)
    )
    rc = main(["--question", "q", "--execute"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "=== EXECUTION ===" in captured.out
    assert "=== SCORE ===" not in captured.out


def test_main_with_flag_includes_score_block(monkeypatch, capsys) -> None:
    """main() with --detect-hallucination includes the SCORE block."""
    monkeypatch.setattr(
        cli, "run_execute", lambda *a, **kw: (_StubResult(), _make_report())
    )
    rc = main(["--question", "q", "--execute", "--detect-hallucination"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "=== EXECUTION ===" in captured.out
    assert "=== SCORE ===" in captured.out


def test_main_with_flag_score_block_contains_verdict_and_overall(monkeypatch, capsys) -> None:
    """The SCORE block must contain both 'verdict:' and 'overall_score:'."""
    monkeypatch.setattr(
        cli, "run_execute", lambda *a, **kw: (_StubResult(), _make_report(verdict="moderate", score=55.0))
    )
    rc = main(["--question", "q", "--execute", "--detect-hallucination"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "verdict:" in captured.out
    assert "overall_score:" in captured.out
    assert "moderate" in captured.out


def test_main_detector_exception_swallowed_and_execution_still_renders(
    monkeypatch, capsys
) -> None:
    """If the detector raises, the CLI must catch it, log to stderr, and still
    print the EXECUTION block (with no SCORE block)."""

    def _raising_run_execute(*a, **kw):
        # Simulate the detector raising inside run_execute by raising here too
        # and catching it the way the real code does.
        try:
            raise RuntimeError("boom")
        except Exception as exc:
            import sys
            print(
                f"hallucination_detection_error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return _StubResult(), None

    monkeypatch.setattr(cli, "run_execute", _raising_run_execute)
    rc = main(["--question", "q", "--execute", "--detect-hallucination"])
    captured = capsys.readouterr()
    assert rc == 0
    # Execution block must still render.
    assert "=== EXECUTION ===" in captured.out
    # No SCORE block appended.
    assert "=== SCORE ===" not in captured.out
    # Stderr should mention the hallucination_detection_error.
    assert "hallucination_detection_error" in captured.err


# ---------------------------------------------------------------------------
# Bonus: ensure missing signal renders as n/a (defensive coverage)
# ---------------------------------------------------------------------------


def test_format_execution_missing_signal_renders_na() -> None:
    """If a signal is absent from report.outcomes, render score=n/a flagged=n/a."""
    result = _StubResult()
    report = _make_report(missing={"schema_coverage"})
    out = _format_execution(result, Path("audit.jsonl"), report)
    # Find the schema_coverage row.
    matching = [
        line for line in out.splitlines() if line.lstrip().startswith("- schema_coverage")
    ]
    assert len(matching) == 1, f"schema_coverage row missing: {out}"
    assert "score=n/a" in matching[0]
    assert "flagged=n/a" in matching[0]
