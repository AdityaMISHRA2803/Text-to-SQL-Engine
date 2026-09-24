"""Report writers for the eval suite.

Produces two artefacts per run:

* **JSON** — full machine-readable copy of the ``EvalReport`` dataclass tree.
* **Markdown** — human-readable summary with top-level metrics, per-category
  breakdown, and a per-case detail section. Designed to be committed alongside
  the eval code so a reviewer can read the numbers without re-running.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .runner import EvalReport


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def write_json_report(report: EvalReport, out_path: Path) -> Path:
    """Serialize the report to JSON.

    Writes atomically: writes to ``<out_path>.tmp`` then renames.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    tmp.replace(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def write_markdown_report(report: EvalReport, out_path: Path) -> Path:
    """Render the report as a Markdown document."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    _add_header(lines, report)
    _add_summary(lines, report)
    _add_per_category(lines, report)
    _add_per_case(lines, report)
    _add_failures(lines, report)
    _add_footer(lines, report)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _add_header(lines: list[str], r: EvalReport) -> None:
    lines.append(f"# Text-to-SQL Eval Report — {r.timestamp}")
    lines.append("")
    lines.append(f"- **Mode:** `{r.mode}`"
                 + (f" (provider=`{r.provider}`, model=`{r.model}`)" if r.mode == "live" else ""))
    lines.append(f"- **Database:** `{'reachable' if r.db_available else 'unreachable'}` at `{r.database_url}`")
    lines.append(f"- **Total cases:** {r.total_cases}")
    lines.append(f"- **Wall time:** {r.summary.get('elapsed_seconds', 0):.1f} s")
    lines.append("")


def _add_summary(lines: list[str], r: EvalReport) -> None:
    s = r.summary
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append("| Metric | Hits / Total | Rate |")
    lines.append("|---|---:|---:|")
    sql_em = s["sql_exact_match"]
    lines.append(f"| SQL exact match | {sql_em['hits']}/{sql_em['applicable']} | "
                 f"**{sql_em['rate'] * 100:.1f}%** |")
    exec_m = s["execution_match"]
    lines.append(f"| Execution match | {exec_m['hits']}/{exec_m['attempted']} | "
                 f"**{exec_m['rate'] * 100:.1f}%** |")
    gr = s["guardrail_effectiveness"]
    lines.append(f"| Guardrail effectiveness | {gr['correct']}/{gr['checked']} | "
                 f"**{gr['rate'] * 100:.1f}%** "
                 f"(FN={gr['false_negatives']}, FP={gr['false_positives']}) |")
    hal = s["hallucination_detection"]
    lines.append(f"| Hallucination detection | {hal['correct']}/{hal['checked']} | "
                 f"**{hal['rate'] * 100:.1f}%** |")
    lines.append("")


def _add_per_category(lines: list[str], r: EvalReport) -> None:
    if not r.per_category:
        return
    lines.append("## Per-Category Breakdown")
    lines.append("")
    lines.append("| Category | n | SQL EM | Exec match | Guardrail | Hallucination |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cat in sorted(r.per_category):
        c = r.per_category[cat]
        lines.append(
            f"| {cat} | {c['total']} | "
            f"{c['sql_em_rate'] * 100:.0f}% ({c['sql_em_hits']}/{c['sql_em_applicable']}) | "
            f"{c['exec_match_rate'] * 100:.0f}% ({c['exec_hits']}/{c['exec_attempted']}) | "
            f"{c['guardrail_rate'] * 100:.0f}% ({c['guardrail_correct']}/{c['guardrail_total']}) | "
            f"{c['hal_rate'] * 100:.0f}% ({c['hal_correct']}/{c['hal_total']}) |"
        )
    lines.append("")


def _add_per_case(lines: list[str], r: EvalReport) -> None:
    lines.append("## Per-Case Results")
    lines.append("")
    lines.append("<details><summary>Click to expand</summary>")
    lines.append("")
    for case in r.cases:
        lines.append(f"### `{case.case_id}` — {case.category} ({case.difficulty})")
        lines.append("")
        lines.append(f"- **Q:** {case.question}")
        if case.golden_sql:
            lines.append(f"- **Golden SQL:**")
            lines.append("")
            lines.append("  ```sql")
            lines.append(f"  {case.golden_sql}")
            lines.append("  ```")
        if case.generated_sql and case.generated_sql != case.golden_sql:
            lines.append(f"- **Generated SQL:**")
            lines.append("")
            lines.append("  ```sql")
            lines.append(f"  {case.generated_sql}")
            lines.append("  ```")
        flags: list[str] = []
        if case.sql_exact_match is not None:
            flags.append(f"SQL EM={'✓' if case.sql_exact_match else '✗'}")
        if case.execution_skipped:
            flags.append(f"exec=skipped ({case.execution_skip_reason or 'n/a'})")
        elif case.execution_match is not None:
            flags.append(f"exec={'✓' if case.execution_match else '✗'}")
        if case.guardrail_allowed is not None:
            if case.guardrail_allowed:
                flags.append("guardrail=allowed")
            else:
                flags.append(f"guardrail=blocked({case.guardrail_blocked_reason})")
        if case.hallucination_flagged is not None:
            expect = "flag" if case.hallucination_should_flag else "pass"
            got = "flag" if case.hallucination_flagged else "pass"
            mark = "✓" if expect == got else "✗"
            flags.append(
                f"hal: expected={expect} got={got} {mark} "
                f"(verdict={case.hallucination_verdict}, score={case.hallucination_score})"
            )
        if flags:
            lines.append(f"- **Result:** {' · '.join(flags)}")
        if case.error:
            lines.append(f"- **Error:** `{case.error}`")
        if case.notes:
            lines.append(f"- **Notes:** {case.notes}")
        if case.detector_notes:
            lines.append(f"- **Detector notes:** {'; '.join(case.detector_notes)}")
        lines.append("")
    lines.append("</details>")
    lines.append("")


def _add_failures(lines: list[str], r: EvalReport) -> None:
    failures = [
        c for c in r.cases
        if c.error
        or (c.sql_exact_match is False)
        or (c.execution_match is False)
        or (c.hallucination_flagged is not None
            and c.hallucination_flagged != c.hallucination_should_flag)
        or (c.guardrail_allowed is False and c.category != "dangerous")
        or (c.guardrail_allowed is True and c.category == "dangerous")
    ]
    if not failures:
        lines.append("## Failures")
        lines.append("")
        lines.append("_None._")
        lines.append("")
        return
    lines.append("## Failures")
    lines.append("")
    lines.append(f"_{len(failures)} of {r.total_cases} cases need attention._")
    lines.append("")
    for c in failures:
        bits: list[str] = [f"`{c.case_id}` ({c.category})"]
        if c.error:
            bits.append(f"error: `{c.error}`")
        if c.sql_exact_match is False:
            bits.append("SQL EM failed")
        if c.execution_match is False:
            bits.append("exec match failed")
        if c.guardrail_allowed is False and c.category != "dangerous":
            bits.append(f"FALSE POSITIVE: guardrail blocked safe query ({c.guardrail_blocked_reason})")
        if c.guardrail_allowed is True and c.category == "dangerous":
            bits.append("FALSE NEGATIVE: guardrail did NOT block dangerous query!")
        if c.hallucination_flagged is not None and c.hallucination_flagged != c.hallucination_should_flag:
            expect = "flag" if c.hallucination_should_flag else "pass"
            got = "flag" if c.hallucination_flagged else "pass"
            bits.append(f"hallucination: expected={expect} got={got}")
        lines.append(f"- {' — '.join(bits)}")
    lines.append("")


def _add_footer(lines: list[str], r: EvalReport) -> None:
    s = r.summary
    lines.append("---")
    lines.append("")
    lines.append(f"_Generated by `python -m evals.run` at {r.timestamp}. "
                 f"Mode={r.mode}, "
                 f"DB={'ok' if r.db_available else 'skipped'}, "
                 f"elapsed={s.get('elapsed_seconds', 0):.1f}s, "
                 f"errors={s['totals']['errors']}._")
    lines.append("")
