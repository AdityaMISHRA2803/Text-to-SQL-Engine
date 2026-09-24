"""Tests for evals/runner.py — DB-free so they run anywhere.

These tests exercise the runner's stub-mode pipeline and report writers
without touching Postgres. The `pg_engine` / `fresh_db` fixtures in
``tests/conftest.py`` are *not* used here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.report import write_json_report, write_markdown_report
from evals.runner import (
    CaseResult,
    EvalReport,
    _build_per_category,
    _build_summary,
    _probe_database,
    _run_case_stub,
    _validate_case,
    load_golden_dataset,
    run_eval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_golden(tmp_path: Path, cases: list[dict]) -> Path:
    d = tmp_path / "golden"
    d.mkdir()
    (d / "00_test.json").write_text(json.dumps(cases), encoding="utf-8")
    return d


def _make_simple_case(**overrides) -> dict:
    base = {
        "id": "simple_count",
        "category": "simple_lookup",
        "question": "How many customers are there?",
        "golden_sql": "SELECT COUNT(*) AS n FROM customers",
        "expect": {"row_count": 1, "columns": ["n"]},
        "should_be_flagged": False,
        "difficulty": "easy",
        "notes": "trivial count",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def test_load_golden_dataset_empty_dir(tmp_path):
    """Empty dir yields no cases but doesn't raise."""
    d = tmp_path / "golden"
    d.mkdir()
    out = load_golden_dataset(d)
    assert out == []


def test_load_golden_dataset_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_golden_dataset(tmp_path / "does_not_exist")


def test_load_golden_dataset_warns_below_50(tmp_path, caplog):
    d = _write_golden(tmp_path, [_make_simple_case() for _ in range(10)])
    with caplog.at_level("WARNING"):
        cases = load_golden_dataset(d)
    assert len(cases) == 10
    assert any(">=50" in r.message for r in caplog.records)


def test_load_golden_dataset_validates_required_fields(tmp_path):
    d = tmp_path / "golden"
    d.mkdir()
    (d / "bad.json").write_text(json.dumps([{"id": "x", "question": "?"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_golden_dataset(d)


def test_load_golden_dataset_validates_golden_sql_needs_expect(tmp_path):
    d = tmp_path / "golden"
    d.mkdir()
    bad = {
        "id": "x",
        "category": "simple_lookup",
        "question": "?",
        "golden_sql": "SELECT 1",
        "should_be_flagged": False,
        "difficulty": "easy",
        "notes": "",
    }
    (d / "bad.json").write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="has golden_sql but no expect"):
        load_golden_dataset(d)


def test_validate_case_ok():
    _validate_case(_make_simple_case(), source="inline")


# ---------------------------------------------------------------------------
# _probe_database (no DB required — we just want the failure path)
# ---------------------------------------------------------------------------


def test_probe_database_returns_none_when_unreachable():
    """A bogus DSN returns (None, None) without raising."""
    engine, readonly = _probe_database("postgresql+psycopg://nope:nope@127.0.0.1:1/nope")
    assert engine is None
    assert readonly is None


# ---------------------------------------------------------------------------
# _run_case_stub — pipeline behaviour per category
# ---------------------------------------------------------------------------


def _stub_no_db(case):
    """Run stub mode with no DB and return the CaseResult."""
    counters = type("C", (), {"bump": lambda self, *a, **kw: None, "get": lambda self, k: 0})()
    return _run_case_stub(
        case,
        admin_engine=None,
        readonly_dsn=None,
        counters=counters,
    )


def test_stub_simple_lookup_trivial_match():
    res = _stub_no_db(_make_simple_case())
    assert res.sql_exact_match is True
    assert res.guardrail_allowed is True
    assert res.guardrail_blocked_reason is None
    assert res.execution_skipped is True
    assert res.execution_skip_reason == "database unavailable"
    assert res.hallucination_flagged is False  # stub back-translator returns original Q
    assert res.error is None


def test_stub_dangerous_drop_table_blocked():
    res = _stub_no_db(_make_simple_case(
        id="drop_x",
        category="dangerous",
        question="Drop the customers table.",
        golden_sql="DROP TABLE customers",
        should_be_flagged=False,
    ))
    assert res.guardrail_allowed is False
    assert res.guardrail_blocked_reason is not None
    # In stub mode, the SQL is "generated" by us, so SQL EM is True; execution is skipped
    # because the guardrail blocked it.
    assert res.execution_skipped is True


def test_stub_dangerous_delete_blocked():
    res = _stub_no_db(_make_simple_case(
        id="delete_all",
        category="dangerous",
        question="Delete everything.",
        golden_sql="DELETE FROM customers",
        should_be_flagged=False,
    ))
    assert res.guardrail_allowed is False


def test_stub_ambiguous_no_golden_sql():
    res = _stub_no_db(_make_simple_case(
        id="revenue_q",
        category="ambiguous",
        question="Show me revenue.",
        golden_sql=None,
        expect={"row_count": None, "columns": []},
        should_be_flagged=True,
    ))
    # No golden SQL → SQL EM not applicable, hallucination not exercised in stub mode.
    assert res.sql_exact_match is None
    assert res.hallucination_flagged is None
    assert res.guardrail_allowed is None


def test_stub_unanswerable():
    res = _stub_no_db(_make_simple_case(
        id="no_employees",
        category="unanswerable",
        question="How many employees do we have?",
        golden_sql=None,
        expect={"row_count": None, "columns": []},
        should_be_flagged=True,
    ))
    assert res.sql_exact_match is None
    assert res.hallucination_flagged is None


def test_stub_aggregation_simple():
    res = _stub_no_db(_make_simple_case(
        id="avg_total",
        category="aggregation",
        question="Average order total?",
        golden_sql="SELECT AVG(total) AS avg_total FROM orders",
    ))
    assert res.sql_exact_match is True
    assert res.guardrail_allowed is True
    assert res.hallucination_flagged is False


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _fake_counters(**tallies):
    class _C:
        def __init__(self, t):
            self._t = t
        def get(self, k):
            return self._t.get(k, 0)
        def rate(self, n, d):
            dd = self.get(d)
            return self.get(n) / dd if dd else 0.0
    return _C(tallies)


def _make_result(*, sql_em: bool | None = True, exec_match: bool | None = None,
                 gr_allowed: bool | None = True, hal_flagged: bool | None = False,
                 hal_should: bool = False, category: str = "simple_lookup") -> CaseResult:
    return CaseResult(
        case_id="x", category=category, question="?", difficulty="easy",
        notes="", golden_sql="SELECT 1", generated_sql="SELECT 1",
        sql_exact_match=sql_em, execution_match=exec_match,
        execution_skipped=(exec_match is None), execution_skip_reason=None,
        guardrail_allowed=gr_allowed, guardrail_blocked_reason=None,
        hallucination_flagged=hal_flagged, hallucination_should_flag=hal_should,
        hallucination_verdict="high" if not hal_flagged else "low",
        hallucination_score=90.0 if not hal_flagged else 30.0,
    )


def test_build_summary_rates():
    counters = _fake_counters(
        execution_attempted=10,
        execution_match_hits=7,
        guardrail_checked=10,
        guardrail_blocked_when_expected=3,
        guardrail_allowed_when_expected=6,
        guardrail_allowed_when_unsafe=1,
        guardrail_blocked_when_safe=0,
        hallucination_checked=5,
        hallucination_correct=4,
    )
    # 8 with SQL EM True, 2 with False.
    results = [_make_result(sql_em=True) for _ in range(8)] + [_make_result(sql_em=False) for _ in range(2)]
    summary = _build_summary(counters, results)
    assert summary["sql_exact_match"]["hits"] == 8
    assert summary["sql_exact_match"]["applicable"] == 10
    assert summary["sql_exact_match"]["rate"] == 0.8
    assert summary["execution_match"]["rate"] == 0.7
    assert summary["guardrail_effectiveness"]["rate"] == 0.9
    assert summary["guardrail_effectiveness"]["false_negatives"] == 1
    assert summary["guardrail_effectiveness"]["false_positives"] == 0
    assert summary["hallucination_detection"]["rate"] == 0.8


def test_build_per_category_rollup():
    res = CaseResult(
        case_id="x", category="simple_lookup", question="?", difficulty="easy",
        notes="", golden_sql="SELECT 1", generated_sql="SELECT 1", sql_exact_match=True,
        execution_match=None, execution_skipped=True, execution_skip_reason=None,
        guardrail_allowed=True, guardrail_blocked_reason=None,
        hallucination_flagged=False, hallucination_should_flag=False,
        hallucination_verdict="high", hallucination_score=95.0,
    )
    res2 = CaseResult(
        case_id="y", category="simple_lookup", question="?", difficulty="easy",
        notes="", golden_sql="SELECT 2", generated_sql="SELECT 2", sql_exact_match=False,
        execution_match=None, execution_skipped=True, execution_skip_reason=None,
        guardrail_allowed=True, guardrail_blocked_reason=None,
        hallucination_flagged=True, hallucination_should_flag=False,
        hallucination_verdict="low", hallucination_score=30.0,
    )
    per_cat = _build_per_category([res, res2])
    assert "simple_lookup" in per_cat
    sc = per_cat["simple_lookup"]
    assert sc["total"] == 2
    assert sc["sql_em_rate"] == 0.5
    assert sc["hal_rate"] == 0.5  # one correct, one wrong


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _sample_report() -> EvalReport:
    return EvalReport(
        timestamp="2026-07-08T00:00:00+00:00",
        mode="stub",
        provider=None,
        model=None,
        database_url="postgresql+psycopg://text2sql:text2sql@localhost:5432/text2sql",
        db_available=False,
        total_cases=1,
        summary={
            "sql_exact_match": {"applicable": 1, "hits": 1, "rate": 1.0},
            "execution_match": {"attempted": 0, "hits": 0, "rate": 0.0},
            "guardrail_effectiveness": {
                "checked": 1, "correct": 1, "rate": 1.0,
                "blocked_when_unsafe": 0, "allowed_when_safe": 1,
                "false_negatives": 0, "false_positives": 0,
            },
            "hallucination_detection": {"checked": 1, "correct": 1, "rate": 1.0},
            "totals": {"cases": 1, "executed": 0, "errors": 0},
            "elapsed_seconds": 0.5,
        },
        per_category={
            "simple_lookup": {
                "total": 1, "sql_em_applicable": 1, "sql_em_hits": 1,
                "exec_attempted": 0, "exec_hits": 0,
                "guardrail_correct": 1, "guardrail_total": 1,
                "hal_correct": 1, "hal_total": 1,
                "sql_em_rate": 1.0, "exec_match_rate": 0.0,
                "guardrail_rate": 1.0, "hal_rate": 1.0,
            },
        },
        cases=[
            CaseResult(
                case_id="sample", category="simple_lookup", question="Q?",
                difficulty="easy", notes="", golden_sql="SELECT 1",
                generated_sql="SELECT 1", sql_exact_match=True,
                execution_match=None, execution_skipped=True, execution_skip_reason="n/a",
                guardrail_allowed=True, guardrail_blocked_reason=None,
                hallucination_flagged=False, hallucination_should_flag=False,
                hallucination_verdict="high", hallucination_score=90.0,
            ),
        ],
    )


def test_write_json_report(tmp_path):
    path = tmp_path / "out.json"
    write_json_report(_sample_report(), path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["mode"] == "stub"
    assert data["total_cases"] == 1
    assert data["cases"][0]["case_id"] == "sample"


def test_write_markdown_report(tmp_path):
    path = tmp_path / "out.md"
    write_markdown_report(_sample_report(), path)
    assert path.exists()
    text = path.read_text()
    assert "# Text-to-SQL Eval Report" in text
    assert "SQL exact match" in text
    assert "Per-Category Breakdown" in text
    assert "Per-Case Results" in text
    assert "simple_lookup" in text


# ---------------------------------------------------------------------------
# run_eval (end-to-end, stub mode, no DB)
# ---------------------------------------------------------------------------


def test_run_eval_stub_no_db(tmp_path):
    """End-to-end stub-mode run with a tiny golden dataset and no DB."""
    d = _write_golden(tmp_path, [
        _make_simple_case(id="c1"),
        _make_simple_case(
            id="c2",
            question="Drop the customers table.",
            category="dangerous",
            golden_sql="DROP TABLE customers",
            should_be_flagged=False,
        ),
        _make_simple_case(
            id="c3",
            question="Show me revenue.",
            category="ambiguous",
            golden_sql=None,
            expect={"row_count": None, "columns": []},
            should_be_flagged=True,
        ),
    ])
    report = run_eval(mode="stub", golden_dir=d, database_url="postgresql+psycopg://nope:nope@127.0.0.1:1/nope")
    assert report.mode == "stub"
    assert report.total_cases == 3
    assert report.db_available is False
    # Cases by id
    by_id = {c.case_id: c for c in report.cases}
    assert by_id["c1"].sql_exact_match is True
    assert by_id["c2"].guardrail_allowed is False
    assert by_id["c3"].sql_exact_match is None
    # Headline metrics reflect the goldens
    assert report.summary["guardrail_effectiveness"]["checked"] == 2  # c1 + c2 only


def test_run_eval_stub_rejects_bad_mode(tmp_path):
    d = _write_golden(tmp_path, [_make_simple_case()])
    with pytest.raises(ValueError, match="Invalid mode"):
        run_eval(mode="invalid", golden_dir=d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_stub_mode_runs(tmp_path, capsys):
    """`python -m evals.run` exits 0 and writes JSON + Markdown."""
    import argparse
    from evals.runner import main

    d = _write_golden(tmp_path, [_make_simple_case(id="cli1")])
    report_dir = tmp_path / "reports"
    rc = main([
        "--mode", "stub",
        "--golden-dir", str(d),
        "--report-dir", str(report_dir),
        "--quiet",
    ])
    assert rc == 0
    files = list(report_dir.glob("phase5_*.json"))
    md_files = list(report_dir.glob("phase5_*.md"))
    assert files, "expected a JSON report"
    assert md_files, "expected a Markdown report"
    captured = capsys.readouterr()
    assert "Eval complete" in captured.out
