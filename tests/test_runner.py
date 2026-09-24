"""Tests for the QueryRunner (README 2.4)."""
import json
from pathlib import Path

from text2sql.execution.audit import AuditLog
from text2sql.execution.runner import ExecutionResult, QueryRunner
from text2sql.execution.sandbox import Sandbox
from text2sql.generation import GeneratedSQL
from text2sql.guardrails import GuardrailValidator


def _make_runner(audit_path: Path, *, readonly_dsn: str) -> QueryRunner:
    validator = GuardrailValidator(log_blocked=False)
    sandbox = Sandbox(dsn=readonly_dsn)
    return QueryRunner(validator=validator, sandbox=sandbox, audit_path=audit_path)


def test_runner_executes_validated_select(
    fresh_db, readonly_dsn: str, tmp_path: Path
) -> None:
    runner = _make_runner(tmp_path / "audit.jsonl", readonly_dsn=readonly_dsn)
    generated = GeneratedSQL(
        sql="SELECT id, email FROM customers ORDER BY id LIMIT 5",
        explanation="first 5 customers",
        confidence=0.9,
        tables_touched=["customers"],
        columns_touched=["customers.id", "customers.email"],
    )
    result = runner.run(generated, "first 5 customers")
    assert isinstance(result, ExecutionResult)
    assert result.allowed is True
    assert result.row_count == 5
    assert isinstance(result.results, list)
    assert all(isinstance(r, dict) for r in result.results)
    assert result.results[0].keys() == {"id", "email"}


def test_runner_execution_time_is_non_negative_int(
    fresh_db, readonly_dsn: str, tmp_path: Path
) -> None:
    runner = _make_runner(tmp_path / "audit.jsonl", readonly_dsn=readonly_dsn)
    generated = GeneratedSQL(
        sql="SELECT 1 AS one",
        explanation="trivial",
        confidence=1.0,
        tables_touched=[],
        columns_touched=[],
    )
    result = runner.run(generated, "trivial")
    assert isinstance(result.execution_time_ms, int)
    assert result.execution_time_ms >= 0  # could be 0 for very fast queries on fast CI


def test_runner_blocks_invalid_sql_via_validator(
    fresh_db, readonly_dsn: str, tmp_path: Path
) -> None:
    runner = _make_runner(tmp_path / "audit.jsonl", readonly_dsn=readonly_dsn)
    generated = GeneratedSQL(
        sql="DROP TABLE customers",
        explanation="destructive",
        confidence=0.5,
        tables_touched=["customers"],
        columns_touched=[],
    )
    result = runner.run(generated, "delete table")
    assert result.allowed is False
    assert result.blocked_reason is not None
    assert result.row_count == 0
    assert result.results == []


def test_runner_writes_audit_log_one_line_per_run(
    fresh_db, readonly_dsn: str, tmp_path: Path
) -> None:
    audit_path = tmp_path / "audit.jsonl"
    runner = _make_runner(audit_path, readonly_dsn=readonly_dsn)
    generated = GeneratedSQL(
        sql="SELECT 1 AS x",
        explanation="trivial",
        confidence=1.0,
        tables_touched=[],
        columns_touched=[],
    )
    runner.run(generated, "trivial")
    runner.run(generated, "trivial")

    records = AuditLog(audit_path).read_all()
    assert len(records) == 2
    required = {
        "question", "sql", "confidence", "tables_touched", "columns_touched",
        "row_count", "execution_time_ms", "allowed", "blocked_reason",
        "iso_timestamp",
    }
    for rec in records:
        assert required.issubset(rec.keys())


def test_runner_results_serializable_to_json(
    fresh_db, readonly_dsn: str, tmp_path: Path
) -> None:
    """Each result dict must be JSON-serialisable so the audit log works."""
    runner = _make_runner(tmp_path / "audit.jsonl", readonly_dsn=readonly_dsn)
    generated = GeneratedSQL(
        sql="SELECT id FROM customers LIMIT 1",
        explanation="first",
        confidence=1.0,
        tables_touched=["customers"],
        columns_touched=["customers.id"],
    )
    result = runner.run(generated, "first")
    # whole result model serialises cleanly
    json_str = result.model_dump_json()
    assert json.loads(json_str)["row_count"] == 1


def test_runner_audit_records_blocked_query(
    fresh_db, readonly_dsn: str, tmp_path: Path
) -> None:
    """Blocked queries must still appear in the audit log (record != silent)."""
    audit_path = tmp_path / "audit.jsonl"
    runner = _make_runner(audit_path, readonly_dsn=readonly_dsn)
    generated = GeneratedSQL(
        sql="DELETE FROM customers",
        explanation="bad",
        confidence=0.1,
        tables_touched=["customers"],
        columns_touched=[],
    )
    result = runner.run(generated, "delete all")
    assert result.allowed is False

    records = AuditLog(audit_path).read_all()
    assert len(records) == 1
    assert records[0]["allowed"] is False
    assert records[0]["blocked_reason"] is not None
    assert records[0]["row_count"] == 0
