"""End-to-end CLI test for Phase 2 (--execute flag)."""
import io
import sys
from pathlib import Path

import pytest

from text2sql import cli as cli_module
from text2sql.ambiguity import ClarificationRequest, Interpretation
from text2sql.cli import main, run_execute
from text2sql.execution.audit import AuditLog
from text2sql.execution.runner import ExecutionResult
from text2sql.generation import GeneratedSQL
from text2sql.generation.generator import SQLGenerator


def _canned_sql() -> str:
    return "SELECT id, email FROM customers ORDER BY id LIMIT 3"


def _patch_sql_generator(monkeypatch: pytest.MonkeyPatch, canned: GeneratedSQL) -> None:
    """Replace SQLGenerator.generate with a stub returning `canned`.

    Uses class-based setattr so the patch lands on the actual class
    object regardless of which import path callers used.
    """

    def stub_generate(self, question, schema):  # noqa: D401
        return canned

    monkeypatch.setattr(SQLGenerator, "generate", stub_generate)


def _patch_ambiguity_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace AmbiguityHandler.detect so the Phase 1 main() path doesn't
    need a real LLM."""

    def stub_detect(self, question, schema):  # noqa: D401
        return ClarificationRequest(
            question=question,
            ambiguous=False,
            interpretations=[
                Interpretation(sql="SELECT 1", explanation="stub"),
            ],
            model_id="stub/test",
        )

    from text2sql.ambiguity.handler import AmbiguityHandler as _Handler

    monkeypatch.setattr(_Handler, "detect", stub_detect)


def _patch_resolve_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CLI to use the deterministic fake embedder so we never hit
    litellm.embedding (which would 401 when no API key is set or fail
    authentication with a stale key in the test environment)."""
    from text2sql import cli as _cli

    monkeypatch.setattr(
        _cli,
        "_resolve_embedder",
        lambda provider, api_key: (_cli._fake_embedder(), "fake:forced"),
    )


def test_run_execute_end_to_end(
    fresh_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Call run_execute(...) directly with a stubbed SQLGenerator.

    Asserts the returned ExecutionResult has the expected row_count,
    the audit log has one matching entry, and execution_time_ms is an int.
    """
    sql = _canned_sql()
    canned = GeneratedSQL(
        sql=sql,
        explanation="stub",
        confidence=0.9,
        tables_touched=["customers"],
        columns_touched=["customers.id", "customers.email"],
    )
    _patch_sql_generator(monkeypatch, canned)
    _patch_resolve_embedder(monkeypatch)

    # Quiet the prompt-to-stderr noise so test stdout stays clean.
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    audit_path = tmp_path / "audit.jsonl"
    result, _hallucination = run_execute(
        "show me first 3 customers",
        provider="openai",
        database_url=fresh_db.url.render_as_string(hide_password=False),
        audit_path=audit_path,
    )

    assert isinstance(result, ExecutionResult)
    assert result.allowed is True
    assert result.row_count == 3
    assert isinstance(result.execution_time_ms, int)
    assert result.execution_time_ms >= 0
    assert result.sql == sql
    assert result.results[0].keys() == {"id", "email"}

    records = AuditLog(audit_path).read_all()
    assert len(records) == 1
    assert records[0]["sql"] == sql
    assert records[0]["row_count"] == 3
    assert records[0]["allowed"] is True
    assert records[0]["question"] == "show me first 3 customers"


def test_run_execute_blocks_destructive_query(
    fresh_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_execute against a GeneratedSQL with a destructive statement must
    return allowed=False with blocked_reason set, AND the audit log must
    still record the attempt."""
    canned = GeneratedSQL(
        sql="DROP TABLE customers",
        explanation="destructive",
        confidence=0.5,
        tables_touched=["customers"],
        columns_touched=[],
    )
    _patch_sql_generator(monkeypatch, canned)
    _patch_resolve_embedder(monkeypatch)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    audit_path = tmp_path / "audit.jsonl"
    result, _hallucination = run_execute(
        "delete the table",
        provider="openai",
        database_url=fresh_db.url.render_as_string(hide_password=False),
        audit_path=audit_path,
    )

    assert result.allowed is False
    assert result.blocked_reason is not None
    assert result.row_count == 0

    records = AuditLog(audit_path).read_all()
    assert len(records) == 1
    assert records[0]["allowed"] is False
    assert records[0]["blocked_reason"] is not None


def test_cli_main_executes_and_prints_block(
    fresh_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full `python -m text2sql.cli --question ... --execute --audit-path ...` end-to-end.

    We monkeypatch SQLGenerator.generate to return canned SQL and capture
    stdout. Asserts the EXECUTION block is printed and the audit log is
    written with the matching entry.
    """
    sql = _canned_sql()
    canned = GeneratedSQL(
        sql=sql,
        explanation="stub",
        confidence=0.9,
        tables_touched=["customers"],
        columns_touched=["customers.id", "customers.email"],
    )
    _patch_sql_generator(monkeypatch, canned)
    _patch_resolve_embedder(monkeypatch)

    audit_path = tmp_path / "audit.jsonl"

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    rc = main([
        "--question", "show me first 3 customers",
        "--provider", "openai",
        "--execute",
        "--audit-path", str(audit_path),
        "--database-url", fresh_db.url.render_as_string(hide_password=False),
    ])

    assert rc == 0
    out = captured.getvalue()

    # EXECUTION block present
    assert "=== EXECUTION ===" in out
    # row_count of 3
    assert "row_count:         3" in out
    # execution_time_ms printed
    assert "execution_time_ms:" in out
    # sql printed (will be the original sql, possibly with LIMIT appended)
    assert sql in out
    # audit log path printed
    assert str(audit_path.resolve()) in out
    # audit log written with one entry
    records = AuditLog(audit_path).read_all()
    assert len(records) == 1
    assert records[0]["row_count"] == 3
    assert records[0]["allowed"] is True


def test_cli_main_without_execute_uses_phase1_path(
    fresh_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --execute, the CLI keeps the Phase 1 contract (PROMPT + AMBIGUITY)."""
    _patch_ambiguity_handler(monkeypatch)
    _patch_resolve_embedder(monkeypatch)

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    rc = main([
        "--question", "show me first 3 customers",
        "--provider", "openai",
        "--database-url", fresh_db.url.render_as_string(hide_password=False),
    ])
    assert rc == 0
    out = captured.getvalue()
    assert "=== PROMPT ===" in out
    assert "=== AMBIGUITY ===" in out
    # Should NOT contain EXECUTION block (no --execute flag)
    assert "=== EXECUTION ===" not in out
