"""Tests for the AuditLog append-only JSON-lines writer (README 2.4)."""
import json
from pathlib import Path

from text2sql.execution.audit import AuditLog


def test_audit_appends_and_reads_back(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "sub" / "queries.jsonl")
    log.append({"question": "q1", "row_count": 5})
    log.append({"question": "q2", "row_count": 0})
    records = log.read_all()
    assert len(records) == 2
    assert records[0]["question"] == "q1"
    assert records[0]["row_count"] == 5
    assert records[1]["question"] == "q2"
    # auto-timestamp
    assert "iso_timestamp" in records[0]


def test_audit_read_all_missing_file(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "nope.jsonl")
    assert log.read_all() == []


def test_audit_one_line_per_record(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "x.jsonl")
    log.append({"a": 1})
    log.append({"a": 2})
    raw = (tmp_path / "x.jsonl").read_text()
    # exactly two non-blank lines
    non_blank = [ln for ln in raw.splitlines() if ln.strip()]
    assert len(non_blank) == 2
    # each line is valid JSON
    for ln in non_blank:
        json.loads(ln)


def test_audit_preserves_explicit_timestamp(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "ts.jsonl")
    log.append({"question": "q", "iso_timestamp": "2025-01-01T00:00:00+00:00"})
    records = log.read_all()
    assert records[0]["iso_timestamp"] == "2025-01-01T00:00:00+00:00"
