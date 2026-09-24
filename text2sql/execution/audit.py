"""Append-only audit log for executed queries (README 2.4).

Each line of the audit file is one JSON object describing one query
execution (or attempted execution). Lines are flushed after every append
so a crash mid-run does not lose entries. The log is read back via
`read_all()` for downstream tooling (dashboards, evals, debugging).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    """Append-only JSON-lines audit log for executed queries.

    Each line is one JSON object. Lines are flushed after every append
    so a crash mid-run does not lose entries.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        """Append one record as a single JSON line.

        Adds an `iso_timestamp` field (UTC, ISO 8601) if not already present.
        Creates parent directories lazily on first append.
        """
        if "iso_timestamp" not in record and "timestamp" not in record:
            record = {**record, "iso_timestamp": datetime.now(timezone.utc).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()

    def read_all(self) -> list[dict[str, Any]]:
        """Read all records. Returns [] if file does not exist.

        Skips blank lines. Tolerant of trailing newline.
        """
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines]


__all__ = ["AuditLog"]
