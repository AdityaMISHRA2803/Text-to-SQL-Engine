"""Sandbox: SELECT-only context manager that ALWAYS rolls back (README 2.3).

Defence in depth: even if a guardrail validator is bypassed (bug, monkeypatch,
misconfiguration), the DB layer refuses to honour destructive operations
because the sandbox connects as a SELECT-only role and wraps every
execution in a transaction that ALWAYS rolls back on exit.

Public API:
    Sandbox(*, dsn: str, readonly_user: str = "text2sql_ro",
            readonly_password: str = "text2sql_ro") -> Sandbox
    Sandbox.readonly_engine() -> Engine
    Sandbox.connection() -> _SandboxConnection (context manager)

    with sandbox.connection() as conn:
        result = conn.execute(text("SELECT ..."))

The context manager:
    - Opens a transaction via engine.begin()
    - Yields the inner Connection
    - ALWAYS rolls back on __exit__, even when no exception was raised
    - Re-raises any exception raised inside the with-block after rollback

The `_SandboxConnection` class is exposed for type-checkers but tests should
always go through `Sandbox.connection()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine, URL


@dataclass
class Sandbox:
    """Sandbox that runs every SQL inside a transaction that always rolls back."""

    dsn: str
    readonly_user: str = "text2sql_ro"
    readonly_password: str = "text2sql_ro"
    _readonly_engine: Optional[Engine] = None

    def readonly_engine(self) -> Engine:
        """Return a SQLAlchemy Engine connected as the readonly user.

        The engine is cached on the instance so multiple `.connection()`
        calls share a single engine (engines are heavy).
        """
        if self._readonly_engine is None:
            url = self._parse_dsn(self.dsn)
            url = url.set(username=self.readonly_user, password=self.readonly_password)
            self._readonly_engine = create_engine(url, future=True)
        return self._readonly_engine

    def connection(self) -> "_SandboxConnection":
        """Open a sandboxed connection that ALWAYS rolls back on exit."""
        return _SandboxConnection(self.readonly_engine())

    @staticmethod
    def _parse_dsn(dsn: str) -> URL:
        """Parse a SQLAlchemy DSN string into a URL object.

        We delegate to ``sqlalchemy.engine.url.make_url`` so any driver
        prefix and quoting rules are honoured identically to SQLAlchemy's
        own config parsing.
        """
        from sqlalchemy.engine.url import make_url

        return make_url(dsn)


class _SandboxConnection:
    """Context manager that yields a Connection inside an always-rollback transaction."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._tx = None
        self._conn: Optional[Connection] = None

    def __enter__(self) -> Connection:
        # engine.connect() returns a plain Connection; conn.begin() opens a
        # Transaction that does NOT auto-commit on clean exit (unlike
        # engine.begin(), which returns a _GeneratorContextManager that
        # commits when __exit__ is called cleanly).
        self._conn = self._engine.connect()
        self._tx = self._conn.begin()
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            # ALWAYS roll back, even if no exception was raised. Rollback
            # is a no-op for SELECTs but undoes any writes that somehow
            # slipped through (e.g. superuser bypass, broken GRANT).
            if self._tx is not None:
                self._tx.rollback()
        finally:
            # Close the inner connection regardless of rollback outcome.
            if self._conn is not None:
                self._conn.close()


__all__ = ["Sandbox", "_SandboxConnection"]
