"""Tests for the Sandbox (README 2.3 — defence-in-depth execution layer).

Verifies:
- Sandbox is a context manager that yields a Connection.
- SELECT inside the sandbox does not corrupt the outer transaction:
  a row inserted via the main engine is still present after a sandbox SELECT.
- Writes inside the sandbox (if any could slip through) are ALWAYS rolled back.
- A monkeypatched-bypass of GuardrailValidator still cannot DROP a table
  because the DB layer's text2sql_ro role has no DROP privilege.
- The readonly role can SELECT from seeded tables (positive smoke).
"""
from __future__ import annotations

import pytest
import sqlalchemy
from sqlalchemy import text

from text2sql.execution import Sandbox
from text2sql.execution.sandbox import _SandboxConnection
from text2sql.guardrails import GuardrailValidator, ValidationResult


# ---------------------------------------------------------------------------
# API smoke (2 tests)
# ---------------------------------------------------------------------------


def test_sandbox_connection_is_a_context_manager(readonly_dsn: str) -> None:
    sandbox = Sandbox(dsn=readonly_dsn)
    cm = sandbox.connection()
    assert isinstance(cm, _SandboxConnection)


def test_sandbox_connection_yields_connection(readonly_dsn: str) -> None:
    sandbox = Sandbox(dsn=readonly_dsn)
    with sandbox.connection() as conn:
        assert hasattr(conn, "execute")
        # Smoke: a trivial SELECT returns one row
        result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1


# ---------------------------------------------------------------------------
# Rollback semantics (2 tests)
# ---------------------------------------------------------------------------


def test_sandbox_select_does_not_affect_outer_transaction(
    readonly_dsn: str, fresh_db
) -> None:
    """Insert a row via the main engine, SELECT it via the sandbox,
    then re-SELECT via the main engine and confirm the row is still there.
    Proves the sandbox's transaction does not interfere with the outer state.
    """
    # Insert via main engine.
    with fresh_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO customers (id, email, first_name, last_name, country, created_at) "
                "VALUES (999001, 'sandbox@test.com', 'Sbx', 'Tester', 'USA', NOW())"
            )
        )

    sandbox = Sandbox(dsn=readonly_dsn)
    with sandbox.connection() as conn:
        # Sandbox SELECT must succeed (readonly user has SELECT privilege).
        rows = conn.execute(
            text("SELECT id FROM customers WHERE id = 999001")
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 999001

    # After exiting the sandbox, the row must STILL be visible via the main engine.
    with fresh_db.connect() as conn:
        rows = conn.execute(
            text("SELECT id FROM customers WHERE id = 999001")
        ).fetchall()
        assert len(rows) == 1


def test_sandbox_always_rolls_back_writes_on_success(pg_engine, readonly_dsn: str) -> None:
    """Even on a successful context-manager exit, any write inside the sandbox
    must be rolled back. This proves the always-rollback contract holds even
    when no exception is raised.

    We use a tiny throwaway table so we don't perturb fresh_db fixtures used
    by other tests in the session. We exercise `_SandboxConnection` directly
    with the MAIN engine (which CAN write) to confirm the rollback behaviour;
    the readonly role privilege test is covered separately.
    """
    # Create and populate a throwaway table on the main engine.
    with pg_engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS _sandbox_rollback_probe"))
        conn.execute(
            text(
                "CREATE TABLE _sandbox_rollback_probe ("
                "  id INT PRIMARY KEY, "
                "  label TEXT NOT NULL"
                ")"
            )
        )
        conn.execute(
            text("INSERT INTO _sandbox_rollback_probe (id, label) VALUES (1, 'before')")
        )

    # Directly exercise _SandboxConnection with the MAIN engine (which has
    # write privileges) so we can prove the always-rollback contract.
    sandbox_conn = _SandboxConnection(pg_engine)
    with sandbox_conn as conn:
        conn.execute(
            text(
                "INSERT INTO _sandbox_rollback_probe (id, label) "
                "VALUES (2, 'inside-sandbox')"
            )
        )

    # The insert must NOT be visible — sandbox always rolls back on exit.
    with pg_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, label FROM _sandbox_rollback_probe ORDER BY id")
        ).fetchall()
    assert rows == [(1, "before")], f"sandbox did not roll back; rows={rows}"

    # Cleanup.
    with pg_engine.begin() as conn:
        conn.execute(text("DROP TABLE _sandbox_rollback_probe"))


# ---------------------------------------------------------------------------
# Defence-in-depth: monkeypatched guardrail bypass still blocked by DB (2 tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def bypass_guardrail(monkeypatch):
    """Force GuardrailValidator.validate to allow any SQL (simulates a bypass)."""

    def _allow_any(self, sql, *, schema=None, engine=None, provider="unknown"):  # noqa: D401
        return ValidationResult(sql=sql, allowed=True, warnings=[])

    monkeypatch.setattr(GuardrailValidator, "validate", _allow_any)


def test_sandbox_blocks_drop_via_readonly_user(
    bypass_guardrail, pg_engine, fresh_db, readonly_dsn: str
) -> None:
    """Even with a guardrail bypass, the sandbox cannot DROP because the
    text2sql_ro role lacks the privilege. The DB layer raises
    InsufficientPrivilege (a subclass of sqlalchemy.exc.DBAPIError).
    """
    sandbox = Sandbox(dsn=readonly_dsn)
    with pytest.raises(sqlalchemy.exc.DBAPIError):
        with sandbox.connection() as conn:
            conn.execute(text("DROP TABLE customers"))


def test_sandbox_blocks_insert_via_readonly_user(
    bypass_guardrail, pg_engine, fresh_db, readonly_dsn: str
) -> None:
    """The readonly role has only SELECT; INSERT must fail with InsufficientPrivilege."""
    sandbox = Sandbox(dsn=readonly_dsn)
    with pytest.raises(sqlalchemy.exc.DBAPIError):
        with sandbox.connection() as conn:
            conn.execute(
                text(
                    "INSERT INTO customers (email, first_name, last_name, country, created_at) "
                    "VALUES ('hacker@x.com', 'X', 'Y', 'USA', NOW())"
                )
            )


# ---------------------------------------------------------------------------
# Readonly SELECT smoke (1 test)
# ---------------------------------------------------------------------------


def test_readonly_user_can_select_seeded_customers(
    pg_engine, fresh_db, readonly_dsn: str
) -> None:
    """The readonly user must be able to SELECT from the seeded tables."""
    sandbox = Sandbox(dsn=readonly_dsn)
    with sandbox.connection() as conn:
        rows = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
    assert rows == 50, f"expected 50 seeded customers, got {rows}"
