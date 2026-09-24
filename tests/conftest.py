"""Shared pytest fixtures for the text2sql test suite."""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from text2sql.seed_dataset import Base, seed

# Read-only role used by the Sandbox for defence-in-depth.
READONLY_ROLE = "text2sql_ro"
READONLY_PASSWORD = "text2sql_ro"


def _start_container():
    """Start a Postgres testcontainer or raise a docker-related exception."""
    # testcontainers 4.x renamed the module from `postgresql` (plural) to
    # `postgres` (singular). Try the new name first, fall back to the old
    # for compatibility with older releases that may still be installed.
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover - older testcontainers
        from testcontainers.postgresql import PostgresContainer

    return PostgresContainer(
        "postgres:16-alpine",
        username="text2sql",
        password="text2sql",
        dbname="text2sql",
    )


@pytest.fixture(scope="session")
def pg_engine() -> Iterator[Engine]:
    """Spin up a Postgres testcontainer for the test session and yield a SQLAlchemy engine.

    Skips the test session cleanly if Docker is not reachable.

    Provisions a `text2sql_ro` role once per session so the sandbox's
    SELECT-only defence-in-depth has somewhere to connect as.
    """
    try:
        container = _start_container()
        container.start()
    except Exception as exc:  # docker.errors.DockerException, ConnectionError, etc.
        pytest.skip(f"Docker not available; skipping Postgres fixture ({exc})")

    try:
        url = container.get_connection_url()
        # testcontainers returns a postgresql:// or postgresql+psycopg2:// URL;
        # normalise to the psycopg3 driver that this project uses.
        if url.startswith("postgresql+psycopg2://"):
            url = url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        engine = create_engine(url, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            # Provision the readonly role. DROP IF EXISTS first so reruns
            # in the same session (or a stuck container) don't fail.
            conn.execute(text("COMMIT"))  # ensure clean tx for DDL
            conn.execute(text(f"DROP USER IF EXISTS {READONLY_ROLE}"))
            conn.execute(
                text(f"CREATE USER {READONLY_ROLE} WITH PASSWORD '{READONLY_PASSWORD}'")
            )
            conn.commit()
        yield engine
    finally:
        try:
            container.stop()
        except Exception:
            pass


@pytest.fixture
def fresh_db(pg_engine: Engine) -> Iterator[Engine]:
    """Recreate all tables, seed a small dataset, and GRANT SELECT to the readonly role."""
    Base.metadata.drop_all(pg_engine)
    Base.metadata.create_all(pg_engine)
    seed(pg_engine, row_count=50)
    # Grant SELECT on every public table to the readonly role so the
    # sandbox's defence-in-depth connection can read seeded data.
    with pg_engine.begin() as conn:
        conn.execute(
            text(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {READONLY_ROLE}")
        )
    try:
        yield pg_engine
    finally:
        # Revoke before drop_all so the role doesn't keep dangling privileges.
        with pg_engine.begin() as conn:
            conn.execute(
                text(f"REVOKE SELECT ON ALL TABLES IN SCHEMA public FROM {READONLY_ROLE}")
            )
        Base.metadata.drop_all(pg_engine)


@pytest.fixture(scope="session")
def readonly_dsn(pg_engine: Engine) -> str:
    """Return a SQLAlchemy URL string for connecting as the readonly role.

    The DSN is identical to the main engine URL except username/password
    are swapped to the SELECT-only `text2sql_ro` role.
    """
    url = pg_engine.url
    new_url = url.set(username=READONLY_ROLE, password=READONLY_PASSWORD)
    return new_url.render_as_string(hide_password=False)
