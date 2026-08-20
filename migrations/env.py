"""Alembic environment (roadmap T3.5.8).

Three choices here are deliberate and each closes a specific failure mode.

*   **The URL comes from the application, not from `alembic.ini`.** `DATABASE_URL` is read
    once, in `src/database/session.py`; this file imports it. A URL duplicated in the ini
    file would eventually diverge from the one the API uses, and the symptom of that -
    migrating a database nobody reads - is indistinguishable from a migration that did
    nothing.

*   **`render_as_batch=True`.** SQLite cannot `ALTER COLUMN` or (before 3.35) `DROP COLUMN`.
    Batch mode makes Alembic emit the copy-and-rename dance instead, so the *same* migration
    script runs on SQLite and PostgreSQL. Without it, any future column alteration would
    produce a migration that works on the developer's PostgreSQL and fails on the laptop
    SQLite the platform is meant to run on.

*   **`compare_type=True` and `compare_server_default=True`.** These affect autogenerate
    only, and they are what make `src/tests/test_migrations.py::test_no_drift` a real guard:
    with type comparison off, changing a column's type in `models.py` would produce an empty
    autogenerate diff and the test would pass while the schema silently disagreed with the
    ORM.

An in-memory SQLite URL is rejected rather than migrated: each connection to
``sqlite:///:memory:`` gets its own private database, so `alembic upgrade` would build a
schema into a database that is discarded when the command exits (defect D22, same root
cause). Tests migrate an in-memory database by passing a live connection through
``config.attributes["connection"]``, which is the supported way to do it.
"""

from __future__ import annotations

import os
import sys

from alembic import context
from sqlalchemy import engine_from_config, pool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.database.session import DATABASE_URL, Base  # noqa: E402
import src.database.models  # noqa: F401,E402  (registers the tables on Base.metadata)

config = context.config
target_metadata = Base.metadata


def _resolved_url() -> str:
    """The URL to migrate: the ini file wins only if it is non-empty."""
    from_ini = config.get_main_option("sqlalchemy.url", None)
    return from_ini or DATABASE_URL


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting - for review, or for a DBA-applied change."""
    context.configure(
        url=_resolved_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _configure_and_run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    # A caller (the test suite, or `src.database.migrate`) may hand us an open connection.
    # This is the only way to migrate an in-memory SQLite database, and it also lets a
    # migration run inside a transaction the caller controls.
    existing = config.attributes.get("connection", None)
    if existing is not None:
        _configure_and_run(existing)
        return

    url = _resolved_url()
    if ":memory:" in url:
        raise RuntimeError(
            "refusing to migrate %r: every connection to an in-memory SQLite database gets "
            "its own private copy, so the schema this command built would be discarded the "
            "moment it exits. Pass a live connection via "
            "config.attributes['connection'] instead." % url)

    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = url
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        _configure_and_run(connection)
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
