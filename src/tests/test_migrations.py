"""Schema migration tests (roadmap T3.5.8).

The acceptance criterion is ``alembic upgrade head`` building the schema from empty, but that
alone is a weak test: it passes for a migration history that has quietly drifted away from
`models.py`, which is the failure this whole slice exists to prevent. The load-bearing test
here is `test_no_drift_against_orm`, which asks Alembic's own autogenerate machinery whether
the migrated schema and the ORM disagree, and fails on any difference in tables, columns,
types, nullability or indexes.

`test_create_all_ignores_missing_columns` documents the defect itself, in code, so the
reason for the migration layer cannot be lost.
"""

from __future__ import annotations

import io
import os

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

from src.database import migrate
from src.database.session import Base
import src.database.models as models  # noqa: F401  (registers tables on Base.metadata)


# --------------------------------------------------------------------------- fixtures

def _engine():
    """A single-connection in-memory engine.

    StaticPool is mandatory, not tidiness: each connection to ``sqlite:///:memory:`` gets a
    private database, so a migration applied on one connection would be invisible on the
    next and every assertion here would test an empty database (defect D22).
    """
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture()
def empty_engine():
    eng = _engine()
    try:
        yield eng
    finally:
        eng.dispose()


def _drift(bind):
    """Differences between the live schema and the ORM, as Alembic sees them."""
    with bind.connect() as conn:
        context = MigrationContext.configure(
            conn, opts={"compare_type": True, "compare_server_default": True})
        return compare_metadata(context, Base.metadata)


def _columns(bind, table):
    return {c["name"] for c in inspect(bind).get_columns(table)}


# --------------------------------------------------------------------------- history

def test_single_head():
    """A branched history would make `upgrade head` ambiguous."""
    assert migrate.head_revision() == "0002"


def test_revisions_are_ordered_and_complete():
    assert migrate.all_revisions() == ["0001", "0002"]


def test_every_version_file_is_reachable_from_head():
    """A stray file in versions/ that no revision points at would never run."""
    import glob

    files = glob.glob(os.path.join(migrate.SCRIPT_LOCATION, "versions", "0*.py"))
    assert len(files) == len(migrate.all_revisions())


# --------------------------------------------------------------------------- upgrade

def test_upgrade_from_empty(empty_engine):
    """The stated acceptance criterion: `upgrade head` builds the schema from nothing."""
    result = migrate.upgrade(empty_engine)
    assert result["applied"] == ["0001", "0002"]
    assert migrate.current_revision(empty_engine) == "0002"
    tables = set(inspect(empty_engine).get_table_names())
    assert {"experiments", "experiment_runs", "lineage_nodes", "lineage_edges",
            "hypotheses"} <= tables


def test_no_drift_against_orm(empty_engine):
    """The migrated schema and `models.py` must be indistinguishable.

    This is the test that keeps the migration history honest. Adding a column to a model
    without writing a migration fails here, which is the whole point: the alternative is
    discovering it as `no such column` in production.
    """
    migrate.upgrade(empty_engine)
    assert _drift(empty_engine) == []


def test_migrated_schema_matches_create_all(empty_engine):
    """`upgrade head` and `create_all` must produce the same tables and columns.

    Compared explicitly as well as via autogenerate, because the two mechanisms fail
    differently and a divergence between them is what makes a test suite that uses
    `create_all` (as conftest does) unable to catch a migration bug.
    """
    migrate.upgrade(empty_engine)
    migrated = {t: _columns(empty_engine, t) for t in inspect(empty_engine).get_table_names()
                if t != "alembic_version"}

    reference = _engine()
    try:
        Base.metadata.create_all(bind=reference)
        expected = {t: _columns(reference, t)
                    for t in inspect(reference).get_table_names()}
    finally:
        reference.dispose()

    assert migrated == expected


def test_indexes_match_create_all(empty_engine):
    """Index names too: `create_all` derives them from `index=True`, 0001 spells them out."""
    migrate.upgrade(empty_engine)
    migrated = {i["name"] for t in inspect(empty_engine).get_table_names()
                for i in inspect(empty_engine).get_indexes(t)}
    reference = _engine()
    try:
        Base.metadata.create_all(bind=reference)
        expected = {i["name"] for t in inspect(reference).get_table_names()
                    for i in inspect(reference).get_indexes(t)}
    finally:
        reference.dispose()
    assert migrated == expected


def test_upgrade_is_idempotent(empty_engine):
    migrate.upgrade(empty_engine)
    second = migrate.ensure_schema(empty_engine)
    assert second["action"] == "none"
    assert second["case"] == "already_current"


# --------------------------------------------------------------------------- downgrade

def test_downgrade_to_base_removes_every_table(empty_engine):
    migrate.upgrade(empty_engine)
    migrate.downgrade(empty_engine, "base")
    remaining = set(inspect(empty_engine).get_table_names()) - {"alembic_version"}
    assert remaining == set()
    assert migrate.current_revision(empty_engine) is None


def test_each_revision_round_trips(empty_engine):
    """Step up one revision at a time, then back down, checking every intermediate state.

    A migration history is only reversible if *each* step is; testing only the extremes
    would miss a broken `downgrade()` in the middle, which is exactly the one a rollback
    would need.
    """
    revisions = migrate.all_revisions()
    for rev in revisions:
        migrate.upgrade(empty_engine, rev)
        assert migrate.current_revision(empty_engine) == rev
    # Only the head is expected to match the ORM; an intermediate revision legitimately
    # differs from it, and asserting otherwise would be asserting nothing.
    assert _drift(empty_engine) == []
    for rev in list(reversed(revisions))[1:] + ["base"]:
        migrate.downgrade(empty_engine, rev)
        expected = None if rev == "base" else rev
        assert migrate.current_revision(empty_engine) == expected


def test_downgrade_0002_removes_exactly_the_five_columns(empty_engine):
    migrate.upgrade(empty_engine)
    assert {"seed", "execution"} <= _columns(empty_engine, "experiment_runs")
    migrate.downgrade(empty_engine, "0001")
    assert not {"seed", "execution"} & _columns(empty_engine, "experiment_runs")
    assert not ({"p_value", "q_value", "n_tests", "statistics"}
                & _columns(empty_engine, "hypotheses"))
    # Everything else must survive: batch mode rebuilds the table, so a mistake here
    # silently drops columns rather than erroring.
    assert {"id", "experiment_id", "parameters", "status", "results", "error_message",
            "created_at", "completed_at"} == _columns(empty_engine, "experiment_runs")


def test_downgrade_0002_preserves_rows(empty_engine):
    """Batch-mode DROP COLUMN rebuilds the table; the data must come with it."""
    migrate.upgrade(empty_engine)
    with empty_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO experiments (id, name, config) VALUES ('e1', 'n', '{}')"))
        conn.execute(text(
            "INSERT INTO experiment_runs (id, experiment_id, parameters, seed) "
            "VALUES ('r1', 'e1', '{}', 42)"))
    migrate.downgrade(empty_engine, "0001")
    with empty_engine.connect() as conn:
        rows = conn.execute(text("SELECT id, experiment_id FROM experiment_runs")).all()
    assert rows == [("r1", "e1")]


# --------------------------------------------------------------------------- the defect

def test_create_all_ignores_missing_columns(empty_engine):
    """The defect this slice closes, asserted rather than described.

    `create_all` adds missing *tables* only. Against a database at revision 0001 it makes no
    change at all and reports success, leaving the ORM mapping five columns that do not
    exist - which is precisely the state the real `spectral_earth.db` was found in
    (`no such column: experiment_runs.seed`).
    """
    migrate.upgrade(empty_engine, "0001")
    Base.metadata.create_all(bind=empty_engine)
    assert "seed" not in _columns(empty_engine, "experiment_runs")
    assert "q_value" not in _columns(empty_engine, "hypotheses")
    # And the migration is what fixes it.
    migrate.upgrade(empty_engine)
    assert "seed" in _columns(empty_engine, "experiment_runs")
    assert "q_value" in _columns(empty_engine, "hypotheses")


# --------------------------------------------------------------------------- adoption

def test_adopt_pre_alembic_database_preserves_rows(empty_engine):
    """A populated, unstamped, pre-slice-8 database is adopted at 0001 and then upgraded.

    Stamping *head* here would be the natural mistake and would be wrong: it would assert
    the five columns exist, skip 0002 forever, and leave a database Alembic believes is
    current and the ORM cannot query.
    """
    migrate.upgrade(empty_engine, "0001")
    with empty_engine.begin() as conn:
        conn.execute(text("DROP TABLE alembic_version"))
        conn.execute(text(
            "INSERT INTO experiments (id, name, config) VALUES ('e1', 'legacy', '{}')"))
        conn.execute(text(
            "INSERT INTO experiment_runs (id, experiment_id, parameters) "
            "VALUES ('r1', 'e1', '{}')"))
    assert migrate.current_revision(empty_engine) is None
    assert migrate.detect_legacy_revision(empty_engine) == "0001"

    result = migrate.ensure_schema(empty_engine)
    assert result["adopted_as"] == "0001"
    assert result["applied"] == ["0002"]
    assert migrate.current_revision(empty_engine) == "0002"
    assert _drift(empty_engine) == []

    with empty_engine.connect() as conn:
        rows = conn.execute(text("SELECT id, seed FROM experiment_runs")).all()
    # The pre-existing run survives and its seed is NULL - the honest value for a run
    # recorded before seeds were captured. A default of 0 would claim a reproducibility
    # that does not exist.
    assert rows == [("r1", None)]


def test_detect_legacy_revision_on_current_create_all_database(empty_engine):
    """A database built by today's `create_all` already matches 0002 and is stamped there."""
    Base.metadata.create_all(bind=empty_engine)
    assert migrate.detect_legacy_revision(empty_engine) == "0002"
    result = migrate.ensure_schema(empty_engine)
    assert result["adopted_as"] == "0002"
    assert result["applied"] == []
    assert _drift(empty_engine) == []


def test_detect_legacy_revision_on_empty_database(empty_engine):
    assert migrate.detect_legacy_revision(empty_engine) is None
    assert migrate.has_tables(empty_engine) is False


def test_unrecognisable_database_is_not_adopted(empty_engine):
    """Tables present but not ours: refuse, and say what to do instead."""
    with empty_engine.begin() as conn:
        conn.execute(text("CREATE TABLE something_else (id INTEGER)"))
    with pytest.raises(migrate.MigrationError) as excinfo:
        migrate.detect_legacy_revision(empty_engine)
    assert "something_else" in str(excinfo.value)
    assert "back it up" in str(excinfo.value)


# --------------------------------------------------------------------------- guard rails

def test_unknown_stamped_revision_is_an_error(empty_engine):
    """A database migrated by a newer checkout must stop us, not be silently ignored."""
    migrate.upgrade(empty_engine)
    with empty_engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = '9999'"))
    with pytest.raises(migrate.MigrationError) as excinfo:
        migrate.pending(empty_engine)
    message = str(excinfo.value)
    assert "9999" in message
    assert "newer checkout" in message


def test_auto_migrate_disabled_refuses_and_names_pending(empty_engine, monkeypatch):
    migrate.upgrade(empty_engine, "0001")
    monkeypatch.setenv("SPECTRALEARTH_AUTO_MIGRATE", "0")
    assert migrate.auto_migrate_enabled() is False
    with pytest.raises(migrate.MigrationError) as excinfo:
        migrate.ensure_schema(empty_engine)
    message = str(excinfo.value)
    assert "0002" in message
    assert "alembic upgrade head" in message
    # And it must not have migrated anyway.
    assert migrate.current_revision(empty_engine) == "0001"


def test_auto_migrate_disabled_refuses_to_adopt(empty_engine, monkeypatch):
    migrate.upgrade(empty_engine, "0001")
    with empty_engine.begin() as conn:
        conn.execute(text("DROP TABLE alembic_version"))
    monkeypatch.setenv("SPECTRALEARTH_AUTO_MIGRATE", "0")
    with pytest.raises(migrate.MigrationError) as excinfo:
        migrate.ensure_schema(empty_engine)
    assert "predates migration control" in str(excinfo.value)


def test_in_memory_url_is_refused_by_env(monkeypatch):
    """`alembic upgrade` against an in-memory URL would build a schema and discard it."""
    config = migrate.alembic_config(url="sqlite:///:memory:")
    with pytest.raises(RuntimeError) as excinfo:
        command.upgrade(config, "head")
    assert "private copy" in str(excinfo.value)


def test_describe_reports_state(empty_engine):
    migrate.upgrade(empty_engine, "0001")
    state = migrate.describe(empty_engine)
    assert state["revision"] == "0001"
    assert state["head"] == "0002"
    assert state["pending"] == ["0002"]
    assert state["up_to_date"] is False
    assert state["error"] is None


def test_describe_surfaces_an_unknown_revision(empty_engine):
    """A broken state must be reported by the health probe, not swallowed into 'ok'."""
    migrate.upgrade(empty_engine)
    with empty_engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = '9999'"))
    state = migrate.describe(empty_engine)
    assert state["up_to_date"] is None
    assert "9999" in state["error"]


# --------------------------------------------------------------------------- portability

@pytest.mark.parametrize("dialect_url", [
    "postgresql://user:pass@localhost/spectralearth",
    "sqlite:///./offline.db",
])
def test_migrations_render_as_sql_for_both_backends(dialect_url):
    """Offline mode renders the DDL for a dialect without needing a server.

    **What this does and does not show.** It proves the scripts *compile* for PostgreSQL -
    no SQLite-only construct, no unrenderable type - which is the half of the acceptance
    criterion that can be checked on a laptop with no PostgreSQL installed. It does not
    prove the DDL executes there. That distinction is stated rather than glossed: the
    roadmap's acceptance criterion names both backends, and only one of them has been run.
    """
    config = migrate.alembic_config(url=dialect_url)
    buffer = io.StringIO()
    # Both, deliberately: `stdout` is where Config.print_stdout writes, `output_buffer` is
    # where the offline MigrationContext writes the DDL. Setting only the first captures the
    # commentary and lets the SQL escape to the terminal, which is how this test first
    # passed nothing to its own assertions.
    config.stdout = buffer
    config.output_buffer = buffer
    command.upgrade(config, "head", sql=True)
    sql = buffer.getvalue()
    assert "CREATE TABLE experiments" in sql
    assert "CREATE TABLE experiment_runs" in sql
    for column in ("seed", "execution", "p_value", "q_value", "n_tests", "statistics"):
        assert column in sql, "revision 0002 did not render column %r" % column


def test_health_endpoint_reports_the_schema_revision(client):
    """Schema state belongs in the readiness probe: 'behind' is a real failure mode."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    state = response.json()["schema_state"]
    assert state["head"] == migrate.head_revision()
    assert state["revision"] == migrate.head_revision()
    assert state["pending"] == []
    assert state["up_to_date"] is True
