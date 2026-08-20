"""Programmatic schema migration (roadmap T3.5.8).

**What this replaces, and why the replacement is not cosmetic.** The schema was created by
`Base.metadata.create_all` in the FastAPI lifespan. `create_all` adds missing *tables*; it
does not alter existing ones. So the moment a slice added a column to an existing table -
slice 8 added `experiment_runs.seed` and `execution`, slice 9 added four statistics columns
to `hypotheses` - every database created before that slice became silently wrong: the ORM
mapped columns the file did not contain, and the first query raised
`no such column: experiment_runs.seed`. `create_all` reported success throughout. A platform
whose results are meant to be defendable cannot have a storage layer whose upgrade path is
"delete the file and lose the runs".

**Adoption, which is the part that is easy to get wrong.** An existing `spectral_earth.db`
has tables but no `alembic_version`, so Alembic considers it un-migrated and
`upgrade head` would try to `CREATE TABLE experiments` and fail. The usual fix is
`alembic stamp head`, and here that would be *wrong*: stamping head asserts that the five
slice-8/9 columns are present, which for a pre-slice-8 file they are not, and the migration
that would have added them is then skipped forever. `adopt_existing` therefore probes for
those columns and stamps **0001 or 0002 by observation**, never by assumption.

**Auto-upgrade is on by default and can be turned off.** For a laptop research tool,
starting the API and having the schema be correct is the right default. For a shared or
production deployment, schema changes should be a deliberate, reviewed step - set
``SPECTRALEARTH_AUTO_MIGRATE=0`` and `ensure_schema` will refuse to migrate, reporting
exactly which revisions are outstanding instead of applying them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Connection, Engine

from src.core.errors import SpectralEarthError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALEMBIC_INI = os.path.join(REPO_ROOT, "alembic.ini")
SCRIPT_LOCATION = os.path.join(REPO_ROOT, "migrations")

#: Revision that introduced the five columns of slices 8 and 9. Used by `adopt_existing`
#: to tell a pre-slice-8 database from a current one.
REVISION_WITH_RUN_SEED = "0002"

#: The probe columns. If `experiment_runs.seed` exists, the database is at 0002 or later.
_PROBE = ("experiment_runs", "seed")


class MigrationError(SpectralEarthError):
    """Raised when the schema cannot be brought to a known, correct state."""

    status_code = 500


def alembic_config(connection: Optional[Connection] = None,
                   url: Optional[str] = None) -> Config:
    """Build an Alembic `Config` pointing at this repository's migrations.

    Passing ``connection`` is how an in-memory SQLite database is migrated: `env.py` picks
    it up from ``config.attributes`` and runs against it rather than opening its own, which
    would get a different (private, empty) in-memory database.
    """
    config = Config(ALEMBIC_INI) if os.path.exists(ALEMBIC_INI) else Config()
    config.set_main_option("script_location", SCRIPT_LOCATION)
    if url is not None:
        config.set_main_option("sqlalchemy.url", url)
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def head_revision() -> str:
    """The newest revision defined in `migrations/versions/`."""
    script = ScriptDirectory.from_config(alembic_config())
    heads = script.get_heads()
    if len(heads) != 1:
        raise MigrationError(
            "the migration history has %d heads (%s). Alembic cannot decide which is "
            "current; the branches must be merged with `alembic merge` before the schema "
            "can be upgraded." % (len(heads), ", ".join(sorted(heads))))
    return heads[0]


def all_revisions() -> List[str]:
    """Every revision, oldest first."""
    script = ScriptDirectory.from_config(alembic_config())
    return [rev.revision for rev in reversed(list(script.walk_revisions()))]


def current_revision(bind: Engine | Connection) -> Optional[str]:
    """The revision stamped on this database, or ``None`` if it has never been migrated."""
    if isinstance(bind, Engine):
        with bind.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()
    return MigrationContext.configure(bind).get_current_revision()


def has_tables(bind: Engine | Connection) -> bool:
    """Whether the database contains any application table (ignoring `alembic_version`)."""
    names = set(inspect(bind).get_table_names())
    names.discard("alembic_version")
    return bool(names)


def pending(bind: Engine | Connection) -> List[str]:
    """Revisions not yet applied to this database, oldest first.

    An unrecognised stamped revision is an error, not an empty list: it means the database
    was migrated by a *newer* checkout than this one, and running against it would corrupt
    data through an ORM that does not match the schema.
    """
    current = current_revision(bind)
    revisions = all_revisions()
    if current is None:
        return list(revisions)
    if current not in revisions:
        raise MigrationError(
            "this database is stamped with revision %r, which does not exist in "
            "migrations/versions/. It was almost certainly migrated by a newer checkout "
            "of the platform. Update the code rather than the database; running an older "
            "ORM against a newer schema silently reads and writes the wrong columns."
            % current, current_revision=current, known=revisions)
    index = revisions.index(current)
    return revisions[index + 1:]


# --------------------------------------------------------------------------- operations

def upgrade(bind: Engine | Connection, revision: str = "head") -> Dict[str, Any]:
    """Apply migrations up to ``revision`` (default: the head)."""
    before = current_revision(bind)
    outstanding = pending(bind)
    if isinstance(bind, Engine):
        with bind.begin() as conn:
            command.upgrade(alembic_config(connection=conn), revision)
    else:
        command.upgrade(alembic_config(connection=bind), revision)
    after = current_revision(bind)
    return {"action": "upgrade", "from": before, "to": after, "applied": outstanding}


def downgrade(bind: Engine | Connection, revision: str) -> Dict[str, Any]:
    """Revert migrations down to ``revision`` (``"base"`` removes the schema entirely)."""
    before = current_revision(bind)
    if isinstance(bind, Engine):
        with bind.begin() as conn:
            command.downgrade(alembic_config(connection=conn), revision)
    else:
        command.downgrade(alembic_config(connection=bind), revision)
    return {"action": "downgrade", "from": before, "to": current_revision(bind)}


def stamp(bind: Engine | Connection, revision: str) -> Dict[str, Any]:
    """Record ``revision`` as applied **without running it**.

    Only correct when the schema already matches that revision - which is why
    `adopt_existing` establishes that by inspection first.
    """
    if isinstance(bind, Engine):
        with bind.begin() as conn:
            command.stamp(alembic_config(connection=conn), revision)
    else:
        command.stamp(alembic_config(connection=bind), revision)
    return {"action": "stamp", "to": current_revision(bind)}


def detect_legacy_revision(bind: Engine | Connection) -> Optional[str]:
    """Which revision an unstamped, already-populated database actually corresponds to.

    Returns ``None`` for an empty database (nothing to adopt - migrate it normally).
    Otherwise returns ``"0002"`` if the slice-8/9 columns are present and ``"0001"`` if
    they are not, so the correct migrations still run afterwards.
    """
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    tables.discard("alembic_version")
    if not tables:
        return None
    table, column = _PROBE
    if table not in tables:
        raise MigrationError(
            "this database has tables (%s) but not %r, so it does not match any revision "
            "in the migration history and cannot be adopted automatically. Inspect it by "
            "hand: back it up, then either migrate the data into a fresh database or write "
            "a migration that describes what it actually contains."
            % (", ".join(sorted(tables)), table), tables=sorted(tables))
    columns = {c["name"] for c in inspector.get_columns(table)}
    return REVISION_WITH_RUN_SEED if column in columns else "0001"


def adopt_existing(bind: Engine | Connection) -> Dict[str, Any]:
    """Bring an un-stamped, pre-Alembic database under migration control.

    Stamps the revision the schema *is*, determined by inspection, then upgrades to head.
    """
    detected = detect_legacy_revision(bind)
    if detected is None:
        return {"action": "none", "reason": "database is empty; nothing to adopt"}
    stamp(bind, detected)
    result = upgrade(bind)
    result["adopted_as"] = detected
    return result


def auto_migrate_enabled() -> bool:
    """Whether `ensure_schema` may apply migrations itself."""
    return os.getenv("SPECTRALEARTH_AUTO_MIGRATE", "1").strip().lower() not in (
        "0", "false", "no", "off")


def ensure_schema(bind: Engine | Connection) -> Dict[str, Any]:
    """Make the schema current, or explain precisely why it is not. The startup path.

    Four cases, each with a different correct action:

    1. **Empty database** - run every migration.
    2. **Populated but never stamped** - a pre-Alembic database: adopt it by inspection,
       then upgrade.
    3. **Stamped and behind** - upgrade, or if ``SPECTRALEARTH_AUTO_MIGRATE=0``, refuse and
       name the outstanding revisions.
    4. **Stamped and current** - do nothing, and say so.
    """
    stamped = current_revision(bind)
    populated = has_tables(bind)

    if stamped is None and populated:
        if not auto_migrate_enabled():
            detected = detect_legacy_revision(bind)
            raise MigrationError(
                "this database predates migration control (it has tables but no "
                "alembic_version) and automatic migration is disabled. Its schema matches "
                "revision %s. Run `python -m src.database.migrate adopt` to bring it under "
                "control, or unset SPECTRALEARTH_AUTO_MIGRATE." % detected,
                detected=detected)
        result = adopt_existing(bind)
        result["case"] = "adopted_pre_alembic_database"
        return result

    outstanding = pending(bind)
    if not outstanding:
        return {"action": "none", "case": "already_current", "revision": stamped,
                "head": head_revision()}

    if not auto_migrate_enabled():
        raise MigrationError(
            "the database is at revision %s but the code expects %s; %d migration(s) are "
            "outstanding (%s) and automatic migration is disabled. Apply them with "
            "`alembic upgrade head`. Serving requests against a stale schema would fail on "
            "the first query touching a new column, which is a worse outcome than refusing "
            "to start."
            % (stamped or "base", head_revision(), len(outstanding),
               ", ".join(outstanding)),
            current_revision=stamped, head=head_revision(), pending=outstanding)

    result = upgrade(bind)
    result["case"] = "empty_database" if stamped is None else "upgraded_existing"
    return result


def describe(bind: Engine | Connection) -> Dict[str, Any]:
    """Schema state for the health endpoint and for run provenance.

    A result is only reproducible if the schema that stored it is identifiable, so the
    stamped revision belongs in provenance alongside the seed and the code revision.
    """
    try:
        stamped = current_revision(bind)
        outstanding = pending(bind)
        error = None
    except Exception as exc:  # an unknown stamp must be reported, not hidden
        stamped, outstanding, error = None, None, str(exc)
    return {
        "revision": stamped,
        "head": head_revision(),
        "pending": outstanding,
        "up_to_date": (outstanding == []) if outstanding is not None else None,
        "auto_migrate": auto_migrate_enabled(),
        "error": error,
    }


def _main(argv: Optional[List[str]] = None) -> int:
    """`python -m src.database.migrate {status|upgrade|adopt|downgrade <rev>}`."""
    import json
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    action = args[0] if args else "status"
    from src.database.session import engine

    if action == "status":
        print(json.dumps(describe(engine), indent=2))
    elif action == "upgrade":
        print(json.dumps(upgrade(engine, args[1] if len(args) > 1 else "head"), indent=2))
    elif action == "adopt":
        print(json.dumps(adopt_existing(engine), indent=2))
    elif action == "downgrade":
        if len(args) < 2:
            print("downgrade needs a target revision (or 'base')", file=sys.stderr)
            return 2
        print(json.dumps(downgrade(engine, args[1]), indent=2))
    else:
        print("unknown action %r; expected status, upgrade, adopt or downgrade"
              % action, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
