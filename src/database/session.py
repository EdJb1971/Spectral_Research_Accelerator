"""Database engine and session factory.

SQLite is configured for concurrent access (roadmap T3.5.19, standard E11). Two settings
matter and neither is the default:

*   **WAL journal mode.** In the default rollback-journal mode a writer blocks all readers.
    With write-ahead logging readers and one writer proceed concurrently, which is what a
    sweep that computes in workers and persists in the parent needs.
*   **busy_timeout.** Without it, a lock contention raises `database is locked` *immediately*
    rather than waiting. That single setting is the difference between a sweep that works
    under parallelism and one that fails intermittently in a way that looks like a bug in
    the science code.

Both are applied per-connection via a `connect` event, because SQLite pragmas are
connection-scoped: setting them once on the engine would leave every pooled connection after
the first with the defaults.
"""

import os

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./spectral_earth.db")

#: How long a writer waits for a lock before giving up, in milliseconds.
SQLITE_BUSY_TIMEOUT_MS = int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "30000"))

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)


if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _configure_sqlite(dbapi_connection, connection_record):
        """Apply the concurrency pragmas to every new connection."""
        cursor = dbapi_connection.cursor()
        try:
            # An in-memory database has a single connection and does not support WAL;
            # attempting it is harmless but pointless, so it is skipped rather than
            # producing a confusing warning in every test run.
            if ":memory:" not in DATABASE_URL:
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=%d" % SQLITE_BUSY_TIMEOUT_MS)
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def sqlite_settings() -> dict:
    """Report the pragmas actually in force, for the health endpoint and provenance."""
    if not DATABASE_URL.startswith("sqlite"):
        return {"backend": DATABASE_URL.split(":")[0]}
    with engine.connect() as conn:
        from sqlalchemy import text
        return {
            "backend": "sqlite",
            "journal_mode": conn.execute(text("PRAGMA journal_mode")).scalar(),
            "busy_timeout_ms": conn.execute(text("PRAGMA busy_timeout")).scalar(),
            "synchronous": conn.execute(text("PRAGMA synchronous")).scalar(),
        }
