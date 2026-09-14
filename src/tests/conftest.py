"""Shared pytest configuration.

Fixes two collection/isolation problems (defects D4 and D22):

D4  There is no ``__init__.py`` anywhere in ``src/`` and no packaging metadata, so
    ``pytest`` invoked as a bare command cannot resolve ``from src.… import …``.
    Inserting the repository root on ``sys.path`` here makes bare ``pytest`` work
    identically to ``python -m pytest``.

D22 ``sqlite:///:memory:`` gives each *connection* its own private database. SQLAlchemy's
    default pool for in-memory SQLite hands out a fresh connection to each thread, and
    Starlette's ``TestClient`` runs the application in a different thread from the test
    body — so tables created by the fixture were invisible to the endpoint under test,
    surfacing as ``no such table: experiments``. ``StaticPool`` forces one shared
    connection, which is what makes an in-memory test database usable at all.
"""
import os
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Test behavior must not depend on a developer's ignored machine-local credentials or network
# opt-in. The loader has its own isolated mapping tests in test_api_infrastructure.py.
os.environ.setdefault("SPECTRALEARTH_LOAD_LOCAL_ENV", "0")

from src.database.session import Base, get_db  # noqa: E402


@pytest.fixture(scope="function")
def db_engine():
    """A single-connection in-memory SQLite engine, visible across threads."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(scope="function")
def session_factory(db_engine):
    """Session factory bound to the test engine, for injection into the engine layer."""
    return sessionmaker(autocommit=False, autoflush=False, bind=db_engine)


@pytest.fixture(scope="function")
def db_session(session_factory):
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client(db_engine, db_session, session_factory, tmp_path):
    """TestClient whose get_db dependency is bound to the test database.

    Entered as a context manager so the FastAPI lifespan actually runs; without this
    the app's own startup hook never fires.
    """
    from fastapi.testclient import TestClient

    from src.api.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    # D24: BackgroundTasks outlive the request session, so the engine layer resolves its
    # own factory from app.state. Bind it to the test database or the sweep writes to
    # the real spectral_earth.db (or fails, if that file has no tables).
    app.state.session_factory = session_factory
    # The same reasoning as D24, applied to the filesystem. Routes that persist - saved manifest
    # revisions, and since TG17.6 the run journals - fall back to `data/` when nothing binds them,
    # so an unbound test writes into the deployed data directory. For runs that is worse than
    # untidy: a run identity is the content address of its manifest, so a test posting a manifest
    # would resume, and then advance, whatever real run that manifest already had.
    app.state.experiment_run_dir = str(tmp_path / 'experiment_runs')
    app.state.experiment_manifest_dir = str(tmp_path / 'experiment_manifests')
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    app.state.session_factory = None
    app.state.experiment_run_dir = None
    app.state.experiment_manifest_dir = None
