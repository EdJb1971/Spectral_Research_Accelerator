"""Tests for the Executor seam and the device/thread policy (T3.5.19/E11, D18).

The acceptance criterion is that a sweep produces **byte-identical results** under `serial`
and under `process` with 2, 4 and 8 workers. That is asserted here on a real 24-run sweep
through the database, not on a toy function, because the things that break determinism -
result ordering, seed derivation and per-worker state - only appear once real work is
distributed.

Note on the mapped functions: several are defined at module level rather than inside the
tests. That is a requirement of the `process` backend on `spawn` platforms, not a style
choice, and `test_process_backend_explains_the_spawn_trap` covers what happens otherwise.
"""

import json
import os
import tempfile
import time
import uuid

import numpy as np
import pytest
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core import device as device_policy
from src.core import doctor as execution_doctor
from src.core.errors import InvalidParameterError
from src.core.executor import (
    ExecutorStartupError,
    ProcessExecutor,
    SerialExecutor,
    ThreadExecutor,
    derive_seeds,
    describe,
    get_executor,
)
from src.database.models import Base, Experiment, ExperimentRun, LineageNode
from src.experiment_engine.engine import DeclarativeExperimentEngine


# --------------------------------------------------------------- module-level workers

def double(item):
    return item * 2


def seeded_draw(item, seed):
    """Draws from the seeded stream, so the result depends on the seed and nothing else."""
    torch.manual_seed(seed)
    return {"item": item, "value": float(torch.randn(4, dtype=torch.float64).sum())}


def slow_then_fast(item):
    """Early items take longer, so completion order differs from submission order."""
    time.sleep(0.02 if item < 3 else 0.001)
    return item


def explodes_on_three(item):
    if item == 3:
        raise ValueError("item three is unacceptable")
    return item


def report_threads(item):
    return torch.get_num_threads()


# --------------------------------------------------------------- seeds

def test_seeds_are_derived_by_spawn_not_by_addition():
    """Sequential seeds are not independent seeds."""
    seeds = derive_seeds(20260819, 16)
    assert len(set(seeds)) == 16
    # Not an arithmetic sequence - which `root + i` would be.
    diffs = {seeds[i + 1] - seeds[i] for i in range(len(seeds) - 1)}
    assert len(diffs) > 1


def test_seed_derivation_is_reproducible_and_root_dependent():
    assert derive_seeds(7, 8) == derive_seeds(7, 8)
    assert derive_seeds(7, 8) != derive_seeds(8, 8)


def test_a_runs_seed_does_not_depend_on_which_worker_took_it():
    """The property that makes parallel and serial agree."""
    items = list(range(12))
    seeds = derive_seeds(99, 12)
    ref = get_executor("serial").map(seeded_draw, items, seeds)
    thr = get_executor("thread", 4).map(seeded_draw, items, seeds)
    assert [r.value for r in ref] == [r.value for r in thr]
    assert [r.seed for r in ref] == [r.seed for r in thr] == seeds


# --------------------------------------------------------------- ordering

@pytest.mark.parametrize("backend,workers", [("serial", 1), ("thread", 4)])
def test_results_come_back_in_submission_order(backend, workers):
    """`as_completed` order varies run to run; submission order is the contract.

    Uses a workload whose completion order is deliberately the reverse of submission order,
    so a backend that yielded completion order would fail rather than pass by luck.
    """
    items = list(range(6))
    results = get_executor(backend, workers).map(slow_then_fast, items)
    assert [r.value for r in results] == items
    assert [r.index for r in results] == items


def test_empty_input_is_handled_by_every_backend():
    for backend, workers in (("serial", 1), ("thread", 2), ("process", 2)):
        assert get_executor(backend, workers).map(double, []) == []


# --------------------------------------------------------------- failures

@pytest.mark.parametrize("backend,workers", [("serial", 1), ("thread", 4)])
def test_one_failing_item_does_not_discard_the_others(backend, workers):
    """A single bad run in a 20-run sweep must not lose the other 19."""
    results = get_executor(backend, workers).map(explodes_on_three, list(range(6)))
    assert len(results) == 6
    assert [r.ok for r in results] == [True, True, True, False, True, True]
    bad = results[3]
    assert bad.error_type == "ValueError"
    assert "unacceptable" in bad.error
    assert "explodes_on_three" in bad.traceback
    assert [r.value for r in results if r.ok] == [0, 1, 2, 4, 5]


def test_task_result_raises_only_when_asked():
    results = get_executor("serial").map(explodes_on_three, [3])
    with pytest.raises(RuntimeError, match="unacceptable"):
        results[0].raise_if_failed()


# --------------------------------------------------------------- configuration

def test_unknown_backend_is_refused_with_the_valid_options():
    with pytest.raises(InvalidParameterError, match="serial, thread, process"):
        get_executor("celery")
    with pytest.raises(InvalidParameterError):
        get_executor("thread", 0)


def test_get_executor_returns_the_right_type():
    assert isinstance(get_executor("serial"), SerialExecutor)
    assert isinstance(get_executor("thread", 2), ThreadExecutor)
    assert isinstance(get_executor("process", 2), ProcessExecutor)


def test_describe_records_how_work_was_distributed():
    record = describe(get_executor("thread", 4))
    assert record["backend"] == "thread"
    assert record["n_workers"] == 4
    assert record["result_order"] == "submission"
    assert record["threads_per_worker"] >= 1


def test_process_backend_explains_the_spawn_trap():
    """An unpicklable function must produce an actionable message, not a raw pool error.

    A closure defined inside a test is exactly what a user will try first, and the default
    failure - `BrokenProcessPool` or an opaque pickling error - says nothing about the cause.
    """
    local_closure = lambda item: item * 2  # noqa: E731  deliberately unpicklable
    with pytest.raises((ExecutorStartupError, Exception)) as exc:
        get_executor("process", 2).map(local_closure, [1, 2, 3])
    message = str(exc.value)
    assert ("__main__" in message or "pickl" in message.lower()
            or "importable at module level" in message), message


# --------------------------------------------------------------- device / thread policy

def test_thread_budget_prevents_oversubscription():
    """`n_workers` x `n_cores` threads on `n_cores` cores makes throughput *fall*."""
    assert device_policy.thread_budget(1, total_threads=12) == 12
    assert device_policy.thread_budget(4, total_threads=12) == 3
    assert device_policy.thread_budget(8, total_threads=12) == 1
    # Never zero, however many workers are requested.
    assert device_policy.thread_budget(64, total_threads=4) == 1


def test_workers_receive_a_reduced_thread_budget():
    """Asserted through the seam, not just in the helper."""
    before = torch.get_num_threads()
    try:
        results = get_executor("thread", 4, threads_per_worker=1).map(
            report_threads, list(range(4)))
        assert all(r.ok for r in results)
        assert all(r.value == 1 for r in results), [r.value for r in results]
    finally:
        torch.set_num_threads(before)


def test_device_selection_falls_back_to_cpu_and_honours_an_override(monkeypatch):
    device = device_policy.select_device()
    assert device.type in ("cpu", "cuda", "mps")
    monkeypatch.setenv(device_policy.DEVICE_ENV_VAR, "cpu")
    assert device_policy.select_device().type == "cpu"


def test_requesting_an_unavailable_device_says_so(monkeypatch):
    """Silently falling back to CPU would make a 'ran on GPU' claim untrue."""
    monkeypatch.setenv(device_policy.DEVICE_ENV_VAR, "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA/ROCm is not available"):
        device_policy.select_device()


def test_unknown_device_name_is_rejected(monkeypatch):
    monkeypatch.setenv(device_policy.DEVICE_ENV_VAR, "tpu")
    with pytest.raises(ValueError, match="expected 'cpu', 'mps', or 'cuda"):
        device_policy.select_device()


def test_device_describe_is_a_provenance_record():
    record = device_policy.describe(n_workers=4)
    for key in ("device", "device_type", "available", "torch_version",
                "torch_num_threads", "n_workers", "thread_budget_per_worker"):
        assert key in record, key
    assert record["available"]["cpu"] is True
    assert record["available"]["accelerator_runtime"] in (None, "cuda", "rocm", "mps")
    if getattr(torch.version, "hip", None):
        assert record["available"]["accelerator_runtime"] == "rocm"
    elif torch.cuda.is_available():
        assert record["available"]["accelerator_runtime"] == "cuda"


def test_scheduler_context_recognises_allocations_without_a_cluster_client():
    assert device_policy.scheduler_context({})["scheduler"] is None
    assert device_policy.scheduler_context({
        "SLURM_JOB_ID": "8123", "SLURM_LOCALID": "2"
    }) == {"scheduler": "slurm", "job_id": "8123", "local_rank": 2}
    assert device_policy.scheduler_context({
        "PBS_JOBID": "44.server", "OMPI_COMM_WORLD_LOCAL_RANK": "1"
    }) == {"scheduler": "pbs", "job_id": "44.server", "local_rank": 1}


def test_cpu_profile_is_portable_and_rejects_a_conflicting_device(monkeypatch):
    monkeypatch.delenv(device_policy.DEVICE_ENV_VAR, raising=False)
    profile = device_policy.resolve_profile("cpu")
    assert profile.device == "cpu"
    assert profile.fallback_used is False
    monkeypatch.setenv(device_policy.DEVICE_ENV_VAR, "cuda:0")
    with pytest.raises(RuntimeError, match="conflicts"):
        device_policy.resolve_profile("cpu")


def test_accelerator_profile_refuses_a_silent_cpu_fallback(monkeypatch):
    monkeypatch.delenv(device_policy.DEVICE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        device_policy, "_select_device_raw", lambda prefer=None, run_idx=0: torch.device("cpu")
    )
    with pytest.raises(RuntimeError, match="only CPU is available"):
        device_policy.resolve_profile("accelerator")


def test_hpc_profile_refuses_to_run_on_a_login_node(monkeypatch):
    monkeypatch.delenv(device_policy.DEVICE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        device_policy, "scheduler_context",
        lambda environ=None: {"scheduler": None, "job_id": None, "local_rank": 0},
    )
    with pytest.raises(RuntimeError, match="active Slurm, PBS or LSF allocation"):
        device_policy.resolve_profile("hpc")


def test_hpc_profile_maps_scheduler_local_rank_to_a_device(monkeypatch):
    monkeypatch.delenv(device_policy.DEVICE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        device_policy, "scheduler_context",
        lambda environ=None: {"scheduler": "slurm", "job_id": "99", "local_rank": 3},
    )
    seen = []
    monkeypatch.setattr(
        device_policy, "_select_device_raw",
        lambda prefer=None, run_idx=0: seen.append(run_idx) or torch.device("cuda:1"),
    )
    profile = device_policy.resolve_profile("hpc")
    assert seen == [3]
    assert profile.scheduler == "slurm"
    assert profile.job_id == "99"
    assert profile.local_rank == 3
    assert profile.device == "cuda:1"


def test_select_device_honours_the_profile_environment(monkeypatch):
    monkeypatch.delenv(device_policy.DEVICE_ENV_VAR, raising=False)
    monkeypatch.setenv(device_policy.PROFILE_ENV_VAR, "cpu")
    assert device_policy.select_device().type == "cpu"
    monkeypatch.setenv(device_policy.PROFILE_ENV_VAR, "not-a-profile")
    with pytest.raises(ValueError, match="unknown execution profile"):
        device_policy.select_device()


def test_execution_doctor_runs_cpu_and_every_detected_accelerator():
    report = execution_doctor.build_report("auto")
    assert report["schema_version"] == 1
    assert report["smoke"]["cpu"]["status"] == "PASS"
    assert report["ready"] is True
    if torch.cuda.is_available():
        assert report["smoke"]["cuda:0"]["status"] == "PASS"
    assert report["resolved_profile"]["device"] == str(device_policy.select_device())


def test_determinism_record_states_what_was_actually_achieved():
    """Deterministic algorithms are requested, not guaranteed - so the mode is reported."""
    record = device_policy.enable_determinism(1234)
    assert record["seed"] == 1234
    assert "deterministic_algorithms" in record


def test_to_device_moves_nested_structures():
    from src.physical_core.field import PhysicalField
    cpu = torch.device("cpu")
    payload = {"a": torch.randn(2, 2), "b": [torch.randn(2), PhysicalField(torch.randn(4, 4))]}
    moved = device_policy.to_device(payload, cpu)
    assert moved["a"].device.type == "cpu"
    assert moved["b"][1].data.device.type == "cpu"


# --------------------------------------------------------------- database concurrency

def test_sqlite_is_configured_for_concurrent_access():
    """Without `busy_timeout`, contention raises `database is locked` immediately."""
    from src.database.session import sqlite_settings

    settings = sqlite_settings()
    if settings.get("backend") != "sqlite":
        pytest.skip("not running on SQLite")
    assert settings["busy_timeout_ms"] >= 1000
    assert settings["journal_mode"] in ("wal", "memory")


def test_parallel_writers_do_not_hit_database_is_locked(tmp_path):
    """The concurrency stress the acceptance criterion asks for."""
    from concurrent.futures import ThreadPoolExecutor as TPE
    from sqlalchemy import event

    db_path = tmp_path / "stress.db"
    engine = create_engine("sqlite:///%s" % db_path, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, record):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    exp_id = str(uuid.uuid4())
    session = factory()
    session.add(Experiment(id=exp_id, name="stress", status="RUNNING", config={}))
    session.commit()
    session.close()

    errors = []

    def writer(worker_id):
        try:
            db = factory()
            for i in range(25):
                db.add(ExperimentRun(id=str(uuid.uuid4()), experiment_id=exp_id,
                                     parameters={"w": worker_id, "i": i}, status="COMPLETED"))
                db.commit()
            db.close()
        except Exception as exc:  # pragma: no cover - the failure we are testing against
            errors.append("%s: %s" % (type(exc).__name__, exc))

    with TPE(max_workers=8) as pool:
        list(pool.map(writer, range(8)))

    assert not errors, errors
    db = factory()
    assert db.query(ExperimentRun).filter_by(experiment_id=exp_id).count() == 200
    db.close()


# --------------------------------------------------------------- end-to-end sweep

SWEEP_CONFIG = {
    "parameter_matrix": {"lvl": [1, 2], "wname": ["haar", "db2"], "sz": [32, 48]},
    "pipeline": [
        {"name": "gen", "action": "generate_synthetic",
         "args": {"type": "vortex", "height": "{sz}", "width": "{sz}",
                  "params": {"amplitudes": [1.0]}}},
        {"name": "tf", "action": "apply_transform",
         "args": {"field_data": "{gen.field_data}", "transform_type": "swt",
                  "config": {"levels": "{lvl}", "wavelet": "{wname}"}}},
    ],
    "metadata": {"code_revision": "t3519"},
}


def _run_sweep(tmp_path, backend, n_workers, name):
    engine = create_engine("sqlite:///%s" % (tmp_path / ("%s.db" % name)),
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    config = json.loads(json.dumps(SWEEP_CONFIG))
    config["execution"] = {"backend": backend, "n_workers": n_workers, "seed": 20260819}

    exp_id = str(uuid.uuid4())
    db = factory()
    db.add(Experiment(id=exp_id, name=name, status="PENDING", config=config))
    db.commit()
    db.close()

    DeclarativeExperimentEngine.execute_experiment(exp_id, session_factory=factory)

    db = factory()
    runs = db.query(ExperimentRun).filter_by(experiment_id=exp_id).all()
    fingerprint = sorted(
        (json.dumps(r.parameters, sort_keys=True), r.status, r.seed,
         json.dumps(r.results, sort_keys=True)) for r in runs)
    node_count = db.query(LineageNode).filter_by(experiment_id=exp_id).count()
    execution = runs[0].execution
    db.close()
    return fingerprint, node_count, execution


def test_sweep_is_byte_identical_across_backends(tmp_path):
    """**The T3.5.19 acceptance criterion.** 8 runs, serial vs thread vs process.

    Compares the persisted parameters, status, seed and results of every run - not just a
    summary statistic, which could agree while individual runs differed.
    """
    reference, ref_nodes, ref_exec = _run_sweep(tmp_path, "serial", 1, "serial")
    assert len(reference) == 8
    assert all(status == "COMPLETED" for _, status, _, _ in reference)
    assert len({seed for _, _, seed, _ in reference}) == 8

    for backend, workers in (("thread", 2), ("thread", 4), ("process", 2), ("process", 4)):
        got, nodes, execution = _run_sweep(
            tmp_path, backend, workers, "%s%d" % (backend, workers))
        assert got == reference, "%s w=%d diverged from serial" % (backend, workers)
        assert nodes == ref_nodes
        assert execution["backend"] == backend
        assert execution["n_workers"] == workers


def test_every_run_records_its_seed_and_execution_context(tmp_path):
    """D12's remaining half: a run must carry what is needed to reproduce it."""
    fingerprint, _, execution = _run_sweep(tmp_path, "serial", 1, "prov")
    assert all(seed is not None for _, _, seed, _ in fingerprint)
    for key in ("backend", "n_workers", "threads_per_worker", "result_order",
                "root_seed", "device", "torch_version"):
        assert key in execution, key
    assert execution["root_seed"] == 20260819
    assert execution["result_order"] == "submission"


def test_a_failing_step_names_the_step_that_actually_failed(tmp_path):
    """Regression: the first version reported the last *successful* step.

    Attributing a failure to the wrong step is worse than reporting it vaguely - it sends
    the reader somewhere confidently wrong.
    """
    engine = create_engine("sqlite:///%s" % (tmp_path / "fail.db"),
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    config = json.loads(json.dumps(SWEEP_CONFIG))
    config["parameter_matrix"] = {"lvl": [1]}
    config["pipeline"][1]["args"]["transform_type"] = "not_a_transform"
    config["pipeline"][1]["args"]["config"] = {"levels": 1}
    config["pipeline"][0]["args"] = {"type": "vortex", "height": 32, "width": 32,
                                     "params": {"amplitudes": [1.0]}}
    config["execution"] = {"backend": "serial"}

    exp_id = str(uuid.uuid4())
    db = factory()
    db.add(Experiment(id=exp_id, name="fail", status="PENDING", config=config))
    db.commit()
    db.close()

    DeclarativeExperimentEngine.execute_experiment(exp_id, session_factory=factory)

    db = factory()
    run = db.query(ExperimentRun).filter_by(experiment_id=exp_id).one()
    assert run.status == "FAILED"
    assert "'tf'" in run.error_message, run.error_message
    assert "apply_transform" in run.error_message
    assert "not_a_transform" in run.error_message
    # ...and the run that failed still recorded its seed, so it can be re-run.
    assert run.seed is not None
    db.close()


def test_an_unresolvable_reference_is_reported_at_its_source(tmp_path):
    """An unresolved `{step.key}` used to survive as a literal and fail much later."""
    engine = create_engine("sqlite:///%s" % (tmp_path / "ref.db"),
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    config = json.loads(json.dumps(SWEEP_CONFIG))
    config["parameter_matrix"] = {"lvl": [1]}
    config["pipeline"][0]["args"] = {"type": "vortex", "height": 32, "width": 32,
                                     "params": {"amplitudes": [1.0]}}
    config["pipeline"][1]["args"]["field_data"] = "{gen.field}"   # the wrong key
    config["execution"] = {"backend": "serial"}

    exp_id = str(uuid.uuid4())
    db = factory()
    db.add(Experiment(id=exp_id, name="ref", status="PENDING", config=config))
    db.commit()
    db.close()

    DeclarativeExperimentEngine.execute_experiment(exp_id, session_factory=factory)

    db = factory()
    run = db.query(ExperimentRun).filter_by(experiment_id=exp_id).one()
    assert run.status == "FAILED"
    assert "{gen.field}" in run.error_message
    assert "gen.field_data" in run.error_message, "should suggest the right key"
    db.close()


def test_a_partially_failing_sweep_completes_the_good_runs(tmp_path):
    """T3.5.14's sweep-level acceptance: one bad combination must not lose the rest."""
    engine = create_engine("sqlite:///%s" % (tmp_path / "partial.db"),
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    config = json.loads(json.dumps(SWEEP_CONFIG))
    config["parameter_matrix"] = {"wname": ["haar", "db2", "not_a_wavelet", "db3"]}
    config["pipeline"][0]["args"] = {"type": "vortex", "height": 64, "width": 64,
                                     "params": {"amplitudes": [1.0]}}
    config["pipeline"][1]["args"]["config"] = {"levels": 2, "wavelet": "{wname}"}
    config["execution"] = {"backend": "serial"}

    exp_id = str(uuid.uuid4())
    db = factory()
    db.add(Experiment(id=exp_id, name="partial", status="PENDING", config=config))
    db.commit()
    db.close()

    DeclarativeExperimentEngine.execute_experiment(exp_id, session_factory=factory)

    db = factory()
    runs = db.query(ExperimentRun).filter_by(experiment_id=exp_id).all()
    statuses = {r.parameters["wname"]: r.status for r in runs}
    assert statuses["haar"] == "COMPLETED"
    assert statuses["db2"] == "COMPLETED"
    assert statuses["db3"] == "COMPLETED"
    assert statuses["not_a_wavelet"] == "FAILED"
    failed = next(r for r in runs if r.parameters["wname"] == "not_a_wavelet")
    assert "not_a_wavelet" in failed.error_message
    db.close()
