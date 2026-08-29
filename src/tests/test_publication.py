"""Acceptance tests for immutable evidence publication on the deployed filesystem (D71)."""

from __future__ import annotations

import errno
import multiprocessing
import os
from pathlib import Path
import queue

import pytest

import src.core.publication as publication
from src.core.publication import publish_new_bytes


def _racing_publisher(target, index, payload, ready, start, outcomes):
    """Spawn-safe worker: all writers cross the publication boundary together."""
    ready.put(os.getpid())
    if not start.wait(30):
        outcomes.put(("timeout", index))
        return
    try:
        method = publish_new_bytes(target, payload, "racing receipt")
        outcomes.put(("published", index, method))
    except FileExistsError:
        outcomes.put(("exists", index))
    except BaseException as exc:  # pragma: no cover - reported in the parent with full detail
        outcomes.put(("error", index, type(exc).__name__, str(exc)))


def test_publication_is_complete_no_replace_and_on_the_workspace_volume(tmp_path):
    target = tmp_path / "nested" / "receipt.json"
    payload = b'{"complete":true}\n'

    method = publish_new_bytes(target, payload, "test receipt")

    assert target.read_bytes() == payload
    assert not list(target.parent.glob(".%s.*.tmp" % target.name))
    if os.name == "nt":
        # pytest.ini deliberately pins tmp_path to the repository drive. This assertion prevents
        # the D71 acceptance from silently moving back to NTFS system temp in a future config.
        assert target.drive.lower() == Path.cwd().drive.lower()
        assert method == "windows-rename-no-replace"

    with pytest.raises(FileExistsError, match="refusing to overwrite test receipt"):
        publish_new_bytes(target, b'{"replacement":true}\n', "test receipt")
    assert target.read_bytes() == payload
    assert not list(target.parent.glob(".%s.*.tmp" % target.name))


def test_publication_race_has_exactly_one_complete_winner(tmp_path):
    target = tmp_path / "race" / "receipt.bin"
    worker_count = 8
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    outcomes = context.Queue()
    start = context.Event()
    payloads = [(("publisher-%02d|" % index).encode("ascii") * 4096)
                for index in range(worker_count)]
    processes = [
        context.Process(
            target=_racing_publisher,
            args=(str(target), index, payload, ready, start, outcomes),
        )
        for index, payload in enumerate(payloads)
    ]
    for process in processes:
        process.start()
    for _ in processes:
        ready.get(timeout=30)
    start.set()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive(), "publication race worker did not terminate"
        assert process.exitcode == 0

    results = []
    for _ in processes:
        try:
            results.append(outcomes.get(timeout=5))
        except queue.Empty:  # pragma: no cover - assertion reports the missing worker result
            pytest.fail("publication race worker returned no outcome")

    published = [result for result in results if result[0] == "published"]
    refused = [result for result in results if result[0] == "exists"]
    errors = [result for result in results if result[0] not in ("published", "exists")]
    assert not errors
    assert len(published) == 1
    assert len(refused) == worker_count - 1
    assert target.read_bytes() == payloads[published[0][1]]
    assert target.read_bytes() in payloads
    assert not list(target.parent.glob(".%s.*.tmp" % target.name))


def test_failed_namespace_publication_leaves_no_target_or_temporary(tmp_path, monkeypatch):
    target = tmp_path / "failure" / "receipt.json"

    def unsupported(source, destination):
        assert source.parent == destination.parent
        assert source.read_bytes() == b"complete payload"
        raise OSError(errno.EOPNOTSUPP, "filesystem has no atomic no-replace primitive")

    monkeypatch.setattr(publication, "_publish_temporary_no_replace", unsupported)
    with pytest.raises(OSError, match="without weakening no-overwrite semantics"):
        publish_new_bytes(target, b"complete payload", "failure receipt")

    assert not target.exists()
    assert not list(target.parent.glob(".%s.*.tmp" % target.name))


def test_publication_refuses_ambiguous_inputs_before_creating_a_file(tmp_path):
    target = tmp_path / "receipt.json"
    with pytest.raises(TypeError, match="must be bytes"):
        publish_new_bytes(target, bytearray(b"mutable"))
    with pytest.raises(ValueError, match="non-empty"):
        publish_new_bytes(target, b"payload", "  ")
    assert not target.exists()
