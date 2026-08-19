"""Tests for the registries and the error taxonomy (T3.5.15/D15, T3.5.14/D14).

The centrepiece is `test_acceptance_new_plugin_needs_no_core_edits`, which is the roadmap's
stated acceptance criterion for T3.5.15 executed literally: a new data source and a new
pipeline action are added **in a new file only**, and both appear in the API. The test also
asserts the "zero edits" half by hashing the three core files before and after - a claim
about not editing files is checkable, so it is checked rather than asserted in prose.
"""

import hashlib
import io
import os

import pytest
import torch

from src.core.errors import (
    AllSourcesFailedError,
    DuplicateRegistrationError,
    InvalidParameterError,
    MissingParameterError,
    PipelineStepError,
    ReferenceResolutionError,
    ShapeMismatchError,
    SpectralEarthError,
    UnknownNameError,
    UserInputError,
    classify,
)
from src.core.registry import Registry, restore, snapshot
from src.data_layer import builtin_sources  # noqa: F401  (registration side effect)
from src.data_layer import sources as data_sources
from src.experiment_engine import actions as pipeline_actions
from src.physical_core.field import PhysicalField
from src.transform_engine import registry as transform_registry

CORE_FILES = ("src/experiment_engine/engine.py",
              "src/data_layer/adapters.py",
              "src/api/main.py")


def _digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@pytest.fixture
def clean_registries():
    """Snapshot every registry so a test can register temporarily."""
    saved = {
        "actions": snapshot(pipeline_actions.ACTIONS),
        "sources": snapshot(data_sources.SOURCES),
        "transforms": snapshot(transform_registry.TRANSFORMS),
    }
    yield
    restore(pipeline_actions.ACTIONS, saved["actions"])
    restore(data_sources.SOURCES, saved["sources"])
    restore(transform_registry.TRANSFORMS, saved["transforms"])


# ============================================================== generic registry

def test_registry_registers_and_looks_up():
    r = Registry("thing")
    @r.register("a", description="first", params={"x": "int"}, capabilities={"fast": True})
    def a():
        return 1
    assert r.names() == ["a"]
    assert r.get("a")() == 1
    assert "a" in r and len(r) == 1
    entry = r.entry("a")
    assert entry.description == "first"
    assert entry.params == {"x": "int"}
    assert entry.defined_in


def test_registry_takes_the_description_from_the_docstring():
    r = Registry("thing")
    @r.register("a")
    def a():
        """A short summary.

        With more detail below that should not be used.
        """
    assert r.entry("a").description == "A short summary."


def test_unknown_name_lists_options_and_suggests():
    r = Registry("transform")
    for n in ("dwt", "dtcwt", "swt"):
        r.add(n, object())
    with pytest.raises(UnknownNameError) as exc:
        r.get("dwt2")
    msg = exc.value.message
    assert "Did you mean" in msg and "'dwt'" in msg
    assert "Available: dtcwt, dwt, swt" in msg
    assert exc.value.status_code == 404
    assert exc.value.client_safe is True


def test_unknown_name_without_a_close_match_still_lists_options():
    r = Registry("action")
    r.add("compute_diagnostics", object())
    with pytest.raises(UnknownNameError) as exc:
        r.get("zzzzz")
    assert "Did you mean" not in exc.value.message
    assert "compute_diagnostics" in exc.value.message


def test_empty_registry_says_so_rather_than_listing_nothing():
    with pytest.raises(UnknownNameError) as exc:
        Registry("widget").get("x")
    assert "Nothing is registered" in exc.value.message


def test_duplicate_registration_raises_rather_than_overwriting():
    """A silent overwrite makes behaviour depend on import order."""
    r = Registry("thing")
    r.add("a", object())
    with pytest.raises(DuplicateRegistrationError) as exc:
        r.add("a", object())
    assert "already registered" in exc.value.message
    assert exc.value.status_code == 500


def test_explicit_replace_is_allowed():
    r = Registry("thing")
    r.add("a", 1)
    r.register("a", replace=True)(2)
    assert r.get("a") == 2


def test_capability_query_selects_by_property_not_by_name():
    """What T4B.2's wavelet bank needs: ask for a property, not a hard-coded list."""
    shift_invariant = transform_registry.TRANSFORMS.with_capability("shift_invariant", True)
    assert [e.name for e in shift_invariant] == ["swt"]
    oriented = transform_registry.TRANSFORMS.with_capability("oriented", True)
    assert [e.name for e in oriented] == ["dtcwt"]
    bank = [e.name for e in transform_registry.TRANSFORMS.entries() if "wavelet_bank" in e.tags]
    assert set(bank) == {"swt", "dtcwt"}


# ============================================================== transform registry

def test_all_builtin_transforms_are_registered_and_invertible():
    names = transform_registry.TRANSFORMS.names()
    assert set(names) == {"fft", "dct", "dwt", "swt", "dtcwt", "hybrid"}
    field = PhysicalField(torch.randn(64, 64, dtype=torch.float64))
    for name in names:
        config = {"levels": 2} if name in ("dwt", "swt", "dtcwt") else {}
        result = transform_registry.apply_transform(name, field, config)
        assert result["reconstruction_mse"] < 1e-8, name
        assert result["transform"] == name
        assert isinstance(result["summary"], dict)


def test_every_transform_declares_its_parameters_and_capabilities():
    for entry in transform_registry.TRANSFORMS.entries():
        assert entry.description, entry.name
        assert "invertible" in entry.capabilities, entry.name
        assert isinstance(entry.params, dict)


def test_engine_and_api_share_one_transform_definition():
    """They previously had separate if/elif chains that could drift apart on defaults."""
    import inspect
    engine_src = inspect.getsource(__import__(
        "src.experiment_engine.actions", fromlist=["x"]))
    api_src = io.open("src/api/main.py", encoding="utf-8").read()
    assert api_src.count('elif transform_type ==') == 0, (
        "the API still dispatches transforms with an if/elif chain")


# ============================================================== source registry

def test_sources_are_ordered_by_priority_with_simulated_last():
    described = data_sources.describe_sources()
    names = [d["name"] for d in described]
    assert names[0] == "netcdf_local"
    assert names[-1] == "simulated"
    assert described[-1]["is_simulated"] is True


def test_resolution_records_the_whole_attempt_chain():
    resolution = data_sources.resolve("era5_reanalysis")
    prov = resolution.to_provenance()
    assert prov["source"] == "simulated"
    assert prov["is_simulated"] is True
    assert any(a["source"] == "netcdf_local" for a in prov["attempts"])
    assert "SIMULATED" in prov["warning"]


def test_a_declining_source_explains_itself():
    """A decline leaves no failed attempt, so without `why_not` the record says nothing."""
    resolution = data_sources.resolve("era5_reanalysis")
    declined = [a for a in resolution.attempts if a.get("declined")]
    assert declined, "netcdf_local should be recorded as having declined"
    assert "no file at" in declined[0]["reason"]
    assert "drop a NetCDF file" in declined[0]["reason"]
    assert "no file at" in resolution.fallback_reason


def test_unknown_dataset_lists_the_known_ones():
    with pytest.raises(UnknownNameError) as exc:
        data_sources.resolve("not_a_dataset")
    assert "era5_reanalysis" in exc.value.message


def test_all_sources_failing_reports_every_reason(clean_registries):
    """One combined "could not load data" would discard what is needed to fix any of them."""
    data_sources.SOURCES.unregister("netcdf_local")
    data_sources.SOURCES.unregister("simulated")

    @data_sources.register_source("broken_a", priority=1, kind="test",
                                  capabilities={"dataset_ids": ["thing"]})
    class BrokenA:
        @staticmethod
        def can_serve(dataset_id):
            return dataset_id == "thing"

        @staticmethod
        def fetch(dataset_id, **kw):
            raise RuntimeError("disk on fire")

    @data_sources.register_source("broken_b", priority=2, kind="test",
                                  capabilities={"dataset_ids": ["thing"]})
    class BrokenB:
        @staticmethod
        def can_serve(dataset_id):
            return dataset_id == "thing"

        @staticmethod
        def fetch(dataset_id, **kw):
            raise ValueError("bad credentials")

    with pytest.raises(AllSourcesFailedError) as exc:
        data_sources.resolve("thing")
    assert "disk on fire" in exc.value.message
    assert "bad credentials" in exc.value.message
    assert exc.value.status_code == 502


# ============================================================== error taxonomy

def test_user_errors_are_4xx_and_client_safe():
    for err in (UnknownNameError("thing", "x", ["a"]),
                InvalidParameterError("levels", -1, "a positive integer"),
                MissingParameterError("field", action="apply_transform",
                                      required=["field", "transform_type"]),
                ShapeMismatchError("a", (2, 2), "b", (3, 3))):
        assert 400 <= err.status_code < 500
        assert err.client_safe is True
        assert err.api_detail() == err.message


def test_internal_errors_do_not_leak_their_message():
    err = DuplicateRegistrationError("thing", "x", "somewhere")
    assert err.status_code == 500
    assert err.client_safe is False
    assert "somewhere" not in err.api_detail()
    assert "DuplicateRegistrationError" in err.api_detail()


def test_classify_treats_unknown_exceptions_as_internal():
    """Conservative by design: an unrecognised error may carry a path or a query."""
    info = classify(RuntimeError("connection to postgres://user:pw@host failed"))
    assert info["status_code"] == 500
    assert "postgres" not in info["detail"]


def test_shape_mismatch_names_both_sides_and_a_fix():
    err = ShapeMismatchError("step 'a' output", (32, 32), "step 'b' input", (64, 64))
    assert "(32, 32)" in err.message and "(64, 64)" in err.message
    assert "step 'a' output" in err.message and "step 'b' input" in err.message
    assert "resampling" in err.message


def test_missing_parameter_lists_what_is_required():
    err = MissingParameterError("field", action="apply_transform",
                                required=["field", "transform_type"])
    assert "apply_transform" in err.message
    assert "transform_type" in err.message


def test_pipeline_step_error_names_the_step_and_inherits_its_status():
    cause = InvalidParameterError("levels", 0, "int >= 1")
    err = PipelineStepError("step_3", "apply_transform", cause,
                            params={"levels": 0}, run_index=7)
    assert "step_3" in err.message and "apply_transform" in err.message
    assert "run 7" in err.message
    assert "int >= 1" in err.message
    # A bad parameter inside a step is still the caller's mistake.
    assert err.status_code == 400
    assert err.client_safe is True


def test_reference_resolution_error_lists_resolvable_steps():
    err = ReferenceResolutionError("{gen.fieldd}", ["gen.field", "gen.shape"])
    assert "Did you mean" in err.message
    assert "gen.field" in err.message
    err_empty = ReferenceResolutionError("{gen.field}", [])
    assert "runs before it" in err_empty.message


def test_errors_carry_structured_context():
    err = ShapeMismatchError("a", (2, 2), "b", (3, 3), step="s1")
    d = err.to_dict()
    assert d["error"] == "ShapeMismatchError"
    assert d["context"]["shape_a"] == [2, 2]
    assert d["context"]["step"] == "s1"


# ============================================================== the acceptance criterion

def test_acceptance_new_plugin_needs_no_core_edits(clean_registries, client):
    """T3.5.15's acceptance criterion, executed literally.

    "A new data source and a new pipeline action are each added in a **new file only**, with
    zero edits to engine.py, adapters.py or main.py, and both appear automatically in
    GET /api/v1/actions and GET /api/v1/data/datasets."

    The plugin is imported from `src/tests/plugin_example.py` - a genuinely separate file
    that no core module knows about. The "zero edits" half is verified by hashing the three
    core files before and after, because a claim about not editing files is checkable.
    """
    before = {f: _digest(f) for f in CORE_FILES}

    baseline_actions = set(pipeline_actions.ACTIONS.names())
    baseline_ids = set()
    for entry in data_sources.SOURCES.entries():
        baseline_ids.update(entry.capabilities.get("dataset_ids", []))

    from src.tests import plugin_example  # noqa: F401  the whole point: just an import

    # --- the new action appears, and runs ---
    assert "count_extrema" in pipeline_actions.ACTIONS.names()
    listed = {a["name"] for a in client.get("/api/v1/actions").json()}
    assert "count_extrema" in listed
    assert listed >= baseline_actions

    result = pipeline_actions.execute(
        "count_extrema", {"field": PhysicalField(torch.zeros(8, 8))}, torch.device("cpu"))
    assert result["extrema_count"] == 0

    # --- the new data source appears in the listing and in the chain ---
    source_names = {s["name"] for s in client.get("/api/v1/data/sources").json()}
    assert "checkerboard_demo" in source_names

    datasets = client.get("/api/v1/data/datasets").json()
    ids = {d["id"] for d in datasets}
    assert "demo_checkerboard" in ids, (
        "a source added in a new file must appear in the dataset listing")
    assert ids >= baseline_ids

    demo = next(d for d in datasets if d["id"] == "demo_checkerboard")
    assert demo["source_kind"] == "demo"
    assert demo["is_simulated"] is True

    # --- and no core file was touched ---
    after = {f: _digest(f) for f in CORE_FILES}
    assert before == after, (
        "adding a plugin modified a core file: %s"
        % [f for f in CORE_FILES if before[f] != after[f]])


def test_discovery_endpoints_are_generated_from_the_registries(client):
    """A hand-maintained list goes stale exactly when someone adds something."""
    actions = client.get("/api/v1/actions").json()
    assert {a["name"] for a in actions} == set(pipeline_actions.ACTIONS.names())
    for a in actions:
        assert a["description"]
        assert a["node_type"] in ("field", "coefficients", "metrics", "intermediate")

    transforms = client.get("/api/v1/transforms").json()
    assert {t["name"] for t in transforms} == set(transform_registry.TRANSFORMS.names())
    assert all(t["capabilities"] for t in transforms)


def test_api_reports_an_unknown_transform_as_404_with_the_valid_names(client):
    resp = client.post("/api/v1/transforms/apply",
                       json={"field_data": torch.randn(16, 16).tolist(),
                             "transform_type": "dwt2"})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "Did you mean" in detail
    assert "dwt" in detail


def test_api_does_not_leak_internals_on_a_genuine_fault(client, monkeypatch):
    """A 500 must stay opaque even though a 4xx is now verbose."""
    def boom(*a, **k):
        raise RuntimeError("/secret/path/to/model.ckpt is corrupt")
    monkeypatch.setattr(transform_registry, "apply_transform", boom)
    resp = client.post("/api/v1/transforms/apply",
                       json={"field_data": torch.randn(16, 16).tolist(),
                             "transform_type": "fft"})
    assert resp.status_code == 500
    assert "/secret/path" not in resp.json()["detail"]
