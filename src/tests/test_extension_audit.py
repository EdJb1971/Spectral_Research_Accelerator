"""TG17.13 slice 1: what the source-edit audit must refuse.

The audit's own failure mode is the one it exists to prevent, so most of these tests are about
the audit reporting a clean surface it did not actually inspect.
"""

from __future__ import annotations

import pytest

from src.core.extension_audit import (BRANCH, DECLARED_OCCURRENCES, FIFTH_ADAPTER_NAMES,
                                      FRAMEWORK_SOURCES, GLUE_KINDS, PROSE, RECIPE, REPO_ROOT,
                                      audit_framework_edits, source_digests)


@pytest.fixture()
def audit():
    return audit_framework_edits()


# ------------------------------------------------------- the claim the gate turns on

def test_no_framework_source_names_the_synthetic_fifth_adapter(audit):
    """The installation claim, and the only half of this audit that is absolute."""
    assert audit["status"] == "MEASURED", audit.get("reasons")
    assert audit["installation_required_framework_edits"] == 0


def test_a_framework_source_naming_the_fifth_adapter_is_refused(tmp_path, monkeypatch):
    """No declaration may excuse it: there is no admissible reason for the name to be there."""
    import src.core.extension_audit as module

    for name in FRAMEWORK_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# clean\n", encoding="utf-8")
    (tmp_path / "src/core/experiment_run.py").write_text(
        "if domain == 'synthetic_rank_sensor':\n    pass\n", encoding="utf-8")

    facts = module.audit_framework_edits(tmp_path, domains=("reanalysis",))
    assert facts["status"] == "NOT_MEASURED"
    assert facts["adapter_specific_framework_edits"] == "NOT_MEASURED"
    assert any("not installed without framework edits" in reason for reason in facts["reasons"])


# ------------------------------------------------------- the audit's own blind spots

def test_an_empty_registry_is_refused_rather_than_scanned(tmp_path):
    """A scan with no names to look for finds nothing, and that is not a clean surface."""
    for name in FRAMEWORK_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# clean\n", encoding="utf-8")

    facts = audit_framework_edits(tmp_path, domains=())
    assert facts["status"] == "NOT_MEASURED"
    assert any("no names to look for" in reason for reason in facts["reasons"])


def test_a_missing_framework_source_is_refused(tmp_path):
    """A source that is not there cannot be reported as a source that is clean."""
    facts = audit_framework_edits(tmp_path, domains=("reanalysis",))
    assert facts["status"] == "NOT_MEASURED"
    assert any("absent from this checkout" in reason for reason in facts["reasons"])


def test_an_undeclared_occurrence_is_refused_not_ignored(tmp_path):
    """The audit must not silently pass over a domain name it has no reason for."""
    for name in FRAMEWORK_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# clean\n", encoding="utf-8")
    (tmp_path / "src/api/mining.py").write_text(
        "if domain == 'invented_domain':\n    pass\n", encoding="utf-8")

    facts = audit_framework_edits(tmp_path, domains=("invented_domain",))
    assert facts["status"] == "NOT_MEASURED"
    assert any("no declared reason" in reason for reason in facts["reasons"])


# ------------------------------------------------------- the number itself

def test_the_glue_count_is_reported_rather_than_asserted_to_be_zero(audit):
    """G17 asks that glue trend to zero. It has not reached zero, and the audit says so."""
    assert audit["adapter_specific_framework_edits"] == 1
    glue = audit["glue"]
    assert len(glue) == 1
    assert glue[0]["source"] == "frontend/src/components/AcquisitionView.tsx"
    assert glue[0]["domain"] == "reanalysis"
    assert "Copernicus" in glue[0]["reason"]


def test_only_a_behaviour_branch_counts_as_glue():
    """Prose and a named recipe's own content are not edits made to accommodate a domain."""
    assert GLUE_KINDS == (BRANCH,)
    kinds = {kind for _, _, kind, _ in DECLARED_OCCURRENCES}
    assert BRANCH in kinds and PROSE in kinds and RECIPE in kinds


def test_every_declaration_states_a_reason():
    """A declaration is a sentence someone can disagree with, not an entry on a suppression list."""
    for source, domain, kind, reason in DECLARED_OCCURRENCES:
        assert source in FRAMEWORK_SOURCES, source
        assert len(reason.split()) >= 4, "%s/%s has no real reason" % (source, domain)
        assert kind in (BRANCH, PROSE, RECIPE, "DEFAULT")


def test_the_declared_occurrences_are_all_still_real(audit):
    """A declaration for an occurrence that no longer exists is a stale excuse."""
    assert audit["occurrences_found"] >= audit["declared_occurrences"]


# ------------------------------------------------------- binding

def test_every_framework_source_digests(audit):
    digests = source_digests()
    assert set(digests) == set(FRAMEWORK_SOURCES)
    assert "ABSENT" not in digests.values()
    assert len({len(value) for value in digests.values()}) == 1


def test_the_fifth_adapter_names_are_the_ones_the_acceptance_test_uses():
    """Restated in the audit deliberately; this asserts the restatement has not drifted.

    Read from source rather than imported. The first version of this test did
    ``from src.tests import test_adapter_registry``, which gives that module a second identity
    beside the one pytest's rewriter already loaded, executes its body twice, and trips the
    lineage registry's refusal to overwrite a reconstructor - a guard that exists precisely so
    conformance cannot depend on import order. Importing a test module to inspect it is the
    wrong tool; the constants are readable without running anything.
    """
    source = (REPO_ROOT / "src/tests/test_adapter_registry.py").read_text(encoding="utf-8")
    assert 'FIFTH_DOMAIN = "synthetic_rank_sensor"' in source
    assert 'name="monotone_rank"' in source
    for name in FIFTH_ADAPTER_NAMES:
        assert name in source, "%s is no longer a name the acceptance test uses" % name
