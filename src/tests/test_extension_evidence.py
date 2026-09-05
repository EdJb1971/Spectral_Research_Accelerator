"""TG17.13 slice 2: what the fifth-adapter gate may say, and every way it must refuse.

The gate's two halves fail differently and the tests keep them apart, because that difference is
the whole design.  The audit is read live, so its refusal means the audit cannot see what it is
checking (`NOT_RUN`) while its measured finding of a framework source naming the fifth adapter is
a measured falsehood (`FAIL`).  The acceptance run is read from a recording, so absence and drift
mean it has not happened for this code (`NOT_RUN`) while a run that happened with a check failing
is a `FAIL`.

None of these tests import `test_adapter_registry`.  They cannot: the fifth adapter is defined
there precisely because the application may not reach it, and importing that module from here
would give it a second identity beside pytest's and re-execute its lineage registration.  The
recorded run is therefore the only view of it these tests get, which is exactly the view the gate
gets.
"""

import json

import pytest

from src.core import extension_evidence as ev


@pytest.fixture()
def recorded():
    """The recording this checkout actually carries, as committed source."""
    path = ev.record_file()
    assert path.is_file(), "the acceptance run's recording is committed source"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def elsewhere(tmp_path, monkeypatch):
    """A crafted recording, with the live sources and live audit left real.

    Only the record's *location* moves. Pointing `root` at a temporary tree instead would make
    every framework source absent, and the audit would then refuse for that reason and mask
    whatever the test was actually about.
    """
    path = tmp_path / "extension_conformance.json"
    monkeypatch.setattr(ev, "record_file", lambda root=None: path)

    def _write(record):
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return path
    return _write


# ------------------------------------------------------------------ what this checkout says

def test_this_checkout_passes_the_gate_on_a_recording_it_carries(recorded):
    facts = ev.read_extension_conformance()
    assert facts["status"] == "PASS", facts["reasons"]
    assert facts["recorded"]["installation_required_framework_edits"] == 0
    assert recorded["conformant"]


def test_the_standing_glue_count_is_published_rather_than_asserted():
    """The count is reported by a *passing* gate. An unpublished count is one that grows."""
    facts = ev.read_extension_conformance()
    assert facts["status"] == "PASS"
    assert facts["recorded"]["adapter_specific_framework_edits"] >= 1
    assert facts["recorded"]["glue"], "a nonzero count must name its instances"
    for item in facts["recorded"]["glue"]:
        assert item["reason"], "glue is declared with a stated reason, never a bare entry"


def test_the_audit_is_read_live_and_not_inherited_from_the_recording():
    """A fact recomputable from committed source is recomputed; a receipt for it could go stale."""
    facts = ev.read_extension_conformance()
    assert facts["audit_read_live_here"] is True
    assert facts["audit"]["status"] == "MEASURED"


def test_every_check_the_contract_names_was_actually_performed(recorded):
    performed = {item["check"] for item in recorded["checks"]}
    assert performed == {name for name, _ in ev.REQUIRED_CHECKS}
    assert all(item["status"] == "PASS" for item in recorded["checks"])


def test_the_adapter_is_defined_outside_the_application_by_its_file_not_its_module_name(recorded):
    """The first version of this check compared a bare module name and would pass for anything."""
    detail = next(item["detail"] for item in recorded["checks"]
                  if item["check"] == "defined_outside_the_application")
    assert "src/tests/test_adapter_registry.py" in detail
    assert not any(detail.endswith(root + "x") for root in ev.APPLICATION_SOURCE_ROOTS)


# ---------------------------------------------------------------------- the recorded half

def test_an_absent_recording_reads_not_run(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "record_file", lambda root=None: tmp_path / "nothing.json")
    facts = ev.read_extension_conformance()
    assert facts["status"] == "NOT_RUN"
    assert "no acceptance run has been recorded" in " ".join(facts["reasons"])


def test_a_recording_made_against_a_different_contract_reads_not_run(recorded, elsewhere):
    elsewhere({**recorded, "contract_sha256": "0" * 64})
    facts = ev.read_extension_conformance()
    assert facts["status"] == "NOT_RUN"
    assert "declared extension contract has changed" in " ".join(facts["reasons"])


def test_a_recording_made_against_different_source_reads_not_run(recorded, elsewhere):
    """Moving any source the measurement depends on returns the gate to `NOT_RUN`, not to a pass."""
    moved = dict(recorded["source_sha256"])
    moved["src/core/extension_audit.py"] = "1" * 64
    elsewhere({**recorded, "source_sha256": moved})
    facts = ev.read_extension_conformance()
    assert facts["status"] == "NOT_RUN"
    assert "src/core/extension_audit.py has changed" in " ".join(facts["reasons"])


def test_a_weakened_acceptance_test_returns_the_gate_to_not_run(recorded, elsewhere):
    """The test module's own source is bound, so editing it un-measures the run until it is rerun."""
    assert "src/tests/test_adapter_registry.py" in recorded["source_sha256"]
    moved = dict(recorded["source_sha256"])
    moved["src/tests/test_adapter_registry.py"] = "2" * 64
    elsewhere({**recorded, "source_sha256": moved})
    facts = ev.read_extension_conformance()
    assert facts["status"] == "NOT_RUN"
    assert "test_adapter_registry.py has changed" in " ".join(facts["reasons"])


def test_a_recording_that_performed_fewer_checks_is_refused_rather_than_read_as_a_pass(
        recorded, elsewhere):
    fewer = [item for item in recorded["checks"] if item["check"] != "conforms_to_the_kit"]
    elsewhere({**recorded, "checks": fewer})
    facts = ev.read_extension_conformance()
    assert facts["status"] == "NOT_RUN"
    assert "did not perform conforms_to_the_kit" in " ".join(facts["reasons"])


def test_a_run_that_happened_and_failed_a_check_reads_fail_not_not_run(recorded, elsewhere):
    """A run that happened and failed is a different fact from a run that has not happened."""
    checks = [{**item, "status": "FAIL"} if item["check"] == "conforms_to_the_kit" else item
              for item in recorded["checks"]]
    elsewhere({**recorded, "checks": checks, "conformant": False,
               "failed_checks": ["conforms_to_the_kit"]})
    facts = ev.read_extension_conformance()
    assert facts["status"] == "FAIL"
    assert "conforms_to_the_kit" in " ".join(facts["reasons"])


def test_a_recording_that_produced_no_measurement_reads_not_run(recorded, elsewhere):
    elsewhere({**recorded, "status": "REFUSED", "reasons": ["the fixture was not offered"]})
    facts = ev.read_extension_conformance()
    assert facts["status"] == "NOT_RUN"
    assert "the fixture was not offered" in " ".join(facts["reasons"])


# ------------------------------------------------------------------------- the live half

def test_an_audit_that_cannot_see_what_it_checks_reads_not_run(monkeypatch):
    """The audit's refusal is not a weaker pass; it is the absence of a measurement."""
    monkeypatch.setattr(ev, "audit_framework_edits", lambda root=None: {
        "status": "NOT_MEASURED", "framework_sources": 17,
        "reasons": ["no domains are registered, so the scan had no names to look for"],
        "adapter_specific_framework_edits": "NOT_MEASURED"})
    facts = ev.read_extension_conformance()
    assert facts["status"] == "NOT_RUN"
    assert "no names to look for" in " ".join(facts["reasons"])


def test_a_framework_source_naming_the_fifth_adapter_reads_fail(monkeypatch):
    """Measured, and false: the installation claim is the half this gate turns on."""
    monkeypatch.setattr(ev, "audit_framework_edits", lambda root=None: {
        "status": "MEASURED", "framework_sources": 17, "reasons": [],
        "installation_required_framework_edits": 1,
        "adapter_specific_framework_edits": 1, "glue": []})
    facts = ev.read_extension_conformance()
    assert facts["status"] == "FAIL"
    assert "not installed without framework edits" in " ".join(facts["reasons"])


# --------------------------------------------------------------------------- the contract

def test_rewriting_a_declared_reason_moves_the_contract_digest(monkeypatch):
    """A declaration is excused by its written reason, so the reason is part of the contract."""
    before = ev.conformance_contract()["contract_sha256"]
    rewritten = tuple((source, domain, kind, "because I said so")
                      for source, domain, kind, _ in ev.DECLARED_OCCURRENCES)
    monkeypatch.setattr(ev, "DECLARED_OCCURRENCES", rewritten)
    assert ev.conformance_contract()["contract_sha256"] != before


def test_widening_what_counts_as_glue_moves_the_contract_digest(monkeypatch):
    before = ev.conformance_contract()["contract_sha256"]
    monkeypatch.setattr(ev, "GLUE_KINDS", ("BRANCH", "DEFAULT"))
    assert ev.conformance_contract()["contract_sha256"] != before


def test_the_contract_names_every_check_with_what_it_means():
    contract = ev.conformance_contract()
    named = {row["check"]: row["means"] for row in contract["required_checks"]}
    assert named.keys() == {name for name, _ in ev.REQUIRED_CHECKS}
    assert all(text.strip().endswith(".") for text in named.values())


# ------------------------------------------------------------- what the measurement refuses

class _Stub:
    """Only enough of an adapter to reach the refusal being tested, and no further."""

    def __init__(self, domain, translate):
        self.declaration = type("_D", (), {"name": domain})()
        self.translate = translate


def _record(domain):
    return type("_R", (), {"domain": domain})()


def test_measuring_another_adapter_is_refused_rather_than_recorded():
    facts = ev.measure_extension_conformance(
        _Stub("argo_float", lambda *a: None), _record("argo_float"), [object()])
    assert facts["status"] == "REFUSED"
    assert "would qualify nothing" in " ".join(facts["reasons"])


def test_a_native_record_from_another_domain_is_refused():
    facts = ev.measure_extension_conformance(
        _Stub(ev.FIFTH_DOMAIN, lambda *a: None), _record("argo_float"), [object()])
    assert facts["status"] == "REFUSED"
    assert "belongs to" in " ".join(facts["reasons"])


def test_a_measurement_with_no_window_is_refused_rather_than_run_empty():
    facts = ev.measure_extension_conformance(
        _Stub(ev.FIFTH_DOMAIN, lambda *a: None), _record(ev.FIFTH_DOMAIN), [])
    assert facts["status"] == "REFUSED"
    assert "silence would not be evidence" in " ".join(facts["reasons"])


def test_an_adapter_whose_mathematics_cannot_be_traced_is_refused_not_passed():
    """An unanswerable question is not a satisfied one."""
    facts = ev.measure_extension_conformance(
        _Stub(ev.FIFTH_DOMAIN, len), _record(ev.FIFTH_DOMAIN), [object()])
    assert facts["status"] == "REFUSED"
    assert "unanswerable rather than satisfied" in " ".join(facts["reasons"])


def test_defining_source_locates_a_callable_inside_this_repository():
    assert ev.defining_source(_Stub(ev.FIFTH_DOMAIN, ev.conformance_contract)) == \
        "src/core/extension_evidence.py"
    assert "src/core/extension_evidence.py".startswith(ev.APPLICATION_SOURCE_ROOTS), \
        "a translate defined in the application must fail the check, not pass it"


def test_defining_source_returns_none_for_a_callable_outside_this_checkout():
    assert ev.defining_source(_Stub(ev.FIFTH_DOMAIN, json.dumps)) is None


#: A module name that is not in `sys.modules`, so `inspect.getsourcefile` can find no loader and
#: gives up - while `__module__` remains a perfectly confident, perfectly useless string. That gap
#: is the one the first version of `defined_outside_the_application` fell into.
BLIND_SPOT = "the.seams.blind.spot"


def _compiled_translate():
    """A callable with no source file at all, carrying a module name that answers nothing."""
    scope = {"__name__": BLIND_SPOT}
    exec(compile("def translate(record, declaration, parameters): return None",
                 "<no file>", "exec"), scope)
    return scope["translate"]


def test_defining_source_returns_none_for_a_callable_with_no_source_file():
    """The check must key on the file. `__module__` here is a string that starts with no
    application root, so a fallback to it would report this adapter as correctly installed."""
    translate = _compiled_translate()
    assert translate.__module__ == BLIND_SPOT
    assert not BLIND_SPOT.startswith(ev.APPLICATION_SOURCE_ROOTS)
    assert ev.defining_source(_Stub(ev.FIFTH_DOMAIN, translate)) is None


def test_an_adapter_compiled_without_a_source_file_is_refused_not_passed():
    facts = ev.measure_extension_conformance(
        _Stub(ev.FIFTH_DOMAIN, _compiled_translate()), _record(ev.FIFTH_DOMAIN), [object()])
    assert facts["status"] == "REFUSED"
    assert "unanswerable rather than satisfied" in " ".join(facts["reasons"])
