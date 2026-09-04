"""TG17.14 slice 1: the live-source gate can ingest evidence and cannot invent it.

No test in this file reaches a network. The records are deliberately small reporter-shaped
objects used to test the decision boundary; the real measurement remains separately opt-in.
"""

import json
import shutil

import pytest

from src.core import live_source_evidence as evidence


@pytest.fixture()
def checkout(tmp_path):
    for name in evidence.SOURCE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(evidence.REPO_ROOT / name, target)
    (tmp_path / evidence.MEASUREMENT_DIR).mkdir()
    return tmp_path


def _outcome(declared, *, status="PASS"):
    network = bool(declared["network_required_for_pass"])
    return {
        "domain": declared["domain"],
        "source_id": declared["source_id"],
        "source_version": declared["source_version"],
        "status": status,
        "started_utc": "2026-09-04T01:00:00Z",
        "finished_utc": "2026-09-04T01:00:01Z",
        "network_used": network,
        "coverage": {"start": "2026-01-01T00:00:00Z",
                     "end": "2026-01-02T00:00:00Z", "exact": True},
        "content": {"sha256": "a" * 64, "observed_values": 12},
        "reason": "" if status == "PASS" else "the archive declined this bounded request",
    }


def _record(checkout, transform=lambda rows: rows):
    contract = evidence.live_source_contract()
    outcomes = transform([_outcome(row) for row in contract["domains"]])
    record = {
        "schema": evidence.SCHEMA,
        "record_kind": evidence.RECORD_KIND,
        "recorded_utc": "2026-09-04T01:00:02Z",
        "contract_sha256": contract["contract_sha256"],
        "source_sha256": evidence.source_digests(checkout),
        "outcomes": outcomes,
        "bespoke_record": {"content_sha256": "a" * 64,
                           "provenance": "reporter-shaped test record",
                           "licence": "test-only"},
        "claim_boundary": contract["claim_boundary"],
    }
    record["record_sha256"] = evidence._digest(record)
    evidence.record_file(checkout).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def test_contract_is_the_flagship_quartet_and_keeps_archive_and_local_access_apart():
    contract = evidence.live_source_contract()
    assert [row["domain"] for row in contract["domains"]] == [
        "reanalysis", "argo_float", "tess_lightcurve", "order_book"]
    assert contract["public_network_domains"] == list(evidence.PUBLIC_NETWORK_DOMAINS)
    assert contract["local_record_domains"] == ["order_book"]
    by_domain = {row["domain"]: row for row in contract["domains"]}
    assert all(by_domain[name]["network_required_for_pass"]
               for name in evidence.PUBLIC_NETWORK_DOMAINS)
    assert by_domain["order_book"]["network_required_for_pass"] is False


def test_no_recording_is_not_run_and_reading_the_gate_uses_no_network(tmp_path):
    facts = evidence.read_live_source_evidence(tmp_path)
    assert facts["status"] == "NOT_RUN"
    assert facts["network_used_here"] is False
    assert "no four-domain" in facts["reasons"][0]


def test_a_complete_current_four_domain_record_passes(checkout):
    _record(checkout)
    facts = evidence.read_live_source_evidence(checkout)
    assert facts["status"] == "PASS", facts["reasons"]
    assert facts["reasons"] == []
    assert len(facts["recorded"]["outcomes"]) == 4


def test_a_partial_green_record_is_not_run(checkout):
    _record(checkout, lambda rows: rows[:-1])
    facts = evidence.read_live_source_evidence(checkout)
    assert facts["status"] == "NOT_RUN"
    assert "missing order_book" in " ".join(facts["reasons"])


def test_a_public_archive_pass_without_network_is_not_a_measurement(checkout):
    def weaken(rows):
        rows[0]["network_used"] = False
        return rows

    _record(checkout, weaken)
    facts = evidence.read_live_source_evidence(checkout)
    assert facts["status"] == "NOT_RUN"
    assert "reanalysis PASS does not demonstrate network use" in " ".join(facts["reasons"])


def test_the_bespoke_record_must_not_invent_an_archive(checkout):
    def weaken(rows):
        rows[-1]["network_used"] = True
        return rows

    _record(checkout, weaken)
    facts = evidence.read_live_source_evidence(checkout)
    assert facts["status"] == "NOT_RUN"
    assert "order_book PASS does not demonstrate no network use" in " ".join(facts["reasons"])


def test_source_drift_returns_a_readable_pass_to_not_run(checkout):
    _record(checkout)
    moved = checkout / evidence.SOURCE_FILES[0]
    moved.write_bytes(moved.read_bytes() + b"\n# changed acquisition semantics\n")
    facts = evidence.read_live_source_evidence(checkout)
    assert facts["status"] == "NOT_RUN"
    assert facts["recorded"] is not None
    assert evidence.SOURCE_FILES[0] in " ".join(facts["reasons"])


def test_operational_refusal_and_executed_failure_stay_distinct(checkout):
    def refused(rows):
        rows[1] = _outcome(evidence.live_source_contract()["domains"][1], status="REFUSED")
        return rows

    _record(checkout, refused)
    facts = evidence.read_live_source_evidence(checkout)
    assert facts["status"] == "REFUSED"
    assert "argo_float refused" in facts["reasons"][0]

    def failed(rows):
        rows[2] = _outcome(evidence.live_source_contract()["domains"][2], status="FAIL")
        return rows

    _record(checkout, failed)
    facts = evidence.read_live_source_evidence(checkout)
    assert facts["status"] == "FAIL"
    assert "tess_lightcurve failed" in facts["reasons"][0]


def test_a_pass_needs_dated_coverage_and_nonempty_content_identity(checkout):
    def undated(rows):
        rows[0]["finished_utc"] = "not-a-date"
        rows[1]["coverage"].pop("end")
        rows[2]["content"] = {"sha256": "short", "observed_values": 0}
        return rows

    _record(checkout, undated)
    facts = evidence.read_live_source_evidence(checkout)
    assert facts["status"] == "NOT_RUN"
    reasons = " ".join(facts["reasons"])
    assert "reanalysis does not record its exact measurement interval" in reasons
    assert "argo_float does not record its archive/record coverage" in reasons
    assert "tess_lightcurve PASS has no non-empty content-addressed record" in reasons


def test_the_release_ledger_reads_the_record_but_does_not_release_other_refusals(
        checkout, monkeypatch):
    from src.core.experiment_qualification import qualification_plan

    _record(checkout)
    monkeypatch.setattr(evidence, "REPO_ROOT", checkout)
    plan = qualification_plan()
    gate = {row["gate_id"]: row for row in plan["gates"]}["live_sources"]
    assert gate["status"] == "PASS"
    assert gate["blocking"] is False
    assert plan["live_source_evidence"]["network_used_here"] is False
    assert plan["verdict"] == "NOT_RELEASEABLE"


def test_the_operator_plan_freezes_small_requests_without_using_network(tmp_path):
    plan = evidence.live_source_plan(tmp_path)
    assert plan["network_used"] is False
    assert plan["network_requirements"]["both_required"] is True
    assert plan["resource_caps"] == {
        "reanalysis_values": 324,
        "argo_profiles": 200,
        "tess_products": 2,
        "tess_download_bytes": 64 * 1024 * 1024,
    }
    assert [row["request"] for row in plan["contract"]["domains"]]


def test_the_reporter_needs_two_permissions_before_calling_any_probe(
        checkout, tmp_path, monkeypatch):
    local = tmp_path / "record.csv"
    local.write_text("t,value\n0,1\n1,2\n", encoding="utf-8")
    calls = []

    def forbidden(*_args):
        calls.append(True)
        raise AssertionError("a probe ran before both permissions were present")

    probes = {domain: forbidden for domain in evidence.PUBLIC_NETWORK_DOMAINS
              + evidence.LOCAL_RECORD_DOMAINS}
    monkeypatch.delenv("SPECTRALEARTH_ALLOW_NETWORK", raising=False)
    with pytest.raises(PermissionError, match="confirm_network_access=True"):
        evidence.measure_live_sources(
            order_book_record=local, order_book_provenance="test fixture",
            order_book_licence="test-only", root=checkout, probes=probes)
    with pytest.raises(PermissionError, match="SPECTRALEARTH_ALLOW_NETWORK=1"):
        evidence.measure_live_sources(
            order_book_record=local, order_book_provenance="test fixture",
            order_book_licence="test-only", root=checkout,
            confirm_network_access=True, probes=probes)
    assert calls == []
    assert not evidence.record_file(checkout).exists()


def test_the_reporter_records_a_complete_fake_run_and_the_reader_decides(
        checkout, tmp_path, monkeypatch):
    local = tmp_path / "record.csv"
    local.write_text("t,value\n0,1\n1,2\n", encoding="utf-8")
    monkeypatch.setenv("SPECTRALEARTH_ALLOW_NETWORK", "1")

    def measured(declared, _root, _local):
        digest = (evidence._content_digest(_local) if declared["domain"] == "order_book"
                  else "b" * 64)
        return {
            "network_used": declared["network_required_for_pass"],
            "coverage": {"start": "2026-01-01T00:00:00Z",
                         "end": "2026-01-02T00:00:00Z", "exact": True},
            "content": {"sha256": digest, "observed_values": 2},
            "detail": {"bounded": True},
        }

    probes = {domain: measured for domain in evidence.PUBLIC_NETWORK_DOMAINS
              + evidence.LOCAL_RECORD_DOMAINS}
    record = evidence.measure_live_sources(
        order_book_record=local, order_book_provenance="test fixture",
        order_book_licence="test-only", root=checkout,
        confirm_network_access=True, probes=probes)
    assert len(record["record_sha256"]) == 64
    assert evidence.read_live_source_evidence(checkout)["status"] == "PASS"


def test_the_reporter_keeps_provider_refusal_and_implementation_failure_distinct(
        checkout, tmp_path, monkeypatch):
    from src.core.errors import InvalidParameterError

    local = tmp_path / "record.csv"
    local.write_text("t,value\n0,1\n1,2\n", encoding="utf-8")
    monkeypatch.setenv("SPECTRALEARTH_ALLOW_NETWORK", "1")

    def measured(declared, _root, _local):
        digest = (evidence._content_digest(_local) if declared["domain"] == "order_book"
                  else "c" * 64)
        return {"network_used": declared["network_required_for_pass"],
                "coverage": {"start": "a", "end": "b", "exact": True},
                "content": {"sha256": digest, "observed_values": 2}}

    probes = {domain: measured for domain in evidence.PUBLIC_NETWORK_DOMAINS
              + evidence.LOCAL_RECORD_DOMAINS}
    probes["argo_float"] = lambda *_: (_ for _ in ()).throw(
        InvalidParameterError("archive", "busy", "a bounded response"))
    record = evidence.measure_live_sources(
        order_book_record=local, order_book_provenance="test fixture",
        order_book_licence="test-only", root=checkout,
        confirm_network_access=True, probes=probes)
    argo = next(row for row in record["outcomes"] if row["domain"] == "argo_float")
    assert argo["network_attempted"] is True and argo["network_used"] is None
    assert evidence.read_live_source_evidence(checkout)["status"] == "REFUSED"

    probes["argo_float"] = measured
    probes["tess_lightcurve"] = lambda *_: (_ for _ in ()).throw(
        RuntimeError("parser invariant broke"))
    record = evidence.measure_live_sources(
        order_book_record=local, order_book_provenance="test fixture",
        order_book_licence="test-only", root=checkout,
        confirm_network_access=True, probes=probes)
    tess = next(row for row in record["outcomes"] if row["domain"] == "tess_lightcurve")
    assert tess["network_attempted"] is True and tess["network_used"] is None
    assert evidence.read_live_source_evidence(checkout)["status"] == "FAIL"


def test_the_plan_cli_is_network_dark_and_the_run_command_demands_an_exact_acknowledgement(
        capsys, monkeypatch):
    monkeypatch.delenv("SPECTRALEARTH_ALLOW_NETWORK", raising=False)
    assert evidence.main(["plan"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["network_used"] is False
    with pytest.raises(SystemExit) as stopped:
        evidence.main(["run", "--order-book-record", "missing.csv",
                       "--confirm-network-access", "yes"])
    assert stopped.value.code == 2


def test_local_measurement_identity_is_exact_bytes_not_source_style_line_normalisation(tmp_path):
    record = tmp_path / "record.csv"
    record.write_bytes(b"t,value\r\n0,1\r\n1,2\r\n")
    crlf = evidence.live_source_plan(order_book_record=record)["order_book_record"]["sha256"]
    record.write_bytes(b"t,value\n0,1\n1,2\n")
    lf = evidence.live_source_plan(order_book_record=record)["order_book_record"]["sha256"]
    assert crlf != lf


def test_editing_a_record_without_resealing_it_is_detected(checkout):
    record = _record(checkout)
    record["outcomes"][0]["content"]["observed_values"] += 1
    evidence.record_file(checkout).write_text(json.dumps(record), encoding="utf-8")
    facts = evidence.read_live_source_evidence(checkout)
    assert facts["status"] == "NOT_RUN"
    assert "record digest does not match" in " ".join(facts["reasons"])


def test_a_committed_fabricated_demo_cannot_satisfy_the_live_local_binding(
        checkout, monkeypatch):
    from src.core.errors import InvalidParameterError

    demo = checkout / "data" / "channels" / "demo.csv"
    demo.parent.mkdir(parents=True)
    demo.write_text("t,value\n0,1\n1,2\n", encoding="utf-8")
    monkeypatch.setenv("SPECTRALEARTH_ALLOW_NETWORK", "1")
    calls = []

    def forbidden(*_args):
        calls.append(True)

    with pytest.raises(InvalidParameterError, match="fabricated demonstrations"):
        evidence.measure_live_sources(
            order_book_record=demo, order_book_provenance="fabricated",
            order_book_licence="test-only", root=checkout, confirm_network_access=True,
            probes={domain: forbidden for domain in evidence.PUBLIC_NETWORK_DOMAINS
                    + evidence.LOCAL_RECORD_DOMAINS})
    assert calls == []
