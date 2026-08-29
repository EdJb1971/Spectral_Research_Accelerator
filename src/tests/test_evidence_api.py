"""TG11.3: the evidence write path, and the two ways a client could climb a rung by typing."""

from __future__ import annotations

import csv
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.findings import STUDY_ROOT_ENV
from src.core.builtin_domains import register_builtin_domains
from src.core.builtin_glossaries import ORDER_BOOK_PHRASES
from src.core.claim_ladder import PRECEDENCE_KEY
from src.core.domain import DOMAIN_DECLARATIONS, AxisSpec, DomainDeclaration
from src.core.evidence import EVIDENCE_FIELDS
from src.core.onboarding import DOMAIN_ONBOARDINGS, onboard_domain
from src.core.registry import restore, snapshot
from src.core.translation import DOMAIN_GLOSSARIES


DOMAIN = "evidence_instrument_log"
ASSOCIATION_ONLY = "evidence_association_only"
BASE = "/api/v1/evidence"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A study store per test. A test that wrote into the real one would publish evidence."""
    monkeypatch.setenv(STUDY_ROOT_ENV, str(tmp_path / "studies"))
    before = (snapshot(DOMAIN_GLOSSARIES), snapshot(DOMAIN_DECLARATIONS),
              snapshot(DOMAIN_ONBOARDINGS))
    register_builtin_domains()
    for name, policy in ((DOMAIN, "declared"), (ASSOCIATION_ONLY, "none")):
        onboard_domain(
            DomainDeclaration(
                name=name, description="Regular synthetic instrument channels.",
                axes=(AxisSpec("t", "time", units="s"),
                      AxisSpec("channel", "category", ordered=False)),
                licence="Fixture licence; no real archive was accessed.",
                violations=("no_physical_metric", "no_propagation_speed", "no_natural_cycle"),
                lag_policy=policy,
                declared_floor_frames=1 if policy == "declared" else None,
                declared_floor_basis=("one instrument reporting interval"
                                      if policy == "declared" else None),
                provenance={"declared_by": __name__}),
            ORDER_BOOK_PHRASES, geometry=None, onboarded_by=__name__)
    try:
        yield
    finally:
        restore(DOMAIN_GLOSSARIES, before[0])
        restore(DOMAIN_DECLARATIONS, before[1])
        restore(DOMAIN_ONBOARDINGS, before[2])


@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app)


def _record(n: int = 240, seed: int = 20260827) -> str:
    rng = np.random.default_rng(seed)
    alpha = rng.normal(size=n)
    beta = rng.normal(size=n)
    beta[2:] = alpha[:-2] + 0.4 * beta[2:]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["t", "alpha", "beta"])
    for i in range(n):
        writer.writerow([i, float(alpha[i]), float(beta[i])])
    return output.getvalue()


def _open(client, study_id="tg11-3", statement="alpha leads beta at two frames",
          prediction="beta is predictable from alpha at lag two"):
    return client.post(BASE + "/studies", json={
        "study_id": study_id, "hypothesis_id": "H1", "statement": statement,
        "prediction": prediction})


def _head(client, study_id="tg11-3") -> str:
    response = client.get("%s/studies/%s/head" % (BASE, study_id))
    assert response.status_code == 200, response.text
    return response.json()["head_sha256"]


def _append(client, category, *, study_id="tg11-3", status="PASS", payload=None,
            head=None, label=None, summary="recorded by a test"):
    return client.post("%s/studies/%s/evidence" % (BASE, study_id), json={
        "expected_head_sha256": head if head is not None else _head(client, study_id),
        "category": category, "label": label or ("%s entry" % category), "status": status,
        "summary": summary, "payload": payload or {"value": 1.0}})


def _precedence(client, text, *, study_id="tg11-3", domain=DOMAIN, n_surrogates=199,
                lags="2", head=None, label="server-computed precedence"):
    return client.post(
        "%s/studies/%s/evidence/precedence" % (BASE, study_id),
        files={"file": ("record.csv", io.BytesIO(text.encode("utf-8")), "text/csv")},
        data={"expected_head_sha256": head if head is not None else _head(client, study_id),
              "label": label, "domain": domain, "time_column": "t", "lags": lags,
              "n_surrogates": n_surrogates, "alpha": 0.05, "bins": 4})


# ------------------------------------------------------------------ opening and appending


def test_opening_a_study_writes_revision_zero_and_claims_nothing(client):
    response = _open(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["revision"] == 0
    assert body["ladder"]["rung"] == "observation"
    assert body["hypothesis"]["statement"] == "alpha leads beta at two frames"
    # A registered hypothesis with no evidence under it is the bottom of the ladder, not a
    # pending one: `observation` is what an empty chain earns.
    assert body["ladder"]["blocking_entries"] == []


def test_a_second_study_under_one_identifier_is_refused(client):
    assert _open(client).status_code == 200
    again = _open(client, statement="a different hypothesis entirely")
    assert again.status_code == 409
    assert "already exists" in again.json()["detail"]


def test_a_study_identifier_that_could_traverse_a_directory_is_refused(client):
    response = _open(client, study_id="../../escaped")
    assert response.status_code in (400, 422)
    assert "study_id" in response.text


def test_an_append_extends_the_head_it_names_and_the_chain_records_the_link(client):
    _open(client)
    anchor = _head(client)
    first = _append(client, "observations")
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["revision"] == 1
    assert body["entry"]["previous_sha256"] == anchor
    assert body["entry"]["sequence"] == 1
    assert body["head_sha256"] == body["entry"]["entry_sha256"]


def test_an_append_naming_a_head_the_chain_has_moved_past_is_refused(client):
    _open(client)
    stale = _head(client)
    assert _append(client, "observations").status_code == 200
    losing = _append(client, "effect_sizes", head=stale)
    assert losing.status_code == 409
    assert "moved under this request" in losing.json()["detail"]
    # And the loser wrote nothing: the chain is where the winner left it.
    after = client.get(BASE + "/studies/tg11-3/head").json()
    assert after["revision"] == 1
    assert len(after["entries"]) == 1


def test_an_earlier_revision_is_kept_rather_than_rewritten(client, tmp_path):
    _open(client)
    _append(client, "observations")
    _append(client, "effect_sizes")
    files = sorted(path.name for path in (tmp_path / "studies").glob("*.json"))
    assert files == ["tg11-3.r00000.json", "tg11-3.r00001.json", "tg11-3.r00002.json"]


def test_the_read_surface_serves_the_latest_revision_of_a_study(client):
    """D66: revision-per-file would otherwise have frozen the read surface at revision zero."""
    _open(client)
    _append(client, "observations")
    _append(client, "effect_sizes")
    served = client.get("/api/v1/findings/studies/tg11-3")
    assert served.status_code == 200, served.text
    assert served.json()["bundle"]["revision"] == 2
    assert served.json()["file"] == "tg11-3.r00002.json"

    listed = client.get("/api/v1/findings/studies").json()
    rows = [row for row in listed if row.get("study_id") == "tg11-3"]
    assert len(rows) == 1, "three revisions of one study are not three studies"
    assert rows[0]["revision"] == 2
    assert rows[0]["superseded_revisions"] == 2


# ------------------------------------------------------------- the rung is never accepted


def test_a_request_carrying_a_rung_is_refused_rather_than_ignored(client):
    _open(client)
    response = client.post(BASE + "/studies/tg11-3/evidence", json={
        "expected_head_sha256": _head(client), "category": "observations", "label": "x",
        "status": "PASS", "summary": "s", "payload": {"value": 1.0},
        "rung": "candidate_precursor"})
    assert response.status_code == 422, response.text
    assert "rung" in response.text


def test_a_payload_asserting_a_rung_is_refused_at_any_depth(client):
    _open(client)
    response = _append(client, "observations",
                       payload={"nested": {"claim_level": "robust_association"}})
    assert response.status_code in (400, 422)
    detail = response.json()["detail"]
    assert "recomputed" in detail and "R22" in detail


def test_a_payload_asserting_temporal_precedence_is_refused_and_names_the_route(client):
    """The one key in the ladder that a free-form payload could otherwise reach."""
    _open(client)
    response = _append(client, "provenance", payload={PRECEDENCE_KEY: True})
    assert response.status_code in (400, 422)
    detail = response.json()["detail"]
    assert "evidence/precedence" in detail
    assert "candidate_precursor" in detail


def test_the_rung_moves_because_the_evidence_moved_it(client):
    _open(client)
    for category in ("observations", "effect_sizes", "uncertainty"):
        response = _append(client, category)
        assert response.status_code == 200, response.text
    body = response.json()
    assert body["ladder"]["rung"] == "association"
    assert body["ladder"]["unblocked_rung"] == "association"
    assert "recomputed" in body["rung_source"]


def test_one_failed_entry_caps_the_chain_at_observation_through_the_wire(client):
    _open(client)
    for category in ("observations", "effect_sizes", "uncertainty"):
        assert _append(client, category).status_code == 200
    response = _append(client, "replication_results", status="FAIL")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ladder"]["rung"] == "observation"
    assert body["ladder"]["blocking_entries"] == [4]
    assert body["blocked"] is True


def test_commentary_has_no_category_here(client):
    _open(client)
    response = _append(client, "commentary")
    assert response.status_code in (400, 422)
    detail = response.json()["detail"]
    assert "Commentary is not scientific evidence" in detail
    assert all(field in detail for field in EVIDENCE_FIELDS)


def test_a_bare_confidence_cannot_be_recorded(client):
    """R9 on the way in, so it is refused by name rather than by a 500 on the way out."""
    _open(client)
    response = _append(client, "effect_sizes", payload={"confidence": 0.91})
    assert response.status_code in (400, 422)
    assert "base_rate" in response.json()["detail"]


def test_the_recorded_time_is_the_servers(client):
    _open(client)
    body = _append(client, "observations").json()
    assert body["entry"]["recorded_at"].endswith("+00:00")
    # An append cannot state when it happened: the field is not on the model, so a request
    # that back-dates an entry is refused rather than believed.
    backdated = client.post(BASE + "/studies/tg11-3/evidence", json={
        "expected_head_sha256": _head(client), "category": "observations", "label": "x",
        "status": "PASS", "summary": "s", "payload": {"value": 1.0},
        "recorded_at": "1999-01-01T00:00:00+00:00"})
    assert backdated.status_code == 422


def test_a_causally_worded_hypothesis_is_registered_with_the_ceiling_stated(client):
    response = _open(client, statement="alpha causes beta to rise")
    assert response.status_code == 200, response.text
    note = response.json()["wording_note"]
    assert note is not None and "R7" in note and "causes" in note


# ------------------------------------------------------------------ the computed verdict


def test_the_precedence_verdict_is_computed_here_and_not_supplied(client):
    _open(client)
    response = _precedence(client, _record())
    assert response.status_code == 200, response.text
    body = response.json()
    entry = body["entry"]
    assert entry["category"] == "provenance"
    assert entry["status"] == "PASS"
    assert entry["payload"][PRECEDENCE_KEY] is True
    assert entry["payload"]["claim_boundary"] == "precedence"
    assert body["computed_here"] is True
    # And it is the ladder, not the route, that decides what the entry is worth.
    assert body["ladder"]["rung"] == "observation"
    assert "candidate_precursor.temporal_precedence_recorded" not in body["unsatisfied_gates"]


def test_the_entry_cites_the_bytes_and_the_configuration_it_was_computed_from(client):
    _open(client)
    entry = _precedence(client, _record()).json()["entry"]
    sources = entry["source_sha256s"]
    assert len(sources) == 2 and len(set(sources)) == 2
    assert entry["payload"]["record_sha256"] in sources
    assert entry["payload"]["analysis_config_sha256"] in sources


def test_an_underpowered_sweep_is_recorded_as_inconclusive_rather_than_as_a_negative(client):
    """R5: a sweep that could not have rejected anything did not check anything."""
    _open(client)
    body = _precedence(client, _record(), n_surrogates=9).json()
    entry = body["entry"]
    assert entry["status"] == "INCONCLUSIVE"
    assert entry["payload"]["can_reject_after_correction"] is False
    # An INCONCLUSIVE provenance entry opens neither the provenance gate nor the precedence one.
    assert "candidate_precursor.provenance_auditable" in body["unsatisfied_gates"]


def test_a_domain_with_no_admissible_lag_floor_writes_nothing(client):
    _open(client)
    response = _precedence(client, _record(), domain=ASSOCIATION_ONLY)
    assert response.status_code in (400, 422), response.text
    assert client.get(BASE + "/studies/tg11-3/head").json()["revision"] == 0


def test_a_precedence_append_naming_a_stale_head_is_refused_before_it_computes(client):
    _open(client)
    stale = _head(client)
    assert _append(client, "observations").status_code == 200
    response = _precedence(client, _record(), head=stale)
    assert response.status_code == 409
    assert client.get(BASE + "/studies/tg11-3/head").json()["revision"] == 1


def test_the_capabilities_say_what_is_computed_rather_than_accepted(client):
    body = client.get(BASE).json()
    assert body["accepts_a_rung"] is False
    assert body["rung_is_computed"] is True
    assert PRECEDENCE_KEY in body["computed_not_accepted"]
    assert list(body["categories"]) == list(EVIDENCE_FIELDS)
    assert "held-out ledger" in body["claim_boundary"]
