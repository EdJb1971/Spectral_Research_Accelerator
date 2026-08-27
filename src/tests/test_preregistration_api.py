"""TG11.2: the generate/confirm split over HTTP, and the two ways of cheating it refuses."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.preregistration import (IDENTIFYING_PROVENANCE, ROOT_ENV, SEALED_RUN_KEYS,
                                     _specification)
from src.analysis_engine.cross_scale import GateProtocol
from src.core.builtin_domains import register_builtin_domains
from src.core.builtin_glossaries import ORDER_BOOK_PHRASES
from src.core.domain import DOMAIN_DECLARATIONS, AxisSpec, DomainDeclaration
from src.core.onboarding import DOMAIN_ONBOARDINGS, onboard_domain
from src.core.registry import restore, snapshot
from src.core.translation import DOMAIN_GLOSSARIES
from src.data_layer.tabular_source import VALUE_MEASURE


DOMAIN = "prereg_instrument_log"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A ledger per test. A test that spent the study's real partition would spend it."""
    monkeypatch.setenv(ROOT_ENV, str(tmp_path / "preregistrations"))
    before = (snapshot(DOMAIN_GLOSSARIES), snapshot(DOMAIN_DECLARATIONS),
              snapshot(DOMAIN_ONBOARDINGS))
    register_builtin_domains()
    onboard_domain(
        DomainDeclaration(
            name=DOMAIN, description="Regular synthetic instrument channels.",
            axes=(AxisSpec("t", "time", units="s"),
                  AxisSpec("channel", "category", ordered=False)),
            licence="Fixture licence; no real archive was accessed.",
            violations=("no_physical_metric", "no_propagation_speed", "no_natural_cycle"),
            lag_policy="declared", declared_floor_frames=1,
            declared_floor_basis="one instrument reporting interval",
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


def _upload(text: str, name: str = "record.csv"):
    return {"file": (name, io.BytesIO(text.encode("utf-8")), "text/csv")}


def _seal(client, text: str, *, study_id: str = "tg11.2", name: str = "record.csv",
          confirm_lags=(2,), generate_lags=(1, 2, 3, 4), n_surrogates: int = 99):
    return client.post(
        "/api/v1/preregistration/seal", files=_upload(text, name),
        data={"domain": DOMAIN, "time_column": "t", "study_id": study_id,
              "generate": json.dumps({"lags": list(generate_lags), "n_surrogates": 999}),
              "confirm": json.dumps({"lags": list(confirm_lags),
                                     "n_surrogates": n_surrogates}),
              "train_ratio": 0.6, "embargo_frames": 4})


# --------------------------------------------------------------------- the family shape


def test_the_sealed_family_is_the_shape_the_sweep_actually_emits():
    """A frozen label the sweep never emits cannot be confirmed, however well it was frozen.

    This pins the specification built at the HTTP boundary to the one the engine's own
    protocol builds. The two agreeing by construction is what makes `confirm_on_held_out`'s
    "extra label" refusal an invariant check rather than an obstacle to route around.
    """
    protocol = GateProtocol(study_id="shape", n_scales=3, lags=(1, 2, 3),
                            expected_frames=200, cadence_seconds=1.0, n_surrogates=99,
                            measure=VALUE_MEASURE)
    mine = _specification(channels=(1, 2, 3), lags=(1, 2, 3), n_surrogates=99, alpha=0.05,
                          correction="benjamini_yekutieli", study_id="shape", notes={})
    assert set(mine.labels()) == set(protocol.search_specification().labels())
    assert mine.family_size == protocol.family_size


# ------------------------------------------------------------------ the partition itself


def test_the_partition_identity_ignores_what_the_file_was_called(client):
    """D65: the ledger's "once" is once per held-out data, not once per filename.

    `PartitionIdentity.from_series` hashes provenance wholesale, and an upload's provenance
    carries its filename. Taking that identity unmodified would let the same held-out bytes be
    confirmed twice by renaming the file between requests, which is the whole of what the
    ledger exists to stop.
    """
    text = _record()
    first = client.post("/api/v1/preregistration/partition", files=_upload(text, "a.csv"),
                        data={"domain": DOMAIN, "time_column": "t", "train_ratio": 0.6,
                              "embargo_frames": 4})
    second = client.post("/api/v1/preregistration/partition", files=_upload(text, "b.csv"),
                         data={"domain": DOMAIN, "time_column": "t", "train_ratio": 0.6,
                               "embargo_frames": 4})
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["held_out"]["digest"] == second.json()["held_out"]["digest"]
    provenance = first.json()["held_out"]["provenance"]
    assert "path_basename" not in provenance and "adapter" not in provenance
    assert "domain" not in provenance
    assert set(IDENTIFYING_PROVENANCE) - {"cadence_seconds"} <= set(provenance)


def test_a_different_split_is_a_different_partition(client):
    """Two windows of one record are two partitions, and the seal names which one."""
    text = _record()
    digests = []
    for ratio in (0.5, 0.6):
        response = client.post("/api/v1/preregistration/partition", files=_upload(text),
                               data={"domain": DOMAIN, "time_column": "t",
                                     "train_ratio": ratio, "embargo_frames": 4})
        digests.append(response.json()["held_out"]["digest"])
    assert digests[0] != digests[1]


def test_describing_a_partition_neither_binds_nor_stores(client):
    text = _record()
    response = client.post("/api/v1/preregistration/partition", files=_upload(text),
                           data={"domain": DOMAIN, "time_column": "t", "train_ratio": 0.6,
                                 "embargo_frames": 4})
    body = response.json()
    assert body["stored"] is False and body["read_only"] is True
    assert body["already_opened"] is False
    assert client.get("/api/v1/preregistration/seals").json()["n_seals"] == 0


# ------------------------------------------------------------------------------ sealing


def test_sealing_freezes_a_narrower_family_and_times_it_by_the_server_clock(client):
    text = _record()
    response = _seal(client, text)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["confirm_family_size"] < body["generate_family_size"]
    assert body["records_evidence"] is False and body["rung_moved"] is False
    assert "server clock" in body["sealed_at_source"]
    settings = body["seal"]["confirm"]["notes"]
    assert not set(SEALED_RUN_KEYS) - set(settings)
    listed = client.get("/api/v1/preregistration/seals").json()
    assert listed["n_seals"] == 1 and listed["seals"][0]["spent"] is False


def test_a_confirmatory_lag_that_was_never_generated_is_refused(client):
    """Confirming a member that was never mined is a fresh search wearing a seal's name."""
    response = _seal(client, _record(), confirm_lags=(9,), generate_lags=(1, 2, 3))
    assert response.status_code == 400
    assert "never generated" in response.text or "generated family" in response.text


def test_the_stored_seal_is_not_offered_as_evidence_about_itself(client):
    """Self-consistency is not a binding; the published digest is the check with weight."""
    digest = _seal(client, _record()).json()["seal_sha256"]
    body = client.get("/api/v1/preregistration/seals/%s" % digest).json()
    assert body["verification"]["seal_sha256"] == digest
    assert body["checked_against_publication"] is False
    assert "cannot rewrite" in body["publication"]


def test_an_edited_seal_names_the_field_that_changed(client, tmp_path):
    """A seal that does not hash to its own contents is not evidence of when it was frozen."""
    digest = _seal(client, _record()).json()["seal_sha256"]
    path = tmp_path / "preregistrations" / ("%s.json" % digest)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["study_id"] = "rewritten-afterwards"
    path.write_text(json.dumps(stored), encoding="utf-8")

    response = client.get("/api/v1/preregistration/seals/%s" % digest)
    assert response.status_code == 400
    assert "study_id" in response.text
    assert client.post("/api/v1/preregistration/seals/%s/confirm" % digest,
                       files=_upload(_record())).status_code == 400


def test_a_seal_checked_against_the_wrong_published_digest_is_refused(client):
    digest = _seal(client, _record()).json()["seal_sha256"]
    response = client.get("/api/v1/preregistration/seals/%s" % digest,
                          params={"published_sha256": "0" * 64})
    assert response.status_code == 400
    assert "published" in response.text


# --------------------------------------------------------------------------- confirming


def test_confirmation_takes_every_setting_from_the_seal_and_spends_the_partition(client):
    """The caller supplies the record and nothing else, so nothing can be tuned afterwards."""
    text = _record()
    sealed = _seal(client, text).json()
    response = client.post("/api/v1/preregistration/seals/%s/confirm" % sealed["seal_sha256"],
                           files=_upload(text))
    assert response.status_code == 200, response.text
    body = response.json()
    receipt = body["receipt"]
    assert receipt["schema"] == "confirmation-receipt/v1"
    assert receipt["correction_unit"] == sealed["confirm_family_size"]
    assert sorted(receipt["labels"]) == sorted(sealed["confirm_labels"])
    assert receipt["generate_family_size"] == sealed["generate_family_size"]
    assert body["records_evidence"] is False and body["rung_moved"] is False
    assert "spent" in body["claim_boundary"]

    listed = client.get("/api/v1/preregistration/seals").json()["seals"][0]
    assert listed["spent"] is True and listed["spent_by"] == sealed["seal_sha256"]


def test_the_same_held_out_data_cannot_be_confirmed_twice_under_a_second_honest_seal(client):
    """The arithmetic R18 exists to stop: two seals over one partition are two tests of it.

    The second seal here is individually perfect -- correctly hashed, correctly narrowed,
    affordable at its own size. That is exactly why it must be refused: each confirmation
    would be defensible alone and the pair would be uncorrected.
    """
    text = _record()
    first = _seal(client, text, study_id="first", confirm_lags=(2,)).json()
    assert client.post("/api/v1/preregistration/seals/%s/confirm" % first["seal_sha256"],
                       files=_upload(text)).status_code == 200

    second = _seal(client, text, study_id="second", confirm_lags=(3,), name="renamed.csv")
    assert second.status_code == 409, second.text
    assert "has not been tested yet" in second.text or "opened at" in second.text


def test_two_seals_written_before_any_opening_still_buy_only_one_look(client):
    """The sharpest form of the attack, and the reason the ledger is keyed on the partition.

    Both seals here are frozen before anything is opened, so neither is post-hoc in any sense:
    no edit, no backdating, no narrowing chosen after a look. They are simply two declarations
    over one held-out partition, and testing it under both is two tests of the same data whose
    corrections were computed as though each were the only one.
    """
    text = _record()
    first = _seal(client, text, study_id="first", confirm_lags=(2,)).json()
    second = _seal(client, text, study_id="second", confirm_lags=(3,)).json()
    assert first["seal_sha256"] != second["seal_sha256"]
    assert first["held_out_digest"] == second["held_out_digest"]

    assert client.post("/api/v1/preregistration/seals/%s/confirm" % first["seal_sha256"],
                       files=_upload(text)).status_code == 200
    spent = client.post("/api/v1/preregistration/seals/%s/confirm" % second["seal_sha256"],
                        files=_upload(text))
    assert spent.status_code == 409, spent.text
    assert first["seal_sha256"] in spent.text


def test_a_confirmation_on_a_partition_the_seal_did_not_name_is_refused(client):
    """Choosing the partition after the declaration is the freedom the split removes."""
    sealed = _seal(client, _record()).json()
    other = _record(seed=999)
    response = client.post("/api/v1/preregistration/seals/%s/confirm" % sealed["seal_sha256"],
                           files=_upload(other))
    assert response.status_code == 400
    assert "did not name" in response.text or "bound itself to" in response.text


def test_a_refused_confirmation_does_not_spend_the_partition(client):
    """A design error must not cost the partition; only a completed confirmation may."""
    text = _record()
    sealed = _seal(client, text).json()
    assert client.post("/api/v1/preregistration/seals/%s/confirm" % sealed["seal_sha256"],
                       files=_upload(_record(seed=999))).status_code == 400
    assert client.get("/api/v1/preregistration/seals").json()["seals"][0]["spent"] is False
    assert client.post("/api/v1/preregistration/seals/%s/confirm" % sealed["seal_sha256"],
                       files=_upload(text)).status_code == 200


def test_confirming_against_an_unknown_seal_is_refused_before_anything_is_read(client):
    response = client.post("/api/v1/preregistration/seals/%s/confirm" % ("a" * 64),
                           files=_upload(_record()))
    assert response.status_code == 404
    assert "frozen before" in response.text


# ------------------------------------------------------------------- the surface itself


def test_the_capabilities_state_what_once_is_measured_over(client):
    body = client.get("/api/v1/preregistration").json()
    assert body["records_evidence"] is False and body["moves_rung"] is False
    assert "held-out data" in body["once_is_per"]
    assert "Not the filename" in body["once_is_per"]
    assert "promise about ordering" in body["claim_boundary"]


def test_the_analysis_surface_refuses_a_gate_on_a_spent_partition(client):
    """TG11.1's gate splits internally, so running it on spent data is a second test of it."""
    text = _record()
    sealed = _seal(client, text).json()
    assert client.post("/api/v1/preregistration/seals/%s/confirm" % sealed["seal_sha256"],
                       files=_upload(text)).status_code == 200

    response = client.post(
        "/api/v1/analysis/run", files=_upload(text),
        data={"operation": "domain_gate", "domain": DOMAIN, "time_column": "t",
              "configuration": json.dumps({"lags": [2], "n_surrogates": 99,
                                           "train_ratio": 0.6, "embargo_frames": 4})})
    assert response.status_code == 409, response.text
    assert "held-out" in response.text or "opened at" in response.text
