"""TG11.4: the structure-mining surface, and the two things it must never accept.

The acceptance for this slice is not that a motif can be found. It is that a motif cannot be
*asserted*. Two tests carry that weight and the rest support them.

`test_no_route_on_this_surface_accepts_a_feature` is the R22-shaped one. A motif is a
configuration of extracted features, so a boundary that took feature coordinates would let a
caller type the shape it wanted and have the programme confirm it on held-out data - with
arithmetic everywhere and nothing for an estimator to notice, because every number afterwards
would be computed correctly from a fabricated premise. The check is structural: no request
model on this surface has a field for a coordinate, a feature or a graph.

`test_the_tolerance_cannot_be_typed` is the second. The match tolerance decides which
configurations count as repeats, and this tree's own benchmark measured a tolerance built the
tempting way at fifty times too wide, at which point every triangle matched every other. It is
therefore calibrated by the server and passed by digest.

The scenes here are planted geometry rather than reanalysis frames: a scalene triangle at a
random rotation and translation in every frame, plus distractors placed under the same minimum
separation the surrogate null draws with. The null record is the same generator with the
triangle removed and the same number of features left in place, so a pass over it examines
exactly the same family and the only thing missing is the repetition.
"""

import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.mining import SEALED_MINING_KEYS, _split
from src.api.preregistration import ROOT_ENV as PREREGISTRATION_ROOT_ENV
from src.api.mining import ROOT_ENV as MINING_ROOT_ENV
from src.core.channel_series import ChannelSeries, split_channel_series
from src.core.builtin_domains import register_builtin_domains
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.family import SearchAxis, SearchSpecification, SearchTerm
from src.core.builtin_glossaries import REANALYSIS_PHRASES
from src.core.domain import DOMAIN_DECLARATIONS
from src.core.onboarding import DOMAIN_ONBOARDINGS, onboard_domain
from src.core.registry import restore, snapshot
from src.core.translation import DOMAIN_GLOSSARIES
from src.core.preregistration import PartitionIdentity, freeze_confirmatory_family


N = 48
SIGMA = 1.8
SIDE = 9.0
ARMS = (0.62, 1.0, 1.45)
MIN_SEPARATION = 7.0
FEATURES_PER_FRAME = 5
EXTRACTION = {"n_surrogates": 29, "seed": 7}


# ------------------------------------------------------------------------------- the fields


def _blob(row: float, col: float) -> np.ndarray:
    y = np.arange(N)[:, None]
    x = np.arange(N)[None, :]
    return np.exp(-((y - row) ** 2 + (x - col) ** 2) / (2 * SIGMA ** 2))


def _frame(positions, seed: int, noise: float = 0.02) -> np.ndarray:
    values = np.zeros((N, N), dtype=np.float64)
    for row, col in positions:
        values = values + _blob(row, col)
    return values + noise * np.random.default_rng(seed).standard_normal((N, N))


def _triangle(centre, rotation_deg: float):
    row, col = centre
    return [(row + SIDE * arm * math.sin(math.radians(rotation_deg + 120.0 * index)),
             col + SIDE * arm * math.cos(math.radians(rotation_deg + 120.0 * index)))
            for index, arm in enumerate(ARMS)]


def _positions(seed: int, *, plant: bool = True, n_features: int = FEATURES_PER_FRAME):
    """A configuration and its distractors, placed no closer than the null can place them."""
    rng = np.random.default_rng(seed)
    positions = []
    if plant:
        positions += _triangle((rng.uniform(20, 28), rng.uniform(20, 28)),
                               rng.uniform(0.0, 360.0))
    for _ in range(4000):
        if len(positions) >= n_features:
            break
        candidate = (float(rng.uniform(8, N - 8)), float(rng.uniform(8, N - 8)))
        if all(math.hypot(candidate[0] - row, candidate[1] - col) >= MIN_SEPARATION
               for row, col in positions):
            positions.append(candidate)
    assert len(positions) == n_features
    return positions


def _npy(frames) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.stack(frames), allow_pickle=False)
    return buffer.getvalue()


def _stack(seeds, *, plant: bool = True, n_features: int = FEATURES_PER_FRAME) -> bytes:
    return _npy([_frame(_positions(seed, plant=plant, n_features=n_features), 2000 + seed)
                 for seed in seeds])


def _replicate_stack(seed: int = 77, n: int = 4) -> bytes:
    """One configuration under `n` noise realisations - what a tolerance is measured from."""
    positions = _positions(seed)
    return _npy([_frame(positions, 900 + index) for index in range(n)])


# ------------------------------------------------------------------------------- the client


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Both stores under the test's own directory.

    The mining root and the preregistration root are separated because a test that spent a
    real held-out partition would spend it permanently, and a test that committed a transfer
    target would make that target unusable for the researcher who owns it.
    """
    monkeypatch.setenv(MINING_ROOT_ENV, str(tmp_path / "mining"))
    monkeypatch.setenv(PREREGISTRATION_ROOT_ENV, str(tmp_path / "preregistrations"))
    before = (snapshot(DOMAIN_GLOSSARIES), snapshot(DOMAIN_DECLARATIONS),
              snapshot(DOMAIN_ONBOARDINGS))
    register_builtin_domains()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        # A domain onboarded for a transfer test must not survive into another test's
        # catalogue: a registry that leaks between tests makes one test's fixture another
        # test's evidence.
        restore(DOMAIN_GLOSSARIES, before[0])
        restore(DOMAIN_DECLARATIONS, before[1])
        restore(DOMAIN_ONBOARDINGS, before[2])


def _admit(client, payload: bytes, *, domain: str = "reanalysis", dataset: str = "planted",
           variable: str = "amplitude", representation: str = "identity",
           extraction=None) -> dict:
    response = client.post(
        "/api/v1/mining/records",
        files={"file": ("frames.npy", payload, "application/octet-stream")},
        data={"domain": domain, "dataset": dataset, "variable": variable,
              "representation": representation, "time_units": "s",
              "extraction": json.dumps(dict(EXTRACTION, **(extraction or {})))})
    return response


def _tolerance(client, record_id: str) -> str:
    response = client.post("/api/v1/mining/tolerance", json={
        "record_id": record_id, "frames": [0, 1, 2, 3],
        "declared_as_replicates_of": "one planted triangle under four noise realisations"})
    assert response.status_code == 200, response.text
    return response.json()["tolerance_sha256"]


def _run_body(record_id: str, tolerance_sha256: str, **overrides) -> dict:
    body = {"record_id": record_id, "tolerance_sha256": tolerance_sha256, "size": 3,
            "n_surrogates": 199, "train_ratio": 0.5, "embargo_frames": 0,
            "study_id": "tg11-4"}
    body.update(overrides)
    return body


@pytest.fixture
def planted(client):
    """A twelve-frame record with a triangle in every frame, and its tolerance."""
    record = _admit(client, _stack(range(100, 112)))
    assert record.status_code == 200, record.text
    replicates = _admit(client, _replicate_stack(), dataset="planted")
    assert replicates.status_code == 200, replicates.text
    return {"record_id": record.json()["record_id"],
            "record": record.json(),
            "tolerance_sha256": _tolerance(client, replicates.json()["record_id"])}


# ================================================================= capabilities and pricing


def test_capabilities_names_every_matcher_and_no_rung(client):
    body = client.get("/api/v1/mining").json()
    assert body["moves_rung"] is False
    matchers = {entry["name"]: entry for entry in body["matchers"]}
    assert matchers["relative_geometry"]["declared_invariance"] == [
        "rotation", "translation", "rescaling"]
    assert matchers["absolute_position"]["declared_invariance"] == []
    assert "reanalysis" in body["minable_domains"]
    assert "order_book" not in body["minable_domains"]
    assert "selection" in body["claim_boundary"]


def test_the_price_of_a_mining_pass_is_computable_before_anything_is_mined(client):
    body = client.post("/api/v1/mining/price", json={
        "n_scenes": 6, "n_features": 6, "size": 3, "n_surrogates": 199}).json()
    assert body["generate"]["family_size"] == 6 * 20
    assert body["generate"]["affordable"] is False
    assert body["split_is_not_optional"] is True
    # The refusal is arithmetic and it is stated with the arithmetic in it.
    assert body["generate"]["surrogates_required"] > 199
    assert body["confirmatory_ceiling"]["max_affordable_family"] >= 1


def test_the_price_agrees_with_the_family_the_mining_pass_actually_searches(client, planted):
    """`/price` answers before a field exists, so it builds the family a different way.

    Two constructions of one number is exactly where a priced family and a searched family
    drift apart without anyone noticing, and a correction unit that is not the number of
    tests is not a correction. They are asserted equal against a real pass.
    """
    generated = client.post("/api/v1/mining/generate",
                            json=_run_body(planted["record_id"],
                                           planted["tolerance_sha256"])).json()
    scenes = len(generated["mining"]["scenes"])
    priced = client.post("/api/v1/mining/price", json={
        "n_scenes": scenes, "n_features": FEATURES_PER_FRAME, "size": 3,
        "n_surrogates": 199}).json()
    assert priced["generate"]["family_size"] == generated["mining"]["generate_family_size"]
    assert generated["mining"]["n_examined"] == priced["generate"]["family_size"]
    assert priced["generate"]["affordable"] is generated["mining"]["affordable_in_one_stage"]


def test_a_configuration_size_this_surface_cannot_price_is_refused(client):
    response = client.post("/api/v1/mining/price", json={
        "n_scenes": 6, "n_features": 8, "size": 5})
    assert response.status_code == 400
    assert "size" in response.json()["detail"]


# ============================================================================= the admission


def test_a_record_is_addressed_by_its_bytes_and_its_declaration(client):
    payload = _stack(range(100, 106))
    first = _admit(client, payload).json()
    again = _admit(client, payload).json()
    relabelled = _admit(client, payload, variable="vorticity").json()
    assert first["record_id"] == again["record_id"]
    assert relabelled["record_id"] != first["record_id"]
    assert first["declaration"]["content_sha256"] == relabelled["declaration"]["content_sha256"]


def test_the_extractor_finds_the_planted_features_in_every_frame(client, planted):
    record = planted["record"]
    assert record["feature_counts"] == [FEATURES_PER_FRAME]
    assert record["minable"] is True
    assert [frame["n_features"] for frame in record["frames"]] == [FEATURES_PER_FRAME] * 12


def test_an_admitted_record_reports_counts_and_no_measurement(client, planted):
    """The split is declared after admission, so a per-frame measurement would be on screen
    while the split was being chosen. Counts and rejection tallies are geometry; a calibrated
    noise floor is a measurement of the frame it came from."""
    for row in planted["record"]["frames"]:
        assert sorted(row) == ["frame", "n_features", "rejected", "scene"]


def test_a_single_frame_is_not_a_record(client):
    buffer = io.BytesIO()
    np.save(buffer, _frame(_positions(1), 1), allow_pickle=False)
    response = _admit(client, buffer.getvalue())
    assert response.status_code == 400
    assert "stack of frames" in response.json()["detail"]


def test_a_pickled_array_is_refused_rather_than_loaded(client):
    buffer = io.BytesIO()
    np.save(buffer, np.array([{"payload": "arbitrary"}], dtype=object), allow_pickle=True)
    response = _admit(client, buffer.getvalue())
    assert response.status_code == 400
    assert "executable content" in response.json()["detail"]


def test_a_domain_with_no_spatial_extent_cannot_hold_a_configuration(client):
    response = _admit(client, _stack(range(100, 104)), domain="order_book")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "two spatial axes" in detail and "order_book" in detail


def test_frames_that_disagree_about_how_many_features_they_hold_are_refused_not_trimmed(client):
    """The tempting repair is named in the refusal, which is why it is not offered."""
    frames = [_frame(_positions(seed), 2000 + seed) for seed in range(100, 106)]
    # Inside the training window, so the refusal is the one the mining pass itself makes.
    frames[1] = _frame(_positions(500, n_features=FEATURES_PER_FRAME + 2), 2500)
    record = _admit(client, _npy(frames)).json()
    assert record["minable"] is False
    assert "trims" in record["why_not_minable"] or "trims" in record["note"]

    replicates = _admit(client, _replicate_stack()).json()
    tolerance = _tolerance(client, replicates["record_id"])
    response = client.post("/api/v1/mining/generate",
                           json=_run_body(record["record_id"], tolerance))
    assert response.status_code == 400
    assert "brightest" in response.json()["detail"]


def test_a_record_whose_declaration_was_edited_is_refused_by_its_own_digest(client, planted):
    root = Path(os.environ[MINING_ROOT_ENV]) / "records"
    path = root / ("%s.json" % planted["record_id"])
    declaration = json.loads(path.read_text(encoding="utf-8"))
    declaration["variable"] = "something_else"
    path.write_text(json.dumps(declaration), encoding="utf-8")

    response = client.post("/api/v1/mining/generate",
                           json=_run_body(planted["record_id"], planted["tolerance_sha256"]))
    assert response.status_code == 409
    assert "edited after the record was admitted" in response.json()["detail"]


# ================================================== the two things that cannot be asserted


def test_no_route_on_this_surface_accepts_a_feature(client):
    """The R22-shaped hole of this slice, closed structurally rather than by review."""
    import inspect

    from pydantic import BaseModel

    from src.api import mining

    forbidden = ("feature", "coordinate", "coords", "graph", "signature", "position",
                 "occurrence", "support", "p_value", "rung", "scene", "tolerance_value")
    offenders = []
    models = [value for _name, value in inspect.getmembers(mining, inspect.isclass)
              if issubclass(value, BaseModel) and value is not BaseModel]
    assert len(models) >= 8, "the request models moved; this check would pass vacuously"
    for model in models:
        for field in model.__fields__:
            # `n_features` and `n_scenes` are counts of things the server found or will
            # look at. A count cannot carry a shape; a coordinate can.
            if field.startswith("n_"):
                continue
            if any(word in field.lower() for word in forbidden):
                offenders.append("%s.%s" % (model.__name__, field))
    assert offenders == [], (
        "these request fields would let a caller supply what the server is supposed to "
        "measure: %s" % offenders)


def test_a_request_carrying_an_unknown_setting_is_refused_rather_than_ignored(client, planted):
    response = client.post("/api/v1/mining/generate", json=_run_body(
        planted["record_id"], planted["tolerance_sha256"], keep_only_the_brightest=3))
    assert response.status_code == 422


def test_the_tolerance_cannot_be_typed(client, planted):
    """A number instead of a measurement is refused by the request model itself."""
    response = client.post("/api/v1/mining/generate", json={
        "record_id": planted["record_id"], "tolerance": 0.4, "size": 3})
    assert response.status_code == 422
    body = response.text
    assert "tolerance_sha256" in body


def test_a_tolerance_receipt_says_whose_claim_the_replicates_are(client, planted):
    receipt = client.post("/api/v1/mining/tolerance", json={
        "record_id": planted["record_id"], "frames": [0, 1],
        "declared_as_replicates_of": "not actually replicates"}).json()
    assert "cannot be checked from the numbers" in receipt["note"]
    assert receipt["receipt"]["declared_as_replicates_of"] == "not actually replicates"
    assert receipt["receipt"]["tolerance"]["n_replicates"] == 2
    assert "biased in the direction that finds no motif" in receipt["claim_boundary"]


def test_one_replicate_is_not_a_calibration(client, planted):
    response = client.post("/api/v1/mining/tolerance", json={
        "record_id": planted["record_id"], "frames": [0],
        "declared_as_replicates_of": "one frame"})
    assert response.status_code == 400
    assert "agrees with itself exactly" in response.json()["detail"]


def test_a_tolerance_measured_on_another_pipeline_is_refused(client, planted):
    other = _admit(client, _replicate_stack(), variable="vorticity").json()
    tolerance = _tolerance(client, other["record_id"])
    response = client.post("/api/v1/mining/generate",
                           json=_run_body(planted["record_id"], tolerance))
    assert response.status_code == 400
    assert "pipeline" in response.json()["detail"]


def test_a_tolerance_calibrated_on_held_out_frames_is_refused(client, planted):
    """Calibrating past the boundary lets the held-out frames set what counts as a repeat."""
    tolerance = client.post("/api/v1/mining/tolerance", json={
        "record_id": planted["record_id"], "frames": [8, 9, 10],
        "declared_as_replicates_of": "held-out frames"}).json()["tolerance_sha256"]
    response = client.post("/api/v1/mining/generate",
                           json=_run_body(planted["record_id"], tolerance))
    assert response.status_code == 400
    assert "held-out" in response.json()["detail"]


# ==================================================================== the split and the mine


def test_frames_are_split_by_the_rule_rows_are_split_by(client):
    """R6, applied to frames, asserted against the implementation it mirrors."""
    for n_frames, ratio, embargo in ((12, 0.5, 0), (12, 0.6, 2), (20, 0.75, 3)):
        series = ChannelSeries(
            channels=["a", "b"],
            times_seconds=np.arange(n_frames, dtype=np.float64),
            measures={"value": np.zeros((n_frames, 2))})
        rows = split_channel_series(series, train_ratio=ratio, embargo_frames=embargo)
        frames = _split(n_frames, ratio, embargo)
        assert frames["train"] == tuple(rows["train"].provenance["split_frames"])
        assert frames["held_out"] == tuple(rows["test"].provenance["split_frames"])


def test_generation_produces_candidates_and_says_they_are_not_findings(client, planted):
    body = client.post("/api/v1/mining/generate",
                       json=_run_body(planted["record_id"],
                                      planted["tolerance_sha256"])).json()
    assert body["mining"]["affordable_in_one_stage"] is False
    assert body["mining"]["generate_family_size"] == 6 * 10
    assert body["chosen"], "the planted triangle should recur across the training frames"
    assert body["chosen"][0]["support"] >= 4
    assert "selection" in body["claim_boundary"]
    assert body["generation_report"]["p_values_uncorrected"] is None


def test_generation_returns_the_held_out_geometry_and_nothing_measured_in_it(client, planted):
    body = client.post("/api/v1/mining/generate",
                       json=_run_body(planted["record_id"],
                                      planted["tolerance_sha256"])).json()
    held = body["held_out_identity"]
    assert held["frames"] == [6, 12] and held["n_times"] == 6
    assert held["n_channels"] == FEATURES_PER_FRAME
    # Every scene named anywhere in this response is a training scene.
    named = json.dumps(body)
    for index in range(6, 12):
        assert "frame%04d|" % index not in named
    assert body["held_out_spent"] is None


# ================================================================= freezing and confirming


def _seal(client, planted, **overrides):
    response = client.post("/api/v1/mining/freeze",
                           json=_run_body(planted["record_id"], planted["tolerance_sha256"],
                                          **overrides))
    assert response.status_code == 200, response.text
    return response.json()


def test_a_seal_carries_every_setting_the_confirmation_needs(client, planted):
    body = _seal(client, planted)
    notes = body["seal"]["confirm"]["notes"]
    for key in SEALED_MINING_KEYS:
        assert key in notes, key
    assert notes["tolerance"] > 0.0
    assert body["frozen_labels"]
    assert "records no evidence" in body["claim_boundary"]


def test_a_mining_seal_is_stored_where_every_other_seal_is(client, planted):
    body = _seal(client, planted)
    listed = client.get("/api/v1/preregistration/seals").json()
    digests = [seal["seal_sha256"] for seal in listed["seals"]]
    assert body["seal_sha256"] in digests


def test_confirmation_takes_a_seal_and_nothing_that_could_change_the_analysis(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ConfirmRequest"]
    assert sorted(schema["properties"]) == ["published_sha256", "seal_sha256"]


def test_the_planted_motif_is_confirmed_on_frames_it_was_not_mined_from(client, planted):
    seal = _seal(client, planted)
    body = client.post("/api/v1/mining/confirm",
                       json={"seal_sha256": seal["seal_sha256"]})
    assert body.status_code == 200, body.text
    receipt = body.json()["receipt"]
    assert receipt["rejected_labels"] == list(seal["frozen_labels"])
    assert receipt["correction_unit"] == len(seal["frozen_labels"])
    assert receipt["n_scenes"] == 6
    assert body.json()["vacuous"] == []
    assert "not a mechanism" in body.json()["claim_boundary"]


def test_the_held_out_frames_are_opened_once(client, planted):
    seal = _seal(client, planted)
    first = client.post("/api/v1/mining/confirm", json={"seal_sha256": seal["seal_sha256"]})
    assert first.status_code == 200
    second = client.post("/api/v1/mining/confirm", json={"seal_sha256": seal["seal_sha256"]})
    assert second.status_code == 409
    assert "opened" in second.json()["detail"]


def test_a_null_record_confirms_nothing(client):
    """The load-bearing gate: the same pass over frames with nothing planted in them.

    The same generator, the same number of features, the same family, the same ensemble.
    The only thing missing is the repetition, which is the only thing the pass is allowed
    to find.
    """
    record = _admit(client, _stack(range(500, 512), plant=False)).json()
    assert record["feature_counts"] == [FEATURES_PER_FRAME]
    replicates = _admit(client, _replicate_stack()).json()
    tolerance = _tolerance(client, replicates["record_id"])

    generated = client.post("/api/v1/mining/generate",
                            json=_run_body(record["record_id"], tolerance)).json()
    assert generated["mining"]["generate_family_size"] == 6 * 10

    seal = client.post("/api/v1/mining/freeze",
                       json=_run_body(record["record_id"], tolerance)).json()
    receipt = client.post("/api/v1/mining/confirm",
                          json={"seal_sha256": seal["seal_sha256"]}).json()["receipt"]
    assert receipt["rejected_labels"] == [], (
        "a pass over unplanted frames confirmed %s" % receipt["rejected_labels"])


def test_a_seal_frozen_by_another_surface_is_refused_here(client, planted):
    """A lag-family seal is the same object and is not this analysis."""
    from src.api.preregistration import store_seal

    specification = SearchSpecification(
        terms=(SearchTerm("ordered_pairs", (SearchAxis("scale", ("a", "b", "c")),)),),
        n_surrogates=199, label_format="{0}->{1}")
    confirm = SearchSpecification(
        terms=(SearchTerm("ordered_pairs", (SearchAxis("scale", ("a", "b")),)),),
        n_surrogates=199, label_format="{0}->{1}")
    held = PartitionIdentity("held_out", 8, 3, ("a", "b", "c"), (8, 16), {"split": "test"})
    seal = freeze_confirmatory_family(specification, confirm, held_out=held,
                                      sealed_at="2026-08-27T00:00:00+00:00")
    store_seal(seal)

    response = client.post("/api/v1/mining/confirm", json={"seal_sha256": seal.seal_sha256})
    assert response.status_code == 409
    assert "frozen by another surface" in response.json()["detail"]


def test_a_published_digest_that_disagrees_with_the_seal_spends_nothing(client, planted):
    """The check runs before the partition is opened, or a refusal costs the data it refused."""
    seal = _seal(client, planted)
    response = client.post("/api/v1/mining/confirm", json={
        "seal_sha256": seal["seal_sha256"], "published_sha256": "0" * 64})
    assert response.status_code == 400
    assert "published" in response.json()["detail"].lower()

    # The held-out frames must still be unopened: a correct confirmation still runs.
    again = client.post("/api/v1/mining/confirm", json={
        "seal_sha256": seal["seal_sha256"], "published_sha256": seal["seal_sha256"]})
    assert again.status_code == 200, again.text


# ======================================================================= publish and transfer


def _second_domain() -> DomainDeclaration:
    declaration = DomainDeclaration(
        name="mining_target_domain",
        description="A second gridded domain, declared so a transfer has somewhere to go.",
        axes=(AxisSpec(name="time", role="time", units="s"),
              AxisSpec(name="row", role="space", units="m", ordinal=0),
              AxisSpec(name="column", role="space", units="m", ordinal=1)),
        licence="Test fixture; no redistribution terms.",
        violations=("no_physical_metric", "no_propagation_speed"),
        lag_policy="none",
        provenance={"declared_by": "src/tests/test_mining_api.py"})
    if declaration.name not in DOMAIN_DECLARATIONS:
        onboard_domain(declaration, REANALYSIS_PHRASES, geometry=None, onboarded_by=__name__)
    return declaration


def _publish(client, planted):
    seal = _seal(client, planted)
    response = client.post("/api/v1/mining/motifs", json={
        "seal_sha256": seal["seal_sha256"], "label": seal["frozen_labels"][0]})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_published_motif_carries_the_licence_of_the_domain_it_came_from(client, planted):
    body = _publish(client, planted)
    assert body["motif"]["origin_domain"]["name"] == "reanalysis"
    assert "Copernicus" in body["origin_licence"]
    assert body["motif_sha256"] and body["definition_sha256"]


def test_a_motif_can_only_be_published_under_a_label_the_seal_froze(client, planted):
    seal = _seal(client, planted)
    response = client.post("/api/v1/mining/motifs", json={
        "seal_sha256": seal["seal_sha256"], "label": "frame0000|0-1-2"})
    if response.status_code == 200:            # the label happened to be the frozen one
        pytest.skip("the fabricated label collided with the frozen exemplar")
    assert response.status_code == 400
    assert "never frozen" in response.json()["detail"]


def test_a_transfer_searches_a_second_domain_with_the_frozen_definition(client, planted):
    _second_domain()
    published = _publish(client, planted)
    target = _admit(client, _stack(range(300, 306)),
                    domain="mining_target_domain", dataset="target").json()
    response = client.post("/api/v1/mining/transfer", json={
        "motif_sha256": published["motif_sha256"],
        "published_sha256": published["motif_sha256"],
        "target_record_id": target["record_id"],
        "target_domain": "mining_target_domain"})
    assert response.status_code == 200, response.text
    receipt = response.json()["receipt"]
    assert receipt["search"]["source"] == "frozen_motif"
    assert receipt["n_scenes"] == 6
    assert "not a corrected relationship" in receipt["claim_boundary"]


def test_a_transfer_target_is_opened_once_whatever_is_transferred_into_it(client, planted):
    _second_domain()
    published = _publish(client, planted)
    target = _admit(client, _stack(range(300, 306)),
                    domain="mining_target_domain", dataset="target").json()
    body = {"motif_sha256": published["motif_sha256"],
            "published_sha256": published["motif_sha256"],
            "target_record_id": target["record_id"],
            "target_domain": "mining_target_domain"}
    assert client.post("/api/v1/mining/transfer", json=body).status_code == 200
    again = client.post("/api/v1/mining/transfer", json=body)
    assert again.status_code == 409
    assert "already been opened" in again.json()["detail"] or \
           "already opened" in again.json()["detail"]


def test_a_transfer_into_the_domain_it_came_from_is_replication_not_transfer(client, planted):
    published = _publish(client, planted)
    target = _admit(client, _stack(range(300, 306)), dataset="another").json()
    response = client.post("/api/v1/mining/transfer", json={
        "motif_sha256": published["motif_sha256"],
        "published_sha256": published["motif_sha256"],
        "target_record_id": target["record_id"],
        "target_domain": "reanalysis"})
    assert response.status_code == 400
    assert "replication" in response.json()["detail"]


def test_a_transfer_whose_published_digest_does_not_match_is_refused(client, planted):
    _second_domain()
    published = _publish(client, planted)
    target = _admit(client, _stack(range(300, 306)),
                    domain="mining_target_domain", dataset="target").json()
    response = client.post("/api/v1/mining/transfer", json={
        "motif_sha256": published["motif_sha256"],
        "published_sha256": "0" * 64,
        "target_record_id": target["record_id"],
        "target_domain": "mining_target_domain"})
    assert response.status_code in (400, 409)


# ================================================================== the invariance audit


def _presentation_record(client) -> dict:
    """One configuration, its replicates, and the same shape rotated and moved."""
    base = _triangle((24.0, 24.0), 0.0)
    frames = [_frame(base, 10)]                                   # 0: reference
    frames += [_frame(base, 11 + index) for index in range(3)]    # 1-3: replicates
    frames += [_frame(_triangle((24.0, 24.0), angle), 20 + int(angle))
               for angle in (37.0, 121.0)]                        # 4-5: rotations
    frames += [_frame([(row + shift, col + shift) for row, col in base], 30 + int(shift))
               for shift in (6.0, 11.0)]                          # 6-7: translations
    return _admit(client, _npy(frames), dataset="presentations").json()


def test_the_audit_measures_each_matcher_against_its_own_declaration(client):
    record = _presentation_record(client)
    assert record["feature_counts"] == [3]
    response = client.post("/api/v1/mining/invariance", json={
        "record_id": record["record_id"],
        "reference_frame": 0,
        "replicate_frames": [1, 2, 3],
        "presentations": [
            {"frame": 4, "transforms": ["rotation"], "name": "rotation/37"},
            {"frame": 5, "transforms": ["rotation"], "name": "rotation/121"},
            {"frame": 6, "transforms": ["translation"], "name": "translation/6"},
            {"frame": 7, "transforms": ["translation"], "name": "translation/11"},
        ]})
    assert response.status_code == 200, response.text
    body = response.json()
    reports = body["reports"]
    assert set(reports) == {"relative_geometry", "absolute_position"}
    # The position-memorising control declares no invariance, and is not asked to have any.
    assert reports["absolute_position"]["declared"] == []
    # `relative_geometry` declares rescaling, and no presentation here rescaled anything, so
    # the audit says the declaration was not supported rather than quietly passing it.
    assert "rescaling" in reports["relative_geometry"]["overclaimed"]
    assert "nowhere near sufficient" in body["claim_boundary"]


def test_an_audit_with_nothing_to_compare_against_is_refused(client):
    record = _presentation_record(client)
    response = client.post("/api/v1/mining/invariance", json={
        "record_id": record["record_id"], "reference_frame": 0,
        "replicate_frames": [1, 2, 3], "presentations": []})
    assert response.status_code == 400
    assert "never looked" in response.json()["detail"]


def test_an_audit_needs_more_than_one_replicate_to_have_a_null(client):
    record = _presentation_record(client)
    response = client.post("/api/v1/mining/invariance", json={
        "record_id": record["record_id"], "reference_frame": 0,
        "replicate_frames": [1],
        "presentations": [{"frame": 4, "transforms": ["rotation"]}]})
    assert response.status_code == 400
    assert "null" in response.json()["detail"]


# ============================================================================ housekeeping


def test_reading_this_surface_creates_no_store(client, tmp_path):
    """A read that created the programme's store would make an empty directory a side effect."""
    client.get("/api/v1/mining")
    client.get("/api/v1/mining/records")
    client.post("/api/v1/mining/price", json={"n_scenes": 4, "n_features": 5})
    assert not (tmp_path / "mining").exists()
    assert not (tmp_path / "preregistrations").exists()


def test_the_store_holds_what_a_receipt_names(client, planted):
    body = client.get("/api/v1/mining/records").json()
    assert planted["record_id"] in [row["record_id"] for row in body["records"]]
    assert "programme state, not a cache" in body["note"]
    stored = Path(os.environ[MINING_ROOT_ENV]) / "records"
    payload = (stored / ("%s.npy" % planted["record_id"])).read_bytes()
    declaration = json.loads(
        (stored / ("%s.json" % planted["record_id"])).read_text(encoding="utf-8"))
    assert hashlib.sha256(payload).hexdigest() == declaration["content_sha256"]
