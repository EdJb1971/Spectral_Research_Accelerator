"""TG11.4b: the cross-domain record over HTTP.

Two questions run through this file. The first is whether a relationship planted across a real
semantic boundary is recovered on data it was not selected from, and whether the same pipeline
over the same generator with the coupling turned off confirms nothing. The second is whether the
three refusals this surface exists to make are structural rather than advisory: no interpolation,
no defaulted unit, and no setting the caller can still turn after the seal.
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Dict, Sequence

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.cross_domain import SOURCE_KEYS
from src.api.preregistration import ROOT_ENV
from src.core.builtin_domains import register_builtin_domains
from src.core.builtin_glossaries import ORDER_BOOK_PHRASES, REANALYSIS_PHRASES
from src.core.domain import DOMAIN_DECLARATIONS, AxisSpec, DomainDeclaration
from src.core.onboarding import DOMAIN_ONBOARDINGS, onboard_domain
from src.core.registry import restore, snapshot
from src.core.translation import DOMAIN_GLOSSARIES


FIRST_DOMAIN = "xd_thermal_observatory"
SECOND_DOMAIN = "xd_demand_ledger"
IRREGULAR_DOMAIN = "xd_irregular_log"

FIRST_CADENCE = 3600.0
SECOND_CADENCE = 10800.0
COMMON_STEPS = 240
PLANTED_LAG_SECONDS = 21600.0
LAG_SECONDS = (10800.0, 21600.0, 32400.0, 43200.0)
SURROGATES = 199

#: The relationship the generator plants, in the label the frozen family uses.
PLANTED_LABEL = "%s::thermal_gradient>%s::demand_pressure@2" % (FIRST_DOMAIN, SECOND_DOMAIN)

FIRST_CHANNELS = {"thermal_gradient": {"semantics": "instantaneous thermal state",
                                       "units": "K"},
                  "pressure_tendency": {"semantics": "instantaneous pressure tendency",
                                        "units": "hPa"}}
SECOND_CHANNELS = {"demand_pressure": {"semantics": "reported electrical demand",
                                       "units": "MW"},
                   "inventory_balance": {"semantics": "reported storage balance",
                                         "units": "MWh"}}


# ------------------------------------------------------------------------------- fixtures


def _declaration(name: str, description: str, *, cadence_name: str, floor_frames: int,
                 floor_basis: str, irregular: bool = False) -> DomainDeclaration:
    violations = ["no_physical_metric", "no_propagation_speed", "no_natural_cycle",
                  "unordered_channels"]
    if irregular:
        violations.append("irregular_sampling")
    return DomainDeclaration(
        name=name, description=description,
        axes=(AxisSpec(cadence_name, "time", units="s"),
              AxisSpec("channel", "category", ordered=False)),
        licence="Fixture licence; no real archive was accessed.",
        violations=tuple(violations), lag_policy="declared",
        declared_floor_frames=floor_frames, declared_floor_basis=floor_basis,
        provenance={"declared_by": __name__})


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A seal store and ledger per test, and two domains that cannot leak out of it."""
    monkeypatch.setenv(ROOT_ENV, str(tmp_path / "preregistrations"))
    before = (snapshot(DOMAIN_GLOSSARIES), snapshot(DOMAIN_DECLARATIONS),
              snapshot(DOMAIN_ONBOARDINGS))
    register_builtin_domains()
    onboard_domain(
        _declaration(FIRST_DOMAIN, "An hourly thermal instrument reporting physical readings.",
                     cadence_name="thermal_time", floor_frames=2,
                     floor_basis="two hourly sensor-response intervals"),
        REANALYSIS_PHRASES, geometry=None, onboarded_by=__name__)
    onboard_domain(
        _declaration(SECOND_DOMAIN, "A three-hourly operational ledger of reported states.",
                     cadence_name="ledger_time", floor_frames=1,
                     floor_basis="one three-hour reporting interval"),
        ORDER_BOOK_PHRASES, geometry=None, onboarded_by=__name__)
    onboard_domain(
        _declaration(IRREGULAR_DOMAIN, "A log whose observations arrive when they arrive.",
                     cadence_name="log_time", floor_frames=1,
                     floor_basis="one reporting interval", irregular=True),
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


# ------------------------------------------------------------------------------- the data


def _red(length: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(length, dtype=np.float64)
    values[0] = rng.standard_normal()
    innovation = math.sqrt(max(1.0 - phi * phi, 1e-12))
    for index in range(1, length):
        values[index] = phi * values[index - 1] + innovation * rng.standard_normal()
    return values


def _table(times: Sequence[float], columns: Dict[str, Sequence[float]],
           *, time_column: str = "t") -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([time_column] + list(columns))
    for index, moment in enumerate(times):
        writer.writerow([float(moment)] + [float(columns[name][index]) for name in columns])
    return output.getvalue()


def _pair(*, coupling: float = 1.0, seed: int = 20260827,
          common_steps: int = COMMON_STEPS, offset_seconds: float = 0.0):
    """Two native clocks, one optional delayed relationship, and nothing else planted.

    The null is the same builder with the coupling knob at zero: the same generator, the same
    channel count, the same cadences and the same family. A null drawn by different code
    would test the other code as much as the pipeline.
    """
    ratio = int(SECOND_CADENCE / FIRST_CADENCE)
    native_steps = common_steps * ratio
    rng = np.random.default_rng(seed)

    latent = _red(native_steps, 0.78, rng)
    thermal = 284.0 + 3.2 * latent + 0.25 * rng.standard_normal(native_steps)
    pressure = 1008.0 + 5.0 * _red(native_steps, 0.35, rng)

    driver = latent[::ratio]
    independent = _red(common_steps, 0.48, rng)
    lag_frames = int(PLANTED_LAG_SECONDS / SECOND_CADENCE)
    target = independent.copy()
    target[lag_frames:] = (coupling * driver[:-lag_frames]
                           + math.sqrt(max(1.0 - coupling * coupling, 0.0))
                           * independent[lag_frames:])
    demand = 1450.0 + 210.0 * target + 35.0 * rng.standard_normal(common_steps)
    inventory = 68.0 + 9.0 * _red(common_steps, 0.48, rng)

    first_times = np.arange(native_steps, dtype=np.float64) * FIRST_CADENCE
    second_times = (np.arange(common_steps, dtype=np.float64) * SECOND_CADENCE
                    + float(offset_seconds))
    return (_table(first_times, {"thermal_gradient": thermal,
                                 "pressure_tendency": pressure}),
            _table(second_times, {"demand_pressure": demand,
                                  "inventory_balance": inventory}))


def _source(domain: str, channels: Dict[str, Dict[str, str]], **extra) -> str:
    return json.dumps({"domain": domain, "time_column": "t", "channels": channels, **extra})


FIRST_SOURCE = _source(FIRST_DOMAIN, FIRST_CHANNELS)
SECOND_SOURCE = _source(SECOND_DOMAIN, SECOND_CHANNELS)


def _files(first_text: str, second_text: str, *, first_name: str = "thermal.csv",
           second_name: str = "demand.csv"):
    return {"first": (first_name, io.BytesIO(first_text.encode("utf-8")), "text/csv"),
            "second": (second_name, io.BytesIO(second_text.encode("utf-8")), "text/csv")}


def _data(**extra) -> Dict[str, object]:
    return {"first_source": FIRST_SOURCE, "second_source": SECOND_SOURCE,
            "name": "tg11.4b", **extra}


def _seal(client, first_text: str, second_text: str, *, study_id: str = "tg11.4b"):
    return client.post(
        "/api/v1/cross-domain/seal", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps(list(LAG_SECONDS)), study_id=study_id,
                   n_surrogates=SURROGATES, fraction=0.55))


# ------------------------------------------------------------------------- the boundary


def test_capabilities_state_that_nothing_is_interpolated(client):
    body = client.get("/api/v1/cross-domain").json()
    assert body["interpolation"] == "none"
    assert body["alignment"] == "exact_timestamp_intersection"
    assert body["lags_declared_in"] == "seconds"
    assert body["records_evidence"] is False and body["moves_rung"] is False
    assert set(body["requires_per_channel"]) == {"semantics", "units"}


def test_two_native_clocks_are_intersected_and_the_discards_are_reported(client):
    """A join that silently kept a third of one record is a different study."""
    first_text, second_text = _pair()
    body = client.post("/api/v1/cross-domain/align", files=_files(first_text, second_text),
                       data=_data()).json()
    alignment = body["alignment"]
    assert alignment["interpolation"] == "none"
    assert alignment["n_common_observations"] == COMMON_STEPS
    assert alignment["common_cadence_seconds"] == SECOND_CADENCE
    retained = alignment["retained_native_observations"]
    discarded = alignment["discarded_native_observations"]
    # The hourly record keeps one observation in three; the three-hourly record keeps all.
    assert retained[FIRST_DOMAIN] == COMMON_STEPS
    assert discarded[FIRST_DOMAIN] == COMMON_STEPS * 2
    assert discarded[SECOND_DOMAIN] == 0
    assert body["stored"] is False


def test_clocks_that_share_no_observation_are_refused_rather_than_resampled(client):
    """The tempting repair is a nearest-neighbour join, and it is not offered."""
    first_text, second_text = _pair(offset_seconds=1800.0)
    response = client.post("/api/v1/cross-domain/align",
                           files=_files(first_text, second_text), data=_data())
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Interpolation is not offered" in detail
    assert "invented values" in detail


def test_an_irregular_native_clock_refuses_precedence_by_name(client):
    """A frame lag on an irregular clock is not a duration, and is refused before any sweep."""
    first_text, second_text = _pair()
    rows = second_text.splitlines()
    # Drop one observation. The domain that admits this declares `irregular_sampling`, so the
    # reader accepts the record and the cross-domain layer is the thing that refuses it.
    broken = "\n".join(rows[:5] + rows[6:]) + "\n"
    response = client.post(
        "/api/v1/cross-domain/align", files=_files(first_text, broken),
        data={"first_source": FIRST_SOURCE,
              "second_source": _source(IRREGULAR_DOMAIN, SECOND_CHANNELS),
              "name": "irregular"})
    assert response.status_code == 400
    assert "irregular_sampling" in response.json()["detail"]


def test_a_column_whose_meaning_was_not_declared_is_refused_not_defaulted(client):
    """R19: a magnitude nobody described is a magnitude nobody chose to compare."""
    first_text, second_text = _pair()
    partial = _source(SECOND_DOMAIN, {"demand_pressure": SECOND_CHANNELS["demand_pressure"]})
    response = client.post(
        "/api/v1/cross-domain/align", files=_files(first_text, second_text),
        data={"first_source": FIRST_SOURCE, "second_source": partial, "name": "x"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "inventory_balance" in detail and "R19" in detail


def test_a_channel_declaration_missing_units_is_refused(client):
    first_text, second_text = _pair()
    dropped = {"demand_pressure": {"semantics": "reported electrical demand"},
               "inventory_balance": SECOND_CHANNELS["inventory_balance"]}
    response = client.post(
        "/api/v1/cross-domain/align", files=_files(first_text, second_text),
        data={"first_source": FIRST_SOURCE,
              "second_source": _source(SECOND_DOMAIN, dropped), "name": "x"})
    assert response.status_code == 400
    assert "units" in response.json()["detail"]


def test_an_unknown_reading_setting_is_refused_rather_than_ignored(client):
    first_text, second_text = _pair()
    response = client.post(
        "/api/v1/cross-domain/align", files=_files(first_text, second_text),
        data={"first_source": FIRST_SOURCE,
              "second_source": _source(SECOND_DOMAIN, SECOND_CHANNELS, resample="linear"),
              "name": "x"})
    assert response.status_code == 400
    assert "resample" in response.json()["detail"]


def test_two_records_from_one_domain_are_not_a_cross_domain_study(client):
    first_text, second_text = _pair()
    response = client.post(
        "/api/v1/cross-domain/align", files=_files(first_text, second_text),
        data={"first_source": FIRST_SOURCE,
              "second_source": _source(FIRST_DOMAIN, SECOND_CHANNELS), "name": "x"})
    assert response.status_code == 400
    assert "two distinct domain declarations" in response.json()["detail"]


# ------------------------------------------------------------------------------ the family


def test_a_lag_family_is_declared_in_seconds_and_converted_per_clock(client):
    first_text, second_text = _pair()
    body = client.post(
        "/api/v1/cross-domain/lags", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps(list(LAG_SECONDS)),
                   n_surrogates=SURROGATES)).json()
    family = body["family"]
    assert family["lag_seconds"] == list(LAG_SECONDS)
    assert family["lag_frames"] == [1, 2, 3, 4]
    assert family["common_cadence_seconds"] == SECOND_CADENCE


def test_only_pairs_that_cross_the_boundary_are_members(client):
    """A within-domain member would spend correction power on a question this does not ask."""
    first_text, second_text = _pair()
    family = client.post(
        "/api/v1/cross-domain/lags", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps(list(LAG_SECONDS)),
                   n_surrogates=SURROGATES)).json()["family"]
    assert family["n_pairs"] == 8
    assert family["family_size"] == 8 * len(LAG_SECONDS)
    for label in family["cross_domain_pairs"]:
        driver, driven = label.split(">")
        assert driver.split("::")[0] != driven.split("::")[0]


def test_a_duration_below_the_physical_floor_is_refused_not_dropped(client):
    """Both domains' floors are converted to seconds before either clock is combined."""
    first_text, second_text = _pair()
    response = client.post(
        "/api/v1/cross-domain/lags", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps([3600.0] + list(LAG_SECONDS)),
                   n_surrogates=SURROGATES))
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "physical floor" in detail
    assert "silently removed" in detail


def test_a_duration_the_common_clock_cannot_express_is_not_rounded(client):
    first_text, second_text = _pair()
    response = client.post(
        "/api/v1/cross-domain/lags", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps([16200.0]), n_surrogates=SURROGATES))
    assert response.status_code == 400
    assert "integer multiple" in response.json()["detail"]


def test_a_duplicated_duration_is_a_duplicated_family_member(client):
    first_text, second_text = _pair()
    response = client.post(
        "/api/v1/cross-domain/lags", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps([21600.0, 21600.0]), n_surrogates=SURROGATES))
    assert response.status_code == 400
    assert "duplicated physical lag" in response.json()["detail"]


def test_the_price_agrees_with_the_family_the_generate_pass_actually_searches(client):
    """A price computed one way and a search performed another is a correction unit that lies."""
    first_text, second_text = _pair()
    priced = client.post(
        "/api/v1/cross-domain/lags", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps(list(LAG_SECONDS)),
                   n_surrogates=SURROGATES)).json()["family"]
    generated = client.post(
        "/api/v1/cross-domain/generate", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps(list(LAG_SECONDS)), study_id="price",
                   n_surrogates=SURROGATES, fraction=0.55)).json()
    assert generated["n_examined"] == priced["family_size"]
    assert generated["family"]["cross_domain_pairs"] == priced["cross_domain_pairs"]
    assert generated["family"]["lag_frames"] == priced["lag_frames"]


# --------------------------------------------------------------------------- the partition


def test_the_partition_carries_an_embargo_at_least_the_records_own_memory(client):
    first_text, second_text = _pair()
    body = client.post("/api/v1/cross-domain/partition",
                       files=_files(first_text, second_text),
                       data=_data(fraction=0.55)).json()
    assert body["embargo_frames"] >= body["recommended_embargo_frames"]
    assert body["already_opened"] is False
    assert body["train"]["n_frames"] + body["held_out"]["n_frames"] <= COMMON_STEPS


def test_the_partition_identity_ignores_what_the_files_were_called(client):
    """D65: once is once per held-out data, not once per upload name."""
    first_text, second_text = _pair()
    digests = []
    for names in (("a.csv", "b.csv"), ("renamed.csv", "also-renamed.csv")):
        body = client.post(
            "/api/v1/cross-domain/partition",
            files=_files(first_text, second_text, first_name=names[0],
                         second_name=names[1]),
            data=_data(fraction=0.55)).json()
        digests.append(body["held_out"]["digest"])
    assert digests[0] == digests[1]


def test_generating_reads_nothing_from_the_held_out_partition(client, tmp_path):
    first_text, second_text = _pair()
    response = client.post(
        "/api/v1/cross-domain/generate", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps(list(LAG_SECONDS)), study_id="generate",
                   n_surrogates=SURROGATES, fraction=0.55))
    assert response.status_code == 200
    body = response.json()
    assert body["stored"] is False
    assert body["candidates"], "the generate stage should carry something forward"
    # Nothing written, and no partition spent: the ledger file does not exist yet.
    assert not (tmp_path / "preregistrations").exists()


def test_the_generate_stage_calls_its_own_output_selection(client):
    first_text, second_text = _pair()
    body = client.post(
        "/api/v1/cross-domain/generate", files=_files(first_text, second_text),
        data=_data(lag_seconds=json.dumps(list(LAG_SECONDS)), study_id="generate",
                   n_surrogates=SURROGATES, fraction=0.55)).json()
    assert "Selection" in body["claim_boundary"]
    assert "is not evidence" in body["claim_boundary"]


# -------------------------------------------------------------------------------- the seal


def test_a_seal_freezes_a_family_the_confirmation_can_pay_for(client):
    first_text, second_text = _pair()
    response = _seal(client, first_text, second_text)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["confirm_family_size"] <= body["generate_family_size"]
    assert body["confirm_labels"]
    assert body["records_evidence"] is False and body["rung_moved"] is False


def test_a_cross_domain_seal_is_visible_where_the_programme_lists_what_it_froze(client):
    """One drawer. A seal list that omitted these would show a study as having frozen less."""
    first_text, second_text = _pair()
    seal = _seal(client, first_text, second_text).json()
    listed = client.get("/api/v1/preregistration/seals").json()
    assert seal["seal_sha256"] in [row["seal_sha256"] for row in listed["seals"]]


def test_the_run_settings_are_sealed_inside_the_specification(client):
    """A setting kept beside the seal is a setting that can be edited after the seal."""
    first_text, second_text = _pair()
    seal = _seal(client, first_text, second_text).json()["seal"]
    notes = seal["confirm"]["notes"]
    assert notes["surface"] == "cross_domain"
    assert notes["lag_seconds"] == list(LAG_SECONDS)
    assert notes["first"]["channels"] == FIRST_CHANNELS
    assert notes["second"]["domain"] == SECOND_DOMAIN


def test_sealing_does_not_open_the_partition(client):
    first_text, second_text = _pair()
    seal = _seal(client, first_text, second_text).json()
    listed = client.get("/api/v1/preregistration/seals").json()["seals"]
    row = [r for r in listed if r["seal_sha256"] == seal["seal_sha256"]][0]
    assert row["spent"] is False


# ----------------------------------------------------------------------------- confirming


def test_the_planted_relationship_is_confirmed_on_data_it_was_not_selected_from(client):
    """The acceptance. One relationship, one direction, one physical lag, on held-out frames."""
    first_text, second_text = _pair(coupling=1.0)
    seal = _seal(client, first_text, second_text).json()
    assert PLANTED_LABEL in seal["confirm_labels"]
    response = client.post(
        "/api/v1/cross-domain/seals/%s/confirm" % seal["seal_sha256"],
        files=_files(first_text, second_text))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["confirmed_labels"] == [PLANTED_LABEL]
    row = [r for r in body["receipt"]["relationships"] if r["label"] == PLANTED_LABEL][0]
    assert row["lag_seconds"] == PLANTED_LAG_SECONDS
    assert row["statistic_units"] == "dimensionless"


def test_the_same_pipeline_over_an_uncoupled_pair_confirms_nothing(client):
    """The other half of the acceptance: the same generator with the knob at zero."""
    first_text, second_text = _pair(coupling=0.0)
    seal = _seal(client, first_text, second_text).json()
    response = client.post(
        "/api/v1/cross-domain/seals/%s/confirm" % seal["seal_sha256"],
        files=_files(first_text, second_text))
    assert response.status_code == 200, response.text
    assert response.json()["confirmed_labels"] == []


def test_a_confirmed_relationship_carries_both_operands_meanings(client):
    """R19: the receipt restores the semantic boundary the statistic deliberately erased."""
    first_text, second_text = _pair()
    seal = _seal(client, first_text, second_text).json()
    body = client.post(
        "/api/v1/cross-domain/seals/%s/confirm" % seal["seal_sha256"],
        files=_files(first_text, second_text)).json()
    row = body["receipt"]["relationships"][0]
    assert row["driver_operand"]["units"] and row["driven_operand"]["units"]
    assert row["driver_operand"]["units"] != row["driven_operand"]["units"]
    assert "no causal mechanism" in row["claim_boundary"]
    assert body["receipt"]["cross_domain"]["interpolation"] == "none"


def test_the_partition_is_opened_once(client):
    first_text, second_text = _pair()
    seal = _seal(client, first_text, second_text).json()
    url = "/api/v1/cross-domain/seals/%s/confirm" % seal["seal_sha256"]
    assert client.post(url, files=_files(first_text, second_text)).status_code == 200
    again = client.post(url, files=_files(first_text, second_text))
    assert again.status_code == 409


def test_a_different_pair_of_records_confirms_nothing_and_costs_nothing(client):
    """A frozen label confirmed on foreign data would be a confirmation of nothing.

    Two independent checks stand between a wrong upload and a spent partition, and which one
    fires depends on the data: the selection is re-derived from the training partition and
    must match the seal, and the held-out partition presented must be the one the seal named.
    Records drawn from the same generator at another seed can still select the same members,
    so it is the second check that usually speaks. What matters is not which refusal it is
    but that both are made before the ledger is written - the honest confirmation below is
    still available afterwards, which is the only way to see that nothing was spent.
    """
    first_text, second_text = _pair(coupling=1.0)
    seal = _seal(client, first_text, second_text).json()
    other_first, other_second = _pair(coupling=1.0, seed=99)
    url = "/api/v1/cross-domain/seals/%s/confirm" % seal["seal_sha256"]
    wrong = client.post(url, files=_files(other_first, other_second))
    assert wrong.status_code in (400, 409), wrong.text
    assert client.post(url, files=_files(first_text, second_text)).status_code == 200


def test_an_edited_seal_is_not_a_seal(client, tmp_path):
    """The stored seal is tamper-evident, and that is what makes the rest of this cheap.

    Editing a frozen family to name a member the training partition never selected is the
    obvious way to launder a search into a confirmation. It does not reach the confirmatory
    code at all: the seal no longer hashes to its own contents, and a seal that does not is
    not evidence that anything was fixed before the partition was opened. Nothing is spent.
    """
    first_text, second_text = _pair()
    seal = _seal(client, first_text, second_text).json()
    path = tmp_path / "preregistrations" / ("%s.json" % seal["seal_sha256"])
    stored = json.loads(path.read_text(encoding="utf-8"))
    reversed_label = "%s::demand_pressure>%s::thermal_gradient@4" % (SECOND_DOMAIN,
                                                                    FIRST_DOMAIN)
    axis = stored["confirm"]["terms"][0]["axes"][0]
    assert axis["name"] == "relationship"
    axis["values"] = [reversed_label]
    path.write_text(json.dumps(stored), encoding="utf-8")

    response = client.post(
        "/api/v1/cross-domain/seals/%s/confirm" % seal["seal_sha256"],
        files=_files(first_text, second_text))
    assert response.status_code == 400, response.text
    assert "does not hash to its own contents" in response.json()["detail"]
    assert not client.get("/api/v1/preregistration/seals").json()["seals"][0]["spent"]


def test_the_frozen_members_are_re_derived_from_the_record(client):
    """Not reconstructed from their own names, which would match themselves by construction.

    Two things follow from deriving them again. The members handed to the confirmatory pass
    are the ones this record's training partition actually selects, so a future change to the
    selection rule cannot silently confirm a stale family under its old labels. And the
    describe() rows the seal returns carry measured training statistics, which a label could
    not have supplied.
    """
    first_text, second_text = _pair()
    seal = _seal(client, first_text, second_text).json()
    labels = [row["label"] for row in seal["frozen"]]
    assert labels == seal["confirm_labels"]
    for row in seal["frozen"]:
        assert row["n_pairs"] > 0 and math.isfinite(row["correlation"])


def test_a_published_digest_that_disagrees_with_the_seal_spends_nothing(client):
    """A refusal that had already opened the partition would cost what it was declining."""
    first_text, second_text = _pair()
    seal = _seal(client, first_text, second_text).json()
    url = "/api/v1/cross-domain/seals/%s/confirm" % seal["seal_sha256"]
    mismatched = client.post(url, files=_files(first_text, second_text),
                             data={"published_sha256": "0" * 64})
    assert mismatched.status_code in (400, 409)
    ok = client.post(url, files=_files(first_text, second_text),
                     data={"published_sha256": seal["seal_sha256"]})
    assert ok.status_code == 200
    assert ok.json()["checked_against_publication"] is True


def test_a_seal_frozen_by_another_surface_is_not_confirmed_here(client):
    """A seal this surface cannot reproduce is refused, not guessed at."""
    from src.core.builtin_glossaries import ORDER_BOOK_PHRASES as _phrases  # noqa: F401

    rng = np.random.default_rng(7)
    alpha = rng.normal(size=240)
    beta = rng.normal(size=240)
    beta[2:] = alpha[:-2] + 0.4 * beta[2:]
    text = _table(np.arange(240.0), {"alpha": alpha, "beta": beta})
    other = client.post(
        "/api/v1/preregistration/seal",
        files={"file": ("r.csv", io.BytesIO(text.encode("utf-8")), "text/csv")},
        data={"domain": SECOND_DOMAIN, "time_column": "t", "study_id": "elsewhere",
              "generate": json.dumps({"lags": [1, 2, 3, 4], "n_surrogates": 999}),
              "confirm": json.dumps({"lags": [2], "n_surrogates": 99}),
              "train_ratio": 0.6, "embargo_frames": 4})
    assert other.status_code == 200, other.text
    first_text, second_text = _pair()
    response = client.post(
        "/api/v1/cross-domain/seals/%s/confirm" % other.json()["seal_sha256"],
        files=_files(first_text, second_text))
    assert response.status_code == 409
    assert "not frozen by the cross-domain surface" in response.json()["detail"]


def test_confirming_accepts_the_records_and_nothing_else(client):
    """Structural: a knob still turnable after the seal is a family member chosen after it."""
    import inspect

    from src.api.cross_domain import confirm as confirm_route

    accepted = set(inspect.signature(confirm_route).parameters)
    assert accepted == {"seal_sha256", "first", "second", "published_sha256"}, accepted


def test_no_route_on_this_surface_accepts_a_lag_in_frames(client):
    """The whole point of the module is that a lag here is a duration, not a frame count."""
    import inspect

    import src.api.cross_domain as module

    named = set()
    routes = 0
    for name, value in vars(module).items():
        if not callable(value) or getattr(value, "__module__", "") != module.__name__:
            continue
        if name.startswith("_"):
            continue
        try:
            parameters = inspect.signature(value).parameters
        except (TypeError, ValueError):
            continue
        if "first" in parameters or "seal_sha256" in parameters:
            routes += 1
            named.update(parameters)
    assert routes >= 5, "the route scan found nothing to check"
    assert "lag_seconds" in named
    assert not {"lags", "lag_frames", "max_lag"} & named, sorted(named)


def test_the_source_reading_cannot_smuggle_an_interpolation_setting():
    """The allow-list is the thing that keeps `resample` from being a silently ignored key."""
    assert SOURCE_KEYS == {"domain", "time_column", "time_units", "delimiter", "channels",
                           "aggregation_window_seconds"}


def test_the_claim_boundary_denies_a_mechanism_and_names_the_ledger_residual(client):
    body = client.get("/api/v1/cross-domain").json()
    boundary = body["claim_boundary"]
    assert "no causal mechanism" in boundary
    assert "two different registered domains are two partitions" in boundary
