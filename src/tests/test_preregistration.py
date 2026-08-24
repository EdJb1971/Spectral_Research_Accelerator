"""Phase G3, TG3.2: mine on train, freeze, and open the held-out partition exactly once.

TG3.1 prices a declared family and refuses it when the declared ensemble cannot reject one
member. R18 permits two remedies and this module is the second, so the acceptance criterion
is the whole walkthrough rather than any one function: the **real T4C.6 family of 36**, read
off the frozen campaign JSON, is refused at an ensemble of 199; a four-member confirmatory
subset of it is affordable at exactly that ensemble; and the sealed four are corrected at
four, on a partition that can then never be opened again.

Around it sit the properties that make the seal worth more than a comment saying the family
was frozen first:

*   a partition is identified from **geometry and lineage only**, and a series whose values
    raise on access proves it, because a seal written after reading the held-out data is not
    a preregistration whatever it hashes to;
*   both ways of editing a seal are detected and are detected *differently* - an edited field
    is named, and an editor who also rewrites the digest table breaks the outer digest;
*   `verify_published` is the check with weight, and a wholesale rewrite that is perfectly
    self-consistent is caught only there;
*   the ledger is keyed by the **partition**, so a second, entirely honest seal against the
    same held-out data is refused - which is the substance of "once"; and
*   a refusal never spends the partition, while a completed confirmation always does.
"""

import dataclasses
import json
import os

import numpy as np
import pytest

from src.analysis_engine import cross_scale as cs
from src.core.channel_series import ChannelSeries
from src.core.errors import InvalidParameterError
from src.core.family import (
    FamilyUnaffordableError,
    SearchAxis,
    SearchSpecification,
    SearchTerm,
)
from src.core.preregistration import (
    SEAL_SCHEMA,
    HeldOutAlreadyOpenedError,
    HeldOutLedger,
    PartitionIdentity,
    PartitionMismatchError,
    Seal,
    SealBrokenError,
    confirm_on_held_out,
    freeze_confirmatory_family,
    report_generation,
)
from src.statistics.multiple_comparisons import adjust

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMPAIGN = os.path.join(REPO_ROOT, "campaigns", "t4c6_nz_era5_temperature_850_v1.json")

#: The ensemble the walkthrough can actually afford. `max_affordable_family` puts the ceiling
#: at four members here, so the frozen family below is affordable and the family it came from
#: is not - which is the entire reason this module exists.
AFFORDABLE_ENSEMBLE = 199


def _t4c6_generate_specification(n_surrogates=AFFORDABLE_ENSEMBLE):
    """The T4C.6 search, read off the frozen campaign file rather than retyped here."""
    with open(CAMPAIGN, encoding="utf-8") as handle:
        document = json.load(handle)
    protocol = cs.GateProtocol.from_mapping(document["campaign"]["gate_plan"]["protocol"])
    return dataclasses.replace(protocol.search_specification(), n_surrogates=n_surrogates)


def _confirm_specification(scales=(1, 2), lags=(3, 4), n_surrogates=AFFORDABLE_ENSEMBLE):
    """A subset of the T4C.6 family, rendering the same `source->target@lag` labels."""
    return SearchSpecification(
        terms=(SearchTerm("ordered_pairs", (SearchAxis("scale", tuple(scales)),)),
               SearchTerm("product", (SearchAxis("lag_frames", tuple(lags)),))),
        n_surrogates=n_surrogates, alpha=0.05, correction="benjamini_yekutieli",
        label_format="{0}->{1}@{2}", study_id="t4c6_nz_era5_temperature_850_v1")


def _partition(name="held_out", n_times=64, frames=(96, 160), split="test", n_channels=3,
               channel_labels=("scale_1", "scale_2", "scale_3"), **provenance):
    return PartitionIdentity(
        name=name, n_times=n_times, n_channels=n_channels,
        channel_labels=channel_labels, frames=frames,
        provenance={"split": split, "source": "era5_nz_t850", **provenance})


def _train_partition():
    return _partition(name="train", n_times=96, frames=(0, 96), split="train")


def _seal(ledger=None, confirm=None, held_out=None):
    return freeze_confirmatory_family(
        _t4c6_generate_specification(), confirm or _confirm_specification(),
        held_out=held_out or _partition(), sealed_at="2026-08-25T09:00:00Z",
        study_id="t4c6_nz_era5_temperature_850_v1", ledger=ledger)


def _p_values(first=0.001):
    """One p-value per frozen member; the first small enough to survive BY at four tests."""
    labels = _confirm_specification().labels()
    return dict(zip(labels, (first, 0.31, 0.47, 0.62)))


# ------------------------------------------------------------------------- the acceptance


def test_the_family_refused_at_this_ensemble_is_confirmable_as_a_frozen_subset_of_itself():
    """The whole R18 remedy in one walkthrough, priced off the frozen campaign file.

    36 members need 3,005 surrogates and are refused at 199. Four members need 166 and are
    not. Nothing here made the search cheaper: the generate stage still enumerated 36 and
    still tested them, and the receipt says so.
    """
    generate = _t4c6_generate_specification()
    assert generate.family_size == 36
    with pytest.raises(FamilyUnaffordableError) as refusal:
        generate.declare()
    assert refusal.value.context["surrogates_required"] == 3005

    confirm = _confirm_specification()
    assert set(confirm.labels()) <= set(generate.labels())
    account = confirm.declare()
    assert account.affordable and account.surrogates_required == 166

    train = _train_partition()
    report = report_generation(generate, train=train, candidates=list(confirm.labels()),
                               p_values={label: 0.004 for label in confirm.labels()})
    assert report["generate_family_size"] == 36
    assert report["generate_sha256"] == generate.fingerprint()
    assert "not findings" in report["claim_boundary"]

    ledger = HeldOutLedger()
    seal = _seal(ledger=ledger)
    published = seal.seal_sha256

    receipt = confirm_on_held_out(seal, p_values=_p_values(), held_out=_partition(),
                                  ledger=ledger, opened_at="2026-08-25T11:00:00Z",
                                  published_sha256=published)
    assert receipt["correction_unit"] == 4
    assert receipt["generate_family_size"] == 36
    assert receipt["rejected_labels"] == ["1->2@3"]
    assert "not corrected for here and are not claims" in receipt["claim_boundary"]

    # And the partition is spent, so a second confirmatory family cannot be tested on it.
    with pytest.raises(HeldOutAlreadyOpenedError):
        confirm_on_held_out(seal, p_values=_p_values(), held_out=_partition(), ledger=ledger,
                            opened_at="2026-08-25T12:00:00Z")


def test_the_correction_unit_is_the_frozen_size_and_not_the_size_it_was_mined_from():
    """Four tests, corrected at four. The receipt records 36 without correcting for it."""
    ledger = HeldOutLedger()
    seal = _seal(ledger=ledger)
    supplied = _p_values()
    receipt = confirm_on_held_out(seal, p_values=supplied, held_out=_partition(),
                                  ledger=ledger, opened_at="2026-08-25T11:00:00Z")
    ordered = [supplied[label] for label in seal.confirm_labels]

    at_four = adjust(ordered, method="benjamini_yekutieli", alpha=0.05, n_tests=4)
    assert receipt["adjusted"] == pytest.approx([float(q) for q in at_four["adjusted"]])
    assert receipt["dependence_assumption"] == "controls FDR under arbitrary dependence"

    # The distinction is not cosmetic: the same p-values corrected over the generated family
    # reject nothing, which is precisely what the split buys and why the freeze must precede
    # the held-out partition being opened.
    at_thirty_six = adjust(ordered, method="benjamini_yekutieli", alpha=0.05, n_tests=36)
    assert not any(at_thirty_six["rejected"])
    assert receipt["n_rejected"] == 1


# ------------------------------------------------------------------- partition identity


class _ValuesExplode:
    """A series with geometry and lineage whose measure arrays raise on any access."""

    n_times = 64
    n_channels = 3
    channels = ("scale_1", "scale_2", "scale_3")
    provenance = {"split": "test", "split_frames": [96, 160]}

    @property
    def measures(self):
        raise AssertionError("a seal must be written without reading the held-out values")

    def to_matrix(self, measure):
        raise AssertionError("a seal must be written without reading the held-out values")


def test_a_partition_is_identified_without_reading_a_single_value():
    identity = PartitionIdentity.from_series(_ValuesExplode(), name="held_out")
    assert identity.n_times == 64 and identity.n_channels == 3
    assert identity.frames == (96, 160)
    assert identity.channel_labels == ("scale_1", "scale_2", "scale_3")


def test_from_series_takes_the_frames_the_split_recorded_and_falls_back_to_the_whole_span():
    times = np.arange(8, dtype=np.float64) * 3600.0
    measures = {"energy": np.ones((8, 2), dtype=np.float64)}
    split = ChannelSeries(channels=("a", "b"), times_seconds=times, measures=measures,
                          provenance={"split": "test", "split_frames": [40, 48]})
    assert PartitionIdentity.from_series(split, name="held_out").frames == (40, 48)

    whole = ChannelSeries(channels=("a", "b"), times_seconds=times, measures=measures)
    assert PartitionIdentity.from_series(whole, name="held_out").frames == (0, 8)


def test_provenance_a_receipt_cannot_hold_is_recorded_as_its_type_not_dropped():
    """A digest is over JSON, and a key with no value reads as a key with no content."""
    times = np.arange(4, dtype=np.float64) * 3600.0
    series = ChannelSeries(channels=("a", "b"), times_seconds=times,
                           measures={"energy": np.ones((4, 2))},
                           provenance={"split": "test", "mask": np.zeros(4)})
    identity = PartitionIdentity.from_series(series, name="held_out")
    assert identity.provenance["mask"] == "<ndarray>"
    assert isinstance(identity.digest(), str) and len(identity.digest()) == 64


def test_a_partition_digest_moves_when_its_geometry_or_lineage_moves():
    base = _partition()
    assert base.digest() == _partition().digest()
    assert base.digest() != _partition(n_times=63).digest()
    assert base.digest() != _partition(frames=(96, 159)).digest()
    assert base.digest() != _partition(source="era5_nz_t500").digest()


@pytest.mark.parametrize("kwargs, field", [
    ({"name": "  "}, "name"),
    ({"n_times": 0}, "partition shape"),
    ({"n_channels": 0, "channel_labels": ()}, "partition shape"),
    ({"frames": (160, 96)}, "frames"),
    ({"channel_labels": ("scale_1", "scale_2")}, "channel_labels"),
])
def test_a_partition_that_cannot_be_told_apart_or_cannot_exist_is_refused(kwargs, field):
    with pytest.raises(InvalidParameterError) as refusal:
        _partition(**kwargs)
    assert refusal.value.context["parameter"] == field


def test_a_partition_whose_count_and_labels_disagree_describes_two_geometries():
    """Found by running this slice: the record bound a contradiction without noticing."""
    with pytest.raises(InvalidParameterError) as refusal:
        _partition(n_channels=4)
    assert "two different geometries" in str(refusal.value)


# -------------------------------------------------------------------------------- seal


def test_a_fresh_seal_verifies_and_says_what_its_own_verification_is_worth():
    result = _seal().verify()
    assert len(result["seal_sha256"]) == 64
    assert "says nothing about whether it was edited" in result["note"]


def test_an_edited_field_is_named_by_the_refusal():
    seal = _seal()
    object.__setattr__(seal, "sealed_at", "2026-08-25T13:00:00Z")
    with pytest.raises(SealBrokenError) as refusal:
        seal.verify()
    assert refusal.value.context["changed_fields"] == ["sealed_at"]
    assert "not a repair of this one" in str(refusal.value)


def test_an_editor_who_rewrites_the_digest_table_too_breaks_the_outer_digest():
    seal = _seal()
    object.__setattr__(seal, "confirm_labels", ("1->2@3",))
    object.__setattr__(seal, "field_sha256", seal.compute_field_digests())
    with pytest.raises(SealBrokenError) as refusal:
        seal.verify()
    assert refusal.value.context["changed_fields"] == []
    assert "the digest table itself" in str(refusal.value)


def test_a_wholesale_rewrite_is_self_consistent_and_caught_only_against_the_published_digest():
    """The boundary this module refuses to blur: a local seal is not evidence about itself."""
    published = _seal().seal_sha256
    rewritten = _seal(confirm=_confirm_specification(scales=(1, 3)))
    rewritten.verify()  # perfectly self-consistent, and worth nothing on its own
    assert rewritten.seal_sha256 != published
    with pytest.raises(SealBrokenError) as refusal:
        rewritten.verify_published(published)
    assert "rewritten wholesale after publication" in str(refusal.value)


def test_verify_published_refuses_to_check_a_seal_against_nothing():
    with pytest.raises(InvalidParameterError) as refusal:
        _seal().verify_published("   ")
    assert refusal.value.context["parameter"] == "published_sha256"
    assert "not evidence about itself" in str(refusal.value)


def test_a_published_digest_that_matches_is_reported_as_matching():
    seal = _seal()
    assert seal.verify_published(seal.seal_sha256.upper())["matches"] is True


def test_a_seal_survives_a_round_trip_through_json_with_its_digests_intact():
    seal = _seal()
    restored = Seal.from_mapping(json.loads(json.dumps(seal.to_mapping())))
    assert restored.seal_sha256 == seal.seal_sha256
    assert restored.held_out.digest() == seal.held_out.digest()
    assert restored.verify_published(seal.seal_sha256)["matches"] is True


def test_a_stored_document_that_is_not_a_seal_is_refused_by_its_schema_tag():
    mapping = _seal().to_mapping()
    mapping["schema"] = "generation-report/v1"
    with pytest.raises(InvalidParameterError) as refusal:
        Seal.from_mapping(mapping)
    assert SEAL_SCHEMA in str(refusal.value)


# ---------------------------------------------------------------------- generate stage


def test_a_candidate_that_was_never_declared_has_no_family_and_is_refused():
    generate = _t4c6_generate_specification()
    with pytest.raises(InvalidParameterError) as refusal:
        report_generation(generate, train=_train_partition(),
                          candidates=["1->2@3", "1->2@99"])
    assert "never part of a priced family" in str(refusal.value)


def test_mining_on_the_held_out_partition_is_refused_by_the_lineage_it_carries():
    generate = _t4c6_generate_specification()
    with pytest.raises(InvalidParameterError) as refusal:
        report_generation(generate, train=_partition(), candidates=["1->2@3"])
    assert refusal.value.context["parameter"] == "train"
    assert "already used the data" in str(refusal.value)


def test_the_generation_report_labels_its_p_values_uncorrected_and_claims_nothing():
    generate = _t4c6_generate_specification()
    report = report_generation(generate, train=_train_partition(), candidates=["1->2@3"],
                               p_values={"1->2@3": 0.0004})
    assert report["stage"] == "generate"
    assert report["p_values_uncorrected"] == {"1->2@3": 0.0004}
    assert report["n_candidates"] == 1
    assert "no p-value here is evidence" in report["claim_boundary"].lower()


# ------------------------------------------------------------------------------ freeze


def test_a_confirmatory_member_that_was_never_generated_is_a_fresh_search_and_is_refused():
    outside = _confirm_specification(lags=(3, 99))
    with pytest.raises(InvalidParameterError) as refusal:
        _seal(confirm=outside)
    assert refusal.value.context["generate_family_size"] == 36
    assert "under the name of a confirmation" in str(refusal.value)


def test_a_confirmatory_family_too_large_for_its_own_ensemble_is_refused_at_the_freeze():
    """The TG3.1 gate applies to the frozen family at its own size - that is the point."""
    with pytest.raises(FamilyUnaffordableError):
        _seal(confirm=_confirm_specification(scales=(1, 2, 3), lags=(3, 4, 5, 6)))


def test_a_seal_with_no_time_on_it_cannot_be_shown_to_predate_anything():
    with pytest.raises(InvalidParameterError) as refusal:
        freeze_confirmatory_family(_t4c6_generate_specification(), _confirm_specification(),
                                   held_out=_partition(), sealed_at="")
    assert refusal.value.context["parameter"] == "sealed_at"


def test_freezing_against_the_partition_the_candidates_were_mined_from_is_refused():
    with pytest.raises(InvalidParameterError) as refusal:
        _seal(held_out=_train_partition())
    assert "measures the selection, not the relationship" in str(refusal.value)


def test_a_family_too_large_to_list_cannot_be_shown_to_contain_another_and_is_refused():
    huge = SearchSpecification(
        terms=(SearchTerm("product", (SearchAxis("a", tuple(range(400))),
                                      SearchAxis("b", tuple(range(300))))),),
        n_surrogates=AFFORDABLE_ENSEMBLE)
    with pytest.raises(InvalidParameterError) as refusal:
        freeze_confirmatory_family(huge, _confirm_specification(), held_out=_partition(),
                                   sealed_at="2026-08-25T09:00:00Z")
    assert refusal.value.context["parameter"] == "generate family_size"


def test_the_seal_binds_both_specifications_and_the_partition_it_will_be_tested_against():
    generate, confirm = _t4c6_generate_specification(), _confirm_specification()
    seal = _seal()
    assert seal.generate_sha256 == generate.fingerprint()
    assert seal.confirm_sha256 == confirm.fingerprint()
    assert seal.confirm_labels == confirm.labels()
    assert seal.confirm_family_size == 4 and seal.generate_family_size == 36
    assert seal.held_out.digest() == _partition().digest()
    assert seal.confirm_account["affordable"] is True


def test_a_partition_already_spent_cannot_have_a_new_family_frozen_against_it():
    ledger = HeldOutLedger()
    first = _seal(ledger=ledger)
    ledger.open(first, _partition(), opened_at="2026-08-25T11:00:00Z")
    with pytest.raises(HeldOutAlreadyOpenedError):
        _seal(ledger=ledger, confirm=_confirm_specification(scales=(1, 3)))


# ------------------------------------------------------------------------------ ledger


def test_a_second_perfectly_honest_seal_against_the_same_partition_is_still_refused():
    """The substance of "once". Each confirmation is defensible; the pair is uncorrected."""
    ledger = HeldOutLedger()
    first = _seal(ledger=ledger)
    ledger.open(first, _partition(), opened_at="2026-08-25T11:00:00Z")

    second = _seal(confirm=_confirm_specification(scales=(1, 3)))
    second.verify()
    assert second.seal_sha256 != first.seal_sha256
    with pytest.raises(HeldOutAlreadyOpenedError) as refusal:
        ledger.open(second, _partition(), opened_at="2026-08-25T12:00:00Z")
    assert refusal.value.context["first_seal_sha256"] == first.seal_sha256
    assert refusal.value.context["second_seal_sha256"] == second.seal_sha256
    assert "one seal covering both families" in str(refusal.value)


def test_a_different_held_out_partition_is_a_different_key_and_is_not_refused():
    ledger = HeldOutLedger()
    ledger.open(_seal(), _partition(), opened_at="2026-08-25T11:00:00Z")
    later = _partition(name="held_out_2", frames=(160, 224))
    ledger.open(_seal(held_out=later), later, opened_at="2026-08-25T12:00:00Z")
    assert len(ledger.records) == 2


def test_the_constraint_outlives_the_process_that_made_it(tmp_path):
    path = str(tmp_path / "held_out_ledger.json")
    HeldOutLedger(path).open(_seal(), _partition(), opened_at="2026-08-25T11:00:00Z")

    reopened = HeldOutLedger(path)
    assert len(reopened.records) == 1
    with pytest.raises(HeldOutAlreadyOpenedError):
        reopened.open(_seal(), _partition(), opened_at="2026-08-25T12:00:00Z")


def test_a_json_file_that_is_not_a_ledger_is_refused_rather_than_read_as_empty(tmp_path):
    path = str(tmp_path / "not_a_ledger.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"schema": "search-specification/v1", "records": {}}, handle)
    with pytest.raises(InvalidParameterError) as refusal:
        HeldOutLedger(path)
    assert refusal.value.context["parameter"] == "ledger schema"


def test_a_partition_opened_at_no_stated_time_is_refused():
    with pytest.raises(InvalidParameterError) as refusal:
        HeldOutLedger().open(_seal(), _partition(), opened_at=" ")
    assert refusal.value.context["parameter"] == "opened_at"


def test_the_ledger_record_says_what_was_spent_and_what_spent_it():
    ledger = HeldOutLedger()
    seal = _seal()
    record = ledger.open(seal, _partition(), opened_at="2026-08-25T11:00:00Z")
    assert record["partition_digest"] == _partition().digest()
    assert record["seal_sha256"] == seal.seal_sha256
    assert record["family_size"] == 4
    assert record["sealed_at"] == "2026-08-25T09:00:00Z"
    assert record["study_id"] == "t4c6_nz_era5_temperature_850_v1"


# ----------------------------------------------------------------------------- confirm


def test_confirming_on_a_partition_the_seal_did_not_name_is_refused():
    ledger = HeldOutLedger()
    seal = _seal(ledger=ledger)
    other = _partition(name="held_out_2", frames=(160, 224))
    with pytest.raises(PartitionMismatchError) as refusal:
        confirm_on_held_out(seal, p_values=_p_values(), held_out=other, ledger=ledger,
                            opened_at="2026-08-25T11:00:00Z")
    assert refusal.value.context["sealed_digest"] == seal.held_out.digest()
    assert refusal.value.context["presented_digest"] == other.digest()
    assert "chosen after the declaration" in str(refusal.value)


def test_a_frozen_member_with_no_p_value_is_a_family_tested_in_part_and_is_refused():
    supplied = _p_values()
    supplied.pop("2->1@4")
    with pytest.raises(InvalidParameterError) as refusal:
        confirm_on_held_out(_seal(), p_values=supplied, held_out=_partition(),
                            ledger=HeldOutLedger(), opened_at="2026-08-25T11:00:00Z")
    assert refusal.value.context["frozen_family_size"] == 4
    assert "tested and not reported" in str(refusal.value)


def test_a_p_value_for_a_member_the_seal_does_not_contain_is_mining_on_held_out_data():
    supplied = _p_values()
    supplied["1->3@5"] = 0.002
    with pytest.raises(InvalidParameterError) as refusal:
        confirm_on_held_out(_seal(), p_values=supplied, held_out=_partition(),
                            ledger=HeldOutLedger(), opened_at="2026-08-25T11:00:00Z")
    assert "whatever it is called" in str(refusal.value)


def test_a_broken_seal_stops_the_confirmation_before_it_starts():
    seal = _seal()
    object.__setattr__(seal, "confirm_family_size", 1)
    with pytest.raises(SealBrokenError):
        confirm_on_held_out(seal, p_values=_p_values(), held_out=_partition(),
                            ledger=HeldOutLedger(), opened_at="2026-08-25T11:00:00Z")


def test_a_confirmation_against_the_wrong_published_digest_is_refused():
    ledger = HeldOutLedger()
    with pytest.raises(SealBrokenError):
        confirm_on_held_out(_seal(ledger=ledger), p_values=_p_values(), held_out=_partition(),
                            ledger=ledger, opened_at="2026-08-25T11:00:00Z",
                            published_sha256="0" * 64)
    assert ledger.records == {}


def test_a_refusal_does_not_spend_the_partition_and_a_confirmation_always_does():
    """Order matters: every check runs before the ledger is written, and none after."""
    ledger = HeldOutLedger()
    seal = _seal(ledger=ledger)
    incomplete = _p_values()
    incomplete.pop("1->2@4")
    with pytest.raises(InvalidParameterError):
        confirm_on_held_out(seal, p_values=incomplete, held_out=_partition(), ledger=ledger,
                            opened_at="2026-08-25T11:00:00Z")
    assert ledger.records == {}

    receipt = confirm_on_held_out(seal, p_values=_p_values(), held_out=_partition(),
                                  ledger=ledger, opened_at="2026-08-25T11:30:00Z")
    assert len(ledger.records) == 1
    assert receipt["ledger_record"]["opened_at"] == "2026-08-25T11:30:00Z"


def test_the_receipt_reports_the_frozen_labels_in_their_frozen_order():
    ledger = HeldOutLedger()
    seal = _seal(ledger=ledger)
    shuffled = dict(reversed(list(_p_values().items())))
    receipt = confirm_on_held_out(seal, p_values=shuffled, held_out=_partition(),
                                  ledger=ledger, opened_at="2026-08-25T11:00:00Z")
    assert receipt["labels"] == list(seal.confirm_labels)
    assert receipt["p_values"] == [_p_values()[label] for label in seal.confirm_labels]


def test_a_confirmation_that_rejects_nothing_is_a_receipt_like_any_other():
    """A null is the expected outcome of an honest split and must not look like a failure."""
    ledger = HeldOutLedger()
    seal = _seal(ledger=ledger)
    receipt = confirm_on_held_out(seal, p_values=_p_values(first=0.28), held_out=_partition(),
                                  ledger=ledger, opened_at="2026-08-25T11:00:00Z")
    assert receipt["n_rejected"] == 0 and receipt["rejected_labels"] == []
    assert receipt["correction_unit"] == 4
    assert len(ledger.records) == 1
