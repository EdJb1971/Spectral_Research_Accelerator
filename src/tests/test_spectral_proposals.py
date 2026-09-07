"""T4F.8: a proposal is a test, or it is a suggestion, and seven ways of telling them apart.

1.  **The proposals are built on rules the record actually produced.** The T4F.7 acceptance
    record is re-used because it already carries both things this task needs: real precursor
    signatures in regions `A`, `B` and `C`, and a real negative in `D`, which carries the same
    two structures with the consequent six frames *before* the antecedent. So the re-test
    proposals below are proposals about a finding, and the power proposals are proposals about
    an absence, rather than about hand-written figures.

2.  **A proposal that cannot come back negative is refused.** The refutation threshold is the
    base rate, and a Wilson upper bound is strictly positive, so a base rate of zero leaves a
    condition no outcome could satisfy. That case is constructed and refused here.

3.  **The prediction is digested before the record is read.** Signing a registration must not
    move its digest, or the digest would identify the signature rather than the design; changing
    any part of the design must move it, or it would identify nothing.

4.  **A rule that did not clear its null gets no re-test**, and a rule that did gets no power
    proposal. Both directions are checked, because a proposal engine that only ever fires on
    successes has publication bias built in and one that fires on everything has no judgement.

5.  **A negative from an adequately powered study is a result.** `D` needs 32 occurrences for a
    doubling of its base rate and carried 24, so its power proposal stands. Ask for a smaller
    effect and the study was already big enough, and the proposal is refused as chasing.

6.  **The required occurrence count is monotone in `n` and this suite pins it.** The design
    interval is taken at `p * n` rather than at a whole number of occurrences: rounding makes
    separability non-monotone, so 336 trials can fail a separation that 335 passes and a search
    for the smallest sufficient design returns an arbitrary member of a jagged set.

7.  **The two design quantities are readable without running the test.** The eligible-anchor
    count and the base rate are properties of the record; neither counts the pair. The suite
    checks that the number a proposal asks for matches what the region's own report later
    measures, on a region the proposal never saw counted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis_engine.spectral_clustering import (
    ConstellationPattern, PatternCatalogue, SignaturePoint, calibrate_signature_tolerance,
)
from src.analysis_engine.spectral_events import ObservationGrid
from src.analysis_engine.spectral_invariance import sign_constellations
from src.analysis_engine.spectral_precursors import (
    NULL_CIRCULAR_SHIFT, PRECURSOR_CLAIM_BOUNDARY, RULE_BASE_RATE_ZERO, RULE_PRECURSOR,
    eligible_anchors, wilson_interval,
)
from src.analysis_engine.spectral_projection import geography_of
from src.analysis_engine.spectral_proposals import (
    CONFIDENCE_RANGE, KIND_POWER, KIND_RE_TEST, MAXIMUM_SEARCHED_OCCURRENCES, OPTIMISER_NOTE,
    PROPOSAL_CLAIM_BOUNDARY, PROPOSAL_REFUSED, PROPOSAL_SCHEMA, PROPOSAL_TESTABLE,
    PROPOSAL_UNDERPOWERED, REGISTRATION_DRAFT, REGISTRATION_REGISTERED, REGISTRATION_SCHEMA,
    REFUTATION_CONDITION, REFUTATION_STATISTIC, Proposal, _refutation, describe_proposal,
    followup_experiment_config, propose_power_increase, propose_re_test, register,
    required_occurrences, separable_at,
)
from src.analysis_engine.spectral_reference import physical_bridge
from src.analysis_engine.spectral_regions import (
    Region, RegionPartition, ROLE_DISCOVERY, ROLE_HELD_OUT, assign_regions, identity_leakage,
    precursor_in_region, region_series,
)
from src.analysis_engine.spectral_sequences import TransitionWindow
from src.core.errors import InvalidParameterError
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence
from src.tests.test_spectral_regions import (
    ANTECEDENT, BOXES, CONSEQUENT, METRIC, PHYSIOGRAPHY, SEED, STEPS, SURROGATES, WINDOW,
    _partition, _record,
)
from src.transform_engine.coefficient_field import decompose_sequence

ALPHA = 0.05
CORRECTION = "benjamini_yekutieli"

_CACHE: dict = {}


def _catalogue(constellations):
    """The declared band-pair catalogue T4F.7's suite builds, rebuilt here for the same reason.

    Declared rather than fitted, because T4E.2's signature is invariant to rotation and cannot
    tell the two band orientations apart -- so `fitted=()` below is the truthful answer and the
    leakage check has something real to be clean about.
    """
    by_key = {tuple(item.key()): item for item in constellations}
    signatures = sign_constellations(constellations, scale_invariant=False).signatures
    groups: dict = {}
    for signature in signatures:
        bands = by_key[tuple(signature.key)].bands
        if len(set(bands)) == 1:
            groups.setdefault(bands, []).append(signature)
    tolerance = calibrate_signature_tolerance(sorted(groups.values(), key=len)[-1][:3],
                                              metric=METRIC)
    patterns = []
    for index, bands in enumerate(sorted(groups)):
        members = tuple(sorted(groups[bands], key=lambda item: (item.time, item.key)))
        points = [SignaturePoint.from_signature(member) for member in members]
        patterns.append(ConstellationPattern(
            pattern_id=index + 1, members=members, centroid=points[0],
            tolerance_radius=tolerance.value,
            observed_radius=max(METRIC.distance(point, points[0]) for point in points)))
    return PatternCatalogue(patterns=tuple(patterns), metric=METRIC, tolerance=tolerance,
                            n_signatures=sum(len(item.members) for item in patterns))


def _ground():
    """Every region's own series and its own T4F.3 rule, computed once (about half a minute)."""
    if "ground" not in _CACHE:
        _sequence, _field, constellations, geography = _record()
        grid = ObservationGrid(frames=tuple(float(i) for i in range(STEPS)),
                               time_units="frames", cadence=1.0)
        partition = _partition()
        assignment = assign_regions(constellations, partition, geography)
        catalogue = _catalogue(constellations)
        leakage = identity_leakage(catalogue, assignment, partition, fitted=())
        series, rules = {}, {}
        for name in "ABCDE":
            found = region_series(catalogue, assignment, name, grid=grid)
            series[name] = found
            if found is None:
                continue
            report = precursor_in_region(
                found, antecedent=ANTECEDENT, consequent=CONSEQUENT, window=WINDOW,
                n_surrogates=SURROGATES, seed=SEED, alpha=ALPHA, correction=CORRECTION,
                null=NULL_CIRCULAR_SHIFT)
            rules[name] = report.rule(ANTECEDENT, CONSEQUENT, WINDOW)
        _CACHE["ground"] = {
            "partition": partition, "series": series, "rules": rules, "leakage": leakage,
            "bridge": physical_bridge(geography, variable="amplitude"), "grid": grid}
    return _CACHE["ground"]


@pytest.fixture(scope="module")
def ground():
    return _ground()


@pytest.fixture(scope="module")
def partition(ground):
    return ground["partition"]


@pytest.fixture(scope="module")
def signature(ground):
    """`A`'s rule: a real precursor signature, and the thing a re-test is proposed for."""
    return ground["rules"]["A"]


@pytest.fixture(scope="module")
def negative(ground):
    """`D`'s rule: the same two structures in the wrong order, not distinguished from its null."""
    return ground["rules"]["D"]


@pytest.fixture(scope="module")
def dated_bridge():
    """A small record that carries a calendar, so the lead can be a number of hours."""
    if "dated" not in _CACHE:
        grid = GridSpec.cartesian((32, 32), dy_m=31000.0)
        generator = torch.Generator().manual_seed(SEED)
        times = np.datetime64("2026-01-01T00", "ns") + np.arange(8) * np.timedelta64(6, "h")
        sequence = FieldSequence(
            [PhysicalField(torch.randn(32, 32, generator=generator, dtype=torch.float64),
                           grid=grid, units="dimensionless") for _ in range(8)], times)
        field = decompose_sequence(sequence, "swt",
                                   {"levels": 2, "wavelet": "haar"}).select(scales=[1])
        _CACHE["dated"] = physical_bridge(
            geography_of(field, time_units="seconds", values=sequence), variable="amplitude")
    return _CACHE["dated"]


def _re_test(ground, rule, target="B", **kwargs):
    options = {"partition": ground["partition"], "supply": ground["series"].get(target),
               "leakage": ground["leakage"], "bridge": ground["bridge"], "alpha": ALPHA,
               "correction": CORRECTION}
    options.update(kwargs)
    return propose_re_test(rule, target=target, **options)


# ---------------------------------------------------------------------------------------------
# section 1: the record the proposals are made about
# ---------------------------------------------------------------------------------------------


def test_the_record_supplies_a_real_signature_and_a_real_negative(ground):
    """Everything below rests on these two being genuinely different results."""
    rules = ground["rules"]
    assert rules["A"].status == RULE_PRECURSOR
    assert rules["D"].status != RULE_PRECURSOR
    assert rules["A"].confidence == 0.5
    assert rules["A"].support == 38 and rules["A"].eligible_antecedents == 76
    assert rules["D"].support == 0 and rules["D"].eligible_antecedents == 24
    assert ground["series"]["E"] is None
    assert ground["leakage"].clean


def test_the_design_quantities_are_readable_without_counting_the_pair(ground):
    """Point 6 of the module docstring, checked on a region the proposal never saw counted."""
    series = ground["series"]["C"]
    anchors = eligible_anchors(series, ANTECEDENT, WINDOW)
    assert len(anchors) == ground["rules"]["C"].eligible_antecedents
    proposal = _re_test(ground, ground["rules"]["A"], target="C")
    assert proposal.power.available == len(anchors)
    assert proposal.power.base_rate == ground["rules"]["C"].base_rate


# ---------------------------------------------------------------------------------------------
# section 2: a proposal that cannot come back negative
# ---------------------------------------------------------------------------------------------


def test_a_re_test_carries_a_named_refutation_with_a_threshold(ground, signature):
    proposal = _re_test(ground, signature)
    assert proposal.refutation.statistic == REFUTATION_STATISTIC
    assert proposal.refutation.condition == REFUTATION_CONDITION
    assert proposal.refutation.reachable
    assert proposal.refutation.threshold == ground["rules"]["B"].base_rate
    assert list(CONFIDENCE_RANGE) == [proposal.refutation.attainable_low,
                                      proposal.refutation.attainable_high]


def test_a_base_rate_of_zero_leaves_no_reachable_refutation_and_the_proposal_is_refused(
        ground, signature):
    """A Wilson upper bound is strictly positive, so 'at or below zero' can never be met."""
    assert wilson_interval(0, 1000, 0.95)[1] > 0.0
    empty = ObservationGrid(frames=tuple(float(i) for i in range(STEPS)), time_units="frames",
                            cadence=1.0)
    barren = _BarrenSeries(ground["series"]["B"], consequent=CONSEQUENT, grid=empty)
    proposal = _re_test(ground, signature, supply=barren)
    assert proposal.status == PROPOSAL_REFUSED
    assert proposal.power is not None and proposal.power.base_rate == 0.0
    assert any("unreachable" in reason for reason in proposal.reasons)
    assert any("strictly above zero" in reason for reason in proposal.reasons)


def test_a_refused_proposal_yields_no_runnable_config(ground, signature):
    proposal = _re_test(ground, signature, target="A")
    assert proposal.status == PROPOSAL_REFUSED
    with pytest.raises(InvalidParameterError) as excinfo:
        followup_experiment_config(proposal, {"parameter_matrix": {}})
    assert "did not refuse" in str(excinfo.value)


def test_a_refused_proposal_cannot_be_registered(ground, signature):
    proposal = _re_test(ground, signature, target="A")
    with pytest.raises(InvalidParameterError) as excinfo:
        register(proposal, alpha=ALPHA, correction=CORRECTION, null=NULL_CIRCULAR_SHIFT,
                 n_surrogates=SURROGATES)
    assert "cannot answer" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# section 3: the ground a follow-up may be proposed on
# ---------------------------------------------------------------------------------------------


def test_the_discovery_region_is_not_a_re_test(ground, signature):
    proposal = _re_test(ground, signature, target="A")
    assert proposal.status == PROPOSAL_REFUSED
    assert any("is the discovery region" in reason for reason in proposal.reasons)


def test_a_region_the_partition_does_not_declare_is_refused(ground, signature):
    proposal = _re_test(ground, signature, target="Z", supply=ground["series"]["B"])
    assert proposal.status == PROPOSAL_REFUSED
    assert any("not in this partition" in reason for reason in proposal.reasons)


def test_leaked_identity_refuses_the_proposal_however_good_the_ground_is(ground, signature):
    """R6's leak with a map in place of a calendar: the identity was fitted where it is re-tested."""
    clean = _re_test(ground, signature)
    assert clean.status == PROPOSAL_TESTABLE
    leaked = _LeakedIdentity(n_leaked=3)
    proposal = _re_test(ground, signature, leakage=leaked)
    assert proposal.status == PROPOSAL_REFUSED
    assert any("partly defined by the data it would be re-tested on" in reason
               for reason in proposal.reasons)


def test_the_partition_digest_travels_with_the_proposal(ground, signature, partition):
    proposal = _re_test(ground, signature)
    assert proposal.partition_digest == partition.digest
    other = RegionPartition(
        tuple(Region(name, BOXES[name][0], BOXES[name][1],
                     ROLE_DISCOVERY if name == "A" else ROLE_HELD_OUT, PHYSIOGRAPHY[name])
              for name in "ABC"),
        declared_by="a different design", declared_on="2026-09-07")
    assert other.digest != partition.digest
    assert _re_test(ground, signature, partition=other).partition_digest == other.digest


# ---------------------------------------------------------------------------------------------
# section 4: which rules get which kind of proposal
# ---------------------------------------------------------------------------------------------


def test_a_rule_that_did_not_clear_its_null_gets_no_re_test(ground, negative):
    proposal = _re_test(ground, negative)
    assert proposal.status == PROPOSAL_REFUSED
    assert any("manufacture a discovery out of a negative result" in reason
               for reason in proposal.reasons)


def test_a_rule_that_cleared_its_null_gets_no_power_proposal(ground, signature):
    proposal = propose_power_increase(signature, supply=ground["series"]["A"], alpha=ALPHA,
                                      correction=CORRECTION)
    assert proposal.status == PROPOSAL_REFUSED
    assert proposal.reasons == (
        "this rule cleared its null, so there is no absence to interpret. A power increase for "
        "a result that already rejected would be asking for data to strengthen a finding rather "
        "than to make an uninformative one informative",)


def test_a_power_proposal_carries_no_prediction_and_no_refutation(ground, negative):
    proposal = propose_power_increase(negative, supply=ground["series"]["D"], alpha=ALPHA,
                                      correction=CORRECTION)
    assert proposal.kind == KIND_POWER
    assert proposal.status == PROPOSAL_TESTABLE
    assert proposal.prediction is None and proposal.refutation is None
    record = describe_proposal(proposal)
    assert record["prediction"] is None and record["refutation"] is None
    assert "can confirm nothing" in followup_experiment_config(
        proposal, {"parameter_matrix": {}})["description"]


def test_a_negative_from_an_adequately_powered_study_is_a_result_not_a_gap(ground, negative):
    """`D` carried 24 occurrences. A doubling needs 32, so the proposal stands; a smaller
    effect needs fewer than 24, and asking for more data to chase it is refused."""
    standing = propose_power_increase(negative, supply=ground["series"]["D"], alpha=ALPHA,
                                      correction=CORRECTION)
    assert standing.power.available == 24 and standing.power.required == 32
    assert standing.status == PROPOSAL_TESTABLE
    chasing = propose_power_increase(negative, supply=ground["series"]["D"],
                                     target_confidence=0.8, alpha=ALPHA, correction=CORRECTION)
    assert chasing.power.required <= 24
    assert chasing.status == PROPOSAL_REFUSED
    assert any("Asking for more data until it changes is chasing it" in reason
               for reason in chasing.reasons)


def test_a_negative_from_a_study_that_could_not_reject_is_still_a_gap(ground, negative):
    """The occurrence count is only half of it: an ensemble too small to reject after correction
    could not have produced a significant result whatever the record carried."""
    proposal = propose_power_increase(negative, supply=ground["series"]["D"],
                                      target_confidence=0.8, alpha=ALPHA,
                                      correction=CORRECTION, n_surrogates=5, family_size=40)
    assert proposal.power.required <= proposal.power.available
    assert not proposal.power.surrogate_power["can_reject_after_correction"]
    assert proposal.status == PROPOSAL_UNDERPOWERED
    assert not any("chasing it" in reason for reason in proposal.reasons)
    assert any("cannot reject after correction whatever the record supplies" in reason
               for reason in proposal.reasons)


def test_a_power_proposal_on_an_unreadable_ground_borrows_no_base_rate_either(ground,
                                                                             negative):
    """The same wall as the re-test's: a supplied record that gave no base rate gets none."""
    undated = ObservationGrid(frames=tuple(float(i) for i in range(STEPS)),
                              time_units="frames", cadence=None)
    proposal = propose_power_increase(
        negative, supply=_ThinSupply(ground["series"]["D"], keep=10 ** 9, grid=undated),
        alpha=ALPHA, correction=CORRECTION)
    assert proposal.status == PROPOSAL_REFUSED
    assert proposal.power is None
    assert any("no base rate on this ground" in reason for reason in proposal.reasons)
    assert any("no number of occurrences that would make the absence mean anything" in reason
               for reason in proposal.reasons)


def test_the_default_effect_size_is_declared_and_not_taken_from_the_observation(ground,
                                                                                negative):
    """Sizing a study on the effect the record happened to show is sizing it on noise."""
    proposal = propose_power_increase(negative, supply=ground["series"]["D"], alpha=ALPHA,
                                      correction=CORRECTION)
    assert proposal.power.predicted_confidence == pytest.approx(
        2.0 * proposal.power.base_rate)
    assert negative.confidence == 0.0
    assert proposal.power.predicted_confidence != negative.confidence


# ---------------------------------------------------------------------------------------------
# section 5: how many occurrences it would take
# ---------------------------------------------------------------------------------------------


def test_separability_needs_both_directions_and_neither_is_the_binding_one():
    """A design that can confirm and cannot retract is what this module exists to stop.

    Requiring both matters only if the two conditions come apart, and they do -- in **both**
    directions. Sweeping every pair of proportions to two decimal places, 2,052 pairs have some
    `n` that could detect the effect but could not retract it, and 1,973 have some `n` the other
    way about. Neither condition subsumes the other, so a design sized on one of them is
    systematically too small about half the time.
    """
    detect_only = [n for n in range(1, 400)
                   if separable_at(n, 0.35, 0.30, ALPHA)["detectable"]
                   and not separable_at(n, 0.35, 0.30, ALPHA)["refutable"]]
    refute_only = [n for n in range(1, 400)
                   if separable_at(n, 0.54, 0.47, ALPHA)["refutable"]
                   and not separable_at(n, 0.54, 0.47, ALPHA)["detectable"]]
    assert detect_only[:5] == [323, 324, 325, 326, 327]
    assert refute_only[:1] == [195]
    for n in detect_only:
        assert required_occurrences(0.35, 0.30, ALPHA) > n
    for n in refute_only:
        assert required_occurrences(0.54, 0.47, ALPHA) > n


def test_the_effect_interval_is_read_at_its_lower_bound_and_the_null_at_its_upper():
    """Reading either from the wrong end would size the study on the half of the interval that
    cannot rule anything out."""
    n, predicted, base = 24, 0.5, 0.3
    level = 1.0 - ALPHA
    effect = wilson_interval(predicted * n, n, level)
    null = wilson_interval(base * n, n, level)
    assert effect[0] > base, "the effect interval already excludes the base rate at n=24"
    assert null[1] == pytest.approx(0.50004, abs=1e-4)
    assert null[1] > predicted, "and the null interval does not yet exclude the prediction"
    assert separable_at(n, predicted, base, ALPHA) == {"detectable": True, "refutable": False}
    # One more occurrence closes it, and 0.00004 is the whole margin: reading either interval
    # from the wrong end would have sized this study at 24 rather than 25.
    assert separable_at(25, predicted, base, ALPHA) == {"detectable": True, "refutable": True}
    assert required_occurrences(predicted, base, ALPHA) == 25


def test_the_required_count_is_monotone_in_n_and_the_search_returns_the_smallest(ground):
    """Point 6: the design interval is taken at `p * n`, not at a whole number of occurrences."""
    for predicted, base in ((0.9, 0.3), (0.5, 0.3), (0.35, 0.3), (0.25, 0.1), (0.99, 0.9)):
        separable = [n for n in range(1, 900)
                     if all(separable_at(n, predicted, base, ALPHA).values())]
        assert separable
        assert separable == list(range(separable[0], 900)), "separability is not monotone in n"
        assert required_occurrences(predicted, base, ALPHA) == separable[0]


def test_the_required_count_is_pinned_on_this_record(ground, signature):
    proposal = _re_test(ground, signature)
    assert proposal.power.predicted_confidence == 0.5
    assert proposal.power.base_rate == ground["rules"]["B"].base_rate
    assert proposal.power.required == 8
    assert proposal.power.available == 81
    assert proposal.power.shortfall == 0
    assert proposal.power.met


def test_a_refutation_is_unreachable_wherever_its_threshold_cannot_be_met():
    """Three ways the condition is a form of words, checked on the constructor itself."""
    assert _refutation(0.0).reachable is False
    assert "strictly above zero" in _refutation(0.0).note
    assert _refutation(None).reachable is False
    assert "no number for a re-test's confidence to fail to clear" in _refutation(None).note
    assert _refutation(1.4).reachable is False
    assert "met by every outcome and distinguishes nothing" in _refutation(1.4).note
    assert _refutation(0.15).reachable is True


def test_a_ground_with_no_base_rate_at_all_refuses_the_proposal(ground, signature):
    """A record whose grid declares no cadence has no base rate to fail to clear."""
    undated = ObservationGrid(frames=tuple(float(i) for i in range(STEPS)),
                              time_units="frames", cadence=None)
    proposal = _re_test(ground, signature,
                        supply=_ThinSupply(ground["series"]["B"], keep=10 ** 9,
                                           grid=undated))
    assert proposal.status == PROPOSAL_REFUSED
    assert proposal.power is None
    assert any("did not produce" in reason for reason in proposal.reasons)
    assert any("no number for a re-test's confidence to fail to clear" in reason
               for reason in proposal.reasons)
    assert any("nothing for an experiment to check it against" in reason
               for reason in proposal.reasons)


def test_a_prediction_no_better_than_the_base_rate_has_nothing_to_separate():
    assert required_occurrences(0.30, 0.30, ALPHA) is None
    assert required_occurrences(0.20, 0.30, ALPHA) is None


def test_two_proportions_too_close_to_separate_return_no_number_rather_than_a_huge_one():
    """A number here would invite an acquisition that could not succeed."""
    assert required_occurrences(0.3001, 0.3, ALPHA, ceiling=1000) is None
    assert required_occurrences(0.35, 0.3, ALPHA, ceiling=MAXIMUM_SEARCHED_OCCURRENCES) == 350


def test_a_target_that_supplies_nothing_yet_names_the_acquisition_it_needs(ground, signature):
    proposal = _re_test(ground, signature, target="C", supply=None)
    assert proposal.status == PROPOSAL_UNDERPOWERED
    assert proposal.power.available is None
    assert proposal.power.base_rate == signature.base_rate
    assert "assumed for the target" in proposal.power.base_rate_source
    assert any("acquisition carrying at least %d" % proposal.power.required in reason
               for reason in proposal.reasons)


def test_a_target_that_falls_short_names_the_shortfall(ground, signature):
    thin = _ThinSupply(ground["series"]["B"], keep=3)
    proposal = _re_test(ground, signature, supply=thin)
    assert proposal.status == PROPOSAL_UNDERPOWERED
    assert proposal.power.available == 3
    assert proposal.power.shortfall == proposal.power.required - 3
    assert any("a shortfall of %d" % proposal.power.shortfall in reason
               for reason in proposal.reasons)
    assert any("supplies 3 eligible antecedent occurrences" in reason
               for reason in proposal.reasons)


def test_a_surrogate_ensemble_too_small_to_reject_makes_the_proposal_underpowered(ground,
                                                                                  signature):
    proposal = _re_test(ground, signature, n_surrogates=5, family_size=40)
    assert proposal.status == PROPOSAL_UNDERPOWERED
    assert not proposal.power.surrogate_power["can_reject_after_correction"]
    assert proposal.power.available >= proposal.power.required
    assert any("cannot reject after correction" in reason for reason in proposal.reasons)


# ---------------------------------------------------------------------------------------------
# section 6: the registration
# ---------------------------------------------------------------------------------------------


def test_the_digest_covers_the_declared_design_and_not_what_the_record_turned_out_to_say(
        ground, signature, dated_bridge):
    """Two proposals of one design, read through different records, share a digest.

    The lead is a property of the record that was read, and the status and the reasons are what
    this module concluded. None of them was declared in advance, so a digest that moved with
    them would identify the reading rather than the design.
    """
    plain = _re_test(ground, signature, target="C", supply=None)
    dated = _re_test(ground, signature, target="C", supply=None, bridge=dated_bridge)
    assert plain.lead.measured is False and dated.lead.measured is True
    assert plain.design() == dated.design()
    kwargs = {"alpha": ALPHA, "correction": CORRECTION, "null": NULL_CIRCULAR_SHIFT,
              "n_surrogates": SURROGATES}
    assert register(plain, **kwargs).digest == register(dated, **kwargs).digest


def test_the_design_names_the_ground_it_was_declared_for(ground, signature):
    for target in ("B", "C"):
        design = _re_test(ground, signature, target=target).design()
        assert design["target_region"] == target
        assert design["rule"]["antecedent"] == ANTECEDENT
        assert design["rule"]["window"] == [1.0, 3.0, "frames"]


def test_a_correction_this_programme_does_not_implement_is_refused_at_registration(
        ground, signature):
    with pytest.raises(InvalidParameterError) as excinfo:
        register(_re_test(ground, signature), alpha=ALPHA,
                 correction="the one that gives the answer I want",
                 null=NULL_CIRCULAR_SHIFT, n_surrogates=SURROGATES)
    assert "one of" in str(excinfo.value)


def test_signing_a_registration_does_not_move_its_digest(ground, signature):
    draft = register(_re_test(ground, signature), alpha=ALPHA, correction=CORRECTION,
                     null=NULL_CIRCULAR_SHIFT, n_surrogates=SURROGATES)
    assert draft.status == REGISTRATION_DRAFT
    signed = draft.register(registered_by="the T4F.8 suite", registered_on="2026-09-07")
    assert signed.status == REGISTRATION_REGISTERED
    assert signed.digest == draft.digest
    assert signed.registered_by == "the T4F.8 suite"
    assert draft.registered_by is None


def test_changing_any_part_of_the_design_moves_the_digest(ground, signature):
    base = register(_re_test(ground, signature), alpha=ALPHA, correction=CORRECTION,
                    null=NULL_CIRCULAR_SHIFT, n_surrogates=SURROGATES)
    others = [
        register(_re_test(ground, signature, target="C"), alpha=ALPHA, correction=CORRECTION,
                 null=NULL_CIRCULAR_SHIFT, n_surrogates=SURROGATES),
        register(_re_test(ground, signature), alpha=0.01, correction=CORRECTION,
                 null=NULL_CIRCULAR_SHIFT, n_surrogates=SURROGATES),
        register(_re_test(ground, signature), alpha=ALPHA, correction="bonferroni",
                 null=NULL_CIRCULAR_SHIFT, n_surrogates=SURROGATES),
        register(_re_test(ground, signature), alpha=ALPHA, correction=CORRECTION,
                 null="uniform_antecedent_relocation", n_surrogates=SURROGATES),
        register(_re_test(ground, signature), alpha=ALPHA, correction=CORRECTION,
                 null=NULL_CIRCULAR_SHIFT, n_surrogates=SURROGATES + 1),
    ]
    digests = {base.digest} | {item.digest for item in others}
    assert len(digests) == len(others) + 1


def test_the_code_will_not_sign_a_registration(ground, signature):
    draft = register(_re_test(ground, signature), alpha=ALPHA, correction=CORRECTION,
                     null=NULL_CIRCULAR_SHIFT, n_surrogates=SURROGATES)
    for kwargs in ({"registered_by": "  ", "registered_on": "2026-09-07"},
                   {"registered_by": "someone", "registered_on": ""},
                   {"registered_by": None, "registered_on": "2026-09-07"}):
        with pytest.raises(InvalidParameterError) as excinfo:
            draft.register(**kwargs)
        assert "cannot make it on their behalf" in str(excinfo.value)


def test_an_underpowered_proposal_may_still_be_registered(ground, signature):
    """It is a design somebody may want to fund. Only a refusal is unregistrable."""
    proposal = _re_test(ground, signature, target="C", supply=None)
    assert proposal.status == PROPOSAL_UNDERPOWERED
    assert register(proposal, alpha=ALPHA, correction=CORRECTION, null=NULL_CIRCULAR_SHIFT,
                    n_surrogates=SURROGATES).digest


def test_the_registration_record_says_what_its_digest_leaves_out(ground, signature):
    signed = register(_re_test(ground, signature), alpha=ALPHA, correction=CORRECTION,
                      null=NULL_CIRCULAR_SHIFT, n_surrogates=SURROGATES).register(
        registered_by="the T4F.8 suite", registered_on="2026-09-07")
    record = signed.as_record()
    assert record["schema"] == REGISTRATION_SCHEMA
    assert "the proposal's status, the signatory and the date" in record["digest_excludes"]
    assert record["proposal"]["schema"] == PROPOSAL_SCHEMA
    assert record["declared"] == {"alpha": ALPHA, "correction": CORRECTION,
                                  "null": NULL_CIRCULAR_SHIFT, "n_surrogates": SURROGATES}


# ---------------------------------------------------------------------------------------------
# section 7: the cost, and closing the loop
# ---------------------------------------------------------------------------------------------


def test_a_record_with_no_calendar_gets_a_refusal_rather_than_a_lead_in_hours(ground,
                                                                              signature):
    proposal = _re_test(ground, signature)
    assert not proposal.lead.measured
    assert "carries no calendar" in proposal.lead.refusal


def test_a_dated_record_gives_the_lead_in_its_own_hours(ground, signature, dated_bridge):
    proposal = _re_test(ground, signature, bridge=dated_bridge, supply=None, target="C")
    assert proposal.lead.measured
    assert (proposal.lead.low, proposal.lead.high) == (6.0, 18.0)
    assert proposal.lead.units == "hours"
    assert proposal.observable == dated_bridge.observable


def test_a_series_on_another_clock_is_refused_rather_than_converted(ground, signature,
                                                                    dated_bridge):
    """T4F.6's clock reconciliation, reached through a proposal."""
    proposal = _re_test(ground, signature, bridge=dated_bridge)
    assert not proposal.lead.measured
    assert "counted on a different clock" in proposal.lead.refusal


def test_no_bridge_means_no_lead_and_no_observable_rather_than_a_default(ground, signature):
    proposal = _re_test(ground, signature, bridge=None)
    assert proposal.lead is None and proposal.observable is None
    assert proposal.status == PROPOSAL_TESTABLE


def test_the_config_is_the_shape_the_experiment_engine_already_runs(ground, signature):
    parent = {"parameter_matrix": {"levels": [3, 4]}, "pipeline": ["decompose", "track"],
              "metadata": {"code_revision": "abc123", "dataset_version": "v3"}}
    config = followup_experiment_config(_re_test(ground, signature), parent,
                                        parent_id="exp-1", parent_name="the discovery run")
    assert set(config) == {"name", "description", "parameter_matrix", "pipeline", "metadata"}
    assert config["parameter_matrix"]["levels"] == [3, 4]
    assert config["parameter_matrix"]["precursor_antecedent"] == [ANTECEDENT]
    assert config["parameter_matrix"]["precursor_consequent"] == [CONSEQUENT]
    assert config["parameter_matrix"]["region"] == ["B"]
    assert config["parameter_matrix"]["transition_window"] == [[1.0, 3.0]]
    assert config["pipeline"] == ["decompose", "track"]
    assert config["metadata"]["code_revision"] == "abc123"
    assert config["metadata"]["parent_experiment_id"] == "exp-1"
    assert parent["parameter_matrix"] == {"levels": [3, 4]}, "the parent was mutated"


def test_the_config_states_the_retraction_condition_in_words(ground, signature):
    config = followup_experiment_config(_re_test(ground, signature), {"parameter_matrix": {}})
    assert "The finding is retracted if" in config["description"]
    assert REFUTATION_STATISTIC in config["description"]
    assert config["metadata"]["proposal"]["refutation"]["reachable"] is True


def test_the_config_carries_the_whole_proposal_so_a_runner_cannot_lose_it(ground, signature):
    config = followup_experiment_config(_re_test(ground, signature), {"parameter_matrix": {}})
    carried = config["metadata"]["proposal"]
    assert carried["claim_boundary"] == PROPOSAL_CLAIM_BOUNDARY
    assert carried["inherited_claim_boundary"] == PRECURSOR_CLAIM_BOUNDARY
    assert carried["power"]["required_eligible_antecedent_occurrences"] == 8


# ---------------------------------------------------------------------------------------------
# section 8: what the record says, and what it refuses
# ---------------------------------------------------------------------------------------------


def test_every_predicted_figure_is_named_as_a_prediction(ground, signature):
    record = describe_proposal(_re_test(ground, signature))
    prediction = record["prediction"]
    assert set(prediction) >= {"predicted_confidence", "predicted_lift",
                               "measured_on_the_discovery_ground", "these_are_predictions"}
    assert "Nothing here has been measured on the target ground" in \
        prediction["these_are_predictions"]
    assert prediction["predicted_confidence"] == signature.confidence
    assert prediction["measured_on_the_discovery_ground"]["support"] == signature.support


def test_the_claim_boundary_refuses_the_words_it_has_to(ground, signature):
    record = describe_proposal(_re_test(ground, signature))
    for word in ("cause", "driver", "mechanism", "trigger", "forecast", "intervention"):
        assert word in record["claim_boundary"]
    assert "registering one makes it traceable, not true" in record["claim_boundary"]
    assert "is not a prediction that the re-test will succeed" in record["claim_boundary"]


def test_the_record_says_the_existing_proposals_are_optimisers_and_not_tests(ground, signature):
    record = describe_proposal(_re_test(ground, signature))
    assert record["optimiser_note"] == OPTIMISER_NOTE
    assert "_propose_numerical_followup" in OPTIMISER_NOTE
    assert "must not be reported as replication" in OPTIMISER_NOTE


def test_the_record_carries_the_kind_and_the_status(ground, signature, negative):
    assert describe_proposal(_re_test(ground, signature))["kind"] == KIND_RE_TEST
    assert describe_proposal(_re_test(ground, signature))["status"] == PROPOSAL_TESTABLE
    power = propose_power_increase(negative, supply=ground["series"]["D"], alpha=ALPHA,
                                   correction=CORRECTION)
    assert describe_proposal(power)["kind"] == KIND_POWER


def test_the_design_a_digest_covers_excludes_the_status_and_the_lead(ground, signature):
    """The lead is a property of the record read, not part of what was declared in advance."""
    design = _re_test(ground, signature).design()
    assert "status" not in design and "lead" not in design and "reasons" not in design
    assert set(design) == {"schema", "kind", "rule", "target_region", "partition_digest",
                           "prediction", "refutation", "power",
                           "observable_the_target_must_carry"}


def test_anything_that_is_not_the_object_it_claims_to_be_is_refused(ground, signature):
    with pytest.raises(InvalidParameterError):
        propose_re_test(object(), target="B", partition=ground["partition"])
    with pytest.raises(InvalidParameterError):
        propose_re_test(signature, target="B", partition=object())
    with pytest.raises(InvalidParameterError):
        propose_re_test(signature, target="B", partition=ground["partition"],
                        correction="the one that gives the answer I want")
    with pytest.raises(InvalidParameterError):
        propose_power_increase(object())
    with pytest.raises(InvalidParameterError):
        describe_proposal(object())
    with pytest.raises(InvalidParameterError):
        register(object(), alpha=ALPHA, correction=CORRECTION, null=NULL_CIRCULAR_SHIFT,
                 n_surrogates=SURROGATES)
    with pytest.raises(InvalidParameterError):
        followup_experiment_config(_re_test(ground, signature), ["not", "a", "mapping"])


def test_a_target_confidence_that_is_not_a_proportion_is_refused(ground, negative):
    for value in (-0.1, 1.5, "half", float("nan")):
        with pytest.raises(InvalidParameterError):
            propose_power_increase(negative, supply=ground["series"]["D"],
                                   target_confidence=value)


# ---------------------------------------------------------------------------------------------
# the two stand-ins the suite needs, which stand in for records rather than for behaviour
# ---------------------------------------------------------------------------------------------


class _LeakedIdentity:
    """An `IdentityLeakage` that is not clean, without re-clustering a record to make one."""

    def __init__(self, n_leaked: int) -> None:
        self.n_leaked = n_leaked
        self.clean = False


class _ThinSupply:
    """A region's series with all but `keep` of the antecedent's occurrences removed.

    A real record with three occurrences would be a different record in every other respect too,
    and the thing under test is what a proposal does when the ground is thin -- so the ground is
    thinned rather than rebuilt.
    """

    def __init__(self, series, keep: int, grid=None) -> None:
        self._series = series
        self.grid = series.grid if grid is None else grid
        kept = 0
        self._events = []
        for event in series:
            if event.pattern_id == ANTECEDENT:
                if kept >= keep:
                    continue
                kept += 1
            self._events.append(event)

    def __iter__(self):
        return iter(self._events)

    def __len__(self):
        return len(self._events)

    def for_pattern(self, pattern_id):
        return [item for item in self._events if item.pattern_id == int(pattern_id)]


class _BarrenSeries(_ThinSupply):
    """A series whose consequent never occurs, so the base rate is zero and nothing can refute."""

    def __init__(self, series, consequent: int, grid) -> None:
        _ThinSupply.__init__(self, series, keep=10 ** 9)
        self._events = [item for item in self._events if item.pattern_id != int(consequent)]
        self.grid = grid
