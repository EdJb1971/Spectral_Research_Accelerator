"""T4E.8 slice 3: the identity target is declared, and inadmissible evidence is refused.

Slice 2 measured a candidate without stating which of three distinct questions it was
answering. These tests pin the declaration, the target/evidence admissibility rule whose
job is to catch that confusion mechanically, and the label sources behind each evidence
class. Nothing here approves a radius or chooses a target.
"""

from __future__ import annotations

import math

import pytest

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, SignatureMetric, calibrate_signature_tolerance, cluster_signatures,
)
from src.analysis_engine.spectral_constellation import extract_constellations
from src.analysis_engine.spectral_identity_audit import (
    EVIDENCE_CLASSES, IDENTITY_TARGETS, PROXY_LABEL_BOUNDARY, catalogue_labels,
    declare_identity_target, labelled_errors, radius_feasibility,
)
from src.analysis_engine.spectral_invariance import signature_for
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError, MissingParameterError, UnknownNameError
from src.core.feature import FeatureLocation, FeatureSet, Quantity, SpectralFeature
from src.core.tracking import MotionBounds, SearchVolume, Track, TrackingResult


ROW, COL = "row", "col"
AXES = (AxisSpec(ROW, "space", units="cells", periodic=False, ordinal=0),
        AxisSpec(COL, "space", units="cells", periodic=False, ordinal=1))
BOUNDS = MotionBounds(max_doublings=0.5)
SCALENE = ((0.0, 0.0), (0.0, 30.0), (40.0, 0.0))
RIGHT = ((0.0, 0.0), (0.0, 120.0), (160.0, 0.0))
METRIC = SignatureMetric(AttributeWeights(geometry=1.0, bearings=1.0, strengths=1.0, scales=1.0))


def _signature(points, *, jitter=((0.0, 0.0),) * 3, key_time=0.0, scale_invariant=False,
               magnitudes=(3.0, 5.0, 7.0), spatial_scales=(8.0, 8.0, 16.0)):
    features = []
    for index, ((row, col), shake, magnitude, spatial_scale) in enumerate(
            zip(points, jitter, magnitudes, spatial_scales)):
        features.append(SpectralFeature(
            domain="synthetic", dataset="t4e8_slice3", variable="amplitude",
            magnitude=Quantity(float(magnitude), None),
            location=FeatureLocation({ROW: row + shake[0] + 200.0, COL: col + shake[1] + 200.0},
                                     AXES),
            time=float(key_time), time_units="frames", representation="manual",
            spatial_scale=Quantity(float(spatial_scale), "cells"),
            provenance={"scale_label": index + 1, "orientation_label": "LH", "threshold": 1.0,
                        "threshold_sigma": 3.0, "phase": None,
                        "alignment_cross_scale_comparable": True}))
    tracks = tuple(Track(index, FeatureSet([feature]), {})
                   for index, feature in enumerate(features))
    tracking = TrackingResult(
        features=FeatureSet(features), associator="hungarian", alpha=0.05, bounds=BOUNDS,
        volume=SearchVolume({ROW: 1024.0, COL: 1024.0}, units="cells"),
        times=(float(key_time),), tracks=tracks)
    constellations = extract_constellations(tracking, cardinalities=(3,))
    assert len(constellations) == 1
    return signature_for(constellations[0], scale_invariant=scale_invariant)


def _replicates(points, scale_invariant=False, first_time=0.0):
    """Four observations of one configuration. `first_time` keeps signature keys distinct.

    A signature's key is its time and track ids, so two planted configurations observed at
    the same frames would collide and the catalogue could not tell them apart by key.
    """
    return tuple(_signature(points, jitter=((0.0, 0.0), (shake, -shake), (-shake, shake)),
                            key_time=first_time + index, scale_invariant=scale_invariant)
                 for index, shake in enumerate((0.0, 0.4, -0.4, 0.25)))


def _planted_catalogue():
    """Two configurations, each observed four times. A reviewed catalogue holds them apart."""
    tolerance = calibrate_signature_tolerance(_replicates(SCALENE), metric=METRIC)
    signatures = list(_replicates(SCALENE)) + list(_replicates(RIGHT, first_time=10.0))
    catalogue = cluster_signatures(signatures, metric=METRIC, tolerance=tolerance)
    return catalogue, signatures


# --------------------------------------------------------------- the declaration is required

def test_an_absent_identity_target_is_refused_and_names_the_three_targets():
    with pytest.raises(MissingParameterError) as raised:
        declare_identity_target(None, "record_derived_proxy")
    message = str(raised.value)
    assert "identity_target" in message
    for name in ("track_continuity", "spatial_persistence", "kind_recurrence"):
        assert name in message


def test_an_absent_evidence_class_is_refused_and_names_the_classes():
    with pytest.raises(MissingParameterError) as raised:
        declare_identity_target("spatial_persistence", None)
    assert "record_derived_proxy" in str(raised.value)
    assert "external_reference" in str(raised.value)


@pytest.mark.parametrize("target,evidence,wrong", [
    ("spatial_persistance", "record_derived_proxy", "spatial_persistence"),
    ("spatial_persistence", "record_derived_prox", "record_derived_proxy"),
])
def test_a_misspelled_name_is_refused_with_its_correction(target, evidence, wrong):
    with pytest.raises(UnknownNameError) as raised:
        declare_identity_target(target, evidence)
    assert wrong in str(raised.value)


# ------------------------------------------------------------------- the circularity refusal

def test_kind_recurrence_against_record_derived_labels_is_refused_as_circular():
    with pytest.raises(InvalidParameterError) as raised:
        declare_identity_target("kind_recurrence", "record_derived_proxy")
    message = str(raised.value)
    assert "validated against itself" in message
    assert "same record and pipeline" in message
    assert "external_reference" in message


def test_kind_recurrence_is_admitted_against_an_external_reference():
    declaration = declare_identity_target("kind_recurrence", "external_reference")
    assert declaration["evidence_independent_of_record"] is True
    assert declaration["label_boundary"] != PROXY_LABEL_BOUNDARY


def test_track_continuity_on_proxy_labels_publishes_its_tracker_agreement_caveat():
    declaration = declare_identity_target("track_continuity", "record_derived_proxy")
    assert "agreement with the tracker" in declaration["caveat"]


def test_spatial_persistence_on_proxy_labels_publishes_its_drift_caveat():
    declaration = declare_identity_target("spatial_persistence", "record_derived_proxy")
    assert "scored as a split" in declaration["caveat"]


@pytest.mark.parametrize("target", ["track_continuity", "spatial_persistence", "kind_recurrence"])
def test_every_target_round_trips_what_it_recognises_and_what_it_does_not_license(target):
    evidence = IDENTITY_TARGETS.get(target).admissible_evidence[0]
    declaration = declare_identity_target(target, evidence)
    assert declaration["identity_target"] == target
    assert declaration["evidence_class"] == evidence
    for field in ("recognises", "does_not_license", "evidence_provenance", "label_boundary"):
        assert declaration[field] and isinstance(declaration[field], str)
    assert declaration["admissible_evidence"] == list(
        IDENTITY_TARGETS.get(target).admissible_evidence)


def test_only_kind_recurrence_demands_evidence_independent_of_the_record():
    independent = {name for name in IDENTITY_TARGETS.names()
                   if all(EVIDENCE_CLASSES.get(evidence).independent_of_record
                          for evidence in IDENTITY_TARGETS.get(name).admissible_evidence)}
    assert independent == {"kind_recurrence"}


# ------------------------------------------------- naming a target must not reword a receipt

def test_the_published_proxy_wording_is_unchanged_verbatim():
    """Slice 1 and 2 receipts are cited. A declaration must not silently reword them."""
    assert PROXY_LABEL_BOUNDARY == (
        "Tracked-key proxy labels, not independent measurements or physical ground truth")
    assert EVIDENCE_CLASSES.get("record_derived_proxy").label_boundary == PROXY_LABEL_BOUNDARY


def test_reports_carry_the_proxy_boundary_by_default_and_the_declared_one_when_given():
    default = labelled_errors([0.1, 0.2], [0.5, 0.6], 0.15)
    assert default["label_boundary"] == PROXY_LABEL_BOUNDARY
    declared = declare_identity_target("kind_recurrence", "external_reference")
    stated = labelled_errors([0.1, 0.2], [0.5, 0.6], 0.15, declared["label_boundary"])
    assert stated["label_boundary"] == declared["label_boundary"]
    assert stated["false_split_rate"] == default["false_split_rate"]
    feasibility = radius_feasibility([0.1, 0.2], [0.5, 0.6], max_split=0.1, max_admission=0.1,
                                     label_boundary=declared["label_boundary"])
    assert feasibility["errors"]["label_boundary"] == declared["label_boundary"]


# --------------------------------------------------------------- the external-reference path

def test_a_reviewed_catalogue_recovers_its_planted_identities():
    catalogue, signatures = _planted_catalogue()
    assert len(catalogue) == 2
    labels = catalogue_labels(catalogue, signatures, unrelated_pairs=8, seed=11)
    assert labels["metadata"]["distinct_patterns"] == 2
    # Four observations of each of two configurations: six within-pattern pairs each.
    assert labels["metadata"]["repeat_pairs"] == 12
    assert labels["metadata"]["unlabelled_signatures"] == 0
    for left, right in labels["unrelated"]:
        assert (left < 4) != (right < 4)


def test_catalogue_labels_refuse_a_catalogue_signed_under_another_family():
    catalogue, _ = _planted_catalogue()
    invariant = list(_replicates(SCALENE, scale_invariant=True)) + list(
        _replicates(RIGHT, scale_invariant=True, first_time=10.0))
    with pytest.raises(InvalidParameterError) as raised:
        catalogue_labels(catalogue, invariant, unrelated_pairs=4, seed=3)
    assert "must not be reported as total disagreement" in str(raised.value)


def test_catalogue_labels_refuse_to_draw_negatives_from_a_single_identity():
    tolerance = calibrate_signature_tolerance(_replicates(SCALENE), metric=METRIC)
    signatures = list(_replicates(SCALENE))
    catalogue = cluster_signatures(signatures, metric=METRIC, tolerance=tolerance)
    assert len(catalogue) == 1
    with pytest.raises(InvalidParameterError) as raised:
        catalogue_labels(catalogue, signatures, unrelated_pairs=4, seed=5)
    assert "at least two distinct reviewed patterns" in str(raised.value)


def test_a_configuration_the_catalogue_does_not_recognise_is_unlabelled_not_negative():
    catalogue, signatures = _planted_catalogue()
    stranger = _signature(((0.0, 0.0), (0.0, 61.0), (17.0, 0.0)), key_time=99.0)
    labels = catalogue_labels(catalogue, list(signatures) + [stranger],
                              unrelated_pairs=8, seed=13)
    assert labels["metadata"]["unlabelled_signatures"] == 1
    assert labels["metadata"]["labelled_signatures"] == len(signatures)
    stranger_index = len(signatures)
    for pair in labels["same"] + labels["unrelated"]:
        assert stranger_index not in pair
    assert "not negative" in labels["metadata"]["boundary"]


def test_catalogue_labels_refuse_a_negative_population_the_patterns_cannot_supply():
    """Rejection sampling terminates, but a hang and a refusal are not the same result.

    A catalogue whose support sits almost entirely in one identity makes cross-pattern
    draws vanishingly rare. The budget turns that into a named refusal rather than a
    process that appears to be working.
    """
    catalogue, signatures = _planted_catalogue()
    with pytest.raises(InvalidParameterError) as raised:
        catalogue_labels(catalogue, signatures, unrelated_pairs=8, seed=0,
                         attempts_per_negative=1)
    assert "support is too concentrated in one identity" in str(raised.value)
    # The same request succeeds once the budget is the production one.
    assert catalogue_labels(catalogue, signatures, unrelated_pairs=8, seed=0
                            )["metadata"]["repeat_pairs"] == 12


def test_catalogue_labels_refuse_something_that_is_not_a_catalogue():
    _, signatures = _planted_catalogue()
    with pytest.raises(InvalidParameterError):
        catalogue_labels({"patterns": []}, signatures, unrelated_pairs=4, seed=7)


def test_labels_and_errors_compose_into_a_report_that_states_its_provenance():
    catalogue, signatures = _planted_catalogue()
    declaration = declare_identity_target("kind_recurrence", "external_reference")
    labels = catalogue_labels(catalogue, signatures, unrelated_pairs=8, seed=11)
    from src.analysis_engine.spectral_clustering import SignaturePoint
    points = [SignaturePoint.from_signature(signature) for signature in signatures]
    same = [METRIC.distance(points[a], points[b]) for a, b in labels["same"]]
    unrelated = [METRIC.distance(points[a], points[b]) for a, b in labels["unrelated"]]
    report = labelled_errors(same, unrelated, max(same), declaration["label_boundary"])
    assert report["false_split_rate"] == 0.0
    assert report["false_admission_rate"] == 0.0
    assert report["auc"] == 1.0
    assert report["label_boundary"] == declaration["label_boundary"]
    assert math.isfinite(report["radius"])


# -------------------------------------- the target aimed at, and the ones measured on the way

def test_a_diagnostic_names_what_it_is_diagnostic_for_and_what_it_does_not_license():
    """The maintainer's declared position: kind_recurrence is the target, the others inform it.

    "The diagnostic passed" and "the target is met" are one careless sentence apart, so the
    boundary travels in the declaration rather than in a reader's memory.
    """
    declaration = declare_identity_target(
        "spatial_persistence", "record_derived_proxy",
        role="diagnostic", diagnostic_for="kind_recurrence")
    assert declaration["role"] == "diagnostic"
    assert declaration["diagnostic_for"] == "kind_recurrence"
    assert "does not license 'kind_recurrence'" in declaration["diagnostic_boundary"]


def test_a_primary_target_carries_no_diagnostic_boundary():
    declaration = declare_identity_target("kind_recurrence", "external_reference")
    assert declaration["role"] == "primary_scientific_target"
    assert declaration["diagnostic_for"] is None
    assert declaration["diagnostic_boundary"] is None


def test_calling_a_circular_evaluation_diagnostic_does_not_admit_it():
    """The role changes what is claimed from a result, never what evidence is admissible."""
    with pytest.raises(InvalidParameterError) as raised:
        declare_identity_target("kind_recurrence", "record_derived_proxy",
                                role="diagnostic", diagnostic_for="spatial_persistence")
    assert "validated against itself" in str(raised.value)


def test_a_diagnostic_without_a_target_is_refused():
    with pytest.raises(MissingParameterError) as raised:
        declare_identity_target("spatial_persistence", "record_derived_proxy",
                                role="diagnostic")
    assert "diagnostic_for" in str(raised.value)


def test_nothing_is_a_diagnostic_for_itself():
    with pytest.raises(InvalidParameterError) as raised:
        declare_identity_target("spatial_persistence", "record_derived_proxy",
                                role="diagnostic", diagnostic_for="spatial_persistence")
    assert "not a diagnostic for itself" in str(raised.value)


def test_a_primary_target_may_not_also_name_a_diagnostic_target():
    with pytest.raises(InvalidParameterError):
        declare_identity_target("kind_recurrence", "external_reference",
                                diagnostic_for="spatial_persistence")


def test_an_unknown_role_is_refused_and_names_the_roles():
    with pytest.raises(InvalidParameterError) as raised:
        declare_identity_target("spatial_persistence", "record_derived_proxy", role="exploratory")
    assert "primary_scientific_target" in str(raised.value)


def test_a_diagnostic_for_an_unknown_target_is_refused_with_its_correction():
    with pytest.raises(UnknownNameError):
        declare_identity_target("spatial_persistence", "record_derived_proxy",
                                role="diagnostic", diagnostic_for="kind_recurrance")


# ------------- T4E.17: the external catalogue kind_recurrence has always required
#
# kind_recurrence admits external_reference evidence alone, and no such evidence has ever
# existed in this programme, so the primary scientific target has been unevaluable from the
# tool. These tests hold the catalogue design: that it is not yet signed, that the data is
# bound by hash rather than committed, and above all why THIS catalogue is admissible when
# most atmospheric catalogues are not.


def _t4e17():
    import json
    from pathlib import Path

    return json.loads(Path(
        "data/identity_calibration/t4e17-external-catalogue-design.json"
    ).read_text(encoding="utf-8"))


def test_the_catalogue_design_is_not_signed_by_code():
    """Code does not sign catalogues, events or scientific preregistrations for a person."""
    body = _t4e17()

    assert body["status"] in ("declared_before_evaluation", "AMENDED_BEFORE_SIGNATURE")
    assert "SIGNED" in body["declared_by"]
    assert "NOT VALID until the maintainer has reviewed, adopted and SIGNED it" in (
        body["declared_by"])
    assert "Code does not sign catalogues for a person" in body["declared_by"]


def test_the_catalogue_is_bound_by_content_identity_and_not_committed():
    """35.5 MB of third-party data does not belong in the repository.

    The same discipline the acquired record and the market records are held to: bind by digest,
    do not commit, and treat a digest that fails to reproduce as invalidating the evaluation.
    """
    from pathlib import Path

    catalogue = _t4e17()["the_catalogue"]
    assert catalogue["sha256"] == (
        "631f76b95c77a6a4e409233466a0d501bb4848324e421fc58e228efea2086c44")
    assert catalogue["bytes"] == 35482417
    assert "must reproduce that digest" in catalogue["the_file_is_NOT_committed"]
    assert not list(Path("data").rglob("ibtracs*")), "the catalogue was committed"


def test_a_reanalysis_derived_catalogue_would_be_circular_and_the_design_says_so():
    """The crux of admissibility, and the thing most easily got wrong.

    Blocking indices, IMILAST-style track intercomparisons and most atmospheric-river
    catalogues are computed FROM reanalysis. Against an ERA5 record they are record-derived
    proxies wearing a catalogue's name, and admitting one would reintroduce exactly the
    circularity external_reference exists to exclude.
    """
    body = _t4e17()

    why = body["why_this_catalogue_is_admissible_as_external_reference"]
    assert "circular" in why["the_requirement"]
    assert "computed FROM reanalysis" in why["why_most_atmospheric_catalogues_FAIL_that_test"]
    assert "not a reanalysis product" in why["why_IBTrACS_passes"]
    assert "not independent of numerical weather prediction in general" in (
        why["the_residual_dependence_that_must_be_disclosed"])


def test_the_unit_of_independence_is_the_storm_and_not_the_observation():
    """20,241 pairs from 20 storms. The easiest way for a result here to be overclaimed."""
    population = _t4e17()["the_population_and_the_unit_of_independence"]

    assert "20,241 cross-storm pairs" in population["the_pairs_that_bear_on_kind_recurrence"]
    unit = population["THE_UNIT_OF_INDEPENDENCE_IS_THE_STORM_NOT_THE_OBSERVATION"]
    assert "only 20 storms" in unit
    assert "storm-clustered interval" in unit
    assert "nearer 20 than 20,241" in unit


def test_within_storm_pairs_are_excluded_because_they_are_a_different_target():
    """Pairs from one track bear on spatial_persistence, which has its own admissible evidence.

    Counting them here would answer the easier question and report it as the harder one.
    """
    population = _t4e17()["the_population_and_the_unit_of_independence"]

    assert "excluded by construction" in population["the_pairs_that_bear_on_kind_recurrence"]
    assert "spatial_persistence" in population["the_pairs_that_bear_on_kind_recurrence"]


def test_the_kind_label_is_fixed_before_any_signature_is_computed():
    """NATURE adjudicates; USA_SSHS is characterisation. Both base rates were seen, so choosing
    the more favourable one afterwards would be a horse race.
    """
    label = _t4e17()["the_kind_label"]

    assert label["primary"].startswith("NATURE")
    assert "intensity ordinal" in label["why_NATURE_and_not_intensity"]
    assert "adjudicates_nothing" in "".join(label.keys())
    assert "0.391" in label["base_rate_measured_before_declaring"]
    assert "0.182" in label["base_rate_measured_before_declaring"]


def test_disagreement_between_agencies_is_refused_and_not_treated_as_a_class():
    """MX means agencies disagreed and NR means nature was not reported.

    Treating either as a class would let the catalogue's own uncertainty enter as ground truth.
    """
    label = _t4e17()["the_kind_label"]

    assert "REFUSED BY NAME" in label["MX_and_NR_are_refusals_not_classes"]
    assert "neither rate" in label["MX_and_NR_are_refusals_not_classes"]


def test_the_forecast_test_period_is_not_opened_by_the_catalogue_design():
    """2022-2023 stays closed. The counts quoted for it come from the CATALOGUE, not the record."""
    body = _t4e17()

    domain = body["the_matching_domain"]
    assert domain["the_development_window_is_2018_2021_ONLY"].startswith("2022-2023")
    disclosure = domain["what_was_counted_in_the_reserved_period_and_why_that_is_disclosed"]
    assert "computed from the CATALOGUE, not from the record" in disclosure
    assert "no ERA5 frame of the forecast-test period was opened" in disclosure
    assert "No frame of the 2022-2023 forecast-test period is opened" in body["claim_boundary"]


def test_the_design_evaluates_no_criterion_and_says_so():
    """A catalogue is evidence, not a rule. The criterion for it needs its own declaration."""
    settles = _t4e17()["what_this_design_does_and_does_not_settle"]

    assert "It evaluates no criterion" in settles["what_it_does_NOT_do"][0]
    assert "seven synthetic candidates" in settles["what_it_does_NOT_do"][1]
    assert "Twenty storms" in settles["the_honest_ceiling"]


def test_the_publisher_s_citation_requirement_is_recorded_with_the_data():
    """An open-access catalogue's binding obligation is its citation, and it travels with it."""
    catalogue = _t4e17()["the_catalogue"]

    assert catalogue["doi"] == "10.25921/82ty-9e16"
    citations = " ".join(catalogue["citation_required_by_the_publisher"])
    assert "Gahtan" in citations and "Knapp" in citations
    assert "non-commercial" in catalogue["licence_position"]


# ------- T4E.17 amended before signature: the radius had no source, and the base rate was wrong
#
# Checked before asking the maintainer to sign. IBTrACS has 174 columns and none reports
# position uncertainty, so the design's stated radius source did not exist. What replaced it is
# the catalogue's own inter-agency disagreement. Applying that refusal, and the design's own
# MX/NR refusal, moved the adjudicating base rate from 0.391 to 0.571.


def test_the_radius_requirement_had_no_source_and_the_amendment_says_so():
    """The design named a column that does not exist. Signing it would have committed the
    maintainer to a matching parameter with no legitimate source.
    """
    amendment = _t4e17()["amendment_2026_09_10_the_radius_had_no_source"]

    assert "174 columns and none of them reports position uncertainty" in (
        amendment["why_this_amendment_exists"])
    assert "could not have been implemented" in amendment["why_this_amendment_exists"]
    assert "chosen by nobody" in amendment["what_replaces_it"]


def test_the_replacement_radius_is_supplied_by_the_catalogue_and_varies_per_observation():
    """Independent agencies disagree, and their disagreement is the catalogue's own statement."""
    measured = _t4e17()["amendment_2026_09_10_the_radius_had_no_source"]["the_measured_radius"]

    assert measured["median_km"] == 15.2
    assert measured["q95_km"] == 89.4
    assert "BELOW one grid cell" in measured["reading"]
    assert "carried per observation rather than summarised" in measured["reading"]


def test_a_single_agency_observation_is_refused_because_the_catalogue_supplies_no_radius():
    """34 of 210. Refused and counted, never given a default."""
    amendment = _t4e17()["amendment_2026_09_10_the_radius_had_no_source"]

    assert "176 have two or more agencies reporting and 34 have only one" in (
        amendment["coverage"])
    assert "not given a default" in amendment["the_34_are_REFUSED_BY_NAME"]


def test_the_adjudicating_base_rate_was_wrong_in_the_flattering_direction():
    """0.571, not 0.391, once MX and NR are refused as the design already required.

    Same-kind pairs are the majority, so a rule answering 'same kind' to everything would be
    right 57.1% of the time. Accuracy is therefore meaningless here and only the two-sided
    error rates may be reported.
    """
    revised = _t4e17()["amendment_2026_09_10_the_population_and_base_rates_are_revised"]

    assert revised["superseded_figures"]["nature_base_rate"] == 0.391
    assert revised["revised_figures"]["nature_base_rate"] == 0.571
    assert revised["revised_figures"]["storms"] == 18
    change = revised["THE_CHANGE_THAT_MATTERS"]
    assert "accuracy is a meaningless summary" in change
    assert "false split and false admission" in change


def test_the_more_favourable_label_was_not_adopted_after_seeing_both_base_rates():
    """USA_SSHS at 0.227 would make any criterion look better. Switching now is the horse race
    the design forbids, and NATURE was chosen on a principle the numbers do not touch.
    """
    revised = _t4e17()["amendment_2026_09_10_the_population_and_base_rates_are_revised"]

    assert revised["revised_figures"]["usa_sshs_base_rate"] == 0.227
    kept = revised["NATURE_REMAINS_THE_ADJUDICATING_LABEL"]
    assert "horse race the design forbids" in kept
    assert "what a system IS rather than how strong it is" in kept


def test_the_check_is_recorded_as_having_happened_before_signature():
    """The point of doing it first: the design is amended, not the result."""
    body = _t4e17()

    assert body["status"] == "AMENDED_BEFORE_SIGNATURE"
    assert "before signature" in body["what_this_check_cost_and_saved"]
    assert "wrong by 0.18 in the direction that would have made any later result look better" in (
        body["what_this_check_cost_and_saved"])
