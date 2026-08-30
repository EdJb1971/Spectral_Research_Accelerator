"""TG17.5: the whole declared search priced before acquisition, and mode-specific nulls.

Two things are being defended here. The first is that a multi-domain family is the product of
every axis a researcher can turn, counted once, in one place — not a formula copied into
preflight, a second into the browser and a third into a receipt. The second is that a null is a
declared object with a mode and a stated list of what it preserves, so a surrogate that destroys
the autocorrelation or the gaps can be refused by name rather than noticed later in a p-value.
"""

import json

import numpy as np
import pytest
from dataclasses import replace
from pydantic import ValidationError

from src.core.errors import InvalidParameterError
from src.core.experiment_family import (CandidateCountCorrectionError, COMPONENT_ORDER,
                                        correction_plan, declared_ensemble, domain_set_labels,
                                        domains_in, experiment_search_specification,
                                        family_expansion, precedence_availability,
                                        screened_search)
from src.core.experiment_manifest import (CrossDomainExperimentSpec, canonical_bytes,
                                          flagship_recipe, manifest_sha256, preflight_manifest)
from src.core.family import FamilyUnaffordableError
from src.core.structural_nulls import (NULL_FAMILIES, NULL_FEATURES, NullRefusal, bind_null,
                                       circular_clock_shift, nulls_for_mode,
                                       reassign_scale_partners, whole_cycle_clock_shift,
                                       within_group_clock_shift)


def _body(**changes):
    body = json.loads(canonical_bytes(flagship_recipe()))
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(body.get(key), dict):
            body[key] = {**body[key], **value}
        else:
            body[key] = value
    return body


def _spec(**changes):
    return CrossDomainExperimentSpec.parse_obj(_body(**changes))


def _confirmatory_only(**changes):
    """The flagship as a single corrected pass, which is what R18 prices at full size."""
    body = _body(**changes)
    body["confirmation"] = {"stage": "confirmatory_only", "held_out_partition": None,
                            "confirmatory_members": None}
    return CrossDomainExperimentSpec.parse_obj(body)


# --------------------------------------------------------------------- the declared family


def test_the_declared_family_is_the_product_of_every_declared_axis():
    search = experiment_search_specification(flagship_recipe())
    assert search.family_size == 288
    assert len(search.labels()) == 288
    assert len(set(search.labels())) == 288


def test_every_member_names_its_domain_set_and_its_relationship():
    search = experiment_search_specification(flagship_recipe())
    member = search.enumerate_family()[0]
    assert len(member) == len(COMPONENT_ORDER)
    assert domains_in(member[COMPONENT_ORDER.index("domain_set")]) == ("argo_float",
                                                                       "order_book")
    assert member[COMPONENT_ORDER.index("relationship")] == "co_occurrence"


def test_domain_combinations_are_unioned_across_arities_rather_than_multiplied():
    """Pairs and triples of four domains are 6 + 4 members, not 6 x 4."""
    labels = domain_set_labels(["a", "b", "c", "d"], [2, 3])
    assert len(labels) == 10
    assert labels[0] == "a+b" and labels[-1] == "b+c+d"


def test_a_combination_larger_than_the_declared_domains_is_refused():
    with pytest.raises(ValidationError, match="between 2 and the 4 declared domains"):
        _spec(family={**_body()["family"], "domain_arities": [2, 5]})


def test_testing_triples_as_well_as_pairs_is_priced_as_both_searches():
    spec = _spec(family={**_body()["family"], "domain_arities": [2, 3]})
    assert experiment_search_specification(spec).family_size == 480
    assert family_expansion(spec)["family_size"] == 480


def test_declaring_lags_multiplies_the_family_it_used_to_leave_out():
    """A lag axis is a knob, and a family priced without it is short by exactly that factor."""
    spec = _spec(family={**_body()["family"], "lags_seconds": [0.0, 3600.0, 7200.0]})
    assert experiment_search_specification(spec).family_size == 288 * 3


def test_a_repeated_axis_value_is_a_member_counted_twice_and_is_refused():
    with pytest.raises(ValidationError, match="counted twice"):
        _spec(family={**_body()["family"], "scales": [1.0, 2.0, 2.0, 4.0]})


def test_the_human_sentence_states_the_multiplication_it_performed():
    expansion = family_expansion(flagship_recipe())
    assert expansion["in_human_terms"] == ("6 domain sets x 3 windows x 4 channels x 4 scales "
                                           "x 1 relationship = 288 declared tests.")
    assert expansion["family_size"] == 288


def test_the_declared_axes_travel_inside_the_manifest_digest():
    before = manifest_sha256(flagship_recipe())
    after = manifest_sha256(_spec(family={**_body()["family"], "lags_seconds": [0.0, 3600.0]}))
    assert before != after


def test_adding_a_domain_and_adding_a_duration_are_priced_before_the_freeze():
    costs = {row["axis"]: row for row in family_expansion(flagship_recipe())["expansion_cost"]}
    assert costs["domain_set"]["family_size_after"] == 480
    assert costs["domain_set"]["members_added"] == 192
    assert costs["window"]["family_size_after"] == 384
    assert costs["window"]["members_added"] == 96
    assert all(not row["affordable_after"] for row in costs.values())


def test_the_declared_ensemble_is_the_smallest_declared_null():
    body = _body()
    body["nulls"] = [dict(body["nulls"][0], replications=2000),
                     dict(body["nulls"][0], name="second", replications=200)]
    assert declared_ensemble(CrossDomainExperimentSpec.parse_obj(body)) == 200


# --------------------------------------------------------------------------- the R18 gate


def test_a_family_the_declared_ensemble_cannot_resolve_is_refused_before_acquisition():
    spec = _confirmatory_only()
    plan = correction_plan(spec)
    assert plan["correction_unit_members"] == 288
    assert plan["affordable"] is False
    assert plan["surrogates_required"] == 35953
    report = preflight_manifest(spec)
    reason = next(row["reason"] for row in report["refusals"]
                  if row["reason"].startswith("R18"))
    assert "288 corrected tests at 200 surrogates" in reason
    assert "35953 surrogates are needed" in reason
    assert "largest affordable family at this ensemble is 4" in reason


def test_the_refusal_names_both_remedies_with_their_arithmetic():
    with pytest.raises(FamilyUnaffordableError) as error:
        experiment_search_specification(_confirmatory_only()).declare()
    message = str(error.value)
    assert "preregistered narrowing" in message
    assert "generate/confirm split" in message
    assert "35953" in message


def test_a_generate_confirm_split_is_priced_at_the_family_it_will_confirm():
    plan = correction_plan(flagship_recipe())
    assert plan["stage"] == "generate_then_confirm"
    assert plan["declared_search_members"] == 288
    assert plan["correction_unit_members"] == 4
    assert plan["affordable"] is True
    assert plan["generate_stage_makes_claims"] is False
    assert "candidates and not a result" in plan["note"]


def test_a_split_that_names_no_held_out_partition_is_refused():
    with pytest.raises(ValidationError, match="must name its held-out partition"):
        _spec(confirmation={"stage": "generate_then_confirm", "held_out_partition": None,
                            "confirmatory_members": 4})


def test_a_split_that_does_not_say_how_many_it_will_confirm_is_refused():
    with pytest.raises(ValidationError, match="how many members it will confirm"):
        _spec(confirmation={"stage": "generate_then_confirm",
                            "held_out_partition": "heldout", "confirmatory_members": None})


def test_a_confirmatory_only_study_may_not_name_a_partition_it_does_not_use():
    with pytest.raises(ValidationError, match="declares neither a held-out partition"):
        _spec(confirmation={"stage": "confirmatory_only", "held_out_partition": "heldout",
                            "confirmatory_members": None})


def test_confirming_more_members_than_were_generated_is_a_fresh_search():
    spec = _spec(confirmation={"stage": "generate_then_confirm",
                               "held_out_partition": "heldout",
                               "confirmatory_members": 400})
    with pytest.raises(InvalidParameterError, match="never generated"):
        correction_plan(spec)


# ------------------------------------------------------------------- screening and confirming


def test_a_pairwise_screen_does_not_shrink_the_correction_unit():
    screened = screened_search(flagship_recipe())
    assert screened.screening.family_size == 288  # four domains give only pairs anyway
    spec = _spec(family={**_body()["family"], "domain_arities": [2, 3]})
    screened = screened_search(spec)
    assert screened.screening.family_size == 288
    assert screened.complete.family_size == 480
    assert screened.correction_unit == 480


def test_correcting_over_the_survivors_of_a_screen_is_refused_by_name():
    screened = screened_search(flagship_recipe())
    with pytest.raises(CandidateCountCorrectionError) as error:
        screened.correct_over_candidates(7)
    message = str(error.value)
    assert "288 members, not the 7 that survived" in message
    assert "chosen by looking at the data" in message


def test_a_named_held_out_partition_is_what_permits_the_smaller_number():
    screened = screened_search(flagship_recipe())
    assert screened.correct_over_candidates(7, held_out="g17_flagship_heldout_v1") == 7
    with pytest.raises(InvalidParameterError, match="a named partition"):
        screened.correct_over_candidates(7, held_out="   ")


def test_the_screen_and_the_complete_search_are_different_declarations():
    spec = _spec(family={**_body()["family"], "domain_arities": [2, 3]})
    screened = screened_search(spec)
    assert screened.screening.fingerprint() != screened.complete.fingerprint()


# ----------------------------------------------------------------------------- precedence


def test_a_domain_without_a_lag_policy_makes_the_precedence_family_unavailable():
    spec = _spec(family={**_body()["family"],
                         "relationships": ["co_occurrence", "precedence"]})
    report = precedence_availability(spec)
    assert report["domains_without_precedence_policy"] == ["order_book", "tess_lightcurve"]
    # Of the six domain sets, five contain a domain with no justified lag policy; those five
    # are unavailable at the one precedence relationship, across every window, channel and scale.
    assert report["unavailable_precedence_members"] == 5 * 3 * 4 * 4
    assert report["association_members_unaffected"] == 576 - 240


def test_the_precedence_audit_never_reduces_the_declared_family():
    spec = _spec(family={**_body()["family"],
                         "relationships": ["co_occurrence", "precedence"]})
    report = precedence_availability(spec)
    assert report["declared_family_size"] == 576
    assert report["family_size_unchanged"] is True
    assert experiment_search_specification(spec).family_size == 576


def test_structural_association_survives_an_absent_precedence_policy():
    report = precedence_availability(flagship_recipe())
    assert report["precedence_relationships_declared"] == []
    assert report["unavailable_precedence_members"] == 0
    assert report["association_members_unaffected"] == 288


def test_a_family_entirely_of_precedence_over_a_refusing_domain_has_no_answerable_member():
    spec = _spec(family={**_body()["family"], "relationships": ["precedence"]})
    report = precedence_availability(spec, admissible={"reanalysis": False,
                                                       "argo_float": False,
                                                       "tess_lightcurve": False,
                                                       "order_book": False})
    assert report["entire_family_unavailable"] is True
    assert report["unavailable_precedence_members"] == 288
    assert report["association_members_unaffected"] == 0


# ---------------------------------------------------------------------------- null families


def test_every_registered_null_declares_its_mode_and_what_it_preserves():
    for entry in NULL_FAMILIES.entries():
        family = entry.value
        assert family.modes
        assert set(family.preserves) <= set(NULL_FEATURES)
        assert set(family.destroys) <= set(NULL_FEATURES)
        assert not set(family.preserves) & set(family.destroys)


def test_a_global_shuffle_is_registered_so_that_it_can_be_refused_by_name():
    family = NULL_FAMILIES.get("global_value_shuffle")
    assert family.admissible is False
    assert "autocorrelation" in family.destroys
    with pytest.raises(NullRefusal, match="Every domain can execute"):
        bind_null("global_value_shuffle", {}, mode="calendar_aligned")


def test_no_registered_adapter_admits_a_null_this_framework_refuses():
    from src.adapters import register_all_adapters
    from src.core.experiment_adapter import EXPERIMENT_ADAPTERS

    register_all_adapters()
    for entry in EXPERIMENT_ADAPTERS.entries():
        for name in entry.value.admissible_nulls:
            assert NULL_FAMILIES.get(name).admissible


def test_each_flagship_adapter_declares_which_nulls_its_support_can_carry():
    from src.adapters import register_all_adapters
    from src.core.experiment_adapter import adapter_for_domain

    register_all_adapters()
    assert "whole_cycle_clock_shift" in adapter_for_domain("reanalysis").admissible_nulls
    assert "within_group_clock_shift" in adapter_for_domain("tess_lightcurve").admissible_nulls
    assert "whole_cycle_clock_shift" not in adapter_for_domain(
        "tess_lightcurve").admissible_nulls
    assert adapter_for_domain("order_book").admissible_nulls == (
        "independent_native_clock_shift",)


@pytest.mark.parametrize("name,mode", [
    ("independent_native_clock_shift", "scale_shape_aligned"),
    ("scale_partner_reassignment", "calendar_aligned"),
])
def test_a_null_belonging_to_the_other_mode_is_refused_where_the_question_is_declared(name, mode):
    with pytest.raises(NullRefusal, match="a null declared for this comparison mode"):
        bind_null(name, {}, mode=mode)


def test_a_manifest_declaring_the_other_modes_null_is_refused():
    with pytest.raises(ValidationError, match="a null declared for this comparison mode"):
        _spec(nulls=[{"name": "wrong", "method": "scale_partner_reassignment",
                      "replications": 200}])


def test_a_null_parameter_has_no_framework_default():
    with pytest.raises(NullRefusal, match="would be a scientific choice nobody made"):
        bind_null("whole_cycle_clock_shift", {}, mode="calendar_aligned")


def test_an_unknown_null_parameter_is_refused_rather_than_ignored():
    with pytest.raises(NullRefusal, match="believes is in force and is not"):
        bind_null("independent_native_clock_shift", {"preserve_gaps": True},
                  mode="calendar_aligned")


def test_a_null_a_participating_adapter_does_not_admit_is_refused_by_name():
    from src.adapters import register_all_adapters
    from src.core.experiment_adapter import adapter_for_domain

    register_all_adapters()
    admissible = {adapter_for_domain(domain).adapter_id:
                  adapter_for_domain(domain).admissible_nulls
                  for domain in ("reanalysis", "order_book")}
    with pytest.raises(NullRefusal, match="order_book.bespoke_record admits"):
        bind_null("whole_cycle_clock_shift", {"cycle_seconds": 86400.0},
                  mode="calendar_aligned", admissible_by_adapter=admissible)


def test_an_unregistered_null_cannot_be_declared_in_a_manifest():
    with pytest.raises(ValidationError, match="not a registered family"):
        _spec(nulls=[{"name": "bespoke", "method": "shuffle_it", "replications": 200}])


def test_a_refused_null_cannot_be_declared_in_a_manifest():
    with pytest.raises(ValidationError, match="is refused"):
        _spec(nulls=[{"name": "shuffle", "method": "global_value_shuffle",
                      "replications": 200}])


def test_the_flagship_preflight_binds_its_null_against_every_participating_adapter():
    body = _body()
    body["nulls"] = [{"name": "seasonal", "method": "whole_cycle_clock_shift",
                      "replications": 200, "parameters": {"cycle_seconds": 86400.0}}]
    report = preflight_manifest(CrossDomainExperimentSpec.parse_obj(body))
    assert any("order_book.bespoke_record admits" in row["reason"]
               for row in report["refusals"])


def test_the_registry_lists_only_this_modes_admissible_families():
    assert nulls_for_mode("scale_shape_aligned") == ("scale_partner_reassignment",)
    assert "global_value_shuffle" not in nulls_for_mode("calendar_aligned")


# ------------------------------------------------------------ what the nulls actually do


def _trajectory(case="shared_calendar_event", domain="reanalysis"):
    from src.benchmarks.family_calibration import case_trajectories

    return case_trajectories(case, focus="planted")[domain]


def test_a_clock_shift_preserves_the_support_the_validity_and_the_marginal():
    original = _trajectory()
    surrogate = circular_clock_shift(original, 20260831)
    assert np.array_equal(original.support_start_seconds, surrogate.support_start_seconds)
    assert np.array_equal(original.support_end_seconds, surrogate.support_end_seconds)
    assert np.array_equal(original.valid_mask, surrogate.valid_mask)
    assert np.array_equal(np.sort(original.channels["standardized_level"]),
                          np.sort(surrogate.channels["standardized_level"]))
    assert not np.array_equal(original.channels["standardized_level"],
                              surrogate.channels["standardized_level"])


def test_a_surrogate_is_labelled_as_one_and_carries_its_own_digests():
    original = _trajectory()
    surrogate = circular_clock_shift(original, 20260831)
    assert surrogate.trajectory_id.startswith("null:")
    assert (surrogate.lineage["standardized_level"].output_sha256
            != original.lineage["standardized_level"].output_sha256)


def test_a_whole_cycle_shift_moves_every_value_by_a_whole_number_of_cycles():
    original = _trajectory()
    step = float(original.support_start_seconds[1] - original.support_start_seconds[0])
    cycle = step * 8
    surrogate = whole_cycle_clock_shift(original, 20260831, cycle_seconds=cycle)
    offset = surrogate.lineage["standardized_level"].parameters["offset"]
    assert offset % 8 == 0 and offset > 0
    assert np.array_equal(original.support_start_seconds, surrogate.support_start_seconds)


def test_a_cycle_the_record_does_not_span_twice_is_refused_rather_than_rounded():
    original = _trajectory()
    span = float(original.support_end_seconds[-1] - original.support_start_seconds[0])
    with pytest.raises(NullRefusal, match="a cycle this record spans more than once"):
        whole_cycle_clock_shift(original, 20260831, cycle_seconds=span)


def test_a_within_group_shift_never_moves_a_value_across_a_group_boundary():
    original = _trajectory()
    starts = np.asarray(original.support_start_seconds)
    group_seconds = 7.0 * 86400.0
    surrogate = within_group_clock_shift(original, 20260831, group_seconds=group_seconds)
    groups = np.floor((starts - starts[0]) / group_seconds).astype(int)
    for group in np.unique(groups):
        assert np.array_equal(
            np.sort(np.asarray(original.channels["standardized_level"])[groups == group]),
            np.sort(np.asarray(surrogate.channels["standardized_level"])[groups == group]))


def test_one_group_is_the_whole_record_and_is_refused_as_a_grouping():
    original = _trajectory()
    with pytest.raises(NullRefusal, match="more than one group"):
        within_group_clock_shift(original, 20260831, group_seconds=1e12)


def test_the_scale_shape_null_reassigns_partners_and_alters_no_record():
    pairs = [("a", 1.0), ("b", 2.0), ("c", 4.0), ("d", 8.0)]
    reassigned = reassign_scale_partners(pairs, 20260831)
    assert [left for left, _ in reassigned] == ["a", "b", "c", "d"]
    assert sorted(right for _, right in reassigned) == [1.0, 2.0, 4.0, 8.0]
    assert all(before[1] != after[1] for before, after in zip(pairs, reassigned))


def test_a_single_correspondence_has_no_alternative_partner_and_is_refused():
    with pytest.raises(NullRefusal, match="at least two declared correspondences"):
        reassign_scale_partners([("a", 1.0)], 20260831)


# ------------------------------------------------------- calibration at the family level


@pytest.fixture(scope="module")
def calibration():
    from src.benchmarks.family_calibration import calibrate_family

    return calibrate_family()


def test_the_planted_calendar_event_is_confirmed_at_the_frozen_family_level(calibration):
    case = next(row for row in calibration["cases"] if row["case"] == "shared_calendar_event")
    assert case["family_size"] == 6
    assert case["n_rejected_after_correction"] == 6
    assert case["met"] is True


@pytest.mark.parametrize("case_name", ["same_window_unrelated", "gap_alias",
                                       "inadmissible_precedence"])
def test_a_false_alignment_fixture_produces_no_rejection_after_correction(calibration, case_name):
    case = next(row for row in calibration["cases"] if row["case"] == case_name)
    assert case["n_rejected_after_correction"] == 0
    assert case["met"] is True


def test_every_calibration_case_meets_the_answer_frozen_with_the_fixture(calibration):
    assert calibration["all_met"] is True
    assert calibration["correction"] == "benjamini_yekutieli"
    assert calibration["replications"] == 999


def test_the_statistic_is_built_from_shared_support_and_not_from_row_counts():
    """The TG17.4 invariant, applied to the statistic the calibration corrects.

    Splitting every support interval in two writes the same record with twice the rows and
    exactly the same support. A comparison built from overlapping row pairs would report four
    times the evidence; one built from shared seconds reports the same number.
    """
    from src.benchmarks.family_calibration import support_weighted_correlation

    left = _trajectory(domain="reanalysis")
    right = _trajectory(domain="argo_float")
    before = support_weighted_correlation(left, right)

    starts = np.asarray(left.support_start_seconds)
    ends = np.asarray(left.support_end_seconds)
    middles = (starts + ends) / 2.0
    dense = replace(
        left,
        support_start_seconds=np.ravel(np.column_stack([starts, middles])),
        support_end_seconds=np.ravel(np.column_stack([middles, ends])),
        valid_mask=np.repeat(np.asarray(left.valid_mask), 2),
        channels={name: np.repeat(np.asarray(values), 2)
                  for name, values in left.channels.items()})
    assert len(dense.support_start_seconds) == 2 * len(starts)
    assert support_weighted_correlation(dense, right) == pytest.approx(before)


# ----------------------------------------------------------------------------------- api


def test_the_null_family_route_shows_which_domains_admit_each_family(client):
    body = client.get("/api/v1/experiment-composer/null-families").json()
    rows = {row["name"]: row for row in body["families"]}
    assert rows["global_value_shuffle"]["admissible"] is False
    assert rows["global_value_shuffle"]["admitted_by"] == []
    assert rows["independent_native_clock_shift"]["usable_across_all_registered_domains"]
    assert "order_book.bespoke_record" not in rows["whole_cycle_clock_shift"]["admitted_by"]


def test_the_family_route_prices_the_declared_search_in_human_terms(client):
    response = client.post("/api/v1/experiment-composer/manifests/family", json=_body())
    assert response.status_code == 200
    body = response.json()
    assert body["family_size"] == 288
    assert body["correction"]["correction_unit_members"] == 4
    assert body["axes"][0]["axis"] == "domain_set"
    assert body["claim_boundary"].startswith("This is an accounting statement")


def test_the_metadata_preflight_carries_the_priced_family(client):
    body = client.post("/api/v1/experiment-composer/manifests/preflight",
                       json=_body()).json()
    assert body["family"]["declared_members"] == 288
    assert body["family"]["specification_sha256"]
    assert body["family"]["precedence"]["domains_without_precedence_policy"] == [
        "order_book", "tess_lightcurve"]
