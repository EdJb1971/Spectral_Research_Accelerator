"""TG17.4: clock, support and coverage semantics.

The load-bearing test here is `test_row_density_alone_cannot_manufacture_support`. Every other
guarantee in this slice — half-open boundaries, the governing scale, declared kernels, the
mode vocabulary — exists to protect the same thing: that shared support is a fact about the
world the records describe, not about how many lines it took to write them down.
"""

import datetime as dt
import json

import pytest
from fastapi.testclient import TestClient

from src.adapters import register_all_adapters
from src.api.main import app
from src.benchmarks.alignment_fixtures import ALIGNMENT_FIXTURES, alignment_fixture, densify
from src.core.experiment_adapter import AdapterConformanceError, adapter_for_domain
from src.core.experiment_manifest import (CrossDomainExperimentSpec, flagship_recipe,
                                          preflight_manifest)
from src.core.structural_alignment import (ALIGNMENT_KERNELS, AlignmentRefusal,
                                           alignment_report, assert_mode_admits_relationship,
                                           bind_kernel, combined_family_multiplier,
                                           elapsed_seconds, intersect_intervals,
                                           merge_intervals, nominal_day_discrepancy,
                                           occupied_seconds, pairwise_overlap,
                                           subtract_intervals,
                                           scale_shape_correspondences, support_profile)
from src.core.structural_trajectory import StructuralScale


HOUR = 3600.0
WINDOW = (0.0, 7 * 86400.0)


@pytest.fixture(scope="module", autouse=True)
def _adapters():
    register_all_adapters()


def _flat(label, starts, ends, scale=HOUR, window=WINDOW, valid=None):
    return support_profile(label=label, domain=label, starts=starts, ends=ends, window=window,
                           native_scale_seconds=scale, valid=valid)


# ------------------------------------------------------------------ the invariant this slice is


@pytest.mark.parametrize("name", [name for name in ALIGNMENT_FIXTURES
                                  if len(alignment_fixture(name).profiles) == 2])
def test_row_density_alone_cannot_manufacture_support(name):
    """Sixty times the rows, the same support, and every derived number unchanged.

    A pipeline that counted overlapping row *pairs* would report a 3600-fold increase in shared
    evidence for a file that gained no information whatsoever. This is the single assertion the
    rest of the module exists to make true.
    """
    fixture = alignment_fixture(name)
    left, right = fixture.profiles.values()
    before = pairwise_overlap(left, right)
    after = pairwise_overlap(densify(left, 60), densify(right, 60))

    assert after.left_row_count == 60 * len(left.intervals)
    assert after.left_row_count != before.left_row_count
    assert after.overlap_seconds == pytest.approx(before.overlap_seconds)
    assert after.effective_sample_size == pytest.approx(before.effective_sample_size)
    assert after.governing_scale_seconds == before.governing_scale_seconds
    assert after.overlap_fraction_of_shorter == pytest.approx(before.overlap_fraction_of_shorter)


def test_the_report_shows_the_row_count_next_to_the_number_that_is_evidence():
    """Reported, and used by nothing. A reader can see what they would have reached for."""
    fixture = alignment_fixture("unequal_cadence")
    left, right = fixture.profiles.values()
    row = pairwise_overlap(left, right).describe()
    assert row["row_counts_not_used"] == {"left": 168, "right": 28}
    assert row["effective_sample_size"] == pytest.approx(28.0)


# ------------------------------------------------------------------------ half-open boundaries


def test_abutting_supports_do_not_overlap():
    fixture = alignment_fixture("abutting_boundary")
    before, after = fixture.profiles["before"], fixture.profiles["after"]
    overlap = pairwise_overlap(before, after)
    assert overlap.overlap_seconds == 0.0
    assert overlap.effective_sample_size == 0.0
    assert overlap.intervals == ()


def test_a_pair_sharing_no_support_refuses_by_name():
    fixture = alignment_fixture("abutting_boundary")
    report = alignment_report(list(fixture.profiles.values()), mode="calendar_aligned")
    assert report["status"] == "REFUSED"
    assert "abut" in report["pairs"][0]["refusals"][0]


def test_a_zero_width_support_is_refused_rather_than_counted():
    with pytest.raises(AlignmentRefusal) as error:
        _flat("instant", [0.0, HOUR], [0.0, 2 * HOUR])
    assert "start < end" in str(error.value)


def test_overlapping_supports_are_unioned_not_summed():
    """Two supports covering the same hour occupy the world once."""
    profile = _flat("bins", [0.0, 1800.0], [3600.0, 5400.0])
    assert profile.occupied_seconds == 5400.0
    assert profile.intervals == ((0.0, 5400.0),)


def test_intersection_of_two_unions_is_itself_half_open():
    left = merge_intervals([0.0, 100.0], [50.0, 150.0])
    assert left == ((0.0, 50.0), (100.0, 150.0))
    assert intersect_intervals(left, [(40.0, 110.0)]) == ((40.0, 50.0), (100.0, 110.0))
    # The gap between the two pieces is a real gap: an interval lying entirely inside it
    # intersects nothing, and an interval abutting the far edge intersects nothing either.
    assert intersect_intervals(left, [(50.0, 100.0)]) == ()
    assert occupied_seconds(intersect_intervals(left, [(150.0, 200.0)])) == 0.0


# ------------------------------------------------------------------------- the governing scale


def test_the_coarser_scale_governs_effective_sample_size():
    """A record cannot borrow resolution from the record it is compared with."""
    fixture = alignment_fixture("unequal_cadence")
    hourly, six_hourly = fixture.profiles["hourly"], fixture.profiles["six_hourly"]
    overlap = pairwise_overlap(hourly, six_hourly)
    assert overlap.governing_scale_seconds == 6 * HOUR
    assert overlap.effective_sample_size == pytest.approx(28.0)
    assert overlap.effective_sample_size != pytest.approx(168.0)


def test_non_stationary_support_labels_its_effective_sample_size_an_upper_bound():
    fixture = alignment_fixture("non_stationary_support")
    drifting, steady = fixture.profiles["drifting"], fixture.profiles["steady"]
    assert drifting.support_is_stationary is False
    assert steady.support_is_stationary is True
    row = pairwise_overlap(drifting, steady).describe()
    assert "upper bound" in row["effective_sample_size_basis"]


# ------------------------------------------------------------------------------- the calendar


def test_a_daylight_saving_day_is_not_86400_seconds():
    """The window a researcher declared, not the window a nominal day would assume."""
    fixture = alignment_fixture("daylight_saving_day")
    clock = fixture.expected
    assert clock["elapsed_seconds"] == 82800.0
    assert clock["discrepancy_seconds"] == -3600.0
    profile = fixture.profiles["hourly"]
    assert profile.covered_fraction == pytest.approx(1.0)
    assert profile.window_seconds == 82800.0


def test_a_nominal_day_denominator_would_understate_complete_coverage():
    fixture = alignment_fixture("daylight_saving_day")
    profile = fixture.profiles["hourly"]
    assert profile.occupied_seconds / 86400.0 == pytest.approx(0.9583, abs=1e-4)
    assert profile.covered_fraction == pytest.approx(1.0)


def test_a_naive_local_timestamp_is_not_a_point_in_time():
    with pytest.raises(AlignmentRefusal) as error:
        elapsed_seconds(dt.datetime(2026, 3, 29), dt.datetime(2026, 3, 30))
    assert "UTC offset" in str(error.value)


def test_nominal_day_discrepancy_reports_a_uniform_clock_as_uniform():
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    facts = nominal_day_discrepancy(start, start + dt.timedelta(days=1))
    assert facts["clock_is_uniform"] is True
    assert facts["discrepancy_seconds"] == 0.0


# ------------------------------------------------------------------------------ sparse and gapped


def test_a_sparse_profile_reports_the_coverage_it_has():
    fixture = alignment_fixture("sparse_profile")
    argo, era5 = fixture.profiles["argo"], fixture.profiles["era5"]
    assert argo.covered_fraction == pytest.approx(0.01667, abs=1e-4)
    overlap = pairwise_overlap(argo, era5)
    assert overlap.overlap_seconds == pytest.approx(9 * 4 * HOUR)
    assert overlap.effective_sample_size == pytest.approx(9.0)


def test_an_observation_gap_survives_a_continuous_partner():
    """A continuous record does not paper over the other record's downlink gap."""
    fixture = alignment_fixture("interrupted_light_curve")
    tess, era5 = fixture.profiles["tess"], fixture.profiles["era5"]
    assert len(tess.gaps) == 1
    assert tess.largest_gap_seconds == 2 * 86400.0
    overlap = pairwise_overlap(tess, era5)
    assert overlap.overlap_seconds == pytest.approx(25 * 86400.0)
    assert len(overlap.intervals) == 2


@pytest.mark.parametrize("name", ALIGNMENT_FIXTURES)
def test_every_adversarial_fixture_matches_its_known_answer(name):
    """Aligned as declared, or refused by name. Never quietly repaired."""
    fixture = alignment_fixture(name)
    expected = fixture.expected
    profiles = list(fixture.profiles.values())
    if len(profiles) == 2:
        overlap = pairwise_overlap(profiles[0], profiles[1])
        if "overlap_seconds" in expected:
            assert overlap.overlap_seconds == pytest.approx(expected["overlap_seconds"])
        if "governing_scale_seconds" in expected:
            assert overlap.governing_scale_seconds == expected["governing_scale_seconds"]
        if "effective_sample_size" in expected:
            assert overlap.effective_sample_size == pytest.approx(
                expected["effective_sample_size"])
    if "covered_fraction" in expected:
        assert profiles[0].covered_fraction == pytest.approx(expected["covered_fraction"])
    if "support_is_stationary" in expected:
        assert profiles[0].support_is_stationary is expected["support_is_stationary"]
    assert expected["answer"]


# ------------------------------------------------------------------------------ declared kernels


def test_nothing_bins_or_widens_unless_the_manifest_names_it():
    """The only kernel that runs unnamed transforms nothing."""
    assert flagship_recipe().alignment.kernel == "exact_support_overlap"
    assert ALIGNMENT_KERNELS.get("exact_support_overlap").manufactures_simultaneity is False
    fixture = alignment_fixture("abutting_boundary")
    left, right = fixture.profiles.values()
    assert pairwise_overlap(left, right).overlap_seconds == 0.0


def test_an_unregistered_kernel_is_refused_rather_than_approximated():
    with pytest.raises(AlignmentRefusal) as error:
        bind_kernel("nearest_neighbour_snap", {})
    assert "exact_support_overlap" in str(error.value)
    assert "nobody can reproduce" in str(error.value)


def test_a_kernel_parameter_has_no_framework_default():
    """A tolerance the framework picked is a scientific choice nobody made."""
    with pytest.raises(AlignmentRefusal) as error:
        bind_kernel("symmetric_tolerance", {})
    assert "tolerance_seconds" in str(error.value)


def test_an_unknown_kernel_parameter_is_refused_rather_than_ignored():
    with pytest.raises(AlignmentRefusal) as error:
        bind_kernel("symmetric_tolerance", {"tolerance_seconds": 60.0, "tolerence": 60.0})
    assert "tolerence" in str(error.value)
    assert "believes is in force" in str(error.value)


def test_a_kernel_reports_the_seconds_of_overlap_it_created():
    """Manufactured simultaneity is a number in the report, not a footnote."""
    fixture = alignment_fixture("abutting_boundary")
    left, right = fixture.profiles.values()
    kernel = bind_kernel("symmetric_tolerance", {"tolerance_seconds": 600.0})
    overlap = pairwise_overlap(left, right, kernel=kernel)
    assert overlap.exact_overlap_seconds == 0.0
    assert overlap.overlap_seconds == pytest.approx(600.0)
    assert overlap.manufactured_overlap_seconds == pytest.approx(600.0)


def test_a_declared_tolerance_is_the_slack_between_two_records_not_within_each():
    """Applying the whole tolerance to both profiles would silently double it."""
    kernel = bind_kernel("symmetric_tolerance", {"tolerance_seconds": 600.0})
    left = _flat("left", [0.0], [HOUR])
    right = _flat("right", [HOUR + 599.0], [2 * HOUR])
    assert pairwise_overlap(left, right, kernel=kernel).overlap_seconds > 0.0
    far = _flat("far", [HOUR + 601.0], [2 * HOUR])
    assert pairwise_overlap(left, far, kernel=kernel).overlap_seconds == 0.0


def test_grid_aggregation_drops_a_bin_the_record_barely_touches():
    """Promoting a barely-touched bin is how a coarse grid manufactures agreement."""
    kernel = bind_kernel("common_grid_aggregate",
                         {"bin_seconds": HOUR, "minimum_bin_fraction": 0.5})
    sparse = _flat("sparse", [0.0, 2 * HOUR], [300.0, 2 * HOUR + 3000.0])
    binned = kernel.apply(sparse)
    assert binned.intervals == ((2 * HOUR, 3 * HOUR),)
    assert binned.manufactured_seconds > 0.0


def test_interval_subtraction_reports_the_new_part_rather_than_a_net_total():
    """A kernel that drops one second of real overlap and adds another manufactured a second."""
    assert subtract_intervals([(0.0, 100.0)], [(20.0, 40.0)]) == ((0.0, 20.0), (40.0, 100.0))
    assert occupied_seconds(subtract_intervals([(10.0, 20.0)], [(0.0, 10.0)])) == 10.0
    assert subtract_intervals([(0.0, 10.0)], [(0.0, 10.0)]) == ()


def test_manufactured_overlap_is_measured_and_not_netted():
    left = _flat("left", [0.0, 7200.0], [1800.0, 9000.0])
    right = _flat("right", [1500.0], [7500.0])
    kernel = bind_kernel("symmetric_tolerance", {"tolerance_seconds": 1200.0})
    exact = pairwise_overlap(left, right)
    widened = pairwise_overlap(left, right, kernel=kernel)
    assert widened.manufactured_overlap_seconds > 0.0
    assert widened.manufactured_overlap_seconds == pytest.approx(
        occupied_seconds(subtract_intervals(widened.intervals, exact.intervals)))
    assert widened.describe()["lost_overlap_seconds"] == 0.0


def test_a_bound_kernel_carries_the_digest_of_what_was_frozen():
    one = bind_kernel("symmetric_tolerance", {"tolerance_seconds": 600.0})
    same = bind_kernel("symmetric_tolerance", {"tolerance_seconds": 600.0})
    other = bind_kernel("symmetric_tolerance", {"tolerance_seconds": 601.0})
    assert one.freeze_sha256 == same.freeze_sha256
    assert one.freeze_sha256 != other.freeze_sha256


# ---------------------------------------------------------- a kernel is an adapter operation


def test_a_value_inventing_kernel_is_refused_over_an_irregular_clock():
    """Interpolating across an irregular clock manufactures the simultaneity under test."""
    order_book = adapter_for_domain("order_book").declaration
    with pytest.raises(AlignmentRefusal) as error:
        bind_kernel("carry_forward", {"maximum_carry_seconds": 600.0},
                    declarations=[order_book])
    assert "irregular_sampling" in str(error.value)
    assert "invent values" in str(error.value)


def test_a_kernel_a_participating_adapter_does_not_admit_is_refused_by_name():
    """The bespoke family admits nothing but the kernel that transforms nothing."""
    with pytest.raises(AlignmentRefusal) as error:
        bind_kernel("common_grid_aggregate",
                    {"bin_seconds": HOUR, "minimum_bin_fraction": 1.0},
                    admissible_by_adapter={"order_book.bespoke_record":
                                           ("exact_support_overlap",)})
    assert "order_book.bespoke_record" in str(error.value)
    assert "adapter operation" in str(error.value)


def test_each_flagship_adapter_declares_which_kernels_its_support_tolerates():
    admitted = {domain: adapter_for_domain(domain).admissible_kernels
                for domain in ("reanalysis", "argo_float", "tess_lightcurve", "order_book")}
    assert admitted["order_book"] == ("exact_support_overlap",)
    assert "common_grid_aggregate" in admitted["tess_lightcurve"]
    assert "symmetric_tolerance" in admitted["argo_float"]
    assert "carry_forward" not in set().union(*(set(item) for item in admitted.values()))


def test_an_adapter_may_not_refuse_the_kernel_that_transforms_nothing():
    adapter = adapter_for_domain("reanalysis")
    with pytest.raises(AdapterConformanceError) as error:
        type(adapter)(adapter_id="x.y", adapter_version="v1", declaration=adapter.declaration,
                      controls=adapter.controls, plan_acquisition=adapter.plan_acquisition,
                      structural_declaration=adapter.structural_declaration,
                      translate=adapter.translate,
                      admissible_kernels=("symmetric_tolerance",))
    assert "exact_support_overlap" in str(error.value)


def test_an_adapter_cannot_admit_a_kernel_that_does_not_exist():
    adapter = adapter_for_domain("reanalysis")
    with pytest.raises(AdapterConformanceError) as error:
        type(adapter)(adapter_id="x.y", adapter_version="v1", declaration=adapter.declaration,
                      controls=adapter.controls, plan_acquisition=adapter.plan_acquisition,
                      structural_declaration=adapter.structural_declaration,
                      translate=adapter.translate,
                      admissible_kernels=("exact_support_overlap", "magic_snap"))
    assert "magic_snap" in str(error.value)


# --------------------------------------------------------------------- the two mode vocabularies


@pytest.mark.parametrize("relationship", ["co_occurrence", "precedence", "causality"])
def test_scale_shape_mode_cannot_emit_calendar_language(relationship):
    with pytest.raises(AlignmentRefusal) as error:
        assert_mode_admits_relationship("scale_shape_aligned", relationship)
    assert "carries no clock" in str(error.value)


def test_calendar_mode_cannot_silently_search_normalized_scale_ratios():
    with pytest.raises(AlignmentRefusal) as error:
        assert_mode_admits_relationship("calendar_aligned", "scale_ratio_similarity")
    assert "normalized scale-ratio search" in str(error.value)


def test_a_manifest_whose_relationship_belongs_to_the_other_mode_is_refused():
    """Refused where the search is declared, not where the result is worded."""
    raw = json.loads(flagship_recipe().json())
    raw["family"]["relationships"] = ["scale_ratio_similarity"]
    with pytest.raises(ValueError) as error:
        CrossDomainExperimentSpec.parse_obj(raw)
    assert "scale_ratio_similarity" in str(error.value)


def test_declaring_both_modes_pays_for_the_combined_family():
    one = combined_family_multiplier(["calendar_aligned"])
    both = combined_family_multiplier(["calendar_aligned", "scale_shape_aligned"])
    assert one["multiplier"] == 1
    assert both["multiplier"] == 2
    assert "union of what was searched" in both["why"]


def test_a_scale_shape_match_retains_both_native_durations():
    """A normalized match is never reported as an unqualified similarity."""
    left = (StructuralScale(coordinate=0.25, native_value=1.8, native_units="hours",
                            mapping="native duration / record duration"),)
    right = (StructuralScale(coordinate=0.25, native_value=46.0, native_units="days",
                             mapping="native duration / record duration"),)
    matches = scale_shape_correspondences("tess", left, "argo", right)
    assert len(matches) == 1
    row = matches[0].describe()
    assert row["left_native_value"] == 1.8 and row["left_native_units"] == "hours"
    assert row["right_native_value"] == 46.0 and row["right_native_units"] == "days"
    assert row["emits_simultaneity"] is False and row["emits_precedence"] is False


# ------------------------------------------------------------------------------------ preflight


def test_preflight_reports_bounded_overlap_where_metadata_cannot_establish_it():
    """A number that would later turn out to be a guess is not offered as a number."""
    report = preflight_manifest(flagship_recipe())
    window = report["alignment"]["windows"][0]
    statuses = {row["status"] for row in window["pairs"]}
    assert statuses == {"BOUNDED_BY_WINDOW"}
    reasons = " ".join(row["reason"] for row in window["pairs"])
    assert "established only after acquisition" in reasons
    assert report["alignment"]["measurement_values_opened"] is False


def test_preflight_states_the_true_elapsed_seconds_of_every_window():
    report = preflight_manifest(flagship_recipe())
    week = report["alignment"]["windows"][0]
    assert week["window_seconds"] == 604800.0
    assert week["clock"]["clock_is_uniform"] is True
    assert "23 or 25 hours" in week["clock_note"]


def test_preflight_refuses_a_kernel_a_participating_adapter_does_not_admit():
    """Every participating domain must admit the kernel, not merely one of them."""
    raw = json.loads(flagship_recipe().json())
    raw["alignment"] = {"kernel": "common_grid_aggregate",
                        "parameters": {"bin_seconds": 3600.0, "minimum_bin_fraction": 1.0}}
    report = preflight_manifest(CrossDomainExperimentSpec.parse_obj(raw))
    assert report["status"] == "REFUSED"
    assert report["alignment"]["kernel"] is None
    assert "argo_float.standardized-level" in report["alignment"]["kernel_refused"]
    assert report["alignment"]["windows"] == []


def test_a_pair_below_the_declared_effective_sample_floor_is_refused_not_reported():
    """Enough shared seconds, and still not enough evidence at the governing scale."""
    fixture = alignment_fixture("sparse_profile")
    profiles = list(fixture.profiles.values())
    report = alignment_report(profiles, mode="calendar_aligned",
                              minimum_effective_samples=30.0)
    assert report["status"] == "REFUSED"
    row = report["pairs"][0]
    assert row["status"] == "REFUSED"
    assert row["overlap_seconds"] > 0.0
    assert "effective sample size is 9.000" in row["refusals"][0]


def test_preflight_carries_the_declared_floors_into_what_it_reports():
    raw = json.loads(flagship_recipe().json())
    raw["alignment"] = {"minimum_overlap_seconds": 86400.0, "minimum_effective_samples": 24.0}
    report = preflight_manifest(CrossDomainExperimentSpec.parse_obj(raw))
    assert report["alignment"]["minimum_overlap_seconds"] == 86400.0
    assert report["alignment"]["minimum_effective_samples"] == 24.0


def test_the_alignment_policy_travels_inside_the_manifest_digest():
    from src.core.experiment_manifest import manifest_sha256

    raw = json.loads(flagship_recipe().json())
    baseline = manifest_sha256(CrossDomainExperimentSpec.parse_obj(raw))
    raw["alignment"] = {"kernel": "symmetric_tolerance", "parameters": {"tolerance_seconds": 60.0}}
    changed = manifest_sha256(CrossDomainExperimentSpec.parse_obj(raw))
    assert baseline != changed


# ------------------------------------------------------------------------------------ the API


def test_the_kernel_route_shows_which_adapters_admit_each_kernel():
    client = TestClient(app)
    payload = client.get("/api/v1/experiment-composer/alignment-kernels").json()
    rows = {row["name"]: row for row in payload["kernels"]}
    assert rows["exact_support_overlap"]["usable_across_all_registered_domains"] is True
    assert rows["carry_forward"]["admitted_by"] == []
    assert payload["default"] == "exact_support_overlap"


def test_the_alignment_route_measures_support_and_says_which_records_it_opened():
    client = TestClient(app)
    spec = json.loads(flagship_recipe().json())
    payload = client.post("/api/v1/experiment-composer/manifests/alignment", json=spec).json()
    assert payload["record_binding"] == "benchmark_known_answer"
    assert payload["row_indices_were_not_compared"] is True
    coverage = {row["label"]: row for row in payload["coverage"]}
    assert coverage["order_book"]["gap_count"] > 0
    assert coverage["reanalysis"]["covered_fraction"] == pytest.approx(1.0)
    assert len(payload["pairs"]) == 6
    assert all(row["governing_scale_seconds"] > 0 for row in payload["pairs"])


def test_the_alignment_route_refuses_a_kernel_no_adapter_admits():
    client = TestClient(app)
    spec = json.loads(flagship_recipe().json())
    spec["alignment"] = {"kernel": "carry_forward",
                         "parameters": {"maximum_carry_seconds": 600.0}}
    response = client.post("/api/v1/experiment-composer/manifests/alignment", json=spec)
    assert response.status_code == 422
    assert "irregular_sampling" in response.json()["detail"]
