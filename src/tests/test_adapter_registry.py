"""TG17.3: the adapter registry, its conformance kit, and the extension seam's acceptance test.

The load-bearing test here is `test_synthetic_fifth_adapter_...`. TG17.3's acceptance requires
that a fifth domain reach the domain selector, the preflight and the coverage matrix **without
editing the orchestrator, the generic API routes or the UI source**. The fifth adapter below is
therefore defined in this module — not in `src/`, not in `extensions/`, and not in any file the
application imports — and it carries genuinely different structural mathematics: a monotone rank
channel, not the shared standardized level. An adapter that merely re-parameterised the existing
one would test the factory's arguments and prove nothing about the seam.
"""

import datetime as dt

import numpy as np
import pytest

from src.core.adapter_conformance import (FAIL, NOT_PROBED, PASS, ConformanceCase,
                                          assert_conformance, run_conformance)
from src.core.builtin_glossaries import ORDER_BOOK_PHRASES
from src.core.builtin_domains import register_builtin_domains
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.experiment_adapter import (EXPERIMENT_ADAPTERS, AcquisitionPlan,
                                         AdapterConformanceError, ControlField, ControlSchema,
                                         DomainExperimentAdapter, adapter_for_domain,
                                         plan_windows, register_experiment_adapter,
                                         registered_domains)
from src.core.onboarding import onboard_domain
from src.core.registry import restore, snapshot
from src.core.structural_trajectory import (ChannelLineage, NativeStructuralRecord,
                                            StructuralAdapterDeclaration,
                                            StructuralChannelDefinition, StructuralScale,
                                            StructuralTrajectory, mine_structural_peak,
                                            register_lineage_reconstructor)


class _Window:
    def __init__(self, name, start, end):
        self.name, self.start_utc, self.end_utc = name, start, end


WEEK = _Window("week", dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
               dt.datetime(2026, 1, 8, tzinfo=dt.timezone.utc))


@pytest.fixture(autouse=True)
def _isolated_registries():
    """Registries are process-global, so a test that registers must put them back."""
    register_builtin_domains()
    from src.core.domain import DOMAIN_DECLARATIONS
    from src.core.onboarding import DOMAIN_ONBOARDINGS
    from src.core.translation import DOMAIN_GLOSSARIES

    saved = [(registry, snapshot(registry)) for registry in
             (EXPERIMENT_ADAPTERS, DOMAIN_DECLARATIONS, DOMAIN_ONBOARDINGS, DOMAIN_GLOSSARIES)]
    try:
        yield
    finally:
        for registry, state in saved:
            restore(registry, state)


def _native(domain, semantics, units, values=None):
    values = np.asarray(values if values is not None else [3.0, 1.0, 4.0, 1.5, 9.0, 2.0])
    return NativeStructuralRecord(
        domain=domain, source_id="test:%s" % domain, variable="probe", semantics=semantics,
        units=units, sample_times_seconds=np.arange(len(values), dtype=float) * 60.0,
        values=values, valid_mask=np.array([True, True, False, True, True, True]),
        native_scale_seconds=60.0, native_locator="test://%s" % domain,
        assumption_violations=("irregular_sampling",))


# ------------------------------------------------------------- the synthetic fifth adapter

FIFTH_DOMAIN = "synthetic_rank_sensor"
RANK_OPERATION = "monotone_rank(native_value) / (count - 1)"

RANK_CHANNEL = StructuralChannelDefinition(
    name="monotone_rank", semantics="within-record monotone rank position", units="dimensionless",
    formula=RANK_OPERATION, benchmark_id="test.monotone-rank-v1")


def _rank(values, _parameters=None):
    order = np.argsort(np.argsort(np.asarray(values, dtype=float), kind="stable"), kind="stable")
    return order.astype(float) / float(max(len(order) - 1, 1))


register_lineage_reconstructor(RANK_OPERATION, lambda values, parameters: _rank(values))


def _fifth_declaration():
    return DomainDeclaration(
        name=FIFTH_DOMAIN,
        description="A synthetic sensor archive used only to exercise the extension seam.",
        axes=(AxisSpec(name="reading_time", role="time", units="s"),
              AxisSpec(name="sensor", role="category", ordered=False)),
        licence="Synthetic test fixture; no redistribution question arises.",
        violations=("no_physical_metric", "no_propagation_speed", "irregular_sampling",
                    "unordered_channels"),
        lag_policy="none")


def _fifth_structural(_parameters):
    return StructuralAdapterDeclaration(
        adapter_id="%s.monotone-rank" % FIFTH_DOMAIN, adapter_version="test-v1",
        domain=FIFTH_DOMAIN, accepted_semantics="arbitrary sensor reading",
        accepted_units="counts", required_axes=("native_time",),
        required_roles=("observation", "validity"),
        invariances=("native_value_translation", "native_value_positive_scaling",
                     "any_strictly_increasing_transform"),
        consumed_information=("native_value",), output_clock="identity_native_clock",
        output_support="[native_time_i, native_time_i + declared native scale)",
        missing_data_behavior="preserve validity exactly",
        legitimate_null_family="independent_native_clock_shift",
        leakage_risks=("rank_ties_hide_magnitude",),
        refused_operations=("causality", "precedence", "raw_magnitude_comparison"),
        channels={RANK_CHANNEL.name: RANK_CHANNEL})


def _fifth_translate(record, declaration, _parameters):
    from src.core.structural_trajectory import _array_digest, _digest

    values = _rank(record.values)
    config_sha = _digest({})
    return StructuralTrajectory(
        schema_id="structural-trajectory/v1",
        trajectory_id="structural:%s" % _digest({"native": record.content_sha256,
                                                 "adapter": declaration.definition_sha256}),
        domain=record.domain, source_id=record.source_id, variable=record.variable,
        native_semantics=record.semantics, native_units=record.units,
        support_start_seconds=record.sample_times_seconds,
        support_end_seconds=record.sample_times_seconds + record.native_scale_seconds,
        valid_mask=record.valid_mask,
        structural_scales=(StructuralScale(1.0, record.native_scale_seconds, "seconds",
                                           "coordinate * native_scale_seconds"),),
        channels={RANK_CHANNEL.name: values}, channel_definitions=declaration.channels,
        adapter_id=declaration.adapter_id, adapter_version=declaration.adapter_version,
        adapter_definition_sha256=declaration.definition_sha256,
        adapter_config_sha256=config_sha, native_record_sha256=record.content_sha256,
        native_record_locator=record.native_locator, native_record_retained=True,
        assumption_violations=tuple(record.assumption_violations),
        lineage={RANK_CHANNEL.name: ChannelLineage(
            channel=RANK_CHANNEL.name, source_variable=record.variable,
            source_indices=np.arange(len(record.values)), operation=RANK_OPERATION,
            parameters={"count": float(len(record.values))},
            output_sha256=_array_digest(values))})


def _install_fifth_adapter():
    """Everything a third party does, and nothing else: declare, onboard, register."""
    declaration = _fifth_declaration()
    onboard_domain(declaration, ORDER_BOOK_PHRASES,
                   geometry=None, glossary_description="Synthetic sensor wording.",
                   onboarded_by="src/tests/test_adapter_registry.py", replace=True)
    from src.core.structural_nulls import circular_clock_shift

    adapter = DomainExperimentAdapter(
        adapter_id="%s.monotone-rank" % FIFTH_DOMAIN, adapter_version="test-v1",
        declaration=declaration,
        controls=ControlSchema((ControlField(
            name="sensor_id", label="Sensor", kind="text", default="probe-1",
            help="Which sensor's readings this observation uses."),)),
        plan_acquisition=lambda parameters, identity=None: AcquisitionPlan(
            source_id="sensor_query:synthetic", source_version="test-v1",
            support_kind="sparse_point_support", access="local_file_binding",
            access_means="A synthetic fixture; no archive is contacted.",
            coverage_exact=False, estimated_bytes_per_day=1_000),
        structural_declaration=_fifth_structural, translate=_fifth_translate,
        build_null=circular_clock_shift)
    return register_experiment_adapter(adapter, replace=True)


def test_synthetic_fifth_adapter_reaches_the_registry_and_conforms_without_framework_edits():
    """The TG17.3 acceptance test: a fifth domain, no orchestrator, route or UI change."""
    before = registered_domains()
    assert FIFTH_DOMAIN not in before

    adapter = _install_fifth_adapter()

    assert FIFTH_DOMAIN in registered_domains()
    assert adapter_for_domain(FIFTH_DOMAIN) is adapter
    # Its controls reach any renderer generically: nothing asks which domain this is.
    fields = {item["name"]: item for item in adapter.describe()["controls"]["fields"]}
    assert fields["sensor_id"]["kind"] == "text"

    record = _native(FIFTH_DOMAIN, "arbitrary sensor reading", "counts")
    report = assert_conformance(adapter, ConformanceCase(
        parameters={}, windows=[WEEK], maximum_planned_bytes=4 * 1024**3, native_record=record))
    assert report.conformant

    # Different mathematics, same domain-blind mining seam.
    trajectory = adapter.translate(record, adapter.structural_declaration({}), {})
    peak = mine_structural_peak(trajectory, channel="monotone_rank")
    assert peak["units"] == "dimensionless"


def test_an_unprobed_invariance_is_reported_as_unverified_rather_than_passing():
    """`any_strictly_increasing_transform` has no executable probe, so it may not pass."""
    adapter = _install_fifth_adapter()
    report = run_conformance(adapter, ConformanceCase(
        parameters={}, windows=[WEEK], maximum_planned_bytes=4 * 1024**3,
        native_record=_native(FIFTH_DOMAIN, "arbitrary sensor reading", "counts")))
    statuses = {item.name: item.status for item in report.checks}
    assert statuses["declared_invariances:any_strictly_increasing_transform"] == NOT_PROBED
    assert statuses["declared_invariances:native_value_positive_scaling"] == PASS
    assert report.conformant, "an unverified claim is visible, not a failure"


def test_conformance_refuses_an_adapter_whose_declared_invariance_it_does_not_have():
    """A declaration is prose until something executes it."""
    adapter = _install_fifth_adapter()

    def _not_invariant(record, declaration, parameters):
        trajectory = _fifth_translate(record, declaration, parameters)
        from dataclasses import replace as dc_replace
        return dc_replace(trajectory, channels={RANK_CHANNEL.name: np.asarray(record.values)})

    broken = DomainExperimentAdapter(
        adapter_id="broken.invariance", adapter_version="test-v1",
        declaration=adapter.declaration, controls=adapter.controls,
        plan_acquisition=adapter.plan_acquisition,
        structural_declaration=adapter.structural_declaration, translate=_not_invariant)
    report = run_conformance(broken, ConformanceCase(
        parameters={}, windows=[WEEK], maximum_planned_bytes=4 * 1024**3,
        native_record=_native(FIFTH_DOMAIN, "arbitrary sensor reading", "counts")))
    failed = {item.name for item in report.failures}
    assert "declared_invariances:native_value_positive_scaling" in failed
    assert not report.conformant


def test_a_lineage_operation_with_no_reconstructor_fails_conformance():
    """A canonical value nobody can rebuild is not provenanced by carrying a digest."""
    from src.core.structural_trajectory import assert_structural_conformance, \
        StructuralConformanceError
    from dataclasses import replace as dc_replace

    adapter = _install_fifth_adapter()
    record = _native(FIFTH_DOMAIN, "arbitrary sensor reading", "counts")
    declaration = adapter.structural_declaration({})
    trajectory = _fifth_translate(record, declaration, {})
    lineage = dict(trajectory.lineage)
    lineage[RANK_CHANNEL.name] = dc_replace(lineage[RANK_CHANNEL.name],
                                            operation="trust_me(native_value)")
    opaque = dc_replace(trajectory, lineage=lineage)
    with pytest.raises(StructuralConformanceError, match="no registered reconstructor"):
        assert_structural_conformance(opaque, record, declaration, config={})


def test_an_adapter_over_an_unonboarded_domain_is_refused_at_construction():
    declaration = DomainDeclaration(
        name="never_onboarded", description="x",
        axes=(AxisSpec(name="t", role="time", units="s"),),
        licence="none", violations=("no_physical_metric", "no_propagation_speed"),
        lag_policy="none")
    with pytest.raises(AdapterConformanceError, match="onboard_domain"):
        DomainExperimentAdapter(
            adapter_id="never.onboarded", adapter_version="v1", declaration=declaration,
            controls=ControlSchema(()), plan_acquisition=lambda p, i=None: None,
            structural_declaration=lambda p: None, translate=lambda r, d, p: None)


# ------------------------------------------------------------------- the registered quartet


def _registered_adapters():
    from src.adapters import register_all_adapters

    register_all_adapters()
    return [EXPERIMENT_ADAPTERS.get(item) for item in
            ("reanalysis.standardized-level", "argo_float.standardized-level",
             "tess_lightcurve.standardized-level", "order_book.bespoke_record")]


@pytest.mark.parametrize("index", range(4))
def test_every_flagship_adapter_passes_the_conformance_kit(index):
    from src.benchmarks.structural_trajectory import known_answer_native

    adapter = _registered_adapters()[index]
    report = assert_conformance(adapter, ConformanceCase(
        parameters={}, windows=[WEEK], maximum_planned_bytes=4 * 1024**3,
        native_record=known_answer_native(adapter.domain)))
    assert report.conformant
    assert all(item.status != FAIL for item in report.checks)


def test_two_of_the_four_adapters_register_from_outside_src():
    """The seam is exercised by this programme's own adapters, not demonstrated separately."""
    _registered_adapters()
    argo = EXPERIMENT_ADAPTERS.get("argo_float.standardized-level")
    tess = EXPERIMENT_ADAPTERS.get("tess_lightcurve.standardized-level")
    # The domain-specific half of each adapter — its acquisition plan — is defined outside
    # `src`. The shared half comes from the factory, which is the point: glue trends to zero.
    assert argo.plan_acquisition.__module__ == "extensions.argo_float"
    assert tess.plan_acquisition.__module__ == "extensions.tess_lightcurve"
    reanalysis = EXPERIMENT_ADAPTERS.get("reanalysis.standardized-level")
    assert reanalysis.plan_acquisition.__module__ == "src.adapters.reanalysis"


def test_sector_and_sparse_support_can_never_report_exact_coverage():
    adapters = _registered_adapters()
    for adapter in adapters:
        plan = adapter.plan_acquisition(adapter.resolve({}), {})
        if plan.support_kind in ("sparse_point_support",
                                 "intersecting_observational_sectors",
                                 "irregular_local_record"):
            assert plan.coverage_exact is False
            assert all(row["coverage_exact"] is False for row in plan_windows(plan, [WEEK]))


def test_a_plan_claiming_exactness_on_intersecting_support_fails_coverage_honesty():
    adapter = _registered_adapters()[0]
    dishonest = DomainExperimentAdapter(
        adapter_id="dishonest.sectors", adapter_version="v1",
        declaration=adapter.declaration, controls=adapter.controls,
        plan_acquisition=lambda parameters, identity=None: AcquisitionPlan(
            source_id="x", source_version="v", support_kind="intersecting_observational_sectors",
            access="public_network", access_means="test", coverage_exact=True,
            native_cadence_seconds=120.0),
        structural_declaration=adapter.structural_declaration, translate=adapter.translate)
    from src.benchmarks.structural_trajectory import known_answer_native

    report = run_conformance(dishonest, ConformanceCase(
        parameters={}, windows=[WEEK], maximum_planned_bytes=4 * 1024**3,
        native_record=known_answer_native(adapter.domain)))
    assert "coverage_honesty" in {item.name for item in report.failures}


def test_a_plan_over_its_byte_cap_refuses_before_acquisition():
    from src.benchmarks.structural_trajectory import known_answer_native

    adapter = _registered_adapters()[0]
    report = run_conformance(adapter, ConformanceCase(
        parameters={}, windows=[WEEK], maximum_planned_bytes=1_000,
        native_record=known_answer_native(adapter.domain)))
    failure = next(item for item in report.failures if item.name == "bounded_resource_plan")
    assert "above its" in failure.detail


def test_unknown_control_parameters_are_refused_rather_than_ignored():
    from src.core.errors import InvalidParameterError

    adapter = _registered_adapters()[0]
    with pytest.raises(InvalidParameterError, match="only controls this domain declares"):
        adapter.resolve({"level_hpa": 850, "not_a_control": 1})


# ----------------------------------------------------------------- the bespoke record family


def test_a_bespoke_domain_is_added_by_declaration_alone_with_no_code():
    """Order book is one instance of this family, not a finance adapter."""
    from src.adapters.bespoke_record import register_bespoke_domain

    declaration = DomainDeclaration(
        name="clinic_appointments",
        description="Appointment starts logged by a clinic's booking system.",
        axes=(AxisSpec(name="booked_at", role="time", units="s"),
              AxisSpec(name="room", role="category", ordered=False)),
        licence="Held locally by the researcher; not redistributable.",
        violations=("no_physical_metric", "no_propagation_speed", "unordered_channels",
                    "irregular_sampling"),
        lag_policy="none")
    onboard_domain(declaration, ORDER_BOOK_PHRASES,
                   geometry=None, glossary_description="Clinic booking wording.",
                   onboarded_by="src/tests/test_adapter_registry.py", replace=True)
    adapter = register_bespoke_domain(declaration, accepted_semantics="appointments per hour",
                                      accepted_units="appointments", replace=True)
    assert "clinic_appointments" in registered_domains()
    # Its lag policy is `none`, so precedence is refused by construction rather than by memory.
    assert "precedence" in adapter.structural_declaration({}).refused_operations
    assert adapter.plan_acquisition(adapter.resolve({}), {}).refusal.startswith(
        "select a content-addressed local record")


def test_an_irregular_record_obliges_a_declaration_it_cannot_satisfy():
    """TG8.4's rule, reused: detection creates an obligation and never discharges one."""
    from src.adapters.bespoke_record import assert_record_admissible
    from src.core.errors import InvalidParameterError

    declaration = DomainDeclaration(
        name="regular_only", description="A domain that has not admitted an irregular clock.",
        axes=(AxisSpec(name="t", role="time", units="s"),
              AxisSpec(name="channel", role="category", ordered=False)),
        licence="local", violations=("no_physical_metric", "no_propagation_speed",
                                     "unordered_channels"),
        lag_policy="none")
    onboard_domain(declaration, ORDER_BOOK_PHRASES, geometry=None,
                   glossary_description="test", onboarded_by="test", replace=True)
    with pytest.raises(InvalidParameterError, match="irregular_sampling"):
        assert_record_admissible(declaration, [0.0, 1.0, 3.5, 9.0])


def test_a_flat_record_cannot_be_read_under_a_domain_declaring_richer_axes():
    from src.adapters.bespoke_record import assert_record_admissible
    from src.core.builtin_domains import REANALYSIS
    from src.core.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError, match="channel table can supply"):
        assert_record_admissible(REANALYSIS, [0.0, 1.0, 2.0])


def test_the_bespoke_family_is_not_the_extension_seam_acceptance_test():
    """A data-driven instance tests an adapter's parameters, not the registry's seam.

    This is asserted rather than only documented because the temptation is real: the bespoke
    family can mint a working domain from the UI, and letting that stand in for the fifth-adapter
    test would mean shipping an extension point nobody had exercised.
    """
    from src.adapters import bespoke_record

    assert bespoke_record.ADAPTER.adapter_id != "%s.monotone-rank" % FIFTH_DOMAIN
    # The fifth adapter shares no implementation with the bespoke family.
    assert bespoke_record.ADAPTER.translate is not _fifth_translate
    fifth = _install_fifth_adapter()
    assert set(fifth.structural_declaration({}).channels) == {"monotone_rank"}
    assert set(bespoke_record.ADAPTER.structural_declaration({}).channels) == {
        "standardized_level"}
