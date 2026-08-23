"""TG2.1: the canonical feature record, and the comparisons it refuses.

Three groups, in the order the slice has to be believed:

1.  **The record holds a real measurement.** Features are built from the
    `advected_vortex_sequence` benchmark - measured positions and measured widths, never the
    recorded truth - and the quantities a tracker will need come back out of them correctly.
    A record that has only ever held a hand-written literal has not been tested.
2.  **Rule R19 is an invariant, not a docstring.** The cross-domain comparisons the record is
    designed to prevent are attempted here, and each one raises.
3.  **The declarations are read, not assumed.** Orientation conventions and significance
    bases come from registries, and a fourth of each drives the record without editing `src`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.benchmarks.core import get_benchmark
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.feature import (
    ORIENTATION_CONVENTIONS, SIGNIFICANCE_BASES, FeatureLocation, FeatureSet, Orientation,
    OrientationConvention, Quantity, SemanticComparisonError, SignificanceBasis,
    Significance, SpectralFeature,
)
from src.core.registry import restore, snapshot

ROW = AxisSpec("row", "space", units="cells", periodic=True, ordinal=0)
COL = AxisSpec("col", "space", units="cells", periodic=True, ordinal=1)
GRID_N = 128


# --------------------------------------------------------------------- measurement helpers


def _centroid_and_width(data: torch.Tensor):
    """Intensity centroid and Gaussian width of the dominant structure, in cells.

    Measured from the field alone. The benchmark's recorded answer is used to check this,
    never to produce it - a record populated from the truth dictionary would prove nothing
    about whether it can hold a measurement.

    Two details are not incidental. The centroid is **circular**, because the benchmark's
    grid is a torus and an arithmetic mean of coordinates either side of the seam lands in
    the middle of the wrong side of the field. And the width is taken from the integral of
    the blob over its peak (a Gaussian of amplitude `A` integrates to `2 pi sigma^2 A`)
    rather than from a second moment, because a second moment over the whole frame is
    dominated by the noise floor at the far corners, which have enormous lever arms.
    """
    n = data.shape[0]
    idx = torch.arange(n, dtype=torch.float64)
    weight = torch.clamp(data - data.mean(), min=0.0) ** 2

    def circular_mean(marginal: torch.Tensor) -> float:
        angle = 2 * math.pi * idx / n
        s = float((marginal * torch.sin(angle)).sum())
        c = float((marginal * torch.cos(angle)).sum())
        return (math.atan2(s, c) % (2 * math.pi)) * n / (2 * math.pi)

    cy = circular_mean(weight.sum(dim=1))
    cx = circular_mean(weight.sum(dim=0))
    base = float(data.median())
    peak = float(data.max()) - base
    integral = float((data - base).sum())
    return cy, cx, math.sqrt(integral / (2.0 * math.pi * peak))


def _vortex_features(steps: int = 24, **overrides):
    """A `FeatureSet` measured off the advected vortex benchmark."""
    bench = get_benchmark("advected_vortex_sequence")
    overrides["steps"] = steps
    seq = bench.make(**overrides)
    features = []
    for t, field in enumerate(seq.fields):
        cy, cx, width = _centroid_and_width(field.data)
        features.append(SpectralFeature(
            domain="synthetic_vortex",
            dataset="advected_vortex_sequence",
            variable="intensity",
            magnitude=Quantity(float(field.data.max()), "dimensionless"),
            location=FeatureLocation({"row": cy, "col": cx}, (ROW, COL),
                                     uncertainty={"row": 0.5, "col": 0.5}),
            time=float(t),
            time_units="frames",
            representation="identity",
            spatial_scale=Quantity(width, "cells", uncertainty=0.5),
            provenance={"benchmark": bench.name, "measured_by": "intensity_centroid"},
        ))
    return FeatureSet(features), seq, bench.truth(**overrides)


# ------------------------------------------------------------ 1. a real measurement


def test_a_measured_sequence_becomes_a_feature_set_in_time_order():
    features, seq, _ = _vortex_features()
    assert len(features) == len(seq.fields)
    assert features.times == tuple(float(t) for t in range(len(seq.fields)))
    assert features.domain == "synthetic_vortex"
    assert features.representation == "identity"
    # One feature per frame is what TG2.3 will associate across.
    assert [len(group) for _, group in features.frames()] == [1] * len(seq.fields)


def test_the_measured_positions_agree_with_the_benchmarks_recorded_trajectory():
    features, _, truth = _vortex_features()
    worst = 0.0
    for feature, (ty, tx) in zip(features, truth["positions_rowcol"]):
        dy = abs(feature.location.coords["row"] - ty)
        dx = abs(feature.location.coords["col"] - tx)
        worst = max(worst, math.hypot(min(dy, GRID_N - dy), min(dx, GRID_N - dx)))
    assert worst < 1.0, "measured centroid is %.3f cells from the known trajectory" % worst


def test_consecutive_separations_recover_the_known_step_velocity():
    features, _, truth = _vortex_features()
    expected = math.hypot(*truth["velocity_cells_per_step"])
    periods = {"row": float(GRID_N), "col": float(GRID_N)}
    steps = [features[i + 1].separation_to(features[i], periods=periods).value
             for i in range(len(features) - 1)]
    assert max(abs(s - expected) for s in steps) < 0.25
    # The separation came back as a quantity that still knows what it is measured in.
    one = features[1].separation_to(features[0], periods=periods)
    assert one.units == "cells" and one.uncertainty is not None


def test_the_measured_scale_ratio_recovers_the_known_doubling_schedule():
    features, _, truth = _vortex_features()
    ratio = features[-1].scale_ratio_to(features[0])
    steps = len(features) - 1
    expected = 2.0 ** (steps / truth["scale_doubling_steps"])
    assert abs(ratio - expected) / expected < 0.1, (
        "measured width ratio %.4f against the known %.4f" % (ratio, expected))


def test_elapsed_time_is_read_from_the_declared_clock():
    features, _, _ = _vortex_features()
    assert features[5].elapsed_to(features[2]) == 3.0
    assert features[0].time_units == "frames"


def _at(row: float, col: float, time: float) -> SpectralFeature:
    return SpectralFeature(
        domain="synthetic_vortex", dataset="advected_vortex_sequence", variable="intensity",
        magnitude=Quantity(1.0, "dimensionless"),
        location=FeatureLocation({"row": row, "col": col}, (ROW, COL)),
        time=time, time_units="frames", representation="identity",
        spatial_scale=Quantity(5.0, "cells"))


def test_a_periodic_axis_without_its_period_is_refused_rather_than_wrapped_wrongly():
    # Two positions two cells apart across the seam of a 128-cell periodic axis.
    before = _at(32.0, 127.0, 0.0)
    after = _at(32.0, 1.0, 1.0)
    with pytest.raises(InvalidParameterError) as excinfo:
        after.separation_to(before)
    assert "periodic" in str(excinfo.value)

    periods = {"row": float(GRID_N), "col": float(GRID_N)}
    assert after.separation_to(before, periods=periods).value == pytest.approx(2.0)
    # The number the refusal exists to prevent: 126 cells apart, on a grid 128 wide, for two
    # positions two cells apart. A tracker fed that would report a death and a birth.
    naive = abs(after.location.coords["col"] - before.location.coords["col"])
    assert naive == pytest.approx(126.0)


def test_the_benchmarks_trajectory_wraps_but_its_field_does_not_defect_d59():
    """D59, found while giving TG2.1 a real user.

    `truth_advected_vortex` takes the vortex position modulo `n`, so its recorded answer is
    a trajectory on a torus - and `build_advected_vortex` draws the blob with a plain
    Euclidean Gaussian that does not wrap. Away from the boundary the two agree to better
    than a cell, which is why nobody has noticed: at the benchmark's own parameters the
    vortex never reaches the edge. Started near it, the recorded position and the field part
    company by several cells for the frames where the blob is clipped by the boundary.

    This matters for TG2.3, whose acceptance criterion is *1 track, 1 birth, 0 deaths* on
    this sequence. A tracker run on a wrapping parameterisation would see the object fade
    out at one edge and appear at the other, and would be right to report a death and a
    birth - against a recorded answer that says there are none. The benchmark is left as it
    is: changing its field builder would move every number it produces, and TG2.1 is not the
    slice that gets to do that. It is written down, and it is checked, so the next slice
    meets it as a fact rather than as a surprise.
    """
    features, _, truth = _vortex_features(start=(32.0, 120.0))
    errors = []
    for feature, (_, tx) in zip(features, truth["positions_rowcol"]):
        delta = abs(feature.location.coords["col"] - tx)
        errors.append(min(delta, GRID_N - delta))
    near_seam = max(errors[1:5])
    away_from_seam = max(errors[8:])
    assert near_seam > 1.5, "the seam frames must actually disagree for this to be D59"
    assert away_from_seam < 1.0, "away from the seam the measurement is sound"


def test_a_period_on_a_non_periodic_axis_is_refused():
    flat_row = AxisSpec("row", "space", units="cells", periodic=False)
    flat_col = AxisSpec("col", "space", units="cells", periodic=False)
    a = FeatureLocation({"row": 1.0, "col": 2.0}, (flat_row, flat_col))
    b = FeatureLocation({"row": 4.0, "col": 6.0}, (flat_row, flat_col))
    assert a.separation_to(b).value == pytest.approx(5.0)
    with pytest.raises(InvalidParameterError) as excinfo:
        a.separation_to(b, periods={"row": 128.0})
    assert "not joined" in str(excinfo.value)


def test_a_location_refuses_axes_in_mixed_units():
    metres = AxisSpec("y", "space", units="m")
    cells = AxisSpec("x", "space", units="cells")
    mixed = FeatureLocation({"y": 1.0, "x": 2.0}, (metres, cells))
    with pytest.raises(SemanticComparisonError) as excinfo:
        mixed.separation_to(mixed)
    assert "do not add to a length" in str(excinfo.value)


def test_a_location_refuses_to_carry_a_second_clock():
    with pytest.raises(InvalidParameterError) as excinfo:
        FeatureLocation({"t": 1.0}, (AxisSpec("t", "time", units="s"),))
    assert "second clock" in str(excinfo.value)


def test_a_location_refuses_a_coordinate_for_an_axis_it_does_not_declare():
    with pytest.raises(InvalidParameterError):
        FeatureLocation({"row": 1.0, "col": 2.0, "level": 3.0}, (ROW, COL))
    with pytest.raises(UnknownNameError):
        FeatureLocation({"row": 1.0, "col": 2.0}, (ROW, COL), uncertainty={"lat": 1.0})


# ------------------------------------------------------------ 2. rule R19 is an invariant


def _one(domain: str, variable: str, units, **overrides) -> SpectralFeature:
    kwargs = dict(
        domain=domain, dataset="%s_v1" % domain, variable=variable,
        magnitude=Quantity(3.0, units),
        location=FeatureLocation({"row": 1.0, "col": 2.0}, (ROW, COL)),
        time=0.0, time_units="frames", representation="identity",
        spatial_scale=Quantity(4.0, "cells"))
    kwargs.update(overrides)
    return SpectralFeature(**kwargs)


def test_a_cross_domain_magnitude_comparison_raises():
    weather = _one("era5", "temperature", "K")
    market = _one("exchange", "volume", "shares")
    with pytest.raises(SemanticComparisonError) as excinfo:
        weather.compare_magnitude_to(market)
    message = str(excinfo.value)
    assert "R19" in message and "era5" in message and "exchange" in message


def test_a_same_domain_magnitude_comparison_is_allowed_and_keeps_its_units():
    hot = _one("era5", "temperature", "K", magnitude=Quantity(290.0, "K"))
    cold = _one("era5", "temperature", "K", magnitude=Quantity(285.0, "K"))
    difference = hot.compare_magnitude_to(cold)
    assert difference.value == pytest.approx(5.0)
    assert difference.units == "K"


def test_two_variables_of_one_dataset_are_still_not_comparable():
    temperature = _one("era5", "temperature", "K")
    wind = _one("era5", "wind_speed", "m s-1")
    wind = SpectralFeature(**{**wind.__dict__, "dataset": temperature.dataset})
    with pytest.raises(SemanticComparisonError):
        temperature.compare_magnitude_to(wind)


def test_a_cross_domain_separation_and_elapsed_time_both_raise():
    weather = _one("era5", "temperature", "K")
    market = _one("exchange", "volume", "shares")
    with pytest.raises(SemanticComparisonError):
        weather.separation_to(market)
    with pytest.raises(SemanticComparisonError):
        weather.elapsed_to(market)


def test_a_scale_ratio_does_cross_a_domain_boundary_but_only_when_the_units_cancel():
    weather = _one("era5", "temperature", "K", spatial_scale=Quantity(8.0, "cells"))
    market = _one("exchange", "volume", "shares", spatial_scale=Quantity(4.0, "cells"))
    assert weather.scale_ratio_to(market) == pytest.approx(2.0)

    metric = _one("era5", "temperature", "K", spatial_scale=Quantity(8.0, "m"))
    with pytest.raises(SemanticComparisonError) as excinfo:
        metric.scale_ratio_to(market)
    assert "do not cancel" in str(excinfo.value)


def test_the_structural_signature_carries_no_magnitude_units_or_variable():
    feature = _one("era5", "temperature", "K",
                   orientation=Orientation(170.0),
                   significance=Significance(0.002, "surrogate_p_value", 4999))
    signature = feature.structural_signature()
    flat = repr(sorted(signature.items()))
    for forbidden in ("magnitude", "units", "variable", "dataset", "temperature", "era5"):
        assert forbidden not in flat
    assert signature["orientation_degrees"] == pytest.approx(170.0)
    assert signature["significance"] == pytest.approx(0.002)


def test_a_feature_set_refuses_to_mix_domains_representations_or_clocks():
    weather = _one("era5", "temperature", "K")
    market = _one("exchange", "volume", "shares")
    with pytest.raises(SemanticComparisonError):
        FeatureSet([weather, market])

    swt = _one("era5", "temperature", "K", representation="swt_db2_level3")
    with pytest.raises(SemanticComparisonError) as excinfo:
        FeatureSet([weather, swt])
    assert "R8" in str(excinfo.value)

    seconds = _one("era5", "temperature", "K", time_units="s")
    with pytest.raises(SemanticComparisonError):
        FeatureSet([weather, seconds])


def test_an_empty_feature_set_is_refused_rather_than_treated_as_a_result():
    with pytest.raises(InvalidParameterError) as excinfo:
        FeatureSet([])
    assert "no features found" in str(excinfo.value)


def test_the_full_record_keeps_every_field_the_roadmap_named():
    features, _, _ = _vortex_features()
    record = features[0].describe()
    for key in ("domain", "dataset", "variable", "units", "location", "time",
                "spatial_scale", "temporal_scale", "orientation", "magnitude",
                "significance", "extent", "uncertainty", "representation", "provenance"):
        assert key in record, key
    assert record["units"] == "dimensionless"
    assert record["uncertainty"]["spatial_scale"] == pytest.approx(0.5)
    assert record["uncertainty"]["spatial_scale_units"] == "cells"
    assert record["provenance"]["benchmark"] == "advected_vortex_sequence"


def test_a_set_receipt_reports_what_it_is_before_what_is_in_it():
    features, _, _ = _vortex_features()
    record = features.describe()
    assert record["count"] == len(features)
    assert record["frame_count"] == len(features)
    assert record["time_span"] == [0.0, float(len(features) - 1)]
    assert record["representation"] == "identity"


# ------------------------------------------------------------ 3. declared, not assumed


def test_an_undirected_axis_and_a_direction_wrap_differently():
    ridge = Orientation(170.0, "axis_180")
    other = Orientation(350.0, "axis_180")
    assert ridge.separation_to(other) == pytest.approx(0.0)

    heading = Orientation(170.0, "direction_360")
    opposed = Orientation(350.0, "direction_360")
    assert heading.separation_to(opposed) == pytest.approx(180.0)


def test_comparing_two_orientations_under_different_conventions_raises():
    with pytest.raises(SemanticComparisonError) as excinfo:
        Orientation(170.0, "axis_180").separation_to(Orientation(170.0, "direction_360"))
    assert "opposite directions" in str(excinfo.value)


def test_an_orientation_is_stored_wrapped_into_its_own_period():
    assert Orientation(190.0, "axis_180").degrees == pytest.approx(10.0)
    assert Orientation(-10.0, "direction_360").degrees == pytest.approx(350.0)
    assert Orientation(190.0, "axis_180").difference_to(
        Orientation(0.0, "axis_180")) == pytest.approx(10.0)


def test_an_unregistered_orientation_convention_is_refused():
    with pytest.raises(UnknownNameError) as excinfo:
        Orientation(10.0, "radians_2pi")
    assert "axis_180" in str(excinfo.value)


def test_a_fourth_convention_drives_the_record_without_editing_src():
    state = snapshot(ORIENTATION_CONVENTIONS)
    try:
        ORIENTATION_CONVENTIONS.add(
            "hexagonal_60",
            OrientationConvention(60.0, False, "A hexagonal lattice repeats every 60."),
            capabilities={"period_degrees": 60.0, "directed": False})
        a = Orientation(5.0, "hexagonal_60")
        b = Orientation(55.0, "hexagonal_60")
        assert a.separation_to(b) == pytest.approx(10.0)
        feature = _one("lattice", "amplitude", "counts", orientation=a)
        assert feature.describe()["orientation"]["period_degrees"] == 60.0
        assert feature.structural_signature()["orientation_convention"] == "hexagonal_60"
    finally:
        restore(ORIENTATION_CONVENTIONS, state)


def test_a_p_value_below_the_surrogate_resolution_floor_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        Significance(1e-4, "surrogate_p_value", 999)
    assert "was not measured" in str(excinfo.value)
    # The floor itself is admissible: 1 / (1 + 999).
    ok = Significance(1.0 / 1000.0, "surrogate_p_value", 999)
    assert ok.resolution_floor == pytest.approx(1e-3)


def test_an_empirical_p_value_without_its_ensemble_size_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        Significance(0.01, "surrogate_p_value")
    assert "resolution floor" in str(excinfo.value)


def test_not_tested_is_a_declared_state_and_not_a_missing_one():
    none = Significance(1.0, "declared_none")
    assert none.describe()["is_p_value"] is False
    assert none.resolution_floor is None


def test_a_fourth_significance_basis_registers_without_editing_src():
    state = snapshot(SIGNIFICANCE_BASES)
    try:
        SIGNIFICANCE_BASES.add(
            "analytic_p_value",
            SignificanceBasis(False, "A closed-form p-value with no ensemble behind it."),
            capabilities={"is_p_value": False, "threshold_free": True})
        sig = Significance(1e-9, "analytic_p_value")
        assert sig.describe()["basis"] == "analytic_p_value"
    finally:
        restore(SIGNIFICANCE_BASES, state)


def test_a_quantity_will_not_hold_a_nan_or_an_empty_unit():
    with pytest.raises(InvalidParameterError):
        Quantity(float("nan"), "K")
    with pytest.raises(InvalidParameterError) as excinfo:
        Quantity(1.0, "  ")
    assert "not recorded" in str(excinfo.value)
    assert Quantity(1.0, None).units is None


def test_a_feature_must_name_the_representation_that_produced_it():
    with pytest.raises(InvalidParameterError) as excinfo:
        _one("era5", "temperature", "K", representation="")
    assert "representation" in str(excinfo.value)


def test_a_bare_float_magnitude_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        _one("era5", "temperature", "K", magnitude=3.0)
    assert "kelvin" in str(excinfo.value)


def test_a_zero_spatial_scale_is_refused_because_it_has_no_ratio():
    with pytest.raises(InvalidParameterError) as excinfo:
        _one("era5", "temperature", "K", spatial_scale=Quantity(0.0, "cells"))
    assert "undefined" in str(excinfo.value) or "positive" in str(excinfo.value)


def test_the_registries_declare_their_periods_as_capabilities():
    assert ORIENTATION_CONVENTIONS.entry("axis_180").capabilities["period_degrees"] == 180.0
    directed = ORIENTATION_CONVENTIONS.with_capability("directed", True)
    assert [entry.name for entry in directed] == ["direction_360"]
    threshold_free = SIGNIFICANCE_BASES.with_capability("threshold_free", True)
    assert len(threshold_free) == len(SIGNIFICANCE_BASES)


def test_numpy_scalars_are_accepted_as_measured_values():
    # Every measurement in this tree arrives as a numpy or torch scalar at some point, and a
    # record that only accepts a Python float would be populated by a cast nobody checks.
    quantity = Quantity(np.float64(2.5), "cells")
    assert isinstance(quantity.value, float) and quantity.value == pytest.approx(2.5)
