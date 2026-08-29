"""TG1.4: a rank-3 domain through the accepted falsification layer, with `PhysicalField` intact.

Section 2.2 of the cross-domain roadmap calls `PhysicalField`'s strict 2D constraint "the single
largest obstacle" to a second domain. Standard E12 forbids relaxing it. This file is the test
that both statements can be true at once: a domain whose observation is a
``(band, detector, repeat)`` block runs the unmodified sweep and the unmodified replication
gate, and `PhysicalField` still refuses a 1D series and a 3D volume exactly as it always did.

Three groups.

*   **The spine is real.** A planted coupling is recovered and an autocorrelated control is
    not. This pair is the whole test, in the same way it was for TG0.2: a detector that finds
    a planted signal but also finds signal in AR(1) noise has found nothing. Without it, a
    sample type is speculative generality with a docstring.
*   **`PhysicalField` is untouched, and the bridge is one-way.** The acceptance criterion.
*   **Declaration, not inference** — the refusals that make a sample a sample rather than an
    array with opinions, plus the reduction registry.

Nothing here reaches the network, and the record is synthetic. As with TG0.2, this is evidence
about the machinery and about no real-world domain.
"""

import numpy as np
import pytest

from src.analysis_engine import domain_analysis as da
from src.analysis_engine.cross_scale import GateProtocol
from src.core.channel_series import GATE_MEASURES, MeasureSpec, require_gate_measure
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError, ShapeMismatchError, UnknownNameError
from src.core.registry import restore, snapshot
from src.core.sample import (SAMPLE_REDUCTIONS, SampleError, SampleReduction,
                             StructuredSample, channel_series_from_samples, gate_measure_of,
                             reduction_names)
from src.physical_core.field import PhysicalField

CADENCE = 60.0
N_FRAMES = 800
PLANTED_LAG = 3
BANDS = ("red", "green", "blue")
N_DETECTORS = 4
N_REPEATS = 5
LICENCE = "CC-BY-4.0 (synthetic fixture; no real archive was accessed)"
VIOLATIONS = ("no_physical_metric", "no_propagation_speed", "no_natural_cycle",
              "unordered_channels")

#: The declared axes of one frame. Two of them have no meaning the platform could have
#: guessed: `detector` is a category, and `repeat` is a category too - repeats are not a
#: metric axis and nothing may be reported per repeat.
FRAME_AXES = (AxisSpec("band", "category", ordinal=0),
              AxisSpec("detector", "category", ordinal=1),
              AxisSpec("repeat", "category", ordinal=2))


def _ar1(rng, n=N_FRAMES, rho=0.6):
    x = np.zeros(n)
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    return x


def _frames(band_series, rng, jitter=0.05):
    """One `(band, detector, repeat)` sample per frame, around each band's own value."""
    samples = []
    stacked = np.stack([band_series[name] for name in BANDS], axis=1)  # (time, band)
    for row in range(N_FRAMES):
        block = (stacked[row][:, None, None]
                 + jitter * rng.normal(size=(len(BANDS), N_DETECTORS, N_REPEATS)))
        samples.append(StructuredSample(
            block, FRAME_AXES,
            coords={"band": np.array(BANDS)},
            units="counts",
            provenance={"instrument": "synthetic three-band photometer"}))
    return samples


def _series(samples, **kwargs):
    return channel_series_from_samples(
        samples, times_seconds=np.arange(len(samples), dtype=float) * CADENCE,
        channel_axis="band",
        support_parent_px={label: 1.0 for label in BANDS},
        provenance={"licence": LICENCE}, **kwargs)


def _planted(seed=11, coupling=0.9):
    """`red` at t sets `blue` at t+3. `green` is an innocent bystander."""
    rng = np.random.default_rng(seed)
    red, green, blue_noise = _ar1(rng), _ar1(rng), _ar1(rng)
    blue = np.empty(N_FRAMES)
    blue[:PLANTED_LAG] = blue_noise[:PLANTED_LAG]
    blue[PLANTED_LAG:] = (coupling * red[:-PLANTED_LAG]
                          + (1.0 - coupling) * blue_noise[PLANTED_LAG:])
    return _frames({"red": red, "green": green, "blue": blue}, rng)


def _independent(seed=23):
    rng = np.random.default_rng(seed)
    return _frames({name: _ar1(rng) for name in BANDS}, rng)


def _declaration(**overrides):
    kwargs = dict(
        name="synthetic_photometer",
        description="A three-band photometer. Each frame is a band-by-detector-by-repeat "
                    "block of counts. No grid, no metric, no transform, no physics.",
        axes=[AxisSpec("time", "time", units="s"),
              AxisSpec("band", "category"),
              AxisSpec("detector", "category", ordinal=1),
              AxisSpec("repeat", "category", ordinal=2)],
        licence=LICENCE, violations=VIOLATIONS,
        lag_policy="declared", declared_floor_frames=2,
        declared_floor_basis="two integration periods: a band's counts are accumulated over "
                             "one period and a change cannot be observed before the next "
                             "period has been read out")
    kwargs.update(overrides)
    return DomainDeclaration(**kwargs)


# ============================================================ 1. the spine is real


def test_a_rank_three_domain_recovers_a_planted_coupling_through_the_unmodified_sweep():
    """The point of the whole slice: data `PhysicalField` cannot hold, gated anyway."""
    result = da.analyse_precedence(
        _series(_planted()), _declaration(), lags=[PLANTED_LAG], cadence_seconds=CADENCE,
        measure="mean", estimator="mutual_information", bins=4, n_surrogates=299,
        alpha=0.05, correction="benjamini_yekutieli", seed=5)

    significant = {(row["source_scale"], row["target_scale"]) for row in result["results"]
                   if row.get("significant")}
    assert ("red", "blue") in significant, result["results"]
    # The bystander is not swept up. A test that reported every pair would have found the
    # record's autocorrelation, not the coupling planted in it.
    assert ("red", "green") not in significant
    assert result["claim_boundary"] == "precedence"
    # And the receipt says which spine produced the numbers.
    assert result["channel_series"]["provenance"]["spine"] == "structured_sample"
    assert result["channel_series"]["provenance"]["sample_shape"] == [3, 4, 5]


def test_independent_autocorrelated_bands_produce_no_significant_relationship():
    """The control. Without it the test above is not evidence of anything."""
    result = da.analyse_precedence(
        _series(_independent()), _declaration(), lags=[PLANTED_LAG], cadence_seconds=CADENCE,
        measure="mean", estimator="mutual_information", bins=4, n_surrogates=299,
        alpha=0.05, correction="benjamini_yekutieli", seed=5)

    assert not [row for row in result["results"] if row.get("significant")], \
        result["results"]


def test_a_sample_domain_passes_the_unmodified_replication_gate():
    """The replication gate (R6) sees a `ChannelSeries` and cannot tell where it came from."""
    protocol = GateProtocol(
        study_id="sample_spine_photometer_v1",
        lags=(PLANTED_LAG,), cadence_seconds=CADENCE, measure="mean",
        estimator="mutual_information", bins=4, n_surrogates=499, alpha=0.05,
        correction="benjamini_yekutieli", seed=31, train_ratio=0.6, embargo_frames=PLANTED_LAG,
        n_scales=len(BANDS), expected_frames=N_FRAMES, require_advection_floor=False)

    gate = da.run_domain_gate(_series(_planted()), _declaration(), protocol)

    assert gate["gate"]["verdict"] == "PASS", gate["gate"]
    assert gate["domain"]["name"] == "synthetic_photometer"
    # The floor that decided admissibility is the one the domain declared, not a borrowed one.
    assert gate["train"]["applied_lag_floor"]["policy"] == "declared"


def test_the_reduction_is_what_the_channel_carries_not_the_raw_block():
    """A sanity check on the arithmetic, computed independently of the module under test."""
    samples = _planted()
    series = _series(samples, reductions=("mean", "variance"))

    expected = np.mean(np.asarray(samples[7].values)[1])
    assert series.to_matrix("mean")[7, 1] == pytest.approx(expected)
    assert series.to_matrix("variance")[7, 2] == pytest.approx(
        np.var(np.asarray(samples[7].values)[2]))
    assert set(series.provenance["reduction_measures"]) == {"mean", "variance"}


def test_the_receipt_says_which_reduction_produced_the_numbers():
    """Two reductions may emit one measure, so the measure name alone does not identify one.

    `from_channel_series` published only the *names* of the provenance keys until TG1.4. That
    was harmless while every producer's record lived on the `DomainDeclaration`; the sample
    spine puts the arithmetic on the series, so the values are now carried too - and anything
    a lineage record cannot hold is named by type rather than dropped.
    """
    samples = _planted()[:8]
    series = channel_series_from_samples(
        samples, times_seconds=np.arange(8.0) * CADENCE, channel_axis="band",
        reductions=("energy_density",),
        provenance={"instrument_object": object()})
    result = da.analyse_precedence(
        series, _declaration(), lags=[PLANTED_LAG], cadence_seconds=CADENCE,
        measure="energy_density", estimator="mutual_information", bins=2,
        n_surrogates=19, alpha=0.05, correction="benjamini_yekutieli", seed=5)

    recorded = result["channel_series"]["provenance"]
    assert recorded["reduction_measures"] == {"energy_density": "energy_density"}
    assert recorded["channel_axis"] == "band"
    assert recorded["sample_rank"] == 3
    assert recorded["instrument_object"] == "<object>"


# ================================================ 2. PhysicalField is untouched (acceptance)


def test_physical_field_still_raises_on_non_2d_input():
    """The acceptance criterion, stated as bluntly as the roadmap states it (E12)."""
    import torch

    for shape in [(16,), (4, 4, 4), (2, 3, 4, 5)]:
        with pytest.raises(ValueError, match="must be 2D"):
            PhysicalField(torch.zeros(*shape))


def test_the_generalisation_is_not_reachable_through_the_analysis_object():
    """`PhysicalField` accepts no sample, by any keyword, at any rank."""
    sample = StructuredSample(np.zeros((4, 5, 6)), FRAME_AXES)

    with pytest.raises((ValueError, TypeError, AttributeError)):
        PhysicalField(sample)
    # There is no adapter hiding on the analysis object either: the bridge is a method of the
    # sample, so the direction of the dependency is visible in the call.
    assert not any("sample" in name.lower() for name in dir(PhysicalField))


def test_a_rank_three_sample_refuses_the_bridge_and_says_what_would_be_needed():
    sample = StructuredSample(np.zeros((4, 5, 6)), FRAME_AXES)

    with pytest.raises(SampleError) as excinfo:
        sample.to_physical_field()

    message = str(excinfo.value)
    assert "exactly two axes" in message
    assert "'space'" in message
    # It also points at the path that does work, rather than leaving a dead end.
    assert "ChannelSeries" in message


def test_a_two_dimensional_sample_of_categories_is_still_refused():
    """Rank 2 is not the condition; two *declared spatial* axes is."""
    panel = StructuredSample(np.zeros((4, 5)),
                             [AxisSpec("station", "category"),
                              AxisSpec("band", "category", ordinal=1)])

    with pytest.raises(SampleError, match="0 declared spatial axes"):
        panel.to_physical_field()


def test_a_declared_spatial_pair_crosses_into_the_analysis_spine():
    values = np.arange(20, dtype=float).reshape(4, 5)
    sample = StructuredSample(
        values, [AxisSpec("y", "space", units="m"), AxisSpec("x", "space", units="m",
                                                             ordinal=1)],
        units="K", split="train")

    field = sample.to_physical_field()

    assert isinstance(field, PhysicalField)
    assert field.data.shape == (4, 5)
    assert field.units == "K"
    assert field.split == "train"
    assert np.allclose(field.data.numpy(), values)
    # No metric is invented on the way across. A sample carries none, so the field gets the
    # honest default rather than a grid nobody asserted.
    assert not field.grid.is_physical


def test_the_bridge_refuses_a_declaration_that_runs_against_array_order():
    """A transposed field still looks like a field. It is refused, never quietly transposed."""
    sample = StructuredSample(
        np.zeros((4, 5)), [AxisSpec("x", "space", ordinal=1), AxisSpec("y", "space",
                                                                      ordinal=0)])

    with pytest.raises(SampleError, match="array order"):
        sample.to_physical_field()


# ============================================== 3. declaration, not inference


def test_an_undeclared_axis_is_refused_rather_than_named_or_positioned():
    """The reason the type exists: `x` and `y` do not make a geometry (E14).

    Refused structurally, not by a check afterwards. An `AxisSpec` cannot exist without a
    declared role, so a sample whose roles were guessed is unconstructible - which is a
    stronger statement than a validation that could be bypassed by another constructor.
    """
    with pytest.raises(SampleError, match="not a role"):
        StructuredSample(np.zeros((4, 5)), ["y", "x"])


def test_every_axis_of_every_sample_records_a_declared_basis():
    """The property the structural refusal buys: no basis is ever `name` or `position`."""
    sample = StructuredSample(np.zeros((4, 5, 6)), FRAME_AXES)

    assert sample.resolution.fully_declared
    assert set(sample.resolution.basis.values()) == {"declared"}
    assert sample.resolution.guessed_axes() == ()


def test_a_bare_string_axis_is_refused_beside_a_declared_one():
    with pytest.raises(SampleError, match="not a role"):
        StructuredSample(np.zeros((4, 5)),
                         [AxisSpec("y", "space"), "x"])


def test_the_axis_count_must_match_the_array_rank():
    with pytest.raises(ShapeMismatchError):
        StructuredSample(np.zeros((4, 5, 6)), FRAME_AXES[:2])


def test_a_coordinate_that_does_not_span_its_axis_is_refused():
    """The D56 failure mode, refused at construction instead of discovered in a result."""
    with pytest.raises(ShapeMismatchError):
        StructuredSample(np.zeros((4, 5)),
                         [AxisSpec("y", "space"), AxisSpec("x", "space", ordinal=1)],
                         coords={"x": np.arange(4.0)})


def test_an_ordered_axis_with_a_wandering_coordinate_is_refused():
    with pytest.raises(SampleError, match="monotonic"):
        StructuredSample(np.zeros((4,)), [AxisSpec("depth", "level")],
                         coords={"depth": np.array([0.0, 5.0, 2.0, 9.0])})


def test_an_unordered_axis_may_carry_any_labels():
    """A category axis has no order to violate; refusing one would be inventing a rule."""
    sample = StructuredSample(
        np.zeros((3,)), [AxisSpec("station", "category", ordered=False)],
        coords={"station": np.array([7.0, 2.0, 5.0])})
    assert sample.labels("station") == (7.0, 2.0, 5.0)


def test_a_sample_is_immutable_once_its_checks_have_passed():
    values = np.ones((2, 3))
    sample = StructuredSample(values, [AxisSpec("a", "category"),
                                       AxisSpec("b", "category", ordinal=1)])

    values[0, 0] = 99.0  # the caller's array is theirs; the sample took a copy
    assert sample.values[0, 0] == 1.0
    with pytest.raises(ValueError):
        sample.values[0, 0] = 99.0


def test_select_drops_an_axis_and_records_which_label_was_taken():
    sample = StructuredSample(np.arange(24, dtype=float).reshape(2, 3, 4), FRAME_AXES,
                              coords={"band": np.array([0.0, 1.0])})

    picked = sample.select("band", 1)

    assert picked.shape == (3, 4)
    assert picked.axis_names == ("detector", "repeat")
    assert picked.provenance["selected"] == {"band": 1.0}
    assert np.allclose(picked.values, np.arange(24, dtype=float).reshape(2, 3, 4)[1])


def test_frames_of_one_record_must_agree_on_their_structure():
    good = StructuredSample(np.zeros((3, 4, 5)), FRAME_AXES)
    other = StructuredSample(np.zeros((3, 4, 6)), FRAME_AXES)

    with pytest.raises(ShapeMismatchError):
        channel_series_from_samples([good, other], times_seconds=[0.0, 60.0],
                                    channel_axis="band")


def test_the_clock_axis_cannot_be_split_into_channels():
    sample = StructuredSample(np.zeros((3, 4)),
                              [AxisSpec("t", "time"), AxisSpec("band", "category")])

    with pytest.raises(InvalidParameterError, match="not the clock"):
        channel_series_from_samples([sample, sample], times_seconds=[0.0, 60.0],
                                    channel_axis="t")


# ------------------------------------------------------------------ the reduction registry


class _BandPeak(SampleReduction):
    """A domain's own arithmetic: the largest single reading in the block."""

    name = "peak_count"

    def apply(self, block):
        return float(np.max(block))


@pytest.fixture
def registered_peak():
    reductions, measures = snapshot(SAMPLE_REDUCTIONS), snapshot(GATE_MEASURES)
    try:
        GATE_MEASURES.add(
            "peak_count", MeasureSpec(True, "an order statistic of the block; no threshold "
                                            "and no normalisation enter it"),
            description="largest single reading in a structured sample")
        SAMPLE_REDUCTIONS.add(
            "peak_count", _BandPeak(),
            capabilities={"gate_measure": "peak_count", "threshold_free": True},
            description="Largest single reading over the reduced axes.")
        yield
    finally:
        restore(SAMPLE_REDUCTIONS, reductions)
        restore(GATE_MEASURES, measures)


def test_a_fourth_reduction_registers_and_runs_a_whole_sweep_without_editing_src(
        registered_peak):
    """The registry acceptance test, matching TG1.3's fourth lag policy."""
    assert "peak_count" in reduction_names()

    result = da.analyse_precedence(
        _series(_planted(), reductions=("peak_count",)), _declaration(),
        lags=[PLANTED_LAG], cadence_seconds=CADENCE, measure="peak_count",
        estimator="mutual_information", bins=4, n_surrogates=299, alpha=0.05,
        correction="benjamini_yekutieli", seed=5)

    assert ("red", "blue") in {(row["source_scale"], row["target_scale"])
                               for row in result["results"] if row.get("significant")}
    assert result["channel_series"]["provenance"]["reductions"] == ["peak_count"]


def test_a_reduction_is_keyed_by_the_gate_measure_it_declares(registered_peak):
    """The capability is load-bearing: it is the name a protocol freezes."""
    series = _series(_planted()[:4], reductions=("peak_count",))
    assert gate_measure_of("peak_count") == "peak_count"
    assert set(series.measures) == {"peak_count"}
    assert require_gate_measure("peak_count").threshold_free


def test_a_reduction_emitting_an_unregistered_measure_is_refused_at_build():
    """A series that could be analysed but never gated is not a usable series."""
    state = snapshot(SAMPLE_REDUCTIONS)
    try:
        SAMPLE_REDUCTIONS.add("orphan", _BandPeak(),
                              capabilities={"gate_measure": "no_such_measure"})
        with pytest.raises(UnknownNameError):
            _series(_planted()[:4], reductions=("orphan",))
    finally:
        restore(SAMPLE_REDUCTIONS, state)


def test_two_reductions_emitting_the_same_measure_are_refused_not_merged():
    state = snapshot(SAMPLE_REDUCTIONS)
    try:
        SAMPLE_REDUCTIONS.add("mean_again", _BandPeak(),
                              capabilities={"gate_measure": "mean"})
        with pytest.raises(InvalidParameterError, match="distinct gate measures"):
            _series(_planted()[:4], reductions=("mean", "mean_again"))
    finally:
        restore(SAMPLE_REDUCTIONS, state)


def test_an_unknown_reduction_names_the_registered_alternatives():
    with pytest.raises(UnknownNameError) as excinfo:
        _series(_planted()[:4], reductions=("median",))
    assert "mean" in str(excinfo.value)


def test_a_footprint_may_be_declared_per_channel_and_is_never_invented():
    """Same discipline as `ChannelSeries`: no default, because a default sets every floor."""
    samples = _planted()[:4]
    without = channel_series_from_samples(
        samples, times_seconds=np.arange(4.0) * CADENCE, channel_axis="band")
    assert all("support_parent_px" not in record
               for record in without.channel_records)

    with pytest.raises(InvalidParameterError, match="footprints for declared channels"):
        channel_series_from_samples(
            samples, times_seconds=np.arange(4.0) * CADENCE, channel_axis="band",
            support_parent_px={"nonexistent": 1.0})
