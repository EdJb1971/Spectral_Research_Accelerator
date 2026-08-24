"""TG2.4 - the representation-induced feature audit.

Five questions, in the order they decide whether the audit means anything.

1.  Does it find nothing on structureless data, and *can* it find anything at all?
2.  Does the null carry the representation, or does it get rebuilt where the artefact is?
3.  Can the null be violated, or is it a test that cannot fail?
4.  Does the audit pay for its own family, or does it get its floor by running forty-five
    tests at 0.05 and reporting the survivors?
5.  Does every plane say what its indices mean, and does every lens get looked through?
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.extraction import ExtractionField, calibrate
from src.core.feature import FeatureSet
from src.core.representation import (
    DEFAULT_AUDIT_SURROGATES,
    FAMILY_CORRECTIONS,
    NULL_STRATEGIES,
    REPRESENTATIONS,
    CorrectedLevel,
    FamilyTooLargeError,
    RepresentationPlane,
    VACUITY_TOLERANCE,
    _fwer_curve,
    audit_all,
    audit_representation,
    decimation,
    plane_field,
    representation_planes,
    searchable_cells,
)
from src.core.registry import restore, snapshot

GRID_N = 64
AXES = (AxisSpec("row", "space", units="cells", ordinal=0),
        AxisSpec("col", "space", units="cells", ordinal=1))
PERIODIC_AXES = (AxisSpec("row", "space", units="cells", periodic=True, ordinal=0),
                 AxisSpec("col", "space", units="cells", periodic=True, ordinal=1))
SEED = 20260825

#: Two lenses, twenty planes: enough family that the correction has to do real work, few
#: enough that `family / alpha` surrogates run in a test. The whole registry at once is the
#: `representation_null_field` benchmark's job, and it costs a thousand decompositions.
SMALL_SET = ("raw", "swt")


def _noise(seed: int = SEED, n: int = GRID_N, scale: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(scale=scale, size=(n, n))


def _correlated(seed: int = SEED, n: int = GRID_N, beta: float = 3.4) -> np.ndarray:
    """A scale-free field: correlated, smooth, edge-bearing, and with nothing planted in it.

    The hard null. White noise gives a lens almost nothing to turn into a localised
    artefact; a smooth field gives it edges, seams and a decimation phase to work with.
    """
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    k[0, 0] = 1.0
    amplitude = k ** (-beta / 2.0)
    amplitude[0, 0] = 0.0
    spectrum = amplitude * np.fft.fft2(rng.normal(size=(n, n)))
    out = np.real(np.fft.ifft2(spectrum))
    return out / float(np.std(out))


def _blob(amplitude: float = 6.0, cy: float = 38.0, cx: float = 25.0, sigma: float = 5.0,
          n: int = GRID_N, seed: int = SEED) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    yy, xx = np.meshgrid(idx, idx, indexing="ij")
    peak = amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
    return peak + np.random.default_rng(seed).normal(size=(n, n))


def _field(values: np.ndarray, axes=AXES, **kwargs) -> ExtractionField:
    params = dict(domain="synthetic", dataset="audit", variable="amplitude", units=None,
                  time=0.0, time_units="frames", representation="identity")
    params.update(kwargs)
    return ExtractionField(values=values, axes=axes, **params)


#: Audits are expensive - one decomposition per surrogate per lens - and deterministic, so
#: the identical call is memoised rather than rerun. Keyed on the bytes of the array and on
#: every argument, so a test that changes any of them gets a fresh audit; `AuditReport` is
#: frozen, so a shared one cannot be mutated by whoever reads it first.
_AUDITS: dict = {}


def _audit(values=None, representations=SMALL_SET, n_surrogates=499, **kwargs):
    array = _correlated() if values is None else values
    key = (array.tobytes(), tuple(representations), int(n_surrogates),
           tuple(sorted(kwargs.items())))
    if key not in _AUDITS:
        _AUDITS[key] = audit_all(_field(array), representations=list(representations),
                                 n_surrogates=n_surrogates, seed=SEED, **kwargs)
    return _AUDITS[key]


def _clean_where_audited(report) -> bool:
    """`AuditReport.clean` minus its coverage clause.

    `clean` is deliberately false whenever a registered transform was not looked through, so
    a deliberately restricted audit can never be clean by that definition - which is the
    point of the clause and the reason it cannot be used to ask the narrower question these
    tests ask. Coverage gets its own tests, over the whole registry.
    """
    return bool(report.auditable) and all(a.clean for a in report.auditable)


# ==================================================================== 1. the criterion


def test_no_registered_representation_manufactures_a_feature_in_structureless_data():
    """The acceptance criterion, on the lenses a test suite can afford.

    A scale-free field has no localised structure at any scale, so every feature reported in
    every plane of every registered representation was made by the representation. The whole
    registry at once is `representation_null_field`; this is the same question of two lenses.
    """
    report = _audit()
    assert report.found == 0, report.findings()
    assert _clean_where_audited(report)


def test_the_audit_can_see_a_feature_that_is_really_there():
    """Without this, "it found nothing" is a statement about the audit, not about the field.

    The same two lenses, the same corrected level, the same ensemble size - and a blob. If
    the corrected family did not leave enough power to see a six-sigma peak through a wavelet
    band, the clean result above would be a machine for passing.
    """
    report = _audit(values=_blob())
    assert report.found > 0
    assert not _clean_where_audited(report)
    assert "raw" in {name for name, _plane, _k in report.findings()}


def test_the_planted_feature_is_reported_where_it_was_planted():
    report = _audit(values=_blob())
    found = report.audit("raw").plane("field").result[0]
    assert math.hypot(found.location.coords["row"] - 38.0,
                      found.location.coords["col"] - 25.0) < 1.5


def test_every_registered_transform_has_a_registered_representation():
    """Coverage is a property of the registry, not of what this module happened to write."""
    from src.transform_engine.registry import TRANSFORMS

    assert set(TRANSFORMS.names()) <= set(REPRESENTATIONS.names())


def test_a_transform_with_no_plane_builder_is_named_rather_than_skipped():
    """The failure mode the benchmark harness's three outcomes exist to prevent."""
    from src.transform_engine.registry import TRANSFORMS, TransformSpec

    state = snapshot(TRANSFORMS)
    try:
        # Not `_audit`: the two calls are identical in their arguments and differ only in
        # the state of a registry, which is exactly what its memo cannot see.
        def run():
            return audit_all(_field(_correlated()), representations=REPRESENTATIONS.names(),
                             n_surrogates=99, correction="none", seed=SEED)

        complete = run()
        assert complete.uncovered_transforms == ()
        TRANSFORMS.add("unaudited_lens", TransformSpec(
            apply=lambda field, config: {}, inverse=lambda c, f: f,
            summarise=lambda c: {}), description="A lens nobody wrote a plane builder for.")
        report = run()
        assert "unaudited_lens" in report.uncovered_transforms
        assert not report.clean, "an unlooked-through lens cannot leave the audit clean"
        assert _clean_where_audited(report), "and it is not a manufactured feature either"
    finally:
        restore(TRANSFORMS, state)


# ========================================================= 2. the null carries the lens


def test_rebuilding_the_null_inside_the_representation_manufactures_features():
    """The measurable difference the second strategy is registered for.

    Phase-randomising a coefficient plane preserves that plane's spectrum and scatters its
    localisation, so an artefact that sits in one place in every realisation is spread out in
    the null and concentrated in the observation - the signature of a discovery, on a field
    with nothing in it.

    Measured on the dual tree, because that is where the difference lives. The undecimated
    stationary transform is shift-invariant and on the parent grid, and rebuilding its null
    in the plane costs nothing measurable here; the dual tree is decimated, redundant and
    complex, and rebuilding its null manufactures features in the same field. A registry
    entry whose harm is invisible in the only place you looked has not been measured.
    """
    kwargs = dict(representations=["dtcwt"], correction="none", n_surrogates=199)
    propagated = _audit(**kwargs)
    rebuilt = _audit(strategy="randomise_in_representation", **kwargs)
    assert propagated.found == 0
    assert rebuilt.found > propagated.found
    assert rebuilt.findings()


def test_the_stationary_transform_is_where_the_two_strategies_agree():
    """Stated rather than hidden: the wrong null is not wrong everywhere by the same amount."""
    kwargs = dict(representations=["swt"], correction="none", n_surrogates=199)
    assert _audit(**kwargs).found == _audit(
        strategy="randomise_in_representation", **kwargs).found == 0


def test_the_propagated_null_of_the_identity_is_the_calibration_the_tree_already_uses():
    """`raw` is the identity, so propagating through it must reproduce `calibrate` exactly.

    This is what makes the default strategy a generalisation of the tree's existing null
    rather than a second opinion about it. Compared at the uncorrected level, because that is
    the one `calibrate` is written to.
    """
    values = _correlated()
    report = audit_all(_field(values), representations=["raw"], correction="none",
                       n_surrogates=99, seed=SEED)
    theirs = calibrate(values, alpha=0.05, n_surrogates=99, seed=SEED)
    mine = report.audit("raw").plane("field").calibration
    assert mine.threshold == pytest.approx(theirs.threshold, rel=0.0, abs=1e-12)
    assert sorted(mine.null_maxima) == pytest.approx(sorted(theirs.null_maxima))


def test_the_receipt_names_the_representation_the_null_was_pushed_through():
    report = _audit()
    calibration = report.audit("swt").plane("level_1/HH").calibration
    assert calibration.propagated_through == "swt/level_1/HH"
    assert "swt/level_1/HH" in calibration.describe()["null_hypothesis"]


def test_a_null_rebuilt_in_the_plane_does_not_claim_to_carry_the_representation():
    report = _audit(strategy="randomise_in_representation", correction="none")
    calibration = report.audit("swt").plane("level_1/HH").calibration
    assert calibration.propagated_through is None
    assert "swt" not in calibration.describe()["null_hypothesis"]


def test_the_strategies_declare_where_their_null_lives():
    carrying = NULL_STRATEGIES.with_capability("carries_the_representation", True)
    assert [entry.name for entry in carrying] == ["propagate_through_representation"]


def test_one_field_ensemble_serves_every_representation():
    """Same seed, same ensemble: auditing two lenses together must not move either null."""
    together = _audit(representations=("raw", "swt"))
    alone = _audit(representations=("swt",), n_surrogates=499)
    a = together.audit("swt").plane("level_2/HH").calibration.null_maxima
    b = alone.audit("swt").plane("level_2/HH").calibration.null_maxima
    assert list(a) == pytest.approx(list(b))


def test_an_unknown_strategy_names_the_registered_ones():
    with pytest.raises(UnknownNameError) as exc:
        _audit(strategy="propagate")
    assert "propagate_through_representation" in str(exc.value)


# ================================================== 3. a null that cannot be violated


def test_phase_randomisation_leaves_the_fft_magnitude_plane_untouched():
    """The degeneracy the vacuity check exists for, measured directly.

    `phase_randomise` preserves the amplitude spectrum to machine precision and the FFT
    magnitude plane *is* the amplitude spectrum. Every surrogate therefore has the same
    plane, so the null has no spread and the threshold is the observation.
    """
    from src.statistics import surrogates as surrogate_module

    values = _correlated()
    observed = np.abs(np.fft.rfft2(values))
    ensemble = surrogate_module.generate(values, method="phase_randomise", n=5, seed=SEED)
    for member in ensemble["members"]:
        assert np.abs(np.abs(np.fft.rfft2(member)) - observed).max() < 1e-8


def test_the_fft_magnitude_plane_is_reported_vacuous():
    report = audit_all(_field(_correlated()), representations=["raw", "fft"],
                       n_surrogates=99, seed=SEED)
    plane = report.audit("fft").plane("magnitude")
    assert plane.vacuous
    assert plane.relative_spread < VACUITY_TOLERANCE
    assert "fft/magnitude" in report.vacuous()


def test_a_wavelet_bands_null_moves_and_is_not_vacuous():
    report = _audit()
    plane = report.audit("swt").plane("level_1/HH")
    assert not plane.vacuous
    assert plane.relative_spread > 0.05, "a real ensemble spreads by tens of percent"


def test_a_vacuous_plane_could_not_be_counted_clean():
    """`clean` is `found == 0` *and* a null with the power to have said otherwise."""
    report = audit_all(_field(_correlated()), representations=["raw", "fft"],
                       n_surrogates=99, seed=SEED)
    plane = report.audit("fft").plane("magnitude")
    assert plane.found == 0
    assert not plane.clean


def test_a_representation_with_no_testable_plane_is_not_counted_as_a_pass():
    report = audit_all(_field(_correlated()), representations=["raw", "fft", "dct"],
                       n_surrogates=99, seed=SEED)
    assert {a.representation for a in report.unauditable} == {"fft", "dct"}
    assert not report.audit("fft").auditable
    assert _clean_where_audited(report), "the lenses that could be audited were, and were clean"


# ============================================== 4. the audit pays for its own family


def test_the_family_is_the_planes_that_are_actually_tested():
    report = _audit()
    testable = [p for a in report.audits for p in a.planes
                if p.plane.extractable and p.searchable_cells > 0]
    assert report.level.family_size == len(testable)


def test_a_frequency_plane_is_not_charged_to_the_family():
    """Charging the audit for a question nobody asked makes every other verdict stricter."""
    with_fft = audit_all(_field(_correlated()), representations=["raw", "fft"],
                         n_surrogates=99, seed=SEED)
    without = audit_all(_field(_correlated()), representations=["raw"],
                        n_surrogates=99, seed=SEED)
    assert with_fft.level.family_size == without.level.family_size == 1


def test_an_ensemble_too_small_for_the_family_is_refused_before_it_is_run():
    """Rule R18 applied to the audit itself: an unaffordable family is refused, not fudged."""
    with pytest.raises(FamilyTooLargeError) as exc:
        _audit(n_surrogates=99)
    message = str(exc.value)
    assert "family-wise level" in message
    assert exc.value.context["required_surrogates"] >= 20 * exc.value.context["family_size"]


def test_the_refusal_says_how_badly_the_undersized_ensemble_would_have_missed():
    with pytest.raises(FamilyTooLargeError) as exc:
        _audit(n_surrogates=99)
    assert exc.value.context["achievable_fwer"] > 0.05


def test_the_uncorrected_audit_rejects_far_more_often_than_it_claims():
    """The number that made the correction necessary rather than tidy.

    Twenty planes tested at a nominal family-wise 0.05 each reject on a large fraction of
    structureless fields, measured on the same ensemble the cuts come from.
    """
    report = _audit()
    uncorrected = report.level.notes["uncorrected_fwer"]
    assert uncorrected > 5.0 * report.level.alpha_family
    assert report.level.measured_fwer <= report.level.alpha_family


def test_max_statistic_is_less_conservative_than_bonferroni_when_planes_agree():
    """Six identical columns are one test wearing six hats, and only one correction knows.

    The extreme case, where the right answer can be written down: if every plane carries the
    same maximum, the family-wise rate *is* the single-plane rate, so the level should barely
    move. Bonferroni cannot know that and divides by six anyway.
    """
    column = np.random.default_rng(SEED).normal(size=(199, 1))
    identical = np.repeat(column, 6, axis=1)
    names = ["plane_%d" % i for i in range(6)]
    measured = FAMILY_CORRECTIONS.get("max_statistic")(identical, alpha=0.05, names=names)
    assumed = FAMILY_CORRECTIONS.get("bonferroni")(identical, alpha=0.05, names=names)
    assert measured.alpha_per_plane > 4.0 * assumed.alpha_per_plane
    assert measured.alpha_per_plane == pytest.approx(0.045, abs=0.005)
    assert measured.measured_fwer <= 0.05


def test_on_the_real_planes_the_two_corrections_land_within_one_rank_of_each_other():
    """The honest version of the same claim, on the planes the audit actually has.

    The bands of one stationary decomposition are dependent, but not nearly identically so,
    and the measured advantage over Bonferroni is smaller than the ensemble's own resolution
    of `1 / (1 + n)`. Asserted in that form rather than as "less conservative", which is what
    the dependent case suggests and what these planes do not show.
    """
    measured = _audit(correction="max_statistic")
    assumed = _audit(correction="bonferroni")
    step = 1.0 / (1.0 + measured.level.n_surrogates)
    assert measured.level.family_size == assumed.level.family_size
    assert abs(measured.level.alpha_per_plane - assumed.level.alpha_per_plane) <= step
    assert measured.level.measured_fwer is not None
    assert assumed.level.measured_fwer is None, "Bonferroni asserts its rate, it never measures it"


def test_only_the_measured_correction_puts_the_cut_on_an_achievable_p_value():
    """Every threshold stays an order statistic of its own null - section 3.6o's rule."""
    measured = _audit(correction="max_statistic").level
    resolution = 1.0 / (1.0 + measured.n_surrogates)
    assert measured.alpha_per_plane / resolution == pytest.approx(
        round(measured.alpha_per_plane / resolution), abs=1e-9)


def test_bonferroni_refuses_a_family_its_own_arithmetic_cannot_resolve():
    with pytest.raises(FamilyTooLargeError):
        _audit(correction="bonferroni", n_surrogates=99)


def test_no_correction_states_what_it_expects_to_find_in_nothing():
    report = _audit(correction="none")
    expected = report.level.notes["expected_findings_on_a_structureless_field"]
    assert expected == pytest.approx(0.05 * report.level.family_size)


def test_the_corrections_declare_whether_they_assume_independence():
    assumers = {e.name for e in FAMILY_CORRECTIONS.with_capability("assumes_independence", True)}
    assert assumers == {"bonferroni"}


def test_a_correction_registered_from_the_test_module_drives_the_audit():
    """The registry is open, and nothing in `src` has to be edited to prove it."""
    state = snapshot(FAMILY_CORRECTIONS)
    try:
        @FAMILY_CORRECTIONS.register(
            "refuse_everything", description="A cut nothing clears.",
            capabilities={"assumes_independence": False})
        def _paranoid(maxima, *, alpha, names):
            return CorrectedLevel(
                method="refuse_everything", alpha_family=alpha,
                alpha_per_plane=1.0 / (1.0 + maxima.shape[0]),
                family_size=int(maxima.shape[1]), n_surrogates=int(maxima.shape[0]))

        strictest = _audit(values=_blob(), correction="refuse_everything")
        uncorrected = _audit(values=_blob(), correction="none")
        assert strictest.level.method == "refuse_everything"
        assert strictest.level.alpha_per_plane < uncorrected.level.alpha_per_plane
        assert strictest.found <= uncorrected.found
        assert strictest.found > 0, (
            "and a six-sigma peak survives even the strictest cut the ensemble can express, "
            "which is what makes the clean results elsewhere mean something")
    finally:
        restore(FAMILY_CORRECTIONS, state)


def test_the_family_wise_rate_is_measured_leaving_each_surrogate_out():
    """One plane, three surrogates: each is the strict maximum of the other two exactly once.

    With a single plane the family-wise rate at rank 0 is the chance of being the largest of
    the others, which for `n` exchangeable draws is `1 / n` per draw and so `1` in total
    across the ensemble only if every draw wins - it does not. Written out on a matrix whose
    answer can be counted by hand.
    """
    maxima = np.array([[3.0], [2.0], [1.0]])
    curve = _fwer_curve(maxima)
    assert curve[0] == pytest.approx(1.0 / 3.0)     # only the 3.0 beats both others
    assert curve[1] == pytest.approx(2.0 / 3.0)     # 3.0 and 2.0 beat at least one
    assert curve[2] == pytest.approx(1.0)


def test_the_family_wise_rate_never_falls_as_the_cut_is_loosened():
    curve = _fwer_curve(np.random.default_rng(SEED).normal(size=(60, 4)))
    assert list(curve) == sorted(curve)


def test_two_planes_reject_more_often_than_one_at_the_same_rank():
    """The whole reason a family costs something."""
    rng = np.random.default_rng(SEED)
    one = _fwer_curve(rng.normal(size=(200, 1)))
    many = _fwer_curve(rng.normal(size=(200, 8)))
    assert many[0] > one[0]


# ======================================== 5. a plane declares what its indices mean


def test_a_plane_with_no_axes_must_say_why():
    with pytest.raises(InvalidParameterError) as exc:
        RepresentationPlane(name="mystery", values=np.zeros((4, 4)))
    assert "reason" in str(exc.value)


def test_a_plane_with_axes_may_not_also_carry_a_refusal():
    with pytest.raises(InvalidParameterError):
        RepresentationPlane(name="both", values=np.zeros((4, 4)), axes=AXES,
                            not_extractable="cannot be both")


def test_a_plane_must_be_two_dimensional():
    with pytest.raises(InvalidParameterError) as exc:
        RepresentationPlane(name="stack", values=np.zeros((4, 4, 6)), axes=AXES)
    assert "one plane per slice" in str(exc.value)


def test_a_plane_carrying_a_nan_is_refused_rather_than_maximised_over():
    values = np.zeros((4, 4))
    values[1, 1] = np.nan
    with pytest.raises(InvalidParameterError):
        RepresentationPlane(name="broken", values=values, axes=AXES)


def test_a_frequency_plane_is_refused_by_name_and_the_reason_is_the_axes():
    report = audit_all(_field(_correlated()), representations=["raw", "fft"],
                       n_surrogates=99, seed=SEED)
    plane = report.audit("fft").plane("magnitude")
    assert plane.refused is not None
    assert "wavenumbers" in plane.refused
    with pytest.raises(InvalidParameterError):
        plane_field(_field(_correlated()), "fft", plane.plane)


def test_the_fft_phase_is_not_offered_as_a_plane_at_all():
    """Its maximum is a property of the branch cut, so a null on it calibrates a convention."""
    planes = representation_planes("fft", _field(_correlated()))
    assert [p.name for p in planes] == ["magnitude"]
    assert "branch" in planes[0].notes["phase_omitted"]


def test_a_decimated_plane_reports_its_positions_in_parent_cells():
    field = _field(_correlated())
    plane = RepresentationPlane(name="coarse", values=np.zeros((16, 16)), axes=AXES,
                                parent_cells_per_sample=(4.0, 4.0))
    derived = plane_field(field, "test", plane)
    assert derived.spacing("row") == pytest.approx(4.0)
    assert derived.to_coordinate("row", 15.0) == pytest.approx(60.0)
    assert derived.axis_length("row") == pytest.approx(float(GRID_N))


def test_reading_a_decimated_plane_in_its_own_samples_is_wrong_by_the_decimation():
    """The error the parent-cell factor exists to prevent: three octaves, in the same units."""
    field = _field(_correlated())
    honest = plane_field(field, "test", RepresentationPlane(
        name="coarse", values=np.zeros((8, 8)), axes=AXES,
        parent_cells_per_sample=(8.0, 8.0)))
    naive = plane_field(field, "test", RepresentationPlane(
        name="coarse", values=np.zeros((8, 8)), axes=AXES,
        parent_cells_per_sample=(1.0, 1.0)))
    assert honest.to_coordinate("row", 5.0) == 8.0 * naive.to_coordinate("row", 5.0)


def test_a_declared_decimation_the_array_does_not_have_is_refused():
    """The defect the dual-tree lowpass had: eight declared, four measured, a position
    outside the frame it was found in."""
    state = snapshot(REPRESENTATIONS)
    try:
        @REPRESENTATIONS.register("overclaimed", description="Claims twice its decimation.")
        def _overclaimed(values, axes, config):
            return (RepresentationPlane(
                name="coarse", values=np.asarray(values)[::4, ::4], axes=axes,
                parent_cells_per_sample=(8.0, 8.0)),)

        with pytest.raises(InvalidParameterError) as exc:
            representation_planes("overclaimed", _field(_correlated()))
        assert "runs off the field" in str(exc.value)
    finally:
        restore(REPRESENTATIONS, state)


def test_the_dual_tree_lowpass_declares_the_decimation_it_actually_has():
    planes = {p.name: p for p in representation_planes("dtcwt", _field(_correlated()))}
    lowpass = planes["lowpass"]
    finest = planes["level_1/orientation_15"]
    assert lowpass.parent_cells_per_sample == decimation((GRID_N, GRID_N), lowpass.shape)
    assert finest.parent_cells_per_sample == (2.0, 2.0)
    assert lowpass.parent_cells_per_sample[0] != 2.0 ** 3


def test_a_plane_with_no_valid_interior_is_refused_rather_than_reported_clean():
    """Eight samples with a six-sample margin on each side is not a small search; it is none."""
    plane = RepresentationPlane(name="tiny", values=np.zeros((8, 8)), axes=AXES,
                                parent_cells_per_sample=(8.0, 8.0),
                                contaminated_halfwidth=6)
    assert searchable_cells(plane) == 0


def test_a_periodic_axis_has_no_contaminated_margin_to_lose():
    plane = RepresentationPlane(name="wrapped", values=np.zeros((8, 8)), axes=PERIODIC_AXES,
                                parent_cells_per_sample=(8.0, 8.0),
                                contaminated_halfwidth=6)
    assert searchable_cells(plane) == 64


def test_a_plane_with_no_interior_is_struck_from_the_family_not_counted_as_a_floor():
    state = snapshot(REPRESENTATIONS)
    try:
        @REPRESENTATIONS.register("blinkered", description="All margin, no interior.")
        def _blinkered(values, axes, config):
            arr = np.asarray(values)
            return (RepresentationPlane(name="field", values=arr, axes=axes),
                    RepresentationPlane(name="blind", values=arr, axes=axes,
                                        contaminated_halfwidth=arr.shape[0]))

        report = audit_all(_field(_correlated()), representations=["blinkered"],
                           n_surrogates=99, seed=SEED)
        blind = report.audit("blinkered").plane("blind")
        assert not blind.audited
        assert "no valid interior" in blind.refused
        assert report.level.family_size == 1
    finally:
        restore(REPRESENTATIONS, state)


def test_the_report_says_how_many_cells_were_actually_searched():
    """What stops "nothing was manufactured" being a statement about how little was looked at."""
    report = _audit()
    assert report.searched > 10_000
    assert report.describe()["searchable_cells"] == report.searched


# ===================================================== 6. the registry, and determinism


def test_a_representation_registered_from_the_test_module_is_audited():
    state = snapshot(REPRESENTATIONS)
    try:
        @REPRESENTATIONS.register(
            "gradient_magnitude", description="A lens registered outside src/.",
            capabilities={"decimated": False, "spatial_planes": True})
        def _gradient(values, axes, config):
            arr = np.asarray(values, dtype=np.float64)
            gy, gx = np.gradient(arr)
            return (RepresentationPlane(name="magnitude", values=np.hypot(gy, gx),
                                        axes=axes),)

        report = audit_all(_field(_correlated()), representations=["gradient_magnitude"],
                           n_surrogates=99, seed=SEED)
        assert report.audit("gradient_magnitude").plane("magnitude").audited
        assert report.found == 0
    finally:
        restore(REPRESENTATIONS, state)


def test_a_representation_that_produces_nothing_to_look_at_is_refused():
    state = snapshot(REPRESENTATIONS)
    try:
        REPRESENTATIONS.add("empty_lens", lambda values, axes, config: (),
                            description="Produces no planes.")
        with pytest.raises(InvalidParameterError) as exc:
            representation_planes("empty_lens", _field(_correlated()))
        assert "pass earned by not looking" in str(exc.value)
    finally:
        restore(REPRESENTATIONS, state)


def test_two_planes_of_one_representation_may_not_share_a_name():
    state = snapshot(REPRESENTATIONS)
    try:
        REPRESENTATIONS.add("twin_lens", lambda values, axes, config: (
            RepresentationPlane(name="band", values=np.asarray(values), axes=axes),
            RepresentationPlane(name="band", values=np.asarray(values), axes=axes)),
            description="Two planes, one name.")
        with pytest.raises(InvalidParameterError) as exc:
            representation_planes("twin_lens", _field(_correlated()))
        assert "attributed by name" in str(exc.value)
    finally:
        restore(REPRESENTATIONS, state)


def test_a_builder_that_renames_its_planes_between_calls_is_refused():
    """Standard E4. The null and the observation must be the same measurement."""
    state = snapshot(REPRESENTATIONS)
    calls = {"n": 0}
    try:
        @REPRESENTATIONS.register("drifting", description="Renames its plane each call.")
        def _drifting(values, axes, config):
            calls["n"] += 1
            return (RepresentationPlane(name="band_%d" % calls["n"],
                                        values=np.asarray(values), axes=axes),)

        with pytest.raises(InvalidParameterError) as exc:
            audit_all(_field(_correlated()), representations=["drifting"],
                      n_surrogates=99, seed=SEED)
        assert "not comparable" in str(exc.value)
    finally:
        restore(REPRESENTATIONS, state)


def test_an_unknown_representation_names_the_registered_ones():
    with pytest.raises(UnknownNameError) as exc:
        representation_planes("wavelet", _field(_correlated()))
    assert "swt" in str(exc.value)


def test_the_audit_is_reproducible_at_the_same_seed():
    """Two genuine runs, not two reads of one - `_audit`'s memo would answer this for free."""
    def run():
        return audit_all(_field(_correlated()), representations=list(SMALL_SET),
                         n_surrogates=199, seed=SEED, correction="none")

    first, second = run(), run()
    assert first is not second
    assert (first.audit("swt").plane("level_1/HH").calibration.threshold
            == second.audit("swt").plane("level_1/HH").calibration.threshold)
    assert first.level.alpha_per_plane == second.level.alpha_per_plane


def test_a_different_seed_moves_the_null_without_changing_the_verdict():
    field = _field(_correlated())
    a = audit_all(field, representations=list(SMALL_SET), n_surrogates=499, seed=SEED)
    b = audit_all(field, representations=list(SMALL_SET), n_surrogates=499, seed=SEED + 1)
    assert (a.audit("swt").plane("level_1/HH").calibration.threshold
            != b.audit("swt").plane("level_1/HH").calibration.threshold)
    assert a.found == b.found == 0


# ============================================================== 7. what lands on a record


def test_a_feature_found_in_a_band_is_labelled_with_that_band():
    """R8's requirement: a manufactured feature must be attributable to what manufactured it."""
    report = _audit(values=_blob())
    for name, plane, _count in report.findings():
        for feature in report.audit(name).plane(plane).result:
            assert feature.representation == "%s/%s" % (name, plane)


def test_features_from_two_bands_cannot_be_pooled_into_one_set():
    report = _audit(values=_blob())
    findings = [report.audit(n).plane(p).result[0] for n, p, _k in report.findings()]
    if len(findings) < 2:
        pytest.skip("this field produced a finding in only one plane")
    with pytest.raises(InvalidParameterError) as exc:
        FeatureSet(findings[:2])
    assert "representation" in str(exc.value)


def test_the_record_carries_the_configuration_the_lens_was_run_with():
    report = _audit(values=_blob())
    plane = report.audit("swt").plane("level_1/HH")
    provenance = plane.result.field.provenance
    assert provenance["parent_representation"] == "identity"
    assert provenance["parent_shape"] == [GRID_N, GRID_N]
    assert "representation_config" in provenance


def test_the_report_describes_the_level_the_family_and_the_gaps():
    described = _audit().describe()
    for key in ("representations", "uncovered_transforms", "planes_audited",
                "planes_refused", "searchable_cells", "found", "findings", "unauditable",
                "vacuous", "level", "clean"):
        assert key in described
    assert described["level"]["correction"] == "max_statistic"


def test_the_default_ensemble_is_too_small_for_the_whole_registry_and_says_so():
    """The honest price of the audit, asserted so it cannot drift into a silent default."""
    assert DEFAULT_AUDIT_SURROGATES < 45 / 0.05
    with pytest.raises(FamilyTooLargeError):
        audit_all(_field(_correlated()), n_surrogates=DEFAULT_AUDIT_SURROGATES, seed=SEED)
