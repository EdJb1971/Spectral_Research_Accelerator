"""Phase 4C, T4C.2: surrogate nulls for a whole record.

The roadmap's acceptance criteria are here - the surrogate PSD matches the source while the
phase correlation is destroyed, an organised field scores and a fractional Brownian field does
not - and so are the two calibrations that decide whether any of it means anything:

*   choosing the wrong null (`per_frame_phase`) makes an ordinary red-noise record test as
    organised, because the null has no autocorrelation and the data does;
*   computing a lagged statistic on the record as a *line* against a circularly stationary
    null rejects **every time**, on data where the null is true by construction.

Both are measured here rather than asserted anywhere.
"""

import math

import numpy as np
import pytest
import torch

from src.analysis_engine.surrogate_null import (DESTROYS, PRESERVES, SEQUENCE_METHODS,
                                                SurrogateNull, SurrogateNullError,
                                                sequence_surrogate, verify_surrogate)
from src.core.errors import InvalidParameterError
from src.physical_core.field import PhysicalField
from src.physical_core.sequence import FieldSequence
from src.statistics.surrogates import phase_randomise


def _red_sequence(n=16, size=16, phi=0.9, seed=100):
    """An AR(1) record: no organisation at all, and heavily autocorrelated.

    The single most important null in this file. Atmospheric fields are strongly red, so a
    test that mistakes autocorrelation for structure will do so on almost every real record.
    """
    generator = torch.Generator().manual_seed(seed)
    frames = [torch.randn(size, size, dtype=torch.float64, generator=generator)]
    innovation = math.sqrt(1.0 - phi ** 2)
    for _ in range(n - 1):
        frames.append(phi * frames[-1] + innovation * torch.randn(
            size, size, dtype=torch.float64, generator=generator))
    return FieldSequence([PhysicalField(f) for f in frames],
                         np.arange(n, dtype=np.float64) * 3600.0)


def _lag_one(circular):
    def statistic(sequence):
        stack = sequence.to_tensor().numpy().reshape(len(sequence), -1)
        if circular:
            return float(np.corrcoef(stack.ravel(), np.roll(stack, -1, axis=0).ravel())[0, 1])
        return float(np.corrcoef(stack[:-1].ravel(), stack[1:].ravel())[0, 1])
    return statistic


# ============================================================ the stated acceptance criteria

def test_the_spatiotemporal_surrogate_preserves_the_whole_spectrum():
    source = _red_sequence()
    report = verify_surrogate(source, sequence_surrogate(source, "spatiotemporal_phase",
                                                         seed=1))
    assert report["psd_relative_error"] < 1e-12
    assert report["max_abs_frame_correlation"] < 0.3


def test_the_per_frame_surrogate_preserves_each_frame_and_not_the_record():
    source = _red_sequence()
    report = verify_surrogate(source, sequence_surrogate(source, "per_frame_phase", seed=1))
    assert report["max_per_frame_psd_relative_error"] < 1e-12
    # The 3D spectrum is not preserved, which is exactly the temporal structure being lost.
    assert report["psd_relative_error"] > 0.5


def test_the_circular_shift_keeps_every_frame_exactly():
    source = _red_sequence()
    report = verify_surrogate(source, sequence_surrogate(source, "circular_time_shift",
                                                         seed=1))
    assert report["psd_relative_error"] == pytest.approx(0.0, abs=1e-12)


def test_phase_randomisation_now_works_in_three_dimensions():
    """The generator was restricted to 1D and 2D; the arithmetic never was."""
    data = np.random.default_rng(4).standard_normal((6, 8, 8))
    surrogate = phase_randomise(data, seed=77)
    assert surrogate.shape == data.shape
    assert np.allclose(np.abs(np.fft.fftn(surrogate)), np.abs(np.fft.fftn(data)))
    assert np.isrealobj(surrogate)
    assert not np.allclose(surrogate, data)


# ============================================================ the two calibrations

def test_the_wrong_null_turns_autocorrelation_into_organisation():
    """Rule R12, measured. The data is AR(1) noise: there is nothing to find.

    Against `per_frame_phase` the null's lag-one autocorrelation is essentially zero while the
    record's is 0.9, so the *autocorrelation itself* beats the null. Against
    `spatiotemporal_phase` the null carries the same autocorrelation and the comparison is
    about organisation, which is what was asked.
    """
    source = _red_sequence()
    per_frame = SurrogateNull(source, "per_frame_phase", n=99, seed=7).test(_lag_one(True))
    spatiotemporal = SurrogateNull(source, "spatiotemporal_phase", n=99,
                                   seed=7).test(_lag_one(True))

    assert abs(per_frame["surrogate_mean"]) < 0.1
    assert spatiotemporal["surrogate_mean"] > 0.7
    assert per_frame["observed"] == pytest.approx(spatiotemporal["observed"])
    assert any("temporal autocorrelation" in w for w in per_frame["warnings"])
    assert spatiotemporal["warnings"] == []


def test_a_linear_lag_against_a_circular_null_rejects_every_time():
    """The reason `cross_scale` wraps its lags, established by counting.

    Ten independent AR(1) records, on which the phase-randomised null is true by
    construction. A lag statistic computed on the record as a line rejects all ten; the same
    statistic computed circularly rejects none.
    """
    linear = circular = 0
    for trial in range(10):
        source = _red_sequence(seed=100 + trial)
        null = SurrogateNull(source, "spatiotemporal_phase", n=99, seed=1000 + trial)
        linear += null.test(_lag_one(False), alternative="two_sided")["p_value"] <= 0.05
        circular += null.test(_lag_one(True), alternative="two_sided")["p_value"] <= 0.05
    assert linear == 10
    assert circular == 0


# ============================================================ organised versus scale-free

def _concentration(sequence):
    """Mean over frames of the Gini coefficient of level-2 detail energy.

    Rises when energy gathers into localised structures and does not when it is spread by a
    scale-free process, which is the distinction the test needs.
    """
    from src.analysis_engine.scale_signature import gini
    from src.transform_engine import stationary as swt

    values = []
    for frame in sequence.fields:
        coefficients = swt.apply_swt2d(frame, levels=2, wavelet="db2")
        detail = torch.cat([coefficients["level_2"][band].flatten()
                            for band in ("LH", "HL", "HH")])
        values.append(gini((detail.to(torch.float64) ** 2).numpy()))
    return float(np.mean(values))


def _blob_sequence(n_frames=4, size=64, seed=3):
    """Localised Gaussian blobs on noise: organisation, by construction."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    fields = []
    for _ in range(n_frames):
        frame = 0.2 * rng.standard_normal((size, size))
        for _ in range(4):
            cy, cx = rng.uniform(12, size - 12, size=2)
            frame += 6.0 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 3.0 ** 2))
        fields.append(PhysicalField(torch.as_tensor(frame, dtype=torch.float64)))
    return FieldSequence(fields, np.arange(n_frames, dtype=np.float64) * 3600.0)


def _fbm_sequence(n_frames=4, size=64, hurst=0.7, seed=3):
    from src.benchmarks.fields import build_fbm
    from src.benchmarks.seeding import derive
    fields = [build_fbm(derive("fbm_sequence_%d" % index, seed), n=size, hurst=hurst)
              for index in range(n_frames)]
    return FieldSequence([PhysicalField(f.data) for f in fields],
                         np.arange(n_frames, dtype=np.float64) * 3600.0)


def test_an_organised_record_scores_and_a_scale_free_one_does_not():
    """T4C.2's acceptance criterion, both halves in one test so neither can drift alone."""
    organised = SurrogateNull(_blob_sequence(), "spatiotemporal_phase", n=99,
                              seed=11).test(_concentration, label="concentration")
    scale_free = SurrogateNull(_fbm_sequence(), "spatiotemporal_phase", n=99,
                               seed=11).test(_concentration, label="concentration")

    assert organised["p_value"] <= 0.05
    assert organised["effect_size"] > 3.0
    assert scale_free["p_value"] > 0.05


# ============================================================ refusals and reporting

def test_an_unknown_method_names_the_alternatives():
    with pytest.raises(InvalidParameterError) as excinfo:
        sequence_surrogate(_red_sequence(), "aaft")
    assert "src.statistics.surrogates" in str(excinfo.value)


def test_a_circular_shift_needs_enough_frames():
    short = _red_sequence(n=2)
    with pytest.raises(SurrogateNullError):
        sequence_surrogate(short, "circular_time_shift", seed=1)


def test_an_empty_ensemble_is_refused():
    with pytest.raises(InvalidParameterError):
        SurrogateNull(_red_sequence(), n=0)


def test_every_method_declares_what_it_preserves_and_destroys():
    for method in SEQUENCE_METHODS:
        assert PRESERVES[method] and DESTROYS[method]


def test_the_result_carries_the_null_identity():
    result = SurrogateNull(_red_sequence(), n=19, seed=5).test(_lag_one(True))
    assert result["surrogate_method"] == "spatiotemporal_phase"
    assert result["p_value_floor"] == pytest.approx(0.05)
    assert "temporal autocorrelation" in " ".join(result["null_preserves"])
    assert result["null_hypothesis"].startswith("the observed statistic is no more extreme")


def test_the_summary_is_lineage_safe():
    import json
    summary = SurrogateNull(_red_sequence(), n=19, seed=5).summary()
    assert json.loads(json.dumps(summary))["n_surrogates"] == 19
    assert summary["p_value_floor"] == pytest.approx(0.05)


def test_the_time_axis_survives_the_surrogate():
    source = _red_sequence()
    surrogate = sequence_surrogate(source, "spatiotemporal_phase", seed=2)
    assert np.array_equal(surrogate.times_seconds, source.times_seconds)
