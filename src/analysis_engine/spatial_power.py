"""Spatial sampling adequacy for a cross-scale gate (T4C.5i, fixes D84).

**Why this module exists.** R13's edge exclusion is exact and is enforced elsewhere: coefficients
within one filter support of a crop boundary are contaminated and are never analysed. What R13
*also* carries is a crop-size floor -- ``MIN_VALID_INTERIOR = 128`` px per side, rounded up to a
power of two -- and that floor is a judgement, not a derivation. Its own comment says so. The
rounding is justified as dyadic tidiness, and the SWT is undecimated, so it has no dyadic size
requirement at all.

The constant is standing in for a real quantity that nothing measured. ``transfer_entropy``
consumes 1-D series; ``scale_signature`` collapses space first, as
``energy_density[t, s] = sum(coefficient**2) / values.size`` over the valid interior. The joint
histogram's samples are therefore **frames, not pixels**, and the estimator's temporal power is
already checked against ``MIN_SAMPLES_PER_CELL``. The valid interior does something different: it
sets how many independent spatial structures contribute to each per-frame scalar.

That makes crop size beyond edge exclusion a **power** criterion rather than a validity one, and
the direction is favourable. Too few independent structures makes the per-frame energy density a
noisy summary of the region, which attenuates a dependence estimate toward zero. An undersized
crop therefore biases toward the null: it cannot forge a PASS, but it can forge a FAIL that is
really *"the instrument could not have seen it"*. The frozen T4C.6 decision rule already
distinguishes those -- *"an adequately powered absence is FAIL; any ... power failure is
INVALID"* -- but nothing derived the term that separates them.

**What this module measures, and what it deliberately does not.** It measures the spatial
decorrelation length of each scale's valid interior and converts it to an effective sample size.
It does not model attenuation analytically. A reliability formula would rest on a
signal/sampling-noise split of the across-frame variance that cannot be verified from the data,
and an unverifiable correction to a power claim is worse than none. The attenuation itself is
established empirically by nested sub-cropping in a later step of T4C.5i; this module supplies
the geometry that step needs, plus the one refusal that requires no model at all.

**The model-free refusal.** If a scale's interior does not decorrelate anywhere within itself,
then it holds on the order of a single independent structure, its per-frame energy density is a
summary of that one structure rather than of a population, and no crop-size constant is needed to
know the sample is inadequate. That is reported as saturation and is the derived counterpart of
the constant it replaces.

Everything here is computed on the generate/train partition only. Spending the confirmatory
partition to decide whether the instrument is adequate would spend it twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidParameterError


#: Autocorrelation below which two samples count as approximately independent. The same 1/e
#: convention `cross_scale.decorrelation_frames` uses on the time axis; stated once here so the
#: spatial and temporal windows cannot drift apart silently.
DECORRELATION_THRESHOLD = 1.0 / math.e

#: Minimum paired samples before a lag's autocorrelation is trusted. Mirrors the temporal
#: estimator's refusal to read a correlation off fewer than four pairs.
MIN_PAIRS = 4

#: Fraction of the interior within which a decorrelation length is believable. Centring a window
#: on its own mean forces the sample autocorrelation to fall away at large lags whether or not
#: the field decorrelates -- the short-series bias -- so a "length" found beyond this horizon is
#: indistinguishable from that artefact. Searching to half the interior, as the temporal
#: estimator can afford to over thousands of frames, would manufacture exactly the reassuring
#: number this task exists to remove. Beyond the horizon the honest report is saturation.
TRUST_HORIZON_FRACTION = 0.25

#: Effective spatial samples below which a scale is reported as inadequate regardless of its
#: pixel count. One independent structure cannot support a claim about a population of them;
#: this is a floor on arithmetic meaning, not a substitute judgement for the constant removed.
MIN_EFFECTIVE_SAMPLES = 2.0

AXIS_NAMES: Tuple[str, str] = ("row", "column")


def _axis_decorrelation(centred: np.ndarray, axis: int) -> Dict[str, Any]:
    """First lag along ``axis`` whose mean autocorrelation falls below 1/e.

    The correlation at each lag is pooled over the perpendicular axis, so a 139x139 interior
    contributes 139 lines to every estimate rather than one. The search stops at
    ``TRUST_HORIZON_FRACTION`` of the interior: a mean-centred window's autocorrelation decays
    at long lags by construction, so a crossing found out there measures the window, not the
    field. Saturation -- no trustworthy crossing -- is returned as a fact rather than as a
    number, because a length the crop cannot contain has not been measured.
    """
    size = centred.shape[axis]
    limit = int(size * TRUST_HORIZON_FRACTION)
    if limit < 1:
        return {"axis": AXIS_NAMES[axis], "decorrelation_px": None, "saturated": True,
                "searched_to_px": 0,
                "reason": "the interior is too small along this axis to test any lag"}
    moved = np.moveaxis(centred, axis, 0)
    for lag in range(1, limit + 1):
        left = moved[:-lag]
        right = moved[lag:]
        if left.size < MIN_PAIRS:
            break
        denominator = math.sqrt(float(np.sum(left ** 2)) * float(np.sum(right ** 2)))
        if denominator <= 0:
            # A constant band has no structure to decorrelate from; one pixel is as
            # informative as all of them.
            return {"axis": AXIS_NAMES[axis], "decorrelation_px": 1, "saturated": False,
                    "searched_to_px": lag,
                    "reason": "the interior is constant along this axis"}
        acf = float(np.sum(left * right)) / denominator
        if abs(acf) < DECORRELATION_THRESHOLD:
            return {"axis": AXIS_NAMES[axis], "decorrelation_px": int(lag), "saturated": False,
                    "searched_to_px": int(lag), "autocorrelation_at_lag": acf}
    return {"axis": AXIS_NAMES[axis], "decorrelation_px": None, "saturated": True,
            "searched_to_px": int(limit),
            "reason": ("the autocorrelation stays above %.3f out to %d px, the furthest lag a "
                       "%d px interior can report without measuring its own mean-centring, "
                       "so the decorrelation length is longer than this crop can measure"
                       % (DECORRELATION_THRESHOLD, limit, size))}


def spatial_decorrelation(band: Any) -> Dict[str, Any]:
    """Per-axis spatial decorrelation length of one 2-D valid interior, in pixels.

    The band is centred on its own mean, exactly as the temporal estimator centres its series.
    No detrending is applied: a large-scale gradient genuinely does mean the field decorrelates
    slowly, and removing it here would flatter the sample count by discarding the structure that
    makes neighbouring pixels dependent.
    """
    values = np.asarray(band, dtype=np.float64)
    if values.ndim != 2:
        raise InvalidParameterError("band", getattr(values, "shape", None),
                                    "a 2-D valid-interior array")
    if not np.all(np.isfinite(values)):
        raise InvalidParameterError(
            "band", "non-finite values",
            "a fully finite interior; a masked or padded interior is not a measured one")
    if values.size < MIN_PAIRS:
        raise InvalidParameterError("band", values.shape,
                                    "an interior with at least %d samples" % MIN_PAIRS)
    centred = values - float(values.mean())
    axes = [_axis_decorrelation(centred, 0), _axis_decorrelation(centred, 1)]
    saturated = any(row["saturated"] for row in axes)
    return {
        "interior_shape": [int(values.shape[0]), int(values.shape[1])],
        "axes": axes,
        "saturated": saturated,
        "threshold": DECORRELATION_THRESHOLD,
        "basis": ("first lag whose autocorrelation, pooled over the perpendicular axis, falls "
                  "below 1/e; the same convention the Theiler window uses on the time axis"),
    }


def effective_spatial_samples(decorrelation: Mapping[str, Any]) -> Dict[str, Any]:
    """Independent structures in one interior: area divided by decorrelation area.

    A saturated axis yields no count. Substituting the searched limit would silently convert
    "longer than we could measure" into "exactly as long as we looked", which is the optimistic
    direction and precisely the assumption this task exists to remove.
    """
    axes = list(decorrelation["axes"])
    shape = list(decorrelation["interior_shape"])
    if decorrelation["saturated"]:
        blocking = [row["axis"] for row in axes if row["saturated"]]
        return {
            "effective_samples": None,
            "adequate": False,
            "saturated_axes": blocking,
            "reason": ("the %s axis does not decorrelate within the interior, so this crop holds "
                       "on the order of one independent structure at this scale and its "
                       "per-frame summary describes that structure rather than a population "
                       "of them" % " and ".join(blocking)),
        }
    lengths = [int(row["decorrelation_px"]) for row in axes]
    per_axis = [size / length for size, length in zip(shape, lengths)]
    ess = float(per_axis[0] * per_axis[1])
    return {
        "effective_samples": ess,
        "independent_lines_per_axis": per_axis,
        "decorrelation_px_per_axis": lengths,
        "adequate": ess >= MIN_EFFECTIVE_SAMPLES,
        "minimum_effective_samples": MIN_EFFECTIVE_SAMPLES,
        "reason": ("%.1f independent structures across a %dx%d interior at %dx%d px "
                   "decorrelation" % (ess, shape[0], shape[1], lengths[0], lengths[1])),
    }


def assess_interior(band: Any) -> Dict[str, Any]:
    """One interior's decorrelation and effective sample size, as a single record."""
    decorrelation = spatial_decorrelation(band)
    return {"decorrelation": decorrelation,
            "effective_samples": effective_spatial_samples(decorrelation)}


# ====================================================================== attenuation

#: Sub-crops are taken at these fractions of the valid interior, concentric so that every crop
#: shares a centre and differs only in how much of the same field it averages. Nested crops are
#: strongly dependent, which is intentional: the curve is a within-field trend, not a set of
#: independent measurements, and nothing here performs inference *across* sizes.
DEFAULT_SIZE_FRACTIONS: Tuple[float, ...] = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0)

#: Points used for the extrapolation, taken from the largest crops. The linearisation below is a
#: weak-dependence approximation and visibly fails under severe attenuation, so the fit is made
#: where attenuation is mildest -- which is also the only region the extrapolation is asked
#: about.
EXTRAPOLATION_POINTS = 4

#: Coefficient of determination below which the extrapolation is not reported as a number. A
#: badly fitting model must produce a refusal, not a confident intercept.
MIN_EXTRAPOLATION_FIT = 0.8


def centred_subcrop(bands: Any, size: int) -> np.ndarray:
    """The concentric ``size x size`` window of a ``(T, H, W)`` stack."""
    values = np.asarray(bands, dtype=np.float64)
    if values.ndim != 3:
        raise InvalidParameterError("bands", getattr(values, "shape", None),
                                    "a (frames, height, width) stack of valid interiors")
    height, width = values.shape[1], values.shape[2]
    if size < 1 or size > min(height, width):
        raise InvalidParameterError("size", size,
                                    "a window no larger than the interior (%d)"
                                    % min(height, width))
    top = (height - size) // 2
    left = (width - size) // 2
    return values[:, top:top + size, left:left + size]


def energy_density_series(bands: Any) -> np.ndarray:
    """Per-frame energy density: the same mean of squares ``scale_signature`` computes."""
    values = np.asarray(bands, dtype=np.float64)
    if values.ndim != 3:
        raise InvalidParameterError("bands", getattr(values, "shape", None),
                                    "a (frames, height, width) stack of valid interiors")
    return (values ** 2).reshape(values.shape[0], -1).mean(axis=1)


def _median_effective_samples(bands: np.ndarray, *, frames: int = 16) -> Optional[float]:
    """Effective samples for a stack, as the median over an evenly spaced frame subsample.

    One frame's decorrelation length is itself a noisy estimate; the median over a spread of
    frames is stable without pretending the field is stationary. A stack whose *median* frame
    saturates has no count, for the same reason a single saturated interior has none.
    """
    count = bands.shape[0]
    indices = np.unique(np.linspace(0, count - 1, min(frames, count)).astype(int))
    measured = []
    for index in indices:
        assessment = effective_spatial_samples(spatial_decorrelation(bands[index]))
        measured.append(assessment["effective_samples"])
    present = [value for value in measured if value is not None]
    if len(present) * 2 <= len(measured):
        return None
    return float(np.median(present))


def attenuation_curve(source_bands: Any, target_bands: Any, *, lag: int,
                      bins: int = 6,
                      size_fractions: Sequence[float] = DEFAULT_SIZE_FRACTIONS) -> Dict[str, Any]:
    """Transfer entropy as a function of how much of the same interior is averaged.

    Only the spatial averaging changes across rows. The frame count, the bin count, the lag and
    the field itself are identical at every size, so the joint histogram's sample count is
    constant and the usual small-sample entropy bias is common to every row. Whatever the curve
    shows is therefore a property of spatial precision, not of estimator sample size -- which is
    what makes it readable as attenuation at all.
    """
    from src.analysis_engine.cross_scale import transfer_entropy

    source = np.asarray(source_bands, dtype=np.float64)
    target = np.asarray(target_bands, dtype=np.float64)
    if source.shape != target.shape:
        raise InvalidParameterError("target_bands", target.shape,
                                    "the same (frames, height, width) shape as source_bands")
    interior = int(min(source.shape[1], source.shape[2]))
    sizes = sorted({max(4, int(round(interior * fraction))) for fraction in size_fractions})
    rows = []
    for size in sizes:
        source_crop = centred_subcrop(source, size)
        target_crop = centred_subcrop(target, size)
        rows.append({
            "interior_px": size,
            "fraction_of_interior": size / interior,
            "effective_samples": _median_effective_samples(source_crop),
            "transfer_entropy_nats": float(transfer_entropy(
                energy_density_series(source_crop),
                energy_density_series(target_crop), lag, bins=bins)),
        })
    return {"lag": int(lag), "bins": int(bins), "full_interior_px": interior, "curve": rows,
            "basis": ("concentric sub-crops of one interior; frames, bins and lag identical at "
                      "every size, so only spatial precision varies")}


def extrapolate_attenuation(curve: Mapping[str, Any]) -> Dict[str, Any]:
    """Estimate the transfer entropy an unlimited crop would have measured.

    **The model, stated so it can be disagreed with.** Averaging over ``E`` independent spatial
    structures leaves the per-frame scalar with sampling variance proportional to ``1/E``. For
    weak dependence a transfer entropy behaves like a squared correlation, and a squared
    correlation is attenuated by a reliability factor ``1 / (1 + c/E)``. So
    ``TE(E) = TE_inf / (1 + c/E)``, and therefore ``1/TE`` is **linear in ``1/E``** with intercept
    ``1/TE_inf``. The intercept is the quantity of interest: the effect an infinite crop would
    have seen.

    The linearisation is a weak-dependence approximation and it visibly breaks under severe
    attenuation, so it is fitted over the largest crops only and its fit quality is reported. A
    fit below ``MIN_EXTRAPOLATION_FIT`` yields no number -- a poorly determined intercept
    presented as a power correction would be worse than admitting the curve is unreadable.
    """
    rows = [row for row in curve["curve"]
            if row["effective_samples"] is not None and row["transfer_entropy_nats"] > 0]
    if len(rows) < 3:
        return {"extrapolated_transfer_entropy_nats": None, "fit_r_squared": None,
                "reason": ("fewer than three crops produced both a positive transfer entropy and "
                           "a measurable effective sample size, so no trend can be read")}
    rows = sorted(rows, key=lambda row: row["effective_samples"])[-EXTRAPOLATION_POINTS:]
    x = np.array([1.0 / row["effective_samples"] for row in rows])
    y = np.array([1.0 / row["transfer_entropy_nats"] for row in rows])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = y - y.mean()
    r_squared = float(1.0 - (residual ** 2).sum() / (total ** 2).sum()) \
        if float((total ** 2).sum()) > 0 else 0.0
    if intercept <= 0:
        # A non-positive intercept says the fitted line reaches zero at a *finite* sample count,
        # so the curve is still climbing steeply and no plateau is in view. That is not a fit
        # problem -- the fit may be excellent -- it is the strongest available statement that
        # the crop is inadequate, and it must not be softened into a number.
        return {"extrapolated_transfer_entropy_nats": None, "fit_r_squared": r_squared,
                "fitted_points": len(rows), "plateau_in_view": False,
                "reason": ("transfer entropy is still rising steeply at the largest crop and the "
                           "attenuation model reaches no ceiling within a finite sample count "
                           "(R^2=%.3f), so this interior is nowhere near resolving the effect "
                           "and no unlimited-crop estimate can be named" % r_squared)}
    if r_squared < MIN_EXTRAPOLATION_FIT:
        return {"extrapolated_transfer_entropy_nats": None, "fit_r_squared": r_squared,
                "fitted_points": len(rows), "plateau_in_view": None,
                "reason": ("1/TE against 1/ESS does not fit the attenuation model well enough "
                           "(R^2=%.3f) to name the effect an unlimited crop would see. The model "
                           "is a weak-dependence approximation and does not hold near the "
                           "log(bins) entropy ceiling" % r_squared)}
    full = max(curve["curve"], key=lambda row: row["interior_px"])
    extrapolated = float(1.0 / intercept)
    measured = float(full["transfer_entropy_nats"])
    return {
        "extrapolated_transfer_entropy_nats": extrapolated,
        "measured_transfer_entropy_nats": measured,
        "retained_fraction": (measured / extrapolated) if extrapolated > 0 else None,
        "fit_r_squared": r_squared,
        "fitted_points": len(rows),
        "plateau_in_view": True,
        "model": "TE(E) = TE_inf / (1 + c/E); 1/TE linear in 1/E, intercept 1/TE_inf",
    }


def power_verdict(curve: Mapping[str, Any], *, detection_threshold_nats: float) -> Dict[str, Any]:
    """Decide whether an absence at the full crop is FAIL or INVALID.

    The question is never "has the curve plateaued" in the abstract; it is whether a larger crop
    would plausibly have changed the verdict. So the measured and extrapolated effects are
    compared against the same detection threshold the study will use:

    *   both below it -- the effect is absent whatever the crop, and an absence is an
        **adequately powered FAIL**;
    *   both above it -- the crop already resolves the effect, and the verdict stands on its own;
    *   they straddle it -- an unlimited crop would have detected what this one cannot, so the
        result is **INVALID for inadequate power**, not a negative finding.

    This spends no threshold of its own. The only constant it needs is the study's, which was
    frozen before any of this ran.
    """
    if not isinstance(detection_threshold_nats, (int, float)) \
            or isinstance(detection_threshold_nats, bool) or detection_threshold_nats <= 0:
        raise InvalidParameterError("detection_threshold_nats", detection_threshold_nats,
                                    "a positive transfer entropy in nats")
    extrapolation = extrapolate_attenuation(curve)
    full = max(curve["curve"], key=lambda row: row["interior_px"])
    measured = float(full["transfer_entropy_nats"])
    extrapolated = extrapolation["extrapolated_transfer_entropy_nats"]
    threshold = float(detection_threshold_nats)

    if extrapolated is None:
        return {"verdict": "INVALID", "reason":
                "the attenuation curve could not be read, so the crop cannot be shown adequate: "
                + str(extrapolation.get("reason", "")),
                "measured_transfer_entropy_nats": measured,
                "detection_threshold_nats": threshold, "extrapolation": extrapolation}
    if measured < threshold <= extrapolated:
        return {"verdict": "INVALID", "reason":
                ("an unlimited crop would have measured %.4f nats against a %.4f detection "
                 "threshold, while this interior measures %.4f. The absence is a property of the "
                 "crop, not of the atmosphere." % (extrapolated, threshold, measured)),
                "measured_transfer_entropy_nats": measured,
                "extrapolated_transfer_entropy_nats": extrapolated,
                "detection_threshold_nats": threshold, "extrapolation": extrapolation}
    return {"verdict": "ADEQUATE", "reason":
            ("the measured %.4f nats and the unlimited-crop estimate %.4f nats fall on the same "
             "side of the %.4f detection threshold, so a larger crop would not change the "
             "verdict" % (measured, extrapolated, threshold)),
            "measured_transfer_entropy_nats": measured,
            "extrapolated_transfer_entropy_nats": extrapolated,
            "retained_fraction": extrapolation.get("retained_fraction"),
            "detection_threshold_nats": threshold, "extrapolation": extrapolation}


# ====================================================== minimum detectable effect (T4C.5i step 4)

#: Multiplicity procedures whose rank-1 rejection threshold this module knows how to derive. The
#: campaign uses `benjamini_yekutieli`; the others are here so a study that declared a different
#: correction gets a derived threshold rather than silently getting BY's.
DETECTION_PROCEDURES = ("bonferroni", "holm", "benjamini_hochberg", "benjamini_yekutieli")


def detection_rank(*, n_tests: int, n_surrogates: int, alpha: float = 0.05,
                   correction: str = "benjamini_yekutieli") -> Dict[str, Any]:
    """How far into the null ensemble a statistic may fall and still be declared significant.

    A surrogate p-value is ``(1 + k) / (1 + n)`` where ``k`` counts surrogates at least as
    extreme as the observation, so the design's whole detection ability reduces to one integer:
    the largest ``k`` that still clears the strictest corrected threshold in the family.

    The strictest threshold is the one at **rank 1**, which is the honest case to plan for: a
    study looking for a single real effect among ``n_tests`` cannot rely on the laxer thresholds
    a step-up procedure grants only once several tests are already rejected. Under
    Benjamini-Yekutieli that threshold is ``alpha / (m * H_m)``, with ``H_m`` the harmonic sum
    that buys validity under arbitrary dependence -- and arbitrary dependence is the right
    assumption here, since the tests share scales, share frames and share a record.

    A negative answer is not an edge case to smooth over. It means no effect of any magnitude
    could have been declared significant, and it is returned as such.
    """
    if correction not in DETECTION_PROCEDURES:
        raise InvalidParameterError("correction", correction,
                                    "one of %s" % (list(DETECTION_PROCEDURES),))
    if not isinstance(n_tests, int) or isinstance(n_tests, bool) or n_tests < 1:
        raise InvalidParameterError("n_tests", n_tests, "a positive number of declared tests")
    if not isinstance(n_surrogates, int) or isinstance(n_surrogates, bool) or n_surrogates < 1:
        raise InvalidParameterError("n_surrogates", n_surrogates, "a positive ensemble size")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise InvalidParameterError("alpha", alpha, "a significance level in (0, 1)")

    penalty = 1.0
    if correction == "benjamini_yekutieli":
        penalty = float(np.sum(1.0 / np.arange(1, n_tests + 1)))
    required_p = alpha / (n_tests * penalty)
    # (1 + k) / (1 + n) <= required_p, solved for the largest integer k.
    max_exceedances = int(math.floor(required_p * (1 + n_surrogates) - 1.0 + 1e-12))
    record = {
        "n_tests": int(n_tests),
        "n_surrogates": int(n_surrogates),
        "alpha": float(alpha),
        "correction": correction,
        "dependence_penalty": float(penalty),
        "required_raw_p": float(required_p),
        "p_value_floor": 1.0 / (1.0 + n_surrogates),
        "max_exceedances": max_exceedances,
        "can_detect": max_exceedances >= 0,
    }
    if max_exceedances < 0:
        record["reason"] = (
            "%d surrogates floor the p-value at %.3g, but rank 1 of a %d-test family under %s "
            "at alpha=%.3g needs a raw p at or below %.3g. No effect of any magnitude could "
            "have been declared significant, so this design has no minimum detectable effect -- "
            "it has no detection at all."
            % (n_surrogates, record["p_value_floor"], n_tests, correction, alpha, required_p))
    return record


def minimum_detectable_effect(null_nats: Any, *, n_tests: int, alpha: float = 0.05,
                              correction: str = "benjamini_yekutieli") -> Dict[str, Any]:
    """The smallest transfer entropy this design could have declared significant, in nats.

    **Read off the measured null, not off a distribution.** ``detection_rank`` says the observed
    statistic may be matched by at most ``k`` surrogates, so it must exceed the ``k + 1``-th
    largest value the ensemble actually produced. With the campaign's 4,999 shifts and its
    36-test family that rank is 1: the observation has to beat **every** surrogate. No normality,
    no variance estimate and no tail extrapolation enters -- the threshold is an order statistic
    of the same circular-shift ensemble the gate itself is referenced against, so it carries the
    estimator's small-sample entropy bias in exactly the way the observation does.

    **This is a boundary, not an estimate, and that distinction is what the preregistration buys.**
    The surrogate seed is declared in the campaign before any data is seen, so the ensemble is
    frozen: the ``k + 1``-th largest of *that* ensemble is the literal decision boundary of the
    exact test the gate will run. An observation above it is significant and one below it is not.
    No confidence interval belongs on it, because the counterfactual an interval would describe
    -- the same study drawn with a different seed -- is precisely what preregistering the seed
    exists to rule out. ``(1 + k) / (1 + n)`` is likewise exactly valid under exchangeability and
    needs no correction of its own.

    Two earlier versions of this function attached uncertainty to the number anyway, and both were
    wrong in ways worth recording. A Clopper-Pearson bound on the threshold's exceedance
    probability compared two quantities that can never meet, since the bound is about
    ``(k + 1 + z*sqrt(k)) / n`` against a required level of about ``(k + 1) / n``. A bootstrap of
    the order statistic was then miscalibrated in the dangerous direction: at rank 1 a resample
    can never exceed the sample maximum, so the interval was one-sided by construction, and four
    independent ensembles of 4,999 draws all landed above its upper limit. Raising the rank to 10
    did not repair it -- coverage was 3 in 20. The fault was never the estimator; it was asking a
    frozen design a question about a study that will not be run.

    The ensemble must be the one measured on the crop and partition under test. A null drawn from
    a larger crop has a different spread, and borrowing it would hide the very attenuation this
    task exists to expose.
    """
    values = np.asarray(null_nats, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise InvalidParameterError("null_nats", int(finite.size),
                                    "at least two finite surrogate statistics")
    rank = detection_rank(n_tests=n_tests, n_surrogates=int(finite.size), alpha=alpha,
                          correction=correction)
    record: Dict[str, Any] = {
        "detection_rank": rank,
        "null_mean_nats": float(finite.mean()),
        "null_max_nats": float(finite.max()),
        "n_finite_surrogates": int(finite.size),
        "n_discarded_surrogates": int(values.size - finite.size),
    }
    if not rank["can_detect"]:
        record.update({"minimum_detectable_effect_nats": None, "reason": rank["reason"]})
        return record

    ordered = np.sort(finite)[::-1]
    index = rank["max_exceedances"]          # 0-based index of the (k + 1)-th largest
    threshold = float(ordered[index])
    record.update({
        "minimum_detectable_effect_nats": threshold,
        "minimum_detectable_excess_nats": threshold - float(finite.mean()),
        "order_statistic_rank": index + 1,
        "basis": ("the %d-th largest of %d circular-shift surrogates: the observation must "
                  "exceed it to leave at most %d exceedance(s) and clear rank 1 of a %d-test "
                  "family under %s at alpha=%.3g"
                  % (index + 1, finite.size, rank["max_exceedances"], rank["n_tests"],
                     rank["correction"], rank["alpha"])),
    })
    return record


__all__ = [
    "AXIS_NAMES",
    "DECORRELATION_THRESHOLD",
    "DETECTION_PROCEDURES",
    "detection_rank",
    "minimum_detectable_effect",
    "DEFAULT_SIZE_FRACTIONS",
    "EXTRAPOLATION_POINTS",
    "MIN_EXTRAPOLATION_FIT",
    "attenuation_curve",
    "centred_subcrop",
    "energy_density_series",
    "extrapolate_attenuation",
    "power_verdict",
    "MIN_EFFECTIVE_SAMPLES",
    "MIN_PAIRS",
    "TRUST_HORIZON_FRACTION",
    "assess_interior",
    "effective_spatial_samples",
    "spatial_decorrelation",
]
