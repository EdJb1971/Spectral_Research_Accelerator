"""TG17.15 slice 4: what the pool-substitution null does when nothing is there.

Slice 3 produced exact ranks and said, in its own claim boundary, that an exact rank is not a
calibration. This module is the measurement that boundary named. It exists because of one result
in this repository's history: T4C.5h's surrogate preserved *exactly* the property its method was
named after and still measured a family-wise false-positive rate of 0.765 against a nominal 0.05.
Arithmetic being right is not the same as a null being the null it claims, and the only way to
tell the two apart is to run the method on records where the answer is known to be "nothing".

**What is measured, and against what.** Four things, on records built here rather than borrowed:

*   A **false-positive rate on a true null** -- left members and observed partners drawn
    independently, so no correspondence exists to find. This is the T4C.5h test.
*   The **shape of the p-value distribution**, not only its tail. A rate can look nominal while
    the distribution is wrong; under exchangeability the observation's rank among its `N` admitted
    alternatives is uniform on `{1, ..., N + 1}` *exactly*, so the whole distribution is predicted
    in advance and can be compared rather than only its 5% tail.
*   **Detection at declared effect sizes**, so power is a measured curve rather than the single
    planted case that only ever demonstrates the easiest question. `EFFECT_WEIGHTS` is not a round
    set of numbers: it was chosen after a coarse sweep found the curve falls off a cliff between
    0.92 and 0.84, and it brackets that cliff instead of straddling it.
*   Two **adversarial nulls**, each aimed at a way the pool contract could be necessary and not
    sufficient: a shared sampling grid every record carries, and an observed partner drawn
    systematically cleaner than the alternatives its own bands admit.

**Why a rate is reported as an interval and judged on its bound.** "Zero false positives in twenty
runs" is not a measured rate of zero; it is consistent with a true rate of 14%. Every rate here
carries a Clopper-Pearson interval and every acceptance is decided on that interval's *upper*
bound, so a run too small to certify a rate reports that it cannot rather than passing quietly.
That is the difference between margin measured and margin assumed, and it is why
`CalibrationOutcome.certifies_rate` is a separate field from `within_expectation`.

**Why these records are built here and not borrowed from `shape_fixtures`.** Those fixtures build
*pairings* for TG17.11's inventory-reassignment null, where a member's alternatives are the other
pairings' right members and no record needs declared marginals because nothing is admitted. The
pool null admits on marginals, so every record here must carry a `RecordProfile` that is a
consequence of how the record was actually built: `cadence_seconds` is `native_seconds` divided by
the rows per cycle the record really has, and `n_samples` is the number of rows it really carries.
A profile attached to a record afterwards would make admission a fiction and the calibration
worthless.

**The dependence the numbers are read under.** One candidate inventory serves every member of a
family, so the members' pools overlap heavily and their p-values are dependent. That is deliberate
-- it is the case Benjamini-Yekutieli was chosen for -- and it means the `family_size` values from
one realisation are not independent draws. Every statistic that requires independence is therefore
computed on **one member per realisation**, and the pooled version is reported beside it as a
diagnostic that is explicitly not a test.

**Why a rank is needed at all, in one number.** `shape_recurrence` returns a magnitude, and the
magnitude of the weighted correlation between two *independent* smooth profiles is not near zero:
measured over these fixtures it averages about 0.26 and reaches 0.73. A threshold on the statistic
would therefore be a threshold on how many harmonics a profile happens to carry. The planted weight
`w` is recovered faithfully where it is above that floor -- 1.00 measures 0.95, 0.80 measures 0.78,
0.50 measures 0.51 -- and is swamped below it, which is the whole argument for ranking an
observation against a declared pool rather than scoring it against a number.

**What a low detection rate here is, and is not.** Where detection falls, this module reports the
fraction of members that reached their own pool's floor. A member sitting at `1/(N + 1)` and still
not rejecting was not short of effect -- it was short of pool. That single number separates "the
correspondence is not there" from "this pool cannot resolve it", and it is what makes a
disappointing row actionable rather than merely disappointing.

Nothing in this module is evidence about the world. It measures whether a declared method behaves
as declared on records whose answers were fixed before the method ran.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.benchmarks.shape_calibration import (
    ShapeComparisonRefusal, ShapeRecord, shape_recurrence,
)
from src.core.correspondence_estimand import ALPHA, CORRECTION
from src.core.errors import InvalidParameterError
from src.core.partner_pool import (
    AdmissionContract, PartnerPool, PoolAdmissionRefusal, RecordProfile, build_partner_pool,
)
from src.core.pool_substitution_null import CorrespondenceFamily, correspondence_family


CALIBRATION_SCHEMA = "pool-substitution-calibration/v1"
DETECTION_SCHEMA = "pool-substitution-detection/v1"

#: The statistic under calibration and the tail its values put the evidence in. `shape_recurrence`
#: returns a magnitude, so larger is more similar. Declared here rather than inferred, for the
#: reason `require_declared_orientation` has no default.
ORIENTATION = "larger_is_more_similar"

#: Every fixture record spans this many native cycles, so the phase window is declared and equal
#: for every comparison rather than inferred from whichever record happens to be shorter.
SPAN_CYCLES = 8.0

#: Native durations across four orders of magnitude, a quarter-hour to four months. This is the
#: range scale/shape mode exists for, and `AdmissionContract` deliberately refuses to band it.
NATIVE_SECONDS_RANGE = (9.0e2, 1.0e7)

#: Rows per native cycle. The floor is well above `shape_calibration.MINIMUM_ROWS_PER_CYCLE`, so
#: no result here can be an artefact of a record too coarse to carry a shape at all.
ROWS_PER_CYCLE_RANGE = (12.0, 40.0)

#: Dimensionless residual scale added to each standardized record.
NOISE_FLOOR_RANGE = (0.05, 0.20)

#: The declared bands. `native_seconds` is absent because `AdmissionContract` refuses to band it,
#: and that has a consequence this module measures rather than assumes: `cadence_seconds` is
#: `native_seconds / rows_per_cycle`, so banding cadence over a bounded row density *transitively*
#: narrows the native durations one pool can contain even though nothing bands them directly.
#: `admission_yield` reports how far.
CALIBRATION_CONTRACT = AdmissionContract(ratio_bands={
    "n_samples": 2.0,
    "effective_sample_size": 2.0,
    "noise_floor": 3.0,
    "cadence_seconds": 4.0,
})

CALIBRATION_FAMILY_SIZE = 6

#: Candidates offered per realisation. Sized from the measured admission yield rather than
#: guessed: the bands keep a minority of a natively diverse inventory, and a pool that fails to
#: resolve is refused outright, so a calibration whose realisations kept being refused would be
#: measuring its own inventory size instead of its null.
CANDIDATES_OFFERED = 1500

#: Weights of the shared shape in the observed partner. The partner traces
#: `w * shared + sqrt(1 - w^2) * own` with both components standardized first, so `w` is the
#: correlation the planting puts there before noise -- a declared effect size, not a label.
#:
#: The grid is dense between 0.80 and 0.95 and sparse elsewhere because a coarse sweep found the
#: detection curve is not gradual: it holds at 1.00 and collapses by 0.84. A ladder spaced evenly
#: over [0, 1] would have reported two flat ends and missed the only interesting interval.
EFFECT_WEIGHTS: Tuple[float, ...] = (1.0, 0.95, 0.92, 0.88, 0.84, 0.80, 0.60, 0.0)

#: Realisations for a rate that is meant to be certified.
#:
#: The count the *rate* needs is derived rather than chosen: `REALISATIONS_FOR_ALPHA` below solves
#: for the smallest run whose zero-count bound clears alpha, and the answer is 59. This number is
#: larger, and the reason is the second measurement rather than the first. The distribution check
#: resolves a departure of about `1.36 / sqrt(n)`: at 59 that is 0.18, blunt enough that only a
#: T4C.5h-scale error would show, and at 200 it is 0.096. A run sized for the tail alone would
#: certify the rate and be unable to see the shape, which is the pair of mistakes this slice is
#: about. Smaller runs are allowed and report `certifies_rate = False` rather than a pass.
DEFAULT_REALISATIONS = 200

#: Realisations per rung of the effect ladder, where the quantity is a detection rate rather than
#: an error rate that has to clear alpha. A coarse detection curve is informative; a coarse error
#: rate is not, which is why the two counts differ.
DEFAULT_LADDER_REALISATIONS = 40

CONFIDENCE = 0.95

#: How many leading realisations of a case are digested as a reproduction witness.
#:
#: `calibrate_case` runs realisation `index` at `seed + index`, so a run of `k` realisations at a
#: given seed *is* the first `k` realisations of any longer run at that seed. A short run is
#: therefore not a similar measurement to the recorded one -- it is a prefix of it, exactly, and
#: comparing digests either agrees or does not.
#:
#: This is what makes a 22-minute recording checkable by a test suite. The calendar calibration is
#: cheap enough to re-run whole on every pass and is; this one is not, and the honest substitute
#: is not a weaker version of the same claim but a different and smaller one: that the code path
#: the recording names still produces, realisation for realisation, what it produced then. Three
#: is enough for that, because a digest over eighteen p-values agrees by coincidence or not at all.
WITNESS_REALISATIONS = 3


# --------------------------------------------------------------------------- rates as intervals


def clopper_pearson(successes: int, trials: int, *, confidence: float = CONFIDENCE) \
        -> Tuple[float, float]:
    """The exact binomial interval, so a zero count is reported as a bound and not as a rate.

    Exact rather than normal-approximate because every interesting count here is at or near zero,
    where the normal approximation returns the degenerate interval `[0, 0]` and would present
    "we saw none" as "the rate is none".
    """
    trials = int(trials)
    successes = int(successes)
    if trials < 0 or successes < 0 or successes > trials:
        raise InvalidParameterError(
            "successes", successes, "a count between 0 and the number of trials (%d)" % trials)
    if trials == 0:
        return (0.0, 1.0)
    from scipy import stats as sps  # declared dependency; see requirements.txt

    tail = (1.0 - float(confidence)) / 2.0
    lower = 0.0 if successes == 0 else float(sps.beta.ppf(tail, successes, trials - successes + 1))
    upper = (1.0 if successes == trials
             else float(sps.beta.ppf(1.0 - tail, successes + 1, trials - successes)))
    return (lower, upper)


def one_sided_upper(successes: int, trials: int, *, confidence: float = CONFIDENCE) -> float:
    """The one-sided Clopper-Pearson upper bound, which is the bound the claim actually needs.

    "The false-positive rate is at or below alpha" is a one-directional claim, so the number that
    supports it is a one-sided bound at the declared confidence. Reading the upper end of a
    two-sided 95% interval instead would be a 97.5% bound announced as a 95% one -- conservative,
    and still a statement whose confidence does not match its arithmetic. The two-sided interval
    is reported alongside because it describes where the rate lies; this is the one acceptance
    reads, and the two are kept as separate fields so neither can quietly stand in for the other.
    """
    trials, successes = int(trials), int(successes)
    if trials == 0:
        return 1.0
    if successes == trials:
        return 1.0
    from scipy import stats as sps  # declared dependency; see requirements.txt

    return float(sps.beta.ppf(float(confidence), successes + 1, trials - successes))


def one_sided_lower(successes: int, trials: int, *, confidence: float = CONFIDENCE) -> float:
    """The one-sided Clopper-Pearson lower bound, for a claim that points the other way.

    A detection claim is "the rate is at least this", so the number supporting it is a lower bound
    at the declared confidence -- the mirror of `one_sided_upper`, and needed for the same reason.
    An expectation written as "the point estimate reaches 1.0" is not a stricter version of this:
    it is a demand that the method never make a type-II error, which no test satisfies and which
    one unlucky draw in a thousand is enough to refute.
    """
    trials, successes = int(trials), int(successes)
    if trials == 0 or successes == 0:
        return 0.0
    from scipy import stats as sps  # declared dependency; see requirements.txt

    return float(sps.beta.ppf(1.0 - float(confidence), successes, trials - successes + 1))


def attains(measured: Mapping[str, Any], floor: float) -> bool:
    """Whether the lower bound clears a declared floor. The detection counterpart of `certifies`."""
    return bool(measured["trials"]) and float(measured["one_sided_lower"]) >= float(floor)


def realisations_to_certify(*, ceiling: float = ALPHA, confidence: float = CONFIDENCE,
                            limit: int = 100000) -> int:
    """Fewest realisations whose zero-count bound clears the ceiling. Solved, not written down.

    A run smaller than this cannot certify its own error rate however clean its result looks, and
    the number is a property of the ceiling and the confidence together -- so it is computed
    against them, the way `minimum_pool_size` is computed against the real correction rather than
    asserted beside it.
    """
    for count in range(1, int(limit) + 1):
        if one_sided_upper(0, count, confidence=confidence) <= float(ceiling):
            return count
    raise InvalidParameterError(  # pragma: no cover - found long before the limit
        "limit", limit, "a search bound large enough to contain a certifiable run size")


#: The smallest run that can certify a family-wise error rate at alpha: 59 realisations.
#: `DEFAULT_REALISATIONS` is larger, for the reason stated there.
REALISATIONS_FOR_ALPHA = realisations_to_certify()


def ks_resolution(n: int) -> float:
    """The departure from uniform a Kolmogorov-Smirnov check can see at this many draws.

    The 5% critical value, `1.36 / sqrt(n)`. Reported because a distribution check that cannot
    resolve the error it is looking for is a check in name only, and the number that says so
    belongs next to the run size rather than in a reader's head.
    """
    return 1.36 / math.sqrt(max(int(n), 1))


def rate(successes: int, trials: int, *, confidence: float = CONFIDENCE) -> Dict[str, Any]:
    """A count as what it is: a point, the interval it stands for, and the bound a claim reads."""
    lower, upper = clopper_pearson(successes, trials, confidence=confidence)
    return {"successes": int(successes), "trials": int(trials),
            "point": (float(successes) / trials) if trials else None,
            "lower": lower, "upper": upper,
            "one_sided_upper": one_sided_upper(successes, trials, confidence=confidence),
            "one_sided_lower": one_sided_lower(successes, trials, confidence=confidence),
            "confidence": float(confidence)}


def certifies(measured: Mapping[str, Any], ceiling: float) -> bool:
    """Whether the *bound* clears the ceiling, which is the only version of this that is true.

    A point estimate of zero says nothing about a rate; a one-sided Clopper-Pearson upper bound
    below the ceiling says the rate is below it at the stated confidence. Everything in this
    module that claims an error rate is under alpha claims it through this function.
    """
    return bool(measured["trials"]) and float(measured["one_sided_upper"]) <= float(ceiling)


# ------------------------------------------------------------------------------ building records


@dataclass(frozen=True)
class FixtureRecord:
    """One record, and the declared marginals that actually follow from how it was built."""

    shape: ShapeRecord
    profile: RecordProfile

    @property
    def record_id(self) -> str:
        return self.profile.record_id


def _harmonics(rng: np.random.Generator, count: int = 3) -> Tuple[Tuple[float, float], ...]:
    return tuple((float(rng.uniform(0.5, 1.5)), float(rng.uniform(0.0, 2.0 * np.pi)))
                 for _ in range(count))


def _profile_values(phase: np.ndarray,
                    harmonics: Sequence[Tuple[float, float]]) -> np.ndarray:
    value = np.zeros_like(phase)
    for order, (amplitude, offset) in enumerate(harmonics):
        value = value + amplitude * np.sin(2.0 * np.pi * (order + 1) * phase + offset)
    return value


def build_record(record_id: str, provenance_key: str, rng: np.random.Generator, *,
                 harmonics: Optional[Sequence[Tuple[float, float]]] = None,
                 shared: Optional[Sequence[Tuple[float, float]]] = None,
                 weight: float = 0.0,
                 noise_floor_range: Tuple[float, float] = NOISE_FLOOR_RANGE,
                 grid_artefact: bool = False) -> FixtureRecord:
    """One record on its own native scale, with a profile computed from the record it describes.

    `weight` is the declared effect size: the record traces `w * shared + sqrt(1 - w^2) * own`
    with both components standardized first, so `w` is the correlation planted before noise rather
    than an amplitude whose meaning would depend on which harmonics happened to be drawn.

    The rng is consumed in a fixed order whatever the weight, so two realisations built from the
    same seed at different effect sizes differ in the planted correlation and in nothing else.
    """
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise InvalidParameterError(
            "weight", weight,
            "a planted correlation in [0, 1]. It is the effect size itself, so a value outside "
            "the range a correlation can take would not name an effect")
    native_seconds = float(np.exp(rng.uniform(*np.log(NATIVE_SECONDS_RANGE))))
    rows_per_cycle = float(rng.uniform(*ROWS_PER_CYCLE_RANGE))
    rows = int(round(SPAN_CYCLES * rows_per_cycle))
    cadence_seconds = native_seconds / rows_per_cycle

    edges = np.linspace(0.0, SPAN_CYCLES * native_seconds, rows + 1)
    starts, ends = edges[:-1], edges[1:]
    phase = ((starts + ends) / 2.0) / native_seconds

    own = _profile_values(phase, harmonics if harmonics is not None else _harmonics(rng))
    own = own / own.std()
    if shared is not None and weight > 0.0:
        planted = _profile_values(phase, shared)
        own = weight * (planted / planted.std()) + math.sqrt(1.0 - weight ** 2) * own
    if grid_artefact:
        period = max(int(round(rows_per_cycle)), 2)
        within = (np.arange(rows) % period) / float(period)
        own = own + 3.0 * np.sign(np.sin(2.0 * np.pi * 5.0 * within))
    values = (own - own.mean()) / own.std()
    noise = float(rng.uniform(*noise_floor_range))
    values = values + noise * rng.standard_normal(rows)

    return FixtureRecord(
        shape=ShapeRecord(label=record_id, starts=starts, ends=ends, values=values,
                          valid=np.ones(rows, dtype=bool), native_seconds=native_seconds),
        profile=RecordProfile(
            record_id=record_id, provenance_key=provenance_key, n_samples=rows,
            effective_sample_size=rows / float(rng.uniform(1.5, 3.0)),
            native_seconds=native_seconds, cadence_seconds=cadence_seconds,
            coverage_fraction=float(rng.uniform(0.9, 1.0)), noise_floor=noise))


# ----------------------------------------------------------------------------------- the cases


#: Each case is a way the world could be, with the answer fixed by what the case *is* rather than
#: by what it was measured to do. `weight` is the planted correlation; the two adversarial nulls
#: plant nothing and differ only in how the observed partner relates to the pool that will stand
#: in for it.
CASES: Dict[str, Dict[str, Any]] = {
    "planted_correspondence": {
        "weight": 1.0,
        "statement": ("left and observed partner trace one shape at unrelated native durations, "
                      "and the pool's alternatives do not")},
    "no_correspondence": {
        "weight": 0.0,
        "statement": ("left and observed partner are independent, so the partner is exchangeable "
                      "with its pool by construction and every rejection is a false one")},
    "shared_grid_alias": {
        "weight": 0.0, "grid_artefact": True,
        "statement": ("every record carries the same strong artefact keyed to position within "
                      "its own cycle: a shared instrument cadence, which is not a shared shape. "
                      "The pool carries it too, so the null must absorb it")},
    "clean_partner_noisy_pool": {
        "weight": 0.0, "clean_partner": True,
        "statement": ("the observed partner is drawn systematically cleaner than the inventory "
                      "while still passing every declared band, so the alternatives its own "
                      "contract admits are noisier than it is. Nothing corresponds; the question "
                      "is whether passing the bands was enough")},
    "unresolvable_inventory": {
        "weight": 0.0, "candidates_offered": 40,
        "statement": ("too few candidates for any pool to reach the size %d correspondences "
                      "need, so the honest answer is a refusal rather than a p-value that could "
                      "not have rejected" % CALIBRATION_FAMILY_SIZE)},
}

#: Noise range for the observed partner in `clean_partner_noisy_pool`. It sits inside the
#: inventory's own range, so the partner is an ordinary record the contract admits without
#: complaint. The point of the case is that "admissible" and "exchangeable" are not the same
#: statement, and slice 2's claim boundary says so in words; this asks it for a number.
CLEAN_PARTNER_NOISE = (0.05, 0.07)


def build_realisation(case: str, seed: int, *, family_size: int = CALIBRATION_FAMILY_SIZE,
                      candidates_offered: Optional[int] = None,
                      weight: Optional[float] = None) \
        -> Tuple[Tuple[PartnerPool, ...], Dict[str, ShapeRecord], Dict[str, Any]]:
    """One family of correspondences and the single inventory every one of them draws from.

    `weight` overrides the case's own effect size and is how the detection ladder is built: the
    same seed produces the same inventory, the same left members and the same marginals at every
    rung, so a rung differs from its neighbour by the planted correlation and by nothing else.
    """
    if case not in CASES:
        raise InvalidParameterError("case", case, "one of %s" % sorted(CASES))
    spec = CASES[case]
    offered = int(candidates_offered if candidates_offered is not None
                  else spec.get("candidates_offered", CANDIDATES_OFFERED))
    planted = float(spec["weight"] if weight is None else weight)
    rng = np.random.default_rng(int(seed))

    records: Dict[str, ShapeRecord] = {}
    candidates: List[RecordProfile] = []
    for index in range(offered):
        built = build_record("cand_%05d" % index, "src_cand_%05d" % index, rng,
                             grid_artefact=bool(spec.get("grid_artefact")))
        records[built.record_id] = built.shape
        candidates.append(built.profile)

    pools: List[PartnerPool] = []
    for member in range(family_size):
        shared = _harmonics(rng)
        left = build_record("left_%d" % member, "src_left_%d" % member, rng, harmonics=shared,
                            grid_artefact=bool(spec.get("grid_artefact")))
        partner = build_record(
            "partner_%d" % member, "src_partner_%d" % member, rng, shared=shared, weight=planted,
            grid_artefact=bool(spec.get("grid_artefact")),
            noise_floor_range=(CLEAN_PARTNER_NOISE if spec.get("clean_partner")
                               else NOISE_FLOOR_RANGE))
        records[left.record_id] = left.shape
        records[partner.record_id] = partner.shape
        pools.append(build_partner_pool(
            left=left.profile, observed_partner=partner.profile, candidates=candidates,
            contract=CALIBRATION_CONTRACT, estimand="per_correspondence",
            tested_correspondences=family_size))

    return tuple(pools), records, {"case": case, "seed": int(seed), "offered": offered,
                                   "weight": planted}


def _statistic(left: ShapeRecord, right: ShapeRecord) -> float:
    """The declared statistic at the declared phase window, with the window checked not assumed.

    Passing `cycles` explicitly is on its own decoration here, and mutation testing said so: with
    the window capped at each record's own declared support by `_shared_phase`, declaring eight
    cycles and inferring the shorter record's span give the same number for every fixture this
    module builds, because every fixture spans exactly eight. What is *not* decoration is
    requiring that to be true. A fixture that silently got shorter would still return a
    comparison, over a window nobody declared and nobody would see; refusing it turns the
    declaration into something that can fail.
    """
    for record in (left, right):
        if record.phase_span < SPAN_CYCLES - 1e-9:
            raise ShapeComparisonRefusal(
                "phase_span", "%s: %.4f" % (record.label, record.phase_span),
                "a record covering the declared window of %.1f native cycles. Comparing over "
                "less would silently narrow the window the calibration reports its numbers for"
                % SPAN_CYCLES)
    return shape_recurrence(left, right, cycles=SPAN_CYCLES)


def run_realisation(case: str, seed: int, *, family_size: int = CALIBRATION_FAMILY_SIZE,
                    candidates_offered: Optional[int] = None, weight: Optional[float] = None,
                    alpha: float = ALPHA,
                    correction: str = CORRECTION) -> CorrespondenceFamily:
    """Build one realisation and run it through the real null, with no shortcut around it."""
    pools, records, _ = build_realisation(case, seed, family_size=family_size,
                                          candidates_offered=candidates_offered, weight=weight)
    return correspondence_family(pools=pools, records=records, statistic=_statistic,
                                 orientation=ORIENTATION, alpha=alpha, correction=correction)


# ------------------------------------------------------------- the distribution, not just a rate


def rank_uniform(p_value: float, reference_size: int) -> float:
    """Map an exact lattice p-value onto (0, 1) so pools of different sizes can be pooled at all.

    Under exchangeability the observation's rank among its `N` alternatives is uniform on
    `{1, ..., N + 1}` exactly, and `p = rank / (N + 1)`. `(rank - 0.5) / (N + 1)` is the lattice's
    mid-point transform, uniform on (0, 1) to within one half-step. That half-step is the
    transform's own error, which is why the pool sizes are reported beside every uniformity figure
    rather than left out of it: a statistic computed over pools of 58 carries a coarser lattice
    than one computed over pools of 440.
    """
    rank = round(float(p_value) * int(reference_size))
    return (rank - 0.5) / float(reference_size)


def uniformity(values: Sequence[float]) -> Dict[str, Any]:
    """How far a set of transformed ranks departs from the uniform distribution it must have.

    The Kolmogorov-Smirnov p-value here is meaningful only when the values are independent, which
    is why `_case_statistics` computes the certified figure from one member per realisation and
    reports the pooled figure separately as a diagnostic rather than as a second test.
    """
    from scipy import stats as sps  # declared dependency; see requirements.txt

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "expected_mean": 0.5, "standard_error_of_mean": None,
                "ks_statistic": None, "ks_p_value": None}
    test = sps.kstest(array, "uniform")
    return {"n": int(array.size), "mean": float(array.mean()), "expected_mean": 0.5,
            "standard_error_of_mean": float(math.sqrt(1.0 / 12.0 / array.size)),
            "ks_statistic": float(test.statistic), "ks_p_value": float(test.pvalue)}


# ------------------------------------------------------------------------------ the calibration


#: What each case must do, stated from what the case is rather than from what it was seen to do.
#: The two adversarial nulls carry the same ceiling as the plain one: an artefact everyone shares
#: and a partner cleaner than its own pool are both worlds in which nothing corresponds, so both
#: must hold the declared rate or the pool contract is necessary-and-not-sufficient in a way that
#: matters rather than only in principle.
CASE_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    # Two criteria, each about a different thing, and each derived from what the case is rather
    # than from what a perfect run would look like.
    #
    # The first asks whether the *statistic* works: a genuinely shared shape should put the true
    # partner top of its own pool. It is judged on a lower confidence bound, not on a point, for
    # the same reason an error rate is judged on an upper one.
    #
    # The second asks whether the *correction* wastes what the statistic found: a member the
    # statistic ranked first that still does not reject was short of pool, and that is the failure
    # worth catching. It is parameter-free.
    #
    # This replaces `minimum_detection: 1.0`, which the first recorded run failed at 1199 of 1200.
    # That expectation was wrong rather than unlucky: with pools of up to 470 alternatives a chance
    # candidate will occasionally beat a real correspondence, so demanding that every member reject
    # was demanding a test with no type-II error. See VERIFICATION.md for the run that exposed it.
    "planted_correspondence": {
        "outcome": "rejects", "minimum_rank_one_rate": 0.90, "maximum_unresolved_at_floor": 0,
        "maximum_refusal_rate": 0.10,
        "expectation": ("a shape genuinely shared: the true partner tops its own pool for the "
                        "overwhelming majority of members, and every member it does top rejects "
                        "after correction")},
    "no_correspondence": {
        "outcome": "does_not_reject", "maximum_family_wise_error": ALPHA,
        "maximum_refusal_rate": 0.10,
        "expectation": ("independent records: the family-wise false-positive rate must clear "
                        "alpha on its upper confidence bound, and the exact ranks must be "
                        "uniform on their own lattice")},
    "shared_grid_alias": {
        "outcome": "does_not_reject", "maximum_family_wise_error": ALPHA,
        "maximum_refusal_rate": 0.10,
        "expectation": ("a shared sampling grid is not a shared shape: the pool carries the same "
                        "artefact, so the null must absorb it and hold the declared rate")},
    "clean_partner_noisy_pool": {
        "outcome": "does_not_reject", "maximum_family_wise_error": ALPHA,
        "maximum_refusal_rate": 0.10,
        "expectation": ("passing every declared band is a necessary condition for "
                        "exchangeability; this case asks whether it was a sufficient one")},
    "unresolvable_inventory": {
        "outcome": "refuses",
        "expectation": ("an inventory that cannot fill a resolving pool is refused, not scored")},
}


def witness_digest(rows: Sequence[Any]) -> str:
    """A digest of the leading realisations of a case, refusals included as themselves.

    A refused realisation is recorded as the string it is rather than skipped, because the case
    that refuses every realisation would otherwise digest an empty list -- a witness that agrees
    with any run that also produced nothing, which is the one thing a witness must not do.
    """
    return hashlib.sha256(json.dumps(list(rows), separators=(",", ":"),
                                     sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CalibrationOutcome:
    """One case, run enough times to say something, with what it can and cannot certify."""

    case: str
    realisations: int
    refusals: int
    seed: int
    witness: str
    family_wise: Dict[str, Any]
    refusal_rate: Dict[str, Any]
    per_member_after_correction: Dict[str, Any]
    per_member_uncorrected: Dict[str, Any]
    attained_floor: Dict[str, Any]
    unresolved_at_floor: int
    independent_uniformity: Dict[str, Any]
    pooled_uniformity: Dict[str, Any]
    pool_sizes: Tuple[int, int]
    seconds: float
    expectation: Mapping[str, Any]

    @property
    def outcome(self) -> str:
        return str(self.expectation["outcome"])

    @property
    def within_expectation(self) -> bool:
        """Did the case do what it was declared to do -- judged on bounds, never on points."""
        if self.outcome == "refuses":
            return self.refusals == self.realisations
        if not certifies(self.refusal_rate, float(self.expectation["maximum_refusal_rate"])):
            return False
        if self.outcome == "rejects":
            return (attains(self.attained_floor,
                            float(self.expectation["minimum_rank_one_rate"]))
                    and self.unresolved_at_floor
                    <= int(self.expectation["maximum_unresolved_at_floor"]))
        return certifies(self.family_wise, float(self.expectation["maximum_family_wise_error"]))

    @property
    def certifies_rate(self) -> bool:
        """Whether this run was large enough for its own claim, separately from whether it passed.

        A zero count in twenty realisations bounds the rate at 14%, which does not clear alpha.
        Reporting that as a pass would be exactly the assumption this module exists to remove, so
        the size question is answered on its own rather than folded into the verdict.
        """
        if self.outcome != "does_not_reject" or not self.realisations:
            return True
        ceiling = float(self.expectation["maximum_family_wise_error"])
        return one_sided_upper(0, self.realisations) <= ceiling

    def describe(self) -> Dict[str, Any]:
        return {
            "case": self.case, "statement": CASES[self.case]["statement"],
            "reproduction_witness": {
                "realisations": int(WITNESS_REALISATIONS), "seed": int(self.seed),
                "sha256": self.witness,
                "basis": (
                    "realisation `i` runs at `seed + i`, so a %d-realisation run at this seed is "
                    "the leading prefix of this one and reproduces it exactly or not at all"
                    % WITNESS_REALISATIONS)},
            "expectation": dict(self.expectation),
            "realisations": self.realisations, "refusals": self.refusals,
            "refusal_rate": self.refusal_rate,
            "refusals_are_not_neutral": (
                "a refused realisation produced no finding, so excluding it flatters a detection "
                "rate and does not flatter an error rate. Detection is therefore reported over "
                "the realisations actually scored, with the refusal rate bounded beside it "
                "rather than folded into it"),
            "family_wise_error": self.family_wise,
            "per_member_after_correction": self.per_member_after_correction,
            "per_member_uncorrected": self.per_member_uncorrected,
            "members_at_their_pool_floor": self.attained_floor,
            "members_at_their_floor_that_did_not_reject": self.unresolved_at_floor,
            "reading_a_member_at_its_floor_that_did_not_reject": (
                "the statistic ranked it first among every alternative its pool admitted and "
                "the correction still could not reject it. That member was short of pool, "
                "not short of effect, and `minimum_pool_size_for_detected_fraction` names "
                "the pool it would have needed"),
            "independent_uniformity": self.independent_uniformity,
            "pooled_uniformity": self.pooled_uniformity,
            "pool_size_range": list(self.pool_sizes),
            "within_expectation": self.within_expectation,
            "certifies_rate": self.certifies_rate,
            "uniformity_basis": (
                "one member per realisation, because members of a family share a candidate "
                "inventory and are dependent; the pooled figure beside it is a diagnostic and "
                "its p-value is not a test"),
            "seconds": round(self.seconds, 1),
        }


def _case_statistics(families: Sequence[CorrespondenceFamily], *, alpha: float) -> Dict[str, Any]:
    independent: List[float] = []
    pooled: List[float] = []
    sizes: List[int] = []
    member_rejections = 0
    uncorrected = 0
    at_floor = 0
    at_floor_unresolved = 0
    members = 0
    for family in families:
        for index, result in enumerate(family.results):
            value = rank_uniform(result.p_value, result.reference_size)
            pooled.append(value)
            if index == 0:
                independent.append(value)
            sizes.append(result.reference_size - 1)
            members += 1
            if result.p_value <= alpha:
                uncorrected += 1
            if result.p_value == result.p_floor:
                at_floor += 1
                if not family.rejected[index]:
                    at_floor_unresolved += 1
        member_rejections += family.n_rejected
    return {"independent": independent, "pooled": pooled, "sizes": sizes, "at_floor": at_floor,
            "at_floor_unresolved": at_floor_unresolved,
            "member_rejections": member_rejections, "uncorrected": uncorrected,
            "members": members}


def calibrate_case(case: str, *, realisations: int = DEFAULT_REALISATIONS, seed: int = 20260904,
                   family_size: int = CALIBRATION_FAMILY_SIZE, alpha: float = ALPHA,
                   correction: str = CORRECTION,
                   candidates_offered: Optional[int] = None) -> CalibrationOutcome:
    """Run one case `realisations` times and report what it did, as intervals."""
    if case not in CASE_EXPECTATIONS:
        raise InvalidParameterError("case", case, "one of %s" % sorted(CASE_EXPECTATIONS))
    if int(realisations) < 1:
        raise InvalidParameterError(
            "realisations", realisations,
            "at least one realisation. A rate needs trials, and zero trials would report the "
            "vacuous interval [0, 1] as though it were a measurement")
    started = time.time()
    families: List[CorrespondenceFamily] = []
    witness_rows: List[Any] = []
    refusals = 0
    for index in range(int(realisations)):
        try:
            family = run_realisation(
                case, int(seed) + index, family_size=family_size, alpha=alpha,
                correction=correction, candidates_offered=candidates_offered)
        except PoolAdmissionRefusal:
            refusals += 1
            if index < WITNESS_REALISATIONS:
                witness_rows.append("REFUSED")
            continue
        families.append(family)
        if index < WITNESS_REALISATIONS:
            # Rounded, so the witness survives a rebuild of the same arithmetic on another
            # machine's last bit. Twelve places is far finer than any pool floor this module
            # reaches, so it cannot round two genuinely different runs into agreement.
            witness_rows.append([round(float(result.p_value), 12) for result in family.results])
    stats = _case_statistics(families, alpha=alpha)
    return CalibrationOutcome(
        case=case, realisations=int(realisations), refusals=refusals,
        seed=int(seed), witness=witness_digest(witness_rows),
        family_wise=rate(sum(1 for family in families if family.n_rejected), len(families)),
        refusal_rate=rate(refusals, int(realisations)),
        per_member_after_correction=rate(stats["member_rejections"], stats["members"]),
        per_member_uncorrected=rate(stats["uncorrected"], stats["members"]),
        attained_floor=rate(stats["at_floor"], stats["members"]),
        unresolved_at_floor=int(stats["at_floor_unresolved"]),
        independent_uniformity=uniformity(stats["independent"]),
        pooled_uniformity=uniformity(stats["pooled"]),
        pool_sizes=((min(stats["sizes"]), max(stats["sizes"])) if stats["sizes"] else (0, 0)),
        seconds=time.time() - started, expectation=CASE_EXPECTATIONS[case])


def detection_profile(*, weights: Sequence[float] = EFFECT_WEIGHTS,
                      realisations: int = DEFAULT_LADDER_REALISATIONS, seed: int = 20260905,
                      family_size: int = CALIBRATION_FAMILY_SIZE, alpha: float = ALPHA,
                      correction: str = CORRECTION) -> Dict[str, Any]:
    """Detection against declared effect size, so power is a curve and not one easy case.

    Every rung is built from the same seeds, so the inventory, the left members and every declared
    marginal are identical across the ladder and a rung differs from its neighbour only in the
    planted correlation.

    The ladder ends at zero, which is a true null reached by a different route from
    `no_correspondence`: the same code path with the effect dialled out rather than a separate
    case. The two agreeing is a check that the ladder measures what the case measures.

    Each rung also reports how many members reached their own pool's floor. Where detection is low
    and that number is high, the binding constraint was pool size and not effect size -- and
    `minimum_pool_size_for_detected_fraction` already says what pool would have been needed.
    """
    rungs: List[Dict[str, Any]] = []
    for weight in weights:
        rejections = members = family_hits = scored = refused = at_floor = uncorrected = 0
        at_floor_unresolved = 0
        inventory: List[str] = []
        sizes: List[int] = []
        for index in range(int(realisations)):
            try:
                family = run_realisation("no_correspondence", int(seed) + index,
                                         family_size=family_size, weight=float(weight),
                                         alpha=alpha, correction=correction)
            except PoolAdmissionRefusal:
                refused += 1
                continue
            scored += 1
            inventory.extend(result.pool_sha256 for result in family.results)
            rejections += family.n_rejected
            members += len(family.results)
            family_hits += 1 if family.n_rejected else 0
            for position, result in enumerate(family.results):
                sizes.append(result.reference_size - 1)
                if result.p_value == result.p_floor:
                    at_floor += 1
                    if not family.rejected[position]:
                        at_floor_unresolved += 1
                if result.p_value <= alpha:
                    uncorrected += 1
        rungs.append({
            "weight": float(weight), "realisations": scored, "refusals": refused,
            "inventory_sha256": hashlib.sha256(
                "|".join(inventory).encode("utf-8")).hexdigest(),
            "member_detection": rate(rejections, members),
            "family_detection": rate(family_hits, scored),
            "member_uncorrected": rate(uncorrected, members),
            "members_at_their_pool_floor": rate(at_floor, members),
            "members_at_their_floor_that_did_not_reject": at_floor_unresolved,
            "pool_size_range": [min(sizes), max(sizes)] if sizes else [0, 0]})
    return {
        "schema": DETECTION_SCHEMA, "alpha": float(alpha), "correction": str(correction),
        "family_size": int(family_size), "realisations_per_rung": int(realisations),
        "seed": int(seed), "rungs": tuple(rungs),
        "effect_size_basis": (
            "the observed partner traces `w * shared + sqrt(1 - w^2) * own` with both components "
            "standardized, so `w` is the correlation planted before noise"),
        "held_fixed": (
            "the candidate inventory, the left members and every declared marginal are drawn from "
            "the same seeds at every rung, so a rung differs from its neighbour in the planted "
            "correlation and in nothing else. `inventory_sha256` is the digest of every pool the "
            "rung ran against and is identical across rungs, so the claim is checked rather than "
            "described"),
        "reading_a_low_rung": (
            "compare `member_detection` with `members_at_their_pool_floor`. A member at its floor "
            "that did not reject was short of pool, not short of effect, and "
            "`minimum_pool_size_for_detected_fraction` names the pool it would have needed"),
        "claim_boundary": (
            "a detection rate against a planted correlation on fixtures. It states what this "
            "family size and these pool sizes can find; it is not a statement about what any "
            "real inventory contains")}


def calibrate_pool_substitution(*, realisations: int = DEFAULT_REALISATIONS,
                                ladder_realisations: int = DEFAULT_LADDER_REALISATIONS,
                                seed: int = 20260904, family_size: int = CALIBRATION_FAMILY_SIZE,
                                alpha: float = ALPHA,
                                correction: str = CORRECTION) -> Dict[str, Any]:
    """Every case, plus the effect ladder, with one verdict that names what it rests on."""
    outcomes = [calibrate_case(case, realisations=realisations, seed=seed,
                               family_size=family_size, alpha=alpha, correction=correction)
                for case in CASE_EXPECTATIONS]
    ladder = detection_profile(realisations=ladder_realisations, seed=int(seed) + 1,
                               family_size=family_size, alpha=alpha, correction=correction)
    uncertified = [item.case for item in outcomes if not item.certifies_rate]
    failed = [item.case for item in outcomes if not item.within_expectation]
    return {
        "schema": CALIBRATION_SCHEMA,
        "statistic": "src.benchmarks.shape_calibration:shape_recurrence",
        "orientation": ORIENTATION,
        "null": "src.core.pool_substitution_null:correspondence_family",
        "family_size": int(family_size), "alpha": float(alpha), "correction": str(correction),
        "realisations": int(realisations), "seed": int(seed),
        "realisations_needed_to_certify_alpha": int(REALISATIONS_FOR_ALPHA),
        "uniformity_resolution": round(ks_resolution(realisations), 4),
        "contract": CALIBRATION_CONTRACT.describe(),
        "candidates_offered": int(CANDIDATES_OFFERED),
        "cases": [item.describe() for item in outcomes],
        "detection_profile": ladder,
        "cases_failing_expectation": failed,
        "cases_whose_run_cannot_certify_a_rate": uncertified,
        "calibrated": not failed and not uncertified,
        "verdict_basis": (
            "every error rate is judged on the upper bound of its exact binomial interval rather "
            "than on its point estimate, and a run too small to bound its own rate is reported "
            "as uncertified rather than passed"),
        "claim_boundary": (
            "a measured false-positive rate and detection curve for this null on records built "
            "here, whose answers were fixed before the method ran. It is evidence that the "
            "arithmetic and the exchangeability hold together on these fixtures. It is NOT "
            "evidence that a pool of real records is exchangeable: that rests on properties the "
            "fixtures were given by construction and a real inventory would have to be shown to "
            "have"),
    }


def admission_yield(*, seed: int = 20260906, offered: int = CANDIDATES_OFFERED,
                    family_size: int = CALIBRATION_FAMILY_SIZE) -> Dict[str, Any]:
    """What fraction of a natively diverse inventory the declared bands actually keep.

    Reported because it is the cost of the contract and is not visible in the contract.
    `native_seconds` is unbandable by design -- bounding it would refuse the comparison scale/shape
    mode exists for -- but `cadence_seconds` is banded and equals `native_seconds` divided by a
    bounded row density, so the cadence band narrows the native durations one pool can hold even
    though nothing bands them directly. That is a real restriction on the pool and it is invisible
    unless something measures it.
    """
    pools, _, meta = build_realisation("no_correspondence", seed, family_size=family_size,
                                       candidates_offered=offered)
    admitted = [len(pool.admitted) for pool in pools]
    natives = [[item.native_seconds for item in pool.admitted] for pool in pools]
    return {
        "offered": int(meta["offered"]),
        "admitted_per_pool": admitted,
        "yield_per_pool": [round(count / float(meta["offered"]), 4) for count in admitted],
        "native_seconds_range_offered": list(NATIVE_SECONDS_RANGE),
        "decades_offered": round(math.log10(
            NATIVE_SECONDS_RANGE[1] / NATIVE_SECONDS_RANGE[0]), 2),
        "decades_admitted_per_pool": [
            round(math.log10(max(values) / min(values)), 2) if values else 0.0
            for values in natives],
        "basis": ("nothing bands native duration; the cadence band narrows it transitively, "
                  "because cadence is native duration divided by a bounded row density"),
    }


__all__ = [
    "CALIBRATION_SCHEMA", "DETECTION_SCHEMA", "ORIENTATION", "SPAN_CYCLES",
    "NATIVE_SECONDS_RANGE", "ROWS_PER_CYCLE_RANGE", "NOISE_FLOOR_RANGE", "CALIBRATION_CONTRACT",
    "CALIBRATION_FAMILY_SIZE", "CANDIDATES_OFFERED", "EFFECT_WEIGHTS", "DEFAULT_REALISATIONS",
    "DEFAULT_LADDER_REALISATIONS", "CONFIDENCE", "REALISATIONS_FOR_ALPHA",
    "WITNESS_REALISATIONS", "witness_digest", "clopper_pearson",
    "one_sided_upper", "one_sided_lower", "attains", "realisations_to_certify",
    "ks_resolution", "rate", "certifies",
    "FixtureRecord", "build_record", "CASES", "CLEAN_PARTNER_NOISE", "build_realisation",
    "run_realisation", "rank_uniform", "uniformity", "CalibrationOutcome", "CASE_EXPECTATIONS",
    "calibrate_case", "detection_profile", "calibrate_pool_substitution", "admission_yield",
]
