"""TG17.15 slice 2: the partner pool, and the circularity that is made impossible rather than forbidden.

Slice 1 chose `per_correspondence`: is *this* left member's affinity for *this* partner special,
against a declared pool of candidates that are not themselves hypotheses? That question is only
answerable if the observed partner is **exchangeable** with the pool under the null. This module is
where that requirement is made checkable.

**The failure mode this slice exists to prevent.** Slice 1's arithmetic cannot be wrong -- it is
counting. Admission is different in kind: if pool members differ systematically from the observed
partner in record length, effective sample size, coverage or noise floor, then the similarity
statistic differs for reasons that have nothing to do with affinity, every p-value is wrong, and
**nothing announces it**. The enumeration bound this design replaces announced itself by refusing;
a badly curated pool returns a confident number. That asymmetry is why the roadmap calls this the
slice most likely to go quietly wrong.

**The circularity is structural, so the defence is structural.** The tempting way to build a pool
is to admit candidates that look like plausible partners -- which means admitting them on how much
they resemble the left member, which is conditioning on the statistic under test. No amount of
care prevents that if the admission function can see both records. So it cannot:
`admit` reads `RecordProfile` objects and never the records themselves, and a `RecordProfile`
carries only **marginal** properties -- quantities computed from one record alone. There is no
value in this module that is a function of two records, so an admission rule keyed on similarity
is not a rule this module can express.

Left-member marginals *may* be read, and that is not circular: a record's own length, cadence and
noise floor are not functions of any pairing. Matching a pool to the record under test is exactly
what exchangeability requires.

**Native duration is deliberately not constrained.** Scale/shape mode exists to compare shapes at
*different* native durations, so a pool banded on native duration would refuse the comparison the
mode is for. Duration is recorded, never bounded. This is the one marginal where wide spread is
the point rather than a defect, and getting it backwards would quietly convert the mode into
something else.

**What admission can and cannot establish.** It is a **necessary condition, not a sufficient one.**
Passing every declared band does not prove exchangeability; it establishes only that no *declared*
marginal visibly violates it. An unmeasured property can still differ systematically, and this
module cannot know about it. The report says so in those words, because a pool that looks like a
proof of exchangeability is more dangerous than no pool at all.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.correspondence_estimand import minimum_pool_size, require_declared_estimand
from src.core.errors import InvalidParameterError, UserInputError


POOL_SCHEMA = "correspondence-partner-pool/v1"

#: Marginals that must agree within a declared band for a candidate to be exchangeable with the
#: observed partner. Named rather than free-form, because "length" and "sample count" would be two
#: spellings of one guarantee and a contract cannot be checked against a synonym.
BANDED_MARGINALS: Tuple[str, ...] = (
    "n_samples", "effective_sample_size", "cadence_seconds", "coverage_fraction", "noise_floor")

#: Recorded on every profile and never banded. Scale/shape mode compares shapes across native
#: durations; bounding this would refuse the comparison the mode exists for.
UNBANDED_MARGINALS: Tuple[str, ...] = ("native_seconds",)


class PoolAdmissionRefusal(UserInputError):
    """A candidate, an observed partner or a whole pool failed a declared admission band."""

    def __init__(self, subject: str, reasons: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(
            "%s was refused admission to the partner pool: %s"
            % (subject, "; ".join(str(row.get("statement", row)) for row in reasons)),
            subject=subject, reasons=list(reasons), silently_dropped=False)


@dataclass(frozen=True)
class RecordProfile:
    """The marginal properties of one record, and nothing that involves a second one.

    Every field here is computable from a single record. That is the whole contract: this object
    is the only thing admission is allowed to see, so an admission rule cannot key on similarity
    to the record under test even by accident.
    """

    record_id: str
    #: Where this record came from. Two records sharing a provenance key are not independent
    #: alternatives -- a duplicate, a smoothed copy or an overlapping window would enter the
    #: reference set as evidence about the left member rather than against it.
    provenance_key: str
    n_samples: int
    #: Autocorrelation-adjusted count (R12). Two records of equal length with different memory
    #: carry different amounts of evidence, and the nominal count would hide that.
    effective_sample_size: float
    native_seconds: float
    cadence_seconds: float
    #: Fraction of the declared window actually observed, in [0, 1].
    coverage_fraction: float
    #: Dimensionless noise scale, residual over total. Lower is cleaner.
    noise_floor: float

    def __post_init__(self) -> None:
        for name in ("record_id", "provenance_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidParameterError(
                    "RecordProfile.%s" % name, value, "a non-empty identifier")
        if isinstance(self.n_samples, bool) or int(self.n_samples) != self.n_samples \
                or self.n_samples < 1:
            raise InvalidParameterError(
                "RecordProfile.n_samples", self.n_samples, "a positive integer sample count")
        for name in ("effective_sample_size", "native_seconds", "cadence_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise InvalidParameterError(
                    "RecordProfile.%s" % name, getattr(self, name),
                    "a positive finite quantity")
        if float(self.effective_sample_size) > float(self.n_samples):
            raise InvalidParameterError(
                "RecordProfile.effective_sample_size", self.effective_sample_size,
                "at most the nominal sample count (%d). An effective count above the nominal one "
                "would claim autocorrelation had added evidence rather than removed it"
                % self.n_samples)
        coverage = float(self.coverage_fraction)
        if not math.isfinite(coverage) or not 0.0 < coverage <= 1.0:
            raise InvalidParameterError(
                "RecordProfile.coverage_fraction", self.coverage_fraction,
                "an observed fraction in (0, 1]")
        noise = float(self.noise_floor)
        if not math.isfinite(noise) or noise < 0.0:
            raise InvalidParameterError(
                "RecordProfile.noise_floor", self.noise_floor,
                "a non-negative dimensionless noise scale")

    def marginal(self, name: str) -> float:
        if name not in BANDED_MARGINALS + UNBANDED_MARGINALS:
            raise InvalidParameterError(
                "marginal", name, "one of %s" % sorted(BANDED_MARGINALS + UNBANDED_MARGINALS))
        return float(getattr(self, name))

    def describe(self) -> Dict[str, Any]:
        return {"record_id": self.record_id, "provenance_key": self.provenance_key,
                "n_samples": int(self.n_samples),
                "effective_sample_size": float(self.effective_sample_size),
                "native_seconds": float(self.native_seconds),
                "cadence_seconds": float(self.cadence_seconds),
                "coverage_fraction": float(self.coverage_fraction),
                "noise_floor": float(self.noise_floor)}


@dataclass(frozen=True)
class AdmissionContract:
    """The declared bands, fixed before any candidate is looked at.

    `ratio_bands` are multiplicative tolerances against the observed partner's own marginal: a
    band of 2.0 on `n_samples` admits candidates between half and twice its length. Multiplicative
    rather than additive because every banded marginal is a positive scale quantity, and an
    additive tolerance would mean different things at different record lengths.
    """

    ratio_bands: Mapping[str, float]
    minimum_coverage: float = 0.5
    #: A candidate sharing the left member's provenance is not an independent alternative.
    require_distinct_provenance: bool = True

    def __post_init__(self) -> None:
        if not self.ratio_bands:
            raise InvalidParameterError(
                "AdmissionContract.ratio_bands", dict(self.ratio_bands),
                "at least one declared band. A pool with no admission criterion is not a curated "
                "inventory, and the exchangeability the p-values rest on would be assumed rather "
                "than checked")
        unknown = sorted(set(self.ratio_bands) - set(BANDED_MARGINALS))
        if unknown:
            offered = sorted(set(BANDED_MARGINALS) - set(self.ratio_bands))
            raise InvalidParameterError(
                "AdmissionContract.ratio_bands", unknown,
                "bands only on %s (unbanded and available: %s). `native_seconds` is deliberately "
                "not bandable: scale/shape mode exists to compare shapes across native durations, "
                "so bounding it would refuse the comparison the mode is for"
                % (sorted(BANDED_MARGINALS), offered))
        for name, value in self.ratio_bands.items():
            band = float(value)
            if not math.isfinite(band) or band < 1.0:
                raise InvalidParameterError(
                    "AdmissionContract.ratio_bands[%s]" % name, value,
                    "a finite multiplicative tolerance of at least 1.0")
        coverage = float(self.minimum_coverage)
        if not math.isfinite(coverage) or not 0.0 < coverage <= 1.0:
            raise InvalidParameterError(
                "AdmissionContract.minimum_coverage", self.minimum_coverage,
                "a required observed fraction in (0, 1]")

    @property
    def digest(self) -> str:
        body = {"ratio_bands": {k: float(v) for k, v in sorted(self.ratio_bands.items())},
                "minimum_coverage": float(self.minimum_coverage),
                "require_distinct_provenance": bool(self.require_distinct_provenance)}
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def describe(self) -> Dict[str, Any]:
        return {"ratio_bands": {k: float(v) for k, v in sorted(self.ratio_bands.items())},
                "minimum_coverage": float(self.minimum_coverage),
                "require_distinct_provenance": bool(self.require_distinct_provenance),
                "unbanded": list(UNBANDED_MARGINALS),
                "unbanded_reason": (
                    "native duration is what scale/shape mode compares across; banding it would "
                    "refuse the comparison the mode exists for"),
                "contract_sha256": self.digest}


def _band_reasons(candidate: RecordProfile, reference: RecordProfile,
                  contract: AdmissionContract, *, left: RecordProfile) -> List[Dict[str, Any]]:
    """Every reason this candidate is not exchangeable with `reference`. Marginals only."""
    reasons: List[Dict[str, Any]] = []
    if candidate.record_id == left.record_id:
        reasons.append({"check": "self_pairing", "statement": (
            "%s is the left member under test; a record compared against itself is maximally "
            "similar by construction" % candidate.record_id)})
    if contract.require_distinct_provenance and candidate.provenance_key == left.provenance_key:
        reasons.append({"check": "provenance_independence", "statement": (
            "%s shares provenance %r with the left member, so it is a derived or duplicate "
            "record rather than an independent alternative"
            % (candidate.record_id, candidate.provenance_key))})
    if float(candidate.coverage_fraction) < float(contract.minimum_coverage):
        reasons.append({"check": "minimum_coverage", "statement": (
            "%s observes %.4f of its window, below the declared minimum %.4f"
            % (candidate.record_id, candidate.coverage_fraction, contract.minimum_coverage))})
    for name, band in sorted(contract.ratio_bands.items()):
        observed = candidate.marginal(name)
        target = reference.marginal(name)
        if target == 0.0:
            ratio = float("inf") if observed > 0.0 else 1.0
        else:
            ratio = observed / target
        if ratio <= 0.0 or not math.isfinite(ratio) \
                or ratio > float(band) or ratio < 1.0 / float(band):
            reasons.append({"check": name, "statement": (
                "%s has %s = %.6g against the observed partner's %.6g, a ratio of %.4g outside "
                "the declared band [%.4g, %.4g]"
                % (candidate.record_id, name, observed, target, ratio,
                   1.0 / float(band), float(band)))})
    return reasons


@dataclass(frozen=True)
class PartnerPool:
    """A sealed inventory of admitted alternatives for one tested correspondence."""

    left: RecordProfile
    observed_partner: RecordProfile
    admitted: Tuple[RecordProfile, ...]
    refused: Tuple[Mapping[str, Any], ...]
    contract: AdmissionContract
    estimand: str
    tested_correspondences: int

    def __len__(self) -> int:
        return len(self.admitted)

    @property
    def resolution_floor(self) -> float:
        """The smallest p this pool can produce: the observation plus its admitted alternatives."""
        return 1.0 / (len(self.admitted) + 1)

    @property
    def digest(self) -> str:
        body = {"left": self.left.describe(),
                "observed_partner": self.observed_partner.describe(),
                "admitted": [item.describe() for item in self.admitted],
                "contract": self.contract.describe(), "estimand": self.estimand}
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def spread(self) -> Dict[str, Any]:
        """How homogeneous the admitted pool is, and where the observation sits inside it.

        Reported rather than checked. A partner at the extreme of its own pool on some marginal is
        not refused -- it is inside every declared band by construction -- but it is exactly the
        situation where an undeclared property is most likely to be doing the work, and a reader
        who cannot see it cannot judge it.
        """
        body: Dict[str, Any] = {}
        for name in BANDED_MARGINALS + UNBANDED_MARGINALS:
            values = sorted(item.marginal(name) for item in self.admitted)
            target = self.observed_partner.marginal(name)
            below = sum(1 for value in values if value < target)
            body[name] = {
                "observed_partner": target,
                "pool_min": values[0] if values else None,
                "pool_max": values[-1] if values else None,
                "pool_median": values[len(values) // 2] if values else None,
                "observed_partner_percentile": (
                    round(100.0 * below / len(values), 2) if values else None),
            }
        return body

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": POOL_SCHEMA,
            "estimand": self.estimand,
            "left": self.left.describe(),
            "observed_partner": self.observed_partner.describe(),
            "n_admitted": len(self.admitted),
            "n_refused": len(self.refused),
            "tested_correspondences": int(self.tested_correspondences),
            "minimum_pool_size": minimum_pool_size(int(self.tested_correspondences)),
            "resolution_floor": self.resolution_floor,
            "contract": self.contract.describe(),
            "admitted": [item.describe() for item in self.admitted],
            "refused": [dict(row) for row in self.refused],
            "spread": self.spread(),
            "pool_sha256": self.digest,
            "exchangeability_basis": (
                "admission is decided on each record's own marginal properties and on the left "
                "member's, never on any quantity computed between a candidate and the record "
                "under test. No value in this module is a function of two records, so an "
                "admission rule keyed on the statistic being tested is not expressible here"),
            "claim_boundary": (
                "passing every declared band is a NECESSARY condition for exchangeability and "
                "not a sufficient one. It establishes that no declared marginal visibly violates "
                "exchangeability; an undeclared property may still differ systematically, and "
                "this pool cannot know about it. A pool is not a proof that its members are "
                "interchangeable"),
        }


def build_partner_pool(*, left: RecordProfile, observed_partner: RecordProfile,
                       candidates: Sequence[RecordProfile], contract: AdmissionContract,
                       estimand: str, tested_correspondences: int) -> PartnerPool:
    """Admit candidates on declared marginals, refusing each failure by name.

    Nothing is dropped silently. A candidate that fails is carried in `refused` with the band it
    failed and by how much, because a pool that quietly shrinks is a pool whose resolution nobody
    can audit.
    """
    declared = require_declared_estimand(estimand)
    if not isinstance(contract, AdmissionContract):
        raise InvalidParameterError(
            "contract", type(contract).__name__,
            "an explicit AdmissionContract declared before the candidates were seen")
    for name, value in (("left", left), ("observed_partner", observed_partner)):
        if not isinstance(value, RecordProfile):
            raise InvalidParameterError(name, type(value).__name__, "a RecordProfile")
    if left.record_id == observed_partner.record_id:
        raise InvalidParameterError(
            "observed_partner", observed_partner.record_id,
            "a record other than the left member. A correspondence of a record with itself has "
            "maximal similarity by construction and no null can be built for it")

    seen: Dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, RecordProfile):
            raise InvalidParameterError(
                "candidates[%d]" % index, type(candidate).__name__, "a RecordProfile")
        if candidate.record_id in seen:
            raise InvalidParameterError(
                "candidates[%d].record_id" % index, candidate.record_id,
                "one entry per record. This id also appears at position %d, and counting a "
                "record twice would inflate the reference set with a duplicate rather than an "
                "alternative" % seen[candidate.record_id])
        seen[candidate.record_id] = index

    # The observation must clear the same bands as its own alternatives. If it does not, the
    # reference set is not exchangeable with the thing it is the reference for, and the p-value
    # would compare a record against alternatives it was never comparable to.
    partner_reasons = _band_reasons(observed_partner, observed_partner, contract, left=left)
    if partner_reasons:
        raise PoolAdmissionRefusal(
            "the observed partner %r" % observed_partner.record_id, partner_reasons)

    admitted: List[RecordProfile] = []
    refused: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate.record_id == observed_partner.record_id:
            refused.append({"record_id": candidate.record_id,
                            "reasons": [{"check": "observed_partner", "statement": (
                                "%s is the observed partner; it is the observation, not one of "
                                "its own alternatives" % candidate.record_id)}]})
            continue
        reasons = _band_reasons(candidate, observed_partner, contract, left=left)
        if reasons:
            refused.append({"record_id": candidate.record_id, "reasons": reasons})
        else:
            admitted.append(candidate)

    required = minimum_pool_size(int(tested_correspondences))
    if len(admitted) < required:
        raise PoolAdmissionRefusal(
            "the pool for %r" % left.record_id,
            [{"check": "resolution", "statement": (
                "%d candidates were admitted of %d offered, and %d tested correspondences need "
                "at least %d for the exact floor 1/(N+1) to clear %s at alpha 0.05. The pool "
                "cannot resolve, so a p-value from it could not reject at any effect size"
                % (len(admitted), len(candidates), int(tested_correspondences), required,
                   "Benjamini-Yekutieli"))}])

    return PartnerPool(left=left, observed_partner=observed_partner,
                       admitted=tuple(admitted), refused=tuple(refused), contract=contract,
                       estimand=declared.name,
                       tested_correspondences=int(tested_correspondences))


__all__ = [
    "POOL_SCHEMA", "BANDED_MARGINALS", "UNBANDED_MARGINALS", "PoolAdmissionRefusal",
    "RecordProfile", "AdmissionContract", "PartnerPool", "build_partner_pool",
]
