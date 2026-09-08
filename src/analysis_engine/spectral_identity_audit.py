"""Descriptive T4E.8 identity errors against tracked-constellation proxy labels.

No null, significance, independent-sample claim or automatic calibration acceptance lives here.
The caller supplies a predeclared radius, or explicitly requests a calibration quantile.
Storage is O(P + U), runtime O(U log U + P log U), for P repeat and U unrelated distances.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidParameterError, MissingParameterError
from src.core.registry import Registry


@dataclass(frozen=True)
class EvidenceClass:
    """Where identity labels came from, and what that provenance can support."""

    provenance: str
    independent_of_record: bool
    label_boundary: str


@dataclass(frozen=True)
class IdentityTarget:
    """What an identity definition claims to recognise, as data rather than commentary."""

    recognises: str
    admissible_evidence: Tuple[str, ...]
    does_not_license: str
    caveats: Tuple[Tuple[str, str], ...] = ()

    def caveat_for(self, evidence: str) -> Optional[str]:
        return dict(self.caveats).get(evidence)


#: The wording every T4E.8 figure to date was published under. Kept verbatim so that
#: naming a target cannot silently reword a receipt that has already been cited.
PROXY_LABEL_BOUNDARY = "Tracked-key proxy labels, not independent measurements or physical ground truth"

EVIDENCE_CLASSES: Registry[EvidenceClass] = Registry("identity evidence class")
IDENTITY_TARGETS: Registry[IdentityTarget] = Registry("identity target")

EVIDENCE_CLASSES.add("record_derived_proxy", EvidenceClass(
    provenance="Labels computed from the same record and pipeline whose identity is under test",
    independent_of_record=False,
    label_boundary=PROXY_LABEL_BOUNDARY),
    description="Tracked-constellation keys. Cheap, always available, and not independent.")
EVIDENCE_CLASSES.add("external_reference", EvidenceClass(
    provenance="Labels from a reviewed catalogue supplied from outside this record",
    independent_of_record=True,
    label_boundary="Reference-catalogue labels; identity is held fixed by the supplied catalogue"),
    description="A signed, frozen catalogue matched at its own declared radius.")

IDENTITY_TARGETS.add("track_continuity", IdentityTarget(
    recognises="The same evolving tracked constellation, observed again while tracking held it",
    admissible_evidence=("record_derived_proxy", "external_reference"),
    does_not_license="Recurrence of a configuration in a different constellation, or any claim "
                     "that the configuration was unchanged between the two observations",
    caveats=(("record_derived_proxy",
              "Tracked keys define this target rather than testing it: a high score "
              "demonstrates agreement with the tracker, not independent identity."),)),
    description="Continuity of an evolving tracked constellation.")
IDENTITY_TARGETS.add("spatial_persistence", IdentityTarget(
    recognises="Persistence of a spatial configuration's geometry within a declared record/grid",
    admissible_evidence=("record_derived_proxy", "external_reference"),
    does_not_license="Location-independent physical morphology, cross-record transfer, or "
                     "recurrence of the same physical kind in another constellation",
    caveats=(("record_derived_proxy",
              "Repeated tracked keys bound how far geometry drifts over a track's lifetime; "
              "genuine morphological change is scored as a split, not as an error."),)),
    description="Persistence of spatial configuration, the slice-2 candidate's target.")
IDENTITY_TARGETS.add("kind_recurrence", IdentityTarget(
    recognises="Recurrence of the same physical kind in a different constellation",
    admissible_evidence=("external_reference",),
    does_not_license="Anything, until evaluated against labels not derived from this record"),
    description="The target the mining, sequence and precursor machinery requires.")


def declare_identity_target(target: str, evidence: str) -> Dict[str, Any]:
    """Admit a target/evidence pairing, or refuse it by name. No default is supplied.

    The refusal that matters is `kind_recurrence` against record-derived labels: those labels
    are produced by the same record and pipeline whose identity is under test, so a definition
    validated against them is validated against itself. That is not a bound to be relaxed.
    """
    if not target:
        raise MissingParameterError("identity_target", "an identity audit",
                                    required=IDENTITY_TARGETS.names())
    if not evidence:
        raise MissingParameterError("evidence_class", "an identity audit",
                                    required=EVIDENCE_CLASSES.names())
    specification = IDENTITY_TARGETS.get(target)
    evidence_specification = EVIDENCE_CLASSES.get(evidence)
    if evidence not in specification.admissible_evidence:
        raise InvalidParameterError(
            "evidence_class", evidence,
            "evidence admissible for target %r: %s. %s labels are %s, so a definition "
            "validated against them is validated against itself; choose admissible labels "
            "rather than relaxing this pairing"
            % (target, ", ".join(specification.admissible_evidence), evidence,
               evidence_specification.provenance.lower()))
    return {
        "identity_target": target,
        "recognises": specification.recognises,
        "evidence_class": evidence,
        "evidence_provenance": evidence_specification.provenance,
        "evidence_independent_of_record": evidence_specification.independent_of_record,
        "label_boundary": evidence_specification.label_boundary,
        "does_not_license": specification.does_not_license,
        "caveat": specification.caveat_for(evidence),
        "admissible_evidence": list(specification.admissible_evidence),
    }


def catalogue_labels(catalogue: Any, signatures: Sequence[Any], *,
                     unrelated_pairs: int, seed: int,
                     attempts_per_negative: int = 1000) -> Dict[str, Any]:
    """Label signatures by which reviewed pattern they fall into, not by their track keys.

    Identity is held fixed by the supplied catalogue: `match_into_catalogue` takes its
    centroids, metric and tolerance radius from that catalogue and moves none of them, so a
    signature joins the nearest pattern whose calibrated radius contains it and joins nothing
    otherwise. Two signatures are a positive pair when they land in the same pattern.

    Signatures matching nothing are reported, never silently treated as negatives: a
    configuration the catalogue does not recognise is unlabelled, not known to be different.
    """
    from src.analysis_engine.spectral_clustering import PatternCatalogue, SignaturePoint
    from src.analysis_engine.spectral_regions import match_into_catalogue

    if not isinstance(catalogue, PatternCatalogue):
        raise InvalidParameterError("catalogue", type(catalogue).__name__,
                                    "a reviewed T4E.3 PatternCatalogue supplied from outside "
                                    "the record under test")
    families = {pattern.centroid.family for pattern in catalogue}
    present = {SignaturePoint.from_signature(signature).family for signature in signatures}
    if not present & families:
        raise InvalidParameterError(
            "catalogue.patterns", sorted(str(family.describe()) for family in families),
            "at least one pattern whose signature family matches the signatures under test; "
            "a catalogue signed under a different mode, cardinality or comparison scope "
            "labels nothing here and must not be reported as total disagreement")

    matched = match_into_catalogue(catalogue, list(signatures))
    assignment: Dict[Any, int] = {}
    for pattern in matched.catalogue:
        for member in pattern.members:
            assignment[tuple(member.key)] = int(pattern.pattern_id)
    groups: Dict[int, list] = {}
    unlabelled = 0
    for index, signature in enumerate(signatures):
        pattern_id = assignment.get(tuple(signature.key))
        if pattern_id is None:
            unlabelled += 1
            continue
        groups.setdefault(pattern_id, []).append(index)

    same = [pair for members in groups.values()
            for pair in itertools.combinations(sorted(members), 2)]
    if len(groups) < 2:
        raise InvalidParameterError(
            "catalogue_labels.groups", len(groups),
            "signatures falling into at least two distinct reviewed patterns; negatives "
            "cannot be drawn from a single identity")
    labelled = [index for members in groups.values() for index in members]
    rng = np.random.default_rng(seed)
    unrelated = []
    # Rejection sampling terminates, but a catalogue whose patterns are wildly unequal in
    # support makes cross-pattern draws rare enough to look like a hang. Refuse by name
    # instead: an unmeasured negative population is not the same as a slow one.
    attempts, budget = 0, attempts_per_negative * max(unrelated_pairs, 1)
    while len(unrelated) < unrelated_pairs:
        if attempts >= budget:
            raise InvalidParameterError(
                "catalogue_labels.unrelated_pairs", unrelated_pairs,
                "a negative population these patterns can supply; %d draws yielded %d "
                "cross-pattern pairs, so support is too concentrated in one identity"
                % (attempts, len(unrelated)))
        attempts += 1
        left, right = (labelled[int(i)] for i in rng.integers(0, len(labelled), size=2))
        if assignment[tuple(signatures[left].key)] != assignment[tuple(signatures[right].key)]:
            unrelated.append((left, right))
    return {
        "same": same, "unrelated": unrelated,
        "metadata": {
            "signatures": len(signatures), "labelled_signatures": len(labelled),
            "unlabelled_signatures": unlabelled, "distinct_patterns": len(groups),
            "repeat_pairs": len(same),
            "unrelated_unique_unordered_pairs": len({tuple(sorted(p)) for p in unrelated}),
            "matched_into_an_existing_identity": matched.n_matched,
            "matched_nothing": matched.n_unmatched,
            "boundary": ("Labels are the supplied catalogue's identities. Unmatched "
                         "signatures are unlabelled, not negative."),
        },
    }


def _distances(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or np.any(~np.isfinite(result)) or np.any(result < 0):
        raise InvalidParameterError("identity_audit.distances", "invalid",
                                    "a one-dimensional sequence of finite nonnegative distances")
    return result


def recall_radius(repeats: Sequence[float], recall: float) -> Optional[float]:
    """Smallest observed radius reaching the declared empirical recall; None if unmeasured."""
    if not math.isfinite(recall) or not 0 < recall <= 1:
        raise InvalidParameterError("identity_audit.recall", recall, "a recall in (0, 1]")
    values = _distances(repeats)
    if not len(values):
        return None
    return float(np.sort(values)[math.ceil(recall * len(values)) - 1])


def labelled_errors(repeats: Sequence[float], unrelated: Sequence[float],
                    radius: Optional[float],
                    label_boundary: str = PROXY_LABEL_BOUNDARY) -> Dict[str, Any]:
    """Both absolute empirical error rates, with ties included inside the radius.

    AUC is P(repeat distance < unrelated distance) + half the tie probability. Empty
    populations remain unmeasured, never zero errors. `label_boundary` travels with every
    report so no reader sees a rate without the provenance of the labels behind it; it
    defaults to the record-derived proxy wording these figures were first published under.
    """
    same, different = _distances(repeats), _distances(unrelated)
    if radius is not None and (not math.isfinite(radius) or radius < 0):
        raise InvalidParameterError("identity_audit.radius", radius,
                                    "a finite nonnegative radius or None when not calibrated")
    auc = None
    if len(same) and len(different):
        sorted_different = np.sort(different)
        below = np.searchsorted(sorted_different, same, side="left")
        through = np.searchsorted(sorted_different, same, side="right")
        auc = float(np.mean((len(different) - through + (through - below) / 2)
                            / len(different)))
    split = int(np.count_nonzero(same > radius)) if radius is not None and len(same) else None
    admitted = (int(np.count_nonzero(different <= radius))
                if radius is not None and len(different) else None)
    return {
        "radius": radius, "repeat_pairs": len(same), "unrelated_pairs": len(different),
        "false_split_count": split, "false_admission_count": admitted,
        "false_split_rate": None if split is None else split / len(same),
        "false_admission_rate": None if admitted is None else admitted / len(different),
        "auc": auc,
        "label_boundary": label_boundary,
    }


def radius_feasibility(repeats: Sequence[float], unrelated: Sequence[float], *,
                       max_split: float, max_admission: float,
                       label_boundary: str = PROXY_LABEL_BOUNDARY) -> Dict[str, Any]:
    """Whether any radius meets both empirical bounds on these supplied proxy populations.

    Admission is monotone in radius. The smallest radius satisfying the split bound
    therefore minimises admission among all split-feasible radii. This is a diagnostic of
    this population only, never an approved operating point or a generalisation bound.
    """
    for name, value in (("max_split", max_split), ("max_admission", max_admission)):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise InvalidParameterError("identity_audit." + name, value, "a bound in [0, 1)")
    same = _distances(repeats)
    # Count admissible rejections directly. Subtracting max_split from 1 first can
    # round 0.3 upward and ceil(0.30000000000000004 * 10) would then require four.
    radius = (float(np.sort(same)[len(same) - math.floor(max_split * len(same)) - 1])
              if len(same) else None)
    report = labelled_errors(repeats, unrelated, radius, label_boundary)
    admission = report["false_admission_rate"]
    return {"smallest_split_feasible_radius": radius, "errors": report,
            "any_radius_meets_both_empirical_bounds": (None if radius is None or admission is None
                                                      else admission <= max_admission),
            "boundary": "Descriptive feasibility on supplied proxy labels; not a mining radius"}
