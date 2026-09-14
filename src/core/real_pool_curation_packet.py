"""Deterministic marginal-admission evidence for human G17 pool curation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from src.core.partner_pool import AdmissionContract, RecordProfile
from src.core.real_pool_readiness import DEFAULT_PROFILE_ROOTS, audit_real_pool_readiness
from src.core.scale_shape_successor import successor_scale_shape_manifest


SCHEMA = "g17-pool-curation-packet/v1"


def _admitted(candidate: RecordProfile, reference: RecordProfile,
              contract: AdmissionContract) -> bool:
    if candidate.record_id == reference.record_id:
        return False
    if contract.require_distinct_provenance \
            and candidate.provenance_key == reference.provenance_key:
        return False
    if candidate.coverage_fraction < contract.minimum_coverage:
        return False
    return all(
        1.0 / float(band) <= candidate.marginal(name) / reference.marginal(name) <= float(band)
        for name, band in contract.ratio_bands.items())


def build_curation_packet(*, roots: Sequence[Path] = DEFAULT_PROFILE_ROOTS) -> Dict[str, Any]:
    """Find a deterministic all-pairs-admissible subset without judging exchangeability."""
    successor = successor_scale_shape_manifest()
    readiness = audit_real_pool_readiness(successor, roots=roots)
    profiles = [RecordProfile(**row["profile"]) for row in readiness["profiles"]]
    contract = AdmissionContract(
        ratio_bands=successor.admission_contract["ratio_bands"],
        minimum_coverage=successor.admission_contract["minimum_coverage"],
        require_distinct_provenance=successor.admission_contract["require_distinct_provenance"])
    graph = {profile.record_id: {
        candidate.record_id for candidate in profiles
        if _admitted(candidate, profile, contract)
    } for profile in profiles}

    required = successor.minimum_pool_size_per_correspondence
    core = set(graph)
    while True:
        removed = {record_id for record_id in core
                   if len(graph[record_id] & core) < required}
        if not removed:
            break
        core -= removed

    def greedy(start: str) -> list[str]:
        clique = [start]
        candidates = set(graph[start]) & core
        while candidates:
            chosen = sorted(
                candidates, key=lambda value: (-len(graph[value] & candidates), value))[0]
            clique.append(chosen)
            candidates &= graph[chosen]
        return sorted(clique)

    choices = [greedy(record_id) for record_id in sorted(core)]
    selected = max(choices, key=lambda values: (len(values), tuple(reversed(values)))) \
        if choices else []
    body: Dict[str, Any] = {
        "schema": SCHEMA,
        "study_id": successor.study_id,
        "successor_declaration_sha256": successor.digest,
        "readiness_assessment_sha256": readiness["assessment_sha256"],
        "admission_contract_sha256": contract.digest,
        "profile_count": len(profiles),
        "required_alternatives": required,
        "minimum_selected_records": required + 1,
        "core_record_ids": sorted(core),
        "core_size": len(core),
        "recommended_record_ids": selected,
        "recommended_size": len(selected),
        "alternatives_per_recommended_record": max(0, len(selected) - 1),
        "excluded_records": [{
            "record_id": record_id,
            "admissible_neighbors": len(graph[record_id]),
            "reason": "not in the deterministic all-pairs-admissible recommended subset",
        } for record_id in sorted(graph) if record_id not in selected],
        "selection_method": (
            "maximum over deterministic degree-first greedy cliques seeded by every record ID; "
            "this proves the returned subset is pairwise admissible, not globally maximum"),
        "exchangeability": "NOT_ASSESSED",
        "claim_boundary": (
            "This packet proves only that every recommended record satisfies the declared "
            "marginal bands against every other recommended record and that each has enough "
            "alternatives. Necessary marginal agreement is not sufficient for exchangeability; "
            "the final decision remains human."),
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["packet_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


__all__ = ["SCHEMA", "build_curation_packet"]