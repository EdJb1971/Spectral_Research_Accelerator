"""TG17.10 deterministic qualification of the configurable experiment apparatus.

This module is deliberately a *release gate*, not a scientific worker.  It exercises the six
frozen duration/mode declarations through metadata preflight, the resumable orchestrator,
portable receipt export and read-only replay.  Every artefact it creates is labelled as a
known-answer rehearsal, and the overall verdict remains ``NOT_RELEASEABLE`` while browser,
mode-specific calibration and live-source gates have not been recorded separately.

That distinction is load-bearing.  A green state-machine rehearsal proves that the apparatus
preserves one plan.  It says nothing about whether an archive was reachable, a statistic was
calibrated, an effect was detected, or a claim should move.

It is also load-bearing that a cell may come back ``REFUSED``.  The frozen quartet includes the
bespoke order-book domain, whose adapter declares it cannot carry ``scale_partner_reassignment``
because a depositor-supplied record has no native duration worth comparing shapes across.  The
three scale/shape cells are therefore refused at preflight, and the gate records that refusal as
the qualification result rather than widening a framework default until the matrix turns green.
A domain's declared refusal is the science; a passing matrix obtained by overruling it would be
the failure this gate exists to catch.

The ``scale_shape_calibration`` gate is the same distinction one level up (TG17.11).  A registered
calibration for that mode now exists and is measured outside orchestration, so the gate no longer
reads ``NOT_IMPLEMENTED``.  It reads ``REFUSED``, because what blocks it is not an absent method
but an inapplicable one: the declared null's support is too small to resolve anything at the sizes
it will draw from, and too large to draw from at the sizes that could resolve.  A calibrated method
the declared plans cannot reach, a method that does not exist, and a method that ran and failed are
three different facts, and the record keeps them apart.

``calendar_calibration`` is the fourth of those facts and the only one that clears (TG17.12): a
method that exists, is applicable at the declared configuration, ran, and met every answer frozen
with its fixtures.  It is still not executed here, for the reason above.  The gate reads a
recording bound to the declared contract and to the source of the modules that decide what the
measurement is, so relaxing an expectation or changing the statistic returns this gate to
``NOT_RUN`` instead of leaving a stale pass behind.  See ``src.core.calibration_record``.

TG17.15 slice 5 changes why that gate refuses without changing that it refuses, and the change is
recorded as a **checked supersession** rather than as an edit.  TG17.11's refusal was about a
method: the declared joint-reassignment null cannot reject at any inventory size its enumerator
will draw from.  A different null answering a different estimand - exact substitution over a
declared partner pool - now exists, is applicable at a pool size the fixtures actually reach, and
is calibrated and recorded.  Deleting the old refusal at that point would leave a repository in
which a limit that was overcome and a limit that was edited away read identically, so the old claim
is kept, **recomputed** on every plan, and superseded only while it is still true on its own terms.
See ``src.core.calibration_record.scale_shape_supersession``.

What the supersession does not do is turn the gate green, and the reasons are computed rather than
asserted.  The qualification manifests declare ``scale_partner_reassignment`` with a replication
count, which is checked against the manifests this module builds; the calibrated method is not the
one they request.  And the calibration's own claim boundary says it is evidence about built
fixtures rather than about whether a pool of *real* records is exchangeable - a curation obligation
that no amount of further measurement on fixtures discharges.  The gate therefore publishes its
blockers, each with what would discharge it and whether this module can decide it at all.

``synthetic_fifth_adapter`` is the fifth, and it is the only gate here whose evidence is partly
readable on the spot (TG17.13).  Its source-edit half is decidable from committed source in
milliseconds, so it is re-run live rather than believed from a receipt; its acceptance-run half
cannot be reached from this side at all, because TG17.3 requires the fifth adapter to be defined in
a module the application never imports, so it is recorded by that test and read back bound to the
declared contract and source.  The gate publishes the standing glue count even when it passes:
what this installation required is asserted to be zero, and what stands in the generic surfaces
today is reported, because a count nobody prints is a count that grows.  See
``src.core.extension_evidence``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from src.benchmarks.multidomain_flagship import EXPERIMENT_CONTRACT
from src.core.calibration_record import scale_shape_supersession
from src.benchmarks.structural_trajectory import known_answer_native
from src.core.experiment_manifest import (
    CrossDomainExperimentSpec,
    canonical_bytes,
    flagship_recipe,
    manifest_sha256,
    preflight_manifest,
)
from src.core.experiment_receipt import publish_bundle, verify_bundle
from src.core.experiment_run import RunStore
from src.core.run_workers import build_suite


SCHEMA = "experiment-qualification/v1"
RECORD_KIND = "deterministic_known_answer_rehearsal_not_acquired_data"
MODES: Tuple[str, ...] = ("calendar_aligned", "scale_shape_aligned")
DURATIONS: Tuple[str, ...] = tuple(EXPERIMENT_CONTRACT["duration_presets"])


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def qualification_manifest(duration: str, mode: str, *, suffix: str = "") \
        -> CrossDomainExperimentSpec:
    """One frozen matrix cell, derived without inventing a second flagship configuration."""
    if duration not in DURATIONS:
        raise ValueError("duration must be one of %s" % (list(DURATIONS),))
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (list(MODES),))

    body = json.loads(canonical_bytes(flagship_recipe()))
    body["study_id"] = "g17_qualification_%s_%s%s" % (
        mode, duration, ("_" + suffix) if suffix else "")
    body["mode"] = mode
    body["title"] = "G17 qualification: %s / %s" % (
        mode.replace("_", " "), duration.replace("_", " "))
    body["windows"] = [row for row in body["windows"] if row["name"] == duration]
    body["confirmation"]["held_out_partition"] = "g17_qualification_heldout_%s_%s%s" % (
        mode, duration, ("_" + suffix) if suffix else "")

    # The order-book fixture is a real content-addressed record.  Binding its digest here is
    # what lets metadata plan the complete quartet without pretending it has a public archive.
    for observation in body["observations"]:
        observation["adapter"]["parameters"]["source_binding"] = "benchmark_known_answer"
        if observation["domain"] == "order_book":
            observation["acquisition"]["identity"]["content_sha256"] = (
                known_answer_native("order_book").content_sha256)

    if mode == "calendar_aligned":
        body["scale_normalization"] = None
        body["family"]["relationships"] = ["co_occurrence"]
        body["nulls"] = [{"name": "domain_preserving_shift",
                          "method": "independent_native_clock_shift",
                          "replications": 200, "parameters": {}}]
        body["notes"]["claim_boundary"] = "co-occurrence only"
    else:
        body["scale_normalization"] = {
            "method": "native_scale_ratio", "reference": "within_domain"}
        body["family"]["relationships"] = ["shape_recurrence"]
        body["nulls"] = [{"name": "scale_partner_reassignment",
                          "method": "scale_partner_reassignment",
                          "replications": 200, "parameters": {}}]
        body["notes"]["claim_boundary"] = (
            "shape recurrence only; no simultaneity, precedence or causality")
    body["notes"]["qualification_record_kind"] = RECORD_KIND
    body["notes"]["duration_selected_before_results"] = True
    return CrossDomainExperimentSpec.parse_obj(body)


def _gate(gate_id: str, title: str, status: str, detail: str,
          *, blocking: bool = True) -> Dict[str, Any]:
    return {"gate_id": gate_id, "title": title, "status": status,
            "blocking": blocking and status != "PASS", "detail": detail}


# ------------------------------------------------- TG17.11 slice 5: the scale/shape gate

#: Where the *superseded* scale/shape calibration lives - TG17.11's, built on the declared
#: joint-reassignment null. It is named rather than executed here, for the reason
#: `calendar_calibration` gives: a calibration is a scientific measurement and this module is a
#: release gate. Running a three-minute family calibration inside plan assembly would also make an
#: HTTP route's cost depend on a benchmark's. It is still named because the refusal it belongs to
#: is still recomputed and still true; see `scale_shape_supersession`.
SCALE_SHAPE_CALIBRATION = "src.benchmarks.shape_fixtures:calibrate_shape_family"

#: The calibration the gate now decides applicability on (TG17.15). Two entry points appear here
#: rather than one because two different questions were asked, and replacing the first name with
#: the second would be the edit this slice exists not to make.
SCALE_SHAPE_CALIBRATION_SUPERSEDING = (
    "src.benchmarks.pool_calibration:calibrate_pool_substitution")

#: The two domain families G17 could offer this null: the frozen quartet compared all-against-all,
#: and the same inventory once `order_book` has declined the null under D83.
SCALE_SHAPE_DOMAIN_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("quartet_all_pairs", ("argo_float", "reanalysis", "tess_lightcurve", "order_book")),
    ("admitting_triple_all_pairs", ("argo_float", "reanalysis", "tess_lightcurve")),
)


def scale_shape_applicability() -> Dict[str, Any]:
    """Whether the scale/shape mode's declared null can resolve anything, decided before power.

    This is the fact the gate turns on, and it is decidable exactly and in milliseconds, without
    acquiring a record or running a calibration. Two bounds, established by TG17.11 slices 1-4,
    close on the same declared null from opposite directions:

    * **From below, the resolution bound.** A member of a ``k``-pairing family has ``k - 1``
      admissible alternative partners, so its exact p-value cannot fall below ``1/k``. Solved
      against the real Benjamini-Yekutieli correction at the declared alpha, no family smaller
      than ``minimum_resolvable_family()`` pairings can reject even when every member is a
      perfect planted match.
    * **From above, the enumeration bound.** ``reassign_scale_partners`` draws uniformly over the
      reassignments an inventory admits, enumerated exactly, and refuses above
      ``MAX_REASSIGNABLE_PAIRINGS`` rather than adopt a sampler whose uniformity is assumed.

    The two do not meet: the largest inventory the declared null will draw from is an order of
    magnitude smaller than the smallest inventory that could reject.  There is therefore no
    inventory size at which this null *as the qualification manifests declare it* — drawn, with a
    replication count — can produce a rejection.  What can is the exact partner test, which
    enumerates the same finite support instead of resampling it, and which the registered
    calibration is built on.  No declared manifest requests it.

    G17's two candidate families are refused for a second and independent reason before size is
    even reached: one admits a single distinguishable reassignment, so its null is a constant
    rather than a distribution, and the other admits none at all.  Both refusals are recorded
    verbatim from the null itself rather than restated here.
    """
    from src.benchmarks.shape_fixtures import ALPHA, CORRECTION, minimum_resolvable_family
    from src.core.structural_nulls import (
        MAX_REASSIGNABLE_PAIRINGS,
        NullRefusal,
        reassign_scale_partners,
    )

    minimum = minimum_resolvable_family()
    families: List[Dict[str, Any]] = []
    for name, domains in SCALE_SHAPE_DOMAIN_FAMILIES:
        pairings = [(left, right) for index, left in enumerate(domains)
                    for right in domains[index + 1:]]
        try:
            reassign_scale_partners(pairings, 20260903)
        except NullRefusal as error:
            refusal = str(error)
        else:  # pragma: no cover - both declared families are refused; TG17.11 slice 1
            refusal = ""
        families.append({
            "family": name,
            "domains": list(domains),
            "pairings": len(pairings),
            "null_refusal": refusal,
            "reaches_resolvable_size": len(pairings) >= minimum,
        })
    supersession = scale_shape_supersession()
    declared = declared_scale_shape_null()
    return {
        "calibration": SCALE_SHAPE_CALIBRATION,
        "calibration_superseding": SCALE_SHAPE_CALIBRATION_SUPERSEDING,
        "calibration_executed_here": False,
        "inference_the_calibration_uses": "exact_partner_p_values",
        "inference_the_manifests_declare": "drawn surrogates with a replication count",
        "alpha": ALPHA,
        "correction": CORRECTION,
        "minimum_resolvable_family": minimum,
        "largest_drawable_inventory": MAX_REASSIGNABLE_PAIRINGS,
        "declared_null_can_ever_reject": MAX_REASSIGNABLE_PAIRINGS >= minimum,
        "declared_families": families,
        "declared_null": declared,
        "supersession": supersession,
        "blockers": scale_shape_blockers(supersession, declared),
        "claim_boundary": (
            "This gate reports applicability and reads a recorded calibration's verdict; it "
            "executes no calibration and asserts no power number of its own. A recorded pass is "
            "evidence about built fixtures, and is not evidence that a pool of real records is "
            "exchangeable."),
    }


def declared_scale_shape_null() -> Dict[str, Any]:
    """Which inference the frozen scale/shape manifests actually request, read from the manifests.

    The load-bearing half of why a calibrated method still leaves the gate refused, and the half
    most easily reduced to a sentence. It is not a sentence: the six frozen declarations are built
    and their declared null is read back, so a manifest that later requested the calibrated
    inference would change this without anybody remembering to change a paragraph.
    """
    declarations = set()
    for duration in DURATIONS:
        for null in qualification_manifest(duration, "scale_shape_aligned").nulls:
            declarations.add((null.name, null.method, int(null.replications)))
    rows = sorted({"name": name, "method": method, "replications": replications}.items()
                  for name, method, replications in declarations)
    return {
        "declared": [dict(row) for row in rows],
        "inference": "drawn surrogates with a replication count",
        "estimand": "joint_structure",
        "manifests_request_the_calibrated_inference": False,
        "read_from": "the six frozen qualification manifests, not from a restated sentence",
    }


def scale_shape_blockers(supersession: Mapping[str, Any],
                         declared: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """What stands between this gate and a pass, each with what would discharge it.

    A gate that reports only a status invites the reading that enough work turns it green. Two of
    these three would be discharged by work, and the third would not be discharged by any amount of
    measuring the same fixtures harder - so `decidable_here` is published beside each rather than
    left for a reader to infer from tone.
    """
    blockers: List[Dict[str, Any]] = []
    if supersession["status"] == "VOID":
        blockers.append({
            "blocker": "supersession_void",
            "detail": " ".join(supersession["reasons"]),
            "discharged_by": ("re-deriving the record: the predecessor's refusal was retired on "
                              "its own terms rather than superseded"),
            "decidable_here": True})
    elif supersession["status"] != "SUPERSEDED":
        blockers.append({
            "blocker": "recorded_calibration",
            "detail": " ".join(supersession["reasons"]) or "no recorded calibration passes.",
            "discharged_by": ("recording %s against this checkout's contract and source"
                              % SCALE_SHAPE_CALIBRATION_SUPERSEDING),
            "decidable_here": True})
    if not declared["manifests_request_the_calibrated_inference"]:
        blockers.append({
            "blocker": "declared_inference",
            "detail": ("every frozen scale/shape manifest declares %s; the calibrated method is "
                       "exact pool substitution over a declared partner pool, which no declared "
                       "manifest requests"
                       % ", ".join("%s at %d replications" % (row["method"], row["replications"])
                                   for row in declared["declared"])),
            "discharged_by": ("a declared manifest that requests the calibrated inference, which "
                              "is a change to the experiment declaration and not to this gate"),
            "decidable_here": True})
    blockers.append({
        "blocker": "pool_exchangeability_on_real_records",
        "detail": ("the recorded calibration is evidence that the arithmetic and the "
                   "exchangeability hold together on records built with the property by "
                   "construction. Whether an inventory of real records has it is the failure mode "
                   "TG17.15 named in advance as the one that can be violated silently"),
        "discharged_by": ("a declared admission criterion shown to hold on an inventory of real "
                          "records. That is a curation obligation, and no further measurement on "
                          "built fixtures discharges it"),
        "decidable_here": False})
    return blockers


def _calendar_gate(title: str) -> Dict[str, Any]:
    """The calendar gate, read from a recorded calibration rather than from a hopeful sentence.

    TG17.12. The measurement lives in `src.core.calibration_record`, which decides between four
    outcomes and blocks on three of them. This gate states the outcome and the arithmetic behind
    it, and never executes the calibration: it reads a recording bound to the contract and source
    the recording was made against, so a relaxed expectation or a changed statistic returns the
    gate to `NOT_RUN` rather than letting a stale pass stand.
    """
    from src.core.calibration_record import read_calendar_calibration

    facts = read_calendar_calibration()
    applicability = facts["applicability"]
    declared = applicability["declared_plan"]
    if facts["status"] == "PASS":
        recorded = facts["recorded"]
        detail = (
            "The frozen family calibration (%s) ran outside orchestration and every case met the "
            "answer frozen with its fixture: %s. Recorded %s at %d replications on a family of "
            "%d, bound to the contract and source it was measured against. Unlike the scale/shape "
            "null this one has no enumeration ceiling - its floor of %.4g is bought with "
            "replications - and the calendar plan the manifests declare resolves %d corrected "
            "members at %d replications, where %d are required."
            % (facts["entry_point"],
               "; ".join("%s %d" % (row["case"], row["n_rejected_after_correction"])
                         for row in recorded["cases"]),
               recorded["recorded_utc"], recorded["replications"], recorded["family_size"],
               applicability["calibration_family"]["p_value_floor"],
               declared["members"], declared["replications"],
               declared["replications_required"]))
    else:
        detail = (
            "The frozen family calibration (%s) must run separately from orchestration and be "
            "recorded at %s. %s"
            % (facts["entry_point"], facts["record_path"],
               " ".join(reason[0].upper() + reason[1:] + "."
                        for reason in facts["reasons"]) or "No outcome is recorded."))
    return _gate("calendar_calibration", title, facts["status"], detail)


def _browser_gate(title: str) -> Dict[str, Any]:
    """The no-glue gate, read from a recorded browser run that this process did not perform.

    TG18.5 slice 4. The gate cannot observe a browser and does not try; it reads a recording that
    is bound to the source of every spec in the suite and to the completeness of the run, and it
    refuses on absence, staleness, a partial run, or a failure. The distinction it keeps is the
    one this module keeps everywhere: a run that has not happened for this code is `NOT_RUN`, and
    a run that happened and failed is `FAIL`.
    """
    from src.core.browser_evidence import browser_run_evidence

    facts = browser_run_evidence()
    if facts["status"] == "PASS":
        recorded = facts["recorded"]
        detail = (
            "A rendered Chromium run of all %d specs in this checkout passed %d tests with %d "
            "failed, recorded %s, bound to the source of every spec it ran and to %d produced "
            "artefacts. Measured through visible controls only; it qualifies apparatus "
            "behaviour and no scientific result."
            % (facts["specs_in_this_checkout"], recorded["passed"], recorded["failed"],
               recorded["recorded_utc"], recorded["artefacts"]))
    else:
        detail = (
            "Must be measured by a rendered browser test through visible controls and recorded "
            "at %s. %s" % (facts["record_path"],
                           " ".join(reason[0].upper() + reason[1:] + "."
                                    for reason in facts["reasons"])))
    return _gate("browser_no_glue", title, facts["status"], detail)


def _extension_gate(title: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """The fifth-adapter gate, half read live from source and half read from a recorded run.

    TG17.13. The split is the interesting part. The source-edit audit is decidable here, exactly,
    from committed source, so it is re-run live rather than trusted from a receipt: a recording of
    a fact that can be recomputed is only a way to be wrong later. The acceptance run cannot be
    read here at all - TG17.3 requires the fifth adapter to live in a module the application never
    imports - so it is recorded by the test and bound to the contract and source it was measured
    against. The gate's `detail` states the standing glue count even when it passes, because that
    count is reported rather than asserted and an unpublished one would let glue accumulate behind
    a green gate.
    """
    audit = facts["audit"]
    if facts["status"] == "PASS":
        recorded = facts["recorded"]
        glue = recorded["glue"]
        detail = (
            "A synthetic fifth domain with different mathematics - a monotone rank channel, not "
            "the shared standardized level - reached the registry, the control schema, the "
            "conformance kit and the domain-blind mining seam from a module the application never "
            "imports, passing all %d checks (recorded %s). No framework source names it, so "
            "installation required 0 framework edits across %d generic surfaces. Standing "
            "adapter-specific glue in those surfaces is %d and is reported rather than asserted: "
            "%s."
            % (recorded["checks"], recorded["recorded_utc"], audit["framework_sources"],
               recorded["adapter_specific_framework_edits"],
               "; ".join("%s:%d renders a bespoke planner for %s"
                         % (item["source"], item["line"], item["domain"]) for item in glue)
               or "none"))
    else:
        detail = (
            "Must be measured by the extension conformance test and source-edit audit, and "
            "recorded at %s. %s" % (facts["record_path"],
                                    " ".join(reason[0].upper() + reason[1:] + "."
                                             for reason in facts["reasons"])))
    return _gate("synthetic_fifth_adapter", title, facts["status"], detail)


def _scale_shape_gate(title: str) -> Dict[str, Any]:
    """The gate, stated from the measurement and from a supersession that is recomputed.

    TG17.15 slice 5. The status does not move, and that is the result rather than a shortfall: a
    method that is calibrated and a method a declared plan can use are two different facts, and
    this gate exists to keep them apart. What moves is the reason, from "no applicable method
    exists" to a narrower and now-measured one, and the old reason is kept beside the new one
    because it is still true when recomputed.
    """
    facts = scale_shape_applicability()
    supersession = facts["supersession"]
    superseded = supersession["superseded"]
    successor = supersession["superseding"]
    refused = [row for row in facts["declared_families"] if row["null_refusal"]]
    head = (
        "%s. The superseded claim (%s, %s) is recomputed on every plan and still holds: no family "
        "smaller than %d pairings can reject under %s at alpha %.2f, the declared null draws only "
        "from inventories of at most %d, and %d of %d declared domain families are refused by the "
        "null before size is reached."
        % (supersession["status"], superseded["record"], SCALE_SHAPE_CALIBRATION,
           facts["minimum_resolvable_family"], facts["correction"], facts["alpha"],
           facts["largest_drawable_inventory"], len(refused), len(facts["declared_families"])))
    if supersession["status"] == "SUPERSEDED":
        recorded = facts["supersession"]["superseding"]
        middle = (
            " Superseded for applicability by %s (%s, %s), recorded %s: %d correspondences need a "
            "pool of %d, which the calibration's own pools reach."
            % (successor["record"], successor["estimand"],
               SCALE_SHAPE_CALIBRATION_SUPERSEDING, recorded["recorded_utc"],
               successor["tested_correspondences"], successor["minimum_pool_size"]))
    else:
        middle = (
            " Not superseded: %s."
            % (" ".join(supersession["reasons"]).rstrip(".")
               or "no successor recording passes"))
    tail = " Blocked by %s." % "; ".join(
        "%s (%s)" % (row["blocker"],
                     "decidable here" if row["decidable_here"] else "not decidable here")
        for row in facts["blockers"])
    return _gate("scale_shape_calibration", title, "REFUSED", head + middle + tail)


def _live_source_gate(title: str, facts: Mapping[str, Any]) -> Dict[str, Any]:
    """Report a dated external measurement without reaching the network during assembly."""
    if facts["status"] == "PASS":
        detail = (
            "A separately dated four-domain source run passed for the source identities in the "
            "flagship manifest, recorded %s. Reanalysis, Argo and TESS used their declared "
            "archives; the bespoke order-book family used a content-addressed local record and "
            "did not pretend to have a public archive."
            % facts["recorded"]["recorded_utc"])
    else:
        detail = (
            "Requires an explicit opt-in run covering three declared archives and one "
            "content-addressed bespoke local record, recorded at %s. %s"
            % (facts["record_path"],
               " ".join(reason[0].upper() + reason[1:] + "."
                        for reason in facts["reasons"])))
    return _gate("live_sources", title, facts["status"], detail)


def qualification_plan() -> Dict[str, Any]:
    """The complete gate before anything is executed; omissions are impossible to hide."""
    from src.core.browser_evidence import browser_run_evidence, scientist_action_evidence
    from src.core.calibration_record import (
        read_calendar_calibration,
        read_scale_shape_calibration,
    )
    from src.core.extension_evidence import read_extension_conformance
    from src.core.live_source_evidence import read_live_source_evidence

    # Read once and passed to the gate: the audit walks seventeen sources, and doing that twice
    # per plan would double an HTTP route's cost for an identical answer.
    extension = read_extension_conformance()
    live_sources = read_live_source_evidence()

    cells = []
    for duration in DURATIONS:
        for mode in MODES:
            spec = qualification_manifest(duration, mode)
            window = spec.windows[0]
            cells.append({
                "cell_id": "%s:%s" % (duration, mode),
                "duration": duration,
                "mode": mode,
                "start_utc": window.start_utc.isoformat().replace("+00:00", "Z"),
                "end_utc": window.end_utc.isoformat().replace("+00:00", "Z"),
                "manifest_sha256": manifest_sha256(spec),
                "family_correction": spec.correction,
                "record_kind": RECORD_KIND,
                "status": "NOT_RUN",
            })
    gates = [
        _gate("offline_matrix", "Six-cell known-answer apparatus matrix", "NOT_RUN",
              "Run all three frozen durations in both comparison modes."),
        _gate("restart_recovery", "Single remote-failure restart recovery", "NOT_RUN",
              "Resume the same manifest after a process-boundary reload and retry only the "
              "failed component."),
        _browser_gate("Clean-browser no-glue path"),
        _extension_gate("Synthetic fifth-adapter no-edit test", extension),
        _calendar_gate("Calendar null calibration and planted power"),
        _scale_shape_gate("Scale/shape null calibration and planted power"),
        _live_source_gate("Four-domain live-source tail", live_sources),
    ]
    result = {
        "schema": SCHEMA,
        "qualification_sha256": "",
        "verdict": "NOT_RELEASEABLE",
        "record_kind": RECORD_KIND,
        "matrix": cells,
        "gates": gates,
        "calendar_calibration": read_calendar_calibration(),
        "scale_shape_calibration": scale_shape_applicability(),
        # TG17.15 slice 5. Kept beside the applicability section rather than folded into it: what
        # the calibration measured is one fact and whether a declared plan can reach the method is
        # another, and this module has said since TG17.11 that reporting the first as the second
        # is how an unusable mode acquires a green gate.
        "scale_shape_evidence": read_scale_shape_calibration(),
        # TG18.5 slice 4: measured by a rendered run or reported as unmeasured, never invented
        # here. `adapter_specific_framework_edits` stays NOT_MEASURED in both cases - a browser
        # cannot observe a source-edit audit. TG17.13 supplies that number from the audit itself,
        # under `extension_evidence`, where the two halves of the fifth-adapter claim are kept
        # apart: what this installation required (0, asserted) and what stands in the generic
        # surfaces today (reported).
        "scientist_actions": scientist_action_evidence(),
        "browser_evidence": browser_run_evidence(),
        "extension_evidence": extension,
        "live_source_evidence": live_sources,
        "claim_boundary": (
            "This plan and its offline rehearsal qualify apparatus behaviour only. They are "
            "not acquired observations, scientific results, evidence, replication or claims."),
    }
    result["qualification_sha256"] = _digest({**result, "qualification_sha256": ""})
    return result


def _cell_result(spec: CrossDomainExperimentSpec, root: Path) -> Dict[str, Any]:
    preflight = preflight_manifest(spec)
    window = spec.windows[0]
    if preflight["refusals"]:
        # No run is opened for a plan the registered declarations refuse. Opening one would
        # manufacture a run identity for an experiment that may not be conducted, and a later
        # reader could not tell a refused plan from an unexecuted one.
        return {
            "cell_id": "%s:%s" % (window.name, spec.mode),
            "duration": window.name,
            "mode": spec.mode,
            "start_utc": window.start_utc.isoformat().replace("+00:00", "Z"),
            "end_utc": window.end_utc.isoformat().replace("+00:00", "Z"),
            "manifest_sha256": manifest_sha256(spec),
            "preflight_status": preflight["status"],
            "family_correction": spec.correction,
            "record_kind": RECORD_KIND,
            "refusals": list(preflight["refusals"]),
            "status": "REFUSED",
        }
    run = RunStore(root).open(spec)
    receipt = run.execute(build_suite("fixture_dry_run", run=run))
    exported = publish_bundle(run, root)["bundle"]
    replay = verify_bundle(exported)
    checks = {
        "no_preflight_refusal": not preflight["refusals"],
        "complete_run": receipt["state"] == "COMPLETE",
        "same_manifest_everywhere": (
            preflight["manifest_sha256"] == manifest_sha256(spec)
            == exported["run_identity"]["manifest_sha256"]
            == replay["run_receipt"]["manifest_sha256"]),
        "receipt_integrity_verified": replay["integrity"] == "VERIFIED",
        "fixture_not_measured": (
            exported["results"]["measured"] is False
            and exported["results"]["artefacts"] == {}),
        "no_evidence_or_claim_promotion": (
            exported["evidence_handoff"]["automatic_actions"] == []
            and all(row["status"] == "ABSENT"
                    for row in exported["evidence_handoff"]["categories"]
                    if row["category"] in ("measured_results", "registered_hypothesis",
                                           "admitted_evidence", "independent_replication",
                                           "claim_promotion"))),
    }
    return {
        "cell_id": "%s:%s" % (window.name, spec.mode),
        "duration": window.name,
        "mode": spec.mode,
        "start_utc": window.start_utc.isoformat().replace("+00:00", "Z"),
        "end_utc": window.end_utc.isoformat().replace("+00:00", "Z"),
        "manifest_sha256": manifest_sha256(spec),
        "run_id": receipt["run_id"],
        "bundle_sha256": exported["bundle_sha256"],
        "preflight_status": preflight["status"],
        "family_correction": spec.correction,
        "record_kind": RECORD_KIND,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _recovery_result(root: Path) -> Dict[str, Any]:
    spec = qualification_manifest("week", "calendar_aligned", suffix="recovery")
    first_store = RunStore(root)
    run = first_store.open(spec)
    if run.state not in ("FAILED", "COMPLETE"):
        run.execute(build_suite("fixture_single_remote_failure", run=run))

    # A fresh store and loaded run is the process-boundary contract: the worker's memory is gone.
    resumed = RunStore(root).load(run.run_id)
    if resumed.state == "FAILED":
        resumed.retry(build_suite("fixture_single_remote_failure", run=resumed))
    final = resumed.receipt()
    failed_component = "tess_lightcurve"
    attempts_before = {item.domain: (1 if item.domain == failed_component else 1)
                       for item in spec.observations}
    attempts_after = {item.domain: resumed.attempts("ACQUIRING", item.domain)
                      for item in spec.observations}
    transitions = [event.get("to_state") for event in resumed.events
                   if event.get("kind") == "transition"]
    retries = [event for event in resumed.events if event.get("kind") == "retry"]
    checks = {
        "first_attempt_failed_operationally": "FAILED" in transitions,
        "same_run_after_restart": final["run_id"] == run.run_id,
        "retry_completed": final["state"] == "COMPLETE",
        "retry_named_only_failed_component": (
            len(retries) == 1 and retries[0].get("components") == [failed_component]),
        "only_failed_component_retried": (
            attempts_after == {name: 2 if name == failed_component else 1
                               for name in attempts_before}),
        "replay_verified_after_recovery": (
            verify_bundle(publish_bundle(resumed, root)["bundle"])["integrity"] == "VERIFIED"),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "run_id": final["run_id"], "failed_component": failed_component,
            "attempts_before_restart": attempts_before,
            "attempts_after_retry": attempts_after, "checks": checks}


def execute_offline_qualification(root: Path) -> Dict[str, Any]:
    """Execute only the deterministic gates and return an honestly blocked release record."""
    root = Path(root)
    matrix = [_cell_result(qualification_manifest(duration, mode), root)
              for duration in DURATIONS for mode in MODES]
    recovery = _recovery_result(root)
    plan = qualification_plan()
    gates = []
    for row in plan["gates"]:
        if row["gate_id"] == "offline_matrix":
            passed = sum(cell["status"] == "PASS" for cell in matrix)
            refused = [cell for cell in matrix if cell["status"] == "REFUSED"]
            if passed == len(matrix):
                status = "PASS"
            elif refused and passed + len(refused) == len(matrix):
                # Not a broken apparatus: a declared scientific refusal the matrix cannot
                # overrule. It blocks release exactly as a failure does, and says why.
                status = "REFUSED"
            else:
                status = "FAIL"
            detail = "%d of %d deterministic cells passed." % (passed, len(matrix))
            if refused:
                reasons = sorted({str(item.get("reason", "")) for cell in refused
                                  for item in cell["refusals"]})
                detail += " %d refused before execution: %s" % (
                    len(refused), " ".join(reasons))
            gates.append(_gate(row["gate_id"], row["title"], status, detail))
        elif row["gate_id"] == "restart_recovery":
            gates.append(_gate(row["gate_id"], row["title"], recovery["status"],
                               "One acquisition component timed out; a fresh store resumed the "
                               "same run and retried only that component."))
        else:
            gates.append(row)
    record = {**plan, "qualification_sha256": "", "matrix": matrix, "gates": gates,
              "recovery": recovery}
    record["verdict"] = ("RELEASEABLE" if all(row["status"] == "PASS" for row in gates)
                         else "NOT_RELEASEABLE")
    record["qualification_sha256"] = _digest(record)
    return record


def verify_qualification_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify the record's outer seal without turning its unrun gates into passes."""
    supplied = str(record.get("qualification_sha256", ""))
    body = dict(record)
    body["qualification_sha256"] = ""
    expected = _digest(body)
    if supplied != expected:
        raise ValueError("qualification_sha256 does not match the complete qualification record")
    if record.get("verdict") == "RELEASEABLE" and any(
            row.get("status") != "PASS" for row in record.get("gates", [])):
        raise ValueError("a RELEASEABLE qualification record contains a gate that did not pass")
    return {"integrity": "VERIFIED", "qualification_sha256": supplied,
            "verdict": record.get("verdict"), "gates": list(record.get("gates", []))}


__all__ = ["DURATIONS", "MODES", "RECORD_KIND", "SCALE_SHAPE_CALIBRATION",
           "SCALE_SHAPE_CALIBRATION_SUPERSEDING", "declared_scale_shape_null",
           "scale_shape_blockers",
           "SCALE_SHAPE_DOMAIN_FAMILIES", "SCHEMA", "execute_offline_qualification",
           "qualification_manifest", "qualification_plan", "scale_shape_applicability",
           "verify_qualification_record"]
