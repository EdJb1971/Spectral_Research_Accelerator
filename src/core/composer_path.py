"""TG17.7 The guided path through one experiment, as a served contract rather than a layout.

**The failure this exists to prevent.** A workbench that shows four acquisition pages and a page
of instructions leaves the order of operations in the researcher's head, and the order of
operations *is* the scientific discipline: the family is priced before acquisition because a
family priced afterwards is priced knowing what the data looked like; the null is admitted by the
domain before the p-value exists because a null chosen after seeing the statistic is not a null.
A UI that lets those steps happen in any order has not made an error yet - it has made the error
undetectable, because a receipt cannot tell an experiment that was planned from one that was
assembled.

So the path is not a layout decision. `COMPOSER_PATH` is a registry of seven steps, each of which
decides its own status from the manifest, and `compose_state` returns exactly **one** next
legitimate action. The browser renders that; it does not compute it. Three consequences follow:

*   A step's precondition is stated once, in the step. A new step registers rather than being
    remembered in a component, a route and a test.
*   "What may I do now" has a single answer with a single owner. Two enabled buttons that mean
    two different scientific commitments cannot both be the next action, because `next_action`
    is one field.
*   An unavailable choice keeps its reason. `domain_menu` and `window_presets` return what is
    refused *with the refusal*, because a control that disappears when it becomes inadmissible
    teaches the researcher that it never existed.

**The ladder is the other half.** `stage_ladder` names four things a researcher can possess -
acquired material, an executed run, a finding, admitted evidence - and reports which are reached.
They are separated because the slide between them is the cheapest mistake in this repository:
downloading four archives feels like having a study, and a completed run feels like a result. No
rung is reached by doing the previous one; each has its own gate, and the ladder says so in the
same breath as it reports progress.

This module opens no archive, runs no statistic and admits no evidence. Every status it reports
is a fact about a *declaration*.
"""

from __future__ import annotations

import calendar
import hashlib
import json
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError, UserInputError
from src.core.experiment_adapter import EXPERIMENT_ADAPTERS, adapter_for_domain
from src.core.experiment_manifest import (CrossDomainExperimentSpec, canonical_bytes,
                                          manifest_sha256, preflight_manifest)
from src.core.registry import Registry
from src.core.structural_alignment import MODE_FORBIDS


SCHEMA = "composer-path/v1"

#: A step is satisfied, is the thing to do next, or cannot be decided until an earlier step is.
#: Three values rather than two, because "you have not done this yet" and "this cannot be done
#: yet" are different sentences and only one of them is the researcher's move.
STEP_STATUSES: Tuple[str, ...] = ("SATISFIED", "ACTION_REQUIRED", "BLOCKED")

#: Durations the composer offers by name, as (unit, amount). A preset is a **label for explicit
#: boundaries**, never a boundary itself: `window_presets` returns the UTC instants it resolved
#: to, and the manifest stores those. A manifest that stored "six months" would mean different
#: things on different days, and its content address would not change when it did.
#:
#: The months are calendar months rather than a fixed day count, which is not a nicety. The
#: flagship recipe declares 2026-01-01 to 2026-07-01, and a "six months" preset defined as 182
#: days resolves that same anchor to 2026-07-02 - so a researcher pressing the preset that
#: describes their own window would silently move its boundary and re-address the manifest.
DURATION_PRESETS: Dict[str, Tuple[str, int]] = {
    "week": ("days", 7), "three_months": ("months", 3), "six_months": ("months", 6)}


def advance(anchor: datetime, unit: str, amount: int) -> datetime:
    """One preset duration after `anchor`, in calendar terms where the preset is calendar."""
    if unit == "days":
        return anchor + timedelta(days=amount)
    month_index = anchor.month - 1 + amount
    year = anchor.year + month_index // 12
    month = month_index % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return anchor.replace(year=year, month=month, day=day)


class EnvelopeIntegrityError(UserInputError):
    """An imported manifest envelope disagrees with its own digest.

    Import is the one place a manifest can arrive from outside this instance, so it is the one
    place the content address can be wrong. Accepting it would give a run an identity that does
    not address its own plan.
    """


# ---------------------------------------------------------------------------- state


@dataclass(frozen=True)
class ComposerState:
    """Everything the path decides from: the plan, and the facts about it that exist.

    `run` is the run receipt when one has been opened and `None` when none has. It is passed in
    rather than looked up because the path must be decidable for a manifest that has never been
    executed - which is every manifest, until it is not.
    """

    spec: CrossDomainExperimentSpec
    preflight: Mapping[str, Any]
    saved: bool = False
    run: Optional[Mapping[str, Any]] = None

    @property
    def run_state(self) -> str:
        return str(self.run.get("state", "")) if self.run else ""

    def refusals_mentioning(self, *fragments: str) -> List[str]:
        """Preflight refusals whose text names one of these subjects.

        Deliberately a filter rather than a category on the refusal: a refusal's job is to be
        read by a person, and forcing every existing refusal to acquire a machine tag before
        this view could exist would have meant editing eight call sites to render one panel.
        The steps below never rely on this alone - each decides from the manifest first and
        uses the matching refusals to say *why* in the archive's own words.
        """
        out = []
        for refusal in self.preflight.get("refusals", []):
            text = str(refusal.get("reason", ""))
            if any(fragment in text for fragment in fragments):
                out.append(("%s: %s" % (refusal["domain"], text)) if refusal.get("domain") else text)
        return out


@dataclass(frozen=True)
class StepVerdict:
    status: str
    reason: str
    detail: Dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STEP_STATUSES:
            raise InvalidParameterError("StepVerdict.status", self.status,
                                        "one of %s" % ", ".join(STEP_STATUSES))


@dataclass(frozen=True)
class PathStep:
    """One decision the researcher makes, with the question it settles and its own gate."""

    step_id: str
    ordinal: int
    title: str
    question: str
    settles: str
    controls: Tuple[str, ...]
    action_label: str
    action_route: str
    claim_boundary: str
    decide: Callable[[ComposerState], StepVerdict]

    def describe(self) -> Dict[str, Any]:
        return {"step_id": self.step_id, "ordinal": self.ordinal, "title": self.title,
                "question": self.question, "settles": self.settles,
                "controls": list(self.controls), "action_label": self.action_label,
                "action_route": self.action_route, "claim_boundary": self.claim_boundary}


#: The seven steps (standard E1). Registered rather than listed, so the browser, the state
#: endpoint and the receipt all read one description of the workflow.
COMPOSER_PATH: Registry[PathStep] = Registry("composer path step")


def register_step(step: PathStep, *, replace: bool = False) -> PathStep:
    COMPOSER_PATH.add(step.step_id, step, description=step.question,
                      capabilities={"ordinal": step.ordinal}, replace=replace)
    return step


def ordered_steps() -> List[PathStep]:
    """The path in the order it must be walked, not in the order names happen to sort."""
    return sorted((entry.value for entry in COMPOSER_PATH.entries()), key=lambda s: s.ordinal)


# ---------------------------------------------------------------------- the seven gates


def _decide_question(state: ComposerState) -> StepVerdict:
    """Settled by the manifest parsing at all: a spec carries exactly one comparison mode."""
    return StepVerdict("SATISFIED",
                       "Comparing in %s mode. %s" % (state.spec.mode, MODE_FORBIDS[state.spec.mode]),
                       {"mode": state.spec.mode, "forbids": MODE_FORBIDS[state.spec.mode],
                        "coverage_requirement": state.spec.coverage_policy.requirement})


def _decide_domains(state: ComposerState) -> StepVerdict:
    """Two or more registered domains, each carrying the assumptions it breaks."""
    rows, unregistered = [], []
    for observation in state.spec.observations:
        try:
            adapter = adapter_for_domain(observation.domain)
        except Exception as error:  # a manifest may name a domain this instance does not have
            unregistered.append(observation.domain)
            rows.append({"domain": observation.domain, "registered": False,
                         "breaks": [], "reason": str(error)})
            continue
        rows.append({"domain": observation.domain, "registered": True,
                     "label": observation.label,
                     "breaks": list(adapter.declaration.violations),
                     "licence": adapter.declaration.licence,
                     "precedence_admissible": adapter.declaration.precedence_admissible,
                     "reason": ""})
    if unregistered:
        return StepVerdict("BLOCKED",
                           "This instance has no adapter for %s, so what those domains refuse is "
                           "unknown and cannot be inherited." % ", ".join(sorted(unregistered)),
                           {"domains": rows})
    breaks = sorted({name for row in rows for name in row["breaks"]})
    return StepVerdict("SATISFIED",
                       "%d domains declared, breaking %d distinct assumptions between them: %s."
                       % (len(rows), len(breaks), ", ".join(breaks) or "none"),
                       {"domains": rows, "assumptions_broken": breaks})


def _decide_observation(state: ComposerState) -> StepVerdict:
    """Every window explicit, and every adapter's controls resolving under its own schema."""
    refused = []
    for observation in state.spec.observations:
        try:
            adapter_for_domain(observation.domain).resolve(observation.adapter.parameters)
        except Exception as error:
            refused.append({"domain": observation.domain, "reason": str(error)})
    windows = [{"name": w.name, "start_utc": w.start_utc.isoformat(),
                "end_utc": w.end_utc.isoformat(),
                "days": round((w.end_utc - w.start_utc).total_seconds() / 86400.0, 4),
                "preset": _preset_name_for(w.start_utc, w.end_utc)}
               for w in state.spec.windows]
    if refused:
        return StepVerdict("ACTION_REQUIRED",
                           "%d domain control set(s) do not satisfy their adapter's own schema."
                           % len(refused), {"refused": refused, "windows": windows})
    return StepVerdict("SATISFIED",
                       "%d explicit UTC window(s) and %d resolved domain control set(s)."
                       % (len(windows), len(state.spec.observations)),
                       {"refused": [], "windows": windows})


def _decide_preflight(state: ComposerState) -> StepVerdict:
    """The metadata verdict, read rather than re-derived - and partial is the policy's call."""
    status = str(state.preflight.get("status", ""))
    reasons = [(("%s: %s" % (row["domain"], row["reason"])) if row.get("domain") else row["reason"])
               for row in state.preflight.get("refusals", [])]
    detail = {"status": status, "refusals": reasons,
              "coverage": list(state.preflight.get("coverage", [])),
              "planned_bytes": state.preflight.get("planned_bytes"),
              "network_used": state.preflight.get("network_used")}
    if status == "REFUSED":
        return StepVerdict("BLOCKED",
                           "Metadata refuses this plan for %d reason(s). Acquiring anyway would "
                           "buy bytes the declared analysis cannot use." % len(reasons), detail)
    if status == "PARTIAL":
        permitted = state.spec.coverage_policy.requirement == "partial_permitted"
        return StepVerdict("SATISFIED" if permitted else "ACTION_REQUIRED",
                           "Metadata establishes partial coverage only. The frozen coverage rule "
                           "%s it." % ("admits" if permitted else "does not admit"), detail)
    return StepVerdict("SATISFIED",
                       "Metadata coverage is ready. No network was used and no measurement value "
                       "was opened.", detail)


def _decide_analysis(state: ComposerState) -> StepVerdict:
    """The family is priced, the correction can resolve it, and every domain admits the null."""
    family = dict(state.preflight.get("family", {}))
    inadmissible = []
    for null in state.spec.nulls:
        for observation in state.spec.observations:
            try:
                adapter = adapter_for_domain(observation.domain)
            except Exception:
                continue
            if null.method not in adapter.admissible_nulls:
                inadmissible.append({"domain": observation.domain, "null": null.method})
    detail = {"family": family, "correction": state.spec.correction, "alpha": state.spec.alpha,
              "nulls": [{"name": n.name, "method": n.method, "replications": n.replications}
                        for n in state.spec.nulls],
              "confirmation": state.spec.confirmation.dict(),
              "seeds": dict(state.spec.seeds),
              "inadmissible_nulls": inadmissible,
              "refusals": state.refusals_mentioning("family", "null", "correction", "cap")}
    if inadmissible:
        return StepVerdict("ACTION_REQUIRED",
                           "A null must be admitted by every domain it is applied to; %d "
                           "domain/null pair(s) are not." % len(inadmissible), detail)
    if family and not dict(family.get("correction", {})).get("affordable", True):
        return StepVerdict("ACTION_REQUIRED",
                           "The declared family cannot be resolved at the declared ensemble. It "
                           "would run, cost the full amount and reject nothing.", detail)
    return StepVerdict("SATISFIED",
                       "%s members corrected by %s at alpha %s, against %d null replication(s)."
                       % (family.get("declared_members", "an unpriced number of"),
                          state.spec.correction, state.spec.alpha,
                          state.spec.nulls[0].replications), detail)


def _decide_freeze_and_run(state: ComposerState) -> StepVerdict:
    """Freezing is the commitment; after it, the run's own state machine is the truth."""
    detail = {"saved": state.saved, "run_state": state.run_state,
              "run_id": (state.run or {}).get("run_id"),
              "manifest_sha256": manifest_sha256(state.spec)}
    if not state.run:
        return StepVerdict("ACTION_REQUIRED",
                           "Not opened. Opening the run posts this manifest; the run identity is "
                           "its content address, so this is resumable and never duplicated.",
                           detail)
    if state.run_state == "COMPLETE":
        return StepVerdict("SATISFIED", "The frozen plan ran to completion.", detail)
    if state.run_state in ("REFUSED", "CANCELLED"):
        return StepVerdict("BLOCKED",
                           "The run is %s and is terminal. A frozen plan is not edited after "
                           "seeing how it went; an editable copy is." % state.run_state, detail)
    return StepVerdict("ACTION_REQUIRED",
                       "The run is %s. It resumes rather than restarts." % state.run_state, detail)


def _decide_interpret(state: ComposerState) -> StepVerdict:
    """Never satisfied by this surface: interpretation ends outside the composer, by design."""
    if state.run_state != "COMPLETE":
        return StepVerdict("BLOCKED",
                           "There is nothing to interpret until the frozen plan has run.",
                           {"run_state": state.run_state})
    return StepVerdict("ACTION_REQUIRED",
                       "The run is complete. A completed run is an executed plan; turning it into "
                       "a finding is a separate, recorded act.",
                       {"run_state": state.run_state,
                        "artefacts": len((state.run or {}).get("artefacts", {}))})


register_step(PathStep(
    step_id="question", ordinal=1, title="Question",
    question="Calendar-aligned co-occurrence, or scale and shape recurrence?",
    settles="Which comparison is being made, and therefore what may never be claimed from it.",
    controls=("Study identity", "Comparison mode", "Coverage rule"),
    action_label="Choose the comparison mode",
    action_route="POST /api/v1/experiment-composer/manifests/validate",
    claim_boundary="Choosing a mode fixes a claim boundary. It does not license the claim.",
    decide=_decide_question))

register_step(PathStep(
    step_id="domains", ordinal=2, title="Domains",
    question="Which two or more sources, and what does each one break?",
    settles="Whose assumptions the comparison inherits.",
    controls=("Domain selection", "Assumptions this domain breaks", "Licence"),
    action_label="Select two or more registered domains",
    action_route="GET /api/v1/experiment-composer/domain-menu",
    claim_boundary="A registered adapter is a declared contract, not proof its archive was read.",
    decide=_decide_domains))

register_step(PathStep(
    step_id="observation", ordinal=3, title="Observation",
    question="Which UTC windows, and which domain-native measure and role in each?",
    settles="Exactly which observations the plan addresses.",
    controls=("Window start and end", "Duration preset", "Adapter controls"),
    action_label="Resolve every adapter's declared controls",
    action_route="GET /api/v1/experiment-composer/window-presets",
    claim_boundary="A preset is a label for explicit boundaries; the boundaries are what is stored.",
    decide=_decide_observation))

register_step(PathStep(
    step_id="preflight", ordinal=4, title="Preflight",
    question="What does metadata alone say about coverage, gaps, cost and remedies?",
    settles="Whether this plan is worth acquiring anything for.",
    controls=("Inspect metadata coverage", "Per-domain coverage", "Refusals and remedies"),
    action_label="Inspect metadata coverage",
    action_route="POST /api/v1/experiment-composer/manifests/preflight",
    claim_boundary="Coverage planning is not acquisition, analysis, evidence or a result.",
    decide=_decide_preflight))

register_step(PathStep(
    step_id="analysis", ordinal=5, title="Analysis",
    question="Which benchmarked recipe, which null, which correction and which confirmation?",
    settles="The complete tested family, before anything is measured.",
    controls=("Null family", "Alignment kernel", "Price the declared family",
              "Correction", "Seeds", "Confirmation design"),
    action_label="Price the declared family and admit its null",
    action_route="POST /api/v1/experiment-composer/manifests/family",
    claim_boundary="An affordable family is arithmetic, not evidence.",
    decide=_decide_analysis))

register_step(PathStep(
    step_id="freeze_and_run", ordinal=6, title="Freeze and run",
    question="Is this exactly the experiment you are committing to?",
    settles="The plan, permanently. After this the manifest is the run identity.",
    controls=("Preregistration summary", "Open or resume the run", "Execute the frozen plan",
              "Retry the operational failure", "Cancel the run"),
    action_label="Open or resume the run",
    action_route="POST /api/v1/experiment-runs",
    claim_boundary="A completed run is an executed plan, not admitted evidence.",
    decide=_decide_freeze_and_run))

register_step(PathStep(
    step_id="interpret", ordinal=7, title="Interpret",
    question="What did the executed plan produce, and what may be said about it?",
    settles="Nothing on its own: a finding and admitted evidence are separate recorded acts.",
    controls=("Native records", "Canonical structure", "Corrected comparisons", "Limitations"),
    action_label="Review the receipt and its limitations",
    action_route="GET /api/v1/experiment-runs/{run_id}",
    claim_boundary="Reading a receipt is not making a finding, and a finding is not evidence.",
    decide=_decide_interpret))


# ------------------------------------------------------------------------- the ladder


#: Four things a researcher can possess, and the gate on each. They are listed together because
#: the mistake is never in one of them - it is in the slide from one to the next.
STAGE_LADDER: Tuple[Dict[str, str], ...] = (
    {"rung": "acquired_material", "title": "Acquired material",
     "is": "Bytes from an archive, retained with their provenance.",
     "is_not": "A study. Downloading four archives is not an experiment about them.",
     "gate": "An acquisition step of a frozen run completes."},
    {"rung": "experiment_run", "title": "Executed run",
     "is": "A frozen plan carried out, with a receipt naming everything it did not produce.",
     "is_not": "A result. Running the plan you preregistered says nothing about what it found.",
     "gate": "Every stage of the run reaches COMPLETE."},
    {"rung": "finding", "title": "Finding",
     "is": "A claim someone recorded from an executed run, at a stated rung of the claim ladder.",
     "is_not": "Admitted evidence. A finding is an assertion with an author.",
     "gate": "A finding is recorded against the run through the findings instrument."},
    {"rung": "admitted_evidence", "title": "Admitted evidence",
     "is": "A finding that survived confirmation on the held-out partition and adversarial review.",
     "is_not": "A conclusion about the world; it remains bounded by the mode's claim boundary.",
     "gate": "The evidence bundle is admitted after review."},
)


def stage_ladder(state: ComposerState) -> List[Dict[str, Any]]:
    """Which rungs this state has reached, and why the others have not.

    Only the first two can be reached from this surface at all. That is the point: the composer
    can show you that you are two gates away from evidence, and it cannot move you across them.
    """
    run = dict(state.run or {})
    artefacts = dict(run.get("artefacts", {}))
    acquired = [key for key in artefacts if key.startswith("ACQUIRING/")]
    reached = {
        "acquired_material": bool(acquired),
        "experiment_run": run.get("state") == "COMPLETE",
        "finding": False,
        "admitted_evidence": False,
    }
    why_not = {
        "acquired_material": "No acquisition step of a run has completed."
                             if not acquired else "",
        "experiment_run": "The run is %s." % (run.get("state") or "not opened")
                          if run.get("state") != "COMPLETE" else "",
        "finding": "This surface does not record findings; the findings instrument does.",
        "admitted_evidence": "Evidence is admitted after confirmation and review, elsewhere.",
    }
    rows = []
    for rung in STAGE_LADDER:
        row = dict(rung)
        row["reached"] = reached[rung["rung"]]
        row["why_not"] = why_not[rung["rung"]]
        rows.append(row)
    return rows


# -------------------------------------------------------------------------- composing


def compose_state(spec: CrossDomainExperimentSpec, *, saved: bool = False,
                  run: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Walk the path over one manifest and name the single next legitimate action.

    The preflight is computed here rather than passed in, because a step status must be a fact
    about the plan and not a fact about which button was clicked last. Metadata-only, so this
    costs nothing a researcher would hesitate to spend.
    """
    state = ComposerState(spec=spec, preflight=preflight_manifest(spec), saved=saved, run=run)
    steps, next_action = [], None
    for step in ordered_steps():
        verdict = step.decide(state)
        steps.append({**step.describe(), "status": verdict.status, "reason": verdict.reason,
                      "detail": verdict.detail})
        if next_action is None and verdict.status != "SATISFIED":
            next_action = {"step_id": step.step_id, "label": step.action_label,
                           "route": step.action_route, "status": verdict.status,
                           "why": verdict.reason,
                           "blocked": verdict.status == "BLOCKED"}
    return {"schema": SCHEMA, "manifest_sha256": manifest_sha256(spec),
            "study_id": spec.study_id, "steps": steps,
            "satisfied": sum(1 for row in steps if row["status"] == "SATISFIED"),
            "next_action": next_action, "ladder": stage_ladder(state),
            "run_state": state.run_state or None,
            "claim_boundary": "The path reports what a declaration permits next. Reaching the end "
                              "of it produces an executed plan, not a finding and not evidence."}


# ---------------------------------------------------------------------------- controls


def _preset_name_for(start: datetime, end: datetime) -> Optional[str]:
    """The preset a window happens to match, or `None` for a window that matches none.

    A window is never *stored* as a preset, so this is presentation only - it lets the composer
    show a researcher that the boundaries they typed are a week, without the manifest acquiring
    a field whose meaning depends on when it is read.
    """
    for name, (unit, amount) in DURATION_PRESETS.items():
        if advance(start, unit, amount) == end:
            return name
    return None


def window_presets(anchor_utc: str, *, stride_seconds: int = 3600) -> Dict[str, Any]:
    """Resolve every offered duration against one anchor, on the server.

    The arithmetic lives here rather than in the browser for the same reason the family size
    does: two implementations of a boundary are two boundaries, and the one a researcher reads
    would not be the one the manifest addresses.
    """
    try:
        anchor = datetime.fromisoformat(anchor_utc.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise InvalidParameterError("anchor_utc", anchor_utc,
                                    "an ISO-8601 UTC instant such as 2024-01-01T00:00:00Z")
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    anchor = anchor.astimezone(timezone.utc)
    if stride_seconds <= 0:
        raise InvalidParameterError("stride_seconds", stride_seconds, "a positive stride")
    rows = []
    for name, (unit, amount) in DURATION_PRESETS.items():
        end = advance(anchor, unit, amount)
        rows.append({"preset": name, "unit": unit, "amount": amount,
                     "days": round((end - anchor).total_seconds() / 86400.0, 6),
                     "start_utc": anchor.isoformat().replace("+00:00", "Z"),
                     "end_utc": end.isoformat().replace("+00:00", "Z"),
                     "stride_seconds": stride_seconds,
                     "label": "%s from the anchor" % name.replace("_", " ")})
    return {"schema": "composer-window-presets/v1", "anchor_utc": anchor.isoformat().replace("+00:00", "Z"),
            "presets": rows,
            "note": ("A preset is a label. What is stored, hashed and acquired is the pair of "
                     "explicit UTC instants shown beside it."),
            "claim_boundary": "A window is a request. Whether the archive covers it is preflight's "
                              "answer, not this route's."}


#: Why a registered adapter can still be unavailable as a study member. Stated once, because
#: the reason has to travel with the disabled control and it is a scientific reason rather than
#: a UI one: an adapter says how a domain would be translated, and a declared observation says
#: what is measured, in which units, in which role, from which record. The second is not
#: derivable from the first, and a menu that guessed it would be inventing the observation.
NO_TEMPLATE_REASON = (
    "no declared observation is registered for this domain. An adapter says how a domain would "
    "be translated; it does not say what is measured, in which units, in which role or from "
    "which record. That is declared by a recipe, and a menu that guessed it would be inventing "
    "the observation rather than offering it.")


def observation_templates() -> Dict[str, Dict[str, Any]]:
    """The declared observation for each domain a registered recipe already describes."""
    from src.core.experiment_manifest import flagship_recipe

    return {row.domain: json.loads(row.json()) for row in flagship_recipe().observations}


def domain_menu(*, selected: Sequence[str] = ()) -> Dict[str, Any]:
    """Every registered domain, what it breaks, and - when it cannot be chosen - why.

    Nothing is filtered out. A domain that cannot join this study is shown disabled beside the
    reason it cannot, because a control that vanishes when it becomes inadmissible is
    indistinguishable from one that was never offered.
    """
    chosen = set(selected)
    templates = observation_templates()
    rows = []
    for entry in EXPERIMENT_ADAPTERS.entries():
        adapter = entry.value
        declaration = adapter.declaration
        template = templates.get(adapter.domain)
        rows.append({
            "domain": adapter.domain, "adapter_id": adapter.adapter_id,
            "label": declaration.description, "licence": declaration.licence,
            "breaks": list(declaration.violations),
            "lag_policy": declaration.lag_policy,
            "precedence_admissible": declaration.precedence_admissible,
            "admissible_kernels": list(adapter.admissible_kernels),
            "admissible_nulls": list(adapter.admissible_nulls),
            "selected": adapter.domain in chosen,
            "selectable": template is not None,
            "unavailable_reason": "" if template is not None else NO_TEMPLATE_REASON,
            "observation": template,
            "onboarding_cost": dict(adapter.onboarding_cost),
        })
    return {"schema": "composer-domain-menu/v1", "domains": rows,
            "minimum_domains": 2,
            "note": ("A second domain is what makes a comparison cross-domain; the assumptions "
                     "column is what it costs. Every row states what that domain breaks before "
                     "it is chosen, not after a result depends on it."),
            "claim_boundary": "Selecting a domain inherits its declared violations. It does not "
                              "verify that the archive behaves as the declaration says."}


def preregistration_summary(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """The frozen plan in sentences, rendered here so the browser cannot paraphrase it.

    A preregistration a researcher signs after reading a summary the UI composed itself is a
    preregistration of the summary. These sentences are generated from the same spec that is
    hashed, and the digest is returned beside them.
    """
    observations = ", ".join("%s (%s, %s)" % (row.label, row.measure, row.units)
                             for row in spec.observations)
    windows = "; ".join("%s from %s to %s" % (row.name, row.start_utc.isoformat(),
                                              row.end_utc.isoformat()) for row in spec.windows)
    null = spec.nulls[0]
    confirmation = spec.confirmation
    sentences = [
        "This study is %r: %s." % (spec.study_id, spec.title),
        "It compares %s." % observations,
        "It compares them in %s mode, which permits %s; %s."
        % (spec.mode,
           "co-occurrence on shared UTC support" if spec.mode == "calendar_aligned"
           else "normalized structural recurrence",
           MODE_FORBIDS[spec.mode]),
        "It addresses these explicit windows: %s." % windows,
        "Coverage is %s at a minimum fraction of %s; anything less is refused rather than "
        "quietly narrowed." % (spec.coverage_policy.requirement,
                               spec.coverage_policy.minimum_fraction),
        "The complete tested family is declared before acquisition and corrected by %s at "
        "alpha %s." % (spec.correction, spec.alpha),
        "Significance is measured against the %r null (%s) at %d replications, which every "
        "declared domain admits over its own support."
        % (null.name, null.method, null.replications),
        "The confirmation design is %s%s." % (confirmation.stage,
                                              (" on the held-out partition %r"
                                               % confirmation.held_out_partition)
                                              if confirmation.held_out_partition else ""),
        "Randomness is fixed by these seeds: %s." % ", ".join(
            "%s=%d" % (key, value) for key, value in sorted(spec.seeds.items())),
        "Freezing this plan fixes it. A frozen plan is never edited after seeing how it went; "
        "a refusal is answered with an editable copy that leaves the frozen run unchanged.",
    ]
    return {"schema": "composer-preregistration-summary/v1",
            "manifest_sha256": manifest_sha256(spec), "study_id": spec.study_id,
            "sentences": sentences,
            "claim_boundary": "This is the plan, in the plan's own words. It is not a result, and "
                              "reading it is not freezing it."}


# ----------------------------------------------------------------- import and export


def export_envelope(spec: CrossDomainExperimentSpec) -> Dict[str, Any]:
    """A machine-readable envelope carrying the canonical manifest and its digest.

    The envelope digest covers the canonical bytes, so a manifest that was edited in a text
    editor between export and import arrives as an integrity error rather than as a plan whose
    run identity addresses something else.
    """
    body = json.loads(canonical_bytes(spec).decode("utf-8"))
    digest = manifest_sha256(spec)
    return {"schema": "composer-manifest-envelope/v1", "manifest_sha256": digest,
            "envelope_sha256": hashlib.sha256(
                json.dumps({"schema": "composer-manifest-envelope/v1", "manifest_sha256": digest},
                           sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "exported_from": "spectral-research-accelerator",
            "manifest": body,
            "note": ("Direct JSON editing is never required to compose an experiment. This "
                     "envelope exists so a plan can be moved between instances and cited, not "
                     "so it can be hand-edited."),
            "claim_boundary": "An exported manifest is a plan. It carries no data and no result."}


def import_envelope(payload: Mapping[str, Any]) -> CrossDomainExperimentSpec:
    """Parse an exported envelope, refusing one whose digest disagrees with its manifest."""
    if not isinstance(payload, Mapping) or "manifest" not in payload:
        raise InvalidParameterError("envelope", type(payload).__name__,
                                    "an object with a 'manifest' key, as produced by export")
    spec = CrossDomainExperimentSpec(**dict(payload["manifest"]))
    declared = str(payload.get("manifest_sha256", ""))
    actual = manifest_sha256(spec)
    if declared and declared != actual:
        raise EnvelopeIntegrityError(
            "this envelope declares manifest %s but its body hashes to %s. A run's identity is "
            "the content address of its plan, so importing this would give the run an identity "
            "that does not address what it would execute." % (declared, actual),
            declared=declared, actual=actual)
    return spec


def describe_path() -> Dict[str, Any]:
    """The whole workflow as a served contract (standard E1)."""
    return {"schema": SCHEMA,
            "steps": [step.describe() for step in ordered_steps()],
            "step_statuses": list(STEP_STATUSES),
            "duration_presets": sorted(DURATION_PRESETS),
            "ladder": [dict(rung) for rung in STAGE_LADDER],
            "note": ("The browser renders this path; it does not compute it. `next_action` has "
                     "one value because two enabled buttons meaning two different scientific "
                     "commitments cannot both be the next legitimate act."),
            "claim_boundary": "Walking the whole path produces an executed plan. A finding and "
                              "admitted evidence are separate recorded acts, elsewhere."}
