# PLAN ? what remains to make SpectralEarth truthfully useful

**This document is the plan, and only the plan.** It states priorities and acceptance,
not the status of record. `roadmap.md` and `roadmap_cross_domain.md` hold task history,
`architecture.md` describes the implementation and defects, and `VERIFICATION.md` holds
measurements. When they disagree with this document, **they are right and this is stale.**

Last revised 2026-09-08, after T4E.8 slice 2.

## 0. The question this sequence serves

> What stands between here and the first trustworthy, independently checkable finding
> this instrument produces about the atmosphere?

A completed, interpretable negative or invalid result is useful progress. A passing
software suite is evidence about the apparatus, not a finding about the atmosphere.
The wider cross-domain programme remains active and has separate dependencies in section 4.

## 1. Current decision point

The record is acquired and T4C.6 passed. The downstream tracking, constellation,
precursor, projection, physical-gate and generalisation machinery exists. The full
mining pass and T4F.6 adjudication have not completed on that record.

T4E.6 and T4E.7 supplied two-sided error reporting and null calibration. T4E.8 slice 1
identified band instability and a nondiscriminating strength component. Slice 2 now
supplies an explicit, registered `spatial_geometry` mode and a reproducible local audit.
It removes detector-band and magnitude dependence at fixed positions, but the measured
candidate fails the declared acquired-record identity criterion. No mining radius is approved.

**The immediate unresolved question is what identity is meant to recognise.** Continuity
of an evolving tracked constellation, persistence of spatial geometry and recurrence of
the same physical kind in different constellations are distinct targets. Track keys are
proxy labels for the first; they are not independent ground truth for the other two.
Do not redefine that target or weaken the acceptance criterion to make the candidate pass.

D99 and D100 are absent from the new candidate's comparable vector, but the original
mining declaration still uses the old mode. D97 remains unresolved; D96 remains downstream.
D98 remains open for recurrence inference and is not repaired by this labelled audit.

## 2. Ordered atmospheric work

### T4E.8 ? Finish a defensible identity definition and its validation ? in progress

Slices 1 and 2 are implemented; the acquired-record acceptance remains unmet.
**Slice 3 is specified in `roadmap.md` and not started.** It makes the identity target a
declared, enumerated field the audit refuses to run without, types the label source by
provenance, refuses `kind_recurrence` against record-derived proxy labels as circular, opens
the external-reference path through `match_into_catalogue`, and surfaces target, evidence and
refusal in the interface. It is deliberately question-preserving: it blocks the unanswered
question rather than answering it, and it must reproduce the slice-2 figures bit-for-bit.
Slice 3 does not close this task; item 1 below remains the maintainer's decision.

Next:

1. Make the intended identity target explicit and review it against the measurements in
   `architecture.md` section 3E.8 and the T4E.8 verification entries. The new mode is
   record-scoped grid geometry, not location-independent physical morphology.
2. Establish labels or an independently justified reference for that target. Keep
   exploratory development windows distinct from any future confirmatory evaluation;
   the three audited windows are already inspected training data.
3. If the definition changes, record a new task/design and its acceptance before evaluating
   it. The declared-catalogue alternative in section 3 is available, but is a scientific
   model choice and must not be selected silently by software.
4. Re-measure both error rates and the support/dependence of the calibration population.
   Any accepted radius must state its population, source/grid scope, metric, provenance
   and limitations. Broad recurrence needs its own validation; a track-continuity radius
   does not establish it.

Acceptance: a scientifically justified identity target, both error rates within declared
bounds on an appropriate evaluation population, and calibration support adequate for the
precision claimed. The declaration must explicitly adopt the new mode before any pipeline
uses it. Analytical band independence alone does not close the acquired-record task.

### T4F.9 ? One interpretable end-to-end pilot ? not started

After identity is defensible, freeze a pilot design small enough to finish and large
enough for its statistics. Run climatology, anomalies, tracking, constellations, identity,
sequences, precursors, projection and the physical gate.

Acceptance: PASS, FAIL or INVALID with a receipt naming the frozen catalogue, declared
design, input identities and each figure's provenance. A scientific failure can complete
the pilot; an uninterpretable verdict cannot. This is not the full T4F.6 acceptance.

Still required: T4E.8 acceptance, a reviewed and signed reference catalogue, and a documented
cyclogenesis event with its source. The existing draft is material for that review, not an
already signed declaration. Code must not supply a person's scientific signature.

### T4E.5 ? Finish scalable identity ? in progress, after the identity decision

The existing accelerator is exact on tested inputs but still uses dense component
cross-distances. Its recorded mutation work is incomplete.

Finish the surviving mutation checks, make component cross-distances sparse where exact
complete-link updates permit it, preserve scalar/accelerated agreement and all refusals,
and measure component structure, runtime and memory under the accepted identity definition
and radius. A different radius changes the workload; old measurements cannot close D96.

### T4F.6 ? Full acquired-record mining and physical adjudication ? not run to completion

Run the full declared training/evaluation design after identity, scaling and the required
scientific declarations are ready. Amend and digest the existing declaration openly if
its signature mode or other scientific choices change. Preserve the train-only catalogue
and climatology boundaries, and bind any held-out evaluation before opening it.

Acceptance: real adjudication with reproducible provenance and the implemented PASS,
FAIL and INVALID meanings. Entries that this crop or single-level record cannot assess
must remain explicitly unassessable. Phase 4G opens only on the required PASS.

## 3. The scientific alternative, if discovery does not meet its acceptance

A reviewed, cited, frozen catalogue matched at a declared radius is already supported
by `match_into_catalogue` and the physical-reference machinery. It asks a different
question from discovering identities from the record. Choosing it requires an explicit
scientific declaration and its own acceptance, not a change that makes the old gate green.

The failed spatial audit does not prove discovery impossible. It shows what this
candidate and these proxy labels cannot yet justify. Measure the intended target first.

## 4. Independent cross-domain work ? retained, not covered by the atmospheric gate

G17 release remains withheld. TG17.15's replacement scale/shape inference is implemented
and calibrated, but release still requires a manifest that requests that inference and
curated real-record partner pools with a justified exchangeability/admission argument.
An atmospheric identity improvement does not discharge either obligation.

G18's interface programme is complete according to its recorded acceptance. G19's
multi-turn conversation with the evidence is specified and not started. Schedule it
explicitly after the current identity decision; it is not an implicit next task in the
atmospheric sequence. `roadmap_cross_domain.md` remains the broader programme's task history.

## 5. Deliberately not next

- Phase 4G representation scoring: gated on the required T4F.6 PASS.
- Phase 4H learned encoder: optional ceiling estimator, not the immediate deliverable.
- Phase 5 real forecast comparison: interfaces exist; the external model and experiment
  remain to be supplied and bound.
- D84 and D85: unresolved negative-result detectability questions; they do not undo the
  already acquired record or T4C.6 PASS.
- D18: cross-device agreement outside the measured CPU/CUDA configurations.
- Accessibility conformance: no WCAG level is claimed without its required rendered and
  assistive-technology evidence.
- Professional licence review: still absent and deprioritised.

## 6. Standing constraints

- Nothing reaches the network without the maintainer's explicit say-so.
- Declare scientific parameters before confirmatory reads; preserve exploratory history
  and record amendments openly. Already inspected data do not become untouched by renaming them.
- Code does not sign catalogues, events or scientific preregistrations for a person.
- A named refusal is preferable to an unsupported number. Optimisation must preserve it.
- Never commit `data/market_records/*.csv`; bind the record by its content identity.
- `ed-dev` stays local.
- Update architecture, task history and captured verification with each implementation
  slice. Keep this plan subordinate to those records.
