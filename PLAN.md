# PLAN ? what remains to make SpectralEarth truthfully useful

**This is the document to work through. This document is the plan, and only the plan.** It
states priorities, the next action and its acceptance -- nothing else. It does not restate
status, because a plan that carries status becomes another thing to keep in step.

**For what is implemented, read `architecture.md` section 0.** That one table is the single
authority. `roadmap.md` and `roadmap_cross_domain.md` hold task history -- what was done and
why -- and `VERIFICATION.md` holds the captured output behind every figure. When any of them
disagrees with this document, **they are right and this is stale.**

Last revised 2026-09-09, after T4E.11 candidate D and the diagnostic that redirects to T4E.12.

## 0. The question this sequence serves

> What stands between here and the first trustworthy, independently checkable finding
> this instrument produces about the atmosphere?

A completed, interpretable negative or invalid result is useful progress. A passing
software suite is evidence about the apparatus, not a finding about the atmosphere.
The wider cross-domain programme remains active and has separate dependencies in section 4.

## 1. Current decision point

> **Decided 2026-09-09 by the maintainer: `kind_recurrence` is the primary scientific
> target. `track_continuity` and `spatial_persistence` are diagnostics for it, and a result
> under either does not license it.**

`declare_identity_target` now records that distinction as data rather than as prose: a
declaration carries a `role`, and a diagnostic must name the target it is diagnostic for and
publishes a boundary saying its result does not license that target. Labelling a circular
evaluation "diagnostic" does not admit it -- `kind_recurrence` against record-derived proxy
labels is still refused, because the role governs what is claimed from a result and never what
evidence is admissible.

**What the decision costs, stated plainly.** `kind_recurrence` admits only `external_reference`
evidence, so the primary target needs a reviewed, cited catalogue and a serialisation for it,
and neither exists. `audit_spatial_identity.py` refuses every evidence class but the proxy one
by name for exactly that reason. So the primary target is **currently unevaluable from the
tool**, the catalogue moves onto the critical path, and it was already a T4F.9 prerequisite.
The diagnostics remain measurable now and are worth measuring; they are not progress toward the
target.

**What it does not change.** T4E.11 is still required and still blocks. Whichever target is
aimed at, the criterion has to survive a change of partition, and T4E.10 measured that the
present one does not.

The three targets, for reference:

Three targets are declared in `spectral_identity_audit.py` and they are not the same
question. **`track_continuity`** -- the same evolving tracked constellation, observed again.
**`spatial_persistence`** -- a spatial configuration keeping its geometry. **`kind_recurrence`**
-- the same physical kind recurring in a different constellation, which is the target the
mining, sequence and precursor machinery downstream actually requires.

Tracked keys are proxy labels for the first only. Slice 3 refuses `kind_recurrence` against
them by name, because labels drawn from the pipeline under test validate a definition against
itself. That refusal is why the decision above puts the catalogue on the critical path rather
than deferring it.

**What each choice costs.** `track_continuity` scores well and means little: it asks identity
to reproduce the tracker that made its labels. `spatial_persistence` is measured and fails its
declared criterion, and the failure is informative -- configurations hold their geometry over
one 6-hour step and lose it over a track's life, which is the regime mining needs.
`kind_recurrence` requires a reviewed, cited catalogue and a serialisation for it, neither of
which exists; that catalogue is already a T4F.9 prerequisite, so it is not additional work.

Do not redefine the target or weaken the acceptance criterion to make a candidate pass, and
do not report a diagnostic result as progress toward the primary target. D96, D97, D98, D99 and
D100 all remain open and none was repaired by making the choice.

**Next, in order.** T4E.11 candidate B was adopted on 2026-09-09 and **falsified on
development evidence the same day**: normalising by each partition's label-free close-pair
scale left the spread wider (1.876x to 1.991x), because that normaliser varies only 1.084x and
measures the nearest-*unrelated* distance rather than the same-configuration one. The reserved
confirmatory blocks are untouched. Candidate C was then declared,
corrected after external review, adopted and **falsified on development evidence the same day**
-- but it produced the first genuinely positive finding of the sequence. Mutual
nearest-neighbour recovers **every** motif pair in **every** block (split 0.0000) despite the
1.876x magnitude spread: **the ordering transfers where the magnitudes do not.** It fails
because it cannot decline -- 116 matches on a null block where the answer is none.

Candidate D was then declared, adopted and **falsified on development evidence the same
day** -- the cleanest failure of the four, and the most informative. D is C narrowed by a
dimensionless margin: a mutual pair survives only when its distance is at most `tau` times the
distance to the second nearest, both directions, with `tau` pinned at **0.8** from Lowe (2004)
for its external provenance.

**The margin costs nothing in recall.** Split stayed at **0.0000 on every block** and the match
counts fell everywhere, exactly as the declaration predicted before measuring: what it discarded
was entirely non-motif. **It fails on rejection** -- null retention **0.5345** against an
accepted 0.10, keeping 62 of candidate C's 116 matches where the answer is none.

**And the diagnostic settles the question that was actually open.** Motif ratios run 0.020 to
0.293; the null block's begin at **0.240**. So the contrast is real and 0.8 is far too
permissive -- but **the distributions overlap**, and no threshold separates them cleanly. The
overlap is narrow, which is exactly what makes it dangerous: a `tau` chosen to sit inside it
would be fitted to blocks now inspected four times over.

**A diagnostic then settled where to go next, and it is the signature.** Before
declaring anything further, `decompose_matched_pairs` asked what the false admissions actually
are -- a question none of the four declarations had asked of a ground truth all four were
measured against. It adopts nothing and reports no operating point.

**The suspicion it was built to test was wrong, and that is worth stating plainly.** Nine of
every twenty configurations in a scene share two motif features, which recur by construction,
so the ground truth might have been scoring real recurrence as error. Only **29%** of candidate
D's non-motif matches hold the same motif vertices, and that class is enriched just 3.2x over
unrelated. **The ground truth is sound.**

**What the signature does well is now measured, not assumed.** 60 of 60 motif pairs recovered.
A background match rate that does not notice whether a motif is present -- 1.054% in the
planted blocks against 1.033% in the null -- so the null's retention was never an artefact of
the null. And the hardest discrimination in the scene, the full motif against a configuration
holding two of its own three features, is **0 matches out of 1080**.

**The constraint is arithmetic and it is a number.** Each scene pair offers 400 candidate pairs
and holds one true positive: prior odds 1:399. Candidate D's per-pair false rate is **1.019%**,
a specificity of 98.98%, and the declared 0.10 bound needs **0.028%** -- a **36-fold**
reduction. No threshold placed on these distances can supply that.

**T4E.12 is the next task and it is about the signature**, not a fifth criterion. **It is
now specified, with acceptance, in `roadmap.md`.** Writing that acceptance required measuring
how the problem scales, because the obvious form of acceptance turns out to be the wrong one.

**Scene richness is now a parameter** -- the generator takes a feature count defaulting to the
frozen six, so every existing caller builds exactly the scenes it built before -- and the sweep
at 6, 9 and 12 features says two things that point opposite ways.

*Recall is untouched.* False split **0.0000 at every richness**: the motif is still the mutual
nearest neighbour against 219 rivals rather than 19. The signature carries what identification
needs.

*The rule matches a constant fraction of whatever it is given* -- about **a quarter** of
configurations at every richness -- because a nearest-neighbour matching returns at most one
pair per configuration. So false admissions grow **linearly** with the configuration count while
true correspondences stay at one, the admission fraction climbs to **0.979** at twelve features,
and the shortfall widens **x33 to x423**.

**That is why acceptance is not an admission rate.** The per-pair rate falls as `1/m` where a
fixed bound demands `1/m^2`, so an admission fraction is a property of the signature and the
scene together and never of the signature alone -- T4E.10's transfer problem in a new place.
**No rule whose match count scales with the configuration count can succeed**, whatever
threshold is placed on its distances.

**So T4E.12 accepts on a per-pair rate measured at three richness levels**, requiring the
matched fraction to fall as richness grows and the shortfall not to widen.

**Candidate 1 is declared and implemented, and waiting on your adoption.** It is not the
attribute weights. Those are all 1.0 and were never calibrated, which made them the obvious
first move, but they **cannot** satisfy the acceptance: a nearest-neighbour matching returns at
most one pair per configuration whatever the weights are, so reweighting changes which pairs
match and never how many, and the matched fraction stays pinned near the measured 0.23. That is
derivable on paper, so declaring it would have spent a preregistration to learn nothing.

What the scaling law actually indicts is **multiplicity**: the rule does not know it is making
726,000 comparisons rather than 400. So the T4E.12 acceptance clause was **amended in the open**
-- as first written it required a candidate to be a change to the signature, which would have
excluded the class of candidate the evidence points at. The four numbered conditions are
unchanged.

Candidate 1 admits a pair only when a distance as small as its own would arise with probability
at most `alpha / m^2` under a tail model fitted to that scene pair's own distances, with
`alpha = 0.05` a declared family-wise error rate rather than a tuned threshold. The bound
tightens by construction as scenes get richer, 1.25e-4 to 6.9e-8, which is the `1/m^2` scaling
the acceptance demands.

**Candidate 1 was adopted and falsified the same day, and it did not test the
signature.** It admitted **nothing** -- not one pair across four blocks at nine features, four
at twelve, or either null. False split 1.0 against an accepted 0.10.

**The null tells us why, and the declaration fixed that reading in advance.** If the tail model
held, admissions per scene pair would equal `alpha` whatever the signature is like. Nominal
0.05, measured **0.0000**: the model under-admits relative to its own nominal rate. So this is a
verdict on the tail model, **not on the signature**.

**Two failures were derivable before adoption and I did not derive them.** Richness 6 can never
support the model -- 50 exceedances at a 1st-percentile threshold needs 5,000 distances and six
features give 400. And the motif sits inside the sample the tail is fitted to, so its estimated
probability cannot fall far below the exceedance rate over the exceedance count. Sampled
directly, the motif's tail probability is 9.1x, 6.0x, 3.5x and 15.3x above the bound -- short by
a single-digit to low-double-digit factor, with **no trend in richness claimed**, because one
block improves and another worsens.

**So the standing lesson is a new check on declarations here.** Candidate D fixed the
falsification-licensing field and that correction held. What was missing this time is a
**feasibility test**: can this criterion admit anything at all, in principle, at the support
declared for it? That check now belongs in every declaration before adoption.

**A condition was found and named while measuring**: at twelve features a few configurations are
too nearly isotropic to carry a bearing block, and the metric refuses to compare them with ones
that do. It is rare -- 1, 1 and 2 of 1,320 in three of four blocks, none at six or nine -- but
it **qualifies the scaling measurement**, which ran on block 100-105 alone and would have failed
on three of the four blocks at twelve features.

**What is not authorised**: raising `alpha`, moving the threshold percentile or exceedance
minimum, or dropping richness 6. Each is tuning against blocks inspected many times over. Five
falsifications have not unreserved the confirmatory blocks.

**Candidate 2 is declared and implemented, and waiting on your adoption.** Every falsified
candidate asked the same question -- is *this pair* a match? None used what the partition
offers. The motif is in every scene, so its matches form a complete graph; a coincidence between
two scenes has no reason to extend to the rest. So a set of configurations, one per scene, is
**consistent** when every member is the mutual nearest neighbour of every other, and only pairs
inside sets spanning the whole partition are admitted.

**It has no radius, ratio, threshold, fitted model or normaliser** -- nothing to transfer,
nothing to estimate wrongly, nothing whose support runs out. It is evaluable at richness 6,
where candidate 1 could only refuse.

**The feasibility check is done and disclosed**, which is the standing lesson from candidate 1.
Across six-scene partitions the motif reached a clique of **6 in every scene of every block**,
no non-motif configuration exceeded **5**, and the null reached **4**. So it can admit the
answer and reject the null. It is not inert.

**Two things to hold in mind when the result comes back.** The probe informed the choice of
`k`, and the declaration says so — development evidence here is weaker than for its
predecessors, not stronger, which is why confirmatory evidence would now be worth more than at
any earlier point. And `k` = every scene **will not transfer to an acquired record**, where a
real pattern need not appear in every window; that limitation is declared rather than left to be
discovered.

**Adopted as a development experiment on 2026-09-09, and it passes.**

```
rich  block      configs   C props  admitted    split    admit  shortfall
6     100-105         20       121        15   0.0000   0.0000      x0.00
6     200-205         20       121        15   0.0000   0.0000      x0.00
6     300-305         20       152        15   0.0000   0.0000      x0.00
6     400-405         20       136        15   0.0000   0.0000      x0.00
9     (four blocks)   84   485-585        15   0.0000   0.0000      x0.00
12    (four blocks)  220 1440-1590        15   0.0000   0.0000      x0.00

NULL rich=6    C proposed 116     ADMITTED 0
NULL rich=9    C proposed 595     ADMITTED 0
NULL rich=12   C proposed 1486    ADMITTED 0

matched fraction by richness: 0.05000, 0.01190, 0.00455  (exactly 1/m)
incomparable configurations refused: 0, 0, 4
```

**All four acceptance conditions met, on development evidence.** Every block at every
richness admits exactly 15 pairs -- C(6,2), the motif's complete clique -- and nothing else.

**Which half is informative.** The recall half was very nearly guaranteed: the scenes are built
with the motif in every scene, this criterion admits configurations present in every scene, and
candidate C had already recovered every motif pair in every block, so a partition-spanning group
exists **by construction**. **The rejection half is the finding** -- zero false admissions, and
a null admitting none of 116, 595 or 1486 proposed pairs, where candidate C returned 116 and
candidate D returned 62.

**The maintainer's ceiling, fixed before the numbers were known.** *"Candidate 2 looks like a
strong diagnostic of whether the signature can sustain coherent identity. It is not yet a
defensible real-world recurrence rule."* This is evidence that the signature **can** sustain
coherent identity across a partition, and is not reported as a recurrence criterion.

**The confirmatory blocks were withheld by decision and remain clean**, because `k` was
influenced by the feasibility probe on these same blocks. That is stricter than the declaration
asked. **No confirmatory evidence for this candidate exists or will exist under this adoption.**

**What it does not establish**: nothing about partial recurrence, since `k = S` rejects a
configuration absent from one scene outright; nothing about recurrence *across* partitions,
which the mining machinery needs; no mining radius, no discharge of T4E.8's acceptance, no
closure of D96 to D100; and nothing about `kind_recurrence`, which still has no catalogue.

**So the next choice is yours, and it is narrower than any before it.** A criterion of this
shape with `k` below the partition size -- declared *before* any probe of how coincidental
groups behave at that `k` -- is the obvious route to something that could transfer to a real
record, and the reserved blocks would then be spendable on a structural choice made blind. That
is the first time in this sequence the confirmatory evidence would be worth what it costs.

**Still unmoved by any of this**: the catalogue `kind_recurrence` requires, which is the primary
scientific target and remains unevaluable from the tool without a reviewed, cited source.

The reserved confirmatory blocks remain untouched, and four falsifications have not unreserved
them.

After that, the catalogue `kind_recurrence` requires -- sourcing it, deciding its independence
class, and a serialisation the audit tool can load.

## 2. Ordered atmospheric work

### T4E.8 ? Finish a defensible identity definition and its validation ? in progress

Slices 1, 2 and 3 are implemented; the acquired-record acceptance remains unmet.

**Slice 3 changed what is blocking.** The audit now refuses to run without a declared
`identity_target` and `evidence_class`, and refuses `kind_recurrence` against record-derived
proxy labels as circular. Naming the slice-2 design's target reproduced its figures
bit-for-bit, so the declaration cost no measurement. Two things it did not deliver are open
and recorded in `roadmap.md`: no interface surface renders the declaration, because none reads
identity-calibration receipts; and `kind_recurrence` is evaluable only in the library, because
no serialisation for a signed `PatternCatalogue` exists for the audit tool to load.

**The decision below is now the only thing holding this task.** It was always a scientific
question; it is now also a refusal the software will not step around.

Next:

1. **Choose the identity target.** `track_continuity`, `spatial_persistence` and
   `kind_recurrence` are declared in `spectral_identity_audit.py` with what each recognises and
   does not license; review them against the measurements in `architecture.md` sections 3E.8
   and 3E.9 and the T4E.8 verification entries. The slice-2 mode is record-scoped grid geometry,
   not location-independent physical morphology. Choosing `kind_recurrence` additionally
   requires a reviewed catalogue and a format for it; that is an input this programme must be
   given, not one it can generate.
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

### T4E.9 ? Certified synthetic ground truth for the identity step ? implemented, reports FAIL

**Done, and it moved the problem.** The benchmark exists and reports: the T4E identity
definition separates a certified motif perfectly (AUC 1.0, zero admissions in 11,985 negative
pairs), and the frozen radius splits 46.7% of motif pairs while a feasible radius exists. D101
is closed. **What remains of D97 is now named precisely: the operating point, not the record.**
The next slice against it is an estimator of the radius that transfers -- measurable entirely on
synthetic scenes, with no catalogue, no atmosphere and a known answer.

**This task closed D101.** Specifying it uncovered that this repository holds two
identity layers over one shared geometric core, and the benchmark suite certifies the wrong
one. `src/core/motif.py` is exercised against planted and null scenes; the T4E path --
`signature_for`, `SignatureMetric`, `cluster_signatures`, the path T4E.8 measures and T4F.6
would adjudicate -- appears nowhere in `src/benchmarks/`. Its only ground-truth-like evidence
is T4E.7's check, which plants replicates at the signature level and so skips detection,
tracking and constellation extraction. **The T4E identity step has never been run against an
answer known by construction.**

**It is also the cheapest thing on the table that could move T4E.8, because it attacks the
ground-truth problem instead of waiting on a catalogue.** Every difficulty in T4E.8 comes from the same
place: a real atmospheric record supplies no replicates (D97), no honest null (D98) and no
independent labels, so `kind_recurrence` waits on a catalogue nobody has signed.

`src/benchmarks/fields.py` already synthesises fields whose answers are derivable independently
of the analysis code -- `S(k) ~ k^-beta` by construction, fBm at a known Hurst exponent. Run
detection, tracking, constellation extraction and identity over such a field and ask whether the
identity step recovers structure that is *provably* present. That is ground truth no person has
to sign.

The external project `certified-invariants` (adamfbentley) works the same substrate from the
other end: exact Fisher-information ceilings and Le Cam minimax floors on Gaussian random fields
with structured spectra, with preregistered gates and a committed failure ledger. Its ceilings
are a candidate certified target for the same experiment, and its own recorded limitation --
certificate estimates that sometimes exceed their ceilings, flagged as estimator failures -- is
the same species as D97 and must be checked before any ceiling is imported as truth. This is
also what Phase 4H already calls an optional ceiling estimator.

**Boundary, stated before anything is built.** This validates the *apparatus*, not the
atmosphere. A pass here does not discharge T4E.8's acquired-record acceptance and does not
license a mining radius. What it settles is whether the identity definition can recover a known
answer at all -- which is currently unknown, because T4E.7's synthetic check plants replicates
by construction rather than recovering a certified quantity. A failure here would be decisive
about the definition and would cost no catalogue.

Needs its own task, design and acceptance before it runs, declared first. Adopting an external
project's ceilings is a scientific choice and must not be made silently by software.

### T4E.11 ? An identity criterion that survives a change of partition ? not specified

**T4E.10 removed the option of fixing this with a better estimator.** Four blocks of six scenes
from one generator, identical parameters, differ **1.88x** in mean same-configuration distance.
A frozen absolute radius therefore cannot transfer between partitions -- not because the
estimator was poor, but because calibration and evaluation are not one population. A real
record's disjoint windows will differ more than synthetic blocks do, not less.

Three candidates, each a scientific choice needing its own task and acceptance before it is
measured:

1. **Per-partition calibration.** Each window sets its own radius. Cheap and honest, but it
   changes what a pattern is between windows, so recurrence across windows needs its own
   argument and may become unstateable.
2. **A distance whose scale is comparable across partitions.** Normalise the metric so a radius
   means the same thing everywhere -- by the partition's own distance distribution, or by a
   quantity the record supplies. This keeps one criterion but changes the metric, which is a
   change to what identity *is*.
3. **An identity criterion that is not a radius.** A relative or rank-based rule -- nearest
   neighbour with a margin, mutual nearest neighbours -- has no absolute scale to transfer.
   `representation_alignment.py` already implements the mutual k-NN machinery and its
   chance floor, though it was written for a different purpose.

Acceptance for whichever is chosen: measured on the T4E.9 scenes across at least four blocks,
both error rates reported or refused by name, and **the criterion must hold on blocks it was not
calibrated on** -- which is precisely the test every estimator has now failed. Do not select one
by measuring all three and keeping the winner; that is R20's forbidden move.

**This is now what blocks the identity target decision from mattering.** Whichever target is
chosen, it needs a criterion that transfers.

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

**G7 and G19 must be able to reach the documents of record, and currently cannot.** G19's own
rule is right and is not in question: *retrieve at the granularity of a complete record, never a
fragment of one*, because a receipt stripped of its `claim_boundary` is the exact failure this
platform exists to prevent. The gap is the corpus, not the rule. G19 addresses receipts,
qualification plans and `EvidenceBundle`s -- but a researcher meeting a refusal asks *why*, and
the answer lives in the defect ledger, the status table and this plan, which are markdown
documents and not bounded structured objects. A conversation that can quote a gate receipt but
cannot reach D97 or D98 will explain a refusal without its reason.

G19 already says what to do about it: *a record too large to enter whole is a signal that the
platform owes a deterministic summary view it computes itself.* `architecture.md` section 0 is
exactly that view for status, and it was written to be entered whole. Before G19 starts, decide
which documents of record are addressable, and supply a deterministic computed view for each one
too large to enter entire -- the defect ledger and the standing rules being the two that matter
most. The same corpus serves G7's adversarial review: a layer that argues with a finding needs
the finding's defects reachable, or it argues with half of it. Neither may touch a gate (R22),
and nothing here changes that.

One piece of exploratory apparatus sits outside every phase and is deliberately unscheduled:
`representation_alignment.py`, written to audit the Platonic Representation Hypothesis, which
supplies the mutual k-NN metric, its closed-form chance floor k/(n-1) and a permutation null.
It gates nothing, no model has been downloaded or evaluated, and running it on real
representations would open a second domain under R17. It is not next unless it is chosen.

G18's interface programme is complete according to its recorded acceptance. G19's
multi-turn conversation with the evidence is specified and not started. Schedule it
explicitly after the current identity decision; it is not an implicit next task in the
atmospheric sequence. `roadmap_cross_domain.md` remains the broader programme's task history.

## 5. The interface, as an instrument rather than a control panel

G18 is complete against its recorded acceptance, and that acceptance was about the interface
working. The standing requirement is stronger and is not yet met everywhere:

> **The interface must expose the scientific contract, not merely operate the backend. A refusal
> ranks equal to a value, and renders as a first-class result rather than an error state.**

Three known gaps, in the order they bite:

1. ~~**The identity declaration is invisible.**~~ **Closed by T4E.8 slice 4.** `/api/v1/identity/*`
   and `IdentityDeclarationView` render the admissibility matrix, the circularity refusal at the
   weight of an admission, the caveat an admitted pairing still owes, receipts with their claim
   boundaries, and the surface's own refusals. Read-only: choosing remains a person's act.
   Evidence is rendered, not described -- see `architecture.md` section 3E.11.
2. **Refusals still read as absence.** Wherever a figure is unavailable because a gate refused,
   the reason, its named defect and its claim boundary must travel with the empty space. An
   unexplained blank is indistinguishable from a bug and teaches a researcher to distrust the
   instrument.
3. **No view states what a number may not be used for.** Every receipt in this programme carries
   a `claim_boundary`; the interface should never show the value without it.

Acceptance for any interface slice: rendered evidence captured in `VERIFICATION.md`, refusals
demonstrated on screen rather than described, and no WCAG level claimed without the required
rendered and assistive-technology evidence. An interface slice that only adds a control panel
over an existing endpoint does not meet this section.

## 6. Deliberately not next

- D96's algorithm work: its workload is set by the radius, and T4E.10 measured that the radius
  does not transfer. Optimising the clustering now would optimise against an unstable number.
  T4E.11 comes first.
- Phase 4G representation scoring: gated on the required T4F.6 PASS.
- Running `representation_alignment.py` on real models: needs downloads and opens a second
  domain under R17. Not next unless chosen.
- Phase 4H learned encoder: optional ceiling estimator, not the immediate deliverable.
- Phase 5 real forecast comparison: interfaces exist; the external model and experiment
  remain to be supplied and bound.
- D84 and D85: unresolved negative-result detectability questions; they do not undo the
  already acquired record or T4C.6 PASS.
- D18: cross-device agreement outside the measured CPU/CUDA configurations.
- Accessibility conformance: no WCAG level is claimed without its required rendered and
  assistive-technology evidence.
- Professional licence review: still absent and deprioritised.

## 7. Standing constraints

- Nothing reaches the network without the maintainer's explicit say-so.
- Declare scientific parameters before confirmatory reads; preserve exploratory history
  and record amendments openly. Already inspected data do not become untouched by renaming them.
- Code does not sign catalogues, events or scientific preregistrations for a person.
- A named refusal is preferable to an unsupported number. Optimisation must preserve it.
- Never commit `data/market_records/*.csv`; bind the record by its content identity.
- `ed-dev` stays local.
- Update architecture, task history and captured verification with each implementation
  slice. Keep this plan subordinate to those records.
