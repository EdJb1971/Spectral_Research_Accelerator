# PLAN ? what remains to make SpectralEarth truthfully useful

**This is the document to work through. This document is the plan, and only the plan.** It
states priorities, the next action and its acceptance -- nothing else. It does not restate
status, because a plan that carries status becomes another thing to keep in step.

**For what is implemented, read `architecture.md` section 0.** That one table is the single
authority. `roadmap.md` and `roadmap_cross_domain.md` hold task history -- what was done and
why -- and `VERIFICATION.md` holds the captured output behind every figure. When any of them
disagrees with this document, **they are right and this is stale.**

Last revised 2026-09-14, after drafting the ninth criterion candidate. **Section 1 retains the sequential
decision record that led to the current position and must not be read as a current-status
summary; `architecture.md` section 0 is the authority wherever a historical next step below has
since been completed or superseded.**

## 0. The question this sequence serves

> What stands between here and the first trustworthy, independently checkable finding
> this shared instrument produces across materially different domains?

A completed, interpretable negative or invalid result is useful progress. A passing
software suite is evidence about the apparatus, not a finding about the atmosphere.
Atmospheric and cross-domain science have different acceptance gates, but they use one engine:
the registries, manifests, execution, provenance, evidence, claim ladder and review machinery are
shared. Acquisition, geometry, support and scientific acceptance remain domain-specific.

### V1 release boundary

V1 is one shared research instrument with two parallel validation lanes and one release gate.
It is relatively complete when all of the following are true:

1. A researcher can compose, adopt, run, cancel or resume, inspect and review a long scientific
  measurement through the shared durable-run machinery; every terminal outcome preserves an
  immutable receipt and renders `PASS`, `FAIL`, `INVALID` or `REFUSED` without collapsing them.
  **T4E.43 closed the outstanding piece of this**: execution is off the event loop, so the
  instrument keeps answering while a measurement runs, and a cancel raised mid-run is observed
  at the next component boundary instead of only reaching a run that was not running.
2. The atmospheric lane has one bounded end-to-end pilot over real acquired data, including an
  independently sourced identity catalogue, a declared criterion, physical adjudication and the
  claim boundary that survives whether the result passes or fails.
3. The cross-domain lane has one frozen manifest using the successor scale/shape inference and
  curated real-record partner pools whose exchangeability and admission argument is recorded;
  G17 either releases or preserves a bounded refusal.
4. The same provenance, evidence, claim-ladder and review contracts govern both lanes. No domain
  receives a private runner, evidence store or weaker meaning of acceptance.
5. A researcher can start from a research object rather than a named domain: V1 demonstrates one
  declared independent sample table and one existing domain-native record passing through
  structural inspection without semantic inference, explicit role/relationship declaration,
  and one capability profile that shows available, unavailable and not-established instruments
  with reasons. Selecting an available instrument hands the object and its declarations into the
  appropriate existing planning or Composer path; it does not open a project-specific screen or
  create a private execution path.
6. Documentation audit, focused backend tests, production frontend build and rendered critical
  workflows pass against the release candidate.

**Ordered route.** T4E.36 has now tested T4E.35's eastward-support explanation on the fixed 18-row
population. It failed both unchanged raw gates at 5/18 and 0/18 against 9/18 required, so eastward
coverage is not a sufficient repair. T4E.37 now implements and freezes the next diagnostic: on
the remaining 13 condition-1 failures it distinguishes an in-radius pre-threshold sampled maximum
filtered by the existing extractor from a sampled field that offers no in-radius maximum at all.
Ed Bentley adopted its exact declaration and the offline evaluation returned
`FIELD_REFERENCE_SEPARATION_DOMINANT`: 12/13 failures had no native-grid local maximum inside the
inherited agency radius before thresholding, while JOSIE alone exposed an extractor-stage miss.
This is not an acceptance and does not prove the cause of field/reference separation. The next
atmospheric design is now frozen as T4E.38: exact-time ERA5 MSLP triangulation on all 18 fixed rows,
using deterministic catalogue-seeded pressure descent and vorticity ascent with no search radius.
Its single-level planner and guarded resumable acquisition are implemented, and Ed Bentley has
adopted the exact declaration for guarded execution testing. Explicit authorization then acquired
all 36 exact-time MSLP shards (18 per dateline segment; 2,326,213 stored bytes); every recorded
content digest matches and every shard reopens with its declared one-time 161x161 `msl` shape.
Offline materialisation then passed all 18 seams exactly, retained 180E once and published the
18x161x321 record at `dc6f5365…`. The frozen basin comparison returned 10 MSLP-closer rows, seven
vorticity-closer rows, one exact tie and no refusals: `VERTICAL_QUANTITY_SEPARATION_DOMINANT`, with
the mandatory `NOT_AN_ACCEPTANCE`. This supports that declared candidate only under the fixed
development-population rule; it proves no vertical separation and changes no extractor or gate.
That independent design was frozen as T4E.39 before opening 2022–2023, then run end to end. It
retained the signed catalogue, dates, domain, synoptic hours, two-agency minimum, two-degree
interior and deepest-per-storm selection; required at least 10 selected storms; and kept
ties/refusals in a strict-majority denominator. Ed Bentley adopted declaration digest `fc7ee60f…`
before any holdout identity was parsed, then authorized the separate network act. The census
selected 12 storms from 76,784 catalogue rows, fixing the strict majority at 7. Acquisition took 48
exact-time shards across ERA5 `msl` and 850 hPa `vo`, parent and wrapped complement (3,640,750
stored bytes; all digests re-verify and reopen at the declared geometry). Materialisation passed all
24 seams exactly and published the 12×161×321 two-field record at `179aa8c6…`. The unchanged basin
rule returned 7 MSLP-closer, 5 vorticity-closer, no ties and no refusals:
`HOLDOUT_SUPPORTS_VERTICAL_QUANTITY_CANDIDATE`, `NOT_AN_ACCEPTANCE`, reproduced exactly on replay.
This is the smallest majority the rule admits — one row would have made it `HOLDOUT_MIXED` — and
that fragility is part of the result. It establishes no generalisation, independence from
assimilation, cyclone identity, vertical tilt or pressure/surface separation, and changes no
extractor or gate. **The reserved period is now spent; it cannot be reopened as a fresh holdout,
and no post-hoc re-selection or reweighting of it is admissible.** T4E.40 then made that whole
flow readable: all eight stages, every artifact, all twelve rows and both basin walks now render
in Platform & evidence, with ties, refusals and the one-row fragility at the weight of the
counts, and with acquisition and re-running refused in the UI by name. It ends in a result
review that is structurally incapable of expressing an acceptance; that review is written and
adopted by a named person in the UI as two separate acts, and remains `NOT_WRITTEN`.
T4E.41 has now done the second of those: T4E.8's acceptance is eight declared conditions, each
carrying what it requires, what would license it, and what is explicitly not sufficient -- the
third clause naming this programme's own near misses, the T4E.38/T4E.39 deterministic
majorities among them. Its state is computable per condition and is `INSUFFICIENT_EVIDENCE`
at 0 of 8; evidence reaches a condition only through an adopted register entry, and no input
lets code emit `T4E8_ACCEPTED`. Ed Bentley adopted the T4E.41 bar on 2026-09-13, binding `bfb3c500...`; the verdict stayed
`INSUFFICIENT_EVIDENCE` at 0 of 8, which is what adopting a standard is supposed to do.
T4E.44 then derived what C6 actually requires before a ninth candidate is written, because the
obvious construction is already dead: T4E.16's surrogate reassembles the set it tests at
0.9547 over 199 replicates, recomputed from first principles, and its own conclusion was that
a feasible null is invalid here while a valid null is infeasible. Six constraints K1-K6 are
published, each bound to the result that established it, with the two temptations -- widening
the partition, and excluding the set under test -- stated beside their costs.
T4E.45 then ran the extremes test on the second of those shapes before anything was declared,
and it refuted the simple form of it. The small end works: a pair is expected by the hundreds,
so a clique-rarity bar refuses groups of two by arithmetic rather than by the special case
candidate 4 needed, and the expectation collapses from 121 to 2.4e-18 between m = 2 and m = 6
with no free parameter. The large end does not: independence predicts coincidental five-cliques
at 1e-13 while candidate 3 measured them in six of twelve blocks, understating by more than
1e12 in the direction that admits coincidences. A nearest-neighbour matching is transitive by
construction.
T4E.46 settled that question and corrected T4E.45 in doing so. The refutation was a chance
model judged against planted blocks, where a motif exists by construction; against the
motif-free null blocks -- which admit zero at m = 5 and m = 6 at every richness against
predictions of 1e-10 to 1e-15 -- independence is consistent. The motif contributes 15 edges
while planted blocks of one richness differ by 31 to 150, and at richness 9 the motif-free
block is denser than all four planted ones, so a motif-free block is not a different object.
**The conditional is answered NO: a bar can be calibrated where the motif is not, the two
shapes are not demonstrably one obstacle, the discovery family is not shown to be exhausted,
and section 3's catalogue alternative stays the alternative.**
What is actually left is a different problem. Candidate 3's false admissions are
motif-dependent -- zero in the null blocks, present only where a genuine set exists for a
near-miss to attach to -- which is the `k < S` relaxation the T4E.13 record already called a
margin one scene wide, not a defect in the null. T4E.47 has now drafted that ninth candidate. It scores a set by its rank diameter -- largest
pairwise distance as a quantile of the partition's own distance distribution -- against a
size-specific bar taken from the motif-free blocks, with no k, no radius and no threshold to
choose. Two features are forced rather than preferred: a rank because K4 forbids carrying an
absolute bar across partitions, and a tightest-subset reference because the nulls hold zero
consistent sets at m = 5 and 6, where candidate 5's absence rule would have admitted every
impostor. The extremes are derived, including two that are declared NOT derivable -- the
attachment comparison the result turns on, and whether it admits anything at all.
**The next atmospheric acts are, in order: a named person reads the holdout result and adopts
that review; a named person reviews and adopts or refuses the T4E.47 declaration, which is
the ninth in a sequence of eight falsifications and carries that multiplicity; then the
feasibility measurement on motif-free blocks alone, which spends no signal evidence and which
withdraws the candidate if it exceeds its declared one-hour budget; and only then one
development pass. Neither reserve is opened by any of that.**
Do not widen the crop or tune the extractor first. The G17
successor declaration is now frozen
against the calibrated exact-pool-substitution contract; curate and justify its real partner pools
in parallel, without inventing identities or exchangeability. The repository readiness audit now
records `NO_INVENTORY`: zero explicit profiles against 48 required per correspondence. A
source-bound local ingress emits profiles only from exact delimited record bytes and methods that
exactly match a named maintainer's still-current adoption of a marginal-method review. Supply
legitimate records and adopted methods before using it. A digest-pinned batch manifest now
preflights a complete local import before writing any immutable profiles; it is an import path,
not a source of records. Once 48 profiles exist, a curation review can bind a named person's
`ESTABLISHED` or `NOT_ESTABLISHED` decision to the exact live inventory and its unmeasured
properties; at 0 profiles no such review can load. Populate that inventory before scientific
curation; do not optimise D96 first. Then close the atmospheric identity
criterion and run T4F.9
before the full T4F.6 gate. Converge the atmospheric and G17 lanes at the V1 release gate above.

TESS is now the bounded candidate source, but archive availability alone is not a G17 record.
The statistic requires an independently declared native cycle. A first sector-1 metadata page
found 48 arbitrary SPOC products (93.3 MiB predicted), but none intersected the bounded catalogue
of unique non-false-positive TOI hosts with external periods resolving eight cycles; that download
was stopped rather than turning cadence or sector span into a period. The catalogue-first path
acquired 64 period-qualified targets (124,652,160 bytes), but its pre-adoption marginal assessment
found no 48-alternative pool; the best admitted 43. Ed Bentley adopted the declared marginal
methods on 2026-09-12. A lineage audit then found that acquisition had copied the final discovery
row's period metadata onto every target. Corrected offline reconstruction from the frozen
discoveries and SHA-256-verified cache retains 112 acquisitions but refuses eight records that no
longer span eight true orbital cycles, leaving 104 source-bound profiles. Sixty-four records now
support at least 48 alternatives under the frozen bands. Inventory is
`READY_FOR_CURATION_REVIEW`; exchangeability remains `NOT_ASSESSED` pending a separate human
curation decision. The corrected review packet identifies a 59-record 48-core and a 57-record
all-pairs-admissible subset, giving each selected record 56 alternatives. The contaminated
artifacts are preserved under named `g17_tess_lineage_bug` archives. This is necessary marginal
evidence for review, not the review's conclusion.
An immutable packet-bound request now presents the exact 57 IDs, both permitted decisions and all
required human-authored fields. It remains `AWAITING_HUMAN_REVIEW`; its distinct schema cannot be
loaded as an adopted review, and no decision or signature was supplied by code.
Platform Status now displays the corrected readiness, eight refusals, assessment, exact packet,
pending request, adoption state and archived lineage-bug evidence. A reviewer can write a decision
and adopt it there as two separate acts. External discovery/acquisition and immutable evidence
rebuild remain explicit CLI jobs; their outputs are visible in the browser, but those operational
jobs are not launched from it. The next scientific action is still the maintainer's authored
exchangeability decision, followed by separate successor/inventory promotion if adopted.
The parallel object-first slice is complete: the capability registry declares existing
sample-table planner destinations, a content-addressed handoff binds the exact object,
declaration, profile and selected operation, and the browser verifies the destination planner's
schema and content digest before continuing in the existing controls. Do not broaden V1 to every
object format. G19 conversation has since been completed as a separate bounded capability.
Phase 4G/4H and external learned-forecast comparison remain post-V1 unless a release criterion
above proves they are required.

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

**Confirmed on reserved evidence, and the reservation was split rather than spent.** Two
of the four reserved partitions were opened under `t4e12-confirmatory-amendment.json` and the
result reproduces exactly:

```
CONFIRMATORY -- partitions built for the first and only time
rich  block      configs   C props  admitted    motif    split     admit
6     700-705         20       148        15       15   0.0000   0.0000
6     710-715         20       122        15       15   0.0000   0.0000
9     700-705         84       546        15       15   0.0000   0.0000
9     710-715         84       530        15       15   0.0000   0.0000
12    700-705        220      1518        15       15   0.0000   0.0000
12    710-715        219      1376        15       15   0.0000   0.0000
```

15 pairs admitted per partition -- the motif's complete clique -- out of 122 to 1518 proposed,
with both error rates 0.0000 at every richness. **720-725 and 730-735 have never been generated
and are refused in code even under `confirmatory=True`.**

**What this does not settle.** `k = S` followed a probe, so fresh scenes cannot make that choice
blind retrospectively. No confirmatory null was declared, so rejection-where-nothing-recurs
stays development evidence. `k = S` still rejects a configuration absent from one scene, so it
says nothing about a record with partial recurrence, and nothing about recurrence *across*
partitions, which is what mining needs.

**So the next step was `k < S`, declared blind, and it is now declared.** T4E.13 candidate 3
is candidate 2's criterion at `k = S - 1`, by the rule *tolerate exactly one absence* -- the
only `k` below `S` nameable without choosing a free fraction, and the smallest step that admits
partial recurrence at all. It is fixed in
`data/identity_calibration/t4e13-partial-recurrence-declaration.json` and in code, **not
adopted, and measured nowhere**: eight tests assert the absence of any measurement, adoption
record or measuring call, because a single run at any `k` below `S` before adoption would spend
the only property this candidate has.

**What is being claimed, exactly.** That the value of `k`, the rule fixing it, and every
structural choice around it were committed before any measurement at any `k` below the partition
size, on any block, planted or null. Candidate 2 could not claim that -- its `k = S` followed a
feasibility probe -- and that gap is the one thing its confirmatory pass could not close.

**What was already known, disclosed rather than hidden.** Candidate 2's exact run shows no
coincidental group reaching 6 anywhere; the greedy probe found some reaching 5 at richness 9.
So `k = S - 1` sits **at** the observed coincidental ceiling. That points away from tuning
rather than towards it: the safe choice was `k = S`, which is already known to work, and this is
the value most likely to fail. What remains unknown is how many coincidental groups reach 5,
which is the only quantity the error rate depends on.

**Two results are arithmetic, not evidence, and are derived before the fact.** Admissions are
non-decreasing as `k` falls, so candidate 3 admits a superset of candidate 2 on every partition.
Candidate 2's false split is 0.0000 everywhere, so candidate 3's is 0.0000 everywhere **before
it is run** -- acceptance condition 1 is passed by arithmetic and carries no evidential weight.
The same monotonicity says the criterion cannot be inert the way candidate 1 was, which admitted
nothing anywhere and was therefore never tested. The entire empirical content is on the
admission side, and the declaration says so in those words rather than discovering it later.

**The relaxation is small and its transfer value is small with it.** On a long record `S` is
large and `S - 1` is nearly as strict as `S`, so tolerating one absence does not solve the
transfer problem candidate 2 named -- it is one rung on that ladder. A criterion that transfers
will need `k` as a proportion of `S`, and the proportion will need its own evidence.

**Adopted, measured, and falsified.** Candidate 3 fails conditions 3 and 2: false admission
0.4000 at richness 9 and 0.8000 at richness 12 against a 0.10 bound, the shortfall widening to
x36.05, and the matched fraction rising on block 300-305 at the richest level. The declaration
predicted that direction in writing beforehand. Condition 4 held -- the null admits 0 of 1486
proposed pairs at richness 12 -- and condition 1's 0.0000 split was arithmetic, derived before
the run and declared to carry no weight.

```
CANDIDATE 3, k = 5 of 6 -- development blocks, adopted at sha256 55631fa2...
rich  block      configs   Cprops  admitted   motif   split    admit  shortfall
6     100-105         20      121        15      15  0.0000   0.0000       0.00
6     200-205         20      121        15      15  0.0000   0.0000       0.00
6     300-305         20      152        15      15  0.0000   0.0000       0.00
6     400-405         20      136        15      15  0.0000   0.0000       0.00
9     100-105         84      568        15      15  0.0000   0.0000       0.00
9     200-205         84      485        25      15  0.0000   0.4000       6.00
9     300-305         84      585        25      15  0.0000   0.4000       6.00
9     400-405         84      566        15      15  0.0000   0.0000       0.00
12    100-105        220     1446        25      15  0.0000   0.4000       6.00
12    200-205        220     1440        25      15  0.0000   0.4000       6.01
12    300-305        220     1590        75      15  0.0000   0.8000      36.05
12    400-405        220     1488        29      15  0.0000   0.4828       8.43

null: 0 admitted at richness 6, 9 and 12, of 116, 595 and 1486 proposed
```

**The margin between recurrence and coincidence in these scenes is exactly one scene wide.**
Consistency across the whole partition excludes coincidences; consistency across all but one
scene does not. Monotonicity settles the rest: `k = 4` and `k = 3` admit supersets of `k = 5`,
so no lower rung can pass conditions `S - 1` already failed. The `k` profile characterises how
fast it degrades and adjudicates nothing.

**What the blind declaration bought.** A real falsification rather than a criterion that failed
to survive its own tuning -- and that is a smaller return than a pass would have been, which is
what a severe test costs when it goes against you. The claim is spent and cannot be made again
for these scenes.

**What this does NOT license.** Not that the signature is inadequate -- it is unchanged from
candidate 2, which passed; what changed is one integer. Not that partition structure is the
wrong resource -- it is the resource candidate 2 used successfully. Not that partial recurrence
cannot be detected: this tests one relaxation, the minimal one, and a criterion built *for*
partial recurrence is a different object from one that merely tolerates it. And specifically not
that `k = S` is the right operating point for an acquired record -- candidate 2's own limitation
stands untouched, so what this establishes is that the obvious repair does not work, not that
the problem has gone away. Raising `k` back towards `S` is forbidden by the declaration and this
outcome does not unlock it. No mining radius, no discharge of T4E.8's acceptance, no closure of
D96 to D100, and nothing about `kind_recurrence`, which still has no catalogue.

**So the position is now this.** Candidate 2 works at `k = S` and does not transfer, for the
reason its own declaration gave. Candidate 3 shows the obvious repair does not work.

**T4E.14 (2026-09-10): the test bed that makes the next question answerable.** A criterion built
for partial recurrence was authorised *if that is what the work needs*. Resolving the conditional
first showed both authorised options were untestable on the evidence that existed -- recurrence
across partitions is not a distinct phenomenon in this generator, and partial recurrence was
absent from the evidence entirely. So the evidence came first.

```
T4E.14 AUDIT -- 39 partial-presence partitions built and checked
condition 1  presence counts                 MET
condition 2  no leakage by configuration count MET
condition 3  labels refused not guessed      MET, 0 refusals
condition 4  absent scenes are ordinary      MET
condition 5  reproducible                    MET
null 850-855 holds nothing anywhere          MET

j        true pairs   example planting patterns
3            3        [0,2,5] [1,3,4] [0,4,5] [2,4,5]
4            6        [0,1,3,5] [0,1,4,5] [0,1,3,4] [0,1,2,5]
5           10        [1,2,3,4,5] [0,2,3,4,5] [0,1,2,3,4] x2

configurations per scene: 20 / 84 / 220 at richness 6 / 9 / 12,
identical whether or not the scene holds the motif
```

**Why this is evidence and not a rule.** Candidates B, C, D, 1, 2 and 3 all ran on partitions
where the motif sits in *every* scene. Recurrence there is total, so no criterion has ever been
shown a scene in which a true configuration is genuinely absent -- candidate 3's false-split
column was 0.0000 partly for that reason, and its falsification, though sound, was measured on
one side only. A criterion built for partial recurrence and measured where recurrence is total
cannot fail for the right reason or pass for the right reason.

**What the audit checked, and what it deliberately did not.** It checked the test bed, not any
rule. No candidate was run on these partitions and nothing here adjudicates one.

**Two construction choices that could each have rigged a later result.** The planting subset is
uniformly random rather than contiguous: these scenes carry no ordering, so a contiguous run
would introduce temporal structure the generator does not have, and a criterion could then score
well by discovering the planting rule instead of the motif. And absent scenes hold the same
number of configurations as present ones -- 20, 84, 220 -- so presence cannot be read off the
size of a scene for free.

**The population moves with `j` and travels with the partition.** A criterion must recover
C(j,2) pairs: 3, 6 and 10, against 15 in the total-recurrence partitions. An error rate divided
by 15 on this evidence would be the quietest possible wrong number.

**Derived in advance, and not findings.** Candidate 2 requires a group spanning every scene, so
at `j < S` no such group containing the motif exists and its false split is 1.0000 here *before
it runs*. Candidate 3 recovers the motif only at `j = 5`. Both were stated in the design
declaration; neither is a discovery about a passing or a falsified candidate.

**A new reservation, made at the only honest moment.** Partial-presence blocks 880-895 were
reserved before a single partial-presence scene existed, and are refused **unconditionally** --
`ReservedPartialPresenceScene`, with no `confirmatory` flag, because no amendment opening them
exists and a flag that could open them would be a door left ajar. Blocks 720-735 stay separately
reserved and are *total*-recurrence partitions, which is exactly why a new reservation was
needed: the property under test is absent from them.

**What this does not repair.** Epoch-to-epoch recurrence. The blocks are disjoint seed ranges
over independent draws, so scenes within a block differ from one another exactly as scenes
across blocks do, and "across partitions" is not a distinct phenomenon in this generator.
Partial presence is fixed here; separated epochs remain owed by the acquired-record path, and
nothing built on these partitions may be reported as evidence about them.

**The `k` profile is complete** and shows how fast the criterion degrades below the margin:
false admission 0.0000, 0.6104, 0.9281, 0.9818 at `k` = 6, 5, 4, 3 at the richest level, with
the motif column constant at 60 throughout because recall is inherited by monotonicity. **The
null breaks between `k = 5` and `k = 4`** -- 0 pairs at both higher rungs, then 6-132 and
72-778 where nothing recurs at all. It adjudicates nothing, and any `k` chosen from it could no
longer claim a blind structural choice.

**T4E.15 (2026-09-10): candidate 4 declared, and it has nothing to tune.** Closure under the
matching -- a consistent set is admitted when no member has a mutual nearest neighbour outside
it, whatever its span. Fixed in `t4e15-closure-declaration.json` and in code, **not adopted and
measured nowhere**; ten tests assert that absence rather than intend it.

**The criterion.** Form the mutual nearest-neighbour matching as candidate C does. A set of
configurations, at most one per scene, is CONSISTENT when every member is the mutual nearest
neighbour of every other, as in candidate 2. It is CLOSED when no member has a mutual nearest
neighbour outside the set. Admit the pairs inside sets that are consistent and closed, whatever
their span.

**There is no `k`, and no parameter of any kind** -- no clique size, no span threshold, no
radius, no ratio, no fitted model, no estimated normaliser. A group of three is admitted on the
same terms as a group of six, which is what makes it a criterion *for* partial recurrence rather
than a criterion with its tolerance widened. Every falsified candidate in this sequence carried
a number that could be moved after the fact; this one carries none, which is why the declaration
can forbid adding one outright.

**What closure reads that a span threshold cannot** is the *absence* of outside partners. A
threshold sees only how far a group reaches and is blind to what its members do elsewhere in the
partition. A coincidental group whose members also match configurations outside it is not a
coherent identity however far it reaches; a true group matching nothing outside itself is one
however short it is. That is what candidate 2 was actually exploiting -- never the number 6, but
a conjunction with no loose ends.

**A design was discarded by derivation and is recorded rather than forgotten.** Ranking groups
by span and admitting the widest is the most direct reading of the maintainer's own phrasing,
and it is dead on arrival: candidate 3 established that coincidental groups reach `S - 1 = 5`,
so at `j = 3` the widest group in the partition is a coincidence and the motif is rejected
outright -- false split 1.0000 before the rule runs. Catching that in advance is the discipline
candidate 1's failure installed.

**Derivable: it cannot be inert on total-recurrence evidence.** A group spanning all `S` scenes
uses a partner in every other scene, and the matching gives at most one partner per scene, so
those are *all* of each member's partners and the motif group is closed by construction. So
candidate 4 admits at least the motif there and cannot fail the way candidate 1 did.

**Not derivable, and the whole substance of the measurement:** whether the motif group is closed
on *partial*-presence evidence. A motif configuration in a present scene may be the mutual
nearest neighbour of something unrelated in an absent scene, and closure would then reject the
motif. Nothing measured so far bears on how often that happens.

**The honest worst case is stated in advance.** Candidate 4 may admit nothing on partial-presence
evidence. That would be a complete result -- and it is *not* candidate 1's failure, which was
inert everywhere including where the answer was easy. **No prediction is offered**, because
unlike candidate 3 there is no basis for one, and a guess dressed as a prediction is worth
nothing when checked.

**Recall is the informative half here, for the first time.** It is counted against C(j,2) and
not C(S,2), and it is genuinely at risk rather than arithmetic. Hallucinated presence -- pairs
touching a scene that holds nothing -- is bounded and reported as a condition of its own,
because a rule that claims recurrence in an empty window is worse for mining than one that
misses a real occurrence, and a pooled admission rate hides which is happening.

**T4E.23: the study trail is on screen, with rendered evidence.** `StudyTrailView.tsx`, a new
"Study trail" tab, and `frontend/e2e/study-trail.spec.ts` -- **six tests passing in Chromium
against the real API and the real frontend**, with three screenshots captured under
`frontend/e2e/artifacts/`. PLAN section 5's acceptance for an interface slice is rendered
evidence with refusals demonstrated on screen rather than described, and this is that evidence
rather than a claim about it.

What the panel draws, and why each rule exists because of something this programme did:

* **A study is the chain it ran** -- declared, adopted or signed, measured -- so which result
  answered which question is not reconstructed from filenames. Thirteen studies render.
* **A verdict never appears without what it may not be used for.** The boundary renders inside
  the result, not beneath it.
* **A correction is a badge on the study.** T4E.17's three amendments and superseded signature,
  T4E.18's `CORRECTION_2026_09_10`, T4E.20's `GATE_CORRECTION` are all visible without opening
  anything, because a corrected record that reads as current is the dangerous case.
* **A question with no answer is shown, not filtered.** T4E.16 renders as *declared, not
  measured* with its withdrawal mark, and T4E.7 likewise. A view that hid T4E.16 would hide the
  cheapest result the programme produced.
* **The surface states what it will not do**, from the server's own refusal list, rather than
  implying it by an absence of buttons.

**A defect the rendering caught that the API tests could not.** The first `BOUNDARY_KEYS` list
omitted `boundary` and `acceptance_boundary` -- the plainest names of all -- so seven older
measurements rendered as *"no stated boundary; this record predates the convention"* when the
clause was right there in the record. **A viewer that under-reports a boundary is worse than one
that omits the field: it makes a false statement about the evidence, on screen.** Every
measurement now reports its boundary; the count of false "predates the convention" claims went
from seven to zero.

Two test defects were also caught by running rather than by reading: a substring selector that
matched `declared_before_measurement` instead of the *Declared* column heading, and -- after the
boundary fix put more text on the buttons -- a `close` control that matched four elements
because several boundaries contain "no closure of D96".

**T4E.22: the results become reviewable.** Until this slice `/api/v1/identity` served
`data/identity_calibration` and nothing else, so a reader could see that a study had been
*declared* and never what it *measured*. Every offset, falsified prediction and corrected
diagnosis lived in `measurements/` and in git, where no interface could reach them. **A
declaration without its result is a promise; a result without its declaration is an assertion;
only the pair is evidence.**

Three additions, all read-only and all in the router's existing idiom:

* **`GET /measurements` and `/measurements/{name}`** serve the measurement store with each
  record's verdict attached, and with **what it may not be used for attached rather than
  beside it**. Fifteen different key names have been used for that clause across this
  programme's records; all fifteen are collected rather than normalised, because renaming keys
  in committed evidence to suit a viewer would be rewriting evidence to fit its display.
* **`GET /studies`** joins each task into the chain the work actually runs -- declaration,
  adoption or signature, measurement, outcome -- so a reader is not left reconstructing from
  filenames which result answered which question, or whether the question was fixed before the
  answer was known.
* **Summaries now carry a verdict and its corrections.** A record corrected or superseded in
  the open is marked in the *summary*, because the summary is what a reader sees first and **a
  corrected record that reads as current is the dangerous case**.

Two states are shown rather than filtered, and both are load-bearing. A **declaration with no
measurement** is a question fixed and deliberately unanswered -- T4E.16 was withdrawn before
adoption by derivation, and a view that hid it would hide the cheapest result the programme
produced. A **measurement with no stated boundary** predates the convention and is listed by
name rather than passed over.

**A defect caught in verification, not in review.** The first study key split on the leading
separator, which made `t4e19_positional_error.json` a study of its own and left every
declaration reading as unanswered -- a view worse than none. It now takes the leading `t4eNN`
token under either convention, and a test pins both spellings.

**What this slice is not.** It is the API half. `IdentityDeclarationView.tsx` renders the
admissibility matrix and does not yet render studies or measurements, and PLAN section 5's
acceptance for an interface slice requires **rendered evidence captured in VERIFICATION.md,
with refusals demonstrated on screen rather than described**. That evidence does not exist for
this surface, so no interface claim is made here beyond what a client can now fetch.

**T4E.21: competition does not explain the gap, and the declared prediction is falsified.** 247
vortices planted across 39 frames at the record's own feature density, every centre known by
construction. `measurements/t4e21_faithful_background.json`.

**The gate is code this time, not prose.** `src/benchmarks/synthetic_backgrounds.py` implements
all three declared conditions -- median inside the band, no frame above the ceiling, and a median
strictly above zero -- and five tests pin it, including the exact case T4E.20 let through. The
redundant zero check is kept deliberately: a redundant condition that names the failure it was
written for is worth more than a tidy one that does not.

```
gate PASSED: median 4 features/frame (declared band [4,10], record median 7), max 9

planted 247 | recovered 167 | NEVER RECOVERED 80 (32%) | spurious 3

symmetric at real density   0.295 cells (8.2 km)
T4E.20 quiet background     0.370 cells (10.3 km)
real record                 1.630 cells (33.8 km)
by stretch: 1.0 -> 0.295    1.5 -> 0.338    2.5 -> 0.633
```

**The prediction said the offset would rise toward 1.63 cells at real density. It fell.** The
declaration named this outcome in advance: neither asymmetry alone nor competition explains the
real offset.

**What density does cost is recall, not accuracy.** Recovery collapses from 91% on the quiet
background to **68%** here -- 80 of 247 plantings never found at all -- while the features that
do survive are placed no worse. Suppression removes the maximum that would have marked a centre;
it does not displace the ones that remain. That separation was not predicted and is the
substantive finding.

**Disclosed: the gate passed at the bottom of its band.** Median 4 against the record's 7, so
competition is under-represented even though the declared band was met. That forbids claiming
density has been tested at the record's own level. It permits the observation that going from 0
to 4 features per frame did not raise the offset at all, which makes it implausible that 4 to 7
would triple it -- an argument, and labelled as one.

**About 25 km of the real 33.8 km remains unexplained.** Neither the estimator's asymmetry bias
at realistic parameters nor competition accounts for it. On this evidence the leading remaining
candidate is that **an 850 hPa relative-vorticity maximum and an agency's reported surface centre
are not the same quantity** -- which would make about 34 km an intrinsic cost of this comparison
rather than an error to be fixed, and would make restating the T4E.18 acceptance against a
defensible tolerance the correct response. That restatement is its own declaration and is not
made here. T4E.19's storm-type test found no dependence on `NATURE`, which is evidence *against*
a transition-driven separation, so this candidate is not yet comfortable either.

**T4E.20: the centroid is pulled toward the broader side, and now it is demonstrated rather than
hinted at.** 397 vortices planted at known centres on phase-randomised real frames, sweeping
scale, amplitude and asymmetry. `measurements/t4e20_synthetic_centre.json`.

**The gate failed by its own declared criterion, and the code said otherwise.** The declaration
required a background yielding feature counts comparable to the record and named "hundreds, **or
none**" as disqualifying. The phase-randomised backgrounds yield a median of **0** features where
the record yields **7**; the coded check tested only an upper bound and never implemented the
"or none" half. So the synthetic problem is *easier* than the real one -- a planted vortex faces
no competition where a real frame has seven features and a suppression rule between them.

That cuts toward the conclusion rather than away from it: an easier regime that still reaches the
observed offset at adverse parameters makes the mechanism more credible. What it forbids is the
quantitative claim that the estimator explains 33.8 km. What it supports is that the mechanism is
real and can reach that size.

```
does the offset point ALONG the stretch axis?   (0 = along it, 45 = unrelated)
  stretch 1.0   45.3 deg  n=107      offset by stretch:  1.0 -> 0.370 cells
  stretch 1.5   34.7      n=107                          1.5 -> 0.427
  stretch 2.5   18.9      n=112                          2.5 -> 0.841

symmetric, by scale:      sigma 2 -> 9 gives 0.234, 0.288, 0.580, 0.557 cells
symmetric, by amplitude:  ratio 8 -> 32 gives 0.686, 0.436, 0.207 cells
scale recovery:           planted 2.00/3.67/6.00/9.00 -> 2.07/3.79/6.01/8.93
at the record's medians:  0.368 cells = 10.2 km      (real record: 1.63 cells = 33.8 km)
at adverse parameters:    4.952 cells = 137.7 km
not found at all:         35 of 397 (8.8%), counted rather than dropped
```

**Cause A's directional prediction is confirmed cleanly**: monotone from unrelated at symmetry to
strongly aligned at 2.5x stretch. **Its magnitude prediction is confirmed too.** And the
estimator *sizes* a feature almost exactly while mislocating it, so this is a centroid problem
and not a scale one -- which also rules out the diverged-scale defect fixed earlier as an
explanation.

**What it does not settle**: that cause A accounts for the *whole* real offset. At the record's
median parameters it gives 10.2 km against 33.8 observed, and the failed gate forbids treating
the synthetic figure as a like-for-like prediction. Nor does it establish that the extractor
should be changed -- a displaced centroid on asymmetric features is a known property of windowed
centroids, and whether a better estimator exists for this field, and what it would cost
elsewhere, is separate declared work. The post-hoc southwest displacement from T4E.19 is
untouched: the stretch axis was drawn uniformly here, so this design cannot see a fixed
geographic bearing and does not claim to.

**T4E.19: the offset decomposed, and none of the three declared causes survives.** Predictions
were fixed before the measurement precisely because three causes that all produce "about 35 km"
are indistinguishable by magnitude. 176 interior observations, 162 paired within 200 km, 154
core after separating the dateline group. `measurements/t4e19_positional_error.json`.

```
core offset: median 33.8 km  q75 54.7  q90 92.2   against a catalogue radius of 15.2
             inside their own radius: 37 of 154        ratio 2.22

prediction tests (Pearson, leave-one-storm-out range; n=154 from 16 STORMS)
  A estimator bias   r(offset, feature sigma) = +0.143   [+0.025, +0.186]
  B catalogue uncert r(offset, radius)        = +0.020   [-0.033, +0.096]
  C physical         r(offset, wind)          = -0.136   [-0.241, +0.066]
  C physical         r(offset, latitude)      = +0.089   [+0.033, +0.114]  wrong sign
  C physical         by storm type: ET 36.2  MX 35.9  SS 32.7  TS 34.0 km
```

**Cause B is ruled out.** The offset does not track the agencies' own disagreement about where
the centre is, so this is *not* a case of comparing at the wrong tolerance -- which had been the
outcome that would have required no code at all.

**Cause C is not supported**, and the sharpest test is storm type: a transitioning or subtropical
system shows the same offset as a tropical one to within 3.5 km. The latitude term runs the
*opposite* way to the prediction and the intensity term straddles zero.

**Cause A survives in sign only** -- the one correlation that stays on one side of zero across
every leave-one-storm-out fit, explaining about 2% of the variance. Far too weak to carry the
explanation.

**So the declared answer is that the three causes are not separated at this sample size**, which
condition 3 anticipated and required to be reported rather than resolved by picking the most
plausible.

**What it does settle**: the offset is about 34 km, roughly 1.6 grid cells, and is largely
indifferent to storm scale, intensity, type and latitude. Whatever produces it is a property of
the extraction rather than of the catalogue or the storms.

**Fenced off as post-hoc and not adjudicated**: in grid cells the feature sits on average 0.598
south and 0.463 west of the catalogue position -- a systematic displacement of about 0.76 cells
with a mean absolute displacement of 1.634, so roughly half the offset is systematic and half is
scatter. It is *not* a fixed coordinate shift (the coefficient of variation is 0.755 in km
against 0.737 in cells; an indexing error would cluster tightly and does not). This pattern was
not among the declared three and was noticed in the data that would have to test it, so it is a
hypothesis for its own declaration -- testable on synthetic cyclone-like fields where the true
centre is known by construction.

**CORRECTION (2026-09-10): the diagnosis above was wrong, and is superseded rather than edited
away.** It claimed the extractor does not find the cyclone and named the phase-randomised
frame-maximum calibration as the cause. Both claims fail on measurement. The temperature case
was measured -- threshold 306.74 K against a field maximum of 302.5 K -- and generalised to
vorticity without ever being checked there.

**The calibration clears comfortably.** On the GITA frame the observed maximum is 1.5175e-3
against phase-randomised surrogate maxima of 1.76e-4 to 2.55e-4 -- a factor of six. Measured per
storm at the catalogue cell +/- three cells, **11 of 18 are accepted**: above threshold, a local
maximum, localised, and not off-frame.

**A second thing that was missed.** Of the five registered surrogate methods, `aaft`, `iaaft`,
`circular_shift` and `block_bootstrap` all preserve the marginal distribution, so the surrogate
frame maximum *equals* the observed maximum and the test has **no power whatever**. Only
`phase_randomise` can reject at all. That is a property of a frame-maximum statistic and is
worth knowing before any of them is proposed as a replacement.

**And raw-field extraction is better than the SWT planes, not worse** -- the reverse of what the
earlier four-storm comparison suggested.

```
                     nearest km (median)   inside radius   features/frame
raw field                        52.1          3 of 18       0-13, med 7
SWT planes                      127.1          2 of 18      73-153, med 123
```

**The two real failure modes.** Four storms sit at longitude **179.0 to 179.8** and are refused by
the R13 off-frame rule, because their own window at 2 sigma overruns the crop's eastern boundary
at 180.0; their nearest features are 1207, 2028, 2465 and 3685 km away. The crop stops at the
dateline because the request layer refuses dateline-crossing requests by design -- and Fiji,
Tonga and Samoa sit exactly there. For every storm *not* at the dateline the nearest feature is
16.6 to 99.3 km, median about 35, against a catalogue radius of 15.3: **one to two grid cells, a
factor of two to three, not an order of magnitude**.

**The verdict is unchanged -- acceptance still fails, 3 of 18 against a bar of 9 -- but the
diagnosis is what a next step would be built on**, and building a replacement calibration on the
wrong cause would have failed for a reason nobody had measured. What stands from the original
record: the variable carries the signal, the latitude band is not the cause, and the acquisition
is sound at 48 shards and 5,844 frames.

**T4E.18 acquired and FAILED its acceptance. The variable carries the signal; the extractor does
not find it.** 48 shards, **5,844 frames matching the expected calendar exactly**, 336.9 MB, 60
minutes of CDS queue, lat -58..-18, each shard carrying its own digest.
`measurements/t4e18_acceptance.json`.

```
condition 1  nearest feature inside the catalogue radius   2 of 18   (needed 9)  FAILED
condition 2  three features inside the radius              1 of 18   (needed 9)  FAILED
condition 3  features per frame 73-153, median 123                               MET
condition 4  refusals by name                                                    MET
nearest feature km: min 29.9  median 127.1  max 2055.9
```

**Three things are established and are not in doubt.** The variable carries the signal: measured
on the field before any extraction, GITA sits at the **0.02nd percentile** of its frame with the
single most cyclonic cell **one grid cell** from the catalogue position, and the declared sign
convention is correct. The crop is no longer the problem: the first probe sampled each storm's
*first* in-box observation, which is always where it entered the box and therefore always at the
boundary the extractor refuses features at -- **a sampling artefact, corrected**; re-sampled at
each storm's deepest interior observation, latitudes -20.0 to -38.6, it still fails. And the
representation is not the problem: raw extraction yields **3 to 7** features per frame against
73 to 153 through the SWT planes, and lands *further* from the storm in three of four cases.

**What it points at is the extractor's calibration.** `local_maximum_extractor` admits a maximum
only if it clears a threshold calibrated from the frame maxima of **phase-randomised
surrogates**, which preserve the power spectrum -- and for smooth geophysical fields those
surrogates routinely produce maxima as large as the observation's. On 850 hPa temperature the
threshold came out at **306.74 K against a field maximum of 302.5 K**, so nothing could clear it
at all. On vorticity, GITA -- the most cyclonic cell in its frame by a wide margin -- yields four
raw features, none within 2,900 km. The extractor's own docstring says it assumes an isotropic
peak on a flat baseline and that the honest response to a field it does not suit is **a second
registered extractor, not a special case inside this one**.

**The finding: the identity path's front end does not detect the phenomenon the signed catalogue
labels, in a field where that phenomenon is unambiguous and dominant.** That is a property of the
instrument -- not of the atmosphere, the catalogue, the crop or the variable.

**What it does not establish.** Not that vorticity is the wrong variable; the signal is
measurably present and correctly signed, and nothing extracted it. Not that the acquisition was
wasted -- it is what made the diagnosis possible. Not that a second extractor would succeed,
which is untested and would be its own declared work. `kind_recurrence` remains unevaluable, now
for a reason one layer further in than it was this morning.

**The variable works. The crop does not.** Two months of 850 hPa relative vorticity were
acquired as a probe before the declared full request -- 14 MB and about 100 seconds against
230 MB and 1.6 hours -- and tested against acceptance condition 1.
`measurements/t4e18_vorticity_probe.json`.

**Confirmed: a cyclone is a strong, correctly-signed, localised signal in this variable.** GITA
sits at the **0.02nd percentile** of its frame and the frame's single most cyclonic cell is
**one grid cell** from the catalogue position; HOLA at the 0.16th, LINDA 2.32nd, IRIS 1.20th.
The sign convention declared before acquisition is right.

**Failed anyway, and not because of the variable.** No storm of four had a feature inside its
catalogue radius. These storms sit at latitude -20.2 to -21.1 and the crop's northern edge is
**-20.0**, so they are one to four cells from the boundary, where the extractor refuses features
by design -- 34 rejected as `outside_valid_interior` on the GITA frame alone.

```
SP cyclone observations, lon 140-180, 2018-2021:   917
  inside the crop's lat band [-60,-20]:            210
  NORTH of the crop, excluded entirely:            707   (77%)
  in-crop median latitude -23.3; 74 within 2 deg of the edge
```

**The record was built for a New Zealand forecast experiment, not for cyclone identity**, and the
two purposes want different domains.

**Moving the box north fixes the geometry and costs the kind label.**

```
box (lat)      obs  w/radius  storms  interior   pairs     NATURE base rate
-60..-20       210      176      18       113    10,721          0.571
-45..-5        908      684      30       683   164,450          0.872
-40..0         900      680      30       678   163,880          0.875
```

At the same 161-pixel shape and 40-degree span the crop planner already assessed, the population
goes from 18 storms to 30 and from 113 interior observations to 683. But the `NATURE` base rate
rises to **0.872**, because at tropical latitudes almost every system is `TS`: a rule answering
"same kind" to everything would be right 87% of the time. **The label that discriminates usefully
in the southern box is close to degenerate in the northern one.**

**This is a decision, not an optimisation.** Choosing whichever box flatters a later result is the
horse race every declaration in this sequence forbids, so the trade-off is recorded and put to
the maintainer rather than resolved. And it is not free: the T4E.17 catalogue was **signed** on
terms naming this crop and a base rate of 0.571, so a different crop is a different evidence base
and the signature would have to be re-given rather than carried over.

**The full acquisition was not run.** Acquiring 230 MB onto a crop that cannot support the
evaluation is the mistake the probe exists to prevent.

**T4E.18 declared: acquire a variable in which a cyclone centre is an extractable feature.**
`data/identity_calibration/t4e18-vorticity-acquisition-design.json`, declared before any request
and **not yet acquired** -- the maintainer authorised the acquisition on 2026-09-10 and the
credentials it needs are not on this machine.

**The variable is `vorticity`** -- ERA5 relative vorticity, distinct from `potential_vorticity`
-- at 850 hPa, the same level, crop and grid as the existing record. A cyclone is a compact
near-isotropic extremum in it, an order of magnitude above a background of ~1e-5 s^-1, which is
the shape the registered extractor declares it assumes.

**Mean sea level pressure would be more commensurate and is not chosen, for a reason that is a
constraint rather than a preference.** IBTrACS records minimum central pressure directly, so
MSLP is the variable closest to the catalogue's own definition of a centre. But `cds_source.py`
refuses any dataset except `reanalysis-era5-pressure-levels` by design, and MSLP is single-level.
The chosen variable is the best available *within the machinery as it stands*, not the best in
principle, and if vorticity fails its acceptance then MSLP is the next candidate and the
acquisition layer has to be extended to reach it.

**The sign convention is declared before the data exists.** `local_maximum_extractor` finds
maxima; in the southern hemisphere cyclonic rotation is **negative** relative vorticity. A
maximum-finder on raw vorticity in this crop would locate anticyclones and miss every catalogue
storm. The field is therefore negated before extraction. Negation is a transformation this
programme chose, not a property of the data, and declaring it now is what stops it becoming a
knob turned after a disappointing result. The crop lies wholly south of the equator, so one sign
applies throughout; a crop spanning the equator is not covered.

**The window stops at 2021-12-31, deliberately.** Not acquiring 2022-2023 makes the
forecast-test reservation **physical** rather than a matter of policy: a frame that does not
exist cannot be opened by accident, by a refactor, or by someone who has not read the
constraint. The cost is a second request later, accepted knowingly -- this session has repeatedly
found that reservations enforced in code outlast reservations recorded in prose.

**Acceptance is declared before the data arrives**, which is the correction to what went wrong in
T4E.17. That catalogue was checked for independence, for the source of its radius and for the
adequacy of its population, and never for whether the record's variable could *see* what the
catalogue labels. The conditions now: the join must close by an order of magnitude, at least half
the storms must have three features inside their own catalogue radius, the extractor must be
neither starved nor swamped, and refusals are counted by name.

**What it still needs from the maintainer**: CDS credentials (`~/.cdsapirc` or CDSAPI_URL /
CDSAPI_KEY -- never pasted into a conversation and never recorded in this repository), a
one-time ERA5 licence acceptance on the CDS account, and the explicit network consent the layer
requires as a separate act.

**The join cannot be made, and the blocker has moved.** With the catalogue signed and the
extractor fixed, the remaining question was geometric: is there a constellation for a storm to
be the identity *of*? Measured in `measurements/t4e17_join_feasibility.json`.

```
nearest feature to the storm:  min 12.3 km   median 153.9 km   max 949.1 km
catalogue radius:              median 15.2 km
features per frame:            42 to 86 across the ten SWT planes

storms with >=3 features within the neighbourhood (any plane / same plane)
   25 km :  0 / 0      100 km :  1 / 0      400 km : 15 / 10
   50 km :  0 / 0      200 km :  7 / 3              of 18 storms
```

**At the catalogue's own declared radius there is usually no feature at all** -- the nearest is a
median ten times that radius away. Only 2 of 18 storms have any feature within 25 km and **none
has three**, which cardinality 3 requires. It is not a shortage of features; they are simply not
where the storms are. Widening the neighbourhood to 200-400 km until a triple appears would
substitute a radius *we* chose for the one the catalogue supplies, and "matched at its own
declared radius" is exactly what `external_reference` evidence means.

**The cause is the record's variable.** Features here come from **850 hPa temperature** -- thermal
structure. IBTrACS positions are cyclone centres, defined operationally by wind and pressure. A
warm core and a circulation centre need not coincide, and under shear or extratropical
transition they can be hundreds of kilometres apart. This is derivable in hindsight and was not
derived in advance; it is a property of the variable the record holds.

**What this does not show.** Not that the catalogue is inadequate -- it is signed, independent
and correctly specified, and nothing here bears on its labels. Not that the signature or any
criterion fails; none was run. Not that `kind_recurrence` is unevaluable in principle -- only
that it is unevaluable **against this record**, whose single variable is 850 hPa temperature.

**What would unblock it** is a record variable in which a cyclone centre is an extractable
feature: relative vorticity, mean sea level pressure, or 10 m wind. That is a new CDS
acquisition and a new record, and it is the maintainer's decision rather than a consequence of
this measurement.

**T4E.17 (2026-09-10): the catalogue exists, and the primary target is evaluable for the first
time.** Network access was granted and used for this alone. IBTrACS v04r01 South Pacific subset,
DOI 10.25921/82ty-9e16, bound by sha256 `631f76b9...` and not committed.

```
T4E.17 -- IBTrACS v04r01, South Pacific subset, sha256 631f76b9..., 35,482,417 bytes
matching domain: the record's own crop, lat -60..-20, lon 140..180, 850 hPa
development window: 2018-01-01 .. 2021-12-31   (2022-2023 stays closed)

in-box observations at synoptic hours   210     from 20 distinct storms
cross-storm pairs (kind_recurrence)  20,241     within-storm pairs excluded
same kind by NATURE                   7,907     base rate 0.391
same kind by USA_SSHS                 3,679     base rate 0.182
storms per season 2018-2021           8, 4, 2, 6
```

**Why this is the primary target and why it has never been evaluable.** `kind_recurrence`
admits `external_reference` evidence *alone* -- "a signed, frozen catalogue matched at its own
declared radius" -- and no such catalogue has existed in this programme. Every measurement to
date has been on record-derived proxies or synthetic scenes. Seven criteria were declared and
measured on synthetic partitions and an eighth withdrawn; none of it touched this.

**The admissibility question is the whole difficulty, and most candidates fail it.** Blocking
indices, IMILAST-style track intercomparisons and most atmospheric-river catalogues are computed
**from reanalysis**. Against an ERA5 record they are record-derived proxies wearing a
catalogue's name, and admitting one would reintroduce precisely the circularity
`external_reference` exists to exclude. IBTrACS passes because it merges operational best-track
data from the meteorological agencies -- BoM, JMA, NHC, the Shanghai Typhoon Institute and
others -- assigned by forecasters and post-season review, not by any algorithm run over ERA5.

**The residual dependence is disclosed rather than assumed.** Best-track analysts use whatever
guidance was operationally available, which can include model fields. The labels are independent
of *this* record and of ERA5 as reprocessed here; they are not independent of numerical weather
prediction in general. That is weaker than a purely observational catalogue would give.

**The unit of independence is the storm, not the observation.** There are 20,241 pairs but only
**20 storms**, and successive six-hourly positions of one cyclone are strongly correlated. Any
interval computed as though the pairs were independent would be wrong by roughly the square root
of the clustering factor. Every rate must carry a storm-clustered interval, and the effective
sample size is nearer 20 than 20,241. This is stated first because it is the single easiest way
for a result here to be overclaimed.

**Within-storm pairs are excluded by construction.** They bear on `spatial_persistence`, a
different declared target with different admissible evidence. Counting them here would answer
the easier question and report it as the harder one.

**The kind label is fixed before any signature is computed.** `NATURE` -- the catalogue's own
storm-type classification -- adjudicates, because `kind_recurrence` asks what a system *is*;
`USA_SSHS` is an intensity ordinal and is recorded as declared characterisation only. Both base
rates were measured before declaring, so choosing the more favourable one afterwards would be a
horse race. `MX` (agencies disagreed) and `NR` (not reported) are **refusals by name**, not
classes: treating the catalogue's own uncertainty as ground truth would corrupt every rate built
on it.

**The prior odds here are nothing like the synthetic ones.** About 1:1.6 on `NATURE` and 1:4.5
on `USA_SSHS`, against **1:399** in the synthetic partitions. The 36-fold specificity shortfall
that dominated seven synthetic candidates is substantially a property of that design rather than
of the identity question, and no rate measured here may be compared directly to the synthetic
ones.

**The data is bound by digest and is not committed.** 35.5 MB of third-party data does not
belong in this repository -- the same discipline the acquired record and the market records are
held to. A re-download that fails to reproduce `631f76b9...` invalidates the evaluation and must
say so.

**2022-2023 stays closed.** The catalogue lists 393 in-box observations from 13 storms there.
That number comes from the **catalogue**, not the record: no ERA5 frame of the forecast-test
period was opened, and none may be. It is disclosed because it was seen.

**What this does NOT do.** It evaluates no criterion -- a rule for this evidence needs its own
declaration, written before these numbers shape it. It says nothing about any of the seven
synthetic candidates, none of which transfers here. It covers one basin, one 40-degree box, four
years and 20 storms, with a coarse six-value kind label two of whose values are refusals. It
approves no mining radius, discharges nothing of T4E.8's acceptance, and closes none of D96 to
D100. And the catalogue supplies identity, not physics: a criterion agreeing with it has agreed
with the contributing agencies' operational judgements, not with the atmosphere.

**Amended before signature, and the check was worth doing.** Two requirements in the design as
first written do not survive contact with the catalogue.

**The radius had no source.** The design said to take the matching tolerance from "the
catalogue's own reported uncertainty for the contributing agency". IBTrACS has **174 columns and
none of them reports position uncertainty**, so that requirement could not have been implemented
and signing it would have committed the maintainer to a matching parameter with no legitimate
source. What replaces it is supplied by the catalogue and chosen by nobody: independent agencies
report their own position for the same observation, and the radius is the greatest distance from
the catalogue's position to any contributing agency's. In this basin and window the agencies are
USA (186 observations), BOM (102), WELLINGTON (83) and NADI (67).

```
radius = max distance from catalogue position to any contributing agency
  median 15.2 km   q90 47.2   q95 89.4   max 145.2
  record grid cell at 40S: 21.3 km longitude, 27.8 km latitude
coverage: 176 of 210 observations have two or more agencies; 34 have one
```

The median radius is **below one grid cell**, so matching is tight at grid scale for half the
observations; q95 is about four cells, so the radius varies and must travel per observation
rather than be summarised -- which is what "matched at its own declared radius" requires anyway.
The **34 single-agency observations are refused by name**: the catalogue supplies no radius for
them and they are not given a default.

**The adjudicating base rate was wrong, in the flattering direction.** Applying that refusal and
the design's own MX/NR refusal leaves **176 observations from 18 storms**, and the `NATURE` base
rate is **0.571, not 0.391** -- same-kind pairs are the *majority*. A rule answering "same kind"
to everything would be right 57.1% of the time, so accuracy is a meaningless summary here and
only the two-sided error rates may be reported. `USA_SSHS` sits at 0.227 and would make any
criterion look better; **switching to it now, having seen both, is exactly the horse race the
design forbids**, and `NATURE` stays adjudicating on the principle that `kind_recurrence` asks
what a system *is* rather than how strong it is.

The original figures are superseded in place, not edited out. The check cost one verification
pass and saved a signature on an unimplementable design plus a base rate wrong by 0.18 in the
direction that would have made any later result look better than it was.

**SIGNED by the maintainer on 2026-09-10**, at design sha256 `c692ea19...` -- the amended
design, not the original. `data/identity_calibration/t4e17-external-catalogue-signature.json`
records the act; it does not perform it, and the binding to a content hash means the terms
signed cannot drift from the terms recorded.

**What the signature makes true: `kind_recurrence` is evaluable from the tool for the first
time.** Its admissible evidence is `external_reference` alone; none has existed in this
programme through T4E.7 to T4E.16, seven measured criteria and one withdrawn. The primary
scientific target has been unevaluable throughout, and is not any more.

The terms signed are the amended ones -- 176 observations from **18 storms**, `NATURE`
adjudicating at a base rate of **0.571**, a per-observation radius from inter-agency spread with
a median of 15.2 km, and 34 single-agency observations refused by name. The superseded figures
(210 observations, 20 storms, 0.391) stay visible in the design, because the correction ran in
the direction that would have flattered a later result.

**Four obligations come with it**: the publisher's citation (Gahtan et al. 2024, DOI
10.25921/82ty-9e16, with Knapp et al. 2010); the catalogue is never committed and a digest that
fails to reproduce invalidates any evaluation built on it; every rate carries a storm-clustered
interval, because the unit of independence is the storm and not the pair; and **only the
two-sided error rates may be reported**, since accuracy is meaningless at a base rate of 0.571.

**Signing evaluates no criterion.** A rule for this evidence needs its own declaration, written
before these base rates shape it, and it can borrow nothing from the synthetic sequence.

**What it needed from you was a signature, and it has one.** Code does not sign catalogues for a
person, so this is a proposal until you say otherwise.

**T4E.16 (2026-09-10): candidate 5 withdrawn before adoption, by derivation.** The size-scaled
repair was declared, and the derivation that follows adoption discipline killed it before it
cost anything: a within-partition permutation surrogate **reassembles the motif** with
probability `S!/S^S` = 0.0154 per replicate, about 0.96 over 199, so it would have rejected the
motif for a reason unrelated to the signature. A valid null must generate fresh scenes, which
returns the cost to the measured 189 hours at richness 12 alone. **A feasible null is invalid
here; a valid null is infeasible.** Nothing was measured, so nothing was added to the
accumulated multiplicity: seven criteria measured, not eight.

**T4E.15 candidate 4: adopted, measured, and falsified on five of six conditions.**

```
CANDIDATE 4 -- partial presence, 36 partitions (totals by richness and j)
false split 1.0000 on 32 of 36; recovered anything on 4
false admission 1.0000 on those 32; 0.5385 to 0.9143 on the other four
hallucinated presence 0.0000 to 0.9118, above the 0.10 bound on all but one
null admitted 3, 29, 87 of 122, 553, 1450 proposed   (required: zero)

CROSS-CHECK -- total recurrence, where candidate 2 admits exactly 15
rich  block      admitted   true  recovered   split    admit
6     100-105          20     15         15  0.0000   0.2500
6     200-205          33     15         15  0.0000   0.5455
9     200-205          44     15         15  0.0000   0.6591
12    300-305          84     15         15  0.0000   0.8214
null admitted 6, 8, 52 of 116, 595, 1486 proposed     (required: zero)
```

**FALSIFIED on five of six conditions, on both evidences.** Recall fails outright: false split
1.0000 on 32 of 36 partial-presence partitions, with the motif recovered on only four. Admission
fails everywhere -- there is no partition at any presence level or richness inside the 0.10
bound. Hallucinated presence reaches 0.9118, so the rule claims recurrence in scenes holding
nothing at up to 91% of its admissions. Both nulls admit where the answer is zero. Only the
refusal condition holds.

**The mechanism, and it is precisely backwards.** Closure rewards isolation. The consistent
group of a node is the maximal agreeing set, so a node whose only partner is one other node
forms a closed group of **two** and is admitted trivially -- the rule is most permissive exactly
where the evidence for an identity is weakest. Meanwhile a motif configuration in a present
scene usually *is* the mutual nearest neighbour of something unrelated in an absent scene, and
that single loose end breaks closure, so the motif group is rejected. It discards the strongest
evidence and admits the weakest.

**The cross-check settles what the derivation could not.** On total-recurrence evidence closure
recovers the motif -- which is arithmetic, derived before the run -- but admits 20 to 84 pairs
per block against candidate 2's 15, and breaks the null there too. So closure is not a
differently-shaped filter than `k = S`; it is a **weaker** one. It fails even where candidate 2
succeeds.

**A derivation that should have been made and was not.** That closure admits pairs trivially was
derivable before adoption. The declaration's feasibility section derived only that the criterion
*can* admit the answer -- that it is not inert the way candidate 1 was -- and never asked what
the rule does at the extremes of its own domain, where a group of two makes closure vacuous.
This is the third time this programme has met that lesson from a different direction: candidate
1's tail model was too conservative at its extreme, candidate 4's closure too permissive at its
own, and both were derivable in advance. **A feasibility check must ask what a rule does at the
smallest and largest cases it admits, not only whether it can reach the right answer.**

**What the blindness claim still bought.** It was honoured: nothing moved after the numbers
appeared, and no parameter was added to rescue anything -- there was none to add. So this is a
real falsification of closure as declared, not a rule that failed to survive its own tuning.

**What this does NOT license.** Not that the signature is inadequate -- it is unchanged from
candidate 2, which passes at `k = S`. Not that partition structure is the wrong resource. Not
that partial recurrence is undetectable: one rule was tested, and its failure mechanism is now
understood well enough to state what a successor must not do -- **it must not treat a group of
two as evidence on the same terms as a group of six**. Not that a parameterised closure rule
would fail; it might not, and it is a different candidate needing its own declaration, one that
could no longer claim blindness because this result has been seen. No mining radius, no
discharge of T4E.8's acceptance, no closure of D96 to D100, nothing about `kind_recurrence`, and
nothing about recurrence across genuinely separated epochs.

**Where the sequence now stands.** Candidate 2 works at `k = S` and does not transfer.
Candidate 3 shows one scene of slack destroys it, and below the margin the null itself breaks.
Candidate 4 shows that dropping the size requirement altogether inverts the rule. Read together
they say something none of them says alone: **the evidence a group carries has to scale with the
group**, and no criterion in this sequence makes it do so. That is a constraint on the next
declaration rather than a design for it.

**Blocks 880-895 have still never been built**, and 720-735 have still never been generated.
Both stay closed; neither was spent on a falsified candidate, because in both cases the
conditionality was recorded before the numbers existed.

**Two reservations now stand, and they are for different things.** 720-735, total-recurrence
partitions kept for a blind structural choice, never generated. 880-895, partial-presence
partitions reserved before any partial-presence scene existed and refused with no flag to open
them. Neither is spendable without its own amendment.

**Still unmoved**: the catalogue `kind_recurrence` requires, which is the primary scientific
target and remains unevaluable from the tool without a reviewed, cited source.

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

### T4E.34 ? The declaration composer ? built 2026-09-11; the commit-alone control is the point

**Built.** A declaration can be written from the surface and committed **by itself**, which is
the act that makes a later evidence bundle possible. T4E.30's survey is the reason: of sixteen
declaration-and-measurement pairs, six cannot be bundled and **not one for being declared after
the fact** -- in every case both files entered git in the same commit, so nothing can prove the
ordering. The composer supplies structure and refusals and never content: no template, no
suggested prediction, no example claim boundary. A prediction with no stated falsifier is
refused, and composing writes `DRAFTED_NOT_ADOPTED` because composing is not adopting.

**Still open.** Running a *measurement* is not on the surface. Tools like
`tools/rerun_t4e18_join.py` remain terminal commands, which is a different shape of problem: the
runs are long and want progress reporting rather than a request and a response.
See `architecture.md` section 3E.48.

### T4E.32 / T4E.33 ? Signing and convening from the surface ? built 2026-09-11

**Built.** Adoption is a button that binds the declaration's sha256, is written once, refuses a
placeholder name, and requires the affirmation typed in full -- a signature producible by one
click is producible by accident. The rule is unchanged: *code does not sign a scientific
declaration for a person*. What changed is that the rule is no longer enforced by the
awkwardness of a text editor.

One route now runs a model, and `src/api/reviews.py` says so instead of claiming otherwise. The
authorisation moved into the request rather than out of the system: `i_authorise_paid_calls` is
required, the key is read from the server's environment and never from the wire, and the call
count is stated before anything is spent. See `architecture.md` section 3E.47.

### T4E.31 ? The round-robin runner ? built 2026-09-11; TWO ATTEMPTS PARTIAL

**Built; two paid attempts ran on 2026-09-12 and neither completed validly.** Attempt 1 stopped at
the independent reassessment after six calls because it originated dissents no challenger had
raised. Attempt 2 reached 11 calls, but all four candidate responses had conceded their dissent
and the reassessment nevertheless reopened all four; the final synthesis retained them and was
refused. Both non-regenerable exchanges are preserved as partial records in `data/reviews/`.

The second refusal exposed a protocol mismatch rather than licensing another call: the closure
rule has always said a concession retires dissent on its own, while the reassessment prompt and
validator allowed every raised dissent to be reopened. They now permit only **rebutted** dissent
role identifiers, with the four-concession attempt shape pinned by a regression test.

**Next:** commit and review this correction before deciding whether a third paid attempt is worth
authorising. Do not infer a completed review or a round-robin outcome from either partial record.

**Known asymmetry, pinned not fixed.** A protocol violation returns the paid call; a response
that does not match its declared schema is refused before the call is recorded, so that one is
lost. Both were paid for. Changing it is a deliberate decision about where validation sits.
See `architecture.md` section 3E.46.

### T4E.30 ? Measurement store to evidence store ? built 2026-09-11; the first bundle exists

**Built.** On 2026-09-11 this repository held 28 measurement records, 43 declarations and **zero
evidence bundles**, so nothing the programme had measured was visible to the review layer.
Registration is proved from git rather than asserted: the commit that first added the
declaration must precede the commit that first added the measurement.

**The survey is the finding.** Ten of sixteen studies can be bundled; six cannot, and not one
for being declared late -- T4E.12, T4E.14, T4E.19, T4E.20, T4E.21 and T4E.24 each landed their
declaration in the same commit as their measurement. Nothing lifts that retrospectively.
`data/studies/t4e28-join-rerun.r7.json` carries seven entries and the ladder caps it at
`observation`, correctly, because it holds a standing contradiction and a FAIL.
See `architecture.md` section 3E.45.

### T4E.29 ? The join distribution on screen ? built 2026-09-11

**Built.** Every distance from each catalogue centre to every extracted feature, per storm, with
an exclusion applied in the open: excluded rows stay in the table and their aggregate is computed
at equal weight beside the kept one. Excluding longitude >= 178 gives a kept maximum of **315.1
km** against the 99.3 km T4E.18's correction published for that population -- the claim is
refuted by performing the exclusion it implied. A storm yielding no feature is drawn as a row and
stays in every denominator. See `architecture.md` section 3E.44.

### T4E.28 ? The join re-run ? measured 2026-09-11; reproduced, and it falsified a claim in the record

**Measured.** Both declared gates REPRODUCED and the SWT extraction parameters -- which existed
only in a `%TEMP%` scratchpad -- CONFIRMED against all 18 recorded rows. T4E.27's condition 2 is
lifted: 1 of 17 on the SWT path, 0 of 17 on the raw path, against a bar of 9, so the acceptance
now fails on both conditions rather than on one with the other refused. Condition 2 on the raw
path was measured for the first time at 0 of 18.

The re-run also falsified a claim in T4E.18's own correction block: away from the dateline the
nearest feature ranges to 315.1 km, not to 99.3 as stated, and one storm at a dateline longitude
does fine. See `architecture.md` section 3E.43.

### T4E.27 ? The T4E.18 acceptance restated ? measured 2026-09-11; the bar was not the problem

**Measured, and it closes the question T4E.21 left open by answering it in the negative.** A
tolerance built from the two components this repository can justify -- the agency's reported
radius and the extractor's own measured localisation error -- admits **2 of 17 judged storms**,
exactly as many as the original bar did, against a declared 9. See `architecture.md` section
3E.42.

**What that settles.** The original bar was wrong for three stated reasons and **replacing it
changes nothing**, so the tolerance was never what was carrying the failure. The separations
themselves are: a median nearest feature of **127 km**, with three storms at 1241, 1581 and 2056
km that no defensible bar will admit.

**CORRECTED 2026-09-11.** The restatement ran against the SWT-plane extraction path, the worse of
the two the record holds, and did not say so. The verdict stands -- the raw path puts 3 of 18
inside the catalogue radius against a bar of 9 -- but the 93.2 km residual belongs to that path,
and the claim below that the two populations are not the same problem is **withdrawn in its
strong form**: on the raw path the non-dateline median is about 35 km against T4E.19's 33.8, which
agree. What remains real is the dateline group, which T4E.19 separated and T4E.18 does not.

**And it separates two things that were being treated as one.** T4E.19's diagnostic population --
154 interior observations paired within 200 km, dateline group removed -- has a median offset of
33.8 km. T4E.18's acceptance population -- the deepest observation per storm, unfiltered -- has a
median nearest feature of 127 km and a median unexplained residual of 93.2 km. **These are not
the same problem**, and T4E.21's ~34 km quantity-difference hypothesis cannot account for the
larger one. Any successor that quotes 33.8 km against the acceptance is quoting the wrong
population.

**What is now open.** The 93.2 km residual is unexplained and is not the quantity difference.
T4E.18's own correction already named two candidates -- the dateline edge and a one-to-two-cell
positional offset -- and T4E.19 separated a dateline group out of its core population without
ever characterising it. Nothing has measured what those three far storms are doing, and that is
the first thing a successor should ask, because it governs whether the record can support
`kind_recurrence` at all.

**A shortfall to carry forward.** Condition 2 could not be evaluated: the third-nearest distance
is not in the committed receipt, and the IBTrACS CSV that T4E.17 bound by sha256 was never
committed and is not on this machine. Any future work on this join needs that file, and needs the
join re-run so it records the full per-storm distance list rather than only the nearest.

**What is still forbidden.** Adopting a tolerance into any pipeline, selecting a point from the
reported acceptance curve, or reading the two admitted storms as successes -- both pass because
their catalogue radii are 103 and 62 km, which is the join closing because the reference is
vague.

### T4E.26 ? A null estimable from real data ? measured 2026-09-11; it works, and it costs something that is not a number

**Measured.** One round of peeled-null calibration closes a median **0.637** of the gap to the
unavailable oracle, lifts feature recovery from 0.6276 to **0.8237**, cuts never-seen features
from **121 to 39**, and more than triples intact-configuration coverage, from 0.0833 to
**0.3000**. The newly recovered plantings are half the brightness of what round zero already had,
so it reaches the faint population rather than re-finding the bright one, and nothing was lost.
See `architecture.md` section 3E.41.

**The declared failure mode is real and is most of the contamination.** 83 of 99 spurious
features are lobes the subtraction itself created, concentrated at stretch 2.5 exactly as
predicted, because `local_maximum_extractor` declares an isotropic shape model and the features
are not isotropic. Total contamination stays well inside its bar, but 84% of it is manufactured.

**The cost that is not a number, and the reason this is not simply good news.** A peeled ensemble
has the residual's power spectrum, not the frame's, so a p-value taken through a peeled cut
answers a different question from the one `NullCalibration` declares. Part of the coverage above
was bought by changing what a detection claims. Whether the residual's spectrum is the *better*
null here is a real argument and it is not settled.

**What is now open, and what it is not.** The coverage problem is improved, not solved: **70% of
configurations still hold a feature the extractor never recovers**, and in a tenth of scenes one
round closes nothing. Nothing is adopted. Three things could follow and none has been declared:

1. **Settle the hypothesis question before anything is adopted.** This is the one that blocks the
   others, because adopting a peeled null without stating its hypothesis is precisely the outcome
   T4E.26's declaration named as the worst available. It is a scientific argument to be made in
   the open, not a measurement.
2. **The artefact is a shape-model problem, and now has a measured size.** An anisotropic
   extractor, or a subtraction that uses a fitted ellipse rather than the isotropic fit, would
   attack 84% of the contamination directly. That is a change to the extractor and needs its own
   task and acceptance (standard E12 asks for a second registered extractor, not a special case
   inside the first).
3. **The bound is still the favourable case.** Every figure in T4E.24, T4E.25 and T4E.26 comes
   from scenes with identical geometry across all six. What jitter, drift and evolution cost has
   never been measured.

**What is still forbidden.** Adopting a peeled null, changing any default, running a second round
and reporting it, or ranking this against another null construction. Each is its own declaration
with its own blindness claim (R20). And these scenes are now on their fourth inspection, so a
successor that needs a blind result needs fresh evidence, not a rename.

### T4E.25 ? What the detection cut costs ? measured 2026-09-11; option 2 is narrowed, not closed

**Measured, and it removes the cheapest repair from the table.** Relaxing the family-wise level
tenfold, from 0.05 to 0.50, lifts intact-configuration coverage only from 0.0833 to 0.1500 and
feature recovery from 0.6276 to 0.7182. **65% of configurations still hold a permanently
invisible feature.** So T4E.24's coverage is a property of this detection scheme across a
declared range, not an artefact of the current operating point, and no choice of alpha rescues
it. See `architecture.md` section 3E.40.

**The reason is worth more than the result.** A scene-calibrated cut sits **3.53x** above its own
background's, because the plantings' power enters every surrogate of the planted field, while
a tenfold alpha change moves the threshold only 1.2x. **Alpha is not what sets this threshold;
the signal is.** That is the null behaving exactly as declared -- the hypothesis is *a
structureless field with this power spectrum*, and the planted field's spectrum includes the
plantings -- so it is a price, not a defect.

**What it does to the options listed under T4E.24.** Option 2 (attack coverage rather than the
criterion) survives, but not via the error level: the lever is the *self-inflation of the null*,
which is a question about what the null should be conditioned on, not about alpha. Naming that
question is the obvious next declaration and it has not been written. Options 1 and 3 are
untouched.

**What is still forbidden.** Choosing an alpha, changing a default, or altering the extractor or
the null against these numbers. Each is its own declaration with its own blindness claim (R20).

### T4E.24 ? The false absence rate ? measured 2026-09-11; it constrains what comes next

**Measured, and it moves where the effort belongs.** At `S = 6`, on the most favourable geometry
available, **121 of 414 genuinely recurrent features are recovered in no scene at all**. Relaxing
the tolerance from `k = 6` to `k = 3` moves admission only from 0.5507 to 0.6304, because no
value of `a` reaches a feature that was never extracted anywhere. `ABSENCES_TOLERATED = 1` sits
at 0.5870. **Coverage, not the tolerance, is the binding constraint on a partial-recurrence
criterion.** See `architecture.md` section 3E.39.

**What this forbids.** Choosing a tolerance against this number, here, silently. R20 forbids the
horse race, and a tolerance calibrated against a measured absence rate is its own declaration
with its own blindness claim. Nothing in this result licenses one.

**What it puts on the table, without choosing between them.** These are options for the
maintainer, not a plan already adopted:

1. **Accept the ceiling and state it.** A criterion at `S = 6` cannot admit more than ~0.63 of
   truly recurrent features on this extractor, whatever its rule. Any future acceptance bound
   stated above that number is unreachable by construction, and declaring the ceiling before the
   next criterion is declared would stop that happening a fourth time.
2. **Attack coverage rather than the criterion.** The loss tracks amplitude against the detection
   cut, not competition: recovery is 0.144 at peak-to-background ratio 8-14 and 0.919 at 26-32.
   That is a property of the cut and of `local_maximum_extractor`'s isotropic-peak model, and it
   is the first time this sequence has had a measured reason to look there. Changing the
   extractor is a declared task with its own acceptance, not an adjustment.
3. **Measure what the bound costs on real geometry.** This rate is a *lower* bound: identical
   geometry is the most favourable case, and real recurrence carries jitter, drift and evolution.
   How much worse it gets under realistic motion is unmeasured.

**What it does not touch.** T4E.8's acquired-record acceptance, the `kind_recurrence` target,
D96-D100, and the reserved blocks 720-735 and 880-895. It supersedes no earlier result,
including T4E.13's.

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

## 4. Cross-domain validation lane ? parallel science over the shared engine

G17 release remains withheld. TG17.15's replacement scale/shape inference is implemented
and calibrated, and its separate successor declaration now requests that inference without
rewriting the six historical manifests. Release still requires curated real-record partner pools
with a justified exchangeability/admission argument; the frozen inventory is explicitly unresolved.
An atmospheric identity improvement does not discharge either obligation, and neither lane waits
for the other's scientific result. They converge only at the V1 release gate in section 0.

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

G18's interface programme and G19's multi-turn conversation with the evidence are complete
according to their recorded acceptance. G19 is private by default, selects only complete records
across the corpus, pins every touched revision, enforces a provider/cost contract and proves
transcript deletion leaves every claim unchanged. `roadmap_cross_domain.md` remains the broader
programme's task history; neither completion changes the atmospheric identity decision.

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
   instrument. **Narrowed by TG19.2 for the tolerance surface**: a refused bar renders as a
   result carrying its reason, the verdict element is absent rather than showing a miss, and a
   browser test requires the result section to render with no error banner. Six Chromium tests
   hold it. **Narrowed again by T4E.41 for every adoption surface**: a refusal was found to
   reach the DOM for two renders and vanish, because it shared the state a successful reload
   clears. Refusals now hold their own state and render as `role="alert"` in the acceptance,
   holdout-review and G17-review forms, with browser tests driving each to a refusal. Still
   open elsewhere, and the remaining panels are still unaudited against it.
3. ~~**No view states what a number may not be used for.**~~ **Closed by T4E.23 for the identity
   path.** `StudyTrailView` renders every verdict with its boundary attached, across the
   eighteen key names this programme has used for that clause, and six Chromium tests hold it
   with screenshots in `frontend/e2e/artifacts/`. It is closed **for this surface only**: the
   other panels are unaudited against this requirement, and gap 2 -- a refusal reading as
   absence elsewhere in the interface -- is narrowed rather than closed.

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
- **Convening the T4E.28 panel again.** T4E.42 found that one ran unintentionally from a
  browser test and recorded a complete finding; it is archived, not adopted. A real attempt
  still requires explicit paid-call authorisation from the maintainer, and the archived
  record must not be promoted into the study to avoid making one.
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
