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
