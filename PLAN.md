# PLAN — what remains to make SpectralEarth truthfully useful

**This document is the plan, and only the plan.** It says what is left and in what order. It is
not the status of record and it holds no evidence: `roadmap.md` and `roadmap_cross_domain.md` are
the task history, `architecture.md` holds the defect ledger and the design, and `VERIFICATION.md`
holds the measurements. When this document and one of those disagree, **they are right and this
is stale.**

Last revised 2026-09-08.

---

## 0. The one question this plan serves

> **What stands between here and the first trustworthy finding this instrument has ever produced
> about the atmosphere?**

Not a feature, not a phase, not a passing suite. A finding: a statement about the real record
that a stranger could check and that would have come out differently if it were false.

Everything below is ordered by that question. Work that does not move it is named in section 5
so that its absence from the top of the list is deliberate rather than forgotten.

---

## 1. Where the instrument actually stands

**Built, tested and honest:** the acquisition line through to a real 8,764-frame ERA5 record
(T4C.5k/m, PASS at T4C.6); feature tracking and cross-scale association (T4D); constellations,
invariant signatures and identity (T4E.1-3); frequency, sequences, precursor tests against a
surrogate null with multiplicity control (T4E.4, T4F.1-3); bidirectional queries (T4F.4);
evidence projection back onto the map (T4F.5); a physical gate that can return PASS, FAIL and
INVALID (T4F.6); cross-region generalisation (T4F.7); refutable follow-up proposals (T4F.8); and a
clustering radius calibrated against a null with no replicates anywhere (T4E.7). The record does
hold replicates under the strictest reading of identity, and T4E.8's first slice measured what they
say (see section 3).
32 documentation guards, mutation testing on every Phase 4 module.

**Never done:** the instrument has never produced a scientific claim about the atmosphere. Not
one. Everything above is apparatus.

**Why not**, in the order the causes actually run:

```
  D99   a tracked feature changes wavelet band between adjacent frames more often
    |     than it keeps it  (4,160 of 6,838 same-configuration pairs, 60.8%),
    |     and scale-specific signing puts that into the comparable vector
    |
  D100  the strengths block is weighted as a discriminator and is a coin flip
    |     (AUC 0.5053 over 6,650 pairs)
    v
  D97   the identity calibration has no defensible radius on a real record
    |     the replicate route was thought to have no valid input -- it has 6,838
    |     labelled pairs, which give 73% grouped against 27% admitted; the null
    |     route (T4E.7) works on synthetic data and returns nothing on the record
    v
  D96   the identity step cannot process a real record
    |     because the replicate tolerance admits most pairs, the tolerance graph is
    |     one component, and clustering it is quadratic-and-worse
    v
  the T4D-T4F.5 mining pass has never run on the acquired record
    |
    v
  T4F.6's gate has never been adjudicated  ->  Phase 4G is gated
```

The chain matters. It was read from D96 upwards until 2026-09-07, and fixing D96 first would have
made an unfounded answer arrive faster. T4E.7 added a top link on 2026-09-08 by building the
calibration and finding that the null it has to run against is not clean -- and T4E.8's first
slice, the same day, replaced that link with these two. **D98 is still open and is no longer at
the top**: the record's own labels bound what any null could achieve at 73% grouped against
27% admitted, so a clean null was never going to produce a defensible radius. What was
stopping it is the band flipping and a component that does not discriminate.

---

## 2. What "truthfully useful" means here

A checkable definition, so that "done" is not a matter of taste. The instrument is truthfully
useful when all five hold:

1. **It can say something.** A mining pass completes on the acquired record and produces ranked
   precursor rules with R9's six figures.
2. **What it says is discriminating.** The identity step's two error rates are *measured and
   published*, and the radius in use is defensible against both — not calibrated from one and
   silent about the other (D97).
3. **What it says could have been otherwise.** The gate reaches a verdict that a different record
   would have changed, and the FAIL and INVALID paths are reachable on real data, not only on
   fixtures.
4. **Someone else could check it.** Every figure traces to a receipt, a digest and a declared
   design fixed before the record was read.
5. **It says only that.** No claim outside the boundary each module already publishes.

Points 1 and 2 are the open ones. 3, 4 and 5 are built and unexercised on real data. Point 2 now
has both a measurement (T4E.6) and a method that needs no replicates (T4E.7); what neither has is
a radius on the acquired record, because the only null available there does not isolate
recurrence (D98).

---

## 3. The ordered plan

Each item names its acceptance. Nothing here is started unless it says so.

### ~~T4E.6 — Publish what the tolerance admits~~ — **complete 2026-09-07**

Make the calibration two-sided. It must return, and refuse to be used without, **both** error
rates: the false-split rate it already states, and the rate at which the radius admits pairs the
record itself does not call the same configuration. It must publish both distributions, their
overlap and their separation, so any later change to the signature can be judged against a
number rather than an impression.

*Acceptance:* a tolerance cannot be obtained without both rates attached; on the acquired record
the receipt reproduces the D97 figures; and a synthetic record with a planted identity shows the
rates moving in opposite directions as the radius is swept.

*Why first:* it is small, it turns an unmeasured assumption into a published measurement, and
every subsequent decision needs the number it produces.

**Outcome.** Both rates are now published or refused by name, `best_operating_point` returns
`None` where no radius holds both, and the measurement found two things that sharpen T4E.7. The
radius is **arbitrary** — a factor of 6.7 across 25 configurations of the same record, with the
pipeline's natural choice sitting mid-distribution rather than at an extreme. And the rate the
old calibration reported is a **tautology**: it is identically zero at its own radius by
construction. On the acquired record no radius holds both rates below 10%. See `roadmap.md` for
the evidence.

### ~~T4E.7 — Calibrate without replicates~~ — **complete 2026-09-08**

A real record has no repeated measurements of one state, so the radius must be found some other
way. Calibrate against a **null**, which is this programme's own idiom already (T4F.3, and
`surrogate_null.py` exists): measure the distance distribution the record produces between
configurations it says are unrelated, and locate the radius where the observed departs from it.
That needs no replicates and works on real data.

*Acceptance:* on a synthetic record with a planted identity, the chosen radius recovers the
planted grouping; on the acquired record a radius is chosen with both error rates published; and
the refusal path is real — where the two distributions do not separate, the calibration says so
and returns no radius rather than an indefensible one.

*Open question this must answer:* whether the T4E.2 signature discriminates at all on real data.
If it does not, that is the finding, and section 4 is what follows from it.

**Outcome.** The method works and the record does not yield to it. On synthetic data the chosen
radius admits none of the pairs that are not one configuration and groups 85.6% of those that
are, without ever seeing a label. On the acquired record it returns **no radius**, at two
sampling budgets that agree, and the reason is now two named facts rather than an impression.
**D98**: the surrogate-record null produces a quarter of the record's signatures, so the only
excess it shows sits in the bulk where a difference between feature populations sits, and in the
close-pair tail the excess is *negative*. **And the ceiling**: under the strictest reading of
identity the record's 69,580 signatures come from 64,153 distinct tracked constellations seen a
mean of 1.08 times, putting the mixture fraction at 2.8e-06 against a band of 0.0387. The open
question is therefore **not yet answered** — what was measured is that this null cannot answer
it. See `roadmap.md` for the evidence.

### T4E.8 — A radius the record's own labels can defend *(fixes D98, D99, D100)*

**Slice 1 is done (2026-09-08) and reversed the task.** A signature carries `track_ids` and a
`time`, so the strictest reading of identity is already labelled in the record: 4,444 tracked
constellations are observed more than once, giving 6,838 within-key pairs. Measured against those
labels — no null, no mixture, both rates absolute — the record groups 73% of genuine repeats while
admitting 27% of unrelated pairs. **A null cannot beat labels**, so building a better one could
never have reached the original acceptance. D98 stays open and stops being the blocker.

What is blocking it is two things the census found. **D99:** a node changes wavelet band in 60.8%
of same-configuration pairs, and `scale_invariant=False` signing puts that into the comparable
vector — restricted to stationary pairs, the ones that keep their band separate at AUC 0.9692 and
the ones that flip at 0.7274, which is unrelated-pair territory with the tracks in the same place.
**D100:** the `strengths` block scores AUC 0.5053 weighted alone, a coin flip, while contributing
4.3% of same-pair squared distance against 1.3% of different-pair.

And the census found a radius. On the 124 pairs one frame apart with no band change and at most
one cell of motion, **0.0972 groups 90% of them and admits 4.5% of unrelated pairs** — the first
identity radius on the acquired record this programme can defend. It rests on 124 pairs and it is
a noise floor, so it will reject a configuration that recurs after moving, which is the thing
clustering exists to find. See `VERIFICATION.md` and
`data/identity_calibration/t4e8-replicate-census.json`.

*What remains.* The maintainer's declaration is taken: **fix the signature before building any
null.** Close D99 first — a band that flips more often than it holds defeats every downstream
radius, and the fix is a choice between stabler band assignment in tracking, scale-invariant
signing, and an explicit band-change term in the metric. Then D100, which is a declaration about
whether a carried coefficient magnitude of this bank belongs in the comparable half at all, and
not a licence to tune weights against labels. Then re-measure the labelled discrimination: the
124-pair stratum should grow as the band stabilises, and it is the benchmark everything else is
scored on.

*Acceptance:* the band-change rate over same-configuration pairs is reported and materially
reduced, or the signature no longer depends on the band; the labelled discrimination is re-measured
on the acquired record and published as two absolute rates; and the clean stratum is large enough
that its 90th percentile does not rest on a dozen pairs. A null is needed only for the **broad**
reading — different constellations that are the same kind — for which no labels exist, and it
should be built last, against a signature already known to separate the repeats it can be checked
on.

*What this still cannot do.* Under the strict reading the mixture-fraction ceiling is 2.8e-06
against a band of 0.0387, so no population-level calibration will ever see it. That bounds a
null-calibrated radius and **not** a labelled one, which needs no mixture fraction — the
distinction slice 1 turned on. The broad reading remains unmeasured and unlabelled.

### T4F.9 — The pilot: one honest end-to-end result

Run the whole loop on a **declared** design small enough to finish and large enough for the
statistics: climatology, anomalies, tracking, constellations, identity, sequences, precursors,
projection, and the gate. Declared as a pilot, explicitly **not** T4F.6's acceptance.

*Acceptance:* a gate verdict — PASS, FAIL or INVALID — with a receipt naming the frozen catalogue
digest, the declared design and every figure's provenance. **A FAIL or an INVALID is a success
for this task.** What is not acceptable is a verdict nobody can interpret.

*Blocked on:* T4E.8 (an identity worth adjudicating — slice 1 done, D99 and D100 open), plus two maintainer declarations that are
not code — a frozen, reviewed reference catalogue, and a documented cyclogenesis event declared
with its source. The draft catalogue exists and matches the record's region exactly; it needs
four envelopes read and either accepted or corrected. The code refuses to sign it, by design.

### T4E.5 — Finish the scalable identity step *(fixes D96)* — **IN PROGRESS**

Exists in the tree, exact where tested (identical receipts against the original on the acquired
record at 79x, 200x and 393x), and **not finished**: 26 of 37 mutants killed, eleven alive —
including the radius condition, the exact verification behind the box filter, the weights in the
vectorised distance, the family check and the pattern ordering. It also does not yet close D96:
the cross-distance matrix is dense per component, so one component of 843,000 needs 5.7 TB.

*Remaining:* close the eleven survivors; make the cross-distances sparse, which the Lance-Williams
update supports naturally because a merged cluster's neighbours are the *intersection* of its
parents'; then re-measure the ceiling.

*Why after the calibration:* scaling should follow knowing what is being scaled. If T4E.8 changes the
radius, the component structure changes with it and the ceiling has to be re-measured anyway.

### T4F.6 — Run the gate *(unblocks Phase 4G)*

The full mining pass over the acquired record and a real adjudication. The declaration is written
and digested (`data/mining_declarations/t4f6-tasman-mining-declaration.json`), and the training-only
climatology is fitted and saved. What remains is everything above, plus the maintainer's two
declarations.

*Known and already measured:* `blocking-onset` is permanently `unassessable` on this crop for a
physical reason — its envelope runs to 6,000 km and the crop is 2,226-4,184 km wide zonally. The
`upper-level-pv-precursor` entry is unassessable too, on a single-level record. Both are correct
answers, and neither is a failure.

---

## 4. The fork this plan is waiting on

T4E.7 was meant to answer a question that decides the architecture. It did not, and the shape
of its non-answer is worth reading carefully:

**Can identity be discovered on this data, or must it be declared?**

T4E.7 tried to answer this and returned a fourth possibility that was not on the list: *not
against any null this programme currently has*. That is neither a yes nor a no, and reading it as
a no would be exactly the pre-emption this section warns against.

Four findings already point one way. T4F.7 measured that the signature is rotation-invariant by
construction and cannot separate band orientations. Both the T4F.5 and T4F.7 suites built their
catalogues by declaration because clustering would not separate what they needed. D97 shows the
calibrated radius admitting most of the record. And T4E.7 adds the fourth, which is the sharpest
of them: on the acquired record the signature pairs are *further apart* than a surrogate null's
throughout the close-pair tail — the record shows less near-identity than chance, not more.

The fourth finding is also the one most easily over-read, and D98 is why. That null produces a
quarter of the record's signatures, so the comparison is against a sparser and more homogeneous
population, and a homogeneous population's few features are generically alike. The finding is
real and it is not yet evidence about the signature.

If discovery cannot be made to work, the alternative is already built and already used: a
declared, cited, frozen catalogue that the code refuses to sign, with configurations **matched
into** it at a fixed radius. `match_into_catalogue` (T4F.7) and the T4F.6 reference catalogue are
exactly that machinery. That would be a change of scientific model, not a workaround, and it
would need its own task and its own acceptance — but it would not need new apparatus.

Do not pre-empt this. Measure first.

---

## 5. Deliberately not next

Named so their absence is a decision rather than an oversight.

* **Phase 4G (`RepresentationScore`)** — gated on the T4F.6 gate returning PASS, which is the
  point of a gate.
* **Phase 4H (learned encoder)** — optional and explicitly a ceiling estimator, not the
  deliverable.
* **Phase 5 forecast integration** — no external model has been supplied; the seam is built and
  the rest is somebody else's code.
* **D84, D85** — they govern whether an *absence* was detectable and matter only to a future
  negative result. T4C.6's PASS did not route through them.
* **D18 (cross-device agreement)** — ROCm/MPS unmeasured, CPU/CUDA parity done for T5.1a-e.
* **Rendered accessibility conformance** — the source contract is done; no WCAG level is claimed
  and none should be until a real assistive-technology pass runs.
* **Licence review** — bespoke text, never professionally reviewed, and enforcement is
  deprioritised.

---

## 6. Standing constraints

Not negotiable, and not re-litigated per task.

* **Nothing reaches the network without explicit say-so** (`roadmap_cross_domain.md` §0).
* **Analysis parameters are declared before the record is read**, digested, and amended only in
  the open. A threshold chosen because it fits a budget is a threshold the software picked.
* **The code does not sign scientific declarations.** Catalogues, events and pre-registrations
  need a person and a date.
* **A refusal beats a guess.** Every module here returns a named refusal rather than a plausible
  number, and that is the property to protect when making anything faster.
* **`data/market_records/*.csv` is never committed**; the gate binds it by sha256.
* **`ed-dev` stays local.**
