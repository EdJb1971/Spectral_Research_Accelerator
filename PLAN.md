# PLAN — what remains to make SpectralEarth truthfully useful

**This document is the plan, and only the plan.** It says what is left and in what order. It is
not the status of record and it holds no evidence: `roadmap.md` and `roadmap_cross_domain.md` are
the task history, `architecture.md` holds the defect ledger and the design, and `VERIFICATION.md`
holds the measurements. When this document and one of those disagree, **they are right and this
is stale.**

Last revised 2026-09-07.

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
INVALID (T4F.6); cross-region generalisation (T4F.7); refutable follow-up proposals (T4F.8).
4,294 backend tests, 29 documentation guards, mutation testing on every Phase 4 module.

**Never done:** the instrument has never produced a scientific claim about the atmosphere. Not
one. Everything above is apparatus.

**Why not**, in the order the causes actually run:

```
  D97   the identity calibration has no valid input on a real record
    |     a tolerance calibrated from "replicates" that a real record cannot supply
    v
  D96   the identity step cannot process a real record
    |     because that tolerance admits most pairs, the tolerance graph is one
    |     component, and clustering it is quadratic-and-worse
    v
  the T4D-T4F.5 mining pass has never run on the acquired record
    |
    v
  T4F.6's gate has never been adjudicated  ->  Phase 4G is gated
```

The chain matters. It was read the other way round until 2026-09-07, and fixing D96 first would
have made an unfounded answer arrive faster.

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

Points 1 and 2 are the open ones. 3, 4 and 5 are built and unexercised on real data.

---

## 3. The ordered plan

Each item names its acceptance. Nothing here is started unless it says so.

### T4E.6 — Publish what the tolerance admits *(fixes half of D97)*

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

### T4E.7 — Calibrate without replicates *(fixes the rest of D97)*

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

### T4F.9 — The pilot: one honest end-to-end result

Run the whole loop on a **declared** design small enough to finish and large enough for the
statistics: climatology, anomalies, tracking, constellations, identity, sequences, precursors,
projection, and the gate. Declared as a pilot, explicitly **not** T4F.6's acceptance.

*Acceptance:* a gate verdict — PASS, FAIL or INVALID — with a receipt naming the frozen catalogue
digest, the declared design and every figure's provenance. **A FAIL or an INVALID is a success
for this task.** What is not acceptable is a verdict nobody can interpret.

*Blocked on:* T4E.7 (an identity worth adjudicating), plus two maintainer declarations that are
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

*Why after T4E.7:* scaling should follow knowing what is being scaled. If T4E.7 changes the
radius, the component structure changes with it and the ceiling has to be re-measured anyway.

### T4F.6 — Run the gate *(unblocks Phase 4G)*

The full mining pass over the acquired record and a real adjudication. The declaration is written
and digested (`data/gate_receipts/t4f6-tasman-mining-declaration.json`), and the training-only
climatology is fitted and saved. What remains is everything above, plus the maintainer's two
declarations.

*Known and already measured:* `blocking-onset` is permanently `unassessable` on this crop for a
physical reason — its envelope runs to 6,000 km and the crop is 2,226-4,184 km wide zonally. The
`upper-level-pv-precursor` entry is unassessable too, on a single-level record. Both are correct
answers, and neither is a failure.

---

## 4. The fork this plan is waiting on

T4E.7 will answer a question that decides the architecture, and it is worth naming now so the
answer is not a surprise:

**Can identity be discovered on this data, or must it be declared?**

Three findings already point one way. T4F.7 measured that the signature is rotation-invariant by
construction and cannot separate band orientations. Both the T4F.5 and T4F.7 suites built their
catalogues by declaration because clustering would not separate what they needed. And D97 shows
the calibrated radius admitting most of the record.

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
