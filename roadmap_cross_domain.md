# Cross-Domain Discovery: Roadmap to a Domain-Agnostic Structural Instrument

*Strategic engineering plan for the `ed-dev` line. Companion to `architecture.md` (what exists)
and `roadmap.md` (the atmospheric programme, frozen for the `master` line at
`freeze-t4c.5h-preregistration`).*

**Research question this line is being built to answer:**

> Can heterogeneous spatiotemporal datasets from different domains be transformed into a common
> structural representation that allows recurring multiscale patterns, temporal motifs and
> cross-domain relationships to be discovered and rigorously tested — without assuming the
> relevant scale, representation, lag or domain in advance?

**And the question behind that one, which this roadmap treats as primary:**

> Is that abstraction real, or does it only look real because it has never been made to carry a
> second domain?

Everything below either serves those questions or gets cut. The programme is designed so that a
**negative answer arrives early and cheaply**. If the abstraction is false, Phase G1 says so for
the cost of a refactor, not for the cost of a platform.

---

## 0. Current position (2026-09-04, `ed-dev`)

Maintained at the top so that the frontier does not have to be reconstructed from the five
thousand lines below. `VERIFICATION.md` carries the captured output behind every figure here, and
`architecture.md` describes what exists rather than what is planned.

**Phase G17 is IN PROGRESS. Phase G18 is DONE.** Every G17 phase from TG17.0 to TG17.14 is
complete; what keeps G17 open is that TG17.10 delivered its apparatus gate with the **release
withheld**, not that a feature is unbuilt. TG17.11 to TG17.14 were written after G18 opened, so
they appear physically *after* the G18 section below. They are G17 phases.

**The release gate.** `src.core.experiment_qualification.qualification_plan()` assembles seven
gates in about 0.15 s without executing any of the measurements they read. On this tree:

| Gate | Status | What it means |
|---|---|---|
| `offline_matrix` | `NOT_RUN` | A run this call did not perform, not a measurement missing from the checkout. `execute_offline_qualification()` resolves it to the measured result: three calendar cells pass and three scale/shape cells are refused by the bespoke adapter. |
| `restart_recovery` | `NOT_RUN` | The same kind of `NOT_RUN` as above. |
| `browser_no_glue` | `PASS` | TG18.5 slice 4. Reads a recorded Playwright run bound to the source of every spec in the suite. |
| `synthetic_fifth_adapter` | `PASS` | TG17.13. A live source-edit audit plus a recorded acceptance run. Publishes a standing glue count of **1** that it deliberately does not block on. |
| `calendar_calibration` | `PASS` | TG17.12. Reads a recording bound to the declared contract and to the source that decides what was measured. |
| `scale_shape_calibration` | `REFUSED` | TG17.11, **superseded and not deleted** by TG17.15 slice 5. The old claim -- the null's resolvable sizes and its drawable sizes do not overlap, and no amount of compute closes the gap -- is recomputed on every plan and still holds. A successor null is calibrated and recorded. What still blocks: no declared manifest requests the calibrated inference, and pool exchangeability on real records is not decidable here. A refusal blocks release exactly as a failure does. |
| `live_sources` | `PASS` | TG17.14. A dated four-domain run: ERA5/CDS, Argo GDAC and MAST SPOC each demonstrating network use, and the bespoke order-book record demonstrating **no** network use. |

The verdict is `NOT_RELEASEABLE`. **Every scientific gate has now been measured** and five of the
seven read `PASS`; the one thing blocking release is `scale_shape_calibration`, and it is a
declared scientific limit rather than unfinished work. TG17.14 reached four archives in four
domains and moved the verdict not at all, which is the arrangement working: a refusal blocks
exactly as a failure does.

**`scale_shape_calibration` is what TG17.15 addressed, and TG17.15 is now complete.** The gate
refused because the null's resolvable sizes and its drawable sizes did not overlap -- a claim that
is still recomputed on every plan and still holds. Two measurements
on 2026-09-04 established that this cannot be fixed by scaling: at its minimum size of 105 the test
has **zero margin** -- one member off the floor and none of the 105 reject -- and lifting the
enumeration cap buys nothing, because each member's statistic sees only k - 1 distinct partners
however many joint reassignments exist. The gap is structural. TG17.15 proposes separating the
partner pool from the tested family, which makes resolution and multiplicity independent knobs;
its costs, slices and falsification conditions are written down there before any code is cut.

`offline_matrix` and `restart_recovery` remain `NOT_RUN` only in the sense that this call did not
perform them; `execute_offline_qualification()` resolves both.

**TG17.15 is delivered in full.** The estimand is declared, the partner pool is built with
the circularity structurally excluded rather than forbidden, and the exact pool-substitution null
runs a family at the size sealed into its pools. Slice 3 also corrected a defect in its own first
draft that the phase's own falsification conditions were written to catch: a pool sized by
`minimum_pool_size` is sized for the world in which every declared correspondence is genuine, and a
family of six that clears it can still reject nothing unless five of the six are real. The receipt
now names that number rather than reporting the favourable case as a green light.

Slice 4 measured the calibration and the null holds: on a true null the family-wise false-positive
rate is 1 of 200 with a one-sided bound of 0.024, and the exact ranks are uniform on their own
lattice rather than merely thin in the tail. Two adversarial nulls -- a sampling artefact every
record shares, and an observed partner cleaner than the pool its own bands admit -- hold the rate
too. It returned two unwelcome answers all the same. Detection collapses between planted
correlations of 0.92 and 0.84: at 0.88, 98.8% of members have an uncorrected p under 0.05 and 35.4%
survive correction, because a mixed family gives its surviving member no step-up rank to hide in.
And the admission contract narrows native duration transitively through the cadence band, so an
inventory spanning 4.05 decades yields pools spanning about one. Slice 4 also failed its own first
recorded run, on an expectation that demanded a test with no type-II error; the expectation was the
defect and was replaced by two criteria derived from the case.

Slice 5 carried that into the gate on 2026-09-05, and **TG17.15 is complete**. The gate still reads
`REFUSED` and the verdict is still `NOT_RELEASEABLE`; what moved is the reason. TG17.11's refusal is
superseded rather than deleted, and the supersession is recomputed on every plan rather than quoted:
if the enumeration cap were ever lifted past the resolvable size, the record reads `VOID` and says
the old limit removed itself instead of crediting this phase with removing it. Two blockers remain,
both computed. Every frozen scale/shape manifest declares `scale_partner_reassignment` at 200
replications -- read back from the manifests, not restated -- and the calibrated method is exact
pool substitution, which none of them requests. And pool exchangeability on *real* records is
published as **not decidable here**, with a guard asserting that blocker survives a passing
recording, because it is exactly the one that gets quietly dropped once everything else goes green.

**So scale/shape mode is now a calibrated method waiting on a declared plan and a curated
inventory, rather than a method that could not work.** That is a better position than TG17.11 left
it in and it is not a release. What would move it: a manifest that declares the per-correspondence
estimand, and an inventory of real records whose admission criterion is shown to hold rather than
assumed.

**Phase G19 is specified and not started.** A researcher meeting a refusal wants to interrogate it
with a model of their choosing, over several turns. G7's recorded-call boundary already supplies
most of what that needs; what it lacks is a conversation, and a conversation adds drift,
staleness across turns and a transcript that is the most quotable and least reproducible artefact
the platform can produce. The phase records the rule that keeps retrieval safe here -- whole
records, never fragments -- before any code is cut.

Standing instruction, unchanged by TG17.14 having been authorised once: nothing in this repository
may reach the network without the maintainer's explicit say-so.

**Also open, and independent of the above:**

* Phase 4E is DONE on the atmospheric line and **Phase 4F is under way**: T4F.1 supplies the
  timed event substrate with the observation grid that separates what was searched from what
  was found, and **T4F.2 now counts on it** -- `A4 -> A8 -> B8 -> C16` is a chain with a
  support and a confidence rather than a shape the record merely permits. The honest part is
  the denominator: an antecedent enters a confidence only if the whole window it could have
  been followed in was searched, so an occurrence the record ended before is censored rather
  than counted as unfollowed. **T4F.3 has now drawn this phase's first null**: a confidence
  is referenced to a base rate measured as a window probability rather than a per-frame one,
  to a surrogate ensemble that rotates the antecedent on the searched lattice so its own
  bursting survives and only the alignment under test is destroyed, and to a correction paid
  on the whole declared family, with a data-chosen lag referenced to the distribution of the
  maximum because a lag chosen by the data is a search. The easy null is provided and priced:
  on one unchanged record the same lift is not distinguished at p = 0.11 by the shifting null
  and is called a precursor at p = 0.01 by the scattering one, so the choice of null decides
  the finding and the receipt names which was drawn. A design that could not have rejected
  anything is refused before any counting happens, so an under-powered absence is never
  produced to be read as a negative result. **T4F.4 has now made both directions of that
  table askable without making either of them a second test.** Top-down and bottom-up are one
  selection: the direction fixes which role the queried pattern plays and which side of a
  measured scale ordering its counterpart must sit on, and nothing is recomputed -- the
  q-values are the report's own, corrected against the family declared before the record was
  read, and the shortcut of narrowing that family afterwards is priced rather than warned
  about, the same rule falling from q = 0.0417 to q = 0.0050 when it is. Coarse and fine come
  from the catalogue's own member scales as an interval order, so patterns whose scale ranges
  overlap or touch are withheld and counted rather than sorted, and the scale-invariant mode
  is refused an ordering outright because it makes every pattern's scale statistic exactly
  one. Ranked by how often it preceded the target -- the phrase the specification uses -- the
  leading answer is a pattern the null did not distinguish; ranked by the corrected p-value it
  is the planted precursor. **T4F.5 has now put that table back on the map.** A pattern
  projects to a footprint -- the parent cells inside the transform's own filter support -- and
  never to a pixel, because a detail coefficient peaks on a structure's flank: on the planted
  vortex the peak cell is not the planted cell in a single one of twenty-four frames, and the
  footprint recovers it only while the level that found the structure can still reach back to
  it. Each grid gives what it has and refuses the rest, and a rule's historical instances come
  from the same per-anchor decision its support was counted with, reconciled against its
  published figures before they are shown. **T4F.6 has now built the gate that stands between
  all of this and Phase 4G, and has not run it.** A pattern is labelled against a declared
  catalogue of known phenomena as `recognised`, `unrecognised` or `unassessable`, the third
  being the label that stops an instrument's blindness from being recorded as a fact about the
  atmosphere; the gate reaches PASS, FAIL and INVALID on the same real patterns, refuses to
  adjudicate under a catalogue no maintainer has signed, and returns INVALID rather than PASS
  when the catalogue excluded nothing, because recognition by imprecision is this task's own
  failure mode. **This remains the larger outstanding body of work in the repository.** The
  T4F.6 gate run, T4F.7-8 and all of 4G remain, and 4G is still gated: the acceptance needs a
  maintainer-frozen catalogue, a documented cyclogenesis event declared with its source, and a
  real mining pass over the acquired 8,764-frame ERA5 record that has never been performed.
  T4F.5's own second acceptance clause waits on the same run.
* Defects **D84** and **D85** are open and **D18** is partial. Ninety-two of ninety-five are
  fixed. This session found three: **D93** (a time unit dropped at the T4E.2 signing seam, T4F.1),
  **D94** (a cache key republished as a `sha256`) and **D95** (untimestamped SPOC cadences), the
  last two by TG17.14's first contact with real archives.
* The last measured **full backend run is 4,190 passed, 4 skipped, 1 xfailed**, exit 0,
  on 2026-09-06 in 0:45:28 -- the tree carrying T4F.6. It replaces the 4,103 measured after
  T4F.5 and the 4,036 after T4F.4. The rise is exactly 87: the 87 test
  functions T4F.6 added, so nothing was lost in between. Do not quote a larger figure without
  running the suite again.

**Two habits this line holds to, because both were learned by being caught out.** A guard that
passes because it cannot see what it is checking has now been met five times (D64, D74, D75, and
twice inside TG17.13) - a silence is explained before it is believed. And deterministic counts are
asserted while wall-clock is recorded under names that say nothing asserts it, because a duration
measures the machine.

---

## 1. Relationship to SpectralEarth, and the claim boundary

SpectralEarth is a rigorous atmospheric research instrument. It continues independently under
`master` for Adam's use and extension, and this line must not destabilise it.

This is a **fork, not a successor.** The two lines share history to `bce1afc` and share the
standing rules R1--R16 and standards E1--E11 in `roadmap.md`, which are inherited here in full
and are not restated. Rules added by this line continue the same numbering from R17 and E12.

Three boundaries hold for the life of this programme:

* **Weather is the first domain implementation, not the definition of a domain.** Where an
  atmospheric assumption is load-bearing it is named as such and given a general replacement, or
  it is recorded as a limit on generality. It is never hidden behind an adapter.
* **Nothing in this line may be cited as SpectralEarth evidence, and nothing here relaxes the
  atmospheric programme's claim boundary.** T4C.6 has since run on this line and returned PASS (T4C.5m, which also closed D43); that verdict
  is atmospheric evidence on the `ed-dev` fork and is not thereby a cross-domain result.
  The frozen campaign `campaigns/t4c6_nz_era5_temperature_850_v1.json`
  (`84f7b53f…75975`) is immutable on this line.
* **Generalisation is never a reason to weaken a working abstraction.** See E12.

---

## 2. Honest starting status

Established by reading the source on 2026-08-23, not by reading `architecture.md`. Line counts
are from `src/`, excluding tests.

### 2.1 What already generalises

| Component | Evidence |
|---|---|
| `src/core/registry.py` | 145 lines, zero domain references. Capability-declaring, duplicate-raising decorator registries. |
| `src/statistics/` (858 lines) | `effective_sample_size`, `correlation_test`, `stationarity_check`, `surrogate_p_value`, `surrogate_test`, `screen`, `adjust` (Bonferroni/Holm/BH/BY), `resolution_limit`, `required_surrogates`, `check_power`, and five surrogate generators — **all** typed on `np.ndarray` / `Sequence[float]`. |
| `cross_scale.py` information core | `mutual_information`, `transfer_entropy`, `lagged_mutual_information`, `decorrelation_frames`, `admissible_shifts` take `Sequence[float]`. The proposal's "information rather than correlation" requirement is already met, generically. |
| `data_layer/sources.py` | `DataSource` Protocol is `can_serve(id)` / `fetch(id) -> Any` / `why_not(id)`, with the fallback chain recorded as a scientific fact rather than a log line. |
| `climatology.harmonic_design_matrix` | Accepts arbitrary `periods_hours`. Only the *defaults* are diurnal and annual. |
| `evaluate_replication_gate` | Deterministic PASS / FAIL / INVALID over result dicts plus a frozen protocol, with design-drift detection. The hard gate G6 and G7 need, in prototype, at one rung. |
| `GateProtocol`, `GateCampaign`, `ArtifactStore` | SHA-256 preregistration, fingerprint binding, atomic no-overwrite receipts. Domain-neutral. |
| `GridSpec.pixel` | An explicit, recorded "this data has no physical metric" — the honest default a cross-domain system needs, and it already exists. |

### 2.2 Where the atmosphere is load-bearing

* **`PhysicalField.__init__` raises on anything not exactly 2D** (`field.py:13`). This is the
  single largest obstacle. It excludes 1D series, 3D volumes, irregular point sets and graphs,
  and everything above it inherits the constraint: `FieldSequence`, `CoefficientField`,
  `ScaleSignature`, the transform registry seam, `surrogate_null`.
  **Answered in TG1.4** (`src/core/sample.py`), and not by relaxing it - `PhysicalField` is
  untouched and still refuses. The obstacle was real but it was in the wrong place: nothing in
  the accepted falsification layer wanted a *field*, and once a sample can be reduced to a
  `ChannelSeries` a rank-3 domain runs the whole gate without a grid existing anywhere.
* **`GridSpec.kind` is a closed enum** `{pixel, cartesian, latlon}` that raises otherwise — a
  direct violation of standard E1. Geometry is the one pluggable thing that never became
  pluggable. **Localised in TG1.2** (`src/physical_core/geometry.py`): a registry with declared
  capabilities, and operators asking what a geometry can do. The audit called this a naming and
  extensibility problem; it was also a correctness one. `laplacian`'s `else` branch would have
  given any future non-uniform geometry the five-point Cartesian stencil and a `value per m^2`
  label — the same shape as D56, one layer up.
* **Axis roles are inferred from coordinate names.** `field.py` branches on `k in ["x","lon"]` /
  `["y","lat"]` to decide which axis to slice. **Localised in TG1.1** (`src/core/axes.py`):
  declaration beats name, name beats position, and every assignment records which. The audit
  named the site and missed the defect sitting in it - the two lists disagreed about the
  fallback, so `latitude`/`longitude` matched neither and a split returned a coordinate vector
  that no longer described its own data (**D56**).
* **`LevelBank(banks: Dict[float, ...])`** keys on pressure in hPa; `ScaleSignature.level_hpa` is
  a typed atmospheric field on an otherwise generic record. **Localised in TG1.5**
  (`src/core/level_axis.py`), and here too the audit named the naming problem and missed the
  arithmetic one underneath it: `vertical_offsets` decided `"upward" if upper < lower`, which
  is a fact about pressure sitting at the one site that reports the direction of a vertical
  precursor relationship. Right for pressure, right for depth, backwards for height.
* **`support_floor(..., advection_speed_m_s)`** is the deepest conceptual assumption, not merely
  a naming one. Rule R4's minimum admissible lag is derived from advective transport. A domain
  with no propagation speed has no such floor and needs a different admissibility rule. This is
  also where **D48** and **D53** both bit — it is already proven to be the subtle part.
  **Localised in TG1.3**, and the assumption ran deeper than this bullet says: the sweep called
  `support_floor` unconditionally, so the atmospheric floor was applied to every domain
  regardless of what it had declared. The declared floor was enforced only at the boundary,
  and the receipt reported the advective one.
* **`BANK_FAMILIES = ("swt","dtcwt")`** is a hardcoded tuple bypassing the transform registry.
* **`GateProtocol` validated its measure against a four-name tuple of wavelet measures.**
  *Found by TG0.3, missed by this audit.* An allow-list implementing a deny-rule: R3 forbids a
  measure that moves with a threshold, not one that lacks a wavelet name, and no generic domain
  can satisfy a list of wavelet vocabulary. Now a registry declaring `threshold_free` per entry
  (E1); `threshold_fraction` is still refused, by name and with its measured reason. **That the
  audit missed it is itself the finding**: assumptions of this shape are invisible from reading
  and appear the moment a second domain is actually pushed through.
* Correctly and permanently atmospheric: `cds_source`, `era5_overlap`, `zarr_source`,
  `regional_forecast`, `gate_campaign`, all of `forecasting/`.

### 2.3 What does not exist at all

`grep -rE 'SpectralFeature|FeatureTrack|constellation|motif|RepresentationScore' src` returns
**five hits across 28,081 lines, all comments saying "not implemented"**.

The entire middle of the proposed pipeline — feature extraction, canonical feature description,
tracking, constellations, motif mining — is unwritten. Phases 4D--4H of `roadmap.md` were never
started. This roadmap is therefore **not** a generalisation of existing machinery; it is the
construction of that machinery with domain-independence as a design constraint from line one.

Two assets survive from the unbuilt phases, and they are more valuable than they look:

* `benchmarks/sequences.py:229` — `advected_vortex_sequence`, gate `4D.tracking`, raising
  `NotImplementedError` with the known answer recorded (1 track, 1 birth, 0 deaths, exact
  positions).
* `benchmarks/fields.py:320` — `build_planted_configuration`, three features in an equilateral
  triangle, parameterised by `rotation_deg` / `scale_factor` / `translation` specifically so that
  *"a 4E matcher that is genuinely translation-, rotation- and scale-invariant must return the
  same pattern for all of them; one that has merely memorised pixel positions will not."*

**The acceptance tests for the hardest claim in this programme were written and committed before
the code they test.** That is the discipline this roadmap depends on, already in the repository.

### 2.4 The honest split

Approximately **25% genuinely domain-free, 40% domain-free in logic but locked to
2D-gridded-with-a-metric, 35% legitimately atmospheric.**

So neither comfortable conclusion holds. It is not true that most of the platform assumes
gridded atmospheric fields. It is also not true that weather is merely an adapter. The accurate
statement is: **weather is largely an adapter; 2D gridded space-time is not.**

### 2.5 The emergent canonical form, stated precisely

There *is* an emergent domain-independent contract, and it is humbler than the proposal imagines.
`ScaleSignature.to_matrix(measure)` yields a `(time × scale)` matrix of scalars, and
`cross_scale_dependency` tests every ordered channel pair at every admissible lag against a
surrogate ensemble with BY correction and a train/test replication requirement.

So the existing interface between "structure" and "inference" is:

> **a set of labelled scalar time series, a per-channel validity mask, and a per-channel minimum
> admissible lag.**

Any domain that can produce that can enter the accepted inference layer **today**, with no new
abstractions. Phase G0 exploits exactly this.

---

## 3. Additional Standing Rules (R17--R24)

Inherited: R1--R16 from `roadmap.md`, unchanged and unrestated. These add the constraints that
only arise once more than one domain is in play.

### R17. A domain is not a second domain until it breaks something.

A dataset that reuses the existing spine unchanged — another latlon gridded geophysical field at
the same cadence — is a second *variable*, not a second domain, and provides no evidence of
generality. Each declared domain must be recorded with **which existing assumption it violates**.
A domain that violates none is not admissible as evidence for the abstraction.

### R18. The declared family is fixed before the feature layer is allowed to enumerate anything.

This is the binding constraint on the whole programme and it is a scientific constraint, not an
engineering one.

D8 is the precedent: the hypothesis engine emitted nine "discoveries" from a nine-run sweep at a
fixed effect-size threshold, including *"strong positive correlation (r = 0.96) between grid size
and floating-point reconstruction error"*. The fix was family-wide BY correction. The current
T4C.6 family is **36 tests and already requires 3,005 surrogates** to retain power under BY.

A constellation sweep over scales × orientations × lags × representations × domains × motif
configurations produces families of arbitrary size, at which point the surrogate count needed for
any power at all becomes unaffordable — and quietly shrinking the declared family to fit the
compute budget is precisely the fraud this architecture exists to prevent.

**Rule:** every mining pass declares its family size *before* execution, and
`check_power(n_surrogates, n_tests, alpha)` must pass at that declared size, or the pass is
refused rather than run. Where the family is too large to correct honestly, the only admissible
remedies are **preregistered narrowing** or a **generate/confirm split** (mine on train, freeze
the declaration, test once on held-out). Discovering the family size after the sweep is
prohibited, and the refusal must name which remedy is required.

### R19. Structural comparison never licenses semantic comparison.

Mapping sea-surface temperature and trading volume into a common structural vocabulary makes
their *structures* comparable. It does not make the quantities comparable, and it never
authorises an inference about one from the other's units, magnitude or physical meaning. Original
units, semantics and provenance travel with every feature and are never dropped because a
canonical description exists. Any output that compares raw magnitudes across domains is a defect.

### R20. A cross-domain motif must be frozen before it is transferred.

Discover motif M in domain A; freeze its structural definition under a content hash; search for M
in domain B **without redefining M to fit B**. Independently mining several domains and then
selecting the patterns that resemble each other is not evidence, and the system must be unable to
express it: transfer requires a frozen prior definition and refuses a definition modified after
domain B was opened.

### R21. Admissible lag is justified per domain, never inherited.

The generalisation of R4. Every domain declares a lag-admissibility policy and its justification.
Advection is one registered policy, not the default. A domain with no propagation mechanism must
either declare an alternative floor with a stated basis or record that no geometric floor exists —
in which case cross-scale precedence claims are inadmissible for that domain and must be refused,
not merely caveated.

### R22. The LLM layer explains and challenges; it never promotes.

An adversarial review layer may summarise evidence, propose alternative explanations, identify
confounders, translate findings into domain language and record unresolved dissent. It may not
change a claim level, override a deterministic gate, or convert a FAIL or INVALID into anything
else. Every LLM output is stored as commentary attached to an `EvidenceBundle`, never as a field
the claim ladder reads. If deleting every LLM output changes any claim level, the layer is
implemented wrongly.

### R23. LLM outputs are recorded evidence, not reproducible computation.

Standard E4 ("determinism is mandatory and captured") **cannot be satisfied by the LLM layer**,
and pretending otherwise would be the exact class of dishonesty this platform is built against.
Sampling parameters (`temperature`, `top_p`, `top_k`) are removed on current Claude models and
rejected with a 400; identical inputs are not guaranteed to produce identical outputs, and a
reasoning trace is never returned in raw form.

**Rule:** every LLM call captures the verbatim request, the verbatim response, the exact model
id, the effort setting, the API request id and the timestamp, because the output **cannot be
regenerated**. The LLM layer is a recorded observation of a non-deterministic process, and it sits
on the far side of every gate. This limitation is declared in the UI wherever LLM commentary is
displayed.

### R24. A scientific configuration is a first-class record, not one-off glue.

If a defensible experiment requires a researcher to write a conversion script, call hidden
endpoints in sequence, edit JSON by hand or move files between domain-specific tools, the
cross-domain abstraction is unfinished. Every scientific choice must be expressible in one
validated experiment manifest, reachable through both the public API and the UI, frozen before
the affected data are opened, and carried into the receipt.

Domain-specific mathematics remains legitimate; domain-specific orchestration does not. A new
conforming domain adapter may add declarations, acquisition controls and a structural translator,
but it may not require a branch in the experiment runner or a bespoke page. The acceptance test
is operational: a scientist can preflight, freeze, run, inspect, resume and export a complete
experiment from a clean browser session without a terminal, handwritten glue or undeclared
defaults.

### R25. A refusal no call can reach is a defect, not a safeguard.

Found three times in a row -- in G0's `require_declared`, and then in T4D.3 and T4E.1/T4E.2 --
and it is a defect rather than harmless duplication because it reads like a guard and guards
nothing. A second copy of a check that an earlier layer already performs makes the reviewer
believe a class of input is handled here when it is refused elsewhere, so the real refusal's
message and its coverage go uninspected.

**Rule:** every refusal must be demonstrated by a test that reaches it. If measurement shows
another mechanism fires first, the unreachable branch is deleted and replaced by a test naming
*which* mechanism fires and what it says. The deletion is recorded in the docstring, so the
absence is a decision rather than an omission.

### R26. An operating point is measured against replicates, never chosen.

R16 says a tolerance comes from the mathematics rather than from what the code happens to pass.
Its counterpart applies where the mathematics gives no number: a match tolerance, a noise floor,
an admission threshold. A plausible-looking constant is an author's opinion wearing the costume
of a measurement, and it silently sets the false-positive rate of everything downstream.

**Rule:** any such constant is produced by a calibration function that re-derives it from
replicates of a known answer, is committed with the report that produced it, and is pinned by a
test asserting the constant equals what the calibration returns. Two instances stand:
TG3.4's `calibrate_match_tolerance`, and T4E.2's `AXIS_ISOTROPY_FLOOR = 1.0421`, the largest
anisotropy a known-isotropic configuration produced over 24 noise realisations. Clearing a
calibrated floor is a **minimum, not a precision claim**, and the documentation must say so
where the constant is defined.

---

## 4. Additional Engineering Standards (E12--E17)

### E12. Generalise beside, never weaken.

When an existing abstraction does not fit a new requirement, add a sibling spine; do not relax the
original's invariants. The precedent is already in the tree and it worked: `training.py`
introduced a second `(B,C,H,W)` spine beside `PhysicalField` rather than making `PhysicalField`
an untyped container, and `architecture.md` §1.2 defends that choice explicitly.

**`PhysicalField` stays strictly 2D, coordinate-aware and metric-carrying.** Any change that lets
it hold arbitrary domain data is a regression, regardless of how much duplication it would save.

### E13. Geometry is a registry.

`GridSpec.kind` becomes a registry entry with declared capabilities (`has_metric`,
`is_periodic`, `supports_radial_binning`, `area_weighting`), exactly as transforms, actions and
sources already are. Operators query capabilities rather than branching on kind. A geometry that
cannot support an operation causes an explanatory refusal, never a silently wrong number — the
D13 failure mode, generalised.

### E14. Axis roles are declared, not inferred from names.

Every axis carries `{name, role, units, periodic, ordered}`. `role` is drawn from a registry
(`time`, `space`, `level`, `member`, `category`, …). No code may infer meaning from a coordinate
being called `lat`. This is a prerequisite for E13 and for any non-geographic domain.

### E15. Every domain declares its violations.

A domain adapter's registration includes the assumptions it breaks (no metric, no propagation
speed, irregular sampling, no natural cycle, non-stationary support). Those declarations are
machine-readable, drive automatic refusals in the analysis layer, and are reported in every
receipt. This makes R17 enforceable rather than aspirational.

### E16. Admissibility is a registered policy, applied where the arithmetic happens.

Added by TG1.3, from what that slice found. Any rule of the form *"this result is only
admissible if ..."* is a registered policy with declared capabilities, not a branch on a
declaration string — and it must be applied **at the point of use**, not only at the boundary
where the caller meets it. R21's lag floor was localised to the domain entry point in TG0.2
and left assumed inside the sweep, so a domain's declared floor was enforced in the refusal and
absent from the receipt. A refusal written where the reader will meet it and an assumption
living where the number is computed are two different places, and localising one does not
localise the other.

### E17. One manifest drives the API, orchestrator, UI and receipt.

The backend owns a versioned experiment schema. Adapter registrations contribute typed parameter
schemas, defaults, units, validation, cost hints and refusal explanations; the UI renders those
contracts rather than duplicating them in domain-specific forms. The same immutable manifest is
accepted by the API, executed by the orchestrator, displayed by the run monitor and embedded in
the evidence export. If any layer maintains a second interpretation of an experiment, schema
drift has already occurred.

---

## 5. Phases

Phase order is chosen so the cheapest phase carries the most information about whether to
continue. **G0 and G1 together are the falsification test for the whole programme.**

| Phase | Question it answers | Kills the programme if |
|---|---|---|
| **G0** | Is the inference layer already domain-free? | It is not, and cannot be made so cheaply |
| **G1** | Do the assumptions localise, or are they diffuse? | The refactor changes atmospheric results |
| **G2** | Can structural features be extracted and tracked domain-independently? | The pre-committed invariance benchmarks cannot pass |
| **G3** | Can constellations be mined within an affordable declared family? | R18 refuses every useful family size |
| **G4** | Does the engine recover planted relationships and reject artefacts? | It cannot distinguish them |
| **G5** | Does a frozen motif transfer across domains? | The flagship experiment — a null here is a real result |
| **G6** | Can evidence be adjudicated deterministically at scale? | — |
| **G7** | Can an adversarial LLM layer add explanation without adding claims? | — |
| **G8** | Does domain onboarding stay cheap as domains multiply? | — |

---

### Phase G0 — Prove the inference layer is already domain-free

Cheapest possible test of the central claim. **No new abstractions.** Section 2.5 established
that the accepted inference layer consumes labelled scalar time series with a validity mask and a
lag floor. G0 feeds it such series from a non-atmospheric source and runs the *existing*,
*unmodified* gate.

**TG0.1 The channel-series contract - DONE.** Extract the implicit contract behind
`ScaleSignature.to_matrix()` into an explicit, documented type — labelled channels, shared clock,
validity mask, per-channel lag floor, provenance — that `ScaleSignature` then satisfies. No
behaviour change.
**Acceptance:** the ERA5 synthetic gate fixture produces a **bit-identical receipt** before and
after. This is the acceptance criterion for every task in G0 and G1 and is not repeated below.

**Met.** `src/core/channel_series.py` states the contract in two tiers, because the two
consumers need different amounts: `ChannelGeometry` (labels, records, provenance - what a lag
floor needs, with no data) and `ChannelSeriesLike` (plus `to_matrix` - what the sweep needs).
`ScaleSignature` satisfies both through aliases and is otherwise untouched; record *keys* stay
as they are, because they are published in every gate receipt. A test asserts that a
`ChannelSeries` holding a signature's own numbers reproduces its `cross_scale_dependency`
result exactly - config hash, p-values, effect sizes, surrogate means, Theiler windows, floors.

The two-tier split was **forced by running the code**: `gate_campaign` audits the lag floor
before any data exists and had been duck-typing that with a `SimpleNamespace`, in the same
function whose grid handling produced D53.

**D55 found here, not introduced here, and present on `master` identically.** `_run_one` seeds
the process-global torch/numpy generators and `ThreadExecutor` workers share them. Measured: 10
of 10 trials corrupted once the seed-to-draw window is held open by 10 ms of work; 0 of 80 when
the task is microseconds long and the pool effectively serialises. Breaks E4 and re-opens D12
for `execution.backend: thread` with `n_workers > 1`. Left **OPEN**: the fix changes seeded
results on both lines.

**TG0.2 A non-atmospheric channel source - DONE (offline).** One adapter producing the contract from a public
dataset with no advection and no annual cycle. Declares its violations under E15, and therefore
declares that it has **no geometric lag floor** — exercising R21's refusal path on the first day.
**Acceptance:** `cross_scale_dependency` runs unmodified; the run either produces a corrected,
surrogate-referenced, train/test-replicated result or an explanatory refusal naming the missing
floor. Both outcomes are successes for G0. A silent number is a failure.

**Met, and both outcomes are exercised.** `core/domain.py` supplies `DomainDeclaration` (axis
roles, a fixed violation vocabulary, licence, lag policy) and enforces R17 by refusing a domain
that declares no violations and no floor. `analysis_engine/domain_analysis.py` refuses a
precedence claim under lag policy `none` before any computation, naming the domain, its declared
violation and the two ways forward, while leaving `association_only` available and
self-labelling. `data_layer/tabular_source.py` reads a local delimited record with a content
hash and no network path.

Measured on three AR(1) channels with no grid, transform, advection or cycle, where `alpha` at
`t` sets `gamma` at `t+3`: the **unmodified** sweep recovers `alpha->gamma@3` at **0.9985 nats**
excess against **0.1458** at lag 2, implicates the bystander channel in nothing, and reports 12
tests adequately powered under BY. On three seeds of independent AR(1) channels it returns
**0 of 12** significant.

**No public dataset has been ingested.** Offline-accepted infrastructure, in the same sense
`cds_source` was before any live transfer. Nothing here is evidence about a real-world domain,
and pointing the adapter at an archive is a separate act that must also record its licence.

**TG0.3 The negative control - DONE.** The same adapter over a record with no cross-channel structure.
**Acceptance:** FAIL, not INVALID, and not PASS. An adequately powered absence must be reported
as an absence — the property T4C.3 already calibrated for the atmospheric case.

**Met, and calibrated rather than asserted.** `run_domain_gate` validates the frozen protocol
before splitting, applies the generic embargoed split `split_channel_series` (R6, embargo
returned rather than dropped), sweeps both partitions through the **unmodified**
`cross_scale_dependency`, and adjudicates with `evaluate_replication_gate` unchanged.

Measured on 900 frames of three AR(1) channels, 6-test family, BY at 0.05, 499 surrogates,
540/3/357 split:

| Record | Verdict |
|---|---|
| `alpha` at `t` sets `gamma` at `t+3` | **PASS**, replicating in both partitions |
| Independent AR(1), 60 trials | **60 FAIL, 0 PASS** — false positives bounded below **4.87%** at 95% confidence |
| A channel dropped from the family | **INVALID**, not FAIL |

The third row carries the weight: two tests corrected as though they were six is a different
experiment, and reporting its empty result as FAIL would present a design error as evidence of
absence. The 60-trial number is stated as the bound a run of 60 supports, not as a claim that
the rate is zero.

**Exit criterion — MET.** The strongest claim in the proposal, that the falsification layer is
genuinely domain-independent, is now demonstrated rather than asserted: a domain with no grid,
no transform, no advection and no annual cycle reaches PASS, FAIL and INVALID verdicts through
inference code that was not modified to accommodate it. The cost was one adapter and one
assumption that had to be localised (§2.2, item 7).

What this does **not** establish: that the *structural* layer generalises. G0 tested the
inference layer on channels somebody else defined. Phases G2 and G3 build the layer that
defines them, and that is where the abstraction is far more likely to break.

---

### Phase G1 — The abstraction audit — **COMPLETE**

Localise the atmospheric assumptions named in §2.2. **No new science.** Every task is a refactor
under the bit-identical-receipt criterion. All five tasks are done; the phase's answer is
written up under *Phase G1 closed* below.

**TG1.1 Axis roles (E14) - DONE.** Name-based axis inference is now one *registered*
convention behind a declaration, in `src/core/axes.py`. `resolve_axis_roles` applies three
stages in a fixed order - declared, registered name, trailing-axes position - and records which
one produced each answer, so a guess leaves a trace instead of vanishing into a result.
`AxisResolution.require_declared` refuses the two inferred bases outright, which is what
`DomainDeclaration.resolve_axes` uses: a sensor archive whose columns are called `x` and `y`
must not silently acquire a geometry, because a geometry licenses per-metre reporting and this
domain has no metre. `importers.inspect` / `read_field` take an optional `axis_roles` and write
the resolution into the record; `PhysicalField` takes one and propagates it through splits and
resamples.

Bit-identical: all five dimension arrangements the importer previously handled resolve to the
same spatial pair, asserted directly. Two changes are not:

*   **A finding, and the reason to do G1 by execution.** `field.py` decided row-versus-column
    from two hardcoded name lists, `["y","lat"]` and `["x","lon"]`, **which disagreed about
    the fallback**. A field spelled `latitude`/`longitude` - CF's spelling, and ERA5's -
    matched neither: a split *cloned* the longitude vector rather than slicing it, returning a
    narrowed field whose coordinate no longer described its own data. Recorded as **D56**,
    fixed here. The audit in section 2.2 named this site and did not see the defect in it.
*   **Half a spatial pair is no longer a pair.** The replaced code searched for a latitude name
    and a longitude name independently and fell back to the trailing two axes when *either* was
    missing - which for dims `(lat, time, cols)` pairs the clock with the columns and calls the
    result spatial. That is a different field, not a transposition, and nothing downstream can
    detect it. It is now a refusal. No file the atmospheric line has met takes this path.

A third, smaller one is worth recording because of where it was found: the documentation guard
required a fixed defect to name a task matching `T` plus a digit. `TG1.1` did not match. The
first fork task to fix a defect found it; reading had not.

**TG1.2 Geometry registry (E13) - DONE.** `GridSpec.kind` is a key into
`src/physical_core/geometry.py`'s `GEOMETRIES`. `pixel`, `cartesian` and `latlon` register as the
first three entries, each declaring `physical_metric`, `uniform_metric`, `spherical`,
`has_latitude` and `length_units`. Eleven branches in `grid.py` and five in `operators.py` became
methods on the registered geometry or capability queries: `is_physical` reads the declaration
instead of testing `kind != "pixel"`, `latitudes()` refuses on `has_latitude`, `gradient` asks
the geometry which way is north, `laplacian` selects Laplace-Beltrami on `spherical`.

**Acceptance met.** `test_geometry_registry.py` registers `polar_scan` - a radar PPI sweep, rows
range gates and columns azimuth - entirely from the test module, and constructs, validates,
measures, crops and serialises it through the seam. It was chosen to be awkward on purpose: its
zonal metric grows with range, so it declares `uniform_metric=False` *without* being a sphere.
A fourth geometry that is `cartesian` under another name would have proved nothing.

Bit-identical: every metric, area weight, crop, resample and provenance record of the three
builtins is asserted against the arithmetic the pre-TG1.2 code held, recomputed inline in the
test rather than compared to itself.

*   **The defect the closed enum was hiding.** `laplacian` read `if kind == "latlon": spherical
    else: cartesian_5point`. That `else` was a silent default, not a fallback: `polar_scan`
    would have been handed one representative spacing for a metric that varies by a factor of
    2.25 across the grid and returned a finite array labelled `value per m^2`. No exception, no
    warning, no NaN. It now states its precondition and refuses. `gradient`, which already
    divided by a per-row `dx_metres()`, generalised for free - the contrast between the two is
    the finding, and it is the shape to look for in TG1.3.
*   **The registry needed `params` to be open at all.** `lat0`, `lon0` and `radius_m` are
    *`latlon`'s* parameters that happen to have named fields on `GridSpec` for historical
    reasons. A geometry with a different parameterisation could not have existed. `GridSpec`
    now carries `params`, a sorted tuple of geometry-specific scalars, and a geometry that
    requires one refuses construction without it.
*   **One deliberate behaviour change.** `from_coords` resolves names through TG1.1's registry,
    so `latitude`/`longitude` spelled in full is recognised as a sphere. It previously matched
    only the abbreviation, and such a field received a **pixel** grid - a spherical metric
    present in the data and discarded. Same family as D56. Nothing here builds a `PhysicalField`
    with those spellings, so no existing result moves.

**D55 closed in the same slice, the expensive way.** Per-task random streams
(`src/core/randomness.py`), bound to a `contextvars.ContextVar` for exactly one task's duration.
The cheap fix - refusing `n_workers > 1` when seeds are supplied - was rejected: it removes the
symptom and leaves process-global randomness inside a unit of work the platform is explicitly
allowed to run concurrently. Torch values are unchanged. **D57** fell out of it: `perturb_field`
advertised a `seed` parameter in its registry entry and dropped it on the floor, so a request
for a reproducible perturbation was honoured in appearance only.

And the finding that is larger than either defect: `test_sweep_is_byte_identical_across_backends`
- the acceptance criterion for "a sweep is byte-identical across executor backends" - ran a
pipeline containing **no random draw at all**. It certified a reproducibility claim with a
payload that had no seed-dependent behaviour to get wrong, which is why it could not catch D55.
The sweep now carries a noise step and a second test changes the root seed to prove the results
move. Three consecutive slices in which execution found what reading missed; the first in which
the thing that missed it was a test.

**TG1.3 Lag-admissibility policies (R21) - DONE.** `src/core/lag_policy.py` holds
`LAG_POLICIES`, and `advective`, `declared` and `none` register as its first three entries.
Each declares `precedence_admissible`, `floor_from_declaration`, `floor_from_geometry`,
`requires_channel_support` and the `parameters` it consumes. `support_floor` moved into the
module unchanged - the D48/D53 physical-grid reconstruction with it - as the advective
policy's implementation, and `cross_scale` re-exports it for the gate modules that audit a
frozen atmospheric plan before any data exists.

Every branch on the policy name is gone: `DomainDeclaration.__post_init__` calls
`validate_declaration`, `precedence_admissible` reads a capability, `analyse_precedence` binds
the policy and asks it to check the lag family, and `cross_scale_dependency` takes the bound
policy instead of assuming one. `domain_analysis.py` names no policy at all.

**Acceptance met.** `test_lag_policy_registry.py` registers `instrument_response` - a floor
per channel from each sensor's settling time - from the test module, and runs a complete sweep
through it: declaration validation, its own parameter, a per-channel floor, its own exclusion
wording and its own fingerprint entries. It was chosen to be awkward on the axis the builtins
are not - `advective` varies per channel but only from a physical grid, `declared` needs no
data but is uniform, this one is neither - because a fourth policy that was `declared` under
another name would have proved nothing.

Bit-identical: the advective block is asserted against `support * dx / U` recomputed inline,
and the atmospheric `analysis_config` is asserted to carry exactly the keys it always carried.

*   **The defect the closed set was hiding.** The floor the *sweep* applied was always the
    advective one. Under `lag_policy='declared'` the domain's floor was enforced at the entry
    point and then every test record reported `support_floor_frames: 1` with an exclusion
    reason citing rule R4 - a rule about wavelet filter geometry, quoted to a domain that had
    declared it has no propagation mechanism. The right tests ran; the receipt described a
    different study. This is the same shape as TG1.2's `laplacian` `else`: not a fallback, a
    silent default.
*   **D58, and a gate that accepted a substitute.** `run_domain_gate` required
    `lag_policy='advective'` for a protocol that froze an advection floor, and had no
    parameter through which a speed could reach the policy that requires one - so that
    combination could not be executed at all. Separately, the replication gate's
    `require_advection_floor` checked only that *a* floor was enforced, which a declared floor
    satisfies; it now names the policy, because rule R18 fixes the design before the run
    rather than accepting a substitute during it.
*   **One deliberate behaviour change.** A domain whose policy cannot read `support_parent_px`
    is no longer required to supply it. Every domain used to pay that entry price because
    `support_floor` ran regardless of the declaration, so a logger whose floor comes from its
    reporting interval had to declare a wavelet footprint that could not reach any floor it
    applied. Required, invented, and inert.

**The finding.** Rule R21 was written to stop the atmosphere's admissibility rule being
inherited silently by domains that cannot support it. It was enforced at the domain boundary
and nowhere else, and the boundary is not where the number is used. TG0.2 built the refusal;
TG1.3 found that the sweep behind it had never been told. The pattern across G0 and G1 is now
consistent enough to state plainly: **the refusals are written where the reader will meet
them, and the assumptions live where the arithmetic happens.** Localising one does not
localise the other, and only running the code shows the gap.

**TG1.4 The sibling sample spine (E12) - DONE.** `src/core/sample.py` holds
`StructuredSample`: an array of any rank whose axes are declared `AxisSpec`s, with a geometry
deliberately absent. `PhysicalField` is untouched and a test asserts it still raises on `(16,)`,
`(4,4,4)` and `(2,3,4,5)`.

**Declaration is structural, not a check.** Roles resolve through TG1.1's registry with both
inference stages off, and an `AxisSpec` cannot exist without a role, so no constructible sample
has a guessed axis and `basis` is `declared` everywhere. An earlier draft called
`require_declared` after construction; that check was unreachable, and an unreachable refusal is
worse than none because it reads as protection. The refusal a caller actually meets is that a
bare axis name is not a role.

**The route to inference does not pass through `PhysicalField`.**
`channel_series_from_samples` reduces each frame along its non-channel axes to one scalar per
channel, giving the `ChannelSeries` that TG0.1 showed `cross_scale_dependency` actually
consumes. Reductions are a registry, each declaring the gate measure it emits - load-bearing,
because that name is what a protocol freezes and what R3 is adjudicated on: an unregistered
measure is refused at build, and two reductions emitting one measure are refused rather than
merged.

**The bridge is one-way, narrow, and refuses rather than repairs.** `to_physical_field` requires
exactly two axes, both declared `space`, in array order. A declaration whose spatial ordinals
run against array order is refused, never transposed - a transposed field still looks like a
field, and every orientation and anisotropy statistic taken from it would be wrong in a way
nothing downstream can detect. `src.core` does not import `physical_core` at module scope, so a
non-gridded domain does not pull in torch to declare an axis.

**Acceptance met by execution, not by construction.** A three-band photometer whose frame is a
`(band, detector, repeat)` block recovers a planted coupling through the unmodified sweep, finds
nothing in the AR(1) control, and passes the unmodified replication gate with the floor its own
declaration set. The pair is the test; a sample type without it is speculative generality with
a docstring.

*   **One change outside the new module, and the reason for it.** `from_channel_series`
    published the *names* of a series' provenance keys and never their values - harmless while
    every producer's real record lived on the `DomainDeclaration`, which travels into each
    result whole. The sample spine puts the arithmetic on the *series*, and two reductions may
    legitimately emit the same gate measure, so a receipt naming only the measure could not say
    which arithmetic produced it. The lineage record now carries the values, with anything it
    cannot hold named by type rather than dropped. No atmospheric receipt passes through this
    function.

**The finding.** The obstacle section 2.2 named was real, and it was in the wrong place.
`PhysicalField`'s 2D constraint blocks nothing the falsification layer needs, because that layer
was never shown a field - TG0.1 established it consumes labelled series. What the constraint
actually blocked was the *habit* of routing every domain through the field type on the way in.
Removing an obstacle here meant building a second door, not widening the first, and the type
that was said to be in the way turned out not to be on the path at all.

**TG1.5 Level as a declared axis - DONE.** `src/core/level_axis.py` holds
`LEVEL_COORDINATES`, and `pressure_hpa`, `height_m` and `depth_m` are its first three entries.
A `LevelCoordinate` declares its units and `increases_upward`, and nothing else.
`LevelCoordinate.axis_spec()` returns a declared `level`-role `AxisSpec`, the same type TG1.1
resolves and TG1.4's `StructuredSample` accepts, so "a declared level-role axis with units" is
literal here rather than nominal.

**The assumption was again deeper than the bullet.** Three sites carried a level and no two
agreed how: `LevelBank` keyed on hPa, `ScaleSignature.level_hpa` had the unit in the attribute
name, and `CoefficientField.level` had no unit anywhere. But the load-bearing line was
`vertical_offsets`, which decided direction by comparing two numbers - correct for pressure and
depth, backwards for height, at the exact site that reports whether a precursor was above or
below. Nothing shipped wrong because no height axis exists in the tree; nothing *could* have
gone right either, and `test_a_height_bank_labels_its_offsets_the_other_way_round` is the test
that could not have been written before this slice.

**The declaration travels with the number.** `zarr_source` stamps `level_axis` on the frame it
reads, because selecting an ERA5 pressure level by name is what entitles a component to declare
the coordinate; `decompose_sequence` reads it by the same rule it already read `level`;
`decompose_levels` stamps every field in a bank; `scale_signature` carries it onto the
signature. A level that arrives without a coordinate reports no units rather than being assumed
to be pressure - the honest report of a bare number, which is a real state of the tree.

*   **`level_hpa` survives as a property that can refuse.** "The 850-hPa signature" is how the
    atmospheric line talks. It now refuses a non-pressure axis rather than returning `None`,
    because a `None` there would read as "this signature has no level" about one that has a
    level in metres.
*   **One correctness change beyond the rename.** The streaming signature refused frame-to-frame
    drift in grid, coordinates, variable, level and units. The vertical coordinate joins that
    list: 500 on a pressure axis and 500 on a height axis are different frames, and a record
    that switched partway would have passed every other check because the number never moved.
*   **Receipt keys changed, deliberately, for the first time in G1.** `levels_hpa` became
    `levels`; `from_level_hpa`/`to_level_hpa`/`offset_hpa` became `from_level`/`to_level`/
    `offset` with `level_units` beside them; `ScaleSignature.summary()["level_hpa"]` became
    `level` + `level_axis` + `level_units`. TG0.1 deferred exactly this rename to TG1.5 and this
    is where it lands. Nothing hashed moves - `analysis_config_sha256`, the artifact digests
    and the frozen campaign hash are untouched - and `gate_run`, `gate_campaign`, `cds_source`,
    `era5_overlap` and `regional_forecast` keep their own `level_hpa` vocabulary, because
    section 2.2 lists them as permanently atmospheric and a rename there would be generality
    theatre.

---

**Exit criterion.** The atmospheric line runs bit-identically, and every assumption in §2.2 is
either registered, declared, or documented as an accepted limit on generality. **Any assumption
that resists localisation is written up as a finding** — that is a real answer about the
abstraction, not a failure of the phase.

### Phase G1 closed

**Against the exit criterion.** The atmospheric line runs bit-identically in every sense that
was ever claimed for it: the same numbers, the same `analysis_config_sha256`, the same artifact
digests, the same frozen campaign hash. Every assumption in section 2.2 is now registered,
declared, or recorded as an accepted limit:

| Assumption | Outcome |
|---|---|
| `PhysicalField` is strictly 2D | **Answered in TG1.4** by a sibling spine. `PhysicalField` untouched and still refusing. |
| `GridSpec.kind` is a closed enum | **Registered in TG1.2**, with capabilities. |
| Axis roles inferred from names | **Declared in TG1.1**, with the basis of every assignment recorded. |
| `LevelBank` / `level_hpa` | **Declared in TG1.5**, pressure one registered coordinate of three. |
| `support_floor(..., advection_speed_m_s)` | **Registered in TG1.3** as one lag policy of three. |
| `BANK_FAMILIES` hardcoded tuple | Registered in TG0.4. |
| `GateProtocol`'s measure allow-list | Registered in TG0.3. |
| `cds_source`, `era5_overlap`, `zarr_source`, `regional_forecast`, `gate_campaign`, `forecasting/` | Accepted limit: permanently atmospheric, and left so deliberately. |

**Four numbered defects, an unreachable refusal, and a pattern in where they were.** D55, D56,
D57 and D58, plus a `require_declared` check in TG1.4's first draft that could never fire, were
all found by *executing* a generalisation and never by reading the code. Three of them sat
inside sites the section 2.2 audit had already named and read: the axis-name lists (D56), the
geometry enum's `else` branch, and the advective floor (D58). TG1.5 repeated it once more - the
audit named `LevelBank`'s pressure keys and did not see the direction rule underneath them. The
pattern is consistent enough to state as the phase's answer: **an assumption is visible where it
is written down and invisible where it is used, and only running the code crosses that gap.**
Standard E16 is the general form.

**Is the abstraction real?** On the evidence of G1: yes, and less of it was load-bearing than
the audit believed. Four of the five obstacles were localised without weakening anything, and
the largest one - `PhysicalField` - turned out not to be on the path at all. What was genuinely
atmospheric was smaller and sharper than expected: the advective lag floor (a real physical
assumption, now one registered policy among several) and the vertical direction convention (a
real fact about pressure, now declared). Neither resisted localisation. The honest caveat is
that every second domain exercised so far is synthetic, so G1 has shown the machinery is not
atmosphere-shaped; it has not yet shown that a real second archive fits it.

---

### Phase G2 — Structural features and tracking

The first genuinely new machinery, built directly against the benchmarks committed in §2.3.

**TG2.1 The canonical feature record - DONE.** `src/core/feature.py` holds
`SpectralFeature`, `FeatureSet`, and the four quantity types the record is built from:
`Quantity` (value, units, uncertainty, kept together), `Orientation`, `Significance` and
`FeatureLocation`. All fifteen roadmap fields are present in `describe()`.

**The record enforces R19 rather than documenting it.** `compare_magnitude_to`,
`separation_to` and `elapsed_to` raise `SemanticComparisonError` across a domain or a dataset
boundary, and `FeatureSet` refuses to hold features from two domains, two variables, two
representations or two clocks. A comment saying "do not compare these" is not an invariant;
these refusals are, and each is exercised by a test that attempts the comparison.

*   **Two views, and only one crosses a boundary.** `structural_signature()` is *built* from
    the quantities that survive being stripped of their units - scale ratios, orientation,
    significance, representation - rather than filtered out of `describe()`. Adding magnitude
    back is then a visible edit to that method instead of an invisible consequence of a key
    not being removed. `scale_ratio_to` is the one comparison that is permitted across a
    domain boundary, and it still refuses when the units do not cancel: a scale in cells over
    one in metres is not dimensionless, and that is arithmetic rather than policy.
*   **Orientation was the trap.** A ridge at 170 degrees and one at 350 are the same axis; a
    vector at 170 and one at 350 are opposed. The wrap period is a property of the quantity
    and not of the number, so `ORIENTATION_CONVENTIONS` registers `axis_180` and
    `direction_360` and the arithmetic reads the declaration (standard E16). TG2.3 gates
    association on orientation difference, and that gate would have been wrong by up to a
    factor of two in one of the two cases with no way to tell which.
*   **Significance carries its own resolution.** An empirical p-value from `n` surrogates
    cannot be smaller than `1 / (1 + n)` on the `(1 + k) / (1 + n)` convention the tree
    already uses, and `Significance` refuses one that is - a record claiming `p = 1e-4` from
    999 surrogates reports a number the ensemble could not have produced, and that number
    would then be BY-corrected, ranked and published.
*   **No `uncertainty` field, deliberately.** A single uncertainty for a record holding a
    magnitude in kelvin, a location in cells, a scale in metres and an angle in degrees is a
    number with no unit and no referent. It lives with each quantity, and
    `describe()["uncertainty"]` assembles the per-quantity view - the field the list asks for,
    without a lie in the dataclass.
*   **A real user, not a docstring.** Features are built from *measured* centroids and
    *measured* widths of the `advected_vortex_sequence` benchmark - never from its recorded
    truth - and recover the known step velocity to 0.25 cells and the known scale-doubling
    ratio to 10%. This is the TG1.4 discipline: a type without a user is speculative
    generality with a docstring.

**D59, found by giving it that user.** `truth_advected_vortex` takes the vortex position
modulo `n`, so its recorded answer is a trajectory on a torus; `build_advected_vortex` draws a
plain Euclidean Gaussian that is clipped at the boundary rather than wrapped. At the
benchmark's own parameters the vortex never reaches an edge, so the two have never disagreed
and `4D.position` passes to better than a cell. Started near one, they part company by several
cells. **This is TG2.3's problem before it is anyone else's:** that slice's acceptance criterion
is *1 track, 1 birth, 0 deaths*, and a tracker on a wrapping parameterisation would see the
object fade out at one edge and appear at the other - and would be right. Left unfixed on
purpose (making the builder periodic moves every number the benchmark produces) and asserted by
a test, so the next slice meets it as a fact rather than as a surprise.

The pattern from G1 held on the first task of G2: the defect was in an asset the audit had
already read twice and quoted in section 2.3, and it was found by *running* something against
it.

**Evidence:** `src/tests/test_feature_record.py`, 36 tests. Full suite 1375 passed, 1 skipped,
1 xfailed.

**TG2.2 Feature extraction as a registry - DONE.** `src/core/extraction.py` holds
`EXTRACTORS`, `ExtractionField`, `NullCalibration`, `Candidate`, `ExtractorReport` and
`ExtractionResult`. `local_maximum` is the first entry in the registry and not the definition of
extraction.

**The framework keeps what must not vary.** An extractor returns `Candidate`s - positions,
magnitudes and scales in the units of the declared axes - and `extract()` builds the records.
`Candidate` has no `domain`, no `representation` and no `significance`, so an extractor has no
way to choose its own: attaching the representation (R8), attaching the surrogate p-value with
its ensemble size, and producing records R19 can refuse by name are exactly the things a plug-in
would otherwise vary. A second extractor registered from the test module drives the whole
pipeline without an edit to `src`, and its records carry all three anyway.

*   **The threshold is calibrated, never chosen (R3).** The cut is an order statistic of the
    distribution of the *maximum* of a phase-randomised surrogate field, so `alpha` means the
    probability that a structureless field with this power spectrum produces any reported feature
    at all. An `alpha` finer than the ensemble's `1 / (1 + n)` floor is refused *before* the
    ensemble is built - otherwise the extractor returns nothing and the receipt says the field
    was empty, which is indistinguishable from a genuine null and silent. The comparison against
    the cut is **strict**: an observation sitting on the threshold ties with a null maximum, and
    a tie is not an exceedance, so `>=` reports 0.06 under a heading that says 0.05.
*   **Both null benchmarks return nothing**, across three seeds each - while still containing
    thousands of local maxima, so the silence is the calibration rather than a blind extractor.
    Loosening alpha to 0.5 turns the same white-noise field into findings, which is the control
    that claim needs. This is TG2.4's floor arriving a slice early, on the raw field; the audit
    proper still has to run it under every registered *representation*.
*   **The suppression radius is measured, not chosen.** Each feature suppresses its neighbourhood
    at a multiple of its *own* measured scale. A fixed radius tuned at `scale_factor=1.0` reports
    eight features at `scale_factor=2.0` where three were planted, every extra one a noise maximum
    on a real blob's shoulder, above threshold, and indistinguishable in a receipt from a
    discovery. Self-scaling, the count is three across a six-fold range of widths.
*   **Two textbook estimators were measured and rejected.** The three-point sub-cell parabola is
    exact for a noiseless Gaussian and free, and its denominator is the second difference - 0.028
    against a noise amplitude of 0.05 on `planted_configuration`. It errs by up to 0.68 cells
    where a windowed centroid errs by 0.27. The curvature of the same fit, used as a scale,
    returned 2.1-2.9 cells for a planted width of 6.0. Neither is wrong in the textbook; both are
    wrong for features wider than a cell or two, and nothing here guarantees narrow features.
*   **The brightest sample is not the amplitude.** It is the largest of many noisy samples, biased
    high by 3% at `sigma = 3` and 12% at `sigma = 18`, and fed to the integral estimator that
    becomes a scale biased *low* by up to 9% - a bias that does **not** cancel in a ratio, because
    it is a function of the feature's own size, and a ratio is the only form in which a scale
    leaves its domain. The magnitude is a de-biased disc mean; the raw sample stays in provenance.
*   **No fabricated uncertainty.** Re-measuring each scale at two window sizes gives a spread of
    0.01-0.24 cells that does not cover the truth: at `sigma = 18` the two agree to 0.1 cells and
    both sit 1.4 low. That is repeatability, not accuracy, and putting it in
    `Quantity.uncertainty` would understate the error most in the records a reader would trust
    most. Left `None`, which TG2.1 defines as "not recorded".
*   **The declared axes change the arithmetic (E14).** On a periodic axis the neighbourhood, the
    window and the reported coordinate wrap together, and a seam-straddling feature is found once
    within 0.25 cells. The same array declared non-periodic is refused instead - R13 sized by the
    feature rather than by a fixed margin, because a peak two cells from a hard edge has half its
    integral outside the frame and comes back three cells out with a scale 24% low. A missing
    feature is a fact a receipt can carry; a confidently mismeasured one is not.
*   **Finding nothing is a result.** `ExtractionResult` is the object `FeatureSet`'s
    refuse-to-be-empty docstring pointed at: it knows what was searched, with what, at what
    threshold, and what was rejected on the way.

**Measured against answers recorded before the module existed.** `planted_configuration`: three
features, always three, position error under 1 cell, widths within 6% of the planted width from
`scale_factor` 0.5 to 3.0, pairwise separations within 1 cell of the planted 20 to 120, under
rotation, translation and rescaling. `advected_vortex_sequence`: exactly one feature in each of 24
frames under a single calibration, every position within 1 cell of the recorded trajectory, and
the measured scales reproducing the known 16-step doubling to within 5% without being told it
exists. That is TG2.3's acceptance criterion reached from the extraction side; what is left
between here and *1 track, 1 birth, 0 deaths* is association.

The benchmark's own `4E.feature_detection` check is deliberately **not** rewired to call this
extractor. A benchmark that validated the code under test with the code under test has stopped
being an independent answer.

**No new defect.** The off-by-one on the threshold comparison was in this slice's own code and was
caught by its own test before it was committed; the two rejected estimators are design decisions,
not defects in the tree. D59 remains open and remains TG2.3's.

**Evidence:** `src/tests/test_feature_extraction.py`, 39 tests (54 cases with parametrisation). Full suite 1429 passed, 1 skipped,
1 xfailed.

**TG2.3 Tracking - DONE.** `src/core/tracking.py` holds `ASSOCIATORS`, `SearchVolume`,
`MotionBounds`, `GateReport`, `Track` and `TrackingResult`. `4D.tracking` has moved from
`NOT_YET_RUNNABLE` to **PASS**, and the benchmark suite is 18 PASS / 0 FAIL / 1 pending.

**The framework/registry split is TG2.2's, unchanged.** A registered associator receives a
cost matrix and a mask of admissible pairs and returns pairs. It never sees a feature, a unit
or a time, so it cannot invent a gate, cannot bridge a gap, cannot decide what a birth is and
cannot attach anything to a record.

*   **The gate is derived, never chosen.** A tracker's one free number is how far a feature may
    move between frames, and it decides how many objects exist exactly as TG2.2's suppression
    radius decided how many features exist. The ceiling is a **coincidence radius**: the
    distance at which the expected number of *unrelated* features from the target frame inside
    the search ball equals the same `alpha` the extraction was calibrated at. One feature in a
    128x128 frame at `alpha = 0.05` gives 16.1 cells against a true step of 2.9 - and
    sixty-four features in the same frame tighten it by a factor of eight, because a gate that
    does not tighten when the field crowds manufactures tracks precisely where the data can
    least support them.
*   **Everything stricter is declared physics, not tuning.** `MotionBounds` carries a maximum
    speed, doubling rate and turn rate, all `None` by default - meaning *not declared* rather
    than *unbounded by assumption*, with the receipt reporting which gates were active. They
    are **rates**, multiplied by the actual elapsed time: the same 12-cell step is admissible
    over six frames and refused over one under a single declared bound of 3 cells per frame,
    where a gate expressed per *frame* would have answered identically in both cases and been
    wrong in one. A declared bound can only tighten the coincidence gate, never widen it.
*   **A gate on a quantity the extractor does not measure is refused.** This slice's acceptance
    names orientation gating, and TG2.2's only registered extractor declares
    `reports_orientation: False`. Passing every pair because the quantity is missing would put
    the gate in the receipt while refusing nothing, and the run would be indistinguishable from
    one whose physics was tested and satisfied.
*   **Orientation is where TG2.1's convention registry pays.** Under `axis_180`, 170 and 350
    degrees are the same ridge and link; under `direction_360` they are opposed and the same
    declared turn rate refuses. The numbers on the records are identical in both runs.
*   **The cost has no weights, because the gates are the weights.** Each active gate contributes
    the square of the fraction of itself the pair used. A hand-set trade-off between "how far it
    moved" and "how much it grew" would be one more free number deciding how many objects exist.
*   **Two associators, because they disagree measurably.** A second implementation is
    speculative generality unless the difference can be shown (TG1.4). On a constructed
    three-object frame where one object sits one cell from another's true partner,
    `greedy_nearest` takes that pair first - the cheapest single link in the frame - and the
    stranded track pays nine times as much: total 35 against the Hungarian assignment's 27,
    with two of three identities swapped. A third associator registered from the test module
    drives the tracker with no edit to `src`, and two rogue ones - one returning a pair the gate
    refused, one claiming an observation twice - are refused by the framework rather than
    trusted.

**Two defects in this slice's own tracker, found by running it against the benchmark rather
than by reading it.** The first version read its clock off the features it was given. A vortex
advected out of a 24-frame sequence after frame 5 produced a `FeatureSet` whose last frame *was*
frame 5, so the track ran to the end of its own evidence and looked complete against a recorded
answer that says one death - **a tracker that could never report a death at the end of a run**.
And an empty frame in the middle vanished entirely, silently bridging exactly the gap the
module's own docstring promises never to bridge. `times` - every frame that was searched - is now
required, `track_extractions()` takes it from the `ExtractionResult`s so a caller cannot supply a
clock shorter than the run, and `empty_frames` is on the receipt. Both are held shut by
regression tests. Neither is a new D-number: both were in code written this slice and caught
before it was committed.

**D59 is closed, by declaring the topology instead of picking a side.** `truth_advected_vortex`
took the vortex position modulo `n` while `build_advected_vortex` drew a Euclidean Gaussian that
clips. The fix makes `periodic` a declared parameter that the builder, the recorded answer and
both checks read (standard E14). At the registered parameters **every number the benchmark
produces is unchanged** - the modulo was a no-op there, which is exactly why it hid for so long.
`track_count`, `births` and `deaths` are now *derived* from an analytic `mass_inside_frame`
rather than asserted as the constants 1, 1 and 0, so a sequence that advects the vortex out
records the death that actually happens; and a second registered benchmark,
`advected_vortex_periodic_sequence`, declares a torus and draws one, making the seam-crossing
case answerable for the first time.

**One measurement went the other way, and was reverted on the evidence.** Making `4D.position`'s
*centroid estimator* read the topology - an ordinary mean on a plane, a circular one on a torus -
made the worst error eight times larger, 0.830 cells against 0.052. The diffuse noise floor
survives the check's weighting everywhere in the frame, and an ordinary mean drags toward the
centre of the array while a circular one lets it cancel. The circular estimator stays; what reads
the declaration is the *distance*, and what protects the estimator's one failure mode is the
measurable-frames filter.

**Measured against answers recorded before the tracker existed.** `advected_vortex_sequence`: one
track, one birth, no deaths over 24 frames; every position within 1 cell of the recorded
trajectory (worst 0.544); velocity 1.5 and 2.5 cells per frame recovered to 0.05; doubling time
16.31 steps against a recorded 16. The tracker was given the fields, the declared axes and an
alpha - never the velocity or the growth rate. `advected_vortex_periodic_sequence`: the same, at
0.735 cells and 16.38 steps, across both seams. Started at (100, 100) so the vortex advects out,
the extractor refuses the clipped blob by rule R13 in exactly the 18 frames the recorded answer
marks unmeasurable, and the tracker reports the one death that answer now derives.

**Evidence:** `src/tests/test_tracking.py`, 47 tests. Full suite 1477 passed, 1 skipped, 1
xfailed; benchmark suite 18 PASS, 0 FAIL, 1 NOT_YET_RUNNABLE.

**TG2.4 Representation-induced feature audit. DONE (`ed-dev`).** `src/core/representation.py`
holds `REPRESENTATIONS`, `NULL_STRATEGIES`, `FAMILY_CORRECTIONS`, `RepresentationPlane`,
`CorrectedLevel` and `AuditReport`. A new null benchmark `representation_null_field` gates
`4E.representation_audit`.

*   **A representation becomes a set of planes.** Named 2D arrays with declared axes, a
    declared decimation in parent cells per sample, and the filter's contaminated margin
    (R13). Six registered transforms plus the identity produce forty-eight planes of a
    128-cell frame; forty-five of them are testable. `raw` is the identity, so TG2.2's floor is an entry in
    the same registry rather than a separate argument - and propagating through it reproduces
    `calibrate`'s null exactly, which is what makes the audit a generalisation of the tree's
    existing calibration rather than a second opinion about it.
*   **The null is propagated through the lens, never rebuilt inside it.** An artefact of a
    representation is in every realisation the representation is pointed at, so it must be in
    the null. Rebuilding the null in the coefficient plane keeps that plane's spectrum and
    scatters its localisation, which turns a fixed artefact into a discovery: on the same
    scale-free field, propagating reports nothing and rebuilding reports **eight to
    thirty-four features**, almost all of them dual-tree subbands. Registered as two
    strategies so that difference is measured rather than asserted - and measured on the dual
    tree, because on the undecimated stationary transform the two agree and a claim tested
    only there would have been worthless.
*   **The audit had to pay for its own family before it could ask its own question.** R18,
    arriving two phases early. Forty-five planes read as one question is a family, and
    measured on its own ensemble the uncorrected procedure rejects on **83-88%** of
    structureless fields. `FAMILY_CORRECTIONS` reads the family-wise rate off the same
    ensemble the cuts come from, leave-one-out, so dependence between bands of one
    decomposition is measured; Bonferroni is registered beside it and, on six identical
    columns, prices one test wearing six hats at a sixth of the level.
*   **Keeping every threshold an order statistic has a price, and it is refused rather than
    fudged.** The strictest cut `n` surrogates can express still rejects about `P / n` of the
    time over `P` planes, so a family-wise 0.05 over forty-five planes needs **about nine
    hundred surrogates**. 499 is a refusal naming the family size, the achievable rate and
    the required ensemble - not a coarser answer.
*   **Three ways to earn a pass without looking, all refused.** A null nothing can violate
    (`phase_randomise` leaves an FFT magnitude plane bit-identical, so its threshold is its
    own observation for ever); a plane R13 leaves no interior in (struck from the family, so
    a test with no possible outcome cannot make the others stricter); and a registered
    transform nobody wrote a plane builder for (named in `uncovered_transforms`, and the
    audit is not clean).
*   **A frequency plane is refused by name, not silently skipped.** The axes of an FFT
    magnitude plane are wavenumbers, and declaring them `space` so a spatial peak-finder
    would take them is the error TG1.1 exists to prevent. `fft` and `dct` carry `axes=None`
    and a stated reason, their nulls are measured, and they are reported as **unauditable
    rather than clean**. Closing that gap needs an extractor with a spectral shape model and
    none is registered; recorded as a limit, not as a silence.
*   **A defect in this slice's own code, found by running it.** The dual-tree lowpass of a
    three-level decomposition is decimated by four, not eight, because it does not go through
    the quad-to-complex step that halves the highpass again. Declared as eight, a feature
    planted at row 70 of a 128-cell frame came back at row 137 - outside the frame it was
    found in, in the frame's own units. The decimation is now measured from the two shapes and
    any builder whose declared factor its own array does not have is refused.

**Acceptance.** `representation_null_field`: no feature in any of the **forty-five** testable
planes of **seven** registered representations - the six transforms plus the identity - with **215,884 cells** searched, family corrected from a nominal 0.05
to **0.001** per plane against a measured family-wise rate of **0.037**. The audit's power is
measured rather than assumed: the same lenses at the same corrected level find a six-sigma blob
nine times over, in the raw field, both approximations, the hybrid's low-pass half and the
coarse detail bands.

**Evidence:** `src/tests/test_representation.py`, 59 tests. Full suite 1536 passed, 1 skipped,
1 xfailed; benchmark suite 19 PASS, 0 FAIL, 1 NOT_YET_RUNNABLE.

**Exit criterion.** Features and tracks exist, are domain-typed rather than weather-typed, and
pass a benchmark written before them.

---

### Phase G3 — Constellations and the declared-family problem

**This phase is where R18 is either solved or the programme is bounded by it.** Do the
multiplicity work *first*; the mining code is comparatively easy and will be worthless without it.

**TG3.1 Family accounting. DONE (`ed-dev`).** `src/core/family.py` holds `SearchAxis`,
`SearchTerm`, `FAMILY_COMBINATORS`, `SearchSpecification`, `FamilyAccount`,
`max_affordable_family` and `FamilyUnaffordableError`, with
`GateProtocol.search_specification()` as the bridge from the frozen atmospheric protocol.

*   **A family is enumerated, not computed from a formula.** `GateProtocol.family_size` is
    correct and describes exactly one search shape; a second formula beside it is a second
    chance to be wrong by a factor nobody notices, and a bare family size cannot fail. A
    specification is terms over named, enumerated axes, and each registered combinator both
    counts (which prices a family too large to build) and enumerates (which a sweep is checked
    against). A test asserts the two agree for every entry, because implementations that drift
    apart would make a refusal and a receipt describe different searches undetectably.
*   **`ordered_pairs` is why the registry exists rather than a product rule.** The T4C.6 family
    is ordered *distinct* pairs of three scales crossed with six lags, and no Cartesian rule
    expresses either the ordering or the distinctness. TG3.3's `k`-feature constellations are a
    third shape; unordered triples register from the test module and drive a declaration.
*   **The refusal computes both R18 remedies rather than naming them.** `max_affordable_family`
    bisects the largest family the declared ensemble can still reject one member of - 2, 4, 8,
    15 and 54 members at 99, 199, 499, 999 and 4,999 surrogates under BY at 0.05 - and reports
    per axis the largest number of values that reaches it. The narrowing is checked by taking
    it and checked to be tight; where no single axis can reach the ceiling the refusal says so
    instead of sending the reader round a loop. The generate/confirm split is stated as what it
    is: not a cheaper family, a correction moved onto one frozen before the held-out partition
    was opened.
*   **A defect in this slice, found by running it.** The narrowing search reported "scale from 5
    to 1 values" as sufficient. `ordered_pairs` over one value has no members, so the family was
    empty and every ceiling was satisfied - the remedy was to empty the search. A zero-member
    term is now refused, and reaching a ceiling by emptying one does not count as achievable.
*   **The admissibility audit never moves the correction unit** (design option A). A lag below
    its support floor was never testable and is worth counting before acquisition, but the
    declared size and `correction_unit` both stay on the receipt: correcting only the members
    that survived a screen is correcting a family chosen after looking. An audit that empties
    the family, or one whose basis is not stated, is refused.
*   **The phase's boundary is now a measurement rather than an argument.** The constellation
    sweep R18 warns about - 5 scales x 4 orientations x 8 lags x 7 representations - is **4,480
    members needing 805,029 surrogates**, about 7.2e9 estimator evaluations at the sweep's own
    cost. That is this module's output, not a claim about it.

**Acceptance met.** Read from `campaigns/t4c6_nz_era5_temperature_850_v1.json` rather than from
a literal, the T4C.6 declaration prices at exactly **36 members and 3,005 surrogates**, agreeing
with the formula it generalises and with the `check_power` result `GateProtocol.validate` already
enforced - and its **label set is compared against a real sweep**, in order, because 36 agreeing
with 36 for two different reasons would pass a count check.

**Evidence:** `src/tests/test_family_accounting.py`, 30 tests. Full suite 1577 passed, 1
skipped, 1 xfailed; benchmark suite unchanged at 19 PASS, 0 FAIL, 1 NOT_YET_RUNNABLE.

**TG3.2 Generate/confirm split. DONE (`ed-dev`).** `src/core/preregistration.py` holds
`PartitionIdentity`, `Seal`, `HeldOutLedger`, `report_generation`,
`freeze_confirmatory_family`, `confirm_on_held_out` and the three refusals `SealBrokenError`,
`HeldOutAlreadyOpenedError` and `PartitionMismatchError`. This is R18's escape hatch and it
exists before any mining does.

*   **It is not a cheaper family, and the module says so.** The generate stage still
    enumerates and still tests; `report_generation` refuses to call the result anything but
    candidates, records its p-values as uncorrected, and refuses a candidate that is not a
    declared member of the generate specification — one nobody enumerated has no family and so
    no correction unit. What the split buys is a *confirmatory* family fixed and hashed before
    the held-out partition is opened, which is why the correction may legitimately be small.
*   **Two ways to cheat, caught two different ways.** A `Seal` carries a digest per sealed
    field and a digest over that table. An edited field is **named**; an editor who recomputes
    the digest table breaks the outer digest instead. Neither is the check that matters:
    `verify` states in its own return value that self-consistency says nothing about whether
    the seal was edited, and `verify_published` — against a digest recorded where the author
    cannot rewrite it — is the one with weight. A test proves the gap by rewriting a seal
    wholesale; it verifies perfectly against itself and fails only against the published
    digest.
*   **The ledger is keyed by the partition, not by the seal.** That is the substance of
    "once". A seal-keyed ledger would admit a second, entirely honest seal against the same
    held-out data: each confirmation individually defensible, the pair uncorrected. The
    refusal names both seals and computes the two remedies.
*   **A partition is identified without being read.** `from_series` takes geometry and the
    lineage the split wrote and touches no measure array — proved by a series whose values
    raise on access, because a seal written after reading the held-out data is not a
    preregistration whatever it hashes to.
*   **A refusal never spends the partition; a confirmation always does.** Seal, partition and
    label set are all checked before the ledger is written and nothing is checked after.
*   **A defect in this slice, found by running it.** `PartitionIdentity` never checked
    `channel_labels` against `n_channels`. A record claiming three channels while carrying two
    labels describes two geometries and the seal would bind both. Now refused at construction.

**Acceptance met.** Priced off `campaigns/t4c6_nz_era5_temperature_850_v1.json` rather than a
literal, the **T4C.6 family of 36 is refused at an ensemble of 199** (it needs 3,005) while a
**four-member frozen subset needs 166 and is not** — and the four are corrected at four, on a
partition that can then never be opened again. The distinction is measured rather than
asserted: the same four p-values corrected over the generated 36 reject nothing.

**Evidence:** `src/tests/test_preregistration.py`, 40 tests. Full suite 1621 passed, 1
skipped, 1 xfailed; benchmark suite unchanged at 19 PASS, 0 FAIL, 1 NOT_YET_RUNNABLE.

**TG3.3 Constellations as attributed graphs. DONE (`ed-dev`).** `src/core/constellation.py`
holds `Relation`, the `RELATIONS` registry of all eight named relations, `RelationValue`,
`RelationContext`, `measurable_relations`, `relation_axis`, `AttributedGraph`, `MatchReport`,
`constellation`, `constellations_from_set` and the two refusals `RelationUnmeasurableError`
and `GraphTooLargeError`.

*   **A relation is a quantity divided by something the features carry themselves.** TG2.1
    already permits `scale_ratio_to` across a domain boundary and refuses `separation_to`
    across one, and that contrast is the whole design: forty cells at a scale of six cells and
    twelve hundred metres at a scale of a hundred and eighty metres are the same relation,
    while forty and twelve hundred are not. Relations are computed *within* a domain, on
    coordinates in a space that exists; what crosses the boundary is the graph.
*   **Sorting the eight by what each needs is a finding, not a taxonomy.** `succession` needs
    nothing but a shared clock, because an ordering is dimensionless already. `co_occurrence`
    needs a temporal scale, because simultaneity is a *tolerance* and a tolerance in seconds is
    not a statement another domain can read. The other six need a scale, an extent or an
    orientation.
*   **Two of the eight cannot be measured at all today, and they register anyway.** The only
    extractor TG2.2 registers declares `reports_orientation: False`, so `direction` and
    `convergence` refuse by name - TG2.3's rule about gates carried into relations, since a
    relation treating an absent quantity as "no evidence against" would appear in a receipt,
    constrain nothing and be indistinguishable from one doing work. `convergence` carries a
    second refusal of the same kind: an undirected axis does not point, so it cannot converge.
*   **The registry reaches TG3.1 through `relation_axis`.** A family priced over eight
    relations when three are measurable declares five tests that could not have happened -
    which is not a conservative rounding but a receipt naming tests that never ran. Three
    measurable relations give an unordered-pair family of 3; all eight give 28.
*   **Matching is exhaustive or refused.** `k!` node correspondences up to eight nodes, and a
    refusal above that rather than a greedy assignment, because an approximate match that
    returns True is a claim. The report names both carried records, so a reader sees that two
    graphs matched while describing an amplitude field and a temperature field (R19) and
    whether they came from different representations (R8) - stated, never compared.
*   **A defect in this slice, found by running it.** A graph built with `strict=False` records
    a refused relation rather than propagating it, and `matches` compared the relations the
    graph *declared* rather than those it carried. Such a graph therefore did not match
    **itself**, and the stated reason blamed the geometry for what was a hole in the record. A
    missing relation is now a refusal naming the edge, and the caller narrows to what was
    measured.

**Acceptance met.** The `planted_configuration` triangle is built twice from the benchmark's
own geometry - once as a dimensionless amplitude on a grid in **cells**, once as kelvin on a
30-metre grid, with a different domain, dataset, variable, units and spacing - and the two
attributed graphs are **equal**, agreeing on 6.67 (40 cells over a 6-cell scale; 1,200 m over
180 m) rather than merely on which edges exist. The negative control is measured, not asserted:
a relation registered from the test module *without* the dimensionless division reports 40 on
one side and 1,200 on the other, and the same two graphs then fail to match.

**Evidence:** `src/tests/test_constellation.py`, 65 tests. Full suite 1686 passed, 1 skipped,
1 xfailed; benchmark suite unchanged at 19 PASS, 0 FAIL, 1 NOT_YET_RUNNABLE.

**TG3.4 Invariant matching. DONE (`ed-dev`)**
**Acceptance:** `4E.invariance` moves from `NOT_YET_RUNNABLE` to **PASS** on
`build_planted_configuration` under rotation, rescaling and translation. A matcher that memorises
pixel positions must fail this, and a test asserts that a deliberately position-memorising matcher
does.

`src/core/invariance.py` — `MATCHERS`, a registry whose entries are functions from features to
an `AttributedGraph`, so the comparison is always TG3.3's exhaustive `matches` and a matcher can
only fail by keying on the wrong quantity; `relative_geometry` (each separation over the
geometric mean of all of them) and `absolute_position` (the position-memorising control);
`scale_normalised`, public and deliberately unregistered; `best_deviation`, minimised over
correspondences; `null_deviations` and `scale_recovery_null`; `calibrate_match_tolerance`;
`match`, `recover_scale_ratio`, `ScaleRatio`; `Presentation`, `InvarianceTest`,
`ScaleRecoveryTest`, `InvarianceReport`; `measure_invariance` and `audit_declared_invariance`.
The `4E.invariance` check in `src/benchmarks/fields.py` is implemented against it.

Findings:

*   **A dimensionless relation is only as invariant as the thing it divided by.** TG3.3's
    `distance` divides a separation by the features' *estimated* spatial scale. On exact
    geometry it is perfectly scale-invariant; on the benchmark it moves 4.8% at
    `scale_factor=3` against a 3.3% noise floor, because the scale estimate drifts from about
    +1.4% at a 6-cell sigma to about -2.6% at 18 cells while the separation is recovered to
    better than 1%. The whole drift lands in the quotient.
*   **The invariant that survives divides one measurement by another of the same kind.** A
    separation over a separation cancels its units exactly and borrows nothing's accuracy. It
    needs three features: with two, one edge over its own mean is 1 for every configuration in
    the world, and a matcher that matches everything reports a discovery on every pair.
*   **A matcher whose declaration cannot be demonstrated does not get registered.**
    `scale_normalised` is not rescaling-invariant when measured directly, but over seven
    rescalings the evidence comes to `p = 0.016` and does not survive the family correction.
    True and unproven at once is not a state a declaration can hold.
*   **The extractor does not return its features in a stable order** - it follows which blob was
    brightest. Comparing replicate node 0 with node 0 measured that shuffle and put the position
    matcher's noise floor at 0.27, wide enough that a 37-degree rotation matched its own
    memorised pixel coordinates. Minimising over correspondences brought it to 0.026.
*   **A noise floor is an operating point, not a test.** Thresholding thirteen presentations at
    the largest of sixty-six noise samples rejects a genuinely invariant matcher about one time
    in five - R18's subject exactly. Invariance is decided by a one-sided rank test of the
    presentations against the whole null, with alpha divided across the tests actually run, and
    a test whose smallest possible p-value sits above its own alpha is `vacuous` and confers
    nothing (TG2.4's rule, carried into a rank test).
*   **A declared capability is measured, not trusted.** Overclaiming fails and so does
    understating, for the reason TG3.1's relation axis gives. The control fails by the general
    rule rather than by a special case, and a future matcher that overclaims fails the same way
    with no edit to the gate.
*   **The scale-aware reading, and the unread field it closes.** `scale_ratio_vs_reference` has
    sat in the benchmark's known answer since T3.5.17 with nothing reading it - the same species
    of defect as a relation that constrains nothing. It is now recovered from the separations
    (better than 1% across a sixfold range, where the scale estimate drifts by 6%), refused by
    name across a unit boundary, and structurally unable to run before the match: it takes the
    `MatchReport`, so R19 holds by construction rather than by comment.

**A defect found by running it.** The gate first judged the recovered scale ratio against the
*shape's* noise floor and failed a correct matcher on it. The shape's floor and the size's floor
are the noise of two different numbers; `scale_recovery_null` measures the right one.

**Acceptance met.** `4E.invariance` reports **PASS**, and it was the last pending gate in the
suite - the benchmark run is now **20 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE**, and
`test_benchmarks.py` asserts the pending count is zero rather than at least one. Two matchers
are audited over 13 presentations against 10 replicates: `relative_geometry` declares rotation,
translation and rescaling and is measured to have all three; `absolute_position` declares
nothing and is measured to have nothing, at `p = 0.0000` against every transform. Six scale
ratios are recovered and checked against their own noise floor. The gate has been confirmed to
pass at four root seeds.

**Evidence:** `src/tests/test_invariance.py`, 56 tests. Full suite 1742 passed, 1 skipped,
1 xfailed; benchmark suite 20 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE.

**TG3.5 Recurring motifs. DONE (`ed-dev`)**
**Exit criterion:** a motif can be mined, and its declared family was affordable and corrected.

`src/core/motif.py` — `Scene`, `Occurrence`, `MotifCandidate`, `MiningResult`,
`MotifEvidence`; `motif_search_specification` and `enumerate_occurrences`, checked against
each other member for member; `candidates_from`, `count_intransitive`,
`deduplicate_candidates`, `mine`; `observed_minimum_separation`, `surrogate_scene`,
`support_of`, `motif_p_value`; `report_motif_generation`, `choose_candidates`,
`confirmatory_specification`, `freeze_motifs`, `confirm_motifs`; and `always_matches`, the
constant-signature control. The `unordered_triples` and `unordered_quadruples` combinators
are registered into TG3.1's `FAMILY_COMBINATORS` from this module, which is the acceptance
TG3.1's own test wrote when it said the next search shape would be a registration rather
than an edit to `family.py`. Two new benchmarks in `src/benchmarks/fields.py`,
`planted_motif` and `motif_null`, carry the gates `4E.motif_recovery` and `4E.motif_null`.

Findings:

*   **The family is the number of configurations looked at, and no ensemble pays for it.**
    `s * C(n, k)`: the benchmark's six scenes of six features at `k = 3` is 120 members,
    which TG3.1 prices at **12,885 surrogates** before one of them could be rejected; eight
    scenes of twelve features is 1,760 members and **283,380**. The generate/confirm split
    is not an optimisation in this slice, it is the only affordable shape - and the gate
    refuses a pass if its own generate family ever becomes affordable in one stage, because
    then the benchmark would have stopped testing the thing it exists to test.
*   **Matching under a tolerance is not transitive.** A matches B and B matches C without A
    matching C, and single-linkage clustering promotes that chain into one motif with a
    support of three. A candidate is a star around one exemplar, and `intransitive_pairs`
    counts how often the difference would have mattered.
*   **Ranking by raw count prefers the promiscuous.** Support saturates at the number of
    scenes, so the ties at the cap decide the ranking, and they have to be broken by
    *fewer* occurrences: a shape matching three configurations per scene is looser than one
    matching exactly one, not a stronger finding.
*   **A confirmatory family topped up to the ceiling spends power on members nobody
    proposed.** Freezing four candidates where the mining pass favoured one moved the
    planted motif's corrected `q` from **0.005 to 0.042** - the difference between a clear
    result and a marginal one. The frozen set is the top support tier, deduplicated (one
    shape enters the ranking once per scene it occurs in, and six names for one hypothesis
    is a correction unit of six paid for one test), capped at TG3.1's ceiling.
*   **A surrogate has to be drawable by the process that produced the data.** Without a
    minimum separation the null fills with near-degenerate triangles the extractor could
    never have returned, supports the motif *less* often than a real arrangement does, and
    makes the observed support look more surprising than it is: over five seeds, dropping
    the constraint roughly **halves the p-value**, from 0.20-0.25 to 0.07-0.14. The
    constraint is read off the data, not declared, and an arrangement that cannot satisfy
    it is refused rather than quietly relaxed.
*   **Invariance is necessary and nowhere near sufficient.** `always_matches` has a constant
    signature. It is exactly invariant to rotation, translation and rescaling, and
    `test_motif.py` registers it into TG3.4's `MATCHERS` and shows the invariance audit
    calling it honest. It is also useless, and what separates it from a matcher that
    measures something is the null: it supports every motif in every surrogate scene, so its
    p-value is exactly 1. Public and unregistered - for the opposite reason to
    `scale_normalised`, whose declaration cannot be demonstrated where this one's is
    demonstrably true and worth nothing.

**Two defects found by running it.** A replicate must be the same configuration, not the
same frame: calibrating the tolerance from six-feature scenes narrowed to "the brightest
three" measured which blobs happened to be brightest and put the tolerance at **0.41 instead
of 0.0083** - fifty times too wide, and wide enough that every triangle matched every other.
It is TG3.4's node-ordering defect in a new place. And TG3.4's `recover_scale_ratio` raised
`KeyError` for a matcher that records no length; it now refuses by name, because a matcher
can recognise two configurations as the same shape without ever having measured how big
either of them was.

**Exit criterion met.** A motif is mined, frozen before the held-out scenes exist, and
confirmed on a partition it was not mined from at **q = 0.005** over a correction unit of 1,
with a held-out support of 6 scenes out of 6. The null benchmark runs the identical pass -
same family of 120, same tolerance, same ensemble, real candidates frozen from its own
training scenes - and confirms **nothing**, which is the load-bearing result. The benchmark
suite is now **22 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE**, confirmed at four root seeds.

**Evidence:** `src/tests/test_motif.py`, 45 tests. Full suite 1787 passed, 1 skipped,
1 xfailed; benchmark suite 22 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE.

**And TG3.1 did prove it, for this shape.** No achievable ensemble pays for a mining pass in
one stage: 120 members needs 12,885 surrogates and 1,760 needs 283,380. That is the
programme's real boundary, and it is written up as such - the family is not narrowed to make
it affordable, the correction is moved onto a confirmatory family frozen before the held-out
partition is opened, which is what TG3.2 exists for.

---

### Phase G4 — Planted ground truth

Before any real cross-domain claim. Extends `synthetic_generator/cascade.py`.

**TG4.1 Planted relationships at unknown scale and lag. DONE (`ed-dev`)**
**Exit criterion:** the engine is not told where the relationship is and must recover it.

`src/core/precedence.py` — `ScaleSeries`, `Record`, `PrecedenceCandidate`, `SweepResult`,
`PrecedenceEvidence`, `BasisCoupling`; `autocorrelation_time` and `record_memory`;
`admissible_lags`, `ordered_pairs_of`, `measure_basis_coupling`, `admissible_pairs` and
`precedence_search_specification`, checked against the sweep member for member;
`lagged_correlation`, `precedence_strength`, `sweep`; `circular_shift`, `permuted_series`,
`precedence_p_value`; `split_with_embargo`; `deduplicate_candidates`,
`indistinguishable_from_best`, `choose_candidates`, `report_precedence_generation`,
`confirmatory_specification`, `freeze_precedence`, `confirm_precedence`. Two new benchmarks in
`src/benchmarks/sequences.py`, `planted_precedence` and `precedence_null`, carry the gates
`4F.precedence_recovery` and `4F.precedence_null`.

**Where the generator went, and why not `cascade.py`.** This phase's preamble says it extends
`synthetic_generator/cascade.py`, and it does not. That module builds a cascade for the offline
UI and declares no truth; a benchmark needs a declared known answer, a gate and a null twin
built by the same code path, which is what `src/benchmarks/` is for. `build_precedence_sequence`
is therefore a benchmark generator with a `coupling` knob whose two ends are the two benchmarks,
and `cascade.py` is left alone.

**What was wrong with the version that already passed.** `coupled_cascade_sequence` has
recovered a planted lag since T4C.3, and its check reads `driver_level` and `driven_level`
straight out of the known answer before taking an `argmax` over lag. That is a search of one
band pair, run by someone who already knew the answer. This slice asks the same question
without being told either the pair or the lag.

Findings:

*   **Being told where to look is a family of one.** `L(L-1) * |lags|` members: four bands and
    twelve lags is 144, which TG3.1 prices at **15,985 surrogates** before one could be
    rejected; eight bands and twenty-four lags is 1,344 and 209,153. Ordered pairs, because a
    search that asks only *fine leads coarse* was told the direction.
*   **The lag is chosen by the data, so the null has to be over the choice.** Measured on
    records with nothing planted: the sweep's winner against a single-lag null is significant
    at 0.05 in **six runs of six**, and against the null of its own maximum in **one of six**.
    It is necessary and not sufficient - it prices the choice of lag, not of band pair - and
    what pays for the rest is the corrected held-out confirmation, where twenty null records
    produce **no confirmation at all**. The confirmation is affordable only because the lag was
    frozen on train: a frozen lag is not a selection, so its null is over a single statistic.
*   **A surrogate must keep the autocorrelation.** A circular shift preserves every value, the
    whole autocorrelation function and the marginal distribution, and destroys only the
    alignment. A shuffle destroys the memory too: measured over twelve null records, at
    `phi = 0` the two agree (median p 0.64 and 0.65), at `phi = 0.7` the shuffle's median p is
    0.34 against 0.55, and at `phi = 0.9` it rejects 3 times in 12 where the shift rejects none
    (median p 0.19 against 0.59). The error grows with exactly what the surrogate discarded.
*   **The decomposition relates bands to each other before the world does.** Found by running
    the null and watching it confirm a relationship at q = 0.0075 on a record with nothing in
    it. Measured on a control built by the same pipeline with **no temporal structure at all**,
    levels 1 and 2 correlate at **0.99 within a frame** whatever is planted, levels 3 and 4 at
    0.78-0.84 and levels 1 and 3 at 0.49-0.53, while the largest cross-band correlation at any
    admissible lag is 0.06-0.18. So the basis couples two bands when it relates them within a
    frame more strongly than it relates anything across one; the limit is read off the control,
    four of twelve ordered pairs survive, and the narrowing is admissible under R18 because it
    comes from a control record rather than from the record under test. The one lag the family
    may never contain - zero - is exactly the lag that identifies a band seen twice.
*   **Frames are not samples, and a lag costs data.** Both p-values travel with every
    candidate (R12); the benchmark's own record has 262 effective pairs out of 395 frames. The
    same arithmetic refuses a family whose largest lag leaves fewer than eight effective pairs,
    and refuses *every* lag on an `AR(1)` record at `phi = 0.98` - a near-unit-root record
    produces a refusal here, not a weak result.
*   **One relationship enters the family once per lag it survives at**, so the frozen set is
    one member per ordered pair, and then only those the training partition could not tell
    apart from the winner - within one standard error on Fisher's z scale. On the planted
    record that freezes one member; on a null record two to four, because when nothing stands
    out nothing is distinguished, and none of them confirm.

**A hypothesis that measurement refuted.** The embargo between train and held-out was expected
to matter, and on twenty records per setting whose bands are independent but slow it does not:
at `phi = 0.9` the train winner reaches the held-out partition at a median |r| of 0.150 without
an embargo and 0.152 with one; at `phi = 0.95`, 0.161 against 0.188. The held-out test is a
surrogate test drawn from the held-out record, so that record's memory is already in its null -
the protection comes from the surrogate, not from the gap. The refusal stays as a declared
constraint and is written up as one whose benefit has not been measurable, not as a repair.

**Exit criterion met.** The engine is told neither the band pair nor the lag. It declares a
family of 48 admissible members, sweeps a training partition, freezes what it found before the
held-out partition is opened, and confirms **one** relationship there - the fine band leading
the coarse band by exactly the five frames that were planted - at **q = 0.0050** over a
correction unit of 1. The null benchmark runs the identical pass, same builder at coupling
zero, same family, same ensemble, real candidates frozen from its own training partition, and
confirms **nothing**, which is the load-bearing result. Both gates hold at five root seeds. The
benchmark suite is now **24 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE**.

**Evidence:** `src/tests/test_precedence.py`, 62 test functions and 63 cases. Full suite 1850 passed, 1 skipped,
1 xfailed; benchmark suite 24 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE.

**TG4.2 The five refusals. DONE (`ed-dev`)** Per the proposal's own validation strategy, the
engine must: recover planted relationships without being told their scale or lag; reject
convincing but artificial correlations; distinguish independent observations from
autocorrelated repetitions; avoid representation-induced motifs; and report *no relationship*
when none exists.
**Acceptance:** five benchmarks, at least three of them nulls, all returning null. Per the
existing Definition of Done, **the null benchmarks returning null is the load-bearing result.**

`src/core/refusal.py` — `Refusal`, `REFUSALS` and `refusal_coverage`; `LagProfile`,
`lag_profile`, `simultaneous_strength`, `explained_by_leakage`, `refuse_leaked_members`;
`calendar_periods`, `remove_calendar`, `require_calendar_removed`; `naive_versus_effective`;
`carry_forward` and `refusal_report`; `EverythingRefusedError` and `CalendarNotRemovedError`.
Three new benchmarks in `src/benchmarks/sequences.py` — `shared_cycle_precedence`,
`slow_independent_precedence` and `leaked_band_precedence` — carrying the gates
`4F.refusal_calendar`, `4F.refusal_autocorrelation` and `4F.refusal_representation`, alongside
TG4.1's `4F.precedence_recovery` and `4F.precedence_null`.

**Acceptance met, and computed rather than asserted.** Five benchmarks, **four of them nulls**,
all five green at five root seeds. `refusal_coverage` checks the register against the live
benchmark registry - every named benchmark exists, declares the gate its refusal claims, and
agrees about whether it is a null - so a deleted or renamed benchmark fails a test rather than
leaving a sentence in a document. All five are **the same builder at five settings**, which is
what makes the four nulls controls rather than separate experiments.

**Two of the three new traps were holes, not demonstrations.** Each of the three gates runs its
own pass twice - guarded, which is the study, and unguarded, which is the trap measured on the
same record - and fails if its trap stops springing.

Findings:

*   **A single band, read through a redundant transform, manufactures four relationships.**
    One modulation, one spatial band, no second process anywhere. A stationary wavelet spreads
    that band's energy over every level, so every level's series is a smeared copy of one
    process and its memory keeps the copies correlated at a lag. TG4.1's basis control cannot
    see this: it excites both bands independently and asks which *pairs* the transform
    confuses. Run the TG4.1 pipeline unchanged and it confirms **all four admissible members
    at q = 0.0104 at every one of five root seeds**, each at the shortest admissible lag.
*   **The rule that refuses it is a ceiling, not a threshold.** Instantaneous leakage gives
    `r(k) = r(0) * rho(k)` with `rho <= 1`, so the simultaneous correlation is the supremum of
    everything a leak can produce and a lead is claimable only when it is **stronger than the
    same-frame relationship it might be a smeared copy of**. Nothing to tune. On the
    single-band record **0 of 48 members survive** at all five seeds - a *refusal of the
    family*, which is a different outcome from "nothing was significant" and is what that
    benchmark's known answer demands. On the planted record 20 to 44 of 48 survive and the true
    member still confirms at **q = 0.0050** at all five seeds: the ceiling costs a real finding
    nothing. The lag the family may never contain is now the reference every member is measured
    against.
*   **A shared cycle defeats the surrogate that was supposed to be enough.** Two independently
    modulated bands both carrying one deterministic cycle, the fine band's crest three frames
    early, nothing coupled. A circular shift preserves the periodicity but moves its phase, so
    most shifts misalign the crests and the observed alignment still looks surprising:
    unguarded, the pipeline **confirms three or four of the four members at every one of five
    seeds**, at q <= 0.0104, the strongest carrying a naive p between 1.7e-71 and 6.4e-54 and
    an ESS-corrected one between 3.6e-13 and 2.8e-10. Neither R12 nor the surrogate refuses it.
*   **The calendar is metadata, not a discovery.** Rule R11 lifted into this pipeline: the
    cycles a record is exposed to are computed from its **own cadence**, fitted on the training
    partition and subtracted from both on one shared clock, and `carry_forward` - the single
    door between a sweep and a seal - refuses a record that still contains them. With the
    removal in place the same five seeds confirm nothing, smallest corrected q 0.105. Detecting
    the period from the data instead was tried and **measured to be impossible at this record
    length**: the strictest of four out-of-sample variants separates a real cycle at 0.13-0.58
    from a red-noise fluke at up to 0.29, over ten seeds and five record types. The clock
    overlaps with nothing.
*   **What that leaves undefended is said out loud.** A periodic confound at a period the clock
    does not name is refused by nothing here, and the measurement above is the measurement of
    how badly it goes: the engine confirms it, repeatably.
*   **The third trap was already refused, and the benchmark measures by how much.** Two
    independent bands at `phi = 0.95` over 288 frames confirm nothing at all five seeds, while
    **22 to 48 of the 48 members** carry a naive p below 0.05 and as many as **20** carry an
    ESS-corrected one, on a record whose effective pairs are 8 to 19 per cent of its frames.
    `naive_versus_effective` reports both, because a corrected count with no naive count beside
    it does not show that the correction did anything.

**Where one harmonic came from.** The calendar reaches a band's energy series through a
nonlinearity, so more harmonics looked likely to help. On a longer record - 384 frames with a
24-frame cycle - the residual confirms a lag-1 relationship at one root seed in five, and does
so at one, two and three harmonics alike, the larger counts removing slightly *less* of the
spurious lead. That says the leak is a false positive of a 0.05 test rather than an unremoved
cycle. It is recorded rather than tuned away, and it is the one place this refusal has been
seen to fail.

**What this slice does not do.** It adds no new statistic. It does not separate a common driver
from a direct relationship - the conditional statistic that would is still absent, as TG4.1
also said. And the leakage ceiling is sufficient for refusal, not necessary: it refuses
everything a leak could have made, which would also refuse a genuine relationship that is
strongest within a frame. The honest reading of an emptied family is "this record cannot carry
this question", not "there is nothing here".

**Evidence:** `src/tests/test_refusal.py`, 72 tests. Full suite 1922 passed, 1 skipped,
1 xfailed; benchmark suite **27 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE**.

**TG4.3 Planted cross-domain relationships. DONE (`ed-dev`)** Two synthetic domains with
different units, semantics and cadences, one carrying information about the other's later
state.

`src/core/cross_domain.py` — `DomainChannel`, `DomainTimeSeries`, `CrossDomainSweep`;
`align_exact`, `cross_domain_metadata`, `cross_domain_pairs`, `physical_lags`,
`sweep_cross_domain` and `confirm_cross_domain`; `ExactClockRequiredError` and
`PhysicalLagRequiredError`. `src/benchmarks/cross_domain.py` adds the paired
`planted_cross_domain` and `cross_domain_null` benchmarks under the gates
`4F.cross_domain_recovery` and `4F.cross_domain_null`.

**The common representation does not erase either domain.** Each operand carries its native
meaning, units, dataset identity, licence, cadence, lag policy and provenance. Channel labels
are namespaced by domain before they enter TG4.1's machinery, and the full semantic record is
bound into the held-out partition identity. Changing `K` to another unit after the seal is a
partition change and is refused; a within-domain member presented to this boundary is refused
as laundering. The receipt reports a dimensionless structural statistic beside the original
Kelvin and megawatt operands and states the R19 boundary: temporal association, no comparison
of raw magnitude, and no causal mechanism established.

**The common clock is an intersection, not an invention.** The first synthetic domain is an
hourly thermal instrument with 1,080 observations; the second is a three-hourly operational
ledger with 360. `align_exact` retains the 360 timestamps both observed, records the 720
unused thermal observations, hashes the parent clock and performs **no interpolation**.
Offset clocks with no exact overlap, irregular native clocks and a non-regular exact
intersection are explanatory refusals. They are not passed to a default resampler whose
choice could move the result.

**A lag is physical before it is a frame count.** The family is declared at 3, 6, 9 and 12
hours. Each domain's R21 floor is converted from its own native cadence first — two hourly
sensor-response frames versus one three-hour reporting frame — and the larger physical floor
is applied before the durations are expressed on the shared clock. A duration below either
floor, between common-clock frames or declared twice refuses the family rather than silently
narrowing it. A domain declaring aggregate values must name the aggregation window too, and
that window raises the floor so two overlapping reports cannot be read as precedence.

**Nothing is told where to look.** Two channels per domain give eight ordered cross-boundary
directions, including both directions for every pair and no within-domain pair; crossed with
four physical lags, the training sweep declares and measures **32 members**. It freezes what
the training partition cannot distinguish before the held-out partition is opened and uses
TG4.1's circular-shift null and TG3.2's one-use ledger unchanged.

**Acceptance met, with the null carrying the weight.** On five root seeds the planted record
confirms exactly
`synthetic_thermal_observatory::thermal_gradient > synthetic_demand_ledger::demand_pressure`
at the planted six-hour lag, corrected q = 0.0050 every time. `cross_domain_null` is the same
builder at coupling zero — same clocks, units, semantics, marginal processes, family and
ensemble. Its own training partition freezes two to four candidates, so the pass is not
vacuous; held-out confirmation returns **nothing** at all five seeds, with its smallest
corrected q from 0.167 to 1.000.

**The refusal boundary remains visible.** This slice supports regular native clocks whose
exact intersection is regular, policies whose physical floor is fixed by the declaration,
and domains declaring no natural cycle. It does not interpolate irregular observations,
remove two different native calendars, distinguish a common driver from a direct relation or
license causal language. Those are named limits rather than behaviours hidden in alignment.

**Evidence:** `src/tests/test_cross_domain.py`, 47 test functions and 50 cases. Full suite
1972 passed, 1 skipped,
1 xfailed; benchmark suite **29 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE**.

---

### Phase G5 — Frozen cross-domain motif transfer

The flagship experiment, and correctly last among the scientific phases.

**TG5.1 Motif freezing. DONE (`ed-dev`).** `src/core/motif_freeze.py` turns one exact
`MotifCandidate` selected from a `MiningResult` into schema `frozen-motif/v1`. The durable
definition contains the dimensionless graph attributes and typed edges, registered matcher,
configuration size, relation set, refusals and measured tolerance under its own SHA-256. The
outer motif hash additionally binds the source family identity and size, exemplar label and
selection counts, training `PartitionIdentity`, study and freeze time, complete originating
`DomainDeclaration`, and every node's carried domain/dataset/variable/representation/unit
record. The structural and semantic halves are stored separately, so recording the origin does
not make it part of the cross-domain comparison.

Nested records are immutable in memory. Canonical JSON publication uses exclusive creation and
refuses every overwrite; reload requires exact schema fields, canonical bytes, a reconstructible
`AttributedGraph`, and matching partition, definition and outer hashes. A separately published
digest catches a wholesale self-consistent rewrite that local hashes alone cannot. Freezing
refuses held-out source data, a source label attached to a different exemplar graph, mixed or
misnamed origins, an unpriced node count and a timestamp without an explicit UTC offset.

**The chronological boundary was assigned to TG5.2.** A hash proves content identity, not when
the content existed. `verify_published` supplies the comparison TG5.2 now places in a ledger
before opening domain B; TG5.1 alone does not claim that a target was blind merely because the
artifact is immutable. No real second domain was opened and no transfer result exists in this
TG5.1 slice.

**Evidence:** `src/tests/test_motif_freezing.py`, 14 tests. Targeted motif/preregistration/
cross-domain integration: 153 passed. Full suite after documentation: **1986 passed, 1 skipped,
1 xfailed**; benchmark suite remains **29 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE**.

**TG5.2 Blind transfer. DONE (`ed-dev`).** `src/core/motif_transfer.py` admits target values
only through an opener callback. Before calling it, `TransferLedger` verifies the exact
`FrozenMotif` against its separately published digest and durably commits that digest,
definition identity, target partition and domain declarations, and strict timezone-bearing
`frozen_at < bound_at < opened_at` chronology. The target is spent at commit even if loading or
validation fails, so an unpromising look cannot be followed by a redefined motif. A second open
of the same target identity is refused across processes.

Search then enumerates the complete pre-bound target scene count. Configuration size, registered
cross-domain matcher, relation set and measured tolerance are read only from the frozen artifact
and are absent from the API's parameters. Match reports keep both domains' carried semantics
visible while excluding them from structural comparison. A content-addressed receipt records
every examined/matching configuration and treats zero matches as complete. Sixteen tests (18
cases) cover chronology, persistence, failure spending, no redefinition surface, exhaustive
search, target-subset/domain laundering, null transfer and ledger/receipt integrity. Targeted G5/
motif/preregistration/cross-domain integration: **171 passed**.

**Claim boundary.** This proves ordering for access through this API, not the absence of earlier
human or out-of-process archive access; that requires external access control. Synthetic domains
exercise the contract, but no real target archive was opened. TG5.2 is descriptive motif search;
the corrected relationship-transfer result is supplied separately by TG5.3.

**TG5.3 Relationship transfer. DONE (`ed-dev`).** `src/core/motif_relationship.py` freezes one
content-addressed `RelationshipPlan`: the published motif, complete target domain, ordered
non-overlapping train/test identities, outcome x positive-lag family, one-sided effect,
circular-shift null, ensemble, alpha, correction and seed. R21 must license precedence, every
lag must meet the domain floor and G3 affordability must pass before either target split can be
opened.

The execution API has no analysis override surface. A durable ledger verifies the separately
published motif and plan digests and commits both partitions before either callback; each
partition is globally one-use even under a new pairing, and failure still spends both. Every
frozen member is tested and corrected over the complete family independently on train and test.
Only a positive label rejecting in both is a transfer `PASS`; an adequately powered empty
intersection is a complete `FAIL`, so **a null result here remains a genuine,
publishable-shaped finding**. Non-estimable members remain inside correction at p=1 rather than
quietly shrinking the family.

**Evidence:** `src/tests/test_motif_relationship.py`, 16 tests. Focused TG5.3: **16 passed**;
G3/G5 integration: **228 passed**. Synthetic planted and null fixtures exercise the instrument;
no real target archive has been opened. This licenses association/precedence only, not mechanism,
causality or predictive utility, and external access control still owns the absence of earlier
out-of-process inspection.

---

### Phase G6 — The claim ladder and `EvidenceBundle`

Deterministic and mandatory. Must exist before G7.

**TG6.1 `EvidenceBundle`. DONE (`ed-dev`).** `src/core/evidence.py` holds one `Hypothesis` —
identifier, statement, prediction, timezone-bearing registration and provenance — plus an ordered
chain of `EvidenceEntry` records under ten first-class fields: `observations`, `effect_sizes`,
`uncertainty`, `null_results`, `replication_results`, `holdout_performance`, `provenance`,
`confounders`, **`contradictory_evidence`** and **`failure_states`**. The last two are ordinary
fields with ordinary accessors carried on the same chain and the same digest as favourable
evidence; no later append can remove or dilute them.

`append()` returns the next immutable snapshot and leaves the receiver unchanged byte for byte.
Each entry hashes its own body including the previous entry's digest, anchored on a digest binding
schema, study id, creation time and hypothesis, so dropping, reordering, relinking, softening or
swapping the hypothesis under an existing chain is detected. Sequence numbers are gap-free,
append chronology non-decreasing, payloads deep-frozen and required to be non-empty finite JSON.
Every entry must name one of the ten fields: there is no `commentary`, `notes` or `interpretation`
route, so **free text cannot enter the structure the TG6.2 gates read** (R22). Canonical
exclusive publication refuses to overwrite; reload re-verifies every entry digest, chain link,
sequence, category placement, exact canonical bytes and the separately published bundle digest,
and refuses an entry relocated into a different first-class field on disk.

**Evidence:** `src/tests/test_evidence_bundle.py`, 24 tests.

**Claim boundary.** This is a tamper-evident container and a routing discipline, not a judgement:
it does not decide whether the evidence inside supports anything — TG6.2's ladder and TG6.3's five
outputs own that. Hashes detect edits to a published bundle; they do not authenticate the author,
and they cannot show that relevant evidence was gathered and simply never appended.

**TG6.2 The ladder. DONE (`ed-dev`).** `src/core/claim_ladder.py` assigns
`observation → association → robust association → candidate precursor → demonstrated predictive
utility` through ten declarative gates. `assess_claim_ladder(bundle)` is a **pure function of the
bundle**: no second argument, no clock, no filesystem, no environment, no randomness, so the
verdict is recomputable by anyone holding the published snapshot.

`association` requires passing `observations`, `effect_sizes` and `uncertainty` entries — a point
estimate without a quantified interval is not an association. `robust_association` adds passing
`replication_results` and `confounders`. `candidate_precursor` adds passing `provenance` and a
passing `provenance` payload recording `temporal_precedence` exactly `true`; a truthy stand-in
does not count, because precedence must be asserted rather than inferred.
`demonstrated_predictive_utility` adds a passing `holdout_performance`. Only `PASS` advances a
gate; `INCONCLUSIVE` and `NOT_APPLICABLE` are recorded honestly and buy no ground.

**Nothing outvotes a failure.** Any `FAIL` or `INVALID` entry anywhere, and any
`contradictory_evidence` or `failure_states` entry recorded as `PASS` — the bundle asserting the
contradiction stands — caps the bundle at `observation`. TG6.1 immutability means a failure cannot
be appended away: the only route higher is a bundle that never carried it. `unblocked_rung` reports
what the remaining evidence would have reached, as a diagnostic; `rung` is what may be claimed.
Causal claims are outside the ladder entirely: `permits()` **refuses** `causal`, `causation`,
`mechanism`, `efficacy`, `cure` and their neighbours naming R7, rather than returning a `False`
that invites a later "not yet" reading (R7). The gates read only category, status and the one
reserved precedence key, so prose in labels, summaries and unread payload keys is inert (R22).

**Evidence:** `src/tests/test_claim_ladder.py`, 30 tests. Purity is property-tested by
determinism to the digest, independence from append order and from all unread text, an identical
rung after a round trip through disk, and agreement with an independent restatement of the rule
over a randomised sweep in which all five rungs occur. The blocking invariant is tested
exhaustively over field × status and over randomised whole bundles in which both the blocked and
clear branches occur; four deliberate mutations of the gate table were each caught.

**Claim boundary.** The ladder grades the evidence that was appended and cannot know what was
never gathered. It is a floor on rigour, not a certificate: the top rung means a holdout result
was recorded and passed, not that the design was sound, that the holdout was honestly held out, or
that the effect transfers. No rung licenses a causal reading.

**TG6.3 The five outputs. DONE (`ed-dev`).** `src/core/five_outputs.py` states, for any bundle:
what can be claimed; what cannot; what evidence contradicts it; what alternative explanations
remain; and which single observation would most efficiently distinguish between them.
`summarise_evidence(bundle)` is a **pure function of the bundle**, like the ladder beneath it, and
`summary_sha256` binds the report to the exact revision it came from.

What can be claimed is the reached rung and every rung below it, with a fixed entitlement sentence
in the programme's wording saying what that rung licenses and what it does not. What cannot be
claimed is the exact complement, each rung naming the gates that stand between; because the ladder
is climbed in order, a rung whose own gates all pass may still be unreachable, and the nearest
rung that actually blocks it is named rather than left blank.

**The evidence against is wider than the ladder's blocking set.** The ladder asks what may be
claimed; this asks what argues against the hypothesis. So it carries every `FAIL` and `INVALID`
entry, every `contradictory_evidence` and `failure_states` entry that is not `NOT_APPLICABLE`, and
every passing `null_results` entry — a recorded null is contrary evidence even where it caps
nothing — each flagged with whether it caps the bundle, so what merely argues against a claim is
visibly distinct from what forbids it.

Alternatives have two origins. Eight **structural** alternatives are mapped one-to-one and totally
onto the eight climbing gates — `chance`, `sample_specific`, `confounding`,
`reverse_or_simultaneous_order`, `in_sample_optimism`, `unauditable_origin`, `nothing_measured`,
`no_estimated_effect` — each open exactly while its gate is unsatisfied. Everything else is
**recorded**: an unaddressed confounder, a standing contradiction, a passing null. The module
never invents a domain alternative. The next observation follows a stated precedence — resolve the
earliest blocking entry, else close the first unmet gate in ladder order, else distinguish the
earliest recorded alternative, else nominate nothing and say that nominating nothing is not the
same as there being nothing.

**Evidence:** `src/tests/test_five_outputs.py`, 38 tests. Two randomised sweeps carry the
invariants: the first checks the claimable/not-claimable partition, the contrary set and the
structural-alternative correspondence over 600 bundles reaching all five rungs; the second checks
the next-observation precedence over 600 more in which every branch, including the empty one,
occurs. Five deliberate mutations of the output rules were each caught. The sweep found a real
defect during development — an unreachable rung whose own gates all passed was reported with an
empty explanation — which is fixed and separately tested.

**Claim boundary.** The fifth output is a precedence rule, not an experiment design: no expected
information gain is computed, because the bundle carries no likelihoods, and "most efficient"
means "the cheapest thing standing in the way". The fourth can only name alternatives its gates
correspond to or that someone recorded, so a domain rival nobody wrote down is invisible; its
absence from the report is not evidence of its absence in fact. The same holds of the third, and
the rendered text says "none recorded; that is not the same as none existing" rather than leaving
silence to be read as reassurance.

---

### Phase G7 — Adversarial review layer

Sits **above** G6's gates and may not touch them (R22). Non-deterministic and recorded as such
(R23).

**TG7.1 The recorded-call boundary. DONE (`ed-dev`).** `src/core/recorded_call.py` is the only
door through which the review layer may speak, and it is built so the layer cannot reach the G6
gates through it. Every call captures the verbatim request, the verbatim response bytes, the
exact model id, the effort setting, the API request id and both timestamps, chained by digest and
labelled `recorded-not-reproducible` in its own body (R23). Sampling parameters are **refused
rather than recorded** at any depth — `temperature`, `top_p`, `top_k`, `seed` — because pinning
one would imply a reproducibility this layer does not have. A `ResponseSchema` is sent as
`output_config.format` with `additionalProperties` closed, and the response must parse as JSON
matching it exactly; free text where a schema was declared is refused rather than parsed
hopefully, and the record keeps both the bytes and the parse and requires them to agree, so the
check survives a round trip through disk rather than holding only at record time.

Commentary lives in a `ReviewRecord` that binds the bundle's digest and revision from outside, is
published in its own file beside the bundle, and is never appended to the evidence chain.
`record_call` hands back the same bundle object it was given; `ReviewedBundle` computes every
claim-bearing property from that bundle alone; `strip_review` deletes the whole review and
returns it unchanged.

**Acceptance: a corpus of bundles standing on all five rungs — plus a blocked one and a
contradicted one — is reviewed by all eight roles with deliberately assertive commentary
demanding promotion, and every claim level, every claimable set and every summary digest is
identical after deleting every LLM output.** `verify_claim_independence` makes that executable
outside the test: it compares the claim digest with the review present and deleted, re-derives it
from the bundle's own canonical bytes, and refuses if any recorded phrase of at least
`SMUGGLING_FLOOR` characters appears inside them — the one way R22 could actually be broken is a
person copying an answer into a summary or payload, and that is caught rather than assumed
absent.

**Evidence:** `src/tests/test_recorded_call.py`, 50 tests (55 cases). Two randomised sweeps: one
over 300 reviewed bundles reaching all five rungs, asserting that no review of any shape moves
any claim level; one over 200 records of one to five calls, asserting the recorded chain stays
self-checking and reloadable. Five deliberate mutations of the boundary were each caught — the
fifth only after a test was added to isolate it, because the chain check had been masking the
bundle-binding check.

**Claim boundary.** This slice records calls and fences them off; it does not conduct a review.
Nothing here judges whether a challenge was any good, and a well-formed false answer is recorded
as faithfully as a true one. The smuggling check finds recorded wording reproduced in a bundle;
it cannot detect a person who reads commentary, is persuaded, and records a genuine-looking
measurement in their own words. No structural check can, and the defence against that is the
provenance the gates already require.

**TG7.2 Adversarial round-robin. DONE (`ed-dev`).** `src/core/round_robin.py` runs the eight
seats in their fixed order over one frozen bundle — candidate synthesis, statistical challenger,
confounder and alternative-explanation challenger, domain-plausibility challenger, provenance and
methodology challenger, one response per dissent raised, independent reassessment, bounded final
synthesis — with every turn taken through the TG7.1 boundary and recorded verbatim beside the
bundle.

The order is replayed rather than trusted: for every prefix of the recorded chain, `RoundRobin`
recomputes the turn the protocol would have demanded and refuses a record whose role, target,
response schema, model or effort is not the one that was due. A challenge cannot be synthesised
over before it has been answered, and a ninth turn cannot be appended to a finished exchange. A
malformed turn is recorded first and refused second — `RecordedTurnRefused` carries the reviewed
bundle including the offending call, because it was made and cannot be regenerated (R23).

**Not majority voting.** Nothing counts verdicts. A dissent is retired only by the candidate
conceding it, or by a rebuttal the independent reassessment declines to reopen; the candidate does
not mark its own homework, and until the reassessment has spoken a rebuttal is provisional. The
reassessment may reopen a dissent but cannot originate one, because nothing downstream would
answer it. **Unresolved dissent is retained in the published record, never reconciled into
consensus:** the final synthesis must name exactly the unresolved dissents and its
`dissent_remains` flag must match, so a synthesis that drops one, invents one, or reports calm
while one stands is refused. The roadmap's *"retained in the bundle"* and R22's *"nothing in the
bundle"* are both honoured by retaining it in the review record published beside the bundle.

**Acceptance: a complete eight-role exchange over a corpus standing on all five rungs — plus a
blocked bundle and a contradicted one — with every seat arguing for promotion by name, moves no
claim level;** and three agreeing challengers do not retire the fourth's objection, which survives
into the outcome with its own words and the alternatives it could not exclude. Changing what the
three agreeing challengers said leaves the retained dissent byte-identical, which is what a hidden
count would have broken. `close_round_robin` re-runs `verify_claim_independence` on the way out.

`ReviewPanel` seats a model and effort per role, digests them, and pins them into the outcome.
Reviewer overlap is **recorded, not refused** — a single-provider panel stays runnable, and
`render()` says plainly when the reassessment was made by the model that wrote the synthesis. The
module names no vendor and opens no socket: `model_id` is any string, `effort` is an abstract knob
a later adapter translates, and the closed response schema is enforced locally on the parse rather
than trusted to the provider.

**Evidence:** `src/tests/test_round_robin.py`, 49 tests (50 cases). A randomised sweep over 120
exchanges with random dissent, response and reopening patterns checks the retained dissent against
an independently written rule and asserts its own coverage of rungs and dissent counts; a second
sweep over 80 exchanges with randomly seated panels re-replays each recorded chain against the
protocol. Six deliberate mutations of the protocol were each caught — the sixth only after a
missing guard was added: removing the ordering check made the exchange non-terminating rather
than wrong, so the mutation hung the suite instead of failing it. Each challenge is now refused a
second answer, which bounds the exchange whatever else is removed.

**Claim boundary.** This slice conducts the exchange; it does not judge it. Nothing measures
whether a challenge was any good, whether a concession was warranted, or whether a rebuttal was
honest, and a lazy panel that raises no dissent produces a clean outcome that means nothing. The
protocol checks independence at the level of the model id and nothing deeper: two sizes of one
family are reported as independent although they share training data and failure modes, so a
tiered single-provider panel buys cost control and a capability gradient rather than the
independence the word suggests. Seating genuinely unrelated reviewers is a configuration decision
the panel records and does not make. Every test uses a recorded transport, so nothing here
shows that a real client behaves as the protocol expects. Retention is not resolution: an outcome
carrying four unresolved dissents is an honest record of an argument nobody won.

**TG7.3 Cost control. DONE (`ed-dev`).** `src/core/review_cost.py` adds a provider-neutral
`ReviewCostPolicy` / `ReviewCostReceipt` audit and the first concrete transport,
`GeminiBatchTransport`, for the GA model id `gemini-3.5-flash`. Every role is pinned before the
review: the four challenger seats use low thinking, the response uses medium, and candidate
synthesis, independent reassessment and final synthesis use high. One provider/model across all
seats is recorded honestly as overlap by TG7.2; the effort gradient is cost control, not
independence.

Every non-interactive turn is translated into one asynchronous inlined Batch GenerateContent
request with the TG7.1 response schema, and the returned `batches/{id}` is its API request
identity. The API key exists only in the request header and cannot enter a record or exception.
An optional explicit `cachedContents/{id}` resource may carry the shared corpus, while implicit
caching remains possible; neither is credited merely because it was configured.
`audit_review_cost` requires `service_mode: batch`, reconciles normalized token counts against
the preserved raw Gemini `usageMetadata`, and refuses the whole review unless measured cached
tokens are non-zero. The immutable receipt binds per-call unique batch identities and aggregate
input/output/cached/total tokens to the exact review-record and policy digests, with atomic
no-overwrite persistence. Monetary cost is reconstructed from a dated price table rather than
frozen into a supposedly timeless receipt; the policy pins the documentation snapshot and the
documented 50% Batch factor.

**Evidence:** `src/tests/test_review_cost.py`, 15 tests (19 cases). The offline HTTP fixture
exercises submission, polling, structured response parsing, raw usage preservation and cache-hit
accounting; an eight-call review proves the model/effort/batch routes and a 90% measured cache-hit
fraction. Cache enabled with zero hit, standard service, effort drift, normalized/raw usage
disagreement, batch-id splicing, duplicate batch identities, receipt tampering and API failures
are refused. The first real call found D60: the completed-operation nesting, structured-output
dialect and thinking-token accounting differed from the offline fixture. All three are fixed and
the live-derived operation shape is now a regression test. A subsequent real batch returned the
exact declared `{status, note}` schema with 52 input and 124 billed output tokens.

**Claim boundary.** The official Gemini model, caching, Batch, thinking and pricing pages were
reviewed 2026-08-26. Authentication, asynchronous Batch polling, structured output and token
usage are live-accepted against Gemini 3.5 Flash. The 52-token smoke is below the 4,096-token
cache floor and correctly reported zero cached tokens, so a real cache hit, full review and live
cost receipt remain **NOT RUN**. A cost receipt says what route and usage were recorded; it says
nothing about review quality and cannot move a G6 claim.

**TG7.4 Translation, bounded. DONE (`ed-dev`).** `src/core/translation.py` renders a finding in
domain language for a reader. This is the first point in the programme where text is produced for
a human to act on, and therefore the point where four rules break at once if nothing stops them:
domain prose reaches naturally for the semantic comparison R19 forbids; a bare confidence figure
is the way R9 says this platform is most likely to mislead its own author; causal verbs enter
through sentences rather than through claim kinds (R7); and "candidate precursor" becomes "early
warning signal" — a promotion carried out entirely in wording, with no gate touched (R22).

**Translation is a projection, not a generation.** There is no model call. Domain wording arrives
as declared, content-hashed data screened when it is registered, and the renderer emits only from
closed template sets bound to structural facts.

**R9 is given a structure it did not have.** The six figures R9 names had **no structured home
anywhere in `src/`** before this slice: the ladder asks only whether an `effect_sizes` entry
passes and never reads what is inside one, so support, confidence, base rate, lift, interval and
surrogate-corrected lift lived unvalidated in a payload mapping. `AssociationFigures` keeps all
six together, refuses a partial set by name, and checks lift against `confidence / base_rate`
rather than trusting it. Its `render` is the only method that can format a percentage, so the
roadmap's own cautionary "82% of the time" is not banned but made honest — the base rate that
decides whether 82% is a finding or noise is in the same string.

**R19 holds by construction, not by inspection.** Two disjoint template sets, selected by
`semantic_key` equality: the within-domain set may reference units, magnitude and the variable;
the cross-domain set may reference only `structural_signature()`. A cross-domain magnitude
sentence cannot be constructed, and widening the cross-domain field set is a visible edit to a
named constant. **R22 holds by signature:** `translate` takes a `FiveOutputs`, never a
`ReviewedBundle`, so review commentary has no parameter through which to arrive; what a caller
passes as commentary is quarantined, excluded from the claim text, and rendered under a heading
saying it moved nothing.

A glossary registers through the same `Registry` orientation conventions use, so a domain supplies
one **without editing `src/`** — the TG8.1 condition met early. Registration refuses a partial map,
causal vocabulary, any digit, any comparative asserting a relation of size, and any rung phrase
borrowing wording reserved to a higher rung. Each entitlement is welded into the same string as
the claim it bounds, so a UI cannot show "this is a candidate precursor" and drop "predictive
utility is not shown".

**Acceptance: a corpus standing on all five rungs, plus a blocked bundle and a contradicted one,
translated into an atmospheric and a financial vocabulary, yields documents that read completely
differently and assert an identical set of structural facts, with every claim digest unchanged;**
and a hostile glossary attempting six promotions in wording alone is refused at registration, by
name, for each one.

**Evidence:** `src/tests/test_translation.py`, 47 tests (60 cases). Three randomised sweeps: 600
documents over both vocabularies checked against an independently written restatement of the fact
set; 400 more checking numeral containment, bare-confidence and causal-vocabulary refusal with and
without figures attached; and 300 feature pairs, half sharing a `semantic_key` and half not,
asserting no unit-bearing field ever leaves a cross-domain rendering. Nine deliberate mutations
were applied to the module itself and each was caught — but only after the first attempt at two of
them proved worthless: one pattern never matched, so the "mutation" ran against unmutated source,
and the other showed the two halves of the bare-confidence guard are fully redundant, differing
only in diagnostics. Both are recorded in the module rather than left as false assurance.

**Claim boundary.** A translation is faithful to the **record**, not to the world. A glossary
mapping a structural term to a misleading-but-non-causal domain word is accepted, because no
structural check knows what "anomaly" means to an oceanographer; the defence is that the glossary
is declared, hashed and reviewable, not that it is correct. Refusing causal *vocabulary* is not
refusing causal *implication*, and a reader who reads "precursor" as "cause" is caught by nothing
here. The R19 guard stops the *system* emitting a cross-domain comparison; it cannot stop a reader
setting two within-domain renderings side by side and drawing one themselves, and layout is out of
scope. Nothing judges whether a finding was worth translating, or whether the domain words chosen
are the ones a practitioner would use.

**Deliberately not built: a generative translation seat.** Rendering domain prose through the
TG7.1 boundary as a ninth role was considered and rejected for this slice. A generated sentence
would make every guard above a post-hoc text check instead of a structural impossibility, and it
would put a non-deterministic step between the gates and the reader. The deterministic renderer is
built first precisely so that a generative seat, if one is ever added, has something it must pass:
it would have to satisfy the same four guards, and its output would be commentary under R23 rather
than claim text. Authoring a glossary is correspondingly constrained — `RESERVED_WORDING` reserves
"signal" to `candidate_precursor`, so common domain words are refused at lower rungs. That is the
intended cost, not an oversight.

---

### Where this line stands, and what comes next

Recorded so a reader arriving cold does not have to infer it from which entries say DONE.

**Complete:** G0 through G7. Every slice from TG0.1 to TG7.4 is marked DONE on `ed-dev`, with
suite, benchmark and documentation evidence in `VERIFICATION.md`.

**Outstanding, in the order intended:**

1. **TG7.3's live tail.** A real cache hit, a live eight-role review and a live cost receipt are
   **NOT RUN**. The accepted smoke was 52 tokens, below Gemini 3.5 Flash's 4,096-token implicit
   cache floor, so it correctly reported zero cached tokens. Closing this needs a prompt above
   that floor and a full eight-call review against the live transport. Sequenced *after* TG7.4
   deliberately: TG7.4 is entirely offline and does not depend on it, and spending real tokens on
   a full review before the translation layer existed would have bought nothing.
2. **Phase G8** — TG8.1 (the onboarding contract) and TG8.4 (the ingestion seam) are **DONE**.
   A third domain, Argo, is live from outside `src/`, and a channel record can be read under any
   declared domain from tab 12. TG8.2 (licence provenance) is **deferred by decision** (see the
   sequence note above); TG8.3 (the domain ledger) remains. **No public dataset has been
   ingested** — the adapter reads local files and never fetches, so an archive adapter for Argo
   or TESS is still its own act. The probe described under *On auto-detecting a domain from its
   data* is partly built: `inspect` already converts observed facts into obligations, and what
   remains is doing the same for a gridded file.
3. **Phase G9 — the findings instrument.** TG9.1 (the read-only claim surface), TG9.2 (the
   findings view) and TG9.4 (accessibility of the new surface) are DONE. TG9.3 (the refusal surface) is DONE, carrying the half of TG9.1 that
   had been declared and not built. **Phase G9 is complete as declared.** Rendered browser
   inspection of the findings tab is **NOT RUN**, and no bundle records the domain that produced
   it, so `unadmitted_reading` describes a chosen vocabulary rather than a verified provenance.

4. **Phases G10-G13 — acquisition, the workbench, and three domains.** Declared 2026-08-27.
   **G10 and G11 are complete:** analysis, preregistration, evidence, motif mining, the
   cross-domain record and recorded review are reachable, and TG11.6 repays the workflow-wide
   accessibility source debt. G12 reaches the ocean, gridded and then Argo; G13 reaches
   the sky and closes the violation vocabulary. **No public dataset has been ingested by any of
   them yet.**

**Open defect:** D43 (the atmospheric real-data gate has not run through a feasible acquisition
path). D18 remains partial — ROCm/MPS and whole-platform device parity are unverified. D71 is
closed by the shared portable immutable-publication boundary after the deployed exFAT volume
exposed the independent-overlap writer's hard-link assumption. D69 is closed by TG12.2a's
explicit presence contract and scientific refusal of asynchronous frame lags. TG12.1 measured a
laptop-feasible long regional *ocean* layout, but that does
not close D43: it neither acquires ERA5 nor supplies the required cross-route overlap evidence.

---

### Phase G8 — Public dataset onboarding

Only once G0--G4 have shown that domains can be added without touching the core.

**TG8.1 The onboarding contract. DONE (`ed-dev`).** `src/core/onboarding.py` makes the recipe a
tuple rather than prose. `REQUIRED_DECLARATIONS` names the seven things a domain must declare —
axes and roles (E14), geometry (E13), a lag policy (R21), violated assumptions (E15/R17), a
licence, provenance, and a glossary (TG7.4) — each with the reason it is required.
`OnboardedDomain.checklist()` is generated from it, `GET /api/v1/findings/onboarding` serves it,
and the tests assert against it, so the documented contract and the enforced one are one object.

**The hole it closes.** Wording and limits registered separately, and either could exist without
the other. Wording without a declaration is a domain that speaks fluently and refuses nothing —
which TG9.1 shipped and served for a slice before TG9.3 caught it. `onboard_domain` registers
glossary, declaration and contract record **atomically**: every screen runs before the first
write, and any failure restores all three registries. `register_builtin_domains` now goes through
it and `register_builtin_glossaries` delegates to it, so no entry point can quietly reproduce the
state the contract forbids; a built-in found half-registered is repaired rather than skipped.

**The one check nothing else could make** is the biconditional between geometry and violations,
which no single object can see because the two facts live in different registries: a domain
declares `no_physical_metric` **if and only if** its geometry offers no physical metric, asked of
the geometry's declared capability rather than its name (E2). `/domains` now lists the union of
both registries, so limits registered without wording are visible rather than absent — the TG9.1
omission with its halves swapped, closed in the same slice that could have reproduced it.
`audit_onboarding` reports a piecemeal domain as `complete: false` with what is missing, instead
of letting it pass for one that was checked whole.

**Acceptance met:** `src/tests/domain_plugin_example.py` onboards a third domain in one file
outside `src/`, and the test hashes the six files a domain would otherwise have had to touch and
asserts they are byte-identical afterwards — the method `test_registries.py` uses for sources and
actions. The domain is **Argo profiling floats**, chosen by the reasoning in *Which domains get
onboarded* above rather than by sector: it breaks `irregular_sampling` and
`non_stationary_support`, which no registered domain had broken, is the first declared domain
where precedence is admissible while the clock is irregular, is the only one exercising
`lag_policy="declared"`, and keeps a physical metric — the side of the biconditional neither
built-in occupies. 33 tests, five deliberate mutations, each caught.

**Claim boundary.** The contract checks a declaration for completeness and internal agreement. It
reads no data file, so it cannot know whether a domain's declarations describe the source it
names, nor whether the wording chosen is wording a practitioner would use. It establishes nothing
about which domain produced any given `EvidenceBundle`, because a bundle still does not record
one. R17's refusals remain enforced in the analysis layer; this makes the declarations they read
from complete, not self-enforcing.

**TG8.2 Licence and terms provenance. DESCOPED (2026-08-28).** This line is a personal research
experiment over public data, not a commercial or redistributed product. There is no export path to
a third party for a source licence to govern, so the enforcement this slice described would guard
a boundary that does not exist here.

**Recorded as a limit, not as done.** No dataset licence, attribution requirement or access term is
carried in provenance, and no export refuses on licence grounds. The archives' own terms still bind
whoever uses the data — descoping the *check* does not descope the *obligation*, and the
attribution caveats already carried per domain are wording, not enforcement. Redistribution of a
derived product, publication, or any commercial use reopens this slice **before** that happens,
because the provenance it would have recorded cannot be reconstructed after the fact.

**TG8.4 The ingestion seam. DONE (`ed-dev`).** A declaration is not a connection. TG8.1 made a
domain declarable from outside `src/`; nothing could read a file under one. `src/api/channels.py`
mounts two routes over `src/data_layer/tabular_source.py`, which had read delimited channel
records since TG0.2 and was referenced **zero times** from `src/api/` and `frontend/src/`.

**The rule the slice is built on:** *detection may create a required declaration; it may never
satisfy one.* `POST /channels/inspect` reports a record's columns, rows and clock, converts what
it finds into **obligations** on whichever domain the reader picks, and names for every onboarded
domain whether it admits the file and the exact refusal if it does not. `POST /channels/read`
reads it against a named domain or refuses by name, and a test asserts the advertised refusal and
the enforced refusal are the same refusal.

**Three refusals that had no enforcement.** A domain whose declared axes include a role a channel
table cannot supply is refused (E14) — so `reanalysis` correctly cannot read a CSV. A channel
given a parent-axis footprint above one sample obliges the domain to declare `aggregated_values`
(E15/R17), which `_resolve_supports` had documented since TG0.2 and nothing had checked. A record
above a declared ceiling is refused by name and **never thinned**.

**Found while writing the tests, not designed in:** given a file whose clock ran backwards, the
first implementation quietly promoted a price column that happened to increase. No column is
substituted for a broken clock now; the candidates are named and the reader chooses.

**Tab 12, *Domain Records*,** is a peer of the meteorological tab: one reads grids, the other
reads channel tables for any declared domain. Three seeded fixtures ship in `data/channels/` —
regular, irregular, and a clock that runs backwards — with a README saying plainly that they are
fabricated and that no public dataset has been ingested.

**D61, found by this slice.** `order_book` described "an irregular trading clock" and did not
declare `irregular_sampling`. Four slices passed without it being noticed because nothing had yet
tried to *read data* under a declaration. Corrected, with the knock-on recorded: half of what
TG8.1 credited to Argo was really this gap.

**Claim boundary.** Reading a file under a domain establishes that the domain's declaration
admits the file's shape, and nothing more — not that the file came from it. The adapter never
fetches. The preview plot is a preview: nothing is mined and no rung moves (R22). 72 tests across
`test_channels_api.py` and `test_tabular_domain.py`, five deliberate mutations, each caught.

**TG8.3 The domain ledger.** For each onboarded domain: which assumptions it violated (R17),
which analyses it is therefore refused, and what onboarding cost. **If that cost is not falling
as domains accumulate, the abstraction is not working** — and the ledger is designed to make that
visible rather than deniable.

#### Which domains get onboarded, and why the obvious list is the wrong one

The question that prompted this section was whether to build ingestion for financial, crypto,
oceanic, celestial, ecological and social sources at once, so a researcher gets a ready-made
dropdown of domains. The dropdown is not the work: `GET /api/v1/findings/domains` is generated
from the registry and a domain appears in it the moment it registers, with its refusals beside
it (TG9.3). The expensive and load-bearing part is the declaration behind each entry, and rule
R17 decides which ones are worth writing: *a domain that violates nothing is not a second
domain.* Six domains that all break the same assumptions are one domain with six labels, and
they would give false confidence that the abstraction generalises — the same trap
`physical_core/geometry.py` names in its own docstring, where *"a fourth geometry that is
merely `cartesian` with a different name proves nothing."*

`KNOWN_VIOLATIONS` has seven entries. Coverage across the two registered domains:

| Violation | reanalysis | order_book |
| --- | --- | --- |
| `no_physical_metric` | – | yes |
| `no_propagation_speed` | – | yes |
| `unordered_channels` | – | yes |
| `aggregated_values` | – | yes |
| `no_natural_cycle` | – | – |
| `irregular_sampling` | – | – |
| `non_stationary_support` | – | – |

**Three assumptions have never been broken by any registered domain**, so the refusal machinery
for them has never fired against a real source. That gap chooses the next two domains:

*   **Oceanic — Argo float profiles.** Breaks `irregular_sampling` and `non_stationary_support`:
    floats surface on their own schedule, drift, fail, and are replaced mid-record. It keeps a
    physical metric and a real transport mechanism, so it is the first domain where **precedence
    is admissible but the clock is not regular** — a combination neither current domain has, and
    the sharpest available test of R21's floor against a source that is physical and untidy at
    once. Openly licensed.
*   **Celestial — TESS/ZTF photometry.** Breaks `no_natural_cycle`, `irregular_sampling` and
    `non_stationary_support`, and breaks the metric assumption *differently* from `order_book`:
    angular separation is a genuine metric that is not a length in metres. Open archives, no
    terms problem.

And two that are deliberately **not** onboarded:

*   **Crypto** is `order_book` with a different venue: the same four violations, the same lag
    policy, no refusal exercised that is not already exercised. Onboarding it would make TG8.3's
    cost ledger fall for the wrong reason, which is worse than not having a ledger.
*   **Social** is deferred rather than rejected. It contributes no violation Argo and TESS do not
    already cover, and it carries the hardest redistribution and personal-data terms in the list.
    It is a TG8.2 problem before it is a TG8.1 one. Financial equities fold into `order_book`.

#### On auto-detecting a domain from its data

Rejected in the form it is usually wanted, and accepted in a narrower one that is more useful.

`src/core/domain.py` opens by saying the failure mode is the analysis layer *silently* not
applying, and `DomainDeclaration.resolve_axes` therefore runs with `allow_name_inference=False`
and `allow_positional_inference=False`: an axis called `lat` must not acquire a spatial geometry
from its spelling. Inferring a whole domain from a file is that same inference at a larger
scale, and it would put the inferred answer in the very fields — `violations`, `lag_policy`,
`licence` — whose only purpose is to have an accountable author behind them.

The rule that keeps ingestion intuitive without making it silent:

> **Detection may create a required declaration. It may never satisfy one.**

A probe reads a candidate source and reports observed facts — dimensions, dtypes, cadence
regularity, gap structure, unit attributes, channel start and stop points — and then converts
what it found into *obligations*. Irregular timestamps do not set `irregular_sampling`; they
make the onboarding refuse to complete until the author either declares it or records why it
does not apply. Channels that begin and end mid-record do the same for
`non_stationary_support`. No unit attribute anywhere forces a geometry decision rather than
defaulting to `pixel`. The provenance then records that the declaration was made by a person and
*prompted* by a probe, which is a true statement; an auto-filled declaration would record a
false one.

#### Sequence

1.  **TG8.1, the onboarding contract** — first and unchanged. A domain can half-exist today:
    `builtin_glossaries` and `builtin_domains` register separately, so a domain can speak
    fluently while declaring nothing about what it refuses. That is exactly the hole TG9.1
    shipped and TG9.3 patched, and onboarding several sources before the contract exists is
    several more chances to repeat it.
2.  **TG8.2, licence provenance** — **deferred by decision, 2026-08-27.** The sequence above
    originally placed this before any archive adapter, on the assumption that redistribution
    mattered. It does not here: this is a personal experiment, not a commercial tool, and it
    will be exercised on public data that is not redistributed. Recorded as a scope decision
    rather than an oversight, and reversible — `DomainDeclaration.licence` is still a required
    field, so nothing has been removed. What is postponed is only the **export refusal** that
    would check it. Revisit if derived products are ever published or shared.
3.  **Argo, then TESS** — two domains, each justified by which unused violation it exercises.
    Note that TG8.4's D61 shrank the gap Argo was said to fill: `order_book` should always have
    declared `irregular_sampling`, so `non_stationary_support` is what Argo uniquely breaks.
4.  **The probe** — after two real adapters exist, so it generalises from cases rather than
    guesses at them.
5.  **TG8.3's ledger** — which measures whether onboarding cost is falling, and can only do that
    honestly if step 3 did not pad the list.

---

### Phase G9 — The findings instrument

Sits **above** G7 and may not reach past it. Where G7 ends, a finding exists as a
`TranslatedFinding`: domain wording, welded entitlements, and R9's six figures or none. G9 puts
that in front of a person.

**The governing principle: the frontend computes and formats no scientific number.** Every
claim-bearing string is produced by the backend and rendered verbatim. This is not a style
preference — it is the only way R9's *"a bare confidence percentage must not be renderable in the
UI; this is a hard constraint on the frontend, not only on the mining code"* becomes structural.
Today that rule depends on whoever writes the JSX. Under this phase a bare confidence is
**unobtainable**: `AssociationFigures.render` is the only thing that can format a percentage, and
it cannot exist without a base rate. R19 and R7 inherit the same protection, and TG7.4's welded
entitlements mean a view cannot show "this is a candidate precursor" while dropping "predictive
utility is not shown", because they are one string.

**The gap this phase starts from.** The cross-domain line has **no HTTP surface at all**. All
twelve core modules — `domain`, `feature`, `motif`, `evidence`, `claim_ladder`, `five_outputs`,
`recorded_call`, `round_robin`, `translation`, `cross_domain`, `constellation`, `family` — have
zero references in `src/api/`. Every existing route belongs to the atmospheric/transform line
(T3–T5). There is nothing for a UI to render until that is fixed, which is why this phase is
ordered API first.

**TG9.1 The read-only claim surface. DONE (`ed-dev`).** `src/api/findings.py` mounts six
read-only routes under `/api/v1/findings`: registered domains, one glossary whole, published
studies, one bundle, its five outputs, and its translation. Nothing appends evidence, records a
call or moves a rung, so a GET cannot change what may be claimed (R22).

**Phase G9's principle, applied to the wire.** A translation is served already rendered as
`rendered_text` beside its structured units and `structural_keys`, so a client displays strings
rather than assembling them. The five outputs are served untranslated as well, because a reader
checking that domain wording changed no fact needs both forms.

**R9 is enforced on the wire, not in each handler.** `refuse_bare_confidence` walks every
response body at any depth and refuses a `confidence` key not accompanied by all six of R9's
figures. The rule is usually described as a frontend constraint, which puts it in the one place
it cannot be enforced. The guard **restates** the six field names rather than importing them
from `AssociationFigures` — a guard that imported its expectations from the thing it guards
would agree with any change made to it — and a test asserts the two statements still agree. A
partial figure set is served as no figures at all: four of six is not four-sixths of a finding.

**Registration is eager, and that is the point.** Defect D35 was a fallback chain that depended
on browsing order, because registration was an import side effect of a lazily imported module.
`DOMAIN_GLOSSARIES` has exactly that shape, so `register_builtin_glossaries()` runs at module
import, is idempotent, and returns the full built-in set so a caller can assert it.
`src/core/builtin_glossaries.py` supplies the first two vocabularies — reanalysis and
order-book — written against TG7.4's four registration screens rather than fixed up afterwards.

**Delivered short of what this slice declared, and recorded rather than glossed.** The TG9.1
declaration said `GET /domains` would carry *"each domain's declared violations (E15) and lag
policy (R21) so a client can show what a domain refuses as readily as what it permits"*. It does
not. What was built lists **glossaries** — registered wording — not `DomainDeclaration`s, so
violations, lag policy and `precedence_admissible` reach no client. The route is honestly named
for what it serves and the acceptance criteria below are genuinely met, but the refusal half of
the declaration was not built and is **moved explicitly to TG9.3**, which is where the rest of the
refusal surface lives. A domain's refusals are still invisible from the API.

**Acceptance, both criteria met.** A response-shape test walks every route's JSON over a corpus
that genuinely does report a confidence — the test refuses to pass vacuously if none is present —
and no route can serve one without its five companions. And a glossary registered from the test
module, outside `src/core` and `src/api` both, reaches `GET /domains` and `GET /glossaries/{name}`
**without editing `src/api/`**: TG8.1's condition carried onto the HTTP layer.

**Evidence:** `src/tests/test_findings_api.py`, 20 tests. One study served through the reanalysis
and order-book vocabularies reads completely differently and returns byte-identical
`structural_keys` — TG7.4's acceptance, now visible over HTTP, which is what the findings view
will render. A GET leaves the bundle bytes and the rung unchanged; an unreadable bundle is
reported rather than skipped; an absent study root is an empty list rather than an error.

**Claim boundary.** A surface that cannot serve a bare confidence does not make the science good;
it removes one way of misreading it. A fluent rendering of a weak result is more persuasive than
a jargon-laden rendering of the same result, which is a risk this surface creates rather than
removes. An empty "evidence against" section means nothing was recorded, not that nothing exists.
The store reads a directory; it does not establish that anything in that directory was worth
publishing.

**TG9.2 The findings view. DONE (`ed-dev`).** `frontend/src/components/FindingsView.tsx` is an
eleventh tab rendering a `TranslatedFinding`: the five outputs as five sections, each claim shown
welded to the bound that qualifies it, with panels for the untranslated claim state, the glossary
that worded it, and the evidence bundle itself. A domain selector renders one study through any
registered vocabulary. Six client methods were added to `services/api.ts`; nothing in the existing
workbench was refactored.

**The rule is asserted over the source, not remembered.** `test_the_findings_view_formats_no_
scientific_number` refuses `toFixed`, `toPrecision` and any percent literal in the findings
components, and `test_the_findings_view_reads_no_claim_bearing_field_directly` refuses reading
*any* of R9's six fields — `confidence`, `base_rate`, `lift`, `support` and the rest — leaving
`figures_text` as the only route to the association strength. Comments are stripped before the
check, so the component can document the constraint without appearing to break it.

**A gap found while writing the view.** The first draft interpolated `figures.support` into its
own panel — no formatting, no arithmetic, and still wrong, because the moment a view builds that
line from parts R9 depends on the author remembering the base rate. The backend now serves
`figures_text`, the assembled line, so the view has nothing to assemble. The stricter test came
from that mistake rather than anticipating it.

**Acceptance met.** One study through two vocabularies reads completely differently and carries
identical `structural_keys`, served over HTTP and rendered verbatim. Three deliberate mutations of
the component — a `toFixed`, a read of `figures.confidence`, and a bare percent literal — were
each caught.

**Evidence:** five new tests in `src/tests/test_frontend_contract.py` (36 total). `npx tsc
--noEmit` clean; `npm run build` succeeds. **Rendered appearance is NOT RUN**: no browser has
displayed this tab, and no screenshot exists, exactly as for the tenth tab before it.

**TG9.4 Accessibility of the new surface. DONE for this surface (`ed-dev`).** The findings views
ship with `role="tablist"`/`role="tab"`, `aria-selected`, `aria-pressed`, `aria-label`, labels
bound with `htmlFor`, visible focus rings and `aria-hidden` on decorative icons, asserted by test.
**The legacy measurement is unchanged and restated rather than quietly improved:** accessibility
across `frontend/src` as a whole remains zero-derived and the transform workbench was not touched.
This slice stops the new surface adding to that debt; it does not repay it.

**TG9.3 The refusal surface. DONE (`ed-dev`).** What the instrument will not do, shown rather
than hidden — and carrying the half of TG9.1 that was declared and not built.

`DOMAIN_DECLARATIONS` is a registry in `src/core/domain.py`, beside the type it holds, so a domain
registers *what it is and what it breaks* the same way it registers its wording.
`src/core/builtin_domains.py` supplies the two declarations behind the built-in vocabularies, and
they are chosen to make the tension visible rather than to look tidy: **reanalysis** breaks nothing
and floors an advective lag, so precedence is admissible; **order_book** breaks four assumptions
and declares `lag_policy="none"`, so a lead-lag reading is inadmissible from it (R21). `/domains`
now serves each declaration, `refusals_for` assembles what it forbids, and the reason shown is
drawn from `KNOWN_VIOLATIONS` rather than restated, so what a reader is told and what the analysis
layer enforces cannot drift apart (R17).

**The constraint that shapes the slice: a bundle does not record which domain produced it.** An
`EvidenceBundle` carries a hypothesis, ten evidence categories and their digests, and nothing
about provenance of domain. So nothing here checks a study against a domain, and presenting the
limits as such a check would fabricate one. `DOMAIN_ATTRIBUTION_CAVEAT` travels with every served
limit, and a contract test refuses a view that restates the caveat in its own words rather than
rendering the one the API vouched for.

**The one genuinely new report.** When a record stands at `candidate_precursor` or above — the
rung at which a claim first asserts temporal ordering — and the selected vocabulary belongs to a
domain declaring no admissible lag floor, `unadmitted_reading` says so. It is careful about what
it means: the ladder is domain-agnostic and nothing here moves a rung (R22). If the study did come
from that domain, it is a contradiction someone must resolve; if it did not, the vocabulary is the
wrong one to read it in. **The surface cannot tell which, and says so.**

**Acceptance met.** A blocked bundle and a contradicted one render their refusals; commentary is
rendered in its own `<section>` and a structural test refuses any `TranslationUnit` field inside
that container, so recorded argument can never be mistaken for what the record permits.

**Evidence:** 9 new tests in `src/tests/test_findings_api.py` (29 total) and 4 in
`test_frontend_contract.py` (40 total). Two deliberate mutations of the view — claim text moved
inside the commentary container, and the caveat restated in the component instead of rendered from
the payload — were each caught.

**Claim boundary.** Showing what a domain refuses does not enforce it: R17's refusals are enforced
in the analysis layer, and this displays the same facts rather than adding a check. A domain that
declares no violations is not thereby unconstrained — reanalysis breaks nothing only because the
inherited assumptions were written against it. And the attribution gap is real: until a bundle
records its domain, `unadmitted_reading` is a statement about a vocabulary a reader chose, not
about a study. Closing that gap means putting domain provenance in the bundle, which is a change
to a G6 structure and belongs to no slice yet declared.

**Claim boundary.** A UI that cannot render a bare confidence does not make the science behind it
good; it removes one way of misreading it. Rendering a finding in domain words does not make the
finding true, and a fluent view of a weak result is more persuasive than a jargon-laden view of
the same result — which is a risk this phase creates rather than removes. The instrument shows
what was recorded: an empty "evidence against" panel means nothing was written down, not that
nothing exists, and the view must say so in those words. Accessibility for the new views is not
accessibility for the platform. Nothing here touches a gate, a rung or a digest (R22).

**What would falsify this phase.** If the findings view cannot be made useful without computing
something scientific in the frontend, the thin-renderer thesis is wrong and the R9 guarantee
cannot be structural. That would be discovered in TG9.2 and is a result, not a failure.

**Explicitly not in this phase.** The case-study explorer described in `roadmap.md` §9 — stepping
through a rule's supporting historical instances frame by frame — is the most valuable view that
document names, and it belongs to the atmospheric line's regional map rather than to the
cross-domain findings surface. Recorded here so its absence is not mistaken for oversight.

---

---

### Phases G10–G13 — Acquisition, the workbench, and three domains

Declared 2026-08-27, after TG8.4 made a domain readable from a local file and the question became
what it would take to reach real archives for the ocean and the sky. **Revised the same day**, when
a survey of the twelve tabs showed the cross-domain engine had no surface at all: G11 (the
workbench) was inserted before the two archive phases, which moved to G12 and G13. Acquiring data
for three domains before anything could analyse it would have built three paths to a dead end.

#### What decides the ordering, and it is not enthusiasm

**Only one inherited assumption is still unbroken.** After D61, six of the seven entries in
`KNOWN_VIOLATIONS` are declared by a registered domain. `no_natural_cycle` is declared by nothing.
That fact settles which of the candidate domains is scientifically load-bearing and which is
merely useful, and rule R17 requires the distinction be stated rather than blurred:

| Candidate | Breaks what nothing else breaks | Therefore |
| --- | --- | --- |
| Gridded ocean (GLORYS/ECCO/OISST) | nothing | a **data source**, not a second domain. Its value is the export capability and a different physical medium — water advects two orders of magnitude slower than air, which is a real test of `lag_policy="advective"` |
| **Argo profiles** | nothing new, but exercises `irregular_sampling` and `non_stationary_support` against **real data** for the first time | closes the gap between declaring a violation and demonstrating one |
| **Celestial photometry** | **`no_natural_cycle`** | the last unbroken assumption; completes coverage of the vocabulary |

**Three acquisition shapes, not one.** `CropSpec` expresses a region-and-time slice of a regular
grid. Argo is a region, a time window and a depth range returning an irregular scatter of
profiles. A light curve is a per-target series with sector gaps. Treating all three as crops is
how an abstraction quietly stops being one, so each gets its own spec type and the shared
machinery is shared deliberately rather than by force.

**Decisions taken with the user, 2026-08-27.** Free archive accounts are acceptable, with
credentials handled exactly as the Gemini key is — header-only, `.env.local`, redacted from `repr`
and from provider errors, never serialised into an artefact. And the interface is **consolidated
before** new domains land, so the sprawl of a tab per archive is never built and then undone.

#### Phase G10 — The acquisition surface

Generalise what tab 9 already does well. **No new archive in this phase.**

**TG10.1 The store catalogue becomes a registry. DONE** (2026-08-27; `src/data_layer/stores.py`,
`src/tests/test_stores.py`, `src/tests/store_plugin_example.py`; architecture.md §3.6zo;
VERIFICATION.md). `zarr_source.CATALOGUE` was a plain dict of four
ERA5 stores carrying ERA5-shaped fields (`resolution_deg`, `cadence_hours`, `levels`). Standard E1
exists to forbid exactly that shape: a fifth store cannot be added from outside `src/`.
`GRIDDED_STORES` becomes a registry whose entries declare the **domain** they belong to, their
access requirement, and their *measured* chunk facts — the existing notes are the model, recording
"measured 51.1x amplification" and a dated live inspection rather than an assumption. `CropSpec`
is generalised on one axis only, the vertical dimension name (`level` for ERA5, `depth` for ocean
products); `select()` already tolerates either latitude ordering, resolves `lat`/`latitude`, and
refuses a meridian wrap rather than guessing, and that logic is reused untouched. `content_key()`
stays machine-independent so the cache remains shareable and the provenance stays a reproduction
recipe rather than a description.
**Acceptance:** a fifth store registers from a file outside `src/` and reaches the catalogue
route, by the method `test_registries.py` already uses.

**Delivered, with three things worth recording.** *(1)* The acceptance criterion is met by the
method named — `src/tests/store_plugin_example.py` registers a `depth`-axis store no core module
imports, reaches `GET /api/v1/data/zarr/catalogue`, and `zarr_source.py` and `main.py` are hashed
byte-identical afterwards. It sits in `src/tests/` rather than literally outside `src/`, which is
where `plugin_example.py` and `domain_plugin_example.py` already live; the substance of the
criterion — no core file edited — is what is checked. *(2)* **A silent failure was found and is
now a test.** A depth-axis store read through the ERA5 path selected no vertical subset at all
and reported nothing, because the hard-coded `"level" in subset.coords` was simply false: the
full depth axis flowed into the cache while the manifest recorded the request. *(3)* **The
content key deliberately does not move.** `vertical_dim` enters the canonical form only when it
is not `level`, because adding it unconditionally would orphan every materialised crop and make
every recorded provenance record name a key that no longer resolves. One key is pinned to a
literal taken from the pre-TG10.1 module, so a future change to the canonical form is a decision
someone takes rather than a number someone updates.

**What TG10.1 did not do.** No store was opened and no live fetch was run — registering a store
is a declaration, and TG10.3 is what makes probing a recorded act and a precondition. The
generalisation reaches the *selection* path only; the cached-crop reader still speaks in pressure
levels and `level_hpa`, which is honest for the four ERA5 stores that exist and is the remaining
half of the job when a real depth-axis store arrives in TG12.1.

**TG10.2 The domain-first acquisition surface. DONE** (2026-08-27;
`src/api/acquisitions.py`, `frontend/src/components/AcquisitionView.tsx`,
`src/tests/test_acquisitions_api.py`; architecture.md §3.6zq; VERIFICATION.md). TG10.3 was
deliberately taken ahead of this slice, so the DONE markers are out of numeric order. Data
currently lives in tab 2, tab 9 and tab 12;
ocean and sky would make five tabs. Instead: **choose a domain, see what can be acquired for it,
make a selection.** One route family lists, per declared domain, its available acquisitions and
each one's shape (`grid_crop`, `profile_query`, `channel_table`). Tab 9 becomes *Acquire*, and the
roughly 255 lines of ERA5 UI presently inline in `App.tsx` move into a component as `FindingsView`
and `ChannelRecords` already are. Every acquisition carries the domain's declared limits, reusing
`refusals_for` and `DOMAIN_ATTRIBUTION_CAVEAT`.
**Acceptance:** every existing ERA5 capability reachable with no regression, and the tab count
does not grow when a domain is added.

**Delivered.** `GET /api/v1/acquisitions` projects the existing registries rather than creating
a parallel catalogue: every declared domain is listed first; registered stores become
`grid_crop` acquisitions; and the E14 channel-table rule decides whether `channel_table` is
available or is shown with its refusal. Every acquisition carries the domain declaration,
derived refusals and attribution caveat. `profile_query` is vocabulary, not a claimed
implementation before TG12.2.

Tab 9 is now **Acquire**. The former Domain Records tab is embedded under a selected
channel-table domain and removed from navigation, while the ERA5 crop fields, opt-in network
gate, probe and transcription ledger, metadata-only inspection, cached-crop readiness claims
and materialisation command remain reachable under grid crops. Navigation therefore remains
eleven tabs when a domain registers; domains and acquisitions are mapped from the API rather
than named in `App.tsx`. TypeScript and the production build pass. Rendered browser inspection
is **NOT RUN**. The complete suite passes: **2509 passed, 2 skipped, 1 xfailed**.

**Claim boundary.** This phase ran no live archive probe and fetched no public data. A listed
path is a capability, not evidence that it ran, and selecting a domain does not establish that
a local file came from it.

**What TG10.3 left for it.** The probe ledger (`GET /api/v1/data/zarr/probes`) and the probe
button now sit inline in the ERA5 tab, added there because a served route nobody can reach is a
capability the platform does not really have. They move into the *Acquire* surface with the
rest of that UI, and the transcription count belongs where a researcher chooses a store rather
than beside a form field.

**TG10.3 Store probing as a recorded act. DONE** (2026-08-27; `src/data_layer/store_probe.py`,
`src/tests/test_store_probe.py`; architecture.md §3.6zp; VERIFICATION.md). Taken **before**
TG10.2 rather than after, because the acquisition surface renders exactly what the probe
produces and building it first would have meant rendering transcribed prose and revising it a
week later. Registering a store whose behaviour nobody measured is
how D43 happened. A probe opens a URI and records its dims, variables, chunk shape, bytes per
chunk and the amplification a stated crop would suffer — and **records the result either way**,
including "unreachable" or "needs credentials", which are results rather than failures. No store
may be registered without one.
**Acceptance:** the four ERA5 stores' recorded notes are reproduced by the probe, and a
deliberately hostile store is characterised as hostile before anyone crops it.

**Delivered.** The hostile half of the acceptance criterion is met offline and the word
*before* is what is checked: a hostile fixture is characterised as hostile with nothing
materialised, no cache entry created, and the amplification agreeing exactly with what
`assess_access_pattern` predicts from chunk metadata. The ERA5 half is met only as far as
offline can take it — the four recorded notes are held as **transcriptions**, `evidence="prior
recorded inspection"`, counted by `transcribed_probes()` and published by the ledger route so
the number can only fall in the open. The opt-in live probe that would check a transcription
against the real WeatherBench store is written and **NOT RUN**.

**Three things worth recording.** *(1)* **A refusal is a result.** `unreachable`, `needs
credentials` and `network is switched off` are recorded rather than raised, and the HTTP route
returns 200 with a recorded outcome where `/inspect` returns 409. *(2)* **The registration gate
is where the slice has teeth**: a store claiming a measurement must cite a probe, of its own
URI, whose figures it does not contradict. *(3)* **Two mutations survived the first pass** and
both were weak tests rather than weak guards — a fixture whose two variables were identically
sized could not tell a maximum from a mean, and an assertion that merely required *something*
to raise let the probe requirement be deleted, because the next check happened to raise too
with an error that would have sent an author looking in the wrong place.

**Two defects found and fixed here, both introduced by TG10.1.** **D62**: `era5_0p7_6h` was
given a per-chunk size of 8.0 MB that no inspection produced, in the registry built to refuse
exactly that — and it passed because `ChunkFacts` *demanded* a positive figure for any method
other than `not measured`, which made the dishonest entry the easy one. **D63**: adding
`vertical_dim` to `CropSpec` changed `to_provenance`, and that record is embedded in
**authenticated** artefacts, so a checked-in signed preregistration stopped loading. Found by
the full suite **after TG10.1 was reported and committed**, because the slice was reported on
the strength of targeted suites while the full run was still going.

#### Phase G11 — The workbench: making the engine reachable

Declared 2026-08-27, and it exists because a survey of the twelve tabs found something the
roadmap had not said out loud.

**The application is two platforms sharing a shell.** Measured, not estimated:

| Line | Tabs | Operates on |
| --- | --- | --- |
| Gridded physical field | 1 Synthetic, 2 Meteorological, 3 Boundary lab, 4 Spectral transforms, 5 Diagnostics, 9 ERA5 Zarr, 10 Forecast evaluation | `PhysicalField` / `GridSpec` — 2-D arrays with a metric |
| Cross-domain channels | 11 Findings, 12 Domain records | `ChannelSeries` and the claim ladder |
| Genuinely generic | 6 Experiment engine, 8 Platform & evidence | registries, health, device probing, benchmark listing |

**Genericising the gridded line is the wrong goal and must not be attempted.** A two-dimensional
dual-tree wavelet transform of an order book is not a cross-domain capability, it is a category
error. Boundary conditions, advective floors and radial binning in physical wavenumber assume a
spatial grid *correctly*; making them domain-agnostic would mean making them produce numbers where
they have no basis, which is precisely what `no_physical_metric` exists to refuse. What that line
needs is honest **labelling** — it is the gridded line, not the whole application — and the
navigation currently implies otherwise.

**The real gap is that the cross-domain engine has no surface at all.** Measured by reference
count:

```text
analysis_engine/domain_analysis.py    api=0  ui=0
core/cross_domain.py                  api=0
core/motif.py  motif_freeze  motif_transfer  motif_relationship    api=0
core/constellation.py  family.py  invariance.py  preregistration.py  api=0
core/round_robin.py  recorded_call.py                              api=0
```

`domain_analysis` offers `analyse_precedence`, `association_only` and `run_domain_gate`, each
taking a `ChannelSeriesLike` and a `DomainDeclaration`. That is the entire scientific payload for
every non-gridded domain, it is tested, and **nothing can reach it**. TG8.4 therefore dead-ends: a
researcher can load an order book or an Argo record and then do nothing whatever with it. Phase
G10 gets data in across three domains and — stated plainly — gives them nothing to do with it.
This phase is what makes acquisition lead somewhere, which is why it is sequenced **before** the
ocean and the sky rather than after.

**One encouraging finding.** The benchmark suite is already cross-domain: `GET /benchmarks`
returns **11 `sequence`, 7 `field` and 2 `cross_domain`** entries, including
`planted_precedence`, `precedence_null` and `coupled_cascade_sequence` — thirteen non-gridded
benchmarks whose correct answer is known before analysis. A channel-analysis surface can therefore
be validated against ground truth on the day it is built, which is unusual and must be exploited
rather than wasted.

**The governing rule of the phase**, and the reason it is not merely UI work:

> **The interface may record evidence. It may never assert a rung.**

A rung is always *derived* by `claim_ladder` from evidence the client supplied; no request body
carries a claim level, exactly as `AssociationFigures.render` is the only thing in the programme
that can format a percentage. Every write path added here is a place where a rung could move for
the wrong reason (R22), so each is added one at a time and each gets the treatment the ladder got
in G6.

**TG11.0 Information architecture. DONE** (2026-08-27;
`frontend/src/App.tsx`, `frontend/src/components/AcquisitionView.tsx`,
`frontend/src/components/ChannelRecords.tsx`, `frontend/src/components/FindingsView.tsx`,
`src/tests/test_frontend_contract.py`; architecture.md §3.6zr; VERIFICATION.md). Twelve flat
numbered tabs already read as a list rather than
an instrument, and this phase adds more. Navigation is grouped into the sections the scientific
workflow actually has — acquire, analyse, evidence, review, read, platform — with the gridded line
labelled as the gridded line. Tab *count* is not the constraint; legibility is. A persistent
selected record and study becomes the app's context, so a researcher chooses a record once rather
than re-selecting it in every panel: this is the single largest usability win available and it
costs almost nothing.

**Delivered.** The eleven destinations are no longer numbered or presented as peers. They sit
under Acquire, Analyse, Evidence, Review, Read and Platform; the spatial-only tools are labelled
**Gridded field line**, so the shell no longer implies that a wavelet transform or boundary
condition applies to every domain. Review was introduced here as an honest labelled waypoint for
TG11.5 rather than a button to a surface that did not exist; TG11.5 now fills that waypoint
without changing this navigation contract.

`selectedRecord` and `selectedStudyId` now belong to `App`, appear in a persistent research-
context strip and are passed into Acquire and Findings. The selected-record context retains the
original browser `File`, clock choice and aggregate supports as well as the API's bounded preview,
so TG11.1 need not analyze truncated preview data or ask for the file again. Loading a channel
record updates shell context; returning to Acquire restores its domain and preview. Selecting
a published study updates the same shell context and survives navigation. Both selections can
be explicitly cleared. The default destination is Acquire, matching the workflow rather than
the former implementation order. The production build and all 52 frontend contract tests pass;
the complete suite passes **2511 passed, 2 skipped, 1 xfailed**. Rendered browser inspection is
**NOT RUN**.

**Claim boundary.** Grouping tools changes reachability and labelling only. It does not make the
gridded line domain-general, and retaining an in-browser `File` is not durable evidence storage.
TG11.0 adds no scientific compute, evidence write path, claim level or confidence figure.

**TG11.1 The analysis surface. DONE** (2026-08-27; `src/api/analysis.py`, `src/api/main.py`,
`frontend/src/components/DomainAnalysisView.tsx`, `frontend/src/App.tsx`,
`frontend/src/services/api.ts`, `frontend/src/types/api.ts`, `src/tests/test_analysis_api.py`,
`src/tests/test_frontend_contract.py`, `src/tests/test_documentation.py`; architecture.md
§3.6zs; VERIFICATION.md). `domain_analysis` over HTTP: run a domain gate, an
association-only analysis or a precedence analysis against a loaded record, with the domain's
refusals enforced (R21 stops a precedence claim from a domain with no lag floor) and reported.
Read-only compute — it stores nothing and moves no rung.
**Acceptance:** the thirteen `sequence` and `cross_domain` benchmarks are reproduced **through the
HTTP layer**, not merely in process, and every null benchmark still answers "there is nothing
here". A surface that finds the planted coupling but also finds structure in the AR(1) nulls has
found nothing (T4C.3).

**Delivered.** Two endpoints, no new science. `GET /api/v1/analysis` publishes the three operations
and the claim boundary; `POST /api/v1/analysis/run` executes exactly one of them by calling
`association_only`, `analyse_precedence` or `run_domain_gate` unmodified. No estimator,
correction, lag floor, surrogate or verdict is reimplemented at the boundary, which is the only
way the HTTP answer means what the in-process answer means.

The surface re-reads the **original** admitted file that TG11.0 retained, never `/channels/read`'s
bounded preview, and derives cadence, frame count, channel count, measure and source identity from
it. The researcher declares only the hypothesis family and estimator settings. Four refusals fire
before any computation: R21 for a domain with no admissible floor, with the engine's own wording
and both remedies intact; an unknown configuration key, refused per operation rather than ignored;
an irregular clock, because a lag in frames is not a physical duration on one; and a non-UTF-8
upload. Every success states `stored: false` and `rung_moved: false` (R22).

**Acceptance met.** All thirteen `sequence` and `cross_domain` benchmarks run through the live
FastAPI test client on the existing `POST /benchmarks/run` route — no duplicate benchmark path —
and report 13 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE with an empty `null_failures` list. **Eight are
nulls** and every check of every one of them answers "there is nothing here". The panel's verdict
chip was corrected in the same slice: `INVALID` now has its own branch and its own sentence, since
presenting a design that did not hold beside FAIL's "there is nothing here" would report a design
error as evidence of absence.

Full suite **2521 passed, 2 skipped, 1 xfailed**; frontend production build 1,390 modules with
JS/CSS emitted; documentation guard passing with the new router registered in `_ROUTE_SOURCES`.
Rendered browser inspection is **NOT RUN**.

**D64, found by mounting the router.** The documentation guard's route scan used a `[^"]+`
path pattern, so a route at its router's own prefix was invisible to it: `GET
/api/v1/acquisitions` had been served and unseen since TG10.2, and the documented route count read
42 against a served 43 while the check passed. Fixed here; the count is now 45.

**Claim boundary.** Reachability is not correctness. The thirteen benchmarks argue for correctness
on the thirteen cases they cover and on nothing else. This slice records no evidence, derives no
rung, preregisters nothing and spends no held-out partition.

**TG11.2 Preregistration first. DONE** (2026-08-27; `src/api/preregistration.py`,
`src/api/main.py`, `src/api/analysis.py`, `src/tests/test_preregistration_api.py`,
`src/tests/test_frontend_contract.py`, `src/tests/test_documentation.py`,
`frontend/src/components/PreregistrationView.tsx`, `frontend/src/App.tsx`,
`frontend/src/services/api.ts`, `frontend/src/types/api.ts`; architecture.md 3.6zt;
VERIFICATION.md)

**Delivered.** Six endpoints over `core/preregistration.py`, which was 574 lines nothing outside
the tests could reach: capabilities, a partition description built from geometry and lineage
without reading a measure value, sealing, a seal listing, a seal read with both digest layers
recomputed, and the one-shot confirmation. No new science - every digest, refusal and correction
is the module's.

**The ordering is the server's.** A confirmation against a seal that does not exist is refused
before the record is read; against a partition the seal did not name, by `PartitionMismatchError`;
against a spent partition, by the ledger. The panel makes the ordering legible and enforces none of
it. TG11.1's `domain_gate` is now gated too: it splits internally and returns a verdict on its own
test partition, so it reads the ledger first and is refused `409` on data already spent.

**The client cannot tune a confirmatory run.** `/confirm` takes the record and an optional
published digest. Lags, ensemble size, alpha, correction, estimator, bins, seed, split, domain,
clock and delimiter all come out of the seal, frozen as the confirmatory specification's `notes`,
so editing one breaks the seal's digest instead of quietly running a different analysis under it.
The sealing time is the server's, because a caller-supplied one could be written after the
partition was opened.

**Acceptance met.** A confirmation refused for want of a seal; an edited seal naming the field that
changed; a wrong published digest refused; and the one that matters - the same held-out data
refused a second confirmation under a second seal that is individually perfect, correctly hashed,
correctly narrowed and affordable at its own size. Each such confirmation would be defensible
alone; the pair would be uncorrected, and that is the arithmetic R18 exists to stop. A refused
confirmation is also shown to leave the partition unspent.

**D65.** The held-out ledger's "once" could be defeated by renaming a file.
`PartitionIdentity.from_series` hashes provenance wholesale and an uploaded record carries
`path_basename` in its provenance, so the same held-out bytes re-uploaded under another name
hashed to another partition and the ledger did not fire. The domain the file was read under had the
same problem for the same reason. Found while writing the double-spend acceptance, which passed
against a same-name re-upload and would have passed for the wrong reason. The identity is now built
at the boundary from the content digest, the clock, the columns and the split window.

**Still unreachable, named rather than left quiet.** `report_generation` has no route. It checks
that a candidate list was drawn from declared members and mined on train rather than on the
held-out partition, and it needs a candidate list from a real training sweep - a mining surface
rather than a preregistration one. A thin route for it here would have made the module look covered
before the thing it guards exists. It belongs with TG11.4.

**Verified.** Full suite **2543 passed, 2 skipped, 1 xfailed**; frontend production build with
JS/CSS emitted; documentation guard passing with the new router registered in `_ROUTE_SOURCES` and
the route count moved to 51. Rendered browser inspection is **NOT RUN**.

**Claim boundary.** A seal is a promise about ordering, not a result: it says a family was fixed
before a partition was opened, and nothing about whether the family is any good. A confirmation
receipt records no evidence and moves no rung (R22); writing one into a bundle is TG11.3. The
generated family is corrected for nowhere here and its members are not claims. Seals and the ledger
are programme state under `data/preregistrations`, not a cache - deleting them destroys the record
of what has been spent - and the JSON ledger has no lock, so "once" is once per server.

**TG11.3 The evidence write path. DONE** (2026-08-27; `src/api/evidence.py`, `src/api/main.py`,
`src/api/findings.py`, `src/tests/test_evidence_api.py`, `src/tests/test_frontend_contract.py`,
`src/tests/test_documentation.py`, `frontend/src/components/EvidenceView.tsx`,
`frontend/src/App.tsx`, `frontend/src/services/api.ts`, `frontend/src/types/api.ts`;
architecture.md 3.6zu; VERIFICATION.md)

**Delivered.** Five endpoints over `core/evidence.py` and `core/claim_ladder.py`: capabilities,
opening a study at revision zero, reading the head, appending one entry, and appending a
provenance entry whose precedence verdict the server computes. No digest, chain check or ladder
gate is reimplemented at the boundary. This is the first surface in the programme that writes
toward a claim, so it is the first that could break R22, and the rule is enforced structurally
rather than by review: no request model has a field for a rung, a claim level or a confidence,
unknown fields are forbidden rather than ignored, and the rung in every response is the ladder
recomputed over the chain that was just written and stored nowhere.

**The key that could have climbed a rung by typing.** The ladder reads exactly one payload key -
`temporal_precedence`, on passing `provenance` entries - and it gates `candidate_precursor`. A
free-form payload would let a client assert its way up a rung with no arithmetic anywhere for an
estimator to notice. It is refused at any depth of a hand-written payload and written only by
`/evidence/precedence`, which runs `analyse_precedence` here and records what it returns, `false`
included; the entry cites the record's `content_sha256` and the sweep's `analysis_config_sha256`,
so what was recorded is checkable against what was computed. An underpowered sweep is recorded
`INCONCLUSIVE` and opens no gate: a family that could not have rejected anything did not check
anything (R5).

**Append-only, and compare-and-swap.** Every append names the head it extends. A bundle is
immutable and `save_evidence_bundle` refuses to overwrite, so each revision is published as its
own file created exclusively - the exclusive create *is* the concurrency control, and two writers
racing from one head produce one append and one 409 rather than a lost entry. Deliberately
stronger than TG11.2's ledger, which takes no lock: evidence is the thing being protected.
`recorded_at` is the server's clock and is on no request model, so an entry cannot be back-dated
into an order it did not happen in.

**Acceptance met.** The rung cannot be reached by typing, tested from both sides: a request
carrying a rung is refused rather than ignored, and a payload asserting `temporal_precedence` is
refused and told which route computes it. The ladder is then shown climbing to `association`
through three appends that named nothing, and one `FAIL` entry pulling it back to `observation`
over favourable evidence already recorded.

**D66, found by asserting through the read surface what the write surface had just returned.**
`StudyStore.load` resolved a study by taking the first parseable file whose `study_id` matched.
Correct while nothing wrote bundles; wrong the moment revisions became separate files, because
sorted-first is `r00000` - the read surface would have served revision zero for ever while the
write path reported the revision it had appended, and one study worked on five times would have
listed as five studies. Resolution is now by chain rather than by name. Two consistent halves of
one store can agree with each other and both be wrong, which is why the check is cross-surface.

**Verified.** Full suite **2570 passed, 2 skipped, 1 xfailed**; frontend production build 1,392
modules with JS/CSS emitted; documentation guard passing with the new router registered in
`_ROUTE_SOURCES` and the route count moved to 56. Rendered browser inspection is **NOT RUN**.

**Claim boundary.** Recording evidence is not establishing a finding: the ladder grades what is in
the chain, and a chain of one favourable observation earns `observation`. A bundle carries no
domain, so nothing written here records which instrument the evidence came from. This surface does
not consult the held-out ledger - running a precedence analysis over data preregistered as held
out spends it outside the record, which the ledger cannot see and the capabilities note cannot
prevent. Commentary has no category here and prose is not evidence (R22). Bundles are programme
state under `data/studies`, not a cache.

**TG11.4 Structure mining. DONE** (2026-08-27; `src/api/mining.py`, `src/api/main.py`,
`src/api/preregistration.py`, `src/core/motif.py`, `src/tests/test_mining_api.py`,
`src/tests/test_frontend_contract.py`, `src/tests/test_documentation.py`,
`frontend/src/components/StructureMiningView.tsx`, `frontend/src/App.tsx`,
`frontend/src/services/api.ts`, `frontend/src/types/api.ts`; architecture.md 3.6zv;
VERIFICATION.md)

**Delivered.** Eleven endpoints over `core\motif.py`, `core\constellation.py`, `core\family.py`,
`core\invariance.py`, `core\motif_freeze.py` and `core\motif_transfer.py` - 4,211 lines that
nothing outside the test suite could call. Admit a field and extract its features;
list what is admitted; price the family a mining pass would examine before mining it; calibrate
a match tolerance; mine the training frames; freeze the confirmatory family against held-out
frames; open those frames once; publish one motif as a durable definition; transfer it into a
second domain through a ledger that spends the target before the target is read; and audit every
registered matcher against its own declared invariance. No matcher, null, correction, tolerance
or p-value is implemented at the boundary.

**The two things that cannot be typed.** A motif is a configuration of extracted features, so a
surface that accepted feature coordinates would let a caller draw the shape they wanted confirmed
- and every number downstream would then be arithmetically correct and empty. No route accepts a
feature: a caller admits a `.npy` stack of frames and the server extracts, under settings that
become part of the record's digest. And the match tolerance decides which configurations count as
repeats, so it travels as the digest of a calibration the server performed rather than as a
number; measured the tempting way it came out fifty times too wide in this tree's own benchmark,
wide enough that every triangle matched every other.

**Refused rather than repaired.** Frames whose feature counts disagree are refused, and the
refusal names the repair it is declining - keep the brightest six - because magnitude ordering
moves between noise realisations, so "the brightest six" is a different configuration in every
frame. A tolerance calibrated through another pipeline, or on frames past the training boundary,
is refused for the same class of reason.

**The confirmation is driven by the seal.** `/confirm` takes a seal digest and an optional
published one and nothing else; the record, the split, the size, the matcher, the tolerance, the
ensemble, the correction and the seed are sealed as notes on the confirmatory specification, and
the training candidates are re-derived by re-running the deterministic mining pass and checked
label for label against what the seal froze. One core change, made beside rather than instead
(E12): `confirmatory_specification` and `freeze_motifs` take an optional `notes` mapping, merged
beside the notes they already write, so the run settings are inside the seal's digest rather than
in a file next to it. Mining seals go into TG11.2's seal store and spend TG11.2's held-out
ledger, because "this partition has been opened" is one fact about the programme.

**Acceptance met.** `test_a_null_record_confirms_nothing` runs the whole chain over frames with
nothing planted in them - same generator, same feature count, same family, same ensemble - and
confirms nothing, which is the gate this phase exists to pass. Beside it,
`test_the_planted_motif_is_confirmed_on_frames_it_was_not_mined_from` shows the pass can still
find what is there. `test_no_route_on_this_surface_accepts_a_feature` and
`test_the_tolerance_cannot_be_typed` are the structural pair.

**Not delivered here, and named rather than dropped: `cross_domain`.** The roadmap bullet listed
`core\cross_domain.py` with the mining modules. It does not belong on this surface: its input is
two channel tables aligned on an exact common clock, not scenes of extracted features, and its
sweep is a lag family - the same shape `/api/v1/analysis` and `/api/v1/preregistration` already
serve. Routing it here would have put a channel pipeline behind a scene vocabulary. It is
carried as **TG11.4b** below, beside the analysis surface where its inputs already live.

**Verified.** Full suite **2619 passed, 2 skipped, 1 xfailed** (4,090 s); `test_mining_api.py` 41;
`test_frontend_contract.py` 65 -> 73; documentation guard 19 passing with the new router in
`_ROUTE_SOURCES` and the route count moved to 67; frontend production build with JS/CSS emitted;
`tsc --noEmit` clean. Rendered browser inspection is **NOT RUN**.

**Claim boundary.** Mining produces candidates, and support on the training frames is selection:
every exemplar is one of the occurrences it is counted among, so its support starts at one by
construction and it was ranked highly for having been counted often. A confirmed motif is a
configuration that recurred on frames it was not mined from more often than the surrogate null
placed it there - not a mechanism, not a cause, and not a claim until something records it, which
is TG11.3's write path (R22). A motif reported `vacuous` was confirmed by an ensemble that could
not have rejected it, which is not a confirmation (R5). A transfer match count is descriptive.
The transfer ledger establishes ordering inside this API and cannot show that nobody looked at
the target before it was admitted. Admitted fields, tolerances, definitions and the transfer
ledger are programme state under `data/mining`, not a cache.

**TG11.4b The cross-domain record. DONE** (2026-08-27; `src/api/cross_domain.py`,
`src/api/main.py`, `src/core/precedence.py`, `frontend/src/components/CrossDomainRecordView.tsx`,
`frontend/src/{App.tsx,services/api.ts,types/api.ts}`, `src/tests/test_cross_domain_api.py`,
`src/tests/test_frontend_contract.py`, `src/tests/test_documentation.py`).

**Delivered.** `core/cross_domain.py` - 505 lines nothing outside the benchmarks could call - is
reachable on `/api/v1/cross-domain` through seven endpoints: capabilities, `align`, `lags`,
`partition`, `generate`, `seal`, and `confirm` addressed by seal digest. This is the slice where
a lag family stops being expressible only in frames of one clock: durations are declared in
**seconds** and converted per domain against each domain's own registered floor, before the two
clocks are combined. No estimator, null, correction or lag floor is implemented at the boundary.

**Three things that cannot be typed here.** There is no resampling control: the records are
aligned by exact timestamp intersection and a pair that shares too little is refused, with the
refusal naming interpolation as the thing it declines. There is no defaulted unit: every column
must carry its `semantics` and `units` before the record can be read, because a channel table
carries neither and R19 does not permit either to be dropped - and both are restored to each
confirmed relationship in the receipt. And there is no setting a caller can still turn after the
seal: `/confirm` takes the two records and, optionally, a published digest.

**Refused rather than repaired.** A duration below either domain's physical floor is refused
rather than dropped from the declared family; one that is not a whole number of the exact common
cadence is refused rather than rounded, because a rounded lead is a lead the clock cannot
express; a within-domain pair is not a member at all. Every response reports what the alignment
retained and discarded per domain.

**Seal-driven confirmation.** The run settings are sealed *inside* the confirmatory
specification, which needed one additive core change: `precedence.confirmatory_specification` and
`precedence.freeze_precedence` take an optional `notes` mapping merged beside the notes they
already write (E12; no existing caller sees a change). The frozen members are re-derived by
re-running the deterministic training sweep rather than reconstructed from their labels. Seals
land in TG11.2's store and spend TG11.2's one held-out ledger.

**Acceptance.** `test_the_planted_relationship_is_confirmed_on_data_it_was_not_selected_from`
runs the whole chain over an hourly domain and a three-hourly one with one delayed relationship
planted across the boundary, and confirms exactly that relationship at exactly the planted
duration on held-out frames. `test_the_same_pipeline_over_an_uncoupled_pair_confirms_nothing` is
the same generator with the coupling knob at zero - same channels, same cadences, same family,
same ensemble - and confirms nothing.

**One residual, recorded not papered over (D65).** The held-out partition identity is built from
the data: two content digests, both clocks, the shared clock digest, the retained and discarded
counts, the split window and the embargo. The filename is not among them, so renaming a file does
not create a second partition. Each domain's declaration *is* among them, so the same two files
aligned under two different registered domains are two partitions to the ledger and could be
opened twice. Stripping the domain out would make two genuinely different analyses collide, so
the residual is stated in the claim boundary every response returns rather than removed.

**Verified.** `test_cross_domain_api.py` 35 passed; `test_frontend_contract.py` 80 passed (was
73); `test_documentation.py` 19 passed; complete suite 2661 passed, 2 skipped, 1 xfailed;
frontend production build 1,394 modules transformed; `tsc --noEmit` clean.

**Claim boundary.** A cross-domain result is a temporal association between structural series:
one domain's series carries information about another's later series. Raw magnitudes keep
different semantics and units and are never compared, and precedence identifies no causal
mechanism (R19, R21). A confirmation receipt records no evidence and moves no rung (R22) - it is
an input to TG11.3's write path. Rendered browser inspection of the new panel is **NOT RUN**.

**TG11.5 The review surface. DONE** (2026-08-28; `src/api/reviews.py`,
`frontend/src/components/ReviewView.tsx`, `frontend/src/{services,types}/api.ts`, `src/api/main.py`,
`src/tests/{test_reviews_api,test_frontend_contract}.py`; architecture.md §3.6zy;
VERIFICATION.md). The adversarial round-robin, its recorded calls and its cost
receipts (`round_robin`, `recorded_call`, `review_cost`). Everything here is R23
recorded-not-reproducible, so the surface must present it as recorded argument and never as
something the record permits — the separation TG9.3 already enforces for commentary.

**Delivered.** This is one GET-only surface under `/api/v1/reviews/studies/{study_id}`. Artifacts
live in a dedicated review root and are classified by declared schema, not filename; every record,
outcome and receipt is reconstructed through the core type so its digest is verified on read.
Unreadable and unknown artifacts remain visible as refusals.

The selected study resolves to its latest immutable bundle first. A record attaches only when
study id, bundle digest and revision all match; an outcome and cost receipt must additionally bind
the record digest. Commentary on an older revision therefore cannot appear current. The UI is a
separate Review workspace rather than a Findings panel, leads with the backend's R23 declaration
and claim boundary, shows the complete core-rendered calls and retained dissent, and presents the
cost receipt as token/route audit with no price or quality claim. Missing records, outcomes and
receipts each state what their absence does not establish.

**Verified.** `test_reviews_api.py` 8 passed; `test_frontend_contract.py` 92 passed (six TG11.5
contracts added); production `tsc` and Vite build pass with 1,395 modules transformed. Complete
suite **2681 passed, 2 skipped, 1 xfailed**. Rendered inspection was attempted through the
configured in-app browser, but its runtime reported no available browser backend; it is **NOT
RUN**.

**Claim boundary.** Everything on this surface is R23 recorded-not-reproducible: recorded
argument, never evidence or something the record permits. No review output enters a bundle, no
GET can move a rung, no vote count is performed, and deleting every review still changes no claim
level (R22, R23). A cost receipt establishes route and token accounting only, not review quality.

**TG11.6 Accessibility, repaid rather than deferred. DONE** (2026-08-28;
`frontend/src/App.tsx`, `frontend/src/index.css`, `frontend/src/components/{Heatmap2D,LineChart,
LineageGraph,FieldImport,EvaluationEvidence,CrossDomainRecordView}.tsx`, the seven workflow
surfaces, `src/tests/test_frontend_contract.py`; architecture.md §3.6zx; VERIFICATION.md).

**Delivered.** The application has one keyboard contract rather than accessible islands. A skip
link reaches a named main landmark; workflow changes move focus to a programmatic workspace
heading; the current workspace and asynchronous state are announced; and the unkeyboardable
backend-retry `span` is a native button. Every legacy control in the monolithic gridded panels is
bound to its visible label. One high-contrast `:focus-visible` rule wins even over the old
`focus:outline-none` utilities, and reduced-motion preference collapses transitions and animation.

The scientific graphics retain a text route: heat maps report their shape, units, axes and valid
inset; line charts report series, axes and logarithmic scales; and every SVG lineage node is a
named pressed-state control activated by Enter or Space. Cross-domain operands are fieldsets whose
file, domain, clock and semantic declarations each have an accessible name. Acquire, Analyse,
Evidence and Read surfaces expose their busy state; errors are alerts and progress is status.

**Verified.** Six new contract tests make the source semantics executable:
`test_frontend_contract.py` 86 passed (was 80); production `tsc` and Vite build pass with 1,394
modules and emitted JS/CSS. Complete suite **2667 passed, 2 skipped, 1 xfailed**. Rendered keyboard
inspection was attempted through the configured in-app browser, but its runtime reported no
available browser backend; it is **NOT RUN**, stated rather than inferred from the build.

**Claim boundary.** This establishes semantic wiring and keyboard paths in source; it does not
establish a WCAG conformance level, screen-reader quality or visual focus placement in a rendered
browser. Figure summaries describe carried metadata and do not interpret the scientific result.

**Claim boundary.** Making a capability reachable is not evidence that it is correct; the
benchmarks are what argue for correctness, and only for the thirteen cases they cover. A grouped
navigation does not make the gridded line domain-general, and this phase deliberately does not try
to. No write path added here may accept a rung, a claim level, or a confidence figure from a
client.

#### Phase G12 — The ocean

**TG12.1 A gridded ocean product. DONE (2026-08-28, `ed-dev`).** Candidates were chosen by
TG10.3's probe rather than by reputation:
GLORYS (Copernicus Marine, free account), ECCO (NASA Earthdata, free account), NOAA OISST (open).
Declared honestly under R17 as **breaking nothing new** - a source, not a second domain - because a
third catalogue entry that looked like a third domain would be the exact false confidence R17
exists to refuse. **It may bear on D43:** ocean products are often chunked more kindly than
WeatherBench's one-timestep-deep layouts, so a laptop-feasible long regional record may exist here.
Recorded as a possibility to measure, not a promise.

#### What the probing established, and what is left

Measurements and their claim boundary are in `VERIFICATION.md` under *TG12.1 - Ocean store probe*.
In summary:

*   **GLORYS is the only candidate that fits the existing machinery.** OISST is per-day netCDF and
    ECCO is netCDF granules; `probe_store` opens Zarr. Only the Copernicus ARCO stores are Zarr, and
    they probe **anonymously** - the `access="credentials"` assumption this slice was planned around
    is wrong.
*   **The D43 possibility holds, but only in one of two layouts.** The persisted exact probe now
    costs `geoChunked.zarr` at **2.02x** for the sealed three-year regional crop against
    **72.52x** for `timeChunked.zarr` and 26.2x for the ERA5
    store D43 is open against. The asset names describe the narrowly chunked dimension rather
    than the access pattern they make cheap, so choosing by name alone registers the hostile
    store and records a false negative against D43.
*   **The vertical axis is `elevation`, not `depth`** - fractional negative metres, 50 values.
    `KNOWN_VERTICAL_DIMENSIONS`, `CropSpec` and the strict HTTP request now express it without
    moving an integer ERA5 identity (D68).
*   **D67 was found and closed here.** `assess_access_pattern` no longer constructs a dask data
    graph: shared xarray indexers operate on coordinates and exact source chunk ids are counted.
    The Acquire tab's **Inspect** path therefore costs GLORYS before a transfer as intended.

**Delivered.** Both live anonymous metadata probes are checked in under `data/store_probes`:
`ccdb0625e7e8fb1d` for the selected `geoChunked` layout and `f9a45764fcf52c2a` for the rejected
hostile layout. `src/data_layer/glorys_store.py` loads those records without network access and
registers `glorys_phy_my_0p083deg_p1d`, citing the selected digest. Acquire reads the extension's
exact surface-elevation crop defaults when the store changes. No module under `src/` imports
`copernicusmarine`; the isolated environment remains a URI-discovery tool, not a runtime
dependency.

**Verified.** Live D67 acceptance completed against both 12227 x 50 x 2041 x 4320 assets without
`MemoryError`; the selected layout returned 2.02x and the rejected layout 72.52x. Focused backend,
frontend-contract and documentation verification is recorded in `VERIFICATION.md`; the production
TypeScript/Vite build emitted 1,395 modules and real JS/CSS. No ocean field value was fetched.

**Environment already prepared.** `requirements-ocean.txt` adds `s3fs` and `earthaccess` to the main
venv as pure additions. `copernicusmarine` **cannot** go there - it requires pydantic >= 2.9.1 and
this project pins `pydantic < 2.0.0` for FastAPI 0.110, a `ResolutionImpossible` checked rather than
assumed. It lives in an isolated `.venv-copernicus` and is invoked as an external command, never
imported, which is the honest shape for a credential-minting tool anyway.

**TG12.1a D70 — readiness that declares what it is about. DONE (2026-08-28, `ed-dev`).** A
follow-on found by using TG12.1 rather than by testing it: GLORYS was selected in Acquire, Probe
and Inspect both behaved correctly, and the question was what to do next. The honest answer is
nothing — materialisation is CLI-only by design, and TG12.1's claim boundary says no analysis
consumes an ocean crop — but the workbench said something wrong rather than saying nothing. The
T5.2 readiness line rendered against every materialised crop truncated GLORYS's fractional
elevations with `int()` (`int(-0.494…)` is `0`, silently) and then reported the five ERA5
variables as *missing*, which reads as a crop that nearly qualified. Levels are now compared as
numbers; `applicable` is derived from the crop's own declared vertical axis; and Inspect states
**before** a 51 GB transfer that materialising is where this store currently stops. ERA5 fields
are unchanged, asserted. Recorded in `architecture.md` §3.6zpb and `VERIFICATION.md`.

**TG12.1b D72 — the generated materialisation command uses the store it names. DONE
(2026-08-29, `ed-dev`).** The Inspect panel emitted a fractional GLORYS elevation that the CLI
then parsed with `int()`, and the CLI constructed `CropSpec` directly so every store inherited
ERA5's `level` axis. Levels now parse integer-first — preserving every integer ERA5 content key
while retaining exact fractional ocean coordinates — and the command enters through
`crop_for_store`, the same registry-aware constructor as the HTTP path. An invalid level is
refused by name. Focused Zarr/documentation acceptance passes; no value transfer is claimed.

**TG12.1c D71 — portable immutable evidence publication. DONE (2026-08-29, `ed-dev`).** Pinning
pytest to the deployed `D:` volume exposed an unconditional hard-link assumption in the ERA5
overlap receipt: exFAT supports atomic rename but not hard links. Five local implementations of
the same scientific boundary are now one `core/publication.py` primitive. It flushes complete
same-directory bytes and publishes with an OS atomic no-replace operation; it never falls back to
check-then-rename or overwrite. The acceptance race launches eight spawned publishers on `D:`,
gets exactly one complete winner and seven refusals, preserves an existing target byte-for-byte,
and cleans up an injected unsupported-filesystem failure. The two CDS cases that found D71 now
pass on exFAT. This establishes process-crash atomicity, not a claim about sudden-power-loss
durability of every storage device.

**TG12.1d D73 — transform-derived acquisition planning. DONE (2026-08-29, `ed-dev`).** A
researcher should not learn that a crop has no defensible transform interior after paying for it.
The store probe remains a source observation; a separate immutable plan now combines its
metadata with the exact transform family, filters and depth. SWT and DTCWT publish their own
support callbacks through the transform registry, so the acquisition layer contains no copied
filter length and an external transform can join without a planner edit.

The plan reports an absolute technical minimum and a named R13 recommended minimum, each derived
from per-level valid interiors. It expands native coordinate indices symmetrically, shifts at
edges, states when the source is too small, and re-prices the proposed bounds against actual
chunks before transfer. Acquire renders all of this, exposes the exact configuration, and applies
the recommended bounds in one action. The CLI preserves the same configuration—including the
analysis depth that D73 found it had dropped—and explicit materialisation refuses below the
recommended threshold before constructing a field selection. Four-level DTCWT currently derives
240 x 240 absolute and 512 x 512 recommended; SWT/db2 derives 47 x 47 and 256 x 256. The
recommendation is filter support plus a declared 128-parent-cell statistical-span policy, not a
claim of power or discovery. No public value was transferred. Focused backend and production
build acceptance pass; rendered browser inspection remains NOT RUN because no browser was
available in the in-app runtime. Recorded in `architecture.md` §3.6zpc and `VERIFICATION.md`.

**Where this leaves G12.** GLORYS delivered the store seam, the coordinate-only cost estimator
(D67) and a vertical axis the request layer can actually express (D68) — and no analysis route,
because R17 admitted it as a *source* under `reanalysis`, not as a second domain. The analysis
route is TG12.2's to build, which is why that slice carries more weight than its position
suggests: it is the first end-to-end non-atmospheric path, archive to falsification layer.

**TG12.2 Argo profiles — a second acquisition shape, and the reduction nobody had named.**
The slice where `argo_float` stops being a declaration. It is planned in four parts because
writing the design down turned up a prerequisite that has to land first, and a decision that was
posed as a fork with two branches when the honest answer is neither of them.

##### The decision, taken in the open

The earlier note said: *either the profile shape supplies those axes or the declaration is
split.* Having costed both, **neither is taken, and the reason is worth stating because it is the
same reason D61 exists.**

Start from what E14 is actually saying. `assert_domain_admits_channel_table` compares declared
axis *roles* against `CHANNEL_TABLE_ROLES = ("time", "category")`. `argo_float` declares
`latitude` and `longitude` as `space` and `pressure` as `level`, so three of its five axes are
unsatisfiable. The refusal is not "you forgot some columns". It is *"you told me your results are
indexed by position and depth, this container cannot index by them, and reading here would
silently change what a result means."*

So the question E14 is really asking is: **is an Argo analysis result indexed by latitude,
longitude and pressure?** The answer depends entirely on a step the platform has never named —
the **reduction** from a scatter of profiles to labelled scalar series. Three defensible
reductions of the same query:

| reduction | channel label | what happens to the other axes |
|---|---|---|
| **per-float scalar** — e.g. temperature at 100 dbar, or 0–200 m heat content | `float_id` | `pressure` consumed by the reduction; `latitude`/`longitude` become per-sample attributes, not indices |
| **per-depth-bin array mean** | `pressure` bin (ordered) | `float_id` and position averaged over; creates `aggregated_values` |
| **per-region mean** | region or grid cell | `float_id` and `pressure` averaged over |

In every case the surviving index is `(time, channel)`. That is not a loophole around E14 — the
reduction is *where the scientific content of the acquisition lives*, and until now it has had no
name, no registration and no content key. Those three rows are different scientific records
produced from the same bytes, and nothing in the platform could currently tell them apart in a
provenance record.

**Therefore:** `ProfileSpec` supplies the axes *at the acquisition layer* — the first branch is
right about acquisition — and a **declared, registered reduction** consumes them on the way to a
`ChannelSeries`, deriving the channel-series declaration from the parent rather than having a
second one hand-written beside it, which is what the second branch would have produced.

**Why derived and not hand-split.** A hand-written second declaration for "Argo as a flat series"
would be a second description of one archive, maintained by hand, free to drift from the first.
That is precisely D61: `order_book`'s description and its violation tuple disagreed for four
slices because both were written by hand and nothing computed one from the other. Repeating that
shape in the slice whose whole purpose is to make a declaration executable would be a poor joke.
A derived declaration cannot drift, because it is computed from the parent declaration and the
reduction's declared axis consumption.

**E14's refusal stays, and gets a test.** `argo_float` continues to be refused a flat channel
table, by name, and TG12.2 asserts that it still is. What changes is that there is now a
legitimate route to a `ChannelSeries` that goes *through* a declared reduction rather than around
the check.

##### The finding that reorders the work: D69

The justification for Argo, after D61 shrank it, is that it is the only domain breaking
**`non_stationary_support`** — *"channels start and stop during the record, so the effective
sample size differs per channel and per pair."* Floats are deployed, drift and die mid-record;
that is the violation, exactly.

Grepping `src/` for consumers of that violation returns **one hit, and it is the dictionary entry
that defines it.** `irregular_sampling` is enforced in three places — `_validate_cadence` refuses
a ragged clock without it, and `DomainTimeSeries.cadence_seconds` refuses a frame-to-duration
conversion with it. `aggregated_values` is enforced in both directions: declare it and you must
supply a physical window, omit it and you may not carry one. `non_stationary_support` is enforced
**nowhere**. `refusals_for` displays its consequence text, and its own docstring says *"that
refusal is enforced deep in the analysis layer"* — which for this violation is not true.

It is worse than an unenforced rule, because the container cannot express the condition either.
`ChannelSeries.usable` is `Optional[Sequence[bool]]` with **one entry per channel**, and
`unusable_reason` is keyed by channel label. There is no per-*sample* presence mask anywhere in
the contract. A float that reported for two years of a five-year record is either wholly usable
or wholly unusable; "present here, absent there" is not sayable. `measures` finiteness is never
validated, so the obvious workaround — a union clock with `NaN` where a float did not surface —
would push `NaN` into an analysis layer that has never been asked what it does with one.

So the missing enforcement and the missing representation are the same hole, and Argo cannot be
delivered honestly on top of it: the acceptance criterion would reduce to *"a record whose
declared violation nothing acts on."* **Logged as D69 and closed first.**

##### The parts

**TG12.2a — Close D69: per-sample presence in the channel-series contract. DONE
(2026-08-29, `ed-dev`).** The prerequisite,
planned in full here because it changes `ChannelSeries`, which is the one interface the accepted
falsification layer consumes. Everything downstream of it inherits whatever this slice gets wrong.

##### What actually happens today, measured rather than assumed

The tempting implementation is a union clock padded with `NaN`. Before designing anything, the
question *"what does the analysis layer do with an absent sample?"* was put to the code. It does
not raise. It returns numbers. Eight sites, each cited, each a distinct failure:

**F1 — The estimators already mask, so nothing announces itself.** `mutual_information`
(`cross_scale.py:370`) and `transfer_entropy` (`cross_scale.py:409`) each build a `finite` mask
over their operands and compute on the intersection, returning `NaN` only below four and six
finite samples respectively. A padded record therefore produces a complete, plausible,
fully-populated result table. There is no error to notice.

**F2 — Reported N is the padded length, and it drives the bias guard.** `per_cell = n_times /
cells` at `cross_scale.py:573` takes `n_times` from `matrix.shape[0]`. That is the guard against
an estimate dominated by its own bias — the `MIN_SAMPLES_PER_CELL` warning — and it would be
computed from the padded clock while the estimate itself used the pairwise intersection. A pair
overlapping in 40 of 100 frames gets the reassurance arithmetic of 100. `n_frames` in the returned
record is padded for the same reason.

**F3 — The Theiler window is computed on a series with its gaps closed up.**
`decorrelation_frames` (`cross_scale.py:453`) does `a = a[np.isfinite(a)]` and then takes the
autocorrelation over *adjacent entries of the compacted array*. Samples either side of a two-year
gap are treated as neighbours. This understates the decorrelation length, which understates the
Theiler window, which makes `admissible_shifts` admit shifts that are not real shuffles. The
direction of the error is **toward false significance**, which is the direction that matters.

**F4 — The surrogate ensemble rotates the presence pattern.** `_shift_null` (`cross_scale.py:520`)
calls `np.roll(source, shift)` on the padded column, so the `NaN`s travel with the values. Each
surrogate therefore overlaps the target in a *different* number of samples than the observed pair
did: an observed statistic computed on 40 samples is referred to a null whose members were
computed on anywhere between none and seventy. Entropy-estimator bias is a strong function of N,
so this is not the observed statistic's null. The direction of the error is **unsigned and
data-dependent**, which is worse than a bias with a known sign.

**F5 — `admissible_shifts` counts padded frames.** Its `n` (`cross_scale.py:489`) is the padded
length, so both exclusion windows, and the refusal that fires when no shift avoids them, are
computed against a record longer than either channel really has.

**F6 — `wrap=True` joins the ends across gaps.** `np.roll(t, -lag)` treats the record as
circular. With gaps, a lag of *k* frames is not *k* frames of anything for a partially-present
channel, and `wrap_fraction` (`cross_scale.py:658`) is measured against the padded `n`.

**F7 — The split copies `usable` wholesale.** `split_channel_series` slices `measures` and
`times_seconds` but passes `usable`, `support_parent_px` and `unusable_reason` through unchanged.
A float deployed after the train/test boundary is wholly absent from the training partition while
still marked usable, and `train_stop = int(n * train_ratio)` is a padded-frame index, so effective
N per channel can differ arbitrarily between the two sides. R6's protection assumes partitions
that are comparable.

**F8 — The partition identity cannot see presence, which reopens D65.** `_identity`
(`preregistration.py:229`) builds a `PartitionIdentity` from `n_times = stop - start`,
`n_channels`, `channel_labels` and the identifying provenance keys. Two records with the same
clock and the same labels but *different presence* hash identically, so `HeldOutLedger` — which is
keyed on the partition precisely so that a second honest seal cannot buy a second look — would not
fire between them. D65 fixed this class of defect by building the identity from what identifies
the data; a new field that identifies the data must enter it, or the fix regresses.

**The conclusion these eight support.** Padding with `NaN` and proceeding does not fail loudly, or
even quietly. It produces a full result table of numbers computed on unrecorded and varying sample
sizes, referred to a null whose bias differs from the observed statistic's, guarded by a
bias-warning computed from a length nothing used, with a Theiler window derived from an
autocorrelation that treats a two-year gap as one frame. That is the precise machine for
generating confident noise, and it is the reason D69 is closed before any Argo bytes are fetched
rather than after.

##### The five decisions, taken here

**A-D1 — Presence is declared, never inferred from `NaN`.** Deriving the mask from
`~np.isfinite(measure)` would be one line and is refused for three reasons. A `NaN` in a measure
already means something else — a value that was observed and failed QC — and conflating "not
observed" with "observed and invalid" destroys exactly the distinction `usable` and R13 exist to
preserve. Inferred presence would also depend on *which measure* you looked at, when presence is a
property of the sample and shared by all of them. And E14's standing position is that meaning is
declared and never read off an appearance; a mask inferred from a value pattern is the same
mistake as a role inferred from an axis being called `lat`.

**A-D2 — The mask is one `(time, channel)` boolean array per series, not one per measure.** A
float either surfaced on a cycle or it did not; every measure derived from that ascent shares the
fact. A measure that is `NaN` where presence is `True` keeps its current meaning — a value that
exists and is not usable — and the two conditions stay distinguishable in the receipt.

**A-D3 — Enforced in both directions, mirroring `aggregated_values` exactly.** A series whose
domain declares `non_stationary_support` must supply a mask; a series whose domain does not
declare it may not carry one. The precedent is `DomainTimeSeries.__post_init__`
(`cross_domain.py:156-168`), which requires a positive physical window when `aggregated_values` is
declared and refuses one when it is not. Copying an enforcement shape that already exists is
worth more than inventing a second one, and it means the vocabulary's three data-shape violations
are finally enforced alike.

**A-D4 — Effective N is a pairwise quantity, computed from mask overlap, reported per test.** The
violation's own consequence text says the effective sample size differs *"per channel and per
pair"*, so a single per-series number would restate the defect. Every test in the family carries
the overlap count it was actually computed on, and the bias guard of F2 is computed from that
number rather than from the clock length.

**A-D5 — The surrogate null is computed at fixed N, on the pairwise-present subsequence.** The
overlap is taken *first*, then shifts are drawn within it, so every surrogate is computed on
exactly the observed sample size and F4 closes. This has a consequence that must be stated rather
than absorbed: the overlap subsequence is **not contiguous in clock time**, so a shift of *k*
positions within it is not a lag of *k* frames of anything.

That last point was the one genuinely open sub-decision in TG12.2a, and was measured rather
than guessed:

*   **Candidate 1 — run within maximal contiguous presence runs.** Admissible lags are evaluated
    inside each maximal run of joint presence; a pair whose longest run does not support
    `lag + theiler` is excluded by name, with the run lengths in the exclusion reason. Honest about
    clock time; costs statistical power, possibly all of it, on a sparse float array.
*   **Candidate 2 — refuse frame lags outright for a masked series.** A domain declaring
    `non_stationary_support` alongside `irregular_sampling` arguably has no business reporting a
    lag in frames at all — `DomainTimeSeries.cadence_seconds` already refuses the frame-to-duration
    conversion for the second violation. This would make Argo an association-only domain until a
    physical-time estimator exists.

**Measured outcome: Candidate 2.** The controlled Argo-like clock contains 20 floats, 146
ten-day cycles and a distinct 12-hour surfacing offset for each float. Its union has 2,920 rows
and 380 ordered pairs. Pairwise overlap is zero throughout: no pair has one simultaneous sample,
let alone the six required for transfer entropy plus a positive lag and Theiler exclusions.
Candidate 1 therefore cannot recover an injected lag without first inventing simultaneity by
binning or interpolation. The instrument refuses every frame-lag sweep on an intermittently
present mask until a physical-time estimator exists. The complete measurement and refusal
context are captured in `VERIFICATION.md`.

##### The work, in order

**A1 — The contract.** `ChannelSeries.present: Optional[np.ndarray]`, boolean, shape
`(n_times, n_channels)`, validated against the clock and channel count with the existing
`ShapeMismatchError` shape. A channel with fewer than two present samples is refused by name — one
sample has no clock and no lag can be taken along it, which is the refusal `times_seconds` already
makes for the series as a whole. `channel_records` gains a per-channel present-count so a receipt
shows what each channel contributed. `ChannelGeometrySpec` is untouched: it is geometry without
data, and presence is data.

**A2 — The enforcement.** Both directions per A-D3, at the point where a series meets its
declaration — `read_channels_for_domain` and the domain-analysis entry, not in `ChannelSeries`
itself, which does not hold a declaration. `refusals_for` keeps publishing the consequence text,
and its docstring's claim that the refusal is enforced in the analysis layer becomes true.
**Acceptance:** a domain declaring the violation with no mask is refused, by name, naming the
violation; a domain not declaring it that supplies a mask is refused, by name; and the existing
`aggregated_values` refusals are shown to be unchanged.

**A3 — Effective N and the bias guard.** Pairwise overlap replaces `n_times` in `per_cell`, in
the `MIN_SAMPLES_PER_CELL` warning text, and in a new per-test `n_effective` field. `n_frames`
stays in the result as the clock length, under a name that says so, so nothing that reads it today
changes meaning.
**Acceptance:** two channels overlapping in 40 of 100 frames report `n_effective` 40; the bias
warning fires on the pair's own arithmetic, not the record's; a series with no mask reports
`n_effective` equal to `n_frames` for every pair, which is what makes every existing result
identical.

**A4 — Gap-correct decorrelation.** `decorrelation_frames` stops compacting. Given a presence
mask it computes the autocorrelation over pairs that are genuinely `lag` frames apart *and* both
present, and refuses rather than guesses when too few such pairs exist at every candidate lag.
Without a mask its behaviour is byte-identical, including the existing `a[np.isfinite(a)]` line
for genuinely-NaN atmospheric input.
**Acceptance:** a series with a gap reports a decorrelation length matching the same series with
the gap absent from the clock entirely, rather than the shorter one compaction produces; a
regression pins the atmospheric value.

**A5 — The null, and the lag decision.** `_shift_null` and `admissible_shifts` operate on the
pairwise-present subsequence at fixed N, and the F5/F6 padded counts follow it. The A-D5
measurement runs here and its outcome is implemented.
**Acceptance:** every surrogate in an ensemble is computed on the same number of samples as the
observed statistic, asserted directly; `wrap_fraction` and the shift-exclusion windows are
computed from the effective length; and the adopted lag candidate recovers a known injected
coupling on the synthetic float array, or the domain refuses frame lags and says so.

**A6 — Splits and identity.** `split_channel_series` slices the mask with the measures, recomputes
per-partition usability, and refuses — rather than silently returning a degenerate partition — when
a channel is wholly absent from one side. `_identity` takes the presence pattern into the
`PartitionIdentity` digest so F8's collision cannot occur.
**Acceptance:** a float present only after the boundary is refused by name at the split rather than
appearing as a usable training channel; two records identical but for presence produce different
partition digests, asserted by the same test shape D65 introduced.

**A7 — Surfacing, and the byte-identity proof.** Presence counts reach the receipt, the
`ChannelRecords` API response and the Acquire tab's channel table, because a per-channel sample
count that only exists inside the estimator is a fact the reader cannot check.
**Acceptance:** the full suite passes with every pinned hash unmoved, and every checked-in
preregistration and evidence bundle loads. No built-in domain declares `non_stationary_support`
today, so every mask is `None` and every existing path must be byte-identical; that is the
strongest available proof that the change is additive, and it is the acceptance criterion rather
than a hope.

##### Edge-case register

Worked through explicitly, so implementation meets them as decided cases rather than as surprises.

| # | Case | Required behaviour |
|---|---|---|
| E1 | A channel present for fewer than two samples | Refused by name at construction (A1) |
| E2 | A channel wholly absent from a partition after splitting | Refused by name at the split, not returned degenerate (A6) |
| E3 | A pair whose presence never overlaps | Excluded from the family with that reason, and *counted* in `n_excluded`, never silently dropped |
| E4 | A pair overlapping in fewer samples than the estimator's own floor (4 for MI, 6 for TE) | Excluded by name, distinguishing "too little overlap" from today's "estimator was not finite" |
| E5 | `NaN` in a measure where presence is `True` | Keeps its existing meaning — observed and invalid — and stays distinguishable from absence in the receipt |
| E6 | Presence `False` where the measure is finite | Refused: a value that was not observed cannot have a number, and permitting it makes the mask advisory |
| E7 | Every channel present everywhere | Mask is all-`True`; results must equal the no-mask case exactly, asserted |
| E8 | A domain declaring the violation on a record that turns out to be fully present | Permitted with a recorded note; the declaration is about the source, not about one query's luck |
| E9 | `wrap=True` on a masked series | Follows the A-D5 outcome; must not silently wrap across a gap under either candidate |
| E10 | The embargo window falls entirely inside a gap | The embargo is still real in clock time; recorded as such, not counted as effective frames |
| E11 | Presence differs between two measures of one series | Impossible by A-D2; the mask is per series, and a caller supplying per-measure presence is refused |
| E12 | An existing series constructed with no mask | Every downstream number byte-identical, which is A7's acceptance |
| E13 | A masked series reaching a path that has not been audited | Refuses by name rather than proceeding; the audit list is F1–F8 and anything outside it must say so |
| E14 | Presence supplied as an integer or float array | Refused; a boolean mask that accepts `0.5` is not a mask |
| E15 | `support_parent_px` on a channel present in only part of the record | Unchanged — the footprint is a property of the representation, not of the sampling — but recorded alongside the present-count so a reader can see both |

##### What TG12.2a does not do

**Acceptance met.** The contract distinguishes absence from observed-invalid values, binds the
mask in both declaration directions, recomputes partition viability, hashes presence into held-out
identity without reading measures, reports present counts through lineage/API/UI, measures
decorrelation only from genuinely lag-separated pairs and fixes the audit null's N before any
shift. A partial mask refuses before any lag statistic is produced; an all-true mask returns the
same dependency result as the no-mask path. The focused presence suite passes, the expanded
cross-domain/API regression passes, and the full-suite result is recorded in `VERIFICATION.md`.

It does not fetch Argo, define `ProfileSpec`, or register a reduction. It does not claim that a
masked series is *analysable* — A-D5's measurement may conclude that frame lags are inadmissible
for such a series, and that refusal would itself be the deliverable. And it does not touch
`ScaleSignature`: the atmospheric path has no absent samples, declares no such violation, and must
end this slice byte-identical.

**TG12.2b — `ProfileSpec` and the profile collection. DONE (2026-08-29, `ed-dev`).** A region, a time window and a depth
range, content-addressed and machine-independent like `CropSpec` and — as previously noted —
unable to borrow it, because the result is a scatter rather than an array. It returns a
`ProfileCollection` genuinely carrying per-profile time, latitude, longitude and a pressure
vector, so the three axes `argo_float` declares are real objects rather than a claim.
**Acceptance:** `assert_domain_admits_channel_table(argo_float)` still raises, by name, naming
all three unsatisfiable axes; and an identical `ProfileSpec` produces an identical content key on
a second machine.

**TG12.2c — The declared reduction. DONE (2026-08-29, `ed-dev`).** `ProfileReduction`: named, registered, carrying its own
content key, and declaring which parent axes it consumes. The channel-series declaration is
**derived** from the parent declaration and that consumption record. At least two reductions are
registered, because a single one is indistinguishable from a hardcoded path and would not
demonstrate that the concept exists.

This is where the slice's sharpest scientific point lands, and it is recorded here rather than
discovered later: **the choice of reduction determines whether Argo still delivers the violation
it was justified by.** The per-float reduction preserves `non_stationary_support` exactly — floats
really do start and stop. Depth-bin and regional averaging *average the non-stationarity away* and
add `aggregated_values` instead. So the per-float reduction is not one option among three; it is
the one TG12.2 is obliged to deliver, and the derived declaration must be shown to carry the
violation through rather than quietly dropping it.

The remaining sub-decision is the shared clock: floats do not surface together, and
`ChannelSeries` requires one strictly increasing clock with full `(time, channel)` matrices. The
union clock plus TG12.2a's presence mask is the intended answer — it is the *representation of*
the violation rather than a workaround for it — but what the analysis layer does with an absent
sample is to be **measured and recorded, not assumed**. Binning onto a common grid is the
rejected alternative, and the rejection is recorded: it would create `aggregated_values` and
destroy the irregular clock that was half the point of the domain.
**Acceptance:** the derived declaration for the per-float reduction carries `irregular_sampling`
and `non_stationary_support` and does *not* carry `aggregated_values`; the depth-bin reduction's
derived declaration carries `aggregated_values` and a physical window; and two reductions of one
`ProfileCollection` produce two different content keys.

**TG12.2d — Real data, end to end. DONE (2026-08-29, `ed-dev`).** A live Argo GDAC query through the acquisition seam, and
`profile_query` surfaced in the Acquire tab beside `grid_crop`. `argo_float` is declared today in
`src/tests/domain_plugin_example.py`, which an acquisition path may not import; it moves to an
extension module registered the way `glorys_store.py` is, preserving TG8.1's
declared-from-outside-`src/` acceptance rather than discarding it.
**Acceptance:** a real Argo query produces a record whose irregular clock is reported as
irregular, under a domain declaring `irregular_sampling` — the first time that refusal fires
against data rather than a fixture — **and** whose per-channel presence differs across floats,
under a domain declaring `non_stationary_support`, which after TG12.2a is a refusal with
something behind it.

**Acceptance met and rechecked after the interrupted handoff.** The bounded live New Zealand
query passed with explicit network opt-in on 2026-08-29, produced at least two floats, unequal
per-float sample counts and a partial presence mask, and retained both declared violations. The
focused presence/profile/acquisition regressions and the production frontend build pass. The
earlier verification entry's claimed full-suite arithmetic was not accepted as evidence: it
listed two skips although the new live Argo test is a third opt-in skip. That record is corrected
in `VERIFICATION.md` rather than repeated here.

##### Claim boundary, declared in advance

TG12.2 delivers an acquisition, a reduction and two enforced violations. It does **not** claim an
oceanographic result. A reduction being registered and content-keyed says the choice was recorded
and is reproducible; it does not say the choice was right for any particular question, and no
cross-scale finding is claimed from Argo in this slice.

#### Phase G13 — The sky, and closing the vocabulary

**TG13.1 Photometry, and the last unbroken assumption. IMPLEMENTED; LIVE MAST ACCEPTANCE OPEN
(2026-08-29, `ed-dev`).** TESS light curves via the official MAST API.
Breaks `no_natural_cycle`: there is no diurnal or annual forcing, so R11's harmonic climatology
has nothing to remove and its default periods would fit noise. This is the only domain that
exercises that refusal. It also breaks `irregular_sampling` and `non_stationary_support` through
sector gaps and targets entering and leaving, and breaks the metric assumption *differently* from
`order_book` — angular separation is a real metric that is not a length in metres. A third
acquisition shape: per-target, sector-based.

**Delivered.** An exact TIC and bounded sector family now enter a metadata-only preflight before
any value transfer. Product and byte caps, archive URI, filename, sector and declared size are
sealed into the plan. Acquisition reads checksum-valid calibrated SPOC LC FITS, validates TIC,
BJD_TDB and ICRS position, retains quality flags, hashes every source byte and every admitted
sample, and publishes canonical no-overwrite collection bytes. The `angular_sky` point geometry
records great-circle degrees without entering raster geometry recognition. API and Acquire UI
surfaces expose the refusal boundary. Synthetic FITS acceptance and collision mutations pass;
the live MAST metadata service exceeded its 45-second bound during this slice, so live acceptance
is explicitly OPEN rather than converted into an empty result or a completion claim.

**TG13.2 The coverage claim, checked rather than asserted. DONE (2026-08-29, `ed-dev`).** A test that every entry in
`KNOWN_VIOLATIONS` is broken by at least one registered domain **with a real data path behind
it**, not merely declared. The claim that the abstraction generalises then rests on something
mechanical rather than on this document.

`GET /api/v1/acquisitions` now emits `violation_coverage` from registered declarations joined to
available acquisition paths. The test requires every `KNOWN_VIOLATIONS` entry to have a nonempty
path and pins `no_natural_cycle` specifically to `lightcurve_query`; a declaration without a path
cannot satisfy it.

#### Phase G14 — File-first scientific ingress

**TG14.1 Explicit sample tables. DONE (2026-08-29, `ed-dev`).** Acquire begins with **Load a
file. Let's analyse it.** A bounded CSV/TSV probe reports only storage facts. The researcher must
declare every role, units and whether rows are independent, grouped or ordered. Independent rows
enter a new sample-table contract and are never assigned a fake clock to fit `ChannelSeries`.

**TG14.2 Representation Audit, first safe recipe. DONE (2026-08-29, `ed-dev`).** The file digest,
declaration, complete raw/PCA candidate family, PCA component count, estimator bins, split, seed,
alpha, BY correction and permutation ensemble are frozen before enumeration. The permutation
resolution must be capable of surviving correction or the plan refuses. PCA is fit on generate
only; every candidate is measured again on a separately held-out confirmation partition and the
complete family is corrected on each side. Nuisance-median strata are learned on generate and
reported descriptively on confirm, explicitly not as conditional MI. Results are candidate
representations, never certified invariants, causes or instructions to use a feature.

**Next bounded extensions.** Parquet is the next sample-table adapter; NetCDF/Zarr continue through
their typed gridded route rather than pretending every file is a table. Group-aware and blocked
resampling remain separate benchmarked recipes. Conditional information and bounded stable
subspaces are specified in G16 below; their roadmap presence is not an implemented capability or
a UI label over mathematics that does not yet exist.

#### Phase G15 — Dataset capability routing

**TG15.1 Content-bound capability profiles. DONE (2026-08-30, `ed-dev`).** A central operation
registry derives scientific paths from the exact declared representation rather than source names
or UI branches. Declared sample tables, admitted channel records, planned grid crops, acquired
Argo reductions and acquired TESS collections emit a hashed profile with tri-state facts,
operation status, stable reason code and explanation. Domain capability is only an upper bound;
record cadence and transform support can narrow it.

**TG15.2 Explained shell gating. DONE (2026-08-30, `ed-dev`).** The application shell owns the
selected profile. Downstream navigation reads its operation decisions: inadmissible gridded,
lagged or record-specific paths remain visible but disabled with the backend's reason. The same
profile renders the fact matrix in Acquire/current context. Backend entry points remain
authoritative refusals rather than trusting browser state.

**Safety correction delivered with G15.** The first Representation Audit had accepted grouped
and ordered declarations while using a row-random split. It now plans only independent samples.
Grouped rows explicitly require group-held-out confirmation; ordered rows require blocked and
embargoed confirmation. Neither dependency structure is silently broken to make the recipe run.

#### Phase G16 — Representation Structure — **COMPLETE**

G14 finds raw or PCA candidates associated with a declared target and confirms them on a held-out
partition. G16 asks the next bounded questions: whether candidates carry duplicate, complementary
or unresolved information; whether a relationship remains after conditioning on a declared
nuisance; and whether a small, frozen linear subspace retains its relationship across held-out
samples and nuisance regions. This is a representation-structure programme, not automatic feature
selection. Nothing in it deletes a column or instructs a researcher which representation to use.

**TG16.0 Admission and benchmark prerequisites. DONE (2026-08-30, `ed-dev`).** The first G16 recipes admit independent
samples only. Grouped data remain unavailable until group-held-out nulls and confirmation exist;
ordered data remain unavailable until blocked, embargoed equivalents exist. Before an operation
enters the capability registry, paired known-answer benchmarks must cover exact duplicates,
independent features, redundant noisy copies, complementary information, a synergistic pair that
is weak individually, nuisance-only association, signal that survives conditioning, a null that
does not, and a collider counterexample. Calibration on nulls and stated power on planted effects
are acceptance criteria, not follow-up polish.

**Delivered.** `require_independent_samples` is the one refusal used by the existing file-first
recipe and reserved for the G16 recipes: grouped rows name group-held-out confirmation with
benchmarked nulls, and ordered rows name blocked/embargoed confirmation with benchmarked nulls.
`representation_structure.py` registers a planted/safeguard pair covering all nine cases above
with separately derived random streams and independent construction oracles. The future
operation-level acceptance family is frozen at 200 replications, alpha 0.05, null rejection rate
at most 0.075 and planted detection power at least 0.80. The benchmark suite is 31 PASS, 0 FAIL,
0 NOT_YET_RUNNABLE. This slice deliberately exposes no G16 operation; the premature
`association_redundancy` G15 registry entry was removed. TG16.1 must earn its registry entry by
running these cases, not merely by consuming their arrays.

**TG16.1 Redundancy Structure Audit. DONE (2026-08-30, `ed-dev`).** Freeze the complete candidate pair/group family,
estimator, discretisation or neighbourhood policy, null, seeds, alpha and correction before
enumeration. Report a structure map whose outcomes distinguish supported redundancy,
complementarity and unresolved relationships. Pairwise association alone must not be described as
the number of independent pieces of information, and a synergistic pair must not be discarded
because neither member ranks highly alone. Output is a **candidate redundancy structure**, never
an instruction to remove features.

**Delivered.** The first bounded recipe freezes every unordered pair from two to six declared raw
features, three hypotheses per pair, equiprobable discretisation with Miller-Madow correction,
the two conditional-permutation nulls, seed, alpha and Benjamini-Yekutieli family correction before
opening a result. Positive calibrated interaction information supports redundancy; two corrected
conditional increments support complementarity; everything else remains unresolved. Exact
candidate identity is separately visible structural evidence. The joint estimator therefore keeps
the XOR pair as supported complementarity even though both singleton mutual informations are weak.
Responses explicitly deny a partial-information decomposition, a count of independent information
pieces, and any instruction to remove or select a feature. The operation is now registered as
`redundancy_structure_audit` and exposed through content-bound plan/run endpoints.

**Acceptance met.** Both G16 known-answer benchmarks now carry `G16.1.redundancy_structure`.
Across the frozen 200-replication family, exact-duplicate and noisy-copy redundancy power is 1.00
and 0.94, complementary and XOR power is 1.00 each, and the independent false-claim rate is 0.02.
All four one-candidate conditioning cases correctly produce no pair claim. The focused gate is 4
PASS, 0 FAIL and 0 NOT_YET_RUNNABLE; the complete benchmark registry is now 33 checks.

**TG16.2 Conditional-Information Audit. DONE (2026-08-30, `ed-dev`).** Estimate the explicitly named quantity
`I(candidate; target | declared nuisance)` only where overlap and effective support meet a frozen
admission rule. A global target permutation is not a conditional-independence null because it also
destroys the target/nuisance relationship. Each estimator therefore brings a benchmarked
conditional null — for example justified within-stratum/local permutation or a sealed conditional
randomisation model — plus calibration and failure conditions. `nuisance` is a researcher-declared
statistical role, not proof that the variable is a confounder: responses say **conditional
association**, never “confounding removed”, “nuisance-free” or causal. Collider and post-treatment
interpretations remain outside what the computation can decide.

**Delivered.** The bounded recipe requires one researcher-declared nuisance and tests every one of
one to six declared raw features. Its content-bound plan freezes equiprobable bins, Miller-Madow
conditional mutual information, the complete candidate family, a five-rows-per-cell
overlap/effective-support rule, a linear conditional-randomisation model, permutations, seed,
alpha and global Benjamini-Yekutieli correction. The null fits target on the declared nuisance,
permutes residuals, reconstructs and rediscretises the target, preserving the fitted
target/nuisance relationship that a global target shuffle would destroy. Inadequate within-region
levels, effective joint support, absolute quadratic residual correlation above 0.20, a
nuisance-stratum residual-variance ratio above 4, inadequate permutation resolution, missing
nuisance roles and dependent rows all refuse before a result.

**Acceptance met.** Both paired G16 benchmarks now carry `G16.2.conditional_information`. Across
200 frozen replications, support admission was 1.00 for every applicable case, signal-survival
power and collider conditional-association detection were 1.00, and nuisance-only/conditional-null
false-claim rates were 0.055/0.045, below the 0.075 ceiling. The collider result is deliberately
reported as conditional association while its causal interpretation remains outside the
computation. The focused paired gate is 6 PASS, 0 FAIL and 0 NOT_YET_RUNNABLE; the complete
benchmark registry is now 35 checks. The operation is registered as
`conditional_information_audit` and exposed through content-bound plan/run endpoints.

**TG16.3 Stable-Subspace Generation. DONE (2026-08-30, `ed-dev`).** Search a bounded linear family only. The complete
dimension range, preprocessing, objective, nuisance-stability criterion, regularisation values,
optimiser, restarts, seeds and correction family are sealed before generation. Scaling and every
subspace parameter are fitted on the generate partition alone. A subspace is identified by its
projector/span rather than arbitrary component signs or rotations, and multi-seed/perturbation
stability is reported so optimiser luck cannot be called scientific stability. Output is a
**candidate compact stable subspace**, not an optimal representation.

**Delivered.** The content-bound recipe accepts two to six declared raw features, reserves but
does not open a confirmation partition, and enumerates every sealed dimension/positive-ridge
combination. Generate-only z-score scaling feeds a regularised supervised-covariance objective
with a declared-nuisance regional-instability penalty where one nuisance is present. Seeded block
power iteration, restarts, target-permutation refits, global Benjamini-Yekutieli correction and
10% multi-seed perturbations are all frozen. Admission requires a corrected generate association,
a maximum projector distance of 0.10 and, when applicable, generate-tertile explained-fraction
range no larger than 0.35. Projectors identify spans independently of basis sign or rotation;
the basis and generate-only scaling are carried solely for later unchanged application.

**Acceptance met.** Both paired G16 benchmarks now carry `G16.3.stable_subspace_generation`.
Across 200 frozen replications, exact-duplicate, noisy-copy and complementary linear candidate
rates were 1.00 each. The nonlinear XOR candidate rate was 0.025 and the independent
false-candidate rate was 0.055, both below the 0.075 ceiling; every one-feature case correctly
produced no compact subspace. The focused paired gate is 8 PASS, 0 FAIL and 0
NOT_YET_RUNNABLE; the complete benchmark registry is now 37 checks. The operation is registered
as `stable_subspace_generation` and exposed through content-bound plan/generate endpoints.

**TG16.4 Held-out and nuisance-region Confirmation. DONE.** Freeze the generated projector,
preprocessing and complete searched family, then apply them unchanged to held-out samples.
Nuisance regions are defined from generate data or an external declaration before confirmation
outcomes are opened; confirmation data may not choose thresholds, regions, dimensions or family
members. The complete searched family is corrected on confirmation, empty regions and inadequate
overlap refuse, and one held-out partition is opened once. Success is internal replication within
one dataset, not an external certificate.

**Delivered.** `/subspace/freeze` binds the TG16.3 result and its content/plan digests, every
family member's projector and application basis, generate-only means/scales, generated-candidate
status, generate-derived nuisance-tertile cuts, the unchanged 0.35 regional-stability threshold,
the complete-family BY correction and confirmation permutation ensemble into a field-digested
seal. `/subspace/confirm` accepts no scientific tuning input. It reconstructs the sealed random
partition, applies every span and preprocessing value unchanged, tests fixed projected scores by
target permutation, corrects over every searched member rather than only selected candidates,
and refuses missing nuisance-region overlap. The shared programme ledger keys consumption by the
content-and-index-bound held-out partition, so a second seal cannot reopen the same rows.

**Acceptance met.** Both paired G16 benchmarks now carry
`G16.4.held_out_subspace_confirmation`. Across 200 frozen replications with 156 generate and 68
confirmation rows, exact-duplicate, noisy-copy and complementary-linear internal-replication
rates were 1.000 each. Nonlinear XOR was 0.005 and the independent null was 0.000, below the
0.075 ceiling; every one-feature case remained outside the compact family. The paired gate is
10 PASS, 0 FAIL and 0 NOT_YET_RUNNABLE; the complete registry is 39 checks. A passing receipt is
called `internally_replicated_candidate`: it stores no evidence, moves no rung and explicitly is
not an external replication certificate.

**TG16.5 External Certification Seam. DONE (2026-08-30, `ed-dev`).** A published candidate may be tested against an
independently acquired, content-addressed dataset under a separately frozen transfer contract.
Preprocessing and the subspace remain unchanged unless adaptation was declared as a different
family before target access. Only this boundary may report an external replication receipt, and
even that receipt certifies the executed test and provenance rather than declaring a universally
optimal or causal representation.

**Delivered.** `/subspace/publish` creates a content-addressed definition only from a generated
candidate frozen by a TG16.4 seal. `/subspace/transfer/freeze` takes candidate digests and target
metadata without target bytes, requires a different content digest plus acquisition identifier,
time, source and an explicit independence declaration, and freezes the complete transfer family,
target declaration and row count, no-adaptation policy, permutation ensemble, BY correction and
target identity. `/subspace/transfer/certify` requires the independently published transfer-seal
digest, spends the target in the shared durable ledger before target-dependent validation, applies
source scaling, bases/projectors and nuisance regions unchanged, and corrects across the complete
published family. A failed post-opening validation still spends the target. The receipt records
declared provenance without pretending the software verified the truth of that declaration.

**Acceptance met.** Both paired G16 benchmarks now carry
`G16.5.external_subspace_certification`. Across 200 frozen replications with 156 source-generate,
68 source-confirmation and 68 independently generated external rows, exact-duplicate, noisy-copy
and complementary-linear external-replication rates were 1.000 each. XOR and the independent null
were 0.000, and every one-feature case remained outside the compact family. The paired gate is 12
PASS, 0 FAIL and 0 NOT_YET_RUNNABLE; the complete registry is 41 checks. The generic-file UI now
exposes TG16.1 through TG16.5 as one progressive structure programme, closing the served-route
reachability gap rather than exempting scientific endpoints from it.

**Capability and claim boundary.** G16 operations enter `DatasetCapabilityProfile` only when their
backend recipe, refusal path and benchmarks exist. Until then their presence in this roadmap is
not an available capability. Every response remains a candidate structure/relationship/subspace,
stores no evidence and moves no claim rung unless a later evidence workflow explicitly admits its
receipt. The intended progression is: **many measurements → informative candidates → mapped
information structure → conditional relationships → compact stable candidates → held-out
replication → external certification**.

#### Phase G17 — Configurable Multi-domain Structural Experiments — **IN PROGRESS**

G17 closes the gap between having domain-capable parts and having an apparatus a scientist can
actually operate. Its flagship experiment is the one the architecture has been promising: select
a shared week, three months or six months; select four genuinely different domains; acquire each
through its legitimate source; translate each into a common **structural** vocabulary; mine and
compare the declared family under appropriate nulls; and inspect or export the result. The common
vocabulary is not “weather”. Weather is its first adapter. Original units, meanings, clocks,
missingness and provenance remain attached throughout (R19).

This phase has two separate scientific modes and the UI must never blur them:

* **Calendar-aligned:** what declared structures occurred during the same UTC observation
  interval? Calendar overlap permits co-occurrence analysis only; precedence still requires every
  participating domain to admit it under R21, and neither licenses causality.
* **Scale/shape-aligned:** where does a frozen structural motif recur after an explicitly declared
  normalization of structural scale, regardless of absolute duration? This mode cannot imply
  simultaneity or lead/lag and must not inherit the calendar mode's language.

The initial flagship family is reanalysis, Argo, TESS and one independently acquired fourth
domain that breaks a different inherited assumption. The fourth domain and source are frozen in
TG17.0; it may not be chosen after seeing which source gives the most interesting answer. A
second gridded geophysical product does not satisfy R17. Live-source acceptance is recorded
separately from deterministic fixture acceptance so a remote outage cannot be mistaken for a
scientific failure.

**Programme acceptance — the no-glue test.** Starting from a clean browser session, a researcher
can choose or load an experiment, resolve coverage, understand every refusal, freeze the complete
family, press **Run experiment**, leave and resume after refresh, inspect the domain-native and
canonical results, and export a reproducible receipt. They use no terminal, notebook, handwritten
JSON, manual CSV conversion, hidden endpoint or file-shuffling step. Authentication and explicit
network consent may be supplied in the UI; they are not scientific glue. Until this test passes,
G17 is not complete even if every backend endpoint exists.

**Delivery rule:** this is a sequence of vertical product slices, not six backend slices followed
by a UI project. TG17.1 establishes the Composer shell, saved draft and preflight view; TG17.2--6
extend that same visible workflow as each contract becomes real. TG17.7 completes the guided
workflow, recovery and accessibility qualification; it is not the first point at which a
scientist sees G17. A slice with an unreachable backend route remains in progress.

**TG17.0 Flagship contract and known-answer benchmarks — DONE (2026-08-30, `ed-dev`).** Freeze the scientific question,
the four domain/source contracts, the two comparison modes and the acceptance thresholds before
building the happy path. Known-answer fixtures cover: a shared structural event with different
units and cadences; unrelated events in the same calendar interval; the same motif at different
absolute durations; misleading similarity created by gaps; a lag-looking relationship where one
domain refuses precedence; and a family large enough to exercise correction and power refusal.
The planted construction oracle is independent from the implementation being tested.

**Acceptance:** null calibration, planted-effect power, coverage/refusal expectations and maximum
family/compute budgets are written before an operation is registered. The deterministic four-
domain fixture is registered on the existing benchmark API/Platform surface; its saved Composer
recipe lands with TG17.1's manifest rather than inventing a second temporary schema. A null result
is a valid flagship result; changing the question after seeing it is not.

**Delivered.** `multidomain_flagship.py` freezes reanalysis, Argo, TESS and the existing
`order_book` declaration as the first quartet. Order book is a licensed, content-addressed user
record, deliberately retained because its irregular aggregated clock, absent metric and explicit
precedence refusal test more than another grid would. The contract separates calendar-aligned
co-occurrence from scale/shape transfer; freezes week, three-month and six-month presets; and sets
the later operation gates at 200 replications, alpha 0.05, null rejection at most 0.075, planted
detection at least 0.80, 10,000 family members, 4 GiB and one hour of planned work.

The independent construction family contains a shared calendar event, one normalized motif at
different native durations and calendar positions, unrelated records sharing only an interval,
independent values behind strongly shared gaps, an apparent event order that order book makes
inadmissible as four-domain precedence, and a 24,576-member pre-acquisition budget refusal. Every
case crosses four clocks, semantics and units and derives each domain from a distinct labelled
random stream. The focused gate is 2 PASS, 0 FAIL and 0 NOT_YET_RUNNABLE; no G17 scientific
operation or capability is registered by this prerequisite slice.

**TG17.1 Versioned experiment manifest and observation contract — DONE (2026-08-30, `ed-dev`).** Introduce one immutable
`CrossDomainExperimentSpec` carrying mode, domains, acquisition identities, measures and roles,
UTC start/end or declared scale-normalization family, window durations/stride, coverage policy,
adapter versions and parameters, family definition, nulls, correction, seeds and resource caps.
Week, three-month and six-month controls are presets that write explicit boundaries into the
manifest, never ambiguous duration labels. Multiple durations form one declared family unless the
researcher preregisters separate experiments.

Metadata preflight resolves each archive's native addressing into the requested interval without
opening measurement values. In particular, TESS sector/cadence coverage is resolved to an
interval and never presented as exact merely because a sector intersects it; Argo profile
coverage remains sparse point support; reanalysis coverage remains a grid/time extent. The result
is a coverage matrix with expected samples, native cadence, gaps, cost/bytes, credentials/network
needs and a stable reason for every refusal.

**Acceptance:** the manifest round-trips byte-stably through API, saved recipe, URL/run identity
and receipt. Browser refresh preserves every selection. Insufficient coverage refuses or remains
an explicitly permitted partial domain according to the frozen policy; it never silently changes
the interval, drops a domain or shrinks the family. The first Composer shell can create, save,
reload and preflight this manifest using visible controls.

**Delivered.** `CrossDomainExperimentSpec` is now the only G17 scientific configuration and has a
canonical byte encoding, content digest and run identity. It freezes the mode, exact UTC window
boundaries/stride, complete quartet, source and planned-adapter identities, native measure,
semantics and units, coverage rule, one duration-crossed family, domain-preserving null,
correction, seeds and TG17.0 resource caps. The saved flagship recipe round-trips through the same
schema. A draft name points to an immutable content-addressed revision, so saving an edit preserves
the old manifest while browser refresh reloads the selected draft.

The metadata-only preflight reports native addressing, support kind, nominal expected samples
only where meaningful, unknown sparse/irregular counts, gap status, byte estimate and access need
for every exact window. It does not call the network or open measurement values in this slice.
ERA5 is a grid extent; Argo remains sparse point support; TESS remains sector-bounded and is never
called exact from intersection alone. The recipe visibly refuses its deliberately absent local
order-book content binding. Once bound, Argo/TESS remain explicit partial coverage under the
recipe policy, while `complete_required` refuses; no path changes a window, drops a domain or
shrinks the 288-member family.

The separate Experiment Composer provides visible mode, coverage and exact-window controls,
complete quartet/family inspection, validate, immutable save/reload and the coverage/refusal
matrix. It preserves the shell study selection and the saved draft across refresh. The legacy
gridded parameter-sweep engine remains available under its honest name. Run is visibly disabled:
TG17.1 does not claim TG17.2 translation or TG17.6 orchestration, and creates no scientific result,
evidence or rung movement. Twelve focused manifest/API tests and the frontend contract cover the
slice; the production TypeScript/Vite build passes.

**TG17.2 Canonical structural-trajectory contract — DONE (2026-08-30, `ed-dev`).** Define the smallest shared record
the mining layer actually needs, provisionally `StructuralTrajectory`: labelled dimensionless or
unit-declared structural channels; native interval support and validity masks; structural scale
coordinate and its mapping to native scale; domain/source/variable semantics and units;
adapter/version/config digests; and every relevant assumption violation. It may carry declared
energy concentration, persistence, entropy, recurrence, change-point, motif or information
channels only where the producing adapter has a benchmarked definition. It is not a bag into
which convenient domain numbers can be renamed.

Each structural adapter declares required axes and roles, invariances, consumed information,
output clock/support, missing-data behaviour, legitimate null family, leakage risks and operations
it refuses. Raw/native records remain available for audit and are never overwritten by the
canonical projection. A canonical representation does not erase the many-to-one translation or
make raw magnitudes comparable (R19).

**Acceptance:** weather, Argo and TESS known-answer objects enter the same mining interface with no
domain branch there, while provenance can reconstruct exactly how every canonical value arose.
Semantic leakage and undeclared interpolation fail conformance. A canonical plot cannot lose the
native clock, unit, support or adapter digest.

Delivered as a vertical product slice. `StructuralTrajectory` preserves exact native interval
support, validity, meaning and units, structural-to-native scale mapping, immutable native record
identity/locator, adapter/version/config digests, assumption violations and reconstructable
per-value lineage. `StructuralAdapterDeclaration` makes required axes/roles, invariances,
consumed information, output support, gap behaviour, legitimate null, leakage risks, refusals and
benchmark-authorized channels executable rather than prose. Its first deliberately narrow
channel is a benchmarked within-record standardized level; it does not launder native semantics
or magnitudes.

Reanalysis, Argo and TESS deterministic known-answer records pass the same domain-blind mining
function. Conformance independently rejects semantic/unit substitution, any changed clock,
support or validity mask, unbenchmarked channels, changed scale mapping, dropped violations and
values that cannot be rebuilt from their native indices and declared formula. The Composer now
has an **Inspect structural contract** action whose manifest-bound preview visibly retains native
support count, coverage, unit, scale, limits and native/adapter digests. It is explicitly labelled
known-answer data, not acquired observations. Live adapter translation, order-book onboarding,
analysis, evidence and rung movement remain later slices. Ten focused trajectory/API tests plus
the frontend contract pass; the production TypeScript/Vite build passes.

**TG17.3 Adapter registry, schema-driven controls and conformance kit — DONE (2026-08-30, `ed-dev`).** Make acquisition
plus structural translation a registered `DomainExperimentAdapter` contract rather than an
orchestrator switch statement. A registration supplies the domain declaration, acquisition
planner, typed UI schema, metadata preflight, materializer, canonical translator, capability
derivation, null builder and provenance renderer. Deliver conforming adapters for reanalysis,
Argo, TESS and the frozen fourth domain in that order, keeping each vertical slice runnable from
the UI as it lands.

The conformance kit tests deterministic translation, content addressing, axis/role validation,
gap preservation, refusal propagation, declared invariances, null suitability, bounded resource
planning and API/UI schema agreement. Domain-specific advanced controls are permitted only through
the registration schema; hardcoded forms and `if domain == ...` branches in the generic composer
or runner fail review.

**Acceptance:** a synthetic fifth adapter installed through the supported extension seam appears
in the domain selector with its controls, preflights and completes the fixture without editing the
orchestrator, generic API routes or UI source. Onboarding-cost entries distinguish unavoidable
domain mathematics from framework glue; glue must trend to zero rather than merely move files.

**Delivered.** `DomainExperimentAdapter` is one registration carrying the domain declaration,
a typed `ControlSchema`, the acquisition planner, translator configuration, materializer,
structural declaration and translator, capability derivation, null builder and provenance
renderer. `EXPERIMENT_ADAPTERS` is an ordinary registry, so the domain selector, the Composer
controls and the metadata preflight read one source. `preflight_manifest` lost its literal
`source_plans` table and its `channel_table:local` special case; a manifest naming an
unregistered domain, the wrong adapter, a source that adapter does not reach, or parameters its
controls refuse now refuses by name before acquisition.

Window arithmetic stayed in the framework. An adapter declares only support kind, native cadence,
exactness, access and cost per day; `plan_windows` derives expected samples, bytes and gap status
identically for all of them. That is the line the onboarding-cost entries are measured against:
`src/adapters/reanalysis.py` is a declaration, two controls and one plan, and the shared
standardized-level translation, null, capability derivation and provenance rendering are written
once in `standardized_level_adapter.py`. Argo and TESS register through the public seam from
`extensions/`, so the extension point is exercised by this programme rather than demonstrated.

The conformance kit runs ten checks and *executes* what a declaration claims. `INVARIANCE_PROBES`
applies each named invariance to the native record and compares every canonical channel, so an
adapter claiming `native_value_positive_scaling` must have it. An invariance with no registered
probe reports `NOT_PROBED` rather than `PASS`. All four flagship adapters pass 11 of 11 checks
against their deterministic known-answer records, visibly, through
`POST /api/v1/experiment-composer/adapters/{adapter_id}/conformance`.

**The fourth domain became a family rather than a finance adapter.** Order book is now one saved
declaration of `bespoke_record`, the seam for any record a researcher holds and no catalogue
describes. Its fence is TG8.4's rule imported unchanged — *detection may create a required
declaration; it may never satisfy one*: an observed irregular clock obliges `irregular_sampling`
on a domain that has already passed `onboard_domain`, an aggregate footprint obliges
`aggregated_values`, a flat record is refused under a domain declaring richer axes, and a domain
with `lag_policy="none"` gets `precedence` added to its refused operations by construction. A new
bespoke domain is added by declaration alone, with no code — demonstrated on a `clinic_appointments`
domain in the test suite.

That flexibility is deliberately **not** treated as the acceptance evidence. A data-driven
instance tests an adapter's parameters, not the registry's extension point, so the acceptance test
installs a synthetic fifth adapter with genuinely different structural mathematics — a monotone
rank channel — from a module the application never imports. It reaches the registry, the control
schema, the conformance kit and the domain-blind mining seam with no edit to the orchestrator, the
generic API routes or the UI source.

**Three defects this slice forced out.** TG17.2's `assert_structural_conformance` reconstructed
values from a hardcoded standardized-level formula and compared every configuration digest against
`{"ddof": 0}` — one domain's mathematics inside the domain-blind pass, which failed the fifth
adapter for having different and correct arithmetic. `LINEAGE_RECONSTRUCTORS` now dispatches on the
declared operation, and an unregistered operation fails rather than passes.

The other two are in the documentation guard, and the second was hiding under the first. **D74**:
the route-count regex matched the words "those routes" written into architecture.md by TG17.1,
373 lines above the heading that carries the claim, so it parsed neither a numeral nor a spelled
number and the comparison was never reached. Repairing it let the comparison run for the first
time since TG17.1 — and it disagreed by far more than this slice had added. **D75**: `_routes()`
enumerated a hand-maintained list of ten source files, and four mounted routers were missing from
it, so **32 served endpoints were invisible to every check in that file**, the whole TG16 ingress
surface among them. architecture.md claimed 75 routes, **107 are served**, and the table headed
*"an undocumented endpoint is an untested contract"* was missing 21 rows. The list is deleted
rather than corrected: routes now come from the application object, which cannot omit a mounted
router. `test_frontend_contract.py` had enumerated `app.routes` since T3.5.22, so the two guards
had disagreed about what the API is for four slices.

`AdapterControls.tsx` renders the declared schema — its only switch is on a control's `kind` — and
the Composer has no per-domain form. 18 focused adapter tests pass (15 functions, 4 parametrised),
the TG17.1 and TG17.2 suites remain green with one deliberately updated refusal-wording assertion,
and the production TypeScript/Vite build passes at 1,402 modules. No live archive is acquired, no
cross-domain statistic runs, no evidence is written and no claim rung moves; the `source_binding`
control makes the known-answer binding a visible manifest-recorded choice and the live binding
refuses by naming TG17.6.

**TG17.4 Clock, support and coverage semantics — DONE (2026-08-31, `ed-dev`).** Calendar mode compares interval support,
not equal row indices. Every structural observation carries `[start, end)` support and presence;
pairwise overlap and effective sample size are computed from those declarations. No path silently
bins, compacts, forward-fills or interpolates an irregular record to manufacture simultaneity.
Any allowed aggregation, tolerance or resampling kernel is an adapter operation declared and
frozen before values are opened, with consequences shown in preflight.

Scale/shape mode uses a separate normalized structural-scale coordinate and retains its mapping
back to each native duration/length. It cannot emit calendar coincidence, precedence or causal
language. Calendar mode cannot silently search normalized scale ratios. A study wanting both
modes declares both and pays for the combined corrected family.

**Acceptance:** adversarial unequal-cadence, boundary, daylight/time-zone, sparse-profile,
interrupted-light-curve and non-stationary-support fixtures either align as declared or refuse by
name. Changing row density alone cannot manufacture support. The UI visualizes actual coverage
before the freeze and the exact support used afterward.


**Delivered.** `src/core/structural_alignment.py` compares half-open `[start, end)` support with
interval arithmetic and nothing else. The invariant it exists to hold is one sentence —
*changing row density alone cannot manufacture support* — and it is the load-bearing test:
splitting every record into sixty times as many rows over the same support leaves the occupied
duration, the overlap, the governing scale and the effective sample size identical. Effective
sample size is overlap **duration** over the coarser of the two native scales, so a fine record
cannot lend a coarse one resolution it does not have; the raw row count travels in every report
and is used by nothing, printed beside the number that is actually evidence. Supports are
unioned rather than summed, a zero-width support is refused, and `[a, b)` next to `[b, c)`
overlaps in nothing — which under a closed convention would have been a coincidence at every
boundary of every regularly sampled record.

Nothing bins, compacts, forward-fills or interpolates by default. `exact_support_overlap` is the
only kernel that runs unnamed, and it transforms nothing. Every other kernel is a declared
adapter operation: frozen in the manifest's `AlignmentPolicy`, inside the manifest digest,
admitted by *every* participating adapter through the adapter contract's new
`admissible_kernels`, with no framework default for any parameter, and reporting in seconds how
much of the resulting overlap it created rather than observed. A value-inventing kernel is
refused outright over a domain declaring `irregular_sampling` or `aggregated_values`. Which
kernels a domain admits is a domain judgement: reanalysis admits tolerance and grid aggregation,
Argo admits tolerance but not a grid the array does not keep, TESS admits a grid but not a
tolerance that would blur the observational gap deciding whether a target was observed at all,
the bespoke family admits only the kernel that transforms nothing, and no adapter admits
`carry_forward`.

The two modes cannot borrow each other's vocabulary, and the manifest refuses the mismatch where
the search is declared rather than where the result is worded. Scale/shape correspondences retain
both native durations, so a match is reported as a shape recurring at 1.8 hours here and 46 days
there. Declaring both modes prices the union of what was searched.

Consequences are visible before the freeze. `preflight_manifest` gains an alignment block that
binds the kernel against every participating adapter, states each window's true elapsed UTC
seconds, and reports a pair as **bounded by the window** rather than inventing an overlap number
where metadata cannot establish one — which, for three of the four flagship domains, is the
honest answer. `POST .../manifests/alignment` then measures the support the known-answer records
actually have, and `CoverageTimeline.tsx` draws it positioned by time rather than by index, with
the gap count, governing scale, effective sample size, kernel-created seconds and the greyed-out
row count beside them.

**Acceptance met.** `src/benchmarks/alignment_fixtures.py` carries the six adversarial cases and
their known answers: unequal cadence resolves to 28 effective observations rather than 168; the
abutting boundary shares nothing and refuses by name; the daylight-saving day is 82,800 seconds,
where a nominal denominator would have reported 95.8% coverage for a record covering the window
completely; the sparse Argo profile shares 36 hours of nine ascents inside ninety days; the
interrupted light curve's two-day downlink gap survives a continuous partner and the overlap
comes back as two intervals; and non-stationary support has its effective sample size labelled an
upper bound rather than corrected.

**Evidence.** `test_structural_alignment.py` 45 functions / 56 cases; the TG17-adjacent suites,
the benchmark and cross-domain suites, the frontend contract and the documentation audit at
362 passed; production build 1,403 modules; `git diff --check` clean. Nothing acquired, no cross-domain statistic run, no evidence written, no claim rung
moved.


**TG17.5 Multi-domain family accounting and domain-legitimate nulls — DONE (2026-08-31, `ed-dev`).** Freeze all tested
domain pairs, triples, quartets, windows, durations, scales, motifs, lags and representations as
one explicit hypothesis family before mining. The plan reports the family expansion in human
terms and runs R18's resolution/power check before acquisition. Pairwise screens may generate
candidates, but confirmation corrects over the complete search that produced them.

Nulls are adapter-declared and mode-specific. They preserve the features that would otherwise
create false structure: autocorrelation, seasonal/cyclic phase, irregular gaps, profile support,
observation windows and any declared grouping. Calendar-co-occurrence nulls are distinct from
scale/shape-similarity nulls. A global shuffle is not accepted merely because every domain can
technically execute it.

**Delivered.** `src/core/experiment_family.py` turns a manifest into one `SearchSpecification`
over eight declared axes — domain set, window, channel, scale, relationship, lag, representation
and motif — replacing the product that was written inline in `preflight_manifest` and copied
again into the browser. Domain combinations are unioned across declared arities rather than
multiplied, so pairs and triples of four domains are 6 + 4 members. `FamilyDefinition` gained
`domain_arities`, `lags_seconds`, `representations` and `motifs`, each inside the manifest
digest. `family_expansion` renders the multiplication as a sentence, prices the same declaration
with one more domain, duration, scale and channel, and reports the surrogate cost of each.
`ScreenedSearch` holds the screen beside the complete search and refuses to correct over the
survivors of a screen unless a held-out partition is named. `precedence_availability` counts the
members an absent lag policy leaves untestable without reducing the family. `structural_nulls.py`
became a registry of mode-tagged `NullFamily` objects with declared preserved and destroyed
features, four admissible families and `global_value_shuffle` registered-and-refused; adapters
declare `admissible_nulls`, and preflight binds the manifest's null against every participating
adapter. Two routes (`GET .../null-families`, `POST .../manifests/family`) and
`frontend/src/components/FamilyPlan.tsx` put all of it in front of the researcher before the
freeze.

**The arithmetic this forced out.** Pricing the flagship through `SearchSpecification.account()`
for the first time showed that its 288 declared tests need about **35,953 surrogates** under
Benjamini-Yekutieli at alpha 0.05, while the frozen acceptance policy declares 200 — at which the
largest affordable family is **four**. The study could have run to completion, cost the full
amount and been arithmetically incapable of rejecting anything (**D76**). The remedy is the one
R18 already admits: the manifest gained a `ConfirmationPolicy`, a `confirmatory_only` study is
priced at its complete family and refused when it cannot resolve it, and the flagship now
declares `generate_then_confirm` against a named held-out partition with four confirmatory
members — with every payload stating that its generate stage produces candidates and not claims.
A second defect fell out of the same pass: the flagship's null carried a `preserve_gaps`
parameter that nothing had ever read (**D77**).

**Acceptance met.** `src/benchmarks/family_calibration.py` calibrates the TG17.0 fixtures at the
frozen family level — six pairs, one Benjamini-Yekutieli correction, 999 replications, the
declared null applied through `bind_null`, and a support-weighted statistic that inherits TG17.4's
density invariance. The planted `shared_calendar_event` is confirmed on **6 of 6** pairs; the
false-alignment fixtures `same_window_unrelated`, `gap_alias` and `inadmissible_precedence` each
reject **0 of 6**, so an identical outer interval and a shared observation gap do not become
shared structure. Adding a domain or a duration updates the visible family and its surrogate
requirement before the freeze (288 → 480 and 288 → 384, with 35,953 → 64,819 and 50,143
surrogates required). Two of the four flagship domains declare no justified lag policy: they take
part in structural association, and the 240 precedence members that pair them are reported
unavailable while the declared family size stays the correction unit.

**Evidence.** `test_experiment_family.py` 54 functions / 57 cases; `test_frontend_contract.py`
117; the TG17 suites, the benchmark and cross-domain suites and the documentation audit run
together; production build clean. Nothing acquired, no confirmatory statistic run, no evidence
written, no claim rung moved.


**TG17.6 Content-addressed, resumable experiment orchestrator — DONE (2026-08-31, `ed-dev`).** Execute one state machine:
`DRAFT -> PREFLIGHTED -> FROZEN -> ACQUIRING -> TRANSLATING -> MINING -> CONFIRMING -> COMPLETE`,
with explicit `REFUSED`, `FAILED` and `CANCELLED` outcomes. Every transition is idempotent and
content-addressed; completed acquisitions and translations are safely reusable, while target
openings and held-out partitions obey their existing ledgers. Retries cannot create a new
scientific plan or reopen a spent target.

Partial acquisition, remote timeout and adapter failure remain visible per domain. They never
silently run a smaller experiment. The frozen policy alone decides whether a declared partial
result is admissible, and the receipt lists every missing component. Progress events expose stage,
domain, bounded work estimate, completed artefact digests and actionable remediation without
leaking unopened results.

**Acceptance:** kill/restart and browser-refresh tests resume the same run without duplicate
network acquisition or scientific drift. A TESS timeout can be retried from the UI; a permanent
coverage refusal returns to an editable copy rather than mutating the frozen run. Re-executing an
identical complete manifest returns the same run identity and immutable artefacts.

**Delivered.** `src/core/experiment_run.py` holds the machine as a transition **table**, so what a
run was allowed to do next is one dictionary rather than a chain of branches. Run identity is a
content address over the schema and the manifest digest and nothing else — no clock, no UUID, no
machine — so executing an identical manifest *is* the same run: `POST /api/v1/experiment-runs`
resumes rather than creates, and the Composer has no "new run" control because there is no such
operation. Each step is keyed by the digest of the run, the stage, the component and the digests of
its declared inputs, published immutably through `publish_new_bytes` and replayed from disk, which
is both the no-duplicate-acquisition guarantee and the drift check: a changed native artefact
re-keys its translation instead of being paired with a stale one. Operational failures are
deliberately not published under a step address, because a timeout is a fact about a network at a
moment and not a function of the declared inputs.

`decide_stage` is a pure function of the frozen `CoveragePolicy` and the component statuses, so the
decision that turns a partial acquisition into either a smaller experiment or a refusal can be
audited without reconstructing a run. `complete_required` refuses; `partial_permitted` admits the
run only above its declared minimum fraction; the receipt names every missing component either way.
A refusal outranks a failure and a failure outranks a missing component, because "we could not ask"
is not "the answer is no". `runs/<run_id>/journal.jsonl` is the only state: append-only, fsync'd
per line, folded on read, and a torn tail is skipped rather than raising. `HeldOutOpenings` records
the run that first spent a partition and refuses a second one by name.

`src/core/run_workers.py` registers the stage-worker suites rather than accepting behaviour from a
request, and everything registered today **acquires nothing**: `fixture_dry_run` rehearses a frozen
plan end to end, and `fixture_transient_failure` times out each acquisition component once and
completes it on retry, reading "first attempt" from the run's journal so it behaves identically
across a restart or four separate HTTP requests. Eight routes and
`frontend/src/components/RunMonitor.tsx` put the machine in the browser: the declared state trail
drawn from the backend's own table, per-component statuses and digests, a bounded progress bar,
the components the run did not produce, and Retry and "open an editable copy" as two buttons that
are never both live.

**Acceptance met.** A killed run resumes and does not re-request the coverage it had already
acquired; a torn journal tail is skipped and the run still resumes; re-executing an identical
complete manifest returns the same run identity and byte-identical artefact digests, and requests
nothing. A timed-out acquisition leaves the run `FAILED` with a remediation and is retried from the
UI over HTTP, re-executing only that component while the rest are replayed; a retry carrying a
different manifest is refused by digest. A permanent refusal is terminal, is not retryable, and is
answered with `editable_copy`, which writes a new draft and leaves the frozen run's journal
unchanged. `ComponentOutcome` has no field a result could occupy, so a progress feed watched during
`MINING` cannot report what mining found.

**A defect the widened verification set found (D78).** `test_analysis_api.py`'s TG11.1 acceptance
test asserted `len(names) == 13` before posting every sequence and cross-domain benchmark to
`/api/v1/benchmarks/run`. TG17.0 registered two more, so the assertion aborted the test **before
the HTTP call**, and from that slice onward `multidomain_flagship_planted` and
`multidomain_flagship_safeguards` - the benchmarks carrying the four-domain flagship's planted and
safeguard answers - were never once exercised across the API boundary the test exists to exercise.
It survived four slices because TG17.1-17.5 each verified against targeted suites that did not
include that file. Fixed by deriving the count from the registry with a floor so coverage cannot
shrink unnoticed, and by naming the two benchmarks explicitly.

**Evidence.** `test_experiment_run.py` 80 tests; `test_frontend_contract.py` 127; the TG17 suites,
the manifest, family, benchmark and cross-domain suites and the documentation audit run together;
production build clean. Nothing acquired, no confirmatory statistic run, no evidence written, no
claim rung moved — the registered suites are rehearsals, and a rehearsal that completes is not a
result.

**TG17.7 Experiment Composer UI — DONE (2026-08-31, `ed-dev`).** Build one guided workbench over the manifest rather
than four acquisition pages plus instructions. The progressive path is:

1. **Question:** calendar-aligned or scale/shape-aligned, with the claim boundary beside the choice.
2. **Domains:** choose two or more sources and see which assumption each breaks.
3. **Observation:** select explicit dates or week/three-month/six-month presets and domain-native
   measures/roles through adapter-supplied controls.
4. **Preflight:** inspect coverage, gaps, credentials, estimates, unavailable operations and
   remedies in one comparison view.
5. **Analysis:** choose a benchmarked recipe; inspect the expanded family, nulls, correction,
   seed and confirmation design with advanced controls disclosed but never hidden.
6. **Freeze and run:** review a plain-language preregistration summary, freeze it, then monitor,
   pause safely where supported, retry operational failures and resume after refresh.
7. **Interpret:** move from native records to canonical structure, corrected comparisons,
   limitations and evidence export without losing the experiment context.

The composer supports saved versioned recipes, clone-to-edit, an advanced manifest inspector and
machine-readable import/export; direct JSON editing is never required. Dataset acquisition does
not silently make a dataset “the study”: the UI distinguishes acquired material, an experiment
run, a finding and admitted evidence, and always shows the next legitimate action.

**Acceptance:** the complete TG17.0 flagship is executed in a browser automation test using only
visible labelled controls. Keyboard navigation, focus, loading/empty/error/refusal states,
responsive layout and destructive-action confirmation are tested. Every served G17 route is
reachable from the shell; selection persists across navigation and refresh; unavailable choices
remain visible with the backend's reason rather than disappearing.

**Delivered.** `src/core/composer_path.py` holds the workflow as a registry rather than a layout.
`COMPOSER_PATH` carries the seven steps, each a `PathStep` that decides its own status from the
manifest, and `compose_state` returns exactly **one** `next_action`. The browser renders that; it
does not compute it. That is the whole slice: the order of operations *is* the scientific
discipline - a family priced after acquisition is priced knowing what the data looked like, a null
chosen after the statistic exists is not a null - and a UI that permits those in any order has not
made an error, it has made the error undetectable, because no receipt can distinguish an
experiment that was declared from one that was assembled.

Statuses are three-valued, because "you have not done this" and "this cannot be done yet" are
different sentences and only one of them is the researcher's move. A blocked step keeps its tab
and its reason: the whole flagship blocks at preflight naming `order_book`, rather than dropping
the domain and reporting a complete three-domain study. `STAGE_LADDER` names the four things a
researcher can possess - acquired material, an executed run, a finding, admitted evidence - each
with what it is, what it is **not** and its own gate; this surface can move a researcher across
the first two and structurally cannot move them across the last two.

Duration presets resolve on the server, in calendar terms, and apply as the explicit instants they
resolved to. Writing them as fixed day counts was caught in test: 182 days from the flagship's own
anchor is 2026-07-02, so a researcher pressing the preset that described their own window would
have moved its boundary and re-addressed the manifest. The domain menu filters nothing and returns
a domain with no declared observation unselectable with the reason. The preregistration summary is
generated from the bytes that are hashed, since a preregistration signed after reading a summary
the UI composed itself is a preregistration of the summary. `POST /path/state` deliberately looks
for a run at the manifest's content address and never opens one, because `RunStore.open` publishes
a frozen manifest and a read of where a draft stands must not be the thing that freezes it.

**Acceptance met, in a real browser.** `frontend/e2e/composer-path.spec.ts` drives Chromium
through `frontend/playwright.config.ts`, which serves the actual API and the actual frontend and
points the backend at a scratch state directory for the same reason the pytest `client` fixture is
bound to `tmp_path`. Eleven tests, and every locator is a role and a visible name - no CSS class,
no test id - because a test that clicks `.btn-primary` proves the DOM has a div, not that a person
could declare an experiment. They cover the seven ordered steps with blocked ones still reachable,
exactly one next action naming its own route, arrow/Home/End keyboard navigation, the place kept
across a browser refresh, the preflight refusal naming `order_book` and its reason, what each
domain breaks shown before it is chosen, the ladder refusing to call a run a finding, empty panels
that read as unasked questions, two-press destructive confirmation, the complete executable plan
declared and run end to end, and a refresh that resumes the same run rather than starting a second.

The one departure from the wording is that the **complete** flagship cannot be executed and should
not be: `order_book` is bespoke, has no public archive, and metadata cannot plan its coverage, so
the flagship blocks at preflight by design. The browser test drives that refusal, resolves it
through the visible domain menu, and runs the resulting three-domain plan - which is the honest
version of the acceptance rather than a weaker one.

**The defect the browser found in its first minutes (D79).** The domain menu's checkbox was bound
to `row.selected` from the `GET /domain-menu` payload rather than to the manifest. Unchecking a
domain updated the plan at once, but the controlled input re-rendered from the previous payload,
snapped back to checked, and flipped again ~200ms later when the refetch landed - so for that
window the control reported the **opposite** of the choice just made. Every source-level and HTTP
test passed throughout, because the manifest and the payload were both correct; only the rendered
control was wrong, and until this slice nothing in this repository rendered anything. That is
precisely the gap TG11.6 recorded and could not close. Fixed by driving the checkbox from the
manifest the browser already holds.

**Evidence.** `test_composer_path.py` 49 tests; `test_frontend_contract.py` 127 -> 144;
`frontend/e2e/composer-path.spec.ts` 11 browser tests. Seven new routes, 119 -> 126. Production
build clean. No archive is acquired, no statistic runs, no finding is recorded and no evidence is
admitted.

**TG17.8 Scientific comparison views — DONE (2026-08-31, `ed-dev`).** Provide linked views that make the abstraction
inspectable rather than magical: a cross-domain coverage timeline; native-record preview beside
canonical trajectories; native-to-structural scale mapping; pair/triple/quartet result matrix;
motif correspondence and transfer view; null distributions, corrected values and power/resolution;
and provenance drill-down to source, adapter, parameters and support. Selecting an apparent match
highlights its contributing native intervals in every domain.

Raw magnitudes from different domains never share a quantitative axis. Colour, ordering and
language distinguish generated candidates, held-out confirmations, external transfers, nulls and
refusals. Accessible tables contain the numerical result behind every visual, and all views state
what may and may not be concluded.

**Acceptance:** TG17.0 semantic-trap fixtures cannot be rendered as magnitude equivalence,
precedence or causality. Sparse or absent coverage is visually distinct from a measured zero.
Every plotted point traces to an immutable artefact and every correction denominator is visible.

**Delivered.** `src/core/comparison_views.py` holds seven views in a registry ordered by ordinal -
coverage timeline, native record beside canonical trajectory, native-to-structural scale mapping,
pair/triple/quartet result matrix, motif correspondence and transfer, nulls with correction and
resolution, and provenance drill-down - each declaring its axes, its legend roles, what it may
conclude and what it may not. Six routes under `/api/v1/comparison-views` (126 -> 132) serve the
contract, the legend, the set, one view, a linked selection and a reading check. The browser
renders `frontend/src/components/ComparisonViews.tsx` inside the composer's Interpret step; it
holds no boundary text of its own.

**The refusals are structural, not advisory.** `Axis` raises `MagnitudeEquivalenceError` when a
`native_magnitude` coordinate is given more than one domain - at *construction*, so a view that
would put two units on one ruler never finishes being built and cannot reach a browser, an export
or a screenshot. `Mark` requires exactly one of an artefact digest and a reason it has none.
`register_encoding` refuses a role duplicating another's colour, marker *or* word, so the
candidate/confirmation distinction survives for a reader who cannot use colour. A coverage cell is
a named state (`COVERED`/`SPARSE`/`ABSENT`/`REFUSED`) in the payload and in the component, which
draws from `CELL_STYLE` with no numeric path into it - absent support has no width to be zero.

**The distinction the slice turns on.** `MODE_RELATIONSHIPS["calendar_aligned"]` admits
`causality`, and that stays true: a study holding an external intervention design may declare and
test it. No view may *draw* it, because every alignment computed here is observational and the
design that licenses the arrow has no field in the manifest. Refusing the declaration would forbid
a legitimate study; permitting the drawing would let any co-occurrence be read as a cause. So the
contract serves `declarable_by_mode` and `renderable_by_mode` as two lists with the reason, and a
manifest declaring `causality` still gets its matrix cells - occupied by the refusal, because a
blank cell is indistinguishable from one nobody thought about. `magnitude_equivalence` and
`semantic_equivalence` are refused in both modes.

**Honest about what is not measured.** No stage worker produces values yet, so result and null
cells read `NOT_YET_MEASURED` with the reason. `ViewContext.results_exist` requires a `MINING/`
artefact rather than trusting `state == "COMPLETE"`, because TG17.6 lets a run complete under
`partial_permitted` with mining components missing by name - without that, a run that completed
having mined nothing would render its matrix as measured and empty. What *is* shown now is the
correction denominator, the declared search size and the p-value floor: arithmetic about the
declaration, computable before a byte exists, and worth reading before committing to the plan.
Like the composer path, these routes look for a run at the manifest's content address and never
open one.

**What the reachability guard caught.** `test_every_api_method_is_reachable_from_the_ui` failed on
four service methods with no caller - the contract, the legend, the single-view render and the
reading check were served and invisible. They are now the legend at the top of the panel, a
*What these views will not draw* disclosure carrying the three refusals, a per-view *Refresh*, and
a control that asks the server whether a chosen reading can be drawn and prints the reason. That
control is the TG17.0 semantic-trap acceptance made operable rather than only asserted.

**Two a11y defects the browser found.** A `<details>` element is exposed as a group whose
accessible name is *not* computed from its `<summary>`, so both disclosure panels were regions a
screen-reader user would meet with no name at all. Found by `getByRole('group', { name })` timing
out; fixed with explicit `aria-label`s. Separately, one full-suite run failed on the motif view
where a helper waited only for the first view to paint; the helper now waits for the seventh, so
each test's assumption that the whole set is present is stated once rather than raced on.

**Evidence.** `test_comparison_views.py` 65 test functions (73 runs with parametrisation);
`test_frontend_contract.py` 144 -> 154; `frontend/e2e/comparison-views.spec.ts` 19 browser tests,
and the complete browser suite is 30. Six new routes, 126 -> 132. Production build clean. No
archive is acquired, no statistic runs, no finding is recorded and no evidence is admitted.

**TG17.9 Receipt, methods report and evidence handoff — DONE (2026-08-31, `ed-dev`).** A completed run exports an
immutable bundle containing the exact manifest; coverage decision; source/acquisition identities;
native and canonical artefact digests; adapter contracts; environment; family/null/correction;
seeds; stage events; results; refusals; and software version. Produce both a machine-readable
bundle and a scientist-readable methods/limitations report suitable for review, without claiming
publication readiness or independent replication where those have not occurred.

Completion creates an experiment receipt, not automatically a study, finding, EvidenceBundle or
claim promotion. The UI offers explicit, reviewable handoffs into those existing workflows and
shows which required evidence categories remain absent. Platform & evidence explains the entire
G17 lineage and its claim limits from the same backend capability/receipt contracts, so the trust
surface cannot lag behind the engine unnoticed.

**Acceptance:** delete UI state and reconstruct the run from the exported bundle; all scientific
identities and conclusions are unchanged. A platform-capability snapshot and documentation audit
fail when a registered G17 operation, adapter, refusal or receipt field has no visible explanation.

**Delivered.** `src/core/experiment_receipt.py` distinguishes TG17.6's live journal projection
from the archival `cross-domain-experiment-bundle/v1`. A completed export carries the exact
manifest and derived run identity; metadata preflight and coverage decision; acquisition identity
beside source plan; full registered adapter contract and translator configuration; native,
canonical, mining and confirmation digests in separate roles; family, nulls, correction, alpha,
confirmation policy and labelled seeds; freeze-time environment; the exact event sequence;
refusals; result identities; evidence-category census; methods report; and claim boundary. The
bundle and Markdown report are content-addressed and published through the no-overwrite boundary.

Replay is semantic, not merely a checksum. It reconstructs the manifest, run identity, transition
history, component outcomes, artefacts, decisions, bounded work and terminal state from the event
sequence and requires the embedded receipt to agree exactly. A caller that forges a state or
artefact and recomputes the outer digest is still refused. An unknown top-level field is refused
rather than accepted as an unexplained claim. Only COMPLETE runs export; a refusal or failure
keeps its honest live receipt.

The first Chromium import found **D81**: Python emitted an integral JSON value as `1.0`, while a
browser round trip emitted the same JSON number as `1`, so the original digest rejected an
untouched bundle. Canonical hashing now normalises integral numbers according to the JSON data
model, with both a focused spelling-loss regression and the real browser export/replay path.

`frontend/src/components/ExperimentReceipt.tsx` is mounted in Interpret and Platform & evidence.
It renders the backend's operation, adapter, refusal, lineage and field registry; exports JSON and
Markdown; and can discard all browser state then reconstruct the exact run from a user-selected
bundle. The handoff displays `registered_hypothesis`, `admitted_evidence`,
`independent_replication` and `claim_promotion` as absent. Its only action navigates to the
separate evidence-study draft; `automatic_actions` is empty and no evidence route is called.
The current registered rehearsal's mining and confirmation markers remain `fixture_artefacts`;
they do not satisfy `measured_results`, because no measurement value was opened.

The documentation audit imports the generated capability snapshot and requires every registered
operation, adapter, refusal and receipt field to have an explicit backticked explanation in
`architecture.md`. This is the same contract the two browser surfaces render, so a new capability
cannot become usable while remaining invisible on the trust surface.

**Evidence.** `test_experiment_receipt.py` has 21 test functions; the TG17.9 additions bring
`test_frontend_contract.py` to 157 and `test_documentation.py` to 20, while all 80 TG17.6
orchestrator tests remain green. Those four suites account for 278 focused tests. The full
Chromium suite is 33/33, including three TG17.9 rendered tests; the production build transforms
1,408 modules; four new routes bring 132 to 136; and the documented test-function inventory is
2,847. The full backend suite has not been rerun since TG17.7, so 3,106 remains the last measured
full-suite figure. No live archive is acquired, no statistic runs, no finding is recorded and no
evidence is admitted by this slice.

**TG17.10 Flagship qualification and no-glue release gate — APPARATUS GATE DELIVERED, RELEASE WITHHELD (2026-08-31, `ed-dev`).** Run the frozen four-domain
known-answer family for a week, three months and six months in both comparison modes, then record
the live-source tail separately with exact dates, archive coverage and any operational refusal.
Qualify the entire browser path, public API path and exported replay against the same manifests.
Measure scientist actions, adapter-specific framework edits, recovery from one remote failure and
time to understand why a requested analysis is unavailable.

**Definition of done:**

* The no-glue browser test passes exactly as stated at the start of G17.
* The synthetic fifth-adapter test requires no generic runner, route or UI edit.
* Calendar and scale/shape modes pass their separate calibration, power and language gates.
* Week, three-month and six-month manifests retain explicit dates, coverage and complete-family
  correction; no duration is selected after results are opened.
* A refresh, process restart and recoverable acquisition failure resume without scientific drift.
* The result can be replayed and audited from its receipt, and its evidence/claim status is
  impossible to confuse with acquisition or experiment completion.
* `architecture.md` describes only the G17 capabilities actually delivered; this roadmap records
  live checks that were not run or did not pass instead of polishing them into availability.

**Delivered.** `src/core/experiment_qualification.py` is a release gate rather than a scientific
worker. It derives six frozen cells — `week`, `three_months` and `six_months` crossed with the two
comparison modes — from the single flagship recipe, so no second scientific configuration exists.
Each cell keeps explicit UTC boundaries, the complete-family correction and an explicit
`duration_selected_before_results` marker, and binds the order-book observation by the content
digest of its known-answer record rather than by a filename. Each mode carries its own
relationship, null family and claim language: `co_occurrence` under
`independent_native_clock_shift`, `shape_recurrence` under `scale_partner_reassignment`.

`execute_offline_qualification` preflights every cell and executes only the admissible ones. A
cell passes only when the preflight refuses nothing, the run completes, one manifest digest
appears in the preflight, run identity, exported bundle and replayed receipt, integrity verifies,
results are unmeasured with no artefacts, and every evidence category through `claim_promotion` is
`ABSENT` with no automatic action. Repeating the qualification resumes the identical runs.

**The headline result is a refusal, not a green matrix.** Three of the six cells come back
`REFUSED` before anything executes. `order_book.bespoke_record` declares that it cannot carry
`scale_partner_reassignment` — that null alters no record, so admitting it would claim the domain
has a native duration worth comparing shapes across, and a depositor-supplied record's native
scale is whatever the depositor wrote down. **The frozen four-domain quartet cannot be qualified
in scale/shape mode at all.** The ledger records that with its reason instead of narrowing the
quartet, and the browser answers the same way: asking for scale/shape in the Composer produces a
written explanation and leaves the plan calendar-aligned, rather than switching silently and
failing later at execution. `REFUSED` is kept distinct from `FAIL` in the record and on screen —
both block release, but nothing in the apparatus broke. A refused cell opens no run, because a run
identity for an experiment that was never conducted cannot later be told apart from an unexecuted
one.

Recovery is measured separately from the existing broad-outage rehearsal.
`fixture_single_remote_failure` times out exactly one remote-shaped acquisition and completes the
other three; the run is then reloaded through a fresh `RunStore`, which is the real process
boundary. The retry must name only the failed component, that component must show two attempts
against one for each other, the run identity must be unchanged, and the recovered run must still
export a bundle that replays as `VERIFIED`.

**What is deliberately not delivered.** The ledger registers seven gates and passes two. It
records `browser_no_glue` and `synthetic_fifth_adapter` as `NOT_RUN` because a backend rehearsal
must not award a gate only a rendered browser test and a source-edit audit can measure;
`calendar_calibration` as `NOT_RUN`; `scale_shape_calibration` as `NOT_IMPLEMENTED`, because no
registered scale/shape mining calibration produces a scientific statistic at all; and
`live_sources` as `NOT_RUN`, because network stays opt-in and archive coverage with its
operational refusals needs a separately dated live record. `scientist_actions` reports
`NOT_MEASURED` for the action count, adapter-specific framework edits and refusal-explanation
time rather than inventing them. The verdict is `NOT_RELEASEABLE` and cannot be otherwise while
any gate is unpassed: `verify_qualification_record` re-derives the record digest *and* refuses a
`RELEASEABLE` verdict carrying a non-passing gate.

Two of those gates are measured, just not here. The no-glue browser test now passes in Chromium,
and `test_adapter_registry.py`'s synthetic fifth adapter — TG17.3's acceptance test — passes with
its 17 siblings. Both gates nevertheless read `NOT_RUN` in the ledger, because a deterministic
backend rehearsal cannot observe a rendered browser or a source-edit audit and must not award
itself a gate on someone else's evidence. What stands between this slice and a completed TG17.10
is therefore: a trustworthy channel that feeds those two external results into the record, the
calendar calibration run, a scale/shape mining calibration that does not yet exist at all, and
the dated live-source tail.

**D83, caught by the full suite rather than by this slice.** Making the scale/shape cells
admissible in the first place was done by widening the *framework default* `admissible_nulls` in
four places — the `DomainExperimentAdapter` dataclass and the three adapter builders. Every
targeted suite, the production build and the entire browser suite passed. What that change did was
answer, for every adapter author including one who has not written their adapter yet, a question
only an adapter author can answer; and it silently overruled the order-book adapter's own
documented refusal. The only objection came from `test_experiment_family.py`'s pinned per-domain
declaration, in an 11-hour full-suite run — which is the argument for running it. The defaults are
reverted, the three domains that do admit the null declare it individually with stated reasons,
and the matrix now reports the refusal the declarations actually imply. This is worth recording
because it is the programme's central failure mode in miniature: a framework default quietly
making a scientific choice, with every fast check green.

**D82, found by the clean-browser gate.** The acceptance test composed a four-domain scale/shape
plan through visible controls and was refused at execution. A held-out confirmation partition is
confirmatory exactly once; the frozen flagship ships with one default partition name; the first
run to open it spends it. Every later plan derived in the Composer inherited that spent name, was
correctly refused, and was told to declare a new partition — through a form with no control for
declaring one. A browser-composed plan was therefore executable at most once per deployment, and
the only escape was hand-editing a manifest, which is exactly what the no-glue promise forbids.
The analysis step now carries an explicit *Held-out confirmation partition* control. Two smaller
findings came from the same test: the observation disclosure panels were `<details>` groups with
no accessible name (the TG17.8 defect recurring in a second view, fixed with `aria-label`), and
the acceptance test's own `addInitScript` cleared `localStorage` on *every* navigation, so the
refresh it called a resume was really a new browser.

**Evidence.** `test_experiment_qualification.py` has 16 test functions (21 runs with
parametrisation); with `test_experiment_family.py` that is 77 passing tests. The complete Chromium
suite is 35/35 from a cleaned `.e2e-state`, including the two TG17.10 browser tests; the production
build transforms 1,409 modules; two new routes bring 136 to 138. The full backend suite was rerun
twice for this slice: **3,238 passed with 1 failed** before D83 was fixed, and **3,240 passed, 4
skipped, 1 xfailed, 0 failed in 39m 40s** after. That clean 3,240 replaces TG17.7's 3,106 as the
last measured full-suite figure. No archive is acquired, no statistic runs, no finding is
recorded, no evidence is admitted and nothing is released by this slice.


**TG17.11 Scale/shape mining calibration — DONE — all five slices (2026-09-03).** *Superseded below: the gate now reads `REFUSED` rather than `NOT_IMPLEMENTED`, and the finding is that the method is calibrated while G17's declared families cannot reach it.* The
`scale_shape_calibration` gate was, when this task was scoped, the only one of TG17.10's seven that read `NOT_IMPLEMENTED`
rather than `NOT_RUN`. The distinction is exact and it is the reason this task exists: the other
unpassed gates have a method that has not been executed or has no channel to report itself, while
this one has no scientific statistic at all. Half of G17's two declared scientific modes is
currently unvalidatable.

**What already exists, and is sound.** The scale/shape foundations are further along than the
gate's wording suggests. `invariance.py` supplies `relative_geometry`, a matcher whose
scale-invariance is *measured* rather than declared — normalising each edge by the geometric mean
of every edge reproduces to 0.37% across translation, rotation and rescaling, inside replicate
noise — together with `calibrate_match_tolerance`, which fixes the comparison tolerance by
re-measuring the same configuration under fresh noise instead of letting an author choose it, and
`recover_scale_ratio`. `motif.py` supplies a complete mining statistic with a null: `support_of`,
`surrogate_scene` and `motif_p_value` in the `(1 + k) / (1 + n)` form. The
`scale_partner_reassignment` null family is registered and implemented.
`structural_alignment.py` supplies `ScaleShapeCorrespondence`, which retains the mapping back to
both native durations so a normalised match is always reportable as "1.8 hours here, 46 days
there" rather than as an unqualified similarity, and `assert_mode_admits_relationship`, which
refuses calendar vocabulary in this mode by name.

**What is missing is the statistic, and the template for it is already written.**
`src/benchmarks/family_calibration.py` is the calendar analogue and is complete: a statistic
(`support_weighted_correlation`, built from shared support *duration* so that rewriting a record
at ten times the row density gives the same number), the declared registered null applied through
`bind_null`, 999 replications giving a per-pair null distribution, the surrogate p-value, one
Benjamini-Yekutieli correction over the whole declared family, and four cases whose expected
answers are frozen in the module rather than in the test — one planted case that must reject every
member and three safeguards that must reject none. Scale/shape needs the same five parts. It has
the null and it has none of the other four.

**D91 blocks everything downstream of it, and was found by probing the null on the family it would
actually run.** `reassign_scale_partners` deranges list positions, not pairings. Its own docstring
states the requirement it violates. The right members of an all-pairs family repeat, so a shuffle
guaranteeing `order[i] != i` still frequently yields `rights[order[i]] == rights[i]`. Measured over
5,000 draws on the three domains that admit this null — `argo_float`, `reanalysis`,
`tess_lightcurve`, with `order_book` declining under D83 — two of the three members are unchanged
from the observation 50% of the time, and one of those pairs a record against itself 50% of the
time. A surrogate equal to the observation satisfies `null >= observed`, so those members carry a
p-value floor near 0.5 before correction and cannot reject at any effect size.

The bias is conservative, and that is precisely what makes it dangerous here. A calibration built
on this null would have measured near-zero planted power and been read as a well-behaved safeguard
result rather than as a broken null — a false negative wearing the appearance of rigour, which is
this programme's stated central failure mode in a second guise. Nothing already recorded is
affected: no scale/shape cell has ever executed, because all three are `REFUSED` before execution
by D83's per-domain declarations.

**The family is three members, not six** — *superseded by slice 1 below, which found that the three-member inventory admits no reassignment at all and so cannot carry this null; kept as written because the pricing argument still holds for a three-member family and only the membership changes.* D83's per-domain declarations mean the scale/shape family
is the three unordered pairs of the three admitting domains. R18 prices that at 59 required
surrogates for alpha 0.05; 999 replications give a p-value floor of 0.001 and
`check_power(999, 3)` reports `can_reject_after_correction: True`. The calibration is affordable.
It is a different family from calendar's six and must be declared as its own, not inherited.

**The statistic is the open scientific question, and one measurement in this tree already
constrains it.** The mode compares a shape at one native duration against a shape at another, so
the statistic must be invariant to each record's native scale — that invariance is the mode's whole
content and must therefore be *measured*, as `invariance.py` measures its matcher, not asserted.
The constraint that measurement already imposes: dividing by a modelled quantity imports that
quantity's bias, which is why `scale_normalised` is kept as a function and deliberately left out of
`MATCHERS` after the extractor's scale estimate was found to drift from +3.6% to -2.6% across a
sixfold range. A shape statistic that normalises by each record's *estimated* native scale would
inherit exactly that drift. The recommended form therefore divides one measurement by another of
the same kind, as `relative_geometry` does, rather than by an estimated scale; the deciding
evidence is an invariance measurement, and the alternative is admitted only if it survives one.

**Fixtures, by direct analogy with the four frozen calendar cases.** A planted case where one shape
genuinely recurs at materially different native durations, which must reject every member at the
declared alpha. A `same_normalisation_unrelated` safeguard — independent shapes put through the
identical normalisation — because standardising two smooth profiles to zero mean and unit variance
makes them correlate, and that is this mode's counterpart to calendar's `same_window_unrelated`. A
`native_scale_alias` safeguard where a coordinate coincidence arises from the scale grid rather
than from shape, the counterpart of `gap_alias`. And a `degenerate_inventory` case whose declared
family admits no valid reassignment, which must produce a refusal rather than a p-value of 1.0 —
the case D91 currently answers silently and wrongly.

**Slices.**

1. **D91.** Derange effective pairings rather than positions; refuse when an inventory admits no
   valid reassignment. Verified by re-running the probe that found it, and by a test asserting the
   refusal on two pairs sharing one right member.
2. **The statistic.** One shape-recurrence statistic over a declared correspondence, with its
   scale-invariance measured across a range of native durations rather than declared, and its
   sensitivity to the normalisation itself reported.
3. **The fixtures.** Four frozen cases with their expected answers written in the module, not the
   test.
4. **The calibration.** The family of three, run under the declared null at 999 replications with
   one Benjamini-Yekutieli correction, reporting planted power and the false-positive rate on the
   safeguards.
5. **The gate.** `scale_shape_calibration` moves off `NOT_IMPLEMENTED` to whatever it has actually
   earned. A calibration that runs and fails its power target is a passed slice and an unpassed
   gate; those are different facts and the record keeps them apart.

**What would falsify this task, stated before it starts.** If the planted case cannot be recovered
at the declared alpha once D91 is fixed, the honest conclusion is that scale/shape mode is not
measurable on this quartet — not that the fixtures need adjusting or the alpha relaxing. D83
already established that the frozen quartet cannot be qualified in this mode at all; this task may
end by establishing that the three domains which do admit the null still cannot support the claim,
and that outcome must be recorded as the result rather than engineered away. No case may be
retuned after its answer is seen.

**Claim boundary.** Nothing in this task is evidence about the world. It measures whether a
declared family behaves as declared on fixtures whose answers are fixed in advance. Passing it
licenses no mining, promotes no claim and admits no evidence.

**TG17.11 slice 1 — D91 fixed, and the declared families answered a different question than
expected (2026-09-03).** The null now enumerates the reassignments an inventory actually admits
rather than shuffling list positions. A substituted partner must be neither the partner the
pairing already had — the D91 fault, which returned the observation as its own surrogate — nor the
pairing's own left member, which would compare a record against itself at maximal similarity.
Assignments are deduplicated by the surrogate they *produce* rather than by the index permutation
that produced it, because where an inventory names one record in several pairings many
permutations spell one surrogate; counting them separately would overstate how much the null
explores and bias the draw towards whichever surrogate has the most spellings. The draw is then
uniform over the distinguishable surrogates. Enumeration is exact and bounded at eight pairings,
above which a uniform draw would have to be argued for rather than demonstrated; no declared
family approaches that bound.

Measured on the inventory that exposed the fault, six pairings over three right members each
appearing twice: **0 of 5,000 draws** leave a pairing unchanged and **0 of 5,000** pair a record
with itself, against 50% for both before. Four guards pin it, and all four fail when the validity
condition is mutated back to position derangement.

**Two findings about the declared families, which change the rest of this task.** They are
properties of the inventories, not defects, and they were invisible until the null was made to
enumerate what it could legally return.

* **Four domains compared all-against-all admit exactly one distinguishable reassignment.** Every
  valid permutation of that family's six pairings spells the same surrogate. Its null is therefore
  a constant, not a distribution: every replication returns the same value, so the p-value can only
  be the floor or 1.0 however many surrogates R18 prices and is paid for.
* **The three domains that admit the null after D83 admit no reassignment at all.** One of the
  three pairings has no substitute partner that is neither its own nor itself. There is no
  surrogate, so there is no test.

Both are now refused by name, each with its own reason, rather than answered. The consequence is
that the family sketched when this task was scoped does not exist: **a scale/shape calibration
cannot be declared over an all-pairs *domain* family at either size.** It must be declared over an
inventory of records, where the members of a pairing are individual frozen shapes rather than
domains, and where the inventory is checked against the null it will run before any statistic is
built on it. Slices 2 to 4 are rewritten on that basis; slice 3's `degenerate_inventory` fixture
is no longer hypothetical, since both declared domain families are now instances of it.

This is the falsification clause of this task's scope doing its work early and cheaply, on the
null rather than on the statistic. Had the statistic been built first, the calibration would have
reported near-zero planted power on a family that could not have produced any, and the honest
reading of that number would have been a redesign of the statistic.

**TG17.11 slice 2 — the statistic exists, and three of its properties are now numbers
(2026-09-03).** `src/benchmarks/shape_calibration.py` supplies `shape_recurrence`: a
support-weighted correlation over shared *phase* rather than shared seconds. Each record's
declared support is divided by its declared native duration, the two phase axes are intersected
by the same linear sweep `family_calibration.py` uses on calendar supports, and each contribution
is weighted by the phase actually shared. Binning both records onto a common phase grid would
have manufactured the correspondence under test, in the same way binning onto a common calendar
grid would manufacture simultaneity, so it is not done.

**Scale invariance is exact, and the reason is the denominator.** `invariance.py` records why
`scale_normalised` is kept as a function and left out of `MATCHERS`: it divides by the extractor's
*estimated* spatial scale, which drifts from about +3.6% to -2.6% across a sixfold range, and the
whole of that drift lands in the quotient. The denominator here is a record's declared
`StructuralScale.native_value` — a quantity the adapter states and the manifest carries, not one
this module infers from the data it is about to test. Nothing is estimated, so nothing drifts, and
the measurement shows it: re-presenting one record across the same sixfold native range moves the
statistic by at most **1.1e-16**, against 0.37% for `relative_geometry`. The test asserts below
1e-12 rather than below a tolerance chosen to fit. On a planted pair whose native durations differ
by a factor of 2,222 the statistic reads **0.99**, against **0.05** for an unrelated shape at the
same native duration.

That argument was not accepted on its own, and two further measurements say what it costs.

**Row density stops mattering, and where it stops is measured rather than picked.** Rewriting a
record at a different cadence over the same span moves the statistic by less than 1e-9 down to
about 16 rows per native cycle, then **2.7% at 8**, **11.7% at 4** and **15.6% at 2**. That
movement is not a failure of invariance and is deliberately not reported as one: two rows per
cycle cannot represent a second harmonic at all, so the lower agreement is real information loss,
and a statistic that reported no movement there would be inventing detail the rows no longer
carry. `measure_cadence_dependence` is therefore a separate function from
`measure_scale_invariance`, because a single sweep varying both would report one number for two
effects and make the invariance claim unfalsifiable. `MINIMUM_ROWS_PER_CYCLE` is set at 8 from
those figures and comparisons below it are refused by name rather than scored.

**A phase comparison is phase-locked, and that trade-off is now stated in advance.** A wrongly
declared native duration drifts the two records apart once per cycle, so its cost compounds with
how much was compared. A 10% mis-declaration costs **15%** of the statistic over one cycle,
**40%** over two and **93%** over six. Comparing more cycles buys statistical support and spends
tolerance to declaration error, and `measure_phase_window_tradeoff` puts that table in the record
so a study cannot choose its phase window — and with it its own sensitivity — by accident. This is
the honest counterpart of the "declared, not modelled" argument above: the phase axis imports no
estimator drift, but it does import the declaration's own error, and here is how much.

**A third finding, which settles what slice 3 must build.** Every G17 flagship record declares its
structural scale as its own row cadence, so each resolves exactly **1.00 rows per native cycle**.
A shape needs more than one sample per cycle to exist, so *none of the TG17.0 calendar fixtures
can be reused as a scale/shape fixture at its declared scale* — the comparison refuses before it
computes anything. Slice 3 must build records that declare a shape-bearing native scale rather
than borrowing the calendar fixtures, and a test pins the finding so that a later slice cannot
quietly borrow them anyway.

Eight guards. Three mutations were run against them: dividing the phase axis by a constant number
of seconds instead of the declared native duration fails seven of the eight; weighting by row
count instead of shared phase duration fails two; removing the resolution floor fails two. No
mutation left the suite green.

**TG17.11 slice 3 — the fixtures, and the inference that had to be corrected before they could be
scored (2026-09-03).** `src/benchmarks/shape_fixtures.py` builds the four cases and freezes their
expected answers in the module. Two things had to be settled first, and both were found by
measurement rather than by reasoning about the design.

**The family size is solved from the null's resolution, not chosen, and it is 105.**
`reassign_scale_partners` replaces a pairing's right member with another right member from the
same inventory, so a member's surrogate statistic can take only as many values as the inventory
has admissible alternative partners — exactly `k - 1` for `k` disjoint pairings. The p-value is
therefore bounded below by `1/k` however many replications are paid for, and Benjamini-Yekutieli
admits a rejection only where `1/k <= alpha/H_k`. `minimum_resolvable_family` solves that against
the real `adjust` rather than against arithmetic written into the module, and at alpha 0.05 the
answer is **105 pairings**: at 104 the most favourable result the null can produce — every member
beating every alternative — still rejects nothing. This is a hard floor on the mode. It is also
the quantitative form of the slice 1 finding: G17's declared families of three and four domains
are not merely too small, they are short by a factor of about twenty-five.

**The Monte Carlo template would have inverted the safeguards, and is registered and refused.**
Applying `family_calibration.py`'s method unchanged — draw 999 whole reassignments, count
surrogates reaching the observation, divide by 1,000 — reports 0.001 for a member that beats its
`k - 1` alternatives, when the exact tail probability of that event is `1/k`. The replications
resample the same handful of values, so the denominator asserts a resolution the null does not
have. The error is a factor of `k` **in the anti-conservative direction**: measured on a wholly
unrelated inventory of six pairings over 200 realisations, the Monte Carlo form rejects at a
family-wise rate of **74%** against a nominal 5%, at 1.11 false rejections per family, while the
exact partner test rejects at **0%**. Calendar mode is not affected and that is measured rather
than assumed: an independent clock shift over records of 56 and 1,344 rows returns 397 distinct
surrogate statistics in 400 draws, so each replication there is a genuinely new surrogate. The
distinction is whether the null's support exceeds the number of draws, which is a property of the
null rather than of the code, so `monte_carlo_partner_p_values` is kept, named and refused with
its measurement attached — the treatment `global_value_shuffle` already gets, for the same reason.

Had D91 not been fixed first, this second fault would have been hidden underneath it: the
conservative bias of the broken null would have masked the anti-conservative bias of the wrong
p-value, and the calibration would have looked approximately calibrated for two compensating wrong
reasons.

**The four cases, with their answers stated from what each case is.** `exact_partner_p_values`
scores each declared pairing by its rank among the alternatives that could legitimately have
replaced it, taking that reference set from `admissible_partners` — the same set the null draws
from, so the test and the null cannot disagree about what the family is.

* `planted_shape_recurrence`: each pairing genuinely shares one profile, presented at two native
  durations drawn independently across four orders of magnitude. **105 of 105 members reject**,
  in 20 of 20 independent realisations.
* `same_normalisation_unrelated`: independent profiles put through the identical standardization,
  which is what makes arbitrary smooth shapes look alike. Own pairings score 0.339 and
  alternatives 0.346 — indistinguishable, as required. **0 of 105 reject**, family-wise error 0%
  over 20 realisations.
* `native_scale_alias`: every record sampled at the same rows per native cycle and carrying the
  same artefact keyed to position within the cycle, the signature of a shared instrument cadence.
  Own pairings score **0.812** and alternatives **0.845**: a raw correlation that would look like
  a spectacular result under any threshold, and which the null absorbs completely. **0 of 105
  reject**, family-wise error 0% over 20 realisations. This is the case that shows the null
  earning its place rather than the statistic being weak.
* `degenerate_inventory`: every pairing names the same right record, so none has an admissible
  substitute. **Refused**, not scored — a p-value of 1.0 here would read as a safeguard passing.

Eleven guards, including one asserting the contrast with the calendar null so that the refusal
reads as the specific finding it is rather than a general suspicion of resampling.

**TG17.11 slice 4 — the calibration runs, every case meets its frozen expectation, and the design
states what it can and cannot resolve (2026-09-03).** `calibrate_shape_family` is the counterpart
of `family_calibration.calibrate_family` and deliberately reports more than it does, because this
mode's null has finite support and a p-value floor: a result saying only that the planted case was
recovered and the safeguards were not would omit the two facts a reader most needs.

**What it measured.** Each scoreable case is run over **20 independent realisations**, each as one
family of 105 corrected once under Benjamini-Yekutieli at alpha 0.05, with no replications at all —
the null's support is enumerated, and resampling it would claim a resolution it does not have.

* `planted_shape_recurrence`: **105 of 105 members reject in 20 of 20 realisations.** Full power.
* `same_normalisation_unrelated`: **0 rejections in 2,100 member tests**, family-wise rate 0.000.
* `native_scale_alias`: **0 rejections in 2,100 member tests**, family-wise rate 0.000.
* `degenerate_inventory`: **refused**, with its reason, rather than scored.

`all_met` is true. Twenty realisations bound a zero count only at 14% by the rule of three, which
is weaker than the alpha being claimed, so the same quantity is measured a second way where draws
are nearly free: under the global null a member's exact p-value is uniform on the lattice
`1/k ... 1`, so a whole family can be drawn without building a record. Over **20,000 draws the
family-wise false-positive rate is 0.000**, a one-sided 95% upper bound of **0.015%**. The two are
reported side by side rather than one standing in for the other, because the lattice measurement
treats members as independent and a shared statistic grid makes that only approximately true.

**The operating characteristic, which is the finding a study most needs and least expects.** Every
genuinely recurring member sits at exactly the same p-value floor, so there is no region of partial
power: at a given inventory size a family either rejects or it does not. The declared family is the
smallest that can reject at all, which puts it on a knife edge — it recovers a wholly recurring
inventory and nothing sparser. Measured at 105: with **104 of 105** pairings genuinely recurring
the family rejects **0.9%** of the time; with 105 of 105 it rejects 100%.

Detecting a sparser recurrence is not a matter of more computation. It is a larger inventory, and
`minimum_family_for_detected_fraction` solves each size against the real correction:

| fraction of the family genuinely recurring | smallest inventory that can detect it |
| --- | --- |
| 100% | 105 |
| 90% | 120 |
| 75% | 149 |
| 50% | 243 |
| 25% | 550 |
| 10% | 1,586 |

A study that declares 105 correspondences and finds that 90 of them recur reports **nothing**,
however strong each individual match is. That is a property of the null and the correction
together, it is now stated before anyone acquires anything, and it is the single most consequential
number this task produced.

**Why the Monte Carlo form is refused rather than documented as a shortcut.** Its error is a factor
of `k` and therefore shrinks as the inventory grows, while the correction's stringency grows with
it. The two cross, measured on wholly unrelated inventories: family-wise false-positive rate
**76.7% at k=6**, **50.0% at k=12**, **10.0% at k=30**, and **0.0% at k=60 and k=105** — against a
nominal 5%, with the exact test at 0.0% throughout. The wrong method is safe only at the inventory
sizes where the right method already works, and catastrophic at the handful-of-domains sizes anyone
would actually reach for. That is the shape of a trap rather than of an approximation.

Seven further guards, seventeen in the file. Two mutations were run against them: reintroducing a
replication denominator on this finite-support null fails two, and replacing the solved family size
with a chosen one fails two. Notably the replication mutation does *not* break the safeguards at
k=105, which is the same crossing measured above and the reason the guard that catches it asserts
the p-value floor directly rather than waiting for a safeguard to fire.

**TG17.11 slice 5 — the gate moved off `NOT_IMPLEMENTED`, and what it moved to is `REFUSED`
(2026-09-03).** The sentence the gate carried — that no registered scale/shape calibration produces
a scientific statistic — is now false, so it could not stay. What replaced it is not a pass. The
gate reads `REFUSED`, the status this apparatus already reserves for a declared scientific limit
that blocks release exactly as a failure does, and `scale_shape_applicability` computes the
determination in milliseconds without acquiring a record or running a calibration.

**The blocker is an inapplicable method, not an absent one, and it is two bounds that do not
meet.** From below, every member of a `k`-pairing family sits at a p-value floor of `1/k`, so
solved against the real correction at alpha 0.05 **no family smaller than 105 pairings can reject**
even when every member is a perfect planted match. From above, `reassign_scale_partners` enumerates
the reassignments an inventory admits exactly and **refuses above 8 pairings** rather than adopt a
sampler whose uniformity is assumed. The gap is an order of magnitude, so **there is no inventory
size at which the null as the qualification manifests declare it — drawn, with a replication count
— can produce a rejection at all.** What can is the exact partner test, which enumerates the same
finite support instead of resampling it; it is what the registered calibration is built on and what
no declared manifest requests. That is a sharper statement than slice 4's, and it is a property of
the declared null rather than of the fixtures, so no fixture work could move it.

G17's two candidate families never reach that argument: the quartet all-pairs admits exactly one
distinguishable reassignment and the post-D83 triple admits none, and both refusals are carried in
the qualification record verbatim from the null itself rather than restated in prose that could go
stale independently of the code.

**The two facts the slice existed to keep apart.** A calibration that runs and meets its targets on
built fixtures is one fact; whether a declared plan can reach the method is another. Reporting the
first as the second is exactly how an unusable mode acquires a green gate. The record therefore
names the calibration's entry point, states `calibration_executed_here: false`, and asserts only
quantities it computed itself — a guard fails if any power or rejection-rate key appears in that
section. A calibrated method the declared plans cannot reach, a method that does not exist, and a
method that ran and failed are three different facts.

**What this establishes about the task's own falsification condition, stated before it started.**
The condition was that if the planted case could not be recovered at the declared alpha once D91
was fixed, the honest conclusion would be that scale/shape mode is not measurable on this quartet.
The planted case *was* recovered — 105 of 105 members in 20 of 20 realisations — and the conclusion
about the quartet holds anyway, for a reason the condition did not anticipate: the method is sound
and the declared families are two orders of magnitude too small to use it. **Scale/shape mode is
not qualifiable on G17's declared families, and this is now recorded as the result rather than
engineered away.** Qualifying it needs a declared inventory of at least 105 correspondences and an
enumerated rather than drawn inference — a different family design, not a retuned fixture.

**Verification.** Five guards added, twenty-one in `test_experiment_qualification.py`, 25 passed.
Three mutations: awarding the gate `PASS` fails two, hard-coding the applicability verdict while
adding a power number fails two, replacing the null's own refusal text with a fixed string fails
one. The verdict remains `NOT_RELEASEABLE` and one fewer gate is unexplained. No archive is
acquired, no statistic runs here, no evidence is admitted and nothing is released by this slice.









**TG8.3 The domain ledger — finally measurable.** With five domains across three acquisition
shapes there is at last a trend to read. *If onboarding cost is not falling, the abstraction is
not working*, and the ledger must be able to say so.

#### Claim boundary, written in advance

*   A store in a catalogue is **not data ingested**. Each phase states plainly whether a live
    fetch has been run, in the way TG7.3's live tail is still recorded as NOT RUN.
*   A gridded ocean product breaks nothing new and must not be presented as evidence that the
    abstraction generalises. Argo and photometry carry that claim.
*   Reading a record under a domain never establishes that it came from that domain, however real
    the archive (`DOMAIN_ATTRIBUTION_CAVEAT`).
*   Network stays opt-in. Reaching the internet must never be a side effect of running a sweep.
*   Credentials are never written to an artefact, a log, a provenance record or a `repr`.

**TG17.12 Calendar calibration recorded into its release gate — DONE (2026-09-03).**
`calendar_calibration` was the last of TG17.10's seven gates whose `NOT_RUN` was true of the record
and false of the world. `family_calibration.calibrate_family` runs the frozen calendar family on
the TG17.0 fixtures and has always passed; nothing carried that measurement into the qualification
record. The gate still does not run it — a calibration is a scientific measurement and the gate is
a release gate, the same separation TG17.11 stated for scale/shape — so what this task builds is
the channel, in `src/core/calibration_record.py`.

A recording is bound to two digests and is read as *unrun* if either moves: the **declared
contract** (the cases and the rejection counts frozen with them, family size, alpha, correction,
replications, channel, null family and seed), digested from `CALIBRATION_CASES` rather than
restated so that relaxing an expectation cannot leave a stale pass agreeing with it; and the
**source** of the four modules that decide what the measurement is, digested with line endings
normalised so a Windows clone and a POSIX one agree. Four outcomes, three of them blocking: absent,
unbound from its contract and unbound from its source all read `NOT_RUN`, because a recording made
against something else is a measurement of a different thing rather than a weaker pass; a recording
whose cases missed their frozen answers reads `FAIL`, because a calibration that ran and failed is
a different fact from one that did not run.

Neither digest is tamper-evidence against an editor of this repository, and the source binding
covers four files rather than the whole import graph. That boundary is stated rather than hidden,
because the backstop is elsewhere: the live calibration already runs in the suite on every pass,
and a guard in `test_experiment_family.py` compares it case by case against what the gate is being
told, so a recording cannot drift from what the calibration actually does.

**What it recorded.** At 999 replications on a family of six, `shared_calendar_event` rejects
**6 of 6** after correction and `same_window_unrelated`, `gap_alias` and `inadmissible_precedence`
each reject **0**. `all_met` is true, so the gate reads **`PASS`** — the first of the seven
scientific gates to clear, and the verdict is unmoved at `NOT_RELEASEABLE`.

**The contrast with TG17.11, computed rather than asserted.** Both modes have a p-value floor and
buy it differently. Scale/shape buys it with domains — a `k`-pairing family cannot go below `1/k`,
and its draw refuses above 8 while its correction needs 105. Calendar buys it with computation —
the floor is `1/(1 + replications)` and there is no enumeration ceiling, because the surrogates are
clock shifts the record itself supports. Both configurations are checked against the real
correction: the calibration family resolves at **999 against 293 required**, and the calendar plan
the manifests actually declare resolves **4 corrected members at 200 replications against 166
required**. Two bounds that do not meet, against two that meet with room to spare.

Twelve guards in a new `test_calibration_record.py` plus the drift backstop, and the gate assembles
in **22 ms warm** with a guard failing above one second.

### Phase G18 — World-class scientific interface — **DONE (2026-09-04, `ed-dev`)**

All six phases TG18.0-TG18.5 are complete; TG18.5's close-out checked the phase against its own opening paragraph rather than against its slice list. G18 does **not** clear §6 condition 19 on its own: that condition names both the clean-browser no-glue test and the synthetic fifth-adapter test, and the second belongs to G17.

The engine is reachable, but reachability is not yet an instrument-quality interaction contract.
The rendered baseline was inspected at an ultra-wide desktop viewport across Acquire, Spectral
Transforms, Experiment Composer and Findings. The visual language is worth retaining: a restrained
dark field, teal selection, explicit status, numbered scientific workflows and claim-boundary
language. The deficits are shared ergonomics rather than a need for a new aesthetic: undersized
secondary text, weak use of wide screens, stretched control cards, warnings that read like log
lines, a research-context bar too quiet to function as orientation, and empty states that occupy a
large canvas without naming the next legitimate action.

This phase changes presentation and navigation only. It may not recompute, summarize, promote or
reinterpret a scientific value; backend-authored claim language remains verbatim under R22/R23,
and capability/refusal decisions remain server-owned under R25.

**TG18.0 Rendered baseline and interaction inventory — DONE (2026-09-02).** Preserve the four
distinct product modes identified in the baseline: interactive instrument (Spectral Transforms),
guided commitment workflow (Composer), read-only claim surface (Findings), and trust/qualification
surface (Platform & evidence). A change that improves one by making another ambiguous is not a
successful redesign.

**TG18.1 Shared instrument foundation — DONE (2026-09-03).** Establish one stable application frame,
readable type/contrast tokens, an independently scrolling workflow rail, a wide-screen workspace
that uses panes instead of stretched forms, a prominent persistent record/study context, standard
surface and status treatments, reduced-motion compliance, and purposeful empty states. The first
slice must improve all supplied baseline views without changing an API request or response.

**First slice delivered (2026-09-02).** The shared shell now has instrument-level typography and
surface tokens, an independently scrolling workflow rail, bounded wide-screen content, structured
notices, and actionable empty Findings. Record/study context persists in application state but is
shown only when populated and does not create a second sticky header. A new Research Archive
indexes the existing study, run, gate, evaluation, acquisition-probe and benchmark ledgers while
keeping their evidence classes visibly distinct. Acquire now exposes researcher-facing source
identity and names the implemented CDS downloader as `PLANNER_NOT_EXPOSED` instead of silently
omitting it. This does **not** complete TG18.1: the CDS browser planner/job surface,
narrow-width rendered inspection and cross-workspace density tuning remain. Route-level bundle
splitting is explicitly **not required** (2026-09-03): this local research instrument may ship a
large bundle unless measured startup or interaction latency establishes a user-visible defect.

**Second foundation slice delivered (2026-09-03).** The shell now has an accessible compact-width
workspace menu with explicit expanded state and automatic closure after selection; the desktop
workflow rail and workspace scroll independently inside the viewport; record/study context wraps
without clipping on narrow screens; and connection startup is rendered as `Checking API` rather
than briefly misreported as an outage. Navigation metadata is no longer set below 10 px. TG18.1
remains in progress for rendered narrow-width inspection, cross-workspace density tuning and the
CDS browser planner and durable job surface. The next slice below completes the planner half.

**Third foundation slice delivered (2026-09-03).** Acquire now exposes the existing CDS route as
a real metadata-only planner. Every variable, date, UTC hour, bound, pressure level, grid spacing
and analysis depth is editable; the backend uses the production `CDSRegionalRequest` to return the
immutable digest, monthly work units, exact frame/grid geometry and a no-compression-credit storage
ceiling. The accepted six-year request reproduces 72 shards and 8,764 frames. Both API and UI state
plainly say planning uses no network and execution is `NOT_MOUNTED`; a valid plan cannot be mistaken
for a submitted job or acquired data. TG18.1 remains in progress for the durable CDS job/progress/
resume surface, rendered narrow-width inspection and cross-workspace density tuning.

**Fourth foundation slice delivered (2026-09-03).** The CDS planner/job portion is complete. An
exact request digest plus affirmative network acknowledgement is required before submission; the
server accepts no client path, repeats the conservative free-space preflight, and journals bounded
monthly progress atomically below server-owned storage. Cancellation is cooperative between
monthly requests and preserves verified work. Jobs left active by process loss become explicitly
`INTERRUPTED` and can be resumed without repeating verified shards. Only `COMPLETE` creates a
self-hashed acquisition record, which is labelled transfer/integrity provenance rather than
analysis or evidence. The browser exposes job history, progress, the in-flight shard, stop/resume,
preflight status and the completion record while keeping a validated plan visibly inert. The
embedded worker registry is deliberately qualified for one API process; a multi-process deployment
still requires an external queue. TG18.1 remains in progress only for rendered narrow-width
inspection and cross-workspace density tuning.

**Fifth foundation slice delivered (2026-09-03).** Cross-workspace density is now a shared shell
contract rather than a panel-by-panel accident: one centred scientific canvas on wide displays,
readable metadata and plot-label floors, consistent control height, compact mobile spacing, safe
wrapping for content identities, and one-column reflow below 480 px. The conditional record/study
context remains pinned beneath the instrument header while the workspace scrolls. The compact menu
is now a fixed scrimmed drawer with background-scroll lock, current-workspace focus, a Tab loop,
Escape and outside-click dismissal, and trigger focus restoration. Connection state collapses to
an accessibly named icon below 400 px so the 320-pixel header does not overflow. All 171 frontend
contract tests and the production build pass. Rendered narrow-width inspection was **NOT RUN** in
that slice: the shared-browser runtime returned no available session after reconnect. The slice
below runs it.

**Sixth foundation slice delivered (2026-09-03) — TG18.1 closed.** The rendered narrow-width
inspection is run, in Chromium, at 320, 375, 414 and 768 CSS pixels across all four product modes.
The blocker in the fifth slice was the shared browser runtime, not the absence of a browser: the
repository already carries a Playwright install and a Chromium binary, and `narrow-width.spec.ts`
now sits beside the four existing acceptance specs under the same config.

*It found two defects the source contract and the production build had both passed*, which is the
whole argument for the task. The shared canvas rule `.workspace-main > *` also matched the two
`sr-only` children; overriding their one-pixel clipped box with a real width and `margin-inline:
auto` let those absolutely positioned elements escape the workspace's clipping, so **every**
workspace scrolled horizontally by 14 px at 320 px. And the below-480-px reflow collapsed the track
count but not `col-span-*`, so a spanning child rebuilt the second column as an implicit track while
the declared template still read as one. A third, smaller finding: the 11/12-px metadata floor
covered `text-[10px]` and `text-[11px]` and never `text-[9px]`, which Acquire, the lineage nodes and
the capability profile all use. All three are fixed and now carry source assertions as well.

The inspection measures the laid-out document rather than restating the CSS: document scroll width,
content past the right edge that no ancestor scrolls, computed font size on every text-owning
element, rendered control height, header containment, and both the used track count and the rendered
row occupancy of every collapsed grid. It drives the drawer end to end — scrim, scroll lock, focus
placement, Tab loop, Escape, outside click, focus restoration — and captures 21 named viewport
artefacts.

**Evidence.** The complete Chromium suite is **54/54 from a cleaned `.e2e-state`** (35 before this
slice, 19 added). `test_frontend_contract.py` and `test_documentation.py` are **197 passed**. The
production build succeeds. The full backend suite was **not** rerun for this slice; the last
measured full-suite figure is **3536** (T4E.2, on this same tree).

**Stated boundaries, so the closure cannot be read as more than it is.** The sticky record/study
context is not covered: it renders only when a record or study is selected, and no workspace this
inspection reaches selects one, so its pinning stays a source-level contract. Plotly label floors are
not covered, because those axes exist only after a transform has run against real data. Below 352 px
the drawer fills the viewport and outside-click dismissal is unavailable by construction — Escape,
the trigger and selection remain, which is why none is treated as optional. This is not an
accessibility conformance audit and does not pre-empt TG18.4; measuring a font size is not certifying
a contrast ratio with a screen reader in the loop.

**TG18.2 Scientific visualization workspace — DONE (2026-09-03).** Add coordinated plot focus, exact-value
inspection, shared colour/axis controls, comparison locking, uncertainty and validity overlays,
resizable panes, and publication/export affordances. Every visual encoding must have a text/table
equivalent and must state units, support, normalization and missingness where they apply.

**First slice delivered (2026-09-03): the figure data contract.** This opens the phase on the
standing gap rather than on new features. Every gridded panel renders through exactly two
components, so the text/table equivalent is implemented once at that seam and reaches all fifteen
call sites. Both previously carried an `sr-only` caption describing the *shape* of the data and
nothing else; the values were reachable only through a hover tooltip, which is mouse-only,
ephemeral, and absent from every exported or printed copy.

The design decision that matters is the boundary, because G18 may not recompute or summarize a
scientific value and the obvious implementation — mean, median, slope and correlation under every
plot — would breach that immediately. The rule adopted is **transcribe what the figure encodes, and
state what it could not encode.** A sample's value and an axis or colour-bar range are already on
the figure, so restating them is transcription; the heat map therefore reports the range *shown on
this figure* and names whether the limits were supplied for comparison or derived from that panel
alone. Non-finite samples and points a log axis discards are what the encoding silently omits — a
gap in a line reads as an absence of structure, and Plotly drops non-positive samples without a
mark — so those counts are part of the contract. A mean is neither, and is refused; the refusal is
printed on the page, not kept in a comment.

Line charts are enumerated point by point with a stated cap at 2,000 rows. A field cannot be, so
the heat map equivalent is addressed rather than listed: name a row and column, read the exact
sample with its coordinates, units and validity. The control clamps to the field instead of
accepting an index it cannot answer, and the panel states the cell count it declines to tabulate.

**Evidence.** The Chromium suite is **63/63 from a cleaned `.e2e-state`** (54 before this slice, 9
added). Its load-bearing assertion is agreement: the exact sample is read out of the live Plotly
trace and compared with what the panel printed, because plausible numbers unrelated to the trace
would pass every structural check. The refusal is asserted in the markup as well as the prose — no
contract term may be labelled Mean, Median, Slope, Correlation or Standard deviation.
`test_frontend_contract.py` and `test_documentation.py` are **198 passed**; `tsc --noEmit` is clean
and the production build succeeds. The full backend suite was **not** rerun; the last measured
full-suite figure is **3536** (T4E.2).

**Not yet done in that slice.** Coordinated plot focus, shared colour and axis controls, comparison
locking, uncertainty and validity overlays, resizable panes and publication export were untouched.
The equivalent also does not cover figures rendered by the comparison views, which carry their own
`AccessibleTable`; the two idioms are **deliberately not unified**, because `AccessibleTable`
transcribes a backend-authored table that the analysis layer stands behind while `FigureTable`
transcribes a client-side figure encoding, and merging them would erase exactly the provenance
distinction the platform exists to preserve.

**Second slice delivered (2026-09-03): the comparison contract.** Coordinated focus, shared colour
and axis controls and comparison locking are one requirement wearing three hats, so they are built
as one contract rather than three widgets.

The defect closed here was structural. Plotly autoscales each panel to its own extremes unless
given explicit limits, and `zRange` was supplied at exactly **one** call site in the entire
frontend. Every other side-by-side pair rendered on independent scales — including original target
against inverse reconstruction, which *is* an error judgement: under independent autoscaling a
reconstruction that lost most of its amplitude produces a near-identical picture, the discrepancy
surviving only in two small colour-bar ranges.

Forcing a shared scale everywhere would trade a silent error for a louder one, so the module
decides whether one is **admissible** and states the reason above the panels. It refuses on
mismatched units (R19–R21: raw magnitudes never share an axis), mismatched quantity, an undeclared
relationship, a panel with no finite sample, and a group of one. The relationship cannot be
inferred — two unitless fields are not related by being equally unitless — so the call site
declares a `quantity` key and omitting it refuses. A pair becomes comparable only through an
explicit, reviewable claim in the source.

Scale comparability and cell correspondence are decided separately: the padded-boundary pair keeps
its shared colour range, which is what makes padding's effect on magnitude visible, while linked
addressing is refused because the grids differ in shape and says so. Where grids do correspond, one
address drives every panel's cell inspector at once.

**Evidence.** The Chromium suite is **75/75 from a cleaned `.e2e-state`** (63 before, 12 added).
The decision function is exercised directly through the dev server's module graph, because the
mismatched-units, mismatched-quantity and undeclared-relationship branches are not reachable
through the current UI where every declared pair agrees — verifying them only on screen would leave
the refusals unchecked until a future call site needed them. The rendered half confirms both traces
carry identical explicit limits rather than two autoscales. `test_frontend_contract.py` and
`test_documentation.py` are **199 passed**; `tsc --noEmit` clean and the production build succeeds.
The full backend suite was **not** rerun; the last measured full-suite figure is **3536** (T4E.2).

**Third slice delivered (2026-09-03): the validity and uncertainty overlay.** These belong to the
same "what this picture cannot tell you" family as missingness, so they extend the figure contract
rather than sitting beside it.

The gap was concrete and entirely backend-authored, which is what made it addressable under G18 at
all. The PSD chart draws every wavenumber bin; the exponents quoted for it are fitted over
`[k_min, k_max]` using `n_points` of them under a stated `weighting`, with a standard error, an
R-squared and an explicit `assumptions` list. On the running platform the fit band is `0.649` to
`pi rad/pixel` while the figure draws from `0.237`, so roughly the lowest third of the plotted
abscissa lies outside the fit and nothing said so. The assumption strings - isotropy averaged over
annuli, a single unbroken power law, the reporting convention - were returned by the API, typed in
`api.ts`, and rendered **nowhere at all**.

The band is now marked on the figure and the numbers stated with it, outside any disclosure, since
a restricted domain governs how the whole curve may be read. Every value is transcribed and the
claim language is carried verbatim (R22, R23); `regime_interpretation` is deliberately not carried,
because it already appears where the exponent is quoted and a conclusion repeated beside a picture
hardens into a caption.

**The refusal is the load-bearing half.** The fitted power law is not drawn, and no plus/minus one
sigma envelope is drawn around it. Both would require the view to evaluate a model at every plotted
abscissa, and a curve rendered by the browser is indistinguishable on screen from measured data -
the mean-under-the-plot temptation of the first slice in better clothes. The refusal is printed on
the page and asserted mechanically: no `Math.exp`, no `Math.pow`, no `intercept_ln_c` in the module.

Refusals cover a degenerate band, limits that are not finite, and a band lying entirely off the
drawn extent - the last mattering most, since a silently absent band is indistinguishable from a
fit that spanned the whole figure. A band overrunning the figure is clamped, not merely described,
which also keeps `log10(0)` away from a logarithmic axis. Shape coordinates are projected into log
space when the axis is logarithmic, because Plotly reads them as `log10` of the value and a fitted
spectrum is read on log-log axes. The shading has a text equivalent: tabulated points carry an
*in declared domain* column, with three states rather than two, because a point outside a band and
a point on a figure with no band are different facts.

`Heatmap2D`'s `validInset` is the precedent this generalises and was deliberately **not**
retrofitted: its only caller draws six panels per level and already states the inset once at level
scope, and an inset is the intersection of two axis bands, so per-axis shading would draw a cross
where a box is correct.

**Evidence.** 19 e2e tests added, **19/19 passing**; `test_frontend_contract.py` and
`test_documentation.py` **200 passed**; `tsc --noEmit` clean. One real prose defect was found by
the rendered half and fixed: the sentence that samples outside the band were not used appeared only
on the unclipped branch, which is precisely the branch the platform's own spectra do not take. The
full backend suite was **not** rerun; the last measured full-suite figure is **3536** (T4E.2).

**Fourth slice delivered (2026-09-03): resizable comparison panes.** The three gridded pairs
already governed by the comparison contract now share one bounded two-pane canvas. The divider
changes presentation only: both figures stay mounted, their shared scale and linked address do not
move, and the page states that boundary. Pointer drag, arrows, accelerated Shift+arrow steps,
Home/End, Enter/Space and double-click are supported through an accessible separator exposing the
current 25--75 percent allocation. Below 768 px the control disappears and the figures return to
one-column document order, because a working separator with no usable horizontal canvas would be a
false affordance. The component notifies Plotly after a track change but never reads or rewrites a
trace.

**Evidence.** The six new rendered tests pass: they measure both pane rectangles after a real drag,
exercise every keyboard bound and reset, compare the two live Plotly data arrays and their shared
limits before and after resizing, and verify narrow-width stacking. The production build succeeds
with 1,416 transformed modules. Focused source/documentation verification is **202 passed** after
the checked inventories were advanced. The complete Chromium inventory is now **100 tests**; the
full suite has not been rerun as one command for this slice. The full backend suite was not rerun;
the last measured full-suite figure is **3536** (T4E.2).

**Fifth slice delivered (2026-09-03) — TG18.2 closed: publication export.** PNG and SVG remain
picture-only conveniences. Every heat-map and line-chart call site now also exposes one
self-contained, printable publication HTML sheet containing a 1200x800 vector snapshot of the
live Plotly figure, its caption, the fully evaluated figure contract, producer-authored validity
and uncertainty statements and their verbatim qualifiers, the no-analysis boundary and an export
timestamp. Export evaluates the deferred missingness scan only when requested, reads the mounted
figure without mutating its trace or layout, contains no script and has no path into the claim
ladder. This is why the pre-existing three PNG/SVG controls did not already satisfy publication
export: an image without the restricted fit domain, uncertainty, assumptions or scale provenance
is not the figure this instrument asked the researcher to read.

**Evidence.** Four rendered download tests pass. They open the generated files and verify the
vector, caption, exact grid and missingness, shared-range provenance, fit domain, uncertainty,
assumptions, no-analysis statement and live-figure immutability. Both figure families acquire the
action at their shared seam. `test_frontend_contract.py` pins the same refusal at source level;
the production build succeeds with 1,416 transformed modules. The complete cleaned Chromium suite
is **104/104**. The focused frontend/documentation pair is **202 passed**. The full backend suite
was not rerun; the last measured full-suite figure is **3536** (T4E.2).

**Closure boundaries and recorded visual debt.** The validity overlay is
declared at one figure (the PSD chart), because it is the only figure whose backend record carries
a fit domain and an uncertainty; no other chart has one to state. The comparison contract governs
the three declared gridded pairs in `App.tsx`; the `DTCWTScientificView` shared range predates it
and has not been migrated, and no line-chart group declares a comparison contract yet, because
axis-range sharing across line charts raises a separate question about log scales that these slices
do not answer. The PSD chart is also still drawn on **linear** axes, which is a genuine weakness now
that the fit band is visible on it - a power law is a straight line only on log-log - but changing
the axes alters the figure rather than describing it, so it is recorded here rather than folded in
silently.

**TG18.3 Guided research journey — DONE (2026-09-03).** `Acquire -> Inspect -> Design -> Run ->
Compare -> Admit -> Report` is now a visible global map above every workspace. It is explicitly
navigation, not evidence status: Design, Run and Compare hand off to the named Composer panel,
while Composer continues to obtain the scientific order, every status, every blocked reason and
the one next legitimate action from its served path contract. The existing acquired-material,
executed-run, finding and admitted-evidence ladder remains visible on Compare, and the map states
that it neither advances nor replaces it.

The shell owns only two ordinary context blockers. Inspect without a selected record names
`Acquire a record` as its one remediation; Admit without a selected study names `Open Composer and
save a study`. Neither is a scientific readiness verdict. Eight legacy gridded workspaces remain
reachable in the original workflow rail and are distinguished from the evidence journey twice:
the non-colour label `Legacy · Gridded field line` and an amber inset/background treatment.

**Evidence.** Five rendered acceptance tests verify the seven ordered stages, both blockers and
their performed remediation, Run and Compare landing on the corresponding served Composer panels,
continued visibility of the distinct claim ladder, direct Report navigation, and all eight legacy
tools remaining labelled and reachable. The existing Composer and narrow-width suites pass with
the new map (**35/35** together). The complete run from a cleaned `.e2e-state` is **109/109** in
Chromium. `test_frontend_contract.py` is **177 passed**. The production build transforms 1,417
modules. The full backend suite was not rerun; the last measured full-suite figure remains
**3536** (T4E.2).

**TG18.4 Responsive and assistive-technology acceptance — DONE (2026-09-03).** TG11.6 remains
the source-level contract. Eleven new Chromium checks add rendered inspection at desktop (1440 x
900), laptop (1024 x 768) and narrow (375 x 667) layouts: the skip link is the first keyboard stop,
the global focus indicator is measurable, fragment navigation and workspace changes focus the
named workspace heading, and the compact drawer returns focus after Escape. A 640 CSS-pixel/device
scale two inspection covers the effective reflow of a 1280-pixel viewport at 200% zoom without
horizontal document scrolling. Computed foreground/background contrast is checked at the normal-
and large-text thresholds; reduced motion is emulated and its durations measured; and a rendered
colour-removal pass leaves the current journey stage, both blockers, their remediations and the
claim-ladder boundary readable in words and semantics.

The first keyboard run found and closed a real defect: React StrictMode replay focused the hidden
workspace heading on mount, stealing the first Tab stop from the skip link. Focus routing now
compares the actual previous/current workspace, while the skip link explicitly focuses the named
heading after navigation. The new suite is **11/11**; TG18.4 plus its narrow-width and journey
compatibility boundaries are **35/35**; the complete run from a cleaned `.e2e-state` is **120/120**
in Chromium. This is bounded browser engineering acceptance, not a screen-reader audit or WCAG
certification. `test_frontend_contract.py` is **179 tests** and the Python inventory is **3161**.
The full backend suite was not rerun; the last measured figure remains **3536** (T4E.2).

**TG18.5 UI qualification gate — DONE (2026-09-04, `ed-dev`).** A clean-browser run exercises one
representative path through each product mode, captures named viewport artefacts, asserts that
every served route remains reachable, and records action count and refusal-to-remediation time. A
production build and the existing no-glue Composer suite are necessary but not sufficient
evidence.

The gate is a suite, not a surface. Nothing it measures is reported back inside the product; a UI
that grades itself on screen publishes a claim about the UI, and the claims this programme
publishes are about the science.

Its scope is set by TG17.10 rather than by presentation. That task shipped `NOT_RELEASEABLE` with
five unpassed gates, two of which name this work directly: `browser_no_glue` reads `NOT_RUN`
because "a deterministic backend rehearsal cannot observe a rendered browser and must not award
itself a gate on someone else's evidence", and `scientist_actions` reads `NOT_MEASURED` for both
the action count and `refusal_explanation_time_seconds`. TG18.5 therefore has to supply a
trustworthy channel from a rendered run into the qualification record. A missing, stale or
digest-mismatched artefact must read `NOT_RUN`; the ledger may ingest a measurement with its
provenance and may never synthesize one it did not receive.

**First slice delivered (2026-09-03) — served-workspace reachability.**
`frontend/e2e/ui-qualification.spec.ts` asserts the shell serves exactly the qualified inventory
in order, that all twenty workspaces open from a clean browser and name themselves, that every
reachable journey destination lands on an inventoried workspace rather than the fallback heading,
and that a clean browser disables nothing and claims no reason it is not entitled to.

**The slice found a coverage hole rather than a defect, and the distinction matters.**
`ResearchJourney.tsx` holds its seven destination identifiers separately from `WORKFLOW_NAV`.
Inspect and Admit are blocked in a clean browser and correctly substitute a remediation for their
own action, so their destinations are never clicked by any test in the repository. Renaming the
journey's `domainWorkbench` target was confirmed to pass the entire rendered suite — TG18.3's own
journey tests included — while sending a researcher who had selected a record to a heading reading
"Scientific workbench workspace". Nothing is currently broken; nothing was guarding it either. A
static cross-check now resolves all seven destinations and every `JOURNEY_STAGE_BY_WORKSPACE` key
against the served identifiers, and was verified to fail on that mutation and pass without it.
Both new static guards were mutation-checked; the rendered inventory and self-naming tests were
confirmed to fail on a workspace rename and the reachability tests correctly to stay green.

**Second slice delivered (2026-09-04) - one representative path per product mode.**
`frontend/e2e/product-modes.spec.ts` walks the characteristic path of each of TG18.0's four modes
and captures a named artefact at the state each reaches, at 1440 and 1920 CSS pixels - the desktop
widths the narrow-width inspection does not cover, so the two files now span 320 to 1920.

The load-bearing assertion is not any of the four paths. TG18.0's constraint is a property of the
modes *together*, so each mode declares a signature - found by role and accessible name, never a
class or a test id - and the suite asserts every signature appears in **exactly one** of the four.
Four per-mode checks could each pass while the modes converged on one another; this cannot.

Each path also asserts one thing its mode must not have. The instrument recomputes on demand and
must still deliver both figure-data equivalents. The commitment workflow serves seven ordered
steps and exactly one next legitimate action naming its own route. The claim surface opens every
panel and offers no way to compute or freeze one. The trust surface shows both a cleared gate and
an uncleared one with its reason - TG17.12 made the first `PASS` available to show.

Nothing asserts a pixel, so an artefact cannot pass or fail anything. Artefacts land in the
gitignored `e2e/artifacts/`, which sets a constraint on the remaining evidence slice: what reaches
the ledger must be a manifest and digest of the artefact set, never the images. Three mutations
against the product source were each caught: a compute affordance on the claim surface fails 2 of
10, relabelling the findings tablist as the composition path fails 4 of 10 including the uniqueness
assertion, and removing the qualification matrix's accessible name fails 4 of 10.

**The slice's full-suite measurement found an intermittent failure in an older spec.**
`composer-path.spec.ts` waited on a heading name that Playwright matches by substring, and the
shell's sr-only "Experiment Composer workspace" heading matches it too, so the locator resolved to
two elements once the served panel rendered and strict mode failed the run. It failed once in a
134-test run and passed 11 of 11 on isolated re-run. Made `exact`; 33 of 33 across three repeats.
Nothing in the product was wrong, but an intermittently failing suite is exactly what the remaining
evidence slice cannot carry: it would produce an intermittently blocking gate, indistinguishable
from a real refusal at the moment a reader most needs to tell them apart.

**Third slice delivered (2026-09-04) - the action count and the refusal-to-remediation
measurement.** `scientist-actions.spec.ts` measures both in a rendered browser, and the design
turns on which of the two an assertion may hold. An action count is deterministic, so it is
asserted: **14 actions** from a clean browser to a `COMPLETE` run of the frozen plan, and **3** to
reach the preflight refusal with **4** more to clear it. Every action is one activation of one
visible control located by role and accessible name, counted by the walk itself rather than
maintained by hand beside it.

A wall-clock duration is not deterministic, so it is recorded and asserted by nothing. How long a
refusal takes to explain itself is a property of the machine that ran the suite; asserting it would
fail the gate for reasons unrelated to the interface, and would let it pass on a fast machine while
the interface got slower. The committed recording carries 0.264 s and 2.835 s as unasserted context. Those are not
the 0.3 s and 4.74 s first measured: slice 4's full-suite run re-recorded them on a
differently loaded machine, so the numbers moved while the interface did not. A gate
asserting either would have failed on that alone, which is the argument for the design
rather than an illustration chosen after the fact.

The count is an upper bound on the shortest route, not a claim about a minimum, and the measurement
says so in its own claim boundary. `adapter_specific_framework_edits` is untouched: it is a
source-edit audit belonging to `synthetic_fifth_adapter` and no browser can observe it. The
measurement lands in the committed `measurements/scientist_actions.json`, because the remaining
slice has to read it. Three mutations against the product were each caught: an extra confirmation
grows the path and fails the count, renaming a control fails both tests rather than silently
finding another route, and rendering the refusal without the domain it refuses fails the refusal
measurement.

**Fourth slice delivered (2026-09-04) - the evidence channel into the qualification ledger.**
TG17.10 registered two gates it could not award itself, and TG18.5 set its own constraint for
closing them: the ledger may ingest a measurement with its provenance and may never synthesize one
it did not receive. Recording and deciding are separate and separately owned.
`frontend/e2e/qualification-reporter.ts` writes `measurements/browser_run.json` - every test that
ran with its spec and outcome, the artefacts with their digests, and the source digest of every
spec in the suite - and decides nothing. `src/core/browser_evidence.py` decides, and every decision
can be a refusal.

Two bindings, and the second is the one that was actually needed. Weakening a spec returns the gate
to `NOT_RUN`, because a recording is a measurement of a particular set of assertions. And a
*partial* run is refused: running one spec is the normal way to work on a test, and Playwright
reports it as `passed`, so the reporter records both the specs that ran and the whole inventory it
found, and the gate refuses when they disagree, naming every spec that did not run. That refusal
was verified before the first full run - a green four-test invocation read `NOT_RUN` and listed the
other fourteen specs.

The outcomes stay apart: absent, stale or partial means the run has not happened for this code and
reads `NOT_RUN`; a run that happened and failed reads `FAIL`; only a complete, clean run of the
suite this checkout contains reads `PASS`. The counts from slice 3 are restated in the module so a
*drifted* count reads `NOT_MEASURED` rather than being reported as the new number, wall-clock is
carried only under names ending `_unasserted`, and `adapter_specific_framework_edits` stays
unmeasured because no rendered run may award a source-edit audit.

Measured: **136 of 136** across all 15 specs from a cleaned `.e2e-state`, so `browser_no_glue` reads
**`PASS`**. Four mutations were each caught: accepting a partial run, dropping the spec-source
binding, collapsing `FAIL` into `NOT_RUN`, and reporting a drifted count instead of refusing it.

**The close-out (2026-09-04), and the condition it found unguarded.** The five declared scope items
are delivered and were checked against the declaration rather than against the commit log. Three
findings are recorded rather than resolved by rewording.

*The phase's own "nothing it measures is reported back inside the product" is narrower than what
shipped, so the declaration is amended and the code is not.* The trust surface renders every gate's
detail, so `browser_no_glue`'s cleared detail now says on screen that a rendered run of all fifteen
specs passed 136 tests. No UI-quality surface exists — the `scientist_actions` counts are declared in
`api.ts` and read by no component, so 14/3/4 and both durations appear nowhere in the product — but
one gate in the release registry that has always rendered now carries a cleared basis. Suppressing it
while continuing to show every refusal's reason would make a `PASS` *less* inspectable than a
refusal. The line actually held is: no UI-quality surface, and no UI-quality measurement outside the
release registry's own verdict and the basis for it.

*The plan is no longer uniformly cold.* Three gates read recordings at assembly time
(`browser_no_glue` and `calendar_calibration` `PASS`, `scale_shape_calibration` `REFUSED`) while
`offline_matrix` and `restart_recovery` stay `NOT_RUN` until a run fills them. Both kinds block
release identically and mean different things: a measurement this checkout has not received, versus
a run this call did not perform.

*Condition 19 is half-met and this phase does not clear it.* §6.19 requires both the clean-browser
no-glue test and the synthetic fifth-adapter test. The first passes; `synthetic_fifth_adapter` is
`NOT_RUN`, so G17 completion stays unclaimable and the verdict is unmoved at `NOT_RELEASEABLE`.

*The substantive finding is about this programme's own bookkeeping.* `roadmap.md` §10.2 admits no
claim of completion without recorded output in `VERIFICATION.md`, and nothing enforced it: TG17.11,
TG17.12 and TG18.0 through TG18.4 all reached a terminal state with that file silent about them,
while the atmospheric line continued to be recorded correctly. The seven entries are backfilled from
the commits that recorded them, transcribed rather than re-measured, because re-measuring today would
attribute this tree to a phase that closed against an earlier one. Two guards now hold the two
directions: a dated phase marked complete must have an entry, and a phase still in progress must not.
The rule keys on the date the heading carries, so the undated seam phases (TG0.x-TG2.x), which were
recorded under the slice headings, are exempt by construction rather than by a list.

TG18.5 is complete. G18 is complete.

**TG17.13 The source-edit audit and the fifth-adapter gate — DONE (2026-09-04, `ed-dev`).**
`synthetic_fifth_adapter` is one of the two gates still blocking release, and the reason it was
recorded `NOT_RUN` turns out to have been wrong. TG17.10 said a deterministic backend rehearsal
must not award itself a gate only a source-edit audit can measure. The real reason is that no such
audit existed: TG17.3's acceptance test is named
`test_synthetic_fifth_adapter_reaches_the_registry_and_conforms_without_framework_edits`, proves
the first half of that name, and asserts nothing about the second. "Without framework edits" was a
claim carried in a test name - the same shape as TG18.5 slice 1's finding, and found the same way,
by asking what a passing test would still allow.

**First slice delivered (2026-09-04) — the audit, and the number it produced.**
`src/core/extension_audit.py` separates two measurements that must not be run together. The
installation claim is absolute: no framework source may name the synthetic fifth adapter, no
declaration may excuse one that does, and the gate turns on this. It holds -
`installation_required_framework_edits` is **0** across seventeen framework sources. The standing
glue count is reported rather than asserted, because "glue must trend to zero rather than merely
move files" is a property of the whole surface over time and not of one installation; a count that
blocked release would make an unrelated archive's acquisition semantics a release decision, and a
count that went unpublished would let glue accumulate behind a green gate.

Every one of the eighteen occurrences where a framework source names a registered domain is
declared with a kind and a reason, or the audit refuses. Only a behaviour branch counts as glue: a
source that names a domain in prose, or carries a named recipe's own content, has not been edited
to make that domain work. **The count is 1.** `AcquisitionView.tsx` renders `CDSPlanner` behind
`domainName === 'reanalysis'`, and Copernicus acquisition being a long-running job the generic
control schema cannot currently express is a reason the glue exists rather than a reason it stops
being glue.

Two things were found by writing it. `AdapterControls.tsx` claimed there was deliberately no
`domain === 'reanalysis'` branch *anywhere* - true of that file, false of the surface, and pointing
a reader away from the one place such a branch lives; narrowed to what it can support with the
exception named. And **the audit's own first version passed vacuously**: it read the registry cold,
before anything loads the adapters, scanned for an empty set of names and reported a clean surface.
An empty registry is now a refusal, because that is D64, D74 and D75 a fourth time - a guard passing
because it could not see what it was checking.

Four mutations were each caught: ignoring an undeclared occurrence, counting prose and recipe
content as glue, scanning an empty registry instead of refusing it, and permitting the fifth
adapter's name in a framework source.

**Second slice delivered (2026-09-04) - the channel, and the half of it that must not exist.**
`src/core/extension_evidence.py` carries the measurement into the ledger, and the design decision
is which half it records. The **source-edit audit is read live**, on every call, from committed
source: it is exact, it costs milliseconds, and a recording of a fact that can be recomputed is
only a way to be wrong later. The **acceptance run is recorded**, because it cannot be read from
this side at all - TG17.3 requires the fifth adapter to be defined in a module the application
never imports, so nothing under `src/core` may reach it. Copying TG17.12's recording shape onto
both halves would have been the easy symmetry and the wrong one.

**Deciding moved to the reading side.** The test supplies apparatus only it owns - the adapter, the
native record, the window - and `measure_extension_conformance` performs all eight checks and
decides whether they passed. A test that deleted its own assertions therefore changes nothing about
what gets recorded. What it can still do is stop calling the recorder, and the answer to that is
`NOT_RUN`, which is the correct answer. The recording binds the declared contract - the fifth
adapter's names, the framework sources, what counts as glue, and **every declared occurrence with
its written reason** - so a recording cannot be made green by widening the list it is judged
against, or by rewriting why an entry is excused.

**`synthetic_fifth_adapter` now reads `PASS`**, and it publishes the number it does not block on. A
synthetic fifth domain with a monotone rank channel reached the registry, the control schema, the
conformance kit and the domain-blind mining seam from `src/tests/test_adapter_registry.py`, passing
all eight checks; installation required **0** framework edits across the seventeen generic
surfaces; and standing adapter-specific glue is **1**, named in the gate's own detail as
`AcquisitionView.tsx:243`. Three of the seven gates now clear. The verdict is unmoved:
`NOT_RELEASEABLE`, with `offline_matrix` and `restart_recovery` unrun in a cold plan,
`scale_shape_calibration` refused, and `live_sources` still the last unmeasured scientific gate.

**A guard passed because it could not see what it was checking, for the fifth time.** The
`defined_outside_the_application` check first compared `translate.__module__` against package
prefixes. Under pytest's import mode that string is the bare `test_adapter_registry`, which starts
with none of them - and would have started with none of them whatever the module was called. The
check is now resolved from the defining *file*, made repository-relative, and a callable whose
source cannot be located is a **refusal** rather than a pass, because an unanswerable question is
not a satisfied one. D64, D74, D75, slice 1's vacuous scan, and now this.

**A mutation pass with no baseline cannot tell a killed mutant from a broken suite.** The first
run of this slice's mutation script reported M3 - `defining_source` falling back to the module name
- as caught. It was not. The single failure that run was a test of my own that was broken: it
asserted `__module__ == "builtins"` for a callable compiled from a string, and the exec scope
carried no `__name__`, so the attribute was `None`. Every other mutation's count was inflated by
the same failing test. The script now runs an unmutated baseline first and refuses to report at all
unless it is green. Against a green baseline of 51 tests, **six mutations, six caught**: staleness
ignored, a missing required check tolerated, the module-name fallback, `DEFAULT` widened into glue,
another adapter accepted for measurement, and the live audit inherited from the recording.

**Fixing that test found a real defect in the code it was testing.** `inspect.getsourcefile` hands
back a pseudo-filename like `<no file>` for code compiled from a string whose module carries a
loader, and on Windows `Path("<no file>").resolve()` produces an absolute path inside the
repository without raising - so `defining_source` would have reported a located source that cannot
be read. It now requires the resolved path to be an existing file.

TG17.13 is complete. `live_sources` is the only scientific gate left, and it requires network.

**TG17.14 Four-domain live-source qualification — DONE (2026-09-04, `ed-dev`).** The first offline slice supplies
the evidence boundary without pretending to have made the measurement. `src/core/live_source_evidence.py`
derives the four required source identities from the flagship manifest, binds a future recording
to every adapter and acquisition module that decides what was measured, and distinguishes an
absent, stale or partial recording (`NOT_RUN`) from an operational refusal (`REFUSED`) and an
executed contract failure (`FAIL`). The release ledger now reads that channel and never reaches
the network while assembling its verdict.

The slice also corrects an ambiguity in the earlier phrase "public archives across four domains".
Reanalysis, Argo and TESS name archives and must demonstrate real network use. The fourth domain
is deliberately the **bespoke local-record family**: a pass must demonstrate a non-empty,
content-addressed researcher-supplied record with *no* network use. Giving order book a convenient
public feed here would replace the abstraction the flagship froze with a finance-specific adapter.

**The opt-in runner is now delivered, still without executing it.** `python -m
src.core.live_source_evidence plan` prints the complete contract with `network_used: false`:
a 324-value one-day ERA5/CDS request, at most 200 Argo profiles, at most two TESS products and
64 MiB, and the exact local-record requirements. The `run` command cannot start unless both
`SPECTRALEARTH_ALLOW_NETWORK=1` and the literal acknowledgement
`--confirm-network-access I_AUTHORIZE_BOUNDED_ARCHIVE_REQUESTS` are present. A missing local
record, provenance declaration or licence declaration is refused before the first provider call,
and the hashes of the committed fabricated demonstration CSVs are explicitly ineligible.
Provider refusals and unexpected implementation
failures are recorded separately; either records network use as unknown rather than inventing a
fact after an exception. The record is self-hashed, and plan assembly reads it without
performing acquisition.

**The authorised live run has now been made, and the gate reads `PASS`.** On 2026-09-04, with the
maintainer's explicit say-so and both locks satisfied, the bounded acquisition ran in all four
domains: 324 ERA5 values through CDS, 93 Argo profiles carrying 10,218 values, 18,279 finite flux
samples from one MAST SPOC product for TIC 261136679, and 34,560 values in 2,880 records from the
researcher-supplied order-book table. The first three demonstrate network use; the fourth
demonstrates **no** network use, which is what its contract requires and the reason the fourth
domain was chosen to be the bespoke family in the first place.

**It did not clear the release, and could not have.** Five of seven gates now read `PASS`, every
scientific gate has been measured, and the verdict is still `NOT_RELEASEABLE` because
`scale_shape_calibration` is `REFUSED`. Four archives in four domains bought nothing past a
declared scientific limit, which is the whole design.

**The order-book record is deliberately not committed.** `order_book` declares a licence under
which redistribution of raw depth is restricted, so the gate binds the record by `sha256` and the
repository stores its README rather than its bytes -- source URL, exact transformation and digest,
enough to rebuild it and check the hash. Committing the file would contradict the licence the
platform states while reading it.

**Two defects, both from first contact with real data, and neither findable offline.** **D94**: the
reanalysis probe republished a 32-character cache key in a field named `sha256`, and the gate
refused the whole record rather than credit a `PASS` whose content binding it could not verify --
the validator working. **D95**: SPOC emits a row per cadence including 815 of 20,076 with no
timestamp, and those were handed to a collection whose invariant is a finite strictly increasing
clock; they are now dropped and *counted*, with the count carried into the record. Every synthetic
fixture had a clean clock, which is exactly why a live gate exists.

**MAST is intermittent and the record must be read knowing it.** Across five attempts that day the
metadata service answered three times and timed out twice; a timeout is recorded `REFUSED` with
`network_used: null`, because after a failed call whether bytes moved is unknown. The binding
constraint on this gate is a metadata service, not the science.

**TG17.15 Rebuilding the scale/shape null so it can resolve - DONE (all five slices, 2026-09-05).**

TG17.11 delivered a calibration and a refusal, and the refusal is correct: the declared null cannot
reject at any inventory size the enumerator can reach. This phase asks what to build instead. It is
a change of scientific question rather than an implementation detail, so it is written down before
any code is cut.

**Two measurements made on 2026-09-04 that sharpen TG17.11's finding, and neither is in that
phase's record.**

*   **At its own minimum the test has no margin at all.** `minimum_resolvable_family()` returns
    105. Put one member of 105 a single step off the p-value floor and **zero** of 105 reject, not
    104. Graceful degradation begins only at k = 106 (105 of 106). So 105 is not a size at which
    the test starts working; it is a knife-edge requiring every one of 105 members to be a perfect
    planted match simultaneously. Reaching it would not have produced a usable instrument.
*   **Lifting the enumeration cap buys nothing, and the reason is the whole finding.**
    `_valid_reassignments` returns the derangement numbers -- 9, 44, 265, 1854, **14,833** at
    k = 8. It is tempting to read that as a reference set giving a floor of 1/(1+|A|) = 6.7e-5. It
    is not one. Each member's statistic depends only on *which partner it received*, and member 0
    has exactly k - 1 = 7 distinct partners across all 14,833 reassignments. Those draws produce
    seven distinct statistic values; using |A| as the denominator counts duplicates as independent
    evidence and is **anticonservative by three orders of magnitude**. The existing 1/k floor is
    right, and this is the mistake a re-implementation is most likely to make.

**So `MAX_REASSIGNABLE_PAIRINGS = 8` and the 105 bound are not two ends of one axis.** A uniform
sampler above 8 is achievable and provable -- rejection sampling from uniform permutations,
accepting only valid ones, is exactly uniform on the valid subset with acceptance about 1/e -- but
it would raise |A| and not per-member resolution. **The gap is structural, not computational**, and
no amount of compute closes it.

**The cause is one inventory doing two jobs.** The k pairings are simultaneously the hypotheses
under test, which sets the multiplicity burden, and the source of alternative partners, which sets
the resolution. Growing k lowers the floor to 1/k and raises the correction burden at nearly the
same rate; the crossover is `H_k / k <= alpha`, which is why the answer is 105 and why it arrives
with no margin.

**The proposed rebuild separates the two roles.** A **partner pool** of N candidate records that
are *not* hypotheses, and m preregistered tested correspondences. Each test substitutes its left
member's partner across the pool: an exact categorical reference set of size N, no factorial
enumeration, so the 8-cap stops being this test's concern. The requirement is
`1/(N + 1) <= alpha / (m * H_m)`, solved against the real correction rather than written down.
Measured against this repository's own `adjust()`:

| tested `m` | pool `N` needed | margin at `N` | margin at `2N` |
|---|---|---|---|
| 1 | 19 | -- | -- |
| 5 | 45 | 0 of 5 | 4 of 5 |
| 6 | 48 | 0 of 6 | 5 of 6 |
| 10 | 58 | 0 of 10 | 9 of 10 |

Six tests need a pool of about 48 rather than a 105-member all-pairs family, and doubling the pool
to about 96 buys real margin **at no multiplicity cost**. That is the property the present design
cannot have: pool size and family size become two independent knobs instead of one.

**What the rebuild costs, recorded before it is built rather than discovered afterwards.**

*   **The estimand narrows, and must be declared rather than slipped in (condition 15).** Joint
    reassignment asks whether the *overall correspondence structure* is special. Pool substitution
    asks whether *this* left member's affinity for *this* partner is special against a declared
    pool. The second is arguably what scale/shape mode already claims -- "this shape at this native
    duration resembles that shape at that one" is a per-correspondence sentence -- but the two are
    different questions with different answers, and the choice belongs in the record.
*   **Exchangeability becomes a data-curation obligation, and this is the new failure mode.** If
    pool members differ systematically in record length, noise floor or sampling density, the null
    is biased and every p-value is wrong. The present design's limit is computational and therefore
    self-announcing; this one's limit is a property of a curated inventory and can be violated
    silently. It needs a declared admission criterion and a guard, and it is the half of this phase
    most likely to go quietly wrong.
*   **Benjamini-Yekutieli stays.** The m tests share one pool and are dependent. BY is valid under
    arbitrary dependence and BH is not, so the existing correction choice is load-bearing rather
    than incidental.

**Slices.**

*   **Slice 1 - the estimand as a declared object. DONE.**
    `src/core/correspondence_estimand.py`; `src/tests/test_correspondence_estimand.py`, 14 test
    functions, **14 passed**. Both estimands are registered, `per_correspondence` is chosen, and
    `joint_structure` is registered *inadmissible* so it is refused by name rather than
    rediscovered -- it is the natural first design and its failure is invisible from inside it.
    `require_declared_estimand` refuses an undeclared estimand rather than defaulting, because the
    two questions can disagree on the same data and a silent default would choose the finding.
    The derangement measurement is computed by `joint_reassignment_resolution` from the null's own
    enumerator rather than restated, so it cannot drift from the object it describes, and the
    anticonservative misreading is pinned at more than a thousandfold in the direction that eases
    rejection. `minimum_pool_size` is solved against the real correction the way
    `minimum_resolvable_family` is. A test asserts the property that motivates the whole rebuild:
    at a pool of 48 one member off the floor rejects nothing, at 96 it rejects five of six, and
    the test count is identical -- margin bought without multiplicity. The slice declares the
    question and stops; a guard asserts no power key appears in its report.
*   **Slice 2 - the partner pool and its admission criterion. DONE.**
    `src/core/partner_pool.py`; `src/tests/test_partner_pool.py`, 20 test functions, **20
    passed**, and four mutations each caught. The criterion is decided on **marginals only**, and
    circularity is made *inexpressible* rather than forbidden: `build_partner_pool` reads
    `RecordProfile` objects and never records, so an admission rule keyed on resemblance to the
    record under test is not a rule this module can express. A test asserts the profile field set
    exactly, so a similarity or distance field cannot be added quietly. `native_seconds` is
    recorded and refuses a band by name, because banding the quantity scale/shape mode compares
    across would refuse the comparison the mode is for. The observed partner must clear the same
    bands as its own alternatives; candidates sharing the left member's provenance are refused as
    leakage; nothing is dropped silently; and a pool below `minimum_pool_size(m)` is refused
    rather than returned. The receipt states that admission is a **necessary and not sufficient**
    condition for exchangeability, because a pool that reads as a proof of it would be worse than
    no pool.
*   **Slice 3 - the exact pool-substitution null, and the pool size derived rather than chosen.
    DONE.** `src/core/pool_substitution_null.py`; `src/tests/test_pool_substitution_null.py`, 35
    test functions, **37 passed** (one parametrised over three values), and ten mutations each caught. Nothing is sampled: the
    reference set is a sealed finite inventory, so `monte_carlo_pool_substitution` is registered
    and refused -- a Monte Carlo denominator is chosen by the caller rather than fixed by the
    pool. The module takes a **callable** and evaluates all `N + 1` values itself, so a
    precomputed observed statistic cannot enter under a different normalisation; the observed pair
    is evaluated twice and a non-deterministic statistic is refused. Orientation is declared with
    no default, because a similarity and a distance invert the tail. Ties count toward the
    numerator. The **family size is sealed in the pool digests** before any p-value exists, so
    narrowing a family after seeing its results contradicts a number that predates them.

    **The slice also corrected a defect in its own first draft, and the correction is the more
    useful half.** `minimum_pool_size(m)` sizes for the world in which *every* declared
    correspondence is genuine, and the first `resolution()` reported that as a green light. It was
    wrong in exactly the way this phase exists to catch: six pools of 58 clear the required 48 and
    report `every_member_can_reject_at_its_own_floor = True`, yet a family with **three genuine
    correspondences of six rejects nothing** -- all three at the exact floor, `q = 0.083` -- because
    members that do not correspond consume the Benjamini-Yekutieli step-up ranks the genuine ones
    need. `sparsest_detectable_count` now measures, from the floors the pools actually have, the
    fewest genuine members the family could ever reject, and the receipt says so in words:
    *"this family can produce a rejection only if at least 5 of its 6 declared correspondences are
    genuine."* `minimum_pool_size_for_detected_fraction` sizes a pool for that world in advance:
    at `m = 6`, all six genuine needs `N = 48`, half needs `N = 97`, one of six needs `N = 293`.

    No calibration was performed and no false-positive rate was measured on real records. The gate
    is unchanged: `scale_shape_calibration` still reads `REFUSED`, verdict `NOT_RELEASEABLE`.
*   **Slice 4 - calibration, with margin measured rather than assumed. DONE**
    (`src/benchmarks/pool_calibration.py`, `src/tests/test_pool_calibration.py`, 50 tests, 15 of 15
    mutations caught after four gaps were found and closed; architecture.md section 7.1f; VERIFICATION.md.)

    **The null does not repeat T4C.5h's defect.** Five declared cases at 200 realisations each,
    `m = 6`, pools of 61 to 470, run through the real `correspondence_family` with the real
    `shape_recurrence` statistic. `no_correspondence` gives a family-wise false-positive rate of
    1/200, one-sided bound **0.0235**; `shared_grid_alias` 2/200, bound **0.0311**;
    `clean_partner_noisy_pool` 0/199, bound **0.0149**. Every bound clears alpha, and
    `unresolvable_inventory` refused 200 of 200 rather than scoring.

    **The tail is not the whole check.** Under exchangeability the observation's rank among its `N`
    alternatives is uniform on `{1, ..., N + 1}` *exactly*, so the whole distribution is predicted
    in advance, not only its 5% tail -- a rate can look nominal while the distribution is wrong.
    Measured on one member per realisation, because members of a family share an inventory and are
    dependent: KS 0.065 (p = 0.35), 0.073 (p = 0.23), 0.068 (p = 0.30). It holds.

    **Slice 2's claim boundary was asked for a number.** Passing every declared band is necessary
    for exchangeability and not sufficient -- so two adversarial nulls attack it: an artefact every
    record carries keyed to position within its own cycle, and an observed partner drawn
    systematically cleaner than the alternatives its own bands admit. Both hold the declared rate.

    **The defect this slice found in its own first recorded run.** `planted_correspondence` was
    declared to pass when detection reached 1.0. The run measured **1,199 of 1,200** and reported
    `calibrated: False`. The expectation was wrong, not the run: with pools of up to 470
    alternatives a chance candidate will occasionally outrank a real correspondence, so demanding
    that every member reject was demanding a test with **no type-II error** -- the point-estimate
    mistake already fixed for error rates, left standing in the opposite direction. It is replaced
    by two criteria derived from the case: the statistic must rank the true partner first for at
    least 90% of members, judged on a **lower** confidence bound; and every member it does rank
    first must reject (`maximum_unresolved_at_floor = 0`, parameter-free).

    **Detection is a curve and it falls off a cliff between `w = 0.92` and `w = 0.84`.** At
    `w = 0.88`, **98.8% of members have an uncorrected p at or under 0.05 and 35.4% survive
    correction**. That gap is slice 3's `sparsest_detectable_count` prediction confirmed: when the
    family is mixed, a surviving member must clear `alpha / (m * H_m) = 0.0034`, which no pool
    below 293 can reach. Every rung reports `members_at_their_floor_that_did_not_reject`, so a low
    number says whether the correspondence was absent or the pool too small.

    **A cost of the admission contract that nothing had measured.** `native_seconds` is unbandable
    by design, but `cadence_seconds` is banded and equals native duration over a bounded row
    density -- so the cadence band narrows native duration *transitively*. `admission_yield`
    measures it: an inventory spanning **4.05 decades** yields pools spanning **0.95 to 1.47**, at
    a yield of 15% to 26%.

    Every rate is an interval and every acceptance reads a bound: `certifies` the one-sided upper,
    `attains` the lower, and `certifies_rate` is separate from `within_expectation` so a run too
    small for its own claim says so. `REALISATIONS_FOR_ALPHA` solves for the smallest certifying
    run rather than asserting it -- 59 -- and the declared 200 is larger because the distribution
    check resolves 0.18 at 59 and 0.096 at 200.

    The gate is unchanged and deliberately so: `scale_shape_calibration` still reads `REFUSED`,
    verdict `NOT_RELEASEABLE`. Carrying this measurement into it is slice 5's work, by checked
    supersession.
*   **Slice 5 - the gate. DONE (2026-09-05).** `scale_shape_calibration` reads the new
    calibration, and **the gate still refuses**. What moved is the reason; the old reason was kept.

    **The supersession is recomputed, not remembered.** `scale_shape_supersession` does not quote
    TG17.11's claim. It recomputes it from the two primitives that claim turned on -- 105 resolvable
    against 8 drawable -- using the same functions the gate's applicability section uses, and a
    guard asserts the two agree. Three outcomes, and the third is the point: `SUPERSEDED`,
    `NOT_SUPERSEDED` when the successor is absent, stale or failed, and **`VOID`** when the
    predecessor's claim has stopped being true. If someone lifts the enumeration cap, the old limit
    removed itself and calling that a supersession would credit this phase with work it did not do.
    A guard drives the record into `VOID` and asserts it says so. Deleting the refusal instead
    would have left a repository in which a limit that was overcome and a limit that was edited
    away read identically.

    **A second recording, with a backstop that is honestly smaller than the first's.** The calendar
    calibration is cheap enough to re-run whole on every test pass, and is. This one costs
    **1,031 s** for its five cases plus an eight-rung ladder. Rather than imply an equivalence, the
    recording carries a **reproduction witness**: `calibrate_case` runs realisation `i` at
    `seed + i`, so a three-realisation run at the recorded seed is the *leading prefix* of the
    recorded run rather than a similar measurement, and the suite recomputes it for every case in
    seconds. The case that refuses every realisation witnesses its refusals rather than the empty
    list, which would agree with any other run that also produced nothing.

    **What was recorded**, reproducing slice 4 exactly: `all_met: true`, family-wise one-sided
    bounds of **0.0235**, **0.0311** and **0.0149** against alpha 0.05, ranks uniform on their own
    lattice (KS 0.0650 / 0.0727 / 0.0684 at p 0.35 / 0.23 / 0.30), the planted case ranking the
    true partner first for 1,189 of 1,200 members at a lower bound of **0.9849** with none of those
    failing to reject, `unresolvable_inventory` refusing **200 of 200**, and one inventory digest
    across all eight rungs.

    **Why it still refuses, computed rather than asserted.** Two blockers, each published with what
    would discharge it and whether this module can decide it at all. `declared_inference` is
    decidable here: every frozen scale/shape manifest declares `scale_partner_reassignment` at 200
    replications, **read back from the six manifests** rather than restated, and the calibrated
    method is exact pool substitution, which none of them requests. Discharging it is a change to
    the experiment declaration, not to the gate. `pool_exchangeability_on_real_records` is **not**
    decidable here, and a guard asserts it survives a passing recording: it is the failure mode
    this phase named in advance as the silently-violable one, and no further measurement on built
    fixtures reaches it. TG17.11 kept three facts apart -- a calibrated method the declared plans
    cannot reach, a method that does not exist, a method that ran and failed. This slice adds a
    fourth: a calibrated, applicable, recorded method that answers a question no declared plan asks.

    **Verification.** 59 guards across the two gate suites (12 to 28 and 22 to 28) plus three for
    the witness (50 to 53). Fifteen mutations; the first pass missed one and it was a real gap --
    every guard read the recording already on disk, so a break in the code that *writes* one would
    have passed everything and surfaced only after the next fifty-minute re-record. `_trim_case` is
    now exercised directly on a one-realisation outcome. Second pass **15 of 15**. The verdict is
    unmoved: `NOT_RELEASEABLE`.

**What would falsify this phase, stated in advance.**

1.  No pool admission criterion can be stated that is both checkable and non-circular -- the
    inventory cannot be shown exchangeable without assuming the answer. The per-correspondence
    estimand is then not testable on real records, and the honest outcome is that scale/shape mode
    supports description but no significance claim at all.
2.  The measured false-positive rate on a true null departs from nominal, as T4C.5's did. The null
    is then not the null it claims, and the design is wrong regardless of its arithmetic.
3.  A pool large enough to resolve cannot be assembled from real records without admitting members
    that are not plausible partners, so N is bought at the cost of the exchangeability the
    p-values depend on.

Any of these is a result. The first would mean scale/shape alignment is a describable but not a
testable mode, which is worth knowing and worth stating plainly.


### Phase G19 - The researcher's conversation with the record - **NOT STARTED**

G7 gives the platform an adversarial review layer that argues with a finding. G19 asks the
adjacent question: a researcher meeting a `REFUSED` gate or a corrected q-value wants to
*interrogate* it, with a model of their choosing, over more than one turn. That is a real need and
the architecture already anticipates most of it. What it does not yet have is a conversation, and a
conversation introduces failure modes a single review does not.

**What already exists, so this phase is an extension rather than a new risk surface.**

*   `ReviewRecord` is already an **append-only hash chain** bound to one exact bundle revision:
    `bundle_sha256`, `bundle_revision`, and `previous_sha256` on every call.
*   `record_call` takes `context` **per call**. Context is passed, never accumulated, so
    re-grounding is already the shape of the API rather than something to retrofit.
*   Every `CallRequest` carries the bundle digest and revision it was asked against, so a stale
    turn is already *detectable*.
*   `verify_claim_independence` makes R22 executable three ways: the claim state must be identical
    with the review present and deleted, must survive a rebuild from the bundle's own bytes, and
    **no phrase from any recorded response may appear anywhere in those bytes**.
*   `ResponseSchema` refuses free text where a schema was declared, and R23's non-reproducibility
    is recorded in the body rather than papered over.

**Retrieval is the new capability, and chunking is the wrong instinct for this instrument.**
Retrieval-augmented generation returns *fragments ranked by similarity*. The asset this whole
programme is built on is that every result travels with its refusals and its claim boundary.
Retrieve three of eight chunks of a gate receipt and a model can state "ten links replicated in
train and test" without "D84 and D85 govern whether an absence was detectable", or surface a `PASS`
stripped of its `claim_boundary`. That is the exact failure the platform exists to prevent, and a
retrieval layer would introduce it invisibly and plausibly ranked.

So the rule this phase is built on:

> **Retrieve at the granularity of a complete record. Never a fragment of one.**

Receipts, qualification plans and `EvidenceBundle`s are bounded structured objects that already
carry their own boundaries, and they fit in a context window whole. Retrieval chooses *which*
records are relevant across a corpus; each chosen record then enters entire. A record too large to
enter whole is a signal that the platform owes a **deterministic summary view** it computes itself,
not an invitation for a chunker to guess which paragraphs mattered.

**Three properties a conversation needs that a single review does not.**

*   **Re-grounding every turn.** The transcript carries dialogue; the scientific context is
    rebuilt from source on each turn and never inherited as the model's own paraphrase. Otherwise
    turn twelve reasons about turn three's summary of a receipt, and the compounding is invisible
    because every individual turn looks reasonable.
*   **Staleness that refuses rather than warns.** A conversation outlives the record it discusses.
    TG17.14 demonstrated this on real code: editing an acquisition module returned `live_sources`
    to `NOT_RUN` and invalidated a passing record. A conversation open against a bundle whose
    digest has moved must refuse to continue, in the same way the gate does.
*   **Independence over the whole transcript, not one review.** `verify_claim_independence` holds
    for one `ReviewedBundle`. A conversation spans several bundles and many turns; the guarantee
    must hold for every bundle it touched, and deleting the entire conversation must change
    nothing anywhere.

**What this phase costs, recorded before it is built.**

*   **Model output becomes model input.** Under R22 that never reaches a claim, so the ladder is
    safe. But turn N-1's answer is turn N's context, so **drift compounds inside the transcript**
    even while every claim stays untouched. The mitigation is re-grounding, and re-grounding is
    only checkable if the context is rebuilt from source rather than diffed against history.
*   **A conversation is the most quotable artefact the platform will produce**, and R23 says it is
    recorded evidence and not reproducible computation. A transcript can therefore never be cited
    as the reason a claim holds. That has to be visible in the rendering, not just true in the
    schema, or the most persuasive object in the system will be the least verifiable one.
*   **Multi-provider is a refusal surface, not a convenience.** "Whatever model they choose" means
    the recorded call must carry provider identity, and a provider that cannot honour a declared
    response schema, or that silently accepts sampling parameters R23 rejects, must be **refused
    by name** rather than accommodated.
*   **Cost multiplies.** TG7.3's provider-neutral accounting exists; a conversation turns one
    review into an open-ended sequence, so a declared budget per conversation is a requirement
    rather than an option.

**Slices.**

*   **G19.1 - the conversation as a bound object.** A transcript spanning one or more bundle
    revisions, each digest pinned, with researcher turns and model turns distinguishable by type
    rather than by convention. A turn whose bundle digest has moved is refused.
*   **G19.2 - re-grounding, and a guard that it happened.** Context assembled from source each
    turn. The check that matters: a turn's context must be **derivable from the records it names**
    and must contain nothing that appears only in an earlier model response.
*   **G19.3 - whole-record retrieval.** Selection across a corpus at record granularity, with the
    refusal that makes it safe: a record that will not fit whole is refused, naming the
    deterministic view that should be built, rather than chunked.
*   **G19.4 - independence extended to the transcript.** `verify_claim_independence` generalised
    over every bundle a conversation touched, plus the deletion test: remove the whole
    conversation and assert every claim digest is unchanged.
*   **G19.5 - the rendering.** The transcript displayed with its non-reproducibility and its claim
    boundary attached to every turn, so the most quotable artefact is also the most clearly
    labelled. R22's structural defence carried into the interface, not restated as a caption.

**What would falsify this phase, stated in advance.**

1.  Re-grounding cannot be checked. If no guard can distinguish a context assembled from source
    from one contaminated by an earlier model turn, then drift is unmeasurable and the layer
    should stay a single-shot review, which is verifiable.
2.  Whole-record retrieval proves impractical -- the records that matter do not fit, and the
    deterministic views needed to shrink them cannot be written without choosing what to omit,
    which is the chunking problem wearing a different hat.
3.  A conversation measurably changes what researchers conclude from the same evidence. That is
    testable by giving the same bundle to readers with and without the layer, and it is the
    outcome that would matter most: a tool that makes people more confident without making them
    more correct is the opposite of this programme's purpose.

The third is the one worth stating loudest, because it is the only one that cannot be found by
reading the code.


## 6. Definition of Done

Inherits all nine conditions from `roadmap.md` §10 unchanged. A phase in this line is done when
those hold **and**:

10. Every new capability is reachable from a **new domain** without editing `src/` (E15, TG8.1) —
    the cross-domain form of the existing registry condition.
11. Every domain in play has declared its violated assumptions, and the analysis layer
    demonstrably refuses what those violations forbid (R17, R21).
12. Every mining pass has a declared family size fixed before execution, with a passing power
    check or an explanatory refusal (R18).
13. No LLM output is an input to any claim level, verified by deleting them all and asserting no
    claim changes (R22, TG7.1).
14. The atmospheric line's accepted receipts remain bit-identical, or the divergence is a named,
    justified, documented defect.
15. A conditional-information result names its estimand and benchmarked conditional null; a
    global permutation or causal interpretation cannot satisfy G16.2.
16. A stable-subspace result binds the complete search family before generation, fits only on the
    generate partition, identifies the span independently of basis rotation/sign, and opens its
    confirmation partition once.
17. A configurable multi-domain experiment is represented by one versioned manifest shared by
    API, orchestrator, UI and receipt; no layer carries a second scientific configuration (R24,
    E17).
18. Calendar-aligned and scale/shape-aligned experiments have separate support semantics, nulls
    and claim language. Neither shared time nor shared structure can be rendered as causality, and
    raw cross-domain magnitudes are never compared (R19--R21).
19. A complete G17 capability passes the clean-browser no-glue test and the synthetic fifth-
    adapter test. Backend-only reachability, handwritten JSON or a domain branch in the generic
    runner/UI cannot satisfy completion.

---

## 7. Explicitly out of scope

Recorded so that omission cannot be mistaken for oversight.

* **The atmospheric acquisition stack**, `era5_overlap`, `gate_campaign`, `forecasting/` and the
  T4C.6 preregistration. Not generalised, not modified, not reinterpreted.
* **Transform implementations.** The transform *seam* generalises; the mathematics does not need
  to.
* **Causal inference.** The ladder stops below it deliberately (R7).
* **Unbounded or neural learned representations.** G16 admits a bounded, sealed linear subspace
  family only. Neural representation learning remains a future registry entry outside this
  programme until it has its own leakage, multiplicity, calibration and transfer contracts.
* **3D and spatiotemporal transforms.** `CoefficientField` remains 2D-per-frame. A 3D transform is
  a different and much more expensive object and calling the current one 3D would misdescribe it.
* **Formal accessibility conformance certification.** TG11.6 supplies the workflow-wide source
  and keyboard contract; an assistive-technology audit and WCAG conformance claim remain outside
  this engineering programme until rendered inspection is available.
* **The LLM layer before G6.** An adversarial review layer above a claim ladder that does not
  exist is a debating society with nothing to constrain it.

---

## 8. What would falsify this programme

Stated in advance, so the answer cannot be negotiated afterwards:

1. **G0 fails** — the inference layer turns out to depend on atmospheric structure it does not
   declare.
2. **G1 cannot localise** — the assumptions are diffuse rather than concentrated, and the
   bit-identical criterion cannot be met without rewriting the analysis layer.
3. **TG3.1 proves the multiplicity is unaffordable** for every scientifically interesting family.
   This is the most likely of the four, and it bounds the programme rather than ending it: the
   answer becomes "this works only for preregistered narrow families", which is still a working
   instrument and an honest one.
4. **TG8.3 shows onboarding cost is not falling** as domains accumulate — the abstraction is being
   maintained by hand, one adapter at a time, and is not real.

Any of these is a result. None of them is a failure of the exercise.
