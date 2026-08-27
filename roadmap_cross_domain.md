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
  atmospheric programme's claim boundary.** T4C.6 remains unrun on both lines; D43 remains open.
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

## 3. Additional Standing Rules (R17--R23)

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

---

## 4. Additional Engineering Standards (E12--E15)

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
   **G10 is complete:** the store catalogue is registered and probe-backed, and acquisition is
   consolidated by domain. **G11 has begun: TG11.0 is complete**, replacing the flat numbered
   navigation with workflow sections and making the selected record and study persistent shell
   context. The remaining G11 slices put the cross-domain engine in front of a person for the
   first time — `domain_analysis` and eight
   core modules currently have `api=0`, so TG8.4's records lead nowhere; G12 reaches the ocean,
   gridded and then Argo; G13 reaches the sky and closes the violation vocabulary. **No public
   dataset has been ingested by any of them yet.**

**Open defects:** D43 (the real-data gate is not laptop-feasible through the catalogued
WeatherBench layouts) and D18 (partial — CUDA-only device probing). Both predate this line and
neither blocks G8. **TG12.1 may bear on D43**: an ocean product chunked more kindly than the
WeatherBench layouts could make the gate reachable, which is a thing to measure rather than
assume.

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

**TG8.2 Licence and terms provenance.** Every public dataset carries its licence, attribution
requirement and access terms in its provenance, and export refuses to emit a derived product
whose source licence forbids it. Redistribution rules differ sharply across public archives and
must be data, not folklore.

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

**TG9.4 Accessibility of the new surface. DONE for this surface (`ed-dev`).** The findings views
ship with `role="tablist"`/`role="tab"`, `aria-selected`, `aria-pressed`, `aria-label`, labels
bound with `htmlFor`, visible focus rings and `aria-hidden` on decorative icons, asserted by test.
**The legacy measurement is unchanged and restated rather than quietly improved:** accessibility
across `frontend/src` as a whole remains zero-derived and the transform workbench was not touched.
This slice stops the new surface adding to that debt; it does not repay it.

**TG9.3 The refusal surface.** What the instrument will not do, shown rather than hidden. Where a
domain declares violations, the view states which analyses are therefore refused and why (R17,
R21). **This slice now also carries the half of TG9.1 that was declared and not built:** exposing
`DomainDeclaration` — declared violations (E15), lag policy (R21) and `precedence_admissible`
(R17) — over HTTP at all. TG9.1's `/domains` lists registered *glossaries* and nothing about what
a domain refuses, so there is currently no route through which a client could learn that a domain
forbids a precedence claim. LLM commentary renders under its R23 label, visually separated, never in the same container
as claim text. **Acceptance:** a blocked bundle and a contradicted one each render their refusals;
an automated check asserts no commentary string shares a container with a claim string.

**TG9.4 Accessibility of the new surface.** Accessibility is **zero, measured** across
`frontend/src` — `0` `aria-*` or `role` attributes and `0` keyboard handlers — and `roadmap.md` §1
records it as deliberately unscheduled rather than overlooked. That position is honest for a
mouse-driven transform workbench. It is harder to defend for something described as an instrument
for reading scientific findings. **This phase does not fix the legacy workbench**, which stays as
recorded. It does require that the findings views ship keyboard-navigable and semantically
labelled, so the new surface does not add to the debt. **Acceptance:** the findings views are
operable without a mouse, asserted by test; the legacy measurement is restated unchanged beside
the new one, so the two are not confused.

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
condition applies to every domain. Review is an honest labelled waypoint for TG11.5, not a
button to a surface that does not exist yet.

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

**TG11.4b The cross-domain record.** `core\cross_domain.py` - `align_exact`, `physical_lags`,
`sweep_cross_domain`, `confirm_cross_domain`. Two channel tables from two declared domains,
aligned on an exact common clock with no interpolation, swept over a lag family declared in
*seconds* and converted to frames per domain. It belongs beside `/api/v1/analysis` rather than
`/api/v1/mining` because its input is a record and its output is a lag result, and it is the
slice where a lag family stops being expressible only in frames of one clock. Sequenced after
TG11.4 because the seal and ledger plumbing it needs is the plumbing TG11.4 shared.

**TG11.5 The review surface.** The adversarial round-robin, its recorded calls and its cost
receipts (`round_robin`, `recorded_call`, `review_cost`). Everything here is R23
recorded-not-reproducible, so the surface must present it as recorded argument and never as
something the record permits — the separation TG9.3 already enforces for commentary.

**TG11.6 Accessibility, repaid rather than deferred.** Accessibility across `frontend/src` is
**zero, measured** — no `aria-*` or `role` attributes and no keyboard handlers outside the
findings and records views. TG9.4 stopped the new surfaces adding to that debt and explicitly did
not repay it. "Professional" is the standard being asked for, so this phase repays it for the
workflow above rather than leaving it as a permanent footnote.

**Claim boundary.** Making a capability reachable is not evidence that it is correct; the
benchmarks are what argue for correctness, and only for the thirteen cases they cover. A grouped
navigation does not make the gridded line domain-general, and this phase deliberately does not try
to. No write path added here may accept a rung, a claim level, or a confidence figure from a
client.

#### Phase G12 — The ocean

**TG12.1 A gridded ocean product.** Candidates in preference order, chosen by TG10.3's probe
rather than by reputation: GLORYS (Copernicus Marine, free account), ECCO (NASA Earthdata, free
account), NOAA OISST (open). Declared honestly under R17 as **breaking nothing new** — a source,
not a second domain — because a third catalogue entry that looked like a third domain would be
the exact false confidence R17 exists to refuse. **It may bear on D43:** ocean products are often
chunked more kindly than WeatherBench's one-timestep-deep layouts, so a laptop-feasible long
regional record may exist here. Recorded as a possibility to measure, not a promise.

**TG12.2 Argo profiles — a second acquisition shape.** The slice where `argo_float` stops being a
declaration. `ProfileSpec` takes a region, a time window and a depth range and returns an
irregular collection of profiles; content-addressed and machine-independent like `CropSpec`, but
it cannot borrow it, because the result is a scatter rather than an array. It feeds a
`ChannelSeries`, so everything TG8.4 built — the clock facts, the obligations, the aggregate check
— applies unchanged. **`argo_float`'s declaration needs revisiting in the open:** it declares
latitude, longitude *and* a pressure level, so TG8.4's E14 check correctly refuses it a flat
channel table. Either the profile shape supplies those axes or the declaration is split, and that
is a design decision to take visibly rather than paper over.
**Acceptance:** a real Argo query produces a record whose irregular clock is reported as
irregular, under a domain declaring `irregular_sampling` — the first time that refusal fires
against data rather than a fixture.

#### Phase G13 — The sky, and closing the vocabulary

**TG13.1 Photometry, and the last unbroken assumption.** TESS or ZTF light curves via MAST/AWS.
Breaks `no_natural_cycle`: there is no diurnal or annual forcing, so R11's harmonic climatology
has nothing to remove and its default periods would fit noise. This is the only domain that
exercises that refusal. It also breaks `irregular_sampling` and `non_stationary_support` through
sector gaps and targets entering and leaving, and breaks the metric assumption *differently* from
`order_book` — angular separation is a real metric that is not a length in metres. A third
acquisition shape: per-target, sector-based.

**TG13.2 The coverage claim, checked rather than asserted.** A test that every entry in
`KNOWN_VIOLATIONS` is broken by at least one registered domain **with a real data path behind
it**, not merely declared. The claim that the abstraction generalises then rests on something
mechanical rather than on this document.

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

---

## 7. Explicitly out of scope

Recorded so that omission cannot be mistaken for oversight.

* **The atmospheric acquisition stack**, `era5_overlap`, `gate_campaign`, `forecasting/` and the
  T4C.6 preregistration. Not generalised, not modified, not reinterpreted.
* **Transform implementations.** The transform *seam* generalises; the mathematics does not need
  to.
* **Causal inference.** The ladder stops below it deliberately (R7).
* **Learned representations.** Admissible as a future registry entry; not in this programme.
* **3D and spatiotemporal transforms.** `CoefficientField` remains 2D-per-frame. A 3D transform is
  a different and much more expensive object and calling the current one 3D would misdescribe it.
* **Accessibility.** Measured at zero in `roadmap.md` §1 and unchanged here. Recorded, not
  scheduled.
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
