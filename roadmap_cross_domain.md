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

**TG2.2 Feature extraction as a registry.** Local maxima above a surrogate-calibrated threshold
with sub-pixel localisation is the first registered extractor, not the definition of extraction.

**TG2.3 Tracking.** Frame-to-frame association with scale and orientation gating; nearest
neighbour first, Hungarian assignment once it must be respectable. Yields velocity, scale
velocity and lifetime.
**Acceptance:** `4D.tracking` moves from `NOT_YET_RUNNABLE` to **PASS** on
`advected_vortex_sequence` — 1 track, 1 birth, 0 deaths, position error < 1 px, correct
scale-doubling detection. The answer is already recorded in the repository and was not written by
this phase.

**TG2.4 Representation-induced feature audit.** Extract features from a field with **no**
structure under each registered representation and confirm the extractor reports none. R8's
lesson: a representation can manufacture a motif, and the null benchmarks must catch it.

**Exit criterion.** Features and tracks exist, are domain-typed rather than weather-typed, and
pass a benchmark written before them.

---

### Phase G3 — Constellations and the declared-family problem

**This phase is where R18 is either solved or the programme is bounded by it.** Do the
multiplicity work *first*; the mining code is comparatively easy and will be worthless without it.

**TG3.1 Family accounting.** Before any mining code: an exact enumerator that, given a search
specification, returns the family size and the surrogate count `required_surrogates` demands, and
**refuses the specification** when `check_power` fails. Refusals name which R18 remedy applies.
**Acceptance:** the enumerator reproduces T4C.6's known family (36 tests, 3,005 surrogates) from
its campaign JSON.

**TG3.2 Generate/confirm split.** Machinery for mining on train, freezing the declaration under a
content hash, and testing once on held-out — with the frozen declaration cryptographically bound
so a post-hoc edit is detectable. This is R18's escape hatch and must exist before mining does.

**TG3.3 Constellations as attributed graphs.** Features plus typed relations — distance, relative
scale, temporal lag, direction, convergence, containment, succession, co-occurrence — expressed
independently of originating domain. Relations are a registry.

**TG3.4 Invariant matching.**
**Acceptance:** `4E.invariance` moves from `NOT_YET_RUNNABLE` to **PASS** on
`build_planted_configuration` under rotation, rescaling and translation. A matcher that memorises
pixel positions must fail this, and a test asserts that a deliberately position-memorising matcher
does.

**TG3.5 Recurring motifs.** Mining for repeated constellation configurations, executed only
through TG3.1's declared family and TG3.2's split.

**Exit criterion.** A motif can be mined, and its declared family was affordable and corrected.
If TG3.1 proves that every scientifically interesting family is unaffordable, that is the
programme's real boundary and it is written up as such.

---

### Phase G4 — Planted ground truth

Before any real cross-domain claim. Extends `synthetic_generator/cascade.py`.

**TG4.1 Planted relationships at unknown scale and lag.** The engine is not told where the
relationship is and must recover it.

**TG4.2 The five refusals.** Per the proposal's own validation strategy, the engine must:
recover planted relationships without being told their scale or lag; reject convincing but
artificial correlations; distinguish independent observations from autocorrelated repetitions;
avoid representation-induced motifs; and report *no relationship* when none exists.
**Acceptance:** five benchmarks, at least three of them nulls, all returning null. Per the
existing Definition of Done, **the null benchmarks returning null is the load-bearing result.**

**TG4.3 Planted cross-domain relationships.** Two synthetic domains with different units,
semantics and cadences, one carrying information about the other's later state.

---

### Phase G5 — Frozen cross-domain motif transfer

The flagship experiment, and correctly last among the scientific phases.

**TG5.1 Motif freezing.** A structural definition serialised under a content hash, with the
originating domain recorded and the definition immutable thereafter.

**TG5.2 Blind transfer.** Search for the frozen motif in a domain opened *after* the freeze.
Machinery must make redefinition impossible, not merely discouraged (R20).

**TG5.3 Relationship transfer.** Whether the motif's relationship to subsequent organisation also
transfers, under the full G3 family accounting.

**Acceptance:** a preregistered transfer with a declared family, a corrected result and a
train/test split. **A null result here is a genuine, publishable-shaped finding** and is an
acceptable outcome of the entire programme.

---

### Phase G6 — The claim ladder and `EvidenceBundle`

Deterministic and mandatory. Must exist before G7.

**TG6.1 `EvidenceBundle`.** Hypothesis, observations, effect sizes, uncertainty, null results,
replication results, holdout performance, provenance, confounders, **contradictory evidence** and
failure states — as a hashed, append-only structure. Contradictory evidence is a first-class
field, not a remark.

**TG6.2 The ladder.** `observation → association → robust association → candidate precursor →
demonstrated predictive utility`. Each rung has deterministic entry conditions computed from the
bundle. Causal claims are **outside the ladder entirely** and unreachable without an explicit
causal-inference framework that this programme does not provide (R7).
**Acceptance:** rung assignment is a pure function of the bundle, property-tested; and a bundle
with a FAIL or INVALID gate cannot reach any rung above `observation` by any input.

**TG6.3 The five outputs.** For any bundle: what can be claimed; what cannot; what evidence
contradicts it; what alternative explanations remain; and which single observation would most
efficiently distinguish between them.

---

### Phase G7 — Adversarial review layer

Sits **above** G6's gates and may not touch them (R22). Non-deterministic and recorded as such
(R23).

**TG7.1 The recorded-call boundary.** Every call captures verbatim request, verbatim response,
exact model id, effort, API request id and timestamp. Structured outputs
(`output_config.format`, or `messages.parse()`) constrain every response to a schema, so
commentary can never arrive as free text that later code parses hopefully.
**Acceptance:** a test deletes every LLM output from a corpus of bundles and asserts **no claim
level changes**. This is R22 made executable and is the single most important test in the phase.

**TG7.2 Adversarial round-robin.** Candidate synthesis → statistical challenger → confounder and
alternative-explanation challenger → domain-plausibility challenger → provenance and methodology
challenger → response and revision → independent reassessment → bounded final synthesis. Not
majority voting. **Unresolved dissent is retained in the bundle, never reconciled into
consensus.**

**TG7.3 Cost control.** The review layer makes many small calls over a shared evidence corpus,
which is the exact shape prompt caching is designed for: the bundle and rubric form a stable
prefix, the challenged claim is the volatile suffix, and `usage.cache_read_input_tokens` is
asserted non-zero by a test rather than assumed. Non-interactive review passes go through the
Batch API at half cost. Challenger roles that are mechanical rather than judgemental run at lower
effort or on a smaller model; the final synthesis does not. Model selection is deferred to
implementation time and pinned per run in the receipt, because a bundle reviewed by a different
model is different evidence.

**TG7.4 Translation, bounded.** Rendering a finding in domain language for a reader. Under R19
this may not introduce a semantic comparison the structural evidence does not support, and under
R9 a bare confidence figure remains unrenderable.

---

### Phase G8 — Public dataset onboarding

Only once G0--G4 have shown that domains can be added without touching the core.

**TG8.1 The onboarding contract.** A documented adapter recipe: declare axes and roles (E14),
declare geometry (E13), declare a lag policy (R21), declare violated assumptions (E15), supply
provenance and a licence record.
**Acceptance:** a domain is onboarded **without editing `src/`**, matching the existing
third-party transform plugin test.

**TG8.2 Licence and terms provenance.** Every public dataset carries its licence, attribution
requirement and access terms in its provenance, and export refuses to emit a derived product
whose source licence forbids it. Redistribution rules differ sharply across public archives and
must be data, not folklore.

**TG8.3 The domain ledger.** For each onboarded domain: which assumptions it violated (R17),
which analyses it is therefore refused, and what onboarding cost. **If that cost is not falling
as domains accumulate, the abstraction is not working** — and the ledger is designed to make that
visible rather than deniable.

---

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
