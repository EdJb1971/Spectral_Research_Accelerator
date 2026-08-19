# SpectralEarth: Roadmap to an Automated Multiscale Discovery Engine
*Strategic engineering plan. Companion to `architecture.md`, which records what exists today.*

**Research question this platform is being built to answer:**

> At what locations, scales and times does predictively useful physical information exist in an atmospheric field, how does structure transform across scale, and which representation of the field carries that information best?

Everything below either serves that question or gets cut.

---

## 1. Honest Technical Status

Verified against the code on 2026-08-20. Every claim here is backed by captured output in
`VERIFICATION.md`; `architecture.md` Section 7 holds the full defect ledger (D1-D31).

| Area | Real status |
|---|---|
| Backend test suite | **271 passed, 1 xfailed.** Trajectory: 19 written / 1 failing / uncollectable -> 65 -> 152 -> 222 -> 271. |
| Ground-Truth Benchmark Suite | **14 PASS, 0 FAIL, 3 NOT_YET_RUNNABLE.** Nine datasets with declared known answers, five of them nulls. CI-ready via `python -m src.benchmarks`. |
| Backend compute modules | **Written, executed and tested.** `physical_core` now carries `GridSpec` + metric-aware operators; `analysis_engine` gained `spectra.py` and `climatology.py`; `transform_engine` gained the undecimated `stationary.py`. |
| Physical units and wavenumbers | **Correct as of T3.5.13.** Gradients metric-aware, spectra on a physical `k` axis, domain statistics area-weighted, and every quantity carries its units. Previously all of it was pixel-space and unlabelled (D13). |
| Turbulence regime classification | **Corrected as of T3.5.13.** Was off by one exponent for the platform's entire history and labelled Kolmogorov fields as Charney (D26). |
| Shift invariance | **Available as of T3.5.7** via the undecimated SWT: 0.00% energy spread against the decimated DWT's 153.50%. |
| Reproducibility | **Seeded generation and perturbation** (T3.5.12). Storing the seed on `ExperimentRun` and replaying a run from lineage is still outstanding, and lands with T3.5.19. |
| Declarative experiment engine + lineage | **Written and executed.** A 9-run sweep completes 9/9 and writes 28 lineage nodes / 54 edges. |
| Hypothesis engine | **Written and executed.** Still no multiple-comparison control (D8), so its p-values are not yet trustworthy. |
| FastAPI surface | **17 endpoints**, executed and smoke-tested. CORS, health and collection endpoints all added (T3.5.2, T3.5.10). |
| React frontend | **Installed and built** (T3.5.0/T3.5.3) - emits 1,378 modules with real JS/CSS. **Rendered appearance in a browser still unverified**, and it does not yet consume the health, experiment-list or benchmark endpoints. |
| DTCWT | **Still not implemented as advertised** (D1). Degenerate second tree, no orientation. Downgraded from a Phase 4 blocker to a 4E-orientation requirement once the SWT landed. |
| Registries / extension seams | **Still if/elif chains** (D15), now carrying extra `swt` branches as acknowledged debt. |
| HPC / executor seam | **Not started** (T3.5.19). Sweeps run strictly sequentially; multi-GPU assignment is cosmetic (D18). |
| Real ERA5 data | **Not started** (T3.5.18). Everything to date runs on synthetic fields and benchmarks. |

**The original baseline audit** (2026-08-19, before any of Phase 3.5) is preserved in
`architecture.md` Section 7 and in the early sections of `VERIFICATION.md`. It recorded that
nothing had ever been executed, that the test suite could not be collected, and that the
"validated / zero-error" claims in earlier revisions were aspirational. That snapshot is
history, not current status, and this table replaces it.

Phases 1 and 2 are done as *code* and now substantially done as *verified software*. Phase 3
is partial. Phase 3.5 is roughly two-thirds complete - see Section 4 for per-task status.

---

## 2. Phase Map

```
+--------------------------------------+
| PHASE 3.5: Green Baseline & Real     |  BLOCKING. Nothing downstream is
|            Multiscale Foundations    |  trustworthy until this is done.
|  - fix D1..D11, get tests green      |
|  - REAL DTCWT (q-shift, shift-inv.)  |
|  - add undecimated SWT               |
|  - Alembic                           |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
| PHASE 4: The Spectral Feature Layer  |  The scientific core.
|                                      |
|  4A  FieldSequence + ArtifactStore   |  (time axis + payload offload)
|  4B  CoefficientField + WaveletBank  |  (wiggles become data)
|  4C  ScaleSignature + SurrogateNull  |  <<< GO / NO-GO GATE >>>
|      + cross-scale lag + alpha fit   |
|  4D  SpectralFeature + Track         |  (structures through time)
|  4E  FeatureConstellation (graph)    |  (bottom-up organisation)
|  4F  Precursor & Transition Mining   |  (which small graphs at t predict
|                                      |   which big structures at t+n)
|  4G  RepresentationScore + baselines |  (which wiggle actually helps)
|  4H  Learned encoder  [OPTIONAL]     |  (ceiling estimator only)
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
| PHASE 5: Neural Weather Model as     |  FourCastNet / ClimaX becomes the
|          Downstream Judge            |  judge of representation quality,
|                                      |  not the object of study.
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
| PHASE 6: HPC & Cluster Orchestration |  Celery/Redis, Slurm, K8s, DDP.
|          (was Phase 5)               |  Bank sweeps will genuinely need it.
+--------------------------------------+
```

**Ordering rule:** 3.5 blocks everything. 4A-4B block 4C. There are **two gates**:

*   **4C (statistical gate)** - if cross-scale organisation does not beat phase-randomised surrogates, 4D-4H are cancelled and the negative result is the deliverable. 4C also blocks 4D-4H scientifically, since the null machinery is shared.
*   **4F.6 (physical gate)** - if the mining recovers no recognisable known phenomena on a period containing documented events, the pipeline is presumed broken and 4G does not start (see R10).

4H is optional and never gates anything. Phase 5 needs 4G. Phase 6 needs nothing, but makes 4B-4F affordable at scale.

---

## 3. Standing Scientific Integrity Rules

These are not a phase. They are constraints on **every** hypothesis this platform is ever allowed to emit. They exist because the single biggest risk to this project is not bugs - it is generating a large volume of confident, well-formatted, statistically meaningless claims.

### R1. The wavelet hierarchy is algebraically coupled, so every cross-scale claim needs a null model.
A coarse-scale coefficient and the fine-scale coefficients beneath it are computed from *the same pixels*; in an undecimated transform they share support outright. Cross-scale coefficient correlation is therefore **guaranteed by construction** for any broadband field - it is the basis of wavelet-domain hidden Markov tree models, and it is why wavelet coefficients are famously *not* independent across scales despite decorrelating within a scale.

**Rule:** no cross-scale, precursor, or emergence claim is emitted without comparison against a **phase-randomised surrogate ensemble** that preserves the field's power spectrum exactly while destroying phase organisation. Signal that survives = organisation. Signal that does not survive = the power spectrum, which `compute_radial_psd` already reported.

### R2. A power law is not a finding; a deviation from the surrogate power law is.
A fractional Brownian field yields a beautiful, clean, high-$R^2$ power law and contains no interesting structure whatsoever. Report $\alpha$ **always** alongside $\alpha_{\text{surrogate}}$ and the effect size between them. Never report $\alpha$ alone.

### R3. Normalise per-scale quantities, or the answer is grid bookkeeping.
Two distinct traps:
*   **Threshold sensitivity:** raw counts of "significant coefficients" per scale are dominated by threshold choice, because each scale has a different coefficient variance. Prefer threshold-free measures (energy fraction, participation ratio, Gini).
*   **Geometric bias:** in a decimated pyramid the *number of available coefficients* per scale already falls as $s^{-2}$, so a raw population count $N(s)$ yields $\alpha = 2 + \text{physics}$. Always divide by available coefficients per scale.

### R4. Temporal precedence must exceed the transform's own support.
A scale-16 coefficient has roughly 16 px of spatial support (and temporal support too, if a spatiotemporal transform is ever added). Detecting "fine scale precedes coarse scale by $\Delta t$" when $\Delta t$ is shorter than that smearing measures **filter geometry, not weather**. Enforce a minimum admissible lag per scale, and report the support alongside every lag in every result.

### R5. Multiple comparisons are controlled, always.
The existing engine tests every parameter x metric pair at $|r| \ge 0.3$ with no correction (D8). Phase 4 multiplies the search space by scales x orientations x lags x wavelet families x constellations - millions of tests. Every mining pass applies **Benjamini-Hochberg FDR control** at a declared $q$, and every emitted hypothesis records the number of tests in its family.

### R6. Predictive claims are evaluated across a strict temporal split with an embargo.
"Wavelet B predicts $t+6$" is trivially fakeable without one. `PhysicalField.split_field()` / `validate_split_guardrails()` already implement exactly this discipline for the **spatial** axis; Phase 4A extends the identical pattern to time, with a gap between train and test at least as long as the longest lag under test.

### R7. Correlation of organisation is not causation.
Small-scale structures preceding a large-scale structure may both be consequences of an unobserved process. The platform is permitted to claim **precursor** and **signature**; it is not permitted to claim **cause**. Hypothesis descriptions must use precursor/signature language, enforced in the text templates.

### R8. Reconstruction tests can hide representation defects.
D1 is the cautionary tale: the fake DTCWT reconstructs perfectly, so the round-trip test passed for months while the transform did nothing it claimed. Every transform needs a **property** test (shift invariance, orientation selectivity, energy conservation), not merely an inverse test.

### R9. Association strength is reported as lift against base rate, never as bare confidence.
"When configuration X occurs, there is an 82% historical association with pattern Y developing later" is **not a finding on its own**. If Y occurs 80% of the time regardless, 82% is noise dressed as insight. Every association must report:
*   support (how many times X occurred),
*   confidence P(Y at t+dt | X at t),
*   **base rate** P(Y at t+dt),
*   **lift** = confidence / base rate, with a confidence interval,
*   and lift against the phase-randomised surrogate ensemble (R1).

A bare confidence percentage must not be renderable in the UI. This is a hard constraint on the frontend, not only on the mining code - a big friendly "82%" on a dashboard is the single most likely way this platform ends up misleading its own author.

### R10. Discovering the already-known is the validation signal, not a failure.
Unsupervised precursor mining will surface mostly *recognisable* phenomena first - cyclogenesis, frontal waves, blocking onset - plus artifacts. That is exactly what a working pipeline should do. If the top-ranked discovered precursors are **not** recognisable to a meteorologist, the prior should be that the pipeline is broken, not that physics has been overturned. Only once the known phenomena are being recovered reliably do the unrecognisable patterns become interesting. Implemented as T4F.6.

### R11. Mine anomalies, not raw fields - and build the climatology on training data only.
The strongest signals in any raw atmospheric field are the **seasonal cycle** and the **diurnal cycle**. Mine raw ERA5 for cross-scale precursors and the top discovery will be, in an elaborate wavelet costume, *"it is winter"*. Every scale signature, feature and constellation is therefore computed on **anomalies** with respect to a climatology, with the seasonal and diurnal cycles removed.

Two further constraints, both easy to get wrong:

*   **The climatology is computed on the training period only.** Computing it over the full record and *then* splitting temporally leaks test-period information into the training normalisation - a textbook leak that R6's embargo does not catch, because it enters through the normalisation rather than through the samples.
*   **ERA5 is not stationary.** The observing system changed radically (satellite era from 1979, major assimilation changes around 1998-2000s) and there is a climate trend. Pooling 1959-2023 naively will manufacture "patterns" that are observing-system artifacts. Analyses are **era-stratified**, and any finding must be shown to hold within eras, not merely across them.

### R12. Consecutive frames are not independent samples.
Hourly ERA5 frames are massively autocorrelated. Treating 8,760 frames per year as 8,760 independent samples inflates significance catastrophically and would make every FDR threshold in R5 and every lift confidence interval in R9 meaningless.

**Rule:** report **effective sample size** derived from the field's autocorrelation time alongside every statistic; use **block bootstrap** for confidence intervals and **block-permutation surrogates** for nulls. This extends R1: surrogates must preserve *temporal* autocorrelation as well as the spatial power spectrum, or the null is easier to beat than reality and everything looks significant.

### R13. A regional domain has edges, and the coarsest scale decides how big it must be.
Regional analysis is the **primary scientific unit**, not merely a cost-saving crop - but a crop has artificial boundaries, and wavelet coefficients within one filter support of an edge are contaminated by whatever padding or windowing was applied. Those contaminated coefficients look exactly like genuine features: strong, localised, oriented along the edge. The existing Boundary-Condition Lab exists precisely because regional domains behave this way, and it becomes a validation instrument here rather than a standalone tab.

**Two hard requirements:**

1.  **Scale-dependent edge exclusion.** Feature detection, constellation extraction and every statistic operate only on the **valid interior**: `interior = N - 2 * halfwidth(j)`, where for an undecimated transform at level *j* with an *L*-tap filter `halfwidth(j) ~ (L-1) * 2^(j-1) / 2`. The mask is per-scale - coarse scales exclude far more than fine ones. The platform **computes and reports** the valid interior per scale rather than assuming it, since the exact figure depends on the chosen filter.

2.  **Crops must be sized from the coarsest scale, and the numbers are larger than intuition suggests.** Valid interior width for a 14-tap filter:

    | Level (scale) | Excluded per side | N=64 | N=128 | N=256 | N=512 |
    |---|---|---|---|---|---|
    | 1 (2) | 6 px | 52 | 116 | 244 | 500 |
    | 2 (4) | 13 px | 38 | 102 | 230 | 486 |
    | 3 (8) | 26 px | 12 | 76 | 204 | 460 |
    | 4 (16) | 52 px | **none** | 24 | 152 | 408 |
    | 5 (32) | 104 px | **none** | **none** | 48 | 304 |

    **A 64x64 crop has zero valid interior at scale 16 and about 12x12 px at scale 8.** Cross-scale analysis - the entire premise of Phase 4C - is therefore *impossible* on a 64x64 region. Practical minimum is **256x256 for four dyadic levels and 512x512 for five**. The `laptop` tier is constrained on frames, bank breadth and surrogate count, **never** by shrinking the grid below its valid-interior floor.

**Acceptance:** a test asserts that a field with a deliberately discontinuous edge produces **no** detected features inside the valid interior, and that requesting more levels than the crop can support raises rather than silently returning contaminated results.

### R14. A pattern that holds in only one region is a local quirk until shown otherwise.
Orography, coastlines and land-sea contrast produce region-specific behaviour that is real but not general. Every mined pattern is therefore re-tested on **held-out regions** with similar and dissimilar physiography, and reports where it holds. This is the domain-level counterpart to R4E.2's translation invariance, and it is the difference between "we found a thing about the Alps" and "we found a thing about the atmosphere" - both valuable, but they are not the same claim and must not be reported as though they were.

### R15. State the spectral convention next to every exponent.
In two dimensions `E(k) = 2*pi*k*S(k)`, so the 1D isotropic energy spectrum and the 2D spectral density differ by exactly one exponent. The reference values everyone quotes - 5/3 for Kolmogorov, 3 for Charney/Kraichnan - are **E-convention**. This is not pedantry: defect D26 was exactly this confusion, and it caused a synthetic field built with a textbook Kolmogorov spectrum to be labelled *"Charney/Kraichnan enstrophy cascade"* - the opposite physical regime - for the entire history of the platform. Any function returning a spectral exponent must report which convention it is in **and** its value in the other, and any regime label must be derived from the E form.

The same rule governs units. The label of a wavenumber axis is a property of the **grid**, not of the argument name the caller passed: reporting `k` in `rad m^-1` for a field that has no physical metric (D29) is the same error one layer out. Report the units of `k` and of the power, derive both from the grid, and refuse combinations that have no meaning rather than relabelling them.

### R16. Set tolerances from the mathematics, not from what the code happens to pass.
A test whose tolerance was chosen so that the current implementation passes cannot discover anything; it can only detect a change. Three defects in T3.5.13 were found solely because the bound came from theory rather than from observation:

*   **D27** (float32 `fftfreq`) sits at 5.8e-8 and is invisible to any Parseval check with a tolerance looser than 1e-7. The identity is exact in exact arithmetic, so the tolerance must be machine epsilon and nothing else.
*   The **Laplacian stencil** is validated by asserting the measured residual *equals the predicted truncation* `(k h)^2 / 12` to within 10%, not by asserting it is "small". An error of the right size for the wrong reason still fails.
*   The **gradient** is validated by **convergence order**: halving the grid spacing must quarter the error. That distinguishes "correct scheme at finite resolution" from "subtly wrong scheme" in a way no single threshold can, and it is why the residual 3.2e-6 could be confidently attributed to truncation rather than to a metric error.

Corollary, learned the hard way in this task: when a measurement contradicts the claim written in the docstring, the claim is what changes. Two such reversals are recorded in `architecture.md` section 7.2e.

---

## 3b. Cross-Cutting Engineering Standards

The R-rules keep the science honest. These keep the platform usable, extensible and trustworthy. They are peers, not subordinates: a rigorous result that cannot be reproduced, explained or extended is not world-class, it is a dead end with good statistics.

### E1. Everything pluggable is a registry, never an if/elif chain.
Today's "seams" are not extension points (D15): `_get_simulated_fallback` hard-codes three dataset ids, `list_datasets` iterates a hard-coded literal, and `_execute_action` is a 13-branch chain. Adding a data source or a pipeline action means editing core engine files, which is the definition of not modular.

**Rule:** four registries, populated by decorator, discovered at import: `@register_source`, `@register_action`, `@register_transform`, `@register_detector` (and later `@register_scorer`). Core engine files must never need editing to add a capability. Every registry entry declares its own JSON schema so the API and UI can enumerate what is available rather than hard-coding option lists.

### E2. Every data source declares capabilities and a fallback chain.
A `DataSource` protocol with `list_variables()`, `capabilities()` (does it have levels? a time axis? what native grid and units?) and `fetch(...) -> xarray.DataArray`. Sources are tried in a declared order (local NetCDF -> cached remote -> live API -> simulated), and **the resolved source is recorded in lineage**. A run must never be silently satisfied by simulated data when the researcher believed they were using ERA5: the fallback is a first-class provenance fact, surfaced in the UI, not a convenience that quietly rewrites what the experiment was.

### E3. Physical units and grid metrics are carried, not assumed.
Currently gradients are per-pixel and radial PSD bins in pixel wavenumber on a lat/lon grid whose zonal spacing varies as `cos(lat)` (D13). For a tool that classifies turbulence regimes, that is the wrong space.

**Rule:** `PhysicalField` carries units and grid metric; all differential operators take real spacing; any global or large-domain statistic is `cos(lat)`-area-weighted; PSD is reported in physical wavenumber with the isotropy assumption stated and the projection recorded. Where a lat/lon domain is too large for the isotropy assumption to hold, the platform says so rather than reporting a slope anyway.

### E4. Determinism is mandatory and captured.
Unseeded RNG in four places today (D12) means the provenance graph cannot reproduce its own runs. **Rule:** every run draws from an explicit seed, stored in `ExperimentRun`; all RNG flows through an injected generator; re-executing a run from its lineage record reproduces its metrics bit-for-bit on the same hardware. This is verified by a test, not by assertion.

### E5. Provenance is captured automatically, never declared by hand.
`code_revision` is currently a free-text string defaulting to `"unknown"` - an honour system, not provenance. **Rule:** each run automatically captures git SHA and dirty-flag, resolved dependency versions, RNG seeds, device and dtype, input data checksums (free from the content-addressed `ArtifactStore`), and the resolved data source per E2. A run is reproducible from its own record or it is not a run.

### E6. Errors are explanatory, or they are defects.
Ten API handlers currently discard the exception and return a fixed sentence (D14). **Rule:** a structured error carries the failing step name, the action, the resolved arguments, the observed vs expected shapes/dtypes/ranges, and a suggested fix; pipeline failures record which step failed and with which parameter combination, so a 200-run sweep with 3 failures tells you *which* 3 and *why*. Scientific-validity failures (embargo violation, insufficient support, lag below the support floor) get their own error class and must be legible to the researcher, not merely logged.

### E7. Synthetic ground truth is a validation instrument, not just an offline fallback.
Every mining stage must be provable against fields whose answer is known in advance - see the Ground-Truth Benchmark Suite (T3.5.17). The current synthetic generators and simulated datasets exist to keep the UI alive offline; that is a different and much weaker job.

### E8. Numerical method choice is declared and justified.
"Top research level" is a claim about specific estimator choices, so they get written down and defended in code comments and docs:
*   Continuous mutual information uses a **Kraskov-Stogbauer-Grassberger (KSG)** k-NN estimator, not naive histogram binning, which is badly biased at the sample sizes a few hundred frames provide.
*   Transfer entropy declares its embedding dimension and lag, with a stationarity check.
*   Surrogate ensembles use **IAAFT** (iterative amplitude-adjusted Fourier transform) where the field is non-Gaussian, not plain phase randomisation.
*   Power-law fits report goodness of fit and are compared against a log-normal alternative, per the Clauset-Shalizi-Newman critique - a high $R^2$ on a log-log plot is famously weak evidence for a power law.
*   Spectral and statistical accumulation runs in **float64** (D16); float32 is for visualisation only.
*   Every epsilon guard (`1e-8` appears throughout) is justified relative to the dtype's precision rather than copied.

### E9. Laptop-first: HPC accelerates, it never enables.
**Every stage must be runnable end-to-end on a single laptop at reduced settings.** The cluster makes it faster and bigger; it is never the thing that makes a capability exist. A stage that only works on a cluster is not finished.

Implemented as **tiers** - one code path, config only, no cluster-only branches:

| Tier | Grid | Levels | Frames | Bank | Surrogates | Target runtime |
|---|---|---|---|---|---|---|
| `smoke` (CI, every commit) | 64x64 | 2 | 8 | 2 families | 8 | seconds |
| `laptop` | 256x256 | 4 | 200 | 3 families | 50 | minutes |
| `workstation` | 512x512 | 5 | 1000 | full bank | 200 | hours |
| `cluster` | native (1440x721) | 6 | full archive | full bank | 1000 | Phase 6 |

**Grid sizes are floors set by R13, not preferences.** The `smoke` tier is explicitly a *mechanism* test at 2 levels - it verifies the code runs, and its results are **never** scientifically meaningful, because 64x64 cannot support cross-scale analysis. Any tier is scaled down by cutting frames, bank breadth or surrogate count - never by cutting the grid below its valid-interior floor.

Any run declares its tier, and the tier is recorded in lineage - because a finding at `laptop` tier with 50 surrogates has genuinely weaker statistics than the same finding at `workstation` tier, and the UI must not present them as equivalent.

### E10. Know the cost model, and cache along the dependency structure.
The naive reading of Phase 4C is multiplicative and hopeless:

```
families x scales x orientations x lags x surrogates(200) x frames
```

But **the terms are not all coupled**, and exploiting that is the difference between hours and weeks:

*   **Surrogates depend only on the source field, not on the wavelet.** Generate the ensemble once per field and reuse it across the entire bank: `surrogates + bank`, not `surrogates x bank`. This is the single largest available win and it is free.
*   **A `CoefficientField` for a given (field, family) is reused across every lag test** - decompose once, test many.
*   **`ScaleSignature` is a small reduction of a large object.** Once computed, the coefficient payload can be evicted; only the signature is needed for 4C.
*   The `ArtifactStore` (T4A.3) is therefore also the **cache layer**, keyed by content hash - identical inputs never recompute. Memory-mapped reads and chunking so a laptop streams coefficient fields rather than holding them resident.

**Rule:** every new stage declares its complexity in the docstring and its measured runtime per tier in `VERIFICATION.md`. A stage whose cost is not understood is not accepted, because the multiplicative terms above make an unmeasured 10x regression invisible until it is a 1000x regression.

### E11. One concurrency seam, from laptop cores to cluster nodes.
Sweeps run strictly sequentially today (`for idx, run_params in enumerate(expanded_runs)`), so a laptop uses one core while surrogate ensembles - the most parallel workload in the entire plan - wait in line.

**Rule:** all parallelism goes through a single `Executor` protocol (`submit`/`map`/`gather`) with interchangeable backends: `serial` (debugging, deterministic), `thread` (tensor work, which releases the GIL), `process` (Python-level graph mining in 4E/4F, which does not), and later `celery` / `slurm` in Phase 6. **Science code never imports a backend** - it asks the executor. Phase 6 then becomes a new backend rather than a rewrite.

Three traps this codebase will hit, each of which must be handled in the seam rather than rediscovered:

1.  **Thread oversubscription.** PyTorch already parallelises intra-op. Wrapping a process pool around it without setting `torch.set_num_threads` per worker gives `N_workers x M_torch_threads` threads fighting over the same cores, which is reliably *slower* than serial. The executor sets the per-worker thread budget explicitly; the total is a declared knob, not an emergent accident.
2.  **SQLite write contention.** `execute_experiment` commits per pipeline step, and SQLite serialises writers. Parallel runs against the default configuration produce `database is locked`. Required: WAL mode, a busy timeout, per-worker sessions (never a shared session across threads - SQLAlchemy sessions are not thread-safe), and ideally a single writer draining a result queue while workers stay read-only.
3.  **Determinism must survive parallelism (E4).** Results must not depend on worker count or completion order. Seeds are derived per-run from a root seed via `SeedSequence.spawn`, never drawn from a shared global RNG; reductions over parallel results use an order-independent or explicitly re-sorted combination.

**Acceptance for any parallel stage:** identical results at `--executor serial` and `--executor process --workers N` for several N, asserted by test. If they differ, the parallelism is wrong - not the test.

---

## 3c. Seam Inventory (the backbone)

The extension points the plan commits to. Each is a protocol with a registry, so capabilities are added in new files rather than by editing core engine code (E1). This is the spine that keeps the platform expandable:

| Seam | Protocol | Added by | Consumed by | Status |
|---|---|---|---|---|
| **Data source** | `DataSource.fetch/list_variables/capabilities` + fallback chain | T3.5.15 | all phases | replaces the hard-coded if/elif (D15) |
| **Pipeline action** | `@register_action` | T3.5.15 | experiment engine | replaces the 13-branch chain (D15) |
| **Transform** | `@register_transform` | T3.5.15 | 4B wavelet bank | makes the bank sweepable without engine edits |
| **Feature detector** | `@register_detector` | 4D | 4D/4E | alternative detection strategies |
| **Surrogate generator** | `@register_surrogate` | 4C | all mining | phase-randomised, AAFT, IAAFT |
| **Scorer term** | `@register_scorer` | 4G | representation scoring | add a score term without touching the scorer |
| **Forecaster** | `Forecaster.predict` | 4G | 4G baselines, Phase 5 | persistence / advection / FourCastNet behind one interface |
| **Executor** | `Executor.submit/map/gather` | T3.5.19 | everything parallel | serial / thread / process / celery / slurm |
| **Artifact store** | `ArtifactStore.put/get` | T4A.3 | everything large | local disk now, object store later |
| **Database** | SQLAlchemy session | exists | everything | genuinely already decoupled |

The data spine running through all of it: `PhysicalField` -> `FieldSequence` (4A) -> `CoefficientField` (4B) -> `ScaleSignature` (4C) / `SpectralFeature` (4D) -> `FeatureConstellation` (4E) -> `TransitionPattern` (4F) -> `RepresentationScore` (4G). Every seam above either produces or consumes one of these types, which is what keeps the engine coherent rather than a bag of scripts.

---

## 4. Phase 3.5 - Green Baseline & Real Multiscale Foundations

**Objective:** an installed, running, test-green platform whose multiscale machinery actually has the properties Phase 4 depends on. No new science.

### T3.5.0 Establish a verified baseline *(blocks all)*
Create a venv, `pip install -r requirements.txt`, `npm install` in `frontend/`, run the full test suite, run both servers, load the UI, exercise all seven tabs.
**Acceptance:** a `VERIFICATION.md` recording actual command output - test pass/fail counts, `npm run build` output with emitted asset names and sizes, and a screenshot per tab. No claim of "validated" appears anywhere in the repo without corresponding output in this file.

### T3.5.1 Fix the failing test and test collection *(D3, D4)*
Give `execute_experiment` an injectable session factory: `execute_experiment(experiment_id, session_factory=None)` defaulting to `SessionLocal`. Pass the test's factory in `test_experiments.py`. Add `src/tests/conftest.py` inserting the repo root on `sys.path`.
**Acceptance:** bare `pytest` collects and passes all 19 tests from the repo root.

### T3.5.2 CORS middleware *(D5)*
`CORSMiddleware` with an origin allowlist from an env var, defaulting to `http://localhost:3000`.
**Acceptance:** a cross-origin `fetch` from port 5173 succeeds without the Vite proxy.

### T3.5.3 Tailwind and PostCSS configuration *(D6)*
Add `tailwind.config.js` (content globs over `index.html` and `src/**/*.{ts,tsx}`) and `postcss.config.js`.
**Acceptance:** `npm run build` emits hashed JS **and** CSS assets into `dist/`; the served page is styled.

### T3.5.4 Fix the launcher and debug config *(D7, D11)*
`Start-Job -ScriptBlock { Set-Location $using:PWD; python -m uvicorn ... }`, plus a readiness poll against a new `/api/v1/health` endpoint before starting Vite, and a loud failure if the backend never comes up. Remove the duplicate `"name"` key in `launch.json`.
**Acceptance:** `.\start_platform.ps1` reliably brings up both services; killing the backend produces a visible error rather than a silently broken UI.

### T3.5.5 Fix the hybrid inverse *(D2)*
`inverse_hybrid` reconstructs `low + inverse_dwt(residual)`. Retain `mixing_weight` only as an explicitly-labelled lossy blend option, defaulting to exact reconstruction.
**Acceptance:** round-trip MSE for `hybrid` is at machine epsilon for all `crossover_freq` values.

### T3.5.6 Implement a real DTCWT *(D1 - highest risk item in Phase 3.5)* - **DONE**
Replace the degenerate tree-B filters with genuine Kingsbury filters: LeGall 5/3 (or near-symmetric 13/19) at level 1, q-shift filters at levels >= 2, forming a proper Hilbert pair. Combine tree outputs into **6 oriented complex subbands** ($\pm 15^\circ, \pm 45^\circ, \pm 75^\circ$).
**Acceptance (property tests, per R8):**
1. *Shift invariance:* translate a synthetic vortex 0-8 px; subband magnitude envelopes stay within 5% - and the same test **fails** against the current Haar DWT, proving the test has teeth.
2. *Orientation selectivity:* a `generate_front` at angle $\theta$ puts peak energy in the subband nearest $\theta$, swept over 0-180 degrees.
3. *Perfect reconstruction* to machine epsilon.
4. Coefficient magnitudes agree with a `PyWavelets`/`dtcwt` reference oracle within tolerance.

**Met.** Delivered as a new module `src/transform_engine/dtcwt.py` with vendored Kingsbury
coefficients (`kingsbury_coeffs.py`, generated by `tools/gen_kingsbury_coeffs.py`); 65 tests
in `test_dtcwt.py`; suite 286 -> 351.

1.  *Shift invariance:* **4.99%** subband-energy spread over an 0-8 px translation, against
    **236.11%** for the old transform - which is *identical* to a plain Haar DWT on the same
    field, the cleanest possible proof that its four trees were one filter bank.
2.  *Orientation:* passband centres measured directly at **75.0 / 45.0 / 15.3 / 164.6 /
    135.0 / 105.4 degrees**, each within 3 degrees of its declared value, and a grating at
    each subband's angle peaks in that subband.
3.  *Perfect reconstruction:* ~1e-31 MSE across 5 shapes (including odd), 4 levels, 3
    level-1 filter sets and 4 q-shift sets.
4.  *Oracle agreement:* elementwise to **1e-13** against the reference `dtcwt` package, and
    magnitudes to 1e-4 against `pytorch_wavelets` - two independent implementations. Both
    stay test-only; the coefficients are vendored so agreement is not circular.

**Two corrections worth recording.** The acceptance criterion's *vortex* probe turned out to
be too weak to distinguish any transform (0.00% for the broken one and for a plain DWT), so a
periodic sharp front is used instead and the weak probe is now an executable test in its own
right. And the first orientation table listed the right angles in the **wrong index order**;
it was corrected by measuring each subband's passband centre rather than assuming the
conventional ordering.

**Scope note.** The DTCWT is *near* shift invariant, not exact. The undecimated SWT (T3.5.7)
remains the exactly-invariant transform and stays the tool for Phase 4D tracking; the DTCWT's
distinct contribution is **orientation**, which a separable transform cannot provide because
its single diagonal band cannot separate +45 from -45 degrees.

### T3.5.7 Add an undecimated Stationary Wavelet Transform (SWT)
A trous / stationary transform in which **every scale keeps the parent grid shape**. This is what makes `CoefficientField` coordinate-clean and makes spatial cross-scale reasoning tractable; the decimated DWT stays for compression/reconstruction paths.
**Rationale:** at scale 8 a decimated transform has 1/64 the samples, so a precursor *configuration* cannot be localised relative to an emergent coarse structure with useful precision.
**Acceptance:** for all scales, `swt_coeffs[s].shape == field.data.shape`; perfect reconstruction; coefficient energy sums correctly across scales.

### T3.5.8 Alembic migrations
Initialise Alembic, autogenerate the baseline migration for the five existing tables, stop relying on `create_all` in the lifespan.
**Acceptance:** `alembic upgrade head` builds the schema from empty on both SQLite and PostgreSQL.

### T3.5.9 Data-layer honesty and hygiene *(D9)*
Add a cache-invalidation path (mtime check or explicit reload endpoint). Either implement `cfgrib` GRIB ingestion or delete the misleading comment - no third option.
**Acceptance:** dropping a new `.nc` file is picked up without restart; test with a real ERA5 cutout, not only the simulated fallback.

### T3.5.10 Add `GET /api/v1/experiments` and `GET /api/v1/health`
The frontend currently cannot list prior experiments, which makes the whole lineage/hypothesis loop unusable across sessions.
**Acceptance:** paginated experiment list rendered in the Experiment Engine tab.

### T3.5.11 Dependency and documentation truth-up *(D10)*
Add `alembic`, `scipy`, `ipywidgets`, `plotly`, `matplotlib`, and the estimator dependencies required by E8 (`scikit-learn` for k-NN neighbour queries in the KSG estimator, `networkx` for the Phase 4E attributed graphs); add `PyWavelets` and `dtcwt` as **test-only** oracles. Reconcile `architecture.md`, `README.md`, `data/README.md`.
**Acceptance:** a clean clone reaches a green test suite and a running UI using only documented commands.

### T3.5.12 Seed discipline and determinism *(D12, implements E4)* - **DONE**
Thread an explicit `torch.Generator` / `numpy.random.Generator` through `PerturbationEngine` and the simulated dataset builders. Store the seed on `ExperimentRun`.
**Acceptance:** a test re-executes a completed run from its lineage record and reproduces every metric bit-for-bit; a second test asserts that two runs with *different* seeds actually differ, so the first test cannot pass trivially.

**Met for generation and perturbation.** `benchmarks/seeding.py` derives independent streams via `SeedSequence.spawn`; `PerturbationEngine.add_noise` takes `seed=` or a threaded `generator=` instead of drawing from global torch state, and labels an unseeded run `seeded: False`. Both halves of the acceptance are asserted, including the "different seeds actually differ" control. The **stable `zlib.crc32` label hash** matters: Python's `hash()` on a string is salted per process, so a label-derived seed using it reproduces within a session and silently changes between sessions.

**Outstanding:** storing the seed on `ExperimentRun` and re-executing a completed run from its lineage record - that part lands with T3.5.19's Executor seam, where `SeedSequence.spawn` per run is what makes serial and process backends agree.

### T3.5.13 Metric-aware differential operators and physical wavenumbers *(D13, implements E3)* - **DONE**
Add units and grid metric to `PhysicalField`. Pass real spacing to every `torch.gradient` call. Area-weight domain statistics by `cos(lat)`. Report PSD in physical wavenumber, and state the isotropy assumption alongside every spectral slope.
**Acceptance:** gradient magnitude of an analytic field on a lat/lon grid matches the analytic answer in physical units at multiple latitudes - a test the current code fails at high latitude. Regime classification is re-validated after the change, since `fit_spectral_slope` interpretations were previously computed in pixel space.

**Met.** Delivered as `physical_core/grid.py` (`GridSpec`: pixel / cartesian / latlon, exact spherical cell areas, physical wavenumber axes, anisotropy reporting, provenance round-trip), `physical_core/operators.py` (metric-aware gradient, Laplace-Beltrami Laplacian, area-weighted statistics) and `analysis_engine/spectra.py` (physical-wavenumber radial spectra, count-weighted power-law fit with a standard error). 70 new tests in `test_grid_operators.py`; suite 152 -> 222 passing.

Evidence: global cell areas reproduce `4*pi*R^2` to 1.2e-16; the zonal gradient matches `cos(lon)/(R cos(lat))` to 3.2e-6 at four latitudes while unit spacing errs by ~1e4 **and by a different factor at each latitude**; gradient accuracy confirmed second-order by grid refinement; the spherical Laplacian matches three harmonic eigenvalues (l=1, l=2 zonal, l=2 sectoral) to 1e-5..1e-8; Parseval exact to 1.0000000000; injected energy slopes 5/3, 2 and 3 recovered to within 0.05.

**The mandated re-validation of regime classification found D26** - the classifier compared an S(k) exponent against E(k) reference values, mislabelling every field by one exponent. Also found D27, D28 and D29, and refuted two of my own stated hypotheses (see `architecture.md` 7.2e). Partially delivers **T3.5.20 (D17)**: the radial-PSD and coherence annulus reductions are now `torch.bincount`, asserted numerically identical to the reference Python loop.

### T3.5.14 Explanatory error taxonomy *(D14, implements E6)*
Structured exception hierarchy with step/action/shape/expectation context; API returns actionable detail (4xx for user error, 5xx only for genuine internal faults) while still not leaking stack traces to the client. Pipeline failures record the failing step and parameter combination.
**Acceptance:** a deliberately malformed pipeline (shape mismatch between two steps) produces a message naming both steps, both shapes, and the fix. Sweep failure reporting is tested with a partially-failing 20-run sweep.

### T3.5.15 Registries for sources, actions, transforms and detectors *(D15, implements E1, E2)*
Decorator-based registries; refactor `_execute_action`'s 13 branches and the adapter's hard-coded dataset list onto them. Define the `DataSource` protocol with capability declaration and an explicit fallback chain, recording the resolved source in lineage.
**Acceptance:** a new data source and a new pipeline action are each added in a **new file only**, with zero edits to `engine.py`, `adapters.py` or `main.py`, and both appear automatically in `GET /api/v1/actions` and `GET /api/v1/data/datasets`. A run served by the simulated fallback is visibly labelled as such in the API response and the UI.

### T3.5.16 Precision policy *(D16, implements E8)*
`PhysicalField(dtype=...)` with float64 available; float64 by default for spectral accumulation, surrogate statistics and fitting; float32 retained for display paths.
**Acceptance:** a documented benchmark of the precision/performance trade-off, so the default is a measured choice rather than a preference.

### T3.5.17 Ground-Truth Benchmark Suite *(implements E7 - the scientific test harness)* - **DONE (9 of 9 datasets; 3 gates pending their stage)**
A named module (`src/benchmarks/`) of synthetic fields and sequences **with analytically known answers**, which every later phase must pass before it is allowed to touch real data:

| Benchmark | Known answer | Gates |
|---|---|---|
| Pure sinusoid at scale $s$ | all energy at scale $s$ | 4C ScaleSignature |
| Fractional Brownian field, given $H$ | exact $\alpha$; **no** organisation | 4C alpha fit + surrogate null (must return null) |
| Synthetic cascade with injected cross-scale coupling | known coupling scales and lag | 4C cross-scale dependency |
| Advected vortex, known trajectory and scale evolution | exact positions, scale-doubling time | 4D tracking |
| Planted three-feature precursor configuration | known geometry, lag, lift | 4E clustering, 4F mining |
| Same configuration, translated / rotated / rescaled | must match the original pattern | 4E invariance (and the scale toggle) |
| Pure noise sequence | **nothing**; every stage must return null | all stages - the false-positive check |
| Seasonal + diurnal cycle only, no weather | **nothing**; the cycles must be removed by R11, not "discovered" | 4C-4F - the single most likely false discovery on real ERA5 |
| Autocorrelated red-noise sequence | **nothing**; catches R12 violations (inflated significance from dependent samples) | 4C-4F |

**Acceptance:** the suite runs in CI on every commit. The two null benchmarks (fBm, pure noise) are the most important entries: **any stage that reports a finding on them is broken**, and that is far more informative than a stage that finds something on real data.

**Met.** `src/benchmarks/` with a registry, nine datasets, declared known answers and a CLI (`python -m src.benchmarks`, exit code 1 on any failure) plus `GET /api/v1/benchmarks` and `POST /api/v1/benchmarks/run`. **14 PASS, 0 FAIL, 3 NOT_YET_RUNNABLE**; 46 tests in `test_benchmarks.py`; suite 222 -> 271.

Five null benchmarks, not two: fBm, white-noise field, pure-noise sequence, seasonal+diurnal, and red noise. `test_no_null_benchmark_ever_fails` is separated from the general pass count and carries its own failure message.

Gates whose stage does not exist yet report **`NOT_YET_RUNNABLE`** naming the missing stage, never a skip - a skipped test is invisible in a summary and would let "all green" mean "we never looked". Pending: `4D.tracking`, `4C.surrogate_null`, `4E.invariance`.

Two measured results: naive significance testing on AR(1) data (phi = 0.85, ESS 32.2) rejects a true null at **40.0%** versus **2.3%** ESS-corrected (R12); and a time-of-day bin climatology leaves **15.5%** of variance on a cycles-only sequence versus **0.017%** for harmonic regression (R11), which motivated building `analysis_engine/climatology.py` with split-aware fitting (R6). Found **D30** (a solve whose answer depended on what ran before it) and **D31**.

### T3.5.18 Cloud-native ERA5 via Zarr, with a rechunked local cache *(implements E2; unblocks all of Phase 4)*
The raw material is not a constraint: **ERA5** (ECMWF/Copernicus reanalysis, hourly, 1940-present, ~31 km, up to 137 levels) is packaged analysis-ready by **WeatherBench 2** as public Zarr on GCS, 1959-2023, at resolutions up to the full 0.25 degree / 1440x721 grid. Zarr is the right adapter, not bulk NetCDF download: `xarray.open_zarr` over `fsspec`/`gcsfs` opens the whole archive lazily and fetches only the chunks a crop touches, so "crop aggressively" stops being a separate step and becomes simply how the store works.

**The trap that decides whether this is fast or unusable: chunk alignment.** SpectralEarth's access pattern is *many timesteps over a small spatial region* - the exact opposite of a store chunked as one full global field per timestep. Against such a layout, every frame of a 64x64 crop drags a whole planet across the network. Reading a decade for one region could move terabytes to analyse megabytes.

**Therefore the adapter has two stages, and the second is not optional:**
1.  **Query** the remote Zarr lazily for the requested region, time window, levels and variables.
2.  **Materialise once into a local, analysis-optimised Zarr cache, rechunked time-contiguous for that region**, content-hashed and registered in the `ArtifactStore`. All Phase 4 work reads the local cache.

This is also excellent provenance (E5): a crop is *exactly* specifiable as `{store URI, dataset version, variable, time range, bbox, levels}` plus a content hash of the materialised snapshot - far stronger than "someone put a `.nc` in a folder", and it makes any finding reproducible by anyone with an internet connection.

**Acceptance:** the adapter reports the remote store's chunk structure and **warns when the requested access pattern is chunk-hostile**; a 256x256 region (the R13 floor for four levels) over one year materialises within the `laptop` tier budget with recorded bytes-transferred; the adapter refuses a crop too small for the requested number of levels, naming the minimum; re-requesting an identical crop hits the cache with zero network traffic; the crop specification round-trips from the lineage record to an identical re-materialisation.

### T3.5.19 Executor seam and concurrency-safe persistence *(implements E11)*
Define the `Executor` protocol with `serial`/`thread`/`process` backends; route the sweep loop and surrogate ensemble generation through it. Enable SQLite WAL mode and a busy timeout; give each worker its own session; move seed derivation to `SeedSequence.spawn`.
**Acceptance:** a 20-run sweep produces byte-identical results under `serial` and under `process` with 2, 4 and 8 workers; a concurrency stress test drives parallel writes without a single `database is locked`; measured speed-up recorded per tier in `VERIFICATION.md`, including the deliberately-misconfigured oversubscribed case to demonstrate the thread-budget knob matters.

### T3.5.20 Vectorise the radial and distance binning *(D17, implements E10)*
Replace all five per-bin loops with `torch.bincount` / `scatter_add` single-pass reductions (`compute_radial_psd`, `compute_spectral_coherence`, `decompose_by_boundary`, `analyze_boundary_artefacts`, and the Tukey window construction). Batch the transform over `(time, scale, orientation)` as tensor dimensions rather than Python iteration.
**Acceptance:** bit-comparable results to the current implementation (within float tolerance) on the benchmark suite, plus a recorded speed-up on a 512x512 field. This is a prerequisite for the `laptop` tier being honest rather than aspirational.

### T3.5.21 Device and thread policy *(D18, implements E9)*
Extend `get_execution_device` to CUDA -> MPS -> CPU with an explicit override; configure CPU thread counts; make every kernel device-agnostic (the current code mixes CPU-constructed tensors with a selected device in places).
**Acceptance:** the `smoke` tier passes identically on CPU, CUDA and MPS, with results agreeing within float tolerance across all three.

---

## 5. Phase 4 - The Spectral Feature Layer

**Objective:** promote transform output from a *result* to a *first-class spatiotemporal dataset*, then mine it in both directions - top-down (what fine structure constitutes a coarse pattern?) and bottom-up (what collective fine organisation precedes a coarse pattern?).

The conceptual ladder:

```
FieldSequence            (4A)   PhysicalField over time
      |
      v
CoefficientField         (4B)   position x scale x orientation x variable x time
      |
      +---> ScaleSignature      (4C)   normalised activity per scale per time
      |          |
      |          +---> cross-scale lagged dependency  (4C)   does s at t predict s' at t+dt?
      |          +---> scale-population exponent      (4C)   N(s) ~ s^-alpha
      |
      +---> SpectralFeature     (4D)   a localised coefficient structure
                 |
                 +---> SpectralFeatureTrack   (4D)   one structure through time
                 +---> FeatureConstellation   (4E)   attributed graph over many structures:
                            |                        nodes = scale/strength/velocity/d(scale)/dt
                            |                        edges = distance/bearing/scale ratio/
                            |                                convergence/time offset
                            v
                       TransitionPattern      (4F)   which small graphs at t precede
                            |                        which large structures at t+n
                            v                        (support, base rate, lift, surrogate lift)
                       RepresentationScore    (4G)   which wiggle is actually useful
```

Every stage is gated by the Section 3 rules.

### Phase 4A - `FieldSequence` and `ArtifactStore`

**T4A.1 `FieldSequence` (`src/physical_core/sequence.py`)**
Ordered `PhysicalField`s plus a time coordinate; validates shape and coordinate consistency across frames. Methods: `.at(t)`, `.map(fn)`, `.to_tensor()` -> `(T, H, W)`.

**T4A.2 `split_temporal(train, val, embargo)` *(implements R6)***
Mirrors the existing spatial guardrail API deliberately: `validate_temporal_guardrails(other)` raises on any time overlap **or** on an embargo gap shorter than the longest lag under test.
**Acceptance:** a deliberate leakage attempt (overlapping windows, or embargo < max lag) raises `ValueError`, with a test asserting the raise.

**T4A.3 `ArtifactStore` (`src/artifact_store/store.py`)**
Content-addressed on-disk store (`.npz`/`.pt` under `artifacts/`, SHA-256 keyed). Steps exchange `ArtifactHandle {ref, shape, dtype, sha256, summary}`. `resolve_value` learns to dereference handles. Lineage `value` columns store the **handle plus summary**, never the payload.
**Rationale:** a `CoefficientField` over 10 frames x 64x64 x 4 scales x 6 orientations is ~10M floats; the current `.tolist()` JSON seam cannot carry it.
**Acceptance:** a 100-frame 64x64 sequence round-trips; every lineage row stays under 4 KB; `analyze_boundary` no longer embeds a full padded field in its node.

**T4A.4 Adapter sequence slicing**
`slice_sequence(dataset_id, variable, time_range, level, lat_range, lon_range) -> FieldSequence`, plus a `slice_sequence` pipeline action.

### Phase 4B - `CoefficientField` and the Wavelet Bank

**T4B.1 `CoefficientField` (`src/transform_engine/coefficient_field.py`)**
Axes `(time, scale, orientation, y, x)` with `wavelet_family`, `source_variable`, and phase where the transform is complex. Backed by SWT/DTCWT so all scales share the parent grid, inheriting coords from the source `PhysicalField`. Deliberately mirrors `PhysicalField`'s API so it can flow through analysis the same way.
**Acceptance:** coordinate alignment test across all scales; perfect reconstruction per family; `.summary()` produces a lineage-safe dict.

**T4B.2 `WaveletBank` config and sweepability**
Declarative `wavelet_bank: {families: [...], scales: [...], orientations: [...]}`. Because these are ordinary parameter-matrix entries, `expand_parameter_matrix` picks them up with **no engine changes** - one of the genuinely free wins in this plan. Keep the 1,000-combination guard.

**T4B.3 New pipeline actions:** `decompose_bank`, `extract_scale_signature`.

**T4B.4 Pressure level as a bank dimension.** ERA5 is `time x level x lat x lon x variable`, and the vertical axis is currently only a *selector* (pick 500 hPa). But the canonical atmospheric precursor relationship is inherently vertical: an upper-level trough preceding surface cyclogenesis. Treat level as a first-class bank dimension alongside scale and orientation - decompose per level, and let 4E constellations span levels with **vertical offset as an edge attribute**.
**Scope discipline:** this is *2D-per-level*, not 3D wavelets. Full 3D transforms are deliberately out of scope for Phase 4 on cost and complexity grounds; per-level decomposition with cross-level edges unlocks the most physically famous precursor structure at a fraction of the price, and is the natural target for T4F.6's known-phenomenon gate.

### Phase 4C - `ScaleSignature`, `SurrogateNull`, Cross-Scale Dependency <<< THE GATE >>>

This phase is self-contained: it needs **nothing** from 4D-4G, and it is where the project finds out whether the central idea is real.

**T4C.1 `ScaleSignature` (`src/analysis_engine/scale_signature.py`) *(implements R3)***
Per `(t, scale)`: energy fraction, participation ratio, Gini coefficient, and - reported but never primary - threshold-based counts with the threshold recorded. Direct generalisation of the existing `compute_wavelet_energy`.
**Acceptance:** signature of a pure sinusoid concentrates at its scale; signature of white noise is flat in energy fraction; both invariant to a global amplitude rescale.

**T4C.2 `SurrogateNull` (`src/analysis_engine/surrogates.py`) *(implements R1)***
Phase-randomised and AAFT surrogate generators preserving the PSD (and, for sequences, per-frame PSD). Ensemble of N (default 200) with effect sizes and empirical p-values against the ensemble.
**Acceptance:** surrogate PSD matches source PSD within tolerance while phase correlation is destroyed; a known-organised synthetic field scores significantly, and a fractional Brownian field does **not**.

**T4C.3 Cross-scale lagged dependency *(implements R4)***
Lagged mutual information and transfer entropy over the $A_t(s)$ matrix, for all scale pairs and admissible lags, with the per-scale support floor enforced and reported.
**Acceptance:** recovers an injected cross-scale coupling in a synthetic cascade; reports **null** on a phase-randomised version of that same field.

**T4C.4 Generic `fit_power_law` and scale-population exponent *(implements R2, R3)***
Factor the log-log least-squares core out of `fit_spectral_slope` into a reusable `fit_power_law(x, y, x_min, x_max)`; keep the Charney/Kolmogorov interpretation as a thin turbulence-specific wrapper so existing behaviour is unchanged. Apply it to $N(s)$ / energy vs scale, normalised per R3, always reported against $\alpha_{\text{surrogate}}$.
**Acceptance:** existing `fit_spectral_slope` tests still pass unchanged; recovers a known exponent from a synthetic multifractal.

**T4C.5 FDR control utility *(implements R5, fixes D8)***
Benjamini-Hochberg helper; retrofit it onto the **existing** hypothesis engine as well as all new mining. Every `Hypothesis` row gains `n_tests_in_family`, `p_value`, `q_value`, `surrogate_effect_size`.

**T4C.6 GATE REVIEW.** Written verdict: does cross-scale organisation exceed the surrogate ensemble at $q < 0.05$, at lags above the support floor, on real ERA5 data?
*   **Pass** -> proceed to 4D.
*   **Fail** -> stop. Write up the negative result. It is a genuine, publishable-shaped finding that the observed cross-scale coefficient structure is explained by the power spectrum alone, and it saves 4D-4G entirely.

### Phase 4D - `SpectralFeature` and `SpectralFeatureTrack`

**T4D.1 Feature detection.** Local maxima in `CoefficientField` above a surrogate-calibrated threshold, with sub-pixel localisation; yields `{id, t, y, x, scale, orientation, strength, phase}`.

**T4D.2 Tracking.** Frame-to-frame association with scale/orientation gating; nearest-neighbour first, Hungarian assignment (`scipy.optimize.linear_sum_assignment`) once it needs to be respectable. Yields `SpectralFeatureTrack {feature_id, wavelet, start_time, positions[], scales[], strengths[], orientations[], velocity, scale_velocity, lifetime}`.
**Acceptance (this is where D1 comes home to roost):** a synthetic vortex advected on a known trajectory with a known scale evolution is recovered with position error < 1 px and correct scale-doubling detection. **This test cannot pass without T3.5.6** - with shift-variant Haar, coefficients flicker with grid parity and the tracker reports births and deaths that are pure aliasing.

**T4D.3 Narrative summaries.** Human-readable track descriptions ("travelled southeast over six frames while dominant scale doubled and coefficient energy rose 43%"), using precursor/signature language per R7.

### Phase 4E - `FeatureConstellation` as an attributed graph (the bottom-up direction)

The primitive here is **an attributed graph, not a summary descriptor.** A pairwise distance/bearing histogram cannot express "A and B are weakening at small scales while C strengthens one scale up" - it throws away which node carries which behaviour. The graph keeps it.

**T4E.1 Constellation extraction as attributed graphs.** A constellation is a small graph over co-occurring features:

*   **Node attributes** (from the 4D tracks): position, scale, orientation, coefficient strength, phase, `d(strength)/dt`, `d(scale)/dt` (the `scale_velocity` already produced by T4D.2), velocity vector, age.
*   **Edge attributes:** separation distance, bearing, **scale ratio**, strength ratio, **relative radial velocity** (signed: converging vs diverging), and time offset between the two features' onsets.

Start with **pairs and triples only** - O(n^2)/O(n^3), not exponential; pairs alone already carry relative geometry, scale ratio and convergence. Hard cap on cardinality, raised only if triples prove informative.
**Rationale:** thousands of local features in arbitrary arrangements is frequent-subgraph mining, not a Cartesian sweep. The 1,000-combination guard in `expand_parameter_matrix` neither applies nor protects here.

**T4E.2 Invariance by construction.** Every attribute above is expressed relatively, so a configuration is recognised anywhere on the grid: distances normalised by the participating features' scales, bearings measured relative to the constellation's own principal axis, strengths normalised within the constellation. Translation and rotation invariance therefore fall out of the representation rather than being bolted on afterwards. Scale invariance is a **separate, explicit toggle** - normalising by absolute scale answers "does this configuration recur at other scales?", and **that query is the universality hook.** It must be first-class in the API from day one, and it must be possible to run the same mining pass with it on and off and compare.

**T4E.3 Approximate matching by clustering, not exact subgraph isomorphism.** Exact attributed-subgraph isomorphism is the wrong tool: the attributes are continuous and noisy, so exact matching would fragment one physical configuration into hundreds of near-duplicate patterns. Instead define a distance metric on attributed graphs (attribute-weighted, with a declared weighting per attribute) and **cluster** constellations in that space; a "pattern" is a cluster centroid with a tolerance radius. More robust, and tractable.
**Acceptance:** a synthetic three-feature configuration, replayed at a different grid location, a different rotation and with 10% attribute jitter, lands in the same cluster; replayed at a different absolute scale it lands in the same cluster **only** when scale invariance is enabled.

**T4E.4 Minimum-support mining.** Support-threshold miner with early pruning over the clustered patterns; configurable minimum support; support counts reported with every pattern; hard wall-clock and candidate-count budgets so a sweep cannot silently run for days.

### Phase 4F - Transition and Precursor Mining

**T4F.1 New substrate.** The existing engine mines *scalar run metrics* out of flattened `results` JSON and structurally **cannot** express `A4 -> A8 -> B8 -> C16`. This needs an event table and sequence counting, not another correlation loop.

**T4F.2 Sequence mining.** Frequent sequences over feature/constellation events with support and confidence; recurrence-interval detection (does the *gap* between events recur?).

**T4F.3 Precursor tests *(R1, R4, R5, R7, R9)***. The core question, stated precisely: **which small attributed graphs at $t$ predict which larger structures at $t+\Delta$?** For each candidate constellation pattern, does its appearance raise the probability of a coarse-scale structure emerging at $t+\Delta$ *above its base rate*, above the surrogate ensemble, at admissible lags, under FDR control? Every emitted rule carries support, confidence, base rate, lift, CI and surrogate-corrected lift (R9), and uses precursor/signature language, never causal language (R7).

**T4F.4 Bidirectional queries.** Top-down (given a coarse structure, what fine configurations most commonly preceded it?) and bottom-up (given a fine configuration, what coarse structure follows?) share one query engine over the same tables. Neither direction requires telling the platform what a cyclone or a front is - the features are discovered from coefficient structure alone, and physical interpretation is the researcher's step afterwards.

**T4F.5 Evidence projection - the "show the why" step.** A mined rule is useless to a researcher as a row in a table. Every pattern must project **back onto the map**: the specific grid cells, lat/lons, pressure levels and timestamps whose coefficients constitute it, overlaid on the physical field they came from. This is why T3.5.7's undecimated SWT matters beyond convenience - because every scale shares the parent grid, the inverse mapping from a coefficient structure to its physical footprint is direct rather than an interpolation guess.

Each rule therefore carries: its constituent features' physical coordinates, the field values there, the anomaly magnitudes, the per-scale contribution, and the list of **historical instances** supporting it. A researcher must be able to answer "why does the platform believe this?" by looking at weather, not at coefficients.
**Acceptance:** for a planted synthetic precursor, projection recovers the exact grid cells that were planted. On real ERA5, a `recognised` pattern from T4F.6 projects onto a physically sensible footprint - verified by eye, which is a legitimate acceptance test for this specific task.

**T4F.6 Known-phenomenon cross-reference *(implements R10 - this is a validation gate, not a feature)***. Maintain a small reference catalogue of known synoptic precursor phenomena with their expected scale ranges, lags and geometries. Every mined pattern is checked against it and labelled `recognised` / `unrecognised`.
**Acceptance:** on a real ERA5 period containing a documented cyclogenesis event, the mining pass ranks a `recognised` pattern corresponding to it in the top results. **If nothing recognisable is recovered, the pipeline is presumed broken and 4G does not start.** Only then are `unrecognised` high-lift patterns promoted for human review.

### Phase 4G - `RepresentationScore`

**T4G.1 Baseline forecasters *(de-risks Phase 5)***
Persistence and optical-flow advection. Cheap, CPU-only, no model weights.
**Additionally, and better:** WeatherBench 2 publishes *precomputed forecasts* from operational and ML models (IFS HRES, GraphCast, Pangu, and others) alongside the ERA5 ground truth. That means `RepresentationScore` can be validated against **real model forecast errors without running a single model**, which pulls the central Phase 5 validation forward into Phase 4G at near-zero cost. If a representation's score fails to track real GraphCast error structure here, the scoring formula is wrong and we learn it before integrating anything.
**Rationale:** if the scorer cannot distinguish representations against persistence, it will not distinguish them against FourCastNet either - and that is worth learning in a week rather than after a model integration.

**T4G.2 The score.** Recurrence + sparsity + temporal persistence + cross-scale coherence + spatial coherence + **predictive information** + generalisation. Predictive information is the **anchor** term; without it the score rewards whichever wavelet is busiest.

**T4G.3 Strict evaluation *(R6)***. All predictive terms computed only across `split_temporal` with embargo. Report per-term contributions, never a single opaque number.
**Acceptance:** the scorer ranks a deliberately-crippled representation (e.g. random orthogonal basis) below a physically-appropriate one, and ranks a busy-but-uninformative wavelet below a sparse-but-predictive one - the distinction that motivated the whole design.

**T4F.7 Cross-region generalisation *(implements R14)***. Re-test every candidate pattern on held-out regions; report where it holds and where it fails, with physiography noted. A pattern is labelled `regional` or `general` accordingly.

**T4F.8 Follow-up experiment proposals.** Extend the existing `_propose_numerical_followup` / `_propose_categorical_followup` pattern to propose experiments that *test* a discovered precursor - the platform closing its own loop.

### Phase 4H - Learned graph encoder (OPTIONAL, ceiling estimator only)

There are two ways to answer "which little graphs predict which bigger graphs", and the choice matters more than it looks:

*   **(a) Symbolic** - frequent attributed-pattern mining with support and lift (T4E, T4F). Auditable, human-readable, slow, combinatorial. **This is the deliverable.** It is what makes SpectralEarth a research instrument rather than another opaque predictor.
*   **(b) Learned** - a GNN over the constellation graphs trained to predict coarse-scale state at $t+\Delta$. More expressive and much faster, but it reintroduces exactly the interpretability problem this platform exists to dissolve. Using a black box to explain a black box (Phase 5's FourCastNet) would be self-defeating.

**Therefore (b) is admitted in one narrow role only: as a ceiling estimator.** Train the GNN purely to measure *how much predictable signal the graph representation contains at all*, which bounds how good the symbolic rules in 4F could ever become. A large gap between the GNN's skill and the symbolic rules' skill means the mining is leaving signal on the table and 4E's attribute set or matching tolerance needs revisiting. A small gap means the symbolic rules have extracted essentially everything available, and the work is done. The GNN's predictions are **never** presented as findings.
**Acceptance:** reports one number - the symbolic-vs-learned skill gap on the held-out temporal split - and nothing else.

---

## 6. Phase 5 - Neural Weather Model as Downstream Judge

**Reframing:** FourCastNet is no longer the goal. It is the *instrument* that tells us whether `RepresentationScore` predicts real forecast skill.

*   **T5.1** Integrate a pre-trained model (GraphCast, FourCastNet or ClimaX) behind the `Forecaster` seam, with `predict` / `fine_tune` pipeline actions. Note that GraphCast's own tooling points researchers at the same WeatherBench 2 ERA5 Zarr used by T3.5.18, so the data path is already built by this point - only the weights and the inference wrapper are new.
*   **T5.2** Feed each candidate representation (raw, Fourier, wavelet, hybrid, selected multiscale) and correlate `RepresentationScore` against actual forecast skill. **If the correlation is null, 4G's scoring formula is wrong and must be revised** - this is the external validation of the entire Phase 4 thesis.
*   **T5.3** Sensitivity benchmarking: perturbed initial fields via the existing `PerturbationEngine`, tracking error trajectories and structural degradation - now expressible per scale and per orientation rather than as a single scalar.
*   **T5.4** Analyse FCN *error* fields with the same coefficient machinery: at what scales, locations and times does the model fail? This is the question the platform's name has always promised.

---

## 7. Phase 6 - HPC and Cluster Orchestration

Unchanged in substance from the previous Phase 5, but explicitly **an accelerator, not a gate** (E9). Every capability already works at `laptop` tier before Phase 6 begins; Phase 6 buys resolution, archive length, bank breadth and surrogate count - that is, statistical power and scope, not new features. If any Phase 4 capability turns out to *require* Phase 6, that is a design failure in Phase 4 to be fixed there, not deferred to here.

*   **T6.1** Celery + Redis as a **new `Executor` backend** (E11), not a rewrite - if T3.5.19 did its job, science code is untouched. Adds retries, cancellation, progress and timeouts.
*   **T6.2** Slurm adapter (`.sbatch` generation, `sbatch` submission, state polling) and Kubernetes job orchestrator (ephemeral GPU pods).
*   **T6.3** DDP/FSDP multi-GPU scaling, channels-last layouts, FP16/BF16 mixed precision. Note that `get_execution_device` already round-robins GPUs but sweeps run sequentially, so this is where that becomes real rather than cosmetic.
*   **T6.4** Surrogate ensembles are embarrassingly parallel and are the natural first workload to distribute.

---

## 8. New Database Tables (Phase 4)

Added via Alembic (hence T3.5.8 being a prerequisite, not a nice-to-have). Kept **separate** from `LineageNode`/`LineageEdge`: physical structures are not provenance records and conflating them would be a category error.

| Table | Phase | Holds |
|---|---|---|
| `scale_signatures` | 4C | per (run, t, scale) normalised activity measures |
| `surrogate_baselines` | 4C | ensemble statistics per (run, test family) |
| `spectral_features` | 4D | localised coefficient structures |
| `feature_tracks` | 4D | tracked structures with kinematics |
| `constellations` | 4E | constellation instances (graph identity, cluster assignment, frame) |
| `constellation_nodes` | 4E | per-instance node attributes (scale, strength, d/dt, velocity) |
| `constellation_edges` | 4E | per-instance edge attributes (distance, bearing, scale ratio, radial velocity, time offset) |
| `constellation_patterns` | 4E | cluster centroids + tolerance radii = the "patterns" themselves |
| `transition_patterns` | 4F | recurring sequences and precursor rules: support, confidence, base rate, lift, CI, q-value, surrogate lift, `recognised` flag |
| `representation_scores` | 4G | per-representation score breakdowns |

`Hypothesis` gains `p_value`, `q_value`, `n_tests_in_family`, `surrogate_effect_size`, `evidence_refs`.

---

## 9. Frontend Additions

Deliberately cheap, because the existing components already fit:

*   **Scale-signature heatmap** (time x scale) - reuses `Heatmap2D` unchanged.
*   **Cross-scale lag plot** and **alpha fit with surrogate band** - reuse `LineChart`.
*   **Emergence graph** (features and constellations across space/scale/time) - reuses `LineageGraph.tsx` **as-is** for rendering. The framing of "a dependency graph across space, scale and time" is precisely a lineage graph for *physics* rather than for provenance, so the existing SVG node-link renderer and tooltip inspector already speak the right visual grammar.
    *   *But do not reuse the lineage **tables**.* `LineageEdge` carries a single `relation` string and no numeric columns, so it cannot hold distance, bearing, scale ratio or radial velocity. The visual grammar transfers; the schema does not. Hence the separate attributed tables in Section 8.
*   **Constellation inspector** - node/edge attribute panel for a selected pattern, with the `recognised` label and its matched known phenomenon where applicable.
*   **Lift, not confidence** - per R9, every association in the UI shows support, base rate, lift and surrogate-corrected lift. A bare confidence percentage is not a renderable component.
*   **Case-study explorer** - *the most important new view, and the one that makes the platform a research instrument rather than a claim generator.* Given a discovered rule, enumerate its supporting historical instances and let the researcher step through them frame by frame on the regional map, with the constellation overlaid, the valid-interior mask shown, and the per-scale contribution visible. This is how a human decides whether a mined pattern is physically meaningful - and it is the only way R10's "is this recognisable?" judgement can actually be made.
*   **Valid-interior overlay** - every regional map shows the R13 exclusion margin for the current coarsest scale, so it is always obvious how much of the domain is genuinely being analysed.
*   **Representation leaderboard** - new, simple table with per-term score breakdown.
*   **Surrogate-comparison affordance** - every displayed finding shows its null-model comparison inline. A finding without a visible null is not shippable UI.

---

## 10. Definition of Done

A phase is done when **all** of the following hold:

1.  Tests pass in CI, including the Ground-Truth Benchmark Suite (T3.5.17) - and critically, the **null benchmarks return null**.
2.  Acceptance criteria are met with **recorded command output** in `VERIFICATION.md`. No claim of "validated" exists anywhere without corresponding output there.
3.  Every new capability was added through a registry (E1), not by editing a core engine file.
4.  Every new result is reproducible from its own lineage record (E4, E5), verified by a test rather than asserted.
5.  Failure modes produce explanatory errors (E6), tested with deliberately broken input.
6.  Any statistical claim carries its null comparison, effect size and q-value (R1, R5, R9), and any predictive claim its temporal split (R6).
7.  `architecture.md` is updated to describe what now exists, **including any new defects discovered while building it**. The Section 7 ledger only shrinks when something is genuinely fixed; it grew from 11 entries to 16 during planning, which is the process working as intended.
8.  The UI surfaces the result *with* its provenance, its null comparison, its tier and its resolved data source - never a bare number.
9.  The stage runs end-to-end at `laptop` tier (E9), with measured per-tier runtimes and a declared complexity recorded in `VERIFICATION.md` (E10).
