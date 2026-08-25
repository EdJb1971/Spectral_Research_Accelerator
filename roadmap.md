# SpectralEarth: Roadmap to an Automated Multiscale Discovery Engine
*Strategic engineering plan. Companion to `architecture.md`, which records what exists today.*

**Research question this platform is being built to answer:**

> At what locations, scales and times does predictively useful physical information exist in an atmospheric field, how does structure transform across scale, and which representation of the field carries that information best?

Everything below either serves that question or gets cut.

---

### External research alignment and the claim boundary

The supplied EGU poster *Spectral representations for regional AI-based weather prediction* by Emily
O'Riordan (Victoria University of Wellington) is relevant external context, not a result of
this project. It asks whether Fourier, DCT, Haar, db2 and DTCWT representations change forecast
skill in a controlled lightweight neural model over the New Zealand ERA5 domain at 850 hPa.
Its initial results motivate this roadmap's boundary, transform and downstream-forecast work.

SpectralEarth currently supplies much of the **measurement apparatus** needed around that
question: the compared 2D transforms, physical grid metadata, boundary diagnostics,
scale/orientation summaries, surrogate nulls, corrected inference, provenance and a verified
single-variable regional Zarr crop path. T5.1a-e now supplies the batched transform API and
T5.2a-b supplies the leakage-safe five-variable PyTorch dataset constructor plus bounded-memory,
worker-safe local Zarr access over a materialised crop. It does **not** yet supply the viable
multi-year real NZ crop (D43), an actual independent
ERA5 route cross-check or the poster's learned forecasting experiment. No current task has
implemented its neural architecture, autoregressive training schedule or matched forecast
comparison. Phase 5's remaining model work is a proposal to integrate that judge, not evidence
that the laboratory architecture or experiment exists inside SpectralEarth. T5.3a-b supplies
the generic adapter, persistence/smoke baselines, verified artifact contract and held-out
evaluator described below; Adam's model and weights have not been supplied or executed.

Consequently:

* poster findings must be labelled external and cannot be used as SpectralEarth validation;
* transform properties (localisation, directionality, shift behaviour) must not be promoted to
  forecast-skill claims without a controlled predictive experiment;
* a future comparison must hold the model, parameter budget, training schedule, data split and
  evaluation protocol fixed enough to isolate representation choice; and
* persistence and appropriate operational/ML baselines, uncertainty across seeds or folds,
  per-variable/per-lead-time families and compute cost belong in the acceptance criteria.

The practical priority is the laboratory's existing regional model. Phase 5A therefore begins
with an importable PyTorch representation layer and a provenance-carrying regional dataset,
then wraps the existing model. Global pre-trained models remain useful later generalisation
judges; they are not allowed to delay a usable local vertical slice.

The clever outcome is not a system that automatically agrees with the motivating poster. It is
one that can explain which measured property of a representation predicts held-out forecast
skill, show the counterexamples, or report that no robust relationship survives.

---

## 1. Honest Technical Status

Verified against the code on 2026-08-22. Every claim here is backed by captured output in
`VERIFICATION.md`; `architecture.md` Section 7 holds the full defect ledger (D1-D59, of which **57 fixed, 1 partial (D18), 1 open (D43)**).

The numbers in this table are checked by `src/tests/test_documentation.py`, which parses them
out of this file and compares them against the source. That guard exists because this table
had itself gone stale — it claimed 271 passing tests and "DTCWT still not implemented as
advertised" several slices after both had changed. A status section that cannot fail is not a
status section, it is a memory.

| Area | Real status |
|---|---|
| **Phase progress** | **Phase 3.5 complete** (25 tasks). **Phase 4A, 4B and the pre-gate 4C instrument are complete:** T4A.1-4, T4B.1-4, T4C.1-5h. T4C.5d-h make the real gate, independent-source check, two-stage acquisition, physical preflight and primary preregistration bounded and content-bound, but **T4C.6 has not run**. **T5.0 is partial:** T5.0a-b supplies the strict versioned/hashable protocol and runtime-binding gate, but the actual laboratory evidence/config has not been supplied or frozen. **T5.1 is partial:** T5.1a-e accepts raw/FFT/DCT/Haar/db2/SWT/DTCWT training representations, optimized and exposed with truthful UI readiness; mixed precision and remaining cross-device acceptance remain. **T5.2 is partial:** T5.2a-d implements the aligned, leakage-safe, train-normalised PyTorch dataset, bounded-memory worker-safe cache, bounded blockwise CDS cache publication, exact calendar splits and verified physical-time leads; the live CDS run, independent overlap and viable multi-year NZ crop remain blocked by D43. **T5.3 is partial:** T5.3a-b supplies the forecaster seam, persistence baseline, deterministic represented smoke run, verified model-artifact contract and persistence-relative evaluator; T5.3c and the actual laboratory model remain open. **T5.6 is partial:** T5.6a-g supplies the offline FCN3 identity, lazy canonical cube/import boundary, exact ERA5 truth bridge, matched ensemble metric engine, atomic reproducible run receipt, portable laptop/HPC runner and verified API/UI reporting; no worker, dependency, checkpoint, initial condition, real forecast or real truth evaluation has run, so the result UI is honestly empty. **Not done:** T4C.6, the real-ERA5 gate review; 4D-4H; T5.4-5, the remaining T5.6 inference/real-data work, T5.7, and the T5.8 irregular-observation/spatial-downscaling track. Per-task evidence blocks sit under each task below; a task without a **DONE** or **PARTIAL** label has not been started. |
| Ownership / licence | **Declared in `LICENSE.md`.** Edward Jonathan Bentley retains the proprietary SpectralEarth core. Adam Frank Bentley has a named perpetual, worldwide, royalty-free grant for lawful personal, academic, research and commercial use/modification, without public redistribution or sublicensing of the core. Independent extensions and upstream contributions remain separately governed. This bespoke text has not been professionally reviewed. |
| **Accessibility** | **Zero, measured.** `0` `aria-*` or `role` attributes and `0` keyboard handlers across `frontend/src`. No focus management. The UI is usable with a mouse and by nobody else. Not scheduled; recorded so it cannot be mistaken for an oversight. |
| Backend test suite | **1787 passed, 1 xfailed.** Plus one explicit skip: the opt-in live-GCS check. Trajectory: 19 written / 1 failing / uncollectable -> 65 -> 152 -> 222 -> 271 -> 351 -> 407 -> 449 -> 478 -> 535 -> 548 -> 593 -> 642 -> 647 -> 709 -> 781 -> 855 -> 859 -> 882 -> 883 -> 890 -> 911 -> 917 -> 933 -> 946 -> 955 -> 957 -> 962 -> 969 -> 981 -> 985 -> 1000 -> 1008 -> 1027 -> 1042 -> 1056 -> 1070 -> 1080 -> 1089 -> 1094 -> 1095 -> 1102 -> 1104 -> 1106 -> 1112 -> 1116 -> 1117 (the `master` freeze) -> 1375 (TG2.1, `ed-dev`) -> 1429 (TG2.2, `ed-dev`) -> 1477 (TG2.3, `ed-dev`) -> 1536 (TG2.4, `ed-dev`) -> 1577 (TG3.1, `ed-dev`) -> 1621 (TG3.2, `ed-dev`) -> 1686 (TG3.3, `ed-dev`) -> 1742 (TG3.4, `ed-dev`) -> 1787 (TG3.5, `ed-dev`). |
| Ground-Truth Benchmark Suite | **22 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE.** Thirteen datasets with declared known answers, seven of them nulls. CI-ready via `python -m src.benchmarks` (exit 0). |
| Backend compute modules | **Written, executed and tested.** `physical_core` carries `GridSpec` + metric-aware operators; `analysis_engine` gained `spectra.py` and `climatology.py`; `transform_engine` gained the undecimated `stationary.py` and a real `dtcwt.py`; `statistics/` and `core/` are new packages. |
| Physical units and wavenumbers | **Correct as of T3.5.13.** Gradients metric-aware, spectra on a physical `k` axis, domain statistics area-weighted, and every quantity carries its units. Previously all of it was pixel-space and unlabelled (D13). |
| Turbulence regime classification | **Corrected as of T3.5.13.** Was off by one exponent for the platform's entire history and labelled Kolmogorov fields as Charney (D26). |
| Shift invariance | **Available as of T3.5.7** via the undecimated SWT: 0.00% energy spread against the decimated DWT's 153.50%. |
| DTCWT | **Implemented as advertised as of T3.5.6** (D1 closed). Real Kingsbury q-shift dual tree, six oriented complex subbands with *measured* passband centres, vendored full-precision coefficients, cross-checked against two independent oracles. Near shift invariant, not exact — the SWT remains the exactly shift-invariant transform. |
| Reproducibility | **Seeded generation, perturbation and run-level seed capture** (T3.5.12/T3.5.19). `ExperimentRun.seed` and `execution` are persisted, and a sweep is byte-identical across the `serial` and `process` backends. **Byte-identical across `thread` as well, since D55 was closed properly on `ed-dev`:** `_run_one` used to seed the process-global torch/numpy generators, which threads share, so a real payload's seed-to-draw window was corrupted by the next worker — measured at 10 of 10 trials once that window was held open by 10 ms of work. Streams are now per task (`core/randomness.py`), and the acceptance sweep carries a random draw, which it previously did not — the criterion had been certified by a fully deterministic payload. Replaying a run from lineage alone is still outstanding. |
| Statistical validity | **Controlled as of T4C.5, and calibrated as of T4C.3** (D8 closed). Bonferroni / Holm / BH / BY with the dependence assumption reported alongside every q-value, five surrogate null models, an ESS correction, a calibrated stationarity gate and an explicit power check. The D8 scenario went from 3 reported "discoveries" on 9-sample noise to 0, while a real effect among 19 nulls at n=40 is still recovered. T4C.3 added the two calibrations that decide whether a lagged test means anything at all: a linear lag against a circularly stationary null falsely rejects **20 of 20** AR(1) records where the null is true, and a shift null that keeps the simultaneous alignment caps the achievable p-value at about 0.005 regardless of ensemble size. |
| Registries / extension seams | **Registry-based as of T3.5.15** (D15 closed). Transforms, pipeline actions and data sources are decorator-registered with capability metadata; the plugin acceptance test registers a third-party transform without editing `src/`. |
| Error reporting | **Taxonomy in place as of T3.5.14** (D14 closed). `SpectralEarthError` subclasses carry their own status code and client-safety, so an HTTP status follows from the error *kind* rather than from the call site. |
| HPC / executor seam | **In place as of T3.5.19.** Serial / thread / process backends behind one interface, submission-order results, thread-budget control and SQLite WAL + `busy_timeout`. Measured honestly: on this workload **serial beat thread(4) and process(4)**, because PyTorch already parallelises the FFT across cores. T5.1a-e CPU/RTX-CUDA parity is verified; whole-platform and ROCm/MPS agreement are not (D18 remains partial). |
| Schema migrations | **In place as of T3.5.8** (D32 closed). Two Alembic revisions, `ensure_schema` at startup, drift against the ORM checked by test. `create_all` had silently left the repository's own database unqueryable. PostgreSQL is verified only as *rendered* DDL, not executed. |
| FastAPI surface | **31 endpoints**, executed and smoke-tested. CORS, health and collection endpoints all added (T3.5.2, T3.5.10); T5.6g adds verified receipt import/list/get. Health reports device, executor, SQLite pragmas and schema revision. |
| React frontend | **Ten modules, wired to the backend, and no longer able to fabricate a result** (T3.5.22/T3.5.23/T5.6g). Export in CSV/JSON/NetCDF4/Zarr/PNG/SVG with provenance embedded in the file; units, spectral convention and slope uncertainty displayed; simulated data labelled where it is used. Health/device/executor/schema, the benchmark suite, the data-source chain with its simulated flags, the ERA5 crop inspector, T5.1 representation readiness, T5.2 manifest readiness, per-hypothesis statistics and receipt-backed forecast evaluation are reachable now; `tsc` is clean and the current build emits 1,386 modules. A contract test asserts every fetched path is served and every field the UI reads exists. **The then-nine modules were rendered in a browser and confirmed working by the user on 2026-08-20** (T3.5.25). No screenshots were captured, so that confirmation is a **user report rather than an artefact in the repository**; T3.5.0 asked for a screenshot per tab and that literal evidence is still absent. The tenth Forecast Evaluation module has compiled and built but has not been visually inspected. |
| Real ERA5 data | **Reading the live archive as of T3.5.18.** Regional crops stream from public WeatherBench 2 Zarr on GCS into a rechunked local cache: 257x257 x 4 levels x 8 days materialised in 186 s (951 MB wire, 19 MB cached, 51.1x chunk amplification measured against 51.1x predicted). Network access is opt-in; a cached crop works offline. **A one-year crop at the R13 floor is 79 GB and ~2 h - measured, not achievable at laptop tier, and stated as such.** |
| Vectorisation | **Complete for D17.** Radial PSD, coherence, both boundary-distance profiles and Tukey construction are single-pass/vectorised. On a 512x512 field with 256 rings, `decompose_by_boundary` fell from 148.8 ms mean to 7.70 ms mean (**19.3x**) while matching an independent oracle. Long-record signature and climatology extraction now stream one frame at a time. |

**The original baseline audit** (2026-08-19, before any of Phase 3.5) is preserved in
`architecture.md` Section 7 and in the early sections of `VERIFICATION.md`. It recorded that
nothing had ever been executed, that the test suite could not be collected, and that the
"validated / zero-error" claims in earlier revisions were aspirational. That snapshot is
history, not current status, and this table replaces it.

Phases 1 and 2 are done as *code* and now substantially done as *verified software*. Phase 3
is partial. Phase 3.5's 25 implementation tasks are complete; the literal screenshot evidence
requested by T3.5.0 is still absent, and D18's cross-device agreement remains partial because
only the T5.1a-e slice has CPU/CUDA parity evidence and ROCm/MPS are unmeasured. Phase 4A-4C.5
are complete; the real-ERA5 T4C.6 gate review and all
of 4D-4H remain undone. See Section 4 for per-task evidence.

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
| PHASE 5: Training-Native Regional    |  The existing laboratory model is
|          Forecast Research           |  the first downstream judge; global
|                                      |  models are later adapters.
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

4H is optional and never gates anything. Phase 5A (T5.0--T5.4) is an intentional fast track:
it needs the verified transform core and a real-data resolution for D43, but does not wait for
4D--4G. T5.5's comparison with `RepresentationScore` does need 4G. Phase 6 needs nothing, but
makes 4B-4F and broad Phase 5 sweeps affordable at scale.

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

1.  **Scale-dependent edge exclusion.** Feature detection, constellation extraction and every statistic operate only on the **valid interior**: `interior = N - 2 * halfwidth(j)`. For a recursively undecimated transform, the complete cascade support is `1 + (L-1)(2^j-1)`; counting only the current level's dilated filter discards the inherited low-pass support (D44). The mask is per-scale - coarse scales exclude far more than fine ones. The platform **computes and reports** the valid interior per scale rather than assuming it, since the exact figure depends on the chosen filter.

2.  **Crops must be sized from the coarsest scale, and the numbers are larger than intuition suggests.** Valid interior width for a 14-tap filter:

    | Level (scale) | Excluded per side | N=64 | N=128 | N=256 | N=512 |
    |---|---|---|---|---|---|
    | 1 (2) | 7 px | 50 | 114 | 242 | 498 |
    | 2 (4) | 20 px | 24 | 88 | 216 | 472 |
    | 3 (8) | 46 px | **none** | 36 | 164 | 420 |
    | 4 (16) | 98 px | **none** | **none** | 60 | 316 |
    | 5 (32) | 202 px | **none** | **none** | **none** | 108 |

    **A 64x64 crop has zero valid interior already at scale 8.** Cross-scale analysis - the entire premise of Phase 4C - is therefore *impossible* on a 64x64 region. With the declared 128-pixel minimum valid interior, the practical floor is **512x512 for four dyadic levels and 1024x1024 for five**. The `laptop` tier is constrained on frames, bank breadth and surrogate count, **never** by shrinking the grid below its valid-interior floor.

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
| `laptop` | 256x256 | 3 | 200 | 3 families | 50 | minutes |
| `workstation` | 512x512 | 4 | 1000 | full bank | 200 | hours |
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
| **Training representation** | `RepresentationModule.forward/inverse` | T5.1 | regional or global PyTorch forecasters | partial: raw/FFT/DCT/Haar/db2/SWT/DTCWT accepted; mixed precision and non-NVIDIA evidence remain |
| **Forecast dataset** | `RegionalForecastDataset` | T5.2 | Phase 5 training/evaluation | partial; aligned 850-hPa t/q/u/v/z histories/targets, pre-sample embargo, train-only normalisation and provenance implemented; real-source acceptance/D43 open |
| **Point observations** | planned `ObservationSet` | T5.8a | observation-conditioned models and station verification | not started; current data contracts accept gridded fields, not irregular station observations |
| **Conditional field model** | planned `ConditionalFieldModel` | T5.8b-e | spatial downscaling and observation enhancement | not started; no DeepSensor or other off-grid model adapter is currently implemented |
| **Feature detector** | `@register_detector` | 4D | 4D/4E | alternative detection strategies |
| **Surrogate generator** | `@register_surrogate` | 4C | all mining | phase-randomised, AAFT, IAAFT |
| **Scorer term** | `@register_scorer` | 4G | representation scoring | add a score term without touching the scorer |
| **Forecaster** | `Forecaster.predict` | 4G | 4G baselines, Phase 5 | persistence / advection / FourCastNet behind one interface |
| **Executor** | `Executor.submit/map/gather` | T3.5.19 | everything parallel | serial / thread / process / celery / slurm |
| **Execution profile** | `resolve_profile` + `python -m src.core.doctor` | T5.1 portability slice | app, experiments, local/HPC preflight | implemented: auto/cpu/accelerator plus allocation-guarded hpc; no remote submission |
| **Artifact store** | `ArtifactStore.put/get` | T4A.3 | everything large | local disk now, object store later |
| **Database** | SQLAlchemy session | exists | everything | genuinely already decoupled |

The data spine running through all of it: `PhysicalField` -> `FieldSequence` (4A) -> `CoefficientField` (4B) -> `ScaleSignature` (4C) / `SpectralFeature` (4D) -> `FeatureConstellation` (4E) -> `TransitionPattern` (4F) -> `RepresentationScore` (4G). Every seam above either produces or consumes one of these types, which is what keeps the engine coherent rather than a bag of scripts.

---

## 4. Phase 3.5 - Green Baseline & Real Multiscale Foundations

**Objective:** an installed, running, test-green platform whose multiscale machinery actually has the properties Phase 4 depends on. No new science.

### T3.5.0 Establish a verified baseline *(blocks all)*
Create a venv, `pip install -r requirements.txt`, `npm install` in `frontend/`, run the full test suite, run both servers, load the UI, exercise all seven tabs.
**Acceptance:** a `VERIFICATION.md` recording actual command output - test pass/fail counts, `npm run build` output with emitted asset names and sizes, and a screenshot per tab. No claim of "validated" appears anywhere in the repo without corresponding output in this file.

**Met except for the screenshots.** `VERIFICATION.md` records captured output for every slice; `npm run build` output with asset names and sizes is recorded; both servers were started and the nine tabs confirmed working by the user on 2026-08-20 (T3.5.25). **No screenshot per tab has been captured**, so that one clause is outstanding - the rendering is attested, not evidenced.

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

### T3.5.8 Alembic migrations - **DONE**
Initialise Alembic, autogenerate the baseline migration for the five existing tables, stop relying on `create_all` in the lifespan.
**Acceptance:** `alembic upgrade head` builds the schema from empty on both SQLite and PostgreSQL.

**Met, with one criterion met only partially and said so.** Two revisions: `0001` the
pre-migration five tables, `0002` the five columns added by T3.5.12 and T4C.5. On SQLite,
`upgrade head` from empty builds the schema and `compare_metadata` against the ORM reports
**zero differences** with type and server-default comparison enabled; each revision
round-trips up and down individually; `downgrade base` leaves no application table. On
**PostgreSQL the migrations are verified only as rendered DDL** in Alembic's offline mode -
they compile for that dialect, but no server was available to execute them, so half of this
acceptance criterion is asserted and half is demonstrated. That distinction is recorded
rather than smoothed over.

**Why it stopped being bookkeeping.** `create_all` adds missing tables and silently ignores
missing columns. The repository's own `spectral_earth.db` predated both column-adding slices,
so it raised `no such column: experiment_runs.seed` on the first query while startup reported
success - defect **D32**. Reproduced on that file, then fixed: `ensure_schema` detected it as
revision `0001`, applied `0002`, and left zero drift against the ORM.

**The trap avoided.** Adopting a pre-Alembic database by `stamp head` - the conventional move -
would have asserted the five columns exist, skipped `0002` permanently, and produced a
database Alembic believed was current and the ORM could not query. `detect_legacy_revision`
probes for `experiment_runs.seed` and stamps **by observation**. 25 tests
(`src/tests/test_migrations.py`), including one that asserts the `create_all` defect itself so
the reason for this layer cannot be lost.

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

### T3.5.14 Explanatory error taxonomy *(D14, implements E6)* - **DONE**
Structured exception hierarchy with step/action/shape/expectation context; API returns actionable detail (4xx for user error, 5xx only for genuine internal faults) while still not leaking stack traces to the client. Pipeline failures record the failing step and parameter combination.
**Acceptance:** a deliberately malformed pipeline (shape mismatch between two steps) produces a message naming both steps, both shapes, and the fix. Sweep failure reporting is tested with a partially-failing 20-run sweep.

**Met for the taxonomy and the API surface.** `core/errors.py` defines the hierarchy;
`ShapeMismatchError` names both sides and the fix, `PipelineStepError` names the step, action,
parameter combination and run index and **inherits the cause's status code** (a bad parameter
inside a step is still a 4xx, not a platform fault), and `ReferenceResolutionError` lists the
resolvable references with a `did you mean`. The API classifies by error *kind*: an unknown
transform is now a 404 quoting the valid names, while a genuine fault stays opaque - both
asserted, including that a 500 does not leak a file path.

**Outstanding:** wiring `PipelineStepError` into the sweep loop so a partially-failing 20-run
sweep reports per-run causes. That lands with T3.5.19, which is where run-level failure
handling belongs.

### T3.5.15 Registries for sources, actions, transforms and detectors *(D15, implements E1, E2)* - **DONE**
Decorator-based registries; refactor `_execute_action`'s 13 branches and the adapter's hard-coded dataset list onto them. Define the `DataSource` protocol with capability declaration and an explicit fallback chain, recording the resolved source in lineage.
**Acceptance:** a new data source and a new pipeline action are each added in a **new file only**, with zero edits to `engine.py`, `adapters.py` or `main.py`, and both appear automatically in `GET /api/v1/actions` and `GET /api/v1/data/datasets`. A run served by the simulated fallback is visibly labelled as such in the API response and the UI.

**Met, and verified literally.** `src/tests/plugin_example.py` adds a data source and a
pipeline action in one new file; `test_acceptance_new_plugin_needs_no_core_edits` imports it,
asserts both appear in `GET /api/v1/actions` and `GET /api/v1/data/datasets`, and **hashes
the three core files before and after** - a claim about not editing files is checkable, so it
is checked rather than asserted in prose.

Delivered as `core/registry.py` + `core/errors.py`, with `transform_engine/registry.py`
(6 transforms, one dispatch point replacing two parallel chains),
`experiment_engine/actions.py` (7 actions; **engine.py 562 -> 315 lines**) and
`data_layer/sources.py` + `builtin_sources.py` (priority-ordered fallback chain). Three new
discovery endpoints. Suite 351 -> 375.

**The acceptance test found a real latent bug the moment a third source existed:**
`list_datasets` inferred `is_simulated` from `kind == "simulated"`, a literal string test, so
any simulated source with a different `kind` label would have been reported to the researcher
as **real observational data**. It now reads the flag the source declared.

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

### T3.5.18 Cloud-native ERA5 via Zarr, with a rechunked local cache *(implements E2; unblocks all of Phase 4)* - **DONE**
The raw material is not a constraint: **ERA5** (ECMWF/Copernicus reanalysis, hourly, 1940-present, ~31 km, up to 137 levels) is packaged analysis-ready by **WeatherBench 2** as public Zarr on GCS, 1959-2023, at resolutions up to the full 0.25 degree / 1440x721 grid. Zarr is the right adapter, not bulk NetCDF download: `xarray.open_zarr` over `fsspec`/`gcsfs` opens the whole archive lazily and fetches only the chunks a crop touches, so "crop aggressively" stops being a separate step and becomes simply how the store works.

**The trap that decides whether this is fast or unusable: chunk alignment.** SpectralEarth's access pattern is *many timesteps over a small spatial region* - the exact opposite of a store chunked as one full global field per timestep. Against such a layout, every frame of a 64x64 crop drags a whole planet across the network. Reading a decade for one region could move terabytes to analyse megabytes.

**Therefore the adapter has two stages, and the second is not optional:**
1.  **Query** the remote Zarr lazily for the requested region, time window, levels and variables.
2.  **Materialise once into a local, analysis-optimised Zarr cache, rechunked time-contiguous for that region**, content-hashed and registered in the `ArtifactStore`. All Phase 4 work reads the local cache.

This is also excellent provenance (E5): a crop is *exactly* specifiable as `{store URI, dataset version, variable, time range, bbox, levels}` plus a content hash of the materialised snapshot - far stronger than "someone put a `.nc` in a folder", and it makes any finding reproducible by anyone with an internet connection.

**Acceptance:** the adapter reports the remote store's chunk structure and **warns when the requested access pattern is chunk-hostile**; a 512x512 region (the corrected R13 floor for four levels) over one year is costed against the `laptop` tier budget with recorded bytes-transferred; the adapter refuses a crop too small for the requested number of levels, naming the minimum; re-requesting an identical crop hits the cache with zero network traffic; the crop specification round-trips from the lineage record to an identical re-materialisation.

**Met, with one criterion measured and found unachievable — reported rather than reworded.**

Verified against the **live** WeatherBench 2 archive (24 ERA5 stores enumerated anonymously at
`gs://weatherbench2/datasets/era5`), not against a mock.

*   **Chunk structure reported, hostility warned.** The 0.25 degree stores are chunked
    `(1, 13, 721, 1440)` — one timestep, all levels, whole planet, **54.0 MB/chunk**.
    `assess_access_pattern` derives an amplification of **51.1x** for a 257x257 four-level crop
    from chunk metadata alone, transferring nothing, and the live measurement came in at 51.1x.
    It also reports the non-obvious part: selecting 4 of 13 levels saves nothing over the
    network, because level is inside the chunk.
*   **Recorded bytes transferred.** 257x257, 4 levels, 8 days at 6 h: predicted 1,727.6 MB
    uncompressed, **measured 951.3 MB wire in 186.3 s**, cached to 19.0 MB.
*   **The R13 floor is refused, not warned about.** After D44, `edge_exclusion` accumulates the
    complete inherited cascade (7/20/46/98 px at levels 1–4 for 14 taps) and
    `minimum_crop_size` returns 512 and 1024 for four and five levels.
    A 64x64 crop at four levels raises, naming the minimum, the contaminated width and which
    dimension to constrain instead.
*   **A repeat request transfers zero bytes**, asserted as `== 0` rather than as "fast": the
    same crop returned `cache_hit` in 0.006 s, and the cached read is 8 chunk reads in 0.05 s
    against 186 s remote.
*   **The crop specification round-trips.** `rematerialise_from_provenance` rebuilt the real
    crop from its lineage record with identical content key *and* content hash. The hash is
    also independent of cache chunking, verified by materialising at two chunk sizes.

**Not met: "one year within the `laptop` tier budget".** Measured, the arithmetic forbids it —
1,464 frames x 54.0 MB is **79.0 GB** to deliver 1.55 GB, about **2 hours** at the 11.6 MB/s this
connection sustains even with 16 concurrent chunk fetches. This is a property of WeatherBench 2's
chunk-1 layout and of consumer bandwidth, not of the adapter. R13 already says which dimension
to give up: *"the `laptop` tier is constrained on frames, bank breadth and surrogate count,
**never** by shrinking the grid below its valid-interior floor."* So the honest laptop-tier ERA5
crop at 0.25 degree is 256x256 x 3 levels x days, or 512x512 x 4 levels x fewer days, not x a
year. The historical 257x257 transfer remains valid transport evidence, but D44 means it was
not valid evidence of a four-level scientific floor. The adapter now says so before download.

**Two defects found while doing it.**

*   **D33** — `xarray` was declared in `requirements.txt` with **no NetCDF engine**, so the
    documented "drop an ERA5 `.nc` into `data/`" path could not open a modern NetCDF4/HDF5 file
    at all. Reproduced, then fixed and covered by a write-and-reopen test.
*   **An unrechunked cache, caught by its own test.** `to_zarr` prefers each variable's
    inherited `encoding["chunks"]` — copied from the *remote* store — so `.chunk()` set the dask
    graph and the write ignored it. The cache came out with one timestep per chunk: the exact
    layout it exists to escape. Nothing errored, and the manifest still recorded the *requested*
    chunking. Measured on the real crop after the fix: cache reads fell from **39 chunks / 1.27 s
    to 8 chunks / 0.05 s**, a 25x improvement in the operation the whole second stage exists for.

58 tests in `src/tests/test_zarr_source.py`, all offline against synthetic Zarr stores built with
the real archive's pathological layout, plus one opt-in live check that asserts the documented
chunk shape still holds — so if WeatherBench 2 rechunks, the test says so rather than the
documentation quietly becoming false.

### T3.5.19 Executor seam and concurrency-safe persistence *(implements E11)* - **DONE**
Define the `Executor` protocol with `serial`/`thread`/`process` backends; route the sweep loop and surrogate ensemble generation through it. Enable SQLite WAL mode and a busy timeout; give each worker its own session; move seed derivation to `SeedSequence.spawn`.
**Acceptance:** a 20-run sweep produces byte-identical results under `serial` and under `process` with 2, 4 and 8 workers; a concurrency stress test drives parallel writes without a single `database is locked`; measured speed-up recorded per tier in `VERIFICATION.md`, including the deliberately-misconfigured oversubscribed case to demonstrate the thread-budget knob matters.

**Met.** `core/executor.py` (serial/thread/process, submission-order results,
`SeedSequence.spawn` seeding, per-worker thread budget) and `core/device.py`
(CUDA/ROCm -> MPS -> CPU with an explicit override). The sweep is split into
`execute_run_payload` - which computes a run with **no database access**, so it can cross a
process boundary - and serial persistence in the parent. 28 tests in `test_executor.py`;
suite 379 -> 407.

*   **Byte-identical** across `serial`, `thread(2)`, `thread(4)`, `process(2)`, `process(4)`
    and `process(8)`, compared on the persisted parameters, status, seed and results of every
    run rather than on a summary.
*   **Concurrency stress:** 8 threads x 25 commits = 200 rows with zero `database is locked`,
    with SQLite in WAL mode and a 30 s `busy_timeout` applied per connection.
*   **Speed-up measured, and the honest answer is that parallelism does not help this
    workload.** PyTorch already parallelises FFT/BLAS across cores: at 1 intra-op thread
    `thread(4)` gives **1.40x**, but at 12 threads it gives **0.56x** - slower than serial.
    The `process` backend costs **~6.3 s** of fixed startup on this platform. `serial` is
    therefore the measured default, and `GET /api/v1/health` reports why.

**Also completes D12's remaining half:** every `ExperimentRun` now records its derived `seed`
and an `execution` provenance block (backend, workers, thread budget, device, torch version),
so a run can be re-executed to the same numbers. And **T3.5.14's sweep-level acceptance**: a
partially-failing sweep completes its good runs and names the failing step - a test asserts 3
of 4 runs COMPLETED with the fourth naming its bad parameter.

### T3.5.20 Vectorise the radial and distance binning *(D17, implements E10)* - **DONE**
Replace all five per-bin loops with `torch.bincount` / `scatter_add` single-pass reductions (`compute_radial_psd`, `compute_spectral_coherence`, `decompose_by_boundary`, `analyze_boundary_artefacts`, and the Tukey window construction). Batch the transform over `(time, scale, orientation)` as tensor dimensions rather than Python iteration.
**Acceptance:** bit-comparable results to the current implementation (within float tolerance) on the benchmark suite, plus a recorded speed-up on a 512x512 field. This is a prerequisite for the `laptop` tier being honest rather than aspirational.

**Met.** The radial PSD/coherence and Tukey paths were already vectorised. The two remaining
distance profiles now label every pixel once, use `bincount` for counts/sums and grouped
`scatter_reduce(amax)` for maxima, and only loop over the small result vector for JSON
serialization. Independent nested-loop oracles preserve the integer interior rings and the
boundary lab's Euclidean padded-corner rings exactly. On float64 512x512 input with 256 rings,
the error-decomposition path measured 148.8 ms mean before and 7.70 ms after (**19.3x**). The
earlier proposal to batch an entire long `(time,scale,orientation)` cube is superseded by the
bounded T4C.5d stream: batching thousands of 512x512 coefficient frames would violate E9/E10;
one-frame transform residency is the laptop-safe seam.

### T3.5.21 Device and thread policy *(D18, implements E9)* - **PARTIAL** (CPU and T5.1a CUDA verified; whole-platform/ROCm/MPS agreement remains)
Extend `get_execution_device` to CUDA/ROCm -> MPS -> CPU with an explicit override; configure CPU thread counts; make every kernel device-agnostic (the current code mixes CPU-constructed tensors with a selected device in places).
**Acceptance:** the `smoke` tier passes identically on CPU, CUDA and MPS, with results agreeing within float tolerance across all three.

**Partially met, and the gap is stated rather than papered over.** `core/device.py` implements
the CUDA/ROCm -> MPS -> CPU chain with a `SPECTRAL_DEVICE` override, configures CPU/BLAS thread
counts, and **refuses** a requested device that is not present instead of silently falling
back to CPU - a run that claims a GPU must have used one. `to_device` centralises moving
nested structures, addressing the mixed CPU/device tensor construction D18 flagged.

The venv now uses `torch 2.13.0+cu130`, and T5.1a-e raw/FFT/DCT/Haar/db2/SWT/DTCWT reconstruction, coefficients and
backward gradients agree between CPU and the RTX 5050 within declared tolerance. PyTorch ROCm
uses the same `cuda` device API; `device.available_devices` records the compiled runtime as
`rocm` or `cuda` so provenance remains vendor-correct. The full smoke tier, AMD ROCm hardware
and Apple MPS remain unmeasured, so D18 is still partial.

### T3.5.22 Surface the backend in the UI *(closes the gap between capability and reach)* - **DONE**
Wire the endpoints that had no consumer into the React workbench: health (device, executor,
SQLite pragmas, schema revision), the Ground-Truth Benchmark Suite, the data-source fallback
chain with its simulated/observational flags, the ERA5 Zarr crop tools, and the statistical
fields on every mined hypothesis. Add a mechanical frontend/backend contract check.
**Acceptance:** every service method is called from the UI; every path the frontend fetches is a
route the API serves; every nested key the UI reads out of an untyped payload exists in the real
response; `npm run build` passes.

**Met.** Two new modules in the workbench — **8. Platform & Evidence** and **9. Real ERA5
(Zarr)** — plus a statistics block on every hypothesis card. `tsc` clean, `vite build` emits 1,378
modules. 13 tests in `src/tests/test_frontend_contract.py`.

**Why a contract test and not just a build.** `npm run build` runs `tsc`, so the frontend's
*internal* types are checked. Nothing checked them against the **backend**: the payloads are typed
by hand, and the ones that matter are `Record<string, any>` because their shape is nested. A
renamed route or a restructured payload compiles cleanly and fails in the browser. So the tests
parse `api.ts` for the paths it fetches and assert each is served (distinguishing a path parameter
from a query string), assert every service method is actually called from `App.tsx`, and assert
each nested key the UI reads is present in a real response from the running app.

**It found a runtime bug immediately.** The hypothesis card was written to render
`statistics.correction` as a string. It is an **object** — `{method, assumption, n_tests,
min_adjusted}` — so React would have thrown *"Objects are not valid as a React child"* on the first
mined hypothesis. `tsc` passed, `vite build` passed, and 547 backend tests passed. Only a check of
the payload's actual shape catches that class of defect.

**D8 as a presentation problem, closed.** The card previously showed `Confidence: 96.0%` and
nothing else — the presentation half of the defect that made nine noise correlations from a 9-run
sweep read as nine discoveries. It now labels that number **effect size**, shows the q-value, the
raw p-value, the family size, the correction procedure **with its dependence assumption**, and the
R7 non-causality caveat. A finding with no correction is shown with an explicit warning rather
than silently, because a card that looks the same either way is what made the original defect
invisible.

**Since met, in part, and the part matters.** At the time of writing this task the UI had never
been seen: no browser was available here, so the nine tabs were verified to compile, to call
routes that exist and to read fields that are present — and **not** to render. On 2026-08-20 the
platform was started (T3.5.25) and the user confirmed the nine tabs render and work. What is
still absent is the *captured* evidence T3.5.0 asked for: no screenshot per tab exists in this
repository, so the rendering claim rests on a user's report rather than on an artefact anyone can
re-examine. `test_browser_rendering_evidence_is_described_accurately` keeps that distinction in
this file, because "someone said it looked fine" and "here is the image" are not the same claim.

### T3.5.23 Scientific integrity and export in the UI - **DONE**
Delete every browser-side fabrication path; label simulated data where it is used; display the
units, the spectral convention and the slope uncertainty the backend already returns; export
fields and tables in the formats a researcher opens.
**Acceptance:** no code path computes a result without the backend; `is_simulated` is visible on
the tab that uses the data; CSV/JSON/NetCDF4/Zarr/PNG/SVG all round-trip with provenance embedded
in the file.

**Met.** 55 new tests (32 export round trips, 23 contract and integrity guards); suite 548 -> 593.

*   **~14 fabrication paths deleted.** An unreachable backend used to make the UI invent fields,
    transforms, diagnostics, boundary analyses and experiment IDs. A header badge said "Offline
    Sandbox Mock Mode"; individual results said nothing, so a spectral slope from `Math.random()`
    looked exactly like one from ERA5. A test now asserts no `if (backendConnected)` branch
    survives - with no backend there is no data, and the UI says so.
*   **Export, which did not exist at all.** `grep` for `download|export|Blob|csv` across the
    frontend previously returned nothing. Four data formats now round-trip through the library a
    researcher would use, asserted by reopening each file; PNG/SVG render client-side because a
    server re-render would be a different picture from the one on screen.
*   **Provenance inside the file**, flattened with dotted keys for NetCDF attributes rather than
    dropped, with the simulated and unreproducible warnings *derived* so no export path can omit
    them.
*   **D34, found while doing it:** the perturbation endpoint never passed a seed, so every
    perturbation over HTTP was irreproducible; the frontend then built its synthetic forecast
    with unseeded `Math.random()`, uniform despite the control being labelled StDev.
*   **Two unconditional validation badges removed** from the transform tab - green ticks shown
    whatever the measured error was - replaced by the measurement judged against a stated 1e-9
    tolerance.

**A self-inflicted defect worth recording.** The LaTeX cleanup replaced `$...$` literals across
App.tsx with a blanket substitution, and `${k}=${v}` in two template literals matched the
pattern. `tsc` caught it as an unused destructure; nothing else would have. Blanket text
substitution over source is a refactor, not a formatting fix, and should be treated as one.

**Still not met:** browser rendering. No browser is available here, so T3.5.0's screenshot-per-tab
criterion remains open, and the export controls have never been *clicked* - only proved to compile,
to call routes that exist, and to produce files that reopen correctly on the server side.
### T3.5.24 Import, evidence on demand, and capability discovery - **DONE**
Read user-supplied fields back in; make the benchmark suite runnable from the UI; surface the
registries. Close the last four served-but-unreachable endpoints.
**Acceptance:** every format the platform exports imports back with values, coords, units and
provenance intact; a multidimensional file refuses to guess which slice; no served route is
unreachable from the UI.

**Met.** 40 new tests (35 import, 5 UI contract); suite 593 -> 642. `tsc` clean, build passes.

*   **Import in four formats**, every test round-tripping through `exporters` rather than a
    hand-written fixture - a fixture can encode the same misunderstanding twice.
*   **A 4D file refuses to guess.** An ERA5 `(time, level, lat, lon)` file has no single field
    in it; taking `[0, 0]` silently would import a slice the researcher did not choose while
    every statistic described that arbitrary timestep. `inspect` reports the axes, `read_field`
    refuses until each is pinned, and the indices go into the provenance.
*   **Axes identified by name before position.** `(lat, lon, time)` read positionally comes back
    **transposed**, and a transposed field still looks like a field - every orientation and
    anisotropy statistic derived from it would be wrong undetectably.
*   **A round trip cannot launder simulated data**, asserted for all four formats. Unknown-origin
    files report `is_simulated: null` rather than `false`, and the UI renders that third state as
    "origin unknown" rather than as observational.
*   **Zip path traversal refused.** `extractall` follows `..`; this is the one place the platform
    accepts arbitrary bytes from outside itself.
*   **The benchmark suite is runnable from the UI** at a chosen root seed, with the three
    outcomes kept separate on screen and a distinct alarm when a *null* benchmark reports a
    discovery. Results export as a table.
*   **The registries are browsable** with their capability flags, generated rather than listed.
*   **A test now asserts no served route is unreachable from the UI**, and any exemption must
    name its reason inside the test.

**Still not met:** browser rendering (T3.5.0), accessibility, and the decomposition of
`App.tsx`. The import panel, the benchmark runner and the registry tables are proved to compile,
to call routes that exist and to exchange payloads whose every field is present - and have never
been seen on screen.

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

**T4A.1 `FieldSequence` (`src/physical_core/sequence.py`)** - **DONE**
Ordered `PhysicalField`s plus a time coordinate; validates shape and coordinate consistency across frames. Methods: `.at(t)`, `.map(fn)`, `.to_tensor()` -> `(T, H, W)`.

**Met.** Grid equality checked on the **metric, not object identity** - two frames cropped from
the same archive are distinct `GridSpec` instances, and refusing those would make the class
unusable on real data, while accepting genuinely different grids would let a mean average
different places together and produce numbers rather than an error. Unsorted times are refused
rather than sorted: a sequence silently reordered is worse than one that errors. Irregular
cadence is *reported*, and `lag_to_seconds` refuses on it, because N frames is not a fixed
duration across a gap.

**T4A.2 `split_temporal(train, val, embargo)` *(implements R6)*** - **DONE**
Mirrors the existing spatial guardrail API deliberately: `validate_temporal_guardrails(other)` raises on any time overlap **or** on an embargo gap shorter than the longest lag under test.
**Acceptance:** a deliberate leakage attempt (overlapping windows, or embargo < max lag) raises `ValueError`, with a test asserting the raise.

**Met, both halves.** Overlap raises; and the subtler one - windows that are *disjoint* so an
overlap check passes, while a training example near the boundary has its target inside the test
window - raises too, with a message naming the gap in frames and the lag it fails against. A gap
of **exactly** one lag also raises: it puts the last training target on the first test frame. The
embargo frames are returned rather than dropped, so a lineage record shows what was excluded.

**T4A.3 `ArtifactStore` (`src/artifact_store/store.py`)** - **DONE**
Content-addressed on-disk store (`.npz`/`.pt` under `artifacts/`, SHA-256 keyed). Steps exchange `ArtifactHandle {ref, shape, dtype, sha256, summary}`. `resolve_value` learns to dereference handles. Lineage `value` columns store the **handle plus summary**, never the payload.
**Rationale:** a `CoefficientField` over 10 frames x 64x64 x 4 scales x 6 orientations is ~10M floats; the current `.tolist()` JSON seam cannot carry it.
**Acceptance:** a 100-frame 64x64 sequence round-trips; every lineage row stays under 4 KB; `analyze_boundary` no longer embeds a full padded field in its node.

**Met, measured.** 100 x 64 x 64 round-trips **exactly** (3.15 MB on disk); the `slice_sequence`
lineage node is **1,338 bytes** against the 4 KB budget; `analyze_boundary` was *already* clean
via the summariser layer, now asserted so it cannot regress. Content addressing deduplicates
(identical output -> one file, verified) and verifies: a tampered artifact raises rather than
returning a subtly wrong array. Arbitrary objects are refused rather than pickled - an artifact
only the writing code can read is not reproducible data. Writes are atomic.

**T4A.4 Adapter sequence slicing** - **DONE**
`slice_sequence(dataset_id, variable, time_range, level, lat_range, lon_range) -> FieldSequence`, plus a `slice_sequence` pipeline action.

**Met.** The action stores the frames and returns a *reference*; `resolve_value` dereferences it,
so an action written before the store existed still receives an array. Two guards: a
`max_frames` limit quantifying the memory it prevents (a whole ERA5 record is ~93,000 frames and
the failure mode is a killed process), and a **count of non-finite values replaced** - a zero is
a value, not an absence, and a spectrum of a field with zeroed gaps has structure the atmosphere
does not.

### Phase 4B - `CoefficientField` and the Wavelet Bank

**T4B.1 `CoefficientField` (`src/transform_engine/coefficient_field.py`)** - **DONE**
Axes `(time, scale, orientation, y, x)` with `wavelet_family`, `source_variable`, and phase where the transform is complex. Backed by SWT/DTCWT so all scales share the parent grid, inheriting coords from the source `PhysicalField`. Deliberately mirrors `PhysicalField`'s API so it can flow through analysis the same way.
**Acceptance:** coordinate alignment test across all scales; perfect reconstruction per family; `.summary()` produces a lineage-safe dict.

**Met, with one claim corrected.** All three criteria measured: every scale of both families
returns a `(64, 64)` band on the parent grid; reconstruction error is **2.0e-15 (swt)** and
**2.2e-15 (dtcwt)**; a ten-frame four-level DTCWT bank summarises to **2,008 bytes** against a
payload of **983,040 complex coefficients**. The byte counts vary by a few percent with the
data, because float repr lengths do; the assertion in the tests is the 4 KB budget, not the
figure quoted here.

The corrected claim is "backed by SWT/DTCWT so all scales share the parent grid". That is true
of SWT, which is undecimated, and **false of DTCWT**, whose level-*j* subband is about
`H/2**j`. The alignment for DTCWT is nearest-neighbour **upsampling**, and the field records
`resampled_to_parent=True` with every native shape (`{1: (32,32), 2: (16,16), 3: (8,8)}` for a
64x64 field), because treating an aligned band as if it resolved parent-grid detail would be a
false claim about the data. Nearest neighbour rather than interpolation: the aligned view is a
labelling of parent pixels, and bilinear blending of complex coefficients mixes phases from
different positions, producing values no filter computed.

**Reconstruction therefore never uses the aligned array** - the native coefficients are retained
and inverted. A field without them **refuses** rather than approximating: inverting the
upsampled view returns something a few percent wrong that is indistinguishable from a real
reconstruction, which is a silent failure of the worst kind.

SWT bands are labelled `LH`/`HL`/`HH` and **not** in degrees, because a separable real wavelet's
`HH` responds to both diagonal signs and cannot distinguish them; writing "45 deg" would assert
selectivity the transform has not got, and that claim would propagate into every figure.

**T4B.2 `WaveletBank` config and sweepability** - **DONE**
Declarative `wavelet_bank: {families: [...], scales: [...], orientations: [...]}`. Because these are ordinary parameter-matrix entries, `expand_parameter_matrix` picks them up with **no engine changes** - one of the genuinely free wins in this plan. Keep the 1,000-combination guard.

**Met, and the "no engine changes" claim is asserted rather than asserted-about.**
`combinations()` expands by calling the engine's *own* `expand_parameter_matrix`, and a test
requires the two to produce identical output - so a bank cannot expand differently from an
ordinary matrix, because it is one. `engine.py` was not touched. The 1,000-combination guard is
kept, duplicated in the bank so the refusal names the bank and quantifies the cost, with a test
asserting the two ceilings are the same number.

**One asymmetry in this task's own wording, corrected.** `families` and `scales` are sweep axes;
**`orientations` is not**. A wavelet transform computes every orientation in one pass, so
sweeping them would run the identical decomposition six times and discard five sixths of each
result. Orientations travel as a *selector* applied within each run, and `summary()` says so.

Also closed here: `int(2.5)` is `2`, so a config asking for 2.5 levels ran two and reported two.
Fractional scales are now refused - there is no half dyadic level.

**T4B.3 New pipeline actions:** `decompose_bank`, `extract_scale_signature`. - **DONE**

**Met.** A four-combination bank over five 64x64 frames is **921,600 coefficients** and its
lineage row is **2,415 bytes**; the signature row is **473 bytes**. Both actions refuse a
*dereferenced payload* by name - `resolve_value` turns a literal `artifact://...` into a bare
array, which has no time axis, no grid and no scale labels, so accepting one would mean
inventing a cadence. The refusal names `{step.sequence_ref}` as the fix.

**Scope stated in the result payload, not just here:** `extract_scale_signature` is the *energy
half* of R3. Participation ratio, Gini and threshold counts are T4C.1, and the action says so
under its own `scope` key, at the point a reader actually looks. *(T4C.1 has since delivered
them; the action now returns all four measures and its `scope` string moved with the code
rather than being deleted, so the test that pinned the caveat still pins the new one.)*

**T4B.4 Pressure level as a bank dimension.** ERA5 is `time x level x lat x lon x variable`, and the vertical axis is currently only a *selector* (pick 500 hPa). But the canonical atmospheric precursor relationship is inherently vertical: an upper-level trough preceding surface cyclogenesis. Treat level as a first-class bank dimension alongside scale and orientation - decompose per level, and let 4E constellations span levels with **vertical offset as an edge attribute**. - **DONE**
**Scope discipline:** this is *2D-per-level*, not 3D wavelets. Full 3D transforms are deliberately out of scope for Phase 4 on cost and complexity grounds; per-level decomposition with cross-level edges unlocks the most physically famous precursor structure at a fraction of the price, and is the natural target for T4F.6's known-phenomenon gate.

**Met.** `slice_level_sequences` returns one sequence per level in a single pass, so the shared
time axis is a property of the construction. `LevelBank` validates shape, family and
**timestamps** across levels - a vertical lead-lag across levels sampled at different times
measures the sampling, not the atmosphere. `vertical_offsets()` is the 4E edge attribute:
signed, in hPa, with direction named, because pressure decreases upward and a reader who got
that backwards would invert every precursor relationship the bank exists to find.

**Two silent failures closed, both found by building it:**

*   `decompose_bank` originally ignored `levels_hpa` - it would have decomposed one sequence
    repeatedly and labelled **identical arrays** 850 hPa and 500 hPa, and every cross-level
    statistic downstream would have measured that fiction. A vertical bank now requires one
    sequence per level and refuses to guess.
*   `sel(method="nearest")` maps a level a dataset lacks onto its neighbour, so two requests can
    return the same data. `slice_level_sequences` compares the arrays and refuses when they are
    identical: a cross-level correlation computed from them is a field correlated with itself, a
    coefficient of 1.0 that means nothing and looks like a discovery. The guard fired
    immediately on `t2m`, which has no vertical axis at all.

The scope discipline is asserted, not just written: `LevelBank.summary()` states that it is 2D
per level and not a 3D transform, and a test requires that sentence to be there.

### Phase 4C - `ScaleSignature`, `SurrogateNull`, Cross-Scale Dependency <<< THE GATE >>>

This phase is self-contained: it needs **nothing** from 4D-4G, and it is where the project finds out whether the central idea is real.

**T4C.1 `ScaleSignature` (`src/analysis_engine/scale_signature.py`) *(implements R3)*** - **DONE**
Per `(t, scale)`: energy fraction, participation ratio, Gini coefficient, and - reported but never primary - threshold-based counts with the threshold recorded. Direct generalisation of the existing `compute_wavelet_energy`.
**Acceptance:** signature of a pure sinusoid concentrates at its scale; signature of white noise is flat in energy fraction; both invariant to a global amplitude rescale.

**Met, and the measures are asserted against their analytic values rather than against
"roughly flat".** A sinusoid of wavelength 8 concentrates at level 3 and one of wavelength 16
at level 4, in both families; white noise is flat to within 15%; and every threshold-free
measure is bit-identical under a 1,000x rescale.

On white noise the coefficient energy of a **real** transform is chi-squared with one degree
of freedom, so the participation ratio is exactly `1/3` and the Gini exactly `2/pi`; for a
**circular complex** band it is exponential, giving `1/2` and `1/2`. Measured: **0.334 / 0.636**
(SWT) and **0.499 / 0.500** (DTCWT level 2). Those are predictions, not snapshots.

**The threshold measure is reported and demonstrably not primary.** Moving the threshold from
2 to 4 sigma changes the threshold count by more than a factor of ten while the other three
measures are *bit-identical* - rule R3's first trap, measured rather than described.

**Three things that would have been silently wrong:**

*   A resampled DTCWT band repeats every native coefficient `4**j` times. Energy *fractions*
    are unaffected (the factor cancels in the ratio - checked, not assumed), but the
    participation ratio is multiplied by exactly `r`. Measures are therefore taken on native
    coefficients, recovered from the aligned view by exact stride subsampling when the native
    arrays are gone.
*   **Rule R13's crop table understates the dual tree by nearly two.** It is derived for one
    14-tap filter at every level; DTCWT's cascade gives a level-4 margin of **97 parent pixels
    against the table's 52**, so a 256x256 crop - this roadmap's stated minimum for four
    levels - leaves DTCWT level 4 a **2x2** interior. Reported as *thin* by name; 512x512
    restores it to 18x18. Logged as **D40**.
*   Level 1 of a DTCWT is not an analytic signal: the q-shift Hilbert pair starts at level 2,
    and the real and imaginary variances of a level-1 subband differ by a factor of **2.19**
    on white noise against 1.00-1.06 above it. Its participation ratio sits at 0.467, between
    the real and circular-complex values, and is predictable from those two variances alone.

**T4C.2 `SurrogateNull` (`src/analysis_engine/surrogate_null.py`) *(implements R1)*** - **DONE**
Phase-randomised and AAFT surrogate generators preserving the PSD (and, for sequences, per-frame PSD). Ensemble of N (default 200) with effect sizes and empirical p-values against the ensemble.
**Acceptance:** surrogate PSD matches source PSD within tolerance while phase correlation is destroyed; a known-organised synthetic field scores significantly, and a fractional Brownian field does **not**.

**Met.** The spatiotemporal surrogate preserves the 3D power spectrum to **3.4e-16** while the
frame-by-frame correlation with the source falls below 0.1; a sequence of localised blobs
scores at the p-value floor with an effect size above 3; a fractional Brownian sequence does
not score.

**One deviation from the task, stated rather than quietly taken.** The module is
`surrogate_null.py`, not `surrogates.py`: two modules of that name in one codebase resolve
differently depending on which package the reader is in, and it invites the
phase-randomisation core to be forked. The generators in `src/statistics/surrogates.py` are
called, not reimplemented - `phase_randomise` was generalised from 2D to any dimension, which
its arithmetic already supported.

**The choice of null is a choice about time, and the wrong one is not subtle.** On an AR(1)
record with no organisation at all:

| null | null lag-one autocorrelation | record's |
|---|---|---|
| `spatiotemporal_phase` (default) | **0.845** | 0.900 |
| `per_frame_phase` | **0.013** | 0.900 |

Preserving the *per-frame* PSD is what this task literally asks for, and it destroys the
record's temporal structure - so the autocorrelation itself beats the null (rule R12). It is
kept, with a warning on every result that uses it, because being able to demonstrate the
failure is worth more than removing it.

**T4C.3 Cross-scale lagged dependency *(implements R4)*** - **DONE**
Lagged mutual information and transfer entropy over the $A_t(s)$ matrix, for all scale pairs and admissible lags, with the per-scale support floor enforced and reported.
**Acceptance:** recovers an injected cross-scale coupling in a synthetic cascade; reports **null** on a phase-randomised version of that same field.

**Met, end to end.** `src/synthetic_generator/cascade.py` builds a record whose fine band at
`t` sets its coarse band at `t + 3`, driven by a *red* modulation - a white one would have made
the test far too easy. Through the whole path (decompose, signature, sweep, 1,999
circular-shift surrogates, Benjamini-Yekutieli):

| record | significant | which |
|---|---|---|
| cascade | **2** | `1 -> 3 @ lag 3`, `2 -> 3 @ lag 3`, `q = 0.0157` |
| the same record, phase-randomised | **0** | - |

Correct lag, correct direction, nothing in reverse.

**Two calibrations that decide whether any of it means anything, both found by building it:**

*   **A linear lag against a circular null rejects every time.** FT surrogates are circularly
    stationary and a record is not. On twenty AR(1) records where the null is true by
    construction: **20 of 20** false rejections with a linear lag, **0 of 20** with a circular
    one (median p 0.010 against 0.485). Lags therefore wrap, and the wrap fraction is reported.
    This one would have produced a gate that passed on pure red noise.
*   **The shift null must exclude the simultaneous alignment as well as the tested one.**
    Rolling the source by `s` measures effective lag `lag + s`, so `s = -lag` puts the series
    at effective lag zero - a real alignment, not a shuffle. On the cascade that single shift
    gave a transfer entropy of **0.412 nats against an observed 0.211**, the largest value in
    the whole "null" ensemble, capping the achievable p-value at about 0.005 however many
    surrogates were drawn.

**Rule R4's floor, stated honestly.** The transform is spatial and applied frame by frame, so
its temporal support is **zero**; the real floor is the advective crossing time of the filter
support, `support * dx / U`. `support_floor` computes it and **refuses to default the wind
speed** - a plausible 10 m/s would set every floor in every result from a number the reader
never chose. Without it the only floor is one frame, and the result says so.

**And a sweep that cannot reject anything says so.** The full 120-test sweep needs **12,885
surrogates** under Benjamini-Yekutieli; `check_power` is consulted before the result is read,
because a study arithmetically incapable of rejecting is otherwise indistinguishable from a
clean negative.

**T4C.4 Generic `fit_power_law` and scale-population exponent *(implements R2, R3)*** - **DONE**
Factor the log-log least-squares core out of `fit_spectral_slope` into a reusable `fit_power_law(x, y, x_min, x_max)`; keep the Charney/Kolmogorov interpretation as a thin turbulence-specific wrapper so existing behaviour is unchanged. Apply it to $N(s)$ / energy vs scale, normalised per R3, always reported against $\alpha_{\text{surrogate}}$.
**Acceptance:** existing `fit_spectral_slope` tests still pass unchanged; recovers a known exponent from a synthetic multifractal.

**Met.** The least-squares core is now `analysis_engine/power_law.loglog_fit`, which knows
nothing about turbulence; `spectra.fit_power_law` calls it and adds the Charney/Kolmogorov
interpretation on top. Existing behaviour is unchanged - the 118 tests in the three files that
exercise the fit (`test_grid_operators.py`, `test_benchmarks.py`, `test_hypothesis.py`) pass
untouched - and a new test pins the generic core and the spectral wrapper together so they
cannot drift apart.

**Rule R2 is enforced in the return value, not in prose.** `compare_exponent_to_null` returns
`reportable: False` until an exponent has been placed beside a surrogate ensemble, because a
fractional Brownian field yields a clean high-`r_squared` power law and contains nothing.
`reportable` is deliberately not conditioned on *significance*: a null result is reportable and
is often the point.

`scale_energy_exponent` fits `A(s) ~ s**-alpha` against the per-available-coefficient density
- rule R3's normalisation, since an unnormalised count in a decimated pyramid already falls as
`s**-2` and would recover `2 + physics` with the two indistinguishable. Recovery of an injected
`s**-1.5` is exact to `r_squared = 1.0`.

**T4C.5 FDR control utility *(implements R5, fixes D8)*** - **DONE**

**Met, and it went further than the task asked.** `src/statistics/` delivers three modules:
`multiple_comparisons.py` (Bonferroni, Holm, BH, BY - the FDR pair validated against
`scipy.stats.false_discovery_control` to **1e-16**), `surrogates.py` (phase randomisation,
AAFT, IAAFT, circular shift, block bootstrap) and `significance.py` (surrogate tests,
ESS-corrected correlation, stationarity gate). 41 tests in `test_statistics.py`; suite
407 -> 446.

**The D8 scenario, before and after.** 20 tests on 9-sample noise - the exact shape of the
smoke sweep that produced 9 spurious "discoveries":

| rule | reported |
|---|---|
| `abs(r) >= 0.5` (the pre-D8 engine) | 3 of 20 |
| raw `p <= 0.05` | 1 of 20 |
| **after Benjamini-Yekutieli FDR** | **0 of 20** |

**And power is retained**, which matters as much: one real effect hidden among 19 nulls at
n = 40 is recovered, and only it. A correction that rejects everything would have "closed" D8
by making the platform mute. The engine is silent when underpowered (12 points, weak effect ->
nothing) and speaks when the evidence is there (40 points, same effect -> found).

**Three additions the task did not specify but defendability required:**

1.  **Surrogate nulls** (rule R1). A threshold is not a null model. Wavelet scales are
    algebraically coupled by the transform, so a cross-scale correlation must be compared
    against data with the same second-order structure and no genuine coupling.
2.  **A stationarity gate.** FT surrogates assume stationarity; without it the test is
    anti-conservative at measured rates of 0.11 / 0.16 / 0.39 / 0.78 (AR(1) phi = 0.80, 0.85,
    0.95, random walk) against a nominal 0.05. Atmospheric series are routinely strongly red,
    so this is the default situation rather than a corner case. The threshold is **calibrated
    against those measured rates**, not chosen (rule R16).
3.  **A power check.** A surrogate p-value floors at `1/(1+n)`, so 99 surrogates cannot reject
    one of 500 tests after correction - such a study reports nothing while looking like a clean
    negative. It now says so.

Every reported `Hypothesis` carries `p_value`, `q_value`, `n_tests` and a `statistics` block
naming the test, the correction, its dependence assumption and the R7 non-causality caveat -
all surfaced through the API, because a finding that travels without them cannot be judged.

**This work made the fBm benchmark's `4C.surrogate_null` gate enforceable, and that gate
immediately caught a real defect in the surrogate machinery** - see `architecture.md` 7.2h.
The suite is now **15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE**.
Benjamini-Hochberg helper; retrofit it onto the **existing** hypothesis engine as well as all new mining. Every `Hypothesis` row gains `n_tests_in_family`, `p_value`, `q_value`, `surrogate_effect_size`.

**T4C.5d Bounded, content-bound gate execution *(fixes D46-D50)* - DONE.** The calibrated
statistics were not yet an executable real study. Four failures were closed before acquiring
the expensive record:

* CDS conversion no longer loads every monthly shard and concatenates the full record in RAM.
  It validates and appends bounded time blocks to a sibling Zarr store, streams the same
  chunk-independent logical content hash, verifies the complete time axis, and atomically
  publishes only after success.
* Before the first CDS client call, storage preflight budgets the remaining NetCDF shards and
  complete temporary Zarr independently without assuming compression, combines requirements
  on a shared volume, and preserves the greater of 5 GiB or 10% working-space reserve.
* `stream_scale_signature` produces the exact eager energy, concentration and threshold
  measures in two passes while retaining one source/coefficient frame. It hashes both passes
  and refuses a source that changes between them. Test thresholds can be supplied from train,
  preventing a secondary threshold diagnostic from fitting on held-out data.
* `fit_harmonic_climatology_stream` uses the same explicit SVD pseudo-inverse as the accepted
  R11 implementation, fits only declared training indices, retains `(parameters,H,W)` rather
  than `(time,H,W)`, and matches eager anomalies to float64 tolerance.
* `GateStudyPlan` now freezes crop, variable, level, transform/filter/boundary semantics,
  climatology, advection speed and `GateProtocol` under one SHA-256. Local-only preflight checks
  exact frames/cadence, actual transform interiors and the full filter-support floor. For a
  frozen real study it enforces R13's 128-parent-pixel minimum from those actual filters; the
  generic 14-tap 512/1024 planning floor is not incorrectly imposed on a known shorter filter.
  The
  runner binds train/test results to that plan, reuses train-fitted thresholds, and atomically
  publishes a tamper-detecting receipt. Synthetic acceptance remains
  `scientific_verdict: NOT_ESTABLISHED`; a real role additionally requires direct CDS provenance
  and a recorded PASS from the independent WeatherBench overlap check.

**T4C.5e Content-bound independent ERA5 overlap *(fixes D51)* - DONE.** The previous
cross-route checker returned only an in-memory dictionary, while the gate trusted a mutable
manifest string. `era5_overlap.py` now compares a small, exact-coordinate local CDS/WeatherBench
overlap in bounded frame blocks with per-variable units and tolerances, writes an immutable
hash-verified receipt, and atomically binds it to the CDS content and acquisition request. The
real gate validates the receipt schema, hash, source identities, variable and pressure level;
a bare or tampered `PASS`, a failed comparison, configuration drift, interpolation or network
fallback cannot authorise T4C.6. This is offline-accepted infrastructure: no live overlap has run.

**T4C.5f Frozen two-stage acquisition campaign *(fixes D52)* - DONE.** A full multi-year
transfer is no longer the first live test of the CDS route. `gate_campaign.py` freezes the full
request, a small CDS canary with identical variables/levels/hours/grid, its exact catalogued
WeatherBench overlap, and the final `GateStudyPlan` under one SHA-256. Offline preflight budgets
all canary/full NetCDF and Zarr artifacts plus the overlap cache together, without compression
credit; checks only the presence (never values) of standard credential configuration,
`cdsapi`, and explicit network consent; and constructs no client. The mandatory order is
WeatherBench overlap -> CDS canary -> require PASS -> full CDS record -> require a new PASS
bound to the full cache -> T4C.6. Unknown or inconsistent nested design fields are refused.
The acquisition mechanism was accepted here without selecting the scientific design; T4C.5h
subsequently supplies that preregistration. No live step has run.

**T4C.5g Pre-acquisition physical geometry audit *(fixes D53-D54)* - DONE.** Review found
that `support_floor` read `GridSpec.dx` before fields explicitly labelled in metres. For a
lat/lon ERA5 grid, `dx=0.25` is degrees; treating it as 0.25 metres collapsed a filter footprint
of hundreds of kilometres to a few metres and admitted contaminated one-frame lags. Physical
grids are now reconstructed from provenance, angular axes are converted to metres, and the
larger cell axis across the crop sets a conservative direction-agnostic floor. The campaign
also derives exact grid points and actual SWT/DTCWT support before transfer, refuses a crop
below R13 or lags below the physical floor, reports split/climatology coverage, family power and
surrogate work, and requires bounds to lie on integer grid intervals. CDS materialisation
independently rejects server-snapped endpoints even when spacing is correct.

The support-floor audit found D48: the code claimed filter support but used `2**level`. It now
uses SWT/DTCWT's exact accumulated `support_parent_px`; for db2 level 3 that is 22 pixels, not
the scale label 8. The synthetic Zarr-to-receipt job passes without network access.

**T4C.5h Primary T4C.6 scientific preregistration - DONE.** The exact primary campaign is now
the authenticated `campaigns/t4c6_nz_era5_temperature_850_v1.json`, rather than parameters
copied from examples at acquisition time. It freezes 2018--2022 at six-hour cadence over the
161x161 Southwest-Pacific/NZ-supporting crop (20--60 S, 140--180 E), one deliberately primary
field (850-hPa temperature), three exactly shift-invariant db2 SWT scales, energy density,
transfer entropy, six 18--48-hour lags, six equiprobable bins, 4,999 circular-shift surrogates,
BY correction at 0.05, an eight-frame embargo and seed 20260821. The 36-test family needs at
least 3,005 surrogates, retains 139x139 valid parent pixels at level 3 and starts exactly at
the conservative three-frame support floor. The split is 4,382 train / 8 embargo / 2,914 test
frames. A zero-network `review` command emits those facts, exact calendar boundaries, hashes
and claim boundary. Secondary variables, levels, transforms or regions may be exploratory
follow-ups but cannot replace this verdict. The choices keep the family singular: temperature
is the scalar field shared with the downstream 850-hPa regional work; SWT avoids position-phase
sensitivity; db2 is the shortest non-Haar accepted wavelet; three levels are the deepest that
retain the R13 interior on this crop; five complete years give nearly three annual cycles to
the train-only climatology and comfortably populate the six-bin conditional estimator; and
the lag window begins at the physical floor rather than at a convenient one-frame lag.

The actual zero-network preflight passed storage: 4.75 GiB working requirement plus a 5 GiB
reserve against 1,180.06 GiB free on D:. It constructed no client and remains blocked on the
three expected local prerequisites: `cdsapi`, standard CDS credential configuration and explicit
network consent; service-side licence acceptance remains unproven. No acquisition or atmospheric
test ran in this task.

**T4C.6 GATE REVIEW.** Written verdict: does cross-scale organisation exceed the surrogate ensemble at $q < 0.05$, at lags above the support floor, on real ERA5 data?
*   **Pass** -> proceed to 4D.
*   **Fail** -> stop. Write up the negative result. It is a genuine, publishable-shaped finding that the observed cross-scale coefficient structure is explained by the power spectrum alone, and it saves 4D-4G entirely.

**Not yet run, and deliberately not pre-judged.** T4C.1-T4C.5 are complete and calibrated, and
on synthetic data the instrument gives the answers a working instrument should: it finds an
injected cascade at the right lag and direction, and finds nothing in that same record once
the alignment is destroyed. The verdict belongs to a run on real ERA5 data; writing it from
synthetic evidence would be exactly the kind of claim this phase exists to prevent.

**The decision protocol is now executable, but the data requirement is not yet met
(T4C.5c).** `GateProtocol` freezes the scale/lag family, estimator, bins, surrogate count,
alpha, correction, split, embargo and seed behind a stable SHA-256 fingerprint. It refuses a
study whose surrogate p-value floor cannot survive correction, whose train or test partition
has fewer than five samples per joint-estimator cell, or whose embargo is shorter than the
longest tested lag. `evaluate_replication_gate` returns PASS only when the same positive,
corrected relationship occurs independently in train and test; an adequately powered absence
is FAIL, while configuration drift, missing advection support or inadequate power is INVALID.

No gate-sized crop is currently cached. Live metadata inspection of the 0.7-degree store found that the
three-year, one-variable record needed to populate independent transfer-entropy partitions
would fetch an estimated **29.88 GB** because every eight-frame chunk still spans all levels
and the globe (26.2x amplification). The 0.25-degree source is worse. T4C.6 is therefore
blocked by **D43**, not complete: add a temporally deep, spatially tiled source (or a direct
regional CDS acquisition path), verify its values/provenance against ERA5, then freeze the
exact crop before transfer. The acquisition and analysis paths are now bounded and executable;
the remaining boundary is external evidence: CDS credentials/licence acceptance, the live
multi-year transfer and a passed independent-route overlap receipt. The sample requirement will
not be relaxed to fit the old layout.

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
**Additionally:** WeatherBench 2 publishes *precomputed forecasts* from operational and ML
models (IFS HRES, GraphCast, Pangu, and others) alongside ERA5 ground truth. Those errors let
the score face an external predictive target without running a model, pulling an informative
diagnostic forward into Phase 4G at low compute cost. This is not a direct validation of
representation choice because those forecasters were not trained on the candidate
representations. Failure to track their error structure is evidence against the score's claimed
generality, but does not by itself identify whether the cause is the formula, low power, domain
shift or forecaster mismatch.
**Rationale:** persistence and precomputed forecast errors are cheap falsification probes. A
failure against either requires diagnosis before model integration; it is not automatically
attributed to one component.

**T4G.2 The score.** Recurrence + sparsity + temporal persistence + cross-scale coherence + spatial coherence + **predictive information** + generalisation. Predictive information is the **anchor** term; without it the score rewards whichever wavelet is busiest.

**T4G.3 Strict evaluation *(R6)***. All predictive terms computed only across `split_temporal` with embargo. Report per-term contributions, never a single opaque number.
**Acceptance:** the scorer ranks a deliberately-crippled representation (e.g. random orthogonal basis) below a physically-appropriate one, and ranks a busy-but-uninformative wavelet below a sparse-but-predictive one - the distinction that motivated the whole design.

**T4F.7 Cross-region generalisation *(implements R14)***. Re-test every candidate pattern on held-out regions; report where it holds and where it fails, with physiography noted. A pattern is labelled `regional` or `general` accordingly.

**T4F.8 Follow-up experiment proposals.** Extend the existing `_propose_numerical_followup` / `_propose_categorical_followup` pattern to propose experiments that *test* a discovered precursor - the platform closing its own loop.

### Phase 4H - Learned graph encoder (OPTIONAL, ceiling estimator only)

There are two ways to answer "which little graphs predict which bigger graphs", and the choice matters more than it looks:

*   **(a) Symbolic** - frequent attributed-pattern mining with support and lift (T4E, T4F). Auditable, human-readable, slow, combinatorial. **This is the deliverable.** It is what makes SpectralEarth a research instrument rather than another opaque predictor.
*   **(b) Learned** - a GNN over the constellation graphs trained to predict coarse-scale state at $t+\Delta$. More expressive and much faster, but it reintroduces exactly the interpretability problem this platform exists to dissolve. Using a black box to explain a downstream forecast model would be self-defeating.

**Therefore (b) is admitted in one narrow role only: as a ceiling estimator.** Train the GNN purely to measure *how much predictable signal the graph representation contains at all*, which bounds how good the symbolic rules in 4F could ever become. A large gap between the GNN's skill and the symbolic rules' skill means the mining is leaving signal on the table and 4E's attribute set or matching tolerance needs revisiting. A small gap means the symbolic rules have extracted essentially everything available, and the work is done. The GNN's predictions are **never** presented as findings.
**Acceptance:** reports one number - the symbolic-vs-learned skill gap on the held-out temporal split - and nothing else.

---

## 6. Phase 5 - Training-Native Regional Forecast Research

**Reframing:** the downstream model is a judge, not the product. The first judge is the
laboratory's existing lightweight New Zealand model because that produces immediate research
value and the sharpest falsifiable question. External global models are later adapters for
generalisation, not prerequisites. FourCastNet 3 is now the first specified external target
because its 850-hPa outputs, probabilistic ensembles and published spectral-fidelity claims are
directly relevant to the NZ study; that prioritisation does not make it implemented.

**Fast-track dependency:** T5.0--T5.4 depend on the verified transform core and a resolution of
D43, not on completion of feature tracking/mining in 4D--4F. They may proceed as a useful
vertical slice while the broader discovery engine continues. Nothing in this section is
implemented unless its acceptance evidence appears in `VERIFICATION.md`.

*   **T5.0 Freeze the motivating experiment contract -- PARTIAL (T5.0a-b contracts accepted; real protocol not supplied).** Obtain the actual repository/config or
    an exported manifest for the laboratory model: domain coordinates and tensor shape,
    variables/levels, cadence, input history, lead times, transform package/version, filters,
    decomposition depth, boundary mode, coefficient packing, normalisation, split dates,
    optimiser, rollout and parameter counts. Poster-derived estimates are not substitutes.
    **Acceptance:** a versioned, hashable protocol object can reproduce the declared design and
    rejects an incomplete configuration.
    **Delivered T5.0a:** `src/forecasting/protocol.py` defines schema
    `motivating-forecast-protocol/v1` with exact nested sections for evidence identity, domain,
    dataset/cadence, history and physical leads, transform semantics, train-only normalisation,
    calendar splits/embargo, optimiser and schedule, rollout semantics and model parameter
    counts. Parsing rejects missing and unknown keys, placeholders, unsupported schema versions,
    invalid hashes, temporal/rollout contradictions and an embargo shorter than the maximum
    lead. Free-form optimiser/model JSON is recursively frozen. Canonical serialization gives a
    full SHA-256; persisted envelopes refuse overwrite and detect tampering. Fourteen executed
    acceptance cases prove completeness across every scientific section, deterministic identity,
    immutability and drift detection.
    **Outstanding:** no genuine laboratory repository/config, coordinate/statistics hashes or
    evidence SHA-256 has been supplied. The test fixture is deliberately synthetic and is not
    Emily's/Adam's protocol. T5.0 remains partial and T5.3c cannot select model semantics from
    poster estimates.
    **Delivered T5.0b:** `src/forecasting/binding.py` compares prepared dataset provenance to
    every observable frozen field: source/version, variables/level, cadence, histories/leads,
    grid geometry plus a recomputed coordinate SHA-256, exact split instants, embargo and a
    recomputed train-statistics SHA-256. A checkpoint binds only when model class/config/counts,
    transform semantics, optimiser, rollout, model version and training provenance all name the
    same protocol and dataset hashes. Bound evaluation schema v3 refuses substituted dataset,
    checkpoint, variables or leads and refuses batch timestamps outside the declared held-out
    split. Dataset coordinate/time/statistics identities are now full 64-hex SHA-256 values;
    ambiguous regional longitude convention must be declared rather than inferred. Eight
    executed acceptance cases cover the valid chain and intentional drift/refusal classes.
    This closes runtime identity mixing, not T5.0 itself: all evidence remains synthetic until
    the real laboratory manifest is supplied, and no UI status is promoted without one.
*   **T5.1 Build `RepresentationModule` -- PARTIAL (T5.1a-e raw/FFT/DCT/Haar/db2/SWT/DTCWT accepted).** Provide raw, FFT, DCT, Haar, db2, SWT and DTCWT
    modules over `(B,C,H,W)` with forward/inverse operations, stable structured outputs,
    explicit complex packing and boundary/support metadata. Filters or cosine matrices are
    cached/registered by device and dtype rather than rebuilt each step. The analysis and
    training paths consume one canonical filter definition.
    **Acceptance:** batch-versus-item equivalence, reconstruction tolerance, `gradcheck`,
    forward-and-inverse gradient flow, CPU/accelerator parity on every available PyTorch
    backend, device/dtype
    migration, deterministic repeatability, mixed-precision policy, and measured runtime,
    activation memory and cache reuse. An unavailable backend is recorded as NOT RUN, never PASS.
    **Delivered T5.1a:** `training.py` defines immutable `EncodedRepresentation`, the abstract
    module contract, a stable factory and accepted raw/FFT/DCT implementations. FFT uses
    real/imaginary channel packing; DCT uses registered, migration-aware buffers built from the
    canonical matrix definition. Batch/item equivalence, odd/even reconstruction, gradcheck,
    malformed-context refusal, cache reuse and float32/float64 migration are tested. CPU/RTX
    CUDA coefficient parity, reconstruction and backward gradients are verified through a
    vendor-neutral accelerator test.
    **Delivered T5.1b:** Haar/db2 implement an invertible, recursively packed Mallat plane with
    explicit periodic boundary convention and dyadic padding/crop metadata. Level-one bands
    agree with PyWavelets `periodization`; multilevel energy, batch/item equivalence, odd-shape
    reconstruction, `gradcheck`, CPU/RTX coefficient parity and backward gradients are tested.
    The support metadata uses the corrected complete cascade from D44. This clear `torch.roll`
    implementation is the accepted numerical reference, not yet the accepted per-step
    performance kernel: measured db2 encode+inverse was 9.20 ms on CPU and 13.74 ms on this RTX
    for float32 `(4,5,120,80)` at three levels.
    **Delivered T5.1c:** the default cached four-band convolution kernel matches the retained
    `implementation="reference"` path in coefficients, inverse and input gradients. On the same
    batch, db2 round trip is 3.92 ms CPU / 1.45 ms RTX and a round-trip-plus-backward training
    step is 14.09 ms CPU / 8.66 ms RTX. Incremental CUDA peak is 5.87 MiB against 5.31 MiB for
    the reference; the speedup therefore costs 0.56 MiB on this workload. Kernel cache reuse and
    dtype migration are tested.
    **Delivered T5.1d:** SWT packs final LL plus levelwise LH/HL/HH on the parent grid, supports
    Haar/db2/db3, reconstructs exactly and is exactly translation-equivariant under periodic
    shifts. The convolution path agrees with both the coordinate-aware analytical SWT and an
    FFT training oracle in coefficients, inverse and coefficient-loss gradients. Auto policy
    uses the measured faster FFT path on CPU and convolution on CUDA/ROCm/MPS. Metadata and the
    new UI readiness endpoint/panel expose `1+3L` redundancy, level gain/energy normalisation,
    support, valid interior, artificial crop wrap, three-band directional limitation, verified
    CPU/RTX evidence and ROCm/MPS/mixed/compile NOT RUN status. For db2 L3 `(4,5,120,80)`, packed
    storage is 7.324 MiB; round-trip/training-step is 19.81/47.22 ms CPU and 6.40/14.53 ms RTX,
    with 25.46 MiB incremental CUDA peak.
    **Delivered T5.1e:** DTCWT vmaps the canonical Kingsbury transform over `(B,C)` and caches
    all analysis/synthesis filters as migration-aware buffers. A lossless four-real-plane
    recursive atlas preserves every native complex coefficient without interpolation and
    supports arbitrary model outputs. Metadata and UI expose nominal/measured orientations,
    native/parent edge margins, valid interiors, level-1 directional uncertainty and artificial
    atlas adjacency. Analytical coefficient maps show complex magnitude on native grids with a
    shared within-level scale and marked valid inset; phase is retained but not painted as a
    physical scalar. The module refuses a level/domain combination with no strict 2D interior.
    Coordinate-aware oracle agreement, exact atlas bijection, coefficient-loss gradients,
    float64 gradcheck and RTX parity pass. On float32 `(2,5,120,80)`, L2, measured
    round-trip/training-step is 46.23/138.34 ms CPU and 35.87/65.91 ms RTX; storage is
    1.465 MiB and maximum error 1.20e-6.
    **Outstanding:** AMD ROCm/MPS hardware evidence, mixed precision and
    compilation policy. Browser rendering of the new panel is NOT RUN in this session because
    no controllable browser was attached; TypeScript production build and API/UI contracts pass.
    **Portable execution companion delivered:** `SPECTRAL_PROFILE=auto` uses acceleration when
    available and remains fully CPU-capable; `cpu` and `accelerator` make intent explicit;
    `hpc` recognises Slurm/PBS/LSF allocation and local rank while refusing login-node use.
    `python -m src.core.doctor` performs CPU and detected-accelerator FFT/backward smoke tests
    and emits attachable JSON. This is local placement/preflight, not cluster submission or
    artifact synchronisation; those require Adam's actual HPC contract.
*   **T5.2 Build `RegionalForecastDataset` -- PARTIAL (T5.2a-d accepted; live data gate open).** Materialise and rechunk a provenance-carrying New
    Zealand crop with aligned 850-hPa `t/q/u/v/z`, timestamps and grid coordinates. Yield input
    histories and lead-time targets as tensors; apply `split_temporal` and a lag-sufficient
    embargo before sample construction; fit every normalisation statistic on training data
    only. Selecting 850 hPa does not reduce transfer from a source whose chunks span all
    levels, so D43 must be solved with a spatially tiled temporal source or verified direct
    regional acquisition rather than hidden by a short record.
    **Acceptance:** values and coordinates cross-check against an independent ERA5 route on a
    small overlap; no input or target crosses a split/embargo boundary; provenance fingerprints
    the source, crop, variables, levels, timestamps, chunking and normalisation artefact.
    **Delivered T5.2a:** `regional_forecast.py` resolves canonical `t/q/u/v/z` from either short
    or WeatherBench names, pins 850 hPa, refuses coordinate mismatch/non-finite data and reuses
    `split_temporal` with an embargo at least as long as the maximum lead. Frames are split
    before histories/targets are indexed. A hashed population mean/std artifact is fitted in
    float64 on training frames only and reused by validation/test. Default-collatable dataset
    items contain input/target tensors, Unix-nanosecond timestamps and original frame indices;
    bundle provenance fingerprints the source manifest, variables, crop/chunking, grid, time,
    split and normalisation. `prepare_cached_regional_forecast` performs a measured local-only
    read from any explicit laptop/HPC cache path. `cross_check_era5_overlap` requires exact
    coordinates and reports per-variable errors. The T5.2a tests include a real local Zarr
    materialise/rechunk/cache round trip and DataLoader collation. The ERA5 UI labels manifest
    structure separately from preparation, train-only fitting and independent-route evidence.
    **Delivered T5.2b:** cached preparation opens metadata lazily, streams float64 train-only
    moments in configurable bounded frame blocks using Chan merging, and reads only requested
    history/target windows. Live xarray handles are excluded from pickling; each spawned worker
    opens its own local Zarr handle, while a PID guard safely reopens after HPC fork. The bundle
    has explicit `close()`/context-manager lifetime. Lazy/eager statistics and tensors agree,
    a two-worker forced-spawn loader passes after a parent read, and provenance records lazy
    mode and both requested/stored block bounds. Oversized or unrecorded on-disk time chunks are
    refused with a rematerialisation instruction, since lazy indexing alone cannot bound them.
    Ten focused tests cover T5.2a-b.
    **Delivered T5.2c offline contract:** `cds_source.py` freezes exact CDS pressure-level
    requests, plans calendar-month shards, requires explicit network consent, writes validated
    NetCDF downloads atomically, resumes only after SHA-256 verification and records no
    credentials. It requires exact returned times/levels/grid, refuses unresolved dimensions or
    non-finite values and converts the shards into the unchanged bounded-time-chunk local Zarr
    contract consumed by `RegionalForecastDataset`. Provenance replay is route-aware. Eight test
    functions (12 executed cases) cover planning/CLI/refusal, network gating, resume/tamper behavior,
    cache replay, exact timestamp enforcement and lazy dataset compatibility. This is offline
    service-contract evidence: no CDS request or ERA5 agreement has run.
    **Delivered T5.2d:** `calendar_boundaries=(validation_start, test_start)` provides exact,
    reproducible split dates alongside the retained ratio mode. Boundaries must occur on the
    returned axis and each is followed by the configured lag-sufficient embargo. An explicitly
    requested cadence must match every interval. Dataset schema v2 derives lead durations from
    timestamps and records whole-axis cadence; evaluation schema v2 independently recomputes
    them, requires one regular frame-to-hour mapping across the complete axis and all
    samples/batches, and reports frames plus hours. Missing,
    irregular or tampered timing is refused. The manifest-only UI truthfully shows ratio dates
    as unfrozen, cadence as `NOT VERIFIED` and physical lead labels as unavailable. This prevents
    Emily's ambiguous "6 hourly forecast steps" from being silently interpreted as six-hourly
    data. Focused backend/UI contracts and the 1,385-module production build pass; browser render
    inspection was not run because no controllable browser was attached.
    **Outstanding:** run the CDS path live, acquire a viable multi-year NZ crop, and execute the
    existing exact coordinate/value overlap checker against WeatherBench on a small common
    window. Record queue time, wire bytes, cache size and returned schema. D43 remains open and
    T5.2 is not complete until those observations exist.
*   **T5.3 Integrate the existing laboratory model -- PARTIAL (T5.3a-b contracts accepted; laboratory model open).** Put its train/evaluate operations behind
    the `Forecaster` seam without copying model logic into the transform engine. Supply a
    minimal importable example and persistence baseline before dashboard or REST integration.
    **Acceptance:** one deliberately tiny end-to-end run executes dataset -> representation ->
    model -> inverse -> loss -> backward, then a fixed small fixture reproduces deterministically.
    A tiny run proves integration only; it is not forecast evidence.
    **Delivered T5.3a:** `src/forecasting/adapter.py` defines a normalised physical-space
    `Forecaster.predict(history, lead_count)` contract. `PersistenceForecaster` repeats the last
    observed state exactly with zero parameters. `ForecasterAdapter` wraps a one-step encoded
    model with an accepted representation/inverse, rejects shape/dtype/device/non-finite output
    drift and performs physical-state autoregressive rollout. The importable tiny runner consumes
    a real `RegionalForecastDataset` batch, computes MSE and calls `backward()`, recording seed,
    device, dtype, shapes, parameter count, loss, finite gradient norm and prediction/gradient
    hashes with a no-skill claim boundary. Five tests prove exact persistence, deterministic CPU
    reproduction, Haar-in-loop gradients, refusal behavior and CPU/RTX prediction/gradient parity
    through the vendor-neutral CUDA/ROCm API.
    **Delivered T5.3b:** `src/forecasting/artifact.py` saves a tensor-only state dict and canonical
    manifest, verifies model/representation config, model class, checkpoint SHA-256 and tensor
    schema before safe strict loading, and embeds that identity into forecaster provenance.
    `evaluation.py` performs bounded-memory, exactly matched held-out comparison with persistence,
    reporting per-lead/per-variable standardized RMSE/MAE/bias and MSE skill. Physical-unit
    errors require explicit training scales; combined-variable metrics remain standardized;
    perfect-persistence skill is undefined rather than infinite. Stream hashes, dataset/split
    provenance and a single-checkpoint/no-uncertainty claim boundary are recorded. Seven tests
    include tamper/config-drift refusal and CPU/RTX evaluator parity through the vendor-neutral
    CUDA/ROCm API.
    **Outstanding:** Adam's actual model code/interface, weights, history semantics and declared
    optimiser/schedule have not been supplied or run. Multiple independently trained seeds,
    uncertainty, temporal dependence and correction-family inference belong to T5.4/T5.5.
    T5.3b proves a usable integration/evaluation contract, not the laboratory experiment or
    representation skill.
    **T5.3c Model-semantics adapters -- NOT STARTED:** after T5.0 freezes the actual protocol,
    select and test the required semantic adapter: final-frame autoregressive, full-history
    autoregressive, or direct multi-horizon. Each must declare history use, rollout and physical
    lead-time mapping in provenance. Do not implement all variants speculatively or coerce Adam's
    model into the current Markov adapter; an unsupported protocol must be refused explicitly.
*   **T5.4 Run the boundary-support/domain-size study.** Cross representation with nested
    domains at fixed resolution, dates and central New Zealand evaluation window. Report
    full-domain, common-valid-interior and distance-to-boundary skill plus a boundary
    masking/zeroing ablation. The primary estimand is the transform-by-domain-size interaction,
    not whether one rank happens to swap. Hold architecture, split, optimiser, training budget
    and rollout fixed; report coefficient width/redundancy, parameter count, compute and memory.
    Use independently seeded fits, temporal block uncertainty, effect sizes and a frozen
    transform-by-variable-by-lead correction family. A shrinking, unchanged or reversed db2
    deficit is equally reportable.
*   **T5.5 General representation-controlled benchmark.** Extend the accepted harness to any
    selected multiscale representation and relate held-out forecast results to
    `RepresentationScore`. Treat a null or contradiction as an outcome to diagnose -- score
    misspecification, low power, dataset shift, forecaster interaction and a genuinely absent
    relationship remain distinct explanations.
*   **T5.6 External forecaster adapters -- PARTIAL (T5.6a-g offline contract/cube/truth/evaluator/orchestrator/portable runner/reporting accepted; FCN3 not run).** Add an external
    global model only after its data, weights, licence, grid, variables, normalisation, rollout
    and runtime contracts are frozen. WeatherBench catalogue presence does not mean the current
    regional training path can supply the model or that inference is laptop-feasible.

    **Why FourCastNet 3:** NVIDIA's July 2025 model card declares a 710,867,670-parameter
    probabilistic spherical neural operator over 72 variables on a 721x1440, 0.25-degree global
    grid, with six-hour steps and outputs including `t850`, `q850`, `u850`, `v850` and `z850`.
    Its paper reports calibrated ensembles, stable long rollouts and preserved atmospheric
    spectra. Those are unusually relevant external hypotheses for this workbench: compare an
    FCN3 ensemble mean and distribution with persistence and the laboratory regional model over
    identical NZ variables/dates/leads, then independently analyse error by scale, orientation,
    location and boundary distance. These are **authors'/publisher claims to test**, not
    SpectralEarth findings. FCN3's learned spherical Morlet kernels are not the same intervention
    as inserting Haar/db2/SWT/DTCWT representations into Adam's regional model.

    **T5.6a FCN3/Earth2Studio inference contract -- PARTIAL:**

    1. Define a versioned, hashable `ExternalForecastRun` request for the exact Earth2Studio,
       FCN3 package/checkpoint, 72-channel order, global coordinate grid, normalisation, initial
       condition source, six-hour steps, ensemble size/seeds/noise process, output variables,
       precision and requested device profile. Planning and validation must be network-free.
    2. Keep Earth2Studio/`torch-harmonics` in an optional isolated worker environment. The main
       app exchanges a request and a content-addressed NetCDF/Zarr result; importing or opening
       the workbench must never require CUDA, Earth2Studio, an NGC account or the FCN3 weights.
    3. Run FCN3 on the required **global** state and crop the physical forecast to NZ only after
       inference. Refuse a regional crop presented as an FCN3 initial condition: spherical/global
       operators, channel order and learned normalisation cannot be preserved by relabelling it.
    4. Bind output to checkpoint/config/data hashes and record ensemble member, initialization,
       lead times, units, grid, precision, device/runtime and worker logs. Import through one
       canonical forecast-cube adapter before common evaluation.
    5. **Implemented as T5.6c:** extend held-out evaluation with ensemble mean/member metrics, CRPS, spread-skill ratio and
       rank diagnostics, plus the existing persistence comparison and multiscale error analysis.
       Freeze dates, NZ crop, variables and correction families before comparing with Adam's
       model. Never use FCN3 evaluation years as a fresh test set without accounting for its
       published 1980-2015 train, 2016-2017 test and 2018-2019 evaluation partitions.

    **Portability/acceptance boundary:** the official model card lists Linux on NVIDIA Turing,
    Ampere and Hopper, recommends bf16 AMP, names A100/H100/L40S test hardware and publishes a
    2.65-GB compressed package. It does not publish a minimum VRAM figure or claim AMD support.
    Therefore the RTX 5050 8-GB run is `NOT RUN`, not presumed feasible; AMD GPU inference is
    unsupported until independently demonstrated; CPU-only `torch-harmonics` availability is
    not evidence that full FCN3 inference is practical. The first real acceptance run belongs on
    suitable allocated HPC hardware, while request preparation, result analysis and the rest of
    the app remain laptop-capable. Record peak VRAM/RAM, download size, initialization latency,
    per-step time and numerical repeatability. An unavailable backend is `NOT RUN`, never PASS.

    **Reviewed primary sources (2026-08-22):** [FCN3 paper](https://arxiv.org/abs/2507.12144),
    [official NGC model card](https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/fourcastnet3),
    [Earth2Studio](https://github.com/NVIDIA/earth2studio),
    [Makani research/training code](https://github.com/NVIDIA/makani), and
    [`torch-harmonics`](https://github.com/NVIDIA/torch-harmonics). The checkpoint/model card and
    Earth2Studio are Apache-2.0 according to their published records, but every pinned asset's
    licence and redistribution terms must still be captured at acquisition time.

    **Delivered T5.6a offline boundary:** `src/forecasting/external_fcn3.py` defines schema
    `external-forecast-run/fcn3-v1` with exact model/package/checkpoint/model-card/licence hashes,
    pinned Earth2Studio/PyTorch/torch-harmonics environment identity, the canonical ordered 72
    channels, global 721x1440 grid and coordinate hash, source and normalisation identity, UTC
    initializations, six-hour stochastic ensemble rollout, output selection/crop policy and an
    explicit prepare-only or Linux/NVIDIA-worker profile. It rejects regional inputs, reordered
    or missing channels, external perturbation, ambiguous time, seed mismatch, unsupported
    worker platforms and placeholders. Requests serialize canonically with full SHA-256 and
    refuse overwrite. Completed-worker manifests bind request identity, selected variables,
    global geometry, runtime/precision/hardware, measured wall time/peak RAM/VRAM, worker-log hash
    and deterministic file or directory-tree content hash. NetCDF must be a file and Zarr a tree;
    result/request and artifact tampering are refused. Nineteen executed acceptance cases pass.

    **Delivered T5.6b canonical cube/import boundary:** `src/forecasting/external_cube.py`
    authenticates the sealed artifact before opening it through portable xarray/Dask. NetCDF4
    and Zarr normalize to a canonical per-variable cube over exact `time`, seeded `ensemble`,
    six-hour `lead_time`, global `lat` and `lon` axes. Dimensions, the 721x1440 coordinate values
    and request hash, variable set, floating dtype and explicit canonical SI unit for every field
    are mandatory. A completed-write marker is required. The finite-value gate first refuses any
    decompressed storage chunk above its byte budget, then examines every forecast value one
    chunk at a time. An explicitly supplied `GeographicBounds` creates an inclusive, lazy NZ
    subset only when all endpoints lie exactly on the source grid and use the declared longitude
    convention; it never rounds, interpolates, wraps or invents the experimental domain. Crop
    provenance binds request, result, artifact, validation, bounds and regional coordinates.
    Fifteen executed acceptance cases use compressed synthetic global cubes in both formats; no
    FCN3 prediction is represented.

    **Delivered T5.6c matched-truth ensemble evaluation:**
    `src/forecasting/ensemble_evaluation.py` requires a validated regional forecast, exact
    held-out verifying truth and the corresponding observed initialization. It rechecks lineage,
    dimensions, times, physical lead durations, grid, variable set and SI units and refuses
    interpolation, silent missing-value deletion or training-labelled truth. Bounded lazy source
    tiles feed cosine-latitude-weighted member/ensemble-mean/persistence RMSE, MAE and bias,
    persistence-relative MSE skill, empirical CRPS, population-spread/RMSE and rank diagnostics.
    Ties receive deterministic fractional rank allocation; zero skill/spread denominators are
    `null`; variables with unlike units are never pooled. The content-bound receipt states that
    rank shape is diagnostic and that no uncertainty, independence, significance, calibration,
    generalisation or superiority follows. Fourteen analytic synthetic cases pass; no real
    meteorology is represented.

    **Delivered T5.6d lazy matched-truth builder:** `src/forecasting/matched_truth.py`
    constructs evaluator-ready truth and persistence-initialization cubes from the canonical
    regional ERA5/CDS dataset without loading field values. It derives valid times exactly from
    initialization plus lead, requires every timestamp, grid coordinate, 850-hPa level, source
    alias and SI unit to match, and refuses nearest matching, regridding or conversion. Both
    initialization and valid time must remain inside the declared held-out interval. A fresh
    holdout label requires every sample from 2020 onward; use of FCN3's published 1980-2019
    partitions is retained only under an explicit diagnostic role. Source-manifest/content and
    complete selection identity are hashed into the lazy cubes and receipt. Fourteen synthetic
    acceptance cases pass; no live ERA5 or FCN3 values were matched.

    **Delivered T5.6e reproducible evaluation-run orchestrator:**
    `src/forecasting/evaluation_run.py` freezes regional bounds, held-out interval/role,
    850-hPa selection and chunk/tile budgets in a versioned hashable config. It composes the
    authenticated global artifact, exact regional crop, existing local-only ERA5 cache,
    matched-truth builder and bounded evaluator, closing both stores under success or failure.
    An existing target is refused before inputs open. A successful run atomically publishes one
    canonical no-overwrite JSON receipt containing the complete result and lineage; reload
    verifies outer and nested hashes plus cross-section forecast/source/builder identity. Ten
    synthetic acceptance cases pass, including a real Zarr-to-Zarr end-to-end fixture. This is
    execution evidence only: the fixture is not meteorology and establishes no skill.

    **Delivered T5.6f portable evaluation job runner:**
    `src/forecasting/evaluation_job.py` packages the accepted request, sealed result, ERA5
    `CropSpec` and evaluation config into schema `external-evaluation-job/fcn3-era5-v1` with one
    canonical SHA-256. Machine paths are excluded from that identity and live in a separately
    hashed `external-evaluation-path-bindings/v1` record bound to the job hash, so the identical
    scientific job can use relative laptop paths or shared HPC paths. `create`, `bind`,
    `preflight` and `run` CLI commands require no Python scripting. Preflight authenticates the
    forecast artifact, opens only the existing local ERA5 cache, checks crop identity and
    refuses an existing receipt without downloading data, invoking FCN3 or writing results.
    Run executes T5.6e and verifies the saved receipt corresponds exactly to the portable job.
    Job/binding publication is atomic and no-overwrite. Nine synthetic executed cases cover
    relocation, tampering, missing inputs, overwrite refusal, the complete CLI workflow and a
    real Zarr-to-Zarr evaluation fixture. No scheduler submission is performed: Phase 6 owns
    Slurm/Celery execution, while this slice produces the portable command they will invoke.

    **Delivered T5.6g receipt-backed API/UI reporting:** `evaluation_report.py` applies the
    T5.6e nested and cross-lineage verifier to uploaded receipt bytes, refuses synthetic or
    non-catalogue truth declarations, removes machine paths from its report contract and stores
    admitted JSON under the receipt SHA-256. List/get reverify every file and never surface a
    corrupt one. Three API routes provide upload/list/get. The tenth workbench tab has an explicit
    no-receipt state with no fabricated metrics, then shows exact unit-preserving variable/lead
    scores, persistence, CRPS, spread/skill, diagnostic rank frequencies, sample scope, hashes,
    holdout role and claim boundaries. Integrity and artifact authentication remain separate
    from `scientific_skill: NOT_ESTABLISHED`. An accepted ERA5 URI is a checked source declaration,
    not a digital signature over source values. Acceptance uses synthetic stores with a controlled
    catalogue declaration only: **no display-eligible real receipt exists in the repository or UI
    store**, and the new tab has compiled/built but has not been visually inspected.

    **Still outstanding:** there is no Earth2Studio worker process, environment lock, real model
    card/package/checkpoint hash, NGC access, 72-channel initial condition, forecast execution,
    real canonical artifact, real matched ERA5 truth run, dependence-aware uncertainty or comparison
    with Adam's model. T5.6a proves bytes/declarations, T5.6b proves schema/units/finiteness/crop
    lineage, T5.6c proves metric implementation, T5.6d-e prove exact synthetic matching and
    reproducible orchestration, T5.6f proves portable offline invocation, and T5.6g proves
    verified evidence presentation; none proves
    meteorological accuracy, calibration, spectral fidelity or real-data skill. T5.6 remains
    partial and every hardware run remains `NOT RUN`.
*   **T5.7 Sensitivity and error-field analysis.** Perturb initial fields through the existing
    `PerturbationEngine`, then analyse forecast errors by scale, orientation, location, lead and
    boundary distance rather than as a single scalar.
*   **T5.8 Irregular observations and observation-conditioned spatial downscaling -- NOT
    STARTED.** Extend the gridded forecast workbench with generic contracts for models that map
    coarse gridded context, irregular observations and high-resolution covariates to a
    deterministic or probabilistic field. This is a separate scientific task from temporal
    forecasting: the existing ERA5/CDS cache and NetCDF/Zarr readers cover part of the gridded
    input path, but they do **not** currently represent station observations, point-context /
    point-target roles, spatial holdouts, target-grid refinement or distribution-valued output.

    **T5.8a Canonical point-observation contract:** define a versioned, hashable
    `ObservationSet` carrying UTC time, stable station/source identity, latitude, longitude,
    optional elevation, variable identity, physical unit, measured value, quality-control
    state, missingness reason and source/content provenance. Require an explicit coordinate
    reference system and longitude convention; reject duplicate observation keys, ambiguous
    units, non-finite accepted values and silently discarded QC failures. Storage may be
    Parquet/Arrow or another measured streaming format, but the scientific contract must not
    depend on a particular table library.

    **T5.8b Conditional-field sample and model seams:** define explicit roles for coarse gridded
    context, irregular point context, withheld point targets, static or time-varying
    high-resolution covariates, land/validity masks and the requested target coordinates or
    grid. Record native and target resolution, grid alignment, CRS, interpolation/regridding
    operator and boundary policy. Add a generic `ConditionalFieldModel` adapter whose output is
    a physical field, samples, or declared distribution parameters; do not make DeepSensor a
    core dependency or encode one library's internal task object as the platform contract.

    **T5.8c Leakage-safe temporal and spatial evaluation:** compose the existing temporal
    embargo with frozen station/region holdouts. Fit every normalisation, transformation,
    imputation and station-selection rule on training data only. A point held out as a target
    must not reappear as context through another table, cached task or collocated identifier.
    Persist the split seed and exact station/time membership. Report interpolation to seen
    locations, prediction at unseen locations and transfer to held-out regions as different
    estimands rather than pooling them.

    **T5.8d Probabilistic verification and baselines:** retain the declared likelihood or sample
    semantics through export. Evaluate physical-unit bias/MAE/RMSE alongside appropriate proper
    scores such as CRPS or log score where mathematically defined, coverage and sharpness, over
    identical held-out observations. Compare against declared coarse-grid interpolation and
    climatology/persistence baselines. Stratify results by variable, location, elevation,
    season, observation density and distance to the nearest context station; predeclare
    correction families and dependence-aware uncertainty before inferential claims.

    **T5.8e Portable adapters, UI and acceptance:** accept canonical xarray NetCDF/Zarr gridded
    inputs plus the canonical point-observation table on laptop and HPC paths. Prove the generic
    seam with synthetic irregular stations, a deliberately withheld station/region and at least
    one minimal external-library compatibility fixture. The UI must show context versus target
    points, native versus target grid, masks, units, QC exclusions, uncertainty semantics,
    split identity and provenance; it must not display a smooth high-resolution map without
    also exposing observation support and validation coverage.

    **Acceptance boundary:** round-trip every contract without semantic loss; detect shuffled
    coordinates, unit/CRS errors, point-target leakage, train/test normalisation leakage and
    distribution-parameter mislabelling; demonstrate bounded-memory loading; and reproduce a
    synthetic known answer against independent metric oracles. Only then add a real DeepSensor,
    ConvNP or other downscaler adapter. Compatibility does not establish forecast/downscaling
    skill, observational representativeness or operational readiness.

    **External interface case reviewed 2026-08-22:** Emily O'Riordan's
    [deepsensorNZ](https://github.com/oriordanemily/deepsensorNZ) at commit
    `68dd21f7aab5c375dfbe34637cd469ba4592243a` motivated this gap: it combines ERA5/WRF gridded
    context, station context/targets, high-resolution auxiliary fields and probabilistic ConvNP
    predictions. It is an interoperability reference, not copied code, evidence for the wavelet
    experiment, or proof that T5.8 exists. No repository-root licence file was observed during
    that review, so no source reuse or vendoring is planned without explicit permission and
    licence clarification.

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
