# SpectralEarth: Roadmap to an Automated Multiscale Discovery Engine
*Strategic engineering plan. Companion to `architecture.md`, which records what exists today.*

**Research question this platform is being built to answer:**

> At what locations, scales and times does predictively useful physical information exist in an atmospheric field, how does structure transform across scale, and which representation of the field carries that information best?

Everything below either serves that question or gets cut.

---

### External research alignment and the claim boundary

The supplied EGU poster *Spectral representations for regional AI-based weather prediction* (author
O'Riordan (Victoria University of Wellington) is relevant external context, not a result of
this project. It asks whether Fourier, DCT, Haar, db2 and DTCWT representations change forecast
skill in a controlled lightweight neural model over the New Zealand ERA5 domain at 850 hPa.
Its initial results motivate this roadmap's boundary, transform and downstream-forecast work.

SpectralEarth currently supplies much of the **measurement apparatus** needed around that
question: the compared 2D transforms, physical grid metadata, boundary diagnostics,
scale/orientation summaries, surrogate nulls, corrected inference, provenance and a verified
single-variable regional Zarr crop path. T5.1a-e now supplies the batched transform API and
T5.2a-b supplies the leakage-safe five-variable PyTorch dataset constructor plus bounded-memory,
worker-safe local Zarr access over a materialised crop. The viable multi-year real NZ crop and the
independent ERA5 route cross-check have since been supplied — T4C.5m acquired the 8,764-frame
record, closed D43 and returned a T4C.6 PASS, and T4C.5n audited a second window. It does
**not** supply the poster's learned forecasting experiment. No current task has
implemented its neural architecture, autoregressive training schedule or matched forecast
comparison. Phase 5's remaining model work is a proposal to integrate that judge, not evidence
that the laboratory architecture or experiment exists inside SpectralEarth. T5.3a-b supplies
the generic adapter, persistence/smoke baselines, verified artifact contract and held-out
evaluator described below; no external model or weights have been supplied or executed.

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

Verified against the code on 2026-09-02. Every claim here is backed by captured output in
`VERIFICATION.md`; `architecture.md` Section 7 holds the full defect ledger (D1-D102, of which **94 fixed, 1 partial (D18), 7 open (D84, D85, D96, D97, D98, D99, D100)**). **`PLAN.md` is where the remaining work is ordered**; this document is the task history and the evidence.

The numbers in this table are checked by `src/tests/test_documentation.py`, which parses them
out of this file and compares them against the source. That guard exists because this table
had itself gone stale — it claimed 271 passing tests and "DTCWT still not implemented as
advertised" several slices after both had changed. A status section that cannot fail is not a
status section, it is a memory.

| Area | Real status |
|---|---|
| **Phase progress** | **Phase 3.5 complete** (25 tasks). **Phase 4A, 4B and the pre-gate 4C instrument are complete:** T4A.1-4, T4B.1-4, T4C.1-5h. T4C.5d-h make the real gate, independent-source check, two-stage acquisition, physical preflight and primary preregistration bounded and content-bound, but **T4C.6 has now run and returned PASS on the real record (T4C.5m)**. **T5.0 is partial:** T5.0a-b supplies the strict versioned/hashable protocol and runtime-binding gate, but the actual laboratory evidence/config has not been supplied or frozen. **T5.1 is partial:** T5.1a-e accepts raw/FFT/DCT/Haar/db2/SWT/DTCWT training representations, optimized and exposed with truthful UI readiness; mixed precision and remaining cross-device acceptance remain. **T5.2 is partial:** T5.2a-d implements the aligned, leakage-safe, train-normalised PyTorch dataset, bounded-memory worker-safe cache, bounded blockwise CDS cache publication, exact calendar splits and verified physical-time leads; the live CDS run, independent overlap and multi-year NZ crop are all now done, D43 having been closed by T4C.5m. **T5.3 is partial:** T5.3a-b supplies the forecaster seam, persistence baseline, deterministic represented smoke run, verified model-artifact contract and persistence-relative evaluator; T5.3c and the actual laboratory model remain open. **T5.6 is partial:** T5.6a-g supplies the offline FCN3 identity, lazy canonical cube/import boundary, exact ERA5 truth bridge, matched ensemble metric engine, atomic reproducible run receipt, portable laptop/HPC runner and verified API/UI reporting; no worker, dependency, checkpoint, initial condition, real forecast or real truth evaluation has run, so the result UI is honestly empty. **T4C.5i is complete (steps 1-8):** the derived spatial-power quantities, the demoted crop constants, the receipt that publishes the derivation, the FAIL/INVALID boundary that refuses to record an inadequately powered absence as a negative finding, and the checked supersession retiring the frozen campaign for a six-year successor that resolves its own declared family. **No real gate has run**, so nothing has been adjudicated by the new boundary and no data has been acquired for either campaign. Its step 5 found D85 -- the frozen campaign's confirmatory partition cannot resolve its own declared family, so T4C.6 cannot PASS as frozen. **T4C.5j is complete:** the T4C gate record is served read-only over seven GET routes and rendered as a workflow destination, so a retired design, its checked retirement and the FAIL/INVALID boundary are readable without opening a file; it adds no science and no receipt has ever been served, because none exists. **T4C.5k is complete and T4C.5m closes D43:** the mandatory acquisition order has now run end to end. Steps 1-3 gave a recorded store probe, the eight-frame WeatherBench overlap, the first live CDS request in the programme's history, and a cross-route check judged in units of the primary route's own GRIB packing step rather than in Kelvin; that check stopped the campaign on its first attempt and found D86 -- the declared tolerance was finer than the CDS route can express, and the rule authorising the spend was not part of the frozen design -- both halves of which are fixed, with campaign v3 freezing the criterion. Steps 4-6 then ran: **the 8,764-frame record was acquired** (72 monthly shards, 338.905 MB, 5,853.7 s, `content_key a07c23ec89f953c1`), the full-cache overlap passed at 0.71875 of a packing step, and **T4C.6 returned PASS** in 753.9 s with ten links replicated in train and test. Step 6 found D87 -- the gate could not admit a record its own campaign had legitimately verified, because the admission path read the absolute criterion's manifest fields and the reader carried only those three names -- fixed by naming the criterion and by running the gate through `run_campaign_gate`, so the rule reaches it from the frozen envelope rather than from the caller. **D84 and D85 remain open and did not gate this result**; they govern whether an absence was detectable, and a PASS does not route through them. **T4C.5n** then audited the record's interior against a second WeatherBench window at 2021-07-01/02, passing at 0.46875 of a packing step; it authorises nothing by construction, because the campaign names exactly one overlap window and this is not it. Two windows out of 8,764 frames are independently verified; the rest is guaranteed structurally rather than against a second archive. **T4D.1 and T4D.2 are DONE**: coefficient maxima are located, sub-pixel, threshold-frozen and R13-masked, and they are linked frame to frame through the existing TG2.3 tracker rather than through a second one. T4D.2's acceptance test found **D88** -- every coefficient's parent-grid position was displaced by half its filter's accumulated support, differently at every level, so no cross-scale association had ever been on a common frame; no existing result changes, because everything before this collapsed a band to a scalar and a circular shift moves no mass. **T4D.3 is DONE**: tracks are rendered as sentences that name the coefficient maximum as their subject, refuse a compass word to a grid that declares no orientation, report energy as the square of magnitude under its own name, and put scale growth in the population -- on the vortex, level 4 weakening by 32% while level 5 strengthens by 18% and is first excited nine frames later -- rather than in any one track. It found **D89**: the programme's one guard against causal vocabulary passed over `co2_causes_warming`, an underscore being a word character, which is the shape author-supplied text most often arrives in; no existing document changes, because what was wrong was the guard's reach and not an output. **T4E.1 is DONE**: a tracking pass is read as TG3.3's attributed graphs rather than through a second graph type -- the co-present tracks of each searched frame enumerated as pairs and triples under refusing budgets, the comparable half left to `constellation()` and the track-derived half (onset, age, local velocity, d(strength)/dt) carried beside it in this record's own units. Three of TG3.3's eight relations are measurable on 4D features and five refuse by name. It found **D90**: `distance`, the one relation carrying geometry, refused on every 4D feature because the position unit was spelled `cells` and the scale unit `parent-grid px` -- two names for one unit on an undecimated bank -- and the failure was silent, since a refusal is recorded on the graph rather than raised; no existing result changes, because no constellation had ever been built from these features. **T4E.2 is DONE**: invariance is a toggle between two matchers TG3.4 had already built and already measured, not a third normalisation -- 135 constellations signed scale-specific, 48 scale-invariant, and the 87 pairs lost by turning the universality hook on are the price of the only mode whose every entry is dimensionless. Translation, rotation, reflection and relabelling leave the signature identical to floating-point precision. It found that `planted_configuration`, the benchmark this programme supplies for invariance, is *equilateral* and therefore has no principal axis: over 24 noise realisations its anisotropy stayed under 1.05 while the axis angle scattered across the half-circle, so the bearings the specification asks for are refused there by a floor that is a measurement rather than a choice. **T4E.3 is DONE**: declared-weight, correspondence-minimised distance is calibrated from same-configuration replicates and clustered by deterministic complete link into centroids with measured radii; location, rotation and 10% jitter join, while absolute rescaling joins only in the scale-invariant mode. It found and fixed D92, where small noise changed the exact canonical order and attached node attributes to the wrong vertices during approximate comparison. **T4E.4 is DONE**: distinct constellation identities supply support, the decreasing-support scan prunes its below-threshold tail, every candidate retains its count, and candidate/time overruns refuse the whole sweep. No null or significance is claimed. **T4F.1 is DONE**: the timed event substrate Phase 4F needs, and the three things it refuses to assume -- the unit of the clock, that an occurrence sits on a searched frame, and that a gap between occurrences was observed rather than merely unread. Simultaneous events are published unordered, because a succession taken from list position would be the one arrow this phase exists not to fabricate. It found and fixed **D93** -- T4E.2 dropped the `time_units` its constellation carried, and nothing had caught it because comparison, clustering and support counting are none of them dimensional. **T4F.2 is DONE**: the chain the substrate was built to express is counted, with a declared transition window whose minimum lag is strictly positive so simultaneity can never become a step, and with a confidence whose denominator admits only antecedents whose window was wholly searched -- an occurrence the record ended before is censored rather than counted unfollowed, which is the asymmetry that would otherwise price the tail of every record as a failure to be followed. Support is proved antimonotone under extension and the scan prunes on it. Recurrence is a repeated gap on the cadence lattice and is published with the number of interval values the record could hold, because a repeat among few of them is expected under no structure at all. No null is drawn and neither receipt claims significance. **T4F.3 is DONE**: the first null this phase draws. A confidence is referenced to a base rate measured as a window probability over the searched positions rather than to a per-frame rate that would make lift a fact about window width; to a surrogate ensemble that rotates the antecedent on the lattice, preserving its count and its own bursting and destroying only the alignment under test; and to a correction paid on the whole declared family, with the lag choice tested against the distribution of the maximum because a lag chosen by the data is a search. The anti-conservative null is provided and its cost measured: on one unchanged record the same lift of 5.25 is not distinguished from the shifting null at p = 0.11 and is called a precursor by the scattering null at p = 0.01, so the choice of null decides the finding. A design that could not have rejected anything is refused before any counting happens, so an under-powered absence is never produced to be misread. Every rule carries R9's six figures through the programme's own carrier or none of them, publishes the ensemble its p-value came from, and claims a precursor signature and nothing above it (R7). **T4F.4 is DONE:** the two directions the phase names are one selection over the table T4F.3 already corrected, and the point of that is what it refuses. A query is a view and not a second test -- no p-value, q-value or status is recomputed, the declared family is published beside every answer, and the shortcut of re-running the inference for one target is priced rather than warned about: the same rule's q-value falls from 0.0417 to 0.0050 when the family is narrowed after the fact. Coarse and fine are read off the catalogue's own member scales as an interval order, so patterns whose scale ranges overlap or touch are not orderable and are withheld and counted rather than sorted, and under the scale-invariant mode the ordering is refused outright because that mode makes every pattern's scale statistic exactly 1.0. Every ranking key is available with its rank on every entry: ranked by how often it preceded the target -- this task's own wording -- the leading answer is a pattern the null did not distinguish, and ranked by the corrected p-value it is the planted precursor, so the choice of ranking decides the answer. Nothing distinguished, nothing orderable and nothing measurable are kept apart as three different empties. **T4F.5 is DONE:** a rule is put back on the parent grid as a *footprint* -- the cells inside the transform's own filter support, taken from the same number R13 cuts the contaminated margin with -- because a detail coefficient peaks on a flank rather than at a centre, and on the planted vortex the flank sits 1.08 structure widths out at level 4 and 1.26 at level 5, so **the peak cell is not the planted cell in a single one of twenty-four frames**. The footprint recovers it, and only while the level that found the structure can still reach back to it: level 4 holds the planted cell to frame 9 and loses it in all fifteen frames after, level 5 holds it throughout. The intersection of two members is not a location either -- one level's LH/HL pair contains the planted cell in 11 of 11 occurrences while one orientation's two levels exclude it in 11 of 11, because those flanks point the same way -- and the receipt warns for the second case. Each grid says only what it can (degrees, or metres named as a distance from the crop's origin, or cells), an undeclared level is a number rather than hectopascals, a frame is dated only where the record carries a calendar, and the field's values and anomalies are read from supplied records rather than derived from each other. The historical instances come from `anchor_verdicts`, the same per-anchor decision `count_sequence` tallies, and every enumerated total is reconciled against the figures the rule published before anything is shown. **Its second acceptance clause is outstanding**: a real-ERA5 `recognised` pattern projecting onto a physically sensible footprint needs T4F.6, which does not exist. **T4F.6 is PARTIAL:** R10's physical gate exists, discriminates, and has not been run. A
mined pattern is measured off the footprints T4F.5 drew, converted into kilometres and hours
through the record's own metric and calendar, and labelled against a declared catalogue of known
phenomena whose every entry carries a citation and whose content is hashed. All three verdicts
are reached on the same real patterns -- PASS against envelopes that fit, FAIL against envelopes
that do not with R10's presumption that the pipeline is broken stated in the receipt, and
INVALID against a catalogue so wide it excluded nothing, which licenses nothing and is not a
FAIL. The ranking key decides the verdict, so it is hashed into the declaration along with the
top-N, the catalogue digest and a required naming of the documented event; a catalogue is a
draft until a maintainer signs it and the gate refuses to adjudicate under a draft, the shipped
southern-ocean reference included, because its envelopes are the implementer's first reading and
not values quoted from any paper. The measurement that bounds the whole exercise comes from the
record: on the T4C.5k crop a cell is 27.80 km meridionally and 13.90 to 26.12 km zonally, so
sixteen cells is 222.4 to 444.8 km and an envelope narrower than that cannot exclude anything.
An entry needing a field the record lacks is unassessable rather than unrecognised, permanently
so for the upper-level entry on a single-level record. **The acceptance has not been run**: it
needs a maintainer-frozen catalogue, a declared documented event, and a real mining pass over
the 8,764-frame record that has never been performed, so **Phase 4G is still gated** and T4F.5's
second acceptance clause is still outstanding. **T4F.7 is DONE:** R14's cross-region re-test, on a record built to give four different
answers at once. Five declared boxes of one field: the rule holds in two held-out regions, does
not hold in a third that carries both patterns in the wrong order, and is not assessable in a
fourth that carries nothing -- and the verdict is `regional` because of the third, or `general`
under a design that declares three held-out boxes rather than four. A region that never carried
the antecedent did not fail the test, and neither did one that carried it but never the
consequent; rendering either as locality would turn missing data into a finding. A held-out
region may supply occurrences but not help define a pattern, so identities are matched into at
T4E.3's own calibrated radius and leakage refuses `general`; measuring that found that T4E.2's
signature cannot tell two band orientations apart at all, being invariant to rotation by
construction. Membership is decided on T4F.5's footprints rather than on a maximum, with the
three ways of not being inside a box counted apart. The gaps between boxes are published in
cells and kilometres and `general` is refused while their independence is unestablished, and
physiography is declared with a source rather than derived from a field that carries no
coastline. It has been exercised on a synthetic record only. **T4F.8 is DONE:** the platform's existing follow-up proposers are optimisers -- they propose the parameter range or category that made the metric better, so no outcome would retract the finding that prompted them -- and this task adds the refutable kind: a re-test on ground the finding was not made on, carrying a prediction digested before the record is read and a named condition that would retract it, refused outright when that condition is one no outcome could satisfy. A rule that did not clear its null gets a power proposal instead, which carries no prediction and can confirm nothing, and is refused when the study was already big enough to detect the effect. The occurrence count a design needs is computed from quantities the record can be read for without counting the pair. No proposal has been run. **Not done:** the real-ERA5 gate
*review*; the T4F.6 gate run itself, 4G and 4H; T5.4-5, the remaining T5.6 inference/real-data work, T5.7, and the T5.8 irregular-observation/spatial-downscaling track. Per-task evidence blocks sit under each task below; a task without a **DONE** or **PARTIAL** label has not been started. |
| **Phase chronology correction (2026-09-03)** | The pre-run sentences embedded in the long phase-progress history are superseded by the later evidence in that same row: campaign v3 acquired the complete 8,764-frame record and T4C.6 returned PASS. A PASS does not exercise the FAIL/INVALID absence adjudication, so D84/D85 remain relevant only to a future negative result; they did not block or invalidate the recorded PASS. The v3 receipt is present and served by the read-only gate record. |
| Ownership / licence | **Declared in `LICENSE.md`.** Edward Jonathan Bentley retains the proprietary SpectralEarth core. A designated Named Licensee may be granted a perpetual, worldwide, royalty-free right of lawful personal, academic, research and commercial use/modification, without public redistribution or sublicensing of the core; no designation is recorded in this repository. Independent extensions and upstream contributions remain separately governed. This bespoke text has not been professionally reviewed. |
| **Accessibility** | **Workflow-wide source contract, TG11.6 DONE.** Skip and route focus, globally visible focus, bound legacy labels, reduced motion, announced asynchronous state, keyboard SVG lineage and figure text equivalents now cover both platform lines. Rendered assistive-technology inspection remains NOT RUN, so no WCAG conformance level is claimed (see `roadmap_cross_domain.md`). |
| Backend test suite | **5038 passed, 1 xfailed** and **7 FAILED** (plus 4 skipped). Measured 2026-09-12 in 4,233.33 s (1:10:33) on the tree carrying T4E.34 and the section-0 truth-up. Six are the documented benchmark/API, browser-evidence and `live_sources` gate failures. The seventh was a Windows `os.replace` permission failure while publishing a temporary Zarr directory; the exact test passed alone immediately afterwards in 38.05 s, so it is recorded as non-reproduced. See `architecture.md` section 7.1. |
| Ground-Truth Benchmark Suite | **29 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE.** Twenty datasets with declared known answers, twelve of them nulls. CI-ready via `python -m src.benchmarks` (exit 0). |
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
| React frontend | **Workflow surfaces are wired to the backend and no longer able to fabricate a result.** Export in CSV/JSON/NetCDF4/Zarr/PNG/SVG with provenance embedded in the file; units, spectral convention and slope uncertainty displayed; simulated data labelled where it is used. The domain-general line now reaches acquisition, analysis, preregistration, evidence, mining, cross-domain confirmation, findings and recorded review. `tsc` is clean and the current build emits 1,395 modules. Contract tests assert every fetched path is served, the review client is GET-only, and claim/commentary boundaries survive the layout. **The then-nine legacy modules were rendered in a browser and confirmed working by the user on 2026-08-20** (T3.5.25); no screenshots were captured, so that is a user report rather than repository evidence. The later workflow panels compile and build but have not been visually inspected. |
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
only the T5.1a-e slice has CPU/CUDA parity evidence and ROCm/MPS are unmeasured. Phase 4A-4C.6
are complete through the recorded real-data PASS; a separate human review of that receipt
remains. T4D.1-3, T4E.1-4 and T4F.1-5 are complete and T4F.6 is partial -- the physical gate is built and discriminates, but it has never been run, so 4G is still gated. T4F.7 and T4F.8 are complete. **Current status for every task above lives in `architecture.md` section 0, which is the
authority; this document is history and does not restate it.** `PLAN.md` orders what remains.
See Section 4 below for per-task evidence.

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

## 2b. The cross-domain line — this document is not the whole programme

The phase map above covers the **gridded physical-field line** only. A second line has run on
branch `ed-dev` since the `master` freeze at 1,117 tests and is now more than half the suite:
declared domains and their violated assumptions, the claim ladder, the adversarial review layer,
bounded translation into domain wording, the onboarding contract and the ingestion seam.

**It is tracked in `roadmap_cross_domain.md`**, which is the primary planning document for it.
Phases G0–G11 are complete and TG12.1/12.1a/12.1b/12.1c/12.1d delivered the gridded-ocean
source, its truthful acquisition boundary, portable immutable receipts and transform-derived
minimum-crop planning before transfer. TG12.2a closes D69 with an
explicit per-sample presence contract and refuses asynchronous frame-lag inference; TG12.2b-d,
Argo acquisition/reduction and G13 remain unstarted. `architecture.md` remains the source of
truth for what is implemented. Anyone picking this project up should read both documents,
because nothing in this one would reveal the detail of the other line.

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

Corollary, learned the hard way in this task: when a measurement contradicts the claim written in the docstring, the claim is what changes. The rule numbering continues at **R17** in `roadmap_cross_domain.md`, which holds R17-R26; R25 (an unreachable refusal is a defect) and R26 (an operating point is calibrated against replicates, not chosen) were both learned on this atmospheric line in Phases 4D-4E and apply here in full. Two such reversals are recorded in `architecture.md` section 7.2e.

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
Add `alembic`, `scipy`, `ipywidgets`, `plotly`, `matplotlib`, and the estimator dependencies required by E8 (`scikit-learn` for k-NN neighbour queries in the KSG estimator, `networkx` for the Phase 4E attributed graphs -- **not adopted at T4E.1**, which builds on TG3.3's `AttributedGraph`: at two and three nodes the graphs are complete and there is no graph algorithm to run, so the dependency would buy a data class; revisit against a use in T4E.3 or 4F); add `PyWavelets` and `dtcwt` as **test-only** oracles. Reconcile `architecture.md`, `README.md`, `data/README.md`.
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
    `minimum_crop_size` returns 324 and 532 for four and five levels (512 and 1024 until
    T4C.5i step 6 removed the power-of-two rounding from the refusal path; `dyadic_crop_size`
    still reports the rounded size, and nothing is refused on it).
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

### UI north star - inspectability is the deliverable, not a disclosure panel *(standing constraint, not a task)*

**The rule.** *Every conclusion must be traceable back to the exact data, assumptions, mathematics
and admissibility checks that produced it.* No screen may present a result in a form that cannot
be expanded into the evidence chain that produced it: the record and its content key, the
representation and its configuration, the declared family and its size, the null and the
correction, the split, the power and resolution limits, what was refused, what was assumed, and
what the result explicitly does **not** establish.

This is not a usability preference. The roadmap's own bet is that the symbolic, auditable path
"is what makes SpectralEarth a research instrument rather than another opaque predictor" - and a
UI that surfaces the number while hiding the contract converts the instrument back into the
predictor without changing a line of the backend. The people who can validate the whole stack are
a small, mixed group - statisticians, signal-processing people, domain scientists, ML researchers
- and no single reviewer owns all of it, which makes surfacing more important rather than less.

Three constraints follow, and each exists because the obvious design violates it:

*   **Refusals and unmeasured states carry the same visual rank as values.** Nearly every result
    this programme produces is qualified by construction: an onset that is left-censored is a
    *lower bound*; a separation between two bands is between two flanks and not two structures; a
    strength ratio across bands is a ratio of filter gains until it is normalised; a relation that
    could not be measured is refused **by name**; a gate whose stage does not exist reports
    `NOT_YET_RUNNABLE`, never a skip. The natural instinct is to show the value and put the
    qualification behind a disclosure control - which inverts the epistemics exactly, because in
    this system the qualification is the load-bearing half. "9-frame precursor lag" and ">=9
    frames, one record, no null, synthetic data" are different claims, and only the second is
    true.
*   **An empty receipt must say which kind of empty it is.** Provenance depth is genuinely uneven
    across the tree: some paths are content-addressed end to end, and some receipts have never
    been served because nothing has produced one. "No run exists" and "the client did not fetch
    it" must be visibly different states. A spinner that resolves to blank is an audit failure,
    not a loading edge case.
*   **A generated summary is a rendering of a receipt, never a substitute for one.** LLM review is
    deliberately outside reproducible computation and cannot promote a claim (R7, and the G-line
    review boundary). The same context that makes the model useful is what makes the system
    inspectable - a real convergence - but it runs backwards the moment a summary occupies the
    place the receipt should. Every summary must be one gesture from the artefact it summarises,
    and must carry its entitlement in the same view rather than in a footnote, which is the rule
    `spectral_narrative` already applies by welding the entitlement into the claim string.

**Status: not scheduled, not started.** This is recorded so it is not designed away one friendly
summary at a time. It is a constraint on 4E-4H's UI work and on any future presentation slice, and
it should be turned into acceptance criteria at the point a task is written against it - not
retrofitted afterwards. Nothing in it is claimed to be met today; T3.5.0's screenshot-per-tab
criterion is still open, so no assertion about what a reviewer can actually see in a browser has
been verified by looking.
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

**T4C.5i Derived spatial power and the FAIL/INVALID boundary *(fixes D84)* - PLANNED.** The
frozen 161x161 crop passes its own preregistered geometry audit and is then refused by the
acquisition layer. Both layers are computing R13 correctly and disagreeing about a constant, which
is the signal that the constant is doing work it was never derived to do.

The arithmetic, verified: db2 SWT level 3 has an accumulated support of 22 px, so 11 px per side
are contaminated and a 161 px crop retains a 139 px valid interior. `MIN_VALID_INTERIOR` is 128,
so 139 passes -- and `gate_campaign` checks exactly this, which is why `review` and `preflight`
both returned clean. `zarr_source.minimum_crop_size` then takes the same 128, adds the 22 px of
support to get a raw minimum of 150, and **rounds up to the next power of two**, giving 256. The
161 px crop is refused against that 256, and the refusal text calls it *statistically recommended*.

Neither number is a derivation. `MIN_VALID_INTERIOR`'s own comment says so -- *"this is a
judgement"* -- and R13 calls it "the declared 128-pixel minimum". The rounding is justified in
`minimum_crop_size` as *"because the dyadic transforms want one and because it matches how a
researcher thinks about crop sizes"*: ergonomics and dyadic tidiness. SWT is undecimated and has no
dyadic size requirement at all. So a preregistered study is currently blocked by a rounding step
with no statistical standing, described to the caller as though it had one.

**What the estimator actually consumes.** `transfer_entropy` takes 1-D sequences. Space is
collapsed before it runs: `energy_density[t, s] = sum(coefficient**2) / values.size` over the valid
interior -- one scalar per frame per scale. The joint histogram's samples are therefore **frames,
not pixels**: `cells = bins**3 = 216`, giving 4,382/216 = 20.3 samples per cell on train and
2,914/216 = 13.5 on test, both above `MIN_SAMPLES_PER_CELL` of 5.0. Temporal power is comfortable
and is already checked.

**So crop size beyond edge exclusion is a power question, not a validity question.** The valid
interior does not supply samples; it sets the *precision* of the per-frame scalar. Fewer
effectively-independent spatial structures means a noisier energy density, and measurement noise in
source and target attenuates estimated MI and TE toward zero. The direction matters: an undersized
crop **biases toward the null**. It cannot manufacture a false PASS by this route. It can
manufacture a FAIL that is really *"the instrument was too noisy to see it"*. Edge contamination
can manufacture a false positive, but that is R13 requirement (1), which is rigorous, separately
enforced and unchanged by this task.

The frozen decision rule already carries the right vocabulary -- *"an adequately powered absence is
FAIL; any contract, source, geometry, overlap or power failure is INVALID"* -- but nothing in the
system derives the spatial-precision term that separates those two verdicts. That is the hole, and
it is not the crop constant.

**Why this blocks acquisition rather than following it.** A negative T4C.6 is an explicitly
permitted deliverable of this programme. A negative result with no power analysis is not
publishable: the first question a reviewer asks is whether the instrument could have detected the
effect being declared absent. Until the attenuation is measured, neither a FAIL nor a PASS can be
defended, and the crop that 2.5 GB would be committed to has not been shown adequate for the
frozen family.

**The work:**

1.  **Spatial decorrelation length per scale**, estimated from the **train partition only** -- the
    direct analogue of `decorrelation_frames`, which already takes the first lag at which the
    autocorrelation falls below 1/e, applied on the latitude and longitude axes instead of time.
2.  **Effective spatial sample size** per scale: valid interior area divided by decorrelation area.
    Raw pixel counts are never treated as independent, for the same reason a global shuffle is
    never treated as a null.
3.  **Attenuation measured, not modelled.** Recompute the transfer entropy over concentric
    sub-crops of the same interior, holding frames, bins and lag fixed so only spatial precision
    varies, and extrapolate to an unlimited crop. *Changed during implementation:* this step was
    first specified as an analytic standard-error and noise-to-signal term. That was dropped
    because the signal/sampling-noise split of the across-frame variance cannot be verified from
    the data, and an unverifiable correction to a power claim is worse than none.
4.  **Minimum detectable effect** at the declared 36-test family, BY correction at 0.05 and
    4,999 surrogates, read off the measured ensemble rather than assumed. `detection_rank`
    reduces the design to the largest exceedance count `k` that still clears the rank-1 corrected
    level -- `k = 0` for this campaign, so the observation must beat every surrogate -- and
    `minimum_detectable_effect` returns the `k + 1`-th largest value the ensemble produced. A
    design that cannot reach the level at any `k` is reported as having no detection at all
    rather than a very large threshold. *Changed during implementation:* no confidence interval
    is attached. The surrogate seed is preregistered, so the ensemble is frozen and the order
    statistic *is* the gate's decision boundary rather than an estimate of one; the two intervals
    attempted before that was recognised are recorded in the Progress block below.
5.  **Refuse on the derived quantity.** Inadequate power returns `INVALID`, names the deficit, and
    states which of crop size, frame count or scale count would close it -- never a bare constant.
    `spatial_power_refusal` combines the derived quantities into one verdict; `crop_for_effect`
    inverts the reported attenuation fit to name the interior that would close an attenuation
    deficit, and refuses to name one where a number would be an invention; `family_for_effect`
    reports the family that would have detected the effect and marks it inadmissible after the
    data are in. *Found while implementing:* `frames_for_resolution` exposed **D85** -- surrogates
    are drawn with replacement, so the *distinct admissible shifts* in a record, not the number of
    draws, bound the attainable p-value, and the frozen campaign's 2,914-frame confirmatory
    partition supplies 2,912 against the 3,005 the declared family needs. The campaign therefore
    cannot replicate at any effect size, and `check_power` reports it as adequately powered because
    it counts draws. The resolution half of the refusal is wired: `review_gate_campaign` reports
    it, `preflight_gate_campaign` refuses on it before any transfer, and `preflight_cached_gate`
    refuses on it for the real gate role. The attenuation half needs the sub-cropped curve, whose
    computation is the same work step 7's receipt fields need, and is built with them.
6.  **Demote both constants.** `MIN_VALID_INTERIOR` and the power-of-two size become *reported
    recommendations*, explicitly labelled as heuristics. The power-of-two rounding is removed from
    the refusal path entirely and its "statistically recommended" wording corrected.
    **Done.** Both constants stay at 128 and stay reported -- an unreported judgement is a number
    in someone's head -- but each now declares in code and in every payload that it is a judgement
    about uncontaminated span rather than a derived power criterion, and names
    `analysis_engine/spatial_power.py` as the thing that answers the question it stood in for.
    `minimum_crop_size` returns the requirement (324 px at four levels, 532 at five) and the new
    `dyadic_crop_size` reports the round-up separately; `assess_shape`'s threshold is the
    alignment-respecting requirement (352 cells for DTCWT level 4) with
    `dyadic_operational_shape` beside it. The refusal reads *R13 heuristic interior*.
    *Consequence:* **D84's contradiction is closed.** The frozen 161 px db2 level-3 crop needs
    150 px, not 256, so `gate_campaign` and the planner now agree about it -- pinned as a test
    against the defect's own case. Whether 139 px of interior is *enough* is still unanswered in
    the running gate, so D84 remains open on the adjudication rather than the disagreement.
7.  **Publish the derivation.** Decorrelation lengths, ESS, attenuation, minimum detectable effect
    and the FAIL/INVALID boundary all enter the receipt, so a reviewer audits the power claim
    rather than trusting a judgement.
    **Done**, and with it step 5's deferred attenuation half, which needed exactly this curve.
    `run_cached_gate` takes one further bounded pass over the train partition and publishes a
    `spatial_power` block: per-scale median decorrelation length and effective sample count,
    the attenuation curve over concentric sub-crops, the minimum detectable effect, the
    `spatial_power_refusal` record with its remedy axes, and a `power_adjudication` block naming
    the rule that produced the scientific verdict. The audit is of the *train* partition's
    smallest-p test, because selecting what to audit after seeing the held-out result is the
    move the split exists to prevent. Its threshold is the sweep's own surrogate ensemble,
    reconstructed through the new `cross_scale.shift_null_ensemble` from `surrogate_seed` and
    checked against the summary the sweep published, because an order statistic of a *different*
    null is not this study's decision boundary. `StreamedAttenuation` builds the curve one frame
    at a time -- the array form would need 677 MB per scale on the frozen crop -- and a test
    asserts it equals `attenuation_curve` exactly rather than approximately.
    *Consequence:* in the real gate role a FAIL now survives only where the derived record says
    the absence was detectable; otherwise the run is INVALID and names its deficit and remedy
    axis. A PASS is never downgraded, and the receipt says why: attenuation biases toward the
    null, so an undersized crop cannot manufacture a positive. The apparatus is complete and
    every branch is pinned. **At the time this apparatus step closed, no real gate had run.**
    T4C.5m subsequently ran v3 and returned PASS; because a PASS bypasses absence adjudication,
    no real atmospheric absence has yet exercised this boundary and the crop's negative-result
    attenuation decision remains unobserved.
8.  **Re-freeze the campaign** against the derived criterion, recorded as a supersession of
    `t4c6_nz_era5_temperature_850_v1` with its reason, not an edit of it.
    **Done.** `CampaignSupersession` is a third immutable artifact naming both campaigns by
    content hash, and **every stated reason is a check that is re-run against both** -- a reason
    is admissible only where the superseded campaign fails it and the successor passes, so the
    record cannot be written for a defect that was not real or claim a repair that did not
    happen. Properties the predecessor already held are declared under `preserved` and must hold
    for both, and a reason naming a check the registry does not implement is refused, because an
    unverifiable reason is a note and a note cannot retire a frozen design. The successor
    extends the record to six whole calendar years, 2018--2023: 8,764 frames, a 5,258/3,498
    split, 3,496 distinct admissible shifts against 3,005 required, with crop, family, lags,
    embargo, seed, canary and overlap unchanged and asserted so.
    *Consequence:* the retired campaign stays loadable and still reports its own defect --
    otherwise the defect could not be recorded against it -- while `preflight_gate_campaign`
    and the CLI refuse to spend on it. **The minimal repair was refused by the record's own
    checks**, which is the finding of this step: 2023-02-27 gives exactly the 3,007 frames
    `frames_required` asked for and resolves at a Theiler window of one frame *and no more*,
    while the sweep derives that window from the series under test, so the minimal design would
    have reproduced D85 after the 2.8 GB transfer instead of before it. `deferred_to_run`
    records what this does **not** establish: D84 is carried through unchanged and left to
    `run_cached_gate`'s adjudication, the declared margin is a design decision rather than a
    measurement of the derived window, and D43 is untouched -- no data has been acquired for
    either campaign.

**Progress (2026-09-01, `ed-dev`).** Steps 1-4, the resolution half of step 5, and step 6 are
implemented
in `src/analysis_engine/spatial_power.py` with 53 test functions (58 runs), all passing and
order-stable, and 215 passing alongside the cross-scale, scale-signature, statistics, gate and
preregistration suites. Step 5's attenuation adjudication inside `run_cached_gate` is not wired,
because it needs the sub-cropped curve that step 7's receipt fields also need and the two are
built together. Step 6 demoted both constants to labelled heuristics and removed the
power-of-two rounding from the refusal path, which closed D84's gate-versus-gate contradiction
without closing D84. Step 7 published the derivation into the receipt and built the FAIL/INVALID
adjudication step 5 had deferred, so the apparatus D84 asked for is complete; what remains
unanswered is the frozen crop's own measurement, which needs the acquisition. Step 8 recorded
the supersession: campaign v2 supplies the frames D85 needs, every reason it states is a check
re-run against both campaigns, and the acquisition path refuses the retired design. **T4C.5i is
now complete.** Both defects remain open, because both now wait on data rather than on code:
D85's derived Theiler window is unmeasured and D84's attenuation curve does not exist until a
field does. Acquisition is blocked by D43.

**T4C.5j -- the gate record as a read-only surface. Done.** Until this slice, nothing in the
whole T4C line was reachable from the browser: no route, no client method, no component. That
was a real gap for a research instrument, because the two distinctions the line exists to draw
are exactly the two a reviewer has to *see* -- that a retired design must not be acquired, and
that an inadequately powered absence is INVALID rather than FAIL. `src/api/gate.py` serves seven
GET routes over the campaign store and the receipt store, and `GateRecordView.tsx` renders them
under Review as a seventeenth workflow destination.
*Consequence:* the surface serves **no other verb**, asserted over the whole `/api/v1/gate`
prefix rather than left to the handlers; there is no preflight route, because a preflight
reports on a machine rather than on the science, and no acquisition route, because acquisition
is a 2.8 GB spend under a mandatory order a button cannot represent. The four refusals are
served as data and rendered, so a missing button reads as a refusal rather than as an unfinished
panel. Retirement is derived from the supersession's **fingerprint** match, the same comparison
`preflight_gate_campaign` refuses on, and survives renaming every file. The retired design is
shown in full with its defect visible, and its retirement banner is rendered above the design
body. An empty receipt store reports `NOT_YET_MEASURED` with the sentence that this is an
absence of runs rather than of findings. A receipt is served with both verdicts and the
adjudication that separates them.
*What it does not do:* it adds no science, closes no defect, reads no field and touches no
network. **No receipt has ever been served, because none exists.**

**T4C.5k -- the first live acquisition, and what agreement means. DONE.**
The mandatory order is `materialise_weatherbench_overlap`, `materialise_cds_canary`,
`require_canary_overlap_PASS_or_stop`, `materialise_full_cds_record`,
`require_full_cache_overlap_PASS_or_stop`, `preflight_and_run_T4C.6`. Steps 1 to 3 have now run
live and passed; the 3.03 GB record has not been requested.
*What now exists that did not before:* a dated probe of the 0.25-degree WeatherBench store
recorded before anything was read from it; the eight-frame independent overlap in the local
cache; **the first live CDS request this programme has ever made**, proving credentials, licence
acceptance, download, NetCDF normalisation and canonical Zarr conversion end to end; a second
independent window acquired purely to derive a criterion without tuning it to the window it
judges; and campaign **v3**, which freezes the agreement rule the earlier designs left implicit.
*What step 3 found first, and why it was not patched away:* the two routes agreed to 7.324e-4 K
against a declared tolerance of 1e-4 K, and the check failed. ERA5 arrives through CDS packed
per GRIB field, each frame on its own binary lattice -- 2^-10 K and 2^-9 K in this canary -- so
the tolerance asked for more precision than the route can express. Recorded as **D86** and fixed
by changing the unit rather than the number: `encoding_step` measures the lattice a frame
actually occupies and the criterion allows one step, half for round-to-nearest and half for the
independent route's undocumented pipeline. Declared, not fitted; the two measured windows reach
0.72 and 0.44 steps.
*The half that mattered more:* the rule authorising a multi-gigabyte transfer lived in module
code, so the one decision in the campaign that no supersession governed was the decision to
spend. It now lives in the campaign envelope, `preflight` blocks without it, and an
unsatisfiable absolute tolerance cannot be preregistered either. v1 and v2 keep the exact
fingerprints they were sealed under, pinned by test.
*What it did not establish, and what T4C.5m then did:* at the close of T4C.5k the 8,764-frame
record had not been acquired, no crop had been frozen against real data, T4C.6 had not run and
no gate verdict existed. All four are now false; see T4C.5m below. The canary remains two days
of one variable at one level. D84 and D85 remain open.

**T4C.5m -- the record, and the first verdict. DONE; D43 closed.**
Steps 4 to 6 of the mandatory order, run live.
*What now exists that did not before:* a complete regional ERA5 record in the canonical cache --
**8,764 frames** of 161x161 at 850 hPa for 2018-01-01 to 2023-12-31, assembled from 72 monthly
CDS shards for **338.905 MB** in **5,853.7 s**, `content_key a07c23ec89f953c1`, `content_hash
e488f5c3d480f072c834dceae1eeea2a`; a full-cache overlap receipt binding it to the WeatherBench
route at **0.71875** of a packing step against 1.0 allowed, zero mismatches over 207,368 values;
and the programme's **first gate receipt**, served from the checked-in store -- **PASS**, ten
links replicated in train and test, no problems, 753.9 s.
*Why the transfer came in 9x under budget:* the 3.03 GB preflight figure assumes float32 with a
2x safety factor and no compression credit. A bound that refuses to promise is not a prediction,
and the quantity actually checked is the frame count, which is verified exactly at conversion
against the complete expected calendar.
*What step 6 found, and why it was not worked around:* the gate refused the record. D86 had moved
the agreement rule into the campaign envelope and taught the acquisition to use it, but nothing
had taught the admission path, which still read the unsuffixed manifest fields belonging to the
absolute criterion; underneath, `CachedFieldReader.source_provenance` hardcoded those same three
names, so criterion-specific evidence never reached the gate at all. Recorded as **D87**. An
instrument that will not accept properly verified data is as broken as one that accepts
unverified data, and the symmetric hazard is worse: an absolute PASS would have admitted a
record whose campaign declared something else.
*The half that mattered more:* a plan does not carry the agreement rule and the envelope does,
so passing the criterion in from the call site would let the rule that admits a record be chosen
after the record is in hand -- the precise failure D86 exists to prevent. `run_campaign_gate`
takes the campaign, refuses one that declares no criterion, refuses one a supplied supersession
has retired, and hands the frozen rule down; `run_cached_gate` defaults to `absolute` so an
undeclared run refuses. No fingerprint changed -- a code defect, not a design change.
*What it does not establish:* the verdict adjudicates the frozen T4C.6 relationship family on
this exact crop -- one variable, one level, one region, six years. It is not causality, not
universality, not forecast skill, not operational readiness. The independent cross-route check
covers eight of the 8,764 frames, which is the window the frozen design specifies; the other
8,756 are guaranteed structurally -- exact equality against the complete expected calendar,
cross-shard coordinate and variable identity, finiteness, and a content hash over the published
store -- rather than against a second archive. A second independent window mid-record would
close that and has not been acquired. **D84 and D85 remain open and did not gate this result**;
they govern whether an absence was detectable, and a PASS does not route through them.

**T4C.5n -- the audit window. DONE.**
The frozen campaign names one overlap window and it is the record's first two days, so the
record's values were independently verified only at their start. This adds a second WeatherBench
window at **2021-07-01/02**, 3.5 years in: 237.7 MB in 62.9 s, `content_key 19c03cdcde90ceb2`,
**passed at 0.46875** of a packing step with zero mismatches over 207,368 values, receipt
`0f32c89a`.
*It is also a third test of D86's diagnosis, on data that had no part in deriving the rule:* all
eight frames sit on 2^-9 K, so the window disagrees **more** in Kelvin than the gate window does
(9.155e-4 against 7.324e-4) while agreeing **better** in steps. This is the first reproduction of
that inversion on a window acquired after the rule was frozen, so it cannot be an artefact of
the derivation.
*Why it authorises nothing:* it was compared after the record was admitted and gated. A labelled
receipt binds under its own manifest fields; `validate_overlap_evidence` accepts only the two
bare criterion names and cannot read a labelled one; the receipt says `authorises: "nothing..."`
in its own body; and a label that could pass for a criterion is refused. Allowing otherwise
would let the evidence admitting a record be chosen after the record was in hand.
*What it does not establish:* two windows out of 8,764 frames is two windows. The rest still
rests on exact calendar equality, cross-shard coordinate identity, finiteness and the content
hash rather than on a second archive.

Three things were learned by building it, recorded so they are not re-derived:

*   **The estimator caught a calibration error in its own test.** The first known-answer helper
    built a field with a Gaussian kernel of `sigma = L/sqrt(2)` and expected the 1/e crossing at
    `L`. A Gaussian autocorrelation of standard deviation `d` crosses 1/e at `d*sqrt(2)`, so the
    module returned 12 where 8 was expected and was right. The tempting repair -- widening the
    tolerance until the module agreed -- would have calibrated the estimator against the mistake.
*   **A mean-centred window manufactures decorrelation.** Searching to half the interior, as the
    temporal estimator safely does over thousands of frames, returns a confident 19 px length for
    a 40 px interior of 60 px structure. `TRUST_HORIZON_FRACTION = 0.25` bounds the search and
    saturation is reported instead.
*   **A summary is not an ensemble.** The sweep published its surrogate ensemble's mean,
    standard deviation and exceedance count, which is everything a p-value needs and nothing a
    *threshold* needs: the minimum detectable effect is an order statistic, and no summary
    recovers one. The cheap repair -- draw a fresh ensemble -- would have produced a confident
    number that was not the study's decision boundary, so step 7 reconstructs the sweep's own
    ensemble from its seed and reports whether the reconstruction matched.
*   **An empty list is a claim.** The receipt route could have returned `[]` and been
    correct. Rendered, it reads as "no relationship was found", which is a finding this
    programme has not made and the most attractive one it could accidentally publish. The
    distinction between an absence of runs and an absence of findings is now carried in the
    payload, not left to whoever writes the panel.
*   **A repair aimed at the stated requirement can reproduce the defect.**
    `frames_for_resolution` reported that 3,007 confirmatory frames would close D85, and it was
    right about the question it was asked: it audits the most favourable Theiler window of one
    frame, because the window does not exist before the data. The sweep derives that window from
    the measured decorrelation of the series it is testing. A design built to the reported
    number resolves at a window of one and nothing larger, so it would have failed at run time
    for the same reason it was rebuilt. The margin is now an admissibility check on the
    supersession rather than a judgement in whoever re-froze it.
*   **A rounded threshold refuses on the rounding.** `minimum_crop_size` raised a 150 px
    requirement to 256 for dyadic tidiness, and that 106 px of ergonomics was the entire reason
    the frozen campaign's crop was admitted by one gate and refused by another -- while the
    refusal called the round number *statistically recommended*. Convenience conventions and
    decision thresholds must be reported separately, because a caller cannot tell which one
    refused them.
*   **Draws are not resolution.** Surrogates are drawn *with replacement* from the admissible
    circular shifts, so asking for 4,999 always returns 4,999 numbers and a nominal p-value floor
    of 1/5000. The exact test's reference set is the shifts themselves, and its attainable
    p-value is bounded by how many the record contains. Every power check in the repository
    counted draws; none counted the reference set. The frozen campaign's confirmatory partition
    turned out to be on the wrong side of that distinction (D85), which no amount of ensemble
    size repairs.
*   **A frozen design's decision boundary is not an estimate.** Step 4 twice tried to attach
    uncertainty to the minimum detectable effect. A Clopper-Pearson bound on the threshold's
    exceedance probability was unattainable by construction, comparing about
    `(k + 1 + z*sqrt(k)) / n` against a required level of about `(k + 1) / n`. A bootstrap of the
    order statistic was worse, being miscalibrated the dangerous way: at rank 1 no resample can
    exceed the sample maximum, so the interval was one-sided and four independent ensembles of
    4,999 draws all fell above it; rank 10 gave 3-in-20 coverage. Both were answering a question
    about a study drawn with a different seed, which preregistering the seed exists to rule out.
    The surrogate p-value `(1 + k) / (1 + n)` is exactly valid under exchangeability, and the
    `k + 1`-th largest of the frozen ensemble *is* the boundary the gate applies.
*   **The attenuation model has a stated domain and must refuse outside it.** Near the
    `log(bins)` entropy ceiling the weak-dependence linearisation fails. Separately, a curve still
    climbing at the largest crop yields a non-positive intercept with an excellent fit (R^2 0.999)
    -- that is the strongest evidence of inadequacy available, and it is reported distinctly from
    a fit failure rather than being softened into a number.

**Definition of done:**

*   The acquisition layer and `gate_campaign` agree on one geometry contract, and a crop admitted
    by one cannot be refused by the other.
*   No refusal in the sizing path cites a judgement constant; every refusal cites a measured or
    derived quantity and names what would close the deficit.
*   The power-of-two rounding appears nowhere in a refusal, and no heuristic is described to a
    caller as statistically recommended.
*   An adequately powered absence and an underpowered absence produce **different verdicts** on
    the same input, demonstrated by a test that varies only the crop.
*   Attenuation is validated against a planted synthetic effect: a known TE injected at a known
    lag is recovered at full crop and measurably attenuated at a reduced one, with the predicted
    attenuation matching the observed within a stated tolerance.
*   The receipt carries the complete derivation, and re-running it from the receipt reproduces the
    same verdict.
*   The superseding campaign records why it supersedes, and the original remains readable.

**Claim boundary.** This task decides only whether the declared family is adequately powered on a
given crop. It is not a T4C.6 verdict, it does not adjudicate the cross-scale hypothesis, and a
passed power check is not evidence of an effect. Measuring power is also not permission to reduce
the sample requirement to fit an existing layout: if the derivation says the crop is inadequate,
the crop grows or the family shrinks.

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

**T4D.1 Feature detection.** Local maxima in `CoefficientField` above a surrogate-calibrated threshold, with sub-pixel localisation; yields `{id, t, y, x, scale, orientation, strength, phase}`. **DONE (`src/analysis_engine/spectral_feature.py`).**

**Met.** 15 test functions in `src/tests/test_spectral_feature.py`, all passing. A planted blob is
recovered sub-pixel at (32, 32), (30.4, 41.7) and (25.5, 25.5); the refinement beats the integer
peak it starts from; the R13 margin excludes a detection the detector demonstrably can see with
the mask off; the threshold is fitted over the whole record so a quiet frame reports nothing
beside a loud one, and a frozen set is inherited verbatim; truncation is recorded with the
strongest kept; a real family reports `phase: None` rather than inventing zero.

**One design change the tests forced.** Maxima are *regional*, not strict. A feature centred
exactly between two samples produces two exactly equal samples and a strict test rejects both --
and that is the position a smoothly advecting feature passes through twice per pixel of travel,
so a tracker built on a strict detector would watch features blink out and back as they drift. A
connected group of equal-valued pixels is one candidate, positioned at its centroid, carrying
`plateau_pixels` so a consumer can see that a flat top is located less sharply than a peak.

**Not met, and stated rather than glossed.** The threshold is `sigma x RMS` fitted on the record,
not surrogate-calibrated. The surrogate-calibrated path exists (`src/core/extraction.calibrate`)
but calibrates on the field, not on its coefficients; nothing here reports a significance, and a
threshold crossing is not offered as one.

**T4D.2 Tracking.** Frame-to-frame association with scale/orientation gating; nearest-neighbour first, Hungarian assignment (`scipy.optimize.linear_sum_assignment`) once it needs to be respectable. Yields `SpectralFeatureTrack {feature_id, wavelet, start_time, positions[], scales[], strengths[], orientations[], velocity, scale_velocity, lifetime}`. **DONE (`src/analysis_engine/spectral_tracking.py`).**

**Built as a bridge, not a second tracker.** `src/core/tracking.py` (TG2.3) already supplies
greedy-nearest and Hungarian association, a coincidence radius *derived* from the alpha the search
was calibrated at and the frame's own density, declared scale and orientation gates that are
refused when the features cannot measure them, an associator whose answer is checked against the
gates that admitted it, and a clock that is every frame searched so an empty frame ends a track
rather than being bridged. `Track` already yields positions, scales, strengths, orientations,
velocity, scale_velocity, lifetime and doubling time. A second tracker would have meant a second
set of those refusals, and the second set is the one that would be weaker. What was missing was
the translation of banded coefficient maxima into `src/core/feature.SpectralFeature`, which is
what this task supplies.

**Met.** 19 test functions in `src/tests/test_spectral_tracking.py`, all passing. On
`advected_vortex_sequence`, whose trajectory, velocity and doubling time were recorded before any
of this code: 4 tracks over 24 frames, the two level-4 bands unbroken end to end (D1's aliasing,
absent); every band's track within **1 px** of the recorded trajectory on its transverse
coordinate (worst 0.667); every band's velocity equal to advection plus the structure's own growth
along the axis its band high-passes, to better than **0.11 cells/step** (worst 0.109, best 0.020),
a prediction with no free parameter; the level-5 tracks born at frame 9 and never before; and no
track spanning two levels, because the one-octave-per-frame gate refuses it.

**The acceptance sentence above is not what was checked, and this is the reason.** "Position error
< 1 px" assumes a detector whose maxima coincide with the structure's centre. A detail wavelet is
derivative-like: a symmetric blob has *zero* detail response exactly at its middle and two maxima
on its flanks, about one analysing width out along the high-passed axis, and that distance grows
as the structure grows. So a detail-coefficient maximum is never at the centre. The `< 1 px`
criterion in field space is met by `src/core/extraction` and is already recorded as the
`4D.position` benchmark check at 0.052 cells; from coefficient maxima it is unattainable, and the
two checks above are what this representation can actually be held to.

**Found: D88.** Pooling bands into one frame -- necessary, because there is no scale ratio to gate
inside a single band -- exposed that an undecimated band has the parent grid's *shape* but was
never registered to it. The analysis filters are anchored at index 0, so a response is displaced
by half the accumulated support: 0.5 px at level 1, **22.5 px at db2 level 4**, growing with
level. Fixed by `stationary.analysis_delay`, `CoefficientField.parent_alignment` and a subtraction
in `detect_features`; and, where arithmetic cannot fix it, by refusal -- only a linear-phase filter
has a single delay, so `db2` and `db3` are refused cross-scale linking by name with the two ways
out stated. No previously recorded result changes: everything before this collapsed a band to a
scalar, and a circular shift moves no mass.

**A growing structure is several tracks, not one that migrates.** The bank is redundant, so a
doubling vortex excites the coarse level *beside* the fine one rather than instead of it. Scale
evolution therefore lives in the population of tracks, not in one track's `scale_velocity`.
Reporting it the other way would require claiming a merge, which this tracker does not claim.
**Acceptance (this is where D1 comes home to roost):** a synthetic vortex advected on a known trajectory with a known scale evolution is recovered with position error < 1 px and correct scale-doubling detection. **This test cannot pass without T3.5.6** - with shift-variant Haar, coefficients flicker with grid parity and the tracker reports births and deaths that are pure aliasing.

**T4D.3 Narrative summaries.** Human-readable track descriptions ("travelled southeast over six frames while dominant scale doubled and coefficient energy rose 43%"), using precursor/signature language per R7. **DONE (`src/analysis_engine/spectral_narrative.py`).**

**Met.** 25 test functions in `src/tests/test_spectral_narrative.py`, all passing. Every number a
sentence contains is checked against the track it came from -- the spoken speed against
`Track.speed()` for all four tracks of the T4D.2 vortex pass, the spoken magnitudes against the
first and last observation. The band ordering is rendered as a **candidate precursor
relationship**, in the same sentence as the facts that it was not tested against a null and that
no merge is claimed. `assert_no_causal_language` imports `OUTSIDE_THE_LADDER` from `claim_ladder`
rather than restating it, and is asserted against every word in that list.

**Three clauses of the example sentence were not sayable as written, and the reasons are
measurements.** *"Southeast"* needs a grid that knows where north is: the sign relating row order
to latitude, and the cosine that stops a degree of longitude counting as long as a degree of
latitude -- at 60 degrees north an equal row and column displacement is a bearing of 26.6 degrees,
not 45, so omitting the cosine changes the compass word rather than only a number. A `latlon` grid
gets the bearing; every other grid gets *"toward increasing row and increasing col"*. *"Dominant
scale doubled"* is not a statement any one track supports, since a track that holds one level has
a scale velocity of exactly zero; what is measured on the vortex is level 4's maxima weakening by
31.6% and 32.7% while level 5's strengthen by 18.9% and 17.0% and level 5 is first excited at
frame 9, and that -- an ordering of two bands over one record -- is where the growth is reported.
*"Coefficient energy rose 43%"* names the wrong quantity: energy is the square of magnitude, so
43% in one is 104.5% in the other, and both now appear under their own names in the same clause.

**Found: D89.** T4D.3's guard was written with the same defect as the programme's existing one and
had it caught by its own test: `\bcauses\b` does not match inside `co2_causes_warming`, because
an underscore is a word character -- and an identifier is exactly the shape of the author-supplied
text that reaches rendered prose through an evidence entry's label or an alternative's wording.
Fixed in both places by flattening punctuation to spaces before the word boundaries are applied.
The remaining limit is stated rather than closed: a word run together with another with no
separator is not caught, and the substring match that would catch it also refuses "causeway".

### Phase 4E - `FeatureConstellation` as an attributed graph (the bottom-up direction)

The primitive here is **an attributed graph, not a summary descriptor.** A pairwise distance/bearing histogram cannot express "A and B are weakening at small scales while C strengthens one scale up" - it throws away which node carries which behaviour. The graph keeps it.

**T4E.1 Constellation extraction as attributed graphs.** - **DONE** A constellation is a small graph over co-occurring features:

*   **Node attributes** (from the 4D tracks): position, scale, orientation, coefficient strength, phase, `d(strength)/dt`, `d(scale)/dt` (the `scale_velocity` already produced by T4D.2), velocity vector, age.
*   **Edge attributes:** separation distance, bearing, **scale ratio**, strength ratio, **relative radial velocity** (signed: converging vs diverging), and time offset between the two features' onsets.

Start with **pairs and triples only** - O(n^2)/O(n^3), not exponential; pairs alone already carry relative geometry, scale ratio and convergence. Hard cap on cardinality, raised only if triples prove informative.
**Rationale:** thousands of local features in arbitrary arrangements is frequent-subgraph mining, not a Cartesian sweep. The 1,000-combination guard in `expand_parameter_matrix` neither applies nor protects here.

**Met, as a bridge rather than a second graph.** `src/analysis_engine/spectral_constellation.py`;
`src/tests/test_spectral_constellation.py`, 45 test functions, **54 passed**.

**The graph is TG3.3's.** The cross-domain line already built the attributed graph this task asks
for, and built it stronger: eight typed relations that each divide by something the two features
carry or declare that they cannot, a value that is dimensionless or `None` with no third case,
relations that refuse by name rather than treating absence as agreement, and a matcher that
refuses above `MAX_MATCH_NODES` instead of approximating. A second graph in this line would be a
second set of those refusals and the weaker set -- the argument T4D.2 already made about the
tracker -- and its edges would be in cells, which is the number TG3.3 exists to refuse. What was
missing is the **enumeration**: `constellations_from_set` is `C(n, k)` over a whole set with no
notion of a frame, which over the 8,764-frame T4C.5m record would pair a maximum in January with
one in March. This slice enumerates the co-present tracks of each *searched frame*, hands each
pair and triple to `constellation()`, and carries the track-derived attributes -- onset, age,
local velocity, `d(strength)/dt`, `scale_velocity` -- alongside the graph rather than inside it,
because they are in cells and frames and R19 refuses them across a domain boundary.

**Three of the eight relations are measurable on 4D features; five refuse by name.** `distance`,
`relative_scale` and `succession` measure. `temporal_lag` and `co_occurrence` have no
`temporal_scale`, `direction` and `convergence` no `orientation`, `containment` no `extent`; each
is reported with the field it lacks rather than dropped. The declaration is made once for the
pass, not per frame, for the reason `relation_axis` exists.

**Three of the specified attributes are not what the roadmap's wording implies, and are carried
accordingly.** *Separation* is between two coefficient maxima, and a detail maximum sits on its
structure's flank: on the vortex at frame 9, `L4/LH` and `L4/HL` following one vortex are 1.4447
scale lengths -- 11.56 cells -- apart. *Strength ratio* across bands is a ratio of filter gains
until each strength is divided by its own band's RMS; at frame 9 the raw `L4/LH -> L5/LH` ratio is
1.755 and the normalised one 0.554, so the two disagree about which band is the stronger, and both
are carried. *Time offset between onsets* is a lower bound when a track was already alive in the
first searched frame; both level-4 tracks are left-censored here, so the nine-frame offset is a
bound. *Bearing* is degrees in the row-column plane with no grid consulted, and says so; the
compass is T4D.3's and the invariant form is TG3.3's `direction`.

**The budgets refuse rather than truncate.** A frame over the node cap, or a sweep over the
constellation budget, stops and names the numbers, because a silently truncated sweep makes
T4E.4's support a count of what fitted. On the T4D.2 vortex the pass emits **135 constellations**
over 24 searched frames -- 87 pairs and 48 triples -- and the count is checked frame by frame
against the combinatorics of its own census.

**Not claimed.** No null, no support count, no significance, no recurrence: a constellation here
is one observation of one arrangement in one frame, and matching is TG3.4's, clustering T4E.3's
and support T4E.4's. `networkx`, which this phase's dependency note lists, was not adopted -- at
two and three nodes the graphs are complete and there is no algorithm to run.

**Found: D90.** `distance` -- the one relation carrying geometry -- refused on every feature the
4D line produces, because T4D.2 named the position axes `cells` and the dyadic octave `parent-grid
px`, and TG3.3 correctly declines to divide a separation in one unit by a scale in another. On an
undecimated bank whose levels are all mapped to the parent grid, those are two spellings of one
unit. The failure was silent: a refusal is recorded on the graph rather than raised, so a mining
pass would have run with the geometry missing. Nothing already recorded changes, because no
constellation had ever been built from these features. Fixed in `_scale_quantity`; the rule is
untouched and a test pins that a scale genuinely in metres still refuses.

- **DONE** **T4E.2 Invariance by construction.** Every attribute above is expressed relatively, so a configuration is recognised anywhere on the grid: distances normalised by the participating features' scales, bearings measured relative to the constellation's own principal axis, strengths normalised within the constellation. Translation and rotation invariance therefore fall out of the representation rather than being bolted on afterwards. Scale invariance is a **separate, explicit toggle** - normalising by absolute scale answers "does this configuration recur at other scales?", and **that query is the universality hook.** It must be first-class in the API from day one, and it must be possible to run the same mining pass with it on and off and compare.

**Met, as a choice between two matchers that already exist rather than a third normalisation.**
`src/analysis_engine/spectral_invariance.py`, 45 tests / 53 cases. TG3.4 had already built both
halves of the specification's first sentence and had already measured that one of them does not
do what the sentence assumes.

*   *"Distances normalised by the participating features' scales"* is TG3.3's `distance`, and
    TG3.4 measured it on real extracted features: it is **not** rescaling-invariant, because it
    divides by an *estimated* scale whose estimate drifts from about +3.6% at sigma 3 to about
    -2.6% at sigma 18 -- 4.8% movement at `scale_factor=3` against a 3.3% noise floor. TG3.4
    deliberately left it out of `MATCHERS`, because a registry entry carries a declaration and
    this one has none it can demonstrate.
*   What survives a rescaling is a separation over *another separation* -- TG3.4's
    `relative_geometry`, reproducing to 0.37% -- and it needs three features, because a pair has
    one separation and its ratio to itself is 1 for every pair in every domain.

**The toggle is first-class and it is priced.** `signature_for(..., scale_invariant=...)` and
`sign_constellations` select between the two matchers; `compare_scale_modes` runs the same pass
both ways and returns the difference. On the vortex pass: **135 signed scale-specific, 48 signed
scale-invariant, 87 lost -- every pair.** That is what turning the universality hook on costs,
as a number rather than an argument. What it buys is rescaling invariance and a signature whose
every entry is dimensionless, which is the only form that could be compared with a configuration
from another domain -- `cross_domain_comparable` is true in exactly one of the two modes.

**Invariance is measured, not declared.** Translation, three rotations, reflection and every
relabelling of the members leave the signature vector identical to floating-point precision in
both modes, on exact geometry where the truth has no noise in it. A uniform rescaling leaves the
scale-free shape alone; an estimator that misses a rescaling moves the scale-specific geometry
by exactly the factor it missed, which is the difference between the two modes made arithmetic.

**Found: the benchmark this programme supplies for invariance has no principal axis.**
`planted_configuration` is an *equilateral* triangle, so its position covariance is isotropic and
its principal axis is whatever the noise decided. Measured over 24 field-noise realisations
through the real extraction pipeline: anisotropy stayed in **[1.0077, 1.0421]** while the
recovered axis angle scattered from **0.78 to 158.08 degrees** -- effectively uniform over the
half-circle, circular standard deviation near 50 degrees -- and the shape ratios over the same
replicates reproduced to **0.218%**. A bearing block written without a guard would have emitted a
confident angle that was pure noise, on the exact configuration nominated here for testing
invariance. `AXIS_ISOTROPY_FLOOR` is that measurement rather than a choice and
`calibrate_axis_admission` re-measures it; clearing it is a minimum, not a precision claim. The
vortex triples clear it by two orders of magnitude (smallest anisotropy 85.22).

**Three things the specification's wording does not imply.** A bearing is folded to [0, 90]
degrees, because an edge is unordered and an axis has no sign -- so the signature is invariant to
*reflection* as well as rotation, which is a consequence of the grid declaring no orientation
rather than a preference. At two and three members the bearings add no degree of freedom: three
pairwise separations determine a triangle up to similarity and reflection, so the angles are a
re-expression of the geometry block, kept because they are the readable form and because 4F must
project a configuration back onto a map. And the canonical order is a lexicographic minimisation
over node correspondences, not a sort of each block -- sorting independently lets two
configurations agree with no single correspondence that makes both blocks true at once, which is
a matcher reporting an agreement it cannot exhibit.

**Not claimed.** No cluster, no match, no support count, no null, no p-value: a signature is a
description of one configuration in one frame, and deciding that two of them are the same needs a
tolerance calibrated against replicates, which is T4E.3. The universality hook is *exercised* on
this record and not *evidenced* by it -- the tracked bank has two levels, so "does this
configuration recur at other scales?" has almost no room to be answered here. Nothing has been
signed from ERA5 or from any second domain. `networkx` remains unadopted for the same reason
T4E.1 gave.


**DONE — T4E.3 Approximate matching by clustering, not exact subgraph isomorphism.** Exact attributed-subgraph isomorphism is the wrong tool: the attributes are continuous and noisy, so exact matching would fragment one physical configuration into hundreds of near-duplicate patterns. Instead define a distance metric on attributed graphs (attribute-weighted, with a declared weighting per attribute) and **cluster** constellations in that space; a "pattern" is a cluster centroid with a tolerance radius. More robust, and tractable.
**Acceptance:** a synthetic three-feature configuration, replayed at a different grid location, a different rotation and with 10% attribute jitter, lands in the same cluster; replayed at a different absolute scale it lands in the same cluster **only** when scale invariance is enabled.

Implemented in `src/analysis_engine/spectral_clustering.py`. All four comparable blocks have
caller-declared weights and a published dimensionless reduction. The only accepted radius is a
`MatchTolerance` measured as the maximum all-pairs distance among declared replicates of the same
physical configuration, bound to both the exact metric digest and the signature family. Clustering
is deterministic complete-link agglomeration: every cross-member distance and every
member-to-centroid distance must remain inside that measured radius, so a single-link bridge cannot
join endpoints the calibration says are different. Each pattern reports its aligned centroid, the
calibrated radius and its observed member radius; its member count is explicitly not yet a
minimum-support claim.

The acceptance configuration passes at another location, after a 73-degree rotation, and under a
held 10% perturbation of geometry, strength and scale. Doubling both its geometry and member scales
splits the scale-specific signatures and joins the scale-invariant signatures. The first run found
and fixed **D92**: T4E.2's exact canonical order can change discontinuously under small noise, so a
distance between canonical vectors attached strengths and scales to the wrong vertices and inflated
the measured noise floor. T4E.3 now minimises over the at-most-six valid node correspondences while
moving edge and node attributes together, and aligns every centroid to a deterministic medoid.

**DONE — T4E.4 Minimum-support mining.** Support-threshold miner with early pruning over the clustered patterns; configurable minimum support; support counts reported with every pattern; hard wall-clock and candidate-count budgets so a sweep cannot silently run for days.

Implemented in `src/analysis_engine/spectral_mining.py`. Support is the count of distinct
`signature.key` constellation identities, not raw list length; a duplicate identity is refused
rather than allowed to manufacture frequency. Candidates are scanned in decreasing support, so
the first below-threshold candidate prunes the complete remaining tail while every pattern still
appears in the receipt with its support and decision. The threshold is inclusive and a value above
every pattern returns a complete empty result rather than a failure. `MiningBudget` requires both a
candidate cap and positive finite wall time; exceeding either raises a client-safe refusal whose
structured context says `partial_result: false`. The wall clock includes identity validation and
support preflight as well as the scan. The acceptance fixture has one five-occurrence and one
two-occurrence cluster: minimum support three retains exactly the former. The receipt states that
this deterministic frequency filter is not recurrence significance, a null test, predictive
evidence or a discovery.

**T4E.5 The identity step at record scale *(fixes D96)* -- IN PROGRESS.** `cluster_signatures`
is complete-linkage agglomerative clustering implemented directly, measured at about O(n^3) on
real signatures, against a training period presenting some 843,000 configurations. Replace the
implementation without touching the definition.

`src/analysis_engine/spectral_identity.py` exists and is exact where tested: identical receipts
against the original on the acquired ERA5 record at 79x, 200x and 393x speedups, and about
O(n^1.6). **It is not finished.** 26 of 37 mutants are killed and eleven are alive, including the
radius condition, the exact verification behind the box filter, the weights in the vectorised
distance, the family check and the pattern ordering. It also does not yet close D96: the
cross-distance matrix is dense per component, and on this record the tolerance graph is a single
component, so 843,000 configurations would need 5.7 TB.

**Acceptance:** every mutant killed or argued equivalent; the cross-distances sparse, which the
Lance-Williams update supports because a merged cluster's neighbours are the intersection of its
parents'; and the ceiling re-measured on the acquired record.

**T4E.6 Publish what the tolerance admits *(fixes half of D97)* -- DONE.**
`calibrate_signature_tolerance` states a false-rejection rate and never computes the
complementary one -- how often the radius admits pairs the record does not call the same
configuration -- and that is the rate deciding whether a pattern means anything. Make the
calibration two-sided: both error rates, both distributions, their overlap and their separation,
returned together and refused apart.

**Acceptance:** a tolerance cannot be obtained without both rates attached; on the acquired record
the receipt reproduces D97's measured figures; and on a synthetic record with a planted identity
the two rates move in opposite directions as the radius is swept.

**Met**, with one clause read rather than followed literally. "Both rates attached" is enforced as
*the question is always answered*: where no contrast population is supplied the tolerance carries
`ADMISSION_RATE_NOT_MEASURED` and no number, rather than the calibration being made impossible.
Requiring a contrast outright would have made a valid one-sided noise-floor measurement
unobtainable; recording the silence as a refusal keeps it honest without that. 22 tests, 19
mutations all killed.

**It found two things nothing had reached before.** The calibrated radius is **arbitrary**: across
25 configurations of the same record with the same metric it spans 0.1683 to 1.1220, a factor of
6.7, with the admission rate running 8.9% to 84.5% -- and the longest run, which the pipeline
naturally reaches for, sits at the 48th percentile rather than at an extreme, so nothing signals
the arbitrariness. And the rate the calibration *did* report is a **tautology at its own operating
point**: the radius is the largest replicate-pair distance, so the measured false-split rate there
is identically zero on any input whatever. Both are pinned by tests, so a calibration that later
reports a real split rate will break them and have to replace them.

On the acquired record no radius holds both rates below 10%; the best achievable is 0.5380 at a
worst rate of 14.9%. **T4E.6 measures the radius and does not move it** -- the suite asserts that
supplying a contrast leaves the value unchanged -- and choosing a defensible one is T4E.7.

**T4E.7 Calibrate without replicates *(the second half of D97)* -- DONE.** The calibration
asks for repeated measurements of one physical configuration and **a real atmospheric record
contains none**. Calibrate against a null instead: measure the distance distribution the record
produces between configurations that are not the same one, and locate the radius where the
observed departs from it. `src/analysis_engine/spectral_null_calibration.py`, verified by
`src/tests/test_spectral_null_calibration.py`.

**Acceptance:** on a synthetic record with a planted identity the chosen radius recovers the
planted grouping; on the acquired record a radius is chosen with both error rates published; and
where the two distributions do not separate the calibration **refuses and returns no radius**
rather than an indefensible one.

**Met on the first and third clauses. The second is answered rather than met, and the answer is
the task's main result.** On a synthetic record of twelve configurations observed ten times each,
the calibration -- which never sees the labels -- chooses radius 0.033979, and against those
labels it admits **none** of the 6,600 pairs that are not one configuration and groups 462 of the
540 that are (85.6%); its mixture fraction reads 0.0696 against a true 0.0756. The refusal path
is exercised on a record where every configuration is observed exactly once, and returns no
radius. **On the acquired record it also returns no radius**, at two sampling budgets that agree
to within 0.001 on the band, so publishing a radius anyway would have been the failure this task
exists to prevent.

**Why the acquired record gives none, in two parts, both measured.** The excess over the
surrogate-record null *is* significant -- 0.0916 at p = 0.05 -- but its maximum sits at
`r = 0.8271`, and no radius clears the simultaneous band before the null already admits a quarter
of its own pairs. In the close-pair tail the excess is **negative**: the record has fewer
near-identical signature pairs than its own surrogate null, because phase randomisation produces
a homogeneous field whose few features are generic and therefore alike while the record's are
diverse. That is **D98**, now in the ledger: the null removes the features as well as their
recurrence (69,580 signatures against a median of 16,090, a factor of 4.3), so what it measures
in the bulk is a difference between two feature populations and not recurrence. Separately, under
the strictest reading of identity -- the same tracked constellation observed again -- the record's
69,580 signatures come from 64,153 distinct constellations seen a mean of 1.08 times, which puts
the mixture fraction's ceiling at **2.8e-06**, some 13,700 times below the band. Exact recurrence
is undetectable on this record by arithmetic rather than by implementation. Recurrence in the
sense clustering exists for -- different constellations that are the same *kind* -- is not
bounded by that number and remains open.

**Four design decisions were forced by measurements that contradicted a first implementation**,
each recorded in `VERIFICATION.md` with the figure that forced it. A Benjamini-Yekutieli sweep
over 64 radii would have needed 7,588 surrogates on this record to be capable of rejecting
anything, each a full pipeline re-run, so the test is one maximum statistic with a simultaneous
band rather than a corrected sweep that would have returned a clean-looking negative. A grid
spaced evenly in quantile cannot see recurrence below about 1.5% of pairs and steps by 13.9% at
64 points over 4,000 pairs, so it is geometric and its size is derived from the declared target,
and a grid too coarse for that target is refused with the count that would suffice. The mixture
fraction read at the null's median returned **negative** values on a record whose planted
grouping the separation test had just detected cleanly, so it is read in the close-pair tail
under a declared cap. And the contamination rate is an **estimate, not a bound** -- an earlier
draft of this module claimed otherwise, and measurement against labels put it below the truth at
17 of 73 radii, by up to 0.11 inside the region a radius is chosen from.

**No absolute split rate is published.** It is not identifiable from a record with neither labels
nor replicates, and the estimate a first implementation did publish read 1.91% against a true
12.53%. This is the successor to T4E.6's finding that the old rate was zero by construction: the
fix is not a better estimate of that quantity but the statement that the quantity cannot be had.
A fifth refusal status was also removed after mutation testing asked what could reach it -- a
non-positive mixture fraction cannot coexist with a significant excess, because the largest
observed distance puts `F_obs` at 1 while `F_null` is at most 1.

77 test functions, 93 cases, 38 mutations all killed across two batches. Eleven of the twelve
first-pass survivors were real gaps and are now bound by tests; the twelfth was the unreachable
status. **D97 is not closed** -- nothing in the pipeline consumes the new calibration yet, and on
real data it still yields no radius -- and **D98 is opened**.

**T4E.8 A radius the record's own labels can defend *(addresses D98, D99, D100)* -- IN PROGRESS; slices 1-5 implemented, acquired-record acceptance unmet. Identity target decided 2026-09-09: `kind_recurrence` primary, `track_continuity` and `spatial_persistence` diagnostic.**

*Specified below as a null-building task. Slice 1 measured that the null was not the binding constraint, so everything from here to the slice-1 block is the superseded specification, kept because the reasoning in it is why the measurement was worth taking.*

T4E.7's calibration is sound and has nothing clean to run against. A `spatiotemporal_phase` surrogate
destroys phase organisation, and phase organisation is what makes a feature, so the null
population is not the record's features without their recurrence -- it is a different and much
sparser population, a quarter the size. Every excess it shows in the bulk is therefore a
comparison between two feature populations, and in the close-pair tail the record's pairs are
*further apart* than the null's. Build a null that keeps the feature population exactly and
destroys only the relationship that makes two signatures the same configuration. Two candidates:
permute which frames a tracked configuration's observations are drawn from, or reassign
signatures between tracks. Choosing between them is a scientific declaration about what "the same
configuration" means, not an implementation detail, and it belongs in the task's declaration.

**Acceptance:** the null produces a signature population within a declared factor of the record's
own, and the factor is published; the calibration recovers a planted grouping against it on
synthetic data as it does against the present one; and on the acquired record it returns either a
radius with both rates published or a refusal whose reason is no longer about the null.
`ATTRIBUTE_PERMUTATION_NULL` already exists in `spectral_null_calibration.py` as the cheap,
deliberately anti-conservative comparison, and its caveat says why it cannot itself be the answer:
permuting signature components independently scatters the null off the surface real signatures
occupy.

**What this cannot fix, and must say so.** Under the strictest reading of identity -- the same
tracked constellation observed again -- the acquired record's 480-frame slice yields 69,580
signatures from 64,153 distinct constellations, a mean of 1.08 observations each, and a mixture
fraction ceiling of 2.8e-06 against a measured band of 0.0387. No null rescues that; the reading
itself has to change. The question clustering exists for is whether *different* constellations are
the same kind, and T4E.8 must state which reading its radius is calibrated for before the radius
means anything.


**Slice 1 (2026-09-08) reversed the premise before anything was built, and is DONE.** A signature
carries `track_ids` and a `time`, so the strictest reading of identity is *already labelled in the
record*: 4,444 tracked constellations are observed more than once, giving 6,838 within-key pairs.
`calibrate_signature_tolerance`'s docstring says "an atmospheric record contains no replicates at
all"; that is wrong in letter, and this slice measured why it is right in effect.

**Met.** Receipt at `data/identity_calibration/t4e8-replicate-census.json`; measurements in
`VERIFICATION.md`.

* **A null cannot beat labels, and the labels give 73%/27%.** Over all 6,838 within-key pairs the
  best balanced operating point groups 0.7164 of genuine repeats while admitting 0.2883 of
  unrelated ones (AUC 0.7827; 0.7995 on geometry alone). So a clean null could not have produced
  a defensible radius under this reading, and **D98, though real, is not the binding constraint**.
  T4E.8 as originally specified -- build a better null -- could not have reached its acceptance.
* **The within-key distance measures physical evolution, not measurement noise.** One frame apart,
  it rises monotonically with how far the tracks moved: 0.0488 at zero cells, 0.1055 at one,
  0.2208 at three, 0.3208 at six, against a median displacement of 3.25 cells.
* **The band flips in 60.8% of same-configuration pairs** (4,160 of 6,838), and scale-specific
  signing puts that into the comparable vector. **D99.**
* **`strengths` scores AUC 0.5053 weighted alone** over 6,650 pairs while contributing 4.3% of
  same-pair squared distance against 1.3% of different-pair. **D100.**
* **A defensible radius exists once both are excluded.** On the 124 pairs one frame apart with no
  band change and at most one cell of motion, a radius of 0.0972 groups 90% of them and admits
  **4.5%** of unrelated pairs -- both rates absolute counts against the record's own labels, no
  null and no mixture anywhere. This is the first identity radius on the acquired record this
  programme can defend. It rests on 124 pairs, and it is a *noise floor*: it admits near-stationary
  repeats and will reject a configuration that recurs after moving, which is the thing clustering
  exists to find.

**What remains, re-specified by that measurement.** The maintainer's declaration is taken: fix the
signature before building any null. In order -- close D99, since a band that flips more often than
it holds defeats every downstream radius; close D100 by deciding whether a carried coefficient
magnitude belongs in the comparable half at all, which is a declaration and not a tuning exercise;
then re-measure the labelled discrimination, since the 124-pair stratum should grow as the band
stabilises. A null is needed only for the *broad* reading -- different constellations that are the
same kind -- for which no labels exist, and it should be built against a signature already known
to separate the repeats it can be checked on.

**The ceiling paragraph above still holds, with one correction slice 1 forces.** The
2.8e-06 ceiling bounds what a *population-level* calibration could ever see, and nothing changes
that. It does not bound a **labelled** radius, which needs no mixture fraction at all and is what
slice 1 measured. So the strict reading is not hopeless as that paragraph implies -- it is
hopeless for a null, and usable with labels. The broad reading remains unmeasured and unlabelled.

**Slice 2 (2026-09-08): an explicit spatial identity, and a failed operating criterion.**
The registered `spatial_geometry` mode compares pairwise separation in a declared record/grid
and admitted bearings. It carries detector scales, bands and strengths without comparing them.
This resolves band/magnitude sensitivity at fixed positions in this candidate definition,
preserves discrimination between differently separated pairs, and requires matching scope,
source and units. Original modes and the frozen mining declaration are not silently changed.
`spectral_identity_audit.py` supplies two-sided descriptive errors and empirical feasibility;
`tools/audit_spatial_identity.py` makes the real-data measurement reproducible from a hashed
exploratory design. Tests and captured output are recorded in `VERIFICATION.md` under T4E.8
slice 2; the architecture's section 3E.8 describes the implementation and boundaries.

The unchanged first window reproduces 69,580 signatures and 6,838 repeated-key pairs. The
candidate's all-pair AUC is 0.8877, and its stationary calibration radius is 0.138075. Applied
unchanged in two disjoint training windows, it gives split/admission rates of **28.81%/9.36%**
and **25.63%/10.895%**. Neither meets the declared 10%/10% criterion. These are tracked-key
proxy labels, not physical ground truth or independent samples. Removing the band condition
grows the calibration stratum to 188 pairs from 169 keys; it does not establish adequate
tail precision or recognition of the same physical kind in another constellation.

**Next dependency.** The remaining question is the identity target and its independent
validation: continuity of evolving tracks, persistence of spatial configuration and recurrence
of a physical kind must not be conflated. A scientific declaration and suitable labels or a
reviewed catalogue are needed before that target can change. No radius from this exploratory
audit is approved for mining. A second, explicitly descriptive design amendment reports whether
*any* radius meets both empirical bounds on these already inspected populations; it changes
no threshold or acceptance criterion. Both audit versions remain in
`data/identity_calibration/`. T4E.8 stays open; a cleaner null, the pilot and scaling do not
turn its failed discrimination criterion into a pass.

**T4E.8 slice 3 -- Make the identity target a declared object the audit must be given -- IMPLEMENTED 2026-09-08; limbs 1 and 2 delivered, limb 3 not delivered.**

*Slice 2 failed its criterion, and the reason it failed is not a threshold. Three distinct
questions -- continuity of an evolving tracked constellation, persistence of a spatial
configuration, and recurrence of the same physical kind in a different constellation -- were
being measured with one mechanism and one label source, and nothing in the design said which
was intended. This slice does not answer that question. It makes the question askable, refuses
to proceed while it is unanswered, and makes the third target evaluable at all. Choosing the
target remains a scientific act for the maintainer, not a default the software supplies.*

**The gap this addresses.** `data/identity_calibration/t4e8-spatial-design-v2.json` has a
`labels` field, but it is prose describing the track-key mechanism; there is no field anywhere
that states what those labels are evidence *for*. `tools/audit_spatial_identity.py:_labels`
derives repeat and unrelated pairs from `sorted(track_ids)` inline, so the tool can only ever
evaluate track continuity and spatial persistence. Recurrence of a physical kind -- the target
the mining, sequence and precursor machinery downstream actually requires -- is not merely
unmeasured, it is currently unevaluable, because the only label source available is derived
from the same record the identity is derived from. `spectral_identity_audit.py` is already
label-agnostic; it takes distance sequences and knows nothing about where they came from. The
defect is above it, in the tool and the design.

**Limb 1: a declared target.** An `identity_target` field becomes required in the audit design.
It is enumerated, not free text -- `track_continuity`, `spatial_persistence`, `kind_recurrence`
-- and each enumerant carries, as data rather than commentary, what it claims to recognise,
which evidence classes can validate it, and what a pass under it does not license. A design
without the field is refused by name before any source value is read. The refusal names the
three targets rather than selecting one.

**Limb 2: label provenance, and the circularity check.** The label source becomes a declared,
typed input with an explicit provenance class: `record_derived_proxy` for tracked keys, and
`external_reference` for labels from a reviewed catalogue supplied from outside the record.
`_labels` becomes one implementation of that interface rather than the only path, and an
external-reference source consuming an existing `PatternCatalogue` through
`spectral_regions.py:match_into_catalogue` becomes the second. Its centroids, metric and radius
already come from the supplied catalogue and none of them moves, which is the property that
makes a held-out evaluation possible.

The pairing of target and evidence class is then checked, and this is the substantive limb.
`kind_recurrence` declared against `record_derived_proxy` labels must be **refused by name**,
because tracked keys are produced by the same record and pipeline whose identity is under test,
and a definition validated against them is validated against itself. Equally, `track_continuity`
against those labels must state in the receipt that a high score demonstrates agreement with
the tracker rather than independent identity. Neither refusal nor caveat may be suppressed by a
flag. This check is the one mechanism that would have caught the present confusion without a
human noticing it.

**Limb 3: the choice is surfaced, not buried.** The declared target, its evidence class, its
claim boundary and any refusal from limb 2 appear in the interface alongside the audit's
numbers, at equal weight to them. A refusal to evaluate a target on inadmissible labels is a
first-class result and renders as one. No accessibility level and no rendered-evidence claim is
made without its captured evidence, per the standing constraint.

**What this slice explicitly does not do.** It does not choose the target. It does not supply,
review or sign a catalogue -- code does not sign a scientific declaration for a person. It does
not approve a mining radius, alter the frozen mining declaration, change any threshold, window
or weight, or re-open the slice-2 verdict. It closes neither T4E.8 nor D97, D98, D99 or D100.
It converts an unaskable question into a blocked one, which is the only progress available while
the target is undecided.

**Acceptance.**

1. A design lacking `identity_target` is refused by name, before source values are read, with
   the three targets and their evidence classes stated in the refusal.
2. Each of the three targets round-trips into the receipt with its evidence class, its claim
   boundary and its provenance.
3. `kind_recurrence` against `record_derived_proxy` labels is refused by name, and the refusal
   states the circularity rather than a generic validation message. `track_continuity` against
   the same labels publishes its tracker-agreement caveat in the receipt.
4. **The slice-2 design, amended only by adding `identity_target: spatial_persistence` and
   `evidence_class: record_derived_proxy`, reproduces `t4e8-spatial-audit-v2-clean.json`
   bit-for-bit on every census, error, AUC, stratum and feasibility figure.** A declaration
   mechanism that moves a measurement is a defect in the mechanism.
5. The `external_reference` path is exercised end to end against a synthetic catalogue with a
   planted identity, recovering it, and against one whose family or scope does not match the
   signatures, refusing it.
6. The interface renders target, evidence class, boundary and refusal, with captured evidence.
7. Targeted mutation testing to the standard set by slice 2 -- every mutation of the new
   admissibility and refusal logic killed by a failing test, with no timeout or collection
   failure counted as a kill.

**Implementation outcome (2026-09-08).** Limbs 1 and 2 are delivered as specified.
`IDENTITY_TARGETS` and `EVIDENCE_CLASSES` are registries; `declare_identity_target` refuses an
absent target, an unknown name and every inadmissible pairing by name, and the audit calls it
before opening a source value. `catalogue_labels` opens the external-reference path through
`match_into_catalogue` and reports unmatched signatures as unlabelled rather than negative.
Acceptance 1, 2, 3, 5 and 7 are met by 21 tests and 12 of 12 killed mutations; acceptance 4 is
met exactly -- the v3 receipt reproduces `t4e8-spatial-audit-v2-clean.json` bit-for-bit at the
same frozen radius and verdict. `architecture.md` section 3E.9 describes the implementation.

**Acceptance 6 is not met, and the specification was wrong to assume it could be.** Limb 3
asked for the declaration to be surfaced in the interface. Nothing in `src/api` or
`frontend/src` reads identity-calibration receipts: the audit is a local-only CLI writing JSON,
and there is no view to extend. Delivering it means a new API endpoint over local receipt files
and a new view, which is its own task with its own acceptance, not a limb of this one. The
declaration is published in the receipt so a future surface has one authority to read.
**A second gap is recorded rather than worked around:** no serialisation for a signed
`PatternCatalogue` exists, so `tools/audit_spatial_identity.py` refuses any evidence class other
than `record_derived_proxy` by name. `kind_recurrence` is evaluable in the library and tested
there, and remains unevaluable from the tool until a reviewed catalogue and a format for it
exist. T4E.8 stays open and no radius is approved.

**Dependency note.** Slice 3 unblocks the maintainer's decision; it does not substitute for it.
Whichever target is chosen, `kind_recurrence` additionally requires a reviewed, cited, frozen
catalogue before it can be evaluated, and that catalogue is an input this programme must be
given rather than one it can generate. T4F.9 and T4F.6 remain gated on the decision and on
whatever validation the chosen target then demands.

**T4E.9 Certified synthetic ground truth for the identity step *(fixes D101, evidence for D97)* -- IMPLEMENTED 2026-09-08; the benchmark reports FAIL.**

*Every difficulty in T4E.8 traces to one absence: a real atmospheric record supplies no
replicates (D97), no honest null (D98) and no independent labels, so `kind_recurrence` waits on
a catalogue nobody has signed. This task asks whether the identity definition can recover an
answer that is known by construction, which requires no catalogue and no person's signature.*

`src/benchmarks/fields.py` already synthesises fields whose answers are derivable independently
of the analysis code: `S(k) ~ k^-beta` built in the Fourier domain has that exponent by
construction, and fBm carries a known Hurst exponent. The task is to run detection, tracking,
constellation extraction and the identity step over such a field under a declared design, and
measure whether identity recovers structure that is provably present -- with both error rates,
as T4E.6 requires, and a refusal rather than a number wherever the population is unmeasured.

**The external reference, and what admitting it would cost.** `certified-invariants`
(adamfbentley) works the same substrate from the other end: exact Fisher-information ceilings
and Le Cam minimax floors over Gaussian random fields with structured spectra, under
preregistered gates with a committed failure ledger. Its ceilings are a *candidate* certified
target for this task and its methodology is close to this programme's own. It is also what
Phase 4H already anticipates as an optional ceiling estimator. **Admitting an external
project's ceilings is a scientific choice and requires its own declaration**, not a silent
import: that project records certificate estimates occasionally exceeding their own ceilings,
flagged as estimator failures, which is the same species of problem as D97 and must be measured
here before any ceiling is treated as truth. Nothing about this task requires network access or
a second domain under R17; the existing local generators are sufficient to start.

**Claim boundary, recorded before anything is built.** This validates the apparatus, not the
atmosphere. A pass does not discharge T4E.8's acquired-record acceptance, does not approve a
mining radius, and closes none of D96-D100. What it settles is whether the identity definition
can recover a certified answer at all -- currently unknown, because T4E.7's synthetic check
plants replicates by construction rather than recovering an independently certified quantity.

**Acceptance.** A declared design fixed before measurement; both error rates reported or refused
by name against a target that is analytic rather than proxy-labelled; the recovered quantity
compared to its construction value with the comparison's own boundary stated; and an explicit
verdict including INVALID. A failure is a complete result and is informative about the
definition. No radius is approved by this task under any outcome.

**Outcome (2026-09-08): the benchmark reports FAIL, and the failure is the useful half.**
`src/benchmarks/identity_certification.py` registers `t4e_identity_certified`. It routes the
T4E path through the existing planted-motif scenes -- extraction, constellations, signing under
`spatial_geometry`, and `SignatureMetric` matching -- with construction labels recovered by
nearest planted position and **refused rather than guessed** when a position has no feature near
it or two positions claim one. Three disjoint seed blocks: calibration freezes the radius, and
neither evaluation partition informs it.

The identity definition separates perfectly: **AUC 1.0**, and **zero admissions across 11,985
different-configuration pairs** spanning the planted evaluation and the null partition. D101 is
closed by that measurement existing. What fails is the radius -- frozen at 0.006563 from 15
calibration pairs, it splits **46.7%** of evaluation motif pairs against a 10% bound, while a
feasible radius exists at 0.010801 giving 6.7% and 0%.

**So the two halves separate.** The definition can recover a certified answer; the
calibrate-at-a-recall-quantile-then-freeze procedure does not transfer at this support, with
perfect labels, total separation and no atmosphere. That is evidence about D97 isolated from
every atmospheric confound, and it names what the next slice of D97 has to address: not a
better record, a better estimator of the operating point. The FAIL stands rather than being
tuned away; no threshold, partition or weight was changed after the first run, and the only
amendment was the descriptive feasibility diagnostic. No mining radius is approved.

**T4E.10 An operating point that transfers *(the live half of D97)* -- IMPLEMENTED 2026-09-09; no estimator holds, and the reason redirects the defect.**

*T4E.9 measured the failure without the atmosphere in the way. With construction labels, total
separation and AUC 1.0, a radius calibrated at 90% recall on 15 cross-scene motif pairs and
frozen still split 46.7% of motif pairs in a partition it had never seen. So the problem is not
the record. `calibrate at a recall quantile, then freeze` is the wrong estimator, and this task
asks which estimator is right and whether the support the programme actually has can carry one.*

**Why the present estimator cannot work, stated before anything is built.** `recall_radius` is
an empirical quantile: r90 of 15 samples is the 14th smallest. What the acceptance needs is not
an estimate of the population's 90th percentile but a bound that will still admit 90% of pairs
it has never seen -- a one-sided nonparametric tolerance bound, not a quantile. Those are
different objects and the second is strictly wider. For the k-th smallest of n samples the
covered proportion is distributed Beta(k, n-k+1), so a bound covering proportion p with
confidence gamma exists only when some k satisfies P(Beta(k, n-k+1) >= p) >= gamma. Using the
maximum, that is 1 - p^n >= gamma, so **n >= log(1-gamma)/log(p)**. At p = gamma = 0.9 the
requirement is **n >= 22**, and T4E.9 calibrated on 15. No choice of order statistic fixes
that: the support was insufficient for the guarantee before the radius was computed.

**What this task builds.** A registry of declared operating-point estimators, so adding one is a
declaration rather than a fork, each stating what it guarantees and what it needs:
`empirical_quantile`, the present behaviour, kept so its failure stays measurable;
`nonparametric_tolerance_bound`, which **refuses by name when the support cannot carry the
declared coverage and confidence, and states the n that would**; and at least one estimator that
widens with uncertainty rather than refusing, so the refusal is a choice and not the only option.

**How it is measured.** Through `t4e_identity_certified`'s scenes, because the answer holds by
construction there and neither an atmosphere nor a catalogue is involved. Each estimator is
calibrated on one partition and applied unchanged to two it never saw, at **two supports**: six
scenes giving 15 cross-scene motif pairs, which is below the requirement, and eight scenes
giving 28, which is above it. Both error rates reported or refused by name; an explicit verdict
including INVALID.

**Acceptance.** An estimator whose declared error rate holds on a partition it never saw, at a
support the programme can actually reach, with the refusal fired at the support that cannot
carry it. **A refusal at 15 pairs is a pass for the estimator and a finding about the design**,
not a failure to be tuned away. If no estimator holds at either support, that is a complete
result and it redirects D96, because a radius that cannot be set is a workload that cannot be
predicted.

**Claim boundary.** This is measured on synthetic scenes and settles a property of the
estimator, not of the atmosphere. It approves no mining radius, discharges no part of T4E.8's
acquired-record acceptance, and closes neither D96 nor D97 on its own. What it can do is tell
this programme whether the operating point is estimable at the support a real record supplies.

**Outcome (2026-09-09): no estimator holds, and the acceptance's own premise is what failed.**
`src/analysis_engine/operating_point.py` registers the three estimators. The support arithmetic
holds exactly: 90% coverage at 90% confidence needs 22 observations, T4E.9 had 15, and
`nonparametric_tolerance_bound` refuses there naming 22 -- which is a pass for the estimator and
the finding about the design the specification anticipated. At 28 pairs it names a radius and
still fails, splitting 20.0% against a 10% bound, though that is less than half the empirical
quantile's 46.7%. Admissions are zero throughout.

The reason is exchangeability rather than support. Four blocks of six scenes from one generator
with identical parameters have mean same-configuration distances spanning **1.88x** (0.004575 to
0.008585), and T4E.9's calibration block is the tightest of the four. Calibration and evaluation
are not one population, so no distribution-free bound calibrated on one carries a guarantee about
the other at any support.

**So the live half of D97 is no longer "find a better estimator".** A frozen absolute radius is
the wrong object when partitions differ this much, and a real record's windows will differ more.
The remaining candidates -- per-partition calibration, a distance whose scale is comparable
across partitions, or an identity criterion that is not a radius at all -- are each a scientific
choice and each needs its own task and acceptance. This also reaches **D96**: its workload is set
by the radius, so a radius that cannot be frozen is a workload that cannot be predicted, and the
identity criterion must settle before the algorithm that consumes it does.

**T4E.11 An identity criterion that survives a change of partition -- IN PROGRESS. Candidates B and C both declared, adopted and falsified on development evidence, 2026-09-09. C produced the sequence's first positive finding: the ordering transfers where magnitudes do not. Candidate D was declared, adopted and falsified on 2026-09-09: the margin costs nothing in recall and does not reject, and the ratio distributions overlap.**

*T4E.10 closed off the estimator route. Four blocks of six scenes from one generator with
identical parameters differ 1.88x in mean same-configuration distance, so a frozen absolute
radius cannot transfer between partitions however well it is estimated. This task asks what
criterion can, and it is a change to what identity is rather than to how a number is computed.*

Three candidates. **Per-partition calibration**: each window sets its own radius, which is cheap
and honest but changes what a pattern is between windows, so recurrence across windows needs its
own argument and may become unstateable -- and recurrence across windows is what the mining,
sequence and precursor machinery exists to find. **A comparably scaled distance**: normalise the
metric so a radius means the same thing in every partition, by the partition's own distance
distribution or by a quantity the record supplies; this keeps one criterion but changes the
metric, which changes identity itself. **A criterion with no absolute scale**: a rank-based rule
such as nearest neighbour with a margin, or mutual nearest neighbours, which has nothing to
transfer; `src/analysis_engine/representation_alignment.py` already implements mutual k-NN and
its closed-form chance floor k/(n-1), written for a different purpose and reusable here.

**Acceptance.** Measured on T4E.9's scenes across at least four blocks, calibrated where
calibration applies and evaluated on blocks it never saw, with both error rates reported or
refused by name and an explicit verdict including INVALID. The criterion must hold on blocks it
was not calibrated on, which is exactly what every estimator in T4E.10 failed to do. **R20
governs the selection**: do not measure all three and keep the winner. Declare which criterion
is being adopted and why before evaluating it, and record an amendment openly if that changes.

**Claim boundary.** Synthetic scenes settle a property of the criterion, not of the atmosphere.
Nothing here approves a mining radius or discharges T4E.8's acquired-record acceptance. A
criterion that holds on synthetic blocks still has to be argued for on a real record, where the
partitions differ by more.

**Candidate D, declared 2026-09-09 and not yet adopted.** Candidate C's failure was named and
narrow -- it recovered every motif pair and could not decline -- so candidate D adds the one
thing it lacked, a rejection test, and adds it in a form that keeps what C had. A mutual
nearest-neighbour pair is kept only when its distance is at most `tau` times the distance to
the second nearest, in both directions. Both distances come from the same scene pair, so the
test is dimensionless and there is nothing to carry between partitions; this is not candidate B
in another costume, because B divided by an *estimated* partition scale and the estimate was
where it failed.

`tau` is pinned at **0.8**, the canonical nearest/second-nearest ratio from Lowe (2004), IJCV
60(2). It is taken for its external provenance -- it is not answerable to this record -- and
explicitly not because it is expected to be optimal here. A `tau` read off these
already-inspected blocks would be fitted and could not be reported as a criterion at all; the
declaration says so in advance and routes any such value to a candidate E evaluated on data
these blocks did not select. **Acceptance** is false split at most 0.10 on every block, null
retention at most 0.10 -- the fraction of candidate C's 116 null matches the margin keeps --
and both materially stable between blocks.

One property is derivable before measurement and was recorded before it: D is a strict
narrowing of C, so its false split can only rise from C's 0.0000 and its match count can only
fall. The whole question was whether that trade is close to free.

**Candidate D outcome (2026-09-09): adopted before measurement, falsified by it, and the
cleanest failure of the four.** The margin **cost nothing in recall**: false split stayed at
0.0000 in all four blocks, so every motif pair survives it, and the counts fell on every
block as predicted. What it discarded was entirely non-motif. It failed the one condition
it was declared against -- **null retention 0.5345** where acceptance required at most 0.10.
It removed 54 of candidate C's 116 null matches and kept 62 where the correct answer is none.

**The diagnostic answers the question the declaration asked, and the answer is the unwelcome
one.** Motif ratios run 0.020 to 0.293 across the four blocks; non-motif ratios reach 0.9997
and the null block's begin at 0.240. So the contrast is genuinely informative and `tau = 0.8`
is far too permissive for it -- but **the distributions overlap**, 0.240 against 0.293, and no
threshold separates them cleanly. The overlap is narrow, which is what makes it dangerous: a
`tau` chosen to sit inside it would be fitted to blocks now inspected four times over. That is
a candidate E and needs an evaluation on data these blocks did not select.

**Two limitations are recorded as limitations rather than results.** Retention was measured
against the single declared null partition, so its stability across nulls is untested. The
diagnostic stores extremes rather than distributions, so the *mass* of the overlap -- how many
motif pairs sit above 0.240 -- is not known from this run.

**The declaration diagnosed its own failure correctly, which the two before it did not.** D's
falsification field was narrowed deliberately in response to that pattern, to license only that
at `tau = 0.8` this contrast does not separate in these scenes. That is exactly and only what
the measurement supports. The narrowing was the right correction.

**What this does not authorise.** Moving `tau`. Adding a second rejection test to D and
re-measuring. Reading an operating point off the diagnostic. The confirmatory blocks remain
reserved; four falsifications have not unreserved them.

**The diagnostic that followed, 2026-09-09, and where it sends the programme.** Four candidates
had been falsified against a ground truth no declaration had examined. `decompose_matched_pairs`
asks what the false admissions are. It adopts nothing and reports no operating point.

| class | available | C matched | rate | D matched | rate |
|---|---|---|---|---|---|
| **motif** | 60 | 60 | **100%** | 60 | **100%** |
| shared_2 | 1620 | 89 | 5.49% | 54 | 3.33% |
| shared_1 | 1620 | 36 | 2.22% | 17 | 1.05% |
| **crossed_2** | 1080 | **0** | **0.000%** | **0** | **0.000%** |
| crossed_1 | 10800 | 170 | 1.57% | 80 | 0.74% |
| unrelated | 8820 | 175 | 1.98% | 93 | 1.05% |
| *null, unrelated* | 6000 | 116 | 1.93% | 62 | **1.03%** |

**The suspicion that prompted it was wrong.** Nine of twenty configurations per scene share two
motif features and therefore recur by construction, so the ground truth might have been scoring
real recurrence as error. It is not: only 29% of candidate D's non-motif matches hold the same
motif vertices, and that class is enriched just 3.2x over unrelated. **The ground truth is
sound.**

**What the signature does well is now measured too.** 60 of 60 motif pairs recovered. A
background rate that does not notice whether a motif is present -- 1.054% planted against 1.033%
null -- which is why the null's retention was never a null-specific artefact. And `crossed_2` at
**0 of 1080**: the full motif is never matched to a configuration holding two of its own three
features.

**The binding constraint is a number.** 400 candidate pairs per scene pair, one true positive:
prior odds 1:399. Candidate D's per-pair false rate is 1.019%, a specificity of 98.98%. The
declared 0.10 bound needs 0.028% -- a **36-fold** reduction. No threshold on these distances can
supply it, and that is the brief for **T4E.12**.

**T4E.12 What the signature must carry -- SPECIFIED 2026-09-09, NOT STARTED.**

*T4E.11 falsified four criteria and its diagnostic said why: the binding constraint is
specificity against combinatorial odds, not the placement of any threshold. This task asks what
the signature would have to carry instead, and its first act was to measure how the problem
scales -- because the acceptance criterion depends on the answer.*

**The scaling measurement (diagnostic, run 2026-09-09 before any criterion was declared).**
`measure_richness_scaling` varies the feature count, which `_motif_scene_positions` now accepts
as a parameter defaulting to the frozen six, so every existing caller builds exactly the scenes
it built before. Candidate D is used as a probe of the signature and is already falsified;
nothing is adopted and no operating point is reported.

| features | configs | candidate pairs | matches | split | admission | pair rate | required | shortfall |
|---|---|---|---|---|---|---|---|---|
| 6 | 20 | 6,000 | 70 | **0.0000** | 0.7857 | 9.19e-3 | 2.79e-4 | **x33** |
| 9 | 84 | 105,840 | 321 | **0.0000** | 0.9533 | 2.89e-3 | 1.58e-5 | **x184** |
| 12 | 220 | 726,000 | 720 | **0.0000** | 0.9792 | 9.71e-4 | 2.30e-6 | **x423** |

**Two findings, and they point opposite ways.**

*The signature keeps its recall completely.* False split is **0.0000 at every richness**: the
planted motif is still the mutual nearest neighbour when it is competing against 219 rival
configurations rather than 19. The information needed to identify the configuration is present.

*The rule matches a constant fraction of whatever it is given.* Roughly **a quarter** of
configurations are matched at every richness -- 0.233, 0.255, 0.218 -- because a
nearest-neighbour matching returns at most one pair per configuration. So false admissions grow
**linearly** with the configuration count while true correspondences stay at one per scene pair,
and the admission fraction climbs towards one: 0.786, 0.953, **0.979**.

**The consequence, which is why the acceptance below is not stated as an admission rate.** The
per-pair false rate does fall as scenes get richer -- a quarter of a growing population is a
shrinking fraction of its square -- but it falls as `1/m` where a fixed admission bound demands
`1/m^2`. The gap therefore *widens*: **x33, x184, x423**. An admission fraction is a property of
the signature and the scene richness together, never of the signature alone, so a bound met on
one record need not hold on a denser one. That is T4E.10's transfer problem again, in a place no
declaration had looked -- and it means **no rule whose match count scales with the configuration
count can succeed**, whatever threshold is placed on its distances.

**Acceptance.** A candidate is declared before it is measured and must satisfy all of:

*(Amended 2026-09-09, in the open, before any candidate was declared under it. As first written
this clause said a candidate is "a change to what the signature carries". That was too narrow
and the scaling measurement above is what shows it: what the measurement indicts is a decision
rule that does not know how many comparisons it is making, and multiplicity is a property of the
procedure rather than of the signature. Restricting candidates to signature changes would have
excluded the class of candidate the evidence actually points at. The four numbered conditions
are unchanged; only the admissible class of candidate is widened. Note in particular that a
change to the attribute weights -- which are all 1.0 and have never been calibrated -- CANNOT
satisfy condition 2 on its own, because a nearest-neighbour matching returns at most one pair
per configuration whatever the weights are, so reweighting changes which pairs match and never
how many.)*

1. **Recall unchanged.** False split at most 0.10 at every richness measured, individually.
2. **A match count that does not track the population.** The matched fraction of configurations
   must fall as richness grows rather than holding constant. This is the property every T4E.11
   candidate lacked and the one the scaling measurement identifies.
3. **A shortfall that does not widen.** Measured at **at least three richness levels**, the
   ratio of achieved to required per-pair false rate must be non-increasing, and the admission
   bound of 0.10 must be met at the richest level tested. Meeting it only at the sparsest level
   is the failure this task exists to prevent.
4. **Refusals by name**, and an explicit verdict including INVALID.

**What acceptance may not be.** An admission rate quoted at a single richness. A weight vector
or threshold calibrated on the four already-inspected blocks -- the attribute weights are
currently all 1.0 and have never been calibrated, which makes them an obvious candidate and
exactly the kind that must be declared first and evaluated on scenes that did not select it.

**R20 governs as it governed T4E.11.** Declare the change before measuring it, evaluate it
alone, and record a falsification as a result rather than trying variants until one passes.

**Claim boundary.** Synthetic scenes at several densities settle a property of the signature,
not of the atmosphere. Nothing here approves a mining radius, discharges T4E.8's acquired-record
acceptance, or closes D96, D97, D98, D99 or D100. The primary target remains `kind_recurrence`,
which needs `external_reference` evidence no signature change supplies. The reserved
confirmatory blocks stay reserved.

**Candidate 1, declared 2026-09-09 and not yet adopted.** A rule that counts its own
comparisons. Candidate C proposes the pairs; a pair is admitted only when a distance as small as
its own would arise with probability at most `alpha / m^2` under a tail model fitted to that
same scene pair's own cross-scene distances. `alpha = 0.05` is a declared family-wise error rate
-- the expected false admissions per ordered scene pair if the model holds -- and not a tuned
threshold. The bound therefore tightens by construction as scenes get richer, 1.25e-4 at six
features to 6.9e-8 at twelve, which is the `1/m^2` scaling the acceptance demands.

**Why not the attribute weights, which are all 1.0 and were never calibrated.** They cannot
satisfy condition 2, and that is derivable rather than measurable: a nearest-neighbour matching
returns at most one pair per configuration whatever the weights are, so reweighting changes
which pairs match and never how many. Declaring it would have spent a preregistration to learn
something available on paper.

**The tail model is peaks-over-threshold** -- the 1st percentile of the scene pair's own
distances as threshold, a generalised Pareto on the exceedances below it -- because
extreme-value theory is what licenses stating a probability of 1e-7 from hundreds of thousands
of samples. A thin population, too few exceedances or a non-converging fit is **refused by
name**: admitting on a failed fit would make the rule most permissive where its model is least
supported. Three assumptions are declared rather than discovered later: the null is contaminated
by the true correspondences it contains, the generalised Pareto is asymptotic theory on a finite
sample, and configurations sharing features are not independent, so the correction is a working
one rather than an exact guarantee.

**A diagnostic separates two failures that would otherwise be confused.** If the tail model is
right, admissions per scene pair equal `alpha` by construction whatever the signature is like.
So the null blocks' admission count tests the tail model rather than the signature: near 0.05
means the model holds and any remaining failure is the signature's; far above it means the fit
is wrong here and the signature has not been tested at all.

**The falsification field is narrowed in the same way candidate D's was**, and for the same
reason: T4E.11's candidates B and C both over-reached there, D's was narrowed in response and
proved correct. A failure here licenses only that a correction of this form, with this tail
model, at this `alpha`, does not separate recurrence from coincidence in these scenes -- not
that multiplicity correction is the wrong idea, and not that the signature is adequate or
hopeless. **Candidate 1 outcome (2026-09-09): adopted before measurement, falsified by it, and it
did not test the signature.** It admitted **nothing**: not one pair across four blocks at nine
features, four at twelve, or either null partition. False split is 1.0 wherever it could be
computed, against an accepted 0.10.

**Richness 6 was never evaluable and that was derivable before adoption.** Fifty exceedances at
a 1st-percentile threshold needs at least 5,000 distances, hence 71 configurations, hence nine
features; six features give 400 distances and four exceedances. Every scene pair refused by
name, at a richness level the declaration itself listed.

**The null carries the reading that matters, and the declaration fixed it in advance.** If the
tail model held, admissions per scene pair would equal `alpha` whatever the signature is like.
Nominal 0.05, measured **0.0000** -- the fitted model under-admits relative to its own nominal
rate. So the failure is the model's and **the signature was not cleanly tested**. The
declaration's diagnostic named "far above alpha" as indicting the model and never named "far
below", which is the direction that occurred; that gap is recorded rather than edited away.

**How short, and what is not claimed.** The motif's fitted tail probability sits 9.1x, 6.0x,
3.5x and 15.3x above the bound in four sampled scene pairs -- a single-digit to low-double-digit
factor, not orders of magnitude. **No trend in richness is claimed**: one block improves with
richness, the other worsens, on two blocks.

**The lesson, which is about drafting rather than about the atmosphere.** Candidate D taught
that the falsification-licensing field must be narrow, and that held here. This candidate
exposes a missing check: a **feasibility test** before adoption -- can the criterion admit
anything at all, in principle, at the support declared for it? Both failures above answer no and
both were available on paper.

**A condition found and named while measuring.** At twelve features a few configurations are so
nearly isotropic that `sign_constellations` refuses their principal axis, so they carry no
bearing block and `SignatureMetric` refuses to compare them with anything that does -- correctly,
because distances between different measured quantities are not numbers. Extent: none at six or
nine features, 1, 1 and 2 of 1,320 in three of four blocks at twelve. `comparable_subset`
refuses them by name and counts them in every row, and refuses outright if the motif itself
falls in the minority family. **This also qualifies the scaling measurement above**, which ran
on block 100-105 alone and would have failed on three of the four blocks at twelve features: its
evidence base is one block per richness level, narrower than first written.

**What this does not authorise.** Raising `alpha`, moving the threshold percentile or the
exceedance minimum, or dropping richness 6 from the declared levels. Each is tuning against
blocks inspected many times over, and each is a further candidate needing its own declaration
and evaluation on data these blocks did not select. The reserved confirmatory blocks remain
reserved; five falsifications have not unreserved them.

**Candidate 2, declared 2026-09-09 and not yet adopted: identity as consistency across the
partition.** All five falsified candidates asked the same question -- given one pair of
configurations, is it a match? None used what the partition offers. The motif is in every scene,
so its mutual nearest-neighbour matches form a complete graph, while a coincidence between two
scenes has no reason to extend to the rest. A set of configurations, at most one per scene, is
**consistent** when every member is the mutual nearest neighbour of every other; only pairs
inside sets spanning at least `k` scenes are admitted, and `k` is the partition size.

**It carries no radius, ratio, threshold, fitted model or estimated normaliser.** So there is
nothing to transfer between partitions, nothing to estimate wrongly, and nothing whose support
can run out -- it is evaluable at richness 6, where candidate 1 refused every scene pair. The
computation is **exact rather than greedy**: mutual nearest-neighbour gives at most one partner
per scene, so the candidate group is determined and the largest agreeing subset is enumerated
outright.

**The feasibility check candidate 1 lacked was done first and is disclosed in the declaration.**

```
clique sizes reached by mutual-nearest-neighbour consistency, six-scene partitions
                        motif          non-motif spread
100-105 rich 6     6,6,6,6,6,6    1:21  2:53  3:32  4:8
100-105 rich 9     6,6,6,6,6,6    1:55  2:249 3:150 4:44
300-305 rich 6     6,6,6,6,6,6    1:7   2:51  3:41  4:15
300-305 rich 9     6,6,6,6,6,6    1:56  2:245 3:141 4:51 5:5
NULL    rich 6              --    1:22  2:51  3:46  4:1
NULL    rich 9              --    1:47  2:260 3:156 4:41
```

The motif reached 6 in every scene of every block; nothing else exceeded 5; the null reached 4.
So the criterion can admit the answer and reject the null -- it is not inert. It does not
establish the error rates, and the probe's greedy walk makes its sizes a lower bound rather than
the criterion's own output.

**The probe informed `k`, and the declaration says so plainly.** "Consistent across every scene
of the window" is defensible a priori and is the strictest setting available, so it is not a
number tuned to a result -- but it was chosen after learning that coincidences top out at 5.
Development evidence for this candidate is therefore weaker than for its predecessors, which is
exactly why confirmatory evidence would be worth more here than at any earlier point.

**Three limitations, declared now.** `k = S` will not transfer to an acquired record, where a
real pattern need not appear in every window and this rejects a configuration absent from one
scene with no partial credit. It inherits the mutual nearest-neighbour rule's assumptions. And
consistency within one partition says nothing about recurrence across partitions, which is what
the mining machinery downstream needs.

**A clause the earlier declarations did not need.** Success is a live possibility for the
first time, so the declaration states what a success would NOT license: not a mining radius, not
T4E.8's acceptance, not D96 to D100, not anything about the atmosphere, and not a confirmed
result -- which needs the reserved blocks, and opening those is a separate decision requiring
its own amendment.

**Candidate 2 outcome (2026-09-09): adopted as a development experiment, and it passes.**

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

### T4E.36 - declare the eastward wrapped acquisition

**T4E.36 (2026-09-13): acquired, materialised and measured after Ed Bentley's exact adoption;
verdict `FAIL`.** The
complementary ERA5 pressure-level request is fixed at relative vorticity, 850 hPa, 2018-2021,
four synoptic hours, latitude -58..-18 and longitude -180..-140. The existing 140..180 record
stays byte-for-byte unchanged; after wrapping, the two records span 140..220 with the shared
180-degree meridian verified and retained once. The production planner resolves the complement
to 48 monthly shards, 5,844 frames and a 161x161 grid without network access.

The same 18 storm-times, catalogue radii, negation, extractor, 99 surrogates, seed 1234 and both
representations are frozen. The reproducibly better raw-field path is primary and must meet both
unchanged T4E.18 gates at 9 of 18; SWT is diagnostic and cannot rescue it. A failure therefore
falsifies added eastward support as a sufficient repair under this procedure, not as a possible
contributor. The declaration adoption is bound to SHA-256 `5b96519a…`; separate experiment and
general network authorisation were subsequently supplied for this one execution.

The four-phase CLI now makes that boundary executable. Planning remains offline. Acquisition
requires a current digest-bound maintainer adoption, an experiment-specific
`--authorise-network` flag, and the CDS layer's general network opt-in. Materialisation verifies
all parent and complement shard receipts, measures the encoding lattice per GRIB field, refuses
an unmeasurable or mismatched seam, retains 180 degrees once, streams a 321-longitude union into
a content-addressed Zarr, and verifies that parent bytes did not change. Evaluation reads the
published T4E.28 rows directly instead of reopening the catalogue, reports old and widened full
distance lists, and lets only the two raw 9-of-18 conditions set the verdict. Six synthetic tests
cover those contracts without reaching a network.

The production execution acquired all 48 monthly complement shards: 5,844 frames and
325,654,005 bytes. All 48 source-encoding seam checks had zero maximum absolute difference; the
parent was preserved byte-for-byte and the 321-longitude wrapped Zarr was published at logical
SHA-256 `9e9b7e50…`. Evaluation held the population at exactly 18. Raw condition 1 improved from
3/18 to **5/18**, while condition 2 remained **0/18**; both required 9/18. Median raw nearest
distance improved 52.15 to 40.83 km and the maximum improved 3,685.34 to 315.09 km. SARAI and
GRETEL moved inside their radii; JOSIE improved to 151.29 km but remained just outside its
145.23 km radius. The result is useful but negative: edge support contributed to several errors,
yet did not repair the declared join. No further widening, extractor adjustment, tolerance,
identity criterion or atmospheric claim follows from it without a new declaration.

### T4E.37 - distinguish field/reference displacement from extractor filtering

**T4E.37 (2026-09-13): adopted and measured; `FIELD_REFERENCE_SEPARATION_DOMINANT` 12/13;
`VERDICT: NOT_AN_ACCEPTANCE`.** The exact 13 raw condition-1 failures from T4E.36
are frozen by identity and by the hashes of the wrapped measurement and record receipts. Passing
rows are excluded, the catalogue is not reopened, and no forecast-test data is touched.

The diagnostic exposes the existing extractor's own pre-threshold candidate rule: finite native
grid samples equal to the maximum of their 3x3 neighbourhood, with its one-cell boundary exclusion
and tied maxima retained. It introduces no new distance tolerance, interpolation, fitted centre or
alternative extractor. A sampled maximum inside the unchanged agency radius but no published
feature is `EXTRACTOR_FILTERING_CANDIDATE`; no sampled maximum inside is
`FIELD_REFERENCE_SEPARATION_CANDIDATE`. Seven of the 13 binary rows names the dominant candidate
mechanism. The unchanged extraction must first reproduce every T4E.36 feature count and sorted
distance or the diagnostic refuses.

This majority is deliberately not an acceptance verdict. A raw sampled maximum is neither a
significant feature nor an identified cyclone, and absence of one inside the radius does not prove
that pressure-level/surface quantity separation caused the offset. The executor therefore reports
`VERDICT: NOT_AN_ACCEPTANCE`, cannot rescue T4E.36, and changes no crop, extractor, tolerance,
mining radius or identity criterion. The separate human adoption binds SHA-256 `2a0f0ce5…`.

The offline execution reproduced every T4E.36 raw row before attribution and published receipt
SHA-256 `f8ff2f88…`. Twelve failures had no native-grid local maximum inside their inherited agency
radius before thresholding. JOSIE alone had two, at 37.99 km and 114.14 km inside its 145.23 km
radius, and both cleared the calibrated threshold even though no final extracted feature was
inside. Nearest sampled maxima across the 13 ranged from 26.12 to 177.98 km, median 44.44 km.

LINDA's inherited radius is zero, so its classification is structurally true but physically
uninformative. A post-result positive-radius sensitivity remains 11/12 field/reference candidates
versus 1/12 extractor filtering and cannot change the declared majority; it is not a replacement
gate. The result rules out extractor filtering as the dominant explanation on these rows, not as
a contributor. It does not establish that vertical pressure/surface quantity separation caused
the offsets. That causal question now requires an independently grounded reference-alignment
design rather than tuning this extractor on inspected development rows.

### T4E.38 - exact-time MSLP reference-alignment acquisition

**T4E.38 (2026-09-13): adopted, acquired, materialised and measured;
`VERTICAL_QUANTITY_SEPARATION_DOMINANT` at 10/18; `NOT_AN_ACCEPTANCE`.** This follows the T4E.18 design's
pre-existing handoff: MSLP is more commensurate with IBTrACS central pressure than 850 hPa
vorticity, but required extending CDS ingress beyond the pressure-level product.

The new single-level request is product-specific and admits only `msl`; it cannot carry a pressure
level. It binds the exact 18 T4E.36 timestamps and emits one request per timestamp in each of the
140..180 and -180..-140 segments. The offline result is 36 shards, two 161x161 samples per row,
and 3,732,624 raw float32 bytes. A one-time request avoids the day/hour cross-products that would
silently expose undeclared frames. Every shard is dataset-hashed, downloaded through an atomic
partial file, validated as NetCDF and byte-checked again on resume.

The later analysis is frozen now: deterministic 3x3 descent from the catalogue-seeded cell to an
MSLP basin minimum, and ascent from the same seed to a negated-vorticity basin maximum. Comparing
catalogue distances introduces no search radius. Ten of 18 MSLP-closer rows names vertical quantity
separation as the dominant candidate; ten vorticity-closer rows names catalogue/reanalysis
alignment; otherwise the outcome is `MIXED`. It is never an acceptance and identifies neither
centre as a cyclone. The adoption binds SHA-256 `0ad4e537…`; adoption itself made no request.
Ed Bentley later supplied the separate experiment authorization with the ordinary network gate
enabled. Acquisition completed 18/18 parent and 18/18 complement shards, totalling 2,326,213
compressed bytes. Every file matches its receipt digest and reopens with one `valid_time`, a
161x161 grid and only `msl`; the segment receipts hash to `35b137ac…` and `0100831c…`.

Materialisation reverified every source byte, found zero seam difference on all 18 timestamps
against a measured 0.0625 Pa source-encoding step, retained 180E once and published the
18x161x321 record at `dc6f5365…` with receipt `4acba0aa…`. Every declared path completed. MSLP was
closer to the catalogue position on 10 rows, negated vorticity on seven, with one exact tie and no
refusals. The pre-acquisition threshold therefore names `VERTICAL_QUANTITY_SEPARATION_DOMINANT`;
measurement receipt `9906ef4e…` replayed exactly. This is not proof of vertical separation, does
not identify a cyclone centre or rescue T4E.36, and changes no extractor, radius, tolerance or
acceptance gate. A further atmospheric slice requires an independently declared holdout design
before opening 2022–2023.

### T4E.45 - the extremes test that refuted the shape it was written to support

**T4E.45 (2026-09-14): T4E.44's recommended derivation, performed. The small end works; the large
end is refuted by measurements already on disk, and with it the simple form of the
recommendation.**

T4E.44 left two shapes for a size-scaled criterion and recommended Shape B -- take the scaling
from the matching graph's own combinatorics rather than from a surrogate null -- explicitly
because its behaviour at the extremes could be derived on paper before adoption rather than
measured after. That was the whole argument for it, so the derivation had to come before the
candidate.

Under independent edges of the observed density,
`E_m = C(S, m) * c**m * p**C(m, 2)` for a one-per-scene set of size `m`. The density is candidate
2's own published count -- 121 pairs proposed of 6,000 candidate pairs at richness 6 -- and at
`m = 2` the expression returns that edge count, which is how the arithmetic is anchored rather
than assumed.

The small end works. A pair is expected by the hundreds at every recorded density, so no pair can
be evidence, and candidate 4's defect -- a group of two admitted on the same terms as a group of
six -- is excluded by arithmetic instead of a special case. The expectation then collapses from
121 at `m = 2` to 2.4e-18 at `m = 6`, because the count grows as `c**m` while the probability
falls as `p**(m(m-1)/2)`. Size scaling with no free parameter.

The large end refutes the null that makes that tractable. Independence predicts coincidental
five-cliques at 6.7e-13 and 7.9e-15 at richness 9 and 12. Candidate 3 measured them in six of
twelve development blocks, at false admission 0.4 rising to 0.8. Independence understates by at
least 7.5e11 and 1.3e14, anti-conservatively. A nearest-neighbour matching is transitive by
construction, so cliques are far more common than independent edges of the same density allow.

The cost was arithmetic over counts already published, and it refuted a recommendation before
anything was declared, adopted or measured. The open question narrows: not whether the graph has
an estimable chance structure, but whether its transitivity can be estimated from a partition
containing the motif whose clique is under test without the estimate being set by the signal --
which is K1 in another form, and suggests the two shapes may share one obstacle. Nine tests,
including one requiring the refuting half to be reported rather than the flattering half.

### T4E.44 - what a size-scaled criterion must survive

**T4E.44 (2026-09-13): the derivation C6 needs before a ninth candidate. Six constraints, no
candidate, no reserve opened, nothing measured on any partition.**

Ed Bentley adopted the T4E.41 acceptance bar, which unblocks declaring a criterion against it, and
C6 -- evidence that scales with the group -- is where the eight falsifications converged. The
obvious construction has already been tried and killed by derivation: T4E.16's candidate 5 set the
bar per size with a within-partition permutation surrogate, and that surrogate can reassemble the
very set it is testing. Its own conclusion was that a feasible null is invalid here and a valid
null is infeasible. A ninth candidate written before that bind is mapped walks into the same wall.

The recorded figure is recomputed rather than restated. A redistribution surrogate rebuilds the
motif whenever its S copies land in S distinct scenes -- `S!/S**S`, 720/46656 = 0.015432 at S = 6,
0.9547 over the declared 199 replicates -- against T4E.16's recorded 0.0154 and its simulations of
0.0153 to 0.0168. The closed form is checked against a 40,000-trial simulation, difference
0.000168.

Six constraints are published, each bound to what established it: a surrogate must not be able to
reconstruct the set it tests; a null that changes pairwise distances per replicate is not runnable;
those two together admit only a pool excluding the set under test or no per-size surrogate at all;
a bar calibrated on one partition may not be applied to another; the partition size stays at six;
and a ninth candidate inherits eight attempts' multiplicity and opens neither reserve.

Two things are published with their costs rather than left to be discovered. The defect falls away
fast with partition size -- 0.9547 at S = 6, 0.0106 at 12 -- so the incentive to widen S is real,
and the prohibition is printed beside the numbers that make it tempting. And the one repair that
keeps the design runnable, excluding the set under test from the surrogate pool, drives reassembly
to zero by construction while biasing the bar toward false admission, because the excluded members
are the partition's tightest group. A candidate using it owes that bias quantified before adoption.

This is a derivation and is not evidence for C6. Registering it there would be a category error:
C6 asks for a criterion, and this is the set of limits one would have to satisfy. Eight tests.

### T4E.43 - a long measurement you can watch and stop

**T4E.43 (2026-09-13): the durable-run machinery generalised for long measurements.** V1
criterion 1 asks that a researcher can compose, adopt, run, cancel or resume, inspect and review
a *long* scientific measurement. The machinery already did the hard parts -- content-addressed
steps, a journal the state is folded from, completed work replayed rather than repeated, a
transition table no code path can leave. It could not be interrupted.

`execute` ran every declared stage to completion whatever the journal said, so `cancel` only ever
reached a run that was not running. And the route was `async def` around that synchronous work, so
a long run sat on the event loop and the API stopped answering -- `/progress` and `/cancel`
included, the two routes that exist for precisely this case. The same shape T4E.42 measured on the
convening route, here in the machinery meant to carry every long measurement.

The repair is small because the design was right: state is folded from the journal on every read,
so a cancel written by another instance is already visible to an executing run. It only needed to
be looked at. Both routes are now plain `def`, threadpooled by FastAPI, and both check for a
cancellation before each component and at each stage boundary -- per component rather than per
stage, because a stage whose components each take an hour is exactly the run someone needs to
stop. Completed components keep their outcome and digest, so paid work survives a stop.
`CANCELLED` remains terminal: re-executing does no work and returns the receipt, because a stop
is a decision rather than a pause.

Six tests hold it and the existing eighty run tests are unchanged.

### T4E.42 - a convening nobody asked for

**T4E.42 (2026-09-13): a browser test convened a real paid panel, and the handler that ran it
froze the API while doing so. Both defects fixed; the paid record preserved outside the study.**

`adoption.spec.ts` held a test called *"convening without a key on the server refuses and says
nothing was sent"*. It ticked authorisation and clicked convene, expecting the no-key refusal.
That was an assumption about the machine rather than a fact the test established: `.env.local`
here supplies `GEMINI_API_KEY`, so the click convened eight seats, made paid calls on the
maintainer's account, put the review bundle in front of a third party, and recorded a complete
panel for a study whose history said no valid panel had ever completed.

It survived several runs because `convene_round_robin` was `async def` over a blocking transport
with an 86,400 second default timeout. The call sat on the event loop, the whole API stopped
answering including `/health`, and every panel in the browser suite rendered its loading line.
The symptom looked like a broken frontend; three "pre-existing" browser failures were this, and
they pass once the wedge is gone.

The route is now a plain `def` that FastAPI runs in a threadpool, with the reason written beside
it. The browser suite's backend clears both key variables, and because `reuseExistingServer`
leaves an existing server's environment alone, the test also reads `key_present` from the panel
plan and skips with a stated reason instead of clicking convene to find out. A companion test
drives the authorisation refusal, decided before any key is read and safe on any machine.

The paid record is archived byte-for-byte under
`data/superseded/t4e42_unintended_panel_20260913/` with a README stating how it came to exist.
Nothing paid for is discarded (R23), and it is kept out of `data/reviews/` because the provenance
of that finding is a test harness rather than a person. Five backend tests hold the route's
synchronicity, the pre-key refusal, key presence reported without the key, an unrelated request
still being served during a blocking call, and the archive staying out of the study record.

### T4E.41 - T4E.8's acceptance, made decidable

**T4E.41 (2026-09-13): the bar written instead of a ninth candidate. Eight conditions, bound to
the record they were set against, computable per condition, and unacceptable by software.
`INSUFFICIENT_EVIDENCE` at 0 of 8; the declaration is `DRAFTED_NOT_ADOPTED`.**
`data/identity_calibration/t4e41-identity-acceptance-declaration.json`,
`src/core/identity_acceptance.py`, `tools/run_t4e41_identity_acceptance.py`,
`frontend/src/components/IdentityAcceptance.tsx`, `frontend/e2e/identity-acceptance.spec.ts`.

Eight criteria have been declared before measurement and falsified by it, and the catalogue
thread ended at a deterministic majority carrying `NOT_AN_ACCEPTANCE`. Throughout, T4E.8's
acceptance stayed one sentence of prose whose every term had since acquired a measured meaning.
Writing a ninth candidate against that sentence would spend evidence that cannot be recovered
without being able to say afterwards whether it had passed.

Each condition states what it requires, what would license it, and what is explicitly not
sufficient. The third clause is where this programme's own results go, by name: the false
rejection rate alone (D97), a rate that is arithmetic rather than evidential, a deterministic
majority over a finite census (T4E.38 and T4E.39), the phase surrogate that produces a quarter
of the record's signatures (D98), `strengths` at AUC 0.5053 (D100) and detector band in the
comparable vector at a 60.8% change rate (D99), the admission fraction the T4E.12 amendment
replaced, the fixed radius T4E.10 measured cannot transfer, closure with no size term (T4E.15),
and R20's forbidden move of measuring several candidates and keeping the winner.

Evidence reaches a condition through an adopted register entry that names the measurement and
binds both digests. Code recomputes the digest and reads the signature; it does not judge the
science. An unadopted claim counts as none, adopted evidence stops counting when its measurement
changes, and a registered `NOT_MET` decides its condition rather than being dropped for a passing
sibling. There is no path to `T4E8_ACCEPTED`: the furthest state code reaches is
`ALL_CONDITIONS_MET_AWAITING_HUMAN_ACCEPTANCE`, and a test registers adopted evidence for all
eight conditions and adopts the bar itself to prove it.

The panel renders every condition with `NO_EVIDENCE` at the weight of a met one, the seven bound
artefacts with their verification state, and the six results that do not discharge the
acceptance. The bar is signed where it is read, through the existing declaration-signing
endpoint rather than a second adoption path, with the whole declaration on the panel so that
`I have read this declaration` is honourable at the point it is typed.

Two things the signing path settled. A refusal was found reaching the DOM for two renders and
then vanishing, because the panel kept it in the state a successful reload clears; refusals now
hold their own state and render as `role="alert"`, and the same repair was made to the holdout
review and the G17 review. That is PLAN section 5's second gap, found in a panel written to
close it. And a single-maintainer instrument no longer asks for the same name and role at every
adoption: `useSignerIdentity` remembers those two strings per browser, written only after an
adoption succeeds. The affirmation is excluded by construction, and so is the reason, which
belongs to one decision.

Eleven backend tests and six browser tests hold both halves. Nothing was adopted by code, and
the bar awaits a named person.

### T4E.40 - the holdout, rendered and reviewable

**T4E.40 (2026-09-13): the complete T4E.39 flow renders in the platform UI, every stage artifact
is examinable, and the flow ends in a human review that cannot become an acceptance.**
`src/core/holdout_review.py`, `src/api/reference_holdout.py`,
`frontend/src/components/ReferenceHoldout.tsx`, `frontend/e2e/reference-holdout.spec.ts`.

Until this slice the atmospheric holdout existed only as files and captured terminal output. A
result only its author can read is not independently checkable, whatever its receipts say. All
eight performed stages now render in order -- declaration, adoption, census, field plan,
acquisition, materialisation, measurement, review -- each naming the party that performed it, its
status, its digest and the boundary that governs it.

Nothing displayed is taken on trust. The surface recomputes the declaration file digest and every
artifact's receipt identity from its own content, requires the census to bind the declaration, the
field record to bind both and the measurement to bind all three, re-plans the four exact field
requests offline on every read and refuses the response if the recomputed shard count no longer
matches what was acquired. A measurement that has lost its mandatory `NOT_AN_ACCEPTANCE` verdict is
refused even when its receipt has been re-sealed around the change.

Refusals are rendered at the weight of values: exact ties and refused rows are tiles beside the two
candidate counts, and the one-row fragility of the 7-of-12 majority is stated on screen rather than
left for a reader to compute. Two operations are refused in the UI and say why -- acquisition
reached a network under a separate named authorization and stays a CLI job, and re-running the
holdout would manufacture a second one from a spent period.

The flow ends in `t4e39-holdout-result-review/v1`. Its permitted decisions are `BOUNDARY_SOUND` and
`BOUNDARY_DISPUTED`; it is structurally incapable of expressing an acceptance, and a review whose
bound digests, outcome, denominator or majority drift from the live result is refused and rendered
as stale. Writing and adopting remain separate acts, and adoption requires the exact typed
affirmation. Eight backend tests and four browser tests hold both lines; no review was written by
code, and the review remains `NOT_WRITTEN`.

### T4E.39 - temporal reference holdout

**T4E.39 (2026-09-13): declared before opening, adopted, censused, acquired, materialised and
measured; `HOLDOUT_SUPPORTS_VERTICAL_QUANTITY_CANDIDATE` at 7 of 12, with `NOT_AN_ACCEPTANCE`.**
Ed Bentley adopted the exact declaration digest `fc7ee60f…` as `ADOPTED_FOR_GUARDED_HOLDOUT_CENSUS`
before any holdout identity was parsed. The declaration binds the exact T4E.38 result and signed
IBTrACS identity; the offline plan verified those bytes and explicitly reported catalogue rows
false, identities false, ERA5 false and network false.

The selection was frozen first: 2022–2023, synoptic hours, −58..−18 latitude, 140..180 catalogue
longitude, two agency fixes, a two-degree interior and the existing deepest-per-storm rule. NATURE,
intensity, development pass/fail status and eventual centre distances cannot select rows. Fewer
than 10 selected storms is `INSUFFICIENT_HOLDOUT_POPULATION`; otherwise all selected storms form
the denominator and strict majority is `floor(n/2)+1`, with exact ties and refused paths retained.

The adoption-guarded census then read the signed catalogue: 76,784 rows, refusing 75,469 outside
the window, 405 outside the box, 650 off synoptic hours and 109 with too few agency fixes. Twelve
storms were selected from the disclosed thirteen in-box storms — above the minimum — so the strict
majority was fixed at 7 before any field was opened.

Ed Bentley then supplied the separate T4E.39 network authorization. Guarded acquisition took 48
exact-time shards across four streams — ERA5 single-level `msl` and pressure-level 850 hPa `vo`,
each over the 140..180 parent and wrapped −180..−140 complement — totalling 3,640,750 stored bytes.
A new exact pressure-level client mirrors the T4E.38 single-level guarantees: one timestamp and one
level per request, atomic content-checked storage and resume without re-download. An independent
pass re-verified all 48 digests and reopened every file at the declared 161x161 geometry.

Offline materialisation passed all 24 seams at exactly 0 difference, retained 180E once and
published the immutable 12x161x321 two-field record at `179aa8c6…`. The unchanged T4E.38
catalogue-seeded basin rule returned 7 MSLP-closer rows, 5 vorticity-closer, no ties and no
refusals; independent replay reproduced the value and receipt exactly.

The majority is the smallest the rule admits: one row changing side would have returned
`HOLDOUT_MIXED`. That is part of the result, not a licence to re-select or reweight. The outcome
says only that the deterministic T4E.38 MSLP-closer majority recurred on the pre-declared holdout
under the identical finite-census rule. It is not statistical generalisation, independence from
assimilation, cyclone identity, vertical tilt or pressure/surface separation; it validates neither
IBTrACS nor either field, approves no radius or extractor change, and rescues neither T4E.36 nor
T4E.8. The frozen design is now spent: the reserved period has been opened once, as declared, and
cannot be reopened as a fresh holdout.

### TG17.15 slice 6 - freeze the calibrated successor declaration

**TG17.15 slice 6 (2026-09-12): declaration frozen; release still refused.** The separately
versioned `g17-scale-shape-successor/v1` declaration now binds the per-correspondence estimand,
exact pool substitution, six-member Benjamini-Yekutieli family, derived pool minimum of 48,
orientation and calibrated marginal admission contract. The release gate reads this declaration
beside the unchanged historical joint-reassignment manifests, so `declared_inference` is no longer
a blocker. Its real-record inventory remains empty and `UNRESOLVED`, with exchangeability
`NOT_ESTABLISHED`; `pool_exchangeability_on_real_records` therefore remains blocking and G17 stays
`NOT_RELEASEABLE`. No real record identity or curation conclusion was manufactured by this slice.

### TG17.15 slice 7 - measure real-pool inventory readiness

**TG17.15 slice 7 (2026-09-12): bounded refusal recorded.** A deterministic repository audit now
searches the declared real-record roots for explicit partner-profile documents,
validates every document as a single-record `RecordProfile`, rejects malformed files and duplicate
record identities, and binds its result to the successor declaration. The current result is
`NO_INVENTORY`: **0 profiles, 48 required, shortfall 48**. The release gate renders those figures
inside `pool_exchangeability_on_real_records` instead of leaving “unresolved” unactionable.

Even 48 profiles yield only `READY_FOR_CURATION_REVIEW`; the audit always reports exchangeability
as `NOT_ASSESSED`. Marginal availability is necessary to begin admission and scientific review,
not evidence that records are interchangeable. The empty result is likewise absence of evidence,
not evidence of non-exchangeability. No source was acquired and no record identity was invented.

### TG17.15 slice 8 - bind partner profiles to real source bytes

**TG17.15 slice 8 (2026-09-12): ingress contract implemented; inventory remains empty.**
`build_g17_partner_profile.py` accepts one local delimited record and an explicit declaration,
binds the artifact to the exact source SHA-256, and derives only storage-level facts: finite sample
count, median cadence and coverage against an explicit declared window. Native scale, effective sample size and noise
floor remain explicit scientific marginals and each requires a named basis or method. The output
is immutable `correspondence-record-profile/v2`; readiness now refuses legacy unbound profile JSON,
duplicate provenance identities and malformed source metadata.

This creates a legitimate path for actual records without choosing a source, acquiring data,
inventing identities or blessing a marginal-estimation method. The regenerated repository audit
is byte-unchanged at `NO_INVENTORY`, 0 of 48, with exchangeability `NOT_ASSESSED`. Reaching 48
source-bound profiles opens human curation review only.

### TG17.15 slice 9 - require adopted marginal methods before profile emission

**TG17.15 slice 9 (2026-09-12): reviewed-method boundary enforced; inventory remains empty.**
The profile CLI now requires a `g17-marginal-method-review/v1` declaration whose exact bytes are
still reached by a named maintainer adoption. Native-scale basis, effective-sample-size method and
noise-floor method must exactly match that adopted declaration. The emitted
`correspondence-record-profile/v3` binds both declaration and adoption by SHA-256; readiness
refuses envelopes without structurally valid review evidence. A changed declaration invalidates
its prior adoption instead of silently carrying approval onto different methods.

This mechanism does not review or choose a method, sign for a maintainer, approve a resulting
profile, establish pool admission or assess exchangeability. No method review or source record was
invented in the repository, so readiness remains `NO_INVENTORY`, 0 of 48, with exchangeability
`NOT_ASSESSED`.

### TG17.15 slices 10-17 - real TESS pool, corrected curation handoff and UI

**TG17.15 slices 10-17 (2026-09-12): implemented through human-review handoff; decision still
pending.** Source-bound batch import, bounded TOI-period discovery, exact SPOC acquisition,
adopted marginal methods and incremental expansion produced 112 acquired targets. A lineage audit
then found that acquisition had copied the final discovery row's period fields onto every target.
The emitter is fixed and regression-tested; reconstruction from frozen discoveries and
SHA-256-verified cached FITS bytes now records all target mappings exactly. Eight targets fail the
eight-native-cycle requirement under their correct periods, leaving 104 qualified profiles.

The corrected assessment has 64 records reaching 48 alternatives and pool sizes 0 / 58.5 / 70.
The deterministic packet has a 59-record 48-core and a 57-record all-pairs-admissible subset, with
56 alternatives per selected record. Contaminated receipts, profiles, readiness and packet remain
under named `g17_tess_lineage_bug` archives. Readiness is `READY_FOR_CURATION_REVIEW`, while
exchangeability remains `NOT_ASSESSED`.

Platform Status now displays readiness, all eight refusals, assessment, packet IDs, pending review
requirements, adoption state and archived evidence. A reviewer can write either permitted
decision and adopt it through separate controls with no supplied scientific defaults. External
TESS discovery/acquisition and immutable reconstruction/export remain explicit CLI jobs; their
results are visible in the UI, but those network and bulk-publication operations are not launched
there. **Still to do:** a named human must author and adopt the exchangeability decision. Only
after that may a separate successor/inventory promotion consume it; no code has made either act.

### T4E.35 - crop-edge support in the failed catalogue join

**T4E.35 (2026-09-12): measured as a post-hoc diagnostic, not an acceptance.** The immutable
T4E.28 per-storm rows were reused without rerunning extraction. Exact great-circle distance to
all four crop boundaries puts seven of eighteen observations within the existing 2-degree
support margin. Their raw-path median nearest-feature error is **1617.7 km**, against **35.9 km**
for eleven interior observations; the four largest finite errors and the sole no-feature row are
all nearest the east edge. The SWT split points the same way, less strongly. Rank correlation is
-0.736 on the raw path and -0.515 on SWT.

This cannot establish causation because the rows were inspected before the diagnostic existed,
and RUBY remains a counterexample to any claim that east-edge longitude is sufficient. It does
select the next experiment: freeze an eastward wrapped-acquisition declaration, acquire the
complementary dateline segment without replacing the current record, and rerun the same storms
and extraction parameters. Do not tune the extractor or the tolerance first.

### T4E.34 - composing a declaration, and committing it alone

**T4E.34 (2026-09-11): the other half of "do the experiments without editing a JSON file".**
`src/core/declaration_composer.py`, `DeclarationComposer.tsx`,
`/identity/declarations/compose`.

**Why this is not a form over a text editor.** T4E.30 surveyed every study here: ten can be
carried into an evidence bundle, six cannot, and NOT ONE of the six was refused for being declared
after the fact. In every case the declaration and the measurement entered git in the same commit.
Those declarations were almost certainly written first; git cannot separate them, so nothing
downstream can check it. That is what happens when writing a declaration means opening an editor
mid-session: it gets saved with everything else.

So the composer commits the declaration **by itself**, and that control is part of the form
rather than an afterthought. A test stages an unrelated file first and checks it is still staged
and uncommitted afterwards.

**Structure and refusals, never content.** No template text, no suggested prediction, no example
claim boundary. Every prediction must carry what would falsify it -- TG19.5 can prove a
prediction impossible when it has the bars and the values, and at declaration time those do not
exist, so what can be required is the question. A declaration with no gate, no prediction, a
placeholder declarer or any empty field is refused with the reason.

**Composing is not adopting.** The status written is always `DRAFTED_NOT_ADOPTED`; signing stays
the separate act T4E.32 built. And a declaration is written once: *"editing one after a run is how
a gate becomes whatever the result was."*

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_declaration_composer.py -q   23 passed
$ npx playwright test adoption.spec.ts                                            9 passed
```

**The chain is now complete on the surface**: compose a declaration and commit it alone, run the
measurement, sign the declaration, build the bundle, convene the panel, read the exchange.

### T4E.32 / T4E.33 - signing and convening, moved into the instrument

**A single-maintainer research instrument whose experiments can only be run by editing JSON and
typing shell commands is not finished.** `src/core/adoption.py`, `AdoptionView.tsx`,
`ConvenePanel.tsx`, and four routes.

**The rule did not change; where it is enforced did.** *Code does not sign a scientific
declaration for a person* stands. Until now it was enforced by the awkwardness of a text editor,
and awkwardness is a poor place to keep a principle. A maintainer who reads a declaration, types
their own name and types the affirmation has signed it -- that is what a signature is.

The substance is kept exactly: the adoption binds the declaration's sha256; it is written once
and never overwritten; it requires *"I have read this declaration and I adopt it"* typed in full,
because a signature producible by one click is producible by accident; a placeholder name is
refused because the artefact IS an attribution; and the surface supplies no default for the name,
the reason or the affirmation. Drift is reported as loudly as adoption -- a declaration amended
after signing reads *"what was signed is not what is on disk"*.

**One route now runs a model, and the module says so instead of claiming otherwise.**
`reviews.py` opened with *"It never runs a model..."*. That is now false in its first clause, so
it was replaced rather than quietly deleted. The authorisation moved into the request rather than
out of the system: `i_authorise_paid_calls` is required, the key is read from the server
environment and never from the wire, the call count is stated before anything is spent, and the
authorisation checkbox is a separate control from the run button.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_adoption.py -q     24 passed (19 functions)
$ npx playwright test adoption.spec.ts ui-qualification.spec.ts        11 passed (1.0m)
```

**Still to do, named rather than implied: declaring an experiment from the surface.** Signing a
declaration is now a button; writing one is still a hand-edited JSON file. That is the larger
half of "do the experiments without editing a JSON file" and it is not done here.

### T4E.31 - the round-robin, wired to a bundle; two real attempts remain partial

**T4E.31 (2026-09-11): the runner.** `tools/review_join_rerun.py`,
`src/tests/test_review_runner.py`.

**Almost none of this is new code.** The eight-seat protocol, the dissent register, the R22
independence check and a Gemini batch transport were built and tested at TG7.1 and TG7.2. What
was missing was a bundle to review -- T4E.30 supplied the first -- and a runner. The runner is a
short loop and a long list of refusals.

**Nothing reaches the network unless the maintainer says so, in the command.**
`--send-to-the-network` is required and a key must be present in the environment, never in an
argument where it would land in the shell history and the process table. `--dry-run` prints the
whole plan and sends nothing. There is deliberately no stub transport in the tool: a review
record asserts that a panel said something, and a fabricated one in `data/reviews/` beside real
ones would be the worst artefact this programme could produce.

**Driving the protocol over the real bundle corrected two assumptions.** An exchange with no
dissent takes SEVEN turns, not eight -- `response_and_revision` is skipped, because a response to
no dissent is a rebuttal of nothing. And a dissent must be answered by name, oldest first; a
response naming the wrong one is refused, and the refusal carries the paid call (R23).

**One asymmetry found and pinned rather than quietly fixed.** A protocol violation comes back
carrying its call; a response that does not match its declared schema is refused before the call
is recorded, so that one is lost. Both were paid for. A test pins the behaviour so changing it is
deliberate.

**Two paid attempts ran on 2026-09-12; no valid panel completed and no outcome exists.** The first
was refused at independent reassessment after six calls for originating dissents no challenger
raised. The second reached 11 calls, but its reassessment reopened four dissents the candidate
had conceded and its final synthesis retained them. Both calls chains are preserved unchanged as
partial records. The mismatch is repaired by allowing reassessment to reopen only rebutted
dissents; concessions remain final. A third paid attempt requires fresh explicit authorisation.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_round_robin.py src/tests/test_review_runner.py -q
64 passed
```

### T4E.30 - the measurement store and the evidence store, joined where git can prove the ordering

**T4E.30 (2026-09-11): a measurement carried into an evidence bundle, or refused by name.**
`src/core/measurement_evidence.py`, `tools/bundle_join_rerun.py`,
`data/studies/t4e28-join-rerun.r7.json`.

**The gap, counted.** The round-robin, the recorded discussion and the claim ladder all read an
`EvidenceBundle`, and `summarise_evidence` is *"a pure function of the bundle: no clock, no I/O,
no other input"* -- so a panel sees only what was appended as evidence entries. On 2026-09-11
this repository held **28 measurement records, 43 declarations and zero bundles**. Every finding
the programme had produced was invisible to the stage built to argue about it.

**Registration is proved from git, not asserted.** A bundle refuses a hypothesis registered after
its evidence, and it is right to: back-dating one manufactures a preregistration, and the panel
would be arguing about a risk nobody took. The commit that first added the declaration must
precede the commit that first added the measurement, both files must be tracked, and both working
copies must match what was committed. T4E.28 qualifies because its gate landed in `70640b4`,
carrying no measurement, seventeen minutes before the run.

```
entries: 7
  replication_results      PASS   both declared gates reproduced
  provenance               PASS   the recovered extraction parameters are the ones that produced the record
  provenance               PASS   the catalogue is the signed one, resolved by digest
  contradictory_evidence   PASS   T4E.18's published non-dateline range is false
  null_results             FAIL   condition 2 on the raw path, measured for the first time
  failure_states           PASS   the acceptance this reproduces still fails
  uncertainty              PASS   what the catalogue's own positional uncertainty is

claim ladder: observation   (standing contradiction, and a FAIL -- both cap it, correctly)
```

**What the module will not do.** It does not decide what a measurement means. Which entries a
record yields, in which category and at which status, is supplied by the caller as
`EvidenceClaim`s, written down in the tool where they can be disagreed with. A module that
inferred them would be authoring the evidence it claims to transport.

**Surveying every study found something worth naming.** Of sixteen declaration-and-measurement
pairs, ten can be bundled and six are refused -- **not one for being declared afterwards**. In
every refused case the declaration and the measurement entered the repository in the *same
commit*: T4E.12, T4E.14, T4E.19, T4E.20, T4E.21 and T4E.24. Those declarations were almost
certainly written first, and git cannot separate them. The refusal says so and names the practice
that lifts it for future work: commit the declaration on its own, before the run, as T4E.28 did.

**What this does not establish.** No claim rung or adoption. A bundle existing made discussion
possible; the two later partial attempts establish that calls happened, not that a valid panel
completed or that an outcome exists.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_measurement_evidence.py -q
21 passed
```

### T4E.29 - every distance, and what an exclusion does to the answer

**T4E.29 (2026-09-11): the distribution, rather than its extremes.** An engineering slice -- it
makes no claim about any world, so it carries no declaration, no adoption and no prediction.
`src/api/identity.py` route `/join-distribution`,
`frontend/src/components/JoinDistributionView.tsx`, `frontend/e2e/join-distribution.spec.ts`, a
population map for the re-run in `src/data_layer/declared_population.py`.

**Why it exists, in one measured failure.** T4E.18's correction stated that away from the
dateline the nearest extracted feature is *"16.6 to 99.3 km"*, *"a factor of two to three, not an
order of magnitude"*. The median was right; the range was not. SETH sits 315.1 km out and HOLA
247.7, neither near a boundary, and GRETEL yields no feature at all. That claim was written,
reviewed, committed and read back for a day. Two things let it stand: the record held **one
number per storm**, so nothing in the repository could contradict it, and the aggregate was taken
over a **subset nobody named**, so nothing could check the subset either.

Neither is a lapse of care. Both are missing surfaces.

```
$ curl '/api/v1/identity/join-distribution?population=raw_field'
everything   storms 18   nearest 16.6 - 52.1 - 3685.3   >=1 inside radius 3 of 18

$ curl '/api/v1/identity/join-distribution?population=raw_field&exclude_longitude_at_or_above=178'
kept         storms 12   nearest 16.6 - 35.9 - 315.1
excluded     storms  6   nearest 67.6 - 2028.0 - 3685.3  (1 with no feature at all)
```

**The kept maximum is 315.1 km.** The claim that survived review is refuted by running the
exclusion it implied and reading the answer.

**Four rules, each from something that went wrong.** No exclusion hides a row -- excluded storms
stay in the table with their reason, and their aggregate is computed at equal weight beside the
kept one rather than beneath it. A storm with no feature is drawn, not skipped: it has no
distance so it cannot appear on a distance axis, which is exactly why it would vanish, and
dropping it would improve every aggregate by removing the worst case. Every distance is plotted
rather than the nearest. And the population must be named, so an unnamed read is refused with
both names instead of served whichever is stored first -- the error T4E.27 made.

**Measured in a real browser**, because "a scientist can see it" is a claim about a screen:

```
$ npx playwright test join-distribution.spec.ts
6 passed (48.6s)
$ npx playwright test ui-qualification.spec.ts
4 passed (41.0s)
```

**What it does not do.** It computes no verdict, adopts nothing, stores nothing, and no route
behind it writes. It does not decide which extraction pass is right. And a chart is not a
finding: what a reader sees is the distribution a measurement recorded, under that measurement's
own claim boundary, which renders beside it.

### T4E.28 - the join re-run: both gates reproduced, and a claim in the record falsified

**T4E.28 (2026-09-11): the join re-run, with its gate committed before the run.** The gate landed
in `70640b4`, which carries no measurement; the numbers came afterwards.
`data/identity_calibration/t4e28-join-rerun-declaration.json`,
`src/benchmarks/catalogue_join.py`, `tools/rerun_t4e18_join.py`,
`measurements/t4e28_join_rerun.json`.

**Two gaps in an adopted record.** T4E.18's acceptance carried per-storm rows for the SWT path
only; the raw-field path its own correction calls *markedly better* existed as four rounded
aggregates. And neither path's extraction parameters were written down anywhere in this
repository -- both runs came from scripts in a session scratchpad under `%TEMP%`, so figures in an
adopted record were reproducible only from a directory nobody would think to preserve. The
parameters were recovered while those scripts still existed and now live in the tool.

```
raw_field    nearest min 16.6  median 52.1  max 3685.3 | condition 1: 3 of 18   GATE REPRODUCED
swt_planes   nearest min 29.9  median 127.1 max 2055.9 | condition 1: 2 of 18   GATE REPRODUCED
recovered SWT parameters against the 18 recorded rows: CONFIRMED
```

**Every aggregate reproduces to 0.1 km and all 18 SWT rows regenerate exactly**, which confirms
the recovered parameters are the ones that produced the record rather than a plausible guess.

**T4E.27's condition 2 is LIFTED**: 1 of 17 judged on the SWT path, 0 of 17 on the raw path,
against a bar of 9. The verdict does not move -- FAILED, now on both conditions rather than one
with the other refused. `tools/restate_position_acceptance.py` computes it from the T4E.28 record
and still refuses by name when that record is absent.

**Condition 2 on the raw path, measured for the first time: 0 of 18**, against the SWT path's 1.
No prediction was offered for it, because there was no prior figure to reproduce. It qualifies the
previous reading: the raw path is better on nearest distance and worse on the count inside the
radius, which with 0-13 features per frame against 73-153 is what should be expected.

**A claim in T4E.18's own correction is falsified.** It stated that away from the dateline the
nearest feature is *"16.6 to 99.3 km"*, a *"factor of two to three, not an order of magnitude"*.
The median is right; the range is not. SETH at longitude 155.9 sits 315.1 km out and HOLA at
175.8 sits 247.7 km out, both far from any boundary and an order of magnitude above the 22.1 km
radius median; GRETEL yields no feature at all. RUBY, at longitude 179.0 inside the band named as
off-frame, is fine at 67.6 km -- so dateline longitude is not sufficient for the failure. The
upper bound had been quoted from the storms that fit the story, and the record held only a
minimum per storm, so nothing in the repository could contradict it. *Two real failure modes* was
too few.

**What it does not do.** It adopts nothing, approves no radius, and does not change the T4E.18
verdict, which remains FAILED. A reproduction of a failing measurement is still a failing
measurement. It says nothing about which extraction path is right, whether the bar of 9 was well
chosen, or what the atmosphere does.

### TG19.5 - can the change you are about to make move the answer at all?

**TG19.5 (2026-09-11): can the change you are about to make move the answer at all?**

An engineering slice. It makes no claim about any world, so it carries no declaration, no adoption
and no prediction. `src/analysis_engine/prediction_sensitivity.py`, wired into
`tools/restate_position_acceptance.py`.

**The failure it prevents, computed on the case that produced it.** T4E.27 declared before
measuring that restating a bar would leave the acceptance failing, *"improving on 2 of 18 but not
reaching 9"*. The failure half held. The improvement half was not a risky prediction that came out
wrong -- it was **impossible**, and the impossibility is three columns of arithmetic:

```
storm      old bar   new bar   separation   could flip
FEHI          8.90     12.10        69.74   no
GITA         89.38     89.76       113.72   no
HOLA         21.99     23.46       252.29   no
LINDA         0.00   refused        74.91   not judged
JOSIE       145.23    145.46       189.65   no
...
swing set: 0 of 17 judged      admitted before 2 -> after 2      INERT
```

No observation has its separation between the old bar and the new one. The bar moves 11.12 to
13.81 at its most generous and 89.38 to 89.76 at its least, against separations of 30 to 2056 km.
**Nothing could change, however the bar was justified.**

**What the module reports.** `verdict_travel` classifies every observation as gained, lost,
admitted either way, rejected either way, or not judged, and exposes the swing set.
`check_prediction` then adjudicates a declared direction against what the arithmetic permits:
`POSSIBLE` when the measurement decides it, `IMPOSSIBLE` when it is settled in advance, and
`TRIVIALLY_TRUE` for "unchanged" on an inert change -- because predicting no change where nothing
can change is true before the run and carries no evidential weight.

Applied to T4E.27's own numbers, `improve` returns **IMPOSSIBLE** and `unchanged` returns
**TRIVIALLY_TRUE**.

**Why this belongs in the instrument rather than in a habit.** A prediction declared before a
measurement is this programme's main guard against reading a result into the answer already
believed. Seven criteria have been adjudicated that way. The guard is worth nothing where the
arithmetic settles the prediction in advance, and **a declaration that reads as though a risk was
taken is worse than one that predicts nothing** -- it buys credibility it has not earned. The
check makes that auditable instead of assumed.

**A refused bar is not movement and not immovability.** An observation whose tolerance was refused
leaves the judged population entirely rather than counting as a verdict that could not change.
LINDA's `0.00` radius is `not_judged`, 17 are judged, and the swing set is computed over those --
the same discipline `PositionTolerance` applies, carried through so the two agree. The inclusive
boundary matches `admits` for the same reason: a different convention here would disagree with the
thing it checks.

**What it does not do, stated in the module itself.** It sees one kind of error -- a threshold test
whose threshold moves. It says nothing about whether the right quantity is being thresholded,
whether the population is the right one, or whether the bar is defensible. Those three are what
actually decided T4E.27, and none is visible here. **A clean report is not a sound design**, and a
module implying otherwise would sell the same false comfort it was written to remove. A test pins
that the report says so.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_prediction_sensitivity.py -q
15 passed
```

`restate_position_acceptance.py` now carries `was_the_prediction_ever_falsifiable` beside its
verdict, reporting `IMPOSSIBLE` for its own improvement half. Every measured figure is unchanged
-- 2 admitted of 17 judged -- because auditing a prediction moves no measurement.

**Where this leaves the sequence.** Three of the six underived facts are now structurally harder to
repeat: the wrong unit (TG19.1's `survival_by_cardinality` reads `ALLOWED_CARDINALITIES` rather
than relying on memory), the unlabelled population (TG19.4), and the unfalsifiable prediction
(this). The remaining ones -- choosing the wrong unit to think in, and reading a correction block
without carrying its distinction forward -- are attention, and no checker catches them.

### TG19.4 - a record holding two answers is read by name, or not at all

**TG19.4 (2026-09-11): a record holding two answers is read by name, or not at all.**

An engineering slice. It makes no claim about any world, so it carries no declaration, no adoption
and no prediction. `src/data_layer/declared_population.py`, wired into
`tools/restate_position_acceptance.py`.

**The failure it prevents had already happened, twice over.**
`measurements/t4e18_acceptance.json` holds two extraction passes with different distances,
different feature counts and different verdicts. Its `per_storm` table is the SWT one, and nothing
in that table says so -- the fields are `storm, lat, nature, time, radius_km, nearest_km,
inside_radius, features`, and the only way to tell is to match feature counts against a prose
block elsewhere in the same file. T4E.27 read that table, restated a bar against it, named no
path, and had its sharpest conclusion withdrawn the next day.

**This repository already had the fix, pointed the other way.** `declare_identity_target` refuses
to run an audit without a declared target and evidence class, because labels drawn from the
pipeline under test validate a definition against itself. That discipline governs the *inputs* of
a measurement and nothing that *reads one back*. `/api/v1/identity/measurements` will serve an
unlabelled table to anyone. This is the same rule applied to consumption.

**Four refusals, and the third is the one that matters.**

```
read_population(record)                 -> REFUSED: holds more than one population and was read
                                           without naming which. It holds: swt_planes, raw_field
read_population(record, "nonsense")     -> REFUSED: holds no population named 'nonsense'
read_population(record, "raw_field")    -> REFUSED: recorded as a summary only, not as rows
read_population(record, "swt_planes")   -> 18 rows, with how they were identified
```

The third is the dangerous one. The raw-field pass is *better* for this purpose by T4E.18's own
correction -- 52.1 km median against 127.1 -- and it exists in that record as **four aggregate
numbers and no rows at all**. A reader asking for it must not receive the SWT rows under its name,
which is exactly the substitution that went wrong. The refusal says what is present, and says that
producing the rest needs the join re-run.

**The map is declared beside the record and never edited into it**, for the same reason T4E.17's
signed design was left alone and its local path recorded elsewhere: committed evidence is not
rewritten to suit a later reader. A test asserts the record still carries no `populations` key and
no per-row `extraction_pass`.

**The identification is checkable, not assertable.** Each population states how it is
distinguished -- the SWT rows carry 73 to 153 features where the raw pass yields 0 to 13 -- and a
test verifies that claim against the rows it describes. The two ranges do not overlap, which is
what makes the mapping decidable rather than a label somebody chose.

**The tool that made the mistake now cannot repeat it.**
`restate_position_acceptance.py` declares `SOURCE_POPULATION = "swt_planes"` because
`read_population` refuses without one, and its receipt carries `extraction_pass`, the evidence for
that identification, and a note naming the other pass and why it cannot be substituted. Every
figure is unchanged -- 2 admitted of 17 judged, 1 refused -- which is the point: naming what was
already being read moved no measurement.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_declared_population.py -q
17 passed
```

**What this does not do.** It maps one record. Every other measurement in the store is still read
whole and at the reader's risk, and a record absent from the registry is refused for a *named*
read rather than being validated -- the module says what it does not know. It does not re-run any
join, does not lift T4E.27's condition 2, and changes no verdict anywhere. And it catches only the
class of error where a record holds two answers: the framing error in T4E.24's first addendum, and
the arithmetic one in T4E.27's prediction, are different failures and are untouched by this.

### TG19.3 - a signed reference resolves by digest, or refuses by name

**TG19.3 (2026-09-11): a signed reference resolves by digest, or refuses by name.**

An engineering slice. It makes no claim about any world, so it carries no declaration, no
adoption and no prediction. `src/data_layer/signed_reference.py`, wired into
`tools/restate_position_acceptance.py`.

**The gap, and it was not hypothetical.** T4E.17 signed the IBTrACS catalogue on terms naming a
specific population, bound it by sha256, and deliberately did not commit 35.5 MB of third-party
data. What the design records is the URL, the digest and the byte count. What it records **no**
path. So nothing in this repository knew where to look, nothing failed loudly when the file was
missing, and the file spent a day in a session scratchpad under `%TEMP%` where it survived by
luck. Losing it would have **voided the signature**, not merely cost a download: IBTrACS v04r01 is
a living archive, so a fresh copy is a different catalogue on which T4E.17's terms do not hold.

That is the lesson T4E.27 drew about a catalogue radius of `0.00`, applied to a file instead of a
field: **a missing input should be refused by name, not discovered later.**

**Three bindings, checked in order, because a chain is only as strong as the link nobody checks.**

1. **Signature against design.** `signs_sha256` in the signature file, against the design's own
   digest. If the design was edited after it was signed, the terms recorded are not the terms
   signed and everything downstream inherits the drift. Nothing had ever checked this.
2. **Design against data.** The digest the file must reproduce.
3. **Byte count first**, as a cheap pre-check, so a truncated download is named before 35 MB are
   hashed to reach the same conclusion.

```
$ python -c "from src.data_layer.signed_reference import resolve; ..."
name                 ibtracs_sp_v04r01
path                 data/catalogues/ibtracs.SP.list.v04r01.csv
available            True
signature_verified   True
bytes                35,482,417 expected, 35,482,417 observed
digest               matches the signed 631f76b9...
citations            2, carried with the resolution
```

**A refusal is a result, not an exception by default.** `resolve()` returns a record saying what
failed and what would lift it; `require()` raises for a caller that cannot proceed. Both carry the
source URL and the expected digest, so a reader who has lost the file learns where to get it and
what it must hash to in the same breath as learning it is gone.

**A digest mismatch is reported as a *different* reference, never a damaged one.** The refusal
says so in terms: the signed population, base rates and claim boundary do not extend to it, and
using it would evaluate against an unsigned catalogue. There is no fallback path to an unverified
copy, and nothing here reaches a network -- acquiring the file is the maintainer's act.

**A refusal that could not tell "missing" from "present" was itself the defect.** The first
version of `restate_position_acceptance.py` stated in prose that the catalogue was absent. It had
no way to check, and it was wrong: the file existed. That clause is now resolved rather than
asserted, and the tool's receipt carries the resolution beside the refusal. **The committed
T4E.27 receipt still says the file was not present on this machine. It was accurate when it was
written and is superseded rather than edited**, which is how this programme records corrections.

What the refusal now says is the part that was always true and is the only part that still is:
condition 2 needs the distance to the third-nearest feature, the committed receipt carries only
the nearest, and recovering it takes a re-run of the join. That re-run is its own work and is not
done here -- but it is no longer blocked on a missing file.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_signed_reference.py -q
16 passed
```

The tests pin each way silence could return: an absent file naming path, URL and digest; a
truncated one named by size without hashing; a different edition refused as different; a design
edited after signature; a signature carrying no digest; a design carrying no digest; an unsigned
design allowed and reported as unverified rather than failed; and `require()` raising by name
instead of returning an unverified path.

**What this does not do.** It downloads nothing, adopts nothing, and changes no default. It does
not lift T4E.27's condition 2, which needs the join re-run. It does not edit the signed design to
record the path -- that would break the sha256 the signature rests on, so the path lives in the
registry beside it. And it verifies bindings, not contents: that the file is the one that was
signed says nothing about whether the terms signed were the right ones.

### TG19.2 - the join's bar, on the wire and on screen

**TG19.2 (2026-09-11): the join's bar, on the wire and on screen, in parts.**

An engineering slice. It makes no claim about any world, so it carries no declaration, no
adoption and no prediction. `src/api/identity.py` (two routes),
`frontend/src/components/PositionToleranceView.tsx`, `frontend/e2e/position-tolerance.spec.ts`.

**Why it exists.** T4E.27 built a tolerance that publishes its parts, and it was reachable only by
importing a Python module. A researcher could read what bar this programme used and could not see
what theirs would be. T4E.27 also found two things a researcher needs and neither was visible:
the bar was wrong for three reasons, *and* replacing it changed nothing. A bar that arrives as a
single number can only be accepted or rejected; one whose components, provenance, exclusions and
refusals are on screen can be disagreed with specifically.

**Two routes, both GET, both computing rather than deciding.** `/identity/tolerance/components`
serves the contract before any observation is supplied -- what the parts are, who supplies each,
how they combine, what is excluded, and why a missing uncertainty is refused.
`/identity/tolerance` computes one bar, and with an optional separation also reports whether it
is admitted and what the justified components fail to explain. No route stores a tolerance,
approves a join, or records an acceptance, and none was added.

**`null` is a third answer and the wire carries it as one.** A refused tolerance returns
`admitted: null`, never `false`, because *we could not say* and *no* are different answers and a
client that conflated them would count a missing catalogue uncertainty as a failed detection --
the exact error T4E.27 exists to correct.

```
$ curl '/api/v1/identity/tolerance?catalogue_radius_km=11.12&separation_km=99.98&observation=OWEN'
total_km 13.81   admitted false   unexplained_residual_km 86.17

$ curl '/api/v1/identity/tolerance?catalogue_radius_km=0.0&separation_km=74.91&observation=LINDA'
total_km null    admitted null    unexplained_residual_km null
```

**Rendered evidence, because PLAN section 5's acceptance is a claim about what a researcher can
see.** Six Chromium tests against the real API and the real frontend:

```
$ cd frontend && ./node_modules/.bin/playwright test e2e/position-tolerance.spec.ts
  ok 1 the bar arrives in parts, each with where it came from (17.2s)
  ok 2 what the bar leaves out is on screen at the weight of what it includes (2.7s)
  ok 3 a missing catalogue uncertainty is refused by name, not scored as a miss (3.0s)
  ok 4 a refusal renders as a result, not as an error state (2.8s)
  ok 5 a computed bar shows its total and the residual it does not explain (2.9s)
  ok 6 the panel states what it will not do, from the server own list (2.7s)
  6 passed (57.3s)
```

Test 3 is the one the panel was written for: LINDA's reported radius in the acquired record is
exactly 0.00, the case that made one storm of eighteen unpassable however good the extraction
was. On screen it renders as a refusal naming its reason, the verdict element is absent
entirely rather than showing a miss, and test 4 requires the surrounding result section to still
render with no error banner -- a bar that could not be built is an answer the instrument is
entitled to give. Test 6 asserts no control matches `accept|approve|save|record|apply`, because
the panel's only control is arithmetic.

**A pre-existing defect this slice found, and it is not this slice's.** Adding a workspace made
`test_the_qualification_gate_inventories_every_served_workspace` fail -- and the drift it
reported was **`Study trail`**, not the new panel. T4E.23 added that workspace on 2026-09-10 and
never added it to `e2e/ui-qualification.spec.ts`; the spec's last commit is T4E.8 slice 4's. So a
served workspace has been outside the qualification gate since then, and the guard that catches
exactly this was not run when it shipped. Both entries are now listed, in shell order. **A
workspace outside the qualification inventory is an unqualified surface that reads as a qualified
one**, which is the failure mode the guard exists for.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_frontend_contract.py src/tests/test_identity_api.py -q
214 passed

$ cd frontend && ./node_modules/.bin/tsc --noEmit -p tsconfig.json
(clean)
```

**What this costs, recorded rather than absorbed.** This slice edits `frontend/src/App.tsx` and
`e2e/ui-qualification.spec.ts`, and recorded evidence in this programme is bound to the source it
was measured against. The G17 release plan will therefore return `browser_no_glue` and
`synthetic_fifth_adapter` to `NOT_RUN`, as it did for T4E.8 slice 4 and as TG17.14 demonstrated
when editing an acquisition module invalidated a passing record. The G17 verdict was already
`NOT_RELEASEABLE` on `scale_shape_calibration`, so no release decision changes. Restoring those
two gates needs a fresh full browser run and its recording, and that has not been done.

**What is not claimed.** No WCAG level: this is a rendered functional inspection, not an
assistive-technology audit. Nothing is adopted, no default changed, and no tolerance approved for
any pipeline. The panel computes a bar for whatever numbers a reader types; it says nothing about
whether that bar is right for their catalogue, and the excluded component is on screen precisely
so they can see what it does not cover.

### TG19.1 - the coverage question, made runnable by someone else

**TG19.1 (2026-09-11): the coverage question, made runnable by someone else.**

An engineering slice, not a scientific one: it makes no claim about any world, so it carries no
declaration, no adoption and no prediction. `src/benchmarks/coverage_report.py`,
`tools/coverage_report.py`, `survival_by_cardinality` in `src/benchmarks/false_absence.py`.

**Why it exists.** T4E.24 through T4E.27 measured what this extractor loses -- 37% of features
present never extracted, 85% of triples never surviving six scenes, a cut that cannot be relaxed
to fix it. Those numbers became *readable* through T4E.22's measurement API and T4E.23's study
trail. They were not *runnable*. A researcher arriving with their own field and their own
extractor could read what this programme measured about its own instrument and could not ask the
same question of theirs. That is the difference between a lab notebook and an instrument.

**What was generalised, and what stayed put.** The T4E.24 measurement was bound to ERA5 shards,
a vorticity sign convention and one extractor. The mechanism was never atmospheric: plant known
structure, extract, count what came back. `coverage_report` now takes a `background_for(i)`
callable, any **registered** extractor by name, and the caller's own declaration of what the
field is -- domain, dataset, variable, units, which R19 refuses a comparison without. The
atmospheric run is unchanged and its receipts are untouched.

**`survival_by_cardinality` turns a derivation into a capability.** T4E.24's second addendum
computed pair and triple survival by hand in a one-off script, because
`ALLOWED_CARDINALITIES` is `(2, 3)` and whole-configuration survival measures something nothing
downstream consumes. That arithmetic is now a function, tested, and reported by every coverage
run. Checked against the committed receipt, it reproduces the addendum exactly: pairs 394/1362
intact and 649/1362 assemblable, triples 418/2805 and 856/2805.

```
$ python -m tools.coverage_report --list-extractors
local_maximum    Local maxima above a surrogate-calibrated frame-maximum threshold, ...

$ python -m tools.coverage_report --output r.json --source fbm --configurations 3 --surrogates 199
VERDICT: EVIDENCE_NOT_REPRESENTATIVE
  median 3 features per frame is outside the declared band [4, 10] taken from the record
```

**The first run refused, and that is the instrument working.** The default band is the one the
atmospheric work declared, and it is wrong for a fractional-Brownian field -- which the tool's own
help says before it is run: *"the default is the one the atmospheric work declared and is
probably wrong for you"*. A caller who states a band their field warrants gets a measurement; a
caller who does not gets a refusal naming the reason. What is deliberately absent is any path
where the background is adjusted until the number improves.

```
$ python -m tools.coverage_report --output r.json --source fbm --configurations 3 \
      --surrogates 199 --median-band 2 10
VERDICT MEASURED   gate median 3   pairs intact 0.181   triples intact 0.079

$ python -m tools.coverage_report --output r.json --source netcdf \
      --path data/cds_downloads/t4e18_vorticity --variable vo --negate \
      --domain atmosphere --configurations 10 --surrogates 199
VERDICT MEASURED   gate median 4   pairs intact 0.257   triples intact 0.107   52.6 s
```

The second runs with **no atmospheric data at all** -- a synthetic field with a declared Hurst
exponent, for a caller who has no record of their own. The third runs the same question over the
acquired record through the generic path and lands near T4E.24's triple figure of 0.149, at a
tenth the configurations and a fifth the surrogates, which is the agreement a smaller sample
should give and not an independent confirmation of it.

**What every report carries.** The extractor's registered capabilities, including its declared
`shape_model`, because a coverage number means something different for an instrument that assumes
isotropy. Which field the cut was taken from, with T4E.25's measured 3.53x attached, so a caller
choosing `background` sees what the choice is worth. That identical geometry across scenes makes
recall an **upper** bound. And a claim boundary saying the report is a property of an instrument
and never of a world -- it plants what it then looks for, so it can say nothing about whether
such structure exists in any real field.

**What this does not do.** It adopts nothing, changes no default, and alters no extractor. It
does not make the atmospheric numbers transferable: a coverage figure belongs to the field and
the planting it was measured on, and the module refuses to imply otherwise. And it is not yet on
the wire or on screen -- reports land in the measurement store and are served by T4E.22's
existing routes, which is reach, not a user interface.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_coverage_report.py -q
17 passed
```

### T4E.27 - a tolerance built from its components, and the T4E.18 acceptance restated

**T4E.27 (2026-09-11): the bar was wrong, and the bar was not what was carrying the failure.**

Adopted 2026-09-11 (`t4e27-position-tolerance-adoption.json`, binding the declaration by sha256
`f4ef59f6...`) and measured the same day. `measurements/t4e27_restated_acceptance.json`,
`src/analysis_engine/position_tolerance.py`, `tools/restate_position_acceptance.py`.

T4E.21 named this work and deferred it in writing. T4E.18's acceptance had failed at 2 of 18 and
1 of 18 against a bar of 9, and had stood as a failure ever since against a tolerance nobody had
argued for.

**Three reasons the original bar was wrong, none of which is that it failed.** A **category
error**: the bar was the catalogue's per-observation radius, an agency's bound on its own
uncertainty about its own quantity, never a bound on the separation between two different
quantities. An **unsatisfiable region**: the radii run 0.00 to 145.23 km and LINDA's is exactly
0.00, so one of eighteen storms could not have passed however good the extraction was -- a
missing agency report written down as a number, in a programme whose named-refusal rule had been
applied to its outputs and not to this input. And an **empirical miss**: T4E.19 measured the
correlation between offset and radius at +0.020, range straddling zero.

**The restated bar is computed, not chosen** -- the quadrature sum of the agency's reported
radius and the extractor's own localisation error, 0.295 cells or 8.19 km, as T4E.21 measured it
on known ground truth for a different purpose. The vorticity-versus-surface-centre separation is
deliberately **not** a component, so the residual measures it.

```
storm        lat   nearest   radius  tolerance  residual  admitted
FEHI       -36.8     69.74     8.90      12.10      57.6   no
GITA       -38.6    113.72    89.38      89.76      24.0   no
HOLA       -31.5    252.29    21.99      23.46     228.8   no
LINDA      -23.8     74.91     0.00    REFUSED         -   could not say
IRIS       -22.7    152.69    15.13      17.20     135.5   no
JOSIE      -21.5    189.65   145.23     145.46      44.2   no
OWEN       -20.8     99.98    11.12      13.81      86.2   no
PENNY      -20.0     51.20    11.12      13.81      37.4   no
OMA        -28.2    162.24    14.82      16.93     145.3   no
SARAI      -20.3   2055.91    24.57      25.89    2030.0   no
UESI       -37.5     60.81   103.47     103.80     -43.0   YES
GRETEL     -31.2    127.10   106.85     107.17      19.9   no
ANA        -22.3   1581.20    34.91      35.86    1545.3   no
LUCAS      -22.7     29.93    61.55      62.09     -32.2   YES
NIRAN      -28.4   1241.04   112.31     112.61    1128.4   no
UNNAMED    -30.1    159.45    22.24      23.70     135.7   no
RUBY       -30.7    107.00    11.12      13.81      93.2   no
SETH       -21.0    111.72    10.38      13.22      98.5   no

condition 1 restated: 2 admitted of 17 judged, 1 refused, needed 9 -- NOT MET
condition 2 restated: REFUSED BY NAME, not evaluable on the available evidence
median unexplained residual: 93.19 km
```

**The prediction was half right, and the half that was wrong was derivable.** The declaration
predicted the acceptance would still fail while *improving* on 2 of 18. It still fails. It did
not improve at all -- the same two storms pass, and no others. The reason is arithmetic that
should have been done in advance: adding 8.19 km in quadrature to radii of 11 to 145 km is very
nearly inert. It moves a bar of 11.12 to 13.81 and a bar of 89.38 to 89.76, against separations
of 50 to 250 km. **The estimator component could never have changed a verdict here, and saying so
required no measurement.** That is the fifth time in this programme a derivable fact has been
left underived, and it is recorded rather than smoothed over.

**So the finding is sharper than either outcome the declaration anticipated.** The bar was wrong
for three good reasons *and* replacing it with a defensible one changes nothing. What was
carrying the failure is not the tolerance. It is the separations themselves: a median nearest
feature of 127 km, with three storms -- SARAI at 2056 km, ANA at 1581, NIRAN at 1241 -- that no
tolerance worth the name will ever admit.

**The two storms that do pass, pass for the wrong reason.** UESI and LUCAS are admitted because
their *catalogue radii* are 103 and 62 km -- the agency was highly uncertain about where those
centres were. A join that closes because the reference is vague is not evidence that the
instrument found the cyclone, and counting it as one would be the same category error in the
opposite direction.

**The residual localises what remains, and it does not fit T4E.21's hypothesis.** The median
unexplained separation is **93.2 km**. T4E.21's leading candidate -- that an 850 hPa vorticity
maximum and a surface centre are different quantities, worth about 34 km -- cannot account for
that. So on this population the quantity difference is **not** the dominant term, and the honest
conclusion is that T4E.18's acceptance population and T4E.19's diagnostic population are not the
same problem. T4E.19 worked on 154 interior observations paired within 200 km with a dateline
group separated out, and got a median of 33.8 km. T4E.18's acceptance takes the deepest
observation per storm with no such filtering, and gets 127 km. Restating the bar exposed that
rather than repairing it.

**The acceptance curve, reported for inspection and not for selection.**

```
tolerance km     5    10    15    20    25    30    35    40    50    65
admitted /17     0     0     0     0     0     1     1     1     1     3

tolerance km    80   100   125   150   200   300   500
admitted /17     4     5     8     9    13    14    14
```

Nine of seventeen -- the declared majority -- first arrives at **150 km**, roughly an order of
magnitude beyond anything the two justified components support, and the curve then saturates at
14 of 17 no matter how far it is pushed. The declared bar decides the verdict; this is here so
the bar can be argued with specifically rather than merely accepted, and choosing a point from it
now would be the horse race R20 forbids.

**Condition 2 is refused by name, and this slice therefore meets its own acceptance only in
part.** Condition 2 asks for three or more features inside tolerance. The committed receipt
carries each storm's nearest distance and a count of features inside the *original* radius, but
not the distance to the third-nearest, so the count at any other bar is not recoverable from it.
Recomputing it needs the IBTrACS CSV, which T4E.17 bound by sha256, did not commit, and which is
not present on this machine. Approximating it was available and was refused. **The shortfall is
reported rather than absorbed:** this task's own acceptance condition 3 asked for both
conditions, and one of them was not delivered.

**The instrument, which was the point of the task.** `PositionTolerance` publishes each
component with its provenance, names what it deliberately excludes, and refuses an unusable input
by name -- returning `None` from `admits`, never `False`, because *we could not say* and *no* are
different answers and conflating them counts a missing agency report as a failed detection. A
refused observation leaves the denominator rather than scoring against the instrument. A bar
built this way can be disagreed with in parts, which is what lets a question this programme did
not think of be asked with the same instrument.

**What is not licensed.** No mining radius, no tolerance adopted into any pipeline, no part of
T4E.8's acquired-record acceptance discharged, and nothing settled about whether the record
supports `kind_recurrence`. `measurements/t4e18_acceptance.json` stands as measured with its own
correction beside it; this restatement is recorded alongside it, never in place of it. The
population, conditions, extractor, null and alpha are all unchanged, and nothing was adopted from
T4E.26. D96 through D100 remain open. No reserved seed block and no frame of the 2022-2023
forecast-test period was read.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_position_tolerance.py -q
17 passed
```

One test failed while writing this and was itself the error: it asserted that an infinite reported
radius should be accepted as a value rather than refused. An infinite bar admits every
separation, which is worse than having no bar at all, so the module was right and the test was
wrong. The test was corrected and the module's refusal wording was tightened to say why a
non-finite radius is refused as well as a zero one.

### T4E.26 - whether a null estimable from real data closes the self-inflation

**T4E.26 (2026-09-11): peeling the null works, manufactures most of its own contamination, and
changes what a detection claims.**

Adopted 2026-09-11 (`t4e26-peeled-null-adoption.json`, binding the declaration by sha256
`3e374738...`) and measured the same day. One round, `alpha = 0.05`, T4E.24's 360 scenes under the
same root seed. `measurements/t4e26_peeled_null.json`. Three ensembles per scene -- the scene's
own, the residual's, and the bare background's -- 71 minutes.

**Acceptance condition 2 first.** Round zero reproduces T4E.24 and T4E.25 exactly: feature
recovery 0.6276, the presence distribution identical key by key (121/12/20/8/10/15/228),
configuration coverage 0.1667 and 0.0833, spurious 0.0417 per scene. Every feature had a usable
width, so nothing was left unsubtracted for want of one.

```
thresholds, in units of the background's own robust width (median over 360 scenes)
   round zero 15.17      peeled 8.64      oracle 4.37

fraction of the oracle gap closed, by decile over 360 scenes
   0.000  0.396  0.488  0.532  0.587  0.637  0.701  0.736  0.798  0.896  1.041

stage        gate med   feature recovery   recoverable    intact      spurious/scene
round zero       4          0.6276         0.1667(10/60)  0.0833(5/60)    0.042
peeled           6          0.8237         0.6000(36/60)  0.3000(18/60)   0.275

features never recovered in any scene:  121 -> 39
newly recovered trials 487, trials lost 0
median peak-to-background ratio: newly recovered 12.51, already found at round zero 24.02
```

**The prediction held on all three limbs, against bars fixed by T4E.25 before the probe that
informed it existed.** Intact-configuration coverage rose **+0.2167** against a 0.15 bar.
Contamination stayed at 0.275 per scene against a bar of 1.0. And the newly recovered plantings
are **half the brightness** of those round zero already had -- 12.51 against 24.02 -- so peeling
reached the faint population rather than re-finding the bright one. Nothing was lost: no trial
recovered at round zero went missing after peeling.

**The declared failure mode is real, and it is most of the contamination.** Of 99 spurious
features, **83 (83.8%) fall within two fitted widths of something that was peeled** -- they are
lobes the subtraction created and the lowered cut then reported. And they concentrate exactly
where the declaration said they would, on the asymmetric features an isotropic fit cannot
represent:

```
stretch of the planting nearest each manufactured feature
   1.0 -> 18      1.5 -> 11      2.5 -> 54
```

So the procedure's cost is not a diffuse rise in background noise. It is a specific, predictable
artefact at stretched features, arising from `local_maximum_extractor`'s declared
`isotropic_gaussian_on_a_flat_baseline` shape model meeting features that are not isotropic. Only
16 of 99 spurious features are ordinary contamination.

**THE REPORTED SIGNIFICANCE MEANS SOMETHING ELSE, and this belongs here rather than in a
footnote.** `NullCalibration` states its hypothesis as no peak exceeding the strongest peak of a
field with **the same power spectrum** as the frame. A peeled ensemble has the residual's
spectrum. So every p-value taken through a peeled cut answers a different question, and the
coverage gained above was gained **partly by changing what a detection claims**. Whether the
residual's spectrum is the better null for the question actually being asked -- whether *this*
peak is distinguishable from the background it sits on, rather than from a field that includes
itself -- is an argument, is labelled as one, and is not settled here. A successor that adopts a
peeled null must state the new hypothesis explicitly.

**What the numbers do and do not say about the coverage problem.** Peeling more than triples
intact-configuration coverage and cuts never-seen features from 121 to 39. It does not solve the
problem: **70% of configurations still hold at least one feature the extractor never recovers**,
and the gap's first decile is 0.000 -- in a tenth of scenes one round closes nothing at all. The
top decile exceeds 1.0, meaning some peeled cuts fall *below* the background oracle; that is
reported rather than clipped, because a peeled null is not bounded by the oracle and pretending
otherwise would hide an overshoot.

**Disclosed: the gate median moved from 4 to 6.** Both are inside the declared band, and 6 is
*closer* to the record's own 7 than round zero's 4 was. So the peeled stage is the better density
match to the record -- which is worth stating plainly, and is also the first time in this
sequence that a gate reading has moved toward the record rather than sitting at the bottom of its
band.

**The blindness claim and its limit, restated because it governs how this may be cited.** A
one-scene feasibility probe preceded the declaration and informed the prediction, which is
therefore **not blind** and is not reported as if it were. The two acceptance bars are T4E.25's
and predate the probe. These are T4E.24's scenes on their **fourth inspection**. Nothing here is
confirmatory: a pass is a failure to fail on the evidence that motivated the question, not a
validation.

**What is not licensed.** No peeled null is adopted, no default is changed, and nothing in
`local_maximum_extractor`, `NullCalibration` or the frame-maximum statistic is altered. No second
round was run and none is reported. No other null construction was evaluated (R20). The oracle is
a ceiling, not an achievable operating point. The 3.53x figure from T4E.25 and the artefact rate
here belong to 850 hPa relative vorticity over this crop under this planting and are quoted for
no other domain -- though a domain adopting a peeled null owes its own artefact measurement,
because that rate depends on how badly its extractor's declared shape model fits its features.
D96 through D100 remain open, T4E.8's acquired-record acceptance is untouched, and no reserved
seed block or frame of the 2022-2023 forecast-test period was read.

**The bound still runs the same way.** Identical geometry across the six scenes remains the most
favourable case, so every figure above is optimistic and the truth under jitter, drift and
evolution is worse.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_peeled_null.py -q
17 passed
```

One test failed while writing this and was itself the error: it asserted a peeled baseline of 5.0
against a flat field that had no peak planted in it, so the code was right and the test was
describing a field it had not built. The test was corrected rather than the module.

### T4E.25 - what the detection cut costs, and what relaxing it buys

**T4E.25 (2026-09-11): the cut is not a lever, and the reason is not the one predicted.**

Adopted 2026-09-11 (`t4e25-coverage-contamination-adoption.json`, binding the declaration by
sha256 `ce3126fb...`) and measured the same day. 60 configurations at `S = 6`, T4E.24's scenes
replanted under the same root seed with the family-wise level `alpha` swept over the four
declared values. `measurements/t4e25_coverage_contamination.json`.

**Acceptance condition 2 first, because the declaration says nothing else may be interpreted
until it passes.** At `alpha = 0.05` the sweep reproduces T4E.24 **bit-for-bit**: marginal false
absence 0.3723832528 against 0.3723832528, 2,484 trials, 15 spurious, gate median 4, the presence
distribution identical key by key, the admission rates identical, and the addendum's
configuration figures at 10 and 5. The pairing is real, not asserted.

```
alpha  gate  feature   configurations      configurations   spurious   spurious   median ratio
       med   recovery  recoverable         intact           per scene  fraction   of recovered
0.05    4     0.6276    0.1667 (10/60)      0.0833 (5/60)     0.042      0.0095      24.0
0.10    5     0.6522    0.1833 (11/60)      0.1000 (6/60)     0.047      0.0104      23.7
0.25    5     0.6864    0.3000 (18/60)      0.1000 (6/60)     0.058      0.0122      23.2
0.50    5     0.7182    0.3500 (21/60)      0.1500 (9/60)     0.075      0.0149      22.8

step          d intact   d recoverable   d feature   spurious/scene   newly    median ratio
                                          recovery                    trials   of the new
0.05 -> 0.10   +0.0167     +0.0167        +0.0246    0.042 -> 0.047     61        16.9
0.10 -> 0.25   +0.0000     +0.1167        +0.0342    0.047 -> 0.058     85        14.0
0.25 -> 0.50   +0.0500     +0.0500        +0.0318    0.058 -> 0.075     79        13.1
```

No alpha was excluded by the gate. Recovery was monotone in alpha at every step, as the design
assumed; no trial was recovered at a lower alpha and lost at a higher one.

**The declared outcome held. The declared mechanism did not, and the difference matters.**

The prediction was: *"Contamination rises faster than configuration coverage at every step, and
no declared alpha brings intact-configuration coverage above 0.50 while keeping spurious features
below one per scene."*

The second clause **holds decisively**. Intact-configuration coverage reaches 0.15 at
`alpha = 0.50` -- a tenfold relaxation of the declared error level -- against a bar of 0.50. No
step met the declared falsification condition of a 0.15 absolute rise, so the verdict is
`PREDICTION_HELD`.

The first clause is **wrong**. Contamination barely moved: 0.042 to 0.075 spurious features per
scene across the whole sweep, never above 1.5% of everything extracted. Coverage rose more in
absolute terms than contamination did. The prediction was right about where the sweep ends and
wrong about what stops it, and reporting only the verdict would hide that.

**What actually stops it, measured rather than asserted.** A post-hoc diagnostic over 8 scenes,
adopting nothing and reporting no operating point, took the threshold in units of the
background's own robust width, from the scene's null and from the bare background's null:

```
alpha    planted-scene null    bare-background null
0.05          15.47 sigma            4.39 sigma
0.50          12.95 sigma            3.77 sigma

span 0.05 -> 0.50:  planted x1.195      background x1.165
the planted-scene null sits x3.53 above the bare-background null at alpha = 0.05
```

Two things follow, and only one of them was in the prediction.

**The tail is steep, as predicted.** A tenfold change in alpha moves the threshold by about 20%,
on the planted scene and on the bare background alike. That half of the predicted mechanism is
confirmed. But a threshold that moves 20% does not admit a flood of noise -- which is why
contamination stayed flat, and why the second half of the mechanism was wrong.

**The dominant term was not in the prediction at all.** The cut on a scene containing signal sits
**3.53x higher** than the cut on that scene's own background, because the plantings' power enters
every surrogate of the planted field. Against a level shift of 3.5x, a lever with 1.2x of travel
is not a lever. Alpha is not what sets this threshold; the signal is.

**This is the null behaving as declared, not a defect.** The hypothesis `NullCalibration` states
is *a structureless field with this power spectrum*, and the planted field's power spectrum
includes the plantings. Phase-randomising a field that contains coherent structure spreads that
structure's power across the frame and raises its own maxima. The result is conservative by
construction. It is the same effect recorded at T4E.24's calibration choice, where background
calibration gave recovery 0.940 against 0.774 -- now quantified as a threshold ratio rather than
inferred from a recall difference.

**What this licenses.** That coverage is not recoverable by relaxing the declared error level
within this scheme: a tenfold relaxation buys 9 percentage points of feature recovery and leaves
**65% of configurations still holding a permanently invisible feature**. T4E.24's coverage figure
is therefore a property of the detection scheme across a declared range, not an artefact of the
0.05 operating point -- which is exactly what this slice was declared to establish.

**What it does not license.** No operating point, no alpha, no change to any default (R20). No
change to the extractor, to `NullCalibration`, or to the frame-maximum statistic, and no
replacement for any of them. It does not establish that a scene-calibrated cut is the wrong
choice -- it is the choice the pipeline makes on the real record, and this slice measured its
price, not its correctness. The 3.53x figure is a diagnostic over 8 scenes with no error bar and
adjudicates nothing. Nothing here touches the atmosphere, T4E.8's acquired-record acceptance, or
D96 through D100. The reserved seed blocks and the 2022-2023 forecast-test period were not read.

**The bound still runs the same way.** These are T4E.24's scenes, so geometry is identical across
the six and the figures remain the favourable case. Real recurrence carries jitter, drift and
evolution; the true coverage is worse than every number above.

```
$ .venv/Scripts/python.exe -m pytest src/tests/test_coverage_contamination.py -q
12 passed
```

### T4E.24 - the false absence rate, and what it permits a tolerance to assume

**Why this task existed.** T4E.13 fixed `ABSENCES_TOLERATED = 1` as the minimal relaxation of
candidate 2 -- the only `k` below `S` nameable without choosing a free fraction -- and said in
its own declaration that the choice was **not** calibrated against any absence mechanism,
because none had been measured. T4E.21 then produced evidence of one from an unexpected
direction: at a feature density below the record's own, 80 of 247 planted features were never
recovered at all. The criterion consumes recovered features, so *absent from this scene* and
*present but suppressed* are the same observation to it, and the tolerance had been set without
knowing how often that happens.

**The quantity is not the one T4E.21 measured.** T4E.21 planted independently in every frame, so
it measured a marginal recovery rate over unrelated features and could not ask whether absences
fall on the same features repeatedly. The criterion does not consume marginal rates; it consumes
the number of scenes a *particular* feature is seen in. That count had never been measured.

**The design.** 60 configurations of 3-10 vortices, each planted with **identical geometry** into
6 independent phase-randomised backgrounds drawn from distinct frames of the acquired record --
360 scenes, 414 features, 2,484 trials, T4E.21's planting envelope and pairing rule unchanged so
the result speaks to the same regime. Geometry is held identical because it is the most
favourable case for recovery available: real recurrence carries jitter, drift and evolution, all
of which can only make recovery harder. So recall here is an **upper** bound and the false
absence rate a **lower** one -- the bound runs against the programme's own argument, which is why
it was accepted.

**Adopted 2026-09-11**, recorded in `data/identity_calibration/t4e24-false-absence-adoption.json`
and binding the declaration by content hash rather than revision, because it was untracked when
adopted.

**What was measured.** The gate passed at median 4 features per frame. Marginal false absence
**0.3724**, reproducing T4E.21's loss on a different design. The presence distribution is
bimodal: **121 of 414 features recovered in no scene at all**, 228 in all six, only 65 in
between. So the admission rate at `k = S - a` moves from **0.5507** at `a = 0` to **0.6304** at
`a = 3` -- eight percentage points for a threefold relaxation. `ABSENCES_TOLERATED = 1` sits at
**0.5870**.

**The finding is that the tolerance is not the constraint.** No value of `a` reaches a feature
that was never extracted anywhere, and 29% of genuinely recurrent features are in that class.
The declaration named this arm in advance as the worse one, because it is silent: a criterion
cannot fail on evidence that never reaches it.

**The mechanism was predicted, confirmed, and its stated cause corrected.** Presence counts are
heavily over-dispersed (variance 7.35 against a binomial 1.40, band [1.24, 1.58]), so the
predicted feature-intrinsic concentration holds. It was close to built in and the declaration
said so before measuring. But the declaration attributed it to suppression relative to
neighbours, and the breakdown says the driver is **amplitude against the detection cut**, nearly
alone: recovery 0.144 at peak-to-background ratio 8-14 against 0.919 at 26-32, while a fivefold
change in nearest-neighbour distance moves it only 0.535 to 0.677. The features that vanish are
the faint ones, not the crowded ones.

**Two things this slice did that its predecessors did not.** The generating code is committed --
`src/benchmarks/false_absence.py` and `tools/measure_false_absence.py` -- where T4E.20 and T4E.21
wrote receipts from scripts that cannot be re-run from this repository. And the detection cut's
source, which the declaration did not fix, is recorded as a choice with both options measured
before either was taken: calibrating on the bare background gives higher recall (0.940 against
0.774) but admits 63 noise peaks across 48 scenes against 0, so calibrating on the scene -- what
the pipeline does on the real record -- was chosen as the faithful and less flattering option.

**What it licenses.** Only that the extractor's coverage, not the tolerance, bounds a
partial-recurrence criterion on this evidence at `S = 6`. It chooses no tolerance and proposes no
proportion of `S` (R20), changes nothing in the extractor, does not supersede T4E.13, and says
nothing about the atmosphere. No reserved seed block and no frame of the 2022-2023 forecast-test
period was read. The rate belongs to 850 hPa relative vorticity over this crop under this
planting and is quoted for no other domain -- though the obligation to measure it transfers to
every domain that adopts the criterion.

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

**T4E.17 (2026-09-10): the external catalogue `kind_recurrence` has always required.** The
primary scientific target, untouched by everything the synthetic sequence did, and unevaluable
from the tool until now for want of a catalogue.

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

**What comes next** is a criterion declared against it -- written before these numbers shape
it, and unable to borrow anything from the synthetic sequence.

**T4E.16 (2026-09-10): candidate 5 declared and withdrawn before adoption, by derivation.**
The direct repair for the constraint the four measured results fix -- and it does not survive
its own feasibility check.

**The design.** The bar on a consistent set is set by a within-partition permutation surrogate,
per group size: admit a set of size `m` only if its diameter is tighter than the tightest set of
size `m` found in any of 199 surrogate replicates. A max statistic, so family-wise error across
every set of that size is controlled at 1/200 with **no alpha divided by anything** -- which is
what made candidate 1 inert. The bar falls with `m` automatically, because large coincidental
sets are rarer under the surrogate than small ones, and nobody chooses the rate at which it
falls. That is the constraint the four measured results jointly fix: **the evidence a group
carries must scale with the group.**

**Withdrawn, because the surrogate can reassemble the motif.** It redistributes the partition's
*actual* configurations. When the `S` motif copies land in `S` distinct scenes they form the
same consistent set with **exactly the same diameter**, so the strictly-tighter rule then
rejects the motif. The rate is combinatorial -- `S!/S^S` = 0.0154 at `S` = 6, confirmed by
simulation at 0.0166, 0.0168 and 0.0153 for `m` = 20, 84 and 220 -- and over 199 replicates that
is **0.9642, 0.9657 and 0.9535**. Candidate 5 would have failed conditions 1 and 5 at every
richness, with about 96% probability, for a reason having nothing to do with the signature. It
is a permutation null invalidated by its own permutations, the known failure of that
construction when the signal is a small set of near-duplicates and `S` is small.

**It cannot simply be repaired.** A valid null must *generate* fresh distractor scenes rather
than redistribute existing ones -- but then every pairwise distance changes per replicate, the
cached matrix that made the design runnable is worthless, and the cost returns to the measured
189 hours at richness 12 alone. **A feasible null is invalid here; a valid null is infeasible.**
Widening `S` would drive `S!/S^S` down fast, but `S` = 6 is the frozen partition size, and
changing it to rescue a criterion would be tuning the evidence to the rule.

**The declaration had labelled the point honestly, which is why it was checked.** Its derivation
section said of the large-`m` argument: *an ARGUMENT, NOT A PROOF ... if some surrogate replicate
produces a size-6 set tighter than the motif, the motif is rejected, and nothing derivable
excludes that.* Labelling it as argued rather than proved is what made it the next thing to
compute.

**What was not done.** Not adopted, not measured, not quietly altered into a variant that
passes. No surrogate was run on the T4E.14 evidence or on any block. The declaration is retained
unedited, superseded rather than deleted, because a design killed by derivation is part of the
record.

**The multiplicity position is unchanged.** A declaration withdrawn before adoption and before
measurement adds nothing to the accumulated multiplicity, because nothing was tested. **Seven
criteria have been measured in this sequence, not eight**, and both reservations -- 720-735 and
880-895 -- remain unspent.

**Two things survive and are kept**, because the next design meets the same arithmetic. The cost
finding: the clique enumeration is free at every richness and the matching is the entire
expense, at 0.8, 13.5 and 95.3 seconds per partition at richness 6, 9 and 12. And
`PooledDistances`, which computes a partition's distances once and keeps the metric's refusals
**as refusals** rather than as numbers -- optimisation must preserve the named refusal.

**What this does NOT license.** Not that size-scaled evidence is the wrong idea: the constraint
the four measured results fix is untouched, and what failed is one null, not the principle. Not
that the signature is inadequate -- no measurement was taken. Not that permutation surrogates
are wrong in general; this one is invalid *here*, at `S` = 6, against a signal of `S`
near-duplicates. And nothing about `kind_recurrence`, D96 to D100, a mining radius, or
recurrence across separated epochs.

**What this says about the sequence.** Seven criteria measured, six falsified, one passing but
non-transferable. The eighth was withdrawn before it could be measured, and for a reason that
is structural rather than incidental: at `S` = 6 a within-partition null cannot be built that is
both valid and affordable. That is a statement about what this evidence can support, not about
the signature -- and it is the strongest argument yet that further criteria on synthetic
partitions have reached diminishing returns.

**T4E.15 (2026-09-10): candidate 4 measured, and falsified.** Closure under the matching --
the first criterion in the sequence with nothing to tune, and the first measured on evidence
where both errors were available at once.

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

**What a successor must not do, which is more than the previous four failures gave.** Treat a
group of two as evidence on the same terms as a group of six. Candidate 2 worked because a
six-scene conjunction is hard to achieve by chance; candidate 3 showed one scene of slack
destroys that; candidate 4 shows that dropping the size requirement altogether inverts the rule.
The evidence a group carries has to scale with the group, and no criterion so far makes it do
so. That is a constraint on the next declaration, not a design for it, and it is the
maintainer's decision.

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

**The `k` profile, completed.** Its rungs reproduce candidate 2 at `k = 6` and candidate 3 at
`k = 5` exactly through a separate code path.

```
T4E.13 k PROFILE -- development blocks, totals over the four blocks
k    rich    admitted     motif  false adm      null
6    6             60        60     0.0000         0
6    9             60        60     0.0000         0
6    12            60        60     0.0000         0
5    6             60        60     0.0000         0
5    9             80        60     0.2500         0
5    12           154        60     0.6104         0
4    6           129        60     0.5349         6
4    9           374        60     0.8396        87
4    12          834        60     0.9281       132
3    6           336        60     0.8214        72
3    9          1196        60     0.9498       314
3    12         3293        60     0.9818       778
```

**The motif column never moves.** 60 pairs at every rung and every richness -- four blocks of
15 -- which is the monotonicity the T4E.13 declaration derived before any of this ran:
admissions are non-decreasing as `k` falls, so recall is inherited and is not a finding at any
rung. Everything informative is in the other three columns.

**The null breaks between `k = 5` and `k = 4`.** At `k = 6` and `k = 5` it admits nothing at any
richness. At `k = 4` it admits 6, 87 and 132 pairs, and at `k = 3`, 72, 314 and 778 -- where
nothing recurs at all. So the one-scene-wide margin is not merely where specificity against
*structured* coincidence runs out; just below it the criterion begins manufacturing identity out
of noise, which is a different and worse failure.

**False admission climbs monotonically** from 0.0000 to 0.9818, and at `k = 3`, richness 12,
3,293 pairs are admitted of which 60 are true. Nothing below `k = S` is recoverable by widening
further, which the same monotonicity already guaranteed.

**This adjudicates nothing.** R20 forbids the horse race and the T4E.13 declaration forbids it
by name: the criterion under evaluation was `k = S - 1` and it is falsified on its own
conditions. A `k` made attractive by this sweep would be a further candidate needing its own
declaration and its own evidence, and it could not claim its structural choice was blind,
because this profile has now been seen. That consequence was recorded in advance of running it.

**T4E.14 (2026-09-10): evidence in which recurrence is partial.** The maintainer authorised a
criterion built for partial recurrence *if that is what the work needs*. Resolving that
conditional against the evidence, rather than assuming it, showed both authorised options were
untestable as things stood: recurrence across partitions is not a distinct phenomenon in this
generator, and partial recurrence was absent from the evidence entirely. So the first thing the
work needed was a test bed, and this is it.

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

**What comes next, and what it may not borrow.** A criterion declared against these partitions,
with its own acceptance and its own blindness claim -- written before the audit's numbers are
used to shape it. It may not borrow candidate 2's `k = S`, which is arithmetically dead here,
nor candidate 3's `k = S - 1`, which is falsified and recovers only at `j = 5`. That declaration
is the maintainer's decision and nothing here authorises it.

**T4E.13 (2026-09-09): candidate 3 measured, and falsified on a choice made blind.** The
minimal relaxation of candidate 2 -- tolerate exactly one absence, `k = S - 1` -- admits
coincidences that `k = S` excludes.

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

**FALSIFIED on conditions 3 and 2.** False admission reaches 0.4000 at richness 9 and 0.8000
at richness 12, against a declared bound of 0.10 at every richness individually -- four to eight
times over -- and the shortfall widens with richness rather than staying non-increasing, which
the same condition forbids separately. Condition 2 fails as well: on block 300-305 the matched
fraction goes 0.0500, 0.0198, 0.0227, rising at the richest level instead of falling. Three
blocks fall as required; the condition is stated per block, not as an average, so one is enough.

**The declaration predicted the direction, before the run and in writing.** Coincidental groups
do reach `S - 1`, and the greedy probe under-counted them. What the measurement adds is the
size: at richness 12 on block 300-305, 75 pairs admitted where 15 are true.

**Condition 4 held, and it is worth recording precisely because the candidate failed.** Where
nothing recurs, tolerating one absence still admits **nothing** -- 0 of 1486 proposed pairs at
richness 12. The relaxation did not break the null; it broke the planted blocks, where
coincidences have real structure to be coincidental with.

**Condition 1 is arithmetic and is not a finding.** The 0.0000 false split everywhere was
derived before the run: admissions are non-decreasing as `k` falls, so candidate 3 inherits
candidate 2's recall by monotonicity. The declaration said this in advance so it could not be
reported as evidence afterwards, and it is not being.

**What the blindness claim bought.** `k`, its rule and every structural choice were committed at
sha256 `55631fa2...` before any measurement at any `k` below `S`, and nothing moved after the
numbers appeared. The claim is now spent -- it cannot be made again for these scenes -- and what
it purchased is that this is a real falsification rather than a criterion that failed to survive
its own tuning. That is the whole return on the slice, and it is a smaller return than a pass
would have been.

**The margin is exactly one scene wide.** Consistency across the whole partition excludes
coincidences; consistency across all but one scene does not. By the same monotonicity the
profile's remaining rungs cannot rescue anything -- `k = 4` and `k = 3` admit supersets of
`k = 5` -- so the profile characterises how fast it degrades and adjudicates nothing.

**Blocks 720-735 stay closed and now will not be spent on this.** The adoption made spending
them conditional on candidate 3 meeting its conditions, and it did not. That conditionality was
recorded before the numbers existed, which is the only reason it is worth anything now.

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

**What would now be worth declaring.** Not another value of `k`: the profile shows the
degradation and monotonicity settles the ordering, so no rung below `S - 1` can pass conditions
`S - 1` already failed. A criterion built *for* partial recurrence -- one that scores how much
of a window a configuration spans rather than thresholding it, or that admits on evidence
accumulated across partitions rather than within one -- is a different object and would need its
own declaration. That is the maintainer's decision and nothing here authorises it.

**T4E.13 (2026-09-09): candidate 3 declared, and the structural choice made blind.** The
step candidate 2's own limitation section demanded: `k = S - 1`, by the rule *tolerate exactly
one absence*, fixed in `t4e13-partial-recurrence-declaration.json` and in code before any
measurement at any `k` below the partition size. Not adopted, and measured nowhere -- eight
tests assert that absence rather than intend it.

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

**What would make the last two reserved partitions worth spending.** Candidate 3 meeting its
conditions on development evidence, and nothing less. 720-735 are the programme's last untouched
partitions and a confirmatory pass on a falsified candidate would spend them for nothing. If
candidate 3 does pass, a confirmatory pass there would establish the thing candidate 2's could
not: that a blind structural choice produced the result.

**Confirmatory evaluation (2026-09-09): the reserved evidence was split, and candidate 2
reproduces.** The maintainer opened **two** of the four reserved partitions -- 700-705 and
710-715 -- under `t4e12-confirmatory-amendment.json`, and kept the other two.

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

Every partition admits 15 pairs, the motif's complete clique, and nothing else, out of 122 to
1518 pairs candidate C proposes. Both error rates 0.0000 throughout. The development result was
not an artefact of the blocks it was developed on.

**720-725 and 730-735 have never been generated and are refused in code even under
`confirmatory=True`** -- `STILL_RESERVED_CONFIRMATORY_SEEDS`, enforced rather than intended.
They are kept for a criterion whose key structural choices are fixed without seeing them, most
obviously a criterion of this shape with `k` below `S` declared before anyone probes how
coincidental groups behave at that `k`. Spending them needs a further amendment.

**What this pass still cannot establish.** Not that a blind structural choice would have
produced it: `k = S` followed a probe, and fresh scenes cannot retrospectively make that choice
blind. Not anything about rejection where nothing recurs -- no confirmatory null was declared,
so that evidence remains development only. Not partial recurrence, not recurrence across
partitions, no mining radius, no discharge of T4E.8's acceptance, no closure of D96 to D100, and
nothing about `kind_recurrence`, which still has no catalogue. The maintainer's framing was
fixed before these numbers existed and is unchanged by them: a strong diagnostic of whether the
signature can sustain coherent identity, not a defensible real-world recurrence rule.

**Candidate B outcome (2026-09-09): adopted before measurement, falsified by it.** The
normaliser was pinned in the declaration -- within a partition, each configuration's distance to
its nearest neighbour among other scenes, then the median -- and adopted in a separate record so
the declared artifact kept its hash. Measured on the four development blocks, the raw spread of
1.876x became **1.991x**: dividing by that scale left the disagreement wider than it found it.

The reason is measured and is narrower than the declaration anticipated. The normaliser varies
only 1.084x while the same-configuration scale varies 1.876x, sits 7-14x above it, and
correlates -0.21 with it. A median over every configuration's nearest cross-scene neighbour is
dominated by the unrelated majority -- 114 of 120 configurations in a block are not the motif --
so it measures the nearest-*unrelated* distance, a different regime from the one the radius
works in. **The declaration's stated falsification reasoning was wrong**: it said failure would
show the shape moving rather than the scale, and that is not established. The scale still moves;
this normaliser does not see it. That correction is recorded in the adoption record rather than
edited into the declaration, which stays byte-identical and still verifies against the hash it
was adopted under.

**What this does not authorise.** Trying normalisers until one works. A different normaliser is
candidate B-prime and needs its own declaration and adoption; the alternative is candidate C.
Either way the next step is a declaration, not a measurement. The reserved confirmatory blocks
were not built and remain reserved.

**Candidate C outcome (2026-09-09): falsified, and informative.** Declared as mutual
nearest-neighbour per ordered scene pair -- no radius, no threshold, no normaliser, so nothing
to carry between partitions. Two corrections were made *before* adoption after external review:
the one-to-one assumption was restated at the instance level, because the draft had misdescribed
the criterion as permitting one match per scene pair when the rule returns a partial matching
that can hold many; and the scope was fixed as a domain-agnostic mechanism carrying atmospheric
evidence only, since cross-domain use needs its own declared experiment under R19.

**Split is 0.0000 in all four blocks.** Every construction-labelled motif pair is recovered,
across blocks whose magnitudes differ 1.876x. Neither a frozen radius nor a normalised one could
do that, so **the ordering transfers where the magnitudes do not** -- a finding about this
signature that survives the candidate's failure. What fails is rejection: mutual
nearest-neighbour always returns a match, so the null block yields **116 matches where the
answer is none**, admission 1.0000, and the planted blocks return 120-150 matches of which 15
are motifs.

**The declaration's escalation was wrong and is corrected.** It said failure would implicate the
signature rather than the criteria. It does not: the shape transferred, and what is missing is a
rejection test, which can itself be rank-based and scale-free. That is candidate D and needs its
own declaration; adding it here and re-measuring is what R20 forbids. This is the second
consecutive declaration whose falsification reasoning over-reached, and that pattern is recorded
in the adoption records rather than left for a reader to notice.

**Dependency.** This blocks D96's algorithm work, because the clustering's workload is set by
whatever identity criterion it consumes, and it is what makes the identity-target decision
actionable: whichever target is chosen needs a criterion that transfers.

### Phase 4F - Transition and Precursor Mining

**T4F.1 New substrate.** The existing engine mines *scalar run metrics* out of flattened `results` JSON and structurally **cannot** express `A4 -> A8 -> B8 -> C16`. This needs an event table and sequence counting, not another correlation loop. - **DONE**

**Met, as the event table only.** `src/analysis_engine/spectral_events.py`;
`src/tests/test_spectral_events.py`, 18 test functions, **18 passed**. Sequence counting is
T4F.2 and this module counts nothing: `events_from_catalogue` reads a T4E.3 `PatternCatalogue`
into an ordered `EventSeries`, and its claim boundary says a span is not a recurrence, a period,
a precursor, a rate, a support count or a discovery.

**The grid is the slice.** A catalogue records what was *found* and can never record what was
*looked at*, so `ObservationGrid` is a separate mandatory argument carrying the searched frames,
their unit, and optionally the cadence they sample. Three refusals follow. The time unit is
mandatory, because T4F.2 asks whether the gap between occurrences recurs and a number without a
unit cannot answer that. An occurrence at a frame the pass never searched is refused by name,
because it asserts a sighting where nothing looked. And a span crossing an instant that was never
read is `SPANS_UNOBSERVED_TIME` with its missing count, not a longer gap -- decidable only against
a declared cadence, and reported `COVERAGE_UNDECLARED` rather than assumed complete when no
cadence is given.

**Simultaneity is not order**, which is the whole point of the substrate the task asks for. Two
events on one frame are published through `co_occurrences()` as unordered; a succession read off
list position would be an arrow supplied by tuple comparison rather than by the record, and
`A4 -> A8` is exactly the arrow that must not be fabricated. Event order is asserted invariant to
input permutation for the same reason.

It found **D93**: T4E.2 signed each constellation with a bare float `time` and dropped the
`time_units` its `FrameConstellation` carried, so every clustered pattern downstream held an
unnamed clock. Nothing had caught it because comparison, clustering and support counting are
none of them dimensional -- T4F.1 is the first consumer to subtract two times. Fixed at the seam
rather than by re-declaring the unit in each consumer, and the grid then checks the members
against its own declaration rather than trusting either side alone. No existing result changes,
because no duration had ever been derived from a signature.

**T4F.2 Sequence mining.** Frequent sequences over feature/constellation events with support and confidence; recurrence-interval detection (does the *gap* between events recur?). - **DONE**

**Met.** `src/analysis_engine/spectral_sequences.py`; `src/tests/test_spectral_sequences.py`,
33 test functions, **33 passed**. `A4 -> A8 -> B8 -> C16` is now counted rather than merely
expressible: the planted four-fold chain is recovered at both lengths with support 4 and
confidence 1.0, and the reversed chain is present in the receipt as a zero rather than omitted.

**A transition is undefined without a declared window**, so `TransitionWindow` is mandatory and
its minimum lag is strictly positive -- a zero minimum would let two events on one frame form a
step, which is the arrow T4F.1 built the substrate to avoid fabricating. A window admitting no
lag on the grid's own cadence lattice is refused rather than counted as zero, because a zero from
an unreachable window measures the window and not the record.

**The denominator is the honest part, and it is computed.** Confidence is the share of antecedent
occurrences that were followed, and an occurrence enters that ratio only if the whole window it
could have been followed in was actually searched. `ObservationGrid.observation_of_window` -- one
method added to T4F.1's grid, because the grid is what knows what was looked at -- separates
`WINDOW_MEASURED` from `WINDOW_TRUNCATED_BY_RECORD` and `WINDOW_SPANS_UNOBSERVED_TIME`.
Right-censoring is the asymmetric one: counting an antecedent the record ended before as
unfollowed drags every confidence down by exactly the tail of the record, so a pattern firing
late would look less predictive than one firing early for a reason about the record's edge rather
than about the pattern. Excluded occurrences are counted rather than silently dropped, and a
completion seen at an inadmissible antecedent is published as a sighting that is not admissible
into a ratio. Without a cadence the count stands and the ratio is refused.

**The pruning rule is proved.** Extending a sequence lengthens the window an antecedent must have
observed and a completing chain's prefix completes, so support is antimonotone under extension;
candidates are generated only from sequences that met the minimum, and the acceptance suite
asserts the inequality over every extension the sweep reached rather than trusting the argument.
T4E.4's `MiningBudget` is reused rather than a second budget declared.

**A repeated gap is not a period.** Spans are tallied on the cadence lattice, which is the finest
interval the pass can resolve. A span crossing unobserved time *bounds* an interval from above
rather than measuring it -- an unseen occurrence inside would split it in two -- so it is excluded
from the tally, counted as excluded, and kept out of the reported longest gap. The receipt
publishes how many distinct interval values the record was long enough to hold, because among few
of those a repeat is expected under no structure at all and a concentration without that
denominator invites over-reading.

Nothing here is significance. Both claim boundaries say so: support and confidence are not a base
rate, a lift, a surrogate comparison, a p-value, a precursor or a cause, and a repeated interval
is not a period, a frequency or an oscillation. Those begin at T4F.3.

**T4F.3 Precursor tests *(R1, R4, R5, R7, R9)***. The core question, stated precisely: **which small attributed graphs at $t$ predict which larger structures at $t+\Delta$?** For each candidate constellation pattern, does its appearance raise the probability of a coarse-scale structure emerging at $t+\Delta$ *above its base rate*, above the surrogate ensemble, at admissible lags, under FDR control? Every emitted rule carries support, confidence, base rate, lift, CI and surrogate-corrected lift (R9), and uses precursor/signature language, never causal language (R7). - **DONE**

**Met.** `src/analysis_engine/spectral_precursors.py`; `src/tests/test_spectral_precursors.py`,
47 test functions and **52 passed**, one of them parametrised six ways. Every rule carries R9's six figures or names which one is
undefined and why, and `PrecursorRule.figures()` builds the programme's existing
`AssociationFigures` rather than a seventh private home for the same six numbers -- so a rule
that cannot satisfy R9's contract is refused by that contract instead of by a check in this file.

**The base rate is a window probability and it is measured, not assumed.** Confidence is the
chance that a declared window following an occurrence contains the consequent, so dividing it by
the fraction of *frames* carrying the consequent would give a lift that grows with the width of
the window and with nothing else. The same window is therefore dropped at every searched position
whose window was wholly observed -- the eligibility rule that decides an antecedent's
admissibility, applied to the reference -- and the rate is published with the number of positions
it was estimated over. The reference is not purged of the antecedent's own occurrences, because
purging would make the denominator depend on which rule is being tested; the direction of that
choice is conservative and it is stated rather than hidden.

**The null is a statement about alignment, and the choice of it decides findings.** The default
rotates the antecedent's occurrence times on the searched lattice: the count is preserved
exactly, every gap but the wrapped one is preserved, the consequent is untouched so the base rate
is invariant, and only the alignment between the two patterns is destroyed. Rotations below the
widest declared lag plus a cadence step are never drawn, because such a surrogate keeps part of
the alignment under test. The uniform-relocation null is provided and is anti-conservative by
construction, and the suite measures what that costs on one unchanged record: a clumped
antecedent in front of a dense block of the consequent reads lift 5.25 either way, is **not
distinguished** from the shifting null at p = 0.11, and is called a precursor by the scattering
null at p = 0.01.

**A lag chosen by the data is a search, so the null is over the choice.** Two families are
reported and each is corrected on its own: every (antecedent, consequent, window) triple, all of
them reported so no selection precedes the correction; and one selected lag per pair, referenced
to the distribution of the maximum across the declared family. That maximum has a null because
one draw is shared across the family per antecedent, and the sharing is held to by test rather
than asserted -- the selected null must be the elementwise maximum of the per-lag ones, and a
window's ensemble must not depend on whether it was declared first or second.

**A design that could not reject is refused before anything is counted.** An under-powered study
reports an absence indistinguishable from a real one, so `check_power` is asked first and an
ensemble too small for the declared family raises. This honours T4C.5i's boundary by never
producing the absence, and the refusal is a fact about the declared design rather than a result.
The ensemble each p-value came from is published on the rule, not only summarised, because a
p-value is a statement about a distribution.

**Nothing is reimplemented and nothing is claimed beyond a signature.** The counting is T4F.2's
`count_sequence`, made public here so a second denominator cannot exist; the p-value is
`significance.surrogate_p_value`; the correction and the power check are
`multiple_comparisons.adjust` and `check_power`. A rejected rule is a precursor signature and the
boundary refuses *cause*, *driver*, *mechanism*, *trigger*, *forecast* and *intervention* by
name (R7), states that both patterns following a third thing this record does not contain is
consistent with a rejection, and states that a rule is about this record and not about the
world.

**T4F.4 Bidirectional queries.** Top-down (given a coarse structure, what fine configurations most commonly preceded it?) and bottom-up (given a fine configuration, what coarse structure follows?) share one query engine over the same tables. Neither direction requires telling the platform what a cyclone or a front is - the features are discovered from coefficient structure alone, and physical interpretation is the researcher's step afterwards. - **DONE**

**Met.** `src/analysis_engine/spectral_queries.py`; `src/tests/test_spectral_queries.py`,
58 test functions and **58 passed**. Both directions are one selection over the table T4F.3
already produced: the direction fixes which role the target plays in the rule and which side of
the scale ordering the counterpart must sit on, and nothing else. A top-down entry and the
matching bottom-up entry are asserted to be the very same row object, so a fix to one direction
cannot fail to reach the other.

**A query is a view, and a view is not a test.** The shortcut this module exists to refuse is
re-running the inference restricted to the pattern being asked about: one target, a handful of
counterparts, a small family, and every q-value falls. That is choosing the family after seeing
the record. Nothing here recomputes a p-value, a q-value or a status; the rows are the report's
own, and the family they were corrected against is published beside every answer. The cost of
the shortcut is measured rather than asserted: on the acceptance record the same rule's q-value
falls from 0.0417 to 0.0050 when the declared family shrinks from four tests to one, and a
caller who genuinely wants the narrower family must declare it before the record is read by
passing `pairs=` to `precursor_report`.

**Coarse and fine are measured from the catalogue, and the record is allowed to refuse them.**
`scale_ordering` reads the member scales in the catalogue's own units and builds an interval
order: a pattern occupies the range of scales its members actually spanned, and one pattern is
finer than another only when those ranges are disjoint. Ranges that overlap -- or merely touch,
since sharing a scale is sharing a scale -- are not orderable, and naming one of them the
coarser would be an arrow taken from a sort rather than from the record. The relation is
partial by construction and the pairs it cannot order are published; on the acceptance record
three of the fifteen pairs are unorderable and one of them is withheld from the top-down answer
for exactly that reason. Cardinality is not scale, and the suite holds the ordering to that.
Under T4E.2's scale-invariant mode the refusal is exact rather than cautious: that mode divides
every signature's scales by their own geometric mean, so the statistic is 1.0 for every pattern
in the catalogue and an ordering built on it would order floating-point residue.

**The ranking is a scientific choice and this task's own wording names the wrong one.** "What
most commonly preceded it", answered literally, ranks by how often the counterpart was followed
by the target and so returns the record's commonest pattern whatever it precedes. Every key is
available, every entry carries its rank under every key, each misleading key carries the
sentence saying how it misleads, and the answer publishes whether the keys disagree about what
comes first. On the acceptance record they do: ranked by frequency the leading answer is a
pattern the null did not distinguish (support 9, lift 1.17, q = 1.0000), and ranked by q-value
it is the planted precursor (support 6, lift 4.02, q = 0.0417). The choice of ranking decides
the answer, as the choice of null decided the finding one task earlier.

**An empty answer says which kind of empty it is.** Nothing distinguished from the null,
nothing on the required side of the ordering, and nothing on that side that could be measured at
all are three different findings and only the first is a negative result. Every considered row
comes back including those that found nothing, and the rows naming the target are counted apart
from the whole family so the denominator is visible.

**Nothing is claimed beyond a signature.** The claim boundary refuses *cause*, *driver*,
*mechanism*, *trigger*, *forecast* and *intervention* by name (R7), states that a query is not a
second test, and states that the direction words are about scale rather than influence: top-down
names a query whose target is the coarser of the two patterns and asserts nothing about which of
them acts on the other. `directional_pair` puts the two orderings of one pair side by side
because confidence divides by a different denominator each way, so neither can be read off the
other. **Mutation testing: 61 mutations in two batches, 37 + 24, all caught after the eleven
gaps the first passes found were closed.**

**T4F.5 Evidence projection - the "show the why" step.** A mined rule is useless to a researcher as a row in a table. Every pattern must project **back onto the map**: the specific grid cells, lat/lons, pressure levels and timestamps whose coefficients constitute it, overlaid on the physical field they came from. This is why T3.5.7's undecimated SWT matters beyond convenience - because every scale shares the parent grid, the inverse mapping from a coefficient structure to its physical footprint is direct rather than an interpolation guess. - **DONE**

Each rule therefore carries: its constituent features' physical coordinates, the field values there, the anomaly magnitudes, the per-scale contribution, and the list of **historical instances** supporting it. A researcher must be able to answer "why does the platform believe this?" by looking at weather, not at coefficients.
**Acceptance:** for a planted synthetic precursor, projection recovers the exact grid cells that were planted. On real ERA5, a `recognised` pattern from T4F.6 projects onto a physically sensible footprint - verified by eye, which is a legitimate acceptance test for this specific task.

**Met.** `src/analysis_engine/spectral_projection.py`, verified by
`src/tests/test_spectral_projection.py` (67 test functions, 67 pytest cases, 3.2 s). A rule is
an ordered pair of integers and an integer is not evidence; this task puts one back on the
parent grid.

**A footprint, never a pixel, and the acceptance measures why.** A coefficient is the response
of a filter covering many parent cells, so a projection is the set of cells inside the
transform's own `filter_support` -- the same number R13 cuts the contaminated margin with -- and
its extent is that measured support rather than the dyadic octave label the scale ratios use (16
cells against 8 at haar level 4). This matters because a detail wavelet is derivative-like: a
symmetric structure has no maximum at its centre and two on its flanks. On the T4D.2 acceptance
record, where the planted vortex's cell is known in every frame, the flank offset is 1.08 times
the vortex's own width at level 4 and 1.26 at level 5, and **not once in twenty-four frames does
a member's peak cell coincide with the planted cell** -- a single-pixel projection would have
been wrong by six to twelve cells while looking precise to two decimals.

**The acceptance clause is met with a boundary in it, and the boundary is the finding.** The
footprint recovers the planted cell exactly while the level that found the structure can still
reach back to it: level 4 reaches eight cells and holds the planted cell out to frame 9, losing
it in every one of the fifteen frames after, and level 5 reaches sixteen and holds it in all
fifteen frames it detects anything. The suite checks containment against the geometry rather
than against the code, so the two agree rather than one asserting the other. A coefficient at a
level far finer than the structure points away from it by more than its own reach, and the
honest projection is then empty of it: **evidence should be read at the level that resolves the
thing.**

**An intersection is not a location either, and the measurement corrected the prose.** Two bands
of one level that resolve different axes have flanks pointing different ways, and their overlap
straddles what excited them: the LH/HL pair contains the planted cell in 11 of 11 occurrences.
Two bands of one orientation at two levels have flanks pointing the *same* way and their overlap
sits beside the structure: 11 of 11 occurrences exclude it. The receipt warns whenever every
member of an occurrence shares an orientation, and both counts are pinned, because the second is
what stops the first being read as a rule.

**Each grid, level and clock says only what it can.** A `latlon` grid gives degrees in its own
convention with the seam wrapped; a `cartesian` grid gives offsets in metres from the crop's own
origin and names them a distance rather than a place; a `pixel` grid gives cells and refuses
degrees by name. The task asks for pressure levels: a level with no declared axis is reported as
a number rather than as hectopascals (TG1.5), and the acceptance record declares no vertical
coordinate at all, which the projection says rather than filling in. A frame is a frame unless
the record supplied carries a calendar -- a decomposition's clock is float seconds by
construction, so the field cannot know whether its numbers are dates and the record can.

**The field values and the anomaly magnitudes are inputs.** Values under a footprint are read
from the record the coefficients came from, checked against the decomposition's length, grid and
clock; an anomaly is read from a supplied anomaly record. A raw value is never called an anomaly
and this record's own time mean is never quietly subtracted to make one, because which baseline
was removed belongs to whoever removed it (R11). With nothing supplied the answer is that the
values are unknown, which is smaller than the wrong one and is not zero.

**The historical instances are the ones the rule was counted on.** The per-anchor decision now
lives once, in `spectral_sequences.anchor_verdicts`, which `count_sequence` tallies into a
support and a denominator and this task lists as instances -- two implementations would have
agreed on the planted case and diverged exactly at the record's edge. Every enumerated total is
reconciled against the figures the rule published before anything is shown, and a series that
disagrees is refused rather than displayed. Anchors that did not support the rule are listed
under their own verdicts, and on the acceptance record the anchor at 118 *is* followed at 119
and is still not support, because the rest of its window was never watched.

**Per-scale contribution is a share of this pattern's own members and of nothing else,** since
the bank is undecimated and therefore redundant: per-scale coefficient energies do not partition
the field's variance, and a structure straddling two levels appears in both shares. The sentence
saying so travels with every share.

**Nothing is claimed beyond a footprint.** The claim boundary refuses *cause*, *driver*,
*mechanism*, *trigger*, *forecast* and *intervention* by name (R7) and states the thing this
task most invites a reader to forget: a footprint is where a coefficient of this transform was,
not where a structure is. **Mutation testing: 67 mutations in two batches, 43 + 24, all caught
except one shown to be equivalent** -- `_interior_halfwidth` takes its level from the scale
label whenever that label is an integer, and `spectral_tracking` refuses any scale label that is
not, so the positional fallback is unreachable for anything that can produce a constellation.

**Not met, and not yet runnable.** The second half of the acceptance -- "on real ERA5, a
`recognised` pattern from T4F.6 projects onto a physically sensible footprint, verified by eye"
-- depends on a task that does not exist yet and on a real-ERA5 mining pass that has not been
run. It is outstanding rather than met, and T4F.6 is where it comes due.

**T4F.6 Known-phenomenon cross-reference *(implements R10 - this is a validation gate, not a feature)*** -- PARTIAL (the gate is built and discriminates; it has not been run, and as of 2026-09-07 it **cannot** be: the mining pass it adjudicates is blocked by **D96**, an identity step measured at about O(n^3) against a training period presenting ~843,000 constellations). Maintain a small reference catalogue of known synoptic precursor phenomena with their expected scale ranges, lags and geometries. Every mined pattern is checked against it and labelled `recognised` / `unrecognised`.
**Acceptance:** on a real ERA5 period containing a documented cyclogenesis event, the mining pass ranks a `recognised` pattern corresponding to it in the top results. **If nothing recognisable is recovered, the pipeline is presumed broken and 4G does not start.** Only then are `unrecognised` high-lift patterns promoted for human review.

**Delivered.** `src/analysis_engine/spectral_reference.py` and 72 tests in
`src/tests/test_spectral_reference.py`. A `Phenomenon` declares a citation, the `observables` a
record must carry, a horizontal-scale envelope in kilometres, a lead envelope in hours and
optionally a cardinality and a member separation; a `ReferenceCatalogue` carries a sha256 over
its own canonical content. Every mined pattern is measured off the footprints T4F.5 drew and
labelled against every entry.

**A gate is worth what it can fail, so the failure is what was built for.** All three verdicts
are reached on the same seven real patterns and the same ranked report: `PASS` against a
catalogue whose envelopes fit the planted level-4 pair, `FAIL` against one whose envelopes do
not, and `INVALID` against one so wide that nothing could have fallen outside it. `INVALID` is
not a `FAIL` and says so in its own reason -- T4C.5i's boundary arriving at the physical gate --
because an under-powered gate recorded as a negative finding is how an instrument's blindness
becomes a result. **The ranking key decides the verdict**, which is T4F.4's finding carried
forward: at a declared top-N of one the same report passes ranked by `q_value` and fails
ranked by `support`, so the key,
the top-N, the catalogue digest and a required `period_justification` naming the documented
event and its source are all hashed into a `GateDeclaration` before the gate runs.

**The catalogue is a scientific declaration and this module refuses to sign one.** It is `draft`
until `freeze(signed_by=...)`, `physical_gate` refuses a draft outright, and the digest excludes
the signature so that signing an unchanged catalogue leaves a receipt traceable to the draft
that was reviewed while editing any envelope moves it. `draft_southern_ocean_reference()` ships
four entries for the T4C.5k record with real citations and envelopes that are **the
implementer's first reading and not values quoted from any paper**, which is why it is a draft.

**The measurement that limits this gate, taken from the record itself.** A scale in cells
becomes a scale in kilometres through the grid's own metric, and on the T4C.5k crop that metric
is not one number: 27.80 km meridionally everywhere, 13.90 to 26.12 km zonally across 40 degrees
of latitude, a factor of 1.879 and a 61.08% variation `GridSpec.anisotropy` already warns about.
So sixteen cells there is 222.4 to 444.8 km -- an interval spanning a factor of two before any
measurement error -- and an envelope narrower than that cannot exclude anything. The gate
therefore counts the exclusions the catalogue actually made and returns `INVALID` when there
were none.

**Three labels, not two.** An entry needing an observable the record does not hold is
`unassessable` for every pattern rather than `unrecognised`, and the shipped catalogue's
`upper-level-pv-precursor` is permanently so: classical cyclogenesis is an upper-level
disturbance over a low-level baroclinic zone, and a single-level record of 850 hPa temperature
has no upper level in it. Consistency established on one criterion alone is also `unassessable`
rather than `recognised`, and the observable check never counts as one of the two.

**A lead in hours needs this record's clock.** The cadence is measured from the record's own
timestamps and refused when they are absent or uneven; a window is converted only after checking
that every frame the event series searched is a frame of the record, so a series counted on
another lattice has its lead refused by name rather than multiplied by a cadence that does not
describe it.

**Not met.** The acceptance above has **not been run and cannot be run yet**, and this task is
therefore `PARTIAL` and **Phase 4G remains gated**. Three things are missing and none of them is
code: (1) a maintainer must review, correct and freeze a reference catalogue, since the shipped
one is explicitly a draft; (2) a documented cyclogenesis event on the 2018-2023 record must be
declared with its source, which is the `period_justification` the declaration requires; (3) the
mining pass itself -- T4D through T4F.5 on the real 8,764-frame record, on anomalies with a
training-only climatology per R11 -- has never been run, on this record or any other. Until all
three happen there is no `recognised` pattern on real data, and **T4F.5's own second acceptance
clause, which waits on one, is still outstanding.**

**T4F.7 Cross-region generalisation *(implements R14)*** -- DONE. Re-test every candidate pattern on held-out regions; report where it holds and where it fails, with physiography noted. A pattern is labelled `regional` or `general` accordingly.

**Met.** `src/analysis_engine/spectral_regions.py` and 54 tests in
`src/tests/test_spectral_regions.py`. A declared `RegionPartition` -- hashed, exactly one
discovery region, no overlaps -- is re-tested with each region's own T4F.3 `precursor_report`,
and the held-out regions are corrected as one family.

**The acceptance is a record built to disagree with itself.** Five boxes of one 260-frame
field: `A` where the rule is found, `B` built identically, `C` built with the same two
structures at a longer lag, `D` built with the consequent *before* the antecedent, and `E` left
empty. The module returns four different answers, and has to: the rule **holds** in `B`
(lift 4.47) and `C` (lift 3.72) after correction over four declared held-out regions, **does not
hold** in `D` (support 0 of 24, lift 0.0), and `E` is **not assessable** because it never carried
the antecedent at all. Verdict `regional`. Declaring three held-out boxes instead of four returns
`general` on the same record.

**A region where nothing occurred did not fail the test; it never took it.** That is the
distinction the task turns on, and it has two forms kept apart: a region carrying no occurrence
of the antecedent, and one carrying the antecedent but never the consequent, where the base rate
is zero and no lift exists. Rendering either as `regional` would turn an absence of data into
evidence of locality.

**A region is held out only if the identity was not fitted on it.** `match_into_catalogue` holds
the centroids and the calibrated radius fixed so a held-out region can supply occurrences without
helping to define what a pattern is, and `identity_leakage` refuses `general` when it did --
R6's leak with a map in place of a calendar. Measuring this found that **T4E.2's signature cannot
distinguish two band orientations, by construction**: it is invariant to rotation, so on this
record the same `L3/HL` signatures sit a median 0.447 from their own centroid and 0.585 from the
`L3/LH` one, both far inside a calibrated radius of 0.959. A catalogue whose patterns differ only
by orientation therefore cannot be matched into by signature distance, and the suite's own
catalogue is declared rather than fitted for that reason.

**Membership is decided on footprints, and there are three ways of not having one.** T4F.5
measured that a maximum sits about one analysing width from what excited it, so placing a
configuration by that cell would put it in the wrong box at a boundary. Of this record's 1,063
configurations, 281 sit wholly inside a declared box; 364 straddle two of them, 397 reach out of
the one box they touch into undeclared ground, and 21 are outside every box. The three failures
are counted apart because they say different things about the *declaration* -- boxes drawn closer
together than the transform's own footprints reach, versus ground nobody declared.

**Adjacent boxes are not independent and this module will not pretend otherwise.** The gap
between every pair is published in cells and in kilometres; with no declared decorrelation length
the independence of the regions is `UNESTABLISHED` and `general` is refused, and with one
declared, held-out regions closer to the discovery region than it are named. On this record at a
declared 1,500 km, `B`, `D` and `E` are named and `C` is not.

**Physiography is declared, not derived**, with a source required for the same reason T4F.6
requires a citation. R14 asks for re-testing on similar *and* dissimilar ground, so `general` is
refused when no assessed region declares a physiography, and refused again when every one of them
declares the same class.

**Not claimed.** This has been exercised on a synthetic record only. Applying it to the acquired
ERA5 record waits on the same mining pass T4F.6's gate waits on, and no atmospheric region has
been compared with any other.

**T4F.8 Follow-up experiment proposals -- DONE.** Extend the existing
`_propose_numerical_followup` / `_propose_categorical_followup` pattern to propose experiments
that *test* a discovered precursor - the platform closing its own loop.

**Met.** `src/analysis_engine/spectral_proposals.py`, verified by
`src/tests/test_spectral_proposals.py` (50 tests, 8.07 s).

**The two existing proposers are optimisers, and that is the whole problem.** One proposes the
parameter range that made the metric better; the other fixes the winning category and re-runs.
No outcome of either would retract the finding that prompted it, so neither is a test, and a
procedure that only ever produces confirmations is closing a circle rather than a loop. Both now
carry that in their own docstrings.

**A proposal that cannot come back negative is refused.** Every re-test names the statistic, the
direction and the threshold that would retract the finding, and the threshold is checked against
the range the statistic can attain. A base rate of zero leaves a condition no experiment could
satisfy, because a Wilson upper bound is strictly positive at any number of trials, and the
proposal is refused rather than dressed up in words.

**The prediction is digested before the record is read**, over the design alone: the rule, the
ground, the prediction, the refutation, the required occurrences and the declared alpha,
correction, null and ensemble size. The status, the lead, the signatory and the date are excluded
because none of them was declared in advance -- so one design read through two different records
has one digest. As in T4F.6, the code will not sign.

**A re-test is proposed only on ground the finding was not made on**, only for a rule that
cleared its own null, and only where the identities were not fitted on the target. A rule that
did *not* clear its null gets the other kind: a power proposal, which carries no prediction and
no refutation and says so, because a tool that proposes follow-ups only for the things that
worked has publication bias built into it. That proposal is refused when the study already
carried the occurrences an effect of the declared size needs -- a negative from an adequately
powered study is a result, and asking for more data until it changes is chasing it.

**Power is computed from quantities the record can be read for without performing the test.**
The eligible-anchor count and the base rate are properties of the record and neither counts the
pair, so reading them does not touch the alignment under test. `required_occurrences` returns the
smallest number of occurrences that separates the predicted effect from the base rate in **both**
directions -- able to confirm and able to retract. Sweeping every pair of proportions to two
decimal places, 2,052 pairs have an `n` that could detect and not retract and 1,973 have one the
other way about, so neither condition subsumes the other. The design interval is taken at `p * n`
rather than at a whole number of occurrences, because rounding makes separability non-monotone in
`n`: 336 trials fail a separation that 335 passes.

On the T4F.7 record the re-test of `A`'s rule onto `B` needs 8 eligible antecedent occurrences
and `B` supplies 81, so it is `TESTABLE`; onto `C` with no record supplied it is `UNDERPOWERED`
and names the acquisition it needs. `D`'s negative needs 32 occurrences for a doubling of its base
rate and carried 24, so its power proposal stands.

**Mutation testing: 58 mutations, 57 killed and one argued equivalent.** The survivors found a
real defect -- a target that was *offered* and turned out to have no readable base rate was
quietly given the discovery region's, a borrowed measurement standing in for a refused one. That
is now its own case and is refused, for both kinds of proposal. The equivalent mutant is an early
return in `required_occurrences` whose absence changes nothing, since the search below reaches
the ceiling and returns the same answer; it is kept for legibility and documented as equivalent.

**Not claimed.** No proposal has been run. This produces designs, and every figure in one is
what the discovery implies rather than anything measured on the target ground. Whether the
platform executes its own proposals is Phase 4G's question, not this one's.

---

**T4F.9 The pilot: one honest end-to-end result -- SPECIFIED, NOT STARTED.** The instrument has
never produced a scientific claim about the atmosphere. Run the whole loop on a **declared**
design small enough to finish and large enough for the statistics -- climatology, anomalies,
tracking, constellations, identity, sequences, precursors, projection and the gate -- declared
as a pilot and explicitly **not** T4F.6's acceptance.

**Acceptance:** a gate verdict, PASS or FAIL or INVALID, with a receipt naming the frozen
catalogue digest, the declared design and every figure's provenance. **A FAIL or an INVALID is a
success for this task.** What is not acceptable is a verdict nobody can interpret.

**Blocked on:** T4E.8, for an identity worth adjudicating; and on two declarations that are not
code -- a frozen, reviewed reference catalogue, and a documented cyclogenesis event declared with
its source. The draft catalogue exists and matches the acquired record's region exactly; it needs
four envelopes read and either accepted or corrected. The code refuses to sign it, by design.

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
    an external protocol. T5.0 remains partial and T5.3c cannot select model semantics from
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
    artifact synchronisation; those require a real HPC contract.
*   **T5.2 Build `RegionalForecastDataset` -- PARTIAL (T5.2a-d and live acquisition accepted; real model use remains downstream).** Materialise and rechunk a provenance-carrying New
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
    an ambiguous "6 hourly forecast steps" from being silently interpreted as six-hourly
    data. Focused backend/UI contracts and the 1,385-module production build pass; browser render
    inspection was not run because no controllable browser was attached.
    **Live acceptance completed later by T4C.5m/T4C.5n:** the CDS path acquired the six-year,
    8,764-frame NZ crop; exact coordinate/value overlap passed against WeatherBench at both the
    opening and an interior window; queue time, transferred bytes, cache identity and returned
    schema are recorded; D43 is closed. T5.2 remains labelled partial only because the accepted
    dataset has not yet been consumed by the real laboratory training/evaluation work owned by
    T5.3-T5.4, not because acquisition evidence is missing.
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
    **Outstanding:** a real model's code/interface, weights, history semantics and declared
    optimiser/schedule have not been supplied or run. Multiple independently trained seeds,
    uncertainty, temporal dependence and correction-family inference belong to T5.4/T5.5.
    T5.3b proves a usable integration/evaluation contract, not the laboratory experiment or
    representation skill.
    **T5.3c Model-semantics adapters -- NOT STARTED:** after T5.0 freezes the actual protocol,
    select and test the required semantic adapter: final-frame autoregressive, full-history
    autoregressive, or direct multi-horizon. Each must declare history use, rollout and physical
    lead-time mapping in provenance. Do not implement all variants speculatively or coerce a real
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
    as inserting Haar/db2/SWT/DTCWT representations into an external regional model.

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
       Freeze dates, NZ crop, variables and correction families before comparing with an external
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
    with an external model. T5.6a proves bytes/declarations, T5.6b proves schema/units/finiteness/crop
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

    **External interface case reviewed 2026-08-22:** the external
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
