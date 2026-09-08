# SpectralEarth: Visual Research Workbench & Discovery Engine

SpectralEarth is a research workbench under active scientific validation for multiscale
atmospheric analysis, regional spectral modelling, data-assimilation diagnostics and
hypothesis screening.

The platform joins a tensor-accelerated computational backend (**PyTorch**, **xarray**,
**SQLAlchemy**) to a React/Vite/Plotly dashboard and a Jupyter playground. It is not yet a
validated forecasting system. The decisive Phase 4C real-ERA5 gate **has** now run and returned
PASS on an 8,764-frame acquired record (T4C.5m, which also closed D43); Phase 5 still has
integration contracts and no completed learned forecasting comparison. Current status, evidence
and known limitations live in `roadmap.md`, `VERIFICATION.md` and `architecture.md`
respectively, and the section immediately below says which of those is authoritative for what.

---

## Where the programme actually is, and which document says so

This README is an orientation document. It is **not** the status of record and must not be cited
as one. Four documents carry the tracked state, each with a different job:

| Document | What it is authoritative for | Machine-checked |
|---|---|---|
| `architecture.md` | What exists in the code today: modules, HTTP routes, the test inventory, and Section 7's full defect ledger. | yes |
| `roadmap.md` | The **atmospheric** programme: Section 1's honest status table, the standing rules R1-R16, and every task with its evidence block. | yes |
| `roadmap_cross_domain.md` | The **cross-domain** programme on `ed-dev`: rules R17 onward, phases G0-G18. | partly |
| `VERIFICATION.md` | Captured output. Every number claimed elsewhere should be findable here as a run. | yes |
| `README.md` (this file) | Installation, layout and an orientation summary. | partly |

"Machine-checked" means `src/tests/test_documentation.py` parses the document and fails when it
contradicts the source or the other documents. That guard exists because these files had gone
stale before while nothing failed; it is the reason the status table cannot quietly drift.

**Two roadmaps, one repository.** `roadmap.md` is the atmospheric line, frozen for `master` at
`freeze-t4c.5h-preregistration` and still advancing on `ed-dev`. `roadmap_cross_domain.md` is a
fork of that line, not a successor, and its results may **not** be cited as SpectralEarth
atmospheric evidence. Both are live on `ed-dev` and work alternates between them; neither
supersedes the other.

**Current frontier (`ed-dev`).**

* Atmospheric line: **Phase 4E is DONE**. T4E.1-3 build, sign and approximately cluster
  constellations; T4E.4 now applies a configurable minimum support over distinct constellation
  identities, reports every candidate's support, prunes the below-threshold tail early, and
  refuses candidate/time budget overruns without returning a partial sweep. Approximate matching
  uses a declared attributed-graph metric, a tolerance measured from same-configuration
  replicates, and deterministic complete-link clusters whose centroid and radius are readable.
  **Phase 4F has opened: T4F.1 is DONE.** The timed event substrate reads a clustered
  catalogue against an explicit grid of the frames that were *searched*, refuses an
  unnamed clock, refuses an occurrence at a frame nothing looked at, and marks a span
  crossing an unread instant rather than reporting it as a longer gap. Events sharing a
  frame are published unordered, because a succession taken from list position would be
  fabricated. It found and fixed D93. **T4F.2 is DONE**: the chain the substrate was built
  to express is now counted, against a declared transition window whose minimum lag is
  strictly positive so simultaneity can never become a step. Confidence admits only
  antecedents whose window was wholly searched, so an occurrence the record ended before is
  censored rather than counted as unfollowed; a repeated gap is reported with the number of
  interval values the record was long enough to hold, and is not called a period. **T4F.3 is
  DONE, and it is where this phase draws its first null.** A confidence is referenced to a
  base rate measured as a window probability over the searched positions, to a surrogate
  ensemble that rotates the antecedent on the lattice so its own bursting survives and only
  the alignment under test is destroyed, and to a correction paid on the whole declared
  family, with a data-chosen lag tested against the distribution of the maximum. The
  anti-conservative null is provided and its cost measured: on one unchanged record the same
  lift of 5.25 is not distinguished from the shifting null at p = 0.11 and is called a
  precursor by the scattering null at p = 0.01. A design that could not have rejected
  anything is refused before any counting happens. Nothing is called a cause (R7).
* **T4F.4 is DONE, and it makes that table askable in both directions without making either
  direction a second test.** Top-down (what finer configurations preceded this structure?) and
  bottom-up (what coarser structure followed this configuration?) are one selection over the
  rows T4F.3 already corrected: the direction fixes which role the queried pattern plays and
  which side of a measured scale ordering its counterpart must sit on. Nothing is recomputed,
  the declared family is published beside every answer, and the tempting shortcut -- re-running
  the inference for the one pattern being asked about -- is priced rather than warned about:
  the same rule's q-value falls from 0.0417 to 0.0050 when the family is narrowed after the
  fact. Coarse and fine are read off the catalogue's own member scales as an interval order, so
  patterns whose scale ranges overlap or merely touch are not orderable and are withheld and
  counted rather than sorted, and under the scale-invariant mode the ordering is refused
  outright because that mode makes every pattern's scale statistic exactly 1.0. Every ranking
  key is available with its rank on every entry: ranked by how often it preceded the target the
  leading answer is a pattern the null did not distinguish, and ranked by the corrected p-value
  it is the planted precursor, so the choice of ranking decides the answer.
* Phase 4 frontier: **T4F.5 is DONE, and it puts a rule back on the map without claiming the
  map knows more than it does.** A pattern projects to a *footprint* -- the parent cells inside
  the transform's own filter support, the same number R13 cuts the contaminated margin with --
  and never to a pixel, because a detail coefficient peaks on a structure's flank rather than at
  its centre. On the planted vortex that flank sits 1.08 structure widths out at level 4 and 1.26
  at level 5, and **the peak cell is not the planted cell in a single one of twenty-four
  frames**; the footprint recovers it, and only while the level that found the structure can
  still reach back to it, so level 4 holds the planted cell to frame 9 and loses it in all
  fifteen frames after while level 5 holds it throughout. Evidence should be read at the level
  that resolves the thing. Two members' overlap is not a location either: one level's LH and HL
  bands contain the planted cell in 11 of 11 occurrences, and one orientation's two levels
  exclude it in 11 of 11 because both flanks point the same way. Each grid says only what it can
  -- degrees, or metres named as a distance from the crop's own origin, or cells and nothing else
  -- an undeclared level stays a number rather than becoming hectopascals, and a frame is dated
  only where the record carries a calendar. The historical instances come from the same
  per-anchor decision the support was counted with, and every total is reconciled against the
  rule's published figures before anything is shown.
* **T4F.6 is PARTIAL: R10's physical gate is built, it discriminates, and it has not been run.**
  A mined pattern is measured off the footprints T4F.5 drew, turned into kilometres and hours
  through the record's own metric and calendar, and labelled `recognised`, `unrecognised` or --
  the third label, which is the one that keeps the other two honest -- `unassessable`, because
  an entry needing a field the record does not carry has not been checked and found absent. All
  three verdicts are reached on the same real patterns: PASS against a catalogue whose envelopes
  fit, FAIL against one whose envelopes do not, with R10's standing presumption that the
  pipeline is broken written into the receipt, and INVALID against a catalogue so wide that
  nothing could have fallen outside it, which licenses nothing and is not a FAIL. The ranking
  key decides the verdict -- with a top-N of one the same report passes ranked by corrected
  p-value and fails ranked by support -- so it is hashed into a declaration along with the top-N, the catalogue's digest
  and a required naming of the documented event and its source. A catalogue is a draft until a
  maintainer signs it, and the gate refuses to adjudicate under a draft including the one this
  repository ships, whose envelopes are a first reading rather than values quoted from any
  paper. What bounds the whole exercise is a measurement of the record: on the acquired ERA5
  crop a cell is 27.80 km north-south and between 13.90 and 26.12 km east-west, so sixteen cells
  is anywhere from 222 to 445 km, and an envelope narrower than that cannot exclude anything.
  **The acceptance has not been run** -- it needs a maintainer-frozen catalogue, a declared
  documented cyclogenesis event, and a real mining pass over the 8,764-frame record that has
  never been performed -- so **Phase 4G is still gated**, and T4F.5's second acceptance clause
  waits with it.
* **T4F.7 is DONE: a rule now has to say where it holds and where it does not.** R14's point is
  that a pattern found in one place is a fact about that place until it is re-tested somewhere
  else -- "we found a thing about the Alps" and "we found a thing about the atmosphere" are both
  valuable and they are not the same claim. Five declared boxes of one record: the rule holds in
  two held-out regions, does not hold in a third that carries both patterns in the wrong order,
  and is **not assessable** in a fourth that carries nothing at all. That third label is the
  point of the task: a region where the pattern never occurred did not fail the test, it never
  took it, and reporting it as failure would turn missing data into evidence of locality. A
  held-out region may supply occurrences but may not help define what a pattern is, and checking
  that turned up something worth knowing -- the signature this platform compares configurations
  with is invariant to rotation by construction, so it cannot tell a zonal structure from a
  meridional one at all. How far apart the boxes are is published rather than assumed, and
  physiography is something a maintainer declares with a source, not something a temperature
  field can be asked. Exercised on a synthetic record only.
* **T4F.8 is DONE: the platform now proposes experiments that could tell it it was wrong.** It
  already proposed follow-ups, and both of them were optimisers -- propose the parameter range
  that made the metric better, fix the category that won and re-run. Neither could produce a
  result that would retract the finding that prompted it, which makes them useful and makes them
  not tests. The new kind is refutable: a re-test on ground the finding was **not** made on,
  carrying a prediction digested before the target record is read and a named condition that
  would retract it -- and refused outright when that condition turns out to be one no possible
  outcome could satisfy, which is what a zero base rate leaves you with. A rule that *failed* its
  null gets a proposal too, because a tool that only follows up its successes has publication
  bias built into it; that proposal carries no prediction, can confirm nothing, and is refused
  when the study was already large enough to have found the effect -- asking for more data until
  a negative goes away is chasing it. How many occurrences a design needs is computed from
  quantities the record can be read for without ever counting the pair under test, and it has to
  be enough to confirm *and* enough to retract. No proposal has been run: this produces designs.
  **Phase 4G is next** on the code line; the next scientific step is still the T4F.6 gate run,
  which needs a maintainer before it needs a machine.
* Cross-domain line: **TG17.14 is DONE and `live_sources` reads `PASS`.** On 2026-09-04 the
  authorised bounded run reached ERA5 through CDS, Argo GDAC and MAST SPOC, each demonstrating
  real network use, while the bespoke order-book family demonstrated a content-addressed local
  record and **no** network use. First contact with real archives found two defects, D94 and D95,
  neither findable offline. **Five of seven release gates now clear and the verdict is still
  `NOT_RELEASEABLE`**, blocked by `scale_shape_calibration`'s declared scientific limit — four
  archives in four domains buy nothing past a refusal. **The scale/shape bound is next, and it is
  a decision rather than a task.**
* Cross-domain line: Phase G18, the instrument the science is read through. TG17.9 and TG17.10
  are DONE, the latter with the **release withheld**; TG18.1 (shared instrument foundation) closed
  on 2026-09-03 after six slices; TG18.2 (scientific visualization workspace) closed the same day
  after five slices; TG18.3 closes the global navigation-only
  `Acquire -> Inspect -> Design -> Run -> Compare -> Admit -> Report` journey without replacing
  Composer's server-owned path or claim ladder. TG18.4 now closes rendered keyboard, focus,
  zoom-equivalent reflow, contrast, reduced-motion and non-colour acceptance without claiming WCAG
  certification. **TG18.5 (the UI qualification gate) is DONE, and with it G18**, scoped by TG17.10's
  withheld release rather than by presentation: it must supply the rendered evidence the
  `browser_no_glue` and `scientist_actions` gates refuse to award themselves. Its first slice
  qualifies served-workspace reachability; its second walks one representative path through each of
  the four product modes at 1440 and 1920 CSS pixels, capturing a named artefact at the state each
  reaches, and asserts that every mode signature appears in exactly one of the four -- the property
  that keeps a redesign from making one mode ambiguous while improving another. That measurement
  also found an intermittent failure in an older spec, whose heading locator matched both the
  panel's own title and the shell's screen-reader-only workspace heading. Its third measures the two numbers
  `scientist_actions` refused to invent -- **14** visible actions from a clean browser to a
  completed run of the frozen plan, and **3** to reach the preflight refusal with **4** more to
  clear it -- asserting the counts while recording wall-clock as context nothing asserts. Its
  fourth supplies the evidence channel: a Playwright reporter records what a run did and binds it
  to the source of every spec plus the completeness of the run, and a Python reader decides. A
  green single-spec run is refused as partial, a weakened spec returns the gate to `NOT_RUN`, and a
  run that failed reads `FAIL` rather than merely unrun. **`browser_no_glue` now reads `PASS`** on
  a measured 136 of 136. The close-out found the gap that mattered most: `roadmap.md`
  §10.2 admits no completion claim without recorded output in `VERIFICATION.md`, and seven
  phases had been marked done while that file stayed silent. The entries are backfilled and
  two guards now hold both directions. **G18 did not clear condition 19 by itself**; TG17.13 later
  supplied the source-edit audit and recorded acceptance run, so `synthetic_fifth_adapter` now
  reads `PASS`. **TG17.11 (the scale/shape calibration) is
  done**, all five slices: D91 fixed, the statistic built with its scale invariance measured at
  1.1e-16 rather than declared, four frozen fixtures, a calibration in which the planted case
  recovers 105 of 105 members in 20 of 20 realisations while both safeguards reject nothing in
  2,100 member tests, and the gate moved off `NOT_IMPLEMENTED` to **`REFUSED`**. Its result is
  that the method is sound and G17's declared families cannot reach it: the null needs an
  inventory of at least **105 declared pairings** before any member can reject at alpha 0.05,
  while the declared draw refuses inventories above 8, so no size satisfies both. Scale/shape
  mode is therefore not qualifiable on the declared families, which is recorded as the finding
  rather than engineered away. **TG17.12 is done**: the calendar calibration, which has always
  run and passed in the suite, now reaches its release gate through a recording bound to the
  declared contract and to the source of the modules that decide what it measures, so a relaxed
  expectation or a changed statistic returns the gate to `NOT_RUN` instead of leaving a stale
  pass. It recorded 6 of 6 rejections on the planted calendar event and 0 on each of the three
  false-alignment fixtures, so `calendar_calibration` reads **`PASS`** — the first of the seven
  gates to clear, with the verdict unmoved at `NOT_RELEASEABLE`. The two modes are opposites
  here: calendar buys its p-value floor with replications and has no enumeration ceiling
  (999 against 293 required, and the declared plan 200 against 166), while scale/shape buys its
  floor with domains and cannot.
* What TG18.2 has delivered, all of it presentation only: every figure carries a text and table
  equivalent of what it draws, a declared statement of whether two panels may share a colour
  scale, the domain a fitted claim was taken over with the uncertainty attached to it, and
  presentation-only resizable panes for the declared gridded comparisons, and self-contained
  vector publication sheets that preserve the figure's reading contract. Under
  G18 none of this may recompute, summarize, promote or reinterpret a scientific value, so each
  addition transcribes what the analysis layer produced or states that it produced nothing.
* Last measured full backend run: **4441 passed, 4 skipped, 1 xfailed**, exit 0 (2026-09-08, 1:04:59; the same day's earlier run of 4348 took 0:42:20 and the difference in duration is unexplained).
* Last measured browser suite: **136/136** in Chromium from a cleaned `.e2e-state` (2026-09-04).
  It is inventoried in `architecture.md` section 7.4a; it is the only check here that proves a
  page renders.
* Open defects: **D84, D85, D96, D97, D98, D99 and D100**; D18 partial. D43 is closed. **`PLAN.md` orders
  what remains.** **D97 is the one that gates everything**: the tolerance deciding what a
  *pattern* is is calibrated from repeated measurements of one physical configuration, and a
  real atmospheric record contains none -- the distance between two observations of one
  tracked configuration grows monotonically with the gap between them, so what comes back is
  physical evolution rather than a noise floor. Measured on three slices spanning the acquired
  record, at the radius that splits 5% of same-configuration pairs, 60-69% of pairs from
  different configurations are admitted. T4E.6 added the missing complementary rate and found
  the published one to be identically zero at its own operating point; T4E.7 built a
  calibration that needs no replicates at all and recovers a planted identity on synthetic data
  without labels, and **on the acquired record it returns no radius**, which is why D97 stays
  open. **D98 is part of why it returns none**: the only null available for the question -- a
  phase-randomised surrogate put through the same pipeline -- produces a quarter of the
  record's signatures, so it destroys the features and not merely their recurrence, and in the
  close-pair tail the record's signature pairs are *further apart* than that null's.
  T4E.8's replicate census then found that the record **does** hold replicates under the
  strictest reading -- 6,838 labelled pairs -- and that they group only 73% of genuine repeats
  against 27% of unrelated pairs admitted. A null cannot beat labels, so D98 was never the
  binding constraint. **D99** is: a tracked feature changes wavelet band between adjacent
  frames in 60.8% of same-configuration pairs, and scale-specific signing puts that into the
  comparable vector. **D100** is a component weighted as a discriminator that scores a coin
  flip. Excluding both, a radius of 0.0972 groups 90% of stationary repeats while admitting
  4.5% of unrelated pairs -- the first identity radius on this record the programme can defend,
  and it rests on 124 pairs.
  **D96 is downstream of it**: because most pairs fall inside the radius, the tolerance graph
  is one connected component, which is what makes the clustering unusably slow. **D96 gates Phase 4G**: the
  identity step that turns signed configurations into patterns is complete-linkage clustering
  implemented directly, measured on real signatures from the acquired record at 5.5 s for 50
  points, 40 s for 100 and 347 s for 200 -- an exponent of about 3 -- against a training
  period that presents roughly 843,000 of them. It was never characterised, because every
  test in the repository clusters a few dozen points. The mining pass the T4F.6 gate
  adjudicates is therefore not merely unperformed; on this implementation it cannot be
  performed. D91, the scale/shape null
  deranging list positions rather than pairings, is fixed in TG17.11's first slice. Fixing
  it established two facts about the declared families themselves: four domains compared
  all-against-all admit exactly one distinguishable reassignment, and the three that admit
  the null after D83 admit none at all. Both are refused by name rather than answered, so a
  scale/shape calibration must be declared over an inventory of records rather than domains.

**What has not been done**, stated once here so it is not inferred from the feature list: no
learned forecast comparison has been run, no laboratory model or config has been supplied, FCN3
has not been executed, no cross-domain mining pass has produced a finding, and no claim has been
promoted from any of the above. Completion of a run is not a finding.

---

## 🗺️ Key Scientific Pillars

### 1. Multiscale Transforms (FFT, DCT, DWT, SWT, DTCWT, Hybrid)
SpectralEarth ships tested 2D FFT, DCT-II/III, decimated Haar DWT, undecimated SWT,
a real Kingsbury q-shift DTCWT, and a hybrid FFT+DWT representation.

> **Honest status:** `src/transform_engine/dtcwt.py` is the real DTCWT implementation,
> cross-checked against two independent reference implementations. The original degenerate
> `apply_dtcwt2d` remains only as a regression-test comparison arm and is not on a runtime
> path. The DTCWT is approximately shift-invariant; SWT is the exactly shift-invariant option.

### 2. Turbulence Spectral Slope ($\beta$) Fitting
To verify if models (including Neural Weather Operators) preserve physical consistency and kinetic energy cascades, the engine computes radial Power Spectral Density (PSD) and fits a power-law exponent ($\beta$) using log-log least-squares regression:
$$\ln(E(k)) = \ln(C) - \beta \ln(k)$$
It automatically classifies flow regimes into **Charney 2D Enstrophy Cascade ($\beta \approx 3.0$)**, **Kolmogorov 3D Mesoscale Cascade ($\beta \approx 5/3$)**, or flat noise spectrums.

### 3. Boundary-Condition Lab
Regional models are prone to boundary artifacts. The lab applies Circular, Reflective, and Replicate padding, alongside Tukey (cosine-tapered), Hann, and Hamming tapering windows to analyze spatial gradient decay and quantify **Spectral Leakage** in Fourier space.

### 4. Dynamic Provenance Lineage DAG
Captures run seeds, execution policy, resolved data source, artefact references and lineage
nodes/edges. This substantially improves reproducibility, but replaying a run from lineage
alone is still outstanding and is not claimed.

### 5. Automated Hypothesis Discovery
The engine screens run tables for numerical and categorical associations and can propose
follow-up experiment configurations. Associations are hypotheses, not proof or causation;
reported findings carry multiple-comparison information and statistical caveats.

### 6. Relationship to regional AI weather forecasting

The project is scientifically aligned with comparing Fourier, cosine and wavelet
representations on limited-area weather domains. Today it can inspect their boundary
behaviour, localisation, scale/orientation structure and representation diagnostics on real
ERA5 crops and construct leakage-safe PyTorch forecast datasets from a materialised crop. It
**cannot yet reproduce a controlled learned-forecast comparison**: no matched spectral neural
network, actual laboratory-model integration, training loop or forecast-skill evaluation across
representations is implemented. That must not be inferred from the current data/transform tools.

The laboratory hand-off is now mechanically strict. `src.forecasting.protocol` accepts a
versioned motivating-experiment protocol only when domain coordinates/shape, data cadence,
histories and physical leads, transform semantics, train-only normalisation, exact splits,
optimiser/training budget, rollout semantics and parameter counts are all explicit. It also
requires a reference and SHA-256 for the source laboratory config. Canonical JSON has a stable
content hash and persisted records are integrity checked. This contract is ready, but no real
laboratory config has yet been supplied: the motivating experiment is therefore **NOT FROZEN**.

`src.forecasting.binding` then checks that execution matches that declaration. It recomputes
coordinate and train-statistics hashes, checks exact UTC splits and data/transform/model/training
semantics, and creates one dataset/protocol/checkpoint identity. Passing that binding to the
evaluator upgrades the record to schema v3 and refuses cross-run substitution or samples outside
the declared held-out split. This is ready for a real manifest, but the current acceptance
fixture is synthetic; there is deliberately no UI "ready" badge claiming a real experiment yet.

FourCastNet 3 is the first partially implemented external global-judge track under T5.6. Its
published 72-variable, six-hour global ensemble outputs contain the same five 850-hPa fields as
the NZ study, making common regional evaluation and independent multiscale error analysis useful.
`src.forecasting.external_fcn3` now implements a dependency-free, versioned request and sealed
result identity: it requires the global grid, all 72 ordered inputs, exact source/model/software
hashes, UTC initializations, stochastic member seeds, six-hour rollout, global-then-crop policy
and explicit worker hardware; returned NetCDF/Zarr bytes and resource measurements are bound to
the request. `src.forecasting.external_cube` authenticates those bytes before opening them,
requires exact time/member/lead/grid/variable axes and SI units, scans every value in bounded
storage chunks, and creates only an exact, explicitly declared regional view. **FCN3 itself is
not integrated or run.** The isolated Earth2Studio worker, checkpoint and real forecast artifact
still do not exist. The main application remains usable without NVIDIA hardware, Earth2Studio,
model weights or network access. An NZ crop will never be passed directly to FCN3, and NVIDIA's
spectral-fidelity claims will be tested rather than repeated as platform findings.

`src.forecasting.evaluation_run` now provides the offline execution path once those artifacts
exist. A versioned config freezes bounds, held-out dates/role and memory ceilings; the runner
authenticates and crops the forecast, opens only an existing local ERA5 cache, matches exact
valid times, evaluates against persistence and atomically writes one no-overwrite JSON result
and provenance receipt. Receipt reload checks nested hashes and cross-section lineage. Current
acceptance uses synthetic Zarr fixtures only, so the UI still shows no FCN3 skill result.

`src.forecasting.evaluation_job` makes that path portable without changing its scientific
identity. A job contains the request/result/crop/evaluation contracts; a separate bindings file
contains machine-local forecast, cache and receipt paths and is pinned to the job hash. Relative
paths resolve from the bindings file, so the same job can be copied between Windows, a laptop
and a shared HPC filesystem while only the bindings change. The command-line workflow is:

```text
python -m src.forecasting.evaluation_job create --forecast-request RUN.json --forecast-result RESULT.json --era5-crop CROP.json --evaluation-config EVAL.json --output JOB.json
python -m src.forecasting.evaluation_job bind --job JOB.json --forecast-artifact FORECAST.zarr --era5-cache-dir ERA5_CACHE --receipt RECEIPT.json --output BINDINGS.json
python -m src.forecasting.evaluation_job preflight --job JOB.json --bindings BINDINGS.json
python -m src.forecasting.evaluation_job run --job JOB.json --bindings BINDINGS.json
```

`preflight` authenticates the saved forecast and opens the existing local ERA5 cache but writes
nothing. `run` performs no download, model inference or scheduler submission; it executes the
accepted evaluator and verifies the resulting receipt against the job. These commands are
tested with synthetic stores only. They do not mean FCN3, real ERA5 verification or HPC has run.

The **Forecast Evaluation** tab is the receipt-backed presentation seam. Import the JSON written
by the final command; the server rechecks its outer/nested hashes and cross-lineage identities,
requires a declared accepted ERA5 catalogue source, stores it under its content hash, and then
shows matched forecast/persistence metrics, CRPS, spread/skill, rank diagnostics, exact scope,
provenance and claim boundaries. It accepts uploads only—never a browser-supplied server path.
With no admitted real-source receipt, the tab contains an explicit empty state and no result
visuals. A verified receipt still says **scientific skill not established**: sampling uncertainty,
dependence, significance and generalisation remain separate work.

The first Phase 5 target is deliberately practical: an importable PyTorch path for the exact
regional workflow used by the motivating research -- batches shaped `(B, C, H, W)`, aligned
850-hPa `t/q/u/v/z` inputs and targets, strict temporal splits with an embargo, differentiable
forward/inverse representations, and a small evaluation harness around an existing lab model.
The training-native path in `src/transform_engine/training.py` now includes raw, FFT, DCT,
multilevel decimated Haar/db2, undecimated SWT and DTCWT modules over `(B,C,H,W)`. They reconstruct differentiably and expose immutable
synthesis context for a model prediction. FFT uses explicit real/imaginary channel packing and
DCT matrices are cached module buffers; Haar/db2 use a PyWavelets-compatible periodisation
phase and a model-ready Mallat coefficient plane. Haar/db2 default to a cached four-band
`conv2d`/`conv_transpose2d` kernel, while `implementation="reference"` retains the clear
per-tap oracle. SWT packs final LL plus three parent-grid detail bands per level, reports its
`1+3L` coefficient expansion and boundary-valid interior, and automatically uses the measured
faster FFT path on CPU or convolution on CUDA/ROCm/MPS. The Spectral Transforms tab reads the
training-readiness contract from the backend. DTCWT uses an exact four-real-plane atlas, reports
native and parent-grid edge margins, and is shown as training accepted only because batch,
inverse, autograd, analytical-oracle and CPU/RTX parity tests pass. Its analytical view displays
native complex magnitudes with one within-level colour scale and a marked valid inset; it does
not interpolate scales or imply atlas adjacency is physical.

`src/data_layer/regional_forecast.py` now supplies the dataset half of that bridge. It resolves
canonical `t/q/u/v/z` aliases at 850 hPa, verifies coordinate alignment, applies the accepted
`split_temporal` embargo before constructing histories/targets, and fits one population
mean/standard deviation per variable using training frames only. Each ordinary PyTorch dataset
item contains `(history,C,H,W)` inputs, `(lead,C,H,W)` targets, timestamps and frame indices;
the shared provenance fingerprints the source manifest, crop, grid, time axis, split and
normalisation artifact. Cached preparation reads metadata eagerly but never the full crop:
float64 train-only moments are merged from bounded frame blocks, while each DataLoader process
opens its own local Zarr handle on first use. Preparation refuses an on-disk time chunk larger
than `statistics_chunk_frames`, because lazy indexing cannot undo an unbounded Zarr chunk;
materialise forecast caches with a matching explicit `time_chunk`. The same
`prepare_cached_regional_forecast(..., cache_dir=...)` call works against a laptop folder or an
HPC shared cache and returns CPU tensors for the training loop to place on CUDA, ROCm or MPS.

```python
from torch.utils.data import DataLoader
from src.data_layer.regional_forecast import (
    RegionalForecastConfig, prepare_cached_regional_forecast)
from src.data_layer.zarr_source import CropSpec

config = RegionalForecastConfig(
    level_hpa=850, history_frames=2, lead_frames=(1, 2, 4), embargo_frames=4,
    expected_cadence_hours=6,
    # Optional reproducible experiment cutoffs: (validation_start, test_start).
    # calendar_boundaries=("2018-01-01T00:00:00", "2019-01-01T00:00:00"),
    statistics_chunk_frames=32)
bundle = prepare_cached_regional_forecast(crop_spec, config, cache_dir=shared_cache)
train_loader = DataLoader(bundle.train, batch_size=8, shuffle=True, num_workers=2)
# Call bundle.close() when persistent loaders/workers are finished, or use `with bundle:`.
```

`crop_spec` and `shared_cache` are intentionally explicit; preparation never downloads or
silently falls back to simulated data. The ERA5 panel reports only manifest-level structural
eligibility until values are opened, and keeps preparation, train-only normalisation and the
independent-route overlap check as separate claims. **This is not the whole forecasting path:**
mixed-precision/compilation acceptance, non-NVIDIA hardware evidence and execution of the
actual laboratory model remain outstanding. The multi-year regional source and the independent
ERA5 overlap run are no longer outstanding: T4C.5m acquired the record and closed D43, and
T4C.5n audited a second window.

T5.3a adds the first forecasting seam without pretending the laboratory model has been
integrated. `PersistenceForecaster` is an exact zero-parameter physical-space baseline.
`ForecasterAdapter` wraps any one-step `torch.nn.Module` whose output preserves the encoded
tensor shape, dtype and device, then reconstructs each step and feeds it back autoregressively.
The importable smoke run exercises a real dataset batch through representation, model, inverse,
MSE and `backward()` while labelling its result as plumbing evidence, not forecast skill:

```python
from src.forecasting import (
    ForecasterAdapter, PersistenceForecaster, run_tiny_deterministic_step)
from src.transform_engine.training import make_representation

batch = next(iter(train_loader))
persistence = PersistenceForecaster().predict(batch["inputs"], lead_count=3)
evidence = run_tiny_deterministic_step(batch, representation="haar", seed=7103)

# Existing one-step coefficient model: encoded (B,C,H,W) -> same shape/dtype/device.
adapter = ForecasterAdapter(existing_model, make_representation("db2", levels=3))
prediction = adapter.predict(batch["inputs"], lead_count=batch["targets"].shape[1])
```

T5.3a uses only the final history frame and records that Markov assumption in provenance. The
actual external/laboratory architecture and its history semantics remain unintegrated; no
current evidence compares representation skill.

T5.3b adds the model-artifact and held-out evaluation contract around that seam. Laboratory code
still constructs its own model—the platform never imports executable code named by a manifest.
The loader verifies the model/representation configuration, model class, parameter schema and
checkpoint SHA-256 before a strict tensor-only `state_dict` load, then embeds that identity in
the forecaster and evaluation provenance:

```python
from src.forecasting import (
    evaluate_against_persistence, load_laboratory_forecaster,
    save_laboratory_artifact)

artifact = save_laboratory_artifact(
    "runs/model-a", existing_model,
    model_config=model_config,
    representation_config={"name": "db2", "levels": 3},
    training_provenance={
        "dataset": bundle.provenance, "seed": 7103,
        "optimizer": optimizer_config, "schedule": schedule_config,
    })
forecaster, artifact = load_laboratory_forecaster(
    "runs/model-a", existing_model, make_representation("db2", levels=3),
    expected_model_config=model_config,
    expected_representation_config={"name": "db2", "levels": 3})
result = evaluate_against_persistence(
    forecaster, test_loader, variables=config.variables,
    lead_frames=config.lead_frames, split="test",
    channel_std=bundle.normalisation.std,
    dataset_provenance=bundle.test.provenance)
```

Evaluation streams exactly matched model and persistence errors. It reports standardized RMSE,
MAE, bias and `1 - model_MSE / persistence_MSE` per lead and variable. Physical-unit errors
appear only when explicit training scales are supplied; cross-variable aggregation stays in
standardized space. A zero-error persistence denominator produces `null` skill, not infinity.
Every result says it is a single-checkpoint evaluation without seed uncertainty or significance,
so this contract makes no scientific-skill claim.

T5.2d makes the temporal meaning equally explicit. Calendar mode requires the declared
validation/test boundary timestamps to exist exactly in the returned time axis and applies the
same lag-sufficient embargo after each boundary; ratio mode remains available for exploratory
fixtures. `expected_cadence_hours`, when supplied, must match every interval exactly. Each sample
derives nanosecond lead durations and the whole-axis cadence from its timestamps, and evaluation
independently recomputes them, requires one regular frame-to-hour mapping across the complete
time axis and all samples/batches, and reports both
frame offsets and hours. Missing, irregular or inconsistent timing is refused instead of being
silently labelled as (for example) a six-hour forecast. The cached-crop UI remains metadata-only:
it displays cadence as `NOT VERIFIED` and physical lead labels as unavailable until actual
dataset preparation has opened and checked the time axis.

### Matched external-ensemble verification

`build_matched_truth(...)` lazily selects the exact 850-hPa ERA5/CDS analyses required by a
regional external forecast. It constructs both `(time, lead_time, lat, lon)` verifying truth and
the `(time, lat, lon)` observed initialization used by persistence. Initialization and valid
times, grid points, level, variables and SI units must match exactly; no nearest-time lookup,
regridding or unit conversion is performed. A held-out split interval must contain every
initialization and valid time. Runs wholly from 2020 onward may be labelled a fresh post-2019
holdout; overlap with FCN3's published 1980-2015 training, 2016-2017 test or 2018-2019
evaluation periods requires the explicit `published_partition_diagnostic` role.

The source manifest hash, its materialized content digest, requested samples, forecast/grid
identity, split, FCN3-period classification and exact selection semantics form a deterministic
builder receipt. Field arrays remain Dask-backed. This proves matching and lineage, not that the
ERA5 route is independent of model inputs or that a forecast is skilful.

`evaluate_regional_ensemble(...)` accepts the lazy NZ forecast produced by the canonical
FCN3 importer, verifying truth on exact `(time, lead_time, lat, lon)` coordinates, and the
corresponding observed initialization used for persistence. It refuses coordinate, variable,
unit, held-out split, lineage, completeness and finite-value mismatches; it does not interpolate
or silently delete missing cells.

For each physical variable and lead it reports every member's RMSE/MAE/bias, ensemble-mean
errors and MSE skill against exact persistence, empirical ensemble CRPS, population ensemble
spread/RMSE, and rank-bin diagnostics with deterministic fractional treatment of ties. Spatial
scores use cosine-latitude area weights. Unlike units are never pooled. Reads remain lazy and
spatially tiled; the recorded byte ceiling covers source arrays materialised per tile, while
working memory remains proportional to ensemble size times tile size.

The builder and evaluator are accepted on synthetic fixtures only. No real FCN3 output or
matched ERA5 truth has been scored, so the UI must not display an FCN3 skill or calibration
result yet.

### Accelerator installation and portability

The source is vendor-neutral at the PyTorch device boundary. NVIDIA CUDA and AMD ROCm both use
`torch.device("cuda")` in PyTorch; execution provenance records whether the compiled runtime is
`cuda` or `rocm` so an AMD run is not mislabeled as NVIDIA. Apple MPS remains part of the device
policy. The current Windows workstation is verified with an RTX 5050 Laptop GPU using:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade `
  --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130
```

An AMD system must install the official ROCm PyTorch build appropriate to its supported OS and
ROCm version; the representation code and accelerator test require no NVIDIA-specific calls.
ROCm hardware and Windows DirectML have **not** been tested in this repository, so portability
to those runtimes is designed but not claimed as verified.

### One workflow from laptop to HPC

Run the same local capability check on any machine:

```powershell
.\.venv\Scripts\python.exe -m src.core.doctor
.\.venv\Scripts\python.exe -m src.core.doctor --json
```

`SPECTRAL_PROFILE` controls placement without changing scientific code:

* `auto` (default) uses an available accelerator and otherwise falls back to CPU;
* `cpu` guarantees the dependency-free path, useful on an unsupported AMD laptop;
* `accelerator` requires CUDA/ROCm or MPS and refuses a silent CPU fallback; and
* `hpc` runs only inside a detected Slurm, PBS or LSF allocation, uses scheduler local rank for
  GPU placement, and refuses accidental execution on a login node.

For example, a laptop can use `auto` while `SPECTRAL_PROFILE=hpc` is set inside an
allocated cluster job. The application does not connect to, submit to or depend on the HPC
system; the same repository and commands run in both places. Remote submission, environment
modules/containers and artifact transfer remain site-specific work until the actual cluster
contract is known.

A priority experiment will test whether transform rankings change with domain size and valid
interior. This is recorded as a falsifiable support-length/boundary hypothesis, not as a claim
that the external poster is wrong. Full-domain, common-interior and boundary-band skill will be
reported separately so predictive boundary information is not silently confused with
padding-contaminated coefficient geometry.

---

## 📂 Codebase Directory Structure

```
├── .vscode/                 # Task and debug configurations (F5 to run)
├── data/                    # Directory for NetCDF4 (.nc) data dumps (GRIB not yet supported)
│   └── README.md            # Data file naming and ingestion guides
├── frontend/                # Vite React Dashboard Workspace
│   ├── src/
│   │   ├── components/      # Heatmap2D, LineChart, and SVG LineageGraph
│   │   ├── services/        # API client fetch routing endpoints
│   │   ├── types/           # TypeScript API interfaces matching Pydantic
│   │   └── App.tsx          # Main visual control center & state manager
│   ├── package.json         # Frontend package manifest
│   └── vite.config.ts       # Reverse-proxy configuration routing to FastAPI
├── src/                     # Computational Python Core
│   ├── physical_core/       # PhysicalField core coordinates & split guardrails
│   ├── transform_engine/    # Wavelets, Dual-Tree DTCWT, manual DCT, and FFT
│   ├── synthetic_generator/ # Affine perturbations and sinusoid/vortex/front generators
│   ├── boundary_lab/        # Windowing (Tukey/Hann) and spatial gradient decay
│   ├── data_layer/          # Simulated, local NetCDF and opt-in ERA5 Zarr sources
│   ├── analysis_engine/     # Power spectral density, coherence, and slope fittings
│   ├── experiment_engine/   # Dynamic Cartesian sweeps and lineage tracking
│   ├── hypothesis_engine/   # Pattern correlation mining
│   └── database/            # SQLAlchemy schemas (SQLite target by default)
├── requirements.txt         # Core Python packages (Optimized for Pydantic v1)
├── start_platform.bat       # Double-clickable Windows launch utility
├── start_platform.ps1       # Automated PowerShell dependency installer & runner
└── research_playground.ipynb# Jupyter exploratory notebook dashboard
```

---

## ⚙️ Quick Installation

### Prerequisites
Before setting up, ensure you have the following installed on your machine:
*   [Python 3.9+](https://www.python.org/downloads/)
*   [Node.js (LTS)](https://nodejs.org/)

### 🚀 Standard Automatic Startup (Recommended)
We have packaged a background launcher script to automate python environment checks, package installations, and start both processes together.

#### On Windows:
Simply double-click **`start_platform.bat`** in your file explorer. 

*Alternatively, run this command in PowerShell inside the workspace root:*
```powershell
.\start_platform.ps1
```
*When prompted `Do you want to check and install/update dependencies? (y/N)`, type **`y`** on your first startup.*

---

## 🛠️ Step-by-Step Manual Setup

If you prefer to run and manage the backend and frontend separately, follow these commands:

### 1. Backend Server Setup
From the workspace root:
```bash
# 1. Install dependencies (Optimized Pydantic v1 package rules)
pip install -r requirements.txt

# 2. Run the FastAPI development server with reload enabled
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
*The database file `spectral_earth.db` will be initialized automatically upon server startup.*

### 2. Frontend Dashboard Setup
In a new terminal:
```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install Node packages
npm install

# 3. Start the Vite hot-reloading development server
npm run dev
```
*The React dashboard will be running at [http://localhost:3000](http://localhost:3000) and will communicate with the API backend via a configured reverse-proxy.*

---

## 📓 Jupyter Notebook Integration (`research_playground.ipynb`)

For exploratory scripting, double-click **`research_playground.ipynb`** at the workspace root to launch the Jupyter research notebook.

It features a 5-step interactive widgets dashboard built with `ipywidgets` and `plotly`.
Those packages and `matplotlib` are declared in `requirements.txt`. The notebook supports
wavelet-level and boundary-taper experiments, spectral-slope fitting and Markdown report
export; it is an exploratory interface, not a substitute for a recorded pipeline run.

---

## 📂 Real Data Ingestion

Place your real-world meteorological cutouts inside the `data/` folder:
*   For ERA5 Reanalysis: Name your file `era5_reanalysis.nc`
*   For GFS Forecast: Name your file `gfs_forecast.nc`
*   For Toy Climate Model: Name your file `toy_climate_model.nc`

The `MeteorologicalDataAdapter` (`src/data_layer/adapters.py`) will automatically identify these files, parse them via `xarray`, and enable interactive coordinate cropping, pressure-level selections, and variable extractions inside the UI. If no files are present, the system defaults to high-fidelity, in-memory simulated weather fields for sandbox testing.

The source registry also supports opt-in, cloud-native ERA5 regional crops from the public
WeatherBench 2 Zarr archive on Google Cloud, with a rechunked local cache and recorded
provenance. Network reads are disabled unless `SPECTRALEARTH_ALLOW_NETWORK` is enabled; cached
crops remain available offline. Local-file cache entries are invalidated when files appear,
disappear or change, so an API restart is not required.

T5.2c adds an optional direct Copernicus CDS acquisition backend for the chunk-hostile regional
long-record case. Install it with `pip install -r requirements-cds.txt`. A
`CDSRegionalRequest` declares exact inclusive dates, UTC hours, north/west/south/east bounds,
canonical variables, pressure levels and grid spacing. It plans monthly requests, requires
explicit network consent, downloads atomically, resumes only hash-verified shards and converts
the result into the same local Zarr cache used by `RegionalForecastDataset`. Credentials remain
in the standard CDS client configuration and never enter provenance. The offline acquisition,
resume and conversion contracts pass. A **real CDS request has now run**: the eight-frame canary
for campaign v3, checked against an independently acquired WeatherBench window and agreeing to
within one step of the CDS route's own GRIB packing. **The multi-year NZ crop has since been acquired
and T4C.6 has run**: 8,764 frames in 72 monthly shards (338.905 MB, 5,853.7 s), the full-cache
overlap passing at 0.71875 of a GRIB packing step, and a **PASS** verdict with ten links
replicated in train and test. D43 is closed (T4C.5m).

Planning is network-free and prints the exact monthly CDS payloads before anything is queued:

```powershell
python -m src.data_layer.cds_source plan `
  --date-start 2020-01-01 --date-end 2020-12-31 --hours 0,6,12,18 `
  --lat -50 -20 --lon 150 180 --pressure-levels 850 --analysis-levels 3
```

The plan also prints a conservative storage estimate. Materialisation rechecks the actual
download and cache volumes before constructing the CDS client: it budgets remaining NetCDF
shards plus the complete temporary Zarr without assuming compression, combines both when they
share a drive, and refuses unless at least 5 GiB or 10% of the working requirement remains free
afterwards. The storage decision is recorded in the acquisition/cache manifest.

Those bounds and dates are an interface example, **not any real experiment specification**.
Materialisation uses the same scientific arguments plus explicit `--download-dir`, `--cache-dir`
and `--time-chunk`; it still refuses unless the network gate and standard CDS credentials are set.

The T4C.6 execution boundary is available in `src.analysis_engine.gate_run`. A versioned
`GateStudyPlan` freezes the crop, transform, climatology and complete statistical protocol;
`preflight_cached_gate` verifies an existing cache without network fallback, and
`run_cached_gate` streams train-only climatology and scale signatures into an atomic,
tamper-detecting receipt. The real-evidence role refuses anything except the direct CDS route
with a content-bound passed independent WeatherBench overlap receipt. Once a small matching
WeatherBench crop is materialised locally, publish that receipt without network fallback with:

```powershell
python -m src.data_layer.era5_overlap `
  --primary-manifest data/zarr_cache/<cds-key>.json `
  --independent-manifest data/zarr_cache/<weatherbench-key>.json `
  --variables t --level-hpa 850 --block-frames 8
```

The comparison requires exact timestamps/grid coordinates and compatible units, streams value
blocks, records per-variable tolerances/errors, and cannot overwrite or reuse evidence from a
different design. Synthetic runs are always labelled
`scientific_verdict: NOT_ESTABLISHED`. No atmospheric T4C.6 verdict has yet been produced.

Before live acquisition, freeze the complete two-stage design with
`src.analysis_engine.gate_campaign`. Its portable JSON binds a small CDS canary and matching
WeatherBench crop to the full request and gate plan; machine paths are supplied only at
preflight. The preflight is zero-network, budgets all campaign artifacts together, checks
`cdsapi`, standard credential-configuration presence and explicit network consent, and emits
the mandatory canary-first order:

```powershell
python -m src.analysis_engine.gate_campaign review `
  --campaign campaigns/t4c6_nz_era5_temperature_850_v2.json
python -m src.analysis_engine.gate_campaign review-supersession `
  --supersession campaigns/t4c6_nz_era5_temperature_850_v1_superseded_by_v2.json
python -m src.analysis_engine.gate_campaign preflight `
  --campaign campaigns/t4c6_nz_era5_temperature_850_v2.json `
  --supersession campaigns/t4c6_nz_era5_temperature_850_v1_superseded_by_v2.json `
  --full-download-dir data/cds/full --canary-download-dir data/cds/canary `
  --cache-dir data/zarr_cache --independent-cache-dir data/zarr_cache
```

The checked-in campaign is the preregistered T4C.6 primary analysis, not an example: six
complete years (2018--2023), 0.25-degree 20--60 S / 140--180 E, 850-hPa temperature, three
db2 SWT scales and 18--48-hour transfer-entropy lags. Its campaign SHA-256 is pinned by a test;
the review command emits exact calendar partitions, transform interiors, physical lag floors,
the 36-test BY family and surrogate resolution without touching the network. A ready preflight
still does not prove credentials, licence acceptance, remote service availability, ERA5
agreement or the hypothesis.

**`..._v1.json` is retired and must not be acquired.** It preregistered 2018--2022, and defect
D85 established that its 2,914-frame confirmatory partition holds 2,912 distinct admissible
circular shifts against the 3,005 its own 36-test BY family needs -- so it could not have
produced a PASS at any effect size. It is left frozen and unedited, because editing a
preregistration destroys the record of what was actually declared. `..._v1_superseded_by_v2.json`
is the retirement: it names both campaigns by content hash and states its reasons as *checks*
that are re-run against both, so a reason is admissible only where v1 fails it and v2 passes.
Passing `--supersession` to `preflight` makes the retirement bite where the transfer would
happen; `review` still reads v1 and still reports its defect, which is the point of keeping it.
The supersession's `deferred_to_run` block names what the re-freeze does **not** settle: D84
(the crop is carried through unchanged), D85's derived Theiler window, and D43.

`..._v3.json` and `..._v2_superseded_by_v3.json` continue the chain for D86. v2 froze the
record, the crop and the protocol but not the rule by which the two ERA5 routes are declared to
agree -- that lived in `era5_overlap.DEFAULT_ATOL` as 1e-4 K, so the one decision authorising a
multi-gigabyte transfer was the one no supersession governed. It was also unsatisfiable: ERA5
arrives through CDS packed per GRIB field, each frame on its own binary lattice, and 1e-4 K is
finer than the step the route can express. v3 freezes an `overlap_criterion` instead --
agreement within one step of the lattice the primary frame actually occupies -- and `preflight`
now refuses any campaign that declares none. The eight-frame canary was acquired and
passed it, and **the 8,764-frame record has since been acquired and the gate has run to a PASS**
under v3 (T4C.5m). D84 and D85 remain open; they govern whether an *absence* would have been
detectable, and a PASS does not route through them.

Both campaigns, the retirement and any published receipts are also readable in the browser under
**Review -> Atmospheric gate record**, served by seven `GET /api/v1/gate/...` routes. That
surface is read-only by construction: it serves no other HTTP verb, has no preflight or
acquisition route, and publishes the reasons for those refusals rather than leaving a missing
button to be read as an unfinished panel. It shows a retired design in full with its defect
visible, and reports an empty receipt store as an absence of *runs* rather than of findings.
The T4C.6 gate has since run and returned PASS.
Before reporting readiness it now derives the exact grid shape and chosen transform supports,
requires at least 128 valid parent-grid pixels at every scale, converts lat/lon degrees to
physical metres for the advection floor, refuses shorter lags, and reports temporal split,
annual-climatology coverage, multiplicity power and the upper-bound surrogate workload. CDS
bounds must align exactly to the requested grid and returned endpoints may not be server-snapped.

**Not supported:** GRIB (`.grib`/`.grib2`) ingestion via `cfgrib`, dateline-crossing CDS boxes
without splitting them into two requests, and NOAA HRRR/GFS object-store retrieval.

---

## Licence

SpectralEarth is proprietary software; it is not released under an open-source licence.
Copyright remains with Edward Jonathan Bentley. [The repository licence](LICENSE.md) grants
a designated Named Licensee a perpetual, worldwide and royalty-free right to use and modify the
platform for lawful personal, academic, scientific and commercial work, while reserving public
redistribution and sublicensing of the SpectralEarth core. Independently authored extensions
remain separate under the terms stated there. Third-party libraries, datasets, papers and model
artifacts retain their own licences and conditions.
