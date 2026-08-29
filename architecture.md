# SpectralEarth: Platform Architecture Document
*An Honest and Comprehensive Technical Assessment of the Current Implementation*

This document provides a detailed, truthful architectural blueprint of the **SpectralEarth Research Platform** as it exists today. It delineates the core design, the unified data spine, the key modules, database representations, API boundaries, and the architectural seams that decouple the platform's layers.

---

## 1. Architectural System Overview

SpectralEarth is designed as a hybrid research workbench that bridges high-performance atmospheric physics (implemented in **PyTorch** and **xarray**), a declarative automated experiment pipeline (orchestrated via **FastAPI** and **SQLAlchemy**), and a reactive visualization dashboard (built in **React/Vite**).

The system architecture is structured hierarchically:

```
                  +-----------------------------------------+
                  |           React Frontend                |
                  | (Reactive REST API Dashboard & Sandbox) |
                  +--------------------+--------------------+
                                       |
                                       | REST APIs (HTTP / JSON)
                                       v
                  +-----------------------------------------+
                  |          FastAPI REST Engine            |
                  |    (Request Validation & Routing)       |
                  +---------+--------------------+----------+
                            |                    |
         Database queries   |                    | Asynchronous Tasks
         & lineage inserts  v                    v
                  +---------+----+      +--------+----------+
                  |  SQLAlchemy  |      |   Declarative     |
                  |  Data Layer  |      | Experiment Engine |
                  |   (SQLite)   |      +--------+----------+
                  +---------+----+               |
                            ^                    | Resolves & Executes
                            |                    v
                            |           +--------+----------+
                            +-----------+   Physical Core   |
                               Lineage  |  (PhysicalField)  |
                                        +--------+----------+
                                                 |
         +--------------------+------------------+------------------+--------------------+
         |                    |                  |                  |                    |
         v                    v                  v                  v                    v
+--------+-----------+ +------+-------+ +--------+---------+ +-----+--------+ +--------+----------+
| Spectral Transform | |   Synthetic  | |    Boundary     | | Meteorological | |   Analysis &     |
|   Engine (FFT,     | |  Generator & | |  Condition Lab   | |  Data Adapter  | | Diagnostics (PSD,  |
|  DCT, DWT, DTCWT)  | | Perturbations| |(Windowing/Pads)  | | (Simulated DS) | |Decompositions, etc)|
+--------------------+ +--------------+ +------------------+ +----------------+ +--------------------+
```

---

### 1.1 Scope boundary: analysis workbench versus learned forecaster

An EGU poster supplied as external research context, *Spectral representations for regional
AI-based weather prediction* (Emily O'Riordan, Victoria University of Wellington), compares
Fourier, DCT, Haar, db2 and DTCWT representations inside a controlled lightweight neural
forecaster over the New Zealand ERA5 domain. It reports initial forecast-RMSE differences by
lead time and variable. Those poster results are **not evidence produced by this repository**
and are not reproduced here.

The present implementation is relevant to that research because it can:

* obtain provenance-carrying regional ERA5 crops;
* execute and validate the named transform families, including a real six-orientation DTCWT;
* measure boundary artefacts, reconstruction, scale activity and directional structure; and
* evaluate statistical claims against declared surrogate nulls and corrected test families.

It cannot yet answer the poster's forecasting question. The source tree now has an accepted
generic forecaster seam, persistence baseline, autoregressive integration fixture and held-out
evaluator, but it has no supplied laboratory model, real training run or representation-
controlled forecast-skill experiment. Those integration contracts are not evidence that the
poster result was reproduced. Until the actual protocol/model/data are bound and the frozen
experiment is run, statements about one representation improving learned regional forecast
skill must cite external results, not SpectralEarth.

### 1.2 Immediate research target: a training-native regional bridge *(partially implemented)*

The first downstream integration is the motivating laboratory workflow, not a large global
model: `(B, C, H, W) -> representation -> existing regional forecaster -> inverse -> loss`.
That makes the platform useful in the current experiment while keeping the interfaces general
enough for another regional model, variable set, pressure level or geographic domain.

This requires a second, deliberately narrow spine beside `PhysicalField`:

* `RepresentationModule`, a `torch.nn.Module`-compatible forward/inverse contract over
  batched tensors, with structured coefficient metadata and an explicit real/imaginary packing
  convention;
* `RegionalForecastDataset`, yielding aligned input/target tensors plus timestamps, grid,
  variables, levels, lead time, source fingerprint and split provenance; and
* `ForecasterAdapter`, which lets an existing laboratory model act as the downstream judge
  without making its architecture part of SpectralEarth.

`PhysicalField` remains the 2D, coordinate-aware analysis object. It must not be weakened into
an untyped container for arbitrary training batches. The two spines share filter definitions,
boundary conventions and provenance schemas; adapters move a selected batch item into the
analysis spine when diagnostics are required. This separation gives training code a normal
PyTorch interface without creating a second scientific definition of each transform.

**Current readiness boundary.** `src/transform_engine/training.py` now implements accepted
raw, FFT, DCT, Haar, db2, SWT and DTCWT modules over `(B,C,H,W)`, with immutable per-batch synthesis
context, exact-shape validation, differentiable inverse operations and cached matrices/filter
banks. FFT
packs real then imaginary coefficients on the channel axis rather than using discontinuous
magnitude/phase. DTCWT uses a lossless four-real-plane recursive atlas per input channel;
every real and imaginary component stays at its native dyadic resolution and arbitrary
model-produced atlases remain invertible. CPU/accelerator reconstruction, coefficient parity and backward
gradients are verified on an RTX 5050 Laptop GPU with PyTorch `2.13.0+cu130`. The accelerator
test is vendor-neutral: an AMD ROCm build appears through PyTorch's `cuda` device API, while
provenance distinguishes runtime `rocm` from `cuda`. ROCm, MPS and DirectML hardware are NOT
RUN; mixed precision and compilation remain unaccepted.

`src/data_layer/regional_forecast.py` now implements `RegionalForecastDataset` and a three-split
bundle over aligned 850-hPa `t/q/u/v/z`. The constructor resolves long/short ERA5 aliases,
refuses misaligned coordinates, missing/non-finite values and an embargo shorter than the
longest lead, then reuses the accepted `split_temporal` guardrail. It assigns frames to
train/validation/test (including returned embargo sequences) before it creates any history or
target index. Population mean/std vectors are fitted in float64 over training frames and grid
cells only, serialized as a hashed `NormalisationArtifact`, then reused unchanged by all splits.
Items are default-collatable dictionaries with `(history,C,H,W)` inputs, `(lead,C,H,W)` targets,
Unix-nanosecond timestamps, timestamp-derived lead durations and original frame indices. Ratio
splits remain available; calendar mode instead requires exact validation/test start timestamps
on the returned axis and applies the embargo after each declared boundary. An optional expected
cadence must match every observed interval. Each sample also carries the measured whole-axis
cadence. Evaluation recomputes duration from timestamps, requires all frame leads and the full
axis to share one regular physical cadence, and records both frames and
hours; missing or inconsistent timing is a hard refusal. Source content hash, materialised crop
manifest/chunking, canonical/resolved variables, 850-hPa selection, full time/grid fingerprints,
split bounds and normalisation lineage travel in the bundle provenance.

The cached path is bounded-memory as of T5.2b. `open_cached_lazy` retains the cache-only
refusal without calling `.load()`. Coordinate vectors are read first; float64 population
moments are then computed only over training indices in configurable frame blocks using Chan's
stable merge formula. `_LazyZarrValues` serialises no live xarray object and opens one local
store per process on first sample read; a PID guard also closes and reopens a handle inherited
through HPC `fork`. `RegionalForecastBundle.close()` and its context-manager form make parent
handle lifetime explicit. Lazy and eager statistics/tensors agree on the fixture, and a forced-
spawn two-worker DataLoader succeeds after a parent read. Training cells are checked for finite
values while statistics stream; validation/test cells are checked when accessed. Cached
preparation refuses a missing or oversized recorded on-disk time chunk and instructs the user
to rematerialise with `time_chunk <= statistics_chunk_frames`; this is what turns the memory
bound into an enforceable storage property rather than an assumption about lazy indexing.

The interface is deliberately placement-neutral: `prepare_cached_regional_forecast` takes a
content-addressed `CropSpec` and an explicit local or shared-HPC cache directory, permits no
network fallback, and returns CPU tensors for a normal PyTorch `DataLoader`; a training loop may
move each batch to CUDA, ROCm or MPS. `cross_check_era5_overlap` requires exact time/grid
coordinates and reports per-variable error under declared tolerances. The checker passes on the
independent local acceptance fixture, but an actual second ERA5 acquisition route has **NOT
RUN**, so source-value agreement is not claimed. D43 still blocks a viable multi-year crop and
the actual laboratory model has not been integrated. T5.3a now supplies the adapter contract,
an exact persistence baseline and deterministic tiny integration evidence described below.

### 1.3 Boundary-support hypothesis for the New Zealand comparison *(proposed study)*

For the implemented `near_sym_b` / `qshift_b` DTCWT, the measured parent-grid contaminated
margin per side is 9, 19, 45 and 97 pixels at levels 1--4. If -- and only if -- the motivating
experiment used these filters, comparable boundary handling, a domain near 120x80 and three or
four levels, its coarsest coefficients would have no strict two-dimensional valid interior.
The poster does not establish all of those configuration facts, so this is an audit hypothesis,
not a finding about its model.

The related hypothesis is that part of the db2-versus-Haar ranking could arise from support
length interacting with a limited domain. It will be tested with nested domains at fixed
resolution and dates, not inferred from filter length alone. The frozen design must include:

* the exact external model configuration, tensor shape, transform library/version, filters,
  levels, padding and coefficient packing;
* identical temporal splits, embargo, training schedule and train-only normalisation;
* a common central New Zealand evaluation window while the surrounding context grows;
* full-domain, common-valid-interior and distance-to-boundary skill, plus a boundary
  masking/zeroing ablation;
* independently seeded fits, temporal block uncertainty, effect sizes and a declared
  transform-by-variable-by-lead correction family; and
* a transform-by-domain-size interaction as the primary test, with parameter count, coefficient
  redundancy, measured compute and memory reported as possible confounders.

Boundary-dependent coefficients are not automatically useless to a predictor. The scientific
claim is narrower: a representation mechanism is supported only if its ranking survives the
declared controls, and either a surviving or disappearing db2 deficit is a reportable result.

### 1.4 FourCastNet 3 external-judge boundary (`src/forecasting/external_fcn3.py`, `external_cube.py`, `matched_truth.py`, `ensemble_evaluation.py`, `evaluation_run.py`, `evaluation_job.py`, `evaluation_report.py`, partial)

FourCastNet 3 (FCN3) is the first concrete T5.6 external target, not part of the motivating
regional model. The official July 2025 NGC model card declares a 710,867,670-parameter
probabilistic spherical neural operator accepting 72 variables on a global 721x1440,
0.25-degree grid at six-hour steps. Its outputs include the five 850-hPa variables selected for
the regional bridge: temperature, specific humidity, zonal/meridional wind and geopotential.
The accompanying paper reports probabilistic calibration, spectral fidelity and rollout
stability to subseasonal leads. These statements motivate independent tests; they are not
verified facts about a run in this repository. Sources reviewed 2026-08-22: the
[paper](https://arxiv.org/abs/2507.12144), [NGC model card](https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/fourcastnet3),
[Earth2Studio](https://github.com/NVIDIA/earth2studio), [Makani](https://github.com/NVIDIA/makani)
and [`torch-harmonics`](https://github.com/NVIDIA/torch-harmonics).

The scientific role is an external global ensemble judge. FCN3 must ingest its complete global
state through its own pinned channel/grid/normalisation contract; only physical outputs may then
be cropped to the same NZ window, dates, variables and lead times used by persistence and the
regional model. Supplying the NZ five-channel crop directly would be a hard refusal, not an
optimization: it changes both the input contract and the spherical/global operator. FCN3's
learned spherical Morlet-wavelet convolution kernels must not be described as the same
experimental treatment as the explicit Haar/db2/SWT/DTCWT representations around Adam's model.

The implemented offline boundary and proposed worker path are file-oriented and vendor-isolated:

```text
portable app -> hashed ExternalForecastRun request
             -> optional Earth2Studio/FCN3 worker on suitable allocated hardware
             -> hashed global NetCDF/Zarr ensemble forecast
             -> canonical forecast-cube import -> NZ crop -> common evaluation/analysis
```

The app must start and prepare requests without Earth2Studio, an NGC account, CUDA or the
checkpoint; once the canonical importer exists, it must likewise analyse saved outputs without
those dependencies. The worker records checkpoint/package/config/data hashes,
initial condition, member identity/noise process, six-hour leads, precision, device/runtime,
units, coordinates and resource measurements. Evaluation will require persistence and the same
truth samples, then add ensemble-member/mean errors, CRPS, spread-skill and rank diagnostics.
Multiscale analysis tests FCN3's published spectral claims rather than assuming them.

T5.6a now implements the first and last identity edges of that diagram. `ExternalForecastRun`
is an exact-key, versioned, canonically hashable declaration covering the reviewed FCN3 model
identity and parameter count; package, checkpoint, model-card and licence evidence; pinned
Earth2Studio/PyTorch/torch-harmonics environment; all 72 input channels in canonical order;
global grid and coordinate identity; source and normalisation hashes; timezone-bearing initial
conditions; six-hour steps; one unique seed per intrinsic stochastic ensemble member; output
variables/format/reference; global-then-crop policy; precision; and prepare-only or external
worker placement. Only Linux/NVIDIA CUDA is accepted as an executed-worker declaration in v1;
other hardware remains a truthful planning refusal rather than an optimistic fallback.

`ExternalForecastResult` can be sealed only for an external-worker request. It binds the request
hash to a deterministic SHA-256 of either a NetCDF file or every named/file-sized byte stream in
a Zarr tree, plus global shape, variables, initializations, cadence, steps, ensemble count,
precision, OS/accelerator/hardware, peak RAM/VRAM, wall time and worker-log hash. The manifest has
its own derived hash, refuses overwrite, reloads strictly, verifies against the originating
request and detects changed artifact bytes. This is byte/provenance acceptance only: it does not
open the forecast arrays or establish their schema, units, values, calibration, spectral fidelity
or skill.

T5.6b adds the array acceptance boundary in `external_cube.py`. A returned NetCDF4 or Zarr
artifact is content-authenticated against both manifests **before xarray opens it**. The
canonical cube uses Earth2Studio's public `time`, `ensemble`, `lead_time`, `lat` and `lon`
coordinate vocabulary and one named array per requested physical variable. It requires the exact
global 721x1440 grid and declared coordinate hash, UTC initializations, member seeds, six-hour
lead durations, requested variable set, floating dtypes and canonical SI units (`K`, `m s-1`,
`kg kg-1`, `m2 s-2`, `Pa` or `kg m-2`, as applicable). NetCDF CF numeric durations and Zarr
`timedelta64` leads normalize to the same exact nanosecond axis. A completed-write marker is
mandatory, and partial or semantically different output is refused.

The value gate is complete but bounded: Dask exposes the on-disk chunks, the importer rejects a
decompressed chunk above the declared byte budget, and then checks every value for finiteness one
chunk at a time. It never calls `.load()` on the global ensemble. `GeographicBounds` requires
explicit, inclusive, grid-aligned endpoints and the same longitude convention as the global run;
implicit rounding, interpolation, antimeridian wrapping and a guessed default "NZ box" are
refused. The resulting `RegionalForecastCube` stays lazy and carries request, result, artifact
and validation hashes plus the exact regional coordinate hash. This follows the official
Earth2Studio [coordinate conventions](https://nvidia.github.io/earth2studio/userguide/about/data.html)
and [chunked-output guidance](https://nvidia.github.io/earth2studio/main/userguide/components/io/)
reviewed 2026-08-22 while making the workbench's stricter semantics explicit.

T5.6c adds the portable matched-truth evaluator in `ensemble_evaluation.py`. Its verifying truth
has exact `time`, `lead_time`, `lat` and `lon` coordinates, while the initialization dataset has
the same `time`, `lat` and `lon` and supplies the physical persistence baseline. Both require a
completed-write marker, content-source SHA-256, explicit non-training split and the same canonical
SI units as the forecast. The regional forecast's request/result/artifact/validation hashes,
shape and coordinate hash are rechecked. Variables, timestamps, leads and grid coordinates must
match exactly; interpolation, unit conversion and missing-value deletion are refused.

Evaluation visits bounded spatial source tiles and leaves all xarray/Dask inputs lazy. For every
variable and lead it records cosine-latitude-area-weighted RMSE, MAE and bias for each seeded
member, the ensemble mean and persistence; `1 - ensemble_mean_MSE / persistence_MSE`; exact
empirical finite-ensemble CRPS; RMS population ensemble standard deviation; and the explicitly
defined uncorrected spread/RMSE ratio. A zero denominator becomes `null`. Rank bins use uniform
fractional allocation across all ranks admissible under an exact truth/member tie and report
both fractional cell counts and area-weighted frequencies. Rank shape is diagnostic only. No
cross-variable score combines unlike physical units. Receipts bind the three provenances,
coordinate identity, value-stream hashes, member/lead identities, weighting and source-tile byte
limit. Temporary working memory is O(members x tile cells), not represented as the source-read
byte limit.

T5.6d adds the lazy observation bridge in `matched_truth.py`. It accepts the canonical regional
ERA5/CDS xarray cube and its mandatory materialized-content manifest, plus a validated regional
forecast. It derives every valid timestamp as initialization plus the exact nanosecond lead,
requires initialization and all valid times to occur exactly once on the strictly increasing
source axis, and uses labelled integer indexing so the field arrays remain Dask-backed. The
850-hPa level, source alias, floating dtype, SI unit and regional latitude/longitude values must
match exactly. Nearest-time matching, regridding, level approximation and unit conversion are
hard refusals.

The caller declares a non-training split with inclusive start/end timestamps; both analyses and
verifying targets must remain inside it. `fresh_post_2019_holdout` is accepted only when every
sample is from 2020 onward. Any use of FCN3's published 1980-2015 training, 2016-2017 test or
2018-2019 evaluation periods must instead be labelled
`published_partition_diagnostic`; pre-1980 dates are refused by the v1 contract because their
relationship to the pinned model is undeclared. The receipt hashes the complete source manifest,
retains its materialized content digest, and binds forecast/grid identity, aliases, level,
variables, split, period classification, initializations, leads and valid-time selection. That
receipt establishes matching and provenance only; it does not prove source independence.

T5.6e composes those gates in `evaluation_run.py`. A versioned, hashable
`EvaluationRunConfig` freezes the exact regional bounds, held-out split and FCN3-period role,
850-hPa bridge and both forecast-chunk and evaluation-tile memory ceilings. The orchestrator
authenticates and completely scans the sealed global artifact, makes an exact grid-aligned crop,
opens an already materialized ERA5 `CropSpec` through the local-only lazy cache seam, constructs
exact matched truth, evaluates it against persistence, and closes both stores on success or
failure. It performs no acquisition and no model inference.

The successful result is one canonical JSON `EvaluationRunReceipt`: request/config, forecast
validation, crop lineage, complete ERA5 manifest, matched-truth receipt, metrics and value-stream
hashes are retained together. Saving uses a flushed same-directory temporary file and an atomic
no-overwrite publish; an existing result is refused before any expensive input opens. Reloading
checks the outer content hash, embedded request/config/evaluation hashes, and cross-section
forecast, validation, ERA5-source and builder identities. This is reproducible execution
evidence, not a digital signature and not evidence that the sources are independent or that the
forecast has scientific skill.

T5.6f adds the portable invocation boundary in `evaluation_job.py`. `EvaluationJob` embeds the
typed T5.6a request/result, exact ERA5 `CropSpec` and T5.6e config in one versioned canonical
record; its SHA-256 therefore changes with any scientific input or memory control. It contains
no machine path. `EvaluationPathBindings` separately binds forecast-artifact, ERA5-cache and
receipt paths to that job hash. Relative paths resolve from the bindings file, allowing the
same job bytes to move between a Windows laptop and a Linux/shared HPC filesystem without
silently changing the experiment.

The four-command CLI creates a job from existing validated manifests, creates machine-local
bindings, preflights them and runs the accepted orchestrator. Preflight verifies request/result
compatibility, authenticates the complete forecast artifact, opens the local-only lazy ERA5
cache, rechecks its crop identity and refuses an existing output; it neither writes a receipt
nor performs network access or inference. Run delegates to T5.6e, reloads the atomic receipt and
checks its config, request, result, artifact and crop identities against the portable job.
Job and binding files are themselves atomically published and never overwritten. This layer is
intentionally not a scheduler: Slurm/Celery submission remains an Executor concern in Phase 6.

T5.6g adds the evidence-to-interface boundary in `evaluation_report.py` and the Forecast
Evaluation tab. Receipt import is an uploaded JSON document, never an arbitrary server path.
The server applies the same outer, nested and cross-lineage verifier as the offline loader,
reduces the receipt to a path-free presentation contract, and stores the original verified JSON
under its receipt SHA-256. Every list/get read verifies it again; corrupted entries disappear
from listing rather than becoming partial reports. An integrity-valid fixture is still refused:
display eligibility additionally requires the truth manifest to name an exact URI from the
accepted ERA5 catalogue and to carry no synthetic/fixture source declaration. This authenticates
the receipt and forecast artifact and checks the declared truth route; it is **not** a third-party
digital signature over ERA5 values, so the UI says `DECLARED_OFFICIAL_ERA5`, not “source signed.”

The tenth UI module has a hard empty state: with no admitted receipt it renders no scores or
plots. With one, it presents unit-preserving variable/lead rows for ensemble mean, persistence,
MSE skill score, CRPS, spread and spread/skill; rank frequencies with their fractional-tie policy
and `diagnostic only` label; exact dates/domain/pressure/split/member/grid scope; content hashes;
holdout role; and both receipt claim boundaries. `scientific_skill` remains
`NOT_ESTABLISHED` regardless of a positive point score because the receipt contains no sampling
uncertainty, significance, dependence or generalisation result. New T5.6g receipts also embed
the already-hashed forecast request/result so model, checkpoint and runner identity can be shown;
older valid receipts truthfully report that only their digests were retained.

These gates prove artifact acceptance and deterministic matched-sample metric calculation. They
do **not** prove meteorological correctness, ensemble calibration, spectral fidelity,
generalisation or skill on real data. Sampling uncertainty and dependence-aware inference remain
future work; no FCN3 or ERA5 result has been evaluated in this repository.

Portability is deliberately asymmetric. The official model card lists Linux/NVIDIA Turing,
Ampere and Hopper; it recommends bf16 and reports A100/H100/L40S testing, but no minimum VRAM or
AMD support. Its 2.65-GB compressed package and 711M parameters do not establish that an 8-GB
RTX can execute it. RTX 5050, AMD GPU and CPU inference are all **NOT RUN**. HPC may enable the
worker, but FCN3 is never a dependency for the regional workflow or ordinary workbench use.
No FCN3 dependency, worker, checkpoint, global initial condition or real forecast artefact
currently exists in this repository. Only the dependency-free request/result/cube contracts,
synthetic matched truth and analytic ensemble-evaluation fixtures exist; no real-data evaluation
exists.

### 1.5 Ownership and extension boundary (`LICENSE.md`)

SpectralEarth is proprietary software owned by Edward Jonathan Bentley, not an open-source
integration project for a laboratory. The root licence gives Adam Frank Bentley a personal,
perpetual, worldwide, royalty-free right to use, modify and operate it for lawful personal,
academic, research and commercial work, including on institutional, cloud and HPC systems. It
does not transfer ownership of the core or permit its public redistribution, sale as a platform
or sublicensing. Narrow collaborator access is allowed only to support Adam's work and creates
no independent licence.

The legal boundary matches the technical one: an independently authored adapter or plugin that
does not reproduce a substantial part of the core can remain its author's work, subject to any
employer, university or funder rights. Incorporation into the core requires separate written
contribution terms. Third-party code, data, papers, services, model weights and other artifacts
remain governed by their own terms; the SpectralEarth licence cannot grant rights Edward does
not hold. This section records the repository's declared terms, not evidence that a lawyer has
reviewed them.

---

## 2. The Core Spine: `PhysicalField`

The primary data structures of atmospheric models (temperature, geopotential, wind velocities) are multidimensional grids. In SpectralEarth, the unified spine of the entire application is the **`PhysicalField`** class (`src/physical_core/field.py`). 

Instead of passing raw PyTorch tensors across analytical modules, every analytical engine,
generator, transform, and adapter operates on a `PhysicalField` object. The proposed
training-native bridge in Section 1.2 is the intentional exception: batches remain tensors and
cross into `PhysicalField` only for coordinate-aware diagnostics.

### Implementation Mechanics:
*   **Grid Representation:** Encapsulates a 2D spatial grid as a `torch.Tensor` (shape: `H, W`), coordinate mappings (e.g., latitude/longitude or local x/y), and metadata dictionary.
*   **Coordinate-Aware Scaling:** Provides `.scale_resolution(target_shape, mode)` to interpolate both the 2D grid field and its coordinate arrays concurrently using bilinear or linear interpolation. This ensures that spatial resizing preserves coordinate mappings.
*   **Leakage Guardrails:** Provides `.split_field()` to partition a region along the X-axis into Train/Val/Test subsets. Crucially, `.validate_split_guardrails(other)` compares coordinate boundaries and raises a `ValueError` if spatial overlap or data leakage is detected.

---

## 3. Truthful Module Assessment

Every module on the backend is implemented in active PyTorch or xarray code; there are no stub functions or hard-coded return values. Two mathematical claims previously made in this document did **not** hold on inspection and have been corrected below: the **DTCWT is not a true dual-tree transform** (§3.1) and the **hybrid inverse is not the algebraic inverse of its forward pass** (§3.1). Section 7 records the full verification status and defect list.

### 3.1 Spectral Transform Engine (`src/transform_engine/transforms.py`)
Provides forward and inverse spectral transforms, working as a core mathematical toolset:
*   **FFT (Fast Fourier Transform):** Computes 2D Fast Fourier Transform returning amplitude and phase matrices. Vectorized using `torch.fft.rfft2` and `torch.fft.irfft2`.
*   **DCT (Discrete Cosine Transform):** Computes 2D DCT Type II and Type III (inverse) manually using pre-computed cosine coefficient matrices. Highly efficient.
*   **DWT (Discrete Wavelet Transform):** Multi-level 2D DWT using Harr filters. Implemented using 2D convolutions with stride 2 and reflection padding to prevent edge artefacts.
*   **DTCWT (Dual-Tree Complex Wavelet Transform) — implemented for real in T3.5.6, in a
    new module `src/transform_engine/dtcwt.py`.** The original `apply_dtcwt2d` in this file
    was not a DTCWT at all: tree B's filters were `[cos(pi/4), sin(pi/4)]`, identical to the
    Haar low-pass, and `g_b = -g_a`, so all four "trees" were one filter bank up to a sign —
    no Hilbert pair, no analytic response, no shift invariance, no orientation. It
    reconstructed perfectly because four copies of one transform average back to the input,
    which is precisely why the round-trip test never saw it (defect D1, rule R8). Measured
    at **236.11%** subband-energy spread over an 0–8 px translation, *identical to a plain
    Haar DWT on the same field*. The old function is retained, unused by any runtime path,
    purely as the comparison arm of the head-to-head regression tests; a test asserts no
    runtime code calls it.
*   **Undecimated / Stationary Wavelet Transform (`src/transform_engine/stationary.py`, T3.5.7):** A trous 2D transform in which **every scale keeps the parent grid shape**, with exact perfect reconstruction (`mode="periodic"`) and **exact shift invariance**. Supports haar, db2 and db3 with full-double-precision vendored coefficients. Measured on a sharp front translated 0-7 px: subband energy spread **0.00%** and aligned envelope correlation **+1.0000**, against **153.50%** and **-0.11** for the decimated DWT on the identical field. Tight-frame constant 4.0 per 2D level (verified to 1e-11); coefficients agree with `pywt.swt2(..., norm=True)` times `2**level` to 1e-11.
    *   This is the transform the Phase 4 layer is built on: coefficients inherit the parent field's coordinates directly (making `CoefficientField` coordinate-clean and T4F.5 evidence projection exact), and shift invariance is what makes feature tracking possible at all. It supersedes the DTCWT as the Phase 4 prerequisite; D1 now matters for *orientation* (Phase 4E) rather than as a blocker.
    *   `valid_interior_halfwidth(wavelet, level)` reports the R13 contaminated margin per level, and `apply_swt2d` refuses a level whose support exceeds the field rather than returning meaningless coarse scales.
    *   `swt_energy_fractions()` is the R3-compliant scale summary. Note that the construction applies a `2**level` amplitude gain, so **raw** per-level energies are not comparable across levels — measured on one field, raw energies read 3302/3192/2740 (apparently flat) while normalised they read 826/200/43 (clean octave decay). Same data, opposite conclusion.
*   **Hybrid Spectral Representation:** Separates fields into a low-frequency component (reconstructed via low-pass FFT filtering) and a high-frequency residual component (reconstructed via Haar DWT). Vectorized with `torch.meshgrid` to avoid loops.
    *   *Known defect:* the forward pass decomposes as `field = low + residual`, but `inverse_hybrid` recombines as `mixing_weight * low + (1 - mixing_weight) * high`. This is not the inverse of the forward split for **any** value of `mixing_weight`, so the `hybrid` transform always reports a non-zero reconstruction MSE for algebraic rather than physical reasons. Fixed in Task 3.5.5.

### 3.1 Training representations (`src/transform_engine/training.py`, T5.1a-e)

`RepresentationModule` is the PyTorch forecasting seam; it does not replace the analytical
registry. `RawRepresentation`, `FFTRepresentation`, `DCTRepresentation`,
`HaarRepresentation` and `DB2Representation` accept float32 or
float64 `(B,C,H,W)` tensors and return an `EncodedRepresentation`. The latter carries the
model-ready tensor plus immutable representation name, original shape, layout and synthesis
metadata. `with_values(model_output)` preserves that context without storing mutable call state
on the module, so concurrent calls cannot overwrite one another.

The FFT uses `rfft2` and packs real channels followed by imaginary channels. This avoids the
branch cut and zero-magnitude singularity of magnitude/phase while remaining consumable by an
ordinary real-valued model. DCT spatial shape is fixed at module construction and its two
orthonormal cosine matrices are registered buffers: forward/inverse allocate no matrices, and
`.to(device/dtype)` regenerates the deterministic cache once at the destination precision.
The shared `get_dct_matrix` evaluates float32 bases in float64 before casting, reducing measured
120x80 float32 round-trip maximum error from roughly `8e-5` to below `2e-6`.

**T5.1b adds the decimated wavelet reference path.** Haar and db2 use the canonical filter
definitions, PyWavelets-compatible `periodization` phase, explicit bottom/right dyadic padding,
and a recursively packed Mallat plane (`LL|LH / HL|HH`). The immutable context records padding,
filter length and the corrected accumulated-cascade margin at every level. Odd shapes reconstruct
after an explicit crop; implicit padding can be refused. Level-one bands agree with independent
PyWavelets oracles, multilevel energy agrees, float64 `gradcheck` passes, and reconstruction and
backward gradients have CPU/RTX-CUDA evidence. `implementation="reference"` retains this clear
per-tap numerical oracle.

**T5.1c adds the training kernel without deleting the oracle.** The default constructs the four
LL/LH/HL/HH kernels once as a registered buffer, evaluates a level with one strided `conv2d`,
and synthesises with one `conv_transpose2d` plus the exact adjoint of circular padding. It
matches reference coefficients, reconstruction and coefficient-loss input gradients to float64
noise. On float32 `(4,5,120,80)`, three levels, the RTX db2 round trip fell from 7.58 ms to
1.45 ms and the full round-trip-plus-backward step from 18.26 ms to 8.66 ms. Incremental CUDA
peak was 5.87 MiB versus 5.31 MiB for the reference, an explicit 0.56 MiB speed/memory tradeoff.
CPU also improved (7.39 to 3.92 ms round trip).

**T5.1d adds the undecimated training comparison.** `SWTRepresentation` packs final LL followed
by levelwise LH/HL/HH for each input channel, so its coefficient ratio is exactly `1+3L` while
every band keeps the parent grid. Haar, db2 and db3 are supported. The default policy follows
measurement rather than vendor preference: FFT circular filtering on CPU, four-band convolution
on CUDA/ROCm/MPS. A retained FFT oracle agrees with the convolution path in coefficients,
inverse and random coefficient-loss gradients; the packed batch also agrees with the existing
coordinate-aware analytical SWT and is exactly translation-equivariant under periodic shifts.
Metadata records accumulated support, valid interior shape at every level, amplitude gain,
energy normalisation, redundancy, boundary warning and the honest directional meaning: three
separable bands, **not** six signed orientations. On `(4,5,120,80)`, db2 L3 expands storage to
7.324 MiB (10x), and the measured auto paths take 19.81/47.22 ms CPU and 6.40/14.53 ms RTX for
round-trip/training-step respectively. RTX incremental peak is 25.46 MiB.

**T5.1e adds the orientation-aware training comparison.** `DTCWTRepresentation` vmaps the
canonical Kingsbury analysis and synthesis over flattened batch/channel items while reusing
registered level-1 and q-shift filter buffers. Its exact four-plane real atlas retains the six
complex bands at every native scale without interpolation: plane 0 recursively nests the
coarse pyramid and planes 1--3 hold level-1 real/imaginary components. Atlas adjacency is
explicitly storage geometry, never physical adjacency. The module reports parent and native
edge margins, native valid interiors, nominal and independently measured direction centres,
and refuses configurations with no two-dimensional valid interior. The analytical UI summary
shows native complex-magnitude maps with a shared within-level colour scale and the valid inset
drawn; it does not resample levels or paint phase as a scalar field. Coordinate-aware oracle
agreement, exact arbitrary-atlas packing, float64 gradcheck, random coefficient-loss gradients,
batch reconstruction, cached filter migration and CPU/RTX-CUDA parity pass. On float32
`(2,5,120,80)` at two levels, measured round-trip/training-step times are 46.23/138.34 ms CPU
and 35.87/65.91 ms RTX; encoded storage is 1.465 MiB and maximum round-trip error is 1.20e-6.

This is **T5.1a-e, not all of T5.1**. Non-NVIDIA hardware evidence, mixed
precision and compilation acceptance remain outstanding.

### 3.1f Motivating-experiment protocol (`src/forecasting/protocol.py`, `binding.py`, T5.0a-b)

`MotivatingExperimentProtocol` is the strict hand-off between laboratory evidence and platform
execution. Schema `motivating-forecast-protocol/v1` requires an evidence reference plus SHA-256,
exact domain bounds/shape/coordinate hash, source/version/variables/levels/cadence, input history,
frame and physical leads, transform package/version/filters/depth/boundary/packing/normalisation,
train-only data normalisation, fully dated splits and embargo, optimiser/scheduler/training budget,
rollout/history semantics, model identity and total/trainable parameter counts. Exact-key parsing
rejects both omissions and silently ignored additions; placeholder strings such as `unknown`,
`TBD` and `N/A` are invalid.

Cross-section validation proves physical durations equal cadence times frame offsets, embargo is
at least the longest lead, loss leads belong to the declared lead family and rollout feedback is
semantically consistent. Caller-owned free-form JSON is recursively frozen, canonical JSON gives
a stable full SHA-256 and the Python object has a stable content-derived hash. Save refuses
overwrite; load checks both schema and envelope hash. This proves a supplied declaration is
complete and unchanged. It does **not** prove that the declaration matches the professor's
experiment: that requires the actual referenced repository/config and its evidence hash, which
have not been supplied. Consequently T5.0 is partial and no T5.3c adapter has been selected.

`binding.py` closes the gap between a complete declaration and the objects actually executed.
It checks prepared dataset provenance against the protocol's source/version, variables/level,
cadence, history/leads, embargo, exact UTC split instants, grid shape/bounds/spacing/order and
explicit longitude convention. Coordinate SHA-256 is recomputed from the recorded coordinate
arrays; train-normalisation SHA-256 is recomputed from the recorded statistics. Dataset time,
grid and statistics hashes are full 64-hex SHA-256 values (the earlier 32-hex truncation was not
truthful under a `sha256` label). A model artefact binds only if its class, configuration,
parameter counts and transform configuration match and its training provenance names the exact
protocol, dataset, model version, optimiser and rollout.

The resulting content-addressed `ExperimentProtocolBinding` is accepted by evaluation schema v3,
which refuses a different dataset hash, checkpoint, variable/lead family, or batch timestamp
outside the declared held-out split. Unbound evaluation schema v2 remains available for generic
integration fixtures and is explicitly not protocol-conformant evidence. Binding proves identity
and declared conformance, not that the source evidence is genuine, the implementation is correct,
or the model has forecast skill. No real binding or UI-ready status exists until Adam's actual
laboratory evidence is supplied.

### 3.1g Forecasting integration (`src/forecasting/`, T5.3a-b)

`Forecaster` is the physical-space contract shared by learned adapters and baselines:
`predict(history, lead_count)` accepts `(B,history,C,H,W)` and returns
`(B,lead,C,H,W)`. `PersistenceForecaster` is the exact, zero-parameter no-change baseline; it
repeats the final observed normalised state without passing through a transform.

`ForecasterAdapter` wraps a one-step coefficient-space `torch.nn.Module` between an accepted
`RepresentationModule` and its inverse. It rejects output shape, dtype, device or finiteness
changes, reconstructs each prediction, and rolls multiple leads autoregressively in physical
space. T5.3a deliberately uses only the last history frame because the motivating inner loop is
`(B,C,H,W)`; that Markov assumption and the rollout policy are carried in provenance rather than
hidden. `TinyResidualCoefficientModel` and `run_tiny_deterministic_step` are importable
integration fixtures. The latter executes a real `RegionalForecastDataset` batch through
representation -> model -> inverse -> MSE -> backward and records loss, gradient norm, tensor
hashes, shapes, parameter count, seed/device/dtype and an explicit no-skill claim boundary.

Acceptance proves exact fixed-fixture reproduction on CPU, non-zero finite parameter gradients,
Haar analysis/synthesis in the loop, explicit contract refusals and CPU/RTX prediction/gradient
agreement through the vendor-neutral PyTorch `cuda` API. This is integration evidence only. The
professor's actual architecture, weights/training schedule, history semantics and forecast-skill
evaluation have not been integrated or run.

T5.3b separates model code, model identity and evaluation. `artifact.py` writes a tensor-only
PyTorch `state_dict` beside a canonical JSON manifest containing caller-declared model and
representation configurations, training provenance, fully qualified model class, parameter
counts, tensor-schema hash and checkpoint SHA-256. Loading verifies these before deserialisation
and uses `weights_only=True` plus strict state-dict matching. It never imports a class named by
the manifest. `load_laboratory_forecaster` binds the verified artifact into adapter provenance.

`evaluation.py` streams held-out batches through the forecaster and exact physical-space
persistence baseline on the same targets. Per lead and variable it reports standardized RMSE,
MAE, bias and MSE skill (`1 - model_MSE/persistence_MSE`). Physical-unit errors appear only with
explicit training standard deviations; combined-variable metrics remain standardized and state
their weighting. Perfect-persistence denominators are undefined (`None`), not infinite.
Prediction/target hashes, split, counts, dataset provenance, artifact-bearing model provenance
and a no-uncertainty/no-skill claim boundary form the evaluation record. This is a
real-model-ready contract, not evidence that the professor's model has been supplied or has skill.

### 3.2 Synthetic Field Generator & Perturbation Engine (`src/synthetic_generator/`)
*   **Deterministic Field Generator (`generator.py`):** Generates analytical 2D fields:
    *   *Sinusoid:* Linear combinations of 2D sine/cosine waves.
    *   *Vortex:* Sum of circular Gaussian plume cores with parameters centers, core radii, and amplitudes.
    *   *Front:* Sharp thermal-like front simulated using a hyperbolic tangent function (`tanh`) rotated via spatial coordinates.
*   **Perturbation Engine (`perturbation.py`):**
    *   Applies rotations and translations using PyTorch's `affine_grid` and `grid_sample` (bilinear mode with zero-padding).
    *   Applies Gaussian, Uniform, and Salt & Pepper noise.
    *   Computes sensitivity metrics including MSE, RMSE, Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index, and Spectral Energy Shift (MSE of the 2D Fourier magnitudes).
    *   *Precision note:* `compute_ssim` evaluates SSIM over a **single global window** (whole-field means, variances and covariance). It is not the standard Wang et al. locally-windowed SSIM with an 11x11 Gaussian kernel, and it is therefore insensitive to spatially localised structural change. It is used as a cheap global structure metric, not as a drop-in for published SSIM values.

### 3.3 Boundary-Condition Lab (`src/boundary_lab/boundary.py`)
Designed to study how boundary treatments affect regional models:
*   **Boundary Padding:** Supports circular (periodic), constant-value (zero), reflection (reflect), and edge-replication (replicate) padding.
*   **Spectral Windowing:** Applies Hann, Hamming, and Tukey windows. The Tukey window provides a smooth cosine taper at the edges while preserving original values in the center.
*   **Artefact Analysis:** Measures spatial error profiles and gradient magnitudes as a function of Euclidean distance from the boundary grid edges. Quantifies **Spectral Leakage** as the ratio of high-frequency energy in the padded field versus the original field.

### 3.4 Data Layer Adapters (`src/data_layer/adapters.py`)
Decouples data access from file systems:
*   **Under the Hood:** Supports direct NetCDF data ingestion from the `./data/` folder (looks for `era5_reanalysis.nc`, `gfs_forecast.nc`, and `toy_climate_model.nc` using `xarray.open_dataset` when files are present).
*   **GRIB is not supported.** No GRIB branch exists and `cfgrib` is not a dependency. Local
    adapters probe `.nc`; separately, T3.5.18 added opt-in ERA5 access through the public
    WeatherBench 2 Zarr archive on GCS with a provenance-carrying local cache. Direct
    Copernicus CDS and NOAA HRRR/GFS retrieval remain unsupported.
*   **Cache invalidation (fixed, T3.5.8):** `get_dataset` now records `{kind, path, mtime}` per dataset in `_sources` and re-resolves whenever the file appears, disappears or changes on disk, so a newly added `.nc` is picked up without restarting the API. `invalidate_cache(dataset_id=None)` forces a re-resolve.
*   **Source transparency (E2):** `get_source_info()` reports where a dataset actually came from, and `list_datasets()` / the `/api/v1/data/datasets` response now carry `source_kind`, `source_path`, `is_simulated` and `fallback_reason`. A run can no longer be silently satisfied by simulated data while the researcher believes they are using reanalysis — the fallback is a visible provenance fact, not a convenience.
*   **Simulated Datasets Fallback:** If the physical files are absent, seamlessly falls back to high-fidelity, in-memory simulated datasets built via `xarray.Dataset`:
    *   *ERA5 Reanalysis:* 2m temperature (`t2m`) and geopotential (`z`) with daily variations and sine waves.
    *   *GFS Forecast:* Temperature, geopotential, and wind components ($u$ and $v$ winds) modeled using latitudinal Gaussian distributions.
    *   *Toy Climate Model:* Simulated Sea Surface Temperature (SST) and Mean Sea Level Pressure (MSLP).
*   **Slicing Engine:** Provides a uniform `.slice_dataset()` interface to extract variables, nearest-neighbor query timestamps, select pressure levels, and crop geographical latitude/longitude bounding boxes.

### 3.5 Analysis & Diagnostics Engine (`src/analysis_engine/`)
*   **Spatial & Gradient Diagnostics (`diagnostics.py`):** Calculates spatial statistics (MSE, RMSE, MAE, Bias, SSIM, and Pearson correlation), gradient-magnitude mean absolute error, and gradient angular errors in degrees. Computes radial Power Spectral Density (PSD) and cross-spectral coherence.
*   **Turbulence Spectral-Slope Fitting (`diagnostics.fit_spectral_slope`):** Fits a power law `E(k) = C * k^(-beta)` by least squares in log-log space over a wavenumber band, returning `slope_beta`, `intercept_ln_c`, `r_squared`, and a `regime_interpretation` string that classifies the fit as a Charney/Kraichnan 2D enstrophy cascade (`beta ~ 3.0`), a Kolmogorov mesoscale kinetic-energy cascade (`beta ~ 5/3`), a flat/noise-dominated spectrum, or an intermediate regime. Invoked automatically for both forecast and ground-truth fields inside `compute_diagnostics`. *This module was fully implemented but undocumented in earlier revisions of this file.* Phase 4C generalises it into a reusable `fit_power_law` helper for scale-population exponents.
*   **Wavelet Energy (`diagnostics.compute_wavelet_energy`):** Sums squared coefficients per DWT subband (`LL`, and `LH`/`HL`/`HH` per level) to give a per-scale energy budget. This is the direct ancestor of the Phase 4C `ScaleSignature` object.
*   **Error Decomposition Engine (`decomposition.py`):**
    *   *Scale Decomposition:* Decomposes errors into low, mid, and high-frequency bands by applying bandpass masks in Fourier space and computing band RMSEs.
    *   *Boundary Decomposition:* Computes spatial error metrics binned by distance from the boundary.
    *   *Lead-Time Trajectory:* Computes error progression over simulated forecast lead times.

### 3.6 Declarative Experiment Engine (`src/experiment_engine/engine.py`)
Executes workflows described by JSON configurations:
*   **Parameter Sweeps:** Automatically expands parameter matrices into lists of Cartesian runs, with a safety threshold of 1,000 combinations.
*   **Pipeline Execution:** Runs step sequences (e.g., `slice_dataset` -> `apply_transform` -> `compute_diagnostics`).
*   **Dynamic Reference Resolution:** Dynamically resolves inter-step references such as `"{step_name.field_data}"` or parameters `"{freq}"` at runtime.
*   **Provenance Lineage Tracking:** Logs every step execution as a `LineageNode` (representing data, code, coefficients, or metrics) and `LineageEdge` (with relations like `executed_by`, `input_to`, `sliced_from`).

### 3.6d Statistical Validity (`src/statistics/`)

Delivered in T4C.5, closing **D8** - the defect that decided whether any reported finding is
defendable. Three modules, because the three ways to get a p-value wrong are independent and a
result needs all three handled at once.

**`multiple_comparisons.py`** - Bonferroni, Holm, Benjamini-Hochberg and Benjamini-Yekutieli.
BH and BY are validated against `scipy.stats.false_discovery_control` to **1e-16**. The default
is **BY**, which controls FDR under *arbitrary* dependence: parameter-metric pairs from the same
runs are dependent in ways not guaranteed to be positive, and assuming PRDS because it is more
convenient would trade validity for power. Every result carries the procedure's dependence
assumption, so a q-value cannot be quoted without it. `n_tests` lets a caller declare the true
family size - correcting only the survivors of a selection is not a correction.

It also polices the **power trap**: a surrogate p-value has a floor of `1/(1+n)`, so a
99-surrogate study cannot reject one of 500 tests after correction. Such a study reports
nothing and looks like a clean negative result. `check_power` says so instead.

**`surrogates.py`** - phase randomisation, AAFT, IAAFT, circular shift and block bootstrap.
Each destroys something specific and the choice *is* the hypothesis: phase randomisation
preserves the power spectrum exactly (measured to 1e-16), so a rejection can only come from
phase organisation, while a circular shift preserves every autocorrelation and destroys only
alignment - the correct null for a lagged claim. p-values use `(1+k)/(1+n)`, never `k/n`.

**`significance.py`** - surrogate tests, ESS-corrected correlation, and a **stationarity gate**.
The gate exists because FT surrogates assume stationarity, and without it the test is wildly
anti-conservative: measured false-positive rates of 0.11 (AR(1) phi=0.80), 0.16 (0.85), 0.39
(0.95) and 0.78 (random walk) against a nominal 0.05. Its threshold is **calibrated against
those measurements** rather than chosen - the test is well behaved to rho1 ~ 0.69 and unusable
beyond ~0.79, so the boundary sits at 0.75. The gate flags 0-15% of reliable series and 86-100%
of unreliable ones.

Calibration on a true null is now **0.045 at a nominal 0.05**. It was **0.765** until a bug in
the phase construction was found - see Section 7.2h.

### 3.7 Automated Hypothesis Engine (`src/hypothesis_engine/engine.py`)
Mines database results to generate hypotheses:
*   **Numerical Correlations:** Runs Pearson correlation coefficient calculations between experiment input parameters and numeric metrics, producing a `Hypothesis` for strong correlations ($|r| \ge \text{threshold}$).
*   **Categorical Optimizers:** Performs categorical group-mean analysis, finding the optimal categories for metrics and computing relative performance improvements.
*   **Adaptive Follow-Up Proposals:** Generates a new `proposed_experiment_config` targeting the optimal parameters (either expanding the range of numerical parameters in the correct correlation direction or fixing a categorical parameter to its best-performing category).

---

### 3.6c Execution: the Executor seam and device policy (`src/core/executor.py`, `device.py`)

Delivered in T3.5.19 and T3.5.21, closing the parallelism half of **D12** and **D18**
(standard E11, E9).

`Executor` is the seam: science code calls `executor.map(fn, items)` and the backend -
`serial`, `thread` or `process` - is configuration. Phase 6's Celery backend is meant to be
additive rather than a rewrite, and this is what makes that possible.

Three traps it exists to handle, each of which produces *wrong* science rather than merely
slow science:

*   **Result order.** Every backend returns results in **submission order**, never completion
    order. A sweep that aggregated in completion order would give different output on every
    execution from identical inputs.
*   **Seeding.** Seeds are derived up front with `SeedSequence.spawn`, one substream per run,
    so a run's randomness depends on its index and the root seed and on nothing else - not on
    which worker happened to pick it up.
*   **Thread oversubscription.** `n_workers` processes each defaulting to one BLAS thread per
    core gives `n_workers x n_cores` threads on `n_cores` cores; throughput *falls*.
    `device.thread_budget` divides the cores and the executor applies it inside each worker.

A fourth, specific to this codebase: a SQLAlchemy session cannot cross a process boundary. The
sweep is therefore split - `execute_run_payload` computes a run with **no database access** and
returns a payload; the parent, which owns the one session, persists in submission order. That
split is what makes serial and parallel byte-identical.

**The measured default is `serial`, and the reason is counter-intuitive enough to record.**
PyTorch already parallelises FFT and BLAS across cores, so run-level workers compete for cores
it is already using. With one intra-op thread, `thread(4)` gives a 1.40x speed-up; with all 12,
it gives **0.56x** - slower than serial. The `process` backend costs **~6.3 s** of fixed worker
startup on this platform (`spawn` re-imports torch per worker), so it only pays off for long
tasks or genuinely GIL-bound work. `GET /api/v1/health` reports this rationale alongside the
available backends, so the choice is informed rather than guessed.

`device.py` selects CUDA/ROCm -> MPS -> CPU with an explicit `SPECTRAL_DEVICE` override, and
**refuses** a device that is not present rather than silently falling back - a run that claims
to have used a GPU must have used one. `describe()` records the device, thread counts and
determinism mode into every run's `execution` column.

The portable profile layer adds `SPECTRAL_PROFILE=auto|cpu|accelerator|hpc`. `auto` is the app
default and preserves a complete CPU fallback; `accelerator` makes a missing accelerator an
error rather than an unnoticed slow run; `hpc` requires an active Slurm, PBS or LSF allocation
and maps scheduler local rank onto the available CUDA/ROCm devices. It deliberately cannot
submit or cancel work, so a home laptop never depends on cluster reachability. The compatibility
wrapper in `experiment_engine/engine.py` now delegates to this one policy rather than retaining
its former CUDA/CPU-only selector.

`src/core/doctor.py` is the zero-network preflight: `python -m src.core.doctor [--json]` reports
OS/Python/PyTorch, compiled CUDA or HIP runtime, devices, scheduler context, resolved profile,
and actual CPU/accelerator FFT reconstruction plus backward-gradient smoke tests. Readiness is
defined for the selected profile; failures on an optional backend do not make an explicitly
selected CPU profile unusable. `--require-accelerator` provides the stricter cluster/job guard.

SQLite is configured per-connection for concurrency: **WAL** journal mode (so readers and one
writer proceed together) and a 30 s **busy_timeout** (without which contention raises
`database is locked` immediately). Both are applied on a `connect` event, because pragmas are
connection-scoped and setting them once on the engine would leave every pooled connection
after the first with the defaults.

### 3.6ca Immutable evidence publication (`src/core/publication.py`, D71)

Every content-bound gate or evaluation receipt now reaches the filesystem through
`publish_new_bytes`. The contract combines two properties which an ordinary rename does not
combine portably: the target appears only after the complete payload has been flushed and
`fsync`'d, and an existing target is never replaced even when publishers race. The temporary
file is random-named and created beside the target, so the namespace operation cannot cross a
filesystem.

The no-replace primitive is selected by operating-system semantics rather than by optimism:
Windows `rename` (including the deployed exFAT volume), Linux
`renameat2(RENAME_NOREPLACE)`, macOS `renamex_np(RENAME_EXCL)`, with atomic hard-link publication
as the remaining POSIX fallback. A filesystem supporting none is refused; there is no
existence-check-plus-rename or overwrite fallback. The API claims process-crash atomicity, not
storage durability across sudden power loss, which additionally depends on the filesystem and
device.

The acceptance test runs under the repository-pinned `tmp_path` on `D:`, launches eight spawned
processes against one target and observes exactly one complete winner and seven refusals. It also
pins existing-target byte identity, cleanup after an injected unsupported-filesystem failure and
the exact primitive exercised. The ERA5 overlap, gate plan/run, campaign and external-evaluation
run/job writers consume this one boundary; their domain-specific error types remain outside it.

### 3.7b Dual-Tree Complex Wavelet Transform (`src/transform_engine/dtcwt.py`, `kingsbury_coeffs.py`)

Delivered in T3.5.6, closing **D1** — the oldest and highest-risk entry in the ledger.

*   **Six oriented complex subbands.** Passband centres measured directly (impulse response,
    FFT, energy-weighted axial mean) at **75.0, 45.0, 15.3, 164.6, 135.0, 105.4 degrees** for
    the wavevector, i.e. 165/135/105/75/45/15 for the feature orientation. Both conventions
    are named constants and both are reported, because conflating them is a silent 90-degree
    error.
*   **Near shift invariance:** 4.99% subband-energy spread over an 0–8 px translation,
    against 236.11% for the retained artefact and for a plain Haar DWT. It is *approximate*,
    not exact — the undecimated SWT (T3.5.7) remains the exactly shift-invariant transform
    and stays the right tool for Phase 4D tracking. The DTCWT is for **orientation**, which
    the SWT cannot supply: a separable transform has one diagonal band and cannot tell +45
    from −45 degrees.
*   **Perfect reconstruction** to ~1e-31 MSE across five shapes (including odd dimensions),
    four levels, three level-1 filter sets and four q-shift sets.
*   **Validated against two independent oracles** — the reference `dtcwt` package (agreement
    to 1e-13 elementwise) and `pytorch_wavelets`. Both are **test-only**: the coefficients are
    vendored in `kingsbury_coeffs.py`, generated by `tools/gen_kingsbury_coeffs.py`, so the
    runtime depends on neither. `pytorch_wavelets` imports the deprecated `pkg_resources`,
    and a platform meant to outlast its dependencies should not need it for ~80 constants.
    A test parses the module's imports to enforce this.

### 3.6b Registries and the Error Taxonomy (`src/core/registry.py`, `src/core/errors.py`)

Delivered in T3.5.15 and T3.5.14, closing **D15** and **D14** (standards E1, E2, E6).

`Registry` is a generic, decorator-based collection. An entry carries a description, a
parameter schema, capability metadata and tags, which is what lets the discovery endpoints be
**generated** rather than hand-maintained — a hand-written list goes stale exactly when
someone adds something. Duplicate names raise rather than overwrite, because a silent
overwrite makes behaviour depend on module import order.

`errors.py` is the explanatory exception hierarchy. Three properties it enforces: an error
says *what to do* (alternatives are listed, with a `did you mean` from edit distance); the
**error kind** decides the HTTP status, so a user's typo is a 404 with a helpful message
while a genuine fault stays an opaque 500; and every error carries structured context, so a
failure inside a 20-run sweep names the step, the action and the parameter combination.

Concrete registries built on it:

*   `transform_engine/registry.py` — six transforms behind one dispatch point, replacing
    **two parallel six-branch chains** in `engine.py` and `main.py` that had to be edited in
    lockstep. Capabilities (`shift_invariant`, `oriented`, `parent_grid`, `complex`) are what
    make T4B.2's wavelet bank selectable by property instead of by a hard-coded list.
*   `experiment_engine/actions.py` — seven actions, each declaring its own lineage node type
    and summary alongside its implementation. All three chains that listed the action names
    are gone and `engine.py` fell from **562 to 295 lines**.
*   `data_layer/sources.py` + `builtin_sources.py` — a priority-ordered fallback chain that
    records every attempt, including sources that **declined** and why. A source can be
    selected by capability (`SOURCES.with_capability("streaming")`), which is how T3.5.18's
    Zarr adapter will slot in.

`src/tests/plugin_example.py` is a worked example and an executable proof: it adds a data
source and a pipeline action in one new file, and the acceptance test verifies both appear in
the API *and* that `engine.py`, `adapters.py` and `main.py` are byte-identical afterwards.

### 3.6e The channel-series contract (`src/core/channel_series.py`, TG0.1, `ed-dev`)

Added on the cross-domain line (`roadmap_cross_domain.md`, Phase G0). It changes no
behaviour; it names an interface that already existed.

`cross_scale_dependency` and `support_floor` are the accepted falsification layer, and they
were written against `ScaleSignature`. That made it look as though the cross-scale question
required a wavelet decomposition of a 2D atmospheric field. Reading the code says otherwise:
between them the two functions touch `to_matrix(measure)`, `channels`, `channel_records` and
`provenance`, and nothing else — not the field, the transform, the grid, the variable or the
pressure level. The real interface between structure and inference is **labelled scalar series
on a shared clock, a per-channel validity mask, and a per-channel minimum-lag basis**.

Two tiers, because the two consumers genuinely need different amounts:

*   `ChannelGeometry` — labels, per-channel records, provenance. What a lag floor needs, with
    no data at all. `ChannelGeometrySpec` is its concrete form.
*   `ChannelSeriesLike` — the above plus `to_matrix`. What the dependency sweep needs.
    `ChannelSeries` is its concrete, domain-neutral form, validating clock monotonicity,
    label uniqueness and per-measure shape, and refusing a fabricated support the way
    `support_floor` already refuses a defaulted advection speed.

`ScaleSignature` satisfies both without being restructured: `channels` and `channel_records`
are added as aliases for `scales` and `interior`. The record *keys* are deliberately unchanged
— `valid_interiors` is published in every gate receipt and TG0.1's acceptance criterion is a
bit-identical receipt — so renaming them waits for TG1.5.

**Splitting the tiers was forced by running the code, not by design taste.** `gate_campaign`
performs its pre-acquisition lag-floor audit before any data exists, and had been expressing
that with a `SimpleNamespace` carrying `scales` and `interior`: duck-typing standing in for an
interface nobody had written down, in the same function whose grid handling produced D53. It
now declares `ChannelGeometrySpec`, as does the D53 regression test.

`test_channel_series.py` carries the Phase G0 claim as an executable assertion: a
`ChannelSeries` holding a signature's own numbers reproduces the signature's own
`cross_scale_dependency` result exactly — same configuration hash, same per-test p-values,
effect sizes, surrogate means, Theiler windows and floors. **If that test ever fails, the
inference layer is not domain-independent and the cross-domain programme has been falsified
at its cheapest point.**

**TG12.2a closes D69 without pretending sparse observations form a regular series.**
`ChannelSeries.present` is an explicit boolean `(time, channel)` observation mask, shared by
all measures and never inferred from `NaN`. A finite value at an absent sample is refused;
`present=True` with a non-finite value remains the distinct observed-but-invalid state. The
`non_stationary_support` declaration and the mask are enforced in both directions wherever a
series meets a domain. Presence is sliced and viability recomputed across train, embargo and
test partitions, hashed into `PartitionIdentity` without reading measures, and surfaced as
per-channel present/absent counts in lineage, API and UI.

Frame-lag inference on an intermittently present mask is deliberately refused. The controlled
Argo-like union clock used 20 floats, 146 ten-day cycles and distinct 12-hour surfacing offsets:
2,920 clock rows, 380 ordered pairs, zero simultaneous samples and zero pairs supporting any
requested contiguous lag. Compaction would invent adjacency and binning would invent
simultaneity, so neither is performed. `masked_frame_lag_assessment` records pairwise effective
N and contiguous-run admissibility; `decorrelation_frames` can measure only genuinely
lag-separated present pairs; and `_shift_null` can hold N fixed on a predeclared overlap for
audit and future physical-time work. No masked lag statistic, bias reassurance or significance
claim is emitted until that future estimator exists. An all-true declared mask takes the
literal pre-existing dependency path and is asserted result-identical to no mask.

### 3.6f Domain declarations and the first non-atmospheric adapter (`src/core/domain.py`, `src/analysis_engine/domain_analysis.py`, `src/data_layer/tabular_source.py`, TG0.2, `ed-dev`)

TG0.1 showed the inference layer *could* take channels from anywhere. TG0.2 sends some through
it, and adds the safety the atmospheric path had for free.

`DomainDeclaration` is what a source supplies besides data: declared axis roles (E14), declared
violations drawn from a fixed vocabulary rather than free text (E15), a licence, and a **lag
policy** (R21). It enforces rule R17 directly — a domain declaring no violations *and* no lag
floor is refused, because a source that breaks nothing is a second variable, not a second
domain, and is no evidence that the abstraction generalises. Licence is required rather than
optional, since TG8.2 will make export refuse a product whose source terms forbid it and an
optional field is one that gets left blank.

Three lag policies: `advective` (the atmospheric path, delegating to `support_floor`),
`declared` (an explicit floor in frames with a recorded basis), and `none` (no floor exists).
Under `none`, `analyse_precedence` refuses **before any computation** and names the domain, the
violation it declared and the two ways forward; `association_only` remains available and labels
its own output, because the difference between association and precedence here is what the
domain can justify, not what was computed. Lags below a declared floor are refused rather than
dropped — a family silently reduced to its testable members is indistinguishable from one that
had nothing to drop (R18).

`domain_analysis.py` is deliberately thin: `cross_scale_dependency` runs **unmodified**, with
the same surrogates, correction, power check and excluded-test accounting it applies to ERA5.
If a second domain had needed its own inference path, the abstraction would already have failed.
The one thing it adds at the boundary is a check that every channel declares its parent-axis
footprint, so a non-wavelet adapter never meets `support_floor`'s message about parent-grid
filter support. That footprint generalises honestly: a raw reading depends on exactly one
sample, a ten-minute mean of one-minute data on ten. It is never guessed — the same reasoning
that denies `advection_speed_m_s` a default.

`tabular_source.py` reads a local delimited file — one clock column, one column per channel —
into a `ChannelSeries` and its declaration, with a content hash and no network path at all. It
refuses an irregular clock unless the domain declares `irregular_sampling`, refuses to sort a
non-monotonic one, and refuses non-finite values rather than imputing them: interpolation
upstream of a dependence estimator manufactures exactly the short-lag structure R4 exists to
exclude.

**Measured, on a record with no grid, no transform, no advection and no annual cycle.** Three
AR(1) channels where `alpha` at `t` sets `gamma` at `t+3` and `beta` is a bystander: the
unmodified sweep recovers `alpha->gamma@3` with **0.9985 nats** of excess against **0.1458** at
lag 2 (the residue of alpha's own autocorrelation), implicates `beta` in nothing, and reports
12 tests adequately powered under BY. On three seeds of independent AR(1) channels it returns
**0 of 12** significant. The null returning null is the load-bearing half — T4C.3 already
measured a naive lag test falsely rejecting 20 of 20 AR(1) records.

**Boundary: no public dataset has been ingested.** This is offline-accepted infrastructure in
the same sense `cds_source` was before any live transfer. Nothing here is evidence about any
real-world domain.

### 3.6g The domain replication gate and its calibration (`domain_analysis.run_domain_gate`, TG0.3, `ed-dev`)

Phase G0 closes here. TG0.1 showed the inference layer could take channels from anywhere;
TG0.2 sent a non-atmospheric record through it. Neither established that a **null** record is
reported as a null rather than as an inconclusive run, and that distinction is the difference
between an instrument and a machine for generating plausible findings.

`run_domain_gate` validates the frozen `GateProtocol` **before** splitting, applies the generic
embargoed split `split_channel_series` (rule R6 — the embargo is returned, not dropped, so a
receipt can show the gap was real), sweeps both partitions through the unmodified
`cross_scale_dependency`, and adjudicates with `evaluate_replication_gate` unchanged. The
verdict therefore means here exactly what it means for ERA5.

**Measured on a domain with no grid, transform, advection or annual cycle** (900 frames, three
AR(1) channels, 6-test family, BY at 0.05, 499 surrogates, 540/3/357 split):

| Record | Verdict |
|---|---|
| `alpha` at `t` sets `gamma` at `t+3` | **PASS**, replicating `alpha->gamma@3` in both partitions |
| Independent AR(1) channels, 60 trials | **60 FAIL, 0 PASS** — false-positive rate bounded below **4.87%** at 95% confidence |
| A channel dropping out of the family | **INVALID**, not FAIL |

The third row is the one that matters most. Two tests corrected as though they were six is a
different experiment, and reporting its empty result as FAIL would present a design error as
evidence of absence. The 60-trial figure is stated as the bound a run of 60 can support, not as
a claim that the rate is zero.

**TG0.3 found a sixth atmospheric assumption that the TG0.1 audit missed.** `GateProtocol`
validated its measure against a literal four-name tuple of wavelet measures — an allow-list
implementing a deny-rule. What R3 forbids is a measure that *moves with a threshold*, not one
that lacks a wavelet name, and no generic domain can satisfy an allow-list of wavelet
vocabulary. Gate measures are now a registry (`GATE_MEASURES`, standard E1) whose entries
declare `threshold_free` and a justification. `threshold_fraction` is still refused — now by
name, carrying the measured reason (moving the threshold from 2 to 4 sigma changed it by more
than a factor of ten while every threshold-free measure was bit-identical). The four
atmospheric measures remain admissible and no accepted design's fingerprint changed.

### 3.6h Axis roles are declared, not read off the coordinate names (`src/core/axes.py`, TG1.1, `ed-dev`)

Phase G1 opens here. The task is not new science: it is to find where the atmosphere is
load-bearing inside code that reads as general, and to localise it **without moving a single
receipt**.

The first such place was `importers._spatial_dims`, which decided what an axis *meant* from
what it was *called* — two tuples of names (`latitude, lat, y, nlat` / `longitude, lon, x,
nlon`), and where those failed, the last two dimensions. Both halves are conventions of
gridded model output. Neither is removed. What changes is that they are now one **registered**
convention among possible others, and that every assignment records its authority:

| Basis | Meaning |
|---|---|
| `declared` | The caller supplied the role. Never overridden by a name. |
| `name` | Matched a registered naming convention (`AXIS_NAME_HINTS`, standard E1). A strong hint, still a hint. |
| `position` | The trailing-axes fallback. Correct for gridded output, unjustified elsewhere. |

`resolve_axis_roles` applies those three in that order and returns a frozen `AxisResolution`
carrying `roles`, `ordinals` and `basis`. `inspect` and `read_field` both accept an optional
`axis_roles` declaration, and both now write the resolution into the record — so *"the spatial
axes were chosen by position"* is a readable caveat on every orientation statistic instead of
an invisible one. `AxisResolution.require_declared` refuses `name` and `position` outright,
which is what `DomainDeclaration.resolve_axes` uses: a sensor archive whose columns happen to
be called `x` and `y` must not silently acquire a geometry, because a geometry is what
licenses per-metre reporting and this domain has no metre.

`AxisSpec` gains an optional `ordinal` — the position within a role, row before column — so a
domain that declares two spatial axes controls which is which. Ordering by ordinal is also
what makes a `(lon, lat)` file come back as `(lat, lon)`: a transposed field still looks like
a field, and every anisotropy number computed from it would be wrong undetectably.

**One deliberate behaviour change, asserted rather than tolerated.** The code replaced here
searched for a latitude name and a longitude name *independently*, and fell back to the
trailing two axes when either was missing. For dims `(lat, time, cols)` that pairs the clock
with the columns and calls the result spatial — not a transposition but a different field.
A single resolved spatial axis is now discarded rather than half-trusted, and the import
refuses. No file the atmospheric line has met takes this path: all five arrangements the
importer previously handled resolve identically, which
`test_every_arrangement_the_importer_met_resolves_as_it_did_before` asserts directly.

Adding an archive's vocabulary is a registration, from outside `src/`, exactly as a transform
is — `test_a_new_naming_convention_registers_without_editing_src` is the acceptance. Two hints
claiming one name are refused rather than resolved by import order.

### 3.6i Geometry is a registry, and operators ask what it can do (`src/physical_core/geometry.py`, TG1.2, `ed-dev`)

`GridSpec.kind` was three strings, and eleven methods in `grid.py` plus five call sites in
`operators.py` branched on them. Standard E13 makes it a registry: `pixel`, `cartesian` and
`latlon` register as the first three entries, each declaring capabilities, and every branch
becomes a method on the registered `Geometry`.

| Capability | What it licenses |
|---|---|
| `physical_metric` | lengths are a physical unit, so anything may be reported per metre |
| `uniform_metric` | one spacing describes the whole grid. **False for `latlon`** |
| `spherical` | the Laplacian is Laplace-Beltrami, not the five-point stencil |
| `has_latitude` | rows carry a latitude, so `cos(lat)` area weighting applies |
| `length_units` | the label printed beside every derived quantity |

`is_physical` and `length_units` now read the declaration instead of testing `kind !=
"pixel"`; `latitudes()` refuses on `has_latitude` rather than on a name; `gradient` asks the
geometry `north_sign` instead of deciding it from `kind == "latlon"`. `GridSpec` keeps the
arithmetic that is true of *any* geometry - endpoint-preserving resample scaling, crop bounds,
wavenumber axes - and delegates the rest.

**The defect the closed enum was hiding.** `laplacian` read `if kind == "latlon": spherical
else: cartesian_5point`. The `else` was not a fallback, it was a silent default: a geometry
whose metric varies across the grid - which is the interesting case, and precisely why
`latlon` needed its own branch - would have been handed one representative spacing and
returned a finite array labelled `value per m^2`. No exception, no warning, no NaN. That
branch now states its precondition (`assert_uniform_metric`), and the acceptance test proves
it fires: `polar_scan`, a radar PPI sweep registered from the test module, has a metric that
grows with range and is refused. `gradient`, which already divided by a per-row `dx_metres()`,
generalised for free - the contrast between the two operators is the finding.

`GridSpec` also gains `params`, a sorted tuple of geometry-specific scalars. Without it the
registry would have been open in name only: `lat0`, `lon0` and `radius_m` are *`latlon`'s*
parameters that happen to have named fields for historical reasons, and a fourth geometry
with a different parameterisation could not have existed. `polar_scan` stores its `r0` there
and refuses construction without it, so a grid with no metric is refused at construction
rather than discovered at the first gradient.

**One deliberate behaviour change.** `from_coords` now resolves coordinate names through
TG1.1's registry, so `latitude`/`longitude` spelled in full - CF's spelling and ERA5's - is
recognised as a sphere. It previously matched only the abbreviation and such a field received
a **pixel** grid: gradients per pixel, no `cos(lat)` weighting, a spherical metric present in
the data and discarded. The same family as D56, found in the same way. Nothing in this
repository builds a `PhysicalField` with those spellings - the importers normalise to
`lat`/`lon` - so no existing result moves, and the five arrangements `from_coords` already
handled are asserted unchanged.

### 3.6j Random streams belong to the task, not to the process (`src/core/randomness.py`, D55, `ed-dev`)

`executor._run_one` opened every task with `torch.manual_seed(seed)` and `np.random.seed(seed)`.
Both mutate a generator owned by the *process*. Under `serial` and `process` exactly one task
sits between its seeding and its draws; under `thread` with more than one worker every thread
shares the one generator, so worker B's seed lands inside worker A's seed-to-draw window.

The window is microseconds for a toy payload, which is why the property test passed. Held open
by 10 ms of work - which is what a sweep payload is - `thread(8)` disagreed with `serial` in
**10 of 10** trials.

Each task now gets a `TaskStreams` bound to a `contextvars.ContextVar` for exactly its own
duration. A `ContextVar` set inside a thread is invisible to every other thread, which is the
isolation required and nothing more; it costs one binding in the serial and process backends;
and payload code reaches the stream without every intervening signature growing a `generator=`
parameter, which would have meant a mechanical rewrite of the action layer.

*   **The values did not move.** `torch.Generator().manual_seed(s)` yields the same sequence
    as the global generator after `torch.manual_seed(s)`, asserted directly.
*   **The cheap fix was rejected on purpose.** Refusing `n_workers > 1` when seeds are supplied
    would have preserved every value too, and left process-global randomness inside a unit of
    work the platform is explicitly allowed to run concurrently.
*   **Unseeded stays unseeded.** Outside a task there is no stream; `add_noise` records
    `rng_source` and `seeded: False` rather than inventing one, because an unreproducible
    result must stay distinguishable from a reproducible one (defect D12's lesson).
*   `enable_determinism` no longer seeds the process. It records `seed_scope`, which reports
    `task`, `unbound` or `task_mismatch`, so a reproducibility claim is checked against the
    binding rather than against an assumption.

**The finding, which is larger than the defect.** `test_sweep_is_byte_identical_across_backends`
is the acceptance criterion for "a sweep is byte-identical across executor backends" - and its
pipeline was a vortex and a wavelet transform. **Entirely deterministic. Not one random draw.**
It certified a reproducibility claim with a payload that had no seed-dependent behaviour to get
wrong, so it could not have caught D55 and did not. The sweep now carries a noise step, and
`test_the_acceptance_sweep_actually_draws_random_numbers` guards the guard by changing the root
seed and requiring the persisted results to move. This is the third consecutive slice in which
execution found what reading missed, and the first in which the thing that missed it was a test.

### 3.6k The lag floor is a registered policy, and the sweep applies the declared one (`src/core/lag_policy.py`, TG1.3, `ed-dev`)

Rule R21 asks one question before any sweep runs: *is a lag admissible at all?* Three answers
existed - an advective crossing time, a domain-declared floor, or no floor - and they were
spread across four files as branches on the string `DomainDeclaration.lag_policy`. Standard
E16 makes them a registry: each policy declares what it licenses, the analysis layer asks,
and the policy is applied where the floor is used rather than only where the caller meets it.

| Capability | What it licenses |
|---|---|
| `precedence_admissible` | rule R21 permits a lead-lag reading of a positive result |
| `floor_from_declaration` | the floor is known from the declaration alone, before any data |
| `floor_from_geometry` | the floor is computed from the record's own physical grid |
| `requires_channel_support` | the policy reads each channel's parent-axis footprint. **False for every policy that does not measure a filter crossing** |
| `parameters` | the extra arguments the policy consumes; anything else is refused, not ignored |

`support_floor` moved into the module unchanged - including the D48/D53 physical-grid
reconstruction - as the `advective` policy's implementation, and `cross_scale` re-exports it
for the two gate modules that call it directly to audit a frozen atmospheric plan.

**The defect the closed set was hiding.** `cross_scale_dependency` did not branch on the
policy at all: it called `support_floor` unconditionally, so the floor that actually excluded
tests was always the atmospheric one. Under `lag_policy='declared'` the domain's floor was
enforced at the entry point, in `analyse_precedence`, and then every test record in the receipt
reported `support_floor_frames: 1` with an exclusion reason citing rule R4 - a rule about
wavelet filter geometry, quoted to a domain that had declared it has no propagation mechanism.
The tests that ran were the right ones. The receipt described a different study, and a receipt
that describes a different study is the failure this system exists to prevent. The sweep now
takes a bound policy and asks it for the floor, the exclusion wording and its own contribution
to the analysis fingerprint. The `advective` policy contributes exactly the key the old code
hard-coded, so every atmospheric `analysis_config_sha256` is byte-identical.

**Two smaller ones, found by the same move.**

*   **D58.** `run_domain_gate` refused `require_advection_floor` unless the domain declared
    `lag_policy='advective'`, and then called `analyse_precedence` with no way to pass a
    speed - which that policy requires and is deliberately denied a default. The advective
    path through the domain gate was therefore unreachable: each half looked correct, and the
    pair could not be executed. It now takes `lag_policy_params`, and a test runs it.
*   **`require_advection_floor` accepted a substitute.** The replication gate checked only
    that *a* floor was enforced. A declared floor is a floor, but it is not the one the
    protocol froze, and rule R18 fixes the design before the run rather than accepting a
    substitute during it. The check now names the policy.

**One deliberate behaviour change.** A domain whose policy cannot read `support_parent_px` is
no longer required to supply it. Before TG1.3 every domain paid that entry price, because
`support_floor` ran whatever the declaration said - so a logger reporting every two minutes,
whose floor comes from the logger and not from a filter, had to declare a wavelet footprint
that could not reach any floor it applied. The number was required, could only be invented,
and changed nothing. It is still accepted and still reported where it is declared; it is only
required by policies that read it.

**Acceptance met.** `test_lag_policy_registry.py` registers `instrument_response` - a floor
per channel from each sensor's settling time - entirely from the test module, and runs a
complete sweep through it. It was chosen to be awkward on the axis the builtins are not:
`advective` varies per channel but only from a physical grid, `declared` needs no data but
gives every channel the same number, and this one needs neither. A fourth policy that was
`declared` under another name would have proved nothing.

### 3.6l The sibling sample spine (`src/core/sample.py`, TG1.4, `ed-dev`)

Section 2.2 of the cross-domain roadmap calls `PhysicalField`'s strict 2D constraint "the
single largest obstacle" to a second domain, and standard E12 forbids relaxing it. TG1.4 is
the demonstration that both hold at once. `StructuredSample` is an array of **any** rank whose
axes are declared `AxisSpec`s, and it is not, and does not become, a field.

**Declaration is structural here, not a check.** Axis roles resolve through TG1.1's registry
with *both* inference stages switched off, and an `AxisSpec` cannot exist without a role - so
there is no constructible sample whose roles were guessed, and `AxisResolution.basis` is
``declared`` for every axis of every sample. That is the refusal the type exists for: an array
whose axes are called `x` and `y` is not a field, because a field licenses per-metre gradients,
radial binning and area weighting, and a domain with no metre must not acquire one by spelling.
A sample therefore carries **no geometry at all**; one is asserted at the bridge, by a caller,
and recorded as theirs.

**The route into the accepted falsification layer does not pass through `PhysicalField`.**
`channel_series_from_samples` reduces each frame along its non-channel axes to one scalar per
channel, producing the `ChannelSeries` that TG0.1 showed `cross_scale_dependency` actually
consumes. Rank, extra axes and the absence of a metric are all reduced away before the analysis
layer sees anything. The reductions are a registry (`SAMPLE_REDUCTIONS`), and each declares the
gate measure it emits - a capability that is load-bearing rather than decorative: the measure
name is what a `GateProtocol` freezes and what rule R3 is adjudicated on, so a reduction
emitting a name no measure registry knows is refused at build, and two reductions emitting one
measure are refused rather than silently merged.

**The bridge is one-way and narrow.** `StructuredSample.to_physical_field` is the counterpart
of `training.py`'s batch boundary in the other direction. It requires exactly two axes, both
declared ``space``, in array order, and refuses everything else with the condition that failed
and a pointer to the path that works. Two properties follow, and both are asserted:

*   Nothing in the analysis spine accepts a sample. The bridge is a method of the *sample*, so
    the direction of the dependency is visible at the call site, and `src.core` does not import
    `physical_core` at module scope - a non-gridded domain does not pull in torch to declare
    an axis.
*   A declaration whose spatial ordinals run against array order is **refused, never
    transposed**. `PhysicalField` reads array dimension 0 as the row, so transposing here would
    fix the array and leave every orientation and anisotropy statistic computed from it
    silently wrong - the failure `NameHint.ordinal` prevents in TG1.1, one layer up.

**Acceptance met, and deliberately not only by construction.** A three-band photometer whose
frame is a ``(band, detector, repeat)`` block runs the unmodified sweep and the unmodified
replication gate: the planted coupling is recovered, the AR(1) control is not, the gate returns
PASS, and the floor that decided admissibility is the one the domain declared. The pair matters
more than the single positive - a detector that finds a planted signal and also finds signal in
autocorrelated noise has found nothing, and without it a sample type is speculative generality
with a docstring. Alongside it, `PhysicalField` is asserted still to raise on shapes `(16,)`,
`(4,4,4)` and `(2,3,4,5)`.

**One change outside the new module.** `from_channel_series` published only the *names* of a
series' provenance keys, never their values. That was harmless while every producer's record
lived on the `DomainDeclaration`, which `describe()` carries into every result whole. The sample
spine puts the arithmetic that produced the numbers on the *series*, and two reductions may
legitimately emit the same gate measure - so a receipt naming only the measure could not say
which arithmetic it described. The lineage record now carries the values, with anything it
cannot hold (an array, a tensor, a `GridSpec`) replaced by its type name rather than dropped: a
key with no value reads as a key with no content. `provenance_keys` is unchanged, and no
atmospheric receipt passes through this function.

### 3.6m The vertical coordinate is declared, and pressure is one instance (`src/core/level_axis.py`, TG1.5, `ed-dev`)

The last Phase G1 task, and the last item in section 2.2's list. Three places carried a
vertical level and no two agreed on how: `LevelBank` was keyed on `Dict[float, ...]` where the
float was hectopascals, `ScaleSignature` carried a field named `level_hpa`, and
`CoefficientField.level` carried a bare number with no units at all. The unit was in an
attribute name, in a docstring, and nowhere.

**The part that was not cosmetic.** `LevelBank.vertical_offsets` decided direction with
`"upward" if upper < lower else "downward"` - right for pressure, right for depth, and exactly
backwards for height or altitude. That is the one place in the tree that reports the *direction*
of a vertical precursor relationship, which is the sign distinguishing an upper-level trough
from a surface one. Nothing shipped wrong, because no height axis exists; the point is that
nothing could have gone right either, and the rule sat at the site where the finding is made.

A `LevelCoordinate` declares two things and no more:

| Field | What it decides |
|---|---|
| `units` | what the number means; a coordinate with no units is refused at construction |
| `increases_upward` | which way along the axis is up - the whole content, and the reason depth and pressure agree with each other and disagree with height |

`LEVEL_COORDINATES` registers `pressure_hpa`, `height_m` and `depth_m`; a fourth registers from
outside `src/` and drives a whole bank. `LevelCoordinate.axis_spec()` returns a declared
`level`-role `AxisSpec`, so the roadmap's phrase is literal rather than nominal - the same spec
type TG1.1 resolves and TG1.4's `StructuredSample` accepts.

**Ordering and direction are separate, and both are recorded.** Levels are still ordered by
ascending coordinate value, because that is what `LevelBank` already did and changing it would
reorder the stacked tensor of every existing bank. Which end of that order is the top is then
the coordinate's business: `summary()` carries `ascending_order_runs`, so a reader never has to
know the convention to read the result.

**The declaration travels with the number.** `zarr_source`'s cached reader stamps
`level_axis: pressure_hpa` on each frame - it selected an ERA5 pressure level by name, so it is
the one place entitled to declare the coordinate. `decompose_sequence` reads it from the
sequence metadata by the same rule it already read `level`, `decompose_levels` stamps every
field in a bank, and `scale_signature` carries it onto the signature. A level that arrives
without a coordinate reports `level_units: None` rather than being assumed to be pressure.

**`level_hpa` survives as a property, and it is now a claim that can be wrong.** "The 850-hPa
signature" is how the atmospheric line talks, and removing the spelling would make every reader
carry the axis check itself. It refuses a non-pressure axis rather than returning `None`,
because a `None` there would read as "this signature has no level" about a signature that has
one.

**One correctness change beyond the rename.** The streaming signature refused frame-to-frame
drift in grid, coordinates, variable, level and units. The vertical coordinate now joins that
list: 500 on a pressure axis and 500 on a height axis are different frames, and a record that
switched axis partway would have passed every other check, because the number never moved.

**Receipt keys changed, deliberately and for the first time in G1.** `levels_hpa` became
`levels`, `from_level_hpa`/`to_level_hpa`/`offset_hpa` became `from_level`/`to_level`/`offset`
with `level_units` beside them, and `ScaleSignature.summary()["level_hpa"]` became
`level` + `level_axis` + `level_units`. TG0.1 explicitly deferred this rename to TG1.5, and
this is where it lands. Nothing hashed moves: `analysis_config_sha256`, the artifact digests
(which hash tensor bytes) and the frozen campaign hash are untouched, and `gate_run`,
`gate_campaign`, `cds_source`, `era5_overlap` and `regional_forecast` keep their own
`level_hpa` vocabulary because section 2.2 lists them as permanently atmospheric.

### 3.6n The canonical feature record (`src/core/feature.py`, TG2.1, `ed-dev`)

The first piece of the machinery section 2.3 of `roadmap_cross_domain.md` records as entirely
absent: `grep -rE 'SpectralFeature|FeatureTrack|constellation|motif' src` returned five hits,
all comments saying "not implemented". TG2.1 is deliberately the **record** rather than the
extractor. What a feature is has to be settled before anything decides how to find one, because
every later phase reads this vocabulary and rule R20 will eventually freeze a motif expressed
in it.

**The record exists to enforce R19.** Mapping sea-surface temperature and trading volume into a
common structural vocabulary makes their structures comparable and their quantities not. So the
canonical description does not replace the original one: domain, dataset, variable and units
travel with every feature, and the comparisons R19 forbids raise `SemanticComparisonError`
rather than being documented as unwise. A comment is not an invariant.

Two views, and only one of them crosses a domain boundary:

| View | Contains | Who may read it |
|---|---|---|
| `describe()` | all fifteen roadmap fields, with units and provenance | receipts, and any reader working inside one domain |
| `structural_signature()` | scale ratios, orientation, significance, representation | a cross-domain matcher (phases G3 and G5) |

`structural_signature()` is **built** from the quantities that survive being stripped of their
units, not filtered out of `describe()`. Adding magnitude back is then a visible edit to that
method rather than an invisible consequence of a key not being removed.

**Every number carries its units, including the ones that have none.** `Quantity` holds value,
units and uncertainty together because those three are separated at exactly the moment somebody
needs them together. `units=None` is a legitimate declaration - the quantity is dimensionless -
and it is distinct from "the units were never recorded", so an empty string is refused.

**Orientation is the trap.** A ridge at 170 degrees and one at 350 degrees are the same axis; a
wind vector at 170 and one at 350 are opposed. The wrap period is a property of the quantity and
not of the number, so `ORIENTATION_CONVENTIONS` registers `axis_180` and `direction_360`, the
arithmetic reads the declaration (standard E16), and comparing two orientations under different
conventions raises. TG2.3's tracker gates association on orientation difference; that gate is
wrong by up to a factor of two in one of these two cases if the convention is assumed.

**Significance carries its resolution.** An empirical p-value from `n` surrogates cannot be
smaller than `1 / (1 + n)`, on the same `(1 + k) / (1 + n)` convention `surrogate_null` already
uses. `Significance` refuses a value below that floor, and refuses an empirical p-value with no
ensemble size at all - a record claiming `p = 1e-4` from 999 surrogates is reporting a number
the ensemble could not have produced, and that number would then be corrected, ranked and
published. `declared_none` makes "not tested" a state a receipt can distinguish from "tested
and weak".

**A location is on declared axes, and knows its own topology.** `FeatureLocation` holds
`AxisSpec`s (standard E14), refuses a `time`-role axis (a feature's time is a field of its own,
and a second copy is a second clock that can disagree), refuses mixed units across its axes, and
**refuses a separation across a periodic axis when the axis length is not supplied.** That last
refusal is the difference between two cells and a hundred and twenty-six.

**There is no `uncertainty` field.** The roadmap's list names one, and a single uncertainty for
a record holding a magnitude in kelvin, a location in cells, a scale in metres and an angle in
degrees would be a number with no unit and no referent. Uncertainty lives with each quantity and
`describe()["uncertainty"]` assembles the per-quantity view, so the field the roadmap asks for
is in the receipt without a lie in the dataclass.

**`FeatureSet` is homogeneous by construction** - one domain, one dataset, one variable, one
representation, one clock. Everything TG2.3 and TG3.3 will do with a set is arithmetic on
coordinates, scales and times, and a set that quietly mixed two domains would let all of it run
and produce comparisons R19 forbids. Mixing representations is refused for the second reason
too: TG2.4's audit asks which representation manufactured a feature, and cannot ask that of a
set whose features came from several.

**Given a real user rather than a docstring.** The tests build features from measured centroids
and measured widths of the `advected_vortex_sequence` benchmark - never from its recorded truth
- and recover the known step velocity to 0.25 cells and the known scale-doubling ratio to 10%.
A record that has only ever held a hand-written literal has not been tested.

**Defect D59, found by giving it that user.** `truth_advected_vortex` takes the vortex position
modulo `n`, so its recorded answer is a trajectory on a torus, and `build_advected_vortex` draws
the blob with a plain Euclidean Gaussian that does not wrap. At the benchmark's own parameters
the vortex never reaches an edge, so the two have never disagreed. Started near one, the
recorded position and the field part company by several cells for the frames where the blob is
clipped. This is TG2.3's problem before it is anyone else's: that slice's acceptance criterion
is *1 track, 1 birth, 0 deaths*, and a tracker run on a wrapping parameterisation would see the
object fade out at one edge and appear at the other, and would be right. The benchmark is left
alone - changing its field builder would move every number it produces, and TG2.1 is not the
slice that gets to do that - and the disagreement is asserted by a test so the next slice meets
it as a fact rather than as a surprise. **TG2.3 closed it** by declaring the topology instead
of picking a side; see Section 3.6p.

### 3.6o Feature extraction as a registry (`src/core/extraction.py`, TG2.2, `ed-dev`)

TG2.1 settled what a feature *is*. TG2.2 settles who decides that one is there, and the answer
is: not this module. `EXTRACTORS` is a registry and `local_maximum` is the **first** entry in
it, not the definition of extraction. A watershed, a persistence filter and a matched filter
would each disagree with it about the same field, and a tree that hard-codes one of them has
asserted that the disagreement does not matter.

**What the framework keeps, and what the registry gets.** A registered extractor receives the
field and the calibration and returns `Candidate`s - positions, magnitudes and scales in the
units of the declared axes - and nothing else. `extract()` calibrates the null, dispatches, and
builds the `SpectralFeature` records itself. `Candidate` has no `domain`, no `representation`
and no `significance` field, so an extractor has no way to choose its own; attaching the
representation (R8), attaching the surrogate p-value with its ensemble size (TG2.1's resolution
floor) and producing records R19 can refuse by name are the parts that must not vary, and a
plug-in is exactly the thing that would vary them.

**The threshold is calibrated, never chosen.** Rule R3 exists because "coefficient > 0.6" is a
number somebody picked, and a discovery pipeline built on picked numbers discovers the picks.
The cut here is an order statistic of the distribution of the **maximum** of a surrogate field
with the same power spectrum and randomised phases: a peak is reported when it exceeds what the
strongest peak of a structureless field with this spectrum does, at a declared family-wise level
over the whole frame. Three consequences follow, and each is a test:

*   The p-value on the record is the tree's existing `(1 + k) / (1 + n)`, so `Significance`'s
    resolution floor applies with no translation - and an `alpha` finer than that floor is
    refused *before* the ensemble is built. Without that refusal the extractor returns nothing
    and the receipt says the field was empty, which is the most expensive failure available
    because it is indistinguishable from a genuine null and it is silent.
*   The comparison against the threshold is **strict**. `k` counts null maxima at least as large
    as the observation, so an observation sitting exactly on the cut ties with it and reports
    `p` one step *above* alpha. Accepting it with `>=` reports 0.06 under a heading that says
    0.05. The off-by-one was live for one test run and is now held shut by a test of its own.
*   Both null benchmarks return **nothing**, across three seeds each - and the same fields still
    contain thousands of local maxima, so the silence is the calibration working rather than an
    extractor that cannot find anything. Loosening alpha to 0.5 turns the same white-noise field
    into findings, which is the control that claim needs.

**The suppression radius is measured, not chosen.** The one free number a peak-finder usually
carries is how far apart two features must be, and it is the number that decides how many
features exist. Each accepted feature is localised first and then suppresses its neighbourhood
out to a multiple of *its own measured scale*. Fixing that radius manufactures features: on
`planted_configuration` at `scale_factor=2.0`, a radius tuned at `scale_factor=1.0` reports
eight features where three were planted, and all five extra ones are noise maxima on the
shoulders of real blobs - above the calibrated threshold, and indistinguishable in a receipt
from a discovery. With self-scaling suppression the count is exactly three across a six-fold
range of feature widths.

**Two textbook estimators were measured against the benchmark and rejected.**

| Estimator | Why it looked right | What it measured |
|---|---|---|
| three-point sub-cell parabola | exact for a noiseless Gaussian, and free | its denominator is the second difference, of order `A / sigma^2` - 0.028 against a noise amplitude of 0.05 on `planted_configuration`. Errs by up to 0.68 cells where a windowed centroid errs by 0.27. Valid only when the feature is a cell or two wide, which nothing in this tree guarantees |
| curvature of the same fit, as a scale | one fit yielding both position and width | returned 2.1-2.9 cells for a planted width of 6.0 |

What is used instead is a windowed centroid whose window sizes itself from the measured scale,
with the integral estimator `sigma = sqrt(I / (2 pi A))` corrected for the square window's
truncation by `erf(r / (sigma sqrt2))^2`.

**The brightest sample is not the amplitude.** It is the largest of many noisy samples, so it is
biased high, and the bias grows as the feature broadens and more cells compete to be the
maximum: on `planted_configuration` the peak sample overstates a unit-amplitude feature by 3% at
`sigma = 3` cells and by 12% at `sigma = 18`. Fed to the integral estimator that becomes a scale
biased *low* by up to 9% - in a direction that does **not** cancel in a ratio, because it is a
function of the feature's own size, and a ratio of scales is the only form in which a scale
leaves its domain (TG2.1). The magnitude is therefore the mean over a disc of `sigma / 2`
divided by the analytic Gaussian disc-mean, which divides the noise by the root of the cell count
and leaves per-feature errors under 5% across a six-fold range of scales. The raw sample stays in
`provenance` so the correction is visible rather than merely applied.

**No fabricated uncertainty.** Re-measuring each scale with a window of `2 sigma` and of
`3 sigma` gives a spread of 0.01-0.24 cells, and that spread does not cover the truth: at
`sigma = 18` the two agree to 0.1 cells while both sit 1.4 cells low. It is a repeatability, not
an accuracy, and putting it in `Quantity.uncertainty` would understate the error by an order of
magnitude in exactly the records a reader would trust most. The field is left `None`, which
TG2.1 defines as "not recorded".

**The declared axes change the arithmetic (standard E14).** A periodic axis wraps: the
neighbourhood comparison, the localisation window and the reported coordinate wrap together, and
a feature straddling the seam is found **once** at its true position (within 0.25 cells) rather
than twice at two false ones. The same array declared non-periodic is a different problem and
gets a different answer - rule R13, sized by the feature rather than by a fixed margin. A peak
two cells from a non-periodic edge has half its integral outside the frame; the truncation
correction then applies for the wrong reason and returns a position three cells out and a scale
24% low, with nothing in the record to say so. It is refused and counted instead. A missing
feature is a fact a receipt can carry; a confidently mismeasured one is not.

**Finding nothing is a result.** `FeatureSet` refuses to be empty, and its docstring says an
empty set "has no domain, and 'no features found' is a result that belongs beside the search that
produced it". `ExtractionResult` is that object: it still knows what was searched, with which
extractor, at what threshold, and what was rejected on the way - `below_threshold`,
`outside_valid_interior`, `suppressed_by_a_stronger_feature`, `unmeasurable_scale`. "No features"
and "four thousand local maxima, none of which cleared the cut" are different statements about a
field, and TG2.4's audit is a question about the second one.

**Measured against answers recorded before the module existed.** On `planted_configuration`:
three features found, always three, position error under 1 cell, measured widths within 6% of the
planted width across `scale_factor` 0.5 to 3.0, and pairwise separations within 1 cell of the
planted 20 to 120 - under rotation, translation and rescaling. On `advected_vortex_sequence`:
exactly one feature in each of 24 frames under a single calibration, every position within 1 cell
of the recorded trajectory, and the measured scales reproducing the known 16-step doubling to
within 5% without being told it exists. The benchmark's own `4E.feature_detection` check is
deliberately **not** rewired to call this extractor: a benchmark that validated the code under
test with the code under test has stopped being an independent answer.

### 3.6p Frame-to-frame association (`src/core/tracking.py`, TG2.3, `ed-dev`)

TG2.1 settled what a feature is; TG2.2 settled who decides one is there. This is the first
claim in the tree that is not a measurement of a single array: nothing in a pair of frames
says two blobs are one object, so association is an *inference*, and the design is entirely
about keeping the inference's assumptions visible instead of dissolving them into constants.

**The framework/registry split is TG2.2's, unchanged.** A registered associator receives a
cost matrix and a boolean mask of admissible pairs and returns pairs. It never sees a
feature, a unit or a time, so it cannot invent a gate, cannot bridge a gap, cannot decide
what a birth is and cannot attach anything to a record. `ASSOCIATORS` holds `greedy_nearest`
and `hungarian`, and a third registered from the test module drives the tracker with no edit
to `src`.

**The gate is derived, never chosen.** A tracker's one free number is how far a feature may
move between frames, and it decides how many objects exist in exactly the way TG2.2's
suppression radius decided how many features exist. The ceiling here is a **coincidence
radius**: the distance at which the expected number of *unrelated* features from the target
frame falling inside the search ball equals the same `alpha` the extraction was already
calibrated at.

    r  =  ( alpha * measure / (count * V_d) ) ** (1/d)

On the vortex benchmark - one feature, a 128x128 frame, `alpha = 0.05` - that is 16.1 cells,
against a true step of 2.9. It is not a motion model. It is the point past which proximity
stops being evidence, and it moves with the frame: sixty-four features in the same frame
tighten it by a factor of eight, because a gate that does not tighten when the field crowds
is a gate that manufactures tracks precisely where the data is least able to support them.

**Everything stricter is declared physics, not tuning.** `MotionBounds` carries a maximum
speed, a maximum scale-doubling rate and a maximum turn rate; all three default to `None`,
which means *not declared* rather than *unbounded by assumption*, and the receipt reports
which gates were active so a run with no declared physics cannot be mistaken for one whose
physics happened to be satisfied. Bounds are **rates**, multiplied by the actual elapsed time
of each link: the same 12-cell step is admissible over six frames and refused over one under
one declared bound of 3 cells per frame, and a gate expressed per *frame* would have answered
the same in both cases and been wrong in one of them. A declared bound can only tighten the
coincidence gate, never widen it - generous physics does not license a link chance already
explains.

**A gate on a quantity the extractor does not measure is refused.** This slice's acceptance
criterion names scale and orientation gating, and the only extractor TG2.2 registered
declares `reports_orientation: False`. Passing every pair because the quantity is missing
would put the gate in the receipt while refusing nothing, and the run would then be
indistinguishable from one whose physics was tested and satisfied. Asking for an orientation
gate on those features raises, naming the capability.

**Orientation is where TG2.1's convention registry pays.** Under `axis_180`, features at 170
and 350 degrees are the same ridge and link; under `direction_360` they are opposed and the
same declared turn rate refuses. The numbers on the records are identical in both runs, and
the wrap period is read from the declaration rather than from the number - the failure TG2.1
predicted would have been wrong by a factor of two with no way to tell which case you were in.

**The cost has no weights, because the gates are the weights.** Each active gate contributes
the square of the fraction of itself the pair used, so an admissible pair costs at most one
per gate. A hand-set trade-off between "how far it moved" and "how much it grew" would be one
more free number deciding how many objects exist.

**Two associators, because they disagree measurably.** Registering a second implementation is
speculative generality unless the difference can be shown (TG1.4's lesson). On a constructed
three-object frame where one object sits one cell from another's true partner, `greedy_nearest`
takes that pair first - it is the cheapest single link in the frame - and the stranded track
then pays nine times as much: total 35 against the Hungarian assignment's 27, with two of the
three identities swapped. The capability flags say `optimal: True` and `optimal: False`, and
`with_capability("optimal", True)` returns exactly one entry.

**A missed frame ends a track, and the clock is required to know that one was missed.** The
first version of `track()` read its clock off the features it was given, and it had two
defects that its own benchmark run exposed before commit. A vortex advected out of a
24-frame sequence after frame 5 produced a `FeatureSet` whose last frame *was* frame 5, so
the track ran to the end of its own evidence and looked complete where the recorded answer
says one death. And an empty frame in the middle vanished entirely, silently bridging exactly
the gap the module's docstring promises never to bridge. `times` - every frame that was
searched - is now a required argument, `track_extractions()` takes it from the
`ExtractionResult`s so a caller cannot supply a clock shorter than the run, and
`empty_frames` is on the receipt. Gap bridging is refused outright: it requires a motion
model good enough to say where the object was while it was invisible, and this tree has
measured no such model. A death followed by a birth is a fact a reader can argue with; a
bridged gap is an assertion that nothing happened in between.

**Everything else is inherited rather than restated.** A `Track` holds a `FeatureSet`, so it
cannot span two domains, two variables, two representations or two clocks - the refusals are
TG2.1's and are not reimplemented here. Velocity is refused on a clock whose units were never
recorded, and on a single sighting, where returning zero would report a measurement that was
never made. A displacement is accumulated link by link rather than taken end to end, because
on a torus a full lap and standing still have the same endpoints. `scale_velocity` is a
least-squares slope of `log2(scale)`, dimensionless by construction because it is built from
ratios of the track's own scales - the one form in which TG2.1 lets a scale leave its domain -
and it refuses a track whose observations do not all carry one. `doubling_time` refuses a
shrinking track rather than returning a negative number that invites reading as a magnitude.

**`4D.tracking` moves from `NOT_YET_RUNNABLE` to PASS.** It has held a recorded answer since
T3.5.17 and raised `NotImplementedError` ever since. On `advected_vortex_sequence`: one track,
one birth, no deaths over 24 frames, worst position error 0.544 cells against a target of one,
and a measured doubling time of 16.31 steps against a recorded 16 - the tracker was given the
fields, the declared axes and an alpha, never the velocity or the growth rate. On
`advected_vortex_periodic_sequence`, the seam-crossing variant added with this slice, the same
numbers: 0.735 cells and 16.38 steps. Started at (100, 100) so the vortex advects out of the
frame, the extractor refuses the clipped blob by rule R13 in exactly the 18 frames the
recorded answer marks unmeasurable, and the tracker reports the one death the recorded answer
now derives. That is defect D59 closed end to end - extractor, tracker and recorded answer on
one declared topology.


### 3.6q Representation-induced feature audit (`src/core/representation.py`, TG2.4, `ed-dev`)

TG2.2 measured the false-positive floor on the raw array: structureless data in, nothing out.
Rule R8 says the other half out loud - *a representation can manufacture a motif* - and a
peak-finder that is honest on an array is not thereby honest on a wavelet band of that array,
because the band is not the array. Every transform in `TRANSFORMS` is a lens, every lens has
its own artefacts, and a discovery pipeline reads the lens rather than the field. This module
points the registered extractor at every plane of every registered representation of a field
that has nothing in it, and requires the answer to be nothing.

**A representation becomes a set of planes.** `REPRESENTATIONS` is a registry keyed by
transform name, plus `raw` for the identity - so the floor TG2.2 measured is an entry here
rather than a separate argument. A plane is a named 2D array with **declared axes**, a declared
decimation in parent cells per sample, and the filter's contaminated margin (rule R13). Six
registered transforms plus the identity produce forty-eight planes of a 128-cell frame;
forty-five of them are testable.

**The null is propagated, never rebuilt.** This is the decision the audit stands on, and it is
a registry with a declared default rather than a constant.

*   `propagate_through_representation` surrogates the **field** and pushes every surrogate
    through the same lens with the same configuration. The null then carries the
    representation, so an artefact present in every realisation raises the cut instead of
    being reported.
*   `randomise_in_representation` surrogates the **coefficient plane**. It is what "calibrate
    where you measure" means and it is wrong: the plane's own spectrum is a product of the
    lens, so randomising phases inside it destroys the artefact the null was supposed to
    account for. Registered so the difference can be measured, and it is: on the same
    scale-free field, propagating reports nothing and rebuilding reports **eight to
    thirty-four features**, almost all of them dual-tree subbands.

The identity check is what makes the default a *generalisation* rather than a second opinion:
propagated through `raw`, the audit's null is `calibrate`'s null to the last bit.

**The audit pays for its own family, and could not afford it at first.** Forty-five planes
tested at a nominal family-wise 0.05 each is forty-five tests, and the answer is read as one
question - *did any lens manufacture a feature?* Measured on the ensemble itself, the
uncorrected procedure rejects on **83-88%** of structureless fields. `FAMILY_CORRECTIONS` is a
third registry: `max_statistic` reads the family-wise rate off the same surrogate ensemble the
cuts come from, leave-one-out, so dependence between bands of one decomposition is measured
rather than assumed; `bonferroni` prices them as strangers; `none` is registered to be measured
against. The corrected level is a shared per-plane rank, which keeps every threshold an order
statistic of its own null - `calibrate`'s no-interpolation rule (§3.6o) applied to a family.

That rule has a price and the price is refused rather than fudged: the strictest cut `n`
surrogates can express is "larger than every null maximum", which over `P` planes still rejects
about `P / n` of the time, so a family-wise 0.05 over forty-five planes **needs about nine
hundred surrogates**. `FamilyTooLargeError` names the family size, the achievable rate and the
required ensemble, and 499 surrogates is a refusal rather than a coarser answer.

**Three ways to earn a pass without looking, all refused.**

*   **A null nothing can violate.** `phase_randomise` preserves the amplitude spectrum to
    machine precision and an FFT magnitude plane *is* the amplitude spectrum, so every
    surrogate has a bit-identical plane, the threshold equals the observation and a strict `>`
    reports nothing on any data whatever. Every plane's null spread is measured relative to
    what it gates; a vacuous one is recorded as `vacuous` and cannot be counted clean.
*   **A plane with nothing left to search.** A dual-tree level-3 subband of a 64-cell frame is
    8 x 8 samples with a six-sample contaminated margin on every side - no valid interior at
    all. Such planes are refused by name and **struck from the family**, so a test with no
    possible outcome cannot make the other tests stricter. The report carries the number of
    cells actually searched: 215,884 across forty-five planes at 128 cells.
*   **A lens nobody looked through.** Coverage is checked against the *transform* registry, not
    against this module's own, so a transform registered without a plane builder is named in
    `uncovered_transforms` and the audit is not clean.

**Not every plane is extractable, and that is a declaration rather than a silence.** The axes
of an FFT magnitude plane are wavenumbers. Declaring them `space` so a spatial peak-finder
would accept them is exactly the error §3.6a exists to prevent, and it would license a
position, a width and a separation in cells that the plane does not have. `fft` and `dct` carry
`axes=None` and a stated reason, their nulls are still measured, and they are reported as
**unauditable rather than clean**. Closing that gap needs a registered extractor with a
spectral shape model, and none exists; the FFT phase is not offered as a plane at all, because
its maximum is a property of the branch cut.

**A defect this slice found in itself, by running it.** The dual-tree lowpass of a three-level
decomposition looks as though it should be decimated by eight, because its six subbands are; it
is decimated by **four**, because the lowpass does not go through the quad-to-complex step that
halves the highpass again. Declared as eight, `plane_field` built a coordinate axis twice as
long as the field, and a feature planted at row 70 of a 128-cell frame came back at **row 137** -
outside the frame it was found in, in the frame's own units, with nothing in the record to say
so. The decimation is now measured from the two shapes, and `representation_planes` refuses any
builder whose declared factor its own array does not have.

**`4E.representation_audit` and `representation_null_field`.** A new null benchmark: scale-free
fBm at 128 cells - correlated, smooth and edge-bearing, which is the material a boundary rule
or a decimation phase turns into a localised artefact, and therefore the harder null. Audited
through every registered representation at 999 surrogates it reports **no feature in any of the
forty-five testable planes**, with the family corrected from a nominal 0.05 to 0.001 per plane
against a measured family-wise rate of 0.037-0.045. The audit's power is measured separately
rather than assumed: the same lenses, the same corrected level and a six-sigma blob produce
nine findings, in the raw field, both approximations, the low-pass half of the hybrid and the
coarse detail bands - so "it found nothing" is a statement about the field.

### 3.6r The declared family is priced before anything enumerates (`src/core/family.py`, TG3.1, `ed-dev`)

Phase G3 opens with the multiplicity work, deliberately before any mining code, because rule
R18 is the phase's binding constraint and the mining is worthless without it. TG2.4 made the
case by hitting it: the audit's own forty-five-plane family was unaffordable at the ensemble
size it started with, and that was a family nobody had budgeted for because nobody had counted
it in advance.

**What existed and why it was not enough.** `GateProtocol.family_size` is
`n_scales * (n_scales - 1) * len(lags)`, which is correct, and which describes exactly one
search shape. A constellation sweep over scales x orientations x lags x representations x
motif configurations is a different shape. A second formula written beside the first is a
second chance to be wrong by a factor nobody notices, and the wrongness would be invisible:
a family size is a bare number, and a bare number cannot fail.

`SearchSpecification` is therefore a declaration made of **terms over named, enumerated
axes**, and each term's combinator both counts and enumerates:

| Combinator | Members | Why a product rule does not cover it |
|---|---|---|
| `product` | Cartesian across its axes | - |
| `ordered_pairs` | ordered distinct pairs from one axis | source->target and target->source are two questions |
| `unordered_pairs` | unordered distinct pairs | a symmetric relation is one test, not two |

`FAMILY_COMBINATORS` is a registry (standard E1) because TG3.3's `k`-feature constellations are
neither a product nor a pair; the acceptance registers unordered triples from the test module
and drives a whole declaration through it. `count` prices a family that may be far too large to
build and `enumerate` builds the one that is not, so a test asserts the two agree for every
registered entry at four axis sizes - two implementations that could drift apart would make a
refusal and a receipt describe different searches with nothing able to detect it.

**The acceptance is the label set, not the count.** `GateProtocol.search_specification()`
renders `source->target@lag`, the same labels `cross_scale_dependency` emits, and the test
compares against a **real sweep** on a three-channel record over the frozen lag family, in
order. Read from `campaigns/t4c6_nz_era5_temperature_850_v1.json` rather than from a literal,
the T4C.6 declaration prices at exactly **36 members and 3,005 surrogates**, agreeing with both
the formula it generalises and the `check_power` result `GateProtocol.validate` already
enforced. 36 agreeing with 36 for two different reasons would have passed a count check.

**`declare()` refuses; `account()` reports.** Mining code calls `declare()`, which raises
`FamilyUnaffordableError` when `check_power` fails, so a pass that cannot reject anything is
refused before it runs instead of run and read as a negative afterwards. The refusal computes
both remedies R18 admits rather than naming them:

*   **preregistered narrowing** - `max_affordable_family` bisects the largest family the
    declared ensemble can still reject one member of (2, 4, 8, 15 and 54 members at 99, 199,
    499, 999 and 4,999 surrogates under BY at 0.05), then reports per axis the largest number
    of values that reaches it. At 4,999 surrogates a five-scale eight-lag sweep (160 members,
    needing 18,097) is fixed by narrowing the lag axis alone, and the test checks that remedy
    by **taking** it - and checks it is tight, since one value more is still refused.
*   **generate/confirm split** (TG3.2), stated as what it is: it does not make the family
    cheaper, it moves the correction onto a confirmatory family frozen before the held-out
    partition is opened.

At 499 surrogates neither axis of that sweep can get there alone, and the refusal says so
rather than sending the reader round a loop.

**A defect in this slice, found by running it.** The narrowing search reported *"scale from 5
to 1 values"* as sufficient at 499 surrogates. `ordered_pairs` over a single value has **no
members**, so the family size was zero and zero is under every ceiling. The remedy was to
empty the search - a pass with nothing in it, and an empty result to read. A term of size zero
is now refused at construction, and reaching a ceiling by emptying a term is not counted as
achievable.

**The programme's boundary is measured here rather than argued.** Phase G3's exit criterion
says that if TG3.1 proves every scientifically interesting family unaffordable, that is the
real boundary and is written up as such. The constellation sweep R18 warns about - 5 scales x
4 orientations x 8 lags x 7 representations - is **4,480 members needing 805,029 surrogates**,
which at the sweep's own `2 x family x n_surrogates` cost is about 7.2e9 estimator evaluations.
That is an output of this module, not a claim about it.

**Option A on admissibility, and it is a refusal rather than a convenience.** A lag below its
support floor is a member that was never testable, and `audit_admissibility` reports how many
there are before acquisition - but it does **not** move `family_size`, and both the declared
size and `correction_unit` stay on the receipt beside the shortfall. Correcting only the
members that survived a screen is correcting a family chosen after looking, which is the
specific move R18 forbids. An audit that empties the family is refused, because a pass with no
possible outcome must not run and report nothing; an exclusion with no stated basis is refused,
because a justification that is not recorded is indistinguishable from one chosen to improve
the answer.

`fingerprint()` is a content hash over the whole declaration - terms, axis values, alpha,
correction and ensemble size - so TG3.2 has exactly one thing to freeze. Nothing here is
evidence: `FamilyAccount.describe()` carries a claim boundary saying that affordability means
the pass is capable of rejecting a member and nothing more.

### 3.6s The held-out partition is opened once, and the declaration is bound before it (`src/core/preregistration.py`, TG3.2, `ed-dev`)

TG3.1 ends by refusing things. The T4C.6 family of 36 needs 3,005 surrogates; the
constellation sweep Phase G3 exists to run needs 805,029. R18 admits exactly two remedies and
`max_affordable_family` computes the first one; this slice is the second, and it is the one
that lets a search stay the size it needs to be.

**What the split does not buy.** Nothing here reduces the number of tests performed. The
generate stage still enumerates 36 and still tests 36, on the training partition, and
`report_generation` refuses to call any of that a result: its p-values are recorded under
`p_values_uncorrected`, its claim boundary says *candidates, not findings*, and a candidate
that is not a declared member of the generate specification is refused outright, because a
candidate nobody enumerated has no family and therefore no correction unit. What the split
buys is that the **confirmatory** family - a subset, small enough to be affordable - is
written down and hashed *before the held-out partition is opened*, so the correction applies
to a family that was fixed without reference to the data it is tested on.

The acceptance is the whole walkthrough rather than any one function, priced off the same
frozen campaign file TG3.1 reads: **36 members are refused at an ensemble of 199, a
four-member subset of them needs 166 and is not**, and the four are corrected at four. That
the distinction is not cosmetic is measured, not asserted - the same four p-values corrected
over the generated 36 reject nothing, and over the frozen four reject one.

**Two ways to cheat, and how each is caught.**

*   **Editing the declaration afterwards.** A `Seal` carries a digest per sealed field and a
    digest over that table, and the two layers fail differently on purpose. An edited *field*
    no longer matches the digest recorded beside it, so `verify` **names it**. An editor who
    also recomputes the digest table breaks the outer digest instead, which is caught but
    cannot say which field moved.
*   **Opening the held-out partition twice.** `HeldOutLedger` records consumption keyed by the
    **partition digest, not by the seal**. That choice is the substance of "once": a ledger
    keyed by seal would let a study write a second, entirely honest seal - correctly hashed,
    correctly frozen, affordable at its own size - and test the same held-out data again. Each
    confirmation would be individually defensible and the pair would be uncorrected, which is
    the arithmetic R18 exists to stop. The refusal names both seals and states the two
    remedies: a partition this study has not touched, or one seal covering both families whose
    combined size is then the correction unit and must clear the TG3.1 gate.

**The boundary, stated rather than blurred.** A content hash detects an unrecorded edit; it
does not prevent one. Anyone who can rewrite the seal file can recompute every digest in it,
and nothing here is signed. `verify` says so in its own return value - *self-consistency only;
this says the local copy was not edited carelessly, and says nothing about whether it was
edited*. `verify_published` is the check that carries weight, and it needs a digest that came
from somewhere the author cannot rewrite - a commit, a registry entry, a preregistration
record. A test proves the gap is real: a wholesale rewrite verifies perfectly against itself
and is caught only against the published digest. Passing no published digest at all is
refused, because a self-consistent seal is not evidence about itself.

**A partition is identified without reading it.** `PartitionIdentity.from_series` takes
channels, frame count, the parent frames the split recorded and the lineage the split wrote,
and touches no measure array - a test hands it a series whose values raise on access, because
a seal written after reading the held-out data is not a preregistration whatever it hashes to.
Provenance goes through `_recordable`, so an array becomes `<ndarray>` rather than being
dropped: the digest is over JSON, and a key with no value reads as a key with no content.

**Refusals that are lineage checks rather than conventions.** Mining on a partition whose own
provenance records it as the held-out one is refused. Freezing against a partition whose
lineage records it as the training split is refused, because confirming on the data the
candidates were selected from measures the selection. A confirmatory member outside the
generated family is refused as a fresh search under the name of a confirmation. A frozen
member with no p-value is refused, because the correction is computed over what was frozen and
an untested member is a claim that it was tested and not reported; a p-value for a member the
seal does not contain is refused as mining on held-out data whatever it is called.

**Order matters, and is tested.** The seal is verified, the presented partition is checked
against the one the seal named, and the p-value set is checked to be exactly the frozen label
set - and only then is the partition spent. So a refusal never consumes the held-out data and
a completed confirmation always does. A confirmation that rejects nothing produces a receipt
like any other, since a null is the expected outcome of an honest split and must not look like
a failure.

**A defect in this slice, found by running it.** `PartitionIdentity` validated its frame
count, its channel count and its name, but never checked `channel_labels` against
`n_channels`. A record claiming three channels while carrying two labels describes two
different geometries, and the seal would bind both - so a confirmation on the partition the
labels name would be indistinguishable from one on the partition the count names. The
mismatch is now refused at construction.

### 3.6t Constellations are attributed graphs, and a relation divides by what the features carry (`src/core/constellation.py`, TG3.3, `ed-dev`)

TG2.1 settled what a feature is, TG2.2 who decides one is there and TG2.3 which feature in one
frame is the same object as one in the next. This module settles what a *configuration* is:
`k` features plus the typed relations between them, described so the description survives
leaving its domain.

**The whole problem is one contrast already present in TG2.1.** `scale_ratio_to` is permitted
across a domain boundary and `separation_to` is refused across one, because a ratio of two
lengths is a number about the world and a distance in cells is a number about an array. So a
relation is not "a quantity computed from two features"; it is that quantity *divided by
something the two features carry themselves*. Forty cells at a scale of six cells and twelve
hundred metres at a scale of a hundred and eighty metres are the same relation. Forty and
twelve hundred are not, and the module contains no way to say otherwise.

**Relations are computed within a domain and compared across one.** A constellation is drawn
from a single dataset - a `FeatureSet` is already one domain, one dataset, one variable and one
representation - so the coordinates the relations read are coordinates in a space that exists.
What crosses the boundary afterwards is the `AttributedGraph`, split into `attributes` (the
dimensionless view, which `matches` reads) and `carried` (domain, dataset, variable,
representation, units, which nothing in `matches` can reach). As in TG2.1's
`structural_signature`, the comparable view is **built rather than filtered**: putting a
carried field back into a comparison is a visible edit to `_node_attributes`, not the
consequence of a key someone forgot to remove. Absolute orientation does not appear there
either - an angle measured from the grid's north is a property of how the array was stored -
and it re-enters as *differences* on the edges, which is where it means something.

**Eight relations, and sorting them by what each needs is a finding rather than a taxonomy.**
`succession` needs nothing but a shared clock, because an ordering is dimensionless already:
what came first came first in frames and in hours alike. `co_occurrence` needs a temporal
scale, because simultaneity is a *tolerance*, and a tolerance in seconds is not a statement
another domain can read. `distance` needs a spatial scale, `containment` an extent,
`temporal_lag` a temporal scale, and `direction` and `convergence` an orientation - which the
only extractor TG2.2 registers declares it does not report.

**So two of the eight cannot be measured at all today, and they register anyway and refuse by
name.** This is the design ruling of the slice. It is TG2.3's rule about gates carried into
relations: a relation that treated an absent quantity as "no evidence against" would appear in
a receipt, constrain nothing, and be indistinguishable from one that was doing work.
`convergence` carries a second refusal of the same kind - it is refused under an *undirected*
orientation convention, because an axis does not point, and a ridge at 170 degrees and one at
350 are the same orientation.

**`relation_axis` is how the registry reaches TG3.1.** A family priced over eight relations
when three are measurable has declared five tests that could not have happened. That is not a
conservative rounding in R18's direction - it is a receipt naming tests that never ran - so a
declaration is built from `measurable_relations` rather than from `RELATIONS.names()`. Three
measurable relations give an unordered-pair family of 3; all eight give 28.

**Matching is exhaustive or refused.** Two graphs describe the same configuration if some
correspondence of their nodes makes every attribute and every relation agree, which is `k!`
comparisons; above `MAX_MATCH_NODES` (8) this refuses rather than falling back on a greedy
assignment, because an approximate match that returns `True` is a claim. The `MatchReport`
names the correspondence, the relations compared and **both carried records**, so a reader can
see that two graphs matched while describing an amplitude field and a temperature field (R19),
and can see whether the two came from different representations (R8) - stated, never compared.

**What this slice deliberately does not do.** It does not match a configuration under rotation,
rescaling and translation; that is TG3.4, whose whole acceptance is `4E.invariance` moving off
`NOT_YET_RUNNABLE`, and claiming it here would leave that slice nothing to prove. A rotated
triangle does match on the relations measured here, because distance and relative scale are
rotation-invariant on their own - the relations that would fail are the two needing an
orientation nothing reports yet. It does not mine for repeated configurations either; TG3.5
does that, through TG3.1's declared family and TG3.2's split.

**A defect in this slice, found by running it.** A graph built with `strict=False` records a
refused relation instead of propagating it, and `matches` compared the relations the graph
*declared* rather than the ones it carried. The result was that such a graph did not match
**itself**, and the stated reason blamed the geometry - "no correspondence makes every
relation agree" - for what was a hole in the record. A missing relation is now a refusal
naming the edge and the reason it was not measured, and the caller narrows `relations` to what
was actually measured.

### 3.6u Invariant matching, and the tolerance that is measured rather than chosen (`src/core/invariance.py`, TG3.4, `ed-dev`)

The `4E.invariance` gate has been defined since T3.5.17 and reported `NOT_YET_RUNNABLE` ever
since, because the benchmark could present the planted triangle rotated, rescaled and
translated but nothing existed to be asked whether it was the same triangle. It now reports
**PASS**, and it was the last pending gate in the suite: the benchmark run was 20 PASS, 0 FAIL,
0 NOT_YET_RUNNABLE (24 since TG3.5 and TG4.1 each added two more).

**A matcher here is not an algorithm, it is a choice of what to measure.** Every entry in
`MATCHERS` is a function from features to an `AttributedGraph`, and the comparison is always
TG3.3's exhaustive `AttributedGraph.matches`. A registered matcher therefore cannot fail by
being careless about the node correspondence, because it does not get to decide the
correspondence; it can only fail by keying on the wrong quantity, which is the failure the
gate is about. Two are registered. `relative_geometry` divides each separation by the
geometric mean of every separation in the configuration and declares invariance to all three
transforms. `absolute_position` keys on coordinates and raw separations, declares invariance
to nothing, and is the position-memorising control the roadmap asked to see fail.

**A dimensionless relation is only as invariant as the thing it divided by.** TG3.3's
`distance` is a separation over the geometric mean of the two features' *estimated* spatial
scales. It is dimensionless, so it looked scale-invariant, and on exact synthetic geometry it
is - the relation does not move at all under a rescaling when the scale is known exactly. On
the benchmark it moves by 4.8% at `scale_factor=3` against a 3.3% noise floor, because the
extractor's scale estimate drifts from about +1.4% at a 6-cell sigma to about -2.6% at 18
cells while the separation itself is recovered to better than 1%. The whole of the drift lands
in the quotient. Neither the relation nor the extractor is at fault; what is measured is that
the two compose badly, and dividing by a *modelled* quantity imports that quantity's bias into
the relation that divided by it.

**The invariant that survives divides one measurement by another of the same kind.** A
separation over a separation cancels its units exactly and involves no estimated scale at all,
and it reproduces to within the noise across all three transforms. The price is that it needs
three features: with two there is one edge, its ratio to its own mean is 1, and a matcher that
returns the same signature for every configuration reports a discovery on every pair it is
shown. `build_signature` refuses a pair for `relative_geometry` on exactly that ground - which
is also why TG3.3 divided by the modelled scale, and was right to for two features.

**`scale_normalised` is public and deliberately unregistered.** It is TG3.3's `distance`
comparison, unchanged, and TG3.5 will want it, because it is the one that works on a pair and
crosses a domain boundary. It is not in `MATCHERS` because a registry entry carries a
declaration and this one has no declaration that can be demonstrated: measured directly it is
not rescaling-invariant, but tested the way `measure_invariance` tests, over seven rescalings,
the evidence comes to `p = 0.016` and does not survive the family correction. True and unproven
at once is not a state a declaration can hold, so the function stays public and the registry
stays honest.

**The extractor does not return its features in a stable order.** The order follows which blob
happened to be brightest, so it shuffles between noise realisations. The first version of the
noise floor compared replicate node 0 with replicate node 0 and measured that shuffle: it put
`absolute_position`'s floor at 0.27, wide enough that a configuration rotated by 37 degrees
matched its own memorised pixel coordinates. `best_deviation` minimises over the same `k!`
correspondences a match searches, which brought the floor to 0.026.

**A noise floor is an operating point; invariance needs a test.** `calibrate_match_tolerance`
re-measures the same configuration under different noise and reports the largest disagreement
it produced. There is no safety factor, because a safety factor is a free parameter and a free
parameter is where a tuned result hides - and the false-rejection rate that comes with it,
about `1/(n_null + 1)`, is written into the tolerance's own basis rather than left implicit.
But thresholding a family of presentations at that value would reject a genuinely invariant
matcher about one time in five, which is R18's subject exactly: thirteen chances at a
one-in-sixty-seven event. `measure_invariance` instead compares the presentations against the
whole null distribution with a one-sided rank test, on the reasoning that a presentation and a
replicate measure the same thing whenever the matcher is invariant, and divides alpha across
the tests actually run - one per transform, plus the scale recovery.

**A test that could not have failed does not confer anything.** `InvarianceTest.power_floor` is
the smallest p-value the comparison could return, and a test whose floor sits above its own
alpha is marked `vacuous` and grants no invariance. This is TG2.4's rule about a plane whose
null nothing could exceed, carried into a rank test: two replicates give one null value, and
one presentation against one null value cannot report a p-value below 0.5 however wrong the
matcher is.

**A declared capability is measured, not trusted.** `audit_declared_invariance` runs every
registered matcher against every presentation and compares what it survived with what it
declared. Overclaiming fails, and so does understating, for the reason TG3.1's relation axis
gives: a receipt naming a property the run did not have is wrong in either direction, and a
matcher that under-declares makes a caller reach for a heavier one it did not need. The
position-memorising control therefore fails by the general rule applied to it rather than by a
special case written for it, and a future matcher that overclaims will fail the same way with
no edit to the gate.

**Only a single transform can attribute a result.** A presentation combining rotation,
translation and rescaling is measured and reported but confers no invariance on its own,
because a combined presentation that matched could have matched because two errors cancelled,
and a cancellation is not a property.

**The scale-aware reading.** The benchmark's known answer has recorded
`scale_ratio_vs_reference` since T3.5.17 and nothing read it, which is the same species of
defect as a relation that constrains nothing. `recover_scale_ratio` takes the `MatchReport`
rather than a flag, so it structurally cannot run before a decision has been made: the
separation scale lives in `carried`, nothing in `matches` can read it, and this reads it only
to describe a match already decided without it (R19). It is recovered from the separations
rather than from the estimated scales, and across the benchmark's sixfold range it returns
`scale_factor` to better than 1% where the scale estimate drifts by 6%. Across a unit boundary
it refuses by name and says why - a configuration in cells is not a number of times bigger than
one in metres, and the match that crossed that boundary crossed it precisely because it had
divided the units out. The recovery is judged against its *own* noise floor, measured by
`scale_recovery_null`; the first version of the gate judged it against the shape's floor, which
is the noise of a different number, and failed a correct matcher on it.

**What the gate refuses to accept as a pass**, beyond the matchers' declarations: a noise floor
built from fewer replicates than the benchmark's known answer requires, because that floor is a
maximum and a maximum over few samples is biased low in the direction that rejects a matcher
which is invariant; any vacuous test; and a run in which no scale ratio was recovered, since a
matcher blind to scale and a matcher that measures scale and states it both match a rescaled
triangle, and only one of them produces the number. The minima live in
`truth_planted_configuration` rather than in the check, because a gate that decides how hard to
look at the moment it looks can always decide to look less hard.

**What this slice does not do.** It does not recover a scale ratio across a domain boundary; a
ratio of separations in cells to separations in metres is not a number, and TG3.5 is where a
cross-domain configuration will need one, if it needs one at all. It does not mine for repeated
configurations - that is TG3.5, through TG3.1's declared family and TG3.2's split. And the
gate's transformed presentations are built from a fixed label rather than from the run's root
seed, so varying the root seed varies the reference and not the presentations; the gate has
been confirmed to pass at four root seeds, which is four references against one presentation
set rather than four independent runs.

### 3.6v Recurring motifs, and the accounting that counting honestly requires (`src/core/motif.py`, TG3.5, `ed-dev`)

A motif is a configuration that occurs more than once, and everything hard about mining for
one follows from that sentence being about *counting*. Two gates were added with the module
and both pass: `4E.motif_recovery` mines six training scenes, freezes what it found before
the held-out scenes exist and confirms one motif on a partition it was not mined from;
`4E.motif_null` runs the identical pass over scenes with nothing planted in them and confirms
nothing. Per this tree's own Definition of Done the second is the load-bearing one.

**The family is the number of configurations looked at, and no ensemble pays for it.** Mining
`k`-feature configurations over `s` scenes of `n` features each examines `s * C(n, k)` of
them, and every one is a chance for a repeat to look surprising. The benchmark's six scenes of
six features at `k = 3` is 120 members - which TG3.1 prices at 12,885 surrogates before its
ceiling allows a single member to be rejected, and eight scenes of twelve features would be
1,760 members and 283,380 surrogates. TG3.2's generate/confirm split is therefore not an
optimisation in this slice, it is the only affordable shape, and
`motif_search_specification` computes that refusal rather than asserting it. The gate refuses a pass if its own generate family ever
becomes affordable in one stage, because then the benchmark would have stopped testing the
thing it exists to test.

**The priced family and the pass are checked to be the same search, member for member.** Two
terms multiplied - which scene, and which `size`-subset of its features - and `mine` compares
`SearchSpecification.labels()` against the labels the enumeration actually emits, in order. A
count check would pass on two different searches that happened to be the same size. The
`k`-subset combinators are registered *from this module* rather than added to `family.py`,
which is the acceptance TG3.1's own test wrote down when it said the next search shape would
be a registration and not an edit.

**A motif is an exemplar, not a cluster.** Matching under a tolerance is reflexive and
symmetric but not transitive: A matches B and B matches C without A matching C, and
single-linkage clustering silently promotes that chain into one motif with a support of
three. A candidate is therefore one occurrence together with the occurrences that match
*that* occurrence - a star, not a chain - and `MiningResult.intransitive_pairs` counts how
often the difference would have mattered, because a design decision whose consequence is
never measured is a preference.

**Overlapping occurrences within a scene are one piece of evidence.** Two triangles in one
scene sharing two of their three features are very nearly the same observation, so support is
the number of *scenes* a motif occurs in and the occurrence count is carried beside it as
description. `support_of` stops at the first match within a scene, so the null is computed
under the same rule as the observation.

**Ranking by raw count prefers the promiscuous.** Support is capped at the number of scenes
and saturates, so the ties at the cap decide the ranking - and they have to be broken by
*fewer* occurrences, not more. A shape matching three configurations per scene is a looser
shape than one matching exactly one, not a stronger finding, and the key written the obvious
way round fills the top of the ranking with shapes that match everything.

**What is frozen, and why it is not simply the ceiling.** `choose_candidates` takes the
shapes that recurred most on train, deduplicated, capped at TG3.1's ceiling. Deduplicated
because a motif that genuinely recurs in six scenes enters the ranking six times, once under
each scene's label, and six names for one hypothesis is a correction unit of six paid for one
test. Only the top support tier, because a family topped up to the ceiling with candidates
the mining pass did not favour spends correction power on members nobody proposed - measured
on this benchmark, it moved the planted motif's corrected `q` from 0.005 to 0.042, the
difference between a clear result and a marginal one. Selecting on train support is
selection, and it is the selection the generate stage exists to perform: it happens before
the seal, on the partition that was mined, and nothing it produces is a claim.

**A surrogate has to be drawable by the process that produced the data.** The null places the
same features, with the same magnitudes and scales and provenance, at random positions inside
the extent where features were actually seen. Without a minimum separation two of them can
land closer than the extractor could have resolved them, and the triangles they make are
near-degenerate slivers the pipeline could never have returned; the null then fills with
shapes unlike anything the data contains, supports the motif *less* often than a real
arrangement would, and makes the observed support look more surprising than it is. Measured
over five seeds, dropping the constraint roughly halves the p-value - from 0.20-0.25 to
0.07-0.14. The constraint is read off the data (`observed_minimum_separation`) rather than
declared, and an arrangement that cannot satisfy it is refused rather than quietly relaxed.

**Invariance is necessary and nowhere near sufficient.** `always_matches` is a matcher whose
signature is a constant. It is exactly invariant to rotation, translation and rescaling -
genuinely, not by a trick - and `test_motif.py` registers it into TG3.4's `MATCHERS` and
shows that `audit_declared_invariance` calls it honest. It is also useless: it reports every
configuration as a repeat of every other. A gate built on invariance alone cannot see the
difference, and what separates the two is the null, where the constant matcher supports every
motif in every surrogate scene and its p-value is exactly 1. It is public and unregistered,
for the opposite reason to `scale_normalised`: that one's declaration cannot be demonstrated,
this one's is demonstrably true and worth nothing.

**A defect in TG3.4 that TG3.5 found by using it.** `recover_scale_ratio` read
`separation_geometric_mean` straight out of `carried` and raised `KeyError` for a matcher that
records no length. It now refuses by name - a matcher can recognise two configurations as the
same shape without ever having measured how big either of them was - and `measure_invariance`
skips the scale-recovery test for such a matcher rather than failing inside it.

**What the gates refuse to accept as a pass**, beyond their own expected outcome: a tolerance
calibrated from fewer replicates than the known answer requires; partitions of the wrong size;
an enumeration that examined a different number of configurations than the family priced; a
generate family that turned out to be affordable in one stage; a run that carried no candidate
forward; a frozen motif tested by an ensemble whose smallest possible p-value sits above the
corrected alpha (TG2.4's `vacuous` rule); and a correction computed over a different number of
members than were frozen. The null gate shares all of them, deliberately: a null that reported
nothing because it proposed nothing is a pass obtained by not looking.

**A replicate must be the same configuration, not the same frame.** The tolerance is
calibrated from six extractions of a *three-feature* field, not from six-feature scenes
narrowed to their brightest three. Magnitude ordering moves between noise realisations, so
"the brightest three" is a different configuration each time it is taken; measured here, that
mistake put the tolerance at 0.41 instead of 0.0083 - fifty times too wide, and wide enough
that every triangle matched every other. It is the same defect TG3.4 found in its node
ordering, in a new place.

**What this slice does not do.** It does not serialise a motif's definition. The seal freezes
the exemplar's *label* and the definition travels beside it in memory, which is enough for a
confirmation inside one run and not enough for TG5.1, where a frozen motif has to survive the
process that made it. It does not mine across a domain boundary. And, like TG3.4's, the
gates' scenes are built from fixed labels rather than from the run's root seed: only the first
training scene's noise varies with the seed, and the extractor's sub-cell localisation is
stable enough that it does not change the outcome. Both gates have been confirmed to pass at
four root seeds, which is four runs of one scene set rather than four independent ones.

### 3.6w Recovering a relationship nobody pointed at (`src/core/precedence.py`, TG4.1, `ed-dev`)

Phase 4C's central claim is that fine-scale activity at `t` precedes coarse-scale activity at
`t + lag`, and `src/benchmarks/sequences.py` has carried a benchmark for it since T4C.3. That
benchmark's check reads `driver_level` and `driven_level` **out of the known answer** and then
takes an `argmax` over lag. It recovers the injected lag, which is worth knowing, and it is not
the claim: a search told which two bands to look at is a search of one band pair, and a study
that could only be run by someone who already knew the answer has not demonstrated recovery.
This module runs the same recovery without being told, and two gates were added with it. Both
pass. `4F.precedence_recovery` confirms one relationship on a partition it was not mined from;
`4F.precedence_null` runs the identical pass over a record whose two modulations were drawn
independently - same band structure, same marginals, same memory, no alignment - and confirms
nothing. Per this tree's own Definition of Done the second is the load-bearing one.

**Being told where to look is a family of one.** Declared honestly, the question over `L` bands
and `|lags|` admissible lags is `L(L-1) * |lags|` members: ordered pairs, because *fine leads
coarse* and *coarse leads fine* are computed from the same two series and are two hypotheses.
Four bands and twelve lags is 144 members, which TG3.1 prices at **15,985 surrogates** before
one of them could be rejected; eight bands and twenty-four lags is 1,344 members and 209,153.
So TG3.2's generate/confirm split is again not an optimisation but the only affordable shape,
and both gates refuse a pass if their own generate family ever becomes affordable in one stage.

**The lag is chosen by the data, so the null has to be over the choice.** The maximum of twelve
lagged correlations is not one correlation - its null is the distribution of a maximum - and
this is the largest effect in the slice. Measured on records with *nothing planted in them*,
the winner of the sweep tested against a single-lag null is significant at 0.05 in **six runs
of six**, and against the null of the same maximum in **one of six**. `precedence_p_value`
takes `selected_over` for that. It is necessary and nowhere near sufficient, because it prices
the choice of lag and not the choice of band pair; what pays for the rest is the corrected
confirmation on a partition the study did not select on, where twenty null records produce **no
confirmation at all**. And the confirmation is affordable precisely because the lag was frozen
on train: a frozen lag is not a selection, so the held-out null is over a single statistic.

**A surrogate must keep the autocorrelation.** The honest surrogate is a circular shift of the
driver band's series - every value, the whole autocorrelation function and the marginal
distribution preserved exactly, and only the alignment destroyed. A shuffle destroys the memory
too, and two red series are far more aligned at a random offset than two white ones, so a
shuffled null sits closer to zero than the truth. Measured over twelve null records at the same
lag: at `phi = 0` the two agree (median p 0.64 and 0.65), at `phi = 0.7` the shuffle's median p
has fallen to 0.34 against 0.55, and at `phi = 0.9` it rejects three times in twelve where the
shift rejects none, median p 0.19 against 0.59. The error grows with exactly the quantity the
surrogate discarded. `permuted_series` is public and unregistered so that this is measurable.

**The decomposition relates bands to each other before the world does.** This was found by
running the null and watching it confirm a relationship at q = 0.0075 on a record with nothing
in it. A redundant wavelet does not return four independent bands, it returns a smear: measured
on a control built by the same pipeline with **no temporal structure at all**, levels 1 and 2
correlate at **0.99 within a frame** whatever is planted, levels 3 and 4 at 0.78 to 0.84, and
levels 1 and 3 at 0.49 to 0.53, while nothing survives one frame - the largest cross-band
correlation at any admissible lag is 0.06 to 0.18. So the basis couples two bands when it
relates them within a frame more strongly than it relates anything across one, the limit is
read off the control rather than chosen, and of twelve ordered pairs four survive. The
narrowing is admissible under R18 for one reason: it is derived from a control record and not
from the record under test, so it is fixed before the study's data is read. The one lag the
family may never contain - zero - is exactly the lag that identifies a band seen twice.

**Frames are not samples, and a lag costs data.** Every candidate carries both p-values, naive
and ESS-corrected, because the ratio is how much serial dependence was inflating it (rule R12);
on the benchmark's own record the effective pairs are 262 of 395 frames. The same arithmetic
decides admissibility: a lag of `k` is tested on `n - k` pairs, a family whose largest lag
leaves fewer than eight effective pairs is refused rather than quietly tested at a power nobody
declared, and a record slow enough - `AR(1)` at `phi = 0.98`, memory measured anywhere from 26
to 115 frames on 360 - has *no* admissible lag and produces a refusal instead of a weak result.

**One relationship enters the family once per lag it survives at.** A driver with a memory of
several frames correlates with its target at `k - 1`, `k` and `k + 1`, so the top of the
ranking is one hypothesis wearing three labels and freezing all three is a correction unit of
three paid for one test. `deduplicate_candidates` keeps the best lag per ordered pair, and what
is frozen is then the members the training partition **could not tell apart** from the winner -
within one standard error on Fisher's z scale, which is the middle ground between pretending to
a precision the record does not have and TG3.5's measured cost of filling the family to the
ceiling. On the planted record that rule freezes one member; on a null record it freezes two to
four, because when nothing stands out nothing is distinguished, and none of them confirm.

**The embargo is hygiene, and it is not what protects the confirmation.** `split_with_embargo`
drops frames between the partitions and `freeze_precedence` refuses to seal one cut closer than
the record's own memory. Then the effect was measured, and on twenty records per setting whose
bands are independent but slow it is not there: at `phi = 0.9` the train winner reaches the
held-out partition at a median |r| of 0.150 without an embargo and 0.152 with one; at
`phi = 0.95`, 0.161 against 0.188. The held-out test is a surrogate test drawn from the
held-out record, so that record's autocorrelation is already in its null - the protection comes
from the surrogate, not from the gap. The refusal stays as a declared constraint, recorded here
as one whose benefit this tree has not been able to measure rather than as a repair.

**What the gates refuse to accept as a pass**, beyond their own expected outcome: an admissible
lag set that does not contain the planted lag; a generate family affordable in one stage; a
sweep that examined a different number of members than the family priced; a basis control that
excluded no pair at all, since a control that measured nothing has not been run; a run that
carried no candidate forward; a frozen member whose ensemble could not have rejected it (TG2.4's
`vacuous` rule); a correction computed over a different number of members than were frozen; and
more frozen members than the ensemble can reject one of. The null gate shares every one of them.

**What this slice does not do.** It measures precedence as a lagged correlation, which is a
statement about alignment and not about mechanism: a common driver of both bands would produce
the same evidence, and separating that needs a conditional statistic this module does not have.
It searches one record at a time and does not cross a domain boundary. Its basis control is
deliberately *not* reseeded with the run - which bands a wavelet can separate is a property of
the transform, not of the record - so the family is stable across seeds by construction; the
records under test do vary, and both gates have been confirmed at five root seeds.

### 3.8 Grid Geometry and Metric-Aware Operators (`src/physical_core/grid.py`, `operators.py`)

Added in T3.5.13 (defect D13, standard E3). `GridSpec` is the physical metric attached to
every `PhysicalField` - `pixel`, `cartesian` or `latlon`, and since TG1.2 whatever else is
registered (section 3.6i) - and the default is deliberately `pixel` rather than `None`, so
"lengths here are array indices" is a recorded fact that travels with the data rather than an
unexamined assumption.

*   Exact spherical cell areas (`R^2 dlon (sin(lat_n) - sin(lat_s))`), validated by summing a
    global grid to `4 pi R^2` to **1.2e-16**.
*   Physical wavenumber axes in `rad_per_m` / `cycles_per_km` / etc., with `legacy_pixel`
    retained so pre-D13 numbers stay reproducible and auditable.
*   `anisotropy()` reports the aspect ratio and the spread of the zonal metric; a 32x32 ERA5
    patch at 60 degrees north has aspect **1.79** with `dx` varying **20%** across it.
*   `resampled()` / `subset()` propagate the grid through interpolation and crops, and a
    grid that does not match its field's shape is refused.

`operators.py` supplies the metric-aware `gradient` (validated against
`cos(lon)/(R cos(lat))` to 3.2e-6 at four latitudes, and confirmed second-order by grid
refinement), a **Laplace-Beltrami** spherical Laplacian in staggered flux form (validated
against three harmonic eigenvalues), and area-weighted domain statistics that return both the
weighted and unweighted values so the size of the weighting effect stays visible.

### 3.9 Radial Spectra and Power-Law Fitting (`src/analysis_engine/spectra.py`)

Added in T3.5.13. Isotropically-averaged spectra on a **physical** wavenumber axis, with the
1D energy convention `E(k)` and the 2D density `S(k)` both computed and each labelled -
`E = 2 pi k S` in 2D, and conflating them was defect **D26**. The Parseval identity holds to
1.0000000000. `fit_power_law` performs count-weighted least squares and returns a **standard
error** with every exponent (rule R2), naming a turbulence regime only when the fit is tight
*and* the reference exponent is within tolerance. Annulus reduction is `torch.bincount`
(part of D17), asserted numerically identical to a reference Python loop.

### 3.10 Climatology Removal (`src/analysis_engine/climatology.py`)

Added in T3.5.17 because a benchmark failure demanded it. Harmonic regression on the diurnal
and annual periods, which removes a *partial* annual cycle from a *partial* year - a
time-of-day bin climatology cannot, and left **15.5%** of the variance on a cycles-only
sequence against **0.017%** for the harmonic fit. Fitting is split-aware (`fit_mask`, rule
R6). The solve is deliberately deterministic: columns are normalised and an SVD pseudo-inverse
with an explicit rank tolerance replaces `torch.linalg.lstsq`, whose driver made a rank
decision that changed with prior BLAS state (defect **D30**).

### 3.6x The five refusals (`src/core/refusal.py`, TG4.2, `ed-dev`)

The proposal's validation strategy names five things this engine must be able to do, and four
of them are refusals: recover a planted relationship without being told where it is, **reject**
a convincing but artificial correlation, **distinguish** independent observations from
autocorrelated repetitions, **avoid** a motif that belongs to the representation rather than to
the world, and **report nothing** when there is nothing. TG4.1 built the recovery and one
refusal. This slice built the other three, and the way it built them is the point: each began
by constructing a record on which the pipeline TG4.1 shipped **produced the false discovery**,
and only then declaring the rule that refuses it. Two of the three traps were holes. That is
recorded here because a slice that only re-confirmed the previous slice would be worth nothing.

The register is machinery rather than a list in a document. `REFUSALS` names all five, the gate
that carries each and the benchmark that gate runs on, and `refusal_coverage` checks that
against the live benchmark registry - five benchmarks, at least three of them nulls, every
named gate actually declared by the benchmark that claims it. TG4.2's acceptance criterion is
therefore computed, and a refusal whose benchmark is deleted or renamed stops being a refusal
that day. All five benchmarks are the **same builder at five settings**, which is what makes
the four nulls controls rather than separate experiments, and every one of the three new gates
runs its own pass twice: guarded, which is the study, and unguarded, which is the trap measured
on the same record. A gate whose trap stops springing has stopped testing its refusal, and says
so.

**A single band, read through a redundant transform, manufactures four relationships.** The
trap is one modulation exciting one spatial band, with no second process anywhere in the
record. A stationary wavelet spreads that band's energy across every level, so every level's
series is a smeared copy of one process, and because the process has memory the copies stay
correlated at a lag. TG4.1's basis control does not catch it, and the reason is worth stating:
that control excites both bands independently and asks which *pairs* the transform confuses, so
it reports levels 1 and 4 as separable at 0.023 - which they are, when both are excited. Run
the TG4.1 pipeline unchanged on the single-band record and it confirms **all four admissible
members at q = 0.0104, at every one of five root seeds**, each at the shortest admissible lag.

**The rule that refuses it is a ceiling, not a threshold.** Instantaneous leakage of one
process into two bands produces `r(k) = r(0) * rho(k)`, where `rho` is the process's own
autocorrelation: maximal at zero, decaying from there, and never larger than the *simultaneous*
correlation. So a lead is claimable only when it is **stronger than the same-frame relationship
it might be a smeared copy of**. There is nothing to tune - `rho <= 1` makes `r(0)` the
supremum of everything a leak can produce - and the lag the family may never contain becomes
the reference every member is measured against. On the single-band record **0 of 48 members
survive** at all five seeds, with the strongest refused member at |r| = 0.74 to 0.90 against a
same-frame 0.93 to 0.98. That is a *refusal of the family*, which `EverythingRefusedError`
reports as a distinct outcome from "nothing was significant", and the benchmark's known answer
demands that outcome rather than silence. On the planted record 20 to 44 of the 48 members
survive and the true one confirms at q = 0.0050 at all five seeds, unchanged: the ceiling costs
a real finding nothing, because a real lead is stronger at its lag than at zero.

This is selection on the training partition and it happens there only, before the seal. It does
not narrow the declared family - 48 members are priced and 48 are measured - and it never runs
on the held-out partition, where the lag is frozen and no choice remains.

**A shared cycle defeats the surrogate that was supposed to be enough.** The second trap is two
independently modulated bands that both carry one deterministic cycle, the fine band's crest
arriving three frames before the coarse band's. Nothing is coupled. The circular shift ought to
be the classical defence, because a shifted copy of a periodic series is still periodic - and
it is not enough: the shift moves the phase, most shifts misalign the crests, and the observed
alignment still looks surprising. Measured over five root seeds, the unguarded pipeline
**confirms three or four of the four members at every one of them**, at q <= 0.0104, with the
strongest member carrying a naive p-value between 1.7e-71 and 6.4e-54 and an ESS-corrected one
between 3.6e-13 and 2.8e-10. Neither rule R12 nor the surrogate refuses this.

**The calendar is metadata, not a discovery.** What refuses it is rule R11 lifted into this
pipeline. The cycles a record is exposed to are computed from its own cadence - a day is
twelve frames at two-hourly sampling whatever the values do - fitted as harmonics **on the
training partition** and subtracted from both partitions on one shared clock, and
`carry_forward` refuses to hand a single candidate to a freeze from a record that still has its
calendar in it. There is no path from a swept record to a frozen claim that does not pass
through that door. With the removal in place the same five seeds confirm nothing, the smallest
corrected q being 0.105.

One harmonic per period, and the count was measured rather than assumed. The calendar does not
reach a band's energy series as a sinusoid - it modulates an amplitude that is read as an
energy in logs - so more harmonics looked likely to help. On a longer record, 384 frames with a
24-frame cycle, the residual confirms a lag-1 relationship at one root seed in five, and it
does so at one, two and three harmonics alike; two and three removed slightly *less* of the
spurious lead than one did. That is what says the leak is a false positive of a 0.05 test
rather than an unremoved cycle, and it is recorded here rather than tuned away: it is the one
place this refusal has been seen to fail.

Detecting the period from the data instead was tried and measured, and it does not work at this
record length. Fitting one harmonic per candidate period on the first half of the training
partition and scoring it by the variance it removes from the second half - the strictest of
four variants tried, over ten seeds and five record types - separates a real cycle at 0.13 to
0.58 from a red-noise fluke at up to 0.29. The distributions overlap, so a detector would
either miss real calendars or invent them. The clock overlaps with nothing.

**What that leaves undefended is said out loud.** A periodic confound at a period the clock
does not name is refused by nothing in this module, and the measurement above is exactly the
measurement of how badly it would go: the engine confirms it, repeatably. What is refused here
is the *calendar*, which is the confound this programme will actually meet on ERA5, and not
periodicity in general.

**The third trap was already refused, and the benchmark exists to measure by how much.** Two
independent bands at `phi = 0.95` over 288 frames confirm nothing at all five seeds, while 22
to 48 of the 48 members carry a naive p-value below 0.05 and as many as 20 carry an
ESS-corrected one below 0.05, on a record whose effective pairs are 8 to 19 per cent of its
frames. On a record drawn the same way the winner has been seen at `level_2>level_4@5` - the
same label, at the same lag, that the planted benchmark confirms. What refuses it is the held-out surrogate, and
`naive_versus_effective` reports both counts because a corrected count with no naive count
beside it does not show that the correction did anything.

**What this slice does not do.** It adds no new statistic: the leakage ceiling is a comparison
between two correlations the sweep already computes, and the calendar removal is a harmonic
regression. It does not separate a common driver from a direct relationship - the conditional
statistic that would is still absent, and TG4.1 said so too. And the leakage ceiling is a
sufficient condition for refusal, not a necessary one: it refuses everything a leak could have
made, which will also refuse a genuine relationship that happens to be strongest within a
frame. On the records this tree has, that costs nothing measurable; on a record where the true
coupling is instantaneous, it would cost the finding, and the honest reading of a refused
family is "this record cannot carry this question", not "there is nothing here".

### 3.6y A cross-domain relationship keeps both domains (`src/core/cross_domain.py`, TG4.3, `ed-dev`)

TG4.3 asks for two synthetic domains with different units, semantics and cadences, one carrying
information about the other's later state. The word *domain* is load-bearing. Concatenating two
arrays would let TG4.1 recover a lag, but would discard exactly the facts rule R19 says may never
be discarded: what each operand means, its native units and clock, its source and licence, and
the rule that licenses a precedence claim in that domain. `DomainChannel` and
`DomainTimeSeries` retain those facts; `align_exact` brings the records onto a common clock
without making their quantities common.

**The common clock contains observations, not estimates.** Alignment is the exact timestamp
intersection. It performs no interpolation and records the parent clock hash, the common
cadence, and how many native observations each domain retained and discarded. The benchmark's
hourly thermal record has 1,080 observations and its three-hourly demand record has 360; the
aligned record has the 360 timestamps both actually observed, and says that 720 thermal
observations were not used. Offset clocks with no exact overlap, a non-regular intersection and
a domain declaring irregular sampling are refusals, not invitations to choose a resampler after
seeing the relationship.

**A frame is not the same duration in two domains.** The search is declared in seconds.
`physical_lags` converts a duration to common-clock frames only after both native R21 floors
have been converted to physical time, and applies the larger of them. The synthetic thermal
instrument declares two hourly response intervals; the demand ledger declares one three-hour
reporting interval, so their conservative joint floor is three hours. A duration below either
floor, between two common-clock frames or declared twice refuses the entire family rather than
silently narrowing it. A domain carrying aggregates must also name its aggregation window, and
that window raises the physical floor so overlapping windows cannot masquerade as precedence.

**The boundary defines the family.** Every channel label is namespaced by its domain.
`cross_domain_pairs` enumerates every ordered pair whose operands cross the boundary, in both
directions, and no within-domain pair. Two channels in each domain therefore produce eight
directions; crossed with physical lags of 3, 6, 9 and 12 hours, `sweep_cross_domain` declares
and measures 32 members. Neither the pair, direction nor lag is read from the known answer.
The statistic is TG4.1's dimensionless magnitude correlation, so the Kelvin and megawatt values
are never compared as magnitudes; their original meanings and units travel beside each operand.

**The semantic record is inside the seal, not added to the prose later.** The aligned
`Record` carries both `DomainDeclaration`s, dataset identities, licences, native cadences, lag
bases and per-channel semantics and units in its provenance. `split_with_embargo` carries that
lineage into both partitions, TG3.2 binds the held-out partition before it is opened, and
`confirm_cross_domain` restores the operand records and physical lag to every relationship in
the receipt. Changing `K` to another unit after the seal changes the partition identity and is
refused. A within-domain candidate is refused at this boundary even if it is presented under a
valid seal. The claim boundary is explicit: a held-out, surrogate-referenced temporal
association between structural series, with no comparison of raw magnitude and no causal
mechanism established.

**The null performs the same selection.** `planted_cross_domain` and `cross_domain_null` are
one builder at coupling one and zero, with the same two domains, clocks, channels, marginals,
family and ensemble. On five root seeds the planted record confirms exactly
`synthetic_thermal_observatory::thermal_gradient > synthetic_demand_ledger::demand_pressure`
at the planted six-hour lag, at corrected q = 0.0050. The null freezes two to four candidates
selected from its own training partition and confirms none on held-out data at all five seeds;
its smallest corrected q ranges from 0.167 to 1.000. Its silence is therefore a measured null,
not a pass earned by carrying no hypothesis forward.

**The present boundary is deliberately narrower than "any two clocks".** It supports regular
native clocks whose exact intersection is itself regular, lag policies whose floor is fixed by
the domain declaration, and domains declaring that they have no natural cycle. It does not
interpolate, solve irregular-time inference, perform native-clock calendar removal, distinguish
a common driver from direct dependence, or make a causal claim. Those absences are refusals or
stated limits rather than freedoms hidden inside the alignment.

### 3.6z A motif that survives the process that found it (`src/core/motif_freeze.py`, TG5.1, `ed-dev`)

TG3.5's generate/confirm seal freezes an exemplar label and checks that the in-memory graph
presented under that label has not been swapped inside the same run. It deliberately does not
serialise the graph, so it cannot establish rule R20 across processes. `FrozenMotif` is that
durable boundary. Schema `frozen-motif/v1` stores the exact dimensionless node attributes,
typed edge values, relation set, refusals, registered matcher name, configuration size and
measured tolerance. The structural definition has its own SHA-256. A second outer SHA-256 binds
that definition to the originating `DomainDeclaration`, every node's carried domain/dataset/
variable/representation/unit record, the training `PartitionIdentity`, source family digest and
size, source exemplar label and selection counts, study identity and timezone-bearing freeze
time.

The structural and semantic halves remain separate in the serialised object: `definition`
contains no carried values, while `origin_domain` and `origin_nodes` are recorded but never enter
graph comparison. Freezing refuses a held-out source partition, a candidate that merely reuses a
declared label for a different graph, a graph whose nodes do not all name the declared origin,
and a node count different from the priced configuration size. Nested mappings become immutable
on construction, and `FrozenMotif.graph` reconstructs the exact `AttributedGraph` without access
to the mining process.

Persistence is canonical UTF-8 JSON with exclusive creation, flush and `fsync`; an existing path
is never replaced. Reload requires the exact versioned schema, exact field sets, canonical bytes,
valid graph structure, the training-partition digest, the structural digest and the outer digest.
`verify_published` additionally compares the internally consistent object with a digest held
outside the file. This distinction is essential: a content hash detects an edit, but someone who
can rewrite the whole artifact can recompute its hash. TG5.2 therefore binds the published
`motif_sha256` in its chronological ledger before it opens the target domain; TG5.1 alone does
not claim blind transfer merely because serialization exists.

Fourteen acceptance tests cover deterministic canonical identity, exact graph round-trip,
nested in-memory immutability, no-overwrite publication, structural/origin tampering, wholesale
self-consistent rewrite versus a published digest, held-out-source refusal, label/signature
redefinition, origin laundering, timezone requirements, strict schemas and non-canonical bytes.
No real domain was opened and no transfer result is represented.

### 3.6za Binding before opening, with no redefinition surface (`src/core/motif_transfer.py`, TG5.2, `ed-dev`)

`blind_transfer` makes the chronology TG5.1 left open one controlled operation. The target
values are not an argument: they sit behind a zero-argument `opener`. Before invoking it, a
durable `TransferLedger` verifies the `FrozenMotif` against the separately published
`motif_sha256`, requires strict timezone-bearing `frozen_at < bound_at < opened_at`, rejects the
source domain and source partition as targets, and commits the motif/definition digests, exact
target `PartitionIdentity`, complete target `DomainDeclaration`, and all three times. The
committed record is canonical JSON, individually content-addressed, flushed and `fsync`ed before
the callback can run. Reload checks exact fields, canonical bytes, record, target-partition and
target-declaration hashes, the published/motif equality, state and chronology.

The target is conservatively spent at that commit. A loader exception, malformed return,
mislabelled feature domain or post-open subset of the pre-bound frame count does not roll it
back, and a new ledger instance refuses a second motif against the same target identity. This is
what prevents a failed or unpromising first look from becoming a free trial before another
definition is chosen.

After opening, exhaustive enumeration constructs every target signature using only the frozen
configuration size and registered matcher, then compares it using only the frozen relation set
and measured tolerance. Those choices do not appear in the public function signature. The
matcher must itself declare `crosses_domains=True`; carried semantic records stay outside the
comparison but both domains remain visible in every match report. The receipt binds the opening
record, exact search definition, all examined and matching labels, scene support, both domain
declarations and its own content digest. Zero matches is a complete descriptive result.

This boundary proves ordering for access performed through this API, not that a person or
another program never inspected an archive earlier; external access control remains the
responsibility of the archive/preregistration system named in target provenance. It also makes
no corrected relationship-transfer claim: TG5.3 owns that family, null and train/test accounting.
The sixteen acceptance tests use two synthetic domains and cover 18 pytest cases; no real target
archive has been opened.

### 3.6zb Corrected relationship transfer on target train and test (`src/core/motif_relationship.py`, TG5.3, `ed-dev`)

TG5.3 asks a narrower second question after structural transfer: whether presence of the frozen
motif is followed by greater values of a declared organisation measure. `RelationshipPlan`
freezes the published motif identity, complete target `DomainDeclaration`, ordered non-overlapping
target train/test `PartitionIdentity` records, outcome x positive-lag family, one-sided statistic,
circular-shift null, surrogate count, alpha, correction and root seed. The target domain must
license precedence under R21 and every lag must meet its declared floor. The resulting G3
`SearchSpecification` must be affordable before a plan can exist.

The plan is canonical, content-addressed, exclusively published and checked against a separately
published digest. `test_relationship_transfer` exposes none of the matcher, tolerance, outcome,
lag, surrogate, alpha, correction or seed choices. A durable `RelationshipLedger` verifies both
published digests and commits both target partitions before either opener runs. Each partition is
globally one-use, including attempts to reuse it with a different counterpart; opener or
validation failure does not return either split to the available pool.

Execution rebuilds frozen motif presence by exhaustive configuration search in every frame. For
every declared outcome x lag member it measures
`mean(outcome[t+lag] | motif[t]) - mean(outcome[t+lag] | no motif[t])`, obtains a one-sided
Monte-Carlo p-value by circularly shifting the outcome relative to the unchanged presence series,
and includes non-estimable members as `p=1` rather than shrinking the family. BY (or the other
predeclared registered correction) is applied over the complete frozen family independently in
train and test. `PASS` requires the same positive label to reject in both; an adequately powered
empty intersection is `FAIL`, a complete null result rather than an omitted finding. The receipt
contains both full corrected families, split identities, chronology, replicated labels, verdict,
and its own digest.

This is a replicated precedence/association instrument, not evidence of mechanism, causality or
predictive utility. Circular shifts preserve each finite observed series but do not eliminate a
shared driver. The sixteen acceptance tests use synthetic planted and null records; no real target
archive has been opened, and the API chronology still depends on external access control for the
claim that no person or out-of-process program inspected it earlier.

### 3.6zc The hashed, append-only evidence bundle (`src/core/evidence.py`, TG6.1, `ed-dev`)

Phase G6 opens with the structure the claim ladder will read. An `EvidenceBundle` is an immutable
snapshot of one `Hypothesis` — identifier, statement, prediction, timezone-bearing registration
time and provenance — together with an ordered chain of `EvidenceEntry` records. The hypothesis
cannot be registered after the bundle that anchors it, and it cannot be swapped underneath an
existing chain, because the chain's anchor digest binds the schema, study id, creation time and
hypothesis digest.

Evidence is appended by `append()`, which returns the **next** bundle; the receiver is unchanged
byte for byte, so a prior revision remains a citable snapshot. Each entry carries a sequence
number, category, label, one of `PASS/FAIL/INVALID/INCONCLUSIVE/NOT_APPLICABLE`, a summary, a
recorded time, a non-empty structured payload, and the digests of the source artifacts it came
from. Each entry hashes its own body **including** the previous entry's digest, so dropping,
reordering, relinking or softening any entry — supporting or otherwise — invalidates the chain.
Append chronology is non-decreasing and sequence numbers are gap-free. Payloads are deep-frozen
and rejected unless they are finite JSON, so an infinity, a NaN or a live Python object cannot
enter the record.

There are ten first-class scientific fields: `observations`, `effect_sizes`, `uncertainty`,
`null_results`, `replication_results`, `holdout_performance`, `provenance`, `confounders`,
**`contradictory_evidence`** and **`failure_states`**. The last two are ordinary fields with
ordinary accessors, not remarks appended to a discussion; they occupy the same chain and the same
digest as favourable evidence, and no later append can remove or dilute them. Every entry must
name one of these categories: there is no `commentary`, `notes` or `interpretation` route, so
free text has no way into the structure the TG6.2 gates will read. TG7 may record adversarial
commentary **beside** this bundle, never inside it (R22).

`save_evidence_bundle` publishes one canonical, exclusively created JSON snapshot and refuses to
overwrite an existing one; `load_evidence_bundle` rebuilds every nested entry, re-verifies each
entry digest, chain link, sequence, category placement and the whole-bundle digest, requires the
bytes to be exactly canonical, and optionally checks a separately published digest. Because the
grouped-by-field publication format is reassembled by sequence, an entry relocated into a
different first-class field on disk is refused rather than silently re-filed.

**Claim boundary.** This is a tamper-evident container and a routing discipline, not a judgement.
It does not decide whether the evidence inside supports anything; TG6.2's ladder and TG6.3's five
outputs own that — see 3.6zd for the ladder, 3.6ze for the five outputs, 3.6zf for the
recorded-call boundary above them and 3.6zg for the round-robin that speaks through it. Nor does
it authenticate the author: hashes detect
edits to a published bundle, they do not prove who wrote it or that some evidence was never
gathered and simply left out. Only what is appended can be weighed.

### 3.6zd The claim ladder over a bundle (`src/core/claim_ladder.py`, TG6.2, `ed-dev`)

`observation → association → robust association → candidate precursor → demonstrated predictive
utility`. `assess_claim_ladder(bundle)` is a **pure function of the bundle**: it takes no other
argument and reads no clock, filesystem, environment or random source, so the same evidence chain
always yields the same verdict and the verdict can be recomputed by anyone holding the published
bundle.

Ten declarative gates decide the climb. `association` needs a passing `observations` entry, a
passing `effect_sizes` entry and a passing `uncertainty` entry — a point estimate without a
quantified interval is not an association. `robust_association` adds passing
`replication_results` and `confounders` entries. `candidate_precursor` adds a passing `provenance`
entry **and** a passing `provenance` entry whose payload records `temporal_precedence` exactly
`true`; a truthy stand-in such as `1` or `"true"` does not count, because precedence must be
asserted, not inferred. `demonstrated_predictive_utility` adds a passing `holdout_performance`
entry. Only `PASS` advances a gate: `INCONCLUSIVE` and `NOT_APPLICABLE` are recorded honestly and
buy no ground.

Two gates sit on the floor rung and dominate everything above it. Any entry recorded as `FAIL` or
`INVALID`, anywhere in the bundle, caps it at `observation`; so does any `contradictory_evidence`
or `failure_states` entry recorded as `PASS`, which is the bundle asserting that the contradiction
or the failure **stands**. No quantity of favourable evidence outvotes one of these, and because
TG6.1 entries are immutable, a failure cannot be appended away — the only route to a higher rung
is a bundle that never carried it. The assessment still reports `unblocked_rung`, the rung the
remaining evidence would have reached, as a **diagnostic**: `rung` is what may be claimed.

Causal claims are outside the ladder entirely. `permits()` answers for a rung, but **refuses** for
`causal`, `causation`, `mechanism`, `efficacy`, `cure` and their neighbours, naming R7 — a
refusal rather than a `False`, because a `False` invites a later "not yet" reading. The top rung
is a statement about out-of-sample prediction and says nothing about mechanism.

The gates read only category, status and the one reserved `temporal_precedence` payload key of
entries that TG6.1 already confines to its ten first-class fields. Labels, summaries and every
other payload key are inert, so no amount of persuasive prose in a bundle can move a rung (R22).
A `ClaimLadderAssessment` is immutable, reports every gate with its requirement and whether it was
satisfied, and carries an `assessment_sha256` binding the verdict to the exact bundle revision and
hypothesis digest it was computed from.

**Claim boundary.** The ladder grades the evidence that was appended; it cannot know what was
never gathered. It is a floor on rigour, not a certificate: reaching `demonstrated_predictive
utility` means a holdout result was recorded and passed, not that the study design was sound, that
the holdout was honestly held out, or that the effect will hold elsewhere. TG5's external access
control and TG6.1's tamper evidence own those questions, and 3.6ze states what a bundle's
verdict leaves open, and no rung of this ladder — including
its top — licenses a causal reading.

### 3.6ze The five outputs of a bundle (`src/core/five_outputs.py`, TG6.3, `ed-dev`)

For any TG6.1 bundle, `summarise_evidence(bundle)` states five things and nothing else: what can
be claimed, what cannot, what evidence contradicts it, which alternative explanations remain, and
which single observation would most efficiently distinguish between them. Like the ladder it sits
on it is a **pure function of the bundle** — no second argument, no clock, no filesystem, no
environment, no randomness — so the report is recomputable by anyone holding the snapshot, and
`summary_sha256` binds it to the exact revision it came from.

**What can be claimed** is the rung the ladder assigned and every rung beneath it, together with a
fixed *entitlement* sentence stating what that rung licenses a reader to say and, explicitly, what
it does not. The entitlement is the programme's own wording, not the study's, so a bundle cannot
supply the sentence that describes it.

**What cannot be claimed** is the exact complement: every rung above, each naming the specific
gates that stand between and their requirements. Because the ladder is climbed in order, a rung
whose own gates all pass may still be unreachable — a floor-rung failure, or an unmet gate on any
rung between. The nearest such rung is named, so no unreachable rung is ever reported without a
reason.

**What contradicts it** is deliberately wider than the ladder's blocking set. The ladder asks what
may be claimed; this output asks what argues against the hypothesis, so it carries every `FAIL` or
`INVALID` entry in any category, every `contradictory_evidence` and `failure_states` entry at any
status but `NOT_APPLICABLE`, and every passing `null_results` entry — a recorded null is contrary
evidence even where it caps nothing. Each carries `caps_at_observation`, which agrees exactly with
the ladder's blocking set, so a reader can tell what merely argues against a claim from what
forbids it.

**What alternatives remain** has two origins, and the distinction is load-bearing. Eight
*structural* alternatives are mapped one-to-one onto the eight climbing gates — `chance`,
`sample_specific`, `confounding`, `reverse_or_simultaneous_order`, `in_sample_optimism`,
`unauditable_origin`, `nothing_measured`, `no_estimated_effect` — and each stays open exactly
while its gate is unsatisfied. The mapping is total over the gate table, so no unmet gate goes
unexplained. Every other alternative is *recorded*: a `confounders` entry that is not passing, a
contradiction that is not `NOT_APPLICABLE`, a passing null result. The module never invents a
domain alternative; it enumerates the ones its gates correspond to and the ones a person wrote
down.

**The single next observation** follows a fixed precedence, stated as such: resolve the earliest
blocking entry if one stands, since while it does no further evidence can lift the bundle at all;
otherwise close the first unmet gate in ladder order; otherwise distinguish the earliest recorded
alternative; otherwise nominate nothing, and say plainly that nominating nothing is not the same
as there being nothing. `render()` produces the five outputs as deterministic plain text in that
order, with the R7 line present at every rung.

Membership of all five outputs is decided by category, status and the gate table alone. Labels and
summaries are carried through to the reader, since a human needs to know which confounder is
open, but they can never move a verdict (R22). `permits()` delegates to the ladder, so causal
claim kinds are refused rather than denied here too, at every rung including the top (R7).

**Claim boundary.** The fifth output is a precedence rule, not an experiment design: no expected
information gain is computed, because the bundle carries no likelihoods to compute one from, and
"most efficient" means "the cheapest thing standing in the way", not "optimal". The fourth output
can only name alternatives its gates correspond to or that someone recorded — a domain-specific
rival explanation nobody wrote down is invisible to it, and its absence from the report is not
evidence of its absence in fact. The same holds of the third: "none recorded" is not "none
existing", and the rendered text says so rather than leaving silence to be read as reassurance.

### 3.6zf The recorded-call boundary (`src/core/recorded_call.py`, TG7.1, `ed-dev`)

Phase G7 puts an adversarial review layer **above** the G6 gates. This module is the only door
through which that layer may speak, and it is built so the layer cannot reach the gates through
it. Three things are enforced.

**Every call is recorded verbatim, because it cannot be regenerated (R23).** A `RecordedCall`
carries the exact request sent, the exact response bytes returned, the exact model id, the effort
setting, the API request id and both timestamps, chained to its predecessor by digest. The record
states `recorded-not-reproducible` in its own body, and a loaded record claiming to be
deterministic is refused by name. Sampling parameters are **refused rather than recorded** at any
depth of the request — `temperature`, `top_p`, `top_k` and `seed` — because they are rejected by
the API on current models and because pinning one would imply a reproducibility this layer does
not have. `render()` prints the R23 declaration above any commentary it displays, and says
plainly when nothing was recorded.

**Every response is constrained to a declared schema.** A `ResponseSchema` of typed, optionally
enumerated fields is sent as the request's `output_config.format` with `additionalProperties`
closed, and the recorded response must parse as JSON matching it exactly: free text where a
schema was declared is a refusal, not a string for later code to parse hopefully. The record
keeps the response bytes *and* the structured parse and requires them to agree, so the parse can
be re-checked at any time; that check runs in `__post_init__` rather than at record time, so it
survives a round trip through disk. A closed schema is also what stops an undeclared
`claim_level` field arriving in an answer.

**No recorded output is an input to any claim level (R22).** Commentary lives in a `ReviewRecord`
that binds the bundle's digest and revision **from outside**, is published in its own file beside
the bundle, and is never appended to the evidence chain — the ten scientific fields contain no
category it could be appended to. A `ReviewedBundle` holds the pair and computes every
claim-bearing property from `self.bundle` alone; `record_call` hands back the *same bundle
object* it was given. `strip_review` deletes the whole review and returns that bundle unchanged,
and `verify_claim_independence` checks the claim digest with the review present and deleted,
re-derives it from the bundle's own canonical bytes, and refuses if any recorded phrase of at
least `SMUGGLING_FLOOR` characters is found inside those bytes — the one way R22 could actually
be broken is a person copying an answer into a summary or payload, and that is caught rather than
assumed absent.

Commentary on one revision is not commentary on another: appending evidence yields a new bundle,
and pairing it with the old review is refused. The eight TG7.2 round-robin roles are fixed here,
so a call cannot invent an authority for itself, and the transport is injected — this module
opens no socket, which is why the boundary is testable without one.

**Claim boundary.** This slice records calls and fences them off; it does not conduct a review.
Nothing here judges whether a challenge was any good, and a well-formed response saying something
false is recorded exactly as faithfully as a true one. The smuggling check finds recorded wording
reproduced in the bundle; it cannot detect a person who reads commentary, is persuaded by it, and
records a genuine-looking measurement in their own words — no structural check can, and the
defence against that is the chain of provenance the gates already require, not this function.
Determinism is not claimed anywhere in the layer: an identical request may return a different
answer tomorrow, which is precisely why the answer is stored rather than recomputed.

### 3.6zg The adversarial round-robin (`src/core/round_robin.py`, TG7.2, `ed-dev`)

Eight seats speak in a fixed order over one frozen bundle: a candidate synthesis, four
challengers, one response per dissent raised, an independent reassessment, and a bounded final
synthesis. Every turn goes through the TG7.1 boundary, so the whole exchange is recorded
verbatim beside the bundle and none of it can reach a claim level. Three things are enforced
here that the boundary alone does not give.

**The order is the protocol, and it is replayed rather than trusted.** `RoundRobin` recomputes,
for every prefix of the recorded chain, the turn the protocol would have demanded at that point,
and refuses a record whose role, target, response schema, model or effort is not the one that was
due. A challenge cannot be synthesised over before it has been answered, each challenge is
refused a second answer — which is what bounds the exchange, since a dissent left permanently open
would keep demanding turns — a ninth turn cannot be appended to a finished exchange, and the panel
is pinned seat by seat because a bundle reviewed by a different model is different evidence. A turn that comes back malformed is recorded first and
refused second: `RecordedTurnRefused` carries the reviewed bundle *including* the offending call,
so nothing that was paid for is discarded because it was disappointing (R23).

**This is not majority voting.** Nothing in the module counts verdicts. A dissent is retired only
by the candidate conceding it, or by a rebuttal that the independent reassessment declines to
reopen — the candidate does not mark its own homework, and until the reassessment has spoken a
rebuttal is provisional and the dissent stands. The reassessment may reopen a dissent but cannot
originate one at that turn, because nothing downstream would answer it. Three challengers
agreeing has no effect on the fourth's objection, and there is no value anywhere in `CLOSURES`
meaning *outvoted*.

**Unresolved dissent is retained, never reconciled.** The final synthesis must name exactly the
unresolved dissents — no more, no fewer — and its `dissent_remains` flag must match whether any
actually stands; a synthesis that drops one, invents one, or reports calm while one is open is
refused. `RoundRobinOutcome` carries each retained dissent with the challenger's own argument and
the alternatives it could not exclude, and `render()` prints them under the R23 declaration. The
roadmap says dissent is retained *in the bundle*; R22 says no LLM output may be in the bundle at
all. Both are honoured by retaining it in the review record, which is published beside the bundle
and travels with it.

**The panel.** `ReviewPanel` seats a model and an effort for each of the eight roles, digests the
result, and pins it into the outcome. Reviewer overlap is **recorded, not refused**: one model in
several seats is a weaker exchange than several — an independent reassessment by the model that
wrote the candidate synthesis is not independent in the usual sense — but refusing it would make a
single-provider panel impossible to run at all. So the overlap is computed, carried in the digest,
and stated plainly by `render()` wherever the outcome is displayed. `close_round_robin` runs
`verify_claim_independence` before building the outcome, so R22 is re-proved on the way out of
every exchange.

**Provider independence.** The module names no vendor and opens no socket; the transport is
injected, `model_id` is any string, and `effort` is an abstract three-valued knob a later adapter
translates (a thinking budget in tokens, for instance). The closed-response schema is enforced
locally by `ResponseSchema.validate` on the parse, so an undeclared field is refused whether or
not the provider honoured `additionalProperties: false`.

**Claim boundary.** This slice conducts the exchange; it does not judge it. Nothing here measures
whether a challenge was any good, whether a concession was warranted, or whether a rebuttal was
honest — a fluent, false objection is retained as faithfully as a sound one, and a lazy panel that
raises no dissent produces a clean outcome that means nothing. Independence is checked at the
level of the model *id* and nothing deeper: `reassessment_is_independent` reports two different
ids as independent, but two sizes of one family share training data, tokenizer and failure modes,
so they differ in capability rather than in perspective — and it is the correlated blind spot that
an adversarial exchange exists to catch. A tiered panel drawn from one provider buys cost control
and a capability gradient, not the independence the word suggests; seating genuinely unrelated
reviewers is a configuration decision this module records and does not make. Every test uses a recorded transport, so nothing here shows that a real client behaves
as the protocol expects. And retention is not resolution: an outcome that carries four unresolved
dissents is an honest record of an argument nobody won, not a finding.

### 3.6zh Cost-controlled review transport (`src/core/review_cost.py`, TG7.3, `ed-dev`)

TG7.3 adds the first provider implementation beneath the provider-neutral TG7.1/TG7.2
contracts: Gemini 3.5 Flash (`gemini-3.5-flash`) through the asynchronous inlined Batch
GenerateContent API. `GeminiBatchTransport` translates the already-recorded system text,
instruction, context, effort and closed response schema into one batch request, polls the named
`batches/{id}` operation, accepts exactly one successful inlined response, and returns TG7.1's
transport mapping. Low, medium and high effort map directly to Gemini thinking levels. The API
key exists only in the `x-goog-api-key` header; it has no serialisation path, its representation
is redacted, and provider failures are reported without response bodies.

`ReviewCostPolicy` fixes the model, effort per role, required service mode, minimum measured
cache hit, the documented Batch discount and the provider-documentation snapshot before a review
runs. The initial single-model panel uses low effort for the four challenger seats, medium for
the response, and high for candidate synthesis, independent reassessment and final synthesis.
This is a cost/capability gradient, not reviewer independence; TG7.2's overlap report remains
unchanged and will say that all eight seats share a model.

`audit_review_cost` does not accept configuration as evidence. Every recorded call must name the
predeclared model and effort, carry `service_mode: batch`, bind its unique batch resource to the
TG7.1 API request id, preserve the raw Gemini `usageMetadata`, and reconcile each normalized
input/output/cached/total token count against that raw object. The review is refused unless the
sum contains a non-zero measured cache hit. The resulting content-addressed
`ReviewCostReceipt` binds those totals and batch identities to the exact review-record and policy
digests; canonical save/load is atomic, no-overwrite by default, and rechecks the digest and
arithmetic on read. Prices are deliberately not stored as timeless facts: the receipt preserves
tokens, route, the reviewed pricing source and its 50% Batch factor so a dated price table can
reconstruct money later.

Official Gemini documentation reviewed 2026-08-26 records `gemini-3.5-flash` as GA with Batch
and context-caching support, a 4,096-token implicit-cache threshold, cache hits in usage metadata,
and Batch at 50% of standard token price. The wire adapter and an eight-call review with non-zero
cache usage are offline-accepted. A live structured Batch smoke run is also accepted: operation
`batches/nd6n27...mb26` returned exactly `{status: "ok", note: ...}` under the declared schema,
with 52 input, 32 visible candidate, 92 thinking and 176 total tokens. The adapter normalizes the
124 visible-plus-thinking tokens as billed output and preserves the raw split. The tiny prompt
reported zero cached tokens, so live cache-hit and cost-receipt acceptance remain NOT RUN.

**Claim boundary.** A lower token bill does not make a review better, and a cache hit does not
make an answer reproducible. This receipt says which paid route was used and what the provider
reported consuming. It neither audits provider billing nor promotes, demotes or judges any
claim; R22 and R23 remain wholly owned by the layers below.

### 3.6zi Translation, bounded (`src/core/translation.py`, TG7.4, `ed-dev`)

Every layer beneath this one speaks in the programme's vocabulary — rungs, gates, categories,
statuses, digests. A domain scientist reading `candidate_precursor, blocked by
robust_association.confounders_addressed` learns nothing. TG7.4 renders a finding in domain
language, and it is the first point in the programme where text is produced for a human to act
on. That is precisely why four rules break here at once if nothing stops them: domain prose
reaches naturally for the semantic comparison R19 forbids; a bare confidence figure is the way R9
says this platform is most likely to mislead its own author; causal verbs enter through sentences
rather than through claim kinds (R7); and "candidate precursor" becomes "early warning signal" — a
promotion carried out entirely in wording, with no gate touched (R22).

**Translation is a projection, not a generation.** There is no model call. Domain wording arrives
as declared, content-hashed data screened when it is registered, and the renderer emits only from
closed template sets bound to structural facts — TG7.1's design, that the dangerous thing is
*refused rather than recorded*, at the boundary, once.

**R9 is given a structure it did not have.** The six figures R9 names — support, confidence, base
rate, lift with an interval, and surrogate-corrected lift — had no structured home anywhere in
`src` before this slice. `claim_ladder` asks only whether an `effect_sizes` entry *passes*; it
never reads what is inside one, so the figures lived unvalidated in a payload mapping.
`AssociationFigures` keeps all six together, refuses a partial set by name, and checks lift
against `confidence / base_rate` rather than trusting it. Its `render` is the **only** method in
the module that can format a percentage, so the roadmap's cautionary "82% of the time" is not
banned but made honest: the base rate that decides whether 82% is a finding or noise is in the
same string.

**R19 holds by construction.** Two disjoint template sets, selected by `semantic_key` equality.
The within-domain set may reference units, magnitude and the variable's identity; the cross-domain
set may reference only `structural_signature()` — the dimensionless view built from the fields
that survive being stripped of their units. A cross-domain magnitude sentence is not caught after
the fact; it cannot be constructed, and widening `CROSS_DOMAIN_FIELDS` is a visible edit to a
named constant rather than an invisible consequence of a filter.

**R22 holds by signature.** `translate` takes a `FiveOutputs`, never a `ReviewedBundle`. Review
commentary cannot reach the claim text because there is no parameter through which it could
arrive; what a caller passes as `commentary` is carried in its own quarantined field, excluded
from `claim_text()`, and rendered under a heading saying it moved nothing.

A glossary is registered through the same `Registry` orientation conventions use, so a domain
supplies one **without editing `src/`** — the TG8.1 onboarding condition met early. Registration
refuses a partial map, any phrase carrying a term from `OUTSIDE_THE_LADDER`, any phrase carrying a
digit, any comparative asserting a relation of size, and any rung phrase borrowing wording
reserved to a higher rung. Each entitlement is welded into the same string as the claim it bounds,
so a UI cannot render "this is a candidate precursor" while dropping "predictive utility is not
shown".

The causal guard scans each unit's *rendered* half and not its *licences*, because several
entitlements name a causal term precisely in order to deny it — *"a statement about prediction,
never about mechanism"*. Scanning those would refuse the sentence whose whole job is to hold the
line, so the curated half and the authored half are scanned differently rather than together.

**Claim boundary.** A translation is faithful to the **record**, not to the world. A glossary
mapping a structural term to a misleading-but-non-causal domain word is accepted, because no
structural check knows what "anomaly" means to an oceanographer; the defence is that the glossary
is declared, hashed and reviewable, not that it is correct. Refusing causal *vocabulary* is not
refusing causal *implication*, and a reader who reads "precursor" as "cause" is caught by nothing
here. The R19 guard stops the *system* emitting a cross-domain comparison; it cannot stop a reader
setting two within-domain renderings side by side and drawing one themselves, and layout is out of
scope. Nothing here judges whether a finding was worth translating, or whether the domain words
chosen are the ones a practitioner would use.

### 3.6zj The read-only claim surface (`src/api/findings.py`, TG9.1, `ed-dev`)

Before this slice the cross-domain line had **no HTTP surface at all**. Every route in
`api/main.py` belonged to the atmospheric/transform line, and all twelve G-line modules —
`domain`, `feature`, `motif`, `evidence`, `claim_ladder`, `five_outputs`, `recorded_call`,
`round_robin`, `translation`, `cross_domain`, `constellation`, `family` — had zero references
there. Nothing a browser could reach knew a claim ladder existed.

`src/api/findings.py` is that surface, mounted under `/api/v1/findings`, and it is **read-only**:
nothing in it appends evidence, records a call or moves a rung, so a GET cannot change what may
be claimed (R22). The surface is six read-only endpoints: registered domains, one glossary
whole, published studies, one bundle, its five outputs, and its translation.

**Phase G9's principle applied to the wire: a client computes and formats no scientific number.**
A translation is served already rendered as `rendered_text`, beside its structured units and its
`structural_keys`, so a client displays strings rather than assembling them. The five outputs are
served untranslated as well, because a reader who wants to check that domain wording changed no
fact needs to see both forms.

**R9 is enforced on the wire rather than in each handler.** `refuse_bare_confidence` walks every
response body, at any depth, and refuses a `confidence` key not accompanied by all six of R9's
figures. The rule is usually described as a frontend constraint, which puts it in the one place
it cannot be enforced; this is the last point before a client sees it. The guard **restates**
R9's six field names rather than importing them from `AssociationFigures`, because a guard that
imported its expectations from the thing it guards would agree with any change made to it — and
a test asserts the two statements still agree. A partial figure set is served as no figures at
all rather than as a subset: four of six is not four-sixths of a finding.

**Registration is eager, and that is the point.** Defect D35 was a fallback chain that depended
on browsing order, because source registration was an import side effect of a module imported
lazily inside its own handler. `DOMAIN_GLOSSARIES` has exactly that shape, so
`register_builtin_glossaries()` runs at module import rather than inside a handler, is
idempotent, and returns the full built-in set so a caller can assert it rather than hope.
`src/core/builtin_glossaries.py` supplies the first two vocabularies — reanalysis and order-book
— written against TG7.4's four registration screens rather than fixed up afterwards.

`StudyStore` is deliberately neither a database nor a cache. A bundle is a tamper-evident
canonical file whose digest `load_evidence_bundle` rechecks on every read; holding a parsed copy
would mean serving a claim state that no longer matches disk, which is the staleness the digests
exist to prevent. A file that will not parse is reported as unreadable rather than skipped,
because silently omitting it would let a corrupted bundle look like a study nobody ever ran.

**Claim boundary.** A surface that cannot serve a bare confidence does not make the science behind
it good; it removes one way of misreading it. Serving a finding in domain words does not make the
finding true, and a fluent rendering of a weak result is more persuasive than a jargon-laden
rendering of the same result — a risk this surface creates rather than removes. An empty
"evidence against" section means nothing was recorded, not that nothing exists, and the rendered
text says so in those words. Nothing here touches a gate, a rung or a digest.

### 3.6zk The findings view (`frontend/src/components/FindingsView.tsx`, TG9.2/TG9.4, `ed-dev`)

An eleventh tab rendering a `TranslatedFinding`: the five outputs as five sections, each claim
welded to the bound that qualifies it, with panels for the untranslated claim state, the glossary
that worded it and the evidence bundle itself. A domain selector renders one study through any
registered vocabulary — the same study, different words, identical facts. Nothing in the existing
transform workbench was refactored; six client methods were added to `services/api.ts`.

**What the component does not contain is the point.** No `toFixed`, no percent literal, no
arithmetic on a claim value, and no read of any of R9's six fields. `figures_text` — the line the
backend assembled — is the only route to the association strength, so a bare confidence is not
withheld by discipline here; there is no code that could produce one. Two contract tests assert
this over the source, with comments stripped first so the component can document the constraint
without appearing to breach it. Three deliberate mutations were each caught.

**A gap found by writing the view.** The first draft interpolated `figures.support` into its own
panel: no formatting, no arithmetic, and still wrong, because a view that builds that line from
parts puts R9 back in the hands of whoever writes the JSX. The API now serves `figures_text` so
the view has nothing to assemble, and the test was tightened from "formats no number" to "reads no
claim-bearing field" as a result.

**Accessibility (TG9.4) applies to this surface only.** The findings views carry
`role="tablist"`/`role="tab"`, `aria-selected`, `aria-pressed`, `aria-label`, labels bound with
`htmlFor`, visible focus rings and `aria-hidden` on decorative icons, asserted by test. The
platform-wide measurement in `roadmap.md` §1 is unchanged: the legacy workbench was not touched,
and this stops the new surface adding to that debt rather than repaying it.

**Claim boundary.** `npx tsc --noEmit` is clean and `npm run build` succeeds, which proves the tab
compiles, calls endpoints that exist and reads fields that are present. **Rendered appearance is NOT
RUN**: no browser has displayed this tab and no screenshot exists in the repository, exactly as for
the tenth tab before it. A view that cannot render a bare confidence does not make the finding it
displays worth reading.

### 3.6zl The refusal surface (`src/core/builtin_domains.py`, TG9.3, `ed-dev`)

TG9.1 shipped a `/domains` route listing **glossaries** — how a domain speaks — and nothing about
what it refuses. That was half of what the slice declared, and the missing half is the more
important one: a reader can be told a finding in fluent domain words while the domain those words
belong to does not admit the claim being made. TG9.3 delivers it.

`DOMAIN_DECLARATIONS` is a registry in `src/core/domain.py`, beside the type it holds, so a domain
registers what it *is and breaks* the same way it registers its wording.
`src/core/builtin_domains.py` supplies the two declarations behind the built-in vocabularies,
chosen to make the tension visible rather than to look tidy. **reanalysis** breaks nothing and
floors an advective lag, so precedence is admissible — R17 permits an empty violation set only
because the domain floors something. **order_book** breaks `no_physical_metric`,
`no_propagation_speed`, `unordered_channels` and `aggregated_values`, and declares
`lag_policy="none"`, which is the honest position for a domain with no propagation mechanism and
is not a failure state: association remains measurable, the lead-lag *interpretation* is refused
(R21).

`refusals_for` assembles what a domain forbids, drawing each consequence straight from
`KNOWN_VIOLATIONS` rather than restating it, so the reason shown to a reader and the reason
enforced in the analysis layer cannot drift apart (R17).

**The constraint that shapes the slice: an `EvidenceBundle` does not record which domain produced
it.** Its fields are the hypothesis, the ten evidence categories and their digests — there is no
domain among them. So nothing here checks a study against a domain, and presenting the limits as
such a check would fabricate one. `DOMAIN_ATTRIBUTION_CAVEAT` travels with every served limit, and
a contract test refuses a view that restates it in its own words instead of rendering the sentence
the API vouched for.

**`unadmitted_reading`** is the one genuinely new report. When a record stands at
`candidate_precursor` or above — the rung at which a claim first asserts temporal ordering — and
the selected vocabulary belongs to a domain declaring no admissible lag floor, the surface says
so. It is careful about what that means: the ladder is domain-agnostic, and nothing here moves a
rung (R22). If the study did come from that domain it is a contradiction someone must resolve; if
it did not, the vocabulary is simply the wrong one to read it in. The surface cannot tell which,
and says so rather than choosing.

Commentary is rendered in its own `<section>` with its own aria-label, and a structural test
refuses any `TranslationUnit` field inside that container, so recorded argument (R23) can never be
mistaken for what the record permits.

**Claim boundary.** Showing what a domain refuses does not enforce it: R17's refusals are enforced
in the analysis layer and this displays the same facts rather than adding a check. A domain
declaring no violations is not thereby unconstrained. And the attribution gap is real: until a
bundle records its domain, `unadmitted_reading` describes a vocabulary a reader chose rather than
a verified provenance. Closing it means putting domain provenance into a G6 structure, which
belongs to no slice yet declared.

### 3.6zm The onboarding contract (`src/core/onboarding.py`, TG8.1, `ed-dev`)

A domain reached this programme through two registries that knew nothing about each other:
`DOMAIN_GLOSSARIES` (TG7.4) for its wording, `DOMAIN_DECLARATIONS` (TG9.3) for what it is and
what it breaks. Either could be registered without the other, and the asymmetry is not neutral.
**Wording without a declaration is a domain that speaks fluently and refuses nothing** — not a
hypothetical, since TG9.1 shipped exactly that and served it for a slice before TG9.3 caught it.
A declaration without wording fails quietly in the other direction: limits that exist and cannot
be read.

`onboard_domain` makes the pair the unit. Glossary, declaration and the record of the contract
itself are registered **atomically**: every screen runs before the first `Registry.add`, and any
failure restores all three registries to the state they had before the call. Registries are
process-global, so a partial write would be a defect of exactly the D35 family — behaviour
depending on which registration happened to have run.

**The recipe is a tuple, not prose.** `REQUIRED_DECLARATIONS` names the seven things a domain
must declare — axes (E14), geometry (E13), lag policy (R21), violations (E15/R17), licence,
provenance, glossary (TG7.4) — each with the reason it is required. `OnboardedDomain.checklist()`
is generated from it, `GET /api/v1/findings/onboarding` serves it, and the tests assert against
it, so the contract a reader is held to and the contract the code enforces are one object rather
than two lists that drift.

**The one check nothing else could make.** Everything on that list is already validated by
`DomainDeclaration` or `DomainGlossary` — except the relationship *between* the geometry and the
violations, which no single object can see because the two facts live in different registries.
The contract is a biconditional: a domain declares `no_physical_metric` **if and only if** its
geometry offers no physical metric, asked of the geometry's declared `physical_metric` capability
rather than of its name (E2). Both halves bite. A domain naming `cartesian` while declaring
`no_physical_metric` has supplied a metre and then renounced it, so code asking the geometry is
told metres while code asking the declaration is refused, and nothing reconciles the two answers.
A domain supplying no geometry and *not* declaring `no_physical_metric` is claiming lengths it
cannot produce — the same failure with the sign flipped, and the more common one, because it is
what an adapter author writes before they have thought about it.

**`audit_onboarding` reports the difference between complete and assembled.**
`DOMAIN_DECLARATIONS` can only answer *does a declaration exist*; the question that matters
downstream is *was this domain ever checked as a whole*. A domain registered piecemeal is
reported `complete: false` with what is missing and a note saying its geometry and violations
were never checked against each other, rather than passing for a checked one. `/domains` now
lists the **union** of both registries, so a domain that declared its limits and never declared
its wording is visible instead of absent — the TG9.1 omission with its halves swapped, closed in
the same slice that could otherwise have reproduced it.

**The built-ins are held to the contract they document.** `register_builtin_domains` goes through
`onboard_domain` rather than adding to the registries directly, and
`register_builtin_glossaries` delegates to it: a way in that quietly produced wording without
limits would leave the hole open beside the fix for it. A built-in found half-registered is
**repaired rather than skipped**, since a name present in some registries and not others is
precisely the state the contract forbids. `reanalysis` declares `latlon`; `order_book` declares
no geometry and `no_physical_metric` with it — the two sides of the biconditional.

**Acceptance, in the form `test_registries.py` uses for sources and actions.**
`src/tests/domain_plugin_example.py` onboards a third domain in one file, outside `src/`, and the
test hashes the six files a domain would otherwise have had to touch and asserts they are
byte-identical afterwards. The domain is Argo profiling floats, chosen by R17's reasoning rather
than by sector: of the seven entries in `KNOWN_VIOLATIONS` the two built-ins between them break
four, and `irregular_sampling` and `non_stationary_support` had never been broken by any
registered domain, so no refusal depending on them had ever fired against a declared source. Argo
breaks exactly those two, for reasons that are physically real — a float surfaces on a cycle it
does not keep precisely, and floats are deployed, fail and are replaced mid-record. It is also
the first declared domain in which **precedence is admissible while the clock is irregular**, it
is the only one exercising `lag_policy="declared"`, and it keeps a physical metric, which is the
side of the biconditional neither built-in occupies.

`onboarding_sha256` binds declaration, wording and geometry into one citable digest, so a result
attributed to a domain can name the exact contract it was read under rather than a name that may
since have been re-registered with different words. `onboarded_by` captures the calling module,
because `Entry.defined_in` records `value.__module__` — for a `DomainGlossary` that is always
`src.core.translation`, the class's home and never the adapter's.

**Claim boundary.** The contract checks a declaration for completeness and internal agreement. It
reads no data file, so it cannot know whether a domain's declarations describe the source it
names; it cannot know whether the wording chosen is wording a practitioner would use; and it
establishes nothing about which domain produced any given `EvidenceBundle`, because a bundle
still does not record one. `DOMAIN_ATTRIBUTION_CAVEAT` continues to travel with every served
limit. Rule R17's refusals remain enforced in the analysis layer; this makes the declarations
they read from complete, not self-enforcing.

### 3.6zn The ingestion seam (`src/api/channels.py`, TG8.4, `ed-dev`)

TG8.1 made a domain **declarable** from outside `src/`. A declaration is not a connection. Until
this slice there was no path from any file to any declared domain, so `/findings/domains`
described vocabularies and limits for data nobody could load, and the platform could ingest three
things — a local NetCDF, ERA5 crops from WeatherBench Zarr, and a fabricated field — all
atmospheric, one of them not data.

Most of what was needed already existed and was unreachable. `src/data_layer/tabular_source.py`
has read delimited channel records since TG0.2: it refuses a non-monotonic clock, refuses an
irregular one unless the domain declares `irregular_sampling`, refuses non-finite values, and
digests the bytes. It called `register_source` nowhere, was referenced **zero times** from
`src/api/` and `frontend/src/`, and its only importer was its own test. TG8.4 gives it a seam.

**Not registered as a `DataSource`, deliberately.** `register_source` wraps a
`can_serve(dataset_id)` / `fetch(...) -> xr.Dataset` protocol — an atmospheric grid-fetch
contract. A channel table returns a `ChannelSeries` and is uploaded rather than named. Forcing it
into `SOURCES` would misrepresent both and would put a CSV into the fallback chain that serves
ERA5. It gets its own router.

**Two calls, and the reason is the one `/import/inspect` already had.** That route exists because
an ERA5 file is `(time, level, lat, lon)` and picking `[0, 0]` on the researcher's behalf imports
a slice they did not choose while every statistic downstream describes that arbitrary timestep.
Here the arbitrary choices are the **clock column** and the **domain**, and neither is made for
the caller.

`POST /channels/inspect` reports what the file is — columns, rows, which columns could serve as a
clock, whether that clock is strictly increasing and regular — and converts those facts into
**obligations**, under the rule this slice is built on:

> **Detection may create a required declaration. It may never satisfy one.**

An irregular clock does not acquire `irregular_sampling`; it acquires a *requirement* that
whichever domain the reader picks already declare it. The report then names, for every onboarded
domain, whether it admits this file and the exact wording of the refusal if it does not — so a
refusal is visible before it is hit. `POST /channels/read` reads the record against a named
onboarded domain, or refuses by name, and a test asserts that the refusal `inspect` advertises
and the refusal `read` enforces are the **same** refusal.

**No column is substituted for a broken clock.** This was found while writing the tests, not
designed in: given a file whose `t` ran backwards, the first implementation quietly promoted
`bid` — a price column that happened to increase — to be the clock, and every lag reported
downstream would have described that substitution. Inspection now stops and names the candidates
instead. A caller who knows their clock is in column three passes `time_column` explicitly.

**Three refusals that had no enforcement before.** Two are new, one was prose:

| Condition | Basis |
|---|---|
| the domain's declared axes include a role a channel table cannot supply | E14 — a domain declaring latitude and a pressure level has declared a record richer than the file, and reading it here would silently drop axes its results are indexed by. `reanalysis` is therefore refused for a CSV, correctly |
| a channel is given a parent-axis footprint above one sample while the domain does not declare `aggregated_values` | E15/R17 — documented in `_resolve_supports` since TG0.2 and enforced by nothing. Declaring the footprint without the violation sets the lag floor correctly and leaves every other refusal that depends on it switched off |
| rows × channels above a declared ceiling | refused by name, **never thinned**. A record silently reduced to fit a response body is indistinguishable from one that was always that size |

**Stateless.** The series is returned and nothing is stored, exactly as `/import/field` returns a
field the browser then holds. A server-side store would be a second place a record could go stale
against the file it came from.

**Tab 12, *Domain Records*,** is a peer of the meteorological tab rather than a section inside it:
one tab reads grids, the other reads channel tables for any declared domain. The multi-domain
interface is a structural fact there rather than a claim in a document. The view formats no
scientific quantity — the clock facts, the refusals, the caveat and the preview note all arrive
as strings or booleans and are rendered as given, Phase G9's rule applied to a tab G9 did not
write. Row counts and raw data values are formatted, and neither is a claim.

**D61, found by this slice.** `order_book` had described *"an irregular trading clock"* in its
description from its first commit while omitting `irregular_sampling` from its violation tuple.
Four slices passed without anyone noticing, because nothing had yet tried to **read data** under
the declaration — which is exactly what R17 is for and exactly what an ingestion seam is for. The
declaration is corrected here, and the consequence is recorded rather than smoothed over: half of
what TG8.1 claimed Argo uniquely contributed was really a gap in an existing declaration, so the
plugin's docstring and the coverage test both now say `non_stationary_support` alone.

**Claim boundary.** Reading a file under a domain does **not** establish that the file came from
that domain; it establishes that the domain's declaration admits the file's shape. That is the
attribution gap `DOMAIN_ATTRIBUTION_CAVEAT` already records for findings, and the caveat is served
with every inspection and every read. The adapter never fetches — pointing a domain at a real
archive remains a separate, deliberate act, and no public dataset has been ingested. The preview
plot is a preview: nothing is mined, no claim exists, no rung moves (R22). And refusing a file is
not validating the data in it: finite, monotonic and regularly sampled says nothing about whether
the values are right.

### 3.6zo The gridded-store registry (`src/data_layer/stores.py`, TG10.1, `ed-dev`)

`zarr_source.CATALOGUE` was four ERA5 stores in a module-level dictionary whose value shape was
ERA5's own — `resolution_deg`, `cadence_hours`, `levels`. Standard E1 exists to forbid exactly
that shape, and the cost was not hypothetical: a fifth store could not be added without editing
`src/`, and the shape had nowhere to record which domain a store belonged to or what cropping it
cost. `GRIDDED_STORES` is a `Registry[GriddedStore]`, `register_builtin_stores()` puts the four
ERA5 entries in it eagerly and idempotently, and `CATALOGUE` survives as a **read-only mapping
view** over the registry so the provenance, overlap and reporting readers are untouched.

**What an entry is now obliged to say**, each because something went wrong without it:

| Field | Why it is required |
| --- | --- |
| `domain` | Rule R17: a domain that violates nothing is not a second domain. A catalogue listing stores without saying whose they are invites three entries reading as three domains — the exact false confidence R17 refuses. Registration resolves the name against `DOMAIN_DECLARATIONS` and refuses one nothing has declared, so a store cannot be selectable and *then* fail where its refusals were needed |
| `access` | One of `anonymous`, `credentials`, `local`. A deployment is told what a store will need before a request fails with a credential error |
| `vertical_dim` | Declared, with **no default**, for the reason `AxisSpec.role` has none (E14). `level` was assumed of every store because every store so far was ERA5 |
| `chunks` | `ChunkFacts` carries the figure, **how it was obtained**, and the date if a live inspection produced it. `not measured` may not quote a size, and a `live inspection` with no date is refused outright: an undated measurement of an archive that may rechunk describes nothing. D43 is what an unmeasured store treated as a known quantity costs, and its note is now attached to the store it is about |
| `note` | Kept from the old catalogue and still required. The 0.25 and 1.5 degree stores differ by more than resolution, and a catalogue that does not say so sends a researcher to the wrong one |

**The view is deliberately unwritable.** Every consumer of the old dictionary only ever asked
whether a store was catalogued and what its URI was; none wrote. Keeping it read-only means a
store cannot enter the catalogue without passing `register_store`, so the domain and chunk checks
cannot be sidestepped by assigning a bare dictionary — which is what one test was doing, and now
registers a `local` fixture store instead. `__getitem__` raises `KeyError` rather than the
registry's `UnknownNameError`, because `CATALOGUE.get(store)` on an uncatalogued name is a
supported question whose answer is "no": it is the path a raw URI takes.

**`CropSpec` is generalised on exactly one axis.** `vertical_dim` names the store's vertical
dimension — `level` for ERA5, `depth` for an ocean product — and `select()` applies the vertical
selection to *that* name instead of a hard-coded `"level"`. Before this, a depth-axis store read
through the ERA5 path selected no vertical subset at all and said nothing about it, because
`"level" in subset.coords` was simply false; the full depth axis flowed into the cache while the
manifest recorded the request. That silence is now a test. Nothing else about a region-and-time
slice needed generalising: `select` already tolerates either latitude ordering, resolves
`lat`/`latitude`, and refuses a meridian wrap rather than guessing.

**The content key does not move, and that was a decision.** `vertical_dim` enters
`canonical()` **only when it is not `level`**. Adding it unconditionally would change the key of
every crop ever materialised, orphaning the local cache and making every recorded provenance
record name a key that no longer resolves — a high price for a distinction that distinguishes
nothing, since `store` is already in the key and a store determines its own vertical axis. One
key is pinned to a literal in `test_stores.py`, taken by running the pre-TG10.1 module rather
than by writing down what the new code produced, so a future change to the canonical form is a
decision someone takes rather than a number someone updates. Older provenance records carry no
`vertical_dim` key and still replay, defaulting to `level`.

**Acceptance, executed literally.** `src/tests/store_plugin_example.py` registers a fifth store
on a `depth` axis in a file no core module imports; the test asserts it reaches
`GET /api/v1/data/zarr/catalogue` and that `zarr_source.py` and `main.py` are byte-identical
afterwards, the method `test_registries.py` already uses for the data-source seam. The example
declares `domain="reanalysis"` rather than a domain of its own, because a gridded store that
breaks no inherited assumption is a source and not a second domain (R17), and it declares
`method="not measured"` because nothing has opened it.

**Claim boundary.** A store in a catalogue is **not data ingested**, and no live fetch was run
for this slice. Registering a store is a declaration; nothing here opens a store, reaches the
network, or verifies that a recorded figure is still true — TG10.3 is the slice that makes
probing a recorded act and a precondition of registration. The generalisation reaches the
*selection and materialisation* path; the forecasting-specific cached reader (`RegionalRecord`)
still speaks in pressure levels and `level_hpa`, which is honest for its ERA5-only contract and
is not used to describe the GLORYS entry.

### 3.6zp Probing a store as a recorded act (`src/data_layer/store_probe.py`, TG10.3, `ed-dev`)

Probing already happened. Somebody opened the WeatherBench stores, read their chunk metadata
and wrote what they found into a prose note — that is how the catalogue came to say *"measured
51.1x amplification"*. A practice that lives in somebody's terminal history produces exactly
what this one produced: figures for the stores anyone happened to look at, silence for the
rest, and **no way to tell the two apart afterwards**. Defect D43 is a year-late discovery of
that shape; defect D62 is the same failure one level in, a per-chunk size written into a field
no inspection had ever filled.

**A refusal is a result.** `unreachable`, `needs_credentials`, `not_readable` and
`network_disabled` are things a probe can conclude *about a store*, and they are the things a
researcher most needs before planning around one. `probe_store` never raises for a store that
declines; it raises only for a fault in the request, such as an empty URI, because that is not
a fact about a store at all. The HTTP route follows suit and, unlike `/inspect`, returns 200
with a recorded `network_disabled` rather than 409 — a deployment that cannot reach a store
needs that written down, not raised.

**Nothing reaches the network by accident.** Importing the module opens nothing and registering
a store opens nothing; `probe_store` honours `SPECTRALEARTH_ALLOW_NETWORK` and records the
refusal when it is unset. The probe is a deliberate act whose *record* is what registration
then requires.

**The record.** `StoreProbe` is frozen and content-addressed. The digest covers what was
observed and the URI it was observed at, and deliberately **not** `probed_on` or `evidence`:
two probes of an unchanged store on different days describe the same store, and letting the
date into the digest would mint a new record every time anyone looked, turning a ledger into a
log. An earlier probe is never erased by a later one — a store that was hostile last year was
hostile last year. Records persist content-addressed, atomically, and never over an existing
file.

**`chunk_hostile` is three-valued and `None` is never folded into `False`.** "Nobody measured"
and "it is fine" are the two states D43 confused, and a store called friendly because nothing
looked is the failure this slice exists to stop. An amplification also may not exist without
the crop it is the ratio *for*: hostility is a property of a pairing, never of a store alone —
the same fixture store amplifies 1x for a request that lines up with its chunks and 30x for one
that straddles them.

**The registration gate**, which is where the slice has teeth. `register_store` refuses a store
whose `chunks.method` is anything but `not measured` unless it cites a probe; the probe must be
of *that store's* URI, or a probe of a friendly store would license a hostile one with a
reference that is present and checkable and about something else; and the entry may not quote a
figure the cited probe never saw, because D62 with a citation attached is D62 improved only in
appearance. `not measured` and a probe digest together are refused too: a store cannot both
cite a look and say nobody looked.

**Two kinds of evidence, and the difference is published.** `evidence="probe run"` means this
code opened that URI. `evidence="prior recorded inspection"` means a transcription of an
inspection run before this module existed — and all four ERA5 entries are of that kind.
Transcriptions are accepted because the alternatives were deleting four true records or
fabricating four live runs, and both are worse. They are counted by `transcribed_probes()` and
the count is served by `GET /api/v1/data/zarr/probes`, so the number can only fall where anyone
can see it. Two of the four carry only what the original note wrote down — a size and no shape,
and no date of their own — and that is left as it stands rather than completed by inference,
which is what D62 was.

**Claim boundary.** TG12.1 has now added two live public-archive metadata probes beside the four
ERA5 transcriptions. They record GLORYS structure and predicted cost and transfer no ocean
values. A probe does not validate the data in a store, and a store characterised as friendly is
friendly *for the crop that was stated*. The registration gate proves an entry cites a look, not
that the look is recent enough for a future acquisition or that the archive has not rechunked.

### 3.6zpa GLORYS registration and coordinate-only costing (`src/data_layer/glorys_store.py`, `zarr_source.py`, TG12.1, `ed-dev`)

TG12.1 is the first real store to enter through the registry seam. `glorys_store.py` loads the
two checked-in `StoreProbe` records and registers the anonymous Copernicus Marine
`geoChunked.zarr` asset as `glorys_phy_my_0p083deg_p1d`. The entry declares
`domain="reanalysis"`: a regularly gridded ocean reanalysis breaks no inherited analysis
assumption, so R17 permits it as a source and forbids presenting it as a second-domain result.
It carries the product/dataset identities, licence statement, daily 1/12-degree grid,
`elevation` axis, exact probe digest and crop defaults used by Acquire. Importing the extension
does not reach the network, and no module under `src/` imports `copernicusmarine`; that
pydantic-2 credential-minting client remains an isolated external command.

**Both layouts survive the decision.** The selected probe `ccdb0625e7e8fb1d` records
`geoChunked.zarr` at `(2081,1,16,16)` and 2.02x amplification for the exact 1993-1995,
20-degree, one-variable, surface-elevation crop. The rejected probe `f9a45764fcf52c2a` records
`timeChunked.zarr` at `(1,1,512,2048)` and 72.52x. Keeping the hostile receipt is what makes the
choice evidence rather than hindsight. Both are live anonymous metadata observations dated
2026-08-28; neither transfers or validates field values.

**D67 is closed at the cause.** `assess_access_pattern` formerly called `select(dataset, spec)`
and made dask construct a sliced graph over a `12227 x 50 x 2041 x 4320` variable merely to read
its sizes. It now applies the same xarray indexers only to one-dimensional coordinates, maps the
selected labels back to source positions and counts the exact chunk ids those positions touch.
The predictor therefore preserves partial-date and descending-axis semantics without building a
data graph, and distinguishes a selection contained in one chunk from an equal-width selection
that straddles two. A regression replaces the data-selection function with `MemoryError` and
proves costing still completes.

**D68 was found by making the declaration executable.** The registry already allowed an axis
named `elevation`, but `CropSpec.levels` and `ZarrCropRequest.levels` accepted integers only.
GLORYS's 50 coordinates are fractional negative metres, ending at
`-0.49402499198913574`; registering it without changing the request type would advertise a store
whose one-level crop could not be expressed. The field now accepts strict finite integers or
floats, preserving integer ERA5 JSON and its pinned hashes byte-for-byte, and exact fractional
selection is tested. Acquire reads extension-supplied defaults when the store changes rather
than sending the ERA5 850-hPa default to an elevation axis.

**Claim boundary.** This is metadata acquisition readiness, not ocean-data ingestion. No GLORYS
value was cropped, cached or analysed. The 2.02x result is specific to the sealed crop and does
not promise that a window crossing a 2081-frame boundary is equally cheap; the UI therefore
keeps Inspect in the path before materialisation. The atmospheric D43 gate remains open because
an affordable ocean layout does not acquire or cross-check ERA5.

### 3.6zpb Readiness that declares what it is about (`data_layer/regional_forecast.py`, TG12.1a, `ed-dev`)

A follow-on fix to TG12.1, found by using it: a researcher registered GLORYS, inspected a crop,
and asked what to do next. There is no next step in the UI by design - materialisation is
CLI-only, because a 51 GB transfer should not begin from a browser button and `Inspect` exists to
show the 2.02x first - but two things were wrong at that point rather than merely absent (D70).

`assess_manifest_readiness` is the T5.2 regional-forecast check rendered against every
materialised crop in the Acquire tab. It asks for five canonical ERA5 variables on the 850 hPa
pressure level, and it read the vertical selection as ``[int(v) for v in spec["levels"]]``. GLORYS
elevations are fractional negative metres, so `int(-0.49402499198913574)` silently became `0` and
the verdict came back `level_available=False` with all five variables reported *missing* - which
reads as a crop that nearly qualified. **The question does not apply to an ocean crop at all.**
The coercion is D68's defect living on in a consumer TG12.1 did not teach alongside `CropSpec` and
`ZarrCropRequest`, which is the third instance of one pattern: D62, D68 and D70 are all the
registry describing a store more confidently than the code behind it could deliver.

Levels are now compared as numbers with no coercion. The assessment derives `applicable` from the
crop's own declared vertical axis - absent means `level`, per D63 - so no store name is hardcoded
and any future non-pressure store inherits the behaviour. When it does not apply it says so in
words, stating that the variable and level rows describe what T5.2 requires rather than anything
this crop failed to supply, and the panel renders that instead of the verdict. `Inspect`
additionally says **before** the transfer that materialising is where this store currently stops:
the crop will be cached, content-addressed and reproducible, and no analysis path reads a crop
from this store yet. Every ERA5 field is unchanged, asserted in the same test.

### 3.6zpc Transform-derived acquisition planning (`data_layer/crop_planner.py`, TG12.1d, `ed-dev`)

A source probe and an analysis plan are deliberately different records. `StoreProbe` continues
to state only what was observed about a URI, its arrays and its chunks. A
`TransformSupportRequest` then names the registered transform, exact filter configuration,
level depth and recommendation policy; `plan_acquisition` combines those declarations with the
selected native coordinates and chunk metadata without reading a field value. This keeps a
research preference out of the source ledger while making the eventual crop decision measured.

Support belongs to the transform implementation. `TransformSpec` has an optional support
callback, and the built-in SWT and DTCWT callbacks call the same filter-support and valid-interior
functions used by their transforms. A third-party transform can therefore become plannable by
registering that callback, without an acquisition-layer edit; a transform with no support
contract is refused before a store is opened. The request and plan are content-addressed. Plan
identity binds the URI, structure, exact latitude/longitude coordinate bytes, crop, transform
configuration, thresholds and suggested bounds, but not field values.

Two thresholds answer two different questions. The **absolute minimum** leaves at least one
uncontaminated coefficient on each coarsest native axis and licenses technical computability
only. The **recommended minimum** applies the named
`r13-cross-scale-valid-interior/v1` policy: 128 uncontaminated parent-grid cells of span at the
coarsest level, translated through the transform sampling and rounded to a dyadic operational
crop. With the implemented filters, four-level DTCWT requires 240 x 240 absolutely and 512 x
512 for the research policy; four-level SWT/db2 requires 47 x 47 and 256 x 256. These are derived
results, not global 256/512 constants.

For an undersized request, native coordinate indices are expanded symmetrically, shifted at
source edges, and re-costed against real chunks. Inspect reports current, absolute and
recommended shapes, every level's support and valid interior, feasibility, exact suggested
bounds, revised transfer/amplification, plan digest and claim boundary. Acquire exposes the
transform/filter/depth controls and can apply the recommended bounds in one action. The generated
CLI preserves that exact analysis request, including `--analysis-levels`; explicit
materialisation refuses anything below the recommended threshold from metadata before a data
selection or transfer is constructed. Callers that omit analysis retain the legacy R13 path and
manifest shape.

**Claim boundary.** The recommendation certifies filter support plus a declared minimum span; it
does not prove statistical power, stationarity, physical relevance or that the eventual analysis
will find an effect. No public field value was transferred. TypeScript and the production build
pass, but the in-app browser runtime exposed no browser, so this new panel has not been visually
or assistive-technology inspected.

### 3.6zq Domain-first acquisition (`src/api/acquisitions.py`, `frontend/src/components/AcquisitionView.tsx`, TG10.2, `ed-dev`)

`GET /api/v1/acquisitions` projects the existing domain declarations, gridded-store registry
and channel-table admission rule into one domain-first catalogue. It is not a second source
registry. Every acquisition carries the selected domain's declaration, derived refusals and
`DOMAIN_ATTRIBUTION_CAVEAT`; a registered grid store appears as `grid_crop`, while
`channel_table` is available only when the atomic onboarding contract is complete and the
domain declares no axes a flat table cannot supply. The `profile_query` shape is declared but
has no implementation before TG12.2.

The frontend's ninth tab is now **Acquire**. It renders domains and acquisitions from this
catalogue, embeds `ChannelRecords` after a channel-table domain is chosen, and retains the ERA5
crop, probe, inspection, cache-readiness and materialisation-command workflow for grid crops.
The former twelfth Domain Records tab is removed, so adding a domain or store changes catalogue
rows and cannot add a tab. This is reachability, not acquisition evidence: no public archive
fetch or live WeatherBench probe was run, and selecting a domain does not establish that a
supplied record came from it.

### 3.6zr Workflow navigation and persistent context (`frontend/src/App.tsx`, TG11.0, `ed-dev`)

The shell groups its eleven destinations by the scientific workflow: **Acquire, Analyse,
Evidence, Review, Read, Platform**. The spatial generator, meteorological reader, boundary lab,
spectral transforms and diagnostics are explicitly labelled the **Gridded field line**; grouping
does not generalise them to channel domains. Review was initially an honest labelled waypoint;
TG11.5 now fills it with recorded argument in a workspace separate from Findings.

The selected channel record and study id are shell-owned context. Because the API response is a
bounded preview, `ChannelRecordSelection` retains the original browser `File`, chosen clock and
aggregate supports beside that response; the next analysis surface can therefore submit the full
admitted input without a second selection or silently analysing the preview. `ChannelRecords`
publishes that context upward, `AcquisitionView` restores its domain and channel-table view when
remounted, and `FindingsView` reads and changes the shell's selected study. A context
strip remains visible across panels and permits either selection to be cleared. This persistence
is browser-memory workflow state, not evidence persistence; it adds no scientific computation,
claim mutation or durable record.

### 3.6zs The analysis surface (`src/api/analysis.py`, `frontend/src/components/DomainAnalysisView.tsx`, TG11.1, `ed-dev`)

`domain_analysis` has offered `association_only`, `analyse_precedence` and `run_domain_gate`
since TG0.2, and until this slice **nothing could reach any of them**. TG8.4 could load an order
book and then do nothing with it. Two endpoints close that gap and add no science: `GET
/api/v1/analysis` describes the three operations and the claim boundary, `POST
/api/v1/analysis/run` executes exactly one of them. No estimator, correction, lag floor,
surrogate or verdict is implemented here; every one is the engine's, called unmodified, which is
the only way the HTTP result means what the in-process result means.

**The record is re-read, never the preview.** `/channels/read` returns a bounded preview, so
analysing what the browser already holds would be analysing a truncated record. TG11.0 retained
the original `File` for precisely this reason and the surface re-uploads it. A server-side record
cache was rejected as the alternative: it would be a second source of truth that can go stale
against the file the researcher believes they selected.

**The client declares the hypothesis, the server derives the record.** Cadence, frame count,
channel count, measure and source identity come from the re-read file; the family, estimator,
bins, surrogate count, alpha, correction and seed come from the researcher. That split is the
substance rather than a convention — a request body that could assert a cadence could assert a
lag family's physical duration, and the receipt would then describe a record that was never read.

**Four refusals, each before any computation.**

*   **R21.** A domain whose registered lag policy supplies no admissible floor cannot be asked
    for precedence, and the engine's own wording — naming the domain, the violation it declared
    and both remedies — reaches the caller intact rather than being re-phrased at the boundary.
    Association stays available and labels its own output as association.
*   **An unknown configuration key** is refused rather than ignored, per operation. An ignored
    family setting would make the receipt describe a different analysis than the one that ran.
*   **An irregular clock** is refused: a lag in frames is not a physical duration on one, and
    this surface will not invent a cadence to make the family expressible.
*   **A non-UTF-8 upload** is refused by name, pointing at the acquisition adapter, rather than
    being decoded leniently.

**Every response says what it did not do.** `stored: false`, `rung_moved: false`, and a claim
boundary stating that this is candidate compute — no evidence recorded, no rung derived (R22), no
preregistration and no held-out spend. Those begin at TG11.2 and TG11.3; the governing rule of
Phase G11 is that the interface may record evidence and may never assert a rung.

**Acceptance, through the HTTP layer rather than in process.** All **thirteen** `sequence` and
`cross_domain` benchmarks run through the live FastAPI test client via the existing
`POST /benchmarks/run` route — no duplicate benchmark path was added — and report 13 PASS, 0
FAIL, 0 NOT_YET_RUNNABLE with an empty `null_failures` list. **Eight of the thirteen are nulls**,
including `precedence_null`, `shared_cycle_precedence`, `slow_independent_precedence`,
`leaked_band_precedence` and `cross_domain_null`, and every check of every one of them answers
"there is nothing here". A surface that found the planted coupling and also found structure in
the AR(1) nulls would have found nothing (T4C.3), so the null half is the load-bearing half.

**The re-read is checked rather than assumed.** Re-uploading the file is the design, and an
unchecked re-upload is only a belief that the server read the same record the panel is showing.
The result header compares the returned `content_sha256` against the selected record's and
refuses the result outright when they differ, and - when the bytes do agree - compares the frame
count the analysis derived against the frame count `/channels/read` reported, because two readings
of one file that disagree on its length were not admitted the same way. Neither disagreement can
arise through today's controls, which is exactly why it would be invisible if it ever did.

**`INVALID` is named on screen, not styled as a leftover.** The panel's verdict chip first
matched `PASS` and `FAIL` and let everything else fall to one unlabelled colour. `INVALID` means
the design did not hold — a family that lost a member is a different experiment — and presenting
it beside FAIL's "there is nothing here" would report a design error as evidence of absence. It
now carries its own statement and its own branch, and an unrecognised verdict is no longer
painted as if it were one of the three.

**D64, found by mounting the router.** The documentation guard that refuses an undocumented
endpoint scanned route decorators with a `[^"]+` path pattern, so a route mounted at its router's
own prefix — `@router.get("")` — was invisible to it. `GET /api/v1/acquisitions` had been served
and unseen since TG10.2, and the documented count read 42 against a served 43 while the check
passed, because the claim was being compared with the guard's blind spot rather than with the API.
The quantifier is now `[^"]*` and the count is 45. See D64.

**Claim boundary.** Making the engine reachable is not evidence that it is correct. The thirteen
benchmarks argue for correctness on the thirteen cases they cover and on nothing else. Rendered
browser inspection of the new panel is **NOT RUN**; the contract tests prove it compiles, calls
routes that exist and reads fields that are present.

### 3.6zt Preregistration (`src/api/preregistration.py`, `frontend/src/components/PreregistrationView.tsx`, TG11.2, `ed-dev`)

**Six endpoints, and no new science.** `core/preregistration.py` is 574 lines - `PartitionIdentity`,
`Seal`, `report_generation`, `freeze_confirmatory_family`, `HeldOutLedger`, `confirm_on_held_out` -
that nothing outside the tests could reach. Every refusal, digest and correction here is that
module's. This is the wire boundary, and it exists because TG11.1 made it possible to look at a
record first and declare a family afterwards, which is the exact freedom R18 removes.

**The ordering is structural, not presentational.** A confirmation against a seal that does not
exist is a 404 before the record is read. A confirmation against a partition the seal did not name
is refused by `PartitionMismatchError`. A second confirmation against a spent partition is refused
by the ledger. None of that depends on which button the panel enables: an interface that enforced
the ordering by rendering would defeat the module while appearing to use it.

**The client cannot tune a confirmatory run.** `POST /seals/{digest}/confirm` accepts the record
and an optional published digest, and nothing else. Lags, ensemble size, alpha, correction,
estimator, bins, seed, split, domain, clock and delimiter are all read back out of the seal, where
they were frozen as the confirmatory specification's `notes` - so editing any of them breaks the
seal's digest rather than silently producing a different analysis under the same seal. A knob left
turnable after sealing is a family member chosen after the declaration.

**The sealing time is the server's.** A caller-supplied `sealed_at` could be written after the
partition was opened, and the time is the whole of what a seal claims.

**The confirmatory family narrows by lag, and only by lag.** `cross_scale_dependency` tests every
ordered channel pair, so the frozen label set is exactly what the sweep on the held-out partition
emits. A narrower family would leave the engine computing tests on held-out data that the
correction did not count - correcting at a smaller family than was actually tested. The axis is
declared over the record's own channel names rather than over `1..n`, because that is what the
sweep renders into a label; a family declared over positions would freeze labels the sweep never
emits, and every frozen member would come back missing at the one moment the partition may be
opened. `test_the_sealed_family_is_the_shape_the_sweep_actually_emits` pins the boundary's
specification to `GateProtocol.search_specification` label for label.

**A refusal never spends the partition.** `confirm_on_held_out` verifies the seal, matches the
partition and checks the label set before it writes the ledger, so a design error costs nothing and
a completed confirmation always costs the data. A test asserts both halves: a mismatched record is
refused and the partition is still spendable afterwards.

**TG11.1's gate is gated.** `run_domain_gate` splits internally and returns a verdict on its own
test partition, so running it *is* a test of that partition. The analysis surface now reads the
ledger before the gate runs and refuses with `409` if the partition is already spent. Nothing is
recorded by that check - the ledger is only read - and the store directory is deliberately not
created by a read, so an analysis that preregistered nothing leaves no trace.

**A seal is not evidence about itself.** The stored copy hashes to itself by construction; anyone
who can rewrite the file can recompute every digest in it, and nothing here is signed. Every
response that hands out a seal says so, `verify_published` is reachable without spending anything,
and the panel labels an unverified confirmation "self-consistency only".

**"Once" is per held-out data (D65).** Not per filename, not per domain, not per seal. See the
ledger entry: `PartitionIdentity.from_series` hashes provenance wholesale, and taking it unmodified
across an HTTP boundary would have let a rename buy a second look.

**Operationally.** Seals and the ledger live under `data/preregistrations`, overridable by
`SPECTRAL_PREREGISTRATION_ROOT` so a test never spends a real partition. This is programme state
rather than a cache: deleting it destroys the record of what has been spent. `HeldOutLedger` reads,
mutates and rewrites a JSON file without a lock, so "once" is once *per server* and a multi-worker
deployment needs a real store.

**What is still unreachable, named rather than left quiet.** `report_generation` - the check that a
candidate list was drawn from declared members of the generate specification, and that it was mined
on train rather than on the held-out partition - has no route. It needs a candidate list from a
real training sweep, which is a mining surface rather than a preregistration one, and inventing a
thin route for it here would have made the module look covered without the thing it guards existing
yet. It belongs with TG11.4.

**Claim boundary.** A seal is a promise about ordering, not a result: it says a family was fixed
before a partition was opened and says nothing about whether the family is any good. A confirmation
receipt records no evidence and moves no rung (R22) - writing it into a bundle is TG11.3. The
generated family is not corrected for anywhere here and its members are not claims. Rendered
browser inspection of the new panel is **NOT RUN**.

### 3.6zu The evidence write path (`src/api/evidence.py`, `frontend/src/components/EvidenceView.tsx`, TG11.3, `ed-dev`)

**The first surface that writes toward a claim.** Everything before it was read-only by
construction: `api/analysis.py` computes and stores nothing, `api/preregistration.py` stores an
ordering promise that records no evidence. This module writes into an `EvidenceBundle`, and a
bundle is the thing the claim ladder grades, so this is where rule R22 stops being a property of
the architecture and becomes a property of five handlers. No new science: every digest, chain
check and gate is `core/evidence.py`'s and `core/claim_ladder.py`'s.

**The rung is recomputed, never accepted.** No request model on this surface has a field for a
rung, a claim level or a confidence, and unknown fields are forbidden rather than ignored, so
`{"rung": "candidate_precursor"}` is a `422` rather than a key that silently does nothing. The
rung in every response is `assess_claim_ladder` run over the chain that was just written, and it
is recomputed per request rather than stored, because a stored rung is a figure that can disagree
with the evidence beneath it.

**The one payload key that could climb a rung by typing.** The ladder reads exactly one key out of
an evidence payload - `temporal_precedence`, on passing `provenance` entries - and it is the gate
for `candidate_precursor`. A free-form payload would therefore let a client assert its way up a
rung, with no arithmetic anywhere for an estimator to notice. The boundary refuses that key at any
depth of a hand-written payload and names the alternative: `POST
/studies/{id}/evidence/precedence` runs `analyse_precedence` server-side and records whatever it
returns, `false` included. The caller chooses the record and the lag family; it does not choose
the answer. The entry cites two digests - the record's `content_sha256` and the sweep's
`analysis_config_sha256` - so what was computed is checkable against what was recorded.

**An underpowered sweep is `INCONCLUSIVE`, not a negative.** If `check_power` says the family
could not have rejected anything after correction, the entry is recorded as `INCONCLUSIVE`, which
opens neither the provenance gate nor the precedence one. A `PASS` there would have let a sweep
that never looked stand in for a check that was made (R5).

**Appends are compare-and-swap, and revisions are files.** An append names the `head_sha256` it
believes it extends; a mismatch is a `409` and writes nothing. A bundle is immutable and
`save_evidence_bundle` refuses to overwrite, so each revision is published as its own file
(`{study_id}.r{NNNNN}.json`) created exclusively - which makes the exclusive create the
concurrency control. Two writers racing from one head produce one append and one `409` rather
than a lost entry. This is deliberately stronger than TG11.2's ledger, which takes no lock:
evidence is the thing being protected.

**Revision-per-file forced a read-surface fix (D66).** `StudyStore.load` returned the first file
whose `study_id` matched, and `summaries` listed one row per file. Under a write path that
publishes revisions as separate files, the read surface would have served revision zero of a
study for ever while the write path reported the revision it had appended, and one study worked on
five times would have listed as five studies. Resolution is now by chain rather than by filename:
the highest revision wins, and earlier ones are folded into its row with `superseded_revisions`.

**What it still cannot do.** It cannot delete, amend or reorder an entry - each entry carries its
predecessor's digest, so an edit is detectable rather than merely discouraged. It cannot record
commentary: the ten fields of `EVIDENCE_FIELDS` are the whole vocabulary and prose is not one of
them (R22). It cannot back-date: `recorded_at` is the server's clock and is not a field on any
request. And `refuse_bare_confidence` - TG9.1's wire guard - is applied to the payload on the way
in as well as the body on the way out, so a bare confidence is refused by name rather than by a
500 (R9).

**A causally worded hypothesis is registered, with the ceiling stated.** A statement containing a
word from `OUTSIDE_THE_LADDER` is not refused - it is a legitimate thing to want to test - but the
response says once, at registration, that no revision of the bundle can reach that wording,
because the ladder tops out at demonstrated predictive utility (R7).

**Operationally.** Bundles live under `data/studies`, the directory TG9.1 already reads, and the
root is overridable by `SPECTRAL_STUDY_ROOT` so a test never publishes evidence into a
researcher's store. This is programme state, not a cache.

**Claim boundary.** Recording evidence is not establishing a finding: the ladder grades what is in
the chain, and a chain of one favourable observation earns `observation`. A bundle carries no
domain, so nothing written here records which instrument the evidence came from. And this surface
does not consult the held-out ledger - running a precedence analysis over data preregistered as
held out spends it outside the record, which the ledger cannot see and the capabilities note
cannot prevent. Rendered browser inspection of the new panel is **NOT RUN**.

### 3.6zv Structure mining (`src/api/mining.py`, `frontend/src/components/StructureMiningView.tsx`, TG11.4, `ed-dev`)

**The largest body of unreachable capability in the tree, made reachable.** `core/motif.py`,
`core/constellation.py`, `core/family.py`, `core/invariance.py`, `core/motif_freeze.py` and
`core/motif_transfer.py` are 4,211 lines that nothing outside the test suite could call. Eleven endpoints reach them. No matcher, null, correction, tolerance or p-value is
implemented at the boundary; every number in every response is computed by the module that owns
it, and the boundary's whole content is *what it refuses to accept*.

**The client cannot draw a motif.** This is the R22-shaped risk of the slice and it is not about
rungs. A motif is a configuration of extracted features, so a surface that accepted feature
coordinates would let a caller type the shape they wanted confirmed - and every number computed
downstream would then be arithmetically correct and scientifically empty, with nothing for an
estimator to notice. No route accepts a feature. A caller admits a *field* - a 3-D `.npy` stack
of frames - and the server extracts, under settings that become part of the record's identity.
A record is addressed by the digest of its bytes **and** of the declaration they are read under,
so re-reading one array as another variable is a different record, and an edited sidecar no
longer hashes to the name it was stored under.

**The client cannot choose what counts as the same shape either.** The match tolerance decides
which configurations are repeats. Built the tempting way it came out at 0.41 against a correct
0.0083 in this tree's own benchmark - fifty times too wide, wide enough that every triangle
matched every other. `/generate` therefore takes a tolerance *digest*, never a number:
`/tolerance` calibrates one with `invariance.calibrate_match_tolerance` over frames the caller
declares to be replicates of one configuration. That declaration is the caller's and the server
cannot check it, which the receipt says in as many words. What the server does check is that the
calibration came through the same pipeline - domain, dataset, variable, units, representation and
every extraction setting - and, when it was measured on the record being mined, that it was
measured inside the training window.

**Frames of unequal feature counts are refused, not trimmed.** The declared family is
`scenes x C(n, k)` and scenes of different `n` have no such number. The obvious repair - keep the
brightest six - is named in the refusal rather than offered as an option, because magnitude
ordering moves between noise realisations, so "the brightest six" is a different configuration in
every frame and a tolerance calibrated across frames that disagree about *which* features they
hold measures that disagreement.

**Nothing measured inside a held-out frame leaves before it is spent.** `/records` reports how
many features each frame yielded and what the extractor rejected, which is admission geometry -
exactly what a `PartitionIdentity` has always carried about a channel table, namely how many
channels it has and what they are called. Nothing measured *from* a frame is reported: not a
position, not a scale, not an amplitude, and not the calibrated threshold, which was removed from
the per-frame rows for exactly this reason - the split is declared after admission, so a per-frame
measurement would have been on screen while the split was being chosen. A test asserts that no
held-out scene is named anywhere in a generation response, and another that no frame row carries a
measured quantity. The residual is stated rather than closed: feature counts have to agree across
frames or the record is refused, so the counts carry almost nothing, but a caller does admit first
and split afterwards. Binding the split into the record's own declaration - so that it is fixed
before extraction runs - is the stronger design and is not what this slice does.

**The confirmatory run is read out of the seal.** TG11.2 established that for lag families; this
extends it to mining. `/confirm` takes a seal digest and an optional published one, and nothing
else: the record, the split, the size, the matcher, the tolerance, the ensemble, the correction
and the seed are sealed as notes on the confirmatory specification, so an edit to any of them
breaks the seal's digest instead of quietly producing a different analysis under the same name.
The candidates are re-derived by re-running the same deterministic mining pass - the enumeration
is combinatorial and randomness enters only at the surrogate ensemble - and `confirm_motifs`
checks the labels against the ones the seal froze, so a redefined motif is refused rather than
confirmed under an old name.

**One core change, made beside rather than instead (E12).** `motif.confirmatory_specification`
and `motif.freeze_motifs` gained an optional `notes` mapping, merged beside the two notes they
already write and therefore inside the specification's fingerprint. Without it the run settings
would have had to live in a file next to the seal, which is a setting the seal does not bind.
No existing caller sees a change.

**Seals and the ledger are shared, not duplicated.** A mining seal is a `Seal` and goes into
TG11.2's store, so `GET /preregistration/seals` answers "what has this programme frozen" rather
than "what has this surface frozen". The held-out ledger is TG11.2's ledger for the same reason:
"this partition has been opened" is one fact about the programme, and two ledgers would let the
same data be spent once on each. `held_out_ledger`, `store_seal` and `load_seal` were exported
from `api/preregistration.py` to make that sharing explicit rather than incidental.

**Transfer is the ordering, and the ordering is the record (R20).** `/motifs` publishes one
frozen exemplar as a durable definition carrying its origin domain and that domain's licence -
a definition that crossed domains having forgotten where it came from is the erasure R17 exists
to prevent. `/transfer` commits the binding to the transfer ledger *before* the target frames
are read, so a failure after that point still spends the target, which is the honest accounting:
a caller who saw an error after the opener ran has still seen the target. Search takes no size,
matcher, relation or tolerance argument; all four come from the verified `FrozenMotif`.

**Why the server's clock is made strictly increasing.** A transfer record is refused unless
`frozen_at < bound_at < opened_at`, and that ordering is the entire content of the record. A
Windows clock ticks about every 15 ms, so two events that really did happen in that order can be
issued one timestamp, and a correctly ordered transfer would then be refused for a reason about
the clock rather than about the science. An instant that would repeat or go backwards is advanced
by a microsecond. Nothing waits and nothing is back-dated: the ordering reported is the ordering
that happened, at a resolution the clock does not have.

**The invariance audit is reachable, and it is not a gate.** `/invariance` measures every
registered matcher against its own declaration, and both directions fail: a matcher that
overclaims makes every cross-scene match suspect, and one that understates makes a caller reach
for a heavier matcher it did not need. Which transform each presentation underwent is the
caller's declaration, because only whoever produced the frames knows. Invariance is necessary for
recognising one configuration in two scenes and nowhere near sufficient - a constant signature is
perfectly invariant to everything and matches everything - so what separates a matcher that
measures something from one that does not is the surrogate null in `/confirm`, not this audit.

**Operationally.** Admitted fields, calibrated tolerances, published definitions and the transfer
ledger live under `data/mining`, overridable by `SPECTRAL_MINING_ROOT`. This is programme state,
not a cache: a seal, a frozen motif and a confirmation receipt all name a record digest, and
deleting the record makes those receipts uncheckable. Extraction is cached per process because it
is a pure function of an immutable record, and `/confirm` re-derives the training candidates in
order to check them against the seal.

**Claim boundary.** Mining produces candidates. Support on the training frames is selection and
not evidence: every exemplar is one of the occurrences it is counted among, so its support starts
at one by construction, and it was ranked highly for having been counted often. A confirmed motif
is a configuration that recurred on frames it was not mined from more often than the surrogate
null placed it there - not a mechanism, not a cause, and not a claim until something records it,
which is TG11.3's write path (R22). A motif reported `vacuous` was confirmed by an ensemble that
could not have rejected it, which is not a confirmation (R5). A transfer match count is
descriptive and is not a corrected transfer result. Rendered browser inspection of the new panel
is **NOT RUN**.

### 3.6zw The cross-domain record (`src/api/cross_domain.py`, `frontend/src/components/CrossDomainRecordView.tsx`, TG11.4b, `ed-dev`)

**The module that makes a lag stop being a frame count.** `core/cross_domain.py` is 505 lines
that only the benchmarks could reach. Everywhere else in this programme a lag family is declared
in frames of one record's clock, which is honest exactly as long as there is one clock. Two
domains sampled hourly and three-hourly have two, and a family declared in frames of either is a
family the other domain cannot read. Seven endpoints on `/api/v1/cross-domain` are where a lag is
declared in **seconds** and converted per domain: capabilities, `align`, `lags`, `partition`,
`generate`, `seal`, and one `confirm` addressed by seal digest. No estimator, null, correction or
lag floor is implemented at the boundary.

**It will not resample.** The two records are aligned by exact timestamp intersection. Where they
share too few observations, or their intersection is irregular, the request is refused and the
refusal names what it is declining: interpolation would manufacture values at times one source
never observed and then let those manufactured values vote in a lagged relationship. Every
response reports how many native observations each domain retained and discarded, because a join
that quietly kept a third of one record is a different study from the one that was declared.

**It will not default a unit (R19).** A channel table carries column names and numbers; it does
not carry what its columns mean. Each side's reading must therefore declare `semantics` and
`units` for every column in the file, and a table with an undeclared or over-declared column is
refused rather than read under `"unknown"`. The statistic is dimensionless and compares no raw
magnitude - what keeps that honest is that both magnitudes still say what they are, all the way
into the confirmatory receipt, where each relationship carries its driver's and its driven
operand's semantics and units beside the lead in seconds.

**A duration is refused, never rounded.** `/lags` converts the declared family onto the exact
common cadence. A duration below either domain's physical floor - each converted from that
domain's own registered lag policy *before* the clocks were combined - is refused rather than
dropped from the family, and one that is not a whole number of common frames is refused rather
than rounded to the nearest, because a rounded lead is a lead the clock cannot express. The
family is priced before any of it is measured, and it contains only ordered pairs that cross the
declared boundary: a within-domain member would spend correction power on a question this surface
does not ask.

**The confirmation cannot be tuned.** As in TG11.2 and TG11.4, `/confirm` reads the domains, the
columns, the delimiters, the declared units, the lag family in seconds, the split, the embargo,
the ensemble, alpha, the correction and the seed back out of the seal; the caller supplies the
two records and, optionally, the digest they published. Those records are checked twice over, by
the seal's own digest - which an edit breaks at load - and by the held-out partition identity,
which a different pair of files cannot reproduce. Both refusals happen before the ledger is
written, and the published digest is compared before anything at all is read, so a refusal costs
nothing. The frozen members are then re-derived by re-running the deterministic training sweep
rather than reconstructed from their labels: a reconstruction built from a label matches that
label by construction, and would let a later change to the selection rule confirm a stale family
under its old names.

**One additive core change (E12).** `precedence.confirmatory_specification` and
`precedence.freeze_precedence` take an optional `notes` mapping, merged beside the notes they
already write, so this surface's run settings sit *inside* the confirmatory specification's
fingerprint rather than in a file beside the seal. No existing caller sees a change. The seals
land in TG11.2's store and spend TG11.2's one held-out ledger, for the reason TG11.4's do.

**Identity, and one residual (D65).** The held-out partition identity is the aligned record's
own, which hashes its provenance wholesale. That is safe here because every field in that
provenance is one this module put there - the two content digests, the two clocks, the shared
clock digest, the retained and discarded counts, the split window and the embargo - and the
filename is not among them, so the same held-out bytes re-uploaded under another name are the
same partition. Each side's `dataset_id` is its content digest for the same reason. What *is*
among them is each domain's declaration, and that is the residual: the same two files aligned
under two different registered domains produce two partition digests, and the ledger would let
them be opened twice. It is recorded rather than papered over - stripping the domain out of the
identity would make two genuinely different analyses collide - and it is stated in the claim
boundary every response on this surface returns.

**Claim boundary.** A cross-domain result is a temporal association between structural series:
one domain's series carries information about another's later series. The two records' raw
magnitudes keep different semantics and units and are never compared, and precedence identifies
no causal mechanism (R19, R21). A confirmation receipt records no evidence and moves no rung
(R22); it is an input to TG11.3's write path exactly as a single-domain precedence receipt is.
Rendered browser inspection of the new panel is **NOT RUN**.

### 3.6zx Accessibility as a workflow contract (`frontend/src/App.tsx`, `index.css`, `Heatmap2D.tsx`, `LineChart.tsx`, `LineageGraph.tsx`, TG11.6, `ed-dev`)

**The debt was interaction, not decoration.** The legacy panels had labels placed visually beside
controls but not bound to them, removed the browser outline without supplying one consistently,
left focus on the navigation control after changing workspaces, and made the disconnected-backend
retry a clickable `span`. The application now has a first-focusable skip link into a named main
landmark; changing workflow moves focus to a programmatic workspace heading; the current workspace
and global asynchronous state are announced; and the retry is a native button. Every legacy
control in `App.tsx` has an explicit `id`/`htmlFor` pair, including the experiment JSON editor and
every range input whose visible value changes.

**One focus rule covers both platforms.** `index.css` supplies a three-pixel high-contrast
`:focus-visible` outline for links, buttons, form controls, summaries and explicit tab stops. It
comes after Tailwind so the old `focus:outline-none` utilities cannot erase it. A reduced-motion
media query collapses animations and transitions without changing any scientific state. The
shell and each acquire/analyse/evidence/read surface expose `aria-busy`; errors are alerts and
non-error progress is status, so waiting and failure are not conveyed by colour or animation
alone.

**Scientific graphics keep a textual door.** Heat maps and line charts are figures with labelled
captions that report shape, series, units, axes, logarithmic scales and validity insets. The SVG
lineage nodes remain spatially arranged, but each is now a named, pressed-state keyboard control
activated by Enter or Space. The cross-domain pair is two fieldsets rather than one visual label
over eight unnamed controls, and the older field import and evaluation receipt controls are
programmatically bound.

**What is verified.** Six TG11.6 contract tests guard skip/focus routing, legacy label bindings,
the global focus and reduced-motion rules, the retry and SVG keyboard paths, figure text
equivalents, and busy/alert/status semantics across the workflow. With TG11.5 added,
`test_frontend_contract.py` passes 92 tests and the production build transforms 1,395 modules
with real JS/CSS assets.
Rendered keyboard and screen-reader inspection was attempted through the configured in-app
browser, but the runtime reported no available browser backend; it remains **NOT RUN**, so this
slice establishes source semantics and build integrity rather than WCAG conformance.

**Claim boundary.** Accessibility metadata does not validate the scientific content it names.
A text equivalent describes what the application knows about a figure; it does not independently
interpret the plotted result. No conformance level is claimed without an assistive-technology and
rendered-browser audit.

### 3.6zy Recorded review, outside the claim surface (`src/api/reviews.py`, `frontend/src/components/ReviewView.tsx`, TG11.5, `ed-dev`)

**Read, do not rerun.** `GET /api/v1/reviews/studies/{study_id}` is the review layer's only HTTP
route and its only verb is GET. It cannot create a panel, call a model, append evidence or accept
a claim state. It reads `ReviewRecord`, `RoundRobinOutcome` and `ReviewCostReceipt` artifacts from
the dedicated `SPECTRAL_REVIEW_ROOT` (`data/reviews` by default), classifies them by their declared
schema rather than their filename, and reconstructs each through its core type so every content
digest is checked again on read. Unknown, malformed and tampered artifacts are reported by
filename rather than silently disappearing.

**Four bindings prevent stale commentary from looking current.** The route first resolves the
latest immutable bundle revision through `StudyStore`. A review must match its study id, bundle
digest and bundle revision; an outcome must additionally name that review-record digest; and a
cost receipt must name it too. Commentary over an earlier evidence revision is therefore not
served under the newer one merely because both files occupy the same directory. An exact revision
with no review returns an explicit absence note: no recorded review is not evidence that nobody
reviewed it or that no criticism exists.

**The layout is part of the boundary.** Review is its own workflow workspace, not a panel inside
Findings. The amber R23 declaration and R22/R23 claim boundary precede every record. Complete core-
rendered calls and round-robin outcomes are shown as recorded argument, including retained
dissent; cost receipts show the provider route's recorded token counts and digest, with no dollar
price and no suggestion that cost measures review quality. The shared study id is navigation
context only. There is no vote count, consensus badge, promotion control or action that can run a
review.

**What is verified.** `test_reviews_api.py` has eight tests for the empty state, the complete
record/outcome/receipt surface, exact-revision binding, record-digest linkage, corrupt-artifact
reporting, 404 behavior, GET-only routing and byte-identical bundle reads. Six additional frontend
contracts guard routing and separation from Findings, the GET-only client, the visible R23 fence,
complete argument/cost rendering, honest empty states and accessibility semantics.
`test_frontend_contract.py` passes 92 tests and the production build transforms 1,395 modules.
Rendered inspection was attempted through the configured in-app browser, whose runtime reported
no available browser backend, so it remains **NOT RUN**.

**Claim boundary.** A recorded argument is an observation of what a non-deterministic process said
once. It is not reproducible computation, evidence, consensus or permission to claim; deleting it
changes no claim level (R22, R23). A cost receipt proves only the recorded route and token
accounting, not that the argument was good.

### 3.6zz Irregular profiles (`src/data_layer/profiles.py`, `src/data_layer/profile_reductions.py`, `src/data_layer/argo_source.py`, `src/api/profiles.py`, `frontend/src/components/ProfileAcquisition.tsx`, TG12.2b-d, `ed-dev`)

**The acquisition of asynchronous profile data.** Completes the TG12.2 program by adding `ProfileSpec` to content-address non-gridded profiles in a given region, time window, and depth range (`src/data_layer/profiles.py`). It enforces that non-stationary support is preserved or purposefully aggregated through explicitly declared reduction methods (`src/data_layer/profile_reductions.py`), and executes a bounded real query against the Argo GDAC (`src/data_layer/argo_source.py`).

**API and UI contract.** `src/api/profiles.py` and `ProfileAcquisition.tsx` expose the bounded acquisition. The interface properly renders the declared irregularities (such as `irregular_sampling` and `non_stationary_support`), and the readiness checks honor the exact characteristics of the profiles instead of making grid assumptions.

### 3.6zza Celestial photometry (`src/data_layer/lightcurves.py`, `src/data_layer/tess_source.py`, `src/api/lightcurves.py`, `extensions/tess_lightcurve.py`, `frontend/src/components/LightCurveAcquisition.tsx`, TG13.1, `ed-dev`)

TESS is a third acquisition shape, not a grid crop or profile scatter. `LightCurveSpec` binds an
exact TIC identifier, one to eight sectors, flux product, quality policy, product cap and byte
cap. `inspect_tess_query` resolves the TIC position, queries only the declared sectors, selects
exact-TIC calibrated SPOC `LC` products by archive URI and reports their declared bytes without
opening a FITS value. Acquisition is network opt-in and cannot start unless both preflight caps
pass.

`LightCurveCollection` validates a finite, strictly increasing BJD_TDB clock and aligned flux,
error, quality and sector arrays. Its digest includes every admitted sample, every product
record and the SHA-256 of every downloaded FITS payload; equal-length light curves with one
changed flux value therefore cannot collide. FITS checksums, TIC identity, time reference and
target coordinates are checked before publication through the same atomic no-overwrite primitive
as the other immutable collections. Quality flags are retained rather than silently dropped.

The `tess_lightcurve` extension declares `no_natural_cycle`, `irregular_sampling` and
`non_stationary_support`, plus `lag_policy=none`. Its `angular_sky` point geometry records a real
great-circle metric in degrees while explicitly declaring `grid_compatible=false`; it is excluded
from raster coordinate recognition and refuses differential-grid use. `GET /api/v1/lightcurves`,
`POST /api/v1/lightcurves/inspect` and `POST /api/v1/lightcurves/acquire` expose the contract and
its refusals. The UI cannot enable acquisition until metadata preflight succeeds.

### 3.6zzb File-first sample ingress and representation audit (`src/data_layer/dataset_ingress.py`, `src/api/ingress.py`, `frontend/src/components/GenericIngress.tsx`, G14.1-G14.2, `ed-dev`)

The Acquire surface now begins with **Load a file. Let's analyse it.** The first adapter accepts a
bounded UTF-8 CSV/TSV sample table. `probe_delimited` reports shape, storage type, missingness,
distinct counts and monotonicity, and returns `semantic_inference=false`: it never promotes a
column because it is named `target`, `time`, `lat` or anything else. A separate
`SampleTableDeclaration` requires an explicit role for every column, an explicit independent,
grouped or ordered sample relationship, and units (including the deliberate word
`dimensionless`) for every numeric target, nuisance and feature.

Independent samples do not enter `ChannelSeries` and acquire no invented clock. The first
`Representation Audit` recipe operates directly on the declared sample table. Its immutable plan
binds the exact file bytes, roles, units, target, complete identity/PCA candidate family, PCA
component count, bins, split ratio, seed, alpha, correction and permutation count before any
association is enumerated. An under-resolved permutation ensemble is refused when its minimum
attainable p-value cannot survive Benjamini-Yekutieli correction for the declared family.

PCA is fitted only on the generate partition and applied unchanged to confirmation. Every raw
feature and declared component is tested on both partitions; nothing is selected on generate and
then omitted from the confirmatory correction. Target permutations are fixed per candidate and
partition. Nuisance splits use a median learned on generate and are reported as descriptive
held-out stability only, explicitly not conditional mutual information. Responses state
`stored=false` and `rung_moved=false` and call survivors candidates rather than certified
invariants, causes or recommendations. The endpoints are `POST /api/v1/ingress/probe`,
`POST /api/v1/ingress/capabilities`, `POST /api/v1/ingress/plan` and
`POST /api/v1/ingress/audit`. The first audit recipe is deliberately restricted to independent
samples. Grouped declarations are routed to an explained group-held-out requirement and ordered
declarations to a blocked/embargoed requirement; either is refused before a row-random plan can
be sealed.

### 3.6zzc Dataset-bound capability routing (`src/core/dataset_capabilities.py`, `frontend/src/components/DatasetCapabilityProfile.tsx`, G15, `ed-dev`)

Acquire now produces one `spectral.dataset-capability-profile.v1` projection for every declared
sample table, admitted channel record, planned gridded crop, acquired profile reduction and
acquired light-curve collection. The profile is hashed over the exact file, request, collection
or reduction identity, its phase, domain, facts, operation decisions and their bases. A domain is
an upper bound: record facts such as irregular cadence or insufficient transform support may
remove a path that the domain could otherwise admit.

Operations declare requirements in one backend registry. Decisions are `available`,
`unavailable`, `needs_declaration` or `insufficient_support`, with a stable reason code and a
researcher-facing explanation. The shell reads those decisions to gate cross-domain analysis,
Boundary Lab, spatial transforms, gridded diagnostics, structure mining, cross-domain records and
forecast evaluation. Disabled navigation remains visible and renders the backend reason; it is
never a silent grey control. The profile card separately shows yes/no/not-established facts.

This is guidance rather than a security boundary. Every scientific endpoint retains its own
authoritative refusal, so editing browser state cannot license an inadmissible computation. No
column name is interpreted: generic-file profiles are derived only after the researcher supplies
roles, units and the sample relationship.

### 3.6zzd Representation-structure admission and benchmark contract (`src/benchmarks/representation_structure.py`, `dataset_ingress.py`, TG16.0, `ed-dev`)

The first G16 recipes have one shared authoritative admission rule: only a declared
`independent` sample relationship is accepted. Grouped samples name group-held-out confirmation
with benchmarked nulls as the missing contract; ordered samples name blocked and embargoed
confirmation with benchmarked nulls. The existing G14 representation plan now uses the same
function, so a later G16 recipe cannot weaken the refusal by copying it.

Two paired sample-table benchmarks freeze nine cases before any G16 estimator exists. The planted
half contains an exact duplicate, distinct noisy copies, complementary candidates, an XOR pair
that is weak individually, and signal that survives conditioning. The safeguard half contains
independent features, nuisance-only marginal association, a conditional null, and a collider
whose conditioning-induced association must never be described as nuisance removal. Each case
uses its own seed-derived stream so adding one fixture cannot change the others.

The acceptance contract fixes 200 replications, alpha 0.05, a maximum null rejection rate of
0.075 and minimum planted-effect detection rate of 0.80 for later operation-level gates. TG16.0
verifies the constructions with elementary independent oracles; it does **not** implement or
validate redundancy, conditional-information or stable-subspace estimation. The premature
`association_redundancy` capability added during G15 has therefore been removed. No G16 operation
is advertised until its backend recipe, refusal, null calibration and planted power check exist.

### 3.6zze Candidate redundancy structure (`src/analysis_engine/representation_structure.py`, `dataset_ingress.py`, `src/api/ingress.py`, TG16.1, `ed-dev`)

The first G16 operation is `redundancy_structure_audit`. Its immutable plan is bound to the exact
file bytes and declaration and freezes every unordered pair from two to six raw features. Group
size is exactly two in this bounded recipe. Each pair pays for three hypotheses: positive
interaction information and the conditional information increment of each member beyond the
other. Equiprobable bins, Miller-Madow entropy correction, conditional nulls, permutation count,
seed, alpha and Benjamini-Yekutieli correction are all sealed before enumeration. At least five
rows per possible three-variable joint cell are required, and a permutation ensemble unable to
survive the full correction is refused before computation.

The redundancy null shuffles one candidate within target bins, preserving both candidate/target
marginals while breaking their remaining arrangement. Each conditional-increment null shuffles
the target within bins of the other candidate. Exact candidate identity is also recorded as
deterministic structural evidence. A pair is `supported_redundancy` when identity or corrected
positive interaction evidence supports it, `supported_complementarity` only when both corrected
conditional increments survive, and otherwise `unresolved`. This ordering means noisy copies are
not renamed complementary merely because each noisy measurement adds a small increment. The XOR
control takes the other path: its singleton information is weak but both joint increments survive,
so it remains visible as supported complementarity.

This is deliberately not a partial-information decomposition. Interaction information is not
reported as the number of independent information pieces, and no outcome removes, selects or
recommends a feature. `POST /api/v1/ingress/structure/plan` and
`POST /api/v1/ingress/structure/audit` expose the content-bound workflow; both retain the shared
independent-sample refusal. The operation entered the capability registry only after the paired
benchmarks ran the frozen 200-replication family. Exact-duplicate/noisy-copy power is 1.00/0.94,
complementary/XOR power is 1.00/1.00, the independent false-claim rate is 0.02, and every
one-candidate safeguard makes no pair claim. The focused gate reports 4 PASS, 0 FAIL, 0
NOT_YET_RUNNABLE.

### 3.11 Ground-Truth Benchmark Suite (`src/benchmarks/`)

Added in T3.5.17 (standard E7). Twenty-two synthetic datasets whose correct answer is known
*before* analysis, of which **twelve are null benchmarks** whose answer is "there is nothing
here". This is distinct from `synthetic_generator/`, which exists to keep the UI alive
offline and declares no truth.

*   `core.py` - registry, `Benchmark` (build + known answer + gated stages + checks), and a
    three-valued `Outcome`. `NOT_YET_RUNNABLE` is never folded into `PASS`, so "all green"
    cannot come to mean "we never looked".
*   `seeding.py` - `SeedSequence.spawn` derivation from a root seed and a **`zlib.crc32`**
    label hash (Python's `hash()` on a string is salted per process and would break
    cross-session reproducibility).
*   `fields.py` / `sequences.py` / `cross_domain.py` - the twenty pre-G16 datasets, including
    `advected_vortex_periodic_sequence`, added in TG2.3 so that a benchmark declaring a
    torus draws one (defect D59), and `planted_motif` / `motif_null`, the TG3.5 pair that
    differ only in whether anything was planted, and `planted_precedence` /
    `precedence_null`, the TG4.1 pair that are one builder at its two ends - joined in TG4.2
    by `shared_cycle_precedence`, `slow_independent_precedence` and `leaked_band_precedence`,
    three further settings of that same builder which carry the other three of the five
    refusals (Section 3.6x); and `planted_cross_domain` / `cross_domain_null`, TG4.3's paired
    records at coupling one and zero, carrying different units, semantics and native clocks
    through one exact-clock, held-out relationship pass (Section 3.6y).
*   `representation_structure.py` - TG16.0's paired
    `representation_structure_planted` / `representation_structure_safeguards` sample-table
    family and the calibration/power thresholds later G16 operations must meet before
    registration; TG16.1 adds the first operation-level calibration gate to both datasets.
*   `runner.py`, `__main__.py` - report and CLI (`python -m src.benchmarks`, exit 1 on any
    failure, usable directly as a CI gate).

Current status: **33 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE**. See Section 7.2f.

### 3.13 Cloud-Native ERA5 over Zarr (`src/data_layer/zarr_source.py`, T3.5.18)

**Real observational data, verified against the live archive.** WeatherBench 2 publishes ERA5
as public Zarr on Google Cloud Storage, readable anonymously — 24 stores enumerated at
`gs://weatherbench2/datasets/era5` on 2026-08-20. Four are catalogued here with a note on what
each is good and bad *for*, because the difference between them is not resolution alone.

**The chunk-hostility trap, measured on the real store rather than predicted.** The 0.25 degree
stores are chunked `(1, 13, 721, 1440)` — one timestep, all 13 pressure levels, the whole
planet, **54.0 MB per chunk**. A 257x257 four-level crop wants 1.05 MB per timestep. Live
measurement:

| | value |
|---|---|
| Crop | 257x257, 4 levels, `temperature`, 8 days at 6 h (32 frames) |
| Predicted transfer (from chunk metadata, before fetching) | 1,727.6 MB uncompressed |
| Measured wire transfer | **951.3 MB** in 186.3 s |
| Amplification | **51.1x** (79.0 GB would be needed for 1.55 GB over one year) |
| Local cache size | 19.0 MB |
| Cache read | **8 chunk reads, 0.05 s** — against 186 s remote |
| Repeat request | **0 bytes, 0.006 s** |

`assess_access_pattern` computes that 51.1x from chunk metadata alone, transferring nothing, so
the warning arrives *before* the download rather than as an explanation afterwards. It also
reports the non-obvious consequence: **selecting fewer levels does not reduce transfer**,
because level sits inside the chunk. Only the variable and time axes actually narrow the wire.

**Two stages, the second not optional.** Query the remote store lazily, then materialise once
into a local Zarr cache **rechunked time-contiguous** for that region, content-hashed, with
bytes transferred recorded. The inversion is the entire point: the remote layout is one
timestep per chunk, the cache is all frames in one chunk, and a cross-scale read of a region
becomes a single seek.

**R13 enforced, not documented.** `edge_exclusion(j) = floor((1 + (L-1)·(2^j-1))/2)` uses the
support of the complete inherited low-pass cascade, not merely the filter applied at level `j`.
For a 14-tap filter it gives 7/20/46/98 px per side at levels 1–4, and
`minimum_crop_size` returns 512 for four levels and 1024 for five when 128 valid pixels are
required. A crop below
the floor is **refused**, naming the minimum and the measured valid interior, because a 64x64
crop has *zero* valid interior at level 4: cross-scale analysis on it is not noisy but
arithmetically impossible, and every edge coefficient looks exactly like a strong, localised,
oriented feature — which is precisely what a discovery engine would report. The refusal also
says which dimension to give up instead, since R13 is explicit that the grid is never it.

**Provenance that is a reproduction recipe (E5).** A `CropSpec` is frozen, hashable and
machine-independent: no local paths, no timestamps. Its `content_key` is order-independent, so
two researchers typing the variables in a different order share one cache entry. The content
hash covers the *data* and is independent of cache chunking — verified by materialising the same
crop at two different chunk sizes and getting the same hash. `rematerialise_from_provenance`
rebuilds a crop from the lineage record alone; verified against the real crop, key and hash both
matching.

**Network access is opt-in** (`SPECTRALEARTH_ALLOW_NETWORK=1`). Reaching the internet must never
be a side effect of running a sweep or a test: it makes results depend on connectivity, and a
mistyped bounding box against a 0.25 degree store moves tens of gigabytes. A **cached crop makes
the source available offline**, which is the payoff of stage 2 — `can_serve` consults the cache,
not the network.

Registered in the fallback chain at priority 20, between `netcdf_local` (10) and `simulated`
(900), exactly where `builtin_sources.py` anticipated it. `sources.py` had promised that
`SOURCES.with_capability("streaming")` was how this adapter would be selected without editing a
dispatch chain; until this slice that query returned an empty list, so the seam was a claim.
It is now a fact, with a test asserting it.

The catalogued 0.7-degree store is **not** the long-record regional solution previously
implied by its "compromise" label. Live metadata inspection on 2026-08-21 found chunks of
`(8, 13, 512, 256)` for temperature: eight frames but all levels and the whole globe. A
three-year, one-variable 255x255 request is estimated at **29.88 GB fetched for 1.14 GB wanted
(26.2x amplification)**. With independent train/test transfer-entropy partitions needing a
multi-year record, this is outside the declared laptop tier. Logged as D43; no T4C.6 real-data
verdict exists until a spatially tiled temporal source or direct regional acquisition path is
implemented and verified.

It declares `crop_dataset_ids` rather than `dataset_ids`, and the distinction is load-bearing:
`list_datasets` promises that every id under `dataset_ids` appears with concrete variables, a
time range, a bounding box and a resolution. A crop *family* has none of those until a crop is
specified — the archive is 64 years of the whole planet — so listing it there would advertise a
dataset `/api/v1/data/slice` cannot serve. Materialised crops, which do have concrete extents,
are listed by `GET /api/v1/data/zarr/cached`.

Operator interface: `python -m src.data_layer.zarr_source {catalogue|cached|inspect|materialise}`.
`inspect` reads metadata only and is the command to run before committing to a download.
Materialisation is deliberately **not** exposed over HTTP: it is a minutes-to-hours job needing
the Phase 4A artifact store and a job record, and an endpoint that held a connection open for
two hours would be a worse answer than no endpoint.

**Joined to the Phase 4 spine (T4C.5b, closing D42).** The original Zarr implementation ended
at its dedicated API: `slice_sequence` called the adapter with only a dataset id, while a Zarr
archive necessarily requires a crop specification. Parameterised source options now pass
through the registered action and adapter; crop identity remains in the Zarr content-addressed
cache rather than the adapter's id-only process cache. WeatherBench's `latitude`/`longitude`
coordinates are normalised to the `PhysicalField` `lat`/`lon` spine, and the resulting
`FieldSequence` records the real source request and resolution provenance. A local
WeatherBench-shaped test executes the complete offline path from cached crop to artifact
handle, with `is_simulated == False`.

### 3.13a Regional ERA5 through CDS (`src/data_layer/cds_source.py`, T5.2c / D43a)

CDS is treated as a queued acquisition backend, not a normal fallback `DataSource`: resolving a
dataset must never silently submit a remote job. `CDSRegionalRequest` freezes canonical
`t/q/u/v/z` selections, inclusive dates, explicit UTC hours, pressure levels, grid spacing and
north/west/south/east area order. Dateline-crossing boxes are refused until split explicitly.
The immutable request and every monthly shard have full SHA-256 identities.

`plan_monthly_shards` follows queue-friendly calendar boundaries. Acquisition is disabled unless
`SPECTRALEARTH_ALLOW_NETWORK=1` or the caller passes explicit consent. Each response is written
to a `.part` file, opened as NetCDF, hashed and atomically renamed before completion state is
updated. Resume re-hashes completed files and refuses untracked or changed shards. Standard CDS
client configuration owns credentials; no token is inspected, passed to our functions or
written into provenance. `cdsapi>=0.7.7` is isolated in `requirements-cds.txt`, so cached
laptop/HPC work has no CDS dependency.

`materialise_cds` normalises reviewed CDS coordinate/name variants, requires the exact requested
timestamps and levels, refuses implicit ensemble/`expver` selection and non-finite values, then
writes the existing content-addressed Zarr cache with bounded time chunks. The unchanged
`prepare_cached_regional_forecast` path consumes it lazily. A cache manifest includes the full
acquisition state and says `independent_overlap_check: NOT RUN`; replay uses
`rematerialise_cds_from_provenance` rather than mistaking a CDS request for a Zarr URI.

Offline acceptance uses deterministic, real NetCDF shards and proves month splitting, network
consent, atomic resume, tamper refusal, exact timestamp enforcement, cache conversion and the
complete lazy Dataset/DataLoader interface. It does **not** prove CDS credentials, queue service,
wire transfer, current NetCDF conversion or ERA5 agreement. No live request or multi-year NZ
crop has run, so this is T5.2c partial infrastructure and D43 remains open.

### 3.13b Calendar splits and physical forecast time (`regional_forecast.py`, `evaluation.py`, T5.2d)

`RegionalForecastConfig.calendar_boundaries=(validation_start, test_start)` selects a reproducible
calendar contract; without it, the existing ratios are retained. Boundaries must occur exactly
in the source time axis—no nearest-time rounding—and the lag-sufficient embargo is excluded at
the start of validation and test. `expected_cadence_hours` is optional because historical
experiments may not have frozen it, but when declared it is checked against every timestamp
interval and a mismatch or irregular axis aborts preparation. Dataset schema v2 records the
split mode and cadence evidence and emits `lead_durations_ns` derived from each target and the
final history timestamp plus `time_axis_cadence_ns` measured over the full source axis.

Forecast evaluation schema v2 does not accept frame labels alone. It requires input/target
timestamps and durations, recomputes their equality, proves each duration is the corresponding
integer frame offset times one common cadence, and proves that mapping is constant across every
sample and batch. Results retain frame-keyed metrics for machine compatibility while adding
`lead_durations_hours` globally and per metric. The crop panel exposes only the weaker manifest
state: dates are not frozen under its default ratio contract, cadence is `NOT VERIFIED`, and
physical lead labels are unavailable. Production TypeScript/build and contract tests pass;
rendered browser inspection was **NOT RUN** because no controllable browser was attached.

### 3.14 Export (`src/data_layer/exporters.py`, T3.5.23)

Before this module the platform **could not emit a single file**. Every field, spectrum, metric,
hypothesis and benchmark result lived and died inside a browser tab. A tool whose output cannot
leave it is not a research tool, whatever the quality of its mathematics.

Four data formats, each earning its place:

| Format | Why it is here |
|---|---|
| `csv` | Opens anywhere, no dependencies. Provenance in `#`-commented header lines, which `numpy.loadtxt`, `pandas.read_csv(comment="#")` and every spreadsheet importer skip — so the record costs the reader nothing. |
| `json` | The only format that round-trips *nested* provenance without flattening it. |
| `netcdf` | **NetCDF4/HDF5**, what an atmospheric scientist actually loads: coords, units and attrs straight into `xarray`. |
| `zarr` | Chunked store, delivered as a zip, for fields too large to want as one blob. |

**NetCDF4 rather than NetCDF3, and the distinction is not pedantic.** `Dataset.to_netcdf()` with
no path only supports the `scipy` engine, which writes NetCDF3 classic — no groups, no
compression, 32-bit offsets. The exporter therefore writes with `h5netcdf` to a temporary file
and returns the bytes; a test asserts the HDF5 magic number, because "it produced a file" is not
the same claim as "it produced the file the researcher meant".

**Provenance travels *inside* the file, never beside it.** A CSV in a downloads folder six months
later, with no record of the seed, the units, the grid or whether the data was simulated, is
indistinguishable from any other CSV — and that is exactly when it matters. NetCDF attributes
cannot hold a nested dict, so nesting is *flattened with dotted keys* rather than dropped;
dropping it would silently lose the crop spec, the correction and the seed.

**The two warnings are derived, not supplied**, so no export path can omit them by forgetting:
`is_simulated` produces `THIS DATA IS SIMULATED … is NOT an observation`, and
`reproducible: false` produces `THIS DATA IS NOT REPRODUCIBLE`. Both are written in words, in
the file, because a flag a reader has to know to look for is not a warning.

**PNG and SVG are deliberately client-side** (`frontend/src/components/FigureExport.tsx`), not
server endpoints. A server-side re-render would be a *different* picture from the one on
screen — different colour limits, aspect ratio, tick choices — and a figure that does not match
what the researcher saw is worse than no figure.

**An empty table still exports.** "Nothing survived correction" is a real result — it is the
answer the null benchmarks exist to produce — and refusing to save it would make the honest
outcome the one you cannot record.

### 3.15 UI scientific integrity (T3.5.23)

The frontend used to **fabricate results** whenever the backend was unreachable: fields,
perturbations, transforms ("approximate reconstruction with tiny errors"), diagnostics, boundary
analyses and experiment IDs, across roughly fourteen code paths. One badge in the header read
"Offline Sandbox Mock Mode"; the individual results said nothing, so a spectral slope computed
from `Math.random()` was indistinguishable from one computed from ERA5. That is precisely the
defect class the backend's `is_simulated` provenance chain exists to prevent, committed one layer
up. **Every one of those paths has been deleted.** With no backend there is no data, the controls
say so, and a test asserts no `if (backendConnected)` branch survives.

Four further presentation defects closed in the same pass:

*   **`is_simulated` was fetched and shown nowhere it mattered.** The dataset selector rendered
    `d.name` only, so choosing "Era5 Reanalysis" and receiving fabricated data looked identical
    to receiving observations. The tab now carries a banner reading the flag the source
    *declared*, with the fallback reason and source kind.
*   **The physical-units work was invisible.** `k_units`, `power_units`, `convention`,
    `convention_note`, `warnings` and `grid` had been returned since T3.5.13 and were absent from
    the frontend's own type definitions, so the UI could not have shown them. R15 requires the
    convention wherever a slope appears: the same field has different exponents under E(k) and
    S(k).
*   **A spectral slope was shown with no uncertainty**, though `slope_standard_error` was
    computed and returned. A β that cannot be compared against −5/3 or −3 is not a measurement.
*   **Two unconditional green ticks** — "Mathematically rigorous floating point calculations",
    "Verified perfect reconstruct limits" — appeared whatever the measured error was. Replaced by
    the measured round-trip error judged against a stated 1e-9 tolerance, with amber and red
    states.

The synthetic "forecast" on the diagnostics tab is now drawn by the backend's **seeded**
perturbation engine with the seed on screen and recorded in the result (defect D34), rather than
by an unseeded `Math.random()` in the browser.

### 3.16 Import (`src/data_layer/importers.py`, T3.5.24)

The counterpart to `exporters.py`, and the half that makes the platform usable on data it did
not produce. Until this existed, real data could arrive exactly two ways: a file placed in
`data/` by hand under one of three fixed names, or a Zarr crop streamed from WeatherBench 2. A
researcher with a NetCDF from their own model, a colleague, a CDS download, or a previous export
of this platform had **no way in**.

Reads `.nc`/`.nc4`/`.netcdf` (NetCDF3 and NetCDF4/HDF5), `.zarr.zip`, `.csv` and `.json` —
including everything `export_field` writes. Every test round-trips through the exporter rather
than through a hand-written fixture, because a fixture can encode the same misunderstanding
twice.

**The hard part is the one that looks trivial: is it 2D?** An ERA5 file is
`(time, level, lat, lon)`. Taking `[0, 0]` silently would import *a* field and never say which,
and every statistic computed afterwards would describe an arbitrary timestep the researcher did
not choose — a wrong answer indistinguishable from a right one. So the flow is **two calls**:
`inspect` reports the variables and which axes still need pinning, then `read_field` refuses
until every non-spatial dimension has an explicit index, and records those indices in the
provenance.

**Axes are identified by name before position.** Falling back to "the last two dimensions" is
right for every convention met so far, but only *after* the name check: a file with dims
`(lat, lon, time)` read positionally comes back **transposed**, and a transposed field still
looks like a field — every anisotropy and orientation statistic derived from it would be wrong
in a way nothing downstream can detect.

**A round trip cannot launder simulated data.** Export a synthetic field, import it back, and it
still says `is_simulated: true` — asserted for all four formats. Without that the platform would
offer a one-step way to turn fabricated data into apparently observational data, which is worse
than never having labelled it.

**Unknown origin reports `null`, not `false`.** A file the platform did not write carries no
claim about whether it is real; recording a confident `False` would be the platform asserting
something nobody told it. The UI renders that third state as *"Origin unknown … the platform
makes no claim about whether this is real data."*

**Uploaded archives are checked for path traversal.** `extractall` follows `..` and absolute
paths, so a zipped-Zarr upload could otherwise write outside its temporary directory. This is
the one place the platform accepts arbitrary bytes from outside itself, and the check is cheap.

A single-row CSV is accepted as a 1×N field rather than refused: the size rule that protects the
science lives with the transforms (`FieldTooSmallError`, which names the minimum for the
requested number of levels), and duplicating that judgement in the reader would put two rules in
two places to diverge.

### 3.17 Evidence and capability discovery in the UI (T3.5.24)

Four endpoints were served and unreachable from the workbench. All four are now wired, and a
test asserts that **no served route is unreachable**, with any exemption having to name its
reason in the test itself.

*   **`POST /benchmarks/run`** — the gates could be *listed* but not *run*, so a researcher
    could see what the platform claims to get right and could not make it prove it. The Platform
    tab now runs the suite at a chosen root seed, shows PASS/FAIL/NOT-YET-RUNNABLE per gate, and
    raises a distinct alarm when a **null** benchmark reports a discovery — that is a false
    positive in the platform itself, not a result. The three outcomes stay separate on screen for
    the same reason they do in the runner: folding NOT_YET_RUNNABLE into PASS would let "all
    green" mean "we never looked". Results export as a table.
*   **`GET /transforms` and `GET /actions`** — the registries, rendered with their declared
    capability flags. Generated, never hand-listed: a hand-written list goes stale precisely
    when someone adds an entry.
*   **`GET /hypothesis/proposals`** — the only exemption, recorded in the test: the discovery
    call returns the same records, so a separate listing adds no capability.

## 3.12 HTTP API Surface

75 routes. Listed here because an undocumented endpoint is an untested contract.

| Method | Route | Notes |
|---|---|---|
| GET | `/api/v1/health` | DB reachability, backend scheme, dataset count, execution device (T3.5.10) |
| GET | `/api/v1/evaluation/receipts` | verified, display-eligible real-source reports; empty means no evidence (T5.6g) |
| GET | `/api/v1/evaluation/receipts/{report_id}` | one content-addressed report, reverified on read; no machine paths |
| POST | `/api/v1/evaluation/receipts/import` | verify and admit JSON; fixtures, tampering and non-catalogue truth are refused |
| POST | `/api/v1/transforms/apply` | fft, dct, dwt, hybrid, **swt**, **dtcwt** (real Kingsbury q-shift, T3.5.6; `level1`/`qshift` sweepable) |
| POST | `/api/v1/synthetic/generate` | vortex, front, turbulence, wave |
| POST | `/api/v1/synthetic/perturb` | rotate, translate, noise (now seedable, T3.5.12) |
| POST | `/api/v1/boundary/analyze` | declares its pixel frame explicitly (D13) |
| GET | `/api/v1/actions` | every registered pipeline action, generated from the registry (T3.5.15) |
| GET | `/api/v1/transforms` | every registered transform with its params and capabilities (T3.5.15) |
| GET | `/api/v1/training/representations` | batched/autograd readiness, verification limits, SWT redundancy, DTCWT atlas geometry and selected R13 interiors (T5.1e) |
| GET | `/api/v1/data/sources` | the data-source fallback chain in priority order (E2) |
| GET | `/api/v1/acquisitions` | domain-first catalogue generated from domain/store/table contracts; each entry carries limits and the attribution caveat (TG10.2) |
| GET | `/api/v1/analysis` | the three reachable engine operations and the read-only claim boundary (TG11.1) |
| POST | `/api/v1/analysis/run` | one `domain_analysis` operation over the re-uploaded **full** record; stores nothing, moves no rung |
| GET | `/api/v1/preregistration` | what a seal binds, what "once" is measured over, and the claim boundary (TG11.2) |
| POST | `/api/v1/preregistration/partition` | the train and held-out identities a seal would bind, from geometry and lineage only |
| POST | `/api/v1/preregistration/seal` | freeze a confirmatory family against the held-out partition; the sealing time is the server's |
| GET | `/api/v1/preregistration/seals` | every stored seal, and whether its partition has been spent |
| GET | `/api/v1/preregistration/seals/{seal_sha256}` | one seal with both digest layers recomputed, optionally against a published digest |
| POST | `/api/v1/preregistration/seals/{seal_sha256}/confirm` | open the held-out partition **once**; every setting comes from the seal |
| GET | `/api/v1/evidence` | the ten evidence categories, what is computed rather than accepted, and the claim boundary (TG11.3) |
| POST | `/api/v1/evidence/studies` | register a hypothesis at revision zero, before any evidence exists to favour it |
| GET | `/api/v1/evidence/studies/{study_id}/head` | the chain, the digest an append must name, and the recomputed ladder verdict |
| POST | `/api/v1/evidence/studies/{study_id}/evidence` | append one entry, compare-and-swap on the head; the rung comes back computed |
| POST | `/api/v1/evidence/studies/{study_id}/evidence/precedence` | run the precedence sweep here and record its verdict, `false` included |
| GET | `/api/v1/mining` | matchers with their declared invariance, relations, extractors, minable domains, and what none of it is evidence of (TG11.4) |
| POST | `/api/v1/mining/records` | admit one stack of frames and extract its features here; the record is addressed by its bytes **and** its declaration |
| GET | `/api/v1/mining/records` | every admitted field by digest; reads no array and extracts nothing |
| POST | `/api/v1/mining/price` | what a mining pass over a shape would cost, computed before anything is mined (R18) |
| POST | `/api/v1/mining/tolerance` | calibrate the match tolerance from declared replicate frames; `/generate` takes its digest, never a number |
| POST | `/api/v1/mining/generate` | mine the training frames; candidates, and explicitly no claims |
| POST | `/api/v1/mining/freeze` | freeze the confirmatory family against held-out frames before they are opened; the seal lands in TG11.2's store |
| POST | `/api/v1/mining/confirm` | open the held-out frames once and correct at the frozen size; takes a seal digest and no setting |
| POST | `/api/v1/mining/motifs` | publish one frozen motif as a durable definition carrying its origin domain's licence |
| POST | `/api/v1/mining/transfer` | bind a published motif, spend the target through the ledger, then open and search it once (R20) |
| POST | `/api/v1/mining/invariance` | measure every registered matcher against its own declared invariance |
| GET | `/api/v1/cross-domain` | what is aligned, what is refused, and that nothing is interpolated (TG11.4b) |
| POST | `/api/v1/cross-domain/align` | intersect two native clocks exactly; reports what each side retained and discarded |
| POST | `/api/v1/cross-domain/lags` | convert a family declared in seconds onto the common cadence and price it before measuring any of it |
| POST | `/api/v1/cross-domain/partition` | split the aligned record with an embargo and show both partition identities |
| POST | `/api/v1/cross-domain/generate` | sweep every crossing direction at every declared duration on the training partition |
| POST | `/api/v1/cross-domain/seal` | freeze the selected members against the held-out partition; the seal lands in TG11.2's store |
| POST | `/api/v1/cross-domain/seals/{seal_sha256}/confirm` | open the held-out partition once on the frozen family; takes the two records and no setting |
| POST | `/api/v1/import/inspect` | describe an uploaded file without committing to a 2D slice of it (T3.5.24) |
| POST | `/api/v1/import/field` | read one pinned 2D field out of an upload, with reconstructed provenance |
| POST | `/api/v1/export/field` | a 2D field as CSV, JSON, NetCDF4 or a zipped Zarr store, provenance embedded (T3.5.23) |
| POST | `/api/v1/export/table` | hypotheses, benchmarks or metrics as CSV or JSON, provenance embedded |
| GET | `/api/v1/data/zarr/catalogue` | known cloud ERA5 stores, the network gate, and the R13 crop floor (T3.5.18) |
| GET | `/api/v1/data/zarr/cached` | crops already materialised locally; works with no network |
| POST | `/api/v1/data/zarr/inspect` | chunk structure and chunk-hostility for a proposed crop - **metadata only** |
| GET | `/api/v1/data/zarr/probes` | the probe ledger, with the transcription debt published (TG10.3) |
| POST | `/api/v1/data/zarr/probe` | probe one store and record the result, refusals included - **metadata only** |
| GET | `/api/v1/benchmarks` | the suite and its declared known answers (T3.5.17) |
| POST | `/api/v1/benchmarks/run` | three outcomes reported separately; 404 on an unknown name (D31) |
| GET | `/api/v1/data/datasets` | carries `source_kind` / `is_simulated` / `fallback_reason` (E2) |
| POST | `/api/v1/data/slice` | region + variable crop |
| POST | `/api/v1/analysis/diagnostics` | area-weighted metrics, physical wavenumbers, units and grid provenance |
| POST | `/api/v1/analysis/error-decomposition` | by scale, by boundary distance, by lead time |
| POST | `/api/v1/experiments` | creates and launches a sweep |
| GET | `/api/v1/experiments` | paginated listing with `limit` / `offset` / `status` (T3.5.10) |
| GET | `/api/v1/experiments/{id}` | one experiment with its runs |
| GET | `/api/v1/experiments/{id}/lineage` | lineage nodes and edges |
| POST | `/api/v1/hypothesis/discover` | correlation + categorical scan (no FDR yet - D8) |
| GET | `/api/v1/hypothesis/proposals` | generated follow-up configurations |
| GET | `/api/v1/findings/domains` | every registered domain — the union of the wording and declaration registries — with what it refuses and whether its declaration is complete (TG9.1/TG9.3/TG8.1) |
| GET | `/api/v1/findings/onboarding` | the adapter recipe itself, generated from `REQUIRED_DECLARATIONS`, plus each domain's audit (TG8.1) |
| POST | `/api/v1/channels/inspect` | what an uploaded channel record is, and the violations a domain must already declare to read it; chooses no clock column and no domain (TG8.4) |
| POST | `/api/v1/channels/read` | one channel record read against a declared domain, with that domain's refusals and the attribution caveat; refused by name where the declaration does not admit it (TG8.4) |
| GET | `/api/v1/findings/glossaries/{name}` | one domain's wording for all 38 structural terms, published so it can be audited |
| GET | `/api/v1/findings/studies` | published studies with the rung each stands on; unreadable files reported, not skipped |
| GET | `/api/v1/findings/studies/{study_id}` | one evidence bundle whole, with its digests |
| GET | `/api/v1/findings/studies/{study_id}/outputs` | the five outputs untranslated, for checking the wording changed no fact |
| GET | `/api/v1/findings/studies/{study_id}/translation` | the finding rendered in one domain's words; R9's six figures whole or absent |
| GET | `/api/v1/reviews/studies/{study_id}` | verified recorded calls, round-robin outcomes and cost receipts bound to the latest exact bundle revision; read-only and never claim permission (TG11.5) |

## 3A. Phase 4A - The Time Axis and the Artifact Store

### 3A.1 `FieldSequence` (`src/physical_core/sequence.py`, T4A.1/T4A.2)

**What was missing, and why nothing above it could exist.** `PhysicalField` is strictly 2D and
raises on any other rank. Every transform, statistic and benchmark in the platform operated on
one snapshot. The research question the project exists to answer — *which small configurations
at t precede which large structures at t+Δ* — could not be **stated**, because nothing
represented "t". `decompose_by_lead_time` only appeared to work because the caller assembled the
list itself, so the time axis lived in a local variable and vanished on return.

A `FieldSequence` is ordered `PhysicalField`s plus a real time coordinate, with the consistency
everything above it will assume checked **once, here**:

*   **Same shape**, or refused — a ragged sequence cannot be stacked or transformed as a block.
*   **Same grid**, or refused — and this is the one that matters. Frames on different grids are
    not observations of one region; a mean or a spectrum across them averages *different places
    together*, which is arithmetic that succeeds and means nothing. Equality is on the **metric,
    not object identity**, because two frames cropped from the same archive are separate
    `GridSpec` instances describing the same region.
*   **Strictly increasing times**, or refused. Unsorted input is *not* sorted silently: a
    sequence whose order was corrected without anyone noticing is worse than one that errors.
    Duplicate timestamps make "the next frame" ambiguous and would make every lag wrong by an
    unknown amount.

**Irregular sampling is reported, never silently accepted.** `cadence_seconds` gives the modal
spacing and `is_regular` says whether every step matches. `lag_to_seconds` **refuses** on an
irregular record, because "3 frames" is not a fixed duration when the record has gaps.

`at_time` refuses an inexact match by default rather than returning the nearest frame: a caller
who asked for 06:00 and silently received 12:00 has a six-hour error in a lag calculation with
nothing to indicate it. Slicing returns a `FieldSequence`, not a list — a list would drop the
time axis, which is the exact failure this class exists to prevent.

### 3A.2 The temporal split (rule R6)

`split_temporal(train, val, embargo_frames)` mirrors the existing *spatial* guardrail
(`PhysicalField.split_field` / `validate_split_guardrails`) deliberately, so the discipline reads
the same on both axes.

**Why an embargo and not just a boundary.** Atmospheric fields are strongly autocorrelated: the
frame immediately after a train/test boundary is nearly a copy of the one before it. Testing on
it measures persistence, not prediction. The embargo is the number of frames *discarded* at each
boundary and must be at least as long as the longest lag under test — otherwise the target of a
training example lies inside the test window.

The discarded frames are **returned** under `embargo_train_val` / `embargo_val_test` rather than
dropped, so a lineage record shows what was excluded and a reader can check the gap was real.

`validate_temporal_guardrails(a, b, max_lag_frames)` raises on two distinct failures, because
they have different causes and different fixes:

| Failure | Why it needs its own check |
|---|---|
| **Overlap** | The windows share timestamps. An ordinary bug. |
| **Insufficient embargo** | The windows are disjoint — *an overlap check passes* — but a training example near the boundary has its target inside the test window. **The split looks clean and leaks.** |

The second is T4A.2's acceptance criterion and the reason the function takes a lag argument at
all. A gap of *exactly* one lag still raises: it puts the last training target on the first test
frame.

### 3A.3 `ArtifactStore` (`src/artifact_store/store.py`, T4A.3)

Lineage `value` columns and inter-step `step_outputs` carried full payloads as nested Python
lists — `architecture.md` has listed that as a hard scaling limit since the first audit, and
Phase 4B is where it stops being theoretical: a `CoefficientField` over 10 frames × 64×64 ×
4 scales × 6 orientations is ~10 million floats, roughly **200 MB of JSON in a database column,
per step, per run**.

Steps now exchange an `ArtifactHandle` — reference, shape, dtype, SHA-256, and a small
*statistical* summary — while the bytes live under `artifacts/` keyed by their own content hash.

Measured against T4A.3's stated acceptance criteria:

| Criterion | Result |
|---|---|
| 100-frame 64×64 sequence round-trips | **exact**, 3.15 MB on disk |
| Every lineage row under 4 KB | **1,338 bytes** for the `slice_sequence` node |
| `analyze_boundary` no longer embeds a padded field | **already true** via the summariser layer — now asserted so it cannot regress |

**Content addressing is the deduplication and the integrity check at once.** Two steps producing
identical output write one file (verified). A handle names the exact bytes it was made from, so a
corrupted artifact is an **error** rather than a subtly wrong array — the failure it catches is
silent, and every number derived from a tampered artifact would be wrong with nothing to
indicate it.

**NaN and infinity counts are first-class in the summary**, not diagnostics: an array that is 3%
NaN poisons every downstream mean, and one that is *entirely* NaN is indistinguishable from a
good one by shape and dtype alone.

**Arbitrary objects are refused rather than pickled.** An artifact only the exact code that wrote
it can read is not reproducible data, whatever the lineage row says. Writes are atomic
(temp-file-then-rename), because a store that can hold a half-written artifact is one whose
checksums start failing for reasons nobody can reconstruct.

`resolve_value` dereferences handles, so an action written before the store existed still
receives an array and no action needs to know the store exists.

### 3A.4 Sequence slicing (`slice_sequence`, T4A.4)

`MeteorologicalDataAdapter.slice_sequence(...)` and a matching pipeline action are the entry
point through which every Phase 4 stage gets its data. The action stores the frames and returns
a **reference**, not a payload.

Two guards worth naming: a `max_frames` limit (a whole ERA5 record at 6-hourly cadence is
~93,000 frames, and the failure mode without it is a killed process rather than an error), and a
**count of non-finite values replaced**. A zero is a value, not an absence — a spectrum of a
field with zeroed gaps has structure the atmosphere does not — so the substitution is recorded
in the provenance rather than being invisible.

---

## 3B. Phase 4B - `CoefficientField` and the Wavelet Bank

### 3B.1 The five-dimensional array (`src/transform_engine/coefficient_field.py`, T4B.1)

Phase 4A gave the platform a time axis. Everything above 4B needs two more: **scale** and
**orientation**. The research question - *which small configurations at t precede which large
structures at t+delta* - is a statement about a five-dimensional array, and until this slice
that array had no representation. `apply_transform` returned a per-frame dict keyed
`"level_1"`, `"level_2"`, ..., so "the energy at scale 3 over time" had to be reassembled from
strings by whoever wanted it - the same way the time axis went missing before `FieldSequence`
existed.

`CoefficientField` is `(time, scale, orientation, y, x)` plus the parent `GridSpec`, with the
invariants checked once: rank five, every axis labelled, and a grid whose shape matches the
spatial axes. That last check is not a formality - a grid of the wrong shape means the
coordinates attached to these coefficients describe *different pixels* than the ones they
label, and every position reported downstream would be wrong by a fixed offset nobody would
look for.

**The parent-grid claim, stated honestly.** The roadmap says "backed by SWT/DTCWT so all scales
share the parent grid". That is true of SWT and **not** true of DTCWT:

| | SWT | DTCWT |
|---|---|---|
| Native band shape at level *j* | `H x W` (undecimated) | about `H/2**j x W/2**j` |
| Aligned to the parent grid by | nothing - it already is | **nearest-neighbour upsampling** |
| `resampled_to_parent` | `False` | `True`, with every native shape recorded |

Nearest neighbour rather than interpolation is deliberate. The aligned view is a *labelling* of
parent pixels by the coefficient covering them, not a smooth field to be interpolated, and
bilinear blending of complex coefficients mixes phases belonging to different spatial
positions - producing values no filter ever computed. **Upsampling invents no information**:
the effective resolution of scale *j* is still `2**j` pixels, which the summary states in those
words, because treating an aligned DTCWT band as if it resolved parent-grid detail would be a
false claim about the data. A test asserts the replication directly - an 8x8 native band cannot
produce more than 64 distinct values once spread over 64x64.

**Reconstruction therefore never uses the aligned array.** The native per-frame coefficients are
retained and `reconstruct()` inverts *those*; both families return the source sequence to about
2e-15, which is what float64 exactness looks like. A field built with `keep_native=False`, or
one restored from an artifact, **refuses** to reconstruct rather than inverting the upsampled
view. That refusal is the point: inverting the aligned view would return a field a few percent
wrong that is indistinguishable from a real reconstruction by inspection - silent, small, and
therefore the worst available failure.

**Orientation labels say only what the transform can support.** DTCWT carries the six feature
orientations in degrees, with the convention named and a pointer to the wavevector convention
that differs from it by 90 degrees. SWT carries `LH`, `HL`, `HH` and *no angles*, because a
separable real wavelet's `HH` band responds to both diagonal signs and cannot distinguish them.
Writing "45 deg" on it would assert a directional selectivity the transform has not got - and
that claim would then propagate into every figure drawn from it. The ambiguity is the classical
motivation for the dual tree, and it is recorded as such.

**Physical scale is reported as a band, or not at all.** A dyadic level *j* covers wavelengths
of roughly `2**j` to `2**(j+1)` samples, so `scale_wavelength_bands()` returns that interval in
metres rather than a single number, which would imply a selectivity the filter does not have. On
a grid with no physical spacing it returns `None` rather than a guess: a wavelength in metres
derived from a pixel grid would be fabricated.

`summary()` is lineage-safe by construction - labels, provenance and per-`(scale, orientation)`
mean energy fraction. A ten-frame four-level DTCWT bank summarises to **2,008 bytes** while the
payload it describes is **983,040 complex coefficients**.

### 3B.2 The bank as an ordinary parameter matrix (`src/transform_engine/bank.py`, T4B.2)

The roadmap calls this "one of the genuinely free wins in this plan": a `wavelet_bank` block
needs no engine changes because its entries are ordinary parameter-matrix entries.
`WaveletBank.to_parameter_matrix()` returns a plain `Dict[str, List]`, and `combinations()`
expands it by calling the engine's **own** `expand_parameter_matrix` - so a bank cannot expand
differently from an ordinary matrix, because it *is* one. The test asserts the two produce
identical output; reimplementing the product would have tested the reimplementation.

The families come from the registry, filtered to those `coefficient_field` can arrange into
(scale, orientation) axes. `fft`, `dct` and `hybrid` are registered transforms but are **not**
banks - they have no such factorisation, and giving them one would invent axes that do not
exist. They are refused at *configuration* time, so a long sweep does not die on its last
combination.

**One asymmetry, stated rather than papered over.** The roadmap lists `families`, `scales` and
`orientations` together as though all three were sweep axes. Two of them are. `orientations` is
not: a wavelet transform computes every orientation in a single pass, so sweeping them would run
the identical decomposition once per orientation and discard five sixths of each result - the
same coefficients at six times the cost. Orientations therefore travel as a **selector** applied
within each run, and `summary()` says so under `orientation_role`.

The 1,000-combination guard is kept and duplicated in the bank so the refusal names the bank and
quantifies the cost, rather than arriving from deep inside the engine. A test asserts the two
ceilings are the same number.

### 3B.3 The two pipeline actions (T4B.3)

`decompose_bank` runs every combination and stores each `CoefficientField` as an artifact,
returning **references**. A four-combination bank over five 64x64 frames is 921,600
coefficients; its lineage row is 2,415 bytes. `extract_scale_signature` collapses a field to
per-`(time, scale)` energy and energy fraction.

**Scope, so the roadmap is not over-claimed:** `extract_scale_signature` is the *energy half* of
rule R3. Participation ratio, the Gini coefficient and threshold counts are T4C.1; the action
says so in its own result payload, at the point a reader actually looks. Energy *fractions*
rather than raw energy are what make signatures comparable at all - raw energy scales with the
amplitude of the field, so the same structure recorded in different units would produce two
different signatures.

Both actions refuse a **dereferenced payload** by name. `resolve_value` turns a literal
`artifact://...` into a bare array before an action sees it, and an array has neither a time
axis nor a grid nor scale labels; accepting one would mean inventing a cadence. The refusal
names the fix (`{step.sequence_ref}`, which passes the reference rather than the payload).

### 3B.4 Pressure level as a bank dimension (T4B.4)

ERA5 is `time x level x lat x lon x variable`, and until this slice the vertical axis was only a
*selector*: analyse 500 hPa. That made the most famous precursor relationship in synoptic
meteorology - an upper-level trough preceding surface cyclogenesis - impossible to express,
because two levels were two unrelated runs with nothing tying their time axes together.

`MeteorologicalDataAdapter.slice_level_sequences` returns one `FieldSequence` per level in a
single pass, so the shared time axis is a property of the construction rather than something a
caller must remember to arrange. `LevelBank` validates that the levels agree on shape, family
and **timestamps** - a vertical lead-lag measured across levels sampled at different times
measures the sampling, not the atmosphere. `vertical_offsets()` produces the 4E edge attribute:
signed, in hPa, with its direction named, because pressure decreases upward and a reader who got
that backwards would invert every precursor relationship the bank exists to find.

**Two silent failures closed here, both found by building the thing rather than by review:**

*   **A vertical bank with one sequence.** `decompose_bank` originally ignored `levels_hpa`. It
    would have decomposed the same frames once per level and labelled the results 850 hPa and
    500 hPa - **identical coefficient arrays under different labels**, and every cross-level
    statistic downstream would then have measured that fiction. A bank declaring `levels_hpa`
    now requires one sequence per level and refuses to guess.
*   **Levels that collapse onto one stored level.** `sel(method="nearest")` means a requested
    level a dataset does not carry silently becomes its neighbour, so two requests can land on
    the same data. `slice_level_sequences` compares the returned arrays and refuses when they
    are identical: a cross-level correlation computed from them is a field correlated with
    itself - a coefficient of 1.0 that means nothing and looks like a discovery. The guard
    fired immediately on `t2m`, which has no vertical axis at all.

**Scope discipline, as the roadmap requires: this is 2D-per-level, not 3D wavelets.** Nothing
here resolves vertical structure *within* a decomposition. Each level is decomposed
independently and the vertical relationship is carried as a cross-level edge attribute by 4E.
`LevelBank.summary()` states that in those words, so a later reader cannot mistake what was
computed.

### 3B.5 Two defects closed in the store on the way (D38, and the missing time axis)

Neither was in the plan; both were found by Phase 4B walking into them.

**D38 - a complex summary that silently described only the real part.** `summarise` called
`float(values.min())` unconditionally. For a complex array that does *not* raise: numpy casts to
real, discards the imaginary part, and emits a warning nobody reads. `[1+2j, 3-1j]` reported
`min = 1.0`. Every DTCWT coefficient field is complex, so the lineage rows of an entire phase
would have carried statistics of the real part alone, labelled as statistics of the array.
Complex arrays are now summarised on their **magnitude** and say so under `statistic_of`.

**A store that dropped the time axis.** `put(sequence)` stored the `(T, H, W)` tensor and the
grid, but not the timestamps - so `load` returned an array that was not a sequence, and any
caller rebuilding one would have had to assume a regular cadence, silently mis-dating every
frame of an irregular record. Artifacts now **carry their own axes**: the time coordinate as an
array member and the labels, grid and metadata as one JSON member inside the same `.npz`, with
`load_sequence` and `load_coefficient_field` rebuilding the real object. `allow_pickle=False`
throughout - the axes are data, not a pickle. The handle is unaffected: a 200-frame sequence
still fits the 4 KB budget, because the axes went into the archive rather than the database row.

The module docstring also claimed a `.pt` format that was never implemented. It now states the
single format and why: a torch checkpoint is a pickle readable only by the version that wrote
it, which is not a property reproducible data should have.

## 3C. Phase 4C - `ScaleSignature`, `SurrogateNull`, Cross-Scale Dependency (THE GATE)

Phase 4C is where the platform stops building instruments and asks its question. Everything
below exists to make one sentence answerable honestly: *does fine-scale activity at `t`
precede coarse-scale activity at `t + dt`, beyond what the power spectrum and the
autocorrelation already explain?*

Three of the four modules are small. What took the work was the calibration - finding the
configurations in which the machinery returns a confident answer that is entirely an artefact,
and closing each one with a measurement rather than a caveat.

### 3C.1 The signature (`src/analysis_engine/scale_signature.py`, T4C.1)

Four numbers per `(time, scale)`, reducing a five-dimensional coefficient array to a matrix
small enough to store, to test against two hundred surrogates, and to reason about: energy
density, energy fraction, participation ratio, Gini coefficient - plus a threshold count that
is reported and never primary.

**Rule R3 is why three of them exist.** A raw count of "significant coefficients" at a scale
has no answer independent of the threshold, because each scale has its own coefficient
variance; and in a decimated pyramid the number of *available* coefficients already falls as
`s**-2`, so an unnormalised per-scale total is reporting the pyramid's geometry. Every measure
here is a density, a fraction, or a population-normalised concentration.

The three threshold-free measures are kept separate because they disagree informatively.
Energy fraction says how the energy divides between scales and nothing about whether it sits
in one structure. The participation ratio, `(sum w)**2 / (n sum w**2)` on coefficient energies,
is `1/n` for a single dominant coefficient and `1` for a perfectly even scale. The Gini
coefficient responds to the whole distribution's shape rather than to its second moment, so a
scale with a long tail of medium coefficients separates from one with a single spike.

**The measures have analytic values on white noise, and the tests assert those rather than
"roughly flat".** For a real transform the coefficient energy is chi-squared with one degree of
freedom, giving a participation ratio of exactly `1/3` and a Gini of exactly `2/pi`; for a
circular complex band it is exponential, giving `1/2` and `1/2`. Measured: `0.334` and `0.636`
for SWT, `0.499` and `0.500` for DTCWT at level 2. That is a test of the arithmetic, not a
snapshot of it.

**A measured property of the dual tree, recorded because it changes what level 1 means.** The
q-shift filters that make the two trees a Hilbert pair only start at level 2. At level 1 the
real and imaginary variances of a subband differ by a factor of **2.19** on white noise, while
at levels 2 to 4 they agree to within 6 percent, and the level-1 participation ratio
correspondingly sits at `0.467` - between the real value `1/3` and the circular-complex value
`1/2`, and predictable from the two variances alone to within 0.02. Level 1 of a DTCWT is not
an analytic signal, and a concentration measure taken there does not mean quite what it means
above.

**Everything is computed on native coefficients, inside the valid interior.** Two corrections
that are silent when wrong:

*   A resampled DTCWT band repeats every native coefficient `4**j` times. Energy *fractions*
    survive that untouched - the parent grid has the same cell count at every scale, so the
    factor cancels in the ratio, and a test asserts the aligned and native fractions agree to
    float64. The participation ratio does **not** survive it: replicating every coefficient
    `r` times multiplies it by exactly `r`, so an aligned-view signature would report the
    coarse scales as far more evenly spread than they are. `CoefficientField.native_band`
    recovers the true coefficients either from the retained native arrays or by subsampling
    the aligned view on the replication stride, which is exact rather than approximate.
*   Coefficients within one filter support of the edge are contaminated by the padding
    (rule R13) and look exactly like strong, localised, oriented features - which is what a
    concentration measure is built to notice. The mask is per-scale and computed from the
    transform's own filter lengths.

**Rule R13's crop-size table understates the dual tree, and now says so.** The table is derived
for a single 14-tap filter repeated at every level. DTCWT is not that: level 1 uses the 19-tap
near-symmetric highpass and levels above use the q-shift pair, so `dtcwt.filter_support`
accumulates the actual cascade and returns a level-4 margin of **97 parent pixels against the
table's 52**. The consequence is concrete: a 256x256 crop - the roadmap's stated practical
minimum for four dyadic levels - leaves DTCWT level 4 a **2x2** valid interior, four
coefficients per orientation. The signature reports that scale as *thin* by name rather than
averaging over it, and 512x512 restores it to 18x18. Logged as D40.

### 3C.2 Surrogate nulls for a record (`src/analysis_engine/surrogate_null.py`, T4C.2)

`src/statistics/surrogates.py` already generates surrogates of an array and is not
reimplemented here; this module is the sequence-level layer above it. It is deliberately not
named `analysis_engine/surrogates.py`, as the roadmap suggested: two modules called
`surrogates` in one codebase resolve differently depending on which package the reader is in,
and it invites the phase-randomisation core to be forked and drift.

`phase_randomise` was restricted to 1D and 2D input. The arithmetic never was - the phases come
from `fftn` of a real field of the same shape and the self-conjugate bins are found by an axis
loop - and the restriction was a statement about what had been tested. It now works in any
number of dimensions, which is what makes the default null possible.

**The choice of null is a choice about time, and getting it wrong is not subtle.**

| null | preserves | measured behaviour on an AR(1) record with no organisation |
|---|---|---|
| `spatiotemporal_phase` (default) | the full 3D spectrum, hence the spatial spectrum **and** the temporal autocorrelation | null lag-one autocorrelation **0.845** against the record's 0.900 |
| `per_frame_phase` | each frame's own spectrum | null lag-one autocorrelation **0.013** - the autocorrelation itself beats the null |
| `circular_time_shift` | every frame exactly | the right null for a lagged claim; destroys only alignment |

`per_frame_phase` is what "preserving the per-frame PSD" literally asks for and it is the wrong
default: it destroys the record's temporal structure, so a statistic that depends on time is
compared against a null that is easier to beat than reality (rule R12). It is kept, with a
warning attached to every result that uses it, because the failure is worth being able to
demonstrate.

T4C.2's acceptance criteria hold: the spatiotemporal surrogate preserves the 3D power spectrum
to **3.4e-16** while the frame-by-frame correlation with the source falls below 0.1; a
sequence of localised blobs scores at the p-value floor with an effect size above 3; and a
fractional Brownian sequence does not score at all.

### 3C.3 Cross-scale lagged dependency (`src/analysis_engine/cross_scale.py`, T4C.3)

Lagged mutual information and transfer entropy over the `A_t(s)` matrix, for every ordered
scale pair and every admissible lag, each against its own surrogate ensemble, with the family
corrected together. Transfer entropy - `I(A_{t+lag}(s') ; A_t(s) | A_t(s'))` - is the one that
speaks to precedence, because conditioning on the target's own present is what stops a series
with a long autocorrelation showing dependence on anything that shares its timescale.

Both estimators use equiprobable bins with the Miller-Madow correction, and **neither is
reported as a raw value**: every number is an excess over an ensemble with the same length,
the same bins and the same marginals, so the plug-in bias is present on both sides.

**Three ways this produces a confident wrong answer. Each was found by building it, and each
is now a measurement.**

**1. A linear lag against a circular null rejects every time.** Fourier-transform surrogates
are circularly stationary; a record is not. A lagged statistic computed on the record as a
line is systematically larger than the same statistic on the surrogates. Measured on twenty
independent AR(1) records, where the null is true by construction:

| lag statistic | false rejections at alpha = 0.05 | median p |
|---|---|---|
| linear (`x[:-k]` against `x[k:]`) | **20 of 20** | 0.010 |
| circular (wrapped) | **0 of 20** | 0.485 |

Lags therefore wrap by default, and the fraction of pairs coming from the wrap is reported so
the dilution stays visible. This is the single most dangerous configuration in Phase 4C: it
would have produced a gate that passed on pure red noise.

**2. The shift null must exclude the simultaneous alignment, not only the tested one.**
Rolling the source by `s` measures the pair at an effective lag of `lag + s`, so the ensemble
has to exclude shifts near `0` - the alternative hypothesis - **and** shifts near `-lag`, which
put the two series at effective lag zero. That second window is easy to forget and it is not
hypothetical: anything varying frame by frame and touching every scale at once couples the
scales instantaneously. On the synthetic cascade the single shift at `s = -lag` produced a
transfer entropy of **0.412 nats against an observed 0.211** - the largest value in the entire
"null" ensemble came from a real relationship in the data - and capped the achievable p-value
at about 0.005 however many surrogates were drawn. Significance limited by a null that was
wrong rather than by evidence that was weak. With both windows excluded the same test reaches
the p-value floor.

**3. Rule R4's support floor, stated honestly.** A coarse coefficient and the fine ones beneath
it are computed from the same pixels. But the transform here is **spatial and applied frame by
frame**, so its temporal support is exactly zero, and quoting one would be an invention. What
is real is the time a structure needs to advect across the filter's spatial support:
`t_cross(s) = support(s) * dx / U`. `support_floor` computes it and **refuses to default the
advection speed** - a plausible-looking 10 m/s would silently set every floor in every result
from a number the reader never chose. Without it, the only floor applied is one frame, and the
result says so in as many words.

**The acceptance criterion, measured end to end.** `src/synthetic_generator/cascade.py` builds
a record in which a fine band's amplitude at `t` sets a coarse band's amplitude at `t + 3`,
driven by a *red* modulation - a white one would have made the test far too easy. Run through
the whole path (decompose, signature, sweep, 1,999 circular-shift surrogates, Benjamini-
Yekutieli):

| record | significant tests | which |
|---|---|---|
| cascade | **2** | `1 -> 3 @ lag 3` and `2 -> 3 @ lag 3`, `q = 0.0157`, excess 0.17 and 0.16 nats |
| the same record, phase-randomised | **0** | - |

Correct lag, correct direction, and nothing in the reverse direction. The phase-randomised
version preserves every spectrum and every autocorrelation and destroys only the alignment, so
a dependency that survived it was never about the alignment; none does.

**A sweep that cannot reject anything says so.** 99 surrogates floor the p-value at 0.01, and
18 tests under Benjamini-Yekutieli need a raw p below about 4e-4; the full 120-test sweep needs
**12,885 surrogates**. `check_power`, built for T4C.5, is called before the result is read and
its warning is attached, because a study that was arithmetically incapable of rejecting
anything is otherwise indistinguishable from a clean negative.

### 3C.4 The power-law core (`src/analysis_engine/power_law.py`, T4C.4)

`spectra.fit_power_law` grew up inside the radial-spectrum code and speaks its language:
annuli, isotropy, `beta_energy_1d`, Charney and Kolmogorov. All correct for a power spectrum
and meaningless for the two other power laws 4C needs - energy against scale, and feature
population against scale. The least-squares core is therefore factored into `loglog_fit`, which
knows nothing about what `x` and `y` are; `spectra.fit_power_law` now calls it and adds the
turbulence interpretation on top, with its behaviour unchanged - the 118 tests in the three files that
exercise the fit pass untouched - and a test pinning the generic core and the spectral wrapper
together.

**Rule R2 is enforced in the return value, not in prose.** A fractional Brownian field gives a
clean, high-`r_squared` power law and contains no organisation whatsoever, so
`compare_exponent_to_null` returns `reportable: False` until an exponent has been placed beside
a surrogate ensemble. `reportable` is deliberately not conditioned on significance: a null
result is reportable, and is often the point.

`scale_energy_exponent` fits `A(s) ~ s**-alpha` against the per-available-coefficient energy
density - rule R3's normalisation, since an unnormalised population count in a decimated
pyramid already falls as `s**-2` and an unnormalised fit would recover `2 + physics` and
attribute both to the same cause.

**A gate is now a frozen object, not prose (T4C.5c).** `cross_scale.GateProtocol` records the
scale/lag family, estimator, measure, bins, surrogate count, alpha, correction, expected frame
count, temporal split, embargo and seed, and hashes that exact design to a stable SHA-256
fingerprint. Validation refuses a surrogate ensemble whose empirical p-value floor cannot
survive the declared correction, a train/test partition below five samples per joint-estimator
cell, or an embargo shorter than the longest tested lag. `evaluate_replication_gate` requires
the same positive corrected relationship in both independent partitions. It returns three
states: PASS, an adequately powered FAIL, or INVALID for configuration drift, missing
advection support or inadequate power. INVALID is not a negative scientific result.

### 3C.5 What 4C does not yet answer

T4C.6, the gate review itself, asks the question **on real ERA5 data**. The analysis and
decision instruments exist and are calibrated; the present WeatherBench layouts do not supply
the required multi-year regional record within the laptop tier (D43). The verdict is not
written here because it has not been run on the atmosphere. Writing it from synthetic evidence
or an estimator-starved short record would be exactly the kind of claim this phase was built
to prevent.

The execution boundary is now complete as T4C.5d in `analysis_engine/gate_run.py`. A frozen
`GateStudyPlan` binds the exact `CropSpec`, variable, pressure level, SWT/DTCWT filters and
boundary convention, train-only harmonic climatology, threshold, declared advection speed and
the complete `GateProtocol` under one SHA-256. `preflight_cached_gate` opens only an existing
local cache and checks source identity, exact frame count/cadence, actual transform interiors
and the filter-support lag floor before expensive work. For a frozen transform it derives the
valid parent-grid shape from that transform's exact accumulated support and enforces R13's
128-pixel minimum directly; the conservative generic 14-tap crop table remains a planning
bound, not a substitute for the selected filter. A real-evidence role additionally
requires direct CDS provenance and a recorded PASS from the independent WeatherBench overlap
check; a local fixture cannot be relabelled as ERA5.

The data and analysis paths are bounded rather than merely lazy. CDS monthly shards are
validated and appended to a temporary Zarr store in declared time blocks; the complete logical
content hash is streamed independently of chunk layout before atomic publication. A cached
`CachedFieldReader` exposes one exact physical frame at a time with a decompressed-chunk byte
ceiling. `fit_harmonic_climatology_stream` retains only one source frame plus the small
`(parameters,H,W)` fit and estimates it on training indices only. `stream_scale_signature`
makes two deterministic passes, retaining one source/coefficient frame, matches every eager
measure, fits thresholds on train and reuses them unchanged on test, and refuses if the input
bytes differ between passes. The final receipt authenticates the plan, climatology, source
streams, train/test results and verdict and is published atomically without overwrite.

This work found five defects rather than hiding them behind the external-data blocker: D46
(whole-record CDS materialisation), D47 (whole-record coefficient/climatology residency), D48
(an advection floor documented as filter support but implemented as `2**level`) and D49 (a gate
verdict not bound to crop/transform/source/result identity), plus D50 (no enforced free-space
budget before a resumable acquisition). All five are fixed with synthetic
end-to-end evidence. **No atmospheric verdict follows:** the accepted fixture is explicitly
`scientific_verdict: NOT_ESTABLISHED`; D43 and T4C.6 remain open until the live acquisition and
independent overlap check exist.

T4C.5e closes the last evidence-binding seam before acquisition. `data_layer/era5_overlap.py`
opens only two existing local caches, selects WeatherBench timestamps and grid coordinates
exactly (never interpolating), and compares each canonical variable in bounded frame blocks
under recorded unit-aware tolerances. It publishes a no-overwrite receipt binding both content
hashes, the CDS request, level, coordinates, variables, tolerances and boundedness evidence,
then atomically attaches that receipt to the CDS manifest. `gate_run` validates the embedded
receipt and its SHA rather than trusting the old mutable `independent_overlap_check: PASS`
string. A FAIL remains durable evidence but cannot authorise the atmospheric gate.

`analysis_engine/gate_campaign.py` adds the T4C.5f acquisition boundary. A portable
`GateCampaign` binds the full CDS request, a same-route/same-grid canary, the exact catalogued
WeatherBench overlap and `GateStudyPlan`; strict reconstruction rejects unknown fields and
recomputes all nested identities. Its zero-network preflight aggregates worst-case storage for
all NetCDF/Zarr artifacts on their physical volumes and reports dependency, standard credential
configuration and explicit-consent presence without reading secrets or constructing a CDS
client. The canary must pass before the multi-year transfer; after that transfer, overlap is run
again so the evidence admitted by `gate_run` belongs to the full cache rather than only the
canary. Machine paths remain outside the campaign fingerprint.

T4C.5g makes that preflight physically meaningful. `cross_scale.support_floor` formerly took
the first truthy `grid.dx`; in a lat/lon `GridSpec` this is degrees, not metres. ERA5's 0.25°
spacing was therefore treated as 0.25 m and the advection floor collapsed toward one frame.
The function now reconstructs physical grids, converts angular axes, and conservatively uses
the largest cell-axis spacing across the crop. `GateCampaign` applies the same calculation
before acquisition, derives the selected SWT/DTCWT's exact supports and valid parent interiors,
and refuses R13-deficient geometry or a lag family below that floor. CDS bounds must contain an
integer number of grid intervals and returned endpoints must match exactly, preventing silent
server snapping from changing the frozen crop.

T4C.5h closes the remaining design-choice boundary without claiming a result. The authenticated
campaign at `campaigns/t4c6_nz_era5_temperature_850_v1.json` preregisters the single primary
family: 2018--2022, six-hour 0.25-degree ERA5 over 20--60 S / 140--180 E, 850-hPa temperature,
three db2 SWT scales, energy-density transfer entropy at 18--48-hour lags, six bins, 4,999
circular-shift surrogates, BY at alpha 0.05, an eight-frame embargo and a fixed seed. This gives
7,304 frames, a 4,382/2,914 train/test split, 36 hypotheses, 139x139 deepest valid interior and
a p-value floor capable of surviving BY (3,005 surrogates required). `gate_campaign review`
recomputes and exposes the exact calendar split, geometry, physical floors, multiplicity and
claim boundary locally. The pinned campaign SHA-256 is
`84f7b53fd25d555c8dcd57c6006288b95c5908f2a1d5c002d10a6572c7875975`. This is a
preregistration, not ERA5 evidence: D43 and T4C.6 remain open.

The frozen campaign's local preflight budgets 4.75 GiB of working artifacts plus a 5 GiB
reserve; D: had 1,180.06 GiB free and therefore passed capacity. Readiness remains BLOCKED on
the absent optional `cdsapi` dependency, absent standard CDS credential configuration and
disabled explicit network consent. The preflight inspected no secret, constructed no client
and used no network.

## 4. Database Schema and State Tracking (`src/database/models.py`, `session.py`, `migrate.py`)

The database layer (`src/database/`) is fully configured using SQLAlchemy and targets a persistent or in-memory SQLite database (`spectral_earth.db`). 

```
                                +-------------------+
                                |    Experiment     |
                                +-------------------+
                                | id (PK)           |
                                | name, description |
                                | status, config    |
                                +---------+---------+
                                          | 1
                                          |
                                          | 0..*
                                +---------v---------+
                                |   ExperimentRun   |
                                +-------------------+
                                | id (PK)           |
                                | experiment_id (FK)|
                                | status, results   |
                                | parameters        |
                                +---------+---------+
                                          | 1
                                          |
                                          | 0..*
+-------------------+           +---------v---------+
|    LineageEdge    |           |    LineageNode    |
+-------------------+           +-------------------+
| id (PK)           | 0..*      | id (PK)           |
| source_id (FK)----+-----------+ experiment_id (FK)|
| target_id (FK)----+           | run_id (FK)       |
| relation          |           | name, type, value |
+-------------------+           +-------------------+
```

*   **`Experiment`:** Represents a declared batch of runs with a unified pipeline config.
*   **`ExperimentRun`:** Represents an individual run with evaluated metrics and specific parameters.
*   **`LineageNode`:** Represents an execution node (dataset, code revision, field, coefficients, or metrics).
*   **`LineageEdge`:** Logs how nodes link together to form a provenance tree.
*   **`Hypothesis`:** Holds automatically discovered scientific relations.

### 4.1 Schema Migrations (`src/database/migrate.py`, `migrations/`, T3.5.8)

The schema is now managed by **Alembic**, not by `Base.metadata.create_all`. This is a
correctness fix, not housekeeping.

**The defect (D32), measured on the repository's own database.** `create_all` adds missing
*tables*. It does not alter existing ones, and it reports success either way. Slice T3.5.12
added `experiment_runs.seed` and `execution`; T4C.5 added `p_value`, `q_value`, `n_tests` and
`statistics` to `hypotheses`. The checked-in `spectral_earth.db` predates both, so the ORM
mapped five columns the file did not contain. Querying it raised
`sqlite3.OperationalError: no such column: experiment_runs.seed` — while startup logged
nothing at all. A platform whose results are meant to be defendable cannot have a storage
layer whose only upgrade path is deleting the file and losing the runs.

**Two revisions, and the baseline is deliberately *not* the current schema.**

| Revision | Contents |
|---|---|
| `0001_initial_schema` | The five tables as they existed before migration control — **without** the five later columns |
| `0002_reproducibility_and_statistics` | `experiment_runs.seed`, `execution`; `hypotheses.p_value`, `q_value`, `n_tests`, `statistics` |

Making 0001 match today's `models.py` would have been the natural shortcut and would have
broken the one job a baseline exists for. Adopting a pre-Alembic database means stamping the
revision it *is*; if 0001 already claimed the five columns, stamping it would assert columns
that are absent, 0002 would be skipped permanently, and the result is a database Alembic
believes is current and the ORM cannot query — the original defect, now with a version
number on it. `detect_legacy_revision` therefore **probes for `experiment_runs.seed`** and
stamps 0001 or 0002 by observation. Verified against the real file: detected `0001`, applied
`0002`, then zero drift against the ORM.

**`ensure_schema` — the startup path.** Four states, four different correct actions: empty
(migrate), populated but unstamped (adopt by inspection, then upgrade), stamped and behind
(upgrade), stamped and current (do nothing, and say so). A stamped revision that does not
exist in `migrations/versions/` is an **error**, not an empty upgrade list: it means the
database was migrated by a newer checkout, and running an older ORM against a newer schema
reads and writes the wrong columns silently.

**Auto-upgrade is on by default and can be switched off.** `SPECTRALEARTH_AUTO_MIGRATE=0`
makes `ensure_schema` refuse and name the outstanding revisions instead of applying them —
the right behaviour for a shared deployment, where a schema change should be a reviewed step.
Refusing to start beats serving requests against a stale schema and failing on the first
query that touches a new column.

**All five new columns are nullable, and that is a scientific statement.** A run recorded
before seed capture genuinely has no seed. `NULL` says so; a default of `0` would claim a
reproducibility that does not exist.

**How the migration history is kept honest.** `alembic upgrade head` from empty is the stated
acceptance criterion, but it passes for a history that has quietly drifted from `models.py` —
the exact failure this layer exists to prevent. So `test_migrations.py` asks Alembic's own
`compare_metadata` whether the migrated schema and the ORM disagree, with `compare_type` and
`compare_server_default` enabled, and fails on any difference in tables, columns, types,
nullability or indexes. Adding a column to a model without writing a migration now fails a
test instead of surfacing in production. Each revision is also round-tripped individually
(`upgrade` then `downgrade`), because a history is only reversible if every step is, and the
broken step would be discovered exactly when a rollback was needed. `test_downgrade_0002_*`
checks that batch-mode `DROP COLUMN` — which rebuilds the table on SQLite — keeps both the
other columns and the rows.

**PostgreSQL: rendered, not run.** `render_as_batch=True` makes one script work on both
backends, and the migrations are asserted to render valid DDL for the `postgresql` dialect in
Alembic's offline mode. That proves they *compile* there — no SQLite-only construct, no
unrenderable type. It does not prove they execute, because no PostgreSQL server was
available. The roadmap's acceptance criterion names both backends and only one has actually
been run; this is stated rather than glossed.

The stamped revision is reported by `GET /api/v1/health` as `schema_state`, recomputed per
request rather than cached from startup — a cached `"up_to_date": true` is the one kind of
health output worse than none. A stored result is only reproducible if the schema that stored
it is identifiable, so the revision belongs in provenance beside the seed and the code
revision.

`python -m src.database.migrate {status|upgrade|adopt|downgrade <rev>}` is the operator
interface; `alembic upgrade head` works unchanged, taking its URL from
`src.database.session.DATABASE_URL` so there is exactly one place the database is named.

---

## 5. Architectural Seams (Decoupling Points)

The architecture is highly modular and maintains clean boundaries at several critical integration points:

1.  **The Data Adapter Seam (`MeteorologicalDataAdapter`):**
    *   *Decoupling:* Completely isolates analytical engines from file formats (NetCDF/GRIB/Zarr) and remote servers (CDS API/AWS S3).
    *   *Seam Interface:* Any data loading method that reads from a source and outputs an `xarray.DataArray` with time, level, lat, and lon dimensions can be plugged into this adapter without modifying any downstream analysis or transform engines.
2.  **The Pipeline Solver Seam (`resolve_value`):**
    *   *Decoupling:* The pipeline is fully descriptive. The engine does not compile code; it parses JSON configurations and evaluates strings.
    *   *Seam Interface:* Step outputs are serialized to python dictionaries (`.tolist()` or float/int dictionaries). This allows running step functions on different hardware (e.g., GPU servers) while passing standard JSON strings between boundaries.
3.  **The Provenance Lineage Seam (`LineageNode`/`LineageEdge`):**
    *   *Decoupling:* Provenance tracking is decoupled from direct pipeline execution. The execution steps return outputs; the experiment engine intercepts these outputs and maps them into graph nodes and edges.
    *   *Seam Interface:* The graphing frontend can query the lineage tables independently, completely unaware of how the pipeline was executed, what hardware was used, or whether the fields were synthetic or real.
4.  **The Database Seam (SQLAlchemy Base):**
    *   *Decoupling:* SQLite is used as a default local engine. Since all queries are handled by standard SQLAlchemy sessions, changing to PostgreSQL, MySQL, or CockroachDB is a single line configuration change.

---

## 6. Front-End Technical Implementation

The React frontend is fully written and structurally complete. It was installed and built in T3.5.0/T3.5.3 (`npm run build` emits hashed JS and CSS into `dist/`) and wired to the previously unreachable endpoints in T3.5.22. Its **rendered appearance was confirmed by the user on 2026-08-20** (T3.5.25): the platform was started, both servers came up, and the then-nine tabs were reported working. T5.6g added a tenth tab, TG9.2 an eleventh and TG8.4 briefly a twelfth Domain Records tab. TG10.2 consolidated that reader into Acquire, leaving eleven destinations; TG11.0 groups those destinations by workflow rather than numbering them. TG11.1-TG11.5 add Cross-domain analysis, Preregistration, Evidence record, Structure mining, Cross-domain record and Recorded review, bringing the workflow to sixteen destinations. The post-T3.5.25 surfaces compile and build but have **not** been visually inspected in a browser. The earlier confirmation is a user report, not an artefact - **no screenshot per tab exists in this repository**, so T3.5.0's literal evidence clause remains outstanding. Contract tests prove all sixteen current destinations compile, call routes that exist and read fields that are present; they do not prove rendered appearance.

*   **Component Visualizations:** `Heatmap2D.tsx` and `LineChart.tsx` wrap `react-plotly.js`; `LineageGraph.tsx` is a hand-rolled SVG node-link renderer with a tooltip inspector and no external graph dependency. All three take reactive props and render spatial fields, PSD curves, coherence ratios, and provenance DAGs.
*   **Accessibility has a workflow-wide source contract (TG11.6).** The shell provides skip and
    route-focus behaviour, every legacy gridded control is programmatically labelled, focus is
    globally visible, reduced motion is honoured, asynchronous state is announced, figures have
    text equivalents and SVG lineage nodes have keyboard operation. Rendered assistive-technology
    inspection is still NOT RUN, so no WCAG conformance level is claimed.
*   **Main Application (`App.tsx`):** shell state and the fourteen destinations grouped under
    Acquire, Analyse, Evidence, Review, Read and Platform, with persistent selected-record and
    selected-study context, loading indicators, dynamic controls and proposal adoption.
*   **Platform & Evidence tab (T3.5.22):** the execution device, core and thread counts, executor backends with the measured rationale for the serial default, the SQLite pragmas actually in force, the stamped Alembic revision with an explicit warning when the schema is behind the code, the data-source fallback chain labelled observational/SIMULATED from each source's own declared flag, and the full Ground-Truth Benchmark Suite with its declared known answers and null benchmarks marked. All of this existed on the backend for several slices with no consumer.
*   **Acquire tab (TG10.2):** domain first, then an acquisition generated from the registries. The ERA5 crop form, R13 floor, recorded probe ledger, metadata-only chunk inspection, cached-crop readiness refusals and materialisation command remain under `grid_crop`; the existing channel reader appears under an admitted `channel_table` domain. Adding a domain changes rows, not navigation.
*   **Spectral Transforms training-readiness and DTCWT evidence panels (T5.1e):** read backend-derived contracts rather than a hand-written capability list. Every candidate shows coefficient expansion, shift behaviour, directional meaning, boundary convention, scientific role, verified and NOT RUN backends, and limitations. SWT and DTCWT show the selected grid's per-level edge exclusion and valid interior. Applying an analytical DTCWT additionally shows six native-resolution complex-magnitude maps at the chosen scale, one shared colour range, measured versus nominal angles and a marked valid inset. No cross-level interpolation is used; phase and atlas adjacency are not mislabelled as physical scalar structure.
*   **Statistics on every hypothesis (defect D8, presentation half):** the card labelled a bare `|r|` as "Confidence" and showed nothing else, which is what made nine noise correlations from a 9-run sweep read as nine discoveries. It now says **effect size**, and shows the q-value, the raw p-value, the family size, the correction procedure **with its dependence assumption**, and the R7 non-causality caveat. A finding with no correction gets an explicit warning rather than looking identical to a corrected one.
*   **Frontend/backend contract, checked mechanically (`src/tests/test_frontend_contract.py`):** `npm run build` runs `tsc`, so the frontend's internal types are checked; nothing checked them against the backend, and the payloads that matter are `Record<string, any>` because their shape is nested. The tests parse `api.ts` for every path it fetches and assert each is served (distinguishing a path parameter from a query string), assert every service method is actually called from `App.tsx`, and assert every nested key the UI reads exists in a real response. It found a live bug on its first run: the hypothesis card rendered `statistics.correction`, which is an **object**, and would have thrown *"Objects are not valid as a React child"* while `tsc`, `vite build` and 547 backend tests all passed.
*   **API Integration:** `src/services/api.ts` covers every backend endpoint via `fetch` against the relative base `/api/v1`. The offline fallback lives in `App.tsx`, not in the client service - each tab catches the network error and substitutes a local mock generator (`getMockDatasets`, `runMockFieldGenerator`, `applyMockPerturbation`, and the mock lineage fixture at `App.tsx:647`). The relative base URL means the frontend normally goes through the Vite dev proxy (`vite.config.ts`); since T3.5.2 the API also declares `CORSMiddleware` with an origin allowlist from `CORS_ALLOW_ORIGINS`, so a direct cross-origin call works too (defect D5).
*   **Not yet verified:** the "zero-error strict TypeScript compile / clean production bundle" claim made in earlier revisions of this document is **not substantiated**. `frontend/node_modules` does not exist, `frontend/dist/` contains only `index.html` with no emitted JS or CSS assets, and `tailwind.config.js` / `postcss.config.js` are both **missing** while `src/index.css` uses `@tailwind` directives and `@apply`. A build would therefore either fail or emit an unstyled page. Fixed and actually verified in Tasks 3.5.3 and 3.5.11.

---

## 7. Verification Status and Known Defects

This section is the honest ledger. It exists so that no future reader has to rediscover these facts, and so that the Phase 4 research layer is not built on top of an unverified base.

### 7.1 Execution status

**As of 2026-08-19 the platform has been installed, executed and tested for the first time.**
See `VERIFICATION.md` for the captured command output behind every statement here.

| Item | Status |
|---|---|
| Python venv + dependencies | installed (torch 2.13.0+cu130, numpy 2.2.6, pydantic 1.10.26, SQLAlchemy 2.0.52, xarray 2025.6.1, FastAPI 0.110.3) |
| Backend test suite | **2745 passed, 1 xfailed** (plus 4 skipped: the opt-in live GCS read, opt-in live store probe, opt-in live Argo acceptance, and opt-in live TESS/MAST acceptance) (was 8 failed / 11 passed at first run; 65 after T3.5.0, 152 after T3.5.7, 222 after T3.5.13, 286 after T3.5.17, 351 after T3.5.6, 379 after T3.5.15, 407 after T3.5.19, 449 after T4C.5, 709 after T4A.4, 781 after T4B.4, 855 after T4C.5, 859 after T4C.5c, 882 after T5.1a CPU acceptance, 883 after RTX acceptance, 890 after portable profiles, 911 after T5.1b/D44, 917 after T5.1c, 933 after T5.1d/D45, 946 after T5.1e, 955 after T5.2a, 957 after T5.2b, 962 after T5.3a, 969 after T5.3b, 981 after T5.2c offline acceptance, 985 after T5.2d, 1000 after T5.0a, 1008 after T5.0b, 1027 after T5.6a offline acceptance, 1042 after T5.6b cube acceptance, 1056 after T5.6c matched evaluation, 1070 after T5.6d truth matching, 1080 after T5.6e orchestration, 1089 after T5.6f portable jobs, 1094 after T5.6g reporting, 1095 after the licence guard, 1102 after T4C.5d gate readiness, 1104 after D50 storage preflight, 1106 after T4C.5e overlap evidence, 1112 after T4C.5f campaign acceptance, 1116 after T4C.5g physical preflight, 1117 after T4C.5h preregistration - the `master` freeze; then on `ed-dev`, 1375 after TG2.1, 1429 after TG2.2, 1477 after TG2.3, 1536 after TG2.4, 1577 after TG3.1, 1621 after TG3.2, 1686 after TG3.3, 1742 after TG3.4, 1787 after TG3.5, 1850 after TG4.1, 1922 after TG4.2, 1972 after TG4.3, 1986 after TG5.1, 2004 after TG5.2 and 2020 after TG5.3, 2044 after TG6.1, 2074 after TG6.2, 2112 after TG6.3, 2167 after TG7.1, 2217 after TG7.2, 2235 after TG7.3, 2236 after TG7.3 live acceptance, 2296 after TG7.4, 2321 after TG9.1/TG9.2 2334 after TG9.3, 2367 after TG8.1, 2420 after TG8.4, 2459 after TG10.1, 2503 after TG10.3, 2509 after TG10.2 2511 after TG11.0, 2521 after TG11.1, 2543 after TG11.2, 2570 after TG11.3, 2619 after TG11.4, 2661 after TG11.4b, 2667 after TG11.6, 2681 after TG11.5, 2694 after TG12.1b/TG12.1c, 2708 after TG12.2a, 2716 after TG12.1d, 2726 after TG12.2b-d, 2741 after TG13/G14 file-first ingress, and 2745 after TG15 capability routing) |
| Ground-Truth Benchmark Suite | **29 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE** (`python -m src.benchmarks`, exit 0) |
| Frontend `npm install` + `npm run build` | passes, emits 1,395 modules + real JS/CSS assets (was: 1 module, no assets) |
| Backend server | starts, serves OpenAPI, all smoke-tested endpoints return 200 |
| End-to-end experiment sweep | 9-run parameter sweep completes 9/9, writes 28 lineage nodes / 54 edges, hypothesis engine returns results |
| Version control | active Git history captures implementation slices; scientific run receipts carry their own content identities rather than treating the current commit as data provenance |

Earlier revisions of this document and of `roadmap.md` claimed the platform was "validated"
and "zero-error". It was not: the first real execution produced 8 test failures and a frontend
that had never rendered. The ledger below has grown from 18 entries to **68** as a direct result
of running the code and building tests against independent answers — **66 are fixed, D18 is
partial, and D43 remains open**.

Defects D26-D31 were all found *after* the code they concern was written and passing, by
tests written against analytic answers rather than against the code's own behaviour. Six of
the thirty-one were found this way, which is the single strongest argument for the standing
tolerance rule R16.

~~**Still not verified:** the frontend's rendered appearance in a browser ... The frontend also
does not yet consume `GET /api/v1/health`, `GET /api/v1/experiments` or the benchmark
endpoints.~~ **Superseded.** The health, experiment-listing, benchmark, statistics, registry,
export, import and ERA5 endpoints are all consumed as of T3.5.22-T3.5.24, and a test asserts no
served route is unreachable from the UI. The platform was started and the nine tabs confirmed
working by the user on 2026-08-20 (T3.5.25). **No screenshot per tab has been captured**, so
that clause of T3.5.0 remains open: the rendering is attested by a user, not evidenced by an
artefact in this repository.

### 7.2 Confirmed defects

| # | Location | Defect | Fixed by |
|---|---|---|---|
| D1 | `transform_engine/transforms.py:148-151` | DTCWT tree-B filters are identical to tree A up to sign - no Hilbert pair, no shift invariance, no directional subbands. Exact reconstruction hides it. | **FIXED** T3.5.6 |
| D2 | `transform_engine/transforms.py` `inverse_hybrid` | Recombines `w*low + (1-w)*high` where the forward pass split `low + residual`; not an inverse for any `w`. | **FIXED** T3.5.5 |
| D3 | `tests/test_experiments.py:49` | `test_experiment_engine_execution` **fails**. The test seeds an in-memory SQLite engine, but `execute_experiment` opens the file-backed `SessionLocal` from `database/session.py:11`, never finds the experiment, and returns early leaving status `PENDING`. | **FIXED** T3.5.1 |
| D4 | repo root | No `conftest.py` and no `__init__.py` anywhere, so bare `pytest` cannot resolve `src.*` imports (only `python -m pytest` works, via implicit cwd insertion). | **FIXED** T3.5.1 |
| D5 | `api/main.py` | No `CORSMiddleware`. The API is reachable only through the Vite dev proxy; any separate-origin or static-hosted deployment breaks. | **FIXED** T3.5.2 |
| D6 | `frontend/` | `tailwind.config.js` and `postcss.config.js` absent while `index.css` uses `@tailwind` and `@apply`. Build fails or emits unstyled output. | **FIXED** T3.5.3 |
| D7 | `start_platform.ps1:48-50` | uvicorn is launched inside `Start-Job`, which gets a fresh runspace rooted at the user home directory, so `src.api.main` is not importable and the backend dies silently while the frontend appears to start fine. | **FIXED** T3.5.4 |
| D8 | `hypothesis_engine/engine.py:110-143` | Emits a hypothesis for every parameter x metric pair exceeding `abs(r) >= 0.3` with **no multiple-comparison control**. Tolerable at current scale; becomes a false-discovery generator the moment Phase 4 adds scales, lags and constellations to the search space. | **FIXED** T4C.5 |
| D9 | `data_layer/adapters.py` | Class-level `_datasets` cache is never invalidated; a newly added `.nc` file is ignored until process restart. GRIB is advertised in a code comment but unimplemented. | **FIXED** T3.5.9 |
| D10 | `requirements.txt` | Missing `ipywidgets`, `plotly` and `matplotlib`, which `research_playground.ipynb` requires per the README. `alembic` also absent. | **FIXED** T3.5.11 |
| D11 | `.vscode/launch.json` | The Chrome configuration declares `"name"` twice; VS Code silently keeps the last, so the compound `Debug Platform (Both)` reference is fragile. | **FIXED** T3.5.4 |
| D12 | `data_layer/adapters.py:79`, `synthetic_generator/perturbation.py:43,46,50` | **Unseeded RNG.** `np.random.randn` and `torch.randn_like`/`rand_like` are called with no seed and no seed capture. Every noise perturbation and the simulated GFS wind field are irreproducible, which **directly contradicts the provenance pillar** - a lineage graph that cannot reproduce its own run is a record, not provenance. | **FIXED** T3.5.19 |
| D13 | `analysis_engine/diagnostics.py:68-69`, `boundary_lab/boundary.py:107` | **Metric-unaware differential operators.** `torch.gradient` is called with no `spacing`, so gradients are per-pixel, not per-metre. On a lat/lon grid the zonal spacing varies as `cos(lat)` and differs from the meridional spacing, so gradient magnitude, gradient angular error and boundary gradient decay are all systematically distorted, worsening toward the poles. Radial PSD likewise bins in pixel wavenumber and assumes isotropy that a lat/lon grid does not have - so the Charney/Kolmogorov regime classification is being made in the wrong space. | **FIXED** T3.5.13 |
| D14 | `api/main.py` (10 sites) | Every `except` re-raises a generic 500 with a fixed string (`"An error occurred during ..."`), discarding the exception entirely. For a research tool this is the difference between a usable diagnostic and a dead end. | **FIXED** T3.5.14 |
| D15 | `data_layer/adapters.py:150,162`, `experiment_engine/engine.py` | **The "seams" described in Section 5 are not extension points.** `_get_simulated_fallback` is a hard-coded if/elif over three dataset ids, `list_datasets` iterates a hard-coded literal list, and `_execute_action` is a 13-branch if/elif chain. Adding a data source or a pipeline action requires editing core engine files. | **FIXED** T3.5.15 |
| D16 | `physical_core/field.py:20` | `PhysicalField.__init__` force-casts to float32 with no opt-out. Acceptable for visualisation; marginal for surrogate ensemble statistics, log-log power-law fits, and mutual-information/transfer-entropy estimation in Phase 4C. | **FIXED** T3.5.16 |
| D17 | `analysis_engine/diagnostics.py`, `analysis_engine/decomposition.py`, `boundary_lab/boundary.py` | **Per-bin Python loops over full arrays.** Five radial/distance paths repeatedly masked the entire field. All now use single-pass grouped reductions or vector construction; independent loop oracles preserve results. The last 512x512/256-ring path measured 148.8 ms mean before and 7.70 ms after (19.3x). | **FIXED** T3.5.20 |
| D18 | `experiment_engine/engine.py:18-22` | `get_execution_device` probes CUDA only. No Apple-silicon MPS branch and no explicit CPU-thread configuration, so a large class of development laptops silently runs the slowest available path. | **PARTIAL** T3.5.21 - selection chain, override, refusal path and thread budget implemented; CUDA execution is now verified on the RTX 5050 for T5.1a, but whole-platform CPU/CUDA agreement and ROCm/MPS hardware remain unverified |

### 7.2b Defects found by executing the code (T3.5.0)

Static reading found D1-D18. Running the platform found seven more, five of them fatal to
code paths that `architecture.md` previously described as implemented and rigorous.

| # | Location | Defect | Status |
|---|---|---|---|
| D19 | `frontend/index.html` | **No `<script type="module" src="/src/main.tsx">` tag.** Vite had no entry point, `main.tsx` was orphaned and `App.tsx` never mounted — the UI had never rendered. `npm run build` nonetheless **exited 0**, transforming 1 module and emitting no JS/CSS, which is exactly how the "produces optimized production bundles cleanly" claim survived. | **FIXED** T3.5.3 |
| D20 | `transform_engine/transforms.py` (`apply_dwt2d`, `apply_dtcwt2d`) | `torch.nn.functional.pad` with a 4-element pad on a **2D** tensor raises `NotImplementedError` on torch 2.x. Every DWT and DTCWT call failed, taking `compute_wavelet_energy` and therefore `compute_diagnostics` with it. | **FIXED** T3.5.0 |
| D21 | `boundary_lab/boundary.py:31` | `torch.cos()` was called on a Python float in the Tukey taper, raising `TypeError`. Tukey is the **default** window in the API and the frontend, so the Boundary-Condition Lab had never worked. | **FIXED** T3.5.0 |
| D22 | `tests/*` | `sqlite:///:memory:` gives each *connection* a private database, and `TestClient` runs the app in a different thread — so fixture-created tables were invisible to the endpoint under test (`no such table: experiments`). Needs `StaticPool`. | **FIXED** T3.5.1 |
| D23 | `transform_engine/transforms.py` `_get_haar_filters` | `1.0 / np.sqrt(2.0)` is a `numpy.float64`, so `torch.tensor([...])` inferred **float64** filters and `conv2d` failed with "expected scalar type Float but found Double" against float32 fields. Latent until numpy 2.x. | **FIXED** T3.5.0 |
| D24 | `api/main.py` `create_experiment` | The background task called `execute_experiment` with no session binding, so sweeps launched via the API always wrote to the file-backed database regardless of caller context. Now resolved from `app.state.session_factory`. | **FIXED** T3.5.0 |
| D25 | `transform_engine/transforms.py` (`inverse_dwt2d`, `inverse_dtcwt2d`) | `torch.stack([...], dim=1)` on **2D** tensors yields `(H, 4, W)` where `conv_transpose2d` needs `(N, C, H, W)`. Every inverse wavelet transform was dead code; multi-level inversion also failed to crop a padded approximation band. | **FIXED** T3.5.0 |
| D26 | `analysis_engine/diagnostics.py` `fit_spectral_slope` | **The turbulence regime classifier was off by one exponent.** The annulus mean of `\|F(k)\|^2` estimates the 2D spectral density `S(k)`, but the reference values it was compared against - 5/3 (Kolmogorov) and 3 (Charney/Kraichnan) - are defined for the 1D energy spectrum `E(k)`, and `E(k) = 2*pi*k*S(k)` in 2D. Confirmed by construction: a synthetic field built with a textbook Kolmogorov spectrum was labelled *"Charney/Kraichnan 2D Enstrophy Cascade"* - the opposite physical regime. **Every regime label the platform ever emitted was wrong.** | **FIXED** T3.5.13 |
| D27 | `physical_core/grid.py` (introduced and fixed within T3.5.13) | `torch.fft.fftfreq` returns **float32** by default; a trailing `.to(float64)` preserves an already-rounded value. Capped the Parseval identity at a systematic 5.8e-8 - half of float32 epsilon - for every field. Same class as D16, found because the identity was checked to machine precision instead of to a plausible tolerance. | **FIXED** T3.5.13 |
| D28 | `physical_core/operators.py` (found within T3.5.13) | The polar-degeneracy guard tested `dx_metres > 0`. `cos(90 degrees)` evaluates to 6.1e-17 in float64, not 0, so a polar row had a zonal spacing of ~1e-12 m: finite, passing the guard, and producing gradients of order 1e11 per metre with no NaN to flag them. Now guarded against a physical length threshold. | **FIXED** T3.5.13 |
| D29 | `api/main.py` `DiagnosticsResponse` (found within T3.5.13) | The API reported `k_units = "rad m^-1"` for a field carrying no physical metric at all - a plot axis in metres over data that never had metres, which is D13's failure mode reappearing in the serialisation layer. Unit labels now derive from the grid kind, and kilometre-based units are refused on a pixel grid. | **FIXED** T3.5.13 |
| D30 | `analysis_engine/climatology.py` (found and fixed within T3.5.17) | **A result that changed depending on what ran before it.** The harmonic climatology basis for a short record is near rank-deficient (condition number 8.4e13, smallest singular value 2.5e-13). `torch.linalg.lstsq`'s default driver made its own rank decision there, and that decision flipped with prior BLAS state: identical data and seed gave a residual variance ratio of 0.000172 in one test ordering and 0.157639 in another. Now solved via column normalisation plus an SVD pseudo-inverse with an explicit rank tolerance; conditioning improved to 1.9e4 and the effective rank is reported. | **FIXED** T3.5.17 |
| D31 | `api/main.py` `/api/v1/benchmarks/run` (found within T3.5.17) | An unknown benchmark name filtered the suite to nothing and returned HTTP 200 with zero failures - a silent no-op that reads as "everything passed". A gate a typo can delete is not a gate. Now 404 with the list of available benchmarks. | **FIXED** T3.5.17 |
| D32 | `api/main.py` lifespan / `database/session.py` (found within T3.5.8) | **A schema that startup reported as correct and could not be queried.** `Base.metadata.create_all` adds missing tables but never adds missing *columns*, and returns successfully either way. The checked-in `spectral_earth.db` predated the five columns added by T3.5.12 and T4C.5, so the ORM mapped columns the file did not contain: `no such column: experiment_runs.seed`, with no warning at any point before the query. Fixed by Alembic revisions 0001/0002 plus `ensure_schema`, which adopts a pre-Alembic database by *inspecting* it rather than assuming its revision. | **FIXED** T3.5.8 |
| D33 | `requirements.txt` / `data_layer/builtin_sources.py` (found within T3.5.18) | **A documented feature that could not work.** `xarray` was declared but *no NetCDF engine* was, so a clean install had only `scipy` (NetCDF3 classic) and `zarr`. `data/README.md` invites the researcher to drop an ERA5 `.nc` file into `data/`, and `LocalNetCDFSource` is the highest-priority source - but modern ERA5 downloads are NetCDF4/HDF5, which failed with *"found the following matches ... but their dependencies may not be installed"*. Reproduced on this machine, then fixed: `h5netcdf` + `h5py` declared and installed, with a test that writes and reopens an HDF5-format file so dependency drift cannot silently undo it. | **FIXED** T3.5.18 |
| D34 | `api/main.py` `/api/v1/synthetic/perturb` (found within T3.5.23) | **Reproducibility that existed only in the Python API.** `PerturbationEngine.add_noise` has accepted a seed since T3.5.12 and this endpoint never passed one, so every perturbation requested over HTTP was drawn from the global RNG. The engine recorded `seeded: False` in its own metadata and nothing surfaced it. The frontend then compounded it, building its synthetic "forecast" in the browser with `Math.random()` - unseeded, and *uniform* despite the control being labelled StDev - so no diagnostic computed from it could ever be reproduced. Seed now threaded through, with `provenance` and a `reproducible` flag on the response. | **FIXED** T3.5.23 |
| D35 | `api/main.py` imports (found by starting the platform, T3.5.25) | **A fallback chain that depended on browsing order.** Source registration is an import side effect, and `zarr_source` was imported only lazily inside its own handlers. Measured on the running server: `GET /data/sources` returned `['netcdf_local', 'simulated']` on a fresh process and `['netcdf_local', 'era5_zarr', 'simulated']` after the ERA5 tab had been visited - so which sources `resolve()` searched depended on which endpoint the researcher happened to call first. Now imported at the composition root, with a subprocess test asserting a fresh interpreter registers all three. | **FIXED** T3.5.25 |
| D36 | `api/main.py`, `analysis_engine/decomposition.py`, `boundary_lab/boundary.py` (found by starting the platform, T3.5.25) | **The double-precision core was discarded at the HTTP boundary.** Ten sites cast every incoming field to `torch.float32`. Measured on the identical field: an FFT round trip is **2.8e-16 in float64 and 1.9e-07 in float32** - nine orders of magnitude. Every number a researcher saw through the UI carried float32 error, while the benchmarks, the Parseval checks and the tight-frame constants were all verified in float64. D27 had been fixed precisely because a float32 `fftfreq` capped Parseval at 5.8e-8; casting the data itself gave all of that back. Found on the first live run by the measured round-trip tolerance that replaced the transform tab's unconditional green ticks. | **FIXED** T3.5.25 |
| D37 | `data_layer/exporters.py` (found by starting the platform, T3.5.25) | **CSV export was lossy and nothing said so.** Fields were written with `%.10g`; IEEE-754 double needs **17** significant digits to round-trip, so seven were silently discarded. Found on a live export/import loop through the running server - CSV was the only format that came back changed, and only comparing the arrays revealed it. This is the same precision D36 was fixed to stop throwing away at the HTTP boundary, discarded again one layer out in the file format. Now `%.17g`, with a bit-exactness test across all four formats; the cost is about 50% more bytes. | **FIXED** T3.5.25 |
| D38 | `artifact_store/store.py` (found while building T4B.1) | **A complex summary that silently described only the real part.** `summarise` called `float(values.min())` unconditionally. For a complex array that does not raise: numpy casts to real, discards the imaginary part, and warns where nobody reads it - `[1+2j, 3-1j]` reported `min = 1.0`. Every DTCWT coefficient field is complex, so the lineage rows of an entire phase would have carried real-part statistics labelled as statistics of the array. Complex arrays are now summarised on their **magnitude** and say so under `statistic_of`. | **FIXED** T4B.1 |
| D39 | `artifact_store/store.py` (found while building T4B.3) | **The store dropped the time axis of a `FieldSequence`.** `put` stored the `(T, H, W)` tensor and the grid but not the timestamps, so `load` returned an array that was not a sequence - and any caller rebuilding one would have assumed a regular cadence, silently mis-dating every frame of an irregular record. The irony is exact: 4A existed to give the platform a time axis, and the store built in the same slice discarded it. Artifacts now carry their own axes inside the `.npz` (times as an array member, labels and grid as one JSON member, `allow_pickle=False`), with `load_sequence` / `load_coefficient_field` rebuilding the real object. The 4 KB handle budget is unaffected: the axes went into the archive, not the database row. | **FIXED** T4B.3 |
| D40 | `roadmap.md` rule R13's crop-size table (found while building T4C.1) | **The valid-interior table understates the dual tree by nearly a factor of two.** The table is derived for a single 14-tap filter repeated at every level; DTCWT uses a 19-tap near-symmetric highpass at level 1 and the q-shift pair above it, so the real level-4 margin is **97 parent pixels against the table's 52**. The consequence is concrete rather than theoretical: a 256x256 crop - the roadmap's stated practical minimum for four dyadic levels - leaves DTCWT level 4 a **2x2** valid interior, four coefficients per orientation, on which a participation ratio is almost pure sampling noise. `dtcwt.filter_support` now accumulates the actual cascade, and `scale_signature` reports such a scale as *thin* by name instead of averaging over it. | **FIXED** T4C.1 |
| D41 | `data_layer/zarr_source.py` vs `transform_engine/stationary.py` (found while building T4C.1) | **Two implementations of rule R13 disagreed by one pixel per side at level 1.** `zarr_source.valid_interior` floored the half-integer radius of an even-length filter while `stationary.valid_interior_halfwidth` used the conservative effective-support halfwidth. The crop module therefore declared one contaminated pixel per side valid at level 1. `edge_exclusion` now derives and halves the full effective support, the R13 table is corrected (`N=64, level 1: 52 -> 50`), and the existing geometry test asserts agreement with the SWT implementation so the definitions cannot drift independently again. | **FIXED** T4C.5a |
| D42 | `data_layer/adapters.py`, `experiment_engine/actions.py` (found preparing T4C.6) | **Real ERA5 existed beside the Phase 4 pipeline, not inside it.** The Zarr API could inspect and materialise a crop, but `slice_sequence` supplied only a dataset id; the registered source requires the crop specification and therefore could never serve the action that every Phase 4 stage uses. The adapter also assumed `lat`/`lon`, while WeatherBench uses `latitude`/`longitude`. Parameterised source options now flow through the action without entering the unsafe id-only cache, coordinates are normalised onto the physical spine, and source request/provenance survives on the sequence. An offline WeatherBench-shaped test runs cached crop -> `FieldSequence` -> registered action -> artifact and asserts observational, non-simulated provenance. | **FIXED** T4C.5b |
| D43 | `data_layer/zarr_source.py`, `data_layer/cds_source.py` / T4C.6 data design | **The real-data gate is not laptop-feasible through the catalogued WeatherBench layouts.** The supposedly compromise 0.7-degree store is chunked `(8,13,512,256)`: every eight-frame read transfers all levels and the globe. Live metadata inspection for a three-year, one-variable, 255x255 request estimated 29.88 GB fetched for 1.14 GB wanted (26.2x); the 0.25-degree archive is worse. T5.2c now supplies an offline-accepted, resumable direct regional CDS acquisition and canonical-cache path, but no live CDS request, multi-year NZ crop or cross-route overlap has run. Close only after that acquisition and verification evidence exists, then freeze the crop and run T4C.6. Do not reduce sample or edge-validity requirements to fit the old layout. | **OPEN - acquisition contract implemented; live data gate not run** |
| D44 | `transform_engine/stationary.py:filter_support` / `data_layer/zarr_source.py:edge_exclusion` (found while building T5.1b) | **The generic R13 budget discarded inherited low-pass support.** It counted only the filter newly applied at level `j`, `(L-1)2^(j-1)+1`, although an SWT coefficient has passed through every preceding low-pass stage. The complete cascade is `1+(L-1)(2^j-1)`. For db2 the level-4 margin changes from 12 to 23 pixels; for the declared generic 14-tap budget it changes from 52 to 98, moving the four/five-level 128-valid-pixel floors from 256/512 to 512/1024. Both implementations, their tests, the tier table and R13 documentation now use the accumulated support. An independent convolution of the dilated filters tests the composition rather than merely repeating the formula. Historical D40/D41 measurements remain recorded but are superseded wherever they relied on the generic table. | **FIXED** T5.1b |
| D45 | `transform_engine/training.py` convolution paths (found by the first full T5.1d run) | **A transform whose numerical result depended on what ran before it.** `enable_determinism` selects a different cuDNN convolution algorithm; with ambient TF32 enabled, the RTX SWT round-trip maximum error changed from **2.38e-7 to 4.48e-4** on the same seeded input. Focused tests passed because they started in fresh process state; the ordered full suite exposed it. All training convolution calls now scope `allow_tf32=False` locally and restore the caller's policy. A regression test deliberately enables deterministic cuDNN plus TF32, asserts the 3e-6 reconstruction tolerance, and asserts the ambient flag is restored. | **FIXED** T5.1d |
| D46 | `data_layer/cds_source.py:materialise_cds` | **A resumable download followed by an unbounded conversion.** Monthly shards were each `.load()`ed, retained in a list and concatenated into the complete multi-year five-variable record before Zarr writing. The proposed D43 route could therefore require tens of GB of RAM even though the final cache was chunked. Conversion now validates and appends bounded time blocks to a sibling store, streams its logical content hash and atomically publishes only after exact whole-axis validation. | **FIXED** T4C.5d |
| D47 | `analysis_engine/climatology.py`, `scale_signature.py`, `transform_engine/coefficient_field.py` | **The real gate's analysis path required whole-record RAM.** Climatology constructed `(T,H,W)` and decomposition constructed `(T,S,O,H,W)` before reducing to a small signature; thousands of 512x512 frames make that path larger than laptop memory. Train-only harmonic fitting and exact record-level signature extraction now retain one source/coefficient frame, carry boundedness evidence and reject source mutation between passes. | **FIXED** T4C.5d |
| D48 | `analysis_engine/cross_scale.py:support_floor` | **The support floor did not use filter support.** Its prose and result basis said advective crossing of the transform filter, but the implementation used `2**level`; db2 level 3 was treated as 8 pixels although its accumulated cascade support is 22, understating the lags contaminated by shared air. `ScaleSignature` now carries the exact SWT/DTCWT parent-grid support and the floor refuses to invent a fallback. | **FIXED** T4C.5d |
| D49 | `analysis_engine/cross_scale.py`, `gate_run.py` | **A PASS/FAIL was not authenticated to the complete study.** `GateProtocol` omitted crop/variable/level/transform/climatology/source identity, and `evaluate_replication_gate` did not require result fingerprints or exact bins/alpha/correction/surrogate count. `GateStudyPlan` freezes the full job; preflight and receipt bind both split results to it, while a real role requires CDS plus independent-overlap evidence. | **FIXED** T4C.5d |
| D50 | `data_layer/cds_source.py` | **Live acquisition had no enforced disk-capacity gate.** Monthly resume and bounded conversion protected correctness and RAM, but a request could still fill the download/cache volume mid-run. `preflight_cds_storage` now budgets remaining NetCDF shards and the complete temporary Zarr without compression credit, combines roles sharing a volume, preserves at least 5 GiB or 10% working-space reserve, records the check and refuses before the first client call. | **FIXED** T4C.5d |
| D51 | `analysis_engine/gate_run.py`, `data_layer/regional_forecast.py` | **The real gate trusted an unauthenticated manifest string.** The independent ERA5 checker returned an in-memory report, but no executable path published it; manually changing `independent_overlap_check` to `PASS` satisfied gate preflight. `era5_overlap.py` now performs bounded exact-coordinate CDS/WeatherBench comparison, publishes immutable content-bound evidence, and the gate recomputes and validates its receipt hash, source/request identity, variable and level. | **FIXED** T4C.5e |
| D52 | `analysis_engine/gate_campaign.py` | **The expensive record was the first live integration test.** Acquisition, independent overlap and gate plans existed separately, so request/grid/cadence drift remained possible and the full multi-year CDS transfer could complete before discovering a decoder or cross-route mismatch. A frozen two-stage campaign now requires a small exact canary PASS first, binds all identities and ordering, and preflights aggregate storage/dependency/configuration/consent without network use. | **FIXED** T4C.5f |
| D53 | `analysis_engine/cross_scale.py:support_floor` | **ERA5 angular spacing was interpreted as metres.** `GridSpec.to_provenance()` stores lat/lon `dx` in degrees, but the support floor selected that field first and multiplied it directly by filter pixels. At 0.25° this understated the physical footprint by roughly five orders of magnitude. Physical grids are now reconstructed, angular spacing converted, and the maximum physical cell axis over the crop used conservatively. | **FIXED** T4C.5g |
| D54 | `analysis_engine/gate_campaign.py`, `data_layer/cds_source.py` | **Spatial invalidity was discovered only after transfer.** Campaign validation bound request identities but did not prove grid-aligned bounds, actual-filter R13 interiors or physical lag admissibility; CDS ingestion checked spacing but not endpoints, so a server-snapped crop could pass. All are now exact pre-acquisition refusals, with returned endpoints independently rechecked. | **FIXED** T4C.5g |
| D55 | `core/executor.py:_run_one` | **Per-task seeding is process-global, so the thread backend corrupts it.** `_run_one` calls `torch.manual_seed` and `np.random.seed` — both process-global — then invokes the task. Under `ThreadExecutor` every worker shares one generator, so a second worker's seed overwrites the first's stream before the first draws. Measured: with the seed-to-draw window held open by 10 ms of work, thread(8) disagreed with serial in **10 of 10** trials; with a microsecond-long task it disagreed in 0 of 80, because the tasks effectively serialise. Real sweep payloads are the former. This silently breaks standard E4 and re-opens D12's guarantee for `execution.backend: thread` with `n_workers > 1`, and it falsifies `roadmap.md`'s claim that "a sweep is byte-identical across executor backends" — true for `serial` and `process`, not for `thread`. `test_a_runs_seed_does_not_depend_on_which_worker_took_it` passed by timing luck and was observed failing once under full-suite load. Fixed properly rather than cheaply: streams are now per task, bound to a `contextvars.ContextVar` in `core/randomness.py` and released with the task, so no worker can reach another's. Torch values are unchanged (`Generator().manual_seed(s)` matches the global generator after `manual_seed(s)`); `enable_determinism` no longer seeds the process and reports `seed_scope` instead. The alternative — refusing `n_workers > 1` when seeds are supplied — was rejected: it removes the symptom and leaves process-global randomness inside a unit of work the platform runs concurrently. See section 3.6j. | **FIXED** TG1.2 (`ed-dev`) |
| D58 | `analysis_engine/domain_analysis.py:run_domain_gate` | **The advective path through the domain gate was unreachable.** `run_domain_gate` refused `require_advection_floor` for any domain not declaring `lag_policy='advective'`, and then called `analyse_precedence` with no parameter through which a transport speed could travel - which the advective policy requires and is deliberately denied a default. Every advective domain gate therefore raised "a declared transport speed under lag_policy='advective'" before reaching the data. Neither half was wrong on its own, which is why no test caught it: the gate tests all used `declared` or `none` domains, and the advective tests never went through the gate. `run_domain_gate` now takes `lag_policy_params`, and `test_the_domain_gate_can_now_run_an_advective_domain` executes the combination. Found while localising the policy branches in TG1.3. | **FIXED** TG1.3 (`ed-dev`) |
| D57 | `experiment_engine/actions.py:perturb_field` | **A declared reproducibility parameter was silently discarded.** The action's registry entry advertised that noise `accepts \`seed\` for reproducibility`; the handler then called `PerturbationEngine.add_noise(field, noise_type, level)` and dropped the seed. A user asking for a reproducible perturbation received an unreproducible one with no error — the worst shape a reproducibility defect can take, because the request looks honoured. The seed is now passed through, and omitting it falls back to the task stream (D55). Found while giving the byte-identity acceptance sweep a payload that actually draws. | **FIXED** TG1.2 (`ed-dev`) |
| D56 | `physical_core/field.py:split_field`, `scale_resolution` | **A coordinate spelled in full was silently mishandled.** Both methods decided which coordinate was the row and which the column from hardcoded name lists - `["y","lat"]` and `["x","lon"]` - and the two lists disagreed about the fallback. A field carrying `latitude`/`longitude`, the spelling CF and ERA5 both use, matched neither: `split_field` cloned the longitude vector instead of slicing it, returning a narrowed field whose coordinate no longer described its own data and which `GridSpec.from_coords` would then read; `scale_resolution` interpolated latitude to the *column* count. Nothing raised. Both now resolve through `core/axes.resolve_axis_roles`, and a caller may declare `axis_roles` instead. The disagreement over genuinely unrecognised coordinates is preserved deliberately and asserted, so no pre-TG1.1 result moves. | **FIXED** TG1.1 (`ed-dev`) |

| D59 | `benchmarks/sequences.py:build_advected_vortex`, `truth_advected_vortex` | **The recorded trajectory wraps and the field does not.** `truth_advected_vortex` takes the vortex position modulo `n`, so its known answer is a trajectory on a torus; `build_advected_vortex` draws the blob with a plain Euclidean Gaussian, which is clipped at the boundary rather than wrapped. At the benchmark's own parameters the vortex never reaches an edge, so the two have never disagreed and the `4D.position` check passes to better than a cell. Started near an edge, the recorded position and the measured centroid part company by several cells for the frames where the blob is clipped. This lands on TG2.3, whose acceptance criterion is *1 track, 1 birth, 0 deaths* on this sequence: a tracker run on a wrapping parameterisation would see the object fade out at one edge and appear at the other, and would be right, against a recorded answer that says neither happened. Left unfixed deliberately - making the builder periodic would move every number the benchmark produces, and TG2.1 is not the slice that gets to do that - and asserted by a test so the next slice meets it as a fact. Found by giving TG2.1's record a real user. **Fixed in TG2.3**: the topology is now a declared parameter (`periodic`) that the builder, the recorded answer and both checks read (standard E14), rather than the builder assuming one and the answer the other. At the registered parameters every number the benchmark produces is unchanged - the modulo was a no-op there, which is exactly why it hid. `track_count`, `births` and `deaths` are now **derived** from an analytic `mass_inside_frame` rather than asserted as the constants 1, 1 and 0, so a sequence that advects the vortex out of the frame records the death that actually happens; and a second registered benchmark, `advected_vortex_periodic_sequence`, declares a torus and draws one, which makes the seam-crossing case answerable for the first time. | **FIXED** TG2.3 (`ed-dev`) |
| D60 | `core/review_cost.py:GeminiBatchTransport` | **The offline Gemini Batch fixture did not describe the live API.** A completed operation returns batch state/times in `metadata` and the `GenerateContentBatchOutput` separately in `response`; the first parser expected the reference's direct resource shape and refused a successful live batch. Worse, the legacy `responseMimeType` / `responseJsonSchema` fields were accepted but silently did not constrain Gemini 3.5 Flash, while current `responseFormat` requires protobuf enum `APPLICATION_JSON` on Batch even though synchronous REST examples show `application/json`. The first locally unvalidated smoke therefore returned valid JSON with the wrong fields. Live usage also separates visible candidates from billed thinking tokens. The parser now normalizes the real operation shape, uses the live-accepted enum dialect, validates output locally before success, and counts candidate plus thinking tokens as output while retaining raw usage. A subsequent live batch returned the exact declared `{status, note}` schema. | **FIXED** TG7.3 (`ed-dev`) |
| D61 | `core/builtin_domains.py:ORDER_BOOK` | **A domain's description and its declared violations disagreed for four slices.** `order_book` has read *"per-instrument order-book channels on an irregular trading clock"* since its first commit while omitting `irregular_sampling` from its violation tuple, so the analysis layer would have accepted a ragged record from it as regularly sampled and reported every lag in frames as a duration. Nothing caught it because nothing had yet tried to **read data** under the declaration — which is what rule R17 exists for and what an ingestion seam is for. Found by TG8.4's `read_channels_for_domain`, whose cadence check consults the declaration. The tuple is corrected, and the consequence is recorded rather than smoothed over: half of what TG8.1 credited to Argo was really this gap, so the plugin docstring and the coverage test now claim `non_stationary_support` alone. | **FIXED** TG8.4 (`ed-dev`) |
| D62 | `data_layer/stores.py:BUILTIN_STORES` | **A fabricated measurement, in the registry built to refuse fabricated measurements.** TG10.1 moved the ERA5 catalogue notes into fields and gave `era5_0p7_6h` a `megabytes_per_chunk` of 8.0. No inspection ever produced that number: the 2026-08-21 note records an amplification of 26.2x and an estimated 29.88 GB total and no per-chunk size, and 8.0 does not even follow from the chunk shape the note describes, which works out at 54.5 MB. It passed because `ChunkFacts` demanded a positive figure for any method other than `not measured`, so filling the field was the only way to record a real inspection — a validation rule that made the dishonest entry the easy one. **Not recorded** is now a third state distinct from **not measured**, and the entry states the amplification without inventing a size. | **FIXED** TG10.3 (`ed-dev`) |
| D63 | `data_layer/zarr_source.py:CropSpec.to_provenance` | **A schema grew under an artefact that was already signed.** TG10.1 added `vertical_dim` to `CropSpec`, and `to_provenance` emits `asdict`, so every lineage record gained a key. That record is embedded in **authenticated** artefacts — a gate campaign's preregistration is fingerprinted over it — so a checked-in, signed preregistration written before the field existed stopped loading, and `_crop_from_mapping`'s exact-fields check refused it by name. The content key had been protected against exactly this and the provenance record had not, which is the same decision applied to one of two consumers. `to_provenance` now omits `vertical_dim` when it is `level`, as `canonical()` does, so an ERA5 record is byte-identical to its pre-TG10.1 form; the campaign reader additionally tolerates the field's absence rather than demanding it of an old record. Found by the full suite, **after TG10.1 was reported and committed** — the slice was reported on the strength of targeted suites while the full run was still going, which is how a one-test regression reached a commit. | **FIXED** TG10.3 (`ed-dev`) |
| D64 | `tests/test_documentation.py:_routes` | **The guard that refuses an undocumented endpoint could not see an endpoint mounted at its router's own prefix.** Its path pattern was `[^"]+`, so `@router.get("")` - a real route at `/api/v1/<prefix>` - matched nothing. `GET /api/v1/acquisitions` had been served and invisible since TG10.2, and the documented route count read 42 against a served 43 while the count check passed, because the guard was comparing the claim against its own blind spot rather than against the API. TG11.1 added a second such route and the arithmetic was still self-consistent. Found by loosening the quantifier to `[^"]*` and watching the count move by two. This is the third time this guard has stopped covering new code silently, and its passing is read as assurance. | **FIXED** TG11.1 (`ed-dev`) |
| D65 | `api/preregistration.py:_identity` | **The held-out ledger's "once" could be defeated by renaming a file.** `PartitionIdentity.from_series` hashes the series' provenance wholesale, and a record read from an upload carries `path_basename` in that provenance. In process that is honest lineage; across an HTTP boundary the same held-out bytes re-uploaded as `data2.csv` would hash to a different partition, and `HeldOutLedger` - which is keyed on the partition precisely so that a second honest seal cannot buy a second look - would not fire. The domain the file was read under had the same problem, for the same reason: two confirmations on the same bytes are two tests of them, however they were labelled. Found while writing the double-spend acceptance, which passed against a re-upload under the same name and would have passed for the wrong reason. The identity is now built at the boundary from `content_sha256`, the clock, the columns and the split window, and a test renames the file between two identical requests and asserts the digest does not move. | **FIXED** TG11.2 (`ed-dev`) |
| D66 | `api/findings.py:StudyStore` | **The read surface resolved a study by filename, and a write path made that wrong.** `load` returned the first parseable file whose `study_id` matched and `summaries` listed one row per file. That was harmless while nothing wrote bundles - TG9.1 read a store that only a researcher populated, one file per study. TG11.3 publishes each revision as its own file, because a bundle is immutable and `save_evidence_bundle` refuses to overwrite, so the sorted-first match would have served `r00000` for ever while the write path reported the revision it had just appended, and one study worked on five times would have listed as five studies. Found by asserting through the read surface what the write surface had just returned, rather than trusting that two consistent stores agreed. Resolution is now by chain rather than by name: the highest revision wins, and superseded ones are counted into its row. | **FIXED** TG11.3 (`ed-dev`) |
| D67 | `data_layer/zarr_source.py:assess_access_pattern` | **The cost estimator could not cost the store it was most needed for.** Probing GLORYS raised `MemoryError` while dask built a sliced graph over a 12227 x 50 x 2041 x 4320 variable merely to read selected sizes. Costing now applies the materialiser's shared xarray indexers to coordinate arrays only, maps selected labels to source positions and counts the exact chunk ids touched. It transfers no values, preserves partial-date and descending-axis semantics, and distinguishes aligned from straddling selections. A regression makes the data-selection path raise `MemoryError` and proves Inspect still completes. | **FIXED** TG12.1 (`ed-dev`) |
| D68 | `data_layer/zarr_source.py:CropSpec`, `api/main.py:ZarrCropRequest` | **The store registry could declare GLORYS's `elevation` axis, but no request could select one of its values.** Both request types accepted integer pressure levels while all 50 GLORYS elevations are fractional negative metres. Registration without this fix would have advertised an unusable vertical selection. The existing `levels` field now accepts strict finite integers or floats; integer ERA5 records and pinned hashes do not move, exact fractional elevation selection is tested, and Acquire loads store-specific defaults rather than carrying 850 hPa into GLORYS. | **FIXED** TG12.1 (`ed-dev`) |
| D69 | `core/domain.py:KNOWN_VIOLATIONS`, `core/channel_series.py:ChannelSeries` | **A declared violation with no consumer, and no way to represent it.** `non_stationary_support` - *"channels start and stop during the record, so the effective sample size differs per channel and per pair"* - has been in the vocabulary since TG0.2 and is enforced nowhere: a grep of `src/` outside tests returns the dictionary entry that defines it and nothing else. Its two neighbours are enforced, which is what makes the gap a defect rather than a design: `irregular_sampling` refuses a ragged clock in `_validate_cadence` and refuses a frame-to-duration conversion in `DomainTimeSeries.cadence_seconds`, and `aggregated_values` is enforced in both directions - declare it and a positive physical window is required, omit it and carrying one is refused. `refusals_for` publishes this violation's consequence text to the API while its own docstring asserts that *"that refusal is enforced deep in the analysis layer"*, which for this entry is untrue. The representation is missing too, and is the same hole: `ChannelSeries.usable` carries **one entry per channel** and `unusable_reason` is keyed by channel label, so there is no per-sample presence mask in the contract - a channel that reported for part of the record is wholly usable or wholly unusable, and "present here, absent there" cannot be said. `measures` finiteness is unvalidated, so a union clock padded with `NaN` would carry absence into an analysis layer that has never been asked what it does with one. This is D61's shape at a larger scale, and it lands on the critical path: after D61 shrank Argo's justification to this one violation, TG12.2's acceptance would otherwise read *a record whose declared violation nothing acts on*. Found while writing TG12.2's design rather than while implementing it. The obvious workaround was then audited rather than assumed, and it is worse than a crash: a `NaN`-padded union clock produces a **full result table**, because both estimators already mask to their finite intersection (`cross_scale.py:370`, `:409`). The numbers would be computed on unrecorded, varying, pairwise sample sizes; referred to a shift-surrogate null that rotates the presence pattern with the values (`_shift_null`, `:520`) so each surrogate carries a different N than the observed statistic; guarded by a bias warning computed from the padded clock length (`per_cell`, `:573`); with a Theiler window taken from an autocorrelation that **closes the gaps up** and treats samples either side of a multi-year absence as adjacent (`decorrelation_frames`, `:453`), understating the window in the direction of false significance. The split copies `usable` wholesale across a boundary a channel may lie entirely on one side of, and `PartitionIdentity` cannot see presence at all, which would reopen D65 for two records identical but for their sampling. Eight sites in total, recorded as F1-F8 in the TG12.2a plan. | **FIXED** TG12.2a (`ed-dev`) — `ChannelSeries.present` now represents explicit boolean `(time, channel)` observation state; finite values at absent samples are refused while present non-finite values remain observed-invalid. The domain declaration and mask are enforced in both directions, splits recompute viability, identities hash the presence pattern without reading measures, and counts reach lineage/API/UI. The controlled Argo-like clock (20 floats, 146 cycles, 2,920 union rows, 380 ordered pairs) had zero pairwise overlap, so the selected scientifically admissible policy refuses intermittent-presence frame lags until a physical-time estimator exists. Gap-aware decorrelation and fixed-N null primitives are retained and tested without licensing compacted-frame inference; all-true masks are result-identical to the pre-existing path. |
| D70 | `data_layer/regional_forecast.py:assess_manifest_readiness`, `frontend/src/components/AcquisitionView.tsx` | **The workbench judged an ocean crop against an atmospheric requirement, and truncated its axis to do it.** The T5.2 readiness assessment shown against every materialised crop in the Acquire tab asks for five canonical ERA5 variables on the 850 hPa pressure level. It read the vertical selection as `[int(v) for v in spec['levels']]`, so GLORYS's fractional negative-metre elevations were silently coerced - `int(-0.49402499198913574)` is `0` - and the verdict returned `level_available=False` with `missing_variables=['t','q','u','v','z']`. Reproduced exactly before the fix. That reads as *a crop that nearly qualified and lacks some fields*, when the truth is that the question does not apply to an ocean crop at all; and the coercion is D68's defect surviving in a consumer that TG12.1 did not teach alongside `CropSpec` and `ZarrCropRequest`. The second half is what the reader would have paid for: the Inspect panel offered a 51 GB materialise command with nothing anywhere saying that no analysis path reads a crop from this store, so the transfer could complete before that was discovered. Levels are now compared as numbers with no coercion; the assessment declares `applicable` from the crop's own declared vertical axis (absent means `level`, per D63) and states in words that the variable and level rows describe what T5.2 requires rather than anything the crop failed to supply; the panel renders that instead of the verdict; and Inspect says before the transfer that materialising is where this store currently stops. ERA5 crops are unchanged in every field, asserted in the same test. Found by a researcher clicking Inspect on GLORYS and asking what to do next. | **FIXED** TG12.1a (`ed-dev`) |
| D71 | `core/publication.py:publish_new_bytes`, `data_layer/era5_overlap.py`, `analysis_engine/gate_run.py`, `analysis_engine/gate_campaign.py`, `forecasting/evaluation_run.py`, `forecasting/evaluation_job.py` | **The independent ERA5 receipt could not be published on the drive the repository lives on, and the same scientific guarantee had five private implementations.** The measured failure was narrower than the first ledger wording: `era5_overlap` called `os.link` unconditionally and failed on `D:` because exFAT has no hard links; the other four already used Windows rename and therefore did not share that particular failure. They did share an unowned semantics boundary whose implementations had already drifted. `publish_new_bytes` now writes and fsyncs a random same-directory temporary, then uses an OS atomic no-replace primitive: Windows rename, Linux `renameat2(RENAME_NOREPLACE)`, macOS `renamex_np(RENAME_EXCL)`, or POSIX hard-link fallback. It never check-then-renames and never replaces. Eight spawned publishers on the actual repository volume produce exactly one complete winner; existing bytes survive, injected failure leaves neither target nor temporary, all five writer suites pass, and both previously failing CDS overlap cases publish and reload on exFAT. The claim is process-crash atomicity; sudden-power-loss durability remains a filesystem/device property. | **FIXED** TG12.1c (`ed-dev`) |
| D72 | `data_layer/zarr_source.py:_main` | **The command the UI generates could not run against the store it names.** The Acquire tab's Inspect panel prints a ready-to-paste `materialise` line, and for GLORYS it failed twice over. `--levels` was parsed with `tuple(int(v) for v in ...)`, so the fractional elevation the panel itself supplied raised `ValueError: invalid literal for int() with base 10: '-0.49402499198913574'` before any work began; and the spec was built by calling `CropSpec(...)` directly rather than `crop_for_store(...)`, so `vertical_dim` took ERA5's default of `level` for every store, meaning even a corrected level would have been selected on an axis GLORYS does not have. `crop_for_store` already existed for exactly this and the HTTP route already used it (`api/main.py:1188`); only the CLI had been left behind. This is the **fourth** appearance of one defect: D68 taught `CropSpec` and `ZarrCropRequest` to carry a fractional level, D70 found the readiness assessment still truncating one with `int()`, and this is the entry point a researcher is actually told to use. Levels now parse integer-first, so ERA5's `850,700,500,300` stays integral and no pinned content key moves, while a non-integer literal is kept as a float and an unparseable one is refused by name. Found by a researcher running the command the UI gave them. | **FIXED** TG12.1b (`ed-dev`) |
| D73 | `data_layer/zarr_source.py`, `data_layer/crop_planner.py`, `api/main.py`, `frontend/components/AcquisitionView.tsx` | **Acquisition discovered transform invalidity after the expensive step, and the command did not preserve the analysis it appeared to plan.** Inspect used one generic 14-tap margin and returned geometry the UI did not render; the generated command omitted `--analysis-levels`, so changing the requested analysis depth still materialised under the CLI default. Crop sizing was therefore a hidden global convention rather than a contract of the requested transform. `TransformSpec` now owns an optional implementation-derived support callback; a content-addressed metadata-only plan reports absolute and named research-policy thresholds, per-level valid interiors, feasible native-coordinate expansion and its re-priced chunk cost. Acquire renders exact transform/filter/depth controls and applies recommended bounds in one action; the generated CLI preserves every setting; explicit materialisation refuses below the recommended threshold before field selection or transfer. Legacy callers remain byte-compatible when no analysis request is supplied. | **FIXED** TG12.1d (`ed-dev`) |

**Root cause common to D20, D23, D25 and D2:** the transform engine — the mathematical core of
the platform — had **no test file at all**. `src/tests/test_transforms.py` now exists (36 cases
across FFT, DCT, DWT, DTCWT and hybrid — two thirds of the whole suite) and covers round-trip exactness, energy conservation,
orthonormality, dtype preservation and shift behaviour.

### 7.2c Empirical confirmation of D1

Running the code settles D1 beyond static argument:

*   `max |LH_AA + LH_BB| == 0.0` exactly — tree BB is the **exact negation** of tree AA, so the
    `_real`/`_imag` pair carries no phase information and there is no Hilbert (analytic) structure.
*   DWT and DTCWT reconstruction MSE are **bit-identical** at every tested shape and level.
*   Shift-variance, measured on a sharp front translated 0-7 px: subband energy varies by
    **153%** and the magnitude envelope decorrelates to **corr < 0**. A shift-invariant
    transform would hold both roughly constant.

`test_dtcwt_is_shift_invariant` is committed as `xfail(strict=True)`, so it will **fail the build
the moment T3.5.6 makes it pass**, forcing the marker's removal rather than letting a stale
exemption linger. A first attempt at this test used a smooth vortex and passed against the
known-broken transform — recorded in the test docstring, because it is a concrete reminder that
a property test is only as good as the structure it probes (R8).

### 7.2d Live confirmation of D8 (no multiple-comparison control)

The end-to-end smoke sweep — 9 runs over `grid_size x transform_type` — caused the hypothesis
engine to emit **9 hypotheses**, including "strong positive correlation (r = 0.96)" between grid
size and floating-point reconstruction error. These are artifacts of testing every parameter
against every metric with no FDR control, on 9 samples. It is a preview of what R5 and T4C.5
exist to prevent, and it is why the current engine must not be pointed at real data until they land.

### 7.2e Two hypotheses of mine that measurement refuted (T3.5.13)

Recorded because both were written into module docstrings as fact before being checked, and
because the corrected versions are more useful than the originals were.

**1. "Pixel-radius binning biases the fitted spectral slope on an anisotropic grid."**
False. For a pure power law the angular anisotropy factor
`<(cos^2/a^2 + sin^2)^(-beta/2)>` is independent of radius, so it rescales the *intercept*
and leaves the slope exact. Measured slope error at aspect ratio 4: **+0.001**. What pixel
binning genuinely destroys is the physical meaning of `k` and the *shape* of the spectrum -
a Gaussian spectral peak on an aspect-3 grid had its power-weighted relative spread go from
**0.021 to 0.36, a factor of ~16**, while physical binning was unchanged. Since a
`ScaleSignature` (T4C.1) is built from spectral features and compared across latitudes,
this matters more than a slope bias would have.

**2. "The Laplacian's `valid_mask` marks the accurate region."**
False as first written. Computing the meridional flux term as gradient-of-gradient widens
the stencil to +/-2 rows, so the first call's first-order edge row contaminated the *second*
row - **25% relative error at row 1** while the true interior held 4e-6, inside a mask that
claimed the row was valid. Replaced with a staggered half-point flux form, which is compact
(one invalid ring, truthfully marked) and conservative. Both are asserted head-to-head in
`test_staggered_flux_form_beats_nested_gradients_at_the_second_row`.

**Method note.** Both were caught by testing against an *analytic* answer rather than
against the code's own inverse, and both were found only because the tolerance was set at
the level the mathematics predicts rather than at a level the code could comfortably pass.
The Parseval check that exposed D27 is the clearest case: at a 1e-6 tolerance it passes and
the float32 defect ships.

### 7.2f The false-positive floor (T3.5.17)

`src/benchmarks/` holds twenty datasets whose correct answer is known before analysis.
Twelve of them are **null benchmarks** - their answer is "there is nothing here". Current
status: **29 PASS, 0 FAIL, 0 NOT_YET_RUNNABLE.**

**Nothing is pending any more.** Three gates were defined before the stages that could
answer them existed, and all three have now graduated to enforced PASSes: `4C.surrogate_null`
in T4C.5 - which immediately caught a real defect in the surrogate machinery (Section 7.2h) -
`4D.tracking` in TG2.3, on `advected_vortex_sequence` and on the
`advected_vortex_periodic_sequence` variant added with it (Section 3.6p), and `4E.invariance`
in TG3.4 (Section 3.6u), which was the last of them. `test_benchmarks.py` now asserts the
pending count is **zero** rather than at least one, so a newly pending gate has to be argued
for there rather than appearing quietly. While a gate is pending it reports
`NOT_YET_RUNNABLE` naming the missing stage
rather than being skipped, because a skipped test is invisible in a summary line and would
let "all green" mean "we never looked".

Two results worth recording:

*   **R12, measured.** On 200-frame AR(1) sequences with phi = 0.85 (effective sample size
    32.2), correlating *independent* series rejects the true null at **40.0%** under naive
    significance testing and **2.3%** with the effective-sample-size correction, against a
    nominal 5%. Treating frames as independent samples inflates the false-positive rate
    eight-fold.
*   **R11, measured.** On a sequence containing only deterministic cycles and 0.05-amplitude
    noise, a time-of-day bin climatology leaves **15.5%** of the original variance, because
    a 40-day record cannot form a day-of-year climatology and the annual cycle passes
    straight through. Harmonic regression leaves **0.017%** - a factor of ~900. That 15.5%
    residual is exactly what a miner would report as weather. The suite also asserts the
    *trap*: the raw sequence's scale energies correlate at r = 0.999, p = 2e-219, which is a
    spectacular finding and is entirely the calendar.

### 7.2g Reproducibility (T3.5.12)

`src/benchmarks/seeding.py` derives independent streams via `SeedSequence.spawn` from a root
seed and a **`zlib.crc32`** label hash. Python's built-in `hash()` on a string is salted per
process, so a derivation using it reproduces within one session and silently changes between
sessions - the worst kind of reproducibility bug, and the reason a literal hash value is
asserted in the tests. `PerturbationEngine.add_noise` now accepts `seed=` or a threaded
`generator=`; the previous `torch.randn_like` drew from global state, which would have made
sweep results depend on executor interleaving once T3.5.19 lands. An unseeded run is
labelled `seeded: False` in its metadata rather than being indistinguishable from a
reproducible one.

### 7.4 Test inventory

Counted as **test functions** (pytest reports more cases, because several are
parametrised). Checked automatically by `src/tests/test_documentation.py`, which fails
if this table drifts from the source - the mechanism that stopped this document going
stale once already. `python tools/audit_docs.py` reports the same facts outside a test
run and exits non-zero on any inconsistency; it was made a committed tool in T3.5.8 after
being retyped from memory once per slice, which is how the roadmap's own status table was
able to sit three slices out of date.

| File | Test functions | Covers |
|---|---|---|
| `test_analysis_data.py` | 7 | diagnostics/data-layer endpoints and independent D17 boundary-ring oracle |
| `test_artifact_store.py` | 33 | content addressing, checksum verification, handle budget, T4A.3 acceptance |
| `test_api_infrastructure.py` | 16 | health, listing, pagination, CORS, data-source transparency, benchmark endpoints |
| `test_benchmarks.py` | 47 | Ground-Truth Benchmark Suite, seed discipline, eager/streamed climatology agreement, D30 determinism, TG16.0 paired-family completeness, and TG16.1 gate registration |
| `test_boundary_synthetic.py` | 8 | boundary treatments, windowing, synthetic generators and independent Euclidean-ring oracle |
| `test_cds_source.py` | 14 | T5.2c monthly CDS planning/CLI, grid-alignment/server-snap refusals, network consent, atomic resume, shard integrity, conservative storage refusal, bounded Zarr publication, plus PASS/FAIL independent-route receipt publication, replay and tamper refusal |
| `test_geometry_registry.py` | 20 | TG1.2 geometry registry: the three builtins' metrics, crops, resamples and provenance unchanged; capability-driven `is_physical`/`length_units`/`latitudes`; a fourth geometry (`polar_scan`) registered from the test module with a non-uniform, non-spherical metric; the Cartesian Laplacian refusing it; `latitude`/`longitude` recognised as a sphere |
| `test_tracking.py` | 47 | TG2.3 frame-to-frame association: `4D.tracking` moving from NOT_YET_RUNNABLE to PASS with the recorded velocity and doubling time recovered from the field alone; the coincidence gate derived from alpha and the frame's own density and tightening when the frame crowds; a declared bound as a rate against an irregular clock; greedy and Hungarian disagreeing measurably, plus a third associator registered from the test module and two rogue ones refused; the seam crossing that is one track on a torus and two on a plane; the orientation gate reading the convention rather than the number and refused outright on an extractor that reports none; and the empty-frame and short-clock regressions |
| `test_representation.py` | 59 | TG2.4 representation-induced feature audit: the floor on every plane of every registered lens, and the planted blob that proves the audit can see; the null propagated through the representation against the same null rebuilt inside it, measured on the dual tree where they differ and on the stationary transform where they do not; the FFT magnitude plane whose null nothing can exceed; the family of forty-five planes that rejects on 86% of structureless fields uncorrected, the ensemble refused as too small for it, and the correction registry that prices six identical columns as one test; the declared decimation an array does not have; and the plane R13 leaves no interior in |
| `test_family_accounting.py` | 30 | TG3.1 family accounting: the T4C.6 declaration priced at 36 members and 3,005 surrogates from the frozen campaign JSON, with its label set compared against a real sweep rather than against its own count; every combinator's cheap count checked against its own enumeration; unordered triples registered from the test module; the unaffordable family refused with both R18 remedies computed, the narrowing checked by taking it and checked to be tight, and the case where no single axis can reach the ceiling; the zero-member term that scored as a remedy; and the admissibility audit that never moves the correction unit |
| `test_preregistration.py` | 40 | TG3.2 the generate/confirm split: the T4C.6 family of 36 refused at an ensemble of 199 while a four-member frozen subset of it is affordable, corrected at four rather than at 36, on a partition opened once; a partition identified from a series whose values raise on access; an edited field named and an editor who rewrites the digest table too caught by the outer digest; the wholesale rewrite that verifies against itself and is caught only against the published digest; the second, entirely honest seal against the same held-out data refused; the lineage refusals for mining on held-out and freezing against train; a refusal never spending the partition while a confirmation always does; and the channel-count/label contradiction found by running it |
| `test_publication.py` | 4 | D71 immutable publication on the repository volume: complete flushed bytes, existing-target identity, eight-process exactly-one-winner race, primitive reporting, unsupported-filesystem refusal and temporary cleanup |
| `test_motif.py` | 45 | TG3.5 recurring motifs: the pair of gates - `4E.motif_recovery` confirming one planted motif on a partition it was not mined from, and `4E.motif_null` confirming nothing where nothing was planted, on the same family, the same tolerance and the same ensemble; the family priced at every subset of every scene and checked against the enumeration member for member; a size with no registered combinator refused rather than computed; matching under a tolerance shown to be non-transitive and the intransitive pairs counted; support counting scenes and not overlapping occurrences; the ranking key that has to prefer the shape which matched fewer things; one shape frozen once however many scenes it appeared in; a surrogate that carries everything but position and respects the separation the data demonstrates, with the unconstrained null measured to halve the p-value; the constant-signature matcher registered into TG3.4's `MATCHERS` and audited as honestly invariant to all three transforms - and priced at exactly 1 by the null; a motif tested under a frozen label whose signature came from elsewhere refused as a redefinition; and the held-out partition spent once |
| `test_motif_freezing.py` | 14 | TG5.1 durable motif definitions: exact structural/origin separation and hashes; canonical stable serialization; process-independent graph round-trip; nested in-memory immutability; exclusive no-overwrite publication; structural and origin tamper detection; whole-artifact rewrite checked against a published digest; held-out-source, label/signature redefinition and origin-domain laundering refusals; timezone-bearing freeze order; strict schema and canonical-byte acceptance |
| `test_motif_transfer.py` | 16 | TG5.2 blind transfer (18 pytest cases): published digest durably bound before the only target opener; strict freeze/bind/open chronology; frozen-only size/matcher/relation/tolerance and cross-domain matcher capability; complete target enumeration and post-open subset refusal; source-domain, wrong-digest and carried-domain refusals before or after the access boundary as appropriate; failure still spending the target and second-open refusal across processes; semantic visibility, null-match receipt, canonical ledger reload, nested identity/hash/schema tamper detection and receipt identity |
| `test_motif_relationship.py` | 16 | TG5.3 relationship transfer: content-addressed outcome x lag family and ordered target train/test identities frozen before access; R21 floor, G3 affordability and no-redefinition refusals; both openings committed before either callback and every partition globally spent once even after failure; exhaustive frozen motif presence, circular-shift p-values, complete-family correction on each split and same-label replication; planted PASS, publishable-shaped null FAIL, structural absence retained at p=1, canonical plan/ledger reload, tamper and receipt identity |
| `test_evidence_bundle.py` | 24 | TG6.1 the hashed, append-only evidence bundle: revision zero complete before any evidence and every one of the ten first-class fields routed and queryable; appending returning a new snapshot with the receiver unchanged byte for byte; each entry binding the previous digest into a gap-free, non-decreasing chain, with entries and their nested payloads read-only; `contradictory_evidence` and `failure_states` as ordinary fields carried on the same chain and undeletable by later appends; commentary, prose-only and empty payloads, non-finite and unserialisable values, bad statuses, malformed or duplicated source digests and offset-less or backwards timestamps all refused; a hypothesis refused after the bundle it anchors and refused when swapped under an existing chain; edited, dropped, reordered, substituted and never-linked entries and a mismatched bundle digest all detected, with substitution caught by the bound previous digest rather than by sequence numbering; canonical exclusive publication, no-overwrite, published-digest verification, and reload refusing dropped, softened, relocated, unknown-field and reskinned files; process-independent reload continuing the same chain; the digest sensitive to append order, not only content |
| `test_claim_ladder.py` | 30 | TG6.2 the claim ladder as a pure function of the bundle: the floor rung claiming nothing; each rung reached only once its own evidence and every rung below it is present, climbing one rung at a time without skipping; every gate individually necessary, with a point estimate lacking quantified uncertainty refused as an association and temporal precedence required to be recorded exactly `true` rather than merely truthy; only `PASS` advancing a gate. A single `FAIL` or `INVALID` entry, or a standing `contradictory_evidence` or `failure_states` `PASS`, capping a fully evidenced bundle at `observation`, unoutvotable by piled favourable evidence and unappealable by a later retraction; the invariant held over an exhaustive field x status sweep and a randomised sweep of whole bundles in which both the blocked and clear branches and all five rungs occur; agreement with an independent restatement of the rule. Determinism to the digest, independence from append order and from labels, summaries and unread payload keys, and the same rung after a round trip through disk; causal claim kinds refused rather than denied, naming R7; the verdict binding the exact bundle revision and reporting why the climb stopped |
| `test_precedence.py` | 62 | TG4.1 recovering a relationship nobody pointed at: the pair of gates - `4F.precedence_recovery` confirming one relationship, at the right band pair and the right lag, on a partition it was not mined from, and `4F.precedence_null` confirming nothing over the same builder with its two modulations drawn independently; the family of ordered band pairs crossed with lags priced at 15,985 surrogates and checked against the sweep member for member, with the declared pair axis shown to price it exactly as TG3.1's `ordered_pairs` combinator does; lag zero refused as not a lead and a near-unit-root record refused every lag rather than given a weak result; the basis control that measures what the decomposition relates when nothing else does, the coupled pairs it excludes, and a control with dynamics of its own refused as no control at all; the circular shift shown to preserve every value and the autocorrelation where the shuffle destroys it, with the shuffle's error measured to grow with exactly the memory it discards; the sweep's winner tested against a single-lag null and against the null of its own maximum; one relationship entering the ranking once per lag and leaving it once; the held-out partition cut too close refused a seal, spent once, and a member tested at a lag other than the frozen one refused as a redefinition |
| `test_refusal.py` | 72 | TG4.2 the five refusals: the register that names all five, the gate each is carried by and the benchmark that gate runs on, checked against the live benchmark registry so that a deleted benchmark or an undeclared gate is a failing test rather than a stale sentence; the lag profile that starts at the one lag no precedence family may contain, and the leakage ceiling read off it - a real lead surviving it, every lag of a single process read four times refused by it, and the unguarded path shown to rank and carry forward exactly the members it refuses; a family emptied by refusal distinguished from a null result; the calendar derived from a record's own cadence rather than from its values, refused when it is faster than the sampling or the record is too short to see it turn twice, fitted on the training partition and subtracted from both on one shared clock, with a mislabelled partition shown to be removed worse; a record that still contains its calendar refused a candidate at the single door between a sweep and a seal, and a partly removed calendar refused too; both sample-size counts shown to travel and their gap to grow with memory |
| `test_cross_domain.py` | 47 | TG4.3 planted cross-domain precedence (50 pytest cases): two native clocks at one and three hours aligned only at exact shared timestamps, with interpolation, offset clocks, irregular clocks and an irregular intersection refused; original units, semantics, licences, dataset identities and lag bases carried into the sealed receipt; the joint physical floor computed before frame conversion and raised by aggregation windows; eight cross-boundary directions crossed with four physical lags and no within-domain pair; the Kelvin-to-megawatt relationship recovered at six hours without being named, its reverse and distractors present in the family, the same-builder null selecting candidates on train and confirming none, unit tampering breaking the partition identity, a within-domain member refused as laundering, and the held-out partition spent once |
| `test_invariance.py` | 56 | TG3.4 invariant matching: the `4E.invariance` gate moving off `NOT_YET_RUNNABLE` to PASS, with the position-memorising control audited beside it and surviving nothing; the shape identical under exact rotation, translation and rescaling and TG3.3's `distance` exactly invariant too when the scale is exact, which is what places the benchmark's 4.8% drift in the extractor's scale estimate rather than in the relation; a scalene configuration refused a match so scale-invariance is not permission to match anything; a pair refused by the shape matcher because one edge over its own mean is 1 for every configuration in the world; deviations minimised over correspondences, the defect that had put the position matcher's noise floor at 0.27; a tolerance refused without a stated basis and `match` refusing a tolerance that is merely a number; a vacuous test conferring no invariance; overclaiming and understating both caught on matchers registered from the test module; the scale ratio recovered from the separations, refused across a unit boundary, structurally unable to precede the decision, and judged against its own noise floor rather than the shape's |
| `test_constellation.py` | 65 | TG3.3 constellations as attributed graphs: the planted triangle built twice, in cells as a dimensionless amplitude and in metres as a temperature, matching as the same attributed graph, with a relation registered from the test module *without* the dimensionless division making the same two graphs disagree; all eight relations registered with their requirements declared; `direction` and `convergence` refusing against TG2.2's own `reports_orientation: False` capability; `convergence` refused on an undirected axis; a bearing refused across a periodic seam and from a point to itself; the geometric-mean reference that does not follow the larger scale; the relation axis a TG3.1 family may be priced over, 3 against 28; matching exhaustive to 8 nodes and refused above it; and the non-strict graph that did not match itself, found by running it |
| `test_feature_extraction.py` | 39 | TG2.2 extraction as a registry: the three planted features recovered across a six-fold range of scales and under rotation, translation and rescaling; both null benchmarks silent across three seeds with the loosened-alpha control that makes the silence mean something; the strict-comparison off-by-one; an unresolvable alpha refused before the ensemble; a second extractor registered from the test module; the periodic-axis seam and the self-scaling R13 refusal; and the one-feature-per-frame handoff to TG2.3 |
| `test_feature_record.py` | 37 | TG2.1 canonical feature record: features measured off the advected-vortex benchmark recovering its known velocity and scale doubling, the R19 refusals (magnitude, separation, elapsed time, mixed sets), the periodic-axis refusal, orientation conventions and the surrogate resolution floor, a fourth convention and a fourth significance basis registered from the test module, and defect D59 |
| `test_level_axis.py` | 19 | TG1.5 vertical coordinates: the registry and its sense of up, a height bank labelling its offsets the opposite way to pressure, a fourth coordinate registered from the test module, the declaration travelling from reader to signature, `level_hpa` refusing a non-pressure axis, and the pressure arithmetic unchanged |
| `test_lag_policy_registry.py` | 21 | TG1.3 lag-admissibility policies: the capability table, a fourth policy (`instrument_response`) registered from the test module with a per-channel floor, the advective arithmetic and fingerprint unchanged, the declared floor now reaching the sweep, D58, the replication gate refusing a floor from the wrong policy |
| `test_task_randomness.py` | 15 | D55 per-task streams: torch value continuity with the old global seeding, stream advance, thread isolation of the context binding, release on failure, `require` refusal, `add_noise` ambient fallback and labelling, D57 seed pass-through, `enable_determinism` scope reporting |
| `test_axis_roles.py` | 38 | TG1.1 declared axis roles: legacy arrangements unchanged, the three resolution bases in provenance, declaration beating a contradicting name, refusal of inference for a domain-general adapter, hint registry and collisions, D56 coordinate-slicing fix |
| `test_channel_series.py` | 14 | TG0.1 channel-series contract: signature/plain-series result equivalence, the two protocol tiers, R21's no-support refusal, clock and shape validation |
| `test_domain_gate.py` | 18 | TG0.3 negative control: PASS/FAIL/INVALID on a non-atmospheric domain, false-positive calibration, R6 embargoed split, gate-measure registry |
| `test_sample_spine.py` | 28 | TG1.4 sibling sample spine: a rank-3 domain through the unmodified sweep and replication gate, the planted/AR(1) pair, `PhysicalField` still refusing non-2D input, the one-way bridge and its transpose refusal, declaration-not-inference refusals, and a fourth reduction registered from the test module |
| `test_tabular_domain.py` | 48 | TG0.2 non-atmospheric domain: planted-coupling recovery and AR(1) null through the unmodified sweep, R21/R17 refusals, declaration and adapter validation |
| `test_coefficient_field.py` | 40 | T4B.1 acceptance: parent-grid alignment, perfect reconstruction per family, lineage-safe summary; DTCWT upsampling declared; LevelBank and level slicing (T4B.4) |
| `test_documentation.py` | 19 | architecture, roadmap and proprietary named-licence boundary against the code/repository |
| `test_dtcwt.py` | 28 | Kingsbury q-shift DTCWT: primitives vs reference, two oracles, orientation, shift invariance, D1 head-to-heads |
| `test_executor.py` | 38 | Executor backends, seed derivation, ordering, portable CPU/accelerator/HPC profiles, doctor, device/thread policy, SQLite concurrency, byte-identical sweeps (now over a payload that actually draws), D55 thread/serial agreement with the seed-to-draw window held open |
| `test_experiments.py` | 3 | declarative sweeps and lineage |
| `test_exports.py` | 32 | CSV/JSON/NetCDF4/Zarr round trips, embedded provenance, seeded perturbation (D34) |
| `test_external_fcn3.py` | 9 | T5.6a offline FCN3 request/result schemas, exact global input and ensemble contracts, portability refusals, canonical persistence, file/tree identity and request/artifact tamper isolation |
| `test_external_ensemble_evaluation.py` | 8 | T5.6c exact truth/initialization alignment, member/mean/persistence errors, analytic CRPS and spread, area-weighted fractional-tie ranks, bounded lazy reads, content identity and scientific refusal contracts |
| `test_external_forecast_cube.py` | 8 | T5.6b authenticated lazy NetCDF/Zarr import, exact dimensions/axes/grid/variables/SI units, bounded complete finite-value scan, explicit grid-aligned NZ crop lineage and pre-open tamper refusal |
| `test_evaluation_job.py` | 7 | T5.6f canonical portable jobs, separately hashed relocatable bindings, authenticated no-write preflight, input/overwrite refusals, exact receipt binding and complete create/bind/preflight/run CLI workflow |
| `test_evaluation_report.py` | 5 | T5.6g report flattening, real-source admission versus synthetic refusal, path-free provenance, content-addressed idempotence, read-time tamper hiding and complete import/list/get API boundary |
| `test_evaluation_run.py` | 6 | T5.6e real synthetic Zarr end-to-end orchestration, versioned/hashable controls, atomic no-overwrite receipt, nested/cross-lineage integrity and handle cleanup on failure |
| `test_matched_truth.py` | 9 | T5.6d lazy exact ERA5 initialization/valid-time selection, source/selection identity, split and FCN3-period guards, evaluator compatibility and time/grid/level/unit/alias refusals |
| `test_forecasting_adapter.py` | 5 | T5.3a exact persistence, represented autoregressive rollout, backward gradients, deterministic evidence, refusal contracts and CPU/RTX vendor-neutral accelerator parity |
| `test_forecasting_artifact_evaluation.py` | 8 | T5.3b/T5.2d checkpoint/config integrity, artifact-bound lineage, persistence-relative metrics, physical-time reporting/refusals, undefined-skill handling and CPU/RTX vendor-neutral accelerator parity |
| `test_forecasting_protocol.py` | 7 | T5.0a exact schema completeness, canonical identity, immutable nested configuration, evidence requirements, temporal/rollout consistency, persistence and tamper/drift refusal |
| `test_forecasting_protocol_binding.py` | 4 | T5.0b exact dataset/protocol/checkpoint binding, recomputed coordinate/statistics identities, drift refusals and bound-evaluation cross-run isolation |
| `test_frontend_contract.py` | 94 | the frontend/backend contract, including dataset-bound navigation gating with visible backend refusal reasons, capability profiles showing yes/no/not-established facts, transform/dataset/cadence readiness claim boundaries, domain-driven acquisition, workflow-grouped navigation, persistent record/study context, the TG11.1 analysis panel's three engine operations, R21 disablement, three-valued verdict and re-read identity check, the TG11.2 preregistration panel's declare-never-decide split, TG11.5's separate GET-only recorded-review workspace with its visible R23 fence, complete argument/cost display and honest empty states, preservation of every ERA5 control, and TG11.6's skip/route focus, bound labels, global focus and reduced-motion rule, keyboard SVG lineage, figure text equivalents and asynchronous-state semantics, plus the UI integrity guards: no fabricated results, no unqualified validation claims, units and slope uncertainty displayed |
| `test_gate_run.py` | 1 | T4C.5d frozen plan, local-only preflight, bounded train-only climatology/signatures, authenticated synthetic gate receipt, no-overwrite and tamper refusal |
| `test_gate_campaign.py` | 6 | T4C.5f-h exact campaign identity, strict nested schema, canary/full/WeatherBench drift refusals, pre-transfer R13/physical-lag audit, aggregate storage/readiness, immutable freeze/load, pinned real preregistration and zero-network CLI (8 pytest cases) |
| `test_grid_operators.py` | 64 | grid metrics, metric-aware gradient/Laplacian, area weighting, physical-wavenumber spectra, D26 |
| `test_hypothesis.py` | 3 | correlation and categorical hypothesis discovery |
| `test_imports.py` | 36 | NetCDF/Zarr/CSV/JSON import, dimension pinning, axis identification, laundering guard, benchmark runs over HTTP |
| `test_migrations.py` | 25 | Alembic history, ORM/schema drift, per-revision round trips, pre-Alembic adoption, auto-migrate refusal, PostgreSQL rendering |
| `test_regional_forecast.py` | 13 | T5.2a-d alignment, ratio/calendar embargo, cadence validation, physical lead durations, train-only normalisation, provenance, eager/lazy equivalence, bounded streaming, multi-worker loading, local/HPC cache seam, independent-route comparison contract, and TG12.1a's D70 acceptance that readiness declares itself inapplicable to a non-pressure-level crop without coercing its axis, while every ERA5 field is unchanged |
| `test_sequence.py` | 37 | FieldSequence validation, cadence, R6 temporal split and leakage guardrails, slice_sequence |
| `test_registries.py` | 30 | registries, error taxonomy, fallback chain, plugin acceptance and DTCWT native-visualisation contract |
| `test_stationary.py` | 19 | undecimated SWT: shift invariance, perfect reconstruction, frame constant, PyWavelets oracle, R3 normalisation |
| `test_statistics.py` | 36 | FDR procedures vs scipy, surrogate preservation properties, calibration on a true null, stationarity gate, screening |
| `test_training_representations.py` | 43 | T5.1a-e raw/FFT/DCT/Haar/db2/SWT/DTCWT batch contract, reconstruction, immutable context, analytical/PyWavelets/FFT oracles, exact complex-atlas bijection, fused/reference coefficient and gradient agreement, translation equivariance, explicit TF32 precision isolation, Mallat/channel packing, support/redundancy metadata, gradcheck, cached buffers, dtype migration and vendor-neutral accelerator parity |
| `test_cross_scale.py` | 26 | T4C.3 acceptance plus the frozen T4C.6 protocol: injected cascade/null twin, Theiler windows, exact metric-aware lat/lon support floor, power check, split sufficiency, embargo and three-state replication verdict |
| `test_scale_signature.py` | 29 | T4C.1 acceptance, analytic white-noise values, eager/streamed exact agreement and source-mutation refusal; threshold sensitivity; R13 interiors; T4C.4 power-law core |
| `test_surrogate_null.py` | 14 | T4C.2 acceptance: spectrum preserved, phase destroyed, organised scores and fBm does not; the two calibrations (wrong null, linear lag) |
| `test_wavelet_bank.py` | 27 | T4B.2 expansion through the engine's own parameter matrix, the 1,000-combination guard, decompose_bank / extract_scale_signature, the vertical-bank refusals |
| `test_transforms.py` | 13 | fft/dct/dwt/dtcwt/hybrid round trips; D1 recorded as a strict xfail |
| `test_zarr_source.py` | 61 | R13 geometry, chunk-hostility, byte counting, streaming content identity, exact chunk-bounded frame reader, cache/provenance round trip, NetCDF engine and HTTP surface; D67 coordinate-only costing that succeeds even when the data-selection path raises `MemoryError` |
| `test_five_outputs.py` | 38 | TG6.3 the five outputs as a pure function of the bundle: what can be claimed given as the reached rung and every rung beneath it with a fixed entitlement stating what that rung does not license; what cannot be claimed as the exact complement, each unreachable rung naming the gates that stand between, including the case where a rung's own gates all pass but a floor failure or a lower unmet gate blocks the climb; the evidence against carrying every `FAIL` and `INVALID` entry, every contradiction and failure state that is not `NOT_APPLICABLE`, and every passing null result, with `caps_at_observation` agreeing exactly with the ladder's blocking set so what merely argues against a claim is distinguished from what forbids it; eight structural alternatives mapped one-to-one and totally onto the climbing gates, each open exactly while its gate is unsatisfied, alongside alternatives someone recorded; the next observation following its stated precedence of unblock, then climb, then resolve, then nominate nothing and say so, verified over a randomised sweep in which every branch including the empty one occurs and all five rungs are reached; determinism to the digest and across a round trip through disk; labels, summaries and unread payload keys carried to the reader but moving no membership; causal claim kinds refused at every rung, naming R7 |
| `test_recorded_call.py` | 50 | TG7.1 the recorded-call boundary: every call capturing the verbatim request, the verbatim response bytes, the exact model id, effort, API request id and both timestamps, chained by digest and labelled `recorded-not-reproducible`, with a loaded record claiming determinism refused by name; sampling parameters refused at any depth of the request; the declared schema sent as `output_config.format` with `additionalProperties` closed, and free text, a missing field, an undeclared `claim_level`, a value outside its enumeration, a wrong type and a parse disagreeing with the response bytes each refused, the check surviving a round trip rather than holding only at record time; commentary bound to one exact bundle revision, published beside the bundle and never over it, never overwritten, and refused when spliced from another bundle even where the chain would accept it; and the acceptance test of the phase — a corpus standing on all five rungs plus a blocked and a contradicted bundle, reviewed by all eight roles with commentary demanding promotion, whose every claim level is identical after deleting every LLM output — with a randomised sweep over 300 reviewed bundles, a second over 200 recorded chains, and the smuggling check that refuses recorded wording found inside the evidence chain (R22, R23) |
| `test_round_robin.py` | 49 | TG7.2 the adversarial round-robin: the eight seats replayed turn by turn against the plan, with a role out of order, a seat answered by a model or at an effort the panel did not seat, an answer against the wrong schema, a second answer to one challenge and a ninth turn on a finished exchange each refused; a malformed turn recorded before it is refused, so nothing paid for is discarded (R23); dissent retired only by concession or by a rebuttal the independent reassessment declines to reopen, with the reassessment able to reopen a dissent but not originate one; the final synthesis refused when it drops an unresolved dissent, invents one, or reports calm while one stands; three agreeing challengers leaving the fourth's objection byte-identical, which is what a hidden count would have broken; a panel needing every seat filled and reporting reviewer overlap rather than refusing it; and a complete exchange over a corpus standing on all five rungs plus a blocked and a contradicted bundle, every seat arguing for promotion by name, moving no claim level — with a randomised sweep over 120 exchanges checking retained dissent against an independently written rule and a second over 80 randomly seated panels re-replaying each recorded chain (R22, R23) |
| `test_review_cost.py` | 15 | TG7.3 provider-neutral cost control and Gemini 3.5 Flash Batch transport: fixed per-role effort routing; exact structured Batch request, poll and response mapping including the first live operation shape; current Batch response-format enum; API-key non-retention; visible-plus-thinking output accounting and raw/normalized token reconciliation; measured non-zero cache-hit acceptance and configured-but-missed refusal; standard-route, effort, identity, arithmetic, provider-error and tamper refusals; and content-addressed atomic no-overwrite receipt persistence over an eight-call review |
| `test_translation.py` | 47 | TG7.4 translation, bounded: the restated gate names checked against the ladder's own so the one line of duplication cannot drift; a glossary refused when partial, when it invents a term, and when a phrase carries causal vocabulary, a digit, a comparative asserting a relation of size, or wording reserved to a higher rung; R9's six figures given a structure they did not have, with each of the six load-bearing and a lift that is not confidence over base rate refused; the roadmap's own "82% of the time" rendered welded to the base rate that defuses it; two features of one variable described with their units while a cross-domain pair renders only `structural_signature`, asserted as the absence of variable, dataset and units; entitlements welded into the same string as the claims they bound; commentary quarantined outside the claim text and refused when reproduced inside it; a stale translation of a superseded revision refused; canonical no-overwrite persistence and a tampered document refused on load; a glossary registered from the test module outside `src/` (TG8.1); and the acceptance test of the phase — a corpus on all five rungs plus a blocked and a contradicted bundle, translated into an atmospheric and a financial vocabulary, reading completely differently and asserting identical facts — with three randomised sweeps and nine deliberate mutations of the module, each caught (R7, R9, R19, R22) |
| `test_findings_api.py` | 29 | TG9.1 the read-only claim surface: the wire guard refusing a bare confidence at any depth of any response body and passing one that travels with all six of R9's figures, with the guard's restated field list asserted to still agree with `AssociationFigures`; the acceptance test of the slice — every route served over a corpus that genuinely does report a confidence, with the test refusing to pass vacuously if none is present; a domain registered from the test module reaching `GET /domains` and `GET /glossaries/{name}` without editing `src/api/`; built-in glossaries registered eagerly at import rather than on first request (D35); a GET leaving the bundle bytes and the rung unchanged (R22); an unreadable bundle reported rather than skipped; an absent study root served as an empty list; unknown study and unknown glossary both 404; a blocked study reported as blocked; a partial figure set served as no figures rather than a subset; and one study in two vocabularies reading differently while serving identical `structural_keys` (R9, R22) |
| `test_domain_onboarding.py` | 33 | TG8.1 the onboarding contract: the adapter recipe as a tuple the API serves, the checklist generates from and the tests assert against, so the documented contract and the enforced one cannot drift; the acceptance criterion of the slice — a third domain onboarded in one file outside `src/`, with the six files it would otherwise have had to touch hashed before and after and asserted byte-identical, live in all three registries and served by `GET /findings/domains`; the geometry/violation biconditional in all four combinations, refusing a domain that names a metric geometry while renouncing the metric and one that supplies neither, with `pixel` shown to sit on the renouncing side and an unregistered geometry refused by the registry that owns the vocabulary; atomicity, with a refused glossary, a refused geometry and an injected failure between the writes each leaving all three registries exactly as they were; re-onboarding refused by name until asked for explicitly, and a name the glossary would normalise differently refused as the half-onboarded state in a subtler form; a piecemeal domain audited as incomplete rather than passing for a checked one, and a declaration registered without wording still appearing in the listing — the TG9.1 omission with its halves swapped; the digest stable across equal declarations and moved by a single changed phrase; the built-ins held to the contract they document, either entry point registering the whole pair, and a half-registered built-in repaired rather than skipped; and the plugin domain asserted to break the two assumptions no registered domain had broken, with five deliberate mutations each caught (E13, E15, R17, R21) |
| `test_channels_api.py` | 24 | TG8.4 the ingestion seam over HTTP: inspection reporting a record's columns, rows and clock while choosing neither a clock column nor a domain, and refusing to substitute a price column for a timestamp that runs backwards even though the price increases; an irregular clock turned into an obligation on whichever domain is chosen rather than filled in, and the aggregate obligation stated because no column declares itself one; the property the two-call shape rests on — the refusal inspection advertises and the refusal reading enforces are the same refusal — checked against a third domain onboarded for the purpose, because both built-ins agree about ragged clocks and could not show the difference; a loaded record carrying its domain's refusals, the attribution caveat and the sentence saying a plot is not an analysis; an aggregate channel marked and its declaration required; a gridded domain refused for a channel table in its own words including E14; the adapter's reasoning surviving to the researcher rather than becoming "invalid file"; a binary upload redirected to the route that reads binary; and a channel honestly named `confidence` served rather than mistaken for a bare claim, which is why channels are a list of named entries and not a mapping (E14, E15, R9, R17, R22) |
| `test_presence.py` | 13 | TG12.2a/D69 explicit per-sample presence: strict boolean shape and binding absent values, observed-invalid distinction, minimum viable support and receipt counts, biconditional domain enforcement, tabular refusal without a mask, partition slicing and gap embargo, physical-lag decorrelation and sparse refusal, fixed-N shift-null audit, the controlled Argo-like candidate measurement, masked frame-lag refusal, all-true result identity, presence-bound partition identity without measure reads, and API/UI wire semantics (E5-E14, R6, D65) |
| `test_stores.py` | 37 | TG10.1/TG12.1 gridded-store registry: the four ERA5 stores and externally registered GLORYS source under declared domains; malformed declarations refused; measured versus unmeasured chunk facts; read-only catalogue compatibility; pinned ERA5 crop identity; declared depth and fractional negative-elevation selection; GLORYS citing the persisted 2.02x probe while the rejected 72.52x layout remains in the ledger; no runtime import of `copernicusmarine`; and the independent fifth-store extension acceptance (E1, E2, E14, R17) |
| `test_store_probe.py` | 34 | TG10.3 probing as a recorded act: a local store's structure and chunk sizes read without transferring data, the worst chunk reported rather than the mean against a fixture whose two variables differ in width because an identical pair could not tell the two apart; the acceptance criterion, a deliberately hostile store characterised as hostile with nothing materialised, no cache entry created and the figure agreeing exactly with the prediction from chunk metadata alone; hostility shown to be a property of a pairing, the same store amplifying 30x for a request that straddles its chunks and under 4x for one that lines up; three-valued hostility with `None` never folded into `False`; a refusal recorded as a result — network switched off, an unopenable path, a directory that is not Zarr, and five open failures classified with their text kept verbatim, because the difference between "no such bucket" and "403" is a typo versus an account; seven incoherent records refused, including an amplification with no crop attached and a refusal with no reason; the digest covering the observation and not the day it was taken; a ledger that keeps an earlier probe rather than replacing it; atomic content-addressed persistence that never rewrites an existing record; the registration gate in five parts — a claimed measurement with no probe, a probe of another store's URI, an unrecorded digest, a figure the cited probe denies, and a store that both cites a look and says nobody looked — with the error asserted to name the claim, the remedy and the honest alternative after a mutation showed a weaker assertion passing; the four transcriptions counted as debt and asserted to carry exactly what was recorded and no more; the probe routes including a recorded result where `/inspect` returns 409; and an opt-in live probe of the real WeatherBench store, **NOT RUN** (E1, E5, D43, D62) |
| `test_crop_planner.py` | 8 | TG12.1d/D73 transform-owned support and derived absolute/recommended crop thresholds; an external support callback without planner edits and pre-source refusal without one; symmetric coordinate expansion, edge/source infeasibility and revised chunk cost; plan identity moving with transform or coordinate observations but not field values; metadata-before-selection materialisation refusal; and invalid filter configuration refused as a client parameter (R13) |
| `test_acquisitions_api.py` | 6 | TG10.2-TG13.2 domain-first catalogue, complete registered grid and light-curve reachability, channel-table E14 admission/refusal, limits and attribution caveats, and mechanical known-violation coverage backed by available acquisition paths |
| `test_analysis_api.py` | 7 | TG11.1 the domain-analysis engine through HTTP: the read-only capability boundary, association over the full re-uploaded record, precedence admitted only by a declared floor, the R21 refusal reaching the caller before any computation, a three-valued gate verdict over server-derived record facts, an unknown configuration key refused rather than ignored, and the acceptance criterion — all thirteen `sequence` and `cross_domain` benchmarks reproduced through the live HTTP client with every null still answering "there is nothing here" |
| `test_preregistration_api.py` | 17 | TG11.2 the generate/confirm split over HTTP: the sealed family matching the shape the sweep actually emits, a partition identity that ignores what the file was called (D65) and separates two splits of one record, sealing that narrows by lag and is timed by the server clock, a confirmatory lag that was never generated refused, an edited seal naming the field that changed, a wrong published digest refused, confirmation taking every setting from the seal and spending the partition, the same held-out data refused a second confirmation under a second individually honest seal, two seals frozen before any opening still buying only one look, a partition the seal did not name refused, a refused confirmation leaving the partition unspent, and TG11.1's gate refused on a spent partition |
| `test_evidence_api.py` | 22 | TG11.3 the evidence write path: a study opened at revision zero claiming nothing, a second study under one identifier refused, an identifier that could traverse a directory refused, an append linked to the head it names, a stale head refused with nothing written, earlier revisions kept rather than rewritten, the read surface serving the latest revision and folding the earlier ones into one row (D66), a request carrying a rung refused rather than ignored, a payload asserting a rung refused at any depth, a payload asserting `temporal_precedence` refused and told which route computes it, the rung moving only because the evidence moved it, one FAIL entry capping the chain at observation through the wire, commentary refused a category, a bare confidence refused on the way in, an entry that cannot be back-dated, a causally worded hypothesis registered with the ceiling stated, the precedence verdict computed here and citing the bytes and the configuration it came from, an underpowered sweep recorded INCONCLUSIVE rather than as a negative, a domain with no admissible lag floor writing nothing, and a stale head refused before the sweep runs |
| `test_mining_api.py` | 41 | TG11.4 the structure-mining surface: no request model on it accepts a feature, a coordinate or a graph; a tolerance cannot be typed and travels as the digest of a calibration the server performed; a tolerance measured through another pipeline or on held-out frames refused; frames of unequal feature counts refused rather than trimmed, with the tempting repair named; a record addressed by its bytes and its declaration, and refused when the stored declaration is edited; a pickled array refused rather than loaded; a domain with no spatial extent refused; the frame split asserted against the row split it mirrors; a generation response carrying held-out geometry and nothing measured inside it; every sealed setting present in the seal, stored where every other seal is; a confirmation that takes a seal digest and nothing else; the planted motif confirmed on frames it was not mined from; the held-out frames opened once; a seal frozen by another surface refused; **a null record confirming nothing**; a published definition carrying its origin licence; a transfer target opened once whatever is transferred into it; a transfer into the origin domain refused as replication; and the invariance audit reporting an unsupported declaration as overclaimed |
| `test_cross_domain_api.py` | 35 | TG11.4b the cross-domain record: two native clocks intersected exactly, with what each side retained and discarded reported; clocks that share no observation refused rather than resampled, and the refusal naming interpolation as the thing it declines; an irregular native clock refusing precedence by name; a column whose semantics or units were not declared refused rather than defaulted (R19); an unknown reading setting refused rather than ignored; two records from one domain refused as not a cross-domain study; a family declared in seconds converted onto the common cadence; only pairs that cross the boundary counted as members; a duration below either domain’s physical floor refused rather than dropped and one the common clock cannot express refused rather than rounded; the price agreeing with the family the generate pass actually searches; the partition identity ignoring what the files were called (D65); generation reading nothing from the held-out partition and writing nothing; every run setting sealed inside the specification and the seal visible where the programme lists what it froze; the frozen members re-derived from the record rather than reconstructed from their labels; an edited seal refused at load and spending nothing; **the planted relationship confirmed on data it was not selected from and the same pipeline over an uncoupled pair confirming nothing**; both operands’ semantics and units restored to the receipt; the partition opened once; a wrong pair of records confirming nothing and costing nothing; a published digest that disagrees with the seal spending nothing; a seal frozen by another surface refused; the confirm route accepting the two records and nothing else; and no route on the surface accepting a lag in frames |
| `test_reviews_api.py` | 8 | TG11.5's read-only recorded-review boundary: explicit absence without reassurance, complete verified record/outcome/cost serving, exact latest-bundle binding, record-digest linkage, malformed and unknown artifacts reported rather than skipped, unknown-study 404, GET-only routing, and a read leaving the evidence bundle byte-identical (R22, R23) |
| `test_profiles.py` | 10 | TG12.2b-d immutable profiles, declared reductions, and bounded Argo seam: profile spec machine-independence and scatter preservation, preflight counts, observed-invalid distinction from absence, per-float reduction enforcing violations, depth-bin aggregation identity shifts, profile collection round trips, argo parent flat-channel refusal, profile reduction registry discoverability, and profile API contract refusal visibility (E15, R17) |
| `test_photometry.py` | 9 | TG13.1 atomic TESS onboarding and precedence refusal, canonical bounded requests, metadata-only exact-product preflight, checksum-valid BJD_TDB parsing and value-bound identity, pre-download caps, immutable collection replay, source discovery, API claim boundaries, and an explicit opt-in bounded live MAST acceptance (E14, E15, R17, R21) |
| `test_dataset_ingress.py` | 9 | G14/G15 file probing without semantic inference, explicit sample roles/relationships/units, content-bound routing with explained spatial refusals, grouped/ordered split-leakage refusal, TG16.0's shared independent-only admission contract, R18 family sealing and permutation-resolution refusal, planted generate/confirm recovery, changed-file and tampered-plan refusal, and the complete multipart HTTP workflow |
| `test_representation_structure.py` | 6 | TG16.1 complete pair enumeration, joint redundancy/complementarity/XOR/null discrimination, sealed estimator/null/family and permutation-resolution refusal, content/tamper binding, non-removal claim boundary, earned capability registration, and multipart plan/run workflow |
  | **total** | **2409** | |
### 7.2h A surrogate null that was not the null it claimed (T4C.5)

The most instructive defect of the project so far, because it passed every structural check.

Phase-randomised surrogates were built by drawing independent uniform phases and
antisymmetrising them by averaging, `phi = (phi_k - phi_-k) / 2`. That *is* antisymmetric, so
the surrogate was real-valued and its power spectrum was preserved **exactly to 1e-16** - which
is the one property the method is defined by, and the only one being asserted.

But the difference of two independent uniform angles, halved, has a **triangular density peaked
at zero**. Every surrogate was therefore biased toward the phase-aligned configuration - a field
with its energy concentrated rather than spread. Measured consequences:

*   The excess kurtosis of level-2 wavelet detail was **90.0 +/- 7.5** for the surrogates
    against **0.75** for the field itself: z = -12.
*   The false-positive rate on a true null was **0.765** against a nominal 0.05.

**Nothing in the codebase would have caught this**, because the tests checked what the method
is *specified* to preserve. It was found by making the fBm benchmark's `4C.surrogate_null` gate
runnable: the null benchmark immediately reported a false discovery, which is precisely the job
the null benchmarks exist to do. Fixed by taking the phases from the FFT of a real white-noise
field, which is Hermitian *and* marginally uniform by construction rather than by imposition.

Two further consequences worth recording:

*   **The fBm generator had the same flaw.** It also took `.real` of a non-Hermitian inverse,
    which averages each mode with its conjugate mirror and so modulates the realised amplitude
    by the random phase difference between +k and -k. The field's true spectrum was therefore
    not the requested one, leaving residual phase structure detectable at z = 2.1. A null
    benchmark that does not quite contain what it claims is worse than no benchmark.
*   **Preserving a spectrum is necessary but not sufficient.** The lesson generalises: a
    surrogate must match the null's *distribution*, not only the summary statistic the method
    is named after.

### 7.3 Precision caveats (not defects, but do not overstate them)

*   `compute_ssim` is single-window global SSIM, not locally-windowed SSIM (Section 3.2).
*   ~~`get_execution_device` round-robins `run_idx % num_gpus`, but sweeps run strictly sequentially, so multi-GPU assignment is cosmetic.~~ **Addressed in T3.5.19/T3.5.21:** runs are distributed through the Executor seam and `device.select_device(run_idx=...)` spreads them across CUDA/ROCm devices. Single-device CUDA is verified on the RTX 5050; multi-GPU, ROCm and MPS execution remain untested.
*   The DWT is **decimated**: each level halves resolution, so scale *n* lives on a different grid from the parent field. This is correct for compression and reconstruction, but it makes cross-scale spatial reasoning awkward - the reason Phase 3.5 adds an undecimated SWT alongside it.
*   `PhysicalField` remains intentionally **strictly 2D** and raises on any other rank
    (`field.py:19`). Phase 4A added `FieldSequence` for `(time, y, x)` records rather than
    weakening that invariant; algorithms must declare whether they consume a frame or a
    sequence.
*   Historical pipeline actions embedded full payloads in lineage. Phase 4A's `ArtifactStore`
    and the later sequence/coefficient actions now offload large arrays behind content-addressed
    handles. Any new action that puts a full coefficient or sequence payload in a database
    lineage value is therefore a regression, not an accepted scaling limitation.
*   `GET /api/v1/experiments`, `GET /api/v1/health` and the benchmark endpoints were added in
    T3.5.10 and are consumed by the frontend as of T3.5.22-T3.5.24. Contract tests assert that
    every fetched path is served and that the response fields read by the UI exist; those tests
    verify the interface contract, not browser rendering.
*   ~~Alembic is absent - the schema is created by `Base.metadata.create_all` in the FastAPI lifespan, so there is no migration path for the seven tables Phase 4 adds.~~ **Fixed in T3.5.8** (Section 4.1). `create_all` was not merely missing a migration path: it silently failed to add columns to tables that already existed, which had already broken the real `spectral_earth.db` (defect D32).
