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

### 1.4 FourCastNet 3 external-judge boundary (`src/forecasting/external_fcn3.py`, `external_cube.py`, `matched_truth.py`, `ensemble_evaluation.py`, `evaluation_run.py`, `evaluation_job.py`, partial)

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

### 3.8 Grid Geometry and Metric-Aware Operators (`src/physical_core/grid.py`, `operators.py`)

Added in T3.5.13 (defect D13, standard E3). `GridSpec` is the physical metric attached to
every `PhysicalField` - `pixel`, `cartesian` or `latlon` - and the default is deliberately
`pixel` rather than `None`, so "lengths here are array indices" is a recorded fact that
travels with the data rather than an unexamined assumption.

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

### 3.11 Ground-Truth Benchmark Suite (`src/benchmarks/`)

Added in T3.5.17 (standard E7). Nine synthetic datasets whose correct answer is known
*before* analysis, of which **five are null benchmarks** whose answer is "there is nothing
here". This is distinct from `synthetic_generator/`, which exists to keep the UI alive
offline and declares no truth.

*   `core.py` - registry, `Benchmark` (build + known answer + gated stages + checks), and a
    three-valued `Outcome`. `NOT_YET_RUNNABLE` is never folded into `PASS`, so "all green"
    cannot come to mean "we never looked".
*   `seeding.py` - `SeedSequence.spawn` derivation from a root seed and a **`zlib.crc32`**
    label hash (Python's `hash()` on a string is salted per process and would break
    cross-session reproducibility).
*   `fields.py` / `sequences.py` - the nine datasets.
*   `runner.py`, `__main__.py` - report and CLI (`python -m src.benchmarks`, exit 1 on any
    failure, usable directly as a CI gate).

Current status: **15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE**. See Section 7.2f.

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

28 routes. Listed here because an undocumented endpoint is an untested contract.

| Method | Route | Notes |
|---|---|---|
| GET | `/api/v1/health` | DB reachability, backend scheme, dataset count, execution device (T3.5.10) |
| POST | `/api/v1/transforms/apply` | fft, dct, dwt, hybrid, **swt**, **dtcwt** (real Kingsbury q-shift, T3.5.6; `level1`/`qshift` sweepable) |
| POST | `/api/v1/synthetic/generate` | vortex, front, turbulence, wave |
| POST | `/api/v1/synthetic/perturb` | rotate, translate, noise (now seedable, T3.5.12) |
| POST | `/api/v1/boundary/analyze` | declares its pixel frame explicitly (D13) |
| GET | `/api/v1/actions` | every registered pipeline action, generated from the registry (T3.5.15) |
| GET | `/api/v1/transforms` | every registered transform with its params and capabilities (T3.5.15) |
| GET | `/api/v1/training/representations` | batched/autograd readiness, verification limits, SWT redundancy, DTCWT atlas geometry and selected R13 interiors (T5.1e) |
| GET | `/api/v1/data/sources` | the data-source fallback chain in priority order (E2) |
| POST | `/api/v1/import/inspect` | describe an uploaded file without committing to a 2D slice of it (T3.5.24) |
| POST | `/api/v1/import/field` | read one pinned 2D field out of an upload, with reconstructed provenance |
| POST | `/api/v1/export/field` | a 2D field as CSV, JSON, NetCDF4 or a zipped Zarr store, provenance embedded (T3.5.23) |
| POST | `/api/v1/export/table` | hypotheses, benchmarks or metrics as CSV or JSON, provenance embedded |
| GET | `/api/v1/data/zarr/catalogue` | known cloud ERA5 stores, the network gate, and the R13 crop floor (T3.5.18) |
| GET | `/api/v1/data/zarr/cached` | crops already materialised locally; works with no network |
| POST | `/api/v1/data/zarr/inspect` | chunk structure and chunk-hostility for a proposed crop - **metadata only** |
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

The React frontend is fully written and structurally complete. It was installed and built in T3.5.0/T3.5.3 (`npm run build` emits hashed JS and CSS into `dist/`) and wired to the previously unreachable endpoints in T3.5.22. Its **rendered appearance was confirmed by the user on 2026-08-20** (T3.5.25): the platform was started, both servers came up, and the nine tabs were reported working. That confirmation is a user report, not an artefact - **no screenshot per tab exists in this repository**, so T3.5.0's literal evidence clause is still outstanding. The contract tests prove the nine tabs compile, call routes that exist and read fields that are present; they still do not prove anything renders, and the distinction is kept explicit because a green suite plus a green build is exactly what makes people assume otherwise.

*   **Component Visualizations:** `Heatmap2D.tsx` and `LineChart.tsx` wrap `react-plotly.js`; `LineageGraph.tsx` is a hand-rolled SVG node-link renderer with a tooltip inspector and no external graph dependency. All three take reactive props and render spatial fields, PSD curves, coherence ratios, and provenance DAGs.
*   **Accessibility: zero, and measured rather than assumed.** `frontend/src` contains
    **0** `aria-*` or `role` attributes and **0** keyboard handlers, with no focus
    management anywhere. The nine tabs are operable with a mouse and by nobody else. This is
    recorded here, in the document that says what exists, so it cannot be read as an
    oversight or discovered later as a surprise; it is not currently scheduled.
*   **Main Application (`App.tsx`):** ~2,400 lines covering state hooks for **nine** tabs (Synthetic Generator, Meteorological Data, Boundary-Condition Lab, Spectral Transforms, Diagnostic & Analysis, Experiment Engine, Automated Hypotheses, **Platform & Evidence**, **Real ERA5 (Zarr)**), loading indicators, dynamic sliders, and follow-up proposal adoption.
*   **Platform & Evidence tab (T3.5.22):** the execution device, core and thread counts, executor backends with the measured rationale for the serial default, the SQLite pragmas actually in force, the stamped Alembic revision with an explicit warning when the schema is behind the code, the data-source fallback chain labelled observational/SIMULATED from each source's own declared flag, and the full Ground-Truth Benchmark Suite with its declared known answers and null benchmarks marked. All of this existed on the backend for several slices with no consumer.
*   **Real ERA5 tab (T3.5.22):** a crop form driven by the store catalogue, the R13 minimum crop size for the chosen number of wavelet levels, and an **inspect** action that reports chunk structure, the predicted amplification, the chunk-hostility warning and the per-level valid interior — metadata only, no transfer — then hands back the CLI command that would materialise it. The network gate is shown when it is off, with the variable that enables it.
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
| Backend test suite | **1089 passed, 1 xfailed** (plus 1 skipped: opt-in live GCS) (was 8 failed / 11 passed at first run; 65 after T3.5.0, 152 after T3.5.7, 222 after T3.5.13, 286 after T3.5.17, 351 after T3.5.6, 379 after T3.5.15, 407 after T3.5.19, 449 after T4C.5, 709 after T4A.4, 781 after T4B.4, 855 after T4C.5, 859 after T4C.5c, 882 after T5.1a CPU acceptance, 883 after RTX acceptance, 890 after portable profiles, 911 after T5.1b/D44, 917 after T5.1c, 933 after T5.1d/D45, 946 after T5.1e, 955 after T5.2a, 957 after T5.2b, 962 after T5.3a, 969 after T5.3b, 981 after T5.2c offline acceptance, 985 after T5.2d, 1000 after T5.0a, 1008 after T5.0b, 1027 after T5.6a offline acceptance, 1042 after T5.6b cube acceptance, 1056 after T5.6c matched evaluation, 1070 after T5.6d truth matching, 1080 after T5.6e orchestration, 1089 after T5.6f portable jobs) |
| Ground-Truth Benchmark Suite | **15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE** (`python -m src.benchmarks`, exit 0) |
| Frontend `npm install` + `npm run build` | passes, emits 1,385 modules + real JS/CSS assets (was: 1 module, no assets) |
| Backend server | starts, serves OpenAPI, all smoke-tested endpoints return 200 |
| End-to-end experiment sweep | 9-run parameter sweep completes 9/9, writes 28 lineage nodes / 54 edges, hypothesis engine returns results |
| Version control | `git init` run to enable E5 provenance capture; no commit made yet |

Earlier revisions of this document and of `roadmap.md` claimed the platform was "validated"
and "zero-error". It was not: the first real execution produced 8 test failures and a frontend
that had never rendered. The ledger below grew from 18 entries to **31** as a direct result of
running the code and of building the tests that check it — **29 of which are now fixed, and one partially**.

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
| D17 | `analysis_engine/diagnostics.py:29,56`, `analysis_engine/decomposition.py:81`, `boundary_lab/boundary.py:31,123` | **Per-bin Python loops over full arrays.** Five functions bin values by radius or distance using `for k in range(...)` with a fresh boolean mask over the *entire* array each iteration - `O(bins x H x W)` where `O(H x W)` suffices via `bincount`/`scatter_add`. On a 512x512 field (`max_r = 256`) `compute_radial_psd` performs ~256 full passes, roughly 67M element visits instead of 262k. These are the innermost functions of the Phase 4C loop, called inside a surrogate ensemble; unfixed, they alone decide whether the platform is usable on a laptop. | T3.5.20 |
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

`src/benchmarks/` holds nine datasets whose correct answer is known before analysis. Five of
them are **null benchmarks** - their answer is "there is nothing here". Current status:
**15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE.**

The two pending entries gate stages that do not exist yet (4D tracking, 4E constellation
matching). A third, `4C.surrogate_null`, **graduated to an enforced PASS** in T4C.5 - and in
doing so immediately caught a real defect in the surrogate machinery (Section 7.2h). They report `NOT_YET_RUNNABLE` naming the missing stage
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
| `test_analysis_data.py` | 6 | diagnostics and data-layer endpoints |
| `test_artifact_store.py` | 33 | content addressing, checksum verification, handle budget, T4A.3 acceptance |
| `test_api_infrastructure.py` | 16 | health, listing, pagination, CORS, data-source transparency, benchmark endpoints |
| `test_benchmarks.py` | 45 | Ground-Truth Benchmark Suite, seed discipline, climatology removal, D30 determinism |
| `test_boundary_synthetic.py` | 7 | boundary treatments, windowing, synthetic generators |
| `test_cds_source.py` | 8 | T5.2c monthly CDS planning/CLI, request refusals, network consent, atomic resume, shard integrity, route-aware replay and canonical lazy dataset compatibility |
| `test_coefficient_field.py` | 40 | T4B.1 acceptance: parent-grid alignment, perfect reconstruction per family, lineage-safe summary; DTCWT upsampling declared; LevelBank and level slicing (T4B.4) |
| `test_documentation.py` | 18 | this document and roadmap.md against the code |
| `test_dtcwt.py` | 28 | Kingsbury q-shift DTCWT: primitives vs reference, two oracles, orientation, shift invariance, D1 head-to-heads |
| `test_executor.py` | 33 | Executor backends, seed derivation, ordering, portable CPU/accelerator/HPC profiles, doctor, device/thread policy, SQLite concurrency, byte-identical sweeps |
| `test_experiments.py` | 3 | declarative sweeps and lineage |
| `test_exports.py` | 32 | CSV/JSON/NetCDF4/Zarr round trips, embedded provenance, seeded perturbation (D34) |
| `test_external_fcn3.py` | 9 | T5.6a offline FCN3 request/result schemas, exact global input and ensemble contracts, portability refusals, canonical persistence, file/tree identity and request/artifact tamper isolation |
| `test_external_ensemble_evaluation.py` | 8 | T5.6c exact truth/initialization alignment, member/mean/persistence errors, analytic CRPS and spread, area-weighted fractional-tie ranks, bounded lazy reads, content identity and scientific refusal contracts |
| `test_external_forecast_cube.py` | 8 | T5.6b authenticated lazy NetCDF/Zarr import, exact dimensions/axes/grid/variables/SI units, bounded complete finite-value scan, explicit grid-aligned NZ crop lineage and pre-open tamper refusal |
| `test_evaluation_job.py` | 7 | T5.6f canonical portable jobs, separately hashed relocatable bindings, authenticated no-write preflight, input/overwrite refusals, exact receipt binding and complete create/bind/preflight/run CLI workflow |
| `test_evaluation_run.py` | 6 | T5.6e real synthetic Zarr end-to-end orchestration, versioned/hashable controls, atomic no-overwrite receipt, nested/cross-lineage integrity and handle cleanup on failure |
| `test_matched_truth.py` | 9 | T5.6d lazy exact ERA5 initialization/valid-time selection, source/selection identity, split and FCN3-period guards, evaluator compatibility and time/grid/level/unit/alias refusals |
| `test_forecasting_adapter.py` | 5 | T5.3a exact persistence, represented autoregressive rollout, backward gradients, deterministic evidence, refusal contracts and CPU/RTX vendor-neutral accelerator parity |
| `test_forecasting_artifact_evaluation.py` | 8 | T5.3b/T5.2d checkpoint/config integrity, artifact-bound lineage, persistence-relative metrics, physical-time reporting/refusals, undefined-skill handling and CPU/RTX vendor-neutral accelerator parity |
| `test_forecasting_protocol.py` | 7 | T5.0a exact schema completeness, canonical identity, immutable nested configuration, evidence requirements, temporal/rollout consistency, persistence and tamper/drift refusal |
| `test_forecasting_protocol_binding.py` | 4 | T5.0b exact dataset/protocol/checkpoint binding, recomputed coordinate/statistics identities, drift refusals and bound-evaluation cross-run isolation |
| `test_frontend_contract.py` | 31 | the frontend/backend contract, including transform/dataset/cadence readiness claim boundaries, plus the UI integrity guards: no fabricated results, no unqualified validation claims, units and slope uncertainty displayed |
| `test_grid_operators.py` | 64 | grid metrics, metric-aware gradient/Laplacian, area weighting, physical-wavenumber spectra, D26 |
| `test_hypothesis.py` | 3 | correlation and categorical hypothesis discovery |
| `test_imports.py` | 36 | NetCDF/Zarr/CSV/JSON import, dimension pinning, axis identification, laundering guard, benchmark runs over HTTP |
| `test_migrations.py` | 25 | Alembic history, ORM/schema drift, per-revision round trips, pre-Alembic adoption, auto-migrate refusal, PostgreSQL rendering |
| `test_regional_forecast.py` | 12 | T5.2a-d alignment, ratio/calendar embargo, cadence validation, physical lead durations, train-only normalisation, provenance, eager/lazy equivalence, bounded streaming, multi-worker loading, local/HPC cache seam and independent-route comparison contract |
| `test_sequence.py` | 37 | FieldSequence validation, cadence, R6 temporal split and leakage guardrails, slice_sequence |
| `test_registries.py` | 30 | registries, error taxonomy, fallback chain, plugin acceptance and DTCWT native-visualisation contract |
| `test_stationary.py` | 19 | undecimated SWT: shift invariance, perfect reconstruction, frame constant, PyWavelets oracle, R3 normalisation |
| `test_statistics.py` | 36 | FDR procedures vs scipy, surrogate preservation properties, calibration on a true null, stationarity gate, screening |
| `test_training_representations.py` | 43 | T5.1a-e raw/FFT/DCT/Haar/db2/SWT/DTCWT batch contract, reconstruction, immutable context, analytical/PyWavelets/FFT oracles, exact complex-atlas bijection, fused/reference coefficient and gradient agreement, translation equivariance, explicit TF32 precision isolation, Mallat/channel packing, support/redundancy metadata, gradcheck, cached buffers, dtype migration and vendor-neutral accelerator parity |
| `test_cross_scale.py` | 25 | T4C.3 acceptance plus the frozen T4C.6 protocol: injected cascade/null twin, Theiler windows, support floor, power check, split sufficiency, embargo and three-state replication verdict |
| `test_scale_signature.py` | 28 | T4C.1 acceptance and the analytic values of every measure on white noise; threshold sensitivity measured; R13 interior refusals; T4C.4 power-law core |
| `test_surrogate_null.py` | 14 | T4C.2 acceptance: spectrum preserved, phase destroyed, organised scores and fBm does not; the two calibrations (wrong null, linear lag) |
| `test_wavelet_bank.py` | 27 | T4B.2 expansion through the engine's own parameter matrix, the 1,000-combination guard, decompose_bank / extract_scale_signature, the vertical-bank refusals |
| `test_transforms.py` | 13 | fft/dct/dwt/dtcwt/hybrid round trips; D1 recorded as a strict xfail |
| `test_zarr_source.py` | 58 | R13 crop geometry, chunk-hostility prediction, byte counting, cache and provenance round trip, the NetCDF engine (D33), zarr HTTP surface |
| **total** | **841** | |

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
