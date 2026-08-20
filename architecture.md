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

## 2. The Core Spine: `PhysicalField`

The primary data structures of atmospheric models (temperature, geopotential, wind velocities) are multidimensional grids. In SpectralEarth, the unified spine of the entire application is the **`PhysicalField`** class (`src/physical_core/field.py`). 

Instead of passing raw PyTorch tensors across modules, every analytical engine, generator, transform, and adapter operates on a `PhysicalField` object. 

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
*   **GRIB is not supported.** No GRIB branch exists, `cfgrib` is not a dependency, and only the `.nc` extension is probed. The misleading code comment has been removed and the class docstring now states this explicitly. Remote sources (CDS API, S3) and Zarr are Task 3.5.20.
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

`device.py` selects CUDA -> MPS -> CPU with an explicit `SPECTRAL_DEVICE` override, and
**refuses** a device that is not present rather than silently falling back - a run that claims
to have used a GPU must have used one. `describe()` records the device, thread counts and
determinism mode into every run's `execution` column.

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

**R13 enforced, not documented.** `edge_exclusion(j) = floor((L-1)·2^(j-1)/2)` reproduces R13's
table exactly (6/13/26/52 px per side at levels 1–4 for a 14-tap filter), and
`minimum_crop_size` returns R13's own figures — 256 for four levels, 512 for five. A crop below
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

27 routes. Listed here because an undocumented endpoint is an untested contract.

| Method | Route | Notes |
|---|---|---|
| GET | `/api/v1/health` | DB reachability, backend scheme, dataset count, execution device (T3.5.10) |
| POST | `/api/v1/transforms/apply` | fft, dct, dwt, hybrid, **swt**, **dtcwt** (real Kingsbury q-shift, T3.5.6; `level1`/`qshift` sweepable) |
| POST | `/api/v1/synthetic/generate` | vortex, front, turbulence, wave |
| POST | `/api/v1/synthetic/perturb` | rotate, translate, noise (now seedable, T3.5.12) |
| POST | `/api/v1/boundary/analyze` | declares its pixel frame explicitly (D13) |
| GET | `/api/v1/actions` | every registered pipeline action, generated from the registry (T3.5.15) |
| GET | `/api/v1/transforms` | every registered transform with its params and capabilities (T3.5.15) |
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
*   **Main Application (`App.tsx`):** ~2,400 lines covering state hooks for **nine** tabs (Synthetic Generator, Meteorological Data, Boundary-Condition Lab, Spectral Transforms, Diagnostic & Analysis, Experiment Engine, Automated Hypotheses, **Platform & Evidence**, **Real ERA5 (Zarr)**), loading indicators, dynamic sliders, and follow-up proposal adoption.
*   **Platform & Evidence tab (T3.5.22):** the execution device, core and thread counts, executor backends with the measured rationale for the serial default, the SQLite pragmas actually in force, the stamped Alembic revision with an explicit warning when the schema is behind the code, the data-source fallback chain labelled observational/SIMULATED from each source's own declared flag, and the full Ground-Truth Benchmark Suite with its declared known answers and null benchmarks marked. All of this existed on the backend for several slices with no consumer.
*   **Real ERA5 tab (T3.5.22):** a crop form driven by the store catalogue, the R13 minimum crop size for the chosen number of wavelet levels, and an **inspect** action that reports chunk structure, the predicted amplification, the chunk-hostility warning and the per-level valid interior — metadata only, no transfer — then hands back the CLI command that would materialise it. The network gate is shown when it is off, with the variable that enables it.
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
| Python venv + dependencies | installed (torch 2.13.0, numpy 2.2.6, pydantic 1.10.26, SQLAlchemy 2.0.52, xarray 2025.6.1, FastAPI 0.110.3) |
| Backend test suite | **647 passed, 1 xfailed** (plus 1 skipped: the opt-in live-GCS check) (was 8 failed / 11 passed at first run; 65 after T3.5.0, 152 after T3.5.7, 222 after T3.5.13, 286 after T3.5.17, 351 after T3.5.6, 379 after T3.5.15, 407 after T3.5.19, 449 after T4C.5) |
| Ground-Truth Benchmark Suite | **15 PASS, 0 FAIL, 2 NOT_YET_RUNNABLE** (`python -m src.benchmarks`, exit 0) |
| Frontend `npm install` + `npm run build` | passes, emits 1,378 modules + real JS/CSS assets (was: 1 module, no assets) |
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
| D18 | `experiment_engine/engine.py:18-22` | `get_execution_device` probes CUDA only. No Apple-silicon MPS branch and no explicit CPU-thread configuration, so a large class of development laptops silently runs the slowest available path. | **PARTIAL** T3.5.21 - selection chain, override, refusal path and thread budget implemented and tested on CPU; the CPU/CUDA/MPS agreement check cannot be run on this CPU-only machine and is not claimed |

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
| `test_api_infrastructure.py` | 16 | health, listing, pagination, CORS, data-source transparency, benchmark endpoints |
| `test_benchmarks.py` | 45 | Ground-Truth Benchmark Suite, seed discipline, climatology removal, D30 determinism |
| `test_boundary_synthetic.py` | 7 | boundary treatments, windowing, synthetic generators |
| `test_documentation.py` | 18 | this document and roadmap.md against the code |
| `test_dtcwt.py` | 28 | Kingsbury q-shift DTCWT: primitives vs reference, two oracles, orientation, shift invariance, D1 head-to-heads |
| `test_executor.py` | 26 | Executor backends, seed derivation, ordering, device/thread policy, SQLite concurrency, byte-identical sweeps |
| `test_experiments.py` | 3 | declarative sweeps and lineage |
| `test_exports.py` | 32 | CSV/JSON/NetCDF4/Zarr round trips, embedded provenance, seeded perturbation (D34) |
| `test_frontend_contract.py` | 28 | the frontend/backend contract, plus the UI integrity guards: no fabricated results, no unqualified validation claims, units and slope uncertainty displayed |
| `test_grid_operators.py` | 64 | grid metrics, metric-aware gradient/Laplacian, area weighting, physical-wavenumber spectra, D26 |
| `test_hypothesis.py` | 3 | correlation and categorical hypothesis discovery |
| `test_imports.py` | 36 | NetCDF/Zarr/CSV/JSON import, dimension pinning, axis identification, laundering guard, benchmark runs over HTTP |
| `test_migrations.py` | 25 | Alembic history, ORM/schema drift, per-revision round trips, pre-Alembic adoption, auto-migrate refusal, PostgreSQL rendering |
| `test_registries.py` | 29 | registries, error taxonomy, fallback chain, and the T3.5.15 plugin acceptance criterion |
| `test_stationary.py` | 19 | undecimated SWT: shift invariance, perfect reconstruction, frame constant, PyWavelets oracle, R3 normalisation |
| `test_statistics.py` | 36 | FDR procedures vs scipy, surrogate preservation properties, calibration on a true null, stationarity gate, screening |
| `test_transforms.py` | 13 | fft/dct/dwt/dtcwt/hybrid round trips; D1 recorded as a strict xfail |
| `test_zarr_source.py` | 58 | R13 crop geometry, chunk-hostility prediction, byte counting, cache and provenance round trip, the NetCDF engine (D33), zarr HTTP surface |
| **total** | **492** | |

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
*   ~~`get_execution_device` round-robins `run_idx % num_gpus`, but sweeps run strictly sequentially, so multi-GPU assignment is cosmetic.~~ **Addressed in T3.5.19/T3.5.21:** runs are distributed through the Executor seam and `device.select_device(run_idx=...)` spreads them across CUDA devices. Untested on real multi-GPU hardware - this machine is CPU-only, and that is stated rather than implied.
*   The DWT is **decimated**: each level halves resolution, so scale *n* lives on a different grid from the parent field. This is correct for compression and reconstruction, but it makes cross-scale spatial reasoning awkward - the reason Phase 3.5 adds an undecimated SWT alongside it.
*   `PhysicalField` is **strictly 2D** and raises on any other rank (`field.py:19`). There is no time axis anywhere in the compute layer: the adapter returns a single timestep, and `decompose_by_lead_time` only works because the caller assembles the list itself. Phase 4A introduces `FieldSequence`.
*   Lineage `value` columns and inter-step `step_outputs` carry **full payloads** as nested Python lists (`analyze_boundary` returns an entire padded field this way). This is a hard scaling wall for coefficient fields, addressed by the `ArtifactStore` in Phase 4A.
*   ~~No `GET /api/v1/experiments` collection endpoint and no health endpoint.~~ **Both added (T3.5.10):** paginated listing with `limit`/`offset`/`status`, and `GET /api/v1/health` reporting database reachability, backend scheme, dataset count and execution device. The frontend does not yet consume either — wiring them into the Experiment Engine tab is outstanding.
*   ~~Alembic is absent - the schema is created by `Base.metadata.create_all` in the FastAPI lifespan, so there is no migration path for the seven tables Phase 4 adds.~~ **Fixed in T3.5.8** (Section 4.1). `create_all` was not merely missing a migration path: it silently failed to add columns to tables that already existed, which had already broken the real `spectral_earth.db` (defect D32).
