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
*   **DTCWT (Dual-Tree Complex Wavelet Transform) — NOT CURRENTLY A DTCWT:** The code in `apply_dtcwt2d` runs four parallel filter-bank trees (AA, AB, BA, BB) and labels their outputs `_real`/`_imag`, but tree B's filters are degenerate. At `transforms.py:148-151`, `h_b = [cos(pi/4), sin(pi/4)] = [0.7071, 0.7071]`, which is **identical** to the Haar low-pass `h_a`, and `g_b = [-sin(pi/4), cos(pi/4)] = -g_a`. All four trees are therefore the same filter bank up to a sign. There is **no Hilbert pair, no analytic complex response, no shift invariance, and no directional subbands** — the `_real` and `_imag` tensors hold identical or sign-flipped data. Reconstruction is still exact (the four trees average back to the input), so the round-trip test passes and masks the defect. Replacing this with genuine Kingsbury q-shift / near-symmetric filters is Task 3.5.6 of the roadmap and is a hard prerequisite for the Phase 4 spectral-feature layer, which depends on shift invariance.
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

### 3.7 Automated Hypothesis Engine (`src/hypothesis_engine/engine.py`)
Mines database results to generate hypotheses:
*   **Numerical Correlations:** Runs Pearson correlation coefficient calculations between experiment input parameters and numeric metrics, producing a `Hypothesis` for strong correlations ($|r| \ge \text{threshold}$).
*   **Categorical Optimizers:** Performs categorical group-mean analysis, finding the optimal categories for metrics and computing relative performance improvements.
*   **Adaptive Follow-Up Proposals:** Generates a new `proposed_experiment_config` targeting the optimal parameters (either expanding the range of numerical parameters in the correct correlation direction or fixing a categorical parameter to its best-performing category).

---

## 4. Database Schema and State Tracking

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

The React frontend is fully written and structurally complete, but **has never been installed or built in this workspace** (see Section 7).

*   **Component Visualizations:** `Heatmap2D.tsx` and `LineChart.tsx` wrap `react-plotly.js`; `LineageGraph.tsx` is a hand-rolled SVG node-link renderer with a tooltip inspector and no external graph dependency. All three take reactive props and render spatial fields, PSD curves, coherence ratios, and provenance DAGs.
*   **Main Application (`App.tsx`):** 1,913 lines covering state hooks for all seven tabs (Synthetic Generator, Meteorological Data, Boundary-Condition Lab, Spectral Transforms, Diagnostic & Analysis, Experiment Engine, Automated Hypotheses), loading indicators, dynamic sliders, and follow-up proposal adoption.
*   **API Integration:** `src/services/api.ts` covers every backend endpoint via `fetch` against the relative base `/api/v1`. The offline fallback lives in `App.tsx`, not in the client service - each tab catches the network error and substitutes a local mock generator (`getMockDatasets`, `runMockFieldGenerator`, `applyMockPerturbation`, and the mock lineage fixture at `App.tsx:647`). The relative base URL means the frontend depends entirely on the Vite dev proxy (`vite.config.ts`), because the API declares no CORS middleware.
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
| Backend test suite | **152 passed, 1 xfailed** (was 8 failed / 11 passed at first run) |
| Frontend `npm install` + `npm run build` | passes, emits 1,378 modules + real JS/CSS assets (was: 1 module, no assets) |
| Backend server | starts, serves OpenAPI, all smoke-tested endpoints return 200 |
| End-to-end experiment sweep | 9-run parameter sweep completes 9/9, writes 28 lineage nodes / 54 edges, hypothesis engine returns results |
| Version control | `git init` run to enable E5 provenance capture; no commit made yet |

Earlier revisions of this document and of `roadmap.md` claimed the platform was "validated"
and "zero-error". It was not: the first real execution produced 8 test failures and a frontend
that had never rendered. The ledger below grew from 18 entries to 25 as a direct result of
running the code — **17 of which are now fixed**.

### 7.2 Confirmed defects

| # | Location | Defect | Fixed by |
|---|---|---|---|
| D1 | `transform_engine/transforms.py:148-151` | DTCWT tree-B filters are identical to tree A up to sign - no Hilbert pair, no shift invariance, no directional subbands. Exact reconstruction hides it. | T3.5.6 |
| D2 | `transform_engine/transforms.py` `inverse_hybrid` | Recombines `w*low + (1-w)*high` where the forward pass split `low + residual`; not an inverse for any `w`. | **FIXED** T3.5.5 |
| D3 | `tests/test_experiments.py:49` | `test_experiment_engine_execution` **fails**. The test seeds an in-memory SQLite engine, but `execute_experiment` opens the file-backed `SessionLocal` from `database/session.py:11`, never finds the experiment, and returns early leaving status `PENDING`. | **FIXED** T3.5.1 |
| D4 | repo root | No `conftest.py` and no `__init__.py` anywhere, so bare `pytest` cannot resolve `src.*` imports (only `python -m pytest` works, via implicit cwd insertion). | **FIXED** T3.5.1 |
| D5 | `api/main.py` | No `CORSMiddleware`. The API is reachable only through the Vite dev proxy; any separate-origin or static-hosted deployment breaks. | **FIXED** T3.5.2 |
| D6 | `frontend/` | `tailwind.config.js` and `postcss.config.js` absent while `index.css` uses `@tailwind` and `@apply`. Build fails or emits unstyled output. | **FIXED** T3.5.3 |
| D7 | `start_platform.ps1:48-50` | uvicorn is launched inside `Start-Job`, which gets a fresh runspace rooted at the user home directory, so `src.api.main` is not importable and the backend dies silently while the frontend appears to start fine. | **FIXED** T3.5.4 |
| D8 | `hypothesis_engine/engine.py:110-143` | Emits a hypothesis for every parameter x metric pair exceeding `abs(r) >= 0.3` with **no multiple-comparison control**. Tolerable at current scale; becomes a false-discovery generator the moment Phase 4 adds scales, lags and constellations to the search space. | T4C.5 |
| D9 | `data_layer/adapters.py` | Class-level `_datasets` cache is never invalidated; a newly added `.nc` file is ignored until process restart. GRIB is advertised in a code comment but unimplemented. | **FIXED** T3.5.9 |
| D10 | `requirements.txt` | Missing `ipywidgets`, `plotly` and `matplotlib`, which `research_playground.ipynb` requires per the README. `alembic` also absent. | **FIXED** T3.5.11 |
| D11 | `.vscode/launch.json` | The Chrome configuration declares `"name"` twice; VS Code silently keeps the last, so the compound `Debug Platform (Both)` reference is fragile. | **FIXED** T3.5.4 |
| D12 | `data_layer/adapters.py:79`, `synthetic_generator/perturbation.py:43,46,50` | **Unseeded RNG.** `np.random.randn` and `torch.randn_like`/`rand_like` are called with no seed and no seed capture. Every noise perturbation and the simulated GFS wind field are irreproducible, which **directly contradicts the provenance pillar** - a lineage graph that cannot reproduce its own run is a record, not provenance. | T3.5.12 |
| D13 | `analysis_engine/diagnostics.py:68-69`, `boundary_lab/boundary.py:107` | **Metric-unaware differential operators.** `torch.gradient` is called with no `spacing`, so gradients are per-pixel, not per-metre. On a lat/lon grid the zonal spacing varies as `cos(lat)` and differs from the meridional spacing, so gradient magnitude, gradient angular error and boundary gradient decay are all systematically distorted, worsening toward the poles. Radial PSD likewise bins in pixel wavenumber and assumes isotropy that a lat/lon grid does not have - so the Charney/Kolmogorov regime classification is being made in the wrong space. | **FIXED** T3.5.13 |
| D14 | `api/main.py` (10 sites) | Every `except` re-raises a generic 500 with a fixed string (`"An error occurred during ..."`), discarding the exception entirely. For a research tool this is the difference between a usable diagnostic and a dead end. | T3.5.14 |
| D15 | `data_layer/adapters.py:150,162`, `experiment_engine/engine.py` | **The "seams" described in Section 5 are not extension points.** `_get_simulated_fallback` is a hard-coded if/elif over three dataset ids, `list_datasets` iterates a hard-coded literal list, and `_execute_action` is a 13-branch if/elif chain. Adding a data source or a pipeline action requires editing core engine files. | T3.5.15 |
| D16 | `physical_core/field.py:20` | `PhysicalField.__init__` force-casts to float32 with no opt-out. Acceptable for visualisation; marginal for surrogate ensemble statistics, log-log power-law fits, and mutual-information/transfer-entropy estimation in Phase 4C. | **FIXED** T3.5.16 |
| D17 | `analysis_engine/diagnostics.py:29,56`, `analysis_engine/decomposition.py:81`, `boundary_lab/boundary.py:31,123` | **Per-bin Python loops over full arrays.** Five functions bin values by radius or distance using `for k in range(...)` with a fresh boolean mask over the *entire* array each iteration - `O(bins x H x W)` where `O(H x W)` suffices via `bincount`/`scatter_add`. On a 512x512 field (`max_r = 256`) `compute_radial_psd` performs ~256 full passes, roughly 67M element visits instead of 262k. These are the innermost functions of the Phase 4C loop, called inside a surrogate ensemble; unfixed, they alone decide whether the platform is usable on a laptop. | T3.5.20 |
| D18 | `experiment_engine/engine.py:18-22` | `get_execution_device` probes CUDA only. No Apple-silicon MPS branch and no explicit CPU-thread configuration, so a large class of development laptops silently runs the slowest available path. | T3.5.21 |

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

### 7.3 Precision caveats (not defects, but do not overstate them)

*   `compute_ssim` is single-window global SSIM, not locally-windowed SSIM (Section 3.2).
*   `get_execution_device` round-robins `run_idx % num_gpus`, but `execute_experiment` runs sweeps strictly sequentially, so multi-GPU assignment is currently cosmetic.
*   The DWT is **decimated**: each level halves resolution, so scale *n* lives on a different grid from the parent field. This is correct for compression and reconstruction, but it makes cross-scale spatial reasoning awkward - the reason Phase 3.5 adds an undecimated SWT alongside it.
*   `PhysicalField` is **strictly 2D** and raises on any other rank (`field.py:19`). There is no time axis anywhere in the compute layer: the adapter returns a single timestep, and `decompose_by_lead_time` only works because the caller assembles the list itself. Phase 4A introduces `FieldSequence`.
*   Lineage `value` columns and inter-step `step_outputs` carry **full payloads** as nested Python lists (`analyze_boundary` returns an entire padded field this way). This is a hard scaling wall for coefficient fields, addressed by the `ArtifactStore` in Phase 4A.
*   ~~No `GET /api/v1/experiments` collection endpoint and no health endpoint.~~ **Both added (T3.5.10):** paginated listing with `limit`/`offset`/`status`, and `GET /api/v1/health` reporting database reachability, backend scheme, dataset count and execution device. The frontend does not yet consume either — wiring them into the Experiment Engine tab is outstanding.
*   Alembic is absent - the schema is created by `Base.metadata.create_all` in the FastAPI lifespan, so there is no migration path for the seven tables Phase 4 adds.
