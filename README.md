# SpectralEarth: Visual Research Workbench & Discovery Engine

SpectralEarth is a research workbench under active scientific validation for multiscale
atmospheric analysis, regional spectral modelling, data-assimilation diagnostics and
hypothesis screening.

The platform joins a tensor-accelerated computational backend (**PyTorch**, **xarray**,
**SQLAlchemy**) to a React/Vite/Plotly dashboard and a Jupyter playground. It is not yet a
validated forecasting system: the decisive Phase 4C real-ERA5 gate has not been run, and Phase 5
has integration contracts but no completed learned forecasting comparison. Current status,
evidence and known limitations live in `roadmap.md`, `VERIFICATION.md` and `architecture.md`
respectively.

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
the declared held-out split. This is ready for Adam's real manifest, but the current acceptance
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
mixed-precision/compilation acceptance, non-NVIDIA hardware evidence, a viable multi-year
regional source (D43), an actual independent ERA5 overlap run and execution of the actual
laboratory model remain outstanding.

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
actual professor/laboratory architecture and its history semantics remain unintegrated; no
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

For example, Adam can use `auto` on his laptop and set `SPECTRAL_PROFILE=hpc` inside an
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
resume and conversion contracts pass; **a real CDS request, multi-year NZ crop and independent
WeatherBench overlap check have NOT RUN**, so D43 remains open.

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

Those bounds and dates are an interface example, **not Emily's experiment specification**.
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

**Not supported:** GRIB (`.grib`/`.grib2`) ingestion via `cfgrib`, dateline-crossing CDS boxes
without splitting them into two requests, and NOAA HRRR/GFS object-store retrieval.

---

## Licence

SpectralEarth is proprietary software; it is not released under an open-source licence.
Copyright remains with Edward Jonathan Bentley. [The repository licence](LICENSE.md) grants
Adam Frank Bentley a named, perpetual, worldwide and royalty-free right to use and modify the
platform for lawful personal, academic, scientific and commercial work, while reserving public
redistribution and sublicensing of the SpectralEarth core. Independently authored extensions
remain separate under the terms stated there. Third-party libraries, datasets, papers and model
artifacts retain their own licences and conditions.
