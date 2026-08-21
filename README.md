# SpectralEarth: Visual Research Workbench & Discovery Engine

SpectralEarth is a research workbench under active scientific validation for multiscale
atmospheric analysis, regional spectral modelling, data-assimilation diagnostics and
hypothesis screening.

The platform joins a tensor-accelerated computational backend (**PyTorch**, **xarray**,
**SQLAlchemy**) to a React/Vite/Plotly dashboard and a Jupyter playground. It is not yet a
validated forecasting system: the decisive Phase 4C real-ERA5 gate has not been run, and the
learned forecasting comparison is future Phase 5 work. Current status, evidence and known
limitations live in `roadmap.md`, `VERIFICATION.md` and `architecture.md` respectively.

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
network, laboratory-model adapter, training loop or forecast-skill evaluation across
representations is implemented. That must not be inferred from the current data/transform tools.

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
normalisation artifact. The same `prepare_cached_regional_forecast(..., cache_dir=...)` call
works against a laptop folder or an HPC shared cache and returns CPU tensors for the training
loop to place on CUDA, ROCm or MPS.

```python
from torch.utils.data import DataLoader
from src.data_layer.regional_forecast import (
    RegionalForecastConfig, prepare_cached_regional_forecast)
from src.data_layer.zarr_source import CropSpec

config = RegionalForecastConfig(
    level_hpa=850, history_frames=2, lead_frames=(1, 2, 4), embargo_frames=4)
bundle = prepare_cached_regional_forecast(crop_spec, config, cache_dir=shared_cache)
train_loader = DataLoader(bundle.train, batch_size=8, shuffle=True, num_workers=0)
```

`crop_spec` and `shared_cache` are intentionally explicit; preparation never downloads or
silently falls back to simulated data. The ERA5 panel reports only manifest-level structural
eligibility until values are opened, and keeps preparation, train-only normalisation and the
independent-route overlap check as separate claims. **This is not the whole forecasting path:**
mixed-precision/compilation acceptance, non-NVIDIA hardware evidence, a viable multi-year
regional source (D43), an actual independent ERA5 overlap run, the model adapter and evaluation
harness remain outstanding.

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

**Not supported:** GRIB (`.grib`/`.grib2`) via `cfgrib`, direct Copernicus CDS retrieval, and
NOAA HRRR/GFS object-store retrieval.
