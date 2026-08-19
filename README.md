# SpectralEarth: Visual Research Workbench & Discovery Engine

SpectralEarth is a high-performance, scientific-grade visual research workbench designed for multi-scale atmospheric physics, regional spectral modeling, data assimilation benchmarking, and automated hypothesis discovery.

The platform bridges a highly optimized, tensor-accelerated computational backend (**PyTorch**, **xarray**, **SQLAlchemy**) with a reactive, publication-ready research dashboard (**React**, **Vite**, **Plotly**), and a structured exploratory **Jupyter Playground**.

---

## 🗺️ Key Scientific Pillars

### 1. Multiscale Transforms (FFT, DCT, DWT, Hybrid) &mdash; DTCWT in progress
SpectralEarth ships working 2D FFT, DCT-II/III, multi-level Haar DWT, and a hybrid FFT+DWT representation, all with exact or near-exact inverses.

> **Honest status:** the function named `apply_dtcwt2d` is **not yet a true Dual-Tree Complex Wavelet Transform.** Its second filter tree is numerically identical to the first up to a sign, so it provides *no* shift invariance and *no* directional subbands, despite the `_real`/`_imag` labelling. Genuine Kingsbury q-shift filters (giving real shift invariance and 6 oriented subbands at $\pm 15^\circ, \pm 45^\circ, \pm 75^\circ$) are **Task 3.5.6** in `roadmap.md`, and are a hard prerequisite for the Phase 4 feature-tracking layer. See `architecture.md` Section 3.1 and defect D1 for the full analysis.

### 2. Turbulence Spectral Slope ($\beta$) Fitting
To verify if models (including Neural Weather Operators) preserve physical consistency and kinetic energy cascades, the engine computes radial Power Spectral Density (PSD) and fits a power-law exponent ($\beta$) using log-log least-squares regression:
$$\ln(E(k)) = \ln(C) - \beta \ln(k)$$
It automatically classifies flow regimes into **Charney 2D Enstrophy Cascade ($\beta \approx 3.0$)**, **Kolmogorov 3D Mesoscale Cascade ($\beta \approx 5/3$)**, or flat noise spectrums.

### 3. Boundary-Condition Lab
Regional models are prone to boundary artifacts. The lab applies Circular, Reflective, and Replicate padding, alongside Tukey (cosine-tapered), Hann, and Hamming tapering windows to analyze spatial gradient decay and quantify **Spectral Leakage** in Fourier space.

### 4. Dynamic Provenance Lineage DAG
Ensures waterproof reproducibility. Every pipeline sweep commits code revisions, input datasets, coordinate configurations, transform coefficients, and final diagnostics as unique nodes and edges inside a directed acyclic graph (DAG), visualized live inside the UI.

### 5. Automated Hypothesis Discovery
The engine mines relational runs tables, computing Pearson's $r$ correlation coefficients and categorical optimization metrics. It automatically generates **adaptive, follow-up experiment configurations** to prove discovered relationships, which can be adopted back into the pipeline in a single click.

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
│   ├── data_layer/          # Ingestion adapters supporting in-memory & NetCDF files
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

It features a 5-step, fully interactive widgets dashboard built with `ipywidgets` and `plotly` that mirrors the frontend dashboard. (Note: `ipywidgets`, `plotly` and `matplotlib` are **not** yet listed in `requirements.txt` - install them separately until Task 3.5.11 lands.) Your son can dynamically tweak wavelet levels, modify Tukey alpha tapers, fit Kolmogorov/Charney turbulence slopes, and **export formal markdown reports** (`spectral_earth_active_report.md`) directly from his notebooks.

---

## 📂 Live Data Ingestion (NetCDF4)

Place your real-world meteorological cutouts inside the `data/` folder:
*   For ERA5 Reanalysis: Name your file `era5_reanalysis.nc`
*   For GFS Forecast: Name your file `gfs_forecast.nc`
*   For Toy Climate Model: Name your file `toy_climate_model.nc`

The `MeteorologicalDataAdapter` (`src/data_layer/adapters.py`) will automatically identify these files, parse them via `xarray`, and enable interactive coordinate cropping, pressure-level selections, and variable extractions inside the UI. If no files are present, the system defaults to high-fidelity, in-memory simulated weather fields for sandbox testing.

**Not yet supported:** GRIB (`.grib`/`.grib2`) parsing via `cfgrib`, and remote retrieval from the Copernicus CDS API or AWS S3 (NOAA HRRR/GFS). Only local `.nc` files are probed. Datasets are also memoised for the process lifetime, so a newly added file requires an API restart. Both are tracked as Task 3.1-remaining and Task 3.5.9 in `roadmap.md`.
