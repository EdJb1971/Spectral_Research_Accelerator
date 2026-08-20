# SpectralEarth Data Ingestion Folder

Place your real scientific NetCDF data streams in this folder:
- For ERA5 Reanalysis: Name your file `era5_reanalysis.nc`
- For GFS Forecast: Name your file `gfs_forecast.nc`
- For Toy Climate Model: Name your file `toy_climate_model.nc`

The `MeteorologicalDataAdapter` (`src/data_layer/adapters.py`) will automatically intercept these files, load them via `xarray`, and enable interactive coordinate cropping, pressure-level selections, and variable extractions inside the UI and experiment sweep engines!

If no files are found here, the platform falls back to fully simulated high-fidelity in-memory meteorology grids so you can work and run sweeps offline without any dataset files.

## Real ERA5 without downloading anything by hand (T3.5.18)

The files above are the *manual* path. The `era5_zarr` source streams regional crops straight
from public WeatherBench 2 Zarr on Google Cloud Storage - anonymously, no account, no credentials
- and caches them here under `zarr_cache/`, rechunked for the access pattern this platform
actually has.

    # what stores exist, and what each is good and bad for
    python -m src.data_layer.zarr_source catalogue

    # metadata only - NO data transfer. Run this first.
    python -m src.data_layer.zarr_source inspect --start 2020-06-01 --end 2020-06-08         --lat -4 60 --lon 0 64 --levels 850,700,500,300

    # the actual transfer (needs SPECTRALEARTH_ALLOW_NETWORK=1)
    SPECTRALEARTH_ALLOW_NETWORK=1 python -m src.data_layer.zarr_source materialise         --start 2020-06-01 --end 2020-06-08 --lat -4 60 --lon 0 64 --levels 850,700,500,300

**Run `inspect` before `materialise`, every time.** The 0.25 degree stores are chunked one
timestep x all levels x the whole globe, 54 MB a chunk, so a regional crop fetches ~51x more
than it uses. Measured: a 257x257 four-level crop over 8 days moves 951 MB and takes 3 minutes;
the same crop over **one year moves 79 GB and takes about two hours**. `inspect` tells you which
of those you have asked for before you wait for it.

Two consequences that are not obvious and that `inspect` will state for you:

*   **Asking for fewer pressure levels saves nothing over the network.** Level is inside the
    chunk. Only narrowing the variables or the time window reduces the transfer.
*   **A crop must be at least 256x256 for four wavelet levels** (512x512 for five), and the
    adapter refuses anything smaller rather than producing edge artefacts that look exactly like
    discoveries. Constrain the number of frames instead - never the grid.

Network access is **off by default**. A materialised crop then works entirely offline, which is
the point of caching it: `python -m src.data_layer.zarr_source cached` lists what you have.
