# SpectralEarth Data Ingestion Folder

Place your real scientific NetCDF data streams in this folder:
- For ERA5 Reanalysis: Name your file `era5_reanalysis.nc`
- For GFS Forecast: Name your file `gfs_forecast.nc`
- For Toy Climate Model: Name your file `toy_climate_model.nc`

The `MeteorologicalDataAdapter` (`src/data_layer/adapters.py`) will automatically intercept these files, load them via `xarray`, and enable interactive coordinate cropping, pressure-level selections, and variable extractions inside the UI and experiment sweep engines!

If no files are found here, the platform falls back to fully simulated high-fidelity in-memory meteorology grids so you can work and run sweeps offline without any dataset files.
