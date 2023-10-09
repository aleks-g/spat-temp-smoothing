# spatial-allocation
Note that this currently contains market-zone demand data from Nordpool which is no longer freely available (it was earlier at the time of downloading). Thus this data cannot be shared (and we only work here with simulated and processed data). There is alternative ENTSO-E data which seems to be flawed, but might be good enough for generating the artificial data.

The analysis can be reproduced as follows:
1. Install the conda environment from `environment.yaml` with ```conda env create -f environment.yaml```
2. We have downloaded reanalysis data and then already saved temperature data in `data/weather-input/ERA5_temperatures_1991-2020_norway_regions.csv` and wind data in `data/weather-input/ERA5_wind_norway-regions_1980-2020.csv`. It can also be regenerated through atlite with code we have provided (commented out in `prepare_inputs.ipynb` and in `temperature-fetching-norway.ipynb`).
3. Run `prepare_inputs.ipynb` to generate the input data for the analysis. These can already be found in `data/processing` or `data/demand-input`.
4. Run `simulate-penalties.ipynb` to generate the results (and sensitivity analyses).
5. Run `analyse_outputs.ipynb` for the analysis of the results.
