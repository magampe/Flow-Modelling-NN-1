To run the main function, please ensure all dependencies are installed on your working environment.

To run the model on the 2016-2021 data, open a command prompt in the root directory and run python main.py


Data sources
- DEM: https://www.hydrosheds.org/hydrosheds-core-downloads
    - ACC                       Accumulation
    - DIR                        Directional
    - DEM                     Digital Elevation Model
    - ASPECT                Aspect
    - ROUGH                Roughness
    - SLOPE                   Slope
    - TRI                        Terrain Ruggedness Index
    - PRISM                  Precipitation
    - NLCD                    Land Cover
    - HYDRO                 Hydrologic Group (stored as digits, rather than categories: 0: NULL, 1: A, 2: B, 3: C, 4: D, 5: BD, 6: CD)
    - MFC                      (Mukey) Field Capacity

- PRISM: https://prism.oregonstate.edu/historical/
    - PPT
    - TMEAN
    - Comes in as .bil files. Need to be cropped to station shpfile area (compute_bil_data.py) and resaved as .tif files.
    - Train/Validation/Test Split (Pay attention to time series lengths)
    - Feed into Custom PyTorch Dataloader


Target Data
- Streamflow: https://waterdata.usgs.gov/nwis/dv?referred_module=sw&search_criteria=state_cd&search_criteria=site_tp_cd&submitted_form=introduction
    - Discharge, m³/s
    - Tab-separated data
    - Filter to appropriate Station
    - Train/Validation/Test Split (Pay attention to time series lengths)