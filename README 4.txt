------------------------------
iPAR Yield model version 2023
------------------------------
Authors: 
	- Urs Christoph Schulthess (CIMMYT-China)
	- Ernesto Giron E. (Independent Researcher)

Last update: Aug 3, 2023

Model that estimates yield under non-stressed conditions as a function of iPAR, 
temperature and solar radiation.

Copyright (C) 2023 CIMMYT-Henan Collaborative Innovation Center
---------------------------------

Model parameters for all runs:

"CROP_OPTIMUM_TEMPERATURE": 18, # TOpt
"RUE": 3, # Radiation use efficiency (3 g/MJ)
"DRYMATTER":0.8,
"FACTOR_TON_HA":0.01,
"YIELD_FACTOR": 0.8 * 0.01, #DRYMATTER * FACTOR_TON_HA,
"TMIN_PERC_FACTOR":0.25,
"CROP_TBASE_GDD": 0,
"DAP": 35, # days after planting
"D2M": 40, # days Period heading to maturity
"CONST_DAYHR_AS" : 10.8, # Day length constant
"GDD_Required_to_EmergenceDate":180, # Thermal time or Growing Degree Days
"NDVI_constantIPAR": 0.19, # constant for Eq. in iPAR
"NDVI_lowerThreshold": 0.16,
"NDVI_Threshold": 0.15, # Threshold for linear eq. for Emergence to Heading
"NDVI_max": 0.95, # Maximum NDVI
"NORM_iPAR_EH_BOUNDS": 0.5,
"TDAYS_THRESHOLD_MATURITY": 42, # threshold to adjust the number of temperature days
"DAYS_GRAIN_FILLING":40, # duration (days) of grain filling 


The weather data is from AgERA5 and the crop data is from IWIN SAWYT dataset from 1992 to 2021

-------------------------- 
This folder contains SAWYT simulation results using estimated phenology (heading and maturity dates):

- Observed Phenology 
Simulation using observed days to heading and observed days to maturity as inputs.
To distinguish the results in the table or CSV file, the header only have E, H, M, EH, EM or HM; or sometimes none of them for example "SimYield" or "cGDD".

- All estimated 
Simulation using estimated heading and estimated maturity as inputs.
To distinguish the results in the table or CSV file, the variables have a "p" before the letter that specific the phenologic stage, eg. pE, pH, pM, pHM or SimYield_pHpM

- Mixed Phenology
Simulation using observed heading and estimated maturity as inputs.
To distinguish the results in the table or CSV file, the variables have a "p" before the letter that specific the phenologic stage, eg. HpM, EpM


--------------------------
Each of the results include a CSV file with the summary of the runs per location-occ-year. 

- Data Properties
The header of file is as follows:

'UID': Unique identifier for the location-occ-year,
'location': numeric code of the location,
'Occ': Occurrence of the trial or replicates,
'loc_code': Trial name,
'country': Country name,
'locationname': Name of the location or institution,
'lat': Latitude of the site,
'lon': Longitude of the site,
'cycle': Cycle of the crop,

'nursery': Trial name group,
'sowing': Date of sowing as self reported by the farmer,
'emergence': Emergence date,
'heading': Heading date,
'maturity': Maturity date,
'Days_To_Heading': Days to heading after sowing date,
'Days_To_Maturity': Days to maturity after sowing date,
'YearofSow': Year of sown,
'YearofEme': Year of Emergence,
'DOYofSow': Day of the Year for Sowing date,
'DOYofEme': Day of the Year for Emergence date,
'Date@35DAS': Date 35 days after sowing date,
'DayLength@35DAS': Daylength 35 days after sowing date,

'Obs_DaysHM': Observed days from Heading to Maturity
'PredEmergence': Estimated Emergence date (180 GDD after sowing date)
'PredDaysToEmergence': Estimated days to emergence after sowing date
'PredDaysToHead': Estimated days to heading after sowing date
'PredHeading': Estimated heading date after sowing date
'PredDaysToMaturity': Estimated days to maturity after sowing date
'PredMaturity': Estimated maturity date after sowing date
'NDVI_atHeading': Estimated NDVI at heading

'Days_SH': Observed days from sowing to heading
'Days_SpH': Observed days from sowing to estimated heading 
'Days_SM': Observed days from sowing to maturity
'Days_SpM': Observed days from sowing to estimated maturity
'Days_HM': Observed days from heading to maturity
'Days_HpM': Days from heading to estimated maturity
'Days_pHM': Days from estimated heading to maturity
'Days_pHpM': Estimated days from heading to maturity

'Tmin': Minimum Temperature in °C
'Tmax': Maximum Temperature in °C
'Tavg': Average Temperature in °C
'Srad': Shortwave Radiation in MJ/m2/day
'Pcp': Precipitation in mm/day

'Tmin_EH': Site Avg. minimum temperature for observed Emergence to Heading period in °C
'Tmax_EH': Site Avg. maximum temperature for observed Emergence to Heading period in °C
'Tavg_EH': Site Avg. mean temperature for observed Emergence to Heading period in °C
'Srad_EH': Site Avg. solar radiation for observed Emergence to Heading period in MJ/m2/day
'Pcp_EH': Site Total amount of precipitation for observed Emergence to Heading period in mm/day

'Tmin_SH': Site Avg. minimum temperature for observed Sowing to Heading period in °C
'Tmax_SH': Site Avg. maximum temperature for observed Sowing to Heading period in °C
'Tavg_SH': Site Avg. mean temperature for observed Sowing to Heading period in °C
'Srad_SH': Site Avg. solar radiation for observed Sowing to Heading period in MJ/m2/day
'Pcp_SH': Site Total amount of precipitation for observed Sowing to Heading period in mm/day

'Tmin_pEH': Site Avg. minimum temperature for estimated Emergence to observed Heading period in °C
'Tmax_pEH': Site Avg. maximum temperature for estimated Emergence to observed Heading period in °C
'Tavg_pEH': Site Avg. mean temperature for estimated Emergence to observed Heading period in °C
'Srad_pEH': Site Avg. solar radiation for estimated Emergence to observed Heading period in MJ/m2/day
'Pcp_pEH': Site Total amount of precipitation for estimated Emergence to observed Heading period in mm/day

'Tmin_HM': Site Avg. minimum temperature for observed Heading to Maturity period in °C
'Tmax_HM': Site Avg. maximum temperature for observed Heading to Maturity period in °C
'Tavg_HM': Site Avg. mean temperature for observed Heading to Maturity period in °C
'Srad_HM': Site Avg. solar radiation for observed Heading to Maturity period in MJ/m2/day
'Pcp_HM': Site Total amount of precipitation for observed Heading to Maturity period in mm/day

'Tmin_SpM': Site Avg. minimum temperature for Sowing to estimated Maturity period in °C
'Tmax_SpM': Site Avg. maximum temperature for Sowing to estimated Maturity period in °C
'Tavg_SpM': Site Avg. mean temperature for Sowing to estimated Maturity period in °C
'Srad_SpM': Site Avg. solar radiation temperature for Sowing to estimated Maturity period in MJ/m2/day
'Pcp_SpM': Site Total amount of precipitation for Sowing to estimated Maturity period in mm/day

'Tmin_SpH': Site Avg. minimum temperature for Sowing to estimated Heading period in °C
'Tmax_SpH': Site Avg. maximum temperature for Sowing to estimated Heading period in °C
'Tavg_SpH': Site Avg. mean temperature for Sowing to estimated Heading period in °C
'Srad_SpH': Site Avg. solar radiation temperature for Sowing to estimated Heading period in MJ/m2/day
'Pcp_SpH': Site Total amount of precipitation for Sowing to estimated Heading period in mm/day

'Tmin_pEpH': Site Avg. minimum temperature for estimated Emergence to estimated Heading period in °C
'Tmax_pEpH': Site Avg. maximum temperature for estimated Emergence to estimated Heading period in °C
'Tavg_pEpH': Site Avg. mean temperature for estimated Emergence to estimated Heading period in °C
'Srad_pEpH': Site Avg. solar radiation for estimated Emergence to estimated Heading period in MJ/m2/day
'Pcp_pEpH': Site Total amount of precipitation for estimated Emergence to estimated Heading period in mm/day

'Tmin_HpM': Site Avg. minimum temperature for observed Heading to estimated Maturity period in °C
'Tmax_HpM': Site Avg. maximum temperature for observed Heading to estimated Maturity period in °C
'Tavg_HpM': Site Avg. mean temperature for observed Heading to estimated Maturity period in °C
'Srad_HpM': Site Avg. solar radiation for observed Heading to estimated Maturity period in MJ/m2/day
'Pcp_HpM': Site Total amount of precipitation for observed Heading to estimated Maturity period in mm/day

'Tmin_pHpM': Site Avg. minimum temperature for estimated Heading to estimated Maturity period in °C
'Tmax_pHpM': Site Avg. maximum temperature for estimated Heading to estimated Maturity period in °C
'Tavg_pHpM': Site Avg. mean temperature for estimated Heading to estimated Maturity period in °C
'Srad_pHpM': Site Avg. solar radiation for estimated Heading to estimated Maturity period in MJ/m2/day
'Pcp_pHpM': Site Total amount of precipitation for estimated Heading to estimated Maturity period in mm/day

'cGDD': Cumulative Growing Degree Days for whole cycle,
'cGDD_SE': Cumulative Growing Degree Days from Sowing to Emergence date
'cGDD_EH': Cumulative Growing Degree Days from Emergence to Heading date
'cGDD_EM': Cumulative Growing Degree Days from Emergence to Maturity date
'cGDD_SM': Cumulative Growing Degree Days from Sowing to Maturity date
'cGDD_SH': Cumulative Growing Degree Days from Sowing to Heading date
'cGDD_SpE': Cumulative Growing Degree Days from Sowing to estimated Emergence date
'cGDD_pEH': Cumulative Growing Degree Days from estimated Emergence to Heading date
'cGDD_pEpH': Cumulative Growing Degree Days from estimated Emergence to estimated Heading date
'cGDD_pEM': Cumulative Growing Degree Days from estimated Emergence to Maturity date
'cGDD_HM': Cumulative Growing Degree Days from Heading to Maturity date
'cGDD_pHM': Cumulative Growing Degree Days from estimated Heading to Maturity date
'cGDD_HpM': Cumulative Growing Degree Days from Heading to estimated Maturity date
'cGDD_pHpM': Cumulative Growing Degree Days from estimated Heading to estimated Maturity date
'cGDD_35DAS_H': Cumulative Growing Degree Days from 35 days after sowing date to Heading date,
'cGDD_35DAS_pH': Cumulative Growing Degree Days from 35 days after sowing date to estimated Heading date

'iPAR': Total light interception
'iPAR_EM': Total light interception from emergence and maturity date
'iPAR_EH': Total light interception from emergence and heading date
'iPAR_HM': Total light interception from heading and maturity date
'iPAR_pEM': Total light interception from estimated emergence and maturity date
'iPAR_pEH': Total light interception from estimated emergence and heading date
'iPAR_pHM': Total light interception from estimated heading and maturity date
'iPAR_pEpH': Total light interception from estimated emergence and estimated heading date
'iPAR_pEpM': Total light interception from estimated emergence and estimated maturity date
'iPAR_HpM': Total light interception from heading and estimated maturity date
'iPAR_pHpM': Total light interception from estimated heading and estimated maturity date
'iPAR_pEpM2': Total light interception from estimated emergence and estimated maturity date used to estimate grain yield (SimYield_pM)
'iPAR_pEpM3': Total light interception from estimated emergence and estimated maturity date used to estimate grain yield (SimYield_pHpM) 

'sumfIPAR': Sum of daily fiPAR
'sumfIPAR_EH': Sum of daily fIPAR from Emergence to Heading date
'sumfIPAR_HM': Sum of daily fiPAR from heading and maturity date 
'sumfIPAR_pHM': Sum of daily fiPAR from estimated heading and maturity date
'sumfIPAR_HpM': Sum of daily fiPAR from heading and estimated maturity date
'sumfIPAR_pHpM': Sum of daily fiPAR from estimated heading and estimated maturity date

'cGPP': Cumulative Gross Primary Production for whole cycle
'cGPP_EH': Cumulative Gross Primary Productionfrom Emergence to Heading date,
'cGPP_HM': Cumulative Gross Primary Production from Heading to Maturity date
'cGPP_pHM': Cumulative Gross Primary Production from estimated Heading to Maturity date
'cGPP_HpM': Cumulative Gross Primary Production from Heading to estimated Maturity date
'cGPP_pHpM': Cumulative Gross Primary Production from estimated Heading to estimated Maturity date
'sumGPP_EH': Sum of daily GPP from Emergence to Heading date
'sumGPP_HM': Sum of daily GPP from Heading to Maturity date
'sumGPP_pHM': Sum of daily GPP from estimated Heading to Maturity date
'sumGPP_HpM': Sum of daily GPP from Heading to estimated Maturity date
'sumGPP_pHpM': Sum of daily GPP from estimated Heading to estimated Maturity date,

'ObsYield': Observed grain yield
'SimYield': Estimated grain yield in t/ha (Observed phenology)
'SimYield_pH': Estimated grain yield in t/ha (Estimated Heading)
'SimYield_pM': Estimated grain yield in t/ha (Estimated Maturity)
'SimYield_pHpM': Estimated grain yield in t/ha (Estimated phenology)


All above variables are intermediate results used for the iPAR model to validate and calibrate the model before release to the public

--------------------------
Contact:
Urs Christoph Schulthess (U.Schulthess@cgiar.org)








