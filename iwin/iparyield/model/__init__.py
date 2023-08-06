# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import sys, os, gc
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm

from . import *
from ..data import *

#sys.append('../../iwin')
sys.path.insert(0, r"../../iwin")
import iwin

__version__ = "iPAR Yield model version 1.0.5.dev"
__author__ = "Urs Christoph schulthess, Ernesto Giron Echeverry"
__copyright__ = "Copyright (C) 2023 CIMMYT-Henan Collaborative Innovation Center"
__license__ = "Public Domain"

# --------------------------
# GLOBAL VARIABLES
# --------------------------

# Model configuration parameters
PARAMETERS = {
    "CROP_OPTIMUM_TEMPERATURE": 18, # TOpt
    "stressFactor": 1.0,
    "RUE": 3, # Radiation use efficiency (3 g/MJ)
    "DRYMATTER":0.8,
    "FACTOR_TON_HA":0.01,
    "YIELD_FACTOR": 0.8 * 0.01, #DRYMATTER * FACTOR_TON_HA,
    "TMIN_PERC_FACTOR":0.25,
    "CROP_TBASE_GDD": 0,
    "DAP": 35, # days after planting
    "D2M": 40, #43, # days Period heading to maturity
    "CONST_DAYHR_AS" : 10.8, # Day length constant
    "GDD_Required_to_EmergenceDate":180, # Thermal time or Growing Degree Days
    "NDVI_constantIPAR": 0.19, # constant for Eq. in iPAR
    "NDVI_lowerThreshold": 0.16,
    "NDVI_Threshold": 0.15, # Threshold for linear eq. for Emergence to Heading
    "NDVI_max": 0.95, # Maximum NDVI
    "NORM_iPAR_EH_BOUNDS": 0.5,
    "TDAYS_THRESHOLD_MATURITY": 42, # threshold to adjust the number of temperature days
    "DAYS_GRAIN_FILLING":40, # duration (days) of grain filling 
}

class iPARModel(object):
    global PARAMETERS
    def __init__(self, config, parameters=None):
        self.config = config
        if (parameters is None):
            self.parameters = PARAMETERS
        else:
            self.parameters = parameters
    
    __version__ = "iPAR Yield model version 1.0.1.dev"
    __author__ = "Urs Christoph schulthess, Ernesto Giron Echeverry"
    __copyright__ = "Copyright (C) 2023 CIMMYT-Henan Collaborative Innovation Center"
    __license__ = "Public Domain"

    
    def load_raw_datasets(self):
        ''' Load raw IWIN and AgERA5 datasets '''
        if (self.config is None):
            print("Configuration not valid")
            return
        
        IWIN_sites_phenology_path = os.path.join(self.config['PROJECT_PATH'], self.config['PHENO_FILE'])
        if (os.path.exists(IWIN_sites_phenology_path)):
            #converters = {
            #    'SowingDateQC': lambda x: datetime.strptime(str(x), "%Y-%m-%d"),
            #    'Heading_date': lambda x: datetime.strptime(str(x), "%Y-%m-%d"),
            #    'Maturity_date': lambda x: datetime.strptime(str(x), "%Y-%m-%d"),
            #    'Emergence_date': lambda x: datetime.strptime(str(x), "%Y-%m-%d"),
            #    #'Number': lambda x: float(x.replace(',', ''))
            #}
            #PhenoFile = pd.read_csv(IWIN_sites_phenology_path, converters=converters, index_col=False)
            PhenoFile = pd.read_csv(IWIN_sites_phenology_path, index_col=False)
            # convert sowing, heading and maturity date to DATE format
            if ('SowingDateQC' in list(PhenoFile)):
                PhenoFile['SowingDateQC']=pd.to_datetime(PhenoFile['SowingDateQC'].astype(str), format='%Y-%m-%d')
            if ('Heading_date' in list(PhenoFile)):
                PhenoFile['Heading_date']=pd.to_datetime(PhenoFile['Heading_date'].astype(str), format='%Y-%m-%d')
            if ('Maturity_date' in list(PhenoFile)):
                PhenoFile['Maturity_date']=pd.to_datetime(PhenoFile['Maturity_date'].astype(str), format='%Y-%m-%d')
            if ('Emergence_date' in list(PhenoFile)):
                PhenoFile['Emergence_date']=pd.to_datetime(PhenoFile['Emergence_date'].astype(str), format='%Y-%m-%d')
            #
        else:
            print("Error reading crop phenology file")
            return
            
        IWIN_sites_WeatherFile_path = os.path.join(self.config['PROJECT_PATH'], self.config['WEATHER_FILE'])
        if (os.path.exists(IWIN_sites_WeatherFile_path)):
            #converters = {
            #    'Date': lambda x: datetime.strptime(x, "%Y-%m-%d"),
            #}
            WeatherFile = pd.read_csv(IWIN_sites_WeatherFile_path, index_col=False)
            # convert Date to DATE format
            WeatherFile['Date'] = pd.to_datetime(WeatherFile['Date'].astype(str), format='%Y-%m-%d')
            
        else:
            print("Error reading weather file")
            return

        self.config['WeatherFile'] = WeatherFile
        self.config['PhenoFile'] = PhenoFile
        return PhenoFile
    
    
    def load_datasets(self):
        ''' Load IWIN and AgERA5 datasets in parquet format
            Reading the Parquet format is much more efficient.
            
        :return: Two dataframes for crop phenology and weather dataset respectively.
        
        '''
        if (self.config is None):
            print("Configuration not valid")
            return
        
        IWIN_sites_phenology_path = os.path.join(self.config['PROJECT_PATH'], self.config['PHENO_FILE'])
        if (os.path.exists(IWIN_sites_phenology_path)):
            PhenoFile = pd.read_parquet(IWIN_sites_phenology_path) #, engine="fastparquet")
        else:
            print("Error reading crop phenology file")
            return
            
        IWIN_sites_WeatherFile_path = os.path.join(self.config['PROJECT_PATH'], self.config['WEATHER_FILE'])
        if (os.path.exists(IWIN_sites_WeatherFile_path)):
            WeatherFile = pd.read_parquet(IWIN_sites_WeatherFile_path) #, engine="fastparquet")
        else:
            print("Error reading weather file")
            return
        
        self.config['WeatherFile'] = WeatherFile
        self.config['PhenoFile'] = PhenoFile
        return PhenoFile
    
    # -----------------------------------
    # Filter phenology dataset
    # -----------------------------------
    def filterPhenologyData(self, data=None, fld=None, value=None, selcols=None, verbose=False):
        '''
            Filter dataset by Nursery
            
            :params data: A table or DF with trial for each site
            
            :results: An array or DF of filtered sites
        '''
        if (data is None):
            data = self.config['PhenoFile']
            if (data is None):
                print("Input data not valid or empty")
                return
        if (fld is None or value is None):
            print("Field or column name is required to filter the dataset")
            return
        if (selcols is not None):
            data = data[selcols] # Filter columns
        #
        df_raw = data[data[fld]==value]
        df_raw.reset_index(drop=True, inplace=True)
        df_raw['UID'] = df_raw.index
        return df_raw
    
    # -----------------------------------
    # Data preparation
    # -----------------------------------
    def prepareData(self, data=None, params=None, selcols=None, verbose=False):
        '''
            Preprocessing dataset to the iPAR yield formats
            
            :params data: A table or DF with phenology for each site
            :params params: Paramaters used to control the phenology simulation
            :params selcols: Attibutes or columns selected to use in the simulations
            
            :results: An array of sites with info setup for run model
        '''
        if (data is None):
            data = self.config['PhenoFile']
            if (data is None):
                print("Input data not valid or empty")
                return 
        
        if (params is None):
            if (verbose is True):
                print("Using Observed phenology found in the table")
            params = {
                "estimateEmergence": False,
                "estimateHeading": False,
                "estimateMaturity": False,
            }
        
        # TODO: Generalize this process using a better standard
        if (selcols is None):
            selcols = [ 'UID', 'location', 'Occ', 'sowing', 'Emergence', 'Heading', 'Maturity', 
                        'Days_To_Heading', 'Days_To_Maturity', 'lat', 'lon', 'ObsYield' 
            ]
        
        # Filter columns
        parcels = data[selcols]
        # rename columns according to the model
        # TODO: Generalize this process using a better standard
        parcels.rename(columns={
            'Emergence':'emergence', 'Heading':'heading', 'Maturity':'maturity'
        },inplace=True)
        # Prepare dataset
        if (verbose is True):
            print("Preprocessing dataset to the iPAR yield formats...")
        sites_to_run = []
        for idx in tqdm(parcels.index):
            uid = parcels.loc[idx, 'UID']
            loc = parcels.loc[idx, 'location']
            attributes = {}
            for a in list(parcels):
                if ((a !='UID') and (a!='location')):
                    value = parcels.loc[idx, a]
                    if (isinstance(value, float) or isinstance(value, int) ):
                        #print("numeric value -> ",value)
                        attributes[a] = value
                    elif (isinstance(value, pd.Timestamp) ):
                        #print("Timestamp value -> ",value)
                        attributes[a] = str(value).split(' ')[0] if str(value)!='NaT' else 'nan'
                    elif (isinstance(value, object) ):
                        #print("object value -> ",value)
                        attributes[a] = value if str(value)!='NaT' else 'nan'
            parcel = iwin.Site(int(uid), int(loc), attributes, params)
            sites_to_run.append(parcel)
        
        return sites_to_run
    
    # ------------------------------------------------------------------
    # Data preparation version for extract phenology dates and weather
    # ------------------------------------------------------------------
    def prepareData_toExtractWeather(self, data=None, selcols=None, verbose=False):
        '''
            Preprocessing dataset to extract weather data in parallel
            
            :params data: A table or DF with phenology for each site
            :params selcols: Attibutes or columns selected to use in the simulations
            
            :results: An array of sites with info setup for run model
        '''
        if (data is None):
            print("Input data not valid or empty")
            return 
        
        # TODO: Generalize this process using a better standard
        if (selcols is None):
            selcols = [ 'UID', 'location', 'Occ', 'sowing', 'Emergence', 'Heading', 'Maturity', 
                        'Days_To_Heading', 'Days_To_Maturity', 'lat', 'lon', 'ObsYield' 
            ]
        
        # Filter columns
        parcels = data.copy() #[selcols]
        # rename columns according to the model
        # TODO: Generalize this process using a better standard
        parcels.rename(columns={
            #'Loc_no':'location', 
            'DAYS_TO_HEADING':'Days_To_Heading', 'DAYS_TO_MATURITY':'Days_To_Maturity',
            'SowingDate':'sowing', 'Emergence':'emergence', 'Heading':'heading', 'Maturity':'maturity', 'Lat':'lat'
        },inplace=True)
        # Prepare dataset
        if (verbose is True):
            print("Preprocessing dataset to extract weather data...")
        sites_to_run = []
        for idx in parcels.index:
            uid = parcels.loc[idx, 'UID']
            loc = parcels.loc[idx, 'location']
            attributes = {}
            for a in list(parcels):
                #if ((a !='UID') and (a!='location')):
                value = parcels.loc[idx, a]
                if (isinstance(value, float) or isinstance(value, int) ):
                    attributes[a] = value
                elif (isinstance(value, pd.Timestamp) ):
                    attributes[a] = str(value).split(' ')[0] if str(value)!='NaT' else 'nan'
                elif (isinstance(value, object) ):
                    attributes[a] = value if str(value)!='NaT' else 'nan'
            parcel = iwin.Site(int(uid), int(loc), attributes, {})
            sites_to_run.append(parcel)
        
        #del parcels
        _ = gc.collect()
        return sites_to_run
    
    #
    #
    # ------------------------------
    # Weather data in growth stages
    # ------------------------------
    def process_ClimateStatsforGrowthStage(self, s, weather=None, verbose=False):
        '''
            Distill the climate data down into meaningful variables.
            Extract weather data from AgERA5 dataset for each location in each growth stage

        '''
        # Get Weather for growing seasons
        if ( weather is None):
            #weather = model.config['WeatherFile']
            weather = self.config['WeatherFile']
            if (weather is None):
                print("Climate data is not valid")
                return

        try:
            sowing = None
            emergence = None
            heading = None
            maturity = None

            if ( ('sowing' in s.attributes) and (str(s.attributes['sowing'])!='nan') ):
                sowing = s.attributes['sowing']
            if ( ('emergence' in s.attributes) and (str(s.attributes['emergence'])!='nan') ):
                emergence = s.attributes['emergence']
            if ( ('heading' in s.attributes) and (str(s.attributes['heading'])!='nan') ):
                heading = s.attributes['heading']
            if ( ('maturity' in s.attributes) and (str(s.attributes['maturity'])!='nan') ):
                maturity = s.attributes['maturity']

            if ( (sowing is not None) and (emergence is not None) and (heading is not None) and (maturity is not None) ):
                # climate means for each genotype at each growth stage in each environment
                if verbose is True:
                    print("Processing climate stats...\n")
                    print(sowing, emergence, heading, maturity)

                if verbose is True:
                    print("Defining new periods")
                # Heading minus 5 days
                heading_minus_5days = str(pd.to_datetime(str(heading)) - pd.DateOffset(days=5)).split(' ')[0]
                s.attributes['heading_minus_5days'] = heading_minus_5days
                if verbose is True:
                    print("Heading minus 5 days: ", heading_minus_5days)

                # Heading plus 5 days
                heading_plus_5days = str(pd.to_datetime(str(heading)) + pd.DateOffset(days=5)).split(' ')[0]
                s.attributes['heading_plus_5days'] = heading_plus_5days
                if verbose is True:
                    print("Heading plus 5 days", heading_plus_5days)

                # Heading minus 30 days
                heading_minus_30days = str(pd.to_datetime(str(heading)) - pd.DateOffset(days=30)).split(' ')[0]
                s.attributes['heading_minus_30days'] = heading_minus_30days
                if verbose is True:
                    print("Heading minus 30 days: ", heading_minus_30days)

                # Heading plus 15 days
                heading_plus_15days = str(pd.to_datetime(str(heading)) + pd.DateOffset(days=15)).split(' ')[0]
                s.attributes['heading_plus_15days'] = heading_plus_15days
                if verbose is True:
                    print("Heading plus 15 days: ", heading_plus_15days)

                # Get weather data
                w_loc = weather[( (weather['Date']>=sowing) & (weather['Date']<=maturity) & (weather['location']==s.loc))]

                if (len(w_loc)>0):
                    # Remove those values taken as outliers
                    w_loc['TMIN'] = w_loc['TMIN'].apply(lambda x: x if x >=0.0 else 0.0)
                    w_loc['TMAX'] = w_loc['TMAX'].apply(lambda x: x if x >=0.0 else 0.0)

                    # Period 1A: Emergence to Heading minus 5 days (Vegetative) 
                    _mask_EHminus5 = ( (w_loc['Date']>=emergence) & (w_loc['Date']<=heading_minus_5days) )
                    w_EHminus5 = w_loc[_mask_EHminus5].reset_index(drop=True)
                    if ( len(w_EHminus5)>1 ):
                        # Sensitivities are presented most in mean values
                        #s.attributes['Period1A_TMin_min'] = round(w_EHminus5['TMIN'].min(), 1)
                        s.attributes['Period1A_TMin_mean'] = round(w_EHminus5['TMIN'].mean(), 1)
                        #s.attributes['Period1A_TMin_max'] = round(w_EHminus5['TMIN'].max(), 1)

                        #s.attributes['Period1A_TAVG_min'] = round(w_EHminus5['TAVG'].min(), 1)
                        s.attributes['Period1A_TAVG_mean'] = round(w_EHminus5['TAVG'].mean(), 1)
                        #s.attributes['Period1A_TAVG_max'] = round(w_EHminus5['TAVG'].max(), 1)

                        #s.attributes['Period1A_TMax_min'] = round(w_EHminus5['TMAX'].min(), 1)
                        s.attributes['Period1A_TMax_mean'] = round(w_EHminus5['TMAX'].mean(), 1)
                        #s.attributes['Period1A_TMax_max'] = round(w_EHminus5['TMAX'].max(), 1)

                        s.attributes['Period1A_Rain_total'] = round(w_EHminus5['PCP'].sum(), 1)
                        #s.attributes['Period1A_Rain_mean'] = round(w_EHminus5['PCP'].mean(), 1)

                        #s.attributes['Period1A_SolRad_min'] = round(w_EHminus5['SolRad'].min(), 1)
                        s.attributes['Period1A_SolRad_mean'] = round(w_EHminus5['SolRad'].mean(), 1)
                        #s.attributes['Period1A_SolRad_max'] = round(w_EHminus5['SolRad'].max(), 1)

                        #s.attributes['Period1A_RHUMn_min'] = round(w_EHminus5['RHUMn'].min(), 1)
                        s.attributes['Period1A_RHUMn_mean'] = round(w_EHminus5['RHUMn'].mean(), 1)
                        #s.attributes['Period1A_RHUMn_max'] = round(w_EHminus5['RHUMn'].max(), 1)

                        #s.attributes['Period1A_RHUMx_min'] = round(w_EHminus5['RHUMx'].min(), 1)
                        s.attributes['Period1A_RHUMx_mean'] = round(w_EHminus5['RHUMx'].mean(), 1)
                        #s.attributes['Period1A_RHUMx_max'] = round(w_EHminus5['RHUMx'].max(), 1)

                        #s.attributes['Period1A_WIND_min'] = round(w_EHminus5['WIND'].min(), 1)
                        s.attributes['Period1A_WIND_mean'] = round(w_EHminus5['WIND'].mean(), 1)
                        #s.attributes['Period1A_WIND_max'] = round(w_EHminus5['WIND'].max(), 1)

                    else:
                        # Period not defined
                        # Period 1A: Emergence to Heading minus 5 days (Vegetative) 
                        #s.attributes['Period1A_TMin_min'] = 'nan'
                        s.attributes['Period1A_TMin_mean'] = 'nan'
                        #s.attributes['Period1A_TMin_max'] = 'nan'
                        #s.attributes['Period1A_TAVG_min'] = 'nan'
                        s.attributes['Period1A_TAVG_mean'] = 'nan'
                        #s.attributes['Period1A_TAVG_max'] = 'nan'
                        #s.attributes['Period1A_TMax_min'] = 'nan'
                        s.attributes['Period1A_TMax_mean'] = 'nan'
                        #s.attributes['Period1A_TMax_max'] = 'nan'
                        s.attributes['Period1A_Rain_total'] = 'nan'
                        #s.attributes['Period1A_Rain_mean'] = 'nan'
                        #s.attributes['Period1A_SolRad_min'] = 'nan'
                        s.attributes['Period1A_SolRad_mean'] = 'nan'
                        #s.attributes['Period1A_SolRad_max'] = 'nan'
                        #s.attributes['Period1A_RHUMn_min'] = 'nan'
                        s.attributes['Period1A_RHUMn_mean'] = 'nan'
                        #s.attributes['Period1A_RHUMn_max'] = 'nan'
                        #s.attributes['Period1A_RHUMx_min'] = 'nan'
                        s.attributes['Period1A_RHUMx_mean'] = 'nan'
                        #s.attributes['Period1A_RHUMx_max'] = 'nan'
                        #s.attributes['Period1A_WIND_min'] = 'nan'
                        s.attributes['Period1A_WIND_mean'] = 'nan'
                        #s.attributes['Period1A_WIND_max'] = 'nan'


                    # Period 2A: Heading minus 5 days to heading plus 5 days (Heading)
                    _mask_HH = ( (w_loc['Date']>=heading_minus_5days) & (w_loc['Date']<=heading_plus_5days) )
                    w_HH = w_loc[_mask_HH].reset_index(drop=True)
                    if ( len(w_HH)>1 ):
                        #s.attributes['Period2A_TMin_min'] = round(w_HH['TMIN'].min(), 1)
                        s.attributes['Period2A_TMin_mean'] = round(w_HH['TMIN'].mean(), 1)
                        #s.attributes['Period2A_TMin_max'] = round(w_HH['TMIN'].max(), 1)
                        #s.attributes['Period2A_TAVG_min'] = round(w_HH['TAVG'].min(), 1)
                        s.attributes['Period2A_TAVG_mean'] = round(w_HH['TAVG'].mean(), 1)
                        #s.attributes['Period2A_TAVG_max'] = round(w_HH['TAVG'].max(), 1)
                        #s.attributes['Period2A_TMax_min'] = round(w_HH['TMAX'].min(), 1)
                        s.attributes['Period2A_TMax_mean'] = round(w_HH['TMAX'].mean(), 1)
                        #s.attributes['Period2A_TMax_max'] = round(w_HH['TMAX'].max(), 1)
                        s.attributes['Period2A_Rain_total'] = round(w_HH['PCP'].sum(), 1)
                        #s.attributes['Period2A_Rain_mean'] = round(w_HH['PCP'].mean(), 1)
                        #s.attributes['Period2A_SolRad_min'] = round(w_HH['SolRad'].min(), 1)
                        s.attributes['Period2A_SolRad_mean'] = round(w_HH['SolRad'].mean(), 1)
                        #s.attributes['Period2A_SolRad_max'] = round(w_HH['SolRad'].max(), 1)
                        #s.attributes['Period2A_RHUMn_min'] = round(w_HH['RHUMn'].min(), 1)
                        s.attributes['Period2A_RHUMn_mean'] = round(w_HH['RHUMn'].mean(), 1)
                        #s.attributes['Period2A_RHUMn_max'] = round(w_HH['RHUMn'].max(), 1)
                        #s.attributes['Period2A_RHUMx_min'] = round(w_HH['RHUMx'].min(), 1)
                        s.attributes['Period2A_RHUMx_mean'] = round(w_HH['RHUMx'].mean(), 1)
                        #s.attributes['Period2A_RHUMx_max'] = round(w_HH['RHUMx'].max(), 1)
                        #s.attributes['Period2A_WIND_min'] = round(w_HH['WIND'].min(), 1)
                        s.attributes['Period2A_WIND_mean'] = round(w_HH['WIND'].mean(), 1)
                        #s.attributes['Period2A_WIND_max'] = round(w_HH['WIND'].max(), 1)
                    else:
                        #s.attributes['Period2A_TMin_min'] = 'nan'
                        s.attributes['Period2A_TMin_mean'] = 'nan'
                        #s.attributes['Period2A_TMin_max'] = 'nan'
                        #s.attributes['Period2A_TAVG_min'] = 'nan'
                        s.attributes['Period2A_TAVG_mean'] = 'nan'
                        #s.attributes['Period2A_TAVG_max'] = 'nan'
                        #s.attributes['Period2A_TMax_min'] = 'nan'
                        s.attributes['Period2A_TMax_mean'] = 'nan'
                        #s.attributes['Period2A_TMax_max'] = 'nan'
                        s.attributes['Period2A_Rain_total'] = 'nan'
                        #s.attributes['Period2A_Rain_mean'] = 'nan'
                        #s.attributes['Period2A_SolRad_min'] = 'nan'
                        s.attributes['Period2A_SolRad_mean'] = 'nan'
                        #s.attributes['Period2A_SolRad_max'] = 'nan'
                        #s.attributes['Period2A_RHUMn_min'] = 'nan'
                        s.attributes['Period2A_RHUMn_mean'] = 'nan'
                        #s.attributes['Period2A_RHUMn_max'] = 'nan'
                        #s.attributes['Period2A_RHUMx_min'] = 'nan'
                        s.attributes['Period2A_RHUMx_mean'] = 'nan'
                        #s.attributes['Period2A_RHUMx_max'] = 'nan'
                        #s.attributes['Period2A_WIND_min'] = 'nan'
                        s.attributes['Period2A_WIND_mean'] = 'nan'
                        #s.attributes['Period2A_WIND_max'] = 'nan'

                    # Period 3A: Heading plus 5 days to maturity (Grain filling)
                    _mask_Hplus5M = ( (w_loc['Date']>=heading_plus_5days) & (w_loc['Date']<=maturity) )
                    w_Hplus5M = w_loc[_mask_Hplus5M].reset_index(drop=True)
                    if ( len(w_Hplus5M)>1 ):
                        #s.attributes['Period3A_TMin_min'] = round(w_Hplus5M['TMIN'].min(), 1)
                        s.attributes['Period3A_TMin_mean'] = round(w_Hplus5M['TMIN'].mean(), 1)
                        #s.attributes['Period3A_TMin_max'] = round(w_Hplus5M['TMIN'].max(), 1)
                        #s.attributes['Period3A_TAVG_min'] = round(w_Hplus5M['TAVG'].min(), 1)
                        s.attributes['Period3A_TAVG_mean'] = round(w_Hplus5M['TAVG'].mean(), 1)
                        #s.attributes['Period3A_TAVG_max'] = round(w_Hplus5M['TAVG'].max(), 1)
                        #s.attributes['Period3A_TMax_min'] = round(w_Hplus5M['TMAX'].min(), 1)
                        s.attributes['Period3A_TMax_mean'] = round(w_Hplus5M['TMAX'].mean(), 1)
                        #s.attributes['Period3A_TMax_max'] = round(w_Hplus5M['TMAX'].max(), 1)
                        s.attributes['Period3A_Rain_total'] = round(w_Hplus5M['PCP'].sum(), 1)
                        #s.attributes['Period3A_Rain_mean'] = round(w_Hplus5M['PCP'].mean(), 1)
                        #s.attributes['Period3A_SolRad_min'] = round(w_Hplus5M['SolRad'].min(), 1)
                        s.attributes['Period3A_SolRad_mean'] = round(w_Hplus5M['SolRad'].mean(), 1)
                        #s.attributes['Period3A_SolRad_max'] = round(w_Hplus5M['SolRad'].max(), 1)
                        #s.attributes['Period3A_RHUMn_min'] = round(w_Hplus5M['RHUMn'].min(), 1)
                        s.attributes['Period3A_RHUMn_mean'] = round(w_Hplus5M['RHUMn'].mean(), 1)
                        #s.attributes['Period3A_RHUMn_max'] = round(w_Hplus5M['RHUMn'].max(), 1)
                        #s.attributes['Period3A_RHUMx_min'] = round(w_Hplus5M['RHUMx'].min(), 1)
                        s.attributes['Period3A_RHUMx_mean'] = round(w_Hplus5M['RHUMx'].mean(), 1)
                        #s.attributes['Period3A_RHUMx_max'] = round(w_Hplus5M['RHUMx'].max(), 1)
                        #s.attributes['Period3A_WIND_min'] = round(w_Hplus5M['WIND'].min(), 1)
                        s.attributes['Period3A_WIND_mean'] = round(w_Hplus5M['WIND'].mean(), 1)
                        #s.attributes['Period3A_WIND_max'] = round(w_Hplus5M['WIND'].max(), 1)
                    else:
                        #s.attributes['Period3A_TMin_min'] = 'nan'
                        s.attributes['Period3A_TMin_mean'] = 'nan'
                        #s.attributes['Period3A_TMin_max'] = 'nan'
                        #s.attributes['Period3A_TAVG_min'] = 'nan'
                        s.attributes['Period3A_TAVG_mean'] = 'nan'
                        #s.attributes['Period3A_TAVG_max'] = 'nan'
                        #s.attributes['Period3A_TMax_min'] = 'nan'
                        s.attributes['Period3A_TMax_mean'] = 'nan'
                        #s.attributes['Period3A_TMax_max'] = 'nan'
                        s.attributes['Period3A_Rain_total'] = 'nan'
                        #s.attributes['Period3A_Rain_mean'] = 'nan'
                        #s.attributes['Period3A_SolRad_min'] = 'nan'
                        s.attributes['Period3A_SolRad_mean'] = 'nan'
                        #s.attributes['Period3A_SolRad_max'] = 'nan'
                        #s.attributes['Period3A_RHUMn_min'] = 'nan'
                        s.attributes['Period3A_RHUMn_mean'] = 'nan'
                        #s.attributes['Period3A_RHUMn_max'] = 'nan'
                        #s.attributes['Period3A_RHUMx_min'] = 'nan'
                        s.attributes['Period3A_RHUMx_mean'] = 'nan'
                        #s.attributes['Period3A_RHUMx_max'] = 'nan'
                        #s.attributes['Period3A_WIND_min'] = 'nan'
                        s.attributes['Period3A_WIND_mean'] = 'nan'
                        #s.attributes['Period3A_WIND_max'] = 'nan'

                    # Period 1B: Heading minus 30 days to Heading
                    _mask_Hminus30H = ( (w_loc['Date']>=heading_minus_30days) & (w_loc['Date']<=heading) )
                    w_Hminus30H = w_loc[_mask_Hminus30H].reset_index(drop=True)
                    if ( len(w_Hminus30H)>1 ):
                        #s.attributes['Period1B_TMin_min'] = round(w_Hminus30H['TMIN'].min(), 1)
                        s.attributes['Period1B_TMin_mean'] = round(w_Hminus30H['TMIN'].mean(), 1)
                        #s.attributes['Period1B_TMin_max'] = round(w_Hminus30H['TMIN'].max(), 1)
                        #s.attributes['Period1B_TAVG_min'] = round(w_Hminus30H['TAVG'].min(), 1)
                        s.attributes['Period1B_TAVG_mean'] = round(w_Hminus30H['TAVG'].mean(), 1)
                        #s.attributes['Period1B_TAVG_max'] = round(w_Hminus30H['TAVG'].max(), 1)
                        #s.attributes['Period1B_TMax_min'] = round(w_Hminus30H['TMAX'].min(), 1)
                        s.attributes['Period1B_TMax_mean'] = round(w_Hminus30H['TMAX'].mean(), 1)
                        #s.attributes['Period1B_TMax_max'] = round(w_Hminus30H['TMAX'].max(), 1)
                        s.attributes['Period1B_Rain_total'] = round(w_Hminus30H['PCP'].sum(), 1)
                        #s.attributes['Period1B_Rain_mean'] = round(w_Hminus30H['PCP'].mean(), 1)
                        #s.attributes['Period1B_SolRad_min'] = round(w_Hminus30H['SolRad'].min(), 1)
                        s.attributes['Period1B_SolRad_mean'] = round(w_Hminus30H['SolRad'].mean(), 1)
                        #s.attributes['Period1B_SolRad_max'] = round(w_Hminus30H['SolRad'].max(), 1)
                        #s.attributes['Period1B_RHUMn_min'] = round(w_Hminus30H['RHUMn'].min(), 1)
                        s.attributes['Period1B_RHUMn_mean'] = round(w_Hminus30H['RHUMn'].mean(), 1)
                        #s.attributes['Period1B_RHUMn_max'] = round(w_Hminus30H['RHUMn'].max(), 1)
                        #s.attributes['Period1B_RHUMx_min'] = round(w_Hminus30H['RHUMx'].min(), 1)
                        s.attributes['Period1B_RHUMx_mean'] = round(w_Hminus30H['RHUMx'].mean(), 1)
                        #s.attributes['Period1B_RHUMx_max'] = round(w_Hminus30H['RHUMx'].max(), 1)
                        #s.attributes['Period1B_WIND_min'] = round(w_Hminus30H['WIND'].min(), 1)
                        s.attributes['Period1B_WIND_mean'] = round(w_Hminus30H['WIND'].mean(), 1)
                        #s.attributes['Period1B_WIND_max'] = round(w_Hminus30H['WIND'].max(), 1)
                    else:
                        #s.attributes['Period1B_TMin_min'] = 'nan'
                        s.attributes['Period1B_TMin_mean'] = 'nan'
                        #s.attributes['Period1B_TMin_max'] = 'nan'
                        #s.attributes['Period1B_TAVG_min'] = 'nan'
                        s.attributes['Period1B_TAVG_mean'] = 'nan'
                        #s.attributes['Period1B_TAVG_max'] = 'nan'
                        #s.attributes['Period1B_TMax_min'] = 'nan'
                        s.attributes['Period1B_TMax_mean'] = 'nan'
                        #s.attributes['Period1B_TMax_max'] = 'nan'
                        s.attributes['Period1B_Rain_total'] = 'nan'
                        #s.attributes['Period1B_Rain_mean'] = 'nan'
                        #s.attributes['Period1B_SolRad_min'] = 'nan'
                        s.attributes['Period1B_SolRad_mean'] = 'nan'
                        #s.attributes['Period1B_SolRad_max'] = 'nan'
                        #s.attributes['Period1B_RHUMn_min'] = 'nan'
                        s.attributes['Period1B_RHUMn_mean'] = 'nan'
                        #s.attributes['Period1B_RHUMn_max'] = 'nan'
                        #s.attributes['Period1B_RHUMx_min'] = 'nan'
                        s.attributes['Period1B_RHUMx_mean'] = 'nan'
                        #s.attributes['Period1B_RHUMx_max'] = 'nan'
                        #s.attributes['Period1B_WIND_min'] = 'nan'
                        s.attributes['Period1B_WIND_mean'] = 'nan'
                        #s.attributes['Period1B_WIND_max'] = 'nan'

                    # Period 2B: Heading to Heading plus 15 days
                    _mask_HHplus15 = ( (w_loc['Date']>=heading) & (w_loc['Date']<=heading_plus_15days) )
                    w_HHplus15 = w_loc[_mask_HHplus15].reset_index(drop=True)
                    if ( len(w_HHplus15)>1 ):
                        #s.attributes['Period2B_TMin_min'] = round(w_HHplus15['TMIN'].min(), 1)
                        s.attributes['Period2B_TMin_mean'] = round(w_HHplus15['TMIN'].mean(), 1)
                        #s.attributes['Period2B_TMin_max'] = round(w_HHplus15['TMIN'].max(), 1)
                        #s.attributes['Period2B_TAVG_min'] = round(w_HHplus15['TAVG'].min(), 1)
                        s.attributes['Period2B_TAVG_mean'] = round(w_HHplus15['TAVG'].mean(), 1)
                        #s.attributes['Period2B_TAVG_max'] = round(w_HHplus15['TAVG'].max(), 1)
                        #s.attributes['Period2B_TMax_min'] = round(w_HHplus15['TMAX'].min(), 1)
                        s.attributes['Period2B_TMax_mean'] = round(w_HHplus15['TMAX'].mean(), 1)
                        #s.attributes['Period2B_TMax_max'] = round(w_HHplus15['TMAX'].max(), 1)
                        s.attributes['Period2B_Rain_total'] = round(w_HHplus15['PCP'].sum(), 1)
                        #s.attributes['Period2B_Rain_mean'] = round(w_HHplus15['PCP'].mean(), 1)
                        #s.attributes['Period2B_SolRad_min'] = round(w_HHplus15['SolRad'].min(), 1)
                        s.attributes['Period2B_SolRad_mean'] = round(w_HHplus15['SolRad'].mean(), 1)
                        #s.attributes['Period2B_SolRad_max'] = round(w_HHplus15['SolRad'].max(), 1)
                        #s.attributes['Period2B_RHUMn_min'] = round(w_HHplus15['RHUMn'].min(), 1)
                        s.attributes['Period2B_RHUMn_mean'] = round(w_HHplus15['RHUMn'].mean(), 1)
                        #s.attributes['Period2B_RHUMn_max'] = round(w_HHplus15['RHUMn'].max(), 1)
                        #s.attributes['Period2B_RHUMx_min'] = round(w_HHplus15['RHUMx'].min(), 1)
                        s.attributes['Period2B_RHUMx_mean'] = round(w_HHplus15['RHUMx'].mean(), 1)
                        #s.attributes['Period2B_RHUMx_max'] = round(w_HHplus15['RHUMx'].max(), 1)
                        #s.attributes['Period2B_WIND_min'] = round(w_HHplus15['WIND'].min(), 1)
                        s.attributes['Period2B_WIND_mean'] = round(w_HHplus15['WIND'].mean(), 1)
                        #s.attributes['Period2B_WIND_max'] = round(w_HHplus15['WIND'].max(), 1)
                    else:
                        #s.attributes['Period2B_TMin_min'] = 'nan'
                        s.attributes['Period2B_TMin_mean'] = 'nan'
                        #s.attributes['Period2B_TMin_max'] = 'nan'
                        #s.attributes['Period2B_TAVG_min'] = 'nan'
                        s.attributes['Period2B_TAVG_mean'] = 'nan'
                        #s.attributes['Period2B_TAVG_max'] = 'nan'
                        #s.attributes['Period2B_TMax_min'] = 'nan'
                        s.attributes['Period2B_TMax_mean'] = 'nan'
                        #s.attributes['Period2B_TMax_max'] = 'nan'
                        s.attributes['Period2B_Rain_total'] = 'nan'
                        #s.attributes['Period2B_Rain_mean'] = 'nan'
                        #s.attributes['Period2B_SolRad_min'] = 'nan'
                        s.attributes['Period2B_SolRad_mean'] = 'nan'
                        #s.attributes['Period2B_SolRad_max'] = 'nan'
                        #s.attributes['Period2B_RHUMn_min'] = 'nan'
                        s.attributes['Period2B_RHUMn_mean'] = 'nan'
                        #s.attributes['Period2B_RHUMn_max'] = 'nan'
                        #s.attributes['Period2B_RHUMx_min'] = 'nan'
                        s.attributes['Period2B_RHUMx_mean'] = 'nan'
                        #s.attributes['Period2B_RHUMx_max'] = 'nan'
                        #s.attributes['Period2B_WIND_min'] = 'nan'
                        s.attributes['Period2B_WIND_mean'] = 'nan'
                        #s.attributes['Period2B_WIND_max'] = 'nan'

                    # Period 3B: Heading plus 15 days until maturity
                    _mask_Hplus15M = ( (w_loc['Date']>=heading_plus_15days) & (w_loc['Date']<=maturity) )
                    w_Hplus15M = w_loc[_mask_Hplus15M].reset_index(drop=True)
                    if ( len(w_Hplus15M)>1 ):
                        #s.attributes['Period3B_TMin_min'] = round(w_Hplus15M['TMIN'].min(), 1)
                        s.attributes['Period3B_TMin_mean'] = round(w_Hplus15M['TMIN'].mean(), 1)
                        #s.attributes['Period3B_TMin_max'] = round(w_Hplus15M['TMIN'].max(), 1)
                        #s.attributes['Period3B_TAVG_min'] = round(w_Hplus15M['TAVG'].min(), 1)
                        s.attributes['Period3B_TAVG_mean'] = round(w_Hplus15M['TAVG'].mean(), 1)
                        #s.attributes['Period3B_TAVG_max'] = round(w_Hplus15M['TAVG'].max(), 1)
                        #s.attributes['Period3B_TMax_min'] = round(w_Hplus15M['TMAX'].min(), 1)
                        s.attributes['Period3B_TMax_mean'] = round(w_Hplus15M['TMAX'].mean(), 1)
                        #s.attributes['Period3B_TMax_max'] = round(w_Hplus15M['TMAX'].max(), 1)
                        s.attributes['Period3B_Rain_total'] = round(w_Hplus15M['PCP'].sum(), 1)
                        #s.attributes['Period3B_Rain_mean'] = round(w_Hplus15M['PCP'].mean(), 1)
                        #s.attributes['Period3B_SolRad_min'] = round(w_Hplus15M['SolRad'].min(), 1)
                        s.attributes['Period3B_SolRad_mean'] = round(w_Hplus15M['SolRad'].mean(), 1)
                        #s.attributes['Period3B_SolRad_max'] = round(w_Hplus15M['SolRad'].max(), 1)
                        #s.attributes['Period3B_RHUMn_min'] = round(w_Hplus15M['RHUMn'].min(), 1)
                        s.attributes['Period3B_RHUMn_mean'] = round(w_Hplus15M['RHUMn'].mean(), 1)
                        #s.attributes['Period3B_RHUMn_max'] = round(w_Hplus15M['RHUMn'].max(), 1)
                        #s.attributes['Period3B_RHUMx_min'] = round(w_Hplus15M['RHUMx'].min(), 1)
                        s.attributes['Period3B_RHUMx_mean'] = round(w_Hplus15M['RHUMx'].mean(), 1)
                        #s.attributes['Period3B_RHUMx_max'] = round(w_Hplus15M['RHUMx'].max(), 1)
                        #s.attributes['Period3B_WIND_min'] = round(w_Hplus15M['WIND'].min(), 1)
                        s.attributes['Period3B_WIND_mean'] = round(w_Hplus15M['WIND'].mean(), 1)
                        #s.attributes['Period3B_WIND_max'] = round(w_Hplus15M['WIND'].max(), 1)
                    else:
                        #s.attributes['Period3B_TMin_min'] = 'nan'
                        s.attributes['Period3B_TMin_mean'] = 'nan'
                        #s.attributes['Period3B_TMin_max'] = 'nan'
                        #s.attributes['Period3B_TAVG_min'] = 'nan'
                        s.attributes['Period3B_TAVG_mean'] = 'nan'
                        #s.attributes['Period3B_TAVG_max'] = 'nan'
                        #s.attributes['Period3B_TMax_min'] = 'nan'
                        s.attributes['Period3B_TMax_mean'] = 'nan'
                        #s.attributes['Period3B_TMax_max'] = 'nan'
                        s.attributes['Period3B_Rain_total'] = 'nan'
                        #s.attributes['Period3B_Rain_mean'] = 'nan'
                        #s.attributes['Period3B_SolRad_min'] = 'nan'
                        s.attributes['Period3B_SolRad_mean'] = 'nan'
                        #s.attributes['Period3B_SolRad_max'] = 'nan'
                        #s.attributes['Period3B_RHUMn_min'] = 'nan'
                        s.attributes['Period3B_RHUMn_mean'] = 'nan'
                        #s.attributes['Period3B_RHUMn_max'] = 'nan'
                        #s.attributes['Period3B_RHUMx_min'] = 'nan'
                        s.attributes['Period3B_RHUMx_mean'] = 'nan'
                        #s.attributes['Period3B_RHUMx_max'] = 'nan'
                        #s.attributes['Period3B_WIND_min'] = 'nan'
                        s.attributes['Period3B_WIND_mean'] = 'nan'
                        #s.attributes['Period3B_WIND_max'] = 'nan'

                    # Season: sowing to maturity
                    _mask_Season = ( (w_loc['Date']>=sowing) & (w_loc['Date']<=maturity) )
                    w_Season = w_loc[_mask_Season].reset_index(drop=True)
                    if ( len(w_Season)>1 ):
                        #s.attributes['Season_TMin_min'] = round(w_Season['TMIN'].min(), 1)
                        s.attributes['Season_TMin_mean'] = round(w_Season['TMIN'].mean(), 1)
                        #s.attributes['Season_TMin_max'] = round(w_Season['TMIN'].max(), 1)
                        #s.attributes['Season_TAVG_min'] = round(w_Season['TAVG'].min(), 1)
                        s.attributes['Season_TAVG_mean'] = round(w_Season['TAVG'].mean(), 1)
                        #s.attributes['Season_TAVG_max'] = round(w_Season['TAVG'].max(), 1)
                        #s.attributes['Season_TMax_min'] = round(w_Season['TMAX'].min(), 1)
                        s.attributes['Season_TMax_mean'] = round(w_Season['TMAX'].mean(), 1)
                        #s.attributes['Season_TMax_max'] = round(w_Season['TMAX'].max(), 1)
                        s.attributes['Season_Rain_total'] = round(w_Season['PCP'].sum(), 1)
                        #s.attributes['Season_Rain_mean'] = round(w_Season['PCP'].mean(), 1)
                        #s.attributes['Season_SolRad_min'] = round(w_Season['SolRad'].min(), 1)
                        s.attributes['Season_SolRad_mean'] = round(w_Season['SolRad'].mean(), 1)
                        #s.attributes['Season_SolRad_max'] = round(w_Season['SolRad'].max(), 1)
                        #s.attributes['Season_RHUMn_min'] = round(w_Season['RHUMn'].min(), 1)
                        s.attributes['Season_RHUMn_mean'] = round(w_Season['RHUMn'].mean(), 1)
                        #s.attributes['Season_RHUMn_max'] = round(w_Season['RHUMn'].max(), 1)
                        #s.attributes['Season_RHUMx_min'] = round(w_Season['RHUMx'].min(), 1)
                        s.attributes['Season_RHUMx_mean'] = round(w_Season['RHUMx'].mean(), 1)
                        #s.attributes['Season_RHUMx_max'] = round(w_Season['RHUMx'].max(), 1)
                        #s.attributes['Season_WIND_min'] = round(w_Season['WIND'].min(), 1)
                        s.attributes['Season_WIND_mean'] = round(w_Season['WIND'].mean(), 1)
                        #s.attributes['Season_WIND_max'] = round(w_Season['WIND'].max(), 1)

                    else:
                        #s.attributes['Season_TMin_min'] = 'nan'
                        s.attributes['Season_TMin_mean'] = 'nan'
                        #s.attributes['Season_TMin_max'] = 'nan'
                        #s.attributes['Season_TAVG_min'] = 'nan'
                        s.attributes['Season_TAVG_mean'] = 'nan'
                        #s.attributes['Season_TAVG_max'] = 'nan'
                        #s.attributes['Season_TMax_min'] = 'nan'
                        s.attributes['Season_TMax_mean'] = 'nan'
                        #s.attributes['Season_TMax_max'] = 'nan'
                        s.attributes['Season_Rain_total'] = 'nan'
                        #s.attributes['Season_Rain_mean'] = 'nan'
                        #s.attributes['Season_SolRad_min'] = 'nan'
                        s.attributes['Season_SolRad_mean'] = 'nan'
                        #s.attributes['Season_SolRad_max'] = 'nan'
                        #s.attributes['Season_RHUMn_min'] = 'nan'
                        s.attributes['Season_RHUMn_mean'] = 'nan'
                        #s.attributes['Season_RHUMn_max'] = 'nan'
                        #s.attributes['Season_RHUMx_min'] = 'nan'
                        s.attributes['Season_RHUMx_mean'] = 'nan'
                        #s.attributes['Season_RHUMx_max'] = 'nan'
                        #s.attributes['Season_WIND_min'] = 'nan'
                        s.attributes['Season_WIND_mean'] = 'nan'
                        #s.attributes['Season_WIND_max'] = 'nan'

            else:
                #s['errors'].append({"uid": s.uid, "loc": s.loc, "error": "Incompleted phenology. Growth stages not defined" })
                if verbose is True:
                    print("Problem in site {} - {}. Incompleted phenology, growth stages not defined. Error: {}".format(s.uid, s.loc, err))

        except Exception as err:
            if verbose is True:
                print("Problem in site {} - {}. Error: {}".format(s.uid, s.loc, err))
            #s['errors'].append({"uid": s.uid, "loc": s.loc, "error": "Distilling the climate data down into meaningful variables. Error: {}".format(err)})

        return s
    
    
    # **************************************************
    # Correction - Climate Stats for Growth Periods
    # **************************************************
    def process_ClimateStatsforGrowthPeriod_v3(self, s, weather=None, 
                                            climatevars=['TMIN', 'TAVG', 'TMAX', 'PCP', 'SolRad','RHUMn','RHUMx', 
                                                         'WIND', 'TDay','PTQ', 'iPAR_PRFT'], 
                                            stats=['min','mean','max', 'sd'], verbose=False):
        '''
            Distill the climate data down into meaningful variables.
            Extract weather data from AgERA5 dataset for each location in each growth stage

        '''
        # Get Weather for growing seasons
        if ( weather is None):
            #weather = model.config['WeatherFile']
            weather = self.config['WeatherFile']
            if (weather is None):
                print("Climate data is not valid")
                return

        try:
            sowing = None
            emergence = None
            heading = None
            #anthesis = None
            maturity = None

            if ( ('sowing' in s.attributes) and (str(s.attributes['sowing'])!='nan') ):
                sowing = s.attributes['sowing']
            if ( ('emergence' in s.attributes) and (str(s.attributes['emergence'])!='nan') ):
                emergence = s.attributes['emergence']
            if ( ('heading' in s.attributes) and (str(s.attributes['heading'])!='nan') ):
                heading = s.attributes['heading']
            #if ( ('anthesis' in s.attributes) and (str(s.attributes['anthesis'])!='nan') ):
            #    anthesis = s.attributes['anthesis']
            if ( ('maturity' in s.attributes) and (str(s.attributes['maturity'])!='nan') ):
                maturity = s.attributes['maturity']

            if ( (sowing is not None) and (emergence is not None) and (heading is not None) and (maturity is not None) ):
                # climate means for each genotype at each growth stage in each environment
                if verbose is True:
                    print("Processing climate stats...\n")
                    #print(sowing, emergence, heading, anthesis, maturity)
                    print(sowing, emergence, heading, maturity)

                if verbose is True:
                    print("Defining new periods")

                # ------------------------------------------
                # Average weather conditions during 3 periods
                # ------------------------------------------
                #a.	Sowing to heading (S to H)
                #b.	Heading to beginning of grain filling (H to H+15d)
                #c.	Grain filling (H+15d to maturity)

                # Sowing to heading
                sowing_to_heading_days = str(pd.to_datetime(str(heading)) - pd.to_datetime(str(sowing))).split(' ')[0]
                s.attributes['sowing_to_heading_days'] = sowing_to_heading_days
                if verbose is True:
                    print("Sowing to heading days: ", sowing_to_heading_days)

                # Heading to beginning of grain filling (H to H+15d)
                # Heading plus 15 days
                heading_plus_15days = str(pd.to_datetime(str(heading)) + pd.DateOffset(days=15)).split(' ')[0]
                s.attributes['grainfilling'] = heading_plus_15days
                if verbose is True:
                    print("Heading plus 15 days: ", heading_plus_15days)

                # Grain filling (H+15d to maturity)
                grainfilling_to_maturity_days = str(pd.to_datetime(str(maturity)) - (pd.to_datetime(str(heading)) + pd.DateOffset(days=15)) ).split(' ')[0]
                s.attributes['grainfilling_to_maturity_days'] = grainfilling_to_maturity_days
                if verbose is True:
                    print("Grain filling to maturity days: ", grainfilling_to_maturity_days)


                def getStatsbyPeriod(s, w_loc, start_date, end_date, name='Period', 
                                     climatevars=['TMIN', 'TAVG', 'TMAX', 'PCP', 'SolRad','RHUMn','RHUMx', 
                                                  'WIND', 'TDay','PTQ', 'iPAR_PRFT'], 
                                     stats=['min','mean','max', 'sd']):
                    ''' Estimate the selected weather statistics given a period '''

                    _mask = ( (w_loc['Date']>=start_date) & (w_loc['Date']<=end_date) )
                    w_for_stats = w_loc[_mask].reset_index(drop=True)
                    if ( len(w_for_stats)>1 ):
                        try:
                            for vr in climatevars:
                                if ((vr=='PTQ') and (vr in w_loc.columns)):
                                    ndec = 3
                                else:
                                    ndec = 2
                                #w_for_stats = w_for_stats.dropna(subset=[vr])
                                if ((vr=='PCP' or vr=='Rain') and (vr in w_loc.columns)):
                                    pcp = np.nansum(w_for_stats[vr])
                                    s.attributes[f'{name}_{vr}_total'] = round(pcp, ndec) if pcp is not None else 'nan'
                                elif ((vr=='iPAR_PRFT') and (vr in w_loc.columns)):
                                    ipar = np.nansum(w_for_stats[vr])
                                    s.attributes[f'{name}_SumiPART'] = round(ipar, ndec) if ipar is not None else 'nan'
                                elif ((vr=='PTQ') and (vr in w_loc.columns)):
                                    ptq_avg = np.nanmean(w_for_stats[vr])
                                    ptq_sum = np.nansum(w_for_stats[vr])
                                    s.attributes[f'{name}_{vr}_mean'] = round(ptq_avg, ndec) if ptq_avg is not None else 'nan'
                                    s.attributes[f'{name}_SumPTQ'] = round(ptq_sum, ndec) if ptq_sum is not None else 'nan'
                                else:
                                    try:

                                        if (('min' in stats) and (vr in w_loc.columns)):
                                            mn = np.nanmin(w_for_stats[vr])
                                            s.attributes[f'{name}_{vr}_min'] = round(mn,ndec) if mn is not None else 'nan'
                                        if (('mean' in stats) and (vr in w_loc.columns)):
                                            avg = np.nanmean(w_for_stats[vr])
                                            s.attributes[f'{name}_{vr}_mean'] = round(avg,ndec) if avg is not None else 'nan'
                                        if (('max' in stats) and (vr in w_loc.columns)):
                                            mx = np.nanmax(w_for_stats[vr])
                                            s.attributes[f'{name}_{vr}_max'] = round(mx, ndec) if mx is not None else 'nan'
                                        if (('sd' in stats) and (vr in w_loc.columns)):
                                            sd = np.nanstd(w_for_stats[vr])
                                            s.attributes[f'{name}_{vr}_sd'] = round(sd, ndec) if sd is not None else 'nan'
                                    except Exception as err:
                                        print("ERROR calculating stats: {} - {} - {}".format(s.uid, s.loc, err))
                                #
                                # Addtional stats
                                if ((vr=='TMIN') and (vr in w_loc.columns)):
                                    try:
                                        # Number of days with TMin < 8ºC
                                        s.attributes[f'{name}_{vr}_lt8C_ndays'] = len(w_for_stats[w_for_stats[vr] < 8])
                                        # Number of days with TMin < 11ºC
                                        s.attributes[f'{name}_{vr}_lt11C_ndays'] = len(w_for_stats[w_for_stats[vr] < 11])
                                        # Number of days with TMin > 25ºC
                                        s.attributes[f'{name}_{vr}_gt25C_ndays'] = len(w_for_stats[w_for_stats[vr] > 25])
                                    except Exception as err:
                                        print("Error calculating Number of days with TMin: {} - {} - {}".format(s.uid, s.loc, err))
                                if ((vr=='TMAX') and (vr in w_loc.columns)):
                                    try:
                                        # Number of days with TMax > 32ºC
                                        s.attributes[f'{name}_{vr}_gt32C_ndays'] = len(w_for_stats[w_for_stats[vr] > 32])
                                    except Exception as err:
                                        print("Error calculating Number of days with TMax: {} - {} - {}".format(s.uid, s.loc, err))
                        except Exception as err:
                            print("ERROR: {} - {} - {}".format(s.uid, s.loc, err))
                    else:
                        #print("ERROR!!!!!",len(w_for_stats), start_date, end_date, _mask)
                        # Period not defined
                        for vr in climatevars:
                            if (vr=='PCP' or vr=='Rain'):
                                s.attributes[f'{name}_{vr}_total'] = 'nan'
                            elif (vr=='iPAR_PRFT'):
                                s.attributes[f'{name}_SumiPART'] = 'nan'
                            elif (vr=='PTQ'):
                                s.attributes[f'{name}_{vr}_mean'] = 'nan'
                                s.attributes[f'{name}_SumPTQ'] = 'nan'
                            else:
                                if ('min' in stats):
                                    s.attributes[f'{name}_{vr}_min'] = 'nan'
                                if ('mean' in stats):
                                    s.attributes[f'{name}_{vr}_mean'] = 'nan'
                                if ('max' in stats):
                                    s.attributes[f'{name}_{vr}_max'] = 'nan'
                                if ('sd' in stats):
                                    s.attributes[f'{name}_{vr}_sd'] = 'nan'
                    # return results
                    return s, w_for_stats, _mask

                # Get weather data for entire season
                w_loc = weather[( (weather['Date']>=sowing) & (weather['Date']<=maturity) & (weather['location']==s.loc))]
                if (len(w_loc)>0):
                    # Remove those values taken as outliers or wrong 
                    # If TMax > 30ºC and TMin < -40ºC,  then TMin = TMax – 15 
                    w_loc['TMIN'] = w_loc.apply(lambda row: row['TMAX'] - 15 if (row['TMAX'] > 30 and row['TMIN'] < -40) else row['TMIN'], axis=1)
                    w_loc['TMIN'] = w_loc['TMIN'].apply(lambda x: x if x >=0.0 else 0.0)
                    w_loc['TMAX'] = w_loc['TMAX'].apply(lambda x: x if x >=0.0 else 0.0)

                    # Additional parameters
                    #w_loc['TDay'] = np.nan
                    w_loc['TDay'] = round(  ((2/3)*w_loc['TMAX']) + ((1/3) * w_loc['TMIN']) , 3)
                    w_loc['TAVG'] = round( (w_loc['TMAX'] + w_loc['TMIN']) / 2.0, 3)
                    # PTQ (photothermal quotient)
                    # (This is a crude correction for cold days. With low temperatures, high PTQ values would result).
                    #w_loc['PTQ'] = np.nan
                    w_loc['PTQ'] = w_loc.apply(lambda row: (row['SolRad'] / row['TAVG']) if (row['TAVG'] > 2.0) else 0.0, axis=1)
                    #w_loc['PTQ'] = w_loc['SolRad'] / w_loc['TAVG'] if (w_loc['TAVG'] > 2.0) else 0.0
                    # PRFT (photosynthesis reduction factor due to temperature, from CERES-Wheat)  
                    # PRFT = 1 - 0.0025 * (TDay - 18)^2
                    #w_loc['PRFT'] = np.nan
                    w_loc['PRFT'] = 1 - 0.0025 * (w_loc['TDay'] - 18)**2
                    #If TDay is less than 0, then PRFT = 0  (before, it said that if PRFT is less than 0, then PRFT=0)
                    w_loc['PRFT'] = w_loc['TDay'].apply(lambda x: x if x >=0.0 else 0.0)

                    # Intercepted photosynthetically active solar radiation corrected for temperature
                    # iPAR_PRFT = (SolRad * 0.5) * PRFT  (SumiPART)
                    #w_loc['iPAR_PRFT'] = np.nan
                    w_loc['iPAR_PRFT'] = (w_loc['SolRad'] * 0.5) * w_loc['PRFT']

                    # Period 1: Sowing to heading (S to H)
                    #_mask_SH = ( (w_loc['Date']>=sowing) & (w_loc['Date']<=heading) )
                    #w_SH = w_loc[_mask_SH].reset_index(drop=True)
                    s, w_for_stats, _mask = getStatsbyPeriod(s, w_loc, sowing, heading, name='S-H', 
                                                             climatevars=climatevars, stats=stats)
                    #if verbose is True:
                    #    print(s)

                    # Period 2: Heading to beginning of grain filling (H to H+15d) - Heading plus 15 days
                    # _mask_HHplus15 = ( (w_loc['Date']>=heading) & (w_loc['Date']<=heading_plus_15days) )
                    s, w_for_stats, _mask = getStatsbyPeriod(s, w_loc, heading, heading_plus_15days, name='H-Hplus15d', 
                                                             climatevars=climatevars, stats=stats)

                    # Period 3: Grain filling (H+15d to maturity)
                    s, w_for_stats, _mask = getStatsbyPeriod(s, w_loc, heading_plus_15days, maturity, name='Hplus15d-M', 
                                                             climatevars=climatevars, stats=stats)
                    # Season: sowing to maturity
                    #_mask_Season = ( (w_loc['Date']>=sowing) & (w_loc['Date']<=maturity) )
                    s, w_Season, _mask_Season = getStatsbyPeriod(s, w_loc, sowing, maturity, name='Season', 
                                                                 climatevars=climatevars, stats=['min','mean','max', 'sd'])
                else:
                    print("Error in weather data. Problem in site {} - {}".format(s.uid, s.loc))

            else:
                #s['errors'].append({"uid": s.uid, "loc": s.loc, "error": "Incompleted phenology. Growth stages not defined" })
                if verbose is True:
                    print("Problem in site {} - {}. Error:Incompleted phenology, growth stages not defined.".format(s.uid, s.loc))

        except Exception as err:
            if verbose is True:
                print("Problem in site {} - {}. Error: {}".format(s.uid, s.loc, err))
            #s['errors'].append({"uid": s.uid, "loc": s.loc, 
            #"error": "Distilling the climate data down into meaningful variables. Error: {}".format(err)})

        return s #.getAttr()
    #
    
    
    #
    def process_GapFillingforPhenology(self, sites_to_run=None, climate=True, verbose=False):
        ''' Filling gaps for phenology dates using algorithms developed by Urs
        '''
        if (sites_to_run is None):
            print("Model parameters not valid")
            return
        #
        processed_parcels_error = []
        processed_parcels = []
        for s in sites_to_run:
            try:
                # Get Phenology dates
                _ = s.getPhenologyDates(m=self, verbose=verbose)
                # Estimate Emergence
                _ = s.getEstimatedEmergence(m=self, verbose=verbose)
                # Estimate Heading
                _ = s.getEstimatedHeading(m=self, verbose=verbose)
                # Estimate Maturity - PredMaturity_H - PredMaturity_pH
                _ = s.getEstimatedMaturity(m=self, verbose=verbose) 

                # Phenology
                sowing = None
                emergence = None
                heading = None
                maturity = None

                if ( ('sowing' in s.attributes) and (str(s.attributes['sowing'])!='nan') ):
                    sowing = s.attributes['sowing']
                if ( ('PredEmergence' in s.attributes) and (str(s.attributes['PredEmergence'])!='nan') ):
                    emergence = s.attributes['PredEmergence']
                    s.attributes['emergence'] = emergence

                if ((str(s.attributes['Days_To_Heading'])!='nan') or ( ('heading' in s.attributes) and (str(s.attributes['heading'])!='nan')) ):
                    heading = s.attributes['heading']
                else:
                    heading = s.attributes['PredHeading']
                    s.attributes['heading'] = heading

                if (str(s.attributes['Days_To_Maturity'])!='nan'):
                    maturity = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(s.attributes['Days_To_Maturity']))).split(' ')[0]
                    s.attributes['maturity'] = maturity
                else:
                    # using TAdjDay = 45 / (175 * np.exp(-0.07 * Tavg))
                    if (str(s.attributes['Days_To_Heading'])!='nan'):
                        maturity = s.attributes['PredMaturity_H']
                        s.attributes['maturity'] = maturity
                    else:
                        maturity = s.attributes['PredMaturity_pH']
                        s.attributes['maturity'] = maturity

                if ( (sowing is not None) and (emergence is not None) and (heading is not None) and (maturity is not None) ):
                    # completed Phenology then get statistics between stages
                    # climate means for each genotype at each growth stage in each environment
                    if (climate is True):
                        s = self.process_ClimateStatsforGrowthStage(s, verbose=verbose)
                else:
                    #s['errors'].append({"uid": s.uid, "loc": s.loc, "error": "Growth stages couldn't be estimated" })
                    if verbose is True:
                        print("Problem in site {} - {}. Error: Growth stages couldn't be estimated".format(s.uid, s.loc))

            except Exception as err:
                if verbose is True:
                    print("Problem in site {} - {}. Error: {}".format(s.uid, s.loc, err))
                #s['errors'].append({"uid": s.uid, "loc": s.loc, 
                #                    "error": "Filling gaps for phenology dates. Error: {}".format(err)})
                processed_parcels_error.append(s.getAttr())
            #
            processed_parcels.append(s.getAttr())

        return processed_parcels, processed_parcels_error
    #
    #
    def process_GapFillingforPhenology_v3(self, m, s, climate=False, verbose=False):
        ''' Filling gaps for phenology dates using algorithms developed by Urs
        '''
        try:
            # Get Phenology dates
            _ = s.getPhenologyDates(m=self, verbose=verbose)
            # Estimate Emergence
            _ = s.getEstimatedEmergence(m=self, verbose=verbose)
            # Estimate Heading
            _ = s.getEstimatedHeading(m=self, verbose=verbose)
            # Estimate Maturity - PredMaturity_H - PredMaturity_pH
            _ = s.getEstimatedMaturity(m=self, verbose=verbose) 

            # Phenology
            sowing = None
            emergence = None
            heading = None
            anthesis = None
            maturity = None

            if ( ('sowing' in s.attributes) and (str(s.attributes['sowing'])!='nan') ):
                sowing = s.attributes['sowing']
            if ( ('PredEmergence' in s.attributes) and (str(s.attributes['PredEmergence'])!='nan') ):
                emergence = s.attributes['PredEmergence']
                s.attributes['emergence'] = emergence

            if ((str(s.attributes['Days_To_Heading'])!='nan') or ( ('heading' in s.attributes) and (str(s.attributes['heading'])!='nan')) ):
                heading = s.attributes['heading']
            else:
                heading = s.attributes['PredHeading']
                s.attributes['heading'] = heading

            if ((str(s.attributes['Days_To_Anthesis'])!='nan') or ( ('anthesis' in s.attributes) and (str(s.attributes['anthesis'])!='nan')) ):
                anthesis = s.attributes['anthesis']
            else:
                anthesis = str(pd.to_datetime(str(heading)) + pd.DateOffset(days=10)).split(' ')[0]
                s.attributes['anthesis'] = anthesis

            if (str(s.attributes['Days_To_Maturity'])!='nan'):
                maturity = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(s.attributes['Days_To_Maturity']))).split(' ')[0]
                s.attributes['maturity'] = maturity
            else:
                # using TAdjDay = 45 / (175 * np.exp(-0.07 * Tavg))
                if (str(s.attributes['Days_To_Heading'])!='nan'):
                    maturity = s.attributes['PredMaturity_H']
                    s.attributes['maturity'] = maturity
                else:
                    maturity = s.attributes['PredMaturity_pH']
                    s.attributes['maturity'] = maturity

            if ( (sowing is not None) and (emergence is not None) and (heading is not None) and (maturity is not None) ):
                # completed Phenology then get statistics between stages
                # climate means for each genotype at each growth stage in each environment
                if (climate is True):
                    #s = process_ClimateStatsforGrowthStage(s, verbose=verbose)
                    s = self.process_ClimateStatsforGrowthPeriod_v3(s, weather=None, #self.config['WeatherFile'],
                                        climatevars=['TMIN', 'TAVG', 'TMAX', 'PCP', 'SolRad','RHUMn','RHUMx', 
                                                     'WIND', 'TDay','PTQ', 'iPAR_PRFT'], 
                                        stats=['mean'], verbose=verbose)
            else:
                #s['errors'].append({"uid": s.uid, "loc": s.loc, "error": "Growth stages couldn't be estimated" })
                print("Problem in site {} - {}. Error: {}".format(s.uid, s.loc, "Growth stages couldn't be estimated"))

        except Exception as err:
            #if verbose is True:
            print("Problem in site {} - {}. Error: {}".format(s.uid, s.loc, err))
            #s['errors'].append({"uid": s.uid, "loc": s.loc, "error": "Estimating Thermal Time. Error: {}".format(err)})

        return s.getAttr()
    
    
    # =============================
    # Run model
    # =============================
    def fit(self, sites_to_run=None, season=True, verbose=False):
        '''
            Run an iPAR Yield model to fit yield
            
        :params sites_to_run: Array of Site objects
        :params season: Display weather statistics for different periods
        
        :resutls: An array of sites with intermediate results
        
        '''
        if (sites_to_run is None):
            print("Model parameters not valid")
            return
        #
        processed_parcels_error = []
        processed_parcels = []
        for s in tqdm(sites_to_run):
            try:
                # Get Phenology dates
                _ = s.getPhenologyDates(m=self, verbose=verbose)
                # Estimate Emergence from GDD
                _ = s.getEstimatedPhenologyDates(m=self, verbose=verbose) 
                # Add weather parameters
                _ = s.getWeatherParameters(m=self, season=season, verbose=verbose)
                # The normalized difference vegetation index (NDVI) 
                _ = s.estimateNDVI(m=self, verbose=verbose)
                # iPAR and Yield
                _ = s.getIPAR(m=self, verbose=verbose)
                #print(s.getAttr())
                s.attributes['UID'] = s.uid
                s.attributes['location'] = s.loc
                processed_parcels.append(s.getAttr())
            except:
                #print("Error in site{} - {}".format(s.uid, s.loc))
                processed_parcels_error.append(s.uid)
        # 
        #if (len(processed_parcels_error) > 0): print("Parcelas con error:", processed_parcels_error)
        #pd.DataFrame(processed_parcels)
        #df = pd.DataFrame(processed_parcels)
        #df_errors = pd.DataFrame.from_dict([s.errors[0] for s in sites_to_run if (len(s.errors) > 0) ])

        return processed_parcels, processed_parcels_error
        
    def runModel(self, sites_to_run=None, season=True, verbose=False):
        '''
            Run an iPAR Yield model to fit yield in Parallel
            
        :params sites_to_run: Array of Site objects
        :params season: Display weather statistics for different periods
        
        :resutls: An array of sites with intermediate results
        
        '''
        with Parallel(n_jobs=4, verbose=5) as parallel:
            delayed_funcs = [delayed(lambda s: s.fit(self, season=season, verbose=verbose))(run) for run in sites_to_run]
            output = parallel(delayed_funcs)
        #print(output)
        #df = pd.DataFrame(output)
        return pd.DataFrame(output)
#

    def extractClimatexBestPedigreeSelection(self, m, sites_to_run=None, batch_start=0, batch_end=100, n_jobs=4,
                                             climate=False, fmt="parquet", saveFile=True, verbose=False):
        '''
            Estimate phenology using Urs' iPAR model and extract weather parameters in Parallel

            :params sites_to_run: Array of Site objects

            :resutls: An array of sites with intermediate results

        '''
        if (sites_to_run is None):
            print("Model parameters not valid")
            return
        output = []
        with Parallel(n_jobs=n_jobs, verbose=5) as parallel:
            delayed_funcs = [delayed(lambda s: self.process_GapFillingforPhenology_v3(m, s, climate, verbose))(run) 
                             for run in sites_to_run[batch_start:batch_end]]
            output = parallel(delayed_funcs)
        #print(output)
        #df = pd.DataFrame(output)
        processed_parcels_df = pd.DataFrame(output)
        processed_parcels_df.drop(columns=['errors'], inplace=True)
        processed_parcels_df.reset_index(drop=True, inplace=True)
        # reformat dataset before saving
        if (climate is True):
            col_to_replace = ['S-H_TMIN_lt8C_ndays', 'S-H_TMIN_lt11C_ndays',
               'S-H_TMIN_gt25C_ndays', 'S-H_TAVG_mean', 'S-H_TMAX_mean',
               'S-H_TMAX_gt32C_ndays', 'S-H_PCP_total', 'S-H_SolRad_mean',
               'S-H_RHUMn_mean', 'S-H_RHUMx_mean', 'S-H_WIND_mean',
               'S-H_TDay_mean', 'S-H_PTQ_mean', 'S-H_SumPTQ', 'S-H_SumiPART',
               'H-Hplus15d_TMIN_mean', 'H-Hplus15d_TMIN_lt8C_ndays',
               'H-Hplus15d_TMIN_lt11C_ndays', 'H-Hplus15d_TMIN_gt25C_ndays',
               'H-Hplus15d_TAVG_mean', 'H-Hplus15d_TMAX_mean',
               'H-Hplus15d_TMAX_gt32C_ndays', 'H-Hplus15d_PCP_total',
               'H-Hplus15d_SolRad_mean', 'H-Hplus15d_RHUMn_mean',
               'H-Hplus15d_RHUMx_mean', 'H-Hplus15d_WIND_mean',
               'H-Hplus15d_TDay_mean', 'H-Hplus15d_PTQ_mean', 'H-Hplus15d_SumPTQ',
               'H-Hplus15d_SumiPART', 'Hplus15d-M_TMIN_mean',
               'Hplus15d-M_TMIN_lt8C_ndays', 'Hplus15d-M_TMIN_lt11C_ndays',
               'Hplus15d-M_TMIN_gt25C_ndays', 'Hplus15d-M_TAVG_mean',
               'Hplus15d-M_TMAX_mean', 'Hplus15d-M_TMAX_gt32C_ndays',
               'Hplus15d-M_PCP_total', 'Hplus15d-M_SolRad_mean',
               'Hplus15d-M_RHUMn_mean', 'Hplus15d-M_RHUMx_mean',
               'Hplus15d-M_WIND_mean', 'Hplus15d-M_TDay_mean',
               'Hplus15d-M_PTQ_mean', 'Hplus15d-M_SumPTQ', 'Hplus15d-M_SumiPART',
               'Season_TMIN_min', 'Season_TMIN_mean', 'Season_TMIN_max',
               'Season_TMIN_sd', 'Season_TMIN_lt8C_ndays',
               'Season_TMIN_lt11C_ndays', 'Season_TMIN_gt25C_ndays',
               'Season_TAVG_min', 'Season_TAVG_mean', 'Season_TAVG_max',
               'Season_TAVG_sd', 'Season_TMAX_min', 'Season_TMAX_mean',
               'Season_TMAX_max', 'Season_TMAX_sd', 'Season_TMAX_gt32C_ndays',
               'Season_PCP_total', 'Season_SolRad_min', 'Season_SolRad_mean',
               'Season_SolRad_max', 'Season_SolRad_sd', 'Season_RHUMn_min',
               'Season_RHUMn_mean', 'Season_RHUMn_max', 'Season_RHUMn_sd',
               'Season_RHUMx_min', 'Season_RHUMx_mean', 'Season_RHUMx_max',
               'Season_RHUMx_sd', 'Season_WIND_min', 'Season_WIND_mean',
               'Season_WIND_max', 'Season_WIND_sd', 'Season_TDay_min',
               'Season_TDay_mean', 'Season_TDay_max', 'Season_TDay_sd',
               'Season_PTQ_mean', 'Season_SumPTQ', 'Season_SumiPART']
            try:
                for col in col_to_replace: 
                    processed_parcels_df[col] = processed_parcels_df[col].replace('nan', np.nan, regex=False)
            except Exception as err:
                print("Problem reformating some climate features. Error:", err)

        if (saveFile is True):
            try:
                # Save in binary format
                hoy = datetime.now().strftime('%Y%m%d')
                if (fmt=="parquet"):
                    processed_parcels_df.to_parquet(os.path.join(
                        self.config['RESULTS_PATH'],f"IWIN_forAI_{hoy}_part{batch_start}_{batch_end}.parquet"), 
                                                    index=False, compression=None)
                elif (fmt=="csv"):
                        processed_parcels_df.to_csv(os.path.join(
                            self.config['RESULTS_PATH'], f"IWIN_forAI_{hoy}_part{batch_start}_{batch_end}.csv"),
                                                    index=False)
            except Exception as err:
                print("Problem saving intermediate files. Error:", err)
        output = None
        del output
        _ = gc.collect()
        return processed_parcels_df

    # ---------------------------------------------------------------------------------------
    # Estimate phenology using Urs' iPAR model and extract weather parameters
    # ---------------------------------------------------------------------------------------
    def estimateWeatherPhenology_byObs(self, model, sites_to_run, batch_size=1000, n_jobs=4, 
                                   climate=True, saveIntermediateFiles=False, saveFile=True, vb=False):
        '''
            Estimate phenology using Urs' iPAR model and extract weather parameters for each observation, running in Parallel

        '''
        hoy = datetime.now().strftime('%Y%m%d')
        nObs = len(sites_to_run)
        if (vb is True):
            print("Hours to complete weather statistics in my old 4 cores server: {:.1f} hrs".format((nObs/50000)*3.5))
        step = np.ceil(nObs / batch_size)
        if (vb is True):
            print("Number of Obs:{} - in {:.0f} steps, using batch size: {}".format(nObs, step, batch_size))
        processed_parcels_merged = None
        # Run in parallel
        for b in range(int(step)):
            part = b + 1
            batch_start= b * batch_size
            batch_end = (b+1) * batch_size
            if (vb is True):
                print("Batch {} - from {} to {}".format(b, batch_start, batch_end))
            processed_parcels_df = self.extractClimatexBestPedigreeSelection(m=model, sites_to_run=sites_to_run, 
                                                                              batch_start=batch_start, batch_end=batch_end, n_jobs=n_jobs, 
                                                                              climate=climate, fmt="parquet", saveFile=saveIntermediateFiles, 
                                                                              verbose=False)
            processed_parcels_merged = pd.concat([processed_parcels_merged, processed_parcels_df])

        print("Phenology and Weather parameters were estimated successfully!")

        # ---------------------------------
        # Format again to share for others
        # ---------------------------------
        if (climate is True and 'Hplus15d-M_SolRad_mean' in processed_parcels_merged.columns):
            # Remove values where days to anthesis is smaller than days to heading
            # Choose any null column for example 'Hplus15d-M_SolRad_mean'
            df_final = processed_parcels_merged[~(processed_parcels_merged['Hplus15d-M_SolRad_mean'].isnull()) ]
        else:
            df_final = processed_parcels_merged.copy()

        # Simulated phenology
        df_final['predDays_To_Emergence'] = df_final['PredDaysToEmergence']
        df_final['predDays_To_Heading'] = df_final['PredDaysToHead'] 
        #df_final['predDays_To_Anthesis'] = df_final['PredDaysToAnthesis'] 
        df_final['predDays_To_Maturity'] = df_final['PredDaysToMaturity_pH'] 
        # Preserved Observed data
        df_final['Days_To_Heading'] = df_final['obsDays_To_Heading'] 
        df_final['Days_To_Anthesis'] = df_final['obsDays_To_Anthesis'] 
        df_final['Days_To_Maturity'] = df_final['obsDays_To_Maturity'] 

        df_final['ObsYield'] = df_final['ObsYield'].astype(float).round(3)
        # Corregir o agregar columnas faltantes para el analisis de los mejores pedigrees en ambientes de bajo rendimiento y clima extremo
        df_final["LocOccCntry"] = df_final["location"].astype(str) +"_"+ df_final["Occ"].astype(str) +"_"+ df_final["country"].astype(str)
        df_final["E"] = df_final["Nursery_Yr"].astype(str) +"_"+ df_final["Occ"].astype(str) +"_"+ df_final["location"].astype(str)

        # format column names back to original
        df_final.rename(columns={ 'nursery':'Nursery', 'trial_name':'Trial name','CID':'Cid', 'SID':'Sid',
            'location':'Loc_no', 'Days_To_Heading':'DAYS_TO_HEADING', 'Days_To_Anthesis':'DAYS_TO_ANTHESIS',
            'Days_To_Maturity':'DAYS_TO_MATURITY', 'sowing':'SowingDate', 
            'country':'Country', 'locationname':'Loc_desc', 'cycle':'Cycle', 
            'ObsYield':'GRAIN_YIELD', 'plant_height':'PLANT_HEIGHT', 
            'grain_weight_1k':'1000_GRAIN_WEIGHT', 'test_weight':'TEST_WEIGHT'
        }, inplace=True)

        # Save all values
        if (saveFile is True):
            try:
                # df_final.to_csv(os.path.join(config['RESULTS_PATH'], f'IWIN_wstats_allFeaturesforAI_{hoy}.csv'), index=False)
                df_final.to_parquet(os.path.join(self.config['RESULTS_PATH'],f"IWIN_wstats_allFeaturesforAI_{hoy}.parquet"), 
                                    index=False, compression=None)
            except Exception as err:
                print("Problem saving final results. Error:", err)
        3
        if (climate is True and 'S-H_TMIN_mean' in df_final.columns):
            cols_sel = ['UID','Nursery', 'Nursery_Yr','Occ','Loc_no', 'E' ,'GID',#'Cid', 'Sid',  'UGID',
                    #'Gen_name', 'Gen_no', 'Rep', 'Sub_block', 'Plot',
                    'Trial name', 'Loc_desc', 'lat', 'lon', 'Country', 'LocOccCntry', 
                    'Cycle',  'HarvestYr', 'CycleStartYr',
                    'SowingDate', 'emergence', 'heading', 'anthesis', 'grainfilling','maturity',#
                    'DAYS_TO_HEADING', 'DAYS_TO_ANTHESIS', 'DAYS_TO_MATURITY',
                    'PLANT_HEIGHT', '1000_GRAIN_WEIGHT', 'TEST_WEIGHT','GRAIN_YIELD', 
                    'Grand_Mean', 'Geno_Variance', 'Res_Variance', 'Replicates',
                    'Heritability', 'LSD', 'CV', 'Genotype_significance',
                    'GRAIN_YIELD_BLUEs', 'Quantiles95(GRAIN_YIELD_BLUEs)', 'normYieldBLUE95Perc', 
                    #'p_DAYS_TO_HEADING_byGID', 'p_DAYS_TO_MAT_byGID', 
                    'Date@35DAS', 'DayLength@35DAS',
                    'S-H_TMIN_mean', 'S-H_TMIN_lt8C_ndays',
                    'S-H_TMIN_lt11C_ndays', 'S-H_TMIN_gt25C_ndays', 'S-H_TAVG_mean',
                    'S-H_TMAX_mean', 'S-H_TMAX_gt32C_ndays', 'S-H_PCP_total',
                    'S-H_SolRad_mean', 'S-H_RHUMn_mean', 'S-H_RHUMx_mean',
                    'S-H_WIND_mean', 'S-H_TDay_mean', 'S-H_PTQ_mean', 'S-H_SumPTQ',
                    'S-H_SumiPART', 'H-Hplus15d_TMIN_mean',
                    'H-Hplus15d_TMIN_lt8C_ndays', 'H-Hplus15d_TMIN_lt11C_ndays',
                    'H-Hplus15d_TMIN_gt25C_ndays', 'H-Hplus15d_TAVG_mean',
                    'H-Hplus15d_TMAX_mean', 'H-Hplus15d_TMAX_gt32C_ndays',
                    'H-Hplus15d_PCP_total', 'H-Hplus15d_SolRad_mean',
                    'H-Hplus15d_RHUMn_mean', 'H-Hplus15d_RHUMx_mean',
                    'H-Hplus15d_WIND_mean', 'H-Hplus15d_TDay_mean',
                    'H-Hplus15d_PTQ_mean', 'H-Hplus15d_SumPTQ', 'H-Hplus15d_SumiPART',
                    'Hplus15d-M_TMIN_mean', 'Hplus15d-M_TMIN_lt8C_ndays',
                    'Hplus15d-M_TMIN_lt11C_ndays', 'Hplus15d-M_TMIN_gt25C_ndays',
                    'Hplus15d-M_TAVG_mean', 'Hplus15d-M_TMAX_mean',
                    'Hplus15d-M_TMAX_gt32C_ndays', 'Hplus15d-M_PCP_total',
                    'Hplus15d-M_SolRad_mean', 'Hplus15d-M_RHUMn_mean',
                    'Hplus15d-M_RHUMx_mean', 'Hplus15d-M_WIND_mean',
                    'Hplus15d-M_TDay_mean', 'Hplus15d-M_PTQ_mean', 'Hplus15d-M_SumPTQ',
                    'Hplus15d-M_SumiPART', 'Season_TMIN_min', 'Season_TMIN_mean',
                    'Season_TMIN_max', 'Season_TMIN_sd', 'Season_TMIN_lt8C_ndays',
                    'Season_TMIN_lt11C_ndays', 'Season_TMIN_gt25C_ndays',
                    'Season_TAVG_min', 'Season_TAVG_mean', 'Season_TAVG_max',
                    'Season_TAVG_sd', 'Season_TMAX_min', 'Season_TMAX_mean',
                    'Season_TMAX_max', 'Season_TMAX_sd', 'Season_TMAX_gt32C_ndays',
                    'Season_PCP_total', 'Season_SolRad_min', 'Season_SolRad_mean',
                    'Season_SolRad_max', 'Season_SolRad_sd', 'Season_RHUMn_min',
                    'Season_RHUMn_mean', 'Season_RHUMn_max', 'Season_RHUMn_sd',
                    'Season_RHUMx_min', 'Season_RHUMx_mean', 'Season_RHUMx_max',
                    'Season_RHUMx_sd', 'Season_WIND_min', 'Season_WIND_mean',
                    'Season_WIND_max', 'Season_WIND_sd', 'Season_TDay_min',
                    'Season_TDay_mean', 'Season_TDay_max', 'Season_TDay_sd',
                    'Season_PTQ_mean', 'Season_SumPTQ', 'Season_SumiPART']

        else:
            cols_sel = ['UID','Nursery', 'Nursery_Yr','Occ','Loc_no','E','GID',#'Cid', 'Sid',  'UGID',
                    #'Gen_name', 'Gen_no', 'Rep', 'Sub_block', 'Plot',
                    'Trial name', 'Loc_desc', 'lat', 'lon', 'Country', 'LocOccCntry', 
                    'Cycle',  'HarvestYr', 'CycleStartYr',
                    'SowingDate', 'emergence', 'heading', 'anthesis', 'grainfilling','maturity',
                    'DAYS_TO_HEADING', 'DAYS_TO_ANTHESIS', 'DAYS_TO_MATURITY',
                    'PLANT_HEIGHT', '1000_GRAIN_WEIGHT', 'TEST_WEIGHT','GRAIN_YIELD', 
                    'Grand_Mean', 'Geno_Variance', 'Res_Variance', 'Replicates',
                    'Heritability', 'LSD', 'CV', 'Genotype_significance',
                    'GRAIN_YIELD_BLUEs', 'Quantiles95(GRAIN_YIELD_BLUEs)', 'normYieldBLUE95Perc', 
                    'Date@35DAS', 'DayLength@35DAS']

        # Save
        if (saveFile is True):
            try:
                df_final_selFts = df_final[cols_sel].sort_values(['Nursery_Yr', 'Occ', 'Loc_no', 'GID']).reset_index(drop=True)
                df_final_selFts.to_parquet(os.path.join(self.config['RESULTS_PATH'], f"IWIN_phenology_wstats_grpByGID_forAI_{hoy}.parquet"), 
                                           index=False, compression=None)
                #df_final_selFts.to_csv(os.path.join(self.config['RESULTS_PATH'], f'IWIN_phenology_wstats_grpByGID_forAI_{hoy}.csv'), index=False)
            except Exception as err:
                print("Problem saving final results. Error:", err)

        processed_parcels_merged = None
        del processed_parcels_merged
        _ = gc.collect()

        return df_final, df_final_selFts

    
    #
    









