# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

__version__ = "IWIN version 2.0.0.dev"
__author__ = "Ernesto Giron Echeverry, Urs Christoph Schulthess et al."
__copyright__ = "Copyright (c) 2023 CIMMYT"
__license__ = "Public Domain"

#from . import *

import sys, os, gc
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date, datetime
from joblib import Parallel, delayed
from tqdm import tqdm

from . import *
from ..util import *  #normalize_Yield, KDE_hist_plot

sys.path.insert(0, r"../iwin")
import iwin


def load_data(config=None):
    ''' Load IWIN and AgERA5 datasets in parquet format
        Reading the Parquet format is much more efficient.

    :return: Two dataframes for crop phenology and weather dataset respectively.

    '''
    if (config is None):
        print("Configuration not valid")
        return

    IWIN_sites_phenology_path = os.path.join(config['PROJECT_PATH'], config['PHENO_FILE'])
    if (os.path.exists(IWIN_sites_phenology_path)):
        PhenoFile = pd.read_parquet(IWIN_sites_phenology_path) #, engine="fastparquet")
    else:
        print("Error reading crop phenology file")
        return

    IWIN_sites_WeatherFile_path = os.path.join(config['PROJECT_PATH'], config['WEATHER_FILE'])
    if (os.path.exists(IWIN_sites_WeatherFile_path)):
        WeatherFile = pd.read_parquet(IWIN_sites_WeatherFile_path) #, engine="fastparquet")
    else:
        print("Error reading weather file")
        return

    config['WeatherFile'] = WeatherFile
    config['PhenoFile'] = PhenoFile
    return PhenoFile


def load_IWINdataset(config=None, imputingData=True, cleanSD=True, missigHisto=True, 
                    saveHistoFig=False, saveRawFile=False, verbose=False):
    '''
        Load updated IWIN dataset 
    '''
    dateparse = lambda x: dt.datetime.strptime(str(x), '%Y-%m-%d') if (str(x)!='' and str(x)!='nan') else None #'%Y-%m-%d %H:%M:%S')
    # Load updated IWIN for analysis
    IWIN_sites_phenology_path = os.path.join(config['PROJECT_PATH'], config['PHENO_FILE'])
    if (os.path.exists(IWIN_sites_phenology_path)):
        #PhenoFile = pd.read_parquet(IWIN_sites_phenology_path) #, engine="fastparquet")
        df_IWIN = pd.read_csv(IWIN_sites_phenology_path, index_col=False, parse_dates=['SowingDate'], date_parser=dateparse)
    else:
        print("Error reading raw IWIN file")
        return
    
    df_IWIN['UID'] = df_IWIN.index + 1
    #df_IWIN['SowingDate'] = pd.to_datetime(df_IWIN['SowingDate'].astype(str), format='%Y-%m-%d')
    # Correct CycleStartYr
    #df_IWIN['CycleStartYr'] = df_IWIN['SowingDate'].dt.year
    df_IWIN['Cycle'] = df_IWIN['Cycle'].astype(str)
    df_IWIN['Lat'] = df_IWIN['Lat'].astype(float).round(3)
    df_IWIN['Long'] = df_IWIN['Long'].astype(float).round(3)

    df_IWIN['SowDOY'] = df_IWIN['SowingDate'].apply(lambda x: pd.Timestamp(x).dayofyear)
    df_IWIN["LocOccCntry"] = df_IWIN["Loc_no"].astype(str) +"_"+ df_IWIN["Occ"].astype(str) +"_"+ df_IWIN["Country"].astype(str)
    #df_IWIN["ID"] = df_IWIN["Nursery_Yr"].astype(str) +"_"+ df_IWIN["Occ"].astype(str) +"_"+ df_IWIN["Loc_no"].astype(str) +"_"+ df_IWIN["Trial name"].astype(str) +"_"+ df_IWIN["GID"].astype(str)
    df_IWIN["E"] = df_IWIN["Nursery_Yr"].astype(str) +"_"+ df_IWIN["Occ"].astype(str) +"_"+ df_IWIN["Loc_no"].astype(str)
    df_IWIN["UGID"] = df_IWIN["Cid"].astype(str) +"_"+ df_IWIN["Sid"].astype(str) +"_"+ df_IWIN["GID"].astype(str)
    
    # change the name of the trial to standarize and reduce space
    df_IWIN['Trial name'] = df_IWIN['Trial name'].replace(
        ['1ST ESWYT', '2ND ESWYT', '3RD ESWYT', '4TH ESWYT', '5TH ESWYT',
           '6TH ESWYT', '7TH ESWYT', '8TH ESWYT', '9TH ESWYT', '10TH ESWYT',
           '11TH ESWYT', '12TH ESWYT', 'V13ESWYT', '14TH ESWYT', '15TH ESWYT',
           '16TH ESWYT', '17TH ESWYT', '18TH ESWYT', '19TH ESWYT',
           '20TH ESWYT', '21ST ESWYT', '22ND ESWYT',
           '23RD ELITE SPRING WHEAT YT', '24TH ELITE SPRING WHEAT YT',
           '25TH ELITE SPRING WHEAT YT', '26TH ELITE SPRING WHEAT YT',
           '27TH ELITE SPRING WHEAT YT', '28TH ELITE SPRING WHEAT YT',
           '29TH ELITE SPRING WHEAT YT', '30TH ELITE SPRING WHEAT YT',
           '31st ELITE SPRING WHEAT YT', '32ND ELITE SPRING WHEAT YT',
           '33RD ELITE SPRING WHEAT YT', '34TH ELITE SPRING WHEAT YT',
           '35TH ELITE SPRING WHEAT YT', '36TH ELITE SPRING WHEAT YT',
           '37TH ELITE SPRING WHEAT YT', '37TH ELITE SELECTION WHEAT YT',
           '38TH ELITE SPRING WHEAT YT', '38TH ELITE SELECTION WHEAT YT',
           '39TH ELITE SPRING WHEAT YT', '40TH ELITE SPRING WHEAT YT',
           '41ST ELITE SPRING WHEAT YT', '1ST HTWYT', '2ND HTWYT',
           '3RD HTWYT', '4TH HTWYT', '5TH HTWYT', '6TH HTWYT', '7TH HTWYT',
           '8TH HTWYT', '9TH HTWYT', '10TH HIGH TEMPERATURE WHEAT YT',
           '11TH HIGH TEMPERATURE WHEAT YT', '12TH HIGH TEMPERATURE WHEAT YT',
           '13TH HIGH TEMPERATURE WHEAT YT', '14TH HIGH TEMPERATURE WHEAT YT',
           '15TH HIGH TEMPERATURE WHEAT YT', '16TH HIGH TEMPERATURE WHEAT YT',
           '17TH HIGH TEMPERATURE WHEAT YT', '18TH HIGH TEMPERATURE WHEAT YT',
           '19TH HIGH TEMPERATURE WHEAT YT', '12IDYN', '13IDYN', '14IDYN',
           '15IDYN', '16IDYN', '17IDYN', '18IDYN', '19IDYN', '20IDYN',
           '21IDYN', '22IDYN', '23RD IDYN', '24IDYN', '25IDYN', '26IDYN',
           '27IDYN', '28IDYN', '29TH IDYN', '30IDYN', '31IDYN', '32IDYN',
           '33IDYN', '34TH INTERNATIONAL DURUM YN',
           '35TH INTERNATIONAL DURUM YN', '36TH INTERNATIONAL DURUM YN',
           '37TH INTERNATIONAL DURUM YN', '38TH INTERNATIONAL DURUM YN',
           '39TH INTERNATIONAL DURUM YN', '40TH INTERNATIONAL DURUM YN',
           '41ST INTERNATIONAL DURUM YN', '42ND INTERNATIONAL DURUM YN',
           '43RD INTERNATIONAL DURUM YN', '44TH INTERNATIONAL DURUM YN',
           '45TH INTERNATIONAL DURUM YN', '46TH INTERNATIONAL DURUM YN',
           '47TH INTERNATIONAL DURUM YN', '48TH INTERNATIONAL DURUM YN',
           '49TH INTERNATIONAL DURUM YN', '50TH INTERNATIONAL DURUM YN',
           '51ST INTERNATIONAL DURUM YN', '52ND INTERNATIONAL DURUM YN',
           '1ST SAWYT', '2ND SAWYT', '3RD SAWYT', '4TH SAWYT', '5TH SAWYT',
           '6TH SAWYT', '7TH SAWYT', '8TH SAWYT', '9TH SAWYT',
           '10TH  SEMI-ARID WHEAT YT', '11TH SEMI-ARID WHEAT YT',
           '12TH SEMI-ARID WHEAT YT', '13TH SEMI-ARID WHEAT YT',
           '14TH SEMI-ARID WHEAT YT', '15TH SEMI-ARID WHEAT YT',
           '16TH SEMI-ARID WHEAT YT', '17TH SEMI-ARID WHEAT YT',
           '18TH SEMI-ARID WHEAT YT', '19TH SEMI-ARID WHEAT YT',
           '20TH SEMI-ARID WHEAT YT', '21ST SEMI-ARID WHEAT YT',
           '22ND SEMI-ARID WHEAT YT', '23RD SEMI-ARID WHEAT YT',
           '24TH SEMI-ARID WHEAT YT', '25TH SEMI-ARID WHEAT YT',
           '26TH SEMI-ARID WHEAT YT', '27TH SEMI-ARID WHEAT YT',
           '28TH SEMI-ARID WHEAT YT'],
        ['1ST ESWYT', '2ND ESWYT', '3RD ESWYT', '4TH ESWYT', '5TH ESWYT',
           '6TH ESWYT', '7TH ESWYT', '8TH ESWYT', '9TH ESWYT', '10TH ESWYT',
           '11TH ESWYT', '12TH ESWYT', '13TH ESWYT', '14TH ESWYT', '15TH ESWYT',
           '16TH ESWYT', '17TH ESWYT', '18TH ESWYT', '19TH ESWYT',
           '20TH ESWYT', '21ST ESWYT', '22ND ESWYT',
           '23RD ESWYT', '24TH ESWYT',
           '25TH ESWYT', '26TH ESWYT',
           '27TH ESWYT', '28TH ESWYT',
           '29TH ESWYT', '30TH ESWYT',
           '31ST ESWYT', '32ND ESWYT',
           '33RD ESWYT', '34TH ESWYT',
           '35TH ESWYT', '36TH ESWYT',
           '37TH ESWYT', '37TH ESWYT',
           '38TH ESWYT', '38TH ESWYT',
           '39TH ESWYT', '40TH ESWYT',
           '41ST ESWYT', '1ST HTWYT', '2ND HTWYT',
           '3RD HTWYT', '4TH HTWYT', '5TH HTWYT', '6TH HTWYT', '7TH HTWYT',
           '8TH HTWYT', '9TH HTWYT', '10TH HTWYT',
           '11TH HTWYT', '12TH HTWYT',
           '13TH HTWYT', '14TH HTWYT',
           '15TH HTWYT', '16TH HTWYT',
           '17TH HTWYT', '18TH HTWYT',
           '19TH HTWYT', '12TH IDYN', '13TH IDYN', '14TH IDYN',
           '15TH IDYN', '16TH IDYN', '17TH IDYN', '18TH IDYN', '19TH IDYN', '20TH IDYN',
           '21ST IDYN', '22TH IDYN', '23RD IDYN', '24TH IDYN', '25TH IDYN', '26TH IDYN',
           '27TH IDYN', '28TH IDYN', '29TH IDYN', '30TH IDYN', '31ST IDYN', '32TH IDYN',
           '33RD IDYN', '34TH IDYN',
           '35TH IDYN', '36TH IDYN',
           '37TH IDYN', '38TH IDYN',
           '39TH IDYN', '40TH IDYN',
           '41ST IDYN', '42ND IDYN',
           '43RD IDYN', '44TH IDYN',
           '45TH IDYN', '46TH IDYN',
           '47TH IDYN', '48TH IDYN',
           '49TH IDYN', '50TH IDYN',
           '51ST IDYN', '52ND IDYN',
           '1ST SAWYT', '2ND SAWYT', '3RD SAWYT', '4TH SAWYT', '5TH SAWYT',
           '6TH SAWYT', '7TH SAWYT', '8TH SAWYT', '9TH SAWYT',
           '10TH SAWYT', '11TH SAWYT',
           '12TH SAWYT', '13TH SAWYT',
           '14TH SAWYT', '15TH SAWYT',
           '16TH SAWYT', '17TH SAWYT',
           '18TH SAWYT', '19TH SAWYT',
           '20TH SAWYT', '21ST SAWYT',
           '22ND SAWYT', '23RD SAWYT',
           '24TH SAWYT', '25TH SAWYT',
           '26TH SAWYT', '27TH SAWYT',
           '28TH SAWYT'],
        regex=True)
    
    df_IWIN_filled = None
    df_IWIN_cleaned = None
    if (imputingData is True):
        if(verbose is True):
            print("Imputing missing values...")
        df_IWIN_filled = imputingMissingValues(df=df_IWIN, verbose=verbose)
        
    if(missigHisto is True):
        if(verbose is True):
            print("Total number of observations: {}".format(len(df_IWIN)))
        dirname_path = os.path.join(config['RESULTS_PATH'] , 'IWIN_GxE_AI')
        cols_missing_values, mis_val_table = missingData(df_GE=df_IWIN, title='IWIN\nMissing data', 
                                                         dirname=dirname_path, fmt='pdf', showFig=True, 
                                                         saveFig=saveHistoFig, verbose=verbose)
    # Clean Sowing Date
    if (cleanSD is True):
        if (df_IWIN_filled is not None):
            # Remove observations with missing sowing date values
            df_IWIN_cleaned = df_IWIN_filled[~(df_IWIN_filled['SowingDate'].isnull())]
            ### Remove observations with missing grain yield values
            #df_IWIN_cleaned = df_IWIN_cleaned[~(df_IWIN_cleaned['GRAIN_YIELD'].isnull())]
        else:
            df_IWIN_cleaned = df_IWIN[~(df_IWIN['SowingDate'].isnull())]
        # Check for more missing values
        #for c in df_IWIN_filled.columns:
        #    print(f"{c}: ",df_IWIN_filled[c].isnull().sum())
        # Correct CycleStartYr
        df_IWIN_cleaned['CycleStartYr'] = df_IWIN_cleaned['SowingDate'].dt.year
        df_IWIN_cleaned['Gen_name'].fillna('', inplace=True)
        df_IWIN_cleaned['Loc_desc'].fillna('', inplace=True)
        df_IWIN_cleaned.reset_index(drop=True, inplace=True)
        if(verbose is True):
            print("-"*80)
            displayIWINSummary(df_IWIN, df_IWIN_cleaned)
            print("-"*80)
    
    if (saveRawFile is True):
        hoy = datetime.now().strftime('%Y%m%d')
        # Save modifications
        df_IWIN.to_parquet(os.path.join(config['PROJECT_PATH'],'data',f"IWIN_raw_{hoy}.parquet"), index=False, compression=None)
        df_IWIN_filled.to_parquet(os.path.join(config['PROJECT_PATH'],'data',f"IWIN_filled_raw_{hoy}_forAnalysis.parquet"), 
                                  index=False, compression=None)
        df_IWIN_cleaned.to_parquet(os.path.join(config['PROJECT_PATH'],'data',f"IWIN_cleaned_raw_{hoy}_forAnalysis.parquet"), 
                                   index=False, compression=None)
    
    # Load weather
    IWIN_sites_WeatherFile_path = os.path.join(config['PROJECT_PATH'], config['WEATHER_FILE'])
    if (os.path.exists(IWIN_sites_WeatherFile_path)):
        WeatherFile = pd.read_parquet(IWIN_sites_WeatherFile_path) #, engine="fastparquet")
    else:
        print("Error reading weather file")
        return
    
    config['WeatherFile'] = WeatherFile
    config['PhenoFile'] = df_IWIN
    return df_IWIN, df_IWIN_cleaned


# -----------------------------------
# Filter phenology dataset
# -----------------------------------
def filterPhenologyData(config=None, data=None, fld=None, value=None, selcols=None, verbose=False):
    '''
        Filter dataset by Nursery

        :params data: A table or DF with trial for each site

        :results: An array or DF of filtered sites
    '''
    if (data is None):
        data = config['PhenoFile']
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


# ---------------------------------------
# Filling out missing values
# ---------------------------------------
def imputingMissingValues(df, verbose=True):
    ''' 
        Imputation preserves all cases by replacing missing data with an estimated value based on the average of the site-nursery-year. 
        Once all missing values have been imputed, the data set can then be analysed using standard techniques for complete data.
    '''
    df_IWIN_filled = df.copy()
    # 1 - Raw value, 2 - Filled out with mean of trial (site-year), 3 - Missing values after filling out
    df_IWIN_filled['QC_GRAIN_YIELD'] = 1
    df_IWIN_filled['QC_1000_GRAIN_WEIGHT'] = 1
    df_IWIN_filled['QC_DAYS_TO_ANTHESIS'] = 1
    df_IWIN_filled['QC_DAYS_TO_HEADING'] = 1
    df_IWIN_filled['QC_DAYS_TO_MATURITY'] = 1
    df_IWIN_filled['QC_PLANT_HEIGHT'] = 1
    df_IWIN_filled['QC_TEST_WEIGHT'] = 1
    df_IWIN_filled['QC_SowingDate'] = 1
    #
    df_IWIN_filled.loc[((df_IWIN_filled['GRAIN_YIELD'].isnull()) ), 'QC_GRAIN_YIELD'] = 2
    df_IWIN_filled.loc[((df_IWIN_filled['1000_GRAIN_WEIGHT'].isnull()) ), 'QC_1000_GRAIN_WEIGHT'] = 2
    df_IWIN_filled.loc[((df_IWIN_filled['DAYS_TO_ANTHESIS'].isnull()) ), 'QC_DAYS_TO_ANTHESIS'] = 2
    df_IWIN_filled.loc[((df_IWIN_filled['DAYS_TO_HEADING'].isnull()) ), 'QC_DAYS_TO_HEADING'] = 2
    df_IWIN_filled.loc[((df_IWIN_filled['DAYS_TO_MATURITY'].isnull()) ), 'QC_DAYS_TO_MATURITY'] = 2
    df_IWIN_filled.loc[((df_IWIN_filled['PLANT_HEIGHT'].isnull()) ), 'QC_PLANT_HEIGHT'] = 2
    df_IWIN_filled.loc[((df_IWIN_filled['TEST_WEIGHT'].isnull()) ), 'QC_TEST_WEIGHT'] = 2
    df_IWIN_filled.loc[((df_IWIN_filled['SowingDate'].isnull()) ), 'QC_SowingDate'] = 2
    #
    if (verbose is True):
        print("GRAIN_YIELD Quality Control", df_IWIN_filled['QC_GRAIN_YIELD'].value_counts().to_dict())
        print("1000_GRAIN_WEIGHT Quality Control", df_IWIN_filled['QC_1000_GRAIN_WEIGHT'].value_counts().to_dict())
        print("DAYS_TO_ANTHESIS Quality Control", df_IWIN_filled['QC_DAYS_TO_ANTHESIS'].value_counts().to_dict())
        print("DAYS_TO_HEADING Quality Control", df_IWIN_filled['QC_DAYS_TO_HEADING'].value_counts().to_dict())
        print("DAYS_TO_MATURITY Quality Control", df_IWIN_filled['QC_DAYS_TO_MATURITY'].value_counts().to_dict())
        print("PLANT_HEIGHT Quality Control", df_IWIN_filled['QC_PLANT_HEIGHT'].value_counts().to_dict())
        print("TEST_WEIGHT Quality Control", df_IWIN_filled['QC_TEST_WEIGHT'].value_counts().to_dict())
        print("SowingDate Quality Control", df_IWIN_filled['QC_SowingDate'].value_counts().to_dict())
    
    # get the mean values for each trial
    data_toFillMissingValues = df_IWIN_filled.groupby(['E'], as_index=False).agg({'GRAIN_YIELD':np.nanmean, 
                                                                                  '1000_GRAIN_WEIGHT':np.nanmean, 
                                                                                  'DAYS_TO_ANTHESIS':np.nanmean, 
                                                                                  'DAYS_TO_HEADING':np.nanmean,
                                                                                  'DAYS_TO_MATURITY':np.nanmean, 
                                                                                  'PLANT_HEIGHT':np.nanmean, 
                                                                                  'TEST_WEIGHT':np.nanmean, 
                                                                                  'SowDOY':np.nanmean})
    # Convert DOY to full date
    def convertDOYToDate(year, sDOY):
        try:
            sd = pd.Timestamp('{}-01-01'.format(int(year)))
            sdoy = pd.DateOffset(days=int(sDOY))
            return sd + sdoy
        except (ValueError, TypeError):
            return np.nan
        
    # fill missing values
    for e in df_IWIN_filled['E'].unique():
        # Average of the site-nursery-year
        GY = data_toFillMissingValues[data_toFillMissingValues['E']==e]['GRAIN_YIELD'].values[0]
        GW = data_toFillMissingValues[data_toFillMissingValues['E']==e]['1000_GRAIN_WEIGHT'].values[0]
        DA = data_toFillMissingValues[data_toFillMissingValues['E']==e]['DAYS_TO_ANTHESIS'].values[0]
        DH = data_toFillMissingValues[data_toFillMissingValues['E']==e]['DAYS_TO_HEADING'].values[0]
        DM = data_toFillMissingValues[data_toFillMissingValues['E']==e]['DAYS_TO_MATURITY'].values[0]
        PH = data_toFillMissingValues[data_toFillMissingValues['E']==e]['PLANT_HEIGHT'].values[0]
        TW = data_toFillMissingValues[data_toFillMissingValues['E']==e]['TEST_WEIGHT'].values[0]
        SD = data_toFillMissingValues[data_toFillMissingValues['E']==e]['SowDOY'].values[0]
        # Fill out 
        df_IWIN_filled.loc[((df_IWIN_filled['E']==e) & (df_IWIN_filled['GRAIN_YIELD'].isnull()) ), 'GRAIN_YIELD'] = GY
        df_IWIN_filled.loc[((df_IWIN_filled['E']==e) & (df_IWIN_filled['1000_GRAIN_WEIGHT'].isnull()) ), '1000_GRAIN_WEIGHT'] = GW
        df_IWIN_filled.loc[((df_IWIN_filled['E']==e) & (df_IWIN_filled['DAYS_TO_ANTHESIS'].isnull()) ), 'DAYS_TO_ANTHESIS'] = DA
        df_IWIN_filled.loc[((df_IWIN_filled['E']==e) & (df_IWIN_filled['DAYS_TO_HEADING'].isnull()) ), 'DAYS_TO_HEADING'] = DH
        df_IWIN_filled.loc[((df_IWIN_filled['E']==e) & (df_IWIN_filled['DAYS_TO_MATURITY'].isnull()) ), 'DAYS_TO_MATURITY'] = DM
        df_IWIN_filled.loc[((df_IWIN_filled['E']==e) & (df_IWIN_filled['PLANT_HEIGHT'].isnull()) ), 'PLANT_HEIGHT'] = PH
        df_IWIN_filled.loc[((df_IWIN_filled['E']==e) & (df_IWIN_filled['TEST_WEIGHT'].isnull()) ), 'TEST_WEIGHT'] = TW
        df_IWIN_filled.loc[((df_IWIN_filled['E']==e) & (df_IWIN_filled['SowDOY'].isnull()) ), 'SowDOY'] = SD
        
        # Convert Sowing date from DOY
        HarvestYr = df_IWIN_filled[df_IWIN_filled['E']==e]['HarvestYr'].values[0]
        df_IWIN_filled.loc[((df_IWIN_filled['E']==e) & (df_IWIN_filled['SowingDate'].isnull()) ), 'SowingDate'] = convertDOYToDate(HarvestYr, SD)

    # Missing values where it's not possible to imputing new values
    df_IWIN_filled.loc[((df_IWIN_filled['GRAIN_YIELD'].isnull()) ), 'QC_GRAIN_YIELD'] = 3
    df_IWIN_filled.loc[((df_IWIN_filled['1000_GRAIN_WEIGHT'].isnull()) ), 'QC_1000_GRAIN_WEIGHT'] = 3
    df_IWIN_filled.loc[((df_IWIN_filled['DAYS_TO_ANTHESIS'].isnull()) ), 'QC_DAYS_TO_ANTHESIS'] = 3
    df_IWIN_filled.loc[((df_IWIN_filled['DAYS_TO_HEADING'].isnull()) ), 'QC_DAYS_TO_HEADING'] = 3
    df_IWIN_filled.loc[((df_IWIN_filled['DAYS_TO_MATURITY'].isnull()) ), 'QC_DAYS_TO_MATURITY'] = 3
    df_IWIN_filled.loc[((df_IWIN_filled['PLANT_HEIGHT'].isnull()) ), 'QC_PLANT_HEIGHT'] = 3
    df_IWIN_filled.loc[((df_IWIN_filled['TEST_WEIGHT'].isnull()) ), 'QC_TEST_WEIGHT'] = 3
    df_IWIN_filled.loc[((df_IWIN_filled['SowingDate'].isnull()) ), 'QC_SowingDate'] = 3

    if (verbose is True):
        print("GRAIN_YIELD Quality Control after filling out", df_IWIN_filled['QC_GRAIN_YIELD'].value_counts().to_dict())
        print("1000_GRAIN_WEIGHT Quality Control after filling out", df_IWIN_filled['QC_1000_GRAIN_WEIGHT'].value_counts().to_dict())
        print("DAYS_TO_ANTHESIS Quality Control after filling out", df_IWIN_filled['QC_DAYS_TO_ANTHESIS'].value_counts().to_dict())
        print("DAYS_TO_HEADING Quality Control after filling out", df_IWIN_filled['QC_DAYS_TO_HEADING'].value_counts().to_dict())
        print("DAYS_TO_MATURITY Quality Control after filling out", df_IWIN_filled['QC_DAYS_TO_MATURITY'].value_counts().to_dict())
        print("PLANT_HEIGHT Quality Control after filling out", df_IWIN_filled['QC_PLANT_HEIGHT'].value_counts().to_dict())
        print("TEST_WEIGHT Quality Control after filling out", df_IWIN_filled['QC_TEST_WEIGHT'].value_counts().to_dict())
        print("SowingDate Quality Control after filling out", df_IWIN_filled['QC_SowingDate'].value_counts().to_dict())
    return df_IWIN_filled

#
def displayIWINSummary(df_IWIN=None, df_IWIN_cleaned=None):
    if (df_IWIN is None or df_IWIN_cleaned is None):
        return
    print("Observations: {}".format(len(df_IWIN)))
    print("Observations after QC: {}".format(len(df_IWIN_cleaned)))
    print("% data cannot be used: {:.2f}%".format( abs((len(df_IWIN_cleaned) - len(df_IWIN)) / len(df_IWIN) )*100 ))

    print("{} Years from {} to {}".format((int(df_IWIN_cleaned['SowingDate'].dt.year.max()) - int(df_IWIN_cleaned['SowingDate'].dt.year.min())),
                      int(df_IWIN_cleaned['SowingDate'].dt.year.min()), int(df_IWIN_cleaned['SowingDate'].dt.year.max())))
    print("locations",len(df_IWIN_cleaned['Loc_no'].unique()))
    print("countries",len(df_IWIN_cleaned['Country'].unique()))
    #print("pedigrees",len(df_IWIN_cleaned['Gen_name'].unique()))
    print("pedigrees",len(df_IWIN_cleaned['GID'].unique()))
    print("Site-years",len(df_IWIN_cleaned['E'].unique()))
    print("ESWYT Observations: {}".format(len(df_IWIN_cleaned[df_IWIN_cleaned['Nursery']=='ESWYT'])))
    print("IDYN Observations: {}".format(len(df_IWIN_cleaned[df_IWIN_cleaned['Nursery']=='IDYN'])))
    print("HTWYT Observations: {}".format(len(df_IWIN_cleaned[df_IWIN_cleaned['Nursery']=='HTWYT'])))
    print("SAWYT Observations: {}".format(len(df_IWIN_cleaned[df_IWIN_cleaned['Nursery']=='SAWYT'])))
    
#
def displaySummaryByNursery(df_N=None, nursery='NURSERY', MIN_PERC=5, MAX_PERC=98, KDEHist=True,
                             sel_cols = ['GRAIN_YIELD', '1000_GRAIN_WEIGHT', 
                                         'DAYS_TO_HEADING', 'DAYS_TO_ANTHESIS',  
                                         'DAYS_TO_MATURITY',  'PLANT_HEIGHT', 'TEST_WEIGHT'
                                        ],
                            verbose=True
                            ):
    ''' 
        
        - Cut-off by using percentiles 5% and 98%
    '''
    if (df_N is None):
        print("Data not found!")
        return
    RANGES_BY_NURSERY = []
    if (verbose is True):
        print("{} Observations: {}".format(nursery, len(df_N)))
        print("Nursery-years",len(df_N['Trial name'].unique()), df_N['Nursery_Yr'].min(), df_N['Nursery_Yr'].max())
        print("{} Years from {} to {}".format(len(df_N['Nursery_Yr'].unique()),
                          int(df_N['SowingDate'].dt.year.min()), int(df_N['SowingDate'].dt.year.max())))
        print("Locations",len(df_N['Loc_no'].unique()))
        print("Countries",len(df_N['Country'].unique()))
        print("Pedigrees",len(df_N['GID'].unique()))
        print("Site-years",len(df_N['E'].unique()))
    
    df = df_N[sel_cols].copy() #.dropna(subset=sel_cols)
    if (KDEHist is True):
        KDE_hist_plot(df)
    print("Ranges using CI [5%, 98%]")
    for col in df.columns:
        df2 = df.dropna(subset=[col])  # Drop null values
        mn, avg, mx = np.percentile(df2[col], MIN_PERC), np.percentile(df2[col], 50), np.percentile(df2[col], MAX_PERC)
        RANGES_BY_NURSERY.append({"nursery": nursery, "trait":col, "min":mn, "mean":avg, "max":mx })
        print("{} - Min:{:.2f} Mean:{:.2f} Max:{:.2f}".format(col,mn, avg, mx))
        
    return pd.DataFrame(RANGES_BY_NURSERY)

#
def joinBLUPnBLUE_andNormilizedYield_GrpByGID(config, df, BLU, stats, nursery='', saveFile=True):
    '''
        Join BLUP and BLUE values, and Normalize Yield
        
        It uses 95th percentile of yield (top 5%) of each site-year as a reference and express the yield 
        of each GID within a site-year in percent of the top yield.
        
        Return a group by GID table
    '''
    # ---------------------------
    # Join BLUP and BLUE values
    # ---------------------------
    statistics_Nursery = pd.merge(BLU,stats, how='left', on=['E','Nursery_Yr','Occ','Loc_no' ])
    statistics_Nursery['Genotype_significance'] = statistics_Nursery['Genotype_significance'].astype(float).round(8)
    statistics_Nursery['Replicates'] = statistics_Nursery['Replicates'].astype(int)
    statistics_Nursery['status'].fillna('', inplace=True)
    statistics_Nursery.drop(columns=['ExpDes','status','nEnv'], inplace=True)
    # 
    df_statistics_Nursery_final = pd.merge(df,statistics_Nursery, how='left', 
                                         on=['E','Nursery_Yr','Occ','Loc_no','GID','Nursery' ],
                                         indicator=True
                                        )
    df_statistics_Nursery_final = df_statistics_Nursery_final[df_statistics_Nursery_final['_merge']=='both']
    df_statistics_Nursery_final.drop(columns=['_merge'], inplace=True)
    # Use the same names
    df_statistics_Nursery_final['GRAIN_YIELD_BLUEs'] = df_statistics_Nursery_final['BLUE_YLD']
    # ------------------
    # Normalize Yield
    # ------------------
    # Use 95th percentile of yield (top 5%) of each site-year as a reference and express the yield 
    # of each GID within a site-year in percent of the top yield.
    for sy in df_statistics_Nursery_final['E'].unique():
        df_sample_siteYr = df_statistics_Nursery_final[df_statistics_Nursery_final['E']==sy]
        perc_95 = np.percentile(df_sample_siteYr['GRAIN_YIELD_BLUEs'], 95) #BLUE_YLD
        df_sample_siteYr['normYieldBLUE95Perc'] = df_sample_siteYr['GRAIN_YIELD_BLUEs'].apply(lambda x: normalize_Yield(x, perc_95))
        # Update records
        df_statistics_Nursery_final.loc[(df_statistics_Nursery_final['E']==sy), 'normYieldBLUE95Perc'] = df_sample_siteYr['normYieldBLUE95Perc']
        df_statistics_Nursery_final.loc[(df_statistics_Nursery_final['E']==sy), 'Quantiles95(GRAIN_YIELD_BLUEs)'] = perc_95
    # ------------------
    # Group by GID
    # ------------------
    df_stats_grpByGID = df_statistics_Nursery_final.groupby(['Nursery_Yr','Occ','Loc_no','E','GID'], 
                                  as_index=False)\
    .agg({
        'Trial name':'first', 'Nursery':'first','Country':'first', 'Loc_desc':'first',
        'Cycle':'last','HarvestYr':'mean', 'CycleStartYr':'mean','SowingDate':'last',
        #'Cid', 'Sid', 'Gen_no', 'Rep':'count', 'Sub_block', 'Plot', 
        'GRAIN_YIELD':'mean',
       '1000_GRAIN_WEIGHT':'mean', 'DAYS_TO_ANTHESIS':'mean', 'DAYS_TO_HEADING':'mean',
       'DAYS_TO_MATURITY':'mean', 'PLANT_HEIGHT':'mean', 'TEST_WEIGHT':'mean', 
       'Lat':'mean', 'Long':'mean', 'BLUP_YLD':'mean', 'BLUE_YLD':'mean', 'Grand_Mean':'mean',
       'Geno_Variance':'mean', 'Res_Variance':'mean', 'Replicates':'mean', 'Heritability':'mean',
       'LSD':'mean', 'CV':'mean', 'Genotype_significance':'mean', 'GRAIN_YIELD_BLUEs':'mean',
        'Quantiles95(GRAIN_YIELD_BLUEs)':'mean','normYieldBLUE95Perc':'mean'
    })\
    .sort_values(['Nursery_Yr','Occ','Loc_no'])

    #df_stats_grpByGID['CycleStartYr'] = df_stats_grpByGID['CycleStartYr'].astype(int)
    df_stats_grpByGID['GRAIN_YIELD'] = df_stats_grpByGID['GRAIN_YIELD'].astype(float).round(3)

    # Save results
    if (saveFile is True):
        hoy = datetime.now().strftime('%Y%m%d')
        nry_min, nyr_max = df_stats_grpByGID['Nursery_Yr'].min(), df_stats_grpByGID['Nursery_Yr'].max()
        df_stats_grpByGID.to_csv(os.path.join(config['RESULTS_PATH'], f'{nursery}_nyr{nry_min}-{nyr_max}_stats_grpByGID_{hoy}.csv'), index=False)
    
    statistics_Nursery = None
    #df_statistics_Nursery_final = None
    del statistics_Nursery #df_statistics_Nursery_final
    _ = gc.collect()
    return df_statistics_Nursery_final, df_stats_grpByGID

#
def getStatsByTrial(config, df_final, fmt='parquet', saveFile=True, verbose=False):
    '''
        Estimate stats (BLUEs, BLUPs, Heritability, CV, LSD, Genotype significance, among others) for each individual trial.
        
    '''
    df=df_final.copy()
    df_statistics_IWIN_final = None
    df_IWIN_stats_grpByGID = None
    for n in df['Nursery'].unique():
        # get predefined stats
        if (n =='ESWYT'):
            output_metrics = pd.read_csv(os.path.join(config['BLUEnBLUPs_PATH'], f'ESWYT_statistics_20230729.csv'), index_col=False)
            BLUEs_BLUPs = pd.read_csv(os.path.join(config['BLUEnBLUPs_PATH'], f'ESWYT_BLUEs_BLUPs_20230729.csv'), index_col=False)
        
        if (n =='IDYN'):
            output_metrics = pd.read_csv(os.path.join(config['BLUEnBLUPs_PATH'], f'IDYN_statistics_20230729.csv'), index_col=False)
            BLUEs_BLUPs = pd.read_csv(os.path.join(config['BLUEnBLUPs_PATH'], f'IDYN_BLUEs_BLUPs_20230729.csv'), index_col=False)
            
        if (n =='HTWYT'):
            output_metrics = pd.read_csv(os.path.join(config['BLUEnBLUPs_PATH'], f'HTWYT_statistics_20230729.csv'), index_col=False)
            BLUEs_BLUPs = pd.read_csv(os.path.join(config['BLUEnBLUPs_PATH'], f'HTWYT_BLUEs_BLUPs_20230729.csv'), index_col=False)
            
        if (n =='SAWYT'):
            output_metrics = pd.read_csv(os.path.join(config['BLUEnBLUPs_PATH'], f'SAWYT_statistics_20230729.csv'), index_col=False)
            BLUEs_BLUPs = pd.read_csv(os.path.join(config['BLUEnBLUPs_PATH'], f'SAWYT_BLUEs_BLUPs_20230729.csv'), index_col=False)
        #
        df_statistics_final, df_stats_grpByGID = joinBLUPnBLUE_andNormilizedYield_GrpByGID(config, 
                                                                df=df[df['Nursery']==n], 
                                                                BLU=BLUEs_BLUPs, stats=output_metrics, 
                                                                nursery=n, saveFile=True)
        #if (verbose is True):
        #    print(df_statistics_final.shape, df_stats_grpByGID.shape)
        # Combine all nurseries
        df_statistics_IWIN_final = pd.concat([df_statistics_IWIN_final, df_statistics_final])
        df_IWIN_stats_grpByGID = pd.concat([df_IWIN_stats_grpByGID, df_stats_grpByGID])

    # Save 
    if (saveFile is True):
        hoy = datetime.now().strftime('%Y%m%d')
        if (fmt=='parquet'):
            df_statistics_IWIN_final.to_parquet(os.path.join(config['RESULTS_PATH'], f"IWIN_raw_phenology_stats_{hoy}.parquet"), 
                                                index=False, compression=None)
            df_IWIN_stats_grpByGID.to_parquet(os.path.join(config['RESULTS_PATH'], f"IWIN_raw_phenology_stats_grpByGID_{hoy}.parquet"), 
                                              index=False, compression=None)
        if (fmt=='csv'):
            df_statistics_IWIN_final.to_csv(os.path.join(config['RESULTS_PATH'], f"IWIN_raw_phenology_stats_{hoy}.csv"), index=False )
            df_IWIN_stats_grpByGID.to_csv(os.path.join(config['RESULTS_PATH'], f"IWIN_raw_phenology_stats_grpByGID_{hoy}.csv"), index=False )
    #
    _ = gc.collect()
    return df_statistics_IWIN_final, df_IWIN_stats_grpByGID


# ------------------------------------------------
# Prepare dataset for IWIN library
# ------------------------------------------------
def createArrayOfObjects_toRunInParallel(df):
    '''
        Create an array of sites with the IWIN observations to speed up further analysis
        This array of objects is useful to run models in parallel.
    '''
    # 
    sites_to_run = []
    parcels = df.copy()
    for idx in tqdm(parcels.index):
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

    #sites_to_run[0].attributes
    return sites_to_run

def formatFeaturesforIWIN(df, removeObsPheno=False, arrObj=True):
    ''' 
        Features or columns in IWIN table need to rename and reformat according to the original IWIN package or library.
        
        - Nursery is needed to evaluate the Equation to use in grain filling
        - Rename columns to match with the IWIN library
        - Create array of objects to run models in parallel
        
    '''
    df_IWIN_wstats = df.copy()
    # rename columns to match with the IWIN library
    df_IWIN_wstats.rename(columns={
        'Cid':'CID', 'Sid':'SID', 'Trial name':'trial_name', 'Loc_no':'location','Nursery':'nursery',
        'Country':'country', 'Loc_desc':'locationname','Cycle':'cycle','Lat':'lat', 'Long':'lon', 'SowingDate':'sowing', 
        'DAYS_TO_HEADING':'Days_To_Heading', 'DAYS_TO_ANTHESIS':'Days_To_Anthesis','DAYS_TO_MATURITY':'Days_To_Maturity', 
        'GRAIN_YIELD':'ObsYield', '1000_GRAIN_WEIGHT':'grain_weight_1k', 'PLANT_HEIGHT':'plant_height','TEST_WEIGHT':'test_weight',

    }, inplace=True)

    # Check out columns format
    df_IWIN_wstats['Days_To_Heading'] = df_IWIN_wstats['Days_To_Heading'].apply(toInt)
    df_IWIN_wstats['Days_To_Anthesis'] = df_IWIN_wstats['Days_To_Anthesis'].apply(toInt)
    df_IWIN_wstats['Days_To_Maturity'] = df_IWIN_wstats['Days_To_Maturity'].apply(toInt)
    
    df_IWIN_wstats['ObsYield'] = df_IWIN_wstats['ObsYield'].astype(float).round(3)
    df_IWIN_wstats['BLUP_YLD'] = df_IWIN_wstats['BLUP_YLD'].astype(float).round(3)
    df_IWIN_wstats['BLUE_YLD'] = df_IWIN_wstats['BLUE_YLD'].astype(float).round(3)
    df_IWIN_wstats['Grand_Mean'] = df_IWIN_wstats['Grand_Mean'].astype(float).round(3)
    df_IWIN_wstats['Geno_Variance'] = df_IWIN_wstats['Geno_Variance'].astype(float).round(4)
    df_IWIN_wstats['Res_Variance'] = df_IWIN_wstats['Res_Variance'].astype(float).round(4)
    df_IWIN_wstats['Genotype_significance'] = df_IWIN_wstats['Genotype_significance'].astype(float).round(5)
    df_IWIN_wstats['GRAIN_YIELD_BLUEs'] = df_IWIN_wstats['GRAIN_YIELD_BLUEs'].astype(float).round(3)
    df_IWIN_wstats['Quantiles95(GRAIN_YIELD_BLUEs)'] = df_IWIN_wstats['Quantiles95(GRAIN_YIELD_BLUEs)'].astype(float).round(3)
    df_IWIN_wstats['normYieldBLUE95Perc'] = df_IWIN_wstats['normYieldBLUE95Perc'].astype(float).round(3)
    
    # Preserve observed dataset
    df_IWIN_wstats['obsDays_To_Heading'] = df_IWIN_wstats['Days_To_Heading'] 
    df_IWIN_wstats['obsDays_To_Anthesis'] = df_IWIN_wstats['Days_To_Anthesis'] 
    df_IWIN_wstats['obsDays_To_Maturity'] = df_IWIN_wstats['Days_To_Maturity'] 
    df_IWIN_wstats['obsGrain_weight_1k'] = df_IWIN_wstats['grain_weight_1k'] 
    df_IWIN_wstats['obsPlant_height'] = df_IWIN_wstats['plant_height'] 
    df_IWIN_wstats['obsTest_weight'] = df_IWIN_wstats['test_weight'] 

    # Using Observed phenology 
    df_IWIN_wstats['emergence'] = ''
    df_IWIN_wstats['heading'] = df_IWIN_wstats[['sowing', 'Days_To_Heading']]\
    .apply(lambda row: getPhenologyDateAfterSowing(row['sowing'], row['Days_To_Heading']), axis=1)
    df_IWIN_wstats['anthesis'] = df_IWIN_wstats[['sowing', 'Days_To_Anthesis']]\
    .apply(lambda row: getPhenologyDateAfterSowing(row['sowing'], row['Days_To_Anthesis']), axis=1)
    df_IWIN_wstats['maturity'] = df_IWIN_wstats[['sowing', 'Days_To_Maturity']]\
    .apply(lambda row: getPhenologyDateAfterSowing(row['sowing'], row['Days_To_Maturity']), axis=1)
    df_IWIN_wstats['Obs_DaysHM'] = df_IWIN_wstats[['maturity', 'heading']].apply(lambda x: getObsDaysHM(x['maturity'], x['heading']), axis=1)

    # Using simulated phenology
    if (removeObsPheno is True):
        df_IWIN_wstats['emergence'] = ''
        df_IWIN_wstats['heading'] = ''
        df_IWIN_wstats['anthesis'] = ''
        df_IWIN_wstats['maturity'] = ''

    df_IWIN_wstats = df_IWIN_wstats.sort_values(['Nursery_Yr', 'Occ', 'location', 'GID'])
    # create an unique ID
    df_IWIN_wstats.reset_index(drop=True, inplace=True)
    df_IWIN_wstats['UID'] = df_IWIN_wstats.index + 1
    
    # Process array of objects for running in parallel
    sites_to_run = createArrayOfObjects_toRunInParallel(df_IWIN_wstats)
    
    return df_IWIN_wstats, sites_to_run

#











