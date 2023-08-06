# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import sys, os, gc
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from . import *
from ..util import custom_rolling, assignME
#from ..util import figures
from ..data import *

sys.path.insert(0, r"../iwin")
import iwin


def prepareDatasetforR(df,  folder='GxE'):
    ''' Prepare the IWIN dataset in a specific format for analysis in R statistical packages
    '''
    observations_byTrial = None
    if 'Nursery' in df.columns:
        for n in df['Nursery'].unique():
            df_n = df[df['Nursery']==n]
            # Es necesario extraer todos los datos en relación con el sitio para que 
            # no hallan faltantes en bloques y demas
            observations_byTrial = df_n[['Plot', 'Rep', 'Sub_block','Gen_no', 'E', 
                                         'G', 'Trial name', 'Country','GRAIN_YIELD']]
            observations_byTrial['Sub_block'].fillna(99, inplace=True)
            #observations_byTrial.loc[observations_byTrial.Sub_block.isna(), 'Sub_block'] = observations_byTrial.Sub_block.fillna(99)
            observations_byTrial['Plot'] = observations_byTrial['Plot'].astype(int)
            try:
                observations_byTrial['Sub_block'] = observations_byTrial['Sub_block'].astype(int)
            except Exception as err:
                print(f"Warning with missing Sub_block value in {n} nursery")
            # Change column names
            observations_byTrial.rename(columns={
                'Gen_no': 'GEN',
                'Sub_block': 'BLOCK',
                'Rep': 'REP',
                #'Country': 'ME',
                #'location': 'ENV',
                'Trial name':'nursery',
                'GRAIN_YIELD': 'GY'
            }, inplace=True)

            # Generamos las categorias o factores para el modelo
            observations_byTrial['GEN'] = "GEN" + observations_byTrial['GEN'].astype(str)
            observations_byTrial['REP'] = "REP" + observations_byTrial['REP'].astype(str)
            observations_byTrial['BLOCK'] = "SB" + observations_byTrial['BLOCK'].astype(str)

            # Temporalmente nombrado de esta manera para agrupar por pais
            observations_byTrial['ME'] = observations_byTrial['Country'].astype('category').cat.codes 
            #observations_byTrial['G'] = observations_byTrial['G'].astype('category').cat.codes 
            observations_byTrial['ENV'] = observations_byTrial['E'].astype('category').cat.codes 

            #observations_byTrial = observations_byTrial[['ENV', 'GEN', 'REP', 'BLOCK', 'GY']]
            # Save to process in R
            dirname_path = os.path.join(RESULTS_IWIN_PATH , folder)
            if not os.path.exists(dirname_path):
                os.makedirs(dirname_path)
            observations_byTrial.to_csv(os.path.join(dirname_path , f'data_observations_byTrial_{n}_forR.csv'), index=False)
    else:
        print("Not nursery attribute found")
    return observations_byTrial
        

def export_to_METAR(nursery='IDYN', nyr=52, occ=76, loc=22263, saveFile=False):
    df = df_IWIN[((df_IWIN['Nursery']==nursery) & (df_IWIN['Nursery_Yr']==nyr) 
                  & (df_IWIN['Occ']==occ) & (df_IWIN['Loc_no']==loc))]
    df_METAR = df[[
        'UID', 'Nursery_Yr', 'Occ', 'Loc_no', 'Trial name', 'GID', 'Gen_name', 'Gen_no', 'Rep', 'Sub_block',
           'Plot', 'Country','GRAIN_YIELD',
           '1000_GRAIN_WEIGHT', 'DAYS_TO_ANTHESIS', 'DAYS_TO_HEADING',
           'DAYS_TO_MATURITY', 'PLANT_HEIGHT', 'TEST_WEIGHT',
    ]] #['Sub_block'].unique()

    df_METAR["LocOccCntry"] = df_METAR["Loc_no"].astype(str) +"_"+ df_METAR["Occ"].astype(str) +"_"+ df_METAR["Country"].astype(str)
    df_METAR["ID"] = df_METAR["Nursery_Yr"].astype(str) +"_"+ df_METAR["Occ"].astype(str) +"_"+ df_METAR["Loc_no"].astype(str) +"_"+ df_METAR["Trial name"].astype(str) +"_"+ df_METAR["GID"].astype(str)
    df_METAR["E"] = df_METAR["Nursery_Yr"].astype(str) +"_"+ df_METAR["Occ"].astype(str) +"_"+ df_METAR["Loc_no"].astype(str)

    df_METAR.rename(columns={
        #'UID', 'Nursery_Yr', 'Occ', 'Trial name','Rep',
        'Loc_no':'Location', 
        'GID':'G',  'Gen_name':'Gname', 'Gen_no':'Gno',   'Sub_block':'BLK', 
        'GRAIN_YIELD':'YLD', '1000_GRAIN_WEIGHT':'GW', 
        'DAYS_TO_ANTHESIS':'DA',  'DAYS_TO_HEADING':'DH',
           'DAYS_TO_MATURITY':'DM', 'PLANT_HEIGHT':'PH', 'TEST_WEIGHT':'TW', 
    }, inplace=True)

    sel_cols_metaR = ['UID', 'G', 'Gname', 'E', 'Plot','Rep', 'BLK', 
                      'YLD', 'GW', 'DA', 'DH', 'DM', 'PH', 'TW', 'Country',
                     ]

    df_METAR = df_METAR[sel_cols_metaR].reset_index(drop=True)
    if (saveFile is True):
        df_METAR.to_csv(os.path.join('./', f'df_METAR_Nyr{nyr}_Occ{occ}_Loc{loc}_metaR.csv'), index=False)

    return df_METAR

#
def joinBLUPnBLUE_andNormilizedYield_GrpByGID(df, BLU, stats, nursery='', saveFile=True):
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
        df_sample_siteYr['normYield95perc'] = df_sample_siteYr['GRAIN_YIELD_BLUEs'].apply(lambda x: normalize_Yield(x, perc_95))
        # Update records
        df_statistics_Nursery_final.loc[(df_statistics_Nursery_final['E']==sy), 
                                        'normYield95perc'] = df_sample_siteYr['normYield95perc']
        df_statistics_Nursery_final.loc[(df_statistics_Nursery_final['E']==sy), 
                                        'Quantiles95(GRAIN_YIELD_BLUEs)'] = perc_95
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
        'Quantiles95(GRAIN_YIELD_BLUEs)':'mean','normYield95perc':'mean'
    })\
    .sort_values(['Nursery_Yr','Occ','Loc_no'])

    #df_stats_grpByGID['CycleStartYr'] = df_stats_grpByGID['CycleStartYr'].astype(int)
    df_stats_grpByGID['GRAIN_YIELD'] = df_stats_grpByGID['GRAIN_YIELD'].astype(float).round(3)

    # Save results
    if (saveFile is True):
        hoy = datetime.now().strftime('%Y%m%d')
        nry_min, nyr_max = df_stats_grpByGID['Nursery_Yr'].min(), df_stats_grpByGID['Nursery_Yr'].max()
        df_stats_grpByGID.to_csv(os.path.join(DATASET_IWIN_PATH, f'{nursery}_nyr{nry_min}-{nyr_max}_stats_grpByGID_{hoy}.csv'), index=False)
    
    statistics_Nursery = None
    #df_statistics_Nursery_final = None
    del statistics_Nursery #df_statistics_Nursery_final
    _ = gc.collect()
    return df_statistics_Nursery_final, df_stats_grpByGID

#







