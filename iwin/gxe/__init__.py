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
from sklearn.linear_model import (LinearRegression, RANSACRegressor)

from . import *
from . import ammi
from . import gge
from ..util import custom_rolling, assignME
from ..util import figures
from ..data import *

sys.path.insert(0, r"../iwin")
import iwin

__version__ = "iPAR Yield model version 1.0.1.dev"
__author__ = "Urs Christoph schulthess, Ernesto Giron Echeverry"
__copyright__ = "Copyright (C) 2023 CIMMYT-Henan Collaborative Innovation Center"
__license__ = "Public Domain"

class Nursery(object):
    def __init__(self, nursery, data, params=None, weather=None):
        self.nursery = nursery
        self.data = data
        self.weather = weather
        self.params = params
    
    def getAvgYieldbyNurseryYear(self, df=None, nursery=None, verbose=False):
        '''For each nursery year, calculate the average yield of each location and Occ.
        
        :params df:
        :params nursery:
        
        :return: 
        
        '''
        if (df is None):
            df = self.data
            if (df is None):
                print("Input data not valid")
                return
        if (nursery is None):
            nursery = self.nursery
            if (nursery is None):
                print("Nursery is required")
                return
        # ------------------------------------------------------
        # Average Yield by Location x Occ in a Nusery-year
        # ------------------------------------------------------
        NURSERY_trial = df[~df['ObsYield'].isnull()].query(f'loc_code == "{nursery}"')
        if (verbose is True):
            print("Number of locations: "+ iwin.HT.bold + iwin.HT.fg.orange + "{}"
                  .format(len(NURSERY_trial['location'].value_counts())) + iwin.HT.reset)
            print("Number of different genotypes: "+ iwin.HT.bold + iwin.HT.fg.green + "{}"
                  .format(len(NURSERY_trial['G'].value_counts())) + iwin.HT.reset)

        AvgNURSERY_Trial_YieldxGID = NURSERY_trial.groupby(['location', 'Occ', 'YearofSow'], as_index=False)\
        .agg({'ObsYield':'mean', 'G':'count'}).rename(columns={'ObsYield':'AvGYxLocOcc', 'G':'countOfGenotypes'})
        # ------------------------------------------------------
        # Average Yield by GID in a Nusery-year
        # ------------------------------------------------------
        AvgNURSERY_Trial_YieldxLocOcc = NURSERY_trial.groupby(['G', 'location', 'Occ', 
                                                               'loc_code', 'YearofSow'], as_index=False)\
        .agg({'ObsYield':'mean', 'loc_code':'count'}).rename(columns={'ObsYield':'AvGYxGID', 
                                                                      'loc_code':'countOfLocations'})

        avgGY_Nursery_Year_Loc_Occ = AvgNURSERY_Trial_YieldxLocOcc.merge(AvgNURSERY_Trial_YieldxGID[['location', 'Occ',
                                                                                                     'AvGYxLocOcc', 
                                                                                                     'countOfGenotypes']], 
                                                                      how='left', on=['location', 'Occ'])
        
        self.avgGY = avgGY_Nursery_Year_Loc_Occ
        del NURSERY_trial
        gc.collect()
        return avgGY_Nursery_Year_Loc_Occ
    
    
    # ------------------------------------------------------
    # Select GIDs by linear regression lines
    # ------------------------------------------------------
    def genotypeSelection_byLinearRegression(self, df_GY=None, nursery=None, methods=['OLS', 'RANSAC'], verbose=True):
        ''' Select GIDs by linear regression lines

            Method 1: OLS
            Method 2: Robust linear regression (RANSAC)
        '''
        if (df_GY is None or nursery is None):
            print("Inputs are not valid")
            return
        if (df_GY is None or nursery is None):
            print("Inputs are not valid")
            return
        
        if (verbose is True):
            print("Genotype Selection using linear regressions... " + iwin.HT.bold + nursery + iwin.HT.reset)
        df = df_GY.copy()
        df['method_1'] = 'OLS'
        df['method_2'] = 'RANSAC'
        df['Nursery'] = nursery
        df['m1'] = None
        df['m2'] = None
        df['b1'] = None
        df['b2'] = None
        df['target_m1'] = 0
        df['target_m2'] = 0
        df['linecolor_m1'] = 'black'
        df['linecolor_m2'] = 'black'
        df['environment_m1'] = ''
        df['environment_m2'] = ''
        GID_environments_Trial = pd.DataFrame()
        target_GIDs_Trial = []
        GID_environments_m1 = []
        GID_environments_m2 = []
        for gid in df['G'].unique():
            df_gid = df[df['G']==gid]
            #LocOccSowyr = {"Location":df_gid['location'], "Occ":df_gid['Occ'], "YearofSow":df_gid['YearofSow']}
            #x = df_gid['AvGYxGID'].reshape(-1, 1)
            #t = df_gid['AvGYxLocOcc']
            #plotline_X = np.arange(x.min(), x.max()).reshape(-1, 1)
            # determine best fit line
            #lr = LinearRegression().fit(x, y)
            y = df_gid['AvGYxGID'].to_numpy().flatten()
            x = df_gid['AvGYxLocOcc'].to_numpy().flatten().reshape(-1, 1)
            plotline_X = np.arange(x.min(), x.max()).reshape(-1, 1)
            lr = LinearRegression().fit(x, y)
            ransac = RANSACRegressor(random_state=42).fit(x, y)
            # Add regression lines
            y_linear_regression = lr.predict(plotline_X)
            y_ransac_regression = ransac.predict(plotline_X)
            m1, c1 = 1.0, 0.0
            # We are interested in the purple and yellow lines (GIDs)
            if ('OLS' in methods):
                # -----------------------------
                # Method 1 - OLS
                # -----------------------------
                # Intercept and slope
                #print(lr.intercept_, lr.coef_[0])
                intercept = lr.intercept_
                m2 = lr.coef_[0]
                if (intercept>0 and m2>=m1):
                    df.loc[ (df['G']==gid), 'm1'] = m2
                    df.loc[ (df['G']==gid), 'b1'] = intercept
                    df.loc[ (df['G']==gid), 'target_m1'] = 1
                    df.loc[ (df['G']==gid), 'linecolor_m1'] = "orange"
                    df.loc[ (df['G']==gid), 'environment_m1'] = "Above Average Poor and Good Environment"
                    GID_environments_m1.append({"method":'OLS', "Nursery":nursery, "G": gid, "m":m1, "b":intercept, "target":1, "linecolor":"orange", "environment":"Above Average Poor and Good Environment"})
                elif (intercept<0 and m1>=m2 ):
                    df.loc[ (df['G']==gid), 'm1'] = m2
                    df.loc[ (df['G']==gid), 'b1'] = intercept
                    df.loc[ (df['G']==gid), 'target_m1'] = 0
                    df.loc[ (df['G']==gid), 'linecolor_m1'] = "red"
                    df.loc[ (df['G']==gid), 'environment_m1'] = "Below Average Poor and Good Environment"
                    GID_environments_m1.append({"method":'OLS', "Nursery":nursery, "G": gid, "m":m1, "b":intercept, "target":0, "linecolor":"red", "environment":"Below Average Poor and Good Environment"})
                elif (intercept>=0 and m1>=m2):
                    df.loc[ (df['G']==gid), 'm1'] = m2
                    df.loc[ (df['G']==gid), 'b1'] = intercept
                    df.loc[ (df['G']==gid), 'target_m1'] = 1
                    df.loc[ (df['G']==gid), 'linecolor_m1'] = "purple"
                    df.loc[ (df['G']==gid), 'environment_m1'] = "Above Average Poor and Below Good Environment"
                    GID_environments_m1.append({"method":'OLS', "Nursery":nursery, "G": gid, "m":m1, "b":intercept, "target":1, "linecolor":"purple", "environment":"Above Average Poor and Below Good Environment"})
                elif (intercept<=0 and m2>=m1):
                    df.loc[ (df['G']==gid), 'm1'] = m2
                    df.loc[ (df['G']==gid), 'b1'] = intercept
                    df.loc[ (df['G']==gid), 'target_m1'] = 0
                    df.loc[ (df['G']==gid), 'linecolor_m1'] = "cyan"
                    df.loc[ (df['G']==gid), 'environment_m1'] = "Below Average Poor and Above Good Environment"
                    GID_environments_m1.append({"method":'OLS', "Nursery":nursery, "G": gid, "m":m1, "b":intercept, "target":0, "linecolor":"cyan", "environment":"Below Average Poor and Above Good Environment"})
                else:
                    GID_environments_m1.append({"method":'OLS', "Nursery":nursery, "G": gid, "linecolor":"black", "environment":None})

            if ('RANSAC' in methods):
                # ------------------------------------
                # Method 2 - Robust regressor (RANSAC)
                # ------------------------------------
                #print(ransac.estimator_.intercept_, ransac.estimator_.coef_[0])
                intercept = ransac.estimator_.intercept_
                m2 = ransac.estimator_.coef_[0]
                if (intercept>0 and m2>=m1):
                    df.loc[ (df['G']==gid), 'm2'] = m2
                    df.loc[ (df['G']==gid), 'b2'] = intercept
                    df.loc[ (df['G']==gid), 'target_m2'] = 1
                    df.loc[ (df['G']==gid), 'linecolor_m2'] = "orange"
                    df.loc[ (df['G']==gid), 'environment_m2'] = "Above Average Poor and Good Environment"
                    GID_environments_m2.append({"method":'RANSAC', "Nursery":nursery, "G": gid, "m":m2, "b":intercept, "target":1, "linecolor":"orange", "environment":"Above Average Poor and Good Environment"})
                elif (intercept<0 and m1>=m2 ):
                    df.loc[ (df['G']==gid), 'm2'] = m2
                    df.loc[ (df['G']==gid), 'b2'] = intercept
                    df.loc[ (df['G']==gid), 'target_m2'] = 0
                    df.loc[ (df['G']==gid), 'linecolor_m2'] = "red"
                    df.loc[ (df['G']==gid), 'environment_m2'] = "Below Average Poor and Good Environment"
                    GID_environments_m2.append({"method":'RANSAC', "Nursery":nursery, "G": gid, "m":m2, "b":intercept, "target":0, "linecolor":"red", "environment":"Below Average Poor and Good Environment"})
                elif (intercept>=0 and m1>=m2):
                    df.loc[ (df['G']==gid), 'm2'] = m2
                    df.loc[ (df['G']==gid), 'b2'] = intercept
                    df.loc[ (df['G']==gid), 'target_m2'] = 1
                    df.loc[ (df['G']==gid), 'linecolor_m2'] = "purple"
                    df.loc[ (df['G']==gid), 'environment_m2'] = "Above Average Poor and Below Good Environment"
                    GID_environments_m2.append({"method":'RANSAC', "Nursery":nursery, "G": gid, "m":m2, "b":intercept, "target":1, "linecolor":"purple", "environment":"Above Average Poor and Below Good Environment"})
                elif (intercept<=0 and m2>=m1):
                    df.loc[ (df['G']==gid), 'm2'] = m2
                    df.loc[ (df['G']==gid), 'b2'] = intercept
                    df.loc[ (df['G']==gid), 'target_m2'] = 0
                    df.loc[ (df['G']==gid), 'linecolor_m2'] = "cyan"
                    df.loc[ (df['G']==gid), 'environment_m2'] = "Below Average Poor and Above Good Environment"
                    GID_environments_m2.append({"method":'RANSAC', "Nursery":nursery, "G": gid, "m":m2, "b":intercept, "target":0, "linecolor":"cyan", "environment":"Below Average Poor and Above Good Environment"})
                else:
                    GID_environments_m2.append({"method":'RANSAC', "Nursery":nursery, "G": gid, "m":m2, "b":intercept, "linecolor":"black", "environment":None})
                #
            #
        # Group
        GID_environments_m1 = pd.DataFrame(GID_environments_m1)
        GID_environments_m2 = pd.DataFrame(GID_environments_m2)
        GID_environments_Trial = pd.concat([GID_environments_Trial, GID_environments_m1, GID_environments_m2])
        # We are interested in the purple and yellow (orange) lines (GIDs)
        target_GIDs_m1 = dict(GID_environments_m1.query('target == 1')['G'].sort_values().value_counts()) 
        target_GIDs_m2 = dict(GID_environments_m2.query('target == 1')['G'].sort_values().value_counts()) 
        if (verbose is True):
            print(iwin.HT.bold +"{}".format(len(target_GIDs_m2)) + iwin.HT.reset + " GIDs out of " + iwin.HT.bold + "{}"
                  .format(len(target_GIDs_m1))+ iwin.HT.reset + " after applying a robust linear regression")
        target_GIDs_Trial.append({"Nursery":nursery, 'GID_method_1':target_GIDs_m1, 'GID_method_2': target_GIDs_m2 })

        del df_GY, df_gid, GID_environments_m1, GID_environments_m2, target_GIDs_m1, target_GIDs_m2
        gc.collect()
        return df, GID_environments_Trial, target_GIDs_Trial
    
    
    # *************************
    # Run Processes
    # *************************
    
    def getEnvironment(self, p, avgGrainYield):
        ''' Use cross product to determine whether a point lies above or below a line. '''
        
        if (p is None and avgGrainYield is None):
            print("Inputs not valid")
            return
        # 1:1 Line from point A to B
        a = np.array([0,0])
        b = np.array([40,40])
        res = np.cross(p-a, b-a) # Cross product
        if ( (p[0] <= avgGrainYield) and (res < 0) ):
            env = "Good in bad environment"
        elif ( (p[0] <= avgGrainYield) and (res >= 0) ):
            env = "Bad in bad environment"
        elif ( (p[0] > avgGrainYield) and (res < 0) ):
            env = "Good in good environment"
        elif ( (p[0] > avgGrainYield) and (res >= 0) ):
            env = "Bad in good environment"
        else:
            env = "Undefined"
        return env
    
    def getClass_old(self, p, avgGrainYield, avgEnv, target, qd=4):
        ''' Use cross product to determine whether a point lies above or below a line. '''

        if (p is None or avgGrainYield is None or avgEnv is None):
            print("Inputs not valid")
            return
        # 1:1 Line from point A to B
        a = np.array([0,0])
        b = np.array([40,40])
        res = np.cross(p-a, b-a) # Cross product
        YL, YM, YH = avgGrainYield
        EL, EM, EH = avgEnv
        # 4 Quadrants
        if (qd==4):
            avgGrainYield = YM
            avgEnv = EM
            # +-----+-----+
            # + 1LA + 2HA +
            # +-----+-----+
            # + 3LB + 4HB +
            # +-----+-----+
            if ((p[0] > avgGrainYield) & (p[1] <= avgEnv) ): # Quad 1
                env = "Above average in low yield environment" #"Good in bad environment"
            elif ((p[0] > avgGrainYield) & (p[1] > avgEnv) ): # Quad 2
                env = "Above average in high yield environment" # "Good in good environment"
                if (res >= 0): # below 1:1 line
                    env = "Above average in high yield environment - below 1:1 line"
                else:
                    env = "Above average in high yield environment - above 1:1 line"
            elif ((p[0] <= avgGrainYield) & (p[1] <= avgEnv) ): # Quad 3
                env = "Below average in low yield environment" #"Bad in bad environment"
                if (res >= 0): # below 1:1 line
                    env = "Below average in low yield environment - below 1:1 line"
                else:
                    env = "Below average in low yield environment - above 1:1 line"
            elif ((p[0] <= avgGrainYield) & (p[1] > avgEnv) ): # Quad 4
                env = "Below average in high yield environment" #"Bad in good environment"
            else:
                env = "Undefined"
        elif (qd==9): # 9 Quadrants
            #   E1    E2    E3    E4
            # Y4 +-----+-----+-----+
            #    + 7HL + 8HM + 9HH +
            # Y3 +-----+-----+-----+
            #    + 4ML + 5MM + 6MH + 
            # Y2 +-----+-----+-----+
            #    + 1LL + 2LM + 3LH + 
            # Y1 +-----+-----+-----+
            # 
            Yrng = YH - YL
            Erng = EH - EL
            Y = Yrng / 3
            E = Erng / 3
            # Yield
            Y1 = 0 #YL
            Y2 = YL + Y
            Y3 = YH - Y #YL + 2*Y
            Y4 = YH #YL + 3*Y
            # Environment
            E1 = 0 #EL
            E2 = EL + E
            E3 = EH - E
            E4 = EH 
            if ((p[0] > Y1) & (p[0] <= Y2) & (p[1] > E1) & (p[1] <= E2) ): # Quad 7LL
                env = "1LL"
            elif ((p[0] > Y1) & (p[0] <= Y2) & (p[1] > E2) & (p[1] <= E3) ): # Quad 2LM
                env = "2LM"
            elif ((p[0] > Y1) & (p[0] <= Y2) & (p[1] > E3) & (p[1] <= E4) ): # Quad 3LH
                env = "3LH"
            elif ((p[0] > Y2) & (p[0] <= Y3) & (p[1] > E1) & (p[1] <= E2) ): # Quad 4ML
                env = "4ML"
            elif ((p[0] > Y2) & (p[0] <= Y3) & (p[1] > E2) & (p[1] <= E3) ): # Quad 5MM
                env = "5MM"
            elif ((p[0] > Y2) & (p[0] <= Y3) & (p[1] > E3) & (p[1] <= E4) ): # Quad 6MH
                env = "6MH"
            elif ((p[0] > Y3) & (p[0] <= Y4) & (p[1] > E1) & (p[1] <= E2) ): # Quad 7HL
                env = "7HL"
            elif ((p[0] > Y3) & (p[0] <= Y4) & (p[1] > E2) & (p[1] <= E3) ): # Quad 8HM
                env = "8HM"
            elif ((p[0] > Y3) & (p[0] <= Y4) & (p[1] > E3) & (p[1] <= E4) ): # Quad 9HH
                env = "9HH"
            else:
                env = "Undefined"

        return env
    
    def getClass(self, p, avgGrainYield, avgEnv, target=None, qd=4):
        ''' Use cross product to determine whether a point lies above or below a line. '''

        if (p is None or avgGrainYield is None or avgEnv is None):
            print("Inputs not valid")
            return
        # 1:1 Line from point A to B
        a = np.array([0,0])
        b = np.array([40,40])
        res = np.cross(p-a, b-a) # Cross product
        YL, YM, YH = avgGrainYield
        EL, EM, EH = avgEnv
        # 4 Quadrants
        p0 = p[0] # Env
        p1 = p[1] # GY
        if (qd==4):
            avgGrainYield = EM
            avgEnv = YM
            # +-----+-----+
            # + 1AL + 2AH +
            # +-----+-----+
            # + 3BL + 4BH +
            # +-----+-----+
            if ((p1 > avgGrainYield) & (p0 <= avgEnv) ): # Quad 1
                env = "AL" #"Above average in low yield environment" #"Good in bad environment"
            elif ((p1 > avgGrainYield) & (p0 > avgEnv) ): # Quad 2
                env = "AH" #"Above average in high yield environment" # "Good in good environment"
                if (res >= 0): # below 1:1 line
                    env = "AH_B" #"Above average in high yield environment - below 1:1 line"
                else:
                    env = "AH_A" #"Above average in high yield environment - above 1:1 line"
            elif ((p1 <= avgGrainYield) & (p0 <= avgEnv) ): # Quad 3
                env = "BL" #"Below average in low yield environment" #"Bad in bad environment"
                if (res >= 0): # below 1:1 line
                    env = "BL_B" #"Below average in low yield environment - below 1:1 line"
                else:
                    env = "BL_A" #"Below average in low yield environment - above 1:1 line"
            elif ((p1 <= avgGrainYield) & (p0 > avgEnv) ): # Quad 4
                env = "BH" #"Below average in high yield environment" #"Bad in good environment"
            else:
                env = "Undefined"
        
        elif (qd==6): # 6 Quadrants
            #      Low  Medium High
            #    +-----+-----+-----+
            #    +  AL +  AM +  AH +
            #    +     +     +  +  +
            #    +     +  +  +     + 
            #    +  +  +     +     + 
            #    +  BL +  BM +  BH + 
            #    +-----+-----+-----+
            #   E1    E2    E3    E4
            #        Environments
            Yrng = YH - YL
            Erng = EH - EL
            Y = Yrng / 3
            E = Erng / 3
            # Environment
            E1 = 0 #EL
            E2 = EL + E
            E3 = EH - E
            E4 = EH 
            if ((p0 <= E2) and (res < 0)):
                env = "AL" # Above average in low yield environment "Good in bad environment"
            elif ((p0 <= E2) and (res >= 0)):
                env = "BL" # Below average in low yield environment "Bad in bad environment"
            elif ((p0 > E2) and (p0 <= E3) and (res < 0)):
                env = "AM" # Above average in medium yield environment
            elif ((p0 > E2) and (p0 <= E3) and (res >= 0)):
                env = "BM" # Below average in medium yield environment
            elif ((p0 > E3) and (p0 <= E4) and (res < 0)):
                env = "AH" # Above average in high yield environment "Good in good environment"
            elif ((p0 > E3) and (p0 <= E4) and (res >= 0)):
                env = "BH" # Below average in high yield environment "Bad in good environment"
            else:
                env = "Undefined"
        
        elif (qd==9): # 9 Quadrants
            #   E1    E2    E3    E4
            # Y4 +-----+-----+-----+
            #    + 7HL + 8HM + 9HH +
            # Y3 +-----+-----+-----+
            #    + 4ML + 5MM + 6MH + 
            # Y2 +-----+-----+-----+
            #    + 1LL + 2LM + 3LH + 
            # Y1 +-----+-----+-----+
            # 
            Yrng = YH - YL
            Erng = EH - EL
            Y = Yrng / 3
            E = Erng / 3
            # Yield
            Y1 = 0 #YL
            Y2 = YL + Y
            Y3 = YH - Y #YL + 2*Y
            Y4 = YH #YL + 3*Y
            # Environment
            E1 = 0 #EL
            E2 = EL + E
            E3 = EH - E
            E4 = EH 
            if ((p1 > Y1) & (p1 <= Y2) & (p0 > E1) & (p0 <= E2) ): # Quad 1LL
                env = "1LL"
            elif ((p1 > Y1) & (p1 <= Y2) & (p0 > E2) & (p0 <= E3) ): # Quad 2LM
                env = "2LM"
            elif ((p1 > Y1) & (p1 <= Y2) & (p0 > E3) & (p0 <= E4) ): # Quad 3LH
                env = "3LH"
            elif ((p1 > Y2) & (p1 <= Y3) & (p0 > E1) & (p0 <= E2) ): # Quad 4ML
                env = "4ML"
            elif ((p1 > Y2) & (p1 <= Y3) & (p0 > E2) & (p0 <= E3) ): # Quad 5MM
                env = "5MM"
            elif ((p1 > Y2) & (p1 <= Y3) & (p0 > E3) & (p0 <= E4) ): # Quad 6MH
                env = "6MH"
            elif ((p1 > Y3) & (p1 <= Y4) & (p0 > E1) & (p0 <= E2) ): # Quad 7HL
                env = "7HL"
            elif ((p1 > Y3) & (p1 <= Y4) & (p0 > E2) & (p0 <= E3) ): # Quad 8HM
                env = "8HM"
            elif ((p1 > Y3) & (p1 <= Y4) & (p0 > E3) & (p0 <= E4) ): # Quad 9HH
                env = "9HH"
            else:
                env = "Undefined"

        return env
    
    

    def _classifyAvgYieldbyGID(self, avgGY_2=None, fld1="AvGYxGID", fld2="AvGYxLocOcc"):
        '''
        
        '''
        df = avgGY_2.copy()
        df.dropna(subset=[fld1, fld2], inplace=True)
        avgGrainYield = df[fld2].mean()
        df['environment_m3'] = df[[fld1, fld2]].apply(lambda row: self.getEnvironment(row, avgGrainYield), axis=1)

        # Selected GIDs
        df['target_m3'] = 0
        df.loc[(df['environment_m3']=='Good in bad environment'), 'target_m3'] = 1
        df.loc[(df['environment_m3']=='Good in good environment'), 'target_m3'] = 1
        target_GIDs_m3 = df[( (df['environment_m3']=='Good in bad environment') 
             | (df['environment_m3']=='Good in good environment') )]

        return df, target_GIDs_m3
    
    def classifyAvgYieldbyGID_old(self, avgGY_2=None, fld1="AvGYxGID", fld2="AvGYxLocOcc", 
                              target=['LA','LB', 'HA', 'HB'], qd=4):
        '''

        '''
        if (target is None or len(target)<=0):
            print("Target is not valid. Please choose options between ['LA','LB', 'HA', 'HB', 'AL','BL', 'AH', 'BH']")
            return
        df = avgGY_2.copy()
        df.dropna(subset=[fld1, fld2], inplace=True)
        #avgGrainYield = df[fld1].mean()
        #avgEnv = df[fld2].mean()
        YL, YM, YH = df[fld1].min(), df[fld1].mean(), df[fld1].max()
        EL, EM, EH = df[fld2].min(), df[fld2].mean(), df[fld2].max()
        #print(YL, YM, YH, EL, EM, EH)
        df['environment_m3oldclass'] = df[[fld1, fld2]].apply(lambda row: self.getEnvironment(row, YM), axis=1)
        df['environment_m3'] = df[[fld1, fld2]].apply(lambda row: self.getClass(row, [YL, YM, YH], [EL, EM, EH], 
                                                                                target, qd), axis=1)

        # Selected GIDs
        df['target_m3'] = 0
        if (qd==4): # 4 Quadrants
            # +----+----+
            # + AL + AH +
            # +----+----+
            # + BL + BH +
            # +----+----+
            if ('LA' in target or 'AL' in target or 'Low Above' in target or 'Above Low' in target 
                or '1' in target or 1 in target): # Quadrant 1 - "Good in bad environment"
                df.loc[(df['environment_m3']=='Above average in low yield environment'), 'target_m3'] = 1

            if ('HA' in target or 'AH' in target or 'High Above' in target or 'Above High' in target 
                or '2' in target or 2 in target): # Quadrant 2 -  "Good in good environment"
                df.loc[(df['environment_m3']=='Above average in high yield environment'), 'target_m3'] = 1

            if ('LB' in target or 'BL' in target or 'Low Below' in target or 'Below Low' in target 
                or '3' in target or 3 in target): # Quadrant 3 - "Bad in bad environment"
                df.loc[(df['environment_m3']=='Below average in low yield environment'), 'target_m3'] = 1

            if ('HB' in target or 'BH' in target or 'High Below' in target or 'Below High' in target 
                or '4' in target or 4 in target): # Quadrant 4 - "Bad in good environment"
                df.loc[(df['environment_m3']=='Below average in high yield environment'), 'target_m3'] = 1

        elif (qd==9): # 9 Quadrants
            #   E1    E2    E3    E4
            # Y4 +-----+-----+-----+
            #    + 7HL + 8HM + 9HH +
            # Y3 +-----+-----+-----+
            #    + 4ML + 5MM + 6MH + 
            # Y2 +-----+-----+-----+
            #    + 1LL + 2LM + 3LH + 
            # Y1 +-----+-----+-----+
            if ('LL' in target or '1' in target or 1 in target): # Quad 7LL
                #env = "1LL"
                df.loc[(df['environment_m3']=='1LL'), 'target_m3'] = 1
            elif ('LM' in target or '2' in target or 2 in target): # Quad 2LM
                #env = "2LM"
                df.loc[(df['environment_m3']=='2LM'), 'target_m3'] = 1
            elif ('LH' in target or '3' in target or 3 in target): # Quad 3LH
                #env = "3LH"
                df.loc[(df['environment_m3']=='3LH'), 'target_m3'] = 1
            elif ('ML' in target or '4' in target or 4 in target): # Quad 4ML
                #env = "4ML"
                df.loc[(df['environment_m3']=='4ML'), 'target_m3'] = 1
            elif ('MM' in target or '5' in target or 5 in target): # Quad 5MM
                #env = "5MM"
                df.loc[(df['environment_m3']=='5MM'), 'target_m3'] = 1
            elif ('MH' in target or '6' in target or 6 in target): # Quad 6MH
                #env = "6MH"
                df.loc[(df['environment_m3']=='6MH'), 'target_m3'] = 1
            elif ('HL' in target or '7' in target or 7 in target): # Quad 7HL
                #env = "7HL"
                df.loc[(df['environment_m3']=='7HL'), 'target_m3'] = 1
            elif ('HM' in target or '8' in target or 8 in target): # Quad 8HM
                #env = "8HM"
                df.loc[(df['environment_m3']=='8HM'), 'target_m3'] = 1
            elif ('HH' in target or '9' in target or 9 in target): # Quad 9HH
                #env = "9HH"
                df.loc[(df['environment_m3']=='9HH'), 'target_m3'] = 1

        target_GIDs_m3 = df[df['target_m3']==1]

        return df, target_GIDs_m3
    
    
    
    def classifyAvgYieldbyGID(self, avgGY_2=None, fld1="AvGYxLocOcc", fld2="AvGYxGID", target=None, qd=4):
        '''

        '''
        if (target is None or len(target)<=0):
            # if q=6 => targets = ['AL','AM','AH','BL','BM','BH']
            # if q=9 => targets = ['LL','LM','LH','ML','MM','MH','HL','HM','HH']
            print("Target is not valid. Please choose options between [ 'AL','BH','AH_A', 'AH_B', 'BL_A', 'BL_B']")
            return
        df = avgGY_2.copy()
        df.dropna(subset=[fld1, fld2], inplace=True)
        #avgGrainYield = df[fld1].mean()
        #avgEnv = df[fld2].mean()
        EL, EM, EH = df[fld1].min(), df[fld1].mean(), df[fld1].max()
        YL, YM, YH = df[fld2].min(), df[fld2].mean(), df[fld2].max()
        #print(YL, YM, YH, EL, EM, EH)
        df['environment_m3oldclass'] = df[[fld1, fld2]].apply(lambda row: self.getEnvironment(row, YM), axis=1)
        df['environment_m3'] = df[[fld1, fld2]].apply(lambda row: self.getClass(row, [YL, YM, YH], 
                                                                                [EL, EM, EH], target, qd), axis=1)

        # Selected GIDs
        df['target_m3'] = 0
        if (qd==4): # 4 Quadrants
            # +----+----+
            # + AL + AH +
            # +----+----+
            # + BL + BH +
            # +----+----+
            # 'AL','BL', 'AH', 'BH','AH_A', 'AH_B', 'BL_A', 'BL_B'
            if ('AL' in target): # "Good in bad environment"
                #df.loc[(df['environment_m3']=='Above average in low yield environment'), 'target_m3'] = 1
                df.loc[(df['environment_m3']=='AL'), 'target_m3'] = 1
            if ('BH' in target ): # "Bad in good environment"
                #df.loc[(df['environment_m3']=='Below average in high yield environment'), 'target_m3'] = 1
                df.loc[(df['environment_m3']=='BH'), 'target_m3'] = 1

            # Estos no aparecen en environment_m3 nunca de acuerdo a la nueva clasificación
            #if ('AH' in target): # "Good in good environment"
            #    #df.loc[(df['environment_m3']=='Above average in high yield environment'), 'target_m3'] = 1
            #    df.loc[(df['environment_m3']=='AH'), 'target_m3'] = 1
            # Estos no aparecen en environment_m3 nunca de acuerdo a la nueva clasificación
            #if ('BL' in target): # "Bad in bad environment"
            #    #df.loc[(df['environment_m3']=='Below average in low yield environment'), 'target_m3'] = 1
            #    df.loc[(df['environment_m3']=='BL'), 'target_m3'] = 1

            if ('AH_A' in target ): # "Good in good environment  - above 1:1 line"
                df.loc[(df['environment_m3']=='AH_A'), 'target_m3'] = 1
            if ('AH_B' in target ): # "Good in good environment  - below 1:1 line"
                df.loc[(df['environment_m3']=='AH_B'), 'target_m3'] = 1
            if ('BL_A' in target ): # "Bad in bad environment  - above 1:1 line"
                df.loc[(df['environment_m3']=='BL_A'), 'target_m3'] = 1
            if ('BL_B' in target ): # "Bad in bad environment  - below 1:1 line"
                df.loc[(df['environment_m3']=='BL_B'), 'target_m3'] = 1
        
        elif (qd==6): # 6 Categories
            if ('AL' in target ): # "Good in low environment  - above 1:1 line"
                df.loc[(df['environment_m3']=='AL'), 'target_m3'] = 1
            if ('BL' in target ): # "Bad in low environment  - below 1:1 line"
                df.loc[(df['environment_m3']=='BL'), 'target_m3'] = 1
            if ('AM' in target ): # "Good in medium environment  - above 1:1 line"
                df.loc[(df['environment_m3']=='AM'), 'target_m3'] = 1
            if ('BM' in target ): # "Bad in medium environment  - below 1:1 line"
                df.loc[(df['environment_m3']=='BM'), 'target_m3'] = 1
            if ('AH' in target ): # "Good in high environment  - above 1:1 line"
                df.loc[(df['environment_m3']=='AH'), 'target_m3'] = 1
            if ('BH' in target ): # "Bad in high environment  - below 1:1 line"
                df.loc[(df['environment_m3']=='BH'), 'target_m3'] = 1
        
        elif (qd==9): # 9 Quadrants
            #   E1    E2    E3    E4
            # Y4 +-----+-----+-----+
            #    + 7HL + 8HM + 9HH +
            # Y3 +-----+-----+-----+
            #    + 4ML + 5MM + 6MH + 
            # Y2 +-----+-----+-----+
            #    + 1LL + 2LM + 3LH + 
            # Y1 +-----+-----+-----+
            if ('LL' in target or '1' in target or 1 in target): # Quad 7LL
                #env = "1LL"
                df.loc[(df['environment_m3']=='1LL'), 'target_m3'] = 1
            elif ('LM' in target or '2' in target or 2 in target): # Quad 2LM
                #env = "2LM"
                df.loc[(df['environment_m3']=='2LM'), 'target_m3'] = 1
            elif ('LH' in target or '3' in target or 3 in target): # Quad 3LH
                #env = "3LH"
                df.loc[(df['environment_m3']=='3LH'), 'target_m3'] = 1
            elif ('ML' in target or '4' in target or 4 in target): # Quad 4ML
                #env = "4ML"
                df.loc[(df['environment_m3']=='4ML'), 'target_m3'] = 1
            elif ('MM' in target or '5' in target or 5 in target): # Quad 5MM
                #env = "5MM"
                df.loc[(df['environment_m3']=='5MM'), 'target_m3'] = 1
            elif ('MH' in target or '6' in target or 6 in target): # Quad 6MH
                #env = "6MH"
                df.loc[(df['environment_m3']=='6MH'), 'target_m3'] = 1
            elif ('HL' in target or '7' in target or 7 in target): # Quad 7HL
                #env = "7HL"
                df.loc[(df['environment_m3']=='7HL'), 'target_m3'] = 1
            elif ('HM' in target or '8' in target or 8 in target): # Quad 8HM
                #env = "8HM"
                df.loc[(df['environment_m3']=='8HM'), 'target_m3'] = 1
            elif ('HH' in target or '9' in target or 9 in target): # Quad 9HH
                #env = "9HH"
                df.loc[(df['environment_m3']=='9HH'), 'target_m3'] = 1


        #target_GIDs_m3 = df[( (df['environment_m3']=='Good in bad environment') 
        #     | (df['environment_m3']=='Good in good environment') )]
        target_GIDs_m3 = df[df['target_m3']==1]

        return df, target_GIDs_m3
    
    
    def getNumOfOcurrences(self, df=None, nursery=None, top=10, target=None, qd=4):
        ''' Get number of occurrences in classification method '''
        if (df is None or nursery is None):
            return

        countOfGIDs, top_GIDs = None, None
        try:
            countOfGIDs = df[['G','AvGYxGID','AvGYxLocOcc', 'location','environment_m3']]\
            .groupby(['G', 'environment_m3'], as_index=False)\
            .agg({'AvGYxGID':'mean','AvGYxLocOcc':'mean','location':'count'})\
            .rename(columns={'location':'countOfOccurrences'})\
            .pivot_table(index=['G'],columns=['environment_m3']).reset_index()
            # rename columns
            countOfGIDs.columns = ['|'.join(col) for col in countOfGIDs.columns]
            #countOfGIDs = countOfGIDs.droplevel(level=1, axis=1)
            if (qd==4):
                countOfGIDs.rename(columns={
                    'G|' :'G',
                    'AvGYxGID|AH_A': 'AvGYxGID_AH_A', 'AvGYxGID|AH_B': 'AvGYxGID_AH_B', 
                    'AvGYxGID|AL': 'AvGYxGID_AL', 'AvGYxGID|BH': 'AvGYxGID_BH',
                    'AvGYxGID|BL_A': 'AvGYxGID_BL_A', 'AvGYxGID|BL_B': 'AvGYxGID_BL_B', 
                    'AvGYxLocOcc|AH_A': 'AvGYxLocOcc_AH_A', 'AvGYxLocOcc|AH_B': 'AvGYxLocOcc_AH_B', 
                    'AvGYxLocOcc|AL': 'AvGYxLocOcc_AL', 'AvGYxLocOcc|BH': 'AvGYxLocOcc_BH',
                    'AvGYxLocOcc|BL_A': 'AvGYxLocOcc_BL_A', 'AvGYxLocOcc|BL_B': 'AvGYxLocOcc_BL_B', 
                    'countOfOccurrences|AH_A':'countOfOccurrences_AH_A',
                    'countOfOccurrences|AH_B':'countOfOccurrences_AH_B', 
                    'countOfOccurrences|AL':'countOfOccurrences_AL',
                    'countOfOccurrences|BH':'countOfOccurrences_BH', 
                    'countOfOccurrences|BL_A':'countOfOccurrences_BL_A',
                    'countOfOccurrences|BL_B':'countOfOccurrences_BL_B',

                    'AvGYxGID|Above average in high yield environment - above 1:1 line': 'AvGYxGID_AH_A',
                    'AvGYxGID|Above average in high yield environment - below 1:1 line': 'AvGYxGID_AH_B',
                    'AvGYxGID|Above average in low yield environment': 'AvGYxGID_AL',
                    'AvGYxGID|Below average in high yield environment': 'AvGYxGID_BH',
                    'AvGYxGID|Below average in low yield environment - above 1:1 line': 'AvGYxGID_BL_A',
                    'AvGYxGID|Below average in low yield environment - below 1:1 line': 'AvGYxGID_BL_B',
                    'AvGYxLocOcc|Above average in high yield environment - above 1:1 line': 'AvGYxLocOcc_AH_A',
                    'AvGYxLocOcc|Above average in high yield environment - below 1:1 line': 'AvGYxLocOcc_AH_B',
                    'AvGYxLocOcc|Above average in low yield environment': 'AvGYxLocOcc_AL',
                    'AvGYxLocOcc|Below average in high yield environment': 'AvGYxLocOcc_BH',
                    'AvGYxLocOcc|Below average in low yield environment - above 1:1 line': 'AvGYxLocOcc_BL_A',
                    'AvGYxLocOcc|Below average in low yield environment - below 1:1 line': 'AvGYxLocOcc_BL_B',
                    'countOfOccurrences|Above average in high yield environment - above 1:1 line': 'countOfOccurrences_AH_A',
                    'countOfOccurrences|Above average in high yield environment - below 1:1 line': 'countOfOccurrences_AH_B',
                    'countOfOccurrences|Above average in low yield environment': 'countOfOccurrences_AL',
                    'countOfOccurrences|Below average in high yield environment': 'countOfOccurrences_BH',
                    'countOfOccurrences|Below average in low yield environment - above 1:1 line': 'countOfOccurrences_BL_A',
                    'countOfOccurrences|Below average in low yield environment - below 1:1 line': 'countOfOccurrences_BL_B'
                }, inplace=True)

                # Number of occurences for each GID
                top_GIDs_AL = countOfGIDs[['G', 'countOfOccurrences_AL']].sort_values(by=['countOfOccurrences_AL'], ascending=False)[:top]
                top_GIDs_AH_A = countOfGIDs[['G', 'countOfOccurrences_AH_A']].sort_values(by=['countOfOccurrences_AH_A'], ascending=False)[:top]
                top_GIDs_AH_B = countOfGIDs[['G', 'countOfOccurrences_AH_B']].sort_values(by=['countOfOccurrences_AH_B'], ascending=False)[:top]
                top_GIDs_BH = countOfGIDs[['G', 'countOfOccurrences_BH']].sort_values(by=['countOfOccurrences_BH'], ascending=False)[:top]
                top_GIDs_BL_A = countOfGIDs[['G', 'countOfOccurrences_BL_A']].sort_values(by=['countOfOccurrences_BL_A'], ascending=False)[:top]
                top_GIDs_BL_B = countOfGIDs[['G', 'countOfOccurrences_BL_B']].sort_values(by=['countOfOccurrences_BL_B'], ascending=False)[:top]

                top_GIDs = {
                    'AL': top_GIDs_AL,
                    'AH_A': top_GIDs_AH_A,
                    'AH_B': top_GIDs_AH_B,
                    'BH':top_GIDs_BH,
                    'BL_A':top_GIDs_BL_A,
                    'BL_B':top_GIDs_BL_B
                }
            
            elif (qd==6):
                countOfGIDs.rename(columns={
                    'G|' :'G',
                    'AvGYxGID|AL':'AvGYxGID_AL', 
                    'AvGYxGID|AM':'AvGYxGID_AM', 
                    'AvGYxGID|AH':'AvGYxGID_AH',
                    'AvGYxGID|BL':'AvGYxGID_BL', 
                    'AvGYxGID|BM':'AvGYxGID_BM', 
                    'AvGYxGID|BH':'AvGYxGID_BH',
                    'AvGYxLocOcc|AL':'AvGYxLocOcc_AL',
                    'AvGYxLocOcc|AM':'AvGYxLocOcc_AM', 
                    'AvGYxLocOcc|AH':'AvGYxLocOcc_AH', 
                    'AvGYxLocOcc|BL':'AvGYxLocOcc_BL', 
                    'AvGYxLocOcc|BM':'AvGYxLocOcc_BM',
                    'AvGYxLocOcc|BH':'AvGYxLocOcc_BH', 
                    'countOfOccurrences|AL':'countOfOccurrences_AL',
                    'countOfOccurrences|AM':'countOfOccurrences_AM', 
                    'countOfOccurrences|AH':'countOfOccurrences_AH',
                    'countOfOccurrences|BL':'countOfOccurrences_BL', 
                    'countOfOccurrences|BM':'countOfOccurrences_BM', 
                    'countOfOccurrences|BH':'countOfOccurrences_BH'
                }, inplace=True)

                # Number of occurences for each GID
                top_GIDs_AL = []
                top_GIDs_AM = []
                top_GIDs_AH = []
                top_GIDs_BL = []
                top_GIDs_BM = []
                top_GIDs_BH = []

                if ('countOfOccurrences_AH' in countOfGIDs.columns):
                    top_GIDs_AH = countOfGIDs[['G', 'countOfOccurrences_AH']].sort_values(by=['countOfOccurrences_AH'], ascending=False)[:top]
                if ('countOfOccurrences_AL' in countOfGIDs.columns):
                    top_GIDs_AL = countOfGIDs[['G', 'countOfOccurrences_AL']].sort_values(by=['countOfOccurrences_AL'], ascending=False)[:top]
                if ('countOfOccurrences_AM' in countOfGIDs.columns):
                    top_GIDs_AM = countOfGIDs[['G', 'countOfOccurrences_AM']].sort_values(by=['countOfOccurrences_AM'], ascending=False)[:top]
                if ('countOfOccurrences_BH' in countOfGIDs.columns):
                    top_GIDs_BH = countOfGIDs[['G', 'countOfOccurrences_BH']].sort_values(by=['countOfOccurrences_BH'], ascending=False)[:top]
                if ('countOfOccurrences_BL' in countOfGIDs.columns):
                    top_GIDs_BL = countOfGIDs[['G', 'countOfOccurrences_BL']].sort_values(by=['countOfOccurrences_BL'], ascending=False)[:top]
                if ('countOfOccurrences_BM' in countOfGIDs.columns):
                    top_GIDs_BM = countOfGIDs[['G', 'countOfOccurrences_BM']].sort_values(by=['countOfOccurrences_BM'], ascending=False)[:top]

                top_GIDs = {
                    'AL': top_GIDs_AL,
                    'AM': top_GIDs_AM,
                    'AH': top_GIDs_AH,
                    'BL': top_GIDs_BL,
                    'BM': top_GIDs_BM,
                    'BH': top_GIDs_BH
                }
            
            elif (qd==9):
                countOfGIDs.rename(columns={
                    'G|' :'G',
                    'AvGYxGID|1LL':'AvGYxGID_1LL', 
                    'AvGYxGID|2LM':'AvGYxGID_2LM', 
                    'AvGYxGID|3LH':'AvGYxGID_3LH',
                    'AvGYxGID|4ML':'AvGYxGID_4ML', 
                    'AvGYxGID|5MM':'AvGYxGID_5MM',
                    'AvGYxGID|6MH':'AvGYxGID_6MH', 
                    'AvGYxGID|7HL':'AvGYxGID_7HL',
                    'AvGYxGID|8HM':'AvGYxGID_8HM', 
                    'AvGYxGID|9HH':'AvGYxGID_9HH', 
                    'AvGYxLocOcc|1LL':'AvGYxLocOcc_1LL',
                    'AvGYxLocOcc|2LM':'AvGYxLocOcc_2LM', 
                    'AvGYxLocOcc|3LH':'AvGYxLocOcc_3LH',
                    'AvGYxLocOcc|4ML':'AvGYxLocOcc_4ML', 
                    'AvGYxLocOcc|5MM':'AvGYxLocOcc_5MM',
                    'AvGYxLocOcc|6MH':'AvGYxLocOcc_6MH', 
                    'AvGYxLocOcc|7HL':'AvGYxLocOcc_7HL',
                    'AvGYxLocOcc|8HM':'AvGYxLocOcc_8HM', 
                    'AvGYxLocOcc|9HH':'AvGYxLocOcc_9HH',
                    'countOfOccurrences|1LL':'countOfOccurrences_1LL', 
                    'countOfOccurrences|2LM':'countOfOccurrences_2LM',
                    'countOfOccurrences|3LH':'countOfOccurrences_3LH',
                    'countOfOccurrences|4ML':'countOfOccurrences_4ML', 
                    'countOfOccurrences|5MM':'countOfOccurrences_5MM',
                    'countOfOccurrences|6MH':'countOfOccurrences_6MH', 
                    'countOfOccurrences|7HL':'countOfOccurrences_7HL',
                    'countOfOccurrences|8HM':'countOfOccurrences_8HM',
                    'countOfOccurrences|9HH':'countOfOccurrences_9HH'
                }, inplace=True)
                
                # Number of occurences for each GID
                top_GIDs_LL = []
                top_GIDs_LM = []
                top_GIDs_LH = []
                top_GIDs_ML = []
                top_GIDs_MM = []
                top_GIDs_MH = []
                top_GIDs_HL = []
                top_GIDs_HM = []
                top_GIDs_HH = []
                
                if ('countOfOccurrences_1LL' in countOfGIDs.columns):
                    top_GIDs_LL = countOfGIDs[['G', 'countOfOccurrences_1LL']].sort_values(by=['countOfOccurrences_1LL'], ascending=False)[:top]
                
                if ('countOfOccurrences_2LM' in countOfGIDs.columns):
                    top_GIDs_LM = countOfGIDs[['G', 'countOfOccurrences_2LM']].sort_values(by=['countOfOccurrences_2LM'], ascending=False)[:top]
                    
                if ('countOfOccurrences_3LH' in countOfGIDs.columns):
                    top_GIDs_LH = countOfGIDs[['G', 'countOfOccurrences_3LH']].sort_values(by=['countOfOccurrences_3LH'], ascending=False)[:top]
                    
                if ('countOfOccurrences_4ML' in countOfGIDs.columns):
                    top_GIDs_ML = countOfGIDs[['G', 'countOfOccurrences_4ML']].sort_values(by=['countOfOccurrences_4ML'], ascending=False)[:top]
                
                if ('countOfOccurrences_5MM' in countOfGIDs.columns):
                    top_GIDs_MM = countOfGIDs[['G', 'countOfOccurrences_5MM']].sort_values(by=['countOfOccurrences_5MM'], ascending=False)[:top]
                
                if ('countOfOccurrences_6MH' in countOfGIDs.columns):
                    top_GIDs_MH = countOfGIDs[['G', 'countOfOccurrences_6MH']].sort_values(by=['countOfOccurrences_6MH'], ascending=False)[:top]
                    
                if ('countOfOccurrences_7HL' in countOfGIDs.columns):
                    top_GIDs_HL = countOfGIDs[['G', 'countOfOccurrences_7HL']].sort_values(by=['countOfOccurrences_7HL'], ascending=False)[:top]
                
                if ('countOfOccurrences_8HM' in countOfGIDs.columns):
                    top_GIDs_HM = countOfGIDs[['G', 'countOfOccurrences_8HM']].sort_values(by=['countOfOccurrences_8HM'], ascending=False)[:top]
                    
                if ('countOfOccurrences_9HH' in countOfGIDs.columns):
                    top_GIDs_HH = countOfGIDs[['G', 'countOfOccurrences_9HH']].sort_values(by=['countOfOccurrences_9HH'], ascending=False)[:top]
                
                top_GIDs = {
                    'LL': top_GIDs_LL,
                    'LM': top_GIDs_LM,
                    'LH': top_GIDs_LH,
                    'ML':top_GIDs_ML,
                    'MM':top_GIDs_MM,
                    'MH':top_GIDs_MH,
                    'HL':top_GIDs_HL,
                    'HM':top_GIDs_HM,
                    'HH':top_GIDs_HH
                }
                
                
        except Exception as err:
            print("Problem number of occurrences in classification method: {}. Error: {}".format(nursery, err))

        return countOfGIDs, top_GIDs

    def _getNumOfOcurrences(self, df=None, nursery=None, top=10):
        ''' Get number of occurrences in classification method'''
        if (df is None or nursery is None):
            return
        
        countOfGIDs, top_GIDs_GoodinBadEnv, top_GIDs_GoodinGoodEnv = None, None, None
        try:
            countOfGIDs = df[['G','AvGYxGID','AvGYxLocOcc', 'location','environment_m3']]\
            .groupby(['G', 'environment_m3'], as_index=False)\
            .agg({'AvGYxGID':'mean','AvGYxLocOcc':'mean','location':'count'})\
            .rename(columns={'location':'countOfOccurrences'})\
            .pivot_table(index=['G'],columns=['environment_m3']).reset_index()

            # 'Good in bad environment', 'Good in good environment'
            countOfGIDs.columns = ['G', 'AvGYxGID_GoodinBadEnv', 'AvGYxGID_GoodinGoodEnv',
                                   'AvGYxLocOcc_GoodinBadEnv', 'AvGYxLocOcc_GoodinGoodEnv',
                                   'countOfOccurrences_GoodinBadEnv', 'countOfOccurrences_GoodinGoodEnv'
                                  ]
            countOfGIDs['Nursery']= nursery

            # 5a. Number of occurrences for each GID for which yield is above average in “bad“ environments.
            top_GIDs_GoodinBadEnv = countOfGIDs[['G', 'countOfOccurrences_GoodinBadEnv']]\
            .sort_values(by=['countOfOccurrences_GoodinBadEnv'], ascending=False)[:top]
            # 5b. Number of occurrences for each GID that its yield is above average in “good“ environments.
            top_GIDs_GoodinGoodEnv = countOfGIDs[['G', 'countOfOccurrences_GoodinGoodEnv']]\
            .sort_values(by=['countOfOccurrences_GoodinGoodEnv'], ascending=False)[:top]
        except Exception as err:
            print("Problem number of occurrences in classification method: {}. Error: {}".format(nursery, err))
        
        return countOfGIDs, top_GIDs_GoodinBadEnv, top_GIDs_GoodinGoodEnv

    #
    def getNumOfOcurrences_old(self, df=None, nursery=None, top=10, target=['AL','BH', 'AH', 'BL' ], qd=4):
        ''' Get number of occurrences in classification method '''
        if (df is None or nursery is None):
            return

        countOfGIDs, top_GIDs = None, None
        try:
            countOfGIDs = df[['G','AvGYxGID','AvGYxLocOcc', 'location','environment_m3']]\
            .groupby(['G', 'environment_m3'], as_index=False)\
            .agg({'AvGYxGID':'mean','AvGYxLocOcc':'mean','location':'count'})\
            .rename(columns={'location':'countOfOccurrences'})\
            .pivot_table(index=['G'],columns=['environment_m3']).reset_index()
            # rename columns
            countOfGIDs.columns = ['|'.join(col) for col in countOfGIDs.columns]
            #countOfGIDs = countOfGIDs.droplevel(level=1, axis=1)
            if (qd==4):
                countOfGIDs.rename(columns={
                    'G|' :'G',
                    'AvGYxGID|Above average in high yield environment - above 1:1 line': 'AvGYxGID_AH_A',
                    'AvGYxGID|Above average in high yield environment - below 1:1 line': 'AvGYxGID_AH_B',
                    'AvGYxGID|Above average in low yield environment': 'AvGYxGID_AL',
                    'AvGYxGID|Below average in high yield environment': 'AvGYxGID_BH',
                    'AvGYxGID|Below average in low yield environment - above 1:1 line': 'AvGYxGID_BL_A',
                    'AvGYxGID|Below average in low yield environment - below 1:1 line': 'AvGYxGID_BL_B',
                    'AvGYxLocOcc|Above average in high yield environment - above 1:1 line': 'AvGYxLocOcc_AH_A',
                    'AvGYxLocOcc|Above average in high yield environment - below 1:1 line': 'AvGYxLocOcc_AH_B',
                    'AvGYxLocOcc|Above average in low yield environment': 'AvGYxLocOcc_AL',
                    'AvGYxLocOcc|Below average in high yield environment': 'AvGYxLocOcc_BH',
                    'AvGYxLocOcc|Below average in low yield environment - above 1:1 line': 'AvGYxLocOcc_BL_A',
                    'AvGYxLocOcc|Below average in low yield environment - below 1:1 line': 'AvGYxLocOcc_BL_B',
                    'countOfOccurrences|Above average in high yield environment - above 1:1 line': 'countOfOccurrences_AH_A',
                    'countOfOccurrences|Above average in high yield environment - below 1:1 line': 'countOfOccurrences_AH_B',
                    'countOfOccurrences|Above average in low yield environment': 'countOfOccurrences_AL',
                    'countOfOccurrences|Below average in high yield environment': 'countOfOccurrences_BH',
                    'countOfOccurrences|Below average in low yield environment - above 1:1 line': 'countOfOccurrences_BL_A',
                    'countOfOccurrences|Below average in low yield environment - below 1:1 line': 'countOfOccurrences_BL_B'
                }, inplace=True)
                # 5a. Number of occurences for each GID for which yield is above average in “bad“ environments.
                top_GIDs_AL = countOfGIDs[['G', 'countOfOccurrences_AL']]\
                .sort_values(by=['countOfOccurrences_AL'], ascending=False)[:top]
                # Number of occurences in other targets
                top_GIDs_AH_A = countOfGIDs[['G', 'countOfOccurrences_AH_A']]\
                .sort_values(by=['countOfOccurrences_AH_A'], ascending=False)[:top]
                top_GIDs_AH_B = countOfGIDs[['G', 'countOfOccurrences_AH_B']]\
                .sort_values(by=['countOfOccurrences_AH_B'], ascending=False)[:top]
                top_GIDs_BH = countOfGIDs[['G', 'countOfOccurrences_BH']]\
                .sort_values(by=['countOfOccurrences_BH'], ascending=False)[:top]
                top_GIDs_BL_A = countOfGIDs[['G', 'countOfOccurrences_BL_A']]\
                .sort_values(by=['countOfOccurrences_BL_A'], ascending=False)[:top]
                top_GIDs_BL_B = countOfGIDs[['G', 'countOfOccurrences_BL_B']]\
                .sort_values(by=['countOfOccurrences_BL_B'], ascending=False)[:top]

                top_GIDs = {
                    'AL': top_GIDs_AL,
                    'AH_A': top_GIDs_AH_A,
                    'AH_B': top_GIDs_AH_B,
                    'BH':top_GIDs_BH,
                    'BL_A':top_GIDs_BL_A,
                    'BL_B':top_GIDs_BL_B
                }
        except Exception as err:
            print("Problem number of occurrences in classification method: {}. Error: {}".format(nursery, err))

        return countOfGIDs, top_GIDs

    # Methods 1 and 2
    def processLRMethods(self, df=None, nursery=None, dirname='./', fmt='pdf', saveFig=True, showFig=True, verbose=True ):
        ''' Process and generate a figure using the classification method proposed by Urs '''
        if (df is None):
            df = self.avgGY.copy()
            if (df is None):
                print("Input data not valid")
                return
        if (nursery is None):
            nursery = self.nursery
            if (nursery is None):
                print("Nursery is not valid")
                return
        df, GID_environments_Trial, target_GIDs_Trial = \
        self.genotypeSelection_byLinearRegression(df_GY=df, nursery=nursery, methods=['OLS', 'RANSAC'], verbose=verbose)

        # Most common GIDs in both methods
        #avgGY_Nursery_Year_Loc_Occ[( (avgGY_Nursery_Year_Loc_Occ['target_m1']==1) & 
        #                             (avgGY_Nursery_Year_Loc_Occ['target_m2']==1) )].reset_index(drop=True)
        if (saveFig is True or showFig is True):
            figures.figure_AvgYieldbyGID_LR(avgGY_1=df, nursery=nursery, hue='G', hue2=None, hue3=None, 
                                    hue4='G', lw=0.8, s=10, alpha=.45, alpha2=.85, s4=20, alpha4=.95, 
                                    loc=2, ncol=4, methods=['OLS', 'RANSAC'], fld1="AvGYxLocOcc", fld2="AvGYxGID",  
                                    saveFig=saveFig, showFig=showFig, dirname=dirname, fmt=fmt)
        
        self.avgGY = df
        return df, GID_environments_Trial, target_GIDs_Trial

    # Method 3
    def _processClassificationMethod(self, df=None, nursery=None, threshold=10, dirname='./', fmt='pdf', 
                                    saveFig=True, showFig=True, verbose=False ):
        '''
            Classify nurseries by avg grain yield
            
        '''
        if (df is None):
            df = self.avgGY
            if (df is None):
                print("Input data not valid")
                return
        if (nursery is None):
            nursery = self.nursery
            if (nursery is None):
                print("Nursery is not valid")
                return
        df, target_GIDs_m3 = self.classifyAvgYieldbyGID(df)
        # Count GIDs only on the selected environments
        countOfGIDs, top_GIDs_GoodinBadEnv, top_GIDs_GoodinGoodEnv = \
        self.getNumOfOcurrences(df=target_GIDs_m3, nursery=nursery, top=threshold)

        # ------------------------------------
        # Get selected GIDs - Method 3 (Urs)
        # ------------------------------------
        selected_GIDs_m3 = df[( (df['G'].isin(top_GIDs_GoodinBadEnv['G'].unique()) ) 
                               | (df['G'].isin(top_GIDs_GoodinGoodEnv['G'].unique()) )
                    )].reset_index(drop=True)
        countOfGIDs_topGinBEnv = countOfGIDs[countOfGIDs['G'].isin(top_GIDs_GoodinBadEnv['G'].unique())].reset_index(drop=True)
        countOfGIDs_topGinBEnv['environment_m3'] = 'Good in bad environment'
        countOfGIDs_topGinGEnv = countOfGIDs[countOfGIDs['G'].isin(top_GIDs_GoodinGoodEnv['G'].unique())].reset_index(drop=True)
        countOfGIDs_topGinGEnv['environment_m3'] = 'Good in good environment'
        numGIDsOfOcurrences = pd.concat([countOfGIDs_topGinBEnv, countOfGIDs_topGinGEnv])

        if (saveFig is True or showFig is True):
            figures.figure_AvgYieldbyGID_classify(avgGY_1=df, avgGY_2=target_GIDs_m3, df_countOfGIDs=countOfGIDs, 
                                      topGinBEnv=top_GIDs_GoodinBadEnv, topGinGEnv=top_GIDs_GoodinGoodEnv,
                                      nursery=nursery, threshold=threshold,
                                      hue='G', hue2=None, hue3=None, hue4='G', lw=0.8, s=10, alpha=.45, 
                                      alpha2=.85, s4=20, alpha4=.95, loc=2, ncol=4, fld1="AvGYxGID", 
                                      fld2="AvGYxLocOcc", dispTxt=True, saveFig=saveFig, showFig=showFig,
                                                 dirname=dirname, fmt=fmt)
        
        self.avgGY = df
        self.selected_GIDs_m3 = selected_GIDs_m3
        return df, selected_GIDs_m3, numGIDsOfOcurrences

    #
    def processClassificationMethod_old(self, df=None, nursery=None, target=['LA','LB', 'HA', 'HB'], qd=9,
                                    threshold=10, dirname='./', fmt='pdf', 
                                    saveFig=True, showFig=True, verbose=False ):
        '''
            Classify nurseries by avg grain yield

        '''
        if (df is None):
            df = self.avgGY
            if (df is None):
                print("Input data not valid")
                return
        if (nursery is None):
            nursery = self.nursery
            if (nursery is None):
                print("Nursery is not valid")
                return
        if (target is None or len(target)<=0):
            print("Target is not valid. Please choose options between ['LA','LB', 'HA', 'HB']")
            return

        df, target_GIDs_m3 = self.classifyAvgYieldbyGID(df, target=target, qd=qd)
        countOfGIDs, top_GIDs = self.getNumOfOcurrences(df, nursery=nursery, top=threshold, target=target, qd=qd)
        # ------------------------------------
        # Get selected GIDs - Method 3 (Urs)
        # ------------------------------------
        selected_GIDs_m3 = df[( (df['G'].isin(top_GIDs['AL']['G'].unique()) ) 
                               | (df['G'].isin(top_GIDs['BH']['G'].unique()) )
                    )].reset_index(drop=True)

        countOfGIDs_topGinBEnv = countOfGIDs[countOfGIDs['G'].isin(top_GIDs['AL']['G'].unique())].reset_index(drop=True)
        countOfGIDs_topGinBEnv['environment_m3oldclass'] = 'Good in bad environment'
        countOfGIDs_topGinBEnv['environment_m3'] = 'Above average in low yield environment'
        countOfGIDs_topGinGEnv = countOfGIDs[countOfGIDs['G'].isin(top_GIDs['BH']['G'].unique())].reset_index(drop=True)
        countOfGIDs_topGinGEnv['environment_m3oldclass'] = 'Bad in good environment'
        countOfGIDs_topGinGEnv['environment_m3'] = 'Below average in high yield environment'
        numGIDsOfOcurrences = pd.concat([countOfGIDs_topGinBEnv, countOfGIDs_topGinGEnv])

        if (saveFig is True or showFig is True):
            figures.figure_AvgYieldbyGID_classify_v2(avgGY_1=df, avgGY_2=target_GIDs_m3, df_countOfGIDs=countOfGIDs, 
                                             topGinBEnv=top_GIDs['AL'], topGinGEnv=top_GIDs['BH'],
                                             nursery=nursery, threshold=threshold,
                                             hue='G', hue2=None, hue3=None, hue4='G', lw=0.8, s=10, alpha=.45, 
                                             alpha2=.85, s4=45, alpha4=.95, loc=2, ncol=4, fld1="AvGYxGID", 
                                             fld2="AvGYxLocOcc", dispTxt=True, 
                                             saveFig=saveFig, showFig=showFig, dirname=dirname, fmt=fmt)

        self.avgGY = df
        self.selected_GIDs_m3 = selected_GIDs_m3
        return df, target_GIDs_m3, numGIDsOfOcurrences
    
    
    def processClassificationMethod(self, df=None, nursery=None, 
                                target=['AL', 'BH','AH_A', 'AH_B', 'BL_A', 'BL_B'], qd=4,
                                threshold=10, dirname='./', fmt='pdf', plot_params=None,
                                saveFig=True, showFig=True, verbose=False ):
        '''
            Classify nurseries by avg grain yield

        '''
        if (df is None):
            df = self.avgGY
            if (df is None):
                print("Input data not valid")
                return
        if (nursery is None):
            nursery = self.nursery
            if (nursery is None):
                print("Nursery is not valid")
                return
        if (target is None or len(target)<=0):
            print("Target is not valid. Please choose options between ['AL', 'BH','AH_A', 'AH_B', 'BL_A', 'BL_B']")
            return

        df, target_GIDs_m3 = self.classifyAvgYieldbyGID(df, target=target, qd=qd)
        countOfGIDs, top_GIDs = self.getNumOfOcurrences(df, nursery=nursery, top=threshold, target=target, qd=qd)
        # ------------------------------------
        # Get selected GIDs - Method 3 (Urs)
        #1. above average in bad environment and below average in good environment (AL, BH, BL_A, AH_B)
        #2. above average in bad environment and above average in good environment
        #3. below average in bad environment and below average in good environment
        #4. below average in bad environment and above average in good environment
        # ------------------------------------
        #selected_GIDs_m3 = df[( (df['G'].isin(top_GIDs['AL']['G'].unique()) ) | (df['G'].isin(top_GIDs['BH']['G'].unique()) )
        #            )].reset_index(drop=True)
        
        def getCountOfGIDs_targetGIDS(df2, tg, top_GIDs, countOfGIDs):
            _gids, countOfGIDs_tg = [], None
            if (len(top_GIDs[tg])>0):
                _gids = df2[df2['G'].isin(top_GIDs[tg]['G'].unique())]['G'].to_numpy()
                _gids = np.unique(np.array(_gids))
                countOfGIDs_tg = countOfGIDs[countOfGIDs['G'].isin(_gids)].reset_index(drop=True)
            return _gids, countOfGIDs_tg

        if (qd==4):
            target_GIDS = []
            AL_gids, countOfGIDs_AL = None, None
            BH_gids, countOfGIDs_BH = None, None
            AH_A_gids, countOfGIDs_AH_A = None, None
            AH_B_gids, countOfGIDs_AH_B = None, None
            BL_A_gids, countOfGIDs_BL_A = None, None
            BL_B_gids, countOfGIDs_BL_B = None, None
            if ('AL' in target and 'AL' in top_GIDs.keys()):
                AL_gids = df[df['G'].isin(top_GIDs['AL']['G'].unique())]['G'].to_numpy()
                AL_gids = np.unique(np.array(AL_gids))
                countOfGIDs_AL = countOfGIDs[countOfGIDs['G'].isin(AL_gids)].reset_index(drop=True)
                target_GIDS.append(AL_gids)
            #if ('BL' in target and 'BL' in top_GIDs.keys()):
            #    BL_gids = df['G'].isin(top_GIDs['BL']['G'].unique())
            #    target_GIDS.append(BL_gids)
            #if ('AH' in target and 'AH' in top_GIDs.keys()):
            #    AH_gids = df['G'].isin(top_GIDs['AH']['G'].unique())
            #    target_GIDS.append(AH_gids)
            if ('BH' in target and 'BH' in top_GIDs.keys()):
                BH_gids = df[df['G'].isin(top_GIDs['BH']['G'].unique())]['G'].to_numpy()
                BH_gids = np.unique(np.array(BH_gids))
                countOfGIDs_BH = countOfGIDs[countOfGIDs['G'].isin(BH_gids)].reset_index(drop=True)
                target_GIDS.append(BH_gids)
            if ('AH_A' in target and 'AH_A' in top_GIDs.keys()):
                AH_A_gids = df[df['G'].isin(top_GIDs['AH_A']['G'].unique())]['G'].to_numpy()
                AH_A_gids = np.unique(np.array(AH_A_gids))
                countOfGIDs_AH_A = countOfGIDs[countOfGIDs['G'].isin(AH_A_gids)].reset_index(drop=True)
                target_GIDS.append(AH_A_gids)
            if ('AH_B' in target and 'AH_B' in top_GIDs.keys()):
                AH_B_gids = df[df['G'].isin(top_GIDs['AH_B']['G'].unique())]['G'].to_numpy()
                AH_B_gids = np.unique(np.array(AH_B_gids))
                countOfGIDs_AH_B = countOfGIDs[countOfGIDs['G'].isin(AH_B_gids)].reset_index(drop=True)
                target_GIDS.append(AH_B_gids)
            if ('BL_A' in target and 'BL_A' in top_GIDs.keys()):
                BL_A_gids = df[df['G'].isin(top_GIDs['BL_A']['G'].unique())]['G'].to_numpy()
                BL_A_gids = np.unique(np.array(BL_A_gids))
                countOfGIDs_BL_A = countOfGIDs[countOfGIDs['G'].isin(BL_A_gids)].reset_index(drop=True)
                target_GIDS.append(BL_A_gids)
            if ('BL_B' in target and 'BL_B' in top_GIDs.keys()):
                BL_B_gids = df[df['G'].isin(top_GIDs['BL_B']['G'].unique())]['G'].to_numpy()
                BL_B_gids = np.unique(np.array(BL_B_gids))
                countOfGIDs_BL_B = countOfGIDs[countOfGIDs['G'].isin(BL_B_gids)].reset_index(drop=True)
                target_GIDS.append(BL_B_gids)

            target_GIDS = np.array(target_GIDS).flatten()
            mask_target_GIDs = ( df['G'].isin(target_GIDS) )
            selected_GIDs_m3 = df[ mask_target_GIDs].reset_index(drop=True)

            #
            topGinBEnv = []
            topBinGEnv = []
            if ((AL_gids is not None) and (BL_A_gids is not None)):
                topGinBEnv = np.unique(np.asarray([AL_gids, BL_A_gids]).flatten())
            if ((BH_gids is not None) and (AH_B_gids is not None)):
                topBinGEnv = np.unique(np.asarray([BH_gids, AH_B_gids]).flatten())
            # 
            countOfGIDs_topGinBEnv = countOfGIDs[countOfGIDs['G'].isin(topGinBEnv)].reset_index(drop=True)
            countOfGIDs_topBinGEnv = countOfGIDs[countOfGIDs['G'].isin(topBinGEnv)].reset_index(drop=True)
            countOfGIDs_topGinBEnv['environment_m3oldclass'] = 'Good in bad environment'
            countOfGIDs_topBinGEnv['environment_m3oldclass'] = 'Bad in good environment'
            countOfGIDs_topGinBEnv['environment_m3'] = '# Occurrences above average in low environment'
            countOfGIDs_topBinGEnv['environment_m3'] = '# Occurrences below average in high environment'

            numGIDsOfOcurrences = pd.concat([countOfGIDs_topGinBEnv, countOfGIDs_topBinGEnv])
            numGIDsOfOcurrences['numOcurrences_target'] = numGIDsOfOcurrences['countOfOccurrences_AL']\
            + numGIDsOfOcurrences['countOfOccurrences_BL_A'] + numGIDsOfOcurrences['countOfOccurrences_BH']\
            + numGIDsOfOcurrences['countOfOccurrences_AH_B']

            numGIDsOfOcurrences['numOcurrences_avgGY'] = (numGIDsOfOcurrences['AvGYxLocOcc_AL']\
            + numGIDsOfOcurrences['AvGYxLocOcc_BL_A'] + numGIDsOfOcurrences['AvGYxLocOcc_BH']\
            + numGIDsOfOcurrences['AvGYxLocOcc_AH_B']) / 4.0
        
        if (qd==6):
            target_GIDS = []
            numGIDsOfOcurrences = None
            AL_gids, countOfGIDs_AL = getCountOfGIDs_targetGIDS(df, 'AL', top_GIDs, countOfGIDs)
            AM_gids, countOfGIDs_AM = getCountOfGIDs_targetGIDS(df, 'AM', top_GIDs, countOfGIDs)
            AH_gids, countOfGIDs_AH = getCountOfGIDs_targetGIDS(df, 'AH', top_GIDs, countOfGIDs)
            BL_gids, countOfGIDs_BL = getCountOfGIDs_targetGIDS(df, 'BL', top_GIDs, countOfGIDs)
            BM_gids, countOfGIDs_BM = getCountOfGIDs_targetGIDS(df, 'BM', top_GIDs, countOfGIDs)
            BH_gids, countOfGIDs_BH = getCountOfGIDs_targetGIDS(df, 'BH', top_GIDs, countOfGIDs)

            if (countOfGIDs_AL is not None): countOfGIDs_AL['environment_m3'] = '# Occurrences above average in low environment'
            if (countOfGIDs_AM is not None): countOfGIDs_AM['environment_m3'] = '# Occurrences above average in medium environment'
            if (countOfGIDs_AH is not None): countOfGIDs_AH['environment_m3'] = '# Occurrences above average in high environment'
            if (countOfGIDs_BL is not None): countOfGIDs_BL['environment_m3'] = '# Occurrences below average in low environment'
            if (countOfGIDs_BM is not None): countOfGIDs_BM['environment_m3'] = '# Occurrences below average in medium environment'
            if (countOfGIDs_BH is not None): countOfGIDs_BH['environment_m3'] = '# Occurrences below average in high environment'

            if ('AL' in target and 'AL' in top_GIDs.keys() and len(AL_gids)>0 ):
                target_GIDS.append(AL_gids)
                # Target of the study - Good in low environments
                numGIDsOfOcurrences = countOfGIDs_AL
                numGIDsOfOcurrences['numOcurrences_target'] = countOfGIDs_AL['countOfOccurrences_AL']
                numGIDsOfOcurrences['numOcurrences_avgGY'] = countOfGIDs_AL['AvGYxLocOcc_AL']
            if ('AM' in target and 'AM' in top_GIDs.keys() and len(AM_gids)>0 ):
                target_GIDS.append(AM_gids)
                numGIDsOfOcurrences = countOfGIDs_AM
                numGIDsOfOcurrences['numOcurrences_target'] = countOfGIDs_AM['countOfOccurrences_AM']
                numGIDsOfOcurrences['numOcurrences_avgGY'] = countOfGIDs_AM['AvGYxLocOcc_AM']
            if ('AH' in target and 'AH' in top_GIDs.keys() and len(AH_gids)>0 ):
                target_GIDS.append(AH_gids)
                numGIDsOfOcurrences = countOfGIDs_AH
                numGIDsOfOcurrences['numOcurrences_target'] = countOfGIDs_AH['countOfOccurrences_AH']
                numGIDsOfOcurrences['numOcurrences_avgGY'] = countOfGIDs_AH['AvGYxLocOcc_AH']
            if ('BL' in target and 'BL' in top_GIDs.keys() and len(BL_gids)>0 ):
                target_GIDS.append(BL_gids)
                numGIDsOfOcurrences = countOfGIDs_BL
                numGIDsOfOcurrences['numOcurrences_target'] = countOfGIDs_BL['countOfOccurrences_BL']
                numGIDsOfOcurrences['numOcurrences_avgGY'] = countOfGIDs_BL['AvGYxLocOcc_BL']
            if ('BM' in target and 'BM' in top_GIDs.keys() and len(BM_gids)>0 ):
                target_GIDS.append(BM_gids)
                numGIDsOfOcurrences = countOfGIDs_BM
                numGIDsOfOcurrences['numOcurrences_target'] = countOfGIDs_BM['countOfOccurrences_BM']
                numGIDsOfOcurrences['numOcurrences_avgGY'] = countOfGIDs_BM['AvGYxLocOcc_BM']
            if ('BH' in target and 'BH' in top_GIDs.keys() and len(BH_gids)>0 ):
                target_GIDS.append(BH_gids)
                numGIDsOfOcurrences = countOfGIDs_BH
                numGIDsOfOcurrences['numOcurrences_target'] = countOfGIDs_BH['countOfOccurrences_BH']
                numGIDsOfOcurrences['numOcurrences_avgGY'] = countOfGIDs_BH['AvGYxLocOcc_BH']

            target_GIDS = np.array(target_GIDS).flatten()
            mask_target_GIDs = ( df['G'].isin(target_GIDS) )
            selected_GIDs_m3 = df[ mask_target_GIDs].reset_index(drop=True)
            

        if (saveFig is True or showFig is True):
            if (plot_params is None):
                plot_params_4 = dict(showTargetText=True, ncol=4, target_fontsize=9)
            try:
                figures.plot_AvgYieldbyGID_classify(avgGY_1=df, selGIDs_m3=selected_GIDs_m3,
                                                numGIDsOfOcurrences=numGIDsOfOcurrences, 
                                                nursery=nursery, threshold=threshold,
                                                title='', ptype=4, qd=qd, target=target, plot_params=plot_params_4, 
                                                saveFig=saveFig, showFig=showFig, dirname=dirname, fmt=fmt)
            except Exception as err:
                print("Error in plot_AvgYieldbyGID_classify: ", err)
                pass

        self.avgGY = df
        self.selected_GIDs_m3 = selected_GIDs_m3
        return df, selected_GIDs_m3, numGIDsOfOcurrences

    #
    def commonGIDsinMethods(self, df_raw=None, df=None, selGIDs_m3=None, nursery=None,
                            lw=0.8, hue='G', s=65, alpha=.85, loc=2, ncol=1, xt_tl=.01, yt_tl=.99, ha='left', va='top',
                            dirname='./', fmt='pdf', showFig=True, saveFig=True, verbose=False):
        ''' 
            Combine all of the methods applied to the genotypes selection 
            
        '''
        if (df_raw is None):
            df_raw = self.data
            if (df_raw is None):
                print("Input raw data not valid")
                return
        if (df is None):
            df = self.avgGY.copy()
            if (df is None):
                print("Input data not valid")
                return
        if (selGIDs_m3 is None):
            selGIDs_m3 = self.selected_GIDs_m3
            if (selGIDs_m3 is None):
                print("Selected GIDs using method 3 are not valid")
                return
        if (nursery is None):
            nursery = self.nursery
            if (nursery is None):
                print("Nursery is not valid")
                return
            
        #df = df_final.copy()
        # Methods 1 & 2
        sel_GIDs_m1m2 = df[(
            (df['target_m1']==1) & (df['target_m2']==1)
        )].reset_index(drop=True)

        # Method 3
        sel_GIDs_m3 = df[(
            (df['target_m3']==1) & (df['G'].isin(selGIDs_m3['G'].unique()))
        )].reset_index(drop=True)

        # Combine all methods
        sel_GIDs_m1m2m3 = df[(
            (df['target_m1']==1) | (df['target_m2']==1) |
            ((df['target_m3']==1) & (df['G'].isin(selGIDs_m3['G'].unique())))
        )].reset_index(drop=True)

        unique_selected_GIDs_in_m1m2m3 = df[(
            (df['target_m1']==1) & (df['target_m2']==1) &
            ((df['target_m3']==1) & (df['G'].isin(selGIDs_m3['G'].unique())))
        )].reset_index(drop=True)
        
        #unique_selected_GIDs_in_m1m2m3['E'] = unique_selected_GIDs_in_m1m2m3['location'].astype(str) + '_' + unique_selected_GIDs_in_m1m2m3['Occ'].astype(str)

        # --------------------------
        # Wheat lines with yield better performance in poor/good environments for each trial
        # --------------------------
        # Include all GIDs per Loc-Occ to avoid gaps in GY
        #Gtemp = df_raw[df_raw['G'].isin(unique_selected_GIDs_in_m1m2m3['G'].unique())]
        #Gtemp['E'] = Gtemp['location'].astype(str) + '_' + Gtemp['Occ'].astype(str)
        #Gtemp = Gtemp[Gtemp['E'].isin(unique_selected_GIDs_in_m1m2m3['E'].unique())]

        sel_cols = ['G', 'location', 'Occ','AvGYxLocOcc', 'AvGYxGID',  
                    'environment_m1', 'environment_m2', 'environment_m3' 
                   ] # 
        final_selected_GIDs = pd.merge(unique_selected_GIDs_in_m1m2m3[sel_cols], df_raw, 
                                       how='left', on=['G', 'location', 'Occ']) # 
        # Extract grand parents
        final_selected_GIDs['GrandParent'] = final_selected_GIDs['Pedigree'].apply(lambda x: x.split('//')[0])

        if (saveFig is True or showFig is True):
            figures.figure_AvgYieldbyGID_combineMethods(df_GY=final_selected_GIDs, nursery=nursery,
                              fld1="AvGYxLocOcc", fld2="AvGYxGID", lw=lw, hue=hue, s=s, alpha=alpha, loc=loc, ncol=ncol, 
                              xt_tl=xt_tl, yt_tl=yt_tl, ha=ha, va=va, showFig=showFig, saveFig=saveFig, 
                                                        dirname=dirname, fmt=fmt )
        
        self.final_selected_GIDs = final_selected_GIDs
        del df
        _ = gc.collect()
        return final_selected_GIDs, sel_GIDs_m1m2, sel_GIDs_m3, sel_GIDs_m1m2m3, unique_selected_GIDs_in_m1m2m3

    
    def commonGIDsinMethods_v2(self, df_raw=None, df=None, selGIDs_m3=None, nursery=None, bothEnvs=True,
                        lw=0.8, hue='G', s=65, alpha=.85, loc=2, ncol=1, xt_tl=.01, yt_tl=.99, ha='left', va='top',
                        dirname='./', fmt='pdf', showFig=True, saveFig=True, verbose=False):
        ''' 
            Combine all of the methods applied to the genotypes selection 

        '''
        if (df_raw is None):
            df_raw = self.data
            if (df_raw is None):
                print("Input raw data not valid")
                return
        if (df is None):
            df = self.avgGY.copy()
            if (df is None):
                print("Input data not valid")
                return
        if (selGIDs_m3 is None):
            selGIDs_m3 = self.selected_GIDs_m3
            if (selGIDs_m3 is None):
                print("Selected GIDs using method 3 are not valid")
                return
        if (nursery is None):
            nursery = self.nursery
            if (nursery is None):
                print("Nursery is not valid")
                return

        #df = df_final.copy()
        # Intersection or Combined Methods
        #target_1 = df[(
        #    (df['linecolor_m1']=='purple') & (df['linecolor_m2']=='purple') & (df['target_m3']==1) # AL
        #)]

        # Methods 1 & 2
        # Due to we need to select only one target - Above average yield in poor environments (purple lines in charts)
        #sel_GIDs_m1m2 = df[(
        #    (df['target_m1']==1) & (df['target_m2']==1)
        #)].reset_index(drop=True)
        sel_GIDs_m1m2 = df[(
            (df['linecolor_m1']=='purple') & (df['linecolor_m2']=='purple')
        )].reset_index(drop=True)

        # Method 3
        # This method propose by Urs not only select purple lines but also the others
        sel_GIDs_m3 = df[(
            (df['target_m3']==1) & (df['G'].isin(selGIDs_m3['G'].unique()))
        )].reset_index(drop=True)

        # Combine all methods
        sel_GIDs_m1m2m3 = df[(
            (df['linecolor_m1']=='purple') | (df['linecolor_m2']=='purple') |
            ((df['target_m3']==1) & (df['G'].isin(selGIDs_m3['G'].unique())))
        )].reset_index(drop=True)

        unique_selected_GIDs_in_m1m2m3 = df[(
            (df['linecolor_m1']=='purple') & (df['linecolor_m2']=='purple') &
            ((df['target_m3']==1) & (df['G'].isin(selGIDs_m3['G'].unique())))
        )].reset_index(drop=True)

        # --------------------------
        # Wheat lines with yield better performance in poor environments for each trial
        # --------------------------
        # Include reps in all GIDs per Loc-Occ to avoid gaps in GY
        #Gtemp = df_raw[df_raw['G'].isin(unique_selected_GIDs_in_m1m2m3['G'].unique())]
        #Gtemp['E'] = Gtemp['location'].astype(str) + '_' + Gtemp['Occ'].astype(str)
        #Gtemp = Gtemp[Gtemp['E'].isin(unique_selected_GIDs_in_m1m2m3['E'].unique())]
        
        df_location_lowEnv, df_location_highEnv = None, None
        # old approach
        if (bothEnvs is True):
            sel_cols = ['G', 'location', 'Occ','AvGYxLocOcc', 'AvGYxGID',  
                        'environment_m1', 'environment_m2', 'environment_m3' 
                       ] # 
            final_selected_GIDs = pd.merge(unique_selected_GIDs_in_m1m2m3[sel_cols], df_raw, 
                                           how='left', on=['G', 'location', 'Occ']) # 
        else: # new approach q==4
            df_location_lowEnv = sel_GIDs_m3[( ((sel_GIDs_m3['environment_m3']=='AL') | (sel_GIDs_m3['environment_m3']=='BL_A'))
                      & (sel_GIDs_m3['linecolor_m1']=='purple') & (sel_GIDs_m3['linecolor_m2']=='purple')
                     )]
            df_location_highEnv = sel_GIDs_m3[( ((sel_GIDs_m3['environment_m3']=='BH') 
                                                 | (sel_GIDs_m3['environment_m3']=='AH_B'))
                                               & (sel_GIDs_m3['linecolor_m1']=='purple') 
                                               & (sel_GIDs_m3['linecolor_m2']=='purple')
                     )]

            final_selected_GIDs = pd.merge(df_location_lowEnv, df_raw, how='left', on=['G', 'location', 'Occ', 'YearofSow']) # 
            sel_cols = ['G', 'E', 'location', 'Occ', 'YearofSow', 'AvGYxGID', 
                   'AvGYxLocOcc', 'm1', 'm2', 'b1', 'b2', 
                        'UID', 'CID', 'SID', 'GID', 'loc_code', 'locationname',
                        'country', 'lat', 'lon', 'cycle', 'SowYear', 'sowing', 'Days_To_Anthesis',
                   'Days_To_Heading', 'Days_To_Maturity', 'ObsYield', '1000_GRAIN_WEIGHT',
                   'Plant_Height', 'Pedigree', 'GrandParent', 'Gen_name', 'Gen_no','Nursery_y'
                       ]
            final_selected_GIDs = final_selected_GIDs[sel_cols][final_selected_GIDs['Rep']==1].reset_index(drop=True)
            final_selected_GIDs.rename(columns={'Nursery_y': 'Nursery'}, inplace=True)
        
        # Extract grand parents
        final_selected_GIDs['GrandParent'] = final_selected_GIDs['Pedigree'].apply(lambda x: x.split('//')[0])


        if (saveFig is True or showFig is True):
            figures.figure_AvgYieldbyGID_combineMethods(df_GY=final_selected_GIDs, nursery=nursery,
                              fld1="AvGYxLocOcc", fld2="AvGYxGID", lw=lw, hue=hue, s=s, alpha=alpha, loc=loc, ncol=ncol, 
                              xt_tl=xt_tl, yt_tl=yt_tl, ha=ha, va=va, showFig=showFig, saveFig=saveFig, 
                                                        dirname=dirname, fmt=fmt )

        self.final_selected_GIDs = final_selected_GIDs
        self.locations_lowEnv = df_location_lowEnv
        self.locations_highEnv = df_location_highEnv
        
        del df
        _ = gc.collect()
        return final_selected_GIDs, sel_GIDs_m1m2, sel_GIDs_m3, sel_GIDs_m1m2m3, unique_selected_GIDs_in_m1m2m3, df_location_lowEnv, df_location_highEnv

    # --------------------------------------------------------------------
    # Prepare dataset for GE models
    # --------------------------------------------------------------------
    def prepareDatasetforGEmodel(self, df_gen=None, nursery=None, verbose=False):
        ''' Prepare data as an input for the GxE models '''
        
        if (df_gen is None):
            df_gen = self.final_selected_GIDs
            if (df_gen is None):
                print("Data input is not valid")
                return
        if (nursery is None):
            print("Nursery is not valid")
            return

        df_gxe = df_gen[df_gen['Nursery']==nursery]
        df_gxe['E'] = df_gxe['E'].str.replace('.0_ESWYT', '', regex=True).replace('_ESWYT', '', regex=True)
        df_gxe['E'] = df_gxe['E'].str.replace('.0_HTWYT', '', regex=True).replace('_HTWYT', '', regex=True)
        df_gxe['E'] = df_gxe['E'].str.replace('.0_IDYN', '', regex=True).replace('_IDYN', '', regex=True)
        df_gxe['E'] = df_gxe['E'].str.replace('.0_SAWYT', '', regex=True).replace('_SAWYT', '', regex=True)

        GEYrGY = df_gxe[['G','location', 'Occ', 'loc_code', 'country', 'YearofSow' ,'ObsYield', 'Pedigree', 'GrandParent']]\
        .sort_values(['YearofSow', 'G',  'loc_code', 'location' ])
        GEYrGY['Trial'] = GEYrGY['loc_code'].astype('category').cat.codes 
        GEYrGY['E'] = GEYrGY['location'].astype(str) + '_' + GEYrGY['Occ'].astype(str) #+ GEYrGY_2['YearofSow'].astype(str)

        GEYrGY = GEYrGY[['Trial','G', 'E', 'loc_code', 'country', 'YearofSow', 'ObsYield', 'Pedigree', 'GrandParent']]\
        .sort_values(['Trial', 'loc_code','YearofSow', 'country' ])

        # Standardize or create common column names across all analysis
        # Categorize o factor Environments and Genotypes
        GEYrGY['GEN'] = GEYrGY['G'].astype('category').cat.codes #.astype(str)
        GEYrGY['ENV'] = GEYrGY['E'].astype('category').cat.codes
        # Anonymize dataset for further publishing
        GEYrGY['GEN'] = GEYrGY['GEN'].apply(lambda x: 'G'+str(x+1)).astype(str)
        GEYrGY['ENV'] = GEYrGY['ENV'].apply(lambda x: 'E'+str(x+1)).astype(str)
        GEYrGY['GY'] = GEYrGY['ObsYield'].astype(float).round(2)
        # Order data by Gen then by Env
        GEYrGY['new'] = GEYrGY['ENV'].astype(str).str.extract('(\d+$)').astype(int)
        GEYrGY = GEYrGY.sort_values(by=['GEN', 'new']).drop('new', axis=1)
        GEYrGY.reset_index(drop=True, inplace=True)
        self.GEYrGY = GEYrGY
        
        del df_gxe
        _ = gc.collect()
        return GEYrGY
    
    # --------------------------------------------------------------------
    # Display Genotype, Environment and Genotype-environment means
    # --------------------------------------------------------------------
    def GEmeans(self, df_ge=None, trial=0, title='', byCountry=False, showFig=True, saveFig=False, annot=True, square=True,
            dirname='./', fmt='pdf', verbose=False):
        '''
            Display Genotype, Environment and Genotype-environment means

            Genotype’s performance across environments shown in line plots and heatmaps
            Spatial variations (Heat-maps) for all Genotypes all Environemnts

        '''
        if (df_ge is None):
            df_ge = self.GEYrGY
            if (df_ge is None):
                print("Data input is not valid")
                return
        
        df_ge = df_ge[df_ge['Trial']==trial].reset_index(drop=True)
        trialname = df_ge[df_ge['Trial']==trial].loc[0, 'loc_code']
        table_means_trial = pd.pivot_table(df_ge[df_ge['Trial']==trial], 
                                           values='ObsYield', index=['G'],columns=['E'], 
                                           aggfunc=np.mean, margins=True, margins_name='AvgYield')
        table_means_byCountry_trial = pd.pivot_table(df_ge[df_ge['Trial']==trial], 
                                           values='ObsYield', index=['G'],columns=['country'], 
                                           aggfunc=np.mean, margins=True, margins_name='AvgYield')
        
        # Fill gaps with the reported GY value, It doesn't matter if the point is above or below the average (1:1 line)
        # Identify the null values (This step was covered in the ge_impute_missing_values function for AMMI and GGE methods)
        # Now we are going to use the existing value
        for col in table_means_trial.columns.values:
            if (col!='AvgYield'):
                gens = table_means_trial[table_means_trial[col].isnull()].index.tolist()
                if (len(gens)<=0):
                    continue
                e, o = col.split('_')
                for g in gens:
                    try:
                        gy = self.data[((self.data['location']==int(e)) 
                                 & (self.data['Occ']==int(o)) & (self.data['G']==g)
                                )]['ObsYield'].values[0]
                    except:
                        gy = np.nan
                        #print(g, e, o, gy)
                    table_means_trial.loc[(table_means_trial.index==g),col] = gy
        # By Country
        for col in table_means_byCountry_trial.columns.values:
            if (col!='AvgYield'):
                gens = table_means_byCountry_trial[table_means_byCountry_trial[col].isnull()].index.tolist()
                if (len(gens)<=0):
                    continue
                for g in gens:
                    try:
                        gy = self.data[((self.data['country']==col) & (self.data['G']==g)
                                )]['ObsYield'].values[0]
                    except:
                        gy = np.nan
                    table_means_byCountry_trial.loc[(table_means_byCountry_trial.index==g),col] = gy

        fname = 'IWIN_GxE_Heatmap'
        fig, (axs) = plt.subplots(1, 1, figsize=(14, 10))
        if (byCountry is True):
            fname = 'IWIN_GxE_byCountryHeatmap'
            # genotype-environment means and effects,
            g1 = sns.heatmap(
                table_means_byCountry_trial, 
                square=square, # make cells square
                cbar_kws={'fraction' : 0.007, 'label': 'Grain yield (t/ha)'}, # shrink colour bar
                cmap='RdYlGn', # use orange/red colour map
                linewidth=0.5, # space between cells
                ax=axs,
                annot=annot, fmt=".1f"
            );
            axs.set_xlabel('Environment (Country)',fontsize=15)
            axs.set_ylabel('Genotype (CID-SID)',fontsize=15)
            axs.set_title('Grain Yield - {}'.format(trialname),fontsize=18)
        else:
            g = sns.heatmap(
                table_means_trial, 
                square=square, # make cells square
                cbar_kws={'fraction' : 0.007, 'label': 'Grain yield (t/ha)'}, # shrink colour bar
                cmap='RdYlGn', # use orange/red colour map
                linewidth=0.5, # space between cells
                ax=axs,
                annot=annot, 
                fmt=".1f"
            );
            axs.set_xlabel('Environment (Loc-Occ)',fontsize=15)
            axs.set_ylabel('Genotype (CID-SID)',fontsize=15)
            axs.set_title('Grain Yield - {}'.format(trialname),fontsize=18)

        # Save in PDF
        if (saveFig is True and fmt=='pdf'):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path, '{}_{}_{}.pdf'
                                     .format(fname, trialname.replace(' ', '_'), hoy)), bbox_inches='tight', 
                        orientation='portrait', 
                        pad_inches=0.5, dpi=300) #papertype='a4', 

        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path,"{}_{}_{}.{}"
                                     .format(fname, trialname.replace(' ', '_'), hoy, fmt)), bbox_inches='tight', 
                        facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                        pad_inches=0.5, dpi=300)

        if (showFig is True):
            if (saveFig is False):
                plt.tight_layout()
            plt.show()
        else:
            del fig
            plt.close();
        
        self.table_means_trial = table_means_trial
        return table_means_trial, table_means_byCountry_trial
    
    # --------------------------------------------------------------------
    def ge_impute_missing_values(self, data=None, threshold=.5, fillna=True, verbose=False):
        ''' Select better environments and Genotypes along sites with no empty values. 
            Remove environments or Genotypes with more than 50% or threshold value of empty data.
            Fill the rest of the null values with the mean by Environment.

            :params data: Table of Genotypes, Environments and GE means
            :params threshold: Upper limit in percentage to remove emtpy Environments and Genotypes records

            :results: A cleaner two-way table of means
        '''
        if data is None:
            print("Input not valid")
            return
        df = data.copy()
        # Number of rows, Number of columns
        nr, nc = df.shape
        # Get Genotypes (rows) with missing values
        rows_missing_values = [x for x in dict(df.isnull().any(axis=1)) if (df[df.index==x].isnull().any(axis=0).any() == True)]
        sel_rows = []
        for r in rows_missing_values:
            mis_vals = df[df.index==r].isnull().sum(axis=1)[0]
            mis_perc = mis_vals/nc #df.shape[1]
            if (mis_perc > threshold): 
                sel_rows.append(r)
            else:
                if (verbose is True):
                    print(r, mis_vals, mis_perc)
        # Remove Genotypes with more than the threshold in percentage
        df_g = df[df.index.isin(sel_rows)]

        # Get columns with missing values
        cols_missing_values = [x for x in dict(df.isna().any()) if (df[x].isna().any() == True)]
        sel_cols = []
        for c in cols_missing_values:
            mis_idxs = pd.isnull(df[c]).to_numpy().nonzero()[0]
            mis_perc = len(mis_idxs)/nr #tbl_means.shape[0]
            if (mis_perc > threshold):
                sel_cols.append(c)
            else:
                if (verbose is True):
                    print(c, len(mis_idxs), mis_perc)
        # Remove Env with more than the threshold
        df_e = df[sel_cols]

        # Filling empty values with the mean by Environment
        if (fillna is True):
            #tbl_means.fillna(method='bfill', axis=1)
            df = df[[c for c in df.columns if c not in sel_cols]][~df.index.isin(sel_rows)]
            nr, nc = df.shape
            if ((nr >= 2) and (nc >=2)):
                df = df.apply(lambda x: round(x.fillna(x.median()), 2), axis=0)
            else:
                df = df.apply(lambda x: round(x.fillna(x.median()), 2), axis=0)

        return df

    # --------------------------------------------------------------------
    # AMMI - Additive Main Effect and Multiplicative interaction model
    # --------------------------------------------------------------------
    def geAMMI(self, data=None, Gen='G', Env='E', trait='GY', trialname=None, twt=False, threshold=.5, fillna=True, 
                 centering=False, scaling=False, n_components=None, verbose=False):
        ''' 
            Additive Main Effect and Multiplicative interaction model

            :params data: table with Genotypes, Environments and traits observations
            :params twt: if the data is Two-way table of means, default False or in long format.

            :returns: AMMI model
        '''
        if (data is None):
            data = self.data
            if (data is None):
                print("Input data not valid")
                return
        if (trialname is None):
            trialname = self.nursery
            if (trialname is None):
                #print("Nursery is not valid")
                trialname=''
                
        # Check out for empty values
        df = data.copy()
        df.dropna(subset=[trait], inplace=True)
        if (twt is False):
            # Create table of means
            tbl_means = pd.pivot_table(df, values=trait, index=[Gen],columns=[Env], 
                                       aggfunc=np.mean, margins=False, margins_name='AvgTrait')
        else:
            tbl_means = df
        if (tbl_means.isnull().any(axis=1).any() or tbl_means.isnull().any(axis=0).any()): 
            if (verbose is True):
                print("Table of means has empty values")
        # Clean dataset
        if (fillna is True):
            tbl_means = self.ge_impute_missing_values(data=tbl_means, threshold=threshold, fillna=fillna, verbose=verbose)
        
        m = None
        if (tbl_means.isna().any().any()):
            print("It's not possible to fit a model...")
        else:
            #print("Fitting an AMMI model")
            m = ammi.AMMI(tbl_means, trait, trialname, 
                             {'centering':centering, 'scaling':scaling, 'n_components':n_components })
            _ = m.fit()
        return m
    
    
    # --------------------------------------------------------------------
    # GGE - Genotype plus Genotype-Environment interaction model
    # --------------------------------------------------------------------
    def geGGE(self, data=None, env='ENV', gen='GEN', trait='GY', trialname=None, params=None, 
              twt=False, impute_missing_threshold=.5, verbose=False):
        ''' Genotype plus Genotype-Environment interaction model '''
        if (data is None):
            data = self.data
            if (data is None):
                print("Input data not valid")
                return
        if (trialname is None):
            trialname = self.nursery
            if (trialname is None):
                #print("Nursery is not valid")
                trialname=''
        # Check out for empty values
        df = data.copy()
        if (twt is False):
            df.dropna(subset=[trait], inplace=True)
        # Fitting an GGE model
        m = gge.GGE(data=df, env=env, gen=gen, trait=trait, trialname=trialname, params=params, 
                    twt=twt, verbose=verbose)
        _ = m.fit(impute_missing_threshold=impute_missing_threshold)
        return m
    
    # --------------------------------------------------------------------
    # Get legend for GGE biplot charts
    # --------------------------------------------------------------------
    def get_legend_gen(self, df, t=None):
        if (df is None):
            df = self.data.copy()
            if (df is None):
                print("Input data not valid")
                return
        leg_gen = df.groupby(['GEN', 'G'], as_index=False).agg({}).set_index('GEN').to_dict()['G']
        new_leg_gen = {}
        for k in leg_gen.keys():
            pedigree = df.query('G =="{}"'.format(leg_gen[k]))['GrandParent'].value_counts().keys()[0]
            if (t=='AMMI'):
                new_leg_gen[leg_gen[k]] = pedigree
            if (t=='GGE'):
                new_leg_gen[k] = leg_gen[k] +' - '+ pedigree
            else:
                new_leg_gen[k] = ''

        return new_leg_gen

    def get_legend_env(self, df=None, t=None):
        if (df is None):
            df = self.data.copy()
            if (df is None):
                print("Input data not valid")
                return
        leg_env = df.groupby(['ENV', 'country', 'E'], as_index=False).agg({}).set_index('ENV').to_dict()['E']
        new_leg_env = {}
        for k in leg_env.keys():
            country = df.query('E =="{}"'.format(leg_env[k]))['country'].value_counts().keys()[0]
            if (t=='AMMI'):
                new_leg_env[leg_env[k]] = country
            if (t=='GGE'):
                new_leg_env[k] = leg_env[k] +' - '+ country
            else:
                new_leg_env[k] = ''

        return new_leg_env
    
    def gge_rankingG(self, data=None, env='ENV', gen='GEN', trait='GY', impute_missing_threshold=0.5,
            params=None, twt=False, title=None, plot_params=None, saveFig=True, showFig=True, 
            dirname='./', fmt='pdf', verbose=False):
        '''
            Ranking Genotypes Relative to the Ideal Genotype

            An ideal genotype should have both high mean performance and high stability across environments.

            This figure defines an “ideal” genotype (the center of the concentric circles) to be a point on the AEA 
            (“absolutely stable”) in the positive direction and has a vector length equal to the longest vectors of 
            the genotypes on the positive side of AEA (“highest mean performance”). Therefore, genotypes located closer 
            to the ‘ideal genotype’ are more desirable than others.

            This Figure illustrates an important concept regarding “stability”. The term “high stability” is meaningful 
            only when associated with mean performance

            Note: it should be easy to see how misleading it can be to search and select for “stability” genes. 
            “Stable” genotypes are desirable only when they have high mean performances.

        '''
        default_params=dict(transform=0, centering=2, scaling=0, svp=2, coordflip=1, n_components=None)
        plot_default_params = dict( labelG=True, labelE=False, arrowG=False, arrowE=False, fontsizeE=8, 
                                   fontsizeG=10, ncol=2, sE=20, ncircles=8, scircles=0.25, colCircles='b', 
                                   lwcircle=1, alphaCircle=0.65, showLegend=True, addGenName=True,
                                   leg_gen=None, leg_env=None
        )
        if (data is None):
            print("Input data is not valid")
            return
        if (params is None):
            params = default_params
        else:
            params = {**default_params, **params}

        if (plot_params is None):
            plot_params = plot_default_params
        else:
            plot_params = {**plot_default_params, **plot_params}

        df = data.copy()
        df.dropna(subset=['ObsYield'],inplace=True)
        gge_model = self.geGGE(data=df, env=env, gen=gen, trait=trait, params=params, twt=twt, 
                               impute_missing_threshold=impute_missing_threshold) # long format table
        #gge_model.fit(impute_missing_threshold=impute_missing_threshold) 
        # If las filas o cols tienen mas del 50% de datos faltantes eliminelas y el resto de datos promedielos por ambiente
        if (plot_params['addGenName'] is True):
            plot_params['leg_gen'] = self.get_legend_gen(df, t='GGE') # - Add the name of Pedrigree instead of Gen codes
            plot_params['leg_env'] = self.get_legend_env(df, t='GGE') # - Add the name of country instead of Envs codes
            
        # Display chart 8
        gge_model.plot(title=title, bptype=8, plot_params=plot_params, saveFig=saveFig, 
                       showFig=showFig, dirname=dirname, fmt=fmt)
        selG = gge_model.best_G()
        genname = ''
        if (plot_params['leg_gen'] is not None):
            genname = selG +' - '+ plot_params['leg_gen'][selG]
        if (verbose is True):
            print("Closest Genotype to the 'ideal genotype' is {}".format(genname))
        return genname, gge_model
    
    
    # ----------------------------------
    # 
    def gge_www(self, data=None, env='ENV', gen='GEN', trait='GY', impute_missing_threshold=0.5,
            params=None, twt=False, title=None, plot_params=None, saveFig=True, showFig=True, 
            dirname='./', fmt='pdf', verbose=False):
        '''
            Which-won-where

            One of the most attractive features of a GGE biplot is its ability to show the which-won-where pattern 
            of a genotype by environment dataset. Many researchers find this use of a biplot intriguing, as it 
            graphically addresses important concepts such as crossover GE, mega-environment differentiation, specific 
            adaptation, etc.

            Genotypes located on the vertices of the polygon performed either the best or the poorest in one or 
            more environments.

            - The perpendicular lines are equality lines between adjacent genotypes on the polygon, which 
              facilitate visual comparison of them.
            - The equality lines divide the biplot into sectors, and the winning genotype for each sector 
              is the one located on the respective vertex.
            - This pattern suggests that the target environment may consist of XXXX different mega-environments 
              and that different cultivars should be selected and deployed for each.


        '''
        
        default_params=dict(transform=0, centering=2, scaling=0, svp=2, coordflip=1, n_components=None)
        plot_default_params = dict( labelG=True, labelE=False, arrowG=False, arrowE=False, fontsizeE=8.5, fontsizeG=8.5,
                       ncol=2, sE=20, hueE='labelME', hueG='labelME', leg_gen=None, leg_env=None, 
                       convhull_marker="o", convhull_markersize=15, convhull_markeredgecolor="blue", 
                       convhull_markerfacecolor="white", convhull_linecolor="blue", convhull_ls='-', convhull_lw=2,
                       convhull_perpline_lw=1, convhull_perpline_ls='--', convhull_perpline_color='blue',
                       me_marker="D", me_markersize=8, me_markeredgecolor='red', me_markerfacecolor='red',
                       me_fontsize=14, me_fontcolor='black', me_ha='center', me_va='bottom', me_linewidth=2, me_alpha=0.2,
                       drawME=True, fillME=True, drawDivisions=True, drawConvexhull=True, showLegend=True, addGenName=True,
                       lim_span=0.25, mxlim=10,
                       me_colors= ['g','r','y','b','orange','brown','c']
        )
        if (data is None):
            print("Input data is not valid")
            return
        if (params is None):
            params = default_params
        else:
            params = {**default_params, **params}

        if (plot_params is None):
            plot_params = plot_default_params
        else:
            plot_params = {**plot_default_params, **plot_params}

        df = data.copy()
        df.dropna(subset=['ObsYield'],inplace=True)
        
        if (len(df[gen].unique()) <=2):
            print("It's not possible to carry out a convex hull for only 2 genotypes.")
            return
            
        gge_model = self.geGGE(data=df, env=env, gen=gen, trait=trait, params=params, twt=twt,
                              impute_missing_threshold=impute_missing_threshold) # long format table
        #gge_model.fit(impute_missing_threshold=impute_missing_threshold) 
        # If las filas o cols tienen mas del 50% de datos faltantes eliminelas y el resto de datos promedielos por ambiente
        if (plot_params['addGenName'] is True):
            plot_params['leg_gen'] = self.get_legend_gen(df, t='GGE') # - Add the name of Pedrigree instead of Gen codes
            plot_params['leg_env'] = self.get_legend_env(df, t='GGE') # - Add the name of country instead of Envs codes
        # Display chart 9
        gge_model.plot(title=title, bptype=9, plot_params=plot_params, saveFig=saveFig, 
                       showFig=showFig, dirname=dirname, fmt=fmt)
        
        return gge_model
    
    
    # -------------------------
    # Weather data
    # -------------------------
    def extract_Weather(self, df_raw=None, df_gen=None, df_weather=None, verbose=False):
        '''
            Extract weather data from AgERA5 dataset for each location

            Use 150 days after planting to define the growing season

        '''
        if (df_raw is None):
            df_raw = self.data
            if (df_raw is None):
                print("Input raw data not valid")
                return
        if (df_gen is None):
            df_gen = self.final_selected_GIDs
            if (df_gen is None):
                print("Data input is not valid")
                return
        # Get Weather for growing seasons
        if (df_weather is None):
            #self.weather = self.getWeather(m.config['WeatherFile'])
            df_weather = self.weather
            if (df_weather is None):
                print("Climate data is not valid")
                return
        # 
        loc_toExtract_weather = df_raw[['location', 'lat', 'lon']][df_raw['E'].isin(df_gen['E'].unique())]\
        .groupby(['location', 'lat', 'lon'], as_index=False).agg({}).reset_index(drop=True)

        # Select only weather data related to the locations in the analysis
        weather_data_sel_loc = df_weather[df_weather['location'].isin(loc_toExtract_weather['location'])]

        sel_columns = ['location', 'sowing', 'lat', 'lon', 'ObsYield']
        data_for_extract_weather = df_raw[sel_columns][df_raw['E'].isin(df_gen['E'].unique())]\
        .groupby(['location', 'sowing'], as_index=False).agg({'lat':'mean', 'lon':'mean', 'ObsYield':'mean'}).reset_index(drop=True)

        # Temperature and precipitation of each location averaged over periods of 10 days 
        # after the reported sowing date
        for idx in data_for_extract_weather.index:
            try:
                loc = data_for_extract_weather.iloc[idx]['location']
                if verbose is True:
                    print(f"Processing location {loc}")
                sowing = str(data_for_extract_weather.iloc[idx]['sowing']).split(' ')[0]
                end_date = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(150))).split(' ')[0]
                #if verbose is True: print(loc, sowing, end_date)
                # Get weather data
                weatherDF = weather_data_sel_loc[( (weather_data_sel_loc['Date']>=sowing) 
                                                  & (weather_data_sel_loc['Date']<end_date)
                                                  & (weather_data_sel_loc['location']==loc))].reset_index(drop=True)

                # Remove those values taken as outliers, same as iPAR Yield model
                # truncate negative values to 0.0
                if verbose is True:
                    print(weatherDF[['TMIN','TMAX']].describe())
                weatherDF['TMIN'] = weatherDF['TMIN'].apply(lambda x: x if x >=0.0 else 0.0)
                weatherDF['TMAX'] = weatherDF['TMAX'].apply(lambda x: x if x >=0.0 else 0.0)
                if verbose is True:
                    print(weatherDF[['TMIN','TMAX']].describe())

                # Masks
                days_10 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(10))).split(' ')[0]
                days_20 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(20))).split(' ')[0]
                days_30 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(30))).split(' ')[0]
                days_40 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(40))).split(' ')[0]
                days_50 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(50))).split(' ')[0]
                days_60 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(60))).split(' ')[0]
                days_70 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(70))).split(' ')[0]
                days_80 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(80))).split(' ')[0]
                days_90 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(90))).split(' ')[0]
                days_100 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(100))).split(' ')[0]
                days_110 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(110))).split(' ')[0]
                days_120 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(120))).split(' ')[0]
                days_130 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(130))).split(' ')[0]
                days_140 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(140))).split(' ')[0]
                days_150 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(150))).split(' ')[0]

                _mask_10d = ( (weatherDF['Date'] >= sowing) & (weatherDF['Date'] < days_10) ) # & (weatherDF['location']==loc) )
                _mask_10_20d = ( (weatherDF['Date'] >= days_10) & (weatherDF['Date'] < days_20) )
                _mask_20_30d = ( (weatherDF['Date'] >= days_20) & (weatherDF['Date'] < days_30) )
                _mask_30_40d = ( (weatherDF['Date'] >= days_30) & (weatherDF['Date'] < days_40) )
                _mask_40_50d = ( (weatherDF['Date'] >= days_40) & (weatherDF['Date'] < days_50) )
                _mask_50_60d = ( (weatherDF['Date'] >= days_50) & (weatherDF['Date'] < days_60) )
                _mask_60_70d = ( (weatherDF['Date'] >= days_60) & (weatherDF['Date'] < days_70) )
                _mask_70_80d = ( (weatherDF['Date'] >= days_70) & (weatherDF['Date'] < days_80) )
                _mask_80_90d = ( (weatherDF['Date'] >= days_80) & (weatherDF['Date'] < days_90) )
                _mask_90_100d = ( (weatherDF['Date'] >= days_90) & (weatherDF['Date'] < days_100) )
                _mask_100_110d = ( (weatherDF['Date'] >= days_100) & (weatherDF['Date'] < days_110) )
                _mask_110_120d = ( (weatherDF['Date'] >= days_110) & (weatherDF['Date'] < days_120) )
                _mask_120_130d = ( (weatherDF['Date'] >= days_120) & (weatherDF['Date'] < days_130) )
                _mask_130_140d = ( (weatherDF['Date'] >= days_130) & (weatherDF['Date'] < days_140) )
                _mask_140_150d = ( (weatherDF['Date'] >= days_140) & (weatherDF['Date'] < days_150) )

                w_10d = weatherDF[_mask_10d].reset_index(drop=True)
                w_20d = weatherDF[_mask_10_20d].reset_index(drop=True)
                w_30d = weatherDF[_mask_20_30d].reset_index(drop=True)
                w_40d = weatherDF[_mask_30_40d].reset_index(drop=True)
                w_50d = weatherDF[_mask_40_50d].reset_index(drop=True)
                w_60d = weatherDF[_mask_50_60d].reset_index(drop=True)
                w_70d = weatherDF[_mask_60_70d].reset_index(drop=True)
                w_80d = weatherDF[_mask_70_80d].reset_index(drop=True)
                w_90d = weatherDF[_mask_80_90d].reset_index(drop=True)
                w_100d = weatherDF[_mask_90_100d].reset_index(drop=True)
                w_110d = weatherDF[_mask_100_110d].reset_index(drop=True)
                w_120d = weatherDF[_mask_110_120d].reset_index(drop=True)
                w_130d = weatherDF[_mask_120_130d].reset_index(drop=True)
                w_140d = weatherDF[_mask_130_140d].reset_index(drop=True)
                w_150d = weatherDF[_mask_140_150d].reset_index(drop=True)
                if ( len(w_10d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_10d), sowing, days_10))
                    data_for_extract_weather.loc[idx, 'TMIN_10d'] = round(w_10d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_10d'] = round(w_10d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_10d'] = round(w_10d['PCP'].sum(), 0)

                if ( len(w_20d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_20d), days_10, days_20))
                    data_for_extract_weather.loc[idx, 'TMIN_20d'] = round(w_20d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_20d'] = round(w_20d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_20d'] = round(w_20d['PCP'].sum(), 0)

                if ( len(w_30d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_30d), days_20, days_30))
                    data_for_extract_weather.loc[idx, 'TMIN_30d'] = round(w_30d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_30d'] = round(w_30d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_30d'] = round(w_30d['PCP'].sum(), 0)

                if ( len(w_40d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_40d), days_30, days_40))
                    data_for_extract_weather.loc[idx, 'TMIN_40d'] = round(w_40d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_40d'] = round(w_40d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_40d'] = round(w_40d['PCP'].sum(), 0)

                if ( len(w_50d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_50d), days_40, days_50))
                    data_for_extract_weather.loc[idx, 'TMIN_50d'] = round(w_50d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_50d'] = round(w_50d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_50d'] = round(w_50d['PCP'].sum(), 0)

                if ( len(w_60d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_60d), days_50, days_60))
                    data_for_extract_weather.loc[idx, 'TMIN_60d'] = round(w_60d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_60d'] = round(w_60d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_60d'] = round(w_60d['PCP'].sum(), 0)

                if ( len(w_70d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_70d), days_60, days_70))
                    data_for_extract_weather.loc[idx, 'TMIN_70d'] = round(w_70d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_70d'] = round(w_70d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_70d'] = round(w_70d['PCP'].sum(), 0)

                if ( len(w_80d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_80d), days_70, days_80))
                    data_for_extract_weather.loc[idx, 'TMIN_80d'] = round(w_80d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_80d'] = round(w_80d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_80d'] = round(w_80d['PCP'].sum(), 0)

                if ( len(w_90d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_90d), days_80, days_90))
                    data_for_extract_weather.loc[idx, 'TMIN_90d'] = round(w_90d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_90d'] = round(w_90d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_90d'] = round(w_90d['PCP'].sum(), 0)

                if ( len(w_100d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_100d), days_90, days_100))
                    data_for_extract_weather.loc[idx, 'TMIN_100d'] = round(w_100d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_100d'] = round(w_100d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_100d'] = round(w_100d['PCP'].sum(), 0)

                if ( len(w_110d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_110d), days_100, days_110))
                    data_for_extract_weather.loc[idx, 'TMIN_110d'] = round(w_110d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_110d'] = round(w_110d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_110d'] = round(w_110d['PCP'].sum(), 0)

                if ( len(w_120d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_120d), days_110, days_120))
                    data_for_extract_weather.loc[idx, 'TMIN_120d'] = round(w_120d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_120d'] = round(w_120d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_120d'] = round(w_120d['PCP'].sum(), 0)

                if ( len(w_130d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_130d), days_120, days_130))
                    data_for_extract_weather.loc[idx, 'TMIN_130d'] = round(w_130d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_130d'] = round(w_130d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_130d'] = round(w_130d['PCP'].sum(), 0)

                if ( len(w_140d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_140d), days_130, days_140))
                    data_for_extract_weather.loc[idx, 'TMIN_140d'] = round(w_140d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_140d'] = round(w_140d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_140d'] = round(w_140d['PCP'].sum(), 0)

                if ( len(w_150d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_150d), days_140, days_150))
                    data_for_extract_weather.loc[idx, 'TMIN_150d'] = round(w_150d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_150d'] = round(w_150d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_150d'] = round(w_150d['PCP'].sum(), 0)
            except Exception as err:
                print("Problem in site {}. Error: {}".format(loc, err))
        #
        # End of the loop
        # Estimate additional climate variables
        sel_cols = ['location', 'sowing', #'ObsYield', 
                'TMIN_10d', 'TMIN_20d','TMIN_30d','TMIN_40d','TMIN_50d',
                'TMIN_60d', 'TMIN_70d','TMIN_80d','TMIN_90d','TMIN_100d',
                'TMIN_110d', 'TMIN_120d', 'TMIN_130d','TMIN_140d','TMIN_150d',

                'TMAX_10d', 'TMAX_20d','TMAX_30d','TMAX_40d','TMAX_50d',
                'TMAX_60d','TMAX_70d', 'TMAX_80d','TMAX_90d','TMAX_100d',
                'TMAX_110d', 'TMAX_120d','TMAX_130d', 'TMAX_140d','TMAX_150d',

                'PCP_10d', 'PCP_20d','PCP_30d','PCP_40d','PCP_50d',
                'PCP_60d','PCP_70d','PCP_80d','PCP_90d','PCP_100d',
                'PCP_110d', 'PCP_120d','PCP_130d','PCP_140d','PCP_150d'
              ]

        tmin_columns = ['TMIN_10d', 'TMIN_20d', 'TMIN_30d', 'TMIN_40d',
               'TMIN_50d', 'TMIN_60d', 'TMIN_70d', 'TMIN_80d', 'TMIN_90d', 'TMIN_100d',
               'TMIN_110d', 'TMIN_120d', 'TMIN_130d', 'TMIN_140d', 'TMIN_150d'
                       ]
        tmax_columns = ['TMAX_10d', 'TMAX_20d', 'TMAX_30d', 'TMAX_40d', 'TMAX_50d', 'TMAX_60d',
               'TMAX_70d', 'TMAX_80d', 'TMAX_90d', 'TMAX_100d', 'TMAX_110d',
               'TMAX_120d', 'TMAX_130d', 'TMAX_140d', 'TMAX_150d'
                       ] 
        pcp_columns = ['PCP_10d',
               'PCP_20d', 'PCP_30d', 'PCP_40d', 'PCP_50d', 'PCP_60d', 'PCP_70d',
               'PCP_80d', 'PCP_90d', 'PCP_100d', 'PCP_110d', 'PCP_120d', 'PCP_130d',
               'PCP_140d', 'PCP_150d']
        # 
        data_for_extract_weather = data_for_extract_weather[sel_cols]

        # Average minimum Temperature of the entirollingeriods
        data_for_extract_weather['Tmin_avg'] = data_for_extract_weather[tmin_columns].mean(axis=1).round(1)
        # Average maximum Temperature of the entire periods
        data_for_extract_weather['Tmax_avg'] = data_for_extract_weather[tmax_columns].mean(axis=1).round(1)
        # Total precipitation of the entire periods
        data_for_extract_weather['Pcp_total'] = data_for_extract_weather[pcp_columns].sum(axis=1).astype(int)
        #Maximum Temperature of Warmest Period: The highest temperature of any 10d period maximum temperature.
        data_for_extract_weather['MaxTemp_WarmestPeriod'] = data_for_extract_weather[tmax_columns].max(axis=1).round(1)

        #Minimum Temperature of Coldest Period: The lowest temperature of any 10d period minimum temperature.
        data_for_extract_weather['MinTemp_ColdestPeriod'] = data_for_extract_weather[tmin_columns].min(axis=1).round(1)

        #Temperature Annual Range: The difference between the Maximum Temperature of Warmest Period and the Minimum Temperature of Coldest Period.
        data_for_extract_weather['Temperature_Period_Range'] = data_for_extract_weather['MaxTemp_WarmestPeriod'] - data_for_extract_weather['MinTemp_ColdestPeriod']

        # ******* ME4 wettest quarter precipitation = 100mm, < 400mm
        #Precipitation of Wettest Period: The precipitation of the wettest 10d period.
        data_for_extract_weather['Precip_WettestPeriod'] = data_for_extract_weather[pcp_columns].apply(lambda row: custom_rolling(row, w=9, op='wettest'), axis=1)

        #Precipitation of Driest Period: The precipitation of the driest 10d period.
        data_for_extract_weather['Precip_DriestPeriod'] = data_for_extract_weather[pcp_columns].apply(lambda row: custom_rolling(row, w=9, op='driest'), axis=1)

        # Precipitation Seasonality: The Coefficient of Variation is the standard deviation of the monthly precipitation 
        # estimates expressed as a percentage of the mean of those estimates (i.e. the growing period mean).

        # average temperature min for the coolest quarter (3 consecutive coolest months - 90 days (9 periods in our table))
        # Rolling 9 times in our dataset and get the most coolest
        # ******* ME4 Average min. temperature for coolest quarter 11°C > Tmin ≥ 3°C
        # Mean Temperature of Coldest Quarter: The coldest quarter of the year is determined (to the nearest month), 
        # and the mean temperature of this period is calculated.
        data_for_extract_weather['MinTemp_ColdestQuarter'] = data_for_extract_weather[tmin_columns].apply(lambda row: custom_rolling(row, w=9, op='coldest'), axis=1)

        # Mean Temperature of Warmest Quarter: The warmest quarter of the year is determined (to the nearest month), 
        # and the mean temperature of this period is calculated.
        data_for_extract_weather['MeanTemp_WarmestQuarter'] = data_for_extract_weather[tmax_columns].apply(lambda row: custom_rolling(row, w=9, op='warmest'), axis=1)

        # Merge some results
        sel_gens_weather = pd.merge(df_raw, data_for_extract_weather, on=['location','sowing'], how='right')
        sel_gens_weather = sel_gens_weather[sel_gens_weather['G'].isin(df_gen['G'].unique())]

        return sel_gens_weather, data_for_extract_weather

    
    #
    def extract_Weather_growthStages(self, df_raw=None, df_gen=None, df_weather=None, verbose=False):
        '''
            Extract weather data from AgERA5 dataset for each location

            Use 150 days after planting to define the growing season

        '''
        if (df_raw is None):
            df_raw = self.data
            if (df_raw is None):
                print("Input raw data not valid")
                return
        if (df_gen is None):
            df_gen = self.final_selected_GIDs
            if (df_gen is None):
                print("Data input is not valid")
                return
        # Get Weather for growing seasons
        if (df_weather is None):
            #self.weather = self.getWeather(m.config['WeatherFile'])
            df_weather = self.weather
            if (df_weather is None):
                print("Climate data is not valid")
                return
        # 
        #loc_toExtract_weather = df_raw[['location', 'lat', 'lon']][df_raw['E'].isin(df_gen['E'].unique())]\
        #.groupby(['location', 'lat', 'lon'], as_index=False).agg({}).reset_index(drop=True)
        
        loc_toExtract_weather = df_gen[['location', 'lat', 'lon']]\
        .groupby(['location', 'lat', 'lon'], as_index=False).agg({}).reset_index(drop=True)

        # Select only weather data related to the locations in the analysis
        weather_data_sel_loc = df_weather[df_weather['location'].isin(loc_toExtract_weather['location'])]

        sel_columns = ['location', 'sowing', 'lat', 'lon']
        data_for_extract_weather = df_gen[sel_columns]\
        .groupby(['location', 'sowing'], as_index=False).agg({'lat':'mean', 'lon':'mean'}).reset_index(drop=True)

        # Temperature and precipitation of each location averaged over periods of 10 days 
        # after the reported sowing date
        for idx in data_for_extract_weather.index:
            try:
                loc = data_for_extract_weather.iloc[idx]['location']
                if verbose is True:
                    print(f"Processing location {loc}")
                sowing = str(data_for_extract_weather.iloc[idx]['sowing']).split(' ')[0]
                end_date = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(150))).split(' ')[0]
                #if verbose is True: print(loc, sowing, end_date)
                # Get weather data
                weatherDF = weather_data_sel_loc[( (weather_data_sel_loc['Date']>=sowing) 
                                                  & (weather_data_sel_loc['Date']<end_date)
                                                  & (weather_data_sel_loc['location']==loc))].reset_index(drop=True)

                # Remove those values taken as outliers, same as iPAR Yield model
                # truncate negative values to 0.0
                if verbose is True:
                    print(weatherDF[['TMIN','TMAX']].describe())
                weatherDF['TMIN'] = weatherDF['TMIN'].apply(lambda x: x if x >=0.0 else 0.0)
                weatherDF['TMAX'] = weatherDF['TMAX'].apply(lambda x: x if x >=0.0 else 0.0)
                if verbose is True:
                    print(weatherDF[['TMIN','TMAX']].describe())

                # Masks
                days_10 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(10))).split(' ')[0]
                days_20 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(20))).split(' ')[0]
                days_30 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(30))).split(' ')[0]
                days_40 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(40))).split(' ')[0]
                days_50 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(50))).split(' ')[0]
                days_60 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(60))).split(' ')[0]
                days_70 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(70))).split(' ')[0]
                days_80 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(80))).split(' ')[0]
                days_90 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(90))).split(' ')[0]
                days_100 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(100))).split(' ')[0]
                days_110 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(110))).split(' ')[0]
                days_120 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(120))).split(' ')[0]
                days_130 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(130))).split(' ')[0]
                days_140 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(140))).split(' ')[0]
                days_150 = str(pd.to_datetime(str(sowing)) + pd.DateOffset(days=int(150))).split(' ')[0]

                _mask_10d = ( (weatherDF['Date'] >= sowing) & (weatherDF['Date'] < days_10) ) # & (weatherDF['location']==loc) )
                _mask_10_20d = ( (weatherDF['Date'] >= days_10) & (weatherDF['Date'] < days_20) )
                _mask_20_30d = ( (weatherDF['Date'] >= days_20) & (weatherDF['Date'] < days_30) )
                _mask_30_40d = ( (weatherDF['Date'] >= days_30) & (weatherDF['Date'] < days_40) )
                _mask_40_50d = ( (weatherDF['Date'] >= days_40) & (weatherDF['Date'] < days_50) )
                _mask_50_60d = ( (weatherDF['Date'] >= days_50) & (weatherDF['Date'] < days_60) )
                _mask_60_70d = ( (weatherDF['Date'] >= days_60) & (weatherDF['Date'] < days_70) )
                _mask_70_80d = ( (weatherDF['Date'] >= days_70) & (weatherDF['Date'] < days_80) )
                _mask_80_90d = ( (weatherDF['Date'] >= days_80) & (weatherDF['Date'] < days_90) )
                _mask_90_100d = ( (weatherDF['Date'] >= days_90) & (weatherDF['Date'] < days_100) )
                _mask_100_110d = ( (weatherDF['Date'] >= days_100) & (weatherDF['Date'] < days_110) )
                _mask_110_120d = ( (weatherDF['Date'] >= days_110) & (weatherDF['Date'] < days_120) )
                _mask_120_130d = ( (weatherDF['Date'] >= days_120) & (weatherDF['Date'] < days_130) )
                _mask_130_140d = ( (weatherDF['Date'] >= days_130) & (weatherDF['Date'] < days_140) )
                _mask_140_150d = ( (weatherDF['Date'] >= days_140) & (weatherDF['Date'] < days_150) )

                w_10d = weatherDF[_mask_10d].reset_index(drop=True)
                w_20d = weatherDF[_mask_10_20d].reset_index(drop=True)
                w_30d = weatherDF[_mask_20_30d].reset_index(drop=True)
                w_40d = weatherDF[_mask_30_40d].reset_index(drop=True)
                w_50d = weatherDF[_mask_40_50d].reset_index(drop=True)
                w_60d = weatherDF[_mask_50_60d].reset_index(drop=True)
                w_70d = weatherDF[_mask_60_70d].reset_index(drop=True)
                w_80d = weatherDF[_mask_70_80d].reset_index(drop=True)
                w_90d = weatherDF[_mask_80_90d].reset_index(drop=True)
                w_100d = weatherDF[_mask_90_100d].reset_index(drop=True)
                w_110d = weatherDF[_mask_100_110d].reset_index(drop=True)
                w_120d = weatherDF[_mask_110_120d].reset_index(drop=True)
                w_130d = weatherDF[_mask_120_130d].reset_index(drop=True)
                w_140d = weatherDF[_mask_130_140d].reset_index(drop=True)
                w_150d = weatherDF[_mask_140_150d].reset_index(drop=True)
                if ( len(w_10d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_10d), sowing, days_10))
                    data_for_extract_weather.loc[idx, 'TMIN_10d'] = round(w_10d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_10d'] = round(w_10d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_10d'] = round(w_10d['PCP'].sum(), 0)

                if ( len(w_20d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_20d), days_10, days_20))
                    data_for_extract_weather.loc[idx, 'TMIN_20d'] = round(w_20d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_20d'] = round(w_20d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_20d'] = round(w_20d['PCP'].sum(), 0)

                if ( len(w_30d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_30d), days_20, days_30))
                    data_for_extract_weather.loc[idx, 'TMIN_30d'] = round(w_30d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_30d'] = round(w_30d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_30d'] = round(w_30d['PCP'].sum(), 0)

                if ( len(w_40d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_40d), days_30, days_40))
                    data_for_extract_weather.loc[idx, 'TMIN_40d'] = round(w_40d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_40d'] = round(w_40d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_40d'] = round(w_40d['PCP'].sum(), 0)

                if ( len(w_50d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_50d), days_40, days_50))
                    data_for_extract_weather.loc[idx, 'TMIN_50d'] = round(w_50d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_50d'] = round(w_50d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_50d'] = round(w_50d['PCP'].sum(), 0)

                if ( len(w_60d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_60d), days_50, days_60))
                    data_for_extract_weather.loc[idx, 'TMIN_60d'] = round(w_60d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_60d'] = round(w_60d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_60d'] = round(w_60d['PCP'].sum(), 0)

                if ( len(w_70d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_70d), days_60, days_70))
                    data_for_extract_weather.loc[idx, 'TMIN_70d'] = round(w_70d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_70d'] = round(w_70d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_70d'] = round(w_70d['PCP'].sum(), 0)

                if ( len(w_80d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_80d), days_70, days_80))
                    data_for_extract_weather.loc[idx, 'TMIN_80d'] = round(w_80d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_80d'] = round(w_80d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_80d'] = round(w_80d['PCP'].sum(), 0)

                if ( len(w_90d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_90d), days_80, days_90))
                    data_for_extract_weather.loc[idx, 'TMIN_90d'] = round(w_90d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_90d'] = round(w_90d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_90d'] = round(w_90d['PCP'].sum(), 0)

                if ( len(w_100d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_100d), days_90, days_100))
                    data_for_extract_weather.loc[idx, 'TMIN_100d'] = round(w_100d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_100d'] = round(w_100d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_100d'] = round(w_100d['PCP'].sum(), 0)

                if ( len(w_110d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_110d), days_100, days_110))
                    data_for_extract_weather.loc[idx, 'TMIN_110d'] = round(w_110d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_110d'] = round(w_110d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_110d'] = round(w_110d['PCP'].sum(), 0)

                if ( len(w_120d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_120d), days_110, days_120))
                    data_for_extract_weather.loc[idx, 'TMIN_120d'] = round(w_120d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_120d'] = round(w_120d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_120d'] = round(w_120d['PCP'].sum(), 0)

                if ( len(w_130d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_130d), days_120, days_130))
                    data_for_extract_weather.loc[idx, 'TMIN_130d'] = round(w_130d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_130d'] = round(w_130d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_130d'] = round(w_130d['PCP'].sum(), 0)

                if ( len(w_140d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_140d), days_130, days_140))
                    data_for_extract_weather.loc[idx, 'TMIN_140d'] = round(w_140d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_140d'] = round(w_140d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_140d'] = round(w_140d['PCP'].sum(), 0)

                if ( len(w_150d)>1 ):
                    if verbose is True:
                        print("{} days period from {} to {}".format(len(w_150d), days_140, days_150))
                    data_for_extract_weather.loc[idx, 'TMIN_150d'] = round(w_150d['TMIN'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'TMAX_150d'] = round(w_150d['TMAX'].mean(), 1)
                    data_for_extract_weather.loc[idx, 'PCP_150d'] = round(w_150d['PCP'].sum(), 0)
            except Exception as err:
                print("Problem in site {}. Error: {}".format(loc, err))
        #
        # End of the loop
        # Estimate additional climate variables
        sel_cols = ['location', 'sowing', #'ObsYield', 
                'TMIN_10d', 'TMIN_20d','TMIN_30d','TMIN_40d','TMIN_50d',
                'TMIN_60d', 'TMIN_70d','TMIN_80d','TMIN_90d','TMIN_100d',
                'TMIN_110d', 'TMIN_120d', 'TMIN_130d','TMIN_140d','TMIN_150d',

                'TMAX_10d', 'TMAX_20d','TMAX_30d','TMAX_40d','TMAX_50d',
                'TMAX_60d','TMAX_70d', 'TMAX_80d','TMAX_90d','TMAX_100d',
                'TMAX_110d', 'TMAX_120d','TMAX_130d', 'TMAX_140d','TMAX_150d',

                'PCP_10d', 'PCP_20d','PCP_30d','PCP_40d','PCP_50d',
                'PCP_60d','PCP_70d','PCP_80d','PCP_90d','PCP_100d',
                'PCP_110d', 'PCP_120d','PCP_130d','PCP_140d','PCP_150d'
              ]

        tmin_columns = ['TMIN_10d', 'TMIN_20d', 'TMIN_30d', 'TMIN_40d',
               'TMIN_50d', 'TMIN_60d', 'TMIN_70d', 'TMIN_80d', 'TMIN_90d', 'TMIN_100d',
               'TMIN_110d', 'TMIN_120d', 'TMIN_130d', 'TMIN_140d', 'TMIN_150d'
                       ]
        tmax_columns = ['TMAX_10d', 'TMAX_20d', 'TMAX_30d', 'TMAX_40d', 'TMAX_50d', 'TMAX_60d',
               'TMAX_70d', 'TMAX_80d', 'TMAX_90d', 'TMAX_100d', 'TMAX_110d',
               'TMAX_120d', 'TMAX_130d', 'TMAX_140d', 'TMAX_150d'
                       ] 
        pcp_columns = ['PCP_10d',
               'PCP_20d', 'PCP_30d', 'PCP_40d', 'PCP_50d', 'PCP_60d', 'PCP_70d',
               'PCP_80d', 'PCP_90d', 'PCP_100d', 'PCP_110d', 'PCP_120d', 'PCP_130d',
               'PCP_140d', 'PCP_150d']
        # 
        data_for_extract_weather = data_for_extract_weather[sel_cols]

        # Average minimum Temperature of the entirollingeriods
        data_for_extract_weather['Tmin_avg'] = data_for_extract_weather[tmin_columns].mean(axis=1).round(1)
        # Average maximum Temperature of the entire periods
        data_for_extract_weather['Tmax_avg'] = data_for_extract_weather[tmax_columns].mean(axis=1).round(1)
        # Total precipitation of the entire periods
        data_for_extract_weather['Pcp_total'] = data_for_extract_weather[pcp_columns].sum(axis=1).astype(int)
        #Maximum Temperature of Warmest Period: The highest temperature of any 10d period maximum temperature.
        data_for_extract_weather['MaxTemp_WarmestPeriod'] = data_for_extract_weather[tmax_columns].max(axis=1).round(1)

        #Minimum Temperature of Coldest Period: The lowest temperature of any 10d period minimum temperature.
        data_for_extract_weather['MinTemp_ColdestPeriod'] = data_for_extract_weather[tmin_columns].min(axis=1).round(1)

        #Temperature Annual Range: The difference between the Maximum Temperature of Warmest Period and the Minimum Temperature of Coldest Period.
        data_for_extract_weather['Temperature_Period_Range'] = data_for_extract_weather['MaxTemp_WarmestPeriod'] - data_for_extract_weather['MinTemp_ColdestPeriod']

        # ******* ME4 wettest quarter precipitation = 100mm, < 400mm
        #Precipitation of Wettest Period: The precipitation of the wettest 10d period.
        data_for_extract_weather['Precip_WettestPeriod'] = data_for_extract_weather[pcp_columns].apply(lambda row: custom_rolling(row, w=9, op='wettest'), axis=1)

        #Precipitation of Driest Period: The precipitation of the driest 10d period.
        data_for_extract_weather['Precip_DriestPeriod'] = data_for_extract_weather[pcp_columns].apply(lambda row: custom_rolling(row, w=9, op='driest'), axis=1)

        # Precipitation Seasonality: The Coefficient of Variation is the standard deviation of the monthly precipitation 
        # estimates expressed as a percentage of the mean of those estimates (i.e. the growing period mean).

        # average temperature min for the coolest quarter (3 consecutive coolest months - 90 days (9 periods in our table))
        # Rolling 9 times in our dataset and get the most coolest
        # ******* ME4 Average min. temperature for coolest quarter 11°C > Tmin ≥ 3°C
        # Mean Temperature of Coldest Quarter: The coldest quarter of the year is determined (to the nearest month), 
        # and the mean temperature of this period is calculated.
        data_for_extract_weather['MinTemp_ColdestQuarter'] = data_for_extract_weather[tmin_columns].apply(lambda row: custom_rolling(row, w=9, op='coldest'), axis=1)

        # Mean Temperature of Warmest Quarter: The warmest quarter of the year is determined (to the nearest month), 
        # and the mean temperature of this period is calculated.
        data_for_extract_weather['MeanTemp_WarmestQuarter'] = data_for_extract_weather[tmax_columns].apply(lambda row: custom_rolling(row, w=9, op='warmest'), axis=1)
        
        
        # ----------------------------
        # 
        # ----------------------------
        
        
        # Merge some results
        #sel_gens_weather = pd.merge(df_raw, data_for_extract_weather, on=['location','sowing'], how='right')
        sel_gens_weather = pd.merge(df_gen, data_for_extract_weather, on=['location','sowing'], how='right')
        #sel_gens_weather = sel_gens_weather[sel_gens_weather['G'].isin(df_gen['G'].unique())]

        return sel_gens_weather, data_for_extract_weather
