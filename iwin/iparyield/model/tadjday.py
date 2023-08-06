# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import numba
#from numba import cuda
from numba import vectorize, int32, int64, float32, float64

def calcTAdjDay(Tavg, threshold=42):
    ''' Calculate daily progress (Adjusted TDays). Depreciated: 20230127
    
    :param Tavg: Number or array of Average Temperatures
    :param threshold: Number of TDays. eg. IWIN: 42, Dhillon: 35, South Asia: 41

    :return: a number or array of adjusted temperature days
    
    ''' 
    TAdjDay = threshold / (150 * np.exp(-0.06 * Tavg))
    return float("{:.3f}".format(TAdjDay))

@numba.vectorize([float64(float64, float64)])
def getTAdjDay(Tavg, threshold=42):
    ''' Calculate daily progress (Adjusted TDays)
    
    :param Tavg: Number or array of Average Temperatures
    :param threshold: Number of TDays. eg. IWIN: 42, Dhillon: 35, South Asia: 41
    
    Modified Jan-27-2023: The algorithm tends to overestimate the short and 
                          underestimate the long grainfilling durations.
                          y=165* exp(-0.07 * TAverage)

    :return: a number or array of adjusted temperature days
    
    ''' 
    #TAdjDay = threshold / (150 * np.exp(-0.06 * Tavg))
    # OJO!!! This need to interchange with the Eq. above meanwhile find a better stable equation
    #TAdjDay = threshold / (165 * np.exp(-0.07 * Tavg)) # This not improve the results 
    TAdjDay = 45 / (175 * np.exp(-0.07 * Tavg))
    #
    # Linear Y=60-0.79*TAverage
    #TAdjDay = 45 / (60 - 0.79 * Tavg)
    
    return float("{:.3f}".format(TAdjDay))

@numba.jit(parallel=True, nopython=False) 
def apply_TAdjDay(Tavg, threshold=42):
    ''' Calculate daily progress (Adjusted TDays)
    
    :param Tavg: Number or array of Average Temperatures
    :param threshold: Number of TDays. eg. IWIN: 42, Dhillon: 35, South Asia: 41

    :return: a number or array of adjusted temperature days
    
    ''' 
    n = len(Tavg)
    result = np.empty(n, dtype="float64")
    for i in range(n):
        result[i] = getTAdjDay(Tavg[i], threshold)
    return result

def estimate_TAdjDay(Tavg=None, threshold=42):
    ''' Calculate daily progress (Adjusted TDays)
    
    :param Tavg: Number or array of Average Temperatures
    :param threshold: Number of TDays. eg. IWIN: 42, Dhillon: 35, South Asia: 41

    :return: a number or array of adjusted temperature days
    
    '''
    result = []
    if (Tavg is None):
        print("Weather data not valid")
        return
    try:
        result = apply_TAdjDay(Tavg, threshold)
    except:
        print("Error calculating Adjusted TDays")

    return result #pd.Series(result, index=w.index, name="TDay")

