# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import numba
#from numba import cuda
from numba import vectorize, int32, int64, float32, float64

def calcTDay(Tmin, Tmax, tminFactor=0.25):
    '''Calculate day time temperature
        TDay = 0.75*Tmax + 0.25*TMin
    
    :param Tn: Number or array of Minimum Temperatures
    :param Tx: Number or array of Maximum Temperatures
    :param tminFactor: Minimum Temperature factor

    :return: a number or array of Day Temperatures

    ''' 
    if (Tmax <= Tmin):
        print ("Error: Maximum temperature is equal or lower than minimum temperature")
        return None
    tmaxFactor = 1 - tminFactor
    TDay = tmaxFactor*Tmax + tminFactor*Tmin
    return float("{:.3f}".format(TDay))

##@numba.jit(parallel=True, nopython=False)
@numba.vectorize([float64(float64, float64, float64)])
def getTDay(Tmin, Tmax, tminFactor=0.25):
    '''Calculate day time temperature
        TDay = 0.75*Tmax + 0.25*TMin
    
    :param Tmin: Number or array of Minimum Temperatures
    :param Tmax: Number or array of Maximum Temperatures
    :param tminFactor: Minimum Temperature factor

    :return: a number or array of Day Temperatures
    
    ''' 
    if (Tmax <= Tmin):
        print ("Error: Maximum temperature is equal or lower than minimum temperature")
        return None
    tmaxFactor = 1 - tminFactor
    TDay = tmaxFactor*Tmax + tminFactor*Tmin
    return float("{:.3f}".format(TDay))

@numba.jit(parallel=True, nopython=False) 
def apply_TDay(Tmin, Tmax, tminFactor):
    ''' Calculate day time temperature
        
    :param Tmin: Numpy array of Minimum Temperatures
    :param Tmax: Numpy array of Maximum Temperatures
    :param tminFactor: Minimum Temperature factor

    :return: a number or array of Day Temperatures
    
    ''' 
    n = len(Tmin)
    result = np.empty(n, dtype="float64")
    assert len(Tmin) == len(Tmax) == n
    for i in range(n):
        result[i] = getTDay(Tmin[i], Tmax[i], tminFactor)
    return result

def estimate_TDay(Tmin=None, Tmax=None, tminFactor=0.25):
    '''Calculate day time temperature

    :param Tmin: Numpy array of Minimum Temperatures
    :param Tmax: Numpy array of Maximum Temperatures
    :param tminFactor: Minimum Temperature factor

    :return: a number or array of Day Temperature

    ''' 
    result = []
    if ( (Tmin is None) or (Tmax is None) ):
        print("Weather data not valid")
        return
    try:
        result = apply_TDay(Tmin, Tmax, tminFactor )
    except:
        print("Error calculating Day temperature")

    return result #pd.Series(result, index=w.index, name="TDay")

