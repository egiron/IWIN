# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import numba
#from numba import cuda
from numba import vectorize, int32, int64, float32, float64

def calcGDD(Tmin=None, Tmax=None, Tbase=0):
    ''' Growing degree days GDD (°F or °C)
        Calculated from: ((Daily Max Temp + Daily Min Temp)/2) - 32 °F (or 
        ((Daily Max Temp + Daily Min Temp)/2)).
        
    :param Tmin: Number or array of Minimum Temperatures
    :param Tmax: Number or array of Maximum Temperatures
    :param Tbase: Temperature base of the crop

    :return: a number or array of Growing degree days (GDD)
    
    '''
    if (Tmin is None or Tmax is None):
        print("Temperature parameters are not valid")
        return
    # If the minimum temperature Tmin is below the Tbase, then Tmin = Tbase
    #GDD1 = max((( Tmax + Tmin ) / 2) - Tbase, 0)
    GDD = (( Tmax + Tbase ) / 2) - Tbase if Tmin < Tbase else max((( Tmax + Tmin ) / 2) - Tbase, 0)
    GDD = max(GDD, 0)
    return int(GDD)

@numba.vectorize([float64(float64, float64, float64)])
def getGDD(Tmin, Tmax, Tbase=0):
    ''' Growing degree days GDD (°F or °C)
        Calculated from: ((Daily Max Temp + Daily Min Temp)/2) - 32 °F (or 
        ((Daily Max Temp + Daily Min Temp)/2)).
        
    :param Tmin: Number or array of Minimum Temperatures
    :param Tmax: Number or array of Maximum Temperatures
    :param Tbase: Temperature base of the crop

    :return: a number or array of Growing degree days (GDD)
    
    '''
    # If the minimum temperature Tmin is below the Tbase, then Tmin = Tbase
    #GDD1 = max((( Tmax + Tmin ) / 2) - Tbase, 0)
    GDD = (( Tmax + Tbase ) / 2) - Tbase if Tmin < Tbase else max((( Tmax + Tmin ) / 2) - Tbase, 0)
    GDD = max(GDD, 0)
    return GDD #int(round(GDD,0))

@numba.jit(parallel=True, nopython=False) #, nogil=True 
def apply_GDD(Tmin, Tmax, Tbase=0):
    ''' Growing degree days GDD (°F or °C)
        Calculated from: ((Daily Max Temp + Daily Min Temp)/2) - 32 °F (or 
        ((Daily Max Temp + Daily Min Temp)/2)).
        
    :param Tmin: Number or array of Minimum Temperatures
    :param Tmax: Number or array of Maximum Temperatures
    :param Tbase: Temperature base of the crop

    :return: a number or array of Growing degree days (GDD)
    
    '''
    n = len(Tmin)
    result = np.empty(n, dtype="int32") #"float64"
    assert len(Tmin) == len(Tmax) == n
    for i in range(n):
        result[i] = getGDD(Tmin[i], Tmax[i], Tbase)
    return result

def calculateGDD(Tmin=None, Tmax=None, Tbase=0):
    ''' Growing degree days GDD (°F or °C)
        Calculated from: ((Daily Max Temp + Daily Min Temp)/2) - 32 °F (or 
        ((Daily Max Temp + Daily Min Temp)/2)).
        
    :param Tmin: Number or array of Minimum Temperatures
    :param Tmax: Number or array of Maximum Temperatures
    :param Tbase: Temperature base of the crop

    :return: a number or array of Growing degree days (GDD)
    
    '''
    if (Tmin is None or Tmax is None):
        print("Temperature parameters are not valid")
        return
    result = []
    try:
        result = apply_GDD(Tmin, Tmax, Tbase )
    except:
        print("Error calculating Growing degree days")
    
    return result #pd.Series(result, index=w.index, name="GDD")