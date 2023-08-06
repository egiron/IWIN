# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import numba
#from numba import cuda
from numba import vectorize, int32, int64, float32, float64


def calcPRFT(TDay, TOpt=18):
    ''' Estimate Photosynthesis reduction factor (PRFT)
        PRFT = 1 – 0.0025 * (TDay – TOpt)^2
        
    :param TDay: Number or array of Day Temperatures
    :param TOpt: Optimum Temperature

    :return: a number or array of PRFT
    
    '''
    PRFT = 0
    if (TDay > 0):
        PRFT = 1 - 0.0025 * (TDay - TOpt) ** 2
    return PRFT

@numba.vectorize([float64(float64, float64)])
def getPRFT(Tday, Topt): 
    prft = 1 - 0.0025*(Tday-Topt)**2 if Tday > 0.0 else 0.0
    return prft
    
#@numba.jit(nopython=True)
#def PRFT(Tday, Topt):
#    prft = 0
#    if (Tday > 0):
#        prft = 1 - 0.0025*(Tday-Topt)**2
#    return(prft)

@numba.jit(parallel=True)
def apply_PRFT(Tday, Topt=18):
    ''' Estimate Photosynthesis reduction factor (PRFT)
        PRFT = 1 – 0.0025 * (TDay – TOpt)^2
        
    :param TDay: Number or array of Day Temperatures
    :param TOpt: Optimum Temperature

    :return: a number or array of PRFT
    
    '''
    n = len(Tday)
    result = np.zeros(n, dtype="float64")
    for i in range(n):
        result[i] = getPRFT(Tday[i], Topt)
    return result

def calculatePRFT(Tday, Topt=18):
    ''' Estimate Photosynthesis reduction factor (PRFT)
        PRFT = 1 – 0.0025 * (TDay – TOpt)^2
        
    :param TDay: Number or array of Day Temperatures
    :param TOpt: Optimum Temperature

    :return: a number or array of PRFT
    
    '''
    if (Tday is None):
        print("Day Temperature parameter is not valid")
        return
    result = []
    try:
        result = apply_PRFT(Tday, Topt)
    except:
        print("Error calculating photosynthesis reduction factor (PRFT)")
    
    return result #pd.Series(result, index=w.index, name="PRFT") int(GDD)