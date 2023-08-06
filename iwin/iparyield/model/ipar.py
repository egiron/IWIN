# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import numba
#from numba import cuda
from numba import vectorize, int32, int64, float32, float64

# --------------------------------------------------------------
# Processing iPAR - Total light interception
# --------------------------------------------------------------
def calcIPAR(Norm_TT_EH=None, Norm_SimNDVI_EH=None, Norm_SimNDVI_HM=None, norm_iPAR_EH_bounds=0.5, 
             NDVI_constantIPAR=0.19, verbose=False):
    '''
        Total light interception - iPAR

        # ** Asrar, G., Fuchs, M., Kanemasu, E.T., Hatfield, J.L., 1984. 
        # Estimating absorbed photosynthetic radiation and leaf area index from spectral reflectance 
        # in wheat. Agron. J. 76, 300–306.
        # - Campos 2018 Remote sensing-based crop biomass with water or light-driven crop growth models in 
        #   wheat commercial fields

        iPAR = NDVI * 1.25 - 0.19 # between heading and maturity (Campos et al. 2018)
        iPAR = NDVI * 1.25 - 0.21 Daughtry et al. (1992)

        -------------
        :params Norm_TT_EH: 
        :params Norm_SimNDVI_EH: 
        :params Norm_SimNDVI_HM: 
        :params norm_iPAR_EH_bounds: Bounds for iPAR multi-linear equations

        :return: An array of Total light interception values

    '''

    if (Norm_SimNDVI_EH is None):
        print("NDVI from Emergence to Heading is not valid")
        return
    if (Norm_SimNDVI_HM is None):
        print("NDVI from Heading to Maturity is not valid")
        return
    if (Norm_TT_EH is None):
        print("Normilize Thermal time from Emergence to Heading is not valid")
        return
    #if (norm_iPAR_EH_bounds is None):
    #    print("")
    #if (NDVI_constantIPAR is None):
    #    print("")
    
    # ----------------------------------------------------------------
    # iPAR - Total light interception from emergence to maturity
    # ----------------------------------------------------------------
    if (verbose is True):
        print("Estimating Total light interception (iPAR) from emergence to maturity...")
    # Note: Usando esta función para todo el periodo ofrece mejores resultados
    NDVI_EM = np.concatenate([Norm_SimNDVI_EH, Norm_SimNDVI_HM[1:]]) # remove duplicated value at Head
    iPAR_EM = NDVI_EM * 1.25 - NDVI_constantIPAR

    # ----------------------------------------------------------------
    # iPAR - from emergence to heading
    # ----------------------------------------------------------------
    if (verbose is True):
        print("Estimating iPAR from emergence to heading...")
    # Calculate IPAR values from EH using Multilinear curve
    #multilinearEH==True
    iPAR_EH = []
    for i in Norm_TT_EH:
        if (i <= 0.15):
            iPAR = 0.0 + (i * 0.333333) 
        elif ((i > 0.15) & (i <= norm_iPAR_EH_bounds)): 
            if (norm_iPAR_EH_bounds==0.5):
                iPAR = (2.714286 * i) - 0.357143
            if (norm_iPAR_EH_bounds==0.6):
                iPAR = (2.111111111 * i) - 0.266666667
            if (norm_iPAR_EH_bounds==0.7):
                iPAR = (1.727272727 * i) - 0.209090909
        elif ((i > norm_iPAR_EH_bounds)):
            iPAR = 1.0
        else:
            iPAR = np.nan
        #
        iPAR_EH.append(float("{:.3f}".format(iPAR))) #denormiPAR(iPAR)

    # ----------------------------------------------------------------
    # iPAR - from heading and maturity
    # ----------------------------------------------------------------
    if (verbose is True):
        print("Estimating iPAR from heading and maturity...")
    iPAR_HM = Norm_SimNDVI_HM * 1.25 - NDVI_constantIPAR

    return NDVI_EM, iPAR_EM, iPAR_EH, iPAR_HM
        


@numba.vectorize([float64(float64, float64)])
def getIPAR_EH(Norm_TT_EH, norm_iPAR_EH_bounds=0.5):
    ''' Calculate daily progress (Adjusted TDays)
    
    :param Tavg: Number or array of Average Temperatures
    :param threshold: Number of TDays. eg. IWIN: 42, Dhillon: 35, South Asia: 41

    :return: a number or array of adjusted temperature days
    
    ''' 
    if (Norm_TT_EH <= 0.15):
        iPAR = 0.0 + (Norm_TT_EH * 0.333333) 
    elif ((Norm_TT_EH > 0.15) & (Norm_TT_EH <= norm_iPAR_EH_bounds)): 
        if (norm_iPAR_EH_bounds==0.5):
            iPAR = (2.714286 * Norm_TT_EH) - 0.357143
        if (norm_iPAR_EH_bounds==0.6):
            iPAR = (2.111111111 * Norm_TT_EH) - 0.266666667
        if (norm_iPAR_EH_bounds==0.7):
            iPAR = (1.727272727 * Norm_TT_EH) - 0.209090909
    elif ((Norm_TT_EH > norm_iPAR_EH_bounds)):
        iPAR = 1.0
    else:
        iPAR = None
            
    return float("{:.3f}".format(iPAR))

@numba.jit(parallel=True, nopython=False) 
def apply_IPAR_EH(Norm_TT_EH, norm_iPAR_EH_bounds=0.5):
    ''' Calculate daily progress (Adjusted TDays)
    
    :param Tavg: Number or array of Average Temperatures
    :param threshold: Number of TDays. eg. IWIN: 42, Dhillon: 35, South Asia: 41

    :return: a number or array of adjusted temperature days
    
    ''' 
    n = len(Norm_TT_EH)
    result = np.empty(n, dtype="float64")
    for i in range(n):
        result[i] = getIPAR_EH( Norm_TT_EH[i], norm_iPAR_EH_bounds )
    return result

def estimate_IPAR(Norm_TT_EH=None, Norm_SimNDVI_EH=None, 
                  Norm_SimNDVI_HM=None, norm_iPAR_EH_bounds=0.5, 
                  NDVI_constantIPAR=0.19, verbose=False):
    '''
        Total light interception - iPAR

        # ** Asrar, G., Fuchs, M., Kanemasu, E.T., Hatfield, J.L., 1984. 
        # Estimating absorbed photosynthetic radiation and leaf area index from spectral reflectance 
        # in wheat. Agron. J. 76, 300–306.
        # - Campos 2018 Remote sensing-based crop biomass with water or light-driven crop growth models in 
        #   wheat commercial fields

        iPAR = NDVI * 1.25 - 0.19 # between heading and maturity (Campos et al. 2018)
        iPAR = NDVI * 1.25 - 0.21 Daughtry et al. (1992)

        -------------
        :params Norm_TT_EH: 
        :params Norm_SimNDVI_EH: 
        :params Norm_SimNDVI_HM: 
        :params norm_iPAR_EH_bounds: Bounds for iPAR multi-linear equations

        :return: An array of Total light interception values

    '''
    if (Norm_SimNDVI_EH is None):
        print("NDVI from Emergence to Heading is not valid")
        return
    if (Norm_SimNDVI_HM is None):
        print("NDVI from Heading to Maturity is not valid")
        return
    if (Norm_TT_EH is None):
        print("Normilize Thermal time from Emergence to Heading is not valid")
        return
    #if (norm_iPAR_EH_bounds is None):
    #    print("")
    #if (NDVI_constantIPAR is None):
    #    print("")
    
    NDVI_EM, iPAR_EM, iPAR_EH, iPAR_HM = None, None, None, None
    #try:
    # iPAR - Total light interception from emergence to maturity
    if (verbose is True):
        print("Estimating Total light interception (iPAR) from emergence to maturity...")
    # Note: Usando esta función para todo el periodo ofrece mejores resultados
    NDVI_EM = np.concatenate([Norm_SimNDVI_EH, Norm_SimNDVI_HM[1:]]) # remove duplicated value at Head
    iPAR_EM = NDVI_EM * 1.25 - NDVI_constantIPAR
    # iPAR - from emergence to heading
    if (verbose is True):
        print("Estimating iPAR from emergence to heading...")
    iPAR_EH = apply_IPAR_EH(Norm_TT_EH, norm_iPAR_EH_bounds)
    # iPAR - from heading and maturity
    if (verbose is True):
        print("Estimating iPAR from heading and maturity...")
    iPAR_HM = Norm_SimNDVI_HM * 1.25 - NDVI_constantIPAR
    
    #except:
    #    print("Error calculating iPAR")
    
    return NDVI_EM, iPAR_EM, iPAR_EH, iPAR_HM

