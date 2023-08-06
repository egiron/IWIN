# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import numba
#from numba import cuda
from numba import vectorize, int32, int64, float32, float64


# ---------------------------------------------------
# The normalized difference vegetation index (NDVI) 
# ---------------------------------------------------
def estimateNDVI_EH(norm_TT_EH=None, NDVI_lowerThreshold=None, 
                    NDVI_Threshold=None, NDVI_max=None, verbose=False):
    '''
    Estimate NDVI values from emergence to heading.

    Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
    from Emergence to Heading

    :params norm_TT_EH: Normilize GDD or Thermal time from Emergence to Heading
    :params NDVI_lowerThreshold: Lower threshold to estimate NDVI
    :params NDVI_Threshold: Threshold to estimate NDVI
    :params NDVI_max: Maximum NDVI value allowed
    :verbose: Display messages during the processing
    
    :return: An array with NDVI values from Emergence to Heading

    '''
    if (norm_TT_EH is None):
        print("Normilize Thermal time from Emergence to Heading is not valid")
        return
    if (NDVI_lowerThreshold is None or NDVI_Threshold is None or NDVI_max is None):
        print("NDVI parameters not defined yet")
        return
    
    norm_SimNDVI = []
    if (verbose is True):
        print("Estimating Normalized NDVI from emergence to heading...")
    # eqEH=='multilinear1'
    NDVI_slope_phase_1 = (0.2 - NDVI_lowerThreshold) / NDVI_Threshold
    NDVI_slope_phase_2 = (NDVI_max - 0.2) / (0.5 - NDVI_Threshold)
    # Look for corresponding NDVI
    for i in norm_TT_EH:
        if (i <= 0.15):
            ndvi = 0.16 + (i * NDVI_slope_phase_1) 
        elif ((i > 0.15) & (i <= 0.5)): 
            ndvi = (NDVI_slope_phase_2 * i) - NDVI_Threshold
        elif ((i > 0.5) & (i <= 1.0)):
            ndvi = NDVI_max
        else:
            ndvi = None
        norm_SimNDVI.append(float("{:.2f}".format(ndvi)))

    return norm_SimNDVI


def estimateNDVI_HM(norm_TT_HM=None, NDVImax=None, NDVI_atHeading=0.94, verbose=False):
    '''
    Estimate NDVI values from Heading to Maturity.

    Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
    from Heading to Maturity.

    :params norm_TT_HM: Normilize GDD or Thermal time from Heading to Maturity
    :params NDVI_max: Maximum NDVI value allowed
    :params NDVI_atHeading: NDVI reached at Heading date
    :verbose: Display messages during the processing
    
    :return: An array with NDVI values from Heading to Maturity
    
    '''
    if (norm_TT_HM is None):
        print("Normilize Thermal time from Heading to Maturity is not valid")
        return
    if (NDVImax is None or NDVI_atHeading is None):
        print("NDVI parameters not defined yet")
        return
    if (verbose is True):
        print("Estimating Normalized NDVI from heading and maturity...")
    
    # Setup maximum NDVI found at Heading date estimated in 'Norm_SimNDVI' variable or in estimateNDVI_EH function
    #NDVImax = Norm_SimNDVI[-1]
    #NDVImax = max(NDVImax, NDVI_atHeading)
    norm_SimNDVI = []
    #eqHM=='multilinear1'):
    for i in norm_TT_HM:
        if (i < 0.3): # NormNDVI_Phase1
            p = np.poly1d([-2.61682076e-16, 1.00000000e+00])
            ndvi = p(i) * NDVImax
        elif (i >= 0.3): # NormNDVI_Phase2
            p = np.poly1d([-1.42857143, 1.42857143])
            ndvi = p(i) * NDVImax
        else:
            ndvi = None
        #
        norm_SimNDVI.append(float("{:.2f}".format(ndvi)))
    
    return norm_SimNDVI
    

# ------------------------------------------------------------------------------------
# NDVI from Emergence to Heading
# ------------------------------------------------------------------------------------
@numba.vectorize([float64(float64, float64, float64, float64)])
def getNDVI_EH(norm_TT_EH, NDVI_lowerThreshold, NDVI_Threshold, NDVI_max):
    '''
    Estimate NDVI values from emergence to heading.

    Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
    from Emergence to Heading

    :params norm_TT_EH: Normilize GDD or Thermal time from Emergence to Heading
    :params NDVI_lowerThreshold: Lower threshold to estimate NDVI
    :params NDVI_Threshold: Threshold to estimate NDVI
    :params NDVI_max: Maximum NDVI value allowed
    :verbose: Display messages during the processing
    
    :return: An array with NDVI values from Emergence to Heading

    '''
    if (norm_TT_EH <= 0.15):
        ndvi = 0.16 + (norm_TT_EH * ((0.2 - NDVI_lowerThreshold) / NDVI_Threshold)) 
    elif ((norm_TT_EH > 0.15) & (norm_TT_EH <= 0.5)): 
        ndvi = (((NDVI_max - 0.2) / (0.5 - NDVI_Threshold)) * norm_TT_EH) - NDVI_Threshold
    elif ((norm_TT_EH > 0.5) & (norm_TT_EH <= 1.0)):
        ndvi = NDVI_max
    else:
        ndvi = None
    
    return float("{:.2f}".format(ndvi))

@numba.jit(parallel=True)
def apply_NDVI_EH(norm_TT_EH, NDVI_lowerThreshold, NDVI_Threshold, NDVI_max):
    '''
    Estimate NDVI values from emergence to heading.

    Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
    from Emergence to Heading

    :params norm_TT_EH: Normilize GDD or Thermal time from Emergence to Heading
    :params NDVI_lowerThreshold: Lower threshold to estimate NDVI
    :params NDVI_Threshold: Threshold to estimate NDVI
    :params NDVI_max: Maximum NDVI value allowed
    :verbose: Display messages during the processing
    
    :return: An array with NDVI values from Emergence to Heading

    '''
    n = len(norm_TT_EH)
    result = np.zeros(n, dtype="float64")
    for i in range(n):
        result[i] = getNDVI_EH(norm_TT_EH[i], NDVI_lowerThreshold, NDVI_Threshold, NDVI_max)
    return result

def calculateNDVI_EH(norm_TT_EH, NDVI_lowerThreshold, NDVI_Threshold, NDVI_max):
    ''' 
    Estimate NDVI values from emergence to heading.

    Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
    from Emergence to Heading

    :params norm_TT_EH: Normilize GDD or Thermal time from Emergence to Heading
    :params NDVI_lowerThreshold: Lower threshold to estimate NDVI
    :params NDVI_Threshold: Threshold to estimate NDVI
    :params NDVI_max: Maximum NDVI value allowed
    :verbose: Display messages during the processing
    
    :return: An array with NDVI values from Emergence to Heading
    
    '''
    if (norm_TT_EH is None):
        print("Normilize Thermal time from Emergence to Heading is not valid")
        return
    if (NDVI_lowerThreshold is None or NDVI_Threshold is None or NDVI_max is None):
        print("NDVI parameters not defined yet")
        return
    result = []
    try:
        result = apply_NDVI_EH(norm_TT_EH, NDVI_lowerThreshold, NDVI_Threshold, NDVI_max)
    except:
        print("Error calculating NDVI from Emergence to Heading")
    
    return result



# ------------------------------------------------------------------------------------
# NDVI from Heading to Maturity
# ------------------------------------------------------------------------------------
@numba.vectorize([float64(float64, float64)])
def getNDVI_HM(norm_TT_HM, NDVImax):
    '''
    Estimate NDVI values from Heading to Maturity.

    Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
    from Emergence to Heading

    :params norm_TT_HM: Normilize GDD or Thermal time from Heading to Maturity
    :params NDVI_max: Maximum NDVI value allowed
    :verbose: Display messages during the processing
    
    :return: An array with NDVI values from Heading to Maturity
    
    '''
    if (norm_TT_HM < 0.3): # NormNDVI_Phase1
        p = np.poly1d([-2.61682076e-16, 1.00000000e+00])
        ndvi = p(norm_TT_HM) * NDVImax
    elif (norm_TT_HM >= 0.3): # NormNDVI_Phase2
        p = np.poly1d([-1.42857143, 1.42857143])
        ndvi = p(norm_TT_HM) * NDVImax
    else:
        ndvi = None
        
    return float("{:.2f}".format(ndvi))

@numba.jit(parallel=True)
def apply_NDVI_HM(norm_TT_HM, NDVImax):
    '''
    Estimate NDVI values from Heading to Maturity.

    Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
    from Heading to Maturity.

    :params norm_TT_HM: Normilize GDD or Thermal time from Heading to Maturity
    :params NDVI_max: Maximum NDVI value allowed
    :verbose: Display messages during the processing
    
    :return: An array with NDVI values from Heading to Maturity
    
    '''
    n = len(norm_TT_HM)
    result = np.zeros(n, dtype="float64")
    for i in range(n):
        result[i] = getNDVI_HM(norm_TT_HM[i], NDVImax)
    return result


def calculateNDVI_HM(norm_TT_HM=None, NDVImax=None, NDVI_atHeading=0.94, verbose=False):
    '''
    Estimate NDVI values from Heading to Maturity.

    Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
    from Heading to Maturity.

    :params norm_TT_HM: Normilize GDD or Thermal time from Heading to Maturity
    :params NDVI_max: Maximum NDVI value allowed
    :params NDVI_atHeading: NDVI reached at Heading date
    :verbose: Display messages during the processing
    
    :return: An array with NDVI values from Heading to Maturity
    
    '''
    if (norm_TT_HM is None):
        print("Normilize Thermal time from Heading to Maturity is not valid")
        return
    if (NDVImax is None or NDVI_atHeading is None):
        print("NDVI parameters not defined yet")
        return
    if (verbose is True):
        print("Estimating Normalized NDVI from heading and maturity...")
    
    result = []
    try:
        #NDVImax = max(NDVImax, NDVI_atHeading)
        result = apply_NDVI_HM(norm_TT_HM, NDVImax)
    except:
        print("Error calculating NDVI from Heading to Maturity.")
    
    return result
    

