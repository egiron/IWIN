# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import numba
#from numba import cuda
from numba import vectorize, int32, int64, float32, float64
from datetime import date, datetime

#from timezonefinder import TimezoneFinder
#from astral import LocationInfo
#from astral.sun import sun
# Calculate Day length
# def calcDaylength(dt, lat, lng, name='', region='', verbose=False):
#     # Get the timezone from Latitude and Longitude
#     tf = TimezoneFinder()
#     timezone_str = tf.timezone_at(lng=lng, lat=lat)  # returns 'Europe/Berlin'
#     # l = LocationInfo('name', 'region', 'timezone/name', 0.1, 1.2)
#     city = LocationInfo("{}".format(name), "{}".format(region), 
#                         "{}".format(timezone_str), lat, lng)
#     if (verbose==True):
#         print((
#             f"Information for {city.name}/{city.region}\n"
#             f"Timezone: {city.timezone}\n"
#             f"Latitude: {city.latitude:.02f}; Longitude: {city.longitude:.02f}\n"
#         ))

#     #s = sun(city.observer, date=date(2022, 4, 30), tzinfo=city.timezone)
#     s = sun(city.observer, date=dt, tzinfo=city.timezone)
#     if (verbose==True):
#         print((
#             f'Dawn:    {s["dawn"]}\n'
#             f'Sunrise: {s["sunrise"]}\n'
#             f'Noon:    {s["noon"]}\n'
#             f'Sunset:  {s["sunset"]}\n'
#             f'Dusk:    {s["dusk"]}\n'
#         ))
#     DL = s["sunset"] - s["sunrise"]
#     return DL.total_seconds() / (3600)

# Code share from Urs
def CBM_daylength(dayOfYear, lat, p=0.0):
    '''
        A model comparison for daylength as a function of latitude and day of year
        William C. Forsythe a,* Edward J. Rykiel Jr. a, Randal S. Stahl a, Hsin-i Wu a, Robert M. Schoolfield b
        Ecological Modelling 80 (1995) 87-95
        
    :param dayOfYear: Day of the Year (DOY) 
    :param lat: Latitude of the site in celsius degrees
    :param p: Sun angle with the horizon. eg. p = 6.0 : civil twilight,
              p = 0.0 : day starts / ends when sun is even with the horizon.
              Default value p=0

    :return: a daylength for the specific site

    '''
    latInRad = np.deg2rad(lat)
    revolutionAngle = 0.2163108 + 2*np.arctan(0.9671396*np.tan(0.00860 *(dayOfYear - 186)))
    declinationAngle = np.arcsin(0.39795*np.cos(revolutionAngle))
    value = (np.sin(np.deg2rad(p)) + (np.sin(latInRad)*np.sin(declinationAngle))) / (np.cos(latInRad)*np.cos(declinationAngle))
    if value <= -1.0: 
        return  0.0
    if value >= 1.0: 
        return 24.0
    else: 
        return 24 - (24/np.pi)*np.arccos(value)

#
@numba.vectorize([float64(float64, float64, float64)])
def getDaylength(dayOfYear, lat, p=0.0):
    ''' Length of the day for a specific site
    
    :param dayOfYear: Day of the Year (DOY) 
    :param lat: Latitude of the site in celsius degrees
    :param p: Sun angle with the horizon. eg. p = 6.0 : civil twilight,
              p = 0.0 : day starts / ends when sun is even with the horizon.
              Default value p=0

    :return: a daylength for the specific site

    '''
    latInRad = np.deg2rad(lat)
    revolutionAngle = 0.2163108 + 2*np.arctan(0.9671396*np.tan(0.00860 *(dayOfYear - 186)))
    declinationAngle = np.arcsin(0.39795*np.cos(revolutionAngle))
    value = (np.sin(np.deg2rad(p)) + (np.sin(latInRad)*np.sin(declinationAngle))) / (np.cos(latInRad)*np.cos(declinationAngle))
    if value <= -1.0: 
        return  0.0
    if value >= 1.0: 
        return 24.0
    else: 
        return 24 - (24/np.pi)*np.arccos(value)

@numba.jit(parallel=True, nopython=False) 
def apply_Daylength(dayOfYear, lat, p=0.0):
    ''' Length of the day for a specific site
    
    :param dayOfYear: Day of the Year (DOY) 
    :param lat: Latitude of the site in celsius degrees
    :param p: Sun angle with the horizon. eg. p = 6.0 : civil twilight,
              p = 0.0 : day starts / ends when sun is even with the horizon.
              Default value p=0

    :return: a daylength for the specific site

    '''
    n = len(lat)
    result = np.empty(n, dtype="int32")
    for i in range(n):
        result[i] = getDaylength(dayOfYear, lat[i], p)
    return result

def calculateDayLength(d=None, lat=None, p=0.0):
    '''Get Day length '''
    if (d is None):
        print("Date for getting daylength is not valid")
        return
    if (lat is None):
        print("Latitude not valid")
        return
    daylength = None
    try:
        day_of_year = datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday
        daylength = CBM_daylength(day_of_year, lat, p)
    except:
        print("Error getting Daylength")
    
    return float("{:.2f}".format(daylength)) 