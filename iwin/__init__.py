# coding=utf-8
# Load libraries and existing datasets
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

__version__ = "IWIN version 2.0.0.dev"
__author__ = "Ernesto Giron Echeverry, Urs Christoph Schulthess et al."
__copyright__ = "Copyright (c) 2023 CIMMYT-Henan Collaborative Innovation Center"
__license__ = "Public Domain"

import gc
import numpy as np
import pandas as pd
from datetime import date, datetime
from tqdm import tqdm

from . import *
from .data import *
from .util import *
from .iparyield import *
from .iparyield.model import tday, tadjday, gdd, daylength, prft, ndvi, ipar

class HT:
    '''Colors class:
    Reset all colors with colors.reset
    Two subclasses fg for foreground and bg for background.
    Use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green
    Also, the generic bold, disable, underline, reverse, strikethrough,
    and invisible work with the main class
    i.e. colors.bold
    '''
    reset='\033[0m'
    bold='\033[01m'
    disable='\033[02m'
    underline='\033[04m'
    reverse='\033[07m'
    strikethrough='\033[09m'
    invisible='\033[08m'
    class fg:
        black='\033[30m'
        red='\033[31m'
        green='\033[32m'
        orange='\033[33m'
        blue='\033[34m'
        purple='\033[35m'
        cyan='\033[36m'
        lightgrey='\033[37m'
        darkgrey='\033[90m'
        lightred='\033[91m'
        lightgreen='\033[92m'
        yellow='\033[93m'
        lightblue='\033[94m'
        pink='\033[95m'
        lightcyan='\033[96m'
    class bg:
        black='\033[40m'
        red='\033[41m'
        green='\033[42m'
        orange='\033[43m'
        blue='\033[44m'
        purple='\033[45m'
        cyan='\033[46m'
        lightgrey='\033[47m'

# 
class Site(object):
    def __init__(self, uid, loc, attributes, params):
        self.uid = uid
        self.loc = loc
        self.attributes = attributes
        self.params = params
        self.pheno_dates = None
        self.weather = None
        self.GDD = None
        self.errors = []
    
    def getAttr(self):
        self.attributes['errors'] = self.errors
        return self.attributes
    
    def __str__(self):
        return f"{self.uid}" # - {self.attributes['sowing']} - {self.attributes['maturity']}"
    
    # ------------------------------
    # Get range dates
    # ------------------------------
    def getRangeDates(self, s=None, e=None, verbose=False):
        ''' Get range dates for growing period 
        
        :param s: Start date eg. Sowing date
        :param e: End date eg. Maturity date
        
        :return: an array or pandas series with the dates in between, or None if error occurs
        
        '''
        self.pheno_dates = None
        try:
            if ((s is None) or (e is None)):
                if ('sowing' in self.attributes.keys()):
                    s = self.attributes['sowing']
                else:
                    print("Error: Site has not sowing date")
                    return 
                if ('maturity' in self.attributes.keys()):
                    e = self.attributes['maturity']
                    if (e=='None' or e is None or str(e)=='nan' or str(e)=='null'):
                        if (self.params['estimateMaturity'] is True):
                            # TODO: define better what to choose
                            if ('PredMaturity_H' in self.attributes.keys()):
                                e = self.attributes['PredMaturity_H']
                            elif ('PredMaturity_pH' in self.attributes.keys()):
                                e = self.attributes['PredMaturity_pH']
                        else:
                            # Here we need to get phenology date beyond the matrurity date, due to simulations
                            e = str(pd.to_datetime(str(s)) + pd.DateOffset(days=int(365))).split(' ')[0]
                else:
                    if (verbose is True):
                        print("Error: Site has not maturity date")
                    # Here we need to get phenology date beyond the matrurity date, due to simulations
                    e = str(pd.to_datetime(str(s)) + pd.DateOffset(days=int(365))).split(' ')[0] 
            # Get dates
            if (s=='None' or s is None or str(s)=='nan' or str(s)=='null' or 
                e=='None' or e is None or str(e)=='nan' or str(e)=='null'):
                raise ("Sowing or Maturity date not valid")
            else:
                self.pheno_dates = pd.date_range(start=s, end=e) #, inclusive="both")
        except Exception as err:
            print("Error getting range dates {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Estimating range dates. Error: {}".format(err)})
        
        return self.pheno_dates
    
    # ------------------------------
    # Get Phenology dates
    # ------------------------------
    def getPhenologyDates(self, m=None, verbose=False):
        ''' Get phenology dates from reported or estimated '''
        sowing_date = None
        emergence_date = None
        heading_date = None
        maturity_date = None
        Sim_Maturity_date, Sim_DaysToMaturity, Sim_DaysHM = None, None, None
        
        try:
            if ('sowing' in self.attributes.keys()):
                sowing_date = self.attributes['sowing']
            if ('emergence' in self.attributes.keys()):
                emergence_date = self.attributes['emergence']
            if ('heading' in self.attributes.keys()):
                heading_date = self.attributes['heading']
            if ('maturity' in self.attributes.keys()):
                maturity_date = self.attributes['maturity']

            # Check for empty values
            # Sowing 
            if (sowing_date=='None' or sowing_date is None or str(sowing_date)=='nan' or str(sowing_date)=='null'):
                print("Error getting phenology dates. Sowing date not valid.")
                return
            else:
                sowing_date = pd.to_datetime(str(sowing_date), format='%Y-%m-%d')

            if (emergence_date!='None' and emergence_date is not None and str(emergence_date)!='nan' 
                and str(emergence_date)!='null'):
                emergence_date = pd.to_datetime(str(emergence_date), format='%Y-%m-%d')
            elif ('Days_To_Emergence' in self.attributes.keys()):
                Days_To_Emergence = self.attributes['Days_To_Emergence']
                if (Days_To_Emergence!='None' and Days_To_Emergence is not None and str(Days_To_Emergence)!='nan' 
                    and str(Days_To_Emergence)!='null'):
                    self.attributes['emergence'] = str(sowing_date + pd.DateOffset(days=int(Days_To_Emergence))).split(' ')[0]

            if (heading_date!='None' and heading_date is not None and str(heading_date)!='nan' 
                and str(heading_date)!='null'):
                heading_date = pd.to_datetime(str(heading_date), format='%Y-%m-%d')
            elif ('Days_To_Heading' in self.attributes.keys()):
                Days_To_Heading = self.attributes['Days_To_Heading']
                if (Days_To_Heading!='None' and Days_To_Heading is not None and str(Days_To_Heading)!='nan' 
                    and str(Days_To_Heading)!='null'):
                    self.attributes['heading'] = str(sowing_date + pd.DateOffset(days=int(Days_To_Heading))).split(' ')[0]

            if (maturity_date!='None' and maturity_date is not None and str(maturity_date)!='nan' 
                and str(maturity_date)!='null'):
                maturity_date = pd.to_datetime(str(maturity_date), format='%Y-%m-%d')
            elif ('Days_To_Maturity' in self.attributes.keys()):
                Days_To_Maturity = self.attributes['Days_To_Maturity']
                if (Days_To_Maturity!='None' and Days_To_Maturity is not None and str(Days_To_Maturity)!='nan' 
                    and str(Days_To_Maturity)!='null'):
                    self.attributes['maturity'] = str(sowing_date + pd.DateOffset(days=int(Days_To_Maturity))).split(' ')[0]

            #if (sowing_date is not None and heading_date is not None):
            #    self.attributes['ObsDays_SH'] = (pd.to_datetime(str(heading_date)) - pd.to_datetime(str(sowing_date))).days

            # Get Weather for growing season
            if (self.weather is None):
                if (verbose is True):
                    print("Getting weather data for the site during the growing cycle...")
                self.weather = self.getWeather(m.config['WeatherFile'])
        except Exception as err:
            print(HT.bold + "Error "+HT.reset +"getting phenology dates {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Estimating phenology dates. Error: {}".format(err)})
        
        #print("PhenoDates -> ", sowing_date, emergence_date, heading_date, maturity_date)
        return sowing_date, emergence_date, heading_date, maturity_date
    
    
    # ---------------------------------------------
    # Estimate Emergence
    # ---------------------------------------------
    def getEstimatedEmergence(self, m=None, verbose=True):
        '''
            Get Estimated Emergence
            
        '''
        # Get Weather for growing season
        if (self.weather is None):
            if (verbose is True):
                print("Getting weather data for the site during the growing cycle...")
            self.weather = self.getWeather(m.config['WeatherFile'])
        
        if (verbose is True):
            print("Estimating Emergence from GDD...")
        if ('sowing' in self.attributes.keys()):
            s = self.attributes['sowing']
            if (s=='None' or s is None or str(s)=='nan' or str(s)=='null'):
                print("Sowing date not valid")
                return
        #
        # Here we should get phenology dates beyond the matrurity date, needed for simulations
        e = str(pd.to_datetime(str(s)) + pd.DateOffset(days=int(365))).split(' ')[0] 
        self.pheno_dates = pd.date_range(start=s, end=e) #, inclusive="both")
        #if (verbose is True): print("Start - End Dates -> ",s, e)
        GDD = None
        try:
            if (verbose is True):
                print("Calculating Growing degree days (GDD) from Heading to Maturity dates...")
            self.GDD = self.getGDD(m, self.weather)
            cGDD = np.cumsum(self.GDD)
            self.attributes['PredEmergence'] = self.getEmergenceDate(m, cGDD, verbose=verbose) #, GDDreq=None)
            self.attributes['PredDaysToEmergence'] = (pd.to_datetime(str(self.attributes['PredEmergence'] )) - 
                                                      pd.to_datetime(str(s))).days
        except Exception as err:
            print(HT.bold +"Error "+HT.reset+"estimating Emergence date {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Estimating Emergence date. Error: {}".format(err)})
        
        del GDD
        _ = gc.collect()
            
    # ----------------------------------------------
    # Estimate Heading
    # ----------------------------------------------
    def getEstimatedHeading(self, m=None, verbose=True):
        '''
            Get Estimated Heading
            
        '''
        if ('sowing' in self.attributes.keys()):
            s = self.attributes['sowing']
            if (s=='None' or s is None or str(s)=='nan' or str(s)=='null'):
                print("Sowing date not valid")
                return
        dAS, dlength, DaysToHead = None, None, None
        try:
            if (verbose is True):
                print("Estimating days to heading after sowing date")
                # Adding date and daylength "35" days after planting
            if (verbose is True):
                print("Getting Date {} days after planting and Daylength".format(m.parameters['DAP']))
            dAS, dlength = self.getDayLengthDAP(m) #DAP=None
            self.attributes['Date@{}DAS'.format(m.parameters['DAP'])] = str(dAS).split(' ')[0]
            self.attributes['DayLength@{}DAS'.format(m.parameters['DAP'])] = dlength

            DaysToHead = self.getDaysToHeading(m, dlength)
            self.attributes['PredDaysToHead'] = DaysToHead
            self.attributes['PredHeading'] = str(pd.to_datetime(str(s)) + pd.DateOffset(days=int(DaysToHead))).split(' ')[0]
        except Exception as err:
            print(HT.bold +"Error "+HT.reset+"estimating Heading date {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Estimating Heading date. Error: {}".format(err)})
            
        del dAS, dlength, DaysToHead
        _ = gc.collect()
            
    # ----------------------------------------------
    # Estimate Maturity
    # ----------------------------------------------
    def getEstimatedMaturity(self, m=None, threshold=42, daysGF=None, verbose=True):
        '''
            Get Estimated Maturity from Observed Heading and Predicted Heading
            
        '''
        if ('sowing' in self.attributes.keys()):
            s = self.attributes['sowing']
            if (s=='None' or s is None or str(s)=='nan' or str(s)=='null'):
                print("Sowing date not valid")
                return
        if (verbose is True):
            print("Estimating Maturity date...")
        try:
            Sim_Maturity_date_H, Sim_DaysToMaturity_H, Sim_DaysHM, \
            Sim_Maturity_date_pH, Sim_DaysToMaturity_pH, Sim_DayspHM = self.getMaturityDate(m, threshold=threshold, 
                                                                                     daysGF=daysGF, verbose=verbose)
            #if ((Sim_Maturity_date is not None) and (Sim_DaysToMaturity is not None) and (Sim_DaysHM is not None) ):
            self.attributes['PredMaturity_H'] = str(Sim_Maturity_date_H).split(' ')[0]
            self.attributes['PredDays_HM'] = Sim_DaysHM
            self.attributes['PredDaysToMaturity_H'] = Sim_DaysToMaturity_H
            
            self.attributes['PredMaturity_pH'] = str(Sim_Maturity_date_pH).split(' ')[0]
            self.attributes['PredDays_pHM'] = Sim_DayspHM
            self.attributes['PredDaysToMaturity_pH'] = Sim_DaysToMaturity_pH
            
        except Exception as err:
            print(HT.bold +"Error "+HT.reset+"estimating Maturity date {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Estimating Maturity date. Error: {}".format(err)})

    # ----------------------------------------------
    # Estimated Phenology
    # ----------------------------------------------
    def getEstimatedPhenologyDates(self, m=None, verbose=False):
        '''
            Estimated Phenology (Emergence, Heading, Maturity)
            
        '''
        try:
            if (verbose is True):
                print("Estimating phenology (Emergence, Heading, Maturity)")
            # Estimate Emergence
            self.getEstimatedEmergence(m, verbose=verbose)
            # Estimate Heading
            self.getEstimatedHeading(m, verbose=verbose)
            # Estimate Maturity
            self.getEstimatedMaturity(m, verbose=verbose) 
        except Exception as err:
            print(HT.bold +"Error "+HT.reset+"estimating phenology date {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Estimating phenology dates. Error: {}".format(err)})
    
    # ---------------
    # Climate data
    # ---------------
    def getWeather(self, weatherDF=None, s=None, e=None):
        ''' Get phenology dates for growing period 
        
        :param weatherDF: A pandas dataframe with weather data
        :param s: Start date eg. Sowing date
        :param e: End date eg. Maturity date

        :return: a pandas dataframe with the filtered weather data, or None if error occurs

        '''
        self.weather = None
        if (weatherDF is None):
            print("Source weather data not found")
            return
        try:
            if (s is None):
                if ('sowing' in self.attributes.keys()):
                    s = self.attributes['sowing']
                if (e is None):
                    # Here we need to get weather data beyond the matrurity date, due to simulations
                    # We are using a year ahead of sowing date
                    e = str(pd.to_datetime(str(s)) + pd.DateOffset(days=int(365))).split(' ')[0]
                
            if (self.loc is None):
                print("Location don't exist or not valid")
                return
            # Get weather
            if (s=='None' or s is None or str(s)=='nan' or str(s)=='null' 
                or e=='None' or e is None or str(e)=='nan' or str(e)=='null'):
                print("Problem getting weather data for the location: {}".format(self.loc))
            else:
                _mask = ((weatherDF['Date'] >= s) & (weatherDF['Date'] <= e) & (weatherDF['location']==self.loc) )
                self.weather = weatherDF[_mask].reset_index(drop=True)
        except Exception as err:
            print("Error getting weather data {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Getting weather data. Error: {}".format(err)})

        return self.weather
        
    def getSowingDOY(self):
        ''' Get the Day of the Year for Sowing date '''
        day_of_year = None
        if ('sowing' in self.attributes.keys()):
            s = self.attributes['sowing']
            #day_of_year = pd.Timestamp(s).dayofyear 
            day_of_year = datetime.strptime(s, "%Y-%m-%d").timetuple().tm_yday
        return day_of_year
    
    # ---------------------------------------
    # Calculate Daylength
    # ---------------------------------------
    def getDayLength(self, d=None, lat=None, p=0.0):
        ''' Get Day length '''
        if (d is None):
            print("Date for getting daylength is not valid")
            return
        if (lat is None):
            if ('lat' in self.attributes.keys()):
                lat = self.attributes['lat']
            else:
                print("Latitude not valid")
                return
        dlength = None
        try:
            dlength = daylength.calculateDayLength(d, lat, p)
        except Exception as err:
            print("Error getting Day length {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Getting Daylength. Error: {}".format(err)})

        return float("{:.2f}".format(dlength))
    
    def getDayLengthDAP(self, m=None, DAP=None):
        ''' Get Date after planting and Day length 
        
        :params DAP: Days after sowing date
        
        :return: tuple with Date and Daylength
        '''
        dAS = None
        dlength = None
        if (DAP is None and m is not None):
            DAP = m.parameters["DAP"]
        elif (DAP is None):
            DAP = model.PARAMETERS["DAP"]
        #
        try:
            if ('sowing' in self.attributes.keys()):
                s = self.attributes['sowing']
                dAS = pd.to_datetime(str(s)) + pd.DateOffset(days=int(DAP))
                dlength = self.getDayLength(str(dAS).split(' ')[0])
        except Exception as err:
            print("Problem getting sowing date for estimating Daylength {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Getting sowing date for estimating Daylength. Error: {}".format(err)})
        
        return dAS, dlength
    
    # ---------------------------------------
    # Calculate day time temperature - TDay
    # ---------------------------------------
    def getTDay(self, m=None, w=None, tminFactor=None):
        '''Calculate day time temperature
        
        :param w: Table of weather data with Minimum and Maximum Temperatures
        :param tminFactor: Minimum Temperature factor

        :return: a number or array of Day Temperature
        
        ''' 
        result = []
        if (w is None):
            print("Weather data not valid")
            return
        if (tminFactor is None and m is not None):
            tminFactor = m.parameters["TMIN_PERC_FACTOR"]
        elif (tminFactor is None):
            tminFactor = model.PARAMETERS["TMIN_PERC_FACTOR"]
        try:
            if (('TMIN' in list(w)) and ('TMAX' in list(w)) ) :
                result = tday.estimate_TDay(w['TMIN'].to_numpy(), w['TMAX'].to_numpy(), tminFactor )
            else:
                print("Values for TMIN and TMAX were not found")
        except Exception as err:
            print("Error calculating Day temperature {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Calculating Day temperature. Error: {}".format(err)})
            
        return result
    
    # ---------------------------------------
    # Calculate Growing degree days - GDD
    # ---------------------------------------
    def getGDD(self, m=None, w=None, Tbase=0):
        ''' Growing degree days GDD (°F or °C)
            Calculated from: ((Daily Max Temp + Daily Min Temp)/2) - 32 °F (or 
            ((Daily Max Temp + Daily Min Temp)/2)).

        :param Tmin: Number or array of Minimum Temperatures
        :param Tmax: Number or array of Maximum Temperatures
        :param Tbase: Temperature base of the crop

        :return: a number or array of Growing degree days (GDD)

        '''
        result = []
        if (w is None):
            print("Weather data not valid")
            return
        if (Tbase is None and m is not None):
            Tbase = m.parameters["CROP_TBASE_GDD"]
        elif (Tbase is None):
            Tbase = model.PARAMETERS["CROP_TBASE_GDD"]
        try:
            if (('TMIN' in list(w)) and ('TMAX' in list(w)) ) :
                result = gdd.calculateGDD(w['TMIN'].to_numpy(), w['TMAX'].to_numpy(), Tbase )
            else:
                print("Values for TMIN and TMAX were not found")
        except Exception as err:
            print("Error calculating Growing degree days {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Calculating Growing degree days. Error: {}".format(err)})
            
        return result
    
    # ------------------------------------------------
    # Estimate Photosynthesis reduction factor - PRFT
    # ------------------------------------------------
    def getPRFT(self, m=None, TDay=None, TOpt=None):
        ''' Estimate Photosynthesis reduction factor (PRFT)
            PRFT = 1 – 0.0025 * (TDay – TOpt)^2

        :param TDay: Number or array of Day Temperatures
        :param TOpt: Optimum Temperature
        
        :return: a number or array of PRFT

        '''
        if (TDay is None):
            print("Day Temperature parameter is not valid")
            return
        if (TOpt is None and m is not None):
            TOpt = m.parameters["CROP_OPTIMUM_TEMPERATURE"]
        elif (TOpt is None):
            TOpt = model.PARAMETERS["CROP_OPTIMUM_TEMPERATURE"]
        try:
            result = prft.calculatePRFT(TDay, TOpt )
        except Exception as err:
            print("Error calculating photosynthesis reduction factor {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Calculating photosynthesis reduction factor. Error: {}".format(err)})
            
        return result
    
    # ------------------------------------------------
    # Estimate emergence date
    # ------------------------------------------------
    def getEmergenceDate(self, m=None, cGDD=None,  GDDreq=None, verbose=False):
        '''Estimate emergence date

           Day on which cumulative thermal time reaches 180 GDD (Tbase = 0ºC)

           :params cGDD: Cumulative Sum of GDD
           :params GDDreq: GDD or thermal time required to emergence
           :params ts: convert date to Timestamp
           
           :return: an emergence date
        '''
        if (GDDreq is None and m is not None):
            GDDreq = m.parameters["GDD_Required_to_EmergenceDate"]
        elif (GDDreq is None):
            GDDreq = model.PARAMETERS["GDD_Required_to_EmergenceDate"]
        if (cGDD is None or len(cGDD)<=0):
            print("Cumulative thermal time not valid")
            return
        if (m is None):
            print("Weather data not valid")
            return
        if (self.pheno_dates is None):
            if (verbose is True):
                print("Get phenology array of dates from Heading to Maturity...")
            _ = self.getRangeDates(verbose=verbose)
        #
        idx = np.argmin(np.abs(cGDD - GDDreq))
        emergdate = None
        try:
            if (self.pheno_dates is not None):
                #emergdate = self.weather.loc[self.weather.index==idx, 'Date']
                emergdate = str(self.pheno_dates[idx]).split(' ')
                #print("emergdate -> ",emergdate)
                if ((emergdate is not None) and (len(emergdate)>0)):
                    emergdate = emergdate[0]
                if (verbose is True):
                    print(HT.bold + "Estimated emergence date ->"+ HT.reset, emergdate)
        except Exception as err:
            print(HT.bold + "Error estimating emergence date"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Estimating emergence date. Error: {}".format(err)})
            
        return emergdate
    
    # ---------------------------------------------------
    # Estimate days to heading
    # ---------------------------------------------------
    def getDaysToHeading(self, m=None, dayHR=None):
        '''Estimate days to heading after sowing date
        
        :params dayHR: Daylength 
        
        The following Curves are only applied for ESWYT, IDYN and HTWYT data sets
        If daylength @ 35 DAS < 10.8 hrs
            DAYS_TO_HEADING = 491.27 - 38.62 * DayLength @ 35DAS  (daylength in hours)
        If daylength @ 35 DAS > 10.8 hrs
            DAYS_TO_HEADING = 115.36 - 3.87 * DayLength @ 35DAS
        
        SAWYT is earlier
        If daylength @ 35 DAS < 10.8 hrs
            DAYS_TO_HEADING = 617.68 - 51.406 * DayLength@35DAS  (daylength in hours)
        If daylength @ 35 DAS > 10.8 hrs
            DAYS_TO_HEADING = 87.38 – 2.36 * DayLength @ 35DAS

        
        :return: Estimated days to heading after sowing date
        '''
        DaysToHead = 0
        if (('Nursery' in self.attributes) and (self.attributes['Nursery']=='SAWYT')):
            # The following Curves are only applied for SAWYT dataset
            if ( dayHR < m.parameters['CONST_DAYHR_AS'] ):
                DaysToHead = 617.68 - (51.406 * dayHR)
            elif ( dayHR >= m.parameters['CONST_DAYHR_AS'] ):
                DaysToHead = 87.38 - (2.36 * dayHR)
        else:
            # The following Curves are only applied for ESWYT, IDYN and HTWYT data sets
            if ( dayHR < m.parameters['CONST_DAYHR_AS'] ):
                #DaysToHead = 528.55 - (42.479 * dayHR) # 35 days
                DaysToHead = 491.27 - (38.62 * dayHR) # 35 days
            elif ( dayHR >= m.parameters['CONST_DAYHR_AS'] ):
                DaysToHead = 115.36 - (3.87 * dayHR) #70
        return int(DaysToHead)
    
    def _getDaysToHeading_SAWYT(self, m=None, dayHR=None):
        '''Estimate days to heading after sowing date
        
        The following Curves are only applied for SAWYT dataset
        
        SAWYT is earlier
        If daylength @ 35 DAS < 10.8 hrs
            DAYS_TO_HEADING = 617.68 - 51.406 * DayLength@35DAS  (daylength in hours)
        If daylength @ 35 DAS > 10.8 hrs
            DAYS_TO_HEADING = 87.38 – 2.36 * DayLength @ 35DAS

        
        :params dayHR: Daylength 
        
        :return: Estimated days to heading after sowing date
        '''
        DaysToHead = 0
        if ( dayHR < m.parameters['CONST_DAYHR_AS'] ):
            #DaysToHead = 607.3989 - (50.045799 * dayHR)
            #DaysToHead = 584.98477 - (48.123794 * dayHR)
            DaysToHead = 617.68 - (51.406 * dayHR)
        elif ( dayHR >= m.parameters['CONST_DAYHR_AS'] ):
            #DaysToHead = 83.480844 - (1.7843309 * dayHR) #58
            DaysToHead = 87.38 - (2.36 * dayHR)
        return int(DaysToHead)
    
    # ---------------------------------------------------
    # Estimate Maturity date
    # ---------------------------------------------------
    def getMaturityDate(self, m=None, threshold=42, daysGF=None, verbose=False):
        ''' Estimate maturity date from weather using TAdjDays 
            
            :params m: model with the global configuration, parameters and data
            :params threshold: A threshold to adjust the number of temperature days
            :params daysGF: Days to grain filling
            
            :return: Estimated days to maturity, Maturity date and number of days between heading to maturity date.
            
        '''
        # Assumption: duration of grain filling is 40 days (we may have to make this longer)
        if (daysGF is None and m is not None):
            daysGF = m.parameters["DAYS_GRAIN_FILLING"]
        elif (daysGF is None):
            daysGF = model.PARAMETERS["DAYS_GRAIN_FILLING"]
        if (threshold is None and m is not None):
            threshold = m.parameters["TDAYS_THRESHOLD_MATURITY"]
        elif (threshold is None):
            threshold = model.PARAMETERS["TDAYS_THRESHOLD_MATURITY"]
        if (self.weather is None):
            print("Weather data is not valid")
            return
        sowing_date = None
        heading_date = None
        pred_heading_date = None
        Sim_Maturity_date_H, Sim_DaysToMaturity_H, Sim_DaysHM = None, None, None
        Sim_Maturity_date_pH, Sim_DaysToMaturity_pH, Sim_DayspHM = None, None, None
        df_tmp = None
        try:
            if ('sowing' in self.attributes.keys()):
                sowing_date = self.attributes['sowing']
            if ('heading' in self.attributes.keys()):
                heading_date = self.attributes['heading']
            
            if (sowing_date=='None' or sowing_date is None or str(sowing_date)=='nan' or str(sowing_date)=='null'):
                sowing_date = None
                print("Input phenology is not valid...")
                return
            else:
                sowing_date = pd.to_datetime(str(sowing_date), format='%Y-%m-%d')
                if (verbose is True):
                    print(HT.bold + "Observed sowing date -> "+ HT.reset, str(sowing_date).split(' ')[0])
            
            if (heading_date=='None' or heading_date is None or str(heading_date)=='nan' or str(heading_date)=='null'):
                heading_date = None
            else:
                heading_date = pd.to_datetime(str(heading_date), format='%Y-%m-%d')
                if (verbose is True):
                    print(HT.bold + "Observed heading date -> "+ HT.reset, str(heading_date).split(' ')[0])
            
            # Use estimated Heading
            if ( 'PredHeading' in self.attributes.keys()):
                pred_heading_date = self.attributes['PredHeading']
                if (pred_heading_date=='None' or pred_heading_date is None or str(pred_heading_date)=='nan' 
                    or str(pred_heading_date)=='null'):
                    pred_heading_date = None
                else:
                    pred_heading_date = pd.to_datetime(str(pred_heading_date), format='%Y-%m-%d')
                    if (verbose is True):
                        print(HT.bold + "Estimated heading date -> " + HT.reset, str(pred_heading_date).split(' ')[0])
            
            if (heading_date is not None):
                if (verbose==True):
                    print("\nEstimating Maturity from Observed Heading...")
                df_tmp_H = self.weather[(self.weather['Date'] >= heading_date)].reset_index(drop=True)
                #DailyTWeight =  150 * np.exp(-0.06 * df_tmp['TAVG'].to_numpy())
                # Calculate daily progress (Adjusted TDays) - IWIN: 42, Dhillon: 35, South Asia: 41
                TAdjDay_H = tadjday.estimate_TAdjDay(df_tmp_H['TAVG'].to_numpy(), threshold)
                cTAdjDays_HM = np.cumsum(TAdjDay_H)
                n_index_H = (np.abs(cTAdjDays_HM - daysGF)).argmin()
                if (verbose==True):
                    print("Nearest index or TAdjDay after observed heading: {}".format(n_index_H))
                    print("Nearest value: {}".format(cTAdjDays_HM[n_index_H]))
                # Get phenology date for Estimated Maturity
                Sim_Maturity_date_H = df_tmp_H.loc[n_index_H, 'Date']
                Sim_DaysHM = (Sim_Maturity_date_H - heading_date).days
                Sim_DaysToMaturity_H = (Sim_Maturity_date_H - sowing_date).days
                if (verbose==True):
                    print(HT.bold + "Estimated Maturity date using observed heading ->"+HT.reset+" {}"
                          .format(str(Sim_Maturity_date_H).split(' ')[0] ))
                    print("Estimated Days to Maturity using observed heading : {}".format(Sim_DaysToMaturity_H))
                    print("Estimated Days from Heading to Maturity: {}".format(Sim_DaysHM))
            
            if (pred_heading_date is not None):
                if (verbose==True):
                    print("\nEstimating Maturity from estimated Heading...")
                df_tmp_pH = self.weather[(self.weather['Date'] >= pred_heading_date)].reset_index(drop=True)
                # Calculate daily progress (Adjusted TDays) - IWIN: 42, Dhillon: 35, South Asia: 41
                TAdjDay_pH = tadjday.estimate_TAdjDay(df_tmp_pH['TAVG'].to_numpy(), threshold)
                cTAdjDays_pHM = np.cumsum(TAdjDay_pH)
                n_index_pH = (np.abs(cTAdjDays_pHM - daysGF)).argmin()
                if (verbose==True):
                    print("Nearest index or TAdjDay after estimated heading: {}".format(n_index_pH))
                    print("Nearest value: {}".format(cTAdjDays_pHM[n_index_pH]))
                # Get phenology date for Estimated Maturity
                Sim_Maturity_date_pH = df_tmp_pH.loc[n_index_pH, 'Date']
                Sim_DayspHM = (Sim_Maturity_date_pH - pred_heading_date).days
                Sim_DaysToMaturity_pH = (Sim_Maturity_date_pH - sowing_date).days
                if (verbose==True):
                    print(HT.bold + "Estimated Maturity date using estimated heading ->"+HT.reset+" {}"
                          .format(str(Sim_Maturity_date_pH).split(' ')[0] ))
                    print("Estimated Days to Maturity using estimated heading : {}".format(Sim_DaysToMaturity_pH))
                    print("Estimated Days from estimated Heading to Maturity: {}".format(Sim_DayspHM))
        except Exception as err:
            print(HT.bold + HT.fg.red + "Problem estimating maturity date from weather using TAdjDays"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Estimating maturity date from weather using TAdjDays. Error: {}".format(err)})
        
        del df_tmp
        _ = gc.collect()
        return Sim_Maturity_date_H, Sim_DaysToMaturity_H, Sim_DaysHM, Sim_Maturity_date_pH, Sim_DaysToMaturity_pH, Sim_DayspHM
    
    # ---------------------------------------------------
    # Get masks or filters for periods
    # ---------------------------------------------------
    def getFilters(self, m=None, season=False, verbose=False):
        '''
        
        '''
        if (self.weather is None):
            print("Weather data is not valid to get additional parameters")
            return
        # Get phenology dates
        sowing_date = None
        emergence_date = None
        heading_date = None
        pred_heading_date = None
        maturity_date = None
        pred_maturity_date = None
        if ('sowing' in self.attributes.keys()):
            sowing_date = self.attributes['sowing']
        if ('emergence' in self.attributes.keys()):
            emergence_date = self.attributes['emergence']
        if ('heading' in self.attributes.keys()):
            heading_date = self.attributes['heading']
        if ('maturity' in self.attributes.keys()):
            maturity_date = self.attributes['maturity']
            
        # Check for empty values
        if (sowing_date=='None' or sowing_date is None or str(sowing_date)=='nan' or str(sowing_date)=='null'):
            print("Error getting phenology dates. Sowing date not valid.")
            return
        if (emergence_date=='None' or emergence_date is None or str(emergence_date)=='nan' or str(emergence_date)=='null'):
            if (verbose is True):
                print("Warning: problem getting phenology dates. Observed emergence date is not valid.")
            emergence_date = None
        if (heading_date=='None' or heading_date is None or str(heading_date)=='nan' or str(heading_date)=='null'):
            if (verbose is True):
                print("Warning: Observed heading date is not valid.")
            heading_date = None
        if (maturity_date=='None' or maturity_date is None or str(maturity_date)=='nan' or str(maturity_date)=='null'):
            if (verbose is True):
                print("Warning: Observed maturity date is not valid.")
            maturity_date = None
        
        #print(sowing_date, emergence_date, heading_date, maturity_date)
        _mask_SE=None
        _mask_SM=None
        _mask_SH=None
        _mask_SpE=None
        _mask_SpM=None
        _mask_SpH=None
        _mask_EH=None
        _mask_EM=None
        _mask_EpH=None
        _mask_pEH=None
        _mask_pEM=None
        _mask_pEpH=None
        _mask_pEpM=None
        _mask_HM=None
        _mask_HpM=None
        _mask_pHM=None
        _mask_pHpM=None
        _mask_dDAS_H=None
        _mask_dDAS_pH=None
        try:
            if (emergence_date is not None):
                _mask_SE = ((self.weather['Date'] > sowing_date) & (self.weather['Date'] <= emergence_date))
            else:
                _mask_SE = None
            
            if ((emergence_date is not None) and (heading_date is not None) ):
                _mask_EH = ((self.weather['Date'] > emergence_date) & (self.weather['Date'] <= heading_date))
            else:
                _mask_EH = None
            
            if ((emergence_date is not None) and (maturity_date is not None) ):
                _mask_EM = ((self.weather['Date'] > emergence_date) & (self.weather['Date'] <= maturity_date))
            else:
                _mask_EM = None
            
            if ((heading_date is not None) and (maturity_date is not None) ):
                _mask_HM = ((self.weather['Date'] > heading_date) & (self.weather['Date'] <= maturity_date))
                if (str(heading_date) >= str(maturity_date)):
                    _mask_HM = None
                    if (verbose is True):
                        print("\n"+ HT.bold + HT.fg.red + 
                              "Error: Observed heading is equal or greater than maturity plot {} - {}"
                              .format(self.uid, self.loc) + HT.reset)
                    self.errors.append({"uid": self.uid, "loc": self.loc, 
                                        "error": "Observed heading is equal or greater than maturity"})
            else:
                _mask_HM = None
                
            if ((sowing_date is not None) and (heading_date is not None) ):
                _mask_SH = ((self.weather['Date'] > sowing_date) & (self.weather['Date'] <= heading_date))
            else:
                _mask_SH = None
                
            if ((sowing_date is not None) and (maturity_date is not None) ):
                _mask_SM = ((self.weather['Date'] > sowing_date) & (self.weather['Date'] <= maturity_date))
            else:
                _mask_SM = None
            
            if ('PredEmergence' in self.attributes.keys()):
                pred_emergence_date = self.attributes['PredEmergence']
                if (pred_emergence_date=='None' or pred_emergence_date is None or str(pred_emergence_date)=='nan' or str(pred_emergence_date)=='null'):
                    print("Estimated emergence not found")
                    _mask_SpE = None
                    _mask_pEH = None
                    _mask_pEM = None
                else:
                    _mask_SpE = ((self.weather['Date'] > sowing_date) & (self.weather['Date'] <= pred_emergence_date))
                    if (heading_date is not None):
                        _mask_pEH = ((self.weather['Date'] > pred_emergence_date) & (self.weather['Date'] <= heading_date))
                    else:
                        _mask_pEH = None
                    if (maturity_date is not None):
                        _mask_pEM = ((self.weather['Date'] > pred_emergence_date) & (self.weather['Date'] <= maturity_date))
                    else:
                        _mask_pEM = None
            else:
                _mask_SpE = None
                _mask_pEH = None
                _mask_pEM = None
            
            if ('PredHeading' in self.attributes.keys()):
                pred_heading_date = self.attributes['PredHeading']
                if (pred_heading_date=='None' or pred_heading_date is None or str(pred_heading_date)=='nan' 
                    or str(pred_heading_date)=='null'):
                    print("Estimated heading not found")
                    _mask_SpH = None
                    _mask_pHM = None
                    _mask_EpH = None
                    _mask_pEpH = None
                else:
                    if (sowing_date is not None):
                        _mask_SpH = ((self.weather['Date'] > sowing_date) & (self.weather['Date'] <= pred_heading_date))
                    else:
                        _mask_SpH = None
                    if (emergence_date is not None):
                        _mask_EpH = ((self.weather['Date'] > emergence_date) & (self.weather['Date'] <= pred_heading_date))
                    else:
                        _mask_EpH = None
                    if (maturity_date is not None):
                        _mask_pHM = ((self.weather['Date'] > pred_heading_date) & (self.weather['Date'] <= maturity_date))
                        # Sometime the estimated heading is greater than maturity
                        if (str(pred_heading_date) >= str(maturity_date)):
                            if (verbose is True):
                                print("\n"+ HT.bold + HT.fg.red + 
                                      "Error: Estimated heading is equal or greater than maturity plot {} - {}"
                                      .format(self.uid, self.loc) + HT.reset)
                            self.errors.append({"uid": self.uid, "loc": self.loc, 
                                                "error": "Estimated heading is equal or greater than maturity"})
                    else:
                        _mask_pHM = None
                    if (pred_emergence_date is not None):
                        _mask_pEpH = ((self.weather['Date'] > pred_emergence_date) & 
                                      (self.weather['Date'] <= pred_heading_date))
                
            else:
                _mask_SpH = None
                _mask_pHM = None
                _mask_EpH = None
                _mask_pEpH = None
            
            if ('PredMaturity_H' in self.attributes.keys()):
                pred_maturity_date = self.attributes['PredMaturity_H']
                if (pred_maturity_date=='None' or pred_maturity_date is None or str(pred_maturity_date)=='nan' 
                    or str(pred_maturity_date)=='null'):
                    print("Estimated maturity not found")
                    _mask_HpM = None
                    _mask_pEpM = None
                else:
                    print(sowing_date, heading_date, pred_maturity_date)
                    if (sowing_date is not None):
                        _mask_SpM = ((self.weather['Date'] > sowing_date) & (self.weather['Date'] <= pred_maturity_date))
                    else:
                        _mask_SpM = None
                    if (heading_date is not None):
                        _mask_HpM = ((self.weather['Date'] > heading_date) & (self.weather['Date'] <= pred_maturity_date))
                        if (str(heading_date) >= str(pred_maturity_date)):
                            _mask_HpM = None
                            if (verbose is True):
                                print("\n"+ HT.bold + HT.fg.red + 
                                      "Error: Observed heading is equal or greater than estimated maturity plot {} - {}"
                                      .format(self.uid, self.loc) + HT.reset)
                            self.errors.append({"uid": self.uid, "loc": self.loc, 
                                                "error": "Observed heading is equal or greater than estimated maturity"})
                    else:
                        _mask_HpM = None
                    if (pred_emergence_date is not None):
                        _mask_pEpM = ((self.weather['Date'] > pred_emergence_date) & 
                                      (self.weather['Date'] <= pred_maturity_date))
                    else:
                        _mask_pEpM = None
                    
            else:
                _mask_HpM = None
                _mask_pEpM = None
            

            if ('PredMaturity_pH' in self.attributes.keys()):
                pred_maturity_date = self.attributes['PredMaturity_pH']
                if (pred_maturity_date=='None' or pred_maturity_date is None or str(pred_maturity_date)=='nan' 
                    or str(pred_maturity_date)=='null'):
                    print("Estimated maturity from estimated heading not found")
                    _mask_pHpM = None
                else:
                    _mask_pHpM = ((self.weather['Date'] > pred_heading_date) & 
                                 (self.weather['Date'] <= pred_maturity_date))
                    if (str(pred_heading_date) >= str(pred_maturity_date)):
                        _mask_pHpM = None
                        if (verbose is True):
                            print("\n"+ HT.bold + HT.fg.red + 
                                  "Error: Estimated heading is equal or greater than estimated maturity plot {} - {}"
                                  .format(self.uid, self.loc) + HT.reset)
                        self.errors.append({"uid": self.uid, "loc": self.loc, 
                                            "error": "Estimated heading is equal or greater than estimated maturity"})
            else:
                _mask_pHpM = None
            
            if ('Date@{}DAS'.format(m.parameters['DAP']) in self.attributes.keys()):
                dDAS = self.attributes['Date@{}DAS'.format(m.parameters['DAP'])]
                if (heading_date is not None):
                    _mask_dDAS_H = ((self.weather['Date'] > dDAS) & (self.weather['Date'] <= heading_date))
                else:
                    _mask_dDAS_H = None
                if (pred_heading_date is not None):
                    _mask_dDAS_pH = ((self.weather['Date'] > dDAS) & (self.weather['Date'] <= pred_heading_date))
                else:
                    _mask_dDAS_pH = None
            
            # ---------------------------------------------------------
            # Number of days from periods
            # ---------------------------------------------------------
            if (_mask_SE is not None):
                # (pd.Timestamp(pred_maturity_date) - pd.Timestamp(pred_heading_date)).days
                self.attributes['Days_SE'] = len(self.weather[_mask_SE]) #-1
                # Double checking for masks with more than 0 rows
                if (self.attributes['Days_SE']<=0): _mask_SE = None 
            if (_mask_SpE is not None):
                self.attributes['Days_SpE'] = len(self.weather[_mask_SpE])
                if (self.attributes['Days_SpE']<=0): _mask_SpE = None
            if (_mask_EH is not None):
                self.attributes['Days_EH'] = len(self.weather[_mask_EH]) #-1
                if (self.attributes['Days_EH']<=0): _mask_EH = None
            if (_mask_pEH is not None):
                self.attributes['Days_pEH'] = len(self.weather[_mask_pEH])
                if (self.attributes['Days_pEH']<=0): _mask_pEH = None
            if (_mask_EpH is not None):
                self.attributes['Days_EpH'] = len(self.weather[_mask_EpH])
                if (self.attributes['Days_EpH']<=0): _mask_EpH = None
            if (_mask_pEpH is not None):
                self.attributes['Days_pEpH'] = len(self.weather[_mask_pEpH])
                if (self.attributes['Days_pEpH']<=0): _mask_pEpH = None
            if (_mask_EM is not None):
                self.attributes['Days_EM'] = len(self.weather[_mask_EM])
                if (self.attributes['Days_EM']<=0): _mask_EM = None
            if (_mask_pEM is not None):
                self.attributes['Days_pEM'] = len(self.weather[_mask_pEM])
                if (self.attributes['Days_pEM']<=0): _mask_pEM = None
            if (_mask_pEpM is not None):
                self.attributes['Days_pEpM'] = len(self.weather[_mask_pEpM])
                if (self.attributes['Days_pEpM']<=0): _mask_pEpM = None
            if (_mask_SM is not None):
                self.attributes['Days_SM'] = len(self.weather[_mask_SM])
                if (self.attributes['Days_SM']<=0): _mask_SM = None
            if (_mask_SpM is not None):
                self.attributes['Days_SpM'] = len(self.weather[_mask_SpM])
                if (self.attributes['Days_SpM']<=0): _mask_SpM = None
            if (_mask_SH is not None):
                self.attributes['Days_SH'] = len(self.weather[_mask_SH])
                if (self.attributes['Days_SH']<=0): _mask_SH = None
            if (_mask_SpH is not None):
                self.attributes['Days_SpH'] = len(self.weather[_mask_SpH])
                if (self.attributes['Days_SpH']<=0): _mask_SpH = None
            if (_mask_HM is not None):
                self.attributes['Days_HM'] = len(self.weather[_mask_HM])
                if (self.attributes['Days_HM']<=0): _mask_HM = None
            if (_mask_HpM is not None):
                self.attributes['Days_HpM'] = len(self.weather[_mask_HpM])
                if (self.attributes['Days_HpM']<=0): _mask_HpM = None
            if (_mask_pHM is not None):
                self.attributes['Days_pHM'] = len(self.weather[_mask_pHM])
                if (self.attributes['Days_pHM']<=0): _mask_pHM = None
            if (_mask_pHpM is not None):
                self.attributes['Days_pHpM'] = len(self.weather[_mask_pHpM])
                if (self.attributes['Days_pHpM']<=0): _mask_pHpM = None
            
            # ---------------------------------------------------------
            # Weather for the whole growing season, and periods
            # ---------------------------------------------------------
            self.updateWeatherparams(_mask_SE, _mask_SpE, _mask_EH, _mask_pEH, _mask_EpH, _mask_pEpH, _mask_EM, _mask_pEM,
                                     _mask_pEpM, _mask_SM, _mask_SpM, _mask_SH, _mask_SpH, _mask_HM, _mask_HpM, _mask_pHM, 
                                     _mask_pHpM, _mask_dDAS_H, _mask_dDAS_pH,
                                     season=season, verbose=verbose)
        except Exception as err:
            print(HT.bold + HT.fg.red + "Error getting filters for weather"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, 
                                "error": "Getting filters for weather. Error: {}".format(err)})
        return  _mask_SE, _mask_SpE, _mask_EH, _mask_pEH, _mask_EpH, _mask_pEpH, _mask_EM, _mask_pEM, _mask_pEpM, _mask_SM, _mask_SpM, _mask_SH, _mask_SpH, _mask_HM, _mask_HpM, _mask_pHM, _mask_pHpM, _mask_dDAS_H, _mask_dDAS_pH
       
    # ---------------------------------------------------
    # Get additional weather parameters
    # ---------------------------------------------------
    def getWeatherParameters(self, m=None, season=False, verbose=False):
        '''
            Get additional weather parameters per site
            - Avg minimum temperature during growing season
            - Avg maximum temperature during growing season
            - Avg mean temperature during growing season
            - Avg solar radiation during growing season
            - Total amount of precipitation during growing season
        
        :params weather: the weather dataset for the specific site
        
        :return: An updated attributes of the site
        
        '''
        if (self.weather is None):
            print("Weather data is not valid to get additional parameters")
            return
        
        df_SM_idxs, df_SH_idxs, df_HM_idxs = None, None, None
        df_EH_idxs, df_SpE_idxs, df_EpH_idxs, df_pEH_idxs, df_pEM_idxs = None, None, None, None, None
        df_pHM_idxs, df_HpM_idxs, df_pHpM_idxs, df_pEpH_idxs = None, None, None, None
        cGDD_EH, cGDD_EpH, cGDD_pEH, cGDD_pEpH = None, None, None, None
        cGDD_HM, cGDD_pHM, cGDD_HpM, cGDD_pHpM = None, None, None, None
        self.Norm_TT_EH, self.Norm_TT_HM = None, None
        self.Norm_TT_pEH, self.Norm_TT_EpH, self.Norm_TT_pHM, self.Norm_TT_HpM = None, None, None, None
        self.Norm_TT_pEpH, self.Norm_TT_pHpM = None, None
        self.Norm_SimNDVI_EH, self.Norm_SimNDVI_HM, self.Norm_SimNDVI_EpH, = None, None, None
        self.Norm_SimNDVI_pEH, self.Norm_SimNDVI_pHM, self.Norm_SimNDVI_HpM = None, None, None
        self.Norm_SimNDVI_pEpH, self.Norm_SimNDVI_pHpM = None, None
        
        # Get filters or masks
        _mask_SE, _mask_SpE, _mask_EH, _mask_pEH, _mask_EpH, _mask_pEpH, _mask_EM, _mask_pEM, _mask_pEpM, _mask_SM, \
        _mask_SpM, _mask_SH, _mask_SpH, _mask_HM, _mask_HpM, _mask_pHM, _mask_pHpM, \
        _mask_dDAS_H, _mask_dDAS_pH = self.getFilters(m, season=season, verbose=verbose)
        
        try:
            # ---------------------------------------------------------
            # Calculate Cumulative GDD from Sowing to Maturity date
            # --------------------------------------------------------- 
            if (_mask_SM is not None):
                df_SM_idxs = self.weather[_mask_SM].index #.reset_index(drop=True)
                if (df_SM_idxs is not None and len(df_SM_idxs)<=0): 
                    df_SM_idxs = None
                    _mask_SM = None
                else:
                    self.attributes['cGDD_SM'] = np.nanmax(np.cumsum(self.GDD[df_SM_idxs]))
            
            # ---------------------------------------------------------
            # Calculate Cumulative GDD from Sowing to Heading date
            # --------------------------------------------------------- 
            if (_mask_SH is not None):
                df_SH_idxs = self.weather[_mask_SH].index
                if (df_SH_idxs is not None and len(df_SH_idxs)<=0): 
                    df_SH_idxs = None
                    _mask_SH = None
                else:
                    self.attributes['cGDD_SH'] = np.nanmax(np.cumsum(self.GDD[df_SH_idxs]))

            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from Sowing to Emergence date
            # ------------------------------------------------------------------
            if (_mask_SE is not None):
                df_SE_idxs = self.weather[_mask_SE].index
                if (df_SE_idxs is not None and len(df_SE_idxs)<=0): 
                    df_SE_idxs = None
                    _mask_SE = None
                else:
                    self.attributes['cGDD_SE'] = np.nanmax(np.cumsum(self.GDD[df_SE_idxs]))

            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from Sowing to estimated Emergence date
            # ------------------------------------------------------------------
            if (_mask_SpE is not None):
                df_SpE_idxs = self.weather[_mask_SpE].index
                if (df_SpE_idxs is not None and len(df_SpE_idxs)<=0): 
                    df_SpE_idxs = None
                    _mask_SpE = None
                else:
                    self.attributes['cGDD_SpE'] = np.nanmax(np.cumsum(self.GDD[df_SpE_idxs]))

            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from Emergence to Heading date
            # ------------------------------------------------------------------
            if (_mask_EH is not None):
                df_EH_idxs = self.weather[_mask_EH].index
                if (df_EH_idxs is not None and len(df_EH_idxs)<=0): 
                    df_EH_idxs = None
                    _mask_EH = None
                else:
                    self.attributes['cGDD_EH'] = np.nanmax(np.cumsum(self.GDD[df_EH_idxs]))
            
            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from Emergence to estimated Heading date
            # ------------------------------------------------------------------
            if (_mask_EpH is not None):
                df_EpH_idxs = self.weather[_mask_EpH].index
                if (df_EpH_idxs is not None and len(df_EpH_idxs)<=0): 
                    df_EpH_idxs = None
                    _mask_EpH = None
                else:
                    self.attributes['cGDD_EpH'] = np.nanmax(np.cumsum(self.GDD[df_EpH_idxs]))
                    
            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from estimated Emergence to Heading date
            # ------------------------------------------------------------------
            if (_mask_pEH is not None):
                df_pEH_idxs = self.weather[_mask_pEH].index
                if (df_pEH_idxs is not None and len(df_pEH_idxs)<=0): 
                    df_pEH_idxs = None
                    _mask_pEH = None
                else:
                    self.attributes['cGDD_pEH'] = np.nanmax(np.cumsum(self.GDD[df_pEH_idxs]))
            
            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from estimated Emergence to estimated Heading date
            # ------------------------------------------------------------------
            if (_mask_pEpH is not None):
                df_pEpH_idxs = self.weather[_mask_pEpH].index
                if (df_pEpH_idxs is not None and len(df_pEpH_idxs)<=0): 
                    df_pEpH_idxs = None
                    _mask_pEpH = None
                else:
                    self.attributes['cGDD_pEpH'] = np.nanmax(np.cumsum(self.GDD[df_pEpH_idxs]))
                
            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from Emergence to Maturity date
            # ------------------------------------------------------------------
            if (_mask_EM is not None):
                df_EM_idxs = self.weather[_mask_EM].index
                if (df_EM_idxs is not None and len(df_EM_idxs)<=0): 
                    df_EM_idxs = None
                    _mask_EM = None
                else:
                    self.attributes['cGDD_EM'] = np.nanmax(np.cumsum(self.GDD[df_EM_idxs]))

            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from estimated Emergence to Maturity date
            # ------------------------------------------------------------------
            if (_mask_pEM is not None):
                df_pEM_idxs = self.weather[_mask_pEM].index
                if (df_pEM_idxs is not None and len(df_pEM_idxs)<=0): 
                    df_pEM_idxs = None
                    _mask_pEM = None
                else:
                    self.attributes['cGDD_pEM'] = np.nanmax(np.cumsum(self.GDD[df_pEM_idxs]))

            # ---------------------------------------------------------
            # Calculate Cumulative GDD from Heading to Maturity date
            # --------------------------------------------------------- 
            if (_mask_HM is not None):
                df_HM_idxs = self.weather[_mask_HM].index
                if (df_HM_idxs is not None and len(df_HM_idxs)<=0): 
                    df_HM_idxs = None
                    _mask_HM = None
                else:
                    self.attributes['cGDD_HM'] = np.nanmax(np.cumsum(self.GDD[df_HM_idxs]))
            
            # ---------------------------------------------------------
            # Calculate Cumulative GDD from Date@35DAS to Heading date
            # --------------------------------------------------------- 
            if (_mask_dDAS_H is not None):
                df_dDAS_H_idxs = self.weather[_mask_dDAS_H].index
                if (df_dDAS_H_idxs is not None and len(df_dDAS_H_idxs)<=0): 
                    df_dDAS_H_idxs = None
                    _mask_dDAS_H = None
                else:
                    self.attributes['cGDD_{}DAS_H'
                                    .format(m.parameters['DAP'])] = np.nanmax(np.cumsum(self.GDD[df_dDAS_H_idxs]))
            
            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from estimated Heading to Maturity date
            # ------------------------------------------------------------------
            if (_mask_pHM is not None):
                df_pHM_idxs = self.weather[_mask_pHM].index
                if (df_pHM_idxs is not None and len(df_pHM_idxs)<=0): 
                    df_pHM_idxs = None
                    _mask_pHM = None
                else:
                    self.attributes['cGDD_pHM'] = np.nanmax(np.cumsum(self.GDD[df_pHM_idxs]))
                
            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from  Heading to estimated Maturity date
            # ------------------------------------------------------------------
            if (_mask_HpM is not None):
                df_HpM_idxs = self.weather[_mask_HpM].index
                if (df_HpM_idxs is not None and len(df_HpM_idxs)<=0): 
                    df_HpM_idxs = None
                    _mask_HpM = None
                else:
                    self.attributes['cGDD_HpM'] = np.nanmax(np.cumsum(self.GDD[df_HpM_idxs]))
            
            # ---------------------------------------------------------------------
            # Calculate Cumulative GDD from Date@35DAS to estimated Heading date
            # ---------------------------------------------------------------------
            if (_mask_dDAS_pH is not None):
                df_dDAS_pH_idxs = self.weather[_mask_dDAS_pH].index
                if (df_dDAS_pH_idxs is not None and len(df_dDAS_pH_idxs)<=0): 
                    df_dDAS_pH_idxs = None
                    _mask_dDAS_pH = None
                else:
                    self.attributes['cGDD_{}DAS_pH'
                                    .format(m.parameters['DAP'])] = np.nanmax(np.cumsum(self.GDD[df_dDAS_pH_idxs]))
            
            # ------------------------------------------------------------------
            # Calculate Cumulative GDD from estimated Heading to estimated Maturity date
            # ------------------------------------------------------------------
            if (_mask_pHpM is not None):
                df_pHpM_idxs = self.weather[_mask_pHpM].index
                if (df_pHpM_idxs is not None and len(df_pHpM_idxs)<=0): 
                    df_pHpM_idxs = None
                    _mask_pHpM = None
                else:
                    self.attributes['cGDD_pHpM'] = np.nanmax(np.cumsum(self.GDD[df_pHpM_idxs]))
            
            # ------------------------------------------------------------------------------------
            # Normalize GDD or Thermal time
            # ------------------------------------------------------------------------------------
            # Thermal time from Emergence to Heading
            if (df_EH_idxs is not None):
                cGDD_EH = np.cumsum(self.GDD[df_EH_idxs])
            else:
                cGDD_EH = None
            if (df_EpH_idxs is not None):
                cGDD_EpH = np.cumsum(self.GDD[df_EpH_idxs])
            else:
                cGDD_EpH = None
            if (df_pEH_idxs is not None):
                cGDD_pEH = np.cumsum(self.GDD[df_pEH_idxs])
            else:
                cGDD_pEH = None
            if (df_pEpH_idxs is not None):
                cGDD_pEpH = np.cumsum(self.GDD[df_pEpH_idxs])
            else:
                cGDD_pEpH = None
            # Thermal time from Heading to Maturity
            if (df_HM_idxs is not None):
                cGDD_HM = np.cumsum(self.GDD[df_HM_idxs])
            else:
                cGDD_HM = None
            if (df_pHM_idxs is not None):
                cGDD_pHM = np.cumsum(self.GDD[df_pHM_idxs])
            else:
                cGDD_pHM = None
            if (df_HpM_idxs is not None):
                cGDD_HpM = np.cumsum(self.GDD[df_HpM_idxs])
            else:
                cGDD_HpM = None
            if (df_pHpM_idxs is not None):
                cGDD_pHpM = np.cumsum(self.GDD[df_pHpM_idxs])
            else:
                cGDD_pHpM = None
            
            
            self.getNormalizeThermalTime(m, cGDD_EH, cGDD_pEH, cGDD_EpH, cGDD_pEpH, cGDD_HM, cGDD_pHM, cGDD_pHpM, cGDD_HpM, 
                                         verbose=verbose)
            
            # ---------------------------------
            # Params to calculate iPAR
            # ---------------------------------
            self.setupIPAR(m, _mask_EH, _mask_pEH, _mask_EpH, _mask_pEpH, _mask_HM, _mask_pHM, _mask_HpM, _mask_pHpM, verbose=verbose)
            
        except Exception as err:
            print(HT.bold + HT.fg.red + "Error calculating additional weather parameters"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, 
                                "error": "Calculating additional weather parameters. Error: {}".format(err)})
        
        del df_SM_idxs, df_SH_idxs, df_HM_idxs 
        del df_EH_idxs, df_SpE_idxs, df_EpH_idxs, df_pEH_idxs, df_pEM_idxs 
        del df_pHM_idxs, df_HpM_idxs, df_pHpM_idxs, df_pEpH_idxs
        _ = gc.collect()
        
    
    # ---------------------------------
    # Params to calculate iPAR
    # ---------------------------------
    def setupIPAR(self, m, _mask_EH=None, _mask_pEH=None, _mask_EpH=None, _mask_pEpH=None, 
                  _mask_HM=None, _mask_pHM=None, _mask_HpM=None, _mask_pHpM=None, verbose=False):
        '''
            Setup parameters to calculate iPAR
        '''
        if (verbose is True):
            print("Calculating iPAR...")
        # Calculate day time temperature - TDay
        if (_mask_EH is not None):
            TDay_EH = self.getTDay(m, self.weather[_mask_EH]) 
            self.TDay_EH = TDay_EH
            # Estimate Photosynthesis reduction factor - PRFT
            if (TDay_EH is not None):
                self.PRFT_EH = self.getPRFT(m, TDay_EH)
        else:
            self.TDay_EH = None
            self.PRFT_EH = None

        if (_mask_pEH is not None):
            TDay_pEH = self.getTDay(m, self.weather[_mask_pEH]) 
            self.TDay_pEH = TDay_pEH
            # Estimate Photosynthesis reduction factor - PRFT
            if (TDay_pEH is not None):
                self.PRFT_pEH = self.getPRFT(m, TDay_pEH)
        else:
            self.TDay_pEH = None
            self.PRFT_pEH = None
        
        if (_mask_EpH is not None):
            TDay_EpH = self.getTDay(m, self.weather[_mask_EpH]) 
            self.TDay_EpH = TDay_EpH
            if (TDay_EpH is not None):
                self.PRFT_EpH = self.getPRFT(m, TDay_EpH)
        else:
            self.TDay_EpH = None
            self.PRFT_EpH = None
            
        if (_mask_pEpH is not None):
            TDay_pEpH = self.getTDay(m, self.weather[_mask_pEpH]) 
            self.TDay_pEpH = TDay_pEpH
            # Estimate Photosynthesis reduction factor - PRFT
            if (TDay_pEpH is not None):
                self.PRFT_pEpH = self.getPRFT(m, TDay_pEpH)
        else:
            self.TDay_pEpH = None
            self.PRFT_pEpH = None

        if (_mask_HM is not None):
            TDay_HM = self.getTDay(m, self.weather[_mask_HM]) 
            self.TDay_HM = TDay_HM
            # Estimate Photosynthesis reduction factor - PRFT
            if (TDay_HM is not None):
                self.PRFT_HM = self.getPRFT(m, TDay_HM)
        else:
            self.TDay_HM = None
            self.PRFT_HM = None

        if (_mask_pHM is not None):
            TDay_pHM = self.getTDay(m, self.weather[_mask_pHM]) 
            self.TDay_pHM = TDay_pHM
            # Estimate Photosynthesis reduction factor - PRFT
            if (TDay_pHM is not None):
                self.PRFT_pHM = self.getPRFT(m, TDay_pHM)
        else:
            self.TDay_pHM = None
            self.PRFT_pHM = None
            
        if (_mask_HpM is not None):
            TDay_HpM = self.getTDay(m, self.weather[_mask_HpM]) 
            self.TDay_HpM = TDay_HpM
            # Estimate Photosynthesis reduction factor - PRFT
            if (TDay_HpM is not None):
                self.PRFT_HpM = self.getPRFT(m, TDay_HpM)
        else:
            self.TDay_HpM = None
            self.PRFT_HpM = None
        
        if (_mask_pHpM is not None):
            TDay_pHpM = self.getTDay(m, self.weather[_mask_pHpM]) 
            self.TDay_pHpM = TDay_pHpM
            # Estimate Photosynthesis reduction factor - PRFT
            if (TDay_pHpM is not None):
                self.PRFT_pHpM = self.getPRFT(m, TDay_pHpM)
        else:
            self.TDay_pHpM = None
            self.PRFT_pHpM = None

        # Solar radiation
        if (_mask_EH is not None):
            self.SolRad_EH = self.weather[_mask_EH]['SolRad']
        else:
            self.SolRad_EH = None
        
        if (_mask_EpH is not None):
            self.SolRad_EpH = self.weather[_mask_EpH]['SolRad']
        else:
            self.SolRad_EpH = None
            
        if (_mask_pEH is not None):
            self.SolRad_pEH = self.weather[_mask_pEH]['SolRad']
        else:
            self.SolRad_pEH = None
            
        if (_mask_pEpH is not None):
            self.SolRad_pEpH = self.weather[_mask_pEpH]['SolRad']
        else:
            self.SolRad_pEpH = None

        if (_mask_HM is not None):
            self.SolRad_HM = self.weather[_mask_HM]['SolRad']
        else:
            self.SolRad_HM = None

        if (_mask_pHM is not None):
            self.SolRad_pHM = self.weather[_mask_pHM]['SolRad']
        else:
            self.SolRad_pHM = None
            
        if (_mask_HpM is not None):
            self.SolRad_HpM = self.weather[_mask_HpM]['SolRad']
        else:
            self.SolRad_HpM = None
            
        if (_mask_pHpM is not None):
            self.SolRad_pHpM = self.weather[_mask_pHpM]['SolRad']
        else:
            self.SolRad_pHpM = None
    
    # ------------------------------------------------------------------------------------
    # Normalize GDD or Thermal time
    # ------------------------------------------------------------------------------------
    def getNormalizeThermalTime(self, m=None, cGDD_EH=None, cGDD_pEH=None, cGDD_EpH=None, cGDD_pEpH=None, 
                                cGDD_HM=None, cGDD_pHM=None, cGDD_pHpM=None, cGDD_HpM=None,
                                verbose=False):
        '''
            Normalize GDD or Thermal time
            
        '''
        if (verbose is True):
            print("\nNormalizing GDD or Thermal time..." )
        try:
            if (cGDD_EH is not None):
                self.Norm_TT_EH = ( cGDD_EH - np.nanmin(cGDD_EH) ) / ( np.nanmax(cGDD_EH) - np.nanmin(cGDD_EH) )
            else:
                self.Norm_TT_EH = None
            if (cGDD_EpH is not None):
                self.Norm_TT_EpH = ( cGDD_EpH - np.nanmin(cGDD_EpH) ) / ( np.nanmax(cGDD_EpH) - np.nanmin(cGDD_EpH) )
            else:
                self.Norm_TT_EpH = None
            if (cGDD_pEH is not None):
                self.Norm_TT_pEH = ( cGDD_pEH - np.nanmin(cGDD_pEH) ) / ( np.nanmax(cGDD_pEH) - np.nanmin(cGDD_pEH) )
            else:
                self.Norm_TT_pEH = None
            if (cGDD_pEpH is not None):
                self.Norm_TT_pEpH = ( cGDD_pEpH - np.nanmin(cGDD_pEpH) ) / ( np.nanmax(cGDD_pEpH) - np.nanmin(cGDD_pEpH) )
            else:
                self.Norm_TT_pEpH = None

        except Exception as err:
            print("Problem normalizing GDD or thermal time from Emergence to Heading for plot {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Normalizing GDD or thermal time from Emergence to Heading. Error: {}".format(err)})

        try:
            # Thermal time from Heading to Maturity
            if (cGDD_HM is not None):
                self.Norm_TT_HM = ( cGDD_HM - np.nanmin(cGDD_HM) ) / ( np.nanmax(cGDD_HM) - np.nanmin(cGDD_HM) )
            else:
                self.Norm_TT_HM = None
            if (cGDD_pHM is not None):
                self.Norm_TT_pHM = ( cGDD_pHM - np.nanmin(cGDD_pHM) ) / ( np.nanmax(cGDD_pHM) - np.nanmin(cGDD_pHM) )
            else:
                self.Norm_TT_pHM = None
            if (cGDD_HpM is not None):
                self.Norm_TT_HpM = ( cGDD_HpM - np.nanmin(cGDD_HpM) ) / ( np.nanmax(cGDD_HpM) - np.nanmin(cGDD_HpM) )
            else:
                self.Norm_TT_HpM = None
            if (cGDD_pHpM is not None):
                self.Norm_TT_pHpM = ( cGDD_pHpM - np.nanmin(cGDD_pHpM) ) / ( np.nanmax(cGDD_pHpM) - np.nanmin(cGDD_pHpM) )
            else:
                self.Norm_TT_pHpM = None
                
        except Exception as err:
            print("Problem normalizing GDD or thermal time from Heading to Maturity for plot {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, "error": "Normalizing GDD or thermal time from Heading to Maturity. Error: {}".format(err)})
    
    
    
    # ------------------------------------------------------------------------------------
    # Update Stats from weather data
    # ------------------------------------------------------------------------------------
    def updateWeatherparams(self, _mask_SE=None, _mask_SpE=None, _mask_EH=None, _mask_pEH=None, _mask_EpH=None, 
                            _mask_pEpH=None, _mask_EM=None, _mask_pEM=None,_mask_pEpM=None, _mask_SM=None, 
                            _mask_SpM=None, _mask_SH=None, _mask_SpH=None, _mask_HM=None, _mask_HpM=None, 
                            _mask_pHM=None, _mask_pHpM=None, _mask_dDAS_H=None, _mask_dDAS_pH=None,
                            season=False, verbose=False):
        '''
            Update weather statistics 
            
            :params _mask_SM: Filter used to get weather data from Sowing to Maturity
            :params _mask_SH: Filter used to get weather data from Sowing to Heading
            :params _mask_EH: Filter used to get weather data from Emergence to Heading
            :params _mask_pEH: Filter used to get weather data from predicted Emergence to Heading
            :params _mask_HM: Filter used to get weather data from Heading to Maturity
            :params _mask_SpM: Filter used to get weather data from Sowing to estimated Maturity
            :params _mask_SpH: Filter used to get weather data from Sowing to estimated Heading
            :params _mask_pEpH: Filter used to get weather data from predicted Emergence to estimated Heading
            :params _mask_HpM: Filter used to get weather data from Heading to estimated Maturity
            :params _mask_pHpM: Filter used to get weather data from estimated Heading to estimated Maturity
            :params _mask_dDAS_H: Filter used to get weather data from 35 days after Sowing to Heading
            :params _mask_dDAS_pH: Filter used to get weather data from 35 days after Sowing to estimated Heading
            
            :params season: Display weather statistics for different periods
            
            :result: A Site with updated weather statistics
        '''
        df = None
        if (_mask_SM is not None):
            df = self.weather[_mask_SM]
            self.attributes['Tmin'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
            self.attributes['Tmax'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
            self.attributes['Tavg'] = float("{:.1f}".format(np.nanmean(df['TAVG'])))
            self.attributes['Srad'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
            self.attributes['Pcp'] = float("{:.0f}".format(np.sum(df['PCP'])))
            if (verbose is True):
                print(HT.bold + "\nWeather statistics for growing season"+ HT.reset)
                print("Site Avg. minimum temperature: {:.1f}°C".format(self.attributes['Tmin']))
                print("Site Avg. maximum temperature: {:.1f}°C".format(self.attributes['Tmax']))
                print("Site Avg. mean temperature: {:.1f}°C".format(self.attributes['Tavg']))
                print("Site Avg. solar radiation: {:.1f} MJ/m2/d".format(self.attributes['Srad']))
                print("Site Total amount of precipitation: {:.0f} mm".format(self.attributes['Pcp']))

        if (season is True):
            # Weather from Sowing to Heading
            if (_mask_SH is not None):
                df = self.weather[_mask_SH]
                self.attributes['Tmin_SH'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_SH'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_SH'] = float("{:.1f}".format(np.nanmean(df['TAVG']) ))
                self.attributes['Srad_SH'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_SH'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for sowing to heading period"+ HT.reset)
                    print("Site Avg. minimum temperature for SH period: {:.1f}°C"
                          .format(self.attributes['Tmin_SH']))
                    print("Site Avg. maximum temperature for SH period: {:.1f}°C"
                          .format(self.attributes['Tmax_SH']))
                    print("Site Avg. mean temperature for SH period: {:.1f}°C"
                          .format(self.attributes['Tavg_SH']))
                    print("Site Avg. solar radiation for SH period: {:.1f} MJ/m2/d"
                          .format(self.attributes['Srad_SH']))
                    print("Site Total amount of precipitation for SH period: {:.0f} mm"
                          .format(self.attributes['Pcp_SH']))

            # Weather from Emergence to Heading
            if (_mask_EH is not None):
                df = self.weather[_mask_EH]
                self.attributes['Tmin_EH'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_EH'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_EH'] = float("{:.1f}".format(np.nanmean(df['TAVG']) ))
                self.attributes['Srad_EH'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_EH'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for emergence to heading period" + HT.reset)
                    print("Site Avg. minimum temperature for EH period: {:.1f}°C"
                          .format(self.attributes['Tmin_EH']))
                    print("Site Avg. maximum temperature for EH period: {:.1f}°C"
                          .format(self.attributes['Tmax_EH']))
                    print("Site Avg. mean temperature for EH period: {:.1f}°C"
                          .format(self.attributes['Tavg_EH']))
                    print("Site Avg. solar radiation for EH period: {:.1f} MJ/m2/d"
                          .format(self.attributes['Srad_EH']))
                    print("Site Total amount of precipitation for EH period: {:.0f} mm"
                          .format(self.attributes['Pcp_EH']))
            
            # Weather from Emergence to estimated Heading
            if (_mask_EpH is not None):
                df = self.weather[_mask_EpH]
                self.attributes['Tmin_EpH'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_EpH'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_EpH'] = float("{:.1f}".format(np.nanmean(df['TAVG']) ))
                self.attributes['Srad_EpH'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_EpH'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for emergence to estimated heading period" + HT.reset)
                    print("Site Avg. minimum temperature for EpH period: {:.1f}°C"
                          .format(self.attributes['Tmin_EpH']))
                    print("Site Avg. maximum temperature for EpH period: {:.1f}°C"
                          .format(self.attributes['Tmax_EpH']))
                    print("Site Avg. mean temperature for EpH period: {:.1f}°C"
                          .format(self.attributes['Tavg_EpH']))
                    print("Site Avg. solar radiation for EpH period: {:.1f} MJ/m2/d"
                          .format(self.attributes['Srad_EpH']))
                    print("Site Total amount of precipitation for EH period: {:.0f} mm"
                          .format(self.attributes['Pcp_EpH']))
                    
            # Weather from estimated Emergence to Heading
            if (_mask_pEH is not None):
                df = self.weather[_mask_pEH]
                self.attributes['Tmin_pEH'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_pEH'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_pEH'] = float("{:.1f}".format(np.nanmean(df['TAVG']) ))
                self.attributes['Srad_pEH'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_pEH'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for estimated emergence to heading period"+ HT.reset)
                    print("Site Avg. minimum temperature for pEH period: {:.1f}°C".format(self.attributes['Tmin_pEH']))
                    print("Site Avg. maximum temperature for pEH period: {:.1f}°C".format(self.attributes['Tmax_pEH']))
                    print("Site Avg. mean temperature for pEH period: {:.1f}°C".format(self.attributes['Tavg_pEH']))
                    print("Site Avg. solar radiation for pEH period: {:.1f} MJ/m2/d".format(self.attributes['Srad_pEH']))
                    print("Site Total amount of precipitation for pEH period: {:.0f} mm".format(self.attributes['Pcp_pEH']))

            # Weather from Heading to Maturity
            if (_mask_HM is not None):
                df = self.weather[_mask_HM]
                self.attributes['Tmin_HM'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_HM'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_HM'] = float("{:.1f}".format(np.nanmean(df['TAVG'])))
                self.attributes['Srad_HM'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_HM'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for heading to maturity period"+ HT.reset)
                    print("Site Avg. minimum temperature for HM period: {:.1f}°C"
                          .format(self.attributes['Tmin_HM']))
                    print("Site Avg. maximum temperature for HM period: {:.1f}°C"
                          .format(self.attributes['Tmax_HM']))
                    print("Site Avg. mean temperature for HM period: {:.1f}°C"
                          .format(self.attributes['Tavg_HM']))
                    print("Site Avg. solar radiation for HM period: {:.1f} MJ/m2/d"
                          .format(self.attributes['Srad_HM']))
                    print("Site Total amount of precipitation for HM period: {:.0f} mm"
                          .format(self.attributes['Pcp_HM']))
            
            if (_mask_SpM is not None):
                df = self.weather[_mask_SpM]
                self.attributes['Tmin_SpM'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_SpM'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_SpM'] = float("{:.1f}".format(np.nanmean(df['TAVG'])))
                self.attributes['Srad_SpM'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_SpM'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for sowing to predicted maturity period"+ HT.reset)
                    print("Site Avg. minimum temperature for SpM period: {:.1f}°C"
                          .format(self.attributes['Tmin_SpM']))
                    print("Site Avg. maximum temperature for SpM period: {:.1f}°C"
                          .format(self.attributes['Tmax_SpM']))
                    print("Site Avg. mean temperature for SpM period: {:.1f}°C"
                          .format(self.attributes['Tavg_SpM']))
                    print("Site Avg. solar radiation for SpM period: {:.1f} MJ/m2/d"
                          .format(self.attributes['Srad_SpM']))
                    print("Site Total amount of precipitation for SpM period: {:.0f} mm"
                          .format(self.attributes['Pcp_SpM']))
                
            if (_mask_SpH is not None):
                df = self.weather[_mask_SpH]
                self.attributes['Tmin_SpH'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_SpH'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_SpH'] = float("{:.1f}".format(np.nanmean(df['TAVG'])))
                self.attributes['Srad_SpH'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_SpH'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for sowing to estimated heading period"+ HT.reset)
                    print("Site Avg. minimum temperature for SpH period: {:.1f}°C"
                          .format(self.attributes['Tmin_SpH']))
                    print("Site Avg. maximum temperature for SpH period: {:.1f}°C"
                          .format(self.attributes['Tmax_SpH']))
                    print("Site Avg. mean temperature for SpH period: {:.1f}°C"
                          .format(self.attributes['Tavg_SpH']))
                    print("Site Avg. solar radiation for SpH period: {:.1f} MJ/m2/d"
                          .format(self.attributes['Srad_SpH']))
                    print("Site Total amount of precipitation for SpH period: {:.0f} mm"
                          .format(self.attributes['Pcp_SpH']))
                
            if (_mask_pEpH is not None):
                df = self.weather[_mask_pEpH]
                self.attributes['Tmin_pEpH'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_pEpH'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_pEpH'] = float("{:.1f}".format(np.nanmean(df['TAVG'])))
                self.attributes['Srad_pEpH'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_pEpH'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for estimated emergence to estimated heading period"+ HT.reset)
                    print("Site Avg. minimum temperature for pEpH period: {:.1f}°C"
                          .format(self.attributes['Tmin_pEpH']))
                    print("Site Avg. maximum temperature for pEpH period: {:.1f}°C"
                          .format(self.attributes['Tmax_pEpH']))
                    print("Site Avg. mean temperature for pEpH period: {:.1f}°C"
                          .format(self.attributes['Tavg_pEpH']))
                    print("Site Avg. solar radiation for pEpH period: {:.1f} MJ/m2/d"
                          .format(self.attributes['Srad_pEpH']))
                    print("Site Total amount of precipitation for pEpH period: {:.0f} mm"
                          .format(self.attributes['Pcp_pEpH']))
            
            if (_mask_HpM is not None):
                df = self.weather[_mask_HpM]
                self.attributes['Tmin_HpM'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_HpM'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_HpM'] = float("{:.1f}".format(np.nanmean(df['TAVG'])))
                self.attributes['Srad_HpM'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_HpM'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for heading to predicted maturity period"+ HT.reset)
                    print("Site Avg. minimum temperature for pHpM period: {:.1f}°C"
                          .format(self.attributes['Tmin_HpM']))
                    print("Site Avg. maximum temperature for pHpM period: {:.1f}°C"
                          .format(self.attributes['Tmax_HpM']))
                    print("Site Avg. mean temperature for pHpM period: {:.1f}°C"
                          .format(self.attributes['Tavg_HpM']))
                    print("Site Avg. solar radiation for pHpM period: {:.1f} MJ/m2/d"
                          .format(self.attributes['Srad_HpM']))
                    print("Site Total amount of precipitation for pHpM period: {:.0f} mm"
                          .format(self.attributes['Pcp_HpM']))
                    
            if (_mask_pHpM is not None):
                df = self.weather[_mask_pHpM]
                self.attributes['Tmin_pHpM'] = float("{:.1f}".format(np.nanmean(df['TMIN'])))
                self.attributes['Tmax_pHpM'] = float("{:.1f}".format(np.nanmean(df['TMAX'])))
                self.attributes['Tavg_pHpM'] = float("{:.1f}".format(np.nanmean(df['TAVG'])))
                self.attributes['Srad_pHpM'] = float("{:.1f}".format(np.nanmean(df['SolRad'])))
                self.attributes['Pcp_pHpM'] = float("{:.0f}".format(np.sum(df['PCP'])))
                if (verbose is True):
                    print(HT.bold + "\nWeather statistics for estimated heading to estimated maturity period"+ HT.reset)
                    print("Site Avg. minimum temperature for pHpM period: {:.1f}°C"
                          .format(self.attributes['Tmin_pHpM']))
                    print("Site Avg. maximum temperature for pHpM period: {:.1f}°C"
                          .format(self.attributes['Tmax_pHpM']))
                    print("Site Avg. mean temperature for pHpM period: {:.1f}°C"
                          .format(self.attributes['Tavg_pHpM']))
                    print("Site Avg. solar radiation for pHpM period: {:.1f} MJ/m2/d"
                          .format(self.attributes['Srad_pHpM']))
                    print("Site Total amount of precipitation for pHpM period: {:.0f} mm"
                          .format(self.attributes['Pcp_pHpM']))
                

        del df
        _ = gc.collect()
        
    
    # ---------------------------------------------------
    # The normalized difference vegetation index (NDVI) 
    # ---------------------------------------------------
    def estimateNDVI(self, m=None, verbose=False):
        '''
        Estimate NDVI values for growing cycle.
        
        Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
        from Emergence to Heading

        :params Norm_TT_EH: Normalize GDD or Thermal time from Emergence to Heading

        :return: An array with NDVI values from Emergence to Heading

        '''
        if (verbose is True):
            print("Estimating the normalized difference vegetation index (NDVI)")
        try:
            if (self.Norm_TT_EH is not None):
                Norm_SimNDVI_EH = self.estimateNDVI_EH(m, Norm_TT_EH=self.Norm_TT_EH, verbose=verbose)
                if (Norm_SimNDVI_EH is not None):
                    self.attributes['NDVI_atHeading_EH'] = Norm_SimNDVI_EH[-1]
                    self.Norm_SimNDVI_EH = Norm_SimNDVI_EH
            
            if (self.Norm_TT_pEH is not None):
                Norm_SimNDVI_pEH = self.estimateNDVI_EH(m, Norm_TT_EH=self.Norm_TT_pEH, verbose=verbose)
                if (Norm_SimNDVI_pEH is not None):
                    self.attributes['NDVI_atHeading_pEH'] = Norm_SimNDVI_pEH[-1]
                    self.Norm_SimNDVI_pEH = Norm_SimNDVI_pEH
                    
            if (self.Norm_TT_EpH is not None):
                Norm_SimNDVI_EpH = self.estimateNDVI_EH(m, Norm_TT_EH=self.Norm_TT_EpH, verbose=verbose)
                if (Norm_SimNDVI_EpH is not None):
                    self.attributes['NDVI_atHeading_EpH'] = Norm_SimNDVI_EpH[-1]
                    self.Norm_SimNDVI_EpH = Norm_SimNDVI_EpH
                    
            if (self.Norm_TT_pEpH is not None):
                Norm_SimNDVI_pEpH = self.estimateNDVI_EH(m, Norm_TT_EH=self.Norm_TT_pEpH, verbose=verbose)
                if (Norm_SimNDVI_pEpH is not None):
                    self.attributes['NDVI_atHeading_pEpH'] = Norm_SimNDVI_pEpH[-1]
                    self.Norm_SimNDVI_pEpH = Norm_SimNDVI_pEpH
            
            
            # 
            if ((self.Norm_TT_HM is not None) and ('NDVI_atHeading_EH' in self.attributes.keys())):
                Norm_SimNDVI_HM = self.estimateNDVI_HM(m, Norm_TT_HM=self.Norm_TT_HM, 
                                                       NDVI_atHeading=self.attributes['NDVI_atHeading_EH'], verbose=verbose)
                if (Norm_SimNDVI_HM is not None):
                    self.Norm_SimNDVI_HM = Norm_SimNDVI_HM
            elif ((self.Norm_TT_HM is not None) and ('NDVI_atHeading_pEH' in self.attributes.keys())):
                Norm_SimNDVI_HM = self.estimateNDVI_HM(m, Norm_TT_HM=self.Norm_TT_HM, 
                                                       NDVI_atHeading=self.attributes['NDVI_atHeading_pEH'],
                                                       verbose=verbose)
                if (Norm_SimNDVI_HM is not None):
                    self.Norm_SimNDVI_HM = Norm_SimNDVI_HM
                
            if ((self.Norm_TT_pHM is not None) and ('NDVI_atHeading_EpH' in self.attributes.keys())):
                Norm_SimNDVI_pHM = self.estimateNDVI_HM(m, Norm_TT_HM=self.Norm_TT_pHM, 
                                                        NDVI_atHeading=self.attributes['NDVI_atHeading_EpH'], 
                                                        verbose=verbose)
                if (Norm_SimNDVI_pHM is not None):
                    self.Norm_SimNDVI_pHM = Norm_SimNDVI_pHM
                    
            elif ((self.Norm_TT_pHM is not None) and ('NDVI_atHeading_pEpH' in self.attributes.keys())):
                Norm_SimNDVI_pHM = self.estimateNDVI_HM(m, Norm_TT_HM=self.Norm_TT_pHM, 
                                                        NDVI_atHeading=self.attributes['NDVI_atHeading_pEpH'], 
                                                        verbose=verbose)
                if (Norm_SimNDVI_pHM is not None):
                    self.Norm_SimNDVI_pHM = Norm_SimNDVI_pHM
            
            if ((self.Norm_TT_HpM is not None) and ('NDVI_atHeading_EH' in self.attributes.keys())):
                Norm_SimNDVI_HpM = self.estimateNDVI_HM(m, Norm_TT_HM=self.Norm_TT_HpM, 
                                                        NDVI_atHeading=self.attributes['NDVI_atHeading_EH'], 
                                                        verbose=verbose)
                if (Norm_SimNDVI_HpM is not None):
                    self.Norm_SimNDVI_HpM = Norm_SimNDVI_HpM
            elif ((self.Norm_TT_HpM is not None) and ('NDVI_atHeading_pEH' in self.attributes.keys())):
                Norm_SimNDVI_HpM = self.estimateNDVI_HM(m, Norm_TT_HM=self.Norm_TT_HpM, 
                                                        NDVI_atHeading=self.attributes['NDVI_atHeading_pEH'], 
                                                        verbose=verbose)
                if (Norm_SimNDVI_HpM is not None):
                    self.Norm_SimNDVI_HpM = Norm_SimNDVI_HpM
            
            if ((self.Norm_TT_pHpM is not None) and ('NDVI_atHeading_EpH' in self.attributes.keys())):
                Norm_SimNDVI_pHpM = self.estimateNDVI_HM(m, Norm_TT_HM=self.Norm_TT_pHpM, 
                                                         NDVI_atHeading=self.attributes['NDVI_atHeading_EpH'], 
                                                         verbose=verbose)
                if (Norm_SimNDVI_pHpM is not None):
                    self.Norm_SimNDVI_pHpM = Norm_SimNDVI_pHpM
            elif ((self.Norm_TT_pHpM is not None) and ('NDVI_atHeading_pEpH' in self.attributes.keys())):
                Norm_SimNDVI_pHpM = self.estimateNDVI_HM(m, Norm_TT_HM=self.Norm_TT_pHpM, 
                                                         NDVI_atHeading=self.attributes['NDVI_atHeading_pEpH'], 
                                                         verbose=verbose)
                if (Norm_SimNDVI_pHpM is not None):
                    self.Norm_SimNDVI_pHpM = Norm_SimNDVI_pHpM

        except Exception as err:
            print(HT.bold + HT.fg.red + "Estimating the normalized difference vegetation index (NDVI)"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, 
                                "error": "Estimating the normalized difference vegetation index (NDVI). Error: {}".format(err)})
        
    
    def estimateNDVI_EH(self, m=None, Norm_TT_EH=None, verbose=False):
        '''
        Estimate NDVI values from emergence to heading.
        
        Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
        from Emergence to Heading

        :params Norm_TT_EH: Normalize GDD or Thermal time from Emergence to Heading

        :return: An array with NDVI values from Emergence to Heading

        '''
        if (Norm_TT_EH is None):
            print(HT.bold + "Normalize Thermal time from Emergence to Heading is not valid"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, "Norm_TT_EH"))
            return
        
        if (m is not None):
            NDVI_lowerThreshold = m.parameters["NDVI_lowerThreshold"]
            NDVI_Threshold = m.parameters["NDVI_Threshold"]
            NDVI_max = m.parameters["NDVI_max"]
        else:
            NDVI_lowerThreshold = model.PARAMETERS["NDVI_lowerThreshold"]
            NDVI_Threshold = model.PARAMETERS["NDVI_Threshold"]
            NDVI_max = model.PARAMETERS["NDVI_max"]
        
        #Norm_SimNDVI = ndvi.estimateNDVI_EH(norm_TT_EH=self.Norm_TT_EH, NDVI_lowerThreshold=NDVI_lowerThreshold, 
        #                                   NDVI_Threshold=NDVI_Threshold, NDVI_max=NDVI_max, verbose=verbose)
        
        # Numba
        Norm_SimNDVI = ndvi.calculateNDVI_EH(norm_TT_EH=Norm_TT_EH, NDVI_lowerThreshold=NDVI_lowerThreshold, 
                                           NDVI_Threshold=NDVI_Threshold, NDVI_max=NDVI_max)
        
        return Norm_SimNDVI
    
    def estimateNDVI_HM(self, m=None, Norm_TT_HM=None, NDVI_atHeading=None, verbose=False):
        '''
        Estimate NDVI values from Heading to Maturity.

        Use of Normalized Thermal Time and Observed NDVI to calculate IPAR 
        from Heading to Maturity.

        :params norm_TT_HM: Normalize GDD or Thermal time from Heading to Maturity
        :params NDVI_max: Maximum NDVI value allowed
        :params NDVI_atHeading: NDVI reached at Heading date
        :verbose: Display messages during the processing

        :return: An array with NDVI values from Heading to Maturity

        '''
        if (Norm_TT_HM is None):
            print(HT.bold + "Normalize Thermal time from Heading to Maturity is not valid"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, "Norm_TT_HM"))
            return
        
        #if ('NDVI_atHeading' in self.attributes.keys()):
        #    NDVI_atHeading = self.attributes['NDVI_atHeading']
        #else:
        #    NDVI_atHeading = 0.98
        
        #Norm_SimNDVI = ndvi.estimateNDVI_HM(norm_TT_HM=self.Norm_TT_HM, NDVImax=NDVI_atHeading, 
        #                                    NDVI_atHeading=NDVI_atHeading, verbose=verbose)
        
        # Numba
        Norm_SimNDVI = ndvi.calculateNDVI_HM(norm_TT_HM=Norm_TT_HM, NDVImax=NDVI_atHeading, verbose=verbose)
        
        return Norm_SimNDVI
    
        
    # --------------------------------------------------------------
    # Processing iPAR - Total light interception
    # --------------------------------------------------------------
    def getIPAR(self, m=None, norm_iPAR_EH_bounds=None, NDVI_constantIPAR=None, 
                RUE=None, YIELD_FACTOR=None, verbose=False):
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
            :params m: Model with configuration and parameters
            :params norm_iPAR_EH_bounds: Bounds for iPAR multi-linear equations
            
            :return: An array of Total light interception values

        '''
        
        if (self.Norm_SimNDVI_EH is None):
            print("Normalized NDVI from observed Emergence to Heading is not valid")
        if (self.Norm_SimNDVI_pEH is None):
            print("Normalized NDVI from estimated Emergence to Heading is not valid")
        if (self.Norm_SimNDVI_EpH is None):
            print("Normalized NDVI from Emergence to estimated Heading is not valid")
        if (self.Norm_SimNDVI_pEpH is None):
            print("Normalized NDVI from estimated Emergence to estimated Heading is not valid")
        if ((self.Norm_SimNDVI_EH is None) and (self.Norm_SimNDVI_EpH is None) and (self.Norm_SimNDVI_pEH is None) 
            and (self.Norm_SimNDVI_pEpH is None) ):
            print("Normalized NDVI from Emergence to Heading is not available")
            return
        
        if (self.Norm_SimNDVI_HM is None):
            print("Normalized NDVI from observed Heading to Maturity is not valid")
        if (self.Norm_SimNDVI_pHM is None):
            print("Normalized NDVI from estimated Heading to Maturity is not valid")
        if (self.Norm_SimNDVI_HpM is None):
            print("Normalized NDVI from observed Heading to estimated Maturity is not valid")
        if (self.Norm_SimNDVI_pHpM is None):
            print("Normalized NDVI from estimated Heading to estimated Maturity is not valid")
        
        #if ((self.Norm_SimNDVI_HM is None) and (self.Norm_SimNDVI_pHM is None) 
        #    and (self.Norm_SimNDVI_pHpM is None) and ((self.Norm_SimNDVI_HpM is None)) ):
        #    print("Normalized NDVI from Heading to Maturity is not available")
        #    return
        
        if (self.Norm_TT_EH is None):
            print("Normalized Thermal time from observed Emergence to Heading is not valid")
        if (self.Norm_TT_EpH is None):
            print("Normalized Thermal time from observed Emergence to estimated Heading is not valid")
        if (self.Norm_TT_pEH is None):
            print("Normalized Thermal time from estimated Emergence to Heading is not valid")
        if (self.Norm_TT_pEpH is None):
            print("Normalized Thermal time from estimated Emergence to estimated Heading is not valid")
        #if ((self.Norm_TT_EH is None) and (self.Norm_TT_EpH is None) and (self.Norm_TT_pEH is None) 
        #    and (self.Norm_TT_pEpH is None) ):
        #    print("Normalized Thermal time is not available")
        #    return
        
        if (norm_iPAR_EH_bounds is None and m is not None):
            norm_iPAR_EH_bounds = m.parameters["NORM_iPAR_EH_BOUNDS"]
        elif (norm_iPAR_EH_bounds is None):
            norm_iPAR_EH_bounds = model.PARAMETERS["NORM_iPAR_EH_BOUNDS"]
        if (NDVI_constantIPAR is None and m is not None):
            NDVI_constantIPAR = m.parameters["NDVI_constantIPAR"]
        elif (NDVI_constantIPAR is None):
            NDVI_constantIPAR = model.PARAMETERS["NDVI_constantIPAR"]
        
        if (RUE is None and m is not None):
            RUE = m.parameters["RUE"]
        elif (RUE is None):
            RUE = model.PARAMETERS["RUE"]
            
        if (YIELD_FACTOR is None and m is not None):
            YIELD_FACTOR = m.parameters["YIELD_FACTOR"]
        elif (YIELD_FACTOR is None):
            YIELD_FACTOR = model.PARAMETERS["YIELD_FACTOR"]
        
        # ----------------------------------------------------------------
        # iPAR - Total light interception
        # ----------------------------------------------------------------
        if (verbose is True):
            print("Estimating Total light interception (iPAR)...")
        
        #NDVI_EM, iPAR_EM, iPAR_EH, iPAR_HM = ipar.calcIPAR(Norm_TT_EH=self.Norm_TT_EH, 
        #                                                   Norm_SimNDVI_EH=self.Norm_SimNDVI_EH, 
        #                                                   Norm_SimNDVI_HM=self.Norm_SimNDVI_HM, 
        #                                                   norm_iPAR_EH_bounds=norm_iPAR_EH_bounds, 
        #                                                   NDVI_constantIPAR=NDVI_constantIPAR, verbose=verbose)
        # Parallel
        try: 
            # Observed phenology
            if ((self.Norm_TT_EH is not None) and (self.Norm_SimNDVI_HM is not None)):
                print("Normalizing Thermal time from observed phenology...")
                NDVI_EM, iPAR_EM, iPAR_EH, iPAR_HM = ipar.estimate_IPAR(Norm_TT_EH=self.Norm_TT_EH, 
                                                                    Norm_SimNDVI_EH=self.Norm_SimNDVI_EH, 
                                                                    Norm_SimNDVI_HM=self.Norm_SimNDVI_HM, 
                                                                    norm_iPAR_EH_bounds=norm_iPAR_EH_bounds, 
                                                                    NDVI_constantIPAR=NDVI_constantIPAR, verbose=verbose)
                self.NDVI_EM = NDVI_EM
                self.iPAR_EM = iPAR_EM
                self.iPAR_EH = iPAR_EH
                self.iPAR_HM = iPAR_HM

                # Total IPAR
                if (iPAR_EM is not None):
                    self.attributes['iPAR_EM'] = float("{:.3f}".format(np.sum(iPAR_EM))) # use only pred NDVI
                if (iPAR_EH is not None and iPAR_HM is not None):
                    self.iPAR_EHHM = np.concatenate([iPAR_EH,iPAR_HM[1:]]) # remove duplicated value at Head
                    # Update site parameters
                    self.attributes['iPAR_EH'] = float("{:.3f}".format(np.sum(iPAR_EH)))
                    self.attributes['iPAR_HM'] = float("{:.3f}".format(np.sum(iPAR_HM)))
                    self.attributes['iPAR_EHHM'] = float("{:.3f}".format(np.sum(self.iPAR_EHHM)))
                else:
                    self.iPAR_EHHM = None

            else:
                self.iPAR_EHHM = None
                self.NDVI_EM = None
                self.iPAR_EM = None
                self.iPAR_EH = None
                self.iPAR_HM = None
                
            # Estimated Emergence
            if ((self.Norm_TT_pEH is not None) and (self.Norm_SimNDVI_HM is not None)):
                print("Normalizing Thermal time from estimated Emergence to observed Heading...")
                NDVI_pEM, iPAR_pEM, iPAR_pEH, iPAR_HM = ipar.estimate_IPAR(Norm_TT_EH=self.Norm_TT_pEH, 
                                                                    Norm_SimNDVI_EH=self.Norm_SimNDVI_pEH, 
                                                                    Norm_SimNDVI_HM=self.Norm_SimNDVI_HM, 
                                                                    norm_iPAR_EH_bounds=norm_iPAR_EH_bounds, 
                                                                    NDVI_constantIPAR=NDVI_constantIPAR, verbose=verbose)
                self.NDVI_pEM = NDVI_pEM
                self.iPAR_pEM = iPAR_pEM
                self.iPAR_pEH = iPAR_pEH
                self.iPAR_HM = iPAR_HM

                # Total IPAR
                if (iPAR_pEM is not None):
                    self.attributes['iPAR_pEM'] = float("{:.3f}".format(np.sum(iPAR_pEM))) # use only pred NDVI
                if (iPAR_pEH is not None and iPAR_HM is not None):
                    self.iPAR_pEHHM = np.concatenate([iPAR_pEH,iPAR_HM[1:]]) # remove duplicated value at Head
                    # Update site parameters
                    self.attributes['iPAR_pEH'] = float("{:.3f}".format(np.sum(iPAR_pEH)))
                    self.attributes['iPAR_HM'] = float("{:.3f}".format(np.sum(iPAR_HM)))
                    self.attributes['iPAR_pEHHM'] = float("{:.3f}".format(np.sum(self.iPAR_pEHHM)))
                else:
                    self.iPAR_pEHHM = None

            else:
                self.iPAR_pEHHM = None
                self.NDVI_pEM = None
                self.iPAR_pEM = None
                self.iPAR_pEH = None
                self.iPAR_HM = None
            
            # Estimated Heading
            if ((self.Norm_TT_EpH is not None) and (self.Norm_SimNDVI_pHM is not None)):
                print("Normalizing Thermal time from observed Emergence to estimated Heading...")
                NDVI_EM, iPAR_EM, iPAR_EpH, iPAR_pHM = ipar.estimate_IPAR(Norm_TT_EH=self.Norm_TT_EpH, 
                                                                    Norm_SimNDVI_EH=self.Norm_SimNDVI_EpH, 
                                                                    Norm_SimNDVI_HM=self.Norm_SimNDVI_pHM, 
                                                                    norm_iPAR_EH_bounds=norm_iPAR_EH_bounds, 
                                                                    NDVI_constantIPAR=NDVI_constantIPAR, verbose=verbose)
                self.NDVI_EM = NDVI_EM
                self.iPAR_EM = iPAR_EM
                self.iPAR_EpH = iPAR_EpH
                self.iPAR_pHM = iPAR_pHM

                # Total IPAR
                if (iPAR_EM is not None):
                    self.attributes['iPAR_EM'] = float("{:.3f}".format(np.sum(iPAR_EM))) # use only pred NDVI
                if (iPAR_EpH is not None and iPAR_pHM is not None):
                    self.iPAR_EpHpHM = np.concatenate([iPAR_EpH,iPAR_pHM[1:]]) # remove duplicated value at Head
                    # Update site parameters
                    self.attributes['iPAR_EpH'] = float("{:.3f}".format(np.sum(iPAR_EpH)))
                    self.attributes['iPAR_pHM'] = float("{:.3f}".format(np.sum(iPAR_pHM)))
                    self.attributes['iPAR_EpHpHM'] = float("{:.3f}".format(np.sum(self.iPAR_EpHpHM)))
                else:
                    self.iPAR_EpHpHM = None

            else:
                self.iPAR_EpHpHM = None
                self.NDVI_EM = None
                self.iPAR_EM = None
                self.iPAR_EpH = None
                self.iPAR_pHM = None
            
            # # Estimated Emergence - estimated Heading
            if ((self.Norm_TT_pEpH is not None) and (self.Norm_SimNDVI_pHM is not None)):
                print("Normalizing Thermal time from estimated Emergence to estimated Heading...")
                NDVI_pEM, iPAR_pEM, iPAR_pEpH, iPAR_pHM = ipar.estimate_IPAR(Norm_TT_EH=self.Norm_TT_pEpH, 
                                                                    Norm_SimNDVI_EH=self.Norm_SimNDVI_pEpH, 
                                                                    Norm_SimNDVI_HM=self.Norm_SimNDVI_pHM, 
                                                                    norm_iPAR_EH_bounds=norm_iPAR_EH_bounds, 
                                                                    NDVI_constantIPAR=NDVI_constantIPAR, verbose=verbose)
                self.NDVI_pEM = NDVI_pEM
                self.iPAR_pEM = iPAR_pEM
                self.iPAR_pEpH = iPAR_pEpH
                self.iPAR_pHM = iPAR_pHM

                # Total IPAR
                if (iPAR_pEM is not None):
                    self.attributes['iPAR_pEM'] = float("{:.3f}".format(np.sum(iPAR_pEM))) # use only pred NDVI
                if (iPAR_pEpH is not None and iPAR_pHM is not None):
                    self.iPAR_pEpHpHM = np.concatenate([iPAR_pEpH,iPAR_pHM[1:]]) # remove duplicated value at Head
                    # Update site parameters
                    self.attributes['iPAR_pEpH'] = float("{:.3f}".format(np.sum(iPAR_pEpH)))
                    self.attributes['iPAR_pHM'] = float("{:.3f}".format(np.sum(iPAR_pHM)))
                    self.attributes['iPAR_pEpHpHM'] = float("{:.3f}".format(np.sum(self.iPAR_pEpHpHM)))
                else:
                    self.iPAR_pEpHpHM = None

            else:
                self.iPAR_pEpHpHM = None
                self.NDVI_pEM = None
                self.iPAR_pEM = None
                self.iPAR_pEpH = None
                self.iPAR_pHM = None
            
            # Estimated Maturity
            if ((self.Norm_TT_EH is not None) and (self.Norm_SimNDVI_HpM is not None)):
                print("Normalizing Thermal time from observed emergence, heading and estimated maturity...")
                NDVI_EpM, iPAR_EpM, iPAR_EH, iPAR_HpM = ipar.estimate_IPAR(Norm_TT_EH=self.Norm_TT_EH, 
                                                                    Norm_SimNDVI_EH=self.Norm_SimNDVI_EH, 
                                                                    Norm_SimNDVI_HM=self.Norm_SimNDVI_HpM, 
                                                                    norm_iPAR_EH_bounds=norm_iPAR_EH_bounds, 
                                                                    NDVI_constantIPAR=NDVI_constantIPAR, verbose=verbose)
                self.NDVI_EpM = NDVI_EpM
                self.iPAR_EpM = iPAR_EpM
                self.iPAR_EH = iPAR_EH
                self.iPAR_HpM = iPAR_HpM

                # Total IPAR
                if (iPAR_EpM is not None):
                    self.attributes['iPAR_EpM'] = float("{:.3f}".format(np.sum(iPAR_EpM))) # use only pred NDVI
                if (iPAR_EH is not None and iPAR_HpM is not None):
                    self.iPAR_EHHpM = np.concatenate([iPAR_EH,iPAR_HpM[1:]]) # remove duplicated value at Head
                    # Update site parameters
                    self.attributes['iPAR_EH'] = float("{:.3f}".format(np.sum(iPAR_EH)))
                    self.attributes['iPAR_HpM'] = float("{:.3f}".format(np.sum(iPAR_HpM)))
                    self.attributes['iPAR_EHHpM'] = float("{:.3f}".format(np.sum(self.iPAR_EHHpM)))
                else:
                    self.iPAR_EHHpM = None

            else:
                self.iPAR_EHHpM = None
                self.NDVI_EpM = None
                self.iPAR_EpM = None
                self.iPAR_EH = None
                self.iPAR_HpM = None
            
            
            # Estimated heading and estimated maturity
            if ((self.Norm_TT_EpH is not None) and (self.Norm_SimNDVI_pHpM is not None)):
                print("Normalizing Thermal time from estimate heading to estimated maturity...")
                NDVI_EpM, iPAR_EpM, iPAR_EpH, iPAR_pHpM = ipar.estimate_IPAR(Norm_TT_EH=self.Norm_TT_EpH, 
                                                                    Norm_SimNDVI_EH=self.Norm_SimNDVI_EpH, 
                                                                    Norm_SimNDVI_HM=self.Norm_SimNDVI_pHpM, 
                                                                    norm_iPAR_EH_bounds=norm_iPAR_EH_bounds, 
                                                                    NDVI_constantIPAR=NDVI_constantIPAR, verbose=verbose)
                self.NDVI_EpM = NDVI_EpM
                self.iPAR_EpM = iPAR_EpM
                self.iPAR_EpH = iPAR_EpH
                self.iPAR_pHpM = iPAR_pHpM

                # Total IPAR
                if (iPAR_EpM is not None):
                    self.attributes['iPAR_EpM'] = float("{:.3f}".format(np.sum(iPAR_EpM))) # use only pred NDVI
                if (iPAR_EpH is not None and iPAR_pHpM is not None):
                    self.iPAR_EpHpHpM = np.concatenate([iPAR_EpH,iPAR_pHpM[1:]]) # remove duplicated value at Head
                    # Update site parameters
                    self.attributes['iPAR_EpH'] = float("{:.3f}".format(np.sum(iPAR_EpH)))
                    self.attributes['iPAR_pHpM'] = float("{:.3f}".format(np.sum(iPAR_pHpM)))
                    self.attributes['iPAR_EpHpHpM'] = float("{:.3f}".format(np.sum(self.iPAR_EpHpHpM)))
                else:
                    self.iPAR_EpHpHpM = None

            else:
                self.iPAR_EpHpHpM = None
                self.NDVI_EpM = None
                self.iPAR_EpM = None
                self.iPAR_EpH = None
                self.iPAR_pHpM = None
            
            
            # Estimated phenology
            if ((self.Norm_TT_pEpH is not None) and (self.Norm_SimNDVI_pHpM is not None)):
                print("Normalizing Thermal time from estimated phenology...")
                NDVI_pEpM, iPAR_pEpM, iPAR_pEpH, iPAR_pHpM = ipar.estimate_IPAR(Norm_TT_EH=self.Norm_TT_pEpH, 
                                                                    Norm_SimNDVI_EH=self.Norm_SimNDVI_pEpH, 
                                                                    Norm_SimNDVI_HM=self.Norm_SimNDVI_pHpM, 
                                                                    norm_iPAR_EH_bounds=norm_iPAR_EH_bounds, 
                                                                    NDVI_constantIPAR=NDVI_constantIPAR, verbose=verbose)
                self.NDVI_pEpM = NDVI_pEpM
                self.iPAR_pEpM = iPAR_pEpM
                self.iPAR_pEpH = iPAR_pEpH
                self.iPAR_pHpM = iPAR_pHpM

                # Total IPAR
                if (iPAR_pEpM is not None):
                    self.attributes['iPAR_pEpM'] = float("{:.3f}".format(np.sum(iPAR_pEpM))) # use only pred NDVI
                if (iPAR_pEpH is not None and iPAR_pHpM is not None):
                    self.iPAR_pEpHpHpM = np.concatenate([iPAR_pEpH,iPAR_pHpM[1:]]) # remove duplicated value at Head
                    # Update site parameters
                    self.attributes['iPAR_pEpH'] = float("{:.3f}".format(np.sum(iPAR_pEpH)))
                    self.attributes['iPAR_pHpM'] = float("{:.3f}".format(np.sum(iPAR_pHpM)))
                    self.attributes['iPAR_pEpHpHpM'] = float("{:.3f}".format(np.sum(self.iPAR_pEpHpHpM)))
                else:
                    self.iPAR_pEpHpHpM = None

            else:
                self.iPAR_pEpHpHpM = None
                self.NDVI_pEpM = None
                self.iPAR_pEpM = None
                self.iPAR_pEpH = None
                self.iPAR_pHpM = None
            
            
            # Observed heading, estimated emergence and estimated maturity
            if ((self.Norm_TT_pEH is not None) and (self.Norm_SimNDVI_HpM is not None)):
                print("Normalizing Thermal time from estimated Emergence to estimated Maturity...")
                NDVI_pEpM, iPAR_pEpM, iPAR_pEH, iPAR_HpM = ipar.estimate_IPAR(Norm_TT_EH=self.Norm_TT_pEH, 
                                                                    Norm_SimNDVI_EH=self.Norm_SimNDVI_pEH, 
                                                                    Norm_SimNDVI_HM=self.Norm_SimNDVI_HpM, 
                                                                    norm_iPAR_EH_bounds=norm_iPAR_EH_bounds, 
                                                                    NDVI_constantIPAR=NDVI_constantIPAR, verbose=verbose)
                self.NDVI_pEpM = NDVI_pEpM
                self.iPAR_pEpM = iPAR_pEpM
                self.iPAR_pEH = iPAR_pEH
                self.iPAR_HpM = iPAR_HpM

                # Total IPAR
                if (iPAR_pEpM is not None):
                    self.attributes['iPAR_pEpM'] = float("{:.3f}".format(np.sum(iPAR_pEpM))) # use only pred NDVI
                if (iPAR_pEH is not None and iPAR_HpM is not None):
                    self.iPAR_pEHHpM = np.concatenate([iPAR_pEH,iPAR_HpM[1:]]) # remove duplicated value at Head
                    # Update site parameters
                    self.attributes['iPAR_pEH'] = float("{:.3f}".format(np.sum(iPAR_pEH)))
                    self.attributes['iPAR_HpM'] = float("{:.3f}".format(np.sum(iPAR_HpM)))
                    self.attributes['iPAR_pEHHpM'] = float("{:.3f}".format(np.sum(self.iPAR_pEHHpM)))
                else:
                    self.iPAR_pEHHpM = None

            else:
                self.iPAR_pEHHpM = None
                self.NDVI_pEpM = None
                self.iPAR_pEpM = None
                self.iPAR_pEH = None
                self.iPAR_HpM = None
                

        except Exception as err:
            print(HT.bold + HT.fg.red + "Estimating Total light interception (iPAR)"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, 
                                "error": "Estimating Total light interception (iPAR). Error: {}".format(err)})
            
        # ----------------------------------------------------------------
        # fIPAR
        # ----------------------------------------------------------------
        if (verbose is True):
            print("Estimating fiPAR...")
        # Observed phenology
        if ((self.PRFT_EH is not None) and (self.PRFT_HM is not None)):
            self.PRFT_EHHM = np.concatenate([self.PRFT_EH,self.PRFT_HM[1:]]) # remove duplicated value at Head
        else:
            self.PRFT_EHHM = None
        # Observed emergence, maturity - estimated heading
        if ((self.PRFT_EpH is not None) and (self.PRFT_pHM is not None)):
            self.PRFT_EpHpHM = np.concatenate([self.PRFT_EpH,self.PRFT_pHM[1:]]) # remove duplicated value at Head
        else:
            self.PRFT_EpHpHM = None
        # Estimated emergence - observed heading, maturity 
        if ((self.PRFT_pEH is not None) and (self.PRFT_HM is not None)):
            self.PRFT_pEHHM = np.concatenate([self.PRFT_pEH,self.PRFT_HM[1:]]) # remove duplicated value at Head
        else:
            self.PRFT_pEHHM = None
        # Observed emergence, heading - estimated maturity 
        if ((self.PRFT_EH is not None) and (self.PRFT_HpM is not None)):
            self.PRFT_EHHpM = np.concatenate([self.PRFT_EH,self.PRFT_HpM[1:]]) # remove duplicated value at Head
        else:
            self.PRFT_EHHpM = None
        # Estimated emergence, Observed heading - estimated maturity
        if ((self.PRFT_pEH is not None) and (self.PRFT_HpM is not None)):
            self.PRFT_pEHHpM = np.concatenate([self.PRFT_pEH,self.PRFT_HpM[1:]]) # remove duplicated value at Head
        else:
            self.PRFT_pEHHpM = None
        # Observed emergence, estimated heading - estimated maturity
        if ((self.PRFT_EpH is not None) and (self.PRFT_pHpM is not None)):
            self.PRFT_EpHpHpM = np.concatenate([self.PRFT_EpH,self.PRFT_pHpM[1:]]) # remove duplicated value at Head
        else:
            self.PRFT_EpHpHpM = None
        # Estimated phenology
        if ((self.PRFT_pEpH is not None) and (self.PRFT_pHpM is not None)):
            self.PRFT_pEpHpHpM = np.concatenate([self.PRFT_pEpH,self.PRFT_pHpM[1:]]) # remove duplicated value at Head
        else:
            self.PRFT_pEpHpHpM = None
            
        # Solar Radiation
        # ---------------
        # Observed phenology
        if ((self.SolRad_EH is not None) and (self.SolRad_HM is not None)):
            self.SolRad_EHHM = np.concatenate([self.SolRad_EH,self.SolRad_HM[1:]]) # remove duplicated value at Head
        else:
            self.SolRad_EHHM = None
        # Observed emergence, maturity - estimated heading
        if ((self.SolRad_EpH is not None) and (self.SolRad_pHM is not None)):
            self.SolRad_EpHpHM = np.concatenate([self.SolRad_EpH,self.SolRad_pHM[1:]]) # remove duplicated value at Head
        else:
            self.SolRad_EpHpHM = None
        # Estimated emergence - observed heading, maturity
        if ((self.SolRad_pEH is not None) and (self.SolRad_HM is not None)):
            self.SolRad_pEHHM = np.concatenate([self.SolRad_pEH,self.SolRad_HM[1:]]) # remove duplicated value at Head
        else:
            self.SolRad_pEHHM = None
        # Observed emergence, heading - estimated maturity
        if ((self.SolRad_EH is not None) and (self.SolRad_HpM is not None)):
            self.SolRad_EHHpM = np.concatenate([self.SolRad_EH,self.SolRad_HpM[1:]]) # remove duplicated value at Head
        else:
            self.SolRad_EHHpM = None
        # Estimated emergence, Observed heading - estimated maturity
        if ((self.SolRad_pEH is not None) and (self.SolRad_HpM is not None)):
            self.SolRad_pEHHpM = np.concatenate([self.SolRad_pEH,self.SolRad_HpM[1:]]) # remove duplicated value at Head
        else:
            self.SolRad_pEHHpM = None
        # Observed emergence, estimated heading - estimated maturity
        if ((self.SolRad_EpH is not None) and (self.SolRad_pHpM is not None)):
            self.SolRad_EpHpHpM = np.concatenate([self.SolRad_EpH,self.SolRad_pHpM[1:]]) # remove duplicated value at Head
        else:
            self.SolRad_EpHpHpM = None
        # Estimated phenology
        if ((self.SolRad_pEpH is not None) and (self.SolRad_pHpM is not None)):
            self.SolRad_pEpHpHpM = np.concatenate([self.SolRad_pEpH,self.SolRad_pHpM[1:]]) # remove duplicated value at Head
        else:
            self.SolRad_pEpHpHpM = None
        
        try:
            # Calculate fiPAR
            # ---------------
            # Observed phenology
            if ((self.iPAR_EHHM is not None) and (self.PRFT_EHHM is not None) and (self.SolRad_EHHM is not None)):
                self.fIPAR_EHHM = self.iPAR_EHHM * self.PRFT_EHHM * self.SolRad_EHHM * 0.5
            else:
                self.fIPAR_EHHM = None
            # Observed emergence, maturity - estimated heading
            if ((self.iPAR_EpHpHM is not None) and (self.PRFT_EpHpHM is not None) and (self.SolRad_EpHpHM is not None)):
                self.fIPAR_EpHpHM = self.iPAR_EpHpHM * self.PRFT_EpHpHM * self.SolRad_EpHpHM * 0.5
            else:
                self.fIPAR_EpHpHM = None
            # Estimated emergence - observed heading, maturity 
            if ((self.iPAR_pEHHM is not None) and (self.PRFT_pEHHM is not None) and (self.SolRad_pEHHM is not None)):
                self.fIPAR_pEHHM = self.iPAR_pEHHM * self.PRFT_pEHHM * self.SolRad_pEHHM * 0.5
            else:
                self.fIPAR_pEHHM = None
            # Observed emergence, heading - estimated maturity
            if ((self.iPAR_EHHpM is not None) and (self.PRFT_EHHpM is not None) and (self.SolRad_EHHpM is not None)):
                self.fIPAR_EHHpM = self.iPAR_EHHpM * self.PRFT_EHHpM * self.SolRad_EHHpM * 0.5
            else:
                self.fIPAR_EHHpM = None
            # Estimated emergence, Observed heading - estimated maturity
            if ((self.iPAR_pEHHpM is not None) and (self.PRFT_pEHHpM is not None) and (self.SolRad_pEHHpM is not None)):
                self.fIPAR_pEHHpM = self.iPAR_pEHHpM * self.PRFT_pEHHpM * self.SolRad_pEHHpM * 0.5
            else:
                self.fIPAR_pEHHpM = None
            # Observed emergence, estimated heading - estimated maturity
            if ((self.iPAR_EpHpHpM is not None) and (self.PRFT_EpHpHpM is not None) and (self.SolRad_EpHpHpM is not None)):
                self.fIPAR_EpHpHpM = self.iPAR_EpHpHpM * self.PRFT_EpHpHpM * self.SolRad_EpHpHpM * 0.5
            else:
                self.fIPAR_EpHpHpM = None
            # Estimated phenology
            if ((self.iPAR_pEpHpHpM is not None) and (self.PRFT_pEpHpHpM is not None) 
                and (self.SolRad_pEpHpHpM is not None)):
                self.fIPAR_pEpHpHpM = self.iPAR_pEpHpHpM * self.PRFT_pEpHpHpM * self.SolRad_pEpHpHpM * 0.5
            else:
                self.fIPAR_pEpHpHpM = None

            
        except Exception as err:
            print(HT.bold + HT.fg.red + "Estimating fiPAR"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, 
                                "error": "Estimating fiPAR. Error: {}".format(err)})
        
        # ------------------------------------------------------------------------------------
        # Sum of daily fIPAR from Emergence to Heading
        #
        # Observed phenology
        if (self.fIPAR_EHHM is not None):
            self.attributes['sumfIPAR_EHHM'] = float("{:.3f}".format(np.sum(self.fIPAR_EHHM)))
        # Observed emergence, maturity - estimated heading
        if (self.fIPAR_EpHpHM is not None):
            self.attributes['sumfIPAR_EpHpHM'] = float("{:.3f}".format(np.sum(self.fIPAR_EpHpHM)))
        # Estimated emergence - observed heading, maturity
        if (self.fIPAR_pEHHM is not None):
            self.attributes['sumfIPAR_pEHHM'] = float("{:.3f}".format(np.sum(self.fIPAR_pEHHM)))
        # Observed emergence, heading - estimated maturity
        if (self.fIPAR_EHHpM is not None):
            self.attributes['sumfIPAR_EHHpM'] = float("{:.3f}".format(np.sum(self.fIPAR_EHHpM)))
        # Estimated emergence, Observed heading - estimated maturity
        if (self.fIPAR_pEHHpM is not None):
            self.attributes['sumfIPAR_pEHHpM'] = float("{:.3f}".format(np.sum(self.fIPAR_pEHHpM)))
        # Observed emergence, estimated heading - estimated maturity
        if (self.fIPAR_EpHpHpM is not None):
            self.attributes['sumfIPAR_EpHpHpM'] = float("{:.3f}".format(np.sum(self.fIPAR_EpHpHpM)))
        # Estimated phenology
        if (self.fIPAR_pEpHpHpM is not None):
            self.attributes['sumfIPAR_pEpHpHpM'] = float("{:.3f}".format(np.sum(self.fIPAR_pEpHpHpM)))
        
        # --------------------------------------------
        # Normalize iPAR
        # --------------------------------------------
        # Observed phenology
        if (self.iPAR_EHHM is not None):
            self.Norm_iPAR_EHHM = (self.iPAR_EHHM-np.nanmin(self.iPAR_EHHM)) / (np.nanmax(self.iPAR_EHHM)-np.nanmin(self.iPAR_EHHM))
        
        # Observed emergence, maturity - estimated heading
        if (self.iPAR_EpHpHM is not None):
            self.Norm_iPAR_EpHpHM = (self.iPAR_EpHpHM-np.nanmin(self.iPAR_EpHpHM)) / (np.nanmax(self.iPAR_EpHpHM)-np.nanmin(self.iPAR_EpHpHM))
        
        # Estimated emergence - observed heading, maturity 
        if (self.iPAR_pEHHM is not None):
            self.Norm_iPAR_pEHHM = (self.iPAR_pEHHM-np.nanmin(self.iPAR_pEHHM)) / (np.nanmax(self.iPAR_pEHHM)-np.nanmin(self.iPAR_pEHHM))
        
        # Observed emergence, heading - estimated maturity
        if (self.iPAR_EHHpM is not None):
            self.Norm_iPAR_EHHpM = (self.iPAR_EHHpM-np.nanmin(self.iPAR_EHHpM)) / (np.nanmax(self.iPAR_EHHpM)-np.nanmin(self.iPAR_EHHpM))
        
        # Estimated emergence, Observed heading - estimated maturity
        if (self.iPAR_pEHHpM is not None):
            self.Norm_iPAR_pEHHpM = (self.iPAR_pEHHpM-np.nanmin(self.iPAR_pEHHpM)) / (np.nanmax(self.iPAR_pEHHpM)-np.nanmin(self.iPAR_pEHHpM))
        
        # Observed emergence, estimated heading - estimated maturity
        if (self.iPAR_EpHpHpM is not None):
            self.Norm_iPAR_EpHpHpM = (self.iPAR_EpHpHpM-np.nanmin(self.iPAR_EpHpHpM)) / (np.nanmax(self.iPAR_EpHpHpM)-np.nanmin(self.iPAR_EpHpHpM))
            
        # Estimated phenology
        if (self.iPAR_pEpHpHpM is not None):
            self.Norm_iPAR_pEpHpHpM = (self.iPAR_pEpHpHpM-np.nanmin(self.iPAR_pEpHpHpM)) / (np.nanmax(self.iPAR_pEpHpHpM)-np.nanmin(self.iPAR_pEpHpHpM))
            
        # -----------------------------------------
        # Estimate GPP - gross primary production
        # TODO: Check out when self.iPAR, self.iPAR_pEH, or self.iPAR_pEpH
        #       haven't the same size of the SolRad and PRFT 
        # -----------------------------------------
        if (verbose is True):
            print("Estimating GPP - gross primary production...")
        
        # Observed phenology
        try:
            if ( (RUE is not None) and (self.SolRad_EHHM is not None) and (self.PRFT_EHHM is not None) 
                and (self.iPAR_EHHM is not None) ):
                self.GPP_EHHM = self.SolRad_EHHM * 0.5 * RUE * self.PRFT_EHHM * self.iPAR_EHHM
                cGPP_EHHM = np.cumsum(self.GPP_EHHM)
                self.attributes['cGPP_EHHM'] = float("{:.3f}".format(np.nanmax(cGPP_EHHM)))
        except Exception as err:
            if (verbose is True):
                print(err)
        # Observed emergence, maturity - estimated heading
        try:
            if ( (RUE is not None) and (self.SolRad_EpHpHM is not None) and (self.PRFT_EpHpHM is not None) 
                and (self.iPAR_EpHpHM is not None) ):
                self.GPP_EpHpHM = self.SolRad_EpHpHM * 0.5 * RUE * self.PRFT_EpHpHM * self.iPAR_EpHpHM
                cGPP_EpHpHM = np.cumsum(self.GPP_EpHpHM)
                self.attributes['cGPP_EpHpHM'] = float("{:.3f}".format(np.nanmax(cGPP_EpHpHM)))
        except Exception as err:
            if (verbose is True):
                print(err)
        # Estimated emergence - observed heading, maturity 
        try:
            if ( (RUE is not None) and (self.SolRad_pEHHM is not None) and (self.PRFT_pEHHM is not None) 
                and (self.iPAR_pEHHM is not None) ):
                self.GPP_pEHHM = self.SolRad_pEHHM * 0.5 * RUE * self.PRFT_pEHHM * self.iPAR_pEHHM
                cGPP_pEHHM = np.cumsum(self.GPP_pEHHM)
                self.attributes['cGPP_pEHHM'] = float("{:.3f}".format(np.nanmax(cGPP_pEHHM)))
        except Exception as err:
            if (verbose is True):
                print(err)
        # Observed emergence, heading - estimated maturity
        try:
            if ( (RUE is not None) and (self.SolRad_EHHpM is not None) and (self.PRFT_EHHpM is not None) 
                and (self.iPAR_EHHpM is not None) ):
                self.GPP_EHHpM = self.SolRad_EHHpM * 0.5 * RUE * self.PRFT_EHHpM * self.iPAR_EHHpM
                cGPP_EHHpM = np.cumsum(self.GPP_EHHpM)
                self.attributes['cGPP_EHHpM'] = float("{:.3f}".format(np.nanmax(cGPP_EHHpM)))
        except Exception as err:
            if (verbose is True):
                print(err)
        # Estimated emergence, Observed heading - estimated maturity
        try:
            if ( (RUE is not None) and (self.SolRad_pEHHpM is not None) and (self.PRFT_pEHHpM is not None) 
                and (self.iPAR_pEHHpM is not None) ):
                self.GPP_pEHHpM = self.SolRad_pEHHpM * 0.5 * RUE * self.PRFT_pEHHpM * self.iPAR_pEHHpM
                cGPP_pEHHpM = np.cumsum(self.GPP_pEHHpM)
                self.attributes['cGPP_pEHHpM'] = float("{:.3f}".format(np.nanmax(cGPP_pEHHpM)))
        except Exception as err:
            if (verbose is True):
                print(err)
        # Observed emergence, estimated heading - estimated maturity
        try:
            if ( (RUE is not None) and (self.SolRad_EpHpHpM is not None) and (self.PRFT_EpHpHpM is not None) 
                and (self.iPAR_EpHpHpM is not None) ):
                self.GPP_EpHpHpM = self.SolRad_EpHpHpM * 0.5 * RUE * self.PRFT_EpHpHpM * self.iPAR_EpHpHpM
                cGPP_EpHpHpM = np.cumsum(self.GPP_EpHpHpM)
                self.attributes['cGPP_EpHpHpM'] = float("{:.3f}".format(np.nanmax(cGPP_EpHpHpM)))
        except Exception as err:
            if (verbose is True):
                print(err)
        # Estimated phenology
        try:
            if ( (RUE is not None) and (self.SolRad_pEpHpHpM is not None) and (self.PRFT_pEpHpHpM is not None) 
                and (self.iPAR_pEpHpHpM is not None) ):
                self.GPP_pEpHpHpM = self.SolRad_pEpHpHpM * 0.5 * RUE * self.PRFT_pEpHpHpM * self.iPAR_pEpHpHpM
                cGPP_pEpHpHpM = np.cumsum(self.GPP_pEpHpHpM)
                self.attributes['cGPP_pEpHpHpM'] = float("{:.3f}".format(np.nanmax(cGPP_pEpHpHpM)))
        except Exception as err:
            if (verbose is True):
                print(err)
                
        # -----------------------------------------
        # Cumulative GPP from Emergence to Heading
        # TODO: Correct NDVI using estimated GPP_pEH, GPP_pEpH
        # -----------------------------------------
        try:
            if ( (RUE is not None) and (self.SolRad_EH is not None) and 
                (self.PRFT_EH is not None) and (self.iPAR_EH is not None) ):
                GPP_EH = self.SolRad_EH * 0.5 * RUE * self.PRFT_EH * self.iPAR_EH
                cGPP_EH = np.cumsum(GPP_EH)
                self.attributes['cGPP_EH'] = float("{:.3f}".format(np.nanmax(cGPP_EH)))
                self.attributes['sumGPP_EH'] = float("{:.3f}".format(np.sum(GPP_EH)))

                # Correct NDVI at Heading
                corrected_NDVI_atHeading = 0.00024355578828840187 * np.nanmax(cGPP_EH) + 0.5755361655424565
                self.attributes['NDVI_atHeading_EH'] = float("{:.3f}".format(corrected_NDVI_atHeading))
                if (self.Norm_TT_HM is not None):
                    corrected_Norm_SimNDVI = ndvi.calculateNDVI_HM(norm_TT_HM=self.Norm_TT_HM, 
                                                                   NDVImax=corrected_NDVI_atHeading, verbose=verbose)
                    self.NDVI = corrected_Norm_SimNDVI
                    # Update site
                    #self.attributes['NDVI_atHeading'] = corrected_NDVI_atHeading
                    #if (verbose==True):
                    #    print("NDVI at Heading date -> before: {:.3f} - after: {:.3f}"
                    #          .format(self.attributes['NDVI_atHeading_EH'], corrected_NDVI_atHeading ))

                    # Los datos vienen por el momento normalizados por que estan estimados a partir de las curvas,
                    # necesitamos re-escalarlos para poder asignar un NDVI acorde a los demas y poder interpolar
                    # Supongo que como las curvas se obtuvieron con datos observados de NDVI entre 0.16 y 0.94, 
                    # debemos tener en cuenta estos limites a la hora de denormalizar
                    # de-normalize
                    Norm_SimNDVI_inverse = []
                    for v in corrected_Norm_SimNDVI:
                        min_val=0.16
                        max_val=0.94
                        #''' inverse function for de-normalize NDVI '''
                        Norm_SimNDVI_inverse.append((v*(max_val - min_val)) + min_val)

                    self.NDVI = Norm_SimNDVI_inverse
            else:
                self.attributes['cGPP_EH'] = None
                self.attributes['sumGPP_EH'] = None
        
        
            # -----------------------------------------
            # Cumulative GPP from Heading to Maturity
            # -----------------------------------------
            if ( (RUE is not None) and (self.SolRad_pEH is not None) and 
                (self.PRFT_pEH is not None) and (self.iPAR_pEH is not None) ):
                GPP_pEH = self.SolRad_pEH * 0.5 * RUE * self.PRFT_pEH * self.iPAR_pEH
                cGPP_pEH = np.cumsum(GPP_pEH)
                self.attributes['cGPP_pEH'] = float("{:.3f}".format(np.nanmax(cGPP_pEH)))
                self.attributes['sumGPP_pEH'] = float("{:.3f}".format(np.sum(GPP_pEH)))

                # Correct NDVI at Heading
                corrected_NDVI_atHeading_pEH = 0.00024355578828840187 * np.nanmax(cGPP_pEH) + 0.5755361655424565
                self.attributes['NDVI_atHeading_pEH'] = float("{:.3f}".format(corrected_NDVI_atHeading_pEH))

            else:
                self.attributes['cGPP_pEH'] = None
                self.attributes['sumGPP_pEH'] = None

            if ( (RUE is not None) and (self.SolRad_EpH is not None) and 
                (self.PRFT_EpH is not None) and (self.iPAR_EpH is not None) ):
                GPP_EpH = self.SolRad_EpH * 0.5 * RUE * self.PRFT_EpH * self.iPAR_EpH
                cGPP_EpH = np.cumsum(GPP_EpH)
                self.attributes['cGPP_EpH'] = float("{:.3f}".format(np.nanmax(cGPP_EpH)))
                self.attributes['sumGPP_EpH'] = float("{:.3f}".format(np.sum(GPP_EpH)))

                corrected_NDVI_atHeading_EpH = 0.00024355578828840187 * np.nanmax(cGPP_EpH) + 0.5755361655424565
                self.attributes['NDVI_atHeading_EpH'] = float("{:.3f}".format(corrected_NDVI_atHeading_EpH))
            else:
                self.attributes['cGPP_EpH'] = None
                self.attributes['sumGPP_EpH'] = None

            if ( (RUE is not None) and (self.SolRad_pEpH is not None) and 
                (self.PRFT_pEpH is not None) and (self.iPAR_pEpH is not None) ):
                GPP_pEpH = self.SolRad_pEpH * 0.5 * RUE * self.PRFT_pEpH * self.iPAR_pEpH
                cGPP_pEpH = np.cumsum(GPP_pEpH)
                self.attributes['cGPP_pEpH'] = float("{:.3f}".format(np.nanmax(cGPP_pEpH)))
                self.attributes['sumGPP_pEpH'] = float("{:.3f}".format(np.sum(GPP_pEpH)))

                corrected_NDVI_atHeading_pEpH = 0.00024355578828840187 * np.nanmax(cGPP_pEpH) + 0.5755361655424565
                self.attributes['NDVI_atHeading_pEpH'] = float("{:.3f}".format(corrected_NDVI_atHeading_pEpH))
            else:
                self.attributes['cGPP_pEpH'] = None
                self.attributes['sumGPP_pEpH'] = None

            if ( (RUE is not None) and (self.SolRad_HM is not None) and 
                (self.PRFT_HM is not None) and (self.iPAR_HM is not None) ):
                GPP_HM = self.SolRad_HM * 0.5 * RUE * self.PRFT_HM * self.iPAR_HM
                cGPP_HM = np.cumsum(GPP_HM)
                self.attributes['cGPP_HM'] = float("{:.3f}".format(np.nanmax(cGPP_HM)))
                self.attributes['sumGPP_HM'] = float("{:.3f}".format(np.sum(GPP_HM)))
            else:
                self.attributes['cGPP_HM'] = None
                self.attributes['sumGPP_HM'] = None

            if ( (RUE is not None) and (self.SolRad_pHM is not None) and 
                (self.PRFT_pHM is not None) and (self.iPAR_pHM is not None) ):
                GPP_pHM = self.SolRad_pHM * 0.5 * RUE * self.PRFT_pHM * self.iPAR_pHM
                cGPP_pHM = np.cumsum(GPP_pHM)
                self.attributes['cGPP_pHM'] = float("{:.3f}".format(np.nanmax(cGPP_pHM)))
                self.attributes['sumGPP_pHM'] = float("{:.3f}".format(np.sum(GPP_pHM)))
            else:
                self.attributes['cGPP_pHM'] = None
                self.attributes['sumGPP_pHM'] = None

            if ( (RUE is not None) and (self.SolRad_HpM is not None) and 
                (self.PRFT_HpM is not None) and (self.iPAR_HpM is not None) ):
                GPP_HpM = self.SolRad_HpM * 0.5 * RUE * self.PRFT_HpM * self.iPAR_HpM
                cGPP_HpM = np.cumsum(GPP_HpM)
                self.attributes['cGPP_HpM'] = float("{:.3f}".format(np.nanmax(cGPP_HpM)))
                self.attributes['sumGPP_HpM'] = float("{:.3f}".format(np.sum(GPP_HpM)))
            else:
                self.attributes['cGPP_HpM'] = None
                self.attributes['sumGPP_HpM'] = None

            if ( (RUE is not None) and (self.SolRad_pHpM is not None) and 
                (self.PRFT_pHpM is not None) and (self.iPAR_pHpM is not None) ):
                GPP_pHpM = self.SolRad_pHpM * 0.5 * RUE * self.PRFT_pHpM * self.iPAR_pHpM
                cGPP_pHpM = np.cumsum(GPP_pHpM)
                self.attributes['cGPP_pHpM'] = float("{:.3f}".format(np.nanmax(cGPP_pHpM)))
                self.attributes['sumGPP_pHpM'] = float("{:.3f}".format(np.sum(GPP_pHpM)))
            else:
                self.attributes['cGPP_pHpM'] = None
                self.attributes['sumGPP_pHpM'] = None
        
        except Exception as err:
            print(HT.bold + HT.fg.red + "Cumulative GPP from Heading to Maturity"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, 
                                "error": "Cumulative GPP from Heading to Maturity. Error: {}".format(err)})
        
        # =============================
        # Estimate Yield from GPP
        # =============================
        if (verbose is True):
            print("Estimating Yield...")
        try:
            # Observed phenology
            if (self.attributes['sumGPP_HM'] is not None):
                self.attributes['SimYield'] = float("{:.2f}".format(self.attributes['sumGPP_HM'] * YIELD_FACTOR))
                if (verbose==True):
                    print("Wheat Yield (observed HM): {:.2f} t/ha".format(self.attributes['SimYield']))
            else:
                self.attributes['SimYield'] = None
                
            # Estimated heading - observed maturity
            if (self.attributes['sumGPP_pHM'] is not None):
                self.attributes['SimYield_pH'] = float("{:.2f}".format(self.attributes['sumGPP_pHM'] * YIELD_FACTOR))
                if (verbose==True):
                    print("Wheat Yield (estimated Heading): {:.2f} t/ha".format(self.attributes['SimYield_pH']))
            else:
                self.attributes['SimYield_pH'] = None
            
            # Observed heading - estimated maturity
            if (self.attributes['sumGPP_HpM'] is not None):
                self.attributes['SimYield_pM'] = float("{:.2f}".format(self.attributes['sumGPP_HpM'] * YIELD_FACTOR))
                if (verbose==True):
                    print("Wheat Yield (estimated Maturity): {:.2f} t/ha".format(self.attributes['SimYield_pM']))
            else:
                self.attributes['SimYield_pM'] = None
            
            # Estimated phenology
            if (self.attributes['sumGPP_pHpM'] is not None):
                self.attributes['SimYield_pHpM'] = float("{:.2f}".format(self.attributes['sumGPP_pHpM'] * YIELD_FACTOR))
                if (verbose==True):
                    print("Wheat Yield (estimated heading and estimated maturity): {:.2f} t/ha".format(self.attributes['SimYield_pHpM']))
            else:
                self.attributes['SimYield_pHpM'] = None
        
        except Exception as err:
            print(HT.bold + HT.fg.red + "Estimating Yield"+ HT.reset +" {} - {}. Error: {}".format(self.uid, self.loc, err))
            self.errors.append({"uid": self.uid, "loc": self.loc, 
                                "error": "Estimating Yield. Error: {}".format(err)})
        
        #
    
    # =============================
    # Run model
    # =============================
    def fit(self, m=None, season=True, verbose=False):
        '''
            Run a iPAR Yield model to fit yield
            
        :params m: Model to run
        :params sites_to_run: Array of Site objects
        :params season: Display weather statistics for different periods
        
        :resutls: An array of Sites with intermediate results
        
        '''
        if (m is None):
            print("Model parameters not valid")
            return
        #
        try:
            _ = self.getPhenologyDates(m, verbose=verbose) # Get Phenology dates
            _ = self.getEstimatedPhenologyDates(m, verbose=verbose) # Estimate Emergence from GDD
            _ = self.getWeatherParameters(m, season=season, verbose=verbose) # Add weather parameters
            _ = self.estimateNDVI(m, verbose=verbose) # NDVI
            _ = self.getIPAR(m, verbose=verbose) # iPAR and Yield
            self.attributes['UID'] = self.uid
            self.attributes['location'] = self.loc
        except:
            print("Error fitting the model in site UID:{} - {}".format(self.uid, self.loc))
            
        return self.getAttr()
        