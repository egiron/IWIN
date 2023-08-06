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

import os, gc
import numpy as np
import pandas as pd
from datetime import date, datetime
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from scipy.linalg import svd
#from matplotlib import patches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pyhull.convex_hull import ConvexHull
import itertools

from . import *
from ..data import *
from ..util import *

class GGE(object):
    ''' Genotype plus Genotype-Environment interaction model 
    '''
    def __init__(self, data, env=None, gen=None, trait=None, trialname='', params=None, twt=False, verbose=False):
        self.data = data
        self.data_twt = None
        self.twt = twt
        self.env = env
        self.gen = gen
        self.trait = trait
        self.trialname = trialname
        self.env_residuals=None
        self.gen_residuals=None
        self.GE_data = None
        self.tbl_means = None
        self.coord_pc2 = None
        self.coord_pc1 = None
        self.eigenvalues = None
        self.varexpl = None
        self.totalvar = None
        default_params = {'transform':0, 'centering':2, 'scaling':0, 'svp':2, 'coordflip':1, 'n_components':None}
        if (params is None):
            self.params = default_params
        else:
            self.params = {**default_params, **params}
    
    def _validate_centering(self, verbose=False):
        ''' Data Centering Prior to Singular Value Decomposition 
        
            In a genotype by environment two-way table Y, the value of
            each cell can be regarded as mixed effect of the grand mean
            (μ) modified by the genotype (row) main effect (αi), the 
            environment (column) main effect (βj), and the specific genotype (row) 
            by environment (column) interaction (φij), plus any random error (εij):
            
            yij =μ+αi +βj +φij +εij
            
            Yij = μ + Gi + Ej + GEij + εij
            
        '''
        if (self.data_twt is None):
            print("Input data not valid...")
            return
        self.env_residuals = None
        if (self.params['centering'] == 1 or self.params['centering'] == 'global'):
            if verbose is True:
                print("Centering data by grand mean")
            grand_mean = np.mean(self.data_twt).mean()
            self.env_residuals = self.data_twt.apply(lambda x: x - grand_mean, axis=1)
        elif (self.params['centering'] == 2 or self.params['centering'] == "environment"):
            if verbose is True:
                print("Centering data by environment...")
            env_mean =  self.data_twt.mean(axis=0)
            self.env_residuals = self.data_twt.apply(lambda x: x - env_mean, axis=1)
        elif (self.params['centering'] == 3 or self.params['centering'] == "double"):
            if verbose is True:
                print("Centering data by environment and genotypes")
            grand_mean = np.mean(self.data_twt).mean()
            env_mean = self.data_twt.mean(axis=0)
            gen_mean = self.data_twt.mean(axis=1)
            self.env_residuals = self.data_twt.copy()
            for i in range(self.data_twt.shape[0]):
                for j in range(self.data_twt.shape[1]):
                    self.env_residuals.iloc[i, j] = self.data_twt.iloc[i, j] + grand_mean - env_mean[j] - gen_mean[i]
        else:
            print("Option not valid")
        #return self.env_residuals
        
    def _validate_scaling(self):
        ''' Data Scaling Prior to Singular Value Decomposition 
            
            (y –μ–β)/s =(α +φ )/s
            
            When sj is the standard deviation for column (environment or trait) j, the data is said to be “standardized” 
            such that all columns are given the same weight (importance)
        '''
        if (self.params['scaling'] == 1 or self.params['scaling'] == "sd"):
            env_scale = self.data_twt.std(axis=0)
            #gen_scale = self.data_twt.std(axis=1)
            self.env_residuals = self.data_twt.apply(lambda x: x / env_scale, axis=1)
    
    def _validate_svp(self):
        ''' Singular value partitioning 
            
            The singular values must be partitioned into the genotype and environment scores before a biplot 
            can be constructed to approximate the two-way data. 
            
            Two methods are particularly useful: column-metric preserving and row-metric 
            preserving (Gabriel 2002; Yan 2002).
            A third method is symmetrical partitioning, which has been the most used, but not necessarily the most useful, 
            singular value partitioning method.
        '''
        coordgen, coordenv = None, None
        d1, d2 = None, None
        U, d, V = self._SVD(self.env_residuals)
        if (self.params['svp'] == 1 or self.params['svp'] == "genotype"): #row-metric preserving
            # this partitioning recovers the Euclidean distances among row factors (here genotypes) and is, 
            # therefore, appropriate for visualizing the similarity/dissimilarity among row factors
            coordgen = np.dot(U[0:len(d),:].T, np.multiply(np.eye(len(d)), d)).T
            coordenv = V.T
            d1 = (max(coordenv[0]) - min(coordenv[0]))/(max(coordgen[0]) - min(coordgen[0]))
            d2 = (max(coordenv[1]) - min(coordenv[1]))/(max(coordgen[1]) - min(coordgen[1]))
            #coordenv = coordenv/max(d1, d2)
            coordenv = coordenv/min(d1, d2)

        elif (self.params['svp'] == 2 or self.params['svp'] == "environment"): #column-metric preserving
            # the singular values are entirely partitioned into the column (here environment) eigenvectors, 
            # referred to as column-metric preserving. Therefore, this partitioning is appropriate 
            # for studying the relationships among column factors.
            coordgen = U.T
            coordenv = np.dot(V[0:len(d),:].T, np.multiply(np.eye(len(d)), d)).T
            d1 = (max(coordgen[0]) - min(coordgen[0]))/(max(coordenv[0]) - min(coordenv[0]))
            d2 = (max(coordgen[1]) - min(coordgen[1]))/(max(coordenv[1]) - min(coordenv[1]))
            #coordgen = coordgen/max(d1, d2) # error
            coordgen = coordgen/min(d1, d2)

        elif (self.params['svp'] == 3 or self.params['svp'] == "symmetrical"):
            coordgen = np.dot(U[0:len(d),:].T, np.multiply(np.eye(len(d)), np.sqrt(d) )).T
            coordenv = np.dot(V[0:len(d),:].T, np.multiply(np.eye(len(d)), np.sqrt(d) )).T
        else:
            print("Option not valid for svp")
            return
        
        # Principal factor coordinates
        if ((coordgen is not None) and (coordenv is not None)):
            coord_pc1 = coordgen[0:2,:].T
            self.coord_pc2 = coordenv[0:2,:].T
            self.coord_pc1 = coord_pc1 * -1 if (self.params['coordflip']==1) else coord_pc1
        
        if (d is not None):
            self.eigenvalues = d
            self.varexpl = ((d**2 / sum(d**2)) * 100).round(2)
            self.totalvar = self.varexpl[0] + self.varexpl[1]
        
    
    def _SVD(self, residuals=None):
        ''' Singular Value Decomposition 
        
        :params residuals: A two-way table of means
        
        '''
        if (residuals is None):
            print("Input not valid...")
            return
        U, d, V = svd(residuals)
        return U, d, V
    
    def best_G(self, verbose=False):
        '''
            Select the best or nearest Genotype after GGE ranking Genotypes relative to the Ideal Genotype

        '''
        if (self.GE_data is None):
            print("Data not found")
            return
        E = self.GE_data[self.GE_data['class']=='Environment']
        G = self.GE_data[self.GE_data['class']=='Genotype']
        x2 = np.mean(E['factor1'])
        y2 = np.mean(E['factor2'])
        xx2 = np.square(G['factor1'])
        yy2 = np.square(G['factor2'])
        # Ranking gens
        mod = max((xx2 + yy2)**0.5)
        xcoord = (mod**2 / (1 + y2**2 / x2**2))**0.5
        ycoord = (y2/x2) * xcoord
        A = G.iloc[:,0:2].to_numpy()
        A = np.array(A)
        meanG = np.array((xcoord, ycoord))
        distances = np.linalg.norm(A-meanG, axis=1)
        min_index = np.argmin(distances)
        selG = G.loc[[min_index]]['label'].values[0]
        if (verbose is True):
            print(f"The closest point is {selG} - {A[min_index]}, at a distance of {distances[min_index]}")
        del E,G, x2,y2, xx2,yy2,mod,xcoord,ycoord,A,meanG,distances,min_index
        _ = gc.collect()
        return selG
    
    def calc_env_residuals(self):
        '''Cálculo de residuales con respecto a la media de los ambientes'''
        self.env_mean =  self.data_twt.mean(axis=0)
        self.env_residuals = self.data_twt.apply(lambda x: x - self.env_mean, axis=1)
        return self.env_residuals
    
    def calc_gen_residuals(self):
        '''Cálculo de residuales con respecto a la media de los genotipos'''
        self.gen_mean =  self.data_twt.mean(axis=1)
        self.gen_residuals = self.data_twt.apply(lambda x: x - self.gen_mean, axis=0)
        return self.gen_residuals
    
    def get_twoway_table(self, impute_missing_threshold=.5, fillna=True, verbose=False):
        ''' Create two-way table of means
        '''
        tbl_means = pd.pivot_table(self.data, values=self.trait, index=[self.gen],columns=[self.env], aggfunc=np.mean)
        # Order Genotypes
        try:
            tbl_means['idx'] = tbl_means.index.str.rsplit('G').str[-1].astype(int)
            tbl_means.reset_index(inplace=True)
            tbl_means = tbl_means.sort_values(['idx']).drop('idx', axis=1)
            tbl_means.set_index(self.gen, inplace=True )
        except Exception as err:
            print("Ordering Genotypes. Error: {}".format(err))
        # Order Environments
        try:
            tbl_means = tbl_means[sorted(tbl_means.columns, key=lambda x: tuple(map(str,x.split('_'))))]
            tbl_means = tbl_means.reindex(sorted(tbl_means.columns, key=lambda x: float(x[1:])), axis=1)
        except Exception as err:
            print("Ordering Environments. Error: {}".format(err))
        # Impute missing values
        try:
            tbl_means = self.impute_missing_values(data=tbl_means, impute_missing_threshold=impute_missing_threshold,
                                                   fillna=fillna, verbose=verbose)
        except Exception as err:
            print("Imputing missing values. Error: {}".format(err))
        
        self.data_twt = tbl_means
        return tbl_means
        
        
    def impute_missing_values(self, data=None, impute_missing_threshold=.5, fillna=True, verbose=False):
        ''' Select better environments and Genotypes along sites with no empty values. 
            Remove environments or Genotypes with more than 50% or threshold value of empty data.
            Fill the rest of the null values with the mean by Environment.

            :params data: Table of Genotypes, Environments and GE means
            :params impute_missing_threshold: Upper limit in percentage to remove empty Environments and Genotypes records

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
            if (mis_perc > impute_missing_threshold): 
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
            if (mis_perc > impute_missing_threshold):
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
    
    def prepare_data(self, verbose=False):
        '''Prepare data for GGE biplots'''
        if (self.env_residuals is None):
            print("Residuals not found")
            return
        if ((self.coord_pc1 is None) and (self.coord_pc2 is None)):
            print("Principal components not found")
            return
            
        try:
            # Create table
            df_coord_gen = pd.DataFrame(self.coord_pc1)
            df_coord_gen['class'] = 'Genotype'
            df_coord_gen['label'] = list(self.data_twt.index)
            df_coord_env = pd.DataFrame(self.coord_pc2)
            df_coord_env['class'] = 'Environment'
            df_coord_env['label'] = list(self.data_twt.columns)
            GE_data = pd.concat([df_coord_gen, df_coord_env], axis=0)
            GE_data.columns = ['factor1', 'factor2', 'class', 'label']
            GE_data[['factor1', 'factor2']] = GE_data[['factor1', 'factor2']].astype(float).round(3)
            self.GE_data = GE_data
            
        except Exception as err:
            print(err)
        
        return self.GE_data
        
    
    def fit(self, impute_missing_threshold=.5, fillna=True, verbose=False):
        if (self.twt is False):
            _ = self.get_twoway_table(impute_missing_threshold=impute_missing_threshold, fillna=fillna, verbose=verbose)
        else:
            self.data_twt = self.data
        # Validate input parameters
        self._validate_centering()
        self._validate_scaling()
        self._validate_svp()
        _ = self.prepare_data()
        
    
    def getMegaEnvironments(self, lim_span=0.25, mxlim=10, verbose=False):
        '''
            Get Mega-Environments using GGE biplot
        '''
        if (self.GE_data is None):
            print("Data not found")
            return
        pt_c = (0,0)
        data = self.GE_data.copy()
        principal_E = data[data['class']=='Environment']
        principal_G = data[data['class']=='Genotype']
        if (len(principal_G['label'].unique()) <=2):
            print("It's not possible to carry out a convex hull for only 2 genotypes.")
            return
        points_E = []
        points_G = []
        points_EG = []
        for i in range(len(principal_G)):
            x = float(principal_G['factor1'].values[i])
            y = float(principal_G['factor2'].values[i])
            points_G.append((x,y))
            points_EG.append((x,y))
        for i in range(len(principal_E)):
            x = float(principal_E['factor1'].values[i])
            y = float(principal_E['factor2'].values[i])
            points_E.append((x,y))
            points_EG.append((x,y))

        # Mega-environment partitions by Genotype
        hull = ConvexHull(points_G) # Genotypes
        # Get parts
        x_min, x_max, y_min, y_max, vert = boundingBox(points_EG)
        rect = []
        line_extr = []
        vertices = []
        #lineas = []
        for s in hull.simplices:
            for d in itertools.combinations(s.coords, len(s.coords)):
                d = np.array(d)
                # vertices
                vx, vy = d[:,0], d[:,1] #print("verdaderos vertices",d[0])
                vx2 = [d[0][0],d[0][1]]
                vertices.append(vx2)
                x4,y4 = perpendicular(vx[0], vy[0], vx[1], vy[1], 0, 0)
                p0, p1 = [0,x4], [0,y4]
                p_prj = [p0[1], p1[1]]
                rect.append(p_prj)
                p_extr = getExtrapoledLine((0,0),(x4*(mxlim/2),y4*(mxlim/2)), mxlim) # Extrapolate
                x,y = (*p_extr.xy,)
                line_extr.append(p_extr)
                #lineas.append(l.get_data())
        #
        # -----------------------
        # Draw rect - get extent
        #rect = sorted(rect, key=clockwiseangle_and_distance)
        #p1, p2, p3, p4, p1 = getRectangle(rect)
        # scale points
        x_min3 = x_min-lim_span
        x_max3 = x_max+lim_span
        y_min3 = y_min-lim_span
        y_max3 = y_max+lim_span
        b = box(x_min3, y_min3, x_max3, y_max3, ccw=False)
        #print(b.wkt)
        poly_rect = Polygon(b)
        
        # -----------------------
        # Add circle
        circle = plt.Circle((0, 0), mxlim, alpha=1, ls='--', lw=2, edgecolor='red', fill=False, clip_on=True)
        c_vertices = circle.get_path().vertices*mxlim
        poly_circle = Polygon(c_vertices)
        # transform the polygon into a LineString
        p_c = LineString(list(poly_circle.exterior.coords))

        # -----------------------
        # Diferenciar Genotypes dentro del convhull y los vertices
        vertices = sorted(vertices, key=clockwiseangle_and_distance) 
        poly_box = Polygon(vertices)
        convhull_vertices = []
        genInside = []
        for i, tp in enumerate(points_G):
            p_obj = Point(tp[0],tp[1])
            if (p_obj.within(poly_box)): #if(poly_box.contains(p_obj)):
                genInside.append((tp[0],tp[1]))
            else:
                convhull_vertices.append((tp[0],tp[1]))
        #
        # -----------------------
        # Order intersection points
        points_sorted_line_extr = []
        for le in line_extr:
            pt_start = np.array(le.coords[0])
            pt_end = np.array(le.coords[len(le.coords)-1])
            points_sorted_line_extr.append(pt_end)

        # -----------------------
        # Debe ordenarse para que produzca los resultados esperados
        points_sorted_line_extr = sorted(points_sorted_line_extr, key=clockwiseangle_and_distance)
        convhull_vertices = sorted(convhull_vertices, key=clockwiseangle_and_distance)
        poly_megaenv = Polygon(convhull_vertices)
        convhull_ME = {}
        convhull_ME['convhull'] = convhull_vertices
        convhull_ME['lines'] = points_sorted_line_extr
        convhull_ME['extent'] = [x_min3, y_min3, x_max3, y_max3]
        
        # Mega-environment polygons
        polyg_ME = {}
        for i,vt in enumerate(convhull_vertices):
            pt = points_sorted_line_extr[i]
            le = LineString([Point(pt_c), Point(pt[0],pt[1])])
            x,y = getInters(p_c, le)
            pt_inter_1 = (x[0], y[0])
            # Get next point to create poly
            #if (i < len(convhull_vertices)-1):
            pt = points_sorted_line_extr[i-1]
            le = LineString([Point(pt_c), Point(pt[0],pt[1])])
            x,y = getInters(p_c, le)
            pt_inter_2 = (x[0], y[0])
            ME = (pt_c, pt_inter_1, pt_inter_2)
            poly_megaenv = Polygon(ME)
            me_intersection = poly_megaenv.intersection(poly_rect) #Polygon(rect_vertices))
            try:
                cx,cy = me_intersection.centroid.xy
                ME_clippled = me_intersection.exterior.coords
                # Polygons
                ME_id = i+1
                polyg_ME[i] = {"ME":ME_id, "poly_megaenv":poly_megaenv, "shape":ME_clippled, "wkt":Polygon(ME_clippled),
                               "centroid":[cx,cy] }
            except:
                pass

        # Observation within Mega-environments
        points_ME = {}
        data.reset_index(inplace=True)
        data.rename(columns={'index':'id'}, inplace=True)
        data['ME'] = -99
        for k in polyg_ME.keys():
            ME_id = polyg_ME[k]['ME']
            poly_megaenv = polyg_ME[k]['poly_megaenv']
            for idx in data.index:
                _class = data['class'][idx]
                x = data['factor1'][idx]
                y = data['factor2'][idx]
                p_obj = Point(x,y)
                if (p_obj.within(poly_megaenv)):
                    points_ME["obs_{}".format(idx)] = { "id":idx, "ME":"ME{}".format(ME_id), 'class':_class }
                    data.loc[idx,'ME'] = ME_id

        # Relate results
        # Genotype
        #for i, p in enumerate(points_G):
        #    if p in convhull_vertices:
        #        print(i, p, principal_G.iloc[i])
        #megaenv = pd.DataFrame(points_ME).T
        #megaenv = megaenv.sort_values(by=['id'])
        #megaenv.reset_index(inplace=True)
        #megaenv.rename(columns={'index':'obs'}, inplace=True)
        
        self.ME = {"polygon":polyg_ME, "convhull":convhull_ME, "extent":[x_min3, y_min3, x_max3, y_max3]}
        self.ME_data = data
        return data, polyg_ME, convhull_ME
    
    def plot(self, title=None, bptype=0, selE=None, selG=None, plot_params=None,  
             saveFig=False, showFig=True, dirname='./', fmt='pdf'):
        ''' Display biplot '''
        plot_default_params = dict(
            limspan=1.2, labelG=True, labelE=True, txtspan=0.02, arrowE=True, arrowG=True, 
            arrowpropsE=dict(arrowstyle= '->', color='red', lw=1, ls='-', alpha=0.7), 
            arrowpropsG=dict(arrowstyle= '-', color='gray', lw=1, ls='-', alpha=0.35),
            fontsizeE=7, colorE='red', sE=20, alphaE=0.75, hueE='label', colorLabelE='red', haE='center', vaE='bottom',
            fontsizeG=9.5, colorG='blue', sG=40, alphaG=0.75, hueG='label', colorLabelG='black', haG='center', vaG='bottom',
            ncircles=3, scircles=0.25, colCircles='gray', lwcircle=1, alphaCircle=0.35,
            limspan2=1.2, labelG2=True, labelE2=True, txtspan2=0.05, arrowE2=False, arrowG2=True, 
            arrowpropsE2=dict(arrowstyle= '-', color='gray', lw=1, ls='--', alpha=0.57), 
            arrowpropsG2=dict(arrowstyle= '-', color='gray', lw=1, ls='--', alpha=0.5),
            fontsizeE2=8.2, colorE2='red', sE2=20, alphaE2=0.75, hueE2='label', colorLabelE2='red', 
            haE2='center', vaE2='bottom', fontsizeG2=9.5, colorG2='blue', sG2=40, 
            alphaG2=0.75, hueG2='label', colorLabelG2='black', haG2='center', vaG2='bottom',
            ncol=2, leg_fontsize=10, leg_gen=None, leg_env=None, showLegend=True,
            convhull_marker="o", convhull_markersize=15, convhull_markeredgecolor="blue", 
            convhull_markerfacecolor="white", convhull_linecolor="blue", convhull_ls='-', convhull_lw=2,
            convhull_perpline_lw=1, convhull_perpline_ls='--', convhull_perpline_color='blue',
            me_marker="D", me_markersize=8, me_markeredgecolor='red', me_markerfacecolor='red',
            me_fontsize=14, me_fontcolor='black', me_ha='center', me_va='bottom', me_linewidth=2, me_alpha=0.1,
            drawME=True, fillME=True, drawDivisions=True, drawConvexhull=True, 
            lim_span=0.25, mxlim=10,
            me_colors=['g','r','y','b','orange','brown','c', '#4203c9', '#16acea', '#e89f1e','#d71b6b']
        )
        if (plot_params is None):
            plot_params = plot_default_params
        else:
            plot_params = {**plot_default_params, **plot_params}
            
        if (self.GE_data is None):
            print("data not found to plot")
            return
        
        if (bptype==0):
            #prepare the target
            #Concat it with target variable to create a complete Dataset
            principal_E = self.GE_data[self.GE_data['class']=='Environment']
            principal_G = self.GE_data[self.GE_data['class']=='Genotype']
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data = principal_E , x = 'factor1',y = 'factor2' , hue=plot_params['hueE'], 
                                 marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'factor1',y = 'factor2' , hue=plot_params['hueG'], 
                                 alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(principal_E['factor1'].min(), principal_G['factor1'].min())) - plot_params['limspan']
            minlim_y = int(min(principal_E['factor2'].min(), principal_G['factor2'].min())) - plot_params['limspan']
            maxlim_x = int(max(principal_E['factor1'].max(), principal_G['factor1'].max())) + plot_params['limspan']
            maxlim_y = int(max(principal_E['factor2'].max(), principal_G['factor2'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            if (title==None): title='Relationship among environments \n'+ self.trialname + ' [' +self.trait+ ']'
            plt.suptitle(title, fontsize=18, y=1.02)
            ax1.set_title("Exp.Var= {:.0f}%, Transform = 0, Scaling = {}, Centering = {}, SVP = {}"
                          .format(self.totalvar, self.params['scaling'], self.params['centering'], 
                                  self.params['svp']), fontsize=8.5)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.varexpl[0]), fontsize=12)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.varexpl[1]), fontsize=12)
            #ax1.set_xlabel("Factor 1 ({:.1f}%)".format(self.varexpl[0])) #self.totalvar = 
            #ax1.set_ylabel("Factor 2 ({:.1f}%)".format(self.varexpl[1]))
            
            # Add circles
            #circle1 = patches.Circle((0, 0), radius=0.5, color='gray', alpha=0.2)
            for c in range(int(min(maxlim_x, maxlim_y))*plot_params['ncircles']):
                circle = plt.Circle((0, 0), c*plot_params['scircles'], alpha=plot_params['alphaCircle'], 
                                    lw=plot_params['lwcircle'], edgecolor=plot_params['colCircles'], fill=False) 
                #ax1.add_patch(circle)
                ax1.add_artist(circle)
            ax1.axis('equal')

            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['factor1'].values[i]
                    ytext = principal_E['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_E['label'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE']) 
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['factor1'].values[i]
                    ytext = principal_G['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_G['label'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
        
        elif (bptype==1):
            principal_E = self.GE_data[self.GE_data['class']=='Environment']
            principal_G = self.GE_data[self.GE_data['class']=='Genotype']
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data = principal_E , x = 'factor1',y = 'factor2' , hue=plot_params['hueE'], 
                                 marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'factor1',y = 'factor2' , hue=plot_params['hueG'], 
                                 alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(principal_E['factor1'].min(), principal_G['factor1'].min())) - plot_params['limspan']
            minlim_y = int(min(principal_E['factor2'].min(), principal_G['factor2'].min())) - plot_params['limspan']
            maxlim_x = int(max(principal_E['factor1'].max(), principal_G['factor1'].max())) + plot_params['limspan']
            maxlim_y = int(max(principal_E['factor2'].max(), principal_G['factor2'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            #ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            #ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            if (title==None): title='Discrimitiveness vs. representativenss of test environments \n'+ self.trialname + ' [' +self.trait+ ']'
            plt.suptitle(title, fontsize=18, y=1.02)
            ax1.set_title("Exp.Var= {:.0f}%, Transform = 0, Scaling = {}, Centering = {}, SVP = {}"
                          .format(self.totalvar, self.params['scaling'], self.params['centering'], self.params['svp']), 
                          fontsize=8.5)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.varexpl[0]), fontsize=12)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.varexpl[1]), fontsize=12)
            
            # ideal test environment
            x2 = np.mean(principal_E['factor1'])
            y2 = np.mean(principal_E['factor2'])
            circle0 = plt.Circle((x2, y2), radius=0.1, alpha=0.8, lw=1, edgecolor='red',fill=False)
            circle1 = plt.Circle((x2, y2), radius=0.01, color='red', alpha=0.8)
            ax1.add_artist(circle0)
            ax1.add_artist(circle1)
            # Add line from the biplot origin to the longest vector of all environments
            ax1.annotate("", xy=(x2, y2), xytext=(0,0), arrowprops=dict(arrowstyle= '->', color='r', 
                                                                        lw=3, ls='-', alpha=0.95),
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            # Add an infinite line
            plt.axline(xy1=(0, 0), xy2=(x2, y2), color='black')
            # Add circles
            for c in range(plot_params['ncircles']): #int(min(maxlim_x, maxlim_y))*
                if (c<=2):
                    circle = plt.Circle((0, 0), c*plot_params['scircles'], alpha=plot_params['alphaCircle'], 
                                        lw=plot_params['lwcircle'], edgecolor=plot_params['colCircles'],
                                        fill=False, clip_on=True)  
                    ax1.add_artist(circle)
                else:
                    circle = plt.Circle((0, 0), c*plot_params['scircles'], alpha=plot_params['alphaCircle'], ls='--', 
                                        lw=plot_params['lwcircle'], edgecolor=plot_params['colCircles'],
                                        fill=False, clip_on=True)  
                    ax1.add_artist(circle)
            #ax1.set_clip_on(True)
            ax1.axis('equal')
            
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['factor1'].values[i]
                    ytext = principal_E['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_E['label'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE']) 
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['factor1'].values[i]
                    ytext = principal_G['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_G['label'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            
        
        elif (bptype==2):
            principal_E = self.GE_data[self.GE_data['class']=='Environment']
            principal_G = self.GE_data[self.GE_data['class']=='Genotype']
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data = principal_E , x = 'factor1',y = 'factor2' , hue=plot_params['hueE'], 
                                 marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'factor1',y = 'factor2' , hue=plot_params['hueG'], 
                                 alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(principal_E['factor1'].min(), principal_G['factor1'].min())) - plot_params['limspan']
            minlim_y = int(min(principal_E['factor2'].min(), principal_G['factor2'].min())) - plot_params['limspan']
            maxlim_x = int(max(principal_E['factor1'].max(), principal_G['factor1'].max())) + plot_params['limspan']
            maxlim_y = int(max(principal_E['factor2'].max(), principal_G['factor2'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            #ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            #ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            if (title==None): title='Ranking environments based on both discriminating ability and representativeness \n'+ self.trialname + ' [' +self.trait+ ']'
            plt.suptitle(title, fontsize=18, y=1.02)
            ax1.set_title("Exp.Var= {:.0f}%, Transform = 0, Scaling = {}, Centering = {}, SVP = {}"
                          .format(self.totalvar, self.params['scaling'], self.params['centering'], 
                                  self.params['svp']), fontsize=8.5)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.varexpl[0]), fontsize=12)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.varexpl[1]), fontsize=12)
            # ideal test environment
            x2 = np.mean(principal_E['factor1'])
            y2 = np.mean(principal_E['factor2'])
            xx2 = np.square(principal_E['factor1'])
            yy2 = np.square(principal_E['factor2'])
            # process gens
            mod = max((xx2 + yy2)**0.5)
            xcoord = (mod**2 / (1 + y2**2 / x2**2))**0.5
            ycoord = (y2/x2) * xcoord
            circle2 = plt.Circle((xcoord, ycoord), radius=0.1, alpha=0.8, lw=1, edgecolor='red',fill=False)
            ax1.add_patch(circle2)
            circle0 = plt.Circle((x2, y2), radius=0.1, alpha=0.8, lw=1, edgecolor='red',fill=False)
            circle1 = plt.Circle((x2, y2), radius=0.01, color='red', alpha=0.8)
            ax1.add_artist(circle0)
            ax1.add_artist(circle1)
            # Add line from the biplot origin to the longest vector of all environments
            ax1.annotate("", xy=(xcoord, ycoord), xytext=(0,0), arrowprops=dict(arrowstyle= '->', color='b', lw=2, 
                                                                                ls='-', alpha=0.95),
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            # Add an infinite line
            plt.axline(xy1=(0, 0), xy2=(xcoord, ycoord), color='black')
            # Add a perpendicular line
            plt.axline(xy1=(0, 0), xy2=(-ycoord, xcoord), color='black')
            # Add circles
            for c in range(plot_params['ncircles']): #int(min(maxlim_x, maxlim_y))*
                    circle = plt.Circle((xcoord, ycoord), c*plot_params['scircles'], alpha=plot_params['alphaCircle'], 
                                        lw=plot_params['lwcircle'], edgecolor=plot_params['colCircles'],
                                        fill=False, clip_on=True) #facecolor, color='b', 
                    ax1.add_artist(circle)
            #
            ax1.axis('equal')
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['factor1'].values[i]
                    ytext = principal_E['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_E['label'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE'])
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['factor1'].values[i]
                    ytext = principal_G['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_G['label'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
        
        elif (bptype==3):
            principal_E = self.GE_data[self.GE_data['class']=='Environment']
            principal_G = self.GE_data[self.GE_data['class']=='Genotype']
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data = principal_E , x = 'factor1',y = 'factor2' , hue=plot_params['hueE'], 
                                 marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'factor1',y = 'factor2' , hue=plot_params['hueG'], 
                                 alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(principal_E['factor1'].min(), principal_G['factor1'].min())) - plot_params['limspan']
            minlim_y = int(min(principal_E['factor2'].min(), principal_G['factor2'].min())) - plot_params['limspan']
            maxlim_x = int(max(principal_E['factor1'].max(), principal_G['factor1'].max())) + plot_params['limspan']
            maxlim_y = int(max(principal_E['factor2'].max(), principal_G['factor2'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            #ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            #ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            #if (title==None): title='Average Environment Coordination: ranking genotypes based on mean performance'
            if (title==None): title='Ranking genotypes based on mean performance\n'+ self.trialname + ' [' +self.trait+ ']'
            plt.suptitle(title, fontsize=18, y=1.02)
            ax1.set_title("Exp.Var= {:.0f}%, Transform = 0, Scaling = {}, Centering = {}, SVP = {}"
                          .format(self.totalvar, self.params['scaling'], self.params['centering'], self.params['svp']), 
                          fontsize=8.5)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.varexpl[0]), fontsize=12)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.varexpl[1]), fontsize=12)
            # Ranking Genotypes Relative to the Ideal Genotype
            # An ideal genotype should have both high mean performance and high stability across environments.
            # ideal test environment
            x2 = np.mean(principal_E['factor1'])
            y2 = np.mean(principal_E['factor2'])
            circle0 = plt.Circle((x2, y2), radius=0.05, alpha=0.8, lw=1, edgecolor='red',fill=False)
            circle1 = plt.Circle((x2, y2), radius=0.01, color='red', alpha=0.8)
            ax1.add_artist(circle0)
            ax1.add_artist(circle1)
            # Add line from the biplot origin to the longest vector of all environments
            ax1.annotate("", xy=(x2, y2), xytext=(0,0), arrowprops=dict(arrowstyle= '->', color='r', lw=3, 
                                                                        ls='-', alpha=0.95),
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)

            # Add an infinite line
            plt.axline(xy1=(0, 0), xy2=(x2, y2), color='black')
            plt.axline(xy1=(0, 0), xy2=(-y2, x2), color='black')
            # Add perpendicular line to each Genotype
            slope_xprj = (y2/x2)
            for idx in principal_G.index: 
                x = float(principal_G['factor1'][idx])
                y = float(principal_G['factor2'][idx])
                # slope of connecting line y = mx+b
                m = -np.reciprocal(slope_xprj)
                b = y-m*x
                # find intersecting point
                zx = b/(slope_xprj-m)
                zy = slope_xprj*zx
                # draw line
                plt.annotate('',(zx,zy),(x,y),arrowprops=dict(linewidth=1,arrowstyle='-',color='gray', ls='--', alpha=0.95))
            
            ax1.axis('equal')
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['factor1'].values[i]
                    ytext = principal_E['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_E['label'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE']) 
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['factor1'].values[i]
                    ytext = principal_G['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_G['label'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
        
        elif (bptype==4):
            if (selE is None):
                print("Environment not defined. please use 'selE' option to specific one")
                return
            principal_E = self.GE_data[self.GE_data['class']=='Environment']
            principal_G = self.GE_data[self.GE_data['class']=='Genotype']
            if (selE not in principal_E['label'].unique()):
                print("Selected environment is not valid. Please choose correct one between {}"
                      .format(list(principal_E['label'].unique())))
                return
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data = principal_E , x = 'factor1',y = 'factor2' , hue=plot_params['hueE'], 
                                 marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'factor1',y = 'factor2' , hue=plot_params['hueG'], 
                                 alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(principal_E['factor1'].min(), principal_G['factor1'].min())) - plot_params['limspan']
            minlim_y = int(min(principal_E['factor2'].min(), principal_G['factor2'].min())) - plot_params['limspan']
            maxlim_x = int(max(principal_E['factor1'].max(), principal_G['factor1'].max())) + plot_params['limspan']
            maxlim_y = int(max(principal_E['factor2'].max(), principal_G['factor2'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            #ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            #ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            #if (title==None): title='Average Environment Coordination: ranking genotypes based on mean performance'
            if (title==None): title='Ranking Genotypes Based on Performance in Environment {}\n'.format(selE) + self.trialname + ' [' +self.trait+ ']'
            plt.suptitle(title, fontsize=18, y=1.02)
            ax1.set_title("Exp.Var= {:.0f}%, Transform = 0, Scaling = {}, Centering = {}, SVP = {}"
                          .format(self.totalvar, self.params['scaling'], self.params['centering'], self.params['svp']),
                          fontsize=8.5)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.varexpl[0]), fontsize=12)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.varexpl[1]), fontsize=12)
            # Select an Environment
            x2 = float(principal_E['factor1'][principal_E['label']==selE])
            y2 = float(principal_E['factor2'][principal_E['label']==selE])
            circle0 = plt.Circle((x2, y2), radius=0.1, alpha=0.8, lw=1, edgecolor='red',fill=False)
            circle1 = plt.Circle((x2, y2), radius=0.01, color='red', alpha=0.8)
            ax1.add_artist(circle0)
            ax1.add_artist(circle1)
            # Add line from the biplot origin to the longest vector of all environments
            ax1.annotate("", xy=(x2, y2), xytext=(0,0), arrowprops=dict(arrowstyle= '->', color='r', lw=3, 
                                                                        ls='-', alpha=0.95),
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            # Add an infinite line
            plt.axline(xy1=(0, 0), xy2=(x2, y2), color='black', clip_on=True)
            plt.axline(xy1=(0, 0), xy2=(-y2, x2), color='black') # Add a perpendicular line
            # Add perpendicular line to each Genotype
            slope_xprj = (y2/x2)
            for idx in principal_G.index: 
                x = float(principal_G['factor1'][idx])
                y = float(principal_G['factor2'][idx])
                # slope of connecting line y = mx+b
                m = -np.reciprocal(slope_xprj)
                b = y-m*x
                # find intersecting point
                zx = b/(slope_xprj-m)
                zy = slope_xprj*zx
                # draw line
                plt.annotate('',(zx,zy),(x,y),arrowprops=dict(linewidth=1,arrowstyle='-',color='gray', ls='--', alpha=0.95))
            
            ax1.axis('equal')
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['factor1'].values[i]
                    ytext = principal_E['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_E['label'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE']) 
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['factor1'].values[i]
                    ytext = principal_G['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_G['label'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
                        
        elif (bptype==5):
            if (selG is None):
                print("Genotype not defined. please use 'selG' option to specific one")
                return
            principal_E = self.GE_data[self.GE_data['class']=='Environment']
            principal_G = self.GE_data[self.GE_data['class']=='Genotype']
            if (selG not in principal_G['label'].unique()):
                print("Selected genotype is not valid. Please choose correct one between {}"
                      .format(list(principal_G['label'].unique())))
                return
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data = principal_E , x = 'factor1',y = 'factor2' , hue=plot_params['hueE'], 
                                 marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'factor1',y = 'factor2' , hue=plot_params['hueG'],
                                 alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(principal_E['factor1'].min(), principal_G['factor1'].min())) - plot_params['limspan']
            minlim_y = int(min(principal_E['factor2'].min(), principal_G['factor2'].min())) - plot_params['limspan']
            maxlim_x = int(max(principal_E['factor1'].max(), principal_G['factor1'].max())) + plot_params['limspan']
            maxlim_y = int(max(principal_E['factor2'].max(), principal_G['factor2'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            #ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            #ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            #if (title==None): title='Average Environment Coordination: ranking genotypes based on mean performance'
            if (title==None): title='Ranking Genotypes Based on Performance in Environment {}\n'.format(selG) + self.trialname + ' [' +self.trait+ ']'
            plt.suptitle(title, fontsize=18, y=1.02)
            ax1.set_title("Exp.Var= {:.0f}%, Transform = 0, Scaling = {}, Centering = {}, SVP = {}"
                          .format(self.totalvar, self.params['scaling'], self.params['centering'], self.params['svp']),
                          fontsize=8.5)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.varexpl[0]), fontsize=12)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.varexpl[1]), fontsize=12)
            # Select an Environment
            x2 = float(principal_G['factor1'][principal_G['label']==selG])
            y2 = float(principal_G['factor2'][principal_G['label']==selG])
            circle0 = plt.Circle((x2, y2), radius=0.1, alpha=0.8, lw=1, edgecolor='red',fill=False)
            circle1 = plt.Circle((x2, y2), radius=0.01, color='red', alpha=0.8)
            ax1.add_artist(circle0)
            ax1.add_artist(circle1)
            # Add line from the biplot origin to the longest vector of all environments
            ax1.annotate("", xy=(x2, y2), xytext=(0,0), arrowprops=dict(arrowstyle= '->', color='r', lw=3, 
                                                                        ls='-', alpha=0.95),
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            # Add an infinite line
            plt.axline(xy1=(0, 0), xy2=(x2, y2), color='black', clip_on=True)
            plt.axline(xy1=(0, 0), xy2=(-y2, x2), color='black')# Add a perpendicular line
            # Add perpendicular line to each Genotype
            slope_xprj = (y2/x2)
            for idx in principal_E.index: 
                x = float(principal_E['factor1'][idx])
                y = float(principal_E['factor2'][idx])
                # slope of connecting line y = mx+b
                m = -np.reciprocal(slope_xprj)
                b = y-m*x
                # find intersecting point
                zx = b/(slope_xprj-m)
                zy = slope_xprj*zx
                # draw line
                plt.annotate('',(zx,zy),(x,y),arrowprops=dict(linewidth=1,arrowstyle='-',color='gray', ls='--', alpha=0.95))
            
            ax1.axis('equal')
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['factor1'].values[i]
                    ytext = principal_E['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_E['label'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE']) 
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['factor1'].values[i]
                    ytext = principal_G['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_G['label'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
                        
        elif (bptype==8):
            # ------------------------------------------------------
            # Ranking Genotypes Relative to the Ideal Genotype
            # ------------------------------------------------------
            principal_E = self.GE_data[self.GE_data['class']=='Environment']
            principal_G = self.GE_data[self.GE_data['class']=='Genotype']
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data = principal_E , x = 'factor1',y = 'factor2' , hue=plot_params['hueE'], 
                                 marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'factor1',y = 'factor2' , hue=plot_params['hueG'],
                                 alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(principal_E['factor1'].min(), principal_G['factor1'].min())) - plot_params['limspan']
            minlim_y = int(min(principal_E['factor2'].min(), principal_G['factor2'].min())) - plot_params['limspan']
            maxlim_x = int(max(principal_E['factor1'].max(), principal_G['factor1'].max())) + plot_params['limspan']
            maxlim_y = int(max(principal_E['factor2'].max(), principal_G['factor2'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            #ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            #ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            if (title==None): title='Ranking genotype based on both mean and stability \n'+ self.trialname + ' [' +self.trait+ ']'
            plt.suptitle(title, fontsize=18, y=1.02)
            ax1.set_title("Exp.Var= {:.0f}%, Transform = 0, Scaling = {}, Centering = {}, SVP = {}"
                          .format(self.totalvar, self.params['scaling'], self.params['centering'], self.params['svp']),
                          fontsize=8.5)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.varexpl[0]), fontsize=12)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.varexpl[1]), fontsize=12)
            
            # ----------------------------------------------
            # TODO: Improve this calculations
            # Ranking Genotypes Relative to the Ideal Genotype
            # An ideal genotype should have both high mean performance and high stability across environments.
            # ideal test environment
            x2 = np.mean(principal_E['factor1'])
            y2 = np.mean(principal_E['factor2'])
            xx2 = np.square(principal_G['factor1'])
            yy2 = np.square(principal_G['factor2'])
            # Ranking gens
            mod = max((xx2 + yy2)**0.5)
            xcoord = (mod**2 / (1 + y2**2 / x2**2))**0.5
            ycoord = (y2/x2) * xcoord
            circle2 = plt.Circle((xcoord, ycoord), radius=0.1, alpha=0.8, lw=1, edgecolor='red',fill=False)
            ax1.add_patch(circle2)
            circle0 = plt.Circle((x2, y2), radius=0.1, alpha=0.8, lw=1, edgecolor='red',fill=False)
            circle1 = plt.Circle((x2, y2), radius=0.01, color='red', alpha=0.8)
            #ax1.add_artist(circle0)
            #ax1.add_artist(circle1)
            # distance to the longest vector of all envs
            #dist = math.sqrt((y2)**2 + (x2)**2)
            # Add line from the biplot origin to the longest vector of all environments
            ax1.annotate("", xy=(xcoord, ycoord), xytext=(0,0), arrowprops=dict(arrowstyle= '->', color='b', 
                                                                                lw=2, ls='-', alpha=0.95),
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            # Add an infinite line
            plt.axline(xy1=(0, 0), xy2=(xcoord, ycoord), color='black')
            plt.axline(xy1=(0, 0), xy2=(-ycoord, xcoord), color='black') # Add a perpendicular line
            # Add circles
            for c in range(plot_params['ncircles']): #int(min(maxlim_x, maxlim_y))*
                if (c<=2):
                    circle = plt.Circle((xcoord, ycoord), c*plot_params['scircles'], alpha=plot_params['alphaCircle'], 
                                        lw=plot_params['lwcircle'], edgecolor=plot_params['colCircles'],
                                        fill=False, clip_on=True) #facecolor, color='b', 
                    ax1.add_artist(circle)
                else:
                    circle = plt.Circle((xcoord, ycoord), c*plot_params['scircles'], 
                                        alpha=plot_params['alphaCircle'], ls='--', 
                                        lw=plot_params['lwcircle'], edgecolor=plot_params['colCircles'],
                                        fill=False, clip_on=True) #facecolor, color='b', 
                    ax1.add_artist(circle)
                    
            ax1.axis('equal')
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['factor1'].values[i]
                    ytext = principal_E['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_E['label'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE'])
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['factor1'].values[i]
                    ytext = principal_G['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_G['label'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
        
        elif (bptype==9):
            data, polyg_ME, convhull_ME = self.getMegaEnvironments(lim_span=plot_params['lim_span'], 
                                                                   mxlim=plot_params['mxlim'])
            
            data['labelME'] = data['ME'].apply(lambda x: 'ME' + str(x))
            principal_E = data[data['class']=='Environment']
            principal_G = data[data['class']=='Genotype']
            if (len(principal_G['label'].unique()) <=2):
                print("It's not possible to carry out a convex hull for only 2 genotypes.")
                return
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data = principal_E , x = 'factor1',y = 'factor2' , hue=plot_params['hueE'], 
                                 marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'factor1',y = 'factor2' , hue=plot_params['hueG'],
                                 alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            if (title==None): title='Which-won-where \n'+ self.trialname + ' [' +self.trait+ ']'
            plt.suptitle(title, fontsize=18, y=1.02)
            ax1.set_title("Exp.Var= {:.0f}%, Transform = 0, Scaling = {}, Centering = {}, SVP = {}"
                          .format(self.totalvar, self.params['scaling'], self.params['centering'], self.params['svp']),
                          fontsize=8.5)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.varexpl[0]), fontsize=12)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.varexpl[1]), fontsize=12)
            
            # set limits
            extent = convhull_ME['extent']
            g1.set(xlim=(extent[0], extent[2]), ylim=(extent[1], extent[3]));
            #x_min3, y_min3, x_max3, y_max3 = convhull_ME['extent']
            #g2.set(xlim=(x_min3, x_max3), ylim=(y_min3, y_max3));

            # Draw Convexhull points and lines for the polygon
            if (plot_params['drawConvexhull'] is True):
                convhull = convhull_ME['convhull']
                for i, pt in enumerate(convhull):
                    plt.plot(pt[0], pt[1], marker=plot_params['convhull_marker'], 
                             markersize=plot_params['convhull_markersize'], 
                             markeredgecolor=plot_params['convhull_markeredgecolor'], 
                             markerfacecolor=plot_params['convhull_markerfacecolor'], zorder=0)
                    #if (plot_params['verticesText'] is True):
                    #    plt.text(pt[0]+0.05, pt[0]+0.05,'G{}'.format(i),fontsize=14, color='black', 
                    #             ha='center', va='bottom', clip_on=True)
    
                    if (i < len(convhull)):
                        p1 = convhull[i]
                        p2 = convhull[i-1]
                        l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], lw=plot_params['convhull_lw'], 
                                          ls=plot_params['convhull_ls'], color=plot_params['convhull_linecolor'], 
                                          axes=ax1, clip_on=True)
                        ax1.add_artist(l)
            # perpendicular lines or divisions
            if (plot_params['drawDivisions'] is True):
                lines = convhull_ME['lines']
                for pt in lines:
                    l = mlines.Line2D([0,pt[0]], [0,pt[1]], plot_params['convhull_perpline_lw'], 
                                      ls=plot_params['convhull_perpline_ls'], color=plot_params['convhull_perpline_color'], 
                                      axes=ax1, clip_on=True)
                    ax1.add_artist(l)

            # Draw ME Polygons
            if (plot_params['drawME'] is True):
                for pg in polyg_ME:
                    ply = polyg_ME[pg]
                    n_me = ply['ME']
                    poly_megaenv = ply['poly_megaenv']
                    shp = ply['shape']
                    #wkt = ply['wkt']
                    cx,cy = ply['centroid']
                    # Add center and texts
                    plt.plot(cx[0],cy[0], marker=plot_params['me_marker'], 
                             markersize=plot_params['me_markersize'],
                             markeredgecolor=plot_params['me_markeredgecolor'], 
                             markerfacecolor=plot_params['me_markerfacecolor'], clip_on=True)
                    plt.text(cx[0]+0.05, cy[0]+0.05,'ME{}'.format(n_me),fontsize=plot_params['me_fontsize'], 
                             color=plot_params['me_fontcolor'], ha=plot_params['me_ha'], va=plot_params['me_va'],
                             clip_on=True)
                    # Add polygon
                    if (plot_params['fillME'] is True):
                        #plt.plot(*wkt.exterior.xy) # Lines # wkt.exterior.coords # fill poly
                        try:
                            ME_polygon_clipped = plt.Polygon(shp, alpha=plot_params['me_alpha'], 
                                                             facecolor=plot_params['me_colors'][pg],
                                                             edgecolor=plot_params['me_colors'][pg], 
                                                             linewidth=plot_params['me_linewidth'], zorder=-1, 
                                                             closed=True, clip_on=True)
                            ax1.add_artist(ME_polygon_clipped)
                        except:
                            print("Problem drawing mega-environment: ME{}".format(n_me))
                            pass

            #ax1.axis('equal') # No usar
            
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['factor1'].values[i]
                    ytext = principal_E['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_E['label'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE']) 
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['factor1'].values[i]
                    ytext = principal_G['factor2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'
                             .format(principal_G['label'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
        
        # Legend
        if (plot_params['showLegend'] is True):
            def getLegend_HandlesLabels(ax, handout, lablout):
                handles, labels = ax.get_legend_handles_labels()
                for h,l in zip(handles,labels):
                    if ((plot_params['leg_gen'] is not None) and (l in plot_params['leg_gen'].keys())):
                        l = l +': '+ plot_params['leg_gen'][l]
                    if ((plot_params['leg_env'] is not None) and (l in plot_params['leg_env'].keys())):
                        l = l +': '+ plot_params['leg_env'][l]
                    if l not in lablout:
                        lablout.append(l)
                        handout.append(h)
                return handout, lablout
            
            if ((plot_params['leg_gen'] is not None) and (plot_params['leg_env'] is not None)):
                handout=[]
                lablout=[]
                handout, lablout = getLegend_HandlesLabels(ax1, handout, lablout)
                plt.legend(handout, lablout, bbox_to_anchor=(1.05, 1), loc=2, ncol=plot_params['ncol'], 
                           borderaxespad=0, fontsize=plot_params['leg_fontsize'])
            else:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=plot_params['ncol'], 
                           borderaxespad=0, fontsize=plot_params['leg_fontsize'])
        else:
            ax1.get_legend().remove()
                
                
        # Save in PDF
        if (saveFig is True and fmt=='pdf'):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path, 'IWIN_GGE_{}_{}.pdf'
                                     .format(title.replace(' ', '_'), hoy)), bbox_inches='tight', orientation='portrait', 
                        pad_inches=0.5, dpi=300) #papertype='a4', 

        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path,"IWIN_GGE_{}_{}.{}"
                                     .format(title.replace(' ', '_'), hoy, fmt)), bbox_inches='tight', 
                        facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                        pad_inches=0.5, dpi=300)

        if (showFig is True):
            if (saveFig is False):
                plt.tight_layout()
            plt.show()
        else:
            del fig
            plt.close();
        
