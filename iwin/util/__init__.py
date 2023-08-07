# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

#from . import *

import os, gc
import numpy as np
import pandas as pd
from datetime import date, datetime
import pandas as pd
#pd.set_option('display.max_columns', None)

import seaborn as sns
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from matplotlib.path import Path
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
#from matplotlib.collections import PatchCollection
#from matplotlib.patches import Polygon

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

from shapely.geometry import shape, box, LineString, Point, Polygon
from shapely.ops import split
from pyhull.convex_hull import ConvexHull
import itertools
import math
from collections import Counter
#import scipy
from scipy.stats import norm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.model_selection import train_test_split
import hdbscan

from IPython.display import display, HTML

def is_number(s):
    try:
        float(s) # for int, long and float
    except ValueError:
        try:
            complex(s) # for complex
        except ValueError:
            return False
    return True

def formatInt(f):
    try:
        f = int(f)
    except:
        f = np.nan
    return f

def toInt(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return np.nan    # leave unchanged
    
#define function to swap columns
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

def getDOY(d):
    ''' Get Day of the Year '''
    day_of_year = datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday
    return day_of_year

# Convert DOY to full date
def convertDOYToDate(year, sDOY):
    try:
        sd = pd.Timestamp('{}-01-01'.format(int(year)))
        sdoy = pd.DateOffset(days=int(sDOY))
        return sd + sdoy
    except (ValueError, TypeError):
        return np.nan

def getDate(str_d):
    try:
        return pd.Timestamp(str_d)
    except (ValueError, TypeError):
        return np.nan    # leave unchanged
    
def getBackEmerDate(YearofEme, DOYofEme):
    try:
        ed = pd.Timestamp('{}-01-01'.format(int(YearofEme)))
        edoy = pd.DateOffset(days=int(DOYofEme))
        return ed + edoy
    except (ValueError, TypeError):
        return np.nan    # leave unchanged
    
def getHeadingDate(sowingdate, daysHead):
    try:
        return sowingdate + pd.DateOffset(days=int(daysHead))
    except (ValueError, TypeError):
        return np.nan

def getMaturityDate(sowingdate, daysMat):
    try:
        return sowingdate + pd.DateOffset(days=int(daysMat))
    except (ValueError, TypeError):
        return np.nan    
    
    
def getPhenologyDateAfterSowing(sowingdate, daysaftersowing):
    try:
        return sowingdate + pd.DateOffset(days=int(daysaftersowing))
    except (ValueError, TypeError):
        return np.nan    
    
def getObsDaysHM(matu, head):
    try:
        return int((matu - head).days)
    except (ValueError, TypeError):
        return np.nan    

def sub_block(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return -99    # leave unchanged

#
    
# ---------------------------------------------
# Find nearest value or index 
# to a user define value from array
# ---------------------------------------------
def find_nearest_value(array, value):
    '''
        Find nearest value to a user define value from array
        
        :params array: Array of values
        :params value: value to find into the array
        
        :return: a number with the nearest value found
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_index(array, values):
    '''
        Find nearest index to a user define value from array
        
        :params array: Array of values
        :params values: value to find into the array
        
        :return: a number with the nearest index found
    '''
    values = np.atleast_1d(values)
    indices = np.abs(np.int64(np.subtract.outer(array, values))).argmin(0)
    out = array[indices]
    return out

def find_nearest(array, value):
    '''
        Find nearest index and value to a user define value from array
        
        :params array: Array of values
        :params value: value to find into the array
        
        :return: a number with the nearest value found
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def getNearestRow(arr=None, value=0, verbose=False):
    if (arr is None or len(arr)<=0):
        print("Please check out your input values...")
        return
    #n_value = find_nearest_value(arr, value)
    #n_index = find_nearest_index(arr, value)[0]
    n_index, n_value = find_nearest(arr, value)
    if (verbose is True):
        #print("Temperature adjusted heading to maturity days:")
        print("Nearest value: {}".format(n_value))
        print("Nearest Index: {}".format(n_index))
    return n_index, n_value

def _getNearestRow(df, fld=None, value=0, verbose=False):
    if (fld is None or len(df)<=0):
        print("Please check out your input values...")
        return
    vl = df[[fld]].dropna()
    n_value = find_nearest_GDD_value(vl.values, value)[0];
    n_index = vl.index[ find_nearest_GDD_index(vl.values, value)[0]][0]
    if (verbose is True):
        #print("Temperature adjusted heading to maturity days:")
        print("Nearest value: {}".format(n_value))
        print("Nearest Index: {}".format(n_index))
    return n_index, n_value


def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__)

def rankdata(a):
    n = len(a)
    ivec=rank_simple(a)
    svec=[a[rank] for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0]*n
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray


# Function to find foot of perpendicular
# from a point in 2 D plane to a Line
def findFoot(a, b, c, x1, y1):
    temp = (-1 * (a * x1 + b * y1 + c) // (a * a + b * b))
    x = temp * a + x1
    y = temp * b + y1
    return (x, y)

def perpendicular(x1, y1, x2, y2, x3, y3):
    k = ((y2-y1) * (x3-x1) - (x2-x1) * (y3-y1)) / ((y2-y1)**2 + (x2-x1)**2)
    x4 = x3 - k * (y2-y1)
    y4 = y3 + k * (x2-x1)
    return (x4,y4)

# ---------------------------
# STATS
# ---------------------------
def getScores(df, fld1=None, fld2=None):
    ''' Get stats for model results '''
    if (df is None):
        print("Input data not valid")
        return
    if (fld1 is None or fld2 is None):
        print("Variable are not valid")
        return
    df_notnull = df[[fld1, fld2]].dropna()
    y_test = df_notnull[fld1].astype('double') #float16
    y_predicted = df_notnull[fld2].astype('double') #float16
    accuracy = getAccuracy(y_test, y_predicted)
    r2score = round(r2_score(y_test.values, y_predicted.values), 2)
    rmse = mean_squared_error(y_test.values, y_predicted.values, squared=False)
    n_rmse = (rmse / y_test.values.mean()) * 100
    d1 = ((y_test.values - y_predicted.values).astype('double') ** 2).sum()
    d2 = ((np.abs(y_predicted.values - y_test.values.mean()) + np.abs(y_test.values - y_test.values.mean())).astype('double') ** 2).sum()
    d_index = round(1 - (d1 / d2) ,3)
    return r2score, rmse, n_rmse, d_index, accuracy

# Calculate accuracy and precision for each year in a nursery site
def getAccuracy(y_true, y_predicted):
    '''
        Calculate accuracy and precision for each year in a nursery site
    '''
    mape = np.mean(np.abs((y_true - y_predicted)/y_true))*100
    if (mape<=100):
        accuracy = np.round((100 - mape), 2)
    else:
        mape = np.mean(np.abs((y_predicted - y_true)/ y_predicted))*100
        accuracy = np.round((100 - mape), 2)
    return accuracy


def missing_values_table(df):
    '''
        Look for missing values in a dataset
        
    '''
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\nThere are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    return mis_val_table_ren_columns

def missing_zero_values_table(df):
    '''
    
    '''
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
    mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
    mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[ mz_table.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\nThere are " + str(mz_table.shape[0]) + " columns that have missing values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
    return mz_table



def missingData(df_GE=None, title='IWIN GxE\nMissing data', dirname='./', fmt='pdf', 
                showFig=True, saveFig=False, verbose=True):
    ''' Looking for missing values
        Check for all the variables in an arrays or pandas DF and visualize the missing data
        
    '''
    if (df_GE is None):
        print("Matrix input not valid")
        return
    
    df = df_GE.copy()
    # Get columns with missing values
    cols_missing_values = [x for x in dict(df.isna().any()) if (df[x].isna().any() == True)]
    df = df[cols_missing_values] # Filter only columns with missing values
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table.rename( columns = {0 : 'Missing Values', 1 : '% of Total Values'}, inplace=True)
    mis_val_table = mis_val_table[ mis_val_table.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    mis_val_table = mis_val_table.reset_index().rename(columns={'index':'Trait'})
    if (verbose is True):
        print ("There are " + str(mis_val_table.shape[0]) + " columns that have missing values.")
    
    if (showFig is True):
        xlabel = 'Traits'
        ylabel = '% of total values'
        fig, (ax1) = plt.subplots(figsize=(10,6))
        #fig.subplots_adjust(right=0.55)
        g1 = sns.barplot(x='Trait', y='% of Total Values', data=mis_val_table, alpha=0.75, color="gray", lw=0, ax=ax1);
        g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax1.set_axisbelow(True)
        ax1.set_title(title, fontsize=18)
        ax1.set_xlabel(xlabel, fontsize=14)
        ax1.set_ylabel(ylabel, fontsize=14)
        ax1.tick_params(labelsize=10)
        ax1.tick_params(axis='x', rotation=90)
        #plt.xticks(rotation=90)
        
        # Save in PDF
        if (saveFig is True and fmt=='pdf'):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path,"{}_{}.pdf".format(title.replace(' ',''), hoy)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', pad_inches=0.5, dpi=300)

        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            hoy = datetime.now().strftime('%Y%m%d')
            #figures_path = os.path.join(config['RESULTS_PATH'] , '{}_{}'.format(dirname, hoy) )
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path,"{}_{}.{}".format(title.replace(' ',''), hoy, fmt)), 
                        bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=300)

        if (showFig is True):
            if (saveFig is False):
                plt.tight_layout()
            fig.show()
        else:
            del fig
            plt.close();
    
    del df
    _ = gc.collect()
    
    return cols_missing_values, mis_val_table


# ------------------------------
# Clustering

# Defining a function to find the optimal number of K-Means' n_clusters parameter
# Note that we are using a dataset that contains only numeric features. This is
# because K-Means calculates the mean of each cluster to update centroids.
def findOptimalNClustersKMeans(transformed_df, range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]):
    # Number of clusters to search for and silhouette_scores list
    silhouette_scores = []
    # Testing n_clusters options
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=7)
        cluster_labels = kmeans.fit_predict(transformed_df)
        # Evaluating clusters created by KMeans
        silhouette_avg = silhouette_score(transformed_df, cluster_labels)
        print("K-Means: for n_clusters =", n_clusters, ", the average silhouette_score is", silhouette_avg)
        # Appending iteration's avg silhouette score to scores list
        silhouette_scores.append(silhouette_avg)
        
    return range_n_clusters, silhouette_scores

# Defining a function to find the optimal number of A.C.'s n_clusters parameter:
def findOptimalNClustersAC(transformed_df, range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]):
    silhouette_scores = []
    # Testing n_clusters options
    for n_clusters in range_n_clusters:
        ac = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = ac.fit_predict(transformed_df)
        # Evaluating clusters created by Agglomerative Clustering
        silhouette_avg = silhouette_score(transformed_df, cluster_labels)
        print("Agglomerative Clustering: for n_clusters =", n_clusters, ", the average silhouette_score is", silhouette_avg)
        # Appending iteration's avg silhouette score to scores list
        silhouette_scores.append(silhouette_avg)
        
    return range_n_clusters, silhouette_scores


def plot_outliers(data, cols, target_col, outlier_col):
    # linear predictive features:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    mapdict = {0: [0,0], 1:[0,1], 2:[1,0], 3:[1,1]}

    for n, c in enumerate(cols):
        axs[mapdict[n][0], mapdict[n][1]].scatter(data[c], data[target_col], c=data[outlier_col], cmap='plasma', vmin=0, vmax=1.5)
        axs[mapdict[n][0], mapdict[n][1]].set_xlabel(c)

    axs[0,0].set_ylabel(target_col)
    axs[1,0].set_ylabel(target_col)
    plt.show()

def plot_k_distance_graph(df):
    nn = NearestNeighbors(n_neighbors=2)
    nbrs = nn.fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.figure(figsize=(12,6))
    plt.plot(distances)
    plt.title('K-distance graph',fontsize=20)
    plt.xlabel('Data points sorted by distance',fontsize=14)
    plt.ylabel('Epsilon',fontsize=14)
    plt.show()

#
def run_dbscan(df, data, cols, target_col='GID', eps=8000, min_samples=6, scaled=''):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    print(dbscan)
    dbs = dbscan.fit(df)
    clusters = dbs.labels_
    data[f'cluster{scaled}'] = clusters
    data[f'no_cluster{scaled}'] = np.where(data[f'cluster{scaled}'] == -1, 1, 0)
    print(len(data[data[f'cluster{scaled}'] == -1]))
    print(data[f'cluster{scaled}'].unique())
    return data
    
def findNearest_point(arr=None, pnt=None, verbose=False):
    '''
        Select the nearest point
        
    '''
    A = np.array(arr)
    p = np.array(pnt) #(p[0], p[1]))
    distances = np.linalg.norm(A-meanG, axis=1)
    min_index = np.argmin(distances)
    closestPoint = A[min_index]
    if (verbose is True):
        print(f"The closest point is {closestPoint}, at a distance of {distances[min_index]}")
    return closestPoint


def clockwiseangle_and_distance(point, origin=(0,0), refvec=[0,1]):
    #origin = [0, 0]
    #refvec = [0, 1]
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

def getPolygIntersectionExtentLine(polygon, line):
    minx, miny, maxx, maxy = polygon.bounds
    bounding_box = box(minx, miny, maxx, maxy)
    a, b = line.boundary
    if a.x == b.x:  # vertical line
        extended_line = LineString([(a.x, miny), (a.x, maxy)])
    elif a.y == b.y:  # horizonthal line
        extended_line = LineString([(minx, a.y), (maxx, a.y)])
    else:
        # linear equation: y = k*x + m
        k = (b.y - a.y) / (b.x - a.x)
        m = a.y - k * a.x
        y0 = k * minx + m
        y1 = k * maxx + m
        x0 = (miny - m) / k
        x1 = (maxy - m) / k
        points_on_boundary_lines = [Point(minx, y0), Point(maxx, y1), 
                                    Point(x0, miny), Point(x1, maxy)]
        points_sorted_by_distance = sorted(points_on_boundary_lines, key=bounding_box.distance)
        extended_line = LineString(points_sorted_by_distance[:2])

    intersection = extended_line.intersection(polygon)
    print(intersection)
    parts = split(polygon, extended_line)
    
    return intersection, parts

def perpendicular(x1, y1, x2, y2, x3, y3):
    k = ((y2-y1) * (x3-x1) - (x2-x1) * (y3-y1)) / ((y2-y1)**2 + (x2-x1)**2)
    x4 = x3 - k * (y2-y1)
    y4 = y3 + k * (x2-x1)
    return (x4,y4)

def boundingBox(point_list):
    x_min = min(point[0] for point in point_list)
    x_max = max(point[0] for point in point_list)
    y_min = min(point[1] for point in point_list)
    y_max = max(point[1] for point in point_list)
    
    p1 = (x_min, y_min)
    p2 = (x_min, y_max)
    p3 = (x_max, y_max)
    p4 = (x_max, y_min)
    vert = [p1,p2,p3,p4]
    
    return x_min, x_max, y_min, y_max, vert

def getExtrapoledLine(p1,p2, EXTRAPOL_RATIO=5, rtn_coods=False):
    'Creates a line extrapoled in p1->p2 direction'
    #EXTRAPOL_RATIO = 10
    a = p1
    b = (p1[0]+EXTRAPOL_RATIO*(p2[0]-p1[0]), p1[1]+EXTRAPOL_RATIO*(p2[1]-p1[1]) )
    if (rtn_coods is True):
        l = [a,b]
    else:
        l = LineString([a,b])
    return l

def getRectangle(bx):
    x_min, x_max, y_min, y_max, vert = boundingBox(bx)
    #print(x_min, x_max, y_min, y_max )
    p1 = vert[0] #(x_min, y_min)
    p2 = vert[1] #(x_min, y_max)
    p3 = vert[2] #(x_max, y_max)
    p4 = vert[3] #(x_max, y_min)
    #plt.plot(*zip(p1,p2)) #plt.plot(*zip(p2,p3))
    return p1, p2, p3, p4, p1

def getInters(c, le):
    #print(c, le)
    try:
        pnt_int_circ = c.intersection(le)
        #print("Intersection:",pnt_int_circ)
        x,y = (*pnt_int_circ.xy,)
        #pt_intercep = (x,y)
    except:
        x,y = ([0],[0])
    return (x,y)

#
def custom_rolling(d, w=2, op='min'):
    ''' Duplica el arreglo para completar todos las columnas 
        con el numero de datos seleccionado para la ventana 
    '''
    resp = np.nan
    n = len(d) #d.shape[1]
    #print(n, n+int(n/2))
    arr = list(np.array(d).flatten())*2
    #print(arr)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=w) #, num_values=10, closed=True)
    if (op=='min'):
        arr = pd.DataFrame([arr]).rolling(window=w, min_periods=1, center=False, axis=1).min().iloc[:,0:n+int(n/2)+1].round(1)
        resp = np.min(arr.to_numpy())
    elif (op=='mean'):
        arr = pd.DataFrame([arr]).rolling(window=w, min_periods=1, center=False, axis=1).mean().iloc[:,0:n+int(n/2)+1].round(1)
        resp = np.mean(arr.to_numpy())
    elif (op=='max'):
        arr = pd.DataFrame([arr]).rolling(window=w, min_periods=1, center=False, axis=1).max().iloc[:,0:n+int(n/2)+1].round(1)
        resp = np.max(arr.to_numpy())
    if (op=='coldest'):
        arr = pd.DataFrame([arr]).rolling(window=w, min_periods=1, center=False, axis=1).min().iloc[:,0:n+int(n/2)+1].round(1)
        resp = np.mean(arr.to_numpy())
    elif (op=='warmest'):
        arr = pd.DataFrame([arr]).rolling(window=w, min_periods=1, center=False, axis=1).max().iloc[:,0:n+int(n/2)+1].round(1)
        resp = np.mean(arr.to_numpy())
    elif (op=='sum'):
        arr = pd.DataFrame([arr]).rolling(window=w, min_periods=1, center=False, axis=1).sum().iloc[:,0:n+int(n/2)+1].round(1)
        resp = np.sum(arr.to_numpy())
    elif (op=='wettest'):
        arr = pd.DataFrame([arr]).rolling(window=w, min_periods=1, center=False, axis=1).sum().iloc[:,0:n+int(n/2)+1].round(1)
        resp = np.max(arr.to_numpy())
    elif (op=='driest'):
        arr = pd.DataFrame([arr]).rolling(window=w, min_periods=1, center=False, axis=1).sum().iloc[:,0:n+int(n/2)+1].round(1)
        resp = np.min(arr.to_numpy())
    return round(resp,1) #, arr

# Mega-Environments
def assignME(df):
    ''' Mega-environments 
        
        Al momento solo ME4
    '''
    df['Mega-Env'] = 'ME4'
    # ME4C - coolest quarter mean min temp >3oC <16oC; wetest quarter precipitaion >100 mm <400 mm.
    df.loc[(
        ((df['Precip_WettestPeriod'] > 100.0) & (df['Precip_WettestPeriod'] < 400.0)) & 
        ((df['MinTemp_ColdestQuarter'] > 3.0) & (df['MinTemp_ColdestQuarter'] < 16.0)) ), 'Mega-Env'] = 'ME4C'
    # ME4A - coolest quarter mean min temp >3oC <11oC; wetest quarter precipitaion >100 mm <400 mm.
    df.loc[(
        ((df['Precip_WettestPeriod'] > 100.0) & (df['Precip_WettestPeriod'] < 400.0)) & 
        ((df['MinTemp_ColdestQuarter'] > 3.0) & (df['MinTemp_ColdestQuarter'] < 11.0)) ), 'Mega-Env'] = 'ME4A'
    # ME4B - coolest quarter mean min temp >3oC <11oC; wetest quarter precipitaion >200 mm <500 mm
    df.loc[(
        ((df['Precip_WettestPeriod'] > 200.0) & (df['Precip_WettestPeriod'] < 500.0)) & 
        ((df['MinTemp_ColdestQuarter'] > 3.0) & (df['MinTemp_ColdestQuarter'] < 11.0)) ), 'Mega-Env'] = 'ME4B'
    
    return df


def normalize_Yield(yld, perc):
    ''' Use 95th percentile of yield (top 5%) of each site-year as a reference and 
        express the yield of each GID within a site-year in percent of the top yield.
    '''
    normYield = np.nan
    try:
        normYield = round((yld / perc) * 100, 2)
        if (normYield > 100.0):
            normYield = 100.0
    except Exception as err:
        normYield = np.nan
    return normYield

def KDE_hist_plot(df):
    for col in df.columns:
        df2 = df.dropna(subset=[col])  # Drop null values
        n_bins = 50
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,5))
        #histogram
        n, bins, patches = axes[1].hist(df2[col], n_bins, density=True, alpha=.1, edgecolor='black' )
        #data = pd.Series(s)
        mu = df2[col].mean()
        sigma = df2[col].std()
        pdf = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/(2*sigma**2))
        median, q1, q3 = np.percentile(df2[col], 50), np.percentile(df2[col], 25), np.percentile(df2[col], 75)

        #probability density function
        axes[1].plot(bins, pdf, color='orange', alpha=.6)

        #axes[1].figsize=(10,20)
        #fill from Q1-1.5*IQR to Q1 and Q3 to Q3+1.5*IQR
        iqr = 1.5 * (q3-q1)
        x1 = np.linspace(q1 - iqr, q1)
        x2 = np.linspace(q3, q3 + iqr)
        pdf1 = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x1-mu)**2/(2*sigma**2))
        pdf2 = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x2-mu)**2/(2*sigma**2))
        axes[1].fill_between(x1, pdf1, 0, alpha=.6, color='orange')
        axes[1].fill_between(x2, pdf2, 0, alpha=.6, color='orange')

        #add text to bottom graph.
        axes[1].annotate("{:.1f}%".format(100*(norm(mu, sigma).cdf(q1)    -norm(mu, sigma).cdf(q1-iqr))), 
                         xy=(q1-iqr/2, 0), ha='center')
        axes[1].annotate("{:.1f}%".format(100*(norm(mu, sigma).cdf(q3)    -norm(mu, sigma).cdf(q1)    )), 
                         xy=(median  , 0), ha='center')
        axes[1].annotate("{:.1f}%".format(100*(norm(mu, sigma).cdf(q3+iqr)-norm(mu, sigma).cdf(q3)    )), 
                         xy=(q3+iqr/2, 0), ha='center')
        axes[1].annotate('q1', xy=(q1, norm(mu, sigma).pdf(q1)), ha='center')
        axes[1].annotate('q3', xy=(q3, norm(mu, sigma).pdf(q3)), ha='center')

        #dashed lines
        plt.axvline(df2[col].quantile(0),color='b', linestyle='-.')
        plt.axvline(df2[col].quantile(0.25),color='g', linestyle='--')
        plt.axvline(df2[col].quantile(0.50),color='g', linestyle='--')
        plt.axvline(df2[col].quantile(0.75),color='b', linestyle='--')
        plt.axvline(df2[col].quantile(1),color='r', linestyle='-.')

        axes[1].set_ylabel('Probability Density')

        #top boxplot
        axes[0].boxplot(df2[col], 0, 'gD', vert=False)
        axes[0].axvline(median, color='orange', alpha=.6, linewidth=.5)
        axes[0].axis('off')
        
        axes[1].set_xlabel(col)
        
#
def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy.
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

# ----------------------------------------------
# Functions to select GIDs by ...
# eg. Matthew 
# ----------------------------------------------

def idenfifyTargetGIDs_byNurseryYear(df_filtered, nursery='ESWYT', path_to_save_results='./', drawFigures=False, verbose=True):
    '''
        Extract the targeted GIDs from a pre-processed IWIN dataset
        
        Classify trials in:
         - Above Avg in Low Yielding Envs - Below Avg in High Yielding Envs
         - Below Avg in Low Yielding Envs - Above Avg in High Yielding Envs
         - Highest/Lowest norm. Yield GIDs
         - Above Avg. Low Temperature GIDs
         - Above Avg. High Temperature GIDs
         - Support or not warmer temperature
         - Warmest Environments
         
        Return 3 tables with linear model parameters
    '''
    df = df_filtered.copy()
    df['slope'] = np.nan
    df['intercept'] = np.nan
    df['targetClass'] = ''
    df['minTemperatureEnvs'] = ''
    df['maxTemperatureEnvs'] = ''
    df_final = pd.DataFrame()
    list_GIDs = []
    GIDs_inTarget_1 = []
    GIDs_inTarget_lowTemperature = []
    GIDs_inTarget_highTemperature = []
    GIDs_LowestHighest_NormilizedYield = []
    for nyr in df['Nursery_Yr'].unique(): #[:1]:
        _mask = (df['Nursery_Yr']==nyr)
        df2 = df[_mask].sort_values(by='Country')
        df2['Country'] = df2['Country'].astype(str)

        if (drawFigures is True):
            fig = plt.figure(figsize=(14, 18), constrained_layout=True)
            #fig.subplots_adjust(wspace=0.02, hspace=0.05)
            ax1 = plt.subplot2grid((3, 2), (0, 0))
            ax2 = plt.subplot2grid((3, 2), (0, 1))
            ax3 = plt.subplot2grid((3, 2), (1, 0))
            ax4 = plt.subplot2grid((3, 2), (1, 1))
            ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2) #, rowspan=2)

            # --------------------
            # Figure 1 
            # --------------------
            # color
            #g1 = sns.scatterplot(x='Quantiles95(GRAIN_YIELD_BLUEs)', y='normYieldBLUE95Perc', data=df2, 
            #                     hue='Country',style='Country', marker='o', s=20, alpha=0.85, ax=ax1, label='Country');
            g1 = sns.scatterplot(x='Quantiles95(GRAIN_YIELD_BLUEs)', y='normYieldBLUE95Perc', data=df2, 
                             color='#cccccc', style='Country', marker='+', s=30, alpha=0.45, ax=ax1, label='Country')
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            # Add hztal and vertical lines
            ax1.axhline(df2['normYieldBLUE95Perc'].mean(), ls='--', c='#444444', zorder=0, linewidth=1, label="Avg. Norm Mean Yield")
            ax1.axvline(df2['Quantiles95(GRAIN_YIELD_BLUEs)' ].mean(), ls='--', c='#444444', zorder=0, linewidth=1, 
                        label="Avg. Quantiles95 Mean Yield")

        # Add linear regression
        # Should be by GID
        x = df2['Quantiles95(GRAIN_YIELD_BLUEs)'].to_numpy()
        y = df2['normYieldBLUE95Perc'].to_numpy()
        # determine best fit line
        par = np.polyfit(x, y, 1, full=True)
        pend=par[0][0]
        intercept=par[0][1]
        if (verbose is True):
            print("{} Nursery Year: {} - y = {:.7f}x + {:.7f}".format(nursery, nyr, pend, intercept))
        y_predicted = [pend*i + intercept  for i in x]
        if (drawFigures is True):
            l1 = sns.lineplot(x,y_predicted, color='black', ax=ax1, ls='-.', lw=1.25, label='fit')

        # Add linear regression by GID
        list_GIDs = []
        for gid in df2['GID'].unique():
            x = df2[(df2['GID']==gid)]['Quantiles95(GRAIN_YIELD_BLUEs)'].to_numpy()
            y = df2[(df2['GID']==gid)]['normYieldBLUE95Perc'].to_numpy()
            # determine best fit line
            par = np.polyfit(x, y, 1, full=True)
            pend=par[0][0]
            intercept=par[0][1]
            y_predicted = [pend*i + intercept  for i in x]
            df.loc[(df['GID']==gid), 'slope'] = pend
            df.loc[(df['GID']==gid), 'intercept'] = intercept
            if (pend < 0):
                df.loc[((df['Nursery_Yr']==nyr) & (df['GID']==gid)), 'targetClass'] = 'Above Avg in Low Yielding Envs - Below Avg in High Yielding Envs'
                list_GIDs.append(gid)
                GIDs_inTarget_1.append({"Nyr":nyr, "GID":gid, "slope":pend, "intercept":intercept})
                #print("GID: {} - y = {:.7f}x + {:.7f}".format(gid, pend, intercept))
                if (drawFigures is True):
                    l1 = sns.lineplot(x,y_predicted, color='red', ax=ax1, ls='-', lw=1.25, label='GID: {}'.format(gid))
            else:
                df.loc[((df['Nursery_Yr']==nyr) & (df['GID']==gid)), 'targetClass'] = 'Below Avg in Low Yielding Envs - Above Avg in High Yielding Envs'
                if (drawFigures is True):
                    l1 = sns.lineplot(x,y_predicted, color='lightblue', ax=ax1, ls='-', lw=0.25) #, label='{} line'.format(gid))

        # Add selected target GIDs
        df3 = df2[df2['GID'].isin(list_GIDs)].reset_index(drop=True)
        df3['GID'] = df3['GID'].astype(str)
        if (drawFigures is True):
            g2 = sns.scatterplot(x='Quantiles95(GRAIN_YIELD_BLUEs)', y='normYieldBLUE95Perc', data=df3, 
                                 color='red', hue='GID', style='GID', s=40, alpha=0.85, ax=ax1, label='Target GIDs')
            #ax1.set_title('NURSERY - Nursery Year {}'.format(nyr), fontsize=18)
            ax1.set_xlabel('Quantiles95 (t/ha)',fontsize=15)
            ax1.set_ylabel('normYieldBLUE95Perc (%)',fontsize=15)
            ax1.get_legend().remove()

        # --------------------
        # Figure 2 
        # --------------------
        df2 = df2.sort_values(by='LocOccCntry')
        if (drawFigures is True):
            g3 = sns.scatterplot(x='LocOccCntry', y='normYieldBLUE95Perc', data=df2, 
                             color='#cccccc', style='Country', marker='+', s=30, alpha=0.45, ax=ax2, label='Country')
            g3.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g3.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax2.set_axisbelow(True)
            if (len(df2['LocOccCntry'].unique())>50):
                ax2.tick_params(axis='x', rotation=90, labelsize=6.5)
            else:
                ax2.tick_params(axis='x', rotation=90, labelsize=8)
            g4 = sns.scatterplot(x='LocOccCntry', y='normYieldBLUE95Perc', data=df3, 
                                 color='red', hue='GID', style='GID', s=40, alpha=0.85, ax=ax2, label='Target GIDs')
        
        
        avgYield = df2['normYieldBLUE95Perc'].mean()
        # Classsify High/Low normalized Yield
        df3['targetClass'] = df3['normYieldBLUE95Perc'].apply(lambda x: 'HY' if x >= avgYield else 'LY')
        df3['GID'] = df3['GID'].astype(str)
        if (drawFigures is True):
            # Add hztal and vertical lines
            ax2.axhline(avgYield, ls='--', c='#444444', zorder=0, linewidth=1, label="Norm Mean Yield")
            g5 = sns.scatterplot(x='LocOccCntry', y='normYieldBLUE95Perc', data=df3, 
                                 color='red', hue='GID', style='GID', s=40, alpha=0.85, ax=ax2) #, label='High/Low Yield Environments')

        df4 = df2.groupby(['GID'], as_index=False).agg({'normYieldBLUE95Perc':'mean'}).sort_values(['normYieldBLUE95Perc'],
                                                                                                   ascending=False).reset_index(drop=True)
        ymin, ymax = df4.loc[(df4['normYieldBLUE95Perc'].idxmin(), 'GID')], df4.loc[(df4['normYieldBLUE95Perc'].idxmax(), 'GID')]
        # Save Highest/Lowest norm. Yield GIDs
        GIDs_LowestHighest_NormilizedYield.append({"nyr":nyr, "lowest":ymin, "highest":ymax})
        df5 = df2[df2['GID'].isin([ymin,ymax])]
        df5['HLYieldGIDs'] = ''
        df5.loc[(df5['GID']==ymin), 'HLYieldGIDs'] = 'Lowest avg. normilized yield ({})'.format(ymin)
        df5.loc[(df5['GID']==ymax), 'HLYieldGIDs'] = 'Highest avg. normilized yield ({})'.format(ymax)
        df5['GID'] = df5['GID'].astype(str)
        if (drawFigures is True):
            g6 = sns.scatterplot(x='LocOccCntry', y='normYieldBLUE95Perc', data=df5, 
                                 color='red', style='HLYieldGIDs', s=80, alpha=0.85, ax=ax2, label='Highest/Lowest norm. Yield GIDs')
            ax2.set_xlabel('Loc-Occ-Country',fontsize=15)
            ax2.set_ylabel('',fontsize=15)
            ax2.get_legend().remove()

        # --------------------
        # Figure 3
        # --------------------
        if (drawFigures is True):
            g7 = sns.scatterplot(x='Season_TMIN_mean', y='normYieldBLUE95Perc', data=df2, 
                             color='#cccccc', style='Country', marker='+', s=30, alpha=0.45, ax=ax3, label='Country')
            g7.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g7.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax3.set_axisbelow(True)
            # Add hztal and vertical lines
            ax3.axhline(df2['normYieldBLUE95Perc'].mean(), ls='--', c='#444444', zorder=0, linewidth=1, label="Avg. Norm Mean Yield")
            ax3.axvline(df2['Season_TMIN_mean' ].mean(), ls='--', c='#444444', zorder=0, linewidth=1, label="Avg. Season_TMIN_mean")

            #g7b = sns.scatterplot(x='Season_TMIN_mean', y='normYieldBLUE95Perc', data=df3, 
            #                     color='red', hue='GID', style='GID', s=40, alpha=0.85, ax=ax3, label='Target GIDs')

        # Add linear regression by GID
        list_GIDs_lowTemperature = []
        for gid in df3['GID'].unique():
            x = df3[(df3['GID']==gid)]['Season_TMIN_mean'].to_numpy()
            y = df3[(df3['GID']==gid)]['normYieldBLUE95Perc'].to_numpy()
            # determine best fit line
            par = np.polyfit(x, y, 1, full=True)
            pend=par[0][0]
            intercept=par[0][1]
            y_predicted = [pend*i + intercept  for i in x]
            if (pend > 0):
                df.loc[((df['Nursery_Yr']==nyr) & (df['GID']==gid)), 'minTemperatureEnvs'] = 'Support warmer temperature'
                list_GIDs_lowTemperature.append(gid)
                GIDs_inTarget_lowTemperature.append({"Nyr":nyr, "GID":gid, "slope":pend, "intercept":intercept})
                if (drawFigures is True):
                    l3 = sns.lineplot(x,y_predicted, color='blue', ax=ax3, ls='-', lw=0.9, label='Low Temp. GID: {}'.format(gid))
            else:
                df.loc[((df['Nursery_Yr']==nyr) & (df['GID']==gid)), 'minTemperatureEnvs'] = 'Not support warmer temperature' #'Support cooler temperature'
                if (drawFigures is True):
                    l3 = sns.lineplot(x,y_predicted, color='lightblue', ax=ax3, ls='-', lw=0.25) #, label='{} line'.format(gid))

        df.loc[( (df['Nursery_Yr']==nyr) & (df['GID'].isin(list_GIDs_lowTemperature)) ), 'minTemperatureEnvs'] = 'Support warmer temperature'
        df6 = df3[df3['GID'].isin(list_GIDs_lowTemperature)].reset_index(drop=True)
        df6['GID'] = df6['GID'].astype(str)
        if (drawFigures is True):
            g7c = sns.scatterplot(x='Season_TMIN_mean', y='normYieldBLUE95Perc', data=df6, 
                                  color='blue', hue='GID', style='GID', s=40, alpha=0.85, ax=ax3, label='Above Avg. Low Temperature GIDs')
            ax3.set_xlabel('Season_TMIN_mean',fontsize=15)
            ax3.set_ylabel('normYieldBLUE95Perc (%)',fontsize=15)
            ax3.get_legend().remove()

        # --------------------
        # Figure 4
        # --------------------
        fld_fg4 = 'Season_TMAX_mean'
        if (drawFigures is True):
            g8 = sns.scatterplot(x=fld_fg4, y='normYieldBLUE95Perc', data=df2, 
                             color='#cccccc', style='Country', marker='+', s=30, alpha=0.45, ax=ax4, label='Country')
            g8.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g8.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax4.set_axisbelow(True)
            # Add hztal and vertical lines
            ax4.axhline(df2['normYieldBLUE95Perc'].mean(), ls='--', c='#444444', zorder=0, linewidth=1, label="Avg. Norm Mean Yield")
            ax4.axvline(df2[fld_fg4].mean(), ls='--', c='#444444', zorder=0, linewidth=1, label=f"Avg. {fld_fg4}")
            #g8b = sns.scatterplot(x='Season_TMAX_mean', y='normYieldBLUE95Perc', data=df3, 
            #                     color='red', hue='GID', style='GID', s=40, alpha=0.85, ax=ax4, label='Target GIDs')

        # Add linear regression by GID
        list_GIDs_highTemperature = []
        for gid in df3['GID'].unique():
            x = df3[(df3['GID']==gid)][fld_fg4].to_numpy()
            y = df3[(df3['GID']==gid)]['normYieldBLUE95Perc'].to_numpy()
            # determine best fit line
            par = np.polyfit(x, y, 1, full=True)
            pend=par[0][0]
            intercept=par[0][1]
            y_predicted = [pend*i + intercept  for i in x]
            if (pend > 0):
                df.loc[((df['Nursery_Yr']==nyr) & (df['GID']==gid)), 'maxTemperatureEnvs'] = 'Support warmest temperature'
                list_GIDs_highTemperature.append(gid)
                GIDs_inTarget_highTemperature.append({"Nyr":nyr, "GID":gid, "slope":pend, "intercept":intercept})
                if (drawFigures is True):
                    l4 = sns.lineplot(x,y_predicted, color='red', ax=ax4, ls='-.', lw=1.25, label='High Temp. GID: {}'.format(gid))
            else:
                df.loc[((df['Nursery_Yr']==nyr) & (df['GID']==gid)), 'maxTemperatureEnvs'] = 'Not support warmest temperature' #'Support coolest temperature'
                if (drawFigures is True):
                    l4 = sns.lineplot(x,y_predicted, color='lightblue', ax=ax4, ls='-', lw=0.25) #, label='{} line'.format(gid))

        df.loc[( ((df['Nursery_Yr']==nyr) & df['GID'].isin(list_GIDs_highTemperature)) ), 'maxTemperatureEnvs'] = 'Support warmest temperature'
        #df7 = df3[df3['GID'].isin(list_GIDs_highTemperature)].groupby(['GID'], as_index=False).agg({'normYieldBLUE95Perc':'mean', 'Season_TMAX_mean':'mean'}).sort_values(['normYieldBLUE95Perc'], ascending=False).reset_index(drop=True)
        df7 = df3[df3['GID'].isin(list_GIDs_highTemperature)].reset_index(drop=True)
        df7['GID'] = df7['GID'].astype(str)
        if (drawFigures is True):
            g8b = sns.scatterplot(x=fld_fg4, y='normYieldBLUE95Perc', data=df7, 
                                  color='red', hue='GID', style='GID', s=40, alpha=0.85, ax=ax4, label='Above Avg. High Temperature GIDs')
            ax4.set_xlabel(f'{fld_fg4}',fontsize=15)
            ax4.set_ylabel('',fontsize=15)
            ax4.get_legend().remove()

        # --------------------
        # Figure 5
        # --------------------
        if (drawFigures is True):
            bottom = 0.7379411764705883
            ax1.set_position([0.125, bottom, 0.3522727272727273, 0.22205882352941175])
            ax2.set_position([0.55, bottom, 0.3522727272727273, 0.22205882352941175])
            #ax5.set_anchor('S') #'C'
            #ax5.set_position([0.125, 0.35, 0.7, 0.35], which='original')

            # ------------------------------
            def add_colorbar(mappable, lbl=''):
                #from mpl_toolkits.axes_grid1 import make_axes_locatable
                last_axes = plt.gca()
                ax = mappable.axes
                fig = ax.figure
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.1)
                cbar = fig.colorbar(mappable, cax=cax)
                cbar.set_label(lbl)
                plt.sca(last_axes)
                return cbar

            # lower left minx miny , upper right maxx maxy
            bounds = [-135.0, -55.0, 179.0, 70.0]
            minx, miny, maxx, maxy = bounds
            w, h = maxx - minx, maxy - miny
            m = Basemap(projection='merc', ellps = 'WGS84', epsg=4326, resolution='i', 
                        llcrnrlon=minx, llcrnrlat=miny, urcrnrlon=maxx, urcrnrlat=maxy, lat_ts=0.0, ax=ax5)
            m.drawcountries(linewidth=0.25, color='gray')
            m.drawcoastlines(linewidth=0.25, color='gray')

            #m.drawmapboundary(fill_color='lightblue')
            m.drawparallels(np.arange(-90.,90.,10.),color='gray',dashes=[1,3],labels=[1,0,0,0])
            m.drawmeridians(np.arange(0.,360.,15.),color='gray',dashes=[1,3],labels=[0,0,0,1])
            #m.fillcontinents(color='beige',lake_color='lightblue')
            m.fillcontinents(color='#f7f7f7',lake_color='lightblue')

            # Data
            df_map_HY = df3[df3['targetClass']=='HY']
            lats_HY = df_map_HY['lat'].to_numpy()
            lons_HY = df_map_HY['lon'].to_numpy()
            x_HY,y_HY = m(lons_HY, lats_HY)
            locs_HY = m.plot(x_HY,y_HY,'x',markersize=5, linewidth=0.25, color='black', zorder=11, 
                             label='Below average in Low Yielding Environments')

            #
            df_map_LY = df3[df3['targetClass']=='LY']
            lats_LY = df_map_LY['lat'].to_numpy()
            lons_LY = df_map_LY['lon'].to_numpy()
            x_LY,y_LY = m(lons_LY, lats_LY)
            locs_LY = m.plot(x_LY,y_LY,'p',linewidth=0.25, markersize=5, markeredgecolor='red', markerfacecolor='#ffffff00', 
                             fillstyle=None,  zorder=11, label='Above average in Low Yielding Environments')

            # -------------
            # we want to use the same normalization for all the scatter plots
            # norm = Normalize(vmin=np.nanmin(df2['GRAIN_YIELD_BLUEs']), vmax=np.nanmax(df2['GRAIN_YIELD_BLUEs']))
            #trgt = df3['targetClass']
            #trgt_list = sorted(set(trgt))
            #marker = {'HY':'x', 'LY':'+'}
            #colormarker = {'HY':'black', 'LY':'red'}
            # create a list of Artists to provide handles to plt.legend
            #scatters = [m.scatter(lons_LY[ix], lats_LY[ix], c= colormarker[trgt], marker=marker[trgt],
            #                      latlon=True, norm=norm, s=20, alpha=1, zorder=11, label=f'{trgt}' )
            #            for trgt in trgt_list for ix in (trgt==trgt,)]
            # -------------


            # -------------
            # High temperatures
            avgLT = df2['Season_TMIN_mean'].mean()
            avgHT = df2['Season_TMAX_mean'].mean()
            #df_map_HT = df7.copy() #df[( (df['Nursery_Yr']==nyr) & (df['maxTemperatureEnvs']=='Support warmest temperature') )]
            #df_map_HT = df_map_HT[df_map_HT[fld_fg4]>avgHT]
            df_map_HT = df[( (df['Nursery_Yr']==nyr) 
                            & (df['minTemperatureEnvs']=='Support warmer temperature')
                            & (df['maxTemperatureEnvs']=='Support warmest temperature')
                           )]
            df_map_HT = df_map_HT[( (df_map_HT['Season_TMIN_mean']>avgLT) & (df_map_HT['Season_TMAX_mean']>avgHT) )]
            lats_HT = df_map_HT['lat'].to_numpy()
            lons_HT = df_map_HT['lon'].to_numpy()
            x_HT,y_HT = m(lons_HT, lats_HT)
            locs_HT = m.plot(x_HT,y_HT,'|',markersize=10, linewidth=0.25, color='orange', zorder=12, label='Warmest Environments')

            # -------------
            lats = df2['lat'].to_numpy()
            lons = df2['lon'].to_numpy()
            clr = df2['normYieldBLUE95Perc'].to_numpy()
            magnitude = df2['GRAIN_YIELD_BLUEs'].to_numpy()
            norm = Normalize(vmin=np.nanmin(clr), vmax=np.nanmax(clr))

            x,y = m(lons, lats)
            locs = m.scatter(x,y,s=20*magnitude, c=clr, fc=None, norm=norm, alpha=0.45, zorder=10) #, latlon=True, label='Map selected GIDs by GY')
            #locs = m.scatter(x,y, norm=norm, lw=0.25, s=20*magnitude, ec='#555555', facecolor=None, 
            #                 linestyle='-', alpha=0.85, zorder=10) #, latlon=True, label='Map selected GIDs by GY')

            add_colorbar(locs, lbl="Normalized Grain Yield (BLUEs)")
            #c = plt.colorbar(locs, ax=ax5, orientation='vertical', use_gridspec=True, pad=0.05)
            #fig.colorbar(im1, cax=cax, orientation='vertical')
            #c.set_label("norm Grain Yield")

            # square up axes and basemap
            #ax5.set_aspect(1)
            ax5.set_aspect('equal')
            #ax5.get_legend().remove()

            # ------------------------------
            # remove labels
            #for ax in fig.get_axes():
            #    ax.label_outer()
            ax2.sharey(ax1)
            ax4.sharey(ax3)

            # ------------------------------
            # Add legend()
            def getLegend_HandlesLabels(ax, handout, lablout):
                handles, labels = ax.get_legend_handles_labels()
                for h,l in zip(handles,labels):
                    if l not in lablout:
                        lablout.append(l)
                        handout.append(h)
                return handout, lablout

            handout=[]
            lablout=[]
            handout, lablout = getLegend_HandlesLabels(ax1, handout, lablout)
            handout, lablout = getLegend_HandlesLabels(ax2, handout, lablout)
            handout, lablout = getLegend_HandlesLabels(ax3, handout, lablout)
            handout, lablout = getLegend_HandlesLabels(ax4, handout, lablout)
            handout, lablout = getLegend_HandlesLabels(ax5, handout, lablout)
            #handout, lablout = getLegend_HandlesLabels(ax6, handout, lablout)

            fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.03), loc="lower center", ncol=8, 
                       borderaxespad=0,bbox_transform=plt.gcf().transFigure, fontsize=8) #, fancybox=True, shadow=True)
            #plt.title('NURSERY - Nursery Year {}'.format(nyr), fontsize=18)
            fig.suptitle('{} - Nursery Year {}'.format(nursery, nyr), fontsize=20, y=1.005)
            hoy = datetime.now().strftime('%Y%m%d')
            output_path = os.path.join(path_to_save_results, "{}_TargetGIDs_NurseryYr_{}".format(nursery, hoy))
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)

            fig.savefig(os.path.join(output_path, "{}_TargetGIDs_Nyr{}_{}.pdf".format(nursery, nyr, hoy)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', 
                        edgecolor='none', transparent=False,
                        pad_inches=0.5, dpi=300)

            fig.show()
    #
    GIDs_inTarget_1 = pd.DataFrame(GIDs_inTarget_1)
    GIDs_inTarget_highTemperature = pd.DataFrame(GIDs_inTarget_highTemperature)
    GIDs_LowestHighest_NormilizedYield = pd.DataFrame(GIDs_LowestHighest_NormilizedYield)
    
    return df, GIDs_inTarget_1, GIDs_inTarget_highTemperature, GIDs_LowestHighest_NormilizedYield

#
def filterTargetGIDs_byThresholdMaxTemperature(df_filtered, GIDs_inTarget_highTemperature, nursery='ESWYT', 
                                               avgLT_threshold=10.0, avgHT_threshold=30.0, gt32C_ndays_threshold=10,
                                               path_to_save_results='./', saveFiles=True, verbose=True):
    '''
        Filter target GIDs by Temperature using `Season_TMAX_mean` and `S-H_TMAX_gt32C_ndays` as references
    '''

    #avgLT_threshold = 11.0 #df2['Season_TMIN_mean'].mean()
    #avgHT_threshold = 30.0 #df2['Season_TMAX_mean'].mean()
    df = df_filtered.copy()
    df['finalTarget'] = 0

    df.loc[( #( (df['Season_TMIN_mean']>avgLT_threshold) | (df['Season_TMAX_mean']>avgHT_threshold) ) & 
        (( df['Season_TMAX_mean']>avgHT_threshold) | ( df['S-H_TMAX_gt32C_ndays']>gt32C_ndays_threshold) )
            & (df['GID'].isin(GIDs_inTarget_highTemperature['GID'].unique())) 
            & (df['targetClass']=='Above Avg in Low Yielding Envs - Below Avg in High Yielding Envs')
            #& (df['Season_TMAX_gt32C_ndays']=='Support warmer temperature')
            #& (df['maxTemperatureEnvs']=='Support warmest temperature')
           ), 'finalTarget'] = 1
    #df_map_GIDs_highTemperature = df[( (df['Season_TMIN_mean']>avgLT_threshold) & (df['Season_TMAX_mean']>avgHT_threshold) )].reset_index(drop=True)
    df_map_GIDs_highTemperature = df[df['finalTarget']==1].reset_index(drop=True)
    if (verbose is True):
        print("Number of GIDs supporting high temperatures: {}".format(len(df_map_GIDs_highTemperature['GID'].unique())))
        print(df_map_GIDs_highTemperature['GID'].unique())
        print("Number of Nursery Year supporting high temperatures: {}".format(len(df_map_GIDs_highTemperature['Nursery_Yr'].unique())))

    #save
    if (saveFiles is True):
        hoy = datetime.now().strftime('%Y%m%d')
        df.to_csv(os.path.join(path_to_save_results, "{}_selectedGIDs_results_{}.csv".format(nursery,hoy)), index=False)
        df_map_GIDs_highTemperature.to_csv(os.path.join(path_to_save_results, "{}_final_TargetGIDs_{}.csv".format(nursery,hoy)), index=False)
    return df, df_map_GIDs_highTemperature


# 
def displayMap_spatialDistSelectedGIDs(df_gids, nursery='ESWYT', path_to_save_results='./', saveFig=True, verbose=False):
    '''
        Spatial distribution of selected IWIN genotypes
    '''
    try:
        df_selectedGIDs_final = df_gids.copy()

        fig = plt.figure(figsize=(12,10))
        #Custom adjust of the subplots
        plt.subplots_adjust(left=0.05,right=0.98,top=0.95,bottom=0.05,wspace=0.05,hspace=0.05)
        ax = plt.subplot(111)

        # lower left minx miny , upper right maxx maxy
        bounds = [-135.0, -55.0, 179.0, 70.0]
        minx, miny, maxx, maxy = bounds
        w, h = maxx - minx, maxy - miny
        m = Basemap(projection='merc', ellps = 'WGS84', epsg=4326, resolution='i', 
                    llcrnrlon=minx, llcrnrlat=miny, urcrnrlon=maxx, urcrnrlat=maxy, lat_ts=0.0, ax=ax)
        m.drawcountries(linewidth=0.25, color='gray')
        m.drawcoastlines(linewidth=0.25, color='gray')

        #m.drawmapboundary(fill_color='lightblue')
        m.drawparallels(np.arange(-90.,90.,10.),color='gray',dashes=[1,3],labels=[1,0,0,0])
        m.drawmeridians(np.arange(0.,360.,15.),color='gray',dashes=[1,3],labels=[0,0,0,1])
        #m.fillcontinents(color='beige',lake_color='lightblue')
        m.fillcontinents(color='#f7f7f7',lake_color='lightblue')

        #df_map_LY = df_selectedGIDs_final[df_selectedGIDs_final['finalTarget']==1]
        #lats_LY = df_map_LY['lat'].to_numpy()
        #lons_LY = df_map_LY['lon'].to_numpy()
        #x_LY,y_LY = m(lons_LY, lats_LY)
        #locs_LY = m.plot(x_LY,y_LY,'p',linewidth=0.15, markersize=4, markeredgecolor='red', 
        #                 markerfacecolor='#ffffff00', fillstyle=None,  zorder=11, label='Above average in Low Yielding Environments')

        # ------------------------------
        def add_colorbar(mappable, lbl=''):
            #from mpl_toolkits.axes_grid1 import make_axes_locatable
            last_axes = plt.gca()
            ax = mappable.axes
            fig = ax.figure
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.1)
            cbar = fig.colorbar(mappable, cax=cax)
            cbar.set_label(lbl)
            plt.sca(last_axes)
            return cbar

        # Data
        # -------------
        lats = df_selectedGIDs_final['lat'].to_numpy()
        lons = df_selectedGIDs_final['lon'].to_numpy()
        #clr = df_selectedGIDs_final['Season_TMAX_mean'].to_numpy()
        clr = df_selectedGIDs_final['Season_TMAX_gt32C_ndays'].to_numpy()
        magnitude = df_selectedGIDs_final['GRAIN_YIELD_BLUEs'].to_numpy()
        norm = Normalize(vmin=np.nanmin(clr), vmax=np.nanmax(clr))

        x,y = m(lons, lats)
        locs = m.scatter(x,y,s=20*magnitude, c=clr, cmap='inferno', fc=None, norm=norm, 
                         alpha=0.45, zorder=10) #, latlon=True, label='Map selected GIDs by GY')

        add_colorbar(locs, lbl='Days of maximum temperature \ngreater than 32 oC during growing season') #lbl="Normalized Grain Yield (BLUEs)")
        #c = plt.colorbar(locs, ax=ax, orientation='vertical', use_gridspec=True, pad=0.05)
        #fig.colorbar(im1, cax=cax, orientation='vertical')
        #c.set_label("norm Grain Yield")

        # square up axes and basemap
        #ax.set_aspect(1)
        ax.set_aspect('equal')

        #fig.suptitle('Spatial distribution of selected NURSERY genotypes\nGIDs above average in low yielding environments in anomalous weather'.format(), 
        #             fontsize=20, y=0.8)
        ax.set_title('Spatial distribution of selected {} genotypes\nGIDs above average in low yielding environments and anomalous weather'.format(nursery), fontsize=20, y=1.01)
        #plt.legend()
        fig.tight_layout()

        if (saveFig is True):
            hoy = datetime.now().strftime('%Y%m%d')
            fig.savefig(os.path.join(path_to_save_results, "{}_map_selected_GIDs_{}.png".format(nursery, hoy)), 
                        alpha=False, transparent=False, dpi=300)
            fig.savefig(os.path.join(path_to_save_results, "{}_map_selected_GIDs_{}.pdf".format(nursery,hoy)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', 
                        edgecolor='none', transparent=False,
                        pad_inches=0.5, dpi=300)
        fig.show()
        
    except Exception as err:
        print("It couldn't display the spatial distribution of selected IWIN genotypes. Error:", err)
    
#
def prepareGIDtoChartSensitivity(df, gid):
    df_ex1 = None
    if (gid in df['GID'].unique()):
        df_ex1 = df[df['GID']==gid]
        # rename some field
        df_ex1.rename(columns={
            'S-H_TMAX_mean':'Vegetative',
            'H-Hplus15d_TMAX_mean':'Heading',
            'Hplus15d-M_TMAX_mean':'Grain Filling',
            'Season_TMAX_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }, inplace=True)

        df_ex1 = df_ex1[['GID', 'Vegetative', 'Heading', 'Grain Filling', 'Season', 'Grain Yield', 'finalTarget']].reset_index(drop=True)
    return df_ex1

def displayGenotypeSensitivitybyWeather(df_final, gid, yFld='Grain Yield', nursery='ESWYT', path_to_save_results='./', saveFig=False):
    '''
        The Eberhart and Russell regression with the grain yield environmental means along 
        the x-axis replaced with a climate variable.
        
        Genotypes differ in their sensitivity to climate variables
    '''
    df = df_final.copy()
    df = prepareGIDtoChartSensitivity(df, gid=gid)
    if (df is None):
        print("GID not found!")
        return
    GScolors = {
        'Vegetative':'green',
        'Heading':'orange',
        'Grain Filling':'brown',
        'Season':'red'
    }
    fig, (ax1) = plt.subplots(figsize=(8,8))
    g1 = sns.scatterplot(x='Vegetative', y=yFld, data=df, 
                         alpha=0.75, color=GScolors['Vegetative'], s=40, lw=1.0,  ax=ax1)
    g2 = sns.scatterplot(x='Heading', y=yFld, data=df, 
                         alpha=0.75, color=GScolors['Heading'], s=40, lw=1.0,  ax=ax1)
    g3 = sns.scatterplot(x='Grain Filling', y=yFld, data=df, 
                         alpha=0.75, color=GScolors['Grain Filling'], s=40, lw=1.0,  ax=ax1)
    #g4 = sns.scatterplot(x='Season', y=yFld, data=df, 
    #                     alpha=0.75, color=GScolors['Season'], s=40, lw=1.0,  ax=ax1)
    
    df2 = df[df['finalTarget']==1]
    g5 = sns.scatterplot(x='Vegetative', y=yFld, data=df2, 
                         alpha=0.75, ec=GScolors['Vegetative'], fc='none', marker="o", 
                         s=120, lw=0.35,  ax=ax1, label='Selected observation (Vegetative)')
    g6 = sns.scatterplot(x='Heading', y=yFld, data=df2, 
                         alpha=0.75, ec=GScolors['Heading'], fc='none', marker="o", 
                         s=120, lw=0.35,  ax=ax1, label='Selected observation (Heading)')
    g7 = sns.scatterplot(x='Grain Filling', y=yFld, data=df2, 
                         alpha=0.75, ec=GScolors['Grain Filling'], fc='none', marker="o", 
                         s=120, lw=0.35,  ax=ax1, label='Selected observation (Grain Filling)')
    
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    title1=f'Temperature vs Grain Yield in 3 growth stages (GID{gid})'
    ax1.set_title('{}'.format(title1), fontsize=18)
    ax1.set_xlabel('Average daily maximum temperature (oC)', fontsize=15)
    ax1.set_ylabel(f'{yFld} (t/ha)', fontsize=15)
    ax1.tick_params(labelsize=12)

    # Add linear regression for Vegetative
    # Should be by GID
    # Vegetative 
    x = df['Vegetative'].to_numpy()
    y = df[yFld].to_numpy()
    # determine best fit line
    par = np.polyfit(x, y, 1, full=True)
    pend=par[0][0]
    intercept=par[0][1]
    print("Vegetative y = {:.7f}x + {:.7f}".format(pend, intercept))
    y_predicted = [pend*i + intercept  for i in x]
    l1 = sns.lineplot(x,y_predicted, color=GScolors['Vegetative'], ax=ax1, lw=2.25, label='Vegetative')

    # Heading 
    x = df['Heading'].to_numpy()
    y = df[yFld].to_numpy()
    # determine best fit line
    par = np.polyfit(x, y, 1, full=True)
    pend=par[0][0]
    intercept=par[0][1]
    print("Heading y = {:.7f}x + {:.7f}".format(pend, intercept))
    y_predicted = [pend*i + intercept  for i in x]
    l1 = sns.lineplot(x,y_predicted, color=GScolors['Heading'], ax=ax1, lw=2.25, label='Heading')

    # Grain filling 
    x = df['Grain Filling'].to_numpy()
    y = df[yFld].to_numpy()
    # determine best fit line
    par = np.polyfit(x, y, 1, full=True)
    pend=par[0][0]
    intercept=par[0][1]
    print("Grain Filling y = {:.7f}x + {:.7f}".format(pend, intercept))
    y_predicted = [pend*i + intercept  for i in x]
    l1 = sns.lineplot(x,y_predicted, color=GScolors['Grain Filling'], ax=ax1, lw=2.25, label='Grain Filling')

    plt.legend(bbox_to_anchor=(0.5, -0.2), loc="center", ncol=2,  borderaxespad=0,fontsize=10)
    #plt.legend(loc='upper left')
    fig.tight_layout()
    if (saveFig is True):
        hoy = datetime.now().strftime('%Y%m%d')
        fig.savefig(os.path.join(path_to_save_results, "{}_TempVsGY_GID{}_{}.png".format(nursery, gid, hoy)), 
                    alpha=False, transparent=False, dpi=300)
        fig.savefig(os.path.join(path_to_save_results, "{}_TempVsGY_GID{}_{}.pdf".format(nursery, gid, hoy)), 
                    bbox_inches='tight', orientation='portrait', papertype='a4', 
                    edgecolor='none', transparent=False,
                    pad_inches=0.5, dpi=300)
    fig.show()

#
def displayTempVsGY_inGrowStagesforAllGIDs(df_filtered, df_HT, nursery='ESWYT', path_to_save_results='./', saveFig=True ):
    '''
        Temperature vs Grain Yield in 3 growth stages (all GIDs)
    '''
    df_selectedGIDs_final = df_HT.copy()
    # Number of pages
    pages = np.ceil(len(df_selectedGIDs_final['GID'].unique())/40)
    for pg in range(int(pages)):
        # Number of figures per page
        lstGIDs = df_selectedGIDs_final['GID'].unique()[40*pg:40*(pg+1)]
        nFigures = len(lstGIDs)
        cols = 4
        rows = int(nFigures/4) + 1
        if (rows>10): # No print more than 20 plots per figure or page
            rows = 10

        GScolors = {
                'Vegetative':'green',
                'Heading':'orange',
                'Grain Filling':'brown',
                'Season':'red'
            }
        def createGSFigure(df, ax, yFld='Grain Yield', s=10, s2=40, lw=1.0, lw2=1.25, alpha=0.45):
            g1 = sns.scatterplot(x='Vegetative', y=yFld, data=df, 
                                 alpha=alpha, color=GScolors['Vegetative'], s=s, lw=lw,  ax=ax)
            g2 = sns.scatterplot(x='Heading', y=yFld, data=df, 
                                 alpha=alpha, color=GScolors['Heading'], s=s, lw=lw,  ax=ax)
            g3 = sns.scatterplot(x='Grain Filling', y=yFld, data=df, 
                                 alpha=alpha, color=GScolors['Grain Filling'], s=s, lw=lw,  ax=ax)
            #g4 = sns.scatterplot(x='Season', y=yFld, data=df, 
            #                     alpha=alpha, color=GScolors['Season'], s=s, lw=lw,  ax=ax)
            df2 = df[df['finalTarget']==1]
            g5 = sns.scatterplot(x='Vegetative', y=yFld, data=df2, 
                                 alpha=0.95, ec=GScolors['Vegetative'], fc='none', marker="o", 
                                 s=s2, lw=0.35,  ax=ax, label='Selected observation (Vegetative)')
            g6 = sns.scatterplot(x='Heading', y=yFld, data=df2, 
                                 alpha=0.95, ec=GScolors['Heading'], fc='none', marker="o", 
                                 s=s2, lw=0.35,  ax=ax, label='Selected observation (Heading)')
            g7 = sns.scatterplot(x='Grain Filling', y=yFld, data=df2, 
                                 alpha=0.95, ec=GScolors['Grain Filling'], fc='none', marker="o", 
                                 s=s2, lw=0.35,  ax=ax, label='Selected observation (Grain Filling)')

            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax.set_axisbelow(True)

            # ----------------------
            # Linear regression
            # ----------------------
            # Add linear regression for Vegetative
            # Should be by GID
            # Vegetative 
            x = df['Vegetative'].to_numpy()
            y = df[yFld].to_numpy()
            # determine best fit line
            par = np.polyfit(x, y, 1, full=True)
            pend=par[0][0]
            intercept=par[0][1]
            #print("Vegetative y = {:.7f}x + {:.7f}".format(pend, intercept))
            y_predicted = [pend*i + intercept  for i in x]
            l1 = sns.lineplot(x,y_predicted, color=GScolors['Vegetative'], ax=ax, lw=lw2, label='Vegetative')

            # Heading 
            x = df['Heading'].to_numpy()
            y = df[yFld].to_numpy()
            # determine best fit line
            par = np.polyfit(x, y, 1, full=True)
            pend=par[0][0]
            intercept=par[0][1]
            #print("Heading y = {:.7f}x + {:.7f}".format(pend, intercept))
            y_predicted = [pend*i + intercept  for i in x]
            l1 = sns.lineplot(x,y_predicted, color=GScolors['Heading'], ax=ax, lw=lw2, label='Heading')

            # Grain filling 
            x = df['Grain Filling'].to_numpy()
            y = df[yFld].to_numpy()
            # determine best fit line
            par = np.polyfit(x, y, 1, full=True)
            pend=par[0][0]
            intercept=par[0][1]
            #print("Grain Filling y = {:.7f}x + {:.7f}".format(pend, intercept))
            y_predicted = [pend*i + intercept  for i in x]
            l1 = sns.lineplot(x,y_predicted, color=GScolors['Grain Filling'], ax=ax, lw=lw2, label='Grain Filling')

        #
        # ------------------------------
        # Add legend()
        def getLegend_HandlesLabels(ax, handout, lablout):
            handles, labels = ax.get_legend_handles_labels()
            for h,l in zip(handles,labels):
                if l not in lablout:
                    lablout.append(l)
                    handout.append(h)
            return handout, lablout


        fig = plt.figure(figsize=(10, 18), tight_layout=True) #, constrained_layout=True) 
        gs = gridspec.GridSpec(nrows=rows, ncols=cols) #, wspace=0, hspace=0)

        handout=[]
        lablout=[]
        nfig = 0
        for nr in range(rows):
            for nc in range(cols):
                if (nfig>0):
                    ax = fig.add_subplot(gs[nr, nc], sharey=ax, sharex=ax)
                    #plt.setp(ax.get_xticklabels(), visible=False)
                    #plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    ax = fig.add_subplot(gs[nr, nc])
                # get data to display in chart
                if (nfig < len(lstGIDs)):
                    df_g = prepareGIDtoChartSensitivity(df_filtered, gid=lstGIDs[nfig])
                    nobs = len(df_g)
                    nselobs = len(df_g[df_g['finalTarget']==1])
                    createGSFigure(df_g, ax=ax, yFld='Grain Yield')
                    #ax.set_title(f'{nr}, {nc} - GID{lstGIDs[nfig]} - #{nselobs}/{nobs}')
                    ax.set_title(f'GID{lstGIDs[nfig]} - {nselobs}/{nobs}')
                    ax.set_xlabel('') #'Average daily maximum temperature (oC)', fontsize=12)
                    ax.set_ylabel('') #'Grain Yield (t/ha)', fontsize=12)
                    ax.tick_params(labelsize=8, color='gray')
                    #ax.set(xticks=[], yticks=[])
                    ax.get_legend().remove()

                    # Legend
                    handout, lablout = getLegend_HandlesLabels(ax, handout, lablout)
                    nfig = nfig + 1

        #
        fig.text(0.5, -0.01, 'Average Daily Maximum Temperature (°C)', ha='center', va='center', fontweight='bold', fontsize=14)
        fig.text(-0.02, 0.5, 'Grain Yield (t/ha)', ha='center', va='center', 
                 rotation='vertical', fontweight='bold', fontsize=14)
        #
        fig.suptitle(f'{nursery} - Climate x Growth stage in targeted genotypes - part {pg+1}', fontsize=20, y=1.005)
        fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.07), loc="lower center", ncol=2, 
                       borderaxespad=0,bbox_transform=plt.gcf().transFigure, fontsize=10) #, fancybox=True, shadow=True)


        hoy = datetime.now().strftime('%Y%m%d')
        #output_path = os.path.join(RESULTS_IWIN_PATH, "NURSERY_TargetGIDs_ClimatexGrowthstage_{}".format(hoy))
        #if not os.path.exists(output_path):
        #    os.makedirs(output_path, exist_ok=True)
        fig.tight_layout()
        if (saveFig is True):
            hoy = datetime.now().strftime('%Y%m%d')
            fig.savefig(os.path.join(path_to_save_results, "{}_TargetGIDs_ClimatexGrowthstage_part_{}_{}.pdf".format(nursery, pg+1, hoy)), 
                        alpha=False, transparent=False, dpi=300)
            fig.savefig(os.path.join(path_to_save_results, "{}_TargetGIDs_ClimatexGrowthstage_part_{}_{}.pdf".format(nursery, pg+1, hoy)), 
                        bbox_inches='tight', orientation='portrait', papertype='a4', 
                        edgecolor='none', transparent=False, pad_inches=0.5, dpi=300)

        fig.show()
        
#
def drawTemperaturebyGrowthStage_acrossYears(df_data=None, nursery='NURSERY', temp='TMAX', s=20, s2=80, alpha=0.75, lw=1.5, bxplot=False,
                                             dirname = 'GrowthStages_Temperature', fname='TemperatureAcrossYrsxGrowthstage',
                                             path_to_save_results='./', saveFig=False, showFig=True, fmt='pdf'):
    ''' 
        Average daily temperature across years vs mean grain yield by GID
    '''
    
    # filter
    #df = df_data[( (df_data['finalTarget']==1) & (df_dssat['DAP']<300) & (df_dssat['CycleStartYr']>=1979) 
    #              )].sort_values(['CycleStartYr']).reset_index(drop=True)
    df = df_data.copy()
    
    # Renanme some features to graph according to climate variables
    if (temp=='TMIN'):
        df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TMIN_mean':'Vegetative',
            'H-Hplus15d_TMIN_mean':'Heading',
            'Hplus15d-M_TMIN_mean':'Grain Filling',
            'Season_TMIN_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }, inplace=True)
        
    elif (temp=='TMAX'):
        df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TMAX_mean':'Vegetative',
            'H-Hplus15d_TMAX_mean':'Heading',
            'Hplus15d-M_TMAX_mean':'Grain Filling',
            'Season_TMAX_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }, inplace=True)
    else:
        df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TAVG_mean':'Vegetative',
            'H-Hplus15d_TAVG_mean':'Heading',
            'Hplus15d-M_TAVG_mean':'Grain Filling',
            'Season_TAVG_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }, inplace=True)
    
    df = df[['GID', 'Year', 'Vegetative', 'Heading', 'Grain Filling', 'Season', 'Grain Yield', 'finalTarget']].reset_index(drop=True)
    df = df.sort_values(['Year']).reset_index(drop=True)
    
    # Growth Stages
    phases_labels = ['Vegetative', 'Heading', 'Grain Filling', 'Season'] #'Planting to Harvest'
    GScolors = {
        'Vegetative':'green',
        'Heading':'orange',
        'Grain Filling':'brown',
        'Season':'red'
    }
    # 
    # ------------------------------
    def createFigAcrossYrs(df, ax, gs=0, fldX='Year', s=20, s2=80, lw=1.0, alpha=0.85, bxplot=bxplot):
        # Box plots
        if (bxplot is True):
            g1 = sns.boxplot(x=fldX, y=df[phases_labels[gs]], data=df, hue=None, ax=ax, 
                             flierprops={"marker": "x", "markersize":5, "markeredgecolor":"lightgray" },
                             boxprops={"facecolor": (.4, .6, .8, .5)}, medianprops={"color": GScolors[phases_labels[gs]]});
        else: 
            g1 = sns.scatterplot(x=fldX, y=df[phases_labels[gs]], data=df, hue=None, marker='o', color=GScolors[phases_labels[gs]],
                                 s=s, lw=lw, alpha=alpha, ax=ax, label=phases_labels[gs])
        g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax.set_axisbelow(True)
        
        if (bxplot is False):
            df2 = df[df['finalTarget']==1]
            g2 = sns.scatterplot(x=fldX, y=df2[phases_labels[gs]], data=df2, 
                                 alpha=0.95, ec=GScolors[phases_labels[gs]], fc='none', marker="o", 
                                 s=s2, lw=0.35,  ax=ax, label=f'Selected observations ({phases_labels[gs]})')
        # Add linear regression
        x = df[fldX].astype(int).to_numpy()
        y = df[phases_labels[gs]].to_numpy()
        # determine best fit line
        par = np.polyfit(x, y, 1, full=True)
        pend=par[0][0]
        intercept=par[0][1]
        #print("{} y = {:.7f}x + {:.7f}".format(phases_labels[0], pend, intercept))
        change_Period = (pend*np.nanmax(x) + intercept) - (pend*np.nanmin(x) + intercept)
        #print(f"{phases_labels[0]} change over time: {change_Period:.2f} °C")
        y_predicted = [pend*i + intercept  for i in x]
        if (bxplot is False):
            l1 = sns.lineplot(x,y_predicted, color=GScolors[phases_labels[gs]], ls='--', lw=0.25, 
                              label=f'Temperature trend ({phases_labels[gs]})', ax=ax)
            #l1 = sns.regplot(x,y_predicted, color=GScolors[phases_labels[gs]], label=f'{phases_labels[gs]}', ax=ax)
        pos_x = 0.01
        pos_x2 = 0.98
        pos_y = 1.15 #0.2 # Top 0.95
        fontsize=10
        if (change_Period > 0):
            ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], 
                    ha='left', va='top', transform=ax.transAxes)
            ax.text(pos_x2, pos_y, r"$\bf{:.2f}°C$ increase".format(change_Period), fontsize=fontsize, ha='right', va='top', transform=ax.transAxes)
        else:
            ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], 
                    ha='left', va='top', transform=ax.transAxes)
            ax.text(pos_x2, pos_y, r"$\bf{:.2f}°C$ decrease".format(change_Period), fontsize=fontsize, ha='right', va='top', transform=ax.transAxes)
        
    # ------------------------------
    # Add legend()
    def getLegend_HandlesLabels(ax, handout, lablout):
        handles, labels = ax.get_legend_handles_labels()
        for h,l in zip(handles,labels):
            if l not in lablout:
                lablout.append(l)
                handout.append(h)
        return handout, lablout
    # ------------------------------
    fig = plt.figure(figsize=(6, 8), tight_layout=True) #, constrained_layout=True) 
    rows = len(phases_labels)
    cols = 1
    gs_tmin = gridspec.GridSpec(nrows=rows, ncols=cols) #, wspace=0, hspace=0)
    gs_tmax = gridspec.GridSpec(nrows=rows, ncols=cols) #, wspace=0, hspace=0)
    
    handout=[]
    lablout=[]
    nfig = 0
    for nr in range(rows):
        for nc in range(cols):
            if (nfig>0):
                ax = fig.add_subplot(gs_tmax[nr, nc], sharey=ax, sharex=ax)
            else:
                ax = fig.add_subplot(gs_tmax[nr, nc])
            
            if (nfig<rows-1):
                plt.setp(ax.get_xticklabels(), visible=False)
                #plt.setp(ax.get_yticklabels(), visible=False)
            # Create figure
            createFigAcrossYrs(df, ax=ax, gs=nfig, s=s, s2=s2, lw=lw, alpha=alpha) #, fldX='Year', s=20, lw=1.0, alpha=0.85)
            #ax.set_title(f'{phases_labels[nfig]}', fontweight='bold', fontsize=10, color='gray')
            ax.set_xlabel('') #'Average daily maximum temperature (oC)', fontsize=12)
            ax.set_ylabel('') #'Grain Yield (t/ha)', fontsize=12)
            ax.tick_params(labelsize=8, color='gray', rotation=90)
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
            #ax.set(xticks=[], yticks=[])
            if (bxplot is False):
                ax.get_legend().remove()
            # Legend
            handout, lablout = getLegend_HandlesLabels(ax, handout, lablout)
            nfig = nfig + 1
    
    #
    fig.text(0.5, -0.01, 'Time (Years)', ha='center', va='center', fontweight='bold', fontsize=14)
    if (temp=='TMIN'):
        fig.text(-0.02, 0.5, 'Average Daily Minimum Temperature (°C)', ha='center', va='center', 
                 rotation='vertical', fontweight='bold', fontsize=14)
    elif (temp=='TMAX'):
        fig.text(-0.02, 0.5, 'Average Daily Maximum Temperature (°C)', ha='center', va='center', 
                 rotation='vertical', fontweight='bold', fontsize=14)
    else:
        fig.text(-0.02, 0.5, 'Average Daily Mean Temperature (°C)', ha='center', va='center', 
                 rotation='vertical', fontweight='bold', fontsize=14)
    #
    title1='{} - Temperature across years by growth stage ({} - {})'.format(nursery, df['Year'].min(), df['Year'].max())
    # Title for the complete figure
    fig.suptitle('{}'.format(title1), fontsize='x-large', fontweight='bold' ) #, fontsize=20, y=1.005)
    if (bxplot is False):
        fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.13), loc="lower center", ncol=4, 
                   borderaxespad=0,bbox_transform=plt.gcf().transFigure, fontsize=8) #, fancybox=True, shadow=True)
    
    fig.tight_layout()
    #fig.tight_layout(rect=(0.08,0.05,1,0.985)) # Left, bottom, right, top
    # Save in PDF
    hoy = datetime.now().strftime('%Y%m%d')
    figures_path = os.path.join(path_to_save_results, '{}_{}'.format(dirname, hoy))
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    if (saveFig is True and fmt=='pdf'):
        fig.savefig(os.path.join(figures_path,"{}_{}_{}.{}".format(nursery, fname, hoy, fmt)), 
                    bbox_inches='tight', orientation='portrait', papertype='a4', 
                    edgecolor='none', transparent=False, pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        fig.savefig(os.path.join(figures_path,"{}_{}_{}.{}".format(nursery, fname, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=300)
    
    if (showFig is True):
        fig.show()
    else:
        del fig
        plt.close();
        
#
def drawTemperaturebyGrowthStage_acrossYears_v2(df_data=None, nursery='NURSERY', s=20, s2=80, alpha=0.75, lw=1.5, bxplot=False,
                                             dirname = 'GrowthStages_Temperature', fname='TemperatureAcrossYrsxGrowthstage_v2',
                                             path_to_save_results='./', saveFig=False, showFig=True, fmt='pdf'):
    ''' 
        Average daily temperature across years vs mean grain yield by GID
    '''
    
    # filter
    df = df_data.copy()
    
    # Renanme some features to graph according to climate variables
    #if (temp=='TMIN'):
    df_TMIN = df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TMIN_mean':'Vegetative',
            'H-Hplus15d_TMIN_mean':'Heading',
            'Hplus15d-M_TMIN_mean':'Grain Filling',
            'Season_TMIN_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }) 
    df_TMAX = df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TMAX_mean':'Vegetative',
            'H-Hplus15d_TMAX_mean':'Heading',
            'Hplus15d-M_TMAX_mean':'Grain Filling',
            'Season_TMAX_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }) 
    df_TAVG = df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TAVG_mean':'Vegetative',
            'H-Hplus15d_TAVG_mean':'Heading',
            'Hplus15d-M_TAVG_mean':'Grain Filling',
            'Season_TAVG_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }) 
    
    sel_cols = ['GID', 'Year', 'Vegetative', 'Heading', 'Grain Filling', 'Season', 'Grain Yield', 'finalTarget']
    df_TMIN = df_TMIN[sel_cols].sort_values(['Year']).reset_index(drop=True)
    df_TMAX = df_TMAX[sel_cols].sort_values(['Year']).reset_index(drop=True)
    df_TAVG = df_TAVG[sel_cols].sort_values(['Year']).reset_index(drop=True)
    
    # Growth Stages
    phases_labels = ['Vegetative', 'Heading', 'Grain Filling', 'Season'] #'Planting to Harvest'
    GScolors = {
        'Vegetative':'green',
        'Heading':'orange',
        'Grain Filling':'brown',
        'Season':'red'
    }
    # 
    # ------------------------------
    def createFigAcrossYrs(df, ax, gs=0, fldX='Year', s=20, s2=80, lw=1.0, alpha=0.85, bxplot=bxplot):
        # Box plots
        if (bxplot is True):
            g1 = sns.boxplot(x=fldX, y=df[phases_labels[gs]], data=df, hue=None, ax=ax, 
                             flierprops={"marker": "x", "markersize":5, "markeredgecolor":"lightgray" },
                             boxprops={"facecolor": (.4, .6, .8, .5)}, medianprops={"color": GScolors[phases_labels[gs]]});
        else: 
            g1 = sns.scatterplot(x=fldX, y=df[phases_labels[gs]], data=df, hue=None, marker='o', color=GScolors[phases_labels[gs]],
                                 s=s, lw=lw, alpha=alpha, ax=ax, label=phases_labels[gs])
        g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax.set_axisbelow(True)
        
        if (bxplot is False):
            df2 = df[df['finalTarget']==1]
            g2 = sns.scatterplot(x=fldX, y=df2[phases_labels[gs]], data=df2, 
                                 alpha=0.95, ec=GScolors[phases_labels[gs]], fc='none', marker="o", 
                                 s=s2, lw=0.35,  ax=ax, label=f'Selected observations ({phases_labels[gs]})')
        # Add linear regression
        x = df[fldX].astype(int).to_numpy()
        y = df[phases_labels[gs]].to_numpy()
        # determine best fit line
        par = np.polyfit(x, y, 1, full=True)
        pend=par[0][0]
        intercept=par[0][1]
        #print("{} y = {:.7f}x + {:.7f}".format(phases_labels[0], pend, intercept))
        change_Period = (pend*np.nanmax(x) + intercept) - (pend*np.nanmin(x) + intercept)
        #print(f"{phases_labels[0]} change over time: {change_Period:.2f} °C")
        y_predicted = [pend*i + intercept  for i in x]
        if (bxplot is False):
            l1 = sns.lineplot(x,y_predicted, color=GScolors[phases_labels[gs]], ls='--', lw=0.25, 
                              label=f'Temperature trend ({phases_labels[gs]})', ax=ax)
            #l1 = sns.regplot(x,y_predicted, color=GScolors[phases_labels[gs]], label=f'{phases_labels[gs]}', ax=ax)
        pos_x = 0.01
        pos_x2 = 0.98
        pos_y = 1.15 #0.2 # Top 0.95
        fontsize=10
        if (change_Period > 0):
            ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], 
                    ha='left', va='top', transform=ax.transAxes)
            ax.text(pos_x2, pos_y, r"$\bf{:.2f}°C$ increase".format(change_Period), fontsize=fontsize, ha='right', va='top', transform=ax.transAxes)
        else:
            ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], 
                    ha='left', va='top', transform=ax.transAxes)
            ax.text(pos_x2, pos_y, r"$\bf{:.2f}°C$ decrease".format(change_Period), fontsize=fontsize, ha='right', va='top', transform=ax.transAxes)
        #
        if (gs<3):
            ax.set_xlabel('') #'Average daily maximum temperature (oC)', fontsize=12)
        else:
            ax.set_xlabel('Time (Years)', fontweight='bold', fontsize=12)
        ax.set_ylabel('') #'Grain Yield (t/ha)', fontsize=12)
        ax.tick_params(labelsize=8, color='gray', rotation=90)
        if (bxplot is False):
            ax.get_legend().remove()
        
        
    # ------------------------------
    # Add legend()
    def getLegend_HandlesLabels(ax, handout, lablout):
        handles, labels = ax.get_legend_handles_labels()
        for h,l in zip(handles,labels):
            if l not in lablout:
                lablout.append(l)
                handout.append(h)
        return handout, lablout
    # ------------------------------
    fig = plt.figure(figsize=(10, 8), tight_layout=True) #, constrained_layout=True) 
    rows = len(phases_labels)
    cols = 1
    
    gs0 = fig.add_gridspec(1, 2, hspace=0.1, wspace=0.35) #left=0.01, right=0.85, 
    gs_tmin = gs0[0].subgridspec(nrows=rows, ncols=cols)
    gs_tmax = gs0[1].subgridspec(nrows=rows, ncols=cols)
    #gs_tmin = gridspec.GridSpec(nrows=rows, ncols=cols) #, wspace=0, hspace=0)
    #gs_tmax = gridspec.GridSpec(nrows=rows, ncols=cols) #, wspace=0, hspace=0)
    
    handout=[]
    lablout=[]
    nfig = 0
    for nr in range(rows):
        for nc in range(cols):
            if (nfig>0):
                ax = fig.add_subplot(gs_tmin[nr, nc], sharey=ax, sharex=ax)
                ax2 = fig.add_subplot(gs_tmax[nr, nc], sharey=ax2, sharex=ax2)
            else:
                ax = fig.add_subplot(gs_tmin[nr, nc])
                ax2 = fig.add_subplot(gs_tmax[nr, nc])
            
            if (nfig<rows-1):
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax2.get_xticklabels(), visible=False)
                #plt.setp(ax.get_yticklabels(), visible=False)
            # Create figure
            createFigAcrossYrs(df_TMIN, ax=ax, gs=nfig, s=s, s2=s2, lw=lw, alpha=alpha) #, fldX='Year', s=20, lw=1.0, alpha=0.85)
            createFigAcrossYrs(df_TMAX, ax=ax2, gs=nfig, s=s, s2=s2, lw=lw, alpha=alpha) #, fldX='Year', s=20, lw=1.0, alpha=0.85)
            
            # Legend
            handout, lablout = getLegend_HandlesLabels(ax, handout, lablout)
            handout, lablout = getLegend_HandlesLabels(ax2, handout, lablout)
            nfig = nfig + 1
    
    #
    #fig.text(0.5, -0.01, 'Time (Years)', ha='center', va='center', fontweight='bold', fontsize=12)
    fig.text(0.08, 0.5, 'Average Daily Minimum Temperature (°C)', ha='center', va='center', 
             rotation='vertical', fontweight='bold', fontsize=12)
    fig.text(0.52, 0.5, 'Average Daily Maximum Temperature (°C)', ha='center', va='center', 
             rotation='vertical', fontweight='bold', fontsize=12)
    
    title1='{} - Temperature across years by growth stage ({} - {})'.format(nursery, df['CycleStartYr'].min(), df['CycleStartYr'].max())
    # Title for the complete figure
    fig.suptitle('{}'.format(title1), fontsize='x-large', fontweight='bold' ) #, fontsize=20, y=1.005)
    if (bxplot is False):
        fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.09), loc="lower center", ncol=4, 
                   borderaxespad=0,bbox_transform=plt.gcf().transFigure, fontsize=8) #, fancybox=True, shadow=True)
    
    fig.tight_layout()
    # Save in PDF
    hoy = datetime.now().strftime('%Y%m%d')
    figures_path = os.path.join(path_to_save_results, '{}_{}'.format(dirname, hoy))
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    if (saveFig is True and fmt=='pdf'):
        fig.savefig(os.path.join(figures_path,"{}_{}_{}.{}".format(nursery, fname, hoy, fmt)), 
                    bbox_inches='tight', orientation='portrait', papertype='a4', 
                    edgecolor='none', transparent=False, pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        fig.savefig(os.path.join(figures_path,"{}_{}_{}.{}".format(nursery, fname, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=300)
    
    if (showFig is True):
        fig.show()
    else:
        del fig
        plt.close();
        
#
def drawRegCoefficientbyGrowthStage_acrossYears(df_data=None, nursery='NURSERY', splitYear=2000, trendVar='Temperature', 
                                                s=20, s2=80, alpha=0.75, lw=1.5, bxplot=False,
                                                dirname = 'GrowthStages_Temperature', fname='TemperatureAcrossYrsxGrowthstage_v3',
                                                path_to_save_results='./', saveFig=False, showFig=True, fmt='pdf'):
    ''' 
        Average daily temperature across years vs mean grain yield by GID
    '''
    
    # filter
    df = df_data.copy()
    
    # Renanme some features to graph according to climate variables
    #if (temp=='TMIN'):
    df_TMIN = df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TMIN_mean':'Vegetative',
            'H-Hplus15d_TMIN_mean':'Heading',
            'Hplus15d-M_TMIN_mean':'Grain Filling',
            'Season_TMIN_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }) 
    df_TMAX = df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TMAX_mean':'Vegetative',
            'H-Hplus15d_TMAX_mean':'Heading',
            'Hplus15d-M_TMAX_mean':'Grain Filling',
            'Season_TMAX_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }) 
    df_TAVG = df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TAVG_mean':'Vegetative',
            'H-Hplus15d_TAVG_mean':'Heading',
            'Hplus15d-M_TAVG_mean':'Grain Filling',
            'Season_TAVG_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }) 
    
    sel_cols = ['GID', 'Year', 'Vegetative', 'Heading', 'Grain Filling', 'Season', 'Grain Yield', 'finalTarget', 'slope', 'intercept']
    df_TMIN = df_TMIN[sel_cols].sort_values(['Year']).reset_index(drop=True)
    df_TMAX = df_TMAX[sel_cols].sort_values(['Year']).reset_index(drop=True)
    df_TAVG = df_TAVG[sel_cols].sort_values(['Year']).reset_index(drop=True)
    
    # Scale linear regression coefficients
    #df_TMIN['slope'] = np.log(df_TMIN['slope'])
    #df_TMAX['slope'] = np.log(df_TMAX['slope'])
    #df_TAVG['slope'] = np.log(df_TAVG['slope'])
    
    # Growth Stages
    phases_labels = ['Vegetative', 'Heading', 'Grain Filling', 'Season'] #'Planting to Harvest'
    GScolors = {
        'Vegetative':'green',
        'Heading':'orange',
        'Grain Filling':'brown',
        'Season':'red'
    }
    # 
    # ------------------------------
    def createFigTrendAcrossYrs(df, ax, gs=0, splitYear=2000, fldX='Year', trendVar='Temperature', s=20, s2=80, lw=1.0, alpha=0.85):
        # Estimate reg. coefficients
        slopeFld = f'slope{trendVar}'
        df[slopeFld] = np.nan
        for gid in df['GID'].unique():
            x = df[(df['GID']==gid)][fldX].astype(int).to_numpy()
            if (trendVar=='Temperature'):
                y = df[(df['GID']==gid)][phases_labels[gs]].to_numpy()
            elif (trendVar=='Grain Yield'):
                y = df[(df['GID']==gid)]['Grain Yield'].to_numpy()
            # determine best fit line
            par = np.polyfit(x, y, 1, full=True)
            pend=par[0][0]
            intercept=par[0][1]
            #print("{} y = {:.7f}x + {:.7f}".format(phases_labels[0], pend, intercept))
            #change_Period = (pend*np.nanmax(x) + intercept) - (pend*np.nanmin(x) + intercept)
            #y_predicted = [pend*i + intercept  for i in x]
            df.loc[(df['GID']==gid), slopeFld] = pend
            
        # Estimate regression between years
        years = df['Year'].sort_values().unique()
        for yr in years:
            df_Yrs = df[((df['Year']==years[0]) | (df['Year']==yr))]
            x = df_Yrs[fldX].astype(int).to_numpy()
            y = df_Yrs[slopeFld].to_numpy()
            # determine best fit line
            par = np.polyfit(x, y, 1, full=True)
            pend=par[0][0]
            intercept=par[0][1]
            #print("{} y = {:.7f}x + {:.7f}".format(phases_labels[gs], pend, intercept))
            #change_Period = (pend*np.nanmax(x) + intercept) - (pend*np.nanmin(x) + intercept)
            #y_predicted = [pend*i + intercept  for i in x]
            df.loc[(df['Year']==yr), 'slopeYear'] = pend
        
        # TODO: obtener el mejor corte para determinar decremento o incremento de la variable a traves de los años
        # Por defecto toma splitYear=2000
        #df[['Year','slopeYear']].groupby(['Year'], as_index=False).agg('mean').head()
        
        g1 = sns.scatterplot(x=fldX, y=slopeFld, data=df, hue=None, marker='o', color=GScolors[phases_labels[gs]],
                                 s=s, lw=lw, alpha=0.35, ax=ax, label=phases_labels[gs])
        ax.axhline(0, ls='--', c='#444444', zorder=0, linewidth=1)

        # Add linear regressions from 1979 to splitYear=XXXXX
        df_Yrs = df[((df['Year']==years[0]) | (df['Year']==splitYear))] 
        x = df_Yrs[fldX].astype(int).to_numpy()
        y = df_Yrs[slopeFld].to_numpy()
        # determine best fit line
        par = np.polyfit(x, y, 1, full=True)
        pend=par[0][0]
        intercept=par[0][1]
        #print("{} y = {:.7f}x + {:.7f}".format(phases_labels[gs], pend, intercept))
        change_Period = (pend*np.nanmax(x) + intercept) - (pend*np.nanmin(x) + intercept)
        y_predicted = [pend*i + intercept  for i in x]
        l1 = sns.lineplot(x,y_predicted, color='black', ls='--', lw=1.8, label=f'{trendVar} trend (decreasing)', ax=ax)

        # Add linear regressions from splitYear to present
        df_Yrs = df[((df['Year']==splitYear) | (df['Year']==years[-1]))] # A partir del año XXXXX se nota las mejoras en variedades resistentes a altas temperaturas 
        x = df_Yrs[fldX].astype(int).to_numpy()
        y = df_Yrs[slopeFld].to_numpy()
        # determine best fit line
        par = np.polyfit(x, y, 1, full=True)
        pend=par[0][0]
        intercept=par[0][1]
        #print("{} y = {:.7f}x + {:.7f}".format(phases_labels[gs], pend, intercept))
        change_Period = (pend*np.nanmax(x) + intercept) - (pend*np.nanmin(x) + intercept)
        y_predicted = [pend*i + intercept  for i in x]
        l1 = sns.lineplot(x,y_predicted, color='blue', ls='-', lw=1.8, label=f'{trendVar} trend (increasing)', ax=ax)
        # Add coefficient text
        pos_x = 0.01
        pos_x2 = 0.98
        pos_y = 1.15 #0.2 # Top 0.95
        fontsize=10
        if (change_Period > 0):
            ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], 
                    ha='left', va='top', transform=ax.transAxes)
            if (trendVar=='Temperature'):
                ax.text(pos_x2, pos_y, r"$\bf{:.2f}°C$ increase".format(change_Period), fontsize=fontsize, 
                        ha='right', va='top', transform=ax.transAxes)
            elif (trendVar=='Grain Yield'):
                ax.text(pos_x2, pos_y, r"$\bf{:.2f} t/ha per °C$ increase".format(change_Period), fontsize=fontsize, 
                        ha='right', va='top', transform=ax.transAxes)
        else:
            ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], 
                    ha='left', va='top', transform=ax.transAxes)
            if (trendVar=='Temperature'):
                ax.text(pos_x2, pos_y, r"$\bf{:.2f}°C$ decrease".format(change_Period), fontsize=fontsize, 
                        ha='right', va='top', transform=ax.transAxes)
            elif (trendVar=='Grain Yield'):
                ax.text(pos_x2, pos_y, r"$\bf{:.2f} t/ha per °C$ decrease".format(change_Period), fontsize=fontsize, 
                        ha='right', va='top', transform=ax.transAxes)
        #
        g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax.set_axisbelow(True)
        if (gs<3):
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Time (Years)', fontweight='bold', fontsize=12)
        ax.set_ylabel('')
        ax.tick_params(labelsize=8, color='gray', rotation=90)
        if (bxplot is False):
            ax.get_legend().remove()
        
    # ------------------------------
    def createFigAcrossYrs(df, ax, gs=0, fldX='Year', s=20, s2=80, lw=1.0, alpha=0.85, bxplot=bxplot):
        # Box plots
        if (bxplot is True):
            g1 = sns.boxplot(x=fldX, y=df[phases_labels[gs]], data=df, hue=None, ax=ax, 
                             flierprops={"marker": "x", "markersize":5, "markeredgecolor":"lightgray" },
                             boxprops={"facecolor": (.4, .6, .8, .5)}, medianprops={"color": GScolors[phases_labels[gs]]});
        else: 
            g1 = sns.scatterplot(x=fldX, y=df[phases_labels[gs]], data=df, hue=None, marker='o', color=GScolors[phases_labels[gs]],
                                 s=s, lw=lw, alpha=alpha, ax=ax, label=phases_labels[gs])
        g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax.set_axisbelow(True)
        
        if (bxplot is False):
            df2 = df[df['finalTarget']==1]
            g2 = sns.scatterplot(x=fldX, y=df2[phases_labels[gs]], data=df2, 
                                 alpha=0.95, ec=GScolors[phases_labels[gs]], fc='none', marker="o", 
                                 s=s2, lw=0.35,  ax=ax, label=f'Selected observations ({phases_labels[gs]})')
        # Add linear regression
        x = df[fldX].astype(int).to_numpy()
        y = df[phases_labels[gs]].to_numpy()
        # determine best fit line
        par = np.polyfit(x, y, 1, full=True)
        pend=par[0][0]
        intercept=par[0][1]
        #print("{} y = {:.7f}x + {:.7f}".format(phases_labels[0], pend, intercept))
        change_Period = (pend*np.nanmax(x) + intercept) - (pend*np.nanmin(x) + intercept)
        #print(f"{phases_labels[0]} change over time: {change_Period:.2f} °C")
        y_predicted = [pend*i + intercept  for i in x]
        if (bxplot is False):
            l1 = sns.lineplot(x,y_predicted, color='black', ls='-', lw=0.35, label=f'Temperature trend', ax=ax)
        pos_x = 0.01
        pos_x2 = 0.98
        pos_y = 1.15 #0.2 # Top 0.95
        fontsize=10
        if (change_Period > 0):
            ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], 
                    ha='left', va='top', transform=ax.transAxes)
            ax.text(pos_x2, pos_y, r"$\bf{:.2f}°C$ increase".format(change_Period), fontsize=fontsize, ha='right', va='top', transform=ax.transAxes)
        else:
            ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], 
                    ha='left', va='top', transform=ax.transAxes)
            ax.text(pos_x2, pos_y, r"$\bf{:.2f}°C$ decrease".format(change_Period), fontsize=fontsize, ha='right', va='top', transform=ax.transAxes)
        #
        if (gs<3):
            ax.set_xlabel('') #'Average daily maximum temperature (oC)', fontsize=12)
        else:
            ax.set_xlabel('Time (Years)', fontweight='bold', fontsize=12)
        ax.set_ylabel('') #'Grain Yield (t/ha)', fontsize=12)
        ax.tick_params(labelsize=8, color='gray', rotation=90)
        if (bxplot is False):
            ax.get_legend().remove()
        
        
    # ------------------------------
    # Add legend()
    def getLegend_HandlesLabels(ax, handout, lablout):
        handles, labels = ax.get_legend_handles_labels()
        for h,l in zip(handles,labels):
            if l not in lablout:
                lablout.append(l)
                handout.append(h)
        return handout, lablout
    # ------------------------------
    fig = plt.figure(figsize=(10, 8), tight_layout=True) #, constrained_layout=True) 
    rows = len(phases_labels)
    cols = 1
    
    gs0 = fig.add_gridspec(1, 2, hspace=0.1, wspace=0.35) #left=0.01, right=0.85, 
    gs_tmin = gs0[0].subgridspec(nrows=rows, ncols=cols)
    gs_tmax = gs0[1].subgridspec(nrows=rows, ncols=cols)
    #gs_tmin = gridspec.GridSpec(nrows=rows, ncols=cols) #, wspace=0, hspace=0)
    #gs_tmax = gridspec.GridSpec(nrows=rows, ncols=cols) #, wspace=0, hspace=0)
    
    handout=[]
    lablout=[]
    nfig = 0
    for nr in range(rows):
        for nc in range(cols):
            if (nfig>0):
                ax = fig.add_subplot(gs_tmin[nr, nc], sharey=ax, sharex=ax)
                ax2 = fig.add_subplot(gs_tmax[nr, nc]) #, sharey=ax2, sharex=ax2)
            else:
                ax = fig.add_subplot(gs_tmin[nr, nc])
                ax2 = fig.add_subplot(gs_tmax[nr, nc])
            
            if (nfig<rows-1):
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax2.get_xticklabels(), visible=False)
                #plt.setp(ax.get_yticklabels(), visible=False)
            # Create figure
            createFigAcrossYrs(df_TMAX, ax=ax, gs=nfig, s=s, s2=s2, lw=lw, alpha=alpha) #, fldX='Year', s=20, lw=1.0, alpha=0.85)
            createFigTrendAcrossYrs(df_TMAX, ax=ax2, splitYear=splitYear, trendVar=trendVar, gs=nfig, s=s, s2=s2, lw=lw, alpha=alpha) #, fldX='Year', s=20, lw=1.0, alpha=0.85)
            
            # Legend
            handout, lablout = getLegend_HandlesLabels(ax, handout, lablout)
            handout, lablout = getLegend_HandlesLabels(ax2, handout, lablout)
            nfig = nfig + 1
    
    #
    #fig.text(0.5, -0.01, 'Time (Years)', ha='center', va='center', fontweight='bold', fontsize=12)
    fig.text(0.08, 0.5, 'Average Daily Maximum Temperature (°C)', ha='center', va='center', 
             rotation='vertical', fontweight='bold', fontsize=12)
    fig.text(0.52, 0.5, 'Linear Regression Coefficients', ha='center', va='center', 
             rotation='vertical', fontweight='bold', fontsize=12)
    #if (trendVar=='Temperature'):
    #    fig.text(0.52, 0.5, 'Linear Regression Coefficients of Maximum Temperature', ha='center', va='center', 
    #             rotation='vertical', fontweight='bold', fontsize=12)
    #elif (trendVar=='Grain Yield'):
    #    fig.text(0.52, 0.5, 'Linear Regression Coefficients of Grain Yield', ha='center', va='center', 
    #             rotation='vertical', fontweight='bold', fontsize=12)
    
    title1='{} - Temperature across years by growth stage ({} - {})'.format(nursery, df['CycleStartYr'].min(), df['CycleStartYr'].max())
    # Title for the complete figure
    fig.suptitle('{}'.format(title1), fontsize='x-large', fontweight='bold' ) #, fontsize=20, y=1.005)
    if (bxplot is False):
        fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.09), loc="lower center", ncol=4, 
                   borderaxespad=0,bbox_transform=plt.gcf().transFigure, fontsize=8) #, fancybox=True, shadow=True)
    
    hoy = datetime.now().strftime('%Y%m%d')

    fig.tight_layout()
    #fig.tight_layout(rect=(0.08,0.05,1,0.985)) # Left, bottom, right, top
    # Save in PDF
    hoy = datetime.now().strftime('%Y%m%d')
    figures_path = os.path.join(path_to_save_results, '{}_{}'.format(dirname, hoy))
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    if (saveFig is True and fmt=='pdf'):
        fig.savefig(os.path.join(figures_path,figures_path,"{}_{}_{}.{}".format(nursery, fname, hoy, fmt)), 
                    bbox_inches='tight', orientation='portrait', papertype='a4', 
                    edgecolor='none', transparent=False, pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        fig.savefig(os.path.join(figures_path,"{}_{}_{}.{}".format(nursery, fname, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=300)
    
    if (showFig is True):
        fig.show()
    else:
        del fig
        plt.close();

#
def drawCompareGYvsTMAXbyGS(df_data=None, GIDs2compare=None, nursery='NURSERY', s=10, s2=30, alpha=0.65, lw=1.25, 
                            dirname = 'GrowthStages_Temperature', fname='CompareGYvsTMAXbyGS',
                            path_to_save_results='./', saveFig=False, showFig=True, fmt='pdf'):
    ''' 
        Comparison of Genotypes vs Maximum Temperature by growth stage
        display the average daily temperature across years vs mean grain yield by selected GID
    '''
    if (GIDs2compare is None):
        print("Define a list of GIDs to compare before continuing...")
        return
    # filter
    df = df_data.copy()
    
    # Renanme some features to graph according to climate variables
    df_TMAX = df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TMAX_mean':'Vegetative',
            'H-Hplus15d_TMAX_mean':'Heading',
            'Hplus15d-M_TMAX_mean':'Grain Filling',
            'Season_TMAX_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }) 
    sel_cols = ['GID', 'Year', 'Vegetative', 'Heading', 'Grain Filling', 'Season', 'Grain Yield', 'finalTarget']
    df_TMAX = df_TMAX[sel_cols].sort_values(['Year']).reset_index(drop=True)
    
    # Growth Stages
    phases_labels = ['Vegetative', 'Heading', 'Grain Filling', 'Season'] #'Planting to Harvest'
    GScolors = {
        'Vegetative':'green',
        'Heading':'orange',
        'Grain Filling':'brown',
        'Season':'red'
    }
    
    fig = plt.figure(figsize=(10, 8), tight_layout=True) #, constrained_layout=True) 
    rows = len(phases_labels)
    cols = len(GIDs2compare)
    gs = gridspec.GridSpec(nrows=rows, ncols=cols) #, wspace=0.0, hspace=0.0)
    
    # ------------------------------
    def createFigCompareGYvsTMAXbyGS(df, gid, ngid, ax, gs=0, s=10, s2=30, lw=1.0, alpha=0.85):
        g1 = sns.scatterplot(x=df[phases_labels[gs]], y='Grain Yield', data=df, hue=None, marker='o', color=GScolors[phases_labels[gs]],
                                 s=s, lw=lw, alpha=alpha, ax=ax, label=phases_labels[gs])
        g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel('')
        ax.set_ylabel('')
        #ax.tick_params(labelsize=8, color='gray', rotation=90)
        #ax.set_title(r"$\bf {}:$ $\bf{:.2f}°C$ increase".format(phases_labels[gs], change_Period), fontweight='bold', fontsize=fontsize, color=GScolors[phases_labels[gs]])
        #
        # Add linear regression
        x = df[phases_labels[gs]].astype(int).to_numpy()
        y = df['Grain Yield'].to_numpy()
        # determine best fit line
        par = np.polyfit(x, y, 1, full=True)
        pend=par[0][0]
        intercept=par[0][1]
        #print("{} y = {:.7f}x + {:.7f}".format(phases_labels[0], pend, intercept))
        change_Period = (pend*np.nanmax(x) + intercept) - (pend*np.nanmin(x) + intercept)
        #print(f"{phases_labels[0]} change over time: {change_Period:.2f} °C")
        y_predicted = [pend*i + intercept  for i in x]
        l1 = sns.lineplot(x,y_predicted, color=GScolors[phases_labels[gs]], ls='-', lw=lw, ax=ax) #label=f'Temperature trend ({phases_labels[gs]})', 
        #l1 = sns.regplot(x,y_predicted, color=GScolors[phases_labels[gs]], label=f'{phases_labels[gs]}', ax=ax)
        pos_x = 0.01
        pos_x2 = 0.98
        pos_y = 0.98 # Top 1.15 #
        fontsize=10
        if (change_Period > 0):
            #ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], ha='left', va='top', transform=ax.transAxes)
            ax.text(pos_x, pos_y, r"$\bf + {:.2f}$ tha per °C".format(change_Period), fontsize=fontsize, color='#444444', ha='left', va='top', transform=ax.transAxes) #, zorder=0)
        else:
            #ax.text(pos_x, pos_y, r"$\bf {}$".format(phases_labels[gs]), fontsize=fontsize, c=GScolors[phases_labels[gs]], ha='left', va='top', transform=ax.transAxes)
            ax.text(pos_x, pos_y, r"$\bf {:.2f}$ tha per °C".format(change_Period), fontsize=fontsize, color='#444444', ha='left', va='top', transform=ax.transAxes) #, zorder=0)
        
        ax.get_legend().remove()
        
    # ------------------------------
    # Add legend()
    def getLegend_HandlesLabels(ax, handout, lablout):
        handles, labels = ax.get_legend_handles_labels()
        for h,l in zip(handles,labels):
            if l not in lablout:
                lablout.append(l)
                handout.append(h)
        return handout, lablout
    # ------------------------------
    
    handout=[]
    lablout=[]
    for i, gid in enumerate(GIDs2compare):
        nfig = 0
        if (gid in df_TMAX['GID'].unique()):
            df_GID = df_TMAX[df_TMAX['GID']==gid] #GIDs2compare[i]]
        else:
            df_GID = None
            print("GID {} not found!".format(gid))
            continue
        gidMinYr = df_GID['Year'].min()
        for nr in range(rows):
            for nc in range(1): #cols):
                if (nfig>0):
                    ax = fig.add_subplot(gs[nr, nc+i], sharey=ax, sharex=ax)
                else:
                    ax = fig.add_subplot(gs[nr, nc+i])
                plt.setp(ax.get_xticklabels(), fontsize=8)
                plt.setp(ax.get_yticklabels(), fontsize=8)
                #if (nfig<rows-1):
                #    plt.setp(ax.get_xticklabels(), visible=False)
                #    #ax.set(xticks=[])
                #if (i>0):
                #    plt.setp(ax.get_yticklabels(), visible=False)
                #    #ax.set(yticks=[])
                # Create figure
                createFigCompareGYvsTMAXbyGS(df_GID, gid=gid, ngid=i, ax=ax, gs=nfig, s=s, s2=s2, lw=lw, alpha=alpha) 
                #ax.set_title(f"{gid} - {i}")
                if (nfig==0):
                    ax.set_title(f"GID-{gid} from {gidMinYr}", fontsize=12, fontweight='bold')
                # Legend
                handout, lablout = getLegend_HandlesLabels(ax, handout, lablout)
                nfig = nfig + 1

    #
    fig.text(0.5, -0.01, 'Average Daily Maximum Temperature (°C)', ha='center', va='center', fontweight='bold', fontsize=12)
    fig.text(-0.02, 0.5, 'Grain Yield (t/ha)', ha='center', va='center', 
             rotation='vertical', fontweight='bold', fontsize=12)

    #title1='NURSERY - Comparison of Genotypes vs Maximum Temperature by growth stage ({} - {})'.format(df['CycleStartYr'].min(), df['CycleStartYr'].max())
    title1='{} - Comparison of Genotypes vs Maximum Temperature by growth stage'.format(nursery)
    # Title for the complete figure
    fig.suptitle('{}'.format(title1), fontsize='x-large', fontweight='bold' ) #, fontsize=20, y=1.005)
    fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.09), loc="lower center", ncol=4, 
               borderaxespad=0,bbox_transform=plt.gcf().transFigure, fontsize=10) #, fancybox=True, shadow=True)

    fig.tight_layout()
    # Save in PDF
    hoy = datetime.now().strftime('%Y%m%d')
    figures_path = os.path.join(path_to_save_results, '{}_{}'.format(dirname, hoy))
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    if (saveFig is True and fmt=='pdf'):
        fig.savefig(os.path.join(figures_path,figures_path,"{}_{}_{}.{}".format(nursery, fname, hoy, fmt)), 
                    bbox_inches='tight', orientation='portrait', papertype='a4', 
                    edgecolor='none', transparent=False, pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        fig.savefig(os.path.join(figures_path,figures_path,"{}_{}_{}.{}".format(nursery, fname, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=300)

    if (showFig is True):
        fig.show()
    else:
        del fig
        plt.close();
        
#
def ranking_selectedGenotypes(df_final):
    ''' Ranking Selected Genotypes '''
    
    # Growth Stages
    phases_labels = ['Vegetative', 'Heading', 'Grain Filling', 'Season'] #'Planting to Harvest'
    GScolors = {
        'Vegetative':'green',
        'Heading':'orange',
        'Grain Filling':'brown',
        'Season':'red'
    }
    df = df_final.copy() #df_finalTarget_updated # Se debe usar es df_final porque contiene todos las observaciones por GID
    df_GY = df.rename(columns={
            'CycleStartYr': 'Year',
            'S-H_TMAX_mean':'Vegetative',
            'H-Hplus15d_TMAX_mean':'Heading',
            'Hplus15d-M_TMAX_mean':'Grain Filling',
            'Season_TMAX_mean':'Season',
            'GRAIN_YIELD_BLUEs': 'Grain Yield'
        }) 
    sel_cols = ['GID', 'Year', 'Vegetative', 'Heading', 'Grain Filling', 'Season', 'Grain Yield', 'finalTarget']
    df_GY = df_GY[sel_cols].reset_index(drop=True)

    df_GY['slopeVegetative'] = np.nan
    df_GY['slopeHeading'] = np.nan
    df_GY['slopeGrainFilling'] = np.nan
    df_GY['slopeSeason'] = np.nan
    for gs, gsname in enumerate(phases_labels):
        for gid in df_GY['GID'].unique():
            x = df_GY[(df_GY['GID']==gid)][phases_labels[gs]].astype(int).to_numpy()
            y = df_GY[(df_GY['GID']==gid)]['Grain Yield'].to_numpy()
            # determine best fit line
            par = np.polyfit(x, y, 1, full=True)
            pend=par[0][0]
            intercept=par[0][1]
            #print("{} y = {:.7f}x + {:.7f}".format(phases_labels[0], pend, intercept))
            change_Period = (pend*np.nanmax(x) + intercept) - (pend*np.nanmin(x) + intercept)
            y_predicted = [pend*i + intercept  for i in x]
            #df_GY.loc[((df_GY['Nursery_Yr']==nyr) & (df_GY['GID']==gid)), 'slopeGY'] = slope
            df_GY.loc[(df_GY['GID']==gid), 'slope{}'.format(gsname.replace(' ',''))] = pend
            #df_GY.loc[(df_GY['GID']==gid), f'intercept{gsname}'] = intercept

    df_GY_sorted = df_GY.sort_values(['slopeSeason', 'Grain Yield','slopeVegetative','slopeHeading',  
                                      'slopeGrainFilling'], ascending=False)
    df_GY_sorted.reset_index(drop=True, inplace=True)
    df_GY_sorted['Ranking'] = sorted(np.array(df_GY_sorted.index+1), reverse=False)
    return df_GY_sorted

#
def getVarietyCounts(df, top=5):
    ''' Count the number of site-year trials using a cultivar '''
    df_finalTarget_updated = df.copy()
    df_finalTarget_updated.dropna(subset=['Gen_name'], inplace=True)
    sublist = []
    for p in df_finalTarget_updated['Gen_name'].unique():
        sublist.append(p.split('//'))
    sublist2 = [item.split('/') for p in sublist for item in p]
    final_list = [item for p in sublist2 for item in p if not item.isdigit()]
    final_list = pd.DataFrame.from_dict({"count": dict(Counter(final_list))}).reset_index()\
    .rename(columns={'index':'Variety'}).sort_values('count', ascending=False)
    return final_list.reset_index(drop=True)[:top]

def searchParenLines(df_raw, df_GY_sorted, top=50):
    ''' Identifying parents lines '''
    # Recover the genotype name or pedigree from the raw dataset
    df_Pedigrees = df_raw.groupby(['GID', 'Gen_name'], as_index=False).agg('count')[['GID', 'Gen_name', 'UID']].rename(columns={'UID':'Count'})\
    .sort_values(['Count'], ascending=False).reset_index(drop=True)
    # df_GY_sorted_finalTarget = df_GY_sorted[df_GY_sorted['finalTarget']==1]
    df_final_Pedigrees = pd.merge(df_GY_sorted[df_GY_sorted['finalTarget']==1], 
                                  df_Pedigrees[['GID', 'Gen_name']], how='left', on='GID').sort_values(['Ranking'], ascending=True)
    # Select only the GID that support high temperatures during the whole growing season
    print("Final selected GIDs that perform well in low yielding environments and support high temperatures\n",
          df_final_Pedigrees[df_final_Pedigrees['slopeSeason']>0]['GID'].unique())
    df_final_Pedigrees[df_final_Pedigrees['slopeSeason']>0]
    #print(df_final_Pedigrees[df_final_Pedigrees['slopeSeason']>0]['Gen_name'].unique())

    # Display figures top 5
    #for gid in df_final_Pedigrees[df_final_Pedigrees['slopeSeason']>0]['GID'].unique()[:5]:
    #    displayGenotypeSensitivitybyWeather(df_final_ESWYT, gid=gid, path_to_save_results=RESULTS_IWIN_PATH, saveFig=False ) #yFld='Grain Yield', 

    # 
    print("Number of observations where parent lines were tested:\n ",df_raw[df_raw['GID'].isin(
        df_final_Pedigrees[df_final_Pedigrees['slopeSeason']>0]['GID'].unique())][['Gen_name']].value_counts())

    # Extract Grand Parent
    df_final_Pedigrees['GrandParent'] = df_final_Pedigrees['Gen_name'].apply(lambda x: str(x).split('//')[0])

    # ---------------------------
    # Top Pedigrees used in the selected Genotypes
    # ---------------------------
    f_selGen = df_final_Pedigrees[df_final_Pedigrees['slopeSeason']>0].copy() # Only those supporting high temperature
    f_selGen['GID2'] = f_selGen['GID']
    f_selGen = f_selGen.groupby(['GID','Gen_name'], as_index=False).agg({'GID2':'count'})\
    .rename(columns={'GID2':'Count'})\
    .sort_values(by=['Count'], ascending=False).reset_index(drop=True)
    # display results
    display(HTML(f_selGen.drop_duplicates(subset=['GID', 'Gen_name']).to_html()))
    
    print("Varieties above average in low yielding environments:")
    display(HTML(getVarietyCounts(df=df_final_Pedigrees, top=5).to_html()))
    print("")
    print("Varieties above average in low yielding and extreme climate environments:")
    df_final_selected_varieties = getVarietyCounts(df=df_final_Pedigrees[df_final_Pedigrees['slopeSeason']>0], top=top)
    display(HTML(getVarietyCounts(df=df_final_Pedigrees[df_final_Pedigrees['slopeSeason']>0], top=5).to_html()))
    return df_final_selected_varieties


# -------------------------------
# Run above processes in one step
# -------------------------------
def seachWheatLinesInExtremeWeather(df_raw, df, n='ESWYT', AVGHT_THRESHOLD=30.0, GT32C_NDAYS_THRESHOLD=10, splitYear=2005, top=30,
                                    path_to_save_results='./', s=10, s2=30, drawFigures=False, showFig=True, verbose=True):
    '''
        Run all processes in one step to identify best pedigrees in low yielding environments and extreme weather.
        
        
    '''
    df_GY_sorted_finalTarget, df_final_selected_varieties = None, None
    # Filter by nursery
    df_filtered = df[df['Nursery']==n]
    # -----------------------------------------------------------
    # 1. Identify target GIDs for each nursery-year
    # -----------------------------------------------------------
    try:
        df_final, GIDs_inTarget_1, GIDs_inTarget_highTemperature, GIDs_LowestHighest_NormilizedYield = \
        idenfifyTargetGIDs_byNurseryYear(df_filtered=df_filtered, nursery=n, 
                                         path_to_save_results=path_to_save_results, 
                                         drawFigures=drawFigures, verbose=verbose)
    except Exception as err:
        print("Problem identifying target GIDs for each nursery-year. Error:", err)

    # -----------------------------------------------------------
    # 2. Filter data using thresholds for Maximum Temperature
    # -----------------------------------------------------------
    try:
        df_final, df_map_GIDs_highTemperature = filterTargetGIDs_byThresholdMaxTemperature(df_filtered=df_final, 
                                                    GIDs_inTarget_highTemperature=GIDs_inTarget_highTemperature, nursery=n, 
                                                    avgLT_threshold=10.0, avgHT_threshold=AVGHT_THRESHOLD, 
                                                    gt32C_ndays_threshold=GT32C_NDAYS_THRESHOLD,
                                                    path_to_save_results=path_to_save_results, saveFiles=True, verbose=verbose)
    except Exception as err:
        print("Problem filtering data using thresholds for Maximum Temperature. Error:", err)

    # -----------------------------------------------------------
    # 3. Spatial distribution of selected IWIN genotypes
    # -----------------------------------------------------------
    try:
        displayMap_spatialDistSelectedGIDs(df_gids=df_map_GIDs_highTemperature, nursery=n, 
                                           path_to_save_results=path_to_save_results, saveFig=True)
    except Exception as err:
        print("Problem displaying spatial distribution of selected IWIN genotypes. Error:", err)

    # -----------------------------------------------------------
    # 4. Climate x Growth stage in target genotypes
    # -----------------------------------------------------------
    try:
        #displayGenotypeSensitivitybyWeather(df_final, gid=2430154, path_to_save_results=RESULTS_IWIN_PATH, saveFig=False ) #yFld='Grain Yield',
        # 4.1 Temperature vs Grain Yield in 3 growth stages (all target GIDs)
        displayTempVsGY_inGrowStagesforAllGIDs(df_filtered=df_final, df_HT=df_map_GIDs_highTemperature, 
                                               nursery=n, path_to_save_results=path_to_save_results, saveFig=True )
    except Exception as err:
        print("Problem displaying emperature vs Grain Yield in 3 growth stages for all target GIDs. Error:", err)
        
    # -----------------------------------------------------------
    # 5. Average daily temperature across years vs mean grain yield by GID
    # -----------------------------------------------------------
    try:
        #drawTemperaturebyGrowthStage_acrossYears_v2(df_data=df_final, nursery=n, s=s, s2=s2, alpha=0.65, lw=0.5, bxplot=False,
        #                                             dirname = 'GrowthStages_Temperature', fname=f'TemperatureAcrossYrsxGrowthstage_v2',
        #                                             path_to_save_results=path_to_save_results, saveFig=True, showFig=showFig, fmt='pdf')

        drawTemperaturebyGrowthStage_acrossYears_v2(df_data=df_final, nursery=n, s=s, s2=s2, alpha=0.65, lw=0.5, bxplot=False,
                                                     dirname = 'GrowthStages_Temperature', fname=f'TemperatureAcrossYrsxGrowthstage_v2',
                                                     path_to_save_results=path_to_save_results, saveFig=True, showFig=showFig, fmt='jpg')
    except Exception as err:
        print("Problem displaying Average daily temperature across years vs mean grain yield by GID. Error:", err)
        
    # -----------------------------------------------------------
    # 6. Genetic Gain across years
    # -----------------------------------------------------------
    try:
        #drawRegCoefficientbyGrowthStage_acrossYears(df_data=df_final, nursery=n, splitYear=splitYear, trendVar='Temperature', 
        #                                            s=s, s2=s2, alpha=0.65, lw=0.5, bxplot=False,
        #                                            dirname = 'GrowthStages_Temperature', fname=f'TemperatureAcrossYrsxGrowthstage_regcoeff_trend',
        #                                            path_to_save_results=path_to_save_results, saveFig=True, showFig=showFig, fmt='pdf')

        drawRegCoefficientbyGrowthStage_acrossYears(df_data=df_final, nursery=n, splitYear=splitYear, trendVar='Temperature', 
                                                    s=s, s2=s2, alpha=0.65, lw=0.5, bxplot=False,
                                                    dirname = 'GrowthStages_Temperature', fname=f'TemperatureAcrossYrsxGrowthstage_regcoeff_trend',
                                                    path_to_save_results=path_to_save_results, saveFig=True, showFig=showFig, fmt='jpg')
    except Exception as err:
        print("Problem displaying Genetic Gain across years. Error:", err)
    
    # -----------------------------------------------------------
    # 7. Comparison of Genotypes vs Maximum Temperature by growth stage
    # -----------------------------------------------------------
    #GIDs2compare = [ 3822784, 2430154, 7400458, 4755402 ] # Support of High Temperature (eg. decrease, stable, increase)
    #drawCompareGYvsTMAXbyGS(df_data=df_final, GIDs2compare=GIDs2compare, nursery=n, s=s, s2=s2, alpha=0.65, lw=1.25,
    #                        dirname = 'GrowthStages_Temperature', fname=f'CompareGYvsTMAXbyGS',
    #                        path_to_save_results=path_to_save_results, saveFig=True, showFig=showFig, fmt='pdf')
    #
    #drawCompareGYvsTMAXbyGS(df_data=df_final, GIDs2compare=GIDs2compare, nursery=n, s=s, s2=s2, alpha=0.65, lw=1.25,
    #                        dirname = 'GrowthStages_Temperature', fname=f'CompareGYvsTMAXbyGS',
    #                        path_to_save_results=path_to_save_results, saveFig=True, showFig=False, fmt='jpg')


    # -----------------------------------------------------------
    # 8. Ranking Selected Genotypes
    # Individual wheat lines that outperform under extreme weather (high temperature) events 
    # relative to average performance in international nurseries
    # -----------------------------------------------------------
    try:
        df_GY_sorted = ranking_selectedGenotypes(df_final)
    except Exception as err:
        print("Problem Ranking selected genotypes. Error:", err)
    # -----------------------------------------------------------
    # 9. Top 2 and Bottom 2
    # -----------------------------------------------------------
    try:
        df_GY_sorted_finalTarget = df_GY_sorted[df_GY_sorted['finalTarget']==1]
        df_GY_sorted_finalTarget = df_GY_sorted_finalTarget[['GID']].drop_duplicates()
        GIDs2compare = list(df_GY_sorted_finalTarget['GID'][:2]) + list(df_GY_sorted_finalTarget['GID'][-2:])
        #drawCompareGYvsTMAXbyGS(df_data=df_final, GIDs2compare=GIDs2compare, nursery=n, s=s, s2=s2, alpha=0.65, lw=1.25,
        #                        dirname = 'GrowthStages_Temperature', fname=f'finalTargetGIDs_v2',
        #                        path_to_save_results=path_to_save_results, saveFig=True, showFig=True, fmt='pdf')

        drawCompareGYvsTMAXbyGS(df_data=df_final, GIDs2compare=GIDs2compare, nursery=n, s=s, s2=s2, alpha=0.65, lw=1.25,
                                dirname = 'GrowthStages_Temperature', fname=f'finalTargetGIDs_v2',
                                path_to_save_results=path_to_save_results, saveFig=True, showFig=showFig, fmt='jpg')
    except Exception as err:
        print("Problem displaying Top 2 and Bottom 2 of selected genotypes. Error:", err)
        
    # -----------------------------------------------------------
    # 10. Identifying parents lines
    # Trials using these varieties have more probabilities to support high temperatures 
    # in low yielding environments preserving the genetic gain across years.
    # -----------------------------------------------------------
    try:
        df_final_selected_varieties = searchParenLines(df_raw=df_raw, df_GY_sorted=df_GY_sorted, top=top)
    except Exception as err:
        print("Problem identifying parents lines. Error:", err)
    
    return df_GY_sorted_finalTarget, df_final_selected_varieties



# 










