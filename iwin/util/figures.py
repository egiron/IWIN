# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

from . import getScores

import os, gc
import re
import numpy as np
import pandas as pd
from datetime import date, datetime
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import (LinearRegression, RANSACRegressor)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, pairwise_distances_argmin_min


# GLOBALS
nursery=['ESWYT','IDYN','SAWYT','HTWYT']
colors=['#4203c9', '#16acea', '#e89f1e','#d71b6b']
paleta = {
    'ESWYT':'#4203c9','IDYN':'#16acea','SAWYT':'#e89f1e','HTWYT':'#d71b6b'
}

def chart_compareResults(df_result=None, fld1=None, fld2=None, alpha=.75, s=15, xy_lim=2, hue=None, 
                         loc_leg=2, ncol=2, ha='left', va='top',
                         title='', xlabel='', ylabel='', dirname='RUNS', fname='iPAR_model_', 
                         dispScore=True, dispLegend=True, saveFig=False, showFig=True, fmt='pdf'):
    '''
    Display a scatter plot to compare two variables in the results
    
    Parameters
    ----------
    :params df_result: A pandas DataFrame with the results and variables to compare
    :params fld1: Variable or column name to compare
    :params fld2: Variable or column name to compare
    :params alpha: Transparency of the points in chart
    :params s: Size of the points in chart
    :params xy_lim: Used to extend the x-axis limit. Default 2 units
    :params hue: Variable to classify or discriminate the results in colors
    :params title: Title of the figure
    :params xlabel: Label of the x-axis
    :params ylabel: Label of the y-axis
    :params dirname: Folder name to save results
    :params fname: File name to save the figure
    :params dispScore: Display the accurracy and others stats of the model
    :params dispLegend: Display the legend of the chart
    :params saveFig: Save file in JPG or PDF format
    :params fmt: Format of the output
    
    Returns
    -------
    :results: A figure in JPG or PDF format with the filename specified into the folder name 
    
    '''
    if (df_result is None):
        print("Input data not valid")
        return
    if (fld1 is None or fld2 is None):
        print("Variable are not valid")
        return
    
    df = df_result.copy()
    df.dropna(subset=[fld1, fld2], inplace=True)
    
    r2score, rmse, n_rmse, d_index, accuracy = getScores(df, fld1=fld1, fld2=fld2)
    fig, (ax1) = plt.subplots(figsize=(10,6))
    fig.subplots_adjust(right=0.55)
    g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, color="#000000", hue=hue, s=s, lw=0, ax=ax1);
    ax1.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df[fld1].max(), df[fld2].max())) + xy_lim
    g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title('iPAR Yield model - IWIN\n{}'.format(title), fontsize=15)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.tick_params(labelsize=12)
    if (dispScore==True):
        ax1.text(0.05,0.96,'Observations:{}\nRMSE:{:.1f}\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}'.format(len(df), rmse, n_rmse, d_index, r2score), fontsize=10, ha=ha, va=va, transform=ax1.transAxes)

    if (dispLegend==True):
        plt.legend(bbox_to_anchor=(1.05, 1), loc=loc_leg, ncol=ncol, borderaxespad=0)
    else:
        #plt.legend.remove()
        ax1.get_legend().remove()
    
    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"{}_{}_{}.pdf".format(fname, title.replace(' ',''), hoy)), 
                    bbox_inches='tight', orientation='portrait',  pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        #figures_path = os.path.join(config['RESULTS_PATH'] , '{}_{}'.format(dirname, hoy) )
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"{}_{}_{}.{}".format(fname, title.replace(' ',''), hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, dpi=300)
    
    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
    
    
def chart_compareYieldResults(df_result=None, title="iPAR Yield model", alpha=.95, s=25, xy_lim=2, 
                              hue='country', loc_leg=2, ncol=4, xt_tl=.03, yt_tl=.99, ha='left', va='top',
                              dirname='Figures', fname='Figure',
                              dispScore=True, saveFig=True, showFig=True, sharex=False, sharey=False, fmt='pdf'):
    '''
        Create a figure to compare several iPAR model simulations results of grain yield
        
        :params df_result: A table or DataFrame with the needed data
        :params title: Name of the figure
        :params alpha: Transparency of the points in chart
        :params s: Size of the points in chart
        :params xy_lim: Used to extend the x-axis limit. Default 2 units
        :params hue: Variable to classify or discriminate the results in colors
        :params ncol: Number of columns to split the legend
        :params dirname: Folder name to save results
        :params fname: File name to save the figure
        :params saveFig: Save file or export the figure
        :params sharex: Share or align all x-axis, it will use the same x scale for all charts if the value is True
        :params sharey: Share or align all y-axis, it will use the same y scale for all charts if the value is True
        :params fmt: Format of the output (PNG, JPG or PDF)
        
        :results: An integrated figure with 4 charts showing result simulations
        
    '''
    if (df_result is None):
        print("Input data is not valid")
        return
    # Figure 1 - Observed phenology
    fld1="ObsYield"
    fld2="SimYield"
    df = df_result.copy()
    df.dropna(subset=[fld1, fld2], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Figure 2 - All estimated phenology
    fld1="ObsYield"
    fld3="SimYield_pHpM"
    df2 = df_result.copy()
    df2.dropna(subset=[fld1, fld3], inplace=True)
    df2.reset_index(drop=True, inplace=True)

    # Figure 3 - Obs. heading - Pred. Maturity
    fld1="ObsYield"
    fld4="SimYield_pM"
    df3 = df_result.copy()
    df3.dropna(subset=[fld1, fld4], inplace=True)
    df3.reset_index(drop=True, inplace=True)

    # Figure 4 - Pred. heading - Obs. Maturity
    fld1="ObsYield"
    fld5="SimYield_pH"
    df4 = df_result.copy()
    df4.dropna(subset=[fld1, fld5], inplace=True)
    df4.reset_index(drop=True, inplace=True)

    # Initialise the subplot function using number of rows and columns
    fig, axis = plt.subplots(2, 2, figsize=(10,10), facecolor='white', constrained_layout=True, sharex=sharex, sharey=sharey) #, sharex=False, sharey=True)
    fig.suptitle('{}'.format(title), fontsize=18, y=1.05)

    fonts_axes = 12
    fonts_titles = 14

    # for use hue and better legends, convert columns to string
    # df[['location']] = df[['location']].astype(str)

    # ------------------------------
    # Chart 1
    # ------------------------------
    ax1 = axis[0, 0]
    title1='Observed phenology'
    ax1.set_title('{}'.format(title1), fontsize=fonts_titles)
    #ax1.set_xlabel('Observed Yield (t/ha)', fontsize=fonts_axes)
    ax1.set_xlabel(' ')
    ax1.set_ylabel('Estimated Yield (t/ha)', fontsize=fonts_axes)
    ax1.tick_params(labelsize=12)
    g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, s=s, color="#000000", hue=hue, lw=1, ax=ax1);
    ax1.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1")
    maxlim = int(max(df[fld1].max(), df[fld2].max())) + xy_lim
    g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    # Add hztal and vertical lines
    ax1.axvline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Observed Yield")
    ax1.axhline(df[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Estimated Yield")

    # Add texts
    minGY = df[fld1].min()
    maxGY = df[fld1].max()
    GY_min = df.loc[[df[fld2].argmin()]]
    GY_max = df.loc[[df[fld2].argmax()]]
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .03
    #yt_tl = .99
    ax1.text(0.01, 0.99, r"$\bf (a)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
    #ax1.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nLocations: " + r"$\bf"+str(len(df['location'].unique()))+"$",
    #         fontsize=10, ha=ha, va=va, transform=ax1.transAxes)
    
    if (dispScore==True):
        r2score, rmse, n_rmse, d_index, accuracy = getScores(df, fld1=fld1, fld2=fld2)
        ax1.text(xt_tl, yt_tl-.1,"Observations: " + r"$\bf" + str(len(df)) + "$" +'\nRMSE:{:.1f}'.format(rmse)+' tha$^{-1}$' + '\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}\nAccuracy: {:.1f}%'.format(n_rmse, d_index, r2score, accuracy), 
                 fontsize=9.5, ha=ha, va=va, transform=ax1.transAxes)

    ax1.get_legend().remove()


    # ------------------------------
    # Chart 2
    # ------------------------------
    ax2 = axis[0, 1]
    title2='All estimated phenology'
    ax2.set_title('{}'.format(title2), fontsize=fonts_titles)
    #ax2.set_xlabel('Observed Yield (t/ha)', fontsize=fonts_axes)
    ax2.set_xlabel(' ')
    #ax2.set_ylabel('Estimated Yield (t/ha)', fontsize=fonts_axes)
    ax2.set_ylabel(' ')
    ax2.tick_params(labelsize=12)
    g2 = sns.scatterplot(x=fld1, y=fld3, data=df2, alpha=alpha, s=s, color="#000000", hue=hue, lw=1, ax=ax2);
    ax2.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1")
    maxlim = int(max(df2[fld1].max(), df2[fld3].max())) + xy_lim
    g2.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g2.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g2.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax2.set_axisbelow(True)
    # Add hztal and vertical lines
    ax2.axvline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Observed Yield")
    ax2.axhline(df[fld3].mean(), ls='--', c='red', linewidth=1, label="Mean Estimated Yield")

    # Add texts
    minGY = df2[fld1].min()
    maxGY = df2[fld1].max()
    GY_min = df2.loc[[df2[fld3].argmin()]]
    GY_max = df2.loc[[df2[fld3].argmax()]]
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .03
    #yt_tl = .99
    ax2.text(0.01, 0.99, r"$\bf (b)$", fontsize=18, ha='left', va='top', transform=ax2.transAxes)
    #ax2.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$",
    #         fontsize=10, ha=ha, va=va, transform=ax2.transAxes)
    
    if (dispScore==True):
        r2score, rmse, n_rmse, d_index, accuracy = getScores(df2, fld1=fld1, fld2=fld3)
        ax2.text(xt_tl, yt_tl-.1,"Observations: " + r"$\bf" + str(len(df2))  + "$" +'\nRMSE:{:.1f}'.format(rmse)+' tha$^{-1}$' + '\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}\nAccuracy: {:.1f}%'.format(n_rmse, d_index, r2score, accuracy),  fontsize=9.5, ha=ha, va=va, transform=ax2.transAxes)

    ax2.get_legend().remove()

    # ------------------------------
    # Chart 3
    # ------------------------------
    ax3 = axis[1, 0]
    title3='Obs. heading - Pred. Maturity'
    ax3.set_title('{}'.format(title3), fontsize=fonts_titles)
    ax3.set_xlabel('Observed Yield (t/ha)', fontsize=fonts_axes)
    #ax3.set_xlabel(' ')
    ax3.set_ylabel('Estimated Yield (t/ha)', fontsize=fonts_axes)
    ax3.tick_params(labelsize=12)
    g3 = sns.scatterplot(x=fld1, y=fld4, data=df3, alpha=alpha, s=s, color="#000000", hue=hue, lw=1, ax=ax3);
    ax3.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1")
    maxlim = int(max(df3[fld1].max(), df3[fld4].max())) + xy_lim
    g3.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g3.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g3.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax3.set_axisbelow(True)
    # Add hztal and vertical lines
    ax3.axvline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Observed Yield")
    ax3.axhline(df[fld4].mean(), ls='--', c='red', linewidth=1, label="Mean Estimated Yield")

    # Add texts
    minGY = df3[fld1].min()
    maxGY = df3[fld1].max()
    GY_min = df3.loc[[df3[fld4].argmin()]]
    GY_max = df3.loc[[df3[fld4].argmax()]]
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .03
    #yt_tl = .99
    ax3.text(0.01, 0.99, r"$\bf (c)$", fontsize=18, ha='left', va='top', transform=ax3.transAxes)
    #ax3.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df3))  + "$" + "\nLocations: " + r"$\bf"+str(len(df3['location'].unique()))+"$",
    #         fontsize=10, ha=ha, va=va, transform=ax3.transAxes)
    
    if (dispScore==True):
        r2score, rmse, n_rmse, d_index, accuracy = getScores(df3, fld1=fld1, fld2=fld4)
        ax3.text(xt_tl, yt_tl-.1,"Observations: " + r"$\bf" + str(len(df3))  + "$" +'\nRMSE:{:.1f}'.format(rmse)+' tha$^{-1}$' + '\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}\nAccuracy: {:.1f}%'.format(n_rmse, d_index, r2score, accuracy), 
                 fontsize=9.5, ha=ha, va=va, transform=ax3.transAxes)

    ax3.get_legend().remove()

    # ------------------------------
    # Chart 4
    # ------------------------------
    ax4 = axis[1, 1]
    title4='Pred. heading - Obs. Maturity'
    ax4.set_title('{}'.format(title4), fontsize=fonts_titles)
    ax4.set_xlabel('Observed Yield (t/ha)', fontsize=fonts_axes)
    #ax4.set_xlabel(' ')
    #ax4.set_ylabel('Estimated Yield (t/ha)', fontsize=fonts_axes)
    ax4.set_ylabel(' ')
    ax4.tick_params(labelsize=12)
    g4 = sns.scatterplot(x=fld1, y=fld5, data=df4, alpha=alpha, s=s, color="#000000", hue=hue, lw=1, ax=ax4);
    ax4.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1")
    maxlim = int(max(df4[fld1].max(), df4[fld5].max())) + xy_lim
    g4.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g4.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g4.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax4.set_axisbelow(True)
    # Add hztal and vertical lines
    ax4.axvline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Observed Yield")
    ax4.axhline(df[fld5].mean(), ls='--', c='red', linewidth=1, label="Mean Estimated Yield")

    # Add texts
    minGY = df4[fld1].min()
    maxGY = df4[fld1].max()
    GY_min = df4.loc[[df4[fld5].argmin()]]
    GY_max = df4.loc[[df4[fld5].argmax()]]
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .03
    #yt_tl = .99
    ax4.text(0.01, 0.99, r"$\bf (d)$", fontsize=18, ha='left', va='top', transform=ax4.transAxes)
    #ax4.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df4))  + "$" + "\nLocations: " + r"$\bf"+str(len(df4['location'].unique()))+"$",
    #         fontsize=10, ha=ha, va=va, transform=ax4.transAxes)
    
    if (dispScore==True):
        r2score, rmse, n_rmse, d_index, accuracy = getScores(df4, fld1=fld1, fld2=fld5)
        ax4.text(xt_tl, yt_tl-.1,"Observations: " + r"$\bf" + str(len(df4))  + "$" +'\nRMSE:{:.1f}'.format(rmse)+' tha$^{-1}$' + '\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}\nAccuracy: {:.1f}%'.format(n_rmse, d_index, r2score, accuracy), 
                 fontsize=9.5, ha=ha, va=va, transform=ax4.transAxes)

    ax4.get_legend().remove()

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

    fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.1), loc="center", ncol=ncol, 
               borderaxespad=0,fontsize=10) #, fancybox=True, shadow=True)

    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, '{}_{}_comparison_{}.pdf'.format(fname, title.replace(' ', '_'), hoy)), 
                    bbox_inches='tight', orientation='portrait',  pad_inches=0.5, dpi=300)


    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        #figures_path = os.path.join(config['RESULTS_PATH'] , '{}_{}'.format(dirname, hoy) )
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"{}_{}_comparison_{}.{}".format(fname, title.replace(' ',''), hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)


    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
    

# ----------------
def chart_compareDaysFromStagesResults(df_result=None, title="iPAR Yield model", alpha=.95, s=25, xy_lim=2,
                                       hue='loc_code', ncol=4, xt_tl=.01, yt_tl=.99, ha='left', va='top', 
                                       dirname='Figures', fname='Figure', dispScore=True, saveFig=True, 
                                       showFig=True, sharex=False, sharey=False, fmt='pdf'):
    '''
        Create a figure to compare several iPAR model simulations results of grain yield
        
        :params df_result: A table or DataFrame with the needed data
        :params title: Name of the figure
        :params alpha: Transparency of the points in chart
        :params s: Size of the points in chart
        :params xy_lim: Used to extend the x-axis limit. Default 2 units
        :params hue: Variable to classify or discriminate the results in colors
        :params ncol: Number of columns to split the legend
        :params dirname: Folder name to save results
        :params fname: File name to save the figure
        :params saveFig: Save file or export the figure
        :params sharex: Share or align all x-axis, it will use the same x scale for all charts if the value is True
        :params sharey: Share or align all y-axis, it will use the same y scale for all charts if the value is True
        :params fmt: Format of the output (PNG, JPG or PDF)
        
        :results: An integrated figure with 4 simulation charts 
        
    '''
    if (df_result is None):
        print("Input data is not valid")
        return
    # Figure 1 - Observed phenology vs All estimated phenology
    fld1="Days_HM"
    fld2="Days_pHpM"
    df = df_result.copy()
    df.dropna(subset=[fld1, fld2], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Figure 2 - Pred. heading - Obs. Maturity
    fld1="Days_HM"
    fld3="Days_pHM"
    df2 = df_result.copy()
    df2.dropna(subset=[fld1, fld3], inplace=True)
    df2.reset_index(drop=True, inplace=True)

    # Figure 3 - Obs. heading - Pred. Maturity
    fld1="Days_HM"
    fld4="Days_HpM"
    df3 = df_result.copy()
    df3.dropna(subset=[fld1, fld4], inplace=True)
    df3.reset_index(drop=True, inplace=True)
    
    # Figure 4 - Days sowing to estimated maturity
    fld5="Days_SM"
    fld6="Days_SpM"
    df4 = df_result.copy()
    df4.dropna(subset=[fld5, fld6], inplace=True)
    df4.reset_index(drop=True, inplace=True)

    
    # Initialise the subplot function using number of rows and columns
    fig, axis = plt.subplots(2, 2, figsize=(10,10), facecolor='white', constrained_layout=True, 
                             sharex=sharex, sharey=sharey)
    fig.suptitle('{}'.format(title), fontsize=18, y=1.05)

    fonts_axes = 12
    fonts_titles = 14

    # for use hue and better legends, convert columns to string
    # df[['location']] = df[['location']].astype(str)

    # ------------------------------
    # Chart 1
    # ------------------------------
    ax1 = axis[0, 0]
    title1='All estimated phenology'
    ax1.set_title('{}'.format(title1), fontsize=fonts_titles)
    ax1.set_xlabel('Observed Days H-M', fontsize=fonts_axes)
    #ax1.set_xlabel(' ')
    ax1.set_ylabel('Estimated Days pH-pM', fontsize=fonts_axes)
    ax1.tick_params(labelsize=12)
    g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, s=s, color="#000000", hue=hue, lw=1, ax=ax1);
    ax1.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1")
    maxlim = int(max(df[fld1].max(), df[fld2].max())) + xy_lim
    g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    # Add hztal and vertical lines
    ax1.axhline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Estimated Days")
    ax1.axvline(df[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Observed Days")

    # Add texts
    minGY = df[fld1].min()
    maxGY = df[fld1].max()
    GY_min = df.loc[[df[fld2].argmin()]]
    GY_max = df.loc[[df[fld2].argmax()]]
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .03
    #yt_tl = .99
    ax1.text(0.01, 0.99, r"$\bf (a)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
    #ax1.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nLocations: " + r"$\bf"+str(len(df['location'].unique()))+"$",
    #         fontsize=10, ha=ha, va=va, transform=ax1.transAxes)
    
    if (dispScore==True):
        r2score, rmse, n_rmse, d_index, accuracy = getScores(df, fld1=fld1, fld2=fld2)
        ax1.text(xt_tl, yt_tl-.1,"Observations: " + r"$\bf" + str(len(df)) + "$" +'\nRMSE:{:.1f}'.format(rmse)+' days' + '\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}'.format(n_rmse, d_index, r2score), 
                 fontsize=9.5, ha=ha, va=va, transform=ax1.transAxes)

    ax1.get_legend().remove()


    # ------------------------------
    # Chart 2
    # ------------------------------
    ax2 = axis[0, 1]
    title2='Pred. heading - Obs. Maturity'
    ax2.set_title('{}'.format(title2), fontsize=fonts_titles)
    ax2.set_xlabel('Observed Days H-M', fontsize=fonts_axes)
    #ax2.set_xlabel(' ')
    ax2.set_ylabel('Estimated Days pH-M', fontsize=fonts_axes)
    #ax2.set_ylabel(' ')
    ax2.tick_params(labelsize=12)
    g2 = sns.scatterplot(x=fld1, y=fld3, data=df2, alpha=alpha, s=s, color="#000000", hue=hue, lw=1, ax=ax2);
    ax2.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1")
    maxlim = int(max(df2[fld1].max(), df2[fld3].max())) + xy_lim
    g2.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g2.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g2.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax2.set_axisbelow(True)
    # Add hztal and vertical lines
    ax2.axhline(df2[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Estimated Days")
    ax2.axvline(df2[fld3].mean(), ls='--', c='red', linewidth=1, label="Mean Observed Days")

    # Add texts
    minGY = df2[fld1].min()
    maxGY = df2[fld1].max()
    GY_min = df2.loc[[df2[fld3].argmin()]]
    GY_max = df2.loc[[df2[fld3].argmax()]]
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .03
    #yt_tl = .99
    ax2.text(0.01, 0.99, r"$\bf (b)$", fontsize=18, ha='left', va='top', transform=ax2.transAxes)
    #ax2.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$",
    #         fontsize=10, ha=ha, va=va, transform=ax2.transAxes)
    
    if (dispScore==True):
        r2score, rmse, n_rmse, d_index, accuracy = getScores(df2, fld1=fld1, fld2=fld3)
        ax2.text(xt_tl, yt_tl-.1,"Observations: " + r"$\bf" + str(len(df2))  + "$" +'\nRMSE:{:.1f}'.format(rmse)+' days' + '\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}'.format(n_rmse, d_index, r2score), 
                 fontsize=9.5, ha=ha, va=va, transform=ax2.transAxes)

    ax2.get_legend().remove()

    # ------------------------------
    # Chart 3
    # ------------------------------
    ax3 = axis[1, 0]
    title3='Obs. heading - Pred. Maturity'
    ax3.set_title('{}'.format(title3), fontsize=fonts_titles)
    ax3.set_xlabel('Observed Days H-M', fontsize=fonts_axes)
    #ax3.set_xlabel(' ')
    ax3.set_ylabel('Estimated Days H-pM', fontsize=fonts_axes)
    ax3.tick_params(labelsize=12)
    g3 = sns.scatterplot(x=fld1, y=fld4, data=df3, alpha=alpha, s=s, color="#000000", hue=hue, lw=1, ax=ax3);
    ax3.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1")
    maxlim = int(max(df3[fld1].max(), df3[fld4].max())) + xy_lim
    g3.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g3.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g3.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax3.set_axisbelow(True)
    # Add hztal and vertical lines
    ax3.axhline(df3[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Estimated Days")
    ax3.axvline(df3[fld4].mean(), ls='--', c='red', linewidth=1, label="Mean Observed Days")

    # Add texts
    minGY = df3[fld1].min()
    maxGY = df3[fld1].max()
    GY_min = df3.loc[[df3[fld4].argmin()]]
    GY_max = df3.loc[[df3[fld4].argmax()]]
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .03
    #yt_tl = .99
    ax3.text(0.01, 0.99, r"$\bf (c)$", fontsize=18, ha='left', va='top', transform=ax3.transAxes)
    #ax3.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df3))  + "$" + "\nLocations: " + r"$\bf"+str(len(df3['location'].unique()))+"$",
    #         fontsize=10, ha=ha, va=va, transform=ax3.transAxes)
    
    if (dispScore==True):
        r2score, rmse, n_rmse, d_index, accuracy = getScores(df3, fld1=fld1, fld2=fld4)
        ax3.text(xt_tl, yt_tl-.1,"Observations: " + r"$\bf" + str(len(df3))  + "$" +'\nRMSE:{:.1f}'.format(rmse)+' days' + '\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}'.format(n_rmse, d_index, r2score), 
                 fontsize=9.5, ha=ha, va=va, transform=ax3.transAxes)

    ax3.get_legend().remove()

    # ------------------------------
    # Chart 4
    # ------------------------------
    ax4 = axis[1, 1]
    title4='Days sowing to estimated maturity'
    ax4.set_title('{}'.format(title4), fontsize=fonts_titles)
    ax4.set_xlabel('Observed Days S-M', fontsize=fonts_axes)
    #ax4.set_xlabel(' ')
    ax4.set_ylabel('Estimated Days S-pM', fontsize=fonts_axes)
    #ax4.set_ylabel(' ')
    ax4.tick_params(labelsize=12)
    g4 = sns.scatterplot(x=fld5, y=fld6, data=df4, alpha=alpha, s=s, color="#000000", hue=hue, lw=1, ax=ax4);
    ax4.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1")
    maxlim = int(max(df4[fld5].max(), df4[fld6].max())) + xy_lim
    g4.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g4.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g4.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax4.set_axisbelow(True)
    # Add hztal and vertical lines
    ax4.axhline(df4[fld5].mean(), ls='--', c='red', linewidth=1, label="Mean Estimated Days")
    ax4.axvline(df4[fld6].mean(), ls='--', c='red', linewidth=1, label="Mean Observed Days")

    # Add texts
    minGY = df4[fld5].min()
    maxGY = df4[fld5].max()
    GY_min = df4.loc[[df4[fld6].argmin()]]
    GY_max = df4.loc[[df4[fld6].argmax()]]
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .03
    #yt_tl = .99
    ax4.text(0.01, 0.99, r"$\bf (d)$", fontsize=18, ha='left', va='top', transform=ax4.transAxes)
    #ax4.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df4))  + "$" + "\nLocations: " + r"$\bf"+str(len(df4['location'].unique()))+"$",
    #         fontsize=10, ha=ha, va=va, transform=ax4.transAxes)
    
    if (dispScore==True):
        r2score, rmse, n_rmse, d_index, accuracy =  getScores(df4, fld1=fld5, fld2=fld6)
        ax4.text(xt_tl, yt_tl-.1,"Observations: " + r"$\bf" + str(len(df4))  + "$" +'\nRMSE:{:.1f}'.format(rmse)+' days' + '\nn-RMSE:{:.1f}%\nd-index:{:.2f}\nR$^2$: {:.2f}'.format(n_rmse, d_index, r2score), 
                 fontsize=9.5, ha=ha, va=va, transform=ax4.transAxes)

    ax4.get_legend().remove()

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

    fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.1), loc="center", ncol=ncol, 
               borderaxespad=0,fontsize=10) #, fancybox=True, shadow=True)

    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, '{}_{}_comparison_{}.pdf'.format(fname, title.replace(' ', '_'), hoy)), 
                    bbox_inches='tight', orientation='portrait',  pad_inches=0.5, dpi=300)


    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        #figures_path = os.path.join(config['RESULTS_PATH'] , '{}_{}'.format(dirname, hoy) )
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"{}_{}_comparison_{}.{}".format(fname, title.replace(' ',''), hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
    
    

# ----------------------------
# IWIN G x E
# ----------------------------

def chartObsYieldTrendsAllNurseries(df_N=None, title='Observed yield trend', s=15, alpha=0.3,
                                    addMaxMinTexts=False, saveFig=True, showFig=True, dirname='./', fmt='pdf'):
    '''
        Display a grain yield trends by year
        
    :params saveFig: Save file in JPG or PDF format
    :params fmt: Format of the output
    
    :results: A figure in JPG or PDF format with the filename specified into the folder name 
    
    '''
    global nursery
    global colors
    global paleta
    
    if (df_N is None):
        print("Input data is not valid")
        return
    
    df=df_N[['country', 'loc_code', 'Nursery', 'YearofSow', 'G',
             'Days_To_Heading', 'Days_To_Maturity', 'ObsYield']]\
    .groupby(['YearofSow','G'], as_index=False).agg({
        'country':'first', 'loc_code':'first', 'Nursery':'first', 'Days_To_Heading':'mean', 
        'Days_To_Maturity':'mean', 'ObsYield':'mean'
    }).reset_index()

    minYear = int(df['YearofSow'].min())
    maxYear = int(df['YearofSow'].max())
    #
    GY_min = df.loc[[df['ObsYield'].argmin()]]
    GY_max = df.loc[[df['ObsYield'].argmax()]]
    #print(GY_min['ObsYield'].values[0], GY_max['ObsYield'].values[0])

    fig,ax1=plt.subplots(nrows=1,ncols=1,figsize=(10,6))
    g1 = sns.scatterplot(x='YearofSow', y='ObsYield', data=df, marker='o', s=s, alpha=alpha, 
                         palette=paleta, hue='Nursery',  ax=ax1);
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    minlim = minYear - 1
    maxlim = maxYear + 2
    g1.set(xlim=(minlim, maxlim)) #, ylim=(0, maxlim)

    # Add trend lines
    for n in range(len(nursery)):
        df2 = df_N[df_N['Nursery'] == nursery[n]]
        df2=df2[['Nursery', 'YearofSow', 'G','Days_To_Heading', 'Days_To_Maturity', 'ObsYield']]\
        .groupby(['YearofSow','G'], as_index=False).agg({
            'Nursery':'first', 'Days_To_Heading':'mean', 'Days_To_Maturity':'mean', 'ObsYield':'mean'
        }).reset_index()
        sns.regplot(df2['YearofSow'],df2['ObsYield'],ax=ax1,truncate=True,
                    scatter=False,line_kws={'lw': 1, 'color': colors[n],'linestyle':'--'},label=nursery[n]+" all lines")

    ax1.set_title('{} ({} - {})'.format(title, minYear, maxYear), fontsize=18)
    ax1.set_xlabel('Year',fontsize=15)
    ax1.set_ylabel('Yield (t/ha)',fontsize=15)
    ax1.text(minYear, GY_max['ObsYield'].values[0],r"Observations: $\bf{" + str(len(df_N))  + "}$",fontsize=10)
    dict_numObsxNursery = dict(df_N['Nursery'].value_counts())
    for i, n in enumerate(dict_numObsxNursery):
        #print(minYear, GY_max['ObsYield'].values[0] ,'{} - {}'.format(n, dict_numObsxNursery[n]))
        ax1.text(minYear, GY_max['ObsYield'].values[0] - 0.6 - i/2,'{} - {}'.format(n, dict_numObsxNursery[n]),
                 color=paleta[n], fontsize=10)
    
    if (addMaxMinTexts is True):
        # Add text of the country to the Maximum values per Nursery
        xtext = GY_min['YearofSow'].values[0]
        ytext = GY_min['ObsYield'].values[0]
        ax1.text(xtext+2, ytext-0.25,'{} - {} - {:.2f} t/ha'.format(GY_min['country'].values[0], GY_min['loc_code'].values[0], ytext),fontsize=8)
        ax1.annotate("", xy=(xtext+2,ytext-0.25), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
        # Maximum
        xtext = GY_max['YearofSow'].values[0]
        ytext = GY_max['ObsYield'].values[0]
        ax1.text(xtext-18, ytext+0.15,'{} - {} - {:.2f} t/ha'.format(GY_max['country'].values[0], GY_max['loc_code'].values[0], ytext),fontsize=8)
        ax1.annotate("", xy=(xtext-2,ytext+0.15), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))

    #ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 0.975), ncol=4, borderaxespad=0, fontsize=12)
    ax1.legend(loc=2,bbox_to_anchor=(1.05, 1), ncol=1, borderaxespad=0, fontsize=12)
    
    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, 'IWIN_GxE_{}_{}-{}_{}.pdf'.format(title.replace(' ', '_'), 
                                                                                  minYear, maxYear, hoy)), 
                    bbox_inches='tight', orientation='portrait',  pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        #figures_path = os.path.join(RESULTS_PATH , 'Figures')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_GxE_{}_{}-{}_{}.{}".format(title.replace(' ', '_'), 
                                                                               minYear, maxYear, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
    

def chartAvgObsYieldTrendsAllNurseries(df_N=None, title='Average observed yield trends', s=50, alpha=0.83,
                                       saveFig=True, showFig=True, dirname='./', fmt='pdf'):
    '''
        Display average grain yield trends by year
        
    :params saveFig: Save file in JPG or PDF format
    :params fmt: Format of the output
    
    :results: A figure in JPG or PDF format with the filename specified into the folder name 
    
    '''
    global nursery 
    global colors 
    global paleta
    
    if (df_N is None):
        print("Input data is not valid")
        return
    df=df_N[['Nursery', 'YearofSow', 'G','Days_To_Heading', 'Days_To_Maturity', 'ObsYield']]\
    .groupby(['YearofSow'], as_index=False).agg({
        'Nursery':'first', 'Days_To_Heading':'mean', 'Days_To_Maturity':'mean', 'ObsYield':'mean'
    }).reset_index()

    minYear = int(df['YearofSow'].min())
    maxYear = int(df['YearofSow'].max())
    #
    fig,ax1=plt.subplots(nrows=1,ncols=1,figsize=(10,6))
    #ax1.scatter(df['YearofSow'],df['ObsYield'],c=df['Nursery'],alpha=0.3,s=5)
    g1 = sns.scatterplot(x='YearofSow', y='ObsYield', data=df, marker='o', s=s, alpha=alpha, 
                         palette=paleta, hue='Nursery',  ax=ax1);
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    minlim = minYear - 1
    maxlim = maxYear + 2
    g1.set(xlim=(minlim, maxlim)) #, ylim=(0, maxlim)

    # Add trend lines
    for n in range(len(nursery)):
        df2 = df_N[df_N['Nursery'] == nursery[n]]
        df2=df2[['Nursery', 'YearofSow', 'G', 'Days_To_Heading', 'Days_To_Maturity', 'ObsYield']]\
        .groupby(['YearofSow'], as_index=False).agg({
            'Nursery':'first', 'Days_To_Heading':'mean', 'Days_To_Maturity':'mean', 'ObsYield':'mean'
        }).reset_index()
        sns.regplot(df2['YearofSow'],df2['ObsYield'],ax=ax1,truncate=True,
                    scatter=False,line_kws={'lw': 1, 'color': colors[n],'linestyle':'--'},label=nursery[n]+" all lines")

    ax1.set_title('{} ({} - {})'.format(title, minYear, maxYear), fontsize=18)
    ax1.set_xlabel('Year',fontsize=15)
    ax1.set_ylabel('Avg. Yield (t/ha)',fontsize=15)

    #ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 0.975), ncol=4, borderaxespad=0, fontsize=12)
    ax1.legend(loc=2,bbox_to_anchor=(1.05, 1), ncol=1, borderaxespad=0, fontsize=12)

    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, 'IWIN_GxE_{}_{}-{}_mean_{}.pdf'.format(title.replace(' ', '_'), 
                                                                                  minYear, maxYear, hoy)), 
                    bbox_inches='tight', orientation='portrait',  pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        #figures_path = os.path.join(RESULTS_PATH , 'Figures')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_GxE_{}_{}-{}_mean_{}.{}".format(title.replace(' ', '_'), 
                                                                               minYear, maxYear, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();

        
#
def figure_AvgYieldbyGID_LR(avgGY_1=None, nursery="", hue='G', hue2=None, hue3=None, hue4=None, 
                            lw=0.8, s=10, s4=20, alpha=.45, alpha2=.85, alpha4=.95, loc=2, ncol=3, 
                            xt_tl=.01, yt_tl=.99, ha='left', va='top',
                            methods=['OLS', 'RANSAC'], fld1="AvGYxLocOcc", fld2="AvGYxGID", 
                            saveFig=True, showFig=True, dirname='./', fmt='pdf'):
    ''' Display a figure with regression lines applied for GIDs 
    
    '''
    df = avgGY_1.copy()
    df.dropna(subset=[fld1, fld2], inplace=True)
    
    target_GIDs_m1m2 = df[( (df['target_m1']==1) & (df['target_m2']==1) )]['G'].unique() #.reset_index(drop=True)
    target_GIDs_m1 = df[ df['target_m1']==1 ]['G'].unique() #.reset_index(drop=True)
    target_GIDs_m2 = df[ df['target_m2']==1 ]['G'].unique() #.reset_index(drop=True)
    # Chart 2
    df2 = df[df['G'].isin(target_GIDs_m1)].reset_index(drop=True)
    # Chart 3
    df3 = df[df['G'].isin(target_GIDs_m2)].reset_index(drop=True)
    # Chart 4
    df4 = df[df['G'].isin(target_GIDs_m1m2)].reset_index(drop=True)

    # Initialise the subplot function using number of rows and columns
    fig, axis = plt.subplots(2, 2, figsize=(10,10), facecolor='white', constrained_layout=True, sharex=True, sharey=True)
    #fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    fig.suptitle('{}'.format(nursery), fontsize=18, y=1.05)

    fonts_axes = 12
    fonts_titles = 14

    # for use hue and better legends
    #df[['G', 'location']] = df[['G', 'location']].astype(str)
    df2[['G', 'location']] = df2[['G', 'location']].astype(str)
    df3[['G', 'location']] = df3[['G', 'location']].astype(str)
    df4[['G', 'location']] = df4[['G', 'location']].astype(str)

    # --------------------------------
    # Chart 1
    # --------------------------------
    #axis[0, 0].scatter(X, Y1, s=s)
    ax1 = axis[0, 0]
    title1='Regression lines by GID'
    ax1.set_title('{}'.format(title1), fontsize=fonts_titles)
    ax1.set_xlabel('Average Yield of site-year', fontsize=fonts_axes)
    ax1.set_ylabel('Yield of GID', fontsize=fonts_axes)
    ax1.tick_params(labelsize=12)

    g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, color="#000000", hue=None, s=s, lw=1, ax=ax1);
    ax1.axline((0, 0), slope=1, color='#444', ls="--", linewidth=1.25, zorder=0, label="line 1:1") #c=".5",
    maxlim_1 = int(max(df[fld1].max(), df[fld2].max())) + 1
    g1.set(xlim=(0, maxlim_1), ylim=(0, maxlim_1))
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    # Add hztal and vertical lines
    ax1.axhline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax1.axvline(df[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    tl_txt = 4.5
    tl_fg = 2
    minGY = df[fld1].min()
    maxGY = df[fld1].max()
    GY_min = df.loc[[df[fld2].argmin()]]
    GY_max = df.loc[[df[fld2].argmax()]]
    # Using the same maximum values of chart 1, due to all charts are share x and y axes
    #ax1.text(minGY, maxlim_1-tl_fg, r"$\bf a)$", fontsize=18)
    #ax1.text(minGY, int(GY_max[fld2].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$",fontsize=10)
    #ax1.text(minGY, maxlim_1-tl_txt, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$",fontsize=10)
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .01
    #yt_tl = .99
    ax1.text(0.01, 0.99, r"$\bf a)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
    ax1.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$", fontsize=10, ha=ha, va=va, transform=ax1.transAxes)

    # Add regression lines
    for gid in df['G'].unique():
        x = df[fld1][df['G']==gid].to_numpy()
        y = df[fld2][df['G']==gid].to_numpy()
        # determine best fit line
        par = np.polyfit(x, y, 1, full=True)
        pend=par[0][0]
        intercept=par[0][1]
        #print("y = {:.7f}x + {:.7f}".format(pend, intercept))
        y_predicted = [pend*i + intercept  for i in x]
        # Intercept with line 1:1
        m1, c1 = 1.0, 0.0
        m2, c2 = pend, intercept
        xi = (c1 - c2) / (m2 - m1)
        yi = m1 * xi + c1
        #plt.axvline(x=xi, color='gray', linestyle='--', linewidth=0.5)
        #plt.axhline(y=yi, color='gray', linestyle='--', linewidth=0.5)
        #plt.scatter(xi, yi, color='black', s=10, marker='x')
        # Define environment
        #sns.lineplot(x=x,y=y_predicted, palette='Set1', ax=ax1, s=0.5, label='y='+str(round(pend,7))+'x + '+str(round(intercept,7)))
        if (intercept>0 and m2>=m1):
            sns.lineplot(x=x,y=y_predicted, color='orange', ax=ax1, lw=lw, 
                         label='Above avg Poor Env, Above avg Good Env'.format(gid))
        elif (intercept<0 and m1>=m2 ):
            sns.lineplot(x=x,y=y_predicted, color='red', ax=ax1, lw=lw, 
                         label='Below avg Poor Env, Below avg Good Env'.format(gid))
        elif (intercept>=0 and m1>=m2):
            sns.lineplot(x=x,y=y_predicted, color='purple', ax=ax1, lw=lw, 
                         label='Above avg Poor Env, Below avg Good Env'.format(gid))
        elif (intercept<=0 and m2>=m1):
            sns.lineplot(x=x,y=y_predicted, color='cyan', ax=ax1, lw=lw, 
                         label='Below avg Poor Env, Above avg Good Env'.format(gid))
        else:
            sns.lineplot(x=x,y=y_predicted, color='black', ax=ax1, lw=0.25)

    ax1.get_legend().remove()


    # ------------------------------
    # Chart 2
    # ------------------------------
    methods=['OLS']
    ax2 = axis[0, 1]
    #title2='OLS lines by GID - Method 1'
    title2='Method 1'
    ax2.set_title('{}'.format(title2), fontsize=fonts_titles)
    ax2.set_xlabel('Average Yield of site-year', fontsize=fonts_axes)
    ax2.set_ylabel('Yield of GID', fontsize=fonts_axes)
    ax2.tick_params(labelsize=12)

    g2 = sns.scatterplot(x=fld1, y=fld2, data=df2, alpha=alpha, color="#000000", style=hue2, hue=hue2, s=s, lw=1, ax=ax2);
    ax2.axline((0, 0), slope=1, color='#444', ls="--", linewidth=1.25, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df2[fld1].max(), df2[fld2].max())) + 1
    g2.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g2.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g2.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax2.set_axisbelow(True)
    # Add hztal and vertical lines
    ax2.axhline(df2[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax2.axvline(df2[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    minGY = df2[fld1].min()
    maxGY = df2[fld1].max()
    GY_min = df2.loc[[df2[fld2].argmin()]]
    GY_max = df2.loc[[df2[fld2].argmax()]]
    #ax2.text(minGY, maxlim_1-tl_fg, r"$\bf b)$", fontsize=18)
    #ax2.text(minGY, int(GY_max[fld2].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(target_GIDs_m1))+"$",fontsize=10)
    #ax2.text(minGY, maxlim_1-tl_txt, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(target_GIDs_m1))+"$",fontsize=10)
    ax2.text(0.01, 0.99, r"$\bf b)$", fontsize=18,  ha='left', va='top', transform=ax2.transAxes)
    ax2.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(target_GIDs_m1))+"$",
             fontsize=10, ha=ha, va=va, transform=ax2.transAxes)

    # Add regression lines
    GID_environments_Inv = []
    for gid in df2['G'].unique():
        x = df2[fld1][df2['G']==gid].to_numpy().reshape(-1, 1)
        y = df2[fld2][df2['G']==gid].to_numpy()
        plotline_X = np.arange(x.min(), x.max()).reshape(-1, 1)
        m1, c1 = 1.0, 0.0
        # We are interested in the purple and yellow lines (GIDs)
        if ('OLS' in methods):
            # determine best fit line
            lr = LinearRegression().fit(x, y)
            y_linear_regression = lr.predict(plotline_X)
            # Intercept and slope
            intercept = lr.intercept_
            m2 = lr.coef_[0]
            x1 = plotline_X.flatten()
            y_predicted = y_linear_regression
            if (intercept>0 and m2>=m1):
                sns.lineplot(x1,y_predicted, color='orange', ax=ax2, lw=lw, 
                             label='Above avg Poor Env, Above avg Good Env'.format(gid))
            elif (intercept<0 and m1>=m2 ):
                sns.lineplot(x1,y_predicted, color='red', ax=ax2, lw=lw, 
                             label='Below avg Poor Env, Below avg Good Env'.format(gid))
            elif (intercept>=0 and m1>=m2):
                sns.lineplot(x1,y_predicted, color='purple', ax=ax2, lw=lw, 
                             label='Above avg Poor Env, Below avg Good Env'.format(gid))
            elif (intercept<=0 and m2>=m1):
                sns.lineplot(x1,y_predicted, color='cyan', ax=ax2, lw=lw, 
                             label='Below avg Poor Env, Above avg Good Env'.format(gid))
            else:
                sns.lineplot(x1,y_predicted, color='black', ax=ax2, lw=0.25)

    ax2.get_legend().remove()

    # ------------------------------
    # Chart 3 - RANSAC (RANdom SAmple Consensus)
    # https://scikit-learn.org/stable/modules/linear_model.html#ransac-regression
    # ------------------------------
    methods=['RANSAC']
    dispOutlier=False
    hue='G'
    ax3 = axis[1, 0]
    #title3='RANSAC lines by GID - Method 2'
    title3='Method 2'
    ax3.set_title('{}'.format(title3), fontsize=fonts_titles)
    ax3.set_xlabel('Average Yield of site-year', fontsize=fonts_axes)
    ax3.set_ylabel('Yield of GID', fontsize=fonts_axes)
    ax3.tick_params(labelsize=12)

    g3 = sns.scatterplot(x=fld1, y=fld2, data=df3, alpha=alpha2, color="#000000", style=hue3, hue=hue3, s=s, lw=1, ax=ax3);
    ax3.axline((0, 0), slope=1, color='#444', ls="--", linewidth=1.25, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df3[fld1].max(), df3[fld2].max())) + 1
    g3.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g3.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g3.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax3.set_axisbelow(True)
    # Add hztal and vertical lines
    ax3.axhline(df3[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax3.axvline(df3[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    minGY = df3[fld1].min()
    maxGY = df3[fld1].max()
    GY_min = df3.loc[[df3[fld2].argmin()]]
    GY_max = df3.loc[[df3[fld2].argmax()]]
    #ax3.text(minGY, maxlim_1-tl_fg, r"$\bf c)$", fontsize=18)
    #ax3.text(minGY, int(GY_max[fld2].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df3))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(target_GIDs_m2))+"$",fontsize=10)
    #ax3.text(minGY, maxlim_1-tl_txt, "Observations: " + r"$\bf" + str(len(df3))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(target_GIDs_m2))+"$",fontsize=10)
    ax3.text(0.01, 0.99, r"$\bf c)$", fontsize=18, ha='left', va='top', transform=ax3.transAxes)
    ax3.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df3))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(target_GIDs_m2))+"$",
             fontsize=10, ha=ha, va=va, transform=ax3.transAxes)

    # Add regression lines
    for gid in df3['G'].unique():
        x = df3[fld1][df3['G']==gid].to_numpy().reshape(-1, 1)
        y = df3[fld2][df3['G']==gid].to_numpy()
        plotline_X = np.arange(x.min(), x.max()).reshape(-1, 1)
        m1, c1 = 1.0, 0.0
        # We are interested in the purple and yellow lines (GIDs)
        if ('RANSAC' in methods):
            # determine best fit line
            ransac = RANSACRegressor(random_state=42).fit(x, y)
            y_ransac_regression = ransac.predict(plotline_X)
            #print(ransac.estimator_.intercept_, ransac.estimator_.coef_[0])
            intercept = ransac.estimator_.intercept_
            m2 = ransac.estimator_.coef_[0]
            # Outlier
            if (dispOutlier is True):
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)
                #plt.scatter( x[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", linewidth=1.25, label="Inliers" )
                plt.scatter( x[outlier_mask], y[outlier_mask], color="red", marker="x", linewidth=1.25, label="Outliers", ax=ax3) #

            x = plotline_X.flatten()
            y_predicted = y_ransac_regression
            if (intercept>0 and m2>=m1):
                sns.lineplot(x=x,y=y_predicted, color='orange', ax=ax3, lw=lw, 
                             label='Above avg Poor Env, Above avg Good Env'.format(gid))
            elif (intercept<0 and m1>=m2 ):
                sns.lineplot(x=x,y=y_predicted, color='red', ax=ax3, lw=lw, 
                             label='Below avg Poor Env, Below avg Good Env'.format(gid))
            elif (intercept>=0 and m1>=m2):
                sns.lineplot(x=x,y=y_predicted, color='purple', ax=ax3, lw=lw, 
                             label='Above avg Poor Env, Below avg Good Env'.format(gid))
            elif (intercept<=0 and m2>=m1):
                sns.lineplot(x=x,y=y_predicted, color='cyan', ax=ax3, lw=lw, 
                             label='Below avg Poor Env, Above avg Good Env'.format(gid))
            else:
                sns.lineplot(x=x,y=y_predicted, color='black', ax=ax3, lw=0.25)

    ax3.get_legend().remove()

    # ------------------------------
    # Chart 4 - Combine Method 1 and 2 and apply linear Regression again (RANSAC)
    # ------------------------------
    methods=['RANSAC']
    dispOutlier=False
    ax4 = axis[1, 1]
    #title4='Selected GIDs - Method 1 & 2'
    title4='Method 1 & 2'
    ax4.set_title('{}'.format(title4), fontsize=fonts_titles)
    ax4.set_xlabel('Average Yield of site-year', fontsize=fonts_axes)
    ax4.set_ylabel('Yield of GID', fontsize=fonts_axes)
    ax4.tick_params(labelsize=12)

    g4 = sns.scatterplot(x=fld1, y=fld2, data=df4, alpha=alpha4, color="#000000", style=hue4, hue=hue4, s=s4, lw=1, ax=ax4);
    ax4.axline((0, 0), slope=1, color='#444', ls="--", linewidth=1.25, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df4[fld1].max(), df4[fld2].max())) + 1
    g4.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g4.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g4.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax4.set_axisbelow(True)
    # Add hztal and vertical lines
    ax4.axhline(df4[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax4.axvline(df4[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    minGY = df4[fld1].min()
    maxGY = df4[fld1].max()
    GY_min = df4.loc[[df4[fld2].argmin()]]
    GY_max = df4.loc[[df4[fld2].argmax()]]
    #ax4.text(minGY, maxlim_1-tl_fg, r"$\bf d)$", fontsize=18)
    #ax4.text(minGY, int(GY_max[fld2].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df4))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(target_GIDs_m1m2))+"$",fontsize=10)
    #ax4.text(minGY, maxlim_1-tl_txt, "Observations: " + r"$\bf" + str(len(df4))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(target_GIDs_m1m2))+"$",fontsize=10)
    ax4.text(0.01, 0.99, r"$\bf d)$", fontsize=18, ha='left', va='top', transform=ax4.transAxes)
    ax4.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df4))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(target_GIDs_m1m2))+"$",
             fontsize=10, ha=ha, va=va, transform=ax4.transAxes)

    # Add regression lines
    for gid in df4['G'].unique():
        x = df4[fld1][df4['G']==gid].to_numpy().reshape(-1, 1)
        y = df4[fld2][df4['G']==gid].to_numpy()
        plotline_X = np.arange(x.min(), x.max()).reshape(-1, 1)
        m1, c1 = 1.0, 0.0
        # We are interested in the purple and yellow lines (GIDs)
        if ('RANSAC' in methods):
            # determine best fit line
            ransac = RANSACRegressor(random_state=42).fit(x, y)
            y_ransac_regression = ransac.predict(plotline_X)
            #print(ransac.estimator_.intercept_, ransac.estimator_.coef_[0])
            intercept = ransac.estimator_.intercept_
            m2 = ransac.estimator_.coef_[0]
            # Outlier
            if (dispOutlier is True):
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)
                #plt.scatter( x[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", 
                # linewidth=1.25, label="Inliers" )
                plt.scatter( x[outlier_mask], y[outlier_mask], color="red", marker="x", 
                            linewidth=1.25, label="Outliers", ax=ax4) #

            x = plotline_X.flatten()
            y_predicted = y_ransac_regression
            if (intercept>0 and m2>=m1):
                sns.lineplot(x=x,y=y_predicted, color='orange', ax=ax4, lw=lw, 
                             label='Above avg Poor Env, Above avg Good Env'.format(gid))
            elif (intercept<0 and m1>=m2 ):
                sns.lineplot(x=x,y=y_predicted, color='red', ax=ax4, lw=lw, 
                             label='Below avg Poor Env, Below avg Good Env'.format(gid))
            elif (intercept>=0 and m1>=m2):
                sns.lineplot(x=x,y=y_predicted, color='purple', ax=ax4, lw=lw, 
                             label='Above avg Poor Env, Below avg Good Env'.format(gid))
            elif (intercept<=0 and m2>=m1):
                sns.lineplot(x=x,y=y_predicted, color='cyan', ax=ax4, lw=lw, 
                             label='Below avg Poor Env, Above avg Good Env'.format(gid))
            else:
                sns.lineplot(x=x,y=y_predicted, color='black', ax=ax4, lw=0.25)

    ax4.get_legend().remove()

    # ------------------------------
    # Remove some labels
    for ax in fig.get_axes():
        ax.label_outer()
    
    # ------------------------------
    # Add Footer
    #plt.figtext(0, -0.3,
    #         r"Figure 1. $\bfGenotypes selection using linear regression methods$.\na) Linear regression lines for all yield environment;\nb) Selected genotypes in poor and good yield environment using Ordinary Least Squares (OLS) linear regression lines;\nc) Selected genotypes in poor and good yield environment using Robust linear model estimation (RANSAC);\nd) Selected genotypes using a mixed method (OLS and RANSAC), common selected GIDs in both methods are chosen.", 
    #         va='bottom', ha="left", fontsize=10)
    
    plt.figtext(0, -0.3,
                r"$\bfFigure$ $\bf1$. $\bfGenotypes$ $\bfselection$ $\bfusing$ $\bflinear$ $\bfregression$ $\bfmethods.$"+ "\n" + 
                r"a) Linear regression lines for all yield environment;"+ "\n" + 
                r"b) Selected genotypes in poor and good yield environment using Ordinary Least Squares (OLS) linear regression lines;"+ "\n" +
                r"c) Selected genotypes in poor and good yield environment using Robust linear model estimation (RANSAC);"+ "\n" + 
                r"d) Selected genotypes using a mixed method (" + r"$\mathit{OLS}$" + " and " + r"$\mathit{RANSAC}$" + "), common selected GIDs in both methods are chosen.", 
             va='bottom', ha="left", fontsize=10)
    
    # Now let's add your additional information
    #plt.annotate('...Additional information...',
    #        xy=(0, 0), xytext=(-10, 0),
    #        xycoords=('axes fraction', 'figure fraction'),
    #        textcoords='offset points',
    #        size=14, ha='center', va='bottom')

    #plt.figtext(0.5, -0.2, "one text and next text", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    
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

    fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.1), loc="center", ncol=ncol, 
               borderaxespad=0,fontsize=10) #, fancybox=True, shadow=True)
    
    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, 'IWIN_GxE_WheatLines_GenSelxEnv_LRm1m2_{}_{}.pdf'
                                 .format(nursery.replace(' ', '_'), hoy)), bbox_inches='tight', orientation='portrait', 
                     pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_GxE_WheatLines_GenSelxEnv_LRm1m2_{}_{}.{}"
                                 .format(title.replace(' ', '_'), hoy, fmt)), bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
        
# ------------------------------------------------------------------------------
# Display a figure with regression lines applied for GIDs
# ------------------------------------------------------------------------------
def figure_AvgYieldbyGID_classify(avgGY_1=None, avgGY_2=None, df_countOfGIDs=None, topGinBEnv=None, topGinGEnv=None,
                                  nursery="", threshold=5, hue='G', hue2=None, hue3=None, hue4=None, 
                                  lw=0.8, s=10, s4=20, alpha=.45, alpha2=.85, alpha4=.95, loc=2, ncol=3, 
                                  xt_tl=.01, yt_tl=.99, ha='left', va='top',
                                  fld1="AvGYxGID", fld2="AvGYxLocOcc", dispTxt=False, saveFig=True, showFig=True,
                                  dirname='./', fmt='pdf'):
    ''' 
        Display a figure with regression lines applied for GIDs 
        
        
    '''
    df = avgGY_1.copy()
    df.dropna(subset=[fld1, fld2], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Chart 2
    df2 = avgGY_2.copy()
    df2.dropna(subset=[fld1, fld2], inplace=True)
    df2.reset_index(drop=True, inplace=True)
    # Chart 3
    df3a = df2[( (df['G'].isin(topGinBEnv['G'].unique()) ) | (df['G'].isin(topGinGEnv['G'].unique()) )
                )].reset_index(drop=True)
    df3 = df_countOfGIDs[df_countOfGIDs['G'].isin(topGinBEnv['G'].unique())].reset_index(drop=True)
    df3['environment_m3'] = '# Occurrences Good in bad environment'
    # Chart 4
    df4 = df_countOfGIDs[df_countOfGIDs['G'].isin(topGinGEnv['G'].unique())].reset_index(drop=True)
    df4['environment_m3'] = '# Occurrences Bad in good environment'
    df5 = pd.concat([df3, df4]).reset_index(drop=True)
    
    # Initialise the subplot function using number of rows and columns
    fig, axis = plt.subplots(2, 2, figsize=(10,10), facecolor='white', constrained_layout=True) #, sharex=False, sharey=True)
    #fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    fig.suptitle('{}'.format(nursery), fontsize=18, y=1.05)

    fonts_axes = 12
    fonts_titles = 14

    # for use hue and better legends
    #df[['G', 'location']] = df[['G', 'location']].astype(str)
    #df2[['G', 'location']] = df2[['G', 'location']].astype(str)
    #df3[['G', 'location']] = df3[['G', 'location']].astype(str)
    df3a[['G', 'location']] = df3a[['G', 'location']].astype(str)
    #df4[['G', 'location']] = df4[['G', 'location']].astype(str)
    
    # ------------------------------
    # Chart 1
    # ------------------------------
    ax1 = axis[0, 0]
    title1='Yield environments - Method 3'
    ax1.set_title('{}'.format(title1), fontsize=fonts_titles)
    #ax1.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
    ax1.set_xlabel(' ')
    ax1.set_ylabel('Observed Grain Yield (Nursery-year)', fontsize=fonts_axes)
    ax1.tick_params(labelsize=12)
    g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, s=s, markers=["o","s","d","^"],facecolor="none",
                             palette={'Good in good environment':'green', 'Good in bad environment':'orange',
                                       'Bad in good environment':'brown','Bad in bad environment':'red'},
                             color="#000000", hue='environment_m3', style=None, lw=1, ax=ax1);
    ax1.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df[fld1].max(), df[fld2].max())) + 1
    g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    # Add hztal and vertical lines
    ax1.axhline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax1.axvline(df[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    tl_txt = 4.5
    tl_fg = 1.5
    minGY = df[fld1].min()
    maxGY = df[fld1].max()
    GY_min = df.loc[[df[fld2].argmin()]]
    GY_max = df.loc[[df[fld2].argmax()]]
    #ax1.text(minGY, maxlim-tl_fg, r"$\bf a)$", fontsize=18)
    #ax1.text(minGY, int(GY_max[fld2].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$",fontsize=10)
    #ax1.text(minGY, maxlim-tl_txt, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$",fontsize=10)
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .01
    #yt_tl = .99
    ax1.text(0.01, 0.99, r"$\bf a)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
    ax1.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$",
             fontsize=10, ha=ha, va=va, transform=ax1.transAxes)

    ax1.get_legend().remove()

    # ------------------------------
    # Chart 2
    # ------------------------------
    ax2 = axis[0, 1]
    title2='Selected GIDs'
    ax2.set_title('{}'.format(title2), fontsize=fonts_titles)
    #ax2.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
    #ax2.set_ylabel('Observed Grain Yield (Nursery-year)', fontsize=fonts_axes)
    ax2.set_xlabel(' ')
    ax2.set_ylabel(' ')
    ax2.tick_params(labelsize=12)
    g2 = sns.scatterplot(x=fld1, y=fld2, data=df2, alpha=alpha, s=s, markers=["o","s"],facecolor="none",
                             palette={'Good in good environment':'green', 'Good in bad environment':'orange',
                                       'Bad in good environment':'brown','Bad in bad environment':'red'},
                             color="#000000", hue='environment_m3', style='environment_m3', lw=1, ax=ax2);
    ax2.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df[fld1].max(), df[fld2].max())) + 1
    g2.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g2.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g2.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax2.set_axisbelow(True)
    # Add hztal and vertical lines
    ax2.axhline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax2.axvline(df[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    tl_txt = 3.5
    tl_fg = 1.5
    minGY = df[fld1].min()
    maxGY = df[fld1].max()
    GY_min = df.loc[[df[fld2].argmin()]]
    GY_max = df.loc[[df[fld2].argmax()]]
    #ax2.text(minGY, maxlim-tl_fg, r"$\bf b)$", fontsize=18)
    #ax2.text(minGY, int(GY_max[fld2].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$",fontsize=10)
    #ax2.text(minGY, maxlim-tl_txt, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$",fontsize=10)
    ax2.text(0.01, 0.99, r"$\bf b)$", fontsize=18, ha='left', va='top', transform=ax2.transAxes)
    ax2.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$",
             fontsize=10, ha=ha, va=va, transform=ax2.transAxes)

    ax2.get_legend().remove()
    
    # ------------------------------
    # Chart 3
    # ------------------------------
    ax3 = axis[1, 0]
    
    title3='Top {} selected GIDs'.format(threshold)
    ax3.set_title('{}'.format(title3), fontsize=fonts_titles)
    ax3.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
    ax3.set_ylabel('Observed Grain Yield (Nursery-year)', fontsize=fonts_axes)
    ax3.tick_params(labelsize=12)
    g3 = sns.scatterplot(x=fld1, y=fld2, data=df3a, alpha=alpha2, s=s4, 
                             #markers=["o","s"],facecolor="none",
                             #palette={'Good in good environment':'green', 'Good in bad environment':'orange',
                             #          'Bad in good environment':'brown','Bad in bad environment':'red'},
                             hue='G', style='G', lw=1, ax=ax3);
    ax3.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df3a[fld1].max(), df3a[fld2].max())) + 1
    g3.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g3.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g3.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax3.set_axisbelow(True)
    # Add hztal and vertical lines
    ax3.axhline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax3.axvline(df[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    tl_txt = 3.5
    tl_fg = 1.5
    minGY = df3a[fld1].min()
    maxGY = df3a[fld1].max()
    GY_min = df3a.loc[[df3a[fld2].argmin()]]
    GY_max = df3a.loc[[df3a[fld2].argmax()]]
    #ax3.text(minGY, maxlim-tl_fg, r"$\bf c)$", fontsize=18)
    #ax3.text(minGY, int(GY_max[fld2].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df3a))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df3a['G'].unique()))+"$",fontsize=10)
    #ax3.text(minGY, maxlim-tl_txt, "Observations: " + r"$\bf" + str(len(df3a))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df3a['G'].unique()))+"$",fontsize=10)
    ax3.text(0.01, 0.99, r"$\bf c)$", fontsize=18, ha='left', va='top', transform=ax3.transAxes)
    ax3.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df3a))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df3a['G'].unique()))+"$",
             fontsize=10, ha=ha, va=va, transform=ax3.transAxes)

    ax3.get_legend().remove()
    
    # Share Axis
    ax1.get_shared_x_axes().join(ax1, ax2, ax3)
    ax1.get_shared_y_axes().join(ax1, ax2, ax3)
    
    
    # fld3="AvGYxLocOcc_GoodinBadEnv"
    # fld4="countOfOccurrences_GoodinBadEnv"
    # #title3='Occurences'
    # #ax3.set_title('{}'.format(title3), fontsize=fonts_titles)
    # ax3.set_title('GIDs - Good in bad environment', fontsize=fonts_titles)
    # ax3.set_xlabel('Mean Grain Yield (Loc-Occ)', fontsize=12)
    # ax3.set_ylabel('# of GID Occurrences', fontsize=fonts_axes)
    # ax3.tick_params(labelsize=12)

    # g3 = sns.scatterplot(x=fld3, y=fld4, data=df3, alpha=.85, style=None, color="orange", marker='o',
    #                      s=35, lw=1, label='# Occurrences (Good in bad environment)', ax=ax3);
    # maxlim = int(max(df3[fld3].max(), df3[fld4].max())) + 1
    # g3.set(xlim=(0, maxlim), ylim=(0, maxlim))
    # g3.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    # g3.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    
    # # Add texts
    # if (dispTxt is True):
    #     minGY = df3[fld3].min()
    #     maxGY = df3[fld3].max()
    #     Oc_min = df3.loc[[df3[fld4].argmin()]]
    #     Oc_max = df3.loc[[df3[fld4].argmax()]]
    #     ax3.text(minGY, maxlim-tl_fg, r"$\bf c)$", fontsize=18)
    #     ax3.text(minGY, int(Oc_max[fld4].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df3))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df3['G'].unique()))+"$",fontsize=10)

    #     # Add text of the country to the Maximum values per Nursery
    #     xtext = Oc_min[fld3].values[0]
    #     ytext = Oc_min[fld4].values[0]
    #     ax3.text(xtext+0.05, ytext,'{} - {:.1f} t/ha'.format(Oc_min['G'].values[0], ytext),fontsize=8)
    #     ax3.annotate("", xy=(xtext+0.05,ytext), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
    #     # Maximum
    #     xtext = Oc_max[fld3].values[0]
    #     ytext = Oc_max[fld4].values[0]
    #     ax3.text(xtext+0.05, ytext,'{} - {:.1f} t/ha'.format(Oc_max['G'].values[0], ytext),fontsize=8)
    #     ax3.annotate("", xy=(xtext+0.05,ytext), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
    #ax3.get_legend().remove()

    # ------------------------------
    # Chart 4
    # ------------------------------
    fld3="AvGYxLocOcc_GoodinGoodEnv"
    fld4="countOfOccurrences_GoodinGoodEnv"
    ax4 = axis[1, 1]
    #title4='Occurences 2'
    #ax4.set_title('{}'.format(title4), fontsize=fonts_titles)
    ax4.set_title('GIDs Occurrences vs Avg. Yield', fontsize=fonts_titles)
    #ax4.set_xlabel('Mean Grain Yield (Loc-Occ)', fontsize=12)
    ax4.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
    ax4.set_ylabel('# of GID Occurrences', fontsize=fonts_axes)
    ax4.tick_params(labelsize=12)

    g4 = sns.scatterplot(x=fld3, y=fld4, data=df5, alpha=.85, markers=['o','o'],
                         #color="green", 
                         palette={'# Occurrences Good in good environment':'green', 
                                  '# Occurrences Good in bad environment':'orange',
                                  '# Occurrences Bad in good environment':'brown',
                                  '# Occurrences Bad in bad environment':'red'},
                        hue='environment_m3', style='environment_m3',
                         s=35, lw=1, ax=ax4 ); #label='# Occurrences',
    maxlim = int(max(df5[fld3].max(), df5[fld4].max())) + 1
    #g4.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g4.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5, zorder=0)
    g4.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5, zorder=0)
    ax4.set_axisbelow(True)
    
    # Add texts
    if (dispTxt is True):
        tl_fg = 1.5
        tl_txt = 3
        minGY = df5[fld3].min()
        maxGY = df5[fld3].max()
        Oc_min = df5.loc[[df5[fld4].argmin()]]
        Oc_max = df5.loc[[df5[fld4].argmax()]]
        #ax4.text(minGY, maxlim-tl_fg, r"$\bf d)$", fontsize=18)
        #ax4.text(minGY, int(Oc_max[fld4].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df5))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df5['G'].unique()))+"$",fontsize=10)
        ax4.text(0.01, 0.99, r"$\bf d)$", fontsize=18, ha='left', va='top', transform=ax4.transAxes)
        ax4.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df5))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df5['G'].unique()))+"$",
             fontsize=10, ha=ha, va=va, transform=ax4.transAxes)

        # Add text of the country to the Maximum values per Nursery
        xtext = Oc_min[fld3].values[0]
        ytext = Oc_min[fld4].values[0]
        #ax4.text(xtext+0.05, ytext,'{}'.format(Oc_min['G'].values[0]),fontsize=8)
        ax4.text(.99, .2,'{}'.format(Oc_min['G'].values[0]),fontsize=8, ha='right', va='bottom', transform=ax4.transAxes)
        #ax4.annotate("", xy=(xtext+0.05,ytext), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
        ax4.annotate("", xy=(xtext, ytext), xytext=(.95,.2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"), 
                     textcoords='axes fraction', ha='center', va='center', transform=ax4.transAxes)

        # Maximum
        xtext = Oc_max[fld3].values[0]
        ytext = Oc_max[fld4].values[0]
        #ax4.text(xtext+0.05, ytext,'{}'.format(Oc_max['G'].values[0]),fontsize=8)
        #ax4.annotate("", xy=(xtext+0.05,ytext), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
        ax4.text(.99, .8,'{}'.format(Oc_max['G'].values[0]),fontsize=8, ha='right', va='top', transform=ax4.transAxes)
        ax4.annotate("", xy=(xtext, ytext), xytext=(.95,.8), 
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"), #"bar,armA=0.0,armB=0.0,fraction=-0.2,angle=180"), 
                     textcoords='axes fraction', ha='center', va='center', transform=ax4.transAxes)

    ax4.get_legend().remove()
    
    
    # ------------------------------
    # Remove some labels
    #for ax in fig.get_axes():
    #    ax.label_outer()
    
    # ------------------------------
    # Add Footer
    #plt.figtext(0, -0.3,
    #            r"$\bfFigure$ $\bf1$. $\bfGenotypes$ $\bfselection$ $\bfusing$ $\bflinear$ $\bfregression$ $\bfmethods.$"+ "\n" + 
    #            r"a) Linear regression lines for all yield environment;"+ "\n" + 
    #            r"b) Selected genotypes in poor and good yield environment using Ordinary Least Squares (OLS) linear regression lines;"+ "\n" +
    #            r"c) Selected genotypes in poor and good yield environment using Robust linear model estimation (RANSAC);"+ "\n" + 
    #            r"d) Selected genotypes using a mixed method (" + r"$\mathit{OLS}$" + " and " + r"$\mathit{RANSAC}$" + "), common selected GIDs in both methods are chosen.", 
    #         va='bottom', ha="left", fontsize=10)
    
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

    fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.1), loc="center", ncol=ncol, 
               borderaxespad=0,fontsize=10) #, fancybox=True, shadow=True)
    
    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, 'IWIN_GxE_WheatLines_GenSelxEnv_classifm3_{}_{}.pdf'
                                 .format(nursery.replace(' ', '_'), hoy)), bbox_inches='tight', orientation='portrait', 
                     pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_WheatLines_GenSelxEnv_classifm3_{}_{}.{}"
                                 .format(title.replace(' ', '_'), hoy, fmt)), bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
#
# ------------------------------------------------------------------------------
def figure_AvgYieldbyGID_classify_v2(avgGY_1=None, avgGY_2=None, df_countOfGIDs=None, topGinBEnv=None, topBinGEnv=None,
                                  nursery="", threshold=5, hue='G', hue2=None, hue3=None, hue4=None, 
                                  lw=0.8, s=10, s4=35, alpha=.45, alpha2=.85, alpha4=.95, loc=4, ncol=3, 
                                  xt_tl=.01, yt_tl=.99, ha='left', va='top',
                                  fld1=" AvGYxGID", fld2="AvGYxLocOcc", dispTxt=False, saveFig=True, showFig=True,
                                  dirname='./', fmt='pdf'):
    ''' 
        Display a figure with regression lines applied for GIDs 
        
        
    '''
    
    target_Colors = {
        'AL': 'red',
        'AH_B': 'green',
        'AH_A': 'green',
        'BL_B': 'red',
        'BL_A': 'red',
        'BH' : 'green',
        
    }
    
    paletaColors = {
        # new categories
        'AL': 'orange',
        'AH_B': '#44c554',
        'AH_A': 'green',
        'BL_B': 'red',
        'BL_A': '#f84615',
        'BH' : 'brown',
        
        #'AH' : "Above average in high yield environment", # "Good in good environment"
        #'BL': "Below average in low yield environment", #"Bad in bad environment"
        
        'Above average in high yield environment - below 1:1 line':'#44c554',
        'Above average in high yield environment - above 1:1 line':'green',
        'Below average in low yield environment - above 1:1 line':'#f84615',
        'Below average in low yield environment - below 1:1 line':'red',
        'Above average in low yield environment':'orange',
        'Below average in high yield environment':'brown',
        #old categories
        'Good in good environment':'green', 
        'Good in bad environment':'orange',
        'Bad in good environment':'brown',
        'Bad in bad environment':'red'
    }
    df = avgGY_1.copy()
    df.dropna(subset=[fld1, fld2], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Chart 2
    df2 = avgGY_2.copy()
    df2.dropna(subset=[fld1, fld2], inplace=True)
    df2.reset_index(drop=True, inplace=True)
    # Chart 3
    df3a = df2[( (df2['G'].isin(topGinBEnv['G'].unique()) ) | (df2['G'].isin(topBinGEnv['G'].unique()) )
                )].reset_index(drop=True)
    df3 = df_countOfGIDs[df_countOfGIDs['G'].isin(topGinBEnv['G'].unique())].reset_index(drop=True)
    df3['environment_m3'] = '# Occurrences Good in bad environment'
    # Chart 4
    df4 = df_countOfGIDs[df_countOfGIDs['G'].isin(topBinGEnv['G'].unique())].reset_index(drop=True)
    df4['environment_m3'] = '# Occurrences Bad in good environment'
    df5 = pd.concat([df3, df4]).reset_index(drop=True)
    
    # Initialise the subplot function using number of rows and columns
    fig, axis = plt.subplots(2, 2, figsize=(10,10), facecolor='white', constrained_layout=True) #, sharex=False, sharey=True)
    #fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    fig.suptitle('{}'.format(nursery), fontsize=18, y=1.05)

    fonts_axes = 12
    fonts_titles = 14

    # for use hue and better legends
    #df[['G', 'location']] = df[['G', 'location']].astype(str)
    #df2[['G', 'location']] = df2[['G', 'location']].astype(str)
    #df3[['G', 'location']] = df3[['G', 'location']].astype(str)
    df3a[['G', 'location']] = df3a[['G', 'location']].astype(str)
    #df4[['G', 'location']] = df4[['G', 'location']].astype(str)
    
    # ------------------------------
    # Chart 1
    # ------------------------------
    ax1 = axis[0, 0]
    title1='Yield environments - Method 3'
    ax1.set_title('{}'.format(title1), fontsize=fonts_titles)
    #ax1.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
    ax1.set_xlabel(' ')
    ax1.set_ylabel('Observed Grain Yield (Nursery-year)', fontsize=fonts_axes)
    ax1.tick_params(labelsize=12)
    g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, s=s, markers=["o","s","d","^"],facecolor="none",
                             palette=paletaColors,
                             color="#000000", hue='environment_m3', style=None, lw=1, ax=ax1);
    ax1.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df[fld1].max(), df[fld2].max())) + 1
    g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    # Add hztal and vertical lines
    ax1.axhline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax1.axvline(df[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    tl_txt = 4.5
    tl_fg = 1.5
    minGY = df[fld1].min()
    maxGY = df[fld1].max()
    GY_min = df.loc[[df[fld2].argmin()]]
    GY_max = df.loc[[df[fld2].argmax()]]
    #ax1.text(minGY, maxlim-tl_fg, r"$\bf a)$", fontsize=18)
    #ax1.text(minGY, int(GY_max[fld2].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$",fontsize=10)
    #ax1.text(minGY, maxlim-tl_txt, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$",fontsize=10)
    # Put text in the same place for all chart, using relative coordinates of the axes
    #xt_tl = .01
    #yt_tl = .99
    ax1.text(0.01, 0.99, r"$\bf a)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
    ax1.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$",
             fontsize=10, ha=ha, va=va, transform=ax1.transAxes)

    ax1.get_legend().remove()

    # ------------------------------
    # Chart 2
    # ------------------------------
    ax2 = axis[0, 1]
    title2='Selected GIDs'
    ax2.set_title('{}'.format(title2), fontsize=fonts_titles)
    #ax2.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
    #ax2.set_ylabel('Observed Grain Yield (Nursery-year)', fontsize=fonts_axes)
    ax2.set_xlabel(' ')
    ax2.set_ylabel(' ')
    ax2.tick_params(labelsize=12)
    g2 = sns.scatterplot(x=fld1, y=fld2, data=df2, alpha=alpha, s=s, #markers="o",facecolor="none", ##["o","s"]
                             palette=target_Colors,
                             color="#000000", hue='environment_m3', style='environment_m3', lw=1, ax=ax2);
    ax2.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df[fld1].max(), df[fld2].max())) + 1
    g2.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g2.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g2.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax2.set_axisbelow(True)
    # Add hztal and vertical lines
    ax2.axhline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax2.axvline(df[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    tl_txt = 3.5
    tl_fg = 1.5
    minGY = df[fld1].min()
    maxGY = df[fld1].max()
    GY_min = df.loc[[df[fld2].argmin()]]
    GY_max = df.loc[[df[fld2].argmax()]]
    #ax2.text(minGY, maxlim-tl_fg, r"$\bf b)$", fontsize=18)
    #ax2.text(minGY, int(GY_max[fld2].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$",fontsize=10)
    #ax2.text(minGY, maxlim-tl_txt, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$",fontsize=10)
    ax2.text(0.01, 0.99, r"$\bf b)$", fontsize=18, ha='left', va='top', transform=ax2.transAxes)
    ax2.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df2))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$",
             fontsize=10, ha=ha, va=va, transform=ax2.transAxes)

    ax2.get_legend().remove()
    
    # ------------------------------
    # Chart 3
    # ------------------------------
    ax3 = axis[1, 0]
    title3='Top {} selected GIDs'.format(threshold)
    ax3.set_title('{}'.format(title3), fontsize=fonts_titles)
    ax3.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
    ax3.set_ylabel('Observed Grain Yield (Nursery-year)', fontsize=fonts_axes)
    ax3.tick_params(labelsize=12)
    g3 = sns.scatterplot(x=fld1, y=fld2, data=df3a, alpha=alpha2, s=s4, 
                             #markers=["o","s"],facecolor="none", palette=paletaColors,
                             hue='G', style='G', lw=1, ax=ax3);
    ax3.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
    #maxlim = int(max(df3a[fld1].max(), df3a[fld2].max())) + 1
    #g3.set(xlim=(0, maxlim), ylim=(0, maxlim))
    
    #minlim_x = int(min(df3a[fld1].min(), df3a[fld1].min())) - 1.2 #plot_params['limspan']
    #minlim_y = int(min(df3a[fld2].min(), df3a[fld2].min())) - 1.2 #plot_params['limspan']
    #maxlim_x = int(max(df3a[fld1].max(), df3a[fld1].max())) + 1.2 #plot_params['limspan']
    #maxlim_y = int(max(df3a[fld2].max(), df3a[fld2].max())) + 1.2 #plot_params['limspan']
    #g3.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
    
    g3.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g3.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax3.set_axisbelow(True)
    # Add hztal and vertical lines
    ax3.axhline(df[fld1].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
    ax3.axvline(df[fld2].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    tl_txt = 3.5
    tl_fg = 1.5
    minGY = df3a[fld1].min()
    maxGY = df3a[fld1].max()
    #GY_min = df3a.loc[[df3a[fld2].argmin()]]
    #GY_max = df3a.loc[[df3a[fld2].argmax()]]
    #ax3.text(minGY, maxlim-tl_fg, r"$\bf c)$", fontsize=18)
    ax3.text(0.01, 0.99, r"$\bf c)$", fontsize=18, ha='left', va='top', transform=ax3.transAxes)
    ax3.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df3a))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df3a['G'].unique()))+"$",
             fontsize=10, ha=ha, va=va, transform=ax3.transAxes)

    ax3.get_legend().remove()
    
    # Share Axis
    ax1.get_shared_x_axes().join(ax1, ax2, ax3)
    ax1.get_shared_y_axes().join(ax1, ax2, ax3)
    
    # ------------------------------
    # Chart 4
    # ------------------------------
    fld3="AvGYxLocOcc_AL" #"AvGYxLocOcc_GoodinGoodEnv"
    fld4="countOfOccurrences_AL" #"countOfOccurrences_GoodinGoodEnv"
    ax4 = axis[1, 1]
    #title4='Occurences 2'
    #ax4.set_title('{}'.format(title4), fontsize=fonts_titles)
    ax4.set_title('GIDs Occurrences vs Avg. Yield', fontsize=fonts_titles)
    #ax4.set_xlabel('Mean Grain Yield (Loc-Occ)', fontsize=12)
    ax4.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
    ax4.set_ylabel('# of GID Occurrences', fontsize=fonts_axes)
    ax4.tick_params(labelsize=12)

    g4 = sns.scatterplot(x=fld3, y=fld4, data=df5, alpha=.85, markers=['o','o'],
                         #color="green", 
                         palette={'# Occurrences Good in good environment': 'green', 
                                  '# Occurrences Good in bad environment':'red', #'orange',
                                  '# Occurrences Bad in good environment':'green', #'brown',
                                  '# Occurrences Bad in bad environment':'red'},
                        hue='environment_m3', style='environment_m3',
                         s=35, lw=1, ax=ax4 ); #label='# Occurrences',
    maxlim = int(max(df5[fld3].max(), df5[fld4].max())) + 1
    #g4.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g4.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5, zorder=0)
    g4.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5, zorder=0)
    ax4.set_axisbelow(True)
    
    # Add texts
    if (dispTxt is True):
        tl_fg = 1.5
        tl_txt = 3
        minGY = df5[fld3].min()
        maxGY = df5[fld3].max()
        Oc_min = df5.loc[[df5[fld4].argmin()]]
        Oc_max = df5.loc[[df5[fld4].argmax()]]
        #ax4.text(minGY, maxlim-tl_fg, r"$\bf d)$", fontsize=18)
        #ax4.text(minGY, int(Oc_max[fld4].values[0])-tl_txt, "Observations: " + r"$\bf" + str(len(df5))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df5['G'].unique()))+"$",fontsize=10)
        ax4.text(0.01, 0.99, r"$\bf d)$", fontsize=18, ha='left', va='top', transform=ax4.transAxes)
        ax4.text(xt_tl, yt_tl-.1, "Observations: " + r"$\bf" + str(len(df5))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df5['G'].unique()))+"$",
             fontsize=10, ha=ha, va=va, transform=ax4.transAxes)

        # Add text of the country to the Maximum values per Nursery
        xtext = Oc_min[fld3].values[0]
        ytext = Oc_min[fld4].values[0]
        #ax4.text(xtext+0.05, ytext,'{}'.format(Oc_min['G'].values[0]),fontsize=8)
        ax4.text(.99, .2,'{}'.format(Oc_min['G'].values[0]),fontsize=8, ha='right', va='bottom', transform=ax4.transAxes)
        #ax4.annotate("", xy=(xtext+0.05,ytext), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
        ax4.annotate("", xy=(xtext, ytext), xytext=(.95,.2),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"), 
                     textcoords='axes fraction', ha='center', va='center', transform=ax4.transAxes)

        # Maximum
        xtext = Oc_max[fld3].values[0]
        ytext = Oc_max[fld4].values[0]
        #ax4.text(xtext+0.05, ytext,'{}'.format(Oc_max['G'].values[0]),fontsize=8)
        #ax4.annotate("", xy=(xtext+0.05,ytext), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
        ax4.text(.99, .8,'{}'.format(Oc_max['G'].values[0]),fontsize=8, ha='right', va='top', transform=ax4.transAxes)
        ax4.annotate("", xy=(xtext, ytext), xytext=(.95,.8), 
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"), #"bar,armA=0.0,armB=0.0,fraction=-0.2,angle=180"), 
                     textcoords='axes fraction', ha='center', va='center', transform=ax4.transAxes)

    ax4.get_legend().remove()
    
    
    # ------------------------------
    # Remove some labels
    #for ax in fig.get_axes():
    #    ax.label_outer()
    
    # ------------------------------
    # Add Footer
    #plt.figtext(0, -0.3,
    #            r"$\bfFigure$ $\bf1$. $\bfGenotypes$ $\bfselection$ $\bfusing$ $\bflinear$ $\bfregression$ $\bfmethods.$"+ "\n" + 
    #            r"a) Linear regression lines for all yield environment;"+ "\n" + 
    #            r"b) Selected genotypes in poor and good yield environment using Ordinary Least Squares (OLS) linear regression lines;"+ "\n" +
    #            r"c) Selected genotypes in poor and good yield environment using Robust linear model estimation (RANSAC);"+ "\n" + 
    #            r"d) Selected genotypes using a mixed method (" + r"$\mathit{OLS}$" + " and " + r"$\mathit{RANSAC}$" + "), common selected GIDs in both methods are chosen.", 
    #         va='bottom', ha="left", fontsize=10)
    
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

    fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.1), loc="center", ncol=ncol, 
               borderaxespad=0,fontsize=10) #, fancybox=True, shadow=True)
    
    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, 'IWIN_GxE_WheatLines_GenSelxEnv_classifm3_{}_{}.pdf'
                                 .format(nursery.replace(' ', '_'), hoy)), bbox_inches='tight', orientation='portrait', 
                     pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_WheatLines_GenSelxEnv_classifm3_{}_{}.{}"
                                 .format(title.replace(' ', '_'), hoy, fmt)), bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
#
# ------------------------------------------------------------------------------
# Display a figure for 9 partitions in classification methods
# ------------------------------------------------------------------------------
def plot_AvgYieldbyGID_classify(avgGY_1=None, selGIDs_m3=None, numGIDsOfOcurrences=None, 
                                nursery=None, threshold=10,
                                title=None, ptype=0, qd=4, target=[], 
                                plot_params=None, saveFig=True, showFig=True, dirname='./', fmt='jpg'):
    
    ''' Display figures with classification results by linear regression and clustering lines '''
    
    target_Colors = {
        'AL': 'red', #d62728
        'AH_B': 'green', #2ca02c
        'AH_A': 'green',
        'BL_B': 'red',
        'BL_A': 'red',
        'BH' : 'green',
        #'blue' : #1f77b4
        #'orange' : #ff7f0e
        #'purple' : #9467bd
        #'brown' : #8c564b
        #'pink' : #e377c2
        #'gray' : #7f7f7f
        #'olive' : #bcbd22
        #'cyan' : #17becf,
        #'AL',
        'AM': 'black',
        'AH': 'green',
        'BL': 'red',
        'BM': 'black',
        #'BH'
    }
    paletaColors = {
        'AL': 'orange',
        'AH_B': 'cyan',
        'AH_A': 'blue',
        'BL_B': 'purple',
        'BL_A': '#f84615',
        'BH' : 'brown',
        # qd6
        'AM': 'cyan',
        'AH': 'blue',
        'BL': 'green',
        'BM': 'purple',
    }
    
    occurrenceColors = {
        '# Occurrences above average in low environment': 'red',
        '# Occurrences above average in medium environment': 'orange',
        '# Occurrences above average in high environment': 'green',
        '# Occurrences below average in low environment': 'red',
        '# Occurrences below average in medium environment': 'orange',
        '# Occurrences below average in high environment': 'green'
    }
    
    legend_names = {
        '1LL':'Low yield in low environment',
        '2LM':'Low yield in medium environment',
        '3LH':'Low yield in high environment',
        '4ML':'Medium yield in low environment',
        '5MM':'Medium yield in medium environment',
        '6MH':'Medium yield in high environment',
        '7HL':'High yield in low environment',
        '8HM':'High yield in medium environment',
        '9HH':'High yield in high environment',
        
        'AL': "Above average in low yield environment", #"Good in bad environment"
        'AH' : "Above average in high yield environment", # "Good in good environment"
        'AH_B': "Above average in high yield environment - below 1:1 line",
        'AH_A': "Above average in high yield environment - above 1:1 line",
        'BL': "Below average in low yield environment", #"Bad in bad environment"
        'BL_B': "Below average in low yield environment - below 1:1 line",
        'BL_A': "Below average in low yield environment - above 1:1 line",
        'BH' : "Below average in high yield environment", #"Bad in good environment"
        
        'AM': "Above average in medium yield environment",
        'BM': "Below average in medium yield environment",
    }
    
    plot_default_params = dict(
        limspan=1.2, txtspan=0.02,
        arrowprops1=dict(arrowstyle= '->', color='red', lw=1, ls='-', alpha=0.7), 
        fld1="AvGYxLocOcc", fld2="AvGYxGID", defaultcolor1="#000000", 
        palette1=paletaColors, palette2=target_Colors,
        hue1='environment_m3', s1=10, lw1=0.8, alpha1=.55,  s2=5, alpha2=0.55, s3=20, alpha3=0.75,
        l1_1_color='#444', l1_1_ls="--", l1_1_lw=1.25, l1_1_z=0, l1_1_label="line 1:1",
        xlabel1='Average Yield of site-year [${tha^{-1}}$]', ylabel1='Yield of GID [${tha^{-1}}$]',
        xylabel_fontsize=13, xt_tl1=.99, yt_tl1=.01, ha1='right', va1='bottom', stats_fontsize1=9,
        xt_tl4=.01, yt_tl4=.99, ha4='left', va4='top',
        fig_sleft=0.01, fig_sbottom=0.01, dispGrid=True, colorGrid='#d3d3d3', lwGrid=0.5,
        loc=2, ncol=1, leg_fontsize=10, showLegend=True, dispTxt=True, dispTxt2=True, dispTxt3=True, dispTxt4=True,
        showTargetText=True, target_fontsize=9
    )
    if (plot_params is None):
        plot_params = plot_default_params
    else:
        plot_params = {**plot_default_params, **plot_params}

    if (avgGY_1 is None):
        print("Input data not valid")
        return
    df = avgGY_1.copy()
    df.dropna(subset=[plot_params['fld1'], plot_params['fld2']], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # --------------------
    # Figure 1
    # --------------------
    if (ptype==0):
        fig, (ax1) = plt.subplots(figsize=(8,8)) #, constrained_layout=True)
        fig.subplots_adjust(left=plot_params['fig_sleft'], bottom=plot_params['fig_sbottom'])
        g1 = sns.scatterplot(x=plot_params['fld1'], y=plot_params['fld2'], data=df, 
                             alpha=plot_params['alpha1'], color=plot_params['defaultcolor1'], 
                             palette=plot_params['palette1'],
                             hue=plot_params['hue1'], s=plot_params['s1'], lw=plot_params['lw1'], ax=ax1);
        ax1.axline((0, 0), slope=1, color=plot_params['l1_1_color'], ls=plot_params['l1_1_ls'], 
                   linewidth=plot_params['l1_1_lw'], zorder=plot_params['l1_1_z'], label=plot_params['l1_1_label'])
        maxlim = int(max(df[plot_params['fld1']].max(), df[plot_params['fld2']].max())) + plot_params['limspan']
        g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
        if (plot_params['dispGrid'] is True):
            g1.grid(visible=True, which='major', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            g1.grid(visible=True, which='minor', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            ax1.set_axisbelow(True)
        if (title==None): title='GID and Environment Classification by Average Yield'
        plt.suptitle(title, fontsize=18, y=0.99)
        ax1.set_title('{}'.format(nursery), fontsize=15)
        ax1.set_xlabel(plot_params['xlabel1'], fontsize=plot_params['xylabel_fontsize'])
        ax1.set_ylabel(plot_params['ylabel1'], fontsize=plot_params['xylabel_fontsize'])
        ax1.tick_params(labelsize=12)
        #ax1.axis('equal')
        
        # Add classification texts
        # Add stats
        minGY = df[plot_params['fld1']].min()
        maxGY = df[plot_params['fld1']].max()
        GY_min = df.loc[[df[plot_params['fld2']].argmin()]]
        GY_max = df.loc[[df[plot_params['fld2']].argmax()]]
        
        if (qd==4):
            # Add hztal and vertical lines
            ax1.axhline(df[plot_params['fld1']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
            ax1.axvline(df[plot_params['fld2']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
            ax1.text(maxlim-0.05, df[plot_params['fld1']].mean()+0.25, 'above yield', 
                     {'ha': 'right', 'va': 'center'}, rotation=0)
            ax1.text(maxlim-0.05, df[plot_params['fld1']].mean()-0.25, 'below yield', 
                     {'ha': 'right', 'va': 'center'}, rotation=0)
            ax1.text(df[plot_params['fld2']].mean()+0.25, maxlim-0.05, 'high environment', 
                     {'ha': 'center', 'va': 'top'}, rotation=90)
            ax1.text(df[plot_params['fld2']].mean()-0.25, maxlim-0.05, 'low environment', 
                     {'ha': 'center', 'va': 'top'}, rotation=90)
            if (plot_params['dispTxt'] is True):
                count_AL = len(df[df['environment_m3']=='AL'])
                #count_AH = len(df[df['environment_m3']=='AH']) 
                count_AH_B = len(df[df['environment_m3']=='AH_B']) 
                count_AH_A = len(df[df['environment_m3']=='AH_A'])
                #count_BL = len(df[df['environment_m3']=='BL'])
                count_BH = len(df[df['environment_m3']=='BH'])
                count_BL_B = len(df[df['environment_m3']=='BL_B'])
                count_BL_A = len(df[df['environment_m3']=='BL_A'])
                count_U = len(df[df['environment_m3']=="Undefined"])

                ax1.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df['country'].unique()))+"$",
                         + "\nAL: " + r"$\bf"+str(count_AL)+"$" #+ ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         + ", AH_B: " + r"$\bf"+str(count_AH_B)+"$" + ", AH_A: " + r"$\bf"+str(count_AH_A)+"$"
                         #+ "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         + "\nBH: " + r"$\bf"+str(count_BH)+"$"
                         + ", BL_B: " + r"$\bf"+str(count_BL_B)+"$" + ", BL_A: " + r"$\bf"+str(count_BL_A)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'],
                         transform=ax1.transAxes)
        
        elif (qd==6):
            YL, YM, YH = df[plot_params['fld1']].min(), df[plot_params['fld1']].mean(), df[plot_params['fld1']].max()
            EL, EM, EH = df[plot_params['fld2']].min(), df[plot_params['fld2']].mean(), df[plot_params['fld2']].max()
            #print(YL, YM, YH, EL, EM, EH)
            Yrng = YH - YL
            Erng = EH - EL
            Y = Yrng / 3.
            E = Erng / 3.
            # Yield
            Y1 = YL
            Y2 = YL + Y
            Y3 = YH - Y #YL + 2*Y # 
            Y4 = YH #YL + 3*Y
            # Environment
            E1 = EL
            E2 = EL + E
            E3 = EH - E
            E4 = EH
            # Add hztal and vertical lines
            ax1.axvline(Y2, ls='--', c='red', linewidth=1) #, label="E2")
            ax1.axvline(Y3, ls='--', c='red', linewidth=1) #, label="E2")
            
            if (plot_params['showTargetText'] is True):
                ax1.text((Erng/6)*1, maxlim-0.25, 'Low yielding \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                ax1.text((Erng/6)*3, maxlim-0.25, 'Medium yielding \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                ax1.text((Erng/6)*5, maxlim-0.25, 'High yielding \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
         
            if (plot_params['dispTxt'] is True):
                count_AL = len(df[df['environment_m3']=='AL'])
                count_AM = len(df[df['environment_m3']=='AM']) 
                count_AH = len(df[df['environment_m3']=='AH']) 
                count_BL = len(df[df['environment_m3']=='BL'])
                count_BM = len(df[df['environment_m3']=='BM']) 
                count_BH = len(df[df['environment_m3']=='BH']) 
                count_U = len(df[df['environment_m3']=="Undefined"])

                ax1.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df['country'].unique()))+"$",
                         + "\nAL: " + r"$\bf"+str(count_AL)+"$" #
                         + ", AM: " + r"$\bf"+str(count_AM)+"$"
                         + ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         + "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         + ", BM: " + r"$\bf"+str(count_BM)+"$"
                         + ", BH: " + r"$\bf"+str(count_BH)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'],
                         transform=ax1.transAxes)
        
        elif (qd==9):
            YL, YM, YH = df[plot_params['fld1']].min(), df[plot_params['fld1']].mean(), df[plot_params['fld1']].max()
            EL, EM, EH = df[plot_params['fld2']].min(), df[plot_params['fld2']].mean(), df[plot_params['fld2']].max()
            #print(YL, YM, YH, EL, EM, EH)
            Yrng = YH - YL
            Erng = EH - EL
            Y = Yrng / 3.
            E = Erng / 3.
            # Yield
            Y1 = YL
            Y2 = YL + Y
            Y3 = YH - Y #YL + 2*Y # 
            Y4 = YH #YL + 3*Y
            # Environment
            E1 = EL
            E2 = EL + E
            E3 = EH - E
            E4 = EH
            # Add hztal and vertical lines
            #ax1.axvline(Y1, ls='--', c='red', linewidth=1) #, label="E2")
            ax1.axvline(Y2, ls='--', c='red', linewidth=1) #, label="E2")
            ax1.axvline(Y3, ls='--', c='red', linewidth=1) #, label="E3")
            ax1.axhline(E2, ls='--', c='red', linewidth=1) #, label="Y2")
            ax1.axhline(E3, ls='--', c='red', linewidth=1) #, label="Y3")
            #ax1.axhline(E1, ls='--', c='red', linewidth=1) #, label="Y3")
            
            ax1.text(maxlim-0.05, E2+0.25, 'Above low yield', {'ha': 'right', 'va': 'center'}, color='gray', rotation=0)
            ax1.text(maxlim-0.05, E2-0.25, 'Below medium yield', {'ha': 'right', 'va': 'center'}, color='gray',rotation=0)
            ax1.text(maxlim-0.05, E3+0.25, 'Above medium yield', {'ha': 'right', 'va': 'center'}, color='gray',rotation=0)
            ax1.text(maxlim-0.05, E3-0.25, 'Below high yield', {'ha': 'right', 'va': 'center'}, color='gray',rotation=0)
            ax1.text(Y2-0.25, maxlim-0.05, 'Below medium environment', {'ha': 'center', 'va': 'top'}, color='gray',rotation=90)
            ax1.text(Y2+0.25, maxlim-0.05, 'Above low environment', {'ha': 'center', 'va': 'top'}, color='gray',rotation=90)
            ax1.text(Y3-0.25, maxlim-0.05, 'Below high environment', {'ha': 'center', 'va': 'top'}, color='gray',rotation=90)
            ax1.text(Y3+0.25, maxlim-0.05, 'Above medium environment', {'ha': 'center', 'va': 'top'}, color='gray',rotation=90)
            if (plot_params['dispTxt'] is True):
                # Count of Observations by target
                #'1LL':'Low yield in low environment'
                count_1LL = len(df[df['environment_m3']=='1LL'])
                #'2LM':'Low yield in medium environment'
                count_2LM = len(df[df['environment_m3']=='2LM'])
                #'3LH':'Low yield in high environment'
                count_3LH = len(df[df['environment_m3']=='3LH'])
                #'4ML':'Medium yield in low environment'
                count_4ML = len(df[df['environment_m3']=='4ML'])
                #'5MM':'Medium yield in medium environment'
                count_5MM = len(df[df['environment_m3']=='5MM'])
                #'6MH':'Medium yield in high environment'
                count_6MH = len(df[df['environment_m3']=='6MH'])
                #'7HL':'High yield in low environment'
                count_7HL = len(df[df['environment_m3']=='7HL'])
                #'8HM':'High yield in medium environment'
                count_8HM = len(df[df['environment_m3']=='8HM'])
                #'9HH':'High yield in high environment'
                count_9HH = len(df[df['environment_m3']=='9HH'])

                ax1.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df['country'].unique()))+"$",
                         + "\nLL: " + r"$\bf"+str(count_1LL)+"$" + ", LM: " + r"$\bf"+str(count_2LM)+"$"+ ", LH: " + r"$\bf"+str(count_3LH)+"$"
                         + "\nML: " + r"$\bf"+str(count_4ML)+"$" + ", MM: " + r"$\bf"+str(count_5MM)+"$"+ ", MH: " + r"$\bf"+str(count_6MH)+"$"
                         + "\nHL: " + r"$\bf"+str(count_7HL)+"$" + ", HM: " + r"$\bf"+str(count_8HM)+"$"+ ", MH: " + r"$\bf"+str(count_9HH)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'], transform=ax1.transAxes)

    
    # --------------------
    # Figure 2
    # --------------------
    elif (ptype==1):
        # Chart 2
        if ((target is None) and (len(target)<=0)):
            print("Target not defined")
            
        df2 = df[df['environment_m3'].isin(target)]
        df2.reset_index(drop=True, inplace=True)
        fig, (ax1) = plt.subplots(figsize=(8,8))
        fig.subplots_adjust(left=plot_params['fig_sleft'], bottom=plot_params['fig_sbottom'])
        g1 = sns.scatterplot(x=plot_params['fld1'], y=plot_params['fld2'], data=df2, 
                             alpha=plot_params['alpha2'], color=plot_params['defaultcolor1'], 
                             palette=plot_params['palette2'],
                             hue='environment_m3', #style='environment_m3',
                             s=plot_params['s2'], lw=plot_params['lw1'], ax=ax1);
        ax1.axline((0, 0), slope=1, color=plot_params['l1_1_color'], ls=plot_params['l1_1_ls'], 
                   linewidth=plot_params['l1_1_lw'], zorder=plot_params['l1_1_z'], label=plot_params['l1_1_label'])
        maxlim = int(max(df2[plot_params['fld1']].max(), df2[plot_params['fld2']].max())) + plot_params['limspan']
        g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
        if (plot_params['dispGrid'] is True):
            g1.grid(visible=True, which='major', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            g1.grid(visible=True, which='minor', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            ax1.set_axisbelow(True)
        if (title==None): title='GID and Environment Classification by Average Yield'
        plt.suptitle(title, fontsize=18, y=0.99)
        ax1.set_title('{}'.format(nursery), fontsize=15)
        ax1.set_xlabel(plot_params['xlabel1'], fontsize=plot_params['xylabel_fontsize'])
        ax1.set_ylabel(plot_params['ylabel1'], fontsize=plot_params['xylabel_fontsize'])
        ax1.tick_params(labelsize=12)
        #ax1.axis('equal')
        
        # Add classification texts
        # Add stats
        minGY = df2[plot_params['fld1']].min()
        maxGY = df2[plot_params['fld1']].max()
        GY_min = df2.loc[[df2[plot_params['fld2']].argmin()]]
        GY_max = df2.loc[[df2[plot_params['fld2']].argmax()]]
        if (qd==4):
            # Add hztal and vertical lines
            ax1.axhline(df[plot_params['fld1']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
            ax1.axvline(df[plot_params['fld2']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
            #ax1.text(maxlim-0.05, df[plot_params['fld1']].mean()+0.25, 'above yield', {'ha': 'right', 'va': 'center'}, rotation=0)
            #ax1.text(maxlim-0.05, df[plot_params['fld1']].mean()-0.25, 'below yield', {'ha': 'right', 'va': 'center'}, rotation=0)
            #ax1.text(df[plot_params['fld2']].mean()+0.25, maxlim-0.05, 'high environment', {'ha': 'center', 'va': 'top'}, rotation=90)
            #ax1.text(df[plot_params['fld2']].mean()-0.25, maxlim-0.05, 'low environment', {'ha': 'center', 'va': 'top'}, rotation=90)
            
            if (plot_params['showTargetText'] is True):
                ax1.text(maxlim/4, maxlim-0.25, 'Above average \nin low yielding \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                ax1.text((maxlim/4)*3, maxlim-0.25, 'Below average \nin high yielding \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
            
            if (plot_params['dispTxt'] is True):
                count_AL = len(df2[df2['environment_m3']=='AL'])
                #count_AH = len(df2[df2['environment_m3']=='AH']) 
                count_AH_B = len(df2[df2['environment_m3']=='AH_B']) 
                count_AH_A = len(df2[df2['environment_m3']=='AH_A'])
                #count_BL = len(df2[df2['environment_m3']=='BL'])
                count_BH = len(df2[df2['environment_m3']=='BH'])
                count_BL_B = len(df2[df2['environment_m3']=='BL_B'])
                count_BL_A = len(df2[df2['environment_m3']=='BL_A'])
                count_U = len(df2[df2['environment_m3']=="Undefined"])

                ax1.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df2))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df2['country'].unique()))+"$",
                         + "\nAL: " + r"$\bf"+str(count_AL)+"$" #+ ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         + ", AH_B: " + r"$\bf"+str(count_AH_B)+"$" + ", AH_A: " + r"$\bf"+str(count_AH_A)+"$"
                         #+ "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         + "\nBH: " + r"$\bf"+str(count_BH)+"$"
                         + ", BL_B: " + r"$\bf"+str(count_BL_B)+"$" + ", BL_A: " + r"$\bf"+str(count_BL_A)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'], transform=ax1.transAxes)
        elif (qd==6):
            YL, YM, YH = df2[plot_params['fld1']].min(), df2[plot_params['fld1']].mean(), df2[plot_params['fld1']].max()
            EL, EM, EH = df2[plot_params['fld2']].min(), df2[plot_params['fld2']].mean(), df2[plot_params['fld2']].max()
            #print(YL, YM, YH, EL, EM, EH)
            Yrng = YH - YL
            Erng = EH - EL
            Y = Yrng / 3.
            E = Erng / 3.
            # Yield
            Y1 = YL
            Y2 = YL + Y
            Y3 = YH - Y #YL + 2*Y # 
            Y4 = YH #YL + 3*Y
            # Environment
            E1 = EL
            E2 = EL + E
            E3 = EH - E
            E4 = EH
            if ((len(target)>1) and (('BH' in target) or ('AH' in target)) ):
                # Add hztal and vertical lines
                ax1.axvline(Y2, ls='--', c='red', linewidth=1) #, label="E2")
                ax1.axvline(Y3, ls='--', c='red', linewidth=1) #, label="E2")
            
                if (plot_params['showTargetText'] is True):
                    ax1.text((maxlim/6)*1, maxlim-0.25, 'Low yielding \nenvironments', 
                             {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                    ax1.text((maxlim/6)*3, maxlim-0.25, 'Medium yielding \nenvironments', 
                             {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                    ax1.text((maxlim/6)*5, maxlim-0.25, 'High yielding \nenvironments', 
                             {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])

            if (plot_params['dispTxt'] is True):
                count_AL = len(df2[df2['environment_m3']=='AL'])
                count_AM = len(df2[df2['environment_m3']=='AM']) 
                count_AH = len(df2[df2['environment_m3']=='AH']) 
                count_BL = len(df2[df2['environment_m3']=='BL'])
                count_BM = len(df2[df2['environment_m3']=='BM']) 
                count_BH = len(df2[df2['environment_m3']=='BH']) 
                count_U = len(df2[df2['environment_m3']=="Undefined"])

                ax1.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df2))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df['country'].unique()))+"$",
                         + "\nAL: " + r"$\bf"+str(count_AL)+"$" #
                         + ", AM: " + r"$\bf"+str(count_AM)+"$"
                         + ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         + "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         + ", BM: " + r"$\bf"+str(count_BM)+"$"
                         + ", BH: " + r"$\bf"+str(count_BH)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'],
                         transform=ax1.transAxes)
        
    # --------------------
    # Figure 3
    # --------------------
    elif (ptype==2):
        # Chart 3
        if ((target is None) and (len(target)<=0)):
            print("Target not defined")
            
        df2 = df[df['environment_m3'].isin(target)]
        df2.reset_index(drop=True, inplace=True)
        
        if (selGIDs_m3 is not None):
            df2 = df2[(
                (df2['target_m3']==1) & (df2['G'].isin(selGIDs_m3['G'].unique()))
            )].reset_index(drop=True)
        
        fig, (ax1) = plt.subplots(figsize=(8,8))
        fig.subplots_adjust(left=plot_params['fig_sleft'], bottom=plot_params['fig_sbottom'])
        g1 = sns.scatterplot(x=plot_params['fld1'], y=plot_params['fld2'], data=df2, 
                             alpha=plot_params['alpha3'], color=plot_params['defaultcolor1'], 
                             #palette=plot_params['palette2'],
                             hue='G', style='G',
                             #hue='environment_m3', #style='environment_m3',
                             s=plot_params['s3'], lw=plot_params['lw1'], ax=ax1);
        ax1.axline((0, 0), slope=1, color=plot_params['l1_1_color'], ls=plot_params['l1_1_ls'], 
                   linewidth=plot_params['l1_1_lw'], zorder=plot_params['l1_1_z'], label=plot_params['l1_1_label'])
        maxlim = int(max(df2[plot_params['fld1']].max(), df2[plot_params['fld2']].max())) + plot_params['limspan']
        g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
        if (plot_params['dispGrid'] is True):
            g1.grid(visible=True, which='major', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            g1.grid(visible=True, which='minor', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            ax1.set_axisbelow(True)
        if (title==None): title='GID and Environment Classification by Average Yield'
        plt.suptitle(title, fontsize=18, y=0.99)
        ax1.set_title('{}'.format(nursery), fontsize=15)
        ax1.set_xlabel(plot_params['xlabel1'], fontsize=plot_params['xylabel_fontsize'])
        ax1.set_ylabel(plot_params['ylabel1'], fontsize=plot_params['xylabel_fontsize'])
        ax1.tick_params(labelsize=12)
        #ax1.axis('equal')
        
        # Add classification texts
        # Add stats
        minGY = df2[plot_params['fld1']].min()
        maxGY = df2[plot_params['fld1']].max()
        GY_min = df2.loc[[df2[plot_params['fld2']].argmin()]]
        GY_max = df2.loc[[df2[plot_params['fld2']].argmax()]]
        if (qd==4):
            # Add hztal and vertical lines
            ax1.axhline(df[plot_params['fld1']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
            ax1.axvline(df[plot_params['fld2']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
            
            if (plot_params['showTargetText'] is True):
                ax1.text(maxlim/4, maxlim-0.25, 'Above average \nin low yielding \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                ax1.text((maxlim/4)*3, maxlim-0.25, 'Below average \nin high yielding \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
            
            if (plot_params['dispTxt'] is True):
                count_AL = len(df2[df2['environment_m3']=='AL'])
                #count_AH = len(df2[df2['environment_m3']=='AH']) 
                count_AH_B = len(df2[df2['environment_m3']=='AH_B']) 
                count_AH_A = len(df2[df2['environment_m3']=='AH_A'])
                #count_BL = len(df2[df2['environment_m3']=='BL'])
                count_BH = len(df2[df2['environment_m3']=='BH'])
                count_BL_B = len(df2[df2['environment_m3']=='BL_B'])
                count_BL_A = len(df2[df2['environment_m3']=='BL_A'])
                count_U = len(df2[df2['environment_m3']=="Undefined"])

                ax1.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df2))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df2['country'].unique()))+"$",
                         + "\nAL: " + r"$\bf"+str(count_AL)+"$" #+ ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         + ", AH_B: " + r"$\bf"+str(count_AH_B)+"$" + ", AH_A: " + r"$\bf"+str(count_AH_A)+"$"
                         #+ "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         + "\nBH: " + r"$\bf"+str(count_BH)+"$"
                         + ", BL_B: " + r"$\bf"+str(count_BL_B)+"$" + ", BL_A: " + r"$\bf"+str(count_BL_A)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'], transform=ax1.transAxes)
        
    
    # --------------------
    # Figure 4
    # --------------------
    elif (ptype==3):
        # Chart 4
        if ((target is None) and (len(target)<=0)):
            print("Target not defined")
        #df2 = df[df['environment_m3'].isin(target)]
        #df2.reset_index(drop=True, inplace=True)
        #fld3="AvGYxLocOcc_AL"
        #fld4="countOfOccurrences_AL"
        fld3="numOcurrences_avgGY"
        fld4="numOcurrences_target"
        
        #df3 = df_countOfGIDs[df_countOfGIDs['G'].isin(topGinBEnv['G'].unique())].reset_index(drop=True)
        #df3['environment_m3'] = '# Occurrences Good in bad environment'
        # Chart 4
        #df4 = df_countOfGIDs[df_countOfGIDs['G'].isin(topBinGEnv['G'].unique())].reset_index(drop=True)
        #df4['environment_m3'] = '# Occurrences Bad in good environment'
        #df5 = pd.concat([df3, df4]).reset_index(drop=True)
        
        df4 = numGIDsOfOcurrences.copy()
        df4.reset_index(drop=True, inplace=True)
        
        fig, (ax1) = plt.subplots(figsize=(8,8))
        fig.subplots_adjust(left=plot_params['fig_sleft'], bottom=plot_params['fig_sbottom'])
        g1 = sns.scatterplot(x=fld3, y=fld4, data=df4, alpha=.85, markers='o',
                             palette=occurrenceColors,
                             hue='environment_m3', style='environment_m3',
                             s=35, lw=1, ax=ax1 ); #label='# Occurrences',
        #maxlim = int(max(df4[fld3].max(), df4[fld4].max())) + 1
        #g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
        if (plot_params['dispGrid'] is True):
            g1.grid(visible=True, which='major', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            g1.grid(visible=True, which='minor', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            ax1.set_axisbelow(True)
        if (title==None): title='GIDs Occurrences vs Avg. Yield'
        plt.suptitle(title, fontsize=18, y=0.99)
        ax1.set_title('{}'.format(nursery), fontsize=15)
        ax1.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=plot_params['xylabel_fontsize'])
        ax1.set_ylabel('# of GID Occurrences', fontsize=plot_params['xylabel_fontsize'])
        ax1.tick_params(labelsize=12)
        # Add texts
        if (plot_params['dispTxt'] is True):
            tl_fg = 1.5
            tl_txt = 3
            minGY = df4[fld3].min()
            maxGY = df4[fld3].max()
            Oc_min = df4.loc[[df4[fld4].argmin()]]
            Oc_max = df4.loc[[df4[fld4].argmax()]]
            ax1.text(0.01, 0.99, r"$\bf d)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
            ax1.text(plot_params['xt_tl4'], plot_params['yt_tl4']-.05, "Observations: " + r"$\bf" + str(len(df4))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df4['G'].unique()))+"$",
                 fontsize=10, ha=plot_params['ha4'], va=plot_params['va4'], transform=ax1.transAxes)

            # Add text of the country to the Maximum values per Nursery
            xtext = Oc_min[fld3].values[0]
            ytext = Oc_min[fld4].values[0]
            #ax1.text(xtext+0.05, ytext,'{}'.format(Oc_min['G'].values[0]),fontsize=8)
            ax1.text(.99, .2,'{}'.format(Oc_min['G'].values[0]),fontsize=8, ha='right', va='bottom', transform=ax1.transAxes)
            ax1.annotate("", xy=(xtext, ytext), xytext=(.95,.2),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"), 
                         textcoords='axes fraction', ha='center', va='center', transform=ax1.transAxes)

            # Maximum
            xtext = Oc_max[fld3].values[0]
            ytext = Oc_max[fld4].values[0]
            #ax1.text(xtext+0.05, ytext,'{}'.format(Oc_max['G'].values[0]),fontsize=8)
            #ax1.annotate("", xy=(xtext+0.05,ytext), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
            ax1.text(.99, .8,'{}'.format(Oc_max['G'].values[0]),fontsize=8, ha='right', va='top', transform=ax1.transAxes)
            ax1.annotate("", xy=(xtext, ytext), xytext=(.95,.8), 
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"), #"bar,armA=0.0,armB=0.0,fraction=-0.2,angle=180"), 
                         textcoords='axes fraction', ha='center', va='center', transform=ax1.transAxes)

        
        
    # --------------------
    # All 4 Figures
    # --------------------
    elif (ptype==4):
        # Initialise the subplot function using number of rows and columns
        fig, axis = plt.subplots(2, 2, figsize=(10,10), facecolor='white', constrained_layout=True) #, sharex=False, sharey=True)
        fig.subplots_adjust(left=0.01, bottom=0.01) #right=0.9, top=0.9, , wspace=0.4, hspace=0.4
        fig.suptitle('{}'.format(nursery), fontsize=18, y=0.95)
        fonts_axes = 12
        fonts_titles = 14
        
        # ------------------------------
        # Chart 1
        # ------------------------------
        ax1 = axis[0, 0]
        title1='Yield environments - Method 3'
        ax1.set_title('{}'.format(title1), fontsize=fonts_titles)
        #ax1.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
        ax1.set_xlabel(' ')
        ax1.set_ylabel('Observed Grain Yield (Nursery-year)', fontsize=fonts_axes)
        ax1.tick_params(labelsize=12)
        
        g1 = sns.scatterplot(x=plot_params['fld1'], y=plot_params['fld2'], data=df, 
                             alpha=plot_params['alpha1'], color=plot_params['defaultcolor1'], 
                             palette=plot_params['palette1'],
                             hue=plot_params['hue1'], s=plot_params['s1'], lw=plot_params['lw1'], ax=ax1);
        ax1.axline((0, 0), slope=1, color=plot_params['l1_1_color'], ls=plot_params['l1_1_ls'], 
                   linewidth=plot_params['l1_1_lw'], zorder=plot_params['l1_1_z'], label=plot_params['l1_1_label'])
        maxlim = int(max(df[plot_params['fld1']].max(), df[plot_params['fld2']].max())) + plot_params['limspan']
        g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
        if (plot_params['dispGrid'] is True):
            g1.grid(visible=True, which='major', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            g1.grid(visible=True, which='minor', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            ax1.set_axisbelow(True)
        #ax1.axis('equal')
        
        # Add classification texts
        # Add stats
        minGY = df[plot_params['fld1']].min()
        maxGY = df[plot_params['fld1']].max()
        GY_min = df.loc[[df[plot_params['fld2']].argmin()]]
        GY_max = df.loc[[df[plot_params['fld2']].argmax()]]
        
        if (qd==4):
            # Add hztal and vertical lines
            ax1.axhline(df[plot_params['fld1']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
            ax1.axvline(df[plot_params['fld2']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
            ax1.text(maxlim-0.05, df[plot_params['fld1']].mean()+0.25, 'above yield', {'ha': 'right', 'va': 'center'}, color='gray', rotation=0)
            ax1.text(maxlim-0.05, df[plot_params['fld1']].mean()-0.25, 'below yield', {'ha': 'right', 'va': 'center'}, color='gray', rotation=0)
            ax1.text(df[plot_params['fld2']].mean()+0.25, maxlim-0.05, 'high environment', {'ha': 'center', 'va': 'top'}, color='gray', rotation=90)
            ax1.text(df[plot_params['fld2']].mean()-0.25, maxlim-0.05, 'low environment', {'ha': 'center', 'va': 'top'}, color='gray', rotation=90)
            
            if (plot_params['dispTxt'] is True):
                count_AL = len(df[df['environment_m3']=='AL'])
                #count_AH = len(df[df['environment_m3']=='AH']) 
                count_AH_B = len(df[df['environment_m3']=='AH_B']) 
                count_AH_A = len(df[df['environment_m3']=='AH_A'])
                #count_BL = len(df[df['environment_m3']=='BL'])
                count_BH = len(df[df['environment_m3']=='BH'])
                count_BL_B = len(df[df['environment_m3']=='BL_B'])
                count_BL_A = len(df[df['environment_m3']=='BL_A'])
                count_U = len(df[df['environment_m3']=="Undefined"])

                ax1.text(0.01, 0.99, r"$\bf a)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
                ax1.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df['country'].unique()))+"$",
                         + "\nAL: " + r"$\bf"+str(count_AL)+"$" #+ ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         + ", AH_B: " + r"$\bf"+str(count_AH_B)+"$" + ", AH_A: " + r"$\bf"+str(count_AH_A)+"$"
                         #+ "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         + "\nBH: " + r"$\bf"+str(count_BH)+"$"
                         + ", BL_B: " + r"$\bf"+str(count_BL_B)+"$" + ", BL_A: " + r"$\bf"+str(count_BL_A)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'], transform=ax1.transAxes)
        
        elif (qd==6):
            YL, YM, YH = df[plot_params['fld1']].min(), df[plot_params['fld1']].mean(), df[plot_params['fld1']].max()
            EL, EM, EH = df[plot_params['fld2']].min(), df[plot_params['fld2']].mean(), df[plot_params['fld2']].max()
            #print(YL, YM, YH, EL, EM, EH)
            Yrng = YH - YL
            Erng = EH - EL
            Y = Yrng / 3.
            E = Erng / 3.
            # Yield
            Y1 = YL
            Y2 = YL + Y
            Y3 = YH - Y #YL + 2*Y # 
            Y4 = YH #YL + 3*Y
            # Environment
            E1 = EL
            E2 = EL + E
            E3 = EH - E
            E4 = EH
            # Add hztal and vertical lines
            ax1.axvline(Y2, ls='--', c='red', linewidth=1) #, label="E2")
            ax1.axvline(Y3, ls='--', c='red', linewidth=1) #, label="E2")
            # 
            tl_fg = 1.5
            tl_txt = 3
            minGY = df[plot_params['fld1']].min()
            maxGY = df[plot_params['fld1']].max()
            Oc_min = df.loc[[df[plot_params['fld2']].argmin()]]
            Oc_max = df.loc[[df[plot_params['fld2']].argmax()]]
            ax1.text(0.01, 0.99, r"$\bf a)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
            
            if (plot_params['showTargetText'] is True):
                ax1.text((Erng/6)*1, maxlim-0.35, 'Low \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                ax1.text((Erng/6)*3, maxlim-0.35, 'Medium \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                ax1.text((Erng/6)*5, maxlim-0.35, 'High \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
         
            if (plot_params['dispTxt'] is True):
                count_AL = len(df[df['environment_m3']=='AL'])
                count_AM = len(df[df['environment_m3']=='AM']) 
                count_AH = len(df[df['environment_m3']=='AH']) 
                count_BL = len(df[df['environment_m3']=='BL'])
                count_BM = len(df[df['environment_m3']=='BM']) 
                count_BH = len(df[df['environment_m3']=='BH']) 
                count_U = len(df[df['environment_m3']=="Undefined"])

                ax1.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df['country'].unique()))+"$",
                         + "\nAL: " + r"$\bf"+str(count_AL)+"$" #
                         + ", AM: " + r"$\bf"+str(count_AM)+"$"
                         + ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         + "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         + ", BM: " + r"$\bf"+str(count_BM)+"$"
                         + ", BH: " + r"$\bf"+str(count_BH)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'],
                         transform=ax1.transAxes)
        
            
        ax1.get_legend().remove()
        
        # ------------------------------
        # Chart 2
        # ------------------------------
        ax2 = axis[0, 1]
        title2='Selected GIDs'
        ax2.set_title('{}'.format(title2), fontsize=fonts_titles)
        #ax2.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
        #ax2.set_ylabel('Observed Grain Yield (Nursery-year)', fontsize=fonts_axes)
        ax2.set_xlabel(' ')
        ax2.set_ylabel(' ')
        ax2.tick_params(labelsize=12)
        
        # Chart 2
        if ((target is None) and (len(target)<=0)):
            print("Target not defined")
            
        df2 = df[df['environment_m3'].isin(target)]
        df2.reset_index(drop=True, inplace=True)
        g2 = sns.scatterplot(x=plot_params['fld1'], y=plot_params['fld2'], data=df2, 
                             alpha=plot_params['alpha2'], color=plot_params['defaultcolor1'], 
                             palette=plot_params['palette2'],
                             hue='environment_m3', #style='environment_m3',
                             s=plot_params['s2'], lw=plot_params['lw1'], ax=ax2);
        ax2.axline((0, 0), slope=1, color=plot_params['l1_1_color'], ls=plot_params['l1_1_ls'], 
                   linewidth=plot_params['l1_1_lw'], zorder=plot_params['l1_1_z'], label=plot_params['l1_1_label'])
        maxlim = int(max(df2[plot_params['fld1']].max(), df2[plot_params['fld2']].max())) + plot_params['limspan']
        g2.set(xlim=(0, maxlim), ylim=(0, maxlim))
        if (plot_params['dispGrid'] is True):
            g2.grid(visible=True, which='major', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            g2.grid(visible=True, which='minor', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            ax2.set_axisbelow(True)
        #ax2.axis('equal')
        
        # Add classification texts
        # Add stats
        minGY = df2[plot_params['fld1']].min()
        maxGY = df2[plot_params['fld1']].max()
        GY_min = df2.loc[[df2[plot_params['fld2']].argmin()]]
        GY_max = df2.loc[[df2[plot_params['fld2']].argmax()]]
        if (qd==4):
            # Add hztal and vertical lines
            #ax2.axhline(df[plot_params['fld1']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
            ax2.axvline(df[plot_params['fld2']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
            
            if (plot_params['showTargetText'] is True):
                ax2.text(maxlim/4, maxlim-0.25, 'Above average \nin low yielding \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                ax2.text((maxlim/4)*3, maxlim-0.25, 'Below average \nin high yielding \nenvironments', 
                         {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
            
            if (plot_params['dispTxt'] is True):
                count_AL = len(df2[df2['environment_m3']=='AL'])
                #count_AH = len(df2[df2['environment_m3']=='AH']) 
                count_AH_B = len(df2[df2['environment_m3']=='AH_B']) 
                count_AH_A = len(df2[df2['environment_m3']=='AH_A'])
                #count_BL = len(df2[df2['environment_m3']=='BL'])
                count_BH = len(df2[df2['environment_m3']=='BH'])
                count_BL_B = len(df2[df2['environment_m3']=='BL_B'])
                count_BL_A = len(df2[df2['environment_m3']=='BL_A'])
                count_U = len(df2[df2['environment_m3']=="Undefined"])

                ax2.text(0.01, 0.99, r"$\bf b)$", fontsize=18, ha='left', va='top', transform=ax2.transAxes)
                ax2.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df2))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df2['country'].unique()))+"$",
                         #+ "\nAL: " + r"$\bf"+str(count_AL)+"$" #+ ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         #+ ", AH_B: " + r"$\bf"+str(count_AH_B)+"$" + ", AH_A: " + r"$\bf"+str(count_AH_A)+"$"
                         ##+ "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         #+ "\nBH: " + r"$\bf"+str(count_BH)+"$"
                         #+ ", BL_B: " + r"$\bf"+str(count_BL_B)+"$" + ", BL_A: " + r"$\bf"+str(count_BL_A)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'], transform=ax2.transAxes)
        
        elif (qd==6):
            YL, YM, YH = df2[plot_params['fld1']].min(), df2[plot_params['fld1']].mean(), df2[plot_params['fld1']].max()
            EL, EM, EH = df2[plot_params['fld2']].min(), df2[plot_params['fld2']].mean(), df2[plot_params['fld2']].max()
            #print(YL, YM, YH, EL, EM, EH)
            Yrng = YH - YL
            Erng = EH - EL
            Y = Yrng / 3.
            E = Erng / 3.
            # Yield
            Y1 = YL
            Y2 = YL + Y
            Y3 = YH - Y #YL + 2*Y # 
            Y4 = YH #YL + 3*Y
            # Environment
            E1 = EL
            E2 = EL + E
            E3 = EH - E
            E4 = EH
            if ((len(target)>1) and (('BH' in target) or ('AH' in target)) ):
                # Add hztal and vertical lines
                ax2.axvline(Y2, ls='--', c='red', linewidth=1) #, label="E2")
                ax2.axvline(Y3, ls='--', c='red', linewidth=1) #, label="E2")
                # 
                tl_fg = 1.5
                tl_txt = 3
                minGY = df2[plot_params['fld1']].min()
                maxGY = df2[plot_params['fld1']].max()
                Oc_min = df2.loc[[df2[plot_params['fld2']].argmin()]]
                Oc_max = df2.loc[[df2[plot_params['fld2']].argmax()]]
                ax2.text(0.01, 0.99, r"$\bf b)$", fontsize=18, ha='left', va='top', transform=ax2.transAxes)

                if (plot_params['showTargetText'] is True):
                    ax2.text((maxlim/6)*1, maxlim-0.35, 'Low \nenvironments', 
                             {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                    ax2.text((maxlim/6)*3, maxlim-0.35, 'Medium \nenvironments', 
                             {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                    ax2.text((maxlim/6)*5, maxlim-0.35, 'High \nenvironments', 
                             {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])

            if (plot_params['dispTxt'] is True):
                count_AL = len(df2[df2['environment_m3']=='AL'])
                count_AM = len(df2[df2['environment_m3']=='AM']) 
                count_AH = len(df2[df2['environment_m3']=='AH']) 
                count_BL = len(df2[df2['environment_m3']=='BL'])
                count_BM = len(df2[df2['environment_m3']=='BM']) 
                count_BH = len(df2[df2['environment_m3']=='BH']) 
                count_U = len(df2[df2['environment_m3']=="Undefined"])
                
                if ((len(target)>1)):
                    ax2.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                             "Observations: " + r"$\bf" + str(len(df2))  + "$" 
                             + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$" 
                             + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$"
                             #+ "\nCountries: " + r"$\bf"+str(len(df['country'].unique()))+"$",
                             + "\nAL: " + r"$\bf"+str(count_AL)+"$" #
                             + ", AM: " + r"$\bf"+str(count_AM)+"$"
                             + ", AH: " + r"$\bf"+str(count_AH)+"$" 
                             + "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                             + ", BM: " + r"$\bf"+str(count_BM)+"$"
                             + ", BH: " + r"$\bf"+str(count_BH)+"$"
                             ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'],
                             transform=ax2.transAxes)
                else:
                    ax2.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df2))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'],
                         transform=ax2.transAxes)
        
        ax2.get_legend().remove()
        # ------------------------------
        # Chart 3
        # ------------------------------
        ax3 = axis[1, 0]
        title3='Top {} selected GIDs'.format(threshold)
        ax3.set_title('{}'.format(title3), fontsize=fonts_titles)
        ax3.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
        ax3.set_ylabel('Observed Grain Yield (Nursery-year)', fontsize=fonts_axes)
        ax3.tick_params(labelsize=12)
        
        if ((target is None) and (len(target)<=0)):
            print("Target not defined")
        
        df2 = df[df['environment_m3'].isin(target)]
        df2.reset_index(drop=True, inplace=True)
        
        if (selGIDs_m3 is not None):
            df2 = df2[(
                (df2['target_m3']==1) & (df2['G'].isin(selGIDs_m3['G'].unique()))
            )].reset_index(drop=True)
        
        g3 = sns.scatterplot(x=plot_params['fld1'], y=plot_params['fld2'], data=df2, 
                             alpha=plot_params['alpha3'], color=plot_params['defaultcolor1'], 
                             #palette=plot_params['palette2'],
                             hue='G', style='G',
                             #hue='environment_m3', #style='environment_m3',
                             s=plot_params['s3'], lw=plot_params['lw1'], ax=ax3);
        ax3.axline((0, 0), slope=1, color=plot_params['l1_1_color'], ls=plot_params['l1_1_ls'], 
                   linewidth=plot_params['l1_1_lw'], zorder=plot_params['l1_1_z'], label=plot_params['l1_1_label'])
        maxlim = int(max(df2[plot_params['fld1']].max(), df2[plot_params['fld2']].max())) + plot_params['limspan']
        g3.set(xlim=(0, maxlim), ylim=(0, maxlim))
        if (plot_params['dispGrid'] is True):
            g3.grid(visible=True, which='major', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            g3.grid(visible=True, which='minor', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
            ax3.set_axisbelow(True)
        #ax3.axis('equal')
        
        # Add classification texts
        # Add stats
        minGY = df2[plot_params['fld1']].min()
        maxGY = df2[plot_params['fld1']].max()
        GY_min = df2.loc[[df2[plot_params['fld2']].argmin()]]
        GY_max = df2.loc[[df2[plot_params['fld2']].argmax()]]
        if (qd==4):
            # Add hztal and vertical lines
            #ax3.axhline(df[plot_params['fld1']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (GID)")
            ax3.axvline(df[plot_params['fld2']].mean(), ls='--', c='red', linewidth=1, label="Mean Yield (site-year)")
            #ax3.text(maxlim-0.05, df[plot_params['fld1']].mean()+0.25, 'above yield', {'ha': 'right', 'va': 'center'}, rotation=0)
            #ax3.text(maxlim-0.05, df[plot_params['fld1']].mean()-0.25, 'below yield', {'ha': 'right', 'va': 'center'}, rotation=0)
            #ax3.text(df[plot_params['fld2']].mean()+0.25, maxlim-0.05, 'high environment', {'ha': 'center', 'va': 'top'}, rotation=90)
            #ax3.text(df[plot_params['fld2']].mean()-0.25, maxlim-0.05, 'low environment', {'ha': 'center', 'va': 'top'}, rotation=90)
            if (plot_params['dispTxt3'] is True):
                count_AL = len(df2[df2['environment_m3']=='AL'])
                #count_AH = len(df2[df2['environment_m3']=='AH']) 
                count_AH_B = len(df2[df2['environment_m3']=='AH_B']) 
                count_AH_A = len(df2[df2['environment_m3']=='AH_A'])
                #count_BL = len(df2[df2['environment_m3']=='BL'])
                count_BH = len(df2[df2['environment_m3']=='BH'])
                count_BL_B = len(df2[df2['environment_m3']=='BL_B'])
                count_BL_A = len(df2[df2['environment_m3']=='BL_A'])
                count_U = len(df2[df2['environment_m3']=="Undefined"])
                
                ax3.text(0.01, 0.99, r"$\bf c)$", fontsize=18, ha='left', va='top', transform=ax3.transAxes)
                ax3.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df2))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df2['country'].unique()))+"$",
                         #+ "\nAL: " + r"$\bf"+str(count_AL)+"$" #+ ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         #+ ", AH_B: " + r"$\bf"+str(count_AH_B)+"$" + ", AH_A: " + r"$\bf"+str(count_AH_A)+"$"
                         #+ "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         #+ "\nBH: " + r"$\bf"+str(count_BH)+"$"
                         #+ ", BL_B: " + r"$\bf"+str(count_BL_B)+"$" + ", BL_A: " + r"$\bf"+str(count_BL_A)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'], transform=ax3.transAxes)
        
        elif (qd==6):
            YL, YM, YH = df2[plot_params['fld1']].min(), df2[plot_params['fld1']].mean(), df2[plot_params['fld1']].max()
            EL, EM, EH = df2[plot_params['fld2']].min(), df2[plot_params['fld2']].mean(), df2[plot_params['fld2']].max()
            #print(YL, YM, YH, EL, EM, EH)
            Yrng = YH - YL
            Erng = EH - EL
            Y = Yrng / 3.
            E = Erng / 3.
            # Yield
            Y1 = YL
            Y2 = YL + Y
            Y3 = YH - Y #YL + 2*Y # 
            Y4 = YH #YL + 3*Y
            # Environment
            E1 = EL
            E2 = EL + E
            E3 = EH - E
            E4 = EH
            if ((len(target)>1) and (('BH' in target) or ('AH' in target)) ):
                # Add hztal and vertical lines
                ax3.axvline(Y2, ls='--', c='red', linewidth=1) #, label="E2")
                ax3.axvline(Y3, ls='--', c='red', linewidth=1) #, label="E2")
                # 
                tl_fg = 1.5
                tl_txt = 3
                minGY = df2[plot_params['fld1']].min()
                maxGY = df2[plot_params['fld1']].max()
                Oc_min = df2.loc[[df2[plot_params['fld2']].argmin()]]
                Oc_max = df2.loc[[df2[plot_params['fld2']].argmax()]]
                ax3.text(0.01, 0.99, r"$\bf c)$", fontsize=18, ha='left', va='top', transform=ax3.transAxes)

                if (plot_params['showTargetText'] is True):
                    ax3.text((maxlim/6)*1, maxlim-0.35, 'Low \nenvironments', 
                             {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                    ax3.text((maxlim/6)*3, maxlim-0.35, 'Medium \nenvironments', 
                             {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
                    ax3.text((maxlim/6)*5, maxlim-0.35, 'High \nenvironments', 
                             {'ha': 'center', 'va': 'top'}, color='gray', rotation=0, fontsize= plot_params['target_fontsize'])
         
            if (plot_params['dispTxt3'] is True):
                count_AL = len(df2[df2['environment_m3']=='AL'])
                count_AM = len(df2[df2['environment_m3']=='AM']) 
                count_AH = len(df2[df2['environment_m3']=='AH']) 
                count_BL = len(df2[df2['environment_m3']=='BL'])
                count_BM = len(df2[df2['environment_m3']=='BM']) 
                count_BH = len(df2[df2['environment_m3']=='BH']) 
                count_U = len(df2[df2['environment_m3']=="Undefined"])

                ax3.text(plot_params['xt_tl1'], plot_params['yt_tl1'], 
                         "Observations: " + r"$\bf" + str(len(df2))  + "$" 
                         + "\nGenotypes: " + r"$\bf"+str(len(df2['G'].unique()))+"$" 
                         + "\nLocations: " + r"$\bf"+str(len(df2['location'].unique()))+"$"
                         #+ "\nCountries: " + r"$\bf"+str(len(df['country'].unique()))+"$",
                         #+ "\nAL: " + r"$\bf"+str(count_AL)+"$" #
                         #+ ", AM: " + r"$\bf"+str(count_AM)+"$"
                         #+ ", AH: " + r"$\bf"+str(count_AH)+"$" 
                         #+ "\nBL: " + r"$\bf"+str(count_BL)+"$" 
                         #+ ", BM: " + r"$\bf"+str(count_BM)+"$"
                         #+ ", BH: " + r"$\bf"+str(count_BH)+"$"
                         ,fontsize=plot_params['stats_fontsize1'], ha=plot_params['ha1'], va=plot_params['va1'],
                         transform=ax3.transAxes)
        
        ax3.get_legend().remove()
        
        # Share Axis
        ax1.get_shared_x_axes().join(ax1, ax2, ax3)
        ax1.get_shared_y_axes().join(ax1, ax2, ax3)
    
        # ------------------------------
        # Chart 4
        # ------------------------------
        ax4 = axis[1, 1]
        #title4='Occurences 2'
        #ax4.set_title('{}'.format(title4), fontsize=fonts_titles)
        ax4.set_title('GIDs Occurrences vs Avg. Yield', fontsize=fonts_titles)
        #ax4.set_xlabel('Mean Grain Yield (Loc-Occ)', fontsize=12)
        ax4.set_xlabel('Mean Observed Grain Yield (Loc-Occ)', fontsize=fonts_axes)
        ax4.set_ylabel('# of GID Occurrences', fontsize=fonts_axes)
        ax4.tick_params(labelsize=12)
        
        noDisplayForNow = False
        if (noDisplayForNow is True):
            # Chart 4
            if ((target is None) and (len(target)<=0)):
                print("Target not defined")
            #df2 = df[df['environment_m3'].isin(target)]
            #df2.reset_index(drop=True, inplace=True)
            #fld3="AvGYxLocOcc_AL"
            #fld4="countOfOccurrences_AL"
            fld3="numOcurrences_avgGY"
            fld4="numOcurrences_target"

            df4 = numGIDsOfOcurrences.copy()
            df4.reset_index(drop=True, inplace=True)

            g4 = sns.scatterplot(x=fld3, y=fld4, data=df4, alpha=.85, markers='o',
                                 #color="black", 
                                 palette=occurrenceColors,
                                 hue='environment_m3', style='environment_m3',
                                 s=35, lw=1, ax=ax4 ); #label='# Occurrences',
            maxlim = int(max(df4[fld3].max(), df4[fld4].max())) + 1
            #g4.set(xlim=(0, maxlim), ylim=(0, maxlim))
            if (plot_params['dispGrid'] is True):
                g4.grid(visible=True, which='major', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
                g4.grid(visible=True, which='minor', color=plot_params['colorGrid'], linewidth=plot_params['lwGrid'])
                ax4.set_axisbelow(True)
            # Add texts
            if (plot_params['dispTxt4'] is True):
                tl_fg = 1.5
                tl_txt = 3
                minGY = df4[fld3].min()
                maxGY = df4[fld3].max()
                Oc_min = df4.loc[[df4[fld4].argmin()]]
                Oc_max = df4.loc[[df4[fld4].argmax()]]
                ax4.text(0.01, 0.99, r"$\bf d)$", fontsize=18, ha='left', va='top', transform=ax4.transAxes)
                ax4.text(plot_params['xt_tl4'], plot_params['yt_tl4']-.1, "Observations: " + r"$\bf" + str(len(df4))  + "$" + "\nGenotypes: " + r"$\bf"+str(len(df4['G'].unique()))+"$",
                     fontsize=10, ha=plot_params['ha4'], va=plot_params['va4'], transform=ax4.transAxes)

                # Add text of the country to the Maximum values per Nursery
                xtext = Oc_min[fld3].values[0]
                ytext = Oc_min[fld4].values[0]
                #ax4.text(xtext+0.05, ytext,'{}'.format(Oc_min['G'].values[0]),fontsize=8)
                ax4.text(.99, .2,'{}'.format(Oc_min['G'].values[0]),fontsize=8, ha='right', va='bottom', transform=ax4.transAxes)
                ax4.annotate("", xy=(xtext, ytext), xytext=(.95,.2),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"), 
                             textcoords='axes fraction', ha='center', va='center', transform=ax4.transAxes)

                # Maximum
                xtext = Oc_max[fld3].values[0]
                ytext = Oc_max[fld4].values[0]
                #ax4.text(xtext+0.05, ytext,'{}'.format(Oc_max['G'].values[0]),fontsize=8)
                #ax4.annotate("", xy=(xtext+0.05,ytext), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
                ax4.text(.99, .8,'{}'.format(Oc_max['G'].values[0]),fontsize=8, ha='right', va='top', transform=ax4.transAxes)
                ax4.annotate("", xy=(xtext, ytext), xytext=(.95,.8), 
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"), #"bar,armA=0.0,armB=0.0,fraction=-0.2,angle=180"), 
                             textcoords='axes fraction', ha='center', va='center', transform=ax4.transAxes)


            ax4.get_legend().remove()
        
    
    # ------------------------------------
    # Add Legend
    if (plot_params['showLegend'] is True):
        def getLegend_HandlesLabels(ax, handout, lablout):
            handles, labels = ax.get_legend_handles_labels()
            for h,l in zip(handles,labels):
                if l not in lablout:
                    if (l in legend_names):
                        if (l !='Undefined'):
                            lablout.append(legend_names[l])
                    else:
                        if (l !='Undefined'):
                            lablout.append(l)
                    handout.append(h)
            return handout, lablout
        if (ptype!=4):
            handout=[]
            lablout=[]
            #handles, labels = ax1.get_legend_handles_labels()
            handout, lablout = getLegend_HandlesLabels(ax1, handout, lablout)
            plt.legend(handout,lablout, bbox_to_anchor=(1.05, 1), loc=plot_params['loc'], ncol=plot_params['ncol'], 
                       borderaxespad=0,fontsize=plot_params['leg_fontsize'])
        else:
            # Add legend()
            def getLegend_HandlesLabels(ax, handout, lablout):
                handles, labels = ax.get_legend_handles_labels()
                for h,l in zip(handles,labels):
                    if l not in lablout:
                        if (l in legend_names):
                            if (l !='Undefined'):
                                lablout.append(legend_names[l])
                        else:
                            if (l !='Undefined'):
                                lablout.append(l)
                        handout.append(h)
                return handout, lablout
            handout=[]
            lablout=[]
            handout, lablout = getLegend_HandlesLabels(ax1, handout, lablout)
            handout, lablout = getLegend_HandlesLabels(ax4, handout, lablout)
            #handout, lablout = getLegend_HandlesLabels(ax2, handout, lablout)
            handout, lablout = getLegend_HandlesLabels(ax3, handout, lablout)

            plt.legend(handout,lablout, bbox_to_anchor=(-0.5, -0.65), loc="center", ncol=plot_params['ncol'], 
                       borderaxespad=0,fontsize=plot_params['leg_fontsize'])


    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, 'IWIN_GxE_WheatLines_GenSelxEnv_classifm3_{}_{}.pdf'
                                 .format(nursery.replace(' ', '_'), hoy)), bbox_inches='tight', orientation='portrait', 
                     pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_WheatLines_GenSelxEnv_classifm3_{}_{}.{}"
                                 .format(title.replace(' ', '_'), hoy, fmt)), bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();


# ------------------------------------------------------------------------------
# Display a figure with all combined methods
# ------------------------------------------------------------------------------
def figure_AvgYieldbyGID_combineMethods(df_GY=None, fld1="AvGYxLocOcc", fld2="AvGYxGID", nursery="", 
                                        lw=0.8, hue='G', s=15, alpha=.85, loc=2, ncol=2, 
                                        xt_tl=.01, yt_tl=.99, ha='left', va='top',
                                        showFig=True, saveFig=True, dirname='./', fmt='pdf'):
    '''
        Display a figure with all combined methods
    '''
    if (df_GY is None):
        print("Please check out your inputs...")
        return
    df = df_GY.copy()
    uGIDs = len(df['G'].unique())
    if (uGIDs < 20):
        df['displayGIDname'] = df[['G', 'GrandParent']].apply(lambda row: str(row['G']) +" "+ str(row['GrandParent']),  axis=1)
        hue = 'displayGIDname'
    df[['G', 'location']] = df[['G', 'location']].astype(str)
    fig, (ax1) = plt.subplots(figsize=(8,8)) #, constrained_layout=True)
    #fig.subplots_adjust(right=0.55)
    g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, color="#000000", hue=hue, s=s, lw=1, ax=ax1);
    ax1.axline((0, 0), slope=1, color='#444', ls="--", linewidth=1.25, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df[fld1].max(), df[fld2].max())) + 1
    g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title('{}'.format(nursery), fontsize=16)
    ax1.set_xlabel('Average Yield of site-year [${tha^{-1}}$]', fontsize=15)
    ax1.set_ylabel('Yield of GID [${tha^{-1}}$]', fontsize=15)

    ax1.tick_params(labelsize=12)

    # Add hztal and vertical lines
    #ax1.axhline(df[fld1].mean(), ls='--', c='red', zorder=0, linewidth=1, label="Mean Yield (GID)")
    #ax1.axvline(df[fld2].mean(), ls='--', c='red', zorder=0, linewidth=1, label="Mean Yield (site-year)")
    
    # Add texts
    #tl_txt = 2.5 tl_fg = 1
    minGY = df[fld1].min()
    maxGY = df[fld1].max()
    GY_min = df.loc[[df[fld2].argmin()]]
    GY_max = df.loc[[df[fld2].argmax()]]
    #xt_tl = .01
    #yt_tl = .99
    #ax1.text(0.01, 0.99, r"$\bf a)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
    ax1.text(xt_tl, yt_tl, 
             "Observations: " + r"$\bf" + str(len(df))  + "$" 
             + "\nGenotypes: " + r"$\bf"+str(uGIDs)+"$" 
             + "\nLocations: " + r"$\bf"+str(len(df['location'].unique()))+"$"
             + "\nCountries: " + r"$\bf"+str(len(df['country'].unique()))+"$",
             fontsize=10, ha=ha, va=va, transform=ax1.transAxes)
    
    # Add legend()
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=3, borderaxespad=0, fontsize=8)
    handles, labels = ax1.get_legend_handles_labels()
    handout=[]
    lablout=[]
    for h,l in zip(handles,labels):
        if l not in lablout:
            lablout.append(l)
            handout.append(h)
    plt.legend(handout,lablout, bbox_to_anchor=(1.05, 1), loc=loc, ncol=ncol, borderaxespad=0,fontsize=10)

    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, 'IWIN_GxE_selGen_{}_finalenvironments_{}.pdf'
                                 .format(nursery.replace(' ', '_'), hoy)), bbox_inches='tight', orientation='portrait', 
                     pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_selGen_{}_finalenvironments_{}.{}"
                                 .format(title.replace(' ', '_'), hoy, fmt)), bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();

#

def _clusterYieldEnvironments(df_selGen=None, nurseryGroup='ESWYT', 
                              fld1="AvGYxGID", fld2="AvGYxLocOcc", dispText=True,
                              loc=2, ncol=1, alpha=.65, s=5, hue='YieldEnv', 
                              xt_tl=.01, yt_tl=.99, ha='left', va='top', saveFig=True, showFig=True, 
                              oneFigure=True, verbose=True, dirname='./', fmt='pdf'):
    ''' 
        
        Display clusters of Yield environments 
    
    '''
    
    title='{} Yield environments'.format(nurseryGroup)
    df = df_selGen.query(f'Nursery == "{nurseryGroup}"').reset_index(drop=True)
    x=df[fld1].to_numpy()
    y=df[fld2].to_numpy()
    data = list(zip(x, y))
    # Clusters
    np.random.seed(12345)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)
    C = kmeans.cluster_centers_

    #vemos el representante del grupo, el GID más cercano a su centroid
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
    if (verbose is True):
        print("nearest points to the center of groups", closest)

    # TODO: Bug: Esto simepre muestra un resultado diferente por lo que no es facil asignar un mismo color coherente siempre
    # Look up table needed to order clusters and labels to always obtain the same results in colors and legend
    idx = np.argsort(kmeans.cluster_centers_.sum(axis=1)) 
    LUT = np.zeros_like(idx)
    LUT[idx] = np.arange(3) #n_clusters
    df['YE'] = LUT[kmeans.labels_] #kmeans.labels_
    df['YieldEnv']=''
    df.loc[df['YE']==0, 'YieldEnv'] = 'LYE - Low Yield Environment'
    df.loc[df['YE']==1, 'YieldEnv'] = 'MYE - Medium Yield Environment'
    df.loc[df['YE']==2, 'YieldEnv'] = 'HYE - High Yield Environment' 
    
    paleta = {
        'LYE - Low Yield Environment': 'purple',
        'MYE - Medium Yield Environment': 'orange',
        'HYE - High Yield Environment': 'green'
    }

    avgGY = df['ObsYield'].mean()
    avgGrainYield = df[fld2].mean()
    fig, (ax1) = plt.subplots(figsize=(10,6))
    fig.subplots_adjust(right=0.55)
    g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, s=s, lw=1, palette=paleta,
                         color="#000000", hue='YieldEnv', ax=ax1);
    plt.scatter(C[:, 0], C[:, 1], marker='*',  s=45, c='r', label='Cluster center') 

    ax1.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
    maxlim = int(max(df[fld1].max(), df[fld2].max())) + 2
    g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title('{}'.format(title), fontsize=16)
    ax1.set_xlabel(r'Avg. Observed Yield [${tha^{-1}}$] (Loc-Occ)', fontsize=14)
    ax1.set_ylabel('Observed Yield [${tha^{-1}}$] (Nursery-year)', fontsize=14)
    ax1.tick_params(labelsize=12)
    # Add vertical lines
    ax1.axvline(avgGrainYield, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
    ax1.axhline(avgGY, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
    
    if (dispText is True):
        #for i, c in enumerate(C):
        #    ax1.text(.99, c[1]/maxlim,'{}'.format(closest[i]),fontsize=8, ha='right', va='center', transform=ax1.transAxes)
        #    ax1.annotate("", xy=(c[0], c[1]), xytext=(.85,c[1]/maxlim),
        #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), 
        #             textcoords='axes fraction', ha='center', va='center', transform=ax1.transAxes)

        # Add texts
        minGY = df[fld1].min()
        maxGY = df[fld1].max()
        GY_min = df.loc[[df[fld2].argmin()]]
        GY_max = df.loc[[df[fld2].argmax()]]
        #xt_tl = .01
        #yt_tl = .99
        #print("Number of Genotypes: "+ HT.bold + HT.fg.red + "{}".format(len(df['G'].value_counts())) + HT.reset)
        ax1.text( .01, .99,
                 "Observations: LYE:" + r"$\bf"+str(len(df[df['YE']==2]))+"$ MYE: " + r"$\bf"+str(len(df[df['YE']==0]))+"$ HYE: " + r"$\bf"+str(len(df[df['YE']==1]))+"$"
                 + "\nGenotypes: LYE:" + r"$\bf"+str(len(df[df['YE']==2]['G'].unique()))+"$ MYE: " + r"$\bf"+str(len(df[df['YE']==0]['G'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df[df['YE']==1]['G'].unique()))+"$"
                 + "\nLocations: LYE:" + r"$\bf"+str(len(df[df['YE']==2]['location'].unique()))+"$ MYE: "+ r"$\bf"+str(len(df[df['YE']==0]['location'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df[df['YE']==1]['location'].unique()))+"$"
                 + "\nCountries: LYE:" + r"$\bf"+str(len(df[df['YE']==2]['country'].unique()))+"$ MYE: " + r"$\bf"+str(len(df[df['YE']==0]['country'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df[df['YE']==1]['country'].unique()))+"$",
                 fontsize=9, ha=ha, va=va, transform=ax1.transAxes)

    # Add legend()
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=3, borderaxespad=0, fontsize=8)
    handles, labels = ax1.get_legend_handles_labels()
    handout=[]
    lablout=[]
    for h,l in zip(handles,labels):
        if l not in lablout:
            lablout.append(l)
            handout.append(h)
    plt.legend(handout,lablout, bbox_to_anchor=(1.05, 1), loc=loc, ncol=ncol, borderaxespad=0,fontsize=10)

    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, 'IWIN_GxE_selGen_{}_clusterenvironments_{}.pdf'
                                 .format(nurseryGroup.replace(' ', '_'), hoy)), bbox_inches='tight', orientation='portrait', 
                     pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_selGen_{}_clusterenvironments_{}.{}"
                                 .format(nurseryGroup.replace(' ', '_'), hoy, fmt)), bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
        
# --------------------------
# Cluster Yield Environment
# --------------------------
def clusterYieldEnvironments(df_selGen=None, nurseryGroup=['ESWYT'], fld1="AvGYxLocOcc", fld2="AvGYxGID", dispText=True, 
                             loc=2, ncol=1, alpha=.65, s=5, hue='YieldEnv',xy_lim=2, 
                             xt_tl = .01, yt_tl = .99, ha='left', va='top',
                             saveFig=True, showFig=True, oneFigure=True, verbose=True, 
                             sharex=False, sharey=False, dirname='./', fmt='pdf'):
    ''' 
        
        Display clusters of Yield environments 
    
    '''
    np.random.seed(12345)
    paleta = {
        'LYE - Low Yield Environment': 'purple',
        'MYE - Medium Yield Environment': 'orange',
        'HYE - High Yield Environment': 'green'
    }
    
    if (oneFigure is False):
        try: 
            if (len(nurseryGroup)>0):
                nurseryGroup = nurseryGroup[0]
            title='{} Yield environments'.format(nurseryGroup)
            df = df_selGen.query(f'Nursery == "{nurseryGroup}"').reset_index(drop=True)
            if (df is None or len(df)<=0):
                print("There are not values for this nursery: {}".format(nurseryGroup))
                return
            x=df[fld1].to_numpy()
            y=df[fld2].to_numpy()
            data = list(zip(x, y))
            # Clusters
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(data)
            #print(kmeans.inertia_)
            C = kmeans.cluster_centers_
            #vemos el representante del grupo, el GID más cercano a su centroid
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
            if (verbose is True):
                print("nearest points to the center of groups", closest)

            # TODO: Bug: Esto simepre muestra un resultado diferente por lo que no es facil asignar un mismo color coherente siempre
            # Look up table needed to order clusters and labels to always obtain the same results in colors and legend
            idx = np.argsort(kmeans.cluster_centers_.sum(axis=1)) 
            LUT = np.zeros_like(idx)
            LUT[idx] = np.arange(3) #n_clusters
            df['YE'] = LUT[kmeans.labels_] #kmeans.labels_
            df['YieldEnv']=''
            df.loc[df['YE']==0, 'YieldEnv'] = 'LYE - Low Yield Environment'
            df.loc[df['YE']==1, 'YieldEnv'] = 'MYE - Medium Yield Environment'
            df.loc[df['YE']==2, 'YieldEnv'] = 'HYE - High Yield Environment' 
        except Exception as err:
            print("Problems in Figure. Error:{}".format(err))
            return None

        avgGY = df['ObsYield'].mean()
        avgGrainYield = df[fld2].mean()
        fig, (ax1) = plt.subplots(figsize=(10,6))
        fig.subplots_adjust(right=0.55)
        g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, s=s, lw=1, palette=paleta,
                             color="#000000", hue='YieldEnv', ax=ax1);
        plt.scatter(C[:, 0], C[:, 1], marker='*',  s=45, c='r', label='Cluster center') 

        ax1.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
        maxlim = int(max(df[fld1].max(), df[fld2].max())) + 2
        g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
        g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax1.set_axisbelow(True)
        ax1.set_title('{}'.format(title), fontsize=16)
        ax1.set_xlabel(r'Avg. Observed Yield [${tha^{-1}}$] (Loc-Occ)', fontsize=14)
        ax1.set_ylabel('Observed Yield [${tha^{-1}}$] (Nursery-year)', fontsize=14)
        ax1.tick_params(labelsize=12)
        # Add vertical lines
        #ax1.axvline(avgGrainYield, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
        #ax1.axhline(avgGY, ls='--', c='red', linewidth=1, label="Mean Grain Yield")

        if (dispText is True):
            #for i, c in enumerate(C):
            #    ax1.text(.99, c[1]/maxlim,'{}'.format(closest[i]),fontsize=8, ha='right', va='center', transform=ax1.transAxes)
            #    ax1.annotate("", xy=(c[0], c[1]), xytext=(.85,c[1]/maxlim),
            #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), 
            #             textcoords='axes fraction', ha='center', va='center', transform=ax1.transAxes)

            # Add texts
            minGY = df[fld1].min()
            maxGY = df[fld1].max()
            GY_min = df.loc[[df[fld2].argmin()]]
            GY_max = df.loc[[df[fld2].argmax()]]
            #xt_tl = .70
            #yt_tl = .2
            #print("Number of Genotypes: "+ HT.bold + HT.fg.red + "{}".format(len(df['G'].value_counts())) + HT.reset)
            ax1.text( xt_tl, yt_tl,
                     "Observations: LYE:" + r"$\bf"+str(len(df[df['YE']==2]))+"$ MYE: " + r"$\bf"+str(len(df[df['YE']==0]))+"$ HYE: " + r"$\bf"+str(len(df[df['YE']==1]))+"$"
                     + "\nGenotypes: LYE:" + r"$\bf"+str(len(df[df['YE']==2]['G'].unique()))+"$ MYE: " + r"$\bf"+str(len(df[df['YE']==0]['G'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df[df['YE']==1]['G'].unique()))+"$"
                     + "\nLocations: LYE:" + r"$\bf"+str(len(df[df['YE']==2]['location'].unique()))+"$ MYE: "+ r"$\bf"+str(len(df[df['YE']==0]['location'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df[df['YE']==1]['location'].unique()))+"$"
                     + "\nCountries: LYE:" + r"$\bf"+str(len(df[df['YE']==2]['country'].unique()))+"$ MYE: " + r"$\bf"+str(len(df[df['YE']==0]['country'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df[df['YE']==1]['country'].unique()))+"$",
                     fontsize=9, ha=ha, va=va, transform=ax1.transAxes)

        # Add legend()
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=3, borderaxespad=0, fontsize=8)
        handles, labels = ax1.get_legend_handles_labels()
        handout=[]
        lablout=[]
        for h,l in zip(handles,labels):
            if l not in lablout:
                lablout.append(l)
                handout.append(h)
        plt.legend(handout,lablout, bbox_to_anchor=(1.05, 1), loc=loc, ncol=ncol, borderaxespad=0,fontsize=10)

        # Save in PDF
        if (saveFig is True and fmt=='pdf'):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path, 'IWIN_GxE_selGen_{}_clusterenvironments_{}.pdf'
                                     .format(nurseryGroup.replace(' ', '_'), hoy)), bbox_inches='tight', orientation='portrait', 
                         pad_inches=0.5, dpi=300)

        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path,"IWIN_selGen_{}_clusterenvironments_{}.{}"
                                     .format(nurseryGroup.replace(' ', '_'), hoy, fmt)), bbox_inches='tight', 
                        facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                        pad_inches=0.5, dpi=300)

        if (showFig is True):
            if (saveFig is False):
                plt.tight_layout()
            fig.show()
        else:
            del fig
            plt.close();
    # One Figure
    else:
        #fld1="AvGYxGID", fld2="AvGYxLocOcc",
        try:
            # ----------------------------
            # Figure 1 - 
            # ----------------------------
            title='{}'.format(nurseryGroup[0])
            df = df_selGen.query(f'Nursery == "{nurseryGroup[0]}"').reset_index(drop=True)
            if (df is None or len(df)<=0):
                print("There are not values for this nursery: {}".format(nurseryGroup))
                return
            #df.dropna(subset=[fld1, fld2], inplace=True)
            x=df[fld1].to_numpy()
            y=df[fld2].to_numpy()
            data_1 = list(zip(x, y))
            # Clusters
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(data_1)
            #print(kmeans.inertia_)
            C = kmeans.cluster_centers_
            #vemos el representante del grupo, el GID más cercano a su centroid
            #closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
            #if (verbose is True):
            #    print("nearest points to the center of groups", closest)
            idx = np.argsort(kmeans.cluster_centers_.sum(axis=1)) 
            LUT = np.zeros_like(idx)
            LUT[idx] = np.arange(3) #n_clusters
            df['YE'] = LUT[kmeans.labels_] #kmeans.labels_
            df['YieldEnv']=''
            df.loc[df['YE']==0, 'YieldEnv'] = 'LYE - Low Yield Environment'
            df.loc[df['YE']==1, 'YieldEnv'] = 'MYE - Medium Yield Environment'
            df.loc[df['YE']==2, 'YieldEnv'] = 'HYE - High Yield Environment' 
            avgGY = df['ObsYield'].mean()
            avgGrainYield = df[fld2].mean()
            
        except Exception as err:
            print("Problems in Figure 1. Error:{}".format(err))
        
        # ----------------------------
        # Figure 2 - 
        # ----------------------------
        try:
            title2='{}'.format(nurseryGroup[1])
            df2 = df_selGen.query(f'Nursery == "{nurseryGroup[1]}"').reset_index(drop=True)
            if (df2 is None or len(df2)<=0):
                print("There are not values for this nursery: {}".format(nurseryGroup))
                return
            #df.dropna(subset=[fld1, fld2], inplace=True)
            x=df2[fld1].to_numpy()
            y=df2[fld2].to_numpy()
            data_2 = list(zip(x, y))
            # Clusters
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(data_2)
            C2 = kmeans.cluster_centers_
            idx = np.argsort(kmeans.cluster_centers_.sum(axis=1)) 
            LUT = np.zeros_like(idx)
            LUT[idx] = np.arange(3) #n_clusters
            df2['YE'] = LUT[kmeans.labels_] #kmeans.labels_
            df2['YieldEnv']=''
            df2.loc[df2['YE']==0, 'YieldEnv'] = 'LYE - Low Yield Environment'
            df2.loc[df2['YE']==1, 'YieldEnv'] = 'MYE - Medium Yield Environment'
            df2.loc[df2['YE']==2, 'YieldEnv'] = 'HYE - High Yield Environment' 
            avgGY2 = df2['ObsYield'].mean()
            avgGrainYield2 = df2[fld2].mean()
        
        except Exception as err:
            print("Problems in Figure 2. Error:{}".format(err))
            
        # ----------------------------
        # Figure 3 - 
        # ----------------------------
        try:
            title3='{}'.format(nurseryGroup[2])
            df3 = df_selGen.query(f'Nursery == "{nurseryGroup[2]}"').reset_index(drop=True)
            if (df3 is None or len(df3)<=0):
                print("There are not values for this nursery: {}".format(nurseryGroup))
                return
            #df.dropna(subset=[fld1, fld2], inplace=True)
            x=df3[fld1].to_numpy()
            y=df3[fld2].to_numpy()
            data_3 = list(zip(x, y))
            # Clusters
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(data_3)
            #print(kmeans.inertia_)
            C3 = kmeans.cluster_centers_
            #vemos el representante del grupo, el GID más cercano a su centroid
            #closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
            #if (verbose is True):
            #    print("nearest points to the center of groups", closest)
            idx = np.argsort(kmeans.cluster_centers_.sum(axis=1)) 
            LUT = np.zeros_like(idx)
            LUT[idx] = np.arange(3) #n_clusters
            df3['YE'] = LUT[kmeans.labels_] #kmeans.labels_
            df3['YieldEnv']=''
            df3.loc[df3['YE']==0, 'YieldEnv'] = 'LYE - Low Yield Environment'
            df3.loc[df3['YE']==1, 'YieldEnv'] = 'MYE - Medium Yield Environment'
            df3.loc[df3['YE']==2, 'YieldEnv'] = 'HYE - High Yield Environment' 
            avgGY3 = df3['ObsYield'].mean()
            avgGrainYield3 = df3[fld2].mean()
        except Exception as err:
            print("Problems in Figure 3. Error:{}".format(err))
        
        # ----------------------------
        # Figure 4 - 
        # ----------------------------
        try:
            title4='{}'.format(nurseryGroup[3])
            df4 = df_selGen.query(f'Nursery == "{nurseryGroup[3]}"').reset_index(drop=True)
            if (df4 is None or len(df4)<=0):
                print("There are not values for this nursery: {}".format(nurseryGroup))
                return
            #df.dropna(subset=[fld1, fld2], inplace=True)
            x=df4[fld1].to_numpy()
            y=df4[fld2].to_numpy()
            data_4 = list(zip(x, y))
            # Clusters
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(data_4)
            #print(kmeans.inertia_)
            C4 = kmeans.cluster_centers_
            idx = np.argsort(kmeans.cluster_centers_.sum(axis=1)) 
            LUT = np.zeros_like(idx)
            LUT[idx] = np.arange(3) #n_clusters
            df4['YE'] = LUT[kmeans.labels_] #kmeans.labels_
            df4['YieldEnv']=''
            df4.loc[df4['YE']==0, 'YieldEnv'] = 'LYE - Low Yield Environment'
            df4.loc[df4['YE']==1, 'YieldEnv'] = 'MYE - Medium Yield Environment'
            df4.loc[df4['YE']==2, 'YieldEnv'] = 'HYE - High Yield Environment' 
            avgGY4 = df4['ObsYield'].mean()
            avgGrainYield4 = df4[fld2].mean()
            
        except Exception as err:
            print("Problems in Figure 4. Error:{}".format(err))

        # ----------
        # Initialise the subplot function using number of rows and columns
        fig, axis = plt.subplots(2, 2, figsize=(10,10), facecolor='white', constrained_layout=True, sharex=sharex, sharey=sharey)
        fig.suptitle('IWIN Yield Environments', fontsize=18, y=1.05)
        fonts_axes = 12
        fonts_titles = 14

        # for use hue and better legends, convert columns to string
        # df[['location']] = df[['location']].astype(str)

        # ------------------------------
        # Chart 1
        # ------------------------------
        ax1 = axis[0, 0]
        g1 = sns.scatterplot(x=fld1, y=fld2, data=df, alpha=alpha, s=s, lw=1, palette=paleta,
                             color="#000000", hue='YieldEnv', ax=ax1);
        sns.scatterplot(C[:, 0], C[:, 1], marker='*',  s=100, color='red', label='Cluster center', ax=ax1) 
        ax1.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
        maxlim = int(max(df[fld1].max(), df[fld2].max())) + xy_lim
        g1.set(xlim=(0, maxlim), ylim=(0, maxlim))
        g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax1.set_axisbelow(True)
        ax1.set_title('{}'.format(title), fontsize=fonts_titles)
        ax1.set_xlabel(r'Avg. Observed Yield [${tha^{-1}}$] (Loc-Occ)', fontsize=fonts_axes)
        ax1.set_ylabel('Observed Yield [${tha^{-1}}$] (Nursery-year)', fontsize=fonts_axes)
        ax1.tick_params(labelsize=12)
        #ax1.axvline(avgGrainYield, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
        #ax1.axhline(avgGY, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
        # Add texts
        #minGY = df[fld1].min() #maxGY = df[fld1].max()
        #GY_min = df.loc[[df[fld2].argmin()]] #GY_max = df.loc[[df[fld2].argmax()]]
        # Put text in the same place for all chart, using relative coordinates of the axes
        #xt_tl = .70
        #yt_tl = .2
        ax1.text(0.01, 0.99, r"$\bf (a)$", fontsize=18, ha='left', va='top', transform=ax1.transAxes)
        if (dispText is True):
            ax1.text( xt_tl, yt_tl-.1,
                     "Observations: LYE:" + r"$\bf"+str(len(df[df['YE']==2]))+"$ MYE: " + r"$\bf"+str(len(df[df['YE']==0]))+"$ HYE: " + r"$\bf"+str(len(df[df['YE']==1]))+"$"
                     + "\nGenotypes: LYE:" + r"$\bf"+str(len(df[df['YE']==2]['G'].unique()))+"$ MYE: " + r"$\bf"+str(len(df[df['YE']==0]['G'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df[df['YE']==1]['G'].unique()))+"$"
                     + "\nLocations: LYE:" + r"$\bf"+str(len(df[df['YE']==2]['location'].unique()))+"$ MYE: "+ r"$\bf"+str(len(df[df['YE']==0]['location'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df[df['YE']==1]['location'].unique()))+"$"
                     + "\nCountries: LYE:" + r"$\bf"+str(len(df[df['YE']==2]['country'].unique()))+"$ MYE: " + r"$\bf"+str(len(df[df['YE']==0]['country'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df[df['YE']==1]['country'].unique()))+"$",
                     fontsize=9, ha=ha, va=va, transform=ax1.transAxes)
        
        ax1.get_legend().remove()
        
        
        # ------------------------------
        # Chart 2
        # ------------------------------
        ax2 = axis[0, 1]
        g2 = sns.scatterplot(x=fld1, y=fld2, data=df2, alpha=alpha, s=s, lw=1, palette=paleta,
                             color="#000000", hue='YieldEnv', ax=ax2);
        sns.scatterplot(C2[:, 0], C2[:, 1], marker='*',  s=100, color='red', label='Cluster center', ax=ax2) 
        ax2.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
        maxlim = int(max(df2[fld1].max(), df2[fld2].max())) + xy_lim
        g2.set(xlim=(0, maxlim), ylim=(0, maxlim))
        g2.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g2.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax2.set_axisbelow(True)
        ax2.set_title('{}'.format(title2), fontsize=fonts_titles)
        ax2.set_xlabel(r'Avg. Observed Yield [${tha^{-1}}$] (Loc-Occ)', fontsize=fonts_axes)
        ax2.set_ylabel('Observed Yield [${tha^{-1}}$] (Nursery-year)', fontsize=fonts_axes)
        ax2.tick_params(labelsize=12)
        #ax2.axvline(avgGrainYield2, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
        #ax2.axhline(avgGY2, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
        # Add texts
        #xt_tl = .70
        #yt_tl = .2
        ax2.text(0.01, 0.99, r"$\bf (b)$", fontsize=18, ha='left', va='top', transform=ax2.transAxes)
        if (dispText is True):
            ax2.text( xt_tl, yt_tl-.1,
                     "Observations: LYE:" + r"$\bf"+str(len(df2[df2['YE']==2]))+"$ MYE: " + r"$\bf"+str(len(df2[df2['YE']==0]))+"$ HYE: " + r"$\bf"+str(len(df2[df2['YE']==1]))+"$"
                     + "\nGenotypes: LYE:" + r"$\bf"+str(len(df2[df2['YE']==2]['G'].unique()))+"$ MYE: " + r"$\bf"+str(len(df2[df2['YE']==0]['G'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df2[df2['YE']==1]['G'].unique()))+"$"
                     + "\nLocations: LYE:" + r"$\bf"+str(len(df2[df2['YE']==2]['location'].unique()))+"$ MYE: "+ r"$\bf"+str(len(df2[df2['YE']==0]['location'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df2[df2['YE']==1]['location'].unique()))+"$"
                     + "\nCountries: LYE:" + r"$\bf"+str(len(df2[df2['YE']==2]['country'].unique()))+"$ MYE: " + r"$\bf"+str(len(df2[df2['YE']==0]['country'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df2[df2['YE']==1]['country'].unique()))+"$",
                     fontsize=9, ha=ha, va=va, transform=ax2.transAxes)
        ax2.get_legend().remove()
        
        # ------------------------------
        # Chart 3
        # ------------------------------
        ax3 = axis[1, 0]
        g3 = sns.scatterplot(x=fld1, y=fld2, data=df3, alpha=alpha, s=s, lw=1, palette=paleta,
                             color="#000000", hue='YieldEnv', ax=ax3);
        sns.scatterplot(C3[:, 0], C3[:, 1], marker='*',  s=100, color='red', label='Cluster center', ax=ax3) 
        ax3.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
        maxlim = int(max(df3[fld1].max(), df3[fld2].max())) + xy_lim
        g3.set(xlim=(0, maxlim), ylim=(0, maxlim))
        g3.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g3.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax3.set_axisbelow(True)
        ax3.set_title('{}'.format(title3), fontsize=fonts_titles)
        ax3.set_xlabel(r'Avg. Observed Yield [${tha^{-1}}$] (Loc-Occ)', fontsize=fonts_axes)
        ax3.set_ylabel('Observed Yield [${tha^{-1}}$] (Nursery-year)', fontsize=fonts_axes)
        ax3.tick_params(labelsize=12)
        #ax3.axvline(avgGrainYield3, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
        #ax3.axhline(avgGY3, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
        # Add texts
        #xt_tl = .70
        #yt_tl = .2
        ax3.text(0.01, 0.99, r"$\bf (c)$", fontsize=18, ha='left', va='top', transform=ax3.transAxes)
        if (dispText is True):
            ax3.text( xt_tl, yt_tl-.1,
                     "Observations: LYE:" + r"$\bf"+str(len(df3[df3['YE']==2]))+"$ MYE: " + r"$\bf"+str(len(df3[df3['YE']==0]))+"$ HYE: " + r"$\bf"+str(len(df3[df3['YE']==1]))+"$"
                     + "\nGenotypes: LYE:" + r"$\bf"+str(len(df3[df3['YE']==2]['G'].unique()))+"$ MYE: " + r"$\bf"+str(len(df3[df3['YE']==0]['G'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df3[df3['YE']==1]['G'].unique()))+"$"
                     + "\nLocations: LYE:" + r"$\bf"+str(len(df3[df3['YE']==2]['location'].unique()))+"$ MYE: "+ r"$\bf"+str(len(df3[df3['YE']==0]['location'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df3[df3['YE']==1]['location'].unique()))+"$"
                     + "\nCountries: LYE:" + r"$\bf"+str(len(df3[df3['YE']==2]['country'].unique()))+"$ MYE: " + r"$\bf"+str(len(df3[df3['YE']==0]['country'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df3[df3['YE']==1]['country'].unique()))+"$",
                     fontsize=9, ha=ha, va=va, transform=ax3.transAxes)
        ax3.get_legend().remove()
        
        # ------------------------------
        # Chart 4
        # ------------------------------
        ax4 = axis[1, 1]
        g4 = sns.scatterplot(x=fld1, y=fld2, data=df4, alpha=alpha, s=s, lw=1, palette=paleta,
                             color="#000000", hue='YieldEnv', ax=ax4);
        sns.scatterplot(C4[:, 0], C4[:, 1], marker='*',  s=100, color='red', label='Cluster center', ax=ax4) 
        ax4.axline((0, 0), slope=1, color='#444', ls="-", linewidth=0.75, zorder=0, label="line 1:1") #c=".5",
        maxlim = int(max(df4[fld1].max(), df4[fld2].max())) + xy_lim
        g4.set(xlim=(0, maxlim), ylim=(0, maxlim))
        g4.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
        g4.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
        ax4.set_axisbelow(True)
        ax4.set_title('{}'.format(title4), fontsize=fonts_titles)
        ax4.set_xlabel(r'Avg. Observed Yield [${tha^{-1}}$] (Loc-Occ)', fontsize=fonts_axes)
        ax4.set_ylabel('Observed Yield [${tha^{-1}}$] (Nursery-year)', fontsize=fonts_axes)
        ax4.tick_params(labelsize=12)
        #ax4.axvline(avgGrainYield4, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
        #ax4.axhline(avgGY4, ls='--', c='red', linewidth=1, label="Mean Grain Yield")
        # Add texts
        ax4.text(0.01, 0.99, r"$\bf (d)$", fontsize=18, ha='left', va='top', transform=ax4.transAxes)
        if (dispText is True):
            ax4.text( xt_tl, yt_tl-.1,
                     "Observations: LYE:" + r"$\bf"+str(len(df4[df4['YE']==2]))+"$ MYE: " + r"$\bf"+str(len(df4[df4['YE']==0]))+"$ HYE: " + r"$\bf"+str(len(df4[df4['YE']==1]))+"$"
                     + "\nGenotypes: LYE:" + r"$\bf"+str(len(df4[df4['YE']==2]['G'].unique()))+"$ MYE: " + r"$\bf"+str(len(df4[df4['YE']==0]['G'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df4[df4['YE']==1]['G'].unique()))+"$"
                     + "\nLocations: LYE:" + r"$\bf"+str(len(df4[df4['YE']==2]['location'].unique()))+"$ MYE: "+ r"$\bf"+str(len(df4[df4['YE']==0]['location'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df4[df4['YE']==1]['location'].unique()))+"$"
                     + "\nCountries: LYE:" + r"$\bf"+str(len(df4[df4['YE']==2]['country'].unique()))+"$ MYE: " + r"$\bf"+str(len(df4[df4['YE']==0]['country'].unique()))+"$ HYE: "+ r"$\bf"+str(len(df4[df4['YE']==1]['country'].unique()))+"$",
                     fontsize=9, ha=ha, va=va, transform=ax4.transAxes)
        ax4.get_legend().remove()
        
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

        fig.legend(handout,lablout, bbox_to_anchor=(0.5, -0.1), loc="center", ncol=ncol, 
                   borderaxespad=0,fontsize=10) #, fancybox=True, shadow=True)

        # Save in PDF
        if (saveFig is True and fmt=='pdf'):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path, 'IWIN_GxE_selGen_clusterenvironments_{}.pdf'
                                     .format(hoy)), bbox_inches='tight', orientation='portrait', 
                         pad_inches=0.5, dpi=300)

        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path,"IWIN_selGen_clusterenvironments_{}.{}"
                                     .format(hoy, fmt)), bbox_inches='tight', 
                        facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                        pad_inches=0.5, dpi=300)

        if (showFig is True):
            if (saveFig is False):
                plt.tight_layout()
            fig.show()
        else:
            del fig
            plt.close();
    
#
#
def chartObsYieldTrendsByNursery(df_pheno=None, GenSel=None, ngrp='ESWYT', 
                                 title='Observed yield trends', marker='o', s=15, alpha=0.53,
                                 addMaxMinTexts=False, showFig=True, saveFig=True, dirname='./', fmt='pdf'):
    '''
    
    '''
    global nursery
    global colors
    global paleta
    
    df=df_pheno[df_pheno['Nursery']==ngrp] #.copy()
    df.reset_index(drop=True, inplace=True)
    
    minYear = int(np.nanmin(df['YearofSow']))
    maxYear = int(np.nanmax(df['YearofSow']))
    GY_min = df.loc[[df['ObsYield'].argmin()]]
    GY_max = df.loc[[df['ObsYield'].argmax()]]

    fig,ax1=plt.subplots(nrows=1,ncols=1,figsize=(10,6))
    g1 = sns.scatterplot(x='YearofSow', y='ObsYield', data=df, marker=marker, s=s, alpha=alpha, 
                         hue='loc_code',  ax=ax1);
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    minlim = minYear - 1
    maxlim = maxYear + 2
    g1.set(xlim=(minlim, maxlim)) #, ylim=(0, maxlim)

    ax1.set_title('{} - {} ({:.0f} - {:.0f})'.format(ngrp, title, minYear, maxYear), fontsize=18)
    ax1.set_xlabel('Year',fontsize=15)
    ax1.set_ylabel('Observed Yield (t/ha)',fontsize=15)

    if (addMaxMinTexts is True):
        # Add text of the Country to the Maximum values per Nursery
        xtext = GY_min['YearofSow'].values[0]
        ytext = GY_min['ObsYield'].values[0]
        ax1.text(xtext+2, ytext-0.25,'{} - {} - {:.2f} t/ha'
                 .format(GY_min['country'].values[0], GY_min['loc_code'].values[0], ytext),fontsize=8)
        ax1.annotate("", xy=(xtext+2,ytext-0.25), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
        # Maximum
        xtext = GY_max['YearofSow'].values[0]
        ytext = GY_max['ObsYield'].values[0]
        ax1.text(xtext+1, ytext+0.1,'{} - {} - {:.2f} t/ha'
                 .format(GY_max['country'].values[0], GY_max['loc_code'].values[0], ytext),fontsize=8)
        ax1.annotate("", xy=(xtext+1,ytext+0.1), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))

    plt.legend(loc=2,bbox_to_anchor=(1.05, 1), ncol=2, borderaxespad=0, fontsize=10)

    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, "IWIN_GxE_{}_{}_{}-{}_{}.pdf"
                                 .format(ngrp, title.replace(' ', '_'), minYear, maxYear, hoy)), 
                    bbox_inches='tight', orientation='portrait',  pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_GxE_{}_{}_{}-{}_{}.{}"
                                 .format(ngrp, title.replace(' ', '_'), minYear, maxYear, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
        
#
def chartErrorBar_ObsYieldTrendsByNursery(df_pheno=None, GenSel=None, ngrp='ESWYT', title='Observed yield trends', 
                                          marker='o', s=15, alpha=0.53,  addMaxMinTexts=False, t='bar', 
                                          saveFig=True, showFig=True, dirname='./', fmt='pdf'):
    '''
    '''
    global nursery
    global colors
    global paleta
    
    df=df_pheno[(df_pheno['Nursery']==ngrp) & (df_pheno['G'].isin(GenSel))]\
    .pivot_table(index=['loc_code','YearofSow'],values=['ObsYield']).reset_index()
    # rename some attributes for better visualization
    df['loc_code'] = df['loc_code'].replace({'ESWYT':'','IDYN':'','HTWYT':'','SAWYT':'',
                                           'ELITE SPRING WHEAT YT':'', 'V13':'13TH', '31st':'31ST',
                                           'ELITE SELECTION WHEAT YT':'',#'ESWYT', 
                                           'INTERNATIONAL DURUM YN':'',#'IDYN',
                                           'HIGH TEMPERATURE WHEAT YT':'',#'HTWYT'
                                             'SEMI-ARID WHEAT YT':'' #'SAWYT'
                                          }, regex=True)
    df['loc_code'] = df['loc_code'].str.strip()
    # Extract the number of trial name to order in chart
    df['numtrial'] = df['loc_code'].apply(lambda x: int(re.findall("\d+", x)[0]))
    df = df.sort_values(by=['numtrial'], ascending=True)
    minYear = int(np.nanmin(df['YearofSow']))
    maxYear = int(np.nanmax(df['YearofSow']))
    GY_min = df.loc[[df['ObsYield'].argmin()]]
    GY_max = df.loc[[df['ObsYield'].argmax()]]

    fig,ax1=plt.subplots(nrows=1,ncols=1,figsize=(10,6))
    if (t=='bar'):
        g1 = sns.barplot(x='loc_code', y='ObsYield', data=df, alpha=alpha, capsize=.2, ax=ax1);
    else:
        g1 = sns.boxplot(x='loc_code', y='ObsYield', data=df,  
                     #notch=True, showcaps=False, flierprops={"marker": "x"},
                     #boxprops={"facecolor": (.4, .6, .8, .5)}, medianprops={"color": "coral"},
                     ax=ax1);

    ax1.set_title('{} - {} ({:.0f} - {:.0f})'.format(ngrp, title, minYear, maxYear), fontsize=18)
    ax1.set_xlabel('Trial name',fontsize=15)
    ax1.set_ylabel('Observed Yield (t/ha)',fontsize=15)
    plt.xticks(rotation=90)

    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, "IWIN_GxE_{}_GenSel_{}_{}-{}_{}_{}.pdf"
                                 .format(ngrp, title.replace(' ', '_'), minYear, maxYear, t, hoy)), 
                    bbox_inches='tight', orientation='portrait',  pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_GxE_{}_GenSel_{}_{}-{}_{}_{}_{}.{}"
                                 .format(ngrp, title.replace(' ', '_'), minYear, maxYear, t, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
    
#
def chartObsYieldTrends(df_GY=None, selGIDs=None, ngrp='ESWYT', n=1, title='Observed yield trends',
                        sx_min=2, sy_min=0.05, sx_max=1, sy_max=0.03, addMaxMinTexts=True, 
                        saveFig=True, showFig=True, dirname='./', fmt='pdf'):
    ''' 
        Display Observed Yield Trends 
    
    '''
    global nursery
    global colors
    global paleta
    
    df=df_GY[['country', 'loc_code','Nursery', 'YearofSow', 'G', 'Days_To_Heading', 'Days_To_Maturity', 'ObsYield']]\
    [(df_GY['Nursery']==ngrp) & (df_GY['G'].isin(selGIDs))]\
    .groupby(['YearofSow'], as_index=False).agg({
        'loc_code':'first', 'Nursery':'first', 'Days_To_Heading':'mean', 'Days_To_Maturity':'mean', 'ObsYield':'mean'
    }).reset_index()

    #minYear = int(df['YearofSow'].min())
    #maxYear = int(df['YearofSow'].max())
    minYear = int(np.nanmin(df['YearofSow']))
    maxYear = int(np.nanmax(df['YearofSow']))
    #
    GY_min = df.loc[[df['ObsYield'].argmin()]]
    GY_max = df.loc[[df['ObsYield'].argmax()]]

    fig,ax1=plt.subplots(nrows=1,ncols=1,figsize=(10,6))
    #ax1.scatter(df['YearofSow'],df['ObsYield'],c=df['Nursery'],alpha=0.3,s=5)
    g1 = sns.scatterplot(x='YearofSow', y='ObsYield', data=df, marker='o', s=50, alpha=0.83, 
                         #color=colors[n], 
                         palette=paleta, 
                         hue='Nursery',  ax=ax1);
    g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
    g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
    ax1.set_axisbelow(True)
    minlim = minYear - 1
    maxlim = maxYear + 2
    g1.set(xlim=(minlim, maxlim)) #, ylim=(0, maxlim)

    # Add trend lines
    #for n in range(len(nursery)):
    df2 = df_GY[df_GY['Nursery']==nursery[n]]
    df2=df2[['country', 'loc_code', 'Nursery', 'YearofSow', 'G', 'Days_To_Heading', 'Days_To_Maturity', 'ObsYield']]\
    .groupby(['YearofSow'], as_index=False).agg({
        'country':'first', 'loc_code':'first', 'Nursery':'first', 'Days_To_Heading':'mean', 
        'Days_To_Maturity':'mean', 'ObsYield':'mean'
    }).reset_index()
    sns.regplot(df2['YearofSow'],df2['ObsYield'],ax=ax1,truncate=False,scatter=False, 
                line_kws={'lw': 1, 'color': colors[n],'linestyle':'--'},label=nursery[n]+"")

    ax1.set_title('{} - {} ({:.0f} - {:.0f})'.format(ngrp, title, minYear, maxYear), fontsize=18)
    ax1.set_xlabel('Year',fontsize=15)
    ax1.set_ylabel('Avg. Yield (t/ha)',fontsize=15)

    #TODO: Add number of observations (Total and per Nursery)
    # - Add text of the Country to the Maximum values per Nursery
    if (addMaxMinTexts is True):
        # Add text of the Country to the Maximum values per Nursery
        xtext = GY_min['YearofSow'].values[0]
        ytext = GY_min['ObsYield'].values[0]
        ax1.text(xtext+sx_min, ytext+sy_min,'{} - {:.2f} t/ha'.format(GY_min['loc_code'].values[0], ytext),fontsize=8)
        ax1.annotate("", xy=(xtext+sx_min, ytext+sy_min), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))
        # Maximum
        xtext = GY_max['YearofSow'].values[0]
        ytext = GY_max['ObsYield'].values[0]
        ax1.text(xtext+sx_max, ytext+sy_max,'{} - {:.2f} t/ha'.format(GY_max['loc_code'].values[0], ytext),fontsize=8)
        ax1.annotate("", xy=(xtext+sx_max,ytext+sy_max), xytext=(xtext, ytext),arrowprops=dict(arrowstyle="->"))

    plt.legend(loc=2,bbox_to_anchor=(1.05, 1), ncol=2, borderaxespad=0, fontsize=10)
    
    # Save in PDF
    if (saveFig is True and fmt=='pdf'):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path, "IWIN_GxE_{}_GenSel_{}_{}-{}_mean_{}.pdf"
                                 .format(ngrp, title.replace(' ', '_'), minYear, maxYear, hoy)), 
                    bbox_inches='tight', orientation='portrait',  pad_inches=0.5, dpi=300)

    if (saveFig==True and (fmt=='jpg' or fmt=='png')):
        hoy = datetime.now().strftime('%Y%m%d')
        figures_path = '{}_{}'.format(dirname, hoy)
        if not os.path.isdir(figures_path):
            os.makedirs(figures_path)
        fig.savefig(os.path.join(figures_path,"IWIN_GxE_{}_GenSel_{}_{}-{}_{}_mean_{}.{}"
                                 .format(ngrp, title.replace(' ', '_'), minYear, maxYear, hoy, fmt)), 
                    bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', transparent=False, 
                    pad_inches=0.5, dpi=300)

    if (showFig is True):
        if (saveFig is False):
            plt.tight_layout()
        fig.show()
    else:
        del fig
        plt.close();
        
#





