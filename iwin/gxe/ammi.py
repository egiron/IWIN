# coding=utf-8
# Load libraries and existing datasets
from __future__ import absolute_import, division, print_function

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

__version__ = "IWIN version 1.0.1.dev"
__author__ = "Ernesto Giron Echeverry, Urs Christoph Schulthess et al."
__copyright__ = "Copyright (c) 2022 CIMMYT-Henan Collaborative Innovation Center"
__license__ = "Public Domain"

import os, gc
import numpy as np
import pandas as pd
from datetime import date, datetime
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import *
from ..data import *
from ..util import *


class AMMI(object):
    def __init__(self, data, trait='', trialname='', params=None):
        self.data = data
        self.trait = trait
        self.trialname = trialname
        self.params = params
        if (params is not None):
            self.params['centering'] = params['centering'] if ('centering' in params.keys()) else False
            self.params['scaling'] = params['scaling'] if ('scaling' in params.keys()) else False
            self.params['n_components'] = params['n_components'] if ('n_components' in params.keys()) else None
            n_samples, n_features = data.shape
            min_components = min(n_samples, n_features) #- 1
            if ((self.params['n_components'] is not None) and ((self.params['n_components']==1) or (self.params['n_components']>min_components))):
                print("Error in number of components to keep")
                return
        else:
            self.params = {'centering':False, 'scaling':False, 'n_components':None}
    
    def cal_env_residuals(self):
        '''Cálculo de residuales con respecto a la media de los ambientes'''
        self.env_mean =  self.data.mean(axis=1)
        self.env_residuals = self.data.apply(lambda x: x - self.env_mean, axis=0)
        return self.env_residuals
    
    def calc_gen_residuals(self):
        '''Cálculo de residuales con respecto a la media de los genotipos'''
        self.gen_mean =  self.data.mean(axis=0)
        self.gen_residuals = self.data.apply(lambda x: x - self.gen_mean, axis=1)
        return self.gen_residuals
    
    def PCA_Env(self, verbose=False):
        # Standardize features by removing the mean and scaling to unit variance.
        env_scaling = StandardScaler(with_mean=self.params['centering'], with_std=self.params['scaling']).fit_transform(self.env_residuals)
        pca_out = PCA(n_components=self.params['n_components']).fit(env_scaling) #svd_solver='full'
        #pca_out = PCA().fit(self.env_residuals)
        loadings = pca_out.components_ # component loadings
        # V matrix of U LAMBDA V' decomposition
        PC1 = loadings[0]*-1
        PC2 = loadings[1] #*-1
        mat1 = pd.DataFrame(data=[list(self.data), PC1, PC2]).T
        mat1.columns=['E','PC1', 'PC2']
        mat1[['PC1', 'PC2']] = mat1[['PC1', 'PC2']].astype(float).round(3)
        #self.PCA_Env = mat1
        # get eigenvalues (variance explained by each PC)  
        #print(pca_out.mean_)
        SV_PC1 = round(float(pca_out.singular_values_[0]), 2)
        SV_PC2 = round(float(pca_out.singular_values_[1]), 2)
        #print("Singular value PC1: {} - PC2: {}".format(SV_PC1, SV_PC2))
        EV_PC1 = round(float(pca_out.explained_variance_ratio_[0]*100),2)
        EV_PC2 = round(float(pca_out.explained_variance_ratio_[1]*100),2)
        #print("Explained variance PC1: {} - PC2: {}".format(EV_PC1, EV_PC2))
        self.EV_PC1 = EV_PC1
        self.EV_PC2 = EV_PC2
        cumEV_PC1 = round(float(np.cumsum(pca_out.explained_variance_ratio_)[0]*100), 2)
        cumEV_PC2 = round(float(np.cumsum(pca_out.explained_variance_ratio_)[1]*100), 2)
        self.cumEV_PC1 = cumEV_PC1
        self.cumEV_PC2 = cumEV_PC2
        if (verbose is True):
            print("Cumulative explained variance PC1: {:.2f}% - PC2: {:.2f}%".format(cumEV_PC1, cumEV_PC2))
        # Sum of Eigenvalues
        #print("Sum of Eigenvalues", EV_PC1 + EV_PC2) #pca_out2.explained_variance_ratio_) # #pca_out2.explained_variance_ratio_) #
        
        mat1_pc1 = mat1.iloc[:,1:2].apply(lambda x: x * np.sqrt(SV_PC1), axis=1)
        mat1_pc2 = mat1.iloc[:,2:3].apply(lambda x: x * np.sqrt(SV_PC2), axis=1)
        mat1a = pd.concat([mat1_pc1, mat1_pc2], axis=1)
        mat1a = mat1a.astype(float).round(2)
        
        self.PCA_Env = mat1a
        return mat1, mat1a
    
    def PCA_Gen(self, verbose=False):
        # Standardize features by removing the mean and scaling to unit variance.
        gen_scaling = StandardScaler(with_mean=self.params['centering'], with_std=self.params['scaling']).fit_transform(self.gen_residuals.T)
        pca_out = PCA(n_components=self.params['n_components']).fit(gen_scaling) #svd_solver='full', arpack
        #pca_out = PCA().fit(self.gen_residuals.T)
        loadings = pca_out.components_
        # U matrix of U LAMBDA V' decomposition
        PC1 = loadings[0]*-1
        PC2 = loadings[1]*-1
        mat2 = pd.DataFrame(data=[list(self.data.index), PC1, PC2]).T
        mat2.columns=['G','PC1', 'PC2']
        mat2[['PC1', 'PC2']] = mat2[['PC1', 'PC2']].astype(float).round(3)
        #self.PCA_Gen = mat2
        
        SV_PC1 = round(float(pca_out.singular_values_[0]), 2)
        SV_PC2 = round(float(pca_out.singular_values_[1]), 2)
        cumEV_PC1 = round(float(np.cumsum(pca_out.explained_variance_ratio_)[0]*100), 2)
        cumEV_PC2 = round(float(np.cumsum(pca_out.explained_variance_ratio_)[1]*100), 2)
        #self.cumEV_PC1 = cumEV_PC1
        #self.cumEV_PC2 = cumEV_PC2
        if (verbose is True):
            print("Cumulative explained variance PC1: {:.2f}% - PC2: {:.2f}%".format(cumEV_PC1, cumEV_PC2))
        
        mat2_pc1 = mat2.iloc[:,1:2].apply(lambda x: x * np.sqrt(SV_PC1), axis=1)
        mat2_pc2 = mat2.iloc[:,2:3].apply(lambda x: x * np.sqrt(SV_PC2), axis=1)
        mat2a = pd.concat([mat2_pc1, mat2_pc2], axis=1)
        mat2a = mat2a.astype(float).round(2)
        
        self.PCA_Gen = mat2a
        return mat2, mat2a
    
    def fit(self):
        _ = self.cal_env_residuals()
        _ = self.calc_gen_residuals()
        _ = self.PCA_Env() # V matrix of U LAMBDA V' decomposition
        _ = self.PCA_Gen() # U matrix of U LAMBDA V' decomposition
    
    
    def plot(self, title=None, bptype=0, plot_params=None, 
             saveFig=True, showFig=True, dirname='./', fmt='pdf'):
        ''' Display biplot '''
        plot_default_params = dict(
                limspan=1.2, labelG=True, labelE=True, txtspan=0.02, arrowE=False, arrowG=True, 
                arrowpropsE=dict(arrowstyle= '->', color='red', lw=1, ls='-', alpha=0.7), 
                arrowpropsG=dict(arrowstyle= '-', color='gray', lw=1, ls='-', alpha=0.5),
                fontsizeE=7, colorE='red', sE=20, alphaE=0.75, hueE=None, colorLabelE='red', haE='center', vaE='bottom',
                fontsizeG=9.5, colorG='blue', sG=40, alphaG=0.75, hueG='G', colorLabelG='black', haG='center', vaG='bottom',
                limspan2=1.2, labelG2=True, labelE2=True, txtspan2=0.05, arrowE2=False, arrowG2=True, 
                arrowpropsE2=dict(arrowstyle= '-', color='gray', lw=1, ls='--', alpha=0.57), 
                arrowpropsG2=dict(arrowstyle= '-', color='gray', lw=1, ls='--', alpha=0.5),
                fontsizeE2=7, colorE2='red', sE2=20, alphaE2=0.75, hueE2=None, colorLabelE2='red', 
                haE2='center', vaE2='bottom', fontsizeG2=9.5, colorG2='blue', sG2=40, alphaG2=0.75, hueG2='G', 
                colorLabelG2='black', haG2='center', vaG2='bottom', 
                ncol=2, leg_fontsize=10, leg_gen=None, leg_env=None, showLegend=True
            )
        if (plot_params is None):
            plot_params = plot_default_params
        else:
            plot_params = {**plot_default_params, **plot_params}

        if (bptype==0):
            #prepare the target
            #target = self.data.index.to_numpy() #df['Genotipo'].iloc[:-1].to_numpy()
            #Concat it with target variable to create a complete Dataset
            principal_E = pd.concat([self.PCA_Env , pd.DataFrame(list(self.data))] , axis = 1).rename(columns={0:'E'})
            principal_G = pd.concat([self.PCA_Gen , pd.DataFrame(self.data.index.to_numpy())] , axis = 1).rename(columns={0:'G'})
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data = principal_E , x = 'PC1',y = 'PC2' , hue=plot_params['hueE'], marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'PC1',y = 'PC2' , hue=plot_params['hueG'], alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(principal_E['PC1'].min(), principal_G['PC1'].min())) - plot_params['limspan']
            minlim_y = int(min(principal_E['PC2'].min(), principal_G['PC2'].min())) - plot_params['limspan']
            maxlim_x = int(max(principal_E['PC1'].max(), principal_G['PC1'].max())) + plot_params['limspan']
            maxlim_y = int(max(principal_E['PC2'].max(), principal_G['PC2'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            if (title==None): title='AMMI model\n'+ self.trialname + ' [' +self.trait + ']'
            ax1.set_title(title, fontsize=18)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.EV_PC1), fontsize=12)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.EV_PC2), fontsize=12)
            ax1.axis('equal')
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['PC1'].values[i]
                    ytext = principal_E['PC2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'.format(principal_E['E'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE']) #, transform=ax1.transAxes)
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['PC1'].values[i]
                    ytext = principal_G['PC2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'.format(principal_G['G'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
        
        elif (bptype==1):
            f1 = pd.concat([self.data.mean(axis=0).reset_index(), self.PCA_Env['PC1'] ], axis=1, ignore_index=True )\
            .rename(columns={0:'E', 1:'AvgTrait', 2:'PC1'})
            f2 = pd.concat([self.data.mean(axis=1).reset_index(), self.PCA_Gen['PC1'] ], axis=1, ignore_index=True )\
            .rename(columns={0:'G', 1:'AvgTrait', 2:'PC1'})
            fig, (ax1) = plt.subplots(figsize=(6,6))
            fig.subplots_adjust(left=0.01, bottom=0.01)
            g1 = sns.scatterplot(data=f1 , x='AvgTrait', y='PC1' , hue=plot_params['hueE'], 
                                 marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data=f2 , x='AvgTrait', y='PC1' , hue=plot_params['hueG'], alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(f1['AvgTrait'].min(), f2['AvgTrait'].min())) - plot_params['limspan']
            minlim_y = int(min(f1['PC1'].min(), f2['PC1'].min())) - plot_params['limspan']
            maxlim_x = int(max(f1['AvgTrait'].max(), f2['AvgTrait'].max())) + plot_params['limspan']
            maxlim_y = int(max(f1['PC1'].max(), f2['PC1'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            # Avg Yield
            ax1.axvline(self.data.mean().mean(), ls='--', c='red', linewidth=0.8, label="Average Trait [" + self.trait + "]")
            if (title==None): title='AMMI model vs trait\n'+ self.trialname + ' [' +self.trait + ']'
            ax1.set_title(title, fontsize=18)
            ax1.tick_params(labelsize=12)
            #ax1.set_xlabel("Average Trait [Yield (t/ha)]", fontsize=12)
            ax1.set_xlabel("Average Trait [" + self.trait + "]", fontsize=12)
            ax1.set_ylabel("Factor 1 - Explains {:.2f}% of the variance".format( self.EV_PC1), fontsize=12)
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(f1)):
                    xtext = f1['AvgTrait'].values[i]
                    ytext = f1['PC1'].values[i]
                    if ('E' in list(f1)):
                        f1['E'] = f1['E'].astype(str)
                        ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'.format(f1['E'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE']) #, transform=ax1.transAxes)
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(xtext,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(f2)):
                    xtext = f2['AvgTrait'].values[i]
                    ytext = f2['PC1'].values[i]
                    if ('G' in list(f2)):
                        f2['G'] = f2['G'].astype(str)
                        ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'.format(f2['G'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(xtext,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
        
        # Display both charts in one figure
        elif (bptype==2):
            # Initialise the subplot function using number of rows and columns
            fig, axis = plt.subplots(1, 2, figsize=(12,6), facecolor='white', constrained_layout=True) 
            fig.subplots_adjust(left=0.01, bottom=0.1) 
            if (title==None): title='IWIN GxE - AMMI model\n' + self.trialname + ' [' +self.trait + ']'
            fig.suptitle('{}'.format(title), fontsize=18, y=1.05)
            fonts_axes = 12
            fonts_titles = 14
            # ------------------------------
            # Chart 1 - AMMI0 model
            # ------------------------------
            ax1 = axis[0]
            title1='AMMI model - ' + self.trait
            ax1.set_title('{}'.format(title1), fontsize=fonts_titles)
            principal_E = pd.concat([self.PCA_Env , pd.DataFrame(list(self.data))] , axis = 1).rename(columns={0:'E'})
            principal_G = pd.concat([self.PCA_Gen , pd.DataFrame(self.data.index.to_numpy())] , axis = 1).rename(columns={0:'G'})
            g1 = sns.scatterplot(data = principal_E , x = 'PC1',y = 'PC2' , hue=plot_params['hueE'], marker='s', alpha=plot_params['alphaE'], 
                                 color=plot_params['colorE'], s=plot_params['sE'], label='Environment', ax=ax1)
            g2 = sns.scatterplot(data = principal_G , x = 'PC1',y = 'PC2' , hue=plot_params['hueG'], alpha=plot_params['alphaG'], 
                                 color=plot_params['colorG'], s=plot_params['sG'], label='Genotype', ax=ax1);
            minlim_x = int(min(principal_E['PC1'].min(), principal_G['PC1'].min())) - plot_params['limspan']
            minlim_y = int(min(principal_E['PC2'].min(), principal_G['PC2'].min())) - plot_params['limspan']
            maxlim_x = int(max(principal_E['PC1'].max(), principal_G['PC1'].max())) + plot_params['limspan']
            maxlim_y = int(max(principal_E['PC2'].max(), principal_G['PC2'].max())) + plot_params['limspan']
            g1.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g1.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g1.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax1.set_axisbelow(True)
            ax1.axvline(0, ls='--', c='gray', linewidth=0.8)
            ax1.axhline(0, ls='--', c='gray', linewidth=0.8)
            ax1.tick_params(labelsize=12)
            ax1.set_xlabel("Factor 1 - Explains {:.2f}% of the variance".format( self.EV_PC1), fontsize=fonts_axes)
            ax1.set_ylabel("Factor 2 - Explains {:.2f}% of the variance".format( self.EV_PC2), fontsize=fonts_axes)
            ax1.get_legend().remove()
            ax1.axis('equal')
            # Add text to the Environment points
            if (plot_params['labelE'] is True):
                for i in range(len(principal_E)):
                    xtext = principal_E['PC1'].values[i]
                    ytext = principal_E['PC2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'.format(principal_E['E'].values[i]),fontsize=plot_params['fontsizeE'], 
                             color=plot_params['colorLabelE'], ha=plot_params['haE'], va=plot_params['vaE']) #, transform=ax1.transAxes)
                    if (plot_params['arrowE'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsE'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            if (plot_params['labelG'] is True):
                for i in range(len(principal_G)):
                    xtext = principal_G['PC1'].values[i]
                    ytext = principal_G['PC2'].values[i]
                    ax1.text(xtext+plot_params['txtspan'], ytext+plot_params['txtspan'],'{}'.format(principal_G['G'].values[i]),fontsize=plot_params['fontsizeG'], 
                             color=plot_params['colorLabelG'], ha=plot_params['haG'], va=plot_params['vaG'])
                    if (plot_params['arrowG'] is True):
                        ax1.annotate("", xy=(xtext, ytext), xytext=(0,0), arrowprops=plot_params['arrowpropsG'],
                                     textcoords='data', ha='center', va='center', transform=ax1.transAxes)
            # ------------------------------
            # Chart 2 - AMMI model PC1 vs Trait
            # ------------------------------
            ax2 = axis[1]
            title2='AMMI model vs Trait [' + self.trait + ']'
            ax2.set_title('{}'.format(title2), fontsize=fonts_titles)
            f1 = pd.concat([self.data.mean(axis=0).reset_index(), self.PCA_Env['PC1'] ], axis=1, ignore_index=True )\
            .rename(columns={0:'E', 1:'AvgTrait', 2:'PC1'})
            f2 = pd.concat([self.data.mean(axis=1).reset_index(), self.PCA_Gen['PC1'] ], axis=1, ignore_index=True )\
            .rename(columns={0:'G', 1:'AvgTrait', 2:'PC1'})
            g3 = sns.scatterplot(data=f1 , x='AvgTrait', y='PC1' , hue=plot_params['hueE2'], 
                                 marker='s', alpha=plot_params['alphaE2'], 
                                 color=plot_params['colorE2'], s=plot_params['sE2'], label='Environment', ax=ax2)
            g4 = sns.scatterplot(data=f2 , x='AvgTrait', y='PC1' , hue=plot_params['hueG2'], alpha=plot_params['alphaG2'], 
                                 color=plot_params['colorG2'], s=plot_params['sG2'], label='Genotype', ax=ax2);
            minlim_x = int(min(f1['AvgTrait'].min(), f2['AvgTrait'].min())) - plot_params['limspan2']
            minlim_y = int(min(f1['PC1'].min(), f2['PC1'].min())) - plot_params['limspan2']
            maxlim_x = int(max(f1['AvgTrait'].max(), f2['AvgTrait'].max())) + plot_params['limspan2']
            maxlim_y = int(max(f1['PC1'].max(), f2['PC1'].max())) + plot_params['limspan2']
            g3.set(xlim=(minlim_x, maxlim_x), ylim=(minlim_y, maxlim_y))
            g3.grid(visible=True, which='major', color='#d3d3d3', linewidth=0.5)
            g3.grid(visible=True, which='minor', color='#d3d3d3', linewidth=0.5)
            ax2.set_axisbelow(True)
            ax2.axvline(0, ls='--', c='gray', linewidth=0.8)
            ax2.axhline(0, ls='--', c='gray', linewidth=0.8)
            ax2.axvline(self.data.mean().mean(), ls='--', c='red', linewidth=0.8, label='Average Trait [' + self.trait + ']')
            ax2.tick_params(labelsize=12)
            ax2.set_xlabel("Average Trait [" + self.trait + "]", fontsize=fonts_axes)
            ax2.set_ylabel("Factor 1 - Explains {:.2f}% of the variance".format( self.EV_PC1), fontsize=fonts_axes)
            ax2.get_legend().remove()
            # Add text to the Environment points
            if (plot_params['labelE2'] is True):
                for i in range(len(f1)):
                    xtext = f1['AvgTrait'].values[i]
                    ytext = f1['PC1'].values[i]
                    if ('E' in list(f1)):
                        ax2.text(xtext+plot_params['txtspan2'], ytext+plot_params['txtspan2'],'{}'
                                 .format(f1['E'].values[i]),fontsize=plot_params['fontsizeE2'], 
                             color=plot_params['colorLabelE2'], ha=plot_params['haE2'], va=plot_params['vaE2']) 
                    if (plot_params['arrowE2'] is True):
                        ax2.annotate("", xy=(xtext, ytext), xytext=(xtext,0), arrowprops=plot_params['arrowpropsE2'],
                                     textcoords='data', ha='center', va='center', transform=ax2.transAxes)
            if (plot_params['labelG2'] is True):
                for i in range(len(f2)):
                    xtext = f2['AvgTrait'].values[i]
                    ytext = f2['PC1'].values[i]
                    if ('G' in list(f2)):
                        ax2.text(xtext+plot_params['txtspan2'], ytext+plot_params['txtspan2'],'{}'
                                 .format(f2['G'].values[i]),fontsize=plot_params['fontsizeG2'], 
                             color=plot_params['colorLabelG2'], ha=plot_params['haG2'], va=plot_params['vaG2'])
                    if (plot_params['arrowG2'] is True):
                        ax2.annotate("", xy=(xtext, ytext), xytext=(xtext,0), arrowprops=plot_params['arrowpropsG2'],
                                     textcoords='data', ha='center', va='center', transform=ax2.transAxes)
        # Legend
        if (plot_params['showLegend'] is True):
            def getLegend_HandlesLabels(ax, handout, lablout):
                handles, labels = ax.get_legend_handles_labels()
                for h,l in zip(handles,labels):
                    if ((plot_params['leg_gen'] is not None) and (l in plot_params['leg_gen'].keys())):
                        #print(l)
                        l = l +': '+ plot_params['leg_gen'][l]
                    if ((plot_params['leg_env'] is not None) and (l in plot_params['leg_env'].keys())):
                        #print(l)
                        l = l +': '+ plot_params['leg_env'][l]
                    if l not in lablout:
                        lablout.append(l)
                        handout.append(h)
                return handout, lablout
            
            if ((plot_params['leg_gen'] is not None) and (plot_params['leg_env'] is not None)):
                handout=[]
                lablout=[]
                handout, lablout = getLegend_HandlesLabels(ax1, handout, lablout)
                if (bptype==2):
                    handout, lablout = getLegend_HandlesLabels(ax2, handout, lablout)
                    plt.legend(handout, lablout, bbox_to_anchor=(0, -0.45), loc="center", ncol=plot_params['ncol'], 
                           borderaxespad=0, fontsize=plot_params['leg_fontsize'])
                else:
                    plt.legend(handout, lablout, bbox_to_anchor=(1.05, 1), loc=2, ncol=plot_params['ncol'], 
                           borderaxespad=0, fontsize=plot_params['leg_fontsize'])
            else:
                if (bptype==2):
                    plt.legend(bbox_to_anchor=(0, -0.45), loc="center", ncol=plot_params['ncol'], 
                           borderaxespad=0, fontsize=plot_params['leg_fontsize'])
                else:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=plot_params['ncol'], 
                           borderaxespad=0, fontsize=plot_params['leg_fontsize'])
                
        # Save in PDF
        if (saveFig is True and fmt=='pdf'):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path, 'IWIN_GxE_{}_{}.pdf'
                                     .format(title.replace(' ', '_'), hoy)), bbox_inches='tight', orientation='portrait', 
                        pad_inches=0.5, dpi=300) #papertype='a4', 

        if (saveFig==True and (fmt=='jpg' or fmt=='png')):
            hoy = datetime.now().strftime('%Y%m%d')
            figures_path = '{}_{}'.format(dirname, hoy)
            if not os.path.isdir(figures_path):
                os.makedirs(figures_path)
            fig.savefig(os.path.join(figures_path,"IWIN_GxE_{}_{}.{}"
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
            
            
