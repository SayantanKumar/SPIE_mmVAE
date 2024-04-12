import pickle
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import itertools
import warnings
warnings.filterwarnings('ignore')

import scipy
import statsmodels
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import fdrcorrection_twostage
import seaborn as sns
import ggseg_python

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from utils import *
from mvae_PoE import *
from dataloaders import *


def plot_mean_deviation(Z_score_all, fs_cols):
    
    Z_score_all['Mean deviation'] = (abs((Z_score_all[fs_cols])).sum(axis = 1).to_numpy()/Z_score_all[fs_cols].shape[1])

    #plt.figure(figsize = (10,7))

    sns.boxplot(x = 'DX_bl', y = 'Mean deviation', data = Z_score_all, order = ['CN', 'SMC', 'EMCI', 'LMCI', 'AD'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('Disease category', fontsize = 18)
    plt.ylabel('Mean deviation across \n all brain regions', fontsize = 18)
    plt.yticks(fontsize = 14)
    plt.xticks(fontsize = 14)
    #plt.legend(fontsize = 18)
    #plt.title('Mean deviation (NormVAE)', FontSize = 18)                                                                         
    
 

#******************************************
#-----------------------------------------

dev_mvae = pd.read_csv('./saved_dataframes/multimodal_deviations.csv')
dev_bvae = pd.read_csv('./saved_dataframes/unimodal_deviations.csv')

with open("./saved_dataframes/cortical_cols", "rb") as fp: 
    cortical_cols = pickle.load(fp)
    
with open("./saved_dataframes/subcortical_cols", "rb") as fp: 
    subcortical_cols = pickle.load(fp)
    
with open("./saved_dataframes/hcm_cols", "rb") as fp: 
    hcm_cols = pickle.load(fp)

non_roi_cols = ['Age', 'Sex', 'site', 'DX_bl', 'RID'] 

fs_cols = list(cortical_cols) + list(subcortical_cols) + list(hcm_cols)

plt.figure(figsize = (7,5))
plt.title('Mean deviations (multimodal)', fontsize = 18)  
plot_mean_deviation(dev_mvae, fs_cols)

plt.figure(figsize = (7,5))
plt.title('Mean deviations (unimodal)', fontsize = 18)  
plot_mean_deviation(dev_bvae, fs_cols)

#----------------------------------------------------------------------------
#*************  Deviation brain maps (multimodal) *****************************
#---------------------------------------------------------------------------

cortical_cols_lh_new, cortical_cols_rh_new, subcortical_cols_ggseg, temp_mat_mm = convert_cols_ggseg(dev_mvae, cortical_cols, subcortical_cols)

temp_mat_mm_CN = temp_mat_mm.loc[temp_mat_mm.DX_bl == 'CN']
temp_mat_mm_SMC = temp_mat_mm.loc[temp_mat_mm.DX_bl == 'SMC']
temp_mat_mm_EMCI = temp_mat_mm.loc[temp_mat_mm.DX_bl == 'EMCI']
temp_mat_mm_LMCI = temp_mat_mm.loc[temp_mat_mm.DX_bl == 'LMCI']
temp_mat_mm_AD = temp_mat_mm.loc[temp_mat_mm.DX_bl == 'AD']

##--------------------------- Cortical -----------------------

print('Number of cortical ROIs shown = {}'.format(len(cortical_cols_lh_new + cortical_cols_rh_new)))


ggseg_python.plot_dk(abs(temp_mat_mm_SMC[cortical_cols_lh_new + cortical_cols_rh_new]).mean().to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 3.5],
                ylabel='Mean cortical deviation', figsize = (8,5), fontsize = 24, title='Multimodal cortical deviation maps for SMC (N = {})'.format(len(temp_mat_mm_SMC)))
#print('temp_mat_mm_SMC_cort : Max = {}, Min = {}'.format(abs(temp_mat_mm_SMC[cortical_cols_lh_new + cortical_cols_rh_new]).mean().max(), abs(temp_mat_mm_SMC[cortical_cols_lh_new + cortical_cols_rh_new]).mean().min()))

    
ggseg_python.plot_dk(abs(temp_mat_mm_EMCI[cortical_cols_lh_new + cortical_cols_rh_new]).mean().to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 3.5],
                ylabel='Mean cortical deviation', figsize = (8,5), fontsize = 24, title='Multimodal cortical deviation maps for EMCI (N = {})'.format(len(temp_mat_mm_EMCI)))
#print('temp_mat_mm_EMCI_cort : Max = {}, Min = {}'.format(abs(temp_mat_mm_EMCI[cortical_cols_lh_new + cortical_cols_rh_new]).mean().max(), abs(temp_mat_mm_EMCI[cortical_cols_lh_new + cortical_cols_rh_new]).mean().min()))



ggseg_python.plot_dk(abs(temp_mat_mm_LMCI[cortical_cols_lh_new + cortical_cols_rh_new]).mean().to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 3.5],
                ylabel='Mean cortical deviation', figsize = (8,5), fontsize = 24, title='Multimodal cortical deviation maps for LMCI (N = {})'.format(len(temp_mat_mm_LMCI)))
#print('temp_mat_mm_LMCI_cort : Max = {}, Min = {}'.format(abs(temp_mat_mm_LMCI[cortical_cols_lh_new + cortical_cols_rh_new]).mean().max(), abs(temp_mat_mm_LMCI[cortical_cols_lh_new + cortical_cols_rh_new]).mean().min()))



ggseg_python.plot_dk(abs(temp_mat_mm_AD[cortical_cols_lh_new + cortical_cols_rh_new]).mean().to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 3.5],
                ylabel='Mean cortical deviation', figsize = (8,5), fontsize = 24, title='Multimodal cortical deviation maps for AD (N = {})'.format(len(temp_mat_mm_AD)))
#print('temp_mat_mm_AD_cort : Max = {}, Min = {}'.format(abs(temp_mat_mm_AD[cortical_cols_lh_new + cortical_cols_rh_new]).mean().max(), abs(temp_mat_mm_AD[cortical_cols_lh_new + cortical_cols_rh_new]).mean().min()))


##--------------------------- Subcortical -----------------------

print('Number of subcortical ROIs shown = {}'.format(len(subcortical_cols_ggseg)))

ggseg_python.plot_aseg(abs(temp_mat_mm_SMC[subcortical_cols_ggseg]).mean().to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 3.5],
                ylabel='Mean cortical deviation', figsize = (8,5), fontsize = 24, title='Multimodal subcortical deviation maps for SMC (N = {})'.format(len(temp_mat_mm_SMC)))
#print('temp_mat_mm_SMC_subcort : Max = {}, Min = {}'.format(abs(temp_mat_mm_SMC[subcortical_cols_ggseg]).mean().max(), abs(temp_mat_mm_SMC[subcortical_cols_ggseg]).mean().min()))


ggseg_python.plot_aseg(abs(temp_mat_mm_EMCI[subcortical_cols_ggseg]).mean().to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 3.5],
                ylabel='Mean cortical deviation', figsize = (8,5), fontsize = 24, title='Multimodal subcortical deviation maps for EMCI (N = {})'.format(len(temp_mat_mm_EMCI)))
#print('temp_mat_mm_EMCI_subcort : Max = {}, Min = {}'.format(abs(temp_mat_mm_EMCI[subcortical_cols_ggseg]).mean().max(), abs(temp_mat_mm_EMCI[subcortical_cols_ggseg]).mean().min()))


ggseg_python.plot_aseg(abs(temp_mat_mm_LMCI[subcortical_cols_ggseg]).mean().to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 3.5],
                ylabel='Mean cortical deviation', figsize = (8,5), fontsize = 24, title='Multimodal subcortical deviation maps for LMCI (N = {})'.format(len(temp_mat_mm_LMCI)))
#print('temp_mat_mm_LMCI_subcort : Max = {}, Min = {}'.format(abs(temp_mat_mm_LMCI[subcortical_cols_ggseg]).mean().max(), abs(temp_mat_mm_LMCI[subcortical_cols_ggseg]).mean().min()))


ggseg_python.plot_aseg(abs(temp_mat_mm_AD[subcortical_cols_ggseg]).mean().to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 3.5],
                ylabel='Mean cortical deviation', figsize = (8,5), fontsize = 24, title='Multimodal subcortical deviation maps for AD (N = {})'.format(len(temp_mat_mm_AD)))
#print('temp_mat_mm_AD_subcort : Max = {}, Min = {}'.format(abs(temp_mat_mm_AD[subcortical_cols_ggseg]).mean().max(), abs(temp_mat_mm_AD[subcortical_cols_ggseg]).mean().min()))


#----------------------------------------------------------------------------
#*************  Mean deviation comparison (unimodal vs multimodal) *****************************
#---------------------------------------------------------------------------

def calculate_slope_2model(cat, dev_bvae_m1, dev_bvae_m2):
    
    from scipy.optimize import curve_fit

    #cat = ['SMC', 'EMCI', 'LMCI', 'AD']
    all_mean_m1 = []
    all_mean_m2 = []

    for i in cat:
        all_mean_m1.append(dev_bvae_m1['Mean deviation'][dev_bvae_m1.DX_bl == i].mean())
        all_mean_m2.append(dev_bvae_m2['Mean deviation'][dev_bvae_m2.DX_bl == i].mean())

    all_mean_m1 = dict(zip(cat, all_mean_m1))
    all_mean_m2 = dict(zip(cat, all_mean_m2))

    from scipy.stats import linregress

    slope_m1, intercept_m1, r_value_m1, p_value_m1, std_err_m1 = linregress([1, 2, 3, 4], list(all_mean_m1.values()))
    slope_m2, intercept_m2, r_value_m2, p_value_m2, std_err_m2 = linregress([1, 2, 3, 4], list(all_mean_m2.values()))
    
    slope_m1 = round(slope_m1,2)
    slope_m2 = round(slope_m2,2)
    
    print('Model 1 slope (without test CN) = {}, intercept = {}\n'.format(slope_m1, intercept_m1))
    print('Case 2 slope (without test CN) = {}, intercept = {}\n'.format(slope_m2, intercept_m2))
    
    return slope_m1, slope_m2
    
    
cat = ['SMC', 'EMCI', 'LMCI', 'AD']
slope_n1, slope_n2 = calculate_slope_2model(cat, dev_mvae, dev_bvae)

dev_mvae['label'] = 'Multimodal' + ' (slope = ' + str(slope_n1) + ')'
dev_bvae['label'] = 'Unimodal' + ' (slope = ' + str(slope_n2) + ')'


dev_all = pd.concat([dev_mvae[['Mean deviation', 'label', 'DX_bl']], dev_bvae[['Mean deviation', 'label', 'DX_bl']]]).reset_index(drop = True)

plt.figure(figsize = (7,5))
sns.boxplot(x = 'DX_bl', y = 'Mean deviation', hue = 'label', data = dev_all, order = ['SMC', 'EMCI', 'LMCI', 'AD'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
plt.xlabel('Disease category', fontsize = 22)
plt.ylabel('Mean deviation (Z-scores) \n across all brain regions', fontsize = 22)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.legend(fontsize = 24)
plt.title('Sensitivity of deviations towards disease staging', fontsize = 26)


#----------------------------------------------------------------------------
#*************  Mean deviation plots with statistical annotations *****************************
#---------------------------------------------------------------------------

from statannot import add_stat_annotation

'''
Statistical test to run. Must be one of:
        - `Levene`
        - `Mann-Whitney`
        - `Mann-Whitney-gt`
        - `Mann-Whitney-ls`
        - `t-test_ind`
        - `t-test_welch`
        - `t-test_paired`
        - `Wilcoxon`
        - `Kruskal`
'''
#plt.figure(figsize = (10,6))
fig, ax = plt.subplots(1, 2, figsize = (24,8), sharey = False)


ax[0] = sns.boxplot(ax = ax[0], x = 'DX_bl', y = 'Mean deviation', data = dev_mvae, order = ['SMC', 'EMCI', 'LMCI', 'AD'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
ax[0].set_ylim([0, 5])
add_stat_annotation(ax[0], data=dev_mvae, x='DX_bl', y='Mean deviation', order=['SMC', 'EMCI', 'LMCI', 'AD'],
                    box_pairs=[("SMC", "EMCI"), ("SMC", "LMCI"), ("SMC", "AD"), ("EMCI", "LMCI"), ("EMCI", "AD"), ("LMCI", "AD")],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=0)

ax[0].set_xlabel('Disease category', fontsize = 22)
ax[0].set_ylabel('Mean deviation (Z-scores) \n across all brain regions', fontsize = 22)
ax[0].set_title('Pairwise comparison between mean deviations \n for different disease stages (multimodal)', fontsize = 26)
ax[0].tick_params(axis='x', labelsize= 20)
ax[0].tick_params(axis='y', labelsize= 20)

###################################
ax[1] = sns.boxplot(ax = ax[1], x = 'DX_bl', y = 'Mean deviation', data = dev_bvae, order = ['SMC', 'EMCI', 'LMCI', 'AD'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
ax[1].set_ylim([0, 5])
add_stat_annotation(ax[1], data=dev_bvae, x='DX_bl', y='Mean deviation', order=['SMC', 'EMCI', 'LMCI', 'AD'],
                    box_pairs=[("SMC", "EMCI"), ("SMC", "LMCI"), ("SMC", "AD"), ("EMCI", "LMCI"), ("EMCI", "AD"), ("LMCI", "AD")],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=1)

ax[1].set_xlabel('Disease category', fontsize = 18)
ax[1].set_ylabel('Mean deviation (Z-scores)', fontsize = 18)
ax[1].set_title('Baseline (Unimodal)', fontsize = 18)
ax[1].tick_params(axis='x', labelsize= 16)


plt.tight_layout()
plt.subplots_adjust(top = 0.8, hspace= 0.2, wspace = 0.2)


#----------------------------------------------------------------------------
#*************  Calculate ANOVA p-values *****************************
#---------------------------------------------------------------------------

def calculate_ANOVA_pvalues(dev_mvae):
    
    from scipy.stats import f_oneway

    SMC = dev_mvae.loc[dev_mvae.DX_bl == 'SMC']['Mean deviation'].values
    EMCI = dev_mvae.loc[dev_mvae.DX_bl == 'EMCI']['Mean deviation'].values
    LMCI = dev_mvae.loc[dev_mvae.DX_bl == 'LMCI']['Mean deviation'].values
    AD = dev_mvae.loc[dev_mvae.DX_bl == 'AD']['Mean deviation'].values

    print('ANOVA performance')
    print('ANOVA across all categories (SMC, EMCI, LMCI, AD)')
    print(f_oneway(SMC, EMCI, LMCI, AD))

    print('\nANOVA across SMC and EMCI')
    print(f_oneway(SMC, EMCI))

    print('\nANOVA across EMCI and LMCI')
    print(f_oneway(EMCI, LMCI))

    print('\nANOVA across LMCI and AD')
    print(f_oneway(LMCI, AD))
    
calculate_ANOVA_pvalues(dev_mvae)
calculate_ANOVA_pvalues(dev_bvae)