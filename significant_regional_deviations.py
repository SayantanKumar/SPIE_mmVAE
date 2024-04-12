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

#-------------------------------------------------------------

def convert_p_fdr(Z_score, fs_cols):
    
    p_value = scipy.stats.norm.sf(abs(Z_score[fs_cols]))
    p_value = pd.DataFrame(p_value, columns = fs_cols)


    corr_p = []
    corr_p_allroi = {}
    for col in fs_cols:
        reject_array, p_array = fdrcorrection(p_value[col], alpha=0.05, method='indep')
        #corr_p.append(p_array.tolist())
        corr_p_allroi[col] = p_array.tolist()
        #corr_p = []

    diff_p_corr = pd.DataFrame.from_dict(corr_p_allroi)


    #import seaborn as sns
    #plt.figure(figsize = (40,6))
    #sns.heatmap(diff_p_corr_bs[fs_cols])


    ##----------------------------------------------------------------------------------

    reject_corr = diff_p_corr[fs_cols].copy()
    for idx in diff_p_corr.index.values:
        for col in fs_cols:
            if diff_p_corr[fs_cols].loc[idx, col] < 0.05:
                reject_corr.loc[idx, col] = 1
            else :
                reject_corr.loc[idx, col] = np.nan


    count_sig_roi = reject_corr.count(axis=1).to_frame().rename(columns = {0:'count'})
    #count_sig_roi_bs.hist()

    sig_count_corr = reject_corr.count().to_frame().rename(columns = {0:'num_significant'})
    
    return diff_p_corr, count_sig_roi, sig_count_corr, reject_corr

#--------------------------------------------------------------------------------


#----------------------------------------------------------------------------
#*************  Heatmap of p-values and number of significant regions *****************************
#---------------------------------------------------------------------------

cortical_cols_lh_new, cortical_cols_rh_new, subcortical_cols_ggseg, temp_mat_mm = convert_cols_ggseg(dev_mvae, cortical_cols, subcortical_cols)

cortical_cols_lh_new, cortical_cols_rh_new, subcortical_cols_ggseg, temp_mat_um = convert_cols_ggseg(dev_bvae, cortical_cols, subcortical_cols)

p_values_corr_1, count_sig_roi_1, sig_count_corr_1, reject_corr_1 = convert_p_fdr(temp_mat_mm.loc[temp_mat_mm.DX_bl != 'CN'].reset_index(drop = True), fs_cols)

p_values_corr_2, count_sig_roi_2, sig_count_corr_2, reject_corr_2 = convert_p_fdr(temp_mat_um.loc[temp_mat_um.DX_bl != 'CN'].reset_index(drop = True), fs_cols)

count_sig_roi_1 = reject_corr_1.count(axis=1).to_frame().rename(columns = {0:'count'})
count_sig_roi_2 = reject_corr_2.count(axis=1).to_frame().rename(columns = {0:'count'})


count_sig_roi_1['model'] = 'Proposed \n (Multimodal)'
count_sig_roi_2['model'] = 'Baseline \n (Unimodal)'


#count_sig_all = pd.concat([count_sig_roi, count_sig_roi_bs, count_sig_roi_gp]).reset_index(drop = True)
count_sig_all = pd.concat([count_sig_roi_1, count_sig_roi_2]).reset_index(drop = True)


plt.figure(figsize = (5,3))
sns.boxplot(x = 'model', y = 'count', data = count_sig_all, showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
plt.ylabel('Number of brain regions',fontsize = 18)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 18)
plt.ylim(-10,120)
plt.title('Number of brain regions for each patient \n with statistically significant deviations', fontsize = 20)
plt.tight_layout()


#----------------------------------------------------------------------------
#*************  Frequency of significance (Number of times each ROI is significant) *****************************
#---------------------------------------------------------------------------

#### Cortical and subcortical ROI ####

#cortical_cols_lh_new, cortical_cols_rh_new, subcortical_cols_ggseg, temp_mat_um = convert_cols_ggseg(dev_bvae_c3_n2, cortical_cols, subcortical_cols)

cols_new = list(cortical_cols_lh_new) + list(cortical_cols_rh_new) + list(subcortical_cols_ggseg) + list(hcm_cols)

p_values_corr_1, count_sig_roi_1, sig_count_corr_1, reject_corr_1 = convert_p_fdr(temp_mat_mm.loc[temp_mat_mm.DX_bl != 'CN'].reset_index(drop = True), cols_new)
sig_count_corr_1 = (sig_count_corr_1/len(reject_corr_1))*100

p_values_corr_2, count_sig_roi_2, sig_count_corr_2, reject_corr_2 = convert_p_fdr(temp_mat_um.loc[temp_mat_um.DX_bl != 'CN'].reset_index(drop = True), cols_new)
sig_count_corr_2 = (sig_count_corr_2/len(reject_corr_2))*100


ggseg_python.plot_dk(sig_count_corr_1.loc[cortical_cols_lh_new + cortical_cols_rh_new].iloc[:,0].to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0,100],
                ylabel='Frequency (%) of Significance', figsize = (8,5), fontsize = 24, title='Multimodal')

ggseg_python.plot_aseg(sig_count_corr_1.loc[subcortical_cols_ggseg].iloc[:,0].to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray', vminmax = [0,100],
                ylabel='Frequency (%) of Significance', figsize = (8,5), fontsize = 24, title='Multimodal')

#------------------------------
ggseg_python.plot_dk(sig_count_corr_2.loc[cortical_cols_lh_new + cortical_cols_rh_new].iloc[:,0].to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray',vminmax = [0,100],
                ylabel='Frequency (%) of Significance', figsize = (8,5), fontsize = 24, title='Unimodal')


ggseg_python.plot_aseg(sig_count_corr_2.loc[subcortical_cols_ggseg].iloc[:,0].to_dict(), cmap='hot',
                background='w', edgecolor='k',  bordercolor='gray',vminmax = [0,100],
                ylabel='Frequency (%) of Significance', figsize = (8,5), fontsize = 24, title='Unimodal')


### Hippocampal ROI ######

width = 0.4
plt.figure(figsize = (32,14))



x_axis = np.arange(len(hcm_cols))

plt.bar(x_axis - 0.2, sig_count_corr_1.loc[hcm_cols].num_significant.values,  width, label='Proposed (Multimodal)')
plt.bar(x_axis + 0.2, sig_count_corr_2.loc[hcm_cols].num_significant.values,  width, label='Baseline (Unimodal)')


plt.xticks(x_axis, hcm_cols, rotation = 45, fontsize = 28)
plt.xlabel('Hippocampal Brain regions', fontsize = 34)
#plt.xticks(FontSize = 20)
plt.yticks(fontsize = 28)
plt.ylabel('Frequency of significance (%)', fontsize = 30)
plt.title('Frequency of statistically significant deviations \n of hippocampal brain regions', fontsize = 36)
plt.legend(loc='best', fontsize = 34)
plt.tight_layout()


######################################################################
#---------- Bar plots of cortical and subcortical frequency of significance ---------
######################################################################

sig_count_corr_1 = reject_corr_1.count().to_frame().rename(columns = {0:'num_significant'})
sig_count_corr_1 = (sig_count_corr_1/len(reject_corr_1))*100

sig_count_corr_2 = reject_corr_2.count().to_frame().rename(columns = {0:'num_significant'})
sig_count_corr_2 = (sig_count_corr_2/len(reject_corr_2))*100


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

width = 0.4
plt.figure(figsize = (30,12))

#subcortical_cols_adni = fs_subcort_adni.columns.values
x_axis = np.arange(len(subcortical_cols_ggseg))

plt.bar(x_axis - 0.2, sig_count_corr_1.loc[subcortical_cols_ggseg].num_significant.values,  width, label='Proposed (Multimodal)')
plt.bar(x_axis + 0.2, sig_count_corr_2.loc[subcortical_cols_ggseg].num_significant.values,  width, label='Baseline (Unimodal)')


plt.xticks(x_axis, subcortical_cols_ggseg, rotation = 90, fontsize = 24)
plt.xlabel('Subcortical Brain regions', fontsize = 28)
#plt.xticks(FontSize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Frequency of significance (%)', fontsize = 26)
plt.legend(loc='best', fontsize = 26)
plt.title('Frequency of statistically significant deviations of subcortical brain regions', fontsize = 30)
plt.tight_layout()
#plt.savefig('/Users/sayantankumar/Desktop/Aris_Work/Submissions/SPIE_2022/Plots/sig_roi_subcortical.pdf', bbox_inches = 'tight', dpi = 600)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


width = 0.4
plt.figure(figsize = (30,12))

#cortical_cols_adni = fs_cort_adni.columns.values

x_axis = np.arange(len(cortical_cols_lh_new + cortical_cols_rh_new))

plt.bar(x_axis - 0.2, sig_count_corr_1.loc[cortical_cols_lh_new + cortical_cols_rh_new].num_significant.values,  width, label='Proposed (Multimodal)')
plt.bar(x_axis + 0.2, sig_count_corr_2.loc[cortical_cols_lh_new + cortical_cols_rh_new].num_significant.values,  width, label='Baseline (Unimodal)')


plt.xticks(x_axis, cortical_cols_lh_new + cortical_cols_rh_new, rotation = 90, fontsize = 24)
plt.xlabel('Cortical Brain regions', fontsize = 28)
#plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylabel('Frequency of significance (%)', fontsize = 26)
plt.title('Frequency of statistically significant deviations of cortical brain regions', fontsize = 30)
plt.legend(loc='best', fontsize = 26)
plt.tight_layout()