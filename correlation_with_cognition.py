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

from scipy.stats import pearsonr
from scipy.stats import spearmanr

#cog_cols = ['MMSE', 'ADAS13', 'mPACCdigit', 'mPACCtrailsB', 'RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','RAVLT_perc_forgetting']

cog_cols = ['ADAS13', 'RAVLT_immediate']

pearson_all_n1 = []
pearson_all_n2 = []

for cog in cog_cols:

    corr_p_n1, _ = pearsonr(dev_mvae[cog], dev_mvae['Mean deviation'])
    corr_p_n2, _ = pearsonr(dev_bvae[cog], dev_bvae['Mean deviation'])

    pearson_all_n1.append(corr_p_n1)
    pearson_all_n2.append(corr_p_n2)

    R2_pearson_n1 = [i**2 for i in pearson_all_n1]
    R2_pearson_n2 = [i**2 for i in pearson_all_n2]


fig, axs = plt.subplots(len(cog_cols), 2, figsize = (10,6))

j = 0
for cog in cog_cols:
    
    m, b = np.polyfit(dev_mvae[cog], dev_mvae['Mean deviation'], 1)
    axs[j,0].plot(dev_mvae[cog], m*dev_mvae[cog] + b, color = 'red')
    
    axs[j,0].scatter(dev_mvae[cog], dev_mvae['Mean deviation'])
    axs[j,0].set_xlabel(cog, fontsize = 16)
    axs[j,0].set_ylabel('Mean \n deviation', fontsize = 16)
    axs[j,0].set_title('Proposed (Multimodal) \n $r$ = {}'.format(round(pearson_all_n1[j], 3)), fontsize = 18)
    
    m, b = np.polyfit(dev_bvae[cog], dev_bvae['Mean deviation'], 1)
    axs[j,1].plot(dev_bvae[cog], m*dev_bvae[cog] + b, color = 'red')

    axs[j,1].scatter(dev_bvae[cog], dev_bvae['Mean deviation'])
    axs[j,1].set_xlabel(cog, fontsize = 16)
    axs[j,1].set_ylabel('Mean \n deviation', fontsize = 16)
    axs[j,1].set_title('Baseline (Unimodal) \n $r$ = {}'.format(round(pearson_all_n2[j], 3)), fontsize = 18)
    
    if cog == 'RAVLT_immediate':
        axs[j,0].set_xlabel('RAVLT', fontsize = 16)
        axs[j,1].set_xlabel('RAVLT', fontsize = 16)

    #fig.suptitle('Pearson Correlation Coeffient between mean deviation and cognitive scores', fontsize = 24)
    
    j = j + 1
    

plt.tight_layout()
plt.subplots_adjust(top = 0.8, hspace=1, wspace = 0.3)
plt.suptitle('Pearson Correlation Coeffient between mean deviation and cognitive scores', fontsize = 20)
plt.show()