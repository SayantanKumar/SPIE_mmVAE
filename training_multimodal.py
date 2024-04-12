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

def add_covariates(roi_demo_both, non_roi_cols, multiply = True):

    roi_demo_both.loc[roi_demo_both.Sex == 0, 'Sex'] = 'Female'
    roi_demo_both.loc[roi_demo_both.Sex == 1, 'Sex'] = 'Male'

    age_df = pd.DataFrame(one_hot_encoding(roi_demo_both, 'Age'), index = roi_demo_both.index.values)
    age_mat = one_hot_encoding(roi_demo_both, 'Age')

    sex_df = pd.DataFrame(one_hot_encoding(roi_demo_both, 'Sex'), index = roi_demo_both.index.values)
    sex_mat = one_hot_encoding(roi_demo_both, 'Sex')

    site_df = pd.DataFrame(one_hot_encoding(roi_demo_both, 'site'), index = roi_demo_both.index.values)
    site_mat = one_hot_encoding(roi_demo_both, 'site')

    if multiply == True:
        combine_arr = []
        age_sex = []
        for i in range(roi_demo_both.shape[0]):
            age_sex_val = np.matmul(sex_mat[i].reshape(-1,1), age_mat[i].reshape(-1,1).T).flatten().reshape(-1,1)
            age_sex.append(age_sex_val)

            age_sex_site_val = np.matmul(age_sex[i], site_mat[i].reshape(-1,1).T).flatten().reshape(-1,1)
            combine_arr.append(age_sex_site_val)

        combine_mat = np.hstack(np.array(combine_arr)).T

        age_sex_df = pd.DataFrame(np.hstack(np.array(age_sex)).T, index = roi_demo_both.index.values)
        age_sex_df[non_roi_cols] = roi_demo_both[non_roi_cols]
    
    else:
        age_sex_df = pd.DataFrame(np.concatenate((age_mat, sex_mat), axis=1), index = roi_demo_both.index.values)
        age_sex_df[non_roi_cols] = roi_demo_both[non_roi_cols]
        
    return age_sex_df


roi_demo_both = pd.read_csv('./saved_dataframes/roi_demo_both.csv')
fs_features_adni = pd.read_csv('./saved_dataframes/fs_features_adni.csv')
fs_cols = [col for col in roi_demo_both.columns if col not in ['Age', 'Sex', 'site', 'DX_bl', 'RID']]

with open("./saved_dataframes/cortical_cols", "rb") as fp: 
    cortical_cols = pickle.load(fp)
    
with open("./saved_dataframes/subcortical_cols", "rb") as fp: 
    subcortical_cols = pickle.load(fp)
    
with open("./saved_dataframes/hcm_cols", "rb") as fp: 
    hcm_cols = pickle.load(fp)

non_roi_cols = ['Age', 'Sex', 'site', 'DX_bl', 'RID'] 

age_sex_df = add_covariates(roi_demo_both, non_roi_cols, multiply = False)

only_CN = roi_demo_both.loc[roi_demo_both.DX_bl == 'CN']

#only_CN = only_CN.sample(n=1000, random_state=1) #----------------sampling 1000 rows randomly for checking code quality -----------
#only_CN = only_CN.loc[only_CN.site == 'UKB']

print('Number of healthy controls selected: {}'.format(len(only_CN)))

rest = roi_demo_both.loc[roi_demo_both['DX_bl'] != 'CN']

y_CN = only_CN['DX_bl']
y_rest = rest['DX_bl']

only_CN_ADNI = only_CN.loc[only_CN.site == 'ADNI']
only_CN_ADNI_test = only_CN_ADNI.sample(n=round(0.15*len(only_CN_ADNI)), random_state=1)

CN_ADNI_train_val = only_CN_ADNI.loc[~only_CN_ADNI.RID.isin(only_CN_ADNI_test.RID.values)]

ADNI_CN_model, ADNI_CN_held_val = train_test_split(CN_ADNI_train_val, test_size=0.35, shuffle = False, random_state = 1000)

only_CN_UKB = only_CN.loc[only_CN.site == 'UKB']

# only_CN_UKB_test = only_CN_UKB.sample(n=round(0.01*len(only_CN_UKB)), random_state=1)
#CN_train_val, CN_test, y_CN_train_val, y_CN_test = train_test_split(only_CN, y_CN, test_size=0.05, shuffle = False, random_state = 1000)


X_test_org = pd.concat([rest, only_CN_ADNI_test]).copy()
y_test_org = pd.concat([y_rest, only_CN_ADNI_test['DX_bl']]).copy()


temp1 = fs_features_adni.loc[fs_features_adni.RID.isin(X_test_org.RID.unique())].reset_index(drop = True)
temp2 = X_test_org.sort_values(by = 'RID').reset_index(drop = True)
other_cols = ['MMSE', 'ADAS13', 'Intracranial_vol', 'ABETA', 'PTAU', 'mPACCdigit','mPACCtrailsB', 'RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','RAVLT_perc_forgetting']
X_test_org = pd.concat([temp2, temp1[other_cols]], axis = 1)

#external_X_test, internal_X_test, external_y_test, internal_y_test = train_test_split(X_test_org, X_test_org['DX_bl'], stratify= X_test_org['DX_bl'], test_size=0.25, random_state = 1000)


X_train_org = only_CN.loc[~only_CN.RID.isin(only_CN_ADNI_test.RID.values)].copy()
y_train_org = only_CN['DX_bl'].copy()


#kf = StratifiedKFold(n_splits= 5, shuffle = False)
X_train_total = X_train_org.reset_index(drop = True)

X_train_total_UKB = X_train_total.loc[X_train_total.site == 'UKB'].reset_index(drop = True)
X_train_total_ADNI = X_train_total.loc[X_train_total.site == 'ADNI'].reset_index(drop = True)


X_train_total = pd.concat([X_train_total_UKB, ADNI_CN_model]).reset_index(drop = True)

print('Number of training (train + val) samples: {}'.format(len(X_train_total)))
print('Number of training samples from UKB: {}'.format(len(X_train_total_UKB)))
#print('Number of training samples from ADNI: {}'.format(len(X_train_total_ADNI)))

print('Number of ADNI CN used for model training/val: {}'.format(len(ADNI_CN_model)))
print('Number of ADNI CN used for parameter estimation: {}'.format(len(ADNI_CN_held_val)))
print('Number of ADNI CN used in test set: {}'.format(len(only_CN_ADNI_test)))

#********************************************************
#********************************************************

batch_size = 256
lr = 5e-5
latent_dim = 64
alpha_1 = 1
alpha_2 = 1
alpha_3 = 1
beta = 1
epochs = 200

k = 1

non_roi_cols = ['Age', 'Sex', 'site', 'DX_bl', 'RID']

non_roi_cols_test = ['Age', 'Sex', 'site', 'DX_bl', 'RID', 'MMSE', 'ADAS13',
       'Intracranial_vol', 'ABETA', 'PTAU', 'mPACCdigit', 'mPACCtrailsB',
       'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
       'RAVLT_perc_forgetting']

                            
X_train_m1_ukb, X_train_m2_ukb, X_val_m1_ukb, X_val_m2_ukb, train_age_group_ukb, val_age_group_ukb, m1_cols, m2_cols, scale_allfold_m1_ukb, scale_allfold_m2_ukb = create_training_data_MVAE_2M(X_train_total_UKB, cortical_cols, subcortical_cols, hcm_cols, non_roi_cols, age_sex_df)
print('Training on UKB')

model, train_loss, val_loss, X_org_val, X_pred_val = train_model_MVAE_2M(X_train_m1_ukb, X_train_m2_ukb, X_val_m1_ukb, X_val_m2_ukb, train_age_group_ukb, val_age_group_ukb, m1_cols, m2_cols, retrain = False)

torch.save(model, './saved_models/trained_MVAE_UKB')

print('Fine-tuning model on ADNI CN')
batch_size = 64

X_train_m1_adni, X_train_m2_adni, X_val_m1_adni, X_val_m2_adni, train_age_group_adni, val_age_group_adni, m1_cols, m2_cols, scale_allfold_m1_adni, scale_allfold_m2_adni = create_training_data_MVAE_2M(X_train_total_ADNI, cortical_cols, subcortical_cols, hcm_cols, age_sex_df)

model_retr, train_loss_retr, val_loss_retr, X_org_val_retr, X_pred_val_retr = train_model_MVAE_2M(X_train_m1_adni, X_train_m2_adni, X_val_m1_adni, X_val_m2_adni, train_age_group_adni, val_age_group_adni, m1_cols, m2_cols, retrain = True)

test_loader, X_test_org, test_age_group, X_test_m1, X_test_m2 = create_test_data_MVAE_2M(X_test_org, non_roi_cols_test, cortical_cols, subcortical_cols, hcm_cols, age_sex_df, scale_allfold_m1_adni, scale_allfold_m2_adni)
    
recon_m1_test, recon_m2_test = test_mvae_2M(test_loader, model_retr)

X_pred_test = concat_pred_modality_2M(X_test_m1, X_test_m2, X_test_org, recon_m1_test, recon_m2_test)

X_org_ho_val =  ADNI_CN_held_val.copy()

val_ho_loader, X_valho_org, val_ho_age_group, X_valho_m1, X_valho_m2 = create_valho_data_MVAE_2M(X_org_ho_val, cortical_cols, subcortical_cols, hcm_cols, age_sex_df, scale_allfold_m1_adni, scale_allfold_m2_adni)
   
recon_m1_valho, recon_m2_valho = test_mvae_2M(val_ho_loader, model_retr)

X_pred_valho = concat_pred_modality_valho_2M(X_valho_m1, X_valho_m2, X_valho_org, recon_m1_valho, recon_m2_valho)
  
    
diff, dev_mvae, mean_df, std_df = calculate_deviations(X_valho_org, X_pred_valho, fs_cols, X_test_org, X_pred_test)
   
#********************************************************
#********************************************************

recon = X_test_org.copy()
recon[fs_cols] = (X_test_org[fs_cols] - X_pred_test[fs_cols])**2
recon['Mean Reconstruction Loss'] = (abs((recon[fs_cols])).sum(axis = 1).to_numpy()/recon[fs_cols].shape[1])
sns.boxplot(x = 'DX_bl', y = 'Mean Reconstruction Loss', data = recon, order = ['SMC', 'EMCI', 'LMCI', 'AD'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
plt.xlabel('Disease category', fontsize = 18)
plt.ylabel('Mean reconstruction loss \n across all brain regions', fontsize = 18)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.title('Mean reconstruction loss \n (Case 3)', fontsize = 18)  


# epochs = range(len(train_loss_c1))
# plt.plot(epochs, train_loss_c1, label='Training loss')
# plt.plot(epochs, val_loss_c1, label='Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()


plt.figure(figsize = (7,5))
epochs = range(len(train_loss))
plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.plot(epochs, train_loss_retr, label='Training loss (fine-tuning)')
plt.plot(epochs, val_loss_retr, label='Validation loss (fine-tuning)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation loss')
plt.legend()

dev_mvae.to_csv('./saved_dataframes/multimodal_deviations.csv')
