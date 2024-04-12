import pickle
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import itertools
import pickle
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

#-------------------------------------------------------------------
#---------------- Extracting healthy controls and disease subjects from UKBiobank 
#-------------------------------------------------------------------

def ukb_control_inclusion(icd_option, mental_health = True):
    
    if icd_option == 'pinaya':
        hc_ukb = screen_icd_ukb(fs_features_ukb)
        
    if icd_option == 'no_icd':
        hc_ukb = fs_features_ukb.loc[(pd.isnull(fs_features_ukb['Diagnosis main ICD10']) == True) & (pd.isnull(fs_features_ukb['Diagnosis secondary ICD10']))]
       
    cog_mh_df = get_cog_mh_ukb(ukb_path, ukb_dict, hc_ukb)
    
    if mental_health == True:
        pds_hc = cog_mh_df.loc[(cog_mh_df['Seen doctor for nerves anxiety tension and depression'] == 0) & (cog_mh_df['Seen psychiatrist for nerves anxiety tension and depression'] == 0)]
        selected_hc = pds_hc.loc[pds_hc.total_rds <= 5]

        hc_ukb = fs_features_ukb.loc[fs_features_ukb.eid.isin(selected_hc.eid.unique())]
    
#     mental_disorder_list = fs_features_ukb.loc[(fs_features_ukb['Diagnosis main ICD10'].str.startswith('F')) | (fs_features_ukb['Diagnosis secondary ICD10'].str.startswith('F'))].eid.unique()
#     nevous_system_list = fs_features_ukb.loc[(fs_features_ukb['Diagnosis main ICD10'].str.startswith('G')) | (fs_features_ukb['Diagnosis secondary ICD10'].str.startswith('G'))].eid.unique()
#     exclude_eid_list = list(itertools.chain(mental_disorder_list, nevous_system_list))
    
    exclude_eid_list = fs_features_ukb.loc[(fs_features_ukb['Diagnosis main ICD10'].str.startswith('G3')) | (fs_features_ukb['Diagnosis main ICD10'].str.startswith('F0'))].eid.unique()
    #exclude_eid_list = fs_features_ukb.loc[(fs_features_ukb['Diagnosis main ICD10'].str.startswith('G3'))].eid.unique()
    
    non_hc_ukb = fs_features_ukb.loc[fs_features_ukb.eid.isin(exclude_eid_list)]
     
    print('Healthy controls selected from UKB = {}'.format(hc_ukb.eid.nunique()))
    print('Non-healthy controls selected as part of UKB internal validation cohort = {}'.format(non_hc_ukb.eid.nunique()))
    
    #assert non_hc_ukb.eid.nunique() + hc_ukb.eid.nunique() == fs_features_ukb.eid.nunique()
    return hc_ukb, non_hc_ukb


ukb_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/UKbiobank'

fs_features_ukb, ukb_dict, cortical_cols_ukb, subcortical_cols_ukb, new_hcm_cols_ukb, demo_cols_ukb = get_roi_demo_ukb(ukb_path)

fs_features_hc_ukb, fs_features_nonhc_ukb = ukb_control_inclusion('pinaya', mental_health = True)

ukb_dict_roi = ukb_dict.loc[ukb_dict.Feature_type.isin(['subcortical', 'cortical', 'hippocampal'])]
ukb_dict_roi['ADNI Name'] = [x.strip(' ') for x in ukb_dict_roi['ADNI Name'].to_list()]


#-------------------------------------------------------------------
#---------------- Extracting healthy controls and disease subjects data from ADNI 
#-------------------------------------------------------------------

tadpole_challenge_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/tadpole_challenge'
roi_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/MR_Image_Analysis'

adnimerge_bl = get_demo_adni(tadpole_challenge_path)
temp_adni = get_roi_adni(roi_path, adnimerge_bl)

other_cols = ['RID', 'DX_bl', 'AGE', 'PTGENDER', 'MMSE', 'ADAS13', 'Intracranial_vol', 'ABETA', 'PTAU', 'mPACCdigit','mPACCtrailsB', 'RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','RAVLT_perc_forgetting']
fs_features_adni, fs_cols_adni, cortical_cols_adni, subcortical_cols_adni, hcm_cols_adni = preprocess_roi_adni(temp_adni, other_cols)
fs_features_adni = fs_features_adni.rename(columns = {'AGE':'Age', 'PTGENDER':'Sex'})


#-------------------------------------------------------------------
#---------------- Combine data from both ADNI and UKBiobank 
#-------------------------------------------------------------------


ukb_dict_hcm = ukb_dict_roi.loc[ukb_dict.Feature_type.isin(['hippocampal'])]
ukb_dict_hcm['Actual_name'] = ukb_dict_hcm['Actual_name'].str.replace('-body ', '')
ukb_dict_hcm['Actual_name'] = ukb_dict_hcm['Actual_name'].str.replace('-head ', '')

ukb_dict_hcm = ukb_dict_hcm.drop_duplicates(subset = ['Actual_name', 'ADNI Name']).reset_index(drop = True)
hcm_rename_dict = {i:j for i,j in zip(ukb_dict_hcm['Actual_name'].to_list(), ukb_dict_hcm['ADNI Name'].to_list())}

ukb_dict_cortical = ukb_dict_roi.loc[ukb_dict.Feature_type.isin(['cortical'])]
cortical_rename_dict = {i:j for i,j in zip(ukb_dict_cortical['Actual_name'].to_list(), ukb_dict_cortical['ADNI Name'].to_list())}

ukb_dict_subcortical = ukb_dict_roi.loc[ukb_dict.Feature_type.isin(['subcortical'])]
subcortical_rename_dict = {i:j for i,j in zip(ukb_dict_subcortical['Actual_name'].to_list(), ukb_dict_subcortical['ADNI Name'].to_list())}

def change_ukb_col_names(fs_features_hc_ukb):
    
    fs_hcm_ukb = fs_features_hc_ukb[new_hcm_cols_ukb].rename(columns=hcm_rename_dict, inplace=False)
    fs_cort_ukb = fs_features_hc_ukb[cortical_cols_ukb].rename(columns=cortical_rename_dict, inplace=False)
    fs_subcort_ukb = fs_features_hc_ukb[subcortical_cols_ukb].rename(columns=subcortical_rename_dict, inplace=False)

    fs_subcort_ukb.columns = fs_subcort_ukb.columns.str.replace(',', '')
    fs_subcort_ukb.columns = fs_subcort_ukb.columns.str.replace("'", "")

    fs_cort_ukb.columns = fs_cort_ukb.columns.str.replace(',', '')
    fs_cort_ukb.columns = fs_cort_ukb.columns.str.replace("'", "")
    
    return fs_cort_ukb, fs_subcort_ukb, fs_hcm_ukb

fs_cort_ukb, fs_subcort_ukb, fs_hcm_ukb = change_ukb_col_names(fs_features_hc_ukb)
fs_cort_nonhc_ukb, fs_subcort_nonhc_ukb, fs_hcm_nonhc_ukb = change_ukb_col_names(fs_features_nonhc_ukb)


#------------------------------------------------------------------- 
#-------------------------------------------------------------------

###### Make sure UKB and ADNI have same ROI names as column headers

def uniform_col_names(fs_features_adni, fs_cort_ukb, fs_subcort_ukb, fs_hcm_ukb):

    fs_hcm_adni = fs_features_adni[hcm_cols_adni]
    #fs_hcm_adni.columns = fs_hcm_ukb.columns

    fs_hcm_adni.columns = fs_hcm_adni.columns.str.replace(',', '')
    fs_hcm_adni.columns = fs_hcm_adni.columns.str.replace("'", "")
    fs_hcm_ukb.columns = fs_hcm_ukb.columns.str.replace(',', '')
    fs_hcm_ukb.columns = fs_hcm_ukb.columns.str.replace("'", "")

    #fs_hcm_adni = fs_hcm_adni[fs_hcm_adni.columns & fs_hcm_ukb.columns]
    #fs_hcm_ukb = fs_hcm_ukb[fs_hcm_adni.columns & fs_hcm_ukb.columns]
    fs_hcm_adni.columns = fs_hcm_ukb.columns

    fs_cort_adni = fs_features_adni[cortical_cols_adni]
    fs_cort_adni = fs_cort_adni[fs_cort_adni.columns & fs_cort_ukb.columns]
    fs_cort_ukb = fs_cort_ukb[fs_cort_adni.columns & fs_cort_ukb.columns]
    fs_cort_adni.columns = fs_cort_ukb.columns

    fs_subcort_adni = fs_features_adni[subcortical_cols_adni]
    fs_subcort_adni = fs_subcort_adni[fs_subcort_adni.columns & fs_subcort_ukb.columns]
    fs_subcort_ukb = fs_subcort_ukb[fs_subcort_adni.columns & fs_subcort_ukb.columns]
    fs_subcort_adni.columns = fs_subcort_ukb.columns
    
    return fs_cort_adni, fs_subcort_adni, fs_hcm_adni, fs_cort_ukb, fs_subcort_ukb, fs_hcm_ukb


fs_cort_adni, fs_subcort_adni, fs_hcm_adni, fs_cort_ukb, fs_subcort_ukb, fs_hcm_ukb = uniform_col_names(fs_features_adni, fs_cort_ukb, fs_subcort_ukb, fs_hcm_ukb)
fs_cort_adni, fs_subcort_adni, fs_hcm_adni, fs_cort_nonhc_ukb, fs_subcort_nonhc_ukb, fs_hcm_nonhc_ukb = uniform_col_names(fs_features_adni, fs_cort_nonhc_ukb, fs_subcort_nonhc_ukb, fs_hcm_nonhc_ukb)


#######

demo_cols_needed = ['Age', 'Sex']

fs_features_adni_roi = pd.concat([fs_cort_adni, fs_subcort_adni, fs_hcm_adni], axis = 1)
roi_demo_adni = pd.concat([fs_features_adni[demo_cols_needed], fs_features_adni_roi], axis = 1)
roi_demo_adni['site'] = 'ADNI'
roi_demo_adni['DX_bl'] = fs_features_adni['DX_bl']
roi_demo_adni['RID'] = fs_features_adni['RID']


fs_features_ukb_roi = pd.concat([fs_cort_ukb, fs_subcort_ukb, fs_hcm_ukb], axis = 1)
roi_demo_ukb = pd.concat([fs_features_hc_ukb[demo_cols_needed], fs_features_ukb_roi], axis = 1).reset_index(drop = True)
roi_demo_ukb['site'] = 'UKB'
roi_demo_ukb['DX_bl'] = 'CN'
roi_demo_ukb['eid'] = fs_features_hc_ukb.reset_index(drop = True)['eid']
roi_demo_ukb = roi_demo_ukb.rename(columns = {'eid':'RID'})

fs_features_ukb_nonhc_roi = pd.concat([fs_cort_nonhc_ukb, fs_subcort_nonhc_ukb, fs_hcm_nonhc_ukb], axis = 1)
roi_demo_nonhc_ukb = pd.concat([fs_features_nonhc_ukb[demo_cols_needed], fs_features_ukb_nonhc_roi], axis = 1).reset_index(drop = True)
roi_demo_nonhc_ukb['site'] = 'UKB'
roi_demo_nonhc_ukb['DX_bl'] = 'Case'
roi_demo_nonhc_ukb['eid'] = fs_features_nonhc_ukb.reset_index(drop = True)['eid']
roi_demo_nonhc_ukb = roi_demo_nonhc_ukb.rename(columns = {'eid':'RID'})


assert(fs_features_ukb_roi.columns.to_list() == fs_features_adni_roi.columns.to_list())
assert(fs_features_ukb_nonhc_roi.columns.to_list() == fs_features_adni_roi.columns.to_list())
fs_cols = fs_features_ukb_roi.columns.to_list()

roi_demo_both = pd.concat([roi_demo_ukb, roi_demo_nonhc_ukb, roi_demo_adni], axis = 0).reset_index(drop = True)

cortical_cols = fs_cort_ukb.columns.values
subcortical_cols = fs_subcort_ukb.columns.values
hcm_cols = fs_hcm_ukb.columns.values

roi_demo_both.to_csv('./saved_dataframes/roi_demo_both.csv')
fs_features_adni.to_csv('./saved_dataframes/fs_features_adni.csv')

with open("./saved_dataframes/cortical_cols", "wb") as fp:
    pickle.dump(cortical_cols, fp)
    
with open("./saved_dataframes/subcortical_cols", "wb") as fp:
    pickle.dump(subcortical_cols, fp)
    
with open("./saved_dataframes/hcm_cols", "wb") as fp:
    pickle.dump(hcm_cols, fp)

# with open("test", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
    