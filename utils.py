import numpy as np
import re
import os
import matplotlib.pyplot as plt
import pandas as pd


#--------------------------------------------------------------
#--------------------------------------------------------------

def get_roi_demo_ukb(ukb_path):
    
    #ukb_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/UKbiobank'
    #ukb_dict = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/UKbiobank/ukb_dictionary.csv')
    
    ukb_dict = pd.read_csv(os.path.join(ukb_path, 'ukb_dictionary.csv'))
    ukb_dict['Actual_name'] = ukb_dict['Actual_name'].str.replace('Volume of ', '')
    ukb_dict['UKB_ID'] = ukb_dict['UKB_ID'].astype(str)
    
    #baal = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/UKbiobank/OUTPUT_ROI.tsv', sep = '\t')
    baal = pd.read_csv(os.path.join(ukb_path, 'OUTPUT_ROI.tsv'), sep = '\t')
    
    
    baal.columns = baal.columns.str.replace('-2.0', '')
    baal.columns = baal.columns.str.replace('-0.0', '')

    baal_new = baal[baal.columns[baal.columns.isin(ukb_dict.UKB_ID.to_list())]].rename(columns=dict(zip(ukb_dict["UKB_ID"], ukb_dict["Actual_name"])))
    baal_new['eid'] = baal['eid']
    baal_new = baal_new.reindex(columns= ['eid'] + ukb_dict.Actual_name.to_list())

    icv_eid_list = baal_new[['eid', 'EstimatedTotalIntraCranial (whole brain)']].dropna().eid.to_list()

    #----------------------------------------------------------

    cortical_cols_ukb = ukb_dict.loc[ukb_dict.Feature_type == 'cortical'].Actual_name.to_list()
    subcortical_cols_ukb = ukb_dict.loc[ukb_dict.Feature_type == 'subcortical'].Actual_name.to_list()
    hcm_cols_ukb = ukb_dict.loc[ukb_dict.Feature_type == 'hippocampal'].Actual_name.to_list()
    demo_cols_ukb = ukb_dict.loc[ukb_dict.Feature_type.isin(['Demographics', 'Site information', 'ICD codes'])].Actual_name.to_list()


    cortical_df = baal_new[baal_new.columns[baal_new.columns.isin(cortical_cols_ukb)]]
    #cortical_df['eid'] = baal_new['eid']

    subcortical_df = baal_new[baal_new.columns[baal_new.columns.isin(subcortical_cols_ukb)]]
    icv = subcortical_df['EstimatedTotalIntraCranial (whole brain)']
    subcortical_df = subcortical_df.drop(columns = 'EstimatedTotalIntraCranial (whole brain)')
    subcortical_cols_ukb = subcortical_df.columns.to_list()

    #subcortical_df['eid'] = baal_new['eid']

    hcm_df = baal_new[baal_new.columns[baal_new.columns.isin(hcm_cols_ukb)]]
    #hcm_df['eid'] = baal_new['eid']

    demo_df = baal_new[baal_new.columns[baal_new.columns.isin(demo_cols_ukb)]]
    #demo_df['eid'] = baal_new['eid']

    #----------------------------------------

    hcm_df['CA1(left hemisphere)'] = hcm_df['CA1-body (left hemisphere)'] + hcm_df['CA1-head (left hemisphere)']
    hcm_df['CA1(right hemisphere)'] = hcm_df['CA1-body (right hemisphere)'] + hcm_df['CA1-head (right hemisphere)']

    hcm_df['CA4(left hemisphere)'] = hcm_df['CA4-body (left hemisphere)'] + hcm_df['CA4-head (left hemisphere)']
    hcm_df['CA4(right hemisphere)'] = hcm_df['CA4-body (right hemisphere)'] + hcm_df['CA4-head (right hemisphere)']

    hcm_df['presubiculum(left hemisphere)'] = hcm_df['presubiculum-body (left hemisphere)'] + hcm_df['presubiculum-head (left hemisphere)']
    hcm_df['presubiculum(right hemisphere)'] = hcm_df['presubiculum-body (right hemisphere)'] + hcm_df['presubiculum-head (right hemisphere)']

    hcm_df['subiculum(left hemisphere)'] = hcm_df['subiculum-body (left hemisphere)'] + hcm_df['subiculum-head (left hemisphere)']
    hcm_df['subiculum(right hemisphere)'] = hcm_df['subiculum-body (right hemisphere)'] + hcm_df['subiculum-head (right hemisphere)']

    hcm_df['CA3(left hemisphere)'] = hcm_df['CA3-body (left hemisphere)'] + hcm_df['CA3-head (left hemisphere)']
    hcm_df['CA3(right hemisphere)'] = hcm_df['CA3-body (right hemisphere)'] + hcm_df['CA3-head (right hemisphere)']

    new_hcm_cols_ukb = ['CA1(left hemisphere)', 'CA1(right hemisphere)', 'CA4(left hemisphere)', 'CA4(right hemisphere)', 'CA3(left hemisphere)', 'CA3(right hemisphere)', 
                        'presubiculum(left hemisphere)', 'presubiculum(right hemisphere)', 'subiculum(left hemisphere)', 'subiculum(right hemisphere)', 'fimbria (left hemisphere)', 
                        'fimbria (right hemisphere)', 'hippocampal-fissure (left hemisphere)', 'hippocampal-fissure (right hemisphere)', 
                        'Hippocampal-tail (left hemisphere)', 'Hippocampal-tail (right hemisphere)', ]

    new_hcm_df =  hcm_df[new_hcm_cols_ukb]

    # all_cols_ukb = demo_cols_ukb + cortical_cols_ukb + subcortical_cols_ukb + new_hcm_cols_ukb # 64 cortical, 37 subcortical, 16 hippocampal 



    final_feature_df = pd.concat([demo_df, cortical_df, subcortical_df, new_hcm_df], axis = 1)
    final_feature_df.insert(loc=0, column='eid', value=baal_new.eid.to_list())


    #-------------------------------------------------------------------------

    all_cols_ukb = cortical_cols_ukb + subcortical_cols_ukb + new_hcm_cols_ukb # 64 cortical, 37 subcortical, 16 hippocampal 

    for col in final_feature_df[all_cols_ukb].columns:
        final_feature_df[col] = final_feature_df[col]/icv

    final_feature_df = final_feature_df.sort_values(by = ['eid', 'Date of attending assessment center']).reset_index(drop = True)

    fs_features_ukb = final_feature_df.drop_duplicates(subset = ['eid'], keep = 'first').reset_index(drop = True)

    for i in all_cols_ukb:
        fs_features_ukb[i] = fs_features_ukb[i].fillna(fs_features_ukb[i].mean())

    #print('{} unique patients total.'.format(fs_features_ukb.eid.nunique()))

    fs_features_ukb = fs_features_ukb.loc[fs_features_ukb.eid.isin(icv_eid_list)]

    print('{} unique patients having ICV values.'.format(fs_features_ukb.eid.nunique()))
    
    return fs_features_ukb, ukb_dict, cortical_cols_ukb, subcortical_cols_ukb, new_hcm_cols_ukb, demo_cols_ukb


#--------------------------------------------------------------
#--------------------------------------------------------------

def screen_icd_ukb(fs_features_ukb):
    
    import itertools

    temp_icd = fs_features_ukb.dropna(subset = ['Diagnosis main ICD10', 'Diagnosis secondary ICD10']).reset_index(drop = True)

    mental_disorder_list = temp_icd.loc[(temp_icd['Diagnosis main ICD10'].str.startswith('F')) | (temp_icd['Diagnosis secondary ICD10'].str.startswith('F'))].eid.unique()
    nevous_system_list = temp_icd.loc[(temp_icd['Diagnosis main ICD10'].str.startswith('G')) | (temp_icd['Diagnosis secondary ICD10'].str.startswith('G'))].eid.unique()
    cerebrovascular_list = temp_icd.loc[(temp_icd['Diagnosis main ICD10'].str.startswith('I6')) | (temp_icd['Diagnosis secondary ICD10'].str.startswith('I6'))].eid.unique()
    benign_neoplasm_meninges_list = temp_icd.loc[(temp_icd['Diagnosis main ICD10'].str.startswith('D32')) | (temp_icd['Diagnosis secondary ICD10'].str.startswith('D32'))].eid.unique()
    benign_neoplasm_cns_list = temp_icd.loc[(temp_icd['Diagnosis main ICD10'].str.startswith('D33')) | (temp_icd['Diagnosis secondary ICD10'].str.startswith('D33'))].eid.unique()
    head_injury_list = temp_icd.loc[(temp_icd['Diagnosis main ICD10'].str.startswith('S09')) | (temp_icd['Diagnosis secondary ICD10'].str.startswith('S09'))].eid.unique()

    exclude_eid_list = list(itertools.chain(mental_disorder_list, nevous_system_list, cerebrovascular_list, benign_neoplasm_meninges_list, benign_neoplasm_cns_list))

    #print('Number of patients excluded = {}'.format(len(list(set(exclude_eid_list)))))

    hc = fs_features_ukb.loc[~fs_features_ukb.eid.isin(list(set(exclude_eid_list)))]
    print('Patients selected to be healthy controls from ICD list = {}'.format(hc.eid.nunique()))

    #hc = hc.loc[(hc.Age.between(47,73)) & (hc['UKB assessment center'] == 11025.0)]
    #hc = hc.loc[(hc.Age.between(47,73))]

    #print('Patients selected after screening for age and centre = {}'.format(hc.eid.nunique()))
    
    return hc

#--------------------------------------------------------------
#--------------------------------------------------------------

def get_cog_mh_ukb(ukb_path, ukb_dict, hc):
    
    #cog_mh = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/UKbiobank/OUTPUT_cog_mh.tsv', sep = '\t')
    cog_mh = pd.read_csv(os.path.join(ukb_path, 'OUTPUT_cog_mh.tsv'), sep = '\t')
    
    
    cog_mh = cog_mh.loc[cog_mh.eid.isin(hc.eid.unique())]
    cog_mh.columns = cog_mh.columns.str.replace('-2.0', '')
    cog_mh.columns = cog_mh.columns.str.replace('-0.0', '')

    #ukb_dict = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/UKbiobank/ukb_dictionary.csv')
    ukb_dict = pd.read_csv(os.path.join(ukb_path, 'ukb_dictionary.csv'))

    PHQ_col_list = ukb_dict.loc[ukb_dict.Feature_type == 'Patient Health Questionaire (PHQ-9)'].Actual_name.unique()
    RDS_col_list = ukb_dict.loc[ukb_dict.Feature_type == 'Recent depressive symptoms (RDS-4)'].Actual_name.unique()
    GAD_col_list = ukb_dict.loc[ukb_dict.Feature_type == 'General anxiety disorder (GAD-7)'].Actual_name.unique()
    neuro_col_list = ukb_dict.loc[ukb_dict.Feature_type == 'Neuroticism (N-12)'].Actual_name.unique()
    pds_col_list = ukb_dict.loc[ukb_dict.Feature_type == 'Probable depression status'].Actual_name.unique()

    ukb_dict['Actual_name'] = ukb_dict['Actual_name'].str.replace('Volume of ', '')
    ukb_dict['UKB_ID'] = ukb_dict['UKB_ID'].astype(str)

    cog_mh_df = cog_mh[cog_mh.columns[cog_mh.columns.isin(ukb_dict.UKB_ID.to_list())]].rename(columns=dict(zip(ukb_dict["UKB_ID"], ukb_dict["Actual_name"])))
    cog_mh_df['eid'] = hc['eid']

    RDS_df = cog_mh_df[RDS_col_list].dropna(subset = RDS_col_list)

    for col in cog_mh_df[RDS_col_list]:
        RDS_df.loc[RDS_df[col] == -818.0, col] = np.nan
        RDS_df.loc[RDS_df[col] == -1.0, col] = np.nan
        RDS_df.loc[RDS_df[col] == -3.0, col] = np.nan

    RDS_df['total_rds'] = RDS_df.sum(axis = 1)
    cog_mh_df['total_rds'] = RDS_df['total_rds']
    
    return cog_mh_df


#--------------------------------------------------------------
#--------------------------------------------------------------


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


#--------------------------------------------------------------
#--------------------------------------------------------------

def get_demo_adni(tadpole_challenge_path):
    
    adnimerge = pd.read_csv(os.path.join(tadpole_challenge_path, 'ADNIMERGE.csv'), low_memory = False)
    adnimerge_bl = adnimerge.loc[adnimerge.VISCODE == 'bl'].reset_index(drop = True)

    adnimerge = pd.read_csv(os.path.join(tadpole_challenge_path, 'ADNIMERGE.csv'), low_memory = False)
    #adnimerge_bl = adnimerge.loc[adnimerge.VISCODE == 'bl'].reset_index(drop = True)
    adnimerge_bl = adnimerge.sort_values(by = ['RID', 'EXAMDATE']).drop_duplicates(subset = ['RID'], keep = 'first').reset_index(drop = True)

    adnimerge_cols = ['RID', 'VISCODE', 'EXAMDATE', 'DX_bl', 'AGE', 'PTGENDER', 'MMSE', 'ADAS13', 'ICV_bl', 'ABETA', 'PTAU', 'mPACCdigit','mPACCtrailsB', 'RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','RAVLT_perc_forgetting']

    adnimerge_bl = adnimerge_bl[adnimerge_cols].rename(columns = {'ICV_bl':'Intracranial_vol'}).drop(columns = 'VISCODE')

    print('{} unique patients in ADNIMerge.'.format(adnimerge.RID.nunique()))

    mean_impute_cols = ['mPACCdigit','mPACCtrailsB', 'RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','RAVLT_perc_forgetting', 'Intracranial_vol']
    median_impute_cols = ['AGE', 'MMSE', 'ADAS13']

    for col in mean_impute_cols:
        adnimerge_bl[col] = adnimerge_bl[col].fillna(adnimerge_bl[col].mean())

    for col in median_impute_cols:
        adnimerge_bl[col] = adnimerge_bl[col].fillna(adnimerge_bl[col].median())

    ##### ABeta and PTau not imputed since there are around 50% missing values

    adnimerge_bl['AGE'] = round(adnimerge_bl['AGE'])
    adnimerge_bl['PTAU'] = pd.to_numeric(adnimerge_bl['PTAU'], errors = 'coerce')
    adnimerge_bl['ABETA'] = pd.to_numeric(adnimerge_bl['ABETA'], errors = 'coerce')
    
    return adnimerge_bl

#--------------------------------------------------------------
#--------------------------------------------------------------

def get_roi_adni(roi_path, adnimerge_bl):
    
    ucsf_data = pd.read_csv(os.path.join(roi_path, 'UCSFFSX51_11_08_19.csv'))

    ucsf_data = ucsf_data.sort_values(by=['RID', 'EXAMDATE']).reset_index(drop = True)

    print('{} unique patients in UCSF data.'.format(ucsf_data.RID.nunique()))

    freesurfer_cols = [col for col in ucsf_data.columns if 'ST' in col]
    all_cols =  ['RID', 'EXAMDATE'] + freesurfer_cols

    ucsf_data = ucsf_data[all_cols].drop(columns = 'STATUS').drop_duplicates(subset = 'RID', keep = 'first')

    ##---------------------------------------------------------------------------------------

    adni3 = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/MR_Image_Analysis/UCSFFSX51_ADNI1_3T_02_01_16.csv')

    adni3 = adni3.sort_values(by=['RID', 'EXAMDATE']).reset_index(drop = True)

    #print('{} unique patients.'.format(adni3.RID.nunique()))

    freesurfer_cols = [col for col in adni3.columns if 'ST' in col]
    all_cols =  ['RID', 'EXAMDATE'] + freesurfer_cols

    adni3t = adni3[all_cols].drop(columns = 'STATUS').drop_duplicates(subset = 'RID', keep = 'first').dropna(axis = 'columns', thresh = 50)

    ##---------------------------------------------------------------------------------------

    adni_all = pd.concat([ucsf_data, adni3t]).sort_values(by = ['RID', 'EXAMDATE']).drop_duplicates(subset = 'RID', keep = 'first')

    temp = pd.merge(adnimerge_bl, adni_all, on = ['RID'], how = 'right').drop(columns = ['EXAMDATE_x', 'EXAMDATE_y', 'ST8SV'])
    
    
    return temp


#--------------------------------------------------------------
#--------------------------------------------------------------

def preprocess_roi_adni(temp, other_cols):
    
    input_data = temp.copy()

    #other_cols = ['RID', 'DX_bl', 'AGE', 'PTGENDER', 'MMSE', 'ADAS13', 'Intracranial_vol', 'ABETA', 'PTAU', 'mPACCdigit','mPACCtrailsB', 'RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','RAVLT_perc_forgetting']


    #####-----------------------------------------------------------------------------------------
    cortical_cols = input_data.loc[:, input_data.columns.str.endswith('CV')].columns.to_list() # 69
    subcortical_cols = input_data.loc[:, input_data.columns.str.endswith('SV')].columns.to_list() # 49
    hcm_cols = input_data.loc[:, input_data.columns.str.endswith('HS')].columns.to_list() # 16

    surface_area_cols = input_data.loc[:, input_data.columns.str.endswith('SA')].columns.to_list() # 70
    mean_cortical_thickness_cols = input_data.loc[:, input_data.columns.str.endswith('TA')].columns.to_list() # 68
    std_cortical_thickness_cols = input_data.loc[:, input_data.columns.str.endswith('TS')].columns.to_list() # 68

    #----------------------------------------------------------------------------


    ucsf_dict = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/MR_Image_Analysis/UCSFFSX51_DICT_08_01_14.csv')
    ucsf_dict = ucsf_dict.loc[ucsf_dict.FLDNAME.str.startswith('ST')][['FLDNAME', 'TEXT']].dropna().set_index('FLDNAME')

    ##-----------------Cortical-------------------------------------------

    a_list_cortical = list(ucsf_dict.loc[cortical_cols]["TEXT"].values) #column to list
    string_cortical = " ".join(a_list_cortical) # list of rows to string
    words_cortical = re.findall("(\w+)", string_cortical) # split to  single list of words

    cortical_roi = [item for item in words_cortical if words_cortical.count(item) == 1] #list of words that appear multiple times

    ##-----------------Subcortical-------------------------------------------

    a_list_subcortical = list(ucsf_dict.loc[subcortical_cols]["TEXT"].values) #column to list
    string_subcortical = " ".join(a_list_subcortical) # list of rows to string
    words_subcortical = re.findall("(\w+)", string_subcortical) # split to  single list of words

    subcortical_roi = [item for item in words_subcortical if words_subcortical.count(item) == 1] #list of words that appear multiple times

    ##-------------------Hippocampal-----------------------------------------

    a_list_hcm = list(ucsf_dict.loc[hcm_cols]["TEXT"].values) #column to list
    string_hcm = " ".join(a_list_hcm) # list of rows to string
    words_hcm = re.findall("(\w+)", string_hcm) # split to  single list of words

    hcm_roi = [item for item in words_hcm if words_hcm.count(item) == 1] #list of words that appear multiple times

    ##-----------------------------------------------------------------

    hcm_rename_dict = {i:j for i,j in zip(hcm_cols,hcm_roi)}
    cortical_rename_dict = {i:j for i,j in zip(cortical_cols,cortical_roi)}
    subcortical_rename_dict = {i:j for i,j in zip(subcortical_cols,subcortical_roi)}

    fs_hcm = input_data[hcm_cols].rename(columns=hcm_rename_dict, inplace=False)
    fs_cort = input_data[cortical_cols].rename(columns=cortical_rename_dict, inplace=False)
    fs_subcort = input_data[subcortical_cols].rename(columns=subcortical_rename_dict, inplace=False)
    
    
    subcort_remove_cols = ['OpticChiasm', 'LeftChoroidPlexus', 'RightChoroidPlexus', 'LeftVessel', 'RightVessel', 'NonWMHypoIntensities', 'LeftVentralDC', 'RightVentralDC', 'LeftInferiorLateralVentricle', 'RightInferiorLateralVentricle', 'FourthVentricle']
    fs_subcort = fs_subcort.drop(columns = subcort_remove_cols)

    cort_remove_cols = ['LeftBankssts', 'Icv', 'RightBankssts']
    fs_cort = fs_cort.drop(columns = cort_remove_cols)

    ##------------------------------------------------------------------------------------
    hcm_cols = fs_hcm.columns.to_list()
    cortical_cols = fs_cort.columns.to_list()
    subcortical_cols = fs_subcort.columns.to_list()

    input_data = pd.concat([input_data[other_cols], fs_cort, fs_subcort, fs_hcm], axis = 1)
    
    ##-------------------------------------------------------------------------
    
    fs_cols = cortical_cols + subcortical_cols + hcm_cols

    fs_features = input_data.copy()


    for col in fs_features[cortical_cols + subcortical_cols + hcm_cols].columns:
        fs_features[col] = fs_features[col]/fs_features['Intracranial_vol']

    fs_features = fs_features.dropna(subset = ['Intracranial_vol'])

    fs_features = fs_features[other_cols + fs_cols].copy()
    
    fs_features = fs_features.drop_duplicates(subset = ['RID'], keep = 'first').reset_index(drop = True)

    for i in fs_cols:
        fs_features[i] = fs_features[i].fillna(fs_features[i].mean())

    print('{} patients selected finally from ADNI after preprocessing.'.format(fs_features.RID.nunique()))

    return fs_features, fs_cols, cortical_cols, subcortical_cols, hcm_cols


#--------------------------------------------------------------
#--------------------------------------------------------------


def one_hot_encoding(table, col): # returns one hot encoding matrix
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(table[col].values)
    #print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    
    return onehot_encoded
    #return (to_categorical(table.Age_group.to_numpy()))


def min_max_scaling(train_df, val_df):
    
    train_df_scaled = MinMaxScaler().fit(train_df).transform(train_df)
    val_df_scaled = MinMaxScaler().fit(train_df).transform(val_df)

    return train_df_scaled, val_df_scaled


def standard_scaling(train_df, val_df):
    
    train_df_scaled = StandardScaler().fit(train_df).transform(train_df)
    val_df_scaled = StandardScaler().fit(train_df).transform(val_df)

    return train_df_scaled, val_df_scaled


def robust_scaling(train_df, val_df):
    
    train_df_scaled = RobustScaler().fit(train_df).transform(train_df)
    val_df_scaled = RobustScaler().fit(train_df).transform(val_df)

    return train_df_scaled, val_df_scaled


def quantile_transformer_scaling(train_df, val_df):
    
    train_df_scaled = QuantileTransformer().fit(train_df).transform(train_df)
    val_df_scaled = QuantileTransformer().fit(train_df).transform(val_df)

    return train_df_scaled, val_df_scaled



#******************************************
#******************************************    

def convert_cols_ggseg(table, cortical_cols, subcortical_cols):
    
    temp_mat = table.copy()

    cortical_cols_lh = [col for col in cortical_cols if 'Left' in col]
    cortical_cols_rh = [col for col in cortical_cols if 'Right' in col]

    cortical_cols_lh_new = list(temp_mat[cortical_cols_lh].columns.str.lower().str.replace('left',''))
    cortical_cols_rh_new = list(temp_mat[cortical_cols_rh].columns.str.lower().str.replace('right',''))

    cortical_cols_lh_new = ['{}_{}'.format(a1, b1) for b1 in ['left'] for a1 in cortical_cols_lh_new]
    cortical_cols_rh_new = ['{}_{}'.format(a2, b2) for b2 in ['right'] for a2 in cortical_cols_rh_new]

    temp_mat[cortical_cols_lh_new] = temp_mat[cortical_cols_lh].rename(columns=dict(zip(cortical_cols_lh_new, cortical_cols_lh)))
    temp_mat[cortical_cols_rh_new] = temp_mat[cortical_cols_rh].rename(columns=dict(zip(cortical_cols_rh_new, cortical_cols_rh)))

    #-----------------------------------------------
    
    subcortical_cols_ggseg = ['Right-Pallidum', 'Right-Putamen', 'Left-Accumbens-Area', 'Right-Thalamus', '3rd-Ventricle',
                         'Wm_Hypointensities', 'Left-Amygdala', 'Left-Caudate', 'Left-Cerebellum-Cortex', 'Left-Cerebellum-White-Matter',
                         'Brain-Stem', 'Left-Hippocampus', 'CC_Anterior', 'Left-Lateral-Ventricle', 'CC_Central', 'Left-Pallidum',
                         'CC_Mid_Anterior', 'Left-Putamen', 'CC_Mid_Posterior', 'Left-Thalamus', 'CC_Posterior', 'Right-Accumbens-Area',
                         'Right-Amygdala', 'Right-Caudate', 'Right-Cerebellum-Cortex', 'Right-Cerebellum-White-Matter', 'Csf', 'Right-Hippocampus',
                         'Right-Lateral-Ventricle', 'Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter']


    subcortical_cols_plot = list(temp_mat[subcortical_cols].drop(columns = ['LeftCorticalGM', 'RightCorticalGM','SubcorticalGM', 'TotalGM']).columns.values)
    temp_mat[subcortical_cols_ggseg] = temp_mat[subcortical_cols_plot].rename(columns=dict(zip(subcortical_cols_ggseg, subcortical_cols_plot)))

    
    return cortical_cols_lh_new, cortical_cols_rh_new, subcortical_cols_ggseg, temp_mat

#cortical_cols_lh_new, cortical_cols_rh_new, subcortical_cols_ggseg, temp_mat = convert_cols_ggseg(dev_bvae_c3_n1, cortical_cols, subcortical_cols)
