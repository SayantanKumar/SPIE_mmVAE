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

batch_size = 256
lr = 5e-5
latent_dim = 64
alpha = 1
beta = 1
epochs = 1000
k = 1


#******************************************
#******************************************

def getting_traindata_ready(X_train_total, age_sex_df, fs_cols):
    
    X_train_allfold, X_val_allfold, y_train_allfold, y_val_allfold, scale_allfold, age_group_trainfolds, age_group_valfolds = prepare_input_for_training(X_train_total, age_sex_df, fs_cols)

    X_train = X_train_allfold[k]
    X_val = X_val_allfold[k]

    original_dim = X_train.shape[1] 

    train_age_group = age_group_trainfolds[k]
    val_age_group = age_group_valfolds[k]

#     X = Input(shape=(X_train.shape[1],))
#     age_cond = Input(shape=(train_age_group.shape[1],))
    
    return X_train, X_val, train_age_group, val_age_group, scale_allfold


#******************************************
#******************************************

def training_step(X_train_total, age_sex_df, fs_cols):
    
    X_train_allfold, X_val_allfold, y_train_allfold, y_val_allfold, scale_allfold, age_group_trainfolds, age_group_valfolds = prepare_input_for_training(X_train_total, age_sex_df, fs_cols)

    X_train = X_train_allfold[k]
    X_val = X_val_allfold[k]

    original_dim = X_train.shape[1] 

    train_age_group = age_group_trainfolds[k]
    val_age_group = age_group_valfolds[k]

    original_dim = X_train.shape[1] 

    X = Input(shape=(X_train.shape[1],))
    age_cond = Input(shape=(train_age_group.shape[1],))
    encoder_inputs = concat([X, age_cond])


    x = Dense(64, activation = 'relu')(encoder_inputs)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dense(512, activation = 'relu')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    z = Sampling()([z_mean, z_log_var])
    z_cond = concat([z, age_cond])
    encoder = Model([X, age_cond], [z_mean, z_log_var, z_cond], name='encoder') 


    decoder_inputs = Input(shape=((latent_dim + train_age_group.shape[1]),), name='z_sampling')
    x = Dense(512, activation = 'relu')(decoder_inputs)
    x = Dense(256, activation = 'relu')(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)

    decoder_outputs = Dense(original_dim, activation='sigmoid')(x)
    decoder = Model(decoder_inputs, decoder_outputs, name='decoder')

    #--------------------------

    def vae_loss(X, outputs):

        reconstruction_loss = mse(X, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(alpha*reconstruction_loss + beta*kl_loss)

        return vae_loss


    callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0, patience=10)

    outputs = decoder(encoder([X, age_cond])[2])
    vae = Model([X, age_cond], outputs, name='vae')
    vae.compile(optimizer=Adam(lr=lr), loss= vae_loss)

    history = vae.fit([X_train, train_age_group], X_train, epochs=epochs, validation_data = ([X_val, val_age_group], X_val), batch_size=batch_size, callbacks=[callback], verbose = 0)

    return encoder, decoder, vae, history


#******************************************
#******************************************

def modality_specific_operation(X_train_total_UKB, ADNI_CN_model, ADNI_CN_held_val, X_test_org, age_sex_df, fs_cols):
    
    print('Training only on UKB CN...')

    encoder_ukb, decoder_ukb, vae_ukb, history = training_step_case3(X_train_total_UKB, age_sex_df, fs_cols)
    
    X_train_adni, X_val_adni, train_age_group_adni, val_age_group_adni, scale_allfold_adni = getting_traindata_ready(ADNI_CN_model.reset_index(drop = True), age_sex_df, fs_cols)

    callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0, patience=10)

    print('Re-training trained model on ADNI CN...')
    history_retr = vae_ukb.fit([X_train_adni, train_age_group_adni], X_train_adni, epochs=epochs, validation_data = ([X_val_adni, val_age_group_adni], X_val_adni), batch_size=64, callbacks=[callback], verbose = 0)

    X_valho_org, valho_age_group = create_val_ho_data(ADNI_CN_held_val, fs_cols, age_sex_df, scale_allfold_adni)

    X_valho_pred = vae_ukb.predict([X_valho_org[fs_cols], valho_age_group])
    X_valho_pred = pd.DataFrame(X_valho_pred, columns = fs_cols)

    X_test, X_pred_test = vae_inference_unimodal(vae_ukb, X_test_org, scale_allfold_adni, fs_cols, age_sex_df)
    
    return X_test, X_pred_test, X_valho_org, X_valho_pred, history_retr, history


#-----------------------------------------
#******************************************
#-----------------------------------------

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

#-----------------------------------------
#******************************************
#-----------------------------------------


X_test, X_pred_test, X_valho_org, X_valho_pred, history_retr, history = modality_specific_operation(X_train_total_UKB, ADNI_CN_model, ADNI_CN_held_val, X_test_org, age_sex_df, fs_cols)

diff, dev_bvae, mean_df, std_df = calculate_deviations(X_valho_org, X_valho_pred, fs_cols, X_test, X_pred_test)

recon = X_test_org.copy()
recon[fs_cols] = (X_test_org[fs_cols] - X_pred_test[fs_cols])**2
recon['Mean Reconstruction Loss'] = (abs((recon[fs_cols])).sum(axis = 1).to_numpy()/recon[fs_cols].shape[1])
sns.boxplot(x = 'DX_bl', y = 'Mean Reconstruction Loss', data = recon, order = ['SMC', 'EMCI', 'LMCI', 'AD'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
plt.xlabel('Disease category', fontsize = 18)
plt.ylabel('Mean reconstruction loss \n across all brain regions', fontsize = 18)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.title('Mean reconstruction loss', fontsize = 18) 


# plt.figure(figsize = (7,5))
# training_loss = history.history['loss']
# val_loss = history.history['val_loss']
# training_loss_retr = history_retr.history['loss']
# val_loss_retr = history_retr.history['val_loss']
# epochs = range(len(training_loss))

# plt.plot(epochs, training_loss, label='Training loss')
# plt.plot(epochs, val_loss, label='Validation loss')
# plt.plot(epochs, training_loss_retr, label='Training loss')
# plt.plot(epochs, val_loss_retr, label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

dev_bvae.to_csv('./saved_dataframes/unimodal_deviations.csv')
