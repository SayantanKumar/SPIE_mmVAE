# Multi-modal Variational Autoencoder ------ 2 modalities
#https://arxiv.org/pdf/1802.05335.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
import random

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

from utils import *
from dataloaders import *

class MVAE_2M(nn.Module): # @param n_latents : number of latent dimensions
    
    def __init__(self, n_latents, m1_data_shape, m2_data_shape, cond_shape):
        
        super(MVAE_2M, self).__init__()
        
        self.cort_subcort_encoder = cort_subcort_encoder(n_latents, m1_data_shape, cond_shape)
        self.hcm_encoder = hcm_encoder(n_latents, m2_data_shape, cond_shape)
        
        self.cort_subcort_decoder = cort_subcort_decoder(n_latents, m1_data_shape, cond_shape)
        self.hcm_decoder = hcm_decoder(n_latents, m2_data_shape, cond_shape)
        
        self.experts = ProductOfExperts()
        self.n_latents = n_latents
        self.m1_data_shape = m1_data_shape
        self.m2_data_shape = m2_data_shape
        self.cond_shape = cond_shape
    
    def reparametrize(self, mu, logvar):
            
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
            
        else:
            return mu
            
        
        
    def forward(self, cort_subcort, hcm, age_cond):
            
        mu, logvar = self.infer(cort_subcort, hcm, age_cond)
        z = self.reparametrize(mu, logvar)
        cort_subcort_recon = self.cort_subcort_decoder(z, age_cond)
        hcm_recon = self.hcm_decoder(z, age_cond)
            
        return cort_subcort_recon, hcm_recon, mu, logvar
        
        
    def infer(self, cort_subcort, hcm, age_cond): 
            
        batch_size = cort_subcort.size(0) if cort_subcort is not None else cort_subcort.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                      use_cuda=use_cuda)
        
        if cort_subcort is not None:
            cort_subcort_mu, cort_subcort_logvar = self.cort_subcort_encoder(cort_subcort, age_cond)
            mu     = torch.cat((mu, cort_subcort_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, cort_subcort_logvar.unsqueeze(0)), dim=0)
            
    
        if hcm is not None:
            hcm_mu, hcm_logvar = self.hcm_encoder(hcm, age_cond)
            mu     = torch.cat((mu, hcm_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, hcm_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
            
        return mu, logvar
    
    
    
    
class cort_subcort_encoder(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m1_data_shape, cond_shape):
        super(cort_subcort_encoder, self).__init__()
        input_shape = m1_data_shape + cond_shape
        self.fc1   = nn.Linear(input_shape, 64)
        self.fc2   = nn.Linear(64, 128)
        self.fc3   = nn.Linear(128, 256)
        self.fc4   = nn.Linear(256, 512)
        self.fc51  = nn.Linear(512, n_latents)
        self.fc52  = nn.Linear(512, n_latents)
        self.relu = Relu()

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)
        #h = self.swish(self.fc1(x.view(-1, 99)))
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.fc51(h), self.fc52(h)



    
class cort_subcort_decoder(nn.Module):
    
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m1_data_shape, cond_shape):
        super(cort_subcort_decoder, self).__init__()
        decoder_shape = n_latents + cond_shape
        self.fc1   = nn.Linear(decoder_shape, 512)
        self.fc2   = nn.Linear(512, 256)
        self.fc3   = nn.Linear(256, 128)
        self.fc4   = nn.Linear(128, 64)
        self.fc5   = nn.Linear(64, m1_data_shape)
        self.relu = Relu()
        self.sigmoid = Sigmoid()

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.sigmoid(self.fc5(h))  # NOTE: no sigmoid here. See train.py


class hcm_encoder(nn.Module):
    """Parametrizes q(z|y).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m3_data_shape, cond_shape):
        super(hcm_encoder, self).__init__()
        input_shape = m3_data_shape + cond_shape
        self.fc1   = nn.Linear(input_shape, 64)
        self.fc2   = nn.Linear(64, 128)
        self.fc3   = nn.Linear(128, 256)
        self.fc4   = nn.Linear(256, 512)
        self.fc51  = nn.Linear(512, n_latents)
        self.fc52  = nn.Linear(512, n_latents)
        self.relu = Relu()

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.fc51(h), self.fc52(h)

    
    

class hcm_decoder(nn.Module):
    """Parametrizes p(y|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m3_data_shape, cond_shape):
        super(hcm_decoder, self).__init__()
        decoder_shape = n_latents + cond_shape
        self.fc1   = nn.Linear(decoder_shape, 512)
        self.fc2   = nn.Linear(512, 256)
        self.fc3   = nn.Linear(256, 128)
        self.fc4   = nn.Linear(128, 64)
        self.fc5   = nn.Linear(64, m3_data_shape)
        self.relu = Relu()
        self.sigmoid = Sigmoid()

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.sigmoid(self.fc5(h))  # NOTE: no softmax here. See train.py


#-------------------------------------------------------------- 
#-------------------------------------------------------------- 
    
def elbo_loss_2M(cort_subcort_recon, cort_subcort, hcm_recon, hcm, mu, logvar, lambda_m1 = 1.0, lambda_m2 = 1.0, beta = 1):
    
    cort_subcort_recon_mse, hcm_mse = 0,0
    mse_loss = torch.nn.MSELoss(reduction = 'mean')
    
    if cort_subcort_recon is not None and cort_subcort is not None:
        cort_subcort_recon_mse = mse_loss(cort_subcort, cort_subcort_recon)
        cort_subcort_recon_mse *= cort_subcort.shape[1]
        
        
    if hcm_recon is not None and hcm is not None:
        hcm_mse = mse_loss(hcm, hcm_recon)
        hcm_mse *= hcm.shape[1]
        
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_m1 * cort_subcort_recon_mse + lambda_m2 * hcm_mse + beta * KLD)
    
    return ELBO

#-------------------------------------------------------------- 
#-------------------------------------------------------------- 

def train_model_MVAE_2M(X_train_m1, X_train_m2, X_val_m1, X_val_m2, train_age_group, val_age_group, m1_cols, m2_cols, retrain = False):
    
    train_data = train_dataloader_2M(X_train_m1, X_train_m2, train_age_group.values)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=False)

    val_data = val_dataloader_2M(X_val_m1, X_val_m2 , val_age_group.values)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=False)

    # for batch_idx, (X_train_m1, X_train_m2) in enumerate(train_loader):
    #     print(batch_idx, X_train_m1.shape, X_train_m2.shape)

    m1_data_shape = X_train_m1.shape[1] 
    m2_data_shape = X_train_m2.shape[1] 
    cond_shape = train_age_group.shape[1]

    # X = Input(shape=(X_train.shape[1],))
    # age_cond = Input(shape=(train_age_group.shape[1],))
    # encoder_inputs = concat([X, age_cond])

    
    if retrain == True:
        model = torch.load('/Users/sayantankumar/Desktop/Aris_Work/Codes/Normative Modeling/saved_models/trained_MVAE_UKB')
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        model = MVAE_2M(latent_dim, m1_data_shape, m2_data_shape, cond_shape)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loss = []
    val_loss = []
    #best_loss_value = 1000
    for epoch in range(1, epochs + 1):
        train_loss_value, val_loss_value, recon_m1_val, recon_m2_val = train_val_2M(model, train_loader, val_loader, epoch, optimizer)
        train_loss.append(float(train_loss_value.numpy()))
        val_loss.append(float(val_loss_value.numpy()))

    # epochs = range(len(train_loss))
    # plt.plot(epochs, train_loss, label='Training loss')
    # plt.plot(epochs, val_loss, label='Validation loss')
    # plt.title('Training and Validation loss')
    # plt.legend()


    X_org_val_m1 = pd.DataFrame(X_val_m1, columns = m1_cols)
    X_org_val_m2 = pd.DataFrame(X_val_m2, columns = m2_cols)
    
    X_org_val = pd.concat([X_org_val_m1[m1_cols], X_org_val_m2[m2_cols]], axis = 1)

    X_pred_val_m1 = pd.DataFrame(recon_m1_val.detach().numpy(), columns = m1_cols)
    X_pred_val_m2 = pd.DataFrame(recon_m2_val.detach().numpy(), columns = m2_cols)
    
    X_pred_val = pd.concat([X_pred_val_m1[m1_cols], X_pred_val_m2[m2_cols]], axis = 1)
    
    return model, train_loss, val_loss, X_org_val, X_pred_val

#-------------------------------------------------------------- 
#-------------------------------------------------------------- 

def train_val_2M(model, train_loader_2M, val_loader_2M, epoch, optimizer):
    
    model.train()
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    for batch_idx, (X_train_m1, X_train_m2, age_cond_train) in enumerate(train_loader_2M):

        m1_train = Variable(X_train_m1)
        m2_train = Variable(X_train_m2)
        age_cond_train = Variable(age_cond_train)
        #y_train = Variable(y_train)
        
        batch_size = len(m1_train)

        optimizer.zero_grad()

        recon_m1_train, recon_m2_train, mu_train, logvar_train = model(m1_train, m2_train, age_cond_train)

        joint_loss_train = elbo_loss_2M(recon_m1_train, m1_train, recon_m2_train, m2_train, mu_train, logvar_train, lambda_m1=alpha_1, lambda_m2=alpha_2, beta=beta)
    
        train_loss = joint_loss_train
        train_loss_meter.update(train_loss.data, batch_size)

        train_loss.backward()
        optimizer.step()
        
    model.eval()
        
    for batch_idx, (X_val_m1, X_val_m2, age_cond_val) in enumerate(val_loader_2M):
        
        m1_val = Variable(X_val_m1, volatile = True)
        m2_val = Variable(X_val_m2, volatile = True)
        
        age_cond_val = Variable(age_cond_val, volatile = True)
        batch_size = len(m1_val)
        
        recon_m1_val, recon_m2_val, mu_val, logvar_val = model(m1_val, m2_val, age_cond_val)
        
        joint_loss_val = elbo_loss_2M(recon_m1_val, m1_val, recon_m2_val, m2_val, mu_val, logvar_val, lambda_m1=alpha_1, lambda_m2=alpha_2, beta=beta)
    
        val_loss = joint_loss_val
        val_loss_meter.update(val_loss.data, batch_size)

    #print('====> Epoch: {}\t Train Loss: {:.4f} \t Val Loss: {:.4f}'.format(epoch, train_loss_meter.avg, val_loss_meter.avg))
    
    return train_loss_meter.avg, val_loss_meter.avg, recon_m1_val, recon_m2_val


#-------------------------------------------------------------- 
#-------------------------------------------------------------- 


def test_mvae_2M(test_loader_2M, model):
    
    model.eval()
        
    for batch_idx, (X_test_m1, X_test_m2, age_cond_test) in enumerate(test_loader_2M):
        
        m1_test = Variable(X_test_m1, volatile = True)
        m2_test = Variable(X_test_m2, volatile = True)
        
        age_cond_test = Variable(age_cond_test, volatile = True)
        batch_size = len(m1_test)
        
        recon_m1_test, recon_m2_test, mu_val_test, logvar_val_test = model(m1_test, m2_test, age_cond_test)
        
    return recon_m1_test, recon_m2_test


