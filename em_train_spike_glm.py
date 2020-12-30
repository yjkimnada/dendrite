from models.spike_glm import Spike_GLM, GLM_Encoder

import numpy as np
import torch
import torch.nn as nn
from tqdm import tnrange
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.autograd import Variable


def em_train_spike_glm(Z, E_neural, I_neural, T_train, T_test,
                T_no, batch_size, epoch_no, C_den, C_syn_e, C_syn_i, 
                device, save_dir):

    Z_train = Z[:T_train].to(device).to(device).float()
    Z_test = Z[T_train:T_train + T_test].to(device).float()
    test_E_neural = E_neural[T_train:T_train+T_test].float().to(device)
    test_I_neural = I_neural[T_train:T_train+T_test].float().to(device)
    train_E_neural = E_neural[:T_train].float().to(device)
    train_I_neural = I_neural[:T_train].float().to(device)
    C_syn_e = C_syn_e.float().to(device)
    C_syn_i = C_syn_i.float().to(device)
    C_den = C_den.float().to(device)
    sub_no = C_den.shape[0]

    batch_no = (T_train - batch_size) * epoch_no
    train_idx = np.empty((epoch_no, T_train - batch_size))
    for i in range(epoch_no):
        part_idx = np.arange(T_train - batch_size)
        np.random.shuffle(part_idx)
        train_idx[i] = part_idx
    train_idx = train_idx.flatten()
    train_idx = torch.from_numpy(train_idx)

    decoder = Spike_GLM(C_den, C_syn_e, C_syn_i, T_no, device)
    encoder = GLM_Encoder(C_den, C_syn_e, C_syn_i, T_no, device)
    decoder.to(device)
    encoder.to(device)
    bce_criterion = nn.BCELoss(reduction="none")
    
    print(sum(p.numel() for p in decoder.parameters() if p.requires_grad))
    print(sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    
    loss_factor = 10

    for it in tnrange(1000):

        dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.005)
        enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)
        
        batch_idx = train_idx[it].long()
        batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_size]
        batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_size]
        batch_Z = Z_train[batch_idx : batch_idx+batch_size]

        ########
        ########
        
        encoder.eval()
        decoder.train()

        Z_hid_enc_raw, L_hid_enc_raw = encoder.test_forward(batch_E_neural, batch_I_neural, batch_Z)
        Z_enc = torch.hstack((batch_Z.reshape(-1,1), Z_hid_enc_raw)).detach()
        
        #Z_hid_enc = torch.heaviside(Z_hid_enc_raw-0.5, torch.tensor([1.0]).float().to(device))
        
        #Z_enc = torch.zeros(batch_size, sub_no).to(device)
        #Z_enc[:,0] = Z_enc[:,0] + batch_Z
        #Z_enc[:,1:] = Z_enc[:,1:] + Z_hid_enc
        
        loss_weights = torch.ones_like(Z_enc).to(device)
        loss_weights[Z_enc == 1] *= loss_factor

        for j in range(300):
            dec_optimizer.zero_grad()

            Z_dec, out_filters = decoder.train_forward(batch_E_neural, batch_I_neural, Z_enc)
            
            loss = torch.mean(bce_criterion(Z_dec, Z_enc) * loss_weights)
            #print(torch.numel(torch.where(torch.isnan(Z_dec) == True)[0]))
            
            

            loss.backward()
            dec_optimizer.step()
        
        ########
        ########
        
        encoder.train()
        decoder.eval()

        #Z_dec, out_filters = decoder.test_forward(batch_E_neural, batch_I_neural)
        #Z_obs_dec = Z_dec[:,0].detach()
        #Z_hid_dec = Z_dec[:,1:].detach()
        
        
        
        #loss_weights = torch.ones_like(Z_hid_dec)
        #loss_weights[Z_hid_dec == 1] *= loss_factor
        
        #Z_prior = torch.ones(batch_size, sub_no-1).to(device)*0.1
        
        for j in range(1):
            enc_optimizer.zero_grad()

            #Z_hid_enc = encoder.train_forward(batch_E_neural, batch_I_neural,
                                             #Z_obs_dec, Z_hid_dec)
            Z_hid_enc, L_hid_enc = encoder.test_forward(batch_E_neural, batch_I_neural, batch_Z)
            Z_enc = torch.hstack((batch_Z.reshape(-1,1), Z_hid_enc)).detach()
            Z_dec, out_filters = decoder.train_forward(batch_E_neural, batch_I_neural, Z_enc)
            Z_hid_dec = Z_dec[:,1:]

            #bce_loss = torch.mean(bce_criterion(Z_hid_enc, Z_hid_dec) * loss_weights)
            
            kl_div = torch.mean(L_hid_enc*torch.log(L_hid_enc/Z_hid_dec+1e-8) + (1-L_hid_enc)*torch.log((1-Z_hid_enc)/(1-Z_hid_dec)+1e-8))
            
            loss =  kl_div
            
            loss.backward()
            enc_optimizer.step()

        ### TEST ###
        decoder.eval()
        encoder.eval()
        Z_dec, out_filters = decoder.test_forward(test_E_neural, test_I_neural)
        Z_pred = Z_dec[:,0]

        good_no = 0
        bad_no = 0

        for x in torch.where(Z_pred == 1)[0]:
            close_count = 0
            for y in torch.where(Z_test == 1)[0]:
                if torch.abs(x-y) <= 7:
                    close_count += 1
            if close_count > 0:
                good_no += 1
            else:
                bad_no += 1
        
        loss_weights = torch.ones_like(Z_test)
        loss_weights[Z_test == 1] *= loss_factor
        #test_bce = torch.mean(bce_criterion(Z_dec[:,0], Z_test) * loss_weights).item()
        #test_bce = bce_criterion(Z_dec_raw[:,0], Z_test).item()

        print(it, "GOOD: ", good_no, "BAD: ", bad_no ,"ENC HID: ", torch.numel(torch.where(Z_hid_enc >= 0.5)[0]),
              "DEC HID: ", torch.sum(Z_dec[:,1:]).item())