from models.alpha_emrootspike_glm import Alpha_EMRootSpike_GLM, Encoder

import numpy as np
import torch
import torch.nn as nn
from tqdm import tnrange
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
import time

def em_train_glm(model_type, V, E_neural, I_neural, T_train, T_test,
                T_no, batch_size, iter_no, epoch_no, C_den, C_syn_e, C_syn_i, 
                sparse_no, device, dec_lr, enc_lr, save_dir):

    V_train = V[:T_train].to(device).to(device)
    V_test = V[T_train:T_train + T_test].to(device)
    test_E_neural = E_neural[T_train:T_train+T_test].float().to(device)
    test_I_neural = I_neural[T_train:T_train+T_test].float().to(device)
    train_E_neural = E_neural[:T_train].float().to(device)
    train_I_neural = I_neural[:T_train].float().to(device)
    E_no = E_neural.shape[1]
    I_no = I_neural.shape[1]
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
    
    

    decoder = Alpha_EMRootSpike_GLM(C_den=C_den,
                                E_no=E_no,
                                I_no=I_no,
                                T_no=T_no,
                                C_syn_e=C_syn_e,
                                C_syn_i=C_syn_i,
                                device=device)
    encoder = Encoder(T_no=T_no,
                        device=device)
    decoder.to(device)
    encoder.to(device)
    
    bce_criterion = nn.BCELoss()
    
    dec_count = 0
    enc_count = 0
    iter_count = 0
    
    Z_prior = torch.ones(batch_size).to(device) * 0.0011
    
    pad_V_train = torch.zeros(T_train + 2*(T_no - 1))
    pad_V_train[T_no-1 : -T_no+1] = V_train
    
    repeat_V_train = torch.zeros(T_train, 2*(T_no)-1)
    for t in range(T_train):
        repeat_V_train[t] = pad_V_train[t:t+2*(T_no)-1]
    repeat_V_train = repeat_V_train.float()
    print("success!")
    
    for it in tnrange(1000):
        s = time.time()

        dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=dec_lr)
        enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=enc_lr)
        
        ### M STEP ###
        encoder.eval()
        decoder.train()
        
        

        for j in tnrange(250):
            dec_optimizer.zero_grad()
            
            batch_idx = train_idx[dec_count].long()
            batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_size]
            batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_size]
            batch_V_repeat = repeat_V_train[batch_idx : batch_idx+batch_size].to(device)
            batch_V = V_train[batch_idx : batch_idx+batch_size]
            
            
            Z_enc_raw = encoder(batch_V_repeat, batch_E_neural, batch_I_neural)
            Z_enc = torch.heaviside(Z_enc_raw, torch.tensor([0.0]).to(device))
            
            V_dec, Z_dec, out_filters = decoder.train_forward(batch_E_neural, batch_I_neural, Z_enc)

            Z_loss = bce_criterion(Z_dec, Z_enc)
            V_loss = torch.var(V_dec - batch_V)

            loss = Z_loss + V_loss
            loss.backward()
            dec_optimizer.step()
            dec_count += 1
            

        """
        ### E STEP ###
        encoder.train()
        decoder.eval()
        
        
        for j in range(250):
            enc_optimizer.zero_grad()
            
            batch_idx = train_idx[enc_count].long()
            batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_size]
            batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_size]
            batch_V_repeat = repeat_V_train[batch_idx : batch_idx+batch_size].to(device)
            
            Z_enc_raw = encoder(batch_V_repeat, batch_E_neural, batch_I_neural)
            #Z_enc_raw = encoder(repeat_V_train, train_E_neural, train_I_neural)
            Z_enc = torch.sigmoid(Z_enc_raw)
            kl_div = torch.mean(Z_enc*torch.log(Z_enc/Z_prior+1e-8) + (1-Z_enc)*torch.log((1-Z_enc)/(1-Z_prior)+1e-8))
            
            loss = kl_div
            loss.backward()
            enc_optimizer.step()
            enc_count += 1
        """
        
        ### SLEEP STEP ###
        encoder.train()
        decoder.eval()
        
        
        
        for j in tnrange(250):
            enc_optimizer.zero_grad()
            
            
            batch_idx = train_idx[enc_count].long()
            batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_size]
            batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_size]
            
            V_dec, Z_dec, out_filters = decoder.test_forward(batch_E_neural, batch_I_neural)
            pad_V_dec = torch.zeros(batch_size + 2*(T_no - 1))
            pad_V_dec[T_no-1 : -T_no+1] = V_dec
            V_dec_repeat = torch.zeros(batch_size, 2*(T_no)-1)
            for t in range(batch_size):
                V_dec_repeat[t] = pad_V_dec[t:t+2*(T_no)-1]
            V_dec_repeat = V_dec_repeat.float().cuda()
            
            Z_enc_raw = encoder(V_dec_repeat, batch_E_neural, batch_I_neural)
            Z_enc = torch.sigmoid(Z_enc_raw)
            
            Z_loss = bce_criterion(Z_enc, Z_dec)
            loss = Z_loss
            loss.backward()
            enc_optimizer.step()
            enc_count += 1

        ### TEST STEP ###
        decoder.eval()
        encoder.eval()
        V_pred, Z_pred, out_filters = decoder.test_forward(test_E_neural, test_I_neural)

        var_exp = metrics.explained_variance_score(y_true=V_test.cpu().detach().numpy(),
                                                      y_pred=V_pred.cpu().detach().numpy(),
                                                      multioutput='uniform_average')
        iter_count += 1
        
        print("ITER: ", it, " VAR EXP: ", var_exp, "TEST SPIKES: ",torch.sum(Z_pred).item())
        print("TIME ELAPSED: ", time.time() - s)
        
        