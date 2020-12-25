from models.alpha_nospike_glm import Alpha_NoSpike_GLM, Encoder

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
    
    

    decoder = Alpha_NoSpike_GLM(C_den=C_den,
                                E_no=E_no,
                                I_no=I_no,
                                T_no=T_no,
                                C_syn_e=C_syn_e,
                                C_syn_i=C_syn_i,
                                device=device)
    encoder = Encoder(sub_no=sub_no,
                        T_no=T_no,
                        device=device)
    decoder.to(device)
    encoder.to(device)
    
    mse_criterion = nn.MSELoss()
    
    dec_count = 0
    enc_count = 0
    iter_count = 0
    
    for it in tnrange(1000):
        s = time.time()

        dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=dec_lr)
        enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=enc_lr)
        
        ### WAKE STEP ###
        encoder.eval()
        decoder.train()

        for j in tnrange(40):
            dec_optimizer.zero_grad()
            
            batch_idx = train_idx[dec_count].long()
            batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_size]
            batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_size]
            batch_V = V_train[batch_idx : batch_idx+batch_size]

            Y_enc = encoder(batch_V, batch_E_neural, batch_I_neural)
            V_dec, Y_dec = decoder.train_forward(batch_E_neural, batch_I_neural, Y_enc)
            
            #Y_enc = encoder(V_train, train_E_neural, train_I_neural)
            #V_dec, Y_dec = decoder.train_forward(train_E_neural, train_I_neural, Y_enc)

            Y_loss = mse_criterion(Y_dec, Y_enc)
            V_loss = torch.var(V_dec - batch_V)
            #V_loss = torch.var(V_dec - V_train)

            loss = Y_loss + V_loss
            loss.backward()
            dec_optimizer.step()
            dec_count += 1

        ### SLEEP STEP ###
        encoder.train()
        decoder.eval()

        for j in tnrange(40):
            enc_optimizer.zero_grad()
            
            batch_idx = train_idx[enc_count].long()
            batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_size]
            batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_size]
            batch_V = V_train[batch_idx : batch_idx+batch_size]
            
            V_dec, Y_dec = decoder.test_forward(batch_E_neural, batch_I_neural)
            Y_enc = encoder(V_dec, batch_E_neural, batch_I_neural)
            
            #V_dec, Y_dec = decoder.test_forward(train_E_neural, train_I_neural)
            #Y_enc = encoder(V_dec, train_E_neural, train_I_neural)

            Y_loss = mse_criterion(Y_enc, Y_dec)
            loss = Y_loss
            loss.backward()
            enc_optimizer.step()
            enc_count += 1


        ### TEST STEP ###
        decoder.eval()
        V_pred, Y_pred = decoder.test_forward(test_E_neural, test_I_neural)

        var_exp = metrics.explained_variance_score(y_true=V_test.cpu().detach().numpy(),
                                                      y_pred=V_pred.cpu().detach().numpy(),
                                                      multioutput='uniform_average')
        iter_count += 1
        
        print("ITER: ", it, " VAR EXP: ", var_exp)
        print("TIME ELAPSED: ", time.time() - s)
        
        