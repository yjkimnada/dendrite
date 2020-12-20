from models.alpha_glm import Alpha_GLM
from models.alpha_cos2_glm import Alpha_Cos2_GLM
from models.alpha_hist_glm import Alpha_Hist_GLM

import numpy as np
import torch
import torch.nn as nn
from tqdm import tnrange
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics

def train_glm(model_type, V, E_neural, I_neural, T_train, T_test,
                T_no, batch_size, iter_no, epoch_no, C_den, C_syn_e, C_syn_i, 
                sparse_no, device, lr, save_dir):

    V_train = V[:T_train].to(device)
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

    if model_type == "alpha":
        model = Alpha_GLM(C_den=C_den,
                         E_no=E_no,
                         I_no=I_no,
                         T_no=T_no,
                         greedy=False,
                         C_syn_e=C_syn_e,
                         C_syn_i=C_syn_i,
                         device = device)

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            ], lr = lr)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
    elif model_type == "alpha_hist":
        model = Alpha_Hist_GLM(C_den=C_den,
                         E_no=E_no,
                         I_no=I_no,
                         T_no=T_no,
                         greedy=False,
                         C_syn_e=C_syn_e,
                         C_syn_i=C_syn_i,
                         device = device)

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            ], lr = lr)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    elif model_type == "alpha_cos":
        model = Alpha_Cos2_GLM(C_den=C_den,
                         E_no=E_no,
                         I_no=I_no,
                         T_no=T_no,
                         greedy=False,
                         C_syn_e=C_syn_e,
                         C_syn_i=C_syn_i,
                         device = device)

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            ], lr = lr)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    model.to(device)

    for i in tnrange(iter_no):
        model.train()
        optimizer.zero_grad()
        
        batch_idx = train_idx[i].long()
        batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_size]
        batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_size]
        batch_V = V_train[batch_idx : batch_idx+batch_size]


        batch_pred, out_filters, C_syn_e, C_syn_i = model(S_e=batch_E_neural,
                                                                S_i=batch_I_neural,
                                                                temp=None,
                                                                test=False)
        var_loss = torch.var(batch_V - batch_pred)
        if model_type == "alpha_hist":
            filter_diff = out_filters[-sub_no:,1:] - out_filters[-sub_no:,:-1]
            smooth_loss = torch.sum(filter_diff**2)
            loss = var_loss + smooth_loss*0.001
        elif (model_type == "alpha") or (model_type == "alpha_cos"):
            loss = var_loss
            
        loss.backward()
        optimizer.step()
        #scheduler.step()

        if i%50 == 0:
            model.eval()
            test_pred, out_filters, C_syn_e, C_syn_i = model(S_e=test_E_neural,
                                                                    S_i=test_I_neural,
                                                                    temp=None,
                                                                    test=True)
            test_score = metrics.explained_variance_score(y_true=V_test.cpu().detach().numpy(),
                                                      y_pred=test_pred.cpu().detach().numpy(),
                                                      multioutput='uniform_average')
            if model_type == "alpha_hist":
                print(i, test_score, var_loss.item(), smooth_loss.item())
            elif (model_type == "alpha") or (model_type == "alpha_cos"):
                print(i, test_score)

    model.train()
    for param in model.parameters():
        param.requires_grad=False

    model.V_o.requires_grad = True

    optimizer = torch.optim.Adam([model.V_o], lr=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.5)

    for i in tnrange(1200):
        optimizer.zero_grad()
        batch_E_neural = train_E_neural[10000:40000]
        batch_I_neural = train_I_neural[10000:40000]
        batch_V = V_train[10000:40000]
        batch_pred, out_filters, C_syn_e, C_syn_i = model(S_e=batch_E_neural,
                                                                S_i=batch_I_neural,
                                                                temp=None,
                                                                test=False)
        mse_loss = torch.mean((batch_V - batch_pred)**2)
        mse_loss.backward()
        optimizer.step()
        scheduler.step()
        if i%100 == 0:
            print(i, mse_loss.item())

    model.eval()
    test_pred, out_filters, C_syn_e, C_syn_i = model(S_e=test_E_neural,
                                                            S_i=test_I_neural,
                                                            temp=None,
                                                            test=True)
    test_pred = test_pred.cpu().detach().numpy()
    C_syn_e = C_syn_e.cpu().detach().numpy()
    C_syn_i = C_syn_i.cpu().detach().numpy()
    out_filters = out_filters.cpu().detach().numpy()

    test_score = metrics.explained_variance_score(y_true=V_test.cpu().detach().numpy(),
                                                    y_pred=test_pred,
                                                    multioutput='uniform_average')
    print(test_score)

    torch.save(model.state_dict(), save_dir+model_type+"_"+"sub"+str(sub_no)+"_model.pt")
    np.savez(save_dir+model_type+"_"+"sub"+str(sub_no)+"_output.npz",
                    test = test_pred,
                    C_syn_e = C_syn_e,
                    C_syn_i = C_syn_i,
                    filters = out_filters)


    
    