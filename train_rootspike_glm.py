from models.alpha_rootspike_glm import Alpha_RootSpike_GLM

import numpy as np
import torch
import torch.nn as nn
from tqdm import tnrange
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics

def train_glm(model_type, V, Z, E_neural, I_neural, T_train, T_test,
                T_no, batch_size, iter_no, epoch_no, C_den, C_syn_e, C_syn_i, 
                device, lr, save_dir):

    V_train = V[:T_train].to(device).float()
    V_test = V[T_train:T_train + T_test].to(device).float()
    Z_train = Z[:T_train].to(device).float()
    Z_test = Z[T_train:T_train + T_test].to(device).float()
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

    if model_type == "alpha_rootspike":
        model = Alpha_RootSpike_GLM(C_den=C_den,
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=9999999, gamma=0.5)
    

    model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    bce_criterion = nn.BCELoss()

    
    model.eval()
    #V_pred, Z_pred, out_filters = model.test_forward(train_E_neural,
                                                    #train_I_neural)
    
    V_pred, Z_pred, out_filters = model.train_forward(train_E_neural,
                                                     train_I_neural,
                                                     Z_train)
    
    
    avg_diff = torch.mean(V_train - V_pred).item()
    old_V_o = model.V_o.item()
    new_V_o = nn.Parameter(torch.ones(1).to(device) * (avg_diff + old_V_o))
    model.V_o = new_V_o
    
    
    for i in tnrange(iter_no):
        model.train()
        optimizer.zero_grad()
        
        batch_idx = train_idx[i].long()
        batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_size]
        batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_size]
        batch_V = V_train[batch_idx : batch_idx+batch_size]
        batch_Z = Z_train[batch_idx : batch_idx+batch_size]
        
        V_pred, Z_pred, out_filters = model.train_forward(batch_E_neural,
                                                         batch_I_neural,
                                                         batch_Z)
        
        
        var_loss = torch.var(batch_V - V_pred)
        #var_loss = torch.mean((batch_V - V_pred)**2)
        bce_loss = bce_criterion(Z_pred, batch_Z)
        
        loss = var_loss + bce_loss
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_score = metrics.explained_variance_score(y_true=batch_V.cpu().detach().numpy(),
                                                      y_pred=V_pred.cpu().detach().numpy(),
                                                      multioutput='uniform_average')

        if (i%100 == 0) or (i == iter_no-1):
            model.eval()
            
            #V_pred, Z_pred, out_filters = model.test_forward(test_E_neural,
                                                            #test_I_neural)
            
            V_pred, Z_pred, out_filters = model.train_forward(test_E_neural,
                                                             test_I_neural,
                                                             Z_test)
            
            
            test_score = metrics.explained_variance_score(y_true=V_test.cpu().detach().numpy(),
                                                      y_pred=V_pred.cpu().detach().numpy(),
                                                      multioutput='uniform_average')
            test_var = torch.var(V_test - V_pred)
            test_bce = bce_criterion(Z_pred ,Z_test)
            print(i, test_score, test_var.item(), test_bce.item())

    model.eval()
    #V_pred, Z_pred, out_filters = model.test_forward(train_E_neural,
                                                    #train_I_neural)
    
    V_pred, Z_pred, out_filters = model.train_forward(train_E_neural,
                                                     train_I_neural,
                                                     Z_train)
    
    
    avg_diff = torch.mean(V_train - V_pred).item()
    old_V_o = model.V_o.item()
    new_V_o = nn.Parameter(torch.ones(1).to(device) * (avg_diff + old_V_o))
    model.V_o = new_V_o
    
        
    model.eval()
    #V_pred, Z_pred, out_filters = model.test_forward(test_E_neural,
                                                    #test_I_neural)
        
    V_pred, Z_pred, out_filters = model.train_forward(test_E_neural,
                                                     test_I_neural,
                                                     Z_test)
    
    test_pred = V_pred.cpu().detach().numpy()
    #C_syn_e = C_syn_e.cpu().detach().numpy()
    #C_syn_i = C_syn_i.cpu().detach().numpy()
    out_filters = out_filters.cpu().detach().numpy()

    test_score = metrics.explained_variance_score(y_true=V_test.cpu().detach().numpy(),
                                                    y_pred=test_pred,
                                                    multioutput='uniform_average')
    print(test_score)
    print(np.mean((V_test.cpu().detach().numpy() - test_pred)**2))

    torch.save(model.state_dict(), save_dir+model_type+"_"+"sub"+str(sub_no)+"_model.pt")
    np.savez(save_dir+model_type+"_"+"sub"+str(sub_no)+"_output.npz",
                    test = test_pred,
                    #C_syn_e = C_syn_e,
                    #C_syn_i = C_syn_i,
                    filters = out_filters)


    
    