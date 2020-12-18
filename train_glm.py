from gp_glm import GP_GLM
from GP_Hist_GLM import GP_Hist_GLM

import numpy as np
import torch
import torch.nn as nn
from tqdm import tnrange
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics

def train_glm(model_type, V, E_neural, I_neural, T_train, T_test,
                T_no, batch_size, iter_no, epoch_no, C_den, C_syn_e, C_syn_i, 
                sparse_no, save_dir):

    V_train = V[:T_train]
    V_test = V[T_train:T_train + T_test].cuda()
    test_E_neural = E_neural[T_train:T_train+T_test].cuda()
    test_I_neural = I_neural[T_train:T_train+T_test].cuda()
    train_E_neural = E_neural[:T_train]
    train_I_neural = I_neural[:T_train]
    E_no = E_neural.shape[1]
    I_no = I_neural.shape[1]

    batch_no = (train_V_ref.shape[0] - batch_size) * epoch_no
    train_idx = np.empty((repeat_no, train_V_ref.shape[0] - batch_size))
    for i in range(epoch_no):
        part_idx = np.arange(train_V_ref.shape[0] - batch_size)
        np.random.shuffle(part_idx)
        train_idx[i] = part_idx
    train_idx = train_idx.flatten()
    train_idx = torch.from_numpy(train_idx)

    if model_type == "gp":
        model = GP_GLM(C_den=C_den,
                            E_no=E_no,
                            I_no=I_no,
                            T_no=T_no,
                            sparse_no=sparse_no,
                            batch_size=batch_size,
                            greedy=False,
                            C_syn_e=C_syn_e,
                            C_syn_i=C_syn_i)

        optimizer = torch.optim.Adam([
            {'params': model.filter_model.parameters()},
            {'params': model.likelihood.parameters()},
            {'params': model.parameters()},
            ], lr = 0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    elif model_type == "gp_hist":
        model == GP_Hist_GLM(C_den=C_den,
                            E_no=E_no,
                            I_no=I_no,
                            T_no=T_no,
                            sparse_no=sparse_no,
                            batch_size=batch_size,
                            greedy=False,
                            C_syn_e=C_syn_e,
                            C_syn_i=C_syn_i)

        optimizer = torch.optim.Adam([
            {'params': model.filter_model.parameters()},
            {'params': model.likelihood.parameters()},
            {'params': model.parameters()},
            ], lr = 0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    model.cuda()

    for i in tnrange(iter_no):
        model.train()
        optimizer.zero_grad()
        
        batch_idx = train_idx[i].long()
        batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_size].cuda()
        batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_size].cuda()
        batch_V = V_train[batch_idx : batch_idx+batch_size].cuda()


        batch_loss, batch_pred, out_filters, C_syn_e, C_syn_i = model(V_ref=batch_V,
                                                                    S_e=batch_E_neural,
                                                                    S_i=batch_I_neural,
                                                                    temp=None,
                                                                    test=False)
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        if i%100 == 0:
            model.eval()
            test_loss, test_pred, out_filters, C_syn_e, C_syn_i = model(V_ref=V_test
                                                                        S_e=test_E_neural,
                                                                        S_i=test_I_neural,
                                                                        temp=None,
                                                                        test=True)
            test_score = metrics.explained_variance_score(y_true=V_test.cpu().detach().numpy(),
                                                      y_pred=test_pred.cpu().detach().numpy(),
                                                      multioutput='uniform_average')
            print(i, test_score)

    model.train()
    for param in model.parameters():
        param.requires_grad=False

    model.likelihood.V_o.requires_grad = True

    optimizer = torch.optim.Adam([model.likelihood.V_o], lr=0.25)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.5)

    for i in range(1200):
        optimizer.zero_grad()
        batch_loss, batch_pred, out_filters, C_syn_e, C_syn_i = model(V_ref=V_train,
                                                                    S_e=train_E_neural,
                                                                    S_i=train_I_neural,
                                                                    temp=None,
                                                                    test=False)
        batch_loss.backward()
        optimizer.step()

    model.eval()
    test_loss, test_pred, out_filters, C_syn_e, C_syn_i = model(V_ref=V_test
                                                                    S_e=test_E_neural,
                                                                    S_i=test_I_neural,
                                                                    temp=None,
                                                                    test=True)
    test_pred = test_pred.cpu().detach().numpy()
    C_syn_e = C_syn_e.cpu().detach().numpy()
    C_syn_i = C_syn_i.cpu().detach().numpy()
    out_filters = out_filters.cpu().detach().numpy()

    test_score = metrics.explained_variance_score(y_true=V_test.cpu().detach().numpy(),
                                                    y_pred=test_pred.cpu().detach().numpy(),
                                                    multioutput='uniform_average')
    print(test_score)

    torch.save(model.state_dict(), save_dir+"gp_model.pt")
    np.savez(save_dir+"gp_output.npz",
                    test = test_pred,
                    C_syn_e = C_syn_e,
                    C_syn_i = C_syn_i,
                    filters = filters)


    
    