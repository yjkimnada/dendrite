import numpy as np
import torch
import torch.nn as nn
from tqdm import tnrange
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
import os

from greedy_models.greedy_glm import Greedy_GLM


class Greedy_Search:
    def __init__(self, max_sub, V, E_neural, I_neural,
        T_train, T_test, T_no, E_no, I_no, iter_no, batch_size,
        reg, rand, device, save_dir):
    
        self.max_sub = max_sub
        self.T_train = T_train
        self.T_test = T_test
        self.T_no = T_no
        self.iter_no = iter_no
        self.batch_size = batch_size
        self.reg = reg
        self.rand = rand
        self.device = device
        self.save_dir = save_dir
        self.E_no = E_no
        self.I_no = I_no

        self.epoch_no = iter_no*batch_size//T_train

        self.V_train = V[:T_train].float().to(self.device)
        self.V_test = V[T_train:T_train+T_test].float().to(self.device)

        self.train_E_neural = E_neural[:T_train].float().to(self.device)
        self.train_I_neural = I_neural[:T_train].float().to(self.device)
        self.test_E_neural = E_neural[T_train:T_train+T_test].float().to(self.device)
        self.test_I_neural = I_neural[T_train:T_train+T_test].float().to(self.device)

    def search(self):
        best_scores = torch.zeros(self.max_sub-1)

        C_den = torch.zeros(1,1).to(self.device)
        
        for sub_no in range(2,self.max_sub+1,1):
            new_C_den_temp = torch.zeros(sub_no,sub_no).to(self.device)
            new_C_den_temp[:-1,:-1] = C_den

            score_list = torch.zeros(sub_no-1)

            for leaf in range(sub_no - 1):
                new_C_den = new_C_den_temp.detach().clone()
                new_C_den[leaf,-1] = 1
                new_C_den = new_C_den.to(self.device)

                score = self.train(new_C_den, leaf)
                score_list[leaf] = score

            best_leaf = torch.argmax(score_list)
            best_scores[sub_no-2] = score_list[best_leaf]

            best_C_den = new_C_den_temp.detach().clone()
            best_C_den[best_leaf,-1] = 1
            C_den = best_C_den.detach().clone()

            for leaf in range(sub_no-1):
                if leaf != best_leaf:
                    
                    if self.rand == True:
                        rand_status = "rand"
                    elif self.rand == False:
                        rand_status = "norand"
                        
                    os.remove(self.save_dir+"sub"+str(sub_no)+"-"+str(leaf)+"_"+rand_status+"_output.npz")

            print("SUB"+str(sub_no),"SCORE LIST:",score_list)
            print("CUM. SCORE LIST:", best_scores)



    def train(self, C_den, leaf):
        sub_no = C_den.shape[0]
        train_idx = np.empty((self.epoch_no, self.T_train // self.batch_size))
        for i in range(self.epoch_no):
            part_idx = np.arange(0, self.T_train, self.batch_size)
            np.random.shuffle(part_idx)
            train_idx[i] = part_idx
        train_idx = train_idx.flatten()
        train_idx = torch.from_numpy(train_idx)


        model = Greedy_GLM(C_den, self.T_no, self.E_no, self.I_no, self.rand, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
        model.to(self.device)
        
        temp_list = [0.5,0.4,0.3,0.2,0.1,
                     0.09,0.08,0.07,0.06,
                     0.05,0.04,0.03,0.02,0.01,
                    0.009,0.008,0.007,0.006,0.005,
                    0.004,0.003,0.002,0.001]
        temp_count = 0
        
        for i in tnrange(self.iter_no):
            model.train()
            optimizer.zero_grad()

            batch_idx = train_idx[i].long()
            batch_E_neural = self.train_E_neural[batch_idx : batch_idx+self.batch_size]
            batch_I_neural = self.train_I_neural[batch_idx : batch_idx+self.batch_size]
            batch_V = self.V_train[batch_idx : batch_idx+self.batch_size]
            
            temp = temp_list[temp_count]
            if (i%500==499) & (temp_count < 22):
                temp_count += 1

            V_pred, out_filters, C_syn_e, C_syn_i = model(batch_E_neural,
                                                    batch_I_neural,
                                                    temp,
                                                    test=False)

            loss = torch.var(batch_V - V_pred)
            loss.backward()
            optimizer.step()

            """
            #######
            if i%1000 == 999:
                model.eval()
                test_V_pred, test_out_filters, test_C_syn_e, test_C_syn_i = model(self.test_E_neural,
                                                                                self.test_I_neural,
                                                                                temp=0,
                                                                                test=True)
                
                test_score = metrics.explained_variance_score(y_true=self.V_test.cpu().detach().numpy(),
                                                      y_pred=test_V_pred.cpu().detach().numpy())
                print(i, test_score)
            ########
            """

        model.eval()
        train_V_pred, train_out_filters, train_C_syn_e, train_C_syn_i = model(self.train_E_neural,
                                                                            self.train_I_neural,
                                                                            temp=0,
                                                                            test=True)
        mean_error = torch.mean(self.V_train - train_V_pred).item()
        old_V_o = model.V_o.item()
        new_V_o = nn.Parameter(torch.ones(1).to(self.device) * (old_V_o + mean_error))
        model.V_o = new_V_o

        model.eval()
        test_V_pred, test_out_filters, test_C_syn_e, test_C_syn_i = model(self.test_E_neural,
                                                                            self.test_I_neural,
                                                                            temp=0,
                                                                            test=True)

        test_V_pred = test_V_pred.cpu().detach().numpy()
        test_C_syn_e = test_C_syn_e.cpu().detach().numpy()
        test_C_syn_i = test_C_syn_i.cpu().detach().numpy()
        test_out_filters = test_out_filters.cpu().detach().numpy()
        test_C_den = C_den.cpu().detach().numpy()
        test_score = metrics.explained_variance_score(y_true=self.V_test.cpu().detach().numpy(),
                                                    y_pred=test_V_pred)
        
        print(test_score)
        
        if self.rand == True:
            rand_status = "rand"
        elif self.rand == False:
            rand_status = "norand"

        np.savez(self.save_dir+"sub"+str(sub_no)+"-"+str(leaf)+"_"+rand_status+"_output.npz",
                    V = test_V_pred,
                    C_den = test_C_den
                    C_syn_e = test_C_syn_e,
                    C_syn_i = test_C_syn_i,
                    filters = test_out_filters)

        return test_score