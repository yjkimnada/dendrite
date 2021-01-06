import numpy as np
import torch
import torch.nn as nn
from tqdm import tnrange
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tnrange
import os

from greedy_models.greedy_alpha_rootspike_glm import Alpha_RootSpike_GLM


class Greedy_Search:
    def __init__(self, model_type, max_sub, V, Z, E_neural, I_neural, T_train, T_test,
                T_no, iter_no, batch_size, epoch_no, device, save_dir):

        self.device = device
        self.max_sub = max_sub
        self.model_type = model_type
        self.iter_no = iter_no
        self.batch_size = batch_size
        self.epoch_no = epoch_no
        self.T_train = T_train
        self.T_test = T_test
        self.T_no = T_no
        self.V_train = V[:self.T_train].to(self.device).float()
        self.V_test = V[self.T_train:self.T_train + self.T_test].to(self.device).float()
        self.Z_train = Z[:self.T_train].to(self.device).float()
        self.Z_test = Z[self.T_train:self.T_train + self.T_test].to(self.device).float()
        self.test_E_neural = E_neural[self.T_train:self.T_train+self.T_test].float().to(self.device)
        self.test_I_neural = I_neural[self.T_train:self.T_train+self.T_test].float().to(self.device)
        self.train_E_neural = E_neural[:self.T_train].float().to(self.device)
        self.train_I_neural = I_neural[:self.T_train].float().to(self.device)
        self.E_no = E_neural.shape[1]
        self.I_no = I_neural.shape[1]
        self.save_dir = save_dir

    def search(self):
        curr_C_den = torch.zeros((1, 1)).to(self.device)
        final_scores = torch.zeros((self.max_sub-1)).to(self.device)

        for i in tnrange(self.max_sub - 1):
            sub_no = i+2
            score_array = torch.empty((i+1)).to(self.device)

            for j in range(i+1):
                new_C_den = torch.zeros((i+2,i+2)).to(self.device)
                new_C_den[:-1,:-1] = curr_C_den
                new_C_den[j, -1] = 1

                score = self.train(new_C_den)
                score_array[j] = score

            best_idx = torch.argmin(score_array)

            curr_C_den_new = torch.zeros((i+2,i+2)).to(self.device)
            curr_C_den_new[:-1,:-1] = curr_C_den
            curr_C_den_new[best_idx, -1] = 1
            curr_C_den = curr_C_den_new
            final_scores[i] = score_array[best_idx]
            
            for j in range(i+1):
                if j != best_idx:
                    os.remove(self.save_dir+self.model_type+"_"+"sub"+str(sub_no)+"-"+str(j)+"_smooth"+str(self.reg)+"_model.pt")
                    os.remove(self.save_dir+self.model_type+"_"+"sub"+str(sub_no)+"-"+str(j)+"_smooth"+str(self.reg)+"_output.npz")

            print(i+2, "SUB: ", score_array.cpu().detach().numpy())
            print("CUMULATIVE: ", final_scores[:i+1].cpu().detach().numpy())
        

    def train(self, C_den):
        sub_no = int(C_den.shape[0])
        new_idx = torch.where(C_den[:,-1] == 1)[0].item()
        if self.model_type == "alpha_rootspike":
            model = Alpha_RootSpike_GLM(C_den=C_den,
                        E_no=self.E_no,
                        I_no=self.I_no,
                        T_no=self.T_no,
                        device = self.device)

            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                ], lr = 0.002)
            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.5)


        model.to(self.device)
        bce_criterion = nn.BCELoss(reduction="mean")

        temp_list = [0.5, 0.4, 0.3, 0.2, 0.1,
                    0.09, 0.08, 0.07, 0.06,
                    0.05, 0.04, 0.03, 0.02, 0.01]
        temp_count = 0
        
        batch_no = (self.T_train - self.batch_size) * self.epoch_no
        train_idx = np.empty((self.epoch_no, self.T_train - self.batch_size))
        for i in range(self.epoch_no):
            part_idx = np.arange(self.T_train - self.batch_size)
            np.random.shuffle(part_idx)
            train_idx[i] = part_idx
        train_idx = train_idx.flatten()
        train_idx = torch.from_numpy(train_idx)

        for i in tnrange(self.iter_no):
            model.train()
            optimizer.zero_grad()

            if (i%500 == 0) & (temp_count < 14):
                temp = temp_list[temp_count]
                temp_count += 1
                
                
            batch_idx = train_idx[i].long()
            batch_E_neural = self.train_E_neural[batch_idx : batch_idx+self.batch_size]
            batch_I_neural = self.train_I_neural[batch_idx : batch_idx+self.batch_size]
            batch_V = self.V_train[batch_idx : batch_idx+self.batch_size]
            batch_Z = self.Z_train[batch_idx : batch_idx+self.batch_size]

            V_pred, Z_pred, out_filters, C_syn_e, C_syn_i = model.train_forward(batch_E_neural,
                                                            batch_I_neural,
                                                            batch_Z,
                                                            temp)

            bce_loss = bce_criterion(Z_pred, batch_Z)
            var_loss = torch.var(batch_V - V_pred)

            loss = bce_loss + var_loss
                        
            loss.backward()
            optimizer.step()
            #scheduler.step()

        model.eval()
        V_pred, Z_pred, L_pred, out_filters, C_syn_e, C_syn_i = model.test_forward(self.train_E_neural,
                                                        self.train_I_neural)

        avg_diff = torch.mean(self.V_train - V_pred).item()
        old_V_o = model.V_o.item()
        new_V_o = nn.Parameter(torch.ones(1).to(self.device) * (avg_diff + old_V_o))
        model.V_o = new_V_o

        model.eval()
        V_pred, Z_pred, L_pred, out_filters, C_syn_e, C_syn_i = model.test_forward(self.test_E_neural,
                                                    self.test_I_neural)

        test_pred = V_pred.cpu().detach().numpy()
        C_syn_e = C_syn_e.cpu().detach().numpy()
        C_syn_i = C_syn_i.cpu().detach().numpy()
        out_filters = out_filters.cpu().detach().numpy()
        out_spikes = Z_pred.cpu().detach().numpy()
        out_probs = L_pred.cpu().detach().numpy()
            
        test_var_exp = metrics.explained_variance_score(y_true=self.V_test.cpu().detach().numpy(),
                                                    y_pred=test_pred)
        test_mse = torch.mean((V_pred - self.V_test)**2).item()
        test_gauss = 0.5*test_mse
        
        test_bce = bce_criterion(L_pred, self.Z_test).item()
        nll = test_bce + test_gauss

        print("VAR EXP:",np.round(test_var_exp,5), "GAUSS:",np.round(test_gauss,5), "BCE:", np.round(test_bce,5))

        good_no = 0
        bad_no = 0
        for x in torch.where(Z_pred == 1)[0]:
            close_count = 0
            for y in torch.where(self.Z_test == 1)[0]:
                if torch.abs(x-y) <= 5:
                    close_count += 1
            if close_count > 0:
                good_no += 1
            else:
                bad_no += 1
        print("GOOD: ", good_no, "BAD: ", bad_no)

        torch.save(model.state_dict(), self.save_dir+self.model_type+"_"+"sub"+str(sub_no)+"-"+str(new_idx)+"_smooth"+str(self.reg)+"_model.pt")
        np.savez(self.save_dir+self.model_type+"_"+"sub"+str(sub_no)+"-"+str(new_idx)+"_smooth"+str(self.reg)+"_output.npz",
                    test = test_pred,
                    C_syn_e = C_syn_e,
                    C_syn_i = C_syn_i,
                    filters = out_filters,
                    spikes = out_spikes,
                    probs = out_probs)

        return(nll)



    
    
    

