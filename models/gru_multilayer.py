import torch 
import torch.nn as nn
import torch.nn.functional as F

class GRU_Multilayer(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, H_no, device):
        super().__init__()
        
        self.H_no = H_no
        self.device = device
        self.sub_no = C_syn_e.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.E_no = C_syn_e.shape[1]
        self.I_no = C_syn_i.shape[1]
        
        self.E_scale = nn.Parameter(torch.zeros(self.E_no))
        self.I_scale = nn.Parameter(torch.zeros(self.I_no))
        
        self.rnn = nn.ModuleList()
        self.linear = nn.ModuleList()
        for s in range(self.sub_no):
            if s%2 == 0:
                self.rnn.append(nn.GRU(1, self.H_no, batch_first=True))
            elif s%2 == 1:
                self.rnn.append(nn.GRU(2, self.H_no, batch_first=True))
            self.linear.append(nn.Linear(self.H_no, 1))

        self.V_o = nn.Parameter(torch.zeros(1))
        
    def forward(self, S_e, S_i):
        T_data = S_e.shape[1]
        batch_size = S_e.shape[0]
        S_e = S_e * torch.exp(self.E_scale.reshape(1,1,-1))
        S_i = S_i * torch.exp(self.I_scale.reshape(1,1,-1))*(-1)
        
        S_e_sub = torch.matmul(S_e, self.C_syn_e.T.unsqueeze(0))
        S_i_sub = torch.matmul(S_i, self.C_syn_i.T.unsqueeze(0))
        S_sub = S_e_sub + S_i_sub
        
        sub_out = torch.zeros(batch_size, T_data, self.sub_no).to(self.device)
        
        for s in range(self.sub_no//2):
            rnn_out, _ = self.rnn[s*2](S_sub[:,:,s*2].unsqueeze(2))
            lin_out = self.linear[s*2](rnn_out.reshape(-1,self.H_no)).reshape(batch_size, T_data)
            rnn_out2, _ = self.rnn[s*2+1](torch.cat((S_sub[:,:,s*2+1].unsqueeze(2), lin_out.unsqueeze(2)), 2))
            lin_out2 = self.linear[s*2+1](rnn_out2.reshape(-1, self.H_no)).reshape(batch_size, T_data)
            sub_out[:,:,s] = sub_out[:,:,s] + lin_out2

        final = torch.sum(sub_out, 2) + self.V_o
        
        return final, sub_out
