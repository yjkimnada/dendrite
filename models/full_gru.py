import torch 
import torch.nn as nn
import torch.nn.functional as F

class Full_GRU(nn.Module):
    def __init__(self, E_no, I_no, H_no, device):
        super().__init__()
        
        self.H_no = H_no
        self.device = device
        self.E_no = E_no
        self.I_no = I_no
        
        self.rnn = nn.GRU(self.E_no + self.I_no, self.H_no, batch_first=True)
        self.linear = nn.Linear(self.H_no, 1)
        self.V_o = nn.Parameter(torch.zeros(1))
        
    def forward(self, S_e, S_i):
        T_data = S_e.shape[1]
        batch_size = S_e.shape[0]
        S = torch.zeros(batch_size, T_data, self.E_no+self.I_no).to(self.device)
        S[:,:,:self.E_no] = S[:,:,:self.E_no] + S_e
        S[:,:,self.E_no:] = S[:,:,self.E_no:] + S_i
        
        rnn_out, _ = self.rnn(S)
        lin_out = self.linear(rnn_out.reshape(-1,self.H_no)).reshape(batch_size, T_data)
        final = lin_out + self.V_o
        
        return final