import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
from tqdm import tnrange

class Custom_GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, sub_size, bias=True):
        super(Custom_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sub_size = sub_size
        
        self.x2h_W = nn.Parameter(torch.randn(sub_size, 3*hidden_size, input_size))
        self.x2h_b = nn.Parameter(torch.randn(sub_size, 3*hidden_size))
        self.h2h_W = nn.Parameter(torch.randn(sub_size, 3*hidden_size, hidden_size))
        self.h2h_b = nn.Parameter(torch.randn(sub_size, 3*hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        # x is size (batch, sub_size)
        # hidden is size (batch, sub_size, hidden)
        
        gate_x = torch.matmul(self.x2h_W.unsqueeze(0), x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1) + self.x2h_b.unsqueeze(0)
        gate_h = torch.matmul(self.h2h_W.unsqueeze(0), hidden.unsqueeze(-1)).squeeze(-1) + self.h2h_b.unsqueeze(0)
        
        i_r, i_i, i_n = gate_x.chunk(3, -1) # (batch, sub, 3H)
        h_r, h_i, h_n = gate_h.chunk(3, -1)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate) # (batch, sub, H)
        
        return hy

class Custom_GRU(nn.Module):
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
        
        self.rnn_cell = Custom_GRUCell(1, H_no, self.sub_no, True)
        self.linear = nn.Parameter(torch.ones(self.sub_no, self.H_no))
        self.V_o = nn.Parameter(torch.zeros(1))
        
    def forward(self, S_e, S_i):
        T_data = S_e.shape[1]
        batch_size = S_e.shape[0]
        S_e = S_e * torch.exp(self.E_scale.reshape(1,1,-1))
        S_i = S_i * torch.exp(self.I_scale.reshape(1,1,-1))*(-1)
        
        S_e_sub = torch.matmul(S_e, self.C_syn_e.T.unsqueeze(0))
        S_i_sub = torch.matmul(S_i, self.C_syn_i.T.unsqueeze(0))
        S_sub = S_e_sub + S_i_sub
        
        raw_out = torch.zeros(batch_size, T_data, self.sub_no, self.H_no).to(self.device)
        hn = Variable(torch.zeros(batch_size, self.sub_no, self.H_no).to(self.device))
        
        for t in tnrange(T_data):
            hn = self.rnn_cell(S_sub[:,t,:], hn)
            raw_out[:,t,:,:] = raw_out[:,t,:,:] + hn
            
        sub_out = torch.sum(raw_out * self.linear.unsqueeze(0).unsqueeze(0), -1)
        final = torch.sum(sub_out, -1) + self.V_o
        
        return final, sub_out
