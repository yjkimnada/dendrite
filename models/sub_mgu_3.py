import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Sub_MGU(nn.Module):
    def __init__(self, sub_no, input_size, hidden_size, device):
        super(Sub_MGU, self).__init__()
        
        self.device = device
        self.sub_no = sub_no
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_hir = nn.ParameterList()
        self.W_hii = nn.ParameterList()
        self.W_hin = nn.ParameterList()
        self.W_hhr = nn.ParameterList()
        self.W_hhi = nn.ParameterList()
        self.W_hhn = nn.ParameterList()
        
        for s in range(sub_no):
            self.W_hir.append(Parameter(torch.randn(hidden_size)*0.01))
            self.W_hii.append(Parameter(torch.randn(hidden_size)*0.01))
            self.W_hin.append(Parameter(torch.randn(hidden_size)*0.01))
            self.W_hhr.append(Parameter(torch.randn(hidden_size, hidden_size)*0.01))
            self.W_hhi.append(Parameter(torch.randn(hidden_size, hidden_size)*0.01))
            self.W_hhn.append(Parameter(torch.randn(hidden_size, hidden_size)*0.01))
            
        self.bias_ih = Parameter(torch.randn(3*sub_no * hidden_size)*0.01)
        self.bias_hh = Parameter(torch.randn(3*sub_no * hidden_size)*0.01)

    def forward(self, input_data, future=0):
        batch_size, timesteps, _ = input_data.size()
        outputs = Variable(torch.zeros(batch_size, timesteps + future, self.hidden_size*self.sub_no), requires_grad=False).to(self.device)
        self.hidden = Variable(torch.zeros(batch_size, self.sub_no*self.hidden_size), requires_grad=False).to(self.device)
                
        W_hi = torch.zeros(self.sub_no*self.hidden_size*3, self.sub_no).to(self.device)
        W_hh = torch.zeros(self.sub_no*self.hidden_size*3, self.sub_no*self.hidden_size).to(self.device)
        
        for s in range(self.sub_no):
            W_hi[s*self.hidden_size:(s+1)*self.hidden_size,s] = W_hi[s*self.hidden_size:(s+1)*self.hidden_size,s] + self.W_hir[s]
            W_hi[(s+self.sub_no)*self.hidden_size:(s+self.sub_no+1)*self.hidden_size,s] = W_hi[(s+self.sub_no)*self.hidden_size:(s+self.sub_no+1)*self.hidden_size,s] + self.W_hii[s]
            W_hi[(s+2*self.sub_no)*self.hidden_size:(s+2*self.sub_no+1)*self.hidden_size,s] = W_hi[(s+2*self.sub_no)*self.hidden_size:(s+2*self.sub_no+1)*self.hidden_size,s] + self.W_hin[s]
            
            W_hh[s*self.hidden_size:(s+1)*self.hidden_size, s*self.hidden_size:(s+1)*self.hidden_size] = W_hh[s*self.hidden_size:(s+1)*self.hidden_size, s*self.hidden_size:(s+1)*self.hidden_size] + self.W_hhr[s]
            W_hh[(s+self.sub_no)*self.hidden_size:(s+self.sub_no+1)*self.hidden_size, s*self.hidden_size:(s+1)*self.hidden_size] = W_hh[(s+self.sub_no)*self.hidden_size:(s+self.sub_no+1)*self.hidden_size, s*self.hidden_size:(s+1)*self.hidden_size] + self.W_hhr[s]
            W_hh[(s+2*self.sub_no)*self.hidden_size:(s+2*self.sub_no+1)*self.hidden_size, s*self.hidden_size:(s+1)*self.hidden_size] = W_hh[(s+2*self.sub_no)*self.hidden_size:(s+2*self.sub_no+1)*self.hidden_size, s*self.hidden_size:(s+1)*self.hidden_size] + self.W_hhn[s]
        
        for i, input_t in enumerate(torch.split(input_data, 1, 1)):
            
            gi = F.linear(input_t.view(batch_size, self.sub_no), W_hi, self.bias_ih)
            gh = F.linear(self.hidden, W_hh, self.bias_hh)
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            self.hidden = newgate + inputgate * (self.hidden - newgate)
            outputs[:,i,:] = self.hidden
        
        return outputs