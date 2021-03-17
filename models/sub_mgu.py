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
        
        self.W_hin = []
        self.W_hif = []
        self.W_hhn = []
        self.W_hhf = []
        
        for s in range(sub_no):
            self.W_hin.append(Parameter(torch.randn(hidden_size)*0.01))
            self.W_hif.append(Parameter(torch.randn(hidden_size)*0.01))
            self.W_hhn.append(Parameter(torch.randn(hidden_size, hidden_size)*0.01))
            self.W_hhf.append(Parameter(torch.randn(hidden_size, hidden_size)*0.01))
            
        self.W_hin = nn.ParameterList(self.W_hin)
        self.W_hif = nn.ParameterList(self.W_hif)
        self.W_hhn = nn.ParameterList(self.W_hhn)
        self.W_hhf = nn.ParameterList(self.W_hhf)
        
        self.bias_hi = Parameter(torch.randn(sub_no*2 * hidden_size)*0.01)
        self.bias_hh = Parameter(torch.randn(sub_no*2 * hidden_size)*0.01)

    def forward(self, input_data, future=0):
        batch_size, timesteps, _ = input_data.size() #(batch, T, sub_no)
        outputs = Variable(torch.zeros(batch_size, timesteps + future, self.hidden_size*self.sub_no), requires_grad=False).to(self.device)
        self.hidden = Variable(torch.zeros(batch_size, self.sub_no*self.hidden_size), requires_grad=False).to(self.device)
        
        W_hi = torch.zeros(self.sub_no*self.hidden_size*2, self.sub_no).to(self.device)
        W_hh = torch.zeros(self.sub_no*self.hidden_size*2, self.sub_no*self.hidden_size).to(self.device)
        
        for s in range(self.sub_no):
            W_hi[s*self.hidden_size:(s+1)*self.hidden_size,s] = W_hi[s*self.hidden_size:(s+1)*self.hidden_size,s] + self.W_hif[s]
            W_hi[(s+self.sub_no)*self.hidden_size:(s+self.sub_no+1)*self.hidden_size,s] = W_hi[(s+self.sub_no)*self.hidden_size:(s+self.sub_no+1)*self.hidden_size,s] + self.W_hin[s]
            W_hh[s*self.hidden_size:(s+1)*self.hidden_size,s*self.hidden_size:(s+1)*self.hidden_size] = W_hh[s*self.hidden_size:(s+1)*self.hidden_size,s*self.hidden_size:(s+1)*self.hidden_size] + self.W_hhf[s]
            W_hh[(s+self.sub_no)*self.hidden_size:(s+self.sub_no+1)*self.hidden_size,s*self.hidden_size:(s+1)*self.hidden_size] = W_hh[(s+self.sub_no)*self.hidden_size:(s+self.sub_no+1)*self.hidden_size,s*self.hidden_size:(s+1)*self.hidden_size] + self.W_hhn[s]

        for i, input_t in enumerate(torch.split(input_data, 1, 1)):
            #input_t is (batch, sub_no)

            gi = F.linear(input_t.view(batch_size, self.sub_no), W_hi, self.bias_hi) #(batch, 2*hidden*sub)
            gh = F.linear(self.hidden, W_hh, self.bias_hh) #(batch, 2*hidden*sub)
            i_f, i_n = gi.chunk(2, 1) #(batch,sub*hidden); first "sub" many are forget, next "sub" are new
            h_f, h_n = gh.chunk(2, 1) #(batch,sub*hidden)

            forgetgate = torch.sigmoid(i_f + h_f) #(batch, sub*hidden)
            newgate = torch.tanh(i_n + forgetgate * h_n)  #(batch, sub*hidden)
            #newgate = torch.tanh(i_n + h_n)
            #self.hidden = newgate
            self.hidden = forgetgate * newgate + (1-forgetgate) * self.hidden
            outputs[:,i,:] = self.hidden
        
        return outputs