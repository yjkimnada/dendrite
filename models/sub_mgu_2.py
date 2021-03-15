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
        #self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        #self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, 1))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

    def forward(self, input_data, future=0):
        batch_size, timesteps, _ = input_data.size()
        outputs = Variable(torch.zeros(batch_size, timesteps + future, self.hidden_size), requires_grad=False)
        
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        
        W_hi = 
        W_hh = 
        
        for i, input_t in enumerate(torch.split(input_data, 1, 1)):
            
            gi = F.linear(input_t.view(batch_size, 1), self.weight_ih, self.bias_ih)
            #gh = F.linear(self.hidden, self.weight_hh, self.bias_hh)
            gh = F.linear()
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            resetgate = self.sigmoid(i_r + h_r)
            inputgate = self.sigmoid(i_i + h_i)
            newgate = self.tanh(i_n + resetgate * h_n)
            self.hidden = newgate + inputgate * (self.hidden - newgate)
            outputs[i] = self.hidden
        
        return outputs