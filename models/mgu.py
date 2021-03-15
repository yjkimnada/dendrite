import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class MGU(nn.Module):
    def __init__(self, input_size, hidden_size, bias, device):
        super(MGU, self).__init__()
        
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(2 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(2 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(2 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        self.hidden = None
    
    def reset_hidden(self):
        self.hidden = None
        self.timesteps = 0
    
    def detach_hidden(self):
        self.hidden.detach_()

    def forward(self, input_data, future=0):
        self.reset_hidden()
        batch_size, timesteps, features = input_data.size()
        outputs = Variable(torch.zeros(batch_size, timesteps + future, self.hidden_size), requires_grad=False).to(self.device)
        if self.hidden is None:
            self.hidden = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False).to(self.device)

        for i, input_t in enumerate(torch.split(input_data, 1, 1)):
            #input_t is (batch, features)

            gi = F.linear(input_t.view(batch_size, features), self.weight_ih, self.bias_ih) #(batch, 2*hidden)
            gh = F.linear(self.hidden, self.weight_hh, self.bias_hh) #(batch, 2*hidden)
            i_f, i_n = gi.chunk(2, 1) #(batch, hidden)
            h_f, h_n = gh.chunk(2, 1) #(batch, hidden)

            forgetgate = torch.sigmoid(i_f + h_f)
            newgate = torch.tanh(i_n + forgetgate * h_n)
            self.hidden = newgate
            #self.hidden = forgetgate * newgate + (1-forgetgate) * self.hidden
            outputs[:,i,:] = self.hidden
        
        return outputs