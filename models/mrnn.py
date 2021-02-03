import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn

class MRNN(nn.Module):

    def __init__(self, input_size, hidden_size, bias, device):
        super(MRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.weight_ih = Parameter(torch.Tensor(2 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(1 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(2 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(1 * hidden_size))
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
    
    def detach_hidden(self):
        self.hidden.detach_()

    def forward(self, input_data, future=0):
        timesteps, batch_size, features = input_data.size()
        outputs = Variable(torch.zeros(timesteps + future, batch_size, self.hidden_size).to(self.device), requires_grad=False)
        
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size).to(self.device), requires_grad=False)
        
        for i, input_t in enumerate(input_data.split(1)):
            
            gi = F.linear(input_t.view(batch_size, features), self.weight_ih, self.bias_ih)
            gh = F.linear(self.hidden, self.weight_hh, self.bias_hh)
            
            i_n, i_f = gi.chunk(2,1)
            
            forgetgate = torch.sigmoid(i_f + gh)
            newgate = i_n
            self.hidden = (1-forgetgate) * newgate + forgetgate * self.hidden
            outputs[i] = self.hidden
        
        return outputs
