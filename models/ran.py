import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn

class RAN(nn.Module):

    def __init__(self, input_size, hidden_size, bias, out_act, device):
        super(RAN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(2 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        
        self.out_act = out_act
        print(self.out_act)

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
            i_i, i_f, i_n = gi.chunk(3, 1)
            h_i, h_f = gh.chunk(2, 1)

            inputgate = torch.sigmoid(i_i + h_i)
            forgetgate = torch.sigmoid(i_f + h_f)
            newgate = i_n
            self.hidden = inputgate * newgate + forgetgate * self.hidden
            
            if self.out_act == "tanh": 
                outputs[i] = torch.tanh(self.hidden)
            elif self.out_act == None:
                outputs[i] = self.hidden
        
        return outputs
