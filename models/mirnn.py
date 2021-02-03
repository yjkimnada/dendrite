import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn

class MIRNN(nn.Module):

    def __init__(self, input_size, hidden_size, bias, device):
        super(MIRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.weight_ih = Parameter(torch.Tensor(1 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(1 * hidden_size, hidden_size))
        self.weight = Parameter(torch.Tensor(1 * hidden_size))
        
        if bias:
            self.bias_ih = Parameter(torch.Tensor(1 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(1 * hidden_size))
            self.bias = Parameter(torch.Tensor(1 * hidden_size))
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
        timesteps, batch_size, features = input_data.size()
        outputs = Variable(torch.zeros(timesteps + future, batch_size, self.hidden_size).to(self.device), requires_grad=False)
        
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size).to(self.device), requires_grad=False)
        
        for i, input_t in enumerate(input_data.split(1)):
            
            #gi = F.linear(input_t.view(batch_size, features), self.weight_ih, self.bias_ih)
            #gh = F.linear(self.hidden, self.weight_hh, self.bias_hh)
            in_hid = self.weight * torch.matmul(self.weight_ih, input_t.flatten()) * torch.matmul(self.weight_hh, self.hidden.flatten())
            in_only = self.bias_ih * torch.matmul(self.weight_ih, input_t.flatten())
            hid_only = self.bias_hh * torch.matmul(self.weight_hh, self.hidden.flatten())  
            
            self.hidden = torch.tanh(in_hid + in_only + hid_only + self.bias)

            outputs[i] = self.hidden.reshape(1,-1)
        
        return outputs