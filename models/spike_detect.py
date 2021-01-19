import torch
from torch import nn
from torch.nn import functional as F

class Spike_Detect(nn.Module):
    def __init__(self, T_no, layer_no, device):
        super().__init__()

        self.T_no = 100
        self.device = device
        self.layer_no = layer_no

        ### TCN ###
        modules = []
        for i in range(self.layer_no):
            if i == 0:
                modules.append(nn.Conv1d(in_channels=1,
                                        out_channels=5,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
                modules.append(nn.LeakyReLU())
            if i == self.layer_no-1:
                modules.append(nn.Conv1d(in_channels=5,
                                        out_channels=1,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
            else:
                modules.append(nn.Conv1d(in_channels=5,
                                        out_channels=5,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
                modules.append(nn.LeakyReLU())
        self.conv_list = nn.Sequential(*modules)
        self.step = Step.apply
        
        self.Tau_out = nn.Parameter(torch.ones(1)*2 , requires_grad=True)
        self.W_out = nn.Parameter(torch.ones(1)*0.25 , requires_grad=True)

    def forward(self, V):
        T_data = V.shape[0]

        nn = self.conv_list(V.reshape(1,1,-1)).flatten()
        Z_out = self.step(nn)
        
        Z_pad = torch.zeros(T_data + self.T_no).to(self.device)
        Z_pad[-T_data:] = Z_pad[-T_data:] + Z_out
        Z_pad = Z_pad.reshape(1,1,-1)
        
        t_out = torch.arange(self.T_no).to(self.device)
        t_tau_out = t_out / torch.exp(self.Tau_out)
        out_kern = t_tau_out * torch.exp(-t_tau_out) * torch.exp(self.W_out)
        out_kern = torch.flip(out_kern, [0]).reshape(1,1,-1)
        
        S_out = F.conv1d(Z_pad, out_kern).flatten()[:-1]
        
        return S_out, Z_out
    
class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        zero = torch.tensor([0.0]).cuda()
        return torch.heaviside(input, zero)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        derivative = 1/(1+torch.abs(input))**2
        #derivative = torch.sigmoid(input)*(1-torch.sigmoid(input))
        output = derivative * grad_input
        return output