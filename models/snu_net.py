import torch
from torch import nn
from torch.nn import functional as F

class SNU_Net(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i,
                 weight_init, decay_init, threshold_init, prop_init,
                 device):
        super().__init__()

        self.C_den = C_den
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        self.sub_no = C_den.shape[0]

        self.weight_raw = nn.Parameter(torch.ones(self.sub_no,2)*weight_init , requires_grad=True) # exponential with sign
        self.decay_raw = nn.Parameter(torch.ones(self.sub_no)*decay_init, requires_grad=True) # sigmoid
        self.threshold_raw = nn.Parameter(torch.ones(self.sub_no)*threshold_init, requires_grad=True) # exponential with sign
        self.prop_raw = nn.Parameter(torch.ones(self.sub_no)*prop_init , requires_grad=True) # exponential with sign

        self.step = Step.apply
        
        self.Tau = nn.Parameter(torch.ones(1)*2 , requires_grad=False)
        self.W = nn.Parameter(torch.ones(1)*0.75 , requires_grad=False)

    def forward(self, S_e, S_i):
        V = torch.zeros(self.sub_no).to(self.device)
        spk = torch.zeros(self.sub_no).to(self.device)

        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)

        weight_e = torch.exp(self.weight_raw[:,0])
        weight_i = torch.exp(self.weight_raw[:,1])*(-1)
        decay = torch.sigmoid(self.decay_raw)
        threshold = torch.exp(self.threshold_raw)*(-1)
        prop = torch.exp(self.prop_raw)

        syn_e_weight = syn_e * weight_e.reshape(1,-1)
        syn_i_weight = syn_i * weight_i.reshape(1,-1)

        T_data = S_e.shape[0]

        spk_out = torch.zeros((T_data, self.sub_no)).to(self.device)

        for t in range(T_data):
            V = V*(1.0 - spk)
            V = V * decay
            V = V + syn_e_weight[t].clone() + syn_i_weight[t].clone() + torch.matmul(spk*prop, self.C_den.T)
            #V = F.leaky_relu(V) ###
            spk = self.step(V + threshold)
            spk_out[t] = spk_out[t] + spk

        t_raw = torch.arange(201).to(self.device)
        t_tau = t_raw / torch.exp(self.Tau)
        kern = t_tau * torch.exp(-t_tau) * torch.exp(self.W)
        kern = torch.flip(kern, [0]).reshape(1,1,-1)
        
        final = F.conv1d(spk_out[:,0].reshape(1,1,-1), kern, padding=100).flatten()

        return final, spk_out

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
        output = derivative * grad_input
        return output