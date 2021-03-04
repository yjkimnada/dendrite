import torch
from torch import nn
from torch.nn import functional as F

class Mono_Sign_GLM(nn.Module):
    def __init__(self, E_no, I_no, T_no, device):
        super().__init__()

        self.T_no = T_no
        self.E_no = E_no
        self.I_no = I_no
        self.device = device

        ### Synapse Parameters ###
        #self.W_e = nn.Parameter(torch.rand(self.E_no, self.T_no)*0.01, requires_grad=True)
        #self.W_i = nn.Parameter(torch.rand(self.I_no, self.T_no)*0.01, requires_grad=True)
        
        ### Cosine Basis ###
        self.cos_basis_no = 22
        self.cos_shift = 1.
        self.cos_scale = 5.5
        self.cos_basis = torch.zeros(self.cos_basis_no, self.T_no).to(self.device)
        for i in range(self.cos_basis_no):
            phi = 1.5707963267948966*i
            xmin = phi - 3.141592653589793
            xmax = phi + 3.141592653589793
            
            x_in = torch.arange(self.T_no).float().to(self.device)
            raw_cos = self.cos_scale * torch.log(x_in + self.cos_shift)
            
            basis = 0.5*torch.cos(raw_cos - phi) + 0.5
            basis[raw_cos < xmin] = 0.0
            basis[raw_cos > xmax] = 0.0 
            self.cos_basis[i] = self.cos_basis[i] + basis
        
        self.W_e = nn.Parameter(torch.ones(self.E_no, self.cos_basis_no)*(-4), requires_grad=True)
        self.W_i = nn.Parameter(torch.ones(self.I_no, self.cos_basis_no)*(-4), requires_grad=True)

        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.ones(1)*0 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(1), requires_grad=True)
    
    def forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        
        #e_kern = torch.flip(self.W_e[:,:], [1])
        #i_kern = torch.flip(self.W_i[:,:], [1])
        
        e_kern = torch.matmul(torch.exp(self.W_e), self.cos_basis)
        i_kern = torch.matmul(torch.exp(self.W_i)*(-1), self.cos_basis)
        e_kern = torch.flip(e_kern, [1])
        i_kern = torch.flip(i_kern, [1])
        e_kern = e_kern.unsqueeze(1)
        i_kern = i_kern.unsqueeze(1)

        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.E_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.I_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + S_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + S_i
        pad_syn_e = pad_syn_e.T.unsqueeze(0)
        pad_syn_i = pad_syn_i.T.unsqueeze(0)

        filtered_e = F.conv1d(pad_syn_e, e_kern, padding=0, groups=self.E_no).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, i_kern, padding=0,  groups=self.I_no).squeeze(0).T
 
        syn_in = torch.sum(filtered_e, 1) + torch.sum(filtered_i, 1)
        nonlin_out = torch.tanh(syn_in)
        final = nonlin_out * torch.exp(self.W_sub) + self.V_o

        e_kern_out = torch.flip(e_kern, [2]).squeeze(1)
        i_kern_out = torch.flip(i_kern, [2]).squeeze(1)
        out_filters = torch.vstack((e_kern_out, i_kern_out))

        return final, out_filters

