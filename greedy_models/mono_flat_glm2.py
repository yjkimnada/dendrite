import torch
from torch import nn
from torch.nn import functional as F

class Mono_Flat_GLM2(nn.Module):
    def __init__(self, sub_no, E_no, I_no, T_no, device):
        super().__init__()

        self.T_no = T_no
        self.E_no = E_no
        self.I_no = I_no
        self.device = device
        self.sub_no = sub_no

        ### Synapse Parameters ###
        self.W_e = nn.Parameter(torch.rand(self.sub_no, self.T_no)*(0.1), requires_grad=True)
        self.W_i = nn.Parameter(torch.rand(self.sub_no, self.T_no)*(-0.1), requires_grad=True)
        #self.W_e = nn.Parameter(torch.ones(self.sub_no, self.T_no)*(-6), requires_grad=True)
        #self.W_i = nn.Parameter(torch.ones(self.sub_no, self.T_no)*(-6), requires_grad=True)
        
        """
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
        
        self.W_e = nn.Parameter(torch.ones(self.E_no, self.cos_basis_no)*(0.01), requires_grad=True)
        self.W_i = nn.Parameter(torch.ones(self.I_no, self.cos_basis_no)*(-0.01), requires_grad=True)
        """
        
        ### C_syn Parameters ###
        self.C_syn_e_logit = nn.Parameter(torch.zeros(self.sub_no, self.E_no), requires_grad=True)
        self.C_syn_i_logit = nn.Parameter(torch.zeros(self.sub_no, self.I_no), requires_grad=True)

        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no+1)*0 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no+1), requires_grad=True)
    
    def forward(self, S_e, S_i, temp, test):
        T_data = S_e.shape[0]
        
        if test == True:
            C_syn_e = torch.zeros_like(self.C_syn_e_logit).to(self.device)
            C_syn_i = torch.zeros_like(self.C_syn_i_logit).to(self.device)
            for i in range(C_syn_e.shape[1]):
                idx = torch.argmax(self.C_syn_e_logit[:,i])
                C_syn_e[idx,i] = 1
            for i in range(C_syn_i.shape[1]):
                idx = torch.argmax(self.C_syn_i_logit[:,i])
                C_syn_i[idx,i] = 1

        elif test == False:
            C_syn_e = F.softmax((self.C_syn_e_logit) / temp, dim=0)
            C_syn_i = F.softmax((self.C_syn_i_logit) / temp, dim=0)
                
        e_kern = torch.flip(self.W_e**2, [1])
        i_kern = torch.flip(self.W_i**2*(-1), [1])
        #e_kern = torch.flip(self.W_e, [1])
        #i_kern = torch.flip(self.W_i, [1])
        #e_kern = torch.matmul(self.W_e, self.cos_basis)
        #i_kern = torch.matmul(self.W_i, self.cos_basis)
        #e_kern = torch.flip(e_kern, [1])
        #i_kern = torch.flip(i_kern, [1])
        e_kern = e_kern.unsqueeze(1)
        i_kern = i_kern.unsqueeze(1)

        syn_e = torch.matmul(S_e, C_syn_e.T)
        syn_i = torch.matmul(S_i, C_syn_i.T)
        
        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.unsqueeze(0)
        pad_syn_i = pad_syn_i.T.unsqueeze(0)

        filtered_e = F.conv1d(pad_syn_e, e_kern, padding=0, groups=self.sub_no).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, i_kern, padding=0,  groups=self.sub_no).squeeze(0).T
 
        syn_in = filtered_e + filtered_i # (T, sub_no)
        #final = torch.sum(syn_in, 1) + self.V_o
        
        root_in = torch.sum(torch.tanh(syn_in + self.Theta[1:].reshape(1,-1))* torch.exp(self.W_sub[1:]) , 1)
        final = root_in + self.V_o

        e_kern_out = torch.flip(e_kern, [2]).squeeze(1)
        i_kern_out = torch.flip(i_kern, [2]).squeeze(1)
        out_filters = torch.vstack((e_kern_out, i_kern_out))

        return final, out_filters

