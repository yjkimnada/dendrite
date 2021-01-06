import torch
from torch import nn
from torch.nn import functional as F

class Cos_GLM(nn.Module):
    def __init__(self, C_den, E_no, I_no, T_no, greedy, C_syn_e, C_syn_i, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.E_no = E_no
        self.I_no = I_no
        self.greedy = greedy
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device

        
        ### Cosine Basis ###
        self.cos_basis_no = 19
        self.cos_shift = 0
        self.cos_scale = 5
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
        
        
        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no, 2) , requires_grad=True)
        #self.W_syn = nn.Parameter(torch.randn(self.sub_no, self.T_no, 2)*0.1, requires_grad=True)

        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no)*1 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### C_syn Parameters ###
        if self.greedy == True:
            self.C_syn_e_logit = nn.Parameter(torch.ones(self.sub_no, self.E_no), requires_grad=True)
            self.C_syn_i_logit = nn.Parameter(torch.ones(self.sub_no, self.I_no), requires_grad=True)
        
        
    
    def forward(self, S_e, S_i, temp, test):
        T_data = S_e.shape[0] 

        if self.greedy == True:
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
                u_e = torch.rand_like(self.C_syn_e_logit).to(self.device)
                u_i = torch.rand_like(self.C_syn_i_logit).to(self.device)
                eps = 1e-8
                g_e = -torch.log(- torch.log(u_e + eps) + eps)
                g_i = -torch.log(- torch.log(u_i + eps) + eps)
                C_syn_e = F.softmax((self.C_syn_e_logit + g_e) / temp, dim=0)
                C_syn_i = F.softmax((self.C_syn_i_logit + g_i) / temp, dim=0)

        elif self.greedy == False:
            C_syn_e = self.C_syn_e
            C_syn_i = self.C_syn_i

        syn_e = torch.matmul(S_e, C_syn_e.T)
        syn_i = torch.matmul(S_i, C_syn_i.T)

        
        full_e_kern = torch.matmul(self.W_syn[:,:,0], self.cos_basis)
        full_i_kern = torch.matmul(self.W_syn[:,:,1], self.cos_basis)
        #full_e_kern = self.W_syn[:,:,0]**2
        #full_i_kern = self.W_syn[:,:,1]**2*(-1)
        
        full_e_kern = torch.flip(full_e_kern, [1])
        full_i_kern = torch.flip(full_i_kern, [1])
        full_e_kern = full_e_kern.unsqueeze(1)
        full_i_kern = full_i_kern.unsqueeze(1)

        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.reshape(1,self.sub_no,-1)
        pad_syn_i = pad_syn_i.T.reshape(1,self.sub_no,-1)

        filtered_e = F.conv1d(pad_syn_e, full_e_kern, padding=0, groups=self.sub_no).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, full_i_kern, padding=0,  groups=self.sub_no).squeeze(0).T
 
        syn_in = filtered_e + filtered_i

        sub_out = torch.zeros(T_data, self.sub_no).to(self.device)
        
        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                nonlin_out = torch.exp(syn_in[:,sub_idx] + self.Theta[sub_idx]) # (T_data,) 
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + nonlin_out
            else:
                leaf_in = sub_out[:,leaf_idx] * self.W_sub[leaf_idx]**2 # (T_data,)
                nonlin_in = syn_in[:,sub_idx] + torch.sum(leaf_in, 1) + self.Theta[sub_idx] # (T_data,)
                nonlin_out = torch.exp(nonlin_in)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + nonlin_out
        
        final_voltage = sub_out[:,0]*self.W_sub[0]**2 + self.V_o

        e_kern_out = torch.flip(full_e_kern, [2]).squeeze(1)
        i_kern_out = torch.flip(full_i_kern, [2]).squeeze(1)
        out_filters = torch.vstack((e_kern_out, i_kern_out))

        return final_voltage, out_filters, C_syn_e, C_syn_i

