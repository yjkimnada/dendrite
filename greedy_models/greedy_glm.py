import torch
from torch import nn
from torch.nn import functional as F

class Greedy_GLM(nn.Module):
    def __init__(self, C_den, T_no, E_no, I_no, rand, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.device = device
        self.E_no = E_no
        self.I_no = I_no
        self.rand = rand

        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.ones(self.sub_no,2)*(-3) , requires_grad=True) # Exp with sign
        self.Tau_syn = nn.Parameter(torch.ones(self.sub_no,2)*(1) , requires_grad=True) # Exp
        self.Delta_syn = nn.Parameter(torch.ones(self.sub_no,2)*(0) , requires_grad=True) # Exp

        ### Ancestor Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no)*0 , requires_grad=True) # Exp

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### C_syn Parameters ###
        self.C_syn_e_raw = nn.Parameter(torch.zeros(self.sub_no, self.E_no), requires_grad=True)
        self.C_syn_i_raw = nn.Parameter(torch.zeros(self.sub_no, self.I_no), requires_grad=True)

    def forward(self, S_e, S_i, temp, test):
        T_data = S_e.shape[0]

        if test == True:
            C_syn_e = torch.zeros_like(self.C_syn_e_raw).to(self.device)
            C_syn_i = torch.zeros_like(self.C_syn_i_raw).to(self.device)
            for i in range(self.E_no):
                idx = torch.argmax(self.C_syn_e_raw[:,i])
                C_syn_e[idx,i] = 1
            for i in range(self.I_no):
                idx = torch.argmax(self.C_syn_i_raw[:,i])
                C_syn_i[idx,i] = 1

        elif test == False:
            if self.rand == True:
                u_e = torch.rand_like(self.C_syn_e_raw).to(self.device)
                u_i = torch.rand_like(self.C_syn_i_raw).to(self.device)
                eps = 1e-8
                g_e = -torch.log(- torch.log(u_e + eps) + eps)
                g_i = -torch.log(- torch.log(u_i + eps) + eps)
                C_syn_e = F.softmax((self.C_syn_e_raw + g_e) / temp, dim=0)
                C_syn_i = F.softmax((self.C_syn_i_raw + g_i) / temp, dim=0)

            elif self.rand == False:
                C_syn_e = F.softmax(self.C_syn_e_raw/temp , dim=0)
                C_syn_i = F.softmax(self.C_syn_i_raw/temp , dim=0)

        syn_e = torch.matmul(S_e, C_syn_e.T)
        syn_i = torch.matmul(S_i, C_syn_i.T)

        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t_e = t - torch.exp(self.Delta_syn[:,0]).reshape(-1,1)
        t_i = t - torch.exp(self.Delta_syn[:,1]).reshape(-1,1)
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0 

        tau_e = torch.exp(self.Tau_syn[:,0].reshape(-1,1))
        tau_i = torch.exp(self.Tau_syn[:,1].reshape(-1,1))
        t_tau_e = t_e / tau_e
        t_tau_i = t_i / tau_i
        W_e = torch.exp(self.W_syn[:,0]).reshape(-1,1)
        W_i = torch.exp(self.W_syn[:,1]).reshape(-1,1)*(-1)
        
        e_kern = t_tau_e * torch.exp(-t_tau_e) * W_e
        i_kern = t_tau_i * torch.exp(-t_tau_i) * W_i
        e_kern = torch.flip(e_kern, [1])
        i_kern = torch.flip(i_kern, [1])

        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.unsqueeze(0)
        pad_syn_i = pad_syn_i.T.unsqueeze(0)

        filt_e = F.conv1d(pad_syn_e, e_kern.unsqueeze(1), groups=self.sub_no).squeeze(0).T
        filt_i = F.conv1d(pad_syn_i, i_kern.unsqueeze(1), groups=self.sub_no).squeeze(0).T
 
        syn = filt_e + filt_i

        sub_out = torch.zeros(T_data, self.sub_no).to(self.device)
        
        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                sub_in = syn[:,sub_idx] + self.Theta[sub_idx]
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.tanh(sub_in)
            else:
                leaf_in = sub_out[:,leaf_idx] * torch.exp(self.W_sub[leaf_idx])
                sub_in = syn[:,sub_idx] + torch.sum(leaf_in.reshape(T_data,-1) , 1) + self.Theta[sub_idx]
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.tanh(sub_in)

        V = sub_out[:,0]*torch.exp(self.W_sub[0]) + self.V_o

        e_kern_out = torch.flip(e_kern, [1]).reshape(self.sub_no,-1)
        i_kern_out = torch.flip(i_kern, [1]).reshape(self.sub_no,-1)
        out_filters = torch.vstack((e_kern_out, i_kern_out))

        return V, out_filters, C_syn_e, C_syn_i

