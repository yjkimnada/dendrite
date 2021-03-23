import torch
from torch import nn
from torch.nn import functional as F

class GLM(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.E_no = C_syn_e.shape[1]
        self.I_no = C_syn_i.shape[1]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device

        ### Kernel Parameters ###
        self.W_syn = nn.Parameter(torch.ones(self.sub_no, 2) * (-3))
        self.Tau_syn = nn.Parameter(torch.ones(self.sub_no, 2) * (1))
        self.Delta_syn = nn.Parameter(torch.ones(self.sub_no, 2) * (-2))

        ### Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.zeros(self.sub_no))

        ### Output Parameters ###
        self.V_o = nn.Parameter(torch.zeros(1))
        self.Theta = nn.Parameter(torch.zeros(self.sub_no))

    def forward(self, S_e, S_i):
        # S is (batch, T, E)
        T_data = S_e.shape[1]
        batch = S_e.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e.T.unsqueeze(0))
        syn_i = torch.matmul(S_i, self.C_syn_i.T.unsqueeze(0))
        pad_syn_e = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        pad_syn_e[:,-T_data:] = pad_syn_e[:,-T_data:] + syn_e
        pad_syn_i[:,-T_data:] = pad_syn_i[:,-T_data:] + syn_i
        pad_syn_e = pad_syn_e.permute(0,2,1)
        pad_syn_i = pad_syn_i.permute(0,2,1)

        t_raw = torch.arange(self.T_no).repeat(self.sub_no, 1).to(self.device)
        t_e = t_raw - torch.exp(self.Delta_syn[:,0]).reshape(-1,1)
        t_i = t_raw - torch.exp(self.Delta_syn[:,1]).reshape(-1,1)
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0

        tau_e = torch.exp(self.Tau_syn[:,0]).reshape(-1,1)
        tau_i = torch.exp(self.Tau_syn[:,1]).reshape(-1,1)
        t_e_tau = t_e / tau_e
        t_i_tau = t_i / tau_i

        e_kern = t_e_tau * torch.exp(-t_e_tau) * torch.exp(self.W_syn[:,0]).reshape(-1,1)
        i_kern = t_i_tau * torch.exp(-t_i_tau) * torch.exp(self.W_syn[:,1]).reshape(-1,1)
        e_kern = torch.flip(e_kern, [1]).unsqueeze(1)
        i_kern = torch.flip(i_kern, [1]).unsqueeze(1)

        filt_e = F.conv1d(pad_syn_e, e_kern, groups=self.sub_no).permute(0,2,1)
        filt_i = F.conv1d(pad_syn_i, i_kern, groups=self.sub_no).permute(0,2,1)
        syn_in = filt_e + filt_i
        
        sub_out = torch.zeros(batch, T_data, self.sub_no).to(self.device)
        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                nonlin_out = torch.tanh(syn_in[:,:,sub_idx] + self.Theta[sub_idx])
                sub_out[:,:,sub_idx] = sub_out[:,:,sub_idx] + nonlin_out

            else:
                leaf_in = sub_out[:,:,leaf_idx] * torch.exp(self.W_sub[leaf_idx]).reshape(1,1,-1)
                nonlin_in = syn_in[:,:,sub_idx] + torch.sum(leaf_in, -1) + self.Theta[sub_idx]
                nonlin_out = torch.tanh(nonlin_in)
                sub_out[:,:,sub_idx] = sub_out[:,:,sub_idx] + nonlin_out

        final = sub_out[:,:,0] * torch.exp(self.W_sub[0]) + self.V_o
        
        return final