import torch
from torch import nn
from torch.nn import functional as F

class Sub_Cos_TCN(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, layer_no, hid_no, device):
        super().__init__()

        self.T_no = T_no
        self.sub_no = C_syn_e.shape[0]
        self.E_no = C_syn_e.shape[1]
        self.I_no = C_syn_i.shape[1]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        self.hid_no = hid_no
        self.layer_no = layer_no

        self.E_scale = nn.Parameter(torch.zeros(self.E_no))
        self.I_scale = nn.Parameter(torch.zeros(self.I_no))
        
        self.cos_basis_no = 20
        self.scale = 5
        self.shift = 1
        
        self.kern_basis = torch.zeros(self.cos_basis_no, T_no).to(device)
        for i in range(self.cos_basis_no):
            phi = 1.5707963267948966*i
            xmin = phi - 3.141592653589793
            xmax = phi + 3.141592653589793

            x_in = torch.arange(0, T_no, 1)
            raw_cos = 3.5  * torch.log(x_in + 1 + 1e-7)

            basis = 0.5*torch.cos(raw_cos - phi) + 0.5
            basis[raw_cos < xmin] = 0.0
            basis[raw_cos > xmax] = 0.0
            self.kern_basis[i] = basis
        
        self.kern_weights = nn.Parameter(torch.randn(self.sub_no*hid_no , self.cos_basis_no)*0.01)
        self.sub_comb_weights = nn.Parameter(torch.zeros(self.sub_no*hid_no))

        self.V_o = nn.Parameter(torch.zeros(1))
        
        self.comb_sum_mat = torch.zeros(self.sub_no, self.sub_no*self.hid_no).to(device)
        for s in range(self.sub_no):
            self.comb_sum_mat[s, s*self.hid_no : (s+1)*self.hid_no] = 1

    def forward(self, S_e, S_i):
        # S is (batch, T, E)
        T_data = S_e.shape[1]
        batch = S_e.shape[0]

        S_e = S_e * torch.exp(self.E_scale.reshape(1,1,-1))
        S_i = S_i * torch.exp(self.I_scale.reshape(1,1,-1)) * (-1)
        syn_e = torch.matmul(S_e, self.C_syn_e.T.unsqueeze(0))
        syn_i = torch.matmul(S_i, self.C_syn_i.T.unsqueeze(0))

        pad_syn_e = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        pad_syn_e[:,-T_data:] = pad_syn_e[:,-T_data:] + syn_e
        pad_syn_i[:,-T_data:] = pad_syn_i[:,-T_data:] + syn_i
        pad_syn_e = pad_syn_e.permute(0,2,1)
        pad_syn_i = pad_syn_i.permute(0,2,1)

        syn_in = pad_syn_e + pad_syn_i
        
        kern = torch.matmul(self.kern_weights, self.kern_basis) # (sub*H, T_no)
        kern = torch.flip(kern, [1]).unsqueeze(1)
        
        kern_out = F.conv1d(syn_in, kern, groups=self.sub_no).permute(0,2,1) # (batch, T, sub*H)
        sub_comb = kern_out.reshape(-1, self.sub_no*self.hid_no) * torch.exp(self.sub_comb_weights).reshape(1,-1)
        
        sub_out = torch.matmul(sub_comb, self.comb_sum_mat.T).reshape(batch, T_data, self.sub_no)
        # SECOND NONLIN AT SUBUNIT OUTPUT ?!?!?!
        # sub_out = torch.tanh(sub_out) * MORE WEIGHTS?!
        
        final = torch.sum(sub_out, -1)

        return final


