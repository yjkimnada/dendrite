import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt

class Sub_Sparse_Clust_Cos_GLM(nn.Module):
    def __init__(self, sub_no, E_no, I_no, T_no, hid_no, device):
        super().__init__()

        self.T_no = T_no
        self.sub_no = sub_no
        self.E_no = E_no
        self.I_no = I_no
        self.device = device
        self.hid_no = hid_no

        self.E_scale = nn.Parameter(torch.zeros(self.E_no))
        self.I_scale = nn.Parameter(torch.zeros(self.I_no))
        
        self.cos_basis_no = 24
        self.scale = 6
        self.shift = 1
        
        self.kern_basis = torch.zeros(self.cos_basis_no, T_no).to(device)
        for i in range(self.cos_basis_no):
            phi = 1.5707963267948966*i
            xmin = phi - 3.141592653589793
            xmax = phi + 3.141592653589793

            x_in = torch.arange(0, T_no, 1)
            raw_cos = self.scale  * torch.log(x_in + self.shift + 1e-7)

            basis = 0.5*torch.cos(raw_cos - phi) + 0.5
            basis[raw_cos < xmin] = 0.0
            basis[raw_cos > xmax] = 0.0
            self.kern_basis[i] = basis
        
        self.W_e_layer1 = nn.Parameter(torch.randn(self.sub_no*hid_no , self.cos_basis_no)*0.01)
        self.W_i_layer1 = nn.Parameter(torch.randn(self.sub_no*hid_no , self.cos_basis_no)*0.01)
        self.W_layer2 = nn.Parameter(torch.ones(self.sub_no, self.hid_no)*(-1))
        self.b_layer1 = nn.Parameter(torch.zeros(self.sub_no*self.hid_no))
        
        self.C_syn_e_raw = nn.Parameter(torch.randn(self.sub_no, self.E_no)*0.1+1/self.sub_no)
        self.C_syn_i_raw = nn.Parameter(torch.randn(self.sub_no, self.I_no)*0.1+1/self.sub_no)
        
        self.V_o = nn.Parameter(torch.zeros(1))
        
    
    def sparsemax(self, v, z=1):
        v_sorted, _ = torch.sort(v, dim=0, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0) - z
        ind = torch.arange(1, 1 + len(v)).float().to(v.device)
        cond = v_sorted - cssv / ind > 0
        rho = ind.masked_select(cond)[-1]
        tau = cssv.masked_select(cond)[-1] / rho
        w = torch.clamp(v - tau, min=0)
        return w / z


    def sparsestmax(self, v, rad_in=0, u_in=None):
        w = self.sparsemax(v)
        if max(w) - min(w) == 1:
            return w
        ind = torch.tensor(w > 0).float()
        u = ind / torch.sum(ind)
        if u_in is None:
            rad = rad_in
        else:
            rad = sqrt(rad_in ** 2 - torch.sum((u - u_in) ** 2))
        distance = torch.norm(w - u)
        if distance >= rad:
            return w
        p = rad * (w - u) / distance + u
        if min(p) < 0:
            return self.sparsestmax(p, rad, u)
        return p.clamp_(min=0, max=1)

    def forward(self, S_e, S_i, rad ,test):
        # S is (batch, T, E)
        T_data = S_e.shape[1]
        batch = S_e.shape[0]
        
        C_syn_e = torch.zeros_like(self.C_syn_e_raw).to(self.device)
        C_syn_i = torch.zeros_like(self.C_syn_i_raw).to(self.device)
        if test == False:
            for e in range(self.E_no):
                part_e = self.sparsestmax(self.C_syn_e_raw[:,e], rad)
                C_syn_e[:,e] = C_syn_e[:,e] + part_e
            for i in range(self.I_no):
                part_i = self.sparsestmax(self.C_syn_i_raw[:,i], rad)
                C_syn_i[:,i] = C_syn_i[:,i] + part_i
        elif test == True:
            for e in range(self.E_no):
                e_idx = torch.argmax(self.C_syn_e_raw[:,e])
                C_syn_e[e_idx,e] = 1
            for i in range(self.I_no):
                i_idx = torch.argmax(self.C_syn_i_raw[:,i])
                C_syn_i[i_idx,i] = 1

        S_e = S_e * torch.exp(self.E_scale.reshape(1,1,-1))
        S_i = S_i * torch.exp(self.I_scale.reshape(1,1,-1))
        syn_e = torch.matmul(S_e, C_syn_e.T.unsqueeze(0))
        syn_i = torch.matmul(S_i, C_syn_i.T.unsqueeze(0))

        pad_syn_e = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        pad_syn_e[:,-T_data:] = pad_syn_e[:,-T_data:] + syn_e
        pad_syn_i[:,-T_data:] = pad_syn_i[:,-T_data:] + syn_i
        pad_syn_e = pad_syn_e.permute(0,2,1)
        pad_syn_i = pad_syn_i.permute(0,2,1)
        
        layer1_e_kern = torch.matmul(self.W_e_layer1, self.kern_basis) # (sub*H, T_no)
        layer1_i_kern = torch.matmul(self.W_i_layer1, self.kern_basis) # (sub*H, T_no)
        layer1_e_kern = torch.flip(layer1_e_kern, [1]).unsqueeze(1)
        layer1_i_kern = torch.flip(layer1_i_kern, [1]).unsqueeze(1)
        
        layer1_e_conv = F.conv1d(pad_syn_e, layer1_e_kern, groups=self.sub_no)
        layer1_i_conv = F.conv1d(pad_syn_i, layer1_i_kern, groups=self.sub_no)
        layer1_out = torch.tanh(layer1_e_conv + layer1_i_conv + self.b_layer1.reshape(1,-1,1))
        #layer1_out = torch.tanh(layer1_e_conv + self.b_layer1.reshape(1,-1,1))
        
        sub_out = F.conv1d(layer1_out, torch.exp(self.W_layer2).unsqueeze(-1), groups=self.sub_no).permute(0,2,1)
        final = torch.sum(sub_out, -1) + self.V_o

        return final, C_syn_e, C_syn_i
        

