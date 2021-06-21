import torch
from torch import nn
from torch.nn import functional as F

class Sub_Cos_TCN(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, hid_no, layer_no, device):
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
        
        ##############
        ##############
        self.cos_basis_no = 30
        self.scale = 7.5
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
            
        self.W_e_layer1 = nn.Parameter(torch.randn(self.hid_no*self.sub_no , self.cos_basis_no)*0.01)        
        self.W_i_layer1 = nn.Parameter(torch.randn(self.hid_no*self.sub_no , self.cos_basis_no)*0.01)
        self.b_layer1 = nn.Parameter(torch.zeros(self.hid_no))
        
        tcn = []
        for l in range(layer_no-1):
            if l == layer_no-2:
                tcn.append(nn.Conv1d(hid_no, 1, 1))
            else:
                tcn.append(nn.Conv1d(hid_no, hid_no, 1))
                tcn.append(nn.LeakyReLU())
        self.tcn = nn.Sequential(*tcn)
        ################
        ################
        

        self.V_o = nn.Parameter(torch.zeros(1))

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

        layer1_e_kern = torch.matmul(self.W_e_layer1, self.kern_basis) # (sub*H, T_no)
        layer1_i_kern = torch.matmul(self.W_i_layer1, self.kern_basis) # (sub*H, T_no)
        layer1_e_kern = torch.flip(layer1_e_kern, [1]).reshape(self.hid_no,self.sub_no,self.T_no)
        layer1_i_kern = torch.flip(layer1_i_kern, [1]).reshape(self.hid_no,self.sub_no,self.T_no)
        
        layer1_e_conv = F.conv1d(pad_syn_e, layer1_e_kern)
        layer1_i_conv = F.conv1d(pad_syn_i, layer1_i_kern)
        layer1_out = F.leaky_relu(layer1_e_conv + layer1_i_conv + self.b_layer1.reshape(1,-1,1))
            
        final = self.tcn(layer1_out).squeeze(1)

        return final