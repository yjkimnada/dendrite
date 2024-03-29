import torch
from torch import nn
from torch.nn import functional as F

class Sub_Cos_GLM_Multilayer(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, hid_no, device):
        super().__init__()

        self.T_no = T_no
        self.sub_no = C_syn_e.shape[0]
        self.E_no = C_syn_e.shape[1]
        self.I_no = C_syn_i.shape[1]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        self.hid_no = hid_no

        self.E_scale = nn.Parameter(torch.zeros(self.E_no))
        self.I_scale = nn.Parameter(torch.zeros(self.I_no))
        
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
        
        self.W_e_layer1 = nn.Parameter(torch.randn(self.sub_no*hid_no , self.cos_basis_no)*0.01)        
        self.W_i_layer1 = nn.Parameter(torch.randn(self.sub_no*hid_no , self.cos_basis_no)*0.01)
        self.b_layer1 = nn.Parameter(torch.zeros(self.sub_no*self.hid_no))
        
        self.W_layer2 = nn.Parameter(torch.randn((self.sub_no-1)//2, hid_no*2)*(0.01))
        self.b_layer2 = nn.Parameter(torch.zeros((self.sub_no-1)//2))
        
        self.W_layer3 = nn.Parameter(torch.ones((self.sub_no-1)//2+hid_no, 1)*(-1))
        #self.b_layer3 = nn.Parameter(torch.zeros(1))

        self.V_o = nn.Parameter(torch.zeros(1))

    def forward(self, S_e, S_i):
        # S is (batch, T, E)
        T_data = S_e.shape[1]
        batch = S_e.shape[0]

        S_e = S_e * torch.exp(self.E_scale.reshape(1,1,-1))
        S_i = S_i * torch.exp(self.I_scale.reshape(1,1,-1))
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
        layer1_e_kern = torch.flip(layer1_e_kern, [1]).unsqueeze(1)
        layer1_i_kern = torch.flip(layer1_i_kern, [1]).unsqueeze(1)
        
        layer1_e_conv = F.conv1d(pad_syn_e, layer1_e_kern, groups=self.sub_no)
        layer1_i_conv = F.conv1d(pad_syn_i, layer1_i_kern, groups=self.sub_no)
        layer1_out = torch.tanh(layer1_e_conv + layer1_i_conv + self.b_layer1.reshape(1,-1,1))
        
        layer2_conv = F.conv1d(layer1_out[:,:-self.hid_no], self.W_layer2.unsqueeze(2) , groups=self.sub_no//2)
        layer2_out = torch.tanh(layer2_conv + self.b_layer2.reshape(1,-1,1))
        #layer2_out = layer2_conv + self.b_layer2.reshape(1,-1,1)
        
        layer2_out = torch.cat((layer2_out, layer1_out[:,-self.hid_no:].reshape(-1,self.hid_no,T_data)), 1)
        sub_out = F.conv1d(layer2_out, torch.exp(self.W_layer3).unsqueeze(-1), groups=(self.sub_no-1)//2+self.hid_no).permute(0,2,1)
        #final = torch.tanh(torch.sum(sub_out, -1) + self.b_layer3) + self.V_o
        final = torch.sum(sub_out, -1) + self.V_o
        layer1_in = layer1_e_conv + layer1_i_conv + self.b_layer1.reshape(1,-1,1)

        return final, sub_out, layer1_in
