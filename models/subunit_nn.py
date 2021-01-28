import torch
from torch import nn
from torch.nn import functional as F

class Subunit_NN(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, sub_no, T_no, H_no, device):
        super().__init__()

        self.T_no = T_no
        #self.sub_no = C_syn_e.shape[0]
        self.sub_no = sub_no
        
        #self.C_syn_e_raw = nn.Parameter(torch.ones(self.sub_no, 2000), requires_grad=True)
        #self.C_syn_i_raw = nn.Parameter(torch.ones(self.sub_no, 200), requires_grad=True)
        
        self.E_no = C_syn_e.shape[1]
        self.I_no = C_syn_i.shape[1]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        self.H_no = H_no
        
        ### Cosine Basis ###
        self.cos_basis_no = 20
        self.cos_shift = 1
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
        
        self.W_conv = nn.Parameter(torch.randn(self.H_no*self.sub_no,2,self.cos_basis_no)*0.1 , requires_grad=True)

        self.ff = []
        for h in range(self.sub_no):
            self.ff.append(nn.Sequential(nn.Linear(self.H_no, self.H_no), nn.LeakyReLU()))
        self.ff = nn.ModuleList(self.ff)
        
        self.ff2 = []
        for h in range(self.sub_no):
            self.ff2.append(nn.Sequential(nn.Linear(self.H_no, self.H_no), nn.LeakyReLU()))
        self.ff2 = nn.ModuleList(self.ff2)

        self.root = nn.Sequential(nn.Linear(self.sub_no*self.H_no, 1))
        
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, S_e, S_i, temp):
        T_data = S_e.shape[0]
        
        C_syn_e = self.C_syn_e
        C_syn_i = self.C_syn_i
        
        S_e_group = torch.matmul(S_e, C_syn_e.T)
        S_i_group = torch.matmul(S_i, C_syn_i.T)

        S = torch.zeros(T_data+self.T_no-1, 2*self.sub_no).to(self.device)
        for s in range(self.sub_no):
            S[-T_data:,2*s] = S[-T_data:,2*s] + S_e_group[:,s]
            S[-T_data:,2*s+1] = S[-T_data:,2*s+1] + S_i_group[:,s]
        S = S.T.unsqueeze(0)
        
        conv_kern = torch.matmul(self.W_conv, self.cos_basis)
        conv_kern = torch.flip(conv_kern, [2])
        conv_out = F.conv1d(S, conv_kern, groups=self.sub_no).squeeze(0).reshape(self.sub_no, self.H_no, -1)
        conv_out = F.leaky_relu(conv_out)
        
        sub_out = torch.zeros(T_data, self.sub_no*self.H_no).to(self.device)

        for s in range(self.sub_no):
            part_conv = conv_out[s].T
            ff_out = self.ff[s](part_conv)
            ff2_out = self.ff2[s](ff_out)
            
            sub_out[:,s*self.H_no:(s+1)*self.H_no] = sub_out[:,s*self.H_no:(s+1)*self.H_no] + ff2_out
        
        root_out = self.root(sub_out)
        
        V = root_out.flatten() + self.V_o

        return V, conv_kern
