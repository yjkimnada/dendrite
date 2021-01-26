import torch
from torch import nn
from torch.nn import functional as F

class Subunit_NN(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, H_no, device):
        super().__init__()

        self.T_no = T_no
        #self.sub_no = C_syn_e.shape[0]
        self.sub_no = 5
        
        self.C_syn_e_raw = nn.Parameter(torch.ones(self.sub_no, 2000), requires_grad=True)
        self.C_syn_i_raw = nn.Parameter(torch.ones(self.sub_no, 200), requires_grad=True)
        
        self.E_no = C_syn_e.shape[1]
        self.I_no = C_syn_i.shape[1]
        #self.C_syn_e = C_syn_e
        #self.C_syn_i = C_syn_i
        self.device = device
        self.H_no = H_no

        self.conv = nn.Conv1d(in_channels=2*self.sub_no,
                            out_channels=self.H_no*self.sub_no,
                            kernel_size=self.T_no,
                            padding=self.T_no//2,
                            groups=self.sub_no)

        self.ff = []
        for h in range(self.sub_no):
            self.ff.append(nn.Sequential(nn.Linear(self.H_no, 1), nn.LeakyReLU()))
        self.ff = nn.ModuleList(self.ff)

        self.root = nn.Sequential(nn.Linear(self.sub_no, 1) , nn.LeakyReLU())
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, S_e, S_i, temp):
        T_data = S_e.shape[0]
        
        C_syn_e = F.softmax(self.C_syn_e_raw/temp, 0)
        C_syn_i = F.softmax(self.C_syn_i_raw/temp, 0)
        
        S_e_group = torch.matmul(S_e, C_syn_e.T)
        S_i_group = torch.matmul(S_i, C_syn_i.T)

        S = torch.zeros(T_data, 2*self.sub_no).to(self.device)
        for s in range(self.sub_no):
            S[:,2*s] = S[:,2*s] + S_e_group[:,s]
            S[:,2*s+1] = S[:,2*s+1] + S_i_group[:,s]
        S = S.T.unsqueeze(0)

        conv_out = self.conv(S)
        conv_out = conv_out.squeeze(0).reshape(self.sub_no, self.H_no, -1)
        sub_out = torch.zeros(T_data, self.sub_no).to(self.device)

        for s in range(self.sub_no):
            part_conv = conv_out[s].T
            ff_out = self.ff[s](part_conv)
            sub_out[:,s] = sub_out[:,s] + ff_out.flatten()

        root_out = self.root(sub_out).flatten() + self.V_o

        return root_out
