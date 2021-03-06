import torch
from torch import nn
from torch.nn import functional as F

class TCN(nn.Module):
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

        tcn = []
        for l in range(layer_no):
            if l == 0:
                tcn.append(nn.Conv1d(self.sub_no, hid_no, T_no))
                tcn.append(nn.LeakyReLU())
            elif l == layer_no-1:
                tcn.append(nn.Conv1d(hid_no, 1, 1))
            else:
                tcn.append(nn.Conv1d(hid_no, hid_no, 1))
                tcn.append(nn.LeakyReLU())
        self.tcn = nn.Sequential(*tcn)

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

        syn_in = pad_syn_e + pad_syn_i
        final = self.tcn(syn_in).squeeze(1)

        return final