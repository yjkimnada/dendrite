import torch
from torch import nn
from torch.nn import functional as F

class TCN_Multilayer(nn.Module):
    def __init__(self, T_no, in_no, layer_no, hid_no, device):
        super().__init__()
        
        self.T_no = T_no
        self.device = device
        self.hid_no = hid_no
        
        #self.conv = nn.Conv1d(in_no, hid_no, T_no)
        self.conv = [nn.Conv2d(1,hid_no,
                              kernel_size=[T_no,in_no],
                              padding=[T_no//2,0]),
                    nn.LeakyReLU()]
        self.conv = nn.Sequential(*self.conv)
        mlp = []
        for i in range(layer_no):
            if i == layer_no-1:
                mlp.append(nn.Linear(hid_no, 1))
            else:
                mlp.append(nn.Linear(hid_no, hid_no))
                mlp.append(nn.LeakyReLU())
                
        self.mlp = nn.Sequential(*mlp)

    def forward(self, S_e, S_i):
        T_data = S_e.shape[1]
        batch = S_e.shape[0]
        
        S_all = torch.cat([S_e, S_i], 2) #(b,T,syn)
        conv_out = self.conv(S_all.unsqueeze(1))
        conv_out = conv_out.squeeze(-1)
        conv_out = conv_out.permute([0,2,1])
        conv_out = conv_out.reshape(-1,self.hid_no)
        
        final = self.mlp(conv_out)
        final = torch.squeeze(final, -1)
        final = final.reshape(batch, T_data)

        return final


