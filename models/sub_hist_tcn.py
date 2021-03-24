import torch
from torch import nn
from torch.nn import functional as F

class Sub_Hist_TCN(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, hid_no, two_nonlin, device):
        super().__init__()

        self.T_no = T_no
        self.sub_no = C_syn_e.shape[0]
        self.E_no = C_syn_e.shape[1]
        self.I_no = C_syn_i.shape[1]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        self.hid_no = hid_no
        self.two_nonlin = two_nonlin

        self.E_scale = nn.Parameter(torch.zeros(self.E_no))
        self.I_scale = nn.Parameter(torch.zeros(self.I_no))
        
        self.W_e_layer1 = nn.Parameter(torch.randn(self.sub_no*self.hid_no, self.T_no)*0.01)
        #self.W_i_layer1 = nn.Parameter(torch.randn(self.sub_no*self.hid_no, self.T_no)*0.01)
        self.W_layer2 = nn.Parameter(torch.ones(self.sub_no, self.hid_no)*(-1))
        self.b_layer1 = nn.Parameter(torch.zeros(self.sub_no*self.hid_no))
        
        self.W_hist = nn.Parameter(torch.zeros(self.sub_no*self.hid_no, self.T_no))
        
        self.W_sub = nn.Parameter(torch.zeros(self.sub_no))
        self.V_o = nn.Parameter(torch.zeros(1))

    def forward(self, S_e, S_i):
        # S is (batch, T, E)
        T_data = S_e.shape[1]
        batch = S_e.shape[0]

        S_e = S_e * torch.exp(self.E_scale.reshape(1,1,-1))
        #S_i = S_i * torch.exp(self.I_scale.reshape(1,1,-1)) * (-1)
        syn_e = torch.matmul(S_e, self.C_syn_e.T.unsqueeze(0))
        #syn_i = torch.matmul(S_i, self.C_syn_i.T.unsqueeze(0))

        pad_syn_e = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        #pad_syn_i = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        pad_syn_e[:,-T_data:] = pad_syn_e[:,-T_data:] + syn_e
        #pad_syn_i[:,-T_data:] = pad_syn_i[:,-T_data:] + syn_i
        pad_syn_e = pad_syn_e.permute(0,2,1)
        #pad_syn_i = pad_syn_i.permute(0,2,1)
        
        layer1_e_conv = F.conv1d(pad_syn_e, torch.flip(self.W_e_layer1, [1]).unsqueeze(1), groups=self.sub_no)
        #layer1_i_conv = F.conv1d(pad_syn_i, torch.flip(self.W_i_layer1.unsqueeze(1), [1]), groups=self.sub_no) #(batch,sub*H,T)
        
        sub_out_raw = torch.zeros(batch, T_data + self.T_no, self.sub_no*self.hid_no).to(self.device)
        hist_kern = torch.flip(self.W_hist, [1]) #(sub*H,T)
        
        for t in range(T_data):
            sub_hist = sub_out_raw[:,t:t+self.T_no,:].clone() #(batch, T, sub*H)
            sub_hist_in = torch.sum(sub_hist * hist_kern.T.unsqueeze(0), 1) #(batch, sub*H)
            
            #sub_in = sub_hist_in + layer1_e_conv[:,:,t] + layer1_i_conv[:,:,t] + self.b_layer1.unsqueeze(0)
            sub_in = sub_hist_in + layer1_e_conv[:,:,t] + self.b_layer1.unsqueeze(0)
            sub_out_raw[:,t+self.T_no,:] = sub_out_raw[:,t+self.T_no,:] + torch.tanh(sub_in)
        
        sub_out = F.conv1d(sub_out_raw[:,self.T_no:,:].permute(0,2,1),
                           torch.exp(self.W_layer2).unsqueeze(-1), groups=self.sub_no).permute(0,2,1)
        final = torch.sum(sub_out, -1) + self.V_o

        return final


