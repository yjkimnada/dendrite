import torch
from torch import nn
from torch.nn import functional as F

class NN_Encoder(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, layer_no, device):
        super().__init__()

        self.T_no = 50
        self.sub_no = C_syn_e.shape[0]
        #self.C_syn_e = C_syn_e
        #self.C_syn_i = C_syn_i
        self.device = device
        self.layer_no = layer_no

        ### TCN ###
        modules = []
        for i in range(self.layer_no):
            if i == 0:
                modules.append(nn.Conv1d(in_channels=1,
                                        out_channels=5,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
                modules.append(nn.LeakyReLU())
            if i == self.layer_no-1:
                modules.append(nn.Conv1d(in_channels=5,
                                        out_channels=1,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
            else:
                modules.append(nn.Conv1d(in_channels=5,
                                        out_channels=5,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
                modules.append(nn.LeakyReLU())
        self.conv_list = nn.Sequential(*modules)

    def forward(self, V, S_e, S_i):
        T_data = V.shape[0]

        #syn_e = torch.matmul(S_e, self.C_syn_e[:].T)
        #syn_i = torch.matmul(S_i, self.C_syn_i[:].T)

        nn = self.conv_list(V.reshape(1,1,-1)).flatten()

        Z_out = torch.sigmoid(nn)
        spk_out_raw = torch.zeros(T_data, 2).to(self.device)
        spk_out_raw[:,0] = spk_out_raw[:,0] + nn
        
        eps = 1e-8
        temp=0.025
        u = torch.rand_like(spk_out_raw)
        g = - torch.log(- torch.log(u + eps) + eps)

        spk_out_pad = F.softmax((spk_out_raw + g) / temp, dim=1)
        spk_out = spk_out_pad[:,0]

        return spk_out, Z_out
    
class NN_Decoder(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.T_no = 50
        self.C_den = C_den
        self.sub_no = C_syn_e.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        
        ### EXP BASIS ###
        self.basis_no = 21
        self.T_max = self.T_no*2+1
        self.sigma = 20
       
        self.basis = torch.zeros(self.basis_no, self.T_max).to(self.device)
        self.points = torch.arange(0, 5*self.basis_no, 5)
        for i in range(self.basis_no):
            point = self.points[i]
            raw = torch.arange(self.T_max).to(self.device)
            part = torch.exp(-(raw - point)**2/self.sigma)
            self.basis[i] = part
        
        ### BASIS WEIGHTS ###
        self.hid_no = 10
        
        self.weight1 = nn.Parameter(torch.randn(self.sub_no, 2, self.basis_no)*0.01 , requires_grad=True)
        #self.weight2 = nn.Parameter(torch.randn(self.hid_no, self.sub_no, self.basis_no)*0.01 , requires_grad=True)
        #self.weight3 = nn.Parameter(torch.randn(self.hid_no, self.hid_no, self.basis_no)*0.01 , requires_grad=True)
        self.weight4 = nn.Parameter(torch.randn(1, self.sub_no, self.basis_no)*0.01 , requires_grad=True)
        
        self.Theta = nn.Parameter(torch.randn(1)*0.01 , requires_grad=True)
        
        ### Spike Parameters ###
        self.W_spk = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Delta_spk = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Tau_spk = nn.Parameter(torch.ones(1)*3, requires_grad=True)
        
        ### C_syn Parameters ###
        #self.C_syn_e_raw = nn.Parameter(torch.ones(self.sub_no, 2000) , requires_grad=True)
        #self.C_syn_i_raw = nn.Parameter(torch.ones(self.sub_no, 200) , requires_grad=True)
        
    def train_forward(self, S_e, S_i, Z):
        T_data = S_e.shape[0]
        
        """
        eps = 1e-8
        temp=0.025
        u_e = torch.rand_like(self.C_syn_e_raw)
        u_i = torch.rand_like(self.C_syn_i_raw)
        g_e = - torch.log(- torch.log(u_e + eps) + eps)
        g_i = - torch.log(- torch.log(u_i + eps) + eps)
        C_syn_e = F.softmax((self.C_syn_e_raw + g_e) / temp, dim=0)
        C_syn_i = F.softmax((self.C_syn_i_raw + g_i) / temp, dim=0)
        """
        
        kern1 = torch.matmul(self.weight1 , self.basis)
        #kern2 = torch.matmul(self.weight2 , self.basis)
        #kern3 = torch.matmul(self.weight3 , self.basis)
        kern4 = torch.matmul(self.weight4 , self.basis)
        
        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)
        
        syn_in = torch.zeros(T_data, 2*self.sub_no).to(self.device)
        for i in range(self.sub_no):
            syn_in[:,2*i] = syn_in[:,2*i] + syn_e[:,i]
            syn_in[:,2*i+1] = syn_in[:,2*i+1] + syn_i[:,i]
        syn_in = syn_in.T.unsqueeze(0)
        
        out1 = torch.tanh(F.conv1d(syn_in, kern1, padding=self.T_no, groups=self.sub_no))
        #out2 = torch.tanh(F.conv1d(out1, kern2, padding=self.T_no, groups=1))
        #out3 = torch.tanh(F.conv1d(out2, kern3, padding=self.T_no))
        out4 = F.conv1d(out1, kern4, padding=self.T_no).flatten()
                
        prob_out = torch.sigmoid(out4 + self.Theta)
        
        pad_Z = torch.zeros(T_data + self.T_no).to(self.device)
        pad_Z[-T_data:] = pad_Z[-T_data:] + Z
        pad_Z = pad_Z.reshape(1,1,-1)
        
        t = torch.arange(self.T_no).to(self.device)
        t_tau = t / self.Tau_spk**2
        spk_kern = t_tau * torch.exp(-t_tau) * self.W_spk**2
        spk_kern = torch.flip(spk_kern, [0]).reshape(1,1,-1)
        spk_filt = F.conv1d(pad_Z, spk_kern).flatten()[:-1]
        
        out_filters = torch.vstack((torch.flip(kern1.reshape(-1,self.T_max), [1]),
                      torch.flip(kern4.reshape(-1,self.T_max), [1])))
        
        return spk_filt, prob_out, out_filters
    
    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        
        """
        eps = 1e-8
        temp=0.025
        u_e = torch.rand_like(self.C_syn_e_raw)
        u_i = torch.rand_like(self.C_syn_i_raw)
        g_e = - torch.log(- torch.log(u_e + eps) + eps)
        g_i = - torch.log(- torch.log(u_i + eps) + eps)
        C_syn_e = F.softmax((self.C_syn_e_raw + g_e) / temp, dim=0)
        C_syn_i = F.softmax((self.C_syn_i_raw + g_i) / temp, dim=0)
        """
        
        kern1 = torch.matmul(self.weight1 , self.basis)
        #kern2 = torch.matmul(self.weight2 , self.basis)
        #kern3 = torch.matmul(self.weight3 , self.basis)
        kern4 = torch.matmul(self.weight4 , self.basis)
        
        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)
        syn_in = torch.zeros(T_data, 2*self.sub_no).to(self.device)
        for i in range(self.sub_no):
            syn_in[:,2*i] = syn_in[:,2*i] + syn_e[:,i]
            syn_in[:,2*i+1] = syn_in[:,2*i+1] + syn_i[:,i]
        syn_in = syn_in.T.unsqueeze(0)

        out1 = torch.tanh(F.conv1d(syn_in, kern1, padding=self.T_no, groups=self.sub_no))
        #out2 = torch.tanh(F.conv1d(out1, kern2, padding=self.T_no, groups=1))
        #out3 = torch.tanh(F.conv1d(out2, kern3, padding=self.T_no))
        out4 = F.conv1d(out1, kern4, padding=self.T_no).flatten()

        spk_pad = torch.zeros(T_data + self.T_no).to(self.device)
        prob_out = torch.sigmoid(out4 + self.Theta)
        spk_out = torch.bernoulli(torch.sigmoid(out4 + self.Theta))
        spk_pad[-T_data:] = spk_pad[-T_data:] + spk_out
        
        t = torch.arange(self.T_no).to(self.device)
        t_tau = t / self.Tau_spk**2
        spk_kern = t_tau * torch.exp(-t_tau) * self.W_spk**2
        spk_kern = torch.flip(spk_kern, [0]).reshape(1,1,-1)
        spk_filt = F.conv1d(spk_pad.reshape(1,1,-1), spk_kern).flatten()[:-1]
        
        out_filters = torch.vstack((torch.flip(kern1.reshape(-1,self.T_max), [1]),
                      torch.flip(kern4.reshape(-1,self.T_max), [1])))
        
        return spk_filt, prob_out, out_filters
        
    