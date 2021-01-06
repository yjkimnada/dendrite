import torch
from torch import nn
from torch.nn import functional as F

class FullVAE_Hist_GLM(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        
        ### Cosine Basis ###
        self.cos_basis_no = 19
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
            
        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no, 2) , requires_grad=True)

        ### History Parameters ###
        self.W_hist = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no) , requires_grad=True)
        
        ### Spike Parameters ###
        #self.W_spk = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)
        self.W_spk = nn.Parameter(torch.ones(self.sub_no), requires_grad=True)
        self.Delta_spk = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        self.Tau_spk = nn.Parameter(torch.ones(self.sub_no)*3, requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        
        self.weight1 = nn.Parameter(torch.randn(self.sub_no, 6)*0.1, requires_grad=True)
        self.weight2 = nn.Parameter(torch.randn(6, 6)*0.1, requires_grad=True)
        self.weight3 = nn.Parameter(torch.randn(6, self.sub_no)*0.1, requires_grad=True)

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]
        
        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)
        
        full_e_kern = torch.matmul(self.W_syn[:,:,0], self.cos_basis)
        full_i_kern = torch.matmul(self.W_syn[:,:,1], self.cos_basis)
        
        full_e_kern = torch.flip(full_e_kern, [1])
        full_i_kern = torch.flip(full_i_kern, [1])
        full_e_kern = full_e_kern.unsqueeze(1)
        full_i_kern = full_i_kern.unsqueeze(1)

        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.reshape(1,self.sub_no,-1)
        pad_syn_i = pad_syn_i.T.reshape(1,self.sub_no,-1)
        
        filtered_e = F.conv1d(pad_syn_e, full_e_kern, padding=0, groups=self.sub_no).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, full_i_kern, padding=0,  groups=self.sub_no).squeeze(0).T
 
        syn = filtered_e + filtered_i
        
        out_filters = torch.vstack((torch.flip(full_e_kern.squeeze(1), [1]),
                                   torch.flip(full_i_kern.squeeze(1), [1]),
                                   ))
        
        return syn, out_filters

    def train_forward(self, S_e, S_i, Z):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i)
        sub_out = torch.zeros(T_data , self.sub_no).to(self.device)

        pad_Z = torch.zeros(T_data + self.T_no, self.sub_no).to(self.device)
        pad_Z[-T_data:] = pad_Z[-T_data:] + Z
        pad_Z = pad_Z.T.unsqueeze(0)

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [1]).unsqueeze(1)
        hist_filt = F.conv1d(pad_Z, hist_kern, groups=self.sub_no).squeeze(0).T[:-1]

        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t = t - self.Delta_spk.reshape(-1,1)
        t[t < 0.0] = 0.0
        t_tau = t / self.Tau_spk.reshape(-1,1)**2
        spk_kern = t_tau * torch.exp( -t_tau) * self.W_spk.reshape(-1,1)**2
        spk_kern = torch.flip(spk_kern, [1]).unsqueeze(1)
        spk_filt = F.conv1d(pad_Z, spk_kern, groups=self.sub_no).squeeze(0).T[:-1]

        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
                
            if torch.numel(leaf_idx) == 0:
                sub_in = syn[:,sub_idx] + self.Theta[sub_idx] + hist_filt[:,sub_idx]
                sub_out_1 = torch.matmul(F.leaky_relu(sub_in).reshape(-1,1) , self.weight1[sub_idx,:].reshape(1,-1))
                sub_out_2 = torch.matmul(F.leaky_relu(sub_out_1) , self.weight2)
                sub_out_3 = torch.matmul(F.leaky_relu(sub_out_2), self.weight3[:,sub_idx].reshape(-1,1)).flatten()
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.sigmoid(sub_out_3)

            else:
                sub_in = syn[:,sub_idx] + self.Theta[sub_idx] + hist_filt[:,sub_idx] + torch.sum(spk_filt[:,leaf_idx].reshape(T_data,-1), 1)
                sub_out_1 = torch.matmul(F.leaky_relu(sub_in).reshape(-1,1) , self.weight1[sub_idx,:].reshape(1,-1))
                sub_out_2 = torch.matmul(F.leaky_relu(sub_out_1) , self.weight2)
                sub_out_3 = torch.matmul(F.leaky_relu(sub_out_2), self.weight3[:,sub_idx].reshape(-1,1)).flatten()
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.sigmoid(sub_out_3)

        final_V = spk_filt[:,0] + self.V_o
        final_Z = sub_out

        out_filters = torch.vstack((syn_filters,
                                torch.flip(hist_kern.squeeze(1) , [1]),
                                torch.flip(spk_kern.squeeze(1) , [1])))

        return final_V, final_Z, out_filters

    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i)

        prob_out = torch.zeros(T_data, self.sub_no).to(self.device)
        spk_out = torch.zeros(T_data + self.T_no, self.sub_no).to(self.device)
        V_out = torch.zeros(T_data, self.sub_no).to(self.device)

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [1])

        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t = t - self.Delta_spk.reshape(-1,1)
        t[t < 0.0] = 0.0
        t_tau = t / self.Tau_spk.reshape(-1,1)**2
        spk_kern = t_tau * torch.exp( -t_tau) * self.W_spk.reshape(-1,1)**2
        spk_kern = torch.flip(spk_kern, [1])
        

        for t in range(T_data):
            spk_hist = spk_out[t:t+self.T_no].clone()
            hist_filt = torch.sum(spk_hist.T * hist_kern , 1)
            spk_filt_raw = torch.sum(spk_hist.T * spk_kern, 1)
            spk_filt = torch.matmul(self.C_den, spk_filt_raw)
            
            sub_in = syn[t] + self.Theta + hist_filt + spk_filt
            sub_out_1 = F.leaky_relu(sub_in).reshape(-1,1) * self.weight1
            sub_out_2 = torch.matmul(F.leaky_relu(sub_out_1) , self.weight2)
            sub_out_3 = torch.sum(F.leaky_relu(sub_out_2) * self.weight3.T , 1)
            prob_out[t] = prob_out[t] + torch.sigmoid(sub_out_3)
            spk_out[t+self.T_no] = spk_out[t+self.T_no] + torch.bernoulli(torch.sigmoid(sub_out_3))
            V_out[t] = V_out[t] + spk_filt_raw

        final_V = V_out[:,0] + self.V_o
        final_spk = spk_out[self.T_no:]
        final_prob = prob_out
       
        out_filters = torch.vstack((syn_filters,
                                torch.flip(hist_kern , [1]),
                                torch.flip(spk_kern , [1])))

        return final_V, final_spk, final_prob, out_filters

class NN_Encoder(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, layer_no, device):
        super().__init__()

        self.T_no = 50
        self.sub_no = C_syn_e.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        self.layer_no = layer_no

        ### Cosine Basis ###
        self.cos_basis_no = 19
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

        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no, 2) , requires_grad=True)
        
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### TCN ###
        modules = []
        for i in range(self.layer_no):
            if i == 0:
                modules.append(nn.Conv1d(in_channels=1,
                                        out_channels=self.sub_no,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
                modules.append(nn.LeakyReLU())
            if i == self.layer_no-1:
                modules.append(nn.Conv1d(in_channels=self.sub_no,
                                        out_channels=self.sub_no,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
            else:
                modules.append(nn.Conv1d(in_channels=self.sub_no,
                                        out_channels=self.sub_no,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
                modules.append(nn.LeakyReLU())
        self.conv_list = nn.Sequential(*modules)

    def forward(self, V, S_e, S_i):
        T_data = V.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e[:].T)
        syn_i = torch.matmul(S_i, self.C_syn_i[:].T)

        e_kern = torch.matmul(self.W_syn[:,:,0], self.cos_basis)
        i_kern = torch.matmul(self.W_syn[:,:,1], self.cos_basis)
        e_kern = torch.flip(e_kern, [1]).unsqueeze(1)
        i_kern = torch.flip(i_kern, [1]).unsqueeze(1)

        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.reshape(1,self.sub_no,-1)
        pad_syn_i = pad_syn_i.T.reshape(1,self.sub_no,-1)

        filtered_e = F.conv1d(pad_syn_e, e_kern, groups=self.sub_no).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, i_kern, groups=self.sub_no).squeeze(0).T

        syn = filtered_e + filtered_i 
        nn = self.conv_list(V.reshape(1,1,-1)).squeeze(0).T

        prob_in = nn + syn + self.Theta
        prob_out = torch.sigmoid(prob_in) 

        spk_out_raw = torch.zeros(T_data, self.sub_no, 2).to(self.device)
        spk_out_raw[:,:,0] = spk_out_raw[:,:,0] + prob_in

        eps = 1e-8
        temp=0.025
        u = torch.rand_like(spk_out_raw)
        g = - torch.log(- torch.log(u + eps) + eps)
        spk_out_pad = F.softmax((spk_out_raw + g) / temp, dim=2)
        spk_out = spk_out_pad[:,:,0]

        return spk_out, prob_out