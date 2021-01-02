import torch
from torch import nn
from torch.nn import functional as F

class VAE_Hist_GLM(nn.Module):
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

        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no)*0.1 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]

        e_kern = torch.matmul(self.W_syn[:,:,0], self.cos_basis)
        i_kern = torch.matmul(self.W_syn[:,:,1], self.cos_basis)
        e_kern = torch.flip(e_kern, [1]).unsqueeze(1)
        i_kern = torch.flip(i_kern, [1]).unsqueeze(1)

        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)

        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.reshape(1,self.sub_no,-1)
        pad_syn_i = pad_syn_i.T.reshape(1,self.sub_no,-1)

        filtered_e = F.conv1d(pad_syn_e, e_kern, groups=self.sub_no).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, i_kern, groups=self.sub_no).squeeze(0).T

        syn = filtered_e + filtered_i 

        out_filters = torch.vstack((torch.flip(e_kern.squeeze(1), [1]),
                                   torch.flip(i_kern.squeeze(1), [1])))
        
        return syn, out_filters

    def train_forward(self, S_e, S_i, V):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i)
        ns_out = torch.zeros(T_data , self.sub_no).to(self.device)

        pad_V = torch.zeros(T_data + self.T_no, self.sub_no).to(self.device)
        pad_V[-T_data:] = pad_V[-T_data:] + V
        pad_V = pad_V.T.unsqueeze(0)

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [1]).unsqueeze(1)
        hist_filt = F.conv1d(pad_V, hist_kern, groups=self.sub_no).squeeze(0).T[:-1]

        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
                
            if torch.numel(leaf_idx) == 0:
                ns_in = syn[:,sub_idx] + self.Theta[sub_idx] + hist_filt[:,sub_idx]
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)

            else:
                ns_in = syn[:,sub_idx] + self.Theta[sub_idx] + hist_filt[:,sub_idx] + torch.sum(ns_out[:,leaf_idx] * self.W_sub[leaf_idx]**2, 1)
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)

        final_V = ns_out[:,0]*self.W_sub[0]**2 + self.V_o
        hist_out = torch.flip(hist_kern.squeeze(1), [1])
        out_filters = torch.vstack((syn_filters, hist_out))

        return final_V, ns_out[:,1:] , out_filters

    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i)
        ns_out = torch.zeros(T_data + self.T_no , self.sub_no).to(self.device)

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [1])

        for t in range(T_data):
            hist_filt = torch.sum(ns_out[t:t+self.T_no].T.clone() * hist_kern , 1)
            ns_prop = torch.matmul(ns_out[t] * self.W_sub**2, self.C_den.T)
            ns_in = syn[t] + hist_filt + ns_prop + self.Theta
            ns_out[t+self.T_no] = ns_out[t+self.T_no] + torch.tanh(ns_in)

        final_V = ns_out[self.T_no:,0]*self.W_sub[0]**2 + self.V_o
        hist_out = torch.flip(hist_kern.squeeze(1), [1])
        out_filters = torch.vstack((syn_filters, hist_out))

        return final_V, out_filters

class NN_Encoder(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, layer_no, device):
        super().__init__()

        self.T_no = T_no
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
        self.W_syn = nn.Parameter(torch.zeros(self.sub_no-1, self.cos_basis_no, 2) , requires_grad=True)

        ### TCN ###
        modules = []
        for i in range(self.layer_no):
            if i == 0:
                modules.append(nn.Conv1d(in_channels=1,
                                        out_channels=self.sub_no-1,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=1))
                modules.append(nn.LeakyReLU())
            if i == self.layer_no-1:
                modules.append(nn.Conv1d(in_channels=self.sub_no-1,
                                        out_channels=self.sub_no-1,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=self.sub_no-1))
            else:
                modules.append(nn.Conv1d(in_channels=self.sub_no-1,
                                        out_channels=self.sub_no-1,
                                        kernel_size=self.T_no*2+1,
                                        padding=self.T_no,
                                        groups=self.sub_no-1))
                modules.append(nn.LeakyReLU())
        self.conv_list = nn.Sequential(*modules)

    def forward(self, V, S_e, S_i):
        T_data = V.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e[1:].T)
        syn_i = torch.matmul(S_i, self.C_syn_i[1:].T)

        e_kern = torch.matmul(self.W_syn[:,:,0], self.cos_basis)
        i_kern = torch.matmul(self.W_syn[:,:,1], self.cos_basis)
        e_kern = torch.flip(e_kern, [1]).unsqueeze(1)
        i_kern = torch.flip(i_kern, [1]).unsqueeze(1)

        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.sub_no-1).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.sub_no-1).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.reshape(1,self.sub_no-1,-1)
        pad_syn_i = pad_syn_i.T.reshape(1,self.sub_no-1,-1)

        filtered_e = F.conv1d(pad_syn_e, e_kern, groups=self.sub_no-1).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, i_kern, groups=self.sub_no-1).squeeze(0).T

        syn = filtered_e + filtered_i 

        nn = self.conv_list(V.reshape(1,1,-1)).squeeze(0).T

        out = nn+syn

        return out