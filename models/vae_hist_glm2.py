import torch
from torch import nn
from torch.nn import functional as F

class VAE_Hist_GLM2(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        self.layer_no = 3

        
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
        #self.W_syn = nn.Parameter(torch.zeros(self.sub_no*3, self.cos_basis_no, 2) , requires_grad=True)
        #self.pre_conv = nn.Sequential(
                        #nn.Conv1d(in_channels=self.sub_no*2,out_channels=self.sub_no*2,
                                  #kernel_size=2*self.T_no+1,padding=self.T_no,groups=self.sub_no*2),
        #)
                        #nn.LeakyReLU())
        self.conv_list1 = nn.Sequential(
                        nn.Conv1d(in_channels=self.sub_no*2,out_channels=self.sub_no*1,
                                  kernel_size=2*self.T_no+1,padding=self.T_no,groups=1),
                        nn.LeakyReLU(),
                        nn.Conv1d(in_channels=self.sub_no*1,out_channels=self.sub_no*1,
                                  kernel_size=2*self.T_no+1,padding=self.T_no,groups=1),
                        #nn.LeakyReLU(),
                        #nn.Conv1d(in_channels=self.sub_no*2,out_channels=self.sub_no*2,
                                  #kernel_size=2*self.T_no+1,padding=self.T_no,groups=self.sub_no),
                        #nn.LeakyReLU(),
                        #nn.Conv1d(in_channels=self.sub_no*1,out_channels=self.sub_no,
                                  #kernel_size=2*self.T_no+1,padding=self.T_no,groups=self.sub_no)
        )
        #self.conv_list2 = nn.Sequential(
                        #nn.Conv1d(in_channels=self.sub_no,out_channels=self.sub_no,
                                  #kernel_size=2*self.T_no+1,padding=self.T_no,groups=self.sub_no),
                        #nn.LeakyReLU(),
                        #nn.Conv1d(in_channels=self.sub_no,out_channels=self.sub_no,
                                  #kernel_size=2*self.T_no+1,padding=self.T_no,groups=self.sub_no),
                        #nn.LeakyReLU(),
                        #nn.Conv1d(in_channels=self.sub_no,out_channels=self.sub_no,
                                  #kernel_size=2*self.T_no+1,padding=self.T_no,groups=self.sub_no)
        #)

        ### History Parameters ###
        self.W_hist = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)

        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no)*0.1 , requires_grad=True)
        
        ### Spike Parameters ###
        #self.W_spk = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)
        self.W_spk = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Delta_spk = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Tau_spk = nn.Parameter(torch.ones(1)*3, requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        
        ### C_syn ###
        #self.C_syn_e_raw = nn.Parameter(torch.ones(self.sub_no, 2000) , requires_grad=True)
        #self.C_syn_i_raw = nn.Parameter(torch.ones(self.sub_no, 200) , requires_grad=True)

    def spike_convolve(self, S_e, S_i, test):
        T_data = S_e.shape[0]
        
        """
        if test == False:
            eps = 1e-8
            temp=0.025
            u_e = torch.rand_like(self.C_syn_e_raw)
            u_i = torch.rand_like(self.C_syn_i_raw)
            g_e = - torch.log(- torch.log(u_e + eps) + eps)
            g_i = - torch.log(- torch.log(u_i + eps) + eps)

            C_syn_e = F.softmax((self.C_syn_e_raw + g_e) / temp, dim=0)
            C_syn_i = F.softmax((self.C_syn_i_raw + g_i) / temp, dim=0)
            
        elif test == True:
            C_syn_e = torch.zeros(self.sub_no, 2000).to(self.device)
            C_syn_i = torch.zeros(self.sub_no, 200).to(self.device)
            
            for i in range(2000):
                max_idx = torch.argmax(self.C_syn_e_raw[:,i])
                C_syn_e[max_idx,i] = 1
            for i in range(200):
                max_idx = torch.argmax(self.C_syn_i_raw[:,i])
                C_syn_i[max_idx,i] = 1
        
        syn_e = torch.matmul(S_e, C_syn_e.T)
        syn_i = torch.matmul(S_i, C_syn_i.T)
        """
        
        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)
        
        syn_in = torch.zeros(T_data, self.sub_no*2).to(self.device)
        for i in range(self.sub_no):
            syn_in[:,i*2] = syn_in[:,i*2] + syn_e[:,i]
            syn_in[:,i*2+1] = syn_in[:,i*2+1] + syn_i[:,i]
            
        syn = self.conv_list1(syn_in.T.unsqueeze(0)).squeeze(0).T
        
        out_filters = torch.vstack((self.conv_list1[0].weight.reshape(-1,self.T_no*2+1),
                                   self.conv_list1[2].weight.reshape(-1,self.T_no*2+1),
                                   #self.conv_list1[4].weight.reshape(-1,self.T_no*2+1)
                                   ))
        #syn = e_out + i_out
        
        return syn, out_filters

    def train_forward(self, S_e, S_i, Z):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i, test=False)
        ns_out = torch.zeros(T_data , self.sub_no).to(self.device)

        pad_Z = torch.zeros(T_data + self.T_no).to(self.device)
        pad_Z[-T_data:] = pad_Z[-T_data:] + Z
        pad_Z = pad_Z.reshape(1,1,-1)

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [0]).reshape(1,1,-1)
        hist_filt = F.conv1d(pad_Z, hist_kern, groups=1).flatten()[:-1]

        for s in range(self.sub_no-1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
                
            if torch.numel(leaf_idx) == 0:
                ns_in = syn[:,sub_idx] + self.Theta[sub_idx]
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + F.leaky_relu(ns_in)

            else:
                ns_in = syn[:,sub_idx] + self.Theta[sub_idx] + torch.sum(ns_out[:,leaf_idx] * self.W_sub[leaf_idx]**2, 1)
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + F.leaky_relu(ns_in)
        
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        root_in = hist_filt + syn[:,0] + torch.sum(ns_out[:,root_leaf_idx] * self.W_sub[root_leaf_idx]**2, 1) + self.Theta[0]
        ns_out[:,0] = ns_out[:,0] + torch.sigmoid(root_in)
        
        final_Z = ns_out[:,0]
        
        t = torch.arange(self.T_no).to(self.device)
        t_tau = t / self.Tau_spk**2
        spk_kern = t_tau * torch.exp(-t_tau) * self.W_spk**2
        #spk_kern = torch.matmul(self.W_spk, self.cos_basis)
        spk_kern = torch.flip(spk_kern, [0]).reshape(1,1,-1)
        spk_filt = F.conv1d(pad_Z, spk_kern).flatten()[:-1]
        
        final_V = spk_filt + self.V_o
        hist_out = torch.flip(hist_kern.flatten(), [0]).reshape(1,-1)
        #out_filters = torch.vstack((syn_filters, hist_out))
        out_filters = syn_filters

        return final_V, final_Z, out_filters

    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i, test=True)
        ns_out = torch.zeros(T_data , self.sub_no).to(self.device)
        Z_out = torch.zeros(T_data+self.T_no).to(self.device)
        spk_out = torch.zeros(T_data+self.T_no).to(self.device)

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [0])
        
        for s in range(self.sub_no-1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
                
            if torch.numel(leaf_idx) == 0:
                ns_in = syn[:,sub_idx] + self.Theta[sub_idx]
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + F.leaky_relu(ns_in)

            else:
                ns_in = syn[:,sub_idx] + self.Theta[sub_idx] + torch.sum(ns_out[:,leaf_idx] * self.W_sub[leaf_idx]**2, 1)
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + F.leaky_relu(ns_in)
        
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        
        for t in range(T_data):
            hist_filt = torch.sum(spk_out[t:t+self.T_no].clone() * hist_kern)
            root_prop = torch.sum(ns_out[t,root_leaf_idx] * self.W_sub[root_leaf_idx]**2)
            root_in = syn[t,0] + hist_filt + root_prop + self.Theta[0]
            Z_out[t+self.T_no] = Z_out[t+self.T_no] + torch.sigmoid(root_in)
            spk_out[t+self.T_no] = spk_out[t+self.T_no] + torch.bernoulli(torch.sigmoid(root_in))
        
        final_Z = Z_out[self.T_no:]
        final_spk = spk_out[self.T_no:]
        
        t = torch.arange(self.T_no).to(self.device)
        t_tau = t / self.Tau_spk**2
        spk_kern = t_tau * torch.exp(-t_tau) * self.W_spk**2
        #spk_kern = torch.matmul(self.W_spk, self.cos_basis)
        spk_kern = torch.flip(spk_kern, [0]).reshape(1,1,-1)
        spk_filt = F.conv1d(spk_out.reshape(1,1,-1), spk_kern).flatten()[:-1]
        final_V = spk_filt + self.V_o
        
        hist_out = torch.flip(hist_kern.flatten(), [0]).reshape(1,-1)
        #out_filters = torch.vstack((syn_filters, hist_out))
        out_filters = syn_filters

        return final_V, final_Z, out_filters

class NN_Encoder(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, layer_no, device):
        super().__init__()

        self.T_no = 100
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
        
        self.Theta = nn.Parameter(torch.zeros(1), requires_grad=True)

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

        raw_syn = filtered_e + filtered_i 
        #syn = torch.sum(raw_syn, 1)

        nn = self.conv_list(V.reshape(1,1,-1)).flatten()

        #Z_out = torch.sigmoid(nn+syn+self.Theta)
        Z_out = torch.sigmoid(nn)
        spk_out_raw = torch.zeros(T_data, 2).to(self.device)
        #spk_out_raw[:,0] = spk_out_raw[:,0] + nn+syn+self.Theta
        spk_out_raw[:,0] = spk_out_raw[:,0] + nn
        
        eps = 1e-8
        temp=0.025
        u = torch.rand_like(spk_out_raw)
        g = - torch.log(- torch.log(u + eps) + eps)

        spk_out_pad = F.softmax((spk_out_raw + g) / temp, dim=1)
        spk_out = spk_out_pad[:,0]

        return spk_out, Z_out