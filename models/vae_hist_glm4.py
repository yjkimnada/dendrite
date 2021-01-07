import torch
from torch import nn
from torch.nn import functional as F

class VAE_Hist_GLM4(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        
        self.plex_no = 1
        
        ### Cosine Basis ###
        self.cos_basis_no = 22
        self.cos_shift = 1
        self.cos_scale = 6
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
        self.W_syn = nn.Parameter(torch.randn(self.sub_no*self.plex_no,2)*0.01 , requires_grad=True)
        self.Tau_syn = nn.Parameter(torch.ones(self.sub_no*self.plex_no,2)*1 , requires_grad=True)
        self.Delta_syn = nn.Parameter(torch.ones(self.sub_no*self.plex_no,2) , requires_grad=True)
        #self.W_syn = nn.Parameter(torch.rand(self.sub_no*self.plex_no, self.cos_basis_no, 2), requires_grad=True)

        self.W_plex = nn.Parameter(torch.randn(self.sub_no, self.plex_no)*0.1, requires_grad=True)

        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.randn(self.sub_no, self.plex_no)*0.1, requires_grad=True)
        
        ### Spike Parameters ###
        self.W_spk = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Delta_spk = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Tau_spk = nn.Parameter(torch.ones(1)*3, requires_grad=True)

        ### Subunit Output Parameters ###
        #self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no, self.plex_no), requires_grad=True)

    def spike_convolve(self, S_e, S_i, test):
        T_data = S_e.shape[0]
        
        syn_e_raw = torch.matmul(S_e, self.C_syn_e.T)
        syn_i_raw = torch.matmul(S_i, self.C_syn_i.T)
        
        syn_e = torch.zeros(T_data, self.sub_no*self.plex_no).to(self.device)
        syn_i = torch.zeros(T_data, self.sub_no*self.plex_no).to(self.device)
        
        for i in range(self.sub_no):
            syn_e[:,i*self.plex_no:(i+1)*self.plex_no] = syn_e[:,i*self.plex_no:(i+1)*self.plex_no] + syn_e_raw[:,i].reshape(-1,1).repeat(1,self.plex_no)
            syn_i[:,i*self.plex_no:(i+1)*self.plex_no] = syn_i[:,i*self.plex_no:(i+1)*self.plex_no] + syn_i_raw[:,i].reshape(-1,1).repeat(1,self.plex_no)
        
        
        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no*self.plex_no,1).to(self.device)
        t_e = t - self.Delta_syn[:,0].reshape(-1,1)
        t_i = t - self.Delta_syn[:,1].reshape(-1,1)
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0 
        t_tau_e = t_e / self.Tau_syn[:,0].reshape(-1,1)**2
        t_tau_i = t_i / self.Tau_syn[:,1].reshape(-1,1)**2
        full_e_kern = t_tau_e * torch.exp(-t_tau_e) * self.W_syn[:,0].reshape(-1,1)
        full_i_kern = t_tau_i * torch.exp(-t_tau_i) * self.W_syn[:,1].reshape(-1,1)
        """
        
        full_e_kern = torch.matmul(self.W_syn[:,:,0], self.cos_basis) ##### CONSTRAIN?!
        full_i_kern = torch.matmul(self.W_syn[:,:,1], self.cos_basis)
        """
        
        full_e_kern = torch.flip(full_e_kern, [1])
        full_i_kern = torch.flip(full_i_kern, [1])
        full_e_kern = full_e_kern.unsqueeze(1)
        full_i_kern = full_i_kern.unsqueeze(1)
        
        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.sub_no*self.plex_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.sub_no*self.plex_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.unsqueeze(0)
        pad_syn_i = pad_syn_i.T.unsqueeze(0)
        
        filt_e = F.conv1d(pad_syn_e, full_e_kern, padding=0, groups=self.sub_no*self.plex_no).squeeze(0).T
        filt_i = F.conv1d(pad_syn_e, full_e_kern, padding=0, groups=self.sub_no*self.plex_no).squeeze(0).T
        
        syn_out = filt_e + filt_i    
        out_filters = torch.vstack((torch.flip(full_e_kern.squeeze(1), [1]),
                                   torch.flip(full_i_kern.squeeze(1), [1]),
                                   ))
        
        return syn_out, out_filters

    def train_forward(self, S_e, S_i, Z):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i, test=False)
        sub_out = torch.zeros(T_data , self.sub_no).to(self.device)

        pad_Z = torch.zeros(T_data + self.T_no).to(self.device)
        pad_Z[-T_data:] = pad_Z[-T_data:] + Z
        pad_Z = pad_Z.reshape(1,1,-1)

        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
            
            if sub_idx == -self.sub_no:
                syn_in = syn[:,sub_idx*self.plex_no:(sub_idx+1)*self.plex_no]
                ancest_in = torch.matmul(sub_out[:,leaf_idx].reshape(T_data,-1) , self.W_sub[leaf_idx].reshape(-1,self.plex_no)**2)
                theta_in = self.Theta[sub_idx,:].reshape(1,-1)
                sub_in = syn_in + ancest_in + theta_in
                nonlin_out = torch.tanh(sub_in)
                final_out = torch.sum(nonlin_out * self.W_plex[sub_idx].reshape(1,-1)**2 , 1)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + final_out
            elif sub_idx == -1:
                syn_in = syn[:,sub_idx*self.plex_no:]
                theta_in = self.Theta[sub_idx,:].reshape(1,-1)
                sub_in = syn_in + theta_in
                nonlin_out = torch.tanh(sub_in)
                final_out = torch.sum(nonlin_out * self.W_plex[sub_idx].reshape(1,-1)**2 , 1)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + final_out
            elif torch.numel(leaf_idx) == 0:
                syn_in = syn[:,sub_idx*self.plex_no:(sub_idx+1)*self.plex_no]
                theta_in = self.Theta[sub_idx,:].reshape(1,-1)
                sub_in = syn_in + theta_in
                nonlin_out = torch.tanh(sub_in)
                final_out = torch.sum(nonlin_out * self.W_plex[sub_idx].reshape(1,-1)**2 , 1)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + final_out          
            else:
                syn_in = syn[:,sub_idx*self.plex_no:(sub_idx+1)*self.plex_no]
                ancest_in = torch.matmul(sub_out[:,leaf_idx].reshape(T_data,-1) , self.W_sub[leaf_idx].reshape(-1,self.plex_no)**2)
                theta_in = self.Theta[sub_idx,:].reshape(1,-1)
                sub_in = syn_in + theta_in + ancest_in
                nonlin_out = torch.tanh(sub_in)
                final_out = torch.sum(nonlin_out * self.W_plex[sub_idx].reshape(1,-1)**2 , 1)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + final_out
        
        final_Z = torch.sigmoid(sub_out[:,0])
        
        t = torch.arange(self.T_no).to(self.device)
        t_tau = t / self.Tau_spk**2
        spk_kern = t_tau * torch.exp(-t_tau) * self.W_spk**2
        spk_kern = torch.flip(spk_kern, [0]).reshape(1,1,-1)
        spk_filt = F.conv1d(pad_Z, spk_kern).flatten()[:-1]
        
        #final_V = spk_filt + self.V_o
        final_V = spk_filt
        out_filters = syn_filters

        return final_V, final_Z, out_filters

    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i, test=False)
        sub_out = torch.zeros(T_data , self.sub_no).to(self.device)
        
        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
            
            if sub_idx == -self.sub_no:
                syn_in = syn[:,sub_idx*self.plex_no:(sub_idx+1)*self.plex_no]
                ancest_in = torch.matmul(sub_out[:,leaf_idx].reshape(T_data,-1) , self.W_sub[leaf_idx].reshape(-1,self.plex_no)**2)
                theta_in = self.Theta[sub_idx,:].reshape(1,-1)
                sub_in = syn_in + ancest_in + theta_in
                nonlin_out = torch.tanh(sub_in)
                final_out = torch.sum(nonlin_out * self.W_plex[sub_idx].reshape(1,-1)**2 , 1)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + final_out
            elif sub_idx == -1:
                syn_in = syn[:,sub_idx*self.plex_no:]
                theta_in = self.Theta[sub_idx,:].reshape(1,-1)
                sub_in = syn_in + theta_in
                nonlin_out = torch.tanh(sub_in)
                final_out = torch.sum(nonlin_out * self.W_plex[sub_idx].reshape(1,-1)**2 , 1)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + final_out
            elif torch.numel(leaf_idx) == 0:
                syn_in = syn[:,sub_idx*self.plex_no:(sub_idx+1)*self.plex_no]
                theta_in = self.Theta[sub_idx,:].reshape(1,-1)
                sub_in = syn_in + theta_in
                nonlin_out = torch.tanh(sub_in)
                final_out = torch.sum(nonlin_out * self.W_plex[sub_idx].reshape(1,-1)**2 , 1)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + final_out          
            else:
                syn_in = syn[:,sub_idx*self.plex_no:(sub_idx+1)*self.plex_no]
                ancest_in = torch.matmul(sub_out[:,leaf_idx].reshape(T_data,-1) , self.W_sub[leaf_idx].reshape(-1,self.plex_no)**2)
                theta_in = self.Theta[sub_idx,:].reshape(1,-1)
                sub_in = syn_in + theta_in + ancest_in
                nonlin_out = torch.tanh(sub_in)
                final_out = torch.sum(nonlin_out * self.W_plex[sub_idx].reshape(1,-1)**2 , 1)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + final_out
        
        final_Z = torch.sigmoid(sub_out[:,0])
        
        final_Z = torch.sigmoid(sub_out[:,0])
        final_spk = torch.bernoulli(torch.sigmoid(sub_out[:,0]))
        spk_out = torch.zeros(T_data + self.T_no).to(self.device)
        spk_out[-T_data:] = spk_out[-T_data:] + final_spk
        
        t = torch.arange(self.T_no).to(self.device)
        t_tau = t / self.Tau_spk**2
        spk_kern = t_tau * torch.exp(-t_tau) * self.W_spk**2
        spk_kern = torch.flip(spk_kern, [0]).reshape(1,1,-1)
        spk_filt = F.conv1d(spk_out.reshape(1,1,-1), spk_kern).flatten()[:-1]
        #final_V = spk_filt + self.V_o
        final_V = spk_filt
        
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