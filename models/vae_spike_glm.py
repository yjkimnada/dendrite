import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_Spike_GLM(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device

        ### Cosine Basis ###
        self.cos_basis_no = 13
        self.cos_shift = 1
        self.cos_scale = 5
        self.cos_basis = torch.zeros(self.cos_basis_no, self.T_no).to(self.device)
        for i in range(self.cos_basis_no):
            phi = 1.5707963267948966*i
            xmin = phi - 3.141592653589793
            xmax = phi + 3.141592653589793
            
            x_in = torch.arange(self.T_no).float().to(self.device)
            raw_cos = self.cos_scale * torch.log(x_in + self.cos_shift + 1e-8)
            
            basis = 0.5*torch.cos(raw_cos - phi) + 0.5
            basis[raw_cos < xmin] = 0.0
            basis[raw_cos > xmax] = 0.0 
            self.cos_basis[i] = self.cos_basis[i] + basis

        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.randn(self.sub_no, self.cos_basis_no, 2)*0.01 , requires_grad=True)

        ### Spiking History Parameters ###
        self.W_hist = nn.Parameter(torch.randn(self.sub_no, self.cos_basis_no)*0.01 , requires_grad=True)

        ### Spike Propagation Parametesr ###
        self.W_prop = nn.Parameter(torch.randn(self.sub_no, self.cos_basis_no)*0.01 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)

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

        out_filters = torch.vstack((torch.flip(e_kern.squeeze(1), [1]),
                                   torch.flip(i_kern.squeeze(1), [1])))
        
        return syn, out_filters

    def train_forward(self, S_e, S_i, Z):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i)

        Z_pad = torch.zeros(T_data + self.T_no, self.sub_no).to(self.device)
        Z_pad[-T_data:] = Z_pad[-T_data:] + Z
        Z_pad = Z_pad.T.reshape(1,self.sub_no,-1)

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        prop_kern = torch.matmul(self.W_prop, self.cos_basis)

        hist_kern = torch.flip(hist_kern, [1]).unsqueeze(1)
        prop_kern = torch.flip(prop_kern, [1]).unsqueeze(1)

        hist_filt = F.conv1d(Z_pad, hist_kern, groups=self.sub_no).squeeze(0).T[:-1]
        prop_filt = F.conv1d(Z_pad, prop_kern, groups=self.sub_no).squeeze(0).T[:-1]

        Z_pred = torch.zeros(T_data , self.sub_no).to(self.device)

        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                sub_in = syn[:,sub_idx] + self.Theta[sub_idx] + hist_filt[:,sub_idx]
                Z_pred[:,sub_idx] = Z_pred[:,sub_idx] + torch.sigmoid(sub_in)

            else:
                sub_in = syn[:,sub_idx] + self.Theta[sub_idx] + hist_filt[:,sub_idx] + torch.sum(prop_filt[:,leaf_idx].reshape(T_data,-1),1)
                Z_pred[:,sub_idx] = Z_pred[:,sub_idx] + torch.sigmoid(sub_in)

        out_filters = torch.vstack((syn_filters,
                                torch.flip(hist_kern.squeeze(1), [1]),
                                torch.flip(prop_kern.squeeze(1), [1])))

        return Z_pred, out_filters
    
    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i)

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        prop_kern = torch.matmul(self.W_prop, self.cos_basis)

        hist_kern = torch.flip(hist_kern, [1])
        prop_kern = torch.flip(prop_kern, [1])

        Z_pred = torch.zeros(T_data + self.T_no, self.sub_no).to(self.device)

        for t in range(T_data):
            Z_hist = Z_pred[t:t+self.T_no,:].clone()
            Z_hist_filt = torch.sum(Z_hist.T * hist_kern, 1)
            Z_prop_filt_raw = torch.sum(Z_hist.T * prop_kern, 1)
            Z_prop_filt = torch.matmul(self.C_den, Z_prop_filt_raw)

            sub_in = syn[t] + self.Theta + Z_hist_filt + Z_prop_filt
            Z_pred[t+self.T_no,:] = Z_pred[t+self.T_no,:] + torch.bernoulli(torch.sigmoid(sub_in))

        out_filters = torch.vstack((syn_filters,
                                torch.flip(hist_kern.squeeze(1), [1]),
                                torch.flip(prop_kern.squeeze(1), [1])))

        return Z_pred[self.T_no:], out_filters

class GLM_Encoder(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device

        ### Cosine Basis ###
        self.cos_basis_no = 13
        self.cos_shift = 1
        self.cos_scale = 5
        self.cos_basis = torch.zeros(self.cos_basis_no, self.T_no).to(self.device)
        for i in range(self.cos_basis_no):
            phi = 1.5707963267948966*i
            xmin = phi - 3.141592653589793
            xmax = phi + 3.141592653589793
            
            x_in = torch.arange(self.T_no).float().to(self.device)
            raw_cos = self.cos_scale * torch.log(x_in + self.cos_shift + 1e-8)
            
            basis = 0.5*torch.cos(raw_cos - phi) + 0.5
            basis[raw_cos < xmin] = 0.0
            basis[raw_cos > xmax] = 0.0 
            self.cos_basis[i] = self.cos_basis[i] + basis

        ### Observed Basis ###
        self.obs_basis_no = 13
        self.obs_shift = 1
        self.obs_scale = 5
        self.obs_basis = torch.zeros(self.obs_basis_no*2-1, self.T_no*2+1).to(self.device)
        for i in range(self.obs_basis_no):
            phi = 1.5707963267948966*i
            xmin = phi - 3.141592653589793
            xmax = phi + 3.141592653589793

            if i == 0:
                x_in = torch.arange(-self.T_no, self.T_no+1, 1).float().to(self.device)
                raw_cos = self.obs_scale * torch.log(torch.abs(x_in) + self.obs_shift + 1e-8)
                basis = 0.5*torch.cos(raw_cos - phi) + 0.5
                basis[raw_cos < xmin] = 0.0
                basis[raw_cos > xmax] = 0.0
                self.obs_basis[i] = basis
            else:
                x_in_pos = torch.arange(0,self.T_no+1,1)
                x_in_neg = torch.arange(-self.T_no,1,1)
                raw_cos_pos = self.obs_scale  * torch.log(torch.abs(x_in_pos) + self.obs_shift + 1e-8)
                raw_cos_neg = self.obs_scale  * torch.log(torch.abs(x_in_neg) + self.obs_shift + 1e-8)
                pos_basis = 0.5*torch.cos(raw_cos_pos - phi)+ 0.5
                neg_basis = 0.5*torch.cos(raw_cos_neg - phi)+ 0.5
                pos_basis[raw_cos_pos < xmin] = 0.0
                pos_basis[raw_cos_pos > xmax] = 0.0
                neg_basis[raw_cos_neg < xmin] = 0.0
                neg_basis[raw_cos_neg > xmax] = 0.0
                self.obs_basis[i*2-1, self.T_no:] = pos_basis
                self.obs_basis[i*2, :self.T_no+1] = neg_basis
        
        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.randn(self.sub_no-1, self.cos_basis_no, 2)*0.01 , requires_grad=True)

        ### Spiking History Parameters ###
        #self.W_hist = nn.Parameter(torch.randn(self.sub_no-1, self.cos_basis_no)*0.01 , requires_grad=True)

        ### Z_observed Parameters ###
        self.W_obs = nn.Parameter(torch.zeros(self.sub_no-1, self.obs_basis_no*2-1) , requires_grad=True)

        ### Output Parameters ###
        self.Theta = nn.Parameter(torch.zeros(self.sub_no-1) , requires_grad=True)

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]
        
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
        pad_syn_i= pad_syn_i.T.reshape(1,self.sub_no-1,-1)

        filtered_e = F.conv1d(pad_syn_e, e_kern, groups=self.sub_no-1).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, i_kern, groups=self.sub_no-1).squeeze(0).T

        syn = filtered_e + filtered_i
        
        return syn

    def forward(self, S_e, S_i, Z_obs, temp):
        T_data = S_e.shape[0]
        syn = self.spike_convolve(S_e, S_i)

        #hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        obs_kern = torch.matmul(self.W_obs, self.obs_basis)

        #hist_kern = torch.flip(hist_kern, [1])
        obs_kern = torch.flip(obs_kern, [1]).unsqueeze(1)

        Z_obs_pad = torch.zeros(T_data + self.T_no*2).to(self.device)
        Z_obs_pad[self.T_no:-self.T_no] = Z_obs_pad[self.T_no:-self.T_no] + Z_obs
        Z_obs_pad = Z_obs_pad.reshape(1,1,-1)
        Z_obs_filt = F.conv1d(Z_obs_pad, obs_kern).squeeze(0).T

        L_hid = torch.zeros(T_data, self.sub_no-1, 2).to(self.device)
        L_hid[:,:,0] = L_hid[:,:,0] + syn+Z_obs_filt+self.Theta
        
        eps=1e-8
        u = torch.rand_like(L_hid)
        g = - torch.log(- torch.log(u + eps) + eps) 
        Z_hid_raw = F.softmax((L_hid+g) / temp, dim=2)
        Z_hid = Z_hid_raw[:,:,0]
        
        
        #L_hid = torch.sigmoid(syn + Z_obs_filt + self.Theta)
        #alpha = L_hid / (1-L_hid)
        #u = torch.rand_like(alpha)
        #l = torch.log(u +1e-8) - torch.log(1-u +1e-8)
        #Z_hid = 1/(1+torch.exp(-(torch.log(alpha +1e-8)+l)/temp))
        
        #print(-(torch.log(alpha +1e-8)+l)/temp)
        #print(self.W_syn, self.W_obs)
        #print(Z_obs_filt, syn)
        #print(L_hid, Z_hid)
        
        
        """
        
        Z_hid = torch.zeros(T_data + self.T_no, self.sub_no-1).to(self.device)
        for t in range(T_data):
            Z_hist = Z_hid[t:t+self.T_no,:].clone()
            Z_hist_filt = torch.sum(Z_hist.T * hist_kern , 1)

            sub_in = syn[t] + self.Theta + Z_hist_filt + Z_obs_filt[t]
            sub_prob = torch.sigmoid(sub_in)
            alpha = sub_prob / (1-sub_prob)
            u = torch.rand_like(alpha)
            l = torch.log(u) - torch.log(1-u)
            x = 1/(1+torch.exp(-(torch.log(alpha)+l)/temp))
            Z_hid[t+self.T_no,:] = Z_hid[t+self.T_no,:] + x
        """
     
        #return Z_hid[self.T_no:]
        return Z_hid, torch.sigmoid(L_hid[:,:,0])
    
class NN_Encoder(nn.Module):
    def __init__(self, T_no, out_no, layer_no, device):
        super().__init__()
        
        self.T_no = T_no
        self.out_no = out_no
        self.layer_no = layer_no
        
        modules = []
        
        for i in range(self.layer_no):
            if i == 0:
                modules.append(nn.Conv1d(in_channels=1,
                                        out_channels=self.out_no,
                                        kernel_size = 2*self.T_no+1,
                                        padding=self.T_no))
                modules.append(nn.LeakyReLU())
            if i == self.layer_no - 1:
                modules.append(nn.Conv1d(in_channels=self.out_no,
                                         out_channels=self.out_no,
                                        kernel_size = 2*self.T_no+1,
                                        padding=self.T_no))
            else:
                modules.append(nn.Conv1d(in_channels=self.out_no,
                                         out_channels=self.out_no,
                                        kernel_size = 2*self.T_no+1,
                                        padding=self.T_no))
                modules.append(nn.LeakyReLU())
                
        self.conv_list = nn.Sequential(*modules)
        
        