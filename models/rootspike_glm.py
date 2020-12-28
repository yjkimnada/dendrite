import torch
import torch.nn as nn
import torch.nn.functional as F


class RootSpike_GLM(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.C_den = C_den
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.T_no = T_no
        self.device = device
        self.sub_no = C_den.shape[0]

        
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
        

        ### Spike Synapse Parameters ###
        self.W_syn_s = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no, 2) , requires_grad=True)
        #self.W_syn_s_e = nn.Parameter(torch.randn(self.sub_no, self.T_no)*0.1 , requires_grad=True)
        #self.W_syn_s_i = nn.Parameter(torch.randn(self.sub_no, self.T_no)*0.1 , requires_grad=True)
        
        #self.W_syn_s = nn.Parameter(torch.rand(self.sub_no,2)*0.1 , requires_grad=True)
        #self.Tau_s = nn.Parameter(torch.ones(self.sub_no,2) , requires_grad=True)
        #self.Delta_s = nn.Parameter(torch.zeros(self.sub_no,2) , requires_grad=True)

        ### Spiking Parameters ###
        self.hist_s_weights = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)
        #self.hist_s = nn.Parameter(torch.zeros(self.T_no ) , requires_grad=True)

        ### Ancestor Subunit Parameters ###
        self.W_sub_s = nn.Parameter(torch.ones(self.sub_no)*0.1 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.Theta_s = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)
        
        ######
        #e_kern_s = self.W_syn_s_e
        #i_kern_s = self.W_syn_s_i
        
        e_kern_s = torch.matmul(self.W_syn_s[:,:,0], self.cos_basis)
        i_kern_s = torch.matmul(self.W_syn_s[:,:,1], self.cos_basis)
        
        #t_raw = torch.arange(self.T_no).to(self.device).reshape(1,-1).repeat(self.sub_no,1)
        #t_e = t_raw - self.Delta_s[:,0].reshape(-1,1)**2
        #t_i = t_raw - self.Delta_s[:,1].reshape(-1,1)**2
        
        #t_e_tau = t_e / self.Tau_s[:,0].reshape(-1,1)**2
        #t_i_tau = t_i / self.Tau_s[:,1].reshape(-1,1)**2
        
        #e_kern_s = t_e_tau * torch.exp(-t_e_tau) * self.W_syn_s[:,0].reshape(-1,1)**2
        #i_kern_s = t_i_tau * torch.exp(-t_i_tau) * self.W_syn_s[:,1].reshape(-1,1)**2*(-1)
        
        ########
        
        e_kern_s = torch.flip(e_kern_s, [1]).unsqueeze(1)
        i_kern_s = torch.flip(i_kern_s, [1]).unsqueeze(1)

        pad_syn_e_s = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i_s = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e_s[-T_data:] = pad_syn_e_s[-T_data:] + syn_e
        pad_syn_i_s[-T_data:] = pad_syn_i_s[-T_data:] + syn_i
        pad_syn_e_s = pad_syn_e_s.T.reshape(1,self.sub_no,-1)
        pad_syn_i_s = pad_syn_i_s.T.reshape(1,self.sub_no,-1)

        filtered_e_s = F.conv1d(pad_syn_e_s, e_kern_s, groups=self.sub_no).squeeze(0).T
        filtered_i_s = F.conv1d(pad_syn_i_s, i_kern_s, groups=self.sub_no).squeeze(0).T

        syn_s = filtered_e_s + filtered_i_s

        out_filters = torch.vstack((torch.flip(e_kern_s.squeeze(1), [1]),
                                   torch.flip(i_kern_s.squeeze(1), [1])))
        
        return syn_s, out_filters

    def train_forward(self, S_e, S_i, Z):
        T_data = S_e.shape[0]
        syn_s, syn_filters = self.spike_convolve(S_e, S_i)
        s_out = torch.zeros(T_data , self.sub_no).to(self.device) #0th column empty! 

        for s in range(self.sub_no - 1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                s_in = syn_s[:,sub_idx] + self.Theta_s[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)

            else:
                s_in = syn_s[:,sub_idx] + torch.sum(s_out[:,leaf_idx]*self.W_sub_s[leaf_idx]**2 , 1) + self.Theta_s[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)

        hist_s_kern = torch.matmul(self.hist_s_weights, self.cos_basis)
        #hist_s_kern = self.hist_s
        hist_s_kern = torch.flip(hist_s_kern, [0]).reshape(1,1,-1)

        pad_Z = torch.zeros(T_data + self.T_no).to(self.device)
        pad_Z[-T_data:] = pad_Z[-T_data:] + Z
        pad_Z = pad_Z.reshape(1,1,-1)
        
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        hist_s_filt = F.conv1d(pad_Z, hist_s_kern).flatten()[:-1]
        root_s_in = hist_s_filt + syn_s[:,0] + torch.sum(s_out[:,root_leaf_idx]*self.W_sub_s[root_leaf_idx]**2 , 1) + self.Theta_s[0]
        s_out[:,0] = s_out[:,0] + torch.sigmoid(root_s_in)

        final_Z = s_out[:,0]
        hist_s_out = torch.flip(hist_s_kern.reshape(1,-1),[1])
        out_filters = torch.vstack((syn_filters, hist_s_out))
        
        return final_Z, out_filters

    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0] 
        syn_s, syn_filters = self.spike_convolve(S_e, S_i)
        s_out = torch.zeros(T_data , self.sub_no).to(self.device) #0th column empty!
        
        root_s_out = torch.zeros(T_data + self.T_no).to(self.device) #Actual 0th column!

        for s in range(self.sub_no - 1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                s_in = syn_s[:,sub_idx] + self.Theta_s[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)

            else:
                s_in = syn_s[:,sub_idx] + torch.sum(s_out[:,leaf_idx]*self.W_sub_s[leaf_idx]**2 , 1) + self.Theta_s[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)

        hist_s_kern = torch.matmul(self.hist_s_weights, self.cos_basis)
        #hist_s_kern = self.hist_s
        hist_s_kern = torch.flip(hist_s_kern, [0])
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        zero = torch.tensor([0.0]).to(self.device)
        for t in range(T_data):
            hist_s_filt = torch.sum(hist_s_kern * root_s_out[t:t+self.T_no].clone())
            root_s_in = hist_s_filt + syn_s[t,0] + torch.sum(s_out[t,root_leaf_idx]*self.W_sub_s[root_leaf_idx]**2) + self.Theta_s[0]
            root_s_out[t+self.T_no] = root_s_out[t+self.T_no] + torch.heaviside(root_s_in, zero)

        
        final_Z = root_s_out[self.T_no:]
        hist_s_out = torch.flip(hist_s_kern.reshape(1,-1),[1])
        out_filters = torch.vstack((syn_filters, hist_s_out))
        
        return final_Z, out_filters

