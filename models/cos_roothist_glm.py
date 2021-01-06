import torch
from torch import nn
from torch.nn import functional as F

class Cos_RootHist_GLM(nn.Module):
    def __init__(self, C_den, T_no, greedy, C_syn_e, C_syn_i, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        
        ### Cosine Basis ###
        self.cos_basis_no = 19
        self.cos_shift = 0
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
        
        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no)*0.01 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        
        ### Spiking Parameters ###
        self.W_hist = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)
    
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
    
    
    def train_forward(self, S_e, S_i, V):
        
        T_data = S_e.shape[0] 

        syn, syn_filters = self.spike_convolve(S_e, S_i)
        ns_out = torch.zeros(T_data , self.sub_no).to(self.device) #0th column empty! 

        for s in range(self.sub_no - 1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
                
            if torch.numel(leaf_idx) == 0:
                ns_in = syn[:,sub_idx] + self.Theta[sub_idx]
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)
                
            else:
                ns_in = syn[:,sub_idx] + torch.sum(ns_out[:,leaf_idx]*self.W_sub[leaf_idx]**2 , 1) + self.Theta[sub_idx]
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)
                
        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [0]).reshape(1,1,-1)
        
        pad_V = torch.zeros(T_data + self.T_no).to(self.device)
        pad_V[-T_data:] = pad_V[-T_data:] + V
        pad_V = pad_V.reshape(1,1,-1)
        
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        hist_filt = F.conv1d(pad_V, hist_kern).flatten()[:-1]
        root_in = hist_filt + syn[:,0] + torch.sum(ns_out[:,root_leaf_idx]*self.W_sub[root_leaf_idx]**2 , 1) + self.Theta[0]
        ns_out[:,0] = ns_out[:,0] + torch.exp(root_in)
        
        final_V = ns_out[:,0]*self.W_sub[0]**2 + self.V_o
        
        hist_out = torch.flip(hist_kern.reshape(1,-1),[1])
        
        out_filters = torch.vstack((syn_filters, hist_out))
        
        return final_V, out_filters
    
    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0] 

        syn, syn_filters = self.spike_convolve(S_e, S_i)

        ns_out = torch.zeros(T_data , self.sub_no).to(self.device) #0th column empty
        root_out = torch.zeros(T_data + self.T_no).to(self.device) #Actual 0th column!

        for s in range(self.sub_no - 1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
                
            if torch.numel(leaf_idx) == 0:
                ns_in = syn[:,sub_idx] + self.Theta[sub_idx]
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + F.softplus(ns_in)
                
            else:
                ns_in = syn[:,sub_idx] + torch.sum(ns_out[:,leaf_idx]*self.W_sub[leaf_idx]**2 , 1) + self.Theta[sub_idx]
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + F.softplus(ns_in)
                
        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [0])
        
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        
        for t in range(T_data):
            hist_filt = torch.sum(hist_kern * root_out[t:t+self.T_no].clone())
            root_in = hist_filt + syn[t,0] + torch.sum(ns_out[t,root_leaf_idx]*self.W_sub[root_leaf_idx]**2) + self.Theta[0]
            root_out[t+self.T_no] = root_out[t+self.T_no] + F.softplus(root_in)
            
        final_V = root_out[self.T_no:]*self.W_sub[0]**2 + self.V_o
        hist_out = torch.flip(hist_kern.reshape(1,-1),[1])
        
        out_filters = torch.vstack((syn_filters, hist_out))
        
        return final_V, ns_out, out_filters
        
        