import torch
from torch import nn
from torch.nn import functional as F

class Cos_RootSpike_GLM(nn.Module):
    def __init__(self, C_den, E_no, I_no, T_no, greedy, C_syn_e, C_syn_i, device):
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
        self.W_syn_ns = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no, 2) , requires_grad=True)
        self.W_syn_s = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no, 2) , requires_grad=True)
        
        ### Ancestor Subunit Parameters ###
        self.W_sub_ns = nn.Parameter(torch.ones(self.sub_no)*0.1 , requires_grad=True)
        self.W_sub_s = nn.Parameter(torch.ones(self.sub_no)*0.1 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta_ns = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        self.Theta_s = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        
        ### Spiking Parameters ###
        self.hist_s_weights = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)
        self.hist_ns_weights = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)
    
    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]
        
        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)
        
        e_kern_ns = torch.matmul(self.W_syn_ns[:,:,0], self.cos_basis)
        i_kern_ns = torch.matmul(self.W_syn_ns[:,:,1], self.cos_basis)
        e_kern_s = torch.matmul(self.W_syn_s[:,:,0], self.cos_basis)
        i_kern_s = torch.matmul(self.W_syn_s[:,:,1], self.cos_basis)
        
        e_kern_ns = torch.flip(e_kern_ns, [1]).unsqueeze(1)
        i_kern_ns = torch.flip(i_kern_ns, [1]).unsqueeze(1)
        e_kern_s = torch.flip(e_kern_s, [1]).unsqueeze(1)
        i_kern_s = torch.flip(i_kern_s, [1]).unsqueeze(1)
        
        pad_syn_e_ns = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i_ns = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e_s = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i_s = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e_ns[-T_data:] = pad_syn_e_ns[-T_data:] + syn_e
        pad_syn_i_ns[-T_data:] = pad_syn_i_ns[-T_data:] + syn_i
        pad_syn_e_s[-T_data:] = pad_syn_e_s[-T_data:] + syn_e
        pad_syn_i_s[-T_data:] = pad_syn_i_s[-T_data:] + syn_i
        pad_syn_e_ns = pad_syn_e_ns.T.reshape(1,self.sub_no,-1)
        pad_syn_i_ns = pad_syn_i_ns.T.reshape(1,self.sub_no,-1)
        pad_syn_e_s = pad_syn_e_s.T.reshape(1,self.sub_no,-1)
        pad_syn_i_s = pad_syn_i_s.T.reshape(1,self.sub_no,-1)
        
        filtered_e_ns = F.conv1d(pad_syn_e_ns, e_kern_ns, groups=self.sub_no).squeeze(0).T
        filtered_i_ns = F.conv1d(pad_syn_i_ns, i_kern_ns, groups=self.sub_no).squeeze(0).T
        filtered_e_s = F.conv1d(pad_syn_e_s, e_kern_s, groups=self.sub_no).squeeze(0).T
        filtered_i_s = F.conv1d(pad_syn_i_s, i_kern_s, groups=self.sub_no).squeeze(0).T
        
        syn_ns = filtered_e_ns + filtered_i_ns
        syn_s = filtered_e_s + filtered_i_s
        
        out_filters = torch.vstack((torch.flip(e_kern_ns.squeeze(1), [1]),
                                   torch.flip(i_kern_ns.squeeze(1), [1]),
                                   torch.flip(e_kern_s.squeeze(1), [1]),
                                   torch.flip(i_kern_s.squeeze(1), [1])))
        
        return syn_ns, syn_s, out_filters
        
    
    
    def train_forward(self, S_e, S_i, Z):
        
        T_data = S_e.shape[0] 

        syn_ns, syn_s, syn_filters = self.spike_convolve(S_e, S_i)

        ns_out = torch.zeros(T_data , self.sub_no).to(self.device) #0th column empty!
        s_out = torch.zeros(T_data , self.sub_no).to(self.device) #0th column empty! 
        

        for s in range(self.sub_no - 1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
                
            if torch.numel(leaf_idx) == 0:
                s_in = syn_s[:,sub_idx] + self.Theta_s[sub_idx]
                ns_in = syn_ns[:,sub_idx] + self.Theta_ns[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)
                
            else:
                s_in = syn_s[:,sub_idx] + torch.sum(s_out[:,leaf_idx]*self.W_sub_s[leaf_idx]**2 , 1) + self.Theta_s[sub_idx]
                ns_in = syn_ns[:,sub_idx] + torch.sum(ns_out[:,leaf_idx]*self.W_sub_ns[leaf_idx]**2 , 1) + self.Theta_ns[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)
                
        hist_s_kern = torch.matmul(self.hist_s_weights, self.cos_basis)
        hist_ns_kern = torch.matmul(self.hist_ns_weights, self.cos_basis)
        
        hist_s_kern = torch.flip(hist_s_kern, [0]).reshape(1,1,-1)
        hist_ns_kern = torch.flip(hist_ns_kern, [0]).reshape(1,1,-1)
        
        pad_Z = torch.zeros(T_data + self.T_no).to(self.device)
        pad_Z[-T_data:] = pad_Z[-T_data:] + Z
        pad_Z = pad_Z.reshape(1,1,-1)
        
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        hist_s_filt = F.conv1d(pad_Z, hist_s_kern).flatten()[:-1]
        root_s_in = hist_s_filt + syn_s[:,0] + torch.sum(s_out[:,root_leaf_idx]*self.W_sub_s[root_leaf_idx]**2 , 1) + self.Theta_s[0]
        s_out[:,0] = s_out[:,0] + torch.sigmoid(root_s_in)
        
        hist_ns_filt = F.conv1d(pad_Z, hist_ns_kern).flatten()[:-1]
        ns_in = hist_ns_filt + syn_ns[:,0] + torch.sum(ns_out[:,root_leaf_idx]*self.W_sub_ns[root_leaf_idx]**2 , 1) + self.Theta_ns[0]
        ns_out[:,0] = torch.tanh(ns_in)
        
        final_V = ns_out[:,0]*self.W_sub_ns[0]**2 + self.V_o
        final_Z = s_out[:,0]
        
        hist_s_out = torch.flip(hist_s_kern.reshape(1,-1),[1])
        hist_ns_out = torch.flip(hist_ns_kern.reshape(1,-1),[1])
        
        out_filters = torch.vstack((syn_filters, hist_ns_out, hist_s_out))
        
        return final_V, final_Z, out_filters
    
    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0] 

        syn_ns, syn_s, syn_filters = self.spike_convolve(S_e, S_i)

        ns_out = torch.zeros(T_data , self.sub_no).to(self.device) #0th column empty!
        s_out = torch.zeros(T_data , self.sub_no).to(self.device) #0th column empty!
        
        root_s_out = torch.zeros(T_data + self.T_no).to(self.device) #Actual 0th column!
        root_l_out =  torch.zeros(T_data + self.T_no).to(self.device) 

        for s in range(self.sub_no - 1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
                
            if torch.numel(leaf_idx) == 0:
                s_in = syn_s[:,sub_idx] + self.Theta_s[sub_idx]
                ns_in = syn_ns[:,sub_idx] + self.Theta_ns[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)
                
            else:
                s_in = syn_s[:,sub_idx] + torch.sum(s_out[:,leaf_idx]*self.W_sub_s[leaf_idx]**2 , 1) + self.Theta_s[sub_idx]
                ns_in = syn_ns[:,sub_idx] + torch.sum(ns_out[:,leaf_idx]*self.W_sub_ns[leaf_idx]**2 , 1) + self.Theta_ns[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)
                
        hist_s_kern = torch.matmul(self.hist_s_weights, self.cos_basis)
        hist_ns_kern = torch.matmul(self.hist_ns_weights, self.cos_basis)
 
        hist_s_kern = torch.flip(hist_s_kern, [0])
        hist_ns_kern = torch.flip(hist_ns_kern, [0]).reshape(1,1,-1)
        
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        
        for t in range(T_data):
            hist_s_filt = torch.sum(hist_s_kern * root_s_out[t:t+self.T_no].clone())
            root_s_in = hist_s_filt + syn_s[t,0] + torch.sum(s_out[t,root_leaf_idx]*self.W_sub_s[root_leaf_idx]**2) + self.Theta_s[0]
            root_s_out[t+self.T_no] = root_s_out[t+self.T_no] + torch.bernoulli(torch.sigmoid(root_s_in))
            root_l_out[t+self.T_no] = root_l_out[t+self.T_no] + torch.sigmoid(root_s_in)
            
        hist_ns_filt = F.conv1d(root_s_out.reshape(1,1,-1), hist_ns_kern).flatten()[:-1]
        root_ns_in = hist_ns_filt + syn_ns[:,0] + torch.sum(ns_out[:,root_leaf_idx]*self.W_sub_ns[root_leaf_idx]**2 , 1) + self.Theta_ns[0]
        root_ns_out = torch.tanh(root_ns_in)
        
        final_V = root_ns_out*self.W_sub_ns[0]**2 + self.V_o
        final_Z = root_s_out[self.T_no:]
        final_L = root_l_out[self.T_no:]
        
        hist_s_out = torch.flip(hist_s_kern.reshape(1,-1),[1])
        hist_ns_out = torch.flip(hist_ns_kern.reshape(1,-1),[1])
        
        out_filters = torch.vstack((syn_filters, hist_ns_out, hist_s_out))
        
        return final_V, final_Z, final_L, out_filters
        
        