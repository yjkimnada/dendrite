import torch
from torch import nn
from torch.nn import functional as F

class Alpha_EMRootSpike_GLM(nn.Module):
    def __init__(self, C_den, E_no, I_no, T_no, C_syn_e, C_syn_i, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.E_no = E_no
        self.I_no = I_no
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        
        ### Non-Spike Synapse Parameters ###
        self.W_syn_raw = torch.rand(self.sub_no, 2) * 0.1
        self.W_syn_raw[:,1] *= -1
        self.W_syn_ns = nn.Parameter(self.W_syn_raw, requires_grad=True)
        self.Tau_syn_ns = nn.Parameter(torch.ones(self.sub_no, 2) * 3.0, requires_grad=True)
        self.Delta_syn_ns = nn.Parameter(torch.zeros(self.sub_no, 2), requires_grad=True)

        ### Make Cosine Basis ###
        self.cos_basis_no = 20
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
            self.cos_basis[i] = basis
        
        ### Spike Synapse Parameters ###
        self.W_syn_s = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no,2) , requires_grad=True)
        
        ### Ancestor Subunit Parameters ###
        self.W_sub_ns = nn.Parameter(torch.ones(self.sub_no)*0.1 , requires_grad=True)
        self.W_sub_s = nn.Parameter(torch.ones(self.sub_no)*0.1 , requires_grad=True)
        
        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta_ns = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        self.Theta_s = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### Non-Spike/Spike History Parameters ###
        self.hist_s_weights = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)
        self.hist_ns_weights = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)

    

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]
        
        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)
        
        t_raw = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        tau_e_ns = self.Tau_syn_ns[:,0].reshape(-1,1)**2
        tau_i_ns = self.Tau_syn_ns[:,1].reshape(-1,1)**2
        
        t_e_ns = t_raw - self.Delta_syn_ns[:,0].reshape(-1,1)
        t_i_ns = t_raw - self.Delta_syn_ns[:,1].reshape(-1,1)
        
        t_e_ns[t_e_ns < 0.0] = 0.0
        t_i_ns[t_i_ns < 0.0] = 0.0
        
        t_e_tau_ns = t_e_ns / tau_e_ns
        t_i_tau_ns = t_i_ns / tau_i_ns
        
        e_kern_ns = t_e_tau_ns * torch.exp(-t_e_tau_ns) * self.W_syn_ns[:,0].reshape(-1,1)**2
        i_kern_ns = t_i_tau_ns * torch.exp(-t_i_tau_ns) * self.W_syn_ns[:,1].reshape(-1,1)**2*(-1)
        e_kern_s = torch.matmul(self.W_syn_s[:,:,0]**2, self.cos_basis)
        i_kern_s = torch.matmul(self.W_syn_s[:,:,1]**2*(-1), self.cos_basis)
        
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
            
        ns_out = torch.zeros(T_data , self.sub_no).to(self.device)
        s_out = torch.zeros(T_data , self.sub_no).to(self.device)
        
        Z_pad = torch.zeros(T_data + self.T_no).to(self.device)
        Z_pad[-T_data:] = Z_pad[-T_data:] + Z
        Z_pad = Z_pad.reshape(1,1,-1)
        
        hist_s_kern = torch.matmul(self.hist_s_weights, self.cos_basis)
        hist_ns_kern = torch.matmul(self.hist_ns_weights, self.cos_basis)
        hist_s_kern = torch.flip(hist_s_kern, [0]).reshape(1,1,-1)
        hist_ns_kern = torch.flip(hist_ns_kern, [0]).reshape(1,1,-1)
        
        hist_ns_filt = F.conv1d(Z_pad, hist_ns_kern).flatten()[:-1]
        hist_s_filt = F.conv1d(Z_pad, hist_s_kern).flatten()[:-1]
        
        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
            
            if sub_idx == -self.sub_no:
                s_in = hist_s_filt + syn_s[:,0] + torch.sum(s_out[:,leaf_idx]*self.W_sub_s[leaf_idx]**2, 1) + self.Theta_s[0]
                ns_in = hist_ns_filt + syn_ns[:,0] + torch.sum(ns_out[:,leaf_idx]*self.W_sub_ns[leaf_idx]**2, 1) + self.Theta_ns[0]
                s_out[:,0] = s_out[:,0] + torch.sigmoid(s_in)
                ns_out[:,0] = ns_out[:,0] + torch.tanh(ns_in)
                
            elif torch.numel(leaf_idx) == 0:
                s_in = syn_s[:,sub_idx] + self.Theta_s[sub_idx]
                ns_in = syn_ns[:,sub_idx] + self.Theta_ns[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)
                
            else:
                s_in = syn_s[:,sub_idx] + torch.sum(s_out[:,leaf_idx]*self.W_sub_s[leaf_idx]**2 , 1) + self.Theta_s[sub_idx]
                ns_in = syn_ns[:,sub_idx] + torch.sum(ns_out[:,leaf_idx]*self.W_sub_ns[leaf_idx]**2 , 1) + self.Theta_ns[sub_idx]
                s_out[:,sub_idx] = s_out[:,sub_idx] + torch.tanh(s_in)
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + torch.tanh(ns_in)

        final_V = ns_out[:,0]*self.W_sub_ns[0]**2 + self.V_o
        final_Z = s_out[:,0]
        
        hist_s_out = torch.flip(hist_s_kern.reshape(1,-1),[1])
        hist_ns_out = torch.flip(hist_ns_kern.reshape(1,-1),[1])
        out_filters = torch.vstack((syn_filters, hist_ns_out, hist_s_out))

        return final_V, final_Z, out_filters   
    
    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]

        syn_ns, syn_s, syn_filters = self.spike_convolve(S_e, S_i)
        
        ns_out = torch.zeros(T_data , self.sub_no-1).to(self.device)
        s_out = torch.zeros(T_data , self.sub_no-1).to(self.device)
        
        root_s_out = torch.zeros(T_data + self.T_no).to(self.device)

        hist_s_kern = torch.matmul(self.hist_s_weights, self.cos_basis)
        hist_ns_kern = torch.matmul(self.hist_ns_weights, self.cos_basis)
        hist_s_kern = torch.flip(hist_s_kern, [0])
        hist_ns_kern = torch.flip(hist_ns_kern, [0]).reshape(1,1,-1)
        
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
        
        root_leaf_idx = torch.where(self.C_den[0,1:] == 1)[0]
        zero = torch.tensor([0.0]).to(self.device)
        
        for t in range(T_data):
            hist_s_filt = torch.sum(hist_s_kern * root_s_out[t:t+self.T_no].clone())
            root_s_in = hist_s_filt + syn_s[t,0] + torch.sum(s_out[t,root_leaf_idx]*self.W_sub_s[root_leaf_idx]**2) + self.Theta_s[0]
            root_s_out[t+self.T_no] = root_s_out[t+self.T_no] + torch.heaviside(root_s_in, zero)
            
        hist_ns_filt = F.conv1d(root_s_out.reshape(1,1,-1), hist_ns_kern).flatten()[:-1]
        root_ns_in = hist_ns_filt + syn_ns[:,0] + torch.sum(ns_out[:,root_leaf_idx]*self.W_sub_ns[root_leaf_idx]**2, 1) + self.Theta_ns[0]
        root_ns_out = torch.tanh(root_ns_in)
        
        final_V = root_ns_out*self.W_sub_ns[0]**2 + self.V_o
        final_Z = root_s_out[self.T_no:]
             
        hist_s_out = torch.flip(hist_s_kern.reshape(1,-1),[1])
        hist_ns_out = torch.flip(hist_ns_kern.reshape(1,-1),[1])
        out_filters = torch.vstack((syn_filters, hist_ns_out, hist_s_out))

        return final_V, final_Z, out_filters

class Encoder(nn.Module):
    def __init__(self, T_no, device):
        super().__init__()

        self.T_no = T_no
        self.device = device
        

        self.ff = nn.Sequential(nn.Linear(2*(self.T_no) - 1, 500),
                                nn.LeakyReLU(),
                                nn.Linear(500, 500),
                                nn.LeakyReLU(),
                                nn.Linear(500, 500),
                                nn.LeakyReLU(),
                                nn.Linear(500, 1))


        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.rand(2) * 0.1, requires_grad=True)
        self.Tau_syn = nn.Parameter(torch.ones(2) * 3, requires_grad=True)
        self.Delta_syn = nn.Parameter(torch.zeros(2), requires_grad=True)


    def forward(self, V, S_e, S_i):
        T_data = S_e.shape[0]

        syn_e = torch.sum(S_e, 1)
        syn_i = torch.sum(S_i, 1)

        t_raw = torch.arange(self.T_no).to(self.device)
        t_e = t_raw - self.Delta_syn[0]
        t_i = t_raw - self.Delta_syn[1]
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0

        tau_e = self.Tau_syn[0] ** 2
        tau_i = self.Tau_syn[1] ** 2
        t_tau_e = t_e / tau_e
        t_tau_i = t_i / tau_i

        e_kern = t_tau_e * torch.exp(-t_tau_e) * self.W_syn[0]**2
        i_kern = t_tau_i * torch.exp(-t_tau_i) * self.W_syn[1]**2*(-1)
        e_kern = torch.flip(e_kern, [0]).reshape(1,1,-1)
        i_kern = torch.flip(i_kern, [0]).reshape(1,1,-1)


        pad_syn_e = torch.zeros(T_data + self.T_no - 1).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.reshape(1,1,-1)
        pad_syn_i = pad_syn_i.reshape(1,1,-1)

        filtered_e = F.conv1d(pad_syn_e, e_kern).flatten()
        filtered_i = F.conv1d(pad_syn_i, i_kern).flatten()
        syn_out = filtered_e + filtered_i
        
        #V_pad = torch.zeros(T_data + 2*(self.T_no-1)).to(self.device)
        #V_pad[self.T_no-1:-self.T_no+1] = V_pad[self.T_no-1:-self.T_no+1] + V
        #ff_out = torch.zeros(T_data).to(self.device)

        #for t in range(T_data):
            #ff_out[t] = ff_out[t] + self.ff(V_pad[t:t+2*self.T_no-1].clone().reshape(1,-1))
           
        ff_out = self.ff(V).flatten()

        raw_out = ff_out + syn_out

        #Z = torch.sigmoid(raw_out)
        Z = raw_out

        return Z








