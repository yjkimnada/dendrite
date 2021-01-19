import torch
from torch import nn
from torch.nn import functional as F

class Det_Hist_GLM(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        
        ### Cosine Basis ###
        self.cos_basis_no = 17
        self.cos_shift = 1
        self.cos_scale = 4
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
    
        ### Synaptic Parameters ###
        self.Tau_syn = nn.Parameter(torch.ones(self.sub_no, 2)*0 , requires_grad=True)
        self.Delta_syn = nn.Parameter(torch.ones(self.sub_no, 2)*0 , requires_grad=True)
        self.W_syn = nn.Parameter(torch.ones(self.sub_no, 2)*(-2) , requires_grad=True)
        
        #self.Tau_syn = nn.Parameter(torch.ones(self.sub_no, 2) , requires_grad=True)
        #self.Delta_syn = nn.Parameter(torch.zeros(self.sub_no, 2) , requires_grad=True)
        #self.W_syn = nn.Parameter(torch.ones(self.sub_no, 2)*(0.1) , requires_grad=True)
       
        ### Ancestor Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no)*(0), requires_grad=True)
    
        ### History Parameters ###
        #self.W_hist = nn.Parameter(torch.zeros(self.cos_basis_no) , requires_grad=True)
        self.W_hist = nn.Parameter(torch.ones(self.cos_basis_no)*(-2) , requires_grad=True)

        ### Output Parameters ###
        self.Theta = nn.Parameter(torch.ones(self.sub_no)*(0) , requires_grad=True)
        self.Tau_out = nn.Parameter(torch.ones(1)*2.5 , requires_grad=True)
        self.W_out = nn.Parameter(torch.ones(1)*1.5 , requires_grad=True)
        
        self.step = Step.apply

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)

        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        
        t_e = t - torch.exp(self.Delta_syn[:,0].reshape(-1,1))
        t_i = t - torch.exp(self.Delta_syn[:,1].reshape(-1,1))
        #t_e = t - self.Delta_syn[:,0].reshape(-1,1)
        #t_i = t - self.Delta_syn[:,1].reshape(-1,1)
        #t_e = t - self.Delta_syn[:,0].reshape(-1,1)**2
        #t_i = t - self.Delta_syn[:,1].reshape(-1,1)**2
        
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0

        t_tau_e = t_e / torch.exp(self.Tau_syn[:,0].reshape(-1,1))
        t_tau_i = t_i / torch.exp(self.Tau_syn[:,1].reshape(-1,1))
        #t_tau_e = t_e / self.Tau_syn[:,0].reshape(-1,1)**2
        #t_tau_i = t_i / self.Tau_syn[:,1].reshape(-1,1)**2
        
        e_kern = t_tau_e * torch.exp(-t_tau_e) * torch.exp(self.W_syn[:,0].reshape(-1,1))
        i_kern = t_tau_i * torch.exp(-t_tau_i) * torch.exp(self.W_syn[:,1].reshape(-1,1))*(-1)
        #e_kern = t_tau_e * torch.exp(-t_tau_e) * self.W_syn[:,0].reshape(-1,1)**2
        #i_kern = t_tau_i * torch.exp(-t_tau_i) * self.W_syn[:,1].reshape(-1,1)**2*(-1)
        
        e_kern = torch.flip(e_kern, [1]).unsqueeze(1)
        i_kern = torch.flip(i_kern, [1]).unsqueeze(1)

        pad_syn_e = torch.zeros(T_data + self.T_no, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no, self.sub_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.unsqueeze(0)
        pad_syn_i = pad_syn_i.T.unsqueeze(0)

        filt_e = F.conv1d(pad_syn_e, e_kern, groups=self.sub_no).squeeze(0).T[:-1]
        filt_i = F.conv1d(pad_syn_i, i_kern, groups=self.sub_no).squeeze(0).T[:-1]

        syn = filt_e + filt_i
        out_filters = torch.vstack((torch.flip(e_kern.squeeze(1), [1]),
                                   torch.flip(i_kern.squeeze(1), [1]),
                                   ))
        
        return syn, out_filters
    
    def train_forward(self, S_e, S_i, Z):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i)

        sub_out = torch.zeros(T_data , self.sub_no).to(self.device)
        for s in range(self.sub_no - 1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
            
            if torch.numel(leaf_idx) == 0:
                sub_in = syn[:,sub_idx] + self.Theta[sub_idx]
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.tanh(sub_in)
                
            else:
                sub_in = syn[:,sub_idx] + torch.sum(sub_out[:,leaf_idx]*torch.exp(self.W_sub[leaf_idx]), 1) + self.Theta[sub_idx]
                #sub_in = syn[:,sub_idx] + torch.sum(sub_out[:,leaf_idx]*self.W_sub[leaf_idx]**2, 1) + self.Theta[sub_idx]
                
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.tanh(sub_in)
                
        #hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.matmul(torch.exp(self.W_hist)*(-1), self.cos_basis)
        hist_kern = torch.flip(hist_kern, [0]).reshape(1,1,-1)
        
        pad_Z = torch.zeros(T_data + self.T_no).to(self.device)
        pad_Z[-T_data:] = pad_Z[-T_data:] + Z
        pad_Z = pad_Z.reshape(1,1,-1)
        
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        hist_filt = F.conv1d(pad_Z, hist_kern).flatten()[:-1]
        leaf_in = torch.sum(sub_out[:,root_leaf_idx] * torch.exp(self.W_sub[root_leaf_idx]), 1)
        #leaf_in = torch.sum(sub_out[:,root_leaf_idx] * self.W_sub[root_leaf_idx]**2, 1)
        
        Z_in = hist_filt + syn[:,0] + leaf_in + self.Theta[0]
        #Z_in = syn[:,0] + leaf_in + self.Theta[0]
        Z_out = self.step(Z_in)
        
        Z_pad = torch.zeros(T_data + self.T_no).to(self.device)
        Z_pad[-T_data:] = Z_pad[-T_data:] + Z_out
        Z_pad = Z_pad.reshape(1,1,-1)
        
        t_out = torch.arange(self.T_no).to(self.device)
        t_tau_out = t_out / torch.exp(self.Tau_out)
        out_kern = t_tau_out * torch.exp(-t_tau_out) * torch.exp(self.W_out)
        out_kern = torch.flip(out_kern, [0]).reshape(1,1,-1)
        
        V_out = F.conv1d(Z_pad, out_kern).flatten()[:-1]
        
        out_filters = torch.vstack((syn_filters,
                                torch.flip(hist_kern.squeeze(1), [1])))
        
        return V_out, Z_out, out_filters
    
    
    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i)

        sub_out = torch.zeros(T_data , self.sub_no).to(self.device)
        for s in range(self.sub_no - 1):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]
            
            if torch.numel(leaf_idx) == 0:
                sub_in = syn[:,sub_idx] + self.Theta[sub_idx]
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.tanh(sub_in)
                
            else:
                sub_in = syn[:,sub_idx] + torch.sum(sub_out[:,leaf_idx]*torch.exp(self.W_sub[leaf_idx]), 1) + self.Theta[sub_idx]
                #sub_in = syn[:,sub_idx] + torch.sum(sub_out[:,leaf_idx]*self.W_sub[leaf_idx]**2, 1) + self.Theta[sub_idx]
                
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.tanh(sub_in)
                
        Z = torch.zeros(self.T_no + T_data).to(self.device)
        P = torch.zeros(T_data).to(self.device)
        
        #hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.matmul(torch.exp(self.W_hist)*(-1), self.cos_basis)
        hist_kern = torch.flip(hist_kern, [0])
        
        root_leaf_idx = torch.where(self.C_den[0] == 1)[0]
        
        for t in range(T_data):
            Z_hist = Z[t:t+self.T_no].clone()
            hist_in = torch.sum(Z_hist * hist_kern)
            leaf_in = torch.sum(sub_out[t,root_leaf_idx]*torch.exp(self.W_sub[root_leaf_idx]))
            #leaf_in = torch.sum(sub_out[t,root_leaf_idx]*self.W_sub[root_leaf_idx]**2)
            
            Z_in = hist_in + syn[t,0] + leaf_in + self.Theta[0]
            #Z_in = syn[t,0] + leaf_in + self.Theta[0]
            Z[t+self.T_no] = Z[t+self.T_no] + self.step(Z_in)
            P[t] = P[t] + torch.sigmoid(Z_in)

        out_filters = torch.vstack((syn_filters,
                                torch.flip(hist_kern, [0]).reshape(1,-1)))
        
        t_out = torch.arange(self.T_no).to(self.device)
        t_tau_out = t_out / torch.exp(self.Tau_out)
        out_kern = t_tau_out * torch.exp(-t_tau_out) * torch.exp(self.W_out)
        out_kern = torch.flip(out_kern, [0]).reshape(1,1,-1)
        
        V_out = F.conv1d(Z.reshape(1,1,-1), out_kern).flatten()[:-1]

        return V_out, Z[self.T_no:], P, out_filters

class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        zero = torch.tensor([0.0]).cuda()
        return torch.heaviside(input, zero)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        derivative = 1/(1+torch.abs(input))**2
        #derivative = torch.sigmoid(input)*(1-torch.sigmoid(input))
        output = derivative * grad_input
        return output