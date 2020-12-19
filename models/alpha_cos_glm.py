import torch
from torch import nn
from torch.nn import functional as F

class Alpha_Cos_GLM(nn.Module):
    def __init__(self, C_den, E_no, I_no, T_no, greedy, C_syn_e, C_syn_i, device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.E_no = E_no
        self.I_no = I_no
        self.greedy = greedy
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device

        ### Synapse Parameters ###
        self.W_syn_raw = torch.rand(self.sub_no,2, 2) * 0.05
        self.W_syn_raw[:,:,1] *= -1
        self.W_syn = nn.Parameter(self.W_syn_raw, requires_grad=True)
        self.Tau_syn_raw = torch.arange(1.1,4,2).reshape(1,-1,1).repeat(self.sub_no,1,2).float()
        self.Tau_syn = nn.Parameter(self.Tau_syn_raw, requires_grad=True)
        self.Delta_syn = nn.Parameter(torch.zeros(self.sub_no,2, 2), requires_grad=True)
        
        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.zeros(self.sub_no)*0.05 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### History Parameters ###
        self.cos_scale = 4
        self.cos_basis_no = 12
        self.cos_shift = 1
        
        self.hist_weights = nn.Parameter(torch.randn(self.sub_no, self.cos_basis_no)*(-0.005) , requires_grad=True)
        self.hist_basis = torch.empty(self.cos_basis_no, self.T_no).to(self.device)
        for i in range(self.cos_basis_no):
            phi = 1.5707963267948966*i
            xmin = phi - 3.141592653589793
            xmax = phi + 3.141592653589793

            x_in = torch.arange(self.T_no).float().to(self.device)
            raw_cos = self.cos_scale * torch.log(x_in + self.cos_shift)
            small_idx = torch.where(raw_cos < xmin)[0]
            big_idx = torch.where(raw_cos > xmax)[0]

            basis = 0.5*torch.cos(raw_cos - phi) + 0.5
            basis[small_idx] = 0.0
            basis[big_idx] = 0.0
            self.hist_basis[i] = basis

        ### C_syn Parameters ###
        if self.greedy == True:
            self.C_syn_e_logit = nn.Parameter(torch.ones(self.sub_no, self.E_no), requires_grad=True)
            self.C_syn_i_logit = nn.Parameter(torch.ones(self.sub_no, self.I_no), requires_grad=True)
    
    def forward(self, S_e, S_i, temp, test):
        T_data = S_e.shape[0] 

        if self.greedy == True:
            if test == True:
                C_syn_e = torch.zeros_like(self.C_syn_e_logit).to(self.device)
                C_syn_i = torch.zeros_like(self.C_syn_i_logit).to(self.device)
                for i in range(C_syn_e.shape[1]):
                    idx = torch.argmax(self.C_syn_e_logit[:,i])
                    C_syn_e[idx,i] = 1
                for i in range(C_syn_i.shape[1]):
                    idx = torch.argmax(self.C_syn_i_logit[:,i])
                    C_syn_i[idx,i] = 1
            
            elif test == False:
                u_e = torch.rand_like(self.C_syn_e_logit).to(self.device)
                u_i = torch.rand_like(self.C_syn_i_logit).to(self.device)
                eps = 1e-8
                g_e = -torch.log(- torch.log(u_e + eps) + eps)
                g_i = -torch.log(- torch.log(u_i + eps) + eps)
                C_syn_e = F.softmax((self.C_syn_e_logit + g_e) / temp, dim=0)
                C_syn_i = F.softmax((self.C_syn_i_logit + g_i) / temp, dim=0)

        elif self.greedy == False:
            C_syn_e = self.C_syn_e
            C_syn_i = self.C_syn_i

        syn_e = torch.matmul(S_e, C_syn_e.T)
        syn_i = torch.matmul(S_i, C_syn_i.T)

        
        full_e_kern = torch.zeros(self.sub_no, self.T_no).to(self.device)
        full_i_kern = torch.zeros(self.sub_no, self.T_no).to(self.device)
        
        for b in range(2):
            t_raw_e = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
            t_raw_i = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)

            t_e = t_raw_e - self.Delta_syn[:,b,0].reshape(-1,1)
            t_i = t_raw_i - self.Delta_syn[:,b,1].reshape(-1,1)
            t_e[t_e < 0.0] = 0.0
            t_i[t_i < 0.0] = 0.0 

            tau_e = torch.exp(self.Tau_syn[:,b,0]).reshape(-1,1)
            tau_i = torch.exp(self.Tau_syn[:,b,1]).reshape(-1,1)
            t_e_tau = t_e / tau_e
            t_i_tau = t_i / tau_i
            part_e_kern = t_e_tau * torch.exp(-t_e_tau) * self.W_syn[:,b,0].reshape(-1,1)
            part_i_kern = t_i_tau * torch.exp(-t_i_tau) * self.W_syn[:,b,1].reshape(-1,1)
            full_e_kern = full_e_kern + part_e_kern
            full_i_kern = full_i_kern + part_i_kern

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
 
        syn_in = filtered_e + filtered_i

        sub_out = torch.zeros(T_data+self.T_no, self.sub_no).to(self.device)
        hist_kern = torch.matmul(self.hist_weights, self.hist_basis)
        #######
        hist_kern = torch.flip(hist_kern, [1])
        
        for t in range(T_data):
            sub_hist = sub_out[t:t+self.T_no,:].clone() 
            #sub_hist_in = F.conv1d(sub_hist.T.unsqueeze(0) , hist_kern, groups=self.sub_no).flatten() #(1, sub_no, 1)
            sub_hist_in = torch.sum(sub_hist.T * hist_kern, 1)
            
            sub_prop = torch.matmul(sub_out[self.T_no+t-1].clone()*self.W_sub , self.C_den.T)
            Y_out = torch.tanh(syn_in[t] + sub_prop + self.Theta + sub_hist_in)
            sub_out[t+self.T_no] = sub_out[t+self.T_no] + Y_out

        final_voltage = sub_out[self.T_no:,0]*self.W_sub[0] + self.V_o

        e_kern_out = torch.flip(full_e_kern, [2]).squeeze(1)
        i_kern_out = torch.flip(full_i_kern, [2]).squeeze(1)
        #########
        #hist_kern_out = torch.flip(hist_kern, [1])
        hist_kern_out = hist_kern
        out_filters = torch.vstack((e_kern_out, i_kern_out, hist_kern_out))

        return final_voltage, out_filters, C_syn_e, C_syn_i