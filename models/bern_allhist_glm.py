import torch
from torch import nn
from torch.nn import functional as F

class Bern_AllHist_GLM(nn.Module):
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
        self.cos_scale = 3
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
        self.Tau_syn = nn.Parameter(torch.ones(self.sub_no, 2) , requires_grad=True)
        self.Delta_syn = nn.Parameter(torch.zeros(self.sub_no, 2) , requires_grad=True)
        self.W_syn = nn.Parameter(torch.ones(self.sub_no, 2)*(0.01) , requires_grad=True)
        #self.W_syn = nn.Parameter(torch.randn(self.sub_no, self.cos_basis_no, 2)*0.01 ,  requires_grad=True)

        ### Spiking Parameters ###
        #self.Tau_spk = nn.Parameter(torch.zeros(self.sub_no) , requires_grad=True)
        #self.Delta_spk = nn.Parameter(torch.zeros(self.sub_no) , requires_grad=True)
        #self.W_spk = nn.Parameter(torch.ones(self.sub_no)*(-5) , requires_grad=True)
        self.W_spk = nn.Parameter(torch.randn(self.sub_no, self.cos_basis_no)*0.01 , requires_grad=True)

        ### History Parameters ###
        self.W_hist = nn.Parameter(torch.randn(self.sub_no, self.cos_basis_no)*0.1 , requires_grad=True)

        ### Output Parameters ###
        self.Theta = nn.Parameter(torch.ones(self.sub_no)*(-5) , requires_grad=True)

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)

        
        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t_e = t - self.Delta_syn[:,0].reshape(-1,1)
        t_i = t - self.Delta_syn[:,1].reshape(-1,1)
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0

        t_tau_e = t_e / self.Tau_syn[:,0].reshape(-1,1)**2
        t_tau_i = t_i / self.Tau_syn[:,1].reshape(-1,1)**2
        e_kern = t_tau_e * torch.exp(-t_tau_e) * self.W_syn[:,0].reshape(-1,1)**2
        i_kern = t_tau_i * torch.exp(-t_tau_i) * self.W_syn[:,1].reshape(-1,1)**2*(-1)
        
        
        #e_kern = torch.matmul(self.W_syn[:,:,0]**2 , self.cos_basis)
        #i_kern = torch.matmul(self.W_syn[:,:,1]**2*(-1) , self.cos_basis)
        
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
        P = torch.zeros(T_data, self.sub_no).to(self.device)

        pad_Z = torch.zeros(T_data + self.T_no, self.sub_no).to(self.device)
        pad_Z[-T_data:] = pad_Z[-T_data:] + Z
        pad_Z = pad_Z.T.unsqueeze(0)

        """
        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t = t - torch.exp(self.Delta_spk).reshape(-1,1)
        t[t < 0.0] = 0.0
        t_tau = t / torch.exp(self.Tau_spk.reshape(-1,1))
        spk_kern = t_tau * torch.exp(-t_tau) * torch.exp(self.W_spk.reshape(-1,1))
        """
        
        spk_kern = torch.matmul(self.W_spk, self.cos_basis)
        
        spk_kern = torch.flip(spk_kern, [1]).unsqueeze(1)
        spk_filt_raw = F.conv1d(pad_Z, spk_kern, groups=self.sub_no).squeeze(0).T[:-1]
        spk_filt = torch.matmul(spk_filt_raw , self.C_den.T)

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [1]).unsqueeze(1)
        hist_filt = F.conv1d(pad_Z, hist_kern, groups=self.sub_no).squeeze(0).T[:-1]

        P = torch.sigmoid(syn + self.Theta.reshape(1,-1) + hist_filt + spk_filt)
        out_filters = torch.vstack((syn_filters,
                                torch.flip(spk_kern.squeeze(1), [1]),
                                torch.flip(hist_kern.squeeze(1), [1])))

        return P, out_filters

    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        syn, syn_filters = self.spike_convolve(S_e, S_i)

        """
        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t = t - torch.exp(self.Delta_spk).reshape(-1,1)
        t[t < 0.0] = 0.0
        t_tau = t / torch.exp(self.Tau_spk.reshape(-1,1))
        spk_kern = t_tau * torch.exp(-t_tau) * torch.exp(self.W_spk.reshape(-1,1))
        """
        
        spk_kern = torch.matmul(self.W_spk, self.cos_basis)
        spk_kern = torch.flip(spk_kern, [1])

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [1])

        Z = torch.zeros(self.T_no + T_data, self.sub_no).to(self.device)
        P = torch.zeros(T_data, self.sub_no).to(self.device)

        for t in range(T_data):
            Z_hist = Z[t:t+self.T_no].clone()
            hist_in = torch.sum(Z_hist.T * hist_kern , 1)
            spk_in = torch.matmul(self.C_den, torch.sum(Z_hist.T * spk_kern , 1))

            sub_in = hist_in + spk_in + syn[t] + self.Theta
            P[t] = P[t] + torch.sigmoid(sub_in)
            Z[t+self.T_no] = Z[t+self.T_no] + torch.bernoulli(torch.sigmoid(sub_in))

        out_filters = torch.vstack((syn_filters,
                                torch.flip(spk_kern.squeeze(1), [1]),
                                torch.flip(hist_kern.squeeze(1), [1])))

        return Z[self.T_no:], P, out_filters
    
    
class GLM_Encoder(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, device):
        super().__init__()

        self.T_no = T_no
        self.sub_no = C_syn_e.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        
        ### Cosine Basis ###
        self.cos_basis_no = 13
        self.cos_shift = 1
        self.cos_scale = 3
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
        #self.Tau_syn = nn.Parameter(torch.ones(self.sub_no-1, 2) , requires_grad=True)
        #self.Delta_syn = nn.Parameter(torch.zeros(self.sub_no-1, 2) , requires_grad=True)
        #self.W_syn = nn.Parameter(torch.ones(self.sub_no-1, 2)*0.01 , requires_grad=True)
       
        self.W_syn = nn.Parameter(torch.randn(self.sub_no-1, self.cos_basis_no, 2)*0.01 ,  requires_grad=True)
        
        self.Theta = nn.Parameter(torch.ones(self.sub_no-1)*(-5), requires_grad=True)
        
        ### Obs Basis ###
        self.obs_basis_no = 15
        self.obs_basis = torch.zeros(self.obs_basis_no*2-1, 2*self.T_no+1).to(self.device)
        self.scale = 4
        self.shift = 1
        
        for i in range(self.obs_basis_no):
            phi = 1.5707963267948966*i
            xmin = phi - 3.141592653589793
            xmax = phi + 3.141592653589793
            if i == 0:
                x_in = torch.arange(-self.T_no, self.T_no+1,1).to(self.device)
                raw_cos = self.scale * torch.log(torch.abs(x_in) + self.shift + 1e-7)
                basis = 0.5*torch.cos(raw_cos - phi) + 0.5
                basis[raw_cos < xmin] = 0.0
                basis[raw_cos > xmax] = 0.0
                self.obs_basis[i] = basis
            else:
                x_in_pos = torch.arange(0, self.T_no+1, 1).to(self.device)
                x_in_neg = torch.arange(-self.T_no, 1, 1).to(self.device)
                raw_cos_pos = self.scale * torch.log(torch.abs(x_in_pos) + self.shift + 1e-7)
                raw_cos_neg = self.scale * torch.log(torch.abs(x_in_neg) + self.shift + 1e-7)
                pos_basis = 0.5*torch.cos(raw_cos_pos - phi)+ 0.5
                neg_basis = 0.5*torch.cos(raw_cos_neg - phi)+ 0.5

                pos_basis[raw_cos_pos < xmin] = 0.0
                pos_basis[raw_cos_pos > xmax] = 0.0
                neg_basis[raw_cos_neg < xmin] = 0.0
                neg_basis[raw_cos_neg > xmax] = 0.0
                
                self.obs_basis[i*2-1, self.T_no:] = pos_basis
                self.obs_basis[i*2, :self.T_no+1] = neg_basis
        
        ### Obs Parameters ###
        self.W_obs = nn.Parameter(torch.rand(self.sub_no-1, self.obs_basis_no*2-1)*0.01 , requires_grad=True)
       
    def forward(self, Z, S_e, S_i):
        T_data = Z.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e[1:].T)
        syn_i = torch.matmul(S_i, self.C_syn_i[1:].T)
        
        """
        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no-1,1).to(self.device)
        t_e = t - self.Delta_syn[:,0].reshape(-1,1)**2
        t_i = t - self.Delta_syn[:,1].reshape(-1,1)**2
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0

        t_tau_e = t_e / self.Tau_syn[:,0].reshape(-1,1)**2
        t_tau_i = t_i / self.Tau_syn[:,1].reshape(-1,1)**2
        e_kern = t_tau_e * torch.exp(-t_tau_e) * self.W_syn[:,0].reshape(-1,1)**2
        i_kern = t_tau_i * torch.exp(-t_tau_i) * self.W_syn[:,1].reshape(-1,1)**2*(-1)
        """
        
        e_kern = torch.matmul(self.W_syn[:,:,0]**2 , self.cos_basis)
        i_kern = torch.matmul(self.W_syn[:,:,1]**2*(-1) , self.cos_basis)
        
        e_kern = torch.flip(e_kern, [1]).unsqueeze(1)
        i_kern = torch.flip(i_kern, [1]).unsqueeze(1)

        pad_syn_e = torch.zeros(T_data + self.T_no, self.sub_no-1).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no, self.sub_no-1).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.unsqueeze(0)
        pad_syn_i = pad_syn_i.T.unsqueeze(0)

        filt_e = F.conv1d(pad_syn_e, e_kern, groups=self.sub_no-1).squeeze(0).T[:-1]
        filt_i = F.conv1d(pad_syn_i, i_kern, groups=self.sub_no-1).squeeze(0).T[:-1]
        syn = filt_e + filt_i
        
        obs_kern = torch.matmul(self.W_obs, self.obs_basis)
        obs_kern = torch.flip(obs_kern, [1]).unsqueeze(1)
        obs_filt = F.conv1d(Z.reshape(1,1,-1), obs_kern, padding=self.T_no).squeeze(0).T
        
        P_raw = syn + obs_filt + self.Theta
        
        P = torch.sigmoid(P_raw)
        Z = torch.bernoulli(P)
        
        return Z, P
        
        
        
        