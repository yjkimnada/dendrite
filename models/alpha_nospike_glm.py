import torch
from torch import nn
from torch.nn import functional as F

class Alpha_NoSpike_GLM(nn.Module):
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
        self.W_syn_ns = nn.Parameter(torch.rand(self.sub_no, 2) * 0.1, requires_grad=True)
        self.Tau_syn_ns = nn.Parameter(torch.ones(self.sub_no, 2) * 3, requires_grad=True)
        self.Delta_syn_ns = nn.Parameter(torch.zeros(self.sub_no, 2), requires_grad=True)

        ### Ancestor Continuous Propagation Parameters ###
        self.W_ns_ns = nn.Parameter(torch.ones(self.sub_no)*1 , requires_grad=True)

        ### Final Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta_ns = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)

        ns_e_kern = torch.zeros(self.sub_no, self.T_no).to(self.device)
        ns_i_kern = torch.zeros(self.sub_no, self.T_no).to(self.device)

        t_raw = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t_e_ns = t_raw - self.Delta_syn_ns[:,0].reshape(-1,1)
        t_i_ns = t_raw - self.Delta_syn_ns[:,1].reshape(-1,1)
        t_e_ns[t_e_ns < 0.0] = 0.0
        t_i_ns[t_i_ns < 0.0] = 0.0

        tau_e_ns = self.Tau_syn_ns[:,0].reshape(-1,1)**2
        tau_i_ns = self.Tau_syn_ns[:,1].reshape(-1,1)**2
        t_e_tau_ns = t_e_ns / tau_e_ns
        t_i_tau_ns = t_i_ns / tau_i_ns

        ns_e_kern = t_e_tau_ns * torch.exp(-t_e_tau_ns) * self.W_syn_ns[:,0].reshape(-1,1)**2
        ns_i_kern = t_i_tau_ns * torch.exp(-t_i_tau_ns) * self.W_syn_ns[:,1].reshape(-1,1)**2*(-1)
        ns_e_kern = torch.flip(ns_e_kern, [1]).unsqueeze(1)
        ns_i_kern = torch.flip(ns_i_kern, [1]).unsqueeze(1)

        pad_syn_e_ns = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i_ns = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e_ns[-T_data:] = pad_syn_e_ns[-T_data:] + syn_e
        pad_syn_i_ns[-T_data:] = pad_syn_i_ns[-T_data:] + syn_i

        filtered_e_ns = F.conv1d(pad_syn_e_ns.T.unsqueeze(0), ns_e_kern, padding=0, groups=self.sub_no).squeeze(0).T # (T_data, sub_no)
        filtered_i_ns = F.conv1d(pad_syn_i_ns.T.unsqueeze(0), ns_i_kern, padding=0, groups=self.sub_no).squeeze(0).T

        syn_in_ns = filtered_e_ns + filtered_i_ns

        return syn_in_ns
    
    
    def test_forward(self, S_e, S_i):
        T_data = S_e.shape[0]

        syn_in_ns = self.spike_convolve(S_e, S_i) # (T_data, sub_no)

        ns_out = torch.zeros(T_data, self.sub_no).to(self.device)

        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                ns = torch.tanh(syn_in_ns[:,sub_idx] + self.Theta_ns[sub_idx])
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + ns

            else:
                prop_ns_ns = torch.sum(ns_out[:,leaf_idx] * self.W_ns_ns[leaf_idx]**2 , 1)
                ns = torch.tanh(syn_in_ns[:,sub_idx] + self.Theta_ns[sub_idx] + prop_ns_ns)
                ns_out[:,sub_idx] = ns 

        V_ns = ns_out[:,0] * self.W_ns_ns[0]**2 + self.V_o

        V = V_ns

        return V, ns_out

    def train_forward(self, S_e, S_i, Y):
        T_data = S_e.shape[0]

        syn_in_ns = self.spike_convolve(S_e, S_i) # (T_data, sub_no)

        Y_pad = torch.zeros(T_data, self.sub_no).to(self.device)
        Y_pad[-T_data:] = Y_pad[-T_data:] + Y
        
        prop_ns_ns = torch.matmul(Y_pad[:] * self.W_ns_ns**2, self.C_den.T)

        ns_out = torch.zeros(T_data, self.sub_no).to(self.device)
        
        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                ns = torch.tanh(syn_in_ns[:,sub_idx] + self.Theta_ns[sub_idx])
                ns_out[:,sub_idx] = ns_out[:,sub_idx] + ns

            else:
                ns = torch.tanh(syn_in_ns[:,sub_idx] + self.Theta_ns[sub_idx] + prop_ns_ns[:,sub_idx])
                ns_out[:,sub_idx] = ns

        V_ns = ns_out[:,0] * self.W_ns_ns[0]**2 + self.V_o

        V = V_ns

        return V, ns_out


class Encoder(nn.Module):
    def __init__(self, sub_no, T_no, device):
        super().__init__()

        self.T_no = T_no
        self.sub_no = sub_no
        self.device = device


        self.ff = nn.Sequential(nn.Linear(2*(self.T_no) - 1, 500),
                                nn.LeakyReLU(),
                                nn.Linear(500, 500),
                                nn.LeakyReLU(),
                                nn.Linear(500, 500),
                                nn.LeakyReLU(),
                                nn.Linear(500, 1*self.sub_no))


        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.rand(self.sub_no, 2) * 0.1, requires_grad=True)
        self.Tau_syn = nn.Parameter(torch.ones(self.sub_no, 2) * 3, requires_grad=True)
        self.Delta_syn = nn.Parameter(torch.zeros(self.sub_no, 2), requires_grad=True)


    def forward(self, V, S_e, S_i):
        T_data = S_e.shape[0]

        syn_e = torch.sum(S_e, 1)
        syn_i = torch.sum(S_i, 1)

        t_raw = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t_e = t_raw - self.Delta_syn[:,0].reshape(-1,1)
        t_i = t_raw - self.Delta_syn[:,1].reshape(-1,1)
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0

        tau_e = self.Tau_syn[:,0].reshape(-1,1) ** 2
        tau_i = self.Tau_syn[:,1].reshape(-1,1) ** 2
        t_tau_e = t_e / tau_e
        t_tau_i = t_i / tau_i

        e_kern = t_tau_e * torch.exp(-t_tau_e) * self.W_syn[:,0].reshape(-1,1)**2
        i_kern = t_tau_i * torch.exp(-t_tau_i) * self.W_syn[:,1].reshape(-1,1)**2*(-1)
        e_kern = torch.flip(e_kern, [1]).unsqueeze(1)
        i_kern = torch.flip(i_kern, [1]).unsqueeze(1)

        pad_syn_e = torch.zeros(T_data + self.T_no - 1).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.reshape(1,1,-1)
        pad_syn_i = pad_syn_i.reshape(1,1,-1)

        filtered_e = F.conv1d(pad_syn_e, e_kern).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, i_kern).squeeze(0).T
        syn_out = filtered_e + filtered_i # (T_data, 2*sub_no)
        
        V_pad = torch.zeros(T_data + 2*(self.T_no-1)).to(self.device)
        V_pad[self.T_no-1:-self.T_no+1] = V_pad[self.T_no-1:-self.T_no+1] + V
        ff_out = torch.zeros(T_data, self.sub_no).to(self.device)

        for t in range(T_data):
            ff_out[t] = ff_out[t] + self.ff(V_pad[t:t+2*self.T_no-1].reshape(1,-1)).flatten()

        raw_out = ff_out + syn_out

        Y = torch.tanh(raw_out[:,:self.sub_no])

        return Y








