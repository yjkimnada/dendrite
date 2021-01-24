import torch
from torch import nn
from torch.nn import functional as F

class SRM_Net(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, 
            W_syn_init, W_spk_init,
            Tau_syn_init, Tau_spk_init,
            Delta_syn_init, Delta_spk_init,
            device):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device

        ### Cosine Basis ###
        self.cos_basis_no = 16
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
        self.Tau_syn_e = nn.Parameter(torch.ones(self.sub_no) * Tau_syn_init ,requires_grad=True)
        self.Tau_syn_i = nn.Parameter(torch.ones(self.sub_no) * Tau_syn_init ,requires_grad=True)
        self.W_syn_e = nn.Parameter(torch.ones(self.sub_no) * W_syn_init ,requires_grad=False) ### FIXED!
        self.W_syn_i = nn.Parameter(torch.ones(self.sub_no) * W_syn_init ,requires_grad=True)
        self.Delta_syn_e = nn.Parameter(torch.ones(self.sub_no) * Delta_syn_init ,requires_grad=True)
        self.Delta_syn_i = nn.Parameter(torch.ones(self.sub_no) * Delta_syn_init ,requires_grad=True)

        ### Spiking Parameters ###
        self.Tau_spk = nn.Parameter(torch.ones(self.sub_no) * Tau_spk_init ,requires_grad=True)
        self.W_spk = nn.Parameter(torch.ones(self.sub_no) * W_spk_init ,requires_grad=True)

        ### History Parameters ###
        self.W_hist = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no) , requires_grad=True)

        ### Output Parameters ###
        self.Theta = nn.Parameter(torch.zeros(self.sub_no) , requires_grad=True)

        self.step = Step.apply

    def forward(self, S_e, S_i):
        T_data = S_e.shape[0]

        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)

        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t_e = t - torch.exp(self.Delta_syn_e.reshape(-1,1))
        t_i = t - torch.exp(self.Delta_syn_i.reshape(-1,1))
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0

        t_tau_e = t_e / torch.exp(self.Tau_syn_e.reshape(-1,1))
        t_tau_i = t_i / torch.exp(self.Tau_syn_i.reshape(-1,1))
        e_kern = t_tau_e * torch.exp(-t_tau_e) * torch.exp(self.W_syn_e.reshape(-1,1))
        i_kern = t_tau_i * torch.exp(-t_tau_i) * torch.exp(self.W_syn_i.reshape(-1,1))*(-1)
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
        
        t_tau_spk = t / torch.exp(self.Tau_spk.reshape(-1,1))
        spk_kern = t_tau_spk * torch.exp(-t_tau_spk) * torch.exp(self.W_spk.reshape(-1,1))
        spk_kern = torch.flip(spk_kern, [1])

        hist_kern = torch.matmul(self.W_hist, self.cos_basis)
        hist_kern = torch.flip(hist_kern, [1])

        spk_out = torch.zeros(self.T_no + T_data, self.sub_no).to(self.device)

        for t in range(T_data):
            spk_hist = spk_out[t:t+self.T_no].clone()
            hist_in = torch.sum(spk_hist.T * hist_kern, 1)
            spk_in_raw = torch.sum(spk_hist.T * spk_kern, 1)
            spk_in = torch.matmul(self.C_den, spk_in_raw)
            sub_in = hist_in + spk_in + syn[t] + self.Theta
            spk_out[t+self.T_no] = spk_out[t+self.T_no] + self.step(sub_in)

        out_filters = torch.vstack((
            torch.flip(e_kern.squeeze(1), [1]),
            torch.flip(i_kern.squeeze(1), [1]),
            torch.flip(spk_kern, [1]),
            torch.flip(hist_kern, [1])
        ))

        return spk_out[self.T_no:], out_filters



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
        output = derivative * grad_input
        return output