import torch
from torch import nn
from torch.nn import functional as F

class Spk_Subunit_NN(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_no, H_no, device):
        super().__init__()

        self.T_no = T_no
        self.sub_no = C_syn_e.shape[0]
        
        self.E_no = C_syn_e.shape[1]
        self.I_no = C_syn_i.shape[1]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.device = device
        self.H_no = H_no
        
        ### Cosine Basis ###
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
            self.cos_basis[i] = self.cos_basis[i] + basis
        
        self.W_conv = nn.Parameter(torch.randn(self.H_no*self.sub_no,2,self.cos_basis_no)*0.1 , requires_grad=True)
        self.threshold = nn.Parameter(torch.ones(self.sub_no)*(0) , requires_grad=True)
        self.step = Step.apply
        
        self.W_ff = nn.Parameter(torch.randn(self.sub_no, self.H_no, H_no)*0.1, requires_grad=True)
        self.W_ff2 = nn.Parameter(torch.randn(self.sub_no, self.H_no)*0.1, requires_grad=True)

        self.Tau_root = nn.Parameter(torch.ones(self.sub_no)*2 , requires_grad=True)
        self.W_root = nn.Parameter(torch.ones(self.sub_no)*1 , requires_grad=True)
        self.W_hist = nn.Parameter(torch.zeros(self.sub_no, self.cos_basis_no) , requires_grad=True)
        
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        
        

    def forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        
        C_syn_e = self.C_syn_e
        C_syn_i = self.C_syn_i
        
        S_e_group = torch.matmul(S_e, C_syn_e.T)
        S_i_group = torch.matmul(S_i, C_syn_i.T)

        S = torch.zeros(T_data+self.T_no-1, 2*self.sub_no).to(self.device)
        for s in range(self.sub_no):
            S[-T_data:,2*s] = S[-T_data:,2*s] + S_e_group[:,s]
            S[-T_data:,2*s+1] = S[-T_data:,2*s+1] + S_i_group[:,s]
        S = S.T.unsqueeze(0)
        
        conv_kern = torch.matmul(self.W_conv, self.cos_basis)
        conv_kern = torch.flip(conv_kern, [2])
        conv_out = F.conv1d(S, conv_kern, groups=self.sub_no).squeeze(0).reshape(self.sub_no, self.H_no, -1)
        conv_out = F.leaky_relu(conv_out) #(sub_no, H_no, T_no)
        
        sub_out = torch.zeros(T_data+self.T_no, self.sub_no).to(self.device)
        sub_thresh = torch.zeros(T_data, self.sub_no).to(self.device)
        
        hist_kern = torch.matmul(self.W_hist, self.cos_basis) #(sub_no, T_no)
        
        for t in range(T_data):
            sub_hist = sub_out[t:t+self.T_no].clone() #(T_no, sub_no)
            hist_in = torch.sum(sub_hist.T * hist_kern , 1) #(sub_no)
            conv_in = F.leaky_relu(torch.matmul(conv_out[:,:,t], self.W_ff)) #(sub_no, H_no)
            conv_in_2 = torch.sum(conv_in * self.W_ff2, 1)
            
            sub_in = hist_in + conv_in_2 + self.threshold
            sub_thresh[t] = sub_thresh[t] + sub_in
            sub_out[t+self.T_no] = sub_out[t+self.T_no] + self.step(sub_in)
        
        sub_out = sub_out.T.unsqueeze(0)
        
        t = torch.arange(self.T_no).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
        t_tau = t / torch.exp(self.Tau_root).reshape(-1,1)
        spk_kern = t_tau * torch.exp(-t_tau) * torch.exp(self.W_root).reshape(-1,1)
        spk_kern = torch.flip(spk_kern, [1]).unsqueeze(1)
        
        root_raw = F.conv1d(sub_out, spk_kern, groups=self.sub_no).squeeze(0).T[:-1]
        V = torch.sum(root_raw, 1)

        return V, conv_kern, hist_kern, sub_out.squeeze(0).T[self.T_no:], sub_thresh

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