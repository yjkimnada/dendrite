import torch
from torch import nn
from torch.nn import functional as F

class Sub_Clust_Cos_GLM(nn.Module):
    def __init__(self, sub_no, E_no, I_no, T_no, hid_no, device):
        super().__init__()

        self.T_no = T_no
        self.sub_no = sub_no
        self.E_no = E_no
        self.I_no = I_no
        self.device = device
        self.hid_no = hid_no

        self.E_scale = nn.Parameter(torch.zeros(self.E_no))
        self.I_scale = nn.Parameter(torch.zeros(self.I_no))
        
        self.cos_basis_no = 24
        self.scale = 6
        self.shift = 1
        
        self.kern_basis = torch.zeros(self.cos_basis_no, T_no).to(device)
        for i in range(self.cos_basis_no):
            phi = 1.5707963267948966*i
            xmin = phi - 3.141592653589793
            xmax = phi + 3.141592653589793

            x_in = torch.arange(0, T_no, 1)
            raw_cos = self.scale  * torch.log(x_in + self.shift + 1e-7)

            basis = 0.5*torch.cos(raw_cos - phi) + 0.5
            basis[raw_cos < xmin] = 0.0
            basis[raw_cos > xmax] = 0.0
            self.kern_basis[i] = basis
        
        self.W_e_layer1 = nn.Parameter(torch.randn(self.sub_no*hid_no , self.cos_basis_no)*0.01)        
        self.W_i_layer1 = nn.Parameter(torch.randn(self.sub_no*hid_no , self.cos_basis_no)*0.01)
        self.W_layer2 = nn.Parameter(torch.ones(self.sub_no, self.hid_no)*(-1))
        self.b_layer1 = nn.Parameter(torch.zeros(self.sub_no*self.hid_no))
        
        self.C_syn_e_raw = nn.Parameter(torch.randn(self.sub_no, self.E_no)*0.01)
        self.C_syn_i_raw = nn.Parameter(torch.randn(self.sub_no, self.I_no)*0.01)
        
        self.V_o = nn.Parameter(torch.zeros(1))
        self.custom_softmax = CustomSoftmax.apply

    def forward(self, S_e, S_i, temp, test, g_e, g_i):
        # S is (batch, T, E)
        T_data = S_e.shape[1]
        batch = S_e.shape[0]
        
        if test == True:
            C_syn_e = F.softmax(self.C_syn_e_raw * 10000,0)
            C_syn_i = F.softmax(self.C_syn_i_raw * 10000,0)
        elif test == False:
            #eps = 1e-8
            #u_e = torch.rand_like(self.C_syn_e_raw)
            #u_i = torch.rand_like(self.C_syn_i_raw)
            #g_e = - torch.log(- torch.log(u_e + eps) + eps)
            #g_i = - torch.log(- torch.log(u_i + eps) + eps)
            
            ##C_syn_e = F.softmax((self.C_syn_e_raw + g_e), 0)
            ##C_syn_i = F.softmax((self.C_syn_i_raw + g_i), 0)
            #C_syn_e = F.softmax((self.C_syn_e_raw + g_e) / temp, 0)
            #C_syn_i = F.softmax((self.C_syn_i_raw + g_i) / temp, 0)
            #C_syn_e = self.custom_softmax(self.C_syn_e_raw / temp)
            #C_syn_i = self.custom_softmax(self.C_syn_i_raw / temp)
            #C_syn_e = self.custom_softmax(self.C_syn_e_raw)
            #C_syn_i = self.custom_softmax(self.C_syn_i_raw)
            #C_syn_e = self.custom_softmax((self.C_syn_e_raw + g_e) / temp)
            #C_syn_i = self.custom_softmax((self.C_syn_i_raw + g_i) / temp)
            C_syn_e = self.custom_softmax((self.C_syn_e_raw + g_e))
            C_syn_i = self.custom_softmax((self.C_syn_i_raw + g_i))

        S_e = S_e * torch.exp(self.E_scale.reshape(1,1,-1))
        S_i = S_i * torch.exp(self.I_scale.reshape(1,1,-1))
        syn_e = torch.matmul(S_e, C_syn_e.T.unsqueeze(0))
        syn_i = torch.matmul(S_i, C_syn_i.T.unsqueeze(0))

        pad_syn_e = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(batch, T_data + self.T_no-1, self.sub_no).to(self.device)
        pad_syn_e[:,-T_data:] = pad_syn_e[:,-T_data:] + syn_e
        pad_syn_i[:,-T_data:] = pad_syn_i[:,-T_data:] + syn_i
        pad_syn_e = pad_syn_e.permute(0,2,1)
        pad_syn_i = pad_syn_i.permute(0,2,1)
        
        layer1_e_kern = torch.matmul(self.W_e_layer1, self.kern_basis) # (sub*H, T_no)
        layer1_i_kern = torch.matmul(self.W_i_layer1, self.kern_basis) # (sub*H, T_no)
        layer1_e_kern = torch.flip(layer1_e_kern, [1]).unsqueeze(1)
        layer1_i_kern = torch.flip(layer1_i_kern, [1]).unsqueeze(1)
        
        layer1_e_conv = F.conv1d(pad_syn_e, layer1_e_kern, groups=self.sub_no)
        layer1_i_conv = F.conv1d(pad_syn_i, layer1_i_kern, groups=self.sub_no)
        layer1_out = torch.tanh(layer1_e_conv + layer1_i_conv + self.b_layer1.reshape(1,-1,1))
        
        sub_out = F.conv1d(layer1_out, torch.exp(self.W_layer2).unsqueeze(-1), groups=self.sub_no).permute(0,2,1)
        final = torch.sum(sub_out, -1) + self.V_o

        return final, sub_out, C_syn_e, C_syn_i
        

class CustomSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return F.softmax(input * 1000, 0)
    
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad_output = (sub, syn)
        input_softmax = F.softmax(input, 0).T #(syn, sub)
        # d_softmax = (syn, sub, sub)
        
        d_softmax = input_softmax.unsqueeze(-1) * torch.eye(input_softmax.shape[1]).cuda().unsqueeze(0) - torch.matmul(input_softmax.unsqueeze(-1), input_softmax.unsqueeze(1))

        grad_input = torch.matmul(d_softmax, grad_output.T.unsqueeze(-1)).squeeze(-1) #(syn, sub)
        return grad_input.T
        
        
      