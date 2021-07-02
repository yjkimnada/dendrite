import torch 
import torch.nn as nn
import torch.nn.functional as F

class Clust_GRU(nn.Module):
    def __init__(self, sub_no, E_no, I_no, H_no, device):
        super().__init__()
        
        self.H_no = H_no
        self.device = device
        self.sub_no = sub_no
        self.E_no = E_no
        self.I_no = I_no
        
        self.E_scale = nn.Parameter(torch.zeros(self.E_no))
        self.I_scale = nn.Parameter(torch.zeros(self.I_no))
        
        self.rnn = nn.ModuleList()
        self.linear = nn.ModuleList()
        for s in range(self.sub_no):
            self.rnn.append(nn.GRU(1, self.H_no, batch_first=True))
            self.linear.append(nn.Linear(self.H_no, 1))

        self.V_o = nn.Parameter(torch.zeros(1))
        
        eps = 1e-8
        u_e = torch.rand(self.sub_no, self.E_no)
        g_e = - torch.log(- torch.log(u_e + eps) + eps)
        u_i = torch.rand(self.sub_no, self.I_no)
        g_i = - torch.log(- torch.log(u_i + eps) + eps)
        self.C_syn_e_raw = nn.Parameter(g_e * 0.01)
        self.C_syn_i_raw = nn.Parameter(g_i * 0.01)
        
        #g_e_help = torch.zeros(self.sub_no, self.E_no)
        #clust_idx = torch.tensor([0,2,3,1])
        #for c in range(4):
            #g_e_help[clust_idx[c], 43+60*c:43+60*(c+1)] = 0.25
        #self.C_syn_e_raw = nn.Parameter(g_e*0.01 + g_e_help)
        
        #self.custom_softmax = CustomSoftmax.apply
        
    def forward(self, S_e, S_i, temp):
        T_data = S_e.shape[1]
        batch_size = S_e.shape[0]
        S_e = S_e * torch.exp(self.E_scale.reshape(1,1,-1))
        S_i = S_i * torch.exp(self.I_scale.reshape(1,1,-1))*(-1)
        
        C_syn_e = F.softmax(self.C_syn_e_raw/temp, 0)
        C_syn_i = F.softmax(self.C_syn_i_raw/temp, 0)
        #C_syn_e = self.custom_softmax(self.C_syn_e_raw)
        #C_syn_i = self.custom_softmax(self.C_syn_i_raw)
        
        S_e_sub = torch.matmul(S_e, C_syn_e.T.unsqueeze(0))
        S_i_sub = torch.matmul(S_i, C_syn_i.T.unsqueeze(0))
        S_sub = S_e_sub + S_i_sub
        
        sub_out = torch.zeros(batch_size, T_data, self.sub_no).to(self.device)
        
        for s in range(self.sub_no):
            rnn_out, _ = self.rnn[s](S_sub[:,:,s].unsqueeze(2))
            lin_out = self.linear[s](rnn_out.reshape(-1,self.H_no)).reshape(batch_size, T_data)
            sub_out[:,:,s] = sub_out[:,:,s] + lin_out

        final = torch.sum(sub_out, 2) + self.V_o
        
        return final, sub_out, C_syn_e, C_syn_i

class CustomSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return F.softmax(input * 10000, 0)
    
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad_output = (sub, syn)
        input_softmax = F.softmax(input, 0).T #(syn, sub)
        # d_softmax = (syn, sub, sub)
        
        d_softmax = input_softmax.unsqueeze(-1) * torch.eye(input_softmax.shape[1]).cuda().unsqueeze(0) - torch.matmul(input_softmax.unsqueeze(-1), input_softmax.unsqueeze(1))

        grad_input = torch.matmul(d_softmax, grad_output.T.unsqueeze(-1)).squeeze(-1) #(syn, sub)
        return grad_input.T