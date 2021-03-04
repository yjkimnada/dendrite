import torch
from torch import nn
from torch.nn import functional as F

class Tree_Stoch_GLM(nn.Module):
    def __init__(self, sub_no, E_no, I_no, T_no, device):
        super().__init__()

        self.T_no = T_no
        self.sub_no = sub_no
        self.E_no = E_no
        self.I_no = I_no
        self.device = device

        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.rand(self.sub_no, self.T_no, 2)*0.01, requires_grad=True)
        
        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no)*0 , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### C_syn Parameters ###
        self.C_syn_e_logit = nn.Parameter(torch.zeros(self.sub_no, self.E_no), requires_grad=True)
        self.C_syn_i_logit = nn.Parameter(torch.zeros(self.sub_no, self.I_no), requires_grad=True)
        
        #self.C_den_list = []
        #for s in range(self.sub_no-2):
            #self.C_den_list.append(nn.Parameter(torch.zeros(s+2) , requires_grad=True))
        self.C_den_raw = nn.Parameter(torch.zeros(self.sub_no*(self.sub_no-1)//2-1) , requires_grad=True)
    
    def forward(self, S_e, S_i, temp, test):
        T_data = S_e.shape[0] 

        if test == True:
            C_syn_e = torch.zeros_like(self.C_syn_e_logit).to(self.device)
            C_syn_i = torch.zeros_like(self.C_syn_i_logit).to(self.device)
            C_den = torch.zeros(self.sub_no, self.sub_no).to(self.device)
            C_den[0,1] = 1
            den_count = 0

            for i in range(C_syn_e.shape[1]):
                idx = torch.argmax(self.C_syn_e_logit[:,i])
                C_syn_e[idx,i] = 1
            for i in range(C_syn_i.shape[1]):
                idx = torch.argmax(self.C_syn_i_logit[:,i])
                C_syn_i[idx,i] = 1
            for i in range(self.sub_no-2):
                idx = torch.argmax(self.C_den_raw[den_count:den_count+i+2])
                C_den[idx,i+2] = 1
                den_count += i+2
                #idx = torch.argmax(self.C_den_list[i])
                #C_den[idx, i+2] = 1

        elif test == False:
            eps = 1e-7
            u_e = torch.rand_like(self.C_syn_e_logit)
            u_i = torch.rand_like(self.C_syn_i_logit)
            u_den = torch.rand_like(self.C_den_raw)
            
            g_e = - torch.log(- torch.log(u_e + eps) + eps)
            g_i = - torch.log(- torch.log(u_i + eps) + eps)
            g_den = - torch.log(- torch.log(u_den + eps) + eps)
            
            C_syn_e = F.softmax((self.C_syn_e_logit + g_e) / temp, dim=0)
            C_syn_i = F.softmax((self.C_syn_i_logit + g_i) / temp, dim=0)
            C_den = torch.zeros(self.sub_no, self.sub_no).to(self.device)
            C_den[0,1] = 1
            den_count = 0

            for i in range(self.sub_no-2):
                #C_den[:i+2,i+2] = C_den[:i+2,i+2] + F.softmax((self.C_den_list[i]) / temp , dim=0)
                C_den[:i+2,i+2] = C_den[:i+2,i+2] + F.softmax(self.C_den_raw[den_count:den_count+i+2] + g_den[den_count:den_count+i+2] / temp , dim=0)
                den_count += i+2

        syn_e = torch.matmul(S_e, C_syn_e.T)
        syn_i = torch.matmul(S_i, C_syn_i.T)

        e_kern = torch.flip(self.W_syn[:,:,0], [1])
        i_kern = torch.flip(self.W_syn[:,:,1], [1])
        e_kern = e_kern.unsqueeze(1)
        i_kern = i_kern.unsqueeze(1)

        pad_syn_e = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_i = torch.zeros(T_data + self.T_no - 1, self.sub_no).to(self.device)
        pad_syn_e[-T_data:] = pad_syn_e[-T_data:] + syn_e
        pad_syn_i[-T_data:] = pad_syn_i[-T_data:] + syn_i
        pad_syn_e = pad_syn_e.T.unsqueeze(0)
        pad_syn_i = pad_syn_i.T.unsqueeze(0)

        filtered_e = F.conv1d(pad_syn_e, e_kern, padding=0, groups=self.sub_no).squeeze(0).T
        filtered_i = F.conv1d(pad_syn_i, i_kern, padding=0,  groups=self.sub_no).squeeze(0).T
 
        syn_in = filtered_e + filtered_i

        sub_out = torch.zeros(T_data+1, self.sub_no).to(self.device)
        
        for t in range(T_data):
            past_in = sub_out[t].clone()
            leaf_in = torch.matmul(C_den, torch.exp(self.W_sub) * past_in)
            now_in = syn_in[t] + self.Theta + leaf_in
            now_out = torch.tanh(now_in)
            sub_out[t+1] = sub_out[t+1] + now_out
        
        final = sub_out[1:,0]*torch.exp(self.W_sub[0]) + self.V_o

        e_kern_out = torch.flip(e_kern, [2]).squeeze(1)
        i_kern_out = torch.flip(i_kern, [2]).squeeze(1)
        out_filters = torch.vstack((e_kern_out, i_kern_out))

        return final, out_filters, C_den, C_syn_e, C_syn_i

