import numpy as np
import torch 
import torch.nn as nn
from tqdm import tnrange
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
import scipy
import time

#from models.sub_clust_cos_glm import Sub_Clust_Cos_GLM
from models.clust_gru import Clust_GRU

base_dir = "/scratch/yjk27/"
experiment = "clust12-20"
cell_type = "CA1"
E_neural_file = "Espikes_neural.npz"
I_neural_file = "Ispikes_neural.npz"
#V_file = "vdata_T10_Ne2000_gA0.6_tauA1_gN0.8_Ni200_gG0.1_gB0.1_noDendNa_Er0.5_Ir7.4_random_NR_rep1000_stimseed1.npy"
V_file = "V_diff_stimseed1.npy"

E_neural = scipy.sparse.load_npz(base_dir+cell_type+"_"+experiment+"/data/"+E_neural_file)
I_neural = scipy.sparse.load_npz(base_dir+cell_type+"_"+experiment+"/data/"+I_neural_file)
#V = np.load(base_dir+cell_type+"_"+experiment+"/data/"+V_file)[:,:50000].flatten()
V = np.load(base_dir+cell_type+"_"+experiment+"/data/"+V_file)
V = torch.from_numpy(V)
#V -= torch.mean(V)

T_train = 980 * 1000 * 50
T_test = 1 * 1000 * 50
H_no = 20
sub_no = 13
E_no = 2000
I_no = 200
#E_no = e_idx.shape[0]
#I_no = i_idx.shape[0]
T_no = 500
device = torch.device("cuda:4")

increment = 50
batch_length = 50000
batch_size = 5
iter_no = 9800*2
epoch_no = iter_no*batch_length*batch_size//T_train

V_train = V[:T_train].float()
#V_test = V[-50000:].to(device).float()
V_test = V[T_train:T_train+T_test].to(device).float()

#test_E_neural = E_neural[-50000:].toarray()
#test_I_neural = I_neural[-50000:].toarray()
test_E_neural = E_neural[T_train:T_train+T_test].toarray()
test_I_neural = I_neural[T_train:T_train+T_test].toarray()
train_E_neural = E_neural[:T_train]
train_I_neural = I_neural[:T_train]

test_E_neural = torch.from_numpy(test_E_neural).float().to(device)
test_I_neural = torch.from_numpy(test_I_neural).float().to(device)

train_idx = np.empty((epoch_no, T_train//batch_length//batch_size))
for i in range(epoch_no):
    part_idx = np.arange(0, T_train, batch_length*batch_size)
    np.random.shuffle(part_idx)
    train_idx[i] = part_idx
train_idx = train_idx.flatten()
train_idx = torch.from_numpy(train_idx)

#model = Sub_Clust_Cos_GLM(sub_no, E_no, I_no, T_no, H_no, device)
model = Clust_GRU(sub_no, E_no, I_no, H_no, device)

syn_params = []
rest_params = []

for name, params in model.named_parameters():
    if (name == "C_syn_e_raw"):
        syn_params.append(params)
    elif (name == "C_syn_i_raw"):
        syn_params.append(params)
    else:
        rest_params.append(params)

# GLM
#optimizer = torch.optim.Adam(rest_params, lr = 0.005/(1.03**100))
#syn_optimizer = torch.optim.Adam(syn_params, lr = 0.005/(1.0**100))
#milestones = np.arange(increment-1, increment*100, increment)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=1.03)
#syn_milestones = np.arange(increment-1, increment*100, increment)
#syn_scheduler = torch.optim.lr_scheduler.MultiStepLR(syn_optimizer, milestones=syn_milestones, gamma=1)

# GRU
optimizer = torch.optim.Adam(rest_params, lr = 0.005)
syn_optimizer = torch.optim.Adam(syn_params, lr = 0.005/(1**100))
milestones = np.arange(increment-1, increment*100, increment)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=1)
syn_milestones = np.arange(increment-1, increment*100, increment)
syn_scheduler = torch.optim.lr_scheduler.MultiStepLR(syn_optimizer, milestones=syn_milestones, gamma=0.961)

model.to(device).float()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

print(sum(p.numel() for p in syn_params if p.requires_grad))
print(sum(p.numel() for p in rest_params if p.requires_grad))
print(milestones.shape)

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

score_list = []
temp_list = np.logspace(0,-3,100)
temp_count = 0

for i in tnrange(iter_no):
    s = time.time()
    model.train()
    optimizer.zero_grad()
    syn_optimizer.zero_grad()
           
    with torch.no_grad():
        model.C_syn_e_raw.copy_(model.C_syn_e_raw - torch.mean(model.C_syn_e_raw, 0).reshape(1,-1))
        model.C_syn_i_raw.copy_(model.C_syn_i_raw - torch.mean(model.C_syn_i_raw, 0).reshape(1,-1))
    
    if (temp_count < 99) & (i%increment == increment-1):
        temp_count += 1
    temp = torch.tensor([temp_list[temp_count]]).to(device).float()
    
    batch_idx = train_idx[i].long()
    batch_E_neural = train_E_neural[batch_idx : batch_idx+batch_length*batch_size].toarray().reshape(batch_size, batch_length, -1)
    batch_I_neural = train_I_neural[batch_idx : batch_idx+batch_length*batch_size].toarray().reshape(batch_size, batch_length, -1)
    batch_E_neural = torch.from_numpy(batch_E_neural).float().to(device)
    batch_I_neural = torch.from_numpy(batch_I_neural).float().to(device)
    batch_V = V_train[batch_idx : batch_idx+batch_length*batch_size].reshape(batch_size, -1).to(device)
    
    V_pred, _, C_syn_e, C_syn_i  = model(batch_E_neural, batch_I_neural, temp)
    
    loss = torch.mean((V_pred - batch_V)**2)
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    syn_optimizer.step()
    syn_scheduler.step()
    time_step = time.time() - s
    
    if (i%50 == 49) or (i == 0):
        model.eval()
        test_V_pred, test_sub_out, test_C_syn_e, test_C_syn_i = model(test_E_neural.unsqueeze(0), test_I_neural.unsqueeze(0), 0.0001)
        test_V_pred = test_V_pred.flatten()
                 
        test_score = explained_variance_score(V_test.cpu().detach().numpy(), test_V_pred.cpu().detach().numpy())
        train_score = explained_variance_score(batch_V[0].cpu().detach().numpy(), V_pred[0].cpu().detach().numpy())
        score_list.append(test_score)
        
        C_syn_idx = torch.argmax(test_C_syn_e[:,880:1120], 0).float()
        
        mode1, mode1_idx = torch.mode(C_syn_idx[:20])
        mode2, mode2_idx = torch.mode(C_syn_idx[20:40])
        mode3, mode3_idx = torch.mode(C_syn_idx[40:60])
        mode4, mode4_idx = torch.mode(C_syn_idx[60:80])
        mode5, mode5_idx = torch.mode(C_syn_idx[80:100])
        mode6, mode6_idx = torch.mode(C_syn_idx[100:120])
        mode7, mode7_idx = torch.mode(C_syn_idx[120:140])
        mode8, mode8_idx = torch.mode(C_syn_idx[140:160])
        mode9, mode9_idx = torch.mode(C_syn_idx[160:180])
        mode10, mode10_idx = torch.mode(C_syn_idx[180:200])
        mode11, mode11_idx = torch.mode(C_syn_idx[200:220])
        mode12, mode12_idx = torch.mode(C_syn_idx[220:240])
        mode1_no = torch.numel(torch.where(C_syn_idx[:20] == mode1)[0])
        mode2_no = torch.numel(torch.where(C_syn_idx[20:40] == mode2)[0])
        mode3_no = torch.numel(torch.where(C_syn_idx[40:60] == mode3)[0])
        mode4_no = torch.numel(torch.where(C_syn_idx[60:80] == mode4)[0])
        mode5_no = torch.numel(torch.where(C_syn_idx[80:100] == mode5)[0])
        mode6_no = torch.numel(torch.where(C_syn_idx[100:120] == mode6)[0])
        mode7_no = torch.numel(torch.where(C_syn_idx[120:140] == mode7)[0])
        mode8_no = torch.numel(torch.where(C_syn_idx[140:160] == mode8)[0])
        mode9_no = torch.numel(torch.where(C_syn_idx[160:180] == mode9)[0])
        mode10_no = torch.numel(torch.where(C_syn_idx[180:200] == mode10)[0])
        mode11_no = torch.numel(torch.where(C_syn_idx[200:220] == mode11)[0])
        mode12_no = torch.numel(torch.where(C_syn_idx[220:240] == mode12)[0])
                        
        print(i, np.round(test_score,6), np.round(train_score,6), time_step)
        print(mode1.item(), mode2.item(), mode3.item(), mode4.item(),
             mode5.item(), mode6.item(), mode7.item(), mode8.item(),
             mode9.item(), mode10.item(), mode11.item(), mode12.item())
        print(mode1_no, mode2_no, mode3_no, mode4_no,
             mode5_no, mode6_no, mode7_no, mode8_no,
             mode9_no, mode10_no, mode11_no, mode12_no)
        print("------------------------")
        
        torch.save(model.state_dict(), "/scratch/yjk27/CA1_clust12-20/clust/gru_s13_h20_i"+str(i)+".pt")