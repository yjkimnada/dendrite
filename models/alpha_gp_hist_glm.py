import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import gpytorch
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.lazy import ZeroLazyTensor
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise, HomoskedasticNoise, Noise
from typing import Any, Optional

from gpytorch.constraints import GreaterThan
from gpytorch.distributions import base_distributions
from gpytorch.functions import add_diag
from gpytorch.lazy import (
    BlockDiagLazyTensor,
    DiagLazyTensor,
    KroneckerProductLazyTensor,
    MatmulLazyTensor,
    RootLazyTensor,
    lazify,
)
from gpytorch.likelihoods import Likelihood, _GaussianLikelihoodBase
from gpytorch.utils.warnings import OldVersionWarning
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise

from models.multitask_gp import MultitaskGPModel
from models.gp_variational_elbo import VariationalELBO

class Alpha_GP_Hist_GLM_Likelihood( _GaussianLikelihoodBase):
    def __init__(self, C_den, sub_no, N, num_tasks, device,
        rank=0,
        task_correlation_prior=None,
        batch_shape=torch.Size(),
        noise_prior=None,
        noise_constraint=None):
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        noise_covar = MultitaskHomoskedasticNoise(
            num_tasks=num_tasks, noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        super().__init__(noise_covar=noise_covar)
        if rank != 0:
            if rank > num_tasks:
                raise ValueError(f"Cannot have rank ({rank}) greater than num_tasks ({num_tasks})")
            tidcs = torch.tril_indices(num_tasks, rank, dtype=torch.long)
            self.tidcs = tidcs[:, 1:]  # (1, 1) must be 1.0, no need to parameterize this
            task_noise_corr = torch.randn(*batch_shape, self.tidcs.size(-1))
            self.register_parameter("task_noise_corr", torch.nn.Parameter(task_noise_corr))
            if task_correlation_prior is not None:
                self.register_prior(
                    "MultitaskErrorCorrelationPrior", task_correlation_prior, lambda: self._eval_corr_matrix
                )
        elif task_correlation_prior is not None:
            raise ValueError("Can only specify task_correlation_prior if rank>0")
        
        self.num_tasks = num_tasks
        self.rank = rank
        self.device = device
                
        self.C_den = C_den
        self.sub_no = sub_no
        self.N = N
        
        self.decay = nn.Parameter(torch.ones(self.num_tasks) , requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(self.num_tasks) , requires_grad=True)
        self.scale = nn.Parameter(torch.ones(self.num_tasks) , requires_grad=True)
        
        ### Between Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no) , requires_grad=True) # POSITIVE

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        
        ### Synapse Parameters ###
        self.syn_basis_no = 1
        self.W_syn_raw = torch.rand(self.sub_no,self.syn_basis_no, 2) * 0.1
        self.W_syn_raw[:,:,1] *= -1
        self.W_syn = nn.Parameter(self.W_syn_raw, requires_grad=True)
        self.Tau_syn_raw = torch.arange(1.1,2*self.syn_basis_no,2).reshape(1,-1,1).repeat(self.sub_no,1,2).float()
        self.Tau_syn = nn.Parameter(self.Tau_syn_raw, requires_grad=True)
        self.Delta_syn = nn.Parameter(torch.zeros(self.sub_no,self.syn_basis_no, 2), requires_grad=True)
    
    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _shaped_noise_covar(self, base_shape, *params):
        if len(base_shape) >= 2:
            *batch_shape, n, _ = base_shape
        else:
            *batch_shape, n = base_shape

        # compute the noise covariance
        if len(params) > 0:
            shape = None
        else:
            shape = base_shape if len(base_shape) == 1 else base_shape[:-1]
        noise_covar = self.noise_covar(*params, shape=shape)

        if self.rank > 0:
            # if rank > 0, compute the task correlation matrix
            # TODO: This is inefficient, change repeat so it can repeat LazyTensors w/ multiple batch dimensions
            task_corr = self._eval_corr_matrix()
            exp_shape = torch.Size([*batch_shape, n]) + task_corr.shape[-2:]
            task_corr_exp = lazify(task_corr.unsqueeze(-3).expand(exp_shape))
            noise_sem = noise_covar.sqrt()
            task_covar_blocks = MatmulLazyTensor(MatmulLazyTensor(noise_sem, task_corr_exp), noise_sem)
        else:
            # otherwise tasks are uncorrelated
            task_covar_blocks = noise_covar

        if len(batch_shape) == 1:
            # TODO: Properly support general batch shapes in BlockDiagLazyTensor (no shape arithmetic)
            tcb_eval = task_covar_blocks.evaluate()
            task_covar = BlockDiagLazyTensor(lazify(tcb_eval), block_dim=-3)
        else:
            task_covar = BlockDiagLazyTensor(task_covar_blocks)

        return task_covar
        
    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, S_e, S_i, *params: Any, **kwargs: Any) -> Tensor:        
        all_F = input.mean.T
             
        decay_dist = torch.arange(self.N).reshape(1,-1).repeat(self.num_tasks,1).to(self.device) + self.shift.reshape(-1,1)
        decay_dist_2 = decay_dist ** 2
        decay_factor = self.scale.reshape(-1,1)**2 * torch.exp(-decay_dist_2 / self.decay.reshape(-1,1)**2)
        
        all_F = all_F * decay_factor
        
        T = S_e.shape[0]
        
        full_e_kern = torch.zeros(self.sub_no, self.N).to(self.device)
        full_i_kern = torch.zeros(self.sub_no, self.N).to(self.device)
        
        for b in range(self.syn_basis_no):
            t_raw_e = torch.arange(self.N).reshape(1,-1).repeat(self.sub_no,1).to(self.device)
            t_raw_i = torch.arange(self.N).reshape(1,-1).repeat(self.sub_no,1).to(self.device)

            t_e = t_raw_e - self.Delta_syn[:,b,0].reshape(-1,1)
            t_i = t_raw_i - self.Delta_syn[:,b,1].reshape(-1,1)
            t_e[t_e < 0.0] = 0.0
            t_i[t_i < 0.0] = 0.0 

            tau_e = torch.exp(self.Tau_syn[:,b,0]).reshape(-1,1)
            tau_i = torch.exp(self.Tau_syn[:,b,1]).reshape(-1,1)
            t_e_tau = t_e / tau_e
            t_i_tau = t_i / tau_i
            part_e_kern = t_e_tau * torch.exp(-t_e_tau) * self.W_syn[:,b,0].reshape(-1,1)**2
            part_i_kern = t_i_tau * torch.exp(-t_i_tau) * self.W_syn[:,b,1].reshape(-1,1)**2*(-1)
            full_e_kern = full_e_kern + part_e_kern
            full_i_kern = full_i_kern + part_i_kern

        flip_full_e_kern = torch.flip(full_e_kern, [1])
        flip_full_i_kern = torch.flip(full_i_kern, [1])
        flip_full_e_kern = flip_full_e_kern.unsqueeze(1)
        flip_full_i_kern = flip_full_i_kern.unsqueeze(1)
        
        pad_S_e = torch.zeros(T + self.N-1, self.sub_no).to(self.device)
        pad_S_i = torch.zeros(T + self.N-1, self.sub_no).to(self.device)
        pad_S_e[-T:] = pad_S_e[-T:] + S_e
        pad_S_i[-T:] = pad_S_i[-T:] + S_i
        pad_S_e = pad_S_e.T.unsqueeze(0)
        pad_S_i = pad_S_i.T.unsqueeze(0)

        filtered_e = F.conv1d(pad_S_e, flip_full_e_kern, padding=0, groups=self.sub_no).squeeze(0).T
        filtered_i = F.conv1d(pad_S_i, flip_full_i_kern, padding=0, groups=self.sub_no).squeeze(0).T

        syn_in = filtered_e + filtered_i

        #----- Combine Subunits -----#
        
        sub_out = torch.zeros(T+self.N, self.sub_no).to(self.device)
        F_hist = all_F
        F_hist = torch.flip(F_hist, [1])

        for t in range(T):
            sub_hist = sub_out[t:t+self.N,:].clone() 
            sub_hist_in = torch.sum(sub_hist.T * F_hist, 1).flatten()
            
            sub_prop = torch.matmul(sub_out[self.N+t-1].clone()*self.W_sub**2 , self.C_den.T)
            Y_out = torch.tanh(syn_in[t] + sub_prop + self.Theta + sub_hist_in)
            sub_out[t+self.N] = sub_out[t+self.N] + Y_out        
        
        final_voltage = sub_out[self.N:,0]*self.W_sub[0]**2 + self.V_o
        res = torch.var(target - final_voltage)
        
        out_F_e = full_e_kern.squeeze(1)
        out_F_i = full_i_kern.squeeze(1)
        out_F_hist = F_hist
        out_filters = torch.vstack((out_F_e, out_F_i, out_F_hist))
        
        return res, final_voltage, out_filters


class Alpha_GP_Hist_GLM(nn.Module):
    def __init__(self, C_den, E_no, I_no, T_no, sparse_no, batch_size, greedy, C_syn_e, C_syn_i, device):
        super().__init__()

        self.C_den = C_den.float().to(device)
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.E_no = E_no
        self.I_no = I_no
        self.greedy = greedy
        self.C_syn_e = C_syn_e.to(device)
        self.C_syn_i = C_syn_i.to(device)
        self.num_tasks = self.sub_no * 1
        self.sparse_no = sparse_no
        self.batch_size = batch_size
        self.device = device

        self.filter_model = MultitaskGPModel(self.num_tasks, self.sparse_no , self.T_no)
        self.likelihood = Alpha_GP_Hist_GLM_Likelihood(self.C_den, self.sub_no, self.T_no, self.num_tasks, device)
        self.mll = VariationalELBO(self.likelihood, self.filter_model, num_data=self.batch_size)

        ### C_syn Parameters ###
        if self.greedy == True:
            self.C_syn_e_logit = nn.Parameter(torch.ones(self.sub_no, self.E_no), requires_grad=True)
            self.C_syn_i_logit = nn.Parameter(torch.ones(self.sub_no, self.I_no), requires_grad=True)

    def forward(self, V_ref, S_e, S_i, temp, test):
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

        x_in = torch.arange(self.T_no).to(self.device)
        filter_output = self.filter_model(x_in)
        var_loss, V_pred, out_filters = self.mll(filter_output, V_ref, syn_e, syn_i)

        return var_loss, V_pred, out_filters, C_syn_e, C_syn_i