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

from multitask_gp import MultitaskGPModel
from gp_variational_elbo import VariationalELBO

class GP_GLM_Likelihood( _GaussianLikelihoodBase):
    def __init__(self, C_den, sub_no, N, num_tasks,
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
                
        self.C_den = C_den
        self.sub_no = sub_no
        self.N = N
        
        self.decay = nn.Parameter(torch.ones(self.num_tasks) , requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(self.num_tasks) , requires_grad=True)
        self.scale = nn.Parameter(torch.ones(self.num_tasks) , requires_grad=True)
        
        ### Between Subunit Parameters ###
        self.W_log = nn.Parameter(torch.zeros(self.sub_no) , requires_grad=True) # POSITIVE

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
    
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
             
        decay_dist = torch.arange(self.N).reshape(1,-1).repeat(self.num_tasks,1).cuda() + self.shift.reshape(-1,1)
        decay_dist_2 = decay_dist ** 2
        decay_factor = self.scale.reshape(-1,1)**2 * torch.exp(-decay_dist_2 / self.decay.reshape(-1,1)**2)
        
        all_F = all_F * decay_factor
        
        T = S_e.shape[0]
        
        F_e = all_F[:self.sub_no].unsqueeze(1)
        F_i = all_F[self.sub_no:].unsqueeze(1)
        flip_F_e = torch.flip(F_e, [2])
        flip_F_i = torch.flip(F_i, [2])
        
        pad_S_e = torch.zeros(T + self.N-1, self.sub_no).cuda()
        pad_S_i = torch.zeros(T + self.N-1, self.sub_no).cuda()
        pad_S_e[-T:] = pad_S_e[-T:] + S_e
        pad_S_i[-T:] = pad_S_i[-T:] + S_i
        pad_S_e = pad_S_e.T.unsqueeze(0)
        pad_S_i = pad_S_i.T.unsqueeze(0)

        filtered_e = F.conv1d(pad_S_e, flip_F_e, padding=0, groups=self.sub_no).squeeze(0).T
        filtered_i = F.conv1d(pad_S_i, flip_F_i, padding=0, groups=self.sub_no).squeeze(0).T

        syn_in = filtered_e + filtered_i

        #----- Combine Subunits -----#

        sub_out = torch.zeros(T, self.sub_no).cuda()
        
        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                nonlin_out = torch.tanh(syn_in[:,sub_idx] + self.Theta[sub_idx]) # (T_data,) 
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + nonlin_out
            else:
                leaf_in = sub_out[:,leaf_idx] * torch.exp(self.W_log[leaf_idx]) # (T_data,)
                nonlin_in = syn_in[:,sub_idx] + torch.sum(leaf_in, 1) + self.Theta[sub_idx]# (T_data,)
                nonlin_out = torch.tanh(nonlin_in)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + nonlin_out
        
        final_voltage = sub_out[:,0]*torch.exp(self.W_log[0]) + self.V_o

        res = torch.var(target - final_voltage)
        
        return res, final_voltage, all_F


class GP_GLM(nn.Module):
    def __init__(self, C_den, E_no, I_no, T_no, sparse_no, batch_size, greedy, C_syn_e, C_syn_i):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.E_no = E_no
        self.I_no = I_no
        self.greedy = greedy
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.num_tasks = self.sub_no * 2
        self.sparse_no = sparse_no
        self.batch_size = batch_size

        self.filter_model = MultitaskGPModel(self.num_tasks, self.sparse_no , self.T_no)
        self.likelihood = GP_GLM_Likelihood(self.C_den, self.sub_no, self.T_no, self.num_tasks)
        self.mll = VariationalELBO(self.likelihood, self.filter_model, num_data=self.batch_size)

        ### C_syn Parameters ###
        if self.greedy == True:
            self.C_syn_e_logit = nn.Parameter(torch.ones(self.sub_no, self.E_no), requires_grad=True)
            self.C_syn_i_logit = nn.Parameter(torch.ones(self.sub_no, self.I_no), requires_grad=True)

    def forward(self, V_ref, S_e, S_i, temp, test):
        T_data = S_e.shape[0]

        if self.greedy == True:
            if test == True:
                C_syn_e = torch.zeros_like(self.C_syn_e_logit).cuda()
                C_syn_i = torch.zeros_like(self.C_syn_i_logit).cuda()
                for i in range(C_syn_e.shape[1]):
                    idx = torch.argmax(self.C_syn_e_logit[:,i])
                    C_syn_e[idx,i] = 1
                for i in range(C_syn_i.shape[1]):
                    idx = torch.argmax(self.C_syn_i_logit[:,i])
                    C_syn_i[idx,i] = 1
            
            elif test == False:
                u_e = torch.rand_like(self.C_syn_e_logit).cuda()
                u_i = torch.rand_like(self.C_syn_i_logit).cuda()
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

        x_in = torch.arange(self.T_no).cuda()
        filter_output = self.filter_model(x_in)
        var_loss, V_pred, out_filters = self.mll(filter_output, V_ref, S_e, S_i)

        return var_loss, V_pred, out_filters, C_syn_e, C_syn_i