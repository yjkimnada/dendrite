import torch
from torch import nn
from torch.nn import functional as F

class RSLDS(nn.Module):
    def __init__(self, T, N, K, H, batch, C_syn, device):
        super().__init__()

        self.K = K
        self.C_syn = C_syn
        self.in_no = C_syn.shape[1]
        self.device = device
        self.N = N
        self.H = H
        self.batch = batch
        self.T = T

        self.W_zu = nn.Parameter(torch.randn(N, K, K))
        self.W_zx = nn.Parameter(torch.randn(N, K, K, H))
        self.W_xx = nn.Parameter(torch.randn(N, K, H, H))
        self.W_xu = nn.Parameter(torch.randn(N, K, H))
        self.W_yx = nn.Parameter(torch.randn(N * H))

        self.b_z = nn.Parameter(torch.randn(N, K, K))
        self.b_x = nn.Parameter(torch.randn(N, K, H))

        self.X_inf = nn.Parameter(torch.zeros(batch, T, N, H))
        self.X_scale = nn.Parameter(torch.ones(N * H))
        self.U_scale = nn.Parameter(torch.ones(self.in_no))

    def normalize_logprob(self, logmat, dim, temperature):
        logmat = logmat / temperature
        normalizer = torch.logsumexp(logmat, dim, True)
        return logmat - normalizer, normalizer

    def forward_backward(self, log_a, log_b, logprob_z1):
        # A is (batch, t, n, k, k); p(z_t | z_{t-1})
        # B is (batch, t, n, k); p(x_t | x_{t-1}, z_t, u_t)
        # z1 is (k); p(z_1)

        T = log_a.shape[1]
        device = log_a.device
        batch = log_a.shape[0]
        N = log_a.shape[2]
        K = log_a.shape[3]

        ### FORWARD ###
        forward_probs = torch.zeros(batch, T, N, K).to(device)
        forward_norms = torch.zeros(batch, T, N, 1).to(device)

        forward_init_updates_unnorm = logprob_z1.reshape(1,1,-1) + log_b[:,0,:,:]
        forward_init_updates = self.normalize_logprob(forward_init_updates_unnorm, -1, 1) # (batch, n, k) and (batch, n, 1)
        forward_probs[:,0,:,:] = forward_init_updates[0]
        forward_norms[:,0,:,:] = forward_init_updates[1]

        forward_prev_prob = forward_init_updates[0] # first forward probability normalized -> (batch, n, 1, k)

        for t in range(T - 1):
            bi_t = log_b[:,t+1,:,:] # log p(x_t | x_{t-1}, z_t, u_t); (batch, n, k) -> (batch, n, k, 1)
            aij_t = log_a[:,t+1,:,:,:] # p(z_t | z_{t-1}); (batch, n, k, k)

            forward_current_updates_in = bi_t.unsqueeze(3) + aij_t + forward_prev_prob.unsqueeze(2) # (batch, n, k, k)
            forward_current_updates_unnorm = torch.logsumexp(forward_current_updates_in, -1) # (batch, n, k)
            forward_current_updates = self.normalize_logprob(forward_current_updates_unnorm, -1, 1) # (batch, n, k) and (batch, n, 1)

            forward_prev_prob = forward_current_updates[0]
            forward_probs[:,t+1,:,:] = forward_current_updates[0]
            forward_norms[:,t+1,:,:] = forward_current_updates[1]

        ### BACKWARD ###
        backward_probs = torch.zeros(batch, T, N, K).to(device)
        backward_norms = torch.zeros(batch, T, N, 1).to(device)

        backward_init_updates = (torch.zeros(batch, N, K).to(device),
                                torch.zeros(batch, N, 1).to(device))
        backward_probs[:,-1,:,:] = backward_init_updates[0]
        backward_norms[:,-1,:,:] = backward_init_updates[1]

        backward_next_prob = backward_init_updates[0]

        for t_rev in range(T - 1):
            t = -t_rev - 2
            bi_tp1 = log_b[:,t+1,:,:]
            aij_tp1 = log_a[:,t+1,:,:]

            backward_current_updates_in = backward_next_prob.unsqueeze(3) + aij_tp1 + bi_tp1.unsqueeze(3)
            backward_current_updates_unnorm = torch.logsumexp(backward_current_updates_in, -2)
            backward_current_updates = self.normalize_logprob(backward_current_updates_unnorm, -1, 1)

            backward_next_prob = backward_current_updates[0]
            backward_probs[:,t,:,:] = backward_current_updates[0]
            backward_norms[:,t,:,:] = backward_current_updates[1]

        #### MARGINALS CALCULATION ###
        gamma1_unnorm = backward_probs + forward_probs

        m_forward = forward_probs[:,:-1,:,:].unsqueeze(3) # (batch, t-1, n, 1, k)
        m_backward = backward_probs[:,1:,:,:].unsqueeze(4) # (batch, t-1, n, k, 1)
        m_a = log_a[:,1:,:,:,:] # (batch, t-1, n, k, k)
        m_b = log_b[:,1:,:,:].unsqueeze(4) # (batch, t-1, n, k, 1)
        gamma2_unnorm = m_forward + m_backward + m_a + m_b

        gamma1, _ = self.normalize_logprob(gamma1_unnorm, -1, 1)
        gamma2_unpad, _ = normalize_logprob(gamma2_unnorm, [-2,-1], 1)
        gamma2 = torch.zeros(batch, T, N, K, K).to(device)
        gamma2[:,-T:,:,:,:] = gamma2_unpad
        
        return forward_probs, backward_probs, gamma1, gamma2

    def calculate_likelihoods_for_z(self, U_raw, temperature):
        # X is (batch, T, N, H)
        # U is (batch, T, N)
                                
        U_scaled = U_raw * self.U_scale.reshape(1,1,-1)
        U = torch.matmul(U_scaled, self.C_syn.T)
                                
        batch = self.batch
        T = self.T
        X_pad = torch.zeros(batch, T+1, self.N, self.H).to(self.device)
        X_pad[:,-T:,:,:] = self.X_inf

        log_A = torch.zeros(batch, T, self.N, self.K, self.K).to(self.device)
        log_B = torch.zeros(batch, T, self.N, self.K).to(self.device)

        B_est = torch.zeros(batch, T, self.N, self.K, self.H).to(self.device)
        log_A[:,0,:,:,:] = torch.ones(batch, self.K, self.K).to(self.device) * 1/self.K

        X_old = X_pad[:,:T,:,:] # (batch, T, N, H)
        X_new = X_pad[:,1:,:,:] # (batch, T, N, H)

        X_part = X_pad[:,1:-1,:,:] # (batch, T-1, N, H)
        U_part = U[:,1:,:] # (batch, T-1, N)

        for k in range(self.K):
            X_est = torch.matmul(self.W_xx[:,k].unsqueeze(0).unsqueeze(0) , X_old.unsqueeze(4)).squeeze(4) \
                + self.W_xu[:,k].unsqueeze(0).unsqueeze(0) * U.unsqueeze(3) \
                + self.b_x[:,k].unsqueeze(0).unsqueeze(0) # (batch, T, N, H)
            log_B[:,:,:,k] = torch.sum((X_est - X_new)**2 , 3)

            A_k_raw = torch.matmul(self.W_zx[:,k].unsqueeze(0).unsqueeze(0) , X_part.unsqueeze(4)).squeeze(4) \
                + self.W_zu[:,k].unsqueeze(0).unsqueeze(0) * U_part.unsqueeze(3) \
                + self.b_z[:,k].unsqueeze(0).unsqueeze(0) # (batch, T-1, N, K)
            A_k = self.normalize_logprob(A_k_raw, 3, temperature)

            ######
            log_A[:,1:,:,:,k] = A_k

        return log_A, log_B

    def calculate_likelihoods_for_x_theta(self, log_gamma1, log_gamma2, U_raw, Y, temperature):
        # gamma1 is (batch, T, N, K) # log posterior unary
        # gamma2 is (batch, T, N, K, K) # log posterior pairwise
        # U is (batch, T, N)
        # Y is (batch, T)
        U_scaled = U_raw * self.U_scale.reshape(1,1,-1)
        U = torch.matmul(U_scaled, self.C_syn.T)

        gamma1 = torch.exp(log_gamma1)
        gamma2 = torch.exp(log_gamma2)

        batch = self.batch
        T = self.T

        X_old = torch.zeros(batch, T, self.N, self.H).to(self.device)
        X_old[:,1:,:,:] = X_old[:,1:,:,:] + self.X_inf[:,1:,:,:]
        X_part = self.X_inf[:,:-1,:,:]
        U_part = U[:,1:,:] # (batch, T-1, N)

        term1 = 0 # p(x_t | x_{t-1}, z_t, u_t)
        term2 = 0 # p(z_t | z_{t-1}, x_t, u_t)
        term3 = 0 # p(y_t | x_t)

        for k in range(self.K):
            # TERM 1 #
            X_est = torch.matmul(self.W_xx[:,k].unsqueeze(0).unsqueeze(0) , X_old.unsqueeze(4)).squeeze(4) \
                + self.W_xu[:,k].unsqueeze(0).unsqueeze(0) * U.unsqueeze(3) \
                + self.b_x[:,k].unsqueeze(0).unsqueeze(0) # (batch, T, N, H)

            term1_cont = (X1_est - X_inf)**2 * gamma1[:,:,:,k].unsqueeze(3)
            term1 = term1 + torch.sum(term1_cont)

            # TERM 2 #
            log_P_zz = torch.zeros(batch, T, self.N, self.K).to(self.device)
            P_zz_0 = torch.ones(batch, self.N, self.K).to(self.device) * 1/self.K
            log_P_zz[:,0,:,:] = log_P_zz[:,0,:,:] + torch.log(P_zz_0) * gamma2[:,0,:,:,k] # (batch, N, K)

            log_P_zz_t_unnorm = torch.matmul(self.W_zx[:,k].unsqueeze(0).unsqueeze(0) , X_part.unsqueeze(4)).squeeze(4) \
                + self.W_zu[:,k].unsqueeze(0).unsqueeze(0) * U_part.unsqueeze(3) \
                + self.b_z[:,k].unsqueeze(0).unsqueeze(0) # (batch, T-1, N, K)
            log_P_zz_t = self.normalize_logprob(log_P_zz_t_unnorm, temperature) # (batch, T-1, N, K)
            log_P_zz[:,1:,:,:] = log_P_zz[:,1:,:,:] + log_P_zz_t * gamma2[:,1:,:,:,k]
            term2 = term2 + torch.sum(log_P_zz)

        # TERM 3 #
        Y_est = torch.sum(self.X_inf.reshape(batch, T, -1) * self.X_scale.reshape(1,1,-1), 2) # (batch, T)
        term3 = term3 + (Y_est - Y)**2

        loss = term1 + term2 + term3
        return loss

"""
    def test(self, U_raw):
        U_scaled = U_raw * self.U_scale.reshape(1,1,-1)
        U = torch.matmul(U_scaled, self.C_syn.T)
        
        X_out = torch.zeros(self.batch, self.T, self.N, self.H).to(self.device)
        X_curr = torch.zeros(self.batch, self.N, self.H).to(self.device)
        Z_curr = torch.zeros(self.batch, self.N, self.K).to(self.device)
        Z_curr[:,:,0] = 1
        
        for t in range(self.T):
            Z_idx = torch.argmax(X_curr)
            W_zu_part = self.W_zu[:,Z_idx,:] # (N, K)
            W_zx_part = self.W_zx[:,Z_idx,:,:] # (N, K, H)
            W_xx_part = self.W_xx[:,Z_idx,:,:] # (N, H, H)
            W_xu_part = self.W_xu[:,Z_idx,:] # (N, H)
            
            b_z_part = self.b_z[:,Z_idx,:] # (N, K)
            b_x_part = self.b_x[:,Z_idx,:] # (N, H)
                                
            Z_curr_raw
"""          
            
                                
                                
                                
                                