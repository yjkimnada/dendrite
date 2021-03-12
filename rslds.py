import torch
from torch import nn
from torch.nn import functional as F

class RSLDS(nn.Module):
    def __init__(self, T, N, K, H, R, batch, C_syn, device):
        super().__init__()

        self.K = K
        self.C_syn = C_syn
        self.in_no = C_syn.shape[1]
        self.device = device
        self.N = N
        self.H = H
        self.batch = batch
        self.T = T
        self.R = R

        self.W_zu = nn.Parameter(torch.randn(N, K, K))
        self.W_zx = nn.Parameter(torch.randn(N, K, K, H))
        self.W_xx = nn.Parameter(torch.randn(N, K, H, H))
        self.W_xu = nn.Parameter(torch.randn(N, K, H))
        self.W_yx = nn.Linear(N*H, 1)

        self.b_z = nn.Parameter(torch.randn(N, K, K))
        self.b_x = nn.Parameter(torch.randn(N, K, H))

        self.U_scale = nn.Parameter(torch.ones(self.in_no))
        self.Z_init = nn.Parameter(torch.zeros(N, K))
        
        self.encoder = nn.GRU(N+1, R, batch_first=True, bidirectional=True)
        self.enc_lin = nn.Linear(2*R, N*H)

    def normalize_logprob(self, logmat, dim, temperature):
        logmat = logmat / temperature
        normalizer = torch.logsumexp(logmat, dim, True)
        return logmat - normalizer, normalizer

    def forward_backward(self, log_a, log_b, logprob_z1):
        # A is (batch, t, n, k, k); p(z_t | z_{t-1}, x_{t-1}, u_t)
        # B is (batch, t, n, k); p(x_t | x_{t-1}, z_t, u_t)
        # z1 is (n, k); p(z_1 | u_1) aka A_1
        
        log_a = log_a.detach()
        log_b = log_b.detach()

        ### FORWARD ###
        forward_probs = torch.zeros(self.batch, self.T, self.N, self.K).to(self.device)
        forward_norms = torch.zeros(self.batch, self.T, self.N, 1).to(self.device)

        forward_init_updates_unnorm = logprob_z1.unsqueeze(0) + log_b[:,0,:,:]
        forward_init_updates = self.normalize_logprob(forward_init_updates_unnorm, -1, 1) # (batch, n, k) and (batch, n, 1)
        forward_probs[:,0,:,:] = forward_init_updates[0].detach()
        forward_norms[:,0,:,:] = forward_init_updates[1].detach()

        forward_prev_prob = forward_init_updates[0].detach() # first forward probability normalized -> (batch, n, 1, k)
        
        ### BACKWARD ###
        backward_probs = torch.zeros(self.batch, self.T, self.N, self.K).to(self.device)
        backward_norms = torch.zeros(self.batch, self.T, self.N, 1).to(self.device)

        backward_init_updates = (torch.zeros(self.batch, self.N, self.K).to(self.device),
                                torch.zeros(self.batch, self.N, 1).to(self.device))
        backward_probs[:,-1,:,:] = backward_init_updates[0].detach()
        backward_norms[:,-1,:,:] = backward_init_updates[1].detach()

        backward_next_prob = backward_init_updates[0].detach()

        for t in range(self.T - 1):
            bi_t = log_b[:,t+1,:,:].detach() # log p(x_t | x_{t-1}, z_t, u_t); (batch, n, k) -> (batch, n, k, 1)
            aij_t = log_a[:,t+1,:,:,:].detach() # log p(z_t | z_{t-1}, x_{t-1}, u_t); (batch, n, k, k)

            forward_current_updates_in = bi_t.unsqueeze(3) + aij_t + forward_prev_prob.unsqueeze(2) # (batch, n, k, k)
            forward_current_updates_unnorm = torch.logsumexp(forward_current_updates_in, -1) # (batch, n, k)
            forward_current_updates = self.normalize_logprob(forward_current_updates_unnorm, -1, 1) # (batch, n, k) and (batch, n, 1)

            forward_prev_prob = forward_current_updates[0].detach()
            forward_probs[:,t+1,:,:] = forward_current_updates[0].detach()
            forward_norms[:,t+1,:,:] = forward_current_updates[1].detach()
            
            t_rev = -t-2
            bi_tp1 = log_b[:,t_rev+1,:,:].detach() # (batch, n, k)
            aij_tp1 = log_a[:,t_rev+1,:,:,:].detach() # (batch, n, k, k)
            
            backward_current_updates_in = backward_next_prob.unsqueeze(3) + aij_tp1 + bi_tp1.unsqueeze(3)
            backward_current_updates_unnorm = torch.logsumexp(backward_current_updates_in, -2)
            backward_current_updates = self.normalize_logprob(backward_current_updates_unnorm, -1, 1)

            backward_next_prob = backward_current_updates[0].detach()
            backward_probs[:,t_rev,:,:] = backward_current_updates[0].detach()
            backward_norms[:,t_rev,:,:] = backward_current_updates[1].detach()

        #### MARGINALS CALCULATION ###
        gamma1_unnorm = backward_probs + forward_probs

        m_forward = forward_probs[:,:-1,:,:].unsqueeze(3) # (batch, t-1, n, 1, k)
        m_backward = backward_probs[:,1:,:,:].unsqueeze(4) # (batch, t-1, n, k, 1)
        m_a = log_a[:,1:,:,:,:] # (batch, t-1, n, k, k)
        m_b = log_b[:,1:,:,:].unsqueeze(4) # (batch, t-1, n, k, 1)
        gamma2_unnorm = m_forward + m_backward + m_a + m_b

        gamma1, _ = self.normalize_logprob(gamma1_unnorm, -1, 1)
        gamma2_unpad, _ = self.normalize_logprob(gamma2_unnorm, [-2,-1], 1)
        gamma2 = torch.zeros(self.batch, self.T, self.N, self.K, self.K).to(self.device)
        gamma2[:,1:,:,:,:] =  gamma2[:,1:,:,:,:] + gamma2_unpad
        
        return forward_probs, backward_probs, gamma1, gamma2

    def calculate_AB(self, X, U, temperature):
        # X is (batch, T, N, H)
        # U is (batch, T, N)

        log_A = torch.zeros(self.batch, self.T, self.N, self.K, self.K).to(self.device)
        #log_B = torch.zeros(self.batch, self.T, self.N, self.K).to(self.device)

        X_old = torch.zeros(self.batch, self.T, self.N, self.H).to(self.device)
        X_old[:,1:,:,:] = X_old[:,1:,:,:] + X[:,:-1,:,:].clone()

        log_A[:,0,:,:,:] = log_A[:,0,:,:,:] + torch.eye(self.K).to(self.device)
        
        # Calculate B: p(x_t | x_{t-1}, z_t, u_t) Continuous
        X_est = X_old.unsqueeze(-2) + torch.matmul(self.W_xx.unsqueeze(0).unsqueeze(0) , X_old.unsqueeze(3).unsqueeze(-1)).squeeze(-1) \
            + self.W_xu.unsqueeze(0).unsqueeze(0) * U.unsqueeze(-1).unsqueeze(-1) \
            + self.b_x.unsqueeze(0).unsqueeze(0) # (batch, T, N, K, H)
        
        #######################
        ### DETACH?!?!?!??! ###
        #######################
        log_B = torch.sum(-(X_est - X.detach().unsqueeze(-2))**2, -1)
        
        A_raw = torch.matmul(self.W_zx.unsqueeze(0).unsqueeze(0) , X[:,:-1,:,:].clone().unsqueeze(-1).unsqueeze(3)).squeeze(-1) \
            + self.W_xu.unsqueeze(0).unsqueeze(0) * U[:,1:,:].unsqueeze(-1).unsqueeze(-1) \
            + self.b_x.unsqueeze(0).unsqueeze(0) # (batch, T-1, N, K, K)
        log_A[:,1:,:,:,:] = log_A[:,1:,:,:,:] + self.normalize_logprob(A_raw, -2 , temperature)[0]

        return log_A, log_B

    def calculate_total_prob(self, Y, X, log_A, log_B, log_gamma1, log_gamma2):
        #######################
        ### DETACH?!?!?!??! ###
        #######################
        gamma1 = torch.exp(log_gamma1).detach() # (batch, T, N, K)
        gamma2 = torch.exp(log_gamma2).detach() # (batch, T, N, K, K)

        # Calculate Initial Loss
        init_prob = torch.sum(gamma1[:,0,:,:] * (log_B[:,0,:,:] + self.Z_init.unsqueeze(0)))

        # Calculate Rest of Sequence Loss
        seq_prob = torch.sum(gamma2[:,1:,:,:,:] * (log_A[:,1:,:,:,:] + log_B[:,1:,:,:].unsqueeze(-2)))

        # Calculate Y Loss
        Y_est = self.W_yx(X.reshape(-1,self.N*self.H)).reshape(self.batch, self.T)
        
        Y_prob = torch.sum(-(Y_est-Y)**2)

        return Y_prob, init_prob, seq_prob

    def forward(self, Y, U_raw, temp, beta):

        U_scaled = U_raw * self.U_scale.reshape(1,1,-1)
        U = torch.matmul(U_scaled, self.C_syn.T) # (self.batch, self.T, self.N)

        enc_in = torch.zeros(self.batch, self.T, self.N+1).to(self.device)
        enc_in[:,:,0] = enc_in[:,:,0] + Y
        enc_in[:,:,1:] = enc_in[:,:,1:] + U
        
        gamma_prior = torch.ones(self.batch, self.T, self.N, self.K).to(self.device) * 1/self.K

        X_enc_raw, _ = self.encoder(enc_in) # (batch, T, 2R)
        X_enc = self.enc_lin(X_enc_raw.reshape(-1, 2*self.R)).reshape(self.batch, self.T, self.N, self.H)
        
        log_A, log_B = self.calculate_AB(X_enc, U, temp)
        _, _, log_gamma1, log_gamma2 = self.forward_backward(log_A, log_B, self.Z_init)
        Y_prob, init_prob, seq_prob = self.calculate_total_prob(Y, X_enc, log_A, log_B, log_gamma1, log_gamma2)
        
        gamma_prior_prob = torch.sum(gamma_prior * log_gamma1)

        return Y_prob, init_prob, seq_prob, gamma_prior_prob
    
    def test(self, U_raw):
        U_scaled = U_raw * self.U_scale.reshape(1,-1)
        U = torch.matmul(U_scaled, self.C_syn.T) # (self.T, self.N)
        
        X_out = torch.zeros(self.T, self.N, self.H).to(self.device)
        Z_out = torch.zeros(self.T, self.N, self.K).to(self.device)
        
        Z_curr = F.softmax(self.Z_init*5000, -1) # (N, K)
        Z_out[0] = Z_curr
        
        part_W_xu = torch.sum(self.W_xu*Z_curr.unsqueeze(-1), 1) # (N, H)
        part_b_x = torch.sum(self.b_x*Z_curr.unsqueeze(-1), 1) # (N, H)
        X_curr = part_W_xu * U[0].unsqueeze(-1) + part_b_x # (N, H)
        X_out[0] = X_curr
        
        for t in range(self.T-1):
            part_W_zu = torch.sum(self.W_zu*Z_curr.unsqueeze(-1), 1) # (N, K)
            part_W_zx = torch.sum(self.W_zx*Z_curr.unsqueeze(-1).unsqueeze(-1), 1) # (N,K,H)
            part_b_z = torch.sum(self.b_z*Z_curr.unsqueeze(-1), 1) # (N,K)
            
            Z_curr_raw = torch.matmul(part_W_zx , X_curr.unsqueeze(-1)).squeeze(-1) \
                + part_W_zu*U[t+1].unsqueeze(-1) + part_b_z # (N, K)
            Z_curr = F.softmax(Z_curr_raw*5000, -1) #(N, K)
            Z_out[t+1] = Z_curr
            
            part_W_xu = torch.sum(self.W_xu*Z_curr.unsqueeze(-1), 1) # (N, H)
            part_W_xx = torch.sum(self.W_xx*Z_curr.unsqueeze(-1).unsqueeze(-1), 1) # (N,H,H)
            part_b_x = torch.sum(self.b_x*Z_curr.unsqueeze(-1), 1) # (N, H)
            
            X_curr = X_curr + torch.matmul(part_W_xx, X_curr.unsqueeze(-1)).squeeze(-1) \
                + part_W_xu*U[t+1].unsqueeze(-1) + part_b_x # (N, H)
            X_out[t+1] = X_curr
            
        Y_out = self.W_yx(X_out.reshape(-1, self.N*self.H)).flatten()
        
        return Y_out, X_out, Z_out
            
        
            
            
        
        
        

        