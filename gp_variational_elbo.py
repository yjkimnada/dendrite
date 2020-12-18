from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
import torch

class VariationalELBO(_ApproximateMarginalLogLikelihood):
    
    def _log_likelihood_term(self, variational_dist_f, target, S_e, S_i, **kwargs):
        error, pred, all_F = self.likelihood.expected_log_prob(target, variational_dist_f, S_e, S_i, **kwargs)
        
        return error.sum(-1), pred, all_F

    def forward(self, approximate_dist_f, target, S_e, S_i, **kwargs):

        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        log_likelihood, pred, all_F = self._log_likelihood_term(approximate_dist_f, target, S_e, S_i,**kwargs)
        log_likelihood = log_likelihood.div(num_batch)
        
        kl_divergence = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for _, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure()).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss , pred, all_F
            #return log_likelihood , pred
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior.div(self.num_data), added_loss
            else:
                return log_likelihood, kl_divergence, log_prior.div(self.num_data)