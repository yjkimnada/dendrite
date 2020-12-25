import torch
import gpytorch


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_tasks, M, N):
        #inducing_points = torch.rand(num_tasks, M, 1)
        
        #inducing_points = torch.arange(0,N-1,N//M).float().reshape(1,-1).repeat(num_tasks,1).unsqueeze(2)
        inducing_points = torch.arange(0,M,1).float().reshape(1,-1).repeat(num_tasks,1).unsqueeze(2)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)