import torch
import torch.nn as nn

class SinkhornSolver(nn.Module):
    """
    Optimal Transport solver under entropic regularisation.
    Based on the code of Gabriel Peyré.
    """
    def __init__(self, epsilon=1e-2, iterations=1000, threshold = 1e-3, L=None):
        super(SinkhornSolver, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.threshold = threshold
        self.C_gpu = -L
        self.C = self.C_gpu.cpu()


    def forward(self, x, y):
        num_x = x.size(-2)
        num_y = y.size(-2)
        
        batch_size = 1 if x.dim() == 2 else x.size(0)

        # Marginal densities are empirical measures
        with torch.no_grad():
            a = x.new_ones((batch_size, num_x), requires_grad=False, dtype=self.C.dtype, device=self.C.device) / num_x
            b = y.new_ones((batch_size, num_y), requires_grad=False,  dtype=self.C.dtype, device=self.C.device) / num_y

            a = a.squeeze()
            b = b.squeeze()
                
        # Initialise approximation vectors in log domain
        u = torch.zeros_like(a, dtype=self.C.dtype, device=self.C.device)
        v = torch.zeros_like(b, dtype=self.C.dtype, device=self.C.device)
        
        # Sinkhorn iterations
        for i in range(self.iterations): 
            u0, v0 = u, v
                        
            # u^{l+1} = a / (K v^l)
            K = self._log_boltzmann_kernel(u, v, self.C)
            u_ = torch.log(a + 1e-8) - torch.logsumexp(K, dim=1)
            u = self.epsilon * u_ + u
                        
            # v^{l+1} = b / (K^T u^(l+1))
            K_t = self._log_boltzmann_kernel(u, v, self.C).transpose(-2, -1)
            v_ = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=1)
            v = self.epsilon * v_ + v
            
            # Size of the change we have performed on u
            diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
            mean_diff = torch.mean(diff)
            if mean_diff.item() < self.threshold:
                break
   
        # print("Finished computing transport plan in {} iterations".format(i))
    
        # Transport plan pi = diag(a)*K*diag(b)
        u = u.to(self.C_gpu.device)
        v = v.to(self.C_gpu.device)
        K = self._log_boltzmann_kernel(u, v, self.C_gpu)
        pi = torch.exp(K)
        
        # Sinkhorn distance
        # gain = torch.sum(pi * self.C_, dim=(-2, -1))
        # print("pi {}".format(pi))
        return None, pi

    def _log_boltzmann_kernel(self, u, v, C=None):
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel