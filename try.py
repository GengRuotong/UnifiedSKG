import torch
fused_cumsum_sub_one = lambda mask: torch.cumsum(mask, dim=0) - 1
from utils.optimal_transport.sinkhorn import SinkhornSolver

# generate two gaussians as the source and target
def gaussian(mean=0, std=10, n=100):
    d = (-(torch.arange(n) - mean)**2 / (2 * std**2)).exp()
    d /= d.sum()
    return d

def greedy_assignment(scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers_sort, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros(
            (scores.size(-1),), dtype=torch.long, device=scores.device
        )
        workers, counts = torch.unique_consecutive(token_to_workers_sort, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from usa
        return token_to_workers, output_splits.tolist(), sort_ordering

x = torch.randn(20,10)
y = torch.randn(4, 10)


score = torch.tensor([[-0.0066, -0.0195,  0.0603, -0.2632],
        [ 0.1687,  0.0857, -0.1615, -0.0939],
        [ 0.1687,  0.0857, -0.1615, -0.0939],
        [-0.1851, -0.0095, -0.3242, -0.4202],
        [ 0.1687,  0.0857, -0.1615, -0.0939],
        [-0.0472,  0.0061, -0.0649, -0.1672],
        [-0.0066, -0.0195,  0.0603, -0.2632],
        [-0.1851, -0.0095, -0.3242, -0.4202],
        [-0.1418, -0.1187,  0.0016, -0.4024],
        [-0.0472,  0.0061, -0.0649, -0.1672],
        [-0.1851, -0.0095, -0.3242, -0.4202],
        [-0.0066, -0.0195,  0.0603, -0.2632],
        [-0.1418, -0.1187,  0.0016, -0.4024],
        [-0.0472,  0.0061, -0.0649, -0.1672],
        [-0.0472,  0.0061, -0.0649, -0.1672],
        [-0.1851, -0.0095, -0.3242, -0.4202],
        [-0.1418, -0.1187,  0.0016, -0.4024],
        [ 0.1687,  0.0857, -0.1615, -0.0939],
        [-0.1418, -0.1187,  0.0016, -0.4024],
        [-0.0066, -0.0195,  0.0603, -0.2632]])
epsilon = 1e-2
solver = SinkhornSolver(epsilon=epsilon, L=score)
gain_distance, pi = solver.forward(x, y)
print(pi)






