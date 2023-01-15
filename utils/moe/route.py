import math
from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.optimal_transport.sinkhorn import SinkhornSolver
fused_cumsum_sub_one = lambda mask: torch.cumsum(mask, dim=0) - 1

# maximum capacity of 1 expert as a fraction of number of tokens in the batch
# Note: setting this to 1.0 causes inference to significantly slow down
EVAL_CAPACITY_TOKEN_FRACTION = 1

def get_gate_instance(
             model_dim: int,
             num_expert: int,
             gate_type: str='Top1Gate',
             use_xmoe: bool=True,
             base_layer_num=12
            ):
    assert gate_type in ['Top1Gate', 'Top2Gate']
    gate = []
    for _ in range(base_layer_num):
        if gate_type == 'Top1Gate':
            gate.append(Top1Gate(model_dim=model_dim, num_experts=num_expert, use_xmoe=use_xmoe))
        elif gate_type == 'Top2Gate':
            gate.append(Top2Gate(model_dim=model_dim, num_experts=num_expert, use_xmoe=use_xmoe))
        else:
            raise ValueError("Other gate_types are not supported yet!")
    gate = nn.ModuleList(gate)
    for param in gate.parameters():
        param.gate = True
    return gate

def balanced_assignment(input_feature, expert_feature, scores):
    cost = -scores
    solver = SinkhornSolver(L=cost)
    _, pi = solver.forward(input_feature, expert_feature)

    token_num = scores.shape[0]
    # Tell other workers how many tokens to expect from us
    return pi * token_num
    
# Assigns each token to the top k experts
def greedy_assignment(scores, k=1):
    # Tell other workers how many tokens to expect from us
    return torch.softmax(scores, dim=1)


def top1gating(
    logits: torch.Tensor,
    input_feature: Optional[torch.Tensor] = None,
    expert_feature: Optional[torch.Tensor] = None,
    input_mask: Optional[torch.Tensor] = None,
    capacity_factor=1.0,
    eval_mode=False,
    moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
    gate_obj=None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top2Gating on logits."""
    metadata = {}
    '''
    if not eval_mode:
        gates = balanced_assignment(input_feature, expert_feature, logits)
    else:
        gates = greedy_assignment(logits)
    '''
    gates = greedy_assignment(logits)
    
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = capacity_factor * S/E
        capacity = int(capacity_factor * math.ceil(num_tokens / num_experts))
    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts, unsqueeze_indices=True)                    
    if input_mask is not None and input_mask.any():
        nonpadding = ~input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    # for logging (percent of tokens routed to each expert)
    token_to_expert, sort_order = torch.sort(indices1_s)
    output_splits = torch.zeros(
            (logits.size(1),), dtype=indices1_s.dtype, device=indices1_s.device
        )
    workers, counts = torch.unique_consecutive(token_to_expert, return_counts=True)
    output_splits[workers] = counts
    metadata["expert_adress_tokens_count"] = output_splits
    # print(metadata)

    gates1_s = 1.5 * (gates * mask1).sum(dim=1)
    
    # Compute locations in capacity buffer
    locations1 = fused_cumsum_sub_one(mask1)


    # Compute l_aux
    '''
    if not eval_mode:
        origin_gates = greedy_assignment(logits)
        me = torch.mean(origin_gates, dim=0)
    else:
        me = torch.mean(gates, dim=0)
    '''
    me = torch.mean(gates, dim=0)
    
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts
    metadata["l_aux"] = l_aux

    # Remove locations outside capacity from mask
    mask1 = mask1 * torch.lt(locations1, capacity)
    # Store the capacity location for each token, represent expert capability uesd after addressing this token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    # locations1_sc = num_tokens * capacity
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1),
        locations1_sc.to(gates1.dtype).unsqueeze(1),
    )
    dispatch_mask = combine1_sec.bool()
    return l_aux, combine1_sec.to(logits.dtype), dispatch_mask, metadata

class Top1Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        capacity_factor=2.0,
        moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
        use_xmoe=True,
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()

        if not use_xmoe:
            expert_centroids = torch.empty(num_experts, model_dim)
        else:
            self.dim_reduction = nn.Linear(model_dim, 4, bias=False)
            expert_centroids = torch.empty(num_experts, 4)
        nn.init.orthogonal_(expert_centroids, gain=0.32)
        self.register_parameter(
            "expert_centroids", nn.Parameter(expert_centroids)
        )
        self.capacity_factor = capacity_factor
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.use_xmoe = use_xmoe
        
    def forward(self, input, mask=None):  # type: ignore
        is_training = input.requires_grad
        if self.use_xmoe:
            input = self.dim_reduction(input)
            with torch.no_grad():
                expert_centroids_norm = self.expert_centroids.norm(p=2.0, dim=1, keepdim=True)
                self.expert_centroids.mul_(1.5 / expert_centroids_norm)
            logits = self._cosine(input, self.expert_centroids)
            logits = self._make_finite(logits)
        else:
            # Compute similarity of each token to each expert, for routing
            logits = input.matmul(
                self.expert_centroids.transpose(0, 1)
            )

        return top1gating(
            logits,
            input_feature=input,
            expert_feature=self.expert_centroids,
            input_mask=mask,
            capacity_factor=self.capacity_factor,
            eval_mode=not is_training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            gate_obj=self,
        )

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores


def one_hot(indices: torch.Tensor, num_classes: int, unsqueeze_indices=False) -> Tensor:
    if unsqueeze_indices:
        indices = indices.unsqueeze(-1)
    assert indices.shape[-1] == 1, "last dimension of indices must be have size 1"
    output = torch.zeros(
        indices.shape[:-1] + (num_classes,), device=indices.device, dtype=indices.dtype
    )
    output.scatter_(len(output.shape) - 1, indices, 1)
    return output

gumbel_map: Dict[torch.device, Callable] = {}

def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)

def top2gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    second_expert_policy="sampling",
    normalize_gate_prob_before_dropping=False,
    eval_mode=False,
    moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    metadata = {}
    gates = greedy_assignment(logits)
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = 2S/E
        capacity = 2 * math.ceil(num_tokens / num_experts)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1, keepdim=True)
    mask1 = one_hot(indices1_s, num_classes=num_experts)
    if second_expert_policy == "sampling":
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    else:
        logits_w_noise = logits
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1, keepdim=True)
    mask2 = one_hot(indices2_s, num_classes=num_experts)

    # for logging (percent of tokens routed to each expert)
    token_to_expert1, sort_order1 = torch.sort(indices1_s.squeeze(dim=-1))
    token_to_expert2, sort_order2 = torch.sort(indices2_s.squeeze(dim=-1))
    output_splits1 = torch.zeros(
            (logits.size(1),), dtype=indices1_s.dtype, device=indices1_s.device
        )
    output_splits2 = torch.zeros(
            (logits.size(1),), dtype=indices2_s.dtype, device=indices2_s.device
        )
    workers1, counts1 = torch.unique_consecutive(token_to_expert1, return_counts=True)
    workers2, counts2 = torch.unique_consecutive(token_to_expert2, return_counts=True)
    output_splits1[workers1] = counts1
    output_splits2[workers2] = counts2
    metadata["expert1_adress_tokens_count"] = output_splits1
    metadata["expert2_adress_tokens_count"] = output_splits2

    gates1_s = 1.5*(gates * mask1).sum(dim=1)
    gates2_s = 1.5*(gates * mask2).sum(dim=1)

    if normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s = gates1_s / denom_s
        gates2_s = gates2_s / denom_s

    if second_expert_policy == "random":
        sampled = (2 * gates2_s) > torch.rand_like(gates2_s)
        mask2 = mask2 * sampled.repeat(num_experts, 1).transpose(1, 0)

    # Compute locations in capacity buffer
    if input_mask is not None and input_mask.any():
        nonpadding = ~input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)
        mask2 = mask2 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    locations1 = fused_cumsum_sub_one(mask1)
    locations2 = fused_cumsum_sub_one(mask2)
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts
    metadata["l_aux"] = l_aux
    # print(metadata)

    # Remove locations outside capacity from mask
    mask1 = mask1 * torch.lt(locations1, capacity)
    mask2 = mask2 * torch.lt(locations2, capacity)


    if not normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        gates1_s = (gates * mask1).sum(dim=1)
        gates2_s = (gates * mask2).sum(dim=1)
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s /= denom_s
        gates2_s /= denom_s


    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    gates2 = gates2_s.unsqueeze(-1) * mask2.to(gates2_s.dtype)  # einsum("s,se->se")
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    locations2_sc = one_hot(locations2_s, num_classes=capacity, unsqueeze_indices=True)
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1),
        locations1_sc.to(gates1.dtype).unsqueeze(1),
    )
    combine2_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates2.unsqueeze(-1),
        locations2_sc.to(gates2.dtype).unsqueeze(1),
    )
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    return l_aux, combine_weights, dispatch_mask, metadata


class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        second_expert_policy="sampling",
        normalize_gate_prob_before_dropping=False,
        moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
        use_xmoe=True,
    ) -> None:
        super().__init__()
        if not use_xmoe:
            expert_centroids = torch.empty(num_experts, model_dim)
        else:
            self.dim_reduction = nn.Linear(model_dim, 4, bias=False)
            expert_centroids = torch.empty(num_experts, 4)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.32)
        self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))
        
        self.second_expert_policy = second_expert_policy
        self.normalize_gate_prob_before_dropping = normalize_gate_prob_before_dropping
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.use_xmoe = use_xmoe

    def forward(self, input, mask=None):  # type: ignore
        is_training = input.requires_grad
        if self.use_xmoe:
            input = self.dim_reduction(input)
            with torch.no_grad():
                expert_centroids_norm = self.expert_centroids.norm(p=2.0, dim=1, keepdim=True)
                self.expert_centroids.mul_(1.5 / expert_centroids_norm)
            logits = self._cosine(input, self.expert_centroids)
            logits = self._make_finite(logits)
        else:
            logits = input.matmul(
                self.expert_centroids.transpose(0, 1)
            )
        return top2gating(
            logits,
            input_mask=mask,
            second_expert_policy=self.second_expert_policy,
            normalize_gate_prob_before_dropping=self.normalize_gate_prob_before_dropping,
            eval_mode=not is_training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
        )

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores




