import math
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from utils.optimal_transport.sinkhorn import SinkhornSolver
fused_cumsum_sub_one = lambda mask: torch.cumsum(mask, dim=0) - 1

# maximum capacity of 1 expert as a fraction of number of tokens in the batch
# Note: setting this to 1.0 causes inference to significantly slow down
EVAL_CAPACITY_TOKEN_FRACTION = 1

def balanced_assignment(input_feature, expert_feature, scores):
    solver = SinkhornSolver(L=scores)
    _, pi = solver.forward(input_feature, expert_feature)

    expert_num = scores.shape[1]
    # Tell other workers how many tokens to expect from us
    return pi * expert_num
    
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
    if not eval_mode:
        gates = balanced_assignment(input_feature, expert_feature, logits)
    else:
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
    
    # Compute locations in capacity buffer
    locations1 = fused_cumsum_sub_one(mask1)
    gates1_s = (gates * mask1).sum(dim=1)


    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)

    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts

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
        capacity_factor=1.2,
        moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()

        expert_centroids = torch.empty(num_experts, model_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.32)
        self.register_parameter(
            "expert_centroids", torch.nn.Parameter(expert_centroids)
        )
        self.capacity_factor = capacity_factor
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        
    def forward(self, input, mask=None):  # type: ignore
        is_training = input.requires_grad
        with torch.no_grad():
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


def one_hot(indices: torch.Tensor, num_classes: int, unsqueeze_indices=False) -> Tensor:
    if unsqueeze_indices:
        indices = indices.unsqueeze(-1)
    assert indices.shape[-1] == 1, "last dimension of indices must be have size 1"
    output = torch.zeros(
        indices.shape[:-1] + (num_classes,), device=indices.device, dtype=indices.dtype
    )
    output.scatter_(len(output.shape) - 1, indices, 1)
    return output
