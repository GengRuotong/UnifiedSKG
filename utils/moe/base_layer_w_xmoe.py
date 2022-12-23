# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import time
from typing import Any, Tuple, cast
from torch import Tensor, nn
from .base_sublayer import BaseSublayer
from .route_w_gate_xmoe import Top1Gate

fused_cumsum_sub_one = lambda mask: torch.cumsum(mask, dim=0) - 1

logger = logging.getLogger(__name__)


class BaseLayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 mid_features: int,
                 out_features: int,
                 base_shuffle=True,
                 moe_expert_count=4,
                 expert_struct: str = 'MLP_split_to_layers_w_share',
                 ):
        super().__init__()
        self.base_shuffle = base_shuffle
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.moe_expert_count = moe_expert_count
        self.expert_struct = expert_struct
        if 'wo_share' in self.expert_struct:
            self.expert_dim = self.in_features
        elif 'w_share' in self.expert_struct:
            self.expert_dim = self.mid_features
        self.gate = Top1Gate(model_dim=self.expert_dim, num_experts=moe_expert_count)

        expert_network =[BaseSublayer(
                        in_features=self.in_features,
                        mid_features=self.mid_features,
                        out_features=self.out_features,
                        expert_struct= self.expert_struct
                        ) for _ in range (self.moe_expert_count)]
        self.experts = nn.ModuleList(expert_network) 
        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.experts.parameters():
            param.expert = True
        self.num_local_experts = len(self.experts)

    def forward(self, hidden_states) -> Tensor:
        # 3d to 2d(t, model_dim)
        features = hidden_states.reshape(-1, hidden_states.size(-1))
        features_shape = features.shape
        is_training = hidden_states.requires_grad
        features_padding_mask = None

        if self.base_shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = features[shuffle_sort]
  
        l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(
            features, features_padding_mask
        )
        dispatch_mask = dispatch_mask.to(hidden_states.dtype).permute(
            1, 2, 0
        )  # S,E,C -> E,C,S
        E, C, S = dispatch_mask.size()
        M = features_shape[1]
        assert features.size() == (S, M)
        # einsum("sec,sm->ecm")
        dispatched_features = torch.mm(
            dispatch_mask.view(E * C, S), features
        )  
        # Re-shape after all-to-all: (e*c,m) -> ecm
        dispatched_features = dispatched_features.reshape(E, C, M)
        
        chunks = dispatched_features.chunk(self.num_local_experts, dim=0)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=0)
        
        # einsum("sec,ecm->sm")
        combined_output = combine_weights.view(S, E * C).mm(
            expert_output.view(E * C, -1)
        )

        if self.base_shuffle and is_training:
            # Undo shuffling
            combined_output = combined_output[self.inverse_sort(shuffle_sort)]
        return combined_output.view(hidden_states.size(0), hidden_states.size(1), -1), l_aux
 

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(
            0, order, torch.arange(0, order.size(0), device=order.device)
        )
