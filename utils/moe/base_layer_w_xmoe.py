# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from torch import Tensor, nn
from .base_sublayer import BaseSublayer, BasePhmSublayer

fused_cumsum_sub_one = lambda mask: torch.cumsum(mask, dim=0) - 1

logger = logging.getLogger(__name__)

def get_phm_rule_expert(base_layer_num=12, 
                        phm_dim=32,
                        expert_struct: str='MLP_split_to_layers_w_share',
                        strategy: str='plus'):
    assert strategy in ['plus', 'concat']
    # phm_rule_expert.data.normal_(mean=0, std=0.0001)
    if expert_struct == 'MLP_split_to_layers_w_share':
        if strategy == 'plus':
            phm_rule_expert = nn.Parameter(torch.FloatTensor(2*base_layer_num * phm_dim * phm_dim, phm_dim))
        else:
            phm_rule_expert = nn.Parameter(torch.FloatTensor(2*base_layer_num * phm_dim * phm_dim, phm_dim // 2))
        phm_rule_expert.data.normal_(mean=0, std=0.0001)
        return phm_rule_expert
    elif expert_struct == 'MLP_per_layer_w_share':
        if strategy == 'plus':
            phm_rule_expert = [nn.Parameter(torch.FloatTensor(2*phm_dim * phm_dim, phm_dim)) for _ in range(base_layer_num)]
        else:
            phm_rule_expert = [nn.Parameter(torch.FloatTensor(2*phm_dim * phm_dim, phm_dim // 2)) for _ in range(base_layer_num)]
        for item in phm_rule_expert:
            item.data.normal_(mean=0, std=0.0001)
        return phm_rule_expert
    else:
        raise ValueError("Other gate_types are not supported yet!")

def get_phm_rule_shared(
                        phm_dim=32,
                        expert_struct: str='MLP_per_layer_w_share',
                        phm_rule_per_layer_share: bool=False,
                        moe_expert_count=8,
                        strategy: str='plus'):
    assert strategy in ['plus', 'concat']
    assert expert_struct in ['MLP_per_layer_w_share', 'MLP_split_to_layers_w_share'] 
    if expert_struct == 'MLP_per_layer_w_share' and phm_rule_per_layer_share:
        phm_rule_shared = []
        if strategy == 'plus':
            for _ in range(moe_expert_count):
                phm_rule_shared.append(nn.Parameter(torch.FloatTensor(2*phm_dim*phm_dim, phm_dim)))
        else:
            for _ in range(moe_expert_count):
                phm_rule_shared.append(nn.Parameter(torch.FloatTensor(2*phm_dim*phm_dim, phm_dim // 2)))
        return phm_rule_shared
    else:
        return None

class BaseLayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 base_shuffle=True,
                 moe_expert_count=4,
                 base_layer_num=1,
                 phm_dim=32,
                 gate=None,
                 phm_expert: bool = False,
                 factorized_phm: bool = True,
                 phm_rank=1,
                 phm_rule_expert = None,
                 phm_rule_shared = None,
                 strategy: str='plus'
                 ):
        super().__init__()
        self.base_shuffle = base_shuffle
        self.in_features = in_features
        self.base_layer_num = base_layer_num
        self.phm_dim = phm_dim
        self.moe_expert_count = moe_expert_count
        self.gate = gate
        self.phm_expert = phm_expert
        self.factorized_phm = factorized_phm
        self.phm_rank = phm_rank
        self.phm_rule_expert = phm_rule_expert
        self.phm_rule_shared = phm_rule_shared
        self.strategy = strategy

        # expert type, phm_expert only support MLP_split_to_layers
        if self.phm_expert:
            self.out_features = out_features // (2*self.base_layer_num)
            if phm_rule_shared != None and self.base_layer_num == 1:
                expert_network =[BasePhmSublayer(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    layer_num=self.base_layer_num,
                    phm_dim=self.phm_dim,
                    phm_rule_expert=self.phm_rule_expert,
                    phm_rule_shared=self.phm_rule_shared[i],
                    factorized_phm=self.factorized_phm,
                    phm_rank=self.phm_rank,
                    strategy=self.strategy
                ) for i in range (self.moe_expert_count)]
            else:
                expert_network =[BasePhmSublayer(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    layer_num=self.base_layer_num,
                    phm_dim=self.phm_dim,
                    factorized_phm=self.factorized_phm,
                    phm_rank=self.phm_rank,
                    phm_rule_expert=self.phm_rule_expert,
                    
                    strategy=self.strategy
                ) for _ in range (self.moe_expert_count)]
        else:
            self.out_features = out_features
            expert_network =[BaseSublayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            ) for _ in range (self.moe_expert_count)]
        self.experts = nn.ModuleList(expert_network) 
        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.experts.parameters():
            param.expert = True

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
        
        chunks = dispatched_features.chunk(self.moe_expert_count, dim=0)
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