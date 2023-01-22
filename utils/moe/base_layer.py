# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from torch import Tensor, nn
from .base_sublayer import BaseSublayer, BasePhmSublayer, BaseLPSublayer, BasePLSublayer

fused_cumsum_sub_one = lambda mask: torch.cumsum(mask, dim=0) - 1

logger = logging.getLogger(__name__)

def get_phm_rule_expert(base_layer_num=12, 
                        phm_dim=32,
                        moe_expert_num=8,
                        strategy: str='plus',
                        phm_rank=1,
                        phm_rule_expert_share: bool=False,
                        share_kv: bool=False):
    assert strategy in ['plus', 'concat', 'mat']
    if share_kv:
        phm_rule_num = 1
    else:
        phm_rule_num = 2
    if phm_rule_expert_share:
        if strategy == 'mat':
            phm_rule_expert = [nn.Parameter(torch.FloatTensor(phm_rule_num * phm_dim * phm_rank, phm_dim)) for _ in range(base_layer_num)]
        elif strategy == 'plus':
            phm_rule_expert = [nn.Parameter(torch.FloatTensor(phm_rule_num * phm_dim * phm_dim, phm_dim)) for _ in range(base_layer_num)]
        else:
            phm_rule_expert = [nn.Parameter(torch.FloatTensor(phm_rule_num * phm_dim * phm_dim, phm_dim // 2)) for _ in range(base_layer_num)]
        for item in phm_rule_expert:
            item.data.normal_(mean=0, std=0.0001)
        return phm_rule_expert
    else:
        phm_rule_expert = []
        for i in range(base_layer_num):
            if strategy == 'mat':
                phm_rule_expert.append([nn.Parameter(torch.FloatTensor(phm_rule_num * phm_dim * phm_rank, phm_dim)) for _ in range(moe_expert_num)])
            elif strategy == 'plus':
                phm_rule_expert.append([nn.Parameter(torch.FloatTensor(phm_rule_num * phm_dim * phm_dim, phm_dim)) for _ in range(moe_expert_num)])
            else:
                phm_rule_expert.append([nn.Parameter(torch.FloatTensor(phm_rule_num * phm_dim * phm_dim, phm_dim // 2)) for _ in range(moe_expert_num)])
        
        for layer_expert in phm_rule_expert:
            for item in layer_expert:
                item.data.normal_(mean=0, std=0.0001)
        return phm_rule_expert

def get_phm_rule_shared(
                        phm_dim=32,
                        phm_rule_per_layer_share: bool=False,
                        moe_expert_count=8,
                        phm_rank=1,
                        strategy: str='mat',
                        share_kv: bool=False):
    assert strategy in ['plus', 'concat','mat'] 
    if phm_rule_per_layer_share:
        phm_rule_shared = []
        if share_kv:
            if strategy == 'plus':
                phm_rule_shared = [nn.Parameter(torch.FloatTensor(phm_dim*phm_dim, phm_dim)) for _ in range(moe_expert_count)]
            elif strategy == 'mat':
                phm_rule_shared = [nn.Parameter(torch.FloatTensor(phm_dim*phm_dim, phm_rank)) for _ in range(moe_expert_count)]
            else:
                phm_rule_shared = [nn.Parameter(torch.FloatTensor(phm_dim*phm_dim, phm_dim // 2)) for _ in range(moe_expert_count)]
        else:
            if strategy == 'plus':
                phm_rule_shared = [nn.Parameter(torch.FloatTensor(2*phm_dim*phm_dim, phm_dim)) for _ in range(moe_expert_count)]
            elif strategy == 'mat':
                phm_rule_shared = [nn.Parameter(torch.FloatTensor(2*phm_dim*phm_dim, phm_rank)) for _ in range(moe_expert_count)]
            else:
                phm_rule_shared = [nn.Parameter(torch.FloatTensor(2*phm_dim*phm_dim, phm_dim // 2)) for _ in range(moe_expert_count)]
        for item in phm_rule_shared:
            item.data.normal_(mean=0, std=0.0001)
        return phm_rule_shared
    else:
        return None

class BaseLayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 mid_features: int,
                 out_features: int,
                 base_shuffle=True,
                 moe_expert_count=8,
                 base_layer_num=1,
                 phm_dim=32,
                 gate=None,
                 factorized_phm: bool = False,
                 phm_rank=1,
                 phm_rule_expert_down = None,
                 phm_rule_expert_up = None,
                 phm_rule_shared_down = None,
                 phm_rule_shared_up = None,
                 strategy: str='mat',
                 share_kv: bool='False',
                 project_struct: str='PP',
                 ):
        super().__init__()
        self.base_shuffle = base_shuffle
        self.in_features = in_features
        self.mid_features = mid_features
        self.base_layer_num = base_layer_num
        self.phm_dim = phm_dim
        self.moe_expert_count = moe_expert_count
        self.gate = gate
        self.factorized_phm = factorized_phm
        self.phm_rank = phm_rank
        self.phm_rule_expert_down = phm_rule_expert_down
        self.phm_rule_expert_up = phm_rule_expert_up
        self.phm_rule_shared_down = phm_rule_shared_down
        self.phm_rule_shared_up = phm_rule_shared_up
        self.strategy = strategy
        self.share_kv = share_kv
        self.project_struct = project_struct

        # expert type, phm_expert only support MLP_split_to_layers
        if self.project_struct == 'PP' or self.project_struct == 'LP':
            self.out_features = out_features // (2*self.base_layer_num)
            if self.project_struct == 'PP':
                    expert_network =[BasePhmSublayer(
                        in_features=self.in_features,
                        mid_features=self.mid_features,
                        out_features=self.out_features,
                        layer_num=self.base_layer_num,
                        phm_dim=self.phm_dim,
                        # phm_rule_expert_down=self.phm_rule_expert_down[i],
                        # phm_rule_expert_up=self.phm_rule_expert_up[i],
                        # phm_rule_shared_down=None,
                        # phm_rule_shared_up=None,
                        phm_rule_expert_down=self.phm_rule_expert_down,
                        phm_rule_expert_up=self.phm_rule_expert_up,
                        phm_rule_shared_down=self.phm_rule_shared_down[i],
                        phm_rule_shared_up=self.phm_rule_shared_up[i],
                        factorized_phm=self.factorized_phm,
                        phm_rank=self.phm_rank,
                        strategy=self.strategy,
                        share_kv= self.share_kv
                    ) for i in range (self.moe_expert_count)]
            elif self.project_struct == 'LP':
                    expert_network =[BaseLPSublayer(
                        in_features=self.in_features,
                        mid_features=self.mid_features,
                        out_features=self.out_features,
                        layer_num=self.base_layer_num,
                        phm_dim=self.phm_dim,
                        phm_rule_expert_down=self.phm_rule_expert_down,
                        phm_rule_expert_up=self.phm_rule_expert_up,
                        phm_rule_shared_down=self.phm_rule_shared_down[i],
                        phm_rule_shared_up=self.phm_rule_shared_up[i],
                        factorized_phm=self.factorized_phm,
                        phm_rank=self.phm_rank,
                        strategy=self.strategy,
                        share_kv= self.share_kv
                    ) for i in range (self.moe_expert_count)]  
        else:
            self.out_features = out_features
            if self.project_struct == 'PL':
                expert_network =[BasePLSublayer(
                    in_features=self.in_features,
                    mid_features=self.mid_features,
                    out_features=self.out_features,
                    layer_num=self.base_layer_num,
                    phm_dim=self.phm_dim,
                    phm_rule_expert_down=self.phm_rule_expert_down,
                    phm_rule_expert_up=self.phm_rule_expert_up,
                    phm_rule_shared_down=self.phm_rule_shared_down[i],
                    phm_rule_shared_up=self.phm_rule_shared_up[i],
                    factorized_phm=self.factorized_phm,
                    phm_rank=self.phm_rank,
                    strategy=self.strategy,
                    share_kv= self.share_kv
                ) for i in range (self.moe_expert_count)]
            else:
                expert_network =[BaseSublayer(
                                in_features=self.in_features,
                                mid_features=self.mid_features,
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
