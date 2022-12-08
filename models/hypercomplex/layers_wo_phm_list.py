# The codes are from https://github.com/bayer-science-for-a-better-life/phc-gnn
import torch
import torch.nn as nn
from typing import Union, Optional
import torch.nn.functional as F
from .inits import glorot_uniform, glorot_normal  #phm_init
from .kronecker import kronecker_product_einsum_batched


def weight_sum_batched(weight: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """
    Functional method to compute depthwise convolution to fuse channel features
    weight is an order-2 tensor of shape (bsz, channels_num)
    feature has shape (channels_num, feature_in, feature_out)
    result has shape (bsz, feature_in, feature_out)
    """
    bsz = weight.shape[0]
    channels_num = feature.shape[0]
    result = []
    assert weight.shape[-1] == feature.shape[0], f"The number of weight channels {weight.shape[-1]} is not equal to the number of input channels {feature.shape[0]}"
    for i in range(bsz):
        weight_per_batch = weight[i]  # (channels_num)
        weight_sum = 0
        for j in range(channels_num):
            weight_sum += weight_per_batch[j] * feature[j]
        result.append(weight_sum)
    return torch.stack(result, dim=0)

def matvec_product(W: torch.Tensor, x: torch.Tensor,
                       bias: Union[None, torch.Tensor],
                       phm_rule: Union[None, torch.Tensor],
                       weight: torch.Tensor,
                       kronecker_prod_gate=False) -> torch.Tensor:
    """
    Functional method to compute the generalized matrix-vector product based on the paper
    "Parameterization of Hypercomplex Multiplications (2020)"
    https://openreview.net/forum?id=rcQdycl0zyk
    y = Hx + b , where W is generated through the sum of kronecker products from the Parameterlist W, i.e.
    W is a an order-3 tensor of size (phm_dim, in_features, out_features)
    x has shape (batch_size, phm_dim*in_features)
    phm_rule is an order-3 tensor of shape (phm_dim, phm_dim, phm_dim)
    H = sum_{i=0}^{d} mul_rule \otimes W[i], where \otimes is the kronecker product
    """
    if not kronecker_prod_gate:
        H = kronecker_product_einsum_batched(phm_rule, W).sum(0)
        y = torch.matmul(input=x, other=H)
    else: 
        features = kronecker_product_einsum_batched(phm_rule, W) 
        H = weight_sum_batched(weight=weight, feature=features) # (bsz, feature_in, feature_out)
        assert x.shape[0] == H.shape[0]
        bsz = H.shape[0]
        temp = []
        for i in range(bsz):
            temp.append(torch.matmul(input=x[i], other=H[i].squeeze(dim=0)))
        y = torch.stack(temp, dim=0)
    if bias is not None:
        y += bias
    return y

class PHMLinear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 phm_dim: int,
                 bias: bool = True,
                 w_init: str = "glorot-uniform",
                 phm_rule: Union[None, torch.Tensor] = None,
                 factorized_phm=True,
                 shared_W_phm=False,
                 phm_rank = 1,
                 phm_init_range=0.0001,
                 kronecker_prod_gate=False) -> None:
        super(PHMLinear, self).__init__()
        assert w_init in ["phm", "glorot-normal", "glorot-uniform", "normal"]
        assert in_features % phm_dim == 0, f"Argument `in_features`={in_features} is not divisble be `phm_dim`{phm_dim}"
        assert out_features % phm_dim == 0, f"Argument `out_features`={out_features} is not divisble be `phm_dim`{phm_dim}"
        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = out_features // phm_dim
        self.phm_rank = phm_rank
        self.phm_rule = phm_rule
        self.phm_init_range = phm_init_range
        self.kronecker_prod_gate=kronecker_prod_gate
        self.bias_flag = bias
        self.w_init = w_init
        self.shared_W_phm = shared_W_phm 
        self.factorized_phm = factorized_phm

        if not self.shared_W_phm:
            if self.factorized_phm:
                self.W_left_para = nn.Parameter(torch.Tensor(size=(self.phm_dim*self._in_feats_per_axis, self.phm_rank)))
                self.W_right_para = nn.Parameter(torch.Tensor(size=(self.phm_dim*self.phm_rank, self._out_feats_per_axis)))
            else:
                self.W_para = nn.Parameter(torch.Tensor(size=(self.phm_dim*self._in_feats_per_axis, self._out_feats_per_axis)))

        if self.bias_flag:
            self.b = nn.Parameter(torch.Tensor(self.out_features))
        else:
            # self.register_parameter('bias', None)
            self.b = None
        self.reset_parameters()


    def init_W(self):
        if self.w_init == "glorot-normal":
            if self.factorized_phm:
                self.W_left_para.data = glorot_normal(self.W_left_para.data)
                self.W_right_para.data = glorot_normal(self.W_right_para.data)
            else:
                self.W_para.data = glorot_normal(self.W_para.data)
        elif self.w_init == "glorot-uniform":
            if self.factorized_phm:
                self.W_left_para.data = glorot_uniform(self.W_left_para.data)
                self.W_right_para.data = glorot_uniform(self.W_right_para.data)
            else:
                self.W_para.data = glorot_uniform(self.W_para.data)
        elif self.w_init == "normal":
            if self.factorized_phm:
                self.W_left_para.data.normal_(mean=0, std=self.phm_init_range)
                self.W_right_para.data.normal_(mean=0, std=self.phm_init_range)
            else:
                self.W_para.data.normal_(mean=0, std=self.phm_init_range)
        else:
            raise ValueError

    def reset_parameters(self):
        if not self.shared_W_phm:
           self.init_W()

        if self.bias_flag:
            self.b.data = torch.zeros_like(self.b.data)

    def set_phm_rule(self, phm_rule=None):
        """If factorized_phm_rules is set, phm_rule is a tuple, showing the left and right
        phm rules, and if this is not set, this is showing  the phm_rule."""
        self.phm_rule = phm_rule 

    def set_gate_weight(self, gate_weight=torch.ones(1)):
        self.weight = gate_weight
    
    def get_phm_rule(self):
        return self.phm_rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factorized_phm:
            W_combined = []
            W_left_tuple = self.W_left_para.split(self._in_feats_per_axis)
            W_right_tuple = self.W_right_para.split(self.phm_rank)
            for i in range(self.phm_dim):
                W_combined.append(torch.mm(W_left_tuple[i], W_right_tuple[i]))
            W = torch.stack(W_combined, dim=0)
            
        else:
            W_tuple = self.W_para.split(self._in_feats_per_axis)
            W = torch.stack(W_tuple, dim=0)

        return matvec_product(
               W=W,
               x=x,
               bias=self.b,
               phm_rule=self.phm_rule, 
               weight=self.weight,
               kronecker_prod_gate=self.kronecker_prod_gate)

class PHMLinearBlock(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 layer_num: int,
                 phm_dim=4,
                 phm_rule: Union[None, torch.Tensor] = None,
                 c_init: str = "normal",
                 bias: bool = True,
                 shared_phm_rule=True,
                 factorized_phm=True,
                 phm_init_range=0.0001,
                 kronecker_prod_gate=False
                 ) -> None:
        super(PHMLinearBlock, self).__init__()
        assert c_init in ["normal", "uniform"]
        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.layer_num = layer_num
        self.phm_rule = phm_rule
        self.shared_phm_rule = shared_phm_rule
        self.c_init = c_init
        self.bias = bias
        self.factorized_phm = factorized_phm
        self.phm_init_range = phm_init_range
        self.kronecker_prod_gate = kronecker_prod_gate

        # Creates and sets a shared phm_rule in case of hypercomplex adapters with a shared phm_rule.
        if self.shared_phm_rule:
            self.phm_rule = nn.Parameter(torch.FloatTensor(2*self.phm_dim*self.phm_dim, self.phm_dim))
            if self.c_init == "normal":
                self.phm_rule.data.normal_(mean=0, std=self.phm_init_range)
            elif self.c_init == "uniform":
                self.phm_rule.data.uniform_(-1, 1)
            else:
                raise NotImplementedError
        
        PhmlBlock = [PHMLinear(
                                in_features=self.in_features, 
                                out_features=self.out_features, 
                                phm_dim=self.phm_dim, 
                                factorized_phm=self.factorized_phm,
                                bias=self.bias, 
                                kronecker_prod_gate=self.kronecker_prod_gate
                              ) for i in range(2*self.layer_num)
                    ]
        self.PhmlBlock = nn.ModuleList(PhmlBlock)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phm_rule_tuple = self.phm_rule.split(self.phm_dim)
        phm_rule_k = torch.stack(phm_rule_tuple[:self.phm_dim], dim=0)
        phm_rule_v = torch.stack(phm_rule_tuple[self.phm_dim:], dim=0)
        combine_tensor = []
        for i in range(len(self.PhmlBlock)):
            if i % 2 == 0:
                self.PhmlBlock[i].set_phm_rule(phm_rule=phm_rule_k)
            else:
                self.PhmlBlock[i].set_phm_rule(phm_rule=phm_rule_v)
            self.PhmlBlock[i].set_gate_weight()
            combine_tensor.append(self.PhmlBlock[i](x))
        return torch.cat(combine_tensor, dim=2)

class PHMLinearGateBlock(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 layer_num: int,
                 phm_dim=4,
                 phm_rule: Union[None, torch.Tensor] = None,
                 c_init: str = "normal",
                 shared_phm_rule=True,
                 factorized_phm=True,
                 phm_init_range=0.0001,
                 kronecker_prod_gate=False
                 ) -> None:
        super(PHMLinearGateBlock, self).__init__()
        assert c_init in ["normal", "uniform"]
        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.layer_num = layer_num
        self.phm_rule = phm_rule
        self.shared_phm_rule = shared_phm_rule
        self.c_init = c_init
        self.factorized_phm = factorized_phm
        self.phm_init_range = phm_init_range
        self.kronecker_prod_gate = kronecker_prod_gate

        # gate
        self.gate = nn.Parameter(torch.FloatTensor(2*self.layer_num*self.in_features, self.phm_dim))
        self.gate.data.fill_(1 / self.phm_dim)
        self.phm_rep_proj = nn.Linear(in_features=self.phm_dim, out_features=1)

        # Creates and sets a shared phm_rule in case of hypercomplex adapters with a shared phm_rule.
        if self.shared_phm_rule:
            self.phm_rule = nn.Parameter(torch.FloatTensor(2*self.phm_dim*self.phm_dim, self.phm_dim))
            if self.c_init == "normal":
                self.phm_rule.data.normal_(mean=0, std=self.phm_init_range)
            elif self.c_init == "uniform":
                self.phm_rule.data.uniform_(-1, 1)
            else:
                raise NotImplementedError
        
        PhmlBlock = [PHMLinear(
                                in_features=self.in_features, 
                                out_features=self.out_features, 
                                phm_dim=self.phm_dim,
                                factorized_phm=self.factorized_phm,
                                kronecker_prod_gate=self.kronecker_prod_gate
                              ) for i in range(2*self.layer_num)
                    ]
        self.PhmlBlock = nn.ModuleList(PhmlBlock)
    
    def gate_score(self, x: torch.Tensor, phm_rule: torch.Tensor, phm_gate: torch.Tensor):
        x_proj = torch.matmul(x, phm_gate) # bsz, seq_len, phm_dim
        x_rep = torch.mean(x_proj, dim=1)  # bsz, phm_dim
        phm_rep = self.phm_rep_proj(phm_rule) # phm_dim, phm_dim, 1
        phm_rep = phm_rep.squeeze(dim=2)  # phm_dim, phm_dim
        score = torch.matmul(x_rep, phm_rep.t()) # bsz, phm_dim
        bsz = score.shape[0]
        for i in range(bsz):
            score[i].data = score[i].data / (torch.norm(score[i].data) * torch.norm(phm_rep))
        return score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phm_gate_tuple = self.gate.split(self.in_features) # 2L*(in_f ,phm_dim)
        phm_rule_tuple = self.phm_rule.split(self.phm_dim)
        phm_rule_k = torch.stack(phm_rule_tuple[:self.phm_dim], dim=0)
        phm_rule_v = torch.stack(phm_rule_tuple[self.phm_dim:], dim=0)
        combine_tensor = []
        for i in range(len(self.PhmlBlock)):
            if i % 2 == 0:
                self.PhmlBlock[i].set_phm_rule(phm_rule=phm_rule_k)
                weight = self.gate_score(x=x, phm_rule=phm_rule_k, phm_gate=phm_gate_tuple[i])
                self.PhmlBlock[i].set_gate_weight(gate_weight=weight)
            else:
                self.PhmlBlock[i].set_phm_rule(phm_rule=phm_rule_v)
                weight = self.gate_score(x=x, phm_rule=phm_rule_v, phm_gate=phm_gate_tuple[i])
                self.PhmlBlock[i].set_gate_weight(gate_weight=weight)
            combine_tensor.append(self.PhmlBlock[i](x))
        return torch.cat(combine_tensor, dim=2)

        