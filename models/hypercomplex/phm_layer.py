# The codes are from https://github.com/bayer-science-for-a-better-life/phc-gnn
import torch
import torch.nn as nn
from typing import Union
from .inits import glorot_uniform, glorot_normal  #phm_init
from .kronecker import kronecker_product_einsum_batched

def matvec_product( W: torch.Tensor, 
                    x: torch.Tensor,
                    bias: Union[None, torch.Tensor],
                    phm_rule: Union[None, torch.Tensor],
                    ) -> torch.Tensor:
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
    _, phm_dim, phm_dim = phm_rule.shape
    H = kronecker_product_einsum_batched(phm_rule, W)
    _, in_dim, out_dim = H.shape
    H = H.reshape(-1, phm_dim, in_dim, out_dim).sum(1)
    H_list = H.split(1)
    H_combined = torch.cat(H_list, dim=-1).squeeze(dim=0)
    y = torch.matmul(input=x, other=H_combined)
    if bias is not None:
        y += bias
    return y

class PHMLinear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 layer_num: int,
                 phm_dim: int,
                 bias: bool = True,
                 w_init: str = "glorot-uniform",
                 c_init: str = "normal",
                 phm_rule: Union[None, torch.Tensor] = None,
                 factorized_phm: bool = True,
                 phm_rule_expert = None,
                 strategy: str = '',
                 phm_rank = 1,
                 phm_init_range=0.0001,
                 ) -> None:
        super(PHMLinear, self).__init__()
        assert w_init in ["glorot-normal", "glorot-uniform", "normal"]
        assert in_features % phm_dim == 0, f"Argument `in_features`={in_features} is not divisble be `phm_dim`{phm_dim}"
        assert out_features % phm_dim == 0, f"Argument `out_features`={out_features} is not divisble be `phm_dim`{phm_dim}"
        self.in_features = in_features
        self.out_features = out_features
        self.layer_num = layer_num
        self.phm_dim = phm_dim
        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = out_features // phm_dim
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.phm_rule_expert = phm_rule_expert
        self.factorized_phm = factorized_phm
        self.bias_flag = bias
        self.w_init = w_init
        self.c_init = c_init
        self.strategy = strategy
      
        if self.factorized_phm:
            self.W_left_para = nn.Parameter(torch.Tensor(size=(2*self.layer_num*self.phm_dim*self._in_feats_per_axis, self.phm_rank)))
            self.W_right_para = nn.Parameter(torch.Tensor(size=(2*self.layer_num*self.phm_dim*self.phm_rank, self._out_feats_per_axis)))
        else:
            self.W_para = nn.Parameter(torch.Tensor(size=(2*self.layer_num*self.phm_dim*self._in_feats_per_axis, self._out_feats_per_axis)))

        if self.bias_flag:
            self.b = nn.Parameter(torch.Tensor(2*self.layer_num*self.out_features))
        else:
            self.register_parameter('b', None)
            self.b = None

        if phm_rule == None:
            # layer_num > 1 means shared, layer_num == 1 means not shared
            if self.strategy == 'mat':
                self.phm_rule = nn.Parameter(torch.FloatTensor(2*self.phm_dim*self.phm_dim, self.phm_rank))
            elif self.strategy == 'concat':
                self.phm_rule = nn.Parameter(torch.FloatTensor(2*self.phm_dim*self.phm_dim, self.phm_dim // 2))
            elif self.strategy == 'plus':
                self.phm_rule = nn.Parameter(torch.FloatTensor(2*self.phm_dim*self.phm_dim, self.phm_dim))
        else:
            self.phm_rule = phm_rule
            

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
        '''
        if self.factorized_phm:
            self.W_left_para.data.zero_()
            self.W_right_para.data.zero_()
        else:
            self.W_para.data.zero_()
        '''

    def init_phm_rule(self):
        if self.c_init == "normal":
            self.phm_rule.data.normal_(mean=0, std=self.phm_init_range)
        elif self.c_init == "uniform":
            self.phm_rule.data.uniform_(-1, 1)
        else:
            raise NotImplementedError

        # self.phm_rule.data.zero_()

    def reset_parameters(self):
        self.init_W()
        self.init_phm_rule()

        if self.bias_flag:
            self.b.data = torch.zeros_like(self.b.data)
    
    def get_phm_rule(self):
        return self.phm_rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factorized_phm:
            W_left_para = self.W_left_para.reshape(2*self.layer_num*self.phm_dim, self._in_feats_per_axis, self.phm_rank)
            W_right_para = self.W_right_para.reshape(2*self.layer_num*self.phm_dim, self.phm_rank, self._out_feats_per_axis)
            W = torch.matmul(W_left_para, W_right_para)
        else:
            W_para = self.W_para.reshape(2*self.layer_num*self.phm_dim, self._in_feats_per_axis, self._out_feats_per_axis)
            W = W_para

        phm_rule_share = self.phm_rule.reshape(2*self.phm_dim, self.phm_dim, -1)
        phm_rule_share = phm_rule_share.repeat(self.layer_num, 1, 1)
        
        if self.phm_rule_expert != None:
            if self.strategy == 'mat':
                phm_rule_expert = self.phm_rule_expert.reshape(2*self.layer_num*self.phm_dim, self.phm_rank, self.phm_dim)
                phm_rule_share = torch.matmul(phm_rule_share, phm_rule_expert)
      
            if self.strategy == 'plus':
                phm_rule_expert = self.phm_rule_expert.reshape(2*self.layer_num*self.phm_dim, self.phm_dim, self.phm_dim)
                phm_rule_share += phm_rule_expert
            elif self.strategy == 'concat':
                phm_rule_expert = self.phm_rule_expert.reshape(2*self.layer_num*self.phm_dim, self.phm_dim, self.phm_dim // 2)
                phm_rule_share = torch.cat([phm_rule_share, phm_rule_expert], dim=-1)
        
        return matvec_product(
               W=W,
               x=x,
               bias=self.b,
               phm_rule=phm_rule_share, 
            )
        
