import torch
import torch.nn as nn
from models.hypercomplex.phm_layer import PHMLinear
class BaseSublayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ff2 = nn.Linear(self.in_features, self.out_features)
        self.ff2.weight.data.zero_()

    def forward(self, x):
        return self.ff2(x)
        

class BasePhmSublayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 layer_num: int,
                 phm_dim: int,
                 phm_rule_expert: None,
                 strategy: str=''
                 ):
        super().__init__()
        self.activation_fn = nn.ReLU()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_num = layer_num
        self.phm_dim = phm_dim
        self.phm_rule_expert = phm_rule_expert
        self.strategy = strategy
        self.ff2 = PHMLinear(
                             in_features=self.in_features,
                             out_features=self.out_features,
                             layer_num=self.layer_num,
                             phm_dim=self.phm_dim,
                             strategy=self.strategy
                            )
        self.ff2.set_phm_rule(phm_rule=self.phm_rule_expert, strategy=self.strategy)

    def forward(self, x):
        return self.ff2(x)
        