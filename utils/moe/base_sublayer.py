import torch
import torch.nn as nn
class BaseSublayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 mid_features: int,
                 out_features: int,
                 expert_struct: str = 'MLP_split_to_layers_w_share'
                 ):
        super().__init__()
        self.expert_struct = expert_struct
        self.activation_fn = nn.ReLU()
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.ff1 = nn.Linear(self.in_features, self.mid_features)
        self.ff2 = nn.Linear(self.mid_features, self.out_features)

    def forward(self, x):
        if 'wo_share' in self.expert_struct:
            return self.ff2(self.activation_fn(self.ff1(x)))
        elif 'w_share' in self.expert_struct:
            return self.ff2(x)
        else:
            raise ValueError("Other expert structures are not supported yet!")