import torch.nn as nn
from utils.hypercomplex.phm_layer import PHMLinear
class BaseSublayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 mid_features: int,
                 out_features: int,
                 ):
        super().__init__()
        self.in_features = in_features
        self.mid_features= mid_features
        self.out_features = out_features
        self.ff1 = nn.Linear(self.in_features, self.mid_features)
        self.activate = nn.ReLU()
        self.ff2 = nn.Linear(self.mid_features, self.out_features)
        self.ff2.weight.data.zero_()

    def forward(self, x):
        return self.ff2(self.activate(self.ff1(x)))
        

class BasePhmSublayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 mid_features: int,
                 out_features: int,
                 layer_num: int,
                 phm_dim: int,
                 phm_rule_expert_down=None,
                 phm_rule_expert_up=None,
                 phm_rule_shared_down=None,
                 phm_rule_shared_up=None,
                 factorized_phm: bool=False,
                 phm_rank=1,
                 strategy: str='mat',
                 share_kv: bool=False,
                 ):
        super().__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.layer_num = layer_num
        self.phm_dim = phm_dim
        self.phm_rule_expert_down = phm_rule_expert_down
        self.phm_rule_expert_up = phm_rule_expert_up
        self.phm_rule_shared_down = phm_rule_shared_down
        self.phm_rule_shared_up = phm_rule_shared_up
        self.factorized_phm = factorized_phm
        self.phm_rank = phm_rank
        self.strategy = strategy
        self.share_kv = share_kv
        self.ff1 = PHMLinear(
                                in_features=self.in_features,
                                out_features=self.mid_features,
                                layer_num=1,
                                phm_dim=self.phm_dim,
                                phm_rule=self.phm_rule_shared_down,
                                phm_rule_expert=self.phm_rule_expert_down,
                                phm_rank=self.phm_rank,
                                factorized_phm=self.factorized_phm,
                                strategy=self.strategy,
                                share_kv=self.share_kv
                            )
        self.activate = nn.ReLU()
        self.ff2 = PHMLinear(
                             in_features=self.mid_features,
                             out_features=self.out_features,
                             layer_num=self.layer_num,
                             phm_dim=self.phm_dim,
                             phm_rule=self.phm_rule_shared_up,
                             phm_rule_expert=self.phm_rule_expert_up,
                             factorized_phm=self.factorized_phm,
                             phm_rank=self.phm_rank,
                             strategy=self.strategy
                            )
        self.ff2.W_para.data.zero_()
        self.ff2.b.data.zero_()

    def forward(self, x):
        return self.ff2(self.activate(self.ff1(x)))
        
class BaseLPSublayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 mid_features: int,
                 out_features: int,
                 layer_num: int,
                 phm_dim: int,
                 phm_rule_expert_down=None,
                 phm_rule_expert_up=None,
                 phm_rule_shared_down=None,
                 phm_rule_shared_up=None,
                 factorized_phm: bool=False,
                 phm_rank=1,
                 strategy: str='mat',
                 share_kv: bool=False,
                 ):
        super().__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.layer_num = layer_num
        self.phm_dim = phm_dim
        self.phm_rule_expert_down = phm_rule_expert_down
        self.phm_rule_expert_up = phm_rule_expert_up
        self.phm_rule_shared_down = phm_rule_shared_down
        self.phm_rule_shared_up = phm_rule_shared_up
        self.factorized_phm = factorized_phm
        self.phm_rank = phm_rank
        self.strategy = strategy
        self.share_kv = share_kv
        self.ff1 = nn.Linear(in_features=self.in_features, out_features=self.mid_features)
        self.activate = nn.ReLU()
        self.ff2 = PHMLinear(
                             in_features=self.mid_features,
                             out_features=self.out_features,
                             layer_num=self.layer_num,
                             phm_dim=self.phm_dim,
                             phm_rule=self.phm_rule_shared_up,
                             phm_rule_expert=self.phm_rule_expert_up,
                             factorized_phm=self.factorized_phm,
                             phm_rank=self.phm_rank,
                             strategy=self.strategy
                            )
        self.ff2.W_para.data.zero_()
        self.ff2.b.data.zero_()

    def forward(self, x):
        return self.ff2(self.activate(self.ff1(x)))

class BasePLSublayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 mid_features: int,
                 out_features: int,
                 layer_num: int,
                 phm_dim: int,
                 phm_rule_expert_down=None,
                 phm_rule_expert_up=None,
                 phm_rule_shared_down=None,
                 phm_rule_shared_up=None,
                 factorized_phm: bool=False,
                 phm_rank=1,
                 strategy: str='mat',
                 share_kv: bool=False,
                 ):
        super().__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.layer_num = layer_num
        self.phm_dim = phm_dim
        self.phm_rule_expert_down = phm_rule_expert_down
        self.phm_rule_expert_up = phm_rule_expert_up
        self.phm_rule_shared_down = phm_rule_shared_down
        self.phm_rule_shared_up = phm_rule_shared_up
        self.factorized_phm = factorized_phm
        self.phm_rank = phm_rank
        self.strategy = strategy
        self.share_kv = share_kv
        self.ff1 = PHMLinear(
                                in_features=self.in_features,
                                out_features=self.mid_features,
                                layer_num=1,
                                phm_dim=self.phm_dim,
                                phm_rule=self.phm_rule_shared_down,
                                phm_rule_expert=self.phm_rule_expert_down,
                                phm_rank=self.phm_rank,
                                factorized_phm=self.factorized_phm,
                                strategy=self.strategy,
                                share_kv=self.share_kv
                            )
        self.activate = nn.ReLU()
        self.ff2 = nn.Linear(in_features=self.mid_features, out_features=self.out_features)
        self.ff2.weight.data.zero_()
    def forward(self, x):
        return self.ff2(self.activate(self.ff1(x)))