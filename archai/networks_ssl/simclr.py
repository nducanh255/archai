import torch.nn as nn
import torch.nn.functional as F
from archai.networks_ssl.resnet import _resnet
from typing import Type, Any, Callable, Union, List, Optional

class Projection(nn.Module):

    def __init__(self, in_features_dim:int = 2048, hidden_dim:int = 2048, out_features_dim:int = 128):
        super().__init__()
        self.out_features_dim = out_features_dim
        self.in_features_dim = in_features_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.in_features_dim, self.hidden_dim), 
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_features_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        out = F.normalize(x, dim=1)
        return out


class ModelSimCLR(nn.Module):
    
    def __init__(self, dataset: str, depth:int, layers: List[int], bottleneck: bool,
        hidden_dim: int, out_features:int, **kwargs: Any):
        super(ModelSimCLR, self).__init__()
        self.backbone = _resnet(dataset, depth, layers, bottleneck, **kwargs)
        if dataset.startswith('cifar'):
            input_dim = 64*(4 if bottleneck else 1)
        elif dataset == 'imagenet':
            input_dim = 512*(4 if bottleneck else 1)
        self.projection = Projection(input_dim, hidden_dim, out_features)

    def forward(self, x):
        h = self.backbone(x)[-1]
        z = self.projection(h)
        return z