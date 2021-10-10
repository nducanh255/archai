import torch.nn as nn
import torch.nn.functional as F
from archai.networks_ssl.resnet import _resnet
from archai.networks_ssl.densenet import _densenet
from archai.networks_ssl.vggnet import _vggnet
from archai.networks_ssl.vit import _vit
from archai.networks_ssl.mobilenet_v2 import mobilenet_v2
from archai.networks_ssl.efficientnet import EfficientNet
from typing import Type, Any, Callable, Union, List, Optional

# from archai.nas.model_desc import ModelDesc
# from archai.nas.model_ssl import ModelSimClr

class Projection(nn.Module):

    def __init__(self, in_features_dim:int = 2048, hidden_dim:int = 2048, out_features_dim:int = 128):
        super().__init__()
        self.out_features_dim = out_features_dim
        self.in_features_dim = in_features_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.in_features_dim, self.hidden_dim, bias=True), 
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_features_dim, bias=False),
            nn.BatchNorm1d(self.out_features_dim, affine=False), 
        )

    def forward(self, x):
        out = self.model(x)
        out = F.normalize(out, dim=1)
        return out


class ModelSimCLRResNet(nn.Module):
    
    def __init__(self, dataset: str, depth:int, layers: List[int], bottleneck: bool,
        compress:bool, hidden_dim: int, out_features:int, **kwargs: Any):
        super(ModelSimCLRResNet, self).__init__()
        self.backbone = _resnet(dataset, depth, layers, bottleneck, compress, **kwargs)
        input_dim = 512*(4 if bottleneck else 1)
        self.projection = Projection(input_dim, hidden_dim, out_features)

    def forward(self, x):
        h = self.backbone(x)[-1]
        z = self.projection(h)
        return z

class ModelSimCLRVGGNet(nn.Module):
    
    def __init__(self, dataset: str, layers: List[int], batch_norm: bool,
        hidden_dim:int, out_features: int, out_features_vgg:int, **kwargs: Any):
        super(ModelSimCLRVGGNet, self).__init__()
        self.backbone = _vggnet(dataset, layers, batch_norm, out_features_vgg = out_features_vgg, **kwargs)
        input_dim = out_features_vgg
        self.projection = Projection(input_dim, hidden_dim, out_features)

    def forward(self, x):
        h = self.backbone(x)[-1]
        z = self.projection(h)
        return z

class ModelSimCLRViT(nn.Module):
    
    def __init__(self, dim:int, hidden_dim:int, out_features: int, **kwargs: Any):
        super(ModelSimCLRViT, self).__init__()
        self.backbone = _vit(dim=dim, **kwargs)
        input_dim = dim
        self.projection = Projection(input_dim, hidden_dim, out_features)

    def forward(self, x):
        h = self.backbone(x)[-1]
        z = self.projection(h)
        return z


class ModelSimCLRDenseNet(nn.Module):
    
    def __init__(self, dataset: str, compress:bool, hidden_dim: int, out_features:int, **kwargs: Any):
        super(ModelSimCLRDenseNet, self).__init__()
        self.backbone = _densenet(dataset, compress, **kwargs)
        input_dim = self.backbone.output_features
        self.projection = Projection(input_dim, hidden_dim, out_features)

    def forward(self, x):
        h = self.backbone(x)[-1]
        z = self.projection(h)
        return z

class ModelSimCLRMobileNet(nn.Module):
    
    def __init__(self, hidden_dim: int, out_features:int, **kwargs: Any):
        super(ModelSimCLRMobileNet, self).__init__()
        self.backbone = mobilenet_v2(**kwargs)
        input_dim = self.backbone.last_channel
        self.projection = Projection(input_dim, hidden_dim, out_features)

    def forward(self, x):
        h = self.backbone(x)[-1]
        z = self.projection(h)
        return z

class ModelSimCLREfficientNet(nn.Module):
    
    def __init__(self, model_name: str, hidden_dim: int, out_features:int, **kwargs: Any):
        super(ModelSimCLREfficientNet, self).__init__()
        self.backbone = EfficientNet._efficientnet(model_name, **kwargs)
        input_dim = self.backbone.output_features
        self.projection = Projection(input_dim, hidden_dim, out_features)

    def forward(self, x):
        h = self.backbone(x)[-1]
        z = self.projection(h)
        return z
