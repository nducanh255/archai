import torch
import torch.nn as nn
# from .._internally_replaced_utils import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        dataset: str,
        features: nn.Module,
        classifier_type: str = "A", # Choose from 'A', 'B', 'C'
        init_weights: bool = True,
        drop_prob: float = 0.5,
        hidden_features_vgg: int = 4096,
        out_features_vgg: int = 4096
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.dataset = dataset
        if self.dataset == 'imagenet':
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.in_dim = 512*7*7
        elif self.dataset.startswith('cifar'):
            self.in_dim = 512
        else:
            raise Exception(f'Not implemented VGG for {self.dataset} dataset')
        if classifier_type == "A":
            self.classifier = nn.Sequential(
                nn.Linear(self.in_dim, hidden_features_vgg),
                nn.ReLU(True),
                nn.Dropout(p = drop_prob),
                nn.Linear(hidden_features_vgg, out_features_vgg),
                nn.ReLU(True),
                nn.Dropout(p = drop_prob)
            )
        elif classifier_type == "B":
            self.classifier = nn.Sequential(
                nn.Linear(self.in_dim, hidden_features_vgg),
                nn.ReLU(True),
                nn.Dropout(p = drop_prob),
                nn.Linear(hidden_features_vgg, out_features_vgg),
                nn.ReLU(True),
            )
        elif classifier_type == "C":
            self.classifier = nn.Sequential(
                nn.Linear(self.in_dim, hidden_features_vgg),
                nn.ReLU(True),
                nn.Dropout(p = drop_prob),
                nn.Linear(hidden_features_vgg, out_features_vgg),
            )
        elif classifier_type == "D":
            self.classifier = nn.Sequential(
                nn.Linear(self.in_dim, out_features_vgg),
            )
        else:
            raise Exception(f"Classifier type {classifier_type} not defined")
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.dataset == 'imagenet':
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return [x]

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vggnet(dataset: str, layers: List[int], batch_norm: bool, pretrained: bool = False, progress: bool = False, **kwargs: Any) -> VGG:
    layer_list = [64]*layers[0]+['M'] + [128]*layers[1]+['M'] + [256]*layers[2] + ['M'] + [512]*layers[3] + ['M'] + [512]*layers[4] + ['M']
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(dataset, make_layers(layer_list, batch_norm=batch_norm), **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model
