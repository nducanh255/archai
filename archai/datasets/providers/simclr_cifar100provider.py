# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from overrides import overrides
from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.datasets.transforms.simclr_transforms import SimCLREvalLinearTransform, SimCLRTrainDataTransform,SimCLREvalDataTransform
from archai.common.config import Config
from archai.common import utils
from torchvision import transforms
import torchvision

# def create_simclr_provider(base_class:DatasetProvider, conf_dataset:Config)->DatasetProvider:
#     print(base_class)
#     class SimClrProvider(base_class):
#         def __init__(self, conf_dataset:Config):
#             super().__init__(conf_dataset)
#             self._dataroot = utils.full_path(conf_dataset['dataroot'])
#             self.jitter_strength = conf_dataset['jitter_strength']
#             self.input_height = conf_dataset['input_height']
#             self.gaussian_blur = conf_dataset['gaussian_blur']
#             self.normalize = conf_dataset['normalize']
#             # ds_name = conf_dataset['name']
#             # ds_provider_type = get_provider_type(ds_name)
#             # self.parent_ds = ds_provider_type(conf_dataset)

#         @overrides
#         def get_transforms(self)->tuple:
#             train_transform = SimCLRTrainDataTransform(self.input_height,
#                 self.gaussian_blur, self.jitter_strength, self.normalize)
#             test_transform = SimCLREvalDataTransform(self.input_height,
#                 self.gaussian_blur, self.jitter_strength, self.normalize)

#             return train_transform, test_transform

#     return SimClrProvider(conf_dataset)

class SimClrCifar100Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.jitter_strength = conf_dataset['jitter_strength']
        self.input_height = conf_dataset['input_height']
        self.gaussian_blur = conf_dataset['gaussian_blur']
        self.mode = conf_dataset['mode'] if 'mode' in conf_dataset else 'pretrain'
        if conf_dataset['normalize']:
            self.normalize = transforms.Normalize(
                                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                )
        else:
            self.normalize = None

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = torchvision.datasets.CIFAR100(root=self._dataroot, train=True,
                download=True, transform=transform_train)
        if load_test:
            testset = torchvision.datasets.CIFAR100(root=self._dataroot, train=False,
                download=True, transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:

        if self.mode == 'pretrain':
            train_transform = SimCLRTrainDataTransform(self.input_height,
                self.gaussian_blur, self.jitter_strength, self.normalize)
            test_transform = SimCLREvalDataTransform(self.input_height,
                self.gaussian_blur, self.jitter_strength, self.normalize)
        elif self.mode == 'eval_linear':
            train_transform = SimCLREvalLinearTransform(self.input_height,
                self.normalize, is_train=True)
            test_transform = SimCLREvalLinearTransform(self.input_height,
                self.normalize, is_train=False)
        elif self.mode == 'transfer':
            train_transform = SimCLREvalLinearTransform(self.input_height,
                self.normalize, is_transfer=True)
            test_transform = SimCLREvalLinearTransform(self.input_height,
                self.normalize, is_transfer=True)

        return train_transform, test_transform

register_dataset_provider('cifar100_simclr', SimClrCifar100Provider)