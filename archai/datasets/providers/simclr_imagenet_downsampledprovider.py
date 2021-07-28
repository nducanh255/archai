# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional

from overrides import overrides, EnforceOverrides
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.datasets.transforms.simclr_transforms import SimCLRTrainDataTransform,SimCLREvalDataTransform
from archai.datasets.imagenet_downsampled import ImageNet64, ImageNet32
from archai.common.config import Config
from archai.common import utils

class SimClrImageNet64Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.jitter_strength = conf_dataset['jitter_strength']
        self.input_height = conf_dataset['input_height']
        self.gaussian_blur = conf_dataset['gaussian_blur']
        MEAN = [0.48145466, 0.4578275, 0.40821073]
        STD = [0.26862954, 0.26130258, 0.27577711]
        
        if conf_dataset['normalize']:
            self.normalize = transforms.Normalize(mean = MEAN, std = STD)
        else:
            self.normalize = None

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = ImageNet64(root=self._dataroot, 
                train=True, transform=transform_train)
        if load_test:
            testset = ImageNet64(root=self._dataroot,
                train=False, transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        custom_transf = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        train_transform = SimCLRTrainDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize, custom_transf)
        test_transform = SimCLREvalDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize, custom_transf)

        return train_transform, test_transform

class SimClrImageNet32Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.jitter_strength = conf_dataset['jitter_strength']
        self.input_height = conf_dataset['input_height']
        self.gaussian_blur = conf_dataset['gaussian_blur']

        MEAN = [0.48109809, 0.45747185, 0.40785507]
        STD = [0.26040889, 0.2532126, 0.26820634]
        
        if conf_dataset['normalize']:
            self.normalize = transforms.Normalize(mean = MEAN, std = STD)
        else:
            self.normalize = None

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = ImageNet32(root=self._dataroot, 
                train=True, transform=transform_train)
        if load_test:
            testset = ImageNet32(root=self._dataroot,
                train=False, transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        custom_transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        train_transform = SimCLRTrainDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize, custom_transf)
        test_transform = SimCLREvalDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize, custom_transf)

        return train_transform, test_transform

register_dataset_provider('imagenet64_simclr', SimClrImageNet64Provider)
register_dataset_provider('imagenet32_simclr', SimClrImageNet32Provider)