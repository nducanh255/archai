# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional

from overrides import overrides, EnforceOverrides
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.datasets.transforms.simclr_transforms import SimCLRTrainDataTransform,SimCLREvalDataTransform
from archai.common.config import Config
from archai.common import utils

class SimClrMnistProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.jitter_strength = conf_dataset['jitter_strength']
        self.input_height = conf_dataset['input_height']
        self.gaussian_blur = conf_dataset['gaussian_blur']
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]

        if conf_dataset['normalize']:
            self.normalize = transforms.Normalize(mean = MEAN, std = STD)
        else:
            self.normalize = None

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                    transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = torchvision.datasets.MNIST(root=self._dataroot,
                train=True, download=True, transform=transform_train)
        if load_test:
            testset = torchvision.datasets.MNIST(root=self._dataroot,
                train=False, download=True, transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                scale=(0.9, 1.1), shear=0.1),
        ]
        train_transform = SimCLRTrainDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize, transf)
        test_transform = SimCLREvalDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize, transf)

        return train_transform, test_transform

register_dataset_provider('mnist_simclr', SimClrMnistProvider)