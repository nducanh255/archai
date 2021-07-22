# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional
import os

from overrides import overrides, EnforceOverrides
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.datasets.transforms.simclr_transforms import SimCLRTrainDataTransform,SimCLREvalDataTransform
from archai.common.config import Config
from archai.common import utils


class SimClrFood101Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.jitter_strength = conf_dataset['jitter_strength']
        self.input_height = conf_dataset['input_height']
        self.gaussian_blur = conf_dataset['gaussian_blur']
        MEAN = [0.5451, 0.4435, 0.3436]
        STD = [0.2171, 0.2251, 0.2260] # TODO: should be [0.2517, 0.2521, 0.2573]
        
        if conf_dataset['normalize']:
            self.normalize = transforms.Normalize(mean = MEAN, std = STD)
        else:
            self.normalize = None

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainpath = os.path.join(self._dataroot, 'food-101', 'train')
            trainset = torchvision.datasets.ImageFolder(trainpath, transform=transform_train)
        if load_test:
            testpath = os.path.join(self._dataroot, 'food-101', 'test')
            testset = torchvision.datasets.ImageFolder(testpath, transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        # TODO: Need to rethink the food101 transforms
        custom_transf = [
            transforms.Resize((32,32)),
            transforms.RandomCrop(32, padding=4),
        ]
        train_transform = SimCLRTrainDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize, custom_transf)
        test_transform = SimCLREvalDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize, custom_transf)

        return train_transform, test_transform

register_dataset_provider('food101_simclr', SimClrFood101Provider)