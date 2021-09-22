# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import List, Tuple, Union, Optional

from overrides import overrides, EnforceOverrides
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms
from torch.utils.data import ConcatDataset

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.datasets.transforms.simclr_transforms import SimCLREvalLinearTransform,SimCLRTrainDataTransform,SimCLREvalDataTransform
from archai.common.config import Config
from archai.common import utils


class SimClrSvhnProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.jitter_strength = conf_dataset['jitter_strength']
        self.input_height = conf_dataset['input_height']
        self.gaussian_blur = conf_dataset['gaussian_blur']
        self.mode = conf_dataset['mode'] if 'mode' in conf_dataset else 'pretrain'
        MEAN = [0.4914, 0.4822, 0.4465]
        STD = [0.2023, 0.1994, 0.20100]
        
        if conf_dataset['normalize']:
            self.normalize = transforms.Normalize(mean = MEAN, std = STD)
        else:
            self.normalize = None

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = torchvision.datasets.SVHN(root=self._dataroot, split='train',
                download=True, transform=transform_train)
            extraset = torchvision.datasets.SVHN(root=self._dataroot, split='extra',
                download=True, transform=transform_train)
            oldtrainset = trainset
            trainset = ConcatDataset([trainset, extraset])
            trainset.targets = np.concatenate((oldtrainset.labels, extraset.labels))
            print(trainset.targets)
        if load_test:
            testset = torchvision.datasets.SVHN(root=self._dataroot, split='test',
                download=True, transform=transform_test)
            testset.targets = testset.labels

        return trainset, testset

    # @overrides
    # def get_transforms(self)->tuple:
    #     custom_transf = [
    #         transforms.RandomCrop(32, padding=4),
    #     ]
    #     train_transform = SimCLRTrainDataTransform(self.input_height,
    #         self.gaussian_blur, self.jitter_strength, self.normalize, custom_transf)
    #     test_transform = SimCLREvalDataTransform(self.input_height,
    #         self.gaussian_blur, self.jitter_strength, self.normalize, custom_transf)

    #     return train_transform, test_transform

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

register_dataset_provider('svhn_simclr', SimClrSvhnProvider)