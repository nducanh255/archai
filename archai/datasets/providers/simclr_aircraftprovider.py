# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional
import os

from overrides import overrides, EnforceOverrides
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms
from archai.datasets.transforms.simclr_transforms import SimCLRTrainDataTransform,SimCLREvalDataTransform
from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.common import utils


class SimClrAircraftProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.jitter_strength = conf_dataset['jitter_strength']
        self.input_height = conf_dataset['input_height']
        self.gaussian_blur = conf_dataset['gaussian_blur']
        self.mode = conf_dataset['mode'] if 'mode' in conf_dataset else 'pretrain'
        # TODO: update, currently mit67 values
        MEAN = [0.4893, 0.4270, 0.3625]
        STD = [0.2631, 0.2565, 0.2582]
        
        if conf_dataset['normalize']:
            self.normalize = transforms.Normalize(mean = MEAN, std = STD)
        else:
            self.normalize = None

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainpath = os.path.join(self._dataroot, 'aircraft', 'train')            
            trainset = torchvision.datasets.ImageFolder(trainpath, transform=transform_train)
        if load_test:
            testpath = os.path.join(self._dataroot, 'aircraft', 'test')
            testset = torchvision.datasets.ImageFolder(testpath, transform=transform_test)

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

register_dataset_provider('aircraft_simclr', SimClrAircraftProvider)