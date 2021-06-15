# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional

from overrides import overrides, EnforceOverrides
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.datasets.transforms.simclr_transforms import SimCLRFinetuneTransform
from archai.common.config import Config
from archai.common import utils


class Cifar100Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.jitter_strength = conf_dataset['jitter_strength']
        self.input_height = conf_dataset['input_height']
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
        # MEAN = [0.507, 0.487, 0.441]
        # STD = [0.267, 0.256, 0.276]
        # transf = [
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip()
        # ]

        # normalize = [
        #     transforms.ToTensor(),
        #     transforms.Normalize(MEAN, STD)
        # ]

        # train_transform = transforms.Compose(transf + normalize)
        # test_transform = transforms.Compose(normalize)

        train_transform = SimCLRFinetuneTransform(self.input_height, self.jitter_strength, self.normalize, False)
        test_transform = SimCLRFinetuneTransform(self.input_height, self.jitter_strength, self.normalize, True)
        return train_transform, test_transform

register_dataset_provider('cifar100', Cifar100Provider)