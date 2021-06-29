# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from overrides import overrides
from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.datasets.transforms.simclr_transforms import SimCLRTrainDataTransform,SimCLREvalDataTransform
from archai.common.config import Config
from archai.common import utils
from torchvision import transforms
from torchvision import datasets
import torchvision

class SimClrImageNetProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.jitter_strength = conf_dataset['jitter_strength']
        self.input_height = conf_dataset['input_height']
        self.gaussian_blur = conf_dataset['gaussian_blur']
        if conf_dataset['normalize']:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.normalize = None

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = datasets.ImageFolder(root=os.path.join(self._dataroot, 'ImageNet', 'train'),
                transform=transform_train)
            # compatibility with older PyTorch
            if not hasattr(trainset, 'targets'):
                trainset.targets = [lb for _, lb in trainset.samples]
        if load_test:
            testset = datasets.ImageFolder(root=os.path.join(self._dataroot, 'ImageNet', 'val'),
                transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        train_transform = SimCLRTrainDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize)
        test_transform = SimCLREvalDataTransform(self.input_height,
            self.gaussian_blur, self.jitter_strength, self.normalize)

        return train_transform, test_transform

register_dataset_provider('imagenet_simclr', SimClrImageNetProvider)