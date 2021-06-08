# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from archai.networks_ssl.simclr import ModelSimCLR
from archai.common.trainer_ssl import TrainerSimClr
from archai.common.config import Config
from archai.common.common import common_init
from archai.datasets import data

def train_test(conf_eval:Config):
    conf_loader = conf_eval['loader']
    conf_trainer = conf_eval['trainer']
    conf_dataset = conf_loader['dataset']
    conf_model = conf_trainer['model']

    # create model
    model = ModelSimCLR(conf_dataset['name'], conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
        conf_model['hidden_dim'], conf_model['out_features'])
    model = model.to(torch.device('cuda', 0))

    # get data
    data_loaders = data.get_data_ssl(conf_loader)

    # train!
    trainer = TrainerSimClr(conf_trainer, model)
    trainer.fit(data_loaders)


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/algos/simclr.yaml;confs/datasets/cifar100.yaml')
    conf_eval = conf['nas']['eval']

    train_test(conf_eval)


