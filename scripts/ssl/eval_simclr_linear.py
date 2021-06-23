# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
import torch
import torch.nn as nn
from archai.networks_ssl.simclr import ModelSimCLR
from archai.common.trainer import TrainerLinear
from archai.common.config import Config
from archai.common.common import common_init
from archai.datasets import data
from archai.common.checkpoint import CheckPoint

def train_test(conf:Config):
    conf_loader = conf['loader']
    conf_trainer = conf['trainer']
    conf_dataset = conf_loader['dataset']
    conf_checkpoint = conf['common']['checkpoint']
    train_config_path = os.path.join(os.path.dirname(conf['common']['load_checkpoint']),"config_used.yaml")
    with open(train_config_path) as f:
        config_train = yaml.load(f, Loader=yaml.Loader)
    # conf_trainer['batch_chunks'] = config_train['trainer']['batch_chunks']
    conf_trainer['model'] = config_train['trainer']['model']
    # conf_dataset['jitter_strength'] = config_train['loader']['dataset']['jitter_strength']

    conf_model = conf[conf_trainer['model']]

    # create model
    model = ModelSimCLR(conf_dataset['name'], conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
        conf_model['compress'], conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
        width_per_group=conf_model['width_per_group'])
    model = model.to(torch.device('cuda', 0))
    ckpt = torch.load(conf['common']['load_checkpoint'])
    print("Loading model from epoch {}".format(ckpt['trainer']['last_epoch']+1))
    model.load_state_dict(ckpt['trainer']['model'])
    # print('Number of trainable params: {:.2f}M'
    #       .format(sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6))
    # exit()
   
    input_dim = (64 if conf_model['compress'] else 512)*(4 if conf_model['bottleneck'] else 1)
    model.fc = nn.Linear(input_dim, conf_dataset['n_classes'])

    # get data
    data_loaders = data.get_data(conf_loader)

    # train!
    ckpt = CheckPoint(conf_checkpoint, load_existing=False)
    trainer = TrainerLinear(conf_trainer, model)
    trainer.fit(data_loaders)


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/algos/simclr_eval.yaml;confs/algos/simclr_resnets.yaml')

    train_test(conf)


