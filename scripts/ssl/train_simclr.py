# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from torch.nn.modules import dropout
import yaml
import time
import torch
import shutil
from archai.networks_ssl.simclr import ModelSimCLRResNet, ModelSimCLRVGGNet, ModelSimCLRViT, ModelSimCLRDenseNet
from archai.common import utils
from archai.common.trainer_ssl import TrainerSimClr
from archai.common.config import Config
from archai.common.common import common_init, create_conf, get_state, init_from, update_envvars
from archai.datasets import data
from archai.common.checkpoint import CheckPoint

def train_test(conf_main:Config):
    conf_loader = conf_main['loader']
    conf_trainer = conf_main['trainer']
    conf_dataset = conf_loader['dataset']
    conf_common = conf_main['common']
    conf_checkpoint = conf_common['checkpoint']
    if "resnet" in conf_trainer['model']:
        with open('confs/algos/simclr_resnets.yaml', 'r') as f:
            conf_models = yaml.load(f, Loader=yaml.Loader)
    elif "vgg" in conf_trainer['model']:
        with open('confs/algos/simclr_vggnets.yaml', 'r') as f:
            conf_models = yaml.load(f, Loader=yaml.Loader)
    elif "vit" in conf_trainer['model']:
        with open('confs/algos/simclr_vits.yaml', 'r') as f:
            conf_models = yaml.load(f, Loader=yaml.Loader)
            conf_models = yaml.load(f, Loader=yaml.Loader)
    elif "densenet" in conf_trainer['model']:
        with open('confs/algos/simclr_densenets.yaml', 'r') as f:
            conf_models = yaml.load(f, Loader=yaml.Loader)
    else:
        raise Exception(f"Not implemented SimCLR for model {conf_trainer['model']}")
        
    conf_model = conf_models[conf_trainer['model']]

    if "resnet" in conf_trainer['model']:
        model = ModelSimCLRResNet(conf_dataset['name'], conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
            conf_model['compress'], conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
            width_per_group=conf_model['width_per_group'])
    elif "vgg" in conf_trainer['model']:
        model = ModelSimCLRVGGNet(conf_dataset['name'], conf_model['layers'], conf_model['batch_norm'], conf_model['hidden_dim'], 
            conf_model['out_features'], conf_model['out_features_vgg'], classifier_type = conf_model['classifier_type'], 
            init_weights = True, drop_prob=conf_model['drop_prob'], hidden_features_vgg=conf_model['hidden_features_vgg'])
    elif "vit" in conf_trainer['model']:
        model = ModelSimCLRViT(image_size = conf_dataset["input_height"], patch_size = conf_model["patch_size"], dim = conf_model["dim"],
                depth = conf_model["depth"], heads = conf_model["heads"], mlp_dim = conf_model["mlp_dim"], pool = conf_model["pool"],
                channels = conf_model["channels"], dim_head = conf_model["dim_head"], dropout = conf_model["dropout"],
                emb_dropout = conf_model["emb_dropout"], hidden_dim = conf_model["hidden_dim"], out_features = conf_model["out_features"])
    elif "densenet" in conf_trainer['model']:
        model = ModelSimCLRDenseNet(conf_dataset['name'], growth_rate = conf_model['growth_rate'], block_config=conf_model['block_config'],
                num_init_features = conf_model['num_init_features'], hidden_dim = conf_model["hidden_dim"], 
                out_features = conf_model["out_features"])
    model = model.to(torch.device('cuda', 0))
    print('Number of trainable params: {:.2f}M'
          .format(sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6))
    # get data
    data_loaders = data.get_data_ssl(conf_loader)

    # train!
    ckpt = CheckPoint(conf_checkpoint, load_existing=False)
    if conf_checkpoint['resume']:
        resumedir = conf_checkpoint['resumedir']
        experiment_name = conf_checkpoint['experiment_name']
        resumedir = utils.full_path(resumedir)
        resumedir = os.path.join(resumedir, experiment_name)
        filename = os.path.basename(utils.full_path(conf_checkpoint['filename']))
        filepath = os.path.join(resumedir, filename)
        if os.path.exists(filepath):
            print("Resuming")
            found = ckpt.resume(filepath)
        else:
            print('Resume ckpt not found')
            # if os.path.exists(conf_common['resumedir']):
            #     shutil.rmtree(conf_common['resumedir'])
            conf_common['resume'] = conf_checkpoint['resume'] = conf_common['apex']['resume'] = \
            conf_common['apex']['resume'] = conf_common['apex']['resume'] = False
            conf_common['resumedir'] = conf_checkpoint['resumedir'] = ''
    trainer = TrainerSimClr(conf_trainer, model, ckpt)
    st = time.time()
    trainer.fit(data_loaders)
    print('Time taken:', time.time()-st)
    # if conf_common['save_intermediate']:
    #     shutil.rmtree(conf_common['intermediatedir'])


if __name__ == '__main__':
    if utils.is_main_process():
        conf = common_init(config_filepath='confs/algos/simclr.yaml')
        print('Running main process')
    else:
        conf = create_conf(config_filepath='confs/algos/simclr.yaml')
        Config.set_inst(conf)
        update_envvars(conf)
        commonstate = get_state()
        init_from(commonstate,recreate_logger=False)
        print('Running child process')
    train_test(conf)


