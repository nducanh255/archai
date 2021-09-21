# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from torch.nn.modules import dropout
import yaml
import time
import torch
import shutil
from archai.networks_ssl.simclr import ModelSimCLRResNet, ModelSimCLRVGGNet, ModelSimCLRViT, ModelSimCLRDenseNet, ModelSimCLREfficientNet
from archai.common import utils
from archai.common.trainer_ssl import TrainerSimClr
from archai.common.config import Config
from archai.common.common import common_init, create_conf, get_state, init_from, update_envvars
from archai.datasets import data
from archai.common.checkpoint import CheckPoint
from archai.common.apex_utils import ApexUtils

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
    elif "densenet" in conf_trainer['model']:
        with open('confs/algos/simclr_densenets.yaml', 'r') as f:
            conf_models = yaml.load(f, Loader=yaml.Loader)
    elif "efficientnet" in conf_trainer['model']:
        with open('confs/algos/simclr_efficientnets.yaml', 'r') as f:
            conf_models = yaml.load(f, Loader=yaml.Loader)
    else:
        raise Exception(f"Not implemented SimCLR for model {conf_trainer['model']}")
        
    conf_model = conf_models[conf_trainer['model']]

    if "efficientnet" in conf_trainer['model']:
        conf_loader['dataset']['input_height'] = conf_model['res']
    # small_datasets = ['cifar10', 'cifar100', 'aircraft', 'mnist', 'fashion_mnist', 'food101', 'svhn', 'imagenet32', 'imagenet64']
    if 'compress' not in conf_model:
        conf_model['compress'] = conf_loader['dataset']['input_height']<=64

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
        model = ModelSimCLRDenseNet(conf_dataset['name'], conf_model['compress'], growth_rate = conf_model['growth_rate'],
                block_config=conf_model['block_config'], num_init_features = conf_model['num_init_features'], 
                hidden_dim = conf_model["hidden_dim"], out_features = conf_model["out_features"])
    elif "efficientnet" in conf_trainer['model']:
        model = ModelSimCLREfficientNet(conf_trainer['model'], hidden_dim = conf_model["hidden_dim"], out_features = conf_model["out_features"],
                load_pretrained = False, width_coefficient = conf_model['width'], depth_coefficient = conf_model['depth'],
                image_size = conf_model['res'], dropout_rate = conf_model['dropout']
                )
    model = model.to(torch.device('cuda', 0))
    print('Number of trainable params: {:.2f}M'
          .format(sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6))
    # get data
    data_loaders = data.get_data_ssl(conf_loader)

    # train!
    ckpt = CheckPoint(conf_checkpoint, load_existing=False)
    apex = ApexUtils(conf_common['apex'], logger=None)
    apex.sync_devices()
    apex.barrier()
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
            conf_checkpoint['resume'] = False
            conf_checkpoint['resumedir'] = ''
    if apex.is_master():
        conf_wandb = conf_common['wandb']
        if conf_wandb['enabled']:
            import wandb
            import hashlib
            id = hashlib.md5(conf_wandb['run_name'].encode('utf-8')).hexdigest()
            wandb.init(project=conf_wandb['project_name'],
                        name=conf_wandb['run_name'],
                        config=conf_main,
                        id=id,
                        resume=conf_common['resume'],
                        dir=os.path.join(conf_common['logdir']),
                        entity=conf_wandb['entity'])
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


