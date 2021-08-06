# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
import torch
import torch.nn as nn
from archai.common import utils
from archai.algos.darts.darts_model_desc_builder_ssl import DartsModelDescBuilderSimClr
from archai.networks_ssl.simclr import ModelSimCLRResNet, ModelSimCLRVGGNet, ModelSimCLRViT
from archai.common.trainer import TrainerLinear
from archai.common.config import Config
from archai.common.common import _create_sysinfo, common_init, create_conf, get_state, init_from, update_envvars
from archai.datasets import data
from archai.nas.model_desc import ModelDesc
from archai.nas.model_ssl import ModelSimCLRDarts
from archai.common.checkpoint import CheckPoint
from archai.nas.model_desc_builder import ModelDescBuilder

def create_model(conf_train:Config, conf_eval:Config, model_desc_builder:ModelDescBuilder, desc_filename:str)->nn.Module:

        assert model_desc_builder is not None, 'Default evaluater requires model_desc_builder'

        # region conf vars
        # if explicitly passed in then don't get from conf
        conf_model_desc   = conf_train['model_desc']
        conf_projection  = conf_train['projection']
        # endregion

        # load model desc file to get template model
        desc_filepath = os.path.join(os.path.dirname(conf_eval['common']['load_checkpoint']),os.path.basename(desc_filename))
        template_model_desc = ModelDesc.load(desc_filepath)
        model_desc = model_desc_builder.build(conf_model_desc,
                                            template=template_model_desc)

        model = ModelSimCLRDarts(model_desc, conf_projection['hidden_dim'], conf_projection['out_features'], droppath=True, affine=True)

        return model, model_desc

def train_test(conf:Config):
    conf_loader = conf['loader']
    conf_trainer = conf['trainer']
    conf_dataset = conf_loader['dataset']
    conf_checkpoint = conf['common']['checkpoint']
    train_config_path = os.path.join(os.path.dirname(conf['common']['load_checkpoint']),"config_used.yaml")
    with open(train_config_path) as f:
        config_train = yaml.load(f, Loader=yaml.Loader)
    # conf_trainer['batch_chunks'] = config_train['trainer']['batch_chunks']
    if "darts" in conf['common']['load_checkpoint']:
        conf_trainer['model'] = 'darts'
    else:
        conf_trainer['model'] = config_train['trainer']['model']

    if "resnet" in conf_trainer['model']:
        with open('confs/algos/simclr_resnets.yaml', 'r') as f:
            conf_models = yaml.load(f, Loader=yaml.Loader)
    elif "vgg" in conf_trainer['model']:
        with open('confs/algos/simclr_vggnets.yaml', 'r') as f:
            conf_models = yaml.load(f, Loader=yaml.Loader)
    elif "vit" in conf_trainer['model']:
        with open('confs/algos/simclr_vits.yaml', 'r') as f:    
            conf_models = yaml.load(f, Loader=yaml.Loader)
    elif "darts" in conf_trainer['model']:
        model_desc_builder = DartsModelDescBuilderSimClr()
        model, model_desc = create_model(config_train['nas']['eval'], conf, model_desc_builder, config_train['nas']['eval']['full_desc_filename'])
    else:
        raise Exception(f"Not implemented SimCLR for model {conf_trainer['model']}")
    if "darts" not in conf_trainer['model']:
        conf_model = conf_models[conf_trainer['model']]

    _create_sysinfo(conf)
    # create model
    if "resnet" in conf_trainer['model']:
        model = ModelSimCLRResNet(config_train['dataset']['name'], conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
            conf_model['compress'], conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
            width_per_group=conf_model['width_per_group'])
    elif "vgg" in conf_trainer['model']:
        model = ModelSimCLRVGGNet(config_train['dataset']['name'], conf_model['layers'], conf_model['batch_norm'], conf_model['hidden_dim'], 
            conf_model['out_features'], conf_model['out_features_vgg'], classifier_type = conf_model['classifier_type'], 
            init_weights = True, drop_prob=conf_model['drop_prob'], hidden_features_vgg=conf_model['hidden_features_vgg'])
    elif "vit" in conf_trainer['model']:
        model = ModelSimCLRViT(image_size = config_train['loader']['dataset']["input_height"], patch_size = conf_model["patch_size"], dim = conf_model["dim"],
                depth = conf_model["depth"], heads = conf_model["heads"], mlp_dim = conf_model["mlp_dim"], pool = conf_model["pool"],
                channels = conf_model["channels"], dim_head = conf_model["dim_head"], dropout = conf_model["dropout"],
                emb_dropout = conf_model["emb_dropout"], hidden_dim = conf_model["hidden_dim"], out_features = conf_model["out_features"])
    model = model.to(torch.device('cuda', 0))
    ckpt = torch.load(conf['common']['load_checkpoint'])
    print(ckpt['trainer']['last_epoch']+1)
    if "darts" in conf['common']['load_checkpoint']:
        if ckpt['trainer']['last_epoch'] +1 != config_train['nas']['eval']['trainer']['epochs']:
            raise Exception("Model training not finished, exiting evaluation...")
    elif ckpt['trainer']['last_epoch'] +1 != ['trainer']['epochs']:
        raise Exception("Model training not finished, exiting evaluation...")
    print("Loading model from epoch {}".format(ckpt['trainer']['last_epoch']+1))
    model_state_dict = ckpt['trainer']['model']
    keys = list(model_state_dict.keys())
    for key in keys:
        if "module" in key:
            model_state_dict[key.split("module.")[-1]] = model_state_dict[key]
            del model_state_dict[key]
    keys = list(model_state_dict.keys())
    if 'darts' in conf['common']['load_checkpoint']:
        for key in keys:
            model_state_dict["backbone."+key] = model_state_dict[key]
            if 'projection' not in key:
                del model_state_dict[key]
    model.load_state_dict(model_state_dict)
    # print('Number of trainable params: {:.2f}M'
    #       .format(sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6))
    # exit()
   
    if "resnet" in conf_trainer['model']:
        input_dim = (64 if conf_model['compress'] else 512)*(4 if conf_model['bottleneck'] else 1)
    elif "vgg" in conf_trainer['model']:
        input_dim = conf_model['out_features_vgg']
    elif "vit" in conf_trainer['model']:
        input_dim = conf_model['dim']
    elif 'darts' in conf_trainer['model']:
        input_dim = model_desc.logits_op.params['conv'].ch_out

    model.fc = nn.Linear(input_dim, conf_dataset['n_classes'])
    model = model.to(torch.device('cuda', 0))

    # get data
    data_loaders = data.get_data(conf_loader)

    # train!
    ckpt = CheckPoint(conf_checkpoint, load_existing=False)
    trainer = TrainerLinear(conf_trainer, model)
    trainer.fit(data_loaders)


if __name__ == '__main__':
    if utils.is_main_process():
        conf = common_init(config_filepath='confs/algos/simclr_eval.yaml')
        print('Running main process')
    else:
        conf = create_conf(config_filepath='confs/algos/simclr_eval.yaml')
        Config.set_inst(conf)
        update_envvars(conf)
        commonstate = get_state()
        init_from(commonstate)
        print('Running child process')
    train_test(conf)


