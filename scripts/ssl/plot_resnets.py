import os
import sys
import yaml
import json
import wandb
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from archai.networks_ssl.simclr import ModelSimCLRResNet, ModelSimCLRVGGNet, ModelSimCLRViT

api = wandb.Api()

# Project is specified by <entity/project-name>
resnets = []
datasets = ['cifar10', 'cifar100', 'flower102', 'mit67', 'sport8', 'aircraft', 'svhn', 'food101']
# datasets = ['aircraft', 'svhn', 'food101']
outputs = {f"resnet_v{i}":{} for i in range(12,95)}
outputs['resnet18'] = {}
outputs['resnet34'] = {}
outputs['resnet50'] = {}
params = {}
# with open('confs/algos/simclr_resnets.yaml', 'r') as f:
#     conf_models = yaml.load(f, Loader=yaml.Loader)

with open('/vulcanscratch/sgirish/dummy/resnet_params.json','r') as f:
    params = json.load(f)

# for resnet in outputs.keys():
#     conf_model = conf_models[resnet]
    # model = ModelSimCLRResNet('cifar10', conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
    #         False, conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
    #         width_per_group=conf_model['width_per_group'])
    # l1, l2, l3, l4 = conf_model['layers']
    # resnet_dict = {'params':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6,
    #                'bottom_params':sum(p.numel() for n,p in model.backbone.named_parameters() \
    #                                    if p.requires_grad and ('layer1' in n or 'layer2' in n))/1e6,
    #                'top_params':sum(p.numel() for n,p in model.backbone.named_parameters() \
    #                                    if p.requires_grad and ('layer3' in n or 'layer4' in n))/1e6,
    #                'l1':l1, 'l2': l2, 'l3': l3, 'l4': l4}
    # params[resnet] = resnet_dict
# with open('/vulcanscratch/sgirish/dummy/resnet_params.json','w') as f:
#     json.dump(params,f)
df = None
for dataset in datasets:
    print(f'Generating plots for {dataset}')
    runs = api.runs(f"nas_ssl/eval_linear_{dataset}-simclr_imagenet_models")

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        try:
            outputs[run.name.split('-')[1]][dataset] = run.summary.best_val_top1
        except Exception as e:
            continue
    accs = [outputs[resnet][dataset] for resnet in outputs if dataset in outputs[resnet]]
    l1 = [params[resnet]['l1'] for resnet in outputs if dataset in outputs[resnet]]
    l2 = [params[resnet]['l2'] for resnet in outputs if dataset in outputs[resnet]]
    l3 = [params[resnet]['l3'] for resnet in outputs if dataset in outputs[resnet]]
    l4 = [params[resnet]['l4'] for resnet in outputs if dataset in outputs[resnet]]
    bottom_params = [params[resnet]['bottom_params'] for resnet in outputs if dataset in outputs[resnet]]
    top_params = [params[resnet]['top_params'] for resnet in outputs if dataset in outputs[resnet]]
    ratio_params = [params[resnet]['top_params']/params[resnet]['bottom_params'] for resnet in outputs if dataset in outputs[resnet]]
    net_params = [params[resnet]['params'] for resnet in outputs if dataset in outputs[resnet]]

    plt.clf()
    if df is None:
        df = pd.DataFrame(list(zip(bottom_params, top_params, ratio_params, net_params, accs, [dataset]*len(accs))), \
                          columns=['Bottom Params', 'Top Params', 'Top Params / Bottom Params', 'Net Params', f'Accuracy', 'Dataset'])
    else:
        df = df.append(pd.DataFrame(list(zip(bottom_params, top_params, ratio_params, net_params, accs, [dataset]*len(accs))), \
                          columns=['Bottom Params', 'Top Params', 'Top Params / Bottom Params', 'Net Params', f'Accuracy', 'Dataset']))
    # plt.scatter(bottom_params,accs)
    # sns.scatterplot(data=df, x= f'Bottom params', y='Accuracy')
    # plt.show()
    # plt.xlabel('No. of params in bottom layers (in M)')
    # plt.ylabel('Accuracy')
    # plt.title(f'No. of bottom params vs {dataset} accuracy')
    # plt.grid(True)
    # plt.savefig(f'/vulcanscratch/sgirish/results_resnets_imagenet/{dataset}_acc_bottom_line.png')
plt.clf()
plt.grid()
grid = sns.FacetGrid(df, col = "Dataset", col_wrap=4)
grid.map(sns.scatterplot, "Bottom Params", "Accuracy")
grid.add_legend()
plt.grid()
plt.show()
plt.savefig('/vulcanscratch/sgirish/results_resnets_imagenet/bottom_params.png')

plt.clf()
plt.grid()
grid = sns.FacetGrid(df, col = "Dataset", col_wrap=4)
grid.map(sns.scatterplot, "Top Params", "Accuracy")
grid.add_legend()
plt.grid()
plt.show()
plt.savefig('/vulcanscratch/sgirish/results_resnets_imagenet/top_params.png')

plt.clf()
plt.grid()
grid = sns.FacetGrid(df, col = "Dataset", col_wrap=4)
grid.map(sns.scatterplot, "Net Params", "Accuracy")
grid.add_legend()
plt.show()
plt.savefig('/vulcanscratch/sgirish/results_resnets_imagenet/net_params.png')

plt.clf()
plt.grid()
grid = sns.FacetGrid(df, col = "Dataset", col_wrap=4)
grid.map(sns.scatterplot, "Top Params / Bottom Params", "Accuracy")
grid.add_legend()
plt.grid()
plt.show()
plt.savefig('/vulcanscratch/sgirish/results_resnets_imagenet/top_bottom_params.png')
