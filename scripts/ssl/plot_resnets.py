import os
import csv
import sys
import yaml
import json
import wandb
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from archai.networks_ssl.simclr import ModelSimCLRResNet, ModelSimCLRVGGNet, ModelSimCLRViT, ModelSimCLRMobileNet



# key = 'resnet'
key = 'mobilenet'
with open(f'/vulcanscratch/sgirish/dummy/resnet_params.json','r') as f:
    resnet_params = json.load(f)
with open(f'/vulcanscratch/sgirish/dummy/mobilenet_params.json','r') as f:
    mobilenet_params = json.load(f)
with open('/vulcanscratch/sgirish/eval_models.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    acc, dataset, model = [[] for _ in range(3)]
    imagenet_resnet_models = {}
    imagenet_mobilenet_models = {}
    for i,row in enumerate(csv_reader):
        acc.append(float(row['epoch_top1_test']))
        name = row['Name']
        model.append(name.strip().split('-')[-1])
        dataset.append(name.strip().split('-')[0].split('_')[-1])
        if name.strip().split('-')[0].split('_')[-1] == 'imagenet':
            if 'resnet' in name.strip().split('-')[-1] and float(row['epoch_top1_test'])>54.0 and float(resnet_params[name.strip().split('-')[-1]]['params']<30):
                imagenet_resnet_models[name.strip().split('-')[-1]] = float(row['epoch_top1_test'])
            elif 'mobilenet' in name.strip().split('-')[-1]and float(mobilenet_params[name.strip().split('-')[-1]]['params']<3):
                imagenet_mobilenet_models[name.strip().split('-')[-1]] = float(row['epoch_top1_test'])

imagenet_models = imagenet_resnet_models if key == 'resnet' else imagenet_mobilenet_models
params = resnet_params if key == 'resnet' else mobilenet_params
# params = {}
# if key == 'resnet':
#     with open('confs/algos/simclr_resnets.yaml', 'r') as f:
#         conf_models = yaml.load(f, Loader=yaml.Loader)
# elif key == 'mobilenet':
#     with open('confs/algos/simclr_mobilenets.yaml', 'r') as f:
#         conf_models = yaml.load(f, Loader=yaml.Loader)
# for net in imagenet_models:
#     conf_model = conf_models[net]
#     if key == 'resnet':
#         model = ModelSimCLRResNet('cifar10', conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
#                 False, conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
#                 width_per_group=conf_model['width_per_group'])
#         l1, l2, l3, l4 = conf_model['layers']
#         net_dict = {'params':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6,
#                     'bottom_params':sum(p.numel() for n,p in model.backbone.named_parameters() \
#                                         if p.requires_grad and ('layer1' in n or 'layer2' in n))/1e6,
#                     'top_params':sum(p.numel() for n,p in model.backbone.named_parameters() \
#                                         if p.requires_grad and ('layer3' in n or 'layer4' in n))/1e6,
#                     'l1':l1, 'l2': l2, 'l3': l3, 'l4': l4}
#         params[net] = net_dict
#     else:
#         inverted_residual_setting = [[conf_model['t'][i],conf_model['c'][i],conf_model['n'][i],conf_model['s'][i]] \
#                                         for i in range(len(conf_model['t']))]
#         model = ModelSimCLRMobileNet(hidden_dim = conf_model["hidden_dim"], out_features = conf_model["out_features"],
#                 inverted_residual_setting = inverted_residual_setting, width_mult = conf_model['width_mult'], compress = False
#                 )
#         params[net] = {'params':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6}

    
# with open(f'/vulcanscratch/sgirish/dummy/{key}_params.json','w') as f:
#     json.dump(params,f)

# exit()

dataset_names = list(np.unique(np.array(dataset)))
model_names = list(np.unique(np.array(list(imagenet_models.keys()))))

nets = model_names
accs = np.zeros((len(nets),len(dataset_names)))
for i in range(len(acc)):
    model_name, dataset_name = model[i], dataset[i]
    if model[i] in nets:
        accs[nets.index(model_name),dataset_names.index(dataset_name)] = acc[i]

keep_model_idx = np.arange(accs.shape[0])[np.all(accs!=0.0,axis=1)]
imagenet_models_subset = {k:v for k,v in imagenet_models.items() if model_names.index(k) in keep_model_idx}

model_plot = [model[i] for i in range(len(acc)) if dataset[i]!='imagenet' and model[i] in imagenet_models_subset]
dataset_plot = [dataset[i] for i in range(len(acc)) if dataset[i]!='imagenet' and model[i] in imagenet_models_subset]
acc_plot_y = [acc[i] for i in range(len(acc)) if dataset[i]!='imagenet' and model[i] in imagenet_models_subset]
acc_plot_x = [imagenet_models[model[i]] for i in range(len(acc)) if dataset[i]!='imagenet' and model[i] in imagenet_models_subset]


df = pd.DataFrame(list(zip(acc_plot_x, acc_plot_y, dataset_plot)), \
                    columns=['ImageNet Accuracy', 'Accuracy','Dataset'])
plt.clf()
plt.grid()
num_cols = 5
sns.set_style("darkgrid")
# grid = sns.FacetGrid(df, col = "Dataset", col_wrap=5, sharey=False, sharex=True, col_order=[n for n in dataset_names if n!='imagenet'])
# grid.map(sns.lmplot, "ImageNet Accuracy", "Accuracy")
# grid.add_legend()
g = sns.lmplot(x="ImageNet Accuracy", y="Accuracy", col="Dataset", col_order=[n for n in dataset_names if n!='imagenet'],
               data=df, col_wrap=5, sharey=False, sharex=True)

in_min = min(list(imagenet_models_subset.values()))
in_max = max(list(imagenet_models_subset.values()))
for i in range(10):
    dataset_name = [n for n in dataset_names if n!='imagenet'][i]
    # print(dataset_name, cur_accs.min(),cur_accs.max())
    cur_accs = [a for i,a in enumerate(acc) if dataset[i]==dataset_name and model[i] in imagenet_models_subset]
    g.axes[i].set_ylim((min(cur_accs)*0.998,max(cur_accs)*1.002))
    g.axes[i].set_xlim((in_min,in_max))
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/downstream_accs_{key}.png')
    
if key == 'resnet':
    l1 = [params[net]['l1'] for net in imagenet_models]
    l2 = [params[net]['l2'] for net in imagenet_models]
    l3 = [params[net]['l3'] for net in imagenet_models]
    l4 = [params[net]['l4'] for net in imagenet_models]
    bottom_params = [params[net]['bottom_params'] for net in imagenet_models]
    top_params = [params[net]['top_params'] for net in imagenet_models]
    ratio_params = [params[net]['top_params']/params[net]['bottom_params'] for net in imagenet_models]
    net_params = [params[net]['params'] for net in imagenet_models]
    accs = [imagenet_models[net] for net in imagenet_models]

    df = pd.DataFrame(list(zip(bottom_params, top_params, ratio_params, net_params, accs)), \
                            columns=['Bottom Params', 'Top Params', 'Top Params / Bottom Params', 'Net Params', f'ImageNet Accuracy'])
    plt.clf()
    g = sns.lmplot('Net Params', 'ImageNet Accuracy', df)
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/params_imagenet_{key}.png')
elif key == 'mobilenet':
    net_params = [params[net]['params'] for net in imagenet_models]
    accs = [imagenet_models[net] for net in imagenet_models]
    df = pd.DataFrame(list(zip(net_params, accs)), \
                            columns=['Net Params', f'ImageNet Accuracy'])
    plt.clf()
    g = sns.lmplot('Net Params', 'ImageNet Accuracy', df)
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/params_imagenet_{key}.png')
exit()





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
