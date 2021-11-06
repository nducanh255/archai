import os
import csv
import sys
import yaml
import json
import wandb
import argparse
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from archai.networks_ssl.simclr import ModelSimCLRResNet, ModelSimCLRVGGNet, ModelSimCLRViT, ModelSimCLRMobileNet


def param_plots(params, models, dataset_name, dataset, key):
    del_nets =  list(set(list(models.keys()))-set(list(params.keys())))
    for del_net in del_nets:
        del models[del_net]

    # l1 = [params[net]['l1'] for net in models]
    # l2 = [params[net]['l2'] for net in models]
    # l3 = [params[net]['l3'] for net in models]
    # l4 = [params[net]['l4'] for net in models]
    if key == 'mobilenet':
        nets = list(models.keys())
        for net in nets:
            if int(net.split('mobilenet_v')[-1])<12 and int(net.split('mobilenet_v')[-1])>1:
                del models[net]
    bottom_params = [params[net]['bottom_params'] for net in models]
    top_params = [params[net]['top_params'] for net in models]
    vals, count = np.unique(np.array(bottom_params),return_counts=True)
    bottom_dict = {k:v for k,v in zip(vals,count) if v>1}
    ratio_params = [params[net]['bottom_params']/params[net]['top_params'] for net in models]
    net_params = [params[net]['params'] for net in models]
    new_accs = [models[net] for net in models]

    df = pd.DataFrame(list(zip(bottom_params, top_params, ratio_params, net_params, new_accs)), \
                            columns=['Bottom Params', 'Top Params', 'Bottom Params / Top Params', 'Net Params', f'{dataset_name} Accuracy'])
    plt.clf()
    g = sns.lmplot('Net Params', f'{dataset_name} Accuracy', df, order=2, truncate=False)
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/params_{dataset}_{key}.png')
    plt.clf()
    g = sns.lmplot('Bottom Params / Top Params', f'{dataset_name} Accuracy', df, order=2, truncate=False)
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/top_bottom_params_{dataset}_{key}.png')
    plt.clf()
    if dataset == 'imagenet':
        cur_top_params = [params[net]['top_params'] for net in models if params[net]['bottom_params'] in bottom_dict]
        cur_bottom_params = [round(params[net]['bottom_params'],2) for net in models if params[net]['bottom_params'] in bottom_dict]
        cur_accs = [models[net] for net in models if params[net]['bottom_params'] in bottom_dict]
        cur_df = pd.DataFrame(list(zip(cur_top_params, cur_bottom_params, cur_accs)), \
                                columns=['Top Params', 'Bottom Params', f'{dataset_name} Accuracy'])
        g = sns.lmplot('Top Params', f'{dataset_name} Accuracy', cur_df, order=2, truncate=False, hue="Bottom Params")
        # plt.legend(labels=legend)
        # handles = g._legend_data.values()
        # labels = g._legend_data.keys()
        # g.fig.legend(handles=handles, labels=labels, loc='upper left')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='lower center', borderaxespad=0)
        plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/top_params_legend_{dataset}_{key}.png')

    plt.clf()
    g = sns.lmplot('Bottom Params', f'{dataset_name} Accuracy', df, order=2, truncate=False)
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/bottom_params_{dataset}_{key}.png')
    plt.clf()
    g = sns.lmplot('Top Params', f'{dataset_name} Accuracy', df, order=2, truncate=False)
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/top_params_{dataset}_{key}.png')
# key = 'resnet'
key = 'resnet'
with open(f'/vulcanscratch/sgirish/dummy/resnet_params.json','r') as f:
    resnet_params = json.load(f)
with open(f'/vulcanscratch/sgirish/dummy/mobilenet_params.json','r') as f:
    mobilenet_params = json.load(f)
params = resnet_params if key == 'resnet' else mobilenet_params
with open('/vulcanscratch/sgirish/eval_models.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    acc, dataset, model = [[] for _ in range(3)]
    imagenet_resnet_models = {}
    imagenet_mobilenet_models = {}
    for i,row in enumerate(csv_reader):
        name = row['Name']
        # if 'mobilenet' in name.strip().split('-')[-1] and int(name.strip().split('-')[-1].split('mobilenet_v')[-1])>1 and \
        #     int(name.strip().split('-')[-1].split('mobilenet_v')[-1])<12:
        #     print(name)
        acc.append(float(row['epoch_top1_test']))
        model.append(name.strip().split('-')[-1])
        dataset.append(name.strip().split('-')[0].split('_')[-1])
        if name.strip().split('-')[0].split('_')[-1] == 'imagenet':
            if 'resnet' in name.strip().split('-')[-1] and float(row['epoch_top1_test'])>54.0 and float(resnet_params[name.strip().split('-')[-1]]['params']<30):
                imagenet_resnet_models[name.strip().split('-')[-1]] = float(row['epoch_top1_test'])
            elif 'mobilenet' in name.strip().split('-')[-1] and float(mobilenet_params[name.strip().split('-')[-1]]['params']<3):
                imagenet_mobilenet_models[name.strip().split('-')[-1]] = float(row['epoch_top1_test'])
imagenet_models = imagenet_resnet_models if key == 'resnet' else imagenet_mobilenet_models

###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!REMEMBER TO CHANGE c10r20 to c100r20 ON VULCAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# params = {}
# if key == 'resnet':
#     with open('confs/algos/simclr_resnets.yaml', 'r') as f:
#         conf_models = yaml.load(f, Loader=yaml.Loader)
#     all_models = ['resnet18','resnet34','resnet50']+[f'resnet_v{i}' for i in range(1,92)]
# elif key == 'mobilenet':
#     with open('confs/algos/simclr_mobilenets.yaml', 'r') as f:
#         conf_models = yaml.load(f, Loader=yaml.Loader)
#     all_models = [f'mobilenet_v{i}' for i in range(1,43)]

# for net in all_models:
#     conf_model = conf_models[net]
#     if key == 'resnet':
#         model = ModelSimCLRResNet('cifar10', conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
#                 False, conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
#                 width_per_group=conf_model['width_per_group'])
#         l1, l2, l3, l4 = conf_model['layers']
#         net_dict = {'params':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6,
#                     'top_params':sum(p.numel() for n,p in model.backbone.named_parameters() \
#                                         if p.requires_grad and ('layer1' in n or 'layer2' in n))/1e6,
#                     'bottom_params':sum(p.numel() for n,p in model.backbone.named_parameters() \
#                                         if p.requires_grad and ('layer3' in n or 'layer4' in n))/1e6,
#                     'l1':l1, 'l2': l2, 'l3': l3, 'l4': l4}
#         params[net] = net_dict
#     else:
#         inverted_residual_setting = [[conf_model['t'][i],conf_model['c'][i],conf_model['n'][i],conf_model['s'][i]] \
#                                         for i in range(len(conf_model['t']))]
#         model = ModelSimCLRMobileNet(hidden_dim = conf_model["hidden_dim"], out_features = conf_model["out_features"],
#                 inverted_residual_setting = inverted_residual_setting, width_mult = conf_model['width_mult'], compress = False
#                 )
#         top_layers = [f'features.{i}.' for i in range(0,sum(conf_model['n'][:4]))]
#         net_dict = {'params':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6,
#                     'top_params':sum(p.numel() for n,p in model.backbone.named_parameters() \
#                                         if p.requires_grad and any([t in n for t in top_layers]))/1e6,
#                     'bottom_params':sum(p.numel() for n,p in model.backbone.named_parameters() \
#                                         if p.requires_grad and not any([t in n for t in top_layers]))/1e6,
#                     }
#         params[net] = net_dict

# if key == 'resnet':
#     params['resnet18_blur'] = params['resnet18']
#     params['resnet34_blur'] = params['resnet34']
#     params['resnet50_blur'] = params['resnet50']
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
accs = accs[keep_model_idx]

model_plot = [model[i] for i in range(len(acc)) if dataset[i]!='imagenet' and model[i] in imagenet_models_subset]
dataset_plot = [dataset[i] for i in range(len(acc)) if dataset[i]!='imagenet' and model[i] in imagenet_models_subset]
acc_plot_y = [acc[i] for i in range(len(acc)) if dataset[i]!='imagenet' and model[i] in imagenet_models_subset]
acc_plot_x = [imagenet_models[model[i]] for i in range(len(acc)) if dataset[i]!='imagenet' and model[i] in imagenet_models_subset]

sns.set(font_scale=1.6)
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
               data=df, col_wrap=5, sharey=False, sharex=True, truncate=False)

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
    
cur_dataset = ['imagenet']+[n for n in dataset_names if n!='cifar10' and n!='imagenet']
cur_dataset_names = {'aircraft':'Aircraft', 'flower102':'Flowers102', 'cars196':'Stanford Cars', 'imagenet':'ImageNet',
                     'cifar10':'Cifar10', 'cifar100':'Cifar100', 'cub200':'Caltech Birds', 'dogs120':'Stanford Dogs',
                     'mit67':'MIT67', 'sport8':'Sports', 'svhn':'SVHN'}
cur_dataset_names = [cur_dataset_names[k] for k in cur_dataset]
if key == 'resnet':
    min_vals = {'aircraft':38.0, 'flower102':91.5, 'cars196':32.0, 'imagenet':54.0, 'cifar10':82.0, 'cifar100':59.0,
                'cub200':28.0, 'dogs120':46.0, 'mit67':65.0, 'svhn':66.0, 'sport8':94.5}
else:
    min_vals = {'aircraft':0.0, 'flower102':0.0, 'cars196':0.0, 'imagenet':40.5, 'cifar10':0.0, 'cifar100':0.0,
                'cub200':0.0, 'dogs120':0.0, 'mit67':0.0, 'svhn':0.0, 'sport8':0.0}
max_params = 30 if key == 'resnet' else 3
acc_y = [acc[i] for i in range(len(acc)) if dataset[i] in cur_dataset and key in model[i] and \
            params[model[i]]['params']<max_params and acc[i]>min_vals[dataset[i]]]
params_x = [params[model[i]]['params'] for i in range(len(acc)) if dataset[i] in cur_dataset and key in model[i] and \
            params[model[i]]['params']<max_params and acc[i]>min_vals[dataset[i]]]
dataset_plot = [dataset[i] for i in range(len(acc)) if dataset[i] in cur_dataset and key in model[i] and \
                params[model[i]]['params']<max_params and acc[i]>min_vals[dataset[i]]]
df = pd.DataFrame(list(zip(params_x, acc_y, dataset_plot)), columns=['Params (in Million)', 'Accuracy','Dataset'])
plt.clf()
plt.grid()
g = sns.lmplot(x="Params (in Million)", y="Accuracy", col="Dataset", col_order=cur_dataset,
            data=df, col_wrap=5, sharey=False, sharex=True, truncate=False, order=2)
for j in range(len(cur_dataset)):
    cur_accs = [acc[i] for i in range(len(acc)) if dataset[i] == cur_dataset[j] and key in model[i] and \
                params[model[i]]['params']<max_params and acc[i]>min_vals[dataset[i]]]
    g.axes[j].set_xlim((min(params_x)*0.998,max(params_x)*1.002))
    g.axes[j].set_ylim((min(cur_accs)*0.998,max(cur_accs)*1.002))
plt.show()
plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/downstream_params_{key}.png')

# param_plots(params, imagenet_models, 'ImageNet', 'imagenet', key)
for plot_dataset, plot_dataset_name, min_acc in zip(cur_dataset,cur_dataset_names,[min_vals[d] for d in cur_dataset]):
    cur_accs = {}
    cur_accs = {model[i]:acc[i] for i in range(len(acc)) if dataset[i]==plot_dataset and key in model[i] \
                                                            and params[model[i]]['params']<max_params and acc[i]>min_acc}
    param_plots(params, cur_accs, plot_dataset_name, plot_dataset, key)


# in_index = dataset_names.index('imagenet')
# accs = np.concatenate((accs[:,in_index:in_index+1],accs[:,:in_index],accs[:,in_index+1:]),axis=1)
# new_dataset_names = ['imagenet']+[n for n in dataset_names if n!='imagenet']
# pears = np.zeros((len(new_dataset_names),len(new_dataset_names)))
# spear = np.zeros((len(new_dataset_names),len(new_dataset_names)))
# for i in range(len(new_dataset_names)):
#     for j in range(len(new_dataset_names)):
#         pears[i,j] = pearsonr(accs[:,i],accs[:,j])[0]
#         spear[i,j] = spearmanr(accs[:,i],accs[:,j])[0]

# sns.set(font_scale=0.9)
# df_cm = pd.DataFrame(pears.round(2), index = [i for i in new_dataset_names],
#                   columns = [i for i in new_dataset_names])
# plt.figure(figsize = (9,7))
# sns.heatmap(df_cm, annot=True)
# plt.show()
# plt.title('Pearson correlation coefficient')
# plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/pearson_correlation_{key}.png')
# plt.clf()
# df_cm = pd.DataFrame(spear.round(2), index = [i for i in new_dataset_names],
#                   columns = [i for i in new_dataset_names])
# plt.figure(figsize = (9,7))
# sns.heatmap(df_cm, annot=True)
# plt.show()
# plt.title('Spearman correlation coefficient')
# plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/spearman_correlation_{key}.png')
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
