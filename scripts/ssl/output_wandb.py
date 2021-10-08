import pandas as pd 
import wandb
import numpy as np
import scipy
import yaml
import scipy.stats
from scipy.stats import pearsonr, spearmanr
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from archai.networks_ssl.simclr import ModelSimCLRResNet

api = wandb.Api()

# Project is specified by <entity/project-name>
resnets = []
outputs = {f"resnet_v{i}":{} for i in range(1,95)}
datasets = ['cifar10', 'cifar100', 'flower102', 'mit67', 'sport8', 'aircraft', 'svhn', 'food101']
# datasets = ['aircraft', 'svhn', 'food101']
outputs['resnet18'] = {}
outputs['resnet34'] = {}
outputs['resnet50'] = {}
params = {}
with open('confs/algos/simclr_resnets.yaml', 'r') as f:
    conf_models = yaml.load(f, Loader=yaml.Loader)

for resnet in outputs.keys():
    conf_model = conf_models[resnet]
    model = ModelSimCLRResNet('cifar10', conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
            False, conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
            width_per_group=conf_model['width_per_group'])
    params[resnet] = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6
for dataset in datasets:
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

final_accs = {}
final_accs_array = None
final_resnets = []
for resnet in outputs.keys():
    accs = list(outputs[resnet].values())
    if len(accs)!=len(datasets):
        continue
    else:
        final_resnets.append(resnet)
        final_accs[resnet] = accs
        if final_accs_array is None:
            final_accs_array = np.array([params[resnet]]+accs)[np.newaxis,:]
        else:
            final_accs_array = np.concatenate((final_accs_array,np.array([params[resnet]]+accs)[np.newaxis,:]))
print_array = final_accs_array
for i,p in enumerate(print_array):
    print(final_resnets[i],p[0],p[1],p[2],p[3],p[4],p[5], p[6], p[7], p[8])

final_accs_array = final_accs_array[:,1:]
pears = np.zeros((len(datasets),len(datasets)))
spear = np.zeros((len(datasets),len(datasets)))
for i in range(len(datasets)):
    for j in range(len(datasets)):
        pears[i,j] = pearsonr(final_accs_array[:,i],final_accs_array[:,j])[0]
        spear[i,j] = spearmanr(final_accs_array[:,i],final_accs_array[:,j])[0]

print(pears, spear)

df_cm = pd.DataFrame(pears, index = [i for i in datasets],
                  columns = [i for i in datasets])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
plt.title('Pearson correlation coefficient')
plt.savefig('pearson.png')
plt.clf()
df_cm = pd.DataFrame(spear, index = [i for i in datasets],
                  columns = [i for i in datasets])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
plt.title('Spearman correlation coefficient')
plt.savefig('spearman.png')
exit()
runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

# runs_df.to_csv("project.csv")