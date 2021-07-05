#test
import os
import sys
import yaml
import numpy as np
from archai.networks_ssl.simclr import ModelSimCLRResNet, ModelSimCLRVGGNet

def parse_arg(key, commandline):
    idx = commandline.find(key)
    args = commandline.split(' ')

    if key not in args:
        raise Exception(f'Key {key} not found in commandline string')
    else:
        value = args.index(key)
        return args[value+1]

# jobs = os.listdir("../amlt/eval_linear-simclr_cifar10_dist_models")
# # print(jobs)
# count = 0
# full_list = []
# for jobname in jobs:
#     model_name = "_".join(jobname.split("_")[-2:])
#     path = os.path.join('../amlt/eval_linear-simclr_cifar10_dist_models', jobname,'SimClr_Cifar10','log.yaml')
#     if not os.path.exists(path):
#         continue
#     with open(path,"r") as f:
#         results = yaml.load(f, Loader=yaml.Loader)
#     if "eval_train" not in results:
#         continue
#     if "best_test" not in results['eval_train']:
#         continue
#     top1_acc = round(results['eval_train']['best_test']['top1']*100,2)

#     path = os.path.join('../amlt/eval_linear-simclr_cifar10_dist_models', jobname,'SimClr_Cifar10','config_used.yaml')
#     with open(path,'r') as f:
#         conf = yaml.load(f, Loader=yaml.Loader)
#     # model_name = conf['trainer']['model']
#     conf_model = conf[model_name]
#     model = ModelSimCLR('cifar10', conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
#             conf_model['compress'], conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
#             width_per_group=conf_model['width_per_group'])
#     n_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6
#     l1, l2, l3, l4 = conf_model['layers']
#     print(count+1,model_name, round(n_params,2), l1, l2, l3, l4, top1_acc)

#     count += 1

# for l in full_list:
#     print(l[0],l[1],l[2],l[3],l[4],l[5],l[6])
# path = '../amlt/eval_linear-simclr_cifar10_js/eval_linear-simclr_cifar10_js_0.5/SimClr_Cifar10/config_used.yaml'
# with open(path,'r') as f:
#     results = yaml.load(f, Loader=yaml.Loader)
# print(results['trainer']['model'])
# exit()
# jobs = os.listdir("../amlt/simclr_cifar_eval_amlt_bs1024_vr0.1-simclr_cifar10_models")
# count = 348
# for jobname in jobs:
#     path = os.path.join('../amlt/simclr_cifar_eval_amlt_bs1024_vr0.1-simclr_cifar10_models', jobname,'SimClr_Cifar10','log.yaml')
#     if not os.path.exists(path):
#         continue
#     with open(path,"r") as f:
#         results = yaml.load(f, Loader=yaml.Loader)
#     print(count+1,results['data']['train_batch_size'], 
#         results['data']['val_ratio'],
#         results['eval_train']['conf_optim']['lr'],
#         parse_arg('--loader.dataset.jitter_strength',results['command_line']),
#         results['eval_train']['total_epochs'],
#         results['eval_train']['conf_optim']['type'],
#         results['eval_train']['conf_sched']['min_lr'],
#         round(results['eval_train']['best_test']['top1']*100,2))
#     count += 1

# root = "../amlt/amlt/eval_linear-simclr_cifar10_vgg_classifier"
# jobs = os.listdir(root)
# # print(jobs)
# count = 0
# full_list = []
# model_name = 'vgg'
# if "resnet" in model_name:
#     with open('confs/algos/simclr_resnets.yaml', 'r') as f:
#         conf_models = yaml.load(f, Loader=yaml.Loader)
# elif "vgg" in model_name:
#     with open('confs/algos/simclr_vggnets.yaml', 'r') as f:
#         conf_models = yaml.load(f, Loader=yaml.Loader)

root = "../amlt/amlt/eval_linear-simclr_cifar10_vgg_layers"
root_train = "../amlt/amlt/simclr_cifar10_vgg_layers"
jobs = os.listdir(root)
jobs_train = os.listdir(root_train)
count = 0
full_list = []
model_name = 'vgg'
if "resnet" in model_name:
    with open('confs/algos/simclr_resnets.yaml', 'r') as f:
        conf_models = yaml.load(f, Loader=yaml.Loader)
elif "vgg" in model_name:
    with open('confs/algos/simclr_vggnets.yaml', 'r') as f:
        conf_models = yaml.load(f, Loader=yaml.Loader)
classifier_type_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
features_dict = {512:1, 4096:2}
drop_prob_dict = {0.0:1, 0.25:2, 0.5:3}
decay = {0.0:0, 1.0e-3:1, 1.0e-4:2, 1.0e-5:3}
order = []
# for i,(jobname, jobname_train) in enumerate(zip(jobs,jobs_train)):
for i,jobname_train in enumerate(jobs_train):
    sys.stdout.write(f'\r{i}/{len(jobs_train)}')
    jobname = 'eval_linear-'+jobname_train
    path = os.path.join(root, jobname,'SimClr_Cifar10','log.yaml')
    if not os.path.exists(path):
        continue
    with open(path,"r") as f:
        results = yaml.load(f, Loader=yaml.Loader)
    if "eval_train" not in results:
        continue
    if "best_test" not in results['eval_train']:
        continue
    top1_acc = round(results['eval_train']['best_test']['top1']*100,2)
    top1_acc_train = round(results['eval_train']['best_train']['top1']*100,2)

    path = os.path.join(root_train, jobname_train,'SimClr_Cifar10','log.log')
    if not os.path.exists(path):
        continue
    with open(path,"r") as f:
        lines = f.readlines()
    for line in lines:
        if "eval_train/best_train" in line and ", loss=" in line:
            loss_train = line.strip().split("=")[-1]

    path = os.path.join(root, jobname,'SimClr_Cifar10','config_used.yaml')
    with open(path,'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    model_name = conf['trainer']['model']

    conf_model = conf_models[model_name]
    if "resnet" in model_name:
        model = ModelSimCLRResNet(conf['dataset']['name'], conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
            conf_model['compress'], conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
            width_per_group=conf_model['width_per_group'])
        l1, l2, l3, l4 = conf_model['layers']
    elif "vgg" in model_name:
        model = ModelSimCLRVGGNet(conf['dataset']['name'], conf_model['layers'], conf_model['batch_norm'], conf_model['hidden_dim'], 
            conf_model['out_features'], conf_model['out_features_vgg'], classifier_type = conf_model['classifier_type'], 
            init_weights = True, drop_prob=conf_model['drop_prob'], hidden_features_vgg=conf_model['hidden_features_vgg'])
        l1, l2, l3, l4, l5 = conf_model['layers']
    n_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6
    # print(count+1,model_name, round(n_params,2), l1, l2, l3, l4, top1_acc)
    # decay_val = float(jobname.split("_")[-1])#conf['trainer']['optimizer']['decay']
    # full_list.append([conf_models[model_name]['classifier_type'], conf_models[model_name]['hidden_features_vgg'], \
    #                   conf_models[model_name]['out_features_vgg'], conf_models[model_name]['drop_prob'], decay_val, n_params, top1_acc, top1_acc_train])

    full_list.append([model_name, n_params, l1, l2, l3, l4, l5, top1_acc, top1_acc_train, loss_train])
    order.append(l1*(10**4)+l2*(10**3)+l3*(10**2)+l4*(10**1)+l5)

    # full_list.append([model_name, n_params, l1, l2, l3, l4, top1_acc, top1_acc_train, loss_train])
    # order.append(l1*(10**4)+l2*(10**3)+l3*(10**2)+l4*(10**1))

    # order.append(classifier_type_dict[conf_models[model_name]['classifier_type']]*(10**4)+\
    #               features_dict[conf_models[model_name]['hidden_features_vgg']]*(10**3)+\
    #               features_dict[conf_models[model_name]['out_features_vgg']]*(10**2)+\
    #               drop_prob_dict[conf_models[model_name]['drop_prob']]*(10**1)+\
    #               decay[decay_val]*1)
    # print(top1_acc)

    count += 1
sys.stdout.write('\r          ')
sys.stdout.write('\r')
idx = np.argsort(np.array(order))
for i,id in enumerate(idx):
    print(i,full_list[id][0],full_list[id][1],full_list[id][2],full_list[id][3],full_list[id][4],full_list[id][5],full_list[id][6],full_list[id][7],full_list[id][8],full_list[id][9])



# root = "../amlt/amlt/simclr_cifar10_vgg_classifier"
# jobs = os.listdir(root)
# # print(jobs)
# count = 0
# full_list = []
# model_name = 'vgg'
# if "resnet" in model_name:
#     with open('confs/algos/simclr_resnets.yaml', 'r') as f:
#         conf_models = yaml.load(f, Loader=yaml.Loader)
# elif "vgg" in model_name:
#     with open('confs/algos/simclr_vggnets.yaml', 'r') as f:
#         conf_models = yaml.load(f, Loader=yaml.Loader)

# full_list = []
# classifier_type_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
# features_dict = {512:1, 4096:2}
# drop_prob_dict = {0.0:1, 0.25:2, 0.5:3}
# decay = {0.0:0, 1.0e-3:1, 1.0e-4:2, 1.0e-5:3}
# order = []
# for i,jobname in enumerate(jobs):
#     sys.stdout.write(f'\r{i}/{len(jobs)}')
#     path = os.path.join(root, jobname,'SimClr_Cifar10','log.log')
#     if not os.path.exists(path):
#         continue
#     with open(path,"r") as f:
#         lines = f.readlines()
#     for line in lines:
#         if "eval_train/best_train" in line and ", loss=" in line:
#             loss_train = line.strip().split("=")[-1]

#     path = os.path.join(root, jobname,'SimClr_Cifar10','config_used.yaml')
#     with open(path,'r') as f:
#         conf = yaml.load(f, Loader=yaml.Loader)

#     model_name = conf['trainer']['model']

#     conf_model = conf_models[model_name]
#     if "resnet" in model_name:
#         model = ModelSimCLRResNet(conf['dataset']['name'], conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
#             conf_model['compress'], conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
#             width_per_group=conf_model['width_per_group'])
#     elif "vgg" in model_name:
#         model = ModelSimCLRVGGNet(conf['dataset']['name'], conf_model['layers'], conf_model['batch_norm'], conf_model['hidden_dim'], 
#             conf_model['out_features'], conf_model['out_features_vgg'], classifier_type = conf_model['classifier_type'], 
#             init_weights = True, drop_prob=conf_model['drop_prob'], hidden_features_vgg=conf_model['hidden_features_vgg'])
#     n_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6
#     # l1, l2, l3, l4 = conf_model['layers']
#     # print(count+1,model_name, round(n_params,2), l1, l2, l3, l4, top1_acc)
#     decay_val = float(jobname.split("_")[-1])#conf['trainer']['optimizer']['decay']
#     full_list.append([conf_models[model_name]['classifier_type'], conf_models[model_name]['hidden_features_vgg'], \
#                       conf_models[model_name]['out_features_vgg'], conf_models[model_name]['drop_prob'], decay_val, n_params, loss_train])

#     order.append(classifier_type_dict[conf_models[model_name]['classifier_type']]*(10**4)+\
#                   features_dict[conf_models[model_name]['hidden_features_vgg']]*(10**3)+\
#                   features_dict[conf_models[model_name]['out_features_vgg']]*(10**2)+\
#                   drop_prob_dict[conf_models[model_name]['drop_prob']]*(10**1)+\
#                   decay[decay_val]*1)
#     # print(top1_acc)

#     count += 1
# sys.stdout.write('\r          ')
# sys.stdout.write('\r')
# idx = np.argsort(np.array(order))
# for i,id in enumerate(idx):
#     print(i,full_list[id][0],full_list[id][1],full_list[id][2],full_list[id][3],full_list[id][4],full_list[id][5],full_list[id][6])
#     # print(full_list[id][5],full_list[id][6])