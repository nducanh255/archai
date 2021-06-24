import os
import yaml
from archai.networks_ssl.simclr import ModelSimCLR

def parse_arg(key, commandline):
    idx = commandline.find(key)
    args = commandline.split(' ')

    if key not in args:
        raise Exception(f'Key {key} not found in commandline string')
    else:
        value = args.index(key)
        return args[value+1]




jobs = os.listdir("../amlt/eval_linear-simclr_cifar10_dist_models")
# print(jobs)
count = 0
full_list = []
for jobname in jobs:
    model_name = "_".join(jobname.split("_")[-2:])
    path = os.path.join('../amlt/eval_linear-simclr_cifar10_dist_models', jobname,'SimClr_Cifar10','log.yaml')
    if not os.path.exists(path):
        continue
    with open(path,"r") as f:
        results = yaml.load(f, Loader=yaml.Loader)
    if "eval_train" not in results:
        continue
    if "best_test" not in results['eval_train']:
        continue
    top1_acc = round(results['eval_train']['best_test']['top1']*100,2)

    path = os.path.join('../amlt/eval_linear-simclr_cifar10_dist_models', jobname,'SimClr_Cifar10','config_used.yaml')
    with open(path,'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    # model_name = conf['trainer']['model']
    conf_model = conf[model_name]
    model = ModelSimCLR('cifar10', conf_model['depth'], conf_model['layers'], conf_model['bottleneck'],
            conf_model['compress'], conf_model['hidden_dim'], conf_model['out_features'], groups = conf_model['groups'],
            width_per_group=conf_model['width_per_group'])
    n_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6
    l1, l2, l3, l4 = conf_model['layers']
    print(count+1,model_name, round(n_params,2), l1, l2, l3, l4, top1_acc)

    count += 1

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