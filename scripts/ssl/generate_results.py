#test
import os
import sys
import yaml
import argparse
import numpy as np
from archai.networks_ssl.simclr import ModelSimCLRResNet, ModelSimCLRVGGNet, ModelSimCLRViT

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir',
                    dest='output_dir',
                    help='Provide destination host. Defaults to localhost',
                    type=str
                    )
args = parser.parse_args()

model_name = 'vit'
if "resnet" in model_name:
    with open('confs/algos/simclr_resnets.yaml', 'r') as f:
        conf_models = yaml.load(f, Loader=yaml.Loader)
elif "vgg" in model_name:
    with open('confs/algos/simclr_vggnets.yaml', 'r') as f:
        conf_models = yaml.load(f, Loader=yaml.Loader)
elif "vit" in model_name:
    with open('confs/algos/simclr_vits.yaml', 'r') as f:
        conf_models = yaml.load(f, Loader=yaml.Loader)

log_yaml = os.path.join(args.output_dir,'SimClr_Cifar10','log.yaml')
if not os.path.exists(log_yaml):
    raise Exception('log yaml not found')

with open(log_yaml,"r") as f:
    results = yaml.load(f, Loader=yaml.Loader)

if "eval_train" not in results:
    raise Exception('log yaml corrupt')
if "best_test" not in results['eval_train']:
    raise Exception('log yaml corrupt')
top1_acc = round(results['eval_train']['best_test']['top1']*100,2)
top1_acc_train = round(results['eval_train']['best_train']['top1']*100,2)

log_log = os.path.join(args.output_dir, 'SimClr_Cifar10','log.log')
# if not os.path.exists(log_log):
#     raise Exception('log log not found')
# with open(log_log,"r") as f:
#     lines = f.readlines()
# loss_train = -1
# for line in lines:
#     if "eval_train/best_train" in line and ", loss=" in line:
#         loss_train = line.strip().split("=")[-1]
# if loss_train==-1:
#     raise Exception('log log corrupt')


path = os.path.join(args.output_dir,'SimClr_Cifar10','config_used.yaml')
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
elif "vit" in model_name:
    model = ModelSimCLRViT(image_size = conf['loader']['dataset']["input_height"], patch_size = conf_model["patch_size"], dim = conf_model["dim"],
                depth = conf_model["depth"], heads = conf_model["heads"], mlp_dim = conf_model["mlp_dim"], pool = conf_model["pool"],
                channels = conf_model["channels"], dim_head = conf_model["dim_head"], dropout = conf_model["dropout"],
                emb_dropout = conf_model["emb_dropout"], hidden_dim = conf_model["hidden_dim"], out_features = conf_model["out_features"])
n_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1e6
# ps{patch_size}_dim{dim}_depth{depth}_heads{heads}_mlpdim{mlp_dim}_dimhead{dim_head}_dropout{dropout}_{pool}
weight = conf_model["dim"]*(10**4)+conf_model["depth"]*(10**3)+conf_model["heads"]*(10**2)+conf_model["mlp_dim"]*(10**1)+conf_model["dim_head"]
out = np.array([n_params, conf_model["dim"], conf_model["depth"], conf_model["heads"], conf_model["mlp_dim"], conf_model["dim_head"], top1_acc, top1_acc_train, weight])

print(out)
path = os.path.abspath(
        os.path.expanduser(
            os.path.expandvars(os.path.join(os.environ['AMLT_OUTPUT_DIR'],"output.npy"))))
np.save(path,out)

# full_list = []
# count = 0
# for jobname in os.listdir(args.output_dir):
#     filepath = os.path.join(args.output_dir, jobname,"output.npy")
#     if os.path.exists(filepath):
#         n_params, dim, depth, heads, mlp_dim, dim_head, top1_acc, top1_acc_train, weight = np.load(filepath)
#         print(n_params, dim, depth, heads, mlp_dim, dim_head, top1_acc, top1_acc_train, weight)