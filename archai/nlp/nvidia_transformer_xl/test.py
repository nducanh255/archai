import os
from pickle import TRUE
import numpy as np
import collections
import yaml
import collections
import argparse
import re
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import torch

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM, MemTransformerLM_flex
from archai.nlp.nvidia_transformer_xl.mem_transformer import PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, DecoderLayer, \
                                                            RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer, AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl.utils import get_list_of_layers
from archai.nlp.nvidia_transformer_xl import data_utils
from archai.nlp.nvidia_transformer_xl.data_utils import get_lm_corpus
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils
from archai.common import utils, common

from gather_results import get_metrics, get_config_name

def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)


model_config_keys = ['n_layer','n_head','d_model','d_head','d_inner','d_embed','div_val']

def recurse_dir(args, path_to_dir):
  results = {}
  for j in os.listdir(path_to_dir):
      j_path = os.path.join(path_to_dir, j)
      if os.path.isdir(j_path):
        results.update(recurse_dir(args, j_path))
      else:
        model_config = None
        if 'config.yaml' in j_path:
          with open(os.path.join(j_path), 'r') as f:
            config = yaml.load(f)
            model_config = {k: config[k] for k in model_config_keys}
        
        if model_config: 
          config_name = get_config_name(j_path) #get_config_name(os.path.basename(os.path.dirname(j_path)))
          results[config_name] = model_config
  
  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Results Analysis.')
  parser.add_argument('--results_dir', type=str, default='/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/',
                      help='path where amulet results are downloaded')
  parser.add_argument('--exp_name', type=lambda s: [item for item in s.split(',')], #required=True,
                      help='name of maulet experiment')
  parser.add_argument('--seed', type=int, default=1111, help='Random seed')
  parser.add_argument('--plot', action='store_true', help='plot the spearman corr and common ratio')
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  for exp_name in args.exp_name:
    yaml_file = os.path.join(args.results_dir, exp_name, 'params_summary.yaml'.format(args.seed))
    with open(yaml_file, 'r') as f:
      print('Loading params summary')
      params = yaml.safe_load(f)

    all_configs = recurse_dir(args, os.path.join(args.results_dir, exp_name))
  
    config_names = np.sort(list(params.keys()))
    params_sorted = np.asarray([params[c]['FFN']+params[c]['Attn'] for c in config_names])

    random_idx = np.random.randint(0, len(params_sorted))
    param_target = params_sorted[random_idx]
    similar_indices = []
    epsilon = 0.05
    for idx, p in enumerate(params_sorted):
      if np.absolute(param_target-p)/param_target <= epsilon:
        similar_indices.append(idx)

    similar_configs = [all_configs[config_names[i]] for i in similar_indices]
    similar_parameters = [(params[c]['FFN'], params[c]['Attn']) for c in config_names[similar_indices]]
    print(similar_configs)
    print(similar_parameters)
    plt.scatter(similar_indices, params_sorted[similar_indices])
    plt.xticks(similar_indices)
    plt.savefig('params.png', bbox_inches="tight")


