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
import torch.utils.benchmark as benchmark

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM, MemTransformerLM_flex, forward_predict_memtransformer, predict
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


model_config_keys = ['n_token', 'n_layer','n_head','d_model','d_head','d_inner','dropout','dropatt', \
                        'd_embed','div_val','pre_lnorm','tgt_len','ext_len','mem_len', \
                        'same_length','attn_type','clamp_len','sample_softmax']

def recurse_dir(args, path_to_dir):
  results = {}
  for j in os.listdir(path_to_dir):
      j_path = os.path.join(path_to_dir, j)
      if os.path.isdir(j_path):
        results.update(recurse_dir(args, j_path))
      else:
        logs = None
        if 'config.yaml' in j_path:
          with open(os.path.join(j_path), 'r') as f:
            config = yaml.load(f)
            model_config = {k: config[k] for k in model_config_keys}
            
          cutoffs, tie_projs = [], [False]
          if config['adaptive']:
              assert config['dataset'] in ['wt103', 'wt2', 'lm1b']
              if config['dataset'] in ['wt103', 'wt2']:
                  cutoffs = [19997, 39997, 199997]
                  tie_projs += [True] * len(cutoffs)
              elif config['dataset'] == 'lm1b':
                  cutoffs = [59997, 99997, 639997]
                  tie_projs += [False] * len(cutoffs)
          model_config['cutoffs'] = cutoffs
          model_config['tie_projs'] = tie_projs
          model_config['tie_weight'] = config['tied']
          model_config['dtype'] = None
          logs = {'config': config, 'model_config': model_config}
        
        if logs: 
          config_name = get_config_name(j_path) #get_config_name(os.path.basename(os.path.dirname(j_path)))
          print(config_name, logs)
          results[config_name] = logs
  
  return results


def get_latencies(args, exp_name):
  path_to_results = os.path.join(args.results_dir, exp_name)
  
  yaml_file = os.path.join(path_to_results, 'latency_summary_sftmax_1thread.yaml'.format(args.seed))
  if os.path.exists(yaml_file):
    with open(yaml_file, 'r') as f:
      print('Loading latency summary')
      latencies = yaml.safe_load(f)
  
  else:
    train_iter = None
    latencies = {}

    configs = recurse_dir(args, path_to_results)
    for config_name, all_config in configs.items():
      config = all_config['config']
      model_config = all_config['model_config']

      if isinstance(model_config['n_head'], list) and len(model_config['n_head'])>1:
        model = MemTransformerLM_flex(**model_config)
      else:
        model = MemTransformerLM(**model_config)
      model = model.to(device='cpu')
      model.forward = types.MethodType(forward_predict_memtransformer, model)
      model.crit.forward = types.MethodType(predict, model.crit)
      model.eval()

      # data = torch.LongTensor(model_config['tgt_len']).random_(0, config['n_token']).to('cpu')
      # data = data.unsqueeze(-1)
      # out = model(data)

      # num_threads = torch.get_num_threads()
      # print(f'Benchmarking on {num_threads} threads')
      t0 = benchmark.Timer(stmt='model(data)',
                          setup='',
                          globals={'data': torch.LongTensor(model_config['tgt_len']).random_(0, config['n_token']).unsqueeze(-1), 'model':model},
                          # num_threads=num_threads,
                          label='Multithreaded model execution')
      info = t0.timeit(10)
      info._lazy_init()
      curr_latency = info._mean
  
      latencies[config_name] = curr_latency
      print(config_name, latencies[config_name])

    print('summarized %d configurations' % len(latencies.keys()))
    with open(yaml_file, 'w') as f:
        yaml.dump(latencies, f)


def plot(args):
  common_ratios = {}
  spr_ranks = {}
  
  latency_list = {}

  val_ppl_list_gt = {}
  sorted_ground_truth = {}

  legend_keys = []
  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)
    legend_key = 'heterogeneous' if 'heterogeneous' in exp_name else 'homogeneous'
    legend_keys.append(legend_key)

    # load the ground-truth rankings
    yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    with open(yaml_file, 'r') as f:
      results_gt = collections.OrderedDict(yaml.safe_load(f))

    yaml_file = os.path.join(path_to_results, 'latency_summary_sftmax_1thread.yaml'.format(args.seed))
    with open(yaml_file, 'r') as f:
      latencies = yaml.safe_load(f)
    
    common_configs = np.intersect1d(list(results_gt.keys()), list(latencies.keys()))
    print('analyzing {} architectures'.format(len(common_configs)))

    # fear_stage_1 results:
    val_ppl_list_gt[legend_key] = []
    for k in common_configs:
      val_ppl_list_gt[legend_key].append(results_gt[k]['valid_perplexity'])
    sorted_ground_truth[legend_key] = np.argsort(val_ppl_list_gt[legend_key])

    # zero-cost score results:
    latency_list[legend_key] = []
    for k in common_configs:
      latency_list[legend_key].append(-(latencies[k]))   # the higher the latency, the better the architecture (reversely correlated with ppl)
    sorted_latencies = np.argsort(latency_list[legend_key])

    # extract common ratio and spearmanrank
    common_ratios[legend_key] = []
    spr_ranks[legend_key] = []
    
    topk_list = range(10,101,10)
    for topk in topk_list:
      common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth[legend_key], sorted_target=sorted_latencies, \
                                            val_ppl_list_gt=val_ppl_list_gt[legend_key], val_ppl_list_target=latency_list[legend_key])
      common_ratios[legend_key].append(common_ratio)
      spr_ranks[legend_key].append(spr_rank)
  
  plt.figure()
  for k in legend_keys:
      plt.scatter(-np.asarray(latency_list[k])[sorted_ground_truth[k]], np.asarray(val_ppl_list_gt[k])[sorted_ground_truth[k]], label=k)
  plt.ylabel('Validation PPL')
  plt.xlabel('Total Latency')
  plt.title('Pareto Curve')
  plt.grid(axis='y')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig('pareto_latency.png', bbox_inches="tight")

  plt.figure()
  for k in legend_keys:
    plt.scatter(topk_list, common_ratios[k], label=k)
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.title('ranking based on latency')
  plt.grid(axis='y')
  plt.ylim((0,1))
  plt.legend(loc='lower right')
  plt.savefig('common_ratio_topk_latency.png', bbox_inches="tight")

  plt.figure()
  for k in legend_keys:
    plt.scatter(topk_list, spr_ranks[k], label=k)
  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.grid(axis='y')
  plt.ylim((0,1))
  plt.legend(loc='lower right')
  plt.title('ranking based on latency')
  plt.savefig('spearman_topk_latency.png', bbox_inches="tight")

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
    get_latencies(args, exp_name)
  
  if args.plot:
    plot(args)
