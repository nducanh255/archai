import os
from pickle import TRUE
import numpy as np
import collections
import yaml
import collections
import argparse
import re
import types
import functools
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM
from archai.nlp.nvidia_transformer_xl.mem_transformer import PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, DecoderLayer, \
                                                            RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer, AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl.utils import get_list_of_layers
from archai.nlp.nvidia_transformer_xl import data_utils
from archai.nlp.nvidia_transformer_xl.data_utils import get_lm_corpus
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils
from archai.common import utils, common

from gather_results import get_metrics
from zero_cost_utils import compute_synflow_per_weight


def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)

def forward_predict_memtransformer(self, data, target, mems):
  # nn.DataParallel does not allow size(0) tensors to be broadcasted.
  # So, have to initialize size(0) mems inside the model forward.
  # Moreover, have to return new_mems to allow nn.DataParallel to piece
  # them together.
  if mems is None:
      mems = self.init_mems()

  tgt_len = target.size(0)
  hidden, new_mems = self._forward(data, mems=mems)

  pred_hid = hidden[-tgt_len:]
  # return pred_hid.view(-1, pred_hid.size(-1))

  if self.sample_softmax > 0 and self.training:
    raise NotImplemented
    # assert self.tie_weight
    # logit = sample_logits(self.word_emb, self.out_layer.bias, target,
    #                         pred_hid, self.sampler)
    # loss = -F.log_softmax(logit, -1)[:, :, 0]
  else:
    output = self.crit.predict(pred_hid.view(-1, pred_hid.size(-1)))
  
  return (output, new_mems)


def get_in_out_shape(self, input, output):
  self.input_size = torch.tensor(input[0].size())
  self.output_size = torch.tensor(output.size())


def get_layer_flops(l):
  if isinstance(l, AdaptiveEmbedding):
    if len(l.emb_projs) > 0:
      return torch.prod(l.output_size) * l.emb_projs[0].size(-1)
    else:
      return torch.tensor([0])
    
  elif isinstance(l, PositionwiseFF):
    return (torch.prod(l.input_size) + torch.prod(l.output_size)) * l.d_inner

  elif isinstance(l, RelPartialLearnableMultiHeadAttn):
    return l.flops

  elif isinstance(l, ProjectedAdaptiveLogSoftmax):
    return l.flops

  else:
    raise NotImplemented

def get_model_flops(model, inp, tgt):
  layers_with_flops = get_list_of_layers(model, layerType=[AdaptiveEmbedding, PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, ProjectedAdaptiveLogSoftmax])
  
  # register forward hooks to record input and output sizes
  hooks = []
  for l in layers_with_flops:
    h = l.register_forward_hook(get_in_out_shape)
    hooks.append(h)
  
  _, mems = model(inp, tgt, None)
  model(inp, tgt, mems)

  flops = 0
  for l in layers_with_flops:
    f = get_layer_flops(l)
    # print(type(l), f)
    flops += f.item()
  
  return flops


def get_flops(args):
  model_config_keys = ['n_token', 'n_layer','n_head','d_model','d_head','d_inner','dropout','dropatt', \
                      'd_embed','div_val','pre_lnorm','tgt_len','ext_len','mem_len', \
                      'same_length','attn_type','clamp_len','sample_softmax']

  yaml_file = os.path.join(args.results_dir, 'flops_summary.yaml'.format(args.seed))
  if os.path.exists(yaml_file):
    with open(yaml_file, 'r') as f:
      flops = yaml.safe_load(f)
  
  else:
    jobs = os.listdir(args.results_dir)
    device = torch.device("cpu")
    flops = {}

    train_iter = None
    for j in jobs:
      j_path = os.path.join(args.results_dir, j)
      if not os.path.isdir(j_path):
        continue
      
      config_name = re.search('(config_[0-9]+)', j).group(1)
      for fname in os.listdir(j_path):
        if 'config.yaml' in fname:
          with open(os.path.join(j_path, fname), 'r') as f:
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

          model = MemTransformerLM(**model_config)
          model.forward = types.MethodType(forward_predict_memtransformer, model)
          model = model.to(device)

          # load data
          if train_iter is None:
            path_to_data = common.default_dataroot()
            path_to_data = utils.full_path(os.path.join(path_to_data,'textpred', exp_utils.dataset_dir_name(config['dataset'])))
            corpus = get_lm_corpus(path_to_data, config['dataset'], config['vocab'], max_size=config['vocab_size'])
            train_iter = corpus.get_iterator('train', 1, config['tgt_len'], device='cpu', ext_len=config['ext_len'])
          
          # B = 1 #batch size 
          # tgt_len, mem_len, ext_len = 192, 192, 0
          # data_len = tgt_len * 10
          # data = torch.LongTensor(data_len*B).random_(0, config['n_token']).to(device)
          # train_iter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

          model.eval()
          for idx, (inp, tgt, seqlen, _) in enumerate(train_iter):
            curr_flops = get_model_flops(model, inp, tgt)
            break
          
          flops[config_name] = curr_flops
          print(config_name, flops[config_name])

    with open(yaml_file, 'w') as f:
        yaml.dump(flops, f)

def plot(args):
  # load the ground-truth rankings
  yaml_file = os.path.join(args.results_dir, 'result_summary.yaml')
  with open(yaml_file, 'r') as f:
    results_gt = collections.OrderedDict(yaml.safe_load(f))

  yaml_file = os.path.join(args.results_dir, 'flops_summary.yaml'.format(args.seed))
  with open(yaml_file, 'r') as f:
    flops = yaml.safe_load(f)
  
  common_configs = np.intersect1d(list(results_gt.keys()), list(flops.keys()))
  print('analyzing {} architectures'.format(len(common_configs)))

  # fear_stage_1 results:
  val_ppl_list_gt = []
  for k in common_configs:
    val_ppl_list_gt.append(results_gt[k]['valid_perplexity'])
  sorted_ground_truth = np.argsort(val_ppl_list_gt)

  # zero-cost score results:
  target_flops = []
  for k in common_configs:
    target_flops.append(flops[k])   # the higher the score, the better the architecture (reversely correlated with ppl)
  sorted_target = np.argsort(target_flops)

  common_ratios = []
  spr_ranks = []
  # extract common ratio and spearmanrank
  topk_list = range(10,101,10)
  for topk in topk_list:
    common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_target, \
                                          val_ppl_list_gt=val_ppl_list_gt, val_ppl_list_target=target_flops)
    common_ratios.append(common_ratio)
    spr_ranks.append(spr_rank)
  
  plt.figure()
  plt.scatter(np.asarray(target_flops)[sorted_ground_truth], np.asarray(val_ppl_list_gt)[sorted_ground_truth])
  plt.ylabel('Validation PPL')
  plt.xlabel('FLOPs')
  plt.title('Pareto Curve')
  plt.grid(axis='y')
  plt.savefig('pareto_flops.png', bbox_inches="tight")

  plt.figure()
  plt.scatter(topk_list, common_ratios)
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.title('ranking based on FLOPs')
  plt.grid(axis='y')
  plt.savefig('common_ratio_topk_flops.png', bbox_inches="tight")

  plt.figure()
  plt.scatter(topk_list, spr_ranks)
  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.ylim(top=1)
  plt.grid(axis='y')
  plt.title('ranking based on FLOPs')
  plt.savefig('spearman_topk_flops.png', bbox_inches="tight")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Results Analysis.')
  parser.add_argument('--results_dir', type=str, default='/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/',
                      help='path where amulet results are downloaded')
  parser.add_argument('--seed', type=int, default=1111, help='Random seed')
  parser.add_argument('--plot', action='store_true', help='plot the spearman corr and common ratio')
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  args.results_dir = os.path.join(args.results_dir, 'fear_stage_1')
  
  get_flops(args)
  
  if args.plot:
    plot(args)
