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
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM, MemTransformerLM_flex
from archai.nlp.nvidia_transformer_xl.mem_transformer import PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, DecoderLayer, \
                                                            RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer, AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl import data_utils

from gather_results import get_metrics
from generate_archs import get_yaml_values
from flops_profile import recurse_dir


def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)

''' forward functions for torch profiler
def _forward_with_output_memtransformer(self, dec_inp, mems=None):
  with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as p:
    qlen, bsz = dec_inp.size()
    with record_function("embedding calculation"):
      word_emb = self.word_emb(dec_inp)

    with record_function("create attention mask"):
      mlen = mems[0].size(0) if mems is not None else 0
      klen = mlen + qlen
      if self.same_length:
          all_ones = word_emb.new_ones(qlen, klen)
          mask_len = klen - self.mem_len - 1
          if mask_len > 0:
              mask_shift_len = qlen - mask_len
          else:
              mask_shift_len = qlen
          dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                            + torch.tril(all_ones, -mask_shift_len)).bool()
      else:
          dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()

    hids = []
    # default
    if self.attn_type == 0:
        with record_function("add positional embedding"):
          pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                                  dtype=word_emb.dtype)
          if self.clamp_len > 0:
              pos_seq.clamp_(max=self.clamp_len)
          pos_emb = self.pos_emb(pos_seq)

          core_out = self.drop(word_emb)
          pos_emb = self.drop(pos_emb)

        for i, layer in enumerate(self.layers):
          with record_function("decoder layer {}".format(i)):
            hids.append(core_out.detach())
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                              self.r_r_bias, dec_attn_mask=dec_attn_mask,
                              mems=mems_i)
    else:
      raise NotImplemented
    
    with record_function("output dropout and memory update"):
      core_out = self.drop(core_out)
      new_mems = self._update_mems(hids, mems, qlen, mlen)

  # print(p.key_averages(group_by_input_shape=True).table(sort_by="cpu_memory_usage", row_limit=-1))
  logs = {}
  for event in p.profiler.function_events:
    key = str(event.key)
    logs[key] = event.cpu_memory_usage

  event_keys = ['embedding calculation', 'create attention mask', 'add positional embedding'] + \
                ['decoder layer {}'.format(i) for i in range(len(self.layers))] + ['output dropout and memory update']
  # for k in event_keys:
  #   print('{} -> {} B'.format(k, logs[k]))

  return core_out, new_mems


def _forward_with_output_memtransformer(self, dec_inp, mems=None):
  qlen, bsz = dec_inp.size()
  # with record_function("embedding calculation"):
  word_emb = self.word_emb(dec_inp)

  # with record_function("create attention mask"):
  mlen = mems[0].size(0) if mems is not None else 0
  klen = mlen + qlen
  if self.same_length:
      all_ones = word_emb.new_ones(qlen, klen)
      mask_len = klen - self.mem_len - 1
      if mask_len > 0:
          mask_shift_len = qlen - mask_len
      else:
          mask_shift_len = qlen
      dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                        + torch.tril(all_ones, -mask_shift_len)).bool()
  else:
      dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()

  hids = []
  # default
  if self.attn_type == 0:
      # with record_function("add positional embedding"):
      pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                              dtype=word_emb.dtype)
      if self.clamp_len > 0:
          pos_seq.clamp_(max=self.clamp_len)
      pos_emb = self.pos_emb(pos_seq)

      core_out = self.drop(word_emb)
      pos_emb = self.drop(pos_emb)

      for i, layer in enumerate(self.layers):
        if i==0:
          with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as p:
            with record_function("decoder layer {}".format(i)):
              hids.append(core_out.detach())
              mems_i = None if mems is None else mems[i]
              core_out = layer(core_out, pos_emb, self.r_w_bias,
                                self.r_r_bias, dec_attn_mask=dec_attn_mask,
                                mems=mems_i)
        else:
          hids.append(core_out.detach())
          mems_i = None if mems is None else mems[i]
          core_out = layer(core_out, pos_emb, self.r_w_bias,
                            self.r_r_bias, dec_attn_mask=dec_attn_mask,
                            mems=mems_i)
  else:
    raise NotImplemented

  # with record_function("output dropout and memory update"):
  core_out = self.drop(core_out)
  new_mems = self._update_mems(hids, mems, qlen, mlen)

  print(p.key_averages(group_by_input_shape=True).table(sort_by="cpu_memory_usage", row_limit=-1))
  logs = {}
  for event in p.profiler.function_events:
    key = str(event.key)
    logs[key] = event.cpu_memory_usage

  # event_keys = ['embedding calculation', 'create attention mask', 'add positional embedding'] + \
                # ['decoder layer {}'.format(i) for i in range(len(self.layers))] + ['output dropout and memory update']
  # for k in event_keys:
  #   print('{} -> {} B'.format(k, logs[k]))

  return core_out, new_mems


def forward_with_output_memtransformer(self, data, target, mems):
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
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as p2:
      # with record_function("softmax calculation"):
    inp = pred_hid.view(-1, pred_hid.size(-1))
    output = self.crit.predict(inp)
  
  # print(p2.key_averages(group_by_input_shape=True).table(sort_by="cpu_memory_usage", row_limit=-1))

  # logs = {}
  # for event in p2.profiler.function_events:
  #   key = str(event.key)
  #   logs[key] = event.cpu_memory_usage

  # event_keys = ['softmax calculation']
  # for k in event_keys:
  #   print('{} -> {} B'.format(k, logs[k]))
  
  return (output, new_mems)
'''

def measure_memory(args, run_command):
  os.system(run_command)
  df = pd.read_csv('test.ps.log')
  mems = df[args.mem_type]
  os.system('rm *.log')
  return mems.max()


def get_memories(args, exp_name):
  path_to_results = os.path.join(args.results_dir, exp_name)
  print(path_to_results)
  
  yaml_file = os.path.join(args.results_dir, 'memory_{}.yaml'.format(args.mem_type))
  if False: #os.path.exists(yaml_file):
    with open(yaml_file, 'r') as f:
      memories = yaml.safe_load(f)
  
  else:
    peak_memories = {}
    configs = recurse_dir(args, path_to_results)
    for config_name, all_config in configs.items():
      model_config = all_config['model_config']

      for k, v in model_config.items():
        model_config[k] = get_yaml_values(v)
      
      print('=> Running', config_name)
      command = 'syrupy.py -i 0.0001 --title test -C --no-raw-process-log --separator=, --no-align python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer {n_layer} --n_token {n_token} --n_head {n_head}\
                  --d_head {d_head} --d_model {d_model} --d_embed {d_embed} --d_inner {d_inner} --div_val {div_val}'.format(**model_config)
      curr_peak_memories = []
      for iter in range(5):
        curr_setup_mem = measure_memory(args, run_command=(command + ' --setup_only'))
        curr_model_mem = measure_memory(args, run_command=command)
        # print(curr_model_mem, curr_setup_mem)
        
        if curr_model_mem < curr_setup_mem:
          iter -=1 
          pass
        curr_peak_memories.append(curr_model_mem - curr_setup_mem)
      print(curr_peak_memories)
        
      peak_memories[config_name] = np.mean(curr_peak_memories).tolist()
      print(config_name, peak_memories[config_name])
    
    print('summarized %d configurations' % len(peak_memories.keys()))
    with open(yaml_file, 'w') as f:
        yaml.dump(peak_memories, f)

def plot(args):
  # load the ground-truth rankings
  yaml_file = os.path.join(args.results_dir, 'result_summary.yaml')
  with open(yaml_file, 'r') as f:
    results_gt = collections.OrderedDict(yaml.safe_load(f))

  scores = {}
  for file in os.listdir(args.results_dir):
    if 'synflow_scores_seed' in file:
      seed = re.search('seed_([0-9]+)', file).group(1)
      with open(os.path.join(args.results_dir, file), 'r') as f:
        print('loading scores for seed ', seed)
        scores[seed] = yaml.safe_load(f)
  
  common_ratios = {}
  spr_ranks = {}
  for seed in scores.keys():
    common_configs = np.intersect1d(list(results_gt.keys()), list(scores[seed].keys()))
    print('analyzing {} architectures'.format(len(common_configs)))

    # fear_stage_1 results:
    val_ppl_list_gt = []
    for k in common_configs:
      val_ppl_list_gt.append(results_gt[k]['valid_perplexity'])
    sorted_ground_truth = np.argsort(val_ppl_list_gt)

    # zero-cost score results:
    target_scores = []
    for k in common_configs:
      target_scores.append(-scores[seed][k])   # the higher the score, the better the architecture (reversely correlated with ppl)
    sorted_target = np.argsort(target_scores)

    common_ratios[seed] = []
    spr_ranks[seed] = []
    # extract common ratio and spearmanrank
    topk_list = range(10,101,10)
    for topk in topk_list:
      common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_target, \
                                            val_ppl_list_gt=val_ppl_list_gt, val_ppl_list_target=target_scores)
      common_ratios[seed].append(common_ratio)
      spr_ranks[seed].append(spr_rank)

  plt.figure()
  for seed in common_ratios.keys():
    plt.scatter(topk_list, common_ratios[seed], label='seed_'+seed)
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.title('ranking based on Synflow')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.grid(axis='y')
  plt.savefig('common_ratio_topk_synflow.png', bbox_inches="tight")

  plt.figure()
  for seed in common_ratios.keys():
    plt.scatter(topk_list, spr_ranks[seed], label='seed_'+seed)
  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.ylim(top=1)
  plt.grid(axis='y')
  plt.title('ranking based on Synflow')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig('spearman_topk_synflow.png', bbox_inches="tight")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Results Analysis.')
  parser.add_argument('--results_dir', type=str, default='/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/',
                      help='path where amulet results are downloaded')
  parser.add_argument('--exp_name', type=lambda s: [item for item in s.split(',')], #required=True,
                      help='name of maulet experiment')
  parser.add_argument('--seed', type=int, default=1111, help='Random seed')
  parser.add_argument('--plot', action='store_true', help='plot the spearman corr and common ratio')
  parser.add_argument('--mem_type', default='RSS', help='choose from [RSS, VSIZE]')
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  for exp_name in args.exp_name:
    get_memories(args, exp_name)
  
  if args.plot:
    plot(args)
