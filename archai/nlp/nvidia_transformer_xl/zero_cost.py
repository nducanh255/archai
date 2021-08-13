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
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import torch
import torch.nn as nn

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM, MemTransformerLM_flex
from archai.nlp.nvidia_transformer_xl.mem_transformer import PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, DecoderLayer, \
                                                            RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer, AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl.utils import get_parameter_breakdown, get_list_of_layers, recurse_dir
from archai.nlp.nvidia_transformer_xl import data_utils
from archai.nlp.nvidia_transformer_xl.train import weights_init

from gather_results import get_metrics, process_parameters
from zero_cost_utils import compute_synflow_per_weight


def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)


def _forward_synflow_memformer(self, dec_inp, mems=None):
  qlen, bsz = dec_inp.size()
  word_emb = self.word_emb(dec_inp)

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
      pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                              dtype=word_emb.dtype)
      if self.clamp_len > 0:
          pos_seq.clamp_(max=self.clamp_len)
      pos_emb = self.pos_emb(pos_seq)

      core_out = self.drop(word_emb)
      pos_emb = self.drop(pos_emb)

      # mask everything to all one for synflow
      core_out = torch.ones_like(core_out, dtype=core_out.dtype, device=core_out.device)
      pos_emb = torch.ones_like(pos_emb, dtype=pos_emb.dtype, device=pos_emb.device)

      for i, layer in enumerate(self.layers):
          hids.append(core_out.detach())
          mems_i = None if mems is None else mems[i]
          core_out = layer(core_out, pos_emb, self.r_w_bias,
                            self.r_r_bias, dec_attn_mask=dec_attn_mask,
                            mems=mems_i)
  else:
    raise NotImplemented
  
  # # learnable
  # elif self.attn_type == 1:
  #     core_out = self.drop(word_emb)
  #     for i, layer in enumerate(self.layers):
  #         hids.append(core_out.detach())
  #         if self.clamp_len > 0:
  #             r_emb = self.r_emb[i][-self.clamp_len:]
  #             r_bias = self.r_bias[i][-self.clamp_len:]
  #         else:
  #             r_emb, r_bias = self.r_emb[i], self.r_bias[i]

  #         mems_i = None if mems is None else mems[i]
  #         core_out = layer(core_out, r_emb, self.r_w_bias[i],
  #                           r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
  # # absolute
  # elif self.attn_type == 2:
  #     pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
  #                             dtype=word_emb.dtype)
  #     if self.clamp_len > 0:
  #         pos_seq.clamp_(max=self.clamp_len)
  #     pos_emb = self.pos_emb(pos_seq)

  #     core_out = self.drop(word_emb + pos_emb[-qlen:])

  #     for i, layer in enumerate(self.layers):
  #         hids.append(core_out.detach())
  #         mems_i = None if mems is None else mems[i]
  #         if mems_i is not None and len(mems_i) and i == 0:
  #             mems_i += pos_emb[:mlen]
  #         core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
  #                           mems=mems_i)
  # elif self.attn_type == 3:
  #     core_out = self.drop(word_emb)

  #     for i, layer in enumerate(self.layers):
  #         hids.append(core_out.detach())
  #         mems_i = None if mems is None else mems[i]
  #         if mems_i is not None and len(mems_i) and mlen > 0:
  #             cur_emb = self.r_emb[i][:-qlen]
  #             cur_size = cur_emb.size(0)
  #             if cur_size < mlen:
  #                 cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
  #                 cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
  #             else:
  #                 cur_emb = cur_emb[-mlen:]
  #             mems_i += cur_emb.view(mlen, 1, -1)
  #         core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

  #         core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
  #                           mems=mems_i)

  core_out = self.drop(core_out)

  new_mems = self._update_mems(hids, mems, qlen, mlen)

  return core_out, new_mems

def _forward_synflow_memformer_flex(self, dec_inp, mems=None):
  qlen, bsz = dec_inp.size()
  word_emb = self.word_emb(dec_inp)

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
      pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                              dtype=word_emb.dtype)
      if self.clamp_len > 0:
          pos_seq.clamp_(max=self.clamp_len)
      pos_emb = self.pos_emb(pos_seq)

      core_out = self.drop(word_emb)
      pos_emb = self.drop(pos_emb)

      # mask everything to all one for synflow
      core_out = torch.ones_like(core_out, dtype=core_out.dtype, device=core_out.device)
      pos_emb = torch.ones_like(pos_emb, dtype=pos_emb.dtype, device=pos_emb.device)

      for i, layer in enumerate(self.layers):
          hids.append(core_out.detach())
          mems_i = None if mems is None else mems[i]
          core_out = layer(core_out, pos_emb, self.r_w_bias[i],
                                 self.r_r_bias[i], dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
  else:
    raise NotImplemented
  
  # # learnable
  # elif self.attn_type == 1:
  #     core_out = self.drop(word_emb)
  #     for i, layer in enumerate(self.layers):
  #         hids.append(core_out.detach())
  #         if self.clamp_len > 0:
  #             r_emb = self.r_emb[i][-self.clamp_len:]
  #             r_bias = self.r_bias[i][-self.clamp_len:]
  #         else:
  #             r_emb, r_bias = self.r_emb[i], self.r_bias[i]

  #         mems_i = None if mems is None else mems[i]
  #         core_out = layer(core_out, r_emb, self.r_w_bias[i],
  #                           r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
  # # absolute
  # elif self.attn_type == 2:
  #     pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
  #                             dtype=word_emb.dtype)
  #     if self.clamp_len > 0:
  #         pos_seq.clamp_(max=self.clamp_len)
  #     pos_emb = self.pos_emb(pos_seq)

  #     core_out = self.drop(word_emb + pos_emb[-qlen:])

  #     for i, layer in enumerate(self.layers):
  #         hids.append(core_out.detach())
  #         mems_i = None if mems is None else mems[i]
  #         if mems_i is not None and len(mems_i) and i == 0:
  #             mems_i += pos_emb[:mlen]
  #         core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
  #                           mems=mems_i)
  # elif self.attn_type == 3:
  #     core_out = self.drop(word_emb)

  #     for i, layer in enumerate(self.layers):
  #         hids.append(core_out.detach())
  #         mems_i = None if mems is None else mems[i]
  #         if mems_i is not None and len(mems_i) and mlen > 0:
  #             cur_emb = self.r_emb[i][:-qlen]
  #             cur_size = cur_emb.size(0)
  #             if cur_size < mlen:
  #                 cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
  #                 cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
  #             else:
  #                 cur_emb = cur_emb[-mlen:]
  #             mems_i += cur_emb.view(mlen, 1, -1)
  #         core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

  #         core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
  #                           mems=mems_i)

  core_out = self.drop(core_out)

  new_mems = self._update_mems(hids, mems, qlen, mlen)

  return core_out, new_mems

def forward_synflow(self, data, target, mems):
    if mems is None:
        mems = self.init_mems()

    tgt_len = target.size(0)
    hidden, new_mems = self._forward(data, mems=mems)

    pred_hid = hidden[-tgt_len:]
    return pred_hid.view(-1, pred_hid.size(-1))

    # logits = None
    # if self.sample_softmax > 0 and self.training:
    #     assert self.tie_weight
    #     logit = sample_logits(self.word_emb, self.out_layer.bias, target,
    #                             pred_hid, self.sampler)
    #     loss = -F.log_softmax(logit, -1)[:, :, 0]
    # else:
    #     loss, logits = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
    #     loss = loss.view(tgt_len, -1)
    # if logits is not None:
    #     return (loss, new_mems, logits)
    # else:
    #     return (loss, new_mems)

def get_scores(args, exp_name):
  model_config_keys = ['n_token', 'n_layer','n_head','d_model','d_head','d_inner','dropout','dropatt', \
                      'd_embed','div_val','pre_lnorm','tgt_len','ext_len','mem_len', \
                      'same_length','attn_type','clamp_len','sample_softmax']
  
  path_to_results = os.path.join(args.results_dir, exp_name) 
  yaml_file = os.path.join(path_to_results, 'synflow_scores_seed_{}.yaml'.format(args.seed))
  if not os.path.exists(yaml_file):
    jobs = os.listdir(path_to_results)
    device = torch.device("cpu")
    scores = {}
    nparams = {}
    for j in jobs:
      j_path = os.path.join(path_to_results, j)
      if not os.path.isdir(j_path):
        continue
      config_name = re.search('(config_[0-9]+)', j).group(1)
      path_to_config = recurse_dir(j_path, filename='config.yaml', path_to_ref=None)
      if path_to_config:
        with open(path_to_config, 'r') as f:
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

        class config_holder():
          pass
        args_init = config_holder
        args_init.proj_init_std = config['proj_init_std']
        args_init.init_std = config['init_std']
        args_init.init_range = config['init_range']
        args_init.init = config['init']

        if isinstance(model_config['n_head'], list) and len(model_config['n_head'])>1:
          model = MemTransformerLM_flex(**model_config)
          model._forward = types.MethodType(_forward_synflow_memformer_flex, model)
        else:
          model = MemTransformerLM(**model_config)
          model._forward = types.MethodType(_forward_synflow_memformer, model)
        model.apply(functools.partial(weights_init, args_init))
        model = model.to(device)
        # print(model)
        model.forward = types.MethodType(forward_synflow, model)

        B = 4 # bytes per data element
        tgt_len, mem_len, ext_len = 192, 192, 0
        data_len = tgt_len
        data = torch.ones(data_len*B).to(device, torch.long)
        diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

        for idx, (inp, tgt, seqlen, _) in enumerate(diter):
          grads_abs = compute_synflow_per_weight(model, inp, tgt)
          score = np.sum([torch.sum(g).detach().numpy() for g in grads_abs])    
          break
        curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = process_parameters(model)
        
        scores[config_name] = score.tolist()
        nparams[config_name] = {'AdaEmb': float(params_adaptive_embedding), 'Sftmax': float(params_adaptive_softmax), \
                                'Attn': float(params_attention), 'FFN': float(params_ff), 'total': float(curr_n_all_param)} 
        
        print(config_name, scores[config_name], nparams[config_name])

    with open(yaml_file, 'w') as f:
        yaml.dump(scores, f)
    with open(os.path.join(path_to_results, 'synflow_params.yaml'), 'w') as f:
      yaml.dump(nparams, f)

def get_statistics(seed, results_gt, scores, nparams_dict, topk_list):
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
    target_scores.append(-scores[seed][k])#*1./param_count)   # the higher the score, the better the architecture (reversely correlated with ppl)
  sorted_target = np.argsort(target_scores)

  # parameters
  nparams = {}
  for k in common_configs:
    for param_type in nparams_dict[k].keys():
      try:
        nparams[param_type].append(nparams_dict[k][param_type])
      except:
        nparams[param_type] = [nparams_dict[k][param_type]]
  param_corr = {}
  for param_type, target_params in nparams.items():
    param_corr[param_type], _ = spearmanr((-np.asarray(target_scores)).tolist(), target_params)

  common_ratios = []
  spr_ranks = []
  # extract common ratio and spearmanrank
  for topk in topk_list:
    common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_target, \
                                          val_ppl_list_gt=val_ppl_list_gt, val_ppl_list_target=target_scores)
    common_ratios.append(common_ratio)
    spr_ranks.append(spr_rank)

  return common_ratios, spr_ranks, param_corr

def plot(args):
  common_ratios = {}
  spr_ranks = {}
  param_corrs = {}
  legend_keys = []
  
  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)
    legend_key = 'heterogeneous' if 'heterogeneous' in exp_name else 'homogeneous'
    legend_keys.append(legend_key)

    # load the ground-truth rankings
    yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    with open(yaml_file, 'r') as f:
      results_gt = collections.OrderedDict(yaml.safe_load(f))

    with open(os.path.join(path_to_results, 'synflow_params.yaml'), 'r') as f:
      nparams_dict = collections.OrderedDict(yaml.safe_load(f))

    scores = {}
    for file in os.listdir(path_to_results):
      if 'synflow_scores_seed' in file:
        seed = re.search('seed_([0-9]+)', file).group(1)
        with open(os.path.join(path_to_results, file), 'r') as f:
          print('loading scores for seed ', seed)
          scores[seed] = yaml.safe_load(f)
      
    common_ratios[legend_key] = {}
    spr_ranks[legend_key] = {}
    param_corrs[legend_key] = {}
    topk_list = range(10,101,10)
    if args.cross_seed:
      for seed in scores.keys():
        common_ratio, spr_rank, param_corr = get_statistics(seed, results_gt, scores, nparams_dict, topk_list)
        common_ratios[legend_key][seed] = common_ratio
        spr_ranks[legend_key][seed] = spr_rank
        param_corrs[legend_key][seed] = param_corr
    else:
      common_ratio, spr_rank, param_corr = get_statistics(str(args.seed), results_gt, scores, nparams_dict, topk_list)
      common_ratios[legend_key][str(args.seed)] = common_ratio
      spr_ranks[legend_key][str(args.seed)] = spr_rank
      param_corrs[legend_key][str(args.seed)] = param_corr
    
  plt.figure()
  param_types = list(param_corr.keys())
  for lk in legend_keys:
    for seed in common_ratios[lk].keys():
      corrs = [param_corrs[lk][seed][pt] for pt in param_types]
      print(corrs)
      plt.scatter(range(1, len(param_types)+1), corrs, label=lk+'_seed_'+seed)
  plt.xticks(range(1, len(param_types)+1), list(param_types))
  plt.legend()
  plt.ylim((0, 1))
  plt.grid(axis='y')
  plt.title('Synflow score correlation with nparams')
  plt.savefig('synflow_params.png', bbox_inches="tight")

  plt.figure()
  for lk in legend_keys:
    for seed in common_ratios[lk].keys():
      plt.scatter(topk_list, common_ratios[lk][seed], label=lk+'_seed_'+seed)
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.title('ranking based on Synflow')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.grid(axis='y')
  plt.savefig('common_ratio_topk_synflow.png', bbox_inches="tight")

  plt.figure()
  for lk in legend_keys:
    for seed in common_ratios[lk].keys():
      plt.scatter(topk_list, spr_ranks[lk][seed], label=lk+'_seed_'+seed)
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
  parser.add_argument('--cross_seed', action='store_true', help='plot the spearman corr and common ratio for all evaluated seeds')
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  for exp_name in args.exp_name:
    get_scores(args, exp_name)
  
  if args.plot:
    plot(args)