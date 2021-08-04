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

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM, MemTransformerLM_flex
from archai.nlp.nvidia_transformer_xl.mem_transformer import PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, DecoderLayer, \
                                                            RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer, AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl.utils import get_parameter_breakdown, get_list_of_layers, recurse_dir
from archai.nlp.nvidia_transformer_xl import data_utils

from gather_results import get_metrics
from zero_cost_utils import compute_synflow_per_weight
from generate_archs import recurse_dir


def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)

# def _forward_with_output_memtransformer(self, dec_inp, mems=None):
#   with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as p:
#     qlen, bsz = dec_inp.size()
#     with record_function("embedding calculation"):
#       word_emb = self.word_emb(dec_inp)

#     with record_function("create attention mask"):
#       mlen = mems[0].size(0) if mems is not None else 0
#       klen = mlen + qlen
#       if self.same_length:
#           all_ones = word_emb.new_ones(qlen, klen)
#           mask_len = klen - self.mem_len - 1
#           if mask_len > 0:
#               mask_shift_len = qlen - mask_len
#           else:
#               mask_shift_len = qlen
#           dec_attn_mask = (torch.triu(all_ones, 1+mlen)
#                             + torch.tril(all_ones, -mask_shift_len)).bool()
#       else:
#           dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()

#     hids = []
#     # default
#     if self.attn_type == 0:
#         with record_function("add positional embedding"):
#           pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
#                                   dtype=word_emb.dtype)
#           if self.clamp_len > 0:
#               pos_seq.clamp_(max=self.clamp_len)
#           pos_emb = self.pos_emb(pos_seq)

#           core_out = self.drop(word_emb)
#           pos_emb = self.drop(pos_emb)

#         for i, layer in enumerate(self.layers):
#           with record_function("decoder layer {}".format(i)):
#             hids.append(core_out.detach())
#             mems_i = None if mems is None else mems[i]
#             core_out = layer(core_out, pos_emb, self.r_w_bias,
#                               self.r_r_bias, dec_attn_mask=dec_attn_mask,
#                               mems=mems_i)
#     else:
#       raise NotImplemented
#     # # learnable
#     # elif self.attn_type == 1:
#     #     core_out = self.drop(word_emb)
#     #     for i, layer in enumerate(self.layers):
#     #         hids.append(core_out.detach())
#     #         if self.clamp_len > 0:
#     #             r_emb = self.r_emb[i][-self.clamp_len:]
#     #             r_bias = self.r_bias[i][-self.clamp_len:]
#     #         else:
#     #             r_emb, r_bias = self.r_emb[i], self.r_bias[i]

#     #         mems_i = None if mems is None else mems[i]
#     #         core_out = layer(core_out, r_emb, self.r_w_bias[i],
#     #                           r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
#     # # absolute
#     # elif self.attn_type == 2:
#     #     pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
#     #                             dtype=word_emb.dtype)
#     #     if self.clamp_len > 0:
#     #         pos_seq.clamp_(max=self.clamp_len)
#     #     pos_emb = self.pos_emb(pos_seq)

#     #     core_out = self.drop(word_emb + pos_emb[-qlen:])

#     #     for i, layer in enumerate(self.layers):
#     #         hids.append(core_out.detach())
#     #         mems_i = None if mems is None else mems[i]
#     #         if mems_i is not None and len(mems_i) and i == 0:
#     #             mems_i += pos_emb[:mlen]
#     #         core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
#     #                           mems=mems_i)
#     # elif self.attn_type == 3:
#     #     core_out = self.drop(word_emb)

#     #     for i, layer in enumerate(self.layers):
#     #         hids.append(core_out.detach())
#     #         mems_i = None if mems is None else mems[i]
#     #         if mems_i is not None and len(mems_i) and mlen > 0:
#     #             cur_emb = self.r_emb[i][:-qlen]
#     #             cur_size = cur_emb.size(0)
#     #             if cur_size < mlen:
#     #                 cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
#     #                 cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
#     #             else:
#     #                 cur_emb = cur_emb[-mlen:]
#     #             mems_i += cur_emb.view(mlen, 1, -1)
#     #         core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

#     #         core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
#     #                           mems=mems_i)

#     with record_function("output dropout and memory update"):
#       core_out = self.drop(core_out)
#       new_mems = self._update_mems(hids, mems, qlen, mlen)

#   # print(p.key_averages(group_by_input_shape=True).table(sort_by="cpu_memory_usage", row_limit=-1))
#   logs = {}
#   for event in p.profiler.function_events:
#     key = str(event.key)
#     logs[key] = event.cpu_memory_usage

#   event_keys = ['embedding calculation', 'create attention mask', 'add positional embedding'] + \
#                 ['decoder layer {}'.format(i) for i in range(len(self.layers))] + ['output dropout and memory update']
#   # for k in event_keys:
#   #   print('{} -> {} B'.format(k, logs[k]))

#   return core_out, new_mems

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


def init_weight(weight, config):
    """Intialize given parameters using specified strategy"""
    if config['init'] == 'uniform':
        nn.init.uniform_(weight, -config['init_range'], config['init_range'])
    elif config['init'] == 'normal': # default
        nn.init.normal_(weight, 0.0, config['init_std'])

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m, config):
    """Initialize weights of module using specified strategy"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight, config)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, config['proj_init_std'])
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, config)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight, config)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, config['proj_init_std'])
        if hasattr(m, 'out_layers_weights'):
            for i in range(len(m.out_layers_weights)):
                if m.out_layers_weights[i] is not None:
                    init_weight(m.out_layers_weights[i], config)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, config['init_std'])
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb, config)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias, config)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias, config)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def get_scores(args):
  model_config_keys = ['n_token', 'n_layer','n_head','d_model','d_head','d_inner','dropout','dropatt', \
                      'd_embed','div_val','pre_lnorm','tgt_len','ext_len','mem_len', \
                      'same_length','attn_type','clamp_len','sample_softmax']
  
  yaml_file = os.path.join(args.results_dir, 'synflow_scores_seed_{}.yaml'.format(args.seed))
  if False: #os.path.exists(yaml_file):
    with open(yaml_file, 'r') as f:
      scores = yaml.safe_load(f)
  
  else:
    jobs = os.listdir(args.results_dir)
    device = torch.device("cpu")
    scores = {}
    seen_div_vals = []
    for j in jobs:
      j_path = os.path.join(args.results_dir, j)
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

        if isinstance(model_config['n_head'], list) and len(model_config['n_head'])>1:
          model = MemTransformerLM_flex(**model_config)
        else:
          model = MemTransformerLM(**model_config)
        model = model.to(device)
        # print(model)
        # model.forward = types.MethodType(forward_with_output_memtransformer, model)
        # model._forward = types.MethodType(_forward_with_output_memtransformer, model)

        # data loader
        B = 4 # bytes per data element
        tgt_len, mem_len, ext_len = 192, 192, 0
        data_len = tgt_len * 10
        data = torch.LongTensor(data_len*B).random_(0, config['n_token']).to(device)
        diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

        if not model_config['div_val'] in seen_div_vals:
          os.makedirs('onnx_models', exist_ok=True)
          seen_div_vals.append(model_config['div_val'])
          print('converting to onnx')
          print(model)
          model.eval()
          mems = None
          for idx, (inp, tgt, seqlen, _) in enumerate(diter):
            _, mems = model(inp, tgt, None)
            torch.onnx.export(model, (inp, tgt, mems), os.path.join('onnx_models', 'memformer_{}.onnx'.format(config_name)), opset_version=13)
            break
          
          # model(inp, tgt, None) 
          # break
          # if idx > 5:
          #   break

    # with open(yaml_file, 'w') as f:
    #     yaml.dump(scores, f)

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
  parser.add_argument('--exp_name', type=str, required=True,
                      help='name of maulet experiment')
  parser.add_argument('--seed', type=int, default=1111, help='Random seed')
  parser.add_argument('--plot', action='store_true', help='plot the spearman corr and common ratio')
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  args.results_dir = os.path.join(args.results_dir, args.exp_name)
  
  get_scores(args)
  
  if args.plot:
    plot(args)
