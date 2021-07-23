import os
from pickle import TRUE
import numpy as np
import collections
import yaml
import collections
import argparse
import json
import re
import pprint
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import tensorwatch as tw

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM
from archai.nlp.nvidia_transformer_xl.mem_transformer import PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, DecoderLayer, \
                                                            RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer, AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl.utils import get_parameter_breakdown, get_list_of_layers

def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)


def get_metrics(topk, sorted_ground_truth, sorted_target, val_ppl_list_gt, val_ppl_list_target, common_configs=None):
  idx = int(topk/100.*len(sorted_ground_truth))
  sorted_ground_truth_binned = sorted_ground_truth[:idx].astype(np.int32)
  sorted_target_binned = sorted_target[:idx].astype(np.int32)

  # print('Calculating common ratio ...')
  # print('ground truth:', np.asarray(val_ppl_list_gt)[sorted_ground_truth_binned])
  # print('topk ground truth configs:', common_configs[sorted_ground_truth_binned])
  # print('baseline:', np.asarray(val_ppl_list_target)[sorted_target_binned])
  # print('topk baseline configs:', common_configs[sorted_target_binned])

  correct = len(np.intersect1d(sorted_target_binned, sorted_ground_truth_binned))
  total = len(sorted_target_binned)
  common_ratio = correct*1./total
  print('Correctly ranked top %d %% (%d) with %.2f accuracy'%(topk, total, correct*1./total))

  # print('Calculating Spearman Corr ...')
  # selected_configs = [common_configs[i] for i in range(len(common_configs)) if i in sorted_ground_truth_binned]
  # print('configs:', selected_configs)
  topk_val_ppl_list_gt = [val_ppl_list_gt[i] for i in range(len(val_ppl_list_gt)) if i in sorted_ground_truth_binned]
  topk_val_ppl_list_target = [val_ppl_list_target[i] for i in range(len(val_ppl_list_target)) if i in sorted_ground_truth_binned]
  spr_rank, _ = spearmanr(topk_val_ppl_list_gt, topk_val_ppl_list_target)
  # print('ground truth:', topk_val_ppl_list_gt)
  # print('baseline:', topk_val_ppl_list_target)
  print('Spearman Correlation on top %d %% (%d): %.3f'%(topk, len(topk_val_ppl_list_gt), spr_rank))

  return common_ratio, spr_rank


def process_parameters(model):
  params_adaptive_embedding, _ = get_parameter_breakdown(model, layerType=[AdaptiveEmbedding])
  params_adaptive_softmax, _ = get_parameter_breakdown(model, layerType=[ProjectedAdaptiveLogSoftmax])
  params_attention, _ = get_parameter_breakdown(model, layerType=[MultiHeadAttn, RelMultiHeadAttn, RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn])
  params_ff, _ = get_parameter_breakdown(model, layerType=[PositionwiseFF])

  params_adaptive_embedding = np.sum(list(params_adaptive_embedding.values()))
  params_adaptive_softmax = np.sum(list(params_adaptive_softmax.values()))
  params_attention = np.sum(list(params_attention.values()))
  params_ff = np.sum(list(params_ff.values()))

  n_all_param = np.sum(params_adaptive_embedding) + np.sum(params_adaptive_softmax) + np.sum(params_attention) + np.sum(params_ff)
  n_nonemb_param = np.sum(params_attention) + np.sum(params_ff)
  print('total parameter size:', n_all_param)
  print('nonemb parameter size:', n_nonemb_param)

  # n_all_param_gt = sum([p.nelement() for p in model.parameters()])
  n_nonemb_param_gt = sum([p.nelement() for p in model.layers.parameters()])
  # assert n_all_param_gt == n_all_param, print(n_all_param_gt, n_all_param)
  assert n_nonemb_param_gt == n_nonemb_param, print(n_nonemb_param_gt, n_nonemb_param)

  return n_all_param, params_adaptive_embedding*100./n_all_param, params_adaptive_softmax*100./n_all_param, params_attention*100./n_all_param, params_ff*100./n_all_param
                          

def extract_date_time(log_str):
  idx_end_time = list(re.search(' - INFO - ', log_str).span())[0]
  idx_start_time = idx_end_time-8
  time_str = log_str[idx_start_time:idx_end_time]

  date_str = log_str[:idx_start_time-1]
  y, mo, d = date_str.split('-')

  h, m, s = time_str.split(':')

  return int(y), int(mo), int(d), int(h), int(m), int(s)


def extract_step(log_str):
  step = log_str.split('|')[1].split('step')[-1]
  return step


def get_info_from_logs(log_file, stage_1=True):
  out_dict = {}
  
  with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  
  ppl_thresholds = None
  for idx, l in enumerate(lines):
    if 'ppl_threshold' in l:
      if ppl_thresholds is None:
        try:
          str = re.search('\[(.+?)\]', l).group(1)
          ppl_thresholds = [int(float(thr)) for thr in str.split(',')]
        except:
          str = re.search('ppl_threshold=([0-9]+),', l).group(1)
          ppl_thresholds = int(float(str))
    
    elif '#params' in l:
      n_params = re.search('#params = ([0-9]+)', l).group(1)
      out_dict['n_params'] = n_params
    
    elif '#non emb params' in l:
      start_y, start_mo, start_d, start_h, start_m, start_s = extract_date_time(lines[idx+1])
    
    elif 'Saving FEAR checkpoint' in l:
      end_y, end_mo, end_d, end_h, end_m, end_s = extract_date_time(l)
      idx_ppl = re.search('ppl', l).span()[-1]
      ppl = float(l.replace('\n','')[idx_ppl:])
      ppl_key = min(ppl_thresholds, key=lambda x:abs(x-int(ppl)))

      out_dict[ppl_key] = {}
      out_dict[ppl_key]['ppl'] = ppl
      out_dict[ppl_key]['time'] = (end_y-start_y)*365*24*3600 + (end_mo-start_mo)*30*24*3600 + (end_d-start_d)*24*3600 + (end_h-start_h)*3600 + (end_m-start_m)*60 + (end_s-start_s)
      assert out_dict[ppl_key]['time'] > 0, print(end_y, end_mo, end_d, end_h, end_m, end_s, start_y, start_mo, start_d, start_h, start_m, start_s)
      step = int(extract_step(lines[idx-2]))
      out_dict[ppl_key]['step'] = step

    elif 'Training time:' in l and stage_1:
      t_minutes = re.search('Training time: ([+-]?([0-9]*[.])?[0-9]+) minutes', l).group(1)
      out_dict['train_time'] = float(t_minutes) * 60
  
  return out_dict


def get_info_from_json(json_file, step=[], type=None):
  '''
    step: step number to extract the ppl log, live empty to get the final ppl 
    type: select from [test, valid, train]
  '''
  out_dict = {}
  key = type+'_perplexity' if type is not None else None
  
  with open(json_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[::-1]
  
    for line in lines:
      str = re.search('DLLL \{(.+?)\}', line)
      str = '{'+str.group(1)+'}}'
      final_train_log = json.loads(str)

      if len(step)>0:
        if final_train_log['step']==[]:
          out_dict['train_elapsed'] = float(final_train_log['data']['train_elapsed'])*60
          for k in final_train_log['data'].keys():
            if 'perplexity' in k:
              out_dict[k] = final_train_log['data'][k]
        
        elif final_train_log['step'][0] in step:
          if final_train_log['step'][0] not in out_dict.keys():
            out_dict[final_train_log['step'][0]] = {}
          for k in final_train_log['data'].keys():
            if 'perplexity' in k:
              out_dict[final_train_log['step'][0]][k] = final_train_log['data'][k]
          # if key is None:
          #   out_dict[final_train_log['step'][0]] = {}
          #   for k in final_train_log['data'].keys():
          #     if 'perplexity' in k:
          #       out_dict[final_train_log['step'][0]][k] = final_train_log['data'][k]
          # elif key in final_train_log['data'].keys():
          #   out_dict[key] = final_train_log['data'][key]

      else:
        try:
          out_dict['train_elapsed'] = float(final_train_log['data']['train_elapsed'])*60
          if key is None:
            for k in final_train_log['data'].keys():
              if 'perplexity' in k:
                out_dict[k] = final_train_log['data'][k]
          elif key in final_train_log['data'].keys():
            out_dict[key] = final_train_log['data'][key]
          break       
        except:
          return None
  
  return out_dict


def get_config_name(job):
  idx = list(re.search('config_', job).span())[0]
  return job[idx:]


def recurse_dir(args, exp_name, path_to_dir, read_log_file=False):
  results = {}
  for j in os.listdir(path_to_dir):
      j_path = os.path.join(path_to_dir, j)
      if os.path.isdir(j_path):
        if args.n_unfreeze is not None and 'unfreeze_{}'.format(args.n_unfreeze) not in j:
          continue
        results.update(recurse_dir(args, exp_name, j_path, read_log_file))
      else:
        if '.log' in j and read_log_file:
          logs_log = get_info_from_logs(j_path, stage_1='stage_1' in exp_name)
          if logs_log: 
            config_name = get_config_name(os.path.basename(os.path.dirname(j_path)))
            print(config_name, logs_log)
            if config_name in results.keys():
              results[config_name].update(logs_log)
            else:
              results[config_name] = logs_log

        if '.json' in j:
          json_logs = get_info_from_json(j_path, step=args.step, type=args.log_type)
          if json_logs: 
            config_name = get_config_name(os.path.basename(os.path.dirname(j_path)))
            print(config_name, json_logs)
            if config_name in results.keys():
              results[config_name].update(json_logs)
            else:
              results[config_name] = json_logs
  
  return results

parser = argparse.ArgumentParser(description='Results Analysis.')
parser.add_argument('--results_dir', type=str, default='/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/',
                    help='path where amulet results are downloaded')
parser.add_argument('--exp_name', type=lambda s: [item for item in s.split(',')], required=True,
                    help='name of maulet experiment')
parser.add_argument('--step', type=lambda s: [int(item) for item in s.split(',')], default=[],
                    help='training step to extract the log from')
parser.add_argument('--cross_step', action='store_true',
                    help='analyze metrics across different steps of fear stage 2')     
parser.add_argument('--log_type', type=str, default=None,
                    help='type of ppl log to extract, select from [test, valid, train]')
parser.add_argument('--n_unfreeze', type=int, default=None,
                    help='number of unforzen layers for fear_stage_2')
parser.add_argument('--analyze', action='store_true',
                    help='analyze yaml results and generate metrics versus topk')
parser.add_argument('--read_jsons', action='store_true',
                    help='read json results and summarize in a yaml file')
parser.add_argument('--generate_plots', action='store_true',
                    help='generate spearman correlation and common ratio plots with baseline included')
parser.add_argument('--analyze_params', action='store_true',
                    help='analyze model parameter size')     
parser.add_argument('--param_ranking', action='store_true',
                    help='generate metrics w.r.t parameter size')    
parser.add_argument('--cross_seeds', action='store_true',
                    help='generate metrics across various seeds')                   

args = parser.parse_args()
# step = [] if args.step==[] else [int(args.step)]

if args.analyze:
  results = {}
  common_ratios = {}
  spr_ranks = {}
  
  fname = 'result_summary.yaml'
  yaml_file = os.path.join(os.path.join(args.results_dir, 'fear_stage_1'), fname)
  assert os.path.exists(yaml_file), 'no result summary for the ground-truth job'
  with open(yaml_file, 'r') as f:
    results['fear_stage_1'] = collections.OrderedDict(yaml.safe_load(f))

  for n_unfreeze in [2,3]:
    for i, exp_name in enumerate(args.exp_name):
      path_to_results = os.path.join(args.results_dir, exp_name)
      assert 'stage_2' in exp_name
      fname = 'result_summary_unfreeze_{}.yaml'.format(n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
      ppl_threshold = 70 if 'ppl' not in exp_name else re.search('ppl_([0-9]+)', exp_name).group(1)
      exp_name = 'fear_stage_2'

      yaml_file = os.path.join(path_to_results, fname)
      if not os.path.exists(yaml_file):
        print('#### no yaml summary found for {} with n_unfreeze={}'.format(args.exp_name[i], n_unfreeze))
        continue
      
      with open(yaml_file, 'r') as f:
        results[exp_name] = collections.OrderedDict(yaml.safe_load(f))

      common_configs = np.intersect1d(list(results['fear_stage_1'].keys()), list(results['fear_stage_2'].keys()))
      print('analyzing {} architectures'.format(len(common_configs)))
      
      # fear_stage_1 results:
      val_ppl_list_stage1 = []
      for k in common_configs:
        val_ppl_list_stage1.append(results['fear_stage_1'][k]['valid_perplexity'])
      sorted_ground_truth = np.argsort(val_ppl_list_stage1)

      # fear_stage_2 results:
      val_ppl_list_stage2 = []
      for k in common_configs:
        val_ppl_list_stage2.append(results['fear_stage_2'][k]['valid_perplexity'])

      sorted_fear = np.argsort(val_ppl_list_stage2)
      
      # extract common ratio and spearmanrank
      key = 'n_unfreeze_{}_ppl_{}'.format(n_unfreeze, ppl_threshold)
      print('--------------- ', key)
      common_ratios[key] = []
      spr_ranks[key] = []

      topk_list = range(10,101,10)
      for topk in topk_list:
        common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth, sorted_target=sorted_fear, \
                                              val_ppl_list_gt=val_ppl_list_stage1, val_ppl_list_target=val_ppl_list_stage2)
        common_ratios[key].append(common_ratio)
        spr_ranks[key].append(spr_rank)

  plt.figure()
  for k in common_ratios.keys():
    plt.plot(topk_list, common_ratios[k], label=k, marker='.', markersize=10)
  
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.legend(loc='lower right')
  plt.grid(axis='y')
  plt.savefig('common_ratio_topk.png', bbox_inches="tight")

  plt.figure()
  for k in spr_ranks.keys():
    plt.plot(topk_list, spr_ranks[k], label=k, marker='.', markersize=10)
    # for i, label in enumerate(spr_ranks[k]):
    #   plt.text(topk_list[i], spr_ranks[k][i]-0.07, '%.2f'%label)

  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.ylim(top=1)
  plt.grid(axis='y')
  plt.legend(loc='lower right')
  plt.savefig('spearman_topk.png', bbox_inches="tight")


if args.cross_seeds:
  results = {}
  common_ratios = {}
  spr_ranks = {}
  
  fname = 'result_summary.yaml'
  yaml_file = os.path.join(os.path.join(args.results_dir, 'fear_stage_1'), fname)
  assert os.path.exists(yaml_file), 'no result summary for the ground-truth job'
  with open(yaml_file, 'r') as f:
    results['fear_stage_1'] = collections.OrderedDict(yaml.safe_load(f))

  for n_unfreeze in [2,3]:
    for i, exp_name in enumerate(args.exp_name):
      path_to_results = os.path.join(args.results_dir, exp_name)
      if 'stage_2' in exp_name:
        fname = 'result_summary_unfreeze_{}.yaml'.format(n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
        ppl_threshold = 70 if 'ppl' not in exp_name else re.search('ppl_([0-9]+)', exp_name).group(1)
        exp_name = 'fear_stage_2'
      else:
        fname = 'result_summary.yaml'

      yaml_file = os.path.join(path_to_results, fname)
      if not os.path.exists(yaml_file):
        print('#### no yaml summary found for {} with n_unfreeze={}'.format(args.exp_name[i], n_unfreeze))
        continue
      with open(yaml_file, 'r') as f:
        results[exp_name] = collections.OrderedDict(yaml.safe_load(f))

      pprint.pprint(results[exp_name])
      exit()

      structured_results = {}
      for k in results[exp_name].keys():
        config_name = re.search('(config_[0-9]+)', k).group(1)
        seed = re.search('(seed_[0-9]+)', k).group(1)
        try:
          structured_results[config_name][seed] = results[exp_name][k]
        except:
          structured_results[config_name] = {}
          structured_results[config_name][seed] = results[exp_name][k]
      
      print(structured_results)
      exit()

      common_configs = np.intersect1d(list(results['fear_stage_1'].keys()), list(structured_results.keys()))
      print('analyzing {} architectures'.format(len(common_configs)))
      
      # fear_stage_1 results:
      val_ppl_list_stage1 = []
      for k in common_configs:
        val_ppl_list_stage1.append(results['fear_stage_1'][k]['valid_perplexity'])
      sorted_ground_truth = np.argsort(val_ppl_list_stage1)

      # target results:
      val_ppl_list_target = {}
      for k in common_configs:
        print('{} has {} seeds'.format(k, len(structured_results[k].keys())))
        for seed in structured_results[k].keys():
          try:
            val_ppl_list_target[seed].append(structured_results[k][seed]['valid_perplexity'])
          except:
            val_ppl_list_target[seed] = [structured_results[k][seed]['valid_perplexity']]
      sorted_target = {}
      for seed in val_ppl_list_target.keys():
        sorted_target[seed] = np.argsort(val_ppl_list_target[seed])

      for s in val_ppl_list_target.keys():
        print(val_ppl_list_target[s][0:6])
        sorted_target[seed][0:6]
      
      exit()
      
      # extract common ratio and spearmanrank
      for seed in val_ppl_list_target.keys():
        print('--------------- ', seed)
        common_ratios[seed] = []
        spr_ranks[seed] = []

        topk_list = range(10,101,10)
        for topk in topk_list:
          common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth, sorted_target=sorted_target[seed], \
                                                val_ppl_list_gt=val_ppl_list_stage1, val_ppl_list_target=val_ppl_list_target[seed])
          common_ratios[seed].append(common_ratio)
          spr_ranks[seed].append(spr_rank)

  plt.figure()
  for k in common_ratios.keys():
    plt.plot(topk_list, common_ratios[k], label=k, marker='.', markersize=10)
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.legend(loc='lower right')
  plt.grid(axis='y')
  plt.savefig('common_ratio_topk_seeds.png', bbox_inches="tight")

  plt.figure()
  for k in spr_ranks.keys():
    plt.plot(topk_list, spr_ranks[k], label=k, marker='.', markersize=10)
    # for i, label in enumerate(spr_ranks[k]):
    #   plt.text(topk_list[i], spr_ranks[k][i]-0.07, '%.2f'%label)
  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.ylim(top=1)
  plt.grid(axis='y')
  plt.legend(loc='lower right')
  plt.savefig('spearman_topk_seeds.png', bbox_inches="tight")
  

elif args.read_jsons:
  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)

    results = {}
    results = recurse_dir(args, exp_name, path_to_results, read_log_file=False)
      
    print('found %d configurations'%len(results.keys()))
    if 'stage_2' in exp_name:
      fname = 'result_summary_unfreeze_{}.yaml'.format('2' if args.n_unfreeze is None else args.n_unfreeze)
    else:
      fname = 'result_summary.yaml'
    yaml_file = os.path.join(path_to_results, fname)
    with open(yaml_file, 'w') as f:
      yaml.dump(results, f)
      print('saved results summary to', fname)


elif args.generate_plots:
  results = {}
  results_structured = {}
  common_ratios = {}
  spr_ranks = {}
  times = {}
  topk_list = [10,20,30,40,50,100]
  
  # load groundtruth results 
  path_to_results = os.path.join(args.results_dir, 'fear_stage_1')
  yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
  with open(yaml_file, 'r') as f:
    results['fear_stage_1'] = collections.OrderedDict(yaml.safe_load(f))

  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)

    if 'stage_2' in exp_name:
      fname = 'result_summary_unfreeze_{}.yaml'.format(args.n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
      target_ppl = 70 if 'ppl' not in exp_name else int(re.search('ppl_([0-9]+)', exp_name).group(1))
      legend_key = 'fear, ppl:{}'.format(target_ppl)
    else:
      fname = 'result_summary.yaml'
      legend_key = exp_name.replace('_', ' ')
      if 'baseline' in exp_name:
        legend_key.replace('fear', '')

    yaml_file = os.path.join(path_to_results, fname)
    with open(yaml_file, 'r') as f:
      results[exp_name] = collections.OrderedDict(yaml.safe_load(f))

    common_ratios[legend_key] = {}
    spr_ranks[legend_key] = {}
    times[legend_key] = {}

    if 'fear_baseline' in exp_name:
      for k, v in results['fear_baseline'].items():
        max_step = k.split('_')[-1]
        config_name = re.search('(config_[0-9]+)_', k).group(1)
        if max_step not in results_structured.keys():
          results_structured[max_step] = {}
        results_structured[max_step][config_name] = v  

      # parse fear_baseline results:
      val_ppl_list_gt = {}
      val_ppl_list_baseline = {}
      timing_baseline = {}
      common_configs = {}
      
      for max_step, v in results_structured.items():
        val_ppl_list_baseline[max_step] = []
        timing_baseline[max_step] = []
        val_ppl_list_gt[max_step] = []
        
        common_configs[max_step] = np.intersect1d(list(results['fear_stage_1'].keys()), list(results_structured[max_step].keys()))
        for k in common_configs[max_step]:
          val_ppl_list_baseline[max_step].append(results_structured[max_step][k]['valid_perplexity'])
          timing_baseline[max_step].append(results_structured[max_step][k]['train_elapsed'])
          val_ppl_list_gt[max_step].append(results['fear_stage_1'][k]['valid_perplexity'])

      for topk in topk_list:
        common_ratios[legend_key][topk] = []
        spr_ranks[legend_key][topk] = []
        times[legend_key][topk] = []
        for max_step in val_ppl_list_baseline.keys():
          print('------------ {} total number of configs with steps={}'.format(len(val_ppl_list_gt[max_step]), max_step))
          sorted_ground_truth = np.argsort(val_ppl_list_gt[max_step])
          sorted_baseline = np.argsort(val_ppl_list_baseline[max_step])
          common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_baseline, \
                                                  val_ppl_list_gt=val_ppl_list_gt[max_step], val_ppl_list_target=val_ppl_list_baseline[max_step], 
                                                  common_configs=common_configs[max_step])
          
          common_ratios[legend_key][topk].append(common_ratio)
          spr_ranks[legend_key][topk].append(spr_rank)
          times[legend_key][topk].append(np.average(timing_baseline[max_step]))
    
    elif 'fear_stage_2' in exp_name:
      common_configs_stage2 = np.intersect1d(list(results['fear_stage_1'].keys()), list(results[exp_name].keys()))
      
      # parse fear_stage_1 results:
      val_ppl_list_gt_for_fear = []
      for k in common_configs_stage2:
        val_ppl_list_gt_for_fear.append(results['fear_stage_1'][k]['valid_perplexity'])
      sorted_ground_truth_for_fear = np.argsort(val_ppl_list_gt_for_fear)
      
      # parse fear_stage_2 results:
      val_ppl_list_stage2 = []
      timing_stage2 = []
      for k in common_configs_stage2:
        val_ppl_list_stage2.append(results[exp_name][k]['valid_perplexity'])
        timing_stage2.append(results[exp_name][k]['train_elapsed'] + results['fear_stage_1'][k][target_ppl]['time'])
      sorted_fear = np.argsort(val_ppl_list_stage2)
      
      # extract common ratio and spearmanrank
      for topk in topk_list:
        common_ratio_fear, spr_rank_fear = get_metrics(topk, sorted_ground_truth=sorted_ground_truth_for_fear, sorted_target=sorted_fear, \
                                                val_ppl_list_gt=val_ppl_list_gt_for_fear, val_ppl_list_target=val_ppl_list_stage2, common_configs=common_configs_stage2)
            
        common_ratios[legend_key][topk] = common_ratio_fear
        spr_ranks[legend_key][topk] = spr_rank_fear
        times[legend_key][topk] = np.average(timing_stage2)

  markers = ['.', 'v', '*', 'd', 'X', 's']
  for topk in topk_list:
    plt.figure()
    for i, k in enumerate(common_ratios.keys()):
      plt.scatter(times[k][topk], common_ratios[k][topk], label=k, marker=markers[i], s=150)
    plt.ylabel('Common ratio')
    plt.xlabel('Time (s)')
    plt.title('Topk = %d %%' % topk)
    plt.grid(axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('common_ratio_topk_{}.png'.format(topk), bbox_inches="tight")

    plt.figure()
    for i, k in enumerate(spr_ranks.keys()):
      plt.scatter(times[k][topk], spr_ranks[k][topk], label=k, marker=markers[i], s=150)
    plt.ylabel('Spearman\'s Correlation')
    plt.xlabel('Time (s)')
    plt.title('Topk = %d %%' % topk)
    plt.ylim(top=1)
    plt.grid(axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('spearman_topk_{}.png'.format(topk), bbox_inches="tight")

  # results = {}
  # results_structured = {}
  # common_ratios = {}
  # spr_ranks = {}

  # for exp_name in args.exp_name:
  #   path_to_results = os.path.join(args.results_dir, exp_name)

  #   if 'stage_2' in exp_name:
  #     fname = 'result_summary_unfreeze_{}.yaml'.format(args.n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
  #     target_ppl = 70 if 'ppl' not in exp_name else int(re.search('ppl_([0-9]+)', exp_name).group(1))
  #     print(target_ppl)
  #     exp_name = 'fear_stage_2'
  #   else:
  #     fname = 'result_summary.yaml'

  #   yaml_file = os.path.join(path_to_results, fname)
  #   with open(yaml_file, 'r') as f:
  #     results[exp_name] = collections.OrderedDict(yaml.safe_load(f))

  # for k, v in results['fear_baseline'].items():
  #   max_step = k.split('_')[-1]
  #   config_name = re.search('(config_[0-9]+)_', k).group(1)
  #   if max_step not in results_structured.keys():
  #     results_structured[max_step] = {}
  #   results_structured[max_step][config_name] = v  
  
  # if 'fear_stage_2' in results.keys():
  #   common_configs_stage2 = np.intersect1d(list(results['fear_stage_1'].keys()), list(results['fear_stage_2'].keys()))
  #   # fear_stage_1 results:
  #   val_ppl_list_gt_for_fear = []
  #   for k in common_configs_stage2:
  #     val_ppl_list_gt_for_fear.append(results['fear_stage_1'][k]['valid_perplexity'])
  #   sorted_ground_truth_for_fear = np.argsort(val_ppl_list_gt_for_fear)
    
  #   # fear_stage_2 results:
  #   val_ppl_list_stage2 = []
  #   timing_stage2 = []
  #   for k in common_configs_stage2:
  #     val_ppl_list_stage2.append(results['fear_stage_2'][k]['valid_perplexity'])
  #     timing_stage2.append(results['fear_stage_2'][k]['train_elapsed'] + results['fear_stage_1'][k][target_ppl]['time'])
  #   sorted_fear = np.argsort(val_ppl_list_stage2)

  # # fear_baseline results:
  # val_ppl_list_gt = {}

  # val_ppl_list_baseline = {}
  # timing_baseline = {}
  # common_configs = {}

  # for max_step, v in results_structured.items():
  #   val_ppl_list_baseline[max_step] = []
  #   timing_baseline[max_step] = []
  #   val_ppl_list_gt[max_step] = []
    
  #   common_configs[max_step] = np.intersect1d(list(results['fear_stage_1'].keys()), list(results_structured[max_step].keys()))
  #   for k in common_configs[max_step]:
  #     val_ppl_list_baseline[max_step].append(results_structured[max_step][k]['valid_perplexity'])
  #     timing_baseline[max_step].append(results_structured[max_step][k]['train_elapsed'])
    
  #     val_ppl_list_gt[max_step].append(results['fear_stage_1'][k]['valid_perplexity'])

  # # extract common ratio and spearmanrank
  # topk_list = [10,20,30,40,50,100]
  # for topk in topk_list:
  #   if 'fear_stage_2' in results.keys():
  #     common_ratio_fear, spr_rank_fear = get_metrics(topk, sorted_ground_truth=sorted_ground_truth_for_fear, sorted_target=sorted_fear, \
  #                                             val_ppl_list_gt=val_ppl_list_gt_for_fear, val_ppl_list_target=val_ppl_list_stage2, common_configs=common_configs_stage2)
        
  #   common_ratios = []
  #   spr_ranks = []
  #   times = []
    
  #   for max_step in val_ppl_list_baseline.keys():
  #     print('------------ {} total number of configs with steps={}'.format(len(val_ppl_list_gt[max_step]), max_step))
  #     sorted_ground_truth = np.argsort(val_ppl_list_gt[max_step])
  #     sorted_baseline = np.argsort(val_ppl_list_baseline[max_step])

  #     common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_baseline, \
  #                                             val_ppl_list_gt=val_ppl_list_gt[max_step], val_ppl_list_target=val_ppl_list_baseline[max_step], 
  #                                             common_configs=common_configs[max_step])
  #     common_ratios.append(common_ratio)
  #     spr_ranks.append(spr_rank)
      
  #     times.append(np.average(timing_baseline[max_step]))

  #   plt.figure()
  #   plt.scatter(times, common_ratios, label='baseline')
  #   if 'fear_stage_2' in results.keys():
  #     plt.scatter(np.average(timing_stage2), common_ratio_fear, label='fear stage 2')
  #   plt.ylabel('Common ratio')
  #   plt.xlabel('Time (s)')
  #   plt.title('Topk = %d %%' % topk)
  #   plt.grid(axis='y')
  #   plt.legend(loc='lower right')
  #   plt.savefig('common_ratio_topk_{}.png'.format(topk), bbox_inches="tight")

  #   plt.figure()
  #   plt.scatter(times, spr_ranks, label='baseline')
  #   if 'fear_stage_2' in results.keys():
  #     plt.scatter(np.average(timing_stage2), spr_rank_fear, label='fear stage 2')
  #   plt.ylabel('Spearman\'s Correlation')
  #   plt.xlabel('Time (s)')
  #   plt.title('Topk = %d %%' % topk)
  #   plt.ylim(top=1)
  #   plt.grid(axis='y')
  #   plt.legend(loc='lower right')
  #   plt.savefig('spearman_topk_{}.png'.format(topk), bbox_inches="tight")


elif args.analyze_params:
  model_config_keys = ['n_token', 'n_layer','n_head','d_model','d_head','d_inner','dropout','dropatt', \
                      'd_embed','div_val','pre_lnorm','tgt_len','ext_len','mem_len', \
                      'same_length','attn_type','clamp_len','sample_softmax']

  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)
    jobs = os.listdir(path_to_results)

    params_adaptive_embedding_list = []
    params_adaptive_softmax_list = []
    params_attention_list = []
    params_ff_list = []
    
    n_all_params = {}
    for j in jobs:
      if args.n_unfreeze is not None and 'unfreeze_{}'.format(args.n_unfreeze) not in j:
        continue
      j_path = os.path.join(path_to_results, j)
      if os.path.isdir(j_path):
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
              
              curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = process_parameters(model)
              params_adaptive_embedding_list.append(params_adaptive_embedding)
              params_adaptive_softmax_list.append(params_adaptive_softmax)
              params_attention_list.append(params_attention)
              params_ff_list.append(params_ff)

              config_name = re.search('(config_[0-9]+)', j).group(1)
              n_all_params[config_name] = int(curr_n_all_param)
              
              # stats = tw.ModelStats(model, input_shape=[192,1])
              # print(stats)
              exit()

    print(n_all_params)
    
    yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(n_all_params, f)

    fig, ax = plt.subplots()
    data = [params_adaptive_embedding_list, params_adaptive_softmax_list, params_attention_list, params_ff_list]
    bp = ax.boxplot(data, sym='k+', showmeans=True)
    # , label=)
    # plt.boxplot(, label=)
    # plt.boxplot(, label=)
    # plt.boxplot(, label=)
    # plt.legend()
    m = [np.mean(d, axis=0) for d in data]
    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = ' μ={:.2f}'.format(m[i])
        if i>0:
          ax.annotate(text, xy=(x-0.2, y+20))
        else:
          ax.annotate(text, xy=(x, y))

    ax.grid(axis='y')
    plt.xticks(range(1, 5), ['AdaEmb', 'Sftmax', 'Attn', 'FFN'])
    plt.savefig('parameter_breakdown.png', bbox_inches="tight")
 

elif args.param_ranking:
  exp_name = args.exp_name[0]
  path_to_results = os.path.join(args.results_dir, exp_name)

  if 'stage_2' in exp_name:
    fname = 'result_summary_unfreeze_{}.yaml'.format(n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
    exp_name = 'fear_stage_2'
  else:
    fname = 'result_summary.yaml'

  results = {}
  yaml_file = os.path.join(path_to_results, fname)
  with open(yaml_file, 'r') as f:
    results[exp_name] = collections.OrderedDict(yaml.safe_load(f))

  yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
  with open(yaml_file, 'r') as f:
      n_all_params = yaml.safe_load(f)

  common_configs = np.intersect1d(list(results['fear_stage_1'].keys()), list(n_all_params.keys()))
  print('analyzing {} architectures'.format(len(common_configs)))

  # fear_stage_1 results:
  val_ppl_list_stage1 = []
  for k in common_configs:
    val_ppl_list_stage1.append(results['fear_stage_1'][k]['valid_perplexity'])
  sorted_ground_truth = np.argsort(val_ppl_list_stage1)

  # n_param results:
  n_params = []
  for k in common_configs:
    n_params.append(n_all_params[k])
  sorted_fear = np.argsort(n_params)

  common_ratios = []
  spr_ranks = []
  # extract common ratio and spearmanrank
  topk_list = range(10,101,10)
  for topk in topk_list:
    common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_fear, \
                                          val_ppl_list_gt=val_ppl_list_stage1, val_ppl_list_target=n_params)
    common_ratios.append(common_ratio)
    spr_ranks.append(spr_rank)

  plt.figure()
  plt.scatter(topk_list, common_ratios)
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.title('ranking based on number of parameters')
  plt.grid(axis='y')
  plt.savefig('common_ratio_topk_nparams.png', bbox_inches="tight")

  plt.figure()
  plt.scatter(topk_list, spr_ranks)
  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.ylim(top=1)
  plt.grid(axis='y')
  plt.title('ranking based on number of parameters')
  plt.savefig('spearman_topk_nparams.png', bbox_inches="tight")


elif args.cross_step:
  results = {}

  # get baseline
  fname = 'result_summary.yaml'
  yaml_file = os.path.join(os.path.join(args.results_dir, 'fear_stage_1'), fname)
  assert os.path.exists(yaml_file), 'no result summary for the ground-truth job'
  with open(yaml_file, 'r') as f:
    results['fear_stage_1'] = collections.OrderedDict(yaml.safe_load(f))

  # get other experiments
  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)
    if 'stage_2' in exp_name:
      fname = 'result_summary_unfreeze_{}.yaml'.format(args.n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
      ppl_threshold = 70 if 'ppl' not in exp_name else re.search('ppl_([0-9]+)', exp_name).group(1)
      exp_name = 'fear_stage_2'
    else:
      fname = 'result_summary.yaml'

    yaml_file = os.path.join(path_to_results, fname)
    with open(yaml_file, 'r') as f:
      results[exp_name] = collections.OrderedDict(yaml.safe_load(f))
  
    common_configs = np.intersect1d(list(results['fear_stage_1'].keys()), list(results['fear_stage_2'].keys()))
    print('analyzing {} architectures'.format(len(common_configs)))
    
    # fear_stage_1 results:
    val_ppl_list_stage1 = []
    for k in common_configs:
      val_ppl_list_stage1.append(results['fear_stage_1'][k]['valid_perplexity'])
    sorted_ground_truth = np.argsort(val_ppl_list_stage1)

    # fear_stage_2 results:
    common_ratios = {}
    spr_ranks = {}

    for step in args.step:
      val_ppl_list_stage2 = []
      for k in common_configs:
        val_ppl_list_stage2.append(results['fear_stage_2'][k][step]['valid_perplexity'])
      sorted_fear = np.argsort(val_ppl_list_stage2)
      
      # extract common ratio and spearmanrank
      key = 'step_{}'.format(step)
      print('--------------- ', key)
      common_ratios[key] = []
      spr_ranks[key] = []

      topk_list = range(10,101,10)
      for topk in topk_list:
        common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_fear, \
                                              val_ppl_list_gt=val_ppl_list_stage1, val_ppl_list_target=val_ppl_list_stage2)
        common_ratios[key].append(common_ratio)
        spr_ranks[key].append(spr_rank)

  plt.figure()
  for k in common_ratios.keys():
    plt.plot(topk_list, common_ratios[k], label=k, marker='.', markersize=10)

  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.legend(loc='lower right')
  plt.title('n_unfreeze:{}, ppl:{}'.format(args.n_unfreeze, ppl_threshold))
  plt.grid(axis='y')
  plt.savefig('common_ratio_topk_steps.png', bbox_inches="tight")

  plt.figure()
  for k in spr_ranks.keys():
    plt.plot(topk_list, spr_ranks[k], label=k, marker='.', markersize=10)
    # for i, label in enumerate(spr_ranks[k]):
    #   plt.text(topk_list[i], spr_ranks[k][i]-0.07, '%.2f'%label)

  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.ylim(top=1)
  plt.legend(loc='lower right')
  plt.grid(axis='y')
  plt.title('n_unfreeze:{}, ppl:{}'.format(args.n_unfreeze, ppl_threshold))
  plt.savefig('spearman_topk_steps.png', bbox_inches="tight")


else:
  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)

    results = recurse_dir(args, exp_name, path_to_results, read_log_file=True)
    
    if 'stage_2' in exp_name:
      fname = 'result_summary_unfreeze_{}.yaml'.format('2' if args.n_unfreeze is None else args.n_unfreeze)
    else:
      fname = 'result_summary.yaml'
    yaml_file = os.path.join(path_to_results, fname)
    with open(yaml_file, 'w') as f:
      yaml.dump(results, f)

    target_ppl = 70 if 'ppl' not in exp_name else re.search('ppl_([0-9]+)', exp_name).group(1)
    all_val_ppls = []  
    all_times = []
    all_steps = []
    for k, v in results.items():
      try:
        all_val_ppls.append(v['valid_perplexity'])
      except:
        continue

      if 'stage_1' in exp_name:
        all_times.append(v[target_ppl]['time'])
        all_steps.append(v[target_ppl]['step'])
    
    n_params_best = []
    for k, v in results.items():
      try:
        if v['valid_perplexity'] == min(all_val_ppls):
          n_params_best.append(v['n_params'])
      except:
        continue

    print('best achieved ppl: {:.2f} with n_params: {}'.format(min(all_val_ppls), n_params_best))
    
    plt.hist(all_val_ppls, bins=50)
    plt.xlabel('validation perplexity')
    plt.ylabel('# archs')
    plt.title(exp_name+'_unfreeze_{}'.format('2' if args.n_unfreeze is None else args.n_unfreeze))
    plt.savefig('valid_ppl_'+exp_name+'_unfreeze_{}.png'.format('2' if args.n_unfreeze is None else args.n_unfreeze), bbox_inches="tight")
      
    if 'stage_1' in exp_name:
      plt.figure()
      plt.scatter(all_times, all_val_ppls)
      plt.ylabel('Final validation perplexity')
      plt.xlabel('Time to reach threshold val ppl (s)')
      plt.savefig('val_ppl_vs_time_'+exp_name+'.png', bbox_inches="tight")

      ratio_good = np.sum(np.asarray(all_times)<3750)*100./len(all_times)
      print('ratio of good architectures:', ratio_good)

