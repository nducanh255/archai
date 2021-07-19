import os
import numpy as np
import collections
import yaml
import collections
import argparse
import json
import re
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})


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


def get_info_from_logs(log_file):
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
      ppl = int(float(l.replace('\n','')[idx_ppl:]))
      ppl_key = min(ppl_thresholds, key=lambda x:abs(x-ppl))

      #TODO: fix timing for 24 hour representation
      out_dict[ppl_key] = {}
      out_dict[ppl_key]['time'] = (end_y-start_y)*365*24*3600 + (end_mo-start_mo)*30*24*3600 + (end_d-start_d)*24*3600 + (end_h-start_h)*3600 + (end_m-start_m)*60 + (end_s-start_s)
      assert out_dict[ppl_key]['time'] > 0, print(end_y, end_mo, end_d, end_h, end_m, end_s, start_y, start_mo, start_d, start_h, start_m, start_s)
      step = int(extract_step(lines[idx-2]))
      out_dict[ppl_key]['step'] = step

    elif 'Training time' in l:
      t_minutes = re.search('Training time: ([+-]?([0-9]*[.])?[0-9]+) minutes', l).group(1)
      out_dict[ppl_key]['train_time'] = float(t_minutes) * 60
  
  return out_dict


def get_ppl_from_logs(json_file, step=[], type=None):
  '''
    step: step number to extract the ppl log, live empty to get the final ppl 
    type: select from [test, valid, train]
  '''
  out_dict = {}
  key = type+'_perplexity' if type is not None else None
  
  with open(json_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  
    for line in lines:
      str = re.search('DLLL \{(.+?)\}', line)
      str = '{'+str.group(1)+'}}'
      final_train_log = json.loads(str)

      if final_train_log['step']==step:
        if key is None:
          for k in final_train_log['data'].keys():
            if 'perplexity' in k:
              out_dict[k] = final_train_log['data'][k]
        elif key in final_train_log['data'].keys():
          out_dict[key] = final_train_log['data'][key]
  
  return out_dict


def get_config_name(job):
  idx = list(re.search('config_', job).span())[0]
  return job[idx:]

parser = argparse.ArgumentParser(description='Results Analysis.')
parser.add_argument('--results_dir', type=str, required=True,
                    help='path where amulet results are downloaded')
parser.add_argument('--exp_name', type=lambda s: [item for item in s.split(',')], required=True,
                    help='name of maulet experiment')
parser.add_argument('--step', default=[],
                    help='training step to extract the log from')
parser.add_argument('--log_type', type=str, default=None,
                    help='type of ppl log to extract, select from [test, valid, train]')
parser.add_argument('--n_unfreeze', type=int, default=None,
                    help='number of unforzen layers for fear_stage_2')
parser.add_argument('--analyze', action='store_true',
                    help='analyze yaml results (must provide two experiment names)')
# parser.add_argument('--read_jsons', action='store_true',
#                     help='read json results and summarize in a yaml file')

args = parser.parse_args()
step = [] if args.step==[] else [int(args.step)]

if args.analyze:
  results = {}
  common_ratios = {}
  spr_ranks = {}
  
  for n_unfreeze in [2,3]:
    common_ratios[n_unfreeze] = []
    spr_ranks[n_unfreeze] = []
    
    for exp_name in args.exp_name:
      path_to_results = os.path.join(args.results_dir, exp_name)

      if 'stage_2' in exp_name:
        fname = 'result_summary_unfreeze_{}.yaml'.format(n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
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
    val_ppl_list_stage2 = []
    for k in common_configs:
      val_ppl_list_stage2.append(results['fear_stage_2'][k]['valid_perplexity'])

    sorted_fear = np.argsort(val_ppl_list_stage2)
    
    # extract common ratio and spearmanrank
    topk_list = range(10,101,10)
    for topk in topk_list:
      idx = int(topk/100.*len(sorted_ground_truth))
      sorted_ground_truth_binned = sorted_ground_truth[:idx]
      sorted_fear_binned = sorted_fear[:idx]

      correct = len(np.intersect1d(sorted_fear_binned, sorted_ground_truth_binned))
      total = len(sorted_fear_binned)

      common_ratios[n_unfreeze].append(correct*1./total)
      print('Correctly ranked top %d %% with %.2f accuracy'%(topk, correct*1./total))

      topk_val_ppl_list_stage1 = [val_ppl_list_stage1[i] for i in range(len(val_ppl_list_stage1)) if i in sorted_ground_truth_binned]
      topk_val_ppl_list_stage2 = [val_ppl_list_stage2[i] for i in range(len(val_ppl_list_stage2)) if i in sorted_ground_truth_binned]
      spr_rank, p_val = spearmanr(topk_val_ppl_list_stage1, topk_val_ppl_list_stage2)
      spr_ranks[n_unfreeze].append(spr_rank)
      print('Spearman Correlation on top %d %%: %.3f'%(topk, spr_rank))

  plt.figure()
  for n_unfreeze in common_ratios.keys():
    plt.scatter(topk_list, common_ratios[n_unfreeze], label='{} unfrozen'.format(n_unfreeze))
  
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.legend(loc='lower right')
  plt.grid(axis='y')
  plt.savefig('common_ratio_topk.png', bbox_inches="tight")

  plt.figure()
  for n_unfreeze in spr_ranks.keys():
    plt.scatter(topk_list, spr_ranks[n_unfreeze], label='{} unfrozen'.format(n_unfreeze))
    # for i, label in enumerate(spr_ranks[n_unfreeze]):
    #   plt.text(topk_list[i], spr_ranks[n_unfreeze][i]-0.07, '%.2f'%label)

  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.ylim(top=1)
  plt.grid(axis='y')
  plt.legend(loc='lower right')
  plt.savefig('spearman_topk.png', bbox_inches="tight")
  

# elif args.read_jsons:
#   for exp_name in args.exp_name:
#     path_to_results = os.path.join(args.results_dir, exp_name)
#     jobs = os.listdir(path_to_results)

#     results = {}
#     for j in jobs:
#       if args.n_unfreeze is not None and 'unfreeze_{}'.format(args.n_unfreeze) not in j:
#         continue
#       j_path = os.path.join(path_to_results, j)
#       if os.path.isdir(j_path):
#         for fname in os.listdir(j_path):
#           if '.json' in fname:
#             logs = get_ppl_from_logs(os.path.join(j_path, fname), step=step, type=args.log_type)
#             if logs: 
#               config_name = get_config_name(j)
#               print(config_name, logs)
#               results[config_name] = logs

#     yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
#     with open(yaml_file, 'w') as f:
#       yaml.dump(results, f)

else:
  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)

    if 'stage_2' in exp_name:
      fname = 'result_summary_unfreeze_{}.yaml'.format('2' if args.n_unfreeze is None else args.n_unfreeze)
    else:
      fname = 'result_summary.yaml'
    yaml_file = os.path.join(path_to_results, fname)
    
    if os.path.exists(yaml_file):
      with open(yaml_file, 'r') as f:
        results = yaml.safe_load(f)
        print('Loading results summary')

    else:
      jobs = os.listdir(path_to_results)

      results = {}
      for j in jobs:
        if args.n_unfreeze is not None and 'unfreeze_{}'.format(args.n_unfreeze) not in j:
          continue
        j_path = os.path.join(path_to_results, j)
        if os.path.isdir(j_path):
          for fname in os.listdir(j_path):
            if '.log' in fname:
              logs_log = get_info_from_logs(os.path.join(j_path, fname))
              if logs_log: 
                config_name = get_config_name(j)
                print(config_name, logs_log)
                results[config_name] = logs_log

            if '.json' in fname:
              logs_json = get_ppl_from_logs(os.path.join(j_path, fname), step=step, type=args.log_type)
              if logs_json: 
                config_name = get_config_name(j)
                print(config_name, logs_json)
                results[config_name].update(logs_json)
      
      with open(yaml_file, 'w') as f:
        yaml.dump(results, f)

    target_ppl = 70
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