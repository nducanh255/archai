import os
import numpy as np
import collections
import yaml
import math
import pprint
import random
import collections
import re

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)

generation_seed = 1111
np.random.seed(generation_seed)
random.seed(generation_seed)

fear_stage = 1   #3: baseline
n_unfreeze = 3
different_seeds = None #[1111,1009,1200,1234,1302,1562,2222,3334,3425,4567]
max_step = 500

targets = ['itpeastusv100cl', 'itpeastusv100cl2', 'itpseasiav100cl', 'itpscusv100cl']  # amulet targets
gpu_config = ['dgx1_4gpu_fp32'] # dgx1_8gpu_fp16, dgx1_1gpu_fp16, toy, default, dgx1_4gpu_fp16
n_gpus = 4

n_configs = 100 # number fo configs to generate
batch_configs = 10 #how many configs in the same bash file
start_config = 0
indep_dhead = False # if set to False, n_head is determined based on n_head and d_model so that n_head * d_head = d_model
homogeneous_layers = False # if set to True, all decoder layers will have the same config within a model

n_layers = [5, 8]
n_heads = [2, 8]
d_models = [64, 512]
#-----embedding layer
d_embeds = [128, 512]
div_vals = [1, 2, 4]
#-----optional
d_heads = [32, 128]
d_inners = [512, 2048]

#TODO: add this to fear stage 2 and 3
def generate_bash_files(path_to_configs, f_name, exp_name=None):
  assert exp_name is not None, 'provide and experiment name for amulet jobs'
  
  bash_idx = 0
  while True:
    bash_file = os.path.join(path_to_configs, f_name+'_'+str(bash_idx)+'.sh')
    if os.path.exists(bash_file):
          os.remove(bash_file)  
    with open(bash_file, 'a') as f:
      for t in range(len(targets)):
        for i in range(batch_configs):
          job_idx = i + t * batch_configs + bash_idx * len(targets) * batch_configs
          print(job_idx)
          if job_idx >= n_configs:
            break
          if different_seeds:
            f.write('amlt run --yes archai/nlp/nvidia_transformer_xl/configs/{} {} -t {}\n'.format('nv_train_{}.yaml'.format(job_idx), exp_name, targets[t]))
          else:
            f.write('amlt run --yes archai/nlp/nvidia_transformer_xl/configs/{} {} -t {}\n'.format('nv_train_{}.yaml'.format(job_idx), exp_name, targets[t]))          
        if job_idx >= n_configs:
            break      
    if job_idx >= n_configs:
            break

    bash_idx += 1


def get_run_command(max_step, config_num, seed=None):
  if seed:
    assert config_num
    command = 'python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py \
                                        --config {config} --config_file wt103_base_FEAR.yaml \
                                        --n_layer {n_layer} --n_head {n_head} --d_model {d_model} --d_head {d_head} \
                                        --d_inner {d_inner} --d_embed {d_embed} --div_val {div_val} \
                                        --max_step %d --seed %d --experiment_name config_%s_seed_%d' % (str(n_gpus), max_step, seed, config_num, seed)
  else:
	  command = 'python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py \
                                        --config {config} --config_file wt103_base.yaml \
                                        --n_layer {n_layer} --n_head {n_head} --d_model {d_model} --d_head {d_head} \
                                        --d_inner {d_inner} --d_embed {d_embed} --div_val {div_val} \
                                        --max_step %d --experiment_name config_%s_%d --scheduler constant' % (str(n_gpus), max_step, config_num, max_step)

  return command


def get_yaml_values(value):
  if isinstance(value, list):
    value_string = ''
    for v in value:
      value_string += (str(v) + ',')
    return value_string[:-1]
  else:
    return value


#TODO: fix encode function
def encode(config):
  # [n_layers, n_heads, d_models, d_heads, d_inners, d_embeds, div_vals]
  code = []
  for n in config.keys():
    if n=='n_layer':
      code.append(config[n]-n_layers[0])
    elif n=='n_head':
      if homogeneous_layers:
        code.append(math.log2(config[n]/n_heads[0]))
      else:
        for n_head in config[n]:
          code.append(math.log2(n_head/n_heads[0]))
    elif n=='d_model':
      code.append(math.log2(config[n]/d_models[0]))
    elif n=='d_embed':
      code.append(math.log2(config[n]/d_embeds[0]))
    elif n=='div_val':
      code.append(math.log2(config[n]/div_vals[0]))
    elif n=='d_inner':
      if homogeneous_layers:
        code.append(math.log2(config[n]/d_inners[0]))
      else:
        for d_inner in config[n]:
          code.append(math.log2(d_inner/d_inners[0]))
    elif n=='d_head':
      if indep_dhead:
        if homogeneous_layers:
          code.append(math.log2(config[n]/d_heads[0]))
        else:
          for d_head in config[n]:
            code.append(math.log2(d_head/d_heads[0]))
    else:
      assert False, 'invalid parameter name'

  code_val = 0
  for i, c in enumerate(code):
    code_val += c*(10**i)

  return code_val


def get_code(param_values, idx):
  config = collections.OrderedDict({k:param_values[k][idx] for k in param_values.keys()})
  return encode(config)


def generate_params_homogeneous(n_configs):
  # generate n_configs with homogeneous decoder layers 
  values = collections.OrderedDict({})
  values['n_layer'] = (np.random.randint(low=n_layers[0], high=n_layers[-1]+1, size=n_configs, dtype=np.int32)).tolist()
  values['n_head'] = (2**np.random.randint(low=np.log2(n_heads[0]), high=np.log2(n_heads[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  values['d_model'] = (2**np.random.randint(low=np.log2(d_models[0]), high=np.log2(d_models[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  values['d_embed'] = (2**np.random.randint(low=np.log2(d_embeds[0]), high=np.log2(d_embeds[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  values['div_val'] = np.random.choice(div_vals, size=n_configs).astype(np.int32).tolist()
  
  if indep_dhead:
    values['d_head'] = (2**np.random.randint(low=np.log2(d_heads[0]), high=np.log2(d_heads[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  else:
    values['d_head'] = [int(values['d_model'][i] / values['n_head'][i]) for i in range(n_configs)]
  
  # values['d_inner']    = (2**np.random.randint(low=np.log2(d_inners[0]), high=np.log2(d_inners[-1]), size=n_configs, dtype=np.int32)).tolist()
  values['d_inner'] = [random.randint(max(int(2*values['d_model'][i]), d_inners[0]), d_inners[-1]) for i in range(n_configs)]

  return values


def generate_params_heterogeneous(n_configs):
  # generate n_configs with heterogeneous decoder layers 
  values = collections.OrderedDict({})
  values['n_layer'] = (np.random.randint(low=n_layers[0], high=n_layers[-1]+1, size=n_configs, dtype=np.int32)).tolist()
  values['n_head'] = [(2**np.random.randint(low=np.log2(n_heads[0]), high=np.log2(n_heads[-1])+1, size=values['n_layer'][i], dtype=np.int32)).tolist() for i in range(n_configs)]
  values['d_model'] = (2**np.random.randint(low=np.log2(d_models[0]), high=np.log2(d_models[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  values['d_embed'] = (2**np.random.randint(low=np.log2(d_embeds[0]), high=np.log2(d_embeds[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  values['div_val'] = np.random.choice(div_vals, size=n_configs).astype(np.int32).tolist()
  
  if indep_dhead:
    values['d_head'] = [(2**np.random.randint(low=np.log2(d_heads[0]), high=np.log2(d_heads[-1])+1, size=values['n_layer'][i], dtype=np.int32)).tolist() for i in range(n_configs)]
  else:
    values['d_head'] = [(values['d_model'][i]//np.asarray(values['n_head'][i])).tolist() for i in range(n_configs)]
  
  values['d_inner'] = [np.random.randint(low=max(int(2*values['d_model'][i]), d_inners[0]), high=d_inners[-1]+1, size=values['n_layer'][i]).tolist() for i in range(n_configs)]
  return values


if __name__ == '__main__':
  path_to_configs = os.path.join('/home/t-mojanj/Projects/archai/archai/nlp/nvidia_transformer_xl', 'configs')
  if not os.path.exists(path_to_configs):
      os.mkdir(path_to_configs)
  
  if fear_stage==1:
      # generate random architectures
      count = 0
      codes = []
      param_values = {}
      while count < n_configs:
        print('generating a new batch of configs')
        if homogeneous_layers:
          new_param_values = generate_params_homogeneous(n_configs)
          # pprint.pprint(new_param_values)
        else:
          new_param_values = generate_params_heterogeneous(n_configs)

        for idx in range(len(new_param_values['n_layer'])):
          curr_code = get_code(new_param_values, idx)
          if curr_code in codes:
            print('duplicate config')
          else:
            for k in new_param_values.keys():
              if k in param_values.keys():
                param_values[k] += [new_param_values[k][idx]]
              else:
                param_values[k] = [new_param_values[k][idx]]
            codes.append(curr_code)
            count += 1
            if count==n_configs:
              break
      pprint.pprint(param_values)

      # create corresponding yaml files for amulet jobs
      for c in range(n_configs): 
        with open('/home/t-mojanj/Projects/archaiphilly/nv_train.yaml') as file:
          amlt_config = yaml.load(file)
          if c==0:
            pprint.pprint(amlt_config)

        amlt_config['search']['job_template']['sku']= 'G'+str(n_gpus)
        amlt_config['search']['job_template']['name']= 'train_xl_config_'+str(c)
        # TODO: add vocab_size when there is support for it
        amlt_config['search']['job_template']['command'][3] = 'python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py \
                                        --config {config} --config_file wt103_base_FEAR.yaml \
                                        --n_layer {n_layer} --n_head {n_head} --d_model {d_model} --d_head {d_head} \
                                        --d_inner {d_inner} --d_embed {d_embed} --div_val {div_val}' % (str(n_gpus)) #--vocab_size {vocab_size} --attn_type 2 
        amlt_config['search']['params'][0]['values'] = gpu_config
        
        names = list(param_values.keys())   #['n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'd_embed', 'div_val']
        for n in names:
          values = get_yaml_values(param_values[n][c])
          try:
            amlt_config['search']['params'].append({'name':n, 'spec':'discrete', 'values': [values]})
          except:
            amlt_config['search']['params'] = [{'name':n, 'spec':'discrete', 'values': [values]}]

        config_file = 'nv_train_'+str(int(start_config + c))+'.yaml'
        f_name = os.path.join(path_to_configs, config_file)
        with open(f_name, 'w') as file:
            yaml.dump(amlt_config, file)

      generate_bash_files(path_to_configs, f_name='amlt_run_fear_stage1', exp_name='fear_stage_1_heterogeneous')

      local_test_bashfile = os.path.join(path_to_configs, 'local_test.sh')
      if os.path.exists(local_test_bashfile):
            os.remove(local_test_bashfile)  
      with open(local_test_bashfile, 'a') as f:
        for c in range(n_configs):
          keys = ['n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'd_embed', 'div_val']
          values = []
          for k in keys:
            values.append(get_yaml_values(param_values[k][c]))
          command = 'python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer {} --n_head {} --d_model {} --d_head {} --d_inner {} --d_embed {} --div_val {} \n'.format(*values)
          f.write(command) 
          f.write('if [ $? -ne 0 ]; then \n echo FAIL \n exit \n fi \n')

  elif fear_stage==2:
    bash_idx = 0
    while True:
      bash_file = os.path.join(path_to_configs, 'amlt_run_fear_stage2_'+str(bash_idx)+'.sh')
      if os.path.exists(bash_file):
            os.remove(bash_file)
          
      with open(bash_file, 'a') as f:
        for t in range(len(targets)):
          for i in range(batch_configs):
            job_idx = i + t * batch_configs + bash_idx * len(targets) * batch_configs
            print(job_idx)
            if job_idx >= n_configs:
              break
            
            if job_idx % batch_configs == 0:
              f.write('amlt map --yes -t {} ~/Projects/archaiphilly/map.yaml :fear_stage2_unfreeze_{} fear_stage_1 :train_xl_config_{}'.format(targets[t], n_unfreeze, job_idx))
            else:
              f.write(' :train_xl_config_{}'.format(job_idx))
          f.write(' fear_stage_2\n')  
          if job_idx >= n_configs:
              break
      
      if job_idx >= n_configs:
              break

      bash_idx += 1
      
  else:
    files = os.listdir(path_to_configs)
    ref_exp_name = 'fear_stage_1' # name of the amlt experiment with full runs to use as the ground-truth configutation
    for f in files:
      if re.search('(nv_train_[0-9]+.yaml)', f):
        with open(os.path.join(path_to_configs, f), 'r') as file:
          prev_config = yaml.load(file)

        job_name = 'train_xl_config_' + f.replace('.yaml','').split('_')[-1]
        gt_config_path = os.path.join('/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/{}'.format(ref_exp_name), job_name, 'config.yaml')
        if os.path.isfile(gt_config_path):
          with open(gt_config_path, 'r') as file:
            gt_config = yaml.load(file)
          
          config_dict = ['n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'd_embed', 'div_val']
          for k in config_dict:
            for param in prev_config['search']['params']:
              if param['name'] == k:
                if gt_config[k] != param['values'][0]:
                  print('mismatch found in {} inside config {}'.format(k, f))
                break
          print('{} and {} checked'.format(f, job_name))
        else:
          print('##### job {} previously did not run'.format(job_name))
        
        config_num = f.replace('.yaml','').split('_')[-1]

        if different_seeds:
          prev_config['search']['job_template']['name'] = 'train_xl_config_%s_%d' % (config_num, max_step)
        else:
          prev_config['search']['job_template']['name'] = 'train_xl_config_%s_500-5000' % (config_num)
        prev_config['search']['max_trials'] = 8
        prev_config['search']['job_template']['command'] = ['set -e -o xtrace', 'bash scripts/apex_install.sh', 'pip install --user -e .']
        if different_seeds:
          prev_config['search']['job_template']['command'] += [get_run_command(max_step, config_num, s) for s in different_seeds]
        else:
          prev_config['search']['job_template']['command'] += [get_run_command(i, config_num) for i in range(500, 5000, 500)]
        
        for idx, param in enumerate(prev_config['search']['params']):  
          if param['name'] == 'max_step':
            del prev_config['search']['params'][idx]
        
        # prev_config['search']['job_template']['name'] = 'train_xl_config_%s_{max_step}' % (f.replace('.yaml','').split('_')[-1])
        # prev_config['search']['max_trials'] = 8
        # prev_config['search']['job_template']['command'][3] = 'python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py \
        #                                 --config {config} --config_file wt103_base_FEAR.yaml \
        #                                 --n_layer {n_layer} --n_head {n_head} --d_model {d_model} --d_head {d_head} \
        #                                 --d_inner {d_inner} --d_embed {d_embed} --div_val {div_val} --max_step {max_step}' % (str(n_gpus)) #--vocab_size {vocab_size} --attn_type 2 

        # already_has_max_step = False
        # for idx, param in enumerate(prev_config['search']['params']):  
        #   if param['name'] == 'max_step':
        #     prev_config['search']['params'][idx] = [{'name':'max_step',
        #                                             'spec':'discrete',
        #                                             'values': [5000*i for i in range(1,8)]}]
        #     already_has_max_step = True
        #     break
        # if not already_has_max_step:
        #   prev_config['search']['params'].append({'name':'max_step',
        #                                     'spec':'discrete',
        #                                     'values': [5000*i for i in range(1,8)]})
        with open(os.path.join(path_to_configs, f), 'w') as file:
          yaml.dump(prev_config, file)

    exp_name = 'fear_baseline_{}_step'.format(max_step)
    # exp_name = 'fear_baseline_constLR'

    generate_bash_files(path_to_configs, f_name='amlt_run_fear_baseline', exp_name=None)