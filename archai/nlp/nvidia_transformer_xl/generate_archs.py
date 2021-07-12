import os
import numpy as np
import collections
import yaml
import math
import pprint


_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_representer(dumper, data):
  return dumper.represent_mapping(_mapping_tag, data.items())

def dict_constructor(loader, node):
  return collections.OrderedDict(loader.construct_pairs(node))

# yaml.add_representer(collections.OrderedDict, dict_representer)
# yaml.add_constructor(_mapping_tag, dict_constructor)		
		

n_configs = 20 # number fo configs to generate
start_config = 0

n_layers = [5, 8]
n_heads = [2, 8]
d_models = [64, 512]
d_heads = [32, 128]
d_inners = [256, 1024]
#-----embedding layer
d_embeds = [128, 512]
div_vals = [1, 2, 4]

if __name__ == '__main__':
    path_to_configs = os.path.join('/home/t-mojanj/Projects/archai/archai/nlp/nvidia_transformer_xl', 'configs')
    if not os.path.exists(path_to_configs):
        os.mkdir(path_to_configs)

    bash_file = os.path.join(path_to_configs, 'amlt_run.sh')
    if os.path.exists(bash_file):
      os.remove(bash_file)

    values = {}
    values['n_layer'] = (np.random.randint(low=n_layers[0], high=n_layers[-1], size=n_configs, dtype=np.int32)).tolist()
    values['n_head'] = (np.random.randint(low=n_heads[0], high=n_heads[-1], size=n_configs, dtype=np.int32)).tolist()
    values['d_model'] = (2**np.random.randint(low=np.log2(d_models[0]), high=np.log2(d_models[-1]), size=n_configs, dtype=np.int32)).tolist()
    values['d_head'] = (2**np.random.randint(low=np.log2(d_heads[0]), high=np.log2(d_heads[-1]), size=n_configs, dtype=np.int32)).tolist()
    values['d_inner'] = (2**np.random.randint(low=np.log2(d_inners[0]), high=np.log2(d_inners[-1]), size=n_configs, dtype=np.int32)).tolist()
    values['d_embed'] = (2**np.random.randint(low=np.log2(d_embeds[0]), high=np.log2(d_embeds[-1]), size=n_configs, dtype=np.int32)).tolist()
    values['div_val'] = np.random.choice(div_vals, size=n_configs).astype(np.int32).tolist()
    pprint.pprint(values)

    for c in range(n_configs): 
      with open('/home/t-mojanj/Projects/archaiphilly/nv_train.yaml') as file:
        amlt_config = yaml.load(file)
        if c==0:
          pprint.pprint(amlt_config)

      config_file = 'nv_train_'+str(int(start_config + c))+'.yaml'
      
      
      amlt_config['search']['job_template']['name']= 'train_xl_config_'+str(c)
      # TODO: add vocab_size when there is support for it
      amlt_config['search']['job_template']['command'][3] = 'python -m torch.distributed.launch --nproc_per_node="8" archai/nlp/nvidia_transformer_xl/train.py \
                                      --config {config} --config_file wt103_base_FEAR.yaml \
                                      --attn_type 2 --n_layer {n_layer} --n_head {n_head} --d_model {d_model} --d_head {d_head} \
                                      --d_inner {d_inner} --d_embed {d_embed} --div_val {div_val}' #--vocab_size {vocab_size}
      
      names = list(values.keys())   #['n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'd_embed', 'div_val']
      for n in names:
        # print(n, ':', values[n][c])
        try:
          amlt_config['search']['params'].append({'name':n,
                                                  'spec':'discrete',
                                                  'values': [values[n][c]]})
        except:
          amlt_config['search']['params'] = [{'name':n,
                                                  'spec':'discrete',
                                                  'values': [values[n][c]]}]

      f_name = os.path.join(path_to_configs, config_file)
      with open(f_name, 'w') as file:
          yaml.dump(amlt_config, file)

      with open(bash_file, 'a') as f:
        f.write('amlt run --yes archai/nlp/nvidia_transformer_xl/configs/{} fear_stage_1 \n'.format(config_file))
        f.write('sleep 10 \n')
            
