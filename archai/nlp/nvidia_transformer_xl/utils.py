import os
import math

from archai.nlp.nvidia_transformer_xl.mem_transformer import AdaptiveEmbedding, DecoderLayer, MultiHeadAttn, PositionwiseFF, ProjectedAdaptiveLogSoftmax

def get_list_of_layers(module, layerType=None):
        # returns a list of layers (optionally with a certain type) that have trainable parameters
        
        submodules = list(module.children())
        list_of_layers = []
        
        if layerType is not None:
            for lt in layerType:
                if isinstance(module, lt):
                    return module
        else:
            if len(submodules)==0 and len(list(module.parameters()))>0:
                return module
        
        for m in submodules:
                try:
                    list_of_layers.extend(get_list_of_layers(m, layerType))
                except TypeError:
                    list_of_layers.append(get_list_of_layers(m, layerType))
        
        return list_of_layers


def get_parameter_breakdown(model, layerType=None):
    layers = get_list_of_layers(model, layerType)
    print('found {} layers for parameter computation.'.format(len(layers)))
    all_params = {}
    params_decoder = {}
    idx = 0
    for l in layers:
        l_name = l.__class__.__name__+ '_' + str(idx)
        idx += 1
        if isinstance(l, DecoderLayer):
            decoder_layers = get_list_of_layers(model, layerType=[MultiHeadAttn, PositionwiseFF])
            for sub_l in decoder_layers:
                params_decoder['Decoder_'+str(idx)+'_'+sub_l.__class__.__name__] = sum(p.nelement() for p in sub_l.parameters())
        
        all_params[l_name] = sum([p.nelement() for p in l.parameters()])
        
    # print('Per-layer Parameters:', all_params)
    # print('Decoder Parameters:', params_decoder)

    # p_sum = 0
    # for k, p in all_params.items():
    #     if 'Decoder' in k:
    #         p_sum += p

    return all_params, params_decoder


def recurse_dir(pth, filename='config.yaml', path_to_ref=None):
  content = os.listdir(pth) 
  for c in content:
    curr_pth = os.path.join(pth, c)
    if os.path.isfile(curr_pth) and filename in c:
        path_to_ref = curr_pth
    elif os.path.isdir(curr_pth):
      path_to_ref = recurse_dir(curr_pth, filename, path_to_ref)
  
  return path_to_ref