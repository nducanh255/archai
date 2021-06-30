import yaml

# resnets = {}
# count = 12
# for l1 in [2,3,4]:
#     for l2 in [2,3,4]:
#         for l3 in [2,3,4]:
#             for l4 in [2,3,4]:
#                 layers = [l1,l2,l3,l4]
#                 if layers == [2,2,2,2]:
#                     continue
#                 cur_resnet = {
#                               "depth": sum(layers)*3+2,
#                               "layers": layers,
#                               "bottleneck": True,
#                               "hidden_dim": 2048,
#                               "out_features": 128,
#                               "compress": False,
#                               "width_per_group": 64,
#                               "groups": 1
#                 }
#                 resnets["resnet_v{}".format(count)] = cur_resnet
#                 count += 1

# with open('confs/algos/simclr_resnets.yaml','a') as f:
#     f.write("\n")
#     for name, cur_resnet in resnets.items():
#         depth, layers, bottleneck, hidden_dim, out_features, compress, width_per_group, groups = cur_resnet.values()
#         f.write(f"\n{name}:\n")
#         f.write(f"    depth: {depth}\n")
#         f.write(f"    layers: [{layers[0]}, {layers[1]}, {layers[2]}, {layers[3]}]\n")
#         f.write(f"    bottleneck: {bottleneck}\n")
#         f.write(f"    hidden_dim: {hidden_dim}\n")
#         f.write(f"    out_features: {out_features}\n")
#         f.write(f"    compress: {compress}\n")
#         f.write(f"    width_per_group: {width_per_group}\n")
#         f.write(f"    groups: {groups}\n")
#     # yaml.dump(resnets, f, sort_keys=False)

# cfgs: Dict[str, List[Union[str, int]]] = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #vgg11 v22
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #vgg13 v278
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], #vgg16 v299
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], #vgg19 v320
# }

# classifier_type: str = "A", # Choose from 'A', 'B', 'C'
# init_weights: bool = True,
# drop_prob: float = 0.5,
# hidden_features_vgg: int = 4096,
# out_features_vgg: int = 4096


vggnets = {}
count = 1
for l1 in [1,2,3]:
    for l2 in [1,2,3]:
        for l3 in [2,3,4]:
            for l4 in [2,3,4]:
                for l5 in [2,3,4]:
                    for hidden_features_vgg in [512,4096]:
                        for out_features_vgg in [512,4096]:
                            if hidden_features_vgg > out_features_vgg:
                                continue
                            for classifier_type in ['A','B','C','D']:
                                for drop_prob in [0.0, 0.25, 0.5]:
                                    for batch_norm in [True,False]:
                                        layers = [l1,l2,l3,l4,l5]
                                        cur_vggnet = {
                                                    "depth": sum(layers)+3,
                                                    "layers": layers,
                                                    "classifier_type": classifier_type,
                                                    "hidden_features_vgg": hidden_features_vgg,
                                                    "out_features_vgg": out_features_vgg,
                                                    "batch_norm": batch_norm,
                                                    "drop_prob": drop_prob,
                                                    "hidden_dim": 2048,
                                                    "out_features": 128,
                                        }
                                        name = f"vggnet_l{''.join(str(l) for l in layers)}_h{hidden_features_vgg}_o{out_features_vgg}_d{drop_prob}_{classifier_type}"
                                        name += ("_bn" if batch_norm else "")
                                        vggnets[name] = cur_vggnet
                                        if hidden_features_vgg == 4096 and out_features_vgg == 4096 and drop_prob == 0.5:
                                            if layers in [[1, 1, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 3, 3, 3], [2, 2, 4, 4, 4,]]:
                                                name = f"vgg{sum(layers)+3}_{classifier_type}"
                                                name += ("_bn" if batch_norm else "")
                                                vggnets[name] = cur_vggnet
                    count += 1

with open('confs/algos/simclr_vggnets.yaml','w') as f:
    f.write("\n")
    for name, cur_vggnet in vggnets.items():
        depth, layers, classifier_type, hidden_features_vgg, out_features_vgg, batch_norm, drop_prob, hidden_dim, out_features = cur_vggnet.values()
        f.write(f"\n{name}:\n")
        f.write(f"    depth: {depth}\n")
        f.write(f"    layers: [{layers[0]}, {layers[1]}, {layers[2]}, {layers[3]}, {layers[4]}]\n")
        f.write(f"    classifier_type: {classifier_type}\n")
        f.write(f"    hidden_features_vgg: {hidden_features_vgg}\n")
        f.write(f"    out_features_vgg: {out_features_vgg}\n")
        f.write(f"    batch_norm: {batch_norm}\n")
        f.write(f"    drop_prob: {drop_prob}\n")
        f.write(f"    hidden_dim: {hidden_dim}\n")
        f.write(f"    out_features: {out_features}\n")
    # yaml.dump(resnets, f, sort_keys=False)