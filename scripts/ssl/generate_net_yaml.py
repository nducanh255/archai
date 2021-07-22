import yaml

resnets = {}
count = 92
for l1 in [5,6]:
    for l2 in [5,6]:
        for l3 in [2,3,4]:
            for l4 in [2,3,4]:
                layers = [l1,l2,l3,l4]
                if layers == [2,2,2,2]:
                    continue
                cur_resnet = {
                              "depth": sum(layers)*3+2,
                              "layers": layers,
                              "bottleneck": True,
                              "hidden_dim": 2048,
                              "out_features": 128,
                              "compress": False,
                              "width_per_group": 64,
                              "groups": 1
                }
                resnets["resnet_v{}".format(count)] = cur_resnet
                count += 1

with open('confs/algos/simclr_resnets.yaml','a') as f:
    f.write("\n")
    for name, cur_resnet in resnets.items():
        depth, layers, bottleneck, hidden_dim, out_features, compress, width_per_group, groups = cur_resnet.values()
        f.write(f"\n{name}:\n")
        f.write(f"    depth: {depth}\n")
        f.write(f"    layers: [{layers[0]}, {layers[1]}, {layers[2]}, {layers[3]}]\n")
        f.write(f"    bottleneck: {bottleneck}\n")
        f.write(f"    hidden_dim: {hidden_dim}\n")
        f.write(f"    out_features: {out_features}\n")
        f.write(f"    compress: {compress}\n")
        f.write(f"    width_per_group: {width_per_group}\n")
        f.write(f"    groups: {groups}\n")
    # yaml.dump(resnets, f, sort_keys=False)

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


# vggnets = {}
# count = 1
# for l1 in [1,2,3]:
#     for l2 in [1,2,3]:
#         for l3 in [2,3,4]:
#             for l4 in [2,3,4]:
#                 for l5 in [2,3,4]:
#                     for hidden_features_vgg in [512,4096]:
#                         for out_features_vgg in [512,4096]:
#                             if hidden_features_vgg > out_features_vgg:
#                                 continue
#                             for classifier_type in ['A','B','C','D']:
#                                 for drop_prob in [0.0, 0.25, 0.5]:
#                                     for batch_norm in [True,False]:
#                                         layers = [l1,l2,l3,l4,l5]
#                                         cur_vggnet = {
#                                                     "depth": sum(layers)+3,
#                                                     "layers": layers,
#                                                     "classifier_type": classifier_type,
#                                                     "hidden_features_vgg": hidden_features_vgg,
#                                                     "out_features_vgg": out_features_vgg,
#                                                     "batch_norm": batch_norm,
#                                                     "drop_prob": drop_prob,
#                                                     "hidden_dim": 2048,
#                                                     "out_features": 128,
#                                         }
#                                         name = f"vggnet_l{''.join(str(l) for l in layers)}_h{hidden_features_vgg}_o{out_features_vgg}_d{drop_prob}_{classifier_type}"
#                                         name += ("_bn" if batch_norm else "")
#                                         vggnets[name] = cur_vggnet
#                                         if hidden_features_vgg == 4096 and out_features_vgg == 4096 and drop_prob == 0.5:
#                                             if layers in [[1, 1, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 3, 3, 3], [2, 2, 4, 4, 4,]]:
#                                                 name = f"vgg{sum(layers)+3}_{classifier_type}"
#                                                 name += ("_bn" if batch_norm else "")
#                                                 vggnets[name] = cur_vggnet
#                     count += 1

# with open('confs/algos/simclr_vggnets.yaml','w') as f:
#     f.write("\n")
#     for name, cur_vggnet in vggnets.items():
#         depth, layers, classifier_type, hidden_features_vgg, out_features_vgg, batch_norm, drop_prob, hidden_dim, out_features = cur_vggnet.values()
#         f.write(f"\n{name}:\n")
#         f.write(f"    depth: {depth}\n")
#         f.write(f"    layers: [{layers[0]}, {layers[1]}, {layers[2]}, {layers[3]}, {layers[4]}]\n")
#         f.write(f"    classifier_type: {classifier_type}\n")
#         f.write(f"    hidden_features_vgg: {hidden_features_vgg}\n")
#         f.write(f"    out_features_vgg: {out_features_vgg}\n")
#         f.write(f"    batch_norm: {batch_norm}\n")
#         f.write(f"    drop_prob: {drop_prob}\n")
#         f.write(f"    hidden_dim: {hidden_dim}\n")
#         f.write(f"    out_features: {out_features}\n")


        # model = ModelSimCLRViT(image_size = conf_model["image_size"], patch_size = conf_model["patch_size"], dim = conf_model["dim"],
        #         depth = conf_model["depth"], heads = conf_model["heads"], mlp_dim = conf_model["mlp_dim"], pool_dim = conf_model["pool_dim"],
        #         channels = conf_models["channels"], dim_head = conf_models["dim_head"], dropout = conf_models["dropout"],
        #         emb_dropout = conf_models["emb_dropout"])

# vits = {}
# count = 1
# for patch_size in [16,32]:
#     for dim in [256, 512, 1024, 2048]:
#         for depth in [2, 4, 6, 8, 10]:
#             for heads in [4, 8, 16, 32]:
#                 for mlp_dim in [256, 512, 1024, 2048]:
#                     for pool in ['cls','mean']:
#                         for dim_head in [32, 64, 128]:
#                             for dropout in [0.0, 0.1, 0.2]:
#                                     cur_vit = {
#                                                 "patch_size": patch_size,
#                                                 "dim": dim,
#                                                 "depth": depth,
#                                                 "heads": heads,
#                                                 "mlp_dim": mlp_dim,
#                                                 "pool": pool,
#                                                 "dim_head": dim_head,
#                                                 "dropout": dropout
#                                     }
#                                     name = f"vit_ps{patch_size}_dim{dim}_depth{depth}_heads{heads}_mlpdim{mlp_dim}_dimhead{dim_head}_dropout{dropout}_{pool}"
#                                     vits[name] = cur_vit

# with open('confs/algos/simclr_vits.yaml','w') as f:
#     f.write("\n")
#     for name, cur_vit in vits.items():
#         patch_size, dim, depth, heads, mlp_dim, pool, dim_head, dropout = cur_vit.values()
#         f.write(f"\n{name}:\n")
#         f.write(f"    patch_size: {patch_size}\n")
#         f.write(f"    dim: {dim}\n")
#         f.write(f"    depth: {depth}\n")
#         f.write(f"    heads: {heads}\n")
#         f.write(f"    mlp_dim: {mlp_dim}\n")
#         f.write(f"    pool: '{pool}'\n")
#         f.write(f"    channels: 3\n")
#         f.write(f"    dim_head: {dim_head}\n")
#         f.write(f"    dropout: {dropout}\n")
#         f.write(f"    emb_dropout: {dropout}\n")
#         f.write(f"    hidden_dim: 2048\n")
#         f.write(f"    out_features: 128\n")