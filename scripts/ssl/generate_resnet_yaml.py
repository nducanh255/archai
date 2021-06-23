import yaml

resnets = {}
count = 12
for l1 in [2,3,4]:
    for l2 in [2,3,4]:
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