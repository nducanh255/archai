import torchvision
import argparse
import numpy as np
import pickle
import shutil
import os
import wandb
import hashlib
m = hashlib.md5("resume_run".encode('utf-8')).hexdigest()
wandb.init(project='my-test-project',name='resume_run', id=m,resume=True)
wandb.define_metric("custom_step")
wandb.define_metric("train_loss", step_metric="custom_step")
wandb.define_metric("validation_loss", step_metric="custom_step")
for i in range(10):
  log_dict = {
      "train_loss": 11/(i+1),
      "custom_step": i**2,
      "validation_loss": 11/(i+1)   
  }
  wandb.log(log_dict)

exit()
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
        # dict = pickle.load(fo, encoding='latin1')
    return dict


def load_databatch(data_folder, idx):
    data_file = os.path.join(data_folder, 'train_data_batch_')
    # data_file = os.path.join(data_folder, 'data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    # mean_image = d['mean']
    return x,y
    
rootpath = '/var/tmp/dataroot/ImageNet64'
# rootpath = '../dataroot/cifar-10-batches-py'
data = []
labels = []
for i in range(1,11):
    x,y = load_databatch(rootpath,i)
    data.append(x)
    labels.append(y)

data = np.vstack(data).reshape(-1, 3, 64, 64)
print(np.mean(data,axis=(0,2,3))/255,np.std(data,axis=(0,2,3))/255)
exit()
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir',
                    dest='output_dir',
                    help='Provide destination host. Defaults to localhost',
                    type=str
                    )
args = parser.parse_args()
args.output_dir = os.path.abspath(
                    os.path.expanduser(
                        os.path.expandvars(args.output_dir)))
print(args.output_dir, os.listdir('/var/tmp/dataroot'))
file_path = os.path.join(args.output_dir,"SimClr_Cifar10")
file_path = args.output_dir
# os.mkdir(args.output_dir)
# shutil.rmtree('/vc_data/users/t-sgirish/{job_name:s}')
if os.path.exists(file_path):
    print("File exists!", file_path)
else:
    print("File no exist", file_path)
    # with open(file_path,"w") as f:
    #     f.write("Yo")

# from torch.distributed.distributed_c10d import broadcast
# import torch.multiprocessing as mp
# import torch.distributed as dist
# import torch

# def main():
#     mp.spawn(main_worker, nprocs=2, args=[2])
# def main_worker(gpu, ngpus_per_node):
#     dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456',
#                                 world_size=ngpus_per_node, rank=gpu)

#     if gpu==0:
#         a = torch.Tensor([1]).to(torch.device(f'cuda:{gpu}'))
#     else:
#         a = torch.Tensor([0]).to(torch.device(f'cuda:{gpu}'))

#     torch.distributed.broadcast(a,0)
#     print(a)


# if __name__ == '__main__':
#     main()