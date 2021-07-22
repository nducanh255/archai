import torchvision
import argparse
import numpy as np
import pickle
import shutil
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(data_folder, idx):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']
    return x,y
    
rootpath = '../dataroot/ImageNet32'
data = []
labels = []
for i in range(1,11):
    x,y = load_databatch(rootpath,i)
    data.append(x)
    labels.append(y)

data = np.vstack(data).reshape(-1, 3, 32, 32)
print(data.shape)
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

