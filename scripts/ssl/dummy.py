import argparse
import shutil
import os
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

