import os
import time
import math
from typing import Union
import torch.backends.cudnn as cudnn
import torch
import wandb
import random
import hashlib
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, LBFGS, SGD
from torch.utils.data import Dataset, DataLoader

def setup_cuda(seed:Union[float, int], rank:int=0):
    seed = int(seed) + rank
    # setup cuda
    cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True # set to false if deterministic
    torch.set_printoptions(precision=10)
    #cudnn.deterministic = False
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class LinearModel(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.linear(x)

def _calc_learning_rate(init_lr, epoch, batch, n_epochs, nBatch, lr_schedule_type = 'cosine'):
    if lr_schedule_type == 'cosine':
        T_total = n_epochs * nBatch
        T_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * T_cur / T_total))
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr

def adjust_learning_rate(optimizer, init_lr, epoch, batch, n_epochs, nBatch):
    """ adjust learning of a given optimizer and return the new learning rate """
    new_lr = _calc_learning_rate(init_lr, epoch, batch, n_epochs, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)

parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--manual_seed', type=int, default=None)
parser.add_argument('--init_lr', type=float, default=0.05)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')

parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=500)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd', 'lars', 'lbfgs'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--use_wandb', action='store_true')  # opt_param
parser.add_argument('--project_name', type=str, default='proxylessnas_imagenet')  # opt_param
parser.add_argument('--run_name', type=str, default='run_default')  # opt_param
parser.add_argument('--entity', type=str, default='sgirish')  # opt_param

parser.add_argument('--resume', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.manual_seed is not None:
        setup_cuda(args.manual_seed)
    features = torch.load(os.path.join(args.path,'features.pt'))
    dim = features['Xtrain'].size(1)
    num_classes = torch.unique(features['Ytrain']).size(0)
    len_dataset_train = features['Xtrain'].size(0)
    len_dataset_test = features['Xtest'].size(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    epochs = args.n_epochs

    model = LinearModel(dim, num_classes)
    model.to(device)
    if args.opt_type == 'sgd':
        optimizer = SGD(model.parameters(), args.init_lr, weight_decay=args.weight_decay, nesterov= not args.no_nesterov,
                        momentum=args.momentum)
    elif args.opt_type == 'lbfgs':
        optimizer = LBFGS(model.parameters(), args.init_lr)
    else:
        raise NotImplementedError

    if args.use_wandb:
        id = hashlib.md5(args.run_name.encode('utf-8')).hexdigest()
        wandb.init(project=args.project_name,
                    name=args.run_name,
                    config=args.__dict__,
                    id=id,
                    resume=args.resume,
                    dir=args.path,
                    entity=args.entity)
        wandb.define_metric("epoch")
        wandb.define_metric("lr", stesp_metric="epoch")
        wandb.define_metric("epoch_loss_train", step_metric="epoch")
        wandb.define_metric("epoch_top1_train", step_metric="epoch")
        wandb.define_metric("epoch_top5_train", step_metric="epoch")
        wandb.define_metric("epoch_timings", step_metric="epoch")
        wandb.define_metric("epoch_top1_val", step_metric="epoch")
        wandb.define_metric("epoch_top5_val", step_metric="epoch")

    if args.resume and os.path.exists(os.path.join(args.path,'checkpoint_linear.pth')):
        ckpt = torch.load(os.path.join(args.path,'checkpoint_linear.pth'))
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0


    shuffle_idx = np.arange(len(features['Xtrain']))
    for epoch in range(start_epoch, epochs):
        running_loss_train = 0.0
        running_acc1_train = 0.0
        running_acc5_train = 0.0
        running_acc1_test = 0.0
        running_acc5_test = 0.0

        np.random.shuffle(shuffle_idx)
        num_batches_train = len_dataset_train//args.train_batch_size +(1 if len_dataset_train%args.train_batch_size!=0 else 0)
        num_batches_test = len_dataset_test//args.test_batch_size +(1 if len_dataset_test%args.test_batch_size!=0 else 0)

        epoch_time = time.time()
        for i in range(num_batches_train):
            cur_lr = adjust_learning_rate(optimizer, args.init_lr, epoch, i, epochs, num_batches_train)

            x = features['Xtrain'][shuffle_idx[i*args.train_batch_size:(i+1)*args.train_batch_size]].to(device)
            y = features['Ytrain'][shuffle_idx[i*args.train_batch_size:(i+1)*args.train_batch_size]].to(device)


            if args.opt_type == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    return loss
                optimizer.step(closure)

            # Forward pass
            out = model(x)

            # Compute loss
            loss = criterion(out, y)
            with torch.no_grad():
                acc1, acc5 = accuracy(out, y, topk=(1, 5))

            if args.opt_type != 'lbfgs':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update the running loss
            running_loss_train += loss.item()*x.size(0)
            running_acc1_train += acc1.item()*x.size(0)
            running_acc5_train += acc5.item()*x.size(0)
        
        for i in range(num_batches_test):

            x = features['Xtest'][i*args.test_batch_size:(i+1)*args.test_batch_size].to(device)
            y = features['Ytest'][i*args.test_batch_size:(i+1)*args.test_batch_size].to(device)

            # Forward pass
            out = model(x)

            with torch.no_grad():
                acc1, acc5 = accuracy(out, y, topk=(1, 5))

            # Update the running loss
            running_acc1_test += acc1.item()*x.size(0)
            running_acc5_test += acc5.item()*x.size(0)

        elapsed_time = time.time()-epoch_time
        epoch_time = time.time()
        print(f"Epoch: {epoch + 1:02}/{epochs} Train Loss: {running_loss_train/len_dataset_train:.5e} "+\
              f"Train Acc Top1: {running_acc1_train/len_dataset_train:.2f}% "+\
              f"Train Acc Top5: {running_acc5_train/len_dataset_train:.2f}% "+\
              f"Test Acc Top1: {running_acc1_test/len_dataset_test:.2f}% "+\
              f"Test Acc Top5: {running_acc5_test/len_dataset_test:.2f}% "+\
              f"Epoch timings: {elapsed_time:.2f}s LR: {cur_lr:.4e}")
        if args.use_wandb:
            wandb.log({"epoch_loss_train": running_loss_train/len_dataset_train,
                        "epoch_top1_train": running_acc1_train/len_dataset_train,
                        "epoch_top5_train": running_acc5_train/len_dataset_train,
                        "epoch_top1_test": running_acc1_test/len_dataset_test,
                        "epoch_top5_test": running_acc5_test/len_dataset_test,
                        "epoch_timings": elapsed_time,
                        "lr": cur_lr,
                        "epoch":epoch})
        
        torch.save({"epoch":epoch+1, "model":model.state_dict(), "optimizer":optimizer.state_dict()},\
                   os.path.join(args.path,"checkpoint_linear.pth"))

    if os.path.exists(os.path.join(args.path,'features.pt')):
        os.remove(os.path.join(args.path,'features.pt'))