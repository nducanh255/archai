import os
import time
import math
import torch
import wandb
import hashlib
import random
import argparse
import numpy as np
import torch.nn as nn
from typing import Union
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import Adam, LBFGS, SGD
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    elif lr_schedule_type == 'step':
        lr = (0.1**(epoch//40))*init_lr
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
parser.add_argument('--n_worker', type=int, default=-1)
parser.add_argument('--kfolds', type=int, default=3)

parser.add_argument('--use_wandb', action='store_true')  # opt_param
parser.add_argument('--project_name', type=str, default='proxylessnas_imagenet')  # opt_param
parser.add_argument('--run_name', type=str, default='run_default')  # opt_param
parser.add_argument('--entity', type=str, default='sgirish')  # opt_param

parser.add_argument('--use-svm', action='store_true')
parser.add_argument('--resume', action='store_true')


def train(args, Xtrain, Ytrain, Xeval, Yeval, model, print_logs=False):
    if args.opt_type == 'sgd':
        optimizer = SGD(model.parameters(), args.init_lr, weight_decay=args.weight_decay, nesterov= not args.no_nesterov,
                        momentum=args.momentum)
    elif args.opt_type == 'lbfgs':
        optimizer = LBFGS(model.parameters(), args.init_lr)
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()

    len_dataset_train = Xtrain.size(0)
    len_dataset_test = Xeval.size(0)

    shuffle_idx = np.arange(len(Xtrain))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(args.n_epochs):
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
            cur_lr = adjust_learning_rate(optimizer, args.init_lr, epoch, i, args.n_epochs, num_batches_train)

            x = Xtrain[shuffle_idx[i*args.train_batch_size:(i+1)*args.train_batch_size]].to(device)
            y = Ytrain[shuffle_idx[i*args.train_batch_size:(i+1)*args.train_batch_size]].to(device)

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

            x = Xeval[i*args.test_batch_size:(i+1)*args.test_batch_size].to(device)
            y = Yeval[i*args.test_batch_size:(i+1)*args.test_batch_size].to(device)

            # Forward pass
            out = model(x)

            with torch.no_grad():
                acc1, acc5 = accuracy(out, y, topk=(1, 5))

            # Update the running loss
            running_acc1_test += acc1.item()*x.size(0)
            running_acc5_test += acc5.item()*x.size(0)

        elapsed_time = time.time()-epoch_time
        epoch_time = time.time()
        if print_logs:
            print(f"Epoch: {epoch + 1:02}/{args.n_epochs} Train Loss: {running_loss_train/len_dataset_train:.5e} "+\
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
    if args.use_wandb and print_logs:
        wandb.run.summary["best_weight_decay"] = args.weight_decay
        wandb.run.summary["best_batch_size"] = args.train_batch_size
        wandb.run.summary["best_init_lr"] = args.init_lr
    return running_acc1_train/len_dataset_train, running_acc5_train/len_dataset_train, \
            running_acc1_test/len_dataset_test, running_acc5_test/len_dataset_test


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
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("epoch_loss_train", step_metric="epoch")
        wandb.define_metric("epoch_top1_train", step_metric="epoch")
        wandb.define_metric("epoch_top5_train", step_metric="epoch")
        wandb.define_metric("epoch_timings", step_metric="epoch")
        wandb.define_metric("epoch_top1_val", step_metric="epoch")
        wandb.define_metric("epoch_top5_val", step_metric="epoch")
        
    if args.use_svm:
        from thundersvm import SVC
        best_acc = 0
        best_alpha = 1.0
        for alpha in  [0.01, 0.1, 1, 10]:
            print(f'Alpha {alpha}')
            kf = KFold(n_splits=3)
            accs = 0.0
            for train_index, val_index in kf.split(features['Xtrain']):
                # clf = LinearSVC(C=alpha,tol=1e-5, max_iter=3000, verbose=True, dual=False)
                clf = SVC(kernel='linear',C=alpha,tol=1e-5, max_iter=3000, verbose=False,\
                          n_jobs=args.n_worker)
                # clf = SVC(kernel='linear',C=alpha,tol=1.0, verbose=0)
                X_train, X_val = features['Xtrain'][train_index], features['Xtrain'][val_index]
                y_train, y_val = features['Ytrain'][train_index], features['Ytrain'][val_index]
                # clf = make_pipeline(StandardScaler(), clf)
                clf.fit(X_train, y_train)
                accs += clf.score(X_val, y_val)
                print(f'Acc {clf.score(X_val, y_val)}')
            accs = accs/3
            print(f'Accuracy: {accs}')
            if accs>best_acc:
                best_acc = accs
                best_alpha = alpha
        clf = SVC(kernel='linear',C=best_alpha,tol=1e-5, max_iter=3000, verbose=False)
        clf.fit(features['Xtrain'], features['Ytrain'])
        acc = clf.score(features['Xtest'],features['Ytest'])
        print(f'Final Acc {acc}')
        exit()
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    epochs = args.n_epochs

    kf = KFold(n_splits=args.kfolds)
    best_params = {'weight_decay':0.0, 'batch_size':128, 'init_lr':5.0}
    best_acc = 0.0
    for weight_decay in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        for batch_size in [128]:
            for init_lr in [5.0]:
                args.weight_decay = weight_decay
                args.train_batch_size = batch_size
                args.init_lr = init_lr
                avg_acc = 0.0
                for train_index, val_index in kf.split(features['Xtrain']):
                    X_train, X_val = features['Xtrain'][train_index], features['Xtrain'][val_index]
                    y_train, y_val = features['Ytrain'][train_index], features['Ytrain'][val_index]
                    if args.manual_seed is not None:
                        setup_cuda(args.manual_seed)
                    model = LinearModel(dim, num_classes)
                    model.to(device)
                    top1_train, top5_train, top1_val, top5_val = train(args, X_train, y_train, X_val, y_val, model, print_logs=False)
                    avg_acc += top1_val
                avg_acc = avg_acc/args.kfolds
                cur_params = {'weight_decay':weight_decay, 'batch_size':batch_size, 'init_lr':init_lr}
                print('Weight decay ', weight_decay, 'Batch Size ', batch_size, 'Init LR', init_lr, 'Accuracy: ',avg_acc)
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_params = cur_params
                    print('Current best params: ',best_params)
                    
    # args.weight_decay = best_params['weight_decay']
    # best_params = {'weight_decay':0.0, 'batch_size':128, 'init_lr':5.0}
    # best_acc = 0.0
    # for weight_decay in [args.weight_decay]:
    #     for batch_size in [128]:
    #         for init_lr in [5.0]:
    #             args.weight_decay = weight_decay
    #             args.train_batch_size = batch_size
    #             args.init_lr = init_lr
    #             avg_acc = 0.0
    #             for train_index, val_index in kf.split(features['Xtrain']):
    #                 X_train, X_val = features['Xtrain'][train_index], features['Xtrain'][val_index]
    #                 y_train, y_val = features['Ytrain'][train_index], features['Ytrain'][val_index]
    #                 if args.manual_seed is not None:
    #                     setup_cuda(args.manual_seed)
    #                 model = LinearModel(dim, num_classes)
    #                 model.to(device)
    #                 top1_train, top5_train, top1_val, top5_val = train(args, X_train, y_train, X_val, y_val, model, print_logs=False)
    #                 avg_acc += top1_val
    #             avg_acc = avg_acc/args.kfolds
    #             cur_params = {'weight_decay':weight_decay, 'batch_size':batch_size, 'init_lr':init_lr}
    #             print('Weight decay ', weight_decay, 'Batch Size ', batch_size, 'Init LR', init_lr, 'Accuracy: ',avg_acc)
    #             if avg_acc > best_acc:
    #                 best_acc = avg_acc
    #                 best_params = cur_params
    #                 print('Current best params: ',best_params)

    args.weight_decay = best_params['weight_decay']
    args.train_batch_size = best_params['batch_size']
    args.init_lr = best_params['init_lr']
    if args.manual_seed is not None:
        setup_cuda(args.manual_seed)
    model = LinearModel(dim, num_classes)
    model.to(device)
    X_train, y_train, X_test, y_test = features['Xtrain'], features['Ytrain'], features['Xtest'], features['Ytest']
    top1_train, top5_train, top1_val, top5_val = train(args, X_train, y_train, X_test, y_test, model, print_logs=True)